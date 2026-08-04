"""Source VMM handles from vLLM's own ``CuMemAllocator``.

Why this exists
---------------
FlexKV's peer->GPU RDMA paths need the KV cache to be a CUDA VMM allocation whose
``requestedHandleTypes`` includes ``POSIX_FILE_DESCRIPTOR`` (see
:mod:`flexkv.common.vmm_handle`).  vLLM has such an allocator already --
``CuMemAllocator``, enabled by ``enable_cumem_allocator``, which
``FlexKVConnectorV1.requires_exportable_kv_cache()`` turns on.  Its allocations
carry ``POSIX_FILE_DESCRIPTOR`` and ``gpuDirectRDMACapable``, so they are exactly
what FlexKV needs and this module exports handles from them rather than
allocating anything.

Exporting from vLLM's pool is not merely an optimisation: FlexKV cannot supply a
``MemPool`` of its own instead.  Nesting one inside vLLM's is *silently* wrong --
torch honours the inner pool, the KV cache lands in FlexKV's blocks, and vLLM's
``pointer_to_data`` stays empty, with no error anywhere.  And a KV cache in
vLLM's pool could not be shared the old way either: ``cudaIpcGetMemHandle``
cannot export VMM memory, it returns ``cudaErrorInvalidValue``.

:class:`~flexkv.common.vmm_handle.VMMAllocator` remains for callers that own
their own KV buffers, such as FlexKV's standalone tests.

Interface
---------
:class:`CumemVMMSource` duck-types the part of ``VMMAllocator`` that
:class:`~flexkv.common.vmm_handle.VMMSharedHandle` uses -- ``is_vmm``,
``block_info``, ``export_fd`` -- minus ``pool()``, since it allocates nothing.
:func:`get_vmm_source` picks the right one for the current process.
"""
from __future__ import annotations

import bisect
import ctypes
import sys
from typing import Any, Dict, Optional, Tuple

import torch

from flexkv.common.debug import flexkv_logger
from flexkv.common.vmm_handle import (
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
    _cu_check,
    _cuda,
)

_cuda.cuMemRetainAllocationHandle.restype = ctypes.c_int
_cuda.cuMemRetainAllocationHandle.argtypes = [
    ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_void_p,
]
_cuda.cuMemExportToShareableHandle.restype = ctypes.c_int
_cuda.cuMemExportToShareableHandle.argtypes = [
    ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_int, ctypes.c_ulonglong,
]


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CUmemAllocationFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _CUmemAllocationFlags),
    ]


_cuda.cuMemGetAllocationPropertiesFromHandle.restype = ctypes.c_int
_cuda.cuMemGetAllocationPropertiesFromHandle.argtypes = [
    ctypes.POINTER(_CUmemAllocationProp), ctypes.c_ulonglong,
]


def live_cumem_allocator() -> Optional[Any]:
    """vLLM's ``CuMemAllocator`` singleton, or ``None``, without importing vLLM.

    ``sys.modules.get`` rather than an import for two reasons: FlexKV's transfer
    worker is a separate process that must never pull vLLM in, and the singleton's
    *existence* is itself the signal we want.  Merely importing
    ``vllm.device_allocator.cumem`` leaves ``CuMemAllocator.instance`` as ``None``;
    it is populated only once vLLM's worker calls ``get_instance()``, which it
    does when ``enable_cumem_allocator`` is set -- and before FlexKV's
    registration runs.
    """
    mod = sys.modules.get("vllm.device_allocator.cumem")
    if mod is None:
        return None
    cls = getattr(mod, "CuMemAllocator", None)
    if cls is None:
        return None
    return getattr(cls, "instance", None)


class CumemVMMSource:
    """Exports handles for KV tensors living in vLLM's cumem pool."""

    def _lookup(self, data_ptr: int) -> Optional[Tuple[int, int]]:
        """``(base, aligned_size)`` of the cumem allocation containing ``data_ptr``.

        An interval search, not a dict lookup.  ``pointer_to_data`` is keyed by
        the *segment* base that vLLM's allocator returned, and torch's caching
        allocator carves individual tensors out of those segments -- four 100 MiB
        KV tensors can share one 512 MiB segment, with only the first one's
        pointer present as a key.  Testing ``ptr in pointer_to_data`` would miss
        the other three.

        Rebuilt on every call by design: the allocator mutates these entries in
        place and its free callback removes keys, so the table must not be
        cached.
        """
        alloc = live_cumem_allocator()
        if alloc is None:
            return None
        table: Dict[int, Any] = alloc.pointer_to_data
        bases = sorted(table)
        i = bisect.bisect_right(bases, data_ptr) - 1
        if i < 0:
            return None
        base = bases[i]
        # handle[1] is the granularity-rounded size that was cuMemCreate'd and
        # mapped -- always >= the tensor's own nbytes, and what cuMemMap needs.
        size = table[base].handle[1]
        if base <= data_ptr < base + size:
            return base, size
        return None

    def is_vmm(self, tensor: torch.Tensor) -> bool:
        """Whether ``tensor`` sits in an exportable vLLM cumem allocation."""
        found = self._lookup(tensor.data_ptr())
        if found is None:
            # Also excludes PYTORCH_CUDA_ALLOC_CONF=expandable_segments tensors:
            # they pass the driver check below but are backed by several handles,
            # so a single fd would map only part of the range.
            return False
        base, _size = found
        handle = ctypes.c_ulonglong(0)
        if _cuda.cuMemRetainAllocationHandle(
            ctypes.byref(handle), ctypes.c_void_p(base)
        ) != 0:
            # The key exists but the VA is not currently mapped.
            return False
        try:
            prop = _CUmemAllocationProp()
            if _cuda.cuMemGetAllocationPropertiesFromHandle(
                ctypes.byref(prop), handle
            ) != 0:
                return False
            # Load-bearing, not cosmetic: an allocation created without this
            # handle type (vLLM prefers FABRIC where the driver claims support)
            # would pass every other check and then fail at export time.
            return bool(
                prop.requestedHandleTypes
                & CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            )
        finally:
            # A retained handle is itself a reference that pins the physical
            # pages, so it must not outlive this check.
            _cuda.cuMemRelease(ctypes.c_ulonglong(handle.value))

    def block_info(self, tensor: torch.Tensor) -> Tuple[int, int]:
        """Return ``(base_ptr, block_size)`` of the block containing ``tensor``."""
        found = self._lookup(tensor.data_ptr())
        if found is None:
            raise ValueError(
                f"tensor at {hex(tensor.data_ptr())} is not inside a vLLM cumem "
                "allocation; was it allocated outside the kv_cache memory pool?"
            )
        return found

    def export_fd(self, tensor: torch.Tensor) -> int:
        """Shareable POSIX fd for the cumem block containing ``tensor``.

        The retained handle is released immediately: it and the exported fd each
        independently pin the physical pages, so keeping both would leave a
        reference nothing ever drops.  The fd's lifetime is the caller's.
        """
        base, _size = self.block_info(tensor)
        handle = ctypes.c_ulonglong(0)
        _cu_check(
            _cuda.cuMemRetainAllocationHandle(
                ctypes.byref(handle), ctypes.c_void_p(base)
            ),
            "cuMemRetainAllocationHandle",
        )
        try:
            fd = ctypes.c_int(-1)
            _cu_check(
                _cuda.cuMemExportToShareableHandle(
                    ctypes.byref(fd), ctypes.c_ulonglong(handle.value),
                    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0,
                ),
                "cuMemExportToShareableHandle",
            )
            return fd.value
        finally:
            _cuda.cuMemRelease(ctypes.c_ulonglong(handle.value))


_source: Optional[Any] = None


def get_vmm_source() -> Any:
    """The handle source for this process: vLLM's cumem pool, or FlexKV's own.

    Call this instead of ``get_vmm_allocator()`` so that under vLLM no
    ``VMMAllocator`` is ever constructed -- it would demand
    ``$FLEXKV_VMM_ALLOCATOR_LIB``, which that configuration does not need.
    """
    global _source
    if _source is None:
        if live_cumem_allocator() is not None:
            flexkv_logger.info(
                "VMM handles will come from vLLM's cumem pool; FlexKV allocates "
                "no pool of its own"
            )
            _source = CumemVMMSource()
        else:
            from flexkv.common.vmm_handle import get_vmm_allocator

            _source = get_vmm_allocator()
    return _source
