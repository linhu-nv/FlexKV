"""CUDA VMM based GPU cache sharing (peer -> local GPU RDMA path).

Why this exists
---------------
``TensorSharedHandle`` shares GPU KV blocks across processes with legacy CUDA
IPC (``cudaIpcGetMemHandle`` / ``cudaIpcOpenMemHandle``).  On some platforms an
IPC-imported pointer cannot be registered as an RDMA memory region at all --
``ibv_reg_mr`` returns EFAULT, and mooncake's dma-buf fallback also fails
because ``cuMemGetHandleForAddressRange`` rejects IPC-imported pointers.  That
makes the peer->GPU RDMA paths (``PEERH2D`` / ``PEERSSD2D``) unusable without
falling back to TCP.

Memory allocated through the CUDA virtual memory management (VMM) API does not
have this problem: a VMM mapping rebuilt in the importing process can itself be
re-exported as a dma-buf, so mooncake derives the fd on its own and registers
the region with ``ibv_reg_dmabuf_mr``.  Nothing in mooncake needs patching --
it only needs ``WITH_NVIDIA_PEERMEM=0`` in the environment so it takes the
dma-buf branch instead of the nvidia-peermem branch.

Two pieces live here:

``VMMAllocator``
    A ``torch.cuda.MemPool`` backed by a pluggable allocator that allocates via
    ``cuMemCreate`` + ``cuMemMap`` with ``requestedHandleTypes`` set to
    ``POSIX_FILE_DESCRIPTOR``.  Only allocations made inside
    ``with allocator.pool():`` use VMM; everything else keeps using torch's
    default caching allocator, so weights/activations are untouched.

``VMMSharedHandle``
    The counterpart of ``TensorSharedHandle`` for VMM memory.  It carries the
    allocation's shareable POSIX fd instead of a 64-byte IPC handle.  The fd
    cannot travel through zmq's ``send_pyobj``, so the handle transports the
    owner's pid plus the fd number and the importer re-acquires it with
    ``pidfd_getfd(2)``; see :meth:`VMMSharedHandle.get_tensor`.
"""
from __future__ import annotations

import array
import ctypes
import errno
import os
import socket
import struct
import tempfile
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch

from flexkv.common.debug import flexkv_logger


# ---------------------------------------------------------------- driver API --

_cuda = ctypes.CDLL("libcuda.so.1")

CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 0x1
CU_MEM_LOCATION_TYPE_DEVICE = 0x1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3

# Fallback path only.  ``pidfd_getfd(2)`` needs PTRACE_MODE_ATTACH_REALCREDS on
# the exporter -- the same check ``gdb -p`` passes -- which a vLLM worker and a
# FlexKV transfer worker do *not* satisfy in a default container: they are
# siblings, and with the common ``/proc/sys/kernel/yama/ptrace_scope=1`` only a
# direct ancestor may attach.  CAP_SYS_PTRACE is usually dropped too, so there is
# nothing to fall back on.  SCM_RIGHTS over a unix socket needs no privilege at
# all and is the primary transport; these stay for the odd case where the
# exporter published no socket (e.g. an older peer).
_SYS_pidfd_open = 434
_SYS_pidfd_getfd = 438

_libc = ctypes.CDLL("libc.so.6", use_errno=True)


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _CUmemLocation), ("flags", ctypes.c_int)]


_cuda.cuMemImportFromShareableHandle.restype = ctypes.c_int
_cuda.cuMemImportFromShareableHandle.argtypes = [
    ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_void_p, ctypes.c_int,
]
_cuda.cuMemAddressReserve.restype = ctypes.c_int
_cuda.cuMemAddressReserve.argtypes = [
    ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_ulonglong, ctypes.c_ulonglong,
]
_cuda.cuMemMap.restype = ctypes.c_int
_cuda.cuMemMap.argtypes = [
    ctypes.c_ulonglong, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_ulonglong, ctypes.c_ulonglong,
]
_cuda.cuMemSetAccess.restype = ctypes.c_int
_cuda.cuMemSetAccess.argtypes = [
    ctypes.c_ulonglong, ctypes.c_size_t, ctypes.POINTER(_CUmemAccessDesc),
    ctypes.c_size_t,
]
_cuda.cuMemUnmap.restype = ctypes.c_int
_cuda.cuMemUnmap.argtypes = [ctypes.c_ulonglong, ctypes.c_size_t]
_cuda.cuMemAddressFree.restype = ctypes.c_int
_cuda.cuMemAddressFree.argtypes = [ctypes.c_ulonglong, ctypes.c_size_t]
_cuda.cuMemRelease.restype = ctypes.c_int
_cuda.cuMemRelease.argtypes = [ctypes.c_ulonglong]


def _cu_check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed with CUDA driver error {rc}")


# ------------------------------------------------------------- the allocator --

# Environment variable naming the pluggable allocator .so.  Built from
# vmm_alloc.cpp with:
#   nvcc -Xcompiler -fPIC -shared -o libflexkv_vmm.so vmm_alloc.cpp -lcuda -lcudart
_VMM_LIB_ENV = "FLEXKV_VMM_ALLOCATOR_LIB"

# Opt-in switch for the whole VMM sharing scheme.  Off by default: it needs the
# allocator .so, and legacy CUDA IPC remains correct on platforms where an
# IPC-imported pointer *can* be registered for RDMA.
_VMM_ENABLE_ENV = "FLEXKV_USE_VMM_SHARING"


def is_vmm_sharing_enabled() -> bool:
    """Whether GPU KV blocks should be shared as CUDA VMM allocations.

    Both the owning process (which must allocate inside the VMM pool) and the
    transfer worker (which must let mooncake use dma-buf registration) read this.
    """
    return os.environ.get(_VMM_ENABLE_ENV, "0").lower() in ("1", "true", "yes", "on")


_allocator: Optional[VMMAllocator] = None


def get_vmm_allocator() -> VMMAllocator:
    """The process-wide VMM allocator, created on first use.

    One pool per process is enough and is what we want: the pool must outlive
    every tensor allocated from it.
    """
    global _allocator
    if _allocator is None:
        _allocator = VMMAllocator()
    return _allocator


class VMMAllocator:
    """A ``MemPool`` whose allocations are exportable across processes.

    Allocate the GPU KV cache inside :meth:`pool` and share the resulting
    tensors with :class:`VMMSharedHandle`.  Allocations made outside the pool
    keep using torch's default caching allocator.
    """

    def __init__(self, lib_path: Optional[str] = None):
        self.lib_path = lib_path or os.environ.get(_VMM_LIB_ENV)
        if not self.lib_path:
            raise ValueError(
                f"VMM allocator library not given; pass lib_path or set ${_VMM_LIB_ENV}"
            )
        if not os.path.exists(self.lib_path):
            raise FileNotFoundError(f"VMM allocator library not found: {self.lib_path}")

        self._lib = ctypes.CDLL(self.lib_path)
        self._lib.vmm_export_fd.restype = ctypes.c_int
        self._lib.vmm_export_fd.argtypes = [ctypes.c_void_p]
        self._lib.vmm_block_info.restype = ctypes.c_int
        self._lib.vmm_block_info.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_ulonglong),
        ]

        self._torch_alloc = torch.cuda.memory.CUDAPluggableAllocator(
            self.lib_path, "vmm_malloc", "vmm_free"
        )
        # A MemPool is bound to the device it is first used on, so keep one per
        # device rather than a single shared pool.
        self._pools: Dict[int, Any] = {}
        flexkv_logger.info(f"VMMAllocator initialized from {self.lib_path}")

    def pool(self, device: Optional[int] = None) -> Any:
        """Context manager routing allocations on ``device`` to the VMM pool."""
        device_id = torch.cuda.current_device() if device is None else device
        if device_id not in self._pools:
            self._pools[device_id] = torch.cuda.MemPool(
                self._torch_alloc.allocator()
            )
        return torch.cuda.use_mem_pool(self._pools[device_id], device=device_id)

    def is_vmm(self, tensor: torch.Tensor) -> bool:
        """Whether ``tensor`` lives in a VMM block owned by this allocator."""
        base = ctypes.c_ulonglong(0)
        size = ctypes.c_ulonglong(0)
        return self._lib.vmm_block_info(
            ctypes.c_void_p(tensor.data_ptr()), ctypes.byref(base), ctypes.byref(size)
        ) == 0

    def block_info(self, tensor: torch.Tensor) -> Tuple[int, int]:
        """Return ``(base_ptr, block_size)`` of the block containing ``tensor``."""
        base = ctypes.c_ulonglong(0)
        size = ctypes.c_ulonglong(0)
        rc = self._lib.vmm_block_info(
            ctypes.c_void_p(tensor.data_ptr()), ctypes.byref(base), ctypes.byref(size)
        )
        if rc != 0:
            raise ValueError(
                f"tensor at {hex(tensor.data_ptr())} is not in a VMM block; "
                "was it allocated inside VMMAllocator.pool()?"
            )
        return base.value, size.value

    def export_fd(self, tensor: torch.Tensor) -> int:
        """Shareable POSIX fd for the block containing ``tensor``.

        Each call produces a new fd owned by this process; it must stay open for
        as long as any importer may still rebuild the mapping.
        """
        fd = self._lib.vmm_export_fd(ctypes.c_void_p(tensor.data_ptr()))
        if fd < 0:
            raise RuntimeError(
                f"cuMemExportToShareableHandle failed for {hex(tensor.data_ptr())}"
            )
        return fd


# --------------------------------------------------------------- fd passing ---

# An fd is an index into the owner's descriptor table, so it cannot travel inside
# a pickled handle: the kernel has to clone the underlying ``struct file`` into
# the importer's table.  ``SCM_RIGHTS`` over a unix socket does exactly that and,
# unlike ``pidfd_getfd(2)``, requires no permission over the peer process.
#
# The exporting process runs a tiny server: importers connect, send the fd number
# they want, and get that descriptor back as ancillary data.  Two checks keep this
# from being a way to hand the KV cache to anyone: the fd must be one this process
# actually exported, and the peer's uid (from ``SO_PEERCRED``, which the kernel
# fills in -- it cannot be forged) must match ours.  The uid check matters because
# an abstract-namespace socket has no filesystem permissions, so without it any
# process in the same network namespace could ask for a mapping.

_FD_REQUEST = struct.Struct("!Q")  # the owner-side fd number being requested
_UCRED = struct.Struct("iII")      # struct ucred: pid, uid, gid


class _FdServer:
    """Hands out this process's exported fds over ``SCM_RIGHTS``.

    One per process, started lazily on the first export.  The listening socket
    lives in the abstract namespace when possible (nothing to unlink, dies with
    the process) and falls back to a filesystem path otherwise.
    """

    def __init__(self) -> None:
        self._offered: Dict[int, None] = {}
        self._lock = threading.Lock()
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # Abstract namespace: a leading NUL means no filesystem entry, so there
        # is nothing to clean up and no collision with a stale file.
        self.address = f"\0flexkv-vmm-fd-{os.getpid()}"
        try:
            self._sock.bind(self.address)
        except OSError:
            path = os.path.join(
                tempfile.gettempdir(), f"flexkv-vmm-fd-{os.getpid()}.sock"
            )
            if os.path.exists(path):
                os.unlink(path)
            self._sock.bind(path)
            self.address = path
        self._sock.listen(64)
        self._thread = threading.Thread(
            target=self._serve, name="flexkv-vmm-fd-server", daemon=True
        )
        self._thread.start()
        flexkv_logger.info(
            f"VMM fd server listening on {self.address!r} "
            f"(SCM_RIGHTS; no ptrace permission needed)"
        )

    def offer(self, fd: int) -> None:
        """Allow ``fd`` to be requested by importers."""
        with self._lock:
            self._offered[fd] = None

    def withdraw(self, fd: int) -> None:
        """Stop serving ``fd`` (it is being closed)."""
        with self._lock:
            self._offered.pop(fd, None)

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:  # socket closed at shutdown
                return
            try:
                with conn:
                    creds = conn.getsockopt(
                        socket.SOL_SOCKET, socket.SO_PEERCRED, _UCRED.size
                    )
                    peer_pid, peer_uid, _gid = _UCRED.unpack(creds)
                    if peer_uid != os.getuid():
                        flexkv_logger.warning(
                            f"VMM fd server: refusing pid={peer_pid} "
                            f"uid={peer_uid} (expected uid {os.getuid()})"
                        )
                        conn.sendall(b"E")
                        continue
                    raw = conn.recv(_FD_REQUEST.size)
                    if len(raw) != _FD_REQUEST.size:
                        continue
                    (want,) = _FD_REQUEST.unpack(raw)
                    with self._lock:
                        known = want in self._offered
                    if not known:
                        # Refuse anything we did not export ourselves; replying
                        # with no ancillary data makes the importer raise.
                        conn.sendall(b"E")
                        continue
                    conn.sendmsg(
                        [b"F"],
                        [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                          array.array("i", [want]))],
                    )
            except Exception as e:  # one bad client must not kill the server
                flexkv_logger.warning(f"VMM fd server: dropped a request: {e}")


_fd_server: Optional[_FdServer] = None
_fd_server_lock = threading.Lock()


def _get_fd_server() -> _FdServer:
    global _fd_server
    with _fd_server_lock:
        if _fd_server is None:
            _fd_server = _FdServer()
        return _fd_server


def _recv_fd(address: str, owner_fd: int) -> int:
    """Fetch ``owner_fd`` from the exporter listening on ``address``."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(30.0)
        sock.connect(address)
        sock.sendall(_FD_REQUEST.pack(owner_fd))
        fds = array.array("i")
        msg, anc, _flags, _addr = sock.recvmsg(
            1, socket.CMSG_SPACE(fds.itemsize)
        )
        for level, ctype, data in anc:
            if level == socket.SOL_SOCKET and ctype == socket.SCM_RIGHTS:
                fds.frombytes(data[: len(data) - (len(data) % fds.itemsize)])
        if not fds:
            raise RuntimeError(
                f"exporter at {address!r} did not return fd {owner_fd} "
                f"(reply={msg!r}); it may have been closed already"
            )
        return fds[0]


# ------------------------------------------------------------------ handle ----

@dataclass
class _ImportedBlock:
    """A VMM block mapped into this process from another one.

    Tracked so the mapping can be torn down at shutdown: while it exists it
    pins the owner's physical pages, so ``cuMemUnmap`` in the owner frees
    nothing.  ``cu_handle`` is ``None`` once released.
    """

    va: int
    size: int
    device: int
    cu_handle: Optional[int]

    @property
    def is_mapped(self) -> bool:
        return self.cu_handle is not None


# Blocks mapped into this process by :meth:`VMMSharedHandle.get_tensor`, keyed by
# ``(owner_pid, device, owner_base)``.  Module-level because the tensors handed
# out reference these mappings while the handles themselves are short-lived (they
# arrive over zmq, get consumed, get dropped).
_mapped_blocks: Dict[Tuple[int, int, int], _ImportedBlock] = {}


def _is_gapless(tensor: torch.Tensor) -> bool:
    """Does ``tensor`` cover one unbroken byte range, ignoring dimension order?

    ``is_contiguous()`` is too strict for what we need: it demands C order, but a
    *permuted* view of a full allocation still occupies every byte exactly once,
    which is all the fd/offset/size handle needs to describe.  What must be
    rejected is a view with holes -- a slice or a padded ``as_strided`` -- because
    the exported region would then include bytes the tensor does not own.

    Dimensions of extent 1 are skipped: their stride is arbitrary and never
    affects addressing.
    """
    dims = [
        (s, d) for s, d in zip(tensor.stride(), tensor.shape, strict=True) if d > 1
    ]
    if not dims:
        return True
    # Walk from the fastest-varying dimension outward; a gapless layout has each
    # stride equal to the product of the extents below it.
    dims.sort(key=lambda sd: sd[0])
    expected = 1
    for stride, size in dims:
        if stride != expected:
            return False
        expected *= size
    return True


def _describe_layout(tensor: torch.Tensor) -> str:
    """Physical dimension order, most-significant first -- for error messages."""
    order = sorted(range(tensor.dim()), key=lambda i: -tensor.stride()[i])
    return f"physical dim order {tuple(order)}"


@dataclass
class VMMSharedHandle:
    """Cross-process handle for a VMM-allocated GPU tensor.

    Unlike :class:`~flexkv.common.memory_handle.TensorSharedHandle` this carries
    a shareable POSIX fd rather than a CUDA IPC handle, which is what allows the
    importing process to re-export the mapping as a dma-buf and hence register
    it for RDMA.

    A file descriptor is not picklable, so the handle records where to fetch it:
    the exporting process's fd-server address plus the fd number.
    :meth:`get_tensor` receives the descriptor over ``SCM_RIGHTS``, which needs no
    permission over the exporter.  If no address is present (an older peer) it
    falls back to ``pidfd_getfd(2)``, which needs ptrace-attach rights and so
    fails in a default container.  Either way the exporting process must stay
    alive and keep the fd open until every importer has rebuilt.
    """

    tensor_shape: Tuple[int, ...]
    tensor_dtype: torch.dtype
    # Element strides of the owner's view.  Carried because a permuted-but-gapless
    # view is accepted: rebuilding it from shape alone would hand the importer a
    # C-contiguous reinterpretation of the same bytes, i.e. a tensor whose
    # elements sit at different indices than the owner's.
    tensor_stride: Tuple[int, ...]
    device: torch.device
    # location of the shareable handle in the exporting process
    owner_pid: int
    owner_fd: int
    # unix socket serving that fd over SCM_RIGHTS; "" if unavailable
    fd_server_address: str
    # geometry of the VMM block and where the tensor sits inside it
    owner_base: int
    block_size: int
    offset: int

    def __init__(
        self,
        tensor: torch.Tensor,
        allocator: VMMAllocator,
        device_id: int = -1,
    ):
        if not tensor.is_cuda:
            raise ValueError("Only support CUDA tensor sharing")
        # Require gapless, NOT C-contiguous.  Which vLLM layout yields a
        # C-contiguous KV view flipped between releases, so testing
        # is_contiguous() here rejects the wrong thing depending on the version:
        #
        #   vLLM <= 0.21 (K/V in their own dim, 5D): NHD is the identity stride
        #     order -> contiguous;  HND permutes (0,1,3,2,4) -> not contiguous.
        #   vLLM >= 0.23 (K/V packed into the last dim, 4D): NHD permutes
        #     (0,2,1,3) -> NOT contiguous;  HND is the identity -> contiguous.
        #
        # Since NHD is the default in both (get_kv_connector_cache_layout()
        # returns "NHD" when no connector declares a layout), an is_contiguous()
        # gate would reject the default deployment on new vLLM -- and admit only
        # HND, which is precisely the layout whose addresses FlexKV computes
        # wrongly.  All four of those views are *gapless*, so that is the property
        # to test: this handle describes a region by (fd, offset, size), which is
        # exact for any view covering an unbroken byte range whatever the
        # dimension order.  Views with holes (a slice, or the page_size_padded
        # as_strided used by MLA backends with an alignment) are still refused:
        # the exported region would cover bytes the tensor does not own.
        #
        # NOTE: gapless is what the *handle* needs.  It does not make every layout
        # transferable -- the workers derive KV addresses from KVCacheLayout's
        # shape-implied strides and never read tensor.stride(), so the layout must
        # still match what KVCacheLayoutType describes.  That belongs to the
        # adapter (which infers the type from the shape) and is why FlexKV should
        # declare get_required_kvcache_layout() rather than rely on the default.
        if not _is_gapless(tensor):
            raise ValueError(
                "VMMSharedHandle requires a gapless tensor (a permuted view is "
                f"fine, a strided/padded one is not), got shape="
                f"{tuple(tensor.shape)} stride={tuple(tensor.stride())} "
                f"({_describe_layout(tensor)}). A KV view with holes usually "
                "means a sliced cache or page_size_padded alignment padding."
            )

        base, block_size = allocator.block_info(tensor)
        self.owner_fd = allocator.export_fd(tensor)
        self.owner_pid = os.getpid()
        # Publish the fd for SCM_RIGHTS retrieval.  Do not let a failure here be
        # fatal: pidfd_getfd still works where ptrace is permitted.
        try:
            server = _get_fd_server()
            server.offer(self.owner_fd)
            self.fd_server_address = server.address
        except Exception as e:
            flexkv_logger.warning(
                f"VMM fd server unavailable ({e}); falling back to pidfd_getfd, "
                "which needs ptrace permission on this process"
            )
            self.fd_server_address = ""
        self.owner_base = base
        self.block_size = block_size
        self.offset = tensor.data_ptr() - base
        self.tensor_shape = tuple(tensor.shape)
        self.tensor_stride = tuple(tensor.stride())
        self.tensor_dtype = tensor.dtype
        self.device = (
            tensor.device if device_id == -1 else torch.device(f"cuda:{device_id}")
        )

        flexkv_logger.info(
            f"VMMSharedHandle exported: device={self.device}, fd={self.owner_fd}, "
            f"block_size={block_size}, offset={self.offset}, shape={self.tensor_shape}"
        )

    def get_tensor(self) -> torch.Tensor:
        """Rebuild the tensor in this process.

        The resulting pointer is a VMM mapping, so mooncake can register it for
        RDMA (with ``WITH_NVIDIA_PEERMEM=0``) and CUDA kernels / D2D copies work
        on it as usual.
        """
        if not torch.cuda.is_initialized():
            torch.cuda.init()
        device_id = self.device.index if self.device.index is not None else 0
        torch.cuda.set_device(device_id)
        # make sure a context exists before touching the driver API
        _ = torch.zeros(1, device=self.device)

        # Several tensors (typically the per-layer views of one KV buffer) can
        # live in a single VMM block.  Map each block once and slice it, so the
        # tensors keep the same relative layout as in the owner and mooncake sees
        # one registerable region per block rather than N overlapping ones.
        cache_key = (self.owner_pid, device_id, self.owner_base)
        block = _mapped_blocks.get(cache_key)
        if block is not None and block.is_mapped:
            return _tensor_from_ptr(
                block.va + self.offset, self.tensor_shape, self.tensor_dtype,
                self.device, self._import_strides(),
            )

        if block is None:
            va = ctypes.c_ulonglong(0)
            _cu_check(
                _cuda.cuMemAddressReserve(
                    ctypes.byref(va), ctypes.c_size_t(self.block_size),
                    ctypes.c_size_t(0), ctypes.c_ulonglong(0),
                    ctypes.c_ulonglong(0),
                ),
                "cuMemAddressReserve",
            )
            block = _ImportedBlock(
                va=va.value, size=self.block_size, device=device_id,
                cu_handle=None,
            )
            _mapped_blocks[cache_key] = block
        elif self.block_size != block.size:
            # The reservation is exactly block_size long; a differently-sized
            # allocation cannot be mapped into it.
            raise RuntimeError(
                f"VMM block at {hex(block.va)} was reserved for "
                f"{block.size} bytes but the new handle describes "
                f"{self.block_size}; the KV cache geometry changed"
            )

        self._map_into(block)

        data_ptr = block.va + self.offset
        flexkv_logger.info(
            f"VMMSharedHandle imported: block va={hex(block.va)}, "
            f"data_ptr={hex(data_ptr)}, shape={self.tensor_shape}"
        )
        return _tensor_from_ptr(
            data_ptr, self.tensor_shape, self.tensor_dtype, self.device,
            self._import_strides(),
        )

    def _import_strides(self) -> Optional[Tuple[int, ...]]:
        """The owner's strides, or None when they are plain C-contiguous.

        Returning None for the contiguous case keeps the common path on
        ``_create_tensor_from_cuda_ptr``'s default and stays compatible with
        handles pickled by a peer that predates ``tensor_stride``.
        """
        strides = getattr(self, "tensor_stride", None)
        if not strides:
            return None
        expected = 1
        c_contig = []
        for size in reversed(self.tensor_shape):
            c_contig.append(expected)
            expected *= size
        if tuple(reversed(c_contig)) == tuple(strides):
            return None
        return tuple(strides)

    def _map_into(self, block: _ImportedBlock) -> None:
        """Back ``block``'s reserved address range with the owner's pages."""
        fd = self._acquire_fd()
        try:
            handle = ctypes.c_ulonglong(0)
            _cu_check(
                _cuda.cuMemImportFromShareableHandle(
                    ctypes.byref(handle), ctypes.c_void_p(fd),
                    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                ),
                "cuMemImportFromShareableHandle",
            )
        finally:
            # the imported allocation holds its own reference to the memory
            os.close(fd)

        try:
            _cu_check(
                _cuda.cuMemMap(
                    ctypes.c_ulonglong(block.va), ctypes.c_size_t(block.size),
                    ctypes.c_size_t(0), handle, ctypes.c_ulonglong(0),
                ),
                "cuMemMap",
            )
            desc = _CUmemAccessDesc()
            desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE
            desc.location.id = block.device
            desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            _cu_check(
                _cuda.cuMemSetAccess(
                    ctypes.c_ulonglong(block.va), ctypes.c_size_t(block.size),
                    ctypes.byref(desc), ctypes.c_size_t(1),
                ),
                "cuMemSetAccess",
            )
        except Exception:
            # Leave no dangling reference to the owner's pages: a retained
            # handle pins them for as long as it is held.
            _cuda.cuMemRelease(handle)
            raise
        block.cu_handle = handle.value

    def _acquire_fd(self) -> int:
        """Get a usable copy of the exporter's shareable fd in this process.

        Prefers ``SCM_RIGHTS`` (no privilege required) and only falls back to
        ``pidfd_getfd(2)`` when the exporter published no socket.
        """
        if self.owner_pid == os.getpid():
            return os.dup(self.owner_fd)

        address = getattr(self, "fd_server_address", "")
        if address:
            try:
                return _recv_fd(address, self.owner_fd)
            except Exception as e:
                flexkv_logger.warning(
                    f"SCM_RIGHTS fetch of fd {self.owner_fd} from {address!r} "
                    f"failed ({e}); trying pidfd_getfd"
                )

        pidfd = _libc.syscall(_SYS_pidfd_open, self.owner_pid, 0)
        if pidfd < 0:
            raise RuntimeError(
                f"pidfd_open({self.owner_pid}) failed: errno="
                f"{ctypes.get_errno()}; cannot import VMM handle"
            )
        try:
            fd = _libc.syscall(_SYS_pidfd_getfd, pidfd, self.owner_fd, 0)
            if fd < 0:
                eno = ctypes.get_errno()
                raise RuntimeError(
                    f"pidfd_getfd(pid={self.owner_pid}, fd={self.owner_fd}) failed: "
                    f"errno={eno}"
                    + (
                        " (EPERM: the importer may not ptrace the exporter -- this "
                        "is expected with yama ptrace_scope=1 for sibling "
                        "processes; the exporter should publish an fd server "
                        "instead)"
                        if eno == errno.EPERM else ""
                    )
                )
            return fd
        finally:
            os.close(pidfd)


def imported_block_range(data_ptr: int) -> Optional[Tuple[int, int]]:
    """``(base, size)`` of the imported VMM block containing ``data_ptr``.

    ``None`` if the pointer is not inside a block imported by this process (e.g.
    it came from a CUDA IPC handle, or is locally allocated).  Callers that
    register memory for RDMA should register the whole block once rather than
    each tensor inside it, since several tensors commonly share one block.

    A released block still answers: its address range is reserved, so the range
    is still the right one to re-register once it is mapped again.
    """
    for block in _mapped_blocks.values():
        if block.va <= data_ptr < block.va + block.size:
            return block.va, block.size
    return None


def release_imported_blocks() -> None:
    """Unmap every imported VMM block *and* give up its address range.

    For shutdown: this invalidates the addresses, so nothing derived from them
    may be used again.
    """
    for block in _mapped_blocks.values():
        if block.is_mapped:
            _cuda.cuMemUnmap(
                ctypes.c_ulonglong(block.va), ctypes.c_size_t(block.size)
            )
            _cuda.cuMemRelease(ctypes.c_ulonglong(block.cu_handle))
            block.cu_handle = None
        _cuda.cuMemAddressFree(
            ctypes.c_ulonglong(block.va), ctypes.c_size_t(block.size)
        )
    _mapped_blocks.clear()


# Either kind of GPU cache handle.  The two are interchangeable to consumers:
# both expose ``get_tensor()`` and ``device``.
SharedHandle = Union["TensorSharedHandle", VMMSharedHandle]  # noqa: F821


# Reuse the pointer->tensor construction already used for CUDA IPC imports so
# both handle types agree on dtype handling (bfloat16 / fp8 need a detour).
def _tensor_from_ptr(
    data_ptr: int,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    strides: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    from flexkv.common.memory_handle import TensorSharedHandle

    # ``strides`` must be passed through when the owner's view was permuted:
    # rebuilding from shape alone would place elements at different addresses than
    # the owner had.  None means C-contiguous, which is the common case.
    return TensorSharedHandle._create_tensor_from_cuda_ptr(
        data_ptr, shape, dtype, device, strides
    )
