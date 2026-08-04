"""Check that vLLM allocated its KV cache as memory FlexKV can share.

For the peer->GPU RDMA paths the KV buffers must be CUDA VMM allocations rather
than ordinary ``cudaMalloc`` memory, because only a VMM mapping can be re-exported
as a dma-buf in the importing process and hence registered as an RDMA MR (see
:mod:`flexkv.common.vmm_handle`).

vLLM owns those buffers -- FlexKV only receives them in ``register_to_server`` --
so getting them from the right allocator is vLLM's job, and it already has the
machinery: ``FlexKVConnectorV1.requires_exportable_kv_cache()`` turns on
``enable_cumem_allocator``, and the worker allocates the KV cache inside
``CuMemAllocator``'s pool.  Those allocations are created with
``POSIX_FILE_DESCRIPTOR`` and ``gpuDirectRDMACapable``, which is exactly what
:class:`~flexkv.common.vmm_handle.VMMSharedHandle` needs.

This module used to wrap ``GPUModelRunner._allocate_kv_cache_tensors`` in a pool
of FlexKV's own instead.  That was fragile (it bound FlexKV to a private method's
name and signature) and, once vLLM had a pool of its own, silently wrong: torch
honours the *inner* pool, so the KV cache landed in FlexKV's blocks while vLLM's
``pointer_to_data`` stayed empty, without raising anywhere.  All that is left
here is the check.
"""
from __future__ import annotations

from flexkv.common.cumem_source import live_cumem_allocator
from flexkv.common.debug import flexkv_logger
from flexkv.common.vmm_handle import is_vmm_sharing_enabled


def require_exportable_kv_cache() -> bool:
    """Verify vLLM will allocate the KV cache where FlexKV can export it.

    Returns whether an exportable allocator is in place; False (a no-op) when VMM
    sharing is disabled.  Raises when sharing is on but vLLM is not using its
    cumem allocator, because the alternative is registration failing much later,
    inside mooncake, with nothing pointing back at the cause.
    """
    if not is_vmm_sharing_enabled():
        return False

    if live_cumem_allocator() is not None:
        flexkv_logger.info(
            "FlexKV VMM sharing: vLLM's cumem allocator is active, so the KV "
            "cache will be exportable"
        )
        return True

    raise RuntimeError(
        "FlexKV VMM sharing is enabled but vLLM's cumem allocator is not "
        "active, so the KV cache would be plain cudaMalloc memory that cannot "
        "be registered as an RDMA memory region by the transfer worker. This "
        "should have been handled by "
        "FlexKVConnectorV1.requires_exportable_kv_cache(); it is missing if "
        "vLLM predates that hook. Set enable_cumem_allocator to get the same "
        "allocator explicitly."
    )
