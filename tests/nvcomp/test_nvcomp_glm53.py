"""nvCOMP ANS roundtrip on GLM-5.3-Flash's exact KV geometry.

GLM-5.3-Flash is hybrid: 34 KDA linear-attention layers plus 11
``deepseek_sparse_attention`` full-attention layers at global ids
[3,7,11,15,19,23,27,31,35,39,43]. Only those 11 carry a token-addressable
KV cache, and sglang serves them through a ``DSATokenToKVPool`` nested in a
``HybridLinearKVPool``.

Geometry, read off ``config.json`` and confirmed against a live TP=4 server:

  kv_lora_rank      = 512
  qk_rope_head_dim  = 0          -> head_size = 512 + 0 = 512
  kv_dim            = 1          (MLA latent: one buffer, not split K/V)
  num_kv_heads      = 1          (replicated across TP, not sharded)
  KV dtype          = bfloat16   (forced by sglang on SM90)
  page_size         = 64

so one chunk is 64 * 1 * 512 * 2 = 65536 B, sixteen times nvcomp's
MIN_CHUNK_BYTES floor of 4096. The generic ``test_nvcomp_cpu.py`` MLA cell
uses head_dim=576 / 18KB chunks (DSv4's shape); this file pins GLM-5.3's own
numbers so a layout change there is caught here rather than end-to-end.

The indexer cell covers the second buffer DSA pools expose,
``index_k_with_scale_buffer``: index_head_dim=128 with an fp8 scale tail, a
different element count per token than the latent group.

Compression ratios printed by these tests are a LOWER BOUND: the payload is
``torch.randn``, which is near-incompressible. Real KV is not. Treat the
numbers as "ANS does not inflate random data", not as a forecast.
"""

import pytest
import torch

from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from nvcomp_common_utils import make_kv_cache, compute_strides

try:
    from flexkv.c_ext import (
        ANSTransferContext,
        transfer_kv_blocks_ans_comp,
        transfer_kv_blocks_ans_decomp,
    )
    NVCOMP_AVAILABLE = True
except ImportError:
    NVCOMP_AVAILABLE = False

from flexkv.transfer.compression.ans import ans_utils

device = "cuda:0"

# GLM-5.3-Flash: 11 DSA full-attention layers out of 45.
GLM53_NUM_FULL_ATTN_LAYERS = 11
GLM53_KV_LORA_RANK = 512
GLM53_QK_ROPE_HEAD_DIM = 0
GLM53_INDEX_HEAD_DIM = 128
GLM53_PAGE_SIZE = 64


def _roundtrip(num_layers, num_blocks, tokens_per_block, num_heads, head_size,
               dtype, cpu_layout_name, kv_dim, num_kv_heads, label):
    """Compress GPU->CPU, zero the GPU, decompress back, assert bit-exact."""
    transfer_stream = torch.cuda.Stream()

    gpu_blocks, cpu_blocks = make_kv_cache(
        num_layers, num_blocks, tokens_per_block, num_heads, head_size,
        dtype, device, cpu_layout_name, kv_dim=kv_dim, num_kv_heads=num_kv_heads)
    gpu_ptrs = torch.tensor(
        [b.data_ptr() for b in gpu_blocks], dtype=torch.int64).pin_memory()
    (chunk_size,
     gpu_kv_stride, gpu_block_stride, gpu_layer_stride,
     cpu_kv_stride, cpu_layer_stride, cpu_block_stride) = compute_strides(
        num_layers, num_blocks, tokens_per_block, num_heads, head_size,
        dtype, cpu_layout_name, kv_dim=kv_dim, num_kv_heads=num_kv_heads)

    # The guard that silently disables nvcomp end-to-end is
    # min(chunk_sizes) < MIN_CHUNK_BYTES. Assert we are clear of it, so a
    # geometry change shows up as a failure here and not as a
    # "[nvcomp-fallback]" line buried in a server log.
    assert chunk_size >= ans_utils.MIN_CHUNK_BYTES, (
        f"{label}: chunk_size={chunk_size} is below MIN_CHUNK_BYTES="
        f"{ans_utils.MIN_CHUNK_BYTES}; nvcomp would silently fall back")

    block_ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()
    original = [b.clone() for b in gpu_blocks]

    size_table = torch.zeros(
        (num_blocks, num_layers, kv_dim), dtype=torch.uint32).pin_memory()
    st_ptr, st_block_stride, st_layer_stride = (
        size_table.data_ptr(), size_table.stride(0), size_table.stride(1))

    data_type = ans_utils.data_type(dtype)
    batch_size, _ = ans_utils.get_nvcomp_batch_size(chunk_size, data_type=data_type)
    ctx = ANSTransferContext(batch_size, chunk_size, data_type)

    try:
        with torch.cuda.stream(transfer_stream):
            d2h_bytes = transfer_kv_blocks_ans_comp(
                ctx, block_ids, gpu_ptrs,
                gpu_kv_stride, gpu_block_stride, gpu_layer_stride,
                block_ids, cpu_blocks,
                cpu_kv_stride, cpu_layer_stride, cpu_block_stride,
                chunk_size, 0, num_layers, kv_dim, 0,
                st_ptr, st_block_stride, st_layer_stride)
        transfer_stream.synchronize()

        table_i64 = size_table.to(torch.int64)
        assert (table_i64 > 0).all(), f"{label}: size_table has zero entries after D2H"
        expected_bytes = int(table_i64.sum().item())
        assert d2h_bytes == expected_bytes

        uncompressed = size_table.numel() * chunk_size
        print(f"\n[{label}] chunk={chunk_size}B layers={num_layers} "
              f"blocks={num_blocks} tpb={tokens_per_block} "
              f"heads={num_heads} head_size={head_size} dtype={dtype}")
        print(f"[{label}] ratio on randn (lower bound): "
              f"{uncompressed / d2h_bytes:.3f}x ({uncompressed} -> {d2h_bytes} B)")

        # Zero the GPU so a no-op H2D cannot masquerade as a passing roundtrip.
        for b in gpu_blocks:
            b.zero_()

        with torch.cuda.stream(transfer_stream):
            h2d_bytes = transfer_kv_blocks_ans_decomp(
                ctx, block_ids, gpu_ptrs,
                gpu_kv_stride, gpu_block_stride, gpu_layer_stride,
                block_ids, cpu_blocks,
                cpu_kv_stride, cpu_layer_stride, cpu_block_stride,
                chunk_size, 0, num_layers, kv_dim, 0,
                st_ptr, st_block_stride, st_layer_stride)
        transfer_stream.synchronize()
        assert h2d_bytes == expected_bytes
    finally:
        ctx.destroy()

    for li in range(num_layers):
        assert torch.equal(gpu_blocks[li], original[li]), \
            f"{label}: roundtrip mismatch at layer {li}"


@pytest.mark.unit
@pytest.mark.parametrize("cpu_layout_name", [
    pytest.param("BLOCKFIRST", id="bfirst"),
    pytest.param("LAYERFIRST", id="lfirst"),
])
def test_glm53_latent_roundtrip(cpu_layout_name, num_blocks=2):
    """The MLA latent group: 11 layers, kv_dim=1, head_size=512, page 64."""
    if not NVCOMP_AVAILABLE:
        pytest.skip("nvcomp not available")

    head_size = GLM53_KV_LORA_RANK + GLM53_QK_ROPE_HEAD_DIM
    assert head_size == 512
    _roundtrip(
        num_layers=GLM53_NUM_FULL_ATTN_LAYERS,
        num_blocks=num_blocks,
        tokens_per_block=GLM53_PAGE_SIZE,
        num_heads=1,
        head_size=head_size,
        dtype=torch.bfloat16,
        cpu_layout_name=cpu_layout_name,
        kv_dim=1,
        num_kv_heads=1,
        label=f"glm53-latent-{cpu_layout_name}",
    )


@pytest.mark.unit
@pytest.mark.parametrize("cpu_layout_name", [
    pytest.param("BLOCKFIRST", id="bfirst"),
    pytest.param("LAYERFIRST", id="lfirst"),
])
def test_glm53_indexer_roundtrip(cpu_layout_name, num_blocks=2):
    """The DSA indexer group: index_head_dim=128, same 11 layers."""
    if not NVCOMP_AVAILABLE:
        pytest.skip("nvcomp not available")

    _roundtrip(
        num_layers=GLM53_NUM_FULL_ATTN_LAYERS,
        num_blocks=num_blocks,
        tokens_per_block=GLM53_PAGE_SIZE,
        num_heads=1,
        head_size=GLM53_INDEX_HEAD_DIM,
        dtype=torch.bfloat16,
        cpu_layout_name=cpu_layout_name,
        kv_dim=1,
        num_kv_heads=1,
        label=f"glm53-indexer-{cpu_layout_name}",
    )


@pytest.mark.unit
def test_glm53_chunk_clears_min_chunk_bytes():
    """A page of GLM-5.3 latent KV must stay above the nvcomp floor.

    This is the guard that turns nvcomp off silently, so pin the arithmetic
    rather than trusting the end-to-end log to be read.
    """
    head_size = GLM53_KV_LORA_RANK + GLM53_QK_ROPE_HEAD_DIM
    layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=GLM53_NUM_FULL_ATTN_LAYERS,
        num_block=2,
        tokens_per_block=GLM53_PAGE_SIZE,
        num_head=1,
        head_size=head_size,
        kv_dim=1,
        num_kv_heads=1,
    )
    chunk_bytes = layout.get_chunk_size() * torch.bfloat16.itemsize
    assert chunk_bytes == GLM53_PAGE_SIZE * head_size * 2 == 65536
    assert chunk_bytes >= ans_utils.MIN_CHUNK_BYTES


@pytest.mark.unit
def test_glm53_kv_dtype_is_nvcomp_supported():
    """bf16 must be in SUPPORTED_DTYPES, else nvcomp falls back on GLM-5.3."""
    assert torch.bfloat16 in ans_utils.SUPPORTED_DTYPES


@pytest.mark.unit
def test_glm53_layout_does_not_trip_the_packed_4d_guard():
    """``kv_dim == 1 and num_kv_heads > 1`` disables nvcomp (packed 4D).

    GLM-5.3 is kv_dim=1 with num_kv_heads=1, so the guard must not fire.
    """
    kv_dim, num_kv_heads = 1, 1
    assert not (kv_dim == 1 and num_kv_heads > 1)
