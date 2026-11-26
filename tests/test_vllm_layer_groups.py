"""vLLM KV cache groups with mixed page sizes (DeepSeek-V4.1, GQA) as FlexKV layer groups."""
import pytest
import torch

if not hasattr(pytest.importorskip("vllm.v1.kv_cache_interface"), "KVCacheGroup"):
    pytest.skip("needs vLLM 0.34.1's KV cache groups", allow_module_level=True)

from vllm.v1.core.single_type_kv_cache_manager import (  # noqa: E402
    CircularBufferManager,
    FullAttentionManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (  # noqa: E402
    AttentionSpec,
    KVCacheConfig,
    KVCacheGroup,
    KVCacheTensor,
    MLAAttentionSpec,
    create_kv_cache_views,
)
from vllm.v1.kv_cache_layout import KVCacheLayout as VllmKVCacheLayout  # noqa: E402

from flexkv.common.config import LayerGroupSpec, block_size_in_bytes_for_cache  # noqa: E402
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType  # noqa: E402
from flexkv.integration.config import FlexKVConfig, VllmSWAGroups  # noqa: E402
from flexkv.integration.vllm.vllm_v1_adapter import FlexKVWorkerConnector  # noqa: E402
from flexkv.transfer.template import compile_gpu_regions  # noqa: E402

pytestmark = pytest.mark.unit

P = "language_model.model.layers"
NUM_BLOCKS = 16
BLOCK = 116928  # one block of the MLA group: every page below, back to back


def _mla(ratio, head, content, block_size=64):
    return MLAAttentionSpec(block_size=block_size, num_kv_heads=1, max_tp_shards=1,
                            head_size=head, dtype=torch.uint8, tokens_per_state=ratio,
                            cache_dtype_str="fp8_ds_mla", alignment=576,
                            model_version="deepseek_v4", state_content_bytes=content)


def _group(manager, specs, **params):
    return KVCacheGroup.from_layers(
        {name: manager.layer_config(spec, **params) for name, spec in specs.items()})


def _dsv41_config(kv_tp_replicas: int = 8, bounded_replay: bool = True) -> KVCacheConfig:
    """The H20 TP8 layout: SWA, compressed KV + indexer K, compressor rings.
    With SWA bounded replay only the MLA group is prefix-cacheable; without
    it so are the SWA layers, in seven groups. Every TP rank holds the same
    caches, which vLLM's kv_tp_replicas says."""
    swa_names = [f"{P}.{i}.attn.swa_cache" for i in range(40)]
    num_swa_groups = 1 if bounded_replay else 7
    swa_groups = [swa_names[g::num_swa_groups] for g in range(num_swa_groups)]
    specs = {}
    for i in (2, 8, 14, 20):
        ratio = 1 if i == 20 else 2
        specs[f"{P}.{i}.attn"] = _mla(ratio, 512, 584)
        specs[f"{P}.{i}.attn.indexer.k_cache"] = _mla(ratio, 132, None)
    ring = AttentionSpec(block_size=8, num_kv_heads=1, max_tp_shards=1, head_size=1024,
                         head_size_v=0, dtype=torch.float32)
    ring_names = [f"{P}.{i}.attn.compressor.state_cache" for i in (2, 8, 14)]
    size = NUM_BLOCKS * BLOCK
    tensors = [KVCacheTensor(size, names, 19008, BLOCK) for names in swa_groups] + [
        KVCacheTensor(size, [f"{P}.{i}.attn" for i in (2, 8, 14)], 19008, BLOCK, 0),
        KVCacheTensor(size, [f"{P}.{i}.attn.indexer.k_cache" for i in (2, 8, 14)], 4608,
                      BLOCK, 57024),
        KVCacheTensor(size, [f"{P}.20.attn"], 37440, BLOCK, 70848),
        KVCacheTensor(size, [f"{P}.20.attn.indexer.k_cache"], 8640, BLOCK, 108288),
        KVCacheTensor(size, ring_names, 32768, BLOCK),
    ]
    groups = [
        _group(SlidingWindowManager, dict.fromkeys(names, _mla(1, 512, 584, block_size=32)),
               sliding_window=128, bounded_replay=bounded_replay)
        for names in swa_groups
    ] + [
        _group(FullAttentionManager, specs),
        _group(CircularBufferManager, dict.fromkeys(ring_names, ring)),
    ]
    config = KVCacheConfig(num_blocks=NUM_BLOCKS, kv_cache_tensors=tensors, kv_cache_groups=groups,
                           kv_tp_replicas=kv_tp_replicas)
    config.kv_cache_layout = "BLHNC"
    return config


def _vllm_cache_views(config, layer_names, kernel_block_size=None, raw=None):
    """The caches of ``layer_names`` as vLLM's worker hands them over: a
    logical ``[B, H, N, C]`` view per layer into one byte allocation, ``raw``
    or a zeroed one."""
    specs = {}
    for group in config.kv_cache_groups:
        specs.update(group.kv_cache_specs)
    if raw is None:
        raw = torch.zeros(config.kv_cache_tensors[0].size, dtype=torch.int8)
    layout = VllmKVCacheLayout[config.kv_cache_layout]
    views = {}
    for tensor in config.kv_cache_tensors:
        if tensor.layers[0] in layer_names:
            views.update(zip(tensor.layers, create_kv_cache_views(
                raw, specs[tensor.layers[0]], config.num_blocks, layout, tensor,
                kernel_block_size)))
    return views


class _Registration:
    """Stands in for KVTPClient: keeps what the connector registers."""

    def register_to_server(self, **kwargs):
        self.kwargs = kwargs


def _register(flexkv_config, caches):
    connector = object.__new__(FlexKVWorkerConnector)
    connector.flexkv_config = flexkv_config
    connector.enable_layerwise_transfer = False
    connector.layerwise_counter_pool = None
    connector.tp_client = _Registration()
    connector._register_layer_groups(caches, flexkv_config.vllm_kv_group, resume=False)
    return connector.tp_client.kwargs


@pytest.mark.parametrize("kv_tp_replicas", [8, 1])
def test_dsv41_mla_group_as_layer_groups(kv_tp_replicas):
    flexkv_config = FlexKVConfig()
    flexkv_config.model_config.tp_size = 8
    flexkv_config._configure_kv_layout_from_vllm_cache_config(_dsv41_config(kv_tp_replicas))

    group = flexkv_config.vllm_kv_group
    assert group.group_id == 1 and group.num_blocks == NUM_BLOCKS
    assert group.layer_names == tuple(f"{P}.{i}.attn" for i in (2, 8, 14, 20))
    assert flexkv_config.cache_config.tokens_per_block == 64
    model_config = flexkv_config.model_config
    assert (model_config.num_layers, model_config.num_kv_heads, model_config.kv_dim) == (4, 1, 1)
    assert [(g.head_size, g.layer_indices, g.compress_ratio)
            for g in model_config.layer_groups] == [
        (19008, [0, 1, 2], 64), (4608, [0, 1, 2], 64), (37440, [3], 64), (8640, [3], 64)]
    assert group.group_layer_names[1] == tuple(
        f"{P}.{i}.attn.indexer.k_cache" for i in (2, 8, 14))
    # Every TP rank holds the same MLA bytes: one copy per host block.
    assert model_config.host_rank_copies == 1


def test_dsv41_layouts_are_one_shared_head_per_page():
    config = _dsv41_config()
    flexkv_config = FlexKVConfig()
    flexkv_config.model_config.tp_size = 8
    flexkv_config._configure_kv_layout_from_vllm_cache_config(config)
    group = flexkv_config.vllm_kv_group

    registration = _register(flexkv_config, _vllm_cache_views(config, group.cache_layer_names))

    assert [(layout.num_head, layout.head_size, layout.num_kv_heads, layout.kv_dim)
            for layout in registration["gpu_layouts"]] == [
        (1, 19008, 1, 1), (1, 4608, 1, 1), (1, 37440, 1, 1), (1, 8640, 1, 1)]


G = "model.layers"
WIDE_SPEC = AttentionSpec(block_size=16, num_kv_heads=2, head_size=128, dtype=torch.bfloat16)
NARROW_SPEC = AttentionSpec(block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16)
WIDE, NARROW = WIDE_SPEC.page_size_bytes, NARROW_SPEC.page_size_bytes
GQA_BLOCK = 2 * WIDE + 2 * NARROW


def _gqa_config(layout: str = "BLHNC", mla_at: int | None = None) -> KVCacheConfig:
    """A GQA model's one KV cache group: four full-attention layers whose
    pages alternate between two sizes, at one TP rank's heads. TP splits the
    heads, so the ranks' pages differ. ``mla_at`` makes one layer's cache MLA."""
    specs = {f"{G}.{i}.self_attn.attn": NARROW_SPEC if i % 2 else WIDE_SPEC
             for i in range(4)}
    if mla_at is not None:
        specs[f"{G}.{mla_at}.self_attn.attn"] = MLAAttentionSpec(
            block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16)
    wide_layers = [f"{G}.{i}.self_attn.attn" for i in (0, 2)]
    narrow_layers = [f"{G}.{i}.self_attn.attn" for i in (1, 3)]
    size = NUM_BLOCKS * GQA_BLOCK
    if layout == "BLHNC":
        tensors = [KVCacheTensor(size, wide_layers, WIDE, GQA_BLOCK),
                   KVCacheTensor(size, narrow_layers, NARROW, GQA_BLOCK, 2 * WIDE)]
    else:
        tensors = [KVCacheTensor(size, wide_layers, NUM_BLOCKS * WIDE, WIDE),
                   KVCacheTensor(size, narrow_layers, NUM_BLOCKS * NARROW, NARROW,
                                 2 * NUM_BLOCKS * WIDE)]
    config = KVCacheConfig(num_blocks=NUM_BLOCKS, kv_cache_tensors=tensors,
                           kv_cache_groups=[_group(FullAttentionManager, specs)])
    config.kv_cache_layout = layout
    return config


@pytest.mark.parametrize("layout", ["BLHNC", "LBHNC"])
def test_gqa_group_with_mixed_pages_as_layer_groups(layout):
    flexkv_config = FlexKVConfig()
    flexkv_config.model_config.tp_size = 4
    flexkv_config._configure_kv_layout_from_vllm_cache_config(_gqa_config(layout))

    group = flexkv_config.vllm_kv_group
    assert group.group_id == 0 and group.num_blocks == NUM_BLOCKS
    assert group.layer_names == tuple(f"{G}.{i}.self_attn.attn" for i in range(4))
    model_config = flexkv_config.model_config
    assert [(g.head_size, g.layer_indices, g.compress_ratio)
            for g in model_config.layer_groups] == [(WIDE, [0, 2], 16), (NARROW, [1, 3], 16)]
    assert model_config.num_kv_heads == 2 * 4
    assert model_config.host_rank_copies == 4
    assert block_size_in_bytes_for_cache(
        model_config, flexkv_config.cache_config) == 4 * GQA_BLOCK


@pytest.mark.parametrize("layout", ["BLHNC", "LBHNC"])
def test_gqa_layouts_give_each_rank_heads(layout):
    config = _gqa_config(layout)
    flexkv_config = FlexKVConfig()
    flexkv_config.model_config.tp_size = 4
    flexkv_config._configure_kv_layout_from_vllm_cache_config(config)

    caches = _vllm_cache_views(config, config.kv_cache_groups[0].layer_names)

    registration = _register(flexkv_config, caches)

    assert [(layout.num_head, layout.head_size, layout.num_kv_heads, layout.kv_dim)
            for layout in registration["gpu_layouts"]] == [
        (2, WIDE // 2, 2 * 4, 1), (1, NARROW, 1 * 4, 1)]
    main = registration["kv_layout"]
    assert (main.num_layer, main.num_head, main.num_kv_heads) == (4, 2, 2 * 4)


def test_group_mixing_mla_and_split_caches_keeps_a_copy_per_rank():
    flexkv_config = FlexKVConfig()
    flexkv_config.model_config.tp_size = 4
    flexkv_config._configure_kv_layout_from_vllm_cache_config(_gqa_config(mla_at=1))

    assert not flexkv_config.vllm_kv_group.kv_shared_across_ranks
    assert flexkv_config.model_config.host_rank_copies == 4


def test_group_led_by_mla_with_split_caches_is_refused():
    flexkv_config = FlexKVConfig()
    flexkv_config.model_config.tp_size = 4

    with pytest.raises(ValueError, match="first layer is MLA"):
        flexkv_config._configure_kv_layout_from_vllm_cache_config(_gqa_config(mla_at=0))


def test_kernel_block_split_view_moves_whole_blocks():
    config = _gqa_config("LBHNC")
    flexkv_config = FlexKVConfig()
    flexkv_config.model_config.tp_size = 4
    flexkv_config._configure_kv_layout_from_vllm_cache_config(config)
    caches = _vllm_cache_views(config, config.kv_cache_groups[0].layer_names,
                               kernel_block_size=8)

    registration = _register(flexkv_config, caches)
    view = registration["handles_per_group"][0][0]
    (region,) = compile_gpu_regions(
        flexkv_config.model_config.layer_groups[:1], [registration["gpu_layouts"][:1]],
        16, 1, torch.uint8, tensors_per_group_device=[[view]])

    assert view.shape[0] == 2 * NUM_BLOCKS
    assert (region.chunk_bytes, region.block_stride) == (WIDE, WIDE)


def test_block_stride_from_interleaved_view():
    raw = torch.zeros(NUM_BLOCKS * BLOCK, dtype=torch.uint8)
    view = torch.as_strided(raw, (NUM_BLOCKS, 1, 32, 584), (BLOCK, 0, 584, 1), 57024)
    group = LayerGroupSpec(num_layers=1, num_kv_heads=1, head_size=4608, layer_indices=[0],
                           dtype=torch.uint8, compress_ratio=64)
    layout = KVCacheLayout(type=KVCacheLayoutType.LAYERFIRST, num_layer=1, num_block=NUM_BLOCKS,
                           tokens_per_block=1, num_head=1, head_size=4608, kv_dim=1,
                           num_kv_heads=1)
    (region,) = compile_gpu_regions([group], [[layout]], 64, 1, torch.uint8,
                                    tensors_per_group_device=[[view]])
    assert (region.chunk_bytes, region.block_stride) == (4608, BLOCK)


def test_caches_vllm_replicates_keep_one_copy():
    config = _gqa_config()
    config.kv_tp_replicas = 4
    flexkv_config = FlexKVConfig()
    flexkv_config.model_config.tp_size = 4
    flexkv_config._configure_kv_layout_from_vllm_cache_config(config)

    model_config = flexkv_config.model_config
    assert flexkv_config.vllm_kv_group.kv_shared_across_ranks
    assert (model_config.num_kv_heads, model_config.host_rank_copies) == (1, 1)


def test_dsv41_without_bounded_replay_snapshots_its_sliding_window_groups():
    config = _dsv41_config(bounded_replay=False)
    flexkv_config = FlexKVConfig()
    flexkv_config.model_config.tp_size = 8
    flexkv_config._configure_kv_layout_from_vllm_cache_config(config)

    group = flexkv_config.vllm_kv_group
    assert group.group_id == 7
    assert (group.swa.group_ids, group.swa.window_blocks, group.swa.region_layer_names) == (
        tuple(range(7)), 4, (f"{P}.0.attn.swa_cache",))
    swa = flexkv_config.cache_config.swa
    assert (swa.pages_per_slot, swa.num_swa_layers, swa.bytes_per_token_per_layer) == (
        28, 1, BLOCK // 64)
    assert flexkv_config.cache_config.enable_swa_transfer

    raw = torch.zeros(config.kv_cache_tensors[0].size, dtype=torch.int8)
    caches = _vllm_cache_views(config, group.cache_layer_names + group.swa.region_layer_names, raw=raw)
    registration = _register(flexkv_config, caches)

    (pages,) = registration["swa_caches"]
    assert pages.shape == (NUM_BLOCKS, BLOCK) and pages.data_ptr() == raw.data_ptr()
    layout = registration["swa_layout"]
    assert (layout.num_block, layout.tokens_per_block, layout.head_size) == (
        NUM_BLOCKS, 64, BLOCK // 64)


def _sliding_window_hybrid_config() -> KVCacheConfig:
    """Four full-attention and four sliding-window layers with layers
    outermost: the two groups overlay the same four layer regions."""
    full = [f"{G}.{i}.self_attn.attn" for i in range(0, 8, 2)]
    swa = [f"{G}.{i}.self_attn.attn" for i in range(1, 8, 2)]
    size = 4 * NUM_BLOCKS * WIDE
    tensors = [KVCacheTensor(size, names, NUM_BLOCKS * WIDE, WIDE) for names in (swa, full)]
    groups = [
        _group(SlidingWindowManager, dict.fromkeys(swa, WIDE_SPEC),
               sliding_window=128, bounded_replay=False),
        _group(FullAttentionManager, dict.fromkeys(full, WIDE_SPEC)),
    ]
    config = KVCacheConfig(num_blocks=NUM_BLOCKS, kv_cache_tensors=tensors, kv_cache_groups=groups)
    config.kv_cache_layout = "LBHNC"
    return config


def test_layer_outermost_sliding_window_groups_snapshot_each_layer_region():
    config = _sliding_window_hybrid_config()
    flexkv_config = FlexKVConfig()
    flexkv_config.cache_config.tokens_per_block = 16
    flexkv_config.model_config.dtype = torch.bfloat16
    flexkv_config._configure_kv_layout_from_vllm_cache_config(config)

    group = flexkv_config.vllm_kv_group
    regions = tuple(f"{G}.{i}.self_attn.attn" for i in range(1, 8, 2))
    assert group.group_id == 1
    assert group.swa == VllmSWAGroups((0,), 16, 8, WIDE, regions)
    swa = flexkv_config.cache_config.swa
    assert (swa.pages_per_slot, swa.num_swa_layers) == (8, 4)

    raw = torch.zeros(config.kv_cache_tensors[0].size, dtype=torch.int8)
    connector = object.__new__(FlexKVWorkerConnector)
    connector.flexkv_config = flexkv_config
    connector.layerwise_counter_pool = None
    registration = connector._swa_registration(
        _vllm_cache_views(config, regions, raw=raw), group)
    assert [r.data_ptr() - raw.data_ptr() for r in registration["swa_caches"]] == [
        i * NUM_BLOCKS * WIDE for i in range(4)]
    assert all(r.shape == (NUM_BLOCKS, WIDE) for r in registration["swa_caches"])
    assert registration["swa_layout"].num_layer == 4

