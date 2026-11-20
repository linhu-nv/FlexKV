from __future__ import annotations

import pytest
import torch

from flexkv.common.config import (
    CacheConfig,
    LayerGroupSpec,
    ModelConfig,
    RankInfo,
    UserConfig,
    block_size_in_bytes_for_cache,
    convert_to_block_num,
    recompute_cache_block_counts,
    update_default_config_from_user_config,
)
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType

pytestmark = pytest.mark.unit


def test_recompute_matches_heterogeneous_layout_block_size() -> None:
    """Deferred sizing must use the same bytes per block as KVCacheLayout."""
    model_config = ModelConfig(
        num_layers=62,
        num_kv_heads=8,
        head_size=128,
        kv_dim=1,
        tp_size=8,
        dtype=torch.bfloat16,
    )
    cache_config = CacheConfig(tokens_per_block=256)
    user_config = UserConfig(cpu_cache_gb=100, ssd_cache_gb=0)

    rank_info = RankInfo(model_config=model_config)
    update_default_config_from_user_config(rank_info, cache_config, user_config)
    uniform_blocks = cache_config.num_cpu_blocks

    model_config.layer_groups = [
        LayerGroupSpec(
            num_layers=21,
            num_kv_heads=1,
            head_size=585,
            layer_indices=list(range(21)),
            compress_ratio=4,
            dtype=torch.uint8,
        ),
        LayerGroupSpec(
            num_layers=20,
            num_kv_heads=1,
            head_size=9,
            layer_indices=list(range(21, 41)),
            compress_ratio=128,
            dtype=torch.uint8,
        ),
        LayerGroupSpec(
            num_layers=21,
            num_kv_heads=1,
            head_size=44,
            layer_indices=list(range(21)),
            compress_ratio=4,
            dtype=torch.uint8,
        ),
    ]

    layout = KVCacheLayout(
        type=KVCacheLayoutType.BLOCKFIRST,
        num_layer=model_config.num_layers,
        num_block=1,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads,
        head_size=model_config.head_size,
        kv_dim=model_config.kv_dim,
        layer_groups=model_config.layer_groups,
        tp_size=model_config.tp_size,
    )
    expected_blocks = convert_to_block_num(
        user_config.cpu_cache_gb, layout.kv_shape[1])

    assert recompute_cache_block_counts(model_config, cache_config) is True
    assert cache_config.num_cpu_blocks == expected_blocks
    assert cache_config.num_cpu_blocks != uniform_blocks


def test_compact_indexer_reclaims_capacity_at_fixed_bytes() -> None:
    num_layers = 8
    active_indexer_layers = [0, 3, 7]
    model_config = ModelConfig(
        num_layers=num_layers,
        num_kv_heads=1,
        head_size=576,
        kv_dim=1,
        tp_size=4,
        dtype=torch.bfloat16,
    )
    cache_config = CacheConfig(tokens_per_block=64)
    user_config = UserConfig(
        # Synthetic fixed byte budget.
        cpu_cache_gb=1,
        ssd_cache_gb=0,
    )
    rank_info = RankInfo(model_config=model_config)
    update_default_config_from_user_config(rank_info, cache_config, user_config)
    model_config.layer_groups = [
        LayerGroupSpec(
            num_layers=num_layers,
            num_kv_heads=1,
            head_size=576,
            layer_indices=list(range(num_layers)),
            compress_ratio=1,
            dtype=torch.bfloat16,
        ),
        LayerGroupSpec(
            num_layers=len(active_indexer_layers),
            num_kv_heads=1,
            head_size=8448,
            layer_indices=active_indexer_layers,
            compress_ratio=64,
            dtype=torch.uint8,
        ),
    ]

    assert recompute_cache_block_counts(model_config, cache_config) is True
    # One copy for all TP ranks: they hold the same (one-head) KV.
    compact_bytes = 8 * 64 * 576 * 2 + 3 * 8448
    aliased_bytes = 8 * 64 * 576 * 2 + 8 * 8448
    assert cache_config.num_cpu_blocks == (1 << 30) // compact_bytes
    assert cache_config.num_cpu_blocks > (1 << 30) // aliased_bytes


@pytest.mark.parametrize("tp_size,nnodes", [(8, 1), (16, 2)])
def test_split_kv_host_block_has_a_slot_per_tp_rank_on_its_node(tp_size, nnodes) -> None:
    page = 4096
    model_config = ModelConfig(
        num_layers=2,
        num_kv_heads=tp_size,
        head_size=page,
        kv_dim=1,
        tp_size=tp_size,
        nnodes=nnodes,
        dtype=torch.uint8,
        layer_groups=[LayerGroupSpec(
            num_layers=2,
            num_kv_heads=1,
            head_size=page,
            layer_indices=[0, 1],
            compress_ratio=16,
            dtype=torch.uint8,
        )],
    )
    cache_config = CacheConfig(tokens_per_block=16)

    assert model_config.host_tp_copies == 8
    assert block_size_in_bytes_for_cache(model_config, cache_config) == 8 * 2 * page
