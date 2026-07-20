import types

import numpy as np
import pytest

from flexkv.cache.cache_engine import GlobalCacheEngine
from flexkv.common.transfer import (
    DeviceType, TransferOp, TransferOpGraph, TransferType,
)
from flexkv.common.type import MatchResult, MatchResultAccel
from flexkv.transfer.utils import RemoteSSD2HMetaInfo


class _FakeNode:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


class _FakeCPUCache:
    def take(self, num_required_blocks, **kwargs):
        return np.arange(100, 100 + num_required_blocks, dtype=np.int64)

    def recycle(self, blocks):
        pass

    def insert(self, sequence_meta, blocks, **kwargs):
        return _FakeNode(len(blocks)), np.array([], dtype=np.int64)

    def lock_node(self, node):
        pass


def _match(blocks, position, node_id=1):
    """Build a MatchResult: LOCAL prefix or a pure PEER prefix (position)."""
    blocks = np.asarray(blocks, dtype=np.int64)
    if position == "local":
        return MatchResult(local=MatchResultAccel(
            num_ready_matched_blocks=len(blocks),
            num_matched_blocks=len(blocks),
            physical_blocks=blocks,
        ))
    if len(blocks) == 0:
        return MatchResult(local=MatchResultAccel())
    remote = MatchResultAccel(
        num_ready_matched_blocks=len(blocks),
        num_matched_blocks=len(blocks),
        physical_blocks=blocks,
        block_node_ids=np.full(len(blocks), node_id, dtype=np.int64),
    )
    return MatchResult(local=MatchResultAccel(), remote=remote)


def _build_graph(peer_tier, direct_to_gpu):
    if peer_tier == "cpu":
        cpu_result = _match([10, 11], "remote")
        # The SSD prefix is equal to the CPU prefix, so there is no SSD suffix.
        ssd_result = _match([20, 21], "local")
    else:
        cpu_result = _match([], "remote")
        ssd_result = _match([20, 21], "remote")

    cpu_cache = _FakeCPUCache()
    empty_lake = MatchResult(local=MatchResultAccel())
    fake = types.SimpleNamespace(
        cache_config=types.SimpleNamespace(
            enable_cpu=True,
            enable_ssd=True,
            enable_lake=False,
            enable_gds=False,
            enable_p2p_cpu=True,
            enable_p2p_gpu=direct_to_gpu,
        ),
        cpu_cache_engine=cpu_cache,
        ssd_cache_engine=types.SimpleNamespace(),
        lake_cache_engine=None,
        cache_engines={DeviceType.CPU: cpu_cache},
        _metrics_collector=None,
        match_all=lambda *args, **kwargs: (cpu_result, ssd_result, empty_lake),
        _release_match_pre_locks=lambda **kwargs: None,
        _handoff_locks=lambda *args, **kwargs: None,
    )
    strategy = types.SimpleNamespace(
        ignore_gpu=False, ignore_ssd=False, ignore_gds=False, ignore_lake=True
    )
    graph, _, _, _, _, _ = GlobalCacheEngine._get_impl_without_lake(
        fake,
        request_id=1,
        sequence_meta=None,
        block_mask_start=0,
        block_mask_end=2,
        gpu_block_ids=np.array([30, 31], dtype=np.int64),
        layer_num=2,
        temp_cache_strategy=strategy,
    )
    return [op.transfer_type for op in graph._op_map.values()]


@pytest.mark.parametrize(
    "peer_tier,direct_to_gpu,expected",
    [
        ("cpu", False, {TransferType.PEERH2H, TransferType.H2D}),
        ("ssd", False, {TransferType.PEERSSD2H, TransferType.H2D}),
        ("cpu", True, {TransferType.PEERH2D}),
        ("ssd", True, {TransferType.PEERSSD2D}),
    ],
)
def test_peer_match_selects_cpu_or_gpu_worker(
    peer_tier, direct_to_gpu, expected
):
    assert set(_build_graph(peer_tier, direct_to_gpu)) == expected


def test_peer_ssd2d_metadata_round_trip():
    meta = RemoteSSD2HMetaInfo(
        task_id=7,
        cpu_block_ids=[30],
        ssd_block_ids=[9],
        peer_engine_addr="127.0.0.1:5201",
        peer_cpu_base_ptr=0,
        peer_zmq_status_addr="tcp://127.0.0.1:6201",
        data_size=128,
        layer_id=1,
        layer_granularity=2,
        gpu_dst_ptrs=[1000, 2000],
        gpu_block_positions=[0, 0],
        gpu_layer_ids=[1, 1],
        gpu_kv_ids=[0, 1],
        gpu_token_ids=[0, 0],
        gpu_head_starts=[0, 4],
        gpu_data_lens=[64, 64],
    )
    restored = RemoteSSD2HMetaInfo.from_dict(meta.to_dict())
    assert restored.gpu_dst_ptrs == [1000, 2000]
    assert restored.gpu_head_starts == [0, 4]
    assert restored.gpu_data_lens == [64, 64]


def test_h2d_rebinds_with_exact_gpu_block_offset():
    graph = TransferOpGraph()
    op = TransferOp(
        graph_id=graph.graph_id,
        transfer_type=TransferType.H2D,
        src_block_ids=np.array([8], dtype=np.int64),
        dst_block_ids=np.array([0], dtype=np.int64),
        gpu_block_offset=1,
    )
    graph.add_transfer_op(op)
    graph.set_gpu_blocks(np.array([30, 31], dtype=np.int64))
    np.testing.assert_array_equal(op.dst_block_ids, np.array([31], dtype=np.int64))
