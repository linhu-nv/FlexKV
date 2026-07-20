import numpy as np

from flexkv.cache.get_planner import (
    build_transfer_graph,
    get_media_list,
    plan_routes,
    route_of,
)
from flexkv.common.transfer import DeviceType, TransferType
from flexkv.common.type import CacheLocality, MatchResult, MatchResultAccel


def _match(local_blocks,
           peer_blocks=None,
           *,
           peer_node_id=7,
           local_node_ids=None,
           peer_node_ids=None):
    """Build a MatchResult(local, remote) from a local prefix + optional peer suffix.

    ``remote`` is the peer-inclusive view indexed by logical block (local slots
    then peer slots), mirroring what the engines emit; its ``block_node_ids``
    place the owning ids in the peer tail.  ``local_node_ids`` / ``peer_node_ids``
    override the per-block ids (used for LAKE PCFS file ids).
    """
    local_blocks = np.asarray(local_blocks, dtype=np.int64)
    local = MatchResultAccel(
        num_ready_matched_blocks=len(local_blocks),
        num_matched_blocks=len(local_blocks),
        physical_blocks=local_blocks,
        block_node_ids=(
            np.asarray(local_node_ids, dtype=np.int64)
            if local_node_ids is not None else None
        ),
    )
    remote = None
    if peer_blocks is not None:
        peer_blocks = np.asarray(peer_blocks, dtype=np.int64)
        combined = np.concatenate([local_blocks, peer_blocks])
        if peer_node_ids is not None:
            ids = np.asarray(peer_node_ids, dtype=np.int64)
        else:
            ids = np.concatenate([
                np.full(len(local_blocks), -1, dtype=np.int64),
                np.full(len(peer_blocks), peer_node_id, dtype=np.int64),
            ])
        remote = MatchResultAccel(
            num_ready_matched_blocks=len(combined),
            num_matched_blocks=len(combined),
            physical_blocks=combined,
            block_node_ids=ids,
        )
    return MatchResult(local=local, remote=remote)


_EMPTY = MatchResult(local=MatchResultAccel())


def _routed(segments, *, enable_gpu=True, enable_gds=False, enable_peer_gpu=False):
    plan_routes(
        segments,
        enable_gpu=enable_gpu,
        enable_gds=enable_gds,
        enable_peer_gpu=enable_peer_gpu,
    )
    return segments


def test_cpu_local_prefix_and_peer_suffix_are_combined():
    cpu = _match([10, 11], [20, 21])
    segments = _routed(get_media_list(
        cpu, _EMPTY, block_mask_start=0, block_mask_end=4))

    assert [(s.tier, s.locality, s.logical_start) for s in segments] == [
        (DeviceType.CPU, CacheLocality.LOCAL, 0),
        (DeviceType.CPU, CacheLocality.PEER, 2),
    ]
    assert [s.primary_type for s in segments] == [None, TransferType.PEERH2H]
    assert all(s.needs_h2d for s in segments)


def test_route_of_matrix():
    peer_cpu = get_media_list(_match([], [20]), _EMPTY,
                              block_mask_start=0, block_mask_end=1)[0]
    assert route_of(peer_cpu, enable_gpu=True, enable_gds=False,
                    enable_peer_gpu=True).primary_type == TransferType.PEERH2D
    assert route_of(peer_cpu, enable_gpu=True, enable_gds=False,
                    enable_peer_gpu=False).primary_type == TransferType.PEERH2H

    local_ssd = get_media_list(_EMPTY, _match([30]),
                               block_mask_start=0, block_mask_end=1)[0]
    assert route_of(local_ssd, enable_gpu=True, enable_gds=True,
                    enable_peer_gpu=False).primary_type == TransferType.DISK2D
    assert route_of(local_ssd, enable_gpu=True, enable_gds=False,
                    enable_peer_gpu=False).primary_type == TransferType.DISK2H

    peer_ssd = get_media_list(_EMPTY, _match([], [40]),
                              block_mask_start=0, block_mask_end=1)[0]
    assert route_of(peer_ssd, enable_gpu=True, enable_gds=False,
                    enable_peer_gpu=True).primary_type == TransferType.PEERSSD2D
    assert route_of(peer_ssd, enable_gpu=True, enable_gds=False,
                    enable_peer_gpu=False).primary_type == TransferType.PEERSSD2H


def test_direct_peer_suffix_uses_exact_gpu_offset():
    cpu = _match([10, 11], [20, 21])
    segments = _routed(
        get_media_list(cpu, _EMPTY, block_mask_start=0, block_mask_end=4),
        enable_peer_gpu=True,
    )
    graph, _finished, _h2d = build_transfer_graph(
        segments,
        staging_blocks=np.array([], dtype=np.int64),
        block_mask_start=0,
        layer_num=2,
    )
    graph.set_gpu_blocks(np.array([100, 101, 102, 103], dtype=np.int64))

    h2d = next(op for op in graph._op_map.values()
               if op.transfer_type == TransferType.H2D)
    peer = next(op for op in graph._op_map.values()
                if op.transfer_type == TransferType.PEERH2D)
    np.testing.assert_array_equal(h2d.dst_block_ids, [100, 101])
    np.testing.assert_array_equal(peer.dst_block_ids, [102, 103])


def test_ssd_only_fills_suffix_after_combined_cpu_prefix():
    cpu = _match([10], [20])
    ssd = _match([30, 31, 32], [40, 41, 42], peer_node_id=9)
    segments = _routed(get_media_list(
        cpu, ssd, block_mask_start=0, block_mask_end=6))

    assert [(s.tier, s.logical_start, s.primary_type) for s in segments] == [
        (DeviceType.CPU, 0, None),
        (DeviceType.CPU, 1, TransferType.PEERH2H),
        (DeviceType.SSD, 2, TransferType.DISK2H),
        (DeviceType.SSD, 3, TransferType.PEERSSD2H),
    ]
    assert plan_routes(segments, enable_gpu=True, enable_gds=False,
                       enable_peer_gpu=False) == 5


def test_lake_fills_suffix_after_cpu_and_ssd_with_file_node_ids():
    cpu = _match([10])
    ssd = _match([20, 21, 22])
    lake = _match(
        [30, 31, 32, 33], [40, 41],
        local_node_ids=[100, 101, 102, 103],
        peer_node_ids=[100, 101, 102, 103, 204, 205],
    )

    segments = get_media_list(cpu, ssd, lake,
                              block_mask_start=0, block_mask_end=6)
    num_staging = plan_routes(segments, enable_gpu=True, enable_gds=False,
                              enable_peer_gpu=False)

    assert [(s.tier, s.logical_start, s.primary_type) for s in segments] == [
        (DeviceType.CPU, 0, None),
        (DeviceType.SSD, 1, TransferType.DISK2H),
        (DeviceType.LAKE, 3, TransferType.LAKE2H),
        (DeviceType.LAKE, 4, TransferType.LAKE2H),
    ]
    assert num_staging == 5
    np.testing.assert_array_equal(segments[2].src_block_node_ids, [103])
    np.testing.assert_array_equal(segments[3].src_block_node_ids, [204, 205])

    graph, _finished, _h2d = build_transfer_graph(
        segments,
        staging_blocks=np.arange(90, 95, dtype=np.int64),
        block_mask_start=0,
        layer_num=2,
    )
    lake_ops = [op for op in graph._op_map.values()
                if op.transfer_type == TransferType.LAKE2H]
    assert len(lake_ops) == 2
    np.testing.assert_array_equal(lake_ops[0].src_block_node_ids, [103])
    np.testing.assert_array_equal(lake_ops[1].src_block_node_ids, [204, 205])


def test_mask_can_start_inside_local_span():
    cpu = _match([10, 11, 12], [20, 21])
    segments = _routed(
        get_media_list(cpu, _EMPTY, block_mask_start=2, block_mask_end=5),
        enable_peer_gpu=True,
    )
    assert [(s.logical_start, s.num_blocks) for s in segments] == [(2, 1), (3, 2)]
    assert sum(s.num_blocks for s in segments) == 3


def test_direct_route_splits_h2d_runs_without_losing_gpu_offsets():
    cpu = _match([10], [20])
    ssd = _match([30, 31, 32])
    segments = _routed(
        get_media_list(cpu, ssd, block_mask_start=0, block_mask_end=3),
        enable_peer_gpu=True,
    )
    graph, _finished, h2d_ops = build_transfer_graph(
        segments,
        staging_blocks=np.array([90], dtype=np.int64),
        block_mask_start=0,
        layer_num=2,
    )
    graph.set_gpu_blocks(np.array([100, 101, 102], dtype=np.int64))

    peer = next(op for op in graph._op_map.values()
                if op.transfer_type == TransferType.PEERH2D)
    assert peer.dst_block_ids.tolist() == [101]
    assert sorted(op.dst_block_ids.item() for op in h2d_ops) == [100, 102]
