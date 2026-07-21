"""Backend-neutral planning for cache GET transfer graphs.

GET is planned in two phases:

* :func:`get_media_list` — *which* media serve the queried prefix.  A greedy
  longest-prefix walk over a fixed priority (``local_cpu > peer_cpu >
  local_ssd > peer_ssd > local_lake > peer_lake``) carves the ready prefix into
  contiguous :class:`MediaSegment`\\s, one per medium.
* :func:`plan_routes` + :func:`build_transfer_graph` — *how* each segment
  reaches the GPU.  :func:`route_of` maps a segment to a concrete transfer path
  given ``enable_gds`` / ``enable_peer_gpu``; the graph builder emits the primary
  ops and coalesced H2D runs.

Both phases are pure (no cache-engine access).  Promotion — writing a staged
intermediate result back into a lower tier's index for future reuse — needs the
engines and lives in ``GlobalCacheEngine._get_impl_with_lake`` /
``_get_impl_without_lake``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

from flexkv.common.source import BlockSource, LakeSource, LocalSource, PeerSource
from flexkv.common.transfer import (
    DeviceType,
    TransferOp,
    TransferOpGraph,
    TransferType,
)
from flexkv.common.type import CacheLocality, MatchResult


@dataclass
class MediaSegment:
    """A contiguous logical block range served by one storage medium.

    ``block_ids[i]`` is the source block for logical position
    ``logical_start + i``; physical block ids need not be contiguous.
    ``source`` names where the segment's blocks come from — a
    :class:`~flexkv.common.source.PeerSource` (single peer node id) for a
    cpu/ssd PEER segment, a :class:`~flexkv.common.source.LakeSource` (per-block
    PCFS file ids) for a LAKE segment, or a
    :class:`~flexkv.common.source.LocalSource` for a purely-local cpu/ssd
    segment.

    The route fields (``primary_type`` / ``needs_staging`` / ``needs_h2d``) are
    filled by :func:`plan_routes`; the op handles (``primary_op`` / ``staging``
    / ``h2d_op``) are filled by :func:`build_transfer_graph` and consumed by the
    promotion step.
    """

    tier: DeviceType
    locality: CacheLocality
    logical_start: int
    block_ids: np.ndarray
    source: BlockSource = field(default_factory=LocalSource)

    # Route (path selection).
    primary_type: Optional[TransferType] = None
    needs_staging: bool = False
    needs_h2d: bool = False

    # Graph handles.
    primary_op: Optional[TransferOp] = None
    staging: Optional[np.ndarray] = None
    h2d_op: Optional[TransferOp] = None

    def __post_init__(self) -> None:
        self.block_ids = np.asarray(self.block_ids, dtype=np.int64)

    @property
    def num_blocks(self) -> int:
        return len(self.block_ids)

    @property
    def logical_end(self) -> int:
        return self.logical_start + self.num_blocks

    @property
    def is_peer(self) -> bool:
        return self.locality == CacheLocality.PEER


class Route(NamedTuple):
    primary_type: Optional[TransferType]
    needs_staging: bool
    needs_h2d: bool


# Storage media in strict priority order: (tier, locality).  LOCAL reads the
# match's `.local` side, PEER its `.remote` side.  The greedy loop hands each
# medium, in this order, the longest ready extension it can append to the prefix
# already covered by higher-priority media.
_MEDIA_PRIORITY = (
    (DeviceType.CPU, CacheLocality.LOCAL),
    (DeviceType.CPU, CacheLocality.PEER),
    (DeviceType.SSD, CacheLocality.LOCAL),
    (DeviceType.SSD, CacheLocality.PEER),
    (DeviceType.LAKE, CacheLocality.LOCAL),
    (DeviceType.LAKE, CacheLocality.PEER),
)


def get_media_list(cpu_match: MatchResult,
                   ssd_match: MatchResult,
                   lake_match: Optional[MatchResult] = None,
                   *,
                   block_mask_start: int,
                   block_mask_end: int) -> List[MediaSegment]:
    """Greedily compose the tiers' ready prefixes into media segments.

    Each medium in :data:`_MEDIA_PRIORITY` extends the running ``cursor`` to the
    furthest ready block it reaches (``side.num_ready_matched_blocks``, capped by
    ``block_mask_end``), contributing the slice ``[cursor, reach)``.  The result
    is a contiguous segmentation of ``[block_mask_start, total_ready)``.
    """
    if block_mask_start < 0 or block_mask_end < block_mask_start:
        raise ValueError("Invalid block mask interval")

    matches = {
        DeviceType.CPU: cpu_match,
        DeviceType.SSD: ssd_match,
        DeviceType.LAKE: lake_match,
    }
    segments: List[MediaSegment] = []
    cursor = block_mask_start
    for tier, locality in _MEDIA_PRIORITY:
        match = matches[tier]
        if match is None:
            continue
        accel = (
            match.local if locality == CacheLocality.LOCAL else match.remote
        )
        if accel is None:
            continue
        reach = min(block_mask_end, accel.num_ready_matched_blocks)
        if reach <= cursor:
            continue
        block_ids = np.asarray(accel.physical_blocks[cursor:reach], dtype=np.int64)
        seg_source = accel.source.slice(cursor, reach)
        # Validate the origin matches what this medium needs. Check LAKE FIRST:
        # a LAKE-PEER segment carries per-block PCFS file ids (a LakeSource), not
        # a peer node id, and must never be routed to a peer.
        if tier == DeviceType.LAKE:
            if not isinstance(seg_source, LakeSource) or not seg_source.covers(
                reach - cursor
            ):
                raise ValueError(
                    f"{tier.name} {locality.value} segment is missing per-block "
                    "PCFS file ids"
                )
        elif locality == CacheLocality.PEER:
            if not isinstance(seg_source, PeerSource):
                raise ValueError(
                    f"{tier.name} peer segment is missing its peer node id"
                )
        segments.append(
            MediaSegment(tier, locality, cursor, block_ids, seg_source)
        )
        cursor = reach
    return segments


def route_of(seg: MediaSegment,
             *,
             enable_gpu: bool,
             enable_gds: bool,
             enable_peer_gpu: bool) -> Route:
    """Map a segment to its concrete transfer path.

    Peer cpu/ssd take a direct ``*2D`` route into GPU when ``enable_peer_gpu``,
    else stage to host and H2D.  Local SSD takes ``DISK2D`` when ``enable_gds``,
    else ``DISK2H`` + H2D.  Local CPU is served straight by H2D; LAKE always
    stages to host first.
    """
    if seg.tier == DeviceType.CPU:
        if not seg.is_peer:
            return Route(None, False, enable_gpu)
        if enable_peer_gpu:
            return Route(TransferType.PEERH2D, False, False)
        return Route(TransferType.PEERH2H, True, enable_gpu)

    if seg.tier == DeviceType.LAKE:
        return Route(TransferType.LAKE2H, True, enable_gpu)

    if seg.tier != DeviceType.SSD:
        raise ValueError(f"Unsupported GET source tier: {seg.tier}")

    if seg.is_peer:
        if enable_peer_gpu:
            return Route(TransferType.PEERSSD2D, False, False)
        return Route(TransferType.PEERSSD2H, True, enable_gpu)

    if enable_gds:
        return Route(TransferType.DISK2D, False, False)
    return Route(TransferType.DISK2H, True, enable_gpu)


def plan_routes(segments: List[MediaSegment],
                *,
                enable_gpu: bool,
                enable_gds: bool,
                enable_peer_gpu: bool) -> int:
    """Assign each segment's route in place; return the CPU staging blocks needed."""
    num_staging = 0
    for seg in segments:
        route = route_of(
            seg,
            enable_gpu=enable_gpu,
            enable_gds=enable_gds,
            enable_peer_gpu=enable_peer_gpu,
        )
        seg.primary_type = route.primary_type
        seg.needs_staging = route.needs_staging
        seg.needs_h2d = route.needs_h2d
        if route.needs_staging:
            num_staging += seg.num_blocks
    return num_staging


def build_transfer_graph(segments: List[MediaSegment],
                         *,
                         staging_blocks: np.ndarray,
                         block_mask_start: int,
                         layer_num: int) -> Tuple[TransferOpGraph, List[int], List[TransferOp]]:
    """Build the GET graph from routed segments (:func:`plan_routes` first).

    Fills each segment's ``staging`` / ``primary_op`` / ``h2d_op`` and returns
    ``(graph, finished_op_ids, h2d_ops)``.  One H2D op is emitted per consecutive
    run of staged/local-cpu segments; keeping runs separate preserves the exact
    ``gpu_block_offset`` when a direct route splits the ready prefix.

    Into-GPU op destinations are placeholders sized to their segment; the real
    GPU blocks are unknown until dispatch, so ``TransferOpGraph.set_gpu_blocks``
    rebinds each one via its ``gpu_block_offset``.
    """
    graph = TransferOpGraph()
    finished: List[int] = []

    staging_offset = 0
    for seg in segments:
        if seg.needs_staging:
            seg.staging = staging_blocks[staging_offset:staging_offset + seg.num_blocks]
            staging_offset += seg.num_blocks

    for seg in segments:
        if seg.primary_type is None:
            continue
        direct = seg.primary_type.name.endswith("2D")
        # Direct routes land in GPU; dst is a placeholder sized to the segment,
        # rebound by set_gpu_blocks via gpu_block_offset once the real GPU slots
        # arrive.  Staged routes write to CPU staging (known now).
        dst = (
            np.zeros(seg.num_blocks, dtype=np.int64) if direct else seg.staging
        )
        assert dst is not None
        op = TransferOp(
            graph_id=graph.graph_id,
            transfer_type=seg.primary_type,
            src_block_ids=seg.block_ids,
            dst_block_ids=dst,
            layer_id=0,
            layer_granularity=layer_num,
            source=seg.source,
            gpu_block_offset=(
                seg.logical_start - block_mask_start if direct else None
            ),
        )
        graph.add_transfer_op(op)
        seg.primary_op = op
        if direct or not seg.needs_h2d:  # `not seg.needs_h2d` for CPU-only GET
            finished.append(op.op_id)

    h2d_ops: List[TransferOp] = []
    i = 0
    n = len(segments)
    while i < n:
        if not segments[i].needs_h2d:  # direct 2D or CPU only GET
            i += 1
            continue
        run_start = i
        while i < n and segments[i].needs_h2d:  # merge continuous H2D to one op
            i += 1
        run = segments[run_start:i]

        src_parts = [
            s.staging if s.staging is not None else s.block_ids for s in run
        ]
        run_blocks = sum(s.num_blocks for s in run)
        h2d_op = TransferOp(
            graph_id=graph.graph_id,
            transfer_type=TransferType.H2D,
            src_block_ids=np.concatenate(src_parts),
            dst_block_ids=np.zeros(run_blocks, dtype=np.int64),
            layer_id=0,
            layer_granularity=layer_num,
            gpu_block_offset=run[0].logical_start - block_mask_start,
        )
        graph.add_transfer_op(h2d_op)
        for s in run:
            if s.primary_op is not None:  # local cpu not need
                graph.add_dependency(h2d_op.op_id, s.primary_op.op_id)
            s.h2d_op = h2d_op  # for promotion use
        finished.append(h2d_op.op_id)
        h2d_ops.append(h2d_op)

    return graph, finished, h2d_ops
