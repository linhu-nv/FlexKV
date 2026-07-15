from dataclasses import dataclass
from typing import Dict, OrderedDict as OrderedDictT, List, Set, Tuple
from collections import OrderedDict

from flexkv.common.transfer import TransferOp, TransferOpGraph, TransferType


class TransferScheduler:
    """Incremental, foreground-priority transfer graph scheduler.

    Two changes vs. the original O(N)-per-wake scheduler:

    (A) Incremental scheduling. The original schedule() re-scanned EVERY pending
        graph each wake (take_ready_ops + all_transfer_ops_completed), which is
        O(backlog). Under a submission burst the backlog explodes and each wake
        gets slower -> a positive-feedback stall (the P999 spike). But a graph
        can only change state when it is newly added or when one of its ops just
        finished: otherwise its _ready_ops set is byte-identical to last wake, so
        take_ready_ops() returns [] and all_transfer_ops_completed() cannot flip.
        So we only process the "dirty" set = newly-added graphs UNION graphs that
        own a finished op. Output is element-identical to the original; only the
        wasted scans on untouched graphs are skipped. Safe because the graph
        objects are mutated solely by the single TE scheduler thread (graphs
        arrive over an mp.Queue as pickled copies; workers return op_ids only).

    (B) Foreground priority. engine GET/PUT transfers (H2D/D2H — a request is
        synchronously waiting on them) go in a foreground bucket; external
        prefetch transfers (DISK2H — fire-and-forget warmup, nothing waits) go in
        a background bucket. Each wake we drain foreground dirty graphs BEFORE
        background ones, so a prefetch submission burst can no longer push engine
        H2D ops to the back of the FIFO. next_ops preserves order: all foreground
        ops (in graph insertion order) precede all background ops.
    """

    def __init__(self) -> None:
        # Foreground = engine (H2D/D2H); background = external prefetch (DISK2H).
        self._fg: OrderedDictT[int, TransferOpGraph] = OrderedDict()
        self._bg: OrderedDictT[int, TransferOpGraph] = OrderedDict()
        # graph_id -> is_prefetch, so a finished op can find its bucket + we can
        # pop the right dict on completion.
        self._graph_bucket: Dict[int, bool] = {}
        # Dirty graph ids for this round, kept as insertion-ordered sets (dict
        # with None values): O(1) add + dedup, preserves order == original
        # `for g in _transfer_graphs.values()` iteration order.
        self._dirty_fg: "OrderedDict[int, None]" = OrderedDict()
        self._dirty_bg: "OrderedDict[int, None]" = OrderedDict()

    def add_transfer_graph(self, graph: TransferOpGraph, is_prefetch: bool = False) -> None:
        """Add a new transfer graph to the scheduler.

        is_prefetch routes it to the background bucket (deprioritized). Default
        False keeps the non-prefetch / single-engine path in the foreground.
        """
        gid = graph.graph_id
        if is_prefetch:
            self._bg[gid] = graph
            self._dirty_bg[gid] = None
        else:
            self._fg[gid] = graph
            self._dirty_fg[gid] = None
        self._graph_bucket[gid] = is_prefetch

    def _drain(self,
               dirty: "OrderedDict[int, None]",
               graphs: OrderedDictT[int, TransferOpGraph],
               next_ops: List[TransferOp]) -> None:
        """Take ready ops from each dirty graph (in insertion order), appending
        real (non-VIRTUAL) ops to next_ops.

        Every ready op (VIRTUAL included) is appended to next_ops, exactly as the
        original did: the transfer_engine loop routes VIRTUAL ops to the
        completed_queue (a completion marker the CE side waits on) and real ops to
        a worker, so VIRTUAL ops MUST still be emitted. VIRTUAL ops are also
        marked completed here (they carry no data), which can unblock successors
        in the SAME graph. The original picked those successors up on the next
        wake (it re-scanned every graph each wake); the incremental scheduler
        can't rely on the graph being dirty again, so we reach a fixpoint within
        this round: re-run take_ready_ops() while the previous pass surfaced a
        VIRTUAL op. For graphs with no VIRTUAL ops (the common DISK2H->H2D case)
        this is one pass + one empty check == negligible. The ops emitted are
        identical to the original; VIRTUAL-unblocked ops are just emitted this
        round instead of the next wake."""
        for graph_id in dirty:
            graph = graphs.get(graph_id)
            if graph is None:
                continue
            while True:
                ready_op_ids = graph.take_ready_ops()
                if not ready_op_ids:
                    break
                saw_virtual = False
                for op_id in ready_op_ids:
                    op = graph._op_map[op_id]
                    if op.transfer_type == TransferType.VIRTUAL:
                        graph.mark_completed(op_id)
                        saw_virtual = True
                    next_ops.append(op)
                if not saw_virtual:
                    break

    def schedule(self,
                finished_ops: List[TransferOp]
               ) -> Tuple[List[int], List[TransferOp]]:
        """
        Schedule transfer operations (incremental, foreground-first).

        Returns:
            Tuple[List[int], List[TransferOp]]:
                - completed transfer graph ids (foreground first)
                - next executable transfer ops (all foreground before background,
                  each in graph insertion order)
        """
        # (a) Mark completed ops + record their owning graph as dirty.
        for op in finished_ops:
            is_pref = self._graph_bucket.get(op.graph_id)
            if is_pref is None:
                continue  # graph already completed/removed
            graphs = self._bg if is_pref else self._fg
            graph = graphs.get(op.graph_id)
            if graph is None:
                continue
            graph.mark_completed(op.op_id)
            (self._dirty_bg if is_pref else self._dirty_fg)[op.graph_id] = None

        # (b) Take next ready ops — foreground bucket first, then background.
        next_ops: List[TransferOp] = []
        self._drain(self._dirty_fg, self._fg, next_ops)
        self._drain(self._dirty_bg, self._bg, next_ops)

        # (c) Completion check — only for graphs touched this round (dirty).
        # A graph can only become fully-completed on a round where one of its
        # ops finished, i.e. it is in the dirty set. Foreground first.
        completed_graph_ids: List[int] = []
        for dirty, graphs in ((self._dirty_fg, self._fg), (self._dirty_bg, self._bg)):
            for graph_id in dirty:
                graph = graphs.get(graph_id)
                if graph is not None and graph.all_transfer_ops_completed():
                    completed_graph_ids.append(graph_id)

        # Remove completed graphs from their bucket + the bucket index.
        for graph_id in completed_graph_ids:
            is_pref = self._graph_bucket.pop(graph_id, False)
            (self._bg if is_pref else self._fg).pop(graph_id, None)

        # Dirty sets are fully consumed this round.
        self._dirty_fg.clear()
        self._dirty_bg.clear()

        return completed_graph_ids, next_ops
