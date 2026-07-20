# SPDX-License-Identifier: Apache-2.0
"""
RadixShmem-backed CacheEngine.

A drop-in replacement for `flexkv.cache.cache_engine.CacheEngineAccel` whose
RadixTree + slot Mempool live in POSIX shared memory (via the `shmradix`
package). Every DP scheduler process can attach to the same shm region and run
prefix queries / inserts in parallel, serialised only by a process-shared
rwlock.

Public surface mirrors `CacheEngineAccel` so that `GlobalCacheEngine` and the
`_get_impl_*`/`_put_*` helpers in `cache_engine.py` treat both backends
uniformly: `match()` returns a `MatchResult`, `insert` returns an opaque
"node" handle with `.size()`, and `lock_node`/`unlock`/`set_ready` operate on
that handle.

==============================  API model  ==============================

This module targets the CURRENT shmradix API, which deliberately does NOT
expose node-id handles (a node id is invalidated by a later split). Instead:

  * `RadixServer` / `RadixClient`        (renamed from TreeServer/TreeClient)
  * `query(hashes, lock=, update_meta=)` returns a `QueryResult` with
    `ready_prefix_len`, `total_hit_length`, `ready_prefix_slots`, and a
    one-shot `finalize` (armed when lock=True).
  * `insert_with_slots(hashes, slots, is_ready=, lock=)` returns an
    `InsertResult` with `matched_prefix`, `inserted_count`, and a one-shot
    `finalize` (armed when is_ready=False / lock=True; it performs
    set_ready + dec_ref in a single call).
  * State ops are HASH-PATH based and split-invariant:
    `set_ready(hashes, start, length, ready)`,
    `inc_ref(hashes, start, length)`, `dec_ref(hashes, start, length)`.

`ShmRadixNode` therefore carries the hash path (`hashes`, `start`, `length`)
plus an optional armed `finalize`. Two flavours flow through `cache_engine.py`:

  - **matched node** (from `match()` / `query(lock=True)`): protected by the
    query's atomic inc_ref. It participates in cache_engine's hand-off
    protocol (lock_node → release pre-lock → callback unlock), which needs
    independent counter inc/dec, so lock_node/unlock map to HASH-PATH
    inc_ref/dec_ref over [0, ready_prefix_len).

  - **inserted node** (from `insert(is_ready=False)`): auto-locked by shmradix,
    carrying an armed `finalize` (= set_ready + dec_ref). `lock_node` on it is
    a NO-OP (it is already protected); the callback's `set_ready` is a no-op
    and `unlock` calls `finalize()` once. This is the "prefer finalize"
    write-path收尾.

Slot IDs returned by shmradix are `int32`; FlexKV expects `int64`. We cast at
the boundary.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from flexkv.common.debug import flexkv_logger
from flexkv.common.transfer import DeviceType
from flexkv.common.type import (
    MatchResult,
    MatchResultAccel,
)

if TYPE_CHECKING:
    from flexkv.common.block import SequenceMeta
    from flexkv.integration.dynamo.collector import KVEventCollector

try:
    import shmradix
except ImportError as e:  # pragma: no cover
    shmradix = None
    _SHMRADIX_IMPORT_ERROR = e
else:
    _SHMRADIX_IMPORT_ERROR = None


_DEVICE_TYPE_NAMES = ['CPU', 'GPU', 'SSD', 'LAKE']


@dataclass
class ShmRadixNode:
    """Hash-path handle into the shared radix tree.

    Replaces the old "bare node_id" handle. Carries the prefix path so that
    `lock_node`/`unlock`/`set_ready` can re-walk the tree (split-invariant),
    plus an optional armed `finalize` for the insert write-path收尾.

    Fields:
      num_blocks : value returned by `.size()` (matched/inserted block count).
      hashes     : uint64 prefix path copy (kept alive for re-walks).
      start,length : the sub-range of `hashes` this handle governs.
      finalize   : armed FinalizeFn from insert(is_ready=False); when set, this
                   handle is an "inserted node" and is auto-protected by
                   shmradix until finalize() runs (set_ready + dec_ref).
    """
    num_blocks: int
    hashes: Optional[np.ndarray] = None
    start: int = 0
    length: int = 0
    finalize: object = None
    query_finalize: object = None

    def size(self) -> int:
        return self.num_blocks

    def is_valid(self) -> bool:
        return (
            self.hashes is not None or self.finalize is not None or
            self.query_finalize is not None
        )

    @property
    def is_inserted(self) -> bool:
        """True if this handle is auto-protected by an armed insert finalize."""
        return self.finalize is not None

    @property
    def is_query_guard(self) -> bool:
        return self.query_finalize is not None

    def run_finalize(self) -> None:
        """Idempotent one-shot: run the armed finalize then disarm."""
        if self.finalize is not None:
            self.finalize()
            self.finalize = None

    def run_query_finalize(self) -> None:
        if self.query_finalize is not None:
            self.query_finalize()
            self.query_finalize = None


def _ensure_shmradix():
    if shmradix is None:
        raise ImportError(
            "shmradix is not installed; install it from radixshmem repo "
            "(pip install -e radixshmem/python). Original error: "
            f"{_SHMRADIX_IMPORT_ERROR}"
        )


class CacheEngineRadixShmem:
    """Radixshmem-backed cache engine for one device (CPU / SSD / LAKE).

    Multiple instances (one per DP scheduler process) attach to the same shm
    region by name and concurrently query / insert.
    """

    def __init__(self,
                 device_type: DeviceType,
                 num_total_blocks: int,
                 tokens_per_block: int,
                 shm_name: str,
                 # tokens_per_block=-1 means "recover it from the region" via
                 # RadixClient.block_size() (written by the owner on create).
                 evict_ratio: float = 0.05,
                 evict_start_threshold: float = 1.0,
                 hit_reward_seconds: int = 0,
                 eviction_policy: str = "lru",
                 event_collector: Optional[KVEventCollector] = None,
                 metrics_collector=None,
                 protected_threshold: int = 2,
                 peer_enabled: bool = False,
                 redis_meta=None,
                 radix_cluster_id: str = "default"):
        """Attach to an existing radix shm region by name. The RadixServer
        owning the region must have been created elsewhere (e.g. by
        `flexkv.server.shm_radix_bootstrap.create_shm_radix_regions`)."""
        _ensure_shmradix()

        if eviction_policy != "lru":
            flexkv_logger.warning(
                f"radixshmem only supports LRU eviction; ignoring "
                f"eviction_policy={eviction_policy!r}"
            )
        if hit_reward_seconds != 0 or protected_threshold != 2:
            flexkv_logger.debug(
                "radixshmem ignores hit_reward_seconds and protected_threshold"
            )

        self.device_type = device_type
        self.num_total_blocks = num_total_blocks
        self.evict_ratio = evict_ratio
        self.evict_start_threshold = evict_start_threshold
        self.shm_name = shm_name

        self.event_collector = event_collector
        self._metrics_collector = metrics_collector
        self.peer_enabled = bool(peer_enabled)
        self._redis_meta = redis_meta
        self._radix_cluster_id = radix_cluster_id
        self._peer_node_ids = {}
        self._trace_peer = os.getenv("FLEXKV_TRACE_RADIX_PEER", "0") == "1"
        # RadixClient always; the RadixServer is owned by the bootstrap process.
        self._tree = shmradix.RadixClient(shm_name)
        if self.peer_enabled and not self._tree.is_distributed():
            flexkv_logger.warning(
                "radixshmem peer matching is enabled but the attached region "
                "has world_size=1; GETs will remain local-only"
            )

        # -1 => recover tokens_per_block from the region itself.
        if tokens_per_block is None or tokens_per_block < 0:
            tokens_per_block = int(self._tree.block_size())
        self.tokens_per_block = tokens_per_block

        # Diagnostics: how often does the insert race fire (caller supplied
        # excess slots / matched_prefix advanced)?
        self._insert_count = 0
        self._race_count = 0
        self._unused_slot_total = 0
        self._race_log_interval = 50

    # ---------- Mempool view (compatibility shims) ----------

    @property
    def mempool(self) -> _MempoolView:
        return _MempoolView(self._tree)

    # ---------- Lifecycle ----------

    def reset(self) -> None:
        flexkv_logger.warning(
            "CacheEngineRadixShmem.reset(): radixshmem has no in-place reset; "
            "tear down and recreate the shm region instead."
        )

    def start(self) -> None:
        """Compatibility with the peer-capable cache engine lifecycle."""

    def _resolve_peer_node_id(self, radix_rank: int) -> int:
        cached = self._peer_node_ids.get(radix_rank)
        if cached is not None:
            return cached
        if self._redis_meta is None:
            raise RuntimeError(
                "radixshmem peer match requires Redis peer metadata"
            )
        node_id = self._redis_meta.resolve_radix_rank(
            self._radix_cluster_id, radix_rank
        )
        if node_id is None:
            raise RuntimeError(
                f"No active FlexKV node is registered for radix rank {radix_rank}"
            )
        self._peer_node_ids[radix_rank] = int(node_id)
        return int(node_id)

    def close(self) -> None:
        self._tree = None

    def publish_ready(self, node: ShmRadixNode) -> None:
        """Make asynchronous distributed RHT publications visible."""
        if self.peer_enabled:
            self._tree.flush()

    # ---------- Hash-path ref helpers (matched nodes) ----------

    def _inc_ref_node(self, node: ShmRadixNode) -> None:
        if node.hashes is not None and node.length > 0:
            self._tree.inc_ref(node.hashes, node.start, node.length)

    def _dec_ref_node(self, node: ShmRadixNode) -> None:
        if node.hashes is not None and node.length > 0:
            self._tree.dec_ref(node.hashes, node.start, node.length)

    # ---------- Match / insert / lock ----------

    def match(self,
              sequence_meta: SequenceMeta,
              *,
              with_peer: bool = True,
              gpu_matched_blocks: int = 0) -> MatchResult:
        """Query the shared tree, returning a ``MatchResult(local, remote)``.

        Two shapes come out of this one code path:
          * ``with_peer=False`` (or a non-distributed tree) → a LOCAL-only
            result: ``remote is None`` and the guard rides ``local``.
          * ``with_peer=True`` on a distributed tree → LOCAL + PEER: ``local``
            covers the local ready prefix and ``remote`` extends it with the
            peer suffix.

        A single distributed query serves both sides at once, so they share ONE
        atomic guard: the ``pre_locked_node`` (and ``last_ready_node``) live on
        whichever side reaches furthest — ``remote`` when a peer suffix exists,
        otherwise ``local`` — and the cache_engine releases it exactly once.

        ``gpu_matched_blocks`` is accepted only for interface parity with the
        accel/hie engines; radixshmem matches the full sequence, so it is unused.
        """
        local_only = not (with_peer and self.peer_enabled)
        sequence_meta.gen_hashes()
        # SequenceMeta.block_hashes is int64; radixshmem expects uint64. Same
        # byte width → view-cast is safe. Keep a contiguous copy on the handles
        # so later inc_ref/dec_ref/set_ready re-walks stay valid independent of
        # the SequenceMeta lifetime.
        hashes = np.ascontiguousarray(sequence_meta.block_hashes).view(np.uint64)

        # lock=True atomically inc_refs the ready prefix under the read_lock,
        # preventing another process from auto-evicting the matched slots (and
        # recycling our slot ids) between match() and the consuming transfer.
        qr = self._tree.query(
            hashes,
            local_only=local_only,
            lock=True,
            update_meta=True,
        )
        if getattr(self, "_trace_peer", False):
            flexkv_logger.info(
                "[RADIX PEER QUERY] "
                f"shm={self.shm_name} local_only={local_only} "
                f"blocks={len(hashes)} first_hash="
                f"{int(hashes[0]) if len(hashes) else None} "
                f"ready={int(qr.ready_prefix_len)} "
                f"local={int(qr.local_hit_length)} "
                f"total={int(qr.total_hit_length)} "
                f"remote_node={int(qr.remote_node_id)} "
                f"remote_hit={int(qr.remote_hit_length)} "
                f"rdma_reads={int(qr.rdma_read_count)} "
                f"rdma_atomics={int(qr.rdma_atomic_count)}"
            )

        # --- decompose the query into local-ready and peer-ready spans --------
        # radixshmem contract: `ready_prefix_slots` holds ONLY the local-tree
        # ready slots, so its length is the LOCAL ready prefix. `ready_prefix_len`
        # equals that in local_only mode, but once a distributed walk extends the
        # ready prefix it becomes the TOTAL (local + peer) ready — so recover the
        # local span with min(), and the peer tail is carried in `remote_slots`.
        ready_len = int(qr.ready_prefix_len)            # total ready (local + peer)
        matched_len = int(qr.total_hit_length)          # total matched (incl. unready)
        local_hit = int(qr.local_hit_length)            # local matched length
        local_ready_len = min(local_hit, ready_len)     # == len(qr.ready_prefix_slots)
        remote_ready_len = ready_len - local_ready_len  # peer suffix == len(qr.remote_slots)
        has_peer = remote_ready_len > 0

        local_ready_blocks = np.asarray(qr.ready_prefix_slots, dtype=np.int64)
        if len(local_ready_blocks) != local_ready_len:
            raise RuntimeError(
                "radixshmem returned inconsistent local ready slot metadata"
            )

        # --- one atomic guard, on the furthest-reaching side ------------------
        # query(lock=True) took a single evict-protection ref over the whole
        # ready prefix — including, for a peer hit, a remote RDMA atomic ref;
        # `qr.finalize` releases all of it in one call. Hand that finalize to a
        # single guard node and attach it to whichever side reaches furthest
        # (`remote` when a peer suffix exists, else `local`), so the peer slots
        # stay evict-protected until the graph callback fires it.
        if ready_len > 0:
            query_guard = ShmRadixNode(
                num_blocks=ready_len,
                hashes=hashes,
                start=0,
                length=local_ready_len,
                query_finalize=qr.finalize,
            )
        else:
            qr.finalize()
            query_guard = None

        # last_node governs the full LOCAL matched range [0, local_hit); it is
        # take()'s protected_node and never carries an insert finalize.
        local_last_node = (
            ShmRadixNode(num_blocks=local_hit, hashes=hashes, start=0,
                         length=local_hit)
            if local_hit > 0 else None
        )

        # `local` always describes [0, local_ready_len). It carries the shared
        # guard ONLY when there is no peer suffix to carry it instead.
        local = MatchResultAccel(
            num_ready_matched_blocks=local_ready_len,
            num_matched_blocks=local_hit,
            last_ready_node=None if has_peer else query_guard,
            last_node=local_last_node,
            last_node_matched_length=local_hit,
            physical_blocks=local_ready_blocks,
            pre_locked_node=None if has_peer else query_guard,
        )
        if not has_peer:
            # Local-only result: the guard (if any) protects [0, ready_len).
            return MatchResult(local=local)

        # --- peer suffix: build `remote`; the shared guard rides here ---------
        remote_ready_blocks = np.asarray(qr.remote_slots, dtype=np.int64)
        if len(remote_ready_blocks) != remote_ready_len:
            qr.finalize()
            raise RuntimeError(
                "radixshmem returned inconsistent remote ready slot metadata"
            )
        try:
            peer_node_id = self._resolve_peer_node_id(int(qr.remote_node_id))
        except Exception:
            qr.finalize()
            raise

        # Peer-inclusive physical view indexed by logical block: the local ready
        # slots followed by the peer suffix. The planner slices only the
        # [local_ready_len, ready_len) tail (the local prefix is served by
        # `local`), so the -1 placeholder node ids below local_ready_len are
        # never read.
        remote_physical = np.concatenate([local_ready_blocks, remote_ready_blocks])
        remote_node_ids = np.concatenate([
            np.full(local_ready_len, -1, dtype=np.int64),
            np.full(remote_ready_len, peer_node_id, dtype=np.int64),
        ])
        remote = MatchResultAccel(
            num_ready_matched_blocks=ready_len,
            num_matched_blocks=matched_len,
            last_ready_node=query_guard,
            physical_blocks=remote_physical,
            block_node_ids=remote_node_ids,
            pre_locked_node=query_guard,
        )
        return MatchResult(local=local, remote=remote)

    def insert(self,
               sequence_meta: SequenceMeta,
               physical_block_ids: np.ndarray,
               num_insert_blocks: int = -1,
               is_ready: bool = True,
               match_result: Optional[MatchResultAccel] = None
               ) -> tuple[Optional[ShmRadixNode], np.ndarray]:
        """Attach `physical_block_ids` as a suffix in the shared radix tree.

        Returns (node, unused_slots) where:
          - `node` is the inserted leaf handle (or None if nothing attached).
            When inserted with is_ready=False it carries the armed finalize
            (= set_ready + dec_ref) — released later via `unlock(node)`.
          - `unused_slots` (int64) is the subset of `physical_block_ids` that
            shmradix did NOT attach (matched_prefix advanced, or the caller
            supplied excess slots). The current shmradix `insert_with_slots`
            consumes the supplied pool front-first and inserts exactly
            `inserted_count` of them, so unused = slots[inserted_count:]. The
            caller MUST recycle these only AFTER any in-flight transfer that
            references them has completed.
        """
        sequence_meta.gen_hashes()
        hashes = np.ascontiguousarray(sequence_meta.block_hashes).view(np.uint64)

        suffix_slots = np.asarray(physical_block_ids, dtype=np.int32)

        if num_insert_blocks > 0:
            target_hashes = hashes[:num_insert_blocks]
        else:
            target_hashes = hashes
        target_hashes = np.ascontiguousarray(target_hashes)

        result = self._tree.insert_with_slots(
            target_hashes, suffix_slots, is_ready=is_ready
        )

        inserted = int(result.inserted_count)
        matched_prefix = int(result.matched_prefix)

        # unused = supplied slots beyond what was actually attached.
        unused_slots = suffix_slots[inserted:]
        unused_slots_i64 = np.asarray(unused_slots, dtype=np.int64)

        # Diagnostics.
        self._insert_count += 1
        num_unused = len(unused_slots_i64)
        if num_unused > 0:
            self._race_count += 1
            self._unused_slot_total += num_unused
        if self._insert_count % self._race_log_interval == 0:
            race_pct = 100.0 * self._race_count / max(1, self._insert_count)
            flexkv_logger.info(
                f"[shmradix race-counter device={_DEVICE_TYPE_NAMES[self.device_type]}] "
                f"inserts={self._insert_count} race_hits={self._race_count} "
                f"({race_pct:.2f}%) cumulative_unused_slots={self._unused_slot_total}"
            )

        if self.event_collector is not None and inserted > 0:
            attached_hashes = sequence_meta.block_hashes[
                matched_prefix : matched_prefix + inserted
            ]
            self.event_collector.publish_stored(
                block_hashes=attached_hashes,
                block_size=self.tokens_per_block,
                medium=_DEVICE_TYPE_NAMES[self.device_type]
            )

        if inserted <= 0:
            # Nothing attached → drop any armed finalize (no protection taken).
            return None, unused_slots_i64

        # Build the inserted-node handle. For is_ready=False the finalize is
        # armed (set_ready + dec_ref over [matched_prefix, matched_prefix+
        # inserted)); for is_ready=True insert takes no ref and finalize is a
        # no-op, so the handle is hash-path only.
        fin = result.finalize if (result.finalize and bool(result.finalize)) else None
        node = ShmRadixNode(
            num_blocks=inserted,
            hashes=hashes,
            start=matched_prefix,
            length=inserted,
            finalize=fin,
        )
        return node, unused_slots_i64

    def lock_node(self, node: ShmRadixNode) -> None:
        if node is None or not node.is_valid():
            return
        # Inserted nodes are already protected by their armed finalize — taking
        # another ref here would leak it. Matched nodes take an independent
        # hash-path ref (the cache_engine hand-off then drops the match's
        # pre-lock).
        if node.is_inserted or node.is_query_guard:
            return
        self._inc_ref_node(node)

    def unlock(self, node: ShmRadixNode) -> None:
        if node is None or not node.is_valid():
            return
        if node.is_query_guard:
            node.run_query_finalize()
            return
        if node.is_inserted:
            # One-shot: set_ready + dec_ref. Idempotent.
            node.run_finalize()
            if self.peer_enabled:
                # Publish the single insert's RHT updates after its data is
                # marked ready.
                self._tree.flush()
        else:
            self._dec_ref_node(node)
            return
        if getattr(self, "_trace_peer", False):
            verify = self._tree.query(
                node.hashes,
                local_only=True,
                lock=False,
                update_meta=False,
            )
            flexkv_logger.info(
                "[RADIX PEER READY] "
                f"shm={self.shm_name} nodes=1 "
                f"flushed={self.peer_enabled} "
                f"local_ready={int(verify.ready_prefix_len)} "
                f"local_total={int(verify.total_hit_length)}"
            )

    def set_ready(self, node: ShmRadixNode, ready: bool, ready_length: int) -> None:
        if node is None or not node.is_valid():
            return
        if node.is_query_guard:
            return
        if node.is_inserted:
            # Handled atomically by the armed finalize in unlock(); the
            # callback always calls set_ready THEN unlock, so flipping ready
            # here would be redundant (and finalize only fires ready=True).
            return
        # Matched node: already ready, but honour an explicit request via the
        # hash path (idempotent, split-invariant).
        if node.hashes is not None and node.length > 0:
            self._tree.set_ready(node.hashes, node.start, node.length, bool(ready))

    def set_ready_path(self,
                       sequence_meta: SequenceMeta,
                       start: int,
                       length: int,
                       ready: bool) -> None:
        """Path-based set_ready that walks a SequenceMeta hash path."""
        sequence_meta.gen_hashes()
        hashes = np.ascontiguousarray(sequence_meta.block_hashes).view(np.uint64)
        self._tree.set_ready(hashes, start, length, ready)

    # ---------- Mempool ops (take/recycle) ----------

    def take(self,
             num_required_blocks: int,
             protected_node: Optional[ShmRadixNode] = None,
             strict: bool = True) -> np.ndarray:
        """Allocate `num_required_blocks` slots from radixshmem's mempool.

        radixshmem's `allocate_slots` auto-evicts LRU entries to satisfy the
        request. `protected_node` is held across the call (hash-path inc_ref /
        dec_ref) so it is not evicted mid-allocation.
        """
        protect = protected_node is not None and protected_node.is_valid()
        if protect:
            self._inc_ref_node(protected_node)
        try:
            slots_i32 = self._tree.allocate_slots(num_required_blocks)
        finally:
            if protect:
                self._dec_ref_node(protected_node)

        slots = np.asarray(slots_i32, dtype=np.int64)

        if strict and len(slots) < num_required_blocks:
            self._tree.recycle_slots(np.asarray(slots, dtype=np.int32))
            raise RuntimeError(
                f"radixshmem: not enough free blocks to take, required: "
                f"{num_required_blocks}, available: {len(slots)}"
            )

        if self._metrics_collector is not None and len(slots) > 0:
            self._metrics_collector.record_allocation(
                _DEVICE_TYPE_NAMES[self.device_type].lower(), len(slots)
            )
        return slots

    def recycle(self, physical_blocks: np.ndarray) -> None:
        if physical_blocks is None or len(physical_blocks) == 0:
            return
        slots_i32 = np.asarray(physical_blocks, dtype=np.int32)
        self._tree.recycle_slots(slots_i32)

    # ---------- Stats passthrough ----------

    @property
    def num_free_blocks(self) -> int:
        return int(self._tree.mempool_free())

    @property
    def num_used_blocks(self) -> int:
        return int(self._tree.mempool_used())

    @property
    def total_nodes(self) -> int:
        return int(self._tree.total_radix_nodes())


@dataclass
class _MempoolView:
    """Read-only mempool view for `engine.mempool.num_free_blocks` etc."""
    _tree: object

    @property
    def num_total_blocks(self) -> int:
        return int(self._tree.mempool_total())

    @property
    def num_free_blocks(self) -> int:
        return int(self._tree.mempool_free())

    @property
    def num_used_blocks(self) -> int:
        return int(self._tree.mempool_used())
