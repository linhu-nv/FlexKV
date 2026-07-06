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
uniformly: `match()` returns a `MatchResultAccel`, `insert` returns an opaque
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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from flexkv.common.debug import flexkv_logger
from flexkv.common.transfer import DeviceType
from flexkv.common.type import MatchResultAccel

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


_DEVICE_TYPE_NAMES = ['CPU', 'GPU', 'SSD', 'REMOTE']


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

    def size(self) -> int:
        return self.num_blocks

    def is_valid(self) -> bool:
        return self.hashes is not None or self.finalize is not None

    @property
    def is_inserted(self) -> bool:
        """True if this handle is auto-protected by an armed insert finalize."""
        return self.finalize is not None

    def run_finalize(self) -> None:
        """Idempotent one-shot: run the armed finalize then disarm."""
        if self.finalize is not None:
            self.finalize()
            self.finalize = None


def _ensure_shmradix():
    if shmradix is None:
        raise ImportError(
            "shmradix is not installed; install it from radixshmem repo "
            "(pip install -e radixshmem/python). Original error: "
            f"{_SHMRADIX_IMPORT_ERROR}"
        )


class CacheEngineRadixShmem:
    """Radixshmem-backed cache engine for one device (CPU / SSD / REMOTE).

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
                 protected_threshold: int = 2):
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

        # RadixClient always; the RadixServer is owned by the bootstrap process.
        self._tree = shmradix.RadixClient(shm_name)

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

    def close(self) -> None:
        self._tree = None

    # ---------- Hash-path ref helpers (matched nodes) ----------

    def _inc_ref_node(self, node: ShmRadixNode) -> None:
        if node.hashes is not None and node.length > 0:
            self._tree.inc_ref(node.hashes, node.start, node.length)

    def _dec_ref_node(self, node: ShmRadixNode) -> None:
        if node.hashes is not None and node.length > 0:
            self._tree.dec_ref(node.hashes, node.start, node.length)

    # ---------- Match / insert / lock ----------

    def match(self, sequence_meta: SequenceMeta) -> MatchResultAccel:
        sequence_meta.gen_hashes()
        # SequenceMeta.block_hashes is int64; radixshmem expects uint64. Same
        # byte width → view-cast is safe. Keep a contiguous copy on the handles
        # so later inc_ref/dec_ref/set_ready re-walks stay valid independent of
        # the SequenceMeta lifetime.
        hashes = np.ascontiguousarray(sequence_meta.block_hashes).view(np.uint64)

        # lock=True atomically inc_refs the ready prefix under the read_lock,
        # preventing another process from auto-evicting the matched slots (and
        # recycling our slot ids) between match() and the consuming transfer.
        qr = self._tree.query(hashes, local_only=True, lock=True, update_meta=True)

        ready_len = int(qr.ready_prefix_len)
        matched_len = int(qr.total_hit_length)

        # last_ready_node: protects [0, ready_len). Same object is reused as
        # pre_locked_node so the cache_engine hand-off (lock_node → release
        # pre-lock → callback unlock) accounts against one handle.
        if ready_len > 0:
            last_ready_node = ShmRadixNode(
                num_blocks=ready_len, hashes=hashes, start=0, length=ready_len)
        else:
            last_ready_node = None

        # last_node: governs the full matched range [0, matched_len). Used as
        # `protected_node` in take(); never carries an insert finalize.
        if matched_len > 0:
            last_node = ShmRadixNode(
                num_blocks=int(qr.local_hit_length) or matched_len,
                hashes=hashes, start=0, length=matched_len)
        else:
            last_node = None

        physical = np.asarray(qr.ready_prefix_slots, dtype=np.int64)

        return MatchResultAccel(
            num_ready_matched_blocks=ready_len,
            num_matched_blocks=matched_len,
            last_ready_node=last_ready_node,
            last_node=last_node,
            last_node_matched_length=matched_len,
            physical_blocks=physical,
            block_node_ids=None,
            matched_pos="local",
            # query(lock=True) atomically inc_ref'd [0, ready_len); the
            # cache_engine layer releases this exactly once (via unlock →
            # hash-path dec_ref on the same handle).
            pre_locked_node=last_ready_node,
        )

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
        if node.is_inserted:
            return
        self._inc_ref_node(node)

    def unlock(self, node: ShmRadixNode) -> None:
        if node is None or not node.is_valid():
            return
        if node.is_inserted:
            # One-shot: set_ready + dec_ref. Idempotent.
            node.run_finalize()
        else:
            self._dec_ref_node(node)

    def set_ready(self, node: ShmRadixNode, ready: bool, ready_length: int) -> None:
        if node is None or not node.is_valid():
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
