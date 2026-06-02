# NUMA-aware CPU pool for FlexKV

**Status:** Phase 1.5 — control plane + cache engine integration done. Opt-in
via `cache_config.enable_numa_aware`. Verified on a real 2-NUMA H20 box:
topology discovery, `mbind`-pinned shm pools (100% pages on target node),
cross-process IPC preserves placement, `cudaHostRegister` on the pinned
shm pool succeeds, and `GlobalCacheEngine.put / get` emit H2D / D2H ops
with the right `home_numa_id` including the cross-DP cross-NUMA hit case.
What's left for Phase 2 is the SSD/Remote/GDS path and arrangement (b)
(TP crosses NUMA).

## 1. Why

FlexKV today allocates **one big CPU tensor** as the offload pool. The
pool is allocated in the TransferManager process, shared to per-DP
transfer worker subprocesses via `torch.multiprocessing`, and pinned
(`cudaHostRegister`) in each worker. With no NUMA hint, all physical
pages end up on whichever NUMA node the allocator thread happened to
live on. On dual-socket machines this gives ~30–50% cross-NUMA penalty
on every H2D/D2H crossing a UPI/Infinity-Fabric hop.

Pinning the pool **after** the pages have landed prevents any later
migration (`cudaHostRegister` locks the pages), so NUMA placement must
be decided up-front. And because the same physical pool needs to be
read/written by workers in **multiple processes** (one per DP), we
can't use `numa_alloc_onnode` directly — its anonymous mmap can't be
shared cross-process.

## 2. Design

### 2.1 Block-level invariant: no duplication

A logical KV-cache block lives in **exactly one** NUMA pool. When a DP
group on NUMA *x* hits a block whose home is NUMA *y*, we accept the
cross-NUMA DMA cost rather than pay double the CPU memory by
replicating. CPU is the limiting resource on most deployments.

### 2.2 Two arrangements

The shape of "per-pool block size" depends on whether a TP group fits
into a single NUMA:

| Arrangement | TP group fits in 1 NUMA? | per-pool `block_size` | per-pool block count | total CPU |
|---|---|---|---|---|
| **(a) TP_WITHIN_NUMA** | yes (vLLM default) | unchanged | `total / N_numa` | `total` |
| **(b) TP_CROSSES_NUMA** | no | `orig × sub_tp / tp_size` | `total` | `total` |

Phase 1 implements (a) only. (b) is gated behind
`cache_config.allow_tp_crosses_numa` and raises `NotImplementedError`
until Phase 2.

Concrete example — TP=2 × DP=4 on 8 GPUs across 2 NUMA (vLLM contiguous
TP groups: DP0={0,1} DP1={2,3} on NUMA0; DP2={4,5} DP3={6,7} on NUMA1):
- 2 pools.
- Per pool: same `block_stride` as before (TP doesn't cross NUMA), half the block count.
- DP0/DP1's home pool = pool 0; DP2/DP3's home pool = pool 1.
- Cache shared across all DPs via a single radix tree; cross-DP cross-NUMA hits go cross-NUMA DMA.

### 2.3 Routing rule

```
op.home_numa_id ← set by cache engine when allocating physical CPU slots
TransferEngine._assign_op_to_worker(op):
    if numa_plan enabled and op.transfer_type in {H2D, D2H}:
        worker = workers_by_pair[(op.dp_id, op.home_numa_id)]
    else:
        worker = workers[op.dp_id]      # legacy
    worker.submit(op)
```

Workers are spawned per `(dp_id, numa_pool_index)` pair: for TP2×DP4
×2 NUMA that's 8 H2D + 8 D2H = 16 worker processes total. Each worker
pins its assigned NUMA pool (shm-backed) inside its own process; CUDA
tracks pin state per-context so independent pin calls from different
processes on the same shm region all succeed.

### 2.4 How the NUMA placement actually sticks

The chain that makes pages physically live on the right node:

1. Main process creates `/dev/shm/flexkv_numa_<hint>_<idx>_<pid>_<uuid>` and ftruncates to the pool's byte size.
2. Wraps it via `torch.UntypedStorage.from_file(shared=True)`. The tensor is now backed by shared memory but the pages haven't been faulted yet.
3. Calls `mbind(MPOL_BIND, node)` on the tensor's VA range. New page faults from *any* process touching this mapping must come from `node`.
4. Spawns a dedicated thread, pins it to a CPU of `node` via `numa_run_on_node`, then `memset(0)` over the whole tensor. This forces every page to be first-faulted on `node` *before* worker subprocesses touch it.
5. Workers receive the tensor via `torch.multiprocessing`. The reducer sees it's already file-backed shared memory and passes only the filename — no copy. The workers `mmap` the same file and call `cudaHostRegister`.

Both `mbind` and `numa_run_on_node` are best-effort: if libnuma fails
or returns EAGAIN beyond a small retry budget, the allocation still
succeeds and `NumaAllocResult.numa_bound` is set to `False`. The
storage engine logs a warning so an operator can detect the silent
fallback.

## 3. Code layout

```
flexkv/numa/
  topology.py    — sysfs / NVML scan, GPU → NUMA mapping, override hook
  planner.py     — NumaPlan, build_numa_plan(arrangement detection)
  allocator.py   — numa_alloc_tensor (shm + mbind + NUMA-bound first-touch)
  mempool.py     — NumaMempool (per-pool free lists + global id space)

flexkv/common/config.py
  CacheConfig.enable_numa_aware        # master toggle (default False)
  CacheConfig.numa_gpu_map             # optional explicit GPU→NUMA override
  CacheConfig.allow_tp_crosses_numa    # Phase 2 escape hatch (must be False today)

flexkv/common/transfer.py
  TransferOp.home_numa_id              # routing field (-1 = legacy dispatch)

flexkv/common/storage.py
  StorageHandle.numa_node              # which NUMA node the tensor lives on
  StorageHandle.numa_pool_index        # position in NumaPlan.pool_nodes

flexkv/storage/allocator.py
  CPUAllocator.allocate_on_numa_node   # shm + libnuma path

flexkv/storage/storage_engine.py
  StorageEngine.allocate_cpu_pools_per_numa(plan)
  StorageEngine.get_cpu_pool_handles()
  StorageEngine.num_cpu_pools()

flexkv/transfer/transfer_engine.py
  TransferEngine.__init__              # accepts cpu_handles_per_numa + numa_plan
  TransferEngine._init_workers         # spawns one worker per (dp, numa)
  TransferEngine._create_gpu_cpu_worker
  TransferEngine._assign_op_to_worker  # NUMA-aware routing for H2D/D2H

flexkv/transfer_manager.py
  TransferManager._build_numa_plan
  TransferManager.initialize_transfer_engine  # detect topology + allocate pools

tests/numa/                  # pure-Python unit tests (no CUDA needed)
  test_topology.py
  test_planner.py            # TP2×DP4, TP1×DP4, uneven splits, arrangement-b rejection
  test_mempool.py
  test_allocator.py          # shm + mbind end-to-end
```

## 4. What's done in Phase 1 + 1.5

✅ Topology discovery (sysfs + NVML, with explicit override).
✅ NUMA planner with arrangement detection.
✅ Shared-memory NUMA-pinned allocator (libnuma mbind + first-touch).
✅ `NumaMempool` (multi-pool free list, global block-id space).
✅ `StorageEngine` per-NUMA pool support, layouts per pool.
✅ `TransferEngine` spawns `(dp, numa)` workers and routes ops by
   `home_numa_id`.
✅ `GlobalCacheEngine` uses `NumaMempool` for the CPU side when
   `enable_numa_aware=True`. The `take()` API gained a
   `home_numa_pool` hint that the engine plumbs from the requesting
   DP's home pool (`dp_to_home_pool[dp_id]`).
✅ Every emitted H2D / D2H op carries `op.home_numa_id`:
   - D2H: the home pool of the DP that triggered the put (the same
     pool we allocated into).
   - H2D: looked up from `NumaMempool.pools_of(src_cpu_blocks)`. When
     a single op's source blocks span multiple pools (the cross-DP
     cross-NUMA cache hit), the helper fans out into one H2D op per
     home pool, preserving dependencies and `finished_ops_ids`.
✅ Config knobs and CLI plumbing.
✅ Backward compat: `enable_numa_aware=False` is the default and
   preserves the legacy single-pool behavior bit-for-bit. `_AllocatorAdapter`
   transparently wraps either `Mempool` or `NumaMempool`.
✅ 31 unit tests + 4 real-hardware integration scripts (`tests/numa/integration_*_real.py`)
   validated on 8× H20-3e / 2 NUMA / 2 TB RAM.

## 5. What's NOT done yet (Phase 2)

❌ **SSD / Remote / GDS paths.** When `enable_numa_aware=True`,
   the transfer engine errors out if any of `enable_ssd /
   enable_remote / enable_gds / enable_p2p_*` is on. To support them
   we need either per-NUMA SSD workers (per-pool pinning) or a single
   SSD worker that knows about all pools and dispatches per-op. The
   `enable_p2p_cpu` path is also explicitly blocked at GlobalCacheEngine
   constructor today.

❌ **Arrangement (b) — TP crosses NUMA.** Requires:
   - Per-pool layout with `block_stride = orig × sub_tp / tp_size`.
   - `NumaMempool` "shared-id-space" mode (one block id present in
     all pools as one slice each).
   - C++ `tp_group_transfer` extended to accept multiple CPU pool
     pointers and per-rank pool routing (or fan out an op to multiple
     per-pool workers at the Python layer).

❌ **Per-pool eviction.** The radix tree's eviction picks blocks by
   LRU globally and recycles each into its home pool. If pool 0 fills
   up faster than pool 1 (skewed workload), evictions may free pool 1
   blocks even though pool 0 was the one under pressure. `take()`
   returns short or raises in that case; the caller bails the request.
   A proper fix is per-pool LRU rings inside the radix tree.

❌ **C++ side never changed.** Phase 1 reuses the existing
   `TPTransferThreadGroup` unchanged — we just hand it the per-NUMA
   pool tensor. Phase 2 will likely refactor it.

## 6. Config knobs

```python
CacheConfig(
    enable_numa_aware=True,
    # Optional: override sysfs/NVML lookup. Keys are CUDA device ids
    # (global, the same id vLLM passes via KVTPClient).
    numa_gpu_map={0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1},
    # Phase 2 escape hatch, must be False today.
    allow_tp_crosses_numa=False,
)
```

Failure modes (all soft, the engine falls back to a single pool with a
warning, **except** when arrangement (b) is detected without
`allow_tp_crosses_numa=True` — that's a hard error):

| Situation | Behavior |
|---|---|
| `enable_numa_aware=False` (default) | Legacy single pool, no change. |
| `enable_numa_aware=True` but topology is unknown (no NVML/sysfs, e.g. container) | Plan becomes DISABLED, logged. |
| `enable_numa_aware=True` and libnuma missing | Allocation falls back to plain shm (no mbind); `numa_bound=False`, logged. |
| `enable_numa_aware=True` and TP group crosses NUMA, `allow_tp_crosses_numa=False` | Hard `ValueError` describing which DPs span which nodes. |
| `enable_numa_aware=True` and SSD/Remote/GDS/p2p also on | Hard `NotImplementedError` until Phase 2. |

## 7. Testing

### Local (no CUDA, no `c_ext`)

```bash
cd FlexKV
python3 -m unittest discover -s tests/numa -v
```

The unit tests use temp dirs for sysfs mocking, exercise the planner
across arrangement (a) shapes, validate the NumaMempool routing
invariants, and end-to-end allocate/free a NUMA-pinned shm tensor.

### Remote (real GPUs + `c_ext` built)

Integration scripts that exercise real hardware (run from `tests/numa/`,
all pass on 8× H20-3e / 2 NUMA):

| Script | What it verifies |
|---|---|
| `integration_topology_real.py` | NVML+sysfs returns the real GPU↔NUMA map; planner accepts contiguous TP groups; striped TP gets rejected with a clear message; `mbind`+first-touch lands 100% of pages on the target NUMA (verified via `/proc/self/numa_maps`). |
| `integration_storage_real.py` | `StorageEngine.allocate_cpu_pools_per_numa` builds one per-NUMA tensor of the right size; spawned subprocess sees the same physical pages (sentinel byte read-back) and the placement survives the IPC. |
| `integration_pin_real.py` | A child worker `cudaHostRegister`s the NUMA-pinned shm pool and runs real H2D/D2H. On a fabric-symmetric H20 box LOCAL vs CROSS bandwidth are within noise; on a 2-socket Xeon a measurable speedup should appear. |
| `integration_cache_engine_real.py` | Real `GlobalCacheEngine.put / get` end-to-end: D2H gets `home_numa_id=dp_to_home_pool[dp_id]`, H2D gets `home_numa_id` from `NumaMempool.pools_of(src)`, cross-DP cross-NUMA hits route to the writer's home pool. |

Suggested manual matrix when standing up a new host:

| Case | Plan | What to verify |
|---|---|---|
| `enable_numa_aware=False` | DISABLED | Bit-identical to today's behavior, same perf. |
| `enable_numa_aware=True`, TP=1, DP=N (≤ 1 GPU per NUMA) | TP_WITHIN_NUMA, `N_numa` pools | Each pool has total/N blocks; DP_i routes only to its own pool. |
| `enable_numa_aware=True`, TP=2×DP=4 contiguous on 2 NUMA | TP_WITHIN_NUMA, 2 pools | DP0/DP1 → pool 0, DP2/DP3 → pool 1; cross-DP hits go cross-NUMA (verify via `nvidia-smi dmon` PCIe RX/TX). |
| `enable_numa_aware=True`, TP=8 striped across NUMA (forced misconfig) | should raise ValueError | clean error mentioning the bad DP groups. |
| `enable_numa_aware=True` + `enable_ssd=True` | NotImplementedError | Phase 2 marker. |

Validation hints:

```bash
# Verify a pool actually landed on the right NUMA node
numastat -p $(pgrep -f flexkv) | head -20

# Page-level placement: should see all of /dev/shm/flexkv_numa_*_0 on node 0
grep "" /sys/fs/numa/node*/numa_stat   # before & after warmup
```

A useful end-to-end benchmark is the existing `benchmarks/` scripts
with `cache_config.enable_numa_aware=True`. Compare H2D/D2H bandwidth
reported by `_log_transfer_performance` between disabled and enabled
runs; on a real 2-socket box we should see ~1.4–1.8× speedup on the
home-NUMA hot path.

## 8. Open questions for the next pass

1. **Eviction skew.** With one big radix tree but per-pool free lists,
   one pool can fill up while the other has room. Phase-1 behavior is
   "evict locally; don't spill" — should we add an
   `allow_numa_spillover` option later for asymmetric workloads?
2. **SSD pool ownership.** When SSD is enabled, do we (a) per-pool
   SSD worker that pins only one pool, (b) one SSD worker that pins
   all pools and dispatches per-op, or (c) keep SSD ↔ pool-0 only?
   Each has different perf/complexity trade-offs.
3. **Arrangement (b) C++ design.** Two clean options: extend
   `TPTransferThreadGroup` to take an array of CPU pool pointers
   (one per NUMA), or keep the C++ as-is and fan out at the Python
   layer into N parallel ops. The second is less work but adds
   scheduler overhead per op.

These are intentionally left for the next session.
