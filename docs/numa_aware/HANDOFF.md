# NUMA-aware CPU pool for FlexKV — 工程交接文档

> **读者：** 接手这个 PR 的下一位同事。本文档假设你了解 FlexKV 的整体结构
> （StorageEngine / GlobalCacheEngine / TransferEngine）以及现有的 CPU pin
> 路径（一整块 `torch.empty(...)` 通过 `torch.multiprocessing` 分享给 worker
> 进程，每个 worker 调 `cudaHostRegister`），但**不**假设你看过 NUMA 设计的讨论。
>
> 同目录的 `README.md` 是面向用户的简介。本文档是工程深度交接：设计依据、
> 文件级改动清单、当前状态、踩坑点、以及具体的 TODO。

## 0. TL;DR

* **目标：** 让 CPU↔GPU offload 的传输尽可能发生在同一 NUMA 上的 CPU 和 GPU
  之间，**且不在 offload cache 里复制数据**。
* **当前 PR 的状态（称为 Phase 1.5）：** 控制面端到端已经打通（拓扑、planner、
  per-NUMA shm-backed 池、NumaMempool、按 `(dp, numa)` 笛卡尔积起 worker），
  cache engine 的 `home_numa_id` 标记也已经实现（D2H 走 writer 的 home pool；
  H2D 按源 CPU 块所属池做 fanout，包括 cross-DP cross-NUMA 命中那条路径）。
  默认关闭（`cache_config.enable_numa_aware=False` ⇒ 完全等价于旧的单池路径）。
* **真机验证过**（8× H20-3e / 2 NUMA / 2 TiB 内存）：31 个 unit test + 4 个集
  成脚本全过，物理页 100% 落在目标 NUMA 上，spawn 之后 IPC 保留 NUMA 落点，
  `cudaHostRegister` 在 shm 池上能成功 pin，cache engine 路由 op 完全正确。
* **性能注意点：** 在我们测过的这台 H20-3e 上，LOCAL 与 CROSS host memory 带宽
  差异在噪声范围内（H2D 都是 ~55 GB/s）。这并不影响设计的正确性，应该是 H20-3e
  的 fabric 本身就比较对称；在 UPI/socket 差异更明显的平台上（比如老 Xeon 双
  路）才会拿到收益。
* **没做完的：** SSD/Remote/GDS 路径的 NUMA 支持、arrangement (b)（TP 跨 NUMA）、
  per-pool LRU eviction、端到端 vLLM 性能 benchmark（容器里的 `c_ext.so` 比
  main 老，缺 `protected_threshold` 参数，阻塞了真实 `CacheEngineAccel` 跑通）。

## 1. 动机和设计约束

### 1.1 为什么 NUMA 重要

FlexKV 现在 CPU offload 池是 TransferManager 进程里一整块 `torch.empty(...)`，
通过 `torch.multiprocessing` 分享给每个 DP 的 worker 子进程，每个 worker 自己
调 `cudaHostRegister` 做 pin。这带来两件事：

1. 没有 NUMA 提示的情况下，所有物理页都落在分配线程所在的 NUMA 节点上（Linux
   first-touch 策略）。
2. `cudaHostRegister` 把页锁在当前位置；pin 之后页就没法迁移了。

在双 socket 机器上这意味着大部分 GPU 都要跨 UPI / Infinity Fabric 访问 CPU
内存，最坏情况下可能损失 30~50% 的 host memory 带宽。

### 1.2 主导设计的三条约束

下面的每一个非平凡的取舍都源自这三条约束：

| # | 约束 | 影响 |
|---|---|---|
| C1 | CPU 池**必须跨多个 worker 进程共享**（每个 DP 一个 transfer worker，将来还有 SSD/Remote worker）。 | 直接否决了 `numa_alloc_onnode` —— 它是匿名 mmap，子进程没法 re-map。必须走 file-backed 共享映射。 |
| C2 | NUMA placement 决策必须在**任何 `cudaHostRegister`（pin）之前**完成 —— pin 会锁页。而且必须在**任何非目标节点第一次写页之前**完成 —— first-touch 决定一切。 | 必须 mmap → `mbind(MPOL_BIND)` → 由绑定到目标节点的线程做 first-touch → 然后再分发给 worker。 |
| C3 | **CPU 内存是 KV offload 的瓶颈资源。** 我们不能为了避免跨 NUMA 而把 block 在多个池里复制 —— 池总容量本来就是用户愿意付的极限了。 | 每个逻辑 block 唯一住在一个池里。Cross-DP cross-NUMA 命中**会**付跨 NUMA DMA 的代价 —— 但不会双写、不会容量减半、不会命中率减半。 |

### 1.3 TP/DP 交互（最容易搞错的部分）

「每个 CPU 池里实际存放什么」的 shape 只由**TP 组怎么映射到 NUMA**决定。DP 只是
在上面加的一个乘数：

* **TP 不跨 NUMA**（vLLM 默认排列，连号 device id）：每池 `block_size` 不变，
  因为同一个 TP 组对一个 logical block 的所有 slice 都在同一个 NUMA 上。每池
  block *数量* 是 `total / N_numa`（我们按 NUMA 把全局 block id 空间切开 ——
  见 §4.2）。

* **TP 跨 NUMA**：每池 `block_size = orig × sub_tp / tp_size`，因为每个池只
  存放物理上落在该节点上的那些 TP rank 的 slice。每池 block *数量* 是 `total`
  （每个 block 在每个池里都有一份切片，互相不复制因为它们是不同的数据）。

当前 PR 只实现了 **TP-within-NUMA**。TP-crosses-NUMA 会被检测出来并报清楚的
错（`flexkv/numa/planner.py` 里的 `build_numa_plan` 抛 `ValueError`，指出
是哪些 DP 跨了 NUMA）。怎么补上去见 §7.2。

DP 不会改变每池 block size。DP 只决定「某个 DP 的 put 落到哪个池」—— 这条
路由规则就是 `dp_to_home_pool[dp_id]`。

#### 实例：TP=2 × DP=4 / 8 GPU / 2 NUMA

连号排布（vLLM 默认）：GPU 0–3 在 NUMA 0，GPU 4–7 在 NUMA 1，TP 组是
(0,1)/(2,3)/(4,5)/(6,7)。

```
NumaPlan(arrangement=TP_WITHIN_NUMA, num_pools=2)
  pool[0] -> NUMA node 0, num_cpu_blocks/2 blocks
  pool[1] -> NUMA node 1, num_cpu_blocks/2 blocks
  dp0: pool_per_rank=[0, 0], home_pool=0   # 两个 TP rank 都在 NUMA 0
  dp1: pool_per_rank=[0, 0], home_pool=0
  dp2: pool_per_rank=[1, 1], home_pool=1   # 两个 TP rank 都在 NUMA 1
  dp3: pool_per_rank=[1, 1], home_pool=1
```

* DP0 PUT → CPU slot 从 pool 0 分配 → D2H op 带 `home_numa_id=0`
  → 路由给 `(dp=0, numa=0)` worker（它在自己进程里 pin 了 pool 0，pin 时本身
  也在 NUMA 0 上）。
* DP2 GET 命中 DP0 之前 put 的 block → 这个 block 住在 pool 0
  → H2D op 带 `home_numa_id=0` → 路由给 `(dp=2, numa=0)` worker
  → DMA 从 pool 0（NUMA 0）拷到 GPU 4/5（NUMA 1）。跨 NUMA，但**没复制数据**。

总 CPU 容量 = 原 `num_cpu_blocks` × block_size，**没膨胀**。

## 2. 端到端数据流

一次 DP_x 的 PUT，在 `enable_numa_aware=True` 模式下怎么变成一个 NUMA 路由的
transfer：

```
KVManager.put_async(..., dp_id=x)
        │
        ▼
KVTaskEngine.put_async(...)  ──── 构造 GlobalCacheEngine 的 PUT graph
        │
        ▼
GlobalCacheEngine.put(..., dp_id=x)
   │
   ├─ home_pool = _home_pool_for_dp(x)  ───────  numa_plan.dp_to_home_pool[x]
   │
   ├─ cpu_block_ids = cpu_cache_engine.take(
   │       num_required_blocks=N,
   │       home_numa_pool=home_pool   ◀──── 新增：把分配约束到该池
   │   )                                    在全局 block-id 空间里的子段。
   │
   └─ TransferOp(type=D2H,
                 src=gpu_blocks, dst=cpu_block_ids,
                 home_numa_id=home_pool)   ◀──── 新增：路由标记
        │
        ▼
TransferEngine._assign_op_to_worker(op)
        │
        ├── 若 numa_plan 启用且 op 是 H2D/D2H:
        │      worker = workers_by_pair[(op.dp_id, op.home_numa_id)]
        │   否则:
        │      worker = workers[op.dp_id]   ◀──── 旧路径兜底
        │
        ▼
WorkerProcess(dp=x, numa=home_pool)
        │
        ├── 在自己进程里对**该池**的 tensor 调 cudaHostRegister
        │   （CUDA 按 context 跟踪 pin 状态，跨进程互不干扰）
        │
        └── 启动 TPTransferThreadGroup → CUDA kernel → 完成。
```

GET 的流程对称，但 H2D 那一侧多一个细节：一次命中里匹配到的 cpu block 序列可能
跨了多个 home pool（当 writer 和 reader 不在同一 NUMA 时）。
`_emit_h2d_with_pool_fanout` 把 src blocks 按池分组，每个池发一个独立的 H2D op，
全部进 `finished_ops_ids` 并共享同一组 predecessor。具体见
`flexkv/cache/cache_engine.py:_emit_h2d_with_pool_fanout`。

## 3. NUMA placement 怎么真的生效 —— 容易写错的那一段

让物理页真的落到目标节点的整条链路在 `flexkv/numa/allocator.py` 的
`numa_alloc_tensor` 里。顺序：

1. **创建文件** `/dev/shm/flexkv_numa_<hint>_<idx>_<pid>_<uuid>`，`ftruncate`
   到池的字节数。
2. **包成 torch tensor**，走 `torch.UntypedStorage.from_file(shared=True)`。这
   时 tensor 已经是 shm-backed，但物理页还没分配。
3. **`mbind(MPOL_BIND, node)`** 作用在 tensor 的 VA 范围上。之后**任何进程**对
   这段映射触发的页错误，内核都必须从 `node` 上分配页。（`mbind` 是 best-effort，
   优先走 `libc.mbind`，否则直接 `syscall(SYS_mbind, ...)` —— 有些 glibc 不导出
   `mbind` 符号。）
4. **由绑定到目标 NUMA 的线程做 first-touch**：起一个 dedicated 线程，先
   `numa_run_on_node(node)`，然后 `memset(0)` 整个 tensor。这一步把每一页都在
   worker 子进程触碰之前提交到目标节点。
5. **通过 `torch.multiprocessing` 分发给 worker**：reducer 看到 tensor 已经
   shm-file-backed，直接传文件名给子进程，子进程自己 `mmap` 同一段物理页。零拷
   贝。NUMA placement 保持。
6. **每个 worker** 调 `cudaHostRegister` pin **自己进程的本地 VA 映射**。CUDA
   按 context 跟踪 pin，多个 worker 在重叠的 VA 上独立 pin 都能成功。

**为什么不用 `numa_alloc_onnode`？** 它返回的是匿名 mmap。匿名映射没法在另一
个进程里 re-attach —— spawn 出来的 worker 跟 parent 不共享 VA。走 `/dev/shm`
是**唯一**同时满足「保留 NUMA placement」+「跨进程共享」的方案。

**降级模式**（都是 soft，只 warn，不 fail）：

* `libnuma` 加载不了 → `HAS_LIBNUMA=False`，`numa_bound=False`，页跟随
  first-touch 策略，跟以前一样。
* `mbind` 返回非零 → `numa_bound=False`，同上。
* `numa_run_on_node` 失败 → first-touch 可能跨节点；页可能落在 touch 线程实际
  跑的那个节点上。

集成测试 `tests/numa/integration_topology_real.py` 通过读 `/proc/self/numa_maps`
验证：H20-3e 上这条链路非常稳，64 MiB 分配的 16384 个 4 KiB 页 100% 落在目标
节点上。

## 4. 组件分解

### 4.1 `flexkv/numa/topology.py` — 拓扑发现

* `list_numa_nodes(sysfs_root=...)` —— 枚举 `/sys/devices/system/node/nodeN`。
* `gpu_numa_node(device_id)` —— 把 CUDA device id 解析成 NUMA node id，先问
  NVML 拿 PCI bus id，然后读 `/sys/bus/pci/devices/<addr>/numa_node`。如果
  NVML 不在，退化到 `nvidia-smi --query-gpu=pci.bus_id`。同时把 NVML 给的
  8 位 domain 标准化成 sysfs 用的 4 位 domain。
* `NumaTopology.detect(device_ids, override)` —— 返回拓扑快照：`nodes`、
  `gpu_to_node`、`is_fabricated=True`（如果解析失败，例如 CPU-only 容器）。
  planner 用 `is_fabricated` 直接 short-circuit 到 disabled plan。

### 4.2 `flexkv/numa/planner.py` — NumaPlan

* `NumaArrangement` 枚举：`DISABLED`、`TP_WITHIN_NUMA`、`TP_CROSSES_NUMA`。
* `NumaPlan` dataclass —— 下游代码路由所依据的唯一对象。包含
  `pool_nodes: List[int]`、`dp_to_pool_per_rank: Dict[dp_id, List[int]]`、
  `dp_to_home_pool: Dict[dp_id, int]`、`arrangement`、`total_num_blocks`。
* `build_numa_plan(...)` 做 arrangement 检测：对每个 DP 组，收集其 TP rank 所
  在的 NUMA 节点集合。如果每个 DP 的集合大小都是 1 → TP_WITHIN_NUMA（我们
  ship 的）。如果有 DP 的集合大小 > 1 → TP_CROSSES_NUMA（Phase 2；raise
  `NotImplementedError`）。混合（部分 DP 是单节点、部分跨节点）→ raise
  `ValueError`，因为这几乎一定是 launcher 配错了。
* `per_pool_num_blocks()` 把全局 block-id 空间均匀切到各池，多出来的余数分给
  靠前的池。比如 total=10001、N=2 → `[5001, 5000]`。每池的首 id 表通过
  `pool_block_id_ranges(plan)` lazy 算。
* `pool_index_for_block_id(plan, block_id)` 是反向查找。

### 4.3 `flexkv/numa/allocator.py` — shm + mbind 分配器

见 §3。对外的函数：`numa_alloc_tensor(num_elements, dtype, node, name_hint,
pool_index, touch=True, keepalive=True)`。返回 `NumaAllocResult`，其
`.tensor` 是可分享的 handle；`.numa_bound` 在 mbind 没生效时为 False；
`.file_path` 是底下的 `/dev/shm/...` 路径（关闭时 `unlink()` —— 我们注册了
atexit，但也可以更早释放）。

### 4.4 `flexkv/numa/mempool.py` — NumaMempool

多池 free list。对外暴露的 block id 是**全局**的（单一连续 id 空间），所以
radix tree 不需要感知池的存在。

* `allocate_blocks(num, home_numa_pool=None)` —— 不带 hint 时选剩余 block 最多
  的池（用于 prefetch / temp buffer 这种「无所谓哪个池」的场景）。带 hint 时，
  若该池满足不了请求就直接报错（默认不 spillover；TODO 见 §7.4）。
* `recycle_blocks(block_ids)` —— 向量化，先 `pools_of(ids)` 分组，再分别 recycle
  到各池的本地 free list。
* `pool_of(block_id)` / `pools_of(block_ids)` —— 反向查找。`_emit_h2d_with_pool_fanout`
  用它决定每个 H2D op 路由到哪。

注意：实现里复制了一份精简版的 `Mempool` 叫 `_PoolSlot`，是为了避免在 NUMA 包
里 import `flexkv.cache.__init__`（那个会加载 `flexkv.c_ext`）。等 cache
engine 彻底重构到 NumaMempool 后可以考虑合并。

### 4.5 `_AllocatorAdapter`（在 `flexkv/cache/cache_engine.py` 里）

包装 `Mempool` 或 `NumaMempool` 的薄壳，对外接口统一。这就是
`CacheEngineAccel.mempool` 和 `CacheEngine.mempool` 现在持有的对象。
`is_numa` 布尔位让调用点只在需要时分支。旧的 `Mempool` 会静默忽略
`home_numa_pool` 参数，所以所有旧调用点都不用改。

### 4.6 `flexkv/common/storage.py:StorageHandle`

加了两个可选字段：

```python
numa_node: Optional[int] = None
numa_pool_index: Optional[int] = None
```

由 `CPUAllocator.allocate_on_numa_node` 设置。`TransferEngine` 构建
per-`(dp, numa)` worker layout 时读这两个值。旧的 handle 保持 `None`。

### 4.7 `flexkv/common/transfer.py:TransferOp.home_numa_id`

```python
home_numa_id: int = -1   # -1 = 旧路径 / 不感知 NUMA
```

`TransferEngine._assign_op_to_worker` 读它决定 H2D/D2H 给哪个
`(dp_id, home_numa_id)` worker。GlobalCacheEngine 直接设置（D2H）或者通过
`_emit_h2d_with_pool_fanout` 设置（H2D）。`merge_to_batch_graph` 里有
assertion 守住合并的 op 必须有相同 `home_numa_id` —— 将来如果谁加了跨 NUMA 的
batch merge，这条断言会立刻报错。

### 4.8 `flexkv/storage/allocator.py:CPUAllocator.allocate_on_numa_node`

`numa_alloc_tensor` 的薄包装，返回一个带 `numa_node`/`numa_pool_index` 的
`StorageHandle`。`HAS_LIBNUMA=False` 时 raise `RuntimeError` —— 调用方要么关
NUMA-aware，要么装 `libnuma1`。

### 4.9 `flexkv/storage/storage_engine.py`

* 只要 `enable_cpu=True`，`_cpu_layout` 一定会被设置（它描述「逻辑上完整池」的
  shape；per-pool tensor 借用它的 shape 但 `num_block` 较小）。
* `allocate_cpu_pools_per_numa(plan)` 按 `plan.pool_nodes` 分配 N 个池。每个池
  在 handle 表里用 `(DeviceType.CPU, pool_index)` 当 key。旧的
  `get_storage_handle(DeviceType.CPU)` 仍然返回 pool 0 —— 给还没 NUMA 化的代码
  路径（SSD/Remote 的 sanity check）一个兜底。
* `get_cpu_pool_handles()` 按池索引顺序枚举 per-pool handle。NUMA-aware 关闭
  时返回 `[legacy_single_handle]` —— 调用方可以用 `len(...) == 1` 检测单池
  状态。

### 4.10 `flexkv/transfer/transfer_engine.py`

* 构造函数新增 `cpu_handles_per_numa: Optional[List[StorageHandle]]` 和
  `numa_plan: Optional[NumaPlan]`。plan 启用时记下来；否则走旧路径。
* `_init_workers` 按 `(dp_id, pool_index)` 笛卡尔积起 H2D + D2H worker。传给
  每个 worker 的 CPU tensor 是该 NUMA 的池 tensor，`block_stride / kv_stride
  / layer_stride` 与全局 layout 一致（只有 `num_block` 不同）。
* `_create_gpu_cpu_worker(dp, gpus, cpu_handle)` 是 per-pair 构造器；`tp_size>1`
  时用 `tpGPUCPUTransferWorker`，否则 `GPUCPUTransferWorker`。
* `_assign_op_to_worker(op)` 在 plan 启用且 op 是 H2D/D2H 时按
  `(op.dp_id, op.home_numa_id)` 路由。其它 op type（SSD/Remote/Peer/VIRTUAL）
  仍走旧的 per-dp 路由。
* Phase-1 兜底：`_init_workers` 启动时，若 NUMA-aware 开着，**且**
  `enable_ssd / enable_remote / enable_gds / enable_kv_sharing` 任何一个也
  开着，就 raise `NotImplementedError`。见 §7.1。

### 4.11 `flexkv/transfer_manager.py`

* `_build_numa_plan(grouped_device_ids)` —— 用**实际注册的** GPU device id（GPU
  注册完成之后）来构建 plan。这是 transfer 子进程视角的「真 plan」。
* `initialize_transfer_engine` —— GPU 注册完之后，调 plan 构造器，再调
  `storage_engine.allocate_cpu_pools_per_numa(plan)`，把两者都传给
  `TransferEngine`。

### 4.12 `flexkv/cache/cache_engine.py:GlobalCacheEngine`

* `_build_initial_numa_plan(cache_config, model_config)` —— 在主进程构建 plan
  （worker 还没注册的时候），用**连号 device id 假设**
  （`{dp: list(range(dp*tp, (dp+1)*tp))}`）。vLLM 默认就是这样排的。如果
  launcher 把 device id 排成非连号（少见），用户必须显式给
  `cache_config.numa_gpu_map`，保证两边都对得上。见 §6.4。
* `_build_cpu_mempool(num_cpu_blocks)` —— plan 启用时返回 `NumaMempool`，否则
  返回 `None`（让 `CacheEngineAccel` 走旧的 `Mempool`）。
* `_home_pool_for_dp(dp_id)` —— `numa_plan.dp_to_home_pool.get(dp_id, 0)` 或
  者 NUMA-aware 关闭时返回 `None`。这个 `None` 哨兵让 `take(home_numa_pool=...)`
  在旧模式下变成 no-op。
* `_emit_h2d_with_pool_fanout(graph, src_cpu, dst_gpu, layer_id,
  layer_granularity, predecessors, finished_ops_ids)` —— 关键的新 helper。
  旧模式下还是建一个 H2D op。NUMA 模式下先算 `pools_of(src_cpu)`，按池分组，
  每池建一个 H2D op，src/dst 各自切片，`home_numa_id` 设到对应池。所有新 op
  继承同一组 `predecessors`，并都加进 `finished_ops_ids`。

#### `_get_impl_local` / `_put_impl_local` 改了哪里

* `dp_id` 通过显式参数透传进来。
* `cpu_cache_engine.take(...)` 调用里加了 `home_numa_pool=self._home_pool_for_dp(dp_id)`。
* `op_d2h = TransferOp(...)` 构造时加了
  `home_numa_id=home_pool if home_pool is not None else -1`。
* `op_h2d = TransferOp(...)` 整段被替换成调 `_emit_h2d_with_pool_fanout(...)`。

`_get_impl_global` / `_put_impl_global`（只在 `enable_remote=True` 时用）**没动**
—— 它们被 TransferEngine 里那条「NUMA + SSD/Remote 不能共存」的 Phase-1 check
盖住了。

## 5. 配置选项

```python
CacheConfig(
    enable_numa_aware=True,                     # 主开关，默认 False
    numa_gpu_map={0: 0, 1: 0, ..., 7: 1},       # 覆盖 sysfs/NVML 的自动探测
    allow_tp_crosses_numa=False,                # Phase 2 的逃生口
)
```

环境变量目前还没接（CacheConfig 没有这几个字段的通用环境变量读取机制）。如果
要加，去 `flexkv/common/config.py` 改。

## 6. 文件改动清单

**新文件：**

| 路径 | LoC | 内容 |
|---|---:|---|
| `flexkv/numa/__init__.py` | 25 | 包外门面，re-export 公共 API |
| `flexkv/numa/topology.py` | 177 | sysfs + NVML 拓扑发现，GPU↔NUMA 映射，支持覆盖 |
| `flexkv/numa/planner.py` | 224 | `NumaPlan`、`NumaArrangement`、`build_numa_plan`、block id range helpers |
| `flexkv/numa/allocator.py` | 349 | shm-backed + `mbind` + first-touch 分配器；含 libc `mbind` 直接调用 / `syscall(SYS_mbind, ...)` 兜底 |
| `flexkv/numa/mempool.py` | 191 | `NumaMempool`（多池 free list，全局 id 空间）|
| `docs/numa_aware/README.md` | 280 | 用户向简介 + 设计概述 |
| `docs/numa_aware/HANDOFF.md` | （本文）| 工程深度交接 |
| `tests/numa/__init__.py` | 0 | 测试包标记 |
| `tests/numa/test_topology.py` | 68 | sysfs 解析 + fabricated topology |
| `tests/numa/test_planner.py` | 160 | TP1×DPN、TP2×DP4、arrangement-b 拒绝、range helper |
| `tests/numa/test_mempool.py` | 85 | 分配/释放、池路由、向量化反查 |
| `tests/numa/test_allocator.py` | 70 | shm + first-touch 端到端验证（不 assert NUMA；mbind 是 best-effort）|
| `tests/numa/test_h2d_fanout.py` | 175 | 纯 Python 的 `_emit_h2d_with_pool_fanout` 语义（不需要 `c_ext`）|
| `tests/numa/integration_topology_real.py` | 165 | 真机：NVML+sysfs 拓扑、TP2×DP4 plan、striped 拒绝、mbind 落点（100% 页在目标）|
| `tests/numa/integration_storage_real.py` | 217 | `StorageEngine.allocate_cpu_pools_per_numa`、spawn 子进程读同一段 shm、IPC 后 NUMA 落点保持 |
| `tests/numa/integration_pin_real.py` | 167 | worker 子进程 `cudaHostRegister` shm 池后跑真 H2D/D2H；报 LOCAL vs CROSS 带宽 |
| `tests/numa/integration_cache_engine_real.py` | 156 | 真 `GlobalCacheEngine.put/get` 端到端；断言 D2H/H2D `home_numa_id` 正确，包括 cross-DP 命中 |

**改动的文件：**

| 路径 | 内容 |
|---|---|
| `flexkv/common/config.py` | `CacheConfig.enable_numa_aware`、`numa_gpu_map`、`allow_tp_crosses_numa` |
| `flexkv/common/transfer.py` | `TransferOp.home_numa_id` 字段；`merge_to_batch_graph` 的混合池守护 |
| `flexkv/common/storage.py` | `StorageHandle.numa_node`、`StorageHandle.numa_pool_index` |
| `flexkv/storage/allocator.py` | `CPUAllocator.allocate_on_numa_node` |
| `flexkv/storage/storage_engine.py` | `_cpu_layout` 总是被设；新增 `allocate_cpu_pools_per_numa(plan)`、`get_cpu_pool_handles()`、`num_cpu_pools()` |
| `flexkv/transfer/transfer_engine.py` | 构造参数加 plan/per-pool handles；按 `(dp, numa)` 起 worker；`_create_gpu_cpu_worker` helper；`_assign_op_to_worker` NUMA 路由；Phase-1 SSD/Remote/GDS/peer-sharing 兜底 |
| `flexkv/transfer_manager.py` | GPU 注册完成后做拓扑发现 + plan 构建；通过 StorageEngine 分配 per-NUMA 池；plan 和 per-pool handle 传给 TransferEngine |
| `flexkv/cache/cache_engine.py` | `_AllocatorAdapter`；`CacheEngineAccel`/`CacheEngine` 接受 `mempool=` 注入和 `take(home_numa_pool=)`；`GlobalCacheEngine` 构建 plan + NumaMempool；`dp_id` 透传到 `_get_impl_local`/`_put_impl_local`；`_emit_h2d_with_pool_fanout`；D2H 设 `home_numa_id` |

**故意没动的：**

* `flexkv/cache/hie_cache_engine.py`（HierarchyLRCacheEngine，给 `enable_p2p_cpu`
  用的）。Phase 1 里 NUMA 跟 p2p 不兼容；`GlobalCacheEngine.__init__` 直接挡了
  这两个一起开。
* `flexkv/cache/radixtree.py`、`flexkv/cache/mempool.py` —— Mempool 保留为旧
  的单池分配器，NumaMempool 是独立的类。
* C++ 端（`csrc/`）—— Phase 1 直接复用 `TPTransferThreadGroup`。Phase 2 做
  arrangement (b) 时才需要改 C++，见 §7.2。

## 7. Open issues 与 next steps

下面不是空想，而是我会丢进下一个 sprint 的工作项。按这个顺序做比较合理。

### 7.1 SSD / Remote / GDS 路径（优先级最高）

现在 `TransferEngine._init_workers` 在 `enable_numa_aware=True` 且
`enable_ssd / enable_remote / enable_gds / enable_p2p_*` 任意一个也开着的时候
直接 raise `NotImplementedError`。SSD 是 CPU 之外最常用的存储层，所以这是下
一个最该解锁的。

选项是：

* **(a) 每个 NUMA 池一个 SSD worker。** 每个 worker 只 pin 自己的池，所以
  SSD↔CPU 转移留在 home NUMA 上。需要把 `H2DISK / DISK2H` op 也路由到 CPU 侧
  所在池的那个 worker。`_assign_op_to_worker` 里就是一个 1-of-N dispatch，
  跟 GPU 侧逻辑对称，比较直接。
* **(b) 一个 SSD worker 知道所有池。** 拓扑简单，但这个 worker 得 pin 所有
  池，损失了 CPU 侧的 NUMA locality。除非你能确认这台机器上 io_uring 吞吐
  对 NUMA 不敏感，否则别走这条路。

具体下手点：

* `flexkv/transfer/transfer_engine.py` 里 SSD worker 设置那一段（搜
  `CPUSSDDiskTransferWorker.create_worker`）。
* `flexkv/transfer/worker.py:CPUSSDDiskTransferWorker.__init__` —— 现在 `cpu_blocks`
  参数收一个 tensor。改成传该池的 tensor，跟 H2D/D2H 那一侧一样。
* Cache engine 一侧：`_put_impl_local` 已经会产出一个 `H2DISK` op 并依赖于
  对应的 D2H。H2DISK 的 src CPU block 跟 D2H 的 dst 共享 home pool（因为是
  同一次 `take` 分配的），所以在 D2H 设 `home_numa_id` 的地方同样设给 H2DISK
  就行。

Remote / GDS 形态一样。GDS 的 `DISK2D` 直接跳过 CPU，不在乎池 —— 但 worker
仍然 cudaSetDevice 到特定 GPU，所以让 worker 绑到该 GPU 的 home NUMA 上还
是有好处（如果你的 fabric 不对称的话）。

### 7.2 Arrangement (b) —— TP 跨 NUMA（次优先级）

为什么重要：用户 launch TP=8 跨双 socket 而 DP 不连号分组（或者 TP=8, DP=1
跨 2 NUMA）时，会直接撞到 planner 的 `ValueError`，整个 feature 用不了。
现在的兜底就是让用户重排，把 TP 组绑到单 NUMA 上。

需要的工作：

1. **Planner**: 让 `build_numa_plan` 接受 `tp_crosses_numa=True`。`NumaPlan`
   已经有 `TP_CROSSES_NUMA` 这个 enum 值，但 `_build_numa_plan` 只把它当占位符
   抛 `NotImplementedError`。改成发真 plan：池数还是 N，但 `per_pool_num_blocks`
   返回 `[total] * N`（每个 block 在每个池里都有，各一份切片）。
2. **NumaMempool 变体** —— 叫 `NumaMempoolShared`。类形态一样，但
   `allocate_blocks` 从一个逻辑 free list 分配，同一个 block id 在所有池里都
   有效（不按 range 切）。cache engine 也不需要知道「在哪个池」—— 同一个 id
   指向所有池。`pools_of(ids)` 返回「所有池」—— 就是 fanout 信号。
3. **`StorageEngine`**: arrangement (b) 的 per-pool layout `block_stride` 较小
   （`orig × sub_tp / tp_size`）。要么改 `KVCacheLayout` 让它把 `sub_tp` 算进
   `block_stride`，要么直接用较小的 `num_head` 算 per-pool layout。
4. **C++ kernel**: `TPTransferThreadGroup` 现在假设一个 CPU pointer 给整个
   TP 组。需要改成接受 pointer 数组 + per-rank → pool index 的映射。两条路：
   * 重构 `TPTransferThreadGroup` 接受多个 cpu_ptr + per-rank offset。最干净
     但要动 C++。
   * 在 Python 层 fanout —— 对每个 TP rank 所在的池 P，起一个
     TPTransferThreadGroup-of-1 来跑 (dp, pool) 对，并行起来。不改 C++，
     但每个 op 加 scheduler 开销，而且会破坏现在 TP 组内的 barrier 优化。
5. **`_emit_h2d_with_pool_fanout`**: arrangement (a) 直接用，但 (b) 里每次都要
   产 N 个 op（每池一个），而且 **DP 没有 home pool 这个概念了** ——
   `dp_to_home_pool` 在 (b) 模式下要么废掉，要么改成「池列表」。

别低估，C++ 那块是真活。

### 7.3 Per-pool LRU eviction

现在 radix tree 是单一全局 LRU。NUMA 模式下，`cpu_cache_engine.take(home_numa_pool=X)`
触发 eviction 用的也是全局 LRU，可能淘汰的是池 Y 的 block（不是池 X 的）。
然后 take() 返回少于请求的数量（`strict=False` 时）或 raise（`strict=True`
时），请求 bail。

均衡负载下基本不会触发。倾斜负载下（比如某个 prompt 源主要打 DP0/DP1，它们
共享池 0），观察得到。

两种方案：

* **per-pool LRU ring 进 radix tree。** 正确做法，C++ 大手术
  （`csrc/radix_tree.h`、`csrc/eviction_strategy.cpp` 那块插件）。早晚要做。
* **Spillover 开关** —— `cache_config.allow_numa_spillover`。home pool 满了
  又允许 spillover 时，从其它池分配。后续命中这些 block 就跨 NUMA，直到被淘
  汰。实现简单（`NumaMempool.allocate_blocks` 里几行，请求池满时 fallthrough
  到别的池）。

我会先 ship spillover 作为 opt-in，等线上看到 skew 证据再做 per-pool LRU。

### 7.4 Eviction skew 监控

把 per-pool free 数加到现有 metrics collector 里，让用户能看到自己的负载是不
是导致了池倾斜。`NumaMempool.num_free_in_pool(i)` 就是数据；导出成
`flexkv_cpu_pool_free_blocks{pool="0"}` 这种 gauge。帮助判断某个部署需不需要
做 7.3。

### 7.5 端到端 vLLM 性能 benchmark

我们用的容器（`flynn95/flexkv-vllm:latest`）里的 `c_ext.so` 比 main 老，缺
`protected_threshold`，导致 `CacheEngineAccel` 没法构造。集成测试里用
`FLEXKV_INDEX_ACCEL=0` 绕过去了（强制走 Python 的 `CacheEngine`），但真要跑
benchmark 不能那样 —— 太慢了。

要跑真 benchmark 需要：

1. 一个 `c_ext.so` 跟当前 main 对得上的容器（或者在容器里现 build：
   `apt install liburing-dev libxxhash-dev libhiredis-dev && ./build.sh`）。
2. 我们 land 的 numa 测试脚本（在 H20 机器上是 `/raid/fly/flexkv-numa-test/FlexKV`）。
3. 标准 FlexKV benchmark（`benchmarks/test_flexkv_cpu_reuse.sh`），把
   `CacheConfig(enable_numa_aware=True, ...)` 配上。

H20-3e 这台 fabric 在 socket 之间出奇地对称（LOCAL vs CROSS 带宽差在噪声范
围 —— 见 §8）。要 demo 收益，正确的平台可能是老 Xeon 双路。至少 bench：

* TP=1 × DP=8（每池一张 GPU 能放大带宽竞争；多数平台上能赢）。
* TP=2 × DP=4（vLLM 实际场景）。
* TP=4 × DP=2（仍在 NUMA 内，TP slice 更大）。
* `enable_numa_aware=False` 同 shape 的 baseline。

对比 H2D 带宽（从 `_log_transfer_performance` 收）的中位数 + p99、TTFT、整体
吞吐。

### 7.6 验证生产环境的 cross-NUMA 命中代价

cross-DP cross-NUMA 命中是唯一**故意**接受较低带宽的路径（DP_x 读 DP_y 的
home pool）。在真实 prompt 共享负载下，cross-DP 命中率决定了「我们当初拒绝
复制（§1.2 C3）」是不是真的划算。

建议收集的数据点：

* 命中率按「同 DP」vs「跨 DP」分类。
* 跨 DP 命中的带宽 vs 同 DP 命中的带宽。
* 我们因为不复制省下来的内存 vs 复制会有的代价。

如果复制看起来其实更划算，我们手里还有个 `allow_replication` 模式（设计讨论
里我们拒绝过的「策略 A」；要捡回来基本就是
`NumaMempool.allocate_blocks(num, home_numa_pool=None)` 的一个 variant，每池
都分配 + PUT 时 fanout）。

## 8. 初步性能测试结果

硬件：`H20-GPU-11` —— 8× NVIDIA H20-3e / 2 NUMA / 每 socket 约 1 TiB RAM；
GPU 0–3 在 NUMA 0，GPU 4–7 在 NUMA 1。

### 8.1 NUMA placement 稳如磐石

`tests/numa/integration_topology_real.py` 在每个节点上分配 64 MiB 然后读
`/proc/self/numa_maps`：

```
=== numa_alloc_tensor placement check ===
  node=0 bound=True file=/dev/shm/... placement={0: 16384} total_pages=16384
    OK: 16384/16384 pages (100.00%) on node 0
  node=1 bound=True file=/dev/shm/... placement={1: 16384} total_pages=16384
    OK: 16384/16384 pages (100.00%) on node 1
```

每个 4 KiB 页都落在目标节点上。`mbind` 生效。

### 8.2 IPC 后落点保持

`tests/numa/integration_storage_real.py` 分配 250 MiB 池，parent 进程在每个池
写一个 sentinel byte，再 `mp.Process(spawn)` 起子进程检查它看到同一段内存：

```
=== Cross-process share via spawn (the IPC contract) ===
  pool node=0 (250.0MB): child sees sentinel=17, placement={0:16}, on_target=100.0%
  pool node=1 (250.0MB): child sees sentinel=34, placement={1:16}, on_target=100.0%
```

child 读到了 parent 写的 sentinel byte（证明是同一段内存），此刻 child 地址
空间里只映射了 16 页（它只 touch 了 1 byte，其它按需 fault in）—— 但已经映射
进来的页，每一页都在对的节点上。

### 8.3 cudaHostRegister + H2D 带宽

`tests/numa/integration_pin_real.py` 让 worker 子进程 `cudaHostRegister` shm
池然后跑 256 MiB H2D/D2H，重复 5 次：

```
pool  gpu   kind       H2D GB/s    D2H GB/s
NUMA0  GPU0   LOCAL         55.20       54.92
NUMA0  GPU4   CROSS         55.13       54.92
NUMA1  GPU0   CROSS         55.17       54.91
NUMA1  GPU4   LOCAL         55.14       54.84

Local/Cross H2D ratio: 1.00x
```

两个重要观察：

1. **pinned shm 池可以当 CUDA pinned host buffer 用。** 这条集成在事前并不显然
   ——现在验证过了。~55 GB/s 接近 PCIe Gen5 x16 饱和。
2. **这台 H20-3e fabric 上 LOCAL vs CROSS 在噪声内。** H20-3e 的 PCIe 拓扑对
   跨 socket pinned host read 不敏感，看不到老 Xeon 那种 30~50% 的代价。设计
   依然正确，只是这里 demo 不出来。见 §7.5 选合适的平台 bench。

### 8.4 Cache engine 端到端路由

`tests/numa/integration_cache_engine_real.py` 跑真 `GlobalCacheEngine` PUT/GET
（`enable_numa_aware=True`）：

```
=== PUT from DP0: D2H must carry home_numa_id=0 ===
  D2H ops: 1, home_numa_ids=[0]

=== PUT from DP2: D2H must carry home_numa_id=1 ===
  D2H ops: 1, home_numa_ids=[1]

=== GET from DP2 hitting DP0's put: H2D should fan out / route to pool 0 ===
  H2D ops: 1
  H2D home_numa_ids = [0]
  cross-DP cache hit -> H2D ops correctly tagged home_numa_id=0

=== GET from DP0 hitting its own put: H2D should also route to pool 0 ===
  H2D ops home_numa_ids = [0]
```

「cross-DP 命中」那条是最关键的：DP2（NUMA 1）命中 DP0（NUMA 0）写的 block。
我们**没有复制**；H2D 被路由到 pool 0 的 worker（它在本地 NUMA 0 上 pin 了
pool 0），DMA 到 DP2 的 GPU（NUMA 1）—— 跨 NUMA。这就是我们设计的契约。

### 8.5 测试总览

* **31 个 unit test** 在本地（CPU-only torch）和容器里都过。
* **4 个集成脚本**在 H20 机器上都过（每个最后都打印 `All ... checks passed`）。
* **`enable_numa_aware=False` 时无回归** —— 默认还是旧的单池路径，bit-for-bit
  不变。用现有的 FlexKV smoke test 关 flag 跑过验证。

## 9. 改完代码怎么测

### 9.1 纯 Python（不需要 CUDA 和 c_ext）

```bash
cd FlexKV
python3 -m unittest discover -s tests/numa
```

应该看到 `Ran 31 tests`（你加了测试以后数字会变）。验证 planner / mempool /
topology / fanout helper，不依赖 CUDA。

### 9.2 H20 容器内

工作目录在 `H20-GPU-11` 的 `/raid/fly/flexkv-numa-test/FlexKV`。起一个新容器：

```bash
docker run -d --name flexkv-numa-test \
  --gpus all --ipc=host --network host \
  --shm-size=8g --cap-add SYS_NICE --cap-add IPC_LOCK \
  -v /raid/fly/flexkv-numa-test:/work --workdir /work \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --entrypoint "" \
  flynn95/flexkv-vllm:latest sleep infinity

docker exec flexkv-numa-test apt-get install -y numactl libnuma-dev
```

跑集成 suite：

```bash
docker exec flexkv-numa-test bash -c '
  cd /work/FlexKV && \
  PYTHONPATH=/work/FlexKV python3 tests/numa/integration_topology_real.py && \
  PYTHONPATH=/work/FlexKV python3 tests/numa/integration_storage_real.py && \
  PYTHONPATH=/work/FlexKV python3 tests/numa/integration_pin_real.py && \
  PYTHONPATH=/work/FlexKV FLEXKV_INDEX_ACCEL=0 python3 tests/numa/integration_cache_engine_real.py
'
```

最后一个测试要 `FLEXKV_INDEX_ACCEL=0`，因为容器里预装的 `c_ext.so` 比 main 老，
缺 `protected_threshold` 参数。等同事把 `c_ext` 重 build 之后就可以去掉这个
env var。

跑完 benchmark（特别是 dp-shm-test 那一类）记得清理：杀掉残留的 vllm 子进程；
有 `<defunct>` 残留就重启容器。（上次 bench session 被这一条坑过。）

### 9.3 给自己的改动加 unit test

如果你动了 cache engine 的 fanout 逻辑，去 `tests/numa/test_h2d_fanout.py` 加
一个 case。那个测试故意把 GlobalCacheEngine 那一面 stub 掉，不依赖 `c_ext`
—— 适合在没有 CUDA 和没 build `c_ext` 的开发机上快速迭代。stub 要跟真 helper
保持同步；漂移了的话下次跑这个测试会发现。

如果你动了 planner，去 `tests/numa/test_planner.py` 加新 shape 的 case。现有
的几个（TP=1 × DP=4、TP=2 × DP=4、不均匀 block-id 切分、striped 拒绝）是不错
的模板。

## 10. 踩过的坑（避免下一个人重学）

开发中让我意外的事，留在这里：

1. **`numa_alloc_onnode` 是匿名 mmap。** 不能分享给 spawn 的子进程。我们改成
   走 `/dev/shm` + `mbind`。第一版 allocator（commit 没留）直接用
   `numa_alloc_onnode`，跨进程测试静默失败 —— worker 进程 first-touch 时把页
   复制到了一份新 mapping。
2. **`mbind` 可能不是 libc 符号。** 有些 glibc 不导出。我们先试
   `libc.mbind`，失败就 fallback 到 `syscall(SYS_mbind_x86_64=237, ...)`。
   我们测试用的容器就没导出 `mbind` —— fallback 路径是真的会跑到的。
3. **`torch.UntypedStorage.from_file(shared=True)` 的语义。** 这是让 IPC 零拷
   贝的关键。如果你改成先 `torch.empty` 再 `share_memory_()`，reducer 在 IPC
   时会复制一份到新 shm —— NUMA placement 就丢了。
4. **CUDA 按进程跟踪 pin 状态。** 多个 worker（每个自己 `mmap` 了同一个 shm
   文件，VA 不同）对重叠 VA 调 `cudaHostRegister` 都能成功。这就是为什么
   per-`(dp, numa)` worker 能各自独立 pin 自己负责的池。
5. **`bind_to_dp_group` 已经会给每个 op 设 dp_id。** 我给 `_get_impl_local` /
   `_put_impl_local` 加了显式的 dp_id 参数 —— 在 helper 返回之后，`get/put`
   顶层的 `bind_to_dp_group` 还会再被调一次。这没问题；我们只读 dp_id 是为了
   查 home pool，跟 op 分配无关。
6. **`merge_to_batch_graph` 之前会静默搞乱路由**，如果合并的 op 有不同的
   `home_numa_id`。`flexkv/common/transfer.py:_merge_ops` 里现在有断言会立刻
   报错。当前 batch 只包含单个 task 的 op，所以不会触发 —— 但如果将来谁加了
   跨 task 合并，断言会告诉作者去改哪。
7. **`flexkv.cache.__init__` 会 import `flexkv.c_ext`。** 这意味着 `tests/numa/`
   下任何 import `flexkv.cache.*` 的测试都会在没 build `c_ext` 的开发机上挂掉。
   fanout 测试就刻意没这么干 —— 它把 helper inline 成 stub。别因为加了
   `from flexkv.cache.something import ...` 把这个特性弄丢。

## 11. 控制面状态机速查

| 状态 | 住在哪 | 谁读 | 谁写 |
|---|---|---|---|
| `NumaTopology`（GPU↔节点映射）| 在 `TransferManager.initialize_transfer_engine` 里现场构造 | planner | NVML + sysfs（`flexkv/numa/topology.py`）|
| `NumaPlan`（主进程拷贝）| `GlobalCacheEngine.numa_plan` | cache engine 的 `_home_pool_for_dp`、`_emit_h2d_with_pool_fanout` | 主进程里的 `_build_initial_numa_plan` |
| `NumaPlan`（transfer 进程拷贝）| `TransferManager.numa_plan` | TransferEngine 的 `_assign_op_to_worker`（间接）| GPU 注册完成后的 `TransferManager._build_numa_plan` |
| `NumaMempool`（free list）| `CacheEngineAccel.mempool._inner` | `take`、`recycle`、`pools_of` | cache engine 构造函数 |
| per-NUMA `StorageHandle`（CPU 池 tensor）| `StorageEngine._storage_handles[(DeviceType.CPU, pool_idx)]` | TransferEngine worker（通过 `get_cpu_pool_handles`）| transfer 子进程里的 `allocate_cpu_pools_per_numa` |
| 每个 H2D/D2H op 上的 `op.home_numa_id` | `TransferOp.home_numa_id` | TransferEngine 路由 | GlobalCacheEngine（D2H 直接设；H2D 通过 fanout helper）|
| pin 状态 | 每个 worker 进程的页表，由 worker init 时的 `cudaHostRegister` 设 | n/a | 各 worker 自己 |

如果上面有两条不同步（比如主进程 plan 和 transfer 进程 plan 的
`dp_to_home_pool` 不一致），路由就会静默错路 —— op 被发到没 pin 该池的 worker，
你会看到 runtime CUDA 错误或者数据不对。连号 device id 假设是目前唯一保证两边
同步的东西；以后改一边就要同时改另一边。

---

*交接文档完。读完有问题随时找我。这个 PR 的结构允许你一次性 ship §7 里任意
一个 TODO，不需要把其它的也一起重写。*
