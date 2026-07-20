# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import List, Tuple, Optional, Dict, Callable, Any, TYPE_CHECKING
from dataclasses import dataclass

import os
import numpy as np
import nvtx
import torch
from flexkv.c_ext import CRadixNode, CRadixTreeIndex
from flexkv.cache.hie_cache_engine import HierarchyLRCacheEngine
from flexkv.cache.redis_meta import RedisMeta, dist_available

from flexkv.cache.mempool import Mempool
from flexkv.cache.get_planner import (
    build_transfer_graph,
    get_media_list,
    plan_routes,
)
from flexkv.cache.transfer_pattern import add_virtal_op_for_mutiple_finished_ops
from flexkv.common.block import SequenceMeta
from flexkv.common.config import CacheConfig, ModelConfig, GLOBAL_CONFIG_FROM_ENV
from flexkv.common.transfer import (
    DeviceType, TransferOpGraph, TransferOp, TransferType
)
from flexkv.common.debug import flexkv_logger
from flexkv.common.type import (
    CacheLocality, MatchResult, MatchResultAccel, RadixNodeLike,
    CacheEngineLike,
)
from flexkv.integration.dynamo.collector import KVEventCollector
from flexkv.metrics import init_global_collector, get_global_collector

if TYPE_CHECKING:
    from flexkv.cache.radix_shmem_engine import CacheEngineRadixShmem

DEVICE_TYPE: List[str] = ['CPU', 'GPU', 'SSD', 'LAKE']
_VALID_EVICTION_POLICIES = {'lru', 'lfu', 'slru', 'fifo', 'mru', 'filo'}

class CacheEngineAccel:
    def __init__(self,
                 device_type: DeviceType,
                 num_total_blocks: int,
                 tokens_per_block: int,
                 evict_ratio: float,
                 hit_reward_seconds: int = 0,
                 evict_start_threshold: float = 1.0,
                 eviction_policy: str = "lru",
                 event_collector: Optional[KVEventCollector] = None,
                 metrics_collector = None,
                 protected_threshold: int = 2):
        if not isinstance(device_type, DeviceType):
            raise ValueError(f"Unknown device type: {device_type}")
        if num_total_blocks <= 0:
            raise ValueError(f"Invalid num_total_blocks: {num_total_blocks}")
        if tokens_per_block <= 0 or (tokens_per_block & (tokens_per_block - 1)) != 0:
            raise ValueError(f"Invalid tokens_per_block: {tokens_per_block}, "
                              f"tokens_per_block must be a power of 2")
        if eviction_policy not in _VALID_EVICTION_POLICIES:
            raise ValueError(f"Invalid eviction_policy: '{eviction_policy}'. "
                              f"Supported policies: {sorted(_VALID_EVICTION_POLICIES)}")
        if not isinstance(protected_threshold, int) or protected_threshold < 1:
            raise ValueError(f"Invalid protected_threshold: {protected_threshold}. "
                              f"protected_threshold must be an integer >= 1")

        self.device_type = device_type

        self.index = CRadixTreeIndex(tokens_per_block, num_total_blocks, hit_reward_seconds, eviction_policy,
                                     protected_threshold)

        self.mempool = Mempool(num_total_blocks=num_total_blocks)

        self.tokens_per_block = tokens_per_block
        self.num_total_blocks = num_total_blocks
        self.evict_ratio = evict_ratio
        self.evict_start_threshold = evict_start_threshold

        self.event_collector = event_collector
        self._metrics_collector = metrics_collector

    def reset(self) -> None:
        self.index.reset()
        self.mempool.reset()

    def match(self,
              sequence_meta: SequenceMeta,
              *,
              with_peer: bool = True,
              gpu_matched_blocks: int = 0) -> MatchResult:
        # A single in-process index is purely local; there is never a peer hit,
        # so `with_peer` is a no-op and `MatchResult.remote` is always None.
        sequence_meta.gen_hashes()
        match_result = self.index.match_prefix(torch.from_numpy(sequence_meta.block_hashes).to(torch.int64),
                                              sequence_meta.num_blocks, True)
        # physical blocks (torch.Tensor -> numpy, zero-copy on CPU)
        phys = match_result.physical_blocks.cpu().numpy()
        # optional block_node_ids
        try:
            bnis = getattr(match_result, "block_node_ids", None)
            if isinstance(bnis, torch.Tensor) and bnis.numel() > 0:
                bnids_np = bnis.cpu().numpy()
            else:
                bnids_np = None
        except Exception:
            bnids_np = None
        local = MatchResultAccel(
            num_ready_matched_blocks=match_result.num_ready_matched_blocks,
            num_matched_blocks=match_result.num_matched_blocks,
            last_ready_node=match_result.last_ready_node,
            last_node=match_result.last_node,
            last_node_matched_length=match_result.last_node_matched_length,
            physical_blocks=phys,
            block_node_ids=bnids_np,
        )
        return MatchResult(local=local)

    def insert(self,
               sequence_meta: SequenceMeta,
               physical_block_ids: np.ndarray,
               num_insert_blocks: int = -1,
               is_ready: bool = True,
               match_result: Optional[MatchResultAccel] = None
               ) -> "tuple[Optional[CRadixNode], np.ndarray]":
        """Attach `physical_block_ids` into the in-process radix index.

        Returns (node, unused_slots) for API parity with
        `CacheEngineRadixShmem.insert`. For the in-process index there is no
        cross-process race, so `unused_slots` is always empty.
        """
        sequence_meta.gen_hashes()
        if match_result is None:
            node = self.index.insert(torch.from_numpy(physical_block_ids).to(torch.int64),
                                     torch.from_numpy(sequence_meta.block_hashes).to(torch.int64),
                                     sequence_meta.num_blocks,
                                     num_insert_blocks,
                                     is_ready)
        else:
            node = self.index.insert(torch.from_numpy(physical_block_ids).to(torch.int64),
                                     torch.from_numpy(sequence_meta.block_hashes).to(torch.int64),
                                     sequence_meta.num_blocks,
                                     num_insert_blocks,
                                     is_ready,
                                     match_result.last_node,
                                     match_result.num_matched_blocks,
                                     match_result.last_node_matched_length)

        if self.event_collector is not None:
            self.event_collector.publish_stored(
                block_hashes=sequence_meta.block_hashes[:None if num_insert_blocks == -1 else num_insert_blocks],
                block_size=self.tokens_per_block,
                medium=DEVICE_TYPE[self.device_type]
            )
        return node, np.array([], dtype=np.int64)

    def lock_node(self, node: CRadixNode) -> None:
        self.index.lock(node)

    def unlock(self, node: CRadixNode) -> None:
        self.index.unlock(node)

    def set_ready(self, node: CRadixNode, ready: bool, ready_length: int) -> None:
        self.index.set_ready(node, ready, ready_length)

    def take(self,
             num_required_blocks: int,
             protected_node: Optional[CRadixNode] = None,
             strict: bool = True) -> np.ndarray:
        # Calculate current utilization
        utilization = (self.mempool.num_total_blocks - self.mempool.num_free_blocks) / self.mempool.num_total_blocks if self.mempool.num_total_blocks > 0 else 0

        # Proactive eviction: trigger when utilization exceeds threshold OR when blocks are needed
        should_evict = (utilization >= self.evict_start_threshold) or (num_required_blocks > self.mempool.num_free_blocks)

        if should_evict:
            if protected_node is not None:
                self.index.lock(protected_node)

            # Calculate how many blocks to evict
            # Goal: maintain free blocks above (1 - evict_start_threshold) ratio
            target_free_blocks = int(self.mempool.num_total_blocks * (1.0 - self.evict_start_threshold))
            evict_to_reach_target = max(0, target_free_blocks - self.mempool.num_free_blocks)

            evict_block_num = max(
                num_required_blocks - self.mempool.num_free_blocks,  # At least meet current demand
                evict_to_reach_target,                               # Or reach target free ratio
                int(self.mempool.num_total_blocks * self.evict_ratio) if self.evict_ratio > 0 else 0  # Or minimum evict_ratio
            )

            if evict_block_num > 0:
                target_blocks = torch.zeros(evict_block_num, dtype=torch.int64)
                evicted_block_hashes = torch.zeros(evict_block_num, dtype=torch.int64)
                num_evicted = self.index.evict(target_blocks, evicted_block_hashes, evict_block_num)
                if num_evicted != evict_block_num:
                    target_blocks.resize_(num_evicted)
                    evicted_block_hashes.resize_(num_evicted)
                target_blocks = target_blocks.numpy()
                self.mempool.recycle_blocks(target_blocks)

                # Record eviction metrics
                if self._metrics_collector is not None and num_evicted > 0:
                    self._metrics_collector.record_eviction(DEVICE_TYPE[self.device_type].lower(), num_evicted)

                if self.event_collector is not None:
                    self.event_collector.publish_removed(
                        block_hashes=evicted_block_hashes.numpy(),
                        medium=DEVICE_TYPE[self.device_type]
                    )
            if protected_node is not None:
                self.index.unlock(protected_node)

        if strict and num_required_blocks > self.mempool.num_free_blocks:
            raise RuntimeError(f"Not enough free blocks to take, "
                               f"required: {num_required_blocks}, "
                               f"available: {self.mempool.num_free_blocks}")
        num_allocated_blocks = min(num_required_blocks, self.mempool.num_free_blocks)
        allocated_blocks = self.mempool.allocate_blocks(num_allocated_blocks)

        # Record allocation metrics
        if self._metrics_collector is not None and num_allocated_blocks > 0:
            self._metrics_collector.record_allocation(DEVICE_TYPE[self.device_type].lower(), num_allocated_blocks)

        return allocated_blocks

    def recycle(self, physical_blocks: np.ndarray) -> None:
        self.mempool.recycle_blocks(physical_blocks)

@dataclass
class CacheStrategy:
    # if True, will not put or get blocks from GPU
    ignore_gpu: bool = False
    # if True, will not put or get blocks from SSD
    ignore_ssd: bool = False
    # if True, will not get blocks from LAKE
    ignore_lake: bool = False
    # if True, will not use GDS
    ignore_gds: bool = False

DEFAULT_CACHE_STRATEGY = CacheStrategy()

CPUONLY_CACHE_STRATEGY = CacheStrategy(ignore_gpu=False, ignore_ssd=True, ignore_lake=True, ignore_gds=True)

class GlobalCacheEngine:
    def __init__(self, cache_config: CacheConfig, model_config: ModelConfig, redis_meta: Optional[RedisMeta] = None,
                 event_collector: Optional[KVEventCollector] = None):
        cache_config.validate_lake_p2p_exclusive()
        self.cache_config = cache_config
        self.model_config = model_config
        self.tokens_per_block = cache_config.tokens_per_block

        self.cpu_cache_engine: Optional[CacheEngineLike[Any]] = None
        self.ssd_cache_engine: Optional[CacheEngineLike[Any]] = None
        self.lake_cache_engine: Optional[CacheEngineLike[Any]] = None

        # When True, replace the per-device CacheEngineAccel with the radixshmem-
        # backed engine so multiple DP processes share a single index in shm.
        self.use_radix_shmem = bool(getattr(GLOBAL_CONFIG_FROM_ENV, "radix_shmem", False))
        if self.use_radix_shmem and cache_config.enable_lake:
            raise ValueError(
                "radixshmem and Lake cannot be enabled at the same time"
            )
        self._shm_radix_server_id = getattr(
            GLOBAL_CONFIG_FROM_ENV, "shm_radix_server_id", "default"
        )
        if cache_config.enable_kv_sharing:
            assert redis_meta is not None
            self.redis_meta = redis_meta
            self.node_id = self.redis_meta.get_node_id()
            self.enable_kv_sharing = True
        else:
            self.enable_kv_sharing = False
        # Only concrete (non-None) engines are ever inserted, one per enabled tier.
        self.cache_engines: Dict[DeviceType, CacheEngineLike[Any]] = {}

        self.evict_ratio = GLOBAL_CONFIG_FROM_ENV.evict_ratio
        self.evict_start_threshold = GLOBAL_CONFIG_FROM_ENV.evict_start_threshold
        self.hit_reward_seconds = GLOBAL_CONFIG_FROM_ENV.hit_reward_seconds
        self.eviction_policy = GLOBAL_CONFIG_FROM_ENV.eviction_policy
        self.protected_threshold = GLOBAL_CONFIG_FROM_ENV.slru_protected_threshold

        # Initialize metrics collector for cache engine monitoring (before creating CacheEngines)
        self._metrics_collector = get_global_collector()
        if self._metrics_collector is None:
            self._metrics_collector = init_global_collector()

        need_dist = (
            (cache_config.enable_cpu and cache_config.enable_p2p_cpu)
            or (cache_config.enable_ssd and cache_config.enable_p2p_ssd)
            or (cache_config.enable_lake and cache_config.enable_kv_sharing)
        )
        if need_dist and not dist_available():
            raise RuntimeError(
                "Config enables distributed KV cache (P2P/Redis), but FlexKV was built without it. "
                "Rebuild with FLEXKV_ENABLE_P2P=1 and install Redis dependencies "
                "(e.g. libhiredis-dev, redis-tools). See README for full list."
            )

        if cache_config.enable_cpu:
            if self.use_radix_shmem:
                self.cpu_cache_engine = self._build_radix_shmem_engine(
                    DeviceType.CPU, cache_config.num_cpu_blocks, event_collector
                )
            elif cache_config.enable_p2p_cpu:
                self.cpu_cache_engine = HierarchyLRCacheEngine.from_cache_config(cache_config, self.node_id, DeviceType.CPU, meta=self.redis_meta)
            else:
                self.cpu_cache_engine = CacheEngineAccel(
                    device_type=DeviceType.CPU,
                    num_total_blocks=cache_config.num_cpu_blocks,
                    tokens_per_block=cache_config.tokens_per_block,
                    evict_ratio=self.evict_ratio,
                    hit_reward_seconds=self.hit_reward_seconds,
                    evict_start_threshold=self.evict_start_threshold,
                    eviction_policy=self.eviction_policy,
                    event_collector=event_collector,
                    metrics_collector=self._metrics_collector,
                    protected_threshold=self.protected_threshold,
                )
            self.cache_engines[DeviceType.CPU] = self.cpu_cache_engine
        if cache_config.enable_ssd:
            if self.use_radix_shmem:
                self.ssd_cache_engine = self._build_radix_shmem_engine(
                    DeviceType.SSD, cache_config.num_ssd_blocks, event_collector
                )
            elif cache_config.enable_p2p_ssd:
                self.ssd_cache_engine = HierarchyLRCacheEngine.from_cache_config(cache_config, self.node_id, DeviceType.SSD, meta=self.redis_meta)
            else:
                self.ssd_cache_engine = CacheEngineAccel(
                    device_type=DeviceType.SSD,
                    num_total_blocks=cache_config.num_ssd_blocks,
                    tokens_per_block=cache_config.tokens_per_block,
                    evict_ratio=self.evict_ratio,
                    hit_reward_seconds=self.hit_reward_seconds,
                    evict_start_threshold=self.evict_start_threshold,
                    eviction_policy=self.eviction_policy,
                    event_collector=event_collector,
                    metrics_collector=self._metrics_collector,
                    protected_threshold=self.protected_threshold,
                )
            self.cache_engines[DeviceType.SSD] = self.ssd_cache_engine
        if cache_config.enable_lake:
            assert cache_config.num_lake_blocks is not None, \
                "num_lake_blocks must be set when enable_lake is True"
            if self.use_radix_shmem:
                self.lake_cache_engine = self._build_radix_shmem_engine(
                    DeviceType.LAKE, cache_config.num_lake_blocks, None
                )
            elif cache_config.enable_kv_sharing:
                # Build PCFSCacheEngine from CacheConfig directly (replacing LakePCFSCacheEngine) TODO
                self.lake_cache_engine = HierarchyLRCacheEngine.from_cache_config(cache_config, self.node_id, DeviceType.LAKE, meta=self.redis_meta)
            else:
                self.lake_cache_engine = CacheEngineAccel(
                    device_type=DeviceType.LAKE,
                    num_total_blocks=cache_config.num_lake_blocks,
                    tokens_per_block=cache_config.tokens_per_block,
                    evict_ratio=self.evict_ratio,
                    hit_reward_seconds=self.hit_reward_seconds,
                    evict_start_threshold=self.evict_start_threshold,
                    eviction_policy=self.eviction_policy,
                    event_collector=None,
                    metrics_collector=self._metrics_collector,
                    protected_threshold=self.protected_threshold,
                )
            self.cache_engines[DeviceType.LAKE] = self.lake_cache_engine

        #TODO move this to kvmanager.start()
        self.start()

        self._empty_get_return: Callable[[int], Tuple[TransferOpGraph, List[int], Dict, Dict, Dict, int]] = \
            lambda request_id: (TransferOpGraph.create_empty_graph(), [], {}, {}, {}, 0)
        self._empty_put_return: Callable[[int], Tuple[TransferOpGraph, List[int], Dict, Dict, Dict, int, int]] = \
            lambda request_id: (TransferOpGraph.create_empty_graph(), [], {}, {}, {}, 0, 0)

    def _release_match_pre_locks(self,
                                  cpu_result=None,
                                  ssd_result=None,
                                  lake_result=None) -> None:
        """Release the atomic match-time inc_refs (radixshmem `lock=True`).

        A tier match is a `MatchResult`; either side (local or remote) may hold
        the single query guard (``pre_locked_node``).  Backends that don't
        pre-lock leave every ``pre_locked_node=None``, so this is a no-op for
        them.  Called on every early-return path, so a match-acquired ref is
        never leaked.  (The success path releases inline in :meth:`_handoff_locks`.)

        Idempotent: clears ``pre_locked_node`` after release so a double-call
        cannot dec_ref twice.
        """
        pairs = (
            (self.cpu_cache_engine, cpu_result),
            (self.ssd_cache_engine, ssd_result),
            (self.lake_cache_engine, lake_result),
        )
        for engine, result in pairs:
            if engine is None or result is None:
                continue
            # The local/remote sides of this match that still hold a pre-lock.
            for side in (result.local, result.remote):
                if side is not None and side.pre_locked_node is not None:
                    engine.unlock(side.pre_locked_node)
                    side.pre_locked_node = None

        # Update mempool stats after releasing any match-only protection.
        self._update_mempool_metrics()

    def _handoff_locks(self,
                       node_to_unlock: Dict[DeviceType, List[RadixNodeLike]],
                       cpu_result=None,
                       ssd_result=None,
                       lake_result=None) -> None:
        """Take over match-time protection with the graph completion callback.

        Per tier, in one pass: acquire independent protection via ``lock_node``
        (an inc_ref for a plain radix node; a no-op for a radixshmem query-guard
        / inserted node, which already self-protects) for every completion node
        that is NOT the tier's own match guard; then, for each match guard,
        either ADOPT it (the guard is itself a completion node — leave its
        pre-lock ref armed so the callback's ``unlock`` runs the finalize exactly
        once) or RELEASE it now (the guard is unused — drop the match ref so it
        is never leaked).

        Covers every tier carrying a match result, not just those with
        completion nodes: a tier that matched but contributed no completion node
        has all its guards released here.
        """
        tiers = (
            (DeviceType.CPU, self.cpu_cache_engine, cpu_result),
            (DeviceType.SSD, self.ssd_cache_engine, ssd_result),
            (DeviceType.LAKE, self.lake_cache_engine, lake_result),
        )
        for _device_type, engine, result in tiers:
            if engine is None:
                continue
            nodes = node_to_unlock.get(_device_type, [])
            node_ids = {id(node) for node in nodes}
            # The local/remote sides of this match that still hold a pre-lock.
            prelocked_sides = [
                side for side in (result.local, result.remote)
                if side is not None and side.pre_locked_node is not None
            ] if result is not None else []
            guard_ids = {id(side.pre_locked_node) for side in prelocked_sides}
            for node in nodes:
                if id(node) in guard_ids:
                    continue  # the match guard already protects these slots
                engine.lock_node(node)
            for side in prelocked_sides:
                if id(side.pre_locked_node) in node_ids:
                    # Adopted: keep the guard armed for the callback's unlock.
                    side.pre_locked_node = None
                else:
                    # Unused match guard: release its match-time ref now.
                    engine.unlock(side.pre_locked_node)
                    side.pre_locked_node = None

        # Update mempool stats after handing off / releasing match protection.
        self._update_mempool_metrics()

    def _build_radix_shmem_engine(self,
                                   device_type: DeviceType,
                                   num_blocks: int,
                                   event_collector) -> "CacheEngineRadixShmem":
        """Attach to a pre-created radixshmem region as a RadixClient.

        The shm region itself (RadixServer) is owned by the KVManager bootstrap
        process via `flexkv.server.shm_radix_bootstrap.create_shm_radix_regions`.
        Non-bootstrap procs poll for region availability before reaching this
        point, so the attach is unconditional here.
        """
        from flexkv.cache.radix_shmem_engine import CacheEngineRadixShmem
        from flexkv.server.shm_radix_bootstrap import shm_name_for

        return CacheEngineRadixShmem(
            device_type=device_type,
            num_total_blocks=num_blocks,
            tokens_per_block=self.cache_config.tokens_per_block,
            shm_name=shm_name_for(
                device_type,
                self._shm_radix_server_id,
                rank=getattr(GLOBAL_CONFIG_FROM_ENV, "radix_rank", 0),
                world_size=getattr(
                    GLOBAL_CONFIG_FROM_ENV, "radix_world_size", 1
                ),
                cluster_id=getattr(
                    GLOBAL_CONFIG_FROM_ENV,
                    "radix_cluster_id",
                    self._shm_radix_server_id,
                ),
            ),
            evict_ratio=self.evict_ratio,
            evict_start_threshold=self.evict_start_threshold,
            hit_reward_seconds=self.hit_reward_seconds,
            eviction_policy=self.eviction_policy,
            event_collector=event_collector,
            metrics_collector=self._metrics_collector,
            protected_threshold=self.protected_threshold,
            peer_enabled=(
                self.cache_config.enable_p2p_cpu
                if device_type == DeviceType.CPU else
                self.cache_config.enable_p2p_ssd
                if device_type == DeviceType.SSD else False
            ),
            redis_meta=getattr(self, "redis_meta", None),
            radix_cluster_id=getattr(
                GLOBAL_CONFIG_FROM_ENV,
                "radix_cluster_id",
                self._shm_radix_server_id,
            ),
        )

    def start(self) -> None:
        if self.cpu_cache_engine and self.cache_config.enable_p2p_cpu:
            self.cpu_cache_engine.start()
        if self.ssd_cache_engine and self.cache_config.enable_p2p_ssd:
            self.ssd_cache_engine.start()
        if self.lake_cache_engine and self.cache_config.enable_3rd_lake:
            self.lake_cache_engine.start()

    def reset(self) -> None:
        if self.cpu_cache_engine:
            self.cpu_cache_engine.reset()
        if self.ssd_cache_engine:
            self.ssd_cache_engine.reset()
        if self.lake_cache_engine:
            self.lake_cache_engine.reset()

    def _update_mempool_metrics(self) -> None:
        """Update memory pool metrics for all cache engines."""
        if self._metrics_collector is None:
            return
        for device_type, engine in self.cache_engines.items():
            if hasattr(engine, 'mempool'):
                # `mempool` is engine-specific (Mempool vs shmem's _MempoolView),
                # not part of CacheEngineLike; getattr keeps it cast-free (and it
                # is guarded by hasattr above).
                mempool = getattr(engine, 'mempool')
                device_label = DEVICE_TYPE[device_type].lower()
                self._metrics_collector.update_mempool_stats(
                    device_label,
                    mempool.num_total_blocks,
                    mempool.num_free_blocks
                )

    def get(self,
            request_id: int,
            token_ids: np.ndarray,
            token_mask: np.ndarray,
            slot_mapping: np.ndarray,
            layer_num: int = -1,
            layer_granularity: int = -1,
            dp_id: int = 0,
            temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY,
            namespace: Optional[List[str]] = None) \
                 -> Tuple[TransferOpGraph, np.ndarray, Callable, Dict, int]:
        self._check_input(token_ids, token_mask, slot_mapping)

        if layer_num == -1:
            layer_num = self.model_config.num_layers
        if layer_granularity == -1:
            layer_granularity = layer_num

        if layer_num != layer_granularity:
            flexkv_logger.error(f"Layerwise transfer is not supported yet, "
                                f"layer_num: {layer_num}, layer_granularity: {layer_granularity}")
            raise NotImplementedError(f"Layerwise transfer is not supported yet, "
                                      f"layer_num: {layer_num}, layer_granularity: {layer_granularity}")

        combine_with_trtllm = os.getenv("FLEXKV_WITH_TRTLLM", "0") == "1"
        if not combine_with_trtllm:
            aligned_length = (token_ids.shape[0] // self.tokens_per_block) * self.tokens_per_block
        else:
            # When using FlexKV with TensorRT-LLM, we ignore the last incomplete block.
            aligned_length = ((token_ids.shape[0] - 1) // self.tokens_per_block) * self.tokens_per_block

        aligned_token_ids = token_ids[:aligned_length]
        token_mask[aligned_length:] = False

        block_start_idx, block_end_idx = self._get_block_range(token_mask)
        assert block_end_idx == aligned_length // self.tokens_per_block
        gpu_block_ids = self.slot_mapping_to_block_ids(slot_mapping,
                                                       self.tokens_per_block)[:block_end_idx-block_start_idx]

        sequence_meta = SequenceMeta(token_ids=aligned_token_ids,
                                     tokens_per_block=self.cache_config.tokens_per_block,
                                     namespace=namespace)

        # LAKE and P2P are mutually exclusive (validated at config), so each
        # impl owns exactly one extended plan: peer reuse / GDS direct routes
        # without lake, or lake staging without any peer.  Mirrors PUT's split.
        if not self.cache_config.enable_lake or temp_cache_strategy.ignore_lake:
            (transfer_graph, finished_ops_ids, node_to_unlock,
             op_node_to_ready, buffer_to_free, num_gpu_blocks_to_transfer) = \
                self._get_impl_without_lake(
                    request_id,
                    sequence_meta,
                    block_start_idx,
                    block_end_idx,
                    gpu_block_ids,
                    layer_num,
                    temp_cache_strategy
                )
        else:
            (transfer_graph, finished_ops_ids, node_to_unlock,
             op_node_to_ready, buffer_to_free, num_gpu_blocks_to_transfer) = \
                self._get_impl_with_lake(
                    request_id,
                    sequence_meta,
                    block_start_idx,
                    block_end_idx,
                    gpu_block_ids,
                    layer_num,
                    temp_cache_strategy
                )

        transfer_graph, task_end_op_id = add_virtal_op_for_mutiple_finished_ops(
            transfer_graph,
            finished_ops_ids
            )

        return_mask = np.zeros_like(token_mask, dtype=np.bool_)
        return_mask[block_start_idx* self.tokens_per_block:
                    (block_start_idx + num_gpu_blocks_to_transfer) * self.tokens_per_block] = True

        transfer_graph.bind_to_dp_group(dp_id)

        # lock_node + match pre-lock release happens inside the _impl methods
        # to avoid a per-match ref leak window between impl return and lock.

        callback = partial(self._transfer_callback,
                           node_to_unlock=node_to_unlock,
                           buffer_to_free=buffer_to_free)

        op_callback_dict = {} # dict, op_id -> callback
        for op_id, (device_type, nodes) in op_node_to_ready.items():
            op_callback_dict[op_id] = partial(self._op_callback,
                                              device_type=device_type,
                                              nodes=nodes)

        # Update mempool metrics after GET operation
        if self._metrics_collector is not None:
            self._update_mempool_metrics()

        return transfer_graph, return_mask, callback, op_callback_dict, task_end_op_id

    @staticmethod
    def _completion_nodes(match: MatchResult, localities) -> List[RadixNodeLike]:
        """The matched ready node(s) protecting the localities a tier serves.

        Returns the ``last_ready_node`` of exactly the sides the plan uses — the
        local prefix and/or the peer suffix.  A radixshmem match collapses to the
        single query guard (only one side carries it); a hie peer match yields
        the two concrete nodes in its two trees.  No group wrapper: the caller
        holds a flat list and locks / releases each node one at a time.
        """
        nodes: List[RadixNodeLike] = []
        if (CacheLocality.LOCAL in localities and
                match.local.last_ready_node is not None):
            nodes.append(match.local.last_ready_node)
        if (CacheLocality.PEER in localities and match.remote is not None and
                match.remote.last_ready_node is not None):
            nodes.append(match.remote.last_ready_node)
        return nodes

    @staticmethod
    def _promote_apply(engine,
                       sequence_meta: SequenceMeta,
                       dest_blocks: np.ndarray,
                       *,
                       num_insert_blocks: int,
                       match_result: Optional[MatchResultAccel],
                       completion_nodes: List[RadixNodeLike],
                       replace: bool,
                       blocks_to_free: List[np.ndarray],
                       op_node_to_ready: Dict,
                       device_type: DeviceType,
                       ready_op_id: Optional[int]) -> None:
        """The mechanical tail shared by every promotion case.

        Insert ``dest_blocks`` as an unready suffix extending ``match_result``'s
        ready prefix, fold the inserted node into the tier's flat completion
        list, wire the optional early-ready op, and recycle any slots the index
        did not attach.  Mutates ``completion_nodes`` / ``blocks_to_free`` /
        ``op_node_to_ready`` in place (no rebind, so aliases stay valid).

        ``replace=True`` drops the matched node — the inserted descendant pins
        the same prefix, so the matched guard is released at hand-off.
        ``replace=False`` keeps the matched node *beside* the inserted one: its
        guard still protects a separate, still-needed source (the peer slots).
        """
        inserted, unused = engine.insert(
            sequence_meta, dest_blocks,
            num_insert_blocks=num_insert_blocks,
            is_ready=False, match_result=match_result,
        )
        if inserted is not None:
            if replace:
                completion_nodes[:] = [inserted]
            else:
                completion_nodes.append(inserted)
            if ready_op_id is not None:
                op_node_to_ready[ready_op_id] = (device_type, [inserted])
        # Runs even when inserted is None: a distributed shmem insert can attach
        # nothing yet still hand back slots the caller must recycle.
        if len(unused) > 0:
            blocks_to_free.append(unused)

    def _get_impl_with_lake(self,
                            request_id: int,
                            sequence_meta: SequenceMeta,
                            block_mask_start: int,
                            block_mask_end: int,
                            gpu_block_ids: np.ndarray,
                            layer_num: int,
                            temp_cache_strategy: CacheStrategy) \
                               -> Tuple[TransferOpGraph, List[int], Dict, Dict, Dict, int]:
        """Plan and build a GET graph across cpu / ssd / lake.

        Phase 1 (:func:`get_media_list`) decides *which* media serve the queried
        prefix; the inlined phase 2 decides *how* each reaches the GPU and applies
        promotion.  The lake path stages every tier to host first (GDS / peer->GPU
        direct routes are unsupported alongside it), and LAKE and P2P are mutually
        exclusive (validated at config), so no peer segment ever appears here.
        """
        nvtx_range = nvtx.start_range(
            message=f"CacheEngine._get_impl_with_lake[{request_id}]", color="cyan")
        enable_gpu = not temp_cache_strategy.ignore_gpu
        enable_ssd = self.cache_config.enable_ssd and not temp_cache_strategy.ignore_ssd
        assert self.cache_config.enable_cpu
        assert self.cpu_cache_engine is not None
        assert self.lake_cache_engine is not None

        cpu_match, ssd_match, lake_match = self.match_all(
            sequence_meta,
            temp_cache_strategy=temp_cache_strategy,
            is_put=False,
            gpu_matched_blocks=block_mask_start,
        )
        segments = get_media_list(
            cpu_match, ssd_match, lake_match,
            block_mask_start=block_mask_start,
            block_mask_end=block_mask_end,
        )
        total_query_blocks = block_mask_end - block_mask_start
        if not segments:
            if self._metrics_collector is not None and total_query_blocks > 0:
                self._metrics_collector.record_cache_miss(total_query_blocks)
            nvtx.end_range(nvtx_range)
            self._release_match_pre_locks(
                cpu_result=cpu_match,
                ssd_result=ssd_match,
                lake_result=lake_match)
            return self._empty_get_return(request_id)

        num_gpu_blocks = segments[-1].logical_end - block_mask_start
        assert num_gpu_blocks <= len(gpu_block_ids)

        # Lake stages every tier to host first, so GDS and direct peer->GPU
        # routes are both off.
        num_staging = plan_routes(
            segments,
            enable_gpu=enable_gpu,
            enable_gds=False,
            enable_peer_gpu=False,
        )

        nvtx.push_range(f"take {num_staging} cpu blocks", color="green")
        cpu_staging_blocks = self.cpu_cache_engine.take(
            num_required_blocks=num_staging,
            protected_node=cpu_match.local.last_node,
            strict=False,
        )
        nvtx.pop_range()
        if len(cpu_staging_blocks) < num_staging:
            self.cpu_cache_engine.recycle(cpu_staging_blocks)
            if self._metrics_collector is not None:
                self._metrics_collector.record_allocation_failure("local")
            nvtx.end_range(nvtx_range)
            self._release_match_pre_locks(
                cpu_result=cpu_match,
                ssd_result=ssd_match,
                lake_result=lake_match)
            return self._empty_get_return(request_id)

        transfer_graph, finished_ops_ids, _h2d_ops = build_transfer_graph(
            segments,
            staging_blocks=cpu_staging_blocks,
            block_mask_start=block_mask_start,
            layer_num=layer_num,
        )
        op_node_to_ready: Dict[int, Tuple[DeviceType, List[RadixNodeLike]]] = {}

        used: Dict[DeviceType, set] = {}
        for seg in segments:
            used.setdefault(seg.tier, set()).add(seg.locality)
        cpu_nodes = GlobalCacheEngine._completion_nodes(
            cpu_match, used.get(DeviceType.CPU, set()))
        ssd_nodes = GlobalCacheEngine._completion_nodes(
            ssd_match, used.get(DeviceType.SSD, set()))
        lake_nodes = GlobalCacheEngine._completion_nodes(
            lake_match, used.get(DeviceType.LAKE, set()))
        cpu_blocks_to_free: List[np.ndarray] = []
        ssd_blocks_to_free: List[np.ndarray] = []
        promoted_ids = set()  # id(seg) whose staging was inserted into an index

        local_cpu = cpu_match.local
        local_cpu_ready = local_cpu.num_ready_matched_blocks

        # (B) Promote a staged lower-tier (SSD / LAKE) suffix into the local CPU
        # index when CPU is a fully-ready local prefix.  The inserted descendant
        # pins that prefix, so it REPLACES the matched node.
        lower_staged = [
            seg for seg in segments
            if seg.tier in (DeviceType.SSD, DeviceType.LAKE) and seg.needs_staging
        ]
        if lower_staged:
            first = lower_staged[0]
            has_lake = any(s.tier == DeviceType.LAKE for s in lower_staged)
            triggers = {
                s.h2d_op.op_id: s.h2d_op
                for s in lower_staged if s.h2d_op is not None
            }
            if not triggers:
                triggers = {
                    s.primary_op.op_id: s.primary_op
                    for s in lower_staged if s.primary_op is not None
                }
            base_ok = (
                local_cpu_ready == local_cpu.num_matched_blocks and
                block_mask_start <= local_cpu_ready and
                first.logical_start == local_cpu_ready
            )
            # A pure-SSD promotion needs one feeding op so it can flip ready
            # early; a lake promotion spans multiple source tiers, so it only
            # flips ready at graph completion.
            if base_ok and (has_lake or len(triggers) == 1):
                ready_op_id = (
                    next(iter(triggers))
                    if not has_lake and len(triggers) == 1 else None
                )
                GlobalCacheEngine._promote_apply(
                    self.cpu_cache_engine, sequence_meta,
                    np.concatenate([s.staging for s in lower_staged]),
                    num_insert_blocks=block_mask_start + num_gpu_blocks,
                    match_result=local_cpu,
                    completion_nodes=cpu_nodes, replace=True,
                    blocks_to_free=cpu_blocks_to_free,
                    op_node_to_ready=op_node_to_ready,
                    device_type=DeviceType.CPU, ready_op_id=ready_op_id,
                )
                promoted_ids.update(id(s) for s in lower_staged)

        # (C) Promote a LAKE suffix into the local SSD index (extra H2DISK) when
        # it cleanly extends a fully-ready local SSD prefix.
        lake_segs = [seg for seg in segments if seg.tier == DeviceType.LAKE]
        if enable_ssd and lake_segs and self.ssd_cache_engine is not None:
            local_ssd = ssd_match.local
            first_lake = lake_segs[0]
            lake_block_count = sum(s.num_blocks for s in lake_segs)
            can_promote_ssd = (
                local_ssd.num_ready_matched_blocks == local_ssd.num_matched_blocks and
                block_mask_start <= local_ssd.num_ready_matched_blocks and
                first_lake.logical_start == local_ssd.num_ready_matched_blocks
            )
            if can_promote_ssd:
                lake_ssd_blocks = self.ssd_cache_engine.take(
                    num_required_blocks=lake_block_count,
                    protected_node=local_ssd.last_node,
                    strict=False,
                )
                if len(lake_ssd_blocks) == lake_block_count:
                    op_h2disk = TransferOp(
                        graph_id=transfer_graph.graph_id,
                        transfer_type=TransferType.H2DISK,
                        src_block_ids=np.concatenate(
                            [s.staging for s in lake_segs]),
                        dst_block_ids=lake_ssd_blocks,
                        layer_id=0,
                        layer_granularity=layer_num,
                    )
                    transfer_graph.add_transfer_op(op_h2disk)
                    for s in lake_segs:
                        assert s.primary_op is not None
                        transfer_graph.add_dependency(
                            op_h2disk.op_id, s.primary_op.op_id)
                    # H2D and H2DISK share the same lake staging blocks and may
                    # run in parallel; keep the graph alive until both consume
                    # them so the callback cannot recycle staging early.
                    finished_ops_ids.append(op_h2disk.op_id)
                    GlobalCacheEngine._promote_apply(
                        self.ssd_cache_engine, sequence_meta, lake_ssd_blocks,
                        num_insert_blocks=first_lake.logical_start + lake_block_count,
                        match_result=local_ssd,
                        completion_nodes=ssd_nodes, replace=True,
                        blocks_to_free=ssd_blocks_to_free,
                        op_node_to_ready=op_node_to_ready,
                        device_type=DeviceType.SSD, ready_op_id=None,
                    )
                else:
                    self.ssd_cache_engine.recycle(lake_ssd_blocks)

        # Recycle staging that no promotion adopted into an index.
        for seg in segments:
            if seg.staging is not None and id(seg) not in promoted_ids:
                cpu_blocks_to_free.append(seg.staging)

        node_to_unlock: Dict[DeviceType, List[RadixNodeLike]] = {}
        if cpu_nodes:
            node_to_unlock[DeviceType.CPU] = cpu_nodes
        if ssd_nodes:
            node_to_unlock[DeviceType.SSD] = ssd_nodes
        if lake_nodes:
            node_to_unlock[DeviceType.LAKE] = lake_nodes

        buffer_to_free: Dict[DeviceType, np.ndarray] = {}
        if cpu_blocks_to_free:
            buffer_to_free[DeviceType.CPU] = np.concatenate(cpu_blocks_to_free)
        if ssd_blocks_to_free:
            buffer_to_free[DeviceType.SSD] = np.concatenate(ssd_blocks_to_free)

        # This is for sync get
        transfer_graph.set_gpu_blocks(gpu_block_ids)

        if self._metrics_collector is not None:
            cpu_ready = min(block_mask_end, cpu_match.peer_ready)
            ssd_ready = min(block_mask_end, ssd_match.peer_ready)
            lake_ready = min(block_mask_end, lake_match.peer_ready)
            self._metrics_collector.record_cache_hit(
                "cpu", max(0, cpu_ready - block_mask_start))
            self._metrics_collector.record_cache_hit(
                "ssd", max(0, ssd_ready - max(cpu_ready, block_mask_start)))
            self._metrics_collector.record_cache_hit(
                "lake",
                max(0, lake_ready - max(cpu_ready, ssd_ready, block_mask_start)))
            miss_blocks = total_query_blocks - num_gpu_blocks
            if miss_blocks > 0:
                self._metrics_collector.record_cache_miss(miss_blocks)

        # Take over protection from the match's atomic pre-lock: lock the
        # completion nodes (their inserted descendant in the promotion cases),
        # then drop the pre-lock so a per-match ref is never leaked.
        self._handoff_locks(
            node_to_unlock,
            cpu_result=cpu_match,
            ssd_result=ssd_match,
            lake_result=lake_match)
        nvtx.end_range(nvtx_range)
        return (
            transfer_graph, finished_ops_ids, node_to_unlock, op_node_to_ready,
            buffer_to_free, num_gpu_blocks if enable_gpu else 0
        )

    def _get_impl_without_lake(self,
                               request_id: int,
                               sequence_meta: SequenceMeta,
                               block_mask_start: int,
                               block_mask_end: int,
                               gpu_block_ids: np.ndarray,
                               layer_num: int,
                               temp_cache_strategy: CacheStrategy) \
            -> Tuple[TransferOpGraph, List[int], Dict, Dict, Dict, int]:
        """Plan and build a GET graph across cpu / ssd (no lake tier).

        Phase 1 (:func:`get_media_list`) decides *which* media serve the queried
        prefix; the inlined phase 2 decides *how* each reaches the GPU and applies
        promotion.  Covers local/peer cpu and local/peer ssd, with optional GDS
        and direct peer->GPU routes.  LAKE and P2P are mutually exclusive
        (validated at config), so no lake segment ever appears here.
        """
        nvtx_range = nvtx.start_range(
            message=f"CacheEngine._get_impl_without_lake[{request_id}]", color="cyan")
        enable_gpu = not temp_cache_strategy.ignore_gpu
        enable_gds = self.cache_config.enable_gds and not temp_cache_strategy.ignore_gds
        enable_peer_gpu = enable_gpu and self.cache_config.enable_p2p_gpu
        assert self.cache_config.enable_cpu
        assert self.cpu_cache_engine is not None

        cpu_match, ssd_match, _lake_match = self.match_all(
            sequence_meta,
            temp_cache_strategy=temp_cache_strategy,
            is_put=False,
            gpu_matched_blocks=block_mask_start,
        )
        segments = get_media_list(
            cpu_match, ssd_match, None,
            block_mask_start=block_mask_start,
            block_mask_end=block_mask_end,
        )
        total_query_blocks = block_mask_end - block_mask_start
        if not segments:
            if self._metrics_collector is not None and total_query_blocks > 0:
                self._metrics_collector.record_cache_miss(total_query_blocks)
            nvtx.end_range(nvtx_range)
            self._release_match_pre_locks(
                cpu_result=cpu_match,
                ssd_result=ssd_match)
            return self._empty_get_return(request_id)

        num_gpu_blocks = segments[-1].logical_end - block_mask_start
        assert num_gpu_blocks <= len(gpu_block_ids)

        num_staging = plan_routes(
            segments,
            enable_gpu=enable_gpu,
            enable_gds=enable_gds,
            enable_peer_gpu=enable_peer_gpu,
        )

        nvtx.push_range(f"take {num_staging} cpu blocks", color="green")
        cpu_staging_blocks = self.cpu_cache_engine.take(
            num_required_blocks=num_staging,
            protected_node=cpu_match.local.last_node,
            strict=False,
        )
        nvtx.pop_range()
        if len(cpu_staging_blocks) < num_staging:
            self.cpu_cache_engine.recycle(cpu_staging_blocks)
            if self._metrics_collector is not None:
                self._metrics_collector.record_allocation_failure("local")
            nvtx.end_range(nvtx_range)
            self._release_match_pre_locks(
                cpu_result=cpu_match,
                ssd_result=ssd_match)
            return self._empty_get_return(request_id)

        transfer_graph, finished_ops_ids, _h2d_ops = build_transfer_graph(
            segments,
            staging_blocks=cpu_staging_blocks,
            block_mask_start=block_mask_start,
            layer_num=layer_num,
        )
        op_node_to_ready: Dict[int, Tuple[DeviceType, List[RadixNodeLike]]] = {}

        used: Dict[DeviceType, set] = {}
        for seg in segments:
            used.setdefault(seg.tier, set()).add(seg.locality)
        cpu_nodes = GlobalCacheEngine._completion_nodes(
            cpu_match, used.get(DeviceType.CPU, set()))
        ssd_nodes = GlobalCacheEngine._completion_nodes(
            ssd_match, used.get(DeviceType.SSD, set()))
        cpu_blocks_to_free: List[np.ndarray] = []
        promoted_ids = set()  # id(seg) whose staging was inserted into an index

        local_cpu = cpu_match.local
        local_cpu_ready = local_cpu.num_ready_matched_blocks
        cpu_has_peer = CacheLocality.PEER in used.get(DeviceType.CPU, set())

        # (A) Promote a staged PEER-CPU suffix into the local CPU index so the
        # next GET hits CPU directly.  Keeps the matched guard AND the inserted
        # node (the inserted copy is independent of the still-needed peer source).
        peer_cpu_segs = [
            seg for seg in segments
            if seg.tier == DeviceType.CPU and seg.is_peer and seg.needs_staging
        ]
        if peer_cpu_segs:
            cpu_matched = max(
                local_cpu.num_matched_blocks,
                cpu_match.remote.num_matched_blocks if cpu_match.remote else 0,
            )
            first = peer_cpu_segs[0]
            can_promote = (
                    cpu_match.peer_ready == cpu_matched and
                    local_cpu_ready == local_cpu.num_matched_blocks and
                    block_mask_start <= local_cpu_ready and
                    first.logical_start == local_cpu_ready
            )
            if can_promote:
                trigger = peer_cpu_segs[-1].primary_op
                assert trigger is not None  # a staged peer-CPU seg always has PEERH2H
                GlobalCacheEngine._promote_apply(
                    self.cpu_cache_engine, sequence_meta,
                    np.concatenate([s.staging for s in peer_cpu_segs]),
                    num_insert_blocks=cpu_match.peer_ready,
                    match_result=(
                        local_cpu if local_cpu.num_matched_blocks > 0 else None),
                    completion_nodes=cpu_nodes, replace=False,
                    blocks_to_free=cpu_blocks_to_free,
                    op_node_to_ready=op_node_to_ready,
                    device_type=DeviceType.CPU, ready_op_id=trigger.op_id,
                )
                promoted_ids.update(id(s) for s in peer_cpu_segs)

        # (B) Promote a staged local/peer SSD suffix into the local CPU index when
        # CPU is a purely-local, fully-ready prefix (no peer CPU).  The inserted
        # descendant pins that prefix, so it REPLACES the matched node.
        ssd_staged = [
            seg for seg in segments
            if seg.tier == DeviceType.SSD and seg.needs_staging
        ]
        if ssd_staged and not cpu_has_peer:
            first = ssd_staged[0]
            triggers = {
                s.h2d_op.op_id: s.h2d_op
                for s in ssd_staged if s.h2d_op is not None
            }
            if not triggers:  # when gpu not enabled
                triggers = {
                    s.primary_op.op_id: s.primary_op
                    for s in ssd_staged if s.primary_op is not None
                }
            base_ok = (
                    local_cpu_ready == local_cpu.num_matched_blocks and
                    block_mask_start <= local_cpu_ready and
                    first.logical_start == local_cpu_ready
            )
            # A pure-SSD promotion needs exactly one feeding op so it can flip
            # ready early (via that op's completion callback).
            if base_ok and len(triggers) == 1:
                GlobalCacheEngine._promote_apply(
                    self.cpu_cache_engine, sequence_meta,
                    np.concatenate([s.staging for s in ssd_staged]),
                    num_insert_blocks=block_mask_start + num_gpu_blocks,
                    match_result=local_cpu,
                    completion_nodes=cpu_nodes, replace=True,
                    blocks_to_free=cpu_blocks_to_free,
                    op_node_to_ready=op_node_to_ready,
                    device_type=DeviceType.CPU, ready_op_id=next(iter(triggers)),
                )
                promoted_ids.update(id(s) for s in ssd_staged)

        # Local SSD direct-to-GPU (GDS): flip the matched SSD node ready as soon
        # as its DISK2D op completes, preserving the legacy ready-callback.
        if ssd_nodes:
            for seg in segments:
                if (seg.tier == DeviceType.SSD and not seg.is_peer and
                        seg.primary_type == TransferType.DISK2D and
                        seg.primary_op is not None):
                    op_node_to_ready[seg.primary_op.op_id] = (
                        DeviceType.SSD, ssd_nodes)

        # Recycle staging that no promotion adopted into an index.
        for seg in segments:
            if seg.staging is not None and id(seg) not in promoted_ids:
                cpu_blocks_to_free.append(seg.staging)

        node_to_unlock: Dict[DeviceType, List[RadixNodeLike]] = {}
        if cpu_nodes:
            node_to_unlock[DeviceType.CPU] = cpu_nodes
        if ssd_nodes:
            node_to_unlock[DeviceType.SSD] = ssd_nodes

        buffer_to_free: Dict[DeviceType, np.ndarray] = {}
        if cpu_blocks_to_free:
            buffer_to_free[DeviceType.CPU] = np.concatenate(cpu_blocks_to_free)

        # This is for sync get
        transfer_graph.set_gpu_blocks(gpu_block_ids)

        if self._metrics_collector is not None:
            cpu_ready = min(block_mask_end, cpu_match.peer_ready)
            ssd_ready = min(block_mask_end, ssd_match.peer_ready)
            self._metrics_collector.record_cache_hit(
                "cpu", max(0, cpu_ready - block_mask_start))
            self._metrics_collector.record_cache_hit(
                "ssd", max(0, ssd_ready - max(cpu_ready, block_mask_start)))
            miss_blocks = total_query_blocks - num_gpu_blocks
            if miss_blocks > 0:
                self._metrics_collector.record_cache_miss(miss_blocks)

        # Take over protection from the match's atomic pre-lock: lock the
        # completion nodes (their inserted descendant in the promotion cases),
        # then drop the pre-lock so a per-match ref is never leaked.
        self._handoff_locks(
            node_to_unlock,
            cpu_result=cpu_match,
            ssd_result=ssd_match)
        nvtx.end_range(nvtx_range)
        return (
            transfer_graph, finished_ops_ids, node_to_unlock, op_node_to_ready,
            buffer_to_free, num_gpu_blocks if enable_gpu else 0
        )

    def put(self,
            request_id: int,
            token_ids: np.ndarray,
            token_mask: np.ndarray,
            slot_mapping: np.ndarray,
            layer_num : int = -1,
            dp_id: int = 0,
            temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY,
            namespace: Optional[List[str]] = None) \
                -> Tuple[TransferOpGraph, np.ndarray, Callable, Dict, int]:
        self._check_input(token_ids, token_mask, slot_mapping)

        if layer_num == -1:
            layer_num = self.model_config.num_layers
        # ignore the last incomplete block
        aligned_length = (token_ids.shape[0] // self.tokens_per_block) * self.tokens_per_block
        aligned_token_ids = token_ids[:aligned_length]
        token_mask[aligned_length:] = False
        block_start_idx, block_end_idx = self._get_block_range(token_mask)

        # the mask should has a prefix of True
        assert block_start_idx == 0

        gpu_block_ids = self.slot_mapping_to_block_ids(slot_mapping,
                                                       self.tokens_per_block)[:block_end_idx-block_start_idx]

        sequence_meta = SequenceMeta(token_ids=aligned_token_ids,
                                     tokens_per_block=self.cache_config.tokens_per_block,
                                     namespace=namespace)

        assert not temp_cache_strategy.ignore_gpu
        if not self.cache_config.enable_lake or temp_cache_strategy.ignore_lake:
            (transfer_graph, finished_ops_ids, node_to_unlock, op_node_to_ready,
             buffer_to_free, num_gpu_blocks_to_transfer, skipped_gpu_blocks) = \
                self._put_impl_without_lake(
                    request_id,
                    sequence_meta,
                    block_start_idx,
                    block_end_idx,
                    gpu_block_ids,
                    layer_num,
                    temp_cache_strategy
                )
        else:
            (transfer_graph, finished_ops_ids, node_to_unlock, op_node_to_ready,
             buffer_to_free, num_gpu_blocks_to_transfer, skipped_gpu_blocks) = \
                self._put_impl_with_lake(
                    request_id,
                    sequence_meta,
                    block_start_idx,
                    block_end_idx,
                    gpu_block_ids,
                    layer_num,
                    temp_cache_strategy
                )

        transfer_graph, task_end_op_id = add_virtal_op_for_mutiple_finished_ops(
            transfer_graph,
            finished_ops_ids
        )

        return_mask = np.zeros_like(token_mask, dtype=np.bool_)
        return_mask[(block_start_idx + skipped_gpu_blocks)* self.tokens_per_block:
                    (block_start_idx + skipped_gpu_blocks + num_gpu_blocks_to_transfer) * self.tokens_per_block] = True
        transfer_graph.bind_to_dp_group(dp_id)

        # lock_node + match pre-lock release happens inside the _impl methods
        # to avoid a per-match ref leak window between impl return and lock.

        callback = partial(self._transfer_callback,
                           node_to_unlock=node_to_unlock,
                           buffer_to_free=buffer_to_free,
                           is_put=True)

        op_callback_dict = {}
        for op_id, (device_type, nodes) in op_node_to_ready.items():
            op_callback_dict[op_id] = partial(self._op_callback,
                                              device_type=device_type,
                                              nodes=nodes)

        # Update mempool metrics after PUT operation
        if self._metrics_collector is not None:
            self._update_mempool_metrics()

        return transfer_graph, return_mask, callback, op_callback_dict, task_end_op_id

    def _put_impl_with_lake(self,
                            request_id: int,
                            sequence_meta: SequenceMeta,
                            block_mask_start: int,
                            block_mask_end: int,
                            gpu_block_ids: np.ndarray,
                            layer_num : int,
                            temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY) \
                -> Tuple[TransferOpGraph, List[int], Dict, Dict, Dict, int, int]:
        """
        transfer pattern:

        GPU:   (skipped)  | fragment1      | fragment2      | (uncompleted block)
                               ↓                ↓
        CPU: (cpu cached) | fragment1(new) | fragment2(new) |
                                                ↓
        SSD:          (ssd cached)         | fragment2(new) |

        CPU:            ...           |     fragment3      |
                                               ↓ (from cpu)
        LAKE:      (lake cached)   |   fragment3(new)   |

        """
        enable_gpu = not temp_cache_strategy.ignore_gpu
        enable_cpu = self.cache_config.enable_cpu
        enable_ssd = self.cache_config.enable_ssd and not temp_cache_strategy.ignore_ssd
        enable_lake = self.cache_config.enable_lake and not temp_cache_strategy.ignore_lake
        assert enable_gpu
        assert enable_cpu
        assert enable_lake
        assert self.cpu_cache_engine is not None
        assert self.lake_cache_engine is not None

        cpu_match, ssd_match, lake_match = self.match_all(
            sequence_meta, temp_cache_strategy=temp_cache_strategy, is_put=True)
        # PUT is local-only; each MatchResult carries just its `.local` side.
        cpu_matched_result = cpu_match.local
        ssd_matched_result = ssd_match.local
        lake_matched_result = lake_match.local
        cpu_matched_blocks = cpu_matched_result.physical_blocks[
            :cpu_matched_result.num_matched_blocks][block_mask_start:block_mask_end]
        ssd_matched_blocks = ssd_matched_result.physical_blocks[
            :ssd_matched_result.num_matched_blocks][block_mask_start:block_mask_end]
        lake_matched_blocks = lake_matched_result.physical_blocks[
            :lake_matched_result.num_matched_blocks][block_mask_start:block_mask_end]

        num_skipped_blocks = len(cpu_matched_blocks)
        fragment12_num_blocks = len(gpu_block_ids) - num_skipped_blocks
        if fragment12_num_blocks == 0:
            self._release_match_pre_locks(
                cpu_result=cpu_match,
                ssd_result=ssd_match,
                lake_result=lake_match)
            return self._empty_put_return(request_id)
        fragment2_num_blocks = len(gpu_block_ids) - len(ssd_matched_blocks)
        if not enable_ssd:
            fragment2_num_blocks = 0
        fragment3_num_blocks = len(gpu_block_ids) - len(lake_matched_blocks)

        fragment12_gpu_blocks = gpu_block_ids[num_skipped_blocks:]

        fragment12_cpu_blocks = self.cpu_cache_engine.take(
            num_required_blocks=fragment12_num_blocks,
            protected_node = cpu_matched_result.last_node,
            strict=False
        )
        if len(fragment12_cpu_blocks) < fragment12_num_blocks:
            self.cpu_cache_engine.recycle(fragment12_cpu_blocks)
            self._release_match_pre_locks(
                cpu_result=cpu_match,
                ssd_result=ssd_match,
                lake_result=lake_match)
            return self._empty_put_return(request_id)
        put_to_ssd = False
        if enable_ssd and fragment2_num_blocks > 0:
            fragment2_ssd_blocks = self.ssd_cache_engine.take(
                num_required_blocks=fragment2_num_blocks,
                protected_node = ssd_matched_result.last_node,
                strict=False
            )
            if len(fragment2_ssd_blocks) == fragment2_num_blocks:
                put_to_ssd = True
            else:
                self.ssd_cache_engine.recycle(fragment2_ssd_blocks)
        else:
            fragment2_ssd_blocks = np.array([], dtype=np.int64)
        put_to_lake = False
        if fragment3_num_blocks > 0:
            fragment3_lake_blocks = self.lake_cache_engine.take(
                num_required_blocks=fragment3_num_blocks,
                protected_node = lake_matched_result.last_node,
                strict=False
            )
            if len(fragment3_lake_blocks) == fragment3_num_blocks:
                put_to_lake = True
            else:
                self.lake_cache_engine.recycle(fragment3_lake_blocks)
        else:
            fragment3_lake_blocks = np.array([], dtype=np.int64)

        transfer_graph = TransferOpGraph()
        finished_ops_ids = []

        op_d2h = TransferOp(
            graph_id = transfer_graph.graph_id,
            transfer_type = TransferType.D2H,
            src_block_ids = fragment12_gpu_blocks,
            dst_block_ids = fragment12_cpu_blocks,
            layer_id = 0,
            layer_granularity = layer_num
        )
        transfer_graph.add_transfer_op(op_d2h)
        finished_ops_ids.append(op_d2h.op_id)

        if put_to_ssd:
            if len(fragment12_cpu_blocks) < fragment2_num_blocks:
                num_needed_from_cpu_matched = fragment2_num_blocks - len(fragment12_cpu_blocks)
                fragment2_cpu_blocks = np.concatenate([cpu_matched_blocks[-num_needed_from_cpu_matched:], \
                    fragment12_cpu_blocks])
            else:
                fragment2_cpu_blocks = fragment12_cpu_blocks[-fragment2_num_blocks:]
            op_h2disk = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.H2DISK,
                src_block_ids = fragment2_cpu_blocks,
                dst_block_ids = fragment2_ssd_blocks,
                layer_id = 0,
                layer_granularity = layer_num
            )
            transfer_graph.add_transfer_op(op_h2disk)

            transfer_graph.add_dependency(op_h2disk.op_id, op_d2h.op_id)
            finished_ops_ids.append(op_h2disk.op_id)

        if put_to_lake:
            if fragment3_num_blocks > fragment12_num_blocks:
                extra_num_cpu_blocks = fragment3_num_blocks - fragment12_num_blocks
                fragment3_cpu_blocks = np.concatenate([
                    cpu_matched_blocks[-extra_num_cpu_blocks:],
                    fragment12_cpu_blocks,
                ])
            else:
                fragment3_cpu_blocks = fragment12_cpu_blocks[-fragment3_num_blocks:]
            op_h2lake = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.H2LAKE,
                src_block_ids = fragment3_cpu_blocks,
                dst_block_ids = fragment3_lake_blocks,
                layer_id = 0,
                layer_granularity = layer_num
            )
            transfer_graph.add_transfer_op(op_h2lake)
            transfer_graph.add_dependency(op_h2lake.op_id, op_d2h.op_id)
            finished_ops_ids.append(op_h2lake.op_id)

        # Defer recycling of slots radixshmem didn't attach (race with another
        # DP that already inserted the same prefix). They're still committed
        # in our TransferGraph; recycling them now would let another DP reuse
        # the slot id mid-transfer and corrupt data.
        buffer_to_free: Dict[DeviceType, np.ndarray] = {}
        cpu_node_to_unlock, cpu_unused = self.cpu_cache_engine.insert(
            sequence_meta, fragment12_cpu_blocks,
            is_ready=False, match_result=cpu_matched_result)
        if cpu_unused.size > 0:
            buffer_to_free[DeviceType.CPU] = cpu_unused
        ssd_node_to_unlock = None
        if put_to_ssd:
            ssd_node_to_unlock, ssd_unused = self.ssd_cache_engine.insert(
                sequence_meta, fragment2_ssd_blocks,
                is_ready=False, match_result=ssd_matched_result)
            if ssd_unused.size > 0:
                buffer_to_free[DeviceType.SSD] = ssd_unused
        lake_node_to_unlock = None
        if put_to_lake:
            lake_node_to_unlock, lake_unused = self.lake_cache_engine.insert(
                sequence_meta, fragment3_lake_blocks,
                is_ready=False, match_result=lake_matched_result)
            if lake_unused.size > 0:
                buffer_to_free[DeviceType.LAKE] = lake_unused
        node_to_unlock = {}
        if cpu_node_to_unlock is not None:
            node_to_unlock[DeviceType.CPU] = [cpu_node_to_unlock]
        if ssd_node_to_unlock is not None:
            node_to_unlock[DeviceType.SSD] = [ssd_node_to_unlock]
        if lake_node_to_unlock is not None:
            node_to_unlock[DeviceType.LAKE] = [lake_node_to_unlock]

        # Take over protection via lock_node, then drop the match's atomic
        # pre-lock so it doesn't accumulate per request.
        self._handoff_locks(
            node_to_unlock,
            cpu_result=cpu_match,
            ssd_result=ssd_match,
            lake_result=lake_match)

        skipped_gpu_blocks = len(cpu_matched_blocks)
        return (
            transfer_graph, finished_ops_ids, node_to_unlock, {}, buffer_to_free,
            len(fragment12_gpu_blocks), skipped_gpu_blocks  # op_node_to_ready: {}
        )

    def _put_impl_without_lake(self,
                               request_id: int,
                               sequence_meta: SequenceMeta,
                               block_mask_start: int,
                               block_mask_end: int,
                               gpu_block_ids: np.ndarray,
                               layer_num : int,
                               temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY) \
                -> Tuple[TransferOpGraph, List[int], Dict, Dict, Dict, int, int]:
        """
        transfer pattern:

        GPU:   (skipped)  | fragment1      | fragment2      | (uncompleted block)
                                ↓                ↓
        CPU: (cpu cached) | fragment1(new) | fragment2(new) |
                                                 ↓
        SSD:          (ssd cached)         | fragment2(new) |

        """
        enable_gpu = not temp_cache_strategy.ignore_gpu
        enable_cpu = self.cache_config.enable_cpu
        enable_ssd = self.cache_config.enable_ssd and not temp_cache_strategy.ignore_ssd
        enable_gds = self.cache_config.enable_gds and not temp_cache_strategy.ignore_gds
        assert enable_gpu
        assert enable_cpu
        assert self.cpu_cache_engine is not None

        cpu_match, ssd_match, _lake_match = self.match_all(
            sequence_meta, temp_cache_strategy=temp_cache_strategy, is_put=True)
        # PUT is local-only; each MatchResult carries just its `.local` side.
        cpu_matched_result = cpu_match.local
        ssd_matched_result = ssd_match.local
        cpu_matched_blocks = cpu_matched_result.physical_blocks[
            :cpu_matched_result.num_matched_blocks][block_mask_start:block_mask_end]
        ssd_matched_blocks = ssd_matched_result.physical_blocks[
            :ssd_matched_result.num_matched_blocks][block_mask_start:block_mask_end]

        num_skipped_blocks = len(cpu_matched_blocks)
        fragment12_num_blocks = len(gpu_block_ids) - num_skipped_blocks
        if fragment12_num_blocks == 0:
            self._release_match_pre_locks(
                cpu_result=cpu_match,
                ssd_result=ssd_match)
            return self._empty_put_return(request_id)
        fragment2_num_blocks = len(gpu_block_ids) - len(ssd_matched_blocks)
        if not enable_ssd:
            fragment2_num_blocks = 0

        fragment12_gpu_blocks = gpu_block_ids[num_skipped_blocks:]

        fragment12_cpu_blocks = self.cpu_cache_engine.take(
            num_required_blocks=fragment12_num_blocks,
            protected_node = cpu_matched_result.last_node,
            strict=False
        )

        if enable_ssd:
            fragment2_ssd_blocks = self.ssd_cache_engine.take(
                num_required_blocks=fragment2_num_blocks,
                protected_node = ssd_matched_result.last_node,
                strict=False
            )
        else:
            fragment2_ssd_blocks = np.array([], dtype=np.int64)

        if len(fragment12_cpu_blocks) < fragment12_num_blocks or \
            len(fragment2_ssd_blocks) < fragment2_num_blocks:
            print(f"[WARNING] PUT request {request_id} FAILED: CPU={len(fragment12_cpu_blocks)}/{fragment12_num_blocks}, SSD={len(fragment2_ssd_blocks)}/{fragment2_num_blocks}")
            self.cpu_cache_engine.recycle(fragment12_cpu_blocks)
            if enable_ssd:
                self.ssd_cache_engine.recycle(fragment2_ssd_blocks)
            self._release_match_pre_locks(
                cpu_result=cpu_match,
                ssd_result=ssd_match)
            return self._empty_put_return(request_id)

        transfer_graph = TransferOpGraph()
        finished_ops_ids = []
        op_node_to_ready = {}

        op_d2h = TransferOp(
            graph_id = transfer_graph.graph_id,
            transfer_type = TransferType.D2H,
            src_block_ids = fragment12_gpu_blocks,
            dst_block_ids = fragment12_cpu_blocks,
            layer_id = 0,
            layer_granularity = layer_num
        )
        transfer_graph.add_transfer_op(op_d2h)
        finished_ops_ids.append(op_d2h.op_id)

        if fragment2_num_blocks > 0:
            if len(fragment12_cpu_blocks) < fragment2_num_blocks:
                flexkv_logger.warning(f"fragment12_cpu_blocks: {len(fragment12_cpu_blocks)}, "
                                      f"fragment2_num_blocks: {fragment2_num_blocks}, "
                                      f"cpu match blocks are bigger than SSD match blocks number. "
                                      f"This should not often happen if CPU cache size is smaller than SSD cache size.")
                num_needed_from_cpu_matched = fragment2_num_blocks - len(fragment12_cpu_blocks)
                fragment2_cpu_blocks = np.concatenate([cpu_matched_blocks[-num_needed_from_cpu_matched:], \
                    fragment12_cpu_blocks])
            else:
                fragment2_cpu_blocks = fragment12_cpu_blocks[-fragment2_num_blocks:]
            op_h2disk = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.H2DISK,
                src_block_ids = fragment2_cpu_blocks,
                dst_block_ids = fragment2_ssd_blocks,
                layer_id = 0,
                layer_granularity = layer_num
            )
            transfer_graph.add_transfer_op(op_h2disk)

            transfer_graph.add_dependency(op_h2disk.op_id, op_d2h.op_id)

        """insert and lock"""
        # Defer recycling of any slots radixshmem didn't attach (race with
        # another DP) — they're still committed in our TransferGraph.
        buffer_to_free: Dict[DeviceType, np.ndarray] = {}
        cpu_node_to_unlock, cpu_unused = self.cpu_cache_engine.insert(
            sequence_meta, fragment12_cpu_blocks,
            is_ready=False, match_result=cpu_matched_result)
        # insert() returns None when nothing was attached (the whole suffix was
        # already present in the shared tree) — then there is no unready node to
        # flip ready after the transfer, so skip the ready-callback bookkeeping.
        if cpu_node_to_unlock is not None:
            op_node_to_ready[op_d2h.op_id] = (DeviceType.CPU, [cpu_node_to_unlock])
        if cpu_unused.size > 0:
            buffer_to_free[DeviceType.CPU] = cpu_unused
        ssd_node_to_unlock = None
        if len(fragment2_ssd_blocks) > 0:
            ssd_node_to_unlock, ssd_unused = self.ssd_cache_engine.insert(
                sequence_meta, fragment2_ssd_blocks,
                is_ready=False, match_result=ssd_matched_result)
            if ssd_node_to_unlock is not None:
                op_node_to_ready[op_h2disk.op_id] = (DeviceType.SSD, [ssd_node_to_unlock])
            if ssd_unused.size > 0:
                buffer_to_free[DeviceType.SSD] = ssd_unused
        node_to_unlock = {}
        if cpu_node_to_unlock is not None:
            node_to_unlock[DeviceType.CPU] = [cpu_node_to_unlock]
        if ssd_node_to_unlock is not None:
            node_to_unlock[DeviceType.SSD] = [ssd_node_to_unlock]

        # Take over protection via lock_node, then drop the match's atomic
        # pre-lock so it doesn't accumulate per request.
        self._handoff_locks(
            node_to_unlock,
            cpu_result=cpu_match,
            ssd_result=ssd_match)

        skipped_gpu_blocks = len(cpu_matched_blocks)
        return (
            transfer_graph, finished_ops_ids, node_to_unlock, op_node_to_ready, buffer_to_free,
            len(fragment12_gpu_blocks), skipped_gpu_blocks
        )

    def _transfer_callback(self,
                           node_to_unlock: Dict[DeviceType, List[RadixNodeLike]],
                           buffer_to_free: Optional[Dict[DeviceType, np.ndarray]] = None,
                           is_put: bool = False) -> None:
        # Order matters: under shmradix the cache index is in shared memory and
        # another DP's auto-evict could pick up a node the moment its ref drops
        # to 0.  Per tier we set_ready EVERY node THEN unlock EVERY node.  For an
        # inserted node the shmradix engine fuses both into one armed `finalize`
        # (set_ready + dec_ref) fired by unlock(), so set_ready() is a no-op and
        # unlock() does the atomic flip-then-release.  For a matched node
        # set_ready is an idempotent no-op and unlock drops the ref.
        for device_type in (DeviceType.CPU, DeviceType.SSD, DeviceType.LAKE):
            nodes = node_to_unlock.get(device_type)
            if not nodes:
                continue
            engine = self.cache_engines[device_type]
            assert engine is not None
            for node in nodes:
                engine.set_ready(node, True, node.size())
            for node in nodes:
                engine.unlock(node)
            if is_put:
                if device_type == DeviceType.CPU:
                    should_publish = self.cache_config.enable_p2p_cpu
                elif device_type == DeviceType.SSD:
                    should_publish = self.cache_config.enable_p2p_ssd
                else:
                    should_publish = self.enable_kv_sharing
                if should_publish:
                    for node in nodes:
                        if hasattr(engine, "publish_ready"):
                            engine.publish_ready(node)
                        else:
                            engine.local_index.insert_and_publish(node)
        if buffer_to_free is not None:
            for device_type in (DeviceType.CPU, DeviceType.SSD, DeviceType.LAKE):
                blocks = buffer_to_free.get(device_type)
                if blocks is not None:
                    engine = self.cache_engines[device_type]
                    assert engine is not None
                    engine.recycle(blocks)

    def _op_callback(self,
                     device_type: DeviceType,
                     nodes: List[RadixNodeLike]) -> None:
        engine = self.cache_engines[device_type]
        assert engine is not None
        for node in nodes:
            engine.set_ready(node, True, node.size())

    @staticmethod
    def _empty_match() -> MatchResult:
        return MatchResult(local=MatchResultAccel())

    @nvtx.annotate("Match Prefix Accel", color="yellow")
    def match_all(self,
                  sequence_meta: SequenceMeta,
                  temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY,
                  is_put: bool = False,
                  gpu_matched_blocks: int = 0) \
                     -> Tuple[MatchResult, MatchResult, MatchResult]:
        """Match every enabled tier, returning ``(cpu, ssd, lake)``.

        A tier that is disabled or ignored comes back empty.  PUT writes only
        locally, so it never queries a peer index (``with_peer=False``); GET
        queries peers.  LAKE and P2P are mutually exclusive, so this single
        matcher serves both the peer and the lake plans.
        """
        with_peer = not is_put
        cpu_match = GlobalCacheEngine._empty_match()
        ssd_match = GlobalCacheEngine._empty_match()
        lake_match = GlobalCacheEngine._empty_match()
        if self.cpu_cache_engine:
            cpu_match = self.cpu_cache_engine.match(
                sequence_meta, with_peer=with_peer,
                gpu_matched_blocks=gpu_matched_blocks,
            )
        if self.ssd_cache_engine and not temp_cache_strategy.ignore_ssd:
            ssd_match = self.ssd_cache_engine.match(
                sequence_meta, with_peer=with_peer,
                gpu_matched_blocks=gpu_matched_blocks,
            )
        if self.lake_cache_engine and not temp_cache_strategy.ignore_lake:
            lake_match = self.lake_cache_engine.match(
                sequence_meta, with_peer=with_peer,
                gpu_matched_blocks=gpu_matched_blocks,
            )
        return cpu_match, ssd_match, lake_match

    def _check_input(self,
                      token_ids: np.ndarray,
                      token_mask: np.ndarray,
                      slot_mapping: np.ndarray) -> None:
        assert token_ids.dtype == np.int64
        # assert token_mask.dtype == np.bool_, f"token_mask.dtype={token_mask.dtype}"
        assert slot_mapping.dtype == np.int64
        assert token_ids.ndim == 1
        assert token_mask.ndim == 1
        assert slot_mapping.ndim == 1
        assert token_ids.size == token_mask.size, f"token_ids.size={token_ids.size}, token_mask.size={token_mask.size}"
        assert slot_mapping.size == token_mask.sum(), \
            f"slot_mapping.size={slot_mapping.size}, token_mask.sum()={token_mask.sum()}"

    @staticmethod
    def slot_mapping_to_block_ids(slot_mapping: np.ndarray, tokens_per_block: int) -> np.ndarray:
        block_ids: np.ndarray = slot_mapping[::tokens_per_block] // tokens_per_block
        return block_ids

    def _get_block_range(self,
                         token_mask: np.ndarray) -> Tuple[int, int]:
        mask_idx = np.where(token_mask)[0]
        if len(mask_idx) == 0:
            return 0, 0
        start_idx = mask_idx[0].item() // self.tokens_per_block
        end_idx = mask_idx[-1].item() // self.tokens_per_block
        return start_idx, end_idx + 1
