from typing import Optional, Tuple, TYPE_CHECKING, List, Dict, Union

import numpy as np
import torch
from numpy import ndarray

from flexkv.c_ext import CRadixNode
from flexkv import c_ext
from flexkv.cache.mempool import Mempool
from flexkv.cache.radix_remote import LocalRadixTree, DistributedRadixTree
from flexkv.cache.redis_meta import RedisMeta
from flexkv.common.block import SequenceMeta
#if TYPE_CHECKING:
from flexkv.common.config import CacheConfig, GLOBAL_CONFIG_FROM_ENV
from flexkv.common.transfer import DeviceType
from flexkv.common.source import BlockSource, LakeSource, LocalSource, PeerSource
from flexkv.common.type import (
    MatchResult,
    MatchResultAccel,
)


class HierarchyLRCacheEngine:
    def __init__(self,
                 num_total_blocks: int,
                 tokens_per_block: int,
                 evict_ratio: float,
                device_type: DeviceType,
                # Optional runtime wiring for remote/local trees
                local_max_num_blocks: Optional[int] = 0,
                local_lease_ttl_ms: int = 100000,
                 local_renew_lease_ms: int = 10000,
                 local_refresh_batch_size: int = 1000,
                 local_idle_sleep_ms: int = 10,
                 remote_max_num_blocks: int = 4000000,
                 redis_node_id: int = 0,
                 remote_refresh_batch_size: int = 1000,
                 remote_rebuild_interval_ms: int = 100,
                 remote_idle_sleep_ms: int = 10,
                 local_safety_ttl_ms: int = 100,
                 evict_start_threshold: float = 1.0,
                 hit_reward_seconds: int = 0,
                 eviction_policy: str = "lru",
                 meta: Optional[RedisMeta] = None) -> None:
        if num_total_blocks <= 0:
            raise ValueError(f"Invalid num_total_blocks: {num_total_blocks}")
        if tokens_per_block <= 0 or (tokens_per_block & (tokens_per_block - 1)) != 0:
            raise ValueError(
                f"Invalid tokens_per_block: {tokens_per_block}, tokens_per_block must be a power of 2"
            )

        self.device_type = device_type
        self._meta: Optional[RedisMeta] = meta # todo: define storage type in meta
        self.redis_node_id = int(redis_node_id)


        # belows are only for 3rd-party lake storage (like pcfs)
        # Mapping: node_id -> list of PCFS file_nodeids
        self.nid_to_file_nodeids: Dict[int, List[int]] = {}
        # Partition parameter used for mapping block_id to file index
        self.round_robin: int = 1

        # Local index (authoritative for mutations)
        self.local_index = LocalRadixTree(
            tokens_per_block=tokens_per_block,
            max_num_blocks=int(num_total_blocks),
            lease_ttl_ms=int(local_lease_ttl_ms),
            renew_lease_ms=int(local_renew_lease_ms),
            refresh_batch_size=int(local_refresh_batch_size),
            idle_sleep_ms=int(local_idle_sleep_ms),
            safety_ttl_ms=int(local_safety_ttl_ms),
            swap_block_threshold=int(evict_ratio * num_total_blocks),
            hit_reward_seconds=int(hit_reward_seconds),
            eviction_policy=eviction_policy,
        )


        # Remote reference index (read-only, built from Redis)
        self.remote_index = DistributedRadixTree(
            tokens_per_block=tokens_per_block,
            max_num_blocks=int(remote_max_num_blocks or (num_total_blocks * 10)),
            node_id=int(redis_node_id),
            refresh_batch_size=int(remote_refresh_batch_size),
            rebuild_interval_ms=int(remote_rebuild_interval_ms),
            idle_sleep_ms=int(remote_idle_sleep_ms),
            lease_renew_ms=int(local_renew_lease_ms),
            hit_reward_seconds=int(hit_reward_seconds),
        )
        # defer channel start to start(meta)

        # Local memory pool for physical blocks on this device
        self.mempool = Mempool(num_total_blocks=int(local_max_num_blocks or num_total_blocks))

        self.tokens_per_block = tokens_per_block
        self.num_total_blocks = num_total_blocks
        self.evict_ratio = evict_ratio
        self.evict_start_threshold = evict_start_threshold
        
        # cumulative statistics: for analyzing distributed KV reuse benefits
        self._stats_total_queried_tokens = 0       # total tokens queried
        self._stats_gpu_matched_tokens = 0         # total tokens matched in GPU memory
        self._stats_local_matched_tokens = 0       # total tokens matched in FlexKV local
        self._stats_distributed_matched_tokens = 0 # total tokens matched in FlexKV global (distributed reuse)
        self._stats_match_count = 0                # match_all call count

    def start(self) -> None:
        if self._meta is None:
            raise ValueError("RedisMeta is not provided; ensure from_cache_config stores it or pass it to start().")
        #TODO can we use like this to distinguish the different tree pairs?
        if self.device_type == DeviceType.LAKE:
            local_ch_block_key = "PCFSB"
            remote_ch_block_key = "PCFSB"
        elif self.device_type == DeviceType.CPU:
            local_ch_block_key = "CPUB"
            remote_ch_block_key = "CPUB"
        elif self.device_type == DeviceType.SSD:
            local_ch_block_key = "SSDB"
            remote_ch_block_key = "SSDB"
        else:
            raise ValueError(f"Invalid device type: {self.device_type}")
        self.remote_ch = self._meta.get_redis_meta_channel(remote_ch_block_key)
        self.local_ch = self._meta.get_redis_meta_channel(local_ch_block_key)
                # Load and store mapping of node_id -> file_nodeids from Redis
        if self.device_type == DeviceType.LAKE:
            try:
                self.nid_to_file_nodeids = self._meta.load_pcfs_file_nodeids()
            except Exception:
                raise ValueError("Failed to load PCFS file nodeids from Redis")
        if not self.local_index.start(self.local_ch):
            raise RuntimeError(
                f"Failed to start local radix tree for device type: {self.device_type}"
            )
        if not self.remote_index.start(self.remote_ch):
            self.local_index.stop()
            raise RuntimeError(
                f"Failed to start distributed radix tree for device type: {self.device_type}"
            )

    def stop(self) -> None:
        self.local_index.stop()
        self.remote_index.stop()

    def reset(self) -> None:
        self.local_index.reset()
        self.mempool.reset()

    def match(self,
              sequence_meta: SequenceMeta,
              *,
              with_peer: bool = True,
              gpu_matched_blocks: int = 0) -> MatchResult:
        """Unified tier match.

        ``with_peer=False`` (PUT) queries only the local tree.  Otherwise both
        the local and the distributed (peer) trees are queried and returned as
        the two sides of a ``MatchResult``; each is an independent prefix hit,
        and the planner composes local-first / peer-suffix from them.
        """
        if not with_peer:
            return MatchResult(local=self.match_local(sequence_meta))

        sequence_meta.gen_hashes()
        block_hashes_t = torch.from_numpy(sequence_meta.block_hashes).to(torch.int64)
        num_blocks = sequence_meta.num_blocks

        mr_local = self.local_index.match_prefix(block_hashes_t, int(num_blocks), True)
        mr_remote = self.remote_index.match_prefix(block_hashes_t, int(num_blocks), True)

        local_ready = int(mr_local.num_ready_matched_blocks)
        remote_ready = int(mr_remote.num_ready_matched_blocks)
        local_matched = int(mr_local.num_matched_blocks)
        remote_matched = int(mr_remote.num_matched_blocks)

        local_phys = mr_local.physical_blocks.cpu().numpy().astype(np.int64, copy=False)
        remote_phys = mr_remote.physical_blocks.cpu().numpy().astype(np.int64, copy=False)

        # Lake blocks are read from concrete PCFS files; without a per-block file
        # node id a block cannot be transferred, so drop that side's readiness.
        local_source_ids = None
        if self.device_type == DeviceType.LAKE and local_ready > 0:
            local_source_ids = self.nodeids_to_file_nodeids(
                np.full(len(local_phys), self.redis_node_id, dtype=np.int64),
                local_phys,
            )
            if local_source_ids is None or len(local_source_ids) < local_ready:
                local_ready = 0

        # Resolve the peer (remote) origin. For LAKE the peer's blocks map to
        # per-block PCFS file ids. For cpu/ssd the distributed tree merges every
        # peer's blocks, so one remote match can span MULTIPLE peers — a shared
        # prefix and its extension may be owned by different nodes. A match
        # returns at most ONE peer to the planner, so keep only the single peer
        # that extends this tier's local prefix.
        remote_source: Optional[BlockSource] = None
        nids = mr_remote.block_node_ids
        if isinstance(nids, torch.Tensor) and nids.numel() > 0:
            nids_np = nids.cpu().numpy()
            if self.device_type == DeviceType.LAKE:
                file_ids = self.nodeids_to_file_nodeids(nids_np, remote_phys)
                if file_ids is not None:
                    remote_source = LakeSource(np.asarray(file_ids, dtype=np.int64))
            elif remote_ready > local_ready:
                # Blocks [0, local_ready) are served locally; the peer only
                # extends [local_ready, remote_ready). Anchor at local_ready and
                # take the contiguous run of the single peer owning it — the
                # maximal single-peer extension the planner can consume.
                peer_ids = nids_np.astype(np.int64, copy=False)
                owner = int(peer_ids[local_ready])
                end = local_ready
                while end < remote_ready and int(peer_ids[end]) == owner:
                    end += 1
                remote_ready = end
                remote_source = PeerSource(owner)
            else:
                # The peer reaches no further than the local prefix; ignore it.
                remote_ready = 0
        # Without a usable origin the peer blocks cannot be transferred; drop the
        # peer readiness so the planner ignores this side.
        if remote_ready > 0 and (
            remote_source is None or not remote_source.covers(remote_ready)
        ):
            remote_ready = 0
            remote_source = None

        # LOCAL side: a plain local prefix hit.  For Lake, its source carries the
        # per-block file node ids the planner needs.
        local = MatchResultAccel(
            num_ready_matched_blocks=local_ready,
            num_matched_blocks=local_matched,
            last_ready_node=mr_local.last_ready_node,
            last_node=mr_local.last_node,
            last_node_matched_length=int(mr_local.last_node_matched_length),
            physical_blocks=local_phys,
            source=(
                LakeSource(np.asarray(local_source_ids, dtype=np.int64))
                if (self.device_type == DeviceType.LAKE and local_ready > 0)
                else LocalSource()
            ),
        )

        # PEER side: only when the peer tree reaches a valid, transferable
        # prefix.  Its source is a single PeerSource (cpu/ssd) or a per-block
        # LakeSource (Lake file ids).
        remote = None
        if remote_ready > 0:
            assert remote_source is not None  # guaranteed by the drop above
            remote = MatchResultAccel(
                num_ready_matched_blocks=remote_ready,
                num_matched_blocks=remote_matched,
                last_ready_node=mr_remote.last_ready_node,
                last_node=mr_remote.last_node,
                last_node_matched_length=int(mr_remote.last_node_matched_length),
                physical_blocks=remote_phys,
                source=remote_source,
            )

        self._log_reuse_stats(
            num_blocks, gpu_matched_blocks, local_matched,
            max(local_matched, remote_matched),
        )
        return MatchResult(local=local, remote=remote)

    def _log_reuse_stats(self,
                         num_blocks: int,
                         gpu_matched_blocks: int,
                         local_matched: int,
                         distributed_matched: int) -> None:
        self._stats_total_queried_tokens += num_blocks * self.tokens_per_block
        self._stats_gpu_matched_tokens += gpu_matched_blocks * self.tokens_per_block
        self._stats_local_matched_tokens += local_matched * self.tokens_per_block
        self._stats_distributed_matched_tokens += distributed_matched * self.tokens_per_block
        self._stats_match_count += 1

        total = self._stats_total_queried_tokens
        gpu_pct = (self._stats_gpu_matched_tokens * 100 / total) if total > 0 else 0
        local_pct = (self._stats_local_matched_tokens * 100 / total) if total > 0 else 0
        distributed_pct = (self._stats_distributed_matched_tokens * 100 / total) if total > 0 else 0
        extra_from_local = self._stats_local_matched_tokens - self._stats_gpu_matched_tokens
        extra_from_distributed = self._stats_distributed_matched_tokens - self._stats_local_matched_tokens
        extra_local_pct = (extra_from_local * 100 / total) if total > 0 else 0
        extra_distributed_pct = (extra_from_distributed * 100 / total) if total > 0 else 0

        print(
            f"[STATS][REUSE] cnt={self._stats_match_count}, queried={total}, "
            f"gpu={self._stats_gpu_matched_tokens} ({gpu_pct:.2f}%), "
            f"flexkv_local={self._stats_local_matched_tokens} ({local_pct:.2f}%, +{extra_local_pct:.2f}%), "
            f"flexkv_global={self._stats_distributed_matched_tokens} ({distributed_pct:.2f}%, "
            f"+{extra_distributed_pct:.2f}%)"
        )

    def nodeids_to_file_nodeids(self,
                                 bnids_np: np.ndarray,
                                 phys: np.ndarray) -> Optional[np.ndarray]:
        """Convert per-block node ids to per-block PCFS file_nodeids.

        Args:
            bnids_np: per-block owning node ids from the radix match
                (the distributed match's per-block node ids, or this node's own
                id broadcast for a local Lake hit)
            phys: physical_blocks from the match

        Returns:
            file_nodeids array with dtype=uint32, or None if conversion fails
        """
        if bnids_np is None or phys is None:
            return None
        try:
            bnids_np = np.asarray(bnids_np, dtype=np.uint32)
            phys_np = np.asarray(phys, dtype=np.int64)
        except Exception:
            return None
        if bnids_np.shape[0] != phys_np.shape[0]:
            return None
        out = np.full(phys_np.shape, fill_value=0, dtype=np.uint32)
        rr = max(1, int(self.round_robin))
        
        for i in range(bnids_np.shape[0]):
            nid = int(bnids_np[i])
            #check if node is active
            is_active = self._meta.is_node_active(nid)
            if not is_active:
                print(f"[DEBUG] Node {nid} is not active, returning None")
                return None
            file_list = self.nid_to_file_nodeids.get(nid)
            #check if file list is empty
            if not file_list:
                return None
            lake_file_num = len(file_list)
            if lake_file_num <= 0:
                return None
            block_id = int(phys_np[i])
            f_idx = (block_id // rr) % lake_file_num
            out[i] = np.uint32(file_list[f_idx])
        return out
    #match local will only be called for put
    def match_local(self, sequence_meta: SequenceMeta) -> MatchResultAccel:
        sequence_meta.gen_hashes()
        block_hashes_t = torch.from_numpy(sequence_meta.block_hashes).to(torch.int64)
        num_blocks = sequence_meta.num_blocks

        mr_local = self.local_index.match_prefix(block_hashes_t, int(num_blocks), True)

        phys_np = mr_local.physical_blocks.cpu().numpy()

        return MatchResultAccel(
            num_ready_matched_blocks=int(mr_local.num_ready_matched_blocks),
            num_matched_blocks=int(mr_local.num_matched_blocks),
            last_ready_node=mr_local.last_ready_node,
            last_node=mr_local.last_node,
            last_node_matched_length=int(mr_local.last_node_matched_length),
            physical_blocks=phys_np,
        )

    def insert(self,
               sequence_meta: SequenceMeta,
               physical_block_ids: Union[np.ndarray, torch.Tensor],
               num_insert_blocks: int = -1,
               is_ready: bool = True,
               match_result: Optional[MatchResultAccel] = None) -> Tuple[Optional[CRadixNode], np.ndarray]:
        sequence_meta.gen_hashes()
        phys_t = torch.from_numpy(physical_block_ids).to(torch.int64) if isinstance(physical_block_ids, np.ndarray) else physical_block_ids.to(torch.int64)
        hashes_t = torch.from_numpy(sequence_meta.block_hashes).to(torch.int64)
        
        if match_result is None:
            node = self.local_index.insert(
                phys_t, hashes_t, int(sequence_meta.num_blocks), int(num_insert_blocks), bool(is_ready)
            )
        else:
            # match_result carries a RadixNodeLike; this engine's tree is a
            # LocalRadixTree keyed on CRadixNode. A hie match always yields
            parent = match_result.last_node
            assert parent is None or isinstance(parent, CRadixNode)
            node = self.local_index.insert(
                phys_t, hashes_t, int(sequence_meta.num_blocks), int(num_insert_blocks), bool(is_ready),
                parent,
                int(match_result.num_matched_blocks), int(match_result.last_node_matched_length)
            )
        # NOTE: Do NOT lock the node here, because the caller (put() method) will lock it
        # The node will be unlocked in _transfer_callback after data transfer completes
        # Return (node, unused_slots) for API parity with the other engines; the
        # in-process index has no cross-process race, so unused_slots is empty.
        return node, np.array([], dtype=np.int64)

    def lock_node(self, node: CRadixNode) -> None:
        if node is None:
            return
        try:
            is_remote_node = bool(node.has_block_node_ids())
        except Exception:
            is_remote_node = False
        if is_remote_node:
            self.remote_index.lock(node) #TODO why do we need to lock the remote node?
        else:
            self.local_index.lock(node)

    def unlock(self, node: CRadixNode) -> None:
        """Unlock a node in the appropriate index (local or remote).
        
        Args:
            node: The radix node to unlock
        """
        if node is None:
            return
        try:
            is_remote_node = bool(node.has_block_node_ids())
        except Exception:
            is_remote_node = False
        if is_remote_node:
            self.remote_index.unlock(node)
        else:
            self.local_index.unlock(node)

    def cleanup(self, node: CRadixNode, cleanup_length: int) -> None:
        if node is None:
            return
        try:
            is_remote_node = bool(node.has_block_node_ids())
        except Exception:
            is_remote_node = False
        if is_remote_node:
            self.remote_index.unlock(node)
            #self.remote_index.set_ready(node, True, cleanup_length)
        else:
            self.local_index.unlock(node)
            #self.local_index.set_ready(node, True, cleanup_length)

    def set_ready(self, node: CRadixNode, ready: bool = True, ready_length: int = -1) -> None:
        """Set the ready state of a node in the appropriate index (local or remote).
        
        Args:
            node: The radix node to set ready state
            ready: Whether the node is ready (default: True)
            ready_length: The ready length (default: -1, meaning use node's current length)
        """
        if node is None:
            return
        try:
            is_remote_node = bool(node.has_block_node_ids())
        except Exception:
            is_remote_node = False
        if is_remote_node:
            self.remote_index.set_ready(node, ready, ready_length)
        else:
            self.local_index.set_ready(node, ready, ready_length)

    def take(self,
             num_required_blocks: int,
             protected_node: Optional[CRadixNode] = None,
             strict: bool = True) -> ndarray:
        # Calculate current utilization
        utilization = (self.mempool.num_total_blocks - self.mempool.num_free_blocks) / self.mempool.num_total_blocks if self.mempool.num_total_blocks > 0 else 0
        
        # Proactive eviction: trigger when utilization exceeds threshold OR when blocks are needed
        should_evict = (utilization >= self.evict_start_threshold) or (num_required_blocks > self.mempool.num_free_blocks)
        
        if should_evict:
            if protected_node is not None:
                self.local_index.lock(protected_node)
            
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
                num_evicted = self.local_index.evict(target_blocks, evict_block_num)
                if num_evicted != evict_block_num:
                    target_blocks.resize_(num_evicted)
                evicted_np = target_blocks.numpy()
                self.mempool.recycle_blocks(evicted_np)
            
            if protected_node is not None:
                self.local_index.unlock(protected_node)
        
        if strict and num_required_blocks > self.mempool.num_free_blocks:
            raise ValueError(
                f"Not enough free blocks to take, required: {num_required_blocks}, available: {self.mempool.num_free_blocks}"
            )
        num_allocated_blocks = min(num_required_blocks, self.mempool.num_free_blocks)
        #print(f"[TAKE STATISTICS] device type: {self.device_type.name}, utilization: {utilization}, ",
        #      f"should_evict: {should_evict}, num_required_blocks: {num_required_blocks}, ",
        #      f"num_allocated_blocks: {num_allocated_blocks}, num_free_blocks: {self.mempool.num_free_blocks}")
        return self.mempool.allocate_blocks(num_allocated_blocks)

    def recycle(self, physical_blocks: np.ndarray) -> None:
        self.mempool.recycle_blocks(physical_blocks)

    #TODO pfcs may not work now
    @classmethod
    def pcfs_ce_from_cache_config(cls, cache_config: "CacheConfig", node_id: int, meta: Optional[RedisMeta] = None) -> "HierarchyLRCacheEngine":
        """Create a PCFSCacheEngine from CacheConfig.

        This replaces LakePCFSCacheEngine. It wires both local and remote
        radix trees using parameters from CacheConfig and the provided node_id.
        """
        num_blocks = int(cache_config.num_lake_blocks or 0)

        # 1) Generate unique lake_file_prefix using uuid and build lake_cache_path
        if cache_config.lake_file_prefix is None:
            raise ValueError("lake_file_prefix must be provided in CacheConfig when enable_lake is True")
        if cache_config.lake_file_num is None or cache_config.lake_file_num <= 0:
            raise ValueError("lake_file_num must be a positive integer in CacheConfig when enable_lake is True")

        # Prefer uuid from RedisMeta to ensure cluster-wide uniqueness, fallback to Python uuid if meta is None
        try:
            unique_suffix = meta.get_uuid() if meta is not None else __import__("uuid").uuid4().hex
        except Exception:
            unique_suffix = __import__("uuid").uuid4().hex

        new_prefix = f"{cache_config.lake_file_prefix}_{unique_suffix}"
        cache_config.lake_file_prefix = new_prefix
        cache_config.lake_cache_path = [
            f"{cache_config.lake_file_prefix}_{i}" for i in range(cache_config.lake_file_num)
        ]

        # 2) Create PCFS instance and lookup/create files to collect nodeids
        lake_cfg = cache_config.lake_config_custom or {}
        pcfs_fsid = lake_cfg.get("pcfs_fsid")
        pcfs_port = lake_cfg.get("pcfs_port")
        pcfs_ip = lake_cfg.get("pcfs_ip")
        pcfs_parent_nodeid = lake_cfg.get("pcfs_parent_nodeid")
        if None in (pcfs_fsid, pcfs_port, pcfs_ip, pcfs_parent_nodeid):
            raise ValueError("Some required PCFS config fields are missing: pcfs_fsid, pcfs_port, pcfs_ip, pcfs_parent_nodeid")

        pcfs = c_ext.Pcfs(pcfs_fsid, pcfs_port, pcfs_ip, False, pcfs_parent_nodeid)
        if not pcfs.init():
            raise ValueError(f"PCFS init failed: fsid={pcfs_fsid}, ip={pcfs_ip}")

        node_ids: List[int] = []
        # Derive file size if available; otherwise, use 0 when not provided (only lookup or create placeholder)
        # Prefer explicit file_size mode
        file_size = 0
        if getattr(cache_config, "lake_cache_size_mode", "file_size") == "file_size":
            file_size = int(cache_config.lake_file_size or 0)

        for lake_path in cache_config.lake_cache_path:
            nodeid = pcfs.lookup_or_create_file(lake_path, file_size, True)
            if nodeid == 0:
                raise ValueError(f"lookup or create file failed for file: {lake_path}")
            node_ids.append(int(nodeid))

        # 3) Register nodeids into Redis for discovery
        if meta is not None:
            meta.add_node_ids(node_ids)

        # Set global pcfs instance for subsequent C++ lake transfers
        try:
            c_ext.set_pcfs_instance(pcfs)
        except Exception:
            pass

        return cls(
            num_total_blocks=num_blocks,
            tokens_per_block=int(cache_config.tokens_per_block),
            evict_ratio=float(cache_config.evict_ratio),
            device_type=DeviceType.LAKE,
            local_lease_ttl_ms=int(GLOBAL_CONFIG_FROM_ENV.lease_ttl_ms),
            local_renew_lease_ms=int(GLOBAL_CONFIG_FROM_ENV.renew_lease_ms),
            local_refresh_batch_size=int(GLOBAL_CONFIG_FROM_ENV.refresh_batch_size),
            local_idle_sleep_ms=int(GLOBAL_CONFIG_FROM_ENV.idle_sleep_ms),
            remote_max_num_blocks=num_blocks,
            redis_node_id=int(node_id),
            remote_refresh_batch_size=int(GLOBAL_CONFIG_FROM_ENV.refresh_batch_size),
            remote_rebuild_interval_ms=int(GLOBAL_CONFIG_FROM_ENV.rebuild_interval_ms),
            remote_idle_sleep_ms=int(GLOBAL_CONFIG_FROM_ENV.idle_sleep_ms),
            local_safety_ttl_ms=int(GLOBAL_CONFIG_FROM_ENV.safety_ttl_ms),
            eviction_policy=GLOBAL_CONFIG_FROM_ENV.eviction_policy,
            meta=meta,
        )

    #TODO is this enough for peercpu and peerssd?
    @classmethod
    def from_cache_config(cls, cache_config: "CacheConfig", node_id: int, device_type: DeviceType, meta: Optional[RedisMeta] = None) -> "HierarchyLRCacheEngine":

        if device_type == DeviceType.LAKE:
            return cls.pcfs_ce_from_cache_config(cache_config, node_id, meta)
        else:
            # select correct blocks configuration based on device_type
            if device_type == DeviceType.CPU:
                local_max_num_blocks = int(cache_config.num_cpu_blocks)
            elif device_type == DeviceType.SSD:
                local_max_num_blocks = int(cache_config.num_ssd_blocks)
            else:
                raise ValueError(f"Invalid device type: {device_type}")
                #local_max_num_blocks = int(cache_config.num_local_blocks or 0)
            
            return cls(
                num_total_blocks=int(local_max_num_blocks or 0),
                tokens_per_block=int(cache_config.tokens_per_block),
                evict_ratio=float(GLOBAL_CONFIG_FROM_ENV.evict_ratio),
                device_type=device_type,
                local_max_num_blocks=local_max_num_blocks,
                local_lease_ttl_ms=int(GLOBAL_CONFIG_FROM_ENV.lease_ttl_ms),
                local_renew_lease_ms=int(GLOBAL_CONFIG_FROM_ENV.renew_lease_ms),
                local_refresh_batch_size=int(GLOBAL_CONFIG_FROM_ENV.refresh_batch_size),
                local_idle_sleep_ms=int(GLOBAL_CONFIG_FROM_ENV.idle_sleep_ms),
                # local_lt_pool_initial_capacity=int(getattr(cache_config, "lt_pool_initial_capacity", 0)),
                remote_max_num_blocks=int(cache_config.num_lake_blocks or 0),
                redis_node_id=int(node_id),
                # remote_node_id=int(node_id),
                # remote_lt_pool_initial_capacity=int(getattr(cache_config, "lt_pool_initial_capacity", 0)),
                remote_refresh_batch_size=int(GLOBAL_CONFIG_FROM_ENV.refresh_batch_size),
                remote_rebuild_interval_ms=int(GLOBAL_CONFIG_FROM_ENV.rebuild_interval_ms),
                remote_idle_sleep_ms=int(GLOBAL_CONFIG_FROM_ENV.idle_sleep_ms),
                local_safety_ttl_ms=int(GLOBAL_CONFIG_FROM_ENV.safety_ttl_ms),
                evict_start_threshold=float(GLOBAL_CONFIG_FROM_ENV.evict_start_threshold),
                hit_reward_seconds=int(GLOBAL_CONFIG_FROM_ENV.hit_reward_seconds),
                eviction_policy=GLOBAL_CONFIG_FROM_ENV.eviction_policy,
                meta=meta,
            )
            raise ValueError("Invalid device type: {cache_config.device_type}")
