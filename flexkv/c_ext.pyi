from __future__ import annotations
import collections.abc
import numpy
import torch
import typing
import typing_extensions
__all__: list[str] = ['BlockMeta', 'CMatchResult', 'CRadixNode', 'CRadixTreeIndex', 'DistributedRadixTree', 'GDSManager', 'Hasher', 'IntQueue', 'LocalRadixTree', 'RedisMetaChannel', 'RefRadixTree', 'SSDIOCTX', 'TPGDSTransferThreadGroup', 'TPTransferThreadGroup', 'configure_cpp_metrics', 'gen_hashes', 'gen_hashes_numpy', 'get_hash_size', 'transfer_kv_blocks', 'transfer_kv_blocks_gds', 'transfer_kv_blocks_ssd']
class BlockMeta:
    def __init__(self) -> None:
        ...
    @property
    def hash(self) -> int:
        ...
    @hash.setter
    def hash(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def lt(self) -> int:
        ...
    @lt.setter
    def lt(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def nid(self) -> int:
        ...
    @nid.setter
    def nid(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def pb(self) -> int:
        ...
    @pb.setter
    def pb(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def ph(self) -> int:
        ...
    @ph.setter
    def ph(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def state(self) -> int:
        ...
    @state.setter
    def state(self, arg0: typing.SupportsInt) -> None:
        ...
class CMatchResult:
    def __init__(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt, arg2: typing.SupportsInt, arg3: CRadixNode, arg4: CRadixNode, arg5: torch.Tensor, arg6: torch.Tensor) -> None:
        ...
    @property
    def block_node_ids(self) -> torch.Tensor:
        ...
    @property
    def last_node(self) -> CRadixNode:
        ...
    @property
    def last_node_matched_length(self) -> int:
        ...
    @property
    def last_ready_node(self) -> CRadixNode:
        ...
    @property
    def num_matched_blocks(self) -> int:
        ...
    @property
    def num_ready_matched_blocks(self) -> int:
        ...
    @property
    def physical_blocks(self) -> torch.Tensor:
        ...
class CRadixNode:
    @typing.overload
    def __init__(self, arg0: CRadixTreeIndex, arg1: bool, arg2: typing.SupportsInt) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: CRadixTreeIndex, arg1: bool, arg2: typing.SupportsInt, arg3: bool) -> None:
        ...
    def has_block_node_ids(self) -> bool:
        ...
    def size(self) -> int:
        ...
    @property
    def parent(self) -> CRadixNode:
        ...
class CRadixTreeIndex:
    def __init__(self, tokens_per_block: typing.SupportsInt, max_num_blocks: typing.SupportsInt = 1000000, hit_reward_seconds: typing.SupportsInt = 0, eviction_policy: str = 'lru', protected_threshold: typing.SupportsInt = 2) -> None:
        ...
    @typing.overload
    def evict(self, evicted_blocks: torch.Tensor, num_evicted: typing.SupportsInt) -> int:
        ...
    @typing.overload
    def evict(self, evicted_blocks: torch.Tensor, evicted_block_hashes: torch.Tensor, num_evicted: typing.SupportsInt) -> int:
        ...
    def insert(self, physical_block_ids: torch.Tensor, block_hashes: torch.Tensor, num_blocks: typing.SupportsInt, num_insert_blocks: typing.SupportsInt, ready: bool = True, node: ... = None, num_matched_blocks: typing.SupportsInt = -1, last_node_matched_length: typing.SupportsInt = -1) -> ...:
        ...
    def is_empty(self) -> bool:
        ...
    def lock(self, node: ...) -> None:
        ...
    def match_prefix(self, block_hashes: torch.Tensor, num_blocks: typing.SupportsInt, update_cache_info: bool) -> ...:
        ...
    def reset(self) -> None:
        ...
    def set_ready(self, node: ..., ready: bool, ready_length: typing.SupportsInt) -> None:
        ...
    def total_cached_blocks(self) -> int:
        ...
    def total_ready_blocks(self) -> int:
        ...
    def total_unready_blocks(self) -> int:
        ...
    def unlock(self, node: ...) -> None:
        ...
class DistributedRadixTree:
    def __init__(self, tokens_per_block: typing.SupportsInt, max_num_blocks: typing.SupportsInt, node_id: typing.SupportsInt, refresh_batch_size: typing.SupportsInt = 128, rebuild_interval_ms: typing.SupportsInt = 1000, idle_sleep_ms: typing.SupportsInt = 10, lease_renew_ms: typing.SupportsInt = 5000, hit_reward_seconds: typing.SupportsInt = 0) -> None:
        ...
    def is_empty(self) -> bool:
        ...
    def lock(self, node: CRadixNode) -> None:
        ...
    def match_prefix(self, block_hashes: torch.Tensor, num_blocks: typing.SupportsInt, update_cache_info: bool = True) -> CMatchResult:
        ...
    def remote_tree_refresh(self) -> ...:
        ...
    def set_ready(self, node: CRadixNode, ready: bool = True, ready_length: typing.SupportsInt = -1) -> None:
        ...
    def start(self, channel: RedisMetaChannel) -> bool:
        ...
    def stop(self) -> None:
        ...
    def unlock(self, node: CRadixNode) -> None:
        ...
class GDSManager:
    def __init__(self, ssd_files: collections.abc.Mapping[typing.SupportsInt, collections.abc.Sequence[str]], num_devices: typing.SupportsInt, round_robin: typing.SupportsInt = 1) -> None:
        """
        Initialize GDS Manager with device-organized files
        """
    def add_file(self, filename: str) -> bool:
        """
        Add and register a file with GDS (creates with O_DIRECT)
        """
    def batch_read(self, operations: list) -> list:
        """
        Batch read operations
        """
    def batch_synchronize(self, batch_id: typing.SupportsInt) -> int:
        """
        Wait for batch operations to complete
        """
    def batch_write(self, operations: list) -> list:
        """
        Batch write operations
        """
    def create_gds_file(self, filename: str, file_size: typing.SupportsInt) -> bool:
        """
        Create and register a GDS file with specified size
        """
    def get_file_count(self) -> int:
        """
        Get number of files currently managed
        """
    def get_file_paths(self, device_id: typing.SupportsInt) -> list[str]:
        """
        Get file paths for a specific device
        """
    def get_last_error(self) -> str:
        """
        Get the last error message
        """
    def get_num_devices(self) -> int:
        """
        Get number of devices
        """
    def get_num_files_per_device(self) -> int:
        """
        Get number of files per device
        """
    def get_round_robin(self) -> int:
        """
        Get round-robin granularity
        """
    def is_ready(self) -> bool:
        """
        Check if GDS manager is ready for operations
        """
    def read(self, filename: str, gpu_buffer: torch.Tensor, file_offset: typing.SupportsInt = 0) -> int:
        """
        Read data from file to GPU memory
        """
    def read_async(self, filename: str, gpu_buffer: torch.Tensor, file_offset: typing.SupportsInt = 0) -> int:
        """
        Read data from file to GPU memory asynchronously
        """
    def remove_file(self, filename: str) -> bool:
        """
        Remove and unregister a file from GDS
        """
    def synchronize(self) -> None:
        """
        Synchronize all internal CUDA streams
        """
    def write(self, filename: str, gpu_data: torch.Tensor, file_offset: typing.SupportsInt = 0) -> int:
        """
        Write data from GPU memory to file
        """
    def write_async(self, filename: str, gpu_data: torch.Tensor, file_offset: typing.SupportsInt = 0) -> int:
        """
        Write data from GPU memory to file asynchronously
        """
class Hasher:
    def __init__(self) -> None:
        ...
    def digest(self) -> int:
        """
        Return the hash value
        """
    def reset(self) -> None:
        ...
    @typing.overload
    def update(self, input: torch.Tensor) -> Hasher:
        """
        Update the hasher with a tensor
        """
    @typing.overload
    def update(self, input: typing_extensions.CapsuleType, size: typing.SupportsInt) -> Hasher:
        """
        Update the hasher with pointer and size
        """
    def update_numpy(self, input: numpy.ndarray) -> None:
        """
        Update the hasher directly from a numpy array buffer
        """
class IntQueue:
    def __init__(self) -> None:
        ...
    def pop(self) -> tuple:
        ...
    def push(self, value: typing.SupportsInt) -> None:
        ...
class LocalRadixTree(CRadixTreeIndex):
    def __init__(self, tokens_per_block: typing.SupportsInt, max_num_blocks: typing.SupportsInt = 1000000, lease_ttl_ms: typing.SupportsInt = 100000, renew_lease_ms: typing.SupportsInt = 0, refresh_batch_size: typing.SupportsInt = 256, idle_sleep_ms: typing.SupportsInt = 10, safety_ttl_ms: typing.SupportsInt = 100, swap_block_threshold: typing.SupportsInt = 1024, hit_reward_seconds: typing.SupportsInt = 0, eviction_policy: str = 'lru', protected_threshold: typing.SupportsInt = 2) -> None:
        ...
    def add_leaf(self, node: CRadixNode) -> None:
        ...
    def add_node(self, node: CRadixNode) -> None:
        ...
    def dec_node_count(self) -> None:
        ...
    def drain_pending_queues(self) -> int:
        ...
    @typing.overload
    def evict(self, evicted_blocks: torch.Tensor, num_evicted: typing.SupportsInt) -> int:
        ...
    @typing.overload
    def evict(self, evicted_blocks: torch.Tensor, evicted_block_hashes: torch.Tensor, num_evicted: typing.SupportsInt) -> int:
        ...
    def inc_node_count(self) -> None:
        ...
    def insert(self, physical_block_ids: torch.Tensor, block_hashes: torch.Tensor, num_blocks: typing.SupportsInt, num_insert_blocks: typing.SupportsInt, ready: bool = True, node: typing.Optional[CRadixNode] = None, num_matched_blocks: typing.SupportsInt = -1, last_node_matched_length: typing.SupportsInt = -1) -> CRadixNode:
        ...
    def insert_and_publish(self, node: CRadixNode) -> bool:
        ...
    def is_empty(self) -> bool:
        ...
    def is_root(self, node: CRadixNode) -> bool:
        ...
    def lock(self, node: CRadixNode) -> None:
        ...
    def match_prefix(self, block_hashes: torch.Tensor, num_blocks: typing.SupportsInt, update_cache_info: bool = True) -> CMatchResult:
        ...
    def remove_leaf(self, node: CRadixNode) -> None:
        ...
    def remove_node(self, node: CRadixNode) -> None:
        ...
    def reset(self) -> None:
        ...
    def set_meta_channel(self, channel: RedisMetaChannel) -> None:
        ...
    def set_ready(self, node: CRadixNode, ready: bool, ready_length: typing.SupportsInt = -1) -> None:
        ...
    def start(self, channel: RedisMetaChannel) -> bool:
        ...
    def stop(self) -> None:
        ...
    def total_cached_blocks(self) -> int:
        ...
    def total_node_num(self) -> int:
        ...
    def total_ready_blocks(self) -> int:
        ...
    def total_unready_blocks(self) -> int:
        ...
    def unlock(self, node: CRadixNode) -> None:
        ...
class RedisMetaChannel:
    def __init__(self, host: str, port: typing.SupportsInt, node_id: typing.SupportsInt, local_ip: str, blocks_key: str = 'blocks', password: str = '') -> None:
        ...
    def connect(self) -> bool:
        ...
    def delete_blockmeta_batch(self, node_id: typing.SupportsInt, hashes: collections.abc.Sequence[typing.SupportsInt], batch_size: typing.SupportsInt = 200) -> bool:
        ...
    def get_local_ip(self) -> str:
        ...
    def get_node_id(self) -> int:
        ...
    def hmget_field_for_keys(self, keys: collections.abc.Sequence[str], field: str) -> list[str]:
        ...
    def hmget_two_fields_for_keys(self, keys: collections.abc.Sequence[str], field1: str, field2: str) -> list[tuple[str, str]]:
        ...
    def list_block_keys(self, node_id: typing.SupportsInt) -> list[str]:
        ...
    def list_keys(self, pattern: str) -> list[str]:
        ...
    def list_node_keys(self) -> list[str]:
        ...
    def load(self, max_items: typing.SupportsInt) -> list[BlockMeta]:
        ...
    def load_metas_by_keys(self, keys: collections.abc.Sequence[str]) -> list[BlockMeta]:
        ...
    def make_block_key(self, node_id: typing.SupportsInt, hash: typing.SupportsInt) -> str:
        ...
    def publish_batch(self, metas: collections.abc.Sequence[BlockMeta], batch_size: typing.SupportsInt = 100) -> bool:
        ...
    def publish_one(self, arg0: BlockMeta) -> bool:
        ...
    def renew_node_leases(self, node_id: typing.SupportsInt, new_lt: typing.SupportsInt, batch_size: typing.SupportsInt = 200) -> bool:
        ...
    def renew_node_leases_with_hashes(self, node_id: typing.SupportsInt, new_lt: typing.SupportsInt, hashes: collections.abc.Sequence[typing.SupportsInt], batch_size: typing.SupportsInt = 200) -> bool:
        ...
    def update_block_state_batch(self, node_id: typing.SupportsInt, hashes: collections.abc.Sequence[typing.SupportsInt], state: typing.SupportsInt, batch_size: typing.SupportsInt = 200) -> bool:
        ...
class RefRadixTree(CRadixTreeIndex):
    def __init__(self, tokens_per_block: typing.SupportsInt, max_num_blocks: typing.SupportsInt = 1000000, lease_renew_ms: typing.SupportsInt = 5000, hit_reward_seconds: typing.SupportsInt = 0, renew_lease_queue: ... = None, lt_pool: ... = None, generation: typing.SupportsInt = 0) -> None:
        ...
    def dec_ref_cnt(self) -> None:
        ...
    def get_generation(self) -> int:
        ...
    def inc_ref_cnt(self) -> None:
        ...
class SSDIOCTX:
    def __init__(self, arg0: collections.abc.Mapping[typing.SupportsInt, collections.abc.Sequence[str]], arg1: typing.SupportsInt, arg2: typing.SupportsInt, arg3: typing.SupportsInt) -> None:
        ...
class TPGDSTransferThreadGroup:
    def __init__(self, num_gpus: typing.SupportsInt, gpu_block_ptrs_flat: collections.abc.Sequence[typing.SupportsInt], num_tensors_per_gpu: typing.SupportsInt, ssd_files: collections.abc.Mapping[typing.SupportsInt, collections.abc.Sequence[str]], dp_group_id: typing.SupportsInt, num_layers: typing.SupportsInt, gpu_kv_strides_in_bytes: collections.abc.Sequence[typing.SupportsInt], gpu_block_strides_in_bytes: collections.abc.Sequence[typing.SupportsInt], gpu_layer_strides_in_bytes: collections.abc.Sequence[typing.SupportsInt], gpu_chunk_sizes_in_bytes: collections.abc.Sequence[typing.SupportsInt], gpu_device_ids: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    def tp_group_transfer(self, gpu_block_id_tensor: torch.Tensor, ssd_block_id_tensor: torch.Tensor, ssd_layer_stride_in_bytes: typing.SupportsInt, ssd_kv_stride_in_bytes: typing.SupportsInt, ssd_block_stride_in_bytes: typing.SupportsInt, ssd_tp_stride_in_bytes: typing.SupportsInt, num_blocks_per_file: typing.SupportsInt, is_read: bool, layer_id: typing.SupportsInt, layer_granularity: typing.SupportsInt, is_mla: bool) -> None:
        ...
class TPTransferThreadGroup:
    def __init__(self, num_gpus: typing.SupportsInt, gpu_block_ptrs_flat: collections.abc.Sequence[typing.SupportsInt], num_tensors_per_gpu: typing.SupportsInt, cpu_blocks_ptr: typing.SupportsInt, dp_group_id: typing.SupportsInt, num_layers: typing.SupportsInt, gpu_kv_strides_in_bytes: collections.abc.Sequence[typing.SupportsInt], gpu_block_strides_in_bytes: collections.abc.Sequence[typing.SupportsInt], gpu_layer_strides_in_bytes: collections.abc.Sequence[typing.SupportsInt], gpu_chunk_sizes_in_bytes: collections.abc.Sequence[typing.SupportsInt], gpu_device_ids: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    def tp_group_transfer(self, gpu_block_id_tensor: torch.Tensor, cpu_block_id_tensor: torch.Tensor, cpu_kv_stride_in_bytes: typing.SupportsInt, cpu_layer_stride_in_bytes: typing.SupportsInt, cpu_block_stride_in_bytes: typing.SupportsInt, cpu_tp_stride_in_bytes: typing.SupportsInt, transfer_num_cta: typing.SupportsInt, is_host_to_device: bool, use_ce_transfer: bool, layer_id: typing.SupportsInt, layer_granularity: typing.SupportsInt, is_mla: bool) -> None:
        ...
def configure_cpp_metrics(enabled: bool, port: typing.SupportsInt) -> None:
    """
    Configure C++ metrics from Python
    """
def gen_hashes(hasher: ..., token_ids: torch.Tensor, tokens_per_block: typing.SupportsInt, block_hashes: torch.Tensor) -> None:
    """
    Generate hashes for a tensor
    """
def gen_hashes_numpy(hasher: ..., token_ids: numpy.ndarray, tokens_per_block: typing.SupportsInt, block_hashes: numpy.ndarray) -> None:
    """
    Generate block hashes directly from numpy buffers
    """
def get_hash_size() -> int:
    """
    Get the size of the hash result
    """
def transfer_kv_blocks(gpu_block_id_tensor: torch.Tensor, gpu_tensor_ptrs_tensor: torch.Tensor, gpu_kv_stride_in_bytes: typing.SupportsInt, gpu_block_stride_in_bytes: typing.SupportsInt, gpu_layer_stride_in_bytes: typing.SupportsInt, cpu_block_id_tensor: torch.Tensor, cpu_tensor: torch.Tensor, cpu_kv_stride_in_bytes: typing.SupportsInt, cpu_layer_stride_in_bytes: typing.SupportsInt, cpu_block_stride_in_bytes: typing.SupportsInt, chunk_size_in_bytes: typing.SupportsInt, start_layer_id: typing.SupportsInt, num_layers: typing.SupportsInt, transfer_num_cta: typing.SupportsInt = 4, is_host_to_device: bool = True, use_ce_transfer: bool = False, is_mla: bool = False, gpu_block_type: typing.SupportsInt = 0) -> None:
    """
    Transfer multi-layer KV-cache between CPU and GPU
    """
def transfer_kv_blocks_gds(gds_manager: GDSManager, gpu_layer_id_list: torch.Tensor, gpu_layer_ptrs_tensor: torch.Tensor, ssd_block_ids: torch.Tensor, gpu_block_ids: torch.Tensor, gpu_kv_stride_in_bytes: typing.SupportsInt, gpu_block_stride_in_bytes: typing.SupportsInt, gpu_layer_stride_in_bytes: typing.SupportsInt, ssd_layer_stride_in_bytes: typing.SupportsInt, ssd_block_stride_in_bytes: typing.SupportsInt, ssd_kv_stride_in_bytes: typing.SupportsInt, block_size_in_bytes: typing.SupportsInt, ssd_copy_off_inside_chunks: typing.SupportsInt, num_blocks_per_file: typing.SupportsInt, total_layers: typing.SupportsInt, is_read: bool, verbose: bool = False, is_mla: bool = False, gpu_block_type: typing.SupportsInt = 0, gpu_device_id: typing.SupportsInt = 0) -> None:
    """
    Transfer KV blocks between GPU and GDS storage
    """
def transfer_kv_blocks_ssd(ioctx: ..., cpu_layer_id_list: torch.Tensor, cpu_tensor_ptr: typing.SupportsInt, ssd_block_ids: torch.Tensor, cpu_block_ids: torch.Tensor, cpu_layer_stride_in_bytes: typing.SupportsInt, cpu_kv_stride_in_bytes: typing.SupportsInt, ssd_layer_stride_in_bytes: typing.SupportsInt, ssd_kv_stride_in_bytes: typing.SupportsInt, chunk_size_in_bytes: typing.SupportsInt, block_stride_in_bytes: typing.SupportsInt, is_read: bool, num_blocks_per_file: typing.SupportsInt, round_robin: typing.SupportsInt = 1, num_threads_per_device: typing.SupportsInt = 16, is_mla: bool = False) -> None:
    """
    Transfer KV blocks between SSD and CPU memory
    """
