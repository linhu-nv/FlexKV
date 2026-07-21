from typing import List, Dict, Optional, Any


def split_contiguous_blocks(
    src_list: List[int], dst_list: List[int]
)-> List[Dict[str, List[int]]]:
    if not src_list:
        return []

    result = []
    current_src = [src_list[0]]
    current_dst = [dst_list[0]]

    for i in range(1, len(src_list)):
        src_cont = src_list[i] == src_list[i - 1] + 1
        dst_cont = dst_list[i] == dst_list[i - 1] + 1

        if src_cont and dst_cont:
            current_src.append(src_list[i])
            current_dst.append(dst_list[i])
        else:
            result.append({"src": current_src, "dst": current_dst})
            current_src = [src_list[i]]
            current_dst = [dst_list[i]]

    result.append({"src": current_src, "dst": current_dst})
    return result


class RemoteSSD2HMetaInfo:
    """Metadata for a remote-SSD read, sent from the requester to the SSD owner.

    Despite the "2H" name it serves both peer-SSD paths:
      * PEERSSD2H — owner stages SSD blocks to host and RDMA-writes them back to
        the requester's CPU buffer (the ``gpu_*`` fields stay None).
      * PEERSSD2D — owner stages then RDMA-writes token/head slices straight into
        the requester's GPU buffers (the ``gpu_*`` fields describe those slices).
    The ``gpu_*`` lists are an index-aligned struct-of-arrays; when
    ``gpu_dst_ptrs`` is set they must all have the same length.
    """
    task_id: int
    cpu_block_ids: List[int]
    ssd_block_ids: List[int]
    peer_engine_addr: str  # mooncake engine addr of the node that initiates the transfer,
                           # used for write data back to the node.
    peer_cpu_base_ptr: int # the cpu buffer base ptr of the peer node, used for calculating the dst ptrs.
    peer_zmq_status_addr: str
    data_size: int
    layer_id: int
    layer_granularity: int

    def __init__(
        self, task_id, cpu_block_ids, ssd_block_ids, peer_engine_addr,
        peer_cpu_base_ptr, peer_zmq_status_addr, data_size, layer_id,
        layer_granularity, gpu_dst_ptrs=None, gpu_block_positions=None,
        gpu_layer_ids=None, gpu_kv_ids=None, gpu_token_ids=None,
        gpu_head_starts=None, gpu_data_lens=None
    ):
        self.task_id = task_id
        self.cpu_block_ids = cpu_block_ids
        self.ssd_block_ids = ssd_block_ids
        self.peer_engine_addr = peer_engine_addr
        self.peer_cpu_base_ptr = peer_cpu_base_ptr
        self.peer_zmq_status_addr = peer_zmq_status_addr
        self.data_size = data_size
        self.layer_id = layer_id
        self.layer_granularity = layer_granularity
        self.gpu_dst_ptrs = gpu_dst_ptrs
        self.gpu_block_positions = gpu_block_positions
        self.gpu_layer_ids = gpu_layer_ids
        self.gpu_kv_ids = gpu_kv_ids
        self.gpu_token_ids = gpu_token_ids
        self.gpu_head_starts = gpu_head_starts
        self.gpu_data_lens = gpu_data_lens
        if gpu_dst_ptrs is not None:
            n = len(gpu_dst_ptrs)
            assert all(
                col is not None and len(col) == n
                for col in (gpu_block_positions, gpu_layer_ids, gpu_kv_ids,
                            gpu_token_ids, gpu_head_starts, gpu_data_lens)
            ), "RemoteSSD2HMetaInfo gpu_* lists must all be equal length"

    @classmethod
    def from_dict(self, data: dict) -> "RemoteSSD2HMetaInfo":
        return RemoteSSD2HMetaInfo(
            task_id = data.get("task_id"),
            cpu_block_ids=data.get("cpu_block_ids"),
            ssd_block_ids=data.get("ssd_block_ids"),
            peer_engine_addr=data.get("peer_engine_addr"),
            peer_cpu_base_ptr = data.get("peer_cpu_base_ptr"),
            peer_zmq_status_addr = data.get("peer_zmq_status_addr"),
            data_size=data.get("data_size"),
            layer_id = data.get("layer_id"),
            layer_granularity = data.get("layer_granularity"),
            gpu_dst_ptrs=data.get("gpu_dst_ptrs"),
            gpu_block_positions=data.get("gpu_block_positions"),
            gpu_layer_ids=data.get("gpu_layer_ids"),
            gpu_kv_ids=data.get("gpu_kv_ids"),
            gpu_token_ids=data.get("gpu_token_ids"),
            gpu_head_starts=data.get("gpu_head_starts"),
            gpu_data_lens=data.get("gpu_data_lens"),
        )
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "cpu_block_ids": self.cpu_block_ids,
            "ssd_block_ids": self.ssd_block_ids,
            "peer_engine_addr": self.peer_engine_addr,
            "peer_cpu_base_ptr": self.peer_cpu_base_ptr,
            "peer_zmq_status_addr": self.peer_zmq_status_addr,
            "data_size": self.data_size,
            "layer_id": self.layer_id,
            "layer_granularity": self.layer_granularity,
            "gpu_dst_ptrs": self.gpu_dst_ptrs,
            "gpu_block_positions": self.gpu_block_positions,
            "gpu_layer_ids": self.gpu_layer_ids,
            "gpu_kv_ids": self.gpu_kv_ids,
            "gpu_token_ids": self.gpu_token_ids,
            "gpu_head_starts": self.gpu_head_starts,
            "gpu_data_lens": self.gpu_data_lens,
        }

class NodeMetaInfo:
    """Node information for flexkv sub-nodes"""

    def __init__(
        self,
        node_id: int,
        engine_addr: Optional[str] = None,
        zmq_addr: Optional[str] = None,
        cpu_bufer_base_ptr: Optional[int] = None,
        ssd_bufer_base_ptr: Optional[int] = None,
    ):
        self.node_id = node_id
        self.engine_addr = engine_addr
        self.zmq_addr = zmq_addr
        self.cpu_bufer_base_ptr = cpu_bufer_base_ptr
        self.ssd_bufer_base_ptr = ssd_bufer_base_ptr

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_id": self.node_id,
            "addr": self.engine_addr,
            "zmq_addr": self.zmq_addr,
            "cpu_buffer_ptr": self.cpu_bufer_base_ptr,
            "ssd_buffer_ptr": self.ssd_bufer_base_ptr,
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeMetaInfo":
        return cls(
            node_id=data.get("node_id"),
            engine_addr=data.get("addr"),
            zmq_addr=data.get("zmq_addr"),
            cpu_bufer_base_ptr=data.get("cpu_buffer_ptr"),
            ssd_bufer_base_ptr=data.get("ssd_buffer_ptr"),
        )

class RDMATaskInfo:
    def __init__(
        self, task_id: int, local_engine_addr: str, peer_engine_addr: str,
        peer_zmq_addr: str, src_ptrs: List[int], dst_ptrs: List[int],
        src_block_ids: List[int], dst_block_ids: List[int], data_lens: List[int], data_size: int
    ):
        self.task_id = task_id
        self.local_engine_addr = local_engine_addr ## the mooncake engine address of local node
        self.peer_engine_addr = peer_engine_addr ## thre mooncake engine address of remote node
        self.src_ptrs = src_ptrs
        self.dst_ptrs = dst_ptrs
        self.peer_zmq_addr = peer_zmq_addr
        self.src_block_ids = src_block_ids
        self.dst_block_ids = dst_block_ids
        self.data_size = data_size
        self.data_lens = data_lens
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "local_engine_addr": self.local_engine_addr,
            "peer_engine_addr": self.peer_engine_addr,
            "peer_zmq_addr": self.peer_zmq_addr,
            "src_ptrs": self.src_ptrs,
            "dst_ptrs": self.dst_ptrs,
            "src_block_ids": self.src_block_ids,
            "dst_block_ids": self.dst_block_ids,
            "data_lens": self.data_lens,
            "data_size": self.data_size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RDMATaskInfo":
        return cls(
            task_id=data.get("task_id", 0),
            local_engine_addr=data.get("local_engine_addr", ""),
            peer_engine_addr=data.get("peer_engine_addr", ""),
            peer_zmq_addr=data.get("peer_zmq_addr", ""),
            src_ptrs=data.get("src_ptrs", []),
            dst_ptrs=data.get("dst_ptrs", []),
            src_block_ids=data.get("src_block_ids"),
            dst_block_ids=data.get("dst_block_ids"),
            data_lens=data.get("data_lens", []),
            data_size=int(data.get("data_size", 0)),
        )
