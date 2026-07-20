from typing import List, Tuple

import numpy as np

from flexkv.common.transfer import TransferType
from flexkv.common.transfer import TransferOp, TransferOpGraph


def add_virtal_op_for_mutiple_finished_ops(
    graph: TransferOpGraph,
    finished_ops_ids: List[int]
)->Tuple[TransferOpGraph, int]:
    if len(finished_ops_ids) == 0:
        return graph, -1
    elif len(finished_ops_ids) == 1:
        return graph, finished_ops_ids[0]
    else:
        op = TransferOp(
            graph_id = graph.graph_id,
            transfer_type = TransferType.VIRTUAL,
            src_block_ids = np.array([], dtype=np.int64),
            dst_block_ids = np.array([], dtype=np.int64),
            layer_id = -1,
            layer_granularity = -1,
        )
        graph.add_transfer_op(op)
        for op_id in finished_ops_ids:
            graph.add_dependency(op.op_id, op_id)
        return graph, op.op_id
