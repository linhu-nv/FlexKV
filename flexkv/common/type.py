from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class MatchResultAccel:
    num_ready_matched_blocks: int = 0
    num_matched_blocks: int = 0
    last_ready_node: Optional['CRadixNode'] = None
    last_node: Optional['CRadixNode'] = None
    last_node_matched_length: int = 0
    physical_blocks: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    block_node_ids: Optional[np.ndarray] = None
    matched_pos: Optional[str] = None
    matched_node_ids: Optional[np.ndarray] = None #TODO id or ids? should we allow one req match results on multiple nodes?
    insert_to_local_cpu_index: bool = True
    # Set by backends whose `match()` performs an atomic inc_ref to protect the
    # matched slots from eviction between the read and the consuming transfer
    # (e.g. CacheEngineRadixShmem with lock=True). The cache_engine layer is
    # responsible for releasing this exactly once — either right after it has
    # acquired its own protection via `lock_node`, or in any early-return path
    # that skips the transfer. Leaving it un-released leaks a ref per match
    # and eventually pins the matched path against eviction.
    pre_locked_node: Optional['CRadixNode'] = None

    def __post_init__(self) -> None:
        assert self.physical_blocks.ndim == 1


