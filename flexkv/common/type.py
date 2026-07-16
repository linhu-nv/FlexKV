from dataclasses import dataclass, field
from typing import Optional, Protocol, TypeVar, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from flexkv.common.block import SequenceMeta


class RadixNodeLike(Protocol):
    """Structural type shared by every radix-node flavor the cache layer moves
    around: the C++ ``CRadixNode`` and the shared-memory ``ShmRadixNode``. They
    have no common base class, but the cache_engine layer only ever calls
    ``size()`` on a node, so that is all the protocol needs to require."""

    def size(self) -> int: ...


# Each engine works with one concrete node type (CRadixNode or ShmRadixNode),
# consistent within an instance — hence a bounded TypeVar rather than a union.
NodeT = TypeVar("NodeT", bound=RadixNodeLike)


@dataclass
class MatchResultAccel:
    num_ready_matched_blocks: int = 0
    num_matched_blocks: int = 0
    last_ready_node: Optional["RadixNodeLike"] = None
    last_node: Optional["RadixNodeLike"] = None
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
    pre_locked_node: Optional["RadixNodeLike"] = None

    def __post_init__(self) -> None:
        assert self.physical_blocks.ndim == 1


class CacheEngineLike(Protocol[NodeT]):
    """Common surface of the three cache engines — ``CacheEngineAccel``,
    ``HierarchyLRCacheEngine`` and ``CacheEngineRadixShmem``. They are duck-typed
    (no shared base class); this protocol declares only the methods *every* engine
    implements. Generic over the engine's node type so a single instance stays
    consistent (CRadixNode vs ShmRadixNode).

    P2P-only surface (``start`` / ``match_local`` / ``match_all`` / ``local_index``)
    lives on ``HierarchyLRCacheEngine`` alone and is deliberately NOT part of this
    protocol; the config-guarded call sites reach it directly, and the type checker
    flagging those accesses is expected."""

    def reset(self) -> None: ...
    def match(self, sequence_meta: "SequenceMeta") -> MatchResultAccel: ...
    def insert(self,
               sequence_meta: "SequenceMeta",
               physical_block_ids: np.ndarray,
               num_insert_blocks: int = ...,
               is_ready: bool = ...,
               match_result: Optional[MatchResultAccel] = ...) -> "tuple[Optional[NodeT], np.ndarray]": ...
    def take(self,
             num_required_blocks: int,
             protected_node: Optional[NodeT] = ...,
             strict: bool = ...) -> np.ndarray: ...
    def recycle(self, physical_blocks: np.ndarray) -> None: ...
    def set_ready(self, node: NodeT, ready: bool, ready_length: int) -> None: ...
    def unlock(self, node: NodeT) -> None: ...
    def lock_node(self, node: NodeT) -> None: ...
