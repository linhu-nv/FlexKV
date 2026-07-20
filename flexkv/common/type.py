from dataclasses import dataclass, field
from enum import Enum
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


class CacheLocality(str, Enum):
    LOCAL = "local"
    PEER = "peer"


@dataclass
class MatchResultAccel:
    """A single-locality prefix match against one cache tier's index.

    ``physical_blocks[i]`` is the block that serves logical position ``i`` of
    the queried sequence.  ``block_node_ids[i]`` (when present) names the
    source that owns block ``i`` — a peer node id for a PEER match, or a PCFS
    file node id for a Lake match.  A tier match is expressed as a
    ``MatchResult`` pairing the LOCAL hit with an optional peer (remote) hit
    that extends it; each side is one of these objects.
    """

    num_ready_matched_blocks: int = 0
    num_matched_blocks: int = 0
    last_ready_node: Optional["RadixNodeLike"] = None
    last_node: Optional["RadixNodeLike"] = None
    last_node_matched_length: int = 0
    physical_blocks: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    block_node_ids: Optional[np.ndarray] = None
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


@dataclass
class MatchResult:
    """Unified result of a tier match: the local prefix hit plus an optional
    peer (remote) hit that continues it.

    ``remote`` is ``None`` when the tier has no peer index or peer matching was
    not requested (e.g. PUT).  When present, ``remote`` covers a ready prefix
    that reaches at least as far as ``local`` and whose ``physical_blocks`` /
    ``block_node_ids`` describe the peer source for every logical position
    beyond ``local.num_ready_matched_blocks``.
    """

    local: MatchResultAccel
    remote: Optional[MatchResultAccel] = None

    @property
    def peer_ready(self) -> int:
        """Ready prefix length reachable once the peer suffix is included."""
        base = self.local.num_ready_matched_blocks
        return max(base, self.remote.num_ready_matched_blocks) if self.remote else base


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
    def match(self,
              sequence_meta: "SequenceMeta",
              *,
              with_peer: bool = ...,
              gpu_matched_blocks: int = ...) -> MatchResult: ...
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
