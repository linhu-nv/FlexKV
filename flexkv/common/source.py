"""Typed descriptors for *where* a matched block's data comes from.

Historically a single per-block ``np.ndarray`` (``block_node_ids``) carried three
unrelated meanings at once: a single peer node id (broadcast to every block),
genuinely per-block peer ids, and per-block PCFS file ids.  This module replaces
that overloaded array with a small typed union so each source states exactly
what it is.

Deployment invariant (P2P): a peer (remote) match is served by exactly ONE peer
node, so peer ownership is a scalar (:class:`PeerSource`), not a per-block array.
LAKE reads still partition across PCFS files per block (:class:`LakeSource`).
Blocks with no external owner use :class:`LocalSource`.

Kept dependency-free (no transfer/torch imports) so ``flexkv.common`` stays a
leaf: descriptors only hold data and support a positional ``slice`` / ``covers``.
The grouping of blocks for RDMA lives in the transfer layer, which reads these
fields.
"""
from dataclasses import dataclass

import numpy as np


class BlockSource:
    """Base marker for a contiguous logical block span's origin.

    ``slice(start, stop)`` narrows the source to the sub-span ``[start, stop)``;
    ``covers(num_blocks)`` checks the source carries enough per-block metadata
    for a span of ``num_blocks``.  The scalar variants override neither meaning-
    fully (position-independent), which is exactly why they need no per-block
    array.
    """

    def slice(self, start: int, stop: int) -> "BlockSource":
        return self

    def covers(self, num_blocks: int) -> bool:
        return True


@dataclass(frozen=True)
class LocalSource(BlockSource):
    """No external owner: local cpu/ssd hits, PUT-only matches, placeholders."""


@dataclass(frozen=True)
class PeerSource(BlockSource):
    """The whole span is served by a single peer node (``node_id``).

    Position-independent: slicing returns the same source because one peer owns
    every block.  This scalar replaces the old per-block broadcast.
    """

    node_id: int

    def slice(self, start: int, stop: int) -> "PeerSource":
        return self


@dataclass(frozen=True)
class LakeSource(BlockSource):
    """Per-block PCFS file node ids for a LAKE read (``file_ids[i]`` owns block i).

    Genuinely per-block: one logical span fans out across several PCFS files, so
    the array is required and slicing narrows it.
    """

    file_ids: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "file_ids", np.asarray(self.file_ids, dtype=np.int64)
        )

    def slice(self, start: int, stop: int) -> "LakeSource":
        return LakeSource(self.file_ids[start:stop])

    def covers(self, num_blocks: int) -> bool:
        return len(self.file_ids) >= num_blocks
