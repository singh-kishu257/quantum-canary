"""Placement of disjoint ion blocks on one linear register."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence, TypeVar

T = TypeVar("T")


def _check_int(name: str, value: object, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an int, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


@dataclass(frozen=True)
class Layout:
    """Disjoint ion blocks in execution order, with unused gap ions between them."""

    blocks: tuple[tuple[int, ...], ...]  # ion indices of each block, in execution order
    sentinels: tuple[int, ...]  # gap ions strictly between blocks (not trailing)
    n_ions: int
    gap: int

    def __post_init__(self) -> None:
        flat = [i for b in self.blocks for i in b]
        if not self.blocks or any(len(b) == 0 for b in self.blocks):
            raise ValueError("a Layout needs at least one block and no empty block")
        if flat != sorted(set(flat)) or flat[0] < 0 or flat[-1] >= self.n_ions:
            raise ValueError("block ion indices must be increasing, disjoint and in [0, n_ions)")

    @property
    def width(self) -> int:
        """Number of ions spanned: max index used + 1."""
        return self.blocks[-1][-1] + 1


def capacity(block_width: int, n_ions: int, gap: int = 1) -> int:
    """Number of blocks of ``block_width`` ions that fit: floor((n_ions + gap) / (block_width + gap))."""
    _check_int("block_width", block_width, 1)
    _check_int("n_ions", n_ions, 1)
    _check_int("gap", gap, 0)
    return (n_ions + gap) // (block_width + gap)


def make_layout(widths: Sequence[int], n_ions: int, gap: int = 1) -> Layout:
    """Place blocks of the given widths left to right with ``gap`` unused ions between them."""
    if len(widths) == 0:
        raise ValueError("widths must not be empty")
    _check_int("n_ions", n_ions, 1)
    _check_int("gap", gap, 0)
    blocks, start = [], 0
    for w in widths:
        w = _check_int("block width", w, 1)
        blocks.append(tuple(range(start, start + w)))
        start += w + gap
    if blocks[-1][-1] >= n_ions:
        raise ValueError(f"layout needs {blocks[-1][-1] + 1} ions but only {n_ions} are available")
    used = {i for b in blocks for i in b}
    sentinels = tuple(i for i in range(blocks[-1][-1] + 1) if i not in used)
    return Layout(tuple(blocks), sentinels, n_ions, gap)


def latin_rotation(items: Sequence[T], shift: int) -> list[T]:
    """Cyclic shift: slot ``s`` holds ``items[(s - shift) % k]``, i.e. item ``i`` lands in slot ``(i + shift) % k``.

    Over ``shift = 0, ..., k-1`` every item visits every slot exactly once (a Latin square).
    """
    shift = _check_int("shift", shift)
    k = len(items)
    return [items[(s - shift) % k] for s in range(k)]
