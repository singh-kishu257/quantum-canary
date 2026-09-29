"""Physical resources (ions and ion pairs) and damping-rate conversions."""

from __future__ import annotations

import math
from typing import Any, Tuple

Resource = Tuple[Any, ...]


def ion(index: int) -> Resource:
    """Resource key for a single ion (readout and single-qubit control)."""
    return ("ion", int(index))


def pair(a: int, b: int) -> Resource:
    """Resource key for the two-qubit ZZ gate between two ions."""
    a, b = int(a), int(b)
    if a == b:
        raise ValueError("a pair needs two distinct ions")
    return ("pair", min(a, b), max(a, b))


def is_pair(resource: Resource) -> bool:
    """True if the resource is an ion pair."""
    return resource[0] == "pair"


def ions_of(resource: Resource) -> tuple[int, ...]:
    """Ions touched by a resource."""
    return tuple(resource[1:])


def pair_damping(infidelity: float) -> float:
    """Damping rate of a two-qubit depolarizing gate with given average infidelity."""
    return -math.log1p(-4.0 * infidelity / 3.0)


def pair_infidelity(damping: float) -> float:
    """Average infidelity of a two-qubit depolarizing gate with given damping rate."""
    return 0.75 * (1.0 - math.exp(-max(damping, 0.0)))


def readout_damping(error: float) -> float:
    """Damping rate of a symmetric readout flip with given error probability."""
    return -math.log1p(-2.0 * error)


def readout_error(damping: float) -> float:
    """Symmetric readout flip probability with given damping rate."""
    return 0.5 * (1.0 - math.exp(-max(damping, 0.0)))


def to_error(resource: Resource, damping: float) -> float:
    """Convert a damping rate to the natural error metric of the resource."""
    return pair_infidelity(damping) if is_pair(resource) else readout_error(damping)


def to_key(resource: Resource) -> str:
    """Serialise a resource key to a string."""
    return ":".join(str(x) for x in resource)


def from_key(key: str) -> Resource:
    """Parse a resource key produced by to_key."""
    kind, *rest = key.split(":")
    if kind == "ion" and len(rest) == 1:
        return ion(int(rest[0]))
    if kind == "pair" and len(rest) == 2:
        return pair(int(rest[0]), int(rest[1]))
    raise ValueError(f"invalid resource key {key!r}")
