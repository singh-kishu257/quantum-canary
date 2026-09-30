"""Physical resources (ions and ion pairs) and damping-rate conversions.

Canary tracks two kinds of resource, both learned only from calibration (Tier-1) circuits:

* ``pair(i, j)``: the dressed ZZ(1/4 turn) cycle on ions ``i`` and ``j``, i.e. the ZZ gate together with its
  randomized-compiling Pauli frames. Its damping rate ``x`` is the per-gate decay of two-qubit Pauli expectation
  values averaged uniformly over the 15 non-identity Paulis, as measured by cycle benchmarking. For a
  depolarizing channel ``x = -log(1 - lambda)`` and the process (entanglement) infidelity is
  ``15/16 * (1 - exp(-x))``; for a general Pauli channel the same conversion holds to first order in the error.
* ``ion(i)``: state preparation and measurement (SPAM) of ion ``i`` under twirled (symmetrised) readout, which damps
  ``<Z>`` by ``1 - 2p`` with ``p`` the symmetrised flip probability.
"""

from __future__ import annotations

import math
from typing import Any, Tuple

Resource = Tuple[Any, ...]

KINDS: tuple[str, ...] = ("ion", "pair")


def ion(index: int) -> Resource:
    """Resource key for the SPAM of a single ion."""
    return ("ion", int(index))


def pair(a: int, b: int) -> Resource:
    """Resource key for the dressed ZZ cycle between two ions."""
    a, b = int(a), int(b)
    if a == b:
        raise ValueError("a pair needs two distinct ions")
    return ("pair", min(a, b), max(a, b))


def is_pair(resource: Resource) -> bool:
    """True if the resource is an ion pair."""
    return resource[0] == "pair"


def kind(resource: Resource) -> str:
    """Resource kind, 'ion' or 'pair'."""
    return str(resource[0])


def ions_of(resource: Resource) -> tuple[int, ...]:
    """Ions touched by a resource."""
    return tuple(resource[1:])


def pair_damping(infidelity: float) -> float:
    """Damping rate of a two-qubit depolarizing gate with given process (entanglement) infidelity."""
    return -math.log1p(-16.0 * infidelity / 15.0)


def pair_infidelity(damping: float) -> float:
    """Process (entanglement) infidelity of a two-qubit depolarizing gate with given damping rate."""
    return (15.0 / 16.0) * (1.0 - math.exp(-max(damping, 0.0)))


def readout_damping(error: float) -> float:
    """Damping rate of a symmetric flip with given probability."""
    return -math.log1p(-2.0 * error)


def readout_error(damping: float) -> float:
    """Symmetric flip probability with given damping rate."""
    return 0.5 * (1.0 - math.exp(-max(damping, 0.0)))


def to_error(resource: Resource, damping: float) -> float:
    """Convert a damping rate to the natural error metric of the resource."""
    return pair_infidelity(damping) if is_pair(resource) else readout_error(damping)


def to_key(resource: Resource) -> str:
    """Serialise a resource key to a string."""
    return ":".join(str(x) for x in resource)


def from_key(key: str) -> Resource:
    """Parse a resource key produced by to_key."""
    kind_, *rest = key.split(":")
    if kind_ == "ion" and len(rest) == 1:
        return ion(int(rest[0]))
    if kind_ == "pair" and len(rest) == 2:
        return pair(int(rest[0]), int(rest[1]))
    raise ValueError(f"invalid resource key {key!r}")
