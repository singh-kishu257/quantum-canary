"""Splitting a packed joint histogram back into per-block results.

Count keys follow the Qiskit/IonQ convention: clbit 0 is the rightmost character. Spaces
(inserted by Qiskit between classical registers) are removed before use.
"""

from __future__ import annotations

from numbers import Integral
from typing import Mapping, Sequence

from .pack import PackedCircuit


def _normalize(key: str, num_clbits: int) -> str:
    k = key.replace(" ", "") if isinstance(key, str) else ""
    if not isinstance(key, str) or len(k) != num_clbits or not set(k) <= {"0", "1"}:
        raise ValueError(f"count key {key!r} is not a {num_clbits}-bit binary string")
    return k


def _extract(key: str, clbits: Sequence[int]) -> str:
    """Marginal key of a normalized key: local clbit 0 rightmost."""
    n = len(key)
    return "".join(key[n - 1 - c] for c in reversed(clbits))


def _check_clbits(clbits: Sequence[int], num_clbits: int) -> None:
    for c in clbits:
        if isinstance(c, bool) or not isinstance(c, Integral) or not 0 <= c < num_clbits:
            raise ValueError(f"clbit {c!r} out of range for {num_clbits} clbits")


def marginal_counts(counts: Mapping[str, float], clbits: Sequence[int], num_clbits: int) -> dict[str, float]:
    """Marginalize ``counts`` onto ``clbits`` (listed in block-local order)."""
    _check_clbits(clbits, num_clbits)
    out: dict[str, float] = {}
    for key, value in counts.items():
        m = _extract(_normalize(key, num_clbits), clbits)
        out[m] = out.get(m, 0) + value
    return out


def split_counts(counts: Mapping[str, float], packed: PackedCircuit) -> list[dict[str, float]]:
    """One marginal histogram per block, keyed in each block's own clbit order."""
    n = packed.circuit.num_clbits
    return [marginal_counts(counts, clbits, n) for clbits in packed.block_clbits]


def sentinel_excitation(counts: Mapping[str, float], packed: PackedCircuit) -> dict[int, float]:
    """Fraction of total weight with each sentinel ion read as 1 (a crosstalk/leakage monitor)."""
    sentinels = packed.layout.sentinels
    if sentinels and not packed.sentinel_clbits:
        raise ValueError("sentinels were not measured (pack(..., measure_sentinels=False))")
    n = packed.circuit.num_clbits
    total, ones = 0.0, [0.0] * len(sentinels)
    for key, value in counts.items():
        k = _normalize(key, n)
        total += value
        for i, c in enumerate(packed.sentinel_clbits):
            if k[n - 1 - c] == "1":
                ones[i] += value
    if total == 0:
        raise ValueError("counts have zero total weight")
    return {ion: ones[i] / total for i, ion in enumerate(sentinels)}
