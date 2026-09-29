"""Echo probes for ion pairs and the Z2 symmetry sentinel."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Mapping, Sequence

from .program import Gate, LogicalProgram
from .resources import Resource, ion, pair
from .results import expectation, floored_log_signal


@lru_cache(maxsize=None)
def echo_program(depth: int) -> LogicalProgram:
    """H.H, then depth ZZ(pi/2) gates (identity for depth % 4 == 0), then H.H and measure."""
    if depth < 4 or depth % 4:
        raise ValueError("echo depth must be a positive multiple of 4")
    h = [Gate("rz", (0,), (math.pi / 2,)), Gate("sx", (0,)), Gate("rz", (0,), (math.pi / 2,))]
    hh = h + [Gate(g.name, (1,), g.params) for g in h]
    gates = hh + [Gate("rzz", (0, 1), (math.pi / 2,)) for _ in range(depth)] + hh
    return LogicalProgram(2, 2, gates, [(0, 0), (1, 1)], name=f"canary_echo_{depth}")


def echo_observations(
    ions: tuple[int, int],
    depth: int,
    freqs: Mapping[str, float],
    shots: float,
    model_sd: float,
) -> list[tuple[dict[Resource, float], float, float, float]]:
    """Known-ideal (+1) observations of <Z_0> and <Z_1> for an echo probe."""
    i, j = ions
    out = []
    for clbit, target in ((0, i), (1, j)):
        m = expectation(freqs, (clbit,))
        y, var = floored_log_signal(m, shots, model_sd)
        out.append(({ion(target): 1.0, pair(i, j): float(depth)}, y, 0.0, var))
    return out


def z2_flags(
    program: LogicalProgram,
    assignment: Sequence[int],
    freqs: Mapping[str, float],
    shots: float,
    threshold: float,
) -> set[int]:
    """Ions whose single-qubit <Z> deviates from the Z2-symmetric ideal value 0."""
    flagged: set[int] = set()
    n = max(float(shots), 1.0)
    for q, c in program.measurements:
        m = expectation(freqs, (c,))
        if abs(m) * math.sqrt(n) > threshold:
            flagged.add(int(assignment[q]))
    return flagged
