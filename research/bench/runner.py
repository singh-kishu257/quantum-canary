"""Evaluation of assignments, baselines and the Canary loop on a simulated device."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from quantum_canary import AssignmentSolver, Canary, CanaryConfig, LogicalProgram, compile_native

from .circuits import hellinger_fidelity, ideal_distribution, mixture
from .device import DeviceTruth, ExactDevice


class Evaluator:
    """Exact (infinite-shot) outcome distributions of a program under assignments."""

    def __init__(self, program: LogicalProgram, device: ExactDevice) -> None:
        self.program = program
        self.device = device
        self.ideal = ideal_distribution(program)
        self._cache: dict[tuple, dict[str, float]] = {}

    def distribution(self, assignment: Sequence[int]) -> dict[str, float]:
        key = (tuple(int(x) for x in assignment), len(self.device.truth.events))
        if key not in self._cache:
            qc = compile_native(self.program, assignment, twirl=False)
            self._cache[key] = self.device.probabilities(qc)
        return self._cache[key]

    def fidelity(self, assignment: Sequence[int]) -> float:
        return hellinger_fidelity(self.distribution(assignment), self.ideal)

    def mixture_fidelity(self, parts: Sequence[tuple[Sequence[int], float]]) -> float:
        return hellinger_fidelity(mixture([(self.distribution(a), w) for a, w in parts]), self.ideal)


def random_assignments(n: int, ions: Sequence[int], k: int, rng: np.random.Generator) -> list[tuple[int, ...]]:
    """k random injective logical-to-ion maps."""
    ions = np.asarray(ions)
    return [tuple(int(x) for x in rng.permutation(ions)[:n]) for _ in range(k)]


def oracle_assignment(
    program: LogicalProgram, truth: DeviceTruth, evaluator: Evaluator, rng: np.random.Generator, restarts: int = 4, steps: int = 20000
) -> tuple[tuple[int, ...], float]:
    """Best assignment found with the true damping tables, confirmed by exact fidelity."""
    wp, wi = program.weights()
    iv, pv = truth.damping_tables()
    best, best_f = None, -1.0
    for r in range(restarts):
        perm = AssignmentSolver(steps, np.random.default_rng(rng.integers(2**31))).solve(wp, wi, pv, iv)
        a = tuple(int(x) for x in perm)
        f = evaluator.fidelity(a)
        if f > best_f:
            best, best_f = a, f
    return best, best_f


@dataclass
class CanaryTrace:
    """Per-batch record of a Canary run."""

    batch_fidelity: list[float] = field(default_factory=list)
    batch_exploit_fidelity: list[float] = field(default_factory=list)
    recommended_fidelity: list[float] = field(default_factory=list)
    recommended: list[tuple[int, ...]] = field(default_factory=list)
    degraded: list[list[str]] = field(default_factory=list)
    trusted_parts: list[tuple[tuple[int, ...], float]] = field(default_factory=list)
    all_parts: list[tuple[tuple[int, ...], float]] = field(default_factory=list)
    algorithm_shots: int = 0
    probe_shots: int = 0
    canary: Canary | None = None


def run_canary(
    program: LogicalProgram,
    device: ExactDevice,
    evaluator: Evaluator,
    batches: int,
    shots_per_batch: int,
    config: CanaryConfig,
    batch_seconds: float = 600.0,
    before_batch: Callable[[int, Canary], None] | None = None,
) -> CanaryTrace:
    """Run Canary for a number of batches and record exact fidelities of what it ran."""
    clock_t = [0.0]

    def clock() -> float:
        return clock_t[0]

    canary = Canary(device, config, clock=clock)
    trace = CanaryTrace(canary=canary)
    for b in range(batches):
        clock_t[0] = (b + 1) * batch_seconds
        if before_batch is not None:
            before_batch(b, canary)
        shots_before = device.shots_used
        res = canary.run(program, shots_per_batch)
        used = device.shots_used - shots_before
        alg = res.shots
        trace.algorithm_shots += alg
        trace.probe_shots += used - alg
        trusted = [(r.assignment, float(r.shots)) for r in res.runs if r.trusted] or [
            (r.assignment, float(r.shots)) for r in res.runs
        ]
        trace.trusted_parts.extend(trusted)
        trace.all_parts.extend((r.assignment, float(r.shots)) for r in res.runs)
        trace.batch_fidelity.append(evaluator.mixture_fidelity(trusted))
        trace.batch_exploit_fidelity.append(evaluator.fidelity(res.runs[0].assignment))
        trace.recommended.append(res.recommended)
        trace.recommended_fidelity.append(evaluator.fidelity(res.recommended))
        trace.degraded.append([":".join(map(str, r)) for r in res.degraded])
    return trace


def geometric_mean(values: Sequence[float]) -> float:
    v = np.asarray(values, dtype=float)
    return float(np.exp(np.mean(np.log(v))))
