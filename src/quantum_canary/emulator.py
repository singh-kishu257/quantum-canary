"""Heterogeneous trapped-ion emulator with known ground truth, for validating Canary offline.

Canary's native circuits are run on qiskit-aer with a depolarizing ZZ error per ion pair, an optional coherent ZZ
over-rotation per pair, a depolarizing error on every single-qubit pulse and asymmetric readout per ion. The
default pair-error distribution (log-normal around 0.46 % process infidelity) is only loosely shaped after
published Forte direct-randomized-benchmarking statistics; it is a test bed, not a model of any IonQ system.
Every job draws a fresh simulator seed from ``seed``: runs are reproducible, and shot noise is independent between
jobs (a fixed Aer ``seed_simulator`` would replay the same samples for the same circuit position in every job).
Requires the ``aer`` extra.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit

from .executors import BackendExecutor


class _EmulatorExecutor(BackendExecutor):
    """Backend executor that seeds every job from its own random stream."""

    def __init__(self, backend: Any, ions: Iterable[int], seed: int) -> None:
        super().__init__(backend, ions=list(ions))
        self._seeds = np.random.default_rng(seed)

    def run(self, circuits: Sequence[QuantumCircuit], shots: int) -> list[Mapping[str, float]]:
        self.run_options["seed_simulator"] = int(self._seeds.integers(2**31 - 1))
        return super().run(circuits, shots)


@dataclass
class EmulatorTruth:
    """Ground truth of an emulated device (ion indices 0..n-1)."""

    pair_error: dict[tuple[int, int], float]
    pair_over_rotation: dict[tuple[int, int], float]
    readout: dict[int, tuple[float, float]]
    pulse_error: float

    def pair_infidelity(self, a: int, b: int) -> float:
        """Process infidelity of the Pauli-twirled ZZ channel: stochastic part plus sin^2(pi delta)."""
        key = (min(a, b), max(a, b))
        return self.pair_error[key] + math.sin(math.pi * self.pair_over_rotation.get(key, 0.0)) ** 2

    def spam_error(self, i: int) -> float:
        """Symmetrised readout flip probability of an ion (preparation is ideal in the emulator)."""
        p01, p10 = self.readout[i]
        return 0.5 * (p01 + p10)


def forte_like_emulator(
    ions: int = 12,
    seed: int = 0,
    median_pair_error: float = 0.0046,
    log_spread: float = 0.4,
    bad_pairs: Iterable[tuple[int, int]] = (),
    bad_pair_error: float = 0.05,
    over_rotation_sd: float = 0.0,
    readout_range: tuple[float, float] = (0.002, 0.008),
    readout_asymmetry: float = 2.0,
    bad_ions: Iterable[int] = (),
    bad_readout_error: float = 0.05,
    pulse_error: float = 2.0e-4,
    **simulator_options: Any,
) -> tuple[BackendExecutor, EmulatorTruth]:
    """Build an Aer-backed executor for ``ions`` ions and return it with the ground truth."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, coherent_unitary_error, depolarizing_error

    rng = np.random.default_rng(seed)
    n = int(ions)
    pair_error: dict[tuple[int, int], float] = {}
    over: dict[tuple[int, int], float] = {}
    for a in range(n):
        for b in range(a + 1, n):
            pair_error[(a, b)] = min(median_pair_error * math.exp(log_spread * rng.standard_normal()), 0.5)
            over[(a, b)] = over_rotation_sd * float(rng.standard_normal()) if over_rotation_sd > 0 else 0.0
    for a, b in bad_pairs:
        pair_error[(min(a, b), max(a, b))] = float(bad_pair_error)
    readout: dict[int, tuple[float, float]] = {}
    bad_set = {int(i) for i in bad_ions}
    for i in range(n):
        p = float(rng.uniform(*readout_range))
        if i in bad_set:
            p = float(bad_readout_error)
        p10 = 2.0 * p * readout_asymmetry / (1.0 + readout_asymmetry)
        readout[i] = (2.0 * p - p10, p10)
    noise = NoiseModel()
    pulse = depolarizing_error(4.0 * pulse_error / 3.0, 1) if pulse_error > 0 else None
    for i in range(n):
        if pulse is not None:
            noise.add_quantum_error(pulse, "u", [i])
        p01, p10 = readout[i]
        noise.add_readout_error(ReadoutError([[1.0 - p01, p01], [p10, 1.0 - p10]]), [i])
    parity = np.array([1.0, -1.0, -1.0, 1.0])
    for (a, b), e in pair_error.items():
        error = depolarizing_error(16.0 * e / 15.0, 2)
        if over[(a, b)] != 0.0:
            error = error.compose(coherent_unitary_error(np.diag(np.exp(-1j * math.pi * over[(a, b)] * parity))))
        noise.add_quantum_error(error, "rzz", [a, b])
        noise.add_quantum_error(error, "rzz", [b, a])
    simulator_options.pop("seed_simulator", None)
    backend = AerSimulator(noise_model=noise, **simulator_options)
    return _EmulatorExecutor(backend, range(n), seed), EmulatorTruth(pair_error, over, readout, pulse_error)
