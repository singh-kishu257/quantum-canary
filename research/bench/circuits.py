"""Benchmark circuits, ideal distributions and fidelity metrics."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantum_canary import LogicalProgram


def ising_trotter(n: int, steps: int, dt: float = 0.25, j: float = 1.0, h: float = 0.9) -> QuantumCircuit:
    """Transverse-field Ising chain, first-order Trotter, from |0...0>."""
    qc = QuantumCircuit(n, name=f"ising_n{n}_s{steps}")
    for _ in range(steps):
        for q in range(n - 1):
            qc.rzz(2 * j * dt, q, q + 1)
        qc.rx(2 * h * dt, range(n))
    qc.measure_all()
    return qc


def qaoa_ring(n: int, p: int, gamma: float = 0.6, beta: float = 0.35) -> QuantumCircuit:
    """MaxCut QAOA on a ring with a linear parameter ramp."""
    qc = QuantumCircuit(n, name=f"qaoa_ring_n{n}_p{p}")
    qc.h(range(n))
    for layer in range(p):
        f = (layer + 1) / p
        for q in range(n):
            qc.rzz(2 * gamma * f, q, (q + 1) % n)
        qc.rx(2 * beta * (1 - f + 1 / p), range(n))
    qc.measure_all()
    return qc


def ideal_distribution(program: LogicalProgram) -> dict[str, float]:
    """Noiseless outcome distribution over the program's clbits."""
    qc = program.to_circuit().remove_final_measurements(inplace=False)
    probs = Statevector.from_instruction(qc).probabilities()
    ncl = program.num_clbits
    out: dict[str, float] = {}
    for idx in np.flatnonzero(probs > 1e-15):
        bits = ["0"] * ncl
        for q, c in program.measurements:
            if (idx >> q) & 1:
                bits[ncl - 1 - c] = "1"
        key = "".join(bits)
        out[key] = out.get(key, 0.0) + float(probs[idx])
    return out


def hellinger_fidelity(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """Hellinger fidelity (Bhattacharyya coefficient squared)."""
    return float(sum(math.sqrt(p.get(k, 0.0) * q.get(k, 0.0)) for k in set(p) | set(q)) ** 2)


def mixture(parts: Sequence[tuple[Mapping[str, float], float]]) -> dict[str, float]:
    """Weighted mixture of distributions."""
    total = sum(w for _, w in parts)
    out: dict[str, float] = {}
    for dist, w in parts:
        for k, v in dist.items():
            out[k] = out.get(k, 0.0) + v * w / total
    return out
