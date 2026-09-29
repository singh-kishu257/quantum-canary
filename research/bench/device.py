"""Simulated trapped-ion device with heterogeneous, drifting pair noise.

Parameter anchors (Chen et al., Quantum 2024, arXiv:2308.05071, IonQ Forte):
median two-qubit DRB error 46.4e-4 with most pairs in 35-100e-4 and a tail to 885e-4
(DRB rates are process/entanglement infidelities; depolarizing parameter = 16 e / 15);
median single-qubit DRB error 2.0e-4; average SPAM error 0.5%.
The outlier fraction and outlier range are modelling assumptions and are swept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from quantum_canary.executors import Executor
from quantum_canary.resources import pair_damping, readout_damping

BULK_MEDIAN = 46.4e-4
BULK_SIGMA = 0.30
OUTLIER_RANGE = (0.01, 0.0885)
ONEQ_ERROR = 2.0e-4
SPAM_MEDIAN = 0.005
SPAM_SIGMA = 0.30


@dataclass
class DeviceTruth:
    """Hidden per-ion and per-pair error rates."""

    pair_error: np.ndarray
    readout_error: np.ndarray
    oneq_error: np.ndarray
    events: list = field(default_factory=list)

    @property
    def num_ions(self) -> int:
        return int(self.readout_error.size)

    def copy(self) -> "DeviceTruth":
        return DeviceTruth(self.pair_error.copy(), self.readout_error.copy(), self.oneq_error.copy(), list(self.events))

    def set_pair(self, i: int, j: int, value: float) -> None:
        self.pair_error[i, j] = self.pair_error[j, i] = float(value)
        self.events.append(("pair", int(i), int(j), float(value)))

    def damping_tables(self) -> tuple[np.ndarray, np.ndarray]:
        iv = np.array([readout_damping(p) for p in self.readout_error])
        pv = np.vectorize(pair_damping)(self.pair_error)
        np.fill_diagonal(pv, 0.0)
        return iv, pv

    def bad_pairs(self, threshold: float = 0.02) -> set[tuple[int, int]]:
        m = self.num_ions
        return {(i, j) for i in range(m) for j in range(i + 1, m) if self.pair_error[i, j] >= threshold}


def sample_device(
    num_ions: int,
    rng: np.random.Generator,
    outlier_fraction: float = 0.05,
    bulk_median: float = BULK_MEDIAN,
    bulk_sigma: float = BULK_SIGMA,
    outlier_range: tuple[float, float] = OUTLIER_RANGE,
) -> DeviceTruth:
    """Draw a device: log-normal bulk pair errors plus log-uniform outliers."""
    m = num_ions
    pe = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            if rng.random() < outlier_fraction:
                v = float(np.exp(rng.uniform(np.log(outlier_range[0]), np.log(outlier_range[1]))))
            else:
                v = float(bulk_median * np.exp(bulk_sigma * rng.standard_normal()))
            pe[i, j] = pe[j, i] = min(v, 0.7)
    ro = np.clip(SPAM_MEDIAN * np.exp(SPAM_SIGMA * rng.standard_normal(m)), 1e-4, 0.45)
    oq = np.full(m, ONEQ_ERROR)
    return DeviceTruth(pe, ro, oq)


class ExactDevice(Executor):
    """Executes native circuits exactly (density matrix) under the truth, then samples shots."""

    def __init__(self, truth: DeviceTruth, seed: int = 0) -> None:
        self.truth = truth
        self.rng = np.random.default_rng(seed)
        self._sim = AerSimulator(method="density_matrix")
        self._ions = tuple(range(truth.num_ions))
        self.shots_used = 0
        self.circuits_run = 0

    @property
    def ions(self) -> tuple[int, ...]:
        return self._ions

    def _compact(self, qc: QuantumCircuit) -> tuple[QuantumCircuit, NoiseModel, list[int], list[tuple[int, int]]]:
        used = sorted({qc.find_bit(q).index for inst in qc.data for q in inst.qubits})
        pos = {ion: k for k, ion in enumerate(used)}
        n = len(used)
        out = QuantumCircuit(n)
        meas: list[tuple[int, int]] = []
        pairs: set[tuple[int, int]] = set()
        for inst in qc.data:
            qs = [qc.find_bit(q).index for q in inst.qubits]
            name = inst.operation.name
            if name == "measure":
                meas.append((pos[qs[0]], qc.find_bit(inst.clbits[0]).index))
            elif name in ("gpi", "gpi2"):
                p = inst.operation.params[0]
                if name == "gpi":
                    out.u(np.pi, 2 * np.pi * p, np.pi - 2 * np.pi * p, pos[qs[0]])
                else:
                    out.u(np.pi / 2, 2 * np.pi * p - np.pi / 2, np.pi / 2 - 2 * np.pi * p, pos[qs[0]])
            elif name == "zz":
                a, b = pos[qs[0]], pos[qs[1]]
                out.rzz(2 * np.pi * float(inst.operation.params[0]), a, b)
                pairs.add((min(a, b), max(a, b)))
            elif name == "barrier":
                continue
            else:
                raise ValueError(f"unsupported native gate {name}")
        nm = NoiseModel(basis_gates=["u", "rzz"])
        for k, ion in enumerate(used):
            r1 = float(self.truth.oneq_error[ion])
            if r1 > 0:
                nm.add_quantum_error(depolarizing_error(2 * r1, 1), "u", [k])
        for a, b in pairs:
            r2 = float(self.truth.pair_error[used[a], used[b]])
            if r2 > 0:
                err = depolarizing_error(min(16 * r2 / 15, 1.0), 2)
                nm.add_quantum_error(err, "rzz", [a, b])
                nm.add_quantum_error(err, "rzz", [b, a])
        out.save_probabilities(label="p")
        return out, nm, used, meas

    def probabilities(self, qc: QuantumCircuit) -> dict[str, float]:
        """Exact outcome distribution over the circuit's clbits, including readout error."""
        return self.probabilities_many([qc])[0]

    def probabilities_many(self, circuits: Sequence[QuantumCircuit]) -> list[dict[str, float]]:
        out = []
        for qc in circuits:
            comp, nm, used, meas = self._compact(qc)
            self._sim.set_options(noise_model=nm)
            probs = np.asarray(self._sim.run(comp, shots=1).result().data(0)["p"], dtype=float)
            n = len(used)
            tensor = probs.reshape((2,) * n)
            for k, ion in enumerate(used):
                p = float(self.truth.readout_error[ion])
                conf = np.array([[1 - p, p], [p, 1 - p]])
                axis = n - 1 - k
                tensor = np.moveaxis(np.tensordot(conf, tensor, axes=([1], [axis])), 0, axis)
            flat = np.clip(tensor.reshape(-1), 0.0, None)
            flat /= flat.sum()
            ncl = qc.num_clbits
            dist: dict[str, float] = {}
            for idx in np.flatnonzero(flat > 1e-15):
                bits = ["0"] * ncl
                for k, c in meas:
                    if (idx >> k) & 1:
                        bits[ncl - 1 - c] = "1"
                key = "".join(bits)
                dist[key] = dist.get(key, 0.0) + float(flat[idx])
            out.append(dist)
        return out

    def run(self, circuits: Sequence[QuantumCircuit], shots: int) -> list[Mapping[str, float]]:
        results = []
        for dist in self.probabilities_many(circuits):
            keys = list(dist)
            p = np.array([dist[k] for k in keys])
            draws = self.rng.multinomial(int(shots), p / p.sum())
            results.append({k: int(c) for k, c in zip(keys, draws) if c})
        self.shots_used += int(shots) * len(circuits)
        self.circuits_run += len(circuits)
        self.last_job_ids = (f"sim-{self.circuits_run}",)
        return results
