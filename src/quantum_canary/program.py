"""Logical programs in the (rz, sx, x, rzz) basis with lightcone analysis."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile

from .resources import Resource, ion, pair

BASIS: tuple[str, ...] = ("rz", "sx", "x", "rzz")
_IGNORED = frozenset({"barrier", "id", "delay"})
EPS = 1e-9
QUARTER_TURN = 0.25


def wrap_angle(theta: float) -> float:
    """Wrap an angle to the interval (-pi, pi]."""
    t = math.remainder(float(theta), 2.0 * math.pi)
    return math.pi if abs(t + math.pi) < EPS else t


def physical_turns(phi: float) -> float:
    """Angle in turns of the positive ZZ gate that realises rzz(phi) after normalisation, in (0, 0.25]."""
    t = abs(wrap_angle(phi))
    if t > math.pi / 2:
        t = math.pi - t
    return t / (2.0 * math.pi)


def angle_weight(phi: float, exponent: float) -> float:
    """Relative exposure of an rzz(phi) gate, (turns / 0.25) ** exponent; 1 for exponent 0."""
    if exponent == 0.0:
        return 1.0
    return (physical_turns(phi) / QUARTER_TURN) ** exponent


@dataclass(frozen=True)
class Gate:
    """A basis gate acting on logical qubits."""

    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()


@dataclass(frozen=True)
class Observable:
    """A Z-string observable on measured logical qubits."""

    qubits: tuple[int, ...]
    clbits: tuple[int, ...]


class LogicalProgram:
    """A measured circuit in the (rz, sx, x, rzz) basis with terminal measurements."""

    def __init__(
        self,
        num_qubits: int,
        num_clbits: int,
        gates: Sequence[Gate],
        measurements: Sequence[tuple[int, int]],
        name: str = "program",
    ) -> None:
        self.num_qubits = int(num_qubits)
        self.num_clbits = int(num_clbits)
        self.gates = tuple(gates)
        self.measurements = tuple((int(q), int(c)) for q, c in measurements)
        self.name = str(name)
        self._validate()
        self._lightcones: dict[tuple[tuple[int, ...], float], dict[tuple[int, int], float]] = {}

    def _validate(self) -> None:
        if self.num_qubits < 1:
            raise ValueError("a program needs at least one qubit")
        if not self.measurements:
            raise ValueError("a program needs at least one measurement")
        for g in self.gates:
            if g.name not in BASIS:
                raise ValueError(f"gate {g.name!r} is not in the basis {BASIS}")
            if any(not 0 <= q < self.num_qubits for q in g.qubits):
                raise ValueError(f"gate {g} acts outside the register")
            if g.name == "rzz" and (len(g.qubits) != 2 or g.qubits[0] == g.qubits[1]):
                raise ValueError(f"invalid rzz {g}")
        qs = [q for q, _ in self.measurements]
        cs = [c for _, c in self.measurements]
        if len(set(qs)) != len(qs) or len(set(cs)) != len(cs):
            raise ValueError("each qubit and each clbit may be measured at most once")
        if any(not 0 <= q < self.num_qubits for q in qs) or any(not 0 <= c < self.num_clbits for c in cs):
            raise ValueError("measurement outside the registers")

    @classmethod
    def from_circuit(
        cls, circuit: QuantumCircuit, optimization_level: int = 1, name: str | None = None
    ) -> "LogicalProgram":
        """Transpile a Qiskit circuit to the Canary basis; adds measure_all if unmeasured."""
        if circuit.parameters:
            raise ValueError("bind all circuit parameters before running Canary")
        qc = circuit.copy()
        if not any(inst.operation.name == "measure" for inst in qc.data):
            qc.measure_all(add_bits=True)
        tqc = transpile(
            qc,
            basis_gates=list(BASIS),
            optimization_level=optimization_level,
            seed_transpiler=0,
        )
        return cls._parse(tqc, name or circuit.name or "program")

    @classmethod
    def _parse(cls, qc: QuantumCircuit, name: str) -> "LogicalProgram":
        gates: list[Gate] = []
        measurements: list[tuple[int, int]] = []
        measured: set[int] = set()
        for inst in qc.data:
            op = inst.operation
            qs = tuple(qc.find_bit(q).index for q in inst.qubits)
            if op.name in _IGNORED:
                continue
            if op.name == "measure":
                measurements.append((qs[0], qc.find_bit(inst.clbits[0]).index))
                measured.add(qs[0])
                continue
            if measured.intersection(qs):
                raise ValueError("mid-circuit measurement is not supported")
            if op.name not in BASIS:
                raise ValueError(f"unsupported operation {op.name!r}")
            params = tuple(float(p) for p in op.params)
            if op.name == "rzz":
                phi = wrap_angle(params[0])
                if abs(phi) < EPS:
                    continue
                if abs(abs(phi) - math.pi) < EPS:
                    gates.append(Gate("rz", (qs[0],), (math.pi,)))
                    gates.append(Gate("rz", (qs[1],), (math.pi,)))
                    continue
                gates.append(Gate("rzz", qs, (phi,)))
                continue
            gates.append(Gate(op.name, qs, params))
        return cls(qc.num_qubits, qc.num_clbits, gates, measurements, name)

    @property
    def measured_qubits(self) -> tuple[int, ...]:
        """Logical qubits that are measured."""
        return tuple(q for q, _ in self.measurements)

    @property
    def interaction_weights(self) -> dict[tuple[int, int], int]:
        """Number of rzz gates per logical qubit pair."""
        counts: Counter = Counter()
        for g in self.gates:
            if g.name == "rzz":
                a, b = g.qubits
                counts[(min(a, b), max(a, b))] += 1
        return dict(counts)

    def weighted_interactions(self, angle_exponent: float = 0.0) -> dict[tuple[int, int], float]:
        """Angle-weighted rzz exposure per logical qubit pair (plain counts for exponent 0)."""
        out: dict[tuple[int, int], float] = {}
        for g in self.gates:
            if g.name == "rzz":
                a, b = g.qubits
                key = (min(a, b), max(a, b))
                out[key] = out.get(key, 0.0) + angle_weight(g.params[0], angle_exponent)
        return out

    @property
    def two_qubit_count(self) -> int:
        """Total number of rzz gates."""
        return sum(1 for g in self.gates if g.name == "rzz")

    def weights(self, angle_exponent: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Pair and ion exposure weights used by the assignment solver."""
        wp = np.zeros((self.num_qubits, self.num_qubits))
        for (a, b), w in self.weighted_interactions(angle_exponent).items():
            wp[a, b] += w
            wp[b, a] += w
        wi = np.zeros(self.num_qubits)
        for q in self.measured_qubits:
            wi[q] = 1.0
        return wp, wi

    def lightcone(self, qubits: Iterable[int], angle_exponent: float = 0.0) -> dict[tuple[int, int], float]:
        """Angle-weighted rzz exposure per logical pair inside the backward lightcone of the qubits.

        With the default exponent 0 every gate counts as one, so the values are plain gate counts.
        """
        key = tuple(sorted(set(int(q) for q in qubits)))
        cache_key = (key, float(angle_exponent))
        if cache_key not in self._lightcones:
            live = set(key)
            counts: dict[tuple[int, int], float] = {}
            for g in reversed(self.gates):
                if g.name == "rzz" and live.intersection(g.qubits):
                    a, b = g.qubits
                    k = (min(a, b), max(a, b))
                    counts[k] = counts.get(k, 0.0) + angle_weight(g.params[0], angle_exponent)
                    live.update(g.qubits)
            self._lightcones[cache_key] = counts
        return self._lightcones[cache_key]

    def observables(self, extra: Iterable[Sequence[int]] = ()) -> tuple[Observable, ...]:
        """Single-qubit Z and interacting-pair ZZ observables, plus optional extras."""
        clbit_of = dict(self.measurements)
        found: list[Observable] = [Observable((q,), (c,)) for q, c in self.measurements]
        for a, b in sorted(self.interaction_weights):
            if a in clbit_of and b in clbit_of:
                found.append(Observable((a, b), (clbit_of[a], clbit_of[b])))
        for qs in extra:
            qs = tuple(sorted(set(int(q) for q in qs)))
            if not qs or any(q not in clbit_of for q in qs):
                raise ValueError(f"observable {qs} contains unmeasured qubits")
            found.append(Observable(qs, tuple(clbit_of[q] for q in qs)))
        unique: dict[tuple[int, ...], Observable] = {}
        for o in found:
            unique.setdefault(o.qubits, o)
        return tuple(unique.values())

    def exposure(
        self, observable: Observable, assignment: Sequence[int], angle_exponent: float = 0.0
    ) -> dict[Resource, float]:
        """Upper-bound physical exposure of an observable under an assignment (every lightcone gate counted).

        Used only to predict relative damping for Tier-2 diagnostics; it is not a calibrated sensitivity.
        """
        out: dict[Resource, float] = {}
        for (a, b), n in self.lightcone(observable.qubits, angle_exponent).items():
            r = pair(assignment[a], assignment[b])
            out[r] = out.get(r, 0.0) + float(n)
        for q in observable.qubits:
            r = ion(assignment[q])
            out[r] = out.get(r, 0.0) + 1.0
        return out

    def total_exposure(self, assignment: Sequence[int], angle_exponent: float = 0.0) -> dict[Resource, float]:
        """Exposure of the whole circuit: every rzz gate and every measured ion once."""
        out: dict[Resource, float] = {}
        for (a, b), w in self.weighted_interactions(angle_exponent).items():
            r = pair(assignment[a], assignment[b])
            out[r] = out.get(r, 0.0) + w
        for q in self.measured_qubits:
            r = ion(assignment[q])
            out[r] = out.get(r, 0.0) + 1.0
        return out

    def used_resources(self, assignment: Sequence[int]) -> set[Resource]:
        """Resources the program touches under an assignment."""
        used = {pair(assignment[a], assignment[b]) for a, b in self.interaction_weights}
        used.update(ion(assignment[q]) for q in self.measured_qubits)
        return used

    def to_circuit(self) -> QuantumCircuit:
        """Rebuild the logical circuit, including measurements."""
        qc = QuantumCircuit(self.num_qubits, self.num_clbits, name=self.name)
        for g in self.gates:
            getattr(qc, g.name)(*g.params, *g.qubits)
        for q, c in self.measurements:
            qc.measure(q, c)
        return qc

    def __repr__(self) -> str:
        return (
            f"LogicalProgram(name={self.name!r}, qubits={self.num_qubits}, "
            f"rzz={self.two_qubit_count}, measured={len(self.measurements)})"
        )
