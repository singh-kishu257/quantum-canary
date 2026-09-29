"""Compile logical programs to IonQ Forte native gates on chosen ions."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from qiskit_ionq import GPI2Gate, GPIGate, ZZGate

from .program import EPS, Gate, LogicalProgram, wrap_angle

_PAULIS = ("i", "x", "y", "z")
_ANTICOMMUTE_Z = frozenset({"x", "y"})
_OPTIMIZE_1Q = PassManager([Optimize1qGatesDecomposition(basis=["rz", "sx", "x"])])
_ALLOWED = frozenset({"rz", "sx", "x", "rzz"})


def twirl_gates(gates: Sequence[Gate], rng: np.random.Generator) -> list[Gate]:
    """Pauli-twirl every rzz, negating its angle when the Pauli anticommutes with ZZ."""
    out: list[Gate] = []
    for g in gates:
        if g.name != "rzz":
            out.append(g)
            continue
        pa = _PAULIS[int(rng.integers(4))]
        pb = _PAULIS[int(rng.integers(4))]
        sign = -1.0 if (pa in _ANTICOMMUTE_Z) != (pb in _ANTICOMMUTE_Z) else 1.0
        frame = [Gate(p, (q,)) for p, q in ((pa, g.qubits[0]), (pb, g.qubits[1])) if p != "i"]
        out.extend(frame)
        out.append(Gate("rzz", g.qubits, (sign * g.params[0],)))
        out.extend(frame)
    return out


def normalize_gates(gates: Sequence[Gate]) -> list[Gate]:
    """Rewrite every rzz so its angle lies in (0, pi/2] using virtual Z and X conjugation."""
    out: list[Gate] = []
    for g in gates:
        if g.name != "rzz":
            out.append(g)
            continue
        a, b = g.qubits
        phi = wrap_angle(g.params[0])
        if abs(phi) < EPS:
            continue
        flip = False
        if phi > math.pi / 2:
            phi -= math.pi
            flip = True
        elif phi < -math.pi / 2:
            phi += math.pi
            flip = True
        if phi < 0:
            out.extend([Gate("x", (a,)), Gate("rzz", (a, b), (-phi,)), Gate("x", (a,))])
        else:
            out.append(Gate("rzz", (a, b), (phi,)))
        if flip:
            out.extend([Gate("rz", (a,), (math.pi,)), Gate("rz", (b,), (math.pi,))])
    return out


def optimize_single_qubit(gates: Sequence[Gate], num_qubits: int) -> list[Gate]:
    """Merge single-qubit runs into the (rz, sx, x) basis."""
    qc = QuantumCircuit(num_qubits)
    for g in gates:
        getattr(qc, g.name)(*g.params, *g.qubits)
    opt = _OPTIMIZE_1Q.run(qc)
    out: list[Gate] = []
    for inst in opt.data:
        name = inst.operation.name
        if name not in _ALLOWED:
            raise RuntimeError(f"single-qubit optimisation produced unexpected gate {name!r}")
        qs = tuple(opt.find_bit(q).index for q in inst.qubits)
        out.append(Gate(name, qs, tuple(float(p) for p in inst.operation.params)))
    return out


def _phase(frame: float) -> float:
    return float((-frame / (2.0 * math.pi)) % 1.0)


def lower_to_native(
    gates: Sequence[Gate],
    assignment: Sequence[int],
    measurements: Sequence[tuple[int, int]],
    num_clbits: int,
    name: str = "canary",
) -> QuantumCircuit:
    """Lower (rz, sx, x, rzz) gates to GPi/GPi2/ZZ on ions, tracking virtual Z frames."""
    ions = [int(i) for i in assignment]
    if any(i < 0 for i in ions) or len(set(ions)) != len(ions):
        raise ValueError("assignment must map logical qubits to distinct non-negative ions")
    width = max(ions) + 1
    qc = QuantumCircuit(QuantumRegister(width, "q"), ClassicalRegister(num_clbits, "c"), name=name)
    frame = [0.0] * len(ions)
    for g in gates:
        if g.name == "rz":
            frame[g.qubits[0]] += g.params[0]
        elif g.name == "sx":
            q = g.qubits[0]
            qc.append(GPI2Gate(_phase(frame[q])), [ions[q]])
        elif g.name == "x":
            q = g.qubits[0]
            qc.append(GPIGate(_phase(frame[q])), [ions[q]])
        elif g.name == "rzz":
            a, b = g.qubits
            theta = g.params[0] / (2.0 * math.pi)
            if not 0.0 < theta <= 0.25 + 1e-12:
                raise ValueError(f"ZZ angle {theta} turns outside (0, 0.25]; normalise first")
            qc.append(ZZGate(min(theta, 0.25)), [ions[a], ions[b]])
        else:
            raise ValueError(f"cannot lower gate {g.name!r}")
    for q, c in measurements:
        qc.measure(ions[q], c)
    return qc


def compile_native(
    program: LogicalProgram | QuantumCircuit,
    assignment: Sequence[int],
    rng: np.random.Generator | None = None,
    twirl: bool = True,
    name: str | None = None,
) -> QuantumCircuit:
    """Compile a program or circuit onto ions in native gates, optionally Pauli-twirled."""
    if not isinstance(program, LogicalProgram):
        program = LogicalProgram.from_circuit(program)
    if len(assignment) != program.num_qubits:
        raise ValueError("assignment length must equal the number of logical qubits")
    gates: Sequence[Gate] = program.gates
    if twirl:
        gates = twirl_gates(gates, rng if rng is not None else np.random.default_rng())
    gates = normalize_gates(gates)
    gates = optimize_single_qubit(gates, program.num_qubits)
    return lower_to_native(
        gates, assignment, program.measurements, program.num_clbits, name or program.name
    )
