"""Compile logical programs to IonQ Forte native gates on chosen ions.

Pipeline: normalise every ZZ rotation to a positive physical angle of at most 0.25 turns (the only ZZ angles
Canary submits, since IonQ documents no range for the native ZZ angle), dress each ZZ gate with Pauli frames,
optionally twirl the measurement, merge single-qubit runs, and lower to GPi/GPi2/ZZ with virtual-Z frame tracking.

Twirl modes
-----------
``"rc"`` (default) is randomized compiling. ZZ(1/4 turn) is a Clifford gate G, so it is dressed with a uniformly
random Pauli P before and G P G^dagger after; the error channel of the gate is then Pauli-twirled, and in
particular a coherent ZZ over-rotation becomes stochastic ZZ errors. A non-Clifford ZZ(theta) can only be dressed
with Paulis that commute with Z(x)Z (the angle cannot be negated physically), which randomises error components
that anticommute with some commuting frame but leaves ZZ over-rotation untouched. Canary measures that residual
coherent error with phase probes and can pre-compensate it (``angle_offset``).

``"commutant"`` uses commuting frames on every ZZ gate; it preserves the ZZ phase and is used by phase probes.
``"none"`` disables twirling.

Z components of the frames are virtual phase updates, so they do not act on errors fixed in the laboratory frame.
ZZ over-rotation of a Clifford ZZ gate is still randomised because G P G^dagger puts a physical X- or Y-type pulse
on exactly one of the two ions with probability 1/2, which flips the sign of a ZZ error.

Measurement twirling applies X to chosen measured qubits just before measurement (merged into the last pulse);
the caller flips the corresponding bits back. Averaged over complementary flip sets, any readout error becomes a
multiplicative factor on Z-string expectations (van den Berg, Minev and Temme, Phys. Rev. A 105, 032620, 2022).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from qiskit_ionq import GPI2Gate, GPIGate, ZZGate

from .program import EPS, Gate, LogicalProgram, wrap_angle
from .results import unflip_frequencies

TWIRL_MODES: tuple[str, ...] = ("rc", "commutant", "none")
_PAULIS = ("i", "x", "y", "z")
_Z_TIMES = {"i": "z", "x": "y", "y": "x", "z": "i"}
_COMMUTANT = (("i", "i"), ("i", "z"), ("z", "i"), ("z", "z"), ("x", "x"), ("x", "y"), ("y", "x"), ("y", "y"))
_CLIFFORD_ZZ = math.pi / 2
_OPTIMIZE_1Q = PassManager([Optimize1qGatesDecomposition(basis=["rz", "sx", "x"])])
_ALLOWED = frozenset({"rz", "sx", "x", "rzz"})

AngleOffset = Callable[[int, int, float], float]


def resolve_twirl(mode: str | bool | None) -> str:
    """Normalise a twirl setting: True -> 'rc', False/None -> 'none'."""
    if mode is True:
        return "rc"
    if mode is False or mode is None:
        return "none"
    if mode not in TWIRL_MODES:
        raise ValueError(f"twirl must be one of {TWIRL_MODES} or a bool, not {mode!r}")
    return str(mode)


def _anticommutes_with_zz(pa: str, pb: str) -> bool:
    return (pa in ("x", "y")) != (pb in ("x", "y"))


def _frame(pa: str, pb: str, a: int, b: int) -> list[Gate]:
    return [Gate(p, (q,)) for p, q in ((pa, a), (pb, b)) if p != "i"]


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


def twirl_gates(gates: Sequence[Gate], rng: np.random.Generator, mode: str | bool = "rc") -> list[Gate]:
    """Dress every normalised rzz with Pauli frames (see the module docstring for the modes)."""
    mode = resolve_twirl(mode)
    if mode == "none":
        return list(gates)
    out: list[Gate] = []
    for g in gates:
        if g.name != "rzz":
            out.append(g)
            continue
        phi = float(g.params[0])
        if not 0.0 < phi <= _CLIFFORD_ZZ + EPS:
            raise ValueError("twirl_gates expects rzz angles normalised to (0, pi/2]; call normalize_gates first")
        a, b = g.qubits
        if mode == "rc" and abs(phi - _CLIFFORD_ZZ) < EPS:
            pa = _PAULIS[int(rng.integers(4))]
            pb = _PAULIS[int(rng.integers(4))]
            post = (_Z_TIMES[pa], _Z_TIMES[pb]) if _anticommutes_with_zz(pa, pb) else (pa, pb)
        else:
            pa, pb = _COMMUTANT[int(rng.integers(len(_COMMUTANT)))]
            post = (pa, pb)
        out.extend(_frame(pa, pb, a, b))
        out.append(g)
        out.extend(_frame(post[0], post[1], a, b))
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
    angle_offset: AngleOffset | None = None,
) -> QuantumCircuit:
    """Lower (rz, sx, x, rzz) gates to GPi/GPi2/ZZ on ions, tracking virtual Z frames.

    ``angle_offset(ion_a, ion_b, turns)`` returns a correction in turns added to each ZZ angle; corrections that
    would leave (0, 0.25] are clipped at 0.25 or skipped.
    """
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
            theta = min(theta, 0.25)
            if angle_offset is not None:
                adjusted = theta + float(angle_offset(ions[a], ions[b], theta))
                if adjusted > 0.25:
                    theta = 0.25
                elif adjusted > 0.0:
                    theta = adjusted
            qc.append(ZZGate(theta), [ions[a], ions[b]])
        else:
            raise ValueError(f"cannot lower gate {g.name!r}")
    for q, c in measurements:
        qc.measure(ions[q], c)
    return qc


@dataclass
class CompiledCircuit:
    """A native circuit plus the clbits whose outcomes were flipped by the measurement twirl."""

    circuit: QuantumCircuit
    flips: tuple[int, ...] = ()

    def untwirl(self, frequencies: Mapping[str, float]) -> dict[str, float]:
        """Map measured frequencies back to logical outcomes."""
        if not self.flips:
            return dict(frequencies)
        return unflip_frequencies(frequencies, self.flips, self.circuit.num_clbits)


def compile_twirled(
    program: LogicalProgram | QuantumCircuit,
    assignment: Sequence[int],
    rng: np.random.Generator | None = None,
    twirl: str | bool = "rc",
    flips: Iterable[int] = (),
    angle_offset: AngleOffset | None = None,
    name: str | None = None,
) -> CompiledCircuit:
    """Compile onto ions with Pauli-frame twirling and a measurement twirl on the given clbits."""
    if not isinstance(program, LogicalProgram):
        program = LogicalProgram.from_circuit(program)
    if len(assignment) != program.num_qubits:
        raise ValueError("assignment length must equal the number of logical qubits")
    rng = rng if rng is not None else np.random.default_rng()
    qubit_of = {c: q for q, c in program.measurements}
    flip_set = sorted({int(c) for c in flips})
    if any(c not in qubit_of for c in flip_set):
        raise ValueError("measurement-twirl flips must refer to measured clbits")
    gates = normalize_gates(program.gates)
    gates = twirl_gates(gates, rng, twirl)
    gates = gates + [Gate("x", (qubit_of[c],)) for c in flip_set]
    gates = optimize_single_qubit(gates, program.num_qubits)
    circuit = lower_to_native(
        gates, assignment, program.measurements, program.num_clbits, name or program.name, angle_offset
    )
    return CompiledCircuit(circuit, tuple(flip_set))


def compile_native(
    program: LogicalProgram | QuantumCircuit,
    assignment: Sequence[int],
    rng: np.random.Generator | None = None,
    twirl: str | bool = True,
    name: str | None = None,
    angle_offset: AngleOffset | None = None,
) -> QuantumCircuit:
    """Compile a program or circuit onto ions in native gates (no measurement twirl; outcomes are logical)."""
    return compile_twirled(program, assignment, rng, twirl, (), angle_offset, name).circuit
