"""Sequential packing of independent circuits onto disjoint ion blocks.

Workflow: transpile each sub-circuit to IonQ native gates first, then ``pack``, then submit
the packed circuit unchanged. Never ``transpile`` a packed circuit: that can interleave the
blocks and destroy the sequential order this module guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import ControlFlowOp

from .layout import Layout

NATIVE_GATES: frozenset[str] = frozenset({"gpi", "gpi2", "ms", "zz"})


@dataclass(frozen=True)
class PackedCircuit:
    """A packed circuit plus the bookkeeping needed to split its counts."""

    circuit: QuantumCircuit
    layout: Layout
    block_clbits: tuple[tuple[int, ...], ...]  # packed clbits of block i, in block i's own clbit order
    sentinel_clbits: tuple[int, ...]  # packed clbits of measured sentinels; () if not measured


def check_circuit(circuit: QuantumCircuit, native: bool = False) -> dict[int, int]:
    """Validate a sub-circuit and return its terminal-measurement map ``{clbit: qubit}``.

    Rejects control flow, ``reset``, ``delay``, classical bits on non-measure instructions,
    repeated measurement of a qubit or clbit, any non-barrier instruction on a qubit after its
    measurement, instructions on no qubits and, if ``native``, gates outside ``NATIVE_GATES``.
    Unmeasured clbits are allowed; they read 0.
    """
    meas: dict[int, int] = {}
    measured: set[int] = set()
    for inst in circuit.data:
        op, name = inst.operation, inst.operation.name
        qubits = [circuit.find_bit(q).index for q in inst.qubits]
        clbits = [circuit.find_bit(c).index for c in inst.clbits]
        if isinstance(op, ControlFlowOp):
            raise ValueError(f"control flow ({name!r}) is not supported")
        if name in ("reset", "delay"):
            raise ValueError(f"{name!r} is not supported (IonQ has no delay; no mid-circuit reset)")
        if name == "barrier":
            continue
        if not qubits:
            raise ValueError(f"instruction {name!r} acts on no qubits")
        if name == "measure":
            if qubits[0] in measured:
                raise ValueError(f"qubit {qubits[0]} is measured twice")
            if clbits[0] in meas:
                raise ValueError(f"clbit {clbits[0]} is written twice")
            measured.add(qubits[0])
            meas[clbits[0]] = qubits[0]
            continue
        if measured.intersection(qubits):
            raise ValueError(f"{name!r} acts on qubit(s) {qubits} after their measurement")
        if clbits:
            raise ValueError(f"non-measure instruction {name!r} has classical bits")
        if native and name not in NATIVE_GATES:
            raise ValueError(f"gate {name!r} is not an IonQ native gate {sorted(NATIVE_GATES)}")
    return meas


def pack(
    circuits: Sequence[QuantumCircuit],
    layout: Layout,
    *,
    native: bool = False,
    measure_sentinels: bool = True,
    barriers: bool = True,
    name: str = "ionpack",
) -> PackedCircuit:
    """Pack ``circuits[i]`` onto ``layout.blocks[i]``; blocks run strictly one after another.

    All gates of block 0 come first, then block 1, and so on; all measurements are emitted at
    the end. Block ``i`` owns packed clbits ``off_i .. off_i + num_clbits_i - 1``; sentinel
    clbits follow. Input barriers and global phase are dropped.
    """
    if len(circuits) != len(layout.blocks):
        raise ValueError(f"{len(circuits)} circuits for {len(layout.blocks)} blocks")
    maps = []
    for i, (circ, block) in enumerate(zip(circuits, layout.blocks)):
        if circ.num_qubits != len(block):
            raise ValueError(f"circuit {i} has {circ.num_qubits} qubits but block {i} has {len(block)} ions")
        maps.append(check_circuit(circ, native=native))
    offsets, n_clbits = [], 0
    for circ in circuits:
        offsets.append(n_clbits)
        n_clbits += circ.num_clbits
    n_sent = len(layout.sentinels) if measure_sentinels else 0
    out = QuantumCircuit(layout.width, n_clbits + n_sent, name=name)
    for circ, block in zip(circuits, layout.blocks):
        for inst in circ.data:
            if inst.operation.name not in ("measure", "barrier"):
                out.append(inst.operation, [block[circ.find_bit(q).index] for q in inst.qubits])
        if barriers:
            out.barrier()
    for block, off, meas in zip(layout.blocks, offsets, maps):
        for j in sorted(meas):
            out.measure(block[meas[j]], off + j)
    sentinel_clbits = tuple(range(n_clbits, n_clbits + n_sent))
    for ion, clbit in zip(layout.sentinels if measure_sentinels else (), sentinel_clbits):
        out.measure(ion, clbit)
    block_clbits = tuple(tuple(range(off, off + c.num_clbits)) for off, c in zip(offsets, circuits))
    return PackedCircuit(out, layout, block_clbits, sentinel_clbits)


def is_sequential(packed: PackedCircuit) -> bool:
    """True iff every gate (non-barrier, non-measure) acts within one block, the block index is
    non-decreasing in program order, and no gate follows the first measurement."""
    owner = {ion: b for b, block in enumerate(packed.layout.blocks) for ion in block}
    circ, current, measuring = packed.circuit, 0, False
    for inst in circ.data:
        name = inst.operation.name
        if name == "barrier":
            continue
        if name == "measure":
            measuring = True
            continue
        if measuring:
            return False
        blocks = {owner.get(circ.find_bit(q).index) for q in inst.qubits}
        if len(blocks) != 1 or None in blocks:
            return False
        b = blocks.pop()
        if b < current:
            return False
        current = b
    return True
