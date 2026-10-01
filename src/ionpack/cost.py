"""Gate counting, per-circuit price models and the saving from packing.

Models, with c = shots * (n1q * price_1q + n2q * price_2q):
  ``floor``:    max(minimum, c)  -- IonQ: small jobs are "rounded up to this minimum value".
  ``additive``: minimum + c      -- the literal Azure Quantum formula AQT = m + ...
  ``per_shot``: per_task + per_shot * shots  -- AWS Braket.
There is no IonQ-direct preset: rates depend on the customer's agreement; supply your own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import ControlFlowOp


@dataclass(frozen=True)
class Pricing:
    """A per-circuit price model (see module docstring)."""

    model: Literal["floor", "additive", "per_shot"]
    price_1q: float = 0.0
    price_2q: float = 0.0
    minimum: float = 0.0
    per_task: float = 0.0
    per_shot: float = 0.0
    source: str = ""

    def __post_init__(self) -> None:
        if self.model not in ("floor", "additive", "per_shot"):
            raise ValueError(f"unknown pricing model {self.model!r}")
        if min(self.price_1q, self.price_2q, self.minimum, self.per_task, self.per_shot) < 0:
            raise ValueError("prices must be non-negative")


def count_gates(circuit: QuantumCircuit) -> tuple[int, int]:
    """(number of 1-qubit, number of 2-qubit) instructions, ignoring ``measure`` and ``barrier``."""
    n1q = n2q = 0
    for inst in circuit.data:
        name = inst.operation.name
        if isinstance(inst.operation, ControlFlowOp) or name in ("reset", "delay"):
            raise ValueError(f"{name!r} cannot be priced")
        if name in ("measure", "barrier"):
            continue
        if len(inst.qubits) == 1:
            n1q += 1
        elif len(inst.qubits) == 2:
            n2q += 1
        else:
            raise ValueError(f"{name!r} acts on {len(inst.qubits)} qubits; decompose to 1- and 2-qubit gates first")
    return n1q, n2q


def circuit_cost(n1q: int, n2q: int, shots: int, pricing: Pricing) -> float:
    """Price of one circuit with ``n1q`` 1-qubit and ``n2q`` 2-qubit gates run for ``shots`` shots."""
    if min(n1q, n2q, shots) < 0:
        raise ValueError("gate counts and shots must be non-negative")
    if pricing.model == "per_shot":
        return pricing.per_task + pricing.per_shot * shots
    c = shots * (n1q * pricing.price_1q + n2q * pricing.price_2q)
    return max(pricing.minimum, c) if pricing.model == "floor" else pricing.minimum + c


def packed_savings(gate_counts: Sequence[tuple[int, int]], shots: int, pricing: Pricing) -> float:
    """[sum_i cost(block_i)] / cost(packed circuit), all at the same ``shots``.

    For k identical blocks of gate cost c under ``floor`` with minimum m: the saving is k if
    k c <= m, m / c if c < m < k c, and 1 if c >= m; i.e. min(k, m / c) when c < m, else 1.
    Under ``per_shot`` it is exactly k.
    """
    if len(gate_counts) == 0:
        raise ValueError("gate_counts must not be empty")
    solo = sum(circuit_cost(n1, n2, shots, pricing) for n1, n2 in gate_counts)
    packed = circuit_cost(sum(g[0] for g in gate_counts), sum(g[1] for g in gate_counts), shots, pricing)
    if packed == 0:
        raise ValueError("packed cost is zero; savings undefined")
    return solo / packed


AZURE_FORTE_RATES: dict = {
    "price_1q": 0.0001645,  # USD per 1-qubit gate-shot
    "price_2q": 0.001121,  # USD per 2-qubit gate-shot
    "minimum_em_off": 25.7899,  # USD minimum per program, error mitigation off
    "minimum_em_on": 168.195,  # USD minimum per program, error mitigation on (default)
    "source": "https://learn.microsoft.com/azure/quantum/pricing (IonQ Forte 1 / Forte Enterprise 1; retrieved 2026-10-01)",
}

BRAKET_FORTE: Pricing = Pricing(
    model="per_shot",
    per_task=0.30,
    per_shot=0.08,
    source="https://aws.amazon.com/braket/pricing/ (IonQ Forte; retrieved 2026-10-01)",
)
