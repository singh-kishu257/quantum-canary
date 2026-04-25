"""
Canary circuits for quantum hardware drift detection.

Each circuit is a lightweight quantum probe that measures one specific
type of noise channel:

    Bell state    — sensitive to 2-qubit entanglement and CZ gate error
    Coherence     — sensitive to T2 dephasing during 20 H-gate identity
    Gate error    — sensitive to single-qubit gate error during 20 X-gate identity

The fidelity values extracted from these circuits form the three input
features for the MLP drift classifier.
"""

from __future__ import annotations
from qiskit import QuantumCircuit


def build_bell_circuit() -> QuantumCircuit:
    """
    Build a 2-qubit Bell state preparation circuit.

    Ideal output state: (|00⟩ + |11⟩) / √2
    On a perfect device, measurement yields 50% '00' and 50% '11'.
    Any '01' or '10' counts indicate noise.

    Returns
    -------
    QuantumCircuit
        2-qubit, 2-classical-bit circuit ready for transpilation.
    """
    qc = QuantumCircuit(2, 2, name="bell_canary")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def build_coherence_circuit(n_pairs: int = 10) -> QuantumCircuit:
    """
    Build a coherence probe — n_pairs of H·H = identity gates.

    H·H = I, so the ideal output is |0⟩ with 100% probability.
    Total runtime ≈ 2 * n_pairs * 56 ns ≈ 1120 ns on IBM Heron r2.
    Sensitive to T2 dephasing accumulated during execution.

    Parameters
    ----------
    n_pairs : int, optional
        Number of H·H pairs (default 10 = 20 H gates total).

    Returns
    -------
    QuantumCircuit
        1-qubit, 1-classical-bit circuit.
    """
    qc = QuantumCircuit(1, 1, name="coherence_canary")
    for _ in range(n_pairs):
        qc.h(0)
        qc.h(0)
    qc.measure(0, 0)
    return qc


def build_gate_error_circuit(n_gates: int = 20) -> QuantumCircuit:
    """
    Build a gate-error probe — n_gates X-gates back-to-back.

    Even number of X gates = identity, so ideal output is |0⟩.
    Errors compound: with per-gate error ε, total error ≈ 1 - (1 - ε)^n.
    Amplifies small per-gate errors into observable signal.

    Parameters
    ----------
    n_gates : int, optional
        Number of X gates (must be even). Default 20.

    Returns
    -------
    QuantumCircuit
        1-qubit, 1-classical-bit circuit.

    Raises
    ------
    ValueError
        If n_gates is odd (would not return to |0⟩ in ideal case).
    """
    if n_gates % 2 != 0:
        raise ValueError(f"n_gates must be even (got {n_gates})")
    qc = QuantumCircuit(1, 1, name="gate_error_canary")
    for _ in range(n_gates):
        qc.x(0)
    qc.measure(0, 0)
    return qc


def build_canary_circuits(
    n_coh_pairs: int = 10,
    n_gate_reps: int = 20,
) -> list[QuantumCircuit]:
    """
    Build all three canary circuits as a list ready for batch submission.

    Parameters
    ----------
    n_coh_pairs : int, optional
        H·H pairs in the coherence probe (default 10).
    n_gate_reps : int, optional
        X-gate repetitions in the gate-error probe (default 20, must be even).

    Returns
    -------
    list[QuantumCircuit]
        [bell_circuit, coherence_circuit, gate_error_circuit]
    """
    return [
        build_bell_circuit(),
        build_coherence_circuit(n_coh_pairs),
        build_gate_error_circuit(n_gate_reps),
    ]


# ── Fidelity extraction helpers ──────────────────────────────────────────────
def fidelity_bell(counts: dict) -> float:
    """Fidelity of Bell state — fraction of '00' and '11' outcomes."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return (counts.get("00", 0) + counts.get("11", 0)) / total


def fidelity_single(counts: dict) -> float:
    """Fidelity of single-qubit identity — fraction of '0' outcomes."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return counts.get("0", 0) / total