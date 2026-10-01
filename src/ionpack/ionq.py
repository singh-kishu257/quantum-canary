"""Thin helpers for submitting packed circuits with qiskit-ionq (optional extra ``ionpack[ionq]``).

``qiskit_ionq`` is imported lazily inside the functions.
"""

from __future__ import annotations

import warnings

from .pack import PackedCircuit, is_sequential


def run_packed(backend, packed: PackedCircuit, shots: int, *, debias: bool = False, memory: bool = False, **options):
    """Submit ``packed.circuit`` unchanged to an IonQ backend; returns the IonQ job.

    Use a native-gateset backend, e.g. ``provider.get_backend("qpu.forte-1", gateset="native")``.
    """
    from qiskit_ionq import ErrorMitigation

    if backend.gateset() != "native":
        warnings.warn(
            "backend gateset is not 'native': IonQ's compiler may reorder gates across blocks, "
            "breaking sequential execution",
            stacklevel=2,
        )
    if not is_sequential(packed):
        raise ValueError("packed circuit is not sequential (was it transpiled or edited after pack?)")
    if debias:
        warnings.warn(
            "debiasing permutes the qubit assignment: gap and sentinel geometry is not preserved, "
            "and results must be aggregated by averaging",
            stacklevel=2,
        )
    mitigation = ErrorMitigation.DEBIASING if debias else ErrorMitigation.NO_DEBIASING
    return backend.run(packed.circuit, shots=shots, error_mitigation=mitigation, memory=memory, **options)


def packed_counts(job) -> dict[str, int]:
    """Joint counts of a finished job, with ``sharpen=False`` (averaging, not plurality voting)."""
    return job.result(sharpen=False).get_counts()
