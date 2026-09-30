"""Execution back-ends returning counts for lists of native circuits."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from qiskit import QuantumCircuit, transpile

_NATIVE = ("gpi", "gpi2", "zz", "ms")


class Executor(ABC):
    """Runs a list of circuits with a common shot count in one job."""

    last_job_ids: tuple[str, ...] = ()

    @property
    @abstractmethod
    def ions(self) -> tuple[int, ...]:
        """Ion indices Canary may use."""

    @property
    def max_shots(self) -> int | None:
        """Maximum shots per circuit, or None if unbounded."""
        return None

    @abstractmethod
    def run(self, circuits: Sequence[QuantumCircuit], shots: int) -> list[Mapping[str, float]]:
        """Execute circuits and return one counts (or probabilities) mapping per circuit."""


def _ionq_backend(name: str, token: str | None) -> Any:
    from qiskit_ionq import IonQProvider

    return IonQProvider(token).get_backend(name, gateset="native")


class IonQExecutor(Executor):
    """IonQ cloud QPU or simulator using the native (GPi, GPi2, ZZ) gateset without debiasing.

    IonQ runs native-gate circuits exactly as submitted, with its compiler and optimizer turned off and error
    mitigation off by default, and a test on qpu.forte-1 confirmed that native circuits address the requested ion
    indices without optimizing the gate sequence. Debiasing is refused because it re-runs circuits under different
    qubit assignments, which would invalidate per-ion attribution.
    """

    def __init__(
        self,
        backend: Any,
        ions: Sequence[int] | None = None,
        job_name: str = "quantum-canary",
        **run_options: Any,
    ) -> None:
        if backend.gateset() != "native":
            raise ValueError('IonQ backend must be created with gateset="native"')
        self.backend = backend
        self.simulator = backend.name.startswith("ionq_simulator")
        if ions is None:
            n = int(backend.num_qubits)
            if n <= 4:
                warnings.warn(
                    f"backend reports only {n} qubits; pass ions= explicitly", UserWarning, stacklevel=2
                )
            ions = range(n)
        self._ions = tuple(sorted({int(i) for i in ions}))
        if len(self._ions) < 2:
            raise ValueError("at least two ions are required")
        self.job_name = job_name
        self.run_options = dict(run_options)
        if not self.simulator:
            from qiskit_ionq import ErrorMitigation

            self.run_options.setdefault("error_mitigation", ErrorMitigation.NO_DEBIASING)
            if self.run_options["error_mitigation"] is not ErrorMitigation.NO_DEBIASING:
                raise ValueError("Canary requires debiasing to be off: it remaps qubits to other ions")

    @classmethod
    def qpu(
        cls, device: str = "forte-1", token: str | None = None, ions: Sequence[int] | None = None, **run_options: Any
    ) -> "IonQExecutor":
        """Executor for an IonQ QPU such as 'forte-1' or 'forte-enterprise-1'."""
        return cls(_ionq_backend(f"qpu.{device}", token), ions=ions, **run_options)

    @classmethod
    def simulator(
        cls,
        noise_model: str = "forte-1",
        token: str | None = None,
        ions: Sequence[int] | None = None,
        **run_options: Any,
    ) -> "IonQExecutor":
        """Executor for the IonQ cloud simulator with a named noise model."""
        run_options.setdefault("noise_model", noise_model)
        return cls(_ionq_backend("simulator", token), ions=ions, **run_options)

    @property
    def ions(self) -> tuple[int, ...]:
        return self._ions

    @property
    def max_shots(self) -> int | None:
        return None if self.simulator else getattr(self.backend, "_max_shots", None)

    def run(self, circuits: Sequence[QuantumCircuit], shots: int) -> list[Mapping[str, float]]:
        allowed = set(self._ions)
        for qc in circuits:
            used = {qc.find_bit(q).index for inst in qc.data for q in inst.qubits}
            if not used <= allowed:
                raise ValueError(f"circuit {qc.name!r} uses ions outside {self._ions}")
        job = self.backend.run(list(circuits), shots=int(shots), name=self.job_name, **self.run_options)
        self.last_job_ids = (str(job.job_id()),)
        result = job.result()
        return [dict(result.get_counts(i)) for i in range(len(circuits))]


class BackendExecutor(Executor):
    """Generic Qiskit backend (e.g. qiskit-aer) running IonQ native circuits via gate definitions."""

    def __init__(self, backend: Any, ions: Sequence[int], **run_options: Any) -> None:
        self.backend = backend
        self._ions = tuple(sorted({int(i) for i in ions}))
        if len(self._ions) < 2:
            raise ValueError("at least two ions are required")
        self.run_options = dict(run_options)

    @property
    def ions(self) -> tuple[int, ...]:
        return self._ions

    def run(self, circuits: Sequence[QuantumCircuit], shots: int) -> list[Mapping[str, float]]:
        compiled = [
            transpile(
                qc.decompose(gates_to_decompose=list(_NATIVE)),
                backend=self.backend,
                optimization_level=0,
                initial_layout=list(range(qc.num_qubits)),
            )
            for qc in circuits
        ]
        job = self.backend.run(compiled, shots=int(shots), **self.run_options)
        self.last_job_ids = (str(job.job_id()),)
        result = job.result()
        return [dict(result.get_counts(i)) for i in range(len(circuits))]
