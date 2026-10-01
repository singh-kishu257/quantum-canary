"""Offline qiskit-ionq checks. Never calls backend.run on a real backend; no network is needed."""

import json
import warnings

import pytest
from qiskit import QuantumCircuit

from ionpack import PackedCircuit, make_layout, pack, packed_counts, run_packed

qi = pytest.importorskip("qiskit_ionq")
from qiskit_ionq.helpers import qiskit_to_ionq  # noqa: E402


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Refuse every HTTP request made through ``requests`` (qiskit-ionq's only transport)."""
    import requests

    def refuse(*args, **kwargs):
        raise requests.exceptions.ConnectionError("network access is disabled in ionpack tests")

    monkeypatch.setattr(requests.sessions.Session, "request", refuse)


def native_block():
    qc = QuantumCircuit(2, 2)
    qc.append(qi.GPI2Gate(0), [0])
    qc.append(qi.ZZGate(0.25), [0, 1])
    qc.append(qi.GPIGate(0.5), [1])
    qc.measure([0, 1], [0, 1])
    return qc


def native_packed():
    return pack([native_block(), native_block()], make_layout([2, 2], 36, 1), native=True)


def test_t13_serialization_preserves_sequential_order():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        backend = qi.IonQProvider(token="dummy").get_backend("qpu.forte-1", gateset="native")
    assert backend.name == "ionq_qpu.forte-1" and backend.gateset() == "native"
    payload = json.loads(qiskit_to_ionq(native_packed().circuit, backend, passed_args={"shots": 100}))
    assert payload["input"]["gateset"] == "native"
    assert payload["input"]["qubits"] == 5
    gates = payload["input"]["circuit"]
    order = [(g["gate"], g.get("targets", [g.get("target")])) for g in gates]
    assert order == [("gpi2", [0]), ("zz", [0, 1]), ("gpi", [1]), ("gpi2", [3]), ("zz", [3, 4]), ("gpi", [4])]
    assert all(g["gate"] != "barrier" for g in gates)


class StubBackend:
    def __init__(self, gateset):
        self._gateset, self.calls = gateset, []

    def gateset(self):
        return self._gateset

    def run(self, circuit, **kwargs):
        self.calls.append((circuit, kwargs))
        return "job"


def test_run_packed_options_and_checks():
    p = native_packed()
    backend = StubBackend("native")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert run_packed(backend, p, 200) == "job"
    circuit, kwargs = backend.calls[0]
    assert circuit is p.circuit
    assert kwargs == {"shots": 200, "error_mitigation": qi.ErrorMitigation.NO_DEBIASING, "memory": False}
    with pytest.warns(UserWarning, match="debiasing"):
        run_packed(backend, p, 3000, debias=True, memory=True)
    assert backend.calls[1][1]["error_mitigation"] is qi.ErrorMitigation.DEBIASING
    assert backend.calls[1][1]["memory"] is True
    with pytest.warns(UserWarning, match="native"):
        run_packed(StubBackend("qis"), p, 100)


def test_run_packed_rejects_non_sequential():
    p = native_packed()
    qc = QuantumCircuit(5, 4)
    qc.append(qi.GPI2Gate(0), [3])
    qc.append(qi.GPI2Gate(0), [0])
    bad = PackedCircuit(qc, p.layout, p.block_clbits, ())
    backend = StubBackend("native")
    with pytest.raises(ValueError):
        run_packed(backend, bad, 100)
    assert backend.calls == []


def test_packed_counts_requests_no_sharpening():
    class Result:
        def get_counts(self):
            return {"00000": 7}

    class Job:
        def result(self, **kwargs):
            self.kwargs = kwargs
            return Result()

    job = Job()
    assert packed_counts(job) == {"00000": 7}
    assert job.kwargs == {"sharpen": False}
