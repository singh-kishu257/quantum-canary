import json
import warnings

import numpy as np

from quantum_canary import IonQExecutor, compile_native, return_probe_program


def test_ionq_payload_uses_native_gates_and_no_debiasing():
    from qiskit_ionq import ErrorMitigation, IonQProvider
    from qiskit_ionq.helpers import qiskit_to_ionq

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        backend = IonQProvider("offline-test-token").get_backend("qpu.forte-1", gateset="native")
    executor = IonQExecutor(backend, ions=range(36))
    assert executor.run_options["error_mitigation"] is ErrorMitigation.NO_DEBIASING
    circuits = [compile_native(return_probe_program(4), (35, 0), rng=np.random.default_rng(0))]
    payload = json.loads(qiskit_to_ionq(circuits, backend, passed_args={"shots": 100, **executor.run_options}))
    assert payload["input"]["gateset"] == "native"
    assert payload["settings"]["error_mitigation"] == {"debiasing": False}
    gates = {g["gate"] for g in payload["input"]["circuits"][0]["circuit"]} if "circuits" in payload["input"] else {
        g["gate"] for g in payload["input"]["circuit"]
    }
    assert gates <= {"gpi", "gpi2", "zz"}
