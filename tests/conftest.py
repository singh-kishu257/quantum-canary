import numpy as np
import pytest
from qiskit.quantum_info import Statevector


def distribution(qc):
    """Exact outcome distribution over clbits of a measured circuit (Qiskit key order)."""
    unitary = qc.remove_final_measurements(inplace=False)
    meas = [
        (qc.find_bit(i.qubits[0]).index, qc.find_bit(i.clbits[0]).index)
        for i in qc.data
        if i.operation.name == "measure"
    ]
    probs = Statevector.from_instruction(unitary).probabilities_dict()
    out = {}
    for key, p in probs.items():
        bits = ["0"] * qc.num_clbits
        for q, c in meas:
            bits[qc.num_clbits - 1 - c] = key[len(key) - 1 - q]
        k = "".join(bits)
        out[k] = out.get(k, 0.0) + p
    return out


def max_deviation(a, b):
    return max(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
