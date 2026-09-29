import math

import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.random import random_circuit

from quantum_canary import LogicalProgram, compile_native, return_probe_program

from conftest import distribution, max_deviation


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("twirl", [False, True])
def test_native_compilation_is_exact(seed, twirl):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 5))
    qc = random_circuit(n, 4, max_operands=2, seed=seed)
    qc.rzz(float(rng.normal() * 4), 0, 1)
    qc.rzz(math.pi, 0, 1)
    qc.rzz(-2.9, 1, 0)
    qc.measure_all()
    prog = LogicalProgram.from_circuit(qc)
    ions = [int(x) for x in rng.permutation(7)[:n]]
    native = compile_native(prog, ions, rng=np.random.default_rng(seed + 99), twirl=twirl)
    names = {i.operation.name for i in native.data}
    assert names <= {"gpi", "gpi2", "zz", "measure"}
    for inst in native.data:
        if inst.operation.name == "zz":
            assert 0.0 < float(inst.operation.params[0]) <= 0.25
    assert max_deviation(distribution(native), distribution(qc)) < 1e-9


def test_partial_measurement_and_multiple_registers():
    q = QuantumRegister(3)
    c1, c2 = ClassicalRegister(1, "a"), ClassicalRegister(2, "b")
    qc = QuantumCircuit(q, c1, c2)
    qc.h(0)
    qc.cx(0, 2)
    qc.ry(0.7, 1)
    qc.measure(2, c1[0])
    qc.measure(0, c2[1])
    prog = LogicalProgram.from_circuit(qc)
    native = compile_native(prog, (4, 0, 2), rng=np.random.default_rng(0))
    assert max_deviation(distribution(native), distribution(qc)) < 1e-9


@pytest.mark.parametrize("depth", [4, 8, 12])
def test_return_probe_ideal_outcome(depth):
    for seed in range(5):
        native = compile_native(return_probe_program(depth), (3, 1), rng=np.random.default_rng(seed))
        assert abs(distribution(native).get("00", 0.0) - 1.0) < 1e-9
        assert {i.operation.name for i in native.data} <= {"gpi", "gpi2", "zz", "measure"}


def test_invalid_programs():
    qc = QuantumCircuit(2, 2)
    qc.measure(0, 0)
    qc.x(0)
    with pytest.raises(ValueError):
        LogicalProgram.from_circuit(qc)
    from qiskit.circuit import Parameter

    qp = QuantumCircuit(1)
    qp.rx(Parameter("t"), 0)
    with pytest.raises(ValueError):
        LogicalProgram.from_circuit(qp)
    qr = QuantumCircuit(1, 1)
    qr.reset(0)
    qr.measure(0, 0)
    with pytest.raises(ValueError):
        LogicalProgram.from_circuit(qr)
    with pytest.raises(ValueError):
        compile_native(LogicalProgram.from_circuit(QuantumCircuit(2)), (1, 1))


def test_lightcone():
    qc = QuantumCircuit(4)
    qc.rzz(0.3, 0, 1)
    qc.rzz(0.3, 2, 3)
    qc.rzz(0.3, 1, 2)
    qc.measure_all()
    prog = LogicalProgram.from_circuit(qc, optimization_level=0)
    assert prog.lightcone([0]) == {(0, 1): 1}
    assert prog.lightcone([1]) == {(1, 2): 1, (0, 1): 1, (2, 3): 1}
