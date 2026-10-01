import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.providers.basic_provider import BasicSimulator

from ionpack import (
    PackedCircuit,
    check_circuit,
    is_sequential,
    make_layout,
    marginal_counts,
    pack,
    sentinel_excitation,
    split_counts,
)


def t7_circuits():
    a = QuantumCircuit(2, 2)
    a.x(0)
    a.measure([0, 1], [0, 1])
    b = QuantumCircuit(3, 3)
    b.x(2)
    b.measure([0, 1, 2], [0, 1, 2])
    c = QuantumCircuit(2, 2)
    c.x(1)
    c.measure(1, 0)
    c.measure(0, 1)
    return [a, b, c]


def t7_packed(**kw):
    return pack(t7_circuits(), make_layout([2, 3, 2], 36, 1), **kw)


# ---------------------------------------------------------------- T4 check_circuit
def _bad_circuits():
    after = QuantumCircuit(1, 1)
    after.measure(0, 0)
    after.x(0)
    reset = QuantumCircuit(1, 1)
    reset.reset(0)
    delay = QuantumCircuit(1, 1)
    delay.delay(100, 0)
    cflow = QuantumCircuit(2, 2)
    cflow.measure(0, 0)
    with cflow.if_test((cflow.clbits[0], 1)):
        cflow.x(1)
    twice_clbit = QuantumCircuit(2, 1)
    twice_clbit.measure(0, 0)
    twice_clbit.measure(1, 0)
    twice_qubit = QuantumCircuit(1, 2)
    twice_qubit.measure(0, 0)
    twice_qubit.measure(0, 1)
    with_clbit = QuantumCircuit(1, 1)
    with_clbit.append(Instruction("foo", 1, 1, []), [0], [0])
    return [after, reset, delay, cflow, twice_clbit, twice_qubit, with_clbit]


@pytest.mark.parametrize("circ", _bad_circuits())
def test_t4_check_circuit_rejects(circ):
    with pytest.raises(ValueError):
        check_circuit(circ)


def test_t4_native_rejects_h():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    assert check_circuit(qc) == {0: 0}
    with pytest.raises(ValueError):
        check_circuit(qc, native=True)


def test_t4_native_accepts_native_gates():
    qi = pytest.importorskip("qiskit_ionq")
    qc = QuantumCircuit(2, 2)
    qc.append(qi.GPI2Gate(0.0), [0])
    qc.append(qi.ZZGate(0.25), [0, 1])
    qc.measure([0, 1], [0, 1])
    assert check_circuit(qc, native=True) == {0: 0, 1: 1}


def test_check_circuit_map_and_unmeasured_clbit():
    qc = QuantumCircuit(2, 3)
    qc.x(1)
    qc.barrier()
    qc.measure(1, 2)
    qc.barrier()
    assert check_circuit(qc) == {2: 1}


# ---------------------------------------------------------------- T5/T6 structure
def test_t5_pack_structure():
    p = t7_packed()
    assert p.circuit.num_qubits == 9
    assert p.circuit.num_clbits == 2 + 3 + 2 + 2
    assert p.block_clbits == ((0, 1), (2, 3, 4), (5, 6))
    assert p.sentinel_clbits == (7, 8)
    assert is_sequential(p)
    ops = [(i.operation.name, [p.circuit.find_bit(q).index for q in i.qubits]) for i in p.circuit.data]
    gates = [o for o in ops if o[0] not in ("barrier", "measure")]
    assert gates == [("x", [0]), ("x", [5]), ("x", [8])]
    assert [o[0] for o in ops].count("barrier") == 3
    meas = [(p.circuit.find_bit(i.qubits[0]).index, p.circuit.find_bit(i.clbits[0]).index)
            for i in p.circuit.data if i.operation.name == "measure"]
    assert meas == [(0, 0), (1, 1), (3, 2), (4, 3), (5, 4), (8, 5), (7, 6), (2, 7), (6, 8)]


def test_t5_not_sequential():
    lay = make_layout([2, 3, 2], 36, 1)
    blocks = ((0, 1), (2, 3, 4), (5, 6))

    def packed(build):
        qc = QuantumCircuit(9, 7)
        build(qc)
        return PackedCircuit(qc, lay, blocks, ())

    assert is_sequential(packed(lambda qc: (qc.x(0), qc.x(3), qc.measure(0, 0))))
    assert not is_sequential(packed(lambda qc: (qc.x(3), qc.x(0))))  # block 0 after block 1
    assert not is_sequential(packed(lambda qc: qc.cx(1, 3)))  # spans two blocks
    assert not is_sequential(packed(lambda qc: qc.x(2)))  # acts on a sentinel
    assert not is_sequential(packed(lambda qc: (qc.measure(0, 0), qc.x(3))))  # gate after measure


def test_t6_no_barriers():
    p = t7_packed(barriers=False)
    assert all(i.operation.name != "barrier" for i in p.circuit.data)
    assert is_sequential(p)


def test_pack_validation():
    lay = make_layout([2, 3, 2], 36, 1)
    with pytest.raises(ValueError):
        pack(t7_circuits()[:2], lay)
    a, b, c = t7_circuits()
    with pytest.raises(ValueError):  # 3-qubit circuit on a 2-ion block
        pack([b, a, c], lay)


# ---------------------------------------------------------------- T7 end to end
def test_t7_end_to_end_basic_simulator():
    p = t7_packed()
    counts = BasicSimulator().run(p.circuit, shots=100).result().get_counts()
    assert counts == {"000110001": 100}
    assert split_counts(counts, p) == [{"01": 100}, {"100": 100}, {"01": 100}]
    assert sentinel_excitation(counts, p) == {2: 0.0, 6: 0.0}


def test_sentinels_not_measured():
    p = t7_packed(measure_sentinels=False)
    assert p.circuit.num_clbits == 7 and p.sentinel_clbits == ()
    counts = BasicSimulator().run(p.circuit, shots=10).result().get_counts()
    assert split_counts(counts, p) == [{"01": 10}, {"100": 10}, {"01": 10}]
    with pytest.raises(ValueError):
        sentinel_excitation(counts, p)


def test_sentinel_excitation_fraction():
    p = t7_packed()
    counts = {"000110001": 75, "100110001": 25}  # clbit 8 = sentinel ion 6
    assert sentinel_excitation(counts, p) == {2: 0.0, 6: 0.25}


# ---------------------------------------------------------------- T8 keys
def test_t8_key_normalization():
    assert marginal_counts({"01 1": 5}, [0], 3) == {"1": 5}
    with pytest.raises(ValueError):
        marginal_counts({"0101": 5}, [0], 3)
    with pytest.raises(ValueError):
        marginal_counts({"012": 5}, [0], 3)
    with pytest.raises(ValueError):
        marginal_counts({"011": 5}, [3], 3)


def test_marginal_order_and_probabilities():
    # key "110": clbit0=0, clbit1=1, clbit2=1; marginal over [0, 2] -> clbit2 clbit0 = "10"
    assert marginal_counts({"110": 0.5, "011": 0.5}, [0, 2], 3) == {"10": 0.5, "01": 0.5}
    assert marginal_counts({"110": 3, "010": 2}, [2, 0], 3) == {"01": 3, "00": 2}


def test_numpy_integer_indices_and_bad_key_types():
    import numpy as np

    assert make_layout([np.int64(2), np.int64(1)], np.int64(36), np.int64(1)).blocks == ((0, 1), (3,))
    assert marginal_counts({"011": 5}, np.array([0, 1]), 3) == {"11": 5}
    with pytest.raises(ValueError):
        marginal_counts({3: 5}, [0], 3)
    with pytest.raises(ValueError):
        marginal_counts({"011": 5}, [True], 3)
