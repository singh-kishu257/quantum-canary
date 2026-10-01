import numpy as np
import pytest
from qiskit import QuantumCircuit

from ionpack import AZURE_FORTE_RATES, BRAKET_FORTE, Pricing, circuit_cost, count_gates, packed_savings
from test_pack_unpack import t7_packed

P1, P2, M_OFF, M_ON, K = 0.0001645, 0.001121, 25.7899, 168.195, 7

FLOOR_OFF = Pricing("floor", price_1q=P1, price_2q=P2, minimum=M_OFF)
ADD_OFF = Pricing("additive", price_1q=P1, price_2q=P2, minimum=M_OFF)
FLOOR_ON = Pricing("floor", price_1q=P1, price_2q=P2, minimum=M_ON)

# (n1q, n2q, shots), block gate cost c, savings floor m_off, additive m_off, floor m_on
T12 = [
    ((30, 12, 1000), 18.387, 1.402616, 2.001557, 7.000000),
    ((30, 12, 300), 5.5161, 4.675387, 3.402689, 7.000000),
    ((16, 6, 1000), 9.358, 2.755920, 2.694922, 7.000000),
    ((60, 40, 1000), 54.71, 1.000000, 1.378558, 3.074301),
]


def test_azure_preset_values():
    assert AZURE_FORTE_RATES["price_1q"] == P1 and AZURE_FORTE_RATES["price_2q"] == P2
    assert AZURE_FORTE_RATES["minimum_em_off"] == M_OFF and AZURE_FORTE_RATES["minimum_em_on"] == M_ON
    assert BRAKET_FORTE.per_task == 0.30 and BRAKET_FORTE.per_shot == 0.08


@pytest.mark.parametrize("block, c, s_floor_off, s_add_off, s_floor_on", T12)
def test_t12_savings_table(block, c, s_floor_off, s_add_off, s_floor_on):
    n1, n2, shots = block
    assert circuit_cost(n1, n2, shots, Pricing("additive", price_1q=P1, price_2q=P2)) == pytest.approx(c, abs=1e-6)
    counts = [(n1, n2)] * K
    assert packed_savings(counts, shots, FLOOR_OFF) == pytest.approx(s_floor_off, abs=1e-6)
    assert packed_savings(counts, shots, ADD_OFF) == pytest.approx(s_add_off, abs=1e-6)
    assert packed_savings(counts, shots, FLOOR_ON) == pytest.approx(s_floor_on, abs=1e-6)
    # exactly k mathematically; 7 * 80.3 / 80.3 is 7.000000000000001 in binary floating point
    assert packed_savings(counts, shots, BRAKET_FORTE) == pytest.approx(7.0, abs=1e-12)


def test_t12_costs():
    assert circuit_cost(30, 12, 1000, FLOOR_OFF) == pytest.approx(25.7899, abs=1e-6)
    assert circuit_cost(30 * K, 12 * K, 1000, FLOOR_OFF) == pytest.approx(128.709, abs=1e-6)
    assert circuit_cost(5, 5, 1000, BRAKET_FORTE) == pytest.approx(0.30 + 0.08 * 1000, abs=1e-9)


def test_t12_floor_closed_form_property():
    rng = np.random.default_rng(12345)
    for _ in range(500):
        m = float(rng.uniform(1, 500))
        c = float(rng.uniform(0.01, m * 0.999))
        k = int(rng.integers(1, 40))
        pricing = Pricing("floor", price_1q=c, minimum=m)
        assert packed_savings([(1, 0)] * k, 1, pricing) == pytest.approx(min(k, m / c), rel=1e-12)
    assert packed_savings([(1, 0)] * 5, 1, Pricing("floor", price_1q=3.0, minimum=2.0)) == 1.0


def test_t12_count_gates_on_packed():
    assert count_gates(t7_packed().circuit) == (3, 0)
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.barrier()
    qc.measure([0, 1], [0, 1])
    assert count_gates(qc) == (1, 1)


def test_count_gates_rejects():
    three = QuantumCircuit(3)
    three.ccx(0, 1, 2)
    reset = QuantumCircuit(1)
    reset.reset(0)
    delay = QuantumCircuit(1)
    delay.delay(10, 0)
    cflow = QuantumCircuit(2, 1)
    cflow.measure(0, 0)
    with cflow.if_test((cflow.clbits[0], 1)):
        cflow.x(1)
    for qc in (three, reset, delay, cflow):
        with pytest.raises(ValueError):
            count_gates(qc)


def test_pricing_validation():
    with pytest.raises(ValueError):
        Pricing("bogus")
    with pytest.raises(ValueError):
        Pricing("floor", price_1q=-1.0)
    with pytest.raises(ValueError):
        circuit_cost(-1, 0, 10, FLOOR_OFF)
    with pytest.raises(ValueError):
        packed_savings([], 10, FLOOR_OFF)
    with pytest.raises(ValueError):
        packed_savings([(1, 1)], 10, Pricing("floor"))
