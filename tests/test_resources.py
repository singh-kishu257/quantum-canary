import math

import pytest
from qiskit.quantum_info import process_fidelity
from qiskit_aer.noise import depolarizing_error

from quantum_canary.resources import (
    from_key,
    ion,
    pair,
    pair_damping,
    pair_infidelity,
    readout_damping,
    readout_error,
    to_key,
)


def test_pair_damping_uses_process_infidelity():
    lam = 0.03
    channel = depolarizing_error(lam, 2).to_quantumchannel()
    e_process = 1 - process_fidelity(channel)
    assert math.isclose(math.exp(-pair_damping(e_process)), 1 - lam, rel_tol=1e-12)


def test_round_trips():
    for e in (0.0, 0.0046, 0.0885, 0.5):
        assert math.isclose(pair_infidelity(pair_damping(e)), e, abs_tol=1e-12)
    for p in (0.0, 0.005, 0.2):
        assert math.isclose(readout_error(readout_damping(p)), p, abs_tol=1e-12)


def test_keys():
    assert pair(5, 2) == ("pair", 2, 5)
    assert from_key(to_key(pair(3, 7))) == pair(3, 7)
    assert from_key(to_key(ion(4))) == ion(4)
    with pytest.raises(ValueError):
        pair(1, 1)
