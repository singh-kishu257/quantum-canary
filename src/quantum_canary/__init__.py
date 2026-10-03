"""Quantum Canary: low-shot recovery of T1, T2 (echo), T2* (Ramsey), qubit
detuning and single-qubit gate error from a fixed set of probe circuits.

Modules
-------
quantum_canary.inversion   the Canary engine: profiles and priors, probe
                           circuits, forward models and the inversion
                           (identical to research/1_inversion.py).
quantum_canary.online      interleaving Canary rounds into a running workflow,
                           per-round health and cross-round drift
                           (research/1b_online.py).
quantum_canary.simulation  the literature-grounded device simulator used to
                           validate Canary (research/1c_simulation_models.py).
quantum_canary.runner      run_ibm() and run_simulation(): one call from
                           qubits to a CanaryReport.

Quick start
-----------
>>> import quantum_canary as canary  
>>> report = canary.run_simulation("superconducting", n_qubits=2)
>>> report.print()

Hardware: superconducting (IBM, Qiskit Runtime). Trapped-ion qubits are
supported in simulation only.
"""

__version__ = "1.0.0"

from . import inversion, online, simulation
from .inversion import (
    ARCH_DEFAULTS,
    NATIVE_INVERSE_PAIRS,
    BackendProfile,
    CalibrationSource,
    InversionResult,
    build_probe_circuits,
    fetch_live_spam,
    forward_echo,
    forward_gate,
    forward_gate_affine,
    forward_ramsey_xy,
    forward_t1,
    get_live_profile,
    interleave_probe_blocks,
    lindblad_inversion,
    merge_block_counts,
    register_native_pair,
    split_qubits_for_op_limit,
)
from .online import (
    CanaryInterleaver,
    OnlineInversionResult,
    build_inline_probe_circuits,
    chi2_scaled_z,
    interleave_circuits,
    online_inversion,
    parse_inline_results,
)
from .runner import (
    DEFAULT_PROBE_BLOCKS,
    DEFAULT_SHOTS,
    CanaryReport,
    QubitReport,
    run_ibm,
    run_simulation,
)
from .display import render_report

__all__ = [
    "__version__", "inversion", "online", "simulation",
    # engine
    "ARCH_DEFAULTS", "NATIVE_INVERSE_PAIRS", "BackendProfile", "CalibrationSource",
    "InversionResult", "build_probe_circuits", "fetch_live_spam", "forward_echo",
    "forward_gate", "forward_gate_affine", "forward_ramsey_xy", "forward_t1",
    "get_live_profile", "interleave_probe_blocks", "lindblad_inversion",
    "merge_block_counts", "register_native_pair", "split_qubits_for_op_limit",
    # online monitoring
    "CanaryInterleaver", "OnlineInversionResult", "build_inline_probe_circuits",
    "chi2_scaled_z", "interleave_circuits", "online_inversion", "parse_inline_results",
    # runners and display
    "DEFAULT_PROBE_BLOCKS", "DEFAULT_SHOTS", "CanaryReport", "QubitReport",
    "run_ibm", "run_simulation", "render_report",
]
