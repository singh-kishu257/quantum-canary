"""Quantum Canary: qubit health checks before, during and after a job on IBM
superconducting hardware.

    import quantum_canary as canary

    before = canary.check(backend, qubits=[13, 14, 2], spares=[3, 4])
    before.print()                 # health, layout to use, dynamical decoupling advice

    run = canary.run_monitored(backend, my_isa_circuits, before, interval=20)
    run.print()                    # qubit health in every Canary round

    after = canary.check(backend, qubits=before.layout)
    canary.compare(before, after).print()    # did anything drift?

Each check runs 17 short probe circuits per qubit and recovers T1, T2 (echo),
T2* (Ramsey), the frequency offset and the single-qubit gate error.

On IonQ trapped-ion hardware, run_ionq() measures the single-qubit gate
error of chosen ions (T1 and T2 there are seconds long).

Modules: inversion (the Canary engine), online (interleaved monitoring),
simulation (the device simulator used to validate Canary), runner (one
probe run), workflow (check, run_monitored, compare), ionq (IonQ gate-error
probe), display, cli.
"""

__version__ = "1.0.0"

from . import inversion, online, simulation
from .workflow import (
    CheckReport,
    DriftReport,
    MonitoredRun,
    check,
    compare,
    run_monitored,
)
from .runner import (
    DEFAULT_PROBE_BLOCKS,
    DEFAULT_SHOTS,
    CanaryReport,
    QubitReport,
    run_ibm,
    run_simulation,
)
from .ionq import IonQReport, run_ionq
from .display import banner, render_report
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

__all__ = [
    "__version__", "inversion", "online", "simulation",
    # the job workflow
    "check", "run_monitored", "compare", "CheckReport", "MonitoredRun", "DriftReport",
    # single probe runs and display
    "run_ibm", "run_simulation", "run_ionq", "IonQReport", "CanaryReport", "QubitReport",
    "DEFAULT_SHOTS", "DEFAULT_PROBE_BLOCKS", "render_report", "banner",
    # engine
    "ARCH_DEFAULTS", "NATIVE_INVERSE_PAIRS", "BackendProfile", "CalibrationSource",
    "InversionResult", "build_probe_circuits", "fetch_live_spam", "forward_echo",
    "forward_gate", "forward_gate_affine", "forward_ramsey_xy", "forward_t1",
    "get_live_profile", "interleave_probe_blocks", "lindblad_inversion",
    "merge_block_counts", "register_native_pair", "split_qubits_for_op_limit",
    # online monitoring
    "CanaryInterleaver", "OnlineInversionResult", "build_inline_probe_circuits",
    "chi2_scaled_z", "interleave_circuits", "online_inversion", "parse_inline_results",
]
