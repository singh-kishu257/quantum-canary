"""Quantum Canary: qubit health checks for IBM superconducting hardware.

    import quantum_canary as canary

    r = canary.check(backend, [13, 14])          # one short job -> Report
    r.print()                                    # numbers, fit (chi2/dof), health, DD advice
    r.save("before.json")

    d = canary.compare(r, canary.load("after.json"))
    d.print(); d.changed                         # what moved beyond its error bars

    mon = canary.Monitor(backend, [13, 14])      # workflows longer than ~30 minutes:
    mon.check()                                  # baseline, then again between jobs
    mon.changed; mon.print()

    canary.simulate(3)                           # no backend needed
    canary.run_ionq(backend, [3, 6], eps_prior=2e-4, spam=(0.006, 0.004))

Each check runs 17 short probe circuits per qubit (8,700 shots) and recovers T1,
T2 (echo), T2* (Ramsey), the frequency offset and the single-qubit gate error.
Qubits are probed in at most two parallel groups, never two neighbours at once.

Advanced, in submodules: runner (run_ibm, run_simulation, CanaryReport),
workflow (the older check with spares, run_monitored for Canary rounds inside a
job, compare), inversion (the Canary engine), simulation (the validation
simulator), online, ionq, display, cli.
"""

__version__ = "1.0.0"

from . import inversion, online, simulation
from . import workflow
from .workflow import (
    CheckReport,
    DriftReport,
    MonitoredRun,
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
from .api import Monitor, QubitResult, Report, check, compare, load, simulate
from . import api
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

# The simple interface. Every name imported above stays importable from the top
# level for existing code, but only these are the documented API.
__all__ = ["__version__", "check", "compare", "load", "Monitor", "simulate", "run_ionq",
           "Report", "QubitResult", "IonQReport"]
