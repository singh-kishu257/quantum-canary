"""Quantum Canary: qubit health checks for IBM (superconducting) and IonQ (trapped-ion) hardware.

    import quantum_canary as canary

    r = canary.check(backend, [13, 14])          # one short job -> Report
    r.print()                                    # numbers, fit (chi2/dof), health, DD advice
    r.save("before.json")

    d = canary.compare(r, canary.load("after.json"))
    d.print(); d.changed                         # what moved beyond its error bars

    mon = canary.Monitor(backend, [13, 14])      # between jobs, as often as you like:
    mon.check()                                  # baseline, then again between jobs
    mon.changed; mon.print()

    canary.simulate(3)                           # no backend needed
    canary.check(ionq_backend, [0, 3])           # IonQ ions: gate error only

On IBM each check runs 74 short probe circuits per qubit (8,640 shots) and recovers T1,
T2 (echo), T2* (Ramsey), the frequency offset and the single-qubit gate error
(randomized benchmarking, error per pulse).
Qubits are probed in at most two parallel groups, never two neighbours at once.

Advanced, in submodules: runner (run_ibm, run_simulation, CanaryReport),
inversion (the Canary engine), simulation (the validation simulator), workflow
(the drift test behind compare), online, ionq, display, cli.
"""

__version__ = "0.9.0"

from . import inversion, online, simulation
from . import workflow
from .workflow import DriftReport
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
from . import api, credentials
from .inversion import (
    ARCH_DEFAULTS,
    BackendProfile,
    CalibrationSource,
    InversionResult,
    build_probe_circuits,
    fetch_live_spam,
    forward_echo,
    forward_ramsey_xy,
    forward_t1,
    get_live_profile,
    interleave_probe_blocks,
    lindblad_inversion,
    merge_block_counts,
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
