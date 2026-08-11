import importlib.util
import multiprocessing as mp
import pathlib
import sys
import csv
import time
import warnings
import numpy as np
from datetime import datetime, timezone
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("inversion", SCRIPT_DIR / "1_inversion.py")
inv = importlib.util.module_from_spec(_spec)
sys.modules["inversion"] = inv
_spec.loader.exec_module(inv)

DATA_DIR = SCRIPT_DIR / "data"
FIG_DIR  = SCRIPT_DIR / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

SEED               = 48
N_INSTANCES        = 200
N_BOOTSTRAP        = 2000
R2_THRESHOLD       = 0.95

CANARY_SHOTS_T1     = 300
CANARY_SHOTS_RAMSEY = 1000
CANARY_SHOTS_GATE   = 500
CANARY_SHOTS_ECHO   = 500
CANARY_TOTAL        = (3 * CANARY_SHOTS_T1 +
                       6 * CANARY_SHOTS_RAMSEY +
                       3 * CANARY_SHOTS_GATE +
                       3 * CANARY_SHOTS_ECHO)

QE_N_DELAYS        = 15
QE_RB_LENGTHS      = [1, 2, 4, 8, 16, 32, 64, 128]
QE_RB_SAMPLES      = 5
QE_SX_PER_CLIFFORD = 1.875
# Plausibility guards on QE's raw curve-fit outputs.
#
# Canary's own optimizer enforces hard physical bounds and fails safely to
# NaN on divergence. QE's ad hoc extraction (T1/T2Ramsey/T2Hahn curve_fit +
# a manual frequency-difference calculation for dw, + RB's EPC/gates-per-
# Clifford for eps) has no equivalent bound, so a diverging fit can
# silently produce a physically impossible value that then dominates
# R^2/NRMSE instead of being counted as a fit failure.
#
# CRITICAL: every guard below is anchored to TRUE_PARAM_RANGES (the actual
# distribution sample_instance() draws true values from in THIS benchmark)
# or to the same dynamic BackendProfile quantity that generated the true
# dw values -- never to ARCH_DEFAULTS' full architectural physical bound
# (T1_max_s, eps_max, the static dw_max_rad_s constant). Those ARCH_DEFAULTS
# bounds describe the full plausible range of real hardware in general, not
# the much narrower range actually sampled in this specific benchmark, and
# are commonly 50-8900x larger than the true sampling range. A guard
# anchored to the wrong (larger) reference is mathematically incapable of
# rejecting anything: earlier versions of this file made exactly this
# mistake for dw (static ARCH_DEFAULTS dw_max_rad_s, off by ~4400x for
# trapped_ion) and, separately, for T1/T2/eps (ARCH_DEFAULTS T1_max_s /
# eps_max, off by 50-5000x) -- both confirmed by 0% guard-rejection rates
# coexisting with R^2 in the -1e5 to -1e7 range, which is only possible if
# the guard threshold is far above anything a diverged fit could produce.
QE_DW_PLAUSIBILITY_MULT  = 2.0
QE_T1_PLAUSIBILITY_MULT  = 5.0
QE_EPS_PLAUSIBILITY_MULT = 5.0

# Counts, per architecture, how often each QE parameter's raw fit output
# was rejected by its plausibility guard. Printed at the end of run_sweep()
# so a still-broken guard (implausible values that keep slipping through)
# or a guard that is rejecting nearly everything (too strict) is visible in
# the job log rather than only discoverable after downloading the CSV.
QE_GUARD_REJECTS = defaultdict(lambda: defaultdict(int))
QE_GUARD_TOTAL   = defaultdict(lambda: defaultdict(int))

import os as _os
_ALL_BUDGETS = [1_000, 2_500, 5_000, 9_900, 15_000, 25_000, 50_000]
_budget_env  = _os.environ.get("BUDGET_SUBSET", "").strip()
if _budget_env:
    SHOT_BUDGETS = [int(b.strip()) for b in _budget_env.split(",") if b.strip()]
else:
    SHOT_BUDGETS = _ALL_BUDGETS

ALL_ARCHITECTURES = ["superconducting", "trapped_ion", "neutral_atom"]
ARCH = _os.environ.get("ARCH", "superconducting").strip()
if ARCH not in ALL_ARCHITECTURES:
    sys.exit(f"ARCH env var must be one of {ALL_ARCHITECTURES}, got {ARCH!r}")

JOB_TAG = _os.environ.get("JOB_TAG", f"full_{ARCH}")

# Real, physically-sampled current-hardware ranges per architecture — same
# ranges used throughout the codebase (2_parity_experiments.py,
# 3_shot_ablation.py) for consistency across all synthetic experiments.
TRUE_PARAM_RANGES = {
    "superconducting": {
        "T1_s": (80e-6,  400e-6),
        "T2_s": (40e-6,  200e-6),
        "eps":  (1e-4,   2e-3),
    },
    "trapped_ion": {
        "T1_s": (100.0,  10000.0),
        "T2_s": (0.1,    3.0),
        "eps":  (1e-4,   2e-3),
    },
    "neutral_atom": {
        "T1_s": (1.0,    100.0),
        "T2_s": (0.3,    3.0),
        "eps":  (1e-3,   1e-2),
    },
}

PARAMS = [
    ("T1",          0),
    ("T2",          1),
    ("delta_omega", 2),
    ("epsilon_sx",  3),
]

BENCH_COLS = [
    "budget", "architecture", "parameter", "method",
    "r2", "r2_lo", "r2_hi",
    "nrmse", "n_valid", "n_total", "mean_time_s",
]

THRESH_COLS = ["architecture", "parameter", "method", "shots_to_R2_095", "shots_to_R2_099"]

PARAM_LATEX = {
    "T1":          r"$T_1$",
    "T2":          r"$T_2$ (combined)",
    "delta_omega": r"$|\Delta\omega|$",
    "epsilon_sx":  r"$\varepsilon_{sx}$",
}

COLOUR = {"Canary": "#1F3864", "QiskitExperiments": "#C00000"}
MARKER = {"Canary": "o",       "QiskitExperiments": "s"}
LABEL  = {
    "Canary":            "Canary (this work)",
    "QiskitExperiments": "Qiskit Experiments (T1+T2R+T2H+RB)",
}
METHODS = ["Canary", "QiskitExperiments"]
PORDER  = ["T1", "T2", "delta_omega", "epsilon_sx"]


def sample_instance(rng, arch_name):
    ranges = TRUE_PARAM_RANGES[arch_name]
    T1_lo, T1_hi = ranges["T1_s"]
    T2_lo, T2_hi = ranges["T2_s"]
    eps_lo, eps_hi = ranges["eps"]

    if T1_hi / T1_lo > 10.0:
        T1 = float(10 ** rng.uniform(np.log10(T1_lo), np.log10(T1_hi)))
    else:
        T1 = float(rng.uniform(T1_lo, T1_hi))
    T_phi = float(rng.uniform(T2_lo, T2_hi))
    T2    = min(1.0 / (1.0 / (2.0 * T1) + 1.0 / T_phi), 2.0 * T1)
    eps   = float(10 ** rng.uniform(np.log10(eps_lo), np.log10(eps_hi)))
    dw_max = inv.BackendProfile.from_architecture(arch_name).dw_max_rad_s
    dw     = float(rng.choice([-1, 1]) * rng.uniform(0.2 * dw_max, dw_max))
    return T1, T2, dw, eps


REALISTIC_NOISE = {
    "superconducting": {
        "sigma_T1_lognormal": 0.15,
        "T2_reduction_max":   0.12,
        "sigma_coherent_rad": 0.010,
        "sigma_SPAM_frac":    0.05,
    },
    "trapped_ion": {
        "sigma_T1_lognormal": 0.00,
        "T2_reduction_max":   0.10,
        "sigma_coherent_rad": 0.005,
        "sigma_SPAM_frac":    0.03,
    },
    "neutral_atom": {
        "sigma_T1_lognormal": 0.03,
        "T2_reduction_max":   0.15,
        "sigma_coherent_rad": 0.015,
        "sigma_SPAM_frac":    0.08,
    },
}


def _perturb_instance(arch_name, T1, T2, eps, p0g1_nom, p1g0_nom, rng):
    """Apply architecture-specific unmodeled hardware physics to one instance.

    Returns effective (T1_eff, T2_eff, eps_eff, p0g1_eff, p1g0_eff) that
    are passed to BOTH run_canary and run_qe, so both methods see identical
    simulated hardware and neither is advantaged or disadvantaged.

    Neither method knows the effective values — Canary uses
    BackendProfile.from_architecture() (arch-typical prior) and QE uses
    arch-typical delay grids. Both are equally blind to the drift.

    Perturbations:
      T1  : log-normal TLS fluctuator (Klimov et al. PRL 2018;
                                        Carroll et al. npj QI 2022)
      T2  : uniform fractional reduction from quasi-static 1/f dephasing
            (same references; also micromotion for trapped-ion,
             laser phase noise for neutral-atom)
      eps : coherent over-rotation delta ~ N(0, sigma_coh), adds
            eps_coh = delta^2/4 to depolarising (Rol et al. PRL 2019,
            first-order Magnus)
      SPAM: calibration drift, multiplicative fractional noise
            (Bultink et al. PRApplied 2018)

    No dw perturbation: sigma_dw interacts with Canary's hard arctan2
    bounds in a way that depends on the per-instance profile dw_max,
    which varies with T2_prior and cannot be safely bounded by a fixed
    sigma. Omitting it is the correct choice — dw recovery is already
    characterised in the ideal Markovian scenario.
    """
    arch = inv.ARCH_DEFAULTS[arch_name]
    cfg  = REALISTIC_NOISE[arch_name]

    if cfg["sigma_T1_lognormal"] > 0.0:
        T1_eff = float(np.clip(
            T1 * np.exp(rng.normal(0.0, cfg["sigma_T1_lognormal"])),
            arch["T1_min_s"], arch["T1_max_s"]))
    else:
        T1_eff = T1

    T2_eff = float(np.clip(
        T2 * (1.0 - rng.uniform(0.0, cfg["T2_reduction_max"])),
        arch["T2_min_s"], min(2.0 * T1_eff, arch["T1_max_s"])))

    delta_coh = rng.normal(0.0, cfg["sigma_coherent_rad"])
    eps_eff   = float(np.clip(eps + delta_coh ** 2 / 4.0, 0.0, arch["eps_max"]))

    p0g1_eff = float(np.clip(
        p0g1_nom * (1.0 + rng.normal(0.0, cfg["sigma_SPAM_frac"])), 0.0, 0.30))
    p1g0_eff = float(np.clip(
        p1g0_nom * (1.0 + rng.normal(0.0, cfg["sigma_SPAM_frac"])), 0.0, 0.30))

    return T1_eff, T2_eff, eps_eff, p0g1_eff, p1g0_eff


def _convert_native_to_unitary(qc, native_gate_names):
    # AerSimulator's C++ backend does not recognise third-party gate names
    # (verified directly: "gpi"/"gpi2" are absent from
    # AerSimulator().configuration().basis_gates, and attempting to run
    # them raises AerError: unknown instruction). Transpiling onto a
    # native Target correctly produces real gpi/gpi2 sequences, but Aer
    # still cannot execute them by name. The fix verified end-to-end
    # before this was written: convert each named native gate instance to
    # a generic UnitaryGate carrying the SAME matrix (native_gate.to_matrix()),
    # which Aer natively supports ("unitary" IS in Aer's basis_gates). This
    # preserves the exact physical operation (same unitary each native
    # pulse implements) while making it something Aer can simulate; it
    # does not change the gate COUNT or PHYSICS, only how Aer labels the
    # instruction internally.
    from qiskit.circuit.library import UnitaryGate
    new_qc = qc.copy_empty_like()
    for inst in qc.data:
        op = inst.operation
        if op.name in native_gate_names:
            new_qc.append(UnitaryGate(op.to_matrix(), label=op.name),
                          inst.qubits, inst.clbits)
        else:
            new_qc.append(inst)
    return new_qc


def _make_backend(qc, T1, T2, eps, p0g1, p1g0, gate_time_ns, dt_ns,
                  noise_gate_names=None):
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (NoiseModel, thermal_relaxation_error,
                                   depolarizing_error, ReadoutError)
    nm   = NoiseModel()
    gate_time_s = gate_time_ns * 1e-9
    sx_e = (thermal_relaxation_error(T1, T2, gate_time_s)
            .compose(depolarizing_error(2.0 * eps, 1)))
    # rz excluded: virtual Z rotation, zero physical pulse, zero noise on
    # every real hardware platform (same convention as 1_inversion.py).
    # noise_gate_names defaults to the superconducting-basis instruction
    # names; callers targeting a native non-superconducting gate set (see
    # run_qe()'s trapped_ion path) pass the correct names instead (e.g.
    # ["unitary"] after native gates are converted via
    # _convert_native_to_unitary), so the SAME physical noise channel is
    # attached to whichever instruction actually represents the noisy
    # native single-qubit operation for that architecture.
    names = noise_gate_names or ["sx", "x", "id", "rx", "u1", "p"]
    nm.add_quantum_error(sx_e, names, [0])
    seen = set()
    for inst in qc.data:
        if inst.operation.name == "delay":
            ds = inv._delay_seconds(inst, dt_ns)
            key = round(ds, 15)
            if key not in seen and ds > 1e-12:
                seen.add(key)
                nm.add_quantum_error(
                    thermal_relaxation_error(T1, T2, ds), ["delay"], [0])
    nm.add_readout_error(
        ReadoutError([[1 - p0g1, p0g1], [p1g0, 1 - p1g0]]), [0])
    return AerSimulator(noise_model=nm)


def _run_exp(exp_obj, T1, T2, eps, p0g1, p1g0, shots, gate_time_ns, dt_ns,
            dw_true=0.0, inject_detuning=False, native_target=None,
            native_gate_names=None):
    from qiskit import transpile
    exp_obj.set_run_options(shots=shots)
    circuits = exp_obj.circuits()
    exp_data = exp_obj._initialize_experiment_data()
    for qc in circuits:
        run_qc = (inv._inject_ramsey_detuning(qc, dw_true, dt_ns)
                  if inject_detuning else qc)
        noise_gate_names = None
        if native_target is not None:
            # Transpile the abstract sx/x/rz circuit onto the real native
            # gate Target (verified working: real GPI/GPI2 sequences,
            # correct delay/measure passthrough). Then convert those named
            # native gates to AerSimulator-executable UnitaryGate instances
            # (same physical operation, different internal label -- see
            # _convert_native_to_unitary docstring for why this step is
            # required rather than optional).
            run_qc = transpile(run_qc, target=native_target, optimization_level=1)
            run_qc = _convert_native_to_unitary(run_qc, native_gate_names)
            noise_gate_names = ["unitary"]
        b    = _make_backend(run_qc, T1, T2, eps, p0g1, p1g0, gate_time_ns,
                             dt_ns, noise_gate_names=noise_gate_names)
        tqc  = run_qc if native_target is not None else transpile(run_qc, b, optimization_level=0)
        cnts = b.run(tqc, shots=shots).result().get_counts()
        meta = {**dict(qc.metadata), "count_ops": [(((0,), g), c) for g, c in tqc.count_ops().items()]}
        exp_data.add_data({"counts": cnts, "shots": shots, "metadata": meta})
    exp_obj.analysis.run(exp_data).block_for_results()
    return exp_data


def _extract(exp_data, name):
    try:
        df  = exp_data.analysis_results(dataframe=True)
        row = df[df["name"] == name]
        if row.empty:
            return float("nan"), float("nan")
        val = row["value"].iloc[0]
        nom = float(val.nominal_value)
        std = float(val.std_dev)
        std = std if np.isfinite(std) and std > 0 else float("nan")
        return nom, std
    except Exception:
        return float("nan"), float("nan")


def _ionq_native_target(gate_time_ns: float):
    # Builds a local, 1-qubit Target using IonQ's real native gate classes
    # (qiskit_ionq.GPIGate / GPI2Gate) and IonQ's own published gate
    # equivalences (qiskit_ionq.add_equivalences()), which is exactly the
    # mechanism qiskit-ionq itself relies on so that abstract circuits
    # (built in sx/x/rz, as every qiskit-experiments class does) compile
    # down to GPI/GPI2 exactly as they would when targeting a real IonQ
    # backend (see qiskit-ionq docs: "The IonQ provider ... includes its
    # own transpilation and compilation pipeline" from an abstract 'qis'
    # basis down to native gates). We build this locally, rather than
    # connecting to IonQProvider(), so no live API token/network access is
    # required to determine the correct native-gate accounting — this
    # script otherwise runs entirely against local AerSimulator, and stays
    # that way; only the *gate-counting* below touches this Target, never
    # the executed circuit (see run_qe()'s RB block).
    # Returns None if qiskit_ionq is unavailable, so callers can fall back
    # to the existing fixed-constant behavior rather than crash a whole
    # matrix job over an optional dependency.
    try:
        from qiskit.transpiler import Target, InstructionProperties
        from qiskit.circuit import Delay, Measure, Parameter
        from qiskit_ionq import GPIGate, GPI2Gate, add_equivalences
    except Exception:
        return None
    add_equivalences()  # idempotent; registers IonQ gate equivalences globally
    target = Target(num_qubits=1)
    theta = Parameter("theta")
    dur = gate_time_ns * 1e-9
    target.add_instruction(GPIGate(theta),  {(0,): InstructionProperties(duration=dur)})
    target.add_instruction(GPI2Gate(theta), {(0,): InstructionProperties(duration=dur)})
    target.add_instruction(Delay(Parameter("t")), {(0,): None})
    target.add_instruction(Measure(), {(0,): InstructionProperties(duration=dur)})
    return target


def _native_gates_per_clifford(rb_circuits, native_target):
    # Empirically measures the average number of noisy, architecture-native
    # single-qubit gates (GPI + GPI2) per Clifford in THIS run's actual RB
    # circuit sample, by transpiling each circuit onto native_target and
    # reading its true gate count — rather than assuming a fixed
    # gates-per-Clifford ratio calibrated for a different basis. This is
    # the standard, correct way to convert a measured error-per-Clifford
    # into an error-per-native-gate for a specific device's real gate set
    # (cf. QE_SX_PER_CLIFFORD, which is the superconducting-basis
    # equivalent of exactly this quantity, historically hardcoded rather
    # than measured). Returns NaN if nothing could be transpiled/measured,
    # so the caller can fail safely to NaN instead of dividing by a bad
    # ratio.
    from qiskit import transpile
    from qiskit.transpiler.exceptions import TranspilerError
    weighted_gates = 0.0
    weighted_cliffords = 0.0
    for qc in rb_circuits:
        n_cliff = qc.metadata.get("xval")
        if not n_cliff:
            continue
        try:
            tqc = transpile(qc, target=native_target, optimization_level=1)
        except TranspilerError:
            continue
        counts = tqc.count_ops()
        n_native = counts.get("gpi", 0) + counts.get("gpi2", 0)
        weighted_gates      += n_native
        weighted_cliffords  += n_cliff
    if weighted_cliffords <= 0:
        return float("nan")
    return weighted_gates / weighted_cliffords


def _qe_backend(arch_name: str = "superconducting"):
    # DISCLOSED METHODOLOGICAL CONSTRAINT (not a bug, not silently fixable):
    # qiskit-experiments' standard T1 / T2Ramsey / T2Hahn / StandardRB
    # experiment classes generate circuits in an abstract sx/x/rz/delay
    # basis; the *executed* circuit here is always built/simulated against
    # that abstract basis via the backend this function returns, regardless
    # of arch_name — changing that would mean changing _make_backend()'s
    # noise-model gate tagging too, which is out of scope for this fix.
    # arch_name is accepted so the caller can log/audit which architecture
    # a given run was nominally targeting.
    #
    # This does NOT mean every downstream quantity is architecture-blind,
    # though: for trapped_ion, run_qe() separately measures the real
    # IonQ-native (GPI/GPI2) gate count per Clifford via a verified local
    # Target built from qiskit_ionq's own gate/equivalence library (see
    # _ionq_native_target / _native_gates_per_clifford below), and uses
    # that — not this function's abstract backend — to convert RB's
    # measured EPC into epsilon_sx correctly for that architecture's real
    # native gate set. No equivalent standard, gate-based qiskit-experiments
    # workflow exists for neutral_atom (Pasqal/QuEra neutral-atom hardware
    # is accessed via analog abstractions — Pulser, Braket — not digital
    # BackendV2 circuits), so neutral_atom has no correction available and
    # remains fully on this abstract-basis proxy; any neutral_atom result
    # should be reported as a best-effort adaptation, not as "the standard
    # way practitioners run it," because no such standard exists today.
    # Any trapped_ion/neutral_atom Δω result specifically should likewise
    # be attributed to this basis-gate proxy rather than to the same
    # Nyquist mechanism claimed for superconducting, where the backend's
    # basis genuinely matches the target architecture.
    try:
        from qiskit.providers.fake_provider import GenericBackendV2
        return GenericBackendV2(
            num_qubits=1,
            basis_gates=["sx", "x", "rz", "measure", "delay"],
            seed=0)
    except Exception:
        from qiskit_ibm_runtime.fake_provider import FakeNairobiV2
        return FakeNairobiV2()


def run_canary(T1_true, T2_true, dw_true, eps_true, seed, budget, arch_name,
               p0g1_eff=None, p1g0_eff=None):
    arch    = inv.ARCH_DEFAULTS[arch_name]
    profile = inv.BackendProfile.from_architecture(arch_name)
    p0g1    = p0g1_eff if p0g1_eff is not None else arch["p0_given_1"]
    p1g0    = p1g0_eff if p1g0_eff is not None else arch["p1_given_0"]

    circuits, meta = inv.build_probe_circuits(profile)

    scale   = budget / CANARY_TOTAL
    sh_t1   = max(10, int(round(CANARY_SHOTS_T1     * scale)))
    sh_ram  = max(10, int(round(CANARY_SHOTS_RAMSEY  * scale)))
    sh_gate = max(10, int(round(CANARY_SHOTS_GATE    * scale)))
    sh_echo = max(10, int(round(CANARY_SHOTS_ECHO    * scale)))

    try:
        counts_list = inv.run_probe_circuits_aer(
            circuits, meta,
            T1_true, T2_true, eps_true,
            p0g1, p1g0,
            profile.dt_ns,
            sh_t1, sh_ram, sh_gate, sh_echo,
            dw_s=dw_true,
        )
        r = inv.lindblad_inversion(
            counts_list,
            meta, profile,
            shots_t1=sh_t1, shots_ramsey=sh_ram,
            shots_gate=sh_gate, shots_echo=sh_echo,
            qubit_id=0,
            timestamp=datetime.now(timezone.utc).isoformat())
        return r.T1_s, r.T2_s, abs(r.delta_omega), r.epsilon_sx
    except Exception:
        return float("nan"), float("nan"), float("nan"), float("nan")


def run_qe(T1_true, T2_true, dw_true, eps_true, budget, seed, arch_name,
           p0g1_eff=None, p1g0_eff=None, profile_dw_max=None):
    from qiskit_experiments.library import (T1 as QE_T1, T2Ramsey,
                                            T2Hahn, StandardRB)
    arch         = inv.ARCH_DEFAULTS[arch_name]
    p0g1         = p0g1_eff if p0g1_eff is not None else arch["p0_given_1"]
    p1g0         = p1g0_eff if p1g0_eff is not None else arch["p1_given_0"]
    T1_pr        = arch["T1_s"]
    T2_pr        = arch["T2_s"]
    eps_max      = arch["eps_max"]
    gate_time_ns = arch["gate_time_ns"]
    # BUG FIX: dw_max was previously arch["dw_max_rad_s"] — a fixed,
    # architecture-wide constant from ARCH_DEFAULTS. sample_instance()
    # (which generates the TRUE dw values this guard is meant to judge
    # plausibility against) uses a completely different, DYNAMIC quantity:
    # inv.BackendProfile.from_architecture(arch_name).dw_max_rad_s, a
    # @property computed as 0.90*pi/ramsey_delays_s[0] from the actual
    # arch_default log-spaced delay grid. For trapped_ion these two
    # differ by ~4400x (dynamic ~14 rad/s vs static ~62,832 rad/s),
    # because trapped_ion's huge T1 range [0.01s, 100000s] forces very
    # long log-spaced delays, giving a tiny dynamic dw_max. Using the
    # static constant meant the guard threshold was ~8,900x-44,000x larger
    # than the largest true dw value ever sampled -- mathematically
    # incapable of rejecting anything, regardless of how badly QE's fit
    # diverged. This is why 0% were rejected while R^2 was still
    # catastrophic (~-930,000): every QE prediction passed the guard
    # trivially, no matter its actual magnitude relative to the true,
    # much smaller signal.
    #
    # profile_dw_max is passed in from run_sweep(), computed ONCE per
    # architecture rather than recomputed on every call: it is fully
    # deterministic given arch_name (arch_default confidence, fixed
    # log-spaced delay grid — no dependency on the instance's true
    # parameters), and BackendProfile.from_architecture() internally
    # calls fetch_live_spam(), which may attempt a live network call
    # before falling back. Recomputing this thousands of times (once per
    # instance x budget) would be both wasteful and a potential source of
    # slow/blocking calls in a multiprocessing worker.
    dw_max = (profile_dw_max if profile_dw_max is not None
              else inv.BackendProfile.from_architecture(arch_name).dw_max_rad_s)

    budget_per_exp = budget // 4
    n_rb_circs     = len(QE_RB_LENGTHS) * QE_RB_SAMPLES
    shots_t1  = max(10, budget_per_exp // QE_N_DELAYS)
    shots_t2r = max(10, budget_per_exp // QE_N_DELAYS)
    shots_t2h = max(10, budget_per_exp // QE_N_DELAYS)
    shots_rb  = max(10, budget_per_exp // n_rb_circs)

    delays_t1 = np.linspace(1e-6, 5.0 * T1_pr, QE_N_DELAYS).tolist()
    delays_t2 = np.linspace(1e-6, 3.0 * T2_pr, QE_N_DELAYS).tolist()

    backend = _qe_backend(arch_name)

    # FIX: dt_ns previously came from arch.get("dt_ns") — the target
    # architecture's own native clock (None for trapped_ion / neutral_atom;
    # 0.2222 ns for superconducting). That is NOT necessarily the clock
    # `backend` (above) used to build/schedule these circuits — qiskit-
    # experiments schedules against `backend`'s own timing, and
    # _make_backend()'s noise model later re-interprets each Delay
    # instruction's duration using whatever dt_ns is passed to it
    # (inv._delay_seconds(inst, dt_ns)). Feeding it a different, unrelated
    # clock than the one that built the circuit is a unit-consistency bug:
    # for trapped_ion/neutral_atom, delays spanning up to 3x T2_pr (up to
    # several seconds) would be built against the backend's own dt and then
    # reinterpreted under dt_ns=None, i.e. as raw seconds — internally
    # inconsistent regardless of what value dt_ns=None resolves to
    # downstream. We now always resolve dt_ns from the backend that will
    # actually schedule the circuits, so the clock used to build a circuit
    # and the clock used to interpret its delays in the noise model are
    # always the same value.
    dt_ns = (backend.dt * 1e9) if getattr(backend, "dt", None) else 0.2222

    # Native execution: for trapped_ion, transpile every experiment's
    # circuits onto a verified real IonQ-native Target (GPI/GPI2, via
    # qiskit_ionq's own gate classes and equivalence library -- see
    # _ionq_native_target()) instead of executing the abstract sx/x/rz
    # proxy circuit. Tested end-to-end before being wired in here:
    # abstract circuit -> native transpile -> GPI/GPI2 -> converted to
    # AerSimulator-executable UnitaryGate (same physical operation,
    # required because Aer's C++ backend does not recognise "gpi"/"gpi2"
    # as instruction names) -> noise tagged onto "unitary" -> executed.
    # superconducting keeps its existing (already correct) abstract-basis
    # path unchanged. neutral_atom has no verified native-gate provider to
    # build a Target from (no standard digital qiskit-experiments workflow
    # exists for that hardware family -- see _qe_backend() docstring) and
    # remains on the disclosed abstract-basis proxy.
    native_target     = None
    native_gate_names = None
    if arch_name == "trapped_ion":
        native_target = _ionq_native_target(gate_time_ns)
        if native_target is not None:
            native_gate_names = ["gpi", "gpi2"]

    T1_qe   = float("nan");  T1_std  = float("nan")
    T2r_qe  = float("nan");  T2r_std = float("nan")
    T2h_qe  = float("nan");  T2h_std = float("nan")
    dw_qe   = float("nan")
    eps_qe  = float("nan")

    # guard_flags: (had_candidate, was_rejected) per parameter, returned to
    # the caller (run_sweep, in the main process) for aggregation -- a
    # plain module-level counter would not aggregate correctly across
    # multiprocessing worker processes.
    guard_flags = {"T1": (False, False), "T2": (False, False),
                   "dw": (False, False), "eps": (False, False)}

    # T1/T2 plausibility reference: the TRUE sampling range for this
    # architecture in THIS benchmark (TRUE_PARAM_RANGES), not
    # ARCH_DEFAULTS["T1_max_s"] (the full physical hardware range, which
    # can be 50-5000x larger -- see constants block comment above).
    T1_true_max = TRUE_PARAM_RANGES[arch_name]["T1_s"][1]
    T1_guard_threshold = QE_T1_PLAUSIBILITY_MULT * T1_true_max

    try:
        data          = _run_exp(QE_T1([0], delays=delays_t1, backend=backend),
                                 T1_true, T2_true, eps_true, p0g1, p1g0, shots_t1,
                                 gate_time_ns, dt_ns,
                                 native_target=native_target,
                                 native_gate_names=native_gate_names)
        T1_candidate, T1_std = _extract(data, "T1")
        if np.isfinite(T1_candidate):
            plausible = (0.0 < T1_candidate <= T1_guard_threshold)
            guard_flags["T1"] = (True, not plausible)
            if plausible:
                T1_qe = T1_candidate
    except Exception:
        pass

    try:
        osc_freq_hz      = arch["dw_typical_khz"] * 1e3
        data             = _run_exp(
            T2Ramsey([0], delays=delays_t2, osc_freq=osc_freq_hz, backend=backend),
            T1_true, T2_true, eps_true, p0g1, p1g0, shots_t2r,
            gate_time_ns, dt_ns,
            dw_true=dw_true, inject_detuning=True,
            native_target=native_target, native_gate_names=native_gate_names)
        T2r_qe, T2r_std  = _extract(data, "T2star")
        freq_hz, _        = _extract(data, "Frequency")
        if np.isfinite(freq_hz):
            dw_candidate = abs(freq_hz - osc_freq_hz) * 2.0 * np.pi
            # dw_max here is the SAME dynamic BackendProfile quantity that
            # generated the true dw values (see docstring above run_qe),
            # not the static ARCH_DEFAULTS constant.
            plausible = dw_candidate <= QE_DW_PLAUSIBILITY_MULT * dw_max
            guard_flags["dw"] = (True, not plausible)
            if plausible:
                dw_qe = dw_candidate
    except Exception:
        pass

    try:
        T2e_true         = min(T2_true, 2.0 * T1_true)
        data             = _run_exp(
            T2Hahn([0], delays=delays_t2, backend=backend),
            T1_true, T2e_true, eps_true, p0g1, p1g0, shots_t2h,
            gate_time_ns, dt_ns,
            native_target=native_target, native_gate_names=native_gate_names)
        T2h_qe, T2h_std  = _extract(data, "T2")
    except Exception:
        pass

    valid = [(v, s) for v, s in [(T2r_qe, T2r_std), (T2h_qe, T2h_std)]
             if np.isfinite(v) and np.isfinite(s) and s > 0]
    if len(valid) == 2:
        w0 = 1.0 / valid[0][1] ** 2
        w1 = 1.0 / valid[1][1] ** 2
        T2_candidate = (w0 * valid[0][0] + w1 * valid[1][0]) / (w0 + w1)
    elif len(valid) == 1:
        T2_candidate = valid[0][0]
    elif np.isfinite(T2r_qe):
        T2_candidate = T2r_qe
    elif np.isfinite(T2h_qe):
        T2_candidate = T2h_qe
    else:
        T2_candidate = float("nan")

    T2_combined = float("nan")
    if np.isfinite(T2_candidate):
        # T2 <= 2*T1 always; T1_true_max (TRUE_PARAM_RANGES-anchored, see
        # above) is therefore also the correct reference scale for T2.
        plausible = (0.0 < T2_candidate <= T1_guard_threshold)
        guard_flags["T2"] = (True, not plausible)
        if plausible:
            T2_combined = T2_candidate

    if np.isfinite(T1_qe) and np.isfinite(T2_combined):
        T2_combined = min(T2_combined, 2.0 * T1_qe)

    try:
        exp_rb = StandardRB([0], lengths=QE_RB_LENGTHS,
                            num_samples=QE_RB_SAMPLES,
                            seed=int(seed), backend=backend)
        exp_rb.analysis.set_options(gate_error_ratio=False)
        data       = _run_exp(exp_rb, T1_true, T2_true, eps_true,
                              p0g1, p1g0, shots_rb, gate_time_ns, dt_ns,
                              native_target=native_target,
                              native_gate_names=native_gate_names)
        epc, _     = _extract(data, "EPC")
        if np.isfinite(epc) and epc > 0:
            # gates_per_clifford converts EPC (error per Clifford) into
            # epsilon_sx (error per native single-qubit gate). This ratio
            # is basis-dependent, so it must reflect the architecture's
            # real native gate set, not always the superconducting one.
            gates_per_clifford = QE_SX_PER_CLIFFORD
            if arch_name == "trapped_ion":
                native_target = _ionq_native_target(gate_time_ns)
                if native_target is not None:
                    # Re-generate this trial's exact RB circuit sample
                    # (same StandardRB object, same seed -> deterministic,
                    # identical circuits to what _run_exp already executed
                    # above) purely to measure its true IonQ-native gate
                    # count. This does not affect what was simulated.
                    measured = _native_gates_per_clifford(
                        exp_rb.circuits(), native_target)
                    if np.isfinite(measured) and measured > 0:
                        gates_per_clifford = measured
                    # else: qiskit_ionq unavailable or measurement failed;
                    # fall back to QE_SX_PER_CLIFFORD rather than crash the
                    # job, but this fallback is then a known-wrong basis
                    # for trapped_ion and should be flagged if it occurs.
            # neutral_atom: no verified native-gate provider exists (see
            # _qe_backend() docstring) -> stays on QE_SX_PER_CLIFFORD as a
            # disclosed, uncorrected proxy.
            eps_candidate = epc / gates_per_clifford
            # eps_max (ARCH_DEFAULTS' full physical bound, ~0.5) is far
            # looser than TRUE_PARAM_RANGES' actual sampled max (as little
            # as 0.002-0.01 depending on architecture) -- anchor the guard
            # to the true sampling range, same principle as T1/T2/dw above.
            eps_true_max = TRUE_PARAM_RANGES[arch_name]["eps"][1]
            eps_guard_threshold = QE_EPS_PLAUSIBILITY_MULT * eps_true_max
            plausible = (0.0 <= eps_candidate <= eps_guard_threshold)
            guard_flags["eps"] = (True, not plausible)
            if plausible:
                eps_qe = float(np.clip(eps_candidate, 0.0, eps_max))
    except Exception:
        pass

    return float(T1_qe), float(T2_combined), float(dw_qe), float(eps_qe), guard_flags


def _r2_point(true_arr, rec_arr):
    t = np.asarray(true_arr, float)
    r = np.asarray(rec_arr,  float)
    mask = np.isfinite(r) & np.isfinite(t)
    if mask.sum() < 2:
        return float("nan")
    t, r   = t[mask], r[mask]
    ss_res = np.sum((r - t) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def bootstrap_r2(true_vals, rec_vals, n_boot=N_BOOTSTRAP, rng_seed=0):
    t = np.asarray(true_vals, float)
    r = np.asarray(rec_vals,  float)
    mask = np.isfinite(r) & np.isfinite(t)
    if mask.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    t, r   = t[mask], r[mask]
    n      = len(t)
    ss_res = np.sum((r - t) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    r2_pt  = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    rng    = np.random.default_rng(rng_seed)
    boot   = []
    for _ in range(n_boot):
        idx  = rng.integers(0, n, size=n)
        ss_r = np.sum((r[idx] - t[idx]) ** 2)
        ss_t = np.sum((t[idx] - t[idx].mean()) ** 2)
        boot.append(float(1.0 - ss_r / ss_t) if ss_t > 0 else 0.0)
    b = np.array(boot)
    return r2_pt, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def nrmse(true_vals, rec_vals):
    t = np.asarray(true_vals, float)
    r = np.asarray(rec_vals,  float)
    mask = np.isfinite(r) & np.isfinite(t)
    if mask.sum() < 2:
        return float("nan")
    t, r  = t[mask], r[mask]
    rmse  = float(np.sqrt(np.mean((r - t) ** 2)))
    span  = float(t.max() - t.min())
    return rmse / span if span > 0 else float("nan")


def _worker_canary(args):
    warnings.filterwarnings("ignore")
    idx, (T1, T2, dw, eps), seed, budget, arch_name, p0g1_eff, p1g0_eff = args
    t0  = time.perf_counter()
    res = run_canary(T1, T2, dw, eps, seed, budget, arch_name, p0g1_eff, p1g0_eff)
    return idx, res, time.perf_counter() - t0


def _worker_qe(args):
    warnings.filterwarnings("ignore")
    idx, (T1, T2, dw, eps), budget, seed, arch_name, p0g1_eff, p1g0_eff, profile_dw_max = args
    t0  = time.perf_counter()
    res = run_qe(T1, T2, dw, eps, budget, seed, arch_name, p0g1_eff, p1g0_eff, profile_dw_max)
    return idx, res, time.perf_counter() - t0


def run_sweep(n_workers):
    rng_master  = np.random.default_rng(SEED)
    instances   = [sample_instance(rng_master, ARCH) for _ in range(N_INSTANCES)]

    rng_perturb = np.random.default_rng(SEED + 999)
    arch_def    = inv.ARCH_DEFAULTS[ARCH]
    perturbed   = [
        _perturb_instance(
            ARCH,
            inst[0], inst[1], inst[3],
            arch_def["p0_given_1"], arch_def["p1_given_0"],
            rng_perturb)
        for inst in instances
    ]

    # Computed ONCE per architecture (deterministic given arch_default
    # confidence's fixed log-spaced delay grid — no per-instance
    # dependency) and threaded through to every run_qe() call, rather than
    # recomputed per instance x budget. See run_qe()'s docstring comment
    # for why this specific value matters: it is the SAME dynamic quantity
    # sample_instance() used to generate the true dw values, so QE's
    # plausibility guard is judged against the correct scale.
    profile_dw_max = inv.BackendProfile.from_architecture(ARCH).dw_max_rad_s
    print(f"  profile_dw_max (dynamic, used for QE's dw guard) = "
          f"{profile_dw_max:.4f} rad/s", flush=True)

    rows = []

    print(f"  ARCH={ARCH}  N={N_INSTANCES}  budgets={SHOT_BUDGETS}  workers={n_workers}", flush=True)
    print(f"  Canary={CANARY_TOTAL} shots | QE=budget/4 per exp | osc_freq=arch-typical | RB lengths={QE_RB_LENGTHS}", flush=True)
    print(f"  Realistic noise: TLS T1 drift, 1/f T2 reduction, coherent over-rotation, SPAM drift", flush=True)
    print(f"  Perturbations applied once per instance; identical effective params fed to both methods.", flush=True)

    for bi, budget in enumerate(SHOT_BUDGETS):
        print(f"\n[{bi+1}/{len(SHOT_BUDGETS)}] budget={budget:,}  arch={ARCH}", flush=True)

        c_seed_base = SEED * 10_000 + bi * 1_000
        q_seed_base = c_seed_base + 500

        c_args = [
            (i,
             (perturbed[i][0], perturbed[i][1], instances[i][2], perturbed[i][2]),
             c_seed_base + i, budget, ARCH,
             perturbed[i][3], perturbed[i][4])
            for i in range(N_INSTANCES)
        ]
        q_args = [
            (i,
             (perturbed[i][0], perturbed[i][1], instances[i][2], perturbed[i][2]),
             budget, q_seed_base + i, ARCH,
             perturbed[i][3], perturbed[i][4], profile_dw_max)
            for i in range(N_INSTANCES)
        ]

        c_res  = [None] * N_INSTANCES;  c_time = [float("nan")] * N_INSTANCES
        q_res  = [None] * N_INSTANCES;  q_time = [float("nan")] * N_INSTANCES

        with mp.Pool(n_workers) as pool:
            for idx, res, t in pool.imap_unordered(_worker_canary, c_args, chunksize=4):
                c_res[idx]  = res;  c_time[idx] = t

        with mp.Pool(n_workers) as pool:
            for idx, res, t in pool.imap_unordered(_worker_qe, q_args, chunksize=4):
                q_res[idx]  = res;  q_time[idx] = t

        for res in q_res:
            gf = res[4]
            for pname, (had_candidate, rejected) in gf.items():
                if had_candidate:
                    QE_GUARD_TOTAL[ARCH][pname]   += 1
                    QE_GUARD_REJECTS[ARCH][pname] += int(rejected)

        true = {
            "T1":          [perturbed[i][0] for i in range(N_INSTANCES)],
            "T2":          [perturbed[i][1] for i in range(N_INSTANCES)],
            "delta_omega": [abs(instances[i][2]) for i in range(N_INSTANCES)],
            "epsilon_sx":  [perturbed[i][2] for i in range(N_INSTANCES)],
        }

        for pname, pidx in PARAMS:
            tv = true[pname]
            cv = [r[pidx] for r in c_res]
            qv = [r[pidx] for r in q_res]

            for method, vals, times in [
                    ("Canary",            cv, c_time),
                    ("QiskitExperiments", qv, q_time)]:
                r2pt, r2lo, r2hi = bootstrap_r2(tv, vals, rng_seed=bi * 100 + pidx)
                nr      = nrmse(tv, vals)
                n_valid = int(sum(1 for v in vals if np.isfinite(v)))
                mean_t  = float(np.nanmean(times))
                rows.append({
                    "budget":      budget,
                    "architecture": ARCH,
                    "parameter":   pname,
                    "method":      method,
                    "r2":          r2pt,
                    "r2_lo":       r2lo,
                    "r2_hi":       r2hi,
                    "nrmse":       nr,
                    "n_valid":     n_valid,
                    "n_total":     N_INSTANCES,
                    "mean_time_s": mean_t,
                })

            rc = _r2_point(tv, cv); nc = sum(1 for v in cv if np.isfinite(v))
            rq = _r2_point(tv, qv); nq = sum(1 for v in qv if np.isfinite(v))
            print(f"  {pname:12s}: Canary R2={rc:+.4f} ({nc}/{N_INSTANCES})"
                  f"  QE R2={rq:+.4f} ({nq}/{N_INSTANCES})", flush=True)

        save_benchmark_csv(rows)
        print(f"  [checkpoint] saved after budget={budget:,} "
              f"({bi+1}/{len(SHOT_BUDGETS)} budgets complete)", flush=True)

    print(f"\n  QE plausibility-guard summary for {ARCH} "
          f"(rejected/had-candidate, across all budgets):", flush=True)
    for pname in ("T1", "T2", "dw", "eps"):
        total    = QE_GUARD_TOTAL[ARCH].get(pname, 0)
        rejected = QE_GUARD_REJECTS[ARCH].get(pname, 0)
        pct      = (100.0 * rejected / total) if total else 0.0
        flag     = "  <-- check this" if pct > 50.0 else ""
        print(f"    {pname:4s}: {rejected:5d}/{total:5d} rejected "
              f"({pct:5.1f}%){flag}", flush=True)

    return rows


def save_benchmark_csv(rows):
    path = DATA_DIR / f"fig7_benchmark_{JOB_TAG}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BENCH_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {path}  ({len(rows)} rows)", flush=True)
    return path


def save_threshold_csv(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["architecture"], row["parameter"], row["method"])].append(
            (row["budget"], row["r2"]))

    thresh_rows = []
    max_b = max(SHOT_BUDGETS)
    for (arch_name, pname, method), pairs in sorted(groups.items()):
        pairs.sort()
        s95 = next((b for b, r2 in pairs if np.isfinite(r2) and r2 >= 0.95), None)
        s99 = next((b for b, r2 in pairs if np.isfinite(r2) and r2 >= 0.99), None)
        thresh_rows.append({
            "architecture":    arch_name,
            "parameter":       pname,
            "method":          method,
            "shots_to_R2_095": s95 if s95 else f"> {max_b:,}",
            "shots_to_R2_099": s99 if s99 else f"> {max_b:,}",
        })

    path = DATA_DIR / "fig7_threshold.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=THRESH_COLS)
        w.writeheader()
        w.writerows(thresh_rows)
    print(f"Saved: {path}", flush=True)

    by_arch_param = defaultdict(dict)
    for r in thresh_rows:
        by_arch_param[(r["architecture"], r["parameter"])][r["method"]] = r["shots_to_R2_095"]
    print(f"\n  Shots to R2>=0.95:")
    print(f"  {'Arch':16s}  {'Parameter':14s}  {'Canary':>10s}  {'QE':>10s}")
    print(f"  {'-'*56}")
    for (arch_name, pname), d in sorted(by_arch_param.items()):
        sc = d.get("Canary", "-")
        sq = d.get("QiskitExperiments", "-")
        print(f"  {arch_name:16s}  {pname:14s}  {str(sc):>10s}  {str(sq):>10s}")

    return path


def plot_results(rows):
    idx_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    architectures_present = []
    for row in rows:
        a = row["architecture"]
        if a not in architectures_present:
            architectures_present.append(a)
        idx_data[a][row["parameter"]][row["method"]][row["budget"]] = row

    architectures_present = [a for a in ALL_ARCHITECTURES if a in architectures_present]
    budgets = sorted(SHOT_BUDGETS)
    n_rows  = len(architectures_present)
    fig, axes = plt.subplots(n_rows, 4, figsize=(15, 3.6 * n_rows), squeeze=False)

    ARCH_LABEL = {"superconducting": "Superconducting", "trapped_ion": "Trapped ion",
                  "neutral_atom": "Neutral atom"}

    for row_i, arch_name in enumerate(architectures_present):
        for col, pname in enumerate(PORDER):
            ax = axes[row_i, col]

            for method in METHODS:
                xs, ys, los, his = [], [], [], []
                for b in budgets:
                    r = idx_data[arch_name][pname][method].get(b)
                    if r is None:
                        continue
                    xs.append(b)
                    if np.isfinite(r["r2"]):
                        ys.append(r["r2"])
                        los.append(r["r2"] - r["r2_lo"])
                        his.append(r["r2_hi"] - r["r2"])
                    else:
                        ys.append(float("nan"))
                        los.append(0.0)
                        his.append(0.0)

                xs  = np.array(xs); ys = np.array(ys)
                los = np.array(los); his = np.array(his)
                finite = np.isfinite(ys)

                if finite.any():
                    ax.fill_between(xs[finite], (ys - los)[finite], (ys + his)[finite],
                                    color=COLOUR[method], alpha=0.12)
                    ax.errorbar(xs[finite], ys[finite],
                                yerr=[los[finite], his[finite]],
                                color=COLOUR[method], marker=MARKER[method],
                                label=LABEL[method],
                                linewidth=1.8, markersize=5, capsize=3,
                                elinewidth=1.0, alpha=0.95)

            ax.axhline(0.95, color="#555", linestyle="--", linewidth=0.9, alpha=0.7)
            ax.axvline(CANARY_TOTAL, color=COLOUR["Canary"], linestyle=":",
                       linewidth=1.0, alpha=0.5)
            ax.set_xscale("log")
            ax.set_xlim(budgets[0] * 0.75, budgets[-1] * 1.35)
            ax.set_ylim(-0.15, 1.05)
            if row_i == 0:
                ax.set_title(PARAM_LATEX[pname], fontsize=12)
            if row_i == n_rows - 1:
                ax.set_xlabel("Total shots", fontsize=9)
            ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(4))
            ax.grid(True, which="major", alpha=0.25)
            ax.grid(True, which="minor", alpha=0.08)
            if col == 0:
                ax.set_ylabel(f"{ARCH_LABEL[arch_name]}\n" + r"$R^2$ (bootstrap median, 95% CI)",
                              fontsize=8.5)
            if col == 0 and row_i == 0:
                ax.legend(fontsize=6.5, loc="lower right", framealpha=0.85)

    fig.suptitle(
        "Fig. 7 - Canary vs. Qiskit Experiments: "
        r"$R^2$ vs. total shot budget across architectures"
        "\n(N=200 instances, 2000-resample bootstrap 95% CI; "
        "QE: osc_freq=arch-typical prior, arch-prior delays, equal budget; "
        "vertical dotted = Canary native budget; dashed = 0.95 target)",
        fontsize=9, y=1.01)

    plt.tight_layout()
    for fmt in ("pdf", "png"):
        p = FIG_DIR / f"fig7_benchmark.{fmt}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved: {p}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    import os
    warnings.filterwarnings("ignore")
    if sys.platform != "win32":
        mp.set_start_method("fork", force=True)
    n_workers = int(os.environ.get("N_WORKERS", max(1, mp.cpu_count() - 1)))

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"[{ts}] Fig. 7 benchmark - Canary vs. Qiskit Experiments", flush=True)
    print(f"  Architecture={ARCH}  Seed={SEED}  N={N_INSTANCES}  bootstrap={N_BOOTSTRAP}  workers={n_workers}", flush=True)

    rows = run_sweep(n_workers)
    save_benchmark_csv(rows)
    print(f"\n[job JOB_TAG={JOB_TAG}] "
         f"Skipping threshold CSV and plot — run 7_merge_results.py "
         f"after all matrix jobs complete to produce final outputs.",
         flush=True)