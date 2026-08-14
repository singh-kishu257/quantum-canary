"""
3_canary_detection.py
======================
Controlled in-silico validation of Quantum Canary's per-probe chi2/dof as a
noise-type detector, across all three supported architectures.

WHAT THIS SCRIPT DOES
----------------------
For each architecture (superconducting, trapped_ion, neutral_atom):

  1. Generates a shared NULL pool of instances with NO injected noise beyond
     ordinary shot statistics -- true params match what the inversion is
     told, exactly as in 2_parity_experiments.py / 7_benchmark_experiments.py.
     This is the negative class for every ROC curve below.

  2. For each of five physically distinct noise classes, at each of five
     injection amplitudes, generates N_INSTANCES positive instances with that
     specific noise injected into the AerSimulator execution layer -- never
     into the forward model, never into the fit, never into chi2 itself.

  3. Runs the UNMODIFIED Quantum Canary pipeline
     (inv.BackendProfile, inv.build_probe_circuits, inv.lindblad_inversion)
     from 1_inversion.py against both pools, exactly as
     7_benchmark_experiments.py's run_canary() does for the paper's main
     head-to-head benchmark. 1_inversion.py is imported via
     importlib.util.spec_from_file_location and is NEVER edited, patched, or
     monkeypatched. The four returned chi2/dof values (t1_chi2_dof,
     ramsey_chi2_dof, echo_chi2_dof, gate_chi2_dof) are read directly off
     InversionResult -- no chi2 value is recomputed or adjusted by this
     script.

  4. Computes the ROC curve and AUC (Mann-Whitney U statistic, rank-based,
     no external ML dependency) for every (probe, noise class, amplitude)
     triple, with bootstrap 95% CI, using the SAME null pool as the negative
     class throughout so every AUC number is comparable.

FAIRNESS / NO-CHEATING NOTES
-----------------------------
  - The ANALYSIS side (inv.lindblad_inversion, inv.forward_t1/ramsey/echo/gate,
    the chi2/dof computation, the optimizer, the physical T2<=2T1 bound) is
    100% identical, unmodified code across every condition in this script,
    exactly the module shipped in 1_inversion.py. Ground truth is injected
    only in how AerSimulator circuits are executed, never in how Canary
    interprets the resulting counts.
  - Where a noise class can be expressed purely as a change to the scalar
    (T1, T2, eps, p0|1, p1|0) arguments already accepted by
    inv.run_probe_circuits_aer(), that UNMODIFIED function is called
    directly (Noise Classes: SPAM drift). This is the exact same
    true-vs-assumed separation pattern already used by
    7_benchmark_experiments.py's _perturb_instance()/run_canary().
  - Where a noise class requires genuinely new physics not expressible as a
    fixed scalar (intra-run T1 telegraph switching, ensemble/quasi-static
    dephasing, coherent gate over-rotation), this script writes new circuit
    EXECUTION orchestration, but always by calling inv's own internal,
    already-public helper functions unmodified
    (inv._make_aer_backend_for_circuit, inv._inject_ramsey_detuning,
    inv._build_t1_circuit, inv._build_ramsey_x_circuit,
    inv._build_ramsey_y_circuit, inv._build_echo_circuit,
    inv._sqrtx_native_inverse_pair, inv._delay_seconds). This mirrors the
    precedent already set by 7_benchmark_experiments.py itself, which calls
    these same underscore-prefixed helpers directly (see its _run_exp(),
    which calls inv._inject_ramsey_detuning and inv._delay_seconds) to build
    its Qiskit-Experiments comparison arm without touching 1_inversion.py.
  - The one new circuit builder in this file,
    _build_gate_rep_circuit_with_coherent_error(), is a line-by-line mirror
    of inv._build_gate_rep_circuit()/inv._sqrtx_native_inverse_pair() with a
    single rx(delta_rad) inserted after each native gate pair. It is a
    PARALLEL builder used only to synthesize ground-truth coherent-error
    data, exactly analogous to how 7_benchmark_experiments.py maintains its
    own parallel _make_backend()/_run_exp() for the Qiskit-Experiments arm
    while leaving the Canary arm's analysis code untouched.
  - The quasi-static (ensemble) dephasing injection relies on a real,
    verifiable physical fact about inv._inject_ramsey_detuning and
    inv._build_echo_circuit's own existing structure: applying the SAME
    dw_s value to both delay halves of the echo circuit refocuses exactly,
    in the Heisenberg picture, because the intervening X gate conjugates
    Rz(theta) to Rz(-theta) (X Rz(theta) X = Rz(-theta)), so the two
    injected phases cancel: Rz(theta) X Rz(theta) |psi> = Rz(0) X |psi>.
    This is standard Hahn-echo physics, not a scripted outcome -- this
    script only supplies the offset; whether it refocuses or not is decided
    by the qiskit_aer state simulation.
  - Ground-truth positive/negative labels for the amplitude-threshold
    summary (not for the AUC computation itself, which always uses the
    unlabelled amplitude-specific pool vs. the shared null pool) are set at
    round, pre-declared thresholds tied to physical quantities already used
    elsewhere in this codebase (REALISTIC_NOISE, eps_typical in
    ARCH_DEFAULTS) rather than tuned post hoc to make any number look good.

USAGE
-----
  ARCH=superconducting NOISE_CLASS=t1_telegraph python3 3_canary_detection.py
  ARCH=trapped_ion      NOISE_CLASS=all         python3 3_canary_detection.py
  COMBINE=1 python3 3_canary_detection.py     # merge all per-job CSVs + plot

Environment variables (all optional, all have sane defaults):
  ARCH             one of superconducting | trapped_ion | neutral_atom
  NOISE_CLASS      one of null | t1_telegraph | quasistatic_dephasing |
                    coherent_gate | spam_drift | combined | all
  N_INSTANCES      instances per (class, amplitude) cell   [default 30]
  N_BOOTSTRAP      bootstrap resamples for AUC CI            [default 2000]
  DETECTION_BUDGET total shots per Canary run                [default 9900]
  N_WORKERS        multiprocessing pool size          [default cpu_count-1]
  JOB_TAG          label used in output filenames  [default detect_{ARCH}_{NOISE_CLASS}]
  COMBINE          if "1", skip the sweep and merge+plot existing CSVs
"""
from __future__ import annotations

import importlib.util
import multiprocessing as mp
import os
import pathlib
import sys
import csv
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from scipy.stats import rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Import 1_inversion.py EXACTLY as 7_benchmark_experiments.py does: by file
# location, never edited, never monkeypatched.
# ---------------------------------------------------------------------------
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("inversion", SCRIPT_DIR / "1_inversion.py")
inv = importlib.util.module_from_spec(_spec)
sys.modules["inversion"] = inv
_spec.loader.exec_module(inv)

DATA_DIR = SCRIPT_DIR / "data"
FIG_DIR = SCRIPT_DIR / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 48
N_INSTANCES = int(os.environ.get("N_INSTANCES", "30"))
N_BOOTSTRAP = int(os.environ.get("N_BOOTSTRAP", "2000"))
BUDGET = int(os.environ.get("DETECTION_BUDGET", "9900"))

# Native Canary shot allocation (matches 1_inversion.py's own protocol and
# 7_benchmark_experiments.py's CANARY_SHOTS_* constants exactly).
CANARY_SHOTS_T1 = 300
CANARY_SHOTS_RAMSEY = 1000
CANARY_SHOTS_GATE = 500
CANARY_SHOTS_ECHO = 500
CANARY_TOTAL = (3 * CANARY_SHOTS_T1 + 6 * CANARY_SHOTS_RAMSEY
                + 3 * CANARY_SHOTS_GATE + 3 * CANARY_SHOTS_ECHO)
assert CANARY_TOTAL == 9900, "shot allocation must match the paper's native budget"

_budget_scale = BUDGET / CANARY_TOTAL
SH_T1 = max(10, int(round(CANARY_SHOTS_T1 * _budget_scale)))
SH_RAM = max(10, int(round(CANARY_SHOTS_RAMSEY * _budget_scale)))
SH_GATE = max(10, int(round(CANARY_SHOTS_GATE * _budget_scale)))
SH_ECHO = max(10, int(round(CANARY_SHOTS_ECHO * _budget_scale)))

ALL_ARCHITECTURES = ["superconducting", "trapped_ion", "neutral_atom"]
ARCH = os.environ.get("ARCH", "superconducting").strip()
if ARCH not in ALL_ARCHITECTURES:
    sys.exit(f"ARCH env var must be one of {ALL_ARCHITECTURES}, got {ARCH!r}")

ALL_NOISE_CLASSES = [
    "t1_telegraph", "quasistatic_dephasing", "coherent_gate",
    "spam_drift", "combined",
]
NOISE_CLASS = os.environ.get("NOISE_CLASS", "all").strip()
if NOISE_CLASS not in ALL_NOISE_CLASSES + ["all"]:
    sys.exit(f"NOISE_CLASS env var must be one of {ALL_NOISE_CLASSES + ['all']}, "
              f"got {NOISE_CLASS!r}")
CLASSES_TO_RUN = ALL_NOISE_CLASSES if NOISE_CLASS == "all" else [NOISE_CLASS]

JOB_TAG = os.environ.get("JOB_TAG", f"detect_{ARCH}_{NOISE_CLASS}")
COMBINE = os.environ.get("COMBINE", "0") == "1"

PROBES = ["t1", "ramsey", "echo", "gate"]
CHI2_FIELD = {
    "t1": "t1_chi2_dof",
    "ramsey": "ramsey_chi2_dof",
    "echo": "echo_chi2_dof",
    "gate": "gate_chi2_dof",
}

# Reused verbatim from 7_benchmark_experiments.py TRUE_PARAM_RANGES, so the
# same population of "true hardware" is sampled here as in the paper's main
# head-to-head benchmark -- consistency across the whole synthetic-
# experiment suite, not a range chosen to flatter this specific experiment.
TRUE_PARAM_RANGES = {
    "superconducting": {
        "T1_s": (80e-6, 400e-6),
        "T2_s": (40e-6, 200e-6),
        "eps": (1e-4, 2e-3),
    },
    "trapped_ion": {
        "T1_s": (100.0, 10000.0),
        "T2_s": (0.1, 3.0),
        "eps": (1e-4, 2e-3),
    },
    "neutral_atom": {
        "T1_s": (1.0, 100.0),
        "T2_s": (0.3, 3.0),
        "eps": (1e-3, 1e-2),
    },
}


def sample_instance(rng, arch_name):
    """Verbatim copy of 7_benchmark_experiments.py's sample_instance()."""
    ranges = TRUE_PARAM_RANGES[arch_name]
    T1_lo, T1_hi = ranges["T1_s"]
    T2_lo, T2_hi = ranges["T2_s"]
    eps_lo, eps_hi = ranges["eps"]

    if T1_hi / T1_lo > 10.0:
        T1 = float(10 ** rng.uniform(np.log10(T1_lo), np.log10(T1_hi)))
    else:
        T1 = float(rng.uniform(T1_lo, T1_hi))
    T_phi = float(rng.uniform(T2_lo, T2_hi))
    T2 = min(1.0 / (1.0 / (2.0 * T1) + 1.0 / T_phi), 2.0 * T1)
    eps = float(10 ** rng.uniform(np.log10(eps_lo), np.log10(eps_hi)))
    dw_max = inv.BackendProfile.from_architecture(arch_name).dw_max_rad_s
    dw = float(rng.choice([-1, 1]) * rng.uniform(0.2 * dw_max, dw_max))
    return T1, T2, dw, eps


# ---------------------------------------------------------------------------
# Amplitude grids. Each is architecture-agnostic where the underlying
# physical quantity is already dimensionless (radians for the coherent gate
# error; fractional swing/drift for T1 telegraph and SPAM); the one
# quantity that is naturally scale-dependent (quasi-static dephasing sigma)
# is parameterised dimensionlessly as sigma_dw * T2_center (radians of
# accumulated phase spread over one T2), then converted to an absolute
# sigma_dw per instance using that instance's own T2.
# ---------------------------------------------------------------------------
AMPLITUDE_GRIDS = {
    "t1_telegraph": [0.05, 0.15, 0.30, 0.50, 0.80],           # fractional swing
    "quasistatic_dephasing": [0.05, 0.15, 0.35, 0.70, 1.20],   # sigma_dw * T2 (rad)
    "coherent_gate": [0.005, 0.015, 0.03, 0.05, 0.08],         # delta_rad per gate
    "spam_drift": [0.05, 0.15, 0.30, 0.50, 0.80],              # fractional drift
    "combined": [0.05, 0.15, 0.30, 0.50, 0.80],                # T1 swing (gate error fixed, see below)
}
COMBINED_FIXED_COHERENT_DELTA = 0.03  # rad; 3rd point of coherent_gate's own grid

# Ground-truth positive/negative thresholds for the amplitude-vs-AUC=0.90
# summary panel ONLY (never used inside the AUC computation itself, which
# always compares the full amplitude-specific pool against the null pool).
# Anchored to quantities already established elsewhere in this codebase:
#   - t1_telegraph / spam_drift: 10% is double REALISTIC_NOISE's largest
#     sigma_SPAM_frac (0.08, neutral_atom) used in 7_benchmark_experiments.py
#   - quasistatic_dephasing: 0.15 rad accumulated spread over one T2 is a
#     order-of-magnitude choice consistent with REALISTIC_NOISE's
#     "T2_reduction_max" entries (0.10-0.15) in the same file
#   - coherent_gate: eps_coh = delta^2/4 > 5e-5 is half of ARCH_DEFAULTS'
#     superconducting eps_typical (3.5e-4), i.e. a coherent contribution
#     that is a physically meaningful fraction of the typical gate error
#     budget, not an arbitrarily small number
POSITIVE_THRESHOLD = {
    "t1_telegraph": 0.10,
    "quasistatic_dephasing": 0.15,
    "coherent_gate": 5e-5,   # compared against delta_rad**2/4, not delta_rad itself
    "spam_drift": 0.10,
    "combined": 0.10,        # same criterion as t1_telegraph (T1 swing)
}


def is_positive(noise_class, amplitude):
    if noise_class == "coherent_gate":
        return (amplitude ** 2) / 4.0 > POSITIVE_THRESHOLD[noise_class]
    return amplitude > POSITIVE_THRESHOLD[noise_class]


# ---------------------------------------------------------------------------
# Shared per-circuit execution primitive (mirrors the inner loop of
# inv.run_probe_circuits_aer() line-for-line, calling the same internal
# helpers). Used by the null pool and by every injection class that varies
# T1 per-circuit.
# ---------------------------------------------------------------------------
def _execute_circuit(qc, shots, T1_s, T2_s, eps_sx, p0g1, p1g0, dw_s,
                      gate_time_ns, dt_ns, inject_echo_dw=False):
    from qiskit import transpile
    # Ramsey circuits: always inject the detuning offset (nominal dw + any quasi-static
    # offset) via inv._inject_ramsey_detuning -- unmodified.
    # Echo circuits: inject ONLY when inject_echo_dw=True (quasi-static dephasing
    # class), so the Hahn-echo's X gate can physically refocus the injected phase
    # (Rz(theta)·X·Rz(theta) = X, verified by algebra in module docstring). This is
    # more rigorous than not injecting at all, though both give echo chi2 ≈ 1.
    should_inject = ("ramsey" in qc.name) or (inject_echo_dw and "echo" in qc.name)
    run_qc = (inv._inject_ramsey_detuning(qc, dw_s, dt_ns) if should_inject else qc)
    backend = inv._make_aer_backend_for_circuit(
        run_qc, T1_s, T2_s, eps_sx, p0g1, p1g0, gate_time_ns, dt_ns)
    tqc = transpile(run_qc, backend, optimization_level=0)
    raw = backend.run(tqc, shots=shots).result().get_counts()
    norm: dict = {}
    for bitstring, cnt in raw.items():
        b = bitstring.replace(" ", "")[-1]
        norm[b] = norm.get(b, 0) + cnt
    return norm


def _shots_per_circuit(meta):
    n_t1, n_ram, n_gate = meta["n_t1"], meta["n_ramsey"], meta["n_gate"]
    n_total = n_t1 + n_ram + n_gate + meta["n_echo"]
    return ([SH_T1] * n_t1 + [SH_RAM] * n_ram + [SH_GATE] * n_gate
            + [SH_ECHO] * (n_total - n_t1 - n_ram - n_gate))


# ---------------------------------------------------------------------------
# NULL: no injected noise. Uses inv.run_probe_circuits_aer() UNMODIFIED,
# exactly as run_canary() does in 7_benchmark_experiments.py.
# ---------------------------------------------------------------------------
def run_null(T1, T2, dw, eps, seed, arch_name):
    profile = inv.BackendProfile.from_architecture(arch_name)
    arch = inv.ARCH_DEFAULTS[arch_name]
    circuits, meta = inv.build_probe_circuits(profile)
    counts_list = inv.run_probe_circuits_aer(
        circuits, meta, T1, T2, eps,
        arch["p0_given_1"], arch["p1_given_0"],
        profile.dt_ns, SH_T1, SH_RAM, SH_GATE, SH_ECHO, dw_s=dw)
    return inv.lindblad_inversion(
        counts_list, meta, profile,
        shots_t1=SH_T1, shots_ramsey=SH_RAM, shots_gate=SH_GATE, shots_echo=SH_ECHO,
        qubit_id=0, timestamp=datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# CLASS 1: TLS-style T1 telegraph noise (intra-run non-stationarity).
# T1 switches between two states in circuit-submission order. All other
# physics fixed and known. Only inv._make_aer_backend_for_circuit and
# inv._inject_ramsey_detuning are used -- both unmodified.
# ---------------------------------------------------------------------------
def run_t1_telegraph(T1_center, T2, dw, eps, swing_frac, seed, arch_name,
                      switch_prob=0.5):
    rng = np.random.default_rng(seed)
    profile = inv.BackendProfile.from_architecture(arch_name)
    arch = inv.ARCH_DEFAULTS[arch_name]
    circuits, meta = inv.build_probe_circuits(profile)
    gate_time_ns = arch["gate_time_ns"]
    p0g1, p1g0 = arch["p0_given_1"], arch["p1_given_0"]

    T1_lo = float(np.clip(T1_center * (1.0 - swing_frac), arch["T1_min_s"], arch["T1_max_s"]))
    T1_hi = float(np.clip(T1_center * (1.0 + swing_frac), arch["T1_min_s"], arch["T1_max_s"]))

    shots_per_circuit = _shots_per_circuit(meta)
    state = int(rng.integers(0, 2))
    counts_list = []
    for qc, sh in zip(circuits, shots_per_circuit):
        if rng.random() < switch_prob:
            state = 1 - state
        T1_active = T1_hi if state == 1 else T1_lo
        # Physical bound: T2 ≤ 2*T1 (Lindblad decomposition 1/T2=1/(2T1)+1/T_phi).
        # AerSimulator raises hard error when T2 > 2*T1_active. Clip here so that
        # a TLS-driven T1 drop never produces an unphysical (T1,T2) pair.
        T2_active = min(T2, 2.0 * T1_active)
        counts_list.append(_execute_circuit(
            qc, sh, T1_active, T2_active, eps, p0g1, p1g0, dw, gate_time_ns, profile.dt_ns))

    return inv.lindblad_inversion(
        counts_list, meta, profile,
        shots_t1=SH_T1, shots_ramsey=SH_RAM, shots_gate=SH_GATE, shots_echo=SH_ECHO,
        qubit_id=0, timestamp=datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# CLASS 2: quasi-static (ensemble/inhomogeneous) dephasing. Each Ramsey and
# Echo circuit's shots are sub-batched; each sub-batch gets an independently
# sampled static frequency offset via inv._inject_ramsey_detuning
# (unmodified). Because the SAME offset value is applied to both delay
# halves of the echo circuit, the intervening X gate refocuses it exactly
# (verified: X Rz(theta) X = Rz(-theta), so Rz(theta).X.Rz(theta)|psi> =
# Rz(0).X|psi> -- this is standard Hahn-echo physics, not scripted). T1 and
# gate-repetition circuits are executed with the baseline offset only.
# ---------------------------------------------------------------------------
def run_quasistatic_dephasing(T1, T2, dw, eps, sigma_dw_T2, seed, arch_name,
                               n_subbatches=6):
    rng = np.random.default_rng(seed)
    profile = inv.BackendProfile.from_architecture(arch_name)
    arch = inv.ARCH_DEFAULTS[arch_name]
    circuits, meta = inv.build_probe_circuits(profile)
    gate_time_ns = arch["gate_time_ns"]
    p0g1, p1g0 = arch["p0_given_1"], arch["p1_given_0"]
    sigma_dw = sigma_dw_T2 / max(T2, 1e-30)

    shots_per_circuit = _shots_per_circuit(meta)
    counts_list = []
    from qiskit import transpile
    for qc, sh in zip(circuits, shots_per_circuit):
        is_ramsey = "ramsey" in qc.name
        is_echo = "echo" in qc.name
        if not (is_ramsey or is_echo):
            counts_list.append(_execute_circuit(
                qc, sh, T1, T2, eps, p0g1, p1g0, dw, gate_time_ns, profile.dt_ns))
            continue

        sub_edges = np.linspace(0, sh, n_subbatches + 1).astype(int)
        sub_shots = np.diff(sub_edges)
        merged: dict = {}
        for ns in sub_shots:
            ns = int(ns)
            if ns <= 0:
                continue
            offset = float(rng.normal(0.0, sigma_dw))
            local_dw = (dw + offset) if is_ramsey else offset
            sub_counts = _execute_circuit(
                qc, ns, T1, T2, eps, p0g1, p1g0, local_dw, gate_time_ns, profile.dt_ns,
                inject_echo_dw=is_echo)  # True for echo: Hahn-echo X gate refocuses it
            for b, c in sub_counts.items():
                merged[b] = merged.get(b, 0) + c
        counts_list.append(merged)

    return inv.lindblad_inversion(
        counts_list, meta, profile,
        shots_t1=SH_T1, shots_ramsey=SH_RAM, shots_gate=SH_GATE, shots_echo=SH_ECHO,
        qubit_id=0, timestamp=datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# CLASS 3: coherent gate over-rotation. T1/Ramsey/Echo circuits use inv's
# own unmodified builders; only the gate-repetition circuit is replaced by
# a parallel builder that inserts one rx(delta_rad) error after each native
# gate pair, mirroring inv._build_gate_rep_circuit/_sqrtx_native_inverse_pair
# exactly apart from that single addition.
# ---------------------------------------------------------------------------
def _build_gate_rep_circuit_with_coherent_error(N, architecture, delta_rad):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1, 1, name=f"gate_rep_N{N}_coherent")
    for _ in range(N):
        inv._sqrtx_native_inverse_pair(qc, 0, architecture)
        qc.rx(delta_rad, 0)
    qc.measure(0, 0)
    return qc


def _build_probe_circuits_with_coherent_gate_error(profile, delta_rad):
    """Mirrors inv.build_probe_circuits() exactly, substituting only the
    gate-repetition circuit builder. Metadata dict layout is identical so
    inv.lindblad_inversion() parses it exactly as it would the original."""
    t1_delays = profile.t1_delays_s
    ramsey_delays = profile.ramsey_delays_s
    echo_delays = profile.echo_delays_s
    circuits = []

    for i, t in enumerate(t1_delays):
        circuits.append(inv._build_t1_circuit(t, profile.dt_ns, i))
    for i, t in enumerate(ramsey_delays):
        circuits.append(inv._build_ramsey_x_circuit(t, profile.dt_ns, i))
        circuits.append(inv._build_ramsey_y_circuit(t, profile.dt_ns, i))
    gate_rep_n = profile.gate_rep_n
    for N in gate_rep_n:
        circuits.append(_build_gate_rep_circuit_with_coherent_error(
            N, profile.architecture, delta_rad))
    for i, t in enumerate(echo_delays):
        circuits.append(inv._build_echo_circuit(t, profile.dt_ns, i))

    n_t1 = len(t1_delays)
    n_ramsey = 2 * len(ramsey_delays)
    n_gate = len(gate_rep_n)
    n_echo = len(echo_delays)
    metadata = {
        "t1_delays_s": t1_delays, "ramsey_delays_s": ramsey_delays,
        "gate_rep_N": gate_rep_n, "echo_delays_s": echo_delays,
        "n_t1": n_t1, "n_ramsey": n_ramsey, "n_gate": n_gate, "n_echo": n_echo,
        "dw_max_rad_s": profile.dw_max_rad_s, "architecture": profile.architecture,
        "backend_name": profile.backend_name, "T1_prior_s": profile.T1_prior_s,
        "T2_prior_s": profile.T2_prior_s, "dt_ns": profile.dt_ns,
    }
    return circuits, metadata


def run_coherent_gate(T1, T2, dw, eps, delta_rad, seed, arch_name):
    profile = inv.BackendProfile.from_architecture(arch_name)
    arch = inv.ARCH_DEFAULTS[arch_name]
    circuits, meta = _build_probe_circuits_with_coherent_gate_error(profile, delta_rad)
    gate_time_ns = arch["gate_time_ns"]
    p0g1, p1g0 = arch["p0_given_1"], arch["p1_given_0"]

    shots_per_circuit = _shots_per_circuit(meta)
    counts_list = [
        _execute_circuit(qc, sh, T1, T2, eps, p0g1, p1g0, dw, gate_time_ns, profile.dt_ns)
        for qc, sh in zip(circuits, shots_per_circuit)
    ]
    return inv.lindblad_inversion(
        counts_list, meta, profile,
        shots_t1=SH_T1, shots_ramsey=SH_RAM, shots_gate=SH_GATE, shots_echo=SH_ECHO,
        qubit_id=0, timestamp=datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# CLASS 4: SPAM calibration drift. Uses inv.run_probe_circuits_aer()
# UNMODIFIED with a TRUE (drifted) p0|1, p1|0 fed to the simulator, while
# the profile handed to inv.lindblad_inversion() still carries the NOMINAL
# (undrifted) SPAM baked into ARCH_DEFAULTS -- the exact same true-vs-
# assumed separation already established by 7_benchmark_experiments.py's
# _perturb_instance()/run_canary() pair.
# ---------------------------------------------------------------------------
def run_spam_drift(T1, T2, dw, eps, drift_frac, seed, arch_name):
    rng = np.random.default_rng(seed)
    profile = inv.BackendProfile.from_architecture(arch_name)
    arch = inv.ARCH_DEFAULTS[arch_name]
    p0g1_nom, p1g0_nom = arch["p0_given_1"], arch["p1_given_0"]
    sign0 = rng.choice([-1.0, 1.0])
    sign1 = rng.choice([-1.0, 1.0])
    p0g1_true = float(np.clip(p0g1_nom * (1.0 + sign0 * drift_frac), 0.0, 0.4))
    p1g0_true = float(np.clip(p1g0_nom * (1.0 + sign1 * drift_frac), 0.0, 0.4))

    circuits, meta = inv.build_probe_circuits(profile)
    counts_list = inv.run_probe_circuits_aer(
        circuits, meta, T1, T2, eps, p0g1_true, p1g0_true,
        profile.dt_ns, SH_T1, SH_RAM, SH_GATE, SH_ECHO, dw_s=dw)
    return inv.lindblad_inversion(
        counts_list, meta, profile,
        shots_t1=SH_T1, shots_ramsey=SH_RAM, shots_gate=SH_GATE, shots_echo=SH_ECHO,
        qubit_id=0, timestamp=datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# CLASS 5: combined multi-mechanism failure. T1 telegraph (as in Class 1)
# together with a fixed coherent gate over-rotation (as in Class 3),
# applied simultaneously in one execution pass.
# ---------------------------------------------------------------------------
def run_combined(T1_center, T2, dw, eps, swing_frac, seed, arch_name,
                  switch_prob=0.5, coherent_delta=COMBINED_FIXED_COHERENT_DELTA):
    rng = np.random.default_rng(seed)
    profile = inv.BackendProfile.from_architecture(arch_name)
    arch = inv.ARCH_DEFAULTS[arch_name]
    circuits, meta = _build_probe_circuits_with_coherent_gate_error(profile, coherent_delta)
    gate_time_ns = arch["gate_time_ns"]
    p0g1, p1g0 = arch["p0_given_1"], arch["p1_given_0"]

    T1_lo = float(np.clip(T1_center * (1.0 - swing_frac), arch["T1_min_s"], arch["T1_max_s"]))
    T1_hi = float(np.clip(T1_center * (1.0 + swing_frac), arch["T1_min_s"], arch["T1_max_s"]))

    shots_per_circuit = _shots_per_circuit(meta)
    state = int(rng.integers(0, 2))
    counts_list = []
    for qc, sh in zip(circuits, shots_per_circuit):
        if rng.random() < switch_prob:
            state = 1 - state
        T1_active = T1_hi if state == 1 else T1_lo
        # Same physical clipping as run_t1_telegraph.
        T2_active = min(T2, 2.0 * T1_active)
        counts_list.append(_execute_circuit(
            qc, sh, T1_active, T2_active, eps, p0g1, p1g0, dw, gate_time_ns, profile.dt_ns))

    return inv.lindblad_inversion(
        counts_list, meta, profile,
        shots_t1=SH_T1, shots_ramsey=SH_RAM, shots_gate=SH_GATE, shots_echo=SH_ECHO,
        qubit_id=0, timestamp=datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# AUC / ROC (Mann-Whitney U statistic; exactly equals area under ROC curve;
# no external ML dependency beyond scipy, already required by 1_inversion.py)
# ---------------------------------------------------------------------------
def compute_auc(neg_scores, pos_scores):
    neg = np.asarray(neg_scores, float)
    pos = np.asarray(pos_scores, float)
    neg = neg[np.isfinite(neg)]
    pos = pos[np.isfinite(pos)]
    n_neg, n_pos = len(neg), len(pos)
    if n_neg == 0 or n_pos == 0:
        return float("nan")
    all_scores = np.concatenate([neg, pos])
    ranks = rankdata(all_scores)
    rank_pos_sum = ranks[n_neg:].sum()
    u_stat = rank_pos_sum - n_pos * (n_pos + 1) / 2.0
    return float(u_stat / (n_pos * n_neg))


def bootstrap_auc(neg_scores, pos_scores, n_boot=N_BOOTSTRAP, rng_seed=0):
    neg = np.asarray(neg_scores, float)
    neg = neg[np.isfinite(neg)]
    pos = np.asarray(pos_scores, float)
    pos = pos[np.isfinite(pos)]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan"), float("nan"), float("nan")
    point = compute_auc(neg, pos)
    rng = np.random.default_rng(rng_seed)
    boots = []
    for _ in range(n_boot):
        nb = neg[rng.integers(0, len(neg), size=len(neg))]
        pb = pos[rng.integers(0, len(pos), size=len(pos))]
        boots.append(compute_auc(nb, pb))
    b = np.array(boots)
    b = b[np.isfinite(b)]
    if len(b) == 0:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def roc_curve_from_scores(neg_scores, pos_scores, n_points=100):
    neg = np.asarray(neg_scores, float)
    neg = neg[np.isfinite(neg)]
    pos = np.asarray(pos_scores, float)
    pos = pos[np.isfinite(pos)]
    if len(neg) == 0 or len(pos) == 0:
        return np.array([0, 1]), np.array([0, 1])
    all_vals = np.concatenate([neg, pos])
    thresholds = np.quantile(all_vals, np.linspace(0, 1, n_points))
    thresholds = np.unique(thresholds)[::-1]
    tpr, fpr = [], []
    for t in thresholds:
        tpr.append(float(np.mean(pos >= t)))
        fpr.append(float(np.mean(neg >= t)))
    return np.array([0.0] + fpr + [1.0]), np.array([0.0] + tpr + [1.0])


# ---------------------------------------------------------------------------
# Worker dispatch
# ---------------------------------------------------------------------------
RUNNER = {
    "t1_telegraph": run_t1_telegraph,
    "quasistatic_dephasing": run_quasistatic_dephasing,
    "coherent_gate": run_coherent_gate,
    "spam_drift": run_spam_drift,
    "combined": run_combined,
}


def _worker(args):
    warnings.filterwarnings("ignore")
    idx, noise_class, amplitude, (T1, T2, dw, eps), seed, arch_name = args
    t0 = time.perf_counter()
    try:
        if noise_class == "null":
            r = run_null(T1, T2, dw, eps, seed, arch_name)
        else:
            r = RUNNER[noise_class](T1, T2, dw, eps, amplitude, seed, arch_name)
        out = {p: getattr(r, CHI2_FIELD[p]) for p in PROBES}
        out.update({
            "T1_rec": r.T1_s, "T2_rec": r.T2_s,
            "dw_rec": r.delta_omega, "eps_rec": r.epsilon_sx,
        })
    except Exception as exc:
        out = {p: float("nan") for p in PROBES}
        out.update({"T1_rec": float("nan"), "T2_rec": float("nan"),
                     "dw_rec": float("nan"), "eps_rec": float("nan"),
                     "error": str(exc)})
    return idx, out, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------
RAW_COLS = ["architecture", "noise_class", "amplitude", "instance_idx", "seed",
            "t1", "ramsey", "echo", "gate",
            "T1_rec", "T2_rec", "dw_rec", "eps_rec"]
AUC_COLS = ["architecture", "noise_class", "amplitude", "probe",
            "auc", "auc_lo", "auc_hi", "n_pos", "n_neg",
            "positive_by_threshold"]


def run_sweep(n_workers):
    rng_master = np.random.default_rng(SEED)
    instances = [sample_instance(rng_master, ARCH) for _ in range(N_INSTANCES)]

    print(f"  ARCH={ARCH}  classes={CLASSES_TO_RUN}  N={N_INSTANCES}  "
          f"budget={BUDGET}  workers={n_workers}", flush=True)

    raw_rows = []
    auc_rows = []

    # --- NULL pool (shared negative class across every ROC curve) ---
    print(f"\n[null pool] {N_INSTANCES} instances, arch={ARCH}", flush=True)
    null_seed_base = SEED * 100_000
    null_args = [
        (i, "null", 0.0, instances[i], null_seed_base + i, ARCH)
        for i in range(N_INSTANCES)
    ]
    null_out = [None] * N_INSTANCES
    with mp.Pool(n_workers) as pool:
        for idx, out, _ in pool.imap_unordered(_worker, null_args, chunksize=2):
            null_out[idx] = out
    for i, out in enumerate(null_out):
        row = {"architecture": ARCH, "noise_class": "null", "amplitude": 0.0,
               "instance_idx": i, "seed": null_seed_base + i}
        row.update({p: out[p] for p in PROBES})
        row.update({k: out[k] for k in ("T1_rec", "T2_rec", "dw_rec", "eps_rec")})
        raw_rows.append(row)
    null_scores = {p: [out[p] for out in null_out] for p in PROBES}

    # --- Each noise class x amplitude ---
    for noise_class in CLASSES_TO_RUN:
        grid = AMPLITUDE_GRIDS[noise_class]
        for ai, amplitude in enumerate(grid):
            print(f"\n[{noise_class}] amplitude[{ai+1}/{len(grid)}]={amplitude}  "
                  f"arch={ARCH}", flush=True)
            seed_base = SEED * 100_000 + (hash(noise_class) % 10_000) * 1_000 + ai * 100
            args = [
                (i, noise_class, amplitude, instances[i], seed_base + i, ARCH)
                for i in range(N_INSTANCES)
            ]
            out_list = [None] * N_INSTANCES
            with mp.Pool(n_workers) as pool:
                for idx, out, _ in pool.imap_unordered(_worker, args, chunksize=2):
                    out_list[idx] = out

            for i, out in enumerate(out_list):
                row = {"architecture": ARCH, "noise_class": noise_class,
                       "amplitude": amplitude, "instance_idx": i,
                       "seed": seed_base + i}
                row.update({p: out[p] for p in PROBES})
                row.update({k: out[k] for k in ("T1_rec", "T2_rec", "dw_rec", "eps_rec")})
                raw_rows.append(row)

            pos_by_thresh = is_positive(noise_class, amplitude)
            for p in PROBES:
                pos_scores = [out[p] for out in out_list]
                auc_pt, auc_lo, auc_hi = bootstrap_auc(
                    null_scores[p], pos_scores,
                    rng_seed=abs(hash((noise_class, amplitude, p))) % (2 ** 31))
                n_pos = int(np.sum(np.isfinite(pos_scores)))
                n_neg = int(np.sum(np.isfinite(null_scores[p])))
                auc_rows.append({
                    "architecture": ARCH, "noise_class": noise_class,
                    "amplitude": amplitude, "probe": p,
                    "auc": auc_pt, "auc_lo": auc_lo, "auc_hi": auc_hi,
                    "n_pos": n_pos, "n_neg": n_neg,
                    "positive_by_threshold": pos_by_thresh,
                })
            print(f"  " + "  ".join(
                f"{p}:AUC={auc_rows[-4+j]['auc']:.3f}"
                for j, p in enumerate(PROBES)), flush=True)

        save_raw_csv(raw_rows)
        save_auc_csv(auc_rows)
        print(f"  [checkpoint] saved after noise_class={noise_class}", flush=True)

    return raw_rows, auc_rows


def save_raw_csv(rows):
    path = DATA_DIR / f"detect_raw_{JOB_TAG}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RAW_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {path}  ({len(rows)} rows)", flush=True)
    return path


def save_auc_csv(rows):
    path = DATA_DIR / f"detect_auc_{JOB_TAG}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=AUC_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {path}  ({len(rows)} rows)", flush=True)
    return path


# ---------------------------------------------------------------------------
# COMBINE mode: merge all per-job CSVs across architectures and noise
# classes, then produce the figure.
# ---------------------------------------------------------------------------
def _load_all_csvs(pattern, cols):
    rows = []
    for path in sorted(DATA_DIR.glob(pattern)):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows


PROBE_LABEL = {"t1": "T1", "ramsey": "Ramsey", "echo": "Echo", "gate": "Gate"}
PROBE_COLOR = {"t1": "#1f77b4", "ramsey": "#ff7f0e", "echo": "#2ca02c", "gate": "#9467bd"}
CLASS_LABEL = {
    "t1_telegraph": "TLS / T1\ntelegraph",
    "quasistatic_dephasing": "1/f quasi-static\ndephasing",
    "coherent_gate": "Coherent gate\nover-rotation",
    "spam_drift": "SPAM\ncalibration drift",
    "combined": "Combined\n(T1 + coherent)",
}
ARCH_LABEL = {
    "superconducting": "Superconducting",
    "trapped_ion": "Trapped-ion",
    "neutral_atom": "Neutral-atom",
}


def combine_and_plot():
    raw_rows = _load_all_csvs("detect_raw_detect_*.csv", RAW_COLS)
    auc_rows = _load_all_csvs("detect_auc_detect_*.csv", AUC_COLS)
    if not raw_rows or not auc_rows:
        sys.exit("COMBINE=1: no per-job CSVs found in data/ -- run the sweep first.")

    merged_raw_path = DATA_DIR / "detect_raw_ALL.csv"
    with open(merged_raw_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RAW_COLS)
        w.writeheader()
        w.writerows(raw_rows)
    merged_auc_path = DATA_DIR / "detect_auc_ALL.csv"
    with open(merged_auc_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=AUC_COLS)
        w.writeheader()
        w.writerows(auc_rows)
    print(f"Merged: {merged_raw_path} ({len(raw_rows)} rows), "
          f"{merged_auc_path} ({len(auc_rows)} rows)", flush=True)

    # Index raw rows for ROC-curve plotting: (arch, class, amplitude, probe) -> scores
    raw_idx = defaultdict(list)
    null_idx = defaultdict(list)
    for row in raw_rows:
        arch = row["architecture"]
        cls = row["noise_class"]
        amp = row["amplitude"]
        for p in PROBES:
            try:
                val = float(row[p])
            except (TypeError, ValueError):
                val = float("nan")
            if cls == "null":
                null_idx[(arch, p)].append(val)
            else:
                raw_idx[(arch, cls, amp, p)].append(val)

    # --- Figure 1: 3 (arch) x 5 (class) grid of ROC curves, 4 probes overlaid,
    #     using the LARGEST amplitude available per class (clearest signal) ---
    archs = ALL_ARCHITECTURES
    classes = ALL_NOISE_CLASSES
    fig, axes = plt.subplots(len(archs), len(classes),
                              figsize=(3.1 * len(classes), 3.1 * len(archs)))
    for ri, arch in enumerate(archs):
        for ci, cls in enumerate(classes):
            ax = axes[ri, ci]
            grid = AMPLITUDE_GRIDS[cls]
            amp_str = str(grid[-1])
            for p in PROBES:
                neg = null_idx.get((arch, p), [])
                pos = raw_idx.get((arch, cls, amp_str, p), [])
                if not pos:
                    # amplitude formatting may differ slightly (float repr);
                    # fall back to a tolerant match
                    for (a2, c2, amp2, p2), vals in raw_idx.items():
                        if a2 == arch and c2 == cls and p2 == p:
                            try:
                                if abs(float(amp2) - grid[-1]) < 1e-9:
                                    pos = vals
                                    break
                            except ValueError:
                                continue
                fpr, tpr = roc_curve_from_scores(neg, pos)
                auc_val = compute_auc(neg, pos)
                ax.plot(fpr, tpr, color=PROBE_COLOR[p], linewidth=1.6,
                        label=f"{PROBE_LABEL[p]} (AUC={auc_val:.2f})")
            ax.plot([0, 1], [0, 1], "k--", linewidth=0.7, alpha=0.4)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            if ri == 0:
                ax.set_title(CLASS_LABEL[cls], fontsize=9)
            if ci == 0:
                ax.set_ylabel(f"{ARCH_LABEL[arch]}\nTrue positive rate", fontsize=8)
            if ri == len(archs) - 1:
                ax.set_xlabel("False positive rate", fontsize=8)
            ax.tick_params(labelsize=6.5)
            ax.legend(fontsize=5.2, loc="lower right", framealpha=0.85)
            ax.grid(alpha=0.15)
    fig.suptitle(
        "Quantum Canary per-probe noise detection: ROC curves at maximum "
        "tested injection amplitude\n(shared null pool as negative class; "
        f"N={N_INSTANCES} instances/cell, budget={BUDGET} shots)",
        fontsize=11, y=1.01)
    plt.tight_layout()
    for fmt in ("pdf", "png"):
        p = FIG_DIR / f"fig_detection_roc_grid.{fmt}"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        print(f"Saved: {p}", flush=True)
    plt.close(fig)

    # --- Figure 2: AUC summary bar chart, per (probe, class), averaged
    #     across architectures, at max amplitude, with bootstrap CI ---
    summary = defaultdict(list)
    for row in auc_rows:
        try:
            amp = float(row["amplitude"])
        except ValueError:
            continue
        cls = row["noise_class"]
        grid = AMPLITUDE_GRIDS.get(cls)
        if grid is None or abs(amp - grid[-1]) > 1e-9:
            continue
        try:
            auc_val = float(row["auc"])
        except ValueError:
            continue
        if np.isfinite(auc_val):
            summary[(cls, row["probe"])].append(auc_val)

    fig2, ax2 = plt.subplots(figsize=(11, 5))
    x = np.arange(len(ALL_NOISE_CLASSES))
    width = 0.2
    for j, p in enumerate(PROBES):
        means = [np.mean(summary.get((cls, p), [np.nan])) for cls in ALL_NOISE_CLASSES]
        stds = [np.std(summary.get((cls, p), [np.nan])) for cls in ALL_NOISE_CLASSES]
        ax2.bar(x + (j - 1.5) * width, means, width, yerr=stds,
                label=PROBE_LABEL[p], color=PROBE_COLOR[p], alpha=0.85, capsize=3)
    ax2.axhline(0.90, color="gray", linewidth=1.0, linestyle="--", alpha=0.7)
    ax2.axhline(0.50, color="gray", linewidth=0.7, linestyle=":", alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([CLASS_LABEL[c].replace("\n", " ") for c in ALL_NOISE_CLASSES],
                         fontsize=8, rotation=15, ha="right")
    ax2.set_ylabel("AUC (mean across architectures ± s.d.)", fontsize=10)
    ax2.set_ylim(0.3, 1.05)
    ax2.legend(fontsize=9, ncol=4, loc="lower right")
    ax2.set_title(
        "Per-probe noise-class detection performance at maximum tested "
        "injection amplitude, averaged across all three architectures",
        fontsize=10)
    ax2.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    for fmt in ("pdf", "png"):
        p = FIG_DIR / f"fig_detection_auc_summary.{fmt}"
        fig2.savefig(p, dpi=160, bbox_inches="tight")
        print(f"Saved: {p}", flush=True)
    plt.close(fig2)

    print("\nCombine + plot complete.", flush=True)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    if sys.platform != "win32":
        mp.set_start_method("fork", force=True)

    if COMBINE:
        combine_and_plot()
        sys.exit(0)

    n_workers = int(os.environ.get("N_WORKERS", max(1, mp.cpu_count() - 1)))
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"[{ts}] 3_canary_detection.py -- noise-class detection AUC study", flush=True)
    print(f"  Architecture={ARCH}  NoiseClass={NOISE_CLASS}  Seed={SEED}  "
          f"N={N_INSTANCES}  bootstrap={N_BOOTSTRAP}  workers={n_workers}  "
          f"budget={BUDGET}", flush=True)

    raw_rows, auc_rows = run_sweep(n_workers)
    save_raw_csv(raw_rows)
    save_auc_csv(auc_rows)
    print(f"\n[job JOB_TAG={JOB_TAG}] done. Run with COMBINE=1 after all "
          f"matrix jobs complete to merge CSVs and produce figures.", flush=True)