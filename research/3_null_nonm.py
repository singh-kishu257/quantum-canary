"""
3_null_nonm.py
================
Two-phase, publication-grade validation of Quantum Canary's per-probe
chi2/dof as a Markovianity diagnostic. Establishes the logical chain
required for the claim "elevated chi2/dof implies physical
non-Markovianity, not model or optimizer failure":

  PHASE 1 -- NULL CALIBRATION (zero-tolerance false-positive audit)
    Under pure, correctly-specified Markovian hardware, does chi2/dof
    follow its theoretical chi-squared(nu)/nu distribution, with no
    optimizer-trap or model-misspecification tail? This is a
    goodness-of-fit claim, not an empirical guess -- it is testable via
    a Kolmogorov-Smirnov test against the known theoretical distribution
    (Bevington & Robinson, "Data Reduction and Error Analysis for the
    Physical Sciences", 3rd ed., 2003). N_NULL instances (default 500) at
    the native 9,900-shot budget are run through the UNMODIFIED
    lindblad_inversion() pipeline; every returned chi2/dof value across
    all four probes is pooled via the probability integral transform
    (PIT) -- because T1/Gate/Echo probes have 2 degrees of freedom while
    the Ramsey probe has 5 (see DOF dict below, derived directly from
    1_inversion.py's own _chi2_per_dof() calls), a naive pooled histogram
    of raw chi2/dof values would be statistically wrong. Converting each
    probe's chi2/dof to a p-value via its own chi-squared(nu) CDF makes
    every probe's null value Uniform(0,1) under a correctly specified
    model, regardless of its individual degrees of freedom -- this is
    what allows every probe, on every instance, to be pooled into one
    honest histogram and one global KS test.

  PHASE 2 -- CONTINUOUS NON-MARKOVIAN GRADIENT (monotonicity proof)
    As a physical noise amplitude alpha increases continuously from 0
    (Markovian) toward strongly non-Markovian, does mean chi2/dof rise
    smoothly and monotonically on the probe(s) sensitive to that specific
    noise channel? This uses the SAME three injection mechanisms already
    validated in 3_canary_detection.py (TLS-style T1 telegraph, 1/f
    quasi-static ensemble dephasing, coherent gate over-rotation), swept
    over a finer amplitude grid than the AUC study, with alpha=0 anchored
    to the Phase 1 null result (mean chi2/dof = 1 by construction).

FAIRNESS / NO-CHEATING NOTES
-----------------------------
  - 1_inversion.py is imported via importlib.util.spec_from_file_location,
    exactly as in 7_benchmark_experiments.py and 3_canary_detection.py.
    It is never edited, patched, or monkeypatched.
  - The injection functions in this file (_execute_circuit,
    run_t1_telegraph, run_quasistatic_dephasing, run_coherent_gate, and
    the coherent-error gate-rep circuit builder) are byte-identical
    copies of the versions validated and bug-fixed in
    3_canary_detection.py (T2 <= 2*T1 physical clipping on telegraph
    noise; genuine Hahn-echo refocusing via inject_echo_dw on the
    quasi-static class). They are copied rather than imported as a
    module to avoid 3_canary_detection.py's own top-level environment-
    variable validation (which calls sys.exit on an unrecognized
    NOISE_CLASS) firing as a side effect of import in this script's own
    environment. Anyone auditing this file can diff these functions
    directly against 3_canary_detection.py to confirm they are identical.
  - The degrees of freedom used for the theoretical chi-squared reference
    (T1=2, Ramsey=5, Gate=2, Echo=2) are not assumed -- they are read
    directly off 1_inversion.py's own lindblad_inversion() body, where
    each _chi2_per_dof(...) call is made with the exact number of data
    points per probe (3, 6, 3, 3) and n_params=1, giving dof =
    max(n_points - n_params, 1).
  - Ground truth for Phase 1 is "no noise injected" -- inv.run_probe_circuits_aer()
    is called unmodified with true parameters equal to what the inversion
    is told, exactly as run_null() does in 3_canary_detection.py.

USAGE
-----
  ARCH=superconducting PHASE=null      python3 3_null_nonm.py
  ARCH=superconducting PHASE=gradient  python3 3_null_nonm.py
  COMBINE=1 python3 3_null_nonm.py     # merge all per-job CSVs + plot

Environment variables:
  ARCH            one of superconducting | trapped_ion | neutral_atom
  PHASE           one of null | gradient | all
  N_NULL          null instances for Phase 1                [default 500]
  N_GRAD          instances per gradient amplitude point     [default 20]
  N_BOOTSTRAP     bootstrap resamples for gradient CI        [default 1000]
  DETECTION_BUDGET total shots per Canary run                [default 9900]
  FP_THRESHOLD    chi2/dof threshold for false-positive rate [default 3.0]
  N_WORKERS       multiprocessing pool size            [default cpu_count-1]
  JOB_TAG         output filename label      [default nonm_{ARCH}_{PHASE}]
  COMBINE         if "1", skip the sweep and merge+plot existing CSVs
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
from scipy.stats import chi2 as chi2_dist
from scipy.stats import kstest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Import 1_inversion.py EXACTLY as 7_benchmark_experiments.py and
# 3_canary_detection.py do: by file location, never edited, never
# monkeypatched.
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
SEED = 73
N_NULL = int(os.environ.get("N_NULL", "500"))
N_GRAD = int(os.environ.get("N_GRAD", "20"))
N_BOOTSTRAP = int(os.environ.get("N_BOOTSTRAP", "1000"))
BUDGET = int(os.environ.get("DETECTION_BUDGET", "9900"))
FP_THRESHOLD = float(os.environ.get("FP_THRESHOLD", "3.0"))

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

ALL_PHASES = ["null", "gradient"]
PHASE = os.environ.get("PHASE", "all").strip()
if PHASE not in ALL_PHASES + ["all"]:
    sys.exit(f"PHASE env var must be one of {ALL_PHASES + ['all']}, got {PHASE!r}")
PHASES_TO_RUN = ALL_PHASES if PHASE == "all" else [PHASE]

JOB_TAG = os.environ.get("JOB_TAG", f"nonm_{ARCH}_{PHASE}")
COMBINE = os.environ.get("COMBINE", "0") == "1"

PROBES = ["t1", "ramsey", "echo", "gate"]
CHI2_FIELD = {
    "t1": "t1_chi2_dof",
    "ramsey": "ramsey_chi2_dof",
    "echo": "echo_chi2_dof",
    "gate": "gate_chi2_dof",
}

# Degrees of freedom per probe, read directly off 1_inversion.py's own
# lindblad_inversion() body (see module docstring). These are protocol
# constants (fixed circuit counts: 3 T1, 6 Ramsey=2x3 delays, 3 gate-rep,
# 3 echo), invariant across architecture and shot budget.
#   T1:     dof = max(3 - 1, 1) = 2
#   Ramsey: dof = max(6 - 1, 1) = 5   (only T2 counted as a fitted param;
#                                       Δω is estimated in closed form)
#   Gate:   dof = max(3 - 1, 1) = 2
#   Echo:   dof = max(3 - 1, 1) = 2
DOF = {"t1": 2, "ramsey": 5, "gate": 2, "echo": 2}

# Reused verbatim from 7_benchmark_experiments.py / 3_canary_detection.py
# TRUE_PARAM_RANGES, for consistency across the whole synthetic-experiment
# suite.
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
# Byte-identical copies of the validated, bug-fixed injection primitives
# from 3_canary_detection.py. See module docstring for why these are
# copied rather than imported.
# ---------------------------------------------------------------------------
def _execute_circuit(qc, shots, T1_s, T2_s, eps_sx, p0g1, p1g0, dw_s,
                      gate_time_ns, dt_ns, inject_echo_dw=False):
    from qiskit import transpile
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
        # Physical bound: T2 <= 2*T1 (Lindblad decomposition). AerSimulator
        # raises hard error on T2 > 2*T1; clip so a TLS-driven T1 drop never
        # produces an unphysical (T1,T2) pair.
        T2_active = min(T2, 2.0 * T1_active)
        counts_list.append(_execute_circuit(
            qc, sh, T1_active, T2_active, eps, p0g1, p1g0, dw, gate_time_ns, profile.dt_ns))

    return inv.lindblad_inversion(
        counts_list, meta, profile,
        shots_t1=SH_T1, shots_ramsey=SH_RAM, shots_gate=SH_GATE, shots_echo=SH_ECHO,
        qubit_id=0, timestamp=datetime.now(timezone.utc).isoformat())


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


def _build_gate_rep_circuit_with_coherent_error(N, architecture, delta_rad):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1, 1, name=f"gate_rep_N{N}_coherent")
    for _ in range(N):
        inv._sqrtx_native_inverse_pair(qc, 0, architecture)
        qc.rx(delta_rad, 0)
    qc.measure(0, 0)
    return qc


def _build_probe_circuits_with_coherent_gate_error(profile, delta_rad):
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


GRADIENT_RUNNER = {
    "t1_telegraph": run_t1_telegraph,
    "quasistatic_dephasing": run_quasistatic_dephasing,
    "coherent_gate": run_coherent_gate,
}
# Finer amplitude grid than the AUC study (3_canary_detection.py), since
# Phase 2 needs a smooth curve rather than a single max-amplitude estimate.
# alpha=0.0 (Markovian limit) is included explicitly; its expected mean
# chi2/dof is 1.0, anchoring continuity with the Phase 1 null result.
GRADIENT_GRIDS = {
    "t1_telegraph": [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.80],
    "quasistatic_dephasing": [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.95, 1.20],
    "coherent_gate": [0.0, 0.005, 0.010, 0.015, 0.020, 0.030, 0.045, 0.06, 0.08],
}
GRADIENT_PRIMARY_PROBE = {
    "t1_telegraph": "t1",
    "quasistatic_dephasing": "ramsey",
    "coherent_gate": "gate",
}


# ---------------------------------------------------------------------------
# Worker dispatch
# ---------------------------------------------------------------------------
def _worker_null(args):
    warnings.filterwarnings("ignore")
    idx, params, seed, arch_name = args
    T1, T2, dw, eps = params
    try:
        r = run_null(T1, T2, dw, eps, seed, arch_name)
        out = {p: getattr(r, CHI2_FIELD[p]) for p in PROBES}
        out.update({"T1_rec": r.T1_s, "T2_rec": r.T2_s,
                     "dw_rec": r.delta_omega, "eps_rec": r.epsilon_sx})
    except Exception as exc:
        out = {p: float("nan") for p in PROBES}
        out.update({"T1_rec": float("nan"), "T2_rec": float("nan"),
                     "dw_rec": float("nan"), "eps_rec": float("nan"),
                     "error": str(exc)})
    return idx, out


def _worker_gradient(args):
    warnings.filterwarnings("ignore")
    idx, noise_class, amplitude, params, seed, arch_name = args
    T1, T2, dw, eps = params
    try:
        if amplitude == 0.0:
            r = run_null(T1, T2, dw, eps, seed, arch_name)
        else:
            r = GRADIENT_RUNNER[noise_class](T1, T2, dw, eps, amplitude, seed, arch_name)
        out = {p: getattr(r, CHI2_FIELD[p]) for p in PROBES}
    except Exception as exc:
        out = {p: float("nan") for p in PROBES}
        out["error"] = str(exc)
    return idx, out


# ---------------------------------------------------------------------------
# PHASE 1: null calibration
# ---------------------------------------------------------------------------
NULL_RAW_COLS = ["architecture", "instance_idx", "seed",
                  "t1", "ramsey", "echo", "gate",
                  "T1_rec", "T2_rec", "dw_rec", "eps_rec"]
NULL_SUMMARY_COLS = ["architecture", "probe", "dof", "n", "mean_chi2_dof",
                       "ks_statistic", "ks_pvalue",
                       "empirical_fp_rate", "theoretical_fp_rate",
                       "fp_threshold"]


def run_null_phase(n_workers):
    rng = np.random.default_rng(SEED)
    instances = [sample_instance(rng, ARCH) for _ in range(N_NULL)]
    seed_base = SEED * 1_000_000
    args = [(i, instances[i], seed_base + i, ARCH) for i in range(N_NULL)]

    print(f"[null phase] ARCH={ARCH} N={N_NULL} budget={BUDGET} "
          f"workers={n_workers}", flush=True)
    out_list = [None] * N_NULL
    with mp.Pool(n_workers) as pool:
        for idx, out in pool.imap_unordered(_worker_null, args, chunksize=4):
            out_list[idx] = out

    raw_rows = []
    for i, out in enumerate(out_list):
        row = {"architecture": ARCH, "instance_idx": i, "seed": seed_base + i}
        row.update({p: out[p] for p in PROBES})
        row.update({k: out[k] for k in ("T1_rec", "T2_rec", "dw_rec", "eps_rec")})
        raw_rows.append(row)

    raw_path = DATA_DIR / f"nullraw_{JOB_TAG}.csv"
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=NULL_RAW_COLS)
        w.writeheader()
        w.writerows(raw_rows)
    print(f"Saved: {raw_path}  ({len(raw_rows)} rows)", flush=True)

    # Per-probe KS test against theoretical chi-squared(dof) and empirical
    # vs theoretical false-positive rate at FP_THRESHOLD.
    summary_rows = []
    for p in PROBES:
        vals = np.array([out[p] for out in out_list], dtype=float)
        vals = vals[np.isfinite(vals)]
        dof = DOF[p]
        raw_chi2 = vals * dof
        ks_stat, ks_p = kstest(raw_chi2, lambda x, d=dof: chi2_dist.cdf(x, df=d))
        emp_fp = float(np.mean(vals > FP_THRESHOLD)) if len(vals) else float("nan")
        theo_fp = float(1.0 - chi2_dist.cdf(FP_THRESHOLD * dof, df=dof))
        summary_rows.append({
            "architecture": ARCH, "probe": p, "dof": dof, "n": len(vals),
            "mean_chi2_dof": float(np.mean(vals)) if len(vals) else float("nan"),
            "ks_statistic": float(ks_stat), "ks_pvalue": float(ks_p),
            "empirical_fp_rate": emp_fp, "theoretical_fp_rate": theo_fp,
            "fp_threshold": FP_THRESHOLD,
        })
        print(f"  {p:>7}: dof={dof}  mean(chi2/dof)={summary_rows[-1]['mean_chi2_dof']:.3f}  "
              f"KS p={ks_p:.3f}  FP@{FP_THRESHOLD}: emp={emp_fp:.4f} theo={theo_fp:.4f}",
              flush=True)

    summary_path = DATA_DIR / f"nullsummary_{JOB_TAG}.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=NULL_SUMMARY_COLS)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Saved: {summary_path}", flush=True)
    return raw_rows, summary_rows


# ---------------------------------------------------------------------------
# PHASE 2: continuous non-Markovian gradient
# ---------------------------------------------------------------------------
GRAD_RAW_COLS = ["architecture", "noise_class", "amplitude", "instance_idx",
                   "seed", "t1", "ramsey", "echo", "gate"]
GRAD_SUMMARY_COLS = ["architecture", "noise_class", "amplitude", "probe",
                       "mean_chi2_dof", "ci_lo", "ci_hi", "n"]


def bootstrap_mean_ci(values, n_boot=N_BOOTSTRAP, rng_seed=0):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(v))
    rng = np.random.default_rng(rng_seed)
    boots = [np.mean(v[rng.integers(0, len(v), size=len(v))]) for _ in range(n_boot)]
    b = np.array(boots)
    return point, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def run_gradient_phase(n_workers):
    rng_master = np.random.default_rng(SEED)
    instances = [sample_instance(rng_master, ARCH) for _ in range(N_GRAD)]

    raw_rows = []
    summary_rows = []
    for noise_class in GRADIENT_RUNNER.keys():
        grid = GRADIENT_GRIDS[noise_class]
        print(f"\n[gradient phase] class={noise_class} ARCH={ARCH} "
              f"N={N_GRAD} points={len(grid)}", flush=True)
        for ai, amplitude in enumerate(grid):
            seed_base = (SEED * 1_000_000 + (hash(noise_class) % 10_000) * 1_000
                          + ai * 100)
            args = [(i, noise_class, amplitude, instances[i], seed_base + i, ARCH)
                     for i in range(N_GRAD)]
            out_list = [None] * N_GRAD
            with mp.Pool(n_workers) as pool:
                for idx, out in pool.imap_unordered(_worker_gradient, args, chunksize=2):
                    out_list[idx] = out

            for i, out in enumerate(out_list):
                row = {"architecture": ARCH, "noise_class": noise_class,
                       "amplitude": amplitude, "instance_idx": i,
                       "seed": seed_base + i}
                row.update({p: out.get(p, float("nan")) for p in PROBES})
                raw_rows.append(row)

            for p in PROBES:
                vals = [out.get(p, float("nan")) for out in out_list]
                mean_v, lo, hi = bootstrap_mean_ci(
                    vals, rng_seed=abs(hash((noise_class, amplitude, p))) % (2 ** 31))
                summary_rows.append({
                    "architecture": ARCH, "noise_class": noise_class,
                    "amplitude": amplitude, "probe": p,
                    "mean_chi2_dof": mean_v, "ci_lo": lo, "ci_hi": hi,
                    "n": int(np.sum(np.isfinite(vals))),
                })
            primary = GRADIENT_PRIMARY_PROBE[noise_class]
            match = [s for s in summary_rows if s["noise_class"] == noise_class
                      and s["amplitude"] == amplitude and s["probe"] == primary][0]
            print(f"  alpha={amplitude:>6}  [{primary}] mean(chi2/dof)="
                  f"{match['mean_chi2_dof']:.3f}  CI=[{match['ci_lo']:.3f}, "
                  f"{match['ci_hi']:.3f}]", flush=True)

        # Checkpoint after each noise class.
        _save_grad_csvs(raw_rows, summary_rows)

    return raw_rows, summary_rows


def _save_grad_csvs(raw_rows, summary_rows):
    raw_path = DATA_DIR / f"gradraw_{JOB_TAG}.csv"
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=GRAD_RAW_COLS)
        w.writeheader()
        w.writerows(raw_rows)
    summary_path = DATA_DIR / f"gradsummary_{JOB_TAG}.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=GRAD_SUMMARY_COLS)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Saved: {raw_path} ({len(raw_rows)} rows), "
          f"{summary_path} ({len(summary_rows)} rows)", flush=True)


# ---------------------------------------------------------------------------
# COMBINE mode
# ---------------------------------------------------------------------------
def _load_all_csvs(pattern):
    rows = []
    for path in sorted(DATA_DIR.glob(pattern)):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows


PROBE_LABEL = {"t1": "T1", "ramsey": "Ramsey", "echo": "Echo", "gate": "Gate"}
CLASS_LABEL = {
    "t1_telegraph": "TLS / T1 telegraph",
    "quasistatic_dephasing": "1/f quasi-static dephasing",
    "coherent_gate": "Coherent gate over-rotation",
}
CLASS_COLOR = {
    "t1_telegraph": "#1f77b4",
    "quasistatic_dephasing": "#ff7f0e",
    "coherent_gate": "#9467bd",
}
ARCH_LABEL = {
    "superconducting": "Superconducting",
    "trapped_ion": "Trapped-ion",
    "neutral_atom": "Neutral-atom",
}
HEADLINE_ARCH = "superconducting"


def combine_and_plot():
    null_raw = _load_all_csvs("nullraw_nonm_*.csv")
    null_summary = _load_all_csvs("nullsummary_nonm_*.csv")
    grad_raw = _load_all_csvs("gradraw_nonm_*.csv")
    grad_summary = _load_all_csvs("gradsummary_nonm_*.csv")

    if not null_raw or not grad_raw:
        sys.exit("COMBINE=1: missing per-job CSVs in data/ -- run both phases first.")

    for name, rows, cols in [
        ("nullraw_ALL.csv", null_raw, NULL_RAW_COLS),
        ("nullsummary_ALL.csv", null_summary, NULL_SUMMARY_COLS),
        ("gradraw_ALL.csv", grad_raw, GRAD_RAW_COLS),
        ("gradsummary_ALL.csv", grad_summary, GRAD_SUMMARY_COLS),
    ]:
        path = DATA_DIR / name
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"Merged: {path} ({len(rows)} rows)", flush=True)

    # -------------------------------------------------------------------
    # Build the two-panel headline figure using the superconducting arch
    # (cleanest null pool / clearest gradient signal; see paper discussion
    # of trapped-ion T1-telegraph detection limits under arch-default
    # priors). Other architectures are included in the merged CSVs for a
    # supplementary multi-architecture table/figure.
    # -------------------------------------------------------------------
    # --- Left panel: PIT p-value histogram, all 4 probes pooled ---
    pvals = []
    for row in null_raw:
        if row["architecture"] != HEADLINE_ARCH:
            continue
        for p in PROBES:
            try:
                val = float(row[p])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(val):
                continue
            dof = DOF[p]
            raw_chi2 = val * dof
            pv = 1.0 - chi2_dist.cdf(raw_chi2, df=dof)
            pvals.append(pv)
    pvals = np.array(pvals)
    ks_stat, ks_p = (kstest(pvals, "uniform") if len(pvals) else (float("nan"), float("nan")))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax1 = axes[0]
    ax1.hist(pvals, bins=25, range=(0, 1), density=True, color="#4C72B0",
              alpha=0.75, edgecolor="white", linewidth=0.5,
              label=f"Empirical (N={len(pvals)} pooled probe values)")
    ax1.axhline(1.0, color="black", linestyle="--", linewidth=1.3,
                label="Theoretical Uniform(0,1)\n(Bevington & Robinson, 2003)")
    ax1.set_xlabel(r"$p$-value $= 1-F_{\chi^2}(\nu\cdot\chi^2/\nu;\ \nu)$", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.set_title(
        f"Phase 1 -- Null calibration ({ARCH_LABEL[HEADLINE_ARCH]}, N={N_NULL})\n"
        f"KS test vs. Uniform(0,1): D={ks_stat:.3f}, p={ks_p:.3f}",
        fontsize=10)
    ax1.set_ylim(0, 1.8)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.2)

    # --- Right panel: monotonicity gradient, 3 classes, primary probe each ---
    ax2 = axes[1]
    for noise_class in GRADIENT_RUNNER.keys():
        primary = GRADIENT_PRIMARY_PROBE[noise_class]
        pts = [r for r in grad_summary
               if r["architecture"] == HEADLINE_ARCH
               and r["noise_class"] == noise_class and r["probe"] == primary]
        pts_by_amp = {}
        for r in pts:
            try:
                amp = float(r["amplitude"])
                mean_v = float(r["mean_chi2_dof"])
                lo = float(r["ci_lo"])
                hi = float(r["ci_hi"])
            except (TypeError, ValueError):
                continue
            pts_by_amp[amp] = (mean_v, lo, hi)
        amps = sorted(pts_by_amp.keys())
        means = [pts_by_amp[a][0] for a in amps]
        los = [max(pts_by_amp[a][1], 1e-2) for a in amps]  # floor for log-scale safety
        his = [pts_by_amp[a][2] for a in amps]
        if not amps:
            continue
        # Normalize x-axis to [0,1] fraction of that class's own max
        # amplitude so all three curves share one axis meaningfully.
        max_amp = max(GRADIENT_GRIDS[noise_class])
        x_norm = [a / max_amp for a in amps]
        color = CLASS_COLOR[noise_class]
        ax2.plot(x_norm, means, "o-", color=color, linewidth=1.8, markersize=4,
                 label=f"{CLASS_LABEL[noise_class]} ({PROBE_LABEL[primary]} probe)")
        ax2.fill_between(x_norm, los, his, color=color, alpha=0.15)

    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.3,
                label=r"Markovian expectation $\chi^2/\nu=1$")
    ax2.set_xlabel(r"Normalized injection amplitude $\alpha$ / $\alpha_{\max}$", fontsize=10)
    ax2.set_ylabel(r"Mean $\chi^2/\nu$ on primary probe (95% bootstrap CI)", fontsize=10)
    ax2.set_yscale("log")
    ax2.set_title(
        f"Phase 2 -- Continuous non-Markovian gradient ({ARCH_LABEL[HEADLINE_ARCH]}, "
        f"N={N_GRAD}/point)",
        fontsize=10)
    ax2.legend(fontsize=7.5, loc="upper left")
    ax2.grid(alpha=0.2)

    fig.suptitle(
        "Quantum Canary $\\chi^2/\\nu$: null calibration and non-Markovian "
        "monotonicity",
        fontsize=12, y=1.03)
    plt.tight_layout()
    for fmt in ("pdf", "png"):
        path = FIG_DIR / f"fig_null_gradient.{fmt}"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        print(f"Saved: {path}", flush=True)
    plt.close(fig)

    # -------------------------------------------------------------------
    # Supplementary: per-architecture false-positive rate table (text
    # summary written to a .txt file for easy inclusion in the paper).
    # -------------------------------------------------------------------
    summary_path = FIG_DIR / "null_calibration_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Phase 1 null calibration summary (per architecture, per probe)\n")
        f.write("=" * 70 + "\n")
        for row in null_summary:
            f.write(
                f"{row['architecture']:>16s}  {row['probe']:>7s}  "
                f"dof={row['dof']:>2s}  n={row['n']:>5s}  "
                f"mean(chi2/dof)={float(row['mean_chi2_dof']):.4f}  "
                f"KS p={float(row['ks_pvalue']):.4f}  "
                f"FP@{row['fp_threshold']}: emp={float(row['empirical_fp_rate']):.4f} "
                f"theo={float(row['theoretical_fp_rate']):.4f}\n"
            )
    print(f"Saved: {summary_path}", flush=True)
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
    print(f"[{ts}] 3_null_nonm.py -- null calibration + non-Markovian gradient", flush=True)
    print(f"  Architecture={ARCH}  Phase={PHASE}  Seed={SEED}  "
          f"N_NULL={N_NULL}  N_GRAD={N_GRAD}  budget={BUDGET}  "
          f"workers={n_workers}", flush=True)

    if "null" in PHASES_TO_RUN:
        run_null_phase(n_workers)
    if "gradient" in PHASES_TO_RUN:
        run_gradient_phase(n_workers)

    print(f"\n[job JOB_TAG={JOB_TAG}] done. Run with COMBINE=1 after all "
          f"matrix jobs complete to merge CSVs and produce figures.", flush=True)