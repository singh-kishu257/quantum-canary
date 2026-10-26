"""Ideal-regime benchmark.

Each instance is a simulated qubit whose noise matches Canary's model, with a
calibration prior a few percent off the truth. Canary builds its probe from
the prior, runs it on the simulator, and we compare what it recovers (T1, T2,
delta_omega, epsilon) with the true values.

The prior alone already scores a high R2 here, so read R2 next to R2_prior,
beats_prior and the |z| coverage. SPAM is given to Canary, not measured.
The noisy test is 3_benchmark.py. Output is one CSV.
"""

from __future__ import annotations

import argparse
import csv
import logging
import multiprocessing as mp
import pathlib
import sys
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_SRC = SCRIPT_DIR.parent / "src"
if REPO_SRC.is_dir():
    sys.path.insert(0, str(REPO_SRC))

from quantum_canary import inversion as inv
from quantum_canary import simulation as sim
from quantum_canary import api, runner

SHOTS_T1, SHOTS_RAMSEY, SHOTS_GATE, SHOTS_ECHO = 300, 600, 24, 500
ARCHITECTURES = ("superconducting", "trapped_ion")
ARCH_SEED_OFFSET = {"superconducting": 0, "trapped_ion": 100_000}
PARAMETERS = ("T1_s", "T2_s", "delta_omega", "epsilon_sx")
CIRCUIT_PREFIXES = ("t1_", "ramsey_X_", "ramsey_Y_", "rb_", "echo_")

CSV_COLUMNS = [
    "architecture", "instance_id", "seed", "parameter",
    "true_value", "canary_value", "absolute_error", "relative_error",
    "success", "fit_status", "sigma_canary", "chi2_dof", "chi2_dof_secondary",
    "health", "health_reason",
    "shots_total", "prior_value", "prior_absolute_error", "prior_relative_error",
    "timestamp_utc",
    "perturbation_frac", "spam_scale", "spam_swap", "spam_spread", "cold_start",
    "dw_population", "dw_aliased",
    "signed_error", "z_score",
]
NAN = float("nan")

DEFAULT_CFG: Dict[str, Any] = {
    "perturbation_frac": None,
    "spam_scale": 1.0,
    "spam_swap": False,
    "spam_spread": 0.0,
    "cold_start": False,
    "dw_population": "estimator_window",
}
SPAM_CLIP_MAX = 0.499


def _abs_error(value: float, truth_value: float) -> float:
    if not np.isfinite(value) or not np.isfinite(truth_value):
        return NAN
    return abs(value - truth_value)


def _rel_error(value: float, truth_value: float) -> float:
    if not np.isfinite(value) or not np.isfinite(truth_value) or truth_value == 0:
        return NAN
    return abs(value - truth_value) / abs(truth_value)


def _signed_error(value: float, truth_value: float) -> float:
    if not np.isfinite(value) or not np.isfinite(truth_value):
        return NAN
    return value - truth_value


def _z_score(value: float, truth_value: float, sigma: float) -> float:
    """(estimate - truth) / sigma, or NaN if that can't be computed."""
    if not (np.isfinite(value) and np.isfinite(truth_value)
            and np.isfinite(sigma) and sigma > 0):
        return NAN
    return (value - truth_value) / sigma


def _r2(truth: np.ndarray, estimate: np.ndarray) -> float:
    """Coefficient of determination of estimate against truth."""
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    if ss_tot <= 0.0:
        return NAN
    return 1.0 - float(np.sum((estimate - truth) ** 2)) / ss_tot


def _row(architecture: str, instance_id: int, seed: int, parameter: str,
         timestamp: str, **kw) -> Dict[str, Any]:
    row = dict.fromkeys(CSV_COLUMNS, NAN)
    row.update(architecture=architecture, instance_id=instance_id, seed=seed,
               parameter=parameter, timestamp_utc=timestamp,
               success=False, fit_status="", shots_total=0)
    row.update(kw)
    return row


def _resolve_cfg(cfg: Any = None) -> Dict[str, Any]:
    out = dict(DEFAULT_CFG)
    if cfg:
        unknown = set(cfg) - set(DEFAULT_CFG)
        if unknown:
            raise ValueError(f"unknown config keys {sorted(unknown)}")
        out.update(cfg)
    return out


def _config_columns(architecture: str, c: Dict[str, Any]) -> Dict[str, Any]:
    """Config values written to every row."""
    pf = c["perturbation_frac"]
    if pf is None:
        pf = sim.resolve_calibration_perturbation(architecture, "ideal")
    return {"perturbation_frac": float(pf), "spam_scale": float(c["spam_scale"]),
            "spam_swap": bool(c["spam_swap"]), "spam_spread": float(c["spam_spread"]),
            "cold_start": bool(c["cold_start"]), "dw_population": str(c["dw_population"])}


def _inversion_spam(prior: Any, c: Dict[str, Any]):
    """Readout errors given to Canary: the prior's, scaled, optionally swapped, clipped."""
    p01 = prior.spam_p0_given_1_prior * c["spam_scale"]
    p10 = prior.spam_p1_given_0_prior * c["spam_scale"]
    if c["spam_swap"]:
        p01, p10 = p10, p01
    return (float(np.clip(p01, 0.0, SPAM_CLIP_MAX)),
            float(np.clip(p10, 0.0, SPAM_CLIP_MAX)))


def run_one_instance(architecture: str, instance_id: int, seed: int,
                     cfg: Any = None) -> List[Dict[str, Any]]:
    timestamp = datetime.now(timezone.utc).isoformat()
    c = _resolve_cfg(cfg)
    cfg_cols = _config_columns(architecture, c)
    try:
        truth = sim.generate_device(architecture, n_qubits=1, seed=seed, regime="ideal",
                                    spam_spread_frac=c["spam_spread"],
                                    perturbation_frac=c["perturbation_frac"],
                                    dw_population=c["dw_population"])
        prior = sim.generate_calibration_prior(truth, qubit=0, t0_s=0.0,
                                               perturbation_frac=c["perturbation_frac"])
        p0_given_1, p1_given_0 = _inversion_spam(prior, c)

        if c["cold_start"]:
            profile = inv.BackendProfile.from_architecture(
                architecture, backend_name=f"ideal_parity_sim::{architecture}",
                p0_given_1=p0_given_1, p1_given_0=p1_given_0)
        else:
            constants = dict(inv.ARCH_DEFAULTS[architecture])
            constants.update(T1_s=prior.T1_prior_s, T2_s=prior.T2_prior_s,
                             eps_typical=prior.epsilon_prior,
                             p0_given_1=p0_given_1,
                             p1_given_0=p1_given_0)
            profile = inv.BackendProfile(
                architecture=architecture,
                T1_prior_s=prior.T1_prior_s,
                T2_prior_s=prior.T2_prior_s,
                dt_ns=inv.ARCH_DEFAULTS[architecture]["dt_ns"],
                backend_name=f"ideal_parity_sim::{architecture}",
                custom_arch=constants,
                prior_confidence="prior_informed")

        circuits, meta = inv.build_probe_circuits(profile, seed=int(seed))
        for qc in circuits:
            if not any(qc.name.startswith(p) for p in CIRCUIT_PREFIXES):
                raise RuntimeError(f"unexpected circuit name {qc.name!r}")

        shots_list = ([SHOTS_T1] * meta["n_t1"] + [SHOTS_RAMSEY] * meta["n_ramsey"]
                      + [SHOTS_GATE] * meta["n_gate"] + [SHOTS_ECHO] * meta["n_echo"])
        if len(shots_list) != len(circuits):
            raise RuntimeError(f"{len(shots_list)} shot entries != {len(circuits)} circuits")

        sim_result = sim.simulate_probe(circuits, truth, shots=shots_list,
                                        t_s=0.0, dt_ns=profile.dt_ns,
                                        seed=int(seed) * 1000)
        result = inv.lindblad_inversion(
            sim_result.counts_list, meta, profile,
            shots_t1=SHOTS_T1, shots_ramsey=SHOTS_RAMSEY, shots_gate=SHOTS_GATE,
            shots_echo=SHOTS_ECHO, qubit_id=0, timestamp=timestamp)

        tv_now = sim.evaluate_truth_at(truth, 0, 0.0)
        true_values = {"T1_s": float(tv_now["T1_s"]),
                       "T2_s": float(tv_now["T2_s"]),
                       "delta_omega": float(tv_now["delta_omega_rad_s"]),
                       "epsilon_sx": runner.rb_gate_error(truth, 0.0)}
        canary = {"T1_s": float(result.T1_s),
                  "T2_s": float(result.T2_s),
                  "delta_omega": float(result.delta_omega),
                  "epsilon_sx": float(result.epsilon_sx)}
        sigmas = {"T1_s": float(result.T1_sigma_s),
                  "T2_s": float(result.T2_sigma_s),
                  "delta_omega": float(result.delta_omega_sigma),
                  "epsilon_sx": float(result.epsilon_sx_sigma)}
        chi2 = {"T1_s": float(result.t1_chi2_dof),
                "T2_s": float(result.echo_chi2_dof),
                "delta_omega": float(result.ramsey_chi2_dof),
                "epsilon_sx": float(result.gate_chi2_dof)}
        chi2_secondary = {"T2_s": float(result.ramsey_chi2_dof)}
        priors = {"T1_s": float(prior.T1_prior_s),
                  "T2_s": float(prior.T2_prior_s),
                  "delta_omega": NAN,
                  "epsilon_sx": float(prior.epsilon_prior)}
        dw_aliased = {"delta_omega": bool(abs(true_values["delta_omega"])
                                          * meta["ramsey_delays_s"][0] > np.pi)}

        health = runner._health(0, result)
        health_reason = ", ".join(api.health_reasons(health, result))
        rows = []
        for param in PARAMETERS:
            tv, cv, sg, pv = true_values[param], canary[param], sigmas[param], priors[param]
            converged = bool(np.isfinite(cv) and np.isfinite(sg))
            rows.append(_row(
                architecture, instance_id, seed, param, timestamp,
                true_value=tv, canary_value=cv,
                absolute_error=_abs_error(cv, tv), relative_error=_rel_error(cv, tv),
                success=converged,
                fit_status="ok" if converged else "non_convergent_infinite_sigma",
                sigma_canary=sg, chi2_dof=chi2[param],
                chi2_dof_secondary=chi2_secondary.get(param, NAN),
                health=health, health_reason=health_reason,
                shots_total=sum(shots_list), prior_value=pv,
                prior_absolute_error=_abs_error(pv, tv),
                prior_relative_error=_rel_error(pv, tv),
                dw_aliased=dw_aliased.get(param, NAN), **cfg_cols,
                signed_error=_signed_error(cv, tv), z_score=_z_score(cv, tv, sg)))
        return rows

    except Exception as exc:
        status = f"exception:{type(exc).__name__}:{exc}"
        return [_row(architecture, instance_id, seed, p, timestamp, fit_status=status,
                     **cfg_cols)
                for p in PARAMETERS]


def _worker(job) -> List[Dict[str, Any]]:
    warnings.filterwarnings("ignore")
    logging.getLogger("qiskit_aer.noise.noise_model").setLevel(logging.ERROR)
    return run_one_instance(*job)


def run_architecture(architecture: str, n_instances: int, seed_offset: int,
                     n_workers: int, cfg: Any = None) -> List[Dict[str, Any]]:
    jobs = [(architecture, i, seed_offset + i, cfg) for i in range(n_instances)]
    print(f"[ideal_parity] {architecture}: {n_instances} instances, seeds "
          f"{seed_offset}..{seed_offset + n_instances - 1}, {n_workers} workers", flush=True)

    results: List[Any] = [None] * n_instances
    if n_workers <= 1:
        for idx, job in enumerate(jobs):
            results[idx] = _worker(job)
    else:
        with mp.Pool(n_workers) as pool:
            for idx, rows in enumerate(pool.imap(_worker, jobs, chunksize=1)):
                results[idx] = rows
    if any(r is None for r in results):
        raise RuntimeError("a worker returned no rows; refusing to emit a partial CSV")
    return [row for rows in results for row in rows]


def validate_rows(rows: List[Dict[str, Any]], n_instances: int,
                  architectures: List[str]) -> List[str]:
    problems: List[str] = []
    by_arch: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_arch.setdefault(row["architecture"], []).append(row)

    if sorted(by_arch) != sorted(architectures):
        problems.append(f"architectures {sorted(by_arch)} != {sorted(architectures)}")

    for architecture, arch_rows in sorted(by_arch.items()):
        ids = sorted({r["instance_id"] for r in arch_rows})
        if ids != list(range(n_instances)):
            problems.append(f"{architecture}: instance_ids are not contiguous "
                            f"0..{n_instances - 1} (a gap means instances were dropped)")
        if len(arch_rows) != n_instances * len(PARAMETERS):
            problems.append(f"{architecture}: {len(arch_rows)} rows, expected "
                            f"{n_instances * len(PARAMETERS)}")
        per_instance: Dict[int, List[str]] = {}
        for r in arch_rows:
            per_instance.setdefault(r["instance_id"], []).append(r["parameter"])
        if any(sorted(p) != sorted(PARAMETERS) for p in per_instance.values()):
            problems.append(f"{architecture}: an instance does not carry exactly {PARAMETERS}")
        if len({r["seed"] for r in arch_rows}) != n_instances:
            problems.append(f"{architecture}: seeds are not unique per instance")
        budgets = {r["shots_total"] for r in arch_rows if r["shots_total"]}
        if len(budgets) > 1:
            problems.append(f"{architecture}: unequal shot budgets {sorted(budgets)}")
        if any(not str(r["fit_status"]).strip() for r in arch_rows):
            problems.append(f"{architecture}: a row has a blank fit_status")

    keys = [(r["architecture"], r["instance_id"], r["parameter"]) for r in rows]
    if len(set(keys)) != len(keys):
        problems.append("duplicate (architecture, instance_id, parameter) rows")
    if len({(r["architecture"], r["seed"]) for r in rows}) != len(architectures) * n_instances:
        problems.append("architecture/seed pairs are not unique")
    return problems


def write_csv(rows: List[Dict[str, Any]], path: pathlib.Path) -> None:
    rank = {p: i for i, p in enumerate(PARAMETERS)}
    ordered = sorted(rows, key=lambda r: (r["architecture"], r["instance_id"],
                                          rank.get(r["parameter"], 99)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(ordered)
    print(f"[ideal_parity] wrote {len(ordered)} rows -> {path}", flush=True)


def summarize(rows: List[Dict[str, Any]]) -> None:
    """Print errors, R2 against the prior-only R2, and |z| coverage per parameter."""
    print("\n[ideal_parity] summary, all instances, nothing removed "
          "(R2 and errors over converged instances; failures counted)")
    print("  R2(prior) is the score of the calibration prior with NO measurement: "
          "compare against it, never read R2 alone.")
    by_arch: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_arch.setdefault(r["architecture"], []).append(r)

    def _fmt(x: float, pct: bool = False) -> str:
        if not np.isfinite(x):
            return "--"
        return f"{100 * x:.1f}%" if pct else f"{x:.3f}"

    for architecture, arch_rows in sorted(by_arch.items()):
        n_inst = len({r["instance_id"] for r in arch_rows})
        n_bad = len({r["instance_id"] for r in arch_rows if not r["success"]})
        print(f"  {architecture} ({n_inst} instances, {n_bad} with a failed parameter)")
        print(f"     {'parameter':12s} {'fail':>5s} {'medRE':>8s} {'medRE_prior':>11s} "
              f"{'beats_prior':>11s} {'R2':>7s} {'R2_prior':>8s} {'|z|<=1':>7s} {'|z|<=2':>7s}")
        for param in PARAMETERS:
            pr = [r for r in arch_rows if r["parameter"] == param]
            tv = np.array([r["true_value"] for r in pr], dtype=float)
            cv = np.array([r["canary_value"] for r in pr], dtype=float)
            pv = np.array([r["prior_value"] for r in pr], dtype=float)
            re = np.array([r["relative_error"] for r in pr], dtype=float)
            pre = np.array([r["prior_relative_error"] for r in pr], dtype=float)
            z = np.array([r["z_score"] for r in pr], dtype=float)
            ok = np.array([bool(r["success"]) for r in pr]) & np.isfinite(cv) & np.isfinite(tv)
            both = ok & np.isfinite(pv)
            fin_re = ok & np.isfinite(re)
            fin_pair = both & np.isfinite(re) & np.isfinite(pre)
            fin_z = np.isfinite(z)
            print(f"     {param:12s} {len(pr) - int(ok.sum()):5d} "
                  f"{_fmt(float(np.median(re[fin_re])) if fin_re.any() else NAN, True):>8s} "
                  f"{_fmt(float(np.median(pre[fin_pair])) if fin_pair.any() else NAN, True):>11s} "
                  f"{_fmt(float(np.mean(re[fin_pair] < pre[fin_pair])) if fin_pair.any() else NAN, True):>11s} "
                  f"{_fmt(_r2(tv[ok], cv[ok]) if ok.sum() > 1 else NAN):>7s} "
                  f"{_fmt(_r2(tv[both], pv[both]) if both.sum() > 1 else NAN):>8s} "
                  f"{_fmt(float(np.mean(np.abs(z[fin_z]) <= 1)) if fin_z.any() else NAN, True):>7s} "
                  f"{_fmt(float(np.mean(np.abs(z[fin_z]) <= 2)) if fin_z.any() else NAN, True):>7s}")
        n_aliased = sum(1 for r in arch_rows
                        if r["parameter"] == "delta_omega" and r["dw_aliased"] is True)
        print(f"     dw_aliased: {n_aliased}/{n_inst} instances")


def _read_csv(path: pathlib.Path) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError(f"{path.name}: columns {reader.fieldnames} != {CSV_COLUMNS}")
        rows = []
        for raw in reader:
            row = dict(raw)
            for key in ("instance_id", "seed", "shots_total"):
                row[key] = int(row[key])
            row["success"] = str(row["success"]).strip().lower() == "true"
            for key in ("true_value", "canary_value", "absolute_error", "relative_error",
                        "sigma_canary", "chi2_dof", "chi2_dof_secondary",
                        "prior_value", "prior_absolute_error", "prior_relative_error",
                        "perturbation_frac", "spam_scale", "spam_spread",
                        "signed_error", "z_score"):
                row[key] = float(row[key]) if row[key] not in ("", None) else NAN
            for key in ("spam_swap", "cold_start"):
                row[key] = str(row[key]).strip().lower() == "true"
            flag = str(row["dw_aliased"]).strip().lower()
            row["dw_aliased"] = {"true": True, "false": False}.get(flag, NAN)
            rows.append(row)
    return rows


def combine(inputs: List[pathlib.Path], out: pathlib.Path, n_instances: int) -> None:
    if not inputs:
        sys.exit("--combine requires at least one --input")
    rows: List[Dict[str, Any]] = []
    for path in inputs:
        if not path.exists():
            sys.exit(f"missing input CSV: {path}")
        rows.extend(_read_csv(path))

    architectures = sorted({r["architecture"] for r in rows})
    problems = validate_rows(rows, n_instances, architectures)
    expected = len(architectures) * n_instances * len(PARAMETERS)
    if len(rows) != expected:
        problems.append(f"aggregate row count {len(rows)} != {expected}")

    write_csv(rows, out)
    summarize(rows)
    _finish(problems, len(rows))


def _finish(problems: List[str], n_rows: int) -> None:
    if problems:
        print("\n[ideal_parity] VALIDATION FAILED:", flush=True)
        for p in problems:
            print(f"  - {p}", flush=True)
        sys.exit(1)
    print(f"\n[ideal_parity] validation passed: {n_rows} rows")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ideal-parity parameter-recovery benchmark.")
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--input", type=pathlib.Path, action="append", default=[])
    ap.add_argument("--architecture", choices=list(ARCHITECTURES) + ["all"])
    ap.add_argument("--n-instances", type=int, default=300)
    ap.add_argument("--seed-offset", type=int, default=None)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--perturbation-frac", type=float, default=None,
                    help="calibration-prior envelope half-width f (default: 1c table); "
                         "also sets the matching generate_device dw ceiling")
    ap.add_argument("--spam-scale", type=float, default=1.0,
                    help="scale the SPAM handed to the inversion (truth untouched)")
    ap.add_argument("--spam-swap", action="store_true",
                    help="swap p0|1 and p1|0 handed to the inversion (truth untouched)")
    ap.add_argument("--spam-spread", type=float, default=0.0,
                    help="generate_device(spam_spread_frac=...)")
    ap.add_argument("--cold-start", action="store_true",
                    help="use the architecture-default profile instead of the prior")
    ap.add_argument("--dw-population", choices=["estimator_window", "independent"],
                    default="estimator_window",
                    help="generate_device(dw_population=...)")
    args = ap.parse_args()

    if args.n_instances <= 0:
        sys.exit("--n-instances must be positive")
    if args.perturbation_frac is not None and not 0.0 <= args.perturbation_frac < 1.0:
        sys.exit("--perturbation-frac must be in [0, 1)")
    if args.spam_scale < 0.0:
        sys.exit("--spam-scale must be non-negative")
    if not 0.0 <= args.spam_spread < 1.0:
        sys.exit("--spam-spread must be in [0, 1)")
    cfg = {"perturbation_frac": args.perturbation_frac, "spam_scale": args.spam_scale,
           "spam_swap": args.spam_swap, "spam_spread": args.spam_spread,
           "cold_start": args.cold_start, "dw_population": args.dw_population}
    if args.combine:
        combine(args.input, args.out, args.n_instances)
        return
    if args.architecture is None:
        sys.exit("--architecture is required (or use --combine)")

    architectures = list(ARCHITECTURES) if args.architecture == "all" else [args.architecture]
    if args.seed_offset is not None and len(architectures) > 1:
        sys.exit("--seed-offset cannot be combined with --architecture all")

    n_workers = args.workers if args.workers is not None else max(1, mp.cpu_count() - 1)
    started = datetime.now(timezone.utc)

    rows: List[Dict[str, Any]] = []
    for architecture in architectures:
        offset = (args.seed_offset if args.seed_offset is not None
                  else ARCH_SEED_OFFSET[architecture])
        rows.extend(run_architecture(architecture, args.n_instances, offset, n_workers, cfg))

    problems = validate_rows(rows, args.n_instances, architectures)
    write_csv(rows, args.out)
    summarize(rows)
    print(f"[ideal_parity] elapsed "
          f"{(datetime.now(timezone.utc) - started).total_seconds():.1f}s")
    _finish(problems, len(rows))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.getLogger("qiskit_aer.noise.noise_model").setLevel(logging.ERROR)
    if sys.platform != "win32":
        try:
            mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    main()