"""3_benchmark.py -- parameter-recovery accuracy on NISQ simulated hardware.

Per instance: 1c.generate_device(regime="nisq") -> hidden DeviceTruth with
drift and coherent error mechanisms; 1c.generate_calibration_prior() -> what
Canary may know; 1_inversion builds its own probe circuits from the prior;
1c.simulate_probe() executes them on Aer at T_MEASURE_S;
1_inversion.lindblad_inversion() recovers T1, T2, delta_omega, epsilon_sx.
Ground truth is read only by the simulator and by the final comparison.

true_value is 1c.evaluate_truth_at() at T_MEASURE_S, the exact parameters the
simulator used. epsilon truth is 1c.effective_gate_epsilon(): the per-gate
error of the simulated channel, including gate-time decoherence and any
coherent residual surviving the native G*G^-1 pair. Probes are single-qubit,
so crosstalk never fires. Drift is zero at T_MEASURE_S = T_CALIB_S = 0; the
calibration-prior envelope carries the staleness.

SPAM is not benchmarked: the inversion consumes it as a known constant and
never estimates it. delta_omega has no prior: CalibrationPrior deliberately
omits it, so prior_value is NaN for that parameter.

Output: one CSV, no figures.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import logging
import multiprocessing as mp
import pathlib
import sys
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


inv = _load("inversion", "1_inversion.py")
sim = _load("sim1c", "1c_simulation_models.py")

REGIME = "nisq"
T_CALIB_S = 0.0
T_MEASURE_S = 0.0
SHOTS_T1, SHOTS_RAMSEY, SHOTS_GATE, SHOTS_ECHO = 300, 1000, 500, 500
ARCHITECTURES = ("superconducting", "trapped_ion")
ARCH_SEED_OFFSET = {"superconducting": 0, "trapped_ion": 100_000}
PARAMETERS = ("T1_s", "T2_s", "delta_omega", "epsilon_sx")
CIRCUIT_PREFIXES = ("t1_", "ramsey_X_", "ramsey_Y_", "gate_rep_N", "echo_")

CSV_COLUMNS = [
    "architecture", "instance_id", "seed", "parameter",
    "true_value", "canary_value", "absolute_error", "relative_error",
    "success", "fit_status", "sigma_canary", "chi2_dof", "chi2_dof_secondary",
    "shots_total", "prior_value", "prior_absolute_error", "prior_relative_error",
    "timestamp_utc",
]
NAN = float("nan")


def _abs_error(value: float, truth_value: float) -> float:
    if not np.isfinite(value) or not np.isfinite(truth_value):
        return NAN
    return abs(value - truth_value)


def _rel_error(value: float, truth_value: float) -> float:
    if not np.isfinite(value) or not np.isfinite(truth_value) or truth_value == 0:
        return NAN
    return abs(value - truth_value) / abs(truth_value)


def _row(architecture: str, instance_id: int, seed: int, parameter: str,
         timestamp: str, **kw) -> Dict[str, Any]:
    row = dict.fromkeys(CSV_COLUMNS, NAN)
    row.update(architecture=architecture, instance_id=instance_id, seed=seed,
               parameter=parameter, timestamp_utc=timestamp,
               success=False, fit_status="", shots_total=0)
    row.update(kw)
    return row


def run_one_instance(architecture: str, instance_id: int, seed: int) -> List[Dict[str, Any]]:
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        truth = sim.generate_device(architecture, n_qubits=1, seed=seed, regime=REGIME)
        prior = sim.generate_calibration_prior(truth, qubit=0, t0_s=T_CALIB_S)

        constants = dict(inv.ARCH_DEFAULTS[architecture])
        constants.update(T1_s=prior.T1_prior_s, T2_s=prior.T2_prior_s,
                         eps_typical=prior.epsilon_prior,
                         p0_given_1=prior.spam_p0_given_1_prior,
                         p1_given_0=prior.spam_p1_given_0_prior)
        profile = inv.BackendProfile(
            architecture=architecture,
            T1_prior_s=prior.T1_prior_s,
            T2_prior_s=prior.T2_prior_s,
            dt_ns=inv.ARCH_DEFAULTS[architecture]["dt_ns"],
            backend_name=f"nisq_benchmark_sim::{architecture}",
            custom_arch=constants,
            prior_confidence="live")

        circuits, meta = inv.build_probe_circuits(profile)
        for qc in circuits:
            if not any(qc.name.startswith(p) for p in CIRCUIT_PREFIXES):
                raise RuntimeError(f"unexpected circuit name {qc.name!r}")

        shots_list = ([SHOTS_T1] * meta["n_t1"] + [SHOTS_RAMSEY] * meta["n_ramsey"]
                      + [SHOTS_GATE] * meta["n_gate"] + [SHOTS_ECHO] * meta["n_echo"])
        if len(shots_list) != len(circuits):
            raise RuntimeError(f"{len(shots_list)} shot entries != {len(circuits)} circuits")

        sim_result = sim.simulate_probe(circuits, truth, shots=shots_list,
                                        t_s=T_MEASURE_S, dt_ns=profile.dt_ns)
        result = inv.lindblad_inversion(
            sim_result.counts_list, meta, profile,
            shots_t1=SHOTS_T1, shots_ramsey=SHOTS_RAMSEY, shots_gate=SHOTS_GATE,
            shots_echo=SHOTS_ECHO, qubit_id=0, timestamp=timestamp)

        tv_now = sim.evaluate_truth_at(truth, 0, T_MEASURE_S)
        true_values = {"T1_s": float(tv_now["T1_s"]),
                       "T2_s": float(tv_now["T2_s"]),
                       "delta_omega": abs(float(tv_now["delta_omega_rad_s"])),
                       "epsilon_sx": sim.effective_gate_epsilon(truth, 0, T_MEASURE_S)}
        canary = {"T1_s": float(result.T1_s),
                  "T2_s": float(result.T2_s),
                  "delta_omega": abs(float(result.delta_omega)),
                  "epsilon_sx": float(result.epsilon_sx)}
        sigmas = {"T1_s": float(result.T1_sigma_s),
                  "T2_s": float(result.T2_sigma_s),
                  "delta_omega": float(result.delta_omega_sigma),
                  "epsilon_sx": float(result.epsilon_sx_sigma)}
        chi2 = {"T1_s": float(result.t1_chi2_dof),
                "T2_s": float(result.ramsey_chi2_dof),
                "delta_omega": float(result.ramsey_chi2_dof),
                "epsilon_sx": float(result.gate_chi2_dof)}
        chi2_secondary = {"T2_s": float(result.echo_chi2_dof)}
        priors = {"T1_s": float(prior.T1_prior_s),
                  "T2_s": float(prior.T2_prior_s),
                  "delta_omega": NAN,
                  "epsilon_sx": float(prior.epsilon_prior)}

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
                shots_total=sum(shots_list), prior_value=pv,
                prior_absolute_error=_abs_error(pv, tv),
                prior_relative_error=_rel_error(pv, tv)))
        return rows

    except Exception as exc:
        status = f"exception:{type(exc).__name__}:{exc}"
        return [_row(architecture, instance_id, seed, p, timestamp, fit_status=status)
                for p in PARAMETERS]


def _worker(job) -> List[Dict[str, Any]]:
    warnings.filterwarnings("ignore")
    logging.getLogger("qiskit_aer.noise.noise_model").setLevel(logging.ERROR)
    return run_one_instance(*job)


def run_architecture(architecture: str, n_instances: int, seed_offset: int,
                     n_workers: int) -> List[Dict[str, Any]]:
    jobs = [(architecture, i, seed_offset + i) for i in range(n_instances)]
    print(f"[nisq_benchmark] {architecture}: {n_instances} instances, seeds "
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
    print(f"[nisq_benchmark] wrote {len(ordered)} rows -> {path}", flush=True)


def summarize(rows: List[Dict[str, Any]]) -> None:
    print("\n[nisq_benchmark] median relative error, all instances, nothing removed")
    by_arch: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_arch.setdefault(r["architecture"], []).append(r)
    for architecture, arch_rows in sorted(by_arch.items()):
        n_inst = len({r["instance_id"] for r in arch_rows})
        n_bad = len({r["instance_id"] for r in arch_rows if not r["success"]})
        print(f"  {architecture} ({n_inst} instances, {n_bad} with a failed parameter)")
        for param in PARAMETERS:
            vals = np.array([r["relative_error"] for r in arch_rows
                             if r["parameter"] == param], dtype=float)
            finite = vals[np.isfinite(vals)]
            med = float(np.median(finite)) * 100 if len(finite) else NAN
            print(f"     {param:12s} {med:8.2f}%   ({len(finite)}/{len(vals)} finite)")


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
                        "prior_value", "prior_absolute_error", "prior_relative_error"):
                row[key] = float(row[key]) if row[key] not in ("", None) else NAN
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
        print("\n[nisq_benchmark] VALIDATION FAILED:", flush=True)
        for p in problems:
            print(f"  - {p}", flush=True)
        sys.exit(1)
    print(f"\n[nisq_benchmark] validation passed: {n_rows} rows")


def main() -> None:
    ap = argparse.ArgumentParser(description="NISQ parameter-recovery benchmark.")
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--input", type=pathlib.Path, action="append", default=[])
    ap.add_argument("--architecture", choices=list(ARCHITECTURES) + ["all"])
    ap.add_argument("--n-instances", type=int, default=300)
    ap.add_argument("--seed-offset", type=int, default=None)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    if args.n_instances <= 0:
        sys.exit("--n-instances must be positive")
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
        rows.extend(run_architecture(architecture, args.n_instances, offset, n_workers))

    problems = validate_rows(rows, args.n_instances, architectures)
    write_csv(rows, args.out)
    summarize(rows)
    print(f"[nisq_benchmark] elapsed "
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
