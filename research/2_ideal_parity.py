"""
2_ideal_parity.py -- end-to-end parameter-recovery accuracy benchmark
(the "ideal parity" / Figure 2 experiment).

Measures how accurately Quantum Canary recovers the four parameters it
actually estimates -- T1, T2, delta_omega, epsilon_sx -- from realistically
simulated ideal (stationary Markovian) hardware, running Canary's REAL
end-to-end workflow rather than any simplified estimator.

Per instance:

    1c.generate_device(regime="ideal")        -> DeviceTruth   (hidden truth)
    1c.generate_calibration_prior(truth)      -> CalibrationPrior
    1_inversion.BackendProfile(prior values)  -> what Canary is allowed to know
    1_inversion.build_probe_circuits(profile) -> Canary's real probe circuits
    1c.simulate_probe(circuits, truth)        -> Aer execution on the device
    1_inversion.lindblad_inversion(counts)    -> Canary's frozen inversion
    ...only then compare against DeviceTruth.

SPAM is deliberately NOT a benchmarked parameter: Canary's frozen inversion
never estimates it. p0|1 and p1|0 are known inputs, taken from the
calibration prior and used to build SPAM-aware forward models that correct
the four parameters above. There is no SPAM estimate in InversionResult to
compare against, and inventing one would misrepresent the method.

-----------------------------------------------------------------------------
SCIENTIFIC-INTEGRITY INVARIANTS (each enforced or structurally guaranteed)
-----------------------------------------------------------------------------
* Ground truth never reaches Canary. DeviceTruth is read in exactly three
  places: (a) generate_calibration_prior(), where 1c's own calibration
  pipeline is designed to observe truth through synthetic binomial
  measurement noise (never to copy it); (b) simulate_probe(), which is the
  physical device; (c) the final ground-truth comparison, after inversion
  has already returned. Note this is also structurally guaranteed:
  build_probe_circuits() and lindblad_inversion() do not accept a truth
  object at all -- there is no parameter through which truth could leak.
* No selection on prior quality. Prior-vs-truth agreement is recorded as a
  diagnostic and never gates, reorders, regenerates or rejects an instance.
  Enforced by the contiguous-instance_id validation check: any "skip bad
  instance" logic would leave a gap and fail the run.
* Failed / non-convergent instances are retained as explicit NaN rows with a
  fit_status string, never silently dropped. Row count is therefore exactly
  4 * n_instances regardless of how many failed.
* Fixed 9,900-shot budget per instance (300*3 + 1000*6 + 500*3 + 500*3),
  which is also 1_inversion.py's own native default allocation.
* Deterministic seeds: seed = seed_offset + instance_id. Results do not
  depend on worker count or execution order.
* No post-hoc tuning: this script does not modify, re-run, or filter any
  instance based on the results it produces.

Output: one CSV. No figures (figure generation lives elsewhere).
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
from typing import Any, Dict, List, Optional

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def _load(module_name: str, filename: str):
    """Load a sibling numerically-named module (not importable normally)."""
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


inv = _load("inversion", "1_inversion.py")
sim = _load("sim1c", "1c_simulation_models.py")

# --- Fixed experimental configuration (never varied per instance) ----------
SHOTS_T1 = 300
SHOTS_RAMSEY = 1000
SHOTS_GATE = 500
SHOTS_ECHO = 500
SHOTS_TOTAL_EXPECTED = 9900

ARCHITECTURES = ("superconducting", "trapped_ion")

# Non-overlapping so the parallel CI jobs can never collide.
ARCH_SEED_OFFSET: Dict[str, int] = {
    "superconducting": 0,
    "trapped_ion": 100_000,
}

# The four parameters Canary actually estimates. Order is fixed so rows are
# deterministic. delta_omega is compared as |delta_omega|, matching the
# manuscript's own Figure 2(c) axes: the sign is genuinely ambiguous outside
# the arctan2 bound (see 1_inversion.py's dw_max clipping).
PARAMETERS = ("T1_s", "T2_s", "delta_omega", "epsilon_sx")

CSV_COLUMNS = [
    "architecture", "instance_id", "seed", "parameter",
    "true_value", "canary_value", "absolute_error", "relative_error",
    "success", "fit_status",
    "sigma_canary", "chi2_dof", "chi2_dof_secondary", "shots_total",
    "prior_value", "prior_absolute_error", "prior_relative_error",
    "prior_confidence", "spam_source", "n_circuits", "timestamp_utc",
]

# Circuit-name prefixes 1_inversion.py's build_probe_circuits() emits. Used
# to verify the circuits actually executed are Canary's real characterization
# circuits and not some substituted/simplified set.
EXPECTED_CIRCUIT_PREFIXES = ("t1_", "ramsey_X_", "ramsey_Y_", "gate_rep_N", "echo_")


def _rel_error(value: float, truth_value: float) -> float:
    if not np.isfinite(value) or not np.isfinite(truth_value) or truth_value == 0:
        return float("nan")
    return abs(value - truth_value) / abs(truth_value)


def _abs_error(value: float, truth_value: float) -> float:
    if not np.isfinite(value) or not np.isfinite(truth_value):
        return float("nan")
    return abs(value - truth_value)


def _failure_rows(architecture: str, instance_id: int, seed: int,
                  fit_status: str, timestamp: str) -> List[Dict[str, Any]]:
    """Rows emitted when an instance raises. Retained, never dropped, so the
    output always contains exactly 4 rows per attempted instance."""
    nan = float("nan")
    return [{
        "architecture": architecture, "instance_id": instance_id, "seed": seed,
        "parameter": param,
        "true_value": nan, "canary_value": nan,
        "absolute_error": nan, "relative_error": nan,
        "success": False, "fit_status": fit_status,
        "sigma_canary": nan, "chi2_dof": nan, "chi2_dof_secondary": nan,
        "shots_total": SHOTS_TOTAL_EXPECTED,
        "prior_value": nan, "prior_absolute_error": nan, "prior_relative_error": nan,
        "prior_confidence": "", "spam_source": "", "n_circuits": 0,
        "timestamp_utc": timestamp,
    } for param in PARAMETERS]


def run_one_instance(architecture: str, instance_id: int, seed: int) -> List[Dict[str, Any]]:
    """Run one complete simulated-device instance through Canary's real
    end-to-end workflow and return exactly 4 parameter-level rows.

    Never raises: any failure is captured as 4 explicit NaN rows carrying a
    fit_status describing the failure, so difficult instances are represented
    rather than removed.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        # --- (1) hidden ground truth -------------------------------------
        truth = sim.generate_device(architecture, n_qubits=1, seed=seed, regime="ideal")

        # --- (2) calibration prior: what Canary is legitimately given -----
        # 1c's calibration pipeline synthesizes real binomial-sampled
        # measurements at architecture-typical probe points and inverts them
        # with a deliberately crude estimator. It never copies truth.
        prior = sim.generate_calibration_prior(truth, qubit=0, t0_s=0.0, shots=200)

        # --- (3) build the profile from PRIOR values only ------------------
        # From here until the final comparison, `truth` is used only by
        # simulate_probe() (the physical device). Nothing below reads it.
        arch_constants = dict(inv.ARCH_DEFAULTS[architecture])
        arch_constants["T1_s"] = prior.T1_prior_s
        arch_constants["T2_s"] = prior.T2_prior_s
        arch_constants["eps_typical"] = prior.epsilon_prior
        arch_constants["p0_given_1"] = prior.spam_p0_given_1_prior
        arch_constants["p1_given_0"] = prior.spam_p1_given_0_prior

        profile = inv.BackendProfile(
            architecture=architecture,
            T1_prior_s=prior.T1_prior_s,
            T2_prior_s=prior.T2_prior_s,
            dt_ns=inv.ARCH_DEFAULTS[architecture]["dt_ns"],
            backend_name=f"ideal_parity_sim::{architecture}",
            custom_arch=arch_constants,
            # A synthetic calibration measurement was actually performed, so
            # Canary is entitled to its Fisher-informed (prior-centred) delay
            # schedule -- the same branch from_ibm_backend()/from_ionq_backend()
            # take for real live calibration data.
            prior_confidence="live",
        )

        # --- (4) Canary's own adaptive workflow ---------------------------
        circuits, meta = inv.build_probe_circuits(profile)

        # Verify these really are Canary's characterization circuits.
        for qc in circuits:
            if not any(qc.name.startswith(p) for p in EXPECTED_CIRCUIT_PREFIXES):
                raise RuntimeError(
                    f"unexpected circuit name {qc.name!r}; expected one of "
                    f"{EXPECTED_CIRCUIT_PREFIXES}")

        shots_list = ([SHOTS_T1] * meta["n_t1"]
                      + [SHOTS_RAMSEY] * meta["n_ramsey"]
                      + [SHOTS_GATE] * meta["n_gate"]
                      + [SHOTS_ECHO] * meta["n_echo"])
        if len(shots_list) != len(circuits):
            raise RuntimeError(
                f"shot allocation length {len(shots_list)} != circuit count {len(circuits)}")
        if sum(shots_list) != SHOTS_TOTAL_EXPECTED:
            raise RuntimeError(
                f"shot budget {sum(shots_list)} != required {SHOTS_TOTAL_EXPECTED}")

        # --- (5) execute Canary's circuits on the simulated device --------
        sim_result = sim.simulate_probe(
            circuits, truth, shots=shots_list, t_s=0.0, dt_ns=profile.dt_ns)

        # --- (6) frozen Canary inversion ----------------------------------
        result = inv.lindblad_inversion(
            sim_result.counts_list, meta, profile,
            shots_t1=SHOTS_T1, shots_ramsey=SHOTS_RAMSEY,
            shots_gate=SHOTS_GATE, shots_echo=SHOTS_ECHO,
            qubit_id=0, timestamp=timestamp)

        # --- (7)+(8) extract, THEN compare against ground truth -----------
        # First read of truth's parameter values anywhere in this function.
        true_values = {
            "T1_s": float(truth.T1_s[0]),
            "T2_s": float(truth.T2_s[0]),
            "delta_omega": abs(float(truth.delta_omega_rad_s[0])),
            "epsilon_sx": float(truth.epsilon_gate[0]),
        }
        canary_values = {
            "T1_s": float(result.T1_s),
            "T2_s": float(result.T2_s),
            "delta_omega": abs(float(result.delta_omega)),
            "epsilon_sx": float(result.epsilon_sx),
        }
        sigmas = {
            "T1_s": float(result.T1_sigma_s),
            "T2_s": float(result.T2_sigma_s),
            "delta_omega": float(result.delta_omega_sigma),
            "epsilon_sx": float(result.epsilon_sx_sigma),
        }
        # Probe-specific goodness of fit. T2 is a chi2-weighted fusion of the
        # Ramsey and echo probes, so it carries both: ramsey as primary,
        # echo as secondary. No single chi2/dof represents it alone.
        chi2 = {
            "T1_s": float(result.t1_chi2_dof),
            "T2_s": float(result.ramsey_chi2_dof),
            "delta_omega": float(result.ramsey_chi2_dof),
            "epsilon_sx": float(result.gate_chi2_dof),
        }
        chi2_secondary = {
            "T1_s": float("nan"),
            "T2_s": float(result.echo_chi2_dof),
            "delta_omega": float("nan"),
            "epsilon_sx": float("nan"),
        }
        # Prior-vs-truth audit. DIAGNOSTIC ONLY -- recorded for analysis of
        # prior informativeness, never used to select, reorder, regenerate or
        # reject any instance.
        prior_values = {
            "T1_s": float(prior.T1_prior_s),
            "T2_s": float(prior.T2_prior_s),
            "delta_omega": abs(float(prior.delta_omega_prior_rad_s)),
            "epsilon_sx": float(prior.epsilon_prior),
        }

        spam_source = ""
        if profile.calibration_source is not None:
            spam_source = profile.calibration_source.spam_source

        rows: List[Dict[str, Any]] = []
        for param in PARAMETERS:
            tv, cv, sg = true_values[param], canary_values[param], sigmas[param]
            converged = bool(np.isfinite(cv) and np.isfinite(sg))
            rows.append({
                "architecture": architecture,
                "instance_id": instance_id,
                "seed": seed,
                "parameter": param,
                "true_value": tv,
                "canary_value": cv,
                "absolute_error": _abs_error(cv, tv),
                "relative_error": _rel_error(cv, tv),
                "success": converged,
                "fit_status": "ok" if converged else "non_convergent_infinite_sigma",
                "sigma_canary": sg,
                "chi2_dof": chi2[param],
                "chi2_dof_secondary": chi2_secondary[param],
                "shots_total": sum(shots_list),
                "prior_value": prior_values[param],
                "prior_absolute_error": _abs_error(prior_values[param], tv),
                "prior_relative_error": _rel_error(prior_values[param], tv),
                "prior_confidence": profile.prior_confidence,
                "spam_source": spam_source,
                "n_circuits": len(circuits),
                "timestamp_utc": timestamp,
            })
        return rows

    except Exception as exc:  # retained, not dropped
        return _failure_rows(architecture, instance_id, seed,
                             f"exception:{type(exc).__name__}:{exc}", timestamp)


def _worker(args) -> List[Dict[str, Any]]:
    warnings.filterwarnings("ignore")
    logging.getLogger("qiskit_aer.noise.noise_model").setLevel(logging.ERROR)
    architecture, instance_id, seed = args
    return run_one_instance(architecture, instance_id, seed)


def run_architecture(architecture: str, n_instances: int, seed_offset: int,
                     n_workers: int) -> List[Dict[str, Any]]:
    """Run every instance for one architecture.

    Every generated instance is executed and recorded. There is deliberately
    no branch here that inspects results, prior quality, or fit success to
    decide whether an instance is kept.
    """
    jobs = [(architecture, i, seed_offset + i) for i in range(n_instances)]

    print(f"[ideal_parity] architecture={architecture} instances={n_instances} "
          f"seeds={seed_offset}..{seed_offset + n_instances - 1} workers={n_workers}",
          flush=True)

    results: List[Optional[List[Dict[str, Any]]]] = [None] * n_instances
    if n_workers <= 1:
        for idx, job in enumerate(jobs):
            results[idx] = _worker(job)
            if (idx + 1) % 25 == 0:
                print(f"  {idx + 1}/{n_instances} complete", flush=True)
    else:
        with mp.Pool(n_workers) as pool:
            for idx, rows in enumerate(pool.imap(_worker, jobs, chunksize=1)):
                results[idx] = rows
                if (idx + 1) % 25 == 0:
                    print(f"  {idx + 1}/{n_instances} complete", flush=True)

    rows: List[Dict[str, Any]] = []
    for r in results:
        if r is None:
            raise RuntimeError("worker returned no rows; refusing to emit a partial CSV")
        rows.extend(r)
    return rows


def validate_rows(rows: List[Dict[str, Any]], expected_instances_per_arch: int,
                  expected_architectures: List[str]) -> List[str]:
    """Return a list of validation failures (empty means everything passed).

    These are hard requirements of the experimental design, not style checks.
    """
    problems: List[str] = []

    by_arch: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_arch.setdefault(row["architecture"], []).append(row)

    missing = set(expected_architectures) - set(by_arch)
    if missing:
        problems.append(f"missing architectures: {sorted(missing)}")
    unexpected = set(by_arch) - set(expected_architectures)
    if unexpected:
        problems.append(f"unexpected architectures: {sorted(unexpected)}")

    for architecture, arch_rows in sorted(by_arch.items()):
        ids = sorted({r["instance_id"] for r in arch_rows})

        # Exactly N instances, and contiguous 0..N-1. A gap here is the
        # signature of instances having been skipped or filtered -- which is
        # exactly what must never happen.
        if len(ids) != expected_instances_per_arch:
            problems.append(
                f"{architecture}: {len(ids)} unique instances, expected "
                f"{expected_instances_per_arch}")
        if ids != list(range(expected_instances_per_arch)):
            problems.append(
                f"{architecture}: instance_ids are not contiguous 0..{expected_instances_per_arch - 1} "
                f"(gaps indicate instances were skipped or filtered)")

        # Exactly the 4 parameters per instance, no more, no fewer.
        per_instance: Dict[int, List[str]] = {}
        for r in arch_rows:
            per_instance.setdefault(r["instance_id"], []).append(r["parameter"])
        for iid, params in sorted(per_instance.items()):
            if sorted(params) != sorted(PARAMETERS):
                problems.append(
                    f"{architecture} instance {iid}: parameters {sorted(params)} "
                    f"!= {sorted(PARAMETERS)}")

        expected_rows = expected_instances_per_arch * len(PARAMETERS)
        if len(arch_rows) != expected_rows:
            problems.append(
                f"{architecture}: {len(arch_rows)} rows, expected {expected_rows}")

        # One unique seed per instance, no duplicates.
        seeds = [r["seed"] for r in arch_rows]
        seed_by_instance = {r["instance_id"]: r["seed"] for r in arch_rows}
        if len(set(seeds)) != expected_instances_per_arch:
            problems.append(
                f"{architecture}: {len(set(seeds))} unique seeds, expected "
                f"{expected_instances_per_arch}")
        if len(set(seed_by_instance.values())) != len(seed_by_instance):
            problems.append(f"{architecture}: duplicate seeds across instances")

        # Fixed shot budget on every row.
        bad_shots = {r["shots_total"] for r in arch_rows} - {SHOTS_TOTAL_EXPECTED}
        if bad_shots:
            problems.append(
                f"{architecture}: shots_total values {sorted(bad_shots)} != {SHOTS_TOTAL_EXPECTED}")

        # Every row must carry an identifiable fit_status.
        blank_status = [r["instance_id"] for r in arch_rows if not str(r["fit_status"]).strip()]
        if blank_status:
            problems.append(
                f"{architecture}: {len(blank_status)} rows have a blank fit_status")

    # Global duplicate check across the whole aggregated file.
    keys = [(r["architecture"], r["instance_id"], r["parameter"]) for r in rows]
    if len(set(keys)) != len(keys):
        problems.append("duplicate (architecture, instance_id, parameter) rows present")
    arch_seed_keys = {(r["architecture"], r["seed"]) for r in rows}
    expected_arch_seed = len(expected_architectures) * expected_instances_per_arch
    if len(arch_seed_keys) != expected_arch_seed:
        problems.append(
            f"{len(arch_seed_keys)} unique (architecture, seed) pairs, expected "
            f"{expected_arch_seed}")

    return problems


def write_csv(rows: List[Dict[str, Any]], path: pathlib.Path) -> None:
    """Write rows in a deterministic order: architecture, instance_id, then
    the fixed PARAMETERS order."""
    param_rank = {p: i for i, p in enumerate(PARAMETERS)}
    ordered = sorted(rows, key=lambda r: (r["architecture"], r["instance_id"],
                                          param_rank.get(r["parameter"], 99)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(ordered)
    print(f"[ideal_parity] wrote {len(ordered)} rows -> {path}", flush=True)


def summarize(rows: List[Dict[str, Any]]) -> None:
    """Print a plain, unfiltered summary. Reports every instance, including
    failures and poor recoveries -- no trimming, no outlier removal."""
    print("\n[ideal_parity] summary (median relative error, ALL instances "
          "including failures; no outliers removed)")
    by_arch: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_arch.setdefault(r["architecture"], []).append(r)
    for architecture, arch_rows in sorted(by_arch.items()):
        n_inst = len({r["instance_id"] for r in arch_rows})
        n_failed = len({r["instance_id"] for r in arch_rows if not r["success"]})
        print(f"  {architecture}  ({n_inst} instances, "
              f"{n_failed} with >=1 non-converged/failed parameter)")
        for param in PARAMETERS:
            vals = np.array([r["relative_error"] for r in arch_rows
                             if r["parameter"] == param], dtype=float)
            finite = vals[np.isfinite(vals)]
            med = float(np.median(finite)) * 100 if len(finite) else float("nan")
            print(f"     {param:12s} median rel.err = {med:8.2f}%   "
                  f"({len(finite)}/{len(vals)} finite)")


def _read_csv(path: pathlib.Path) -> List[Dict[str, Any]]:
    """Read a per-architecture CSV back, restoring the types the validation
    checks rely on. Values are otherwise passed through untouched."""
    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError(
                f"{path.name}: unexpected columns.\n  got:      {reader.fieldnames}\n"
                f"  expected: {CSV_COLUMNS}")
        for raw in reader:
            row = dict(raw)
            row["instance_id"] = int(row["instance_id"])
            row["seed"] = int(row["seed"])
            row["shots_total"] = int(row["shots_total"])
            row["n_circuits"] = int(row["n_circuits"])
            row["success"] = str(row["success"]).strip().lower() == "true"
            for key in ("true_value", "canary_value", "absolute_error", "relative_error",
                        "sigma_canary", "chi2_dof", "chi2_dof_secondary",
                        "prior_value", "prior_absolute_error", "prior_relative_error"):
                row[key] = float(row[key]) if row[key] not in ("", None) else float("nan")
            rows.append(row)
    return rows


def combine(inputs: List[pathlib.Path], out: pathlib.Path,
            expected_instances_per_arch: int) -> None:
    """Deterministically merge per-architecture CSVs into the final file.

    Concatenation only -- no filtering, deduplication, re-ordering by value,
    or re-running of any instance. Fails loudly rather than emitting an
    incomplete or silently-truncated result.
    """
    if not inputs:
        sys.exit("--combine requires at least one --input")

    all_rows: List[Dict[str, Any]] = []
    for path in inputs:
        if not path.exists():
            sys.exit(f"missing input CSV: {path}")
        rows = _read_csv(path)
        arches = sorted({r["architecture"] for r in rows})
        print(f"[ideal_parity] {path.name}: {len(rows)} rows, architectures={arches}")
        all_rows.extend(rows)

    architectures = sorted({r["architecture"] for r in all_rows})
    problems = validate_rows(all_rows, expected_instances_per_arch, architectures)

    write_csv(all_rows, out)
    summarize(all_rows)

    expected_total = len(architectures) * expected_instances_per_arch * len(PARAMETERS)
    if len(all_rows) != expected_total:
        problems.append(f"aggregate row count {len(all_rows)} != expected {expected_total}")

    if problems:
        print("\n[ideal_parity] AGGREGATE VALIDATION FAILED:", flush=True)
        for p in problems:
            print(f"  - {p}", flush=True)
        sys.exit(1)
    print(f"\n[ideal_parity] aggregate validation passed: {len(all_rows)} rows "
          f"({len(architectures)} architectures x {expected_instances_per_arch} instances "
          f"x {len(PARAMETERS)} parameters)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ideal-parity end-to-end parameter-recovery benchmark (Figure 2).")
    ap.add_argument("--combine", action="store_true",
                    help="Merge per-architecture CSVs into the final aggregate file.")
    ap.add_argument("--input", type=pathlib.Path, action="append", default=[],
                    help="Input CSV for --combine (repeatable).")
    ap.add_argument("--architecture",
                    choices=list(ARCHITECTURES) + ["all"])
    ap.add_argument("--n-instances", type=int, default=300)
    ap.add_argument("--seed-offset", type=int, default=None,
                    help="Defaults to the architecture's reserved offset "
                         "(superconducting=0, trapped_ion=100000).")
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
        sys.exit("--seed-offset cannot be combined with --architecture all "
                 "(each architecture needs its own non-overlapping range)")

    n_workers = args.workers if args.workers is not None else max(1, mp.cpu_count() - 1)

    started = datetime.now(timezone.utc)
    print(f"[ideal_parity] start {started.isoformat()}")
    print(f"[ideal_parity] shot budget per instance = {SHOTS_TOTAL_EXPECTED} "
          f"(T1 {SHOTS_T1}x3, Ramsey {SHOTS_RAMSEY}x6, gate {SHOTS_GATE}x3, echo {SHOTS_ECHO}x3)")

    all_rows: List[Dict[str, Any]] = []
    for architecture in architectures:
        offset = (args.seed_offset if args.seed_offset is not None
                  else ARCH_SEED_OFFSET[architecture])
        all_rows.extend(run_architecture(architecture, args.n_instances, offset, n_workers))

    problems = validate_rows(all_rows, args.n_instances, architectures)
    write_csv(all_rows, args.out)
    summarize(all_rows)

    if problems:
        print("\n[ideal_parity] VALIDATION FAILED:", flush=True)
        for p in problems:
            print(f"  - {p}", flush=True)
        sys.exit(1)

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    print(f"\n[ideal_parity] validation passed; {len(all_rows)} rows in {elapsed:.1f}s")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.getLogger("qiskit_aer.noise.noise_model").setLevel(logging.ERROR)
    if sys.platform != "win32":
        try:
            mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    main()
