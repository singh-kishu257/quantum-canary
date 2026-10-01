"""3b_robustness_envelope.py -- how far each model mismatch can grow before
Canary's parameter recovery degrades, measured one mechanism at a time.

Every run is 3_benchmark.py's mismatched benchmark (same device generator,
calibration prior, probe, time-resolved execution and truth definitions).
All mechanisms sit at their literature values (1c MODEL_PROVENANCE, the
3_benchmark.py defaults) except the one being swept, which is set to a
multiple of its literature value. For each (architecture, mechanism, level)
the script reports, per parameter, R2 of Canary, R2 of the calibration prior
alone, a bootstrap 95% interval for R2, the median relative error and the
|z| <= 2 coverage, and then the ENVELOPE: the largest swept level at which
every parameter keeps R2 >= --threshold.

What the envelope does and does not claim. The levels and the threshold are
fixed in this file before any result is seen (MECHANISMS, --threshold
default 0.95), and every level is reported, including those that fail. The
envelope is the range of mismatch over which the claim holds; it is read
against the literature value (multiplier 1.0) of each mechanism, which is
the realistic operating point, never in place of it.

Output: one long-format CSV (one row per architecture, mechanism, level and
parameter) and a printed envelope table.
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
import zlib
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


bm = _load("nisq_benchmark", "3_benchmark.py")
sim = bm.sim

PARAMETERS = bm.PARAMETERS
ARCHITECTURES = bm.ARCHITECTURES
N_BOOTSTRAP = 1000

# Mechanism -> (3_benchmark cfg key, architectures it acts on, multipliers of
# the literature value). Multiplier 1.0 is the literature operating point;
# levels below it are reported too, so the curve is visible on both sides.
# literature_value(arch) gives the absolute value at multiplier 1.0.
MECHANISMS: Dict[str, Dict[str, Any]] = {
    # Residual qubit detuning population half-width (Hz).
    "dw_halfwidth": {
        "cfg_key": "dw_halfwidth_hz",
        "architectures": ("superconducting", "trapped_ion"),
        "multipliers": (0.5, 1.0, 2.0, 4.0),
        "literature_value": lambda arch: sim.MODEL_PROVENANCE[
            f"{'sc' if arch == 'superconducting' else 'ti'}"
            ".dw_population.literature_halfwidth_hz"]["value"],
    },
    # Quasi-static (shot-to-shot) dephasing std, scaling the per-qubit sigma
    # drawn from the measured T2R/T2E ratios (superconducting only in 1c).
    "quasi_static": {
        "cfg_key": "quasi_static_scale",
        "architectures": ("superconducting",),
        "multipliers": (0.5, 1.0, 2.0, 4.0),
        "literature_value": lambda arch: 1.0,
    },
    # Share of each qubit's gate error realised as coherent amplitude error.
    "coherent_fraction": {
        "cfg_key": "coherent_fraction",
        "architectures": ("superconducting", "trapped_ion"),
        "multipliers": (1.0, 2.0, 3.0, 5.0),
        "literature_value": lambda arch: sim.get_architecture_profile(arch).extra[
            "coherent_fraction"],
    },
    # Stationary amplitude of the detuning drift process.
    "detuning_drift": {
        "cfg_key": "detuning_drift_scale",
        "architectures": ("superconducting", "trapped_ion"),
        "multipliers": (1.0, 2.0, 4.0, 8.0),
        "literature_value": lambda arch: 1.0,
    },
    # Stationary std of the log-T1 OU drift (superconducting only in 1c).
    "t1_drift": {
        "cfg_key": "t1_drift_scale",
        "architectures": ("superconducting",),
        "multipliers": (1.0, 1.5, 2.0, 3.0),
        "literature_value": lambda arch: 1.0,
    },
    # Calibration-record staleness envelope (T1/T2/eps and SPAM priors).
    "prior_staleness": {
        "cfg_key": "perturbation_frac",
        "architectures": ("superconducting", "trapped_ion"),
        "multipliers": (1.0, 1.5, 2.0, 3.0),
        "literature_value": lambda arch: sim.resolve_calibration_perturbation(arch, "nisq"),
    },
    # Time between the calibration record and the probe run (s).
    "calibration_age": {
        "cfg_key": "t_measure_s",
        "architectures": ("superconducting", "trapped_ion"),
        "multipliers": (1.0, 6.0, 24.0),
        "literature_value": lambda arch: bm.T_MEASURE_S,
    },
}

CSV_COLUMNS = ["architecture", "mechanism", "cfg_key", "multiplier", "value",
               "parameter", "n_instances", "n_failed", "R2", "R2_lo95", "R2_hi95",
               "R2_prior", "median_rel_error", "z_cov_2", "passes_threshold",
               "timestamp_utc"]


def _level_cfg(mech: Dict[str, Any], arch: str, multiplier: float) -> Dict[str, Any]:
    value = float(mech["literature_value"](arch)) * float(multiplier)
    cfg: Dict[str, Any] = {mech["cfg_key"]: value}
    if mech["cfg_key"] == "perturbation_frac":
        if not 0.0 <= value < 1.0:
            raise ValueError(f"perturbation_frac level {value} outside [0, 1)")
        # The SPAM record is as stale as the rest of the record (3_benchmark default).
        cfg["spam_perturbation_frac"] = value
    if mech["cfg_key"] == "coherent_fraction" and not 0.0 <= value <= 1.0:
        raise ValueError(f"coherent_fraction level {value} outside [0, 1]")
    return cfg


def _bootstrap_r2(tv: np.ndarray, cv: np.ndarray, rng: np.random.Generator):
    n = len(tv)
    if n < 3:
        return float("nan"), float("nan")
    vals = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        r = bm._r2(tv[idx], cv[idx])
        if np.isfinite(r):
            vals.append(r)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarize_level(rows: List[Dict[str, Any]], threshold: float,
                    seed: int) -> Dict[str, Dict[str, float]]:
    """Per parameter: R2 (failed fits count as failures, never dropped
    silently: a level with any failed fit cannot pass), bootstrap interval,
    prior R2, median relative error and |z| <= 2 coverage."""
    rng = np.random.default_rng(seed)
    out: Dict[str, Dict[str, float]] = {}
    for param in PARAMETERS:
        pr = [r for r in rows if r["parameter"] == param]
        tv = np.array([r["true_value"] for r in pr], dtype=float)
        cv = np.array([r["canary_value"] for r in pr], dtype=float)
        pv = np.array([r["prior_value"] for r in pr], dtype=float)
        re = np.array([r["relative_error"] for r in pr], dtype=float)
        z = np.array([r["z_score"] for r in pr], dtype=float)
        ok = np.array([bool(r["success"]) for r in pr]) & np.isfinite(cv) & np.isfinite(tv)
        both = ok & np.isfinite(pv)
        r2 = bm._r2(tv[ok], cv[ok]) if ok.sum() > 1 else float("nan")
        lo, hi = _bootstrap_r2(tv[ok], cv[ok], rng)
        n_failed = int(len(pr) - ok.sum())
        out[param] = {
            "n_instances": len(pr), "n_failed": n_failed,
            "R2": r2, "R2_lo95": lo, "R2_hi95": hi,
            "R2_prior": bm._r2(tv[both], pv[both]) if both.sum() > 1 else float("nan"),
            "median_rel_error": float(np.median(re[ok & np.isfinite(re)]))
                                if (ok & np.isfinite(re)).any() else float("nan"),
            "z_cov_2": float(np.mean(np.abs(z[np.isfinite(z)]) <= 2))
                       if np.isfinite(z).any() else float("nan"),
            "passes_threshold": bool(np.isfinite(r2) and r2 >= threshold and n_failed == 0),
        }
    return out


def run(architectures: List[str], mechanisms: List[str], n_instances: int,
        n_workers: int, threshold: float, out: pathlib.Path) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    records: List[Dict[str, Any]] = []
    for arch in architectures:
        for name in mechanisms:
            mech = MECHANISMS[name]
            if arch not in mech["architectures"]:
                continue
            for mult in mech["multipliers"]:
                cfg = _level_cfg(mech, arch, mult)
                print(f"[envelope] {arch} {name} x{mult:g} -> {cfg}", flush=True)
                # Same seeds at every level: paired comparison across levels.
                rows = bm.run_architecture(arch, n_instances, bm.ARCH_SEED_OFFSET[arch],
                                           n_workers, cfg)
                # Deterministic bootstrap seed (str hash() is randomized per process).
                summary = summarize_level(rows, threshold,
                                          seed=zlib.crc32(f"{arch}:{name}:{mult}".encode()))
                for param, s in summary.items():
                    records.append({"architecture": arch, "mechanism": name,
                                    "cfg_key": mech["cfg_key"], "multiplier": float(mult),
                                    "value": float(cfg[mech["cfg_key"]]), "parameter": param,
                                    "timestamp_utc": timestamp, **s})
                    print(f"    {param:12s} R2 {s['R2']:.4f} [{s['R2_lo95']:.4f}, "
                          f"{s['R2_hi95']:.4f}]  prior {s['R2_prior']:.4f}  "
                          f"medRE {s['median_rel_error']:.3f}  |z|<=2 {s['z_cov_2']:.2f}  "
                          f"fail {s['n_failed']}", flush=True)
                _write(records, out)
    print_envelope(records, threshold)


def _write(records: List[Dict[str, Any]], out: pathlib.Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def print_envelope(records: List[Dict[str, Any]], threshold: float) -> None:
    """For each (architecture, mechanism): the largest multiplier at which
    every parameter passes, and whether the literature level (1.0) passes."""
    print(f"\n[envelope] largest level with R2 >= {threshold} for EVERY parameter "
          "(multiplier of the literature value; all levels are in the CSV)")
    keys = sorted({(r["architecture"], r["mechanism"]) for r in records})
    for arch, name in keys:
        levels = sorted({r["multiplier"] for r in records
                         if r["architecture"] == arch and r["mechanism"] == name})
        passing = [m for m in levels
                   if all(r["passes_threshold"] for r in records
                          if r["architecture"] == arch and r["mechanism"] == name
                          and r["multiplier"] == m)]
        lit = 1.0 in passing
        worst = {m: min((r["R2"] for r in records if r["architecture"] == arch
                         and r["mechanism"] == name and r["multiplier"] == m),
                        default=float("nan")) for m in levels}
        best = max(passing) if passing else None
        print(f"  {arch:16s} {name:18s} literature level passes: {'yes' if lit else 'NO '}  "
              f"envelope: {'none' if best is None else f'up to x{best:g}'}  "
              f"min R2 per level: " + ", ".join(f"x{m:g}={worst[m]:.3f}" for m in levels))


def main() -> None:
    ap = argparse.ArgumentParser(description="One-at-a-time robustness envelope.")
    ap.add_argument("--architecture", choices=list(ARCHITECTURES) + ["all"], default="all")
    ap.add_argument("--mechanisms", nargs="+", choices=sorted(MECHANISMS),
                    default=sorted(MECHANISMS))
    ap.add_argument("--n-instances", type=int, default=60)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()
    if args.n_instances < 3:
        sys.exit("--n-instances must be >= 3")
    archs = list(ARCHITECTURES) if args.architecture == "all" else [args.architecture]
    n_workers = args.workers if args.workers is not None else max(1, mp.cpu_count() - 1)
    run(archs, args.mechanisms, args.n_instances, n_workers, args.threshold, args.out)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.getLogger("qiskit_aer.noise.noise_model").setLevel(logging.ERROR)
    if sys.platform != "win32":
        try:
            mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    main()
