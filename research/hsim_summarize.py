"""Summarise hsim_before_after.py output: mean fidelity, RMSE, paired dF vs identity with bootstrap CI, wins."""

from __future__ import annotations

import argparse
import json

import numpy as np

REFERENCE = "identity (raw native)"


def summarize(rows: list[dict], resamples: int = 4000, seed: int = 0) -> dict[str, dict]:
    methods = [k[3:] for k in rows[0] if k.startswith("F::")]
    ref = np.array([r[f"F::{REFERENCE}"] for r in rows])
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(rows), size=(resamples, len(rows)))
    out = {}
    for m in methods:
        f = np.array([r[f"F::{m}"] for r in rows])
        d = f - ref
        boot = d[idx].mean(axis=1)
        out[m] = {
            "mean_F": float(f.mean()),
            "rmse": float(np.mean([r[f"R::{m}"] for r in rows])),
            "dF": float(d.mean()),
            "dF_median": float(np.median(d)),
            "dF_min": float(d.min()),
            "dF_max": float(d.max()),
            "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5)),
            "wins": int(np.sum(d > 0)),
            "worse_by_0.03": int(np.sum(d < -0.03)),
            "n": len(rows),
        }
    return out


def table(summary: dict[str, dict]) -> str:
    lines = ["| Method | Mean F | RMSE | dF vs identity [95% CI] | median / min / max dF | wins |", "|---|---|---|---|---|---|"]
    for m, s in summary.items():
        dci = "—" if m == REFERENCE else f"{s['dF']:+.3f} [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]"
        spread = "—" if m == REFERENCE else f"{s['dF_median']:+.3f} / {s['dF_min']:+.3f} / {s['dF_max']:+.3f}"
        wins = "—" if m == REFERENCE else f"{s['wins']}/{s['n']}"
        lines.append(f"| {m} | {s['mean_F']:.3f} | {s['rmse']:.4f} | {dci} | {spread} | {wins} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    args = ap.parse_args()
    rows = json.load(open(args.results))["rows"]
    print(table(summarize(rows)))
    for key in ("cal_fraction", "cal_fraction_sweep", "pairs_measured", "pairs_measured_sweep", "untrusted_batches"):
        if key in rows[0]:
            print(f"{key}: mean {np.mean([r[key] for r in rows]):.3f}")


if __name__ == "__main__":
    main()
