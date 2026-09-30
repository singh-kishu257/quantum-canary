"""Summarise hsim_before_after.py output: mean fidelity, RMSE, paired dF vs identity with bootstrap CI, wins.

``--gains`` adds the fraction of the perfect-screening and oracle gains (over the mean of random maps) that Canary
captures; ``--ablations`` compares several result files (one per variant).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

REFERENCE = "identity (raw native)"
RESAMPLES = 4000


def _boot_idx(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(n, size=(RESAMPLES, n))


def summarize(rows: list[dict]) -> dict[str, dict]:
    methods = [k[3:] for k in rows[0] if k.startswith("F::")]
    ref = np.array([r[f"F::{REFERENCE}"] for r in rows])
    idx = _boot_idx(len(rows))
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
    lines = [
        "| Method | Mean F | Obs. RMSE | dF vs identity [95% CI] | median / min / max dF | wins |",
        "|---|---|---|---|---|---|",
    ]
    for m, s in summary.items():
        dci = "—" if m == REFERENCE else f"{s['dF']:+.3f} [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]"
        spread = "—" if m == REFERENCE else f"{s['dF_median']:+.3f} / {s['dF_min']:+.3f} / {s['dF_max']:+.3f}"
        wins = "—" if m == REFERENCE else f"{s['wins']}/{s['n']}"
        lines.append(f"| {m} | {s['mean_F']:.3f} | {s['rmse']:.4f} | {dci} | {spread} | {wins} |")
    return "\n".join(lines)


def captured(rows: list[dict], method: str, target: str, base: str) -> tuple[float, float, float]:
    """Ratio of mean gains over ``base`` (method vs target) with a paired bootstrap 95% CI."""
    b = np.array([r[f"F::{base}"] for r in rows])
    g = np.array([r[f"F::{method}"] for r in rows]) - b
    h = np.array([r[f"F::{target}"] for r in rows]) - b
    idx = _boot_idx(len(rows), 1)
    ratios = g[idx].mean(axis=1) / h[idx].mean(axis=1)
    return float(g.mean() / h.mean()), float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def gains(rows: list[dict]) -> str:
    base = next(k[3:] for k in rows[0] if k.startswith("F::random map (mean of"))
    targets = [k[3:] for k in rows[0] if k.startswith("F::perfect screening")] + ["oracle map"]
    methods = [
        "Canary final map (raw native)",
        "Canary final map (deployed)",
        "Canary delivered (all batches)",
        "Canary + sweep final map (deployed)",
        "Canary + sweep delivered (all batches)",
        "identity (raw native)",
    ]
    lines = [f"Gain over '{base}' (mean F {np.mean([r['F::' + base] for r in rows]):.3f}):", ""]
    lines.append("| Method | gain | " + " | ".join(f"fraction of {t} gain [95% CI]" for t in targets) + " |")
    lines.append("|---|---|" + "---|" * len(targets))
    for m in methods:
        gain = np.mean([r[f"F::{m}"] - r[f"F::{base}"] for r in rows])
        cells = [f"{c[0]:.2f} [{c[1]:.2f}, {c[2]:.2f}]" for c in (captured(rows, m, t, base) for t in targets)]
        lines.append(f"| {m} | {gain:+.3f} | " + " | ".join(cells) + " |")
    for t in targets:
        gain = np.mean([r[f"F::{t}"] - r[f"F::{base}"] for r in rows])
        lines.append(f"| {t} | {gain:+.3f} | " + " | ".join("—" for _ in targets) + " |")
    return "\n".join(lines)


def diagnostics(rows: list[dict]) -> str:
    keys = [
        "cal_fraction", "cal_fraction_actual", "cal_fraction_sweep", "cal_fraction_sweep_batches", "sweep_shots",
        "shots_spent", "total_shots", "total_shots_sweep", "pairs_measured", "pairs_measured_sweep",
        "untrusted_batches", "Q_identity", "Q_random", "Q_final", "Q_final_sweep", "Q_oracle",
    ]
    keys += [k for k in rows[0] if k.startswith("Q::")]
    lines = ["| Quantity | mean | min | max |", "|---|---|---|---|"]
    for k in keys:
        if k in rows[0]:
            v = np.array([r[k] for r in rows], dtype=float)
            lines.append(f"| {k} | {v.mean():.4g} | {v.min():.4g} | {v.max():.4g} |")
    if "dilution_factor" in rows[0]:
        f = [r["dilution_factor"] for r in rows]
        fitted = [x for x in f if x is not None]
        lines.append(
            f"| dilution_factor (fitted in {len(fitted)}/{len(f)}) | "
            + (f"{np.mean(fitted):.3g} | {min(fitted):.3g} | {max(fitted):.3g} |" if fitted else "— | — | — |")
        )
    if "verdicts" in rows[0]:
        tot = {k: sum(r["verdicts"][k] for r in rows) for k in rows[0]["verdicts"]}
        lines.append(f"| verdicts (total over sessions) | {tot} | | |")
    return "\n".join(lines)


def ablations(paths: list[str]) -> str:
    lines = [
        "| Variant | delivered dF [95% CI] | final dF [95% CI] | worst delivered dF | Q_final | pairs | cal fraction |",
        "|---|---|---|---|---|---|---|",
    ]
    for path in paths:
        data = json.load(open(path))
        rows = data["rows"]
        s = summarize(rows)
        d, f = s["Canary delivered (all batches)"], s["Canary final map (deployed)"]
        lines.append(
            f"| {data.get('variant', path)} | {d['dF']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] | "
            f"{f['dF']:+.3f} [{f['ci_low']:+.3f}, {f['ci_high']:+.3f}] | {d['dF_min']:+.3f} | "
            f"{np.mean([r['Q_final'] for r in rows]):.3f} | {np.mean([r.get('pairs_measured', np.nan) for r in rows]):.1f} | "
            f"{np.mean([r.get('cal_fraction_actual', r['cal_fraction']) for r in rows]):.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--gains", action="store_true")
    ap.add_argument("--ablations", action="store_true")
    args = ap.parse_args()
    if args.ablations:
        print(ablations(args.results))
        return
    rows = json.load(open(args.results[0]))["rows"]
    print(table(summarize(rows)))
    print()
    print(diagnostics(rows))
    if args.gains:
        print()
        print(gains(rows))


if __name__ == "__main__":
    main()
