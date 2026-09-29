"""Figures and summary tables for experiments 1-4 (whatever data exists)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "research" / "data"
FIGS = ROOT / "research" / "figures"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})
COLORS = {
    "naive": "#9a9a9a",
    "random": "#bdbdbd",
    "debias": "#6b8fb5",
    "canary_cumulative": "#e6a24c",
    "canary_final": "#d9661f",
    "oracle": "#2f7d4f",
}
LABELS = {
    "naive": "Identity map",
    "random": "Random map",
    "debias": "Debias (25 maps, avg.)",
    "canary_cumulative": "Canary (all batches)",
    "canary_final": "Canary (final map)",
    "oracle": "Oracle map",
}


def save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def summarize(df: pd.DataFrame, methods: list[str], ref: str) -> pd.DataFrame:
    rows = []
    for m in methods:
        inf = 1 - df[m]
        rows.append(
            {
                "method": m,
                "fidelity_mean": df[m].mean(),
                "fidelity_sem": df[m].std(ddof=1) / np.sqrt(len(df)),
                "infidelity_mean": inf.mean(),
                "infidelity_median": inf.median(),
                "infidelity_ratio_vs_ref_geomean": float(np.exp(np.mean(np.log((1 - df[ref]) / inf)))),
                "win_rate_vs_ref": float(np.mean(df[m] > df[ref])),
            }
        )
    return pd.DataFrame(rows)


def exp1() -> None:
    path = DATA / "exp1" / "instances.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    methods = ["naive", "random", "debias", "canary_cumulative", "canary_final", "oracle"]
    table = summarize(df, methods, "random")
    gap = (df["canary_final"] - df["random"]) / (df["oracle"] - df["random"])
    extra = {
        "instances": len(df),
        "oracle_gap_recovered_median": float(gap.median()),
        "oracle_gap_recovered_mean": float(gap.mean()),
        "canary_final_beats_naive": float(np.mean(df["canary_final"] > df["naive"])),
        "canary_final_beats_debias": float(np.mean(df["canary_final"] > df["debias"])),
        "infidelity_reduction_vs_naive_geomean": float(np.exp(np.mean(np.log((1 - df["naive"]) / (1 - df["canary_final"]))))),
        "infidelity_reduction_vs_debias_geomean": float(np.exp(np.mean(np.log((1 - df["debias"]) / (1 - df["canary_final"]))))),
        "probe_shot_fraction": float(df["probe_shots"].sum() / (df["probe_shots"].sum() + df["algorithm_shots"].sum())),
        "naive_uses_bad_pair_fraction": float(np.mean(df["naive_uses_bad_pair"] > 0)),
    }
    sub = df[df["naive_uses_bad_pair"] > 0]
    if len(sub):
        extra["subset_naive_hits_bad_pair_n"] = len(sub)
        extra["subset_infidelity_reduction_vs_naive_geomean"] = float(
            np.exp(np.mean(np.log((1 - sub["naive"]) / (1 - sub["canary_final"]))))
        )
    table.to_csv(DATA / "exp1" / "summary_table.csv", index=False)
    (DATA / "exp1" / "summary.json").write_text(json.dumps(extra, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), gridspec_kw={"width_ratios": [1.3, 1, 1.1]})
    ax = axes[0]
    data = [1 - df[m] for m in methods]
    parts = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
    for patch, m in zip(parts["boxes"], methods):
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(0.85)
    for k, m in enumerate(methods):
        x = np.random.default_rng(k).normal(k + 1, 0.05, len(df))
        ax.scatter(x, 1 - df[m], s=6, color="k", alpha=0.35, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(methods) + 1), [LABELS[m] for m in methods], rotation=35, ha="right")
    ax.set_ylabel("Infidelity  1 − F_H")
    ax.set_title("(a) Output infidelity per device")

    ax = axes[1]
    hit = df["naive_uses_bad_pair"] > 0
    lo, hi = (1 - df[["naive", "canary_final"]]).min().min() * 0.8, (1 - df[["naive", "canary_final"]]).max().max() * 1.2
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.8, ls="--")
    ax.scatter(1 - df.loc[~hit, "naive"], 1 - df.loc[~hit, "canary_final"], s=14, color=COLORS["debias"], label="identity map avoids outliers")
    ax.scatter(1 - df.loc[hit, "naive"], 1 - df.loc[hit, "canary_final"], s=14, color=COLORS["canary_final"], label="identity map hits outlier pair")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Infidelity, identity map")
    ax.set_ylabel("Infidelity, Canary final map")
    ax.set_title("(b) Paired devices")
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    ax = axes[2]
    traces = json.loads((DATA / "exp1" / "traces.json").read_text())
    rec = np.array([t["recommended_fidelity"] for t in traces])
    trusted = np.array([t["batch_trusted_fidelity"] for t in traces])
    b = np.arange(1, rec.shape[1] + 1)
    for arr, m, lab in ((trusted, "canary_cumulative", "Canary, executed (trusted)"), (rec, "canary_final", "Canary, recommended map")):
        mu = 1 - arr.mean(0)
        se = arr.std(0, ddof=1) / np.sqrt(arr.shape[0])
        ax.plot(b, mu, marker="o", ms=3, color=COLORS[m], label=lab)
        ax.fill_between(b, mu - se, mu + se, color=COLORS[m], alpha=0.2)
    for m in ("random", "debias", "oracle"):
        ax.axhline(1 - df[m].mean(), color=COLORS[m], ls="--", lw=1, label=LABELS[m])
    ax.set_xlabel("Batch (4000 shots each)")
    ax.set_ylabel("Mean infidelity")
    ax.set_title("(c) Online learning")
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig1_before_after")


def exp2() -> None:
    path = DATA / "exp2" / "instances.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    r = df[df["sweep"] == "reuse"]
    g = r.groupby("steps")
    for m in ("random", "debias", "canary_final", "oracle"):
        mu = 1 - g[m].mean()
        ax = axes[0]
        ax.plot(mu.index, mu.values, marker="o", ms=3, color=COLORS[m], label=LABELS[m])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Trotter steps (ZZ uses per pair)")
    axes[0].set_ylabel("Mean infidelity")
    axes[0].set_title("(a) Infidelity vs pair reuse")
    axes[0].legend(frameon=False, fontsize=7)
    ratio = r.assign(gain=(1 - r["random"]) / (1 - r["canary_final"])).groupby("steps")["gain"]
    axes[1].errorbar(ratio.mean().index, ratio.mean().values, yerr=ratio.sem().values, marker="o", ms=3, color=COLORS["canary_final"], capsize=2)
    axes[1].axhline(1, color="0.6", ls="--", lw=0.8)
    axes[1].set_xlabel("Trotter steps (ZZ uses per pair)")
    axes[1].set_ylabel("Infidelity reduction vs random map")
    axes[1].set_title("(b) Gain vs pair reuse")
    h = pd.concat([df[df["sweep"] == "heterogeneity"], r[r["steps"] == 8]])
    hg = h.assign(gain=(1 - h["random"]) / (1 - h["canary_final"])).groupby("outlier_fraction")["gain"]
    axes[2].errorbar(hg.mean().index, hg.mean().values, yerr=hg.sem().values, marker="s", ms=3, color=COLORS["canary_final"], capsize=2)
    axes[2].axhline(1, color="0.6", ls="--", lw=0.8)
    axes[2].set_xlabel("Fraction of outlier ion pairs")
    axes[2].set_ylabel("Infidelity reduction vs random map")
    axes[2].set_title("(c) Gain requires heterogeneity")
    fig.tight_layout()
    save(fig, "fig2_reuse_law")
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    sc = ax.scatter(r["predicted_log_gain"], r["measured_log_gain"], c=r["steps"], s=12, cmap="viridis")
    lim = [min(r["predicted_log_gain"].min(), r["measured_log_gain"].min()), max(r["predicted_log_gain"].max(), r["measured_log_gain"].max())]
    ax.plot(lim, lim, color="0.6", ls="--", lw=0.8)
    corr = np.corrcoef(r["predicted_log_gain"], r["measured_log_gain"])[0, 1]
    ax.set_xlabel("Predicted Δ log fidelity (error model)")
    ax.set_ylabel("Measured Δ log F_H")
    ax.set_title(f"(d) Predictive law, r = {corr:.2f}")
    fig.colorbar(sc, ax=ax, label="Trotter steps")
    fig.tight_layout()
    save(fig, "fig2d_predictive_law")


def exp3() -> None:
    for target in ("static", "canary"):
        _exp3(target)


def _exp3(target: str) -> None:
    d = DATA / f"exp3_{target}"
    path = d / "series.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    cfg = json.loads((d / "config.json").read_text())
    summ = json.loads((d / "summary.json").read_text())
    g = df.groupby("batch")
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for col, color, lab in (
        ("static_oracle", COLORS["oracle"], "Static best map (perfect pre-event calibration)"),
        ("debias", COLORS["debias"], LABELS["debias"]),
        ("canary_trusted", COLORS["canary_cumulative"], "Canary, executed (trusted)"),
        ("canary_recommended", COLORS["canary_final"], "Canary, recommended map"),
    ):
        mu = 1 - g[col].mean()
        se = g[col].sem()
        ax.plot(mu.index + 1, mu.values, marker="o", ms=3, color=color, label=lab)
        ax.fill_between(mu.index + 1, mu - se, mu + se, color=color, alpha=0.2)
    ax.axvline(cfg["event_batch"] + 1, color="k", ls=":", lw=1)
    ax.text(cfg["event_batch"] + 1.2, ax.get_ylim()[1] * 0.95, "pair degrades", fontsize=7, va="top")
    ax.set_xlabel("Batch (10 min apart)")
    ax.set_ylabel("Mean infidelity")
    ax.set_title(f"Drift event on {'static-map' if target == 'static' else 'Canary-map'} pair")
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    save(fig, f"fig3_drift_{target}")
    lat = [s["reroute_latency_batches"] for s in summ]
    det = [s["detect_latency_batches"] for s in summ]
    out = {
        "instances": len(summ),
        "reroute_latency_median": float(np.median([x for x in lat if x is not None])) if any(x is not None for x in lat) else None,
        "rerouted_fraction": float(np.mean([x is not None for x in lat])),
        "detect_latency_median": float(np.median([x for x in det if x is not None])) if any(x is not None for x in det) else None,
        "detected_fraction": float(np.mean([x is not None for x in det])),
        "event_pair_in_static_fraction": float(np.mean([s["event_pair_in_static"] for s in summ])),
        "event_pair_in_canary_fraction": float(np.mean([s.get("event_pair_in_canary", True) for s in summ])),
    }
    (d / "latency_summary.json").write_text(json.dumps(out, indent=2))


def exp4() -> None:
    path = DATA / "exp4" / "instances.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    order = ["qaoa_p1", "ising_s2", "qaoa_p4", "ising_s8"]
    order = [w for w in order if w in set(df["workload"])]
    methods = ["naive", "random", "debias", "canary_final", "oracle"]
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    width = 0.16
    for k, m in enumerate(methods):
        mu = [1 - df[df["workload"] == w][m].mean() for w in order]
        se = [df[df["workload"] == w][m].sem() for w in order]
        ax.bar(np.arange(len(order)) + (k - 2) * width, mu, width, yerr=se, color=COLORS[m], label=LABELS[m], capsize=1.5)
    ax.set_xticks(range(len(order)), order)
    ax.set_yscale("log")
    ax.set_ylabel("Mean infidelity")
    ax.set_title("Algorithm families")
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    fig.tight_layout()
    save(fig, "fig4_algorithms")
    tab = df.assign(gain_vs_random=(1 - df["random"]) / (1 - df["canary_final"])).groupby("workload").agg(
        rzz_total=("rzz_total", "first"),
        max_pair_reuse=("max_pair_reuse", "first"),
        gain_vs_random_mean=("gain_vs_random", "mean"),
        canary_final=("canary_final", "mean"),
        random=("random", "mean"),
        debias=("debias", "mean"),
        oracle=("oracle", "mean"),
    )
    tab.to_csv(DATA / "exp4" / "summary_table.csv")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["exp1", "exp2", "exp3", "exp4"]
    for t in targets:
        globals()[t]()
