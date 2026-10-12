"""Parity figures from the two benchmark CSVs in research/data.

  python research/fig_generation/make_figures.py

  ideal.png           Canary vs ground truth on the ideal simulator (ideal_parity.csv),
                      one panel per parameter.
  nisq_benchmark.png  Canary (left) vs qiskit-experiments (right) on the NISQ simulator
                      (nisq_benchmark.csv), one row per parameter, same axes per row.

Superconducting is blue, trapped ion orange. R2 is computed exactly as in
ideal_parity.py / benchmark.py (against y = x, linear values, converged instances
only), per architecture, and printed next to R2 of the prior alone: read R2 next to
the prior, never alone. Errors are median |relative error|.
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
OUT_DIR = HERE.parent / "figures"

ARCHS = (("superconducting", "Superconducting", "#2a78d6"),
         ("trapped_ion", "Trapped ion", "#eb6834"))
# parameter: (panel title, axis quantity, unit, scale factor to that unit, axis scale)
PARAMS = {
    "T1_s": (r"$T_1$", r"$T_1$", "s", 1.0, "log"),
    "T2_s": (r"$T_2$ (echo)", r"$T_2$", "s", 1.0, "log"),
    "delta_omega": (r"Frequency offset $\Delta\omega/2\pi$", r"$\Delta\omega/2\pi$", "Hz",
                    1.0 / (2.0 * np.pi), "symlog"),
    "epsilon_sx": (r"Gate error $\epsilon_{sx}$", r"$\epsilon_{sx}$", "", 1.0, "log"),
}
SYMLOG_LINTHRESH_HZ = 1.0

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Segoe UI", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 9,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.labelsize": 9,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.facecolor": SURFACE,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "savefig.dpi": 300,
})


def _r2(truth: np.ndarray, estimate: np.ndarray) -> float:
    """Coefficient of determination of estimate against truth (as in ideal_parity.py)."""
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    if ss_tot <= 0.0:
        return np.nan
    return 1.0 - float(np.sum((estimate - truth) ** 2)) / ss_tot


def _pct(x: float) -> str:
    return "n/a" if not np.isfinite(x) else (f"{100 * x:.2g}%" if x < 0.1 else f"{100 * x:.0f}%")


def _num(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.3f}"


def _series(df: pd.DataFrame, arch: str, param: str, method: str):
    """(truth, estimate, stats) of converged rows, in display units."""
    d = df[(df.architecture == arch) & (df.parameter == param)]
    scale = PARAMS[param][3]
    if method == "canary":
        tv, ev = d.true_value.to_numpy(float), d.canary_value.to_numpy(float)
        ok = d.success.astype(bool).to_numpy() & np.isfinite(tv) & np.isfinite(ev)
        re = d.relative_error.to_numpy(float)
        pv = d.prior_value.to_numpy(float)
        both = ok & np.isfinite(pv)
        r2_prior = _r2(tv[both], pv[both]) if both.sum() > 1 else np.nan
    else:
        tv, ev = d.qe_true_value.to_numpy(float), d.qe_value.to_numpy(float)
        ok = (d.qe_fit_status == "ok").to_numpy() & np.isfinite(tv) & np.isfinite(ev)
        re = d.qe_relative_error.to_numpy(float)
        r2_prior = np.nan
    fin_re = ok & np.isfinite(re)
    stats = {"n": len(d), "ok": int(ok.sum()),
             "r2": _r2(tv[ok], ev[ok]) if ok.sum() > 1 else np.nan,
             "r2_prior": r2_prior,
             "med_err": float(np.median(re[fin_re])) if fin_re.any() else np.nan}
    return tv[ok] * scale, ev[ok] * scale, stats


def _limits(values: np.ndarray, how: str) -> Optional[Tuple[float, float]]:
    v = values[np.isfinite(values)]
    if how == "log":
        v = v[v > 0]
        if not len(v):
            return None
        return float(v.min()) / 1.6, float(v.max()) * 1.6
    if not len(v):
        return None
    m = float(np.abs(v).max()) * 1.6
    return -m, m


def _style_axes(ax, param: str, lims, xlabel: str, ylabel: str) -> None:
    how = PARAMS[param][4]
    if how == "symlog":
        ax.set_xscale("symlog", linthresh=SYMLOG_LINTHRESH_HZ)
        ax.set_yscale("symlog", linthresh=SYMLOG_LINTHRESH_HZ)
    else:
        ax.set_xscale(how)
        ax.set_yscale(how)
    if lims is not None:
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.plot(lims, lims, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
    ax.tick_params(which="both", length=3, width=0.6)
    ax.tick_params(which="minor", length=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _label(name: str, s: Dict, with_prior: bool) -> str:
    head = f"{name}  (n = {s['ok']}" + (f" of {s['n']})" if s["ok"] != s["n"] else ")")
    body = f"R² {_num(s['r2'])}"
    if with_prior:
        body += f"  (prior {_num(s['r2_prior'])})"
    body += f"   median |err| {_pct(s['med_err'])}"
    return head + "\n" + body


def _panel(ax, df: pd.DataFrame, param: str, method: str, lims, ylabel_who: str,
           xlabel_suffix: str = "") -> int:
    """Scatter both architectures; returns how many converged points it drew."""
    _, qty, unit, _, how = PARAMS[param]
    u = f" ({unit})" if unit else ""
    drawn = 0
    handles, labels = [], []
    for arch, name, color in ARCHS:
        tv, ev, s = _series(df, arch, param, method)
        if how == "log":
            keep = (tv > 0) & (ev > 0)
            tv, ev = tv[keep], ev[keep]
        h = ax.scatter(tv, ev, s=16, color=color, alpha=0.8, edgecolors=SURFACE,
                       linewidths=0.5, zorder=3)
        drawn += len(tv)
        handles.append(h)
        labels.append(_label(name, s, with_prior=(method == "canary")))
    _style_axes(ax, param, lims, f"true {qty}{u}{xlabel_suffix}", f"{ylabel_who} {qty}{u}")
    if drawn:
        leg = ax.legend(handles, labels, loc="upper left", frameon=True, borderpad=0.6,
                        handletextpad=0.4, labelspacing=0.7, markerscale=1.4)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor(GRID)
        leg.get_frame().set_linewidth(0.6)
        for text in leg.get_texts():
            text.set_color(INK)
    return drawn


def _row_limits(df: pd.DataFrame, param: str, methods) -> Optional[Tuple[float, float]]:
    how = "log" if PARAMS[param][4] == "log" else "sym"
    vals = []
    for method in methods:
        for arch, _, _ in ARCHS:
            tv, ev, _ = _series(df, arch, param, method)
            vals += [tv, ev]
    return _limits(np.concatenate(vals) if vals else np.array([]), how)


def ideal_figure(csv: Path, out: Path) -> None:
    df = pd.read_csv(csv)
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 9.8), layout="constrained")
    for ax, param in zip(axes.flat, PARAMS):
        _panel(ax, df, param, "canary", _row_limits(df, param, ["canary"]), "Canary")
        ax.set_title(PARAMS[param][0])
    fig.suptitle("Canary vs ground truth, ideal simulator", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, -0.01, "Dashed line: y = x. R² against y = x per architecture over converged "
             "instances; prior = R² of the calibration prior with no measurement.",
             ha="left", va="top", fontsize=7.5, color=MUTED)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def nisq_figure(csv: Path, out: Path) -> None:
    df = pd.read_csv(csv)
    fig, axes = plt.subplots(4, 2, figsize=(9.6, 19.0), layout="constrained")
    qe_fail = df.qe_fit_status.fillna("").astype(str)
    for row, param in zip(axes, PARAMS):
        lims = _row_limits(df, param, ["canary", "qe"])
        _panel(row[0], df, param, "canary", lims, "Canary")
        row[0].set_title(f"{PARAMS[param][0]}: Canary")
        drawn = _panel(row[1], df, param, "qe", lims, "qiskit-experiments", ", QE window")
        row[1].set_title(f"{PARAMS[param][0]}: qiskit-experiments")
        if not drawn:
            status = qe_fail[df.parameter == param]
            top = status.value_counts().index[0] if len(status) else "no rows"
            n_bad = int((status == top).sum())
            reason = "\n".join(textwrap.wrap(top.replace(":", ": "), 42)[:3])
            row[1].text(0.5, 0.5, f"No qiskit-experiments fits in this CSV\n\n"
                        f"{n_bad} of {len(status)} rows:\n{reason}",
                        transform=row[1].transAxes, ha="center", va="center",
                        fontsize=8, color=INK_2, zorder=5, linespacing=1.4,
                        bbox=dict(boxstyle="round,pad=0.8", facecolor="white",
                                  edgecolor=GRID, linewidth=0.6))
    fig.suptitle("Canary vs qiskit-experiments, NISQ simulator", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, -0.005, "Dashed line: y = x. Both columns of a row share axes. Each method is "
             "scored against the truth averaged over its own measurement window.",
             ha="left", va="top", fontsize=7.5, color=MUTED)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ideal_figure(args.data_dir / "ideal_parity.csv", args.out_dir / "ideal.png")
    nisq_figure(args.data_dir / "nisq_benchmark.csv", args.out_dir / "nisq_benchmark.png")


if __name__ == "__main__":
    main()
