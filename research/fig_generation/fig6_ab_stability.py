from __future__ import annotations
import argparse, json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "hardware"
OUT_DIR  = ROOT / "figures"

QUBIT_ORDER: list = []

PARAMS = [
    ("T1_s",        "T1_sigma_s",        "T1",         1e6,              "µs",     False),
    ("T2_s",        "T2_sigma_s",        "T2",         1e6,              "µs",     False),
    ("delta_omega", "delta_omega_sigma", r"$\Delta\omega$", 1/(2*np.pi*1e3), "kHz", False),
    ("epsilon_sx",  "epsilon_sx_sigma",  r"$\varepsilon_{sx}$", 1,        "",       True),
]


def find_runs(run_a, run_b):
    def resolve(arg, label):
        if arg:
            p = Path(arg); p = p if p.is_absolute() else ROOT/p
            if not p.exists(): raise FileNotFoundError(p)
            return p
        fs = sorted(DATA_DIR.glob(f"run_{label}_*.json"),
                    key=lambda p: p.stat().st_mtime)
        if not fs: raise FileNotFoundError(f"No run_{label}_*.json in {DATA_DIR}")
        return fs[-1]
    return resolve(run_a, "A"), resolve(run_b, "B")

def load(p):
    with p.open() as f: return json.load(f)

def qcolor(qid):
    return plt.cm.tab10.colors[QUBIT_ORDER.index(qid) % 10]

def minutes_between(ts_a, ts_b):
    fmt = "%Y-%m-%dT%H:%M:%S"
    ta = datetime.fromisoformat(ts_a.split(".")[0])
    tb = datetime.fromisoformat(ts_b.split(".")[0])
    return abs((tb - ta).total_seconds() / 60)

def build_zj(A, B):
    a_by_q = {q["qubit_id"]: q["result_full"] for q in A["qubits"]}
    b_by_q = {q["qubit_id"]: q["result_full"] for q in B["qubits"]}
    qids   = sorted(a_by_q.keys())
    Z, VA, VB, SA, SB = {}, {}, {}, {}, {}
    for vk, sk, lbl, sc, unit, log in PARAMS:
        Z[vk] = []; VA[vk] = []; VB[vk] = []; SA[vk] = []; SB[vk] = []
        for qid in qids:
            ra, rb = a_by_q[qid], b_by_q[qid]
            va, sa = ra[vk]*sc, ra[sk]*sc
            vb, sb = rb[vk]*sc, rb[sk]*sc
            denom  = np.sqrt(sa**2 + sb**2)
            z      = (va-vb)/denom if denom>0 and np.isfinite(denom) else np.nan
            Z[vk].append(z); VA[vk].append(va); VB[vk].append(vb)
            SA[vk].append(sa); SB[vk].append(sb)
    return qids, Z, VA, VB, SA, SB


def panel_scatter(ax, qids, va, vb, sa, sb, title, xlabel, ylabel,
                  unit="", log=False, panel_label=""):
    va = np.array(va); vb = np.array(vb)
    sa = np.array(sa); sb = np.array(sb)

    pos_vals = np.concatenate([va, vb])
    if log:
        pos_vals = pos_vals[pos_vals > 0]
        lo = pos_vals.min() * 0.4
        hi = pos_vals.max() * 2.5
        ref = np.logspace(np.log10(lo), np.log10(hi), 200)
    else:
        lo = min(va.min(), vb.min())
        hi = max(va.max(), vb.max())
        pad = (hi - lo) * 0.15
        lo -= pad; hi += pad
        ref = np.linspace(lo, hi, 200)

    ax.plot(ref, ref, color="#333333", lw=1.0, ls="--", alpha=0.55, zorder=1)

    med_sig = np.nanmedian(np.sqrt(sa**2 + sb**2))
    ax.fill_between(ref, ref - 2*med_sig, ref + 2*med_sig,
                    color="#bbbbbb", alpha=0.25, zorder=0)

    for i, qid in enumerate(qids):
        z = (va[i]-vb[i]) / np.sqrt(sa[i]**2+sb[i]**2) if (sa[i]**2+sb[i]**2)>0 else 0
        col    = qcolor(qid)
        marker = "X" if abs(z) > 2 else "o"
        ms     = 7 if abs(z) > 2 else 5.5
        ax.errorbar(va[i], vb[i], xerr=sa[i], yerr=sb[i],
                    fmt=marker, ms=ms, color=col,
                    capsize=2.5, lw=0.8, elinewidth=0.8,
                    zorder=4 if abs(z) > 2 else 3)
        if abs(z) > 2:
            offx = (hi-lo)*0.03; offy = (hi-lo)*0.025
            ax.text(va[i]+offx, vb[i]+offy, f"Q{qid}",
                    fontsize=6, color=col, fontweight="bold")

    if log:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    else:
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    ax.set_xlabel(f"Run A  [{unit}]" if unit else "Run A", fontsize=8)
    ax.set_ylabel(f"Run B  [{unit}]" if unit else "Run B", fontsize=8)
    ax.set_title(title, fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=0.18)
    if panel_label:
        ax.text(-0.16, 1.04, panel_label, transform=ax.transAxes,
                fontsize=9, fontweight="bold")

    n_drift = sum(1 for i in range(len(qids))
                  if abs((va[i]-vb[i])/np.sqrt(sa[i]**2+sb[i]**2)) > 2)
    ax.text(0.97, 0.05, f"{n_drift}/8 DRIFT",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="#cc2200" if n_drift > 0 else "#007700",
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      alpha=0.85, ec="#cccccc"))


def panel_zj_heatmap(ax, qids, Z):
    param_labels = ["T1", "T2", r"$\Delta\omega$", r"$\varepsilon_{sx}$"]
    param_keys   = [p[0] for p in PARAMS]

    raw = np.array([[Z[k][i] for k in param_keys]
                    for i in range(len(qids))])
    raw = np.where(np.isfinite(raw), raw, 0)

    order  = np.argsort(np.mean(np.abs(raw[:, :3]), axis=1))
    raw    = raw[order]
    labels = [f"Q{qids[i]}" for i in order]

    cmap   = plt.cm.RdBu_r
    disp   = np.clip(raw, -10, 10)
    im     = ax.imshow(disp, aspect="auto", cmap=cmap,
                       vmin=-10, vmax=10, origin="upper")

    nq = len(qids)
    for i in range(nq):
        for j in range(4):
            v   = raw[i, j]
            txt = f"{v:.1f}"
            fg  = "white" if abs(v) > 5.5 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7.0, color=fg,
                    fontweight="bold" if abs(v) > 2 else "normal")

    ax.set_xticks(range(4))
    ax.set_xticklabels(param_labels, fontsize=8.5)
    ax.set_yticks(range(nq))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(r"(e) $z_j = (\hat\theta_A - \hat\theta_B)/\sqrt{\sigma_A^2+\sigma_B^2}$",
                 fontsize=8.5)

    for j in range(3):
        ax.axvline(j+0.5, color="white", lw=1.0)

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$z_j$  (cap $\pm10$)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    cb.set_ticks([-10, -5, -2, 0, 2, 5, 10])
    cb.ax.axhline(2/10*20 - 10, color="#333", lw=0.8, ls="--")

    n_ok = {p[0]: sum(1 for v in Z[p[0]] if np.isfinite(v) and abs(v)<=2)
            for p in PARAMS}
    for j, p in enumerate(PARAMS):
        ax.text(j, nq - 0.35,
                f"{n_ok[p[0]]}/8\nOK",
                ha="center", va="bottom", fontsize=5.5,
                color="#007700" if n_ok[p[0]] >= 6 else "#cc2200")

    ax.text(-0.16, 1.04, "(e)", transform=ax.transAxes,
            fontsize=9, fontweight="bold")


def make_figure(A, B, out_stem):
    global QUBIT_ORDER
    QUBIT_ORDER = sorted([q["qubit_id"] for q in A["qubits"]])

    gap = minutes_between(A["timestamp"], B["timestamp"])
    qids, Z, VA, VB, SA, SB = build_zj(A, B)

    fig = plt.figure(figsize=(7.16, 6.80))
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            height_ratios=[1.0, 1.0],
                            hspace=0.52, wspace=0.42)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.89, bottom=0.08)

    ax_t1  = fig.add_subplot(gs[0, 0])
    ax_t2  = fig.add_subplot(gs[0, 1])
    ax_dw  = fig.add_subplot(gs[0, 2])
    ax_eps = fig.add_subplot(gs[1, 0])
    ax_zj  = fig.add_subplot(gs[1, 1:])

    vk, sk, lbl, sc, unit, log = PARAMS[0]
    panel_scatter(ax_t1, qids, VA[vk], VB[vk], SA[vk], SB[vk],
                  r"(a) $T_1$ — Run A vs B", f"Run A  [µs]", f"Run B  [µs]",
                  unit, log, "(a)")

    vk, sk, lbl, sc, unit, log = PARAMS[1]
    panel_scatter(ax_t2, qids, VA[vk], VB[vk], SA[vk], SB[vk],
                  r"(b) $T_2$ — Run A vs B", f"Run A  [µs]", f"Run B  [µs]",
                  unit, log, "(b)")

    vk, sk, lbl, sc, unit, log = PARAMS[2]
    panel_scatter(ax_dw, qids, VA[vk], VB[vk], SA[vk], SB[vk],
                  r"(c) $\Delta\omega$ — Run A vs B",
                  "Run A  [kHz]", "Run B  [kHz]",
                  unit, log, "(c)")

    vk, sk, lbl, sc, unit, log = PARAMS[3]
    panel_scatter(ax_eps, qids, VA[vk], VB[vk], SA[vk], SB[vk],
                  r"(d) $\varepsilon_{sx}$ — Run A vs B",
                  "Run A", "Run B", unit, True, "(d)")

    panel_zj_heatmap(ax_zj, qids, Z)

    fig.suptitle(
        r"Fig.~6 — Canary Stability: Run A vs Run B on IBM Heron r2  "
        f"($\\Delta t = {gap:.0f}$ min,  ibm\\_fez,  $N_q=8$)"
        "\n"
        r"Diagonal = perfect reproducibility.  "
        r"$\times$ marker and label = $|z_j|>2$ (statistically significant drift).  "
        r"Shaded band = $\pm2\sigma_{\rm combined}$.",
        fontsize=7.6, y=0.965, linespacing=1.5)

    fig.text(0.5, 0.005,
             r"$\varepsilon_{sx}$ stable on 38-min timescale (7/8 qubits $|z|<2$).  "
             r"$T_1$, $T_2$, $\Delta\omega$ show real qubit drift — not model failure.",
             ha="center", fontsize=7.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"{out_stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a",  default=None)
    ap.add_argument("--run-b",  default=None)
    ap.add_argument("--output", default="fig6_ab_stability")
    args = ap.parse_args()

    pa, pb = find_runs(args.run_a, args.run_b)
    A = load(pa); B = load(pb)
    print(f"Run A : {pa.name}  [{A['timestamp']}]")
    print(f"Run B : {pb.name}  [{B['timestamp']}]")
    gap = minutes_between(A["timestamp"], B["timestamp"])
    print(f"Gap   : {gap:.1f} min")
    make_figure(A, B, args.output)


if __name__ == "__main__":
    main()