from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "hardware"
OUT_DIR  = ROOT / "figures"

QUBIT_ORDER: list = []

REP_RAMSEY = 15
REP_ECHO   = 13
REP_GATE   = 9


def fwd_t1(tau, T1, p0g1=0.0, p1g0=0.0):
    p = np.exp(-np.asarray(tau) / T1)
    return p * (1 - p0g1) + (1 - p) * p1g0

def fwd_ramsey_x(tau, T2, dw):
    return 0.5 * (1 - np.exp(-np.asarray(tau) / T2) * np.cos(dw * np.asarray(tau)))

def fwd_ramsey_y(tau, T2, dw):
    return 0.5 * (1 - np.exp(-np.asarray(tau) / T2) * np.sin(dw * np.asarray(tau)))

def fwd_echo(tau, T2e, p0g1=0.0, p1g0=0.0):
    p = 0.5 * (1 - np.exp(-np.asarray(tau) / T2e))
    return p * (1 - p0g1) + (1 - p) * p1g0

def fwd_gate(N, eps, p0g1=0.0, p1g0=0.0):
    p = 0.5 * (1 + (1 - 2 * eps) ** (2 * np.asarray(N, float)))
    return p * (1 - p0g1) + (1 - p) * p1g0


def find_run(arg):
    if arg:
        p = Path(arg)
        p = p if p.is_absolute() else ROOT / p
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    fs = list(DATA_DIR.glob("run_A_*.json"))
    if not fs:
        raise FileNotFoundError(f"No run_A_*.json in {DATA_DIR}")
    return max(fs, key=lambda p: p.stat().st_mtime)

def load(p):
    with p.open() as f:
        return json.load(f)

def get_meta(q, run):
    return q.get("meta") or run.get("meta", {})

def qdata(run, qid):
    return next(q for q in run["qubits"] if q["qubit_id"] == qid)

def recs(q, probe):
    return [r for r in q["model_fit"]["records"] if r["probe"] == probe]

def ra(rs, key):
    return np.asarray([r[key] for r in rs], float)

def qcolor(qid):
    return plt.cm.tab10.colors[QUBIT_ORDER.index(qid) % 10]

def annotate_chi2(ax, chi2, x=0.97, y=0.97):
    ax.text(x, y, f"$\\chi^2/\\nu={chi2:.2f}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      alpha=0.88, ec="#cccccc"))

def data_eb(ax, rs_list, scale=1.0, **kw):
    x = ra(rs_list, "condition") * scale
    y = ra(rs_list, "p_meas")
    s = ra(rs_list, "sigma")
    o = np.argsort(x)
    ax.errorbar(x[o], y[o], yerr=s[o], **kw)


def panel_t1_all(ax, run):
    x_ref = np.linspace(0, 3.8, 400)
    ax.plot(x_ref, np.exp(-x_ref), color="black", lw=1.1,
            ls="--", alpha=0.40, label=r"$e^{-\tau/T_1}$", zorder=1)

    mean_chi2 = np.mean([q["result_full"]["t1_chi2_dof"]
                         for q in run["qubits"]])
    for q in run["qubits"]:
        r    = q["result_full"]
        cal  = q["calibration"]
        T1   = r["T1_s"]
        p0g1 = cal["p0_given_1"]
        p1g0 = cal["p1_given_0"]
        col  = qcolor(q["qubit_id"])
        rs_t1  = recs(q, "T1")
        tau    = ra(rs_t1, "condition")
        pmeas  = ra(rs_t1, "p_meas")
        sigma  = ra(rs_t1, "sigma")
        x_norm = tau / T1
        o      = np.argsort(x_norm)
        x_q    = np.linspace(0, max(x_norm) * 1.15, 200)
        ax.plot(x_q, fwd_t1(x_q * T1, T1, p0g1, p1g0),
                color=col, lw=1.0, alpha=0.50, zorder=2)
        ax.errorbar(x_norm[o], pmeas[o], yerr=sigma[o],
                    fmt="o", ms=4.5, color=col, capsize=2.0,
                    lw=0.8, label=f"Q{q['qubit_id']}", zorder=3)

    ax.set_xlim(0, 3.8)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel(r"Normalized delay $\tau/T_1$", fontsize=8)
    ax.set_ylabel(r"$P(|1\rangle)$", fontsize=8)
    ax.set_title(r"(a) $T_1$ decay — all 8 qubits", fontsize=8.5)
    ax.grid(alpha=0.18)
    annotate_chi2(ax, mean_chi2, x=0.97, y=0.38)

    handles = [Line2D([0],[0], color="k", lw=1.1, ls="--", alpha=0.4,
                      label=r"$e^{-\tau/T_1}$")]
    for q in run["qubits"]:
        handles.append(Line2D([0],[0], marker="o",
                               color=qcolor(q["qubit_id"]),
                               ms=4.5, ls="-", lw=1.0, alpha=0.9,
                               label=f"Q{q['qubit_id']}"))
    ax.legend(handles=handles, fontsize=5.8, frameon=False, ncol=3,
              loc="upper right", columnspacing=0.5,
              handlelength=1.2, handletextpad=0.4)


def panel_ramsey(ax, q, run):
    r    = q["result_full"]
    meta = get_meta(q, run)
    T2   = r["T2_ramsey_s"] or r["T2_s"]
    dw   = r["delta_omega"]
    qid  = q["qubit_id"]

    tau_max  = max(meta["ramsey_delays_s"]) * 1.20
    tau_plot = np.linspace(0, tau_max, 700)

    ax.plot(tau_plot * 1e6, fwd_ramsey_x(tau_plot, T2, dw),
            color="#2ca02c", lw=1.8, label=r"$X$ model", zorder=2)
    ax.plot(tau_plot * 1e6, fwd_ramsey_y(tau_plot, T2, dw),
            color="#ff7f0e", lw=1.8, label=r"$Y$ model", zorder=2)

    data_eb(ax, recs(q, "Ramsey-X"), scale=1e6,
            fmt="s", ms=5.5, color="#2ca02c", capsize=2.5,
            lw=1.0, label=r"$X$ data", zorder=3)
    data_eb(ax, recs(q, "Ramsey-Y"), scale=1e6,
            fmt="^", ms=5.5, color="#ff7f0e", capsize=2.5,
            lw=1.0, label=r"$Y$ data", zorder=3)

    ax.axhline(0.5, color="#aaaaaa", lw=0.7, ls="--")
    ax.set_xlim(0, tau_max * 1e6)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"Delay $\tau$ (µs)", fontsize=8)
    ax.set_ylabel(r"$P(|1\rangle)$", fontsize=8)
    ax.set_title(rf"(b) Ramsey $X/Y$ — Q{qid}", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower center",
              columnspacing=0.8, handlelength=1.2)
    ax.grid(alpha=0.18)
    annotate_chi2(ax, r["ramsey_chi2_dof"])

    dw_khz = dw / (2 * np.pi * 1e3)
    sig    = r["delta_omega_sigma"] / (2 * np.pi * 1e3)
    sig_s  = f"{sig:.2f}" if np.isfinite(sig) else r"\infty"
    ax.text(0.03, 0.97,
            f"$T_2^{{\\rm Ram}}={T2*1e6:.0f}\\pm{r['T2_ramsey_sigma_s']*1e6:.0f}$ µs\n"
            f"$\\Delta\\omega={dw_khz:.2f}\\pm{sig_s}$ kHz",
            transform=ax.transAxes, ha="left", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      alpha=0.88, ec="#cccccc"))


def panel_echo(ax, q, run):
    r    = q["result_full"]
    cal  = q["calibration"]
    meta = get_meta(q, run)
    T2e  = r["T2_echo_s"]
    T2r  = r["T2_ramsey_s"] or r["T2_s"]
    p0g1 = cal["p0_given_1"]
    p1g0 = cal["p1_given_0"]
    qid  = q["qubit_id"]

    tau_max  = max(meta["echo_delays_s"]) * 1.25
    tau_plot = np.linspace(0, tau_max, 400)

    ax.plot(tau_plot * 1e6, fwd_echo(tau_plot, T2e, p0g1, p1g0),
            color="#d62728", lw=1.8, label=r"Canary model ($T_2^{\rm echo}$)",
            zorder=2)
    ax.plot(tau_plot * 1e6, fwd_echo(tau_plot, T2r, p0g1, p1g0),
            color="#d62728", lw=1.0, ls=":", alpha=0.5,
            label=r"$T_2^{\rm Ram}$ reference", zorder=1)

    data_eb(ax, recs(q, "Echo"), scale=1e6,
            fmt="v", ms=5.5, color="#d62728", capsize=2.5,
            lw=1.0, label="IBM data", zorder=3)

    ax.axhline(0.5, color="#aaaaaa", lw=0.7, ls="--")
    ax.set_xlim(0, tau_max * 1e6)
    ax.set_ylim(-0.03, 0.62)
    ax.set_xlabel(r"Echo delay $\tau$ (µs)", fontsize=8)
    ax.set_ylabel(r"$P(|1\rangle)$", fontsize=8)
    ax.set_title(rf"(c) Hahn echo — Q{qid}", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.grid(alpha=0.18)
    annotate_chi2(ax, r["echo_chi2_dof"])
    ax.text(0.97, 0.05,
            f"$T_2^{{\\rm echo}}={T2e*1e6:.0f}$ µs\n"
            f"$T_2^{{\\rm Ram}}={T2r*1e6:.0f}$ µs",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      alpha=0.88, ec="#cccccc"))


def panel_gate(ax, q, run):
    r    = q["result_full"]
    cal  = q["calibration"]
    meta = get_meta(q, run)
    eps  = r["epsilon_sx"]
    p0g1 = cal["p0_given_1"]
    p1g0 = cal["p1_given_0"]
    qid  = q["qubit_id"]
    Nv   = meta["gate_rep_N"]

    N_plot = np.linspace(0, max(Nv) * 1.15, 400)
    ax.plot(N_plot, fwd_gate(N_plot, eps, p0g1, p1g0),
            color="#9467bd", lw=1.8, label="Canary model", zorder=2)

    data_eb(ax, recs(q, "Gate"), scale=1.0,
            fmt="D", ms=5.5, color="#9467bd", capsize=2.5,
            lw=1.0, label="IBM data", zorder=3)

    ax.axhline(0.5, color="#aaaaaa", lw=0.7, ls="--")
    ax.set_xlim(0, max(Nv) * 1.20)
    ax.set_ylim(0.40, 1.03)
    ax.set_xlabel(r"Gate pairs $N$  [SX$\cdot$SX$^\dagger$]", fontsize=8)
    ax.set_ylabel(r"$P(|0\rangle)$", fontsize=8)
    ax.set_title(rf"(d) Gate rep — Q{qid}", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.grid(alpha=0.18)
    annotate_chi2(ax, r["gate_chi2_dof"])
    ax.text(0.03, 0.05,
            f"$\\varepsilon_{{sx}}={eps:.2e}\\pm{r['epsilon_sx_sigma']:.1e}$",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      alpha=0.88, ec="#cccccc"))


def panel_t2_comparison(ax, run):
    qids  = [q["qubit_id"] for q in run["qubits"]]
    t2r   = [q["result_full"]["T2_ramsey_s"] * 1e6 for q in run["qubits"]]
    t2e   = [q["result_full"]["T2_echo_s"]   * 1e6 for q in run["qubits"]]
    t2r_e = [q["result_full"]["T2_ramsey_sigma_s"] * 1e6 for q in run["qubits"]]
    order = np.argsort(t2r)
    y     = np.arange(len(qids))

    for i, idx in enumerate(order):
        col = qcolor(qids[idx])
        ax.errorbar(t2r[idx], y[i] + 0.15, xerr=t2r_e[idx],
                    fmt="s", ms=5.5, color=col,
                    capsize=2.5, lw=1.0, elinewidth=1.0, zorder=3)
        ax.plot(t2e[idx], y[i] - 0.15,
                marker="D", ms=5.5, color=col, alpha=0.65,
                ls="none", zorder=3)
        ax.plot([t2r[idx], t2e[idx]], [y[i], y[i]],
                color=col, lw=0.9, alpha=0.40, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels([f"Q{qids[i]}" for i in order], fontsize=8)
    ax.set_xlabel(r"$T_2$ (µs)", fontsize=8)
    ax.set_title(r"(e) $T_2^{\rm Ram}$ (■) vs $T_2^{\rm echo}$ (◆)", fontsize=8.5)
    ax.set_xlim(left=0)
    ax.grid(axis="x", alpha=0.22)
    ax.tick_params(labelsize=8)
    els = [Line2D([0],[0], marker="s", color="#444", ms=5, ls="none",
                  label=r"$T_2^{\rm Ram}$ ±1σ"),
           Line2D([0],[0], marker="D", color="#444", ms=5, ls="none",
                  alpha=0.65, label=r"$T_2^{\rm echo}$")]
    ax.legend(handles=els, fontsize=6.5, frameon=False, loc="lower right")


def panel_chi2_heatmap(ax, run):
    qids   = [q["qubit_id"] for q in run["qubits"]]
    probes = [r"$T_1$", "Ramsey", "Echo", "Gate"]
    keys   = ["t1_chi2_dof", "ramsey_chi2_dof",
              "echo_chi2_dof", "gate_chi2_dof"]
    raw    = np.array([[q["result_full"][k] for k in keys]
                       for q in run["qubits"]])
    order  = np.argsort(np.mean(raw[:, :3], axis=1))
    raw    = raw[order]
    labels = [f"Q{qids[i]}" for i in order]

    cmap = LinearSegmentedColormap.from_list(
        "chi2", ["#f7fbff", "#c6dbef", "#fdae6b", "#d62728"], N=256)
    disp = np.clip(raw, 0, 15)
    im   = ax.imshow(disp, aspect="auto", cmap=cmap,
                     vmin=0, vmax=15, origin="upper")

    nq = len(qids)
    for i in range(nq):
        for j in range(4):
            v   = raw[i, j]
            txt = (f"{v:.2f}" if v < 10
                   else f"{v:.1f}" if v < 100
                   else f">{int(v // 100) * 100}")
            fg  = "white" if disp[i, j] > 10 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=6.5 if j < 3 else 6.0, color=fg)

    ax.set_xticks(range(4))
    ax.set_xticklabels(probes, fontsize=8)
    ax.set_yticks(range(nq))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(r"(f) $\chi^2/\nu$ per probe — all qubits", fontsize=8.5)
    ax.axvline(2.5, color="#444444", lw=1.2, ls="--", zorder=5)
    ax.text(3, nq - 0.45, "SX·SX†\ngate prob.",
            ha="center", va="bottom", fontsize=5.5, color="#333333")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\chi^2/\nu$ (cap 15)", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    cb.set_ticks([0, 5, 10, 15])


def make_figure(run: dict, out_stem: str):
    global QUBIT_ORDER
    QUBIT_ORDER = [q["qubit_id"] for q in run["qubits"]]

    q_ram  = qdata(run, REP_RAMSEY)
    q_echo = qdata(run, REP_ECHO)
    q_gate = qdata(run, REP_GATE)

    chi2_vals  = [q["model_fit"]["chi2_dof_joint"] for q in run["qubits"]]
    mean_joint = np.mean(chi2_vals)
    best_qid   = run["qubits"][int(np.argmin(chi2_vals))]["qubit_id"]

    fig = plt.figure(figsize=(7.16, 7.50))
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            height_ratios=[1.10, 1.00, 1.00],
                            hspace=0.52, wspace=0.38)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.91, bottom=0.07)

    ax_t1   = fig.add_subplot(gs[0, 0])
    ax_ram  = fig.add_subplot(gs[0, 1])
    ax_echo = fig.add_subplot(gs[1, 0])
    ax_gate = fig.add_subplot(gs[1, 1])
    ax_t2c  = fig.add_subplot(gs[2, 0])
    ax_chi  = fig.add_subplot(gs[2, 1])

    panel_t1_all(ax_t1, run)
    panel_ramsey(ax_ram, q_ram, run)
    panel_echo(ax_echo, q_echo, run)
    panel_gate(ax_gate, q_gate, run)
    panel_t2_comparison(ax_t2c, run)
    panel_chi2_heatmap(ax_chi, run)

    for ax in [ax_t1, ax_ram, ax_echo, ax_gate]:
        ax.tick_params(labelsize=7.5)

    fig.suptitle(
        r"Fig.~5 — IBM Heron r2 Hardware Validation  "
        r"(ibm\_fez, $N_q=8$, 9\,900 shots/qubit,  per-qubit scheduling,  "
        r"SX$\cdot$SX$^\dagger$ gate probe)"
        "\n"
        f"(a) all 8 qubits;  (b–d) representative qubits "
        f"Q{REP_RAMSEY}/Q{REP_ECHO}/Q{REP_GATE};  "
        f"mean joint $\\chi^2/\\nu={mean_joint:.2f}$,  "
        f"best Q{best_qid} at $\\chi^2/\\nu={min(chi2_vals):.2f}$",
        fontsize=7.6, y=0.968, linespacing=1.5)

    fig.text(0.5, 0.005,
             "Points: measured probabilities ± binomial s.e.   "
             r"Curves: Canary forward model at recovered $\hat{\theta}=(T_1,T_2,\Delta\omega,\varepsilon_{sx})$.",
             ha="center", fontsize=6.8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"{out_stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run",    default=None)
    ap.add_argument("--output", default="fig5_ibm_hardware_full")
    args = ap.parse_args()
    rp  = find_run(args.run)
    run = load(rp)
    print(f"Run  : {rp.name}  ({len(run['qubits'])} qubits)")
    make_figure(run, args.output)


if __name__ == "__main__":
    main()