"""Fig. 5 -- IBM Kingston, all 154 operational qubits: Canary against qiskit-experiments (QE) in IBM's standard
configuration (research/data/hardware/ibm_kingston_20261026T014902Z, sandwich C1 QE1 C2 QE2 C3).
(a-d) Bland-Altman of Canary (mean of C1 and C2) against QE1, with the band predicted in silico; QE default fits
and QE's own analysis re-run with the decay floor fixed at its physical value; (e) repeatability between repeats;
(f) epsilon against IBM's published sx error; (g) QE's raw RB decay and its two fits; (h) cost per run.
IEEE double column (7.16 in).
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import hw_common as h
import nisq_common as c

c.style()
d, refits, dw_range, cal, timing = h.load(), h.qe_refits(), h.canary_dw_range_hz(), h.ibm_calibration(), h.timing()
QE_C = c.TOOL["qe"]
PRED = {"T1_s": (-1.4, -26.5, 19.6), "T2_s": (-1.6, -22.6, 19.6), "delta_omega": (0.0981, -0.7562, 1.2396),
        "epsilon_sx": (2.3, -67.3, 86.0)}
rng = np.random.default_rng(5)


def floor_fixed(p, step):
    return refits[(refits.parameter == p) & (refits.step == step)].set_index("qubit").floor_fixed


def ba_panel(ax, p, scale, xlabel, ylim, xlog=False, xlim=None):
    v, _ = h.wide(d, p)
    can = v[["C1", "C2"]].mean(axis=1, skipna=False)
    pb, plo, phi = PRED[p]
    ax.axhspan(plo, phi, color="#E4E4E4", lw=0, zorder=0)
    ax.axhline(pb, color=c.GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
    sets = [("default", v.QE1, "white")]
    if p in ("T1_s", "epsilon_sx"):
        sets.append(("floor fixed", floor_fixed(p, "QE1"), QE_C))
    for name, q, face in sets:
        x, m = h.diff(can, q, p)
        if p == "delta_omega":
            x, m = x / 1e3, m / 1e3
            inside = q.reindex(x.index).abs() < dw_range.reindex(x.index)
            xin, xout = x[inside], x[~inside]
            ax.scatter(m[inside], xin, s=6, facecolor=face, edgecolor=QE_C, lw=0.5, zorder=3, rasterized=True)
            off = np.clip(xout, ylim[0] * 0.94, ylim[1] * 0.94)
            ax.scatter(np.clip(m[~inside], xlim[0] * 0.95, xlim[1] * 0.95), off, s=12, marker="x", color=c.INK,
                       lw=0.6, zorder=3, rasterized=True)
            s = h.bland_altman(xin)
        else:
            ax.scatter(m * scale, x, s=6, facecolor=face, edgecolor=QE_C, lw=0.5, zorder=3, rasterized=True)
            s = h.bland_altman(x)
        ls = "-" if face != "white" or len(sets) == 1 else (0, (1, 1))
        col = QE_C
        ax.axhline(s["bias"], color=col, lw=1.0, ls=ls, zorder=4)
        for lim in (s["loa_lo"], s["loa_hi"]):
            ax.axhline(lim, color=col, lw=0.6, ls=(0, (4, 2)) if ls == "-" else (0, (1, 1)), zorder=4)
    ax.set_ylim(*ylim)
    if xlog:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)


W, H = c.FULL_WIDTH, 5.6
fig = plt.figure(figsize=(W, H))
left, gap, w = 0.075, 0.075, 0.17
top_y, top_h, bot_y, bot_h = 0.585, 0.31, 0.145, 0.265
axes_top = [fig.add_axes([left + i * (w + gap), top_y, w, top_h]) for i in range(4)]

ba_panel(axes_top[0], "T1_s", 1e6, r"Mean $T_1$ ($\mu$s)", (-60, 90))
ba_panel(axes_top[1], "T2_s", 1e6, r"Mean $T_2$ ($\mu$s)", (-90, 90), xlog=True)
ba_panel(axes_top[2], "delta_omega", 1, r"Mean $\Delta\omega/2\pi$ (kHz)", (-6, 6), xlim=(-25, 25))
ba_panel(axes_top[3], "epsilon_sx", 1e4, r"Mean $\varepsilon$ ($10^{-4}$)", (-260, 110), xlog=True)
axes_top[0].set_ylabel("Canary $-$ QE (% of mean)")
axes_top[2].set_ylabel("Canary $-$ QE (kHz)", labelpad=1)
for ax, lab in zip(axes_top, ("(a) $T_1$", "(b) $T_2$", r"(c) $\Delta\omega$", r"(d) $\varepsilon$")):
    ax.text(0.5, -0.30, lab, transform=ax.transAxes, ha="center", va="top")
qe_dw = h.wide(d, "delta_omega")[0].QE1.dropna()
n_out = int((qe_dw.abs() >= dw_range.reindex(qe_dw.index)).sum())
handles = [mpl.lines.Line2D([], [], marker="o", ls="none", ms=3.2, mfc="white", mec=QE_C, label="vs QE default"),
           mpl.lines.Line2D([], [], marker="o", ls="none", ms=3.2, mfc=QE_C, mec=QE_C, label="vs QE floor fixed"),
           mpl.lines.Line2D([], [], color=QE_C, lw=1.0, label="bias"),
           mpl.lines.Line2D([], [], color=QE_C, lw=0.6, ls=(0, (4, 2)), label="95% LoA"),
           mpl.patches.Patch(color="#E4E4E4", label="95% LoA predicted in silico"),
           mpl.lines.Line2D([], [], marker="x", ls="none", ms=3.6, color=c.INK,
                            label=f"QE beyond Canary's $\\Delta\\omega$ range ({n_out} qubits, clipped)")]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 1.0), ncol=3, frameon=False,
           handletextpad=0.3, columnspacing=1.0, handlelength=1.4)

# (e) repeatability between repeats about 20 min apart
ax_e = fig.add_axes([left, bot_y, 0.17, bot_h])
ax_e2 = fig.add_axes([left + 0.205, bot_y, 0.04, bot_h])
groups = [("T1_s", "$T_1$"), ("T2_s", "$T_2$"), ("epsilon_sx", r"$\varepsilon$")]


def med_change(a, b, p):
    x, _ = h.diff(b, a, p)
    x = np.abs(x.values)
    boots = [np.median(rng.choice(x, len(x))) for _ in range(2000)]
    return np.median(x), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def bar(ax, x, val, col, face):
    m, lo, hi = val
    ax.bar(x, m, width=0.26, color=face, edgecolor=col, lw=0.7, zorder=2)
    ax.plot([x, x], [lo, hi], color=c.INK, lw=0.7, zorder=3)


for i, (p, lab) in enumerate(groups):
    v, _ = h.wide(d, p)
    bar(ax_e, i - 0.28, med_change(v.C1, v.C2, p), c.INK, c.INK)
    bar(ax_e, i, med_change(v.QE1, v.QE2, p), QE_C, "white")
    if p != "T2_s":
        bar(ax_e, i + 0.28, med_change(floor_fixed(p, "QE1"), floor_fixed(p, "QE2"), p), QE_C, QE_C)
ax_e.set_xticks(range(3))
ax_e.set_xticklabels([lab for _, lab in groups])
ax_e.set_xlim(-0.6, 2.6)
ax_e.set_ylabel("Median |change| (%)")
v, _ = h.wide(d, "delta_omega")
keep = v.QE1.abs() < dw_range.reindex(v.index)
vk = v[keep.values]
bar(ax_e2, -0.16, tuple(x / 1e3 for x in med_change(vk.C1, vk.C2, "delta_omega")), c.INK, c.INK)
bar(ax_e2, 0.16, tuple(x / 1e3 for x in med_change(vk.QE1, vk.QE2, "delta_omega")), QE_C, "white")
ax_e2.set_xticks([0])
ax_e2.set_xticklabels([r"$\Delta\omega$"])
ax_e2.set_xlim(-0.45, 0.45)
ax_e2.set_title("(kHz)", fontsize=c.SIZE, pad=3)
ax_e.legend(handles=[mpl.patches.Patch(facecolor=c.INK, label="Canary"),
                     mpl.patches.Patch(facecolor="white", edgecolor=QE_C, label="QE default"),
                     mpl.patches.Patch(facecolor=QE_C, label="QE floor fixed")],
            loc="upper left", frameon=False, handlelength=1.0, handletextpad=0.4, borderaxespad=0.1)
fig.text(left + 0.1225, bot_y - 0.36 * bot_h, "(e) Repeatability, about 20 min apart", ha="center", va="top")

# (f) epsilon against IBM's published sx error
ax_f = fig.add_axes([left + 0.33, bot_y, 0.14, bot_h])
v, _ = h.wide(d, "epsilon_sx")
series = [("Canary", v.C1, c.INK, c.INK), ("QE\ndefault", v.QE1, QE_C, "white"),
          ("QE floor\nfixed", floor_fixed("epsilon_sx", "QE1"), QE_C, QE_C)]
for i, (lab, s, col, face) in enumerate(series):
    r = (s / cal.epsilon_sx).dropna().values
    ax_f.scatter(i + rng.uniform(-0.18, 0.18, len(r)), r, s=4, facecolor=face, edgecolor=col, lw=0.4, alpha=0.8,
                 zorder=2, rasterized=True)
    m, lo, hi = c.median_ci(r)
    ax_f.plot([i - 0.25, i + 0.25], [m, m], color=c.INK, lw=1.3, zorder=4, solid_capstyle="butt")
ax_f.axhline(1, color=c.GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
ax_f.set_yscale("log")
ax_f.set_ylim(0.1, 30)
ax_f.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
ax_f.set_xticks(range(3))
ax_f.set_xticklabels([s[0] for s in series])
ax_f.set_xlim(-0.5, 2.5)
ax_f.set_ylabel(r"$\varepsilon$ / IBM's sx error")
ax_f.text(0.5, -0.36, r"(f) $\varepsilon$ vs IBM calibration", transform=ax_f.transAxes, ha="center", va="top")

# (g) QE's raw RB decay and its two fits
ax_g = fig.add_axes([left + 0.545, bot_y, 0.17, bot_h])
raw = h._qe_raw("QE1", "rb")
lengths = (10, 100, 500)
p0 = {L: [] for L in lengths}
for q, entries in raw.items():
    for L in lengths:
        n0 = sum(e["counts"]["0"] for e in entries if e["metadata"]["xval"] == L)
        n = sum(e["shots"] for e in entries if e["metadata"]["xval"] == L)
        p0[L].append(n0 / n)
med = np.array([np.median(p0[L]) for L in lengths])
q25 = np.array([np.percentile(p0[L], 25) for L in lengths])
q75 = np.array([np.percentile(p0[L], 75) for L in lengths])
m = np.array(lengths, dtype=float)
with __import__("warnings").catch_warnings():
    __import__("warnings").simplefilter("ignore")
    free, _ = curve_fit(lambda x, a, al, b: a * al ** x + b, m, med, p0=[0.5, 0.999, 0.5], maxfev=20000)
fixed, _ = curve_fit(lambda x, a, al: a * al ** x + 0.5, m, med, p0=[0.5, 0.999], maxfev=20000)
mm = np.geomspace(5, 3000, 400)
ax_g.plot(mm, free[0] * free[1] ** mm + free[2], color=QE_C, lw=1.0, ls=(0, (1, 1)),
          label=f"free floor ({free[2]:.2f})")
ax_g.plot(mm, fixed[0] * fixed[1] ** mm + 0.5, color=QE_C, lw=1.0, label="floor fixed, 0.5")
ax_g.errorbar(m, med, yerr=[med - q25, q75 - med], fmt="o", ms=3.2, color=c.INK, lw=0.8, capsize=0, zorder=4,
              label="raw median, IQR")
ax_g.axvspan(10, 500, color="#F1F1F1", lw=0, zorder=0)
ax_g.text(70, 0.52, "QE lengths", ha="center", va="bottom", color=c.GREY, fontsize=c.SIZE - 1)
ax_g.set_xscale("log")
ax_g.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
ax_g.set_xlim(5, 3000)
ax_g.set_ylim(0.5, 1.0)
ax_g.set_xlabel("Clifford length $m$")
ax_g.set_ylabel("Survival $P_0$")
ax_g.legend(loc="lower left", bbox_to_anchor=(0.0, 0.12), frameon=False, handlelength=1.4, borderaxespad=0.1)
ax_g.text(0.5, -0.36, "(g) QE's RB fits on its own data", transform=ax_g.transAxes, ha="center", va="top")

# (h) cost per run
ax_h = fig.add_axes([left + 0.79, bot_y, 0.13, bot_h])
metrics = [("QPU", "qpu_s"), ("Classical", "classical_s")]
for j, (lab, key) in enumerate(metrics):
    cv = [timing[s][key] for s in h.CANARY_STEPS]
    qv = [timing[s][key] for s in h.QE_STEPS]
    for k, (vals, col, face) in enumerate(((cv, c.INK, c.INK), (qv, QE_C, "white"))):
        x = j + (k - 0.5) * 0.36
        ax_h.bar(x, np.mean(vals), width=0.32, color=face, edgecolor=col, lw=0.7, zorder=2)
        ax_h.scatter([x] * len(vals), vals, s=5, color=c.GREY, zorder=3)
        ax_h.text(x, np.mean(vals) * 1.15, f"{np.mean(vals):.0f}", ha="center", va="bottom", fontsize=c.SIZE - 1)
ax_h.set_yscale("log")
ax_h.set_ylim(5, 8000)
ax_h.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
ax_h.set_xticks(range(len(metrics)))
ax_h.set_xticklabels([m_[0] for m_ in metrics])
ax_h.set_xlim(-0.6, 1.6)
ax_h.set_ylabel("Seconds per run")
ax_h.text(0.5, 0.97, "8,640 vs 92,000\nshots/qubit", transform=ax_h.transAxes, ha="center", va="top",
          fontsize=c.SIZE - 1)
ax_h.text(0.5, -0.36, "(h) Cost, 154 qubits", transform=ax_h.transAxes, ha="center", va="top")

c.save(fig, "fig5")
