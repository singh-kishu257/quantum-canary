"""Fig. 4 -- accuracy against shot budget (research/data/nisq_benchmark.csv, sweep instances).
Median |error| with its 95% CI against total shots per qubit for Canary and for IBM's QE configuration
scaled to the same budgets; T1, T2, eps in %, delta-omega/2pi in Hz (SC) and mHz (TI).
QE failed fits are excluded (conservative for Canary; counts in Table I). IEEE double column (7.16 in).
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import nisq_common as c

c.style()
d = c.load()
N_SWEEP = d[d.arm == "canary_b1080"].instance_id.nunique()
sweep = d[d.instance_id < N_SWEEP]
assert set(d[d.arm.str.startswith("qe_b")].instance_id) == set(range(N_SWEEP))
CURVES = {"canary": ["canary_b1080", "canary_b2160", "canary_b4320", "canary_x1", "canary_b17280",
                     "canary_b34560", "canary_b69120", "canary_b138240"],
          "qe": [f"qe_b{b}" for b in c.BUDGETS]}
UNIT = {"T1_s": ("%", "%"), "T2_s": ("%", "%"), "delta_omega": ("Hz", "mHz"), "epsilon_sx": ("%", "%")}
SCALE = {"Hz": 1.0, "mHz": 1e3, "%": 1.0}


def curve(arch, arms, p, scale):
    rows = []
    for arm in arms:
        g = c.sel(sweep, arch, arm, p)
        rows.append((g.shots_total.median(), *(scale * np.array(c.median_ci(g.err)))))
    return np.array(rows)


W, H = c.FULL_WIDTH, 3.35
fig, axes = plt.subplots(2, 4, figsize=(W, H))
fig.subplots_adjust(left=0.075, right=0.99, top=0.865, bottom=0.135, wspace=0.42, hspace=0.42)
for r, (arch, tag) in enumerate(c.ARCH):
    for k, p in enumerate(c.PARAMS):
        ax = axes[r, k]
        unit = UNIT[p][r]
        for tool, marker in (("canary", "o"), ("qe", "s")):
            v = curve(arch, CURVES[tool], p, SCALE[unit])
            col = c.TOOL[tool]
            ax.fill_between(v[:, 0], v[:, 2], v[:, 3], color=col, alpha=0.18, lw=0, zorder=1)
            ax.plot(v[:, 0], v[:, 1], color=col, lw=1.0, marker=marker, ms=2.8, mew=0, zorder=3)
        dflt = curve(arch, ["canary_x1"], p, SCALE[unit])[0]
        ax.plot(dflt[0], dflt[1], marker="o", ms=5.0, mfc="none", mec=c.TOOL["canary"], mew=0.8, zorder=4)
        std = curve(arch, ["qe_ibm"], p, SCALE[unit])[0]
        ax.plot(std[0], std[1], marker="D", ms=4.2, mfc="white", mec=c.TOOL["qe"], mew=0.9, zorder=4)
        n = np.array(c.BUDGETS, dtype=float)
        ax.plot(n, dflt[1] * np.sqrt(dflt[0] / n), color=c.GREY, lw=0.6, ls=(0, (3, 2)), zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(800, 2.0e5)
        ax.set_xticks([1e3, 1e4, 1e5])
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
        lo, hi = ax.get_ylim()
        if hi / lo < 12:
            mid = np.sqrt(lo * hi)
            ax.set_ylim(mid / 3.6, mid * 3.6)
        ax.set_title(f"{tag}, {c.LABEL[p]}  ({unit})", fontsize=c.SIZE, pad=3)
        if r == 1:
            ax.set_xlabel("Total shots per qubit")
    axes[r, 0].set_ylabel("Median |error|")
handles = [mpl.lines.Line2D([], [], color=c.TOOL["canary"], marker="o", ms=2.8, mew=0, lw=1.0, label="Canary"),
           mpl.lines.Line2D([], [], color=c.TOOL["qe"], marker="s", ms=2.8, mew=0, lw=1.0,
                            label="QE (IBM design, scaled)"),
           mpl.lines.Line2D([], [], ls="none", marker="o", ms=5.0, mfc="none", mec=c.TOOL["canary"], mew=0.8,
                            label="Canary default"),
           mpl.lines.Line2D([], [], ls="none", marker="D", ms=4.2, mfc="white", mec=c.TOOL["qe"], mew=0.9,
                            label="QE standard (92,000)"),
           mpl.lines.Line2D([], [], color=c.GREY, lw=0.6, ls=(0, (3, 2)), label="$\\propto 1/\\sqrt{\\mathrm{shots}}$"),
           mpl.patches.Patch(color=c.GREY, alpha=0.3, lw=0, label="95% CI")]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 1.0), ncol=6, frameon=False,
           handletextpad=0.4, columnspacing=1.1, handlelength=1.6)
c.save(fig, "fig4")

for arch, tag in c.ARCH:
    for p in c.PARAMS:
        for tool in ("canary", "qe"):
            v = curve(arch, CURVES[tool], p, 1.0)
            print(tag, p, tool, " ".join(f"{m:.3g}" for m in v[:, 1]))
