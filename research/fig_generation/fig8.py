"""Fig. 8 -- IonQ Forte-1: Canary's gate error on ten ions measured in parallel (simultaneous randomized benchmarking),
runs collected by research/ionq_hardware.py (research/data/hardware/ionq_qpu.forte-1_<run>_<time>_ionq.json).
(a) epsilon per ion in every run, ion 0 measured alone, and IonQ's published median; (b) run-to-run consistency:
each run against the other runs of the same ion, in units of their combined error bar; (c) cost of one ten-ion check
in parallel and one ion at a time, from IonQ's rate card. RevTeX single column (3.40 in).
Usage: python fig8.py [RUN_GLOB]   (default: the QPU runs A-D; a relative glob is taken in research/data/hardware)
"""
import glob
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import nisq_common as c

HARDWARE = c.HERE.parent / "data" / "hardware"
RUN_GLOB = (sys.argv[1:2] or ["ionq_qpu.forte-1_[A-D]_*_ionq.json"])[0]
pattern = RUN_GLOB if os.path.isabs(RUN_GLOB) else str(HARDWARE / RUN_GLOB)
runs = [json.load(open(f)) for f in sorted(glob.glob(pattern))]
isolated = [json.load(open(f)) for f in sorted(glob.glob(str(HARDWARE / "ionq_qpu.forte-1_ion0_isolated_*_ionq.json")))]
if not runs:
    sys.exit(f"no runs match {pattern}")
ions = runs[0]["ions"]
assert all(r["ions"] == ions for r in runs), "runs measured different ions"
eps = np.array([[x["epsilon_sx"] for x in r["results"]] for r in runs])
sig = np.array([[x["epsilon_sx_sigma"] for x in r["results"]] for r in runs])
RUN_COL = [c.COL[k] for k in ("T1_s", "T2_s", "delta_omega", "epsilon_sx")]
prior = runs[0]["eps_prior"]

c.style()
W, H = c.COL_WIDTH, 4.8
fig = plt.figure(figsize=(W, H))
ax_a = fig.add_axes([0.155, 0.66, 0.825, 0.29])
ax_b = fig.add_axes([0.155, 0.35, 0.825, 0.16])
ax_c = fig.add_axes([0.30, 0.105, 0.33, 0.08])
x = np.arange(len(ions))
offset = lambda k: (k - (len(runs) - 1) / 2) * 0.14

for k in range(len(runs)):
    ax_a.errorbar(x + offset(k), 1e4 * eps[k], yerr=1e4 * sig[k], fmt="o", ms=3.0, mew=0, lw=0.7, capsize=0,
                  color=RUN_COL[k % 4], label=f"run {chr(65 + k)}", zorder=3)
for iso in isolated:
    for q, res in zip(iso["ions"], iso["results"]):
        if q in ions:
            ax_a.errorbar([ions.index(q) - 0.42], [1e4 * res["epsilon_sx"]], yerr=[1e4 * res["epsilon_sx_sigma"]],
                          fmt="D", ms=3.6, mfc="white", mec=c.INK, color=c.INK, lw=0.7, capsize=0, zorder=4,
                          label=f"ion {q} alone")
ax_a.axhline(1e4 * prior, color=c.GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
ax_a.text(len(ions) - 0.45, 1e4 * prior, "IonQ published median", ha="right", va="bottom", color=c.GREY,
          fontsize=c.SIZE - 1)
ax_a.set_xticks(x)
ax_a.set_xticklabels([str(q) for q in ions])
ax_a.set_xlim(-0.7, len(ions) - 0.4)
ax_a.set_ylim(0, 1.45e4 * max(prior, np.nanmax(eps + sig)))
ax_a.set_ylabel(r"$\varepsilon$ ($10^{-4}$)")
ax_a.set_xlabel("Ion")
ax_a.legend(loc="upper left", ncol=3, frameon=False, handletextpad=0.2, columnspacing=0.8, borderaxespad=0.1,
            fontsize=c.SIZE - 1)
fig.text(0.5675, 0.565, "(a) Gate error per ion, ions driven in parallel", ha="center", va="top")

z = np.full(eps.shape, np.nan)
if len(runs) > 1:
    for k in range(len(runs)):
        rest = [j for j in range(len(runs)) if j != k]
        w = 1 / sig[rest] ** 2
        mean, err = (w * eps[rest]).sum(0) / w.sum(0), 1 / np.sqrt(w.sum(0))
        z[k] = (eps[k] - mean) / np.hypot(sig[k], err)
ax_b.axhspan(-2, 2, color="#EFEFEF", lw=0, zorder=0)
ax_b.axhspan(-1, 1, color="#E2E2E2", lw=0, zorder=0)
for k in range(len(runs)):
    ax_b.plot(x + offset(k), z[k], "o", ms=3.0, mew=0, color=RUN_COL[k % 4], zorder=3)
zf = z[np.isfinite(z)]
inside = int(np.sum(np.abs(zf) <= 2))
lo, hi = c.clopper(inside, len(zf)) if len(zf) else (np.nan, np.nan)
ax_b.axhline(0, color=c.GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
ax_b.set_xticks(x)
ax_b.set_xticklabels([str(q) for q in ions])
ax_b.set_xlim(-0.7, len(ions) - 0.4)
lim = max(3.5, np.ceil(np.nanmax(np.abs(zf)) + 0.5)) if len(zf) else 3.5
ax_b.set_ylim(-lim, lim)
ax_b.set_ylabel("$z$ vs other runs")
ax_b.set_xlabel("Ion")
summary = (f"\n$|z| \\leq 2$: {inside}/{len(zf)} (95% CI {100 * lo:.0f}-{100 * hi:.0f}%; nominal 95%)" if len(zf)
           else " (needs two or more runs)")
fig.text(0.5675, 0.262, "(b) Run-to-run consistency" + summary, ha="center", va="top", linespacing=1.3)

est = [e for e in runs[0].get("ionq_estimates") or [] if e]
if est and est[0].get("cost") is not None:
    parallel = float(np.mean([sum(e["cost"] for e in r["ionq_estimates"]) for r in runs]))
    rate = est[0]["raw"][0]["rate_card"]["rates"][0]
    n_circ = len(est[0]["raw"])
    gates_per_ion = sum(raw["estimate_context"]["1q_gates"] for raw in est[0]["raw"]) / len(ions)
    per_circuit_alone = gates_per_ion / n_circ * runs[0]["shots"] * rate["cost_1q_gate"]
    sequential = len(ions) * n_circ * max(per_circuit_alone, rate["job_cost_minimum"])
    for i, (val, face) in enumerate(((parallel, c.INK), (sequential, "white"))):
        ax_c.barh(i, val / 1e3, color=face, edgecolor=c.INK, lw=0.7, height=0.6)
        ax_c.text(val / 1e3 + 0.3, i, f"\\${val:,.0f}", va="center", fontsize=c.SIZE - 1)
    ax_c.set_yticks([0, 1])
    ax_c.set_yticklabels(["in parallel", "one at a time"])
    ax_c.set_xlim(0, sequential / 1e3 * 1.45)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("k\\$ per check", labelpad=1)
    times = [t.get("execution_duration_ms") for r in runs for t in r.get("ionq_job_times") or []]
    note = (f"execution\n{np.sum(times) / len(runs) / 6e4:.0f} min per check\n(measured)" if times and all(times)
            else f"execution\n{est[0]['execution_time'] / 6e4:.0f} min per check\n(IonQ estimate)")
    fig.text(0.74, 0.125, note, ha="left", va="center", fontsize=c.SIZE - 1)
else:
    ax_c.text(0.5, 0.5, "no IonQ cost estimate (simulator)", transform=ax_c.transAxes, ha="center", va="center")
    ax_c.set_xticks([])
    ax_c.set_yticks([])
fig.text(0.5675, 0.004, "(c) Cost of one ten-ion check", ha="center", va="bottom")

c.save(fig, "fig8" if RUN_GLOB.startswith("ionq_qpu") else "fig8_test")
for k in range(len(runs)):
    print(f"run {chr(65 + k)}: median eps {np.median(eps[k]):.2e}, poor fits {sum(x['poor_fit'] for x in runs[k]['results'])}")
print(f"|z| <= 2: {inside}/{len(zf)}")
