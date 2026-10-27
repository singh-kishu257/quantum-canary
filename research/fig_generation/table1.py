"""Table I and supplementary tables S1-S5 from research/data/nisq_benchmark.csv, written as LaTeX to
research/figures/table1.tex and research/figures/tables_supp.tex, and as readable Markdown to research/figures/tables.md.
I: median |error| [95% CI] per arm; S1: full accuracy and calibration per arm; S2: compare() false alarms on
repeated checks of unchanged qubits; S3: robustness against coherent fraction and prior error (thirds);
S4: Canary chi2 flags and health in the NISQ regime; S5: QE failed fits along the sweep; S6: robustness, every
condition x parameter x tool x platform (Fig. 5).
"""
import re
import numpy as np
from scipy.stats import chi2, norm
import nisq_common as c

d = c.load()
MAIN = (("canary_x1", "Canary"), ("qe_equal", "QE, equal shots"), ("qe_ibm", "QE standard"))
FULL = MAIN[:1] + (("canary_x1_repeat", "Canary (repeat)"),) + MAIN[1:]
UNIT = {"T1_s": "\\%", "T2_s": "\\%", "delta_omega": "Hz", "epsilon_sx": "\\%"}
PROBE = {"T1_s": 2, "T2_s": 2, "delta_omega": 6, "epsilon_sx": 1}
NOMINAL = 0.0125
HEAD = {**c.LABEL, "delta_omega": r"$\Delta\omega/2\pi$"}


def g3(x):
    if not np.isfinite(x):
        return "--"
    if x == 0:
        return "0"
    s = f"{x:#.3g}".rstrip(".")
    return s if "e" not in s else f"{x:.2e}"


def g3t(x):
    return g3(x) if x < 1000 else f"{float(f'{x:.3g}'):,.0f}"


def med(x):
    m, lo, hi = c.median_ci(x)
    return f"{g3(m)} [{g3(lo)}, {g3(hi)}]"


def latex(caption, label, cols, header, rows, wide=True):
    env = "table*" if wide else "table"
    out = [f"\\begin{{{env}}}[!t]", "\\centering", "\\footnotesize", f"\\caption{{{caption}}}",
           f"\\label{{{label}}}", "\\begin{adjustbox}{max width=\\linewidth}", f"\\begin{{tabular}}{{{cols}}}",
           "\\hline", " & ".join(header) + " \\\\", "\\hline"]
    minus = lambda s: re.sub(r"(?<![\w$.\-])-(?=\d)", "$-$", s)
    for r in rows:
        out.append("\\hline" if r is None else " & ".join(map(minus, r)) + " \\\\")
    out += ["\\hline", "\\end{tabular}", "\\end{adjustbox}", f"\\end{{{env}}}", ""]
    return "\n".join(out)


PLAIN = ((r"$T_2^*/T_2 < 0.8$", "T₂*/T₂ < 0.8"), (r"$T_2^*/T_2$", "T₂*/T₂"), ("$T_1$", "T₁"), ("$T_2$", "T₂"),
         (r"$\Delta\omega/2\pi$", "Δω/2π"), (r"$\Delta\omega$", "Δω"), (r"$\varepsilon$", "ε"), (r"$\chi^2$", "χ²"),
         (r"$|z| \leq 1$", "|z| ≤ 1"), (r"$|z| \leq 2$", "|z| ≤ 2"), (r"$\sigma$", "σ"), ("$-$", "−"),
         ("$n$", "n"), ("$z$", "z"), (r"\texttt{compare()}", "`compare()`"), (r"\%", "%"), ("--", "–"),
         (r"2$\sigma$", "2σ"), (r"$\rho$", "ρ"), ("$<$", "<"), ("$p$", "p"))


def plain(s):
    for a, b in PLAIN:
        s = s.replace(a, b)
    return s


def markdown(title, cap, header, rows):
    cell = lambda s: plain(s).replace("|", r"\|")
    out = [f"## {title}", "", plain(cap), "", "| " + " | ".join(cell(h) for h in header) + " |",
           "|" + "---|" * len(header)]
    out += ["| " + " | ".join(cell(x) for x in r) + " |" for r in rows if r is not None]
    return "\n".join(out + [""])


def table_main():
    header = ["Platform", "Method", "Shots", "Circuit time (s)"] + \
             [f"{HEAD[p]} ({UNIT[p]})" for p in c.PARAMS] + ["Failed fits"]
    rows = []
    for arch, tag in c.ARCH:
        for arm, name in MAIN:
            g = d[(d.architecture == arch) & (d.arm == arm)]
            rows.append([tag, name, f"{int(g.shots_total.median()):,}", g3t(g.run_duration_s.median())] +
                        [med(c.sel(d, arch, arm, p).err) for p in c.PARAMS] +
                        [f"{int((~g.ok).sum())}/{len(g)}"])
        rows.append(None)
    rows.pop()
    cap = ("Median absolute error [95\\% CI] against the simulator's truth, 500 simulated qubits per platform "
           "(relative error for $T_1$, $T_2$, $\\varepsilon$; absolute error of $\\Delta\\omega/2\\pi$). "
           "Circuit time: median delays plus gate time per check, a lower bound (no repetition delay or readout). "
           "Failed fits: estimates QE did not return, excluded from the medians.")
    return header, rows, cap, latex(cap, "tab:nisq", "ll" + "r" * 2 + "c" * 4 + "r", header, rows)


def table_s1():
    header = ["Platform", "Method", "Parameter", "$n$", "Median", "90th pct.", "Median signed [95\\% CI]",
              "Cov. 68.3", "Cov. 95.4", "sd($z$)"]
    rows = []
    for arch, tag in c.ARCH:
        for arm, name in FULL:
            for p in c.PARAMS:
                g = c.sel(d, arch, arm, p)
                z = np.abs(g.z_score.values)
                rows.append([tag, name, f"{HEAD[p]} ({UNIT[p]})", f"{len(g)}", g3(np.median(g.err)),
                             g3(np.percentile(g.err, 90)), med(g.signed),
                             f"{100 * np.mean(z <= 1):.1f}", f"{100 * np.mean(z <= 2):.1f}",
                             g3(np.std(g.z_score.values))])
            rows.append(None)
    rows.pop()
    cap = ("Accuracy and error-bar calibration per arm (successful fits only; $n$ of 500). Coverage: \\% of "
           "qubits with $|z| \\leq 1$ and $|z| \\leq 2$, $z$ = (estimate $-$ truth)/$\\sigma$.")
    return header, rows, cap, latex(cap, "tab:s1", "lllrrrrrrr", header, rows)


def table_s2():
    alpha = 2 * norm.sf(2.0)
    header = ["Platform", "$n$"] + [c.LABEL[p] for p in c.PARAMS] + ["Any parameter"]
    rows = []
    for arch, tag in c.ARCH:
        a = d[(d.architecture == arch) & (d.arm == "canary_x1")].set_index(["instance_id", "parameter"])
        b = d[(d.architecture == arch) & (d.arm == "canary_x1_repeat")].set_index(["instance_id", "parameter"])
        ids = sorted(set(a.index.get_level_values(0)) & set(b.index.get_level_values(0)))
        flags = np.zeros((len(ids), len(c.PARAMS)), dtype=bool)
        for i, q in enumerate(ids):
            p_vals = []
            for p in c.PARAMS:
                ra, rb = a.loc[(q, p)], b.loc[(q, p)]
                infl_a = 1.0 if p == "T2_s" else np.sqrt(max(ra.chi2_dof, 1.0))
                infl_b = 1.0 if p == "T2_s" else np.sqrt(max(rb.chi2_dof, 1.0))
                z = (rb.estimate - ra.estimate) / np.hypot(ra.sigma * infl_a, rb.sigma * infl_b)
                p_vals.append(2 * norm.sf(abs(z)))
            flags[i] = c.holm(p_vals) < alpha
        n = len(ids)
        cell = lambda k: f"{100 * k / n:.1f} [{100 * c.clopper(k, n)[0]:.1f}, {100 * c.clopper(k, n)[1]:.1f}]"
        rows.append([tag, f"{n}"] + [cell(int(flags[:, j].sum())) for j in range(len(c.PARAMS))] +
                    [cell(int(flags.any(axis=1).sum()))])
    cap = ("False alarms of \\texttt{compare()} between two Canary checks of the same unchanged qubit (independent "
           "RB sequences and shot noise): \\% of qubits flagged [95\\% CI], with $\\chi^2$-inflated $\\sigma$ and "
           f"Holm correction over the four parameters at family level {100 * alpha:.2f}\\%.")
    return header, rows, cap, latex(cap, "tab:s2", "lrccccc", header, rows)


def table_s3():
    header = ["Platform", "Factor", "Third"] + [f"{HEAD[p]} ({UNIT[p]})" for p in c.PARAMS]
    rows = []
    for arch, tag in c.ARCH:
        for col, name in (("coherent_fraction", "Coherent fraction"), ("perturbation_frac", "Prior error")):
            q = d[(d.architecture == arch) & (d.arm == "canary_x1")].groupby("instance_id")[col].first()
            edges = np.quantile(q.values, [0, 1 / 3, 2 / 3, 1])
            for t in range(3):
                inside = q[(q >= edges[t]) & ((q < edges[t + 1]) if t < 2 else (q <= edges[t + 1]))].index
                cells = []
                for p in c.PARAMS:
                    ca = c.sel(d, arch, "canary_x1", p)
                    qe = c.sel(d, arch, "qe_ibm", p)
                    cells.append(f"{g3(np.median(ca[ca.instance_id.isin(inside)].err))} / "
                                 f"{g3(np.median(qe[qe.instance_id.isin(inside)].err))}")
                rows.append([tag, name, f"{g3(edges[t])}--{g3(edges[t + 1])}"] + cells)
        rows.append(None)
    rows.pop()
    cap = ("Robustness: median absolute error of Canary / QE standard in thirds of the coherent share of the gate "
           "error and of the calibration prior's relative error (about 167 qubits each).")
    return header, rows, cap, latex(cap, "tab:s3", "lllcccc", header, rows)


def table_s4():
    header = ["Platform", "Arm"] + [f"{c.LABEL[p]} $\\chi^2$ flag" for p in c.PARAMS] + ["GREEN", "YELLOW", "RED", "True $T_2^*/T_2 < 0.8$"]
    rows = []
    for arch, tag in c.ARCH:
        for arm, name in FULL[:2]:
            cells = []
            for p in c.PARAMS:
                x = c.sel(d, arch, arm, p).chi2_dof.values
                lim = chi2.ppf(1 - NOMINAL, PROBE[p]) / PROBE[p]
                cells.append(f"{100 * np.mean(x > lim):.1f}")
            h = d[(d.architecture == arch) & (d.arm == arm)].groupby("instance_id").health.first()
            r = d[(d.architecture == arch) & (d.arm == arm)].groupby("instance_id").t2_ramsey_over_echo.first()
            low = f"{100 * np.mean(r < 0.8):.1f}" if r.notna().all() else "--"
            rows.append([tag, name] + cells + [f"{100 * np.mean(h == s):.1f}" for s in ("GREEN", "YELLOW", "RED")] +
                        [low])
    cap = ("Canary's model checks in the NISQ regime: \\% of qubits over each probe's $\\chi^2$/dof limit "
           f"(nominal {100 * NOMINAL:.2f}\\% under a correct model), \\% per health status, and \\% of qubits whose "
           "simulated $T_2^*/T_2$ is below 0.8, the YELLOW decoupling rule (not stored for trapped ions).")
    return header, rows, cap, latex(cap, "tab:s4", "llccccrrrr", header, rows)


def table_s5():
    header = ["Platform", "Shots"] + [c.LABEL[p] for p in c.PARAMS]
    rows = []
    for arch, tag in c.ARCH:
        for b in c.BUDGETS:
            g = d[(d.architecture == arch) & (d.arm == f"qe_b{b}")]
            rows.append([tag, f"{int(g.shots_total.median()):,}"] +
                        [f"{int((~g[g.parameter == p].ok).sum())}/{int((g.parameter == p).sum())}" for p in c.PARAMS])
        rows.append(None)
    rows.pop()
    cap = "Failed QE fits along the shot sweep (Canary: none at any budget)."
    return header, rows, cap, latex(cap, "tab:s5", "lrcccc", header, rows, wide=False)


def table_s6():
    grid = c.robustness_grid(d, c.conditions(d))
    names = {"coherent": "Coherent share of gate error", "readout": "Readout error", "switching": "T1 switching rate",
             "ratio": "T2*/T2 (lower worse)", "detuning": "Detuning size", "prior": "Prior error"}
    header = ["Platform", "Condition", "Parameter", "Tool", "Median |error|, best to worst fifth",
              "2$\\sigma$ coverage, best to worst fifth (\\%)", "Spearman $\\rho$", "$p$ (Holm)"]
    rows = []
    for r in grid.itertuples():
        unit = "Hz" if r.parameter == "delta_omega" else "\\%"
        rows.append([r.platform, names[r.condition], HEAD[r.parameter], r.tool,
                     f"{g3(r.error_best)} to {g3(r.error_worst)} {unit}", f"{r.coverage_best:.0f} to {r.coverage_worst:.0f}",
                     f"{r.rho:+.2f}", f"{r.p_holm:.1e}" + (" *" if r.p_holm < 0.05 else "")])
    cap = ("Robustness, every condition that differs between the simulated qubits against every parameter and both "
           "tools (Canary; QE in IBM's standard configuration). Qubits in fifths of the condition; Spearman $\\rho$ "
           "of the condition (signed so that positive means the error grows as the condition worsens) with the "
           f"per-qubit |error|; $p$ Holm-corrected over all {len(grid)} tests (* $<$ 0.05). Fixed-strength "
           "mechanisms (T1 drift, frequency drift, motional heating, detuning during gates) are not varied.")
    tex = (latex(cap + " Part 1: superconducting qubits.", "tab:s6", "llllcccc", header, [r for r in rows if r[0] == "SC"])
           + "\\addtocounter{table}{-1}\n"
           + latex("(Continued.) Part 2: trapped ions.", "tab:s6b", "llllcccc", header, [r for r in rows if r[0] == "TI"]))
    return header, rows, cap, tex


assert not (~d[d.tool == "canary"].ok).any(), "Table S5 states Canary never failed"
md, supp = ["# NISQ benchmark tables (research/data/nisq_benchmark.csv)", ""], []
for title, fn in (("Table I", table_main), ("Table S1", table_s1), ("Table S2", table_s2), ("Table S3", table_s3),
                  ("Table S4", table_s4), ("Table S5", table_s5), ("Table S6", table_s6)):
    h, r, cap, tex = fn()
    md.append(markdown(title, cap, h, r))
    if title == "Table I":
        (c.OUT_DIR / "table1.tex").write_text(tex, encoding="utf-8")
    else:
        supp.append(tex)
(c.OUT_DIR / "tables_supp.tex").write_text("\n".join(supp), encoding="utf-8")
(c.OUT_DIR / "tables.md").write_text("\n".join(md), encoding="utf-8")
print("saved:", c.OUT_DIR / "table1.tex", c.OUT_DIR / "tables_supp.tex", c.OUT_DIR / "tables.md")
