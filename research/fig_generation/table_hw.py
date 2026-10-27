"""Hardware tables from the IBM Kingston full-chip sandwich C1 QE1 C2 QE2 C3 (hw_common.RUN), written as readable
Markdown (research/figures/tables_hw.md) and LaTeX (research/figures/tables_hw.tex).
H1 Bland-Altman of Canary (mean of the two checks around each QE run) against QE; H2 repeatability and whether it
explains the Canary-QE spread; H3 drift, error-bar and chi2 tests; H4 epsilon against IBM's calibration; H5 the
delta-omega range of Canary's Ramsey probe; H6 cost per run; H7 agreement of the two tools' changes (Fig. 7); H8
confirmation of Canary's drift flags by QE in the same window; H9 health verdicts and their reasons.
"""
import re
import numpy as np
import pandas as pd
from scipy.stats import chi2
import hw_common as h
from nisq_common import OUT_DIR

d, refits, dw_range = h.load(), h.qe_refits(), h.canary_dw_range_hz()
cal = h.ibm_calibration()
PARAMS = ("T1_s", "T2_s", "delta_omega", "epsilon_sx")
NAME = {"T1_s": "T1", "T2_s": "T2", "delta_omega": "Δω/2π", "epsilon_sx": "ε"}
UNIT = {"T1_s": "%", "T2_s": "%", "delta_omega": "Hz", "epsilon_sx": "%"}
PREDICTED = h.predicted()
NOMINAL = 0.0125


def f(x, digits=3):
    if not np.isfinite(x):
        return "–"
    return f"{x:,.0f}" if abs(x) >= 10 ** digits else f"{x:#.{digits}g}".rstrip(".")


def qe(p, step, kind):
    if kind == "floor fixed":
        return refits[(refits.parameter == p) & (refits.step == step)].set_index("qubit").floor_fixed
    return h.wide(d, p)[0][step]


def comparators(p):
    out = [("QE default", "default", None)]
    if p in ("T1_s", "epsilon_sx"):
        out.append(("QE, floor fixed", "floor fixed", None))
    if p == "delta_omega":
        out.append(("QE default, within Canary range", "default", "in_range"))
    return out


def table_h1():
    header = ["Parameter", "Comparator", "Sandwich", "n", "Bias [95% CI]", "95% LoA (±CI)", "2.5–97.5th pct.",
              "Predicted (in silico)"]
    rows = []
    for p in PARAMS:
        v, _ = h.wide(d, p)
        for label, kind, sel in comparators(p):
            for step, (a, b) in h.SANDWICH.items():
                q = qe(p, step, kind)
                if sel == "in_range":
                    q = q[q.abs() < dw_range.reindex(q.index)]
                x, _ = h.diff(v[[a, b]].mean(axis=1, skipna=False), q, p)
                s = h.bland_altman(x)
                pb, plo, phi = PREDICTED[p]
                rows.append([f"{NAME[p]} ({UNIT[p]})", label, f"mean({a},{b}) − {step}", str(s["n"]),
                             f"{f(s['bias'])} [{f(s['bias'] - s['bias_ci'])}, {f(s['bias'] + s['bias_ci'])}]",
                             f"{f(s['loa_lo'])} to {f(s['loa_hi'])} (±{f(s['loa_ci'], 2)})",
                             f"{f(s['p2_5'])} to {f(s['p97_5'])}", f"{f(pb)}; {f(plo)} to {f(phi)}"])
    cap = ("Bland–Altman agreement of Canary (mean of the two checks either side of each QE run) with "
           "qiskit-experiments in IBM's standard configuration on all operational qubits of ibm_kingston. "
           "Differences are Canary − QE, relative to the pair mean (Δω: in Hz). 'Floor fixed': qiskit-experiments' "
           "own analysis re-run on its raw counts with the decay floor fixed at its physical value (RB asymptote 1/2; "
           "T1 offset at IBM's P(1|0)); the default re-run reproduces every saved value. 'Within Canary range': qubits "
           "whose QE detuning lies inside Canary's unambiguous Ramsey range (Table H5). Predicted: mean and 2.5–97.5th "
           "percentiles of Canary − QE standard in the NISQ simulation.")
    return "Table H1", cap, header, rows


def table_h2():
    header = ["Parameter", "QE analysis", "Robust SD, C1→C2", "Robust SD, QE1→QE2", "Predicted SD of Canary − QE",
              "Observed SD of Canary − QE", "Median |change| C1→C2", "Median |change| QE1→QE2"]
    rows = []
    for p in PARAMS:
        v, _ = h.wide(d, p)
        for label, kind, sel in comparators(p):
            if sel is None and p == "delta_omega":
                continue
            q1, q2 = qe(p, "QE1", kind), qe(p, "QE2", kind)
            cc, _ = h.diff(v.C2, v.C1, p)
            qq, _ = h.diff(q2, q1, p)
            ba, _ = h.diff(v[["C1", "C2"]].mean(axis=1, skipna=False), q1, p)
            if sel == "in_range":
                keep = q1.index[(q1.abs() < dw_range.reindex(q1.index)).values]
                cc, qq, ba = cc[cc.index.isin(keep)], qq[qq.index.isin(keep)], ba[ba.index.isin(keep)]
            pred = np.sqrt(h.robust_sd(cc) ** 2 / 4 + h.robust_sd(qq) ** 2 / 2)
            rows.append([f"{NAME[p]} ({UNIT[p]})", label, f(h.robust_sd(cc)), f(h.robust_sd(qq)), f(pred),
                         f(h.robust_sd(ba)), f(np.median(np.abs(cc))), f(np.median(np.abs(qq)))])
    cap = ("Repeatability of each method between repeats about 20 minutes apart (real drift included) and whether it "
           "explains the Canary − QE spread: if the two methods disagree only through their own noise and drift, the SD "
           "of mean(C1,C2) − QE1 is about √(SD²(C1→C2)/4 + SD²(QE1→QE2)/2). Robust SD = 1.4826 × MAD.")
    return "Table H2", cap, header, rows


def table_h3():
    header = ["Parameter", "Changes C1→C2 > 3σ", "QE1 between C1 and C2: large / small change",
              "Mean change C1→C2 [95% CI]", "|z| ≤ 2, C1→C2 / C2→C3 (nominal 95.4%)", "χ² over limit, C1/C2/C3",
              "χ² over limit, simulated (Table S4)", "Median |Canary − QE|: fits pass / over limit"]
    import nisq_common as nc
    nisq = nc.load()
    sim = {p: 100 * np.mean(nc.sel(nisq, "superconducting", "canary_x1", p).chi2_dof.values
                            > chi2.ppf(1 - NOMINAL, h.PROBE_DOF[p]) / h.PROBE_DOF[p]) for p in PARAMS}
    rows = []
    for p in PARAMS:
        v, s = h.wide(d, p)
        z = (v.C2 - v.C1) / np.hypot(s.C1, s.C2)
        big = z.abs() > 3
        lo, hi = np.minimum(v.C1, v.C2), np.maximum(v.C1, v.C2)
        between = (v.QE1 >= lo) & (v.QE1 <= hi) & v.QE1.notna()
        nb, ns = int((big & v.QE1.notna()).sum()), int((~big & z.notna() & v.QE1.notna()).sum())
        kb, ks = int((between & big).sum()), int((between & ~big & z.notna()).sum())
        x, _ = h.diff(v.C2, v.C1, p)
        z23 = ((v.C3 - v.C2) / np.hypot(s.C2, s.C3)).dropna()
        lim = chi2.ppf(1 - NOMINAL, h.PROBE_DOF[p]) / h.PROBE_DOF[p]
        c = d[(d.parameter == p) & (d.tool == "canary")].pivot(index="qubit", columns="step", values="chi2_dof")
        flags = " / ".join(f"{100 * np.mean(c[k] > lim):.0f}%" for k in h.CANARY_STEPS)
        ba, _ = h.diff(v[["C1", "C2"]].mean(axis=1, skipna=False), qe(p, "QE1", "floor fixed" if p in ("T1_s", "epsilon_sx") else "default"), p)
        flagged = (c.C1 > lim).reindex(ba.index)
        rows.append([f"{NAME[p]} ({UNIT[p]})", f"{int(big.sum())}/{int(z.notna().sum())}",
                     f"{kb}/{nb} ({100 * kb / max(nb, 1):.0f}%) / {ks}/{ns} ({100 * ks / max(ns, 1):.0f}%)",
                     f"{f(x.mean())} [{f(x.mean() - 1.96 * x.std() / np.sqrt(len(x)))}, {f(x.mean() + 1.96 * x.std() / np.sqrt(len(x)))}]",
                     f"{100 * np.mean(z.dropna().abs() <= 2):.1f}% / {100 * np.mean(z23.abs() <= 2):.1f}%", flags, f"{sim[p]:.1f}%",
                     f"{f(np.median(np.abs(ba[~flagged])))} / {f(np.median(np.abs(ba[flagged])))}"])
    cap = ("Drift, error-bar and model-fit tests. Drift: for qubits whose two Canary checks differ by more than three "
           "combined standard errors, a drifting qubit puts QE1 (measured in between) between them. |z| between repeats "
           "tests Canary's error bars where drift is small (ε) and shows drift where it is not (T1, Δω). χ² limits as in "
           f"the simulation (nominal {100 * NOMINAL:.2f}% over limit for a correct model). Last column: agreement with QE "
           "(floor fixed for T1 and ε) on qubits whose Canary fit passes vs is over its χ² limit in C1.")
    return "Table H3", cap, header, rows


def table_h4():
    header = ["Method", "Median ε", "Median ratio to IBM [IQR]", "Bias vs IBM, % [95% CI]", "95% LoA vs IBM (%)"]
    rows = []
    ibm = cal.epsilon_sx
    v, _ = h.wide(d, "epsilon_sx")
    for label, series in (("Canary C1", v.C1), ("Canary C2", v.C2), ("QE1 default", v.QE1),
                          ("QE1 floor fixed", qe("epsilon_sx", "QE1", "floor fixed")),
                          ("QE2 floor fixed", qe("epsilon_sx", "QE2", "floor fixed"))):
        x, _ = h.diff(series, ibm, "epsilon_sx")
        ratio = (series / ibm).dropna()
        s = h.bland_altman(x)
        rows.append([label, f"{np.median(series.dropna()):.3e}",
                     f"{np.median(ratio):.2f} [{np.percentile(ratio, 25):.2f}, {np.percentile(ratio, 75):.2f}]",
                     f"{f(s['bias'])} [{f(s['bias'] - s['bias_ci'])}, {f(s['bias'] + s['bias_ci'])}]",
                     f"{f(s['loa_lo'])} to {f(s['loa_hi'])}"])
    rows.append(["IBM calibration (sx gate error)", f"{np.median(ibm.dropna()):.3e}", "1", "–", "–"])
    cap = ("Gate error against IBM's own published sx gate error (IBM's randomized benchmarking, measured about 12 h "
           "before the run; not independent of drift since then).")
    return "Table H4", cap, header, rows


def table_h5():
    v, _ = h.wide(d, "delta_omega")
    header = ["Quantity", "Value"]
    out = []
    q1 = v.QE1.dropna()
    beyond = q1[q1.abs() >= dw_range.reindex(q1.index)]
    clipped = sum(int(np.isclose(abs(v[s][q]), dw_range[q], rtol=1e-3)) for s in h.CANARY_STEPS for q in v.index
                  if np.isfinite(v[s][q]))
    out.append(["Canary unambiguous range |Δω/2π| (median, min–max over qubits)",
                f"{f(np.median(dw_range))} Hz ({f(dw_range.min())}–{f(dw_range.max())} Hz)"])
    out.append(["Qubits whose QE1 detuning lies beyond Canary's range", f"{len(beyond)}/{len(q1)} ({100 * len(beyond) / len(q1):.0f}%)"])
    can = v[["C1", "C2"]].mean(axis=1)
    out.append(["of these, Canary's value off by more than 3 kHz", str(int(((can - v.QE1).abs() > 3000).reindex(beyond.index).sum()))])
    out.append(["Canary values at the range limit (any check)", str(clipped)])
    hl = d[(d.tool == "canary") & (d.parameter == "delta_omega")].pivot(index="qubit", columns="step", values="health")
    near = {"C1": "QE1", "C2": "QE2", "C3": "QE2"}
    green = [(s, q) for s in h.CANARY_STEPS for q in beyond.index if hl[s].get(q) == "GREEN"]
    err = [abs(v[s][q] - v[near[s]][q]) for s, q in green]
    wrong = [e for e in err if e > 3000]
    out.append(["GREEN reports on these qubits (C1–C3); of them off by more than 3 kHz from the nearest QE run",
                f"{len(green)}/{3 * len(beyond)}; {len(wrong)}" + (f" ({f(min(wrong) / 1e3)}–{f(max(wrong) / 1e3)} kHz)"
                                                                    if wrong else "")])
    out.append(["Median |Canary − QE1| within range / beyond range",
                f"{f(np.median(np.abs((can - v.QE1).drop(beyond.index).dropna())))} Hz / "
                f"{f(np.median(np.abs((can - v.QE1).reindex(beyond.index))))} Hz"])
    cap = ("Δω outside Canary's design range. Canary's Ramsey probe resolves |Δω| < 0.9π/τ_min (τ_min: its shortest "
           "Ramsey delay, set by the T2 prior); beyond it the estimate aliases by multiples of 1/τ_min or stops at the "
           "limit. QE's Ramsey (delays 1–49 μs with a 100 kHz offset) resolves about ±150 kHz.")
    return "Table H5", cap, header, out


def table_h6():
    t = h.timing()
    header = ["Run", "Shots per qubit", "Jobs", "QPU (s)", "Classical (s)", "IBM-side (s)", "Queue (s)", "Wall (s)",
              "CPU (s)"]
    rows = []
    for s in ("C1", "QE1", "C2", "QE2", "C3"):
        x = t[s]
        rows.append([s, "8,640" if s.startswith("C") else "92,000", str(len(x["job_ids"])), f(x["qpu_s"]),
                     f(x["classical_s"]), f(x["ibm_s"]), f(x["queue_s"], 2), f(x["wall_s"]), f(x["cpu_s"])])
    cap = ("Cost of each run on 154 qubits. QPU: IBM's measured usage. IBM-side: merged created-to-finished time of "
           "the run's jobs (queue included); classical = wall − IBM-side (circuit generation, transpilation, upload, "
           "download, analysis on one laptop). Queue time depends on IBM's load and is not a property of the method.")
    return "Table H6", cap, header, rows


def drift_table():
    """hw_common.drift() with the time-matched z of each tool's change and its flag (Holm over every qubit and
    parameter, per tool)."""
    from scipy.stats import norm
    from nisq_common import holm
    dr = h.drift()
    dr["zc"], dr["zq"] = dr.canary_change / dr.canary_change_sigma, dr.qe_change / dr.qe_change_sigma
    dr["fc"] = holm(2 * norm.sf(dr.zc.abs().values)) < 0.05
    dr["fq"] = holm(2 * norm.sf(dr.zq.abs().values)) < 0.05
    return dr


def table_h7():
    from scipy.stats import spearmanr
    from nisq_common import holm
    dr = drift_table()
    groups = [(p, dr[dr.parameter == p]) for p in PARAMS]
    rho = [spearmanr(g.canary_change, g.qe_change) for _, g in groups]
    ph = holm([r.pvalue for r in rho])
    header = ["Parameter", "n", "Changed over C1–C3 (Canary)", "Spearman ρ [95% CI]", "p (Holm)",
              "Disagreement SD (robust)", "|z| ≤ 2 (%)"]
    rows = []
    for (p, g), r, pv in zip(groups, rho, ph):
        n = len(g)
        lo, hi = np.tanh(np.arctanh(r.statistic) + np.array([-1.96, 1.96]) * np.sqrt(1.06 / (n - 3)))
        z = (g.canary_change - g.qe_change) / np.hypot(g.canary_change_sigma, g.qe_change_sigma)
        rows.append([f"{NAME[p]}", str(n), f"{int((g.canary_p_holm < 0.05).sum())}/{n}",
                     f"{r.statistic:+.2f} [{lo:+.2f}, {hi:+.2f}]", f"{pv:.1e}",
                     f"{np.std(z, ddof=1):.2f} ({h.robust_sd(z):.2f})", f"{100 * np.mean(np.abs(z) <= 2):.0f}"])
    cap = ("Drift on ibm_kingston over the sandwich (Fig. 7). Changed: Canary's three checks differ beyond their error "
           "bars (Cochran's Q, 2 dof, on the error bars canary.compare uses; Holm over every qubit and parameter). "
           "Spearman ρ: per qubit, QE's change QE2 − QE1 against Canary's change over the same interval (its "
           "trajectory interpolated linearly to the times of QE's jobs); 95% CI from Fieller's standard error, Holm "
           "over the four parameters. Disagreement: (Canary change − QE change) / combined standard error; SD 1 and "
           "95% within ±2 if both tools see the same change and their error bars are right. QE: floor-fixed refits "
           "for T1 and ε; Δω only where both QE values lie inside Canary's range.")
    return "Table H7", cap, header, rows


def table_h8():
    from scipy.stats import fisher_exact
    from nisq_common import clopper
    dr = drift_table()
    header = ["Parameter", "n", "Flagged: Canary / QE / both", "QE also flagged, of Canary's flags [95% CI]",
              "Chance (QE's flag rate)", "Same direction", "Fisher p (one-sided)"]
    rows = []
    for p in PARAMS:
        g = dr[dr.parameter == p]
        n, a = len(g), int((g.fc & g.fq).sum())
        kc, kq = int(g.fc.sum()), int(g.fq.sum())
        lo, hi = clopper(a, kc) if kc else (np.nan, np.nan)
        both = g[g.fc & g.fq]
        p_f = fisher_exact([[a, kc - a], [kq - a, n - kc - kq + a]], alternative="greater").pvalue
        rows.append([f"{NAME[p]}", str(n), f"{kc} / {kq} / {a}",
                     f"{a}/{kc} ({100 * a / kc:.0f}%) [{100 * lo:.0f}, {100 * hi:.0f}]" if kc else "–",
                     f"{100 * kq / n:.0f}%", f"{int((np.sign(both.zc) == np.sign(both.zq)).sum())}/{len(both)}",
                     f"{p_f:.1e}"])
    cap = ("Do Canary's alarms hold up? Each tool's own change over the same interval (QE1 to QE2; Canary "
           "interpolated to QE's job times, as in Table H7) is flagged when it exceeds its standard error (two-sided "
           "z, Holm over every qubit and parameter, per tool). A Canary flag QE could not see (a change outside QE's "
           "window) is not counted here. Chance: the share QE flags among all qubits. QE is not ground truth: its "
           "error bars are as reported.")
    return "Table H8", cap, header, rows


def table_h9():
    import quantum_canary as canary
    from quantum_canary import api
    header = ["Check", "GREEN", "YELLOW", "RED", "YELLOW: fit over limit only", "YELLOW: T2*/T2 rule only",
              "YELLOW: both", "Probes over limit (T1 / Ramsey / echo / RB)"]
    rows = []
    for step in h.CANARY_STEPS:
        qs = canary.load(h.RUN / f"{step}.json").raw.qubits
        n = {s: sum(q.health == s for q in qs) for s in ("GREEN", "YELLOW", "RED")}
        why = {"fit": 0, "rule": 0, "both": 0}
        over = {k: 0 for k in ("T1", "Ramsey", "echo", "gate")}
        for q in qs:
            if q.health != "YELLOW":
                continue
            fit = api.fit_over_limit(q.result)
            rule = any(x.startswith("T2*/T2") for x in api.health_reasons(q.health, q.result))
            why["both" if fit and rule else "fit" if fit else "rule"] += 1
            for s in fit:
                over[s.split()[0]] += 1
        rows.append([step, str(n["GREEN"]), str(n["YELLOW"]), str(n["RED"]), str(why["fit"]), str(why["rule"]),
                     str(why["both"]), " / ".join(str(over[k]) for k in over)])
    cap = ("Canary's health verdicts on the 154 qubits of ibm_kingston, and why qubits were YELLOW: a probe's χ²/dof over "
           "its limit, T2* and T2 differing by more than 20% of T2 (slow frequency noise), or both. Last column: YELLOW "
           "qubits whose probe was over its limit, per probe.")
    return "Table H9", cap, header, rows


def markdown(title, cap, header, rows):
    cell = lambda s: s.replace("|", r"\|")
    out = [f"## {title}", "", cap, "", "| " + " | ".join(map(cell, header)) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(map(cell, r)) + " |" for r in rows]
    return "\n".join(out + [""])


MATH = (("τ\\_min", r"$\tau_{\min}$"), ("Δω/2π", r"$\Delta\omega/2\pi$"), ("Δω", r"$\Delta\omega$"),
        ("ε", r"$\varepsilon$"), ("±", r"$\pm$"), ("√", r"$\surd$"), ("²", r"$^2$"), ("χ", r"$\chi$"), ("≤", r"$\leq$"),
        ("→", r"$\to$"), ("−", "$-$"), ("–", "--"), ("π", r"$\pi$"), ("μ", r"$\mu$"), ("|", r"$|$"),
        ("×", r"$\times$"), ("σ", r"$\sigma$"), ("ρ", r"$\rho$"))


def esc(s):
    """Plain table text as pdfLaTeX text: special characters escaped, symbols as math, a hyphen before a digit as a
    minus sign (not in ranges such as 2.5--97.5 or exponents such as 1e-03)."""
    s = s.replace("%", r"\%").replace("_", r"\_").replace(">", "$>$").replace("<", "$<$")
    s = re.sub(r"(?<![A-Za-z])'", "`", s)
    for a, b in MATH:
        s = s.replace(a, b)
    s = re.sub(r"\bT([12])\b", r"$T_\1$", s)
    return re.sub(r"(?<![\w$.\-])-(?=\d)", "$-$", s)


def latex(title, cap, header, rows):
    body = [r"\begin{table*}[!t]", r"\centering", r"\footnotesize", f"\\caption{{{esc(cap)}}}",
            f"\\label{{tab:{title.split()[-1].lower()}}}", r"\begin{adjustbox}{max width=\linewidth}",
            f"\\begin{{tabular}}{{{'l' * len(header)}}}", r"\hline", " & ".join(esc(x) for x in header) + r" \\", r"\hline"]
    body += [" & ".join(esc(x) for x in r) + r" \\" for r in rows]
    return "\n".join(body + [r"\hline", r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}", ""])


tables = [fn() for fn in (table_h1, table_h2, table_h3, table_h4, table_h5, table_h6, table_h7, table_h8, table_h9)]
(OUT_DIR / "tables_hw.md").write_text("# IBM Kingston hardware tables (" + h.RUN.name + ")\n\n" +
                                      "\n".join(markdown(*t) for t in tables), encoding="utf-8")
(OUT_DIR / "tables_hw.tex").write_text("\n".join(latex(*t) for t in tables), encoding="utf-8")
print("saved:", OUT_DIR / "tables_hw.md", OUT_DIR / "tables_hw.tex")
