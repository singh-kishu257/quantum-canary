"""Loading of the IBM Kingston full-chip sandwich (C1 QE1 C2 QE2 C3, research/data/hardware) into one table."""
import ast
import json
import pathlib
import numpy as np
import pandas as pd
from nisq_common import holm

RUN = pathlib.Path(__file__).resolve().parents[1] / "data" / "hardware" / "ibm_kingston_20261026T014902Z"
CANARY_STEPS, QE_STEPS = ("C1", "C2", "C3"), ("QE1", "QE2")
SANDWICH = {"QE1": ("C1", "C2"), "QE2": ("C2", "C3")}
RAMSEY_OSC_HZ = 1e5
CANARY_FIELDS = {"T1_s": ("T1_s", "T1_sigma_s", "t1_chi2_dof"), "T2_s": ("T2_s", "T2_sigma_s", "echo_chi2_dof"),
                 "delta_omega": ("delta_omega", "delta_omega_sigma", "ramsey_chi2_dof"),
                 "epsilon_sx": ("epsilon_sx", "epsilon_sx_sigma", "gate_chi2_dof")}
QE_FIELDS = {"T1_s": ("t1", "T1"), "T2_s": ("hahn", "T2"), "delta_omega": ("ramsey", "Frequency"),
             "epsilon_sx": ("rb", "EPG_sx")}
PROBE_DOF = {"T1_s": 2, "T2_s": 2, "delta_omega": 6, "epsilon_sx": 1}


def _canary(step):
    d = json.load(open(RUN / f"{step}.json"))
    rows = []
    for q in d["report"]["qubits"]:
        r = q["result"]
        for p, (v, s, c) in CANARY_FIELDS.items():
            value, sigma = r[v], r[s]
            if p == "delta_omega":
                value, sigma = value / (2 * np.pi), sigma / (2 * np.pi)
            rows.append(dict(step=step, tool="canary", qubit=q["qubit"], parameter=p, value=value, sigma=sigma,
                             chi2_dof=r[c], health=q["health"], quality=""))
    return rows


def _parse(x):
    """A field saved either as its Python value or as that value's text."""
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return x
    return x


def _qe(step):
    d = json.load(open(RUN / f"{step}.json"))
    rows = []
    for p, (fam, name) in QE_FIELDS.items():
        for r in d[fam]["results"]:
            if r["name"] != name:
                continue
            comp = _parse(r["components"])
            q = int(str(comp[0])[1:])
            val = _parse(r["value"])
            value, sigma = (float(val[0]), float(val[1])) if isinstance(val, (list, tuple)) and len(val) == 2 else (np.nan, np.nan)
            if p == "delta_omega":
                value = value - RAMSEY_OSC_HZ
            chi = _parse(r["chisq"])
            rows.append(dict(step=step, tool="qe", qubit=q, parameter=p, value=value, sigma=sigma,
                             chi2_dof=float(chi) if isinstance(chi, (int, float)) else np.nan,
                             health="", quality=r["quality"]))
    return rows


def load():
    """One row per (step, qubit, parameter); delta_omega in Hz (Canary: delta_omega/2pi; QE: fitted frequency
    minus the 100 kHz offset, as validated in silico); ok = finite value and error bar."""
    rows = [r for s in CANARY_STEPS for r in _canary(s)] + [r for s in QE_STEPS for r in _qe(s)]
    d = pd.DataFrame(rows)
    d["ok"] = np.isfinite(d.value) & np.isfinite(d.sigma) & (d.sigma > 0)
    return d


def wide(d, parameter):
    """Qubit x step tables of value and sigma for one parameter (NaN where not ok)."""
    g = d[d.parameter == parameter]
    v = g.pivot(index="qubit", columns="step", values="value").where(g.pivot(index="qubit", columns="step", values="ok"))
    s = g.pivot(index="qubit", columns="step", values="sigma").where(v.notna())
    return v, s


def _qe_raw(step, family):
    """Per qubit, QE's raw counts for one family as qiskit-experiments data entries (marginal on that qubit)."""
    groups = json.load(open(RUN / "qubits.json"))["groups"]
    d = json.load(open(RUN / f"{step}.json"))
    out = {}
    for r in d[family]["raw"]:
        top = r["metadata"]
        par = top["composite_metadata"][0]
        for j, sub in enumerate(par["composite_metadata"]):
            q = groups[top["composite_index"][0]][par["composite_index"][j]]
            bit = par["composite_clbits"][j][0]
            n1 = sum(n for k, n in r["counts"].items() if k.replace(" ", "")[-1 - bit] == "1")
            out.setdefault(q, []).append({"counts": {"0": r["shots"] - n1, "1": n1}, "metadata": sub, "shots": r["shots"]})
    return out


def qe_refits():
    """qiskit-experiments' own analyses re-run on QE's raw counts: default (must reproduce the saved values) and
    with the decay floor fixed at its physical value (RB asymptote b = 1/2; T1 base = IBM's P(1|0) for the
    qubit). Cached in research/figures/hw_qe_refits.csv."""
    cache = pathlib.Path(__file__).resolve().parents[1] / "figures" / "hw_qe_refits.csv"
    if cache.exists():
        return pd.read_csv(cache)
    import warnings
    from qiskit_experiments.framework import ExperimentData
    from qiskit_experiments.library import StandardRB, T1
    warnings.filterwarnings("ignore")
    cal = json.load(open(RUN / "calibration_C1.json"))
    p10 = {i: {p["name"]: p["value"] for p in q}.get("prob_meas1_prep0", np.nan) for i, q in enumerate(cal["qubits"])}
    delays = [x * 1e-6 for x in (1, 20, 40, 80, 200, 400)]
    rows = []
    for step in QE_STEPS:
        for family, parameter, name, make, fixed in (
                ("rb", "epsilon_sx", "EPG_sx", lambda q: StandardRB((q,), [10, 100, 500], num_samples=10, seed=q),
                 lambda q: {"b": 0.5}),
                ("t1", "T1_s", "T1", lambda q: T1((q,), delays), lambda q: {"base": p10[q]})):
            for q, data in _qe_raw(step, family).items():
                exp, res = make(q), {}
                for tag, opts in (("default", {}), ("floor_fixed", {"fixed_parameters": fixed(q)})):
                    an = exp.analysis.copy()
                    an.set_options(plot=False, **opts)
                    ed = ExperimentData(experiment=exp)
                    ed.add_data(data)
                    an.run(ed, replace_results=True).block_for_results()
                    f = ed.analysis_results(dataframe=True)
                    x = f[f.name == name]
                    res[tag] = (x.value.iloc[0].nominal_value, x.value.iloc[0].std_dev) if len(x) else (np.nan, np.nan)
                rows.append(dict(step=step, qubit=q, parameter=parameter, default=res["default"][0],
                                 floor_fixed=res["floor_fixed"][0], floor_fixed_sigma=res["floor_fixed"][1]))
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False)
    return out


def canary_dw_range_hz():
    """Per qubit, Canary's unambiguous detuning range |delta_omega|/2pi < 0.9 pi / (2 pi tau_min) (Hz), from the
    Ramsey delays C1 chose with its priors (inversion.BackendProfile.dw_max_rad_s)."""
    from quantum_canary import inversion as inv
    out = {}
    for q in json.load(open(RUN / "C1.json"))["report"]["qubits"]:
        prof = inv.BackendProfile(architecture="superconducting", T1_prior_s=q["T1_prior_s"], T2_prior_s=q["T2_prior_s"],
                                  dt_ns=q["dt_ns"] or 0.2222, prior_confidence="prior_informed",
                                  custom_arch=dict(inv.ARCH_DEFAULTS["superconducting"], eps_typical=q["eps_prior"]))
        out[q["qubit"]] = prof.dw_max_rad_s / (2 * np.pi)
    return pd.Series(out)


def diff(c, q, parameter):
    """Paired difference over qubits where both are finite: relative (%) of the pair mean, delta_omega in Hz."""
    j = c.dropna().index.intersection(q.dropna().index)
    c, q = c.loc[j], q.loc[j]
    if parameter == "delta_omega":
        return c - q, (c + q) / 2
    return 100 * (c - q) / ((c + q) / 2), (c + q) / 2


def bland_altman(x):
    """n, bias with its 95% CI, SD, limits of agreement bias +- 1.96 SD with their approximate 95% CI
    (Bland & Altman 1999: SE = sqrt(3 SD^2 / n)), and the empirical 2.5th/97.5th percentiles."""
    from scipy.stats import t
    x = np.asarray(x, dtype=float)
    n, m, s = len(x), float(np.mean(x)), float(np.std(x, ddof=1))
    ci = float(t.ppf(0.975, n - 1) * s / np.sqrt(n))
    ci_loa = float(t.ppf(0.975, n - 1) * np.sqrt(3 * s * s / n))
    return dict(n=n, bias=m, bias_ci=ci, sd=s, loa_lo=m - 1.96 * s, loa_hi=m + 1.96 * s, loa_ci=ci_loa,
                p2_5=float(np.percentile(x, 2.5)), p97_5=float(np.percentile(x, 97.5)), median=float(np.median(x)))


def robust_sd(x):
    x = np.asarray(x, dtype=float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def chip_layout():
    """(x, y) of every qubit and the coupled pairs, from the two-qubit gates in IBM's calibration: Heron r2 rows of
    16 qubits (qubit 20k + c at column c of row k), each bridge qubit between the two rows it couples, at the mean
    column of its neighbours."""
    c = json.load(open(RUN / "calibration_C1.json"))
    pairs = sorted({tuple(sorted(g["qubits"])) for g in c["gates"] if len(g["qubits"]) == 2})
    pos = {q: (q % 20, -2 * (q // 20)) for q in range(len(c["qubits"])) if q % 20 < 16}
    for q in range(len(c["qubits"])):
        if q % 20 >= 16:
            nb = [p for a, b in pairs for p in (a, b) if q in (a, b) and p != q and p in pos]
            pos[q] = (float(np.mean([pos[p][0] for p in nb])), float(np.mean([pos[p][1] for p in nb])))
    return pos, pairs


def job_times(at="mid"):
    """When each measurement ran on the QPU (POSIX s): per Canary step its one job, per QE step and family its own
    job; the midpoint of IBM's running-to-finished window (at="mid") or its start (at="start")."""
    from datetime import datetime
    ts = lambda x: datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp()
    when = lambda j: (ts(j["running"]) + ts(j["finished"])) / 2 if at == "mid" else ts(j["running"])
    out = {}
    for step in CANARY_STEPS + QE_STEPS:
        jobs = {j["job_id"]: j for j in json.load(open(RUN / f"timing_{step}.json"))["jobs"]}
        for p in CANARY_FIELDS:
            ids = list(jobs) if step in CANARY_STEPS else json.load(open(RUN / f"{step}.json"))[QE_FIELDS[p][0]]["job_ids"]
            assert len(ids) == 1
            out[(step, p)] = when(jobs[ids[0]])
    return out


def drift(at="mid"):
    """Per (qubit, parameter), over the whole sandwich:
    Canary's change across C1, C2, C3: Cochran's Q (2 dof; with two checks Q is the square of canary.compare's z),
    on the error bars canary.compare uses (each inflated by sqrt(max(chi2/dof, 1)) where it inflates them, T2 as
    reported), with its p, Holm over every qubit and parameter, and
    the between-check SD beyond the error bars, tau (DerSimonian-Laird), relative to the weighted mean (%; delta-
    omega in Hz). QE's change QE1 to QE2 likewise (Q with 1 dof). The time-matched signed changes: Canary's
    trajectory interpolated linearly to the times of QE's two jobs (job_times), its change between them against
    QE2 - QE1 (relative to each tool's own mean, %; delta-omega in Hz). QE: floor-fixed refits for T1 and epsilon,
    default fits otherwise. Delta-omega only where both QE values lie inside Canary's unambiguous range."""
    from scipy.stats import chi2
    from quantum_canary.workflow import _DRIFT_PARAMS
    d, ref, rng, t = load(), qe_refits(), canary_dw_range_hz(), job_times(at)
    inflate = {name: chi is not None for name, _, chi in _DRIFT_PARAMS}
    rows = []
    for p in CANARY_FIELDS:
        v, s = wide(d, p)
        sc = s[list(CANARY_STEPS)]
        if inflate[p]:
            chi = d[(d.tool == "canary") & (d.parameter == p)].pivot(index="qubit", columns="step", values="chi2_dof")
            sc = sc * np.sqrt(np.maximum(chi[list(CANARY_STEPS)], 1))
        if p in ("T1_s", "epsilon_sx"):
            r = ref[ref.parameter == p]
            qv = r.pivot(index="qubit", columns="step", values="floor_fixed")
            qs = r.pivot(index="qubit", columns="step", values="floor_fixed_sigma")
        else:
            qv, qs = v[list(QE_STEPS)], s[list(QE_STEPS)]
        tc = [t[(c, p)] for c in CANARY_STEPS]
        t1, t2 = t[("QE1", p)], t[("QE2", p)]
        assert tc[0] < t1 < tc[1] < t2 < tc[2]
        for q in v.index:
            x, sx = v.loc[q, list(CANARY_STEPS)].values.astype(float), sc.loc[q].values.astype(float)
            y = np.array([qv[c].get(q, np.nan) for c in QE_STEPS], dtype=float)
            sy = np.array([qs[c].get(q, np.nan) for c in QE_STEPS], dtype=float)
            if not (np.all(np.isfinite(x)) and np.all(sx > 0) and np.all(np.isfinite(y)) and np.all(sy > 0)):
                continue
            if p == "delta_omega" and not np.all(np.abs(y) < rng[q]):
                continue
            row = dict(qubit=q, parameter=p)
            for tool, z, sz in (("canary", x, sx), ("qe", y, sy)):
                w = 1 / sz ** 2
                m = np.sum(w * z) / np.sum(w)
                Q = float(np.sum(w * (z - m) ** 2))
                tau = np.sqrt(max(0.0, (Q - (len(z) - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w))))
                row.update({f"{tool}_Q": Q, f"{tool}_p": float(chi2.sf(Q, len(z) - 1)), f"{tool}_mean": m,
                            f"{tool}_tau": tau if p == "delta_omega" else 100 * tau / m})
            dc = np.interp(t2, tc, x) - np.interp(t1, tc, x)
            dq = y[1] - y[0]
            row["canary_change"] = dc if p == "delta_omega" else 100 * dc / np.mean(x)
            row["qe_change"] = dq if p == "delta_omega" else 100 * dq / np.mean(y)
            rows.append(row)
    out = pd.DataFrame(rows)
    out["canary_p_holm"] = holm(out.canary_p.values)
    out["qe_p_holm"] = holm(out.qe_p.values)
    return out


def timing():
    return {s: json.load(open(RUN / f"timing_{s}.json")) for s in CANARY_STEPS + QE_STEPS}


def ibm_calibration():
    """IBM's published T1, T2 (s) and sx gate error per qubit, from the calibration snapshot taken at C1."""
    c = json.load(open(RUN / "calibration_C1.json"))
    out = {}
    for i, q in enumerate(c["qubits"]):
        props = {p["name"]: p["value"] for p in q}
        out[i] = {"T1_s": props.get("T1", np.nan) * 1e-6, "T2_s": props.get("T2", np.nan) * 1e-6}
    for g in c["gates"]:
        if g["gate"] == "sx":
            for p in g["parameters"]:
                if p["name"] == "gate_error":
                    out[g["qubits"][0]]["epsilon_sx"] = p["value"]
    return pd.DataFrame(out).T
