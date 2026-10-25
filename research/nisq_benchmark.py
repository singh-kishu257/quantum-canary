"""NISQ benchmark: Canary vs qiskit-experiments, on the same simulated qubits.

Every simulated qubit is
generated with every NISQ noise mechanism of quantum_canary.simulation on (T1 drift and telegraph
switching, 1/f or OU frequency drift, quasi-static dephasing, detuning during pulses, coherent gate
error, heterogeneous readout error, motional heating for ions), a calibration prior taken an hour
before, and time-resolved execution (the qubit keeps drifting while it is measured), on held-out seeds, with two factors drawn per qubit instead of fixed: the coherent share of the gate
error, U(0, 0.3), and the calibration prior's error envelope, U(0, 0.5). Each qubit is then measured by
these arms, all starting at the same simulated time and each scored against the truth averaged over its
own run:

  canary_x1          the package as shipped (8,640 shots)
  canary_x1_repeat   a second, independent check (null for compare: nothing changed)
  qe_ibm             qiskit-experiments 0.14.2 as IBM runs it (Real-time benchmarking tutorial): T1 and
                     T2Hahn at delays 1, 20, 40, 80, 200, 400 us, StandardRB at lengths 10, 100, 500 x 10
                     samples, 1,000 shots per circuit; T2Ramsey as in the qiskit-experiments manual
                     (delays 1-49 us in 2 us steps, osc_freq 100 kHz, 2,000 shots); default analysis.
                     Trapped ions: the same grids rescaled by the prior (longest T1 delay 3*T1, longest
                     T2Hahn delay 2.5*T2, T2Ramsey delays x T2/100 us and osc_freq / that factor).
  qe_equal           qe_ibm's design with Canary's shots per parameter (T1 900, T2Hahn 1,500,
                     T2Ramsey 4,800, RB 1,440: 8,640 in total)
  canary_b<B>, qe_b<B> shot sweep at 8 shared total budgets B = 1,080 ... 138,240 (8,640 x 1/8 ... 16, in
                     doublings), on instances below --n-sweep only: Canary with every circuit's shots
                     scaled (its B = 8,640 point is canary_x1); qiskit-experiments with qe_ibm's design
                     and allocation, every circuit's shots scaled by B / 92,000 (rounded)

Output: one long CSV, one row per (qubit, arm, parameter). Usage:
  python research/nisq_benchmark.py --architecture superconducting --n-instances 300 --shard 0 --n-shards 10 --out x.csv
  python research/nisq_benchmark.py --combine --n-instances 300 --n-sweep 100 --input a.csv --input b.csv --out y.csv"""
from __future__ import annotations

import argparse
import csv
import logging
import multiprocessing as mp
import pathlib
import sys
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
if (SCRIPT_DIR.parent / "src").is_dir():
    sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

from quantum_canary import api, inversion as inv, runner, simulation as sim

REGIME, T_CALIB_S, T_MEASURE_S, PROBE_BLOCKS = "nisq", 0.0, 3600.0, 10
PARAMETERS = ("T1_s", "T2_s", "delta_omega", "epsilon_sx")
TRUTH_FAMILIES = {"T1_s": ("t1_",), "T2_s": ("echo_",),
                  "delta_omega": ("ramsey_X_", "ramsey_Y_"), "epsilon_sx": ("rb_",)}
SPAM_CLIP_MAX, QE_BASIS = 0.499, ["rz", "sx", "x"]
ARCHITECTURES = ("superconducting", "trapped_ion")
SEED_OFFSET = {"superconducting": 500_000, "trapped_ion": 600_000}
COHERENT_RANGE, PRIOR_RANGE = (0.0, 0.3), (0.0, 0.5)
CANARY_SHOTS = dict(runner.DEFAULT_SHOTS)
SWEEP_FACTORS = (1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16)
CANARY_TOTAL, QE_IBM_TOTAL = 8640, 92000
ARMS = {"canary_x1": ("canary", 1.0), "canary_x1_repeat": ("canary", 1.0),
        "qe_ibm": ("qe", 1.0), "qe_equal": ("qe", "equal")}
ARMS.update({f"canary_b{round(CANARY_TOTAL * f)}": ("canary", f) for f in SWEEP_FACTORS if f != 1})
ARMS.update({f"qe_b{round(CANARY_TOTAL * f)}": ("qe", CANARY_TOTAL * f / QE_IBM_TOTAL) for f in SWEEP_FACTORS})
SWEEP_ARMS = tuple(list(ARMS)[4:])
AER_BASE = {"superconducting": 1_000_000_000, "trapped_ion": 1_050_000_000}
IBM_DELAYS = [1e-6, 20e-6, 40e-6, 80e-6, 200e-6, 400e-6]
IBM_RB_LENGTHS, IBM_RB_SAMPLES, IBM_SHOTS = [10, 100, 500], 10, 1000
MANUAL_RAMSEY_DELAYS = [float(x) for x in np.arange(1e-6, 50e-6, 2e-6)]
MANUAL_RAMSEY_OSC, MANUAL_RAMSEY_SHOTS, MANUAL_RAMSEY_T2 = 1e5, 2000, 100e-6
QE_IBM_SHOTS = {"t1": IBM_SHOTS, "hahn": IBM_SHOTS, "ramsey": MANUAL_RAMSEY_SHOTS, "rb": IBM_SHOTS}
QE_EQUAL_SHOTS = {"t1": 150, "hahn": 250, "ramsey": 192, "rb": 48}
COLUMNS = ["architecture", "instance_id", "seed", "arm", "tool", "shots_multiplier", "parameter",
           "true_value", "estimate", "sigma", "signed_error", "abs_error", "rel_error", "z_score",
           "chi2_dof", "health", "qe_quality", "fit_status", "shots_total", "run_duration_s",
           "coherent_fraction", "perturbation_frac", "prior_value", "t2_ramsey_over_echo",
           "rb_seed", "aer_seed_base"]
NAN = float("nan")


def _signed_error(value, truth_value):
    return value - truth_value if np.isfinite(value) and np.isfinite(truth_value) else NAN


def _abs_error(value, truth_value):
    return abs(value - truth_value) if np.isfinite(value) and np.isfinite(truth_value) else NAN


def _rel_error(value, truth_value):
    if not np.isfinite(value) or not np.isfinite(truth_value) or truth_value == 0:
        return NAN
    return abs(value - truth_value) / abs(truth_value)


def _z_score(value, truth_value, sigma):
    if not (np.isfinite(value) and np.isfinite(truth_value) and np.isfinite(sigma) and sigma > 0):
        return NAN
    return (value - truth_value) / sigma


def _ufloat_parts(value):
    """Value and error of a qiskit-experiments result."""
    if hasattr(value, "nominal_value"):
        return float(value.nominal_value), float(value.std_dev)
    try:
        return float(value), NAN
    except (TypeError, ValueError):
        return NAN, NAN


def _time_resolved_truth(truth, sim_result):
    """Canary's truth per parameter: averaged over the shots of the circuits that measure it."""
    out = {}
    for param, prefixes in TRUTH_FAMILIES.items():
        vals, weights = [], []
        for name, recs in zip(sim_result.circuit_names, sim_result.chunk_records):
            if not name.startswith(prefixes):
                continue
            for rec in recs:
                vals.append(runner.rb_gate_error(truth, rec["t_mid_s"]) if param == "epsilon_sx" else
                            float(rec["truth"][0][{"T1_s": "T1_s", "T2_s": "T2_s",
                                                   "delta_omega": "delta_omega_rad_s"}[param]]))
                weights.append(float(rec["shots"]))
        v, w = np.array(vals, dtype=float), np.array(weights, dtype=float)
        out[param] = float(v[0] + np.sum(w * (v - v[0])) / np.sum(w))
    return out


def instance_factors(seed: int):
    """Coherent fraction and prior envelope of one qubit, from their own RNG stream."""
    rng = np.random.default_rng([int(seed), 20261025])
    return float(rng.uniform(*COHERENT_RANGE)), float(rng.uniform(*PRIOR_RANGE))


def make_qubit(architecture: str, seed: int):
    """One NISQ qubit and its calibration prior, with the two drawn factors."""
    sc = architecture == "superconducting"
    cf, pf = instance_factors(seed)
    truth = sim.generate_device(architecture, n_qubits=1, seed=seed, regime=REGIME, overrides=None,
                                crosstalk_scale=1.0, detuning_1_over_f=sc, enable_heating_gate_error=not sc,
                                spam_spread_frac=0.5, perturbation_frac=None, dw_population="literature",
                                t1_telegraph=sc, quasi_static_dephasing=True, detuning_during_gates=True)
    truth.architecture_specific["coherent_fraction"] = cf
    prior = sim.generate_calibration_prior(truth, qubit=0, t0_s=T_CALIB_S, perturbation_frac=pf,
                                           spam_perturbation_frac=pf)
    return truth, prior, cf, pf


def run_canary(architecture, truth, prior, mult, rb_seed, aer_seed, t_measure):
    """One Canary check at mult x the default shots, block-interleaved, run in simulated real time."""
    shots = {k: int(round(v * mult)) for k, v in CANARY_SHOTS.items()}
    p01, p10 = (float(np.clip(prior.spam_p0_given_1_prior, 0, SPAM_CLIP_MAX)),
                float(np.clip(prior.spam_p1_given_0_prior, 0, SPAM_CLIP_MAX)))
    constants = dict(inv.ARCH_DEFAULTS[architecture])
    constants.update(T1_s=prior.T1_prior_s, T2_s=prior.T2_prior_s, eps_typical=prior.epsilon_prior,
                     p0_given_1=p01, p1_given_0=p10)
    profile = inv.BackendProfile(architecture=architecture, T1_prior_s=prior.T1_prior_s,
                                 T2_prior_s=prior.T2_prior_s, dt_ns=inv.ARCH_DEFAULTS[architecture]["dt_ns"],
                                 backend_name=f"nisq_benchmark::{architecture}", custom_arch=constants,
                                 prior_confidence="prior_informed")
    circuits, meta = inv.build_probe_circuits(profile, seed=rb_seed)
    shots_list = runner._shots_list(meta, shots)
    bc, bs, bi = inv.interleave_probe_blocks(circuits, shots_list, PROBE_BLOCKS)
    if len(bc) > 1000:
        raise RuntimeError(f"{len(bc)} block entries would reuse Aer seeds")
    res = sim.simulate_probe(bc, truth, shots=bs, t_s=t_measure, dt_ns=profile.dt_ns, seed=aer_seed,
                             time_resolved=True)
    counts = inv.merge_block_counts(res.counts_list, bi, len(circuits))
    r = inv.lindblad_inversion(counts, meta, profile, shots_t1=shots["t1"], shots_ramsey=shots["ramsey"],
                               shots_gate=shots["gate"], shots_echo=shots["echo"], qubit_id=0)
    truths = _time_resolved_truth(truth, res)
    health = runner._health(0, r)
    est = {"T1_s": (r.T1_s, r.T1_sigma_s, r.t1_chi2_dof), "T2_s": (r.T2_s, r.T2_sigma_s, r.echo_chi2_dof),
           "delta_omega": (r.delta_omega, r.delta_omega_sigma, r.ramsey_chi2_dof),
           "epsilon_sx": (r.epsilon_sx, r.epsilon_sx_sigma, r.gate_chi2_dof)}
    return {p: {"value": float(v), "sigma": float(s), "chi2": float(c), "truth": float(truths[p]),
                "health": health, "quality": "", "status": "ok" if np.isfinite(v) and np.isfinite(s) else "failed"}
            for p, (v, s, c) in est.items()}, sum(shots_list), float(res.run_duration_s), int(rb_seed)


def qe_experiments(architecture, prior, rb_seed):
    """qe_ibm's four experiments (trapped ions: grids rescaled by the prior)."""
    from qiskit_experiments.library import StandardRB, T1, T2Hahn, T2Ramsey
    if architecture == "superconducting":
        t1_d, hahn_d = list(IBM_DELAYS), list(IBM_DELAYS)
        ram_d, osc = list(MANUAL_RAMSEY_DELAYS), MANUAL_RAMSEY_OSC
    else:
        t1_d = [d * 3.0 * prior.T1_prior_s / IBM_DELAYS[-1] for d in IBM_DELAYS]
        hahn_d = [d * 2.5 * prior.T2_prior_s / IBM_DELAYS[-1] for d in IBM_DELAYS]
        k = prior.T2_prior_s / MANUAL_RAMSEY_T2
        ram_d, osc = [d * k for d in MANUAL_RAMSEY_DELAYS], MANUAL_RAMSEY_OSC / k
    exps = {"t1": T1((0,), delays=t1_d), "ramsey": T2Ramsey((0,), delays=ram_d, osc_freq=osc),
            "hahn": T2Hahn((0,), delays=hahn_d, num_echoes=1),
            "rb": StandardRB((0,), lengths=IBM_RB_LENGTHS, num_samples=IBM_RB_SAMPLES, seed=int(rb_seed))}
    for exp in exps.values():
        exp.set_transpile_options(basis_gates=QE_BASIS, num_processes=1)
        exp.analysis.set_options(plot=False)
    return exps, osc


def run_qe(architecture, truth, prior, mult, rb_seed, aer_seed, t_measure):
    """qiskit-experiments on the same qubit; mult = 'equal' or a multiplier of qe_ibm's shots."""
    from qiskit_experiments.framework import ExperimentData
    exps, osc = qe_experiments(architecture, prior, rb_seed)
    per = QE_EQUAL_SHOTS if mult == "equal" else {k: int(round(v * mult)) for k, v in QE_IBM_SHOTS.items()}
    circuits, shots, family = [], [], []
    for fam, exp in exps.items():
        tc = exp._transpiled_circuits()
        circuits += tc
        shots += [per[fam]] * len(tc)
        family += [fam] * len(tc)
    if len(circuits) > 1000:
        raise RuntimeError("qiskit-experiments Aer seeds would overlap")
    res = sim.simulate_probe(circuits, truth, shots=shots, t_s=t_measure, dt_ns=None, seed=aer_seed,
                             time_resolved=True)
    found = {}
    for fam, exp in exps.items():
        idx = [i for i, f in enumerate(family) if f == fam]
        data = ExperimentData(experiment=exp)
        data.add_data([{"counts": res.counts_list[i], "metadata": circuits[i].metadata, "shots": shots[i]}
                       for i in idx])
        exp.analysis.run(data, replace_results=True).block_for_results()
        frame = data.analysis_results(dataframe=True)
        found[fam] = ({str(n): (row["value"], row["quality"]) for n, row in zip(frame["name"], frame.to_dict("records"))}
                      if len(frame) else {})

    def window_truth(param, fam):
        vals, weights = [], []
        for i in (i for i, f in enumerate(family) if f == fam):
            for rec in res.chunk_records[i]:
                vals.append(runner.rb_gate_error(truth, rec["t_mid_s"]) if param == "epsilon_sx" else
                            float(rec["truth"][0][{"T1_s": "T1_s", "T2_s": "T2_s",
                                                   "delta_omega": "delta_omega_rad_s"}[param]]))
                weights.append(float(rec["shots"]))
        v, w = np.array(vals), np.array(weights)
        return float(v[0] + np.sum(w * (v - v[0])) / np.sum(w))

    out = {}
    for param, (fam, name) in {"T1_s": ("t1", "T1"), "T2_s": ("hahn", "T2"),
                               "delta_omega": ("ramsey", "Frequency"), "epsilon_sx": ("rb", "EPG_sx")}.items():
        entry = found[fam].get(name)
        if entry is None:
            value, sigma, quality, status = NAN, NAN, "", "qe_no_result"
        else:
            value, sigma = _ufloat_parts(entry[0])
            quality = str(entry[1])
            if param == "delta_omega":
                value, sigma = 2.0 * np.pi * (value - osc), 2.0 * np.pi * sigma
            status = "ok" if np.isfinite(value) and np.isfinite(sigma) else "qe_non_finite"
        out[param] = {"value": value, "sigma": sigma, "chi2": NAN, "truth": window_truth(param, fam),
                      "health": "", "quality": quality, "status": status}
    return out, int(sum(shots)), float(res.run_duration_s), int(rb_seed)


def run_one_instance(architecture: str, instance_id: int, seed: int, n_sweep: int) -> List[Dict[str, Any]]:
    """All arms on one simulated qubit (sweep arms only if instance_id < n_sweep)."""
    rows = []
    t_measure = T_MEASURE_S
    try:
        truth, prior, cf, pf = make_qubit(architecture, seed)
        r_ratio = float(np.asarray(truth.architecture_specific.get("t2_ramsey_over_echo", [NAN]), dtype=float)[0])
        prior_of = {"T1_s": prior.T1_prior_s, "T2_s": prior.T2_prior_s, "delta_omega": NAN,
                    "epsilon_sx": prior.epsilon_prior}
    except Exception as exc:
        return [dict(dict.fromkeys(COLUMNS, NAN), architecture=architecture, instance_id=instance_id, seed=seed,
                     arm="setup", tool="", parameter=p, fit_status=f"exception:{type(exc).__name__}:{exc}")
                for p in PARAMETERS]
    for j, (arm, (tool, mult)) in enumerate(ARMS.items()):
        if arm in SWEEP_ARMS and instance_id >= n_sweep:
            continue
        rb_seed, aer_seed = int(seed) * 100 + j, AER_BASE[architecture] + (j * 1000 + instance_id) * 1000
        try:
            fn = run_canary if tool == "canary" else run_qe
            est, n_shots, duration, _ = fn(architecture, truth, prior, mult, rb_seed, aer_seed, t_measure)
        except Exception as exc:
            est = {p: {"value": NAN, "sigma": NAN, "chi2": NAN, "truth": NAN, "health": "", "quality": "",
                       "status": f"exception:{type(exc).__name__}:{exc}"} for p in PARAMETERS}
            n_shots, duration = 0, NAN
        for p in PARAMETERS:
            e = est[p]
            rows.append(dict(architecture=architecture, instance_id=instance_id, seed=seed, arm=arm, tool=tool,
                             shots_multiplier=mult, parameter=p, true_value=e["truth"], estimate=e["value"],
                             sigma=e["sigma"], signed_error=_signed_error(e["value"], e["truth"]),
                             abs_error=_abs_error(e["value"], e["truth"]),
                             rel_error=_rel_error(e["value"], e["truth"]),
                             z_score=_z_score(e["value"], e["truth"], e["sigma"]), chi2_dof=e["chi2"],
                             health=e["health"], qe_quality=e["quality"], fit_status=e["status"],
                             shots_total=n_shots, run_duration_s=duration, coherent_fraction=cf,
                             perturbation_frac=pf, prior_value=prior_of[p], t2_ramsey_over_echo=r_ratio,
                             rb_seed=rb_seed, aer_seed_base=aer_seed))
    return rows


def _worker(job):
    warnings.filterwarnings("ignore")
    logging.getLogger("qiskit_aer.noise.noise_model").setLevel(logging.ERROR)
    return run_one_instance(*job)


def expected_arms(instance_id: int, n_sweep: int) -> List[str]:
    return [a for a in ARMS if a not in SWEEP_ARMS or instance_id < n_sweep]


def validate(rows, architectures, ids, n_sweep) -> List[str]:
    """Every instance in ids present once per arm and parameter; nothing dropped or duplicated."""
    problems = []
    keys = [(r["architecture"], int(r["instance_id"]), r["arm"], r["parameter"]) for r in rows]
    if len(keys) != len(set(keys)):
        problems.append("duplicate (architecture, instance, arm, parameter) rows")
    have = set(keys)
    for a in architectures:
        for i in ids:
            for arm in expected_arms(i, n_sweep):
                for p in PARAMETERS:
                    if (a, i, arm, p) not in have:
                        problems.append(f"missing {a} instance {i} {arm} {p}")
    extra = len(have) - sum(len(expected_arms(i, n_sweep)) for i in ids) * len(PARAMETERS) * len(architectures)
    if extra > 0:
        problems.append(f"{extra} unexpected rows")
    return problems[:20]


def write_csv(rows, path: pathlib.Path):
    rows = sorted(rows, key=lambda r: (r["architecture"], int(r["instance_id"]), list(ARMS).index(r["arm"])
                                       if r["arm"] in ARMS else -1, PARAMETERS.index(r["parameter"])))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    from provenance import write_meta
    write_meta(path)
    print(f"[nisq_benchmark] wrote {len(rows)} rows -> {path} (+ .meta.json)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--architecture", choices=ARCHITECTURES)
    ap.add_argument("--n-instances", type=int, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--n-sweep", type=int, default=100)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--input", type=pathlib.Path, action="append", default=[])
    a = ap.parse_args()
    if a.combine:
        rows = []
        for path in a.input:
            with open(path, newline="", encoding="utf-8") as f:
                rows += list(csv.DictReader(f))
        archs = sorted({r["architecture"] for r in rows})
        problems = validate(rows, archs, range(a.n_instances), a.n_sweep)
        write_csv(rows, a.out)
    else:
        if a.architecture is None:
            sys.exit("--architecture is required")
        if not 0 <= a.shard < a.n_shards:
            sys.exit("--shard must be in 0..n_shards-1")
        if a.n_instances > 1000:
            sys.exit("--n-instances must be <= 1000 (Aer seed blocks are 1000 instances wide)")
        ids = [i for i in range(a.n_instances) if i % a.n_shards == a.shard]
        jobs = [(a.architecture, i, SEED_OFFSET[a.architecture] + i, a.n_sweep) for i in ids]
        workers = a.workers or max(1, mp.cpu_count() - 1)
        print(f"[nisq_benchmark] {a.architecture}: shard {a.shard}/{a.n_shards}, {len(ids)} instances "
              f"(i mod {a.n_shards} == {a.shard}), {workers} workers", flush=True)
        if workers <= 1:
            rows = [r for job in jobs for r in _worker(job)]
        else:
            with mp.Pool(workers) as pool:
                rows = [r for rs in pool.imap(_worker, jobs, chunksize=1) for r in rs]
        problems = validate(rows, [a.architecture], ids, a.n_sweep)
        write_csv(rows, a.out)
    bad = sum(1 for r in rows if r["fit_status"] != "ok")
    print(f"[nisq_benchmark] {len(rows)} rows, {bad} not ok (failures are kept and counted)")
    if problems:
        print("[nisq_benchmark] VALIDATION FAILED:\n  - " + "\n  - ".join(problems))
        sys.exit(1)
    print("[nisq_benchmark] validation passed")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.getLogger("qiskit_aer.noise.noise_model").setLevel(logging.ERROR)
    if sys.platform != "win32":
        try:
            mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    main()
