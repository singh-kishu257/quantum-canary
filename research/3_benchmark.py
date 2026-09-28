"""3_benchmark.py -- parameter-recovery accuracy on NISQ simulated hardware.

Per instance: 1c.generate_device(regime="nisq") -> hidden DeviceTruth with
drift and coherent error mechanisms; 1c.generate_calibration_prior() -> what
Canary may know; 1_inversion builds its own probe circuits from the prior;
1c.simulate_probe() executes them on Aer in a run starting at T_MEASURE_S;
1_inversion.lindblad_inversion() recovers T1, T2, delta_omega, epsilon_sx.
Ground truth is read by the simulator, by the calibration-prior generator
(which perturbs T1, T2, eps by U(1±f) and passes SPAM through exactly -- an
oracle-SPAM idealisation; see --spam-scale/--spam-swap), and by the final
comparison.

true_value is always built from the exact parameters the simulator used.
epsilon truth is 1c.effective_gate_epsilon(): the per-gate
error of the simulated channel, including gate-time decoherence and any
coherent residual surviving the native G*G^-1 pair. The prior is generated at
T_CALIB_S = 0 while the device is simulated and evaluated at T_MEASURE_S
(default 3600 s, --t-measure), so the prior's staleness is the
calibration-prior envelope plus the drift over T_MEASURE_S.

Time-resolved probe run (--time-resolved; available but OFF by default: its
superconducting effect is negligible and its trapped-ion effect is set by the
Class C correlation time 1c ti.detuning_drift.correlation_time_s, which has
no source; the CSV column time_resolved records the mode).
1c.simulate_probe(time_resolved=True) starts the run at T_MEASURE_S and
executes the circuits one after another in probe order (circuit-major: all
shots of a circuit before the next; an assumption, a provider may
interleave). One shot lasts its delays plus its noisy native gates times the
gate duration plus a repetition overhead t_rep = 0 s. t_rep = 0 is a lower
bound: IBM's Qiskit Runtime documentation gives the default rep_delay only
as backend-specific (backend.default_rep_delay), no IonQ value was verified,
and readout time is not counted. Each circuit's shots are simulated in
chunks at the truth of the chunk mid-time; a chunk spans at most 1/10 of the
shortest drift correlation time, with at most 100 chunks per circuit (a
runtime cap). What varies during a run is exactly what the drift processes
already attached by 1c vary, with no new parameter: superconducting log-T1
(OU, 1800 s) and detuning (low-pass, correlation time 1800 s/(2*pi)), and
trapped-ion detuning (OU, 100 s); epsilon follows T1/T2 through gate-time
decoherence. A superconducting run lasts about 1 s, far shorter than those
times, so its truth does not change within the run. A trapped-ion run lasts
hours (the T1 probe alone has delays of tens to hundreds of seconds), so the
detuning varies within and between Ramsey circuits: shot-to-shot
quasi-static noise plus within-run drift. The comparison truth per parameter
is the shot-weighted time average over the chunks of the probe family that
measures it (TRUTH_FAMILIES): T1 <- t1 circuits; T2 <- Ramsey + echo;
delta_omega <- Ramsey X/Y (|signed mean|); epsilon <- gate-repetition
circuits (effective_gate_epsilon at the chunk times). truth_rel_spread is
(max - min)/|mean| of the parameter over those chunks, and run_duration_s is
the length of the run. By default (without --time-resolved), every circuit
and shot is simulated with the truth at T_MEASURE_S, true_value is
1c.evaluate_truth_at() there, truth_rel_spread = 0 and run_duration_s is NaN.

Two-state (telegraph) T1 switching (--t1-telegraph, superconducting only) is
OFF by default: Burnett et al.'s switching rates are Class A, but the
switching amplitude is Class C (see 1c sc.T1_telegraph.*). Motional heating
is off (the trapped-ion
heating-to-gate-error term is left disabled), and crosstalk is inactive:
probes are single-qubit, so --crosstalk-scale has no effect in this benchmark.

SPAM is not benchmarked: the inversion consumes it as a known constant and
never estimates it. delta_omega has no prior: CalibrationPrior deliberately
omits it, so prior_value is NaN for that parameter.

Output: one CSV, no figures.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import logging
import multiprocessing as mp
import pathlib
import sys
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


inv = _load("inversion", "1_inversion.py")
sim = _load("sim1c", "1c_simulation_models.py")

REGIME = "nisq"
T_CALIB_S = 0.0
T_MEASURE_S = 3600.0
SHOTS_T1, SHOTS_RAMSEY, SHOTS_GATE, SHOTS_ECHO = 300, 1000, 500, 500
ARCHITECTURES = ("superconducting", "trapped_ion")
ARCH_SEED_OFFSET = {"superconducting": 0, "trapped_ion": 100_000}
PARAMETERS = ("T1_s", "T2_s", "delta_omega", "epsilon_sx")
CIRCUIT_PREFIXES = ("t1_", "ramsey_X_", "ramsey_Y_", "gate_rep_N", "echo_")

CSV_COLUMNS = [
    "architecture", "instance_id", "seed", "parameter",
    "true_value", "canary_value", "absolute_error", "relative_error",
    "success", "fit_status", "sigma_canary", "chi2_dof", "chi2_dof_secondary",
    "shots_total", "prior_value", "prior_absolute_error", "prior_relative_error",
    "timestamp_utc",
    "perturbation_frac", "spam_scale", "spam_swap", "spam_spread", "cold_start",
    "dw_population", "t_measure_s", "coherent_fraction", "crosstalk_scale",
    "detuning_1_over_f", "alpha", "dw_aliased",
    "time_resolved", "t1_telegraph", "run_duration_s", "truth_rel_spread",
]
NAN = float("nan")

# Benchmark-design configuration (CLI flags). These defaults reproduce the
# pre-flag benchmark exactly; each value is recorded (resolved) in the CSV.
DEFAULT_CFG: Dict[str, Any] = {
    "perturbation_frac": None,          # None -> 1c _CALIB_PERTURBATION table
    "spam_scale": 1.0,                  # multiplies the SPAM handed to the inversion
    "spam_swap": False,                 # swap p0|1 <-> p1|0 handed to the inversion
    "spam_spread": 0.0,                 # 1c generate_device(spam_spread_frac=...)
    "cold_start": False,                # arch_default profile instead of the prior
    "dw_population": "estimator_window",  # 1c generate_device(dw_population=...)
    "t_measure_s": T_MEASURE_S,         # simulation / truth-evaluation time
    "coherent_fraction": None,          # None -> 1c registry value
    "crosstalk_scale": 1.0,             # no effect: probes are single-qubit
    "detuning_1_over_f": False,         # superconducting only
    "alpha": None,                      # 1/f alpha override; needs detuning_1_over_f
    "time_resolved": False,             # 1c simulate_probe(time_resolved=...)
    "t1_telegraph": False,              # superconducting only; Class C amplitude
}
# Probe family (circuit-name prefixes) whose chunks define each parameter's
# comparison truth in a time-resolved run.
TRUTH_FAMILIES = {"T1_s": ("t1_",), "T2_s": ("ramsey_X_", "ramsey_Y_", "echo_"),
                  "delta_omega": ("ramsey_X_", "ramsey_Y_"),
                  "epsilon_sx": ("gate_rep_N",)}
# Upper clip for the SPAM handed to the inversion: keeps each readout error
# below 0.5 and 1 - p0|1 - p1|0 >= 0.002 > 0, so the readout contrast used by
# the forward models never vanishes or inverts.
SPAM_CLIP_MAX = 0.499


def _abs_error(value: float, truth_value: float) -> float:
    if not np.isfinite(value) or not np.isfinite(truth_value):
        return NAN
    return abs(value - truth_value)


def _rel_error(value: float, truth_value: float) -> float:
    if not np.isfinite(value) or not np.isfinite(truth_value) or truth_value == 0:
        return NAN
    return abs(value - truth_value) / abs(truth_value)


def _row(architecture: str, instance_id: int, seed: int, parameter: str,
         timestamp: str, **kw) -> Dict[str, Any]:
    row = dict.fromkeys(CSV_COLUMNS, NAN)
    row.update(architecture=architecture, instance_id=instance_id, seed=seed,
               parameter=parameter, timestamp_utc=timestamp,
               success=False, fit_status="", shots_total=0)
    row.update(kw)
    return row


def _resolve_cfg(cfg: Any = None) -> Dict[str, Any]:
    out = dict(DEFAULT_CFG)
    if cfg:
        unknown = set(cfg) - set(DEFAULT_CFG)
        if unknown:
            raise ValueError(f"unknown config keys {sorted(unknown)}")
        out.update(cfg)
    return out


def _config_columns(architecture: str, c: Dict[str, Any]) -> Dict[str, Any]:
    """Resolved config values recorded in every row of the instance."""
    extra = sim.get_architecture_profile(architecture).extra
    pf = c["perturbation_frac"]
    if pf is None:
        pf = sim.resolve_calibration_perturbation(architecture, REGIME)
    cf = c["coherent_fraction"]
    if cf is None:
        cf = extra["coherent_fraction"]
    alpha = NAN
    if c["detuning_1_over_f"]:
        alpha = (c["alpha"] if c["alpha"] is not None
                 else extra.get("detuning_1_over_f_alpha", NAN))  # SC-only key
    return {"perturbation_frac": float(pf), "spam_scale": float(c["spam_scale"]),
            "spam_swap": bool(c["spam_swap"]), "spam_spread": float(c["spam_spread"]),
            "cold_start": bool(c["cold_start"]), "dw_population": str(c["dw_population"]),
            "t_measure_s": float(c["t_measure_s"]), "coherent_fraction": float(cf),
            "crosstalk_scale": float(c["crosstalk_scale"]),
            "detuning_1_over_f": bool(c["detuning_1_over_f"]), "alpha": float(alpha),
            "time_resolved": bool(c["time_resolved"]),
            "t1_telegraph": bool(c["t1_telegraph"])}


def _inversion_spam(prior: Any, c: Dict[str, Any]):
    """SPAM handed to the inversion: the prior's (oracle) SPAM scaled by
    spam_scale, then swapped if spam_swap, then clipped to [0, SPAM_CLIP_MAX].
    Truth and the prior_value columns are untouched."""
    p01 = prior.spam_p0_given_1_prior * c["spam_scale"]
    p10 = prior.spam_p1_given_0_prior * c["spam_scale"]
    if c["spam_swap"]:
        p01, p10 = p10, p01
    return (float(np.clip(p01, 0.0, SPAM_CLIP_MAX)),
            float(np.clip(p10, 0.0, SPAM_CLIP_MAX)))


def _time_resolved_truth(truth: Any, sim_result: Any):
    """Comparison truth of a time-resolved run: per parameter, the
    shot-weighted time average over the chunks of the probe family that
    measures it (TRUTH_FAMILIES), using the exact per-chunk truth the
    simulator used; epsilon is 1c.effective_gate_epsilon() at the chunk
    mid-times; delta_omega is |shot-weighted mean of the signed detuning|.
    Also returns (max - min)/|mean| over those chunks."""
    true_values: Dict[str, float] = {}
    spread: Dict[str, float] = {}
    for param, prefixes in TRUTH_FAMILIES.items():
        vals, weights = [], []
        for name, recs in zip(sim_result.circuit_names, sim_result.chunk_records):
            if not name.startswith(prefixes):
                continue
            for rec in recs:
                if param == "epsilon_sx":
                    vals.append(sim.effective_gate_epsilon(truth, 0, rec["t_mid_s"]))
                else:
                    key = {"T1_s": "T1_s", "T2_s": "T2_s",
                           "delta_omega": "delta_omega_rad_s"}[param]
                    vals.append(float(rec["truth"][0][key]))
                weights.append(float(rec["shots"]))
        v, w = np.array(vals, dtype=float), np.array(weights, dtype=float)
        # Offset by v[0] so a parameter constant over the chunks averages to
        # exactly that value (no floating-point rounding of sum(w*v)/sum(w)).
        mean = float(v[0] + np.sum(w * (v - v[0])) / np.sum(w))
        spread[param] = float((v.max() - v.min()) / abs(mean)) if mean != 0 else NAN
        true_values[param] = abs(mean) if param == "delta_omega" else mean
    return true_values, spread


def run_one_instance(architecture: str, instance_id: int, seed: int,
                     cfg: Any = None) -> List[Dict[str, Any]]:
    timestamp = datetime.now(timezone.utc).isoformat()
    c = _resolve_cfg(cfg)
    t_measure_s = c["t_measure_s"]
    cfg_cols = _config_columns(architecture, c)
    try:
        if (c["detuning_1_over_f"] and architecture != "superconducting") or \
                (c["alpha"] is not None and not c["detuning_1_over_f"]):
            raise ValueError("detuning_1_over_f is superconducting-only and alpha "
                             "requires detuning_1_over_f")
        if c["t1_telegraph"] and architecture != "superconducting":
            raise ValueError("t1_telegraph is superconducting-only")
        truth = sim.generate_device(architecture, n_qubits=1, seed=seed, regime=REGIME,
                                    crosstalk_scale=c["crosstalk_scale"],
                                    detuning_1_over_f=c["detuning_1_over_f"],
                                    spam_spread_frac=c["spam_spread"],
                                    perturbation_frac=c["perturbation_frac"],
                                    dw_population=c["dw_population"],
                                    t1_telegraph=c["t1_telegraph"])
        if c["coherent_fraction"] is not None:
            truth.architecture_specific["coherent_fraction"] = float(c["coherent_fraction"])
        if c["alpha"] is not None:
            truth.drift_processes["detuning"].params["alpha"] = float(c["alpha"])
        prior = sim.generate_calibration_prior(truth, qubit=0, t0_s=T_CALIB_S,
                                               perturbation_frac=c["perturbation_frac"])
        p0_given_1, p1_given_0 = _inversion_spam(prior, c)

        if c["cold_start"]:
            # Architecture-default grid and eps_typical; the prior is still
            # generated above and recorded in the prior_* columns.
            profile = inv.BackendProfile.from_architecture(
                architecture, backend_name=f"nisq_benchmark_sim::{architecture}",
                p0_given_1=p0_given_1, p1_given_0=p1_given_0)
        else:
            constants = dict(inv.ARCH_DEFAULTS[architecture])
            constants.update(T1_s=prior.T1_prior_s, T2_s=prior.T2_prior_s,
                             eps_typical=prior.epsilon_prior,
                             p0_given_1=p0_given_1,
                             p1_given_0=p1_given_0)
            profile = inv.BackendProfile(
                architecture=architecture,
                T1_prior_s=prior.T1_prior_s,
                T2_prior_s=prior.T2_prior_s,
                dt_ns=inv.ARCH_DEFAULTS[architecture]["dt_ns"],
                backend_name=f"nisq_benchmark_sim::{architecture}",
                custom_arch=constants,
                prior_confidence="prior_informed")

        circuits, meta = inv.build_probe_circuits(profile)
        for qc in circuits:
            if not any(qc.name.startswith(p) for p in CIRCUIT_PREFIXES):
                raise RuntimeError(f"unexpected circuit name {qc.name!r}")

        shots_list = ([SHOTS_T1] * meta["n_t1"] + [SHOTS_RAMSEY] * meta["n_ramsey"]
                      + [SHOTS_GATE] * meta["n_gate"] + [SHOTS_ECHO] * meta["n_echo"])
        if len(shots_list) != len(circuits):
            raise RuntimeError(f"{len(shots_list)} shot entries != {len(circuits)} circuits")

        # Independent shot noise per circuit: circuit i uses Aer seed seed*1000 + i
        # (a probe has far fewer than 1000 circuits, so seeds never collide across instances).
        sim_result = sim.simulate_probe(circuits, truth, shots=shots_list,
                                        t_s=t_measure_s, dt_ns=profile.dt_ns,
                                        seed=int(seed) * 1000,
                                        time_resolved=c["time_resolved"])
        result = inv.lindblad_inversion(
            sim_result.counts_list, meta, profile,
            shots_t1=SHOTS_T1, shots_ramsey=SHOTS_RAMSEY, shots_gate=SHOTS_GATE,
            shots_echo=SHOTS_ECHO, qubit_id=0, timestamp=timestamp)

        if c["time_resolved"]:
            true_values, truth_spread = _time_resolved_truth(truth, sim_result)
            run_duration_s = float(sim_result.run_duration_s)
        else:
            tv_now = sim.evaluate_truth_at(truth, 0, t_measure_s)
            true_values = {"T1_s": float(tv_now["T1_s"]),
                           "T2_s": float(tv_now["T2_s"]),
                           "delta_omega": abs(float(tv_now["delta_omega_rad_s"])),
                           "epsilon_sx": sim.effective_gate_epsilon(truth, 0, t_measure_s)}
            # Frozen run: no within-run variation; no schedule is computed.
            truth_spread = dict.fromkeys(PARAMETERS, 0.0)
            run_duration_s = NAN
        canary = {"T1_s": float(result.T1_s),
                  "T2_s": float(result.T2_s),
                  "delta_omega": abs(float(result.delta_omega)),
                  "epsilon_sx": float(result.epsilon_sx)}
        sigmas = {"T1_s": float(result.T1_sigma_s),
                  "T2_s": float(result.T2_sigma_s),
                  "delta_omega": float(result.delta_omega_sigma),
                  "epsilon_sx": float(result.epsilon_sx_sigma)}
        chi2 = {"T1_s": float(result.t1_chi2_dof),
                "T2_s": float(result.ramsey_chi2_dof),
                "delta_omega": float(result.ramsey_chi2_dof),
                "epsilon_sx": float(result.gate_chi2_dof)}
        chi2_secondary = {"T2_s": float(result.echo_chi2_dof)}
        priors = {"T1_s": float(prior.T1_prior_s),
                  "T2_s": float(prior.T2_prior_s),
                  "delta_omega": NAN,
                  "epsilon_sx": float(prior.epsilon_prior)}
        # Post-inversion comparison only: would the true |delta_omega| alias
        # (accumulate > pi) at the shortest Ramsey delay actually probed?
        dw_aliased = {"delta_omega": bool(true_values["delta_omega"]
                                          * meta["ramsey_delays_s"][0] > np.pi)}

        rows = []
        for param in PARAMETERS:
            tv, cv, sg, pv = true_values[param], canary[param], sigmas[param], priors[param]
            converged = bool(np.isfinite(cv) and np.isfinite(sg))
            rows.append(_row(
                architecture, instance_id, seed, param, timestamp,
                true_value=tv, canary_value=cv,
                absolute_error=_abs_error(cv, tv), relative_error=_rel_error(cv, tv),
                success=converged,
                fit_status="ok" if converged else "non_convergent_infinite_sigma",
                sigma_canary=sg, chi2_dof=chi2[param],
                chi2_dof_secondary=chi2_secondary.get(param, NAN),
                shots_total=sum(shots_list), prior_value=pv,
                prior_absolute_error=_abs_error(pv, tv),
                prior_relative_error=_rel_error(pv, tv),
                dw_aliased=dw_aliased.get(param, NAN), **cfg_cols,
                run_duration_s=run_duration_s, truth_rel_spread=truth_spread[param]))
        return rows

    except Exception as exc:
        status = f"exception:{type(exc).__name__}:{exc}"
        return [_row(architecture, instance_id, seed, p, timestamp, fit_status=status,
                     **cfg_cols)
                for p in PARAMETERS]


def _worker(job) -> List[Dict[str, Any]]:
    warnings.filterwarnings("ignore")
    logging.getLogger("qiskit_aer.noise.noise_model").setLevel(logging.ERROR)
    return run_one_instance(*job)


def run_architecture(architecture: str, n_instances: int, seed_offset: int,
                     n_workers: int, cfg: Any = None) -> List[Dict[str, Any]]:
    jobs = [(architecture, i, seed_offset + i, cfg) for i in range(n_instances)]
    print(f"[nisq_benchmark] {architecture}: {n_instances} instances, seeds "
          f"{seed_offset}..{seed_offset + n_instances - 1}, {n_workers} workers", flush=True)

    results: List[Any] = [None] * n_instances
    if n_workers <= 1:
        for idx, job in enumerate(jobs):
            results[idx] = _worker(job)
    else:
        with mp.Pool(n_workers) as pool:
            for idx, rows in enumerate(pool.imap(_worker, jobs, chunksize=1)):
                results[idx] = rows
    if any(r is None for r in results):
        raise RuntimeError("a worker returned no rows; refusing to emit a partial CSV")
    return [row for rows in results for row in rows]


def validate_rows(rows: List[Dict[str, Any]], n_instances: int,
                  architectures: List[str]) -> List[str]:
    problems: List[str] = []
    by_arch: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_arch.setdefault(row["architecture"], []).append(row)

    if sorted(by_arch) != sorted(architectures):
        problems.append(f"architectures {sorted(by_arch)} != {sorted(architectures)}")

    for architecture, arch_rows in sorted(by_arch.items()):
        ids = sorted({r["instance_id"] for r in arch_rows})
        if ids != list(range(n_instances)):
            problems.append(f"{architecture}: instance_ids are not contiguous "
                            f"0..{n_instances - 1} (a gap means instances were dropped)")
        if len(arch_rows) != n_instances * len(PARAMETERS):
            problems.append(f"{architecture}: {len(arch_rows)} rows, expected "
                            f"{n_instances * len(PARAMETERS)}")
        per_instance: Dict[int, List[str]] = {}
        for r in arch_rows:
            per_instance.setdefault(r["instance_id"], []).append(r["parameter"])
        if any(sorted(p) != sorted(PARAMETERS) for p in per_instance.values()):
            problems.append(f"{architecture}: an instance does not carry exactly {PARAMETERS}")
        if len({r["seed"] for r in arch_rows}) != n_instances:
            problems.append(f"{architecture}: seeds are not unique per instance")
        budgets = {r["shots_total"] for r in arch_rows if r["shots_total"]}
        if len(budgets) > 1:
            problems.append(f"{architecture}: unequal shot budgets {sorted(budgets)}")
        if any(not str(r["fit_status"]).strip() for r in arch_rows):
            problems.append(f"{architecture}: a row has a blank fit_status")
        for flag in ("time_resolved", "t1_telegraph"):
            if len({r[flag] for r in arch_rows}) > 1:
                problems.append(f"{architecture}: mixed {flag} values across rows")

    keys = [(r["architecture"], r["instance_id"], r["parameter"]) for r in rows]
    if len(set(keys)) != len(keys):
        problems.append("duplicate (architecture, instance_id, parameter) rows")
    if len({(r["architecture"], r["seed"]) for r in rows}) != len(architectures) * n_instances:
        problems.append("architecture/seed pairs are not unique")
    return problems


def write_csv(rows: List[Dict[str, Any]], path: pathlib.Path) -> None:
    rank = {p: i for i, p in enumerate(PARAMETERS)}
    ordered = sorted(rows, key=lambda r: (r["architecture"], r["instance_id"],
                                          rank.get(r["parameter"], 99)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(ordered)
    print(f"[nisq_benchmark] wrote {len(ordered)} rows -> {path}", flush=True)


def summarize(rows: List[Dict[str, Any]]) -> None:
    print("\n[nisq_benchmark] median relative error, all instances, nothing removed")
    by_arch: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_arch.setdefault(r["architecture"], []).append(r)
    for architecture, arch_rows in sorted(by_arch.items()):
        n_inst = len({r["instance_id"] for r in arch_rows})
        n_bad = len({r["instance_id"] for r in arch_rows if not r["success"]})
        print(f"  {architecture} ({n_inst} instances, {n_bad} with a failed parameter)")
        for param in PARAMETERS:
            vals = np.array([r["relative_error"] for r in arch_rows
                             if r["parameter"] == param], dtype=float)
            finite = vals[np.isfinite(vals)]
            med = float(np.median(finite)) * 100 if len(finite) else NAN
            print(f"     {param:12s} {med:8.2f}%   ({len(finite)}/{len(vals)} finite)")
        n_aliased = sum(1 for r in arch_rows
                        if r["parameter"] == "delta_omega" and r["dw_aliased"] is True)
        print(f"     dw_aliased: {n_aliased}/{n_inst} instances")


def _read_csv(path: pathlib.Path) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError(f"{path.name}: columns {reader.fieldnames} != {CSV_COLUMNS}")
        rows = []
        for raw in reader:
            row = dict(raw)
            for key in ("instance_id", "seed", "shots_total"):
                row[key] = int(row[key])
            row["success"] = str(row["success"]).strip().lower() == "true"
            for key in ("true_value", "canary_value", "absolute_error", "relative_error",
                        "sigma_canary", "chi2_dof", "chi2_dof_secondary",
                        "prior_value", "prior_absolute_error", "prior_relative_error",
                        "perturbation_frac", "spam_scale", "spam_spread", "t_measure_s",
                        "coherent_fraction", "crosstalk_scale", "alpha",
                        "run_duration_s", "truth_rel_spread"):
                row[key] = float(row[key]) if row[key] not in ("", None) else NAN
            for key in ("spam_swap", "cold_start", "detuning_1_over_f",
                        "time_resolved", "t1_telegraph"):
                row[key] = str(row[key]).strip().lower() == "true"
            flag = str(row["dw_aliased"]).strip().lower()
            row["dw_aliased"] = {"true": True, "false": False}.get(flag, NAN)
            rows.append(row)
    return rows


def combine(inputs: List[pathlib.Path], out: pathlib.Path, n_instances: int) -> None:
    if not inputs:
        sys.exit("--combine requires at least one --input")
    rows: List[Dict[str, Any]] = []
    for path in inputs:
        if not path.exists():
            sys.exit(f"missing input CSV: {path}")
        rows.extend(_read_csv(path))

    architectures = sorted({r["architecture"] for r in rows})
    problems = validate_rows(rows, n_instances, architectures)
    expected = len(architectures) * n_instances * len(PARAMETERS)
    if len(rows) != expected:
        problems.append(f"aggregate row count {len(rows)} != {expected}")

    write_csv(rows, out)
    summarize(rows)
    _finish(problems, len(rows))


def _finish(problems: List[str], n_rows: int) -> None:
    if problems:
        print("\n[nisq_benchmark] VALIDATION FAILED:", flush=True)
        for p in problems:
            print(f"  - {p}", flush=True)
        sys.exit(1)
    print(f"\n[nisq_benchmark] validation passed: {n_rows} rows")


def main() -> None:
    ap = argparse.ArgumentParser(description="NISQ parameter-recovery benchmark.")
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--input", type=pathlib.Path, action="append", default=[])
    ap.add_argument("--architecture", choices=list(ARCHITECTURES) + ["all"])
    ap.add_argument("--n-instances", type=int, default=300)
    ap.add_argument("--seed-offset", type=int, default=None)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--perturbation-frac", type=float, default=None,
                    help="calibration-prior envelope half-width f (default: 1c table); "
                         "also sets the matching generate_device dw ceiling")
    ap.add_argument("--spam-scale", type=float, default=1.0,
                    help="scale the SPAM handed to the inversion (truth untouched)")
    ap.add_argument("--spam-swap", action="store_true",
                    help="swap p0|1 and p1|0 handed to the inversion (truth untouched)")
    ap.add_argument("--spam-spread", type=float, default=0.0,
                    help="generate_device(spam_spread_frac=...)")
    ap.add_argument("--cold-start", action="store_true",
                    help="use the architecture-default profile instead of the prior")
    ap.add_argument("--dw-population", choices=["estimator_window", "independent"],
                    default="estimator_window",
                    help="generate_device(dw_population=...)")
    ap.add_argument("--t-measure", type=float, default=T_MEASURE_S,
                    help="simulation/truth-evaluation time in s (prior is at T_CALIB_S = 0)")
    ap.add_argument("--coherent-fraction", type=float, default=None,
                    help="override the truth coherent_fraction (default: 1c registry, 0.10)")
    ap.add_argument("--crosstalk-scale", type=float, default=1.0,
                    help="generate_device(crosstalk_scale=...); no effect in this "
                         "single-qubit benchmark")
    ap.add_argument("--detuning-1-over-f", action="store_true",
                    help="superconducting only: 1/f^alpha detuning drift")
    ap.add_argument("--alpha", type=float, default=None,
                    help="1/f^alpha exponent override; requires --detuning-1-over-f")
    ap.add_argument("--time-resolved", action="store_true",
                    help="opt-in time-resolved probe run starting at --t-measure "
                         "(default: truth frozen at --t-measure for every circuit "
                         "and shot; trapped-ion effect set by a Class C correlation time)")
    ap.add_argument("--t1-telegraph", action="store_true",
                    help="superconducting only: opt-in two-state T1 switching "
                         "(Class C switching amplitude; off by default)")
    args = ap.parse_args()

    if args.n_instances <= 0:
        sys.exit("--n-instances must be positive")
    if args.perturbation_frac is not None and not 0.0 <= args.perturbation_frac < 1.0:
        sys.exit("--perturbation-frac must be in [0, 1)")
    if args.spam_scale < 0.0:
        sys.exit("--spam-scale must be non-negative")
    if not 0.0 <= args.spam_spread < 1.0:
        sys.exit("--spam-spread must be in [0, 1)")
    if args.t_measure < 0.0:
        sys.exit("--t-measure must be non-negative")
    if args.coherent_fraction is not None and not 0.0 <= args.coherent_fraction <= 1.0:
        sys.exit("--coherent-fraction must be in [0, 1]")
    if args.crosstalk_scale < 0.0:
        sys.exit("--crosstalk-scale must be non-negative")
    if args.alpha is not None and not args.detuning_1_over_f:
        sys.exit("--alpha is only valid with --detuning-1-over-f")
    if args.detuning_1_over_f and args.architecture in ("trapped_ion", "all"):
        sys.exit("--detuning-1-over-f is superconducting only "
                 "(use --architecture superconducting)")
    if args.t1_telegraph and args.architecture in ("trapped_ion", "all"):
        sys.exit("--t1-telegraph is superconducting only "
                 "(use --architecture superconducting)")
    cfg = {"perturbation_frac": args.perturbation_frac, "spam_scale": args.spam_scale,
           "spam_swap": args.spam_swap, "spam_spread": args.spam_spread,
           "cold_start": args.cold_start, "dw_population": args.dw_population,
           "t_measure_s": args.t_measure, "coherent_fraction": args.coherent_fraction,
           "crosstalk_scale": args.crosstalk_scale,
           "detuning_1_over_f": args.detuning_1_over_f, "alpha": args.alpha,
           "time_resolved": args.time_resolved, "t1_telegraph": args.t1_telegraph}
    if args.combine:
        combine(args.input, args.out, args.n_instances)
        return
    if args.architecture is None:
        sys.exit("--architecture is required (or use --combine)")

    architectures = list(ARCHITECTURES) if args.architecture == "all" else [args.architecture]
    if args.seed_offset is not None and len(architectures) > 1:
        sys.exit("--seed-offset cannot be combined with --architecture all")

    n_workers = args.workers if args.workers is not None else max(1, mp.cpu_count() - 1)
    started = datetime.now(timezone.utc)

    rows: List[Dict[str, Any]] = []
    for architecture in architectures:
        offset = (args.seed_offset if args.seed_offset is not None
                  else ARCH_SEED_OFFSET[architecture])
        rows.extend(run_architecture(architecture, args.n_instances, offset, n_workers, cfg))

    problems = validate_rows(rows, args.n_instances, architectures)
    write_csv(rows, args.out)
    summarize(rows)
    print(f"[nisq_benchmark] elapsed "
          f"{(datetime.now(timezone.utc) - started).total_seconds():.1f}s")
    _finish(problems, len(rows))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.getLogger("qiskit_aer.noise.noise_model").setLevel(logging.ERROR)
    if sys.platform != "win32":
        try:
            mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    main()