"""NISQ benchmark: Canary vs qiskit-experiments on the same simulated qubits.

Each instance is one simulated qubit with every noise mechanism the simulator
has switched on (crosstalk excepted, since there is only one qubit): T1 drift
and telegraph switching, 1/f and OU frequency drift, quasi-static dephasing,
a residual detuning that also acts during pulses, coherent over-rotation,
readout error that varies between qubits, and motional heating for ions.
The calibration prior is taken an hour before the run and is +/-25% (SC) or
+/-15% (ions) off, readout errors included. Circuits run in simulated real
time, so the qubit keeps drifting while they execute.

Canary gets the counts and the prior. qiskit-experiments (T1, T2Hahn,
T2Ramsey, StandardRB) runs on the same qubit from the same start time, with
its settings picked from the same prior. Each tool is scored against the
truth averaged over its own run.

Truth definitions: T2 is the echo T2, delta_omega is signed, and epsilon is
the average gate infidelity of the simulated sx, 1 - F_avg, the quantity
randomized benchmarking estimates and IBM and IonQ report. Canary's
inverse-pair probe does not see coherent errors, so its epsilon reads low by
the coherent share. Read R2 next to R2_prior, beats_prior and the |z|
coverage. Output is one CSV with both tools side by side.
"""

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
REPO_SRC = SCRIPT_DIR.parent / "src"
if REPO_SRC.is_dir():
    sys.path.insert(0, str(REPO_SRC))

from quantum_canary import inversion as inv
from quantum_canary import simulation as sim

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
    "gate_detuning", "quasi_static", "spam_perturbation_frac", "dw_halfwidth_hz",
    "t2_ramsey_over_echo", "dw_true_rad_s",
    "spam_p0_given_1_true", "spam_p1_given_0_true",
    "spam_p0_given_1_used", "spam_p1_given_0_used",
    "signed_error", "z_score", "probe_blocks",
    "quasi_static_scale", "detuning_drift_scale", "t1_drift_scale",
    "heating_gate_error",
    "qe_value", "qe_sigma", "qe_true_value", "qe_absolute_error",
    "qe_relative_error", "qe_signed_error", "qe_z_score", "qe_quality",
    "qe_fit_status", "qe_quantity", "qe_shots_total", "qe_run_duration_s",
]
NAN = float("nan")

QE_BASIS = ["rz", "sx", "x"]
QE_SHOTS = {"t1": 500, "ramsey": 500, "hahn": 500, "rb": 200}
QE_N_DELAYS = {"t1": 20, "ramsey": 40, "hahn": 20}
QE_T1_SPAN, QE_RAMSEY_SPAN, QE_HAHN_SPAN = 3.0, 2.0, 3.0
QE_RAMSEY_OSC = 4.0
QE_RB_N_LENGTHS, QE_RB_SAMPLES = 10, 10
QE_RB_LMIN, QE_RB_LMAX = 50, 2000
QE_SEED_BASE = 1_000_000_000
QE_FAMILY_OF = {"T1_s": "t1", "T2_s": "hahn", "delta_omega": "ramsey", "epsilon_sx": "rb"}
QE_QUANTITY = {"T1_s": "T1 (qiskit_experiments T1)",
               "T2_s": "T2 (qiskit_experiments T2Hahn, 1 echo)",
               "delta_omega": "2*pi*(Frequency - osc_freq) (qiskit_experiments T2Ramsey)",
               "epsilon_sx": "EPG_sx (qiskit_experiments StandardRB)"}

DEFAULT_CFG: Dict[str, Any] = {
    "perturbation_frac": None,
    "spam_scale": 1.0,
    "spam_swap": False,
    "spam_spread": 0.5,
    "cold_start": False,
    "dw_population": "literature",
    "t_measure_s": T_MEASURE_S,
    "coherent_fraction": None,
    "crosstalk_scale": 1.0,
    "detuning_1_over_f": True,
    "alpha": None,
    "time_resolved": True,
    "t1_telegraph": True,
    "heating_gate_error": True,
    "qiskit_experiments": True,
    "gate_detuning": True,
    "quasi_static": True,
    "spam_perturbation_frac": None,
    "dw_halfwidth_hz": None,
    "probe_blocks": 10,
    "quasi_static_scale": 1.0,
    "detuning_drift_scale": 1.0,
    "t1_drift_scale": 1.0,
}
TRUTH_FAMILIES = {"T1_s": ("t1_",), "T2_s": ("ramsey_X_", "ramsey_Y_", "echo_"),
                  "delta_omega": ("ramsey_X_", "ramsey_Y_"),
                  "epsilon_sx": ("gate_rep_N",)}
SPAM_CLIP_MAX = 0.499


def _avg_gate_infidelity(truth: Any, t_s: float) -> float:
    """1 - F_avg of qubit 0's sx as the simulator plays it at t_s: relaxation and
    depolarizing after the pulse, the amplitude over-rotation, and the detuning
    during the pulse averaged over the quasi-static offset."""
    from qiskit.circuit.library import SXGate
    from qiskit.quantum_info import Operator, SuperOp, average_gate_fidelity
    err, f = sim._gate_noise_channel(truth, 0, t_s)
    g = Operator(SXGate())
    core = SuperOp(g).compose(SuperOp(err))
    if f > 0.0:
        core = core.compose(SuperOp(g.power(f)))
    during = bool(truth.architecture_specific.get("detuning_during_gates", False))
    dw = sim.evaluate_truth_at(truth, 0, t_s)["delta_omega_rad_s"] if during else 0.0
    qs = truth.architecture_specific.get("quasi_static_dw_std_rad_s")
    std = float(np.asarray(qs, dtype=float)[0]) if (during and qs is not None) else 0.0
    nodes, weights = np.polynomial.hermite_e.hermegauss(9)
    t_g = float(truth.gate_duration_ns) * 1e-9
    channel = None
    for x, w in zip(nodes, weights / weights.sum()):
        a = (dw + std * x) * t_g / 2.0
        rz = SuperOp(Operator(np.diag([np.exp(-0.5j * a), np.exp(0.5j * a)])))
        term = rz.compose(core).compose(rz) * w
        channel = term if channel is None else channel + term
    return 1.0 - float(average_gate_fidelity(channel, target=g))


def _abs_error(value: float, truth_value: float) -> float:
    if not np.isfinite(value) or not np.isfinite(truth_value):
        return NAN
    return abs(value - truth_value)


def _rel_error(value: float, truth_value: float) -> float:
    if not np.isfinite(value) or not np.isfinite(truth_value) or truth_value == 0:
        return NAN
    return abs(value - truth_value) / abs(truth_value)


def _signed_error(value: float, truth_value: float) -> float:
    if not np.isfinite(value) or not np.isfinite(truth_value):
        return NAN
    return value - truth_value


def _z_score(value: float, truth_value: float, sigma: float) -> float:
    """(estimate - truth) / sigma, or NaN if that can't be computed."""
    if not (np.isfinite(value) and np.isfinite(truth_value)
            and np.isfinite(sigma) and sigma > 0):
        return NAN
    return (value - truth_value) / sigma


def _r2(truth: np.ndarray, estimate: np.ndarray) -> float:
    """Coefficient of determination of estimate against truth."""
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    if ss_tot <= 0.0:
        return NAN
    return 1.0 - float(np.sum((estimate - truth) ** 2)) / ss_tot


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


def _effective_mechanisms(architecture: str, c: Dict[str, Any]) -> Dict[str, bool]:
    """Which architecture-specific mechanisms actually apply to this qubit."""
    sc = architecture == "superconducting"
    return {"detuning_1_over_f": bool(c["detuning_1_over_f"]) and sc,
            "t1_telegraph": bool(c["t1_telegraph"]) and sc,
            "heating_gate_error": bool(c["heating_gate_error"]) and not sc}


def _config_columns(architecture: str, c: Dict[str, Any]) -> Dict[str, Any]:
    """Config values written to every row."""
    extra = sim.get_architecture_profile(architecture).extra
    eff = _effective_mechanisms(architecture, c)
    pf = c["perturbation_frac"]
    if pf is None:
        pf = sim.resolve_calibration_perturbation(architecture, REGIME)
    cf = c["coherent_fraction"]
    if cf is None:
        cf = extra["coherent_fraction"]
    alpha = NAN
    if eff["detuning_1_over_f"]:
        alpha = (c["alpha"] if c["alpha"] is not None
                 else extra.get("detuning_1_over_f_alpha", NAN))
    spf = c["spam_perturbation_frac"]
    if spf is None:
        spf = pf
    w_hz = NAN
    if c["dw_population"] == "literature":
        key = "sc" if architecture == "superconducting" else "ti"
        w_hz = (c["dw_halfwidth_hz"] if c["dw_halfwidth_hz"] is not None
                else sim.MODEL_PROVENANCE[f"{key}.dw_population.literature_halfwidth_hz"]["value"])
    return {"gate_detuning": bool(c["gate_detuning"]),
            "quasi_static": bool(c["quasi_static"]) and architecture == "superconducting",
            "spam_perturbation_frac": float(spf), "dw_halfwidth_hz": float(w_hz),
            "perturbation_frac": float(pf), "spam_scale": float(c["spam_scale"]),
            "spam_swap": bool(c["spam_swap"]), "spam_spread": float(c["spam_spread"]),
            "cold_start": bool(c["cold_start"]), "dw_population": str(c["dw_population"]),
            "t_measure_s": float(c["t_measure_s"]), "coherent_fraction": float(cf),
            "crosstalk_scale": float(c["crosstalk_scale"]),
            "detuning_1_over_f": eff["detuning_1_over_f"], "alpha": float(alpha),
            "time_resolved": bool(c["time_resolved"]),
            "t1_telegraph": eff["t1_telegraph"],
            "heating_gate_error": eff["heating_gate_error"],
            "probe_blocks": int(c["probe_blocks"]),
            "quasi_static_scale": float(c["quasi_static_scale"]),
            "detuning_drift_scale": float(c["detuning_drift_scale"]),
            "t1_drift_scale": float(c["t1_drift_scale"])}


def _inversion_spam(prior: Any, c: Dict[str, Any]):
    """Readout errors given to Canary: the prior's, scaled, optionally swapped, clipped."""
    p01 = prior.spam_p0_given_1_prior * c["spam_scale"]
    p10 = prior.spam_p1_given_0_prior * c["spam_scale"]
    if c["spam_swap"]:
        p01, p10 = p10, p01
    return (float(np.clip(p01, 0.0, SPAM_CLIP_MAX)),
            float(np.clip(p10, 0.0, SPAM_CLIP_MAX)))


def _apply_mechanism_scales(truth: Any, architecture: str, c: Dict[str, Any]) -> None:
    """Scale quasi-static dephasing, frequency drift and T1 drift (all 1.0 by default)."""
    qs, dd, td = (float(c["quasi_static_scale"]), float(c["detuning_drift_scale"]),
                  float(c["t1_drift_scale"]))
    if min(qs, dd, td) < 0.0:
        raise ValueError("mechanism scales must be non-negative")
    if architecture != "superconducting" and (qs != 1.0 or td != 1.0):
        raise ValueError("quasi_static_scale and t1_drift_scale are superconducting-only")
    spec = truth.architecture_specific
    if qs != 1.0 and "quasi_static_dw_std_rad_s" in spec:
        spec["quasi_static_dw_std_rad_s"] = qs * np.asarray(spec["quasi_static_dw_std_rad_s"],
                                                            dtype=float)
    if dd != 1.0 and "detuning" in truth.drift_processes:
        params = truth.drift_processes["detuning"].params
        key = "sigma" if truth.drift_processes["detuning"].process_type == "ou" else "amplitude"
        params[key] = dd * float(params[key])
    if td != 1.0 and "T1" in truth.drift_processes:
        params = truth.drift_processes["T1"].params
        params["sigma"] = td * float(params["sigma"])


def _time_resolved_truth(truth: Any, sim_result: Any):
    """Truth for each parameter, averaged over the shots of the circuits that measure it."""
    true_values: Dict[str, float] = {}
    spread: Dict[str, float] = {}
    for param, prefixes in TRUTH_FAMILIES.items():
        vals, weights = [], []
        for name, recs in zip(sim_result.circuit_names, sim_result.chunk_records):
            if not name.startswith(prefixes):
                continue
            for rec in recs:
                if param == "epsilon_sx":
                    vals.append(_avg_gate_infidelity(truth, rec["t_mid_s"]))
                else:
                    key = {"T1_s": "T1_s", "T2_s": "T2_s",
                           "delta_omega": "delta_omega_rad_s"}[param]
                    vals.append(float(rec["truth"][0][key]))
                weights.append(float(rec["shots"]))
        v, w = np.array(vals, dtype=float), np.array(weights, dtype=float)
        mean = float(v[0] + np.sum(w * (v - v[0])) / np.sum(w))
        spread[param] = float((v.max() - v.min()) / abs(mean)) if mean != 0 else NAN
        true_values[param] = mean
    return true_values, spread


def _qe_experiments(prior: Any, seed: int):
    """Build the four qiskit-experiments experiments from the prior."""
    from qiskit_experiments.library import T1, T2Ramsey, T2Hahn, StandardRB
    t1p, t2p, epsp = float(prior.T1_prior_s), float(prior.T2_prior_s), float(prior.epsilon_prior)
    osc = QE_RAMSEY_OSC / t2p
    lmax = max(QE_RB_LMIN, min(QE_RB_LMAX, int(1.0 / (2.0 * epsp))))
    lengths = sorted({int(round(x)) for x in np.geomspace(1, lmax, QE_RB_N_LENGTHS)})
    grid = lambda span, n: [float(x) for x in np.linspace(0.0, span, n + 1)[1:]]
    exps = {
        "t1": T1((0,), delays=grid(QE_T1_SPAN * t1p, QE_N_DELAYS["t1"])),
        "ramsey": T2Ramsey((0,), delays=grid(QE_RAMSEY_SPAN * t2p, QE_N_DELAYS["ramsey"]),
                           osc_freq=osc),
        "hahn": T2Hahn((0,), delays=grid(QE_HAHN_SPAN * t2p, QE_N_DELAYS["hahn"]),
                       num_echoes=1),
        "rb": StandardRB((0,), lengths=lengths, num_samples=QE_RB_SAMPLES, seed=int(seed)),
    }
    for exp in exps.values():
        exp.set_transpile_options(basis_gates=QE_BASIS, optimization_level=0)
        exp.analysis.set_options(plot=False)
    return exps, osc


def _ufloat_parts(value: Any):
    """Value and error of a qiskit-experiments result."""
    if hasattr(value, "nominal_value"):
        return float(value.nominal_value), float(value.std_dev)
    try:
        return float(value), NAN
    except (TypeError, ValueError):
        return NAN, NAN


def _run_qiskit_experiments(truth: Any, prior: Any, seed: int, t_measure_s: float,
                            time_resolved: bool) -> Dict[str, Any]:
    """Run qiskit-experiments on the same simulated qubit and score it against its own window."""
    from qiskit_experiments.framework import ExperimentData
    exps, osc = _qe_experiments(prior, seed)
    circuits, shots, family = [], [], []
    for fam, exp in exps.items():
        tc = exp._transpiled_circuits()
        circuits.extend(tc)
        shots.extend([QE_SHOTS[fam]] * len(tc))
        family.extend([fam] * len(tc))
    seed_base = int(seed) * 1000 + QE_SEED_BASE
    if len(circuits) > 1000 or seed_base + len(circuits) >= 2**31 - 1:
        raise RuntimeError("qiskit-experiments Aer seeds would overlap or overflow")
    res = sim.simulate_probe(circuits, truth, shots=shots, t_s=t_measure_s, dt_ns=None,
                             seed=seed_base, time_resolved=time_resolved)

    results: Dict[str, Dict[str, Any]] = {}
    for fam, exp in exps.items():
        idx = [i for i, f in enumerate(family) if f == fam]
        data = ExperimentData(experiment=exp)
        data.add_data([{"counts": res.counts_list[i], "metadata": circuits[i].metadata,
                        "shots": shots[i]} for i in idx])
        exp.analysis.run(data, replace_results=True).block_for_results()
        frame = data.analysis_results(dataframe=True)
        results[fam] = {str(name): (row["value"], row["quality"])
                        for name, row in zip(frame["name"], frame.to_dict("records"))} \
            if len(frame) else {}

    def _truth(param: str) -> float:
        fam = QE_FAMILY_OF[param]
        idx = [i for i, f in enumerate(family) if f == fam]
        if not time_resolved:
            if param == "epsilon_sx":
                return float(_avg_gate_infidelity(truth, t_measure_s))
            key = {"T1_s": "T1_s", "T2_s": "T2_s", "delta_omega": "delta_omega_rad_s"}[param]
            return float(sim.evaluate_truth_at(truth, 0, t_measure_s)[key])
        vals, weights = [], []
        for i in idx:
            for rec in res.chunk_records[i]:
                if param == "epsilon_sx":
                    vals.append(_avg_gate_infidelity(truth, rec["t_mid_s"]))
                else:
                    key = {"T1_s": "T1_s", "T2_s": "T2_s",
                           "delta_omega": "delta_omega_rad_s"}[param]
                    vals.append(float(rec["truth"][0][key]))
                weights.append(float(rec["shots"]))
        v, w = np.array(vals, dtype=float), np.array(weights, dtype=float)
        return float(v[0] + np.sum(w * (v - v[0])) / np.sum(w))

    pick = {"T1_s": ("t1", "T1"), "T2_s": ("hahn", "T2"),
            "delta_omega": ("ramsey", "Frequency"), "epsilon_sx": ("rb", "EPG_sx")}
    out: Dict[str, Any] = {"shots_total": int(sum(shots)),
                           "run_duration_s": float(res.run_duration_s) if time_resolved
                           else float(sim.probe_run_duration_s(circuits, truth, shots,
                                                               t_s=t_measure_s))}
    for param, (fam, name) in pick.items():
        entry = results[fam].get(name)
        if entry is None:
            value, sigma, quality, status = NAN, NAN, "", "qe_no_result"
        else:
            value, sigma = _ufloat_parts(entry[0])
            quality = str(entry[1])
            if param == "delta_omega":
                value, sigma = 2.0 * np.pi * (value - osc), 2.0 * np.pi * sigma
            status = "ok" if np.isfinite(value) and np.isfinite(sigma) else "qe_non_finite"
        out[param] = {"value": value, "sigma": sigma, "quality": quality,
                      "status": status, "truth": _truth(param)}
    return out


def run_one_instance(architecture: str, instance_id: int, seed: int,
                     cfg: Any = None) -> List[Dict[str, Any]]:
    timestamp = datetime.now(timezone.utc).isoformat()
    c = _resolve_cfg(cfg)
    t_measure_s = c["t_measure_s"]
    cfg_cols = _config_columns(architecture, c)
    eff = _effective_mechanisms(architecture, c)
    try:
        if c["alpha"] is not None and not eff["detuning_1_over_f"]:
            raise ValueError("alpha requires detuning_1_over_f (superconducting only)")
        if c["dw_halfwidth_hz"] is not None and c["dw_population"] != "literature":
            raise ValueError("dw_halfwidth_hz requires dw_population='literature'")
        overrides = ({"dw_halfwidth_hz": float(c["dw_halfwidth_hz"])}
                     if c["dw_halfwidth_hz"] is not None else None)
        truth = sim.generate_device(architecture, n_qubits=1, seed=seed, regime=REGIME,
                                    overrides=overrides,
                                    crosstalk_scale=c["crosstalk_scale"],
                                    detuning_1_over_f=eff["detuning_1_over_f"],
                                    enable_heating_gate_error=eff["heating_gate_error"],
                                    spam_spread_frac=c["spam_spread"],
                                    perturbation_frac=c["perturbation_frac"],
                                    dw_population=c["dw_population"],
                                    t1_telegraph=eff["t1_telegraph"],
                                    quasi_static_dephasing=c["quasi_static"],
                                    detuning_during_gates=c["gate_detuning"])
        if c["coherent_fraction"] is not None:
            truth.architecture_specific["coherent_fraction"] = float(c["coherent_fraction"])
        _apply_mechanism_scales(truth, architecture, c)
        if c["alpha"] is not None:
            truth.drift_processes["detuning"].params["alpha"] = float(c["alpha"])
        prior = sim.generate_calibration_prior(
            truth, qubit=0, t0_s=T_CALIB_S, perturbation_frac=c["perturbation_frac"],
            spam_perturbation_frac=cfg_cols["spam_perturbation_frac"])
        p0_given_1, p1_given_0 = _inversion_spam(prior, c)

        if c["cold_start"]:
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

        block_circuits, block_shots, block_index = inv.interleave_probe_blocks(
            circuits, shots_list, int(c["probe_blocks"]))
        if len(block_circuits) > 1000:
            raise RuntimeError(f"{len(block_circuits)} block entries would reuse Aer seeds")
        sim_result = sim.simulate_probe(block_circuits, truth, shots=block_shots,
                                        t_s=t_measure_s, dt_ns=profile.dt_ns,
                                        seed=int(seed) * 1000,
                                        time_resolved=c["time_resolved"])
        counts_list = inv.merge_block_counts(sim_result.counts_list, block_index,
                                             len(circuits))
        result = inv.lindblad_inversion(
            counts_list, meta, profile,
            shots_t1=SHOTS_T1, shots_ramsey=SHOTS_RAMSEY, shots_gate=SHOTS_GATE,
            shots_echo=SHOTS_ECHO, qubit_id=0, timestamp=timestamp)

        tv_now = sim.evaluate_truth_at(truth, 0, t_measure_s)
        if c["time_resolved"]:
            true_values, truth_spread = _time_resolved_truth(truth, sim_result)
            run_duration_s = float(sim_result.run_duration_s)
        else:
            true_values = {"T1_s": float(tv_now["T1_s"]),
                           "T2_s": float(tv_now["T2_s"]),
                           "delta_omega": float(tv_now["delta_omega_rad_s"]),
                           "epsilon_sx": _avg_gate_infidelity(truth, t_measure_s)}
            truth_spread = dict.fromkeys(PARAMETERS, 0.0)
            run_duration_s = sim.probe_run_duration_s(block_circuits, truth, block_shots,
                                                      t_s=t_measure_s, dt_ns=profile.dt_ns)
        canary = {"T1_s": float(result.T1_s),
                  "T2_s": float(result.T2_s),
                  "delta_omega": float(result.delta_omega),
                  "epsilon_sx": float(result.epsilon_sx)}
        sigmas = {"T1_s": float(result.T1_sigma_s),
                  "T2_s": float(result.T2_sigma_s),
                  "delta_omega": float(result.delta_omega_sigma),
                  "epsilon_sx": float(result.epsilon_sx_sigma)}
        chi2 = {"T1_s": float(result.t1_chi2_dof),
                "T2_s": float(result.echo_chi2_dof),
                "delta_omega": float(result.ramsey_chi2_dof),
                "epsilon_sx": float(result.gate_chi2_dof)}
        chi2_secondary = {"T2_s": float(result.ramsey_chi2_dof)}
        priors = {"T1_s": float(prior.T1_prior_s),
                  "T2_s": float(prior.T2_prior_s),
                  "delta_omega": NAN,
                  "epsilon_sx": float(prior.epsilon_prior)}
        dw_aliased = {"delta_omega": bool(abs(true_values["delta_omega"])
                                          * meta["ramsey_delays_s"][0] > np.pi)}
        instance_cols = {
            "t2_ramsey_over_echo": float(np.asarray(truth.architecture_specific.get(
                "t2_ramsey_over_echo", [NAN]), dtype=float)[0]),
            "dw_true_rad_s": float(true_values["delta_omega"]),
            "spam_p0_given_1_true": float(tv_now["spam_p0_given_1"]),
            "spam_p1_given_0_true": float(tv_now["spam_p1_given_0"]),
            "spam_p0_given_1_used": float(p0_given_1),
            "spam_p1_given_0_used": float(p1_given_0),
        }

        qe_cols: Dict[str, Dict[str, Any]] = {p: {} for p in PARAMETERS}
        if c["qiskit_experiments"]:
            try:
                qe = _run_qiskit_experiments(truth, prior, seed, t_measure_s,
                                             bool(c["time_resolved"]))
                for param in PARAMETERS:
                    q = qe[param]
                    qe_cols[param] = {
                        "qe_value": q["value"], "qe_sigma": q["sigma"],
                        "qe_true_value": q["truth"],
                        "qe_absolute_error": _abs_error(q["value"], q["truth"]),
                        "qe_relative_error": _rel_error(q["value"], q["truth"]),
                        "qe_signed_error": _signed_error(q["value"], q["truth"]),
                        "qe_z_score": _z_score(q["value"], q["truth"], q["sigma"]),
                        "qe_quality": q["quality"], "qe_fit_status": q["status"],
                        "qe_quantity": QE_QUANTITY[param],
                        "qe_shots_total": qe["shots_total"],
                        "qe_run_duration_s": qe["run_duration_s"]}
            except Exception as exc:
                for param in PARAMETERS:
                    qe_cols[param] = {"qe_fit_status": f"qe_exception:{type(exc).__name__}:{exc}",
                                      "qe_quantity": QE_QUANTITY[param]}
        else:
            for param in PARAMETERS:
                qe_cols[param] = {"qe_fit_status": "not_run"}

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
                dw_aliased=dw_aliased.get(param, NAN), **cfg_cols, **instance_cols,
                run_duration_s=run_duration_s, truth_rel_spread=truth_spread[param],
                signed_error=_signed_error(cv, tv), z_score=_z_score(cv, tv, sg),
                **qe_cols[param]))
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
                     n_workers: int, cfg: Any = None,
                     instance_start: int = 0) -> List[Dict[str, Any]]:
    """Run instances instance_start to instance_start + n_instances - 1."""
    ids = range(instance_start, instance_start + n_instances)
    jobs = [(architecture, i, seed_offset + i, cfg) for i in ids]
    print(f"[nisq_benchmark] {architecture}: instances {ids.start}..{ids.stop - 1}, seeds "
          f"{seed_offset + ids.start}..{seed_offset + ids.stop - 1}, {n_workers} workers",
          flush=True)

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
                  architectures: List[str], instance_start: int = 0) -> List[str]:
    problems: List[str] = []
    by_arch: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_arch.setdefault(row["architecture"], []).append(row)

    if sorted(by_arch) != sorted(architectures):
        problems.append(f"architectures {sorted(by_arch)} != {sorted(architectures)}")

    expected_ids = list(range(instance_start, instance_start + n_instances))
    for architecture, arch_rows in sorted(by_arch.items()):
        ids = sorted({r["instance_id"] for r in arch_rows})
        if ids != expected_ids:
            problems.append(f"{architecture}: instance_ids are not contiguous "
                            f"{expected_ids[0]}..{expected_ids[-1]} (a gap means "
                            f"instances were dropped)")
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
        for flag in ("time_resolved", "t1_telegraph", "gate_detuning", "quasi_static",
                     "dw_population", "probe_blocks", "detuning_1_over_f",
                     "heating_gate_error", "spam_spread"):
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
    """Print errors, R2 against the prior-only R2, |z| coverage, and the qiskit-experiments comparison."""
    print("\n[nisq_benchmark] summary, all instances, nothing removed "
          "(R2 and errors over converged instances; failures counted)")
    print("  R2(prior) is the score of the calibration prior with NO measurement: "
          "compare against it, never read R2 alone.")
    by_arch: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_arch.setdefault(r["architecture"], []).append(r)

    def _fmt(x: float, pct: bool = False) -> str:
        if not np.isfinite(x):
            return "--"
        return f"{100 * x:.1f}%" if pct else f"{x:.3f}"

    for architecture, arch_rows in sorted(by_arch.items()):
        n_inst = len({r["instance_id"] for r in arch_rows})
        n_bad = len({r["instance_id"] for r in arch_rows if not r["success"]})
        print(f"  {architecture} ({n_inst} instances, {n_bad} with a failed parameter)")
        print(f"     {'parameter':12s} {'fail':>5s} {'medRE':>8s} {'medRE_prior':>11s} "
              f"{'beats_prior':>11s} {'R2':>7s} {'R2_prior':>8s} {'|z|<=1':>7s} {'|z|<=2':>7s}")
        for param in PARAMETERS:
            pr = [r for r in arch_rows if r["parameter"] == param]
            tv = np.array([r["true_value"] for r in pr], dtype=float)
            cv = np.array([r["canary_value"] for r in pr], dtype=float)
            pv = np.array([r["prior_value"] for r in pr], dtype=float)
            re = np.array([r["relative_error"] for r in pr], dtype=float)
            pre = np.array([r["prior_relative_error"] for r in pr], dtype=float)
            z = np.array([r["z_score"] for r in pr], dtype=float)
            ok = np.array([bool(r["success"]) for r in pr]) & np.isfinite(cv) & np.isfinite(tv)
            both = ok & np.isfinite(pv)
            fin_re = ok & np.isfinite(re)
            fin_pair = both & np.isfinite(re) & np.isfinite(pre)
            fin_z = np.isfinite(z)
            print(f"     {param:12s} {len(pr) - int(ok.sum()):5d} "
                  f"{_fmt(float(np.median(re[fin_re])) if fin_re.any() else NAN, True):>8s} "
                  f"{_fmt(float(np.median(pre[fin_pair])) if fin_pair.any() else NAN, True):>11s} "
                  f"{_fmt(float(np.mean(re[fin_pair] < pre[fin_pair])) if fin_pair.any() else NAN, True):>11s} "
                  f"{_fmt(_r2(tv[ok], cv[ok]) if ok.sum() > 1 else NAN):>7s} "
                  f"{_fmt(_r2(tv[both], pv[both]) if both.sum() > 1 else NAN):>8s} "
                  f"{_fmt(float(np.mean(np.abs(z[fin_z]) <= 1)) if fin_z.any() else NAN, True):>7s} "
                  f"{_fmt(float(np.mean(np.abs(z[fin_z]) <= 2)) if fin_z.any() else NAN, True):>7s}")
        n_aliased = sum(1 for r in arch_rows
                        if r["parameter"] == "delta_omega" and r["dw_aliased"] is True)
        durations = np.array([r["run_duration_s"] for r in arch_rows
                              if r["parameter"] == "T1_s"], dtype=float)
        durations = durations[np.isfinite(durations)]
        print(f"     dw_aliased: {n_aliased}/{n_inst} instances; probe run duration "
              f"(lower bound) median {_fmt(float(np.median(durations)) if len(durations) else NAN)} s, "
              f"max {_fmt(float(np.max(durations)) if len(durations) else NAN)} s")

        qe_rows = [r for r in arch_rows if r["parameter"] == "T1_s"]
        if not any(str(r.get("qe_fit_status", "")) not in ("", "not_run") for r in qe_rows):
            continue
        c_shots = np.array([r["shots_total"] for r in qe_rows], dtype=float)
        q_shots = np.array([r["qe_shots_total"] for r in qe_rows], dtype=float)
        q_dur = np.array([r["qe_run_duration_s"] for r in qe_rows], dtype=float)
        print(f"     qiskit-experiments, same instances (shots/qubit: Canary "
              f"{_fmt(float(np.nanmedian(c_shots)))}, qiskit-experiments "
              f"{_fmt(float(np.nanmedian(q_shots)))}; QE run duration median "
              f"{_fmt(float(np.nanmedian(q_dur)))} s)")
        print(f"     {'parameter':12s} {'fail':>5s} {'medRE_QE':>9s} {'medRE_Can':>9s} "
              f"{'Can<QE':>7s} {'R2_QE':>7s} {'R2_Can':>7s} {'QE|z|<=2':>9s}  quantity")
        for param in PARAMETERS:
            pr = [r for r in arch_rows if r["parameter"] == param]
            qt = np.array([r["qe_true_value"] for r in pr], dtype=float)
            qv = np.array([r["qe_value"] for r in pr], dtype=float)
            qre = np.array([r["qe_relative_error"] for r in pr], dtype=float)
            qz = np.array([r["qe_z_score"] for r in pr], dtype=float)
            tv = np.array([r["true_value"] for r in pr], dtype=float)
            cv = np.array([r["canary_value"] for r in pr], dtype=float)
            cre = np.array([r["relative_error"] for r in pr], dtype=float)
            qok = np.array([r["qe_fit_status"] == "ok" for r in pr]) & np.isfinite(qv) & np.isfinite(qt)
            cok = np.array([bool(r["success"]) for r in pr]) & np.isfinite(cv) & np.isfinite(tv)
            both = qok & cok & np.isfinite(qre) & np.isfinite(cre)
            fin_qz = np.isfinite(qz)
            print(f"     {param:12s} {len(pr) - int(qok.sum()):5d} "
                  f"{_fmt(float(np.median(qre[qok & np.isfinite(qre)])) if (qok & np.isfinite(qre)).any() else NAN, True):>9s} "
                  f"{_fmt(float(np.median(cre[both])) if both.any() else NAN, True):>9s} "
                  f"{_fmt(float(np.mean(cre[both] < qre[both])) if both.any() else NAN, True):>7s} "
                  f"{_fmt(_r2(qt[qok], qv[qok]) if qok.sum() > 1 else NAN):>7s} "
                  f"{_fmt(_r2(tv[cok], cv[cok]) if cok.sum() > 1 else NAN):>7s} "
                  f"{_fmt(float(np.mean(np.abs(qz[fin_qz]) <= 2)) if fin_qz.any() else NAN, True):>9s}  "
                  f"{QE_QUANTITY[param]}")


def _read_csv(path: pathlib.Path) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError(f"{path.name}: columns {reader.fieldnames} != {CSV_COLUMNS}")
        rows = []
        for raw in reader:
            row = dict(raw)
            for key in ("instance_id", "seed", "shots_total", "probe_blocks"):
                row[key] = int(row[key])
            row["success"] = str(row["success"]).strip().lower() == "true"
            for key in ("true_value", "canary_value", "absolute_error", "relative_error",
                        "sigma_canary", "chi2_dof", "chi2_dof_secondary",
                        "prior_value", "prior_absolute_error", "prior_relative_error",
                        "perturbation_frac", "spam_scale", "spam_spread", "t_measure_s",
                        "coherent_fraction", "crosstalk_scale", "alpha",
                        "quasi_static_scale", "detuning_drift_scale", "t1_drift_scale",
                        "run_duration_s", "truth_rel_spread",
                        "spam_perturbation_frac", "dw_halfwidth_hz", "t2_ramsey_over_echo",
                        "dw_true_rad_s", "spam_p0_given_1_true", "spam_p1_given_0_true",
                        "spam_p0_given_1_used", "spam_p1_given_0_used",
                        "signed_error", "z_score",
                        "qe_value", "qe_sigma", "qe_true_value", "qe_absolute_error",
                        "qe_relative_error", "qe_signed_error", "qe_z_score",
                        "qe_shots_total", "qe_run_duration_s"):
                row[key] = float(row[key]) if row[key] not in ("", None) else NAN
            for key in ("spam_swap", "cold_start", "detuning_1_over_f",
                        "time_resolved", "t1_telegraph", "gate_detuning", "quasi_static",
                        "heating_gate_error"):
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
                         "also sets the matching generate_device dw ceiling "
                         "(--dw-population estimator_window only)")
    ap.add_argument("--spam-scale", type=float, default=1.0,
                    help="scale the SPAM handed to the inversion (truth untouched)")
    ap.add_argument("--spam-swap", action="store_true",
                    help="swap p0|1 and p1|0 handed to the inversion (truth untouched)")
    ap.add_argument("--spam-spread", type=float, default=DEFAULT_CFG["spam_spread"],
                    help="per-instance readout-error heterogeneity, "
                         "generate_device(spam_spread_frac=...) (default 0.5; 0 = off)")
    ap.add_argument("--cold-start", action="store_true",
                    help="use the architecture-default profile instead of the prior")
    ap.add_argument("--dw-population", choices=["literature", "estimator_window", "independent"],
                    default="literature",
                    help="generate_device(dw_population=...); 'literature' (default) is "
                         "independent of the estimator window, 'estimator_window' is the "
                         "earlier estimator-tailored population")
    ap.add_argument("--dw-halfwidth-hz", type=float, default=None,
                    help="half-width W/2pi (Hz) of the 'literature' detuning population "
                         "(default: 1c registry, 15 kHz SC / 0.5 Hz TI)")
    ap.add_argument("--gate-detuning", action=argparse.BooleanOptionalAction, default=True,
                    help="detuning also acts during gate pulses (default on; "
                         "--no-gate-detuning to ablate)")
    ap.add_argument("--quasi-static", action=argparse.BooleanOptionalAction, default=True,
                    help="superconducting: per-shot quasi-static dephasing from measured "
                         "T2R/T2E ratios (default on; --no-quasi-static to ablate; "
                         "no effect for trapped ions)")
    ap.add_argument("--spam-perturbation-frac", type=float, default=None,
                    help="staleness of the SPAM in the calibration prior, truth*U(1+-f) "
                         "(default: the prior's own envelope; 0 = oracle SPAM)")
    ap.add_argument("--t-measure", type=float, default=T_MEASURE_S,
                    help="simulation/truth-evaluation time in s (prior is at T_CALIB_S = 0)")
    ap.add_argument("--coherent-fraction", type=float, default=None,
                    help="override the truth coherent_fraction (default: 1c registry, 0.10)")
    ap.add_argument("--crosstalk-scale", type=float, default=1.0,
                    help="generate_device(crosstalk_scale=...); no effect in this "
                         "single-qubit benchmark")
    ap.add_argument("--detuning-1-over-f", action=argparse.BooleanOptionalAction, default=True,
                    help="1/f^alpha detuning drift, applied to superconducting qubits "
                         "(default on; --no-detuning-1-over-f for the low-pass process)")
    ap.add_argument("--alpha", type=float, default=None,
                    help="1/f^alpha exponent override (superconducting, with "
                         "--detuning-1-over-f)")
    ap.add_argument("--time-resolved", action=argparse.BooleanOptionalAction, default=True,
                    help="time-resolved probe run starting at --t-measure (default on; "
                         "--no-time-resolved freezes the truth at --t-measure for every "
                         "circuit and shot)")
    ap.add_argument("--t1-telegraph", action=argparse.BooleanOptionalAction, default=True,
                    help="two-state T1 switching, applied to superconducting qubits "
                         "(Class C switching amplitude; default on)")
    ap.add_argument("--heating-gate-error", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="motional-heating gate error, applied to trapped ions "
                         "(1c enable_heating_gate_error, Class C mapping; default on)")
    ap.add_argument("--qiskit-experiments", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="also run qiskit-experiments (T1, T2Hahn, T2Ramsey, StandardRB) "
                         "on the same instances (default on)")
    ap.add_argument("--instance-start", type=int, default=0,
                    help="first instance id (for parallel shards; instance i uses seed "
                         "seed_offset + i)")
    ap.add_argument("--probe-blocks", type=int, default=DEFAULT_CFG["probe_blocks"],
                    help="block-interleaved execution: run the probe list this many "
                         "times with a share of each circuit's shots "
                         "(quantum_canary.interleave_probe_blocks; 1 = circuit-major)")
    args = ap.parse_args()

    if args.n_instances <= 0:
        sys.exit("--n-instances must be positive")
    if args.instance_start < 0:
        sys.exit("--instance-start must be non-negative")
    if args.probe_blocks < 1:
        sys.exit("--probe-blocks must be >= 1")
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
    if args.alpha is not None and (not args.detuning_1_over_f
                                   or args.architecture != "superconducting"):
        sys.exit("--alpha requires --detuning-1-over-f and --architecture superconducting")
    if args.spam_perturbation_frac is not None and not 0.0 <= args.spam_perturbation_frac < 1.0:
        sys.exit("--spam-perturbation-frac must be in [0, 1)")
    if args.dw_halfwidth_hz is not None:
        if args.dw_population != "literature":
            sys.exit("--dw-halfwidth-hz requires --dw-population literature")
        if not (np.isfinite(args.dw_halfwidth_hz) and args.dw_halfwidth_hz > 0.0):
            sys.exit("--dw-halfwidth-hz must be positive")
        if args.architecture == "all":
            sys.exit("--dw-halfwidth-hz is architecture-specific "
                     "(use a single --architecture)")
    cfg = {"perturbation_frac": args.perturbation_frac, "spam_scale": args.spam_scale,
           "spam_swap": args.spam_swap, "spam_spread": args.spam_spread,
           "cold_start": args.cold_start, "dw_population": args.dw_population,
           "t_measure_s": args.t_measure, "coherent_fraction": args.coherent_fraction,
           "crosstalk_scale": args.crosstalk_scale,
           "detuning_1_over_f": args.detuning_1_over_f, "alpha": args.alpha,
           "time_resolved": args.time_resolved, "t1_telegraph": args.t1_telegraph,
           "gate_detuning": args.gate_detuning, "quasi_static": args.quasi_static,
           "spam_perturbation_frac": args.spam_perturbation_frac,
           "dw_halfwidth_hz": args.dw_halfwidth_hz,
           "probe_blocks": args.probe_blocks,
           "heating_gate_error": args.heating_gate_error,
           "qiskit_experiments": args.qiskit_experiments}
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
        rows.extend(run_architecture(architecture, args.n_instances, offset, n_workers, cfg,
                                     instance_start=args.instance_start))

    problems = validate_rows(rows, args.n_instances, architectures,
                             instance_start=args.instance_start)
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