"""3_benchmark.py -- parameter-recovery accuracy on NISQ simulated hardware.

Per instance: 1c.generate_device(regime="nisq") -> hidden DeviceTruth with
drift and coherent error mechanisms; 1c.generate_calibration_prior() -> what
Canary may know; 1_inversion builds its own probe circuits from the prior;
1c.simulate_probe() executes them on Aer in a run starting at T_MEASURE_S;
1_inversion.lindblad_inversion() recovers T1, T2, delta_omega, epsilon_sx.

WHAT CANARY IS GIVEN. Only the counts, the probe metadata and a profile built
from the calibration prior: T1, T2 and epsilon priors drawn as truth x
U(1-f, 1+f) at T_CALIB_S = 0 (f = 0.25 superconducting, 0.15 trapped ion; 1c
_CALIB_PERTURBATION), while the truth keeps drifting until T_MEASURE_S
(default 3600 s, --t-measure); and the prior's readout errors, which by
default are ALSO stale, truth x U(1-f_spam, 1+f_spam) with f_spam = f (1c
calib.spam_perturbation, Class C; --spam-perturbation-frac 0 restores the
oracle SPAM of earlier versions, --spam-scale/--spam-swap add deterministic
misspecification). No truth value enters the inversion. The gate duration and
dt the engine uses equal the simulator's: those are control settings a
provider reports exactly, not calibrated noise parameters.

MODEL MISMATCH, ON BY DEFAULT (each has an ablation flag; the forward models
of 1_inversion.py represent none of them):
  --gate-detuning: the detuning keeps acting while pulses are played
      (Rz(dw*t_g/2) G Rz(dw*t_g/2)). It does not cancel in the G*G^-1 pair
      of the gate-repetition probe, so it tests that probe's claim to isolate
      incoherent error. No new parameter.
  --quasi-static (superconducting): every shot sees its own static detuning
      offset, sized per qubit from measured Ramsey/echo ratios T2R/T2E
      (Place et al., 1c sc.quasi_static.t2_ramsey_over_echo, Class B), so the
      Ramsey decay is Gaussian-exponential with T2* < T2 while the echo is
      refocused. Ignored for trapped ions (no source found).
  --dw-population literature: dw ~ U(-W, W) with W from measured frequency
      excursions (15 kHz, Dhieb et al.; 0.5 Hz, Proctor et al.; Class C
      supports; --dw-halfwidth-hz to sweep), independent of the qubit's truth
      and of the estimator's window, so it contains near-zero and aliasing
      detunings. The earlier default "estimator_window" drew dw inside the
      window the estimator resolves, conditioned on each qubit's true T2.
  --time-resolved: the probe runs circuit after circuit from T_MEASURE_S and
      each shot chunk sees the truth at its own time (details below).
  Also on (unchanged): log-T1 OU drift and low-pass detuning drift
  (superconducting), OU detuning drift (trapped ion), sign-following
  drive-amplitude error (coherent_fraction 0.10, --coherent-fraction).
OFF by default (Class C magnitudes): --t1-telegraph, --detuning-1-over-f;
motional heating stays disabled; crosstalk is inactive because the probes are
single-qubit, so --crosstalk-scale has no effect here.

TRUTH DEFINITIONS (true_value is always built from the exact parameters the
simulator used): T1 as sampled and drifted. T2 is the Markovian T2, i.e. the
Hahn-echo T2 (what IBM reports as T2), NOT the Ramsey T2*; with the
quasi-static option on, T2* = r*T2 and a Ramsey-based estimate is biased low
against this truth. epsilon is 1c.effective_gate_epsilon(): the incoherent
per-gate error of the simulated channel, including gate-time decoherence and
any amplitude-error residual surviving G*G^-1, and EXCLUDING detuning, so a
detuning-induced epsilon bias is scored as error. delta_omega is SIGNED (an
estimate with the wrong sign is an error; |dw| was compared before).

Time-resolved probe run (1c.simulate_probe(time_resolved=True)): circuits
execute one after another in probe order (circuit-major: all shots of a
circuit before the next; an assumption, a provider may interleave). One shot
lasts its delays plus its noisy native gates times the gate duration plus a
repetition overhead t_rep = 0 s. t_rep = 0 is a lower bound: IBM's Qiskit
Runtime documentation gives the default rep_delay only as backend-specific
(backend.default_rep_delay), no IonQ value was verified, and readout time is
not counted. Each circuit's shots are simulated in chunks at the truth of the
chunk mid-time; a chunk spans at most 1/10 of the shortest drift correlation
time, with at most 100 chunks per circuit (a runtime cap). What varies during
a run is what the drift processes attached by 1c vary: superconducting log-T1
(OU, 1800 s) and detuning (low-pass, correlation time 1800 s/(2*pi)), and
trapped-ion detuning (OU, 637 s, from the drift band Proctor et al. detected);
epsilon follows T1/T2 through gate-time decoherence. A superconducting run
lasts about 1 s, so its truth does not change within the run. A trapped-ion
run lasts hours (the T1 probe alone has delays of tens to hundreds of
seconds), so the detuning varies within and between Ramsey circuits. The
comparison truth per parameter is then the shot-weighted time average over
the chunks of the probe family that measures it (TRUTH_FAMILIES): T1 <- t1
circuits; T2 <- Ramsey + echo; delta_omega <- Ramsey X/Y (signed mean);
epsilon <- gate-repetition circuits (effective_gate_epsilon at the chunk
times). truth_rel_spread is (max - min)/|mean| of the parameter over those
chunks. With --no-time-resolved every circuit and shot uses the truth at
T_MEASURE_S and truth_rel_spread = 0. run_duration_s (the probe's wall-clock
length, a lower bound) is reported in both modes.

REPORTING. Besides the per-row errors, the CSV carries signed_error, z_score
= (canary - truth)/sigma_canary, the prior-only baseline (prior_value,
prior_relative_error) and the SPAM actually handed to the inversion. The
printed summary gives, per parameter, R2 of Canary NEXT TO R2 of the prior
alone: the truth populations are log-uniform over wide ranges, so R2 is high
even for a guess that measures nothing (roughly 0.9-0.998 for the prior
here), and it must not be read alone. The paired error against the prior,
the coverage of |z| <= 1 and <= 2 (nominal 68.3% / 95.4%) and the failure
count carry the evidence.

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
    "gate_detuning", "quasi_static", "spam_perturbation_frac", "dw_halfwidth_hz",
    "t2_ramsey_over_echo", "dw_true_rad_s",
    "spam_p0_given_1_true", "spam_p1_given_0_true",
    "spam_p0_given_1_used", "spam_p1_given_0_used",
    "signed_error", "z_score", "probe_blocks",
    "quasi_static_scale", "detuning_drift_scale", "t1_drift_scale",
]
NAN = float("nan")

# Benchmark-design configuration (CLI flags); each value is recorded
# (resolved) in the CSV. The defaults are the model-mismatched benchmark
# described in the module docstring. The earlier, model-consistent defaults
# are: dw_population="estimator_window", time_resolved=False,
# gate_detuning=False, quasi_static=False, spam_perturbation_frac=0.0.
DEFAULT_CFG: Dict[str, Any] = {
    "perturbation_frac": None,          # None -> 1c _CALIB_PERTURBATION table
    "spam_scale": 1.0,                  # multiplies the SPAM handed to the inversion
    "spam_swap": False,                 # swap p0|1 <-> p1|0 handed to the inversion
    "spam_spread": 0.0,                 # 1c generate_device(spam_spread_frac=...)
    "cold_start": False,                # arch_default profile instead of the prior
    "dw_population": "literature",      # 1c generate_device(dw_population=...)
    "t_measure_s": T_MEASURE_S,         # simulation / truth-evaluation time
    "coherent_fraction": None,          # None -> 1c registry value
    "crosstalk_scale": 1.0,             # no effect: probes are single-qubit
    "detuning_1_over_f": False,         # superconducting only
    "alpha": None,                      # 1/f alpha override; needs detuning_1_over_f
    "time_resolved": True,              # 1c simulate_probe(time_resolved=...)
    "t1_telegraph": False,              # superconducting only; Class C amplitude
    "gate_detuning": True,              # 1c generate_device(detuning_during_gates=...)
    "quasi_static": True,               # 1c generate_device(quasi_static_dephasing=...); SC only
    "spam_perturbation_frac": None,     # None -> the resolved perturbation_frac
    "dw_halfwidth_hz": None,            # None -> 1c registry; "literature" population only
    "probe_blocks": 10,                 # inv.interleave_probe_blocks(n_blocks=...); 1 = circuit-major
    "quasi_static_scale": 1.0,          # x quasi-static detuning std (SC); _apply_mechanism_scales
    "detuning_drift_scale": 1.0,        # x detuning drift stationary amplitude
    "t1_drift_scale": 1.0,              # x log-T1 drift stationary std (SC)
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


def _signed_error(value: float, truth_value: float) -> float:
    if not np.isfinite(value) or not np.isfinite(truth_value):
        return NAN
    return value - truth_value


def _z_score(value: float, truth_value: float, sigma: float) -> float:
    """(estimate - truth) / reported sigma; NaN when either is non-finite or
    sigma is not positive. Calibrated uncertainties give |z| <= 1 (<= 2) in
    ~68.3% (~95.4%) of instances."""
    if not (np.isfinite(value) and np.isfinite(truth_value)
            and np.isfinite(sigma) and sigma > 0):
        return NAN
    return (value - truth_value) / sigma


def _r2(truth: np.ndarray, estimate: np.ndarray) -> float:
    """1 - SS_res/SS_tot of estimate against truth (NaN if SS_tot = 0)."""
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
    spf = c["spam_perturbation_frac"]
    if spf is None:
        spf = pf
    w_hz = NAN
    if c["dw_population"] == "literature":
        key = "sc" if architecture == "superconducting" else "ti"
        w_hz = (c["dw_halfwidth_hz"] if c["dw_halfwidth_hz"] is not None
                else sim.MODEL_PROVENANCE[f"{key}.dw_population.literature_halfwidth_hz"]["value"])
    return {"gate_detuning": bool(c["gate_detuning"]),
            # 1c applies quasi-static dephasing to superconducting qubits only.
            "quasi_static": bool(c["quasi_static"]) and architecture == "superconducting",
            "spam_perturbation_frac": float(spf), "dw_halfwidth_hz": float(w_hz),
            "perturbation_frac": float(pf), "spam_scale": float(c["spam_scale"]),
            "spam_swap": bool(c["spam_swap"]), "spam_spread": float(c["spam_spread"]),
            "cold_start": bool(c["cold_start"]), "dw_population": str(c["dw_population"]),
            "t_measure_s": float(c["t_measure_s"]), "coherent_fraction": float(cf),
            "crosstalk_scale": float(c["crosstalk_scale"]),
            "detuning_1_over_f": bool(c["detuning_1_over_f"]), "alpha": float(alpha),
            "time_resolved": bool(c["time_resolved"]),
            "t1_telegraph": bool(c["t1_telegraph"]),
            "probe_blocks": int(c["probe_blocks"]),
            "quasi_static_scale": float(c["quasi_static_scale"]),
            "detuning_drift_scale": float(c["detuning_drift_scale"]),
            "t1_drift_scale": float(c["t1_drift_scale"])}


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


def _apply_mechanism_scales(truth: Any, architecture: str, c: Dict[str, Any]) -> None:
    """Scale the strength of three mismatch mechanisms of a generated truth
    (all 1.0 by default, which leaves the literature values untouched; used by
    3b_robustness_envelope.py): quasi_static_scale multiplies each qubit's
    quasi-static detuning std (superconducting only), detuning_drift_scale
    the stationary amplitude of the detuning drift process, t1_drift_scale
    the stationary std of the log-T1 OU drift (superconducting only)."""
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
    """Comparison truth of a time-resolved run: per parameter, the
    shot-weighted time average over the chunks of the probe family that
    measures it (TRUTH_FAMILIES), using the exact per-chunk truth the
    simulator used; epsilon is 1c.effective_gate_epsilon() at the chunk
    mid-times; delta_omega is the shot-weighted mean of the signed detuning.
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
        true_values[param] = mean  # delta_omega stays signed
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
        if c["dw_halfwidth_hz"] is not None and c["dw_population"] != "literature":
            raise ValueError("dw_halfwidth_hz requires dw_population='literature'")
        overrides = ({"dw_halfwidth_hz": float(c["dw_halfwidth_hz"])}
                     if c["dw_halfwidth_hz"] is not None else None)
        truth = sim.generate_device(architecture, n_qubits=1, seed=seed, regime=REGIME,
                                    overrides=overrides,
                                    crosstalk_scale=c["crosstalk_scale"],
                                    detuning_1_over_f=c["detuning_1_over_f"],
                                    spam_spread_frac=c["spam_spread"],
                                    perturbation_frac=c["perturbation_frac"],
                                    dw_population=c["dw_population"],
                                    t1_telegraph=c["t1_telegraph"],
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

        # Block-interleaved execution (inv.interleave_probe_blocks): the probe
        # list runs probe_blocks times, each with its share of every circuit's
        # shots, and the block counts are merged per circuit before inversion.
        block_circuits, block_shots, block_index = inv.interleave_probe_blocks(
            circuits, shots_list, int(c["probe_blocks"]))
        # Independent shot noise per block entry: entry i uses Aer seed seed*1000 + i
        # (at most 1000 entries, so seeds never collide across instances).
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
                           "epsilon_sx": sim.effective_gate_epsilon(truth, 0, t_measure_s)}
            # Frozen run: no within-run variation; the wall-clock length the
            # probe would take is still reported (same lower-bound accounting).
            truth_spread = dict.fromkeys(PARAMETERS, 0.0)
            run_duration_s = sim.probe_run_duration_s(block_circuits, truth, block_shots,
                                                      t_s=t_measure_s, dt_ns=profile.dt_ns)
        # delta_omega is compared SIGNED: a wrong-sign estimate is an error.
        canary = {"T1_s": float(result.T1_s),
                  "T2_s": float(result.T2_s),
                  "delta_omega": float(result.delta_omega),
                  "epsilon_sx": float(result.epsilon_sx)}
        sigmas = {"T1_s": float(result.T1_sigma_s),
                  "T2_s": float(result.T2_sigma_s),
                  "delta_omega": float(result.delta_omega_sigma),
                  "epsilon_sx": float(result.epsilon_sx_sigma)}
        chi2 = {"T1_s": float(result.t1_chi2_dof),
                "T2_s": float(result.echo_chi2_dof),  # T2_s is the echo T2
                "delta_omega": float(result.ramsey_chi2_dof),
                "epsilon_sx": float(result.gate_chi2_dof)}
        chi2_secondary = {"T2_s": float(result.ramsey_chi2_dof)}
        priors = {"T1_s": float(prior.T1_prior_s),
                  "T2_s": float(prior.T2_prior_s),
                  "delta_omega": NAN,
                  "epsilon_sx": float(prior.epsilon_prior)}
        # Post-inversion comparison only: would the true |delta_omega| alias
        # (accumulate > pi) at the shortest Ramsey delay actually probed?
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
                signed_error=_signed_error(cv, tv), z_score=_z_score(cv, tv, sg)))
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
        for flag in ("time_resolved", "t1_telegraph", "gate_detuning", "quasi_static",
                     "dw_population", "probe_blocks"):
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
    """Per architecture and parameter: failures (never dropped silently),
    median relative error of Canary and of the calibration prior alone, the
    fraction of instances where Canary beats the prior, R2 of each (the
    prior's R2 shows how much of R2 needs no measurement at all), and the
    coverage of |z| <= 1 / <= 2 (nominal 68.3% / 95.4%)."""
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
                        "signed_error", "z_score"):
                row[key] = float(row[key]) if row[key] not in ("", None) else NAN
            for key in ("spam_swap", "cold_start", "detuning_1_over_f",
                        "time_resolved", "t1_telegraph", "gate_detuning", "quasi_static"):
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
    ap.add_argument("--spam-spread", type=float, default=0.0,
                    help="generate_device(spam_spread_frac=...)")
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
    ap.add_argument("--detuning-1-over-f", action="store_true",
                    help="superconducting only: 1/f^alpha detuning drift")
    ap.add_argument("--alpha", type=float, default=None,
                    help="1/f^alpha exponent override; requires --detuning-1-over-f")
    ap.add_argument("--time-resolved", action=argparse.BooleanOptionalAction, default=True,
                    help="time-resolved probe run starting at --t-measure (default on; "
                         "--no-time-resolved freezes the truth at --t-measure for every "
                         "circuit and shot)")
    ap.add_argument("--t1-telegraph", action="store_true",
                    help="superconducting only: opt-in two-state T1 switching "
                         "(Class C switching amplitude; off by default)")
    ap.add_argument("--probe-blocks", type=int, default=DEFAULT_CFG["probe_blocks"],
                    help="block-interleaved execution: run the probe list this many "
                         "times with a share of each circuit's shots "
                         "(1_inversion.interleave_probe_blocks; 1 = circuit-major)")
    args = ap.parse_args()

    if args.n_instances <= 0:
        sys.exit("--n-instances must be positive")
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
    if args.alpha is not None and not args.detuning_1_over_f:
        sys.exit("--alpha is only valid with --detuning-1-over-f")
    if args.detuning_1_over_f and args.architecture in ("trapped_ion", "all"):
        sys.exit("--detuning-1-over-f is superconducting only "
                 "(use --architecture superconducting)")
    if args.t1_telegraph and args.architecture in ("trapped_ion", "all"):
        sys.exit("--t1-telegraph is superconducting only "
                 "(use --architecture superconducting)")
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
           "probe_blocks": args.probe_blocks}
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