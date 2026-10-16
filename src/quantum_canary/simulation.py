"""
1c_simulation_models.py -- literature-grounded physics simulation layer for
Quantum Canary's research pipeline.

Models quantum hardware in-silico for superconducting and trapped-ion qubits
under two regimes:

    ideal / model_consistent  -- correctly specified, stationary Markovian
                                 noise (the null baseline for the T1, gate
                                 and echo chi2/dof, which follow chi2(2)).
                                 With the earlier single-delay arctan2
                                 Ramsey estimator its chi2/dof was inflated
                                 (measured median 1.21 SC / 1.09 TI vs 0.84
                                 for chi2(4)/4; KS p ~ 1e-9); not yet
                                 re-measured for the unwrapped multi-delay
                                 estimator (8 points, dof 6).
    nisq  / model_mismatched  -- adds the mechanisms listed below.

NISQ MECHANISMS. [ON] = active in the single-qubit 3_benchmark.py by
default; [OFF] = exists but inactive there. Evidence classes are those of
the MODEL_PROVENANCE keys named (see EVIDENCE CLASSES below).
  Superconducting:
    [ON]  log-T1 drift, OU: stationary std sc.T1_drift.amplitude_frac (B),
          mean-reversion time sc.T1_drift.timescale_s (C).
    [ON]  detuning drift, single-pole low-frequency noise: std
          sc.detuning_drift.amplitude_rad_s (C); cutoff 1/sc.T1_drift.
          timescale_s (C).
    [OFF] 1/f^alpha detuning drift (detuning_1_over_f=True): alpha
          sc.detuning_1_over_f.alpha (A, device-specific); band low edge
          sc.detuning_1_over_f.f_min_hz (C).
    [ON]  coherent drive-amplitude error: sc.gate.coherent_fraction (C).
    [OFF] coherent ZZ crosstalk kicks (needs >= 2 qubits):
          sc.crosstalk.zz_shift_khz_min/_max (A) times sc.gate.duration_ns (C).
    [OFF] two-state T1 switching (t1_telegraph=True): rate range
          sc.T1_telegraph.switching_rate_hz (A); amplitude
          sc.T1_telegraph.amplitude_frac (C), hence opt-in.
    [ON]  quasi-static (shot-to-shot) low-frequency dephasing
          (generate_device(quasi_static_dephasing=True); library default
          False, ON in 3_benchmark.py): each shot sees its own static
          detuning offset delta ~ N(0, sigma_q), which gives Ramsey a
          Gaussian envelope (T2* < T2) while the Hahn echo refocuses it.
          sigma_q is set per qubit from a measured T2R/T2E ratio,
          sc.quasi_static.t2_ramsey_over_echo (B, Place et al.).
  Trapped ion:
    [ON]  detuning drift, additive OU: std ti.detuning_drift.amplitude_rad_s
          (C); mean-reversion time ti.detuning_drift.correlation_time_s
          (C, from the drift band Proctor et al. detected on a 171Yb+ qubit).
    [ON]  coherent drive-amplitude error: ti.gate.coherent_fraction (C).
    [OFF] coherent crosstalk kicks (needs >= 2 qubits):
          ti.crosstalk.coherent_angle_rad_high (A).
    [OFF] motional-heating gate error (enable_heating_gate_error=True):
          ti.heating.baseline_axial_quanta_per_s (A) times
          ti.heating_to_gate_error.mapping_coefficient (C).
  Both architectures:
    [ON]  stale calibration prior: calib.perturbation.* (C), plus the drift
          accumulated up to the benchmark's measurement time.
    [ON]  stale SPAM in the calibration prior (generate_calibration_prior(
          spam_perturbation_frac=f); library default 0.0 = oracle SPAM, ON in
          3_benchmark.py with f equal to the prior's own envelope,
          calib.spam_perturbation (C)).
    [ON]  detuning during gate pulses (generate_device(detuning_during_gates=
          True); library default False, ON in 3_benchmark.py): the qubit's
          detuning keeps acting while a pulse is played, implemented as
          Rz(dw*t_g/2) G Rz(dw*t_g/2). No new parameter (dw and the gate
          durations above). It does not cancel in the G*G^-1 pair of the
          gate-repetition probe, so it tests that probe's claim to isolate
          incoherent error.
    [ON]  literature detuning population (generate_device(dw_population=
          "literature"); library default "estimator_window", ON in
          3_benchmark.py): dw ~ U(-W, W) with W from measured frequency
          excursions (sc/ti.dw_population.literature_halfwidth_hz, C),
          independent of the estimator's window and of each qubit's truth.
    [ON in 3_benchmark.py] time-resolved probe execution (simulate_probe(
          time_resolved=True); library default False): circuits run
          sequentially and each shot chunk sees the truth at its own time,
          so the drift processes above act within a run and shot to shot.
          No new physical parameter: shot durations use the circuits'
          delays and the gate durations sc.gate.duration_ns (C) /
          ti.gate.duration_ns (A); the repetition overhead is 0 s (a lower
          bound, see _DEFAULT_REP_DELAY_S). A superconducting run (~1 s) is
          far shorter than its drift times, so the mode's effect there is
          negligible; a trapped-ion run lasts hours, and there it realises
          the Proctor-anchored detuning drift inside the run.
  Gate-time decoherence, readout error and detuning phase evolution during
  free evolution act in both regimes. The ideal regime is unaffected by
  every NISQ option.


SCOPE (see spec Section 17 for the full list): this module is the physics/
data-generation layer only. It does not compute R^2, chi2/dof, shot-budget
benchmarks, or run VQE/QAOA/QEC/plotting -- those live in 2_*, 3_*, 4_*.

-----------------------------------------------------------------------------
SENSITIVITY SWEEP PARAMETERS (must be swept and reported, not fixed)
-----------------------------------------------------------------------------
The following generate_device() knobs have no single defensible literature
value. Each is either explicitly phenomenological (Class C) or spans a wide,
device-dependent physical range with no universal "typical" figure. Any
manuscript figure built on the NISQ regime must report results across a
swept range of each and demonstrate the conclusion is insensitive to the
specific value chosen -- picking one point value and treating it as ground
truth would misrepresent a stress-test parameter as a measurement.

  perturbation_frac (generate_calibration_prior): calibration-observation
      staleness/imperfection. No literature value pins how far a live
      calibration prior should legitimately be allowed to drift from the
      device's true state at measurement time. Defaults are now resolved per
      (architecture, regime) from _CALIB_PERTURBATION (0.10/0.25 sc
      ideal/nisq, 0.05/0.15 ti ideal/nisq) rather than one flat 0.10, but all
      four entries remain Class C and must still be swept.
  crosstalk_scale (generate_device): multiplies every crosstalk kick angle.
      Default 1.0 reproduces the cited literature magnitude as-is (see
      sc.crosstalk.zz_shift_khz_min/_max, ti.crosstalk.coherent_angle_rad_high);
      sweeping toward 0 tests how much of any reported effect is
      crosstalk-driven versus intrinsic to the other mechanisms.
  coherent_fraction (sc.gate.coherent_fraction, ti.gate.coherent_fraction):
      share of each qubit's gate error realised as a sign-following drive-
      amplitude error. Class C (0.10 default; GST on a few IonQ Forte qubit
      pairs bounds it to a minority only qualitatively). Sweep 0-0.5.
  detuning drift amplitude (sc/ti.detuning_drift.amplitude_rad_s): Class C
      order-of-magnitude anchor / consistency bound, not a measured
      cloud-hardware spectrum. Sweep.
  alpha (generate_device(detuning_1_over_f=True), sc.detuning_1_over_f.alpha):
      Class A -- 0.57 +/- 0.03 is reported by Slichter et al., PRL 109,
      153601 (2012), arXiv:1206.6946, p.4 (1/f^alpha fit to 90 minutes of
      Ramsey fringes), verified against that text in the 2026-09-26
      external audit. It is device-specific: sub-0.5 Hz flux noise of a
      flux-biased laboratory transmon, and the paper itself notes other work
      finding alpha = 0.9-1.0. Still sweep it across a plausible range
      (e.g. 0.5-1.5) instead of relying on one point value.
  spam_perturbation_frac (generate_calibration_prior): staleness of the
      readout errors in the calibration record (calib.spam_perturbation,
      Class C). No measured readout-error staleness for cloud devices was
      found; 3_benchmark.py reuses the prior's own envelope. Sweep it,
      including 0 (oracle SPAM) as the ablation.
  literature detuning half-width (generate_device(dw_population=
      "literature"), sc/ti.dw_population.literature_halfwidth_hz): Class C
      supports built on measured excursion bounds (Dhieb et al., Proctor et
      al.). Sweep it (overrides={"dw_halfwidth_hz": W}); detuning also
      drives the gate-repetition bias when detuning_during_gates=True.

-----------------------------------------------------------------------------
EVIDENCE CLASSES (see MODEL_PROVENANCE)
-----------------------------------------------------------------------------
  A = directly measured/reported in the cited paper (number or range).
  B = mathematically derived from an A-class quantity via an explicit,
      documented relationship (e.g. p = 1 - exp(-rate * duration)).
  C = phenomenological/stress-test assumption. The cited paper supports the
      MECHANISM but not a specific number; the value here is a reasonable
      order-of-magnitude choice for exercising the mechanism, never to be
      read as a measured fact. Every C-class entry says why in its
      "class_c_rationale" field (enforced by validate_parameter_provenance).

The model's literature-facing numerical defaults, including the per-qubit
truth-sampling ranges (sc./ti.truth_range.*, from which _TRUTH_RANGES is
built), live in MODEL_PROVENANCE with their source, the location within
that source (or "abstract"/"figure N"/etc.), their evidence class, and how
they map into this module's code. Call validate_parameter_provenance() to check the
registry's internal consistency (it cannot verify the citations
themselves; see the next section).

-----------------------------------------------------------------------------
A NOTE ON VERIFICATION
-----------------------------------------------------------------------------
The citations in MODEL_PROVENANCE were checked against the primary texts in
an external audit (2026-09-26); each entry's notes record what was verified
and what remains an assumption or a modelling choice.
In particular: Wang et al. (2021) report a coherence time near 5500 s for an
actively-stabilized single 171Yb+ ion -- that is a best-case demonstration,
not the trapped-ion T2 this module samples by default (which instead follows
1_inversion.py's existing ARCH_DEFAULTS convention for an operationally
typical device, consistent with the manuscript's own trapped-ion defaults).
The two numbers should not be conflated, and this module never cites Wang's
record value as the source of its default T2 sampling range.
"""

from __future__ import annotations

import hashlib
import json
import uuid
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

ARCHITECTURES: Tuple[str, ...] = ("superconducting", "trapped_ion")
_REGIME_ALIASES: Dict[str, str] = {
    "ideal": "ideal", "model_consistent": "ideal",
    "nisq": "nisq", "model_mismatched": "nisq",
}


def _canonical_regime(regime: str) -> str:
    try:
        return _REGIME_ALIASES[regime]
    except KeyError:
        raise ValueError(
            f"Invalid regime {regime!r}; must be one of {sorted(_REGIME_ALIASES)}") from None


# =============================================================================
# 1. MODEL_PROVENANCE REGISTRY
# =============================================================================
#
# Each entry: value, unit, evidence_class (A/B/C), source (author/journal/
# year), location (equation/table/figure or "abstract" if that is as
# precise as could be confirmed), and notes describing exactly what was
# verified vs. assumed, and how the number maps into the code. Class C
# entries also carry class_c_rationale (why the value is not A/B).

MODEL_PROVENANCE: Dict[str, Dict[str, Any]] = {
    # ---------------- Superconducting ----------------
    "sc.T1.baseline_s": {
        "value": 200e-6, "unit": "s", "evidence_class": "B",
        "source": "Sharma & Lau, 'From Circuits to Hardware', arXiv:2607.11637 "
                 "(2026); equal to 1_inversion.py ARCH_DEFAULTS['superconducting']['T1_s']",
        "location": "Table 4: IBM Heron r1/r2 (ibm_fez, ibm_marrakesh, ibm_torino) "
                   "representative T1 100-300 us, T2 80-200 us",
        "notes": ("Architecture-typical value only (per-qubit truth is sampled "
                  "from _TRUTH_RANGES). 200 us is the centre of the reported "
                  "Heron T1 band. Previously 150 us while claiming to match "
                  "the engine's 200 us. CROSS-CHECK of the sampled population: "
                  "T1 80-400 us covers the Heron band; derived T2 ~32-159 us "
                  "overlaps Heron's 80-200 us but extends lower, i.e. it also "
                  "represents older/weaker cloud transmons."),
    },
    "sc.T2.baseline_s": {
        "value": 140e-6, "unit": "s", "evidence_class": "B",
        "source": "Sharma & Lau, arXiv:2607.11637, Table 4; equal to "
                 "1_inversion.py ARCH_DEFAULTS['superconducting']['T2_s']",
        "location": "Table 4 (T2 80-200 us)",
        "notes": "Centre of the reported Heron T2 band; see sc.T1.baseline_s.",
    },
    "sc.T1_drift.diffusivity_MHz_per_sqrt_hr": {
        "value": 2.5, "unit": "MHz/hr^0.5", "evidence_class": "A",
        "source": "Klimov et al., Phys. Rev. Lett. 121, 090502 (2018), arXiv:1809.01043",
        "location": "main text (D = 2.5 +/- 0.1 MHz/hr^0.5); not in abstract",
        "notes": ("Klimov et al. report a single diffusivity, consolidated over "
                  "the transition frequencies of 13 defects: 'sigma(t) = "
                  "2Dt^1/2, with the diffusivity D = 2.5 -/+ 0.1 MHz "
                  "(hour)^-1/2' (arXiv:1809.01043 p.3; repeated p.4 and Suppl. "
                  "S9). Previously recorded as 2.35 with an unsupported "
                  "'~2.2-2.5, abstract' location; corrected after the "
                  "2026-09-26 audit. Recorded only: not read by "
                  "generate_device(). "
                  "This is a TLS *frequency* diffusivity: individual two-level-"
                  "system defects random-walk in frequency, occasionally "
                  "sweeping through resonance with the qubit and suppressing "
                  "its T1 while resonant. It is NOT directly a T1(t) amplitude "
                  "or timescale. Used only as evidence for the CORRELATION "
                  "TIMESCALE of T1(t) fluctuations (see sc.T1_drift.timescale_s), "
                  "not as a literal formula for T1's magnitude."),
    },
    "sc.T1_drift.timescale_s": {
        "value": 1800.0, "unit": "s", "evidence_class": "C",
        "source": "Klimov et al., PRL 121, 090502 (2018); Google AI Blog, "
                 "'Understanding Performance Fluctuations in Quantum Processors' (2018)",
        "location": "abstract / blog summary",
        "notes": ("Klimov et al. and the accompanying Google Research blog post "
                  "both describe T1 hot/cold spots moving on timescales of "
                  "'minutes to hours' as TLS defects drift through resonance. "
                  "30 minutes (1800s) is chosen as a representative midpoint of "
                  "that stated range, used as the mean-reversion time of a "
                  "phenomenological Ornstein-Uhlenbeck process on log(T1) -- "
                  "see generate_device(). This is a reduced-order stand-in for "
                  "the actual TLS spectral-diffusion process, not a claim that "
                  "T1(t) literally follows an OU process; it reproduces the "
                  "qualitative timescale Klimov et al. report. Class C (was "
                  "B): a chosen point in a qualitative range is not a "
                  "derivation from a reported number."),
        "class_c_rationale": ("The sources state only a qualitative 'minutes "
                              "to hours' range; 1800 s is a chosen point in it, "
                              "used as the mean-reversion time of a "
                              "phenomenological OU process."),
    },
    "sc.T1_drift.amplitude_frac": {
        "value": 0.22, "unit": "dimensionless (stationary std of log T1)", "evidence_class": "B",
        "source": "Derived from Burnett et al., npj Quantum Information 5, 54 "
                 "(2019); cross-checked against Dhieb et al., arXiv:2512.18037 (2025)",
        "location": "Burnett Fig. 1b / Fig. 4d-f Gaussian fits (mu, sigma) quoted "
                   "in calib.perturbation.sc_nisq; Dhieb Sec. III A and Fig. 5",
        "notes": ("DERIVATION: Burnett's measured (mu, sigma) pairs give "
                  "sigma/mu = 20.2-26.3% over hours on a fixed-frequency "
                  "transmon; 0.22 is the centre of that band, used as the "
                  "stationary std of log(T1) (sigma/mu ~ std(log T1) for "
                  "small spread). Dhieb et al. independently report TLS-driven "
                  "T1 fluctuation over 95 h with sigma_T1 growing with <T1> "
                  "and occasional dropouts below 50% of baseline, which a "
                  "stationary OU does not reproduce (see calib.perturbation."
                  "sc_nisq for the telegraph/dropout caveat). Replaces an "
                  "earlier undocumented 0.30."),
    },
    # -- Opt-in two-state (telegraph) T1 switching: generate_device(
    # t1_telegraph=True), superconducting nisq only, default OFF (also OFF in
    # 3_benchmark.py) because its switching amplitude is Class C.
    "sc.T1_telegraph.switching_rate_hz": {
        "value": (71.4e-6, 1851.8e-6), "unit": "Hz (Lorentzian 1/tau_0)", "evidence_class": "A",
        "source": "Burnett, Bengtsson, Scigliuzzo, Niepce, Kudra, Delsing, "
                 "Bylander, 'Decoherence benchmarking of superconducting "
                 "qubits', npj Quantum Information 5, 54 (2019), arXiv:1901.04417",
        "location": "Sec. II D: 'Collectively, we find switching rates ranging "
                   "from 71.4 uHz to 1.9 mHz'; Table II columns 1/tau_0 (uHz), "
                   "extremes 71.4 (QA C6) and 1851.8 (QB C1)",
        "notes": ("Rates of the Lorentzians S(f) = 4*A^2*tau_0/(1 + (2*pi*f*"
                  "tau_0)^2) (Methods Eq. 4) fitted to T1-vs-time data of two "
                  "fixed-frequency transmons over several cooldowns. With "
                  "t1_telegraph=True, generate_device draws one 1/tau_0 per "
                  "device log-uniform in this range (the log-uniform shape is "
                  "a modelling choice) from its own RNG stream. DERIVATION of "
                  "the process rate: a symmetric two-state process switching "
                  "at rate r in each direction has autocorrelation "
                  "exp(-2*r*|t|), i.e. the Lorentzian above with tau_0 = "
                  "1/(2r), so _proc_telegraph is given r = (1/tau_0)/2."),
    },
    "sc.T1_telegraph.amplitude_frac": {
        "value": -0.12, "unit": "dimensionless (fractional T1 change on a switch)",
        "evidence_class": "C",
        "source": "Order of magnitude from Burnett et al., npj QI 5, 54 (2019), "
                 "arXiv:1901.04417, Table II (Lorentzian amplitudes A) and "
                 "Suppl. Table III (mean T1 per cooldown)",
        "location": "Table II A_Lor1/A_Lor2 (us); Table III mean T1: QA C2 57.95, "
                   "C3 49.72, C5 44.49, C6 46.68 us; QB C1 74.56, C5 81.63, "
                   "C6 71.22 us",
        "notes": ("Burnett et al. report the amplitude A (us) of each fitted "
                  "Lorentzian, not the size of a T1 jump. READING USED HERE: "
                  "for a symmetric two-state signal with levels a apart, the "
                  "one-sided PSD is a^2*tau_0/(1 + (2*pi*f*tau_0)^2), so "
                  "Eq. 4 gives a = 2A (a two-sided PSD normalisation would "
                  "give a = 2*sqrt(2)*A). 2A/<T1> over the 13 Lorentzians of "
                  "Table II spans 0.062-0.234, median 0.117, rounded to 0.12. "
                  "The sign is a choice: the switched state has the LOWER T1 "
                  "(a TLS moving into resonance suppresses T1), so starting "
                  "from the t=0 state the first switch is a dropout. "
                  "evaluate_truth_at() multiplies T1 by x(t)/x(0) of a "
                  "_proc_telegraph trajectory x with levels 1 and 1+value, on "
                  "top of the OU log-T1 drift; note that Burnett's Gaussian "
                  "T1 spread, from which sc.T1_drift.amplitude_frac is "
                  "derived, already contains these switches, so enabling "
                  "both slightly double counts T1 variance. Sweep it."),
        "class_c_rationale": ("The paper reports a Lorentzian noise amplitude, "
                              "not a fractional T1 jump; converting it assumes "
                              "a symmetric two-state fluctuator and a one-sided "
                              "PSD normalisation, and the paper itself notes "
                              "that two Lorentzians cannot distinguish two "
                              "two-state TLS from one TLS with four states. "
                              "The sign is a modelling choice."),
    },
    "sc.detuning_drift.amplitude_rad_s": {
        "value": 2 * np.pi * 2.0e3, "unit": "rad/s", "evidence_class": "C",
        "source": "Order-of-magnitude anchor: Dhieb et al., 'Stability studies on "
                 "subtractively-fabricated CMOS-compatible superconducting "
                 "transmon qubits', arXiv:2512.18037 (2025)",
        "location": "Sec. III A: qubit A.2 Ramsey-tracked frequency over 95 h, "
                   "'the observed frequency shift was small (below 15 kHz)'; "
                   "across all qubits 'remains below 100 kHz'",
        "notes": ("Used as the STATIONARY STD of the slow detuning deviation "
                  "(see _proc_lowpass_filtered, which is normalised to its "
                  "theoretical stationary std, never a sample std). A 2 kHz std "
                  "is consistent with Dhieb's <15 kHz total excursion for a "
                  "typical qubit; the <100 kHz figure includes TLS-driven "
                  "jumps this Gaussian process does not model. Lab device, "
                  "not a cloud transmon, hence Class C order-of-magnitude "
                  "only. Consistency check (validation test P3): a quasi-static "
                  "std sigma limits Ramsey T2* to sqrt(2)/sigma = 113 us, above "
                  "the median sampled superconducting T2, so the drift does not "
                  "contradict the coherence the module samples. "
                  "Process is single-pole low-frequency colored noise, NOT 1/f."),
        "class_c_rationale": ("Order-of-magnitude anchor only: Dhieb et al. "
                              "report a bound on total frequency excursion for "
                              "a laboratory device, not a stationary std for "
                              "cloud transmons; 2 kHz is chosen to be "
                              "consistent with that bound."),
    },
    "sc.gate.coherent_fraction": {
        "value": 0.10, "unit": "dimensionless", "evidence_class": "C",
        "source": "Phenomenological; supporting evidence that calibration-derived "
                 "IBM noise models carry no coherent term: Bautra et al., "
                 "arXiv:2603.14607 (2026)",
        "location": "Bautra Sec. 3.5: IBM twins map calibration data to "
                   "thermal-relaxation + depolarizing + readout channels only",
        "notes": ("Share of each qubit's sampled gate error budget eps realised "
                  "as a coherent AMPLITUDE (over-rotation) error: every native "
                  "rotation G(theta) is executed as G(theta*(1+f)), with "
                  "f = 2*sqrt(coherent_fraction*eps)/(pi/2) so that an isolated "
                  "sx carries eps_coh = delta^2/4 in 1_inversion.py's Bloch "
                  "convention. The error follows the sign of the rotation, as a "
                  "miscalibrated drive amplitude does, so an sx*sxdg pair "
                  "cancels it. Replaces sc.gate.coherent_overrotation_rad, "
                  "which applied a fixed-sign Rx(+3.5e-4) after every gate "
                  "(including inverses and virtual s/sdg) and whose value did "
                  "not match its own stated derivation (2*sqrt(3.5e-4)=0.037 "
                  "rad, not 3.5e-4). Sweep 0-0.5; no measured split for cloud "
                  "transmons was found."),
        "class_c_rationale": ("Phenomenological: no measured coherent/"
                              "stochastic split of gate error for cloud "
                              "transmons was found; Bautra et al. only show "
                              "that calibration-derived IBM noise models carry "
                              "no coherent term. Sweep 0-0.5."),
    },
    "sc.gate.stochastic_error": {
        "value": 3.5e-4, "unit": "dimensionless", "evidence_class": "C",
        "source": "Engine-compatibility default (historical 1_inversion.py eps_typical)",
        "location": "n/a (engine-compatibility default)",
        "notes": ("Architecture-typical gate error recorded on ArchitectureProfile; "
                  "per-qubit truth is sampled from _TRUTH_RANGES['eps'] instead."),
        "class_c_rationale": ("Engine-compatibility default with no "
                              "literature source; recorded on "
                              "ArchitectureProfile only and not used to sample "
                              "per-qubit truth."),
    },
    "sc.spam.p0_given_1": {
        "value": 0.030, "unit": "dimensionless", "evidence_class": "C",
        "source": "Sharma & Lau, arXiv:2607.11637 (2026), Table 4; values equal to "
                 "1_inversion.py ARCH_DEFAULTS['superconducting']",
        "location": "Table 4: IBM Heron readout error 1.25e-2 (ibm_marrakesh), "
                   "1.45e-2 (ibm_fez), 2.41e-2 (ibm_torino)",
        "notes": ("DERIVATION: (p0|1 + p1|0)/2 = (0.030 + 0.008)/2 = 1.9%, inside "
                  "the reported 1.25-2.41% Heron band (Class A); the asymmetric "
                  "split (decay during readout makes p0|1 > p1|0) is the "
                  "engine's and is not independently confirmed. REPLACES "
                  "0.0092/0.0009 (0.5% average), 2.5-5x more optimistic than "
                  "cloud Heron readout."),
        "class_c_rationale": ("The mean (p0|1 + p1|0)/2 = 1.9% sits inside "
                              "the verified Sharma & Lau Table 4 Heron band "
                              "(Class A anchor), but the p0|1/p1|0 split is "
                              "an engine assumption, not independently "
                              "confirmed."),
    },
    "sc.spam.p1_given_0": {
        "value": 0.008, "unit": "dimensionless", "evidence_class": "C",
        "source": "Sharma & Lau, arXiv:2607.11637, Table 4; see sc.spam.p0_given_1",
        "location": "Table 4 (Heron readout error)",
        "notes": "See sc.spam.p0_given_1 for the derivation.",
        "class_c_rationale": ("The mean (p0|1 + p1|0)/2 = 1.9% sits inside "
                              "the verified Sharma & Lau Table 4 Heron band "
                              "(Class A anchor), but the p0|1/p1|0 split is "
                              "an engine assumption, not independently "
                              "confirmed."),
    },
    "sc.gate.duration_ns": {
        "value": 50.0, "unit": "ns", "evidence_class": "C",
        "source": "Consistent with 1_inversion.py ARCH_DEFAULTS['superconducting']['gate_time_ns']",
        "location": "n/a (engine-compatibility default)",
        "notes": ("Order-of-magnitude single-qubit (sx) duration for fixed-frequency "
                  "transmons, not a measured cloud-device value. Sets the "
                  "gate-time decoherence that "
                  "effective_gate_epsilon() folds into the epsilon truth. "
                  "Intentionally equal to 1_inversion.py ARCH_DEFAULTS"
                  "['superconducting']['gate_time_ns'] in simulation: the "
                  "engine computes its gate-model z* from T1, T2 and "
                  "gate_time_ns; on IBM hardware it reads the live sx "
                  "gate_length instead. Live IBM snapshots show 24-32 ns sx "
                  "durations (qubit 0: ibm_fez 24 ns, ibm_torino 32 ns; "
                  "qiskit-ibm-runtime 0.50.0 fake-backend calibration "
                  "snapshots), below this 50 ns. Sensitivity: a +/-30% error "
                  "in the engine's gate time biases its eps by about +/-4% "
                  "through z* (median over 50 simulated superconducting "
                  "truths at infinite shots; worst +/-9.4%)."),
        "class_c_rationale": ("Engine-compatibility default, not a measured "
                              "cloud-device sx duration (IBM snapshots show "
                              "24-32 ns), so 50 ns is an order-of-magnitude "
                              "value only."),
    },
    "sc.crosstalk.coherent_angle_rad_uncompensated": {
        "value": 14.4e-3, "unit": "rad", "evidence_class": "A",
        "source": "Rudinger et al., PRX Quantum 2, 040338 (2021)",
        "location": "reported Q6 active-gate rotation-axis variation, uncompensated crosstalk",
        "notes": ("Simultaneous-gate-set-tomography measurement on the Advanced "
                  "Quantum Testbed (transmon) platform: driving a neighboring "
                  "qubit shifts Q6's own gate rotation axis by up to 14.4(1.0) "
                  "mrad, and its idle-gate phase by 13.1(2) mrad, with the "
                  "paper's headline finding that crosstalk on this platform is "
                  "PREDOMINANTLY COHERENT rather than stochastic. Used directly "
                  "as the rotation angle of a coherent_unitary_error kick "
                  "applied to a qubit's neighbor when both are found to occupy "
                  "the same simultaneous circuit layer -- see "
                  "_inject_crosstalk_kicks()."),
    },
    # REMOVED (deliberately, do not reintroduce):
    #   sc.crosstalk.coherent_angle_rad_compensated
    #
    # It held 0.016e-3 * 1000 = 0.016 rad = 16.0 mrad, which is 1.11x LARGER
    # than sc.crosstalk.coherent_angle_rad_uncompensated (14.4 mrad). That is
    # backwards on its face: compensation must reduce crosstalk, never
    # increase it, so the entry falsified itself.
    #
    # The arithmetic error: Rudinger et al. report 0.016(8)% residual
    # context-to-context error after their compensation. 0.016% = 1.6e-4, but
    # the expression above reads the 0.016 as already-in-percent AND then
    # multiplies by 1000, landing 100x too high.
    #
    # It is deleted rather than corrected to 1.6e-4 because a PERCENTAGE ERROR
    # cannot be converted into a ROTATION ANGLE without an extra assumption
    # about the error's geometry, and this module does not have grounds to
    # pick one. Reading 1.6e-4 as a small-angle radian proxy would be an
    # invention dressed as a citation. Nothing referenced this entry: it was
    # never used by generate_device(), get_architecture_profile(), or any
    # experiment script, so removing it changes no result. If a compensated
    # device ever needs modelling, derive the angle explicitly and document
    # the assumption rather than restoring this value.
    # -- Always-on ZZ shift (CHANGE 1) ------------------------------------
    # The quantity that produces a COHERENT conditional phase on an idling
    # neighbor is the LONGITUDINAL always-on ZZ shift (zeta / beta), not the
    # TRANSVERSE exchange coupling J. A previous version of this module
    # sampled J ~ 0.4-1.1 MHz (Alghadeer et al.) and fed it straight into
    # angle = 2*pi*J*t_gate. That is a category error: J is the bare
    # qubit-qubit exchange rate, and the residual static ZZ it induces in the
    # dispersive regime is 1-2 orders of magnitude smaller. The result was a
    # ~174 mrad mean kick, an order of magnitude above the largest directly
    # measured coherent crosstalk angle in the literature (Rudinger et al.,
    # 14.4 mrad). The entries below replace it with a measured always-on
    # interaction strength in kHz.
    "sc.crosstalk.zz_shift_khz_min": {
        "value": 15.0, "unit": "kHz", "evidence_class": "A",
        "source": "Mueller (Muller) et al., 'Towards a Digital Twin of Noisy "
                 "Quantum Computers: Calibration-Driven Emulation of Transmon "
                 "Qubits', arXiv:2504.08313",
        "location": "Table I, row 'Always-on interaction-strength [kHz]'. "
                   "Verified by direct text extraction from the arXiv PDF: the "
                   "row reads '15.0  22.2  19.2  65.7' for qubit pairs "
                   "0-2, 1-2, 3-2, 4-2 respectively. 15.0 kHz is the smallest.",
        "notes": ("Smallest of the four fitted always-on interaction strengths "
                  "for the tested QPU. Used as the LOWER bound of a uniform "
                  "per-pair sampling distribution in generate_device(); the "
                  "angle formula angle = 2*pi*zeta*t_gate is unchanged, only "
                  "the magnitude fed into it. At the module's 50 ns gate "
                  "duration this is 2*pi*15.0e3*50e-9 = 4.7 mrad. "
                  "THREE CAVEATS, stated so a reviewer does not have to find "
                  "them: "
                  "(1) INTERPRETATION. Table I's caption says 'the Always-on "
                  "interaction strength corresponds to J in Eq. (10)', where "
                  "Eq. (10) is H_ZZ = beta*Z_u*Z_v with "
                  "beta = J^2*(1/(Delta-alpha_u) - 1/(Delta-alpha_v)). Taking "
                  "the tabulated kHz values as beta directly (as done here) is "
                  "an INTERPRETATION, not a verbatim claim of the paper. It is "
                  "adopted because Eq. (10) as printed is self-inconsistent: "
                  "it yields beta = 0 identically whenever alpha_u == alpha_v, "
                  "and Sec. IV of the very same paper states that this "
                  "device's 'qubit anharmonicities are approximately 200 MHz' "
                  "for all qubits -- i.e. the printed formula predicts zero "
                  "crosstalk on the device it was fitted to. It is also "
                  "dimensionally implausible at these magnitudes: with J = 15 "
                  "kHz and Delta - alpha of order 1e8 Hz, beta would be ~1 Hz, "
                  "far too small to matter in the exp(-i*beta*d) crosstalk "
                  "operator of their Eq. (11) over a 45 ns layer. The "
                  "tabulated kHz values are therefore treated as the always-on "
                  "ZZ shift itself. EQ. (10) IS NOT IMPLEMENTED. "
                  "(2) FITTED, NOT SPECTROSCOPIC. These are free parameters "
                  "found by Differential Evolution against shot-distribution "
                  "TVD, and the paper says so explicitly: 'The fidelity and "
                  "crosstalk interaction strength differ from the values "
                  "expected from a purely physical analysis. This freedom was "
                  "intentionally granted to the optimizer, allowing these "
                  "parameters to account for errors that are not captured by "
                  "our model.' They are measured in the sense of being fitted "
                  "to real hardware data over 33 hours, not in the sense of "
                  "being read off a spectroscopy trace. "
                  "(3) DEVICE MISMATCH. Sec. IV: the device is a five-qubit "
                  "'Soprano-D QPU from QuantWare, featuring flux-tunable "
                  "transmon qubits' in a star topology with fixed couplers -- "
                  "NOT a fixed-frequency IBM-style cloud transmon, which is "
                  "the platform Quantum Canary otherwise targets. The paper "
                  "itself flags this contrast in its Discussion. Treat the "
                  "resulting angles as order-of-magnitude anchors."),
    },
    "sc.crosstalk.zz_shift_khz_max": {
        "value": 65.7, "unit": "kHz", "evidence_class": "A",
        "source": "Mueller (Muller) et al., arXiv:2504.08313",
        "location": "Table I, row 'Always-on interaction-strength [kHz]', "
                   "largest of the four verbatim values 15.0 / 22.2 / 19.2 / "
                   "65.7 (the 65.7 kHz entry is the Q4-Q2 pair, retained in "
                   "the model because Q4 'remains coupled to the active "
                   "qubits' despite being excluded from the experiments).",
        "notes": ("Upper bound of the uniform per-pair sampling distribution. "
                  "At 50 ns this is 2*pi*65.7e3*50e-9 = 20.6 mrad. Same three "
                  "caveats as sc.crosstalk.zz_shift_khz_min. "
                  "TRIANGULATION (the reason this replacement is trusted): the "
                  "sampled range [15.0, 65.7] kHz maps to kick angles of "
                  "4.7-20.6 mrad at 50 ns, which BRACKETS the 14.4 mrad "
                  "coherent rotation-axis error that Rudinger et al. measured "
                  "directly by simultaneous gate set tomography on an entirely "
                  "different transmon platform "
                  "(sc.crosstalk.coherent_angle_rad_uncompensated). Two "
                  "independent devices, two independent measurement "
                  "techniques, one order of magnitude. The superseded J-based "
                  "model produced ~174 mrad, which agrees with neither."),
    },
    "sc.crosstalk.zz_shift_khz_typical": {
        "value": 30.5, "unit": "kHz", "evidence_class": "B",
        "source": "Derived from Mueller (Muller) et al., arXiv:2504.08313, Table I",
        "location": "Arithmetic mean of the four tabulated always-on "
                   "interaction strengths.",
        "notes": ("DERIVATION, stated in full: "
                  "(15.0 + 22.2 + 19.2 + 65.7) / 4 = 122.1 / 4 = 30.525 kHz, "
                  "rounded to 30.5 kHz. The paper does NOT report this mean; "
                  "it is computed here, hence Class B rather than Class A. "
                  "Reported for interpretability only (it corresponds to a "
                  "9.6 mrad kick at 50 ns, close to Rudinger's 14.4 mrad). "
                  "generate_device() does NOT use it: per-pair values are "
                  "drawn UNIFORMLY on [min, max], because with only four "
                  "measured pairs -- one of which (65.7) is a clear outlier "
                  "driven by a single strongly coupled spectator qubit -- "
                  "there is no basis for asserting a Gaussian shape, and a "
                  "uniform draw over the measured support makes no "
                  "distributional claim the data cannot carry."),
    },
    # -- Calibration-prior staleness envelope (CHANGE 3) -------------------
    # These four numbers set the DEFAULT half-width of the bounded
    # multiplicative envelope in generate_calibration_prior() -- see
    # _CALIB_PERTURBATION. They are ALL CLASS C. No published dataset measures
    # calibration-log staleness for cloud-deployed hardware, because doing so
    # requires exactly the independent ground truth a cloud user does not
    # have. The cited papers bound the SCALE of the underlying parameter
    # wander; converting that into a staleness envelope is this module's
    # inference, not theirs.
    "calib.perturbation.sc_nisq": {
        "value": 0.25, "unit": "dimensionless (envelope half-width)", "evidence_class": "C",
        "source": "Scale bounded by Burnett et al., npj Quantum Information 5, "
                 "54 (2019), 'Decoherence benchmarking of superconducting qubits'",
        "location": "Fig. 1b and Fig. 4d-f, verified by direct text extraction "
                   "from the Chalmers open-access PDF of the published paper: "
                   "Gaussian fits to T1 histograms give Qubit A mu = 46.18 us, "
                   "sigma = 10.24 us (Fig. 1b, cooldown 6); Qubit B "
                   "mu = 70.72 us, sigma = 14.31 us (Fig. 1b); and across "
                   "separate cooldowns of qubit A, mu/sigma = 57.95/12.26 "
                   "(C2), 44.49/11.7 (C5), 46.68/10.24 (C6).",
        "notes": ("DERIVATION, stated in full so it can be disagreed with: the "
                  "sigma/mu ratios implied by the measured pairs above are "
                  "22.2%, 20.2%, 21.2%, 26.3% and 21.9% -- i.e. T1 on a real "
                  "fixed-frequency transmon wanders by roughly 20-26% (1 "
                  "sigma) about its own mean over hours, reproducibly, across "
                  "thermal cycles. Those PERCENTAGES ARE COMPUTED HERE; the "
                  "paper reports mu and sigma, not their ratio. 0.25 is chosen "
                  "as a uniform-envelope half-width at the top of that "
                  "measured band, on the reasoning that a stale calibration "
                  "record can be anywhere in the wander, not near its centre. "
                  "A uniform half-width of 0.25 has standard deviation "
                  "0.25/sqrt(3) = 14.4%, so this is if anything CONSERVATIVE "
                  "relative to a 20-26% Gaussian sigma.\n"
                  "Burnett et al. also report structure a symmetric envelope "
                  "cannot represent and which this module does NOT claim to "
                  "capture: T1 switches telegraphically ('approximately 3% of "
                  "the iterations' show a jump mid-trace), roughly 5% of decay "
                  "profiles show TLS revivals, and the switching is Lorentzian "
                  "with rates 'ranging from 71.4 uHz to 1.9 mHz' -- i.e. dwell "
                  "times of hours. That non-Gaussian, non-stationary structure "
                  "is the reason this entry is Class C: the envelope is a "
                  "stress-test envelope calibrated to the right order of "
                  "magnitude, not "
                  "a fit."),
        "class_c_rationale": ("No published dataset measures calibration-log "
                              "staleness for cloud hardware; Burnett et al. "
                              "bound only the scale of T1 wander, and their "
                              "telegraphic, non-Gaussian structure is not "
                              "represented by a symmetric envelope."),
    },
    "calib.perturbation.sc_ideal": {
        "value": 0.10, "unit": "dimensionless (envelope half-width)", "evidence_class": "C",
        "source": "No direct source. Set by construction as the "
                 "recently-recalibrated counterpart of calib.perturbation.sc_nisq.",
        "location": "n/a -- phenomenological",
        "notes": ("DERIVATION: the ideal regime is defined as a device whose "
                  "calibration record is fresh, so the prior-truth gap should "
                  "be a fraction of the full nisq wander. 0.10 is ~40% of the "
                  "0.25 nisq envelope and preserves the historical default of "
                  "this function, so existing sweeps remain comparable.\n"
                  "TWO CANDIDATE ANCHORS WERE CONSIDERED AND WITHDRAWN. Both "
                  "withdrawals are recorded because a reviewer might otherwise "
                  "expect to see them cited:\n"
                  "  (a) arXiv:2402.19293 -- does not itself measure "
                  "calibration staleness; it defers to its own Ref. [36] for "
                  "the figure. Citing it would be citing a citation.\n"
                  "  (b) arXiv:2505.22276 (Alghadeer et al., 'Low Crosstalk in "
                  "a Scalable Superconducting Quantum Lattice') -- a "
                  "purpose-built laboratory 4x4 lattice characterised by its "
                  "own authors under controlled conditions, NOT a "
                  "cloud-deployed device running a public queue between "
                  "calibration cycles. Its coherence figures (T1/T2 of order "
                  "71+/-5, 51+/-4 and 78+/-5 us) describe how good a lab "
                  "lattice can be, which is the opposite of the quantity "
                  "wanted here. It must not be used as a cloud "
                  "calibration-drift anchor.\n"
                  "Supporting (not anchoring) evidence that extreme staleness "
                  "is not the common case: Mueller et al., arXiv:2504.08313, "
                  "Fig. 7 tracks TVD between emulator and QPU 'over time since "
                  "the last calibration' across 33 hours of data and reports a "
                  "'mean TVD value over all data of the full model' of 0.15, "
                  "with no dramatic monotonic degradation across that window. "
                  "That argues against modelling day-scale staleness as "
                  "catastrophic, but it measures whole-model agreement on shot "
                  "distributions, not per-parameter drift, so it cannot set "
                  "this number."),
        "class_c_rationale": ("No direct source; set by construction as a "
                              "fraction of calib.perturbation.sc_nisq for a "
                              "freshly calibrated device."),
    },
    "calib.perturbation.ti_nisq": {
        "value": 0.15, "unit": "dimensionless (envelope half-width)", "evidence_class": "C",
        "source": "No direct source. Inferred by contrast with "
                 "calib.perturbation.sc_nisq.",
        "location": "n/a -- phenomenological",
        "notes": ("DERIVATION: trapped-ion qubit frequencies are set by atomic "
                  "structure and magnetic field rather than by a fabricated "
                  "Josephson junction, so the TLS bath that drives the "
                  "hours-scale T1 telegraph noise Burnett et al. measured in "
                  "transmons (see calib.perturbation.sc_nisq) has no "
                  "counterpart. What does drift on an ion trap is the gate "
                  "CALIBRATION -- laser power and beam pointing, motional-mode "
                  "frequencies, magnetic-field-induced qubit frequency -- "
                  "which is why ion systems recalibrate gates periodically "
                  "rather than continuously re-measuring coherence. 0.15 is "
                  "set to 60% of the superconducting nisq envelope to encode "
                  "'meaningfully more stable, but not stable enough to "
                  "ignore'. THE RATIO IS A JUDGEMENT CALL, NOT A MEASUREMENT. "
                  "No paper was found that reports the distribution of "
                  "trapped-ion calibration-log error for a queued, "
                  "cloud-accessed device; if one exists, this entry should be "
                  "replaced by it."),
        "class_c_rationale": ("No direct source; the 60% ratio to the "
                              "superconducting nisq envelope is a judgement "
                              "call, not a measurement."),
    },
    "calib.perturbation.ti_ideal": {
        "value": 0.05, "unit": "dimensionless (envelope half-width)", "evidence_class": "C",
        "source": "No direct source. Set by construction, same reasoning as "
                 "calib.perturbation.sc_ideal applied to "
                 "calib.perturbation.ti_nisq.",
        "location": "n/a -- phenomenological",
        "notes": ("DERIVATION: one third of the trapped-ion nisq envelope, "
                  "mirroring the sc_ideal/sc_nisq contraction, giving the "
                  "tightest prior in the table. This is the correct place for "
                  "the tightest value: a freshly calibrated ion trap is the "
                  "most predictable of the four cases modelled here. It is "
                  "still Class C."),
        "class_c_rationale": ("No direct source; set by construction as one "
                              "third of calib.perturbation.ti_nisq."),
    },
    "spam.per_qubit_spread_frac": {
        "value": 0.0, "unit": "dimensionless (fractional half-width)", "evidence_class": "C",
        "source": "No single literature value. Readout-error spread across a "
                 "register is device-, layout- and calibration-specific: it "
                 "is reported per-qubit in provider calibration data rather "
                 "than as a population statistic anyone has published for "
                 "cloud hardware generally.",
        "location": "n/a -- phenomenological stress-test knob",
        "notes": ("Default 0.0, which makes SPAM homogeneous across the "
                  "register and reproduces every result generated before this "
                  "knob existed BIT-FOR-BIT (at 0.0 the sampler consumes no "
                  "random numbers, so no downstream draw shifts). Set "
                  "generate_device(spam_spread_frac=f) with f > 0 to draw each "
                  "qubit's p0|1 and p1|0 independently log-uniform in "
                  "[(1-f)*base, (1+f)*base], clipped to [0,1].\n"
                  "WHY IT EXISTS: real registers are heterogeneous in readout "
                  "error -- epsilon_gate in this module already varies "
                  "per-qubit while SPAM did not, so any multi-qubit claim, and "
                  "qubit-routing arguments in particular, were being made "
                  "against an unrealistically uniform register.\n"
                  "THIS IS A SWEEP PARAMETER, NOT A MEASUREMENT. No value of "
                  "f is endorsed here. Any figure that switches it on must "
                  "report results across a swept range and show the "
                  "conclusion does not depend on the point chosen.\n"
                  "INTERACTION WITH regime='ideal': generate_calibration_prior() "
                  "passes SPAM through unperturbed, and the ideal regime is "
                  "DEFINED as a correctly-specified forward model, so ideal-"
                  "regime devices must stay homogeneous. The 0.0 default "
                  "guarantees that. An explicit f > 0 is still honoured in the "
                  "ideal regime -- callers keep control -- but doing so makes "
                  "the register heterogeneous in a parameter the inversion "
                  "treats as known, which is a deliberate departure from the "
                  "ideal regime's definition and should be labelled as such."),
        "class_c_rationale": ("Stress-test sweep knob with no published "
                              "population statistic for cloud readout-error "
                              "spread; no value of f is endorsed."),
    },
    "dw_population.independent_halfwidth_factor": {
        "value": 2.0, "unit": "dimensionless (multiple of 0.9*pi/(0.5*T2_ref))",
        "evidence_class": "C",
        "source": "No literature value. Chosen by construction for the "
                 "generate_device(dw_population='independent') stress test.",
        "location": "n/a -- phenomenological stress-test knob",
        "notes": ("K in the 'independent' detuning population: delta_omega ~ "
                  "U(-W, W), W = K * 0.9*pi / (0.5*T2_ref), T2_ref = the "
                  "architecture baseline T2 (sc.T2.baseline_s = 140 us, "
                  "ti.T2.baseline_s = 1 s). With K = 2.0: superconducting "
                  "W = 2*0.9*pi/70e-6 = 8.08e4 rad/s (W/2pi = 12.9 kHz); "
                  "trapped ion W = 2*0.9*pi/0.5 = 11.3 rad/s (W/2pi = 1.8 Hz). "
                  "The population is independent of each qubit's T2, of the "
                  "calibration prior and of 1_inversion.py's estimator window, "
                  "so it includes |dw| ~ 0 and values beyond that window "
                  "(aliasing instances). Only used when "
                  "dw_population='independent'; the default "
                  "'estimator_window' population does not read it."),
        "class_c_rationale": ("Stress-test sweep knob with no published "
                              "population statistic for cloud-device "
                              "detunings; K = 2 is chosen so the population "
                              "extends to twice the estimator window of a "
                              "qubit whose T2 prior equals T2_ref, not "
                              "derived from data."),
    },
    "sc.dw_population.literature_halfwidth_hz": {
        "value": 15.0e3, "unit": "Hz", "evidence_class": "C",
        "source": "Dhieb et al., 'Stability studies on subtractively-fabricated "
                 "CMOS-compatible superconducting transmon qubits', "
                 "arXiv:2512.18037 (2025); recalibration cadence: Dasgupta & "
                 "Humble, 'Characterizing the Stability of NISQ Devices', IEEE "
                 "QCE 2020, arXiv:2008.09612",
        "location": "Dhieb Sec. III A: Ramsey-tracked frequency of qubit A.2 over "
                   "95 h, 'the observed frequency shift was small (below 15 "
                   "kHz)'; across all qubits it 'remains below 100 kHz' (same "
                   "location as sc.detuning_drift.amplitude_rad_s). Dasgupta & "
                   "Humble Sec. III: 'IBM has been re-calibrating only once per "
                   "day' (IBM Yorktown, Mar 2019 - Mar 2020).",
        "notes": ("W/2pi in generate_device(dw_population='literature'): "
                  "dw ~ U(-W, W), W = 2*pi*15 kHz = 9.42e4 rad/s, drawn "
                  "independently of the qubit's T2, the calibration prior and "
                  "1_inversion.py's estimator window. The support is the "
                  "typical-qubit frequency excursion over 95 h WITHOUT "
                  "recalibration; a cloud qubit is recalibrated about daily, "
                  "so this is a pessimistic (wide) support for the residual "
                  "detuning at job time. It excludes the rare TLS-driven jumps "
                  "(< 100 kHz). The uniform shape is a choice (only a bound is "
                  "reported). With 1_inversion.py's window 0.9*pi/(0.5*T2_prior) "
                  "(12.9 kHz at T2_prior = 70 us), qubits with longer T2 alias "
                  "for part of this support; the benchmark flags them "
                  "(dw_aliased). Laboratory device, not an IBM Heron qubit. "
                  "Override with overrides={'dw_halfwidth_hz': W} to sweep."),
        "class_c_rationale": ("Dhieb et al. report a bound on the total "
                              "frequency excursion of a laboratory device, not "
                              "a distribution of residual detunings on cloud "
                              "hardware; using it as a uniform support is a "
                              "modelling choice. Sweep it."),
    },
    "ti.dw_population.literature_halfwidth_hz": {
        "value": 0.5, "unit": "Hz", "evidence_class": "C",
        "source": "Proctor et al., Nat. Commun. 11, 5396 (2020), arXiv:1907.13608",
        "location": "p.4 (Fig. 1E discussion): the estimated qubit detuning of a "
                   "171Yb+ ion 'varies slowly between approximately -0.5 Hz and "
                   "0.5 Hz' over ~8 h of uncompensated Ramsey data",
        "notes": ("W/2pi in generate_device(dw_population='literature') for "
                  "trapped ions: dw ~ U(-W, W), W = 2*pi*0.5 Hz = 3.14 rad/s. "
                  "The support equals the measured excursion range of an "
                  "uncompensated qubit; Proctor et al. then added feedback, so "
                  "a frequency-locked cloud device may be tighter. Laboratory "
                  "surface trap, not a cloud IonQ system. The uniform shape is "
                  "a choice. Override with overrides={'dw_halfwidth_hz': W}."),
        "class_c_rationale": ("A measured excursion range of one laboratory "
                              "ion used as a uniform support; not a "
                              "distribution of cloud-device detunings. Sweep "
                              "it."),
    },
    "calib.spam_perturbation": {
        "value": "perturbation_frac (same envelope as calib.perturbation.*)",
        "unit": "dimensionless (envelope half-width)", "evidence_class": "C",
        "source": "No quantitative source for readout-error staleness on cloud "
                 "devices. Recalibration cadence: Dasgupta & Humble, IEEE QCE "
                 "2020, arXiv:2008.09612 (Sec. III: once per day on IBM "
                 "Yorktown); the readout error there drifts over the 13-month "
                 "record (Sec. III A, Fig. 5), shown only graphically.",
        "location": "n/a -- rule, see notes",
        "notes": ("Used by generate_calibration_prior(spam_perturbation_frac=f): "
                  "the SPAM handed to the inversion becomes truth * U(1-f, 1+f), "
                  "independently for p0|1 and p1|0, from its own RNG stream (no "
                  "other prior draw moves). Library default f = 0.0 keeps the "
                  "historical oracle SPAM, so every existing caller is "
                  "unchanged; 3_benchmark.py sets f equal to the prior's own "
                  "envelope (calib.perturbation.*), on the reasoning that a "
                  "provider's readout-error record is refreshed on the same "
                  "cadence as its T1/T2/gate-error record and is as stale. "
                  "The ideal regime is defined by oracle SPAM and is not "
                  "changed. Sweep f, with 0 as the oracle ablation."),
        "class_c_rationale": ("No published number measures how far a cloud "
                              "provider's reported readout error sits from the "
                              "true value at job time; reusing the prior's "
                              "Class C envelope is a consistency choice, not a "
                              "measurement."),
    },
    "sc.quasi_static.t2_ramsey_over_echo": {
        "value": (98 / 153, 40 / 44, 47 / 76, 18 / 49, 41 / 53, 101 / 144,
                  68 / 87, 53 / 71, 58 / 78, 105 / 201),
        "unit": "dimensionless (T2 Ramsey / T2 echo, same qubit and cycle)",
        "evidence_class": "B",
        "source": "Place, Rodgers, Mundada, ..., de Leon, Houck, 'New material "
                 "platform for superconducting transmon qubits with coherence "
                 "times exceeding 0.3 milliseconds', arXiv:2003.00024 "
                 "(published in Nature Communications, 2021)",
        "location": "arXiv:2003.00024v1 Supplementary Table S1, columns "
                   "'Average T2R (us)' and 'Average T2E (us)', every 2D device "
                   "with both: 2b 98/153, 4a 40/44, 5a 47/76, 7a 18/49, 11a "
                   "41/53, 11b 101/144, 11c 68/87, 12a 53/71, 17a 58/78, 18a "
                   "105/201 (3D device 3D1 excluded)",
        "notes": ("DERIVATION: the ten ratios r = T2R/T2E span 0.367-0.909, "
                  "median 0.722. generate_device(quasi_static_dephasing=True) "
                  "draws one r per superconducting qubit from these ten values "
                  "(an empirical distribution, own RNG stream) and sets a "
                  "quasi-static detuning std sigma so that the Ramsey 1/e time "
                  "is r*T2: with the envelope exp(-t/T2 - sigma^2 t^2/2) "
                  "(a Gaussian-distributed static offset averages the fringe "
                  "to its characteristic function exp(-sigma^2 t^2/2)), the "
                  "1/e condition at t = r*T2 gives sigma = sqrt(2*(1 - r)) / "
                  "(r*T2). T2 (the sampled Markovian T2) plays the role of "
                  "T2E: a Hahn echo cancels any static offset exactly. "
                  "Place et al. fit T2R with a pure exponential and T2E with a "
                  "stretched exponential; both fitted times are 1/e times, so "
                  "the ratio is read as a ratio of 1/e times. CAVEATS: "
                  "laboratory tantalum transmons, not IBM Heron qubits; "
                  "entries 7a (both) and 17a, 18a (T2R) had three or fewer "
                  "repetitions; real low-frequency noise also degrades the "
                  "echo slightly (the quasi-static limit refocuses it "
                  "perfectly); shot-to-shot independence is assumed, so run-"
                  "to-run correlation (over-dispersion) is not modelled."),
    },
    "sc.detuning_1_over_f.psd_reference_hz": {
        "value": 1.0, "unit": "Hz", "evidence_class": "A",
        "source": "Confirmed by direct fetch of arXiv:2404.02989 (Dephasing in "
                 "Fluxonium Qubits from Coherent Quantum Phase Slips), Appendix B",
        "location": "Appendix B, Eq. B1, confirmed verbatim: "
                   "S_Phi(omega) = A_Phi^2 * (2*pi*1Hz / |omega|)",
        "notes": ("This confirms the standard 1/f power-spectral-density "
                  "FORM used by _proc_one_over_f() (the value stored here is "
                  "the reference frequency in that equation, 1 Hz, not a "
                  "measured device parameter). The paper's reported flux-"
                  "noise amplitude (A_Phi ~ 3-5 uPhi0/sqrt(Hz) across 6 "
                  "qubits, Q6 specifically at ~5.9 uPhi0/sqrt(Hz)) is for Q6, "
                  "an explicitly FLUX-TUNABLE qubit operated away from its "
                  "half-flux sweet spot to enhance flux-noise sensitivity -- "
                  "cited here ONLY for the spectral form and the sum-of-"
                  "Lorentzians generation method, never as a flux-noise claim "
                  "about fixed-frequency transmons (e.g. IBM Heron), which "
                  "are dominated by photon shot noise and TLS instead. The "
                  "amplitude actually used for fixed-frequency superconducting "
                  "detuning drift remains sc.detuning_drift.amplitude_rad_s, "
                  "unchanged by this entry."),
    },
    "sc.detuning_1_over_f.alpha": {
        "value": 0.57, "unit": "dimensionless", "evidence_class": "A",
        "source": "Slichter, Vijay, Weber, Boutin, Boissonneault, Gambetta, "
                 "Blais, Siddiqi, 'Measurement-induced qubit state mixing in "
                 "circuit QED from up-converted dephasing noise', Phys. Rev. "
                 "Lett. 109, 153601 (2012), arXiv:1206.6946",
        "location": "arXiv:1206.6946v2 p.4 and Fig. 4: Ramsey fringes measured "
                   "once per second for 90 minutes; 1/f^alpha fit gives "
                   "alpha = 0.57 +/- 0.03",
        "notes": ("Verified against the source text in the 2026-09-26 external "
                  "audit: 'First, we measured Ramsey fringes at a rate of one "
                  "per second for 90 minutes' and 'The fit coefficients give "
                  "alpha = 0.57 +/- 0.03' (p.4). An earlier note here stated "
                  "that the paper contains neither the experiment nor the "
                  "value; that statement was wrong and has been withdrawn. "
                  "DEVICE-SPECIFIC: the value describes sub-0.5 Hz flux noise "
                  "of a flux-biased laboratory transmon, not the detuning "
                  "noise of a fixed-frequency cloud transmon, and the paper "
                  "itself notes that other work found alpha = 0.9-1.0. Still "
                  "a required sensitivity-sweep parameter (see the module "
                  "docstring's Sensitivity Sweep Parameters section): the "
                  "manuscript should report insensitivity to alpha across a "
                  "swept range, not rely on this single value."),
    },
    "sc.detuning_1_over_f.f_min_hz": {
        "value": 1.0 / 86_400.0, "unit": "Hz", "evidence_class": "C",
        "source": "No source; phenomenological band edge (one-day window)",
        "location": "n/a -- phenomenological",
        "notes": ("Low edge f_min_hz of the 1/f^alpha band generated by "
                  "_proc_one_over_f() when generate_device("
                  "detuning_1_over_f=True). Fixed (rather than 1/(n*dt)) so "
                  "the trajectory never depends on how far it is queried. "
                  "Previously hard-coded in generate_device() without a "
                  "registry entry; the value is unchanged. Only used when "
                  "detuning_1_over_f=True (OFF by default)."),
        "class_c_rationale": ("No source: a one-day low-frequency band edge "
                              "chosen so the generated trajectory is "
                              "independent of the query horizon, not a "
                              "measured spectral corner."),
    },
    # ---------------- Trapped ion ----------------
    "ti.T1.baseline_s": {
        "value": 100.0, "unit": "s", "evidence_class": "C",
        "source": "Consistent with 1_inversion.py ARCH_DEFAULTS['trapped_ion']['T1_s']",
        "location": "n/a (engine-compatibility default)",
        "notes": ("This is NOT Wang et al.'s record coherence figure -- see "
                  "ti.T2.record_s below for that, kept clearly separate. "
                  "Architecture-typical value only (per-qubit truth is sampled "
                  "from _TRUTH_RANGES, 10-200 s). Previously 1000 s while "
                  "claiming to match the engine's 100 s; corrected."),
        "class_c_rationale": ("Engine-compatibility default matching "
                              "1_inversion.py ARCH_DEFAULTS; no literature "
                              "citation is attached to this entry, and "
                              "per-qubit truth is sampled from "
                              "ti.truth_range.* instead."),
    },
    "ti.T2.baseline_s": {
        "value": 1.0, "unit": "s", "evidence_class": "C",
        "source": "Consistent with 1_inversion.py ARCH_DEFAULTS['trapped_ion']['T2_s']",
        "location": "n/a (engine-compatibility default)",
        "notes": "Operationally typical value, not Wang et al.'s record. See ti.T2.record_s.",
        "class_c_rationale": ("Engine-compatibility default matching "
                              "1_inversion.py ARCH_DEFAULTS; no literature "
                              "citation is attached to this entry."),
    },
    "ti.T2.record_s": {
        "value": 5500.0, "unit": "s", "evidence_class": "A",
        "source": "Wang et al., Nature Communications 12, 233 (2021)",
        "location": "abstract (fitted exponential time constant from data up to 960s)",
        "notes": ("Best-case, actively-stabilized single-171Yb+-ion coherence "
                  "time, achieved by suppressing magnetic-field fluctuation, "
                  "microwave frequency instability, and reference-oscillator "
                  "leakage. NOT used as a default sampling value -- exposed "
                  "only as an optional 'best_case' override for a caller who "
                  "explicitly wants to stress-test against a record-quality "
                  "device, so this number is never silently substituted for "
                  "the operational default above."),
    },
    "ti.detuning_drift.amplitude_rad_s": {
        "value": float(np.sqrt(2.0)), "unit": "rad/s", "evidence_class": "C",
        "source": "Derived from the sampled trapped-ion coherence (ti.T2.baseline_s "
                 "= 1 s; _TRUTH_RANGES T_phi 0.1-3 s) via the quasi-static "
                 "dephasing relation T2* = sqrt(2)/sigma",
        "location": "n/a -- internal-consistency bound, stated in full in notes",
        "notes": ("DERIVATION: a slow Gaussian detuning deviation of std sigma, "
                  "averaged over the shots of a Ramsey measurement, decays the "
                  "fringe as exp(-sigma^2 t^2 / 2), i.e. limits T2* to "
                  "sqrt(2)/sigma. For the drift not to contradict the ~1 s "
                  "coherence this module samples for 171Yb+ clock qubits, "
                  "sigma <= sqrt(2)/(1 s) = 1.41 rad/s (0.23 Hz; below the "
                  "sqrt(2)/median-sampled-T2 bound of test P3); that bound "
                  "is used as the stationary std of an additive OU deviation "
                  "from the base detuning. It is an upper bound, not a "
                  "measurement: no cloud trapped-ion qubit-frequency drift "
                  "record was found. REPLACES 2*pi*30 Hz, which was a "
                  "motional-mode secular-frequency proxy (Sci. Rep. 2026) -- "
                  "the same category error that removed the power-line term "
                  "below -- and which, fed in as an OU diffusion coefficient "
                  "without the sqrt(2*theta) conversion, produced a ~212 Hz "
                  "stationary spread: two orders of magnitude above what a "
                  "1 s T2 allows, and it aliased 290/300 benchmark instances. "
                  "Class C (was B): the bound is derived from the module's "
                  "own Class C T2 assumption, not from a Class A quantity."),
        "class_c_rationale": ("Internal-consistency upper bound derived from "
                              "the Class C trapped-ion T2 default (1 s); no "
                              "cloud trapped-ion qubit-frequency drift record "
                              "was found. It is an upper bound, not a "
                              "measurement."),
    },
    "ti.detuning_drift.correlation_time_s": {
        "value": 1.0 / (2.0 * np.pi * 250e-6), "unit": "s", "evidence_class": "C",
        "source": "Derived from Proctor, Revelle, Nielsen, Rudinger, Lobser, "
                 "Maunz, Blume-Kohout, Young, 'Detecting and tracking drift in "
                 "quantum information processors', Nat. Commun. 11, 5396 "
                 "(2020), arXiv:1907.13608",
        "location": "p.3: Ramsey on a 171Yb+ qubit (surface trap, microwave "
                   "control), 6000 rasters over ~8 h; 'The detected frequencies "
                   "range from the lowest Fourier basis frequency for this "
                   "experiment duration -- approximately 15 uHz -- up to "
                   "approximately 250 uHz.' p.4: the detuning 'varies slowly "
                   "between approximately -0.5 Hz and 0.5 Hz', correlated with "
                   "lab temperature (Spearman |rho| = 0.92).",
        "notes": ("Mean-reversion time 1/theta of the additive OU detuning "
                  "deviation attached by generate_device() (trapped ion, "
                  "regime='nisq'); its stationary std is "
                  "ti.detuning_drift.amplitude_rad_s. DERIVATION: tau = "
                  "1/(2*pi*f_max) with f_max = 250 uHz, the highest drift "
                  "frequency Proctor et al. detected, gives 636.6 s: the "
                  "SHORTEST correlation time their data support, so it is the "
                  "most aggressive choice consistent with the measurement "
                  "(slower components were also present). Cross-check of the "
                  "amplitude: an excursion of +/-0.5 Hz has a std of ~0.35 Hz "
                  "for a sinusoid, within a factor 1.6 of the 0.225 Hz "
                  "(sqrt(2) rad/s) used by ti.detuning_drift.amplitude_rad_s. "
                  "REPLACES 100 s, which had no source. Caveat: a laboratory "
                  "surface trap, not a cloud IonQ system. A trapped-ion probe "
                  "run lasts hours, so time-resolved execution (simulate_probe("
                  "time_resolved=True), ON in 3_benchmark.py) realises this "
                  "drift inside the run. Sweep it."),
        "class_c_rationale": ("A single OU correlation time cannot reproduce "
                              "the multi-frequency drift spectrum Proctor et "
                              "al. report; 1/(2*pi*f_max) is an interpretation "
                              "of their detected band, on a laboratory trap, "
                              "not a cloud-device measurement. Sweep it."),
    },
    # ---------------------------------------------------------------------
    # REMOVED (deliberately, do not reintroduce): ti.detuning_drift.powerline_hz
    # (60.0 Hz) and ti.detuning_drift.powerline_amplitude_hz (2.0 Hz).
    #
    # Both numbers were correctly transcribed from the Scientific Reports
    # (2026) paper, but were being applied to the wrong physical quantity and
    # on the wrong timescale:
    #
    # (a) CATEGORY ERROR. The paper measures 60 Hz modulation of the
    #     MOTIONAL-MODE SECULAR FREQUENCY (radial mode), not the qubit drive
    #     detuning that delta_omega represents. This module already draws
    #     exactly this distinction for the Teller et al. heating rates
    #     ("motional heating rate, NOT computational T1"); it was missed here.
    #
    # (b) TIMESCALE ERROR. 60 Hz is a 16.7 ms period, while trapped-ion Ramsey
    #     delays span T2 ~ 0.1-3 s (see _TRUTH_RANGES) -- i.e. 6 to 180 full
    #     cycles within a single delay. Evaluating the sinusoid at one instant
    #     t_s and holding it fixed across such a delay is not physical: a fast
    #     oscillation averaged over a long delay integrates toward zero and
    #     manifests as CONTRAST DECAY, not as a static frequency offset. The
    #     previous implementation produced a static offset.
    #
    # Neither digital-twin reference modelling qubit detuning
    # (arXiv:2504.08313, arXiv:2603.14607) includes a power-line term on
    # detuning. The stochastic OU drift above is kept.
    # ---------------------------------------------------------------------
    "ti.gate.error_total": {
        "value": 1.3e-4, "unit": "dimensionless (1_inversion epsilon convention)",
        "evidence_class": "B",
        "source": "Chen et al. (IonQ Forte), arXiv:2308.05071, Fig. 3",
        "location": "single-qubit DRB error rate, median 2.0e-4",
        "notes": ("Architecture-typical value only: 2.0e-4 / 1.5 (see "
                  "ti.gate.error_range for the conversion). Previously 5e-4 "
                  "while claiming to match the engine's 3.0e-4."),
    },
    "ti.gate.coherent_fraction": {
        "value": 0.10, "unit": "dimensionless", "evidence_class": "C",
        "source": "Qualitative bound from Chen et al. (IonQ), 'Benchmarking a "
                 "trapped-ion quantum computer with 30 qubits', Quantum (2024), "
                 "arXiv:2308.05071",
        "location": "Sec. 4: 'This is supported by 1) results from gate set "
                   "tomography (GST) [31] on just a few qubit pairs that "
                   "indicate the dominant errors are stochastic in nature'; "
                   "Sec. 2: each single-qubit gate is an SK1 composite "
                   "sequence; Conclusion (non-Markovian caveat, see notes)",
        "notes": ("Chen et al. report GST on just a few qubit PAIRS (the GST "
                  "data are not published in the paper; [31] is the GST "
                  "methods paper), finding the dominant errors stochastic; "
                  "SK1 composite pulses are designed to cancel amplitude "
                  "miscalibration. Together these suggest, only indirectly "
                  "for single-qubit gates, that the coherent share is a "
                  "minority. CAVEAT: Chen et al.'s Conclusion hypothesises "
                  "that the observed noise is stochastic spin-phase noise and "
                  "states: 'Due to a non-trivial power spectral density, this "
                  "noise cannot be decomposed into Markovian gate-level noise, "
                  "which leads to context dependency' -- a limitation for any "
                  "Markovian model of Forte, including this one. "
                  "0.10 is a phenomenological point inside that qualitative "
                  "bound (no number is reported); sweep 0-0.3. Realised as a "
                  "sign-following amplitude error (see sc.gate.coherent_"
                  "fraction). REPLACES 0.5 applied as a fixed-sign Rx after "
                  "every gate, which did not cancel in a G*G^-1 pair, "
                  "accumulated as 2*N*theta, and drove the gate-repetition "
                  "chi2/dof to 100-250."),
        "class_c_rationale": ("No coherent fraction is reported; GST on a "
                              "few qubit pairs bounds it only qualitatively, so "
                              "0.10 is a phenomenological point. Sweep 0-0.3."),
    },
    "ti.heating.baseline_axial_quanta_per_s": {
        "value": 13.0, "unit": "quanta/s", "evidence_class": "A",
        "source": "M. Teller, D. A. Fioretto, P. C. Holz, P. Schindler, "
                 "V. Messerer, K. Schueppert, Y. Zou, R. Blatt, J. Chiaverini, "
                 "J. Sage, T. E. Northup, 'Heating of a Trapped Ion Induced by "
                 "Dielectric Materials', Phys. Rev. Lett. 126, 230505 (2021), "
                 "arXiv:2103.13846",
        "location": "p.4: reference axial-mode heating rate measured before the "
                   "trap was integrated with the cavity, 13(3) phonons/s at "
                   "2*pi*1.5 MHz (cited there to their unpublished Ref. [26])",
        "notes": ("Reference value for a 40Ca+ ion in a room-temperature 3D "
                  "linear Paul trap before cavity integration -- not a "
                  "171Yb+ cloud device. Motional heating rate, NOT "
                  "computational T1. Enters only the heating-to-gate-error "
                  "mapping below."),
    },
    "ti.heating.baseline_radial_quanta_per_s": {
        "value": 29.0, "unit": "quanta/s", "evidence_class": "B",
        "source": "Teller et al., PRL 126, 230505 (2021), arXiv:2103.13846; "
                 "see ti.heating.baseline_axial_quanta_per_s",
        "location": "p.4: reference radial-mode heating rates measured before "
                   "cavity integration, 26(6) and 32(8) phonons/s at "
                   "2*pi*{3.2,3.4} MHz",
        "notes": ("DERIVATION: midpoint (26 + 32)/2 = 29 of the two reported "
                  "radial rates, hence Class B (was A). Same 40Ca+ "
                  "reference-trap caveat as "
                  "ti.heating.baseline_axial_quanta_per_s."),
    },
    "ti.heating.distance_scaling_exponent": {
        "value": 4.2, "unit": "dimensionless", "evidence_class": "C",
        "source": "Teller et al., PRL 126, 230505 (2021), arXiv:2103.13846",
        "location": "p.4 and Suppl.: power-law fit S_E = A*d^-alpha to eight "
                   "SIMULATED electric-field-noise values, alpha_ax = "
                   "4.016(6), alpha_rad = 4.413(6); 4.2 used as their midpoint",
        "notes": ("The exponents are fits to finite-element SIMULATIONS of the "
                  "specific ion/dielectric-fibre geometry, not measurements. "
                  "Teller et al. note that they diverge from the 1/d^3 "
                  "scaling of an infinite dielectric plane, underscoring the "
                  "need for finite-geometry modelling. Class C (was A). Used "
                  "in the optional distance-dependent heating helper "
                  "heating_rate_at_distance()."),
        "class_c_rationale": ("The source values are power-law fits to "
                              "simulated field noise for one ion/dielectric "
                              "geometry, not measured scaling for cloud traps; "
                              "4.2 is the rounded midpoint (4.016 + 4.413)/2 = "
                              "4.2145."),
    },
    "ti.heating_to_gate_error.mapping_coefficient": {
        "value": 1e-7, "unit": "dimensionless per (quanta/s)", "evidence_class": "C",
        "source": "No universal heating-rate-to-single-qubit-gate-infidelity "
                 "mapping exists in the literature reviewed for this audit",
        "location": "n/a -- explicitly phenomenological",
        "notes": ("Motional heating primarily degrades MULTI-qubit (e.g. "
                  "Molmer-Sorensen) gates via the Debye-Waller factor, not "
                  "single-qubit rotations, which barely couple to motion. "
                  "This coefficient maps heating rate to an ADDITIONAL, small "
                  "single-qubit depolarizing contribution purely as an "
                  "illustrative stress-test channel -- it is NOT a validated "
                  "physical prediction and should not be read as one. See "
                  "spec Section 7's explicit warning that 'motional heating "
                  "is not computational T1'; this entry is the analogous "
                  "warning for gate error."),
        "class_c_rationale": ("No heating-rate-to-single-qubit-gate-"
                              "infidelity mapping was found in the literature "
                              "reviewed; illustrative stress-test channel "
                              "only, off by default."),
    },
    "ti.spam.p0_given_1": {
        "value": 0.0059, "unit": "dimensionless", "evidence_class": "B",
        "source": "Chen et al. (IonQ Forte), Quantum (2024), arXiv:2308.05071, "
                 "split by 1_inversion.py's _TI_SPAM_RATIO_P0_OVER_P1 (Mai, "
                 "Zhang & Lu, arXiv:2402.18868, Fig. 4, hyperfine detection)",
        "location": "Chen Sec. 2: 'The average state preparation and measurement "
                   "(SPAM) error is measured to be 0.5% on each qubit'; Mai "
                   "Fig. 4 caption: hyperfine detection p(1|0) = 6.4(7)e-3, "
                   "p(0|1) = 9.3(8)e-3",
        "notes": ("DERIVATION: (p0|1 + p1|0)/2 = 0.005 (Chen, Class A) with "
                  "r = p0|1/p1|0 = 0.0093/0.0064 = 1.453 gives p0|1 = "
                  "0.01*r/(1+r) = 0.0059 and p1|0 = 0.01/(1+r) = 0.0041, "
                  "identical to 1_inversion.py ARCH_DEFAULTS['trapped_ion']. "
                  "RATIO SOURCE: Forte reads out by standard hyperfine "
                  "(resonant-fluorescence) detection (Chen Sec. 2), and r is "
                  "Mai et al.'s ratio for that scheme, so the ratio is now "
                  "matched to the detection scheme. Caveats: r comes from a "
                  "single 171Yb+ ion in a laboratory trap (threshold 0), "
                  "Forte's own per-state asymmetry is not published, and "
                  "reading Chen's 'average' as the mean of p0|1 and p1|0 is an "
                  "interpretation. REPLACES 0.0022/0.0078, which used Mai's "
                  "2D3/2 shelving-detection ratio (0.0005/0.0018 = 0.278), a "
                  "scheme Forte does not use, with the opposite asymmetry."),
    },
    "ti.spam.p1_given_0": {
        "value": 0.0041, "unit": "dimensionless", "evidence_class": "B",
        "source": "Chen et al. (IonQ Forte), arXiv:2308.05071; see ti.spam.p0_given_1",
        "location": "Sec. 2 (0.5% average SPAM)",
        "notes": "See ti.spam.p0_given_1 for the split derivation.",
    },
    "ti.gate.duration_ns": {
        "value": 110_000.0, "unit": "ns", "evidence_class": "A",
        "source": "Chen et al. (IonQ Forte), Quantum (2024), arXiv:2308.05071",
        "location": "Sec. 2: 'The total duration of each single-qubit gate is "
                   "110 us, which comprises nine pi/2 pulses according to the "
                   "SK1 dynamical correction sequence'",
        "notes": ("Replaces 135 us (engine default, uncited). Sets the gate-time "
                  "decoherence folded into effective_gate_epsilon(). "
                  "Intentionally equal to 1_inversion.py ARCH_DEFAULTS"
                  "['trapped_ion']['gate_time_ns'] in simulation: the engine "
                  "computes its gate-model z* from T1, T2 and gate_time_ns "
                  "(on IBM hardware the engine reads the live sx gate_length "
                  "instead; from_ionq_backend keeps this value)."),
    },
    "ti.gate.error_range": {
        "value": (1e-4, 2e-3), "unit": "dimensionless (1_inversion epsilon convention)",
        "evidence_class": "C",
        "source": "Chen et al. (IonQ Forte), arXiv:2308.05071, Fig. 3; "
                 "Wright et al. (IonQ 11-qubit), arXiv:1903.08181, Table I",
        "location": "Forte single-qubit DRB error: median 2.0e-4, 10th/90th "
                   "percentile 1.8e-4/2.6e-4. 11-qubit system: RB fidelities "
                   "99.18-99.64% (average 99.5%).",
        "notes": ("CONVERSIONS (per-source conventions). Chen: the DRB error "
                  "rate is an entanglement infidelity, and a depolarizing "
                  "channel with Bloch shrink (1-2*eps) has entanglement "
                  "infidelity 1.5*eps, so Forte's median maps to eps ~ 1.3e-4 "
                  "(10-90%: 1.2e-4-1.7e-4). Wright: the tabulated 'Gate "
                  "Fidelity' is the RB decay base p of the fit B*p^L + 1/2 "
                  "('the base p is the gate fidelity'); with p = 1 - 2*eps, "
                  "eps = (1-p)/2 = 1.8e-3 (99.64%) to 4.1e-3 (99.18%), 2.5e-3 "
                  "at the 99.5% average. Each RB step also contains a random "
                  "pi rotation or identity, so the per-pi/2 eps is at most "
                  "(1-p)/2. The sampled log-uniform range 1e-4-2e-3 spans "
                  "Forte-class devices and overlaps only the best 2019 qubits "
                  "(1.8e-3, 1.9e-3); most of the 2019 generation is not "
                  "sampled. Class C (was B): a chosen sampling support, not a "
                  "derived number. The sampler reads the identical value from "
                  "ti.truth_range.eps (via _TRUTH_RANGES)."),
        "class_c_rationale": ("A chosen log-uniform sampling support "
                              "bracketed by, not derived from, the Chen and "
                              "Wright figures."),
    },
    "ti.crosstalk.coherent_angle_rad_low": {
        "value": 33e-3, "unit": "rad", "evidence_class": "A",
        "source": "Rudinger et al., PRX Quantum 2, 040338 (2021)",
        "location": "reported QSCOUT Q0 rotation-angle context variation, up to 33(7) mrad",
        "notes": ("Coherent crosstalk on the less-susceptible of the two "
                  "characterized QSCOUT ions. Both platforms in this paper "
                  "show predominantly coherent, not stochastic, crosstalk."),
    },
    "ti.crosstalk.coherent_angle_rad_high": {
        "value": 170e-3, "unit": "rad", "evidence_class": "A",
        "source": "Rudinger et al., PRX Quantum 2, 040338 (2021)",
        "location": "reported QSCOUT Q1 rotation-angle context variation, up to 170(40) mrad",
        "notes": ("The more crosstalk-susceptible of the two characterized "
                  "ions -- 3-5x larger than Q0. Used as the default trapped-"
                  "ion crosstalk magnitude in regime='nisq' since it is the "
                  "reported, not a hypothetical worst case."),
    },
    # ---------------- Per-qubit truth-sampling ranges ----------------
    # generate_device() draws per-qubit T1, T_phi and eps log-uniform from
    # these (lo, hi) ranges; _TRUTH_RANGES is built from these entries (see
    # the comment above _TRUTH_RANGES for the full rationale).
    "sc.truth_range.T1_s": {
        "value": (80e-6, 400e-6), "unit": "s", "evidence_class": "C",
        "source": "Chosen sampling support, compared with Sharma & Lau, "
                 "arXiv:2607.11637 (2026), Table 4",
        "location": "Table 4: representative IBM Heron T1 100-300 us",
        "notes": ("Per-qubit truth T1 is drawn log-uniform in this range. It "
                  "covers the reported Heron T1 band (100-300 us, see "
                  "sc.T1.baseline_s) and extends beyond it on both sides."),
        "class_c_rationale": ("A chosen sampling range wider than the "
                              "reported band, not derived from a reported "
                              "number."),
    },
    "sc.truth_range.T_phi_s": {
        "value": (40e-6, 200e-6), "unit": "s", "evidence_class": "C",
        "source": "Chosen sampling support, compared with Sharma & Lau, "
                 "arXiv:2607.11637 (2026), Table 4",
        "location": "Table 4: representative IBM Heron T1 100-300 us, "
                   "T2 80-200 us",
        "notes": ("PURE DEPHASING time; T2 is derived as 1/T2 = 1/(2*T1) + "
                  "1/T_phi and capped at 2*T1. Read back through that "
                  "relation, the Heron bands give T_phi ~133-300 us (100/80 "
                  "us -> 133 us; 300/200 us -> 300 us). The sampled 40-200 us "
                  "range is deliberately broader toward WEAKER coherence: it "
                  "also represents older/weaker cloud transmons. Measured with "
                  "generate_device('superconducting', 5000 qubits, "
                  "seed=12345): derived T2 32.1-158.8 us, median 69.6 us, "
                  "61.4% below 80 us (the lower edge of the Heron T2 band), "
                  "none above 160 us. The population is therefore NOT "
                  "Heron-representative."),
        "class_c_rationale": ("A chosen sampling range, not derived from a "
                              "source; it deliberately extends below the "
                              "coherence implied by Sharma & Lau Table 4."),
    },
    "sc.truth_range.eps": {
        "value": (1e-4, 2e-3), "unit": "dimensionless (1_inversion epsilon convention)",
        "evidence_class": "C",
        "source": "No literature source registered for this range",
        "location": "n/a -- chosen sampling support",
        "notes": ("Per-qubit truth gate error is drawn log-uniform in this "
                  "range. The engine default eps_typical (3.0e-4) and "
                  "sc.gate.stochastic_error (3.5e-4) lie inside it."),
        "class_c_rationale": ("No superconducting gate-error range is cited; "
                              "a chosen sampling support."),
    },
    "ti.truth_range.T1_s": {
        "value": (10.0, 200.0), "unit": "s", "evidence_class": "C",
        "source": "IonQ vendor specification (Aria and Forte product pages "
                 "list T1 & T2 as '10-100s, ~1s'; checked in the 2026-09-26 "
                 "audit), upper bound widened",
        "location": "ionq.com Aria / Forte product pages",
        "notes": ("The vendor states T1 10-100 s; the upper bound is widened "
                  "to 200 s for per-qubit spread. The previous (100, 10_000) "
                  "s range had no citation and put its upper end 100x above "
                  "the stated 100 s maximum."),
        "class_c_rationale": ("Vendor range widened by choice (200 s vs the "
                              "stated 100 s); a sampling support, not a "
                              "measured per-qubit distribution."),
    },
    "ti.truth_range.T_phi_s": {
        "value": (0.1, 3.0), "unit": "s", "evidence_class": "C",
        "source": "Chosen sampling support around the IonQ vendor T2 "
                 "(~1 s); see ti.truth_range.T1_s",
        "location": "ionq.com Aria / Forte product pages",
        "notes": ("Because T1 >> T_phi here, the 1/(2*T1) term is negligible "
                  "and T2 ~ T_phi to within ~1%. Measured with "
                  "generate_device('trapped_ion', 5000 qubits, seed=12345): "
                  "derived T2 0.100-2.96 s, median 0.53 s."),
        "class_c_rationale": ("A chosen range bracketing the vendor's ~1 s "
                              "T2, not derived from a reported distribution."),
    },
    "ti.truth_range.eps": {
        "value": (1e-4, 2e-3), "unit": "dimensionless (1_inversion epsilon convention)",
        "evidence_class": "C",
        "source": "Same range as ti.gate.error_range (Chen et al., "
                 "arXiv:2308.05071; Wright et al., arXiv:1903.08181)",
        "location": "see ti.gate.error_range",
        "notes": ("Operative sampling range for trapped-ion gate error. "
                  "ti.gate.error_range records the same value together with "
                  "the per-source conversions; the two must stay equal."),
        "class_c_rationale": ("Same chosen sampling support as "
                              "ti.gate.error_range: bracketed by, not derived "
                              "from, the Chen and Wright figures."),
    },
}


def get_parameter_provenance(key: Optional[str] = None) -> Dict[str, Any]:
    """Return the full registry, or one entry's record if key is given."""
    if key is None:
        return {k: dict(v) for k, v in MODEL_PROVENANCE.items()}
    if key not in MODEL_PROVENANCE:
        raise KeyError(f"Unknown provenance key: {key!r}")
    return dict(MODEL_PROVENANCE[key])


def validate_parameter_provenance() -> List[str]:
    """Structural validation of MODEL_PROVENANCE (field completeness and
    internal consistency). This checks the registry's own bookkeeping, not
    whether the citations are correct: the citations were checked in an
    external audit (2026-09-26), and each entry's 'notes' record what was
    verified. Class C entries must carry a non-empty 'class_c_rationale'."""
    issues: List[str] = []
    required = {"value", "unit", "evidence_class", "source", "location", "notes"}
    for key, entry in MODEL_PROVENANCE.items():
        missing = required - set(entry.keys())
        if missing:
            issues.append(f"{key}: missing fields {sorted(missing)}")
        ec = entry.get("evidence_class")
        if ec not in ("A", "B", "C"):
            issues.append(f"{key}: invalid evidence_class {ec!r}")
        if ec == "A" and not entry.get("source"):
            issues.append(f"{key}: evidence_class A requires a source")
        if ec == "C":
            rationale = entry.get("class_c_rationale")
            if not (isinstance(rationale, str) and rationale.strip()):
                issues.append(f"{key}: evidence_class C requires a non-empty 'class_c_rationale'")
    return issues


# =============================================================================
# 2. CORE DATACLASSES
# =============================================================================

@dataclass
class ArchitectureProfile:
    """Static, architecture-level typical values (not a per-instance truth)."""
    architecture: str
    T1_s: float
    T2_s: float
    gate_duration_ns: float
    spam_p0_given_1: float
    spam_p1_given_0: float
    gate_error_total: float
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"Invalid architecture: {self.architecture!r}")
        if self.T1_s <= 0:
            raise ValueError(f"T1_s must be positive, got {self.T1_s}")
        if self.T2_s <= 0:
            raise ValueError(f"T2_s must be positive, got {self.T2_s}")
        if self.T2_s > 2.0 * self.T1_s * (1 + 1e-9):
            raise ValueError(f"T2_s={self.T2_s} > 2*T1_s={2*self.T1_s}")
        if self.gate_duration_ns < 0:
            raise ValueError("gate_duration_ns must be non-negative")
        for name, val in (("spam_p0_given_1", self.spam_p0_given_1),
                          ("spam_p1_given_0", self.spam_p1_given_0)):
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0,1], got {val}")


@dataclass
class DriftProcess:
    """Specification for one temporal stochastic process (not the trajectory itself)."""
    process_type: str  # "stationary"|"ou"|"telegraph"|"random_walk"|"lowpass_filtered"|"one_over_f"
    params: Dict[str, Any] = field(default_factory=dict)
    seed_salt: str = ""

    _VALID = ("stationary", "ou", "telegraph", "random_walk", "lowpass_filtered", "one_over_f")

    def __post_init__(self) -> None:
        if self.process_type not in self._VALID:
            raise ValueError(f"Invalid process_type {self.process_type!r}; must be one of {self._VALID}")


@dataclass
class CrosstalkMechanism:
    """One coherent-crosstalk mechanism: a rotation kick applied to qubit b's
    circuit whenever qubit a (its paired neighbor) is found to have a gate in
    the same simultaneous circuit layer. Matches Rudinger et al.'s finding
    that crosstalk on both superconducting and trapped-ion platforms is
    predominantly coherent, not stochastic."""
    enabled: bool = False
    angle_rad: float = 0.0
    axis: str = "z"  # "x" | "y" | "z" -- rotation axis of the injected kick
    affected_pairs: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class DeviceTruth:
    """Hidden physical parameters for one simulated device instance. Contains
    everything needed to reproduce the simulation. NEVER passed to
    1_inversion.py -- only synthetic measurement counts derived from it are."""
    architecture: str
    n_qubits: int
    seed: int
    regime: str
    T1_s: np.ndarray
    T2_s: np.ndarray
    delta_omega_rad_s: np.ndarray
    epsilon_gate: np.ndarray
    spam_p0_given_1: np.ndarray
    spam_p1_given_0: np.ndarray
    gate_duration_ns: float
    drift_processes: Dict[str, DriftProcess] = field(default_factory=dict)
    crosstalk: List[CrosstalkMechanism] = field(default_factory=list)
    architecture_specific: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"Invalid architecture: {self.architecture!r}")
        if self.n_qubits <= 0:
            raise ValueError(f"n_qubits must be positive, got {self.n_qubits}")
        self.regime = _canonical_regime(self.regime)
        for name in ("T1_s", "T2_s", "delta_omega_rad_s", "epsilon_gate",
                     "spam_p0_given_1", "spam_p1_given_0"):
            arr = getattr(self, name)
            if arr.shape != (self.n_qubits,):
                raise ValueError(f"{name} shape {arr.shape} != ({self.n_qubits},)")
        if np.any(self.T1_s <= 0):
            raise ValueError("T1_s must be strictly positive for all qubits")
        if np.any(self.T2_s <= 0):
            raise ValueError("T2_s must be strictly positive for all qubits")
        if np.any(self.T2_s > 2.0 * self.T1_s * (1 + 1e-9)):
            raise ValueError("T2_s > 2*T1_s for some qubit; violates the Lindblad bound")
        for name in ("spam_p0_given_1", "spam_p1_given_0"):
            arr = getattr(self, name)
            if np.any(arr < 0) or np.any(arr > 1):
                raise ValueError(f"{name} must be in [0,1] for all qubits")
        if np.any(self.epsilon_gate < 0) or np.any(self.epsilon_gate > 0.5):
            raise ValueError("epsilon_gate must be in [0, 0.5] for all qubits")


@dataclass
class CalibrationPrior:
    """What Canary is allowed to know before it measures anything: the values
    a provider's calibration log would report for this qubit, plus the age of
    that report.

    There is deliberately NO delta_omega field. Drive detuning is not a
    parameter providers calibrate and publish -- it is actively nulled by
    frequency locking -- and 1_inversion.py recovers it in closed form from
    the shortest Ramsey delay, so it needs no warm-start value. Only the
    parameters that seed a fit (T1, T2, epsilon) or enter the forward model
    as known constants (SPAM) appear here."""
    T1_prior_s: float
    T2_prior_s: float
    epsilon_prior: float
    spam_p0_given_1_prior: float
    spam_p1_given_0_prior: float
    t0_s: float
    source: str = "synthetic_calibration"

    def staleness_s(self, now_s: float) -> float:
        return now_s - self.t0_s


@dataclass
class SimulationResult:
    """Output of simulate_probe(): counts_list is directly compatible with
    1_inversion.py's lindblad_inversion(counts_list, meta, profile, ...).

    The fields after truth_evaluation_time_s are filled only by
    simulate_probe(..., time_resolved=True); truth_evaluation_time_s is then
    the start time of the run. circuit_start_times_s / circuit_end_times_s
    are per-circuit wall-clock times (s), run_duration_s is their total span,
    and chunk_records[i] lists circuit i's shot chunks as dicts with keys
    t_start_s, t_end_s, t_mid_s, shots, seed and truth (evaluate_truth_at()
    at t_mid_s for every qubit: the exact parameters that chunk was
    simulated with)."""
    counts_list: List[Dict[str, int]]
    shots_list: List[int]
    circuit_names: List[str]
    manifest_id: str
    truth_evaluation_time_s: float
    time_resolved: bool = False
    circuit_start_times_s: Optional[List[float]] = None
    circuit_end_times_s: Optional[List[float]] = None
    run_duration_s: Optional[float] = None
    chunk_records: Optional[List[List[Dict[str, Any]]]] = None


@dataclass
class SimulationManifest:
    """Full reproducibility record. truth and calibration are recorded as
    separate top-level fields. The calibration T1/T2/epsilon priors are
    perturbed from truth; the calibration SPAM fields are copied verbatim
    from truth (oracle SPAM) unless generate_calibration_prior was given
    spam_perturbation_frac > 0 (see its docstring)."""
    instance_id: str
    seed: int
    architecture: str
    regime: str
    n_qubits: int
    truth: Dict[str, Any]
    calibration: Dict[str, Any]
    simulation_config: Dict[str, Any]
    drift: Dict[str, Any]
    crosstalk: List[Dict[str, Any]]
    gate_duration_ns: float
    shots: int
    provenance_keys_used: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# =============================================================================
# 3. DETERMINISTIC RNG STREAMS
# =============================================================================

def _sub_rng(master_seed: int, *salt: Any) -> np.random.Generator:
    """Derive a deterministic, independent RNG stream from a master seed plus
    an arbitrary salt tuple. No wall-clock time or process state is ever
    used -- identical (master_seed, salt) always yields identical output."""
    material = f"{master_seed}:" + ":".join(str(s) for s in salt)
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    seed_seq = np.random.SeedSequence(int.from_bytes(digest[:8], "big"))
    return np.random.default_rng(seed_seq)


# =============================================================================
# 4. STOCHASTIC PROCESS LIBRARY
# =============================================================================

def _proc_stationary(rng: np.random.Generator, n: int, mean: float, std: float) -> np.ndarray:
    return np.full(n, mean) if std <= 0 else rng.normal(mean, std, size=n)


def _proc_ou(rng: np.random.Generator, n: int, mean: float, sigma: float,
             theta: float, dt: float, x0: Optional[float] = None) -> np.ndarray:
    """Ornstein-Uhlenbeck: dx = theta*(mean-x)*dt + sigma*dW, exact discretization."""
    x = np.empty(n)
    x[0] = mean if x0 is None else x0
    if theta <= 0 or n <= 1:
        x[1:] = x[0] + (rng.normal(0, sigma * np.sqrt(dt), n - 1) if sigma > 0 else 0.0)
        return x
    decay = np.exp(-theta * dt)
    step_std = sigma * np.sqrt((1.0 - decay ** 2) / (2.0 * theta))
    noise = rng.normal(0.0, step_std, size=n - 1) if step_std > 0 else np.zeros(n - 1)
    for i in range(n - 1):
        x[i + 1] = mean + (x[i] - mean) * decay + noise[i]
    return x


def _proc_telegraph(rng: np.random.Generator, n: int, dt: float, rate: float,
                    baseline: float, amplitude_frac: float) -> np.ndarray:
    """Two-state telegraph process switching between baseline and
    baseline*(1+amplitude_frac) at the given switching rate (Hz)."""
    p_switch = 1.0 - np.exp(-rate * dt)
    switches = rng.random(n) < p_switch
    state = np.zeros(n, dtype=int)
    cur = 0
    for i in range(n):
        if switches[i]:
            cur = 1 - cur
        state[i] = cur
    return baseline * (1.0 + amplitude_frac * state)


def _proc_random_walk(rng: np.random.Generator, n: int, step_std: float, x0: float) -> np.ndarray:
    steps = rng.normal(0.0, step_std, size=n - 1) if step_std > 0 else np.zeros(n - 1)
    return x0 + np.concatenate(([0.0], np.cumsum(steps)))


def _proc_lowpass_filtered(rng: np.random.Generator, n: int, amplitude: float,
                           cutoff_hz: float, dt: float) -> np.ndarray:
    """First-order lowpass-filtered white noise -- i.e. low-frequency colored
    (single-pole/Lorentzian) noise, NOT a 1/f (pink-noise) process: its power
    spectral density is flat below cutoff_hz and rolls off as 1/f^2 above it,
    whereas true 1/f noise has a 1/f rolloff across the whole band. This
    function and every caller of it are deliberately named "low-frequency
    colored noise", never "1/f", to avoid that conflation.

    The recursion y[i] = a*y[i-1] + (1-a)*w[i] is started from its stationary
    distribution and scaled by its THEORETICAL stationary std,
    sqrt((1-a)/(1+a)), so amplitude is the process's stationary std and the
    first k samples do not depend on n. The previous version divided by the
    SAMPLE std of the returned trajectory, which (a) blew up for short
    trajectories -- two nearly equal samples have a near-zero sample std --
    producing detuning excursions up to ~180x amplitude, and (b) rescaled
    earlier points whenever the trajectory was extended, so the same instant
    had different values depending on how far ahead it was queried."""
    if n < 1:
        return np.zeros(0)
    a = np.exp(-2.0 * np.pi * cutoff_hz * dt)
    std_stat = np.sqrt((1.0 - a) / (1.0 + a))
    if std_stat <= 0.0:
        return np.zeros(n)
    white = rng.normal(0.0, 1.0, size=n)
    y = np.empty(n)
    y[0] = std_stat * white[0]
    for i in range(1, n):
        y[i] = a * y[i - 1] + (1.0 - a) * white[i]
    return y * (amplitude / std_stat)


def _proc_one_over_f(rng: np.random.Generator, n: int, amplitude: float,
                     dt: float, alpha: float = 1.0, n_lorentzians: int = 20,
                     f_min_hz: Optional[float] = None) -> np.ndarray:
    """Genuine 1/f^alpha noise via a sum of Lorentzians (distributed
    relaxators), the standard construction for approximating a 1/f^alpha
    power spectral density from a finite sum of first-order (OU/Lorentzian)
    processes -- see e.g. Kogan, "Electronic Noise and Fluctuations in
    Solids". Each component i is a _proc_lowpass_filtered() draw with its own
    corner frequency f_i, log-spaced across [f_min_hz, 1/(2*dt)]; weighting
    each component's variance by f_i^(1-alpha) generalizes the equal-variance
    1/f sum to 1/f^alpha.

    amplitude is the THEORETICAL stationary std (components have unit
    stationary variance and the weights are normalised to unit total
    variance), and each component draws from its own sub-generator, so the
    first k samples do not depend on n. f_min_hz defaults to 1/(n*dt) only for
    direct spectral checks; drift callers pass a fixed band so that extending
    a trajectory never changes its past.
    """
    if n < 1:
        return np.zeros(0)
    f_nyquist = 0.5 / dt
    f_min = (1.0 / (n * dt)) if f_min_hz is None else float(f_min_hz)
    f_min = min(f_min, f_nyquist)
    corner_freqs = np.logspace(np.log10(f_min), np.log10(f_nyquist), n_lorentzians)
    weights = corner_freqs ** ((1.0 - alpha) / 2.0)
    weights = weights / np.sqrt(np.sum(weights ** 2))
    sub_seeds = rng.integers(0, 2**63 - 1, size=n_lorentzians)
    y = np.zeros(n)
    for f_i, w_i, sd in zip(corner_freqs, weights, sub_seeds):
        y += w_i * _proc_lowpass_filtered(np.random.default_rng(int(sd)), n,
                                          amplitude=1.0, cutoff_hz=f_i, dt=dt)
    return y * amplitude


def sample_trajectory(process: DriftProcess, master_seed: int, salt: Any,
                      n_steps: int, dt_s: float, base_value: float) -> np.ndarray:
    """Materialize n_steps samples of `process`, spaced dt_s apart, centered
    on base_value, from a deterministic RNG stream keyed by (master_seed, salt)."""
    rng = _sub_rng(master_seed, salt, process.process_type, process.seed_salt)
    p = process.params
    pt = process.process_type
    if pt == "stationary":
        return _proc_stationary(rng, n_steps, base_value, p.get("std", 0.0))
    if pt == "ou":
        mean, sigma, theta = p.get("mean", base_value), p.get("sigma", 0.0), p.get("theta", 1.0)
        x0 = p.get("x0", base_value)
        if isinstance(x0, str) and x0 == "stationary":
            x0 = mean + (rng.normal(0.0, sigma / np.sqrt(2.0 * theta)) if theta > 0 else 0.0)
        return _proc_ou(rng, n_steps, mean=mean, sigma=sigma, theta=theta, dt=dt_s, x0=x0)
    if pt == "telegraph":
        return _proc_telegraph(rng, n_steps, dt=dt_s, rate=p.get("rate", 1e-3),
                               baseline=base_value, amplitude_frac=p.get("amplitude_frac", 0.0))
    if pt == "random_walk":
        return _proc_random_walk(rng, n_steps, p.get("step_std", 0.0), base_value)
    if pt == "lowpass_filtered":
        return _proc_lowpass_filtered(rng, n_steps, amplitude=p.get("amplitude", 0.0),
                                      cutoff_hz=p.get("cutoff_hz", 1e-3), dt=dt_s)
    if pt == "one_over_f":
        return _proc_one_over_f(rng, n_steps, amplitude=p.get("amplitude", 0.0),
                                dt=dt_s, alpha=p.get("alpha", 1.0),
                                n_lorentzians=p.get("n_lorentzians", 20),
                                f_min_hz=p.get("f_min_hz"))
    raise ValueError(f"Unknown process_type {pt!r}")


def evaluate_at(process: DriftProcess, master_seed: int, salt: Any,
                t_s: float, base_value: float, dt_s: float = 1.0) -> float:
    """Value of process at time t_s: the trajectory is materialised from
    t=0 on a fixed dt_s grid and the sample at t_s is returned. The grid has
    round(t_s/dt_s)+1 points, so t_s=0 returns the t=0 sample exactly. (The
    previous max(2, ...) grid returned the SECOND point at t_s=0, i.e. applied
    one full drift step of dt_s -- 90 s for superconducting, 5 s for trapped
    ion -- to a device that had not aged at all.)"""
    n_steps = max(1, int(round(max(t_s, 0.0) / dt_s)) + 1)
    traj = sample_trajectory(process, master_seed, salt, n_steps, dt_s, base_value)
    return float(traj[-1])


def evaluate_deviation_at(process: DriftProcess, master_seed: int, salt: Any,
                          t_s: float, dt_s: float = 1.0) -> float:
    """Drift of a zero-mean additive process between t=0 and t_s, i.e.
    x(t_s) - x(0). Zero at t_s=0 by construction, so a DeviceTruth's stored
    base value is the value at t=0 and drift only accumulates afterwards."""
    n_steps = max(1, int(round(max(t_s, 0.0) / dt_s)) + 1)
    traj = sample_trajectory(process, master_seed, salt, n_steps, dt_s, 0.0)
    return float(traj[-1] - traj[0])


# =============================================================================
# 5. ARCHITECTURE PROFILES
# =============================================================================

def get_architecture_profile(architecture: str) -> ArchitectureProfile:
    """Canonical, literature-cross-referenced typical values for one architecture."""
    P = MODEL_PROVENANCE
    if architecture == "superconducting":
        return ArchitectureProfile(
            architecture="superconducting",
            T1_s=P["sc.T1.baseline_s"]["value"], T2_s=P["sc.T2.baseline_s"]["value"],
            gate_duration_ns=P["sc.gate.duration_ns"]["value"],
            spam_p0_given_1=P["sc.spam.p0_given_1"]["value"],
            spam_p1_given_0=P["sc.spam.p1_given_0"]["value"],
            gate_error_total=P["sc.gate.stochastic_error"]["value"],
            extra={
                "coherent_fraction": P["sc.gate.coherent_fraction"]["value"],
                "T1_drift_timescale_s": P["sc.T1_drift.timescale_s"]["value"],
                "T1_drift_amplitude_frac": P["sc.T1_drift.amplitude_frac"]["value"],
                "detuning_drift_amplitude_rad_s": P["sc.detuning_drift.amplitude_rad_s"]["value"],
                "crosstalk_angle_rad": P["sc.crosstalk.coherent_angle_rad_uncompensated"]["value"],
                "crosstalk_zz_shift_khz_min": P["sc.crosstalk.zz_shift_khz_min"]["value"],
                "crosstalk_zz_shift_khz_max": P["sc.crosstalk.zz_shift_khz_max"]["value"],
                "crosstalk_zz_shift_khz_typical": P["sc.crosstalk.zz_shift_khz_typical"]["value"],
                "detuning_1_over_f_alpha": P["sc.detuning_1_over_f.alpha"]["value"],
            },
        )
    if architecture == "trapped_ion":
        return ArchitectureProfile(
            architecture="trapped_ion",
            T1_s=P["ti.T1.baseline_s"]["value"], T2_s=P["ti.T2.baseline_s"]["value"],
            gate_duration_ns=P["ti.gate.duration_ns"]["value"],
            spam_p0_given_1=P["ti.spam.p0_given_1"]["value"],
            spam_p1_given_0=P["ti.spam.p1_given_0"]["value"],
            gate_error_total=P["ti.gate.error_total"]["value"],
            extra={
                "coherent_fraction": P["ti.gate.coherent_fraction"]["value"],
                "detuning_drift_amplitude_rad_s": P["ti.detuning_drift.amplitude_rad_s"]["value"],
                "heating_axial_quanta_per_s": P["ti.heating.baseline_axial_quanta_per_s"]["value"],
                "heating_radial_quanta_per_s": P["ti.heating.baseline_radial_quanta_per_s"]["value"],
                "heating_distance_exponent": P["ti.heating.distance_scaling_exponent"]["value"],
                "heating_to_gate_error_coeff": P["ti.heating_to_gate_error.mapping_coefficient"]["value"],
                "crosstalk_angle_rad": P["ti.crosstalk.coherent_angle_rad_high"]["value"],
                "T2_record_s": P["ti.T2.record_s"]["value"],
            },
        )
    raise ValueError(f"Unknown architecture: {architecture!r}; must be one of {ARCHITECTURES}")


# =============================================================================
# 6. DEVICE TRUTH GENERATION
# =============================================================================

# Log-uniform truth-sampling ranges: the population of devices a cloud user
# could plausibly be assigned. These are the RANGE OF TRUE VALUES, and are a
# separate concept from 1_inversion.py's ARCH_DEFAULTS T1_min_s/T1_max_s,
# which are optimizer safety bounds (deliberately generous, so a fit is never
# silently clipped). The two must not be conflated: sampling truth across an
# optimizer bound's full span generates devices that do not exist.
#
# WHICH QUANTITY IS SAMPLED: T1 and eps are sampled directly. The second
# coherence key is "T_phi_s" -- the PURE DEPHASING time -- and NOT T2. T2 is
# DERIVED from it via the standard Lindblad decomposition
#     1/T2 = 1/(2*T1) + 1/T_phi,
# then capped at 2*T1. The decomposition is correct and is not being changed;
# the key was simply misnamed "T2_s", which made the declared range a claim
# about a quantity the sampler never draws. Concretely, for superconducting
# the old key declared 40-200 us while the T2 values actually produced span
# ~32-159 us, with 8.2% of qubits landing BELOW the declared minimum. The
# literature anchoring below therefore describes T_phi, and the derived T2
# span is recorded alongside it so the distribution that experiments actually
# see is documented rather than inferred.
#
# superconducting: T_phi 40-200 us. Reading the Sharma & Lau (arXiv:2607.11637)
#   Table 4 Heron bands (T1 100-300 us, T2 80-200 us) back through the
#   decomposition above gives T_phi ~133-300 us, so the sampled range is
#   deliberately broader toward WEAKER coherence than Heron: it also
#   represents older/weaker cloud transmons. (An earlier version of this
#   comment called the range consistent with uncited IBM median T1/T2 figures;
#   read back through the decomposition, those medians did not support it.)
#   DERIVED T2, generate_device('superconducting', 5000 qubits, seed=12345):
#   32.1-158.8 us, median 69.6 us, 61.4% below 80 us, none above 160 us.
# trapped_ion: IonQ's published Aria specification states T1 10-100 s and
#   T2 ~1 s; T1's upper bound widened to 200 s for per-qubit spread. The
#   previous (100, 10_000) s range had no citation and put its upper end
#   50-100x above any measured cloud trapped-ion T1 -- it is what produced
#   true-T1 ~ 100 s paired with a ~10,000 s prior. Because T1 >> T_phi here,
#   the 1/(2*T1) term is negligible and T2 ~ T_phi to within ~1%.
#   DERIVED T2, generate_device('trapped_ion', 5000 qubits, seed=12345):
#   0.100-2.96 s, median 0.53 s.
#
# The numbers are registered in MODEL_PROVENANCE ("sc./ti.truth_range.*",
# with evidence class and rationale) and _TRUTH_RANGES is built from them.
_TRUTH_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    arch: {q: tuple(MODEL_PROVENANCE[f"{pfx}.truth_range.{q}"]["value"])
           for q in ("T1_s", "T_phi_s", "eps")}
    for arch, pfx in (("superconducting", "sc"), ("trapped_ion", "ti"))
}


def generate_device(
    architecture: str,
    n_qubits: int,
    seed: int,
    regime: str = "ideal",
    dw_max_rad_s: Optional[float] = None,
    overrides: Optional[Dict[str, Any]] = None,
    crosstalk_scale: float = 1.0,
    detuning_1_over_f: bool = False,
    enable_heating_gate_error: bool = False,
    spam_spread_frac: float = 0.0,
    perturbation_frac: Optional[float] = None,
    dw_population: str = "estimator_window",
    t1_telegraph: bool = False,
    quasi_static_dephasing: bool = False,
    detuning_during_gates: bool = False,
) -> DeviceTruth:
    """Sample a complete, hidden DeviceTruth.

    Per-qubit T1, T_phi and epsilon are drawn log-uniform from _TRUTH_RANGES
    (keys "T1_s", "T_phi_s", "eps"). T2 is NOT sampled -- it is DERIVED from
    the sampled T1 and T_phi via 1/T2 = 1/(2*T1) + 1/T_phi and then capped at
    2*T1, so the T2 values a caller sees are narrower than, and shifted below,
    the declared T_phi range (superconducting: T_phi 40-200 us in, T2 ~32-159
    us out). delta_omega is drawn according to dw_population (below); by
    default |delta_omega| ~ U[0.2, 1] * ceiling with a random sign, where the
    ceiling is an aliasing bound that accounts for prior inflation (see the
    comment at the sampling site). regime='ideal' attaches no drift/crosstalk processes at all
    (stationary Markovian noise -- the chi2/dof null baseline) and is
    unaffected by every parameter below; regime='nisq' attaches the
    architecture-specific mechanisms in Sections 6-8 of the module spec,
    each traceable to MODEL_PROVENANCE.

    crosstalk_scale: multiplies every crosstalk kick angle (both
        architectures). Default 1.0 reproduces the literature-grounded
        magnitude as-is; pass e.g. 0.5 or 0.0 to re-run a NISQ experiment at
        reduced/zero crosstalk as an explicit sensitivity check (see the
        module docstring's Sensitivity Sweep Parameters section).
    detuning_1_over_f: superconducting only. When False (default), the
        detuning-drift mechanism is unchanged from before this feature was
        added (a single-pole low-frequency-colored process, see
        _proc_lowpass_filtered). When True, uses the genuine 1/f^alpha
        generator (_proc_one_over_f) instead, with alpha from
        sc.detuning_1_over_f.alpha (Class A but device-specific -- see that
        entry's notes -- so alpha should be swept, not fixed, when this is
        enabled).
    enable_heating_gate_error: trapped-ion only, default False. When False,
        Teller et al.'s (PRL 126, 230505) motional-heating rates are still recorded in
        DeviceTruth.architecture_specific for reference, but the ungrounded
        heating-to-gate-error mapping (Class C, self-documented as "not a
        validated physical prediction") contributes zero extra gate error.
        Set True only for an explicitly-labelled ablation study, never in a
        headline figure.
    spam_spread_frac: per-qubit readout-error heterogeneity, default 0.0.
        At 0.0 every qubit gets the architecture's single base SPAM pair and
        NO random numbers are consumed, so all pre-existing results are
        bit-for-bit unchanged. At f > 0 each qubit's p0|1 and p1|0 are drawn
        independently log-uniform in [(1-f)*base, (1+f)*base] and clipped to
        [0,1]. Class C stress-test knob with no literature value -- sweep it,
        do not fix it (see spam.per_qubit_spread_frac). Leave it at 0.0 for
        regime='ideal', which is defined as a correctly-specified forward
        model and whose prior passes SPAM through unperturbed.
    perturbation_frac: default None, which reproduces the historical
        behaviour exactly: the "estimator_window" ceiling is divided by
        (1 + resolve_calibration_perturbation(architecture, regime)). Pass the
        same explicit value given to generate_calibration_prior(...,
        perturbation_frac=...) so that the ceiling matches the prior envelope
        actually used. Ignored when dw_max_rad_s is given or when
        dw_population="independent".
    dw_population: how delta_omega is drawn.
        "estimator_window" (default, bit-for-bit the historical draw):
            |dw| ~ U[0.2, 1] * ceiling with a random sign, ceiling =
            0.9*pi / (0.5*T2) / (1 + perturbation_frac) per qubit (or
            dw_max_rad_s if given). Without dw_max_rad_s, and with
            perturbation_frac equal to the prior's envelope, every drawn
            (static) value lies inside the window 1_inversion.py resolves
            for any prior that envelope can produce.
        "independent": dw ~ U(-W, W) with W = K * 0.9*pi / (0.5*T2_ref),
            T2_ref = get_architecture_profile(architecture).T2_s (the
            architecture baseline T2) and K from
            dw_population.independent_halfwidth_factor in MODEL_PROVENANCE.
            This is an architecture-level population independent of each
            qubit's T2, of the prior and of the estimator window: it includes
            |dw| ~ 0 and values beyond the window (i.e. aliasing instances).
            It is drawn as random sign * U(0, W), which has the same
            distribution as U(-W, W) and consumes the same random numbers as
            the default draw, so every later draw is unchanged.
            Incompatible with dw_max_rad_s (raises ValueError).
        "literature": dw ~ U(-W, W) with W = 2*pi times
            sc/ti.dw_population.literature_halfwidth_hz (measured frequency
            excursions: 15 kHz, Dhieb et al.; 0.5 Hz, Proctor et al.), or
            2*pi*overrides["dw_halfwidth_hz"]. Independent of each qubit's
            truth, of the prior and of the estimator window; same draw
            consumption as the other branches. Incompatible with
            dw_max_rad_s (raises ValueError).
    t1_telegraph: superconducting, regime='nisq' only (ignored otherwise,
        like detuning_1_over_f); default False, which changes nothing. When
        True, attaches drift_processes["T1_telegraph"]: two-state T1
        switching (Burnett et al., npj QI 5, 54), combined multiplicatively
        with the OU log-T1 drift in evaluate_truth_at(). The switching rate
        is drawn from the Class A range sc.T1_telegraph.switching_rate_hz on
        a separate RNG stream (no other draw moves); the switching amplitude
        sc.T1_telegraph.amplitude_frac is Class C, hence opt-in.
    quasi_static_dephasing: superconducting, regime='nisq' only (ignored
        otherwise); default False, which changes nothing. When True, each
        qubit draws a Ramsey/echo ratio r from sc.quasi_static.t2_ramsey_
        over_echo (Place et al., own RNG stream, no other draw moves) and
        gets a quasi-static detuning std sigma = sqrt(2*(1 - r))/(r*T2),
        stored in architecture_specific["quasi_static_dw_std_rad_s"].
        simulate_circuit() then gives every shot its own static offset
        delta ~ N(0, sigma) (exact per-shot sampling) in any circuit where
        free evolution is followed by a pulse (Ramsey, echo). The truth T2
        stays the Markovian (echo) T2; T2* = r*T2 is not a truth parameter.
    detuning_during_gates: regime='nisq' only (ignored otherwise); default
        False. When True, the detuning (plus any quasi-static offset) also
        acts during every native pulse, as Rz(dw*t_g/2) G Rz(dw*t_g/2)
        (architecture_specific["detuning_during_gates"]). This is a
        coherent error that does NOT cancel in the gate-repetition probe's
        G*G^-1 pair; effective_gate_epsilon() deliberately excludes it, so a
        benchmark measures how well the probe rejects it.
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Invalid architecture: {architecture!r}")
    if n_qubits <= 0:
        raise ValueError(f"n_qubits must be positive, got {n_qubits}")
    regime = _canonical_regime(regime)
    ov = overrides or {}
    profile = get_architecture_profile(architecture)
    ranges = _TRUTH_RANGES[architecture]
    rng = _sub_rng(seed, "truth", architecture, n_qubits)

    def _log_uniform(lo: float, hi: float, size: int) -> np.ndarray:
        return 10.0 ** rng.uniform(np.log10(lo), np.log10(hi), size=size)

    T1_lo, T1_hi = ov.get("T1_range_s", ranges["T1_s"])
    # "T2_range_s" is a DEPRECATED ALIAS for "T_phi_range_s". It always bounded
    # T_phi, never T2 -- the name was wrong, the behaviour was not -- so it is
    # still honoured (identical numerics) but warns, rather than breaking
    # existing callers silently or, worse, being silently ignored.
    if "T2_range_s" in ov and "T_phi_range_s" not in ov:
        warnings.warn(
            "generate_device(overrides={'T2_range_s': ...}) is deprecated: the "
            "key bounds the PURE DEPHASING time T_phi, not T2 (T2 is derived "
            "as 1/T2 = 1/(2*T1) + 1/T_phi and capped at 2*T1). Rename it to "
            "'T_phi_range_s'; the numerical behaviour is unchanged.",
            DeprecationWarning, stacklevel=2)
    T_phi_lo, T_phi_hi = ov.get("T_phi_range_s",
                                ov.get("T2_range_s", ranges["T_phi_s"]))
    eps_lo, eps_hi = ov.get("eps_range", ranges["eps"])

    T1 = _log_uniform(T1_lo, T1_hi, n_qubits)
    T_phi = _log_uniform(T_phi_lo, T_phi_hi, n_qubits)
    T2 = np.minimum(1.0 / (1.0 / (2.0 * T1) + 1.0 / T_phi), 2.0 * T1)
    eps = _log_uniform(eps_lo, eps_hi, n_qubits)

    # Detuning is drawn below each qubit's OWN aliasing ceiling. 1_inversion.py
    # recovers delta_omega unambiguously only for |dw| <= 0.9*pi/t1, where t1 is
    # the shortest Ramsey delay. Using a single median-T2 ceiling for the whole
    # register would hand any qubit with a shorter-than-median T2 a detuning it
    # cannot resolve even in principle, which shows up as an estimator failure
    # that is really a construction artifact.
    #
    # THE CEILING MUST ACCOUNT FOR PRIOR INFLATION. The engine builds its delay
    # grid from T2_PRIOR, not T2_TRUE, so the shortest Ramsey delay is
    # 0.5*T2_prior. Accumulated phase is then bounded by
    #     |dw| * 0.5*T2_prior  <=  0.9*pi * (T2_prior / T2_true),
    # which exceeds pi -- i.e. aliases -- whenever the prior/truth ratio
    # exceeds 1/0.9 = 1.111. generate_calibration_prior() draws that ratio
    # uniform in [1-frac, 1+frac], so the failure mode switches on exactly when
    # frac > 0.111: measured over 400 seeds, superconducting/nisq (frac=0.25)
    # put 29.0% of instances above the 1.111 ratio and 2.0% into actual
    # aliasing, while superconducting/ideal (frac=0.10) gave 0.0% of both --
    # structurally, not by luck, since 1.10 < 1.111.
    #
    # Dividing the ceiling by the worst-case inflation (1+frac) makes the bound
    # hold for every draw the prior can produce, so no instance is
    # unrecoverable by construction at the prior time t0 with the default
    # prior envelope and the prior-informed delay grid. Drift after t0 (e.g.
    # nisq at T_MEASURE_S=3600 s: 4/300 SC, 19/300 TI aliased in the
    # 300-instance benchmark), the arch_default
    # (cold-start) grid, a larger swept perturbation_frac, or
    # dw_population='independent' can alias -- flagged by the benchmark
    # dw_aliased column. An explicit dw_max_rad_s still overrides
    # this entirely.
    if dw_population == "estimator_window":
        if dw_max_rad_s is not None:
            dw_max_per_q = np.full(n_qubits, float(dw_max_rad_s))
        else:
            if perturbation_frac is None:
                _prior_frac = resolve_calibration_perturbation(architecture, regime)
            else:
                if not 0.0 <= perturbation_frac < 1.0:
                    raise ValueError(f"perturbation_frac must be in [0,1), got {perturbation_frac}")
                _prior_frac = float(perturbation_frac)
            dw_max_per_q = 0.9 * np.pi / (0.5 * T2) / (1.0 + _prior_frac)
        dw = rng.choice([-1.0, 1.0], size=n_qubits) * rng.uniform(
            0.2 * dw_max_per_q, dw_max_per_q, size=n_qubits)
    elif dw_population == "independent":
        if dw_max_rad_s is not None:
            raise ValueError("dw_max_rad_s cannot be combined with dw_population='independent'")
        # Architecture-level half-width, independent of the sampled T2, the
        # prior and the estimator window (see the docstring and the
        # dw_population.independent_halfwidth_factor provenance entry).
        _K = MODEL_PROVENANCE["dw_population.independent_halfwidth_factor"]["value"]
        _W = _K * 0.9 * np.pi / (0.5 * profile.T2_s)
        # sign * U(0, W) is distributed as U(-W, W) and consumes the same
        # draws as the estimator_window branch, keeping later draws aligned.
        dw = rng.choice([-1.0, 1.0], size=n_qubits) * rng.uniform(
            0.0, _W, size=n_qubits)
    elif dw_population == "literature":
        if dw_max_rad_s is not None:
            raise ValueError("dw_max_rad_s cannot be combined with dw_population='literature'")
        _W_hz = ov.get("dw_halfwidth_hz", MODEL_PROVENANCE[
            f"{'sc' if architecture == 'superconducting' else 'ti'}"
            ".dw_population.literature_halfwidth_hz"]["value"])
        if not (np.isfinite(_W_hz) and _W_hz > 0):
            raise ValueError(f"dw_halfwidth_hz must be positive and finite, got {_W_hz!r}")
        # Same draw consumption as the other branches (later draws aligned).
        dw = rng.choice([-1.0, 1.0], size=n_qubits) * rng.uniform(
            0.0, 2.0 * np.pi * float(_W_hz), size=n_qubits)
    else:
        raise ValueError(f"Unknown dw_population {dw_population!r}; expected "
                         "'estimator_window', 'independent' or 'literature'")

    if not 0.0 <= spam_spread_frac < 1.0:
        raise ValueError(f"spam_spread_frac must be in [0,1), got {spam_spread_frac}")
    if spam_spread_frac > 0.0:
        # Log-uniform, matching how every other per-qubit truth parameter in
        # this function is drawn. Clipped to [0,1] because these are
        # probabilities; with the base values and f < 1 the clip never binds,
        # but it is cheap insurance against an override that pushes the base
        # near 1.
        def _spam_draw(base: float) -> np.ndarray:
            return np.clip(_log_uniform((1.0 - spam_spread_frac) * base,
                                        (1.0 + spam_spread_frac) * base,
                                        n_qubits), 0.0, 1.0)
        p01 = _spam_draw(profile.spam_p0_given_1)
        p10 = _spam_draw(profile.spam_p1_given_0)
    else:
        # Consume no randomness: this branch is what keeps spam_spread_frac=0.0
        # bit-for-bit identical to the behaviour before the knob existed.
        p01 = np.full(n_qubits, profile.spam_p0_given_1)
        p10 = np.full(n_qubits, profile.spam_p1_given_0)

    drift: Dict[str, DriftProcess] = {}
    crosstalk: List[CrosstalkMechanism] = []
    arch_specific: Dict[str, Any] = {}

    if regime == "nisq":
        if architecture == "superconducting":
            drift["T1"] = DriftProcess("ou", {
                "mean": None,  # filled per-qubit at evaluation time (base_value)
                "sigma": profile.extra["T1_drift_amplitude_frac"],  # fractional units on log(T1)
                "theta": 1.0 / profile.extra["T1_drift_timescale_s"],
            }, seed_salt="T1_ou")
            if t1_telegraph:
                # One Lorentzian rate 1/tau_0 per device, log-uniform over the
                # reported range, from its own RNG stream so that the main
                # truth stream (and hence every other draw) is not advanced.
                lo_hz, hi_hz = MODEL_PROVENANCE["sc.T1_telegraph.switching_rate_hz"]["value"]
                inv_tau0 = float(10.0 ** _sub_rng(seed, "T1_telegraph", architecture, n_qubits)
                                 .uniform(np.log10(lo_hz), np.log10(hi_hz)))
                drift["T1_telegraph"] = DriftProcess("telegraph", {
                    "rate": 0.5 * inv_tau0,  # per-direction rate r = (1/tau_0)/2
                    "amplitude_frac": MODEL_PROVENANCE["sc.T1_telegraph.amplitude_frac"]["value"],
                    "lorentzian_rate_hz": inv_tau0,
                }, seed_salt="T1_telegraph")
            if detuning_1_over_f:
                drift["detuning"] = DriftProcess("one_over_f", {
                    "amplitude": profile.extra["detuning_drift_amplitude_rad_s"],
                    "alpha": profile.extra["detuning_1_over_f_alpha"],
                    # cutoff_hz sets evaluate_truth_at()'s grid; f_min_hz fixes
                    # the band's low edge (one-day window, phenomenological)
                    # so the trajectory never depends on how far it is queried.
                    "cutoff_hz": 1.0 / profile.extra["T1_drift_timescale_s"],
                    "f_min_hz": MODEL_PROVENANCE["sc.detuning_1_over_f.f_min_hz"]["value"],
                }, seed_salt="detuning_1overf")
            else:
                drift["detuning"] = DriftProcess("lowpass_filtered", {
                    "amplitude": profile.extra["detuning_drift_amplitude_rad_s"],
                    "cutoff_hz": 1.0 / profile.extra["T1_drift_timescale_s"],
                }, seed_salt="detuning_lpf")
            if n_qubits >= 2:
                # Per-pair ALWAYS-ON ZZ SHIFT (zeta), drawn uniformly over the
                # span of the four values measured by Mueller et al.
                # (arXiv:2504.08313, Table I). This is the longitudinal shift
                # that actually generates a conditional phase; it is NOT the
                # transverse exchange coupling J, which an earlier version of
                # this module used by mistake and which is 1-2 orders of
                # magnitude larger. Uniform rather than Gaussian because four
                # measured pairs (one of them a clear outlier) cannot support a
                # shape claim -- see sc.crosstalk.zz_shift_khz_typical.
                pairs = [(i, i + 1) for i in range(n_qubits - 1)]
                zz_khz = rng.uniform(profile.extra["crosstalk_zz_shift_khz_min"],
                                     profile.extra["crosstalk_zz_shift_khz_max"],
                                     size=len(pairs))
                gate_duration_s = profile.gate_duration_ns * 1e-9
                for (a, b), zz in zip(pairs, zz_khz):
                    angle = crosstalk_scale * 2.0 * np.pi * (zz * 1e3) * gate_duration_s
                    crosstalk.append(CrosstalkMechanism(
                        enabled=True, angle_rad=float(angle), axis="z",
                        affected_pairs=[(a, b)]))
            arch_specific["coherent_fraction"] = profile.extra["coherent_fraction"]

        elif architecture == "trapped_ion":
            # Additive zero-mean OU deviation from the base detuning. "sigma"
            # is the STATIONARY std; evaluate_truth_at() converts it to the
            # diffusion coefficient with sqrt(2*theta), exactly as for T1.
            drift["detuning"] = DriftProcess("ou", {
                "mean": 0.0, "sigma": profile.extra["detuning_drift_amplitude_rad_s"],
                "theta": 1.0 / MODEL_PROVENANCE["ti.detuning_drift.correlation_time_s"]["value"],
            }, seed_salt="detuning_ou")
            # No power-line term on detuning: the cited 60 Hz measurement is of
            # the MOTIONAL-mode secular frequency, not qubit detuning, and a
            # 16.7 ms period sampled once per Ramsey delay (6-180 cycles long)
            # would produce a static offset where the real effect is contrast
            # decay. See the REMOVED note in MODEL_PROVENANCE.
            if n_qubits >= 2:
                crosstalk.append(CrosstalkMechanism(
                    enabled=True,
                    angle_rad=crosstalk_scale * profile.extra["crosstalk_angle_rad"], axis="x",
                    affected_pairs=[(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]))
            heating_axial = profile.extra["heating_axial_quanta_per_s"]
            arch_specific["heating_axial_quanta_per_s"] = heating_axial
            arch_specific["heating_extra_gate_error"] = float(
                heating_axial * profile.extra["heating_to_gate_error_coeff"]
            ) if enable_heating_gate_error else 0.0
            arch_specific["coherent_fraction"] = profile.extra["coherent_fraction"]

        if quasi_static_dephasing and architecture == "superconducting":
            # One measured T2R/T2E ratio per qubit (Place et al., Table S1),
            # from its own RNG stream so no other draw moves. sigma sets the
            # Ramsey 1/e time to r*T2 for the envelope
            # exp(-t/T2 - sigma^2 t^2 / 2) (see the provenance entry).
            ratios = np.asarray(MODEL_PROVENANCE[
                "sc.quasi_static.t2_ramsey_over_echo"]["value"], dtype=float)
            r_q = _sub_rng(seed, "quasi_static", architecture, n_qubits).choice(
                ratios, size=n_qubits)
            arch_specific["t2_ramsey_over_echo"] = r_q
            arch_specific["quasi_static_dw_std_rad_s"] = (
                np.sqrt(2.0 * (1.0 - r_q)) / (r_q * T2))
        if detuning_during_gates:
            arch_specific["detuning_during_gates"] = True

    return DeviceTruth(
        architecture=architecture, n_qubits=n_qubits, seed=seed, regime=regime,
        T1_s=T1, T2_s=T2, delta_omega_rad_s=dw, epsilon_gate=eps,
        spam_p0_given_1=p01, spam_p1_given_0=p10,
        gate_duration_ns=profile.gate_duration_ns,
        drift_processes=drift, crosstalk=crosstalk, architecture_specific=arch_specific,
    )


def evaluate_truth_at(truth: DeviceTruth, qubit: int, t_s: float) -> Dict[str, float]:
    """Time-evaluate qubit `qubit`'s parameters at time t_s, applying any
    attached drift processes. regime='ideal' has no drift processes, so this
    is a no-op returning the stationary values."""
    T1 = float(truth.T1_s[qubit])
    T2 = float(truth.T2_s[qubit])
    dw = float(truth.delta_omega_rad_s[qubit])

    if "T1" in truth.drift_processes:
        proc = truth.drift_processes["T1"]
        log_T1_0 = np.log(T1)
        amplitude_frac = proc.params.get("sigma", 0.0)  # intended STATIONARY std of log(T1)
        theta = proc.params.get("theta", 1.0)
        # An OU process's stationary variance is sigma_diffusion^2/(2*theta),
        # not sigma_diffusion^2 -- amplitude_frac is the stationary std we
        # actually want (see sc.T1_drift.amplitude_frac's provenance note:
        # "T1 fluctuates by roughly this fraction"), so the diffusion
        # coefficient fed to the OU recursion must be scaled by sqrt(2*theta).
        # Skipping this conversion previously turned an intended ~30% swing
        # into a several-thousand-x swing (theta=1/1800 makes sqrt(2*theta)
        # ~0.033, a >30x amplification if sigma is used unscaled).
        sigma = amplitude_frac * np.sqrt(2.0 * theta) if theta > 0 else amplitude_frac
        # Resolution is fixed by the PROCESS's own correlation time (1/theta),
        # never by the queried t_s -- evaluate_truth_at(truth, q, 50) and
        # evaluate_truth_at(truth, q, 100) must walk the SAME discretized
        # trajectory (the second simply continuing further), not two
        # independently-resolved, mutually inconsistent re-samplings of it.
        dt_fixed = max(1.0, (1.0 / theta) / 20.0) if theta > 0 else 1.0
        walked = evaluate_at(DriftProcess("ou", {"mean": log_T1_0, "sigma": sigma, "theta": theta},
                                          seed_salt=proc.seed_salt),
                             truth.seed, ("T1", qubit), t_s, log_T1_0, dt_s=dt_fixed)
        T1 = float(np.exp(walked))
    if "T1_telegraph" in truth.drift_processes:
        # Two-state switching, multiplicative on top of the OU drift: T1 is
        # scaled by x(t)/x(0) of a telegraph trajectory with levels 1 and
        # 1 + amplitude_frac, so T1(0) is the sampled (OU) value and each
        # switch moves T1 between two levels a factor (1 + amplitude_frac)
        # apart. Grid fixed by the process's correlation time 1/(2*rate).
        proc = truth.drift_processes["T1_telegraph"]
        rate = proc.params.get("rate", 1e-3)
        dt_fixed = max(1.0, (1.0 / (2.0 * rate)) / 20.0) if rate > 0 else 1.0
        n_steps = max(1, int(round(max(t_s, 0.0) / dt_fixed)) + 1)
        x_tel = sample_trajectory(proc, truth.seed, ("T1_telegraph", qubit),
                                  n_steps, dt_fixed, 1.0)
        T1 = T1 * float(x_tel[-1] / x_tel[0])
    if "detuning" in truth.drift_processes:
        # Every detuning process is a stationary zero-mean fluctuation x(t)
        # whose amplitude/sigma is its STATIONARY std, and the truth drifts
        # by its change since t=0: dw(t) = dw_base + [x(t) - x(0)]. So
        # dw(0) = dw_base exactly, and the drift's spread grows to
        # sqrt(2)*sigma over a few correlation times, for both architectures.
        proc = truth.drift_processes["detuning"]
        if proc.process_type in ("lowpass_filtered", "one_over_f"):
            cutoff_hz = proc.params.get("cutoff_hz", 1e-3)
            dt_fixed = max(1.0, (1.0 / max(cutoff_hz, 1e-9)) / 20.0)
            dev_proc = proc
        else:
            theta_dw = proc.params.get("theta", 1.0)
            dt_fixed = max(1.0, (1.0 / theta_dw) / 20.0) if theta_dw > 0 else 1.0
            sigma_stat = proc.params.get("sigma", 0.0)
            dev_proc = DriftProcess("ou", {
                "mean": 0.0, "x0": "stationary", "theta": theta_dw,
                "sigma": sigma_stat * np.sqrt(2.0 * theta_dw) if theta_dw > 0 else sigma_stat,
            }, seed_salt=proc.seed_salt)
        dw = dw + evaluate_deviation_at(dev_proc, truth.seed, ("detuning", qubit),
                                        t_s, dt_s=dt_fixed)

    T2 = min(T2, 2.0 * T1)
    return {"T1_s": T1, "T2_s": T2, "delta_omega_rad_s": dw,
            "epsilon_gate": float(truth.epsilon_gate[qubit]),
            "spam_p0_given_1": float(truth.spam_p0_given_1[qubit]),
            "spam_p1_given_0": float(truth.spam_p1_given_0[qubit])}


# =============================================================================
# 7. CALIBRATION PRIOR (no truth leakage, no inversion mathematics)
# =============================================================================

# Default staleness-envelope half-width, resolved per (architecture, regime).
# All four values are Class C -- see the calib.perturbation.* entries in
# MODEL_PROVENANCE for the full derivation of each and for the two candidate
# anchors that were considered and withdrawn. An explicit perturbation_frac
# argument always overrides this table.
_CALIB_PERTURBATION: Dict[Tuple[str, str], float] = {
    ("superconducting", "ideal"): 0.10,
    ("superconducting", "nisq"):  0.25,
    ("trapped_ion",     "ideal"): 0.05,
    ("trapped_ion",     "nisq"):  0.15,
}

# Used when a (architecture, regime) pair is somehow absent from the table.
# Falling back to the historical flat default is safer than raising, because
# this function sits on the path of every experiment in the repository.
_CALIB_PERTURBATION_FALLBACK = 0.10


def resolve_calibration_perturbation(architecture: str, regime: str) -> float:
    """Default envelope half-width for one (architecture, regime) pair."""
    return _CALIB_PERTURBATION.get((architecture, _canonical_regime(regime)),
                                   _CALIB_PERTURBATION_FALLBACK)


def generate_calibration_prior(
    truth: DeviceTruth, qubit: int = 0, t0_s: float = 0.0,
    perturbation_frac: Optional[float] = None,
    calibration_seed: Optional[int] = None,
    spam_perturbation_frac: float = 0.0,
) -> CalibrationPrior:
    """Produce what a provider's calibration log would report for this qubit.

    A real calibration value is measured at some earlier time and is already
    stale by the time a job runs: T1, T2 and gate error drift between
    calibration cycles, so the reported number is close to -- but not equal
    to -- the parameter that is true right now. That gap is the entire reason
    a prior is not the same thing as ground truth, and it is what keeps an
    ideal-regime experiment from being circular: an inversion handed its own
    answer as a starting point has not been tested.

    The gap is modelled directly as a bounded multiplicative envelope: each
    parameter is independently drawn uniform in
    [1 - perturbation_frac, 1 + perturbation_frac] times its true value.

    perturbation_frac defaults to None, in which case the half-width is looked
    up in _CALIB_PERTURBATION by (truth.architecture, truth.regime): 0.10 /
    0.25 for superconducting ideal/nisq and 0.05 / 0.15 for trapped ion. A
    single flat default was wrong in both directions at once -- it understated
    how stale a superconducting calibration record gets between cycles (real
    transmon T1 wanders 20-26% over hours; Burnett et al., npj QI 5, 54) while
    overstating the gap on an ion trap, which has no TLS bath to drive it.
    Passing an explicit float overrides the table, and 0.0 still reproduces
    truth exactly, so sweeps are unaffected.

    This is a STRESS-TEST envelope, not a fit to a measured staleness
    distribution. ALL FOUR DEFAULTS ARE CLASS C. No published dataset
    quantifies calibration-log staleness for cloud-deployed hardware, because
    measuring it requires exactly the independent ground truth that cloud
    users do not have; the cited work bounds the scale of the underlying
    parameter wander, and the conversion into an envelope is this module's
    own. Callers should therefore SWEEP perturbation_frac and report that
    their result is insensitive to it, rather than treating any single value
    as correct. See the calib.perturbation.* entries in MODEL_PROVENANCE for
    each derivation, including the two candidate anchors that were considered
    and withdrawn.

    SPAM: with spam_perturbation_frac = 0.0 (the default) the prior's
    spam_p0_given_1_prior / spam_p1_given_0_prior are copied verbatim from
    truth. This is an oracle-SPAM idealisation -- the inversion, which treats
    SPAM as a known constant in its forward models, is given perfect readout
    calibration. With spam_perturbation_frac = f > 0 each readout error is
    instead truth * U(1-f, 1+f) (calib.spam_perturbation, Class C), drawn
    from a separate RNG stream so the T1/T2/epsilon draws are unchanged.
    3_benchmark.py uses f = the prior's envelope; the ideal regime keeps
    oracle SPAM. Deterministic misspecification is tested separately by the
    benchmark flags --spam-scale / --spam-swap.
    """
    if perturbation_frac is None:
        perturbation_frac = resolve_calibration_perturbation(truth.architecture,
                                                             truth.regime)
    if not 0.0 <= perturbation_frac < 1.0:
        raise ValueError(f"perturbation_frac must be in [0,1), got {perturbation_frac}")

    seed = calibration_seed if calibration_seed is not None else (truth.seed + 9_000_000)
    rng = _sub_rng(seed, "calib_prior", qubit, t0_s)
    tv = evaluate_truth_at(truth, qubit, t0_s)

    lo, hi = 1.0 - perturbation_frac, 1.0 + perturbation_frac

    T1_prior = float(tv["T1_s"] * rng.uniform(lo, hi))
    # The prior must itself be physically admissible: a T2 prior above 2*T1
    # would seed 1_inversion.py with a delay grid its own hard bound rejects.
    T2_prior = float(min(tv["T2_s"] * rng.uniform(lo, hi), 2.0 * T1_prior))
    eps_prior = float(np.clip(effective_gate_epsilon(truth, qubit, t0_s)
                              * rng.uniform(lo, hi), 1e-9, 0.5))

    if not 0.0 <= spam_perturbation_frac < 1.0:
        raise ValueError(f"spam_perturbation_frac must be in [0,1), "
                         f"got {spam_perturbation_frac}")
    p01_prior = float(tv["spam_p0_given_1"])
    p10_prior = float(tv["spam_p1_given_0"])
    if spam_perturbation_frac > 0.0:
        # Own stream: the T1/T2/epsilon draws above are unchanged.
        rng_spam = _sub_rng(seed, "calib_prior_spam", qubit, t0_s)
        s_lo, s_hi = 1.0 - spam_perturbation_frac, 1.0 + spam_perturbation_frac
        p01_prior = float(np.clip(p01_prior * rng_spam.uniform(s_lo, s_hi), 0.0, 1.0))
        p10_prior = float(np.clip(p10_prior * rng_spam.uniform(s_lo, s_hi), 0.0, 1.0))

    return CalibrationPrior(
        T1_prior_s=T1_prior,
        T2_prior_s=T2_prior,
        epsilon_prior=eps_prior,
        spam_p0_given_1_prior=p01_prior,
        spam_p1_given_0_prior=p10_prior,
        t0_s=t0_s,
    )


# =============================================================================
# 8. AER NOISE MODEL CONSTRUCTION
# =============================================================================

def _delay_seconds(inst, dt_ns: Optional[float]) -> float:
    """Convert one qiskit Delay instruction's duration to seconds, handling
    every unit qiskit's delay() accepts ('s','ms','us','ns','dt')."""
    dur = inst.operation.params[0]
    unit = getattr(inst.operation, "unit", "dt")
    if unit == "s":
        return float(dur)
    if unit == "ms":
        return float(dur) * 1e-3
    if unit == "us":
        return float(dur) * 1e-6
    if unit == "ns":
        return float(dur) * 1e-9
    if unit == "dt":
        return float(dur) * (dt_ns if dt_ns is not None else 0.2222) * 1e-9
    return float(dur)


def build_noise_model(
    truth: DeviceTruth, t_s: float = 0.0,
    native_gate_names: Optional[List[str]] = None,
    circuit: Optional[Any] = None, dt_ns: Optional[float] = None,
):
    """Build a qiskit_aer NoiseModel reflecting truth's parameters evaluated
    at time t_s (regime='ideal' truth is time-independent; regime='nisq'
    truth's drift processes are evaluated at t_s per qubit).

    Holds only the incoherent channels (gate-time relaxation + depolarizing,
    idle relaxation, readout). Coherent mechanisms -- drive-amplitude error,
    crosstalk, detuning -- are injected into the circuit itself by
    simulate_circuit() as explicit unitaries, because a NoiseModel keys errors
    by gate NAME and so cannot make an error follow a gate's rotation sign or
    angle (gpi2(0) and gpi2(0.5) share one name).

    If `circuit` is given, its actual `delay` instructions are scanned per
    qubit and a matching thermal_relaxation_error is attached for each
    distinct delay duration found -- Aer's NoiseModel has no native concept
    of "this gate's error depends on its own numeric duration parameter", so
    (matching the pattern already used elsewhere in this codebase, e.g.
    3_null_nonm.py's _make_backend) a fresh NoiseModel is built per circuit
    rather than shared across circuits with different delays. Without a
    circuit, delay instructions carry no error (only gate-time relaxation
    on the listed gate_names applies) -- callers running their own delay-
    bearing circuits should pass `circuit` to get physically meaningful
    idle-time decay.
    """
    try:
        from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, ReadoutError
    except ImportError as e:
        raise ImportError("qiskit-aer is required for build_noise_model().") from e

    gate_names = native_gate_names or list(_NOISY_GATES)
    nm = NoiseModel()

    for q in range(truth.n_qubits):
        tv = evaluate_truth_at(truth, q, t_s)
        T1, T2_eff = tv["T1_s"], min(tv["T2_s"], 2.0 * tv["T1_s"])
        p01, p10 = tv["spam_p0_given_1"], tv["spam_p1_given_0"]

        gate_error, _ = _gate_noise_channel(truth, q, t_s)
        nm.add_quantum_error(gate_error, gate_names, [q])

        if circuit is not None:
            seen_durations: set = set()
            for inst in circuit.data:
                if inst.operation.name != "delay":
                    continue
                q_idxs = [circuit.find_bit(qb).index for qb in inst.qubits]
                if q not in q_idxs:
                    continue
                ds = _delay_seconds(inst, dt_ns)
                key = round(ds, 15)
                if key in seen_durations or ds <= 1e-12:
                    continue
                seen_durations.add(key)
                nm.add_quantum_error(thermal_relaxation_error(T1, T2_eff, ds), ["delay"], [q])

        p00, p11 = 1.0 - p10, 1.0 - p01
        nm.add_readout_error(ReadoutError([[p00, p10], [p01, p11]]), [q])

    return nm


# Gates that carry gate-time decoherence and stochastic error. "s"/"sdg" are
# EXCLUDED alongside "rz": on both IBM and IonQ hardware Z-type rotations are
# virtual frame updates with no duration and no error (a previous version
# charged s/sdg -- used by the Ramsey-Y probe -- a full gate of decoherence and
# coherent kick). Injected Rz (detuning, crosstalk) and injected amplitude-error
# unitaries are likewise noiseless: they represent physics already accounted
# for, not additional gates. None of 1_inversion.py's probes contain an Rz.
_NOISY_GATES: Tuple[str, ...] = ("sx", "sxdg", "x", "h", "id", "rx", "ry", "r",
                                 "u", "gpi", "gpi2")
_AMPLITUDE_ERROR_GATES = frozenset(g for g in _NOISY_GATES if g != "id")


def _depolarizing_from_eps(eps_value: float):
    """Convert a gate error in 1_inversion.py's epsilon convention into
    Qiskit's depolarizing-channel parameter. forward_gate() models per-gate
    Bloch survival as (1 - 2*epsilon) while depolarizing_error(p, 1) gives
    survival (1 - p), so p = 2*epsilon, NOT epsilon."""
    from qiskit_aer.noise import depolarizing_error
    return depolarizing_error(float(np.clip(2.0 * eps_value, 0.0, 0.75)), 1)


def _amplitude_error_frac(truth: DeviceTruth, qubit: int, t_s: float = 0.0) -> float:
    """Fractional drive-amplitude error f of qubit `qubit` (every native
    rotation G(theta) is executed as G(theta*(1+f))). Zero in the ideal
    regime. Sized so an isolated pi/2 gate carries eps_coh = delta^2/4 =
    coherent_fraction*eps in 1_inversion.py's Bloch convention."""
    if truth.regime != "nisq":
        return 0.0
    cf = float(truth.architecture_specific.get("coherent_fraction", 0.0))
    if cf <= 0.0:
        return 0.0
    eps = evaluate_truth_at(truth, qubit, t_s)["epsilon_gate"]
    return 2.0 * np.sqrt(cf * eps) / (np.pi / 2.0)


def _gate_noise_channel(truth: DeviceTruth, qubit: int, t_s: float = 0.0):
    """The incoherent error attached after every native gate on `qubit`:
    thermal relaxation over the gate duration composed with depolarizing
    error for the stochastic part of the sampled budget,
    eps*(1 - coherent_fraction) (+ opt-in trapped-ion heating term). Returns
    (QuantumError, amplitude_error_frac). Shared by build_noise_model() and
    effective_gate_epsilon(), so the truth is computed from exactly the
    channel the simulator applies."""
    from qiskit_aer.noise import thermal_relaxation_error
    tv = evaluate_truth_at(truth, qubit, t_s)
    T1, T2_eff, eps = tv["T1_s"], min(tv["T2_s"], 2.0 * tv["T1_s"]), tv["epsilon_gate"]
    nisq = truth.regime == "nisq"
    cf = float(truth.architecture_specific.get("coherent_fraction", 0.0)) if nisq else 0.0
    extra = float(truth.architecture_specific.get("heating_extra_gate_error", 0.0)) if nisq else 0.0
    stoch = eps * (1.0 - cf) + extra
    err = thermal_relaxation_error(T1, T2_eff, truth.gate_duration_ns * 1e-9)
    if stoch > 0:
        err = err.compose(_depolarizing_from_eps(stoch))
    return err, _amplitude_error_frac(truth, qubit, t_s)


_FRACTIONAL_POWER_CACHE: Dict[Tuple[Any, ...], Any] = {}


def _fractional_power(op, f: float):
    """G^f as a UnitaryGate: the over-rotation G(theta*f) about G's own axis,
    with the sign of G's rotation (principal branch)."""
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Operator
    key = (op.name, tuple(float(x) for x in op.params), round(f, 15))
    if key not in _FRACTIONAL_POWER_CACHE:
        _FRACTIONAL_POWER_CACHE[key] = UnitaryGate(Operator(op).power(f), label="amp_err")
    return _FRACTIONAL_POWER_CACHE[key]


def _h_amplitude_error(f: float):
    """Amplitude error of a Hadamard executed as hardware does: one pi/2
    pulse between virtual Z frame changes, H = Rz(pi/2)*SX*Rz(pi/2) up to a
    global phase. Only the SX pulse is over-rotated (SX -> SX^f*SX), so the
    executed gate is Rz(pi/2)*SX^f*SX*Rz(pi/2) = E*H with
    E = Rz(pi/2)*SX^f*Rz(-pi/2): a (pi/2)*f rotation, not the pi*f rotation
    H^f. E does not depend on which Z-pi/2-Z decomposition is used."""
    from qiskit.circuit.library import SXGate, UnitaryGate
    from qiskit.quantum_info import Operator
    key = ("h_native_pi2", round(f, 15))
    if key not in _FRACTIONAL_POWER_CACHE:
        rz_p, rz_m = _rotation_unitary("z", np.pi / 2), _rotation_unitary("z", -np.pi / 2)
        e_mat = rz_p @ Operator(SXGate()).power(f).data @ rz_m
        _FRACTIONAL_POWER_CACHE[key] = UnitaryGate(e_mat, label="amp_err")
    return _FRACTIONAL_POWER_CACHE[key]


def _inject_amplitude_errors(circuit, truth: DeviceTruth, t_s: float = 0.0):
    """Return a copy of `circuit` with G^f appended after every native
    single-qubit rotation G, f = _amplitude_error_frac(). Because the error
    follows the sign of the rotation (as a miscalibrated drive amplitude
    does), G(theta)*G(-theta) cancels exactly, as the gate-repetition probe's
    G*G^-1 design assumes; errors between non-inverse gates still accumulate.
    "h" is not a native pulse: it receives _h_amplitude_error(f), the error of
    its single pi/2 pulse, instead of H^f. ("x" keeps X^f: X^f*X equals the
    two-sx execution (SX^f*SX)^2 up to a global phase.)
    No-op when every qubit has f = 0 (always, in the ideal regime)."""
    fracs = {q: _amplitude_error_frac(truth, q, t_s) for q in range(truth.n_qubits)}
    if not any(f > 0.0 for f in fracs.values()):
        return circuit
    new_qc = circuit.copy_empty_like()
    for inst in circuit.data:
        new_qc.append(inst.operation, inst.qubits, inst.clbits)
        if inst.operation.name not in _AMPLITUDE_ERROR_GATES or len(inst.qubits) != 1:
            continue
        q_idx = circuit.find_bit(inst.qubits[0]).index
        f = fracs.get(q_idx, 0.0)
        if f > 0.0:
            err_gate = (_h_amplitude_error(f) if inst.operation.name == "h"
                        else _fractional_power(inst.operation, f))
            new_qc.append(err_gate, inst.qubits, [])
    return new_qc


def _native_inverse_pair(architecture: str):
    """The native (G, G^-1) pair 1_inversion.py's gate-repetition probe uses."""
    if architecture == "trapped_ion":
        try:
            from qiskit_ionq.ionq_gates import GPI2Gate
            return GPI2Gate(0.0), GPI2Gate(0.5)
        except ImportError:
            from qiskit.circuit.library import RGate
            return RGate(np.pi / 2, 0.0), RGate(np.pi / 2, np.pi)
    from qiskit.circuit.library import SXGate, SXdgGate
    return SXGate(), SXdgGate()


def effective_gate_epsilon(truth: DeviceTruth, qubit: int = 0, t_s: float = 0.0) -> float:
    """Per-gate error of qubit `qubit` in 1_inversion.py's epsilon convention,
    computed exactly from the channel the simulator applies.

    The native pair G, noise, G^f, G^-1, noise, (G^-1)^f is composed as a
    superoperator; its Pauli-transfer-matrix element R_zz is the per-pair
    survival of the Z Bloch component, and (1 - 2*eps)^2 = R_zz defines eps.
    R_zz is exactly the rate parameter of the affine gate-repetition model
    the engine fits (1_inversion.forward_gate_affine, z -> R_zz*z + R_z0);
    the non-unital offset R_z0 (amplitude damping toward |0> during the
    gates) enters through that model's asymptote z*, which the engine
    computes from its T1/T2 estimates and the gate time (same channel
    form: relaxation over t_g then depolarizing). This is the
    quantity the gate-repetition probe measures: it INCLUDES decoherence
    during the gate (the coherence-limited error of McKay et al.,
    arXiv:2311.05933, Eq. B2) and any coherent residual that survives the
    pair. It is not identical to a randomized-benchmarking average gate
    infidelity, which differs from it by (t_g/12)*(1/T2 - 1/T1) (to first
    order in t_g/T1, t_g/T2) plus any coherent residual. DeviceTruth.epsilon_gate is only the sampled
    depolarizing budget; comparing a recovered epsilon against it omits the
    gate-time T1/T2 contribution (~1e-4 for a 50 ns gate on a ~100 us
    transmon) and biases every epsilon benchmark high.

    Detuning (static dw, drift and the quasi-static offset) is EXCLUDED by
    construction: it is a coherent frame error, not the incoherent per-gate
    error the engine's epsilon claims to be. With
    generate_device(detuning_during_gates=True) the simulated gate-repetition
    probe therefore also carries a detuning-induced rotation that this truth
    does not contain, and any resulting epsilon bias is scored as error."""
    from qiskit.quantum_info import Operator, PTM, SuperOp
    err, f = _gate_noise_channel(truth, qubit, t_s)
    noise = SuperOp(err)

    def _step(op):
        ch = SuperOp(Operator(op)).compose(noise)
        if f > 0.0:
            ch = ch.compose(SuperOp(Operator(op).power(f)))
        return ch

    g, g_inv = _native_inverse_pair(truth.architecture)
    r_zz = float(np.real(PTM(_step(g).compose(_step(g_inv))).data[3, 3]))
    return float(0.5 * (1.0 - np.sqrt(max(r_zz, 0.0))))


def _rotation_unitary(axis: str, angle_rad: float) -> np.ndarray:
    """U = exp(-i*angle*Sigma/2) for Sigma in {X,Y,Z}."""
    c, s = np.cos(angle_rad / 2.0), np.sin(angle_rad / 2.0)
    if axis == "x":
        return np.array([[c, -1j * s], [-1j * s, c]])
    if axis == "y":
        return np.array([[c, -s], [s, c]])
    if axis == "z":
        return np.array([[np.exp(-1j * angle_rad / 2.0), 0], [0, np.exp(1j * angle_rad / 2.0)]])
    raise ValueError(f"axis must be 'x','y','z', got {axis!r}")


# =============================================================================
# 9. CIRCUIT-AWARE SIMULATION (timing + simultaneous-operation crosstalk)
# =============================================================================

def _circuit_layers(circuit) -> List[List[Tuple[int, Tuple[int, ...]]]]:
    """Greedy list-scheduling into simultaneous layers: instruction i's layer
    = 1 + max(current layer of any qubit it touches); every qubit it touches
    is then advanced to that layer. Returns, per layer, a list of
    (instruction_index_in_circuit.data, qubit_indices) tuples. This is what
    makes crosstalk injection circuit-aware rather than a blanket per-qubit
    channel: a gate only triggers a crosstalk kick on its neighbor if they
    are found in the same layer."""
    qubit_layer: Dict[int, int] = {}
    layers: Dict[int, List[Tuple[int, Tuple[int, ...]]]] = {}
    for idx, inst in enumerate(circuit.data):
        name = inst.operation.name
        if getattr(inst.operation, "label", None) == _DETUNE_GATE_LABEL:
            # Detuning phase accumulated DURING an adjacent pulse: part of
            # that pulse's time window, not an operation of its own.
            continue
        q_idxs = tuple(circuit.find_bit(q).index for q in inst.qubits)
        if name == "barrier":
            # A barrier is a synchronization point, not a gap: every qubit it
            # touches is brought up to the same layer, so nothing before it
            # can appear simultaneous with anything after it. It carries no
            # error itself and is never added to `layers`.
            sync_layer = max((qubit_layer.get(q, 0) for q in q_idxs), default=0)
            for q in q_idxs:
                qubit_layer[q] = sync_layer
            continue
        layer = 1 + max((qubit_layer.get(q, 0) for q in q_idxs), default=0)
        for q in q_idxs:
            qubit_layer[q] = layer
        if name in ("measure", "delay"):
            continue  # advances timing but carries no crosstalk-relevant operation
        layers.setdefault(layer, []).append((idx, q_idxs))
    return [layers[k] for k in sorted(layers)]


def _inject_crosstalk_kicks(circuit, truth: DeviceTruth):
    """Return a copy of `circuit` with an explicit coherent-rotation
    instruction appended immediately after every layer in which two members
    of a crosstalk-affected pair both have a gate -- i.e. crosstalk only
    fires when operations genuinely overlap in the same circuit layer,
    per Rudinger et al.'s circuit-aware crosstalk model E_i = E_i^(0) o
    E_ij^XT. No-op (returns an unmodified copy) if truth has no enabled
    crosstalk mechanism or fewer than 2 qubits are touched."""
    active = [m for m in truth.crosstalk if m.enabled and m.angle_rad != 0.0]
    if not active or circuit.num_qubits < 2:
        return circuit.copy()

    new_qc = circuit.copy()
    layers = _circuit_layers(circuit)

    insertions: List[Tuple[int, int, float, str]] = []  # (after_instruction_idx, qubit, angle, axis)
    for layer in layers:
        touched_qubits = {q for _, qs in layer for q in qs}
        # Insert after the LAST instruction of the whole layer, not after
        # either neighbor's own specific gate -- a kick placed mid-layer
        # would sit ahead of a layer-mate's gate in program order, which
        # misrepresents crosstalk as happening before the shared time
        # window it is supposed to follow has actually completed.
        layer_end_idx = max(idx for idx, _ in layer)
        for mech in active:
            for a, b in mech.affected_pairs:
                if a in touched_qubits and b in touched_qubits:
                    # Each unordered pair is listed once in affected_pairs
                    # (never symmetrized into both (a,b) and (b,a)), so both
                    # members are kicked exactly once per simultaneous layer.
                    insertions.append((layer_end_idx, a, mech.angle_rad, mech.axis))
                    insertions.append((layer_end_idx, b, mech.angle_rad, mech.axis))

    if not insertions:
        return new_qc
    from qiskit.circuit.library import UnitaryGate
    # Kicks are UnitaryGates, not RX/RY/RZ: "rx"/"ry" are native gate names that
    # carry gate noise, and a crosstalk kick is a coherent error, not a gate.
    # Insert in reverse index order so earlier insertions don't shift later indices.
    for after_idx, qubit, angle, axis in sorted(insertions, key=lambda t: -t[0]):
        new_qc.data.insert(after_idx + 1, new_qc.data[after_idx].replace(
            operation=UnitaryGate(_rotation_unitary(axis, angle), label="xt_kick"),
            qubits=(new_qc.qubits[qubit],), clbits=()))
    return new_qc


def _inject_detuning_phases(circuit, truth: DeviceTruth, t_s: float = 0.0,
                            dt_ns: Optional[float] = None,
                            during_gates: bool = False):
    """_inject_detuning() without per-shot parameters: returns only the
    circuit (the historical interface; during_gates=False is the historical
    behaviour exactly)."""
    return _inject_detuning(circuit, truth, t_s=t_s, dt_ns=dt_ns,
                            during_gates=during_gates)[0]


def _phase_reaches_measurement(circuit, q_idx: int) -> bool:
    """True iff qubit q_idx has a free-evolution delay followed later by a
    physical pulse, so that a Z phase accumulated in the delay is mapped onto
    the measured Z basis (Ramsey, echo). False for a T1-type circuit (pulse,
    delay, measure): its initial |0> is invariant under Rz and thermal
    relaxation is phase covariant, so every detuning phase commutes to the
    Z-basis measurement and has no effect. False for delay-free gate trains
    (see simulate_circuit for that limitation)."""
    seen_delay = False
    for inst in circuit.data:
        if q_idx not in [circuit.find_bit(qb).index for qb in inst.qubits]:
            continue
        name = inst.operation.name
        if name == "delay" and float(inst.operation.params[0]) > 0:
            seen_delay = True
        elif seen_delay and name in _NOISY_GATES:
            return True
    return False


_DETUNE_GATE_LABEL = "detune_gate"


def _inject_detuning(circuit, truth: DeviceTruth, t_s: float = 0.0,
                     dt_ns: Optional[float] = None, during_gates: bool = False,
                     parametrize_qubits: Any = ()):
    """Return (circuit_copy, plan). The copy carries an explicit Rz(delta_omega * delta_t)
    inserted immediately after every delay instruction, implementing the
    coherent detuning evolution

        U_delta(t) = exp(-i * delta_omega * t * Z / 2)

    as ACTUAL phase evolution on the simulated circuit, rather than leaving
    delta_omega as an unused DeviceTruth field. Applied in BOTH regimes:
    detuning is a fundamental device parameter, not a NISQ-only mechanism.

    Applied uniformly after every delay, which reproduces the correct probe
    physics without special-casing circuit types:
      - T1 (x, delay, measure): a Z-rotation cannot change a Z-basis
        measurement of a state along +/-Z, so T1 is correctly unaffected.
      - Ramsey X/Y: the phase accumulates and is measured -- this is the
        probe delta_omega is actually recovered from.
      - Hahn echo (h, delay/2, x, delay/2, h): the mid-sequence X pulse
        refocuses static detuning, since Rz(phi) X Rz(phi) = X. The echo is
        therefore correctly insensitive to static delta_omega, matching
        1_inversion.py's forward_echo(), which carries no detuning term.

    Mirrors the reference implementation removed from 1_inversion.py by
    commit b1a8385 (_inject_ramsey_detuning), generalized to apply each
    qubit's own delta_omega to the delays acting on that qubit rather than
    hardcoding qubit 0.

    during_gates=True (NISQ, generate_device(detuning_during_gates=True))
    also lets the detuning act while each native single-qubit pulse
    (_NOISY_GATES) is played, as Rz(dw*t_g/2) G Rz(dw*t_g/2) with t_g =
    truth.gate_duration_ns: the symmetric split of a pulse played on a
    detuned qubit, exact to first order in dw*t_g. These Rz carry the label
    _DETUNE_GATE_LABEL and no gate noise (rz is virtual here). Virtual gates
    (rz, s, sdg) take no time and get none. The echo stays refocused
    (Rz(a) X Rz(a) = X), and so does any static phase around a single pulse,
    but the phases between the two pulses of a G*G^-1 pair do not cancel.

    parametrize_qubits: for these qubits every inserted Rz angle is a
    qiskit Parameter instead of a number, and `plan` lists (Parameter,
    qubit, duration_s); simulate_circuit binds each to (dw_q + delta)*
    duration_s with a per-shot quasi-static offset delta. For all other
    qubits plan is empty and angles are dw_q * duration_s, as before.
    """
    has_delay = any(inst.operation.name == "delay" for inst in circuit.data)
    if not has_delay and not during_gates:
        return circuit, []
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import RZGate

    t_g_s = float(truth.gate_duration_ns) * 1e-9
    param_q = set(int(q) for q in parametrize_qubits)
    plan: List[Tuple[Any, int, float]] = []
    gate_params: Dict[int, Any] = {}
    dw_cache: Dict[int, float] = {}

    def _dw(q: int) -> float:
        if q not in dw_cache:
            dw_cache[q] = evaluate_truth_at(truth, q, t_s)["delta_omega_rad_s"]
        return dw_cache[q]

    def _gate_angle(q: int):
        if q in param_q:
            if q not in gate_params:
                gate_params[q] = Parameter(f"detune_gate_q{q}")
                plan.append((gate_params[q], q, 0.5 * t_g_s))
            return gate_params[q]
        return _dw(q) * 0.5 * t_g_s

    new_qc = circuit.copy_empty_like()
    for inst in circuit.data:
        name = inst.operation.name
        gate_q = None
        if during_gates and t_g_s > 0.0 and name in _NOISY_GATES and len(inst.qubits) == 1:
            q = circuit.find_bit(inst.qubits[0]).index
            if q < truth.n_qubits and (q in param_q or _dw(q) != 0.0):
                gate_q = q
        if gate_q is not None:
            new_qc.append(RZGate(_gate_angle(gate_q), label=_DETUNE_GATE_LABEL),
                          [new_qc.qubits[gate_q]], [])
        new_qc.append(inst.operation, inst.qubits, inst.clbits)
        if gate_q is not None:
            new_qc.append(RZGate(_gate_angle(gate_q), label=_DETUNE_GATE_LABEL),
                          [new_qc.qubits[gate_q]], [])
        if name != "delay":
            continue
        ds = _delay_seconds(inst, dt_ns)
        if ds <= 0.0:
            continue
        for qb in inst.qubits:
            q_idx = circuit.find_bit(qb).index
            if q_idx >= truth.n_qubits:
                continue
            if q_idx in param_q:
                p = Parameter(f"detune_delay_q{q_idx}_{len(plan)}")
                plan.append((p, q_idx, ds))
                new_qc.rz(p, new_qc.qubits[q_idx])
                continue
            phase = _dw(q_idx) * ds
            if phase != 0.0:
                new_qc.rz(phase, new_qc.qubits[q_idx])
    return new_qc, plan


_AER_NATIVE = frozenset({"sx", "sxdg", "x", "h", "id", "rx", "ry", "rz", "r", "u",
                         "s", "sdg", "unitary", "delay", "measure", "barrier"})


def _to_aer_native(circuit):
    """Replace IonQ GPI(phi)/GPI2(phi) with Qiskit's R gate, which is the same
    operator: GPI2(phi) = R(pi/2, 2*pi*phi) exactly and GPI(phi) =
    i*R(pi, 2*pi*phi) (a global phase). "r" is a noisy native gate name, so
    the noise model still applies, and Aer runs it without transpilation."""
    if not any(inst.operation.name in ("gpi", "gpi2") for inst in circuit.data):
        return circuit
    from qiskit.circuit.library import RGate
    new_qc = circuit.copy_empty_like()
    for inst in circuit.data:
        op = inst.operation
        if op.name == "gpi2":
            op = RGate(np.pi / 2, 2.0 * np.pi * float(op.params[0]))
        elif op.name == "gpi":
            op = RGate(np.pi, 2.0 * np.pi * float(op.params[0]))
        new_qc.append(op, inst.qubits, inst.clbits)
    return new_qc


def simulate_circuit(
    circuit, truth: DeviceTruth, shots: int = 1024, t_s: float = 0.0,
    native_gate_names: Optional[List[str]] = None, seed: Optional[int] = None,
    dt_ns: Optional[float] = None,
) -> Dict[str, int]:
    """Simulate one Qiskit QuantumCircuit under `truth` evaluated at time t_s,
    including coherent detuning evolution and circuit-aware coherent
    crosstalk. Returns raw Aer counts.

    `dt_ns` is required to convert delay durations expressed in hardware
    "dt" units back to seconds (1_inversion.py's _snap() emits dt-unit
    delays whenever a profile carries a non-None dt_ns, i.e. for
    superconducting). Pass profile.dt_ns; leaving it None falls back to
    0.2222 ns, which is only coincidentally correct for superconducting.

    NISQ options read from truth.architecture_specific (both absent, hence
    no change, unless generate_device() was asked for them):
      detuning_during_gates: detuning also acts during pulses
        (see _inject_detuning).
      quasi_static_dw_std_rad_s: every SHOT gets its own static offset
        delta_q ~ N(0, sigma_q) added to dw_q, sampled exactly (one Aer
        experiment per shot via parameter binding), in circuits where a
        delay is followed by a pulse on that qubit (_phase_reaches_
        measurement: Ramsey, echo). In T1-type circuits the offset provably
        has no effect and is not sampled. LIMITATION: in delay-free gate
        trains (the gate-repetition probe) only the static dw acts during
        the pulses; per-shot sampling there costs ~60 ms per shot and a
        fixed quadrature misrepresents the long-time average, so the
        offset's dephasing of the detuning-induced rotation is omitted.
    """
    try:
        from qiskit_aer import AerSimulator
        from qiskit import transpile
    except ImportError as e:
        raise ImportError("qiskit-aer is required for simulate_circuit().") from e

    nisq = truth.regime == "nisq"
    during_gates = nisq and bool(truth.architecture_specific.get("detuning_during_gates", False))
    base = _to_aer_native(circuit)
    qs_std = truth.architecture_specific.get("quasi_static_dw_std_rad_s") if nisq else None
    qs_qubits: List[int] = []
    if qs_std is not None:
        qs_std = np.asarray(qs_std, dtype=float)
        qs_qubits = [q for q in range(min(base.num_qubits, truth.n_qubits))
                     if qs_std[q] > 0.0 and _phase_reaches_measurement(base, q)]
    qc, plan = _inject_detuning(base, truth, t_s=t_s, dt_ns=dt_ns,
                                during_gates=during_gates, parametrize_qubits=qs_qubits)
    if nisq:
        qc = _inject_crosstalk_kicks(qc, truth)
        qc = _inject_amplitude_errors(qc, truth, t_s=t_s)
    nm = build_noise_model(truth, t_s=t_s, native_gate_names=native_gate_names,
                           circuit=qc, dt_ns=dt_ns)
    sim_seed = seed if seed is not None else truth.seed
    sim = AerSimulator(noise_model=nm, seed_simulator=sim_seed)
    # Run without transpiling whenever every instruction is Aer-native, so
    # injected coherent errors (amplitude error, crosstalk kicks) reach Aer
    # as noiseless "unitary" instructions. Transpiling against the noise
    # model's basis decomposed them into "u" gates, which carry gate noise,
    # charging every coherent kick a full extra gate of relaxation and
    # depolarizing error. IonQ gpi/gpi2 were already mapped onto the exactly
    # equivalent "r" gate by _to_aer_native().
    if all(inst.operation.name in _AER_NATIVE for inst in qc.data):
        tqc = qc
    else:
        tqc = transpile(qc, sim, optimization_level=0)
    if not plan:
        result = sim.run(tqc, shots=shots).result()
        return result.get_counts(0)
    if int(shots) <= 0:
        return {}
    # Quasi-static offsets: one independent draw per shot and qubit, from a
    # stream salted by this run's seed; each shot is one bound experiment.
    rng_qs = _sub_rng(sim_seed, "quasi_static_delta")
    delta = {q: rng_qs.normal(0.0, float(qs_std[q]), size=int(shots)) for q in qs_qubits}
    dw_q = {q: evaluate_truth_at(truth, q, t_s)["delta_omega_rad_s"] for q in qs_qubits}
    binds = {p: ((dw_q[q] + delta[q]) * dur).tolist() for p, q, dur in plan}
    result = sim.run(tqc, parameter_binds=[binds], shots=1).result()
    counts: Dict[str, int] = {}
    for i in range(int(shots)):
        for bitstring, cnt in result.get_counts(i).items():
            counts[bitstring] = counts.get(bitstring, 0) + cnt
    return counts


# Time-resolved probe execution (simulate_probe(time_resolved=True)).
# Per-shot repetition overhead: 0.0 s is a documented LOWER BOUND, not a
# measurement. IBM's Qiskit Runtime documentation gives no single value
# ("Default is given by backend.default_rep_delay"; "Each backend has a
# different default rep_delay"), and no per-shot overhead was verified for
# IonQ systems. Readout duration is not counted either, so every run
# duration computed below is a lower bound.
_DEFAULT_REP_DELAY_S = 0.0
# Each shot chunk spans at most this fraction of the shortest correlation
# time of the truth's drift processes, so parameters are near-constant within
# a chunk (a discretisation choice, not a physical parameter).
_CHUNK_FRACTION_OF_CORRELATION_TIME = 0.1
# Runtime cap on chunks per circuit (each chunk is one Aer run); when it
# binds, chunks span more than the fraction above.
_DEFAULT_MAX_CHUNKS_PER_CIRCUIT = 100


def _shortest_drift_correlation_time_s(truth: DeviceTruth) -> float:
    """Shortest correlation time (s) of truth's attached drift processes, as
    evaluate_truth_at() realises them; inf when there are none (ideal). OU:
    1/theta. lowpass_filtered: y[i] = a*y[i-1] + ..., a = exp(-2*pi*cutoff*dt),
    decorrelates as exp(-2*pi*cutoff*t), i.e. 1/(2*pi*cutoff). one_over_f:
    its fastest Lorentzian has its corner at the Nyquist frequency of the
    evaluation grid dt, i.e. dt/pi. telegraph (rate r per direction):
    1/(2r)."""
    taus: List[float] = []
    for proc in truth.drift_processes.values():
        p, pt = proc.params, proc.process_type
        if pt == "ou":
            theta = p.get("theta", 1.0)
            if theta > 0:
                taus.append(1.0 / theta)
        elif pt == "lowpass_filtered":
            taus.append(1.0 / (2.0 * np.pi * max(p.get("cutoff_hz", 1e-3), 1e-9)))
        elif pt == "one_over_f":
            dt_grid = max(1.0, (1.0 / max(p.get("cutoff_hz", 1e-3), 1e-9)) / 20.0)
            taus.append(dt_grid / np.pi)
        elif pt == "telegraph":
            rate = p.get("rate", 1e-3)
            if rate > 0:
                taus.append(1.0 / (2.0 * rate))
        else:
            raise ValueError(f"no correlation time defined for process_type {pt!r}")
    return min(taus) if taus else float("inf")


def _probe_schedule(circuits: List[Any], truth: DeviceTruth, shots_list: List[int],
                    t_s: float, dt_ns: Optional[float], rep_delay_s: float,
                    max_chunks: int) -> List[Dict[str, Any]]:
    """Wall-clock schedule of a probe run: circuits execute one after another
    from t_s in list order (circuit-major, all shots of circuit i before
    circuit i+1 -- an assumption; a provider may interleave). One shot of a
    circuit lasts the sum of its delay durations (exact for single-qubit
    probes, an upper bound if delays on different qubits run in parallel)
    plus truth.gate_duration_ns per noisy native gate (the gates the noise
    model charges gate-time relaxation) plus rep_delay_s. Each circuit's
    shots are split into K near-equal consecutive chunks, K = ceil(duration /
    (_CHUNK_FRACTION_OF_CORRELATION_TIME * shortest correlation time)),
    capped by max_chunks and by the shot count."""
    max_span = _CHUNK_FRACTION_OF_CORRELATION_TIME * _shortest_drift_correlation_time_s(truth)
    t_now = float(t_s)
    schedule: List[Dict[str, Any]] = []
    for qc, sh in zip(circuits, shots_list):
        qn = _to_aer_native(qc)
        idle_s = sum(_delay_seconds(inst, dt_ns) for inst in qn.data
                     if inst.operation.name == "delay")
        n_gates = sum(1 for inst in qn.data if inst.operation.name in _NOISY_GATES)
        per_shot_s = idle_s + n_gates * truth.gate_duration_ns * 1e-9 + rep_delay_s
        duration_s = per_shot_s * int(sh)
        k_need = (int(np.ceil(duration_s / max_span))
                  if np.isfinite(max_span) and max_span > 0 else 1)
        n_chunks = max(1, min(k_need, int(max_chunks), int(sh)))
        base, rem = divmod(int(sh), n_chunks)
        chunks, done = [], 0
        for k in range(n_chunks):
            n_k = base + (1 if k < rem else 0)
            c0, c1 = t_now + done * per_shot_s, t_now + (done + n_k) * per_shot_s
            chunks.append({"t_start_s": c0, "t_end_s": c1, "t_mid_s": 0.5 * (c0 + c1),
                           "shots": n_k})
            done += n_k
        schedule.append({"t_start_s": t_now, "t_end_s": t_now + duration_s,
                         "chunks": chunks})
        t_now += duration_s
    return schedule


def probe_run_duration_s(circuits: List[Any], truth: DeviceTruth, shots_list: List[int],
                         t_s: float = 0.0, dt_ns: Optional[float] = None,
                         rep_delay_s: float = _DEFAULT_REP_DELAY_S) -> float:
    """Wall-clock length (s) of running `circuits` one after another, with
    the same accounting as simulate_probe(time_resolved=True) (_probe_
    schedule: delays + noisy gates * gate duration + rep_delay_s per shot).
    A LOWER BOUND (rep_delay_s = 0 and readout time is not counted).
    Available in the frozen mode too, so a benchmark can always report how
    long its probe would occupy the device."""
    schedule = _probe_schedule(circuits, truth, list(shots_list), t_s, dt_ns,
                               rep_delay_s, 1)
    return float(schedule[-1]["t_end_s"] - float(t_s)) if schedule else 0.0


def _truth_snapshot(truth: DeviceTruth, t_s: float):
    """(snapshot, per-qubit truth): a drift-free copy of truth whose stored
    T1/T2/delta_omega equal evaluate_truth_at(truth, q, t_s) for every qubit.
    evaluate_truth_at() returns identical values for the snapshot at any
    time, so simulating it equals simulating truth at t_s, while each drift
    trajectory is materialised once instead of at every internal call."""
    from dataclasses import replace as _dc_replace
    tvs = [evaluate_truth_at(truth, q, t_s) for q in range(truth.n_qubits)]
    snap = _dc_replace(
        truth, drift_processes={},
        T1_s=np.array([tv["T1_s"] for tv in tvs]),
        T2_s=np.array([tv["T2_s"] for tv in tvs]),
        delta_omega_rad_s=np.array([tv["delta_omega_rad_s"] for tv in tvs]))
    return snap, tvs


def simulate_probe(
    circuits: List[Any], truth: DeviceTruth, shots: Any = 1024, t_s: float = 0.0,
    native_gate_names: Optional[List[str]] = None, seed: Optional[int] = None,
    dt_ns: Optional[float] = None, time_resolved: bool = False,
    rep_delay_s: float = _DEFAULT_REP_DELAY_S,
    max_chunks_per_circuit: int = _DEFAULT_MAX_CHUNKS_PER_CIRCUIT,
) -> SimulationResult:
    """Simulate a list of probe circuits (e.g. from 1_inversion.py's
    build_probe_circuits()) and return counts directly compatible with
    lindblad_inversion(counts_list, meta, profile, ...).

    `shots` may be a single int (applied to every circuit) or a list of
    per-circuit shot counts matching `circuits`' length.

    `dt_ns` should be the originating profile's dt_ns, so delays emitted in
    hardware "dt" units are converted to seconds correctly for both the
    thermal-relaxation channel and the injected detuning phase.

    Circuit i is simulated with Aer seed `seed + i` when `seed` is given;
    when `seed` is None it gets a distinct deterministic seed derived from
    (truth.seed, i), so circuits never share one Aer RNG stream.

    time_resolved=False (default): every circuit and shot sees the truth
    frozen at t_s. time_resolved=True: the run starts at t_s and follows
    _probe_schedule() (circuit-major order; shot duration = delays + noisy
    gates * gate duration + rep_delay_s, default 0.0 = lower bound). Each
    circuit's shots are split into chunks spanning at most 1/10 of the
    shortest drift correlation time (at most max_chunks_per_circuit chunks);
    chunk k is simulated with the truth at its mid-time and Aer seed
    circ_seed for k = 0, else one derived from (circ_seed, k), and the chunk
    counts are summed. This realises within-run drift and shot-to-shot
    quasi-static noise from the drift processes already attached to truth
    (no new parameter); with no drift (ideal) or K = 1 for every circuit
    and no change of truth, it reproduces the frozen counts. The schedule
    and per-chunk truth are returned in the SimulationResult.
    """
    n = len(circuits)
    shots_list = [shots] * n if isinstance(shots, int) else list(shots)
    if len(shots_list) != n:
        raise ValueError(f"shots list length {len(shots_list)} != number of circuits {n}")
    if time_resolved:
        if rep_delay_s < 0.0 or int(max_chunks_per_circuit) < 1:
            raise ValueError("rep_delay_s must be >= 0 and max_chunks_per_circuit >= 1")
        schedule = _probe_schedule(circuits, truth, shots_list, t_s, dt_ns,
                                   rep_delay_s, max_chunks_per_circuit)
        chunk_records: List[List[Dict[str, Any]]] = []

    counts_list: List[Dict[str, int]] = []
    names: List[str] = []
    for i, (qc, sh) in enumerate(zip(circuits, shots_list)):
        # seed=None: per-circuit seed from a salted sub-stream of truth.seed
        # (never the bare truth.seed for every circuit); upper bound 2**31 - 1
        # keeps it a non-negative 32-bit int for Aer's seed_simulator.
        circ_seed = ((seed + i) if seed is not None
                     else int(_sub_rng(truth.seed, "aer_circuit", i).integers(0, 2**31 - 1)))
        if not time_resolved:
            raw = simulate_circuit(qc, truth, shots=sh, t_s=t_s,
                                   native_gate_names=native_gate_names,
                                   seed=circ_seed,
                                   dt_ns=dt_ns)
        else:
            raw, recs = {}, []
            for k, chunk in enumerate(schedule[i]["chunks"]):
                ch_seed = (circ_seed if k == 0 else
                           int(_sub_rng(circ_seed, "aer_chunk", k).integers(0, 2**31 - 1)))
                snap, tvs = _truth_snapshot(truth, chunk["t_mid_s"])
                part = simulate_circuit(qc, snap, shots=chunk["shots"], t_s=chunk["t_mid_s"],
                                        native_gate_names=native_gate_names,
                                        seed=ch_seed, dt_ns=dt_ns)
                for bitstring, cnt in part.items():
                    raw[bitstring] = raw.get(bitstring, 0) + cnt
                recs.append(dict(chunk, seed=ch_seed, truth=tvs))
            chunk_records.append(recs)
        norm: Dict[str, int] = {}
        for bitstring, cnt in raw.items():
            b = bitstring.replace(" ", "")[-1]
            norm[b] = norm.get(b, 0) + cnt
        counts_list.append(norm)
        names.append(getattr(qc, "name", f"circuit_{i}"))

    manifest_id = str(uuid.uuid5(uuid.NAMESPACE_OID,
                                 f"{truth.seed}:{truth.architecture}:{truth.n_qubits}:{truth.regime}:{t_s}"))
    if time_resolved:
        return SimulationResult(
            counts_list=counts_list, shots_list=shots_list, circuit_names=names,
            manifest_id=manifest_id, truth_evaluation_time_s=t_s, time_resolved=True,
            circuit_start_times_s=[s["t_start_s"] for s in schedule],
            circuit_end_times_s=[s["t_end_s"] for s in schedule],
            run_duration_s=(schedule[-1]["t_end_s"] - float(t_s)) if schedule else 0.0,
            chunk_records=chunk_records)
    return SimulationResult(counts_list=counts_list, shots_list=shots_list,
                            circuit_names=names, manifest_id=manifest_id,
                            truth_evaluation_time_s=t_s)


# =============================================================================
# 10. MANIFEST
# =============================================================================

def make_manifest(
    truth: DeviceTruth, prior: CalibrationPrior, shots: int,
    simulation_config: Optional[Dict[str, Any]] = None,
) -> SimulationManifest:
    """Full reproducibility record. truth and calibration are recorded under
    separate top-level keys. Calibration T1/T2/epsilon are perturbed from
    truth; the calibration SPAM fields are the prior's (verbatim truth,
    i.e. oracle SPAM, unless spam_perturbation_frac > 0; see
    generate_calibration_prior's docstring)."""
    return SimulationManifest(
        instance_id=str(uuid.uuid4()), seed=truth.seed, architecture=truth.architecture,
        regime=truth.regime, n_qubits=truth.n_qubits,
        truth={
            "T1_s": truth.T1_s.tolist(), "T2_s": truth.T2_s.tolist(),
            "delta_omega_rad_s": truth.delta_omega_rad_s.tolist(),
            "epsilon_gate": truth.epsilon_gate.tolist(),
            "spam_p0_given_1": truth.spam_p0_given_1.tolist(),
            "spam_p1_given_0": truth.spam_p1_given_0.tolist(),
            "architecture_specific": truth.architecture_specific,
        },
        calibration={
            "T1_prior_s": prior.T1_prior_s, "T2_prior_s": prior.T2_prior_s,
            "epsilon_prior": prior.epsilon_prior,
            "spam_p0_given_1_prior": prior.spam_p0_given_1_prior,
            "spam_p1_given_0_prior": prior.spam_p1_given_0_prior,
            "t0_s": prior.t0_s, "source": prior.source,
        },
        simulation_config=simulation_config or {},
        drift={k: {"process_type": v.process_type, "params": v.params, "seed_salt": v.seed_salt}
               for k, v in truth.drift_processes.items()},
        crosstalk=[{"enabled": m.enabled, "angle_rad": m.angle_rad, "axis": m.axis,
                   "affected_pairs": m.affected_pairs} for m in truth.crosstalk],
        gate_duration_ns=truth.gate_duration_ns, shots=shots,
        provenance_keys_used=sorted(MODEL_PROVENANCE.keys()),
    )


__all__ = [
    "ARCHITECTURES",
    "MODEL_PROVENANCE", "get_parameter_provenance", "validate_parameter_provenance",
    "ArchitectureProfile", "DriftProcess", "CrosstalkMechanism", "DeviceTruth",
    "CalibrationPrior", "SimulationResult", "SimulationManifest",
    "get_architecture_profile", "generate_device", "evaluate_truth_at",
    "generate_calibration_prior", "effective_gate_epsilon",
    "build_noise_model",
    "simulate_circuit", "simulate_probe", "probe_run_duration_s", "make_manifest",
    "sample_trajectory", "evaluate_at", "evaluate_deviation_at",
]


# =============================================================================
# 11. VALIDATION SELF-TEST
# =============================================================================

def _run_validation_tests() -> None:
    import copy
    import warnings
    import logging
    warnings.filterwarnings("ignore")
    logging.getLogger("qiskit_aer.noise.noise_model").setLevel(logging.ERROR)
    failures: List[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        if cond:
            print(f"[PASS] {name}")
        else:
            print(f"[FAIL] {name} {detail}")
            failures.append(f"{name} {detail}")

    print("=" * 70)
    print("1c_simulation_models.py validation suite")
    print("=" * 70)

    # 10. Provenance completeness
    issues = validate_parameter_provenance()
    check("10. provenance completeness", not issues, str(issues))

    # 9. Physical parameter validation (rejects invalid inputs)
    def _raises(fn: Callable[[], Any]) -> bool:
        try:
            fn()
            return False
        except ValueError:
            return True
    check("9a. rejects negative T1", _raises(lambda: ArchitectureProfile(
        "superconducting", T1_s=-1.0, T2_s=1.0, gate_duration_ns=50,
        spam_p0_given_1=0.01, spam_p1_given_0=0.001, gate_error_total=1e-3)))
    check("9b. rejects T2>2T1", _raises(lambda: ArchitectureProfile(
        "superconducting", T1_s=100e-6, T2_s=300e-6, gate_duration_ns=50,
        spam_p0_given_1=0.01, spam_p1_given_0=0.001, gate_error_total=1e-3)))
    check("9c. rejects invalid architecture", _raises(lambda: generate_device("bogus", 1, 42)))
    check("9d. rejects n_qubits=0", _raises(lambda: generate_device("superconducting", 0, 42)))
    check("9e. rejects negative gate duration", _raises(lambda: ArchitectureProfile(
        "superconducting", T1_s=1e-4, T2_s=1e-4, gate_duration_ns=-1,
        spam_p0_given_1=0.01, spam_p1_given_0=0.001, gate_error_total=1e-3)))
    check("9f. rejects invalid SPAM probability", _raises(lambda: ArchitectureProfile(
        "superconducting", T1_s=1e-4, T2_s=1e-4, gate_duration_ns=50,
        spam_p0_given_1=1.5, spam_p1_given_0=0.001, gate_error_total=1e-3)))

    # 2. T2 <= 2*T1 across many samples, all architectures, both regimes
    ok = True
    for arch in ARCHITECTURES:
        for regime in ("ideal", "nisq"):
            t = generate_device(arch, 10, seed=123, regime=regime)
            ok &= bool(np.all(t.T2_s <= 2.0 * t.T1_s * (1 + 1e-9)))
    check("2. T2 <= 2*T1 enforced", ok)

    # 5. Deterministic reproducibility
    t1 = generate_device("superconducting", 4, seed=42, regime="nisq")
    t2 = generate_device("superconducting", 4, seed=42, regime="nisq")
    check("5. reproducible DeviceTruth", bool(np.allclose(t1.T1_s, t2.T1_s) and np.allclose(t1.epsilon_gate, t2.epsilon_gate)))
    p1 = generate_calibration_prior(t1, qubit=0, calibration_seed=7)
    p2 = generate_calibration_prior(t2, qubit=0, calibration_seed=7)
    check("5b. reproducible CalibrationPrior", p1.T1_prior_s == p2.T1_prior_s and p1.epsilon_prior == p2.epsilon_prior)

    # 6. Architecture-specific distinction
    sc = generate_device("superconducting", 1, seed=1, regime="nisq")
    ti = generate_device("trapped_ion", 1, seed=1, regime="nisq")
    check("6. architectures produce distinct mechanisms",
          set(sc.drift_processes) != set(ti.drift_processes) or set(sc.architecture_specific) != set(ti.architecture_specific))

    # 7. Ideal vs NISQ
    ideal = generate_device("superconducting", 2, seed=99, regime="ideal")
    nisq = generate_device("superconducting", 2, seed=99, regime="nisq")
    check("7a. ideal has no drift/crosstalk", len(ideal.drift_processes) == 0 and len(ideal.crosstalk) == 0)
    check("7b. nisq has drift and crosstalk", len(nisq.drift_processes) > 0 and any(m.enabled for m in nisq.crosstalk))

    # 8. Calibration prior: bounded, non-degenerate, physically admissible
    truth8 = generate_device("superconducting", 1, seed=77, regime="ideal")
    FRAC = 0.10
    priors = [generate_calibration_prior(truth8, qubit=0, calibration_seed=s,
                                         perturbation_frac=FRAC) for s in range(200)]
    T1_true = float(truth8.T1_s[0])
    ratios = np.array([p.T1_prior_s / T1_true for p in priors])
    check("8a. prior is never a verbatim copy of truth", bool(np.all(ratios != 1.0)))
    check("8b. prior stays inside the +/-10% envelope (no unbounded blowups)",
          bool(np.all(ratios >= 1 - FRAC - 1e-9) and np.all(ratios <= 1 + FRAC + 1e-9)),
          f"observed ratio range [{ratios.min():.4f}, {ratios.max():.4f}]")
    check("8c. prior actually varies with the calibration seed", bool(np.std(ratios) > 0))
    check("8d. prior respects T2 <= 2*T1 (physically admissible seed)",
          all(p.T2_prior_s <= 2 * p.T1_prior_s * (1 + 1e-9) for p in priors))
    check("8e. no delta_omega prior is produced (detuning is not a calibrated parameter)",
          not hasattr(priors[0], "delta_omega_prior_rad_s"))
    check("8f. SPAM passes through unperturbed (read live, not inherited from a stale record)",
          priors[0].spam_p0_given_1_prior == truth8.spam_p0_given_1[0]
          and priors[0].spam_p1_given_0_prior == truth8.spam_p1_given_0[0])
    check("8g. perturbation_frac=0 reproduces truth exactly (sweepable down to the no-gap limit)",
          generate_calibration_prior(truth8, 0, perturbation_frac=0.0).T1_prior_s == T1_true)
    check("8h. rejects an out-of-range perturbation_frac",
          _raises(lambda: generate_calibration_prior(truth8, 0, perturbation_frac=1.5)))

    # 8i-8k. The default envelope half-width is resolved per
    # (architecture, regime), and an explicit argument still wins. Each of the
    # four cases is checked through the actual sampling path -- the observed
    # spread of 400 priors must fill its own table entry's envelope and must
    # not exceed it -- so this cannot pass if the table is read but ignored.
    expected_table = {
        ("superconducting", "ideal"): 0.10,
        ("superconducting", "nisq"):  0.25,
        ("trapped_ion",     "ideal"): 0.05,
        ("trapped_ion",     "nisq"):  0.15,
    }
    resolve_ok, sample_ok, detail = True, True, []
    for (arch_k, regime_k), frac_k in expected_table.items():
        resolve_ok &= resolve_calibration_perturbation(arch_k, regime_k) == frac_k
        t_k = generate_device(arch_k, 1, seed=4242, regime=regime_k)
        r_k = np.array([generate_calibration_prior(t_k, 0, calibration_seed=s).T1_prior_s
                        / evaluate_truth_at(t_k, 0, 0.0)["T1_s"] for s in range(400)])
        within = bool(np.all(r_k >= 1 - frac_k - 1e-9) and np.all(r_k <= 1 + frac_k + 1e-9))
        # Fills at least 85% of its own envelope: rules out a narrower default
        # silently being applied (e.g. the old flat 0.10 under a nisq key).
        fills = bool((r_k.max() - r_k.min()) > 0.85 * 2 * frac_k)
        sample_ok &= within and fills
        detail.append(f"{arch_k[:2]}/{regime_k}:{frac_k:.2f}"
                      f"[{r_k.min():.3f},{r_k.max():.3f}]")
    check("8i. _CALIB_PERTURBATION resolves per (architecture, regime)",
          resolve_ok, str(expected_table))
    check("8j. the resolved default is the envelope actually sampled from",
          sample_ok, "  ".join(detail))
    # Override: a trapped-ion nisq device defaults to 0.15, so forcing 0.25
    # must widen it past what the table alone could ever produce.
    t_ov = generate_device("trapped_ion", 1, seed=4242, regime="nisq")
    T1_ov = evaluate_truth_at(t_ov, 0, 0.0)["T1_s"]
    r_def = np.array([generate_calibration_prior(t_ov, 0, calibration_seed=s).T1_prior_s / T1_ov
                      for s in range(400)])
    r_ov = np.array([generate_calibration_prior(t_ov, 0, calibration_seed=s,
                                                perturbation_frac=0.25).T1_prior_s / T1_ov
                     for s in range(400)])
    check("8k. an explicit perturbation_frac overrides the table",
          bool(r_def.max() <= 1.15 + 1e-9 and r_ov.max() > 1.15
               and np.all(r_ov <= 1.25 + 1e-9)),
          f"default max ratio {r_def.max():.4f} (table 0.15) vs "
          f"explicit-0.25 max ratio {r_ov.max():.4f}")

    # 3. SPAM bounds
    ok = True
    for arch in ARCHITECTURES:
        t = generate_device(arch, 5, seed=55)
        ok &= bool(np.all((t.spam_p0_given_1 >= 0) & (t.spam_p0_given_1 <= 1)))
        ok &= bool(np.all((t.spam_p1_given_0 >= 0) & (t.spam_p1_given_0 <= 1)))
    check("3. SPAM probabilities in [0,1]", ok)

    # 1. T1 exponential relaxation, verified via a real Aer simulation, not just the formula
    try:
        from qiskit import QuantumCircuit
        t_ideal = generate_device("superconducting", 1, seed=5, regime="ideal",
                                  overrides={"T1_range_s": (100e-6, 100e-6), "T_phi_range_s": (100e-6, 100e-6)})
        t_ideal.T2_s[:] = np.minimum(t_ideal.T2_s, 2 * t_ideal.T1_s)
        delays_ns = [0, 20_000, 50_000, 100_000]
        rel_errs = []
        for dns in delays_ns:
            qc = QuantumCircuit(1, 1, name="t1")
            qc.x(0)
            if dns > 0:
                qc.delay(dns, 0, unit="ns")
            qc.measure(0, 0)
            counts = simulate_circuit(qc, t_ideal, shots=20_000, t_s=0.0)
            p1 = counts.get("1", 0) / 20_000
            p01 = t_ideal.spam_p0_given_1[0]; p10 = t_ideal.spam_p1_given_0[0]
            expected = np.exp(-dns * 1e-9 / 100e-6) * (1 - p01) + (1 - np.exp(-dns * 1e-9 / 100e-6)) * p10
            rel_errs.append(abs(p1 - expected))
        check("1. T1 exponential relaxation matches Aer within shot noise", max(rel_errs) < 0.03,
              f"max abs error {max(rel_errs):.4f}")
    except ImportError:
        print("[SKIP] 1. T1 exponential relaxation (qiskit-aer not installed)")

    # 4. Zero-noise limit
    try:
        t_zero = generate_device("superconducting", 1, seed=0, regime="ideal",
                                 overrides={"eps_range": (1e-12, 1e-12)})
        t_zero.spam_p0_given_1[:] = 0.0
        t_zero.spam_p1_given_0[:] = 0.0
        t_zero.T1_s[:] = 1.0
        t_zero.T2_s[:] = 1.0
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(1, 1)
        qc.x(0); qc.measure(0, 0)
        counts = simulate_circuit(qc, t_zero, shots=2000)
        check("4. zero-noise limit recovers deterministic outcome", counts.get("1", 0) / 2000 > 0.99)
    except ImportError:
        print("[SKIP] 4. zero-noise limit (qiskit-aer not installed)")

    # Circuit-aware crosstalk: simultaneous vs sequential gates must differ
    try:
        from qiskit import QuantumCircuit
        t_xt = generate_device("superconducting", 2, seed=3, regime="nisq")
        t_xt.crosstalk[0].angle_rad = np.pi / 2  # exaggerate for a clean structural test
        qc_simul = QuantumCircuit(2, 2, name="simul")
        qc_simul.h(0); qc_simul.h(1); qc_simul.measure([0, 1], [0, 1])
        qc_seq = QuantumCircuit(2, 2, name="seq")
        qc_seq.h(0); qc_seq.barrier(); qc_seq.h(1); qc_seq.measure([0, 1], [0, 1])
        n_kicks_simul = len(_inject_crosstalk_kicks(qc_simul, t_xt).data) - len(qc_simul.data)
        n_kicks_seq = len(_inject_crosstalk_kicks(qc_seq, t_xt).data) - len(qc_seq.data)
        check("crosstalk fires on simultaneous layer, not on barrier-separated gates",
              n_kicks_simul > 0 and n_kicks_seq == 0,
              f"simul={n_kicks_simul} seq={n_kicks_seq}")
    except ImportError:
        print("[SKIP] crosstalk circuit-awareness (qiskit not installed)")

    # Detuning regression test: varying ONLY delta_omega must actually change
    # the simulated Ramsey outcomes. Before _inject_detuning_phases() existed,
    # delta_omega was carried in DeviceTruth, sampled, estimated by the
    # calibration prior and written to the manifest, but never applied to any
    # circuit -- so Ramsey counts were bit-for-bit identical across a 50,000
    # rad/s swing in true detuning and Canary correctly recovered ~0.
    # Also asserts the physics is right per probe: T1 must be UNaffected (a
    # Z-rotation cannot change a Z-basis measurement of a state along +/-Z)
    # and Hahn echo must be UNaffected (the mid-sequence X refocuses static
    # detuning: Rz(phi) X Rz(phi) = X), matching forward_echo()'s lack of a
    # detuning term in the frozen engine.
    try:
        from . import inversion as _inv2  # package build: packaged 1_inversion.py

        _prof = _inv2.BackendProfile(
            architecture="superconducting", T1_prior_s=150e-6, T2_prior_s=90e-6,
            dt_ns=_inv2.ARCH_DEFAULTS["superconducting"]["dt_ns"],
            backend_name="dw_regression", prior_confidence="live")
        _circs, _meta = _inv2.build_probe_circuits(_prof)
        _by_kind = {k: [c for c in _circs if c.name.startswith(k)]
                    for k in ("ramsey", "t1_", "echo")}

        def _probe_p1(kind, dw_value):
            t = generate_device("superconducting", 1, seed=7, regime="ideal")
            t.delta_omega_rad_s[:] = dw_value
            res = simulate_probe(_by_kind[kind], t, shots=[8000] * len(_by_kind[kind]),
                                 t_s=0.0, seed=123, dt_ns=_prof.dt_ns)
            return np.array([c.get("1", 0) / 8000 for c in res.counts_list])

        _ram_0 = _probe_p1("ramsey", 0.0)
        _ram_hi = _probe_p1("ramsey", 3.0e4)
        _t1_0, _t1_hi = _probe_p1("t1_", 0.0), _probe_p1("t1_", 3.0e4)
        _ec_0, _ec_hi = _probe_p1("echo", 0.0), _probe_p1("echo", 3.0e4)

        _ram_delta = float(np.max(np.abs(_ram_hi - _ram_0)))
        _t1_delta = float(np.max(np.abs(_t1_hi - _t1_0)))
        _ec_delta = float(np.max(np.abs(_ec_hi - _ec_0)))
        check("detuning: Ramsey outcomes respond to delta_omega",
              _ram_delta > 0.05, f"max |dP(1)| = {_ram_delta:.4f} (expected >> 0)")
        check("detuning: T1 probe correctly insensitive to delta_omega",
              _t1_delta < 0.02, f"max |dP(1)| = {_t1_delta:.4f} (expected ~0)")
        check("detuning: Hahn echo correctly refocuses static delta_omega",
              _ec_delta < 0.02, f"max |dP(1)| = {_ec_delta:.4f} (expected ~0)")
    except ImportError:
        print("[SKIP] detuning regression (qiskit-aer not installed)")

    # F. Regression test: the trapped-ion gate-repetition probe built via the
    # REAL GPI2 native pair (1_inversion.py's _default_trapped_ion_pair emits
    # "gpi"/"gpi2" instructions whenever qiskit_ionq is installed) must show
    # genuine epsilon-dependent decay. Before item E added "gpi"/"gpi2" to
    # build_noise_model's default gate_names, these instructions carried zero
    # simulated error and P(0) stayed flat regardless of N.
    try:
        from . import inversion as _inv  # package build: packaged 1_inversion.py

        truth_ti = generate_device("trapped_ion", 1, seed=11, regime="ideal",
                                   overrides={"eps_range": (0.01, 0.01)})
        qc_lo = _inv._build_gate_rep_circuit(5, architecture="trapped_ion")
        qc_hi = _inv._build_gate_rep_circuit(100, architecture="trapped_ion")
        uses_gpi2 = any(inst.operation.name in ("gpi", "gpi2") for inst in qc_lo.data)
        p0_lo = simulate_circuit(qc_lo, truth_ti, shots=4000, t_s=0.0).get("0", 0) / 4000
        p0_hi = simulate_circuit(qc_hi, truth_ti, shots=4000, t_s=0.0).get("0", 0) / 4000
        check("F. trapped-ion gate-repetition (gpi2 native pair) shows real eps-dependent decay",
              uses_gpi2 and p0_lo > 0.8 and p0_hi < 0.65,
              f"uses_gpi2={uses_gpi2} p0(N=5)={p0_lo:.3f} p0(N=100)={p0_hi:.3f}")
    except ImportError:
        print("[SKIP] F. trapped-ion gpi2 gate-repetition decay (qiskit-aer/qiskit_ionq not installed)")

    # G. Crosstalk is a per-pair distribution, not one worst-case constant,
    # AND its magnitude agrees with an independent measurement of the same
    # physical quantity on different hardware.
    all_angles = []
    for s in range(60):
        t = generate_device("superconducting", 4, seed=100_000 + s, regime="nisq")
        all_angles.extend(m.angle_rad for m in t.crosstalk)
    all_angles = np.array(all_angles)
    check("G1. crosstalk angles are a distribution, not one repeated constant",
          len(set(np.round(all_angles, 12))) > 5,
          f"{len(set(np.round(all_angles, 12)))} distinct values across {len(all_angles)} samples")

    # G2 is a CROSS-VALIDATION, deliberately not a self-consistency check. The
    # test it replaces compared the sampled distribution back against the same
    # provenance number that generated it, so it passed happily on a model
    # that was off by an order of magnitude. Instead: the ZZ shifts come from
    # Mueller et al. (fitted digital twin, QuantWare flux-tunable device);
    # Rudinger et al. measured a coherent crosstalk ANGLE directly by
    # simultaneous gate set tomography on a completely different transmon
    # platform. Two sources, two techniques, one number -- so the mean kick
    # angle this module produces must land near Rudinger's 14.4 mrad. The
    # second assertion is the regression guard against the J-vs-ZZ category
    # error, whose signature was a ~174 mrad mean.
    mean_angle_mrad = float(np.mean(all_angles)) * 1e3
    rudinger_mrad = MODEL_PROVENANCE[
        "sc.crosstalk.coherent_angle_rad_uncompensated"]["value"] * 1e3
    buggy_j_model_mrad = 174.0
    check("G2. mean crosstalk kick cross-validates against Rudinger et al. "
          "(within a factor of 2 of 14.4 mrad)",
          0.5 * rudinger_mrad <= mean_angle_mrad <= 2.0 * rudinger_mrad,
          f"mean angle = {mean_angle_mrad:.2f} mrad vs Rudinger "
          f"{rudinger_mrad:.1f} mrad (allowed {0.5 * rudinger_mrad:.1f}-"
          f"{2.0 * rudinger_mrad:.1f})")
    check("G2b. mean crosstalk kick is NOT within an order of magnitude of the "
          "superseded J-based value (regression guard)",
          mean_angle_mrad < buggy_j_model_mrad / 10.0,
          f"mean angle = {mean_angle_mrad:.2f} mrad; the J-vs-ZZ category "
          f"error produced {buggy_j_model_mrad:.0f} mrad, so anything above "
          f"{buggy_j_model_mrad / 10.0:.1f} mrad means it is back")
    t_scaled_half = generate_device("superconducting", 2, seed=100_000, regime="nisq", crosstalk_scale=0.5)
    t_scaled_full = generate_device("superconducting", 2, seed=100_000, regime="nisq", crosstalk_scale=1.0)
    check("G3. crosstalk_scale linearly scales the kick angle",
          bool(np.isclose(t_scaled_half.crosstalk[0].angle_rad,
                          0.5 * t_scaled_full.crosstalk[0].angle_rad)),
          f"scale=0.5 angle={t_scaled_half.crosstalk[0].angle_rad:.6f} "
          f"scale=1.0 angle={t_scaled_full.crosstalk[0].angle_rad:.6f}")

    # H. _proc_one_over_f() produces a genuine 1/f^alpha spectrum (checked via
    # periodogram slope, not just "the function runs"). Tolerance is loose
    # (+/-0.25) because a finite sum of n_lorentzians Lorentzians only
    # approximates the target exponent, with a small consistent bias
    # verified empirically before writing this test.
    for alpha_test in (0.57, 1.0):
        rng_h = np.random.default_rng(3)
        n_h = 32768
        y_h = _proc_one_over_f(rng_h, n_h, amplitude=1.0, dt=1.0, alpha=alpha_test, n_lorentzians=20)
        freqs_h = np.fft.rfftfreq(n_h, d=1.0)[1:]
        psd_h = (np.abs(np.fft.rfft(y_h)) ** 2)[1:]
        lo_h, hi_h = int(0.01 * len(freqs_h)), int(0.5 * len(freqs_h))
        slope_h, _ = np.polyfit(np.log10(freqs_h[lo_h:hi_h]), np.log10(psd_h[lo_h:hi_h]), 1)
        check(f"H. one_over_f(alpha={alpha_test}) periodogram slope matches -alpha within 0.25",
              abs(slope_h - (-alpha_test)) < 0.25,
              f"fitted slope={slope_h:+.3f}, expected~{-alpha_test:+.2f}")
    check("H2. 'one_over_f' registered and reachable via sample_trajectory()",
          bool(np.any(sample_trajectory(
              DriftProcess("one_over_f", {"amplitude": 1.0, "alpha": 1.0}),
              master_seed=1, salt="t", n_steps=100, dt_s=1.0, base_value=0.0) != 0.0)))

    # I. (removed) -- previously asserted that a deterministic 60 Hz power-line
    # sinusoid modulated trapped-ion detuning. That mechanism was deleted as a
    # category error (the cited measurement is of the MOTIONAL-mode secular
    # frequency, not qubit detuning) compounded by a timescale error (a 16.7 ms
    # period held static across a 0.1-3 s Ramsey delay). Its replacement is the
    # regression guard below, which asserts the term is GONE.
    t_nopl = generate_device("trapped_ion", 1, seed=5, regime="nisq")
    dw_a = evaluate_truth_at(t_nopl, 0, 0.0)["delta_omega_rad_s"]
    dw_b = evaluate_truth_at(t_nopl, 0, 1.0 / (4.0 * 60.0))["delta_omega_rad_s"]
    check("I. no power-line sinusoid on trapped-ion detuning (regression guard)",
          "detuning_powerline_hz" not in t_nopl.architecture_specific
          and "detuning_powerline_amplitude_hz" not in t_nopl.architecture_specific
          and abs(dw_b - dw_a) < 1e-9,
          f"quarter-60Hz-period change in dw = {abs(dw_b - dw_a):.3e} rad/s (must be ~0)")

    # J. enable_heating_gate_error defaults OFF (Change 4): the Class-C,
    # self-documented-as-unvalidated heating-to-gate-error mapping must
    # contribute exactly zero unless explicitly opted into.
    t_heat_off = generate_device("trapped_ion", 1, seed=9, regime="nisq")
    t_heat_on = generate_device("trapped_ion", 1, seed=9, regime="nisq", enable_heating_gate_error=True)
    check("J. heating-to-gate-error mapping is opt-in and defaults to zero",
          t_heat_off.architecture_specific["heating_extra_gate_error"] == 0.0
          and t_heat_on.architecture_specific["heating_extra_gate_error"] > 0.0,
          f"off={t_heat_off.architecture_specific['heating_extra_gate_error']} "
          f"on={t_heat_on.architecture_specific['heating_extra_gate_error']}")

    # K. No sampled instance is unrecoverable by construction. The engine's
    # shortest Ramsey delay is 0.5*T2_PRIOR, so the phase that actually
    # accumulates is |dw_true| * 0.5 * T2_prior, and it must stay below pi for
    # delta_omega to be identifiable at all. This is checked against the real
    # prior (not the truth), across every architecture x regime, because the
    # failure it guards was invisible to any test that used T2_true.
    alias_detail, alias_ok = [], True
    for arch_k in ARCHITECTURES:
        for regime_k in ("ideal", "nisq"):
            worst, n_alias = 0.0, 0
            for sd in range(300):
                t_k = generate_device(arch_k, 1, seed=sd, regime=regime_k)
                pr_k = generate_calibration_prior(t_k, 0)
                phase = abs(float(t_k.delta_omega_rad_s[0])) * 0.5 * pr_k.T2_prior_s
                worst = max(worst, phase)
                n_alias += phase >= np.pi
            alias_ok &= (n_alias == 0)
            alias_detail.append(f"{arch_k[:2]}/{regime_k}:{n_alias}/300 "
                                f"worst={worst / np.pi:.3f}pi")
    check("K. no instance aliases against its own calibration-prior delay grid",
          alias_ok, "  ".join(alias_detail))

    # L. _TRUTH_RANGES declares what it samples. The second coherence key
    # bounds T_phi; T2 is derived, so the sampled T2 must sit inside the range
    # the T1 and T_phi ranges jointly imply -- and NOT inside the T_phi range,
    # which is the misdeclaration this replaced.
    key_ok = all(set(v) == {"T1_s", "T_phi_s", "eps"} for v in _TRUTH_RANGES.values())
    check("L1. _TRUTH_RANGES keys name the quantity actually sampled (T_phi_s, not T2_s)",
          key_ok, str({a: sorted(v) for a, v in _TRUTH_RANGES.items()}))
    derived_ok, derived_detail = True, []
    for arch_k in ARCHITECTURES:
        r_k = _TRUTH_RANGES[arch_k]
        t_k = generate_device(arch_k, 2000, seed=11, regime="ideal")
        # Monotone in both arguments, so the extremes of the derived T2 range
        # are attained at the corners of the (T1, T_phi) box.
        def _t2(t1, tphi):
            return min(1.0 / (1.0 / (2.0 * t1) + 1.0 / tphi), 2.0 * t1)
        lo_k = _t2(r_k["T1_s"][0], r_k["T_phi_s"][0])
        hi_k = _t2(r_k["T1_s"][1], r_k["T_phi_s"][1])
        derived_ok &= bool(t_k.T2_s.min() >= lo_k * (1 - 1e-9)
                           and t_k.T2_s.max() <= hi_k * (1 + 1e-9))
        derived_detail.append(f"{arch_k[:2]} derived[{lo_k:.4g},{hi_k:.4g}] "
                              f"sampled[{t_k.T2_s.min():.4g},{t_k.T2_s.max():.4g}]")
    check("L2. sampled T2 lies inside the range DERIVED from T1 x T_phi",
          derived_ok, "  ".join(derived_detail))
    # The deprecated alias must still work, warn, and give identical numbers.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t_old = generate_device("superconducting", 4, seed=21, regime="ideal",
                                overrides={"T2_range_s": (50e-6, 60e-6)})
    t_new = generate_device("superconducting", 4, seed=21, regime="ideal",
                            overrides={"T_phi_range_s": (50e-6, 60e-6)})
    check("L3. deprecated 'T2_range_s' override still works, warns, and is numerically identical",
          bool(np.array_equal(t_old.T2_s, t_new.T2_s))
          and any(issubclass(w.category, DeprecationWarning) for w in caught),
          f"{len(caught)} warning(s); T2 arrays equal="
          f"{bool(np.array_equal(t_old.T2_s, t_new.T2_s))}")

    # M. Compensation cannot make crosstalk worse. The entry that violated
    # this was deleted (see the REMOVED note in MODEL_PROVENANCE); the check is
    # written to hold either way, so it stays meaningful if one is restored.
    unc = MODEL_PROVENANCE["sc.crosstalk.coherent_angle_rad_uncompensated"]["value"]
    comp_entry = MODEL_PROVENANCE.get("sc.crosstalk.coherent_angle_rad_compensated")
    check("M. no provenance entry claims compensated crosstalk exceeds uncompensated",
          comp_entry is None or comp_entry["value"] < unc,
          "entry absent (deleted: 0.016% residual cannot be read as an angle "
          "without an invented assumption)" if comp_entry is None
          else f"compensated={comp_entry['value']:.3e} < uncompensated={unc:.3e}")

    # N. Per-qubit SPAM spread: off by default and bit-for-bit inert, on when
    # asked, and bounded by the band requested.
    t_s0a = generate_device("superconducting", 10, seed=31, regime="nisq")
    t_s0b = generate_device("superconducting", 10, seed=31, regime="nisq",
                            spam_spread_frac=0.0)
    base01 = get_architecture_profile("superconducting").spam_p0_given_1
    check("N1. spam_spread_frac=0.0 is the default and changes nothing bit-for-bit",
          bool(np.array_equal(t_s0a.spam_p0_given_1, t_s0b.spam_p0_given_1)
               and np.array_equal(t_s0a.spam_p1_given_0, t_s0b.spam_p1_given_0)
               and np.array_equal(t_s0a.T1_s, t_s0b.T1_s)
               and np.array_equal(t_s0a.delta_omega_rad_s, t_s0b.delta_omega_rad_s)
               and np.all(t_s0a.spam_p0_given_1 == base01)),
          f"{len(np.unique(t_s0a.spam_p0_given_1))} unique p0|1 across 10 qubits "
          f"(downstream draws unshifted: T1 and dw arrays identical)")
    F_SP = 0.30
    t_sp = generate_device("superconducting", 10, seed=31, regime="nisq",
                           spam_spread_frac=F_SP)
    base10 = get_architecture_profile("superconducting").spam_p1_given_0
    in_band = bool(
        np.all(t_sp.spam_p0_given_1 >= (1 - F_SP) * base01 - 1e-12)
        and np.all(t_sp.spam_p0_given_1 <= (1 + F_SP) * base01 + 1e-12)
        and np.all(t_sp.spam_p1_given_0 >= (1 - F_SP) * base10 - 1e-12)
        and np.all(t_sp.spam_p1_given_0 <= (1 + F_SP) * base10 + 1e-12))
    in_range = bool(np.all((t_sp.spam_p0_given_1 >= 0) & (t_sp.spam_p0_given_1 <= 1))
                    and np.all((t_sp.spam_p1_given_0 >= 0) & (t_sp.spam_p1_given_0 <= 1)))
    check("N2. spam_spread_frac>0 gives n_qubits distinct values, in band and in [0,1]",
          len(np.unique(t_sp.spam_p0_given_1)) == 10
          and len(np.unique(t_sp.spam_p1_given_0)) == 10 and in_band and in_range,
          f"{len(np.unique(t_sp.spam_p0_given_1))} unique p0|1 in "
          f"[{t_sp.spam_p0_given_1.min():.5f}, {t_sp.spam_p0_given_1.max():.5f}] "
          f"vs band [{(1 - F_SP) * base01:.5f}, {(1 + F_SP) * base01:.5f}]")
    check("N3. rejects an out-of-range spam_spread_frac",
          _raises(lambda: generate_device("superconducting", 2, seed=1,
                                          spam_spread_frac=1.0)))

    # P. Drift regression guards. (P1) at t=0 every parameter equals its
    # sampled base value -- a previous max(2, ...) grid applied one full drift
    # step at t=0, which pushed 290/300 trapped-ion and 116/300
    # superconducting benchmark instances past the Ramsey aliasing limit.
    # (P2) the drift spread matches the DECLARED stationary std (x(t)-x(0)
    # tends to sqrt(2)*sigma), which fails for both previous bugs: the
    # sample-std normalisation (heavy tail up to ~180x) and the missing
    # sqrt(2*theta) conversion on the trapped-ion OU (~150x). (P3) the
    # declared drift cannot by itself limit Ramsey T2* = sqrt(2)/sigma below
    # the architecture's baseline T2, so drift does not contradict the
    # coherence the module samples.
    p1_ok = True
    for arch_k in ARCHITECTURES:
        for regime_k in ("ideal", "nisq"):
            for sd in range(20):
                t_k = generate_device(arch_k, 1, seed=sd, regime=regime_k)
                tv_k = evaluate_truth_at(t_k, 0, 0.0)
                p1_ok &= (tv_k["delta_omega_rad_s"] == float(t_k.delta_omega_rad_s[0])
                          and bool(np.isclose(tv_k["T1_s"], float(t_k.T1_s[0]), rtol=1e-12)))
    check("P1. evaluate_truth_at(t=0) returns the sampled base values exactly", p1_ok)
    p2_ok, p2_detail = True, []
    for arch_k, t_late in (("superconducting", 50_000.0), ("trapped_ion", 2_000.0)):
        amp_k = get_architecture_profile(arch_k).extra["detuning_drift_amplitude_rad_s"]
        dev_k = np.array([evaluate_truth_at(generate_device(arch_k, 1, seed=sd, regime="nisq"),
                                            0, t_late)["delta_omega_rad_s"]
                          - generate_device(arch_k, 1, seed=sd, regime="nisq").delta_omega_rad_s[0]
                          for sd in range(300)])
        ratio_k = float(np.std(dev_k) / (np.sqrt(2.0) * amp_k))
        p2_ok &= 0.75 < ratio_k < 1.25 and float(np.max(np.abs(dev_k))) < 6.0 * np.sqrt(2.0) * amp_k
        p2_detail.append(f"{arch_k[:2]}: std/(sqrt2*sigma)={ratio_k:.2f} "
                         f"max={np.max(np.abs(dev_k)) / amp_k:.1f}sigma")
    check("P2. detuning drift spread matches its declared stationary std (no heavy tail)",
          p2_ok, "  ".join(p2_detail))
    p3_ok, p3_detail = True, []
    for arch_k in ARCHITECTURES:
        sigma_k = get_architecture_profile(arch_k).extra["detuning_drift_amplitude_rad_s"]
        t2_med = float(np.median(generate_device(arch_k, 2000, seed=17, regime="ideal").T2_s))
        bound_k = np.sqrt(2.0) / t2_med
        p3_ok &= sigma_k <= bound_k
        p3_detail.append(f"{arch_k[:2]}: sigma={sigma_k:.3g} <= sqrt2/median(T2)={bound_k:.3g} rad/s")
    check("P3. drift amplitude does not contradict the sampled T2 (T2* = sqrt2/sigma >= median T2)",
          p3_ok, "  ".join(p3_detail))
    _tr = generate_device("superconducting", 1, seed=5, regime="nisq")
    _pa = sample_trajectory(_tr.drift_processes["detuning"], 5, "p4", 10, 90.0, 0.0)
    _pb = sample_trajectory(_tr.drift_processes["detuning"], 5, "p4", 50, 90.0, 0.0)
    check("P4. extending a drift trajectory never rewrites its past", bool(np.allclose(_pa, _pb[:10])))

    # Q-S. Gate-error physics, checked through real Aer simulations.
    try:
        from qiskit import QuantumCircuit

        def _p0(qc_, truth_, shots_=20_000):
            return simulate_circuit(qc_, truth_, shots=shots_, seed=11).get("0", 0) / shots_

        def _zero_spam(t_):
            t_.spam_p0_given_1[:] = 0.0
            t_.spam_p1_given_0[:] = 0.0
            return t_

        # Q. The drive-amplitude error follows the rotation sign: it cancels in
        # the G*G^-1 pair the gate-repetition probe is built from, but
        # accumulates between two identical gates. The previous fixed-sign
        # Rx(+theta) accumulated in both, driving the trapped-ion probe's
        # chi2/dof to 100-250.
        q_ok, q_detail = True, []
        for arch_k in ARCHITECTURES:
            base = generate_device(arch_k, 1, seed=3, regime="nisq",
                                   overrides={"eps_range": (2e-3, 2e-3),
                                              "T1_range_s": (1e3, 1e3), "T_phi_range_s": (1e3, 1e3)})
            base = _zero_spam(base)
            base.gate_duration_ns = 0.0
            base.architecture_specific["coherent_fraction"] = 0.5
            stoch_k = 2e-3 * 0.5
            g, g_inv = _native_inverse_pair(arch_k)
            n_rep = 60
            qc_pair = QuantumCircuit(1, 1)
            qc_same = QuantumCircuit(1, 1)
            for _ in range(n_rep):
                qc_pair.append(g, [0]); qc_pair.append(g_inv, [0])
                qc_same.append(g, [0]); qc_same.append(g, [0])
                qc_same.append(g, [0]); qc_same.append(g, [0])
            qc_pair.measure(0, 0); qc_same.measure(0, 0)
            d_pair = abs(_p0(qc_pair, base) - 0.5 * (1 + (1 - 2 * stoch_k) ** (2 * n_rep)))
            d_same = abs(_p0(qc_same, base) - 0.5 * (1 + (1 - 2 * stoch_k) ** (4 * n_rep)))
            q_ok &= d_pair < 0.012 and d_same > 0.05
            q_detail.append(f"{arch_k[:2]}: |P0 - depol-only| pair={d_pair:.3f} same-sign={d_same:.3f}")
        check("Q. amplitude error cancels in G*G^-1 and accumulates in G*G", q_ok, "  ".join(q_detail))

        # R. effective_gate_epsilon() is computed from the same channel Aer
        # applies. R1: the exact pair channel, iterated N times, reproduces the
        # simulated gate-repetition P0 (this caught injected coherent-error
        # unitaries being transpiled into noisy "u" gates). R2: in the
        # trapped-ion case, where T1 recovery toward |0> is negligible, the
        # engine's own forward_gate() with eps = effective_gate_epsilon
        # matches the simulation, while the sampled depolarizing budget alone
        # does not. (For transmons, amplitude damping toward |0> over ~50 us
        # of gates adds an offset forward_gate() does not model; the engine's
        # affine model forward_gate_affine() computes z* from T1, T2 and the
        # gate time, so R2 is asserted only for the trapped ion, reported otherwise.)
        from qiskit.quantum_info import Operator, PTM, SuperOp
        r1_ok, r2_ok, r_detail = True, True, []
        for arch_k, t2_k in (("superconducting", 60e-6), ("trapped_ion", 0.2)):
            for regime_k in ("ideal", "nisq"):
                t_k = _zero_spam(generate_device(
                    arch_k, 1, seed=8, regime=regime_k,
                    overrides={"eps_range": (2e-4, 2e-4), "T_phi_range_s": (t2_k, t2_k)}))
                e_eff = effective_gate_epsilon(t_k, 0, 0.0)
                g, g_inv = _native_inverse_pair(arch_k)
                n_rep = int(round(0.25 / e_eff))
                qc_r = QuantumCircuit(1, 1)
                for _ in range(n_rep):
                    qc_r.append(g, [0]); qc_r.append(g_inv, [0])
                qc_r.measure(0, 0)
                p_sim = _p0(qc_r, t_k, 40_000)
                err_k, f_k = _gate_noise_channel(t_k, 0, 0.0)

                def _stp(op):
                    ch = SuperOp(Operator(op)).compose(SuperOp(err_k))
                    return ch.compose(SuperOp(Operator(op).power(f_k))) if f_k > 0 else ch
                ptm_pair = np.real(PTM(_stp(g).compose(_stp(g_inv))).data)
                r_vec = np.linalg.matrix_power(ptm_pair, n_rep) @ np.array([1.0, 0.0, 0.0, 1.0])
                p_chan = 0.5 * (1.0 + r_vec[3])
                p_eff = 0.5 * (1 + (1 - 2 * e_eff) ** (2 * n_rep))
                p_dep = 0.5 * (1 + (1 - 2 * float(t_k.epsilon_gate[0])) ** (2 * n_rep))
                r1_ok &= abs(p_sim - p_chan) < 0.012
                if arch_k == "trapped_ion":
                    r2_ok &= abs(p_sim - p_eff) < 0.012 and abs(p_sim - p_dep) > 0.03
                r_detail.append(f"{arch_k[:2]}/{regime_k}: sim={p_sim:.3f} channel={p_chan:.3f} "
                                f"fwd(eff)={p_eff:.3f} fwd(depol-only)={p_dep:.3f}")
        check("R1. the exact gate channel reproduces the simulated gate-repetition decay",
              r1_ok, "  ".join(r_detail))
        check("R2. forward_gate with effective_gate_epsilon matches simulation (trapped ion)",
              r2_ok, "  ".join(r_detail))
    except ImportError:
        print("[SKIP] Q/R gate-error physics (qiskit-aer not installed)")

    check("S. virtual Z-type gates (s, sdg, rz) carry no gate noise",
          not ({"s", "sdg", "rz"} & set(_NOISY_GATES)), str(_NOISY_GATES))

    # T-X. NISQ options used by 3_benchmark.py (all opt-in in this library).
    # T. Quasi-static dephasing: per-shot offsets reproduce the Ramsey
    # envelope exp(-t/T2 - sigma^2 t^2/2), the echo keeps the Markovian
    # exp(-t/T2), and T1-type circuits are left untouched.
    try:
        from qiskit import QuantumCircuit
        t_q = generate_device("superconducting", 1, seed=3, regime="nisq",
                              quasi_static_dephasing=True,
                              overrides={"T1_range_s": (300e-6, 300e-6),
                                         "T_phi_range_s": (100e-6, 100e-6),
                                         "eps_range": (1e-12, 1e-12)})
        t_q.architecture_specific["coherent_fraction"] = 0.0
        t_q.drift_processes = {}
        t_q.spam_p0_given_1[:] = 0.0
        t_q.spam_p1_given_0[:] = 0.0
        t_q.gate_duration_ns = 0.0
        sig_q = 2.0e4
        t_q.architecture_specific["quasi_static_dw_std_rad_s"] = np.array([sig_q])
        # dw = 0 puts the full envelope on P(1) (cos = 1), so the check is
        # sensitive to it: analytic 0.379 vs Markov-only 0.252 here.
        t_q.delta_omega_rad_s[:] = 0.0
        T2_q, n_q = float(t_q.T2_s[0]), 6000
        tau_q = 0.7 * T2_q
        qc_r = QuantumCircuit(1, 1); qc_r.h(0); qc_r.delay(tau_q, 0, unit="s"); qc_r.h(0); qc_r.measure(0, 0)
        qc_e = QuantumCircuit(1, 1); qc_e.h(0); qc_e.delay(tau_q / 2, 0, unit="s"); qc_e.x(0)
        qc_e.delay(tau_q / 2, 0, unit="s"); qc_e.h(0); qc_e.measure(0, 0)
        qc_t = QuantumCircuit(1, 1); qc_t.x(0); qc_t.delay(tau_q, 0, unit="s"); qc_t.measure(0, 0)
        cosp = np.cos(t_q.delta_omega_rad_s[0] * tau_q)
        p_r = simulate_circuit(qc_r, t_q, shots=n_q, seed=101).get("1", 0) / n_q
        p_e = simulate_circuit(qc_e, t_q, shots=n_q, seed=102).get("1", 0) / n_q
        pa_r = 0.5 * (1 - np.exp(-tau_q / T2_q) * np.exp(-(sig_q * tau_q) ** 2 / 2) * cosp)
        pm_r = 0.5 * (1 - np.exp(-tau_q / T2_q) * cosp)       # Markov-only Ramsey
        pa_e = 0.5 * (1 - np.exp(-tau_q / T2_q))
        sd_r, sd_e = np.sqrt(pa_r * (1 - pa_r) / n_q), np.sqrt(pa_e * (1 - pa_e) / n_q)
        check("T. quasi-static offsets: Ramsey envelope exact, echo refocused, T1 untouched",
              abs(p_r - pa_r) < 4 * sd_r and abs(p_r - pm_r) > 6 * sd_r
              and abs(p_e - pa_e) < 4 * sd_e
              and not _phase_reaches_measurement(qc_t, 0) and _phase_reaches_measurement(qc_r, 0),
              f"Ramsey sim={p_r:.4f} analytic={pa_r:.4f} markov-only={pm_r:.4f}; "
              f"echo sim={p_e:.4f} analytic={pa_e:.4f}")
        t_q2 = generate_device("superconducting", 50, seed=8, regime="nisq",
                               quasi_static_dephasing=True)
        r_q2 = t_q2.architecture_specific["t2_ramsey_over_echo"]
        s_q2 = t_q2.architecture_specific["quasi_static_dw_std_rad_s"]
        # 1/e time of exp(-t/T2 - s^2 t^2/2) must be r*T2 (the drawn ratio).
        e_q2 = r_q2 + 0.5 * (s_q2 * r_q2 * t_q2.T2_s) ** 2
        t_q3 = generate_device("superconducting", 50, seed=8, regime="nisq")
        check("T2. quasi-static sigma realises the drawn T2R/T2E ratio; other draws unmoved",
              bool(np.allclose(e_q2, 1.0)
                   and set(np.round(r_q2, 12)) <= set(np.round(MODEL_PROVENANCE[
                       "sc.quasi_static.t2_ramsey_over_echo"]["value"], 12))
                   and np.array_equal(t_q2.T1_s, t_q3.T1_s)
                   and np.array_equal(t_q2.delta_omega_rad_s, t_q3.delta_omega_rad_s)
                   and "quasi_static_dw_std_rad_s" not in t_q3.architecture_specific
                   and "quasi_static_dw_std_rad_s" not in generate_device(
                       "trapped_ion", 2, seed=8, regime="nisq",
                       quasi_static_dephasing=True).architecture_specific),
              f"max|1/e condition - 1| = {np.max(np.abs(e_q2 - 1.0)):.2e}")

        # U. Detuning during pulses: the gate-repetition circuit follows the
        # exact product of Rz(a) G Rz(a) Rz(a) G^-1 Rz(a) (no other noise),
        # and is untouched with the option off.
        from . import inversion as _inv3  # package build: packaged 1_inversion.py
        from qiskit.quantum_info import Operator as _Op
        u_ok, tw_ok, u_detail = True, True, []
        for arch_k, dw_k in (("superconducting", 2 * np.pi * 5e3), ("trapped_ion", 2 * np.pi * 0.5)):
            t_u = generate_device(arch_k, 1, seed=4, regime="nisq", detuning_during_gates=True,
                                  overrides={"T1_range_s": (1e3, 1e3), "T_phi_range_s": (1e3, 1e3),
                                             "eps_range": (1e-12, 1e-12)})
            t_u.architecture_specific["coherent_fraction"] = 0.0
            t_u.drift_processes = {}
            t_u.spam_p0_given_1[:] = 0.0
            t_u.spam_p1_given_0[:] = 0.0
            t_u.delta_omega_rad_s[:] = dw_k
            a_k = dw_k * t_u.gate_duration_ns * 1e-9 / 2
            rz_k = np.diag([np.exp(-1j * a_k / 2), np.exp(1j * a_k / 2)])
            g_k, gi_k = _native_inverse_pair(arch_k)
            u_pair = rz_k @ _Op(gi_k).data @ rz_k @ rz_k @ _Op(g_k).data @ rz_k
            n_u = 400
            psi = np.linalg.matrix_power(u_pair, n_u) @ np.array([1.0, 0.0])
            # Untwirled train: this test checks the simulator's pulse physics
            # against the exact product of identical pairs.
            qc_u = _inv3._build_gate_rep_circuit(n_u, arch_k, twirl=False)
            qc_tw = _inv3._build_gate_rep_circuit(n_u, arch_k, twirl=True)
            p_on = simulate_circuit(qc_u, t_u, shots=20_000, seed=5).get("0", 0) / 20_000
            p_tw = simulate_circuit(qc_tw, t_u, shots=20_000, seed=5).get("0", 0) / 20_000
            t_u.architecture_specific["detuning_during_gates"] = False
            p_off = simulate_circuit(qc_u, t_u, shots=20_000, seed=5).get("0", 0) / 20_000
            u_ok &= abs(p_on - abs(psi[0]) ** 2) < 0.012 and p_off > 0.99
            # U2: the virtual-Z frame twirl of 1_inversion's gate probe turns
            # the coherent detuning build-up into a random walk, so the
            # twirled train stays within shot noise of the detuning-free one.
            tw_ok &= abs(p_tw - p_off) < 0.01
            u_detail.append(f"{arch_k[:2]}: P0 on={p_on:.4f} exact={abs(psi[0]) ** 2:.4f} "
                            f"off={p_off:.4f} twirled_on={p_tw:.4f}")
        check("U. detuning during pulses matches the exact pair unitary; off changes nothing",
              u_ok, "  ".join(u_detail))
        check("U2. frame-twirled gate train is insensitive to detuning during pulses",
              tw_ok, "  ".join(u_detail))
    except ImportError:
        print("[SKIP] T/U quasi-static and gate-detuning physics (qiskit-aer not installed)")

    # V. Stale SPAM in the prior: default oracle; f > 0 stays in its envelope,
    # differs from truth, and moves no other prior draw.
    t_v = generate_device("superconducting", 1, seed=77, regime="nisq")
    pv0 = generate_calibration_prior(t_v, 0, calibration_seed=3)
    pv_list = [generate_calibration_prior(t_v, 0, calibration_seed=s, spam_perturbation_frac=0.25)
               for s in range(200)]
    rv = np.array([p.spam_p0_given_1_prior / t_v.spam_p0_given_1[0] for p in pv_list])
    pv1 = generate_calibration_prior(t_v, 0, calibration_seed=3, spam_perturbation_frac=0.25)
    check("V. spam_perturbation_frac: oracle by default, bounded when on, no other draw moved",
          pv0.spam_p0_given_1_prior == t_v.spam_p0_given_1[0]
          and bool(np.all(rv >= 0.75 - 1e-9) and np.all(rv <= 1.25 + 1e-9) and np.all(rv != 1.0))
          and (rv.max() - rv.min()) > 0.85 * 0.5
          and pv1.T1_prior_s == pv0.T1_prior_s and pv1.epsilon_prior == pv0.epsilon_prior
          and _raises(lambda: generate_calibration_prior(t_v, 0, spam_perturbation_frac=1.0)),
          f"p0|1 prior/truth in [{rv.min():.3f}, {rv.max():.3f}]")

    # W. Literature detuning population: inside +/-2*pi*W, fills it, is
    # independent of T2, and leaves every other draw aligned.
    w_ok, w_detail = True, []
    for arch_k, key_k in (("superconducting", "sc"), ("trapped_ion", "ti")):
        w_k = 2 * np.pi * MODEL_PROVENANCE[f"{key_k}.dw_population.literature_halfwidth_hz"]["value"]
        t_w = generate_device(arch_k, 2000, seed=13, regime="nisq", dw_population="literature")
        t_w0 = generate_device(arch_k, 2000, seed=13, regime="nisq")
        dw_w = t_w.delta_omega_rad_s
        w_ok &= bool(np.all(np.abs(dw_w) <= w_k) and np.max(np.abs(dw_w)) > 0.99 * w_k
                     and np.min(np.abs(dw_w)) < 0.01 * w_k
                     and np.array_equal(t_w.T1_s, t_w0.T1_s)
                     and np.array_equal(t_w.epsilon_gate, t_w0.epsilon_gate))
        w_detail.append(f"{arch_k[:2]}: |dw| in [{np.min(np.abs(dw_w)):.3g}, "
                        f"{np.max(np.abs(dw_w)):.4g}] vs W={w_k:.4g} rad/s")
    t_wo = generate_device("superconducting", 50, seed=13, regime="nisq",
                           dw_population="literature", overrides={"dw_halfwidth_hz": 1.0e3})
    w_ok &= bool(np.all(np.abs(t_wo.delta_omega_rad_s) <= 2 * np.pi * 1.0e3))
    check("W. literature detuning population bounded, filled, overridable, draws aligned",
          w_ok, "  ".join(w_detail))

    # X. probe_run_duration_s: the frozen-mode run length equals the
    # time-resolved schedule's, and a trapped-ion T1 delay dominates it.
    try:
        from qiskit import QuantumCircuit
        t_x = generate_device("trapped_ion", 1, seed=2, regime="ideal")
        qc_x = QuantumCircuit(1, 1); qc_x.x(0); qc_x.delay(10.0, 0, unit="s"); qc_x.measure(0, 0)
        d_x = probe_run_duration_s([qc_x, qc_x], t_x, [300, 300])
        d_expect = 2 * 300 * (10.0 + t_x.gate_duration_ns * 1e-9)
        check("X. probe_run_duration_s matches delays + gate time per shot (lower bound)",
              abs(d_x - d_expect) < 1e-6 * d_expect, f"{d_x:.3f} s vs {d_expect:.3f} s")
    except ImportError:
        print("[SKIP] X. probe_run_duration_s (qiskit not installed)")

    # End-to-end chain into the frozen engine: DeviceTruth -> simulate_probe -> lindblad_inversion
    try:
        from . import inversion as inv  # package build: packaged 1_inversion.py

        truth_e2e = generate_device("superconducting", 1, seed=21, regime="ideal",
                                    overrides={"T1_range_s": (150e-6, 150e-6), "T_phi_range_s": (90e-6, 90e-6)})
        profile = inv.BackendProfile.from_architecture("superconducting")
        circuits, meta = inv.build_probe_circuits(profile)
        result = simulate_probe(circuits, truth_e2e, shots=[300]*meta["n_t1"] + [600]*meta["n_ramsey"]
                                + [500]*meta["n_gate"] + [500]*meta["n_echo"], t_s=0.0)
        inv_result = inv.lindblad_inversion(
            result.counts_list, meta, profile,
            shots_t1=300, shots_ramsey=600, shots_gate=500, shots_echo=500)
        rel_T1_err = abs(inv_result.T1_s - truth_e2e.T1_s[0]) / truth_e2e.T1_s[0]
        check("end-to-end: DeviceTruth -> simulate_probe -> frozen lindblad_inversion recovers T1",
              rel_T1_err < 0.5, f"true={truth_e2e.T1_s[0]*1e6:.1f}us recovered={inv_result.T1_s*1e6:.1f}us")
    except ImportError:
        print("[SKIP] end-to-end chain (qiskit-aer not installed)")
    except Exception as e:
        check("end-to-end: DeviceTruth -> simulate_probe -> frozen lindblad_inversion", False, f"raised {e!r}")

    print("=" * 70)
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("All validation tests passed.")
    print("=" * 70)


if __name__ == "__main__":
    _run_validation_tests()