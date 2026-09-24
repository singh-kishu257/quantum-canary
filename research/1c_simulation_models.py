"""
1c_simulation_models.py -- literature-grounded physics simulation layer for
Quantum Canary's research pipeline.

Models quantum hardware in-silico for superconducting and trapped-ion qubits
under two regimes:

    ideal / model_consistent  -- correctly specified, stationary Markovian
                                 noise (the null baseline for chi2/dof).
    nisq  / model_mismatched  -- adds specific, literature-motivated
                                 violations of that assumption (temporal
                                 drift, coherent crosstalk, motional heating).


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
      amplitude error. Class C (0.10 default; GST on IonQ Forte bounds it to
      a minority only qualitatively). Sweep 0-0.5.
  detuning drift amplitude (sc/ti.detuning_drift.amplitude_rad_s): Class C /
      B consistency bound, not a measured cloud-hardware spectrum. Sweep.
  alpha (generate_device(detuning_1_over_f=True), sc.detuning_1_over_f.alpha):
      Class C -- the specific value 0.57 could not be independently verified
      against its originally-cited source (see that entry's notes for what
      was actually found) and must not be treated as a measurement. Sweep
      across a plausible range (e.g. 0.5-1.5) instead of relying on one
      point value.

-----------------------------------------------------------------------------
EVIDENCE CLASSES (see MODEL_PROVENANCE)
-----------------------------------------------------------------------------
  A = directly measured/reported in the cited paper (number or range).
  B = mathematically derived from an A-class quantity via an explicit,
      documented relationship (e.g. p = 1 - exp(-rate * duration)).
  C = phenomenological/stress-test assumption. The cited paper supports the
      MECHANISM but not a specific number; the value here is a reasonable
      order-of-magnitude choice for exercising the mechanism, never to be
      read as a measured fact. Every C-class entry says so explicitly in
      its "notes" field.

Every numerical default lives in MODEL_PROVENANCE with its source, the
location within that source (or "abstract"/"figure N"/etc.), its evidence
class, and how it maps into this module's code. Call
validate_parameter_provenance() to check the registry's internal
consistency (it cannot verify the citations themselves -- that was done by
hand against the actual papers before writing this file; see the docstring
of each MODEL_PROVENANCE entry for what was and was not independently
confirmed).

-----------------------------------------------------------------------------
A NOTE ON WHAT COULD AND COULD NOT BE VERIFIED
-----------------------------------------------------------------------------
Every entry below with evidence_class "A" reflects a number I could
independently confirm by fetching the actual paper (arXiv abstract/HTML or a
Nature/PRL page), not one reconstructed from memory of the paper's general
reputation. Numbers I could not confirm to this standard are marked "C" and
say so, even where a specific-looking number would have been easy to invent.
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
# verified vs. assumed, and how the number maps into the code.

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
        "value": 2.35, "unit": "MHz/hr^0.5", "evidence_class": "A",
        "source": "Klimov et al., Phys. Rev. Lett. 121, 090502 (2018)",
        "location": "abstract (reported ~2.2-2.5 MHz/hr^0.5 across independently "
                    "analyzed TLS defects; 2.35 is the midpoint used here)",
        "notes": ("This is a TLS *frequency* diffusivity: individual two-level-"
                  "system defects random-walk in frequency, occasionally "
                  "sweeping through resonance with the qubit and suppressing "
                  "its T1 while resonant. It is NOT directly a T1(t) amplitude "
                  "or timescale. Used only as evidence for the CORRELATION "
                  "TIMESCALE of T1(t) fluctuations (see sc.T1_drift.timescale_s), "
                  "not as a literal formula for T1's magnitude."),
    },
    "sc.T1_drift.timescale_s": {
        "value": 1800.0, "unit": "s", "evidence_class": "B",
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
                  "qualitative timescale Klimov et al. report."),
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
    },
    "sc.gate.stochastic_error": {
        "value": 3.5e-4, "unit": "dimensionless", "evidence_class": "C",
        "source": "Engine-compatibility default (historical 1_inversion.py eps_typical)",
        "location": "n/a (engine-compatibility default)",
        "notes": ("Architecture-typical gate error recorded on ArchitectureProfile; "
                  "per-qubit truth is sampled from _TRUTH_RANGES['eps'] instead."),
    },
    "sc.spam.p0_given_1": {
        "value": 0.030, "unit": "dimensionless", "evidence_class": "B",
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
    },
    "sc.spam.p1_given_0": {
        "value": 0.008, "unit": "dimensionless", "evidence_class": "B",
        "source": "Sharma & Lau, arXiv:2607.11637, Table 4; see sc.spam.p0_given_1",
        "location": "Table 4 (Heron readout error)",
        "notes": "See sc.spam.p0_given_1 for the derivation.",
    },
    "sc.gate.duration_ns": {
        "value": 50.0, "unit": "ns", "evidence_class": "C",
        "source": "Consistent with 1_inversion.py ARCH_DEFAULTS['superconducting']['gate_time_ns']",
        "location": "n/a (engine-compatibility default)",
        "notes": ("Order-of-magnitude single-qubit (sx) duration for fixed-frequency "
                  "transmons; no cloud-device value was independently confirmed "
                  "for this audit. Sets the gate-time decoherence that "
                  "effective_gate_epsilon() folds into the epsilon truth."),
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
        "value": 0.57, "unit": "dimensionless", "evidence_class": "C",
        "source": "UNVERIFIED CITATION -- originally specified as 'measured "
                 "alpha=0.57 from arXiv:1206.6946, fitted to 90 min of "
                 "repeated Ramsey fringes.' Direct fetch of arXiv:1206.6946 "
                 "('Measurement-induced qubit state mixing in circuit QED "
                 "from up-converted dephasing noise') does not contain this "
                 "value or this experiment: that paper extracts an "
                 "equivalent flux-noise PSD from the qubit's measurement-"
                 "induced transition rate at ~1 GHz and compares it to a "
                 "1/f^alpha extrapolation of separately measured sub-1-Hz "
                 "flux-noise data, without stating a specific alpha in the "
                 "retrievable text, and does not describe a 90-minute "
                 "repeated-Ramsey experiment at all.",
        "location": "n/a -- citation could not be confirmed; not used as fact",
        "notes": ("This number is NOT Class A and must not be cited as "
                  "arXiv:1206.6946's result. It is retained only as a "
                  "phenomenological default -- sub-1 exponents for near-1/f "
                  "dephasing are qualitatively common in the superconducting-"
                  "qubit literature, but this specific value was not "
                  "independently verified for this audit. Treated as a "
                  "required sensitivity-sweep parameter (see the module "
                  "docstring's Sensitivity Sweep Parameters section and "
                  "generate_device()'s alpha kwarg): the manuscript should "
                  "report insensitivity to alpha across a swept range, not "
                  "rely on this single value."),
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
    },
    "ti.T2.baseline_s": {
        "value": 1.0, "unit": "s", "evidence_class": "C",
        "source": "Consistent with 1_inversion.py ARCH_DEFAULTS['trapped_ion']['T2_s']",
        "location": "n/a (engine-compatibility default)",
        "notes": "Operationally typical value, not Wang et al.'s record. See ti.T2.record_s.",
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
        "value": float(np.sqrt(2.0)), "unit": "rad/s", "evidence_class": "B",
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
                  "1 s T2 allows, and it aliased 290/300 benchmark instances."),
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
    #     exactly this distinction for the Hite et al. heating rates
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
        "location": "Sec. 4: 'results from gate set tomography (GST) ... indicate "
                   "the dominant errors are stochastic in nature'; Sec. 2: each "
                   "single-qubit gate is an SK1 composite sequence",
        "notes": ("GST on IonQ Forte finds errors predominantly stochastic, and "
                  "SK1 composite pulses are designed to cancel amplitude "
                  "miscalibration, so the coherent share must be a minority. "
                  "0.10 is a phenomenological point inside that qualitative "
                  "bound (no number is reported); sweep 0-0.3. Realised as a "
                  "sign-following amplitude error (see sc.gate.coherent_"
                  "fraction). REPLACES 0.5 applied as a fixed-sign Rx after "
                  "every gate, which did not cancel in a G*G^-1 pair, "
                  "accumulated as 2*N*theta, and drove the gate-repetition "
                  "chi2/dof to 100-250."),
    },
    "ti.heating.baseline_axial_quanta_per_s": {
        "value": 13.0, "unit": "quanta/s", "evidence_class": "A",
        "source": "Hite et al., Phys. Rev. Lett. 126, 230505 (2021)",
        "location": "reported baseline (no dielectric sample) axial-mode heating rate, 13(3) quanta/s at 2*pi*1.5 MHz",
        "notes": "Motional heating rate, NOT computational T1. Enters only the heating-to-gate-error mapping below.",
    },
    "ti.heating.baseline_radial_quanta_per_s": {
        "value": 29.0, "unit": "quanta/s", "evidence_class": "A",
        "source": "Hite et al., PRL 126, 230505 (2021)",
        "location": "reported baseline radial-mode heating rates, 26(6) and 32(8) "
                   "quanta/s at 2*pi*{3.2,3.4} MHz; 29 used as their midpoint",
        "notes": "See ti.heating.baseline_axial_quanta_per_s.",
    },
    "ti.heating.distance_scaling_exponent": {
        "value": 4.2, "unit": "dimensionless", "evidence_class": "A",
        "source": "Hite et al., PRL 126, 230505 (2021)",
        "location": "reported electric-field-noise distance scaling S_E ~ d^-alpha, "
                   "alpha_axial=4.016(6), alpha_radial=4.413(6); 4.2 used as their midpoint",
        "notes": ("Explicitly reported as differing from the naive 1/d^3 "
                  "infinite-dielectric-plane prediction due to finite sample "
                  "geometry. Used in the optional distance-dependent heating "
                  "helper heating_rate_at_distance()."),
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
    },
    "ti.spam.p0_given_1": {
        "value": 0.0022, "unit": "dimensionless", "evidence_class": "B",
        "source": "Chen et al. (IonQ Forte), Quantum (2024), arXiv:2308.05071, "
                 "split by 1_inversion.py's _TI_SPAM_RATIO_P0_OVER_P1",
        "location": "Sec. 2: 'The average state preparation and measurement "
                   "(SPAM) error is measured to be 0.5% on each qubit'",
        "notes": ("DERIVATION: (p0|1 + p1|0)/2 = 0.005 (Class A) with "
                  "p0|1/p1|0 = 0.2778 (the engine's ratio, itself unvalidated, "
                  "Class C) gives p0|1 = 0.0022, p1|0 = 0.0078, identical to "
                  "1_inversion.py ARCH_DEFAULTS['trapped_ion']. The previous "
                  "0.0005/0.0018 (0.12% average) was 4x below the published "
                  "Forte figure while claiming to match the engine."),
    },
    "ti.spam.p1_given_0": {
        "value": 0.0078, "unit": "dimensionless", "evidence_class": "B",
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
                  "1_inversion.py's gate_time_ns is not used by the inversion, "
                  "so the two need not agree."),
    },
    "ti.gate.error_range": {
        "value": (1e-4, 2e-3), "unit": "dimensionless (1_inversion epsilon convention)",
        "evidence_class": "B",
        "source": "Chen et al. (IonQ Forte), arXiv:2308.05071, Fig. 3; "
                 "Wright et al. (IonQ 11-qubit), arXiv:1903.08181, Table I",
        "location": "Forte single-qubit DRB error: median 2.0e-4, 10th/90th "
                   "percentile 1.8e-4/2.6e-4. 11-qubit system: RB fidelities "
                   "99.18-99.64% (average 99.5%).",
        "notes": ("CONVERSION: a depolarizing channel with Bloch shrink (1-2*eps) "
                  "has entanglement infidelity 1.5*eps, so Forte's median DRB "
                  "rate maps to eps ~ 1.3e-4 (10-90%: 1.2e-4-1.7e-4); the 2019 "
                  "11-qubit system sits at eps ~ 2.4e-3-5.5e-3. The sampled "
                  "log-uniform range 1e-4-2e-3 spans Forte-class devices up to "
                  "just below the 2019 generation (so worst-case older ions are "
                  "not sampled). Recorded for provenance; the sampler reads "
                  "_TRUTH_RANGES['trapped_ion']['eps']."),
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
    whether the citations are correct -- that verification was done by hand
    against the source papers when each entry was written; see each entry's
    'notes' for what was and was not independently confirmed."""
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
            haystack = " ".join(str(entry.get(f, "")) for f in ("notes", "source", "location")).lower()
            disclosed = any(kw in haystack for kw in (
                "phenomenological", "stress-test", "not independently confirmed",
                "no ", "engine-compatibility", "order-of-magnitude"))
            if not disclosed:
                issues.append(f"{key}: evidence_class C entries must disclose why in notes/source/location")
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
    1_inversion.py's lindblad_inversion(counts_list, meta, profile, ...)."""
    counts_list: List[Dict[str, int]]
    shots_list: List[int]
    circuit_names: List[str]
    manifest_id: str
    truth_evaluation_time_s: float


@dataclass
class SimulationManifest:
    """Full reproducibility record. truth and calibration are recorded as
    separate top-level fields so a consumer can audit that the calibration
    branch never read the truth branch's exact values."""
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
# superconducting: T_phi 40-200 us is consistent with published medians on
#   cloud-deployed IBM hardware (Heron ~168/130 us, Eagle ~268/183 us,
#   ibm_washington ~100/95 us) once those reported T2 figures are read back
#   through the decomposition above, widened modestly for per-qubit spread.
#   DERIVED T2, measured over 2000 sampled qubits: ~32-159 us.
# trapped_ion: IonQ's published Aria specification states T1 10-100 s and
#   T2 ~1 s; T1's upper bound widened to 200 s for per-qubit spread. The
#   previous (100, 10_000) s range had no citation and put its upper end
#   50-100x above any measured cloud trapped-ion T1 -- it is what produced
#   true-T1 ~ 100 s paired with a ~10,000 s prior. Because T1 >> T_phi here,
#   the 1/(2*T1) term is negligible and T2 ~ T_phi to within ~1%.
#   DERIVED T2, measured over 2000 sampled qubits: ~0.100-2.94 s.
_TRUTH_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "superconducting": {"T1_s": (80e-6, 400e-6), "T_phi_s": (40e-6, 200e-6), "eps": (1e-4, 2e-3)},
    "trapped_ion":     {"T1_s": (10.0, 200.0),   "T_phi_s": (0.1, 3.0),      "eps": (1e-4, 2e-3)},
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
) -> DeviceTruth:
    """Sample a complete, hidden DeviceTruth.

    Per-qubit T1, T_phi and epsilon are drawn log-uniform from _TRUTH_RANGES
    (keys "T1_s", "T_phi_s", "eps"). T2 is NOT sampled -- it is DERIVED from
    the sampled T1 and T_phi via 1/T2 = 1/(2*T1) + 1/T_phi and then capped at
    2*T1, so the T2 values a caller sees are narrower than, and shifted below,
    the declared T_phi range (superconducting: T_phi 40-200 us in, T2 ~32-159
    us out). delta_omega is drawn uniform in [-dw_max, dw_max] with random
    sign, below an aliasing ceiling that accounts for prior inflation (see
    the comment at the sampling site). regime='ideal' attaches no drift/crosstalk processes at all
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
        sc.detuning_1_over_f.alpha (itself Class C -- see that entry's
        notes -- so alpha should be swept, not fixed, when this is enabled).
    enable_heating_gate_error: trapped-ion only, default False. When False,
        Hite et al.'s (Class A) motional-heating rates are still recorded in
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
    # unrecoverable by construction. An explicit dw_max_rad_s still overrides
    # this entirely.
    if dw_max_rad_s is not None:
        dw_max_per_q = np.full(n_qubits, float(dw_max_rad_s))
    else:
        _prior_frac = resolve_calibration_perturbation(architecture, regime)
        dw_max_per_q = 0.9 * np.pi / (0.5 * T2) / (1.0 + _prior_frac)
    dw = rng.choice([-1.0, 1.0], size=n_qubits) * rng.uniform(
        0.2 * dw_max_per_q, dw_max_per_q, size=n_qubits)

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
            if detuning_1_over_f:
                drift["detuning"] = DriftProcess("one_over_f", {
                    "amplitude": profile.extra["detuning_drift_amplitude_rad_s"],
                    "alpha": profile.extra["detuning_1_over_f_alpha"],
                    # cutoff_hz sets evaluate_truth_at()'s grid; f_min_hz fixes
                    # the band's low edge (one-day window, phenomenological)
                    # so the trajectory never depends on how far it is queried.
                    "cutoff_hz": 1.0 / profile.extra["T1_drift_timescale_s"],
                    "f_min_hz": 1.0 / 86_400.0,
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
                "theta": 1.0 / 100.0,
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

    SPAM is passed through unperturbed. Readout error is read from the
    provider at run time rather than inherited from a stale calibration
    record, and 1_inversion.py treats it as a known constant in its forward
    models. Perturbing it here would inject a forward-model misspecification,
    which is precisely what the ideal regime is defined to exclude.
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

    return CalibrationPrior(
        T1_prior_s=T1_prior,
        T2_prior_s=T2_prior,
        epsilon_prior=eps_prior,
        spam_p0_given_1_prior=float(tv["spam_p0_given_1"]),
        spam_p1_given_0_prior=float(tv["spam_p1_given_0"]),
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


def _inject_amplitude_errors(circuit, truth: DeviceTruth, t_s: float = 0.0):
    """Return a copy of `circuit` with G^f appended after every native
    single-qubit rotation G, f = _amplitude_error_frac(). Because the error
    follows the sign of the rotation (as a miscalibrated drive amplitude
    does), G(theta)*G(-theta) cancels exactly, as the gate-repetition probe's
    G*G^-1 design assumes; errors between non-inverse gates still accumulate.
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
            new_qc.append(_fractional_power(inst.operation, f), inst.qubits, [])
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
    survival of the Z Bloch component, and forward_gate()'s model
    (1 - 2*eps)^2 = R_zz defines eps. This is the quantity the gate-repetition
    probe -- and a provider's randomized benchmarking -- actually measures: it
    INCLUDES decoherence during the gate (the coherence-limited error of
    McKay et al., arXiv:2311.05933, Eq. B2) and any coherent residual that
    survives the pair. DeviceTruth.epsilon_gate is only the sampled
    depolarizing budget; comparing a recovered epsilon against it omits the
    gate-time T1/T2 contribution (~1e-4 for a 50 ns gate on a ~100 us
    transmon) and biases every epsilon benchmark high."""
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
                            dt_ns: Optional[float] = None):
    """Return a copy of `circuit` with an explicit Rz(delta_omega * delta_t)
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
    """
    if not any(inst.operation.name == "delay" for inst in circuit.data):
        return circuit

    dw_cache: Dict[int, float] = {}
    new_qc = circuit.copy_empty_like()
    for inst in circuit.data:
        new_qc.append(inst.operation, inst.qubits, inst.clbits)
        if inst.operation.name != "delay":
            continue
        ds = _delay_seconds(inst, dt_ns)
        if ds <= 0.0:
            continue
        for qb in inst.qubits:
            q_idx = circuit.find_bit(qb).index
            if q_idx >= truth.n_qubits:
                continue
            if q_idx not in dw_cache:
                dw_cache[q_idx] = evaluate_truth_at(truth, q_idx, t_s)["delta_omega_rad_s"]
            phase = dw_cache[q_idx] * ds
            if phase != 0.0:
                new_qc.rz(phase, new_qc.qubits[q_idx])
    return new_qc


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
    """
    try:
        from qiskit_aer import AerSimulator
        from qiskit import transpile
    except ImportError as e:
        raise ImportError("qiskit-aer is required for simulate_circuit().") from e

    qc = _inject_detuning_phases(_to_aer_native(circuit), truth, t_s=t_s, dt_ns=dt_ns)
    if truth.regime == "nisq":
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
    result = sim.run(tqc, shots=shots).result()
    return result.get_counts(0)


def simulate_probe(
    circuits: List[Any], truth: DeviceTruth, shots: Any = 1024, t_s: float = 0.0,
    native_gate_names: Optional[List[str]] = None, seed: Optional[int] = None,
    dt_ns: Optional[float] = None,
) -> SimulationResult:
    """Simulate a list of probe circuits (e.g. from 1_inversion.py's
    build_probe_circuits()) and return counts directly compatible with
    lindblad_inversion(counts_list, meta, profile, ...).

    `shots` may be a single int (applied to every circuit) or a list of
    per-circuit shot counts matching `circuits`' length.

    `dt_ns` should be the originating profile's dt_ns, so delays emitted in
    hardware "dt" units are converted to seconds correctly for both the
    thermal-relaxation channel and the injected detuning phase.
    """
    n = len(circuits)
    shots_list = [shots] * n if isinstance(shots, int) else list(shots)
    if len(shots_list) != n:
        raise ValueError(f"shots list length {len(shots_list)} != number of circuits {n}")

    counts_list: List[Dict[str, int]] = []
    names: List[str] = []
    for i, (qc, sh) in enumerate(zip(circuits, shots_list)):
        raw = simulate_circuit(qc, truth, shots=sh, t_s=t_s,
                               native_gate_names=native_gate_names,
                               seed=(seed + i) if seed is not None else None,
                               dt_ns=dt_ns)
        norm: Dict[str, int] = {}
        for bitstring, cnt in raw.items():
            b = bitstring.replace(" ", "")[-1]
            norm[b] = norm.get(b, 0) + cnt
        counts_list.append(norm)
        names.append(getattr(qc, "name", f"circuit_{i}"))

    manifest_id = str(uuid.uuid5(uuid.NAMESPACE_OID,
                                 f"{truth.seed}:{truth.architecture}:{truth.n_qubits}:{truth.regime}:{t_s}"))
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
    separate top-level keys, and calibration never contains any field copied
    verbatim from truth (see generate_calibration_prior's docstring)."""
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
    "simulate_circuit", "simulate_probe", "make_manifest",
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
        import importlib.util as _ilu2, pathlib as _pl2, sys as _sys2
        _spec2 = _ilu2.spec_from_file_location(
            "inversion_dw", _pl2.Path(__file__).parent / "1_inversion.py")
        _inv2 = _ilu2.module_from_spec(_spec2)
        _sys2.modules["inversion_dw"] = _inv2
        _spec2.loader.exec_module(_inv2)

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
        import importlib.util as _ilu, pathlib as _pl, sys as _sys
        _spec = _ilu.spec_from_file_location(
            "inversion_f", _pl.Path(__file__).parent / "1_inversion.py")
        _inv = _ilu.module_from_spec(_spec)
        _sys.modules["inversion_f"] = _inv
        _spec.loader.exec_module(_inv)

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
        # of gates adds an offset forward_gate() does not model; that is a
        # property of the engine's forward model, reported, not asserted.)
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

    # End-to-end chain into the frozen engine: DeviceTruth -> simulate_probe -> lindblad_inversion
    try:
        import importlib.util, pathlib, sys
        spec = importlib.util.spec_from_file_location(
            "inversion", pathlib.Path(__file__).parent / "1_inversion.py")
        inv = importlib.util.module_from_spec(spec)
        sys.modules["inversion"] = inv
        spec.loader.exec_module(inv)

        truth_e2e = generate_device("superconducting", 1, seed=21, regime="ideal",
                                    overrides={"T1_range_s": (150e-6, 150e-6), "T_phi_range_s": (90e-6, 90e-6)})
        profile = inv.BackendProfile.from_architecture("superconducting")
        circuits, meta = inv.build_probe_circuits(profile)
        result = simulate_probe(circuits, truth_e2e, shots=[300]*meta["n_t1"] + [1000]*meta["n_ramsey"]
                                + [500]*meta["n_gate"] + [500]*meta["n_echo"], t_s=0.0)
        inv_result = inv.lindblad_inversion(
            result.counts_list, meta, profile,
            shots_t1=300, shots_ramsey=1000, shots_gate=500, shots_echo=500)
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