"""
1c_simulation_models.py -- literature-grounded physics simulation layer for
Quantum Canary's research pipeline.

Models quantum hardware in-silico for superconducting,
trapped-ion, and neutral-atom qubits under two regimes:

    ideal / model_consistent  -- correctly specified, stationary Markovian
                                 noise (the null baseline for chi2/dof).
    nisq  / model_mismatched  -- adds specific, literature-motivated
                                 violations of that assumption (temporal
                                 drift, coherent crosstalk, motional heating,
                                 Rydberg-specific mechanisms, atom loss).


SCOPE (see spec Section 17 for the full list): this module is the physics/
data-generation layer only. It does not compute R^2, chi2/dof, shot-budget
benchmarks, or run VQE/QAOA/QEC/plotting -- those live in 2_*, 3_*, 4_*.

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
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

ARCHITECTURES: Tuple[str, ...] = ("superconducting", "trapped_ion", "neutral_atom")
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
        "value": 150e-6, "unit": "s", "evidence_class": "C",
        "source": "Consistent with 1_inversion.py ARCH_DEFAULTS['superconducting']['T1_s']",
        "location": "n/a (engine-compatibility default)",
        "notes": ("No single value is 'the' superconducting T1 -- it is device-"
                  "dependent. Kept identical to the frozen engine's arch-typical "
                  "prior so DeviceTruth samples land inside the delay grids "
                  "1_inversion.py's BackendProfile.from_architecture() builds."),
    },
    "sc.T2.baseline_s": {
        "value": 90e-6, "unit": "s", "evidence_class": "C",
        "source": "Consistent with 1_inversion.py ARCH_DEFAULTS['superconducting']['T2_s']",
        "location": "n/a (engine-compatibility default)",
        "notes": "Same rationale as sc.T1.baseline_s.",
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
        "value": 0.30, "unit": "dimensionless", "evidence_class": "C",
        "source": "Mechanism per Klimov et al., PRL 121, 090502 (2018) "
                 "(TLS resonance suppresses T1 when a defect sweeps through "
                 "the qubit frequency); no specific fractional-amplitude "
                 "figure was independently confirmed",
        "location": "n/a -- phenomenological",
        "notes": ("The MECHANISM (T1 dips when a TLS is resonant) is A-class; "
                  "the specific magnitude of a typical dip is not something "
                  "this audit could pin to one number in the paper without "
                  "re-deriving it from the full spectroscopic dataset. Treated "
                  "as a stress-test parameter: T1 fluctuates by roughly this "
                  "fraction around its OU mean. Do not read this as a measured "
                  "figure."),
    },
    "sc.detuning_drift.amplitude_rad_s": {
        "value": 2 * np.pi * 2.0e3, "unit": "rad/s", "evidence_class": "C",
        "source": "Mechanism only; no superconducting-specific quantitative "
                 "low-frequency detuning noise spectrum was independently "
                 "confirmed for this audit",
        "location": "n/a -- phenomenological",
        "notes": ("Low-frequency detuning drift is a real, widely reported "
                  "phenomenon in fixed-frequency transmons (flux noise, "
                  "photon-shot-noise dephasing, often loosely called '1/f' "
                  "in the literature), but this audit did not confirm a "
                  "specific spectral amplitude to cite, and the process used "
                  "here (single-pole/Lorentzian low-pass-filtered white "
                  "noise, see _proc_lowpass_filtered) is NOT a 1/f spectrum "
                  "-- its power spectral density is flat below cutoff_hz and "
                  "rolls off as 1/f^2 above it, whereas true 1/f (pink) noise "
                  "has a 1/f rolloff across the entire band. It is named and "
                  "documented as 'low-frequency colored noise' throughout "
                  "this module specifically to avoid that conflation. 2 "
                  "kHz*2pi is order-of-magnitude consistent with "
                  "1_inversion.py's own dw_typical_khz=5.0 default and is "
                  "used purely to exercise "
                  "the mechanism, not as a measured value."),
    },
    "sc.gate.coherent_overrotation_rad": {
        "value": 3.5e-4, "unit": "rad", "evidence_class": "B",
        "source": "Derived from 1_inversion.py ARCH_DEFAULTS['superconducting']"
                 "['eps_typical']=3.5e-4 via eps_coh = delta^2/4",
        "location": "n/a (reuses the existing engine's own established "
                   "eps<->delta relationship, already used in "
                   "3_null_nonm.py's _perturb_instance)",
        "notes": ("delta = 2*sqrt(eps_typical) so that squaring and dividing "
                  "by 4 reproduces eps_typical exactly at delta -- i.e. this "
                  "coherent angle is calibrated so the ideal-regime gate error "
                  "matches the frozen engine's own eps_typical when used "
                  "alone, rather than inventing an independent number."),
    },
    "sc.gate.stochastic_error": {
        "value": 3.5e-4, "unit": "dimensionless", "evidence_class": "C",
        "source": "Consistent with 1_inversion.py ARCH_DEFAULTS['superconducting']['eps_typical']",
        "location": "n/a (engine-compatibility default)",
        "notes": "Used as the ideal-regime depolarizing rate; see also the coherent entry above.",
    },
    "sc.spam.p0_given_1": {
        "value": 0.0092, "unit": "dimensionless", "evidence_class": "C",
        "source": "Consistent with 1_inversion.py ARCH_DEFAULTS['superconducting']",
        "location": "n/a (engine-compatibility default; engine attributes this "
                   "figure to Chen et al., npj Quantum Inf. 9, 26 (2023))",
        "notes": "Kept identical to the frozen engine so SPAM in synthetic data matches SPAM assumed at inversion time when regime='ideal'.",
    },
    "sc.spam.p1_given_0": {
        "value": 0.0009, "unit": "dimensionless", "evidence_class": "C",
        "source": "Consistent with 1_inversion.py ARCH_DEFAULTS['superconducting']",
        "location": "n/a (engine-compatibility default)",
        "notes": "See sc.spam.p0_given_1.",
    },
    "sc.gate.duration_ns": {
        "value": 50.0, "unit": "ns", "evidence_class": "C",
        "source": "Consistent with 1_inversion.py ARCH_DEFAULTS['superconducting']['gate_time_ns']",
        "location": "n/a (engine-compatibility default)",
        "notes": "Typical single-qubit (sx) gate duration order of magnitude for fixed-frequency transmons.",
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
    "sc.crosstalk.coherent_angle_rad_compensated": {
        "value": 0.016e-3 * 1000, "unit": "rad", "evidence_class": "A",
        "source": "Rudinger et al., PRX Quantum 2, 040338 (2021)",
        "location": "reported Q6 context-to-context error with crosstalk compensation applied",
        "notes": ("0.016(8)% context-to-context error variation after the "
                  "paper's own crosstalk compensation -- i.e. compensation "
                  "reduces (not eliminates) coherent crosstalk. Exposed for "
                  "callers who want to model a compensated device; not used "
                  "by default (regime='nisq' uses the uncompensated value, "
                  "since Quantum Canary's whole premise is characterizing "
                  "hardware as it actually runs)."),
    },
    # ---------------- Trapped ion ----------------
    "ti.T1.baseline_s": {
        "value": 1000.0, "unit": "s", "evidence_class": "C",
        "source": "Consistent with 1_inversion.py ARCH_DEFAULTS['trapped_ion']['T1_s']",
        "location": "n/a (engine-compatibility default)",
        "notes": ("This is NOT Wang et al.'s record coherence figure -- see "
                  "ti.T2.record_s below for that, kept clearly separate. "
                  "1000s represents an operationally typical (not "
                  "record-setting) hyperfine-qubit relaxation floor, matching "
                  "the frozen engine's own default so DeviceTruth samples "
                  "land inside 1_inversion.py's expected delay-grid range."),
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
        "value": 2 * np.pi * 100.0, "unit": "rad/s", "evidence_class": "C",
        "source": "Mechanism per Wang et al. (2021) (limiting factors: magnetic "
                 "field fluctuation, microwave frequency instability); no "
                 "specific pre-mitigation frequency-noise spectrum for a "
                 "*typical* (non-record) device was independently confirmed",
        "location": "n/a -- phenomenological",
        "notes": ("100 Hz*2pi is order-of-magnitude consistent with "
                  "1_inversion.py's dw_typical_khz=0.5 default for trapped "
                  "ions. Used to exercise the mechanism (slow OU drift in "
                  "detuning from magnetic-field/laser noise), not as a "
                  "measured spectral density."),
    },
    "ti.gate.error_total": {
        "value": 5e-4, "unit": "dimensionless", "evidence_class": "C",
        "source": "Consistent with 1_inversion.py ARCH_DEFAULTS['trapped_ion']['eps_typical']",
        "location": "n/a (engine-compatibility default)",
        "notes": "Total single-qubit gate error (coherent + stochastic combined).",
    },
    "ti.gate.coherent_fraction": {
        "value": 0.5, "unit": "dimensionless", "evidence_class": "C",
        "source": "No trapped-ion-specific coherent/stochastic gate-error "
                 "split was independently confirmed for this audit",
        "location": "n/a -- phenomenological",
        "notes": "Even 50/50 split between coherent over-rotation and stochastic depolarizing, used only to exercise both channels.",
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
        "value": 0.0005, "unit": "dimensionless", "evidence_class": "C",
        "source": "Consistent with 1_inversion.py ARCH_DEFAULTS['trapped_ion'] "
                 "(engine attributes this to Mai et al., arXiv:2402.18868 (2024))",
        "location": "n/a (engine-compatibility default)",
        "notes": "Kept identical to the frozen engine.",
    },
    "ti.spam.p1_given_0": {
        "value": 0.0018, "unit": "dimensionless", "evidence_class": "C",
        "source": "Consistent with 1_inversion.py ARCH_DEFAULTS['trapped_ion']",
        "location": "n/a (engine-compatibility default)",
        "notes": "See ti.spam.p0_given_1.",
    },
    "ti.gate.duration_ns": {
        "value": 135_000.0, "unit": "ns", "evidence_class": "C",
        "source": "Consistent with 1_inversion.py ARCH_DEFAULTS['trapped_ion']['gate_time_ns']",
        "location": "n/a (engine-compatibility default)",
        "notes": "Typical single-qubit gate duration order of magnitude for trapped-ion platforms.",
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
    # ---------------- Neutral atom ----------------
    "na.T1.baseline_s": {
        "value": 10.0, "unit": "s", "evidence_class": "C",
        "source": "Consistent with 1_inversion.py ARCH_DEFAULTS['neutral_atom']['T1_s']",
        "location": "n/a (engine-compatibility default)",
        "notes": ("Ground/hyperfine-state qubit lifetime (trap lifetime, "
                  "background-gas collisions) -- explicitly NOT the Rydberg "
                  "state lifetime, which is a separate, much shorter, gate-"
                  "level quantity (see na.rydberg.*)."),
    },
    "na.T2.baseline_s": {
        "value": 1.0, "unit": "s", "evidence_class": "C",
        "source": "Consistent with 1_inversion.py ARCH_DEFAULTS['neutral_atom']['T2_s']",
        "location": "n/a (engine-compatibility default)",
        "notes": "Ground-state coherence time, limited by laser phase noise / magnetic field gradients.",
    },
    "na.gate_2q.duration_s": {
        "value": 262e-9, "unit": "s", "evidence_class": "A",
        "source": "Evered et al., Nature 622, 268 (2023)",
        "location": "derived from reported Omega*T/(2*pi)=1.215 with Omega/2pi=4.6 MHz",
        "notes": ("This is the two-QUBIT Rydberg-mediated entangling gate "
                  "duration, not a single-qubit gate. See the scoping note "
                  "in build_noise_model()/RYDBERG MECHANISM SCOPING below: "
                  "1_inversion.py's Canary gate-repetition probe uses a "
                  "single-qubit native gate pair on all three architectures "
                  "(NATIVE_INVERSE_PAIRS falls back to sx/sxdg for "
                  "unregistered architectures, including neutral_atom), which "
                  "does not invoke Rydberg excitation. The Rydberg-specific "
                  "mechanisms below are therefore applied only to circuit "
                  "instructions explicitly identified as 2-qubit Rydberg "
                  "gates, never to Canary's own single-qubit probes, unless "
                  "the caller explicitly opts in."),
    },
    "na.gate_2q.total_infidelity": {
        "value": 0.0048, "unit": "dimensionless", "evidence_class": "A",
        "source": "Evered et al., Nature 622, 268 (2023)",
        "location": "abstract / Extended Data Fig. 4 (99.52% fidelity -> 0.48% infidelity)",
        "notes": "Two-qubit parallel Rydberg CZ gate, up to 60 atoms in parallel.",
    },
    "na.rydberg.lifetime_effective_us": {
        "value": 88.0, "unit": "us", "evidence_class": "A",
        "source": "Evered et al., Nature 622, 268 (2023)",
        "location": "reported radiative lifetime 170us (n=53), blackbody-limited "
                   "128us, combined effective 88us after subtracting the "
                   "1013nm scattering contribution",
        "notes": "Used as tau_eff in p_decay = 1 - exp(-t_gate/tau_eff) for the Rydberg-decay channel.",
    },
    "na.rydberg.scattering_rate_per_s": {
        "value": 6800.0, "unit": "1/s", "evidence_class": "B",
        "source": "Derived from Evered et al., Nature 622, 268 (2023)",
        "location": "back-calculated from the reported ~0.15-0.20% intermediate-"
                   "state-scattering infidelity contribution over the 262ns gate",
        "notes": ("gamma_e = -ln(1 - p_sc) / t_gate with p_sc ~ 0.0018 "
                  "(midpoint of the reported 0.15-0.20% range) and "
                  "t_gate=262ns gives gamma_e ~ 6.8e3 /s. Used as "
                  "p_scatter = 1 - exp(-gamma_e * t_gate) for the "
                  "intermediate-state-scattering channel."),
    },
    "na.rydberg.dephasing_T2star_us": {
        "value": 3.0, "unit": "us", "evidence_class": "A",
        "source": "Evered et al., Nature 622, 268 (2023)",
        "location": "reported ground-Rydberg coherence time T2*, dominated by laser light-shift fluctuations",
        "notes": "Applied as a dephasing channel during Rydberg-state occupation only.",
    },
    "na.atom_loss.rate_per_s": {
        "value": 0.5, "unit": "1/s", "evidence_class": "C",
        "source": "Mechanism per Evered et al. (2023) and neutral-atom tweezer-"
                 "array literature generally (finite atom-survival probability "
                 "per shot); no single per-second loss rate for this specific "
                 "gate/device was independently confirmed for this audit",
        "location": "n/a -- phenomenological",
        "notes": ("Atom loss is architecturally an ERASURE channel (the qubit "
                  "leaves the computational subspace entirely), fundamentally "
                  "different from depolarizing noise, and is kept as a "
                  "distinguishable mechanism in DeviceTruth.architecture_specific "
                  "and in the simulated counts' metadata rather than folded "
                  "into a generic error rate."),
    },
    "na.gate_1q.error": {
        "value": 1e-3, "unit": "dimensionless", "evidence_class": "C",
        "source": "Consistent with a Raman/microwave-driven ground-state "
                 "rotation, order-of-magnitude only; not derived from Evered "
                 "et al.'s 2-qubit Rydberg gate budget (which is a distinct "
                 "physical mechanism, see na.gate_2q.* above)",
        "location": "n/a -- phenomenological, chosen order-of-magnitude "
                   "consistent with 1_inversion.py's eps_typical for neutral_atom",
        "notes": ("This is the number actually used for Canary's own single-"
                  "qubit gate-repetition probe. Kept explicitly and "
                  "deliberately separate from the Rydberg mechanisms so that "
                  "a 2-qubit-gate error budget is never silently applied to a "
                  "1-qubit gate."),
    },
    "na.spam.p0_given_1": {
        "value": 0.0060, "unit": "dimensionless", "evidence_class": "C",
        "source": "Consistent with 1_inversion.py ARCH_DEFAULTS['neutral_atom'] "
                 "(engine attributes this to Evered et al. (2023))",
        "location": "n/a (engine-compatibility default)",
        "notes": "Kept identical to the frozen engine.",
    },
    "na.spam.p1_given_0": {
        "value": 0.0040, "unit": "dimensionless", "evidence_class": "C",
        "source": "Consistent with 1_inversion.py ARCH_DEFAULTS['neutral_atom']",
        "location": "n/a (engine-compatibility default)",
        "notes": "See na.spam.p0_given_1.",
    },
    "na.gate_1q.duration_ns": {
        "value": 500.0, "unit": "ns", "evidence_class": "C",
        "source": "Consistent with 1_inversion.py ARCH_DEFAULTS['neutral_atom']['gate_time_ns']",
        "location": "n/a (engine-compatibility default)",
        "notes": "Order-of-magnitude typical single-qubit gate duration; not independently sourced from Evered et al. (whose reported timing is for the 2-qubit gate; see na.gate_2q.duration_s).",
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
    process_type: str  # "stationary" | "ou" | "telegraph" | "random_walk" | "lowpass_filtered"
    params: Dict[str, Any] = field(default_factory=dict)
    seed_salt: str = ""

    _VALID = ("stationary", "ou", "telegraph", "random_walk", "lowpass_filtered")

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
class CalibrationObservation:
    """Raw synthetic measurement counts from a coarse calibration experiment
    at time t0 -- an intermediate stage between DeviceTruth and
    CalibrationPrior. Never contains DeviceTruth's exact values, only shot
    counts sampled from truth's forward probability at t0."""
    qubit: int
    t0_s: float
    t1_delay_s: float
    t1_counts: Tuple[int, int]        # (n0, n1)
    echo_delay_s: float
    echo_counts: Tuple[int, int]
    ramsey_delay_s: float
    ramsey_x_counts: Tuple[int, int]
    ramsey_y_counts: Tuple[int, int]
    gate_N: Tuple[int, int]
    gate_counts: Tuple[Tuple[int, int], Tuple[int, int]]
    spam_shots: int
    spam_counts_prep0: Tuple[int, int]
    spam_counts_prep1: Tuple[int, int]


@dataclass
class CalibrationPrior:
    """What Canary is allowed to know: a point estimate derived from
    CalibrationObservation via simple closed-form estimators (deliberately
    cruder than 1_inversion.py's own bounded/weighted fits -- this module
    must not duplicate that inversion mathematics), plus a staleness age."""
    T1_prior_s: float
    T2_prior_s: float
    delta_omega_prior_rad_s: float
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
    colored noise", never "1/f", to avoid that conflation. It is a standard,
    transparent reduced-order stand-in for slow detuning/parameter drift, not
    a claim of reproducing any specific measured spectral density unless the
    caller's amplitude/cutoff came from one (see MODEL_PROVENANCE entries
    using this process, which are marked C where no spectrum number was
    independently confirmed)."""
    alpha = np.exp(-2.0 * np.pi * cutoff_hz * dt)
    white = rng.normal(0.0, 1.0, size=n)
    y = np.empty(n)
    y[0] = white[0]
    for i in range(1, n):
        y[i] = alpha * y[i - 1] + (1.0 - alpha) * white[i]
    std = np.std(y)
    return y * (amplitude / std) if std > 0 else y


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
        return _proc_ou(rng, n_steps, mean=p.get("mean", base_value),
                        sigma=p.get("sigma", 0.0), theta=p.get("theta", 1.0),
                        dt=dt_s, x0=p.get("x0", base_value))
    if pt == "telegraph":
        return _proc_telegraph(rng, n_steps, dt=dt_s, rate=p.get("rate", 1e-3),
                               baseline=base_value, amplitude_frac=p.get("amplitude_frac", 0.0))
    if pt == "random_walk":
        return _proc_random_walk(rng, n_steps, p.get("step_std", 0.0), base_value)
    if pt == "lowpass_filtered":
        return _proc_lowpass_filtered(rng, n_steps, amplitude=p.get("amplitude", 0.0),
                                      cutoff_hz=p.get("cutoff_hz", 1e-3), dt=dt_s)
    raise ValueError(f"Unknown process_type {pt!r}")


def evaluate_at(process: DriftProcess, master_seed: int, salt: Any,
                t_s: float, base_value: float, dt_s: float = 1.0) -> float:
    """Evaluate a DriftProcess at a single time t_s by materializing its
    trajectory from t=0 up to t_s at resolution dt_s and returning the last
    point. Used to time-evaluate DeviceTruth for a simulation call at time t."""
    n_steps = max(2, int(round(t_s / dt_s)) + 1)
    traj = sample_trajectory(process, master_seed, salt, n_steps, dt_s, base_value)
    return float(traj[-1])


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
                "coherent_overrotation_rad": P["sc.gate.coherent_overrotation_rad"]["value"],
                "T1_drift_timescale_s": P["sc.T1_drift.timescale_s"]["value"],
                "T1_drift_amplitude_frac": P["sc.T1_drift.amplitude_frac"]["value"],
                "detuning_drift_amplitude_rad_s": P["sc.detuning_drift.amplitude_rad_s"]["value"],
                "crosstalk_angle_rad": P["sc.crosstalk.coherent_angle_rad_uncompensated"]["value"],
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
    if architecture == "neutral_atom":
        return ArchitectureProfile(
            architecture="neutral_atom",
            T1_s=P["na.T1.baseline_s"]["value"], T2_s=P["na.T2.baseline_s"]["value"],
            gate_duration_ns=P["na.gate_1q.duration_ns"]["value"],
            spam_p0_given_1=P["na.spam.p0_given_1"]["value"],
            spam_p1_given_0=P["na.spam.p1_given_0"]["value"],
            gate_error_total=P["na.gate_1q.error"]["value"],
            extra={
                "gate_2q_duration_s": P["na.gate_2q.duration_s"]["value"],
                "gate_2q_total_infidelity": P["na.gate_2q.total_infidelity"]["value"],
                "rydberg_lifetime_eff_us": P["na.rydberg.lifetime_effective_us"]["value"],
                "rydberg_scattering_rate_per_s": P["na.rydberg.scattering_rate_per_s"]["value"],
                "rydberg_dephasing_T2star_us": P["na.rydberg.dephasing_T2star_us"]["value"],
                "atom_loss_rate_per_s": P["na.atom_loss.rate_per_s"]["value"],
            },
        )
    raise ValueError(f"Unknown architecture: {architecture!r}; must be one of {ARCHITECTURES}")


# =============================================================================
# 6. DEVICE TRUTH GENERATION
# =============================================================================

# Log-uniform truth-sampling ranges. Deliberately independent of
# 1_inversion.py's ARCH_DEFAULTS bounds (no truth leakage into the prior
# construction), but chosen to overlap them so a downstream BackendProfile's
# delay grids remain sensible. Order-of-magnitude consistent with the ranges
# already used in 3_null_nonm.py's TRUE_PARAM_RANGES (Class C: representative
# operating envelope, not a literature-measured distribution).
_TRUTH_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "superconducting": {"T1_s": (80e-6, 400e-6), "T2_s": (40e-6, 200e-6), "eps": (1e-4, 2e-3)},
    "trapped_ion":     {"T1_s": (100.0, 10_000.0), "T2_s": (0.1, 3.0),   "eps": (1e-4, 2e-3)},
    "neutral_atom":    {"T1_s": (1.0, 100.0),      "T2_s": (0.3, 3.0),  "eps": (1e-3, 1e-2)},
}


def generate_device(
    architecture: str,
    n_qubits: int,
    seed: int,
    regime: str = "ideal",
    dw_max_rad_s: Optional[float] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> DeviceTruth:
    """Sample a complete, hidden DeviceTruth.

    Per-qubit T1, T2, epsilon are drawn log-uniform from _TRUTH_RANGES
    (T2 additionally capped at 2*T1 via the standard 1/T2=1/(2T1)+1/Tphi
    decomposition); delta_omega is drawn uniform in [-dw_max, dw_max] with
    random sign. regime='ideal' attaches no drift/crosstalk processes at all
    (stationary Markovian noise -- the chi2/dof null baseline); regime='nisq'
    attaches the architecture-specific mechanisms in Sections 6-8 of the
    module spec, each traceable to MODEL_PROVENANCE.
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
    T2_lo, T2_hi = ov.get("T2_range_s", ranges["T2_s"])
    eps_lo, eps_hi = ov.get("eps_range", ranges["eps"])

    T1 = _log_uniform(T1_lo, T1_hi, n_qubits)
    T_phi = _log_uniform(T2_lo, T2_hi, n_qubits)
    T2 = np.minimum(1.0 / (1.0 / (2.0 * T1) + 1.0 / T_phi), 2.0 * T1)
    eps = _log_uniform(eps_lo, eps_hi, n_qubits)

    dw_max = dw_max_rad_s if dw_max_rad_s is not None else 0.9 * np.pi / (0.5 * float(np.median(T2)))
    dw = rng.choice([-1.0, 1.0], size=n_qubits) * rng.uniform(0.2 * dw_max, dw_max, size=n_qubits)

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
            drift["detuning"] = DriftProcess("lowpass_filtered", {
                "amplitude": profile.extra["detuning_drift_amplitude_rad_s"],
                "cutoff_hz": 1.0 / profile.extra["T1_drift_timescale_s"],
            }, seed_salt="detuning_lpf")
            if n_qubits >= 2:
                crosstalk.append(CrosstalkMechanism(
                    enabled=True, angle_rad=profile.extra["crosstalk_angle_rad"], axis="z",
                    affected_pairs=[(i, i + 1) for i in range(n_qubits - 1)]))
            arch_specific["coherent_overrotation_rad"] = profile.extra["coherent_overrotation_rad"]

        elif architecture == "trapped_ion":
            drift["detuning"] = DriftProcess("ou", {
                "mean": 0.0, "sigma": profile.extra["detuning_drift_amplitude_rad_s"],
                "theta": 1.0 / 100.0,
            }, seed_salt="detuning_ou")
            if n_qubits >= 2:
                crosstalk.append(CrosstalkMechanism(
                    enabled=True, angle_rad=profile.extra["crosstalk_angle_rad"], axis="x",
                    affected_pairs=[(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]))
            heating_axial = profile.extra["heating_axial_quanta_per_s"]
            arch_specific["heating_extra_gate_error"] = float(
                heating_axial * profile.extra["heating_to_gate_error_coeff"])
            arch_specific["heating_axial_quanta_per_s"] = heating_axial
            arch_specific["coherent_fraction"] = profile.extra["coherent_fraction"]

        elif architecture == "neutral_atom":
            arch_specific["rydberg_lifetime_eff_us"] = profile.extra["rydberg_lifetime_eff_us"]
            arch_specific["rydberg_scattering_rate_per_s"] = profile.extra["rydberg_scattering_rate_per_s"]
            arch_specific["rydberg_dephasing_T2star_us"] = profile.extra["rydberg_dephasing_T2star_us"]
            arch_specific["atom_loss_rate_per_s"] = profile.extra["atom_loss_rate_per_s"]
            drift["atom_loss"] = DriftProcess("stationary", {"std": 0.0}, seed_salt="atom_loss")

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
        proc = truth.drift_processes["detuning"]
        if proc.process_type == "lowpass_filtered":
            cutoff_hz = proc.params.get("cutoff_hz", 1e-3)
            dt_fixed = max(1.0, (1.0 / max(cutoff_hz, 1e-9)) / 20.0)
        else:
            theta_dw = proc.params.get("theta", 1.0)
            dt_fixed = max(1.0, (1.0 / theta_dw) / 20.0) if theta_dw > 0 else 1.0
        walked = evaluate_at(proc, truth.seed, ("detuning", qubit), t_s, dw, dt_s=dt_fixed)
        dw = dw + walked if proc.process_type == "lowpass_filtered" else walked

    T2 = min(T2, 2.0 * T1)
    return {"T1_s": T1, "T2_s": T2, "delta_omega_rad_s": dw,
            "epsilon_gate": float(truth.epsilon_gate[qubit]),
            "spam_p0_given_1": float(truth.spam_p0_given_1[qubit]),
            "spam_p1_given_0": float(truth.spam_p1_given_0[qubit])}


# =============================================================================
# 7. CALIBRATION: OBSERVATION -> ESTIMATOR -> PRIOR (no truth leakage)
# =============================================================================
#
# true device -> synthetic calibration observation (binomial shot noise at a
# few fixed, architecture-typical delays a real calibrator would pick without
# knowing this specific device instance) -> a crude closed-form estimator
# (deliberately simpler than 1_inversion.py's bounded/weighted fits, so this
# module never duplicates that inversion mathematics) -> CalibrationPrior.
# No step ever computes truth * noise_factor.

def _binomial(rng: np.random.Generator, shots: int, p: float) -> Tuple[int, int]:
    n1 = int(rng.binomial(shots, float(np.clip(p, 0.0, 1.0))))
    return shots - n1, n1


def run_synthetic_calibration_experiment(
    truth: DeviceTruth, qubit: int, t0_s: float = 0.0,
    shots: int = 200, calibration_seed: Optional[int] = None,
) -> CalibrationObservation:
    """Simulate a coarse, fixed-delay calibration measurement at time t0_s.

    Delays/N are chosen from the ARCHITECTURE PROFILE (what a calibrator
    would guess for this device type), not from this instance's true T1/T2 --
    that is exactly the boundary that keeps this a real observation rather
    than truth leaking through a suspiciously well-chosen probe point.
    """
    seed = calibration_seed if calibration_seed is not None else (truth.seed + 9_000_000)
    rng = _sub_rng(seed, "calib_obs", qubit, t0_s)
    profile = get_architecture_profile(truth.architecture)
    tv = evaluate_truth_at(truth, qubit, t0_s)
    T1, T2, dw = tv["T1_s"], tv["T2_s"], tv["delta_omega_rad_s"]
    p01, p10, eps = tv["spam_p0_given_1"], tv["spam_p1_given_0"], tv["epsilon_gate"]

    t1_delay = profile.T1_s
    p_t1 = np.exp(-t1_delay / T1) * (1 - p01) + (1 - np.exp(-t1_delay / T1)) * p10
    t1_counts = _binomial(rng, shots, p_t1)

    echo_delay = profile.T2_s
    p_echo = 0.5 * (1 - np.exp(-echo_delay / T2)) * (1 - p01) + 0.5 * (1 + np.exp(-echo_delay / T2)) * p10
    echo_counts = _binomial(rng, shots, p_echo)

    ramsey_delay = 0.5 * profile.T2_s
    decay = np.exp(-ramsey_delay / T2)
    p_x_ideal = 0.5 * (1 - decay * np.cos(dw * ramsey_delay))
    p_y_ideal = 0.5 * (1 - decay * np.sin(dw * ramsey_delay))
    p_x = p_x_ideal * (1 - p01) + (1 - p_x_ideal) * p10
    p_y = p_y_ideal * (1 - p01) + (1 - p_y_ideal) * p10
    ramsey_x_counts = _binomial(rng, shots, p_x)
    ramsey_y_counts = _binomial(rng, shots, p_y)

    N_lo = max(1, int(round(1.0 / (8.0 * profile.gate_error_total))))
    N_hi = max(2, 2 * N_lo)
    gate_probs = []
    for N in (N_lo, N_hi):
        p_ideal = 0.5 * (1 + (1 - 2 * eps) ** (2 * N))
        p0 = p_ideal * (1 - p10) + (1 - p_ideal) * p01
        gate_probs.append(_binomial(rng, shots, 1.0 - p0))  # (n0,n1) with n0 the "0" bucket via 1-p
    # store as (n0,n1) pairs matching forward_gate's P(measure 0) convention
    gate_counts = tuple((shots - c[1], c[1]) for c in gate_probs)

    spam_shots = max(shots, 1000)
    spam_prep0 = _binomial(rng, spam_shots, p10)   # prepared 0, measure 1 w.p. p10
    spam_prep1 = _binomial(rng, spam_shots, 1 - p01)  # prepared 1, measure 1 w.p. 1-p01

    return CalibrationObservation(
        qubit=qubit, t0_s=t0_s,
        t1_delay_s=t1_delay, t1_counts=t1_counts,
        echo_delay_s=echo_delay, echo_counts=echo_counts,
        ramsey_delay_s=ramsey_delay, ramsey_x_counts=ramsey_x_counts, ramsey_y_counts=ramsey_y_counts,
        gate_N=(N_lo, N_hi), gate_counts=gate_counts,
        spam_shots=spam_shots, spam_counts_prep0=spam_prep0, spam_counts_prep1=spam_prep1,
    )


def _estimate_prior_from_observation(obs: CalibrationObservation) -> CalibrationPrior:
    """Crude, closed-form point estimates from raw calibration counts.
    Deliberately simpler than 1_inversion.py's bounded/weighted curve_fit
    machinery (no chi2 diagnostics, no SPAM-aware joint fit) -- this is a
    coarse calibration estimate, not a Canary inversion."""
    n0, n1 = obs.spam_counts_prep1
    p1_given_1 = n1 / max(n0 + n1, 1)
    p0_given_1 = 1.0 - p1_given_1
    n0b, n1b = obs.spam_counts_prep0
    p1_given_0 = n1b / max(n0b + n1b, 1)

    n0t, n1t = obs.t1_counts
    p1_t1 = n1t / max(n0t + n1t, 1)
    p1_t1_corrected = float(np.clip((p1_t1 - p1_given_0) / max(1e-6, (1 - p0_given_1) - p1_given_0), 1e-6, 1 - 1e-6))
    T1_hat = -obs.t1_delay_s / np.log(1.0 - p1_t1_corrected) if p1_t1_corrected < 1.0 else obs.t1_delay_s

    n0e, n1e = obs.echo_counts
    p1_echo = n1e / max(n0e + n1e, 1)
    p_echo_corrected = float(np.clip((p1_echo - p1_given_0) / max(1e-6, (1 - p0_given_1) - p1_given_0), 1e-6, 1 - 1e-6))
    T2_hat = -obs.echo_delay_s / np.log(max(1e-6, 1.0 - 2.0 * p_echo_corrected)) if p_echo_corrected < 0.5 else T1_hat

    xn0, xn1 = obs.ramsey_x_counts
    yn0, yn1 = obs.ramsey_y_counts
    x_bar = 1.0 - 2.0 * xn1 / max(xn0 + xn1, 1)
    y_bar = 1.0 - 2.0 * yn1 / max(yn0 + yn1, 1)
    dw_hat = float(np.arctan2(y_bar, x_bar) / max(obs.ramsey_delay_s, 1e-12))

    (n0_lo, n1_lo), (n0_hi, n1_hi) = obs.gate_counts
    N_lo, N_hi = obs.gate_N
    p0_lo = n0_lo / max(n0_lo + n1_lo, 1)
    p0_hi = n0_hi / max(n0_hi + n1_hi, 1)
    ratio = float(np.clip(2 * p0_hi - 1, -0.999, 0.999))
    eps_hat = 0.5 * (1.0 - abs(ratio) ** (1.0 / max(2 * N_hi, 1))) if abs(ratio) > 0 else 1e-3
    eps_hat = float(np.clip(eps_hat, 1e-6, 0.5))

    return CalibrationPrior(
        T1_prior_s=max(T1_hat, 1e-9), T2_prior_s=max(min(T2_hat, 2 * T1_hat), 1e-9),
        delta_omega_prior_rad_s=dw_hat, epsilon_prior=eps_hat,
        spam_p0_given_1_prior=float(np.clip(p0_given_1, 0.0, 1.0)),
        spam_p1_given_0_prior=float(np.clip(p1_given_0, 0.0, 1.0)),
        t0_s=obs.t0_s,
    )


def generate_calibration_prior(
    truth: DeviceTruth, qubit: int = 0, t0_s: float = 0.0,
    shots: int = 200, calibration_seed: Optional[int] = None,
) -> CalibrationPrior:
    """true device -> synthetic calibration observation -> estimator -> prior.
    This is the only sanctioned path from DeviceTruth to a prior; it never
    computes truth * noise_factor."""
    obs = run_synthetic_calibration_experiment(truth, qubit, t0_s, shots, calibration_seed)
    return _estimate_prior_from_observation(obs)


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

    Coherent mechanisms (over-rotation, crosstalk) use
    qiskit_aer.noise.coherent_unitary_error with an explicit unitary --
    never approximated as depolarizing noise.

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
        from qiskit_aer.noise import (
            NoiseModel, thermal_relaxation_error, depolarizing_error,
            coherent_unitary_error, ReadoutError,
        )
    except ImportError as e:
        raise ImportError("qiskit-aer is required for build_noise_model().") from e

    # Includes IonQ's native "gpi"/"gpi2" (qiskit_ionq.ionq_gates.GPIGate/
    # GPI2Gate) alongside the generic gate set: 1_inversion.py's
    # _default_trapped_ion_pair() emits these directly whenever qiskit_ionq
    # is installed, and a circuit built that way would otherwise carry zero
    # noise on its trapped-ion gate-repetition probe (see the regression
    # test in _run_validation_tests() that specifically exercises this path).
    # "rz" is deliberately EXCLUDED: this module injects Rz instructions to
    # represent coherent detuning evolution during delays
    # (_inject_detuning_phases) and coherent crosstalk kicks
    # (_inject_crosstalk_kicks). Those are free precession / coherent
    # errors, not applied gates, and must not additionally pick up
    # depolarizing + thermal-relaxation error -- the delay's own
    # thermal_relaxation_error already accounts for decoherence over that
    # idle period, so attaching gate noise to the injected Rz would
    # double-count it. This matches the gate list used by the reference
    # implementation removed from 1_inversion.py by commit b1a8385, which
    # likewise omitted "rz". (On real superconducting hardware Rz is a
    # virtual frame change with no duration and negligible error.)
    # None of 1_inversion.py's probe circuits contain an Rz of their own.
    gate_names = native_gate_names or ["sx", "sxdg", "x", "h", "s", "sdg", "id",
                                       "rx", "ry", "r", "u", "gpi", "gpi2"]
    nm = NoiseModel()
    gate_dur_s = truth.gate_duration_ns * 1e-9
    arch = truth.architecture

    def _depolarizing_from_eps(eps_value: float):
        """Convert a gate error in 1_inversion.py's epsilon convention into
        Qiskit's depolarizing-channel parameter.

        The frozen engine's forward_gate() models per-gate Bloch-vector
        survival as (1 - 2*epsilon):

            P(0; N) = 0.5 * (1 + (1 - 2*epsilon)^(2N))

        whereas qiskit_aer's depolarizing_error(p, 1) gives survival (1 - p).
        The depolarizing parameter is therefore p = 2*epsilon, NOT epsilon.
        Without this factor, DeviceTruth.epsilon_gate and
        InversionResult.epsilon_sx silently mean different things by 2x and
        any epsilon benchmark is invalid. This matches the factor used by the
        reference implementation removed from 1_inversion.py by commit
        b1a8385 (depolarizing_error(2.0 * eps_sx, 1)).
        """
        return depolarizing_error(float(np.clip(2.0 * eps_value, 0.0, 0.75)), 1)

    for q in range(truth.n_qubits):
        tv = evaluate_truth_at(truth, q, t_s)
        T1, T2, eps = tv["T1_s"], tv["T2_s"], tv["epsilon_gate"]
        p01, p10 = tv["spam_p0_given_1"], tv["spam_p1_given_0"]
        T2_eff = min(T2, 2.0 * T1)

        relax = thermal_relaxation_error(T1, T2_eff, gate_dur_s)
        nm.add_quantum_error(relax, gate_names, [q])

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

        if arch == "superconducting":
            stoch = eps
            if stoch > 0:
                nm.add_quantum_error(_depolarizing_from_eps(stoch), gate_names, [q])
            if truth.regime == "nisq":
                coh_rad = truth.architecture_specific.get("coherent_overrotation_rad", 0.0)
                if coh_rad > 0:
                    U = _rotation_unitary("x", coh_rad)
                    nm.add_quantum_error(coherent_unitary_error(U), gate_names, [q])

        elif arch == "trapped_ion":
            coh_frac = truth.architecture_specific.get("coherent_fraction", 0.0) if truth.regime == "nisq" else 0.0
            stoch = eps * (1.0 - coh_frac)
            extra = truth.architecture_specific.get("heating_extra_gate_error", 0.0) if truth.regime == "nisq" else 0.0
            if stoch + extra > 0:
                nm.add_quantum_error(_depolarizing_from_eps(stoch + extra), gate_names, [q])
            if coh_frac > 0:
                coh_rad = 2.0 * np.sqrt(max(eps * coh_frac, 0.0))
                nm.add_quantum_error(coherent_unitary_error(_rotation_unitary("x", coh_rad)), gate_names, [q])

        elif arch == "neutral_atom":
            if eps > 0:
                nm.add_quantum_error(_depolarizing_from_eps(eps), gate_names, [q])
            # Rydberg-specific mechanisms (decay, scattering, dephasing) are
            # explicitly NOT applied here -- Canary's own gate-repetition
            # probe is a single-qubit ground/hyperfine-state rotation that
            # never invokes Rydberg excitation (see na.gate_2q.duration_s's
            # provenance note). They are applied only inside
            # apply_rydberg_gate_noise(), for callers building circuits with
            # an explicit 2-qubit Rydberg-mediated gate.

        p00, p11 = 1.0 - p10, 1.0 - p01
        nm.add_readout_error(ReadoutError([[p00, p10], [p01, p11]]), [q])

    return nm


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


def apply_rydberg_gate_noise(circuit, qubits: Tuple[int, int], truth: DeviceTruth):
    """Attach Evered et al.-derived Rydberg decay/scattering/dephasing noise
    to an explicit 2-qubit Rydberg-mediated gate on `qubits`. Only meaningful
    for architecture='neutral_atom' and only for a caller's circuit that
    actually contains a real 2-qubit entangling gate (e.g. a future VQE/QAOA
    circuit in 4_online_application.py) -- never applied to Canary's own
    single-qubit gate-repetition probe. Returns a NoiseModel fragment
    (a list of (QuantumError, qubits) pairs) the caller composes into their
    own NoiseModel; kept separate from build_noise_model() so the single-
    qubit/two-qubit mechanism boundary stays explicit and cannot be applied
    by accident.
    """
    try:
        from qiskit_aer.noise import thermal_relaxation_error, depolarizing_error
    except ImportError as e:
        raise ImportError("qiskit-aer is required for apply_rydberg_gate_noise().") from e

    if truth.architecture != "neutral_atom":
        raise ValueError("apply_rydberg_gate_noise is only defined for architecture='neutral_atom'")
    prof = get_architecture_profile("neutral_atom")
    t_g = prof.extra["gate_2q_duration_s"]
    tau_eff = prof.extra["rydberg_lifetime_eff_us"] * 1e-6
    gamma_e = prof.extra["rydberg_scattering_rate_per_s"]
    t2star = prof.extra["rydberg_dephasing_T2star_us"] * 1e-6

    p_decay = 1.0 - np.exp(-t_g / tau_eff)
    p_scatter = 1.0 - np.exp(-gamma_e * t_g)
    errors = []
    for q in qubits:
        decay_err = depolarizing_error(min(p_decay, 0.75), 1)
        scatter_err = depolarizing_error(min(p_scatter, 0.75), 1)
        dephase_err = thermal_relaxation_error(1e12, t2star, t_g)  # T1>>t_g: pure dephasing
        errors.append((decay_err.compose(scatter_err).compose(dephase_err), [q]))
    return errors


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
    from qiskit.circuit.library import RXGate, RYGate, RZGate
    gate_cls = {"x": RXGate, "y": RYGate, "z": RZGate}
    # Insert in reverse index order so earlier insertions don't shift later indices.
    for after_idx, qubit, angle, axis in sorted(insertions, key=lambda t: -t[0]):
        new_qc.data.insert(after_idx + 1, new_qc.data[after_idx].replace(
            operation=gate_cls[axis](angle), qubits=(new_qc.qubits[qubit],), clbits=()))
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

    qc = _inject_detuning_phases(circuit, truth, t_s=t_s, dt_ns=dt_ns)
    if truth.regime == "nisq":
        qc = _inject_crosstalk_kicks(qc, truth)
    nm = build_noise_model(truth, t_s=t_s, native_gate_names=native_gate_names,
                           circuit=qc, dt_ns=dt_ns)
    sim_seed = seed if seed is not None else truth.seed
    sim = AerSimulator(noise_model=nm, seed_simulator=sim_seed)
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
            "delta_omega_prior_rad_s": prior.delta_omega_prior_rad_s,
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
    "CalibrationObservation", "CalibrationPrior", "SimulationResult", "SimulationManifest",
    "get_architecture_profile", "generate_device", "evaluate_truth_at",
    "run_synthetic_calibration_experiment", "generate_calibration_prior",
    "build_noise_model", "apply_rydberg_gate_noise",
    "simulate_circuit", "simulate_probe", "make_manifest",
    "sample_trajectory", "evaluate_at",
]


# =============================================================================
# 11. VALIDATION SELF-TEST
# =============================================================================

def _run_validation_tests() -> None:
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
    na = generate_device("neutral_atom", 1, seed=1, regime="nisq")
    check("6. architectures produce distinct mechanisms",
          set(sc.drift_processes) != set(ti.drift_processes) or set(sc.architecture_specific) != set(na.architecture_specific))

    # 7. Ideal vs NISQ
    ideal = generate_device("superconducting", 2, seed=99, regime="ideal")
    nisq = generate_device("superconducting", 2, seed=99, regime="nisq")
    check("7a. ideal has no drift/crosstalk", len(ideal.drift_processes) == 0 and len(ideal.crosstalk) == 0)
    check("7b. nisq has drift and crosstalk", len(nisq.drift_processes) > 0 and any(m.enabled for m in nisq.crosstalk))

    # 8. No truth leakage: prior must differ from truth, but correlate with it
    truth8 = generate_device("superconducting", 1, seed=77, regime="ideal")
    priors = [generate_calibration_prior(truth8, qubit=0, calibration_seed=s, shots=150) for s in range(30)]
    T1_priors = np.array([p.T1_prior_s for p in priors])
    check("8a. prior varies under independent calibration noise (not a deterministic copy)",
          bool(np.std(T1_priors) > 0))
    check("8b. prior is in the right ballpark of truth (not decoupled nonsense)",
          bool(abs(np.median(T1_priors) - truth8.T1_s[0]) / truth8.T1_s[0] < 1.0))
    check("8c. CalibrationPrior dataclass has no field that is a verbatim copy of a DeviceTruth array",
          not hasattr(priors[0], "T1_s"))  # structural: CalibrationPrior has no truth-shaped field at all

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
                                  overrides={"T1_range_s": (100e-6, 100e-6), "T2_range_s": (100e-6, 100e-6)})
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

    # End-to-end chain into the frozen engine: DeviceTruth -> simulate_probe -> lindblad_inversion
    try:
        import importlib.util, pathlib, sys
        spec = importlib.util.spec_from_file_location(
            "inversion", pathlib.Path(__file__).parent / "1_inversion.py")
        inv = importlib.util.module_from_spec(spec)
        sys.modules["inversion"] = inv
        spec.loader.exec_module(inv)

        truth_e2e = generate_device("superconducting", 1, seed=21, regime="ideal",
                                    overrides={"T1_range_s": (150e-6, 150e-6), "T2_range_s": (90e-6, 90e-6)})
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
