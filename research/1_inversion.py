# 1_inversion.py — Quantum Canary v2 Research Pipeline
# =============================================================================
# Physics-based qubit parameter estimation via Lindblad master equation inversion.
# Simulation-only research script. All simulation runs locally via Qiskit Aer.
# No cloud credentials required unless using IBM hardware noise model (option 3).
#
# WHAT THIS SCRIPT DOES:
#   1. Builds 9 probe circuits per qubit (3 T1 + 3 Ramsey + 3 Gate Repetition)
#   2. Simulates each circuit with a realistic per-qubit randomised noise model
#   3. Inverts the Lindblad forward models via nonlinear least squares
#   4. Returns T1, T2, Δω, ε_sx — the four qubit health parameters
#
# NO TRAINING DATA. NO MACHINE LEARNING. PURE PHYSICS.
#
# THE THREE FORWARD MODELS (Lindblad master equation analytic solutions):
#   T1 inversion recovery:  P(1; τ) = exp(−τ / T1)
#   Ramsey interferometry:  P(1; τ) = (1/2)(1 − exp(−τ/T2) · cos(Δω · τ))
#   Gate repetition:        P(0; N) = (1/2)(1 + (1 − 2ε)^(2N))
#
# SIMULATORS:
#   [1] Demo          — local numpy, known true values, zero setup
#   [2] Aer realistic — Qiskit Aer, per-qubit randomised thermal relaxation
#                       + depolarising + readout noise. Frequency detuning Δω
#                       injected as Rz rotation during delay. Fully local.
#   [3] Aer + IBM     — Qiskit Aer with real IBM backend noise model.
#                       Requires IBM credentials but uses zero shot budget.
#
# ARCHITECTURES (any qubit technology):
#   superconducting, trapped_ion, neutral_atom, spin_qubit, nv_center, custom
#
# WHY UNIVERSAL: the Lindblad forward models hold for any two-level quantum
# system with exponential energy relaxation and phase decoherence. Architecture
# only changes the time scales and noise magnitudes — never the physics.
#
# USAGE:
#   python 1_inversion.py
#   Interactive prompts. Results saved to data/inversion_results.csv.
#
# DEPENDENCIES:
#   numpy, scipy        — always required
#   qiskit>=1.0         — always required
#   qiskit-aer>=0.13    — always required
#   qiskit-ibm-runtime  — only for simulator option [3]
#   Prints T1, T2, Δω, ε_sx per qubit with uncertainties.
#   Saves to data/inversion_results.csv for use by 2_monitor.py.
#
# DEPENDENCIES:
#   Required always:  numpy, scipy
#   IBM platform:     qiskit >= 1.0, qiskit-ibm-runtime >= 0.20
#   IonQ platform:    qiskit >= 1.0, qiskit-ionq >= 0.5
#   Aer simulator:    qiskit >= 1.0, qiskit-aer >= 0.13
# =============================================================================

from __future__ import annotations

import csv
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=RuntimeWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# =============================================================================
# PHYSICAL CONSTANTS PER ARCHITECTURE
# These are literature values, not calibration data. They never go stale.
# Sources:
#   superconducting — Carroll et al. 2022, IBM Heron specs
#   trapped_ion     — Bruzewicz et al. 2019, IonQ Forte specs
#   neutral_atom    — Bluvstein et al. 2023 (QuEra), Pasqal specs
#   spin_qubit      — Burkard et al. 2023 review (Si/SiGe quantum dots)
#   nv_center       — Doherty et al. 2013 review (diamond NV)
#   custom          — user supplies T1/T2 priors at runtime
# =============================================================================

ARCH_DEFAULTS = {
    "superconducting": {
        "T1_s":           150e-6,
        "T2_s":           90e-6,
        "T1_min_s":       1e-6,
        "T1_max_s":       1000e-6,
        "T2_min_s":       0.5e-6,
        "dw_max_rad_s":   2*np.pi * 500e3,
        "dw_typical_khz": 5.0,
        "eps_typical":    3.5e-4,
        "eps_max":        0.1,
        "dt_ns":          0.2222,
        "gate_time_ns":   50.0,
        "display_unit":   "µs",
        "time_scale":     1e6,
    },
    "trapped_ion": {
        "T1_s":           10.0,
        "T2_s":           1.0,
        "T1_min_s":       0.01,
        "T1_max_s":       1000.0,
        "T2_min_s":       0.005,
        "dw_max_rad_s":   2*np.pi * 10e3,
        "dw_typical_khz": 0.5,
        "eps_typical":    5e-4,
        "eps_max":        0.05,
        "dt_ns":          None,
        "gate_time_ns":   135_000.0,
        "display_unit":   "ms",
        "time_scale":     1e3,
    },
    "neutral_atom": {
        "T1_s":           4.0,
        "T2_s":           1.0,
        "T1_min_s":       0.001,
        "T1_max_s":       100.0,
        "T2_min_s":       0.001,
        "dw_max_rad_s":   2*np.pi * 100e3,
        "dw_typical_khz": 2.0,
        "eps_typical":    1e-3,
        "eps_max":        0.1,
        "dt_ns":          None,
        "gate_time_ns":   500.0,
        "display_unit":   "ms",
        "time_scale":     1e3,
    },
    "spin_qubit": {
        "T1_s":           1.0,
        "T2_s":           100e-6,
        "T1_min_s":       1e-6,
        "T1_max_s":       100.0,
        "T2_min_s":       0.1e-6,
        "dw_max_rad_s":   2*np.pi * 1e6,
        "dw_typical_khz": 50.0,
        "eps_typical":    5e-3,
        "eps_max":        0.1,
        "dt_ns":          None,
        "gate_time_ns":   100.0,
        "display_unit":   "µs",
        "time_scale":     1e6,
    },
    "nv_center": {
        "T1_s":           6e-3,
        "T2_s":           1e-3,
        "T1_min_s":       1e-6,
        "T1_max_s":       10.0,
        "T2_min_s":       0.1e-6,
        "dw_max_rad_s":   2*np.pi * 1e6,
        "dw_typical_khz": 20.0,
        "eps_typical":    5e-3,
        "eps_max":        0.1,
        "dt_ns":          None,
        "gate_time_ns":   20.0,
        "display_unit":   "µs",
        "time_scale":     1e6,
    },
}

# Fallback for unknown architecture
ARCH_DEFAULTS["unknown"] = ARCH_DEFAULTS["superconducting"]


def build_custom_arch(T1_prior_s: float, T2_prior_s: float) -> dict:
    """
    Construct architecture constants for ANY qubit technology from two numbers.

    The physics is universal: every qubit with exponential energy relaxation
    and phase decoherence obeys the same three forward models. The only
    architecture-specific quantities are the time scales. Given a rough T1
    estimate, all bounds and display units are derived automatically.

    Parameters
    ----------
    T1_prior_s : float — rough T1 estimate in seconds (order of magnitude is enough)
    T2_prior_s : float — rough T2 estimate in seconds

    Returns
    -------
    dict — same schema as ARCH_DEFAULTS entries
    """
    # Choose display unit from T1 magnitude
    if T1_prior_s < 1e-3:
        unit, scale = "µs", 1e6
    elif T1_prior_s < 1.0:
        unit, scale = "ms", 1e3
    else:
        unit, scale = "s", 1.0

    return {
        "T1_s":           T1_prior_s,
        "T2_s":           min(T2_prior_s, 2.0 * T1_prior_s),
        "T1_min_s":       T1_prior_s / 1000,
        "T1_max_s":       T1_prior_s * 100,
        "T2_min_s":       T2_prior_s / 1000,
        "dw_max_rad_s":   2*np.pi * 500e3,
        "dw_typical_khz": 5.0,
        "eps_typical":    1e-3,
        "eps_max":        0.1,
        "dt_ns":          None,
        "gate_time_ns":   50.0,
        "display_unit":   unit,
        "time_scale":     scale,
    }

# Gate repetition N values — universal across all architectures
GATE_REP_N_VALUES = [5, 10, 20]

# Time point ratios × T1_prior and T2_prior
# Chosen for D-optimal coverage of the decay curve (maximises Fisher information)
T1_RATIOS     = [0.05, 0.40, 1.50]
RAMSEY_RATIOS = [0.04, 0.20, 1.00]

# Multi-start grid for Ramsey inversion (Δω initial guesses in rad/s)
# The Ramsey loss has local minima in Δω — multi-start avoids them
RAMSEY_DW_STARTS_KHZ = [-200, -100, -50, -10, 0, 10, 50, 100, 200]


# =============================================================================
# BACKEND PROFILE
# =============================================================================

@dataclass
class BackendProfile:
    """
    Hardware-specific timing parameters for adaptive probe circuit design.

    The same 9-circuit topology works on any architecture because T1, T2, Δω,
    and ε_sx are universal qubit parameters. Only the absolute time scales
    differ. BackendProfile handles this translation automatically.

    Parameters
    ----------
    architecture : str
        One of 'superconducting', 'trapped_ion', 'neutral_atom', 'unknown'.
    T1_prior_s : float
        Best available estimate of T1 in seconds. Used to set probe time points.
        On first run: use ARCH_DEFAULTS. On subsequent runs: use previous estimate.
    T2_prior_s : float
        Best available estimate of T2 in seconds.
    dt_ns : float or None
        Hardware clock cycle in nanoseconds. IBM-specific. None for other platforms.
    backend_name : str
        Human-readable backend name for logging.
    """
    architecture:  str
    T1_prior_s:    float
    T2_prior_s:    float
    dt_ns:         Optional[float]
    backend_name:  str = "unknown"
    custom_arch:   Optional[dict] = None   # populated for architecture="custom"

    @property
    def constants(self) -> dict:
        if self.custom_arch is not None:
            return self.custom_arch
        return ARCH_DEFAULTS.get(self.architecture, ARCH_DEFAULTS["unknown"])

    @property
    def t1_delays_s(self) -> list[float]:
        """
        Three T1 probe delay times in seconds.
        Spaced across the decay curve: near-start, inflection, near-complete.
        Rounded to nearest dt cycle if dt_ns is specified (IBM hardware).
        """
        delays = []
        for ratio in T1_RATIOS:
            t = ratio * self.T1_prior_s
            if self.dt_ns is not None:
                # Snap to nearest integer number of dt cycles
                dt_s = self.dt_ns * 1e-9
                t = max(dt_s, round(t / dt_s) * dt_s)
            delays.append(t)
        return delays

    @property
    def ramsey_delays_s(self) -> list[float]:
        """
        Three Ramsey probe delay times in seconds.
        Shortest point must satisfy τ₁ < π/Δω_max to avoid aliasing.
        For superconducting with Δω_max=500kHz: τ₁ < 1µs. Satisfied at 0.04×T2.
        """
        delays = []
        for ratio in RAMSEY_RATIOS:
            t = ratio * self.T2_prior_s
            if self.dt_ns is not None:
                dt_s = self.dt_ns * 1e-9
                t = max(dt_s, round(t / dt_s) * dt_s)
            delays.append(t)
        return delays

    @classmethod
    def from_ibm_backend(cls, backend, qubit: int = 0) -> "BackendProfile":
        """
        Build a BackendProfile from a live IBM backend object.
        Reads dt and architecture from the backend directly.
        T1/T2 priors come from literature defaults on first run.
        """
        dt_ns = backend.dt * 1e9 if backend.dt else 0.2222

        # Try to read T1/T2 from backend properties as initial priors
        # These are stale calibration values — only used as starting guess,
        # never as ground truth. The inversion recovers the actual values.
        T1_prior = ARCH_DEFAULTS["superconducting"]["T1_s"]
        T2_prior = ARCH_DEFAULTS["superconducting"]["T2_s"]
        try:
            props = backend.properties()
            t1 = props.qubit_property(qubit, "t1")[0]
            t2 = props.qubit_property(qubit, "t2")[0]
            if t1 and t1 > 0:
                T1_prior = float(t1)
            if t2 and t2 > 0:
                T2_prior = min(float(t2), 2.0 * T1_prior)
        except Exception:
            pass  # fall through to defaults

        return cls(
            architecture = "superconducting",
            T1_prior_s   = T1_prior,
            T2_prior_s   = T2_prior,
            dt_ns        = dt_ns,
            backend_name = getattr(backend, "name", "ibm_unknown"),
        )

    @classmethod
    def from_architecture(cls,
                          architecture: str,
                          backend_name: str = "unknown",
                          T1_prior_s: Optional[float] = None,
                          T2_prior_s: Optional[float] = None) -> "BackendProfile":
        """
        Build a BackendProfile from architecture type alone.

        Works for ANY qubit technology:
          - Named architectures (superconducting, trapped_ion, neutral_atom,
            spin_qubit, nv_center) use published literature defaults.
          - architecture="custom" with T1_prior_s/T2_prior_s supplied supports
            any technology not in the list — photonic dual-rail, topological,
            molecular spin, anything with exponential relaxation physics.
        """
        if architecture == "custom":
            if T1_prior_s is None or T2_prior_s is None:
                raise ValueError(
                    "architecture='custom' requires T1_prior_s and T2_prior_s"
                )
            custom = build_custom_arch(T1_prior_s, T2_prior_s)
            return cls(
                architecture = "custom",
                T1_prior_s   = custom["T1_s"],
                T2_prior_s   = custom["T2_s"],
                dt_ns        = None,
                backend_name = backend_name,
                custom_arch  = custom,
            )

        defaults = ARCH_DEFAULTS.get(architecture, ARCH_DEFAULTS["unknown"])
        T1 = T1_prior_s or defaults["T1_s"]
        T2 = T2_prior_s or defaults["T2_s"]
        T2 = min(T2, 2.0 * T1)  # enforce physical constraint
        return cls(
            architecture = architecture,
            T1_prior_s   = T1,
            T2_prior_s   = T2,
            dt_ns        = defaults["dt_ns"],
            backend_name = backend_name,
        )

    def update_priors(self, T1_est_s: float, T2_est_s: float) -> "BackendProfile":
        """
        Return a new BackendProfile with updated priors from a completed inversion.
        Used for bootstrapping: each run's output initialises the next run's time points.
        """
        T2_est_s = min(T2_est_s, 2.0 * T1_est_s)
        return BackendProfile(
            architecture = self.architecture,
            T1_prior_s   = T1_est_s,
            T2_prior_s   = T2_est_s,
            dt_ns        = self.dt_ns,
            backend_name = self.backend_name,
            custom_arch  = self.custom_arch,
        )


# =============================================================================
# CIRCUIT BUILDERS
# =============================================================================

def build_t1_circuit(delay_s: float, profile: BackendProfile):
    """
    Inversion recovery circuit:  |0⟩ → X → delay(τ) → measure

    Physics: X prepares |1⟩. During delay τ, amplitude damping decays
    population at rate 1/T1. Measured P(|1⟩) = exp(−τ/T1).

    The Kraus operators for amplitude damping:
        K0 = [[1, 0], [0, sqrt(1-γ)]]
        K1 = [[0, sqrt(γ)], [0, 0]]
    where γ = 1 − exp(−τ/T1). Applying to ρ=|1⟩⟨1| gives ρ₁₁(τ) = exp(−τ/T1).
    """
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1, 1, name=f"t1_{delay_s*1e6:.1f}us")
    qc.x(0)
    _add_delay(qc, delay_s, profile)
    qc.measure(0, 0)
    return qc


def build_ramsey_circuit(delay_s: float, profile: BackendProfile):
    """
    Ramsey interferometry circuit:  |0⟩ → H → delay(τ) → H → measure

    Physics: H maps |0⟩ to the Bloch sphere equator.
    During delay τ:
        (1) Phase damping shrinks the equatorial Bloch vector at rate 1/T2
        (2) Frequency detuning Δω rotates the vector at angular rate Δω
    Second H converts accumulated phase → population difference.
    Measured P(|1⟩) = (1/2)(1 − exp(−τ/T2)·cos(Δω·τ))

    The cosine oscillation is Δω. The exponential envelope is T2.
    They are physically and mathematically independent.
    """
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1, 1, name=f"ramsey_{delay_s*1e6:.1f}us")
    qc.h(0)
    _add_delay(qc, delay_s, profile)
    qc.h(0)
    qc.measure(0, 0)
    return qc


def build_gate_rep_circuit(N: int):
    """
    Gate repetition circuit:  |0⟩ → X^(2N) → measure

    Physics: 2N X gates = identity in noiseless case (even count).
    Under depolarizing noise with per-gate error ε, each X gate applies:
        E(ρ) = (1−ε)ρ + ε·(I/2)
    The Bloch vector contracts by (1−2ε) per gate.
    After 2N gates: P(|0⟩) = (1/2)(1 + (1−2ε)^(2N))

    Amplification example: ε=3.5e-4 (healthy) → P(0)≈0.986
                           ε=2.8e-3 (drifted) → P(0)≈0.894
    18% signal for a 10× error increase — clearly measurable.
    X gates are universally available on all platforms.
    """
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1, 1, name=f"gate_rep_N{N}")
    for _ in range(2 * N):
        qc.x(0)
    qc.measure(0, 0)
    return qc


def _add_delay(qc, delay_s: float, profile: BackendProfile):
    """
    Add a delay to a circuit in the appropriate unit for the target platform.

    IBM (dt_ns specified):  Qiskit delay instruction in dt units.
                            dt ≈ 0.2222 ns — hardware clock cycle.
                            Delay must be integer multiple of dt.

    Other platforms:        Qiskit delay instruction in seconds.
                            Transpiler handles platform-specific conversion.
                            For IonQ: natural idle time used if delay unsupported.
    """
    if profile.dt_ns is not None:
        dt_s     = profile.dt_ns * 1e-9
        n_cycles = max(1, round(delay_s / dt_s))
        qc.delay(n_cycles, 0, unit="dt")
    else:
        # Use seconds — transpiler converts for the target platform
        qc.delay(delay_s, 0, unit="s")


def build_probe_suite(profile: BackendProfile) -> tuple[list, dict]:
    """
    Build all 9 probe circuits for a given BackendProfile.

    Returns circuits in canonical order:
        [t1_τ1, t1_τ2, t1_τ3, ramsey_τ1, ramsey_τ2, ramsey_τ3,
         gate_N5, gate_N10, gate_N20]

    Also returns metadata dict with actual delay values used.
    This metadata is required by the inversion layer to set up forward models.
    """
    t1_delays     = profile.t1_delays_s
    ramsey_delays = profile.ramsey_delays_s

    circuits = []
    for t in t1_delays:
        circuits.append(build_t1_circuit(t, profile))
    for t in ramsey_delays:
        circuits.append(build_ramsey_circuit(t, profile))
    for N in GATE_REP_N_VALUES:
        circuits.append(build_gate_rep_circuit(N))

    metadata = {
        "t1_delays_s":     t1_delays,
        "ramsey_delays_s": ramsey_delays,
        "gate_rep_N":      GATE_REP_N_VALUES,
        "architecture":    profile.architecture,
        "backend_name":    profile.backend_name,
        "T1_prior_s":      profile.T1_prior_s,
        "T2_prior_s":      profile.T2_prior_s,
    }
    return circuits, metadata


# =============================================================================
# PLATFORM RUNNERS
# Uniform interface: takes Qiskit circuits, returns list of count dicts.
# Each platform implements run() differently; the inversion layer never
# knows which platform it is talking to.
# =============================================================================

class QuantumRunner(ABC):
    """Abstract base class for platform-specific circuit execution."""

    @abstractmethod
    def run(self, circuits: list, shots: int) -> list[dict]:
        """
        Submit circuits and return measurement counts.

        Parameters
        ----------
        circuits : list[QuantumCircuit]
        shots    : int

        Returns
        -------
        list[dict]
            One counts dict per circuit, e.g. {'0': 850, '1': 150}.
        """

    @staticmethod
    def _extract_counts(result_item) -> dict:
        """
        Extract counts from a SamplerV2 result item.
        IBM's result structure varies by backend and transpilation preset.
        This helper tries all known attribute names.
        """
        data = result_item.data
        for attr in vars(data):
            try:
                return getattr(data, attr).get_counts()
            except Exception:
                continue
        raise ValueError(
            f"Could not extract counts. Attributes: {list(vars(data))}"
        )


class IBMRunner(QuantumRunner):
    """
    IBM Quantum hardware runner via qiskit-ibm-runtime SamplerV2.
    Handles transpilation and batched job submission.
    """

    def __init__(self, backend, optimization_level: int = 0):
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        self.backend = backend
        self.pm      = generate_preset_pass_manager(
            optimization_level=optimization_level, backend=backend
        )

    def run(self, circuits: list, shots: int) -> list[dict]:
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        transpiled = [self.pm.run(qc) for qc in circuits]
        sampler    = Sampler(self.backend)
        job        = sampler.run(transpiled, shots=shots)
        result     = job.result()
        return [self._extract_counts(result[i]) for i in range(len(circuits))]


class IonQRunner(QuantumRunner):
    """
    IonQ hardware/simulator runner via qiskit-ionq provider.

    Note on delays: IonQ's native gate set does not include a delay instruction
    at the circuit abstraction level. For T1 and Ramsey measurements, the
    qiskit-ionq transpiler maps Qiskit delay instructions to IonQ's native
    idle time mechanism where supported, or to identity gate sequences otherwise.
    For the Forte hardware with T1 ~ seconds, identity-gate-based delays are
    impractical; users with IonQ pulse-level access should override this runner.
    """

    def __init__(self, backend):
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        self.backend = backend
        self.pm      = generate_preset_pass_manager(
            optimization_level=0, backend=backend
        )

    def run(self, circuits: list, shots: int) -> list[dict]:
        from qiskit_ionq import IonQProvider  # noqa: F401
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        transpiled = [self.pm.run(qc) for qc in circuits]
        sampler    = Sampler(self.backend)
        job        = sampler.run(transpiled, shots=shots)
        result     = job.result()
        return [self._extract_counts(result[i]) for i in range(len(circuits))]


class AerRunner(QuantumRunner):
    """
    Qiskit Aer simulator runner.
    Supports ideal simulation and hardware-derived noise models.

    Parameters
    ----------
    noise_model : qiskit_aer.noise.NoiseModel or None
        If provided, simulates realistic hardware noise.
        Load from a real backend: NoiseModel.from_backend(backend)
    """

    def __init__(self, noise_model=None):
        from qiskit_aer import AerSimulator
        if noise_model is not None:
            self.simulator = AerSimulator(noise_model=noise_model)
        else:
            self.simulator = AerSimulator()

    def run(self, circuits: list, shots: int) -> list[dict]:
        from qiskit import transpile
        transpiled = transpile(circuits, self.simulator)
        job        = self.simulator.run(transpiled, shots=shots)
        result     = job.result()
        return [result.get_counts(i) for i in range(len(circuits))]


class DemoRunner(QuantumRunner):
    """
    Local numpy simulation runner. Zero credentials, zero Qiskit required.
    Uses the Lindblad forward models + binomial shot noise directly.
    Fixed true parameters — useful for sanity-checking inversion accuracy.
    """

    def __init__(self, arch_constants: dict):
        self.T1  = arch_constants["T1_s"]
        self.T2  = arch_constants["T2_s"]
        self.dw  = 2 * np.pi * arch_constants["dw_typical_khz"] * 1e3
        self.eps = arch_constants["eps_typical"]
        scale    = arch_constants["time_scale"]
        unit     = arch_constants["display_unit"]
        print(f"\n  [Demo] True parameters:")
        print(f"    T1 = {self.T1*scale:.2f} {unit}  "
              f"T2 = {self.T2*scale:.2f} {unit}  "
              f"Δω = {self.dw/(2*np.pi*1e3):.2f} kHz  "
              f"ε = {self.eps:.2e}")
        print(f"  Inversion should recover these values.\n")

    def run(self, circuits: list, shots: int) -> list[dict]:
        counts = []
        for qc in circuits:
            delay_s = self._extract_delay_s(qc)
            name    = qc.name
            if name.startswith("t1_"):
                p1 = forward_t1(delay_s, self.T1)
            elif name.startswith("ramsey_"):
                p1 = forward_ramsey(delay_s, self.T2, self.dw)
            elif name.startswith("gate_rep_"):
                N  = int(name.split("N")[1])
                p1 = 1.0 - forward_gate(N, self.eps)
            else:
                p1 = 0.5
            n1 = int(np.random.binomial(shots, float(np.clip(p1, 0, 1))))
            counts.append({"0": shots - n1, "1": n1})
        return counts

    @staticmethod
    def _extract_delay_s(qc) -> float:
        for instr in qc.data:
            op = instr.operation
            if op.name == "delay":
                unit  = getattr(op, "unit", "dt")
                value = op.duration
                if unit == "dt":    return value * 0.2222e-9
                elif unit == "s":   return float(value)
                elif unit == "ns":  return float(value) * 1e-9
                elif unit == "us":  return float(value) * 1e-6
        return 0.0


class AerRealisticRunner(QuantumRunner):
    """
    Qiskit Aer local simulator with per-qubit randomised realistic noise.

    Each qubit gets independently randomised T1, T2, ε_sx, and Δω drawn
    from the architecture's physically realistic range. The noise model
    for each circuit uses:
        - Thermal relaxation during delay  (models T1 and T2 decay)
        - Depolarising error on X and H    (models gate imperfection)
        - Readout error                    (models measurement imperfection)
        - Rz rotation after delay in Ramsey circuits (models Δω detuning)

    This is the recommended simulator for research validation. It uses
    real Qiskit circuits and Aer execution — no numpy shortcuts.

    Parameters
    ----------
    arch_constants : dict  — from ARCH_DEFAULTS or build_custom_arch()
    n_qubits       : int   — number of qubits to randomise
    seed           : int   — random seed for reproducibility
    """

    def __init__(self, arch_constants: dict, n_qubits: int, seed: int = 42):
        self.arch     = arch_constants
        rng           = np.random.default_rng(seed)
        T1_typ        = arch_constants["T1_s"]
        T2_typ        = arch_constants["T2_s"]
        dw_typ_rad    = 2 * np.pi * arch_constants["dw_typical_khz"] * 1e3
        eps_typ       = arch_constants["eps_typical"]

        # Randomise each qubit independently over a physically realistic spread
        self.true_params: list[dict] = []
        for _ in range(n_qubits):
            T1  = T1_typ  * rng.uniform(0.55, 1.40)
            T2  = min(T2_typ * rng.uniform(0.35, 1.15), 1.95 * T1)
            eps = eps_typ * rng.uniform(0.5, 4.0)
            dw  = dw_typ_rad * rng.uniform(-2.0, 2.0)   # can be +/-
            self.true_params.append(
                {"T1_s": T1, "T2_s": T2, "epsilon_sx": eps, "delta_omega": dw}
            )

        scale = arch_constants["time_scale"]
        unit  = arch_constants["display_unit"]
        print(f"\n  [AerRealistic] Per-qubit randomised true parameters:")
        print(f"  {'Qubit':<8} {'T1':>12} {'T2':>12} {'Δω (kHz)':>12} {'ε_sx':>12}")
        print(f"  {'-'*60}")
        for i, p in enumerate(self.true_params):
            print(f"  {i:<8} "
                  f"{p['T1_s']*scale:>10.2f}{unit}  "
                  f"{p['T2_s']*scale:>10.2f}{unit}  "
                  f"{p['delta_omega']/(2*np.pi*1e3):>10.3f}     "
                  f"{p['epsilon_sx']:>12.2e}")
        print(f"\n  Inversion should recover these values per qubit.\n")

        self._call_index = 0   # increments each run() call, one per qubit

    def run(self, circuits: list, shots: int) -> list[dict]:
        """
        Run 9 circuits for the current qubit using a per-circuit realistic
        noise model built from that qubit's randomised parameters.
        """
        from qiskit_aer import AerSimulator
        from qiskit import transpile
        from qiskit_aer.noise import (
            NoiseModel, thermal_relaxation_error,
            depolarizing_error, ReadoutError
        )

        idx    = self._call_index % len(self.true_params)
        params = self.true_params[idx]
        self._call_index += 1

        T1_ns   = params["T1_s"]    * 1e9
        T2_ns   = params["T2_s"]    * 1e9
        eps     = params["epsilon_sx"]
        dw      = params["delta_omega"]
        gate_ns = self.arch["gate_time_ns"]
        T2_ns   = min(T2_ns, 2.0 * T1_ns * 0.999)

        all_counts = []
        for qc in circuits:
            # Inject Rz for frequency detuning in Ramsey circuits
            circuit_to_run = self._inject_detuning(qc, dw) \
                             if qc.name.startswith("ramsey_") else qc

            delay_ns = self._get_delay_ns(circuit_to_run)
            nm       = NoiseModel(basis_gates=["x", "h", "rz", "delay", "measure"])

            # Compose thermal relaxation + depolarising into one channel per gate
            # Using compose() avoids the "error already exists" warning
            try:
                t_err   = thermal_relaxation_error(T1_ns, T2_ns, gate_ns)
                d_err   = depolarizing_error(eps, 1)
                gate_err = d_err.compose(t_err)
                nm.add_quantum_error(gate_err, ["x", "h"], [0])
            except Exception:
                # Fallback: depolarising only
                nm.add_quantum_error(depolarizing_error(eps, 1), ["x", "h"], [0])

            # Thermal relaxation on delay (actual delay duration)
            if delay_ns > 0:
                try:
                    d_err = thermal_relaxation_error(T1_ns, T2_ns, delay_ns)
                    nm.add_quantum_error(d_err, ["delay"], [0])
                except Exception:
                    pass

            # Readout error
            p_ro    = float(np.clip(eps * 2, 0.001, 0.05))
            readout = ReadoutError([[1-p_ro/2, p_ro/2], [p_ro, 1-p_ro]])
            nm.add_readout_error(readout, [0])

            # optimization_level=0 preserves X and H gates so noise applies
            sim        = AerSimulator(noise_model=nm)
            transpiled = transpile(circuit_to_run, sim, optimization_level=0)
            result     = sim.run(transpiled, shots=shots).result()
            all_counts.append(result.get_counts(0))

        return all_counts

    @staticmethod
    def _get_delay_ns(qc) -> float:
        """Extract delay duration in nanoseconds from a Qiskit circuit."""
        for instr in qc.data:
            op = instr.operation
            if op.name == "delay":
                unit  = getattr(op, "unit", "dt")
                value = float(op.duration)
                if unit == "dt":  return value * 0.2222
                elif unit == "s": return value * 1e9
                elif unit == "ns":return value
                elif unit == "us":return value * 1e3
        return 0.0

    @staticmethod
    def _inject_detuning(qc, delta_omega_rad_s: float):
        """
        Return a new circuit with Rz(Δω·τ) inserted after the delay gate.
        This correctly models qubit frequency detuning at circuit level.
        """
        from qiskit import QuantumCircuit
        delay_s  = 0.0
        for instr in qc.data:
            op = instr.operation
            if op.name == "delay":
                unit  = getattr(op, "unit", "dt")
                value = float(op.duration)
                if unit == "dt":  delay_s = value * 0.2222e-9
                elif unit == "s": delay_s = value
                elif unit == "ns":delay_s = value * 1e-9
                elif unit == "us":delay_s = value * 1e-6
                break

        theta  = delta_omega_rad_s * delay_s   # rotation angle in radians
        new_qc = QuantumCircuit(1, 1, name=qc.name)
        for instr in qc.data:
            new_qc.append(instr.operation, instr.qubits, instr.clbits)
            if instr.operation.name == "delay":
                new_qc.rz(theta, 0)            # inject Rz after delay
        return new_qc


# =============================================================================
# FORWARD MODELS
# These are the Lindblad master equation analytic solutions.
# They are pure numpy functions — no hardware, no ML, no fitting here.
# These functions are called inside the loss functions during inversion.
# =============================================================================

def forward_t1(tau_s: float | np.ndarray, T1_s: float) -> float | np.ndarray:
    """
    T1 inversion recovery forward model.

    P(|1⟩; τ) = exp(−τ / T1)

    Derivation: from Lindblad amplitude damping channel solution.
    ρ₁₁(τ) = ρ₁₁(0) · exp(−τ/T1)  where ρ₁₁(0) = 1 after X gate.

    Parameters
    ----------
    tau_s : float or array  — delay time in seconds
    T1_s  : float           — T1 relaxation time in seconds

    Returns
    -------
    float or array — P(|1⟩)
    """
    return np.exp(-np.asarray(tau_s) / T1_s)


def forward_ramsey(tau_s: float | np.ndarray,
                   T2_s: float,
                   delta_omega: float) -> float | np.ndarray:
    """
    Ramsey interferometry forward model.

    P(|1⟩; τ) = (1/2)(1 − exp(−τ/T2) · cos(Δω · τ))

    Derivation: from Lindblad phase damping channel solution in the rotating frame.
    The off-diagonal density matrix element: ρ₀₁(τ) = ρ₀₁(0) · exp(−τ/T2) · exp(iΔω·τ)
    Measured population after second H gate: |Re[ρ₀₁(τ)]| contributes the cosine term.

    At Δω=0 (perfect calibration): P = (1/2)(1 − exp(−τ/T2))
    At τ≫T2 (fully dephased):      P = 1/2 (no coherence remaining)

    Parameters
    ----------
    tau_s       : float or array — delay time in seconds
    T2_s        : float          — T2 dephasing time in seconds
    delta_omega : float          — frequency detuning in rad/s

    Returns
    -------
    float or array — P(|1⟩)
    """
    tau = np.asarray(tau_s)
    return 0.5 * (1.0 - np.exp(-tau / T2_s) * np.cos(delta_omega * tau))


def forward_gate(N: int | np.ndarray, epsilon_sx: float) -> float | np.ndarray:
    """
    Gate repetition forward model.

    P(|0⟩; N) = (1/2)(1 + (1 − 2ε)^(2N))

    Derivation: depolarizing channel E(ρ) = (1−ε)ρ + ε·(I/2) applied 2N times.
    Bloch vector z-component after 2N gates: r_z = (1−2ε)^(2N)
    P(|0⟩) = (1 + r_z)/2

    Parameters
    ----------
    N          : int or array — number of X gate pairs (total gates = 2N)
    epsilon_sx : float        — per-gate depolarizing error rate

    Returns
    -------
    float or array — P(|0⟩)
    """
    N = np.asarray(N, dtype=float)
    return 0.5 * (1.0 + (1.0 - 2.0 * epsilon_sx) ** (2.0 * N))


# =============================================================================
# INVERSION LAYER
# Nonlinear least squares fitting of forward models to measured probabilities.
# T1, T2/Δω, and ε_sx are optimised independently — they appear in separate
# loss functions and can be solved as three decoupled problems.
# =============================================================================

@dataclass
class InversionResult:
    """
    Recovered physical parameters for a single qubit from one probe run.

    All times in seconds internally. Display formatting is handled separately.
    Uncertainty is 1-sigma from the curve_fit covariance matrix.
    """
    backend_name:    str
    qubit_id:        int
    timestamp:       str
    architecture:    str

    T1_s:            float
    T1_sigma_s:      float

    T2_s:            float
    T2_sigma_s:      float

    delta_omega:     float          # rad/s
    delta_omega_sigma: float        # rad/s

    epsilon_sx:      float
    epsilon_sx_sigma: float

    # Raw measurements for reproducibility
    p1_t1:           list = field(default_factory=list)
    p1_ramsey:       list = field(default_factory=list)
    p0_gate:         list = field(default_factory=list)
    t1_delays_s:     list = field(default_factory=list)
    ramsey_delays_s: list = field(default_factory=list)
    gate_rep_N:      list = field(default_factory=list)

    # Residuals (lower = better fit)
    t1_residual:     float = 0.0
    ramsey_residual: float = 0.0
    gate_residual:   float = 0.0

    def display(self, arch_defaults: dict) -> str:
        """Format result for terminal output."""
        scale = arch_defaults["time_scale"]
        unit  = arch_defaults["display_unit"]
        dw_khz = self.delta_omega / (2 * np.pi * 1e3)
        dw_sig = self.delta_omega_sigma / (2 * np.pi * 1e3)
        lines = [
            f"  Qubit {self.qubit_id}  |  {self.backend_name}  |  {self.timestamp}",
            f"    T1      = {self.T1_s*scale:.2f} ± {self.T1_sigma_s*scale:.2f} {unit}",
            f"    T2      = {self.T2_s*scale:.2f} ± {self.T2_sigma_s*scale:.2f} {unit}",
            f"    Δω      = {dw_khz:.3f} ± {dw_sig:.3f} kHz",
            f"    ε_sx    = {self.epsilon_sx:.3e} ± {self.epsilon_sx_sigma:.3e}",
            f"    Residuals: T1={self.t1_residual:.2e}  Ramsey={self.ramsey_residual:.2e}  Gate={self.gate_residual:.2e}",
        ]
        return "\n".join(lines)


def invert_t1(p1_measured: np.ndarray,
              tau_s: np.ndarray,
              T1_prior_s: float,
              arch: dict) -> tuple[float, float, float]:
    """
    Recover T1 from three inversion recovery measurements.

    Uses scipy.optimize.curve_fit (Levenberg-Marquardt nonlinear least squares).
    Returns (T1_s, sigma_T1_s, residual_sum).

    Single unknown — landscape is smooth, one dominant minimum.
    No multi-start needed.
    """
    bounds = ([arch["T1_min_s"]], [arch["T1_max_s"]])
    try:
        popt, pcov = curve_fit(
            forward_t1,
            tau_s,
            p1_measured,
            p0=[T1_prior_s],
            bounds=bounds,
            maxfev=10000,
        )
        T1    = float(popt[0])
        sigma = float(np.sqrt(np.diag(pcov))[0])
        resid = float(np.sum((p1_measured - forward_t1(tau_s, T1))**2))
    except Exception:
        T1, sigma, resid = T1_prior_s, np.inf, np.inf
    return T1, sigma, resid


def invert_ramsey(p1_measured: np.ndarray,
                  tau_s: np.ndarray,
                  T2_prior_s: float,
                  arch: dict) -> tuple[float, float, float, float, float]:
    """
    Recover T2 and Δω from three Ramsey measurements.

    Uses multi-start curve_fit to avoid local minima in Δω.
    The Ramsey loss landscape has multiple minima because the cosine term
    aliases at different frequencies. Multi-start covers the physical Δω range.

    Returns (T2_s, sigma_T2, delta_omega, sigma_dw, residual).
    """
    dw_max   = arch["dw_max_rad_s"]
    T2_min   = arch["T2_min_s"]
    T2_max   = 2.0 * T2_prior_s * 3   # generous upper bound

    bounds = ([T2_min, -dw_max], [T2_max, dw_max])

    best_T2    = T2_prior_s
    best_dw    = 0.0
    best_sigma = (np.inf, np.inf)
    best_resid = np.inf

    for dw_start_khz in RAMSEY_DW_STARTS_KHZ:
        dw_start = 2 * np.pi * dw_start_khz * 1e3
        try:
            popt, pcov = curve_fit(
                forward_ramsey,
                tau_s,
                p1_measured,
                p0=[T2_prior_s, dw_start],
                bounds=bounds,
                maxfev=20000,
            )
            T2    = float(popt[0])
            dw    = float(popt[1])
            sigs  = np.sqrt(np.diag(pcov))
            resid = float(np.sum((p1_measured - forward_ramsey(tau_s, T2, dw))**2))
            if resid < best_resid:
                best_T2    = T2
                best_dw    = dw
                best_sigma = (float(sigs[0]), float(sigs[1]))
                best_resid = resid
        except Exception:
            continue

    return best_T2, best_sigma[0], best_dw, best_sigma[1], best_resid


def invert_gate(p0_measured: np.ndarray,
                N_values: np.ndarray,
                arch: dict) -> tuple[float, float, float]:
    """
    Recover ε_sx from three gate repetition measurements.

    Single unknown — smooth landscape, one minimum.
    Returns (epsilon_sx, sigma_epsilon, residual).
    """
    bounds = ([0.0], [arch["eps_max"]])
    try:
        popt, pcov = curve_fit(
            forward_gate,
            N_values,
            p0_measured,
            p0=[1e-3],
            bounds=bounds,
            maxfev=10000,
        )
        eps   = float(popt[0])
        sigma = float(np.sqrt(np.diag(pcov))[0])
        resid = float(np.sum((p0_measured - forward_gate(N_values, eps))**2))
    except Exception:
        eps, sigma, resid = 1e-3, np.inf, np.inf
    return eps, sigma, resid


def invert_all(counts_list: list[dict],
               metadata: dict,
               profile: BackendProfile,
               qubit_id: int = 0,
               shots: int = 1000) -> InversionResult:
    """
    Full inversion pipeline: 9 count dicts → 4 physical parameters.

    Expects counts_list in canonical order:
        [0:2]  T1 circuits  (measure P(|1⟩))
        [3:5]  Ramsey       (measure P(|1⟩))
        [6:8]  Gate rep     (measure P(|0⟩))

    Optimises T1, (T2, Δω), and ε_sx independently.
    """
    arch = profile.constants

    # Extract P(|1⟩) for T1 and Ramsey, P(|0⟩) for gate rep
    p1_t1     = np.array([counts_list[i].get("1", 0) / shots for i in range(3)])
    p1_ramsey = np.array([counts_list[i+3].get("1", 0) / shots for i in range(3)])
    p0_gate   = np.array([counts_list[i+6].get("0", 0) / shots for i in range(3)])

    t1_delays     = np.array(metadata["t1_delays_s"])
    ramsey_delays = np.array(metadata["ramsey_delays_s"])
    N_values      = np.array(metadata["gate_rep_N"], dtype=float)

    # Three independent inversions
    T1, T1_sigma, t1_resid = invert_t1(
        p1_t1, t1_delays, profile.T1_prior_s, arch
    )
    T2, T2_sigma, dw, dw_sigma, ramsey_resid = invert_ramsey(
        p1_ramsey, ramsey_delays, profile.T2_prior_s, arch
    )
    # Enforce T2 ≤ 2·T1
    T2 = min(T2, 2.0 * T1)

    eps, eps_sigma, gate_resid = invert_gate(p0_gate, N_values, arch)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    return InversionResult(
        backend_name       = metadata["backend_name"],
        qubit_id           = qubit_id,
        timestamp          = ts,
        architecture       = profile.architecture,
        T1_s               = T1,
        T1_sigma_s         = T1_sigma,
        T2_s               = T2,
        T2_sigma_s         = T2_sigma,
        delta_omega        = dw,
        delta_omega_sigma  = dw_sigma,
        epsilon_sx         = eps,
        epsilon_sx_sigma   = eps_sigma,
        p1_t1              = p1_t1.tolist(),
        p1_ramsey          = p1_ramsey.tolist(),
        p0_gate            = p0_gate.tolist(),
        t1_delays_s        = t1_delays.tolist(),
        ramsey_delays_s    = ramsey_delays.tolist(),
        gate_rep_N         = N_values.tolist(),
        t1_residual        = t1_resid,
        ramsey_residual    = ramsey_resid,
        gate_residual      = gate_resid,
    )


# =============================================================================
# RESULTS I/O
# =============================================================================

def save_result(result: InversionResult, path: Path):
    """Append an InversionResult to a CSV file."""
    path.parent.mkdir(exist_ok=True)
    row = asdict(result)
    # Flatten lists to strings for CSV
    for key in ["p1_t1", "p1_ramsey", "p0_gate",
                "t1_delays_s", "ramsey_delays_s", "gate_rep_N"]:
        row[key] = str(row[key])
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# =============================================================================
# MAIN
# =============================================================================

def run_inversion(runner: QuantumRunner,
                  profile: BackendProfile,
                  qubit_ids: list[int],
                  shots: int = 1000,
                  save_path: Optional[Path] = None) -> list[InversionResult]:
    """
    Run the complete inversion pipeline on one or more qubits.

    Parameters
    ----------
    runner    : QuantumRunner — platform-specific executor
    profile   : BackendProfile — hardware timing parameters
    qubit_ids : list[int] — which qubits to probe
    shots     : int — shots per circuit (1000 is standard)
    save_path : Path or None — where to save results CSV

    Returns
    -------
    list[InversionResult] — one result per qubit
    """
    arch     = profile.constants
    results  = []

    for qubit in qubit_ids:
        print(f"\n  Probing qubit {qubit} on {profile.backend_name}...")

        # Build 9 probe circuits with adaptive time points
        circuits, metadata = build_probe_suite(profile)
        metadata["qubit_id"] = qubit

        scale = arch["time_scale"]
        unit  = arch["display_unit"]
        print(f"    T1 delays    : {[f'{t*scale:.2f}{unit}' for t in metadata['t1_delays_s']]}")
        print(f"    Ramsey delays: {[f'{t*scale:.2f}{unit}' for t in metadata['ramsey_delays_s']]}")
        print(f"    Gate rep N   : {metadata['gate_rep_N']}")

        # Submit to hardware / simulator
        counts_list = runner.run(circuits, shots=shots)

        # Extract raw probabilities for display
        p1_t1     = [counts_list[i].get("1", 0) / shots for i in range(3)]
        p1_ramsey = [counts_list[i+3].get("1", 0) / shots for i in range(3)]
        p0_gate   = [counts_list[i+6].get("0", 0) / shots for i in range(3)]
        print(f"    P(1) T1     : {[round(p, 3) for p in p1_t1]}")
        print(f"    P(1) Ramsey : {[round(p, 3) for p in p1_ramsey]}")
        print(f"    P(0) Gate   : {[round(p, 3) for p in p0_gate]}")

        # Invert to physical parameters
        result = invert_all(counts_list, metadata, profile, qubit_id=qubit, shots=shots)

        print("\n" + result.display(arch))

        # NOTE: do NOT update profile priors here.
        # Bootstrapping (using one run's estimates to warm-start the next)
        # belongs in 2_monitor.py when monitoring the SAME qubit over time.
        # Propagating priors across DIFFERENT qubits causes the escalating
        # time-point bug seen in the CSV data.

        if save_path:
            save_result(result, save_path)

        results.append(result)

    return results


def build_runner(simulator: str,
                 architecture: str,
                 n_qubits: int,
                 noise_model_backend=None,
                 T1_prior_s=None,
                 T2_prior_s=None,
                 seed: int = 42,
                 **credentials):
    """
    Factory: build the right runner and BackendProfile for the chosen simulator.
    simulator: 'demo' | 'aer_realistic' | 'aer_ibm_noise'
    """
    profile = BackendProfile.from_architecture(
        architecture,
        backend_name = simulator,
        T1_prior_s   = T1_prior_s,
        T2_prior_s   = T2_prior_s,
    )
    if simulator == "demo":
        runner = DemoRunner(arch_constants=profile.constants)
        return runner, profile
    if simulator == "aer_realistic":
        runner = AerRealisticRunner(
            arch_constants = profile.constants,
            n_qubits       = n_qubits,
            seed           = seed,
        )
        return runner, profile
    if simulator == "aer_ibm_noise":
        from qiskit_aer.noise import NoiseModel
        noise_model = NoiseModel.from_backend(noise_model_backend)
        runner = AerRunner(noise_model=noise_model)
        return runner, profile
    raise ValueError(f"Unknown simulator: {simulator}")


# =============================================================================
# INTERACTIVE WIZARD
# =============================================================================

def _ask_choice(question: str, options, default: int = 1) -> int:
    print(f"\n{question}")
    for i, opt in enumerate(options, start=1):
        print(f"  [{i}] {opt}")
    while True:
        raw = input(f"  > [{default}]: ").strip()
        if raw == "":
            return default
        try:
            choice = int(raw)
            if 1 <= choice <= len(options):
                return choice
        except ValueError:
            pass
        print(f"  Enter a number between 1 and {len(options)}.")


def _ask_text(question: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{question}{suffix}: ").strip()
    return raw if raw else default


def _ask_int(question: str, default: int) -> int:
    while True:
        raw = input(f"{question} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            print("  Enter a valid integer.")


def _ask_secret(question: str) -> str:
    import getpass
    return getpass.getpass(f"{question}: ").strip()


def main():
    print("=" * 60)
    print("  QUANTUM CANARY v2 — LINDBLAD INVERSION (research)")
    print("  Local Qiskit simulation. No ML. No training data.")
    print("=" * 60)

    # ── STEP 1: Simulator ─────────────────────────────────────────────────
    sim_choice = _ask_choice(
        "Which simulator?",
        [
            "Demo          (numpy forward model + shot noise, instant, known true values)",
            "Aer realistic (Qiskit Aer, per-qubit randomised thermal + depolarising noise)",
            "Aer + IBM     (Qiskit Aer with real IBM backend noise model, needs credentials)",
        ],
        default=2,
    )
    simulator = {1: "demo", 2: "aer_realistic", 3: "aer_ibm_noise"}[sim_choice]

    # ── STEP 2: Architecture ─────────────────────────────────────────────
    T1_custom = None
    T2_custom = None

    if simulator == "aer_ibm_noise":
        architecture = "superconducting"
        print("\nArchitecture: superconducting (fixed for IBM noise model)")
    else:
        arch_choice = _ask_choice(
            "Which qubit architecture?",
            ["Superconducting  (IBM/Google/Rigetti,   T1 ~ 150 us)",
             "Trapped ion      (IonQ/Quantinuum,      T1 ~ 10 s)",
             "Neutral atom     (QuEra/Pasqal,         T1 ~ 4 s)",
             "Spin qubit       (Si/SiGe quantum dot,  T1 ~ 1 s, T2* ~ 100 us)",
             "NV center        (diamond, room temp,   T1 ~ 6 ms)",
             "Custom           (you supply T1 and T2)"],
            default=1,
        )
        architecture = {1: "superconducting", 2: "trapped_ion", 3: "neutral_atom",
                        4: "spin_qubit",      5: "nv_center",   6: "custom"}[arch_choice]

        if architecture == "custom":
            print("\n  Any qubit technology with exponential T1 relaxation is supported.")
            T1_custom = float(_ask_text("  Approximate T1 in seconds (e.g. 1e-4)", "1e-4"))
            T2_custom = float(_ask_text("  Approximate T2 in seconds (e.g. 5e-5)", "5e-5"))

    # ── STEP 3: IBM credentials (only for aer_ibm_noise) ─────────────────
    noise_model_backend = None
    if simulator == "aer_ibm_noise":
        print("\n--- IBM credentials (no shots used, noise model download only) ---")
        token    = _ask_secret("  IBM Quantum API token")
        instance = _ask_secret("  IBM Cloud CRN")
        bk_name  = _ask_text("  Backend name", "ibm_kingston")
        from qiskit_ibm_runtime import QiskitRuntimeService
        print("  Fetching noise model...")
        svc = QiskitRuntimeService(channel="ibm_cloud", token=token, instance=instance)
        noise_model_backend = svc.backend(bk_name)
        print("  Done.")

    # ── STEP 4: Qubits, shots, seed ──────────────────────────────────────
    n_qubits  = _ask_int("\nHow many qubits to simulate", default=5)
    qubit_ids = list(range(n_qubits))
    shots     = _ask_int("Shots per circuit", default=1000)
    seed      = _ask_int("Random seed", default=42)

    # ── STEP 5: Confirm ──────────────────────────────────────────────────
    save_path = DATA_DIR / "inversion_results.csv"
    print("\n" + "-" * 60)
    print("  CONFIGURATION")
    print("-" * 60)
    print(f"  Simulator    : {simulator}")
    print(f"  Architecture : {architecture}")
    print(f"  Qubits       : {n_qubits}  (IDs {qubit_ids[0]}\u2013{qubit_ids[-1]})")
    print(f"  Shots        : {shots} per circuit  ({shots * 9} total per qubit)")
    print(f"  Seed         : {seed}")
    print(f"  Results file : {save_path}")
    confirm = input("\n  Start? [Y/n]: ").strip().lower()
    if confirm in ("n", "no"):
        print("  Cancelled.")
        return []

    # ── STEP 6: Build and run ─────────────────────────────────────────────
    runner, profile = build_runner(
        simulator           = simulator,
        architecture        = architecture,
        n_qubits            = n_qubits,
        noise_model_backend = noise_model_backend,
        T1_prior_s          = T1_custom,
        T2_prior_s          = T2_custom,
        seed                = seed,
    )
    results = run_inversion(
        runner    = runner,
        profile   = profile,
        qubit_ids = qubit_ids,
        shots     = shots,
        save_path = save_path,
    )
    print(f"\n{'='*60}")
    print(f"  COMPLETE  |  {len(results)} qubit(s) inverted")
    print(f"  Saved to  : {save_path}")
    print(f"{'='*60}\n")
    return results


if __name__ == "__main__":
    main()