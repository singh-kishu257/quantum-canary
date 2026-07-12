# 1_inversion.py — Quantum Canary v2 Research Pipeline
# =============================================================================
# Physics-based qubit parameter estimation via Lindblad master equation inversion.
#
# WHAT THIS SCRIPT DOES:
#   1. Connects to a quantum backend (IBM, IonQ, simulator, or any Qiskit provider)
#   2. Builds and submits 9 probe circuits:
#        3 x Inversion Recovery  → measures T1
#        3 x Ramsey              → measures T2 and Δω
#        3 x Gate Repetition     → measures ε_sx
#   3. Receives 9 measurement probabilities from hardware
#   4. Inverts the Lindblad forward models via nonlinear least squares
#   5. Returns T1, T2, Δω, ε_sx — the four qubit health parameters
#
# NO TRAINING DATA. NO MACHINE LEARNING. PURE PHYSICS.
#
# THE THREE FORWARD MODELS (Lindblad master equation analytic solutions):
#   T1 inversion recovery:  P(1; τ) = exp(−τ / T1)
#   Ramsey interferometry:  P(1; τ) = (1/2)(1 − exp(−τ/T2) · cos(Δω · τ))
#   Gate repetition:        P(0; N) = (1/2)(1 + (1 − 2ε)^(2N))
#
# SUPPORTED PLATFORMS:
#   ibm          — IBM Quantum via qiskit-ibm-runtime (SamplerV2)
#   ionq         — IonQ hardware via qiskit-ionq provider
#   aer          — Qiskit Aer simulator (ideal or with noise model)
#   ionq_sim     — IonQ cloud simulators (Forte noisy, Aria 1 noisy, ideal)
#   demo         — local numpy simulation, zero credentials required
#
# SUPPORTED ARCHITECTURES:
#   superconducting  — IBM Heron, Eagle, Falcon, etc. (T1 ~ 100-200 µs)
#   trapped_ion      — IonQ Forte, Aria, etc. (T1 ~ seconds)
#   neutral_atom     — generic interface, specify priors manually
#   unknown          — fallback to superconducting defaults
#
# USAGE:
#   python 1_inversion.py --demo                        # no credentials, local sim
#   python 1_inversion.py --platform aer               # Aer simulator
#   python 1_inversion.py --platform ibm               # IBM hardware
#   python 1_inversion.py --platform ionq              # IonQ hardware
#   python 1_inversion.py --platform ionq_sim          # IonQ noisy simulator
#
# OUTPUT:
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

import argparse
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
# Sources: Carroll et al. 2022 (superconducting), Bruzewicz et al. 2019 (trapped ion)
# =============================================================================

ARCH_DEFAULTS = {
    "superconducting": {
        "T1_s":          150e-6,      # 150 µs typical
        "T2_s":          90e-6,       # 90 µs typical
        "T1_min_s":      1e-6,        # 1 µs minimum physically meaningful
        "T1_max_s":      1000e-6,     # 1000 µs upper bound
        "T2_min_s":      0.5e-6,
        "dw_max_rad_s":  2*np.pi * 500e3,   # ±500 kHz max detuning
        "eps_max":       0.1,
        "dt_ns":         0.2222,      # IBM clock cycle in nanoseconds
        "display_unit":  "µs",
        "time_scale":    1e6,         # multiply seconds → display unit
    },
    "trapped_ion": {
        "T1_s":          10.0,        # ~10 seconds typical for IonQ Forte
        "T2_s":          1.0,         # ~1 second typical
        "T1_min_s":      0.01,
        "T1_max_s":      1000.0,
        "T2_min_s":      0.005,
        "dw_max_rad_s":  2*np.pi * 10e3,    # ±10 kHz (tighter — magnetically sensitive)
        "eps_max":       0.05,
        "dt_ns":         None,        # no clock cycle constraint at circuit level
        "display_unit":  "ms",
        "time_scale":    1e3,
    },
    "neutral_atom": {
        "T1_s":          1.0,
        "T2_s":          0.5,
        "T1_min_s":      0.001,
        "T1_max_s":      100.0,
        "T2_min_s":      0.001,
        "dw_max_rad_s":  2*np.pi * 100e3,
        "eps_max":       0.1,
        "dt_ns":         None,
        "display_unit":  "ms",
        "time_scale":    1e3,
    },
}

# Fallback for unknown architecture
ARCH_DEFAULTS["unknown"] = ARCH_DEFAULTS["superconducting"]

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

    @property
    def constants(self) -> dict:
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
        Uses published literature defaults if priors not specified.
        Appropriate for IonQ, neutral atom, and first-run scenarios.
        """
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
    Local numpy simulation runner. Zero credentials required.
    Simulates realistic noisy hardware using the forward model equations
    with added shot noise. Used for development, testing, and demonstration.

    Parameters
    ----------
    T1_s        : float  — true T1 (set by user to test recovery)
    T2_s        : float  — true T2
    delta_omega : float  — true Δω in rad/s
    epsilon_sx  : float  — true gate error
    architecture: str
    """

    def __init__(self,
                 T1_s:        float = 150e-6,
                 T2_s:        float = 90e-6,
                 delta_omega: float = 2*np.pi*5e3,
                 epsilon_sx:  float = 3.5e-4,
                 architecture: str  = "superconducting"):
        self.T1          = T1_s
        self.T2          = T2_s
        self.dw          = delta_omega
        self.eps         = epsilon_sx
        self.architecture = architecture
        print(f"\n  [DemoRunner] True parameters:")
        print(f"    T1          = {T1_s*1e6:.1f} µs")
        print(f"    T2          = {T2_s*1e6:.1f} µs")
        print(f"    Δω          = {delta_omega/(2*np.pi*1e3):.2f} kHz")
        print(f"    ε_sx        = {epsilon_sx:.2e}")
        print(f"  Inversion should recover these values.\n")

    def run(self, circuits: list, shots: int) -> list[dict]:
        """
        Simulate 9 circuits using forward models + binomial shot noise.
        Circuit identity is inferred from the circuit name.
        """
        counts_list = []
        for qc in circuits:
            name = qc.name
            p1 = self._simulate_circuit(qc, name)
            # Binomial shot noise: sqrt(p(1-p)/N) ~ 0.016 for p=0.5, N=1000
            n1 = np.random.binomial(shots, p1)
            n0 = shots - n1
            counts_list.append({"0": int(n0), "1": int(n1)})
        return counts_list

    def _simulate_circuit(self, qc, name: str) -> float:
        """Return P(|1⟩) for this circuit using the true forward model."""
        # Parse delay from circuit
        delay_s = self._extract_delay_s(qc)
        if name.startswith("t1_"):
            return forward_t1(delay_s, self.T1)
        elif name.startswith("ramsey_"):
            return forward_ramsey(delay_s, self.T2, self.dw)
        elif name.startswith("gate_rep_"):
            N  = int(name.split("N")[1])
            p0 = forward_gate(N, self.eps)
            return 1.0 - p0  # circuit returns P(|0⟩), we want P(|1⟩) for uniformity
        return 0.5

    @staticmethod
    def _extract_delay_s(qc) -> float:
        """Extract delay duration in seconds from a Qiskit circuit."""
        for instruction in qc.data:
            op = instruction.operation
            if op.name == "delay":
                unit  = getattr(op, "unit", "dt")
                value = op.duration
                if unit == "dt":
                    return value * 0.2222e-9
                elif unit == "s":
                    return float(value)
                elif unit == "ns":
                    return float(value) * 1e-9
                elif unit == "us":
                    return float(value) * 1e-6
        return 0.0


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
    arch = ARCH_DEFAULTS.get(profile.architecture, ARCH_DEFAULTS["unknown"])

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
    arch     = ARCH_DEFAULTS.get(profile.architecture, ARCH_DEFAULTS["unknown"])
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

        # Bootstrap: update profile priors for next qubit (warm start)
        profile = profile.update_priors(result.T1_s, result.T2_s)

        if save_path:
            save_result(result, save_path)

        results.append(result)

    return results


def build_runner(platform: str,
                 architecture: str,
                 backend_name: Optional[str] = None,
                 noise_model_backend=None,
                 **credentials) -> tuple[QuantumRunner, BackendProfile]:
    """
    Factory function — builds the appropriate runner and profile for a platform.

    Parameters
    ----------
    platform         : str — 'ibm', 'ionq', 'aer', 'ionq_sim', 'demo'
    architecture     : str — 'superconducting', 'trapped_ion', 'neutral_atom'
    backend_name     : str — backend identifier string
    noise_model_backend : backend object for Aer noise model (optional)
    **credentials    : token, instance, api_key etc. as needed by platform
    """

    if platform == "demo":
        runner  = DemoRunner(architecture=architecture)
        profile = BackendProfile.from_architecture(architecture, backend_name="demo")
        return runner, profile

    if platform == "ibm":
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService(
            channel  = "ibm_cloud",
            token    = credentials["token"],
            instance = credentials["instance"],
        )
        backend = service.backend(backend_name)
        runner  = IBMRunner(backend)
        profile = BackendProfile.from_ibm_backend(backend)
        return runner, profile

    if platform in ("ionq", "ionq_sim"):
        from qiskit_ionq import IonQProvider
        provider = IonQProvider(token=credentials["api_key"])
        backend  = provider.get_backend(backend_name)
        runner   = IonQRunner(backend)
        profile  = BackendProfile.from_architecture(
            "trapped_ion", backend_name=backend_name
        )
        return runner, profile

    if platform == "aer":
        noise_model = None
        if noise_model_backend is not None:
            from qiskit_aer.noise import NoiseModel
            noise_model = NoiseModel.from_backend(noise_model_backend)
        runner  = AerRunner(noise_model=noise_model)
        profile = BackendProfile.from_architecture(
            architecture, backend_name=backend_name or "aer_simulator"
        )
        return runner, profile

    raise ValueError(f"Unknown platform: {platform}. "
                     f"Choose from: ibm, ionq, ionq_sim, aer, demo")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantum Canary v2 — Lindblad parameter inversion",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--platform", default="demo",
        choices=["ibm", "ionq", "ionq_sim", "aer", "demo"],
        help=(
            "ibm       — IBM Quantum hardware\n"
            "ionq      — IonQ hardware\n"
            "ionq_sim  — IonQ cloud simulator\n"
            "aer       — Qiskit Aer local simulator\n"
            "demo      — local numpy demo (no credentials)"
        )
    )
    parser.add_argument(
        "--architecture", default="superconducting",
        choices=["superconducting", "trapped_ion", "neutral_atom", "unknown"],
        help="Qubit technology. Controls time point scaling and physical bounds."
    )
    parser.add_argument("--backend",  default=None, help="Backend name string.")
    parser.add_argument("--qubits",   default="0", help="Comma-separated qubit IDs, e.g. 0,1,2")
    parser.add_argument("--shots",    default=1000, type=int, help="Shots per circuit.")
    parser.add_argument("--no-save",  action="store_true", help="Do not save results to CSV.")
    return parser.parse_args()


def main():
    args     = parse_args()
    qubit_ids = [int(q.strip()) for q in args.qubits.split(",")]
    save_path = None if args.no_save else DATA_DIR / "inversion_results.csv"

    print("=" * 60)
    print("  QUANTUM CANARY v2 — LINDBLAD INVERSION")
    print("=" * 60)
    print(f"  Platform     : {args.platform}")
    print(f"  Architecture : {args.architecture}")
    print(f"  Backend      : {args.backend or 'auto'}")
    print(f"  Qubits       : {qubit_ids}")
    print(f"  Shots        : {args.shots}")
    print(f"  Save results : {save_path or 'disabled'}")

    credentials = {}

    if args.platform == "ibm":
        import getpass
        credentials["token"]    = getpass.getpass("\n  IBM Quantum API token: ")
        credentials["instance"] = getpass.getpass("  IBM Cloud CRN:         ")

    if args.platform in ("ionq", "ionq_sim"):
        import getpass
        credentials["api_key"] = getpass.getpass("\n  IonQ API key: ")

    runner, profile = build_runner(
        platform     = args.platform,
        architecture = args.architecture,
        backend_name = args.backend,
        **credentials,
    )

    results = run_inversion(
        runner    = runner,
        profile   = profile,
        qubit_ids = qubit_ids,
        shots     = args.shots,
        save_path = save_path,
    )

    print(f"\n{'='*60}")
    print(f"  COMPLETE  |  {len(results)} qubit(s) inverted")
    if save_path:
        print(f"  Saved to  : {save_path}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    main()



