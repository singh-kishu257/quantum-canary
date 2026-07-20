from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=RuntimeWarning)

__all__ = [
    "ARCH_DEFAULTS", "build_custom_arch", "BackendProfile",
    "InversionResult", "forward_t1", "forward_ramsey", "forward_gate",
    "build_probe_circuits", "lindblad_inversion",
]

ARCH_DEFAULTS: dict[str, dict] = {
    "superconducting": {
        "T1_s": 150e-6, "T2_s": 90e-6,
        "T1_min_s": 1e-6, "T1_max_s": 1000e-6, "T2_min_s": 0.5e-6,
        "dw_max_rad_s": 2 * np.pi * 500e3, "dw_typical_khz": 5.0,
        "eps_typical": 3.5e-4, "eps_max": 0.1,
        "dt_ns": 0.2222, "gate_time_ns": 50.0,
        "display_unit": "µs", "time_scale": 1e6,
    },
    "trapped_ion": {
        "T1_s": 10.0, "T2_s": 1.0,
        "T1_min_s": 0.01, "T1_max_s": 1000.0, "T2_min_s": 0.005,
        "dw_max_rad_s": 2 * np.pi * 10e3, "dw_typical_khz": 0.5,
        "eps_typical": 5e-4, "eps_max": 0.05,
        "dt_ns": None, "gate_time_ns": 135_000.0,
        "display_unit": "ms", "time_scale": 1e3,
    },
    "neutral_atom": {
        "T1_s": 4.0, "T2_s": 1.0,
        "T1_min_s": 0.001, "T1_max_s": 100.0, "T2_min_s": 0.001,
        "dw_max_rad_s": 2 * np.pi * 100e3, "dw_typical_khz": 2.0,
        "eps_typical": 1e-3, "eps_max": 0.1,
        "dt_ns": None, "gate_time_ns": 500.0,
        "display_unit": "ms", "time_scale": 1e3,
    },
    "spin_qubit": {
        "T1_s": 1.0, "T2_s": 100e-6,
        "T1_min_s": 1e-6, "T1_max_s": 100.0, "T2_min_s": 0.1e-6,
        "dw_max_rad_s": 2 * np.pi * 1e6, "dw_typical_khz": 50.0,
        "eps_typical": 5e-3, "eps_max": 0.1,
        "dt_ns": None, "gate_time_ns": 100.0,
        "display_unit": "µs", "time_scale": 1e6,
    },
    "nv_center": {
        "T1_s": 6e-3, "T2_s": 1e-3,
        "T1_min_s": 1e-6, "T1_max_s": 10.0, "T2_min_s": 0.1e-6,
        "dw_max_rad_s": 2 * np.pi * 1e6, "dw_typical_khz": 20.0,
        "eps_typical": 5e-3, "eps_max": 0.1,
        "dt_ns": None, "gate_time_ns": 20.0,
        "display_unit": "µs", "time_scale": 1e6,
    },
}
ARCH_DEFAULTS["unknown"] = ARCH_DEFAULTS["superconducting"]

_T1_BASE_RATIO     = 0.50
_RAMSEY_BASE_RATIO = 0.04
_GATE_N_BASE       = 5
GATE_REP_N         = [_GATE_N_BASE, 2 * _GATE_N_BASE, 4 * _GATE_N_BASE]
_RAMSEY_DW_STARTS_KHZ = [-50, -20, -10, -5, 0, 5, 10, 20, 50]


def build_custom_arch(T1_prior_s: float, T2_prior_s: float) -> dict:
    if T1_prior_s < 1e-3:
        unit, scale = "µs", 1e6
    elif T1_prior_s < 1.0:
        unit, scale = "ms", 1e3
    else:
        unit, scale = "s", 1.0
    return {
        "T1_s": T1_prior_s, "T2_s": min(T2_prior_s, 2.0 * T1_prior_s),
        "T1_min_s": T1_prior_s / 1000, "T1_max_s": T1_prior_s * 100,
        "T2_min_s": T2_prior_s / 1000,
        "dw_max_rad_s": 2 * np.pi * 500e3, "dw_typical_khz": 5.0,
        "eps_typical": 1e-3, "eps_max": 0.1,
        "dt_ns": None, "gate_time_ns": 50.0,
        "display_unit": unit, "time_scale": scale,
    }


@dataclass
class BackendProfile:
    architecture: str
    T1_prior_s:   float
    T2_prior_s:   float
    dt_ns:        Optional[float]
    backend_name: str = "unknown"
    custom_arch:  Optional[dict] = None

    @property
    def constants(self) -> dict:
        if self.custom_arch is not None:
            return self.custom_arch
        return ARCH_DEFAULTS.get(self.architecture, ARCH_DEFAULTS["unknown"])

    @property
    def t1_delays_s(self) -> list[float]:
        base = _T1_BASE_RATIO * self.T1_prior_s
        delays = [base, 2 * base, 3 * base]
        if self.dt_ns is not None:
            dt_s = self.dt_ns * 1e-9
            delays = [max(dt_s, round(d / dt_s) * dt_s) for d in delays]
        return delays

    @property
    def ramsey_delays_s(self) -> list[float]:
        ratios = [0.04, 0.20, 1.00]
        delays = [r * self.T2_prior_s for r in ratios]
        if self.dt_ns is not None:
            dt_s = self.dt_ns * 1e-9
            delays = [max(dt_s, round(d / dt_s) * dt_s) for d in delays]
        return delays

    @classmethod
    def from_ibm_backend(cls, backend, qubit: int = 0) -> "BackendProfile":
        dt_ns    = (backend.dt * 1e9) if backend.dt else 0.2222
        T1_prior = ARCH_DEFAULTS["superconducting"]["T1_s"]
        T2_prior = ARCH_DEFAULTS["superconducting"]["T2_s"]
        try:
            props = backend.properties()
            t1    = props.qubit_property(qubit, "t1")[0]
            t2    = props.qubit_property(qubit, "t2")[0]
            if t1 and t1 > 0:
                T1_prior = float(t1)
            if t2 and t2 > 0:
                T2_prior = min(float(t2), 2.0 * T1_prior)
        except Exception:
            pass
        return cls(architecture="superconducting", T1_prior_s=T1_prior,
                   T2_prior_s=T2_prior, dt_ns=dt_ns,
                   backend_name=getattr(backend, "name", "ibm_unknown"))

    @classmethod
    def from_architecture(cls,
                          architecture: str,
                          backend_name: str = "unknown",
                          T1_prior_s: Optional[float] = None,
                          T2_prior_s: Optional[float] = None) -> "BackendProfile":
        if architecture == "custom":
            if T1_prior_s is None or T2_prior_s is None:
                raise ValueError("architecture='custom' requires T1_prior_s and T2_prior_s")
            custom = build_custom_arch(T1_prior_s, T2_prior_s)
            return cls(architecture="custom", T1_prior_s=custom["T1_s"],
                       T2_prior_s=custom["T2_s"], dt_ns=None,
                       backend_name=backend_name, custom_arch=custom)
        defaults = ARCH_DEFAULTS.get(architecture, ARCH_DEFAULTS["unknown"])
        T1 = T1_prior_s or defaults["T1_s"]
        T2 = min(T2_prior_s or defaults["T2_s"], 2.0 * T1)
        return cls(architecture=architecture, T1_prior_s=T1, T2_prior_s=T2,
                   dt_ns=defaults["dt_ns"], backend_name=backend_name)

    def updated(self, T1_est_s: float, T2_est_s: float) -> "BackendProfile":
        T2_est_s = min(T2_est_s, 2.0 * T1_est_s)
        return BackendProfile(
            architecture=self.architecture, T1_prior_s=T1_est_s,
            T2_prior_s=T2_est_s, dt_ns=self.dt_ns,
            backend_name=self.backend_name, custom_arch=self.custom_arch)


def _snap_delay(delay_s: float, dt_ns: Optional[float]) -> tuple[float, str]:
    if dt_ns is not None:
        dt_s     = dt_ns * 1e-9
        n_cycles = max(1, round(delay_s / dt_s))
        return float(n_cycles), "dt"
    return delay_s, "s"


def _build_t1_circuit(delay_s: float, dt_ns: Optional[float]):
    from qiskit import QuantumCircuit
    dur, unit = _snap_delay(delay_s, dt_ns)
    qc = QuantumCircuit(1, 1, name=f"t1_{delay_s*1e6:.2f}us")
    qc.x(0)
    qc.delay(dur, 0, unit=unit)
    qc.measure(0, 0)
    return qc


def _build_ramsey_circuit(delay_s: float, dt_ns: Optional[float],
                           delta_omega: float = 0.0):
    from qiskit import QuantumCircuit
    dur, unit = _snap_delay(delay_s, dt_ns)
    qc = QuantumCircuit(1, 1, name=f"ramsey_{delay_s*1e6:.2f}us")
    qc.h(0)
    qc.delay(dur, 0, unit=unit)
    if abs(delta_omega) > 0:
        qc.rz(delta_omega * delay_s, 0)
    qc.h(0)
    qc.measure(0, 0)
    return qc


def _build_gate_rep_circuit(N: int):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1, 1, name=f"gate_rep_N{N}")
    for _ in range(2 * N):
        qc.x(0)
    qc.measure(0, 0)
    return qc


def build_probe_circuits(profile: BackendProfile,
                          delta_omega_hint: float = 0.0) -> tuple[list, dict]:
    t1_delays     = profile.t1_delays_s
    ramsey_delays = profile.ramsey_delays_s
    circuits      = []
    for t in t1_delays:
        circuits.append(_build_t1_circuit(t, profile.dt_ns))
    for t in ramsey_delays:
        circuits.append(_build_ramsey_circuit(t, profile.dt_ns, delta_omega_hint))
    for N in GATE_REP_N:
        circuits.append(_build_gate_rep_circuit(N))
    metadata = {
        "t1_delays_s":     t1_delays,
        "ramsey_delays_s": ramsey_delays,
        "gate_rep_N":      GATE_REP_N,
        "architecture":    profile.architecture,
        "backend_name":    profile.backend_name,
        "T1_prior_s":      profile.T1_prior_s,
        "T2_prior_s":      profile.T2_prior_s,
        "dt_ns":           profile.dt_ns,
    }
    return circuits, metadata


def forward_t1(tau_s, T1_s: float):
    return np.exp(-np.asarray(tau_s, dtype=float) / T1_s)


def forward_ramsey(tau_s, T2_s: float, delta_omega: float):
    tau = np.asarray(tau_s, dtype=float)
    return 0.5 * (1.0 - np.exp(-tau / T2_s) * np.cos(delta_omega * tau))


def forward_gate(N, epsilon_sx: float):
    N = np.asarray(N, dtype=float)
    return 0.5 * (1.0 + (1.0 - 2.0 * epsilon_sx) ** (2.0 * N))


def _algebraic_t1_guess(p1: float, p2: float, p3: float,
                         tau_s: float) -> Optional[float]:
    denom = p1 - p2
    if abs(denom) < 1e-6:
        return None
    x = (p2 - p3) / denom
    if x <= 0 or x >= 1:
        return None
    return -tau_s / np.log(x)


def _algebraic_eps_guess(p1: float, p2: float, p3: float,
                          N: int) -> Optional[float]:
    denom = p1 - p2
    if abs(denom) < 1e-6:
        return None
    x = (p2 - p3) / denom
    if x <= 0 or x >= 1:
        return None
    return (1.0 - x ** (1.0 / (2.0 * N))) / 2.0


def _algebraic_ramsey_guess(p_short: float, tau_short_s: float,
                              p_long: float, tau_long_s: float,
                              dw_max_rad_s: float
                              ) -> tuple[Optional[float], Optional[float]]:
    val      = np.clip(1.0 - 2.0 * p_short, -1.0, 1.0)
    dw_guess = np.arccos(val) / tau_short_s
    if dw_guess > dw_max_rad_s:
        return None, None
    cos_term = np.cos(dw_guess * tau_long_s)
    if abs(cos_term) < 0.05:
        return None, dw_guess
    inner = (1.0 - 2.0 * p_long) / cos_term
    if abs(inner) >= 1.0 or abs(inner) < 1e-9:
        return None, dw_guess
    T2_guess = -tau_long_s / np.log(abs(inner))
    return T2_guess, dw_guess


def _invert_t1(p1_measured: np.ndarray, tau_s: np.ndarray,
               T1_prior_s: float, arch: dict) -> tuple[float, float, float]:
    T1_guess = _algebraic_t1_guess(
        float(p1_measured[0]), float(p1_measured[1]), float(p1_measured[2]),
        tau_s[0]) or T1_prior_s
    T1_guess = float(np.clip(T1_guess, arch["T1_min_s"], arch["T1_max_s"]))
    bounds   = ([arch["T1_min_s"]], [arch["T1_max_s"]])
    try:
        popt, pcov = curve_fit(forward_t1, tau_s, p1_measured,
                               p0=[T1_guess], bounds=bounds, maxfev=10_000)
        T1    = float(popt[0])
        sigma = float(np.sqrt(np.diag(pcov))[0])
        resid = float(np.sum((p1_measured - forward_t1(tau_s, T1)) ** 2))
    except Exception:
        T1, sigma, resid = T1_guess, np.inf, np.inf
    return T1, sigma, resid


def _invert_ramsey(p1_measured: np.ndarray, tau_s: np.ndarray,
                   T2_prior_s: float, arch: dict
                   ) -> tuple[float, float, float, float, float]:
    dw_max   = arch["dw_max_rad_s"]
    T2_min   = arch["T2_min_s"]
    T2_max   = min(2.0 * T2_prior_s * 3, 2.0 * arch["T1_s"])
    T2_alg, dw_alg = _algebraic_ramsey_guess(
        float(p1_measured[0]), tau_s[0],
        float(p1_measured[2]), tau_s[2], dw_max)
    T2_guess = float(np.clip(T2_alg or T2_prior_s, T2_min, T2_max))
    dw_guess = float(dw_alg or 2 * np.pi * arch["dw_typical_khz"] * 1e3)
    starts   = [dw_guess, -dw_guess] + [
        2 * np.pi * k * 1e3 for k in _RAMSEY_DW_STARTS_KHZ]
    bounds   = ([T2_min, -dw_max], [T2_max, dw_max])
    best     = {"T2": T2_guess, "dw": dw_guess,
                "sT2": np.inf, "sdw": np.inf, "resid": np.inf}
    for dw_start in starts:
        dw_start = float(np.clip(dw_start, -dw_max, dw_max))
        try:
            popt, pcov = curve_fit(
                forward_ramsey, tau_s, p1_measured,
                p0=[T2_guess, dw_start], bounds=bounds, maxfev=20_000)
            T2, dw = float(popt[0]), float(popt[1])
            sigs   = np.sqrt(np.diag(pcov))
            resid  = float(np.sum(
                (p1_measured - forward_ramsey(tau_s, T2, dw)) ** 2))
            if resid < best["resid"]:
                best = {"T2": T2, "dw": dw,
                        "sT2": float(sigs[0]), "sdw": float(sigs[1]),
                        "resid": resid}
        except Exception:
            continue
    return best["T2"], best["sT2"], best["dw"], best["sdw"], best["resid"]


def _invert_gate(p0_measured: np.ndarray, N_values: np.ndarray,
                 arch: dict) -> tuple[float, float, float]:
    eps_guess = _algebraic_eps_guess(
        float(p0_measured[0]), float(p0_measured[1]), float(p0_measured[2]),
        int(N_values[0])) or arch["eps_typical"]
    eps_guess = float(np.clip(eps_guess, 0.0, arch["eps_max"]))
    bounds    = ([0.0], [arch["eps_max"]])
    try:
        popt, pcov = curve_fit(forward_gate, N_values, p0_measured,
                               p0=[eps_guess], bounds=bounds, maxfev=10_000)
        eps   = float(popt[0])
        sigma = float(np.sqrt(np.diag(pcov))[0])
        resid = float(np.sum((p0_measured - forward_gate(N_values, eps)) ** 2))
    except Exception:
        eps, sigma, resid = eps_guess, np.inf, np.inf
    return eps, sigma, resid


@dataclass
class InversionResult:
    backend_name:       str
    qubit_id:           int
    timestamp:          str
    architecture:       str
    T1_s:               float
    T1_sigma_s:         float
    T2_s:               float
    T2_sigma_s:         float
    delta_omega:        float
    delta_omega_sigma:  float
    epsilon_sx:         float
    epsilon_sx_sigma:   float
    t1_residual:        float = 0.0
    ramsey_residual:    float = 0.0
    gate_residual:      float = 0.0
    p1_t1:              list = field(default_factory=list)
    p1_ramsey:          list = field(default_factory=list)
    p0_gate:            list = field(default_factory=list)
    t1_delays_s:        list = field(default_factory=list)
    ramsey_delays_s:    list = field(default_factory=list)
    gate_rep_N:         list = field(default_factory=list)

    def summary(self, arch: Optional[dict] = None) -> str:
        if arch is None:
            arch = ARCH_DEFAULTS["superconducting"]
        s  = arch["time_scale"]
        u  = arch["display_unit"]
        dw = self.delta_omega / (2 * np.pi * 1e3)
        return (f"Qubit {self.qubit_id} | "
                f"T1={self.T1_s*s:.2f}±{self.T1_sigma_s*s:.2f}{u}  "
                f"T2={self.T2_s*s:.2f}±{self.T2_sigma_s*s:.2f}{u}  "
                f"Δω={dw:.2f}kHz  ε={self.epsilon_sx:.2e}")


def lindblad_inversion(counts_list: list[dict],
                        metadata:    dict,
                        profile:     BackendProfile,
                        qubit_id:    int = 0,
                        shots:       int = 1000,
                        timestamp:   str = "") -> InversionResult:
    if len(counts_list) != 9:
        raise ValueError(f"Expected 9 count dicts, got {len(counts_list)}")
    arch      = profile.constants
    p1_t1     = np.array([counts_list[i].get("1", 0) / shots   for i in range(3)])
    p1_ramsey = np.array([counts_list[i+3].get("1", 0) / shots for i in range(3)])
    p0_gate   = np.array([counts_list[i+6].get("0", 0) / shots for i in range(3)])
    t1_delays     = np.array(metadata["t1_delays_s"])
    ramsey_delays = np.array(metadata["ramsey_delays_s"])
    N_values      = np.array(metadata["gate_rep_N"], dtype=float)
    T1, sT1, r_t1          = _invert_t1(p1_t1, t1_delays, profile.T1_prior_s, arch)
    T2, sT2, dw, sdw, r_ram = _invert_ramsey(p1_ramsey, ramsey_delays, profile.T2_prior_s, arch)
    T2 = min(T2, 2.0 * T1)
    eps, seps, r_gate       = _invert_gate(p0_gate, N_values, arch)
    return InversionResult(
        backend_name      = metadata.get("backend_name", profile.backend_name),
        qubit_id          = qubit_id,
        timestamp         = timestamp,
        architecture      = profile.architecture,
        T1_s              = T1,  T1_sigma_s        = sT1,
        T2_s              = T2,  T2_sigma_s        = sT2,
        delta_omega       = dw,  delta_omega_sigma = sdw,
        epsilon_sx        = eps, epsilon_sx_sigma  = seps,
        t1_residual       = r_t1,
        ramsey_residual   = r_ram,
        gate_residual     = r_gate,
        p1_t1             = p1_t1.tolist(),
        p1_ramsey         = p1_ramsey.tolist(),
        p0_gate           = p0_gate.tolist(),
        t1_delays_s       = t1_delays.tolist(),
        ramsey_delays_s   = ramsey_delays.tolist(),
        gate_rep_N        = N_values.tolist(),
    )