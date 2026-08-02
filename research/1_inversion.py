from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=RuntimeWarning)

__all__ = [
    "ARCH_DEFAULTS", "build_custom_arch", "BackendProfile",
    "InversionResult", "forward_t1", "forward_ramsey_xy",
    "forward_gate",
    "build_probe_circuits", "lindblad_inversion",
]

ARCH_DEFAULTS: dict[str, dict] = {
    "superconducting": {
        "T1_s": 150e-6, "T2_s": 90e-6,
        "T1_min_s": 1e-6, "T1_max_s": 1000e-6, "T2_min_s": 0.5e-6,
        "dw_max_rad_s": 2*np.pi*500e3, "dw_typical_khz": 5.0,
        "eps_typical": 3.5e-4, "eps_max": 0.5,
        "dt_ns": 0.2222, "gate_time_ns": 50.0,
        "display_unit": "µs", "time_scale": 1e6,
    },
    "trapped_ion": {
        "T1_s": 10.0, "T2_s": 1.0,
        "T1_min_s": 0.01, "T1_max_s": 1000.0, "T2_min_s": 0.005,
        "dw_max_rad_s": 2*np.pi*10e3, "dw_typical_khz": 0.5,
        "eps_typical": 5e-4, "eps_max": 0.5,
        "dt_ns": None, "gate_time_ns": 135_000.0,
        "display_unit": "ms", "time_scale": 1e3,
    },
    "neutral_atom": {
        "T1_s": 4.0, "T2_s": 1.0,
        "T1_min_s": 0.001, "T1_max_s": 100.0, "T2_min_s": 0.001,
        "dw_max_rad_s": 2*np.pi*100e3, "dw_typical_khz": 2.0,
        "eps_typical": 1e-3, "eps_max": 0.5,
        "dt_ns": None, "gate_time_ns": 500.0,
        "display_unit": "ms", "time_scale": 1e3,
    },
    "spin_qubit": {
        "T1_s": 1.0, "T2_s": 100e-6,
        "T1_min_s": 1e-6, "T1_max_s": 100.0, "T2_min_s": 0.1e-6,
        "dw_max_rad_s": 2*np.pi*1e6, "dw_typical_khz": 50.0,
        "eps_typical": 5e-3, "eps_max": 0.5,
        "dt_ns": None, "gate_time_ns": 100.0,
        "display_unit": "µs", "time_scale": 1e6,
    },
    "nv_center": {
        "T1_s": 6e-3, "T2_s": 1e-3,
        "T1_min_s": 1e-6, "T1_max_s": 10.0, "T2_min_s": 0.1e-6,
        "dw_max_rad_s": 2*np.pi*1e6, "dw_typical_khz": 20.0,
        "eps_typical": 5e-3, "eps_max": 0.5,
        "dt_ns": None, "gate_time_ns": 20.0,
        "display_unit": "µs", "time_scale": 1e6,
    },
}
ARCH_DEFAULTS["unknown"] = ARCH_DEFAULTS["superconducting"]

GATE_REP_N_DT = [20, 40, 80]


def build_custom_arch(T1_prior_s: float, T2_prior_s: float,
                      dw_max_rad_s: Optional[float] = None,
                      eps_typical:  Optional[float] = None,
                      gate_time_ns: Optional[float] = None) -> dict:
    if T1_prior_s < 1e-3:
        unit, scale = "µs", 1e6
    elif T1_prior_s < 1.0:
        unit, scale = "ms", 1e3
    else:
        unit, scale = "s", 1.0
    return {
        "T1_s": T1_prior_s, "T2_s": min(T2_prior_s, 2.0*T1_prior_s),
        "T1_min_s": T1_prior_s/1000, "T1_max_s": T1_prior_s*100,
        "T2_min_s": T2_prior_s/1000,
        "dw_max_rad_s": dw_max_rad_s if dw_max_rad_s is not None else 2*np.pi*500e3,
        "dw_typical_khz": 5.0,
        "eps_typical": eps_typical if eps_typical is not None else 1e-3,
        "eps_max": 0.5,
        "dt_ns": None,
        "gate_time_ns": gate_time_ns if gate_time_ns is not None else 50.0,
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
        base   = 0.50 * self.T1_prior_s
        delays = [base, 2*base, 3*base]
        if self.dt_ns is not None:
            dt_s   = self.dt_ns * 1e-9
            delays = [max(dt_s, round(d/dt_s)*dt_s) for d in delays]
        return delays

    @property
    def ramsey_delays_s(self) -> list[float]:
        T2     = self.T2_prior_s
        delays = [0.5*T2, 1.0*T2, 1.5*T2]
        if self.dt_ns is not None:
            dt_s   = self.dt_ns * 1e-9
            delays = [max(dt_s, round(d/dt_s)*dt_s) for d in delays]
        return delays

    @property
    def dw_max_rad_s(self) -> float:
        t1 = self.ramsey_delays_s[0]
        return 0.90 * np.pi / t1

    @classmethod
    def from_ibm_backend(cls, backend, qubit: int = 0) -> "BackendProfile":
        dt_ns    = (backend.dt*1e9) if backend.dt else 0.2222
        T1_prior = ARCH_DEFAULTS["superconducting"]["T1_s"]
        T2_prior = ARCH_DEFAULTS["superconducting"]["T2_s"]
        try:
            props = backend.properties()
            t1    = props.qubit_property(qubit, "t1")[0]
            t2    = props.qubit_property(qubit, "t2")[0]
            if t1 and t1 > 0: T1_prior = float(t1)
            if t2 and t2 > 0: T2_prior = min(float(t2), 2.0*T1_prior)
        except Exception:
            pass
        return cls(architecture="superconducting", T1_prior_s=T1_prior,
                   T2_prior_s=T2_prior, dt_ns=dt_ns,
                   backend_name=getattr(backend, "name", "ibm_unknown"))

    @classmethod
    def from_architecture(cls, architecture: str, backend_name: str = "unknown",
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
        T2 = min(T2_prior_s or defaults["T2_s"], 2.0*T1)
        return cls(architecture=architecture, T1_prior_s=T1, T2_prior_s=T2,
                   dt_ns=defaults["dt_ns"], backend_name=backend_name)

    def updated(self, T1_est_s: float, T2_est_s: float) -> "BackendProfile":
        T2_est_s = min(T2_est_s, 2.0*T1_est_s)
        return BackendProfile(architecture=self.architecture, T1_prior_s=T1_est_s,
                              T2_prior_s=T2_est_s, dt_ns=self.dt_ns,
                              backend_name=self.backend_name, custom_arch=self.custom_arch)


def _snap(delay_s: float, dt_ns: Optional[float]) -> tuple[float, str]:
    if dt_ns is not None and delay_s > 0:
        dt_s = dt_ns*1e-9
        return float(max(1, round(delay_s/dt_s))), "dt"
    return delay_s, "s"


def _build_t1_circuit(delay_s: float, dt_ns: Optional[float], idx: int):
    from qiskit import QuantumCircuit
    dur, unit = _snap(delay_s, dt_ns)
    qc = QuantumCircuit(1, 1, name=f"t1_{idx}")
    qc.x(0)
    if delay_s > 0:
        qc.delay(dur, 0, unit=unit)
    qc.measure(0, 0)
    return qc


def _build_ramsey_x_circuit(delay_s: float, dt_ns: Optional[float], idx: int):
    from qiskit import QuantumCircuit
    dur, unit = _snap(delay_s, dt_ns)
    qc = QuantumCircuit(1, 1, name=f"ramsey_X_{idx}")
    qc.h(0)
    qc.delay(dur, 0, unit=unit)
    qc.h(0)
    qc.measure(0, 0)
    return qc


def _build_ramsey_y_circuit(delay_s: float, dt_ns: Optional[float], idx: int):
    from qiskit import QuantumCircuit
    dur, unit = _snap(delay_s, dt_ns)
    qc = QuantumCircuit(1, 1, name=f"ramsey_Y_{idx}")
    qc.h(0)
    qc.delay(dur, 0, unit=unit)
    qc.sdg(0)
    qc.h(0)
    qc.measure(0, 0)
    return qc


def _build_gate_rep_circuit(N: int):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1, 1, name=f"gate_rep_N{N}")
    for _ in range(2*N):
        qc.x(0)
    qc.measure(0, 0)
    return qc


def build_probe_circuits(profile: BackendProfile) -> tuple[list, dict]:
    t1_delays     = profile.t1_delays_s
    ramsey_delays = profile.ramsey_delays_s
    circuits      = []

    for i, t in enumerate(t1_delays):
        circuits.append(_build_t1_circuit(t, profile.dt_ns, i))

    for i, t in enumerate(ramsey_delays):
        circuits.append(_build_ramsey_x_circuit(t, profile.dt_ns, i))
        circuits.append(_build_ramsey_y_circuit(t, profile.dt_ns, i))

    for N in GATE_REP_N_DT:
        circuits.append(_build_gate_rep_circuit(N))

    n_t1     = len(t1_delays)
    n_ramsey = 2 * len(ramsey_delays)
    n_gate   = len(GATE_REP_N_DT)

    metadata = {
        "t1_delays_s":    t1_delays,
        "ramsey_delays_s": ramsey_delays,
        "gate_rep_N":     GATE_REP_N_DT,
        "n_t1":           n_t1,
        "n_ramsey":       n_ramsey,
        "n_gate":         n_gate,
        "dw_max_rad_s":   profile.dw_max_rad_s,
        "architecture":   profile.architecture,
        "backend_name":   profile.backend_name,
        "T1_prior_s":     profile.T1_prior_s,
        "T2_prior_s":     profile.T2_prior_s,
        "dt_ns":          profile.dt_ns,
    }
    return circuits, metadata


def forward_t1(tau_s, T1_s: float):
    return np.exp(-np.asarray(tau_s, dtype=float) / T1_s)


def forward_ramsey_xy(t: float, T2_s: float, delta_omega: float):
    decay = np.exp(-t / T2_s)
    p1_x  = 0.5 * (1.0 - decay * np.cos(delta_omega * t))
    p1_y  = 0.5 * (1.0 - decay * np.sin(delta_omega * t))
    return p1_x, p1_y


def forward_gate(N, epsilon_sx: float):
    N = np.asarray(N, dtype=float)
    return 0.5 * (1.0 + (1.0 - 2.0*epsilon_sx)**(2.0*N))


def _invert_t1(p1_measured: np.ndarray, tau_s: np.ndarray,
               T1_prior_s: float, arch: dict) -> tuple[float, float, float]:
    denom    = float(p1_measured[0]) - float(p1_measured[1])
    T1_guess = T1_prior_s
    if abs(denom) > 1e-6:
        x = (float(p1_measured[1]) - float(p1_measured[2])) / denom
        if 0 < x < 1:
            T1_guess = float(np.clip(-tau_s[0]/np.log(x),
                                     arch["T1_min_s"], arch["T1_max_s"]))
    try:
        popt, pcov = curve_fit(forward_t1, tau_s, p1_measured,
                               p0=[T1_guess],
                               bounds=([arch["T1_min_s"]], [arch["T1_max_s"]]),
                               maxfev=10_000)
        T1    = float(popt[0])
        sigma = float(np.sqrt(np.diag(pcov))[0])
        resid = float(np.sum((p1_measured - forward_t1(tau_s, T1))**2))
    except Exception:
        T1, sigma, resid = T1_guess, np.inf, np.inf
    return T1, sigma, resid


def _invert_ramsey_3t(p1_x_arr: np.ndarray, p1_y_arr: np.ndarray,
                       ramsey_delays: np.ndarray,
                       T2_prior_s: float, arch: dict,
                       T1_estimate_s: float, shots_ramsey: int
                       ) -> tuple[float, float, float, float, float]:
    x_means = np.clip(1.0 - 2.0*p1_x_arr, -1.0, 1.0)
    y_means = np.clip(1.0 - 2.0*p1_y_arr, -1.0, 1.0)
    amps    = np.clip(np.sqrt(x_means**2 + y_means**2), 1e-6, 1.0)

    T2_min = arch["T2_min_s"]
    T2_max = min(T2_prior_s * 20, 2.0 * T1_estimate_s)

    T2_guess = T2_prior_s
    denom    = float(amps[0]) - float(amps[1])
    if abs(denom) > 1e-6:
        x = (float(amps[1]) - float(amps[2])) / denom
        if 0 < x < 1:
            T2_guess = float(np.clip(-ramsey_delays[0]/np.log(x),
                                     T2_min, T2_max))
    try:
        popt, pcov = curve_fit(forward_t1, ramsey_delays, amps,
                               p0=[T2_guess],
                               bounds=([T2_min], [T2_max]),
                               maxfev=10_000)
        T2    = float(popt[0])
        sT2   = float(np.sqrt(np.diag(pcov))[0])
        resid = float(np.sum((amps - forward_t1(ramsey_delays, T2))**2))
    except Exception:
        T2, sT2, resid = T2_guess, np.inf, np.inf

    t1     = float(ramsey_delays[0])
    angle  = float(np.arctan2(y_means[0], x_means[0]))
    dw     = angle / t1
    amp_t1 = float(amps[0])
    sigma_angle = 1.0 / (max(amp_t1, 0.05) * np.sqrt(max(shots_ramsey, 1)))
    sdw    = float(np.clip(sigma_angle / t1, 0, np.inf))

    return T2, sT2, dw, sdw, resid


def _invert_gate(p0_measured: np.ndarray, N_values: np.ndarray,
                  arch: dict) -> tuple[float, float, float]:
    denom    = float(p0_measured[0]) - float(p0_measured[1])
    eps_guess = arch["eps_typical"]
    if abs(denom) > 1e-6:
        x = (float(p0_measured[1]) - float(p0_measured[2])) / denom
        if 0 < x < 1:
            eps_guess = float(np.clip((1 - x**(1.0/(2.0*N_values[0])))/2.0,
                                      0.0, arch["eps_max"]))
    try:
        popt, pcov = curve_fit(forward_gate, N_values, p0_measured,
                               p0=[eps_guess],
                               bounds=([0.0], [arch["eps_max"]]),
                               maxfev=10_000)
        eps   = float(popt[0])
        sigma = float(np.sqrt(np.diag(pcov))[0])
        resid = float(np.sum((p0_measured - forward_gate(N_values, eps))**2))
    except Exception:
        eps, sigma, resid = eps_guess, np.inf, np.inf
    return eps, sigma, resid


def _chi2_per_dof(measured: np.ndarray, predicted: np.ndarray,
                  shots: int, n_params: int) -> float:
    var  = np.clip(predicted * (1.0 - predicted), 1e-8, None) / shots
    chi2 = float(np.sum((measured - predicted)**2 / var))
    dof  = max(len(measured) - n_params, 1)
    return chi2 / dof


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
    t1_chi2_dof:        float = 0.0
    ramsey_chi2_dof:    float = 0.0
    gate_chi2_dof:      float = 0.0

    def summary(self, arch: Optional[dict] = None) -> str:
        if arch is None:
            arch = ARCH_DEFAULTS["superconducting"]
        s  = arch["time_scale"]; u = arch["display_unit"]
        dw = self.delta_omega / (2*np.pi*1e3)
        return (f"Qubit {self.qubit_id} | "
                f"T1={self.T1_s*s:.2f}±{self.T1_sigma_s*s:.2f}{u}  "
                f"T2={self.T2_s*s:.2f}±{self.T2_sigma_s*s:.2f}{u}  "
                f"Δω={dw:.3f}kHz  ε={self.epsilon_sx:.2e}")


def lindblad_inversion(counts_list: list[dict],
                        metadata:    dict,
                        profile:     BackendProfile,
                        shots_t1:    int = 300,
                        shots_ramsey:int = 1000,
                        shots_gate:  int = 500,
                        qubit_id:    int = 0,
                        timestamp:   str = "") -> InversionResult:
    arch     = profile.constants
    n_t1     = metadata["n_t1"]
    n_ramsey = metadata["n_ramsey"]
    n_gate   = metadata["n_gate"]
    expected = n_t1 + n_ramsey + n_gate
    if len(counts_list) != expected:
        raise ValueError(f"Expected {expected} count dicts, got {len(counts_list)}")

    t1_counts     = counts_list[:n_t1]
    ramsey_counts = counts_list[n_t1:n_t1+n_ramsey]
    gate_counts   = counts_list[n_t1+n_ramsey:]

    p1_t1 = np.array([c.get("1",0)/shots_t1 for c in t1_counts])
    t1d   = np.array(metadata["t1_delays_s"])
    T1, sT1, r_t1 = _invert_t1(p1_t1, t1d, profile.T1_prior_s, arch)

    t1_chi2 = _chi2_per_dof(p1_t1, forward_t1(t1d, T1), shots_t1, n_params=1)

    n_times = n_ramsey // 2
    ramsey_delays = np.array(metadata["ramsey_delays_s"])
    p1_x_arr = np.array([ramsey_counts[2*i].get("1",0)/shots_ramsey   for i in range(n_times)])
    p1_y_arr = np.array([ramsey_counts[2*i+1].get("1",0)/shots_ramsey for i in range(n_times)])
    T2, sT2, dw, sdw, r_ram = _invert_ramsey_3t(
        p1_x_arr, p1_y_arr, ramsey_delays, profile.T2_prior_s, arch,
        T1_estimate_s=T1, shots_ramsey=shots_ramsey)

    px_pred, py_pred = forward_ramsey_xy(ramsey_delays, T2, dw)
    ramsey_meas = np.concatenate([p1_x_arr, p1_y_arr])
    ramsey_pred = np.concatenate([px_pred, py_pred])
    ramsey_chi2 = _chi2_per_dof(ramsey_meas, ramsey_pred, shots_ramsey, n_params=2)

    p0_gate = np.array([c.get("0",0)/shots_gate for c in gate_counts])
    N_vals  = np.array(metadata["gate_rep_N"], dtype=float)
    eps, seps, r_gate = _invert_gate(p0_gate, N_vals, arch)
    gate_chi2 = _chi2_per_dof(p0_gate, forward_gate(N_vals, eps), shots_gate, n_params=1)

    return InversionResult(
        backend_name=metadata.get("backend_name", profile.backend_name),
        qubit_id=qubit_id, timestamp=timestamp, architecture=profile.architecture,
        T1_s=T1, T1_sigma_s=sT1,
        T2_s=T2, T2_sigma_s=sT2,
        delta_omega=dw, delta_omega_sigma=sdw,
        epsilon_sx=eps, epsilon_sx_sigma=seps,
        t1_residual=r_t1, ramsey_residual=r_ram, gate_residual=r_gate,
        t1_chi2_dof=t1_chi2, ramsey_chi2_dof=ramsey_chi2, gate_chi2_dof=gate_chi2)