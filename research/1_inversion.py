from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar

warnings.filterwarnings("ignore", category=RuntimeWarning)

__all__ = [
    "ARCH_DEFAULTS", "build_custom_arch", "BackendProfile",
    "InversionResult", "forward_t1", "forward_ramsey_xy",
    "forward_rpe", "build_probe_circuits", "lindblad_inversion",
]

ARCH_DEFAULTS: dict[str, dict] = {
    "superconducting": {
        "T1_s": 150e-6, "T2_s": 90e-6,
        "T1_min_s": 1e-6, "T1_max_s": 1000e-6, "T2_min_s": 0.5e-6,
        "dw_max_rad_s": 2 * np.pi * 500e3, "dw_typical_khz": 5.0,
        "eps_typical": 3.5e-4, "eps_max": 0.5,
        "dt_ns": 0.2222, "gate_time_ns": 50.0,
        "display_unit": "µs", "time_scale": 1e6,
        "t1_mode": "3point",
        "t1_delays_s": None,
    },
    "trapped_ion": {
        "T1_s": 10.0, "T2_s": 1.0,
        "T1_min_s": 0.01, "T1_max_s": 1000.0, "T2_min_s": 0.005,
        "dw_max_rad_s": 2 * np.pi * 10e3, "dw_typical_khz": 0.5,
        "eps_typical": 5e-4, "eps_max": 0.5,
        "dt_ns": None, "gate_time_ns": 135_000.0,
        "display_unit": "ms", "time_scale": 1e3,
        "t1_mode": "2point",
        "t1_delays_s": [0.0, 1.0],
    },
    "neutral_atom": {
        "T1_s": 4.0, "T2_s": 1.0,
        "T1_min_s": 0.001, "T1_max_s": 100.0, "T2_min_s": 0.001,
        "dw_max_rad_s": 2 * np.pi * 100e3, "dw_typical_khz": 2.0,
        "eps_typical": 1e-3, "eps_max": 0.5,
        "dt_ns": None, "gate_time_ns": 500.0,
        "display_unit": "ms", "time_scale": 1e3,
        "t1_mode": "2point",
        "t1_delays_s": [0.0, 1.0],
    },
    "spin_qubit": {
        "T1_s": 1.0, "T2_s": 100e-6,
        "T1_min_s": 1e-6, "T1_max_s": 100.0, "T2_min_s": 0.1e-6,
        "dw_max_rad_s": 2 * np.pi * 1e6, "dw_typical_khz": 50.0,
        "eps_typical": 5e-3, "eps_max": 0.5,
        "dt_ns": None, "gate_time_ns": 100.0,
        "display_unit": "µs", "time_scale": 1e6,
        "t1_mode": "3point",
        "t1_delays_s": None,
    },
    "nv_center": {
        "T1_s": 6e-3, "T2_s": 1e-3,
        "T1_min_s": 1e-6, "T1_max_s": 10.0, "T2_min_s": 0.1e-6,
        "dw_max_rad_s": 2 * np.pi * 1e6, "dw_typical_khz": 20.0,
        "eps_typical": 5e-3, "eps_max": 0.5,
        "dt_ns": None, "gate_time_ns": 20.0,
        "display_unit": "µs", "time_scale": 1e6,
        "t1_mode": "3point",
        "t1_delays_s": None,
    },
}
ARCH_DEFAULTS["unknown"] = ARCH_DEFAULTS["superconducting"]

RPE_DEPTHS   = [0, 1, 2, 3, 4, 5, 6]
RPE_N_GATES  = [2**j for j in RPE_DEPTHS]


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
        "eps_typical": 1e-3, "eps_max": 0.5,
        "dt_ns": None, "gate_time_ns": 50.0,
        "display_unit": unit, "time_scale": scale,
        "t1_mode": "3point", "t1_delays_s": None,
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
        c = self.constants
        if c["t1_mode"] == "2point":
            return list(c["t1_delays_s"])
        base   = 0.50 * self.T1_prior_s
        delays = [base, 2 * base, 3 * base]
        if self.dt_ns is not None:
            dt_s   = self.dt_ns * 1e-9
            delays = [max(dt_s, round(d / dt_s) * dt_s) for d in delays]
        return delays

    @property
    def ramsey_t_opt_s(self) -> float:
        t_opt = 1.0 / (1.0 / self.T2_prior_s)
        if self.dt_ns is not None:
            dt_s  = self.dt_ns * 1e-9
            t_opt = max(dt_s, round(t_opt / dt_s) * dt_s)
        return t_opt

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
    if dt_ns is not None and delay_s > 0:
        dt_s     = dt_ns * 1e-9
        n_cycles = max(1, round(delay_s / dt_s))
        return float(n_cycles), "dt"
    return delay_s, "s"


def _build_t1_circuit(delay_s: float, dt_ns: Optional[float], idx: int):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1, 1, name=f"t1_{idx}_{delay_s*1e3:.3f}ms")
    qc.x(0)
    if delay_s > 0:
        dur, unit = _snap_delay(delay_s, dt_ns)
        qc.delay(dur, 0, unit=unit)
    qc.measure(0, 0)
    return qc


def _build_ramsey_x_circuit(t_opt_s: float, dt_ns: Optional[float]):
    from qiskit import QuantumCircuit
    dur, unit = _snap_delay(t_opt_s, dt_ns)
    qc = QuantumCircuit(1, 1, name="ramsey_X")
    qc.h(0)
    qc.delay(dur, 0, unit=unit)
    qc.h(0)
    qc.measure(0, 0)
    return qc


def _build_ramsey_y_circuit(t_opt_s: float, dt_ns: Optional[float]):
    from qiskit import QuantumCircuit
    dur, unit = _snap_delay(t_opt_s, dt_ns)
    qc = QuantumCircuit(1, 1, name="ramsey_Y")
    qc.h(0)
    qc.delay(dur, 0, unit=unit)
    qc.sdg(0)
    qc.h(0)
    qc.measure(0, 0)
    return qc


def _build_rpe_circuit(n_gates: int, basis: str):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1, 1, name=f"rpe_{basis}_n{n_gates}")
    for _ in range(n_gates):
        qc.sx(0)
    if basis == "Y":
        qc.sdg(0)
        qc.h(0)
    qc.measure(0, 0)
    return qc


def build_probe_circuits(profile: BackendProfile) -> tuple[list, dict]:
    t1_delays = profile.t1_delays_s
    t_opt     = profile.ramsey_t_opt_s
    circuits  = []

    for i, t in enumerate(t1_delays):
        circuits.append(_build_t1_circuit(t, profile.dt_ns, i))

    circuits.append(_build_ramsey_x_circuit(t_opt, profile.dt_ns))
    circuits.append(_build_ramsey_y_circuit(t_opt, profile.dt_ns))

    for n in RPE_N_GATES:
        circuits.append(_build_rpe_circuit(n, "Y"))
        circuits.append(_build_rpe_circuit(n, "Z"))

    n_t1     = len(t1_delays)
    n_ramsey = 2
    n_rpe    = 2 * len(RPE_N_GATES)

    metadata = {
        "t1_delays_s":   t1_delays,
        "t1_mode":       profile.constants["t1_mode"],
        "ramsey_t_opt_s": t_opt,
        "rpe_n_gates":   RPE_N_GATES,
        "rpe_depths":    RPE_DEPTHS,
        "n_t1":          n_t1,
        "n_ramsey":      n_ramsey,
        "n_rpe":         n_rpe,
        "architecture":  profile.architecture,
        "backend_name":  profile.backend_name,
        "T1_prior_s":    profile.T1_prior_s,
        "T2_prior_s":    profile.T2_prior_s,
        "dt_ns":         profile.dt_ns,
    }
    return circuits, metadata


def forward_t1(tau_s, T1_s: float):
    return np.exp(-np.asarray(tau_s, dtype=float) / T1_s)


def forward_ramsey_xy(t: float, T2_s: float, delta_omega: float):
    decay   = np.exp(-t / T2_s)
    x_mean  = decay * np.cos(delta_omega * t)
    y_mean  = decay * np.sin(delta_omega * t)
    p1_x    = 0.5 * (1.0 - x_mean)
    p1_y    = 0.5 * (1.0 - y_mean)
    return p1_x, p1_y


def forward_rpe(n_gates: int, epsilon: float) -> tuple[float, float]:
    theta = n_gates * (np.pi / 2.0 + epsilon)
    p1_y  = 0.5 * (1.0 + np.sin(theta))
    p1_z  = 0.5 * (1.0 - np.cos(theta))
    return p1_y, p1_z


GATE_REP_N_DT = [5, 10, 20]


def forward_gate(N, epsilon_sx: float):
    N = np.asarray(N, dtype=float)
    return 0.5 * (1.0 + (1.0 - 2.0 * epsilon_sx) ** (2.0 * N))


def _build_gate_rep_circuit(N: int):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1, 1, name=f"gate_rep_N{N}")
    for _ in range(2 * N):
        qc.x(0)
    qc.measure(0, 0)
    return qc


def _invert_gate(p0_measured: np.ndarray, N_values: np.ndarray,
                  arch: dict) -> tuple[float, float, float]:
    from scipy.optimize import curve_fit as _cf
    denom = float(p0_measured[0]) - float(p0_measured[1])
    eps_guess = arch["eps_typical"]
    if abs(denom) > 1e-6:
        x = (float(p0_measured[1]) - float(p0_measured[2])) / denom
        if 0 < x < 1:
            eps_guess = float(np.clip(
                (1 - x ** (1.0 / (2.0 * N_values[0]))) / 2.0,
                0.0, arch["eps_max"]))
    try:
        popt, pcov = _cf(forward_gate, N_values, p0_measured,
                         p0=[eps_guess], bounds=([0.0], [arch["eps_max"]]),
                         maxfev=10_000)
        eps   = float(popt[0])
        sigma = float(np.sqrt(np.diag(pcov))[0])
        resid = float(np.sum((p0_measured - forward_gate(N_values, eps)) ** 2))
    except Exception:
        eps, sigma, resid = eps_guess, np.inf, np.inf
    return eps, sigma, resid


def _invert_t1_3point(p1_measured: np.ndarray, tau_s: np.ndarray,
                       T1_prior_s: float, arch: dict) -> tuple[float, float, float]:
    tau_s = np.asarray(tau_s)
    denom = float(p1_measured[0]) - float(p1_measured[1])
    T1_guess = T1_prior_s
    if abs(denom) > 1e-6:
        x = (float(p1_measured[1]) - float(p1_measured[2])) / denom
        if 0 < x < 1:
            T1_guess = float(np.clip(-tau_s[0] / np.log(x),
                                     arch["T1_min_s"], arch["T1_max_s"]))
    try:
        popt, pcov = curve_fit(forward_t1, tau_s, p1_measured,
                               p0=[T1_guess],
                               bounds=([arch["T1_min_s"]], [arch["T1_max_s"]]),
                               maxfev=10_000)
        T1    = float(popt[0])
        sigma = float(np.sqrt(np.diag(pcov))[0])
        resid = float(np.sum((p1_measured - forward_t1(tau_s, T1)) ** 2))
    except Exception:
        T1, sigma, resid = T1_guess, np.inf, np.inf
    return T1, sigma, resid


def _invert_t1_2point(p_t0: float, p_t1: float, delay_s: float,
                       T1_prior_s: float, arch: dict) -> tuple[float, float, float]:
    p_spam  = float(p_t0)
    p_delay = float(p_t1)
    ratio   = p_delay / p_spam if p_spam > 0.01 else p_delay
    ratio   = float(np.clip(ratio, 1e-9, 1.0 - 1e-9))
    T1      = float(np.clip(-delay_s / np.log(ratio),
                            arch["T1_min_s"], arch["T1_max_s"]))
    sigma_T1 = T1 ** 2 / delay_s * np.sqrt(
        1.0 / max(p_spam, 1e-6) + 1.0 / max(p_delay, 1e-6)) / 50.0
    resid = 0.0
    return T1, float(sigma_T1), resid


def _invert_ramsey_xy(p1_x: float, p1_y: float, t_opt: float,
                       T2_prior_s: float, arch: dict) -> tuple[float, float, float, float, float]:
    x_mean  = float(np.clip(1.0 - 2.0 * p1_x, -1.0, 1.0))
    y_mean  = float(np.clip(1.0 - 2.0 * p1_y, -1.0, 1.0))
    amp     = np.sqrt(x_mean ** 2 + y_mean ** 2)
    amp     = float(np.clip(amp, 1e-9, 1.0))
    T2      = float(np.clip(-t_opt / np.log(amp),
                            arch["T2_min_s"], 2.0 * T2_prior_s * 3))
    dw      = float(np.arctan2(y_mean, x_mean) / t_opt)
    N       = 50.0
    dT2     = T2 ** 2 / (t_opt * amp) * np.sqrt(2.0) / np.sqrt(N)
    ddw     = 1.0 / (t_opt * amp * np.sqrt(N))
    resid   = 0.0
    return T2, float(dT2), dw, float(ddw), resid


def _invert_rpe(p1_y_arr: np.ndarray, p1_z_arr: np.ndarray,
                n_gates_arr: list, arch: dict) -> tuple[float, float, float]:
    epsilon_est = 0.0
    for j, n in enumerate(n_gates_arr):
        py      = float(p1_y_arr[j])
        pz      = float(p1_z_arr[j])
        sin_est = float(np.clip(2.0 * py - 1.0, -1.0, 1.0))
        cos_est = float(np.clip(1.0 - 2.0 * pz, -1.0, 1.0))
        theta_j = np.arctan2(sin_est, cos_est)
        epsilon_est += theta_j / n
    epsilon_est /= len(n_gates_arr)
    epsilon_est  = float(np.clip(epsilon_est, -arch["eps_max"], arch["eps_max"]))

    best_eps = epsilon_est
    best_res = np.inf
    for eps_init in [epsilon_est, 0.0, arch["eps_typical"], -arch["eps_typical"]]:
        try:
            def loss(e):
                total = 0.0
                for j, n in enumerate(n_gates_arr):
                    py_pred, pz_pred = forward_rpe(n, e)
                    total += (p1_y_arr[j] - py_pred) ** 2 + (p1_z_arr[j] - pz_pred) ** 2
                return total
            res = minimize_scalar(loss,
                                  bounds=(-arch["eps_max"], arch["eps_max"]),
                                  method="bounded")
            if res.fun < best_res:
                best_eps = float(res.x)
                best_res = float(res.fun)
        except Exception:
            continue

    sigma = arch["eps_typical"] * 0.01
    return best_eps, sigma, best_res


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

    def summary(self, arch: Optional[dict] = None) -> str:
        if arch is None:
            arch = ARCH_DEFAULTS["superconducting"]
        s  = arch["time_scale"]
        u  = arch["display_unit"]
        dw = self.delta_omega / (2 * np.pi * 1e3)
        return (f"Qubit {self.qubit_id} | "
                f"T1={self.T1_s*s:.2f}±{self.T1_sigma_s*s:.2f}{u}  "
                f"T2={self.T2_s*s:.2f}±{self.T2_sigma_s*s:.2f}{u}  "
                f"Δω={dw:.3f}kHz  ε={self.epsilon_sx:.2e}")


def lindblad_inversion(counts_list: list[dict],
                        metadata:    dict,
                        profile:     BackendProfile,
                        shots_t1:    int = 200,
                        shots_ramsey:int = 600,
                        shots_rpe:   int = 200,
                        qubit_id:    int = 0,
                        timestamp:   str = "") -> InversionResult:
    arch     = profile.constants
    n_t1     = metadata["n_t1"]
    n_ramsey = metadata["n_ramsey"]
    n_rpe    = metadata["n_rpe"]
    expected = n_t1 + n_ramsey + n_rpe
    if len(counts_list) != expected:
        raise ValueError(f"Expected {expected} count dicts, got {len(counts_list)}")

    t1_counts     = counts_list[:n_t1]
    ramsey_counts = counts_list[n_t1:n_t1 + n_ramsey]
    rpe_counts    = counts_list[n_t1 + n_ramsey:]

    if metadata["t1_mode"] == "2point":
        p_t0 = t1_counts[0].get("1", 0) / shots_t1
        p_t1 = t1_counts[1].get("1", 0) / shots_t1
        T1, sT1, r_t1 = _invert_t1_2point(
            p_t0, p_t1,
            metadata["t1_delays_s"][1],
            profile.T1_prior_s, arch)
    else:
        p1_t1 = np.array([c.get("1", 0) / shots_t1 for c in t1_counts])
        t1d   = np.array(metadata["t1_delays_s"])
        T1, sT1, r_t1 = _invert_t1_3point(p1_t1, t1d, profile.T1_prior_s, arch)

    p1_x  = ramsey_counts[0].get("1", 0) / shots_ramsey
    p1_y  = ramsey_counts[1].get("1", 0) / shots_ramsey
    t_opt = metadata["ramsey_t_opt_s"]
    T2, sT2, dw, sdw, r_ram = _invert_ramsey_xy(
        p1_x, p1_y, t_opt, profile.T2_prior_s, arch)
    T2 = min(T2, 2.0 * T1)

    n_depths  = len(RPE_DEPTHS)
    p1_y_arr  = np.array([rpe_counts[2*j].get("1",   0) / shots_rpe for j in range(n_depths)])
    p1_z_arr  = np.array([rpe_counts[2*j+1].get("1", 0) / shots_rpe for j in range(n_depths)])
    eps, seps, r_gate = _invert_rpe(p1_y_arr, p1_z_arr, RPE_N_GATES, arch)

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
    )