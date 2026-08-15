from __future__ import annotations
import datetime
import warnings
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.optimize import curve_fit



__all__ = [
    "ARCH_DEFAULTS", "build_custom_arch", "BackendProfile",
    "CalibrationSource", "InversionResult", "forward_t1", "forward_ramsey_xy",
    "forward_gate", "forward_echo",
    "build_probe_circuits", "run_probe_circuits_aer", "lindblad_inversion",
    "get_live_profile", "fetch_live_spam",
    "_sqrtx_native_inverse_pair", "_build_gate_rep_circuit",
    "split_qubits_for_op_limit",
]

_MAI_2024_P0_GIVEN_1 = 0.0005
_MAI_2024_P1_GIVEN_0 = 0.0018


ARCH_DEFAULTS: dict[str, dict] = {
    "superconducting": {
        "T1_s": 150e-6, "T2_s": 90e-6,
        "T1_min_s": 100e-9,
        
        "T1_max_s": 1e-3, "T2_min_s": 100e-9,
        "dw_max_rad_s": 2*np.pi*500e3, "dw_typical_khz": 5.0,
        "eps_typical": 3.5e-4, "eps_max": 0.5,
        "dt_ns": 0.2222, "gate_time_ns": 50.0,
        
        "p0_given_1": 0.0092, "p1_given_0": 0.0009,
        "display_unit": "µs", "time_scale": 1e6,
    },
    "trapped_ion": {
        "T1_s": 1000.0, "T2_s": 1.0,
        "T1_min_s": 0.01,
        
        "T1_max_s": 100000.0, "T2_min_s": 0.001,
        "dw_max_rad_s": 2*np.pi*10e3, "dw_typical_khz": 0.5,
        "eps_typical": 5e-4, "eps_max": 0.5,
        "dt_ns": None, "gate_time_ns": 135_000.0,
        
        "p0_given_1": 0.0005, "p1_given_0": 0.0018,
        "display_unit": "ms", "time_scale": 1e3,
    },
    "neutral_atom": {
        "T1_s": 10.0, "T2_s": 1.0,
        "T1_min_s": 0.001,
        
        "T1_max_s": 100000.0, "T2_min_s": 0.001,
        "dw_max_rad_s": 2*np.pi*100e3, "dw_typical_khz": 2.0,
        "eps_typical": np.sqrt(1e-3 * 1e-2), "eps_max": 0.5,  # ~3.16e-3
        "dt_ns": None, "gate_time_ns": 500.0,
        
        "p0_given_1": 0.0060, "p1_given_0": 0.0040,
        "display_unit": "ms", "time_scale": 1e3,
    },
}
ARCH_DEFAULTS["unknown"] = ARCH_DEFAULTS["superconducting"]

class _DeprecatedGateRepN(list):
    """Deprecated module-level gate-rep constant. Use BackendProfile.gate_rep_n."""
    def __iter__(self):
        warnings.warn(
            "GATE_REP_N_DT is deprecated and no longer used by "
            "build_probe_circuits(); use BackendProfile.gate_rep_n instead.",
            DeprecationWarning, stacklevel=2)
        return super().__iter__()


GATE_REP_N_DT = _DeprecatedGateRepN([20, 40, 80])


def build_custom_arch(T1_prior_s: float, T2_prior_s: float,
                      dw_max_rad_s: Optional[float] = None,
                      eps_typical:  Optional[float] = None,
                      gate_time_ns: Optional[float] = None,
                      p0_given_1:   float = 0.0,
                      p1_given_0:   float = 0.0) -> dict:
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
        "p0_given_1": p0_given_1,
        "p1_given_0": p1_given_0,
        "display_unit": unit, "time_scale": scale,
    }


def fetch_live_spam(
    architecture: str,
    backend=None,
    qubit_id: int = 0,
    ionq_api_token: Optional[str] = None,
    ionq_backend_name: str = "forte-1",
) -> tuple[float, float, str, str]:
    """
    This fetches the live SPAM error rates from a provider where possible,
    otherwise falls back to default error rates from ARCH_DEFAULTS.

    Returns (p0_given_1, p1_given_0, spam_source, spam_note). 
    """
    if architecture == "superconducting":
        try:
            if backend is None:
                raise ValueError("no backend provided")
            props = backend.properties()
            p0 = props.qubit_property(qubit_id, "prob_meas0_prep1")[0]
            p1 = props.qubit_property(qubit_id, "prob_meas1_prep0")[0]
            if p0 and p1 and p0 > 0 and p1 > 0:
                return (float(p0), float(p1), "live_asymmetric",
                        f"IBM: p0|1={p0:.4f}, p1|0={p1:.4f} from "
                        f"backend.properties() qubit {qubit_id}")
        except Exception:
            pass
        arch = ARCH_DEFAULTS["superconducting"]
        return (arch["p0_given_1"], arch["p1_given_0"], "literature_fallback",
                "IBM properties() unavailable — using Chen et al. (2023) defaults")

    if architecture == "trapped_ion":
        try:
            if ionq_api_token is None:
                raise ValueError("no ionq_api_token provided")
            import requests
            url = (f"https://api.ionq.co/v0.3/characterizations/backends/" 
                   f"qpu.{ionq_backend_name}/current")
            resp = requests.get(
                url, headers={"Authorization": f"apiKey {ionq_api_token}"}, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            spam_error = None
            def _extract_scalar(v):
                if isinstance(v, dict):
                    v = v.get("mean")
                return v
            for path in (
                lambda d: d.get("spam_error"),
                lambda d: _extract_scalar(d.get("fidelity", {}).get("spam")),
                lambda d: d.get("characteristic", {}).get("spam_error"),
                lambda d: _extract_scalar(
                    d.get("characteristic", {}).get("fidelity", {}).get("spam")),
            ):
                try:
                    v = path(data)
                    if v is not None:
                        raw = float(v)
                        spam_error = (1.0 - raw) if raw > 0.5 else raw
                        break
                except Exception:
                    continue

            if spam_error is not None and spam_error > 0:
                ratio = _MAI_2024_P0_GIVEN_1 / _MAI_2024_P1_GIVEN_0
                p0 = spam_error * 2.0 * ratio / (1.0 + ratio)
                p1 = spam_error * 2.0 / (1.0 + ratio)
                return (p0, p1, "live_scaled_from_combined",
                        f"IonQ: spam_error={spam_error:.6f} rescaled using "
                        f"Mai et al. (2024) ratio "
                        f"{_MAI_2024_P0_GIVEN_1}/{_MAI_2024_P1_GIVEN_0}")
        except Exception:
            pass
        arch = ARCH_DEFAULTS["trapped_ion"]
        return (arch["p0_given_1"], arch["p1_given_0"], "literature_fallback",
                "IonQ API unavailable or field not found — using Mai et al. "
                "(2024) defaults")

    if architecture == "neutral_atom":
        arch = ARCH_DEFAULTS["neutral_atom"]
        return (arch["p0_given_1"], arch["p1_given_0"], "literature_fallback",
                "No provider SPAM data available — using Evered et al. (2023) "
                f"defaults (p0|1={arch['p0_given_1']:.4f}, "
                f"p1|0={arch['p1_given_0']:.4f})")

    arch = ARCH_DEFAULTS.get(architecture, ARCH_DEFAULTS["unknown"])
    return (arch["p0_given_1"], arch["p1_given_0"], "literature_fallback",
            f"No live SPAM path for architecture={architecture!r} — using "
            "architecture defaults")


@dataclass
class CalibrationSource:
    """Provenance record for BackendProfile priors. This audits which values came
    from a live provider calibration data or an architecture default."""
    T1_source:  str   
    T2_source:  str
    eps_source: str
    dw_source:  str
    timestamp:  str   # 
    backend:    str   
    spam_source: str = "literature_fallback"
    spam_note:  str = ""


@dataclass
class BackendProfile:
    """
    architecture, T1_prior_s/T2_prior_s: platform and prior decoherence estimates.
    dt_ns: hardware sample-time quantization, if applicable.
    backend_name, custom_arch, calibration_source: provenance/bookkeeping.
    prior_confidence: "live" | "arch_default". "live" means T1_prior_s/T2_prior_s
      were read from a provider calibration API and are trusted; "arch_default"
      means they are architecture-typical literature values that may be off by
      an order of magnitude or more for the actual device.

    Delay-grid strategy: when prior_confidence == "live", t1_delays_s /
    ramsey_delays_s / echo_delays_s use linear multiples of the prior
    (0.5x/1.0x/1.5x etc) — this is Fisher-optimal when the prior is close to
    the true decay constant. When prior_confidence == "arch_default", the
    prior may be wrong by 10x or more, so delays are instead log-spaced across
    the architecture's full plausible [T1_min_s, T1_max_s] (or T2 equivalent)
    range at the 15th/50th/85th percentile in log space — this maximizes the
    chance that at least one delay point lands where the signal has partially
    decayed, regardless of where the true timescale actually falls.

    gate_rep_n: architecture-appropriate gate-repetition counts, computed from
    eps_typical so the total depolarizing signal neither saturates nor stays
    flat (see BackendProfile.gate_rep_n).
    """
    architecture: str
    T1_prior_s:   float
    T2_prior_s:   float
    dt_ns:        Optional[float]
    backend_name: str = "unknown"
    custom_arch:  Optional[dict] = None
    calibration_source: Optional[CalibrationSource] = None
    prior_confidence: str = "arch_default"  # "live" | "arch_default"

    @property
    def constants(self) -> dict:
        if self.custom_arch is not None:
            return self.custom_arch
        return ARCH_DEFAULTS.get(self.architecture, ARCH_DEFAULTS["unknown"])

    def _snap_all(self, delays: list[float]) -> list[float]:
        if self.dt_ns is not None:
            dt_s = self.dt_ns * 1e-9
            return [max(dt_s, round(d/dt_s)*dt_s) for d in delays]
        return delays

    @staticmethod
    def _log_spaced(lo: float, hi: float) -> list[float]:
        log_min, log_max = np.log10(lo), np.log10(hi)
        return [float(10**(log_min + frac*(log_max-log_min)))
                for frac in (0.15, 0.50, 0.85)]

    @property
    def t1_delays_s(self) -> list[float]:
        if self.prior_confidence == "live":
            base   = 0.50 * self.T1_prior_s
            delays = [base, 2*base, 3*base]
        else:
            arch   = self.constants
            delays = self._log_spaced(arch["T1_min_s"], arch["T1_max_s"])
        return self._snap_all(delays)

    @property
    def ramsey_delays_s(self) -> list[float]:
        if self.prior_confidence == "live":
            T2     = self.T2_prior_s
            delays = [0.5*T2, 1.0*T2, 1.5*T2]
        else:
            arch        = self.constants
            t1_delays   = self.t1_delays_s
            T1_estimate = t1_delays[1]
            T2c         = arch["T2_s"]
            lo, hi      = arch["T2_min_s"], 2.0 * T1_estimate
            delays      = [float(np.clip(d, lo, hi))
                           for d in (T2c / 5.0, T2c, T2c * 5.0)]
        return self._snap_all(delays)

    @property
    def echo_delays_s(self) -> list[float]:
        return self.ramsey_delays_s

    @property
    def dw_max_rad_s(self) -> float:
        t1 = self.ramsey_delays_s[0]
        return 0.90 * np.pi / t1

    @property
    def gate_rep_n(self) -> list[int]:
        eps    = self.constants["eps_typical"]
        n_max  = max(5, min(int(1.0 / (4.0 * eps)), 2000))
        n_mid  = max(3, n_max // 2)
        n_low  = max(1, n_max // 4)
        return sorted(set([n_low, n_mid, n_max]))

    @classmethod
    def from_ibm_backend(cls, backend, qubit: int = 0) -> "BackendProfile":
        arch         = ARCH_DEFAULTS["superconducting"]
        dt_ns        = (backend.dt*1e9) if backend.dt else 0.2222
        backend_name = getattr(backend, "name", "ibm_unknown")

        T1_prior,  T1_source  = arch["T1_s"], "arch_default"
        T2_prior,  T2_source  = arch["T2_s"], "arch_default"
        eps_prior, eps_source = arch["eps_typical"], "arch_default"
        dw_source = "arch_default"  # IBM properties() does not expose Δω

        try:
            props = backend.properties()
        except Exception:
            props = None

        if props is not None:
            try:
                t1 = props.qubit_property(qubit, "t1")[0]
                if t1 and t1 > 0:
                    T1_prior, T1_source = float(t1), "live_calibration"
            except Exception:
                pass

            try:
                t2 = props.qubit_property(qubit, "t2")[0]
                if t2 and t2 > 0:
                    T2_prior, T2_source = float(t2), "live_calibration"
            except Exception:
                pass

            eps_live = None
            try:
                eps_live = props.gate_error("sx", qubit)
            except Exception:
                try:
                    eps_live = props.gate_error("x", qubit)
                except Exception:
                    eps_live = None
            if eps_live is not None:
                try:
                    eps_prior  = float(np.clip(eps_live, 0.0, arch["eps_max"]))
                    eps_source = "live_calibration"
                except Exception:
                    pass

        T2_prior = min(T2_prior, 2.0*T1_prior)

        p0_given_1, p1_given_0, spam_source, spam_note = fetch_live_spam(
            "superconducting", backend=backend, qubit_id=qubit)

        custom_arch = None
        if eps_source == "live_calibration" or spam_source != "literature_fallback":
            custom_arch = dict(arch)
            custom_arch["eps_typical"] = eps_prior
            custom_arch["p0_given_1"] = p0_given_1
            custom_arch["p1_given_0"] = p1_given_0

        calibration_source = CalibrationSource(
            T1_source=T1_source, T2_source=T2_source,
            eps_source=eps_source, dw_source=dw_source,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            backend=backend_name,
            spam_source=spam_source, spam_note=spam_note,
        )

        print(f"[Canary] IBM live priors — qubit {qubit}: "
              f"T1={T1_prior*1e6:.1f}µs ({T1_source}), "
              f"T2={T2_prior*1e6:.1f}µs ({T2_source}), "
              f"ε_sx={eps_prior:.2e} ({eps_source})  "
              f"SPAM: p0|1={p0_given_1:.4f}, p1|0={p1_given_0:.4f} ({spam_source})")

        prior_confidence = ("live" if T1_source == "live_calibration"
                            and T2_source == "live_calibration" else "arch_default")
        return cls(architecture="superconducting", T1_prior_s=T1_prior,
                   T2_prior_s=T2_prior, dt_ns=dt_ns,
                   backend_name=backend_name, custom_arch=custom_arch,
                   calibration_source=calibration_source,
                   prior_confidence=prior_confidence)

    @classmethod
    def from_ionq_backend(cls, api_token: str, backend_name: str = "forte-1",
                          qubit_id: int = 0) -> "BackendProfile":
        arch = ARCH_DEFAULTS["trapped_ion"]

        T1_prior,  T1_source  = arch["T1_s"], "arch_default"
        T2_prior,  T2_source  = arch["T2_s"], "arch_default"
        eps_prior, eps_source = arch["eps_typical"], "arch_default"
        dw_source = "arch_default"  

        try:
            try:
                import requests
            except ImportError as exc:
                raise RuntimeError("requests is not installed") from exc

            url = (f"https://api.ionq.co/v0.3/characterizations/backends/"
                   f"qpu.{backend_name}/current")
            resp = requests.get(url, headers={"Authorization": f"apiKey {api_token}"},
                                timeout=10)
            resp.raise_for_status()
            data = resp.json()

            timing = data.get("timing", {}) or {}
            t1 = timing.get("t1")
            if t1 is not None and float(t1) > 0:
                T1_prior, T1_source = float(t1), "live_calibration"

            t2 = timing.get("t2")
            if t2 is not None and float(t2) > 0:
                T2_prior, T2_source = float(t2), "live_calibration"

            fidelity_1q = None
            fidelity_1q_stat_used = None
            fid_dict = data.get("fidelity", {}) or {}
            f1q = fid_dict.get("1q")
            f1q_mean   = None
            f1q_median = None
            if isinstance(f1q, dict):
                f1q_mean   = f1q.get("mean")
                f1q_median = f1q.get("median")
                if f1q_median is not None:
                    fidelity_1q = f1q_median
                    fidelity_1q_stat_used = "median"
                elif f1q_mean is not None:
                    fidelity_1q = f1q_mean
                    fidelity_1q_stat_used = "mean (median unavailable)"
            elif isinstance(f1q, (int, float)):
                fidelity_1q = f1q
                fidelity_1q_stat_used = "scalar"
            if fidelity_1q is not None:
                eps_prior  = float(np.clip(1.0 - float(fidelity_1q), 0.0, arch["eps_max"]))
                eps_source = "live_calibration"
                diag_parts = []
                if f1q_median is not None:
                    diag_parts.append(f"median_eps={1.0-float(f1q_median):.2e}")
                if f1q_mean is not None:
                    diag_parts.append(f"mean_eps={1.0-float(f1q_mean):.2e}")
                if diag_parts:
                    print(f"[Canary] IonQ 1Q fidelity stats — using {fidelity_1q_stat_used}: "
                          f"{', '.join(diag_parts)}")

        except Exception:
            print("[Canary] IonQ API unavailable - using arch defaults for trapped_ion")
            return cls.from_architecture("trapped_ion")

        T2_prior = min(T2_prior, 2.0*T1_prior)

        p0_given_1, p1_given_0, spam_source, spam_note = fetch_live_spam(
            "trapped_ion", ionq_api_token=api_token, ionq_backend_name=backend_name,
            qubit_id=qubit_id)

        custom_arch = None
        if eps_source == "live_calibration" or spam_source != "literature_fallback":
            custom_arch = dict(arch)
            custom_arch["eps_typical"] = eps_prior
            custom_arch["p0_given_1"] = p0_given_1
            custom_arch["p1_given_0"] = p1_given_0

        calibration_source = CalibrationSource(
            T1_source=T1_source, T2_source=T2_source,
            eps_source=eps_source, dw_source=dw_source,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            backend=f"ionq:{backend_name}",
            spam_source=spam_source, spam_note=spam_note,
        )

        print(f"[Canary] IonQ live priors — backend {backend_name} qubit {qubit_id}: "
              f"T1={T1_prior:.2f}s ({T1_source}), T2={T2_prior:.2f}s ({T2_source}), "
              f"ε_sx={eps_prior:.2e} ({eps_source})  "
              f"SPAM: p0|1={p0_given_1:.4f}, p1|0={p1_given_0:.4f} ({spam_source})")

        prior_confidence = ("live" if T1_source == "live_calibration"
                            and T2_source == "live_calibration" else "arch_default")
        return cls(architecture="trapped_ion", T1_prior_s=T1_prior,
                   T2_prior_s=T2_prior, dt_ns=arch["dt_ns"],
                   backend_name=f"ionq:{backend_name}", custom_arch=custom_arch,
                   calibration_source=calibration_source,
                   prior_confidence=prior_confidence)
    #Theoretical proof-of-concept as experiment hasn't been tested on neutral atom processors
    @classmethod
    def from_braket_backend(cls,
                            device_arn: str = "arn:aws:braket:us-east-1::device/qpu/quera/Aquila",
                            qubit_id: int = 0) -> "BackendProfile":
        arch = ARCH_DEFAULTS["neutral_atom"]
        try:
            from braket.aws import AwsDevice
            try:
                device = AwsDevice(device_arn)
                _ = device.properties
            except Exception:
                pass
            print("[Canary] Braket/QuEra: T1/T2/ε priors not available from "
                  "provider — using neutral_atom arch defaults")
        except ImportError:
            pass

        p0_given_1, p1_given_0, spam_source, spam_note = fetch_live_spam(
            "neutral_atom", qubit_id=qubit_id)

        calibration_source = CalibrationSource(
            T1_source="arch_default", T2_source="arch_default",
            eps_source="arch_default", dw_source="arch_default",
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            backend=device_arn,
            spam_source=spam_source, spam_note=spam_note,
        )

        print(f"[Canary] Braket/QuEra priors — qubit {qubit_id}: "
              f"SPAM: p0|1={p0_given_1:.4f}, p1|0={p1_given_0:.4f} ({spam_source})")

        return cls(architecture="neutral_atom", T1_prior_s=arch["T1_s"],
                   T2_prior_s=arch["T2_s"], dt_ns=arch["dt_ns"],
                   backend_name=device_arn, calibration_source=calibration_source,
                   prior_confidence="arch_default")

    @classmethod
    def from_architecture(cls, architecture: str, backend_name: str = "unknown",
                          T1_prior_s: Optional[float] = None,
                          T2_prior_s: Optional[float] = None,
                          dw_max_rad_s: Optional[float] = None,
                          eps_typical: Optional[float] = None,
                          gate_time_ns: Optional[float] = None,
                          p0_given_1: float = 0.0,
                          p1_given_0: float = 0.0) -> "BackendProfile":
        if architecture == "custom":
            if T1_prior_s is None or T2_prior_s is None:
                raise ValueError("architecture='custom' requires T1_prior_s and T2_prior_s")
            custom = build_custom_arch(T1_prior_s, T2_prior_s,
                                       dw_max_rad_s=dw_max_rad_s,
                                       eps_typical=eps_typical,
                                       gate_time_ns=gate_time_ns,
                                       p0_given_1=p0_given_1,
                                       p1_given_0=p1_given_0)
            return cls(architecture="custom", T1_prior_s=custom["T1_s"],
                       T2_prior_s=custom["T2_s"], dt_ns=None,
                       backend_name=backend_name, custom_arch=custom,
                       prior_confidence="arch_default")
        defaults = ARCH_DEFAULTS.get(architecture, ARCH_DEFAULTS["unknown"])
        T1 = T1_prior_s or defaults["T1_s"]
        T2 = min(T2_prior_s or defaults["T2_s"], 2.0*T1)

        live_p0, live_p1, spam_source, spam_note = fetch_live_spam(architecture)
        calibration_source = CalibrationSource(
            T1_source="arch_default", T2_source="arch_default",
            eps_source="arch_default", dw_source="arch_default",
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            backend=backend_name,
            spam_source=spam_source, spam_note=spam_note,
        )
       
        return cls(architecture=architecture, T1_prior_s=T1, T2_prior_s=T2,
                   dt_ns=defaults["dt_ns"], backend_name=backend_name,
                   calibration_source=calibration_source,
                   prior_confidence="arch_default")

    @classmethod
    def from_true_params(cls, architecture: str,
                         T1_true_s: float,
                         T2_true_s: float) -> "BackendProfile":
        """
        Constructs a BackendProfile with priors set to the true parameter
        values and prior_confidence="live", so Strategy A (linear delays)
        is used and delays are perfectly centred on the true decay.
        Used exclusively for Fig. 2 (ideal hardware validation) in
        2_parity_experiments.py. Never used for real hardware runs.
        """
        defaults = ARCH_DEFAULTS.get(architecture, ARCH_DEFAULTS["unknown"])
        T1 = T1_true_s
        T2 = min(T2_true_s, 2.0 * T1_true_s)
        live_p0, live_p1, spam_source, spam_note = fetch_live_spam(architecture)
        calibration_source = CalibrationSource(
            T1_source="arch_default", T2_source="arch_default",
            eps_source="arch_default", dw_source="arch_default",
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            backend="from_true_params",
            spam_source=spam_source, spam_note=spam_note)
        return cls(architecture=architecture, T1_prior_s=T1, T2_prior_s=T2,
                   dt_ns=defaults["dt_ns"], backend_name="from_true_params",
                   calibration_source=calibration_source,
                   prior_confidence="live")

    def updated(self, T1_est_s: float, T2_est_s: float) -> "BackendProfile":
        T2_est_s = min(T2_est_s, 2.0*T1_est_s)
        return BackendProfile(architecture=self.architecture, T1_prior_s=T1_est_s,
                              T2_prior_s=T2_est_s, dt_ns=self.dt_ns,
                              backend_name=self.backend_name, custom_arch=self.custom_arch,
                              prior_confidence=self.prior_confidence)


def get_live_profile(backend, qubit_id: int = 0,
                     ionq_token: Optional[str] = None) -> "BackendProfile":
    """
    Auto-detect the backend type and return a BackendProfile with
    live calibration priors where available.

    Accepts:
      - A Qiskit IBMBackend object    → calls from_ibm_backend()
      - The string "ionq:{backend}"  → calls from_ionq_backend()
        e.g. "ionq:forte-1", requires ionq_token
      - A Braket Device ARN string   → calls from_braket_backend()
      - An architecture string       → calls from_architecture()
        e.g. "superconducting", "trapped_ion", "neutral_atom"
    """
    if isinstance(backend, str):
        if backend.startswith("ionq:"):
            ionq_backend_name = backend.split(":", 1)[1]
            if not ionq_token:
                raise ValueError(
                    "get_live_profile: 'ionq_token' is required for 'ionq:' backends")
            return BackendProfile.from_ionq_backend(
                ionq_token, backend_name=ionq_backend_name, qubit_id=qubit_id)
        if backend.startswith("arn:"):
            return BackendProfile.from_braket_backend(device_arn=backend, qubit_id=qubit_id)
        if backend in ARCH_DEFAULTS:
            return BackendProfile.from_architecture(backend)

    is_ibm_backend = False
    try:
        from qiskit.providers import BackendV1, BackendV2
        if isinstance(backend, (BackendV1, BackendV2)):
            is_ibm_backend = True
    except Exception:
        pass
    if not is_ibm_backend:
        try:
            from qiskit_ibm_runtime import IBMBackend
            if isinstance(backend, IBMBackend):
                is_ibm_backend = True
        except Exception:
            pass
    if is_ibm_backend:
        return BackendProfile.from_ibm_backend(backend, qubit=qubit_id)

    raise ValueError(
        "get_live_profile: unrecognized backend input. Accepted inputs are: "
        "a Qiskit IBMBackend object, a string 'ionq:{backend_name}' "
        "(with ionq_token), a Braket device ARN string ('arn:...'), or an "
        f"architecture string in {sorted(ARCH_DEFAULTS.keys())}. "
        f"Got: {backend!r}"
    )


def _snap(delay_s: float, dt_ns: Optional[float]) -> tuple[float, str]:
    if dt_ns is not None and delay_s > 0:
        dt_s = dt_ns*1e-9
        return float(max(1, round(delay_s/dt_s))), "dt"
    return delay_s, "s"


def _delay_seconds(inst, dt_ns: Optional[float]) -> float:
    dur = inst.operation.params[0]
    unit = getattr(inst.operation, "unit", "dt")
    if unit == "s":
        return float(dur)
    elif unit == "ns":
        return float(dur) * 1e-9
    elif unit == "us":
        return float(dur) * 1e-6
    elif unit == "ms":
        return float(dur) * 1e-3
    elif unit == "dt":
        return float(dur) * (dt_ns if dt_ns is not None else 0.2222) * 1e-9
    return float(dur)


def _make_aer_backend_for_circuit(qc, T1_s: float, T2_s: float,
                                   eps_sx: float, p0g1: float, p1g0: float,
                                   gate_time_ns: float, dt_ns: Optional[float]):
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (NoiseModel, thermal_relaxation_error,
                                   depolarizing_error, ReadoutError)

    nm = NoiseModel()

    gate_time_s = gate_time_ns * 1e-9
    gate_err = (thermal_relaxation_error(T1_s, T2_s, gate_time_s)
                .compose(depolarizing_error(2.0 * eps_sx, 1)))
    nm.add_quantum_error(gate_err,
        ["sx", "sxdg", "x", "h", "s", "sdg", "id",
         "rx", "ry", "r", "u", "u1", "u2", "u3", "p"], [0])

    seen: set = set()
    for inst in qc.data:
        if inst.operation.name == "delay":
            ds = _delay_seconds(inst, dt_ns)
            key = round(ds, 15)
            if key not in seen and ds > 1e-12:
                seen.add(key)
                nm.add_quantum_error(
                    thermal_relaxation_error(T1_s, T2_s, ds), ["delay"], [0])

    nm.add_readout_error(
        ReadoutError([[1.0 - p0g1, p0g1], [p1g0, 1.0 - p1g0]]), [0])

    return AerSimulator(noise_model=nm)


def _inject_ramsey_detuning(qc, dw_s: float, dt_ns: Optional[float]):
    from qiskit import QuantumCircuit
    qc_new = QuantumCircuit(qc.num_qubits, qc.num_clbits, name=qc.name)
    for inst in qc.data:
        qc_new.append(inst.operation, inst.qubits, inst.clbits)
        if inst.operation.name == "delay":
            ds = _delay_seconds(inst, dt_ns)
            qc_new.rz(dw_s * ds, 0)
    return qc_new


def run_probe_circuits_aer(
    circuits: list,
    meta: dict,
    T1_s: float,
    T2_s: float,
    eps_sx: float,
    p0g1: float,
    p1g0: float,
    dt_ns: Optional[float],
    shots_t1: int,
    shots_ramsey: int,
    shots_gate: int,
    shots_echo: int,
    dw_s: float = 0.0,
) -> list:
    from qiskit import transpile

    arch_name = meta.get("architecture", "superconducting")
    arch = ARCH_DEFAULTS.get(arch_name, ARCH_DEFAULTS["superconducting"])
    gate_time_ns = arch.get("gate_time_ns", 50.0)

    n_t1     = meta["n_t1"]
    n_ramsey = meta["n_ramsey"]
    n_gate   = meta["n_gate"]

    shots_per_circuit = (
        [shots_t1]     * n_t1 +
        [shots_ramsey] * n_ramsey +
        [shots_gate]   * n_gate +
        [shots_echo]   * (len(circuits) - n_t1 - n_ramsey - n_gate)
    )

    counts_list = []
    for qc, sh in zip(circuits, shots_per_circuit):
        run_qc = (_inject_ramsey_detuning(qc, dw_s, dt_ns)
                  if "ramsey" in qc.name else qc)
        backend = _make_aer_backend_for_circuit(
            run_qc, T1_s, T2_s, eps_sx, p0g1, p1g0, gate_time_ns, dt_ns)
        tqc = transpile(run_qc, backend, optimization_level=0)
        raw = backend.run(tqc, shots=sh).result().get_counts()
        norm: dict = {}
        for bitstring, cnt in raw.items():
            b = bitstring.replace(" ", "")[-1]
            norm[b] = norm.get(b, 0) + cnt
        counts_list.append(norm)

    return counts_list


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


def _build_echo_circuit(delay_s: float, dt_ns: Optional[float], idx: int):
    from qiskit import QuantumCircuit
    half = delay_s / 2.0
    dur1, unit1 = _snap(half, dt_ns)
    dur2, unit2 = _snap(half, dt_ns)
    qc = QuantumCircuit(1, 1, name=f"echo_{idx}")
    qc.h(0)
    if half > 0:
        qc.delay(dur1, 0, unit=unit1)
    qc.x(0)
    if half > 0:
        qc.delay(dur2, 0, unit=unit2)
    qc.h(0)
    qc.measure(0, 0)
    return qc


def _sqrtx_native_inverse_pair(qc, q, architecture: str):
    """
    Applies one native (G, G^-1) pair on qubit q, where G is the
    architecture's physical sqrt-X-equivalent single-qubit gate:
      - trapped_ion: GPI2(0), GPI2(0.5)  (native IonQ pulses; GPI2(0.5) is
        verified equal to GPI2(0)^dagger, so the pair nets to identity
        exactly as sx/sxdg does)
      - all other architectures: sx, sxdg (Qiskit standard gate, native on
        IBM superconducting hardware; used as the generic stand-in for
        neutral_atom/unknown, matching pre-existing behaviour)
    """
    if architecture == "trapped_ion":
        try:
            from qiskit_ionq.ionq_gates import GPI2Gate
            qc.append(GPI2Gate(0.0), [q])
            qc.append(GPI2Gate(0.5), [q])
        except ImportError:
            qc.r(np.pi / 2, 0.0, q)
            qc.r(np.pi / 2, np.pi, q)
    else:
        qc.sx(q)
        qc.sxdg(q)


def _build_gate_rep_circuit(N: int, architecture: str = "superconducting"):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1, 1, name=f"gate_rep_N{N}")
    for _ in range(N):
        _sqrtx_native_inverse_pair(qc, 0, architecture)
    qc.measure(0, 0)
    return qc


def split_qubits_for_op_limit(target_qubits: list, ops_per_qubit: int,
                              max_total_ops: int = 24_000) -> list:
    """
    Proactively splits target_qubits into batches so that
    ops_per_qubit * len(batch) never exceeds max_total_ops.

    This avoids ever submitting a circuit that would be rejected for
    exceeding a hardware provider's per-job operation-count limit —
    computed BEFORE submission, so no job is ever charged and then
    fails. Each returned batch is a literal subset of target_qubits
    (order preserved), so physical qubit identity is never altered —
    "qubit 7" in a split batch is exactly the same physical qubit as
    "qubit 7" in the unsplit case.

    max_total_ops=24_000 is a conservative default based on IonQ Forte
    QuantumCircuitComplexityError behaviour observed empirically:
    16 qubits x 1252 ops/qubit = 20,032 ops succeeded;
    16 qubits x 2500 ops/qubit = 40,000 ops failed.
    24,000 sits safely below the failure threshold with margin.
    """
    if ops_per_qubit <= 0 or len(target_qubits) == 0:
        return [list(target_qubits)]
    max_qubits_per_batch = max(1, int(max_total_ops // ops_per_qubit))
    n_qubits = len(target_qubits)
    if max_qubits_per_batch >= n_qubits:
        return [list(target_qubits)]

    num_batches = -(-n_qubits // max_qubits_per_batch)  # ceil division
    base_size   = n_qubits // num_batches
    remainder   = n_qubits % num_batches

    batches = []
    idx = 0
    for b in range(num_batches):
        size = base_size + (1 if b < remainder else 0)
        batches.append(list(target_qubits[idx:idx + size]))
        idx += size
    return batches


def build_probe_circuits(profile: BackendProfile) -> tuple[list, dict]:
    t1_delays     = profile.t1_delays_s
    ramsey_delays = profile.ramsey_delays_s
    echo_delays   = profile.echo_delays_s
    circuits      = []

    for i, t in enumerate(t1_delays):
        circuits.append(_build_t1_circuit(t, profile.dt_ns, i))

    for i, t in enumerate(ramsey_delays):
        circuits.append(_build_ramsey_x_circuit(t, profile.dt_ns, i))
        circuits.append(_build_ramsey_y_circuit(t, profile.dt_ns, i))

    gate_rep_n = profile.gate_rep_n
    for N in gate_rep_n:
        circuits.append(_build_gate_rep_circuit(N, profile.architecture))

    for i, t in enumerate(echo_delays):
        circuits.append(_build_echo_circuit(t, profile.dt_ns, i))

    n_t1     = len(t1_delays)
    n_ramsey = 2 * len(ramsey_delays)
    n_gate   = len(gate_rep_n)
    n_echo   = len(echo_delays)

    metadata = {
        "t1_delays_s":    t1_delays,
        "ramsey_delays_s": ramsey_delays,
        "gate_rep_N":     gate_rep_n,
        "echo_delays_s":  echo_delays,
        "n_t1":           n_t1,
        "n_ramsey":       n_ramsey,
        "n_gate":         n_gate,
        "n_echo":         n_echo,
        "dw_max_rad_s":   profile.dw_max_rad_s,
        "architecture":   profile.architecture,
        "backend_name":   profile.backend_name,
        "T1_prior_s":     profile.T1_prior_s,
        "T2_prior_s":     profile.T2_prior_s,
        "dt_ns":          profile.dt_ns,
    }
    return circuits, metadata


def forward_t1(tau_s, T1_s: float, p0_given_1: float = 0.0, p1_given_0: float = 0.0):
    p_ideal = np.exp(-np.asarray(tau_s, dtype=float) / T1_s)
    return p_ideal * (1.0 - p0_given_1) + (1.0 - p_ideal) * p1_given_0


def forward_ramsey_xy(t: float, T2_s: float, delta_omega: float):
    decay = np.exp(-t / T2_s)
    p1_x  = 0.5 * (1.0 - decay * np.cos(delta_omega * t))
    p1_y  = 0.5 * (1.0 - decay * np.sin(delta_omega * t))
    return p1_x, p1_y


def forward_echo(tau_s, T2_echo_s: float, p0_given_1: float = 0.0, p1_given_0: float = 0.0):
    p_ideal = 0.5 * (1.0 - np.exp(-np.asarray(tau_s, dtype=float) / T2_echo_s))
    return p_ideal * (1.0 - p0_given_1) + (1.0 - p_ideal) * p1_given_0


def forward_gate(N, epsilon_sx: float, p0_given_1: float = 0.0, p1_given_0: float = 0.0):
    N = np.asarray(N, dtype=float)
    p_ideal = 0.5 * (1.0 + (1.0 - 2.0*epsilon_sx)**(2.0*N))
    return p_ideal * (1.0 - p0_given_1) + (1.0 - p_ideal) * p1_given_0


def _invert_t1(p1_measured: np.ndarray, tau_s: np.ndarray,
               T1_prior_s: float, arch: dict, shots_t1: int) -> tuple[float, float, float]:
    p0_given_1 = arch.get("p0_given_1", 0.0)
    p1_given_0 = arch.get("p1_given_0", 0.0)
    spam_model = lambda tau_s, T1_s: forward_t1(tau_s, T1_s, p0_given_1, p1_given_0)

    denom    = float(p1_measured[0]) - float(p1_measured[1])
    T1_guess = T1_prior_s
    if abs(denom) > 1e-6:
        x = (float(p1_measured[1]) - float(p1_measured[2])) / denom
        if 0 < x < 1:
            T1_guess = float(np.clip(-tau_s[0]/np.log(x),
                                     arch["T1_min_s"], arch["T1_max_s"]))

    p_guess = spam_model(tau_s, T1_guess)
    fit_sigma = np.maximum(np.sqrt(np.clip(p_guess*(1.0-p_guess), 0, None)/shots_t1), 1e-3)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            popt, pcov = curve_fit(spam_model, tau_s, p1_measured,
                                   p0=[T1_guess],
                                   sigma=fit_sigma, absolute_sigma=True,
                                   bounds=([arch["T1_min_s"]], [arch["T1_max_s"]]),
                                   maxfev=10_000)
        T1    = float(popt[0])
        sigma = float(np.sqrt(np.diag(pcov))[0])
        resid = float(np.sum((p1_measured - spam_model(tau_s, T1))**2))
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

    t1     = float(ramsey_delays[0])
    angle  = float(np.arctan2(y_means[0], x_means[0]))
    dw     = angle / t1
    amp_t1 = float(amps[0])
    # Cramér-Rao lower bound check: below this amplitude the angle estimate
    # carries no reliable information at this shot count, so mark it
    # infinite rather than papering over it with an arbitrary amplitude
    # floor — downstream chi2-weighted combination then discounts it
    # naturally instead of silently trusting a noisy angle.
    if amp_t1 < 3.0 / np.sqrt(max(shots_ramsey, 1)):
        sigma_angle = np.inf
    else:
        sigma_angle = 1.0 / (amp_t1 * np.sqrt(max(shots_ramsey, 1)))
    sdw    = float(np.clip(sigma_angle / t1, 0, np.inf))

    xy_meas = np.concatenate([p1_x_arr, p1_y_arr])
    def xy_model(_, T2_s):
        px_pred, py_pred = forward_ramsey_xy(ramsey_delays, T2_s, dw)
        return np.concatenate([px_pred, py_pred])

    p_guess   = xy_model(None, T2_guess)
    fit_sigma = np.maximum(np.sqrt(np.clip(p_guess*(1.0-p_guess), 0, None)/shots_ramsey), 1e-3)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            popt, pcov = curve_fit(xy_model, ramsey_delays, xy_meas,
                                   p0=[T2_guess],
                                   sigma=fit_sigma, absolute_sigma=True,
                                   bounds=([T2_min], [T2_max]),
                                   maxfev=10_000)
        T2    = float(popt[0])
        sT2   = float(np.sqrt(np.diag(pcov))[0])
        resid = float(np.sum((xy_meas - xy_model(None, T2))**2))
    except Exception:
        T2, sT2, resid = T2_guess, np.inf, np.inf

    return T2, sT2, dw, sdw, resid


def _invert_gate(p0_measured: np.ndarray, N_values: np.ndarray,
                  arch: dict, shots_gate: int) -> tuple[float, float, float]:
    p0_given_1 = arch.get("p0_given_1", 0.0)
    p1_given_0 = arch.get("p1_given_0", 0.0)
    spam_model = lambda N_values, epsilon_sx: forward_gate(N_values, epsilon_sx, p0_given_1, p1_given_0)

    if len(p0_measured) < 2:
        return arch["eps_typical"], float("inf"), float("inf")

    denom    = float(p0_measured[0]) - float(p0_measured[1])
    eps_guess = arch["eps_typical"]
    if len(p0_measured) >= 3 and abs(denom) > 1e-6:
        x = (float(p0_measured[1]) - float(p0_measured[2])) / denom
        if 0 < x < 1:
            eps_guess = float(np.clip((1 - x**(1.0/(2.0*N_values[0])))/2.0,
                                      0.0, arch["eps_max"]))
    p_guess = spam_model(N_values, eps_guess)
    fit_sigma = np.maximum(np.sqrt(np.clip(p_guess*(1.0-p_guess), 0, None)/shots_gate), 1e-3)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            popt, pcov = curve_fit(spam_model, N_values, p0_measured,
                                   p0=[eps_guess],
                                   sigma=fit_sigma, absolute_sigma=True,
                                   bounds=([0.0], [arch["eps_max"]]),
                                   maxfev=10_000)
        eps   = float(popt[0])
        sigma = float(np.sqrt(np.diag(pcov))[0])
        resid = float(np.sum((p0_measured - spam_model(N_values, eps))**2))
    except Exception:
        eps, sigma, resid = eps_guess, np.inf, np.inf
    return eps, sigma, resid


def _invert_echo(p1_measured: np.ndarray, tau_s: np.ndarray,
                 T1_prior_s: float, arch: dict,
                 T1_estimate_s: float, shots_echo: int) -> tuple[float, float, float]:
    p0_given_1 = arch.get("p0_given_1", 0.0)
    p1_given_0 = arch.get("p1_given_0", 0.0)
    spam_model = lambda tau_s, T2_echo_s: forward_echo(tau_s, T2_echo_s, p0_given_1, p1_given_0)

    denom     = float(p1_measured[0]) - float(p1_measured[1])
    T2_min    = arch["T2_min_s"]
    T2_max    = 2.0 * T1_estimate_s
    T2_guess  = float(np.clip(T2_max * 0.5, T2_min, T2_max))
    if abs(denom) > 1e-6:
        x = (float(p1_measured[1]) - float(p1_measured[2])) / denom
        if 0 < x < 1:
            T2_guess = float(np.clip(-tau_s[0]/np.log(x), T2_min, T2_max))

    p_guess = spam_model(tau_s, T2_guess)
    fit_sigma = np.maximum(np.sqrt(np.clip(p_guess*(1.0-p_guess), 0, None)/shots_echo), 1e-3)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            popt, pcov = curve_fit(spam_model, tau_s, p1_measured,
                                   p0=[T2_guess],
                                   sigma=fit_sigma, absolute_sigma=True,
                                   bounds=([T2_min], [T2_max]),
                                   maxfev=10_000)
        T2_echo    = float(popt[0])
        sigma      = float(np.sqrt(np.diag(pcov))[0])
        resid      = float(np.sum((p1_measured - spam_model(tau_s, T2_echo))**2))
    except Exception:
        T2_echo, sigma, resid = T2_guess, np.inf, np.inf
    return T2_echo, sigma, resid


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
    T2_ramsey_s:        float = 0.0
    T2_ramsey_sigma_s:  float = 0.0
    T2_echo_s:          float = 0.0
    T2_echo_sigma_s:    float = 0.0
    echo_residual:      float = 0.0
    echo_chi2_dof:      float = 0.0
    calibration_source: Optional[CalibrationSource] = None
    prior_warning:      str = ""

    def summary(self, arch: Optional[dict] = None) -> str:
        if arch is None:
            arch = ARCH_DEFAULTS["superconducting"]
        s  = arch["time_scale"]; u = arch["display_unit"]
        dw = self.delta_omega / (2*np.pi*1e3)
        line = (f"Qubit {self.qubit_id} | "
                f"T1={self.T1_s*s:.2f}±{self.T1_sigma_s*s:.2f}{u}  "
                f"T2={self.T2_s*s:.2f}±{self.T2_sigma_s*s:.2f}{u}  "
                f"Δω={dw:.3f}kHz  ε={self.epsilon_sx:.2e}")
        if self.calibration_source is not None:
            cs = self.calibration_source
            line += (f"  | priors: T1={cs.T1_source}, T2={cs.T2_source}, "
                     f"ε={cs.eps_source}, Δω={cs.dw_source} "
                     f"(pulled {cs.timestamp} from {cs.backend})")
            if cs.spam_source != "literature_fallback":
                line += f"  | SPAM: live ({cs.spam_source})"
            else:
                line += "  | SPAM: literature defaults"
        if self.prior_warning:
            line += f"\n{self.prior_warning}"
        return line


def lindblad_inversion(counts_list: list[dict],
                        metadata:    dict,
                        profile:     BackendProfile,
                        shots_t1:    int = 300,
                        shots_ramsey:int = 1000,
                        shots_gate:  int = 500,
                        shots_echo:  int = 500,
                        qubit_id:    int = 0,
                        timestamp:   str = "") -> InversionResult:
    arch     = profile.constants
    n_t1     = metadata["n_t1"]
    n_ramsey = metadata["n_ramsey"]
    n_gate   = metadata["n_gate"]
    n_echo   = metadata["n_echo"]
    expected = n_t1 + n_ramsey + n_gate + n_echo
    if len(counts_list) != expected:
        raise ValueError(f"Expected {expected} count dicts, got {len(counts_list)}")

    t1_counts     = counts_list[:n_t1]
    ramsey_counts = counts_list[n_t1:n_t1+n_ramsey]
    gate_counts   = counts_list[n_t1+n_ramsey:n_t1+n_ramsey+n_gate]
    echo_counts   = counts_list[n_t1+n_ramsey+n_gate:]

    p1_t1 = np.array([c.get("1",0)/shots_t1 for c in t1_counts])
    t1d   = np.array(metadata["t1_delays_s"])
    T1, sT1, r_t1 = _invert_t1(p1_t1, t1d, profile.T1_prior_s, arch, shots_t1)

    t1_chi2 = _chi2_per_dof(p1_t1,
        forward_t1(t1d, T1, arch.get("p0_given_1", 0.0), arch.get("p1_given_0", 0.0)),
        shots_t1, n_params=1)

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
    ramsey_chi2 = _chi2_per_dof(ramsey_meas, ramsey_pred, shots_ramsey, n_params=1)

    p0_gate = np.array([c.get("0",0)/shots_gate for c in gate_counts])
    N_vals  = np.array(metadata["gate_rep_N"], dtype=float)
    eps, seps, r_gate = _invert_gate(p0_gate, N_vals, arch, shots_gate)
    gate_chi2 = _chi2_per_dof(p0_gate,
        forward_gate(N_vals, eps, arch.get("p0_given_1", 0.0), arch.get("p1_given_0", 0.0)),
        shots_gate, n_params=1)

    p1_echo   = np.array([c.get("1",0)/shots_echo for c in echo_counts])
    echo_delays = np.array(metadata["echo_delays_s"])
    T2_echo, sT2_echo, r_echo = _invert_echo(
        p1_echo, echo_delays, profile.T2_prior_s, arch,
        T1_estimate_s=T1, shots_echo=shots_echo)
    echo_chi2 = _chi2_per_dof(p1_echo,
        forward_echo(echo_delays, T2_echo, arch.get("p0_given_1", 0.0), arch.get("p1_given_0", 0.0)),
        shots_echo, n_params=1)

    ramsey_chi2_factor = max(ramsey_chi2, 1.0) if np.isfinite(ramsey_chi2) else 1.0
    echo_chi2_factor   = max(echo_chi2, 1.0) if np.isfinite(echo_chi2) else 1.0
    w_ramsey = 1.0/(sT2**2 * ramsey_chi2_factor) if np.isfinite(sT2) and sT2 > 0 else 0.0
    w_echo   = 1.0/(sT2_echo**2 * echo_chi2_factor) if np.isfinite(sT2_echo) and sT2_echo > 0 else 0.0
    if w_ramsey == 0.0 and w_echo == 0.0:
        T2_combined, sT2_combined = T2, sT2
    elif w_ramsey == 0.0:
        T2_combined, sT2_combined = T2_echo, sT2_echo
    elif w_echo == 0.0:
        T2_combined, sT2_combined = T2, sT2
    else:
        T2_combined = (w_ramsey*T2 + w_echo*T2_echo) / (w_ramsey + w_echo)
        sT2_combined = float(np.sqrt(1.0/(w_ramsey + w_echo)))

    prior_warning = ""
    if profile.prior_confidence == "arch_default":
        chi2_dof_checks = [
            ("T1", t1_chi2, t1d, "T1_prior_s"),
            ("Ramsey/T2", ramsey_chi2, ramsey_delays, "T2_prior_s"),
            ("Echo/T2", echo_chi2, echo_delays, "T2_prior_s"),
            ("Gate-rep/eps", gate_chi2, N_vals, "eps_typical"),
        ]
        messages = []
        for label, chi2, delays_arr, prior_name in chi2_dof_checks:
            if np.isfinite(chi2) and chi2 > 7.0:
                vals = [float(v) for v in np.asarray(delays_arr)]
                messages.append(
                    f"WARNING: {label} chi2/dof={chi2:.1f} >> 1. Delays may not "
                    f"span the true timescale. Consider re-running with updated "
                    f"{prior_name}. Current values: {vals}.")
        if messages:
            prior_warning = "\n".join(messages)
            for msg in messages:
                print(f"[Canary] {msg}")

    return InversionResult(
        backend_name=metadata.get("backend_name", profile.backend_name),
        qubit_id=qubit_id, timestamp=timestamp, architecture=profile.architecture,
        T1_s=T1, T1_sigma_s=sT1,
        T2_s=T2_combined, T2_sigma_s=sT2_combined,
        delta_omega=dw, delta_omega_sigma=sdw,
        epsilon_sx=eps, epsilon_sx_sigma=seps,
        t1_residual=r_t1, ramsey_residual=r_ram, gate_residual=r_gate,
        t1_chi2_dof=t1_chi2, ramsey_chi2_dof=ramsey_chi2, gate_chi2_dof=gate_chi2,
        T2_ramsey_s=T2, T2_ramsey_sigma_s=sT2,
        T2_echo_s=T2_echo, T2_echo_sigma_s=sT2_echo,
        echo_residual=r_echo, echo_chi2_dof=echo_chi2,
        calibration_source=profile.calibration_source,
        prior_warning=prior_warning)


if __name__ == "__main__":
    print("=" * 70)
    print("Canary self-test: live calibration priors (get_live_profile)")
    print("=" * 70)

    print("\n[1] get_live_profile('superconducting', qubit_id=0)")
    sc_profile = get_live_profile("superconducting", qubit_id=0)
    print(sc_profile)
    print("calibration_source:", sc_profile.calibration_source)

    print("\n[2] get_live_profile('trapped_ion')")
    ti_profile = get_live_profile("trapped_ion")
    print(ti_profile)
    print("calibration_source:", ti_profile.calibration_source)
    print(f"trapped_ion t1_delays_s (log-spaced [0.01, 100000]): {ti_profile.t1_delays_s}")
    print(f"trapped_ion ramsey_delays_s: {ti_profile.ramsey_delays_s}")

    na_profile = get_live_profile("neutral_atom")
    print(f"neutral_atom t1_delays_s (log-spaced [0.001, 100000]): {na_profile.t1_delays_s}")
    print(f"neutral_atom ramsey_delays_s: {na_profile.ramsey_delays_s}")

    print("\n[2b] gate_rep_n for all three architectures")
    for arch_name in ("superconducting", "trapped_ion", "neutral_atom"):
        p = get_live_profile(arch_name)
        print(f"{arch_name}: eps_typical={p.constants['eps_typical']:.2e} "
              f"gate_rep_n={p.gate_rep_n}")

    print("\n[3] get_live_profile('ionq:forte-1', ionq_token='dummy_token')"
          " — expect graceful fallback")
    ionq_profile = get_live_profile("ionq:forte-1", ionq_token="dummy_token")
    print(ionq_profile)
    print("calibration_source:", ionq_profile.calibration_source)

    print("\n[4] build_probe_circuits() + lindblad_inversion() end-to-end "
          "with a get_live_profile() profile (synthetic all-zero counts)")
    probe_profile = get_live_profile("superconducting", qubit_id=0)
    circuits, metadata = build_probe_circuits(probe_profile)
    n_circuits = len(circuits)
    print(f"built {n_circuits} probe circuits")

    zero_counts = [{"0": 300, "1": 0} for _ in range(n_circuits)]
    result = lindblad_inversion(
        zero_counts, metadata, probe_profile,
        shots_t1=300, shots_ramsey=300, shots_gate=300, shots_echo=300,
        qubit_id=0, timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    print(result.summary(arch=probe_profile.constants))

    print("\n[4b] build_probe_circuits() + lindblad_inversion() on all three "
          "architectures with synthetic all-zero counts")
    for arch_name in ("superconducting", "trapped_ion", "neutral_atom"):
        p = get_live_profile(arch_name)
        circs, meta = build_probe_circuits(p)
        zc = [{"0": 300, "1": 0} for _ in range(len(circs))]
        res = lindblad_inversion(
            zc, meta, p, shots_t1=300, shots_ramsey=300, shots_gate=300,
            shots_echo=300, qubit_id=0,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
        assert np.isfinite(res.T1_s) and np.isfinite(res.T2_s), \
            f"{arch_name}: non-finite T1/T2"
        print(f"{arch_name}: {len(circs)} circuits, T1={res.T1_s:.4g}s "
              f"T2={res.T2_s:.4g}s (finite: OK)")

    print("\n" + "=" * 70)
    print("Canary self-test: live SPAM ingestion (fetch_live_spam)")
    print("=" * 70)

    print("\n[5] fetch_live_spam('superconducting', backend=None)"
          " — expect literature_fallback")
    p0, p1, src, note = fetch_live_spam("superconducting", backend=None)
    print(f"p0|1={p0:.4f}, p1|0={p1:.4f}, source={src}\nnote: {note}")

    print("\n[6] fetch_live_spam('trapped_ion', ionq_api_token='dummy')"
          " — expect graceful fallback")
    p0, p1, src, note = fetch_live_spam("trapped_ion", ionq_api_token="dummy")
    print(f"p0|1={p0:.4f}, p1|0={p1:.4f}, source={src}\nnote: {note}")

    print("\n[7] fetch_live_spam('neutral_atom') — expect literature_fallback")
    p0, p1, src, note = fetch_live_spam("neutral_atom")
    print(f"p0|1={p0:.4f}, p1|0={p1:.4f}, source={src}\nnote: {note}")

    print("\n[8] IonQ ratio-rescaling formula check "
          "(simulated spam_error=0.003305)")
    spam_error = 0.003305
    ratio = _MAI_2024_P0_GIVEN_1 / _MAI_2024_P1_GIVEN_0
    p0_sim = spam_error * 2.0 * ratio / (1.0 + ratio)
    p1_sim = spam_error * 2.0 / (1.0 + ratio)
    print(f"spam_error={spam_error}, ratio={ratio:.4f} -> "
          f"p0|1={p0_sim:.6f}, p1|0={p1_sim:.6f}")
    print(f"check: (p0+p1)/2 = {(p0_sim+p1_sim)/2:.6f} (should equal spam_error)")

    print("\nAll self-tests completed without crashing.")