"""Terminal rendering of a CanaryReport (no third-party dependencies).

Colour is used only when writing to a terminal and NO_COLOR is not set
(https://no-color.org); render_report(color=False) gives plain text.
"""
from __future__ import annotations

import math
import os
import re
import sys
from datetime import datetime
from typing import List, Optional

__all__ = ["render_report", "render_drift", "banner"]

_BANNER = r"""
║  ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗██╗   ██╗███╗   ███╗  ║
║ ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██║   ██║████╗ ████║  ║
║ ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║   ██║   ██║██╔████╔██║  ║
║ ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║   ██║   ██║██║╚██╔╝██║  ║
║ ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║   ╚██████╔╝██║ ╚═╝ ██║  ║
║  ╚════▀▀  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝  ║
║                                                                    ║
║   ██████╗ █████╗ ███╗   ██╗ █████╗ ██████╗ ██╗   ██╗               ║
║  ██╔════╝██╔══██╗████╗  ██║██╔══██╗██╔══██╗╚██╗ ██╔╝               ║
║  ██║     ███████║██╔██╗ ██║███████║██████╔╝ ╚████╔╝                ║
║  ██║     ██╔══██║██║╚██╗██║██╔══██║██╔══██╗  ╚██╔╝                 ║
║  ╚██████╗██║  ██║██║ ╚████║██║  ██║██║  ██║   ██║                  ║
║   ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝                  ║
║                                                                    ║
║           (o>            QUANTUM CANARY  v{version}                    ║
║        \\_//)           qubit health checks before,                ║
║         \_/_)            during and after your job                 ║
║          _|_                                                       ║
║                                                                    ║
"""


def banner() -> str:
    """The startup banner."""
    from . import __version__
    return _BANNER.format(version=__version__).rstrip("\n")

_WIDTH = 78

_ANSI = {"bold": "1", "dim": "2", "green": "32", "yellow": "33", "red": "31",
         "cyan": "36"}
_HEALTH_COLOR = {"GREEN": "green", "YELLOW": "yellow", "RED": "red"}
_HEALTH_MEANING = {
    "GREEN": "every probe fits the model; T2* and T2 agree",
    "YELLOW": "a probe misfits the model (chi2/dof above its limit) or T2* and T2 "
              "disagree by > 20%: non-Markovian or coherent noise, or significantly stale priors",
    "RED": "a probe fit failed to converge",
}
_SOURCE_LABEL = {
    "live_calibration": "live", "arch_default": "arch default",
    "updated_from_estimate": "updated", "live_asymmetric": "live",
    "live_magnitude_literature_ratio": "live (literature split)",
    "literature_fallback": "literature default", "user_override": "user override",
    "unknown": "unknown",
}


def _use_color(color: Optional[bool]) -> bool:
    if color is not None:
        return color
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


class _Style:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def __call__(self, text: str, *styles: str) -> str:
        if not self.enabled or not styles:
            return text
        codes = ";".join(_ANSI[s] for s in styles)
        return f"\033[{codes}m{text}\033[0m"


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _time_units(value_s) -> tuple:
    """(scale, unit) that shows a time in seconds with 1-3 integer digits."""
    v = abs(float(value_s)) if _finite(value_s) else 0.0
    if v >= 1.0:
        return 1.0, "s"
    if v >= 1e-3:
        return 1e3, "ms"
    return 1e6, "µs"


def _time(value_s) -> str:
    if not _finite(value_s):
        return "—"
    scale, unit = _time_units(value_s)
    return f"{float(value_s) * scale:.4g} {unit}"


def _dw_units(architecture: str):
    # Detuning shown as delta_omega/2pi: kHz for superconducting, Hz for ions.
    return (1e3, "kHz") if architecture != "trapped_ion" else (1.0, "Hz")


def _pm(value, sigma, scale: float, signed: bool = False, sci: bool = False) -> str:
    """value ± sigma with sigma to 2 significant figures and value rounded to
    the same decimal place (scientific notation for sci=True)."""
    if not _finite(value):
        return "—  (fit failed)"
    v = float(value) * scale
    sign = "+" if signed else ""
    if sci:
        if not _finite(sigma):
            return f"{v:{sign}.3e} ± ∞"
        return f"{v:{sign}.3e} ± {float(sigma) * scale:.1e}"
    if not _finite(sigma):
        return f"{v:{sign}.4g} ± ∞"
    sg = abs(float(sigma) * scale)
    if sg == 0.0:
        return f"{v:{sign}.4g} ± 0"
    decimals = max(0, 1 - int(math.floor(math.log10(sg))))
    return f"{v:{sign}.{decimals}f} ± {sg:.{decimals}f}"


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _rule(s: _Style, title: str = "", char: str = "─") -> str:
    """A full-width rule, optionally titled; title may already be styled."""
    if not title:
        return s(char * _WIDTH, "dim")
    visible = len(_ANSI_RE.sub("", title))
    return (s(char * 2, "dim") + " " + title + " "
            + s(char * max(0, _WIDTH - visible - 4), "dim"))


def _kv(s: _Style, key: str, value: str) -> str:
    return f"  {s(f'{key:<12}', 'dim')}{value}"


def _prior_lines(s: _Style, q, mode: str, regime: Optional[str]) -> List[str]:
    p = q.profile
    priors = f"T1 {_time(p.T1_prior_s)} · T2 {_time(p.T2_prior_s)}"
    const = p.constants
    spam = f"p0|1 {const.get('p0_given_1', 0.0):.4f} · p1|0 {const.get('p1_given_0', 0.0):.4f}"
    lines = []
    if mode == "simulation":
        age = "1 h old, ±" if regime == "nisq" else "fresh, ±"
        from . import simulation as sim
        frac = sim.resolve_calibration_perturbation(p.architecture, regime or "nisq")
        lines.append(_kv(s, "Priors", s("SIMULATED CALIBRATION RECORD", "cyan")
                         + s(f"  ({age}{frac:.0%} of truth)", "dim")))
    elif p.prior_confidence == "live":
        lines.append(_kv(s, "Priors", s("LIVE", "green")
                         + s("  (read from the backend's calibration)", "dim")))
    elif p.prior_confidence == "arch_default":
        lines.append(_kv(s, "Priors", s("ARCHITECTURE DEFAULTS", "yellow")
                         + s("  (backend T1/T2 unavailable: delays log-spaced "
                             "over the plausible range)", "dim")))
    else:
        lines.append(_kv(s, "Priors", s("SUPPLIED", "cyan")
                         + s(f"  ({p.prior_confidence})", "dim")))
    cs = p.calibration_source
    if cs is not None and mode == "hardware":
        lab = lambda v: _SOURCE_LABEL.get(v, v)
        lines.append(_kv(s, "", s(f"sources: T1 {lab(cs.T1_source)} · T2 {lab(cs.T2_source)} · "
                                  f"ε {lab(cs.eps_source)} · SPAM {lab(cs.spam_source)}", "dim")))
    lines.append(_kv(s, "", priors))
    lines.append(_kv(s, "", "SPAM  " + spam))
    return lines


def _param_rows(q, with_truth: bool):
    r, arch = q.result, q.profile.architecture
    dw_div, dw_unit = _dw_units(arch)
    dw_scale = 1.0 / (2.0 * math.pi * dw_div)
    truth = q.truth or {}
    # One time unit per parameter, chosen from its estimate (or its truth).
    t1_scale, t1_unit = _time_units(r.T1_s if _finite(r.T1_s) else truth.get("T1_s"))
    t2_scale, t2_unit = _time_units(r.T2_s if _finite(r.T2_s) else truth.get("T2_s"))
    tr_scale, tr_unit = _time_units(r.T2_ramsey_s)
    rows = [
        ("T1", _pm(r.T1_s, r.T1_sigma_s, t1_scale), t1_unit, r.t1_chi2_dof,
         truth.get("T1_s"), r.T1_s, t1_scale),
        ("T2 (echo)", _pm(r.T2_s, r.T2_sigma_s, t2_scale), t2_unit, r.echo_chi2_dof,
         truth.get("T2_s"), r.T2_s, t2_scale),
        ("T2* (Ramsey)", _pm(r.T2_ramsey_s, r.T2_ramsey_sigma_s, tr_scale), tr_unit,
         r.ramsey_chi2_dof, None, None, None),
        ("Δω/2π", _pm(r.delta_omega, r.delta_omega_sigma, dw_scale, signed=True), dw_unit,
         None, truth.get("delta_omega"), r.delta_omega, dw_scale),
        ("ε_sx", _pm(r.epsilon_sx, r.epsilon_sx_sigma, 1.0, sci=True), "", r.gate_chi2_dof,
         truth.get("epsilon_sx"), r.epsilon_sx, 1.0),
    ]
    out = []
    for name, est, unit, chi2, tv, cv, scale in rows:
        chi = f"{chi2:.2f}" if _finite(chi2) else ""
        true_s = err_s = ""
        if with_truth and tv is not None and scale is not None:
            fmt = ".3e" if name == "ε_sx" else (("+.4g") if name.startswith("Δω") else ".4g")
            true_s = f"{float(tv) * scale:{fmt}}"
            if _finite(cv) and float(tv) != 0.0:
                err_s = f"{100.0 * (float(cv) - float(tv)) / abs(float(tv)):+.1f}%"
        out.append((name, est, unit, chi, true_s, err_s))
    return out


def _qubit_block(s: _Style, q, mode: str, regime: Optional[str]) -> List[str]:
    color = _HEALTH_COLOR.get(q.health, "bold")
    lines = [_rule(s, s(f"Qubit {q.qubit}", "bold") + "  " + s(f"● {q.health}", color, "bold"))]
    lines.extend(_prior_lines(s, q, mode, regime))
    with_truth = q.truth is not None
    lines.append("")
    if with_truth:
        header = f"  {'Parameter':<14}{'Canary estimate':<28}{'Unit':<6}{'χ²/dof':>7}  {'True':>11}  {'Error':>7}"
    else:
        header = f"  {'Parameter':<14}{'Canary estimate':<28}{'Unit':<6}{'χ²/dof':>7}"
    lines.append(s(header, "dim"))
    for name, est, unit, chi, true_s, err_s in _param_rows(q, with_truth):
        row = f"  {name:<14}{est:<28}{unit:<6}{chi:>7}"
        if with_truth:
            row += f"  {true_s:>11}  {err_s:>7}"
        lines.append(row)
    warnings = [ln for ln in (q.result.prior_warning or "").splitlines() if ln.strip()]
    for w in warnings:
        lines.append("  " + s("⚠ " + w.replace("WARNING: ", ""), "yellow"))
    return lines


def _summary_table(s: _Style, report) -> List[str]:
    if not report.qubits:
        return []
    arch = report.architecture
    dw_div, dw_unit = _dw_units(arch)
    lines = [_rule(s, s("Summary", "bold")),
             s(f"  {'Qubit':<7}{'Health':<9}{'T1':>12}{'T2':>12}"
               f"{'T2*':>12}{'Δω/2π ' + dw_unit:>13}{'ε_sx':>12}", "dim")]
    for q in report.qubits:
        r = q.result
        dw = (f"{float(r.delta_omega) / (2 * math.pi * dw_div):+.4g}"
              if _finite(r.delta_omega) else "—")
        eps = f"{float(r.epsilon_sx):.3e}" if _finite(r.epsilon_sx) else "—"
        health = s(f"{q.health:<9}", _HEALTH_COLOR.get(q.health, "bold"))
        lines.append(f"  {q.qubit:<7}{health}{_time(r.T1_s):>12}{_time(r.T2_s):>12}"
                     f"{_time(r.T2_ramsey_s):>12}{dw:>13}{eps:>12}")
    counts = {h: sum(q.health == h for q in report.qubits) for h in ("GREEN", "YELLOW", "RED")}
    lines.append("")
    for h in ("GREEN", "YELLOW", "RED"):
        if counts[h]:
            lines.append(f"  {s(f'{h:<6}', _HEALTH_COLOR[h], 'bold')} {counts[h]} qubit(s): "
                         + s(_HEALTH_MEANING[h], "dim"))
    return lines


def render_report(report, color: Optional[bool] = None) -> str:
    """Render a CanaryReport as a terminal block."""
    from . import __version__
    s = _Style(_use_color(color))
    if report.mode == "hardware" and report.backend.startswith("fake_"):
        where = s("fake IBM backend · local dry run (simulated noise model)", "yellow")
    elif report.mode == "hardware":
        where = s("IBM hardware", "green")
    else:
        where = s(f"in-silico simulation · {report.regime} regime", "cyan")
    shots = report.shots
    try:
        when = datetime.fromisoformat(report.timestamp).strftime("%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        when = report.timestamp
    lines = [_rule(s, s(f"Quantum Canary v{__version__}", "bold"), "━"),
             _kv(s, "Backend", f"{s(report.backend, 'bold')} · {report.architecture} · {where}")]
    if report.job_id:
        lines.append(_kv(s, "Job", report.job_id))
    lines.append(_kv(s, "Qubits", ", ".join(str(q.qubit) for q in report.qubits)))
    blocks = (f"{report.probe_blocks} interleaved blocks" if report.probe_blocks > 1
              else "circuit-by-circuit")
    lines.append(_kv(s, "Probe", f"{report.circuits_per_qubit} circuits/qubit · "
                                 f"{report.shots_per_qubit:,} shots/qubit · {blocks}"))
    lines.append(_kv(s, "", s(f"per circuit: T1 {shots['t1']} · Ramsey {shots['ramsey']} · "
                              f"gate-rep {shots['gate']} · echo {shots['echo']} shots", "dim")))
    lines.append(_kv(s, "Time", when))
    if report.mode == "simulation":
        eps_truth = ("ε = the engine's own ε" if report.regime == "ideal"
                     else "ε = sx average gate infidelity")
        lines.append(_kv(s, "", s("True values are the simulator's hidden parameters "
                                  f"(T2 = echo T2; {eps_truth}).", "dim")))
    lines.append("")
    for q in report.qubits:
        lines.extend(_qubit_block(s, q, report.mode, report.regime))
        lines.append("")
    lines.extend(_summary_table(s, report))
    lines.append(_rule(s, char="━"))
    return "\n".join(lines)


def _cell_health(s: _Style, health: str, width: int = 8) -> str:
    return s(f"{health:<{width}}", _HEALTH_COLOR.get(health, "bold"), "bold")


def render_drift(drift, color: Optional[bool] = None) -> str:
    """Render a DriftReport: what changed between two checks."""
    s = _Style(_use_color(color))
    names = (("T1_s", "T1"), ("T2_s", "T2"), ("delta_omega", "Δω"), ("epsilon_sx", "ε_sx"))
    lines = [_rule(s, s("Post-job drift", "bold"), "━"),
             s(f"  {'Qubit':<7}{'Before':<9}{'After':<9}"
               + "".join(f"{'z ' + lbl:>9}" for _, lbl in names) + "  Drifted", "dim")]
    for q, z in drift.z_scores.items():
        moved = [lbl for key, lbl in names if drift.drifted[q][key]]
        zs = "".join(f"{z[key]:>+9.2f}" if _finite(z[key]) else f"{'—':>9}" for key, _ in names)
        lines.append(f"  {q:<7}{_cell_health(s, drift.health_before[q], 9)}"
                     f"{_cell_health(s, drift.health_after[q], 9)}{zs}  "
                     + (s(", ".join(moved), "yellow") if moved else "—"))
    lines.append("")
    if drift.stable:
        lines.append(_kv(s, "Verdict", s("stable: the qubits did not change between the two checks", "green")))
    else:
        lines.append(_kv(s, "Verdict", s(f"qubit(s) {drift.drifted_qubits} changed: results that "
                                         "depend on them may not hold; re-run or re-learn "
                                         "error mitigation", "yellow")))
    lines.append(_kv(s, "", s(f"drift = Holm-corrected two-sample z-test at |z| > "
                              f"{drift.z_threshold:g}", "dim")))
    lines.append(_rule(s, char="━"))
    return "\n".join(lines)
