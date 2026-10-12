"""Command-line interface: `quantum-canary` (or `python -m quantum_canary`).

  quantum-canary check ibm_kingston 13,14            check qubits on IBM hardware
  quantum-canary check fake_fez 13,14                the same, locally, no account
  quantum-canary check ibm_kingston 13,14 --json before.json
  quantum-canary compare before.json after.json      what changed between two checks

Everything else (simulation, monitoring inside a job, IonQ) is in the Python
API: import quantum_canary as canary.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import List, Optional

import numpy as np

PROBES = (("t1", "T1"), ("ramsey", "Ramsey"), ("gate", "gate"), ("echo", "echo"))
EXPLAIN = """\
Fit     chi2/dof of each probe against its limit. On a qubit that follows the model
        each probe goes over its limit 1.25% of the time; when it does, that probe's
        numbers may be wrong.
Health  GREEN   every fit converged and passed, and T2* is within 20% of T2
        YELLOW  a fit failed its limit, or T2* differs from T2 by more than 20%
                (slow frequency noise, which dynamical decoupling refocuses)
        RED     a fit did not converge; re-run the check
        Health describes the measurement, not how good the qubit is: read the
        numbers for that."""


def _qubits(text: str) -> List[int]:
    try:
        qubits = [int(t) for t in text.replace(" ", "").split(",") if t]
    except ValueError:
        raise argparse.ArgumentTypeError(f"qubits must be comma-separated integers, got {text!r}")
    if not qubits:
        raise argparse.ArgumentTypeError("list at least one qubit")
    return qubits


def _say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _backend(name: str, instance: Optional[str]):
    """(backend, service): a fake backend for names starting with fake_, otherwise
    the IBM backend from the saved account."""
    if name.startswith("fake_"):
        from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2
        return FakeProviderForBackendV2().backend(name), None
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(**({"instance": instance} if instance else {}))
    return service.backend(name), service


def _ibm_usage(service, job_id: Optional[str]) -> Optional[float]:
    if service is None or not job_id:
        return None
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            usage = service.job(job_id).usage()
        except Exception:
            return None
        if usage:
            return float(usage)
        time.sleep(5)
    return None


def _fit(result) -> List[str]:
    """Probes whose chi2/dof is over its limit, as 'Ramsey 5.1 > 2.7'."""
    from .online import _green_chi2_threshold
    out = []
    for key, label in PROBES:
        value, limit = getattr(result, f"{key}_chi2_dof"), _green_chi2_threshold(key)
        if np.isfinite(value) and value > limit:
            out.append(f"{label} {value:.1f} > {limit:.1f}")
    return out


def _health(q) -> str:
    """The health verdict with its reason."""
    r = q.result
    if q.health == "RED":
        return "RED (fit did not converge)"
    reasons = ["fit"] if _fit(r) else []
    if r.T2_s > 0 and np.isfinite(r.T2_ramsey_s) and np.isfinite(r.T2_echo_s):
        if abs(r.T2_ramsey_s - r.T2_echo_s) / r.T2_s > 0.20:
            reasons.append(f"T2*/T2 {r.T2_ramsey_s / r.T2_s:.2f}")
    return q.health + (f" ({', '.join(reasons)})" if reasons else "")


def _us(value: float, sigma: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value * 1e6:.0f} ± {sigma * 1e6:.0f} us" if np.isfinite(sigma) else f"{value * 1e6:.0f} us"


def _table(qubits) -> str:
    head = f"{'qubit':<6}{'T1':<16}{'T2':<16}{'T2*':<16}{'dw (kHz)':<16}{'eps':<20}{'fit':<26}health"
    lines = [head, "-" * len(head)]
    for q in qubits:
        r = q.result
        dw = (f"{r.delta_omega / 2e3 / np.pi:+.2f} ± {r.delta_omega_sigma / 2e3 / np.pi:.2f}"
              if np.isfinite(r.delta_omega) else "n/a")
        eps = (f"{r.epsilon_sx:.2e} ± {r.epsilon_sx_sigma:.0e}" if np.isfinite(r.epsilon_sx) else "n/a")
        fit = "; ".join(_fit(r)) or "ok"
        lines.append(f"{q.qubit:<6}{_us(r.T1_s, r.T1_sigma_s):<16}{_us(r.T2_s, r.T2_sigma_s):<16}"
                     f"{_us(r.T2_ramsey_s, r.T2_ramsey_sigma_s):<16}{dw:<16}{eps:<20}{fit:<26}{_health(q)}")
    return "\n".join(lines)


def _cmd_check(args) -> int:
    from .runner import run_ibm
    from .workflow import _dd_reasons
    backend, service = _backend(args.backend, args.instance)
    _say(f"checking qubits {args.qubits} on {backend.name}"
         + (" (local fake backend)" if service is None else "") + " ...")
    report = run_ibm(backend, args.qubits)
    usage = _ibm_usage(service, report.job_id)
    print(_table(report.qubits))
    dd = {q.qubit: _dd_reasons(q) for q in report.qubits}
    advised = {q: r for q, r in dd.items() if r}
    print()
    if advised:
        for q, reasons in advised.items():
            print(f"dynamical decoupling advised on qubit {q}: {'; '.join(reasons)}")
    else:
        print("dynamical decoupling: not needed on these qubits")
    print()
    print(EXPLAIN)
    print()
    print(f"backend {backend.name}, job {report.job_id or 'local'}, "
          f"{report.shots_per_qubit:,} shots per qubit, probed in groups {report.groups}"
          + (f", IBM measured usage {usage:.1f} s" if usage is not None else
             ("" if service is None else ", IBM usage not reported yet")))
    if args.json:
        payload = {"backend": backend.name, "job_id": report.job_id, "ibm_usage_s": usage,
                   "health": {q.qubit: _health(q) for q in report.qubits},
                   "fit_over_limit": {q.qubit: _fit(q.result) for q in report.qubits},
                   "dynamical_decoupling": dd, "report": report.to_dict()}
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        _say(f"saved {args.json}")
    return 0


def _load(path: str):
    """A CanaryReport rebuilt from a JSON file written by check --json (or by
    CanaryReport.to_dict()), for compare()."""
    from dataclasses import fields
    from .inversion import InversionResult
    from .runner import CanaryReport, QubitReport
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    rep = data.get("report", data)
    names = {f.name for f in fields(InversionResult)}
    qubits = []
    for q in rep["qubits"]:
        result = {k: v for k, v in q["result"].items() if k in names}
        result["calibration_source"] = None
        qubits.append(QubitReport(qubit=int(q["qubit"]), result=InversionResult(**result),
                                  profile=None, health=q["health"]))
    return CanaryReport(mode=rep.get("mode", "hardware"), backend=rep.get("backend", ""),
                        architecture=rep.get("architecture", "superconducting"),
                        shots=rep.get("shots", {}), probe_blocks=rep.get("probe_blocks", 1),
                        circuits_per_qubit=rep.get("circuits_per_qubit", 0), qubits=qubits,
                        timestamp=rep.get("timestamp", ""))


def _cmd_compare(args) -> int:
    from .workflow import compare
    before, after = _load(args.before), _load(args.after)
    drift = compare(before, after)
    names = (("T1_s", "T1"), ("T2_s", "T2"), ("delta_omega", "dw"), ("epsilon_sx", "eps"))
    head = f"{'qubit':<6}{'before':<9}{'after':<9}" + "".join(f"{'z ' + n:>9}" for _, n in names) + "   changed"
    print(head)
    print("-" * len(head))
    for q, z in drift.z_scores.items():
        moved = [n for key, n in names if drift.drifted[q][key]]
        print(f"{q:<6}{drift.health_before[q]:<9}{drift.health_after[q]:<9}"
              + "".join(f"{z[key]:>+9.2f}" if np.isfinite(z[key]) else f"{'n/a':>9}" for key, _ in names)
              + "   " + (", ".join(moved) or "-"))
    print()
    print(f"changed = |z| > {drift.z_threshold:g} after a Holm correction over each qubit's "
          "four parameters; z uses both checks' error bars")
    print("stable: nothing changed beyond the error bars" if drift.stable else
          f"changed qubits: {drift.drifted_qubits}; results that depend on them may not hold")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from . import __version__
    ap = argparse.ArgumentParser(prog="quantum-canary",
                                 description="Check the health of qubits on IBM hardware.")
    ap.add_argument("--version", action="version", version=f"quantum-canary {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check", help="measure T1, T2, T2*, frequency offset and gate error of qubits")
    p.add_argument("backend", help="IBM backend, e.g. ibm_kingston, or a fake backend, e.g. fake_fez")
    p.add_argument("qubits", type=_qubits, help="comma-separated qubits, e.g. 13,14")
    p.add_argument("--json", metavar="FILE", help="also save the result, for compare")
    p.add_argument("--instance", default=None, help="IBM instance, if your account has several")
    p.set_defaults(func=_cmd_check)
    c = sub.add_parser("compare", help="what changed between two saved checks")
    c.add_argument("before", help="JSON file from check --json")
    c.add_argument("after", help="JSON file from check --json")
    c.set_defaults(func=_cmd_compare)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
