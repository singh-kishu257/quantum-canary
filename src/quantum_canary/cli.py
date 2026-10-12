"""Command-line interface: `quantum-canary` (or `python -m quantum_canary`).

  quantum-canary check ibm_kingston 13,14            check qubits on IBM hardware
  quantum-canary check fake_fez 13,14                the same, locally, no account
  quantum-canary check ibm_kingston 13,14 --json before.json
  quantum-canary compare before.json after.json      what changed between two checks
  quantum-canary compare c0.json c1.json c2.json     every later check against the first

Everything else (simulation, Canary rounds inside a job, IonQ) is in the Python
API: import quantum_canary as canary.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional


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
    """A fake backend for names starting with fake_, otherwise the IBM backend
    from the saved account."""
    if name.startswith("fake_"):
        from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2
        return FakeProviderForBackendV2().backend(name)
    from qiskit_ibm_runtime import QiskitRuntimeService
    return QiskitRuntimeService(**({"instance": instance} if instance else {})).backend(name)


def _cmd_check(args) -> int:
    from .api import check
    backend = _backend(args.backend, args.instance)
    _say(f"checking qubits {args.qubits} on {backend.name}"
         + (" (local fake backend)" if args.backend.startswith("fake_") else "") + " ...")
    report = check(backend, args.qubits)
    print(report.render())
    if args.json:
        report.save(args.json)
        _say(f"saved {args.json}")
    return 0


def _cmd_compare(args) -> int:
    from .api import compare, load, timeline
    reports = [load(path) for path in args.checks]
    if len(reports) == 2:
        print(compare(reports[0], reports[1]).render())
    else:
        print(f"baseline {args.checks[0]}")
        print(timeline(reports[0], reports[1:], labels=args.checks[1:]))
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
    c = sub.add_parser("compare", help="what changed between saved checks; the first file is the "
                                       "baseline, every later one is compared with it")
    c.add_argument("checks", nargs="+", metavar="FILE", help="JSON files from check --json")
    c.set_defaults(func=_cmd_compare)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "compare" and len(args.checks) < 2:
        parser.error("compare needs at least two files")
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
