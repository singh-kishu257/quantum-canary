"""Command-line interface: `quantum-canary` (or `python -m quantum_canary`).

  quantum-canary check --backend ibm_fez --qubits 13,14,2 --spares 3,4
  quantum-canary check --fake fake_fez --qubits 13,14      # local dry run, no account
  quantum-canary ibm --backend ibm_fez --qubits 13,14,2    # full per-qubit report
  quantum-canary simulate --qubits 3
  quantum-canary selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional


def _parse_qubits(text: str) -> List[int]:
    try:
        qubits = [int(t) for t in text.replace(" ", "").split(",") if t != ""]
    except ValueError:
        raise argparse.ArgumentTypeError(f"--qubits must be comma-separated integers, got {text!r}")
    if not qubits:
        raise argparse.ArgumentTypeError("--qubits must list at least one qubit")
    return qubits


def _status(msg: str) -> None:
    print(f"[quantum-canary] {msg}", file=sys.stderr, flush=True)


def _emit(report, args) -> None:
    color = False if args.no_color else None
    print(report.render(color=color))
    if args.verbose:
        for q in report.qubits:
            for line in q.engine_log:
                print(f"  [engine q{q.qubit}] {line}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        _status(f"wrote {args.json}")


def _backend(args):
    """The IBM backend (or fake backend) named on the command line, or None."""
    try:
        import qiskit_ibm_runtime  # noqa: F401
    except ImportError:
        _status("qiskit-ibm-runtime is missing: pip install --upgrade quantum-canary")
        return None
    if args.fake:
        from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2
        backend = FakeProviderForBackendV2().backend(args.fake)
        _status(f"local dry run on {backend.name} (fake backend noise model, no account)")
        return backend
    from qiskit_ibm_runtime import QiskitRuntimeService
    kwargs = {k: v for k, v in (("channel", args.channel), ("instance", args.instance))
              if v is not None}
    _status("connecting to IBM Quantum (saved account)...")
    return QiskitRuntimeService(**kwargs).backend(args.backend)


def _cmd_check(args) -> int:
    from .workflow import check
    backend = _backend(args)
    if backend is None:
        return 2
    _status(f"{backend.name}: checking qubits {args.qubits}"
            + (f" and spares {args.spares}" if args.spares else "") + " in one job...")
    result = check(backend, args.qubits, spares=args.spares or ())
    print(result.render(color=False if args.no_color else None))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        _status(f"wrote {args.json}")
    return 0


def _cmd_ibm(args) -> int:
    from .runner import run_ibm
    backend = _backend(args)
    if backend is None:
        return 2
    _status(f"{backend.name}: building probes for qubits {args.qubits} and submitting one job...")
    report = run_ibm(backend, args.qubits, probe_blocks=args.probe_blocks)
    _emit(report, args)
    return 0


def _cmd_simulate(args) -> int:
    from .runner import run_simulation
    _status(f"simulating {args.qubits} {args.architecture} qubit(s), {args.regime} regime...")
    report = run_simulation(args.architecture, n_qubits=args.qubits, regime=args.regime,
                            seed=args.seed, probe_blocks=args.probe_blocks)
    _emit(report, args)
    return 0


def _cmd_selftest(args) -> int:
    from . import simulation
    simulation._run_validation_tests()
    return 0


def build_parser() -> argparse.ArgumentParser:
    from . import __version__
    from .runner import DEFAULT_PROBE_BLOCKS
    ap = argparse.ArgumentParser(
        prog="quantum-canary",
        description="Quantum Canary: qubit health checks before, during and after a "
                    "job on IBM superconducting hardware.")
    ap.add_argument("--version", action="version", version=f"quantum-canary {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    def _output_flags(p):
        p.add_argument("--json", metavar="PATH", help="also write the full report as JSON")
        p.add_argument("--no-color", action="store_true", help="plain-text output")
        p.add_argument("--verbose", action="store_true",
                       help="also print the engine's own log lines")
        p.add_argument("--probe-blocks", type=int, default=DEFAULT_PROBE_BLOCKS,
                       help=f"block-interleaved execution (default {DEFAULT_PROBE_BLOCKS}; "
                            "1 = circuit by circuit)")

    def _target_flags(p):
        target = p.add_mutually_exclusive_group(required=True)
        target.add_argument("--backend", help="IBM backend name, e.g. ibm_fez")
        target.add_argument("--fake", metavar="NAME",
                            help="qiskit-ibm-runtime fake backend for a local dry run, "
                                 "e.g. fake_fez")
        p.add_argument("--qubits", required=True, type=_parse_qubits,
                       help="physical qubits, comma-separated, e.g. 13,14,2")
        p.add_argument("--channel", default=None, help="QiskitRuntimeService channel")
        p.add_argument("--instance", default=None, help="QiskitRuntimeService instance")

    p_check = sub.add_parser("check", help="check qubits before a job: health, layout, "
                                           "dynamical decoupling")
    _target_flags(p_check)
    p_check.add_argument("--spares", type=_parse_qubits, default=None,
                         help="spare qubits to swap in for RED ones, e.g. 3,4")
    p_check.add_argument("--json", metavar="PATH", help="also write the result as JSON")
    p_check.add_argument("--no-color", action="store_true", help="plain-text output")
    p_check.set_defaults(func=_cmd_check)

    p_ibm = sub.add_parser("ibm", help="full per-qubit report on an IBM backend")
    _target_flags(p_ibm)
    _output_flags(p_ibm)
    p_ibm.set_defaults(func=_cmd_ibm)

    p_sim = sub.add_parser("simulate", help="run in silico on simulated qubits")
    p_sim.add_argument("--architecture", choices=["superconducting", "trapped_ion"],
                       default="superconducting")
    p_sim.add_argument("--qubits", type=int, default=3, help="number of simulated qubits")
    p_sim.add_argument("--regime", choices=["nisq", "ideal"], default="nisq")
    p_sim.add_argument("--seed", type=int, default=None,
                       help="seed of qubit 0 (default: the benchmarks' offset)")
    _output_flags(p_sim)
    p_sim.set_defaults(func=_cmd_simulate)

    p_test = sub.add_parser("selftest", help="run the simulation layer's validation tests")
    p_test.set_defaults(func=_cmd_selftest)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    # The report uses non-ASCII symbols (µ, ±, Δ, ε); write UTF-8 even when
    # stdout is redirected to a file or pipe with a narrower locale encoding.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    if args.command != "selftest":
        from .display import banner
        print(banner())
        print()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
