"""
Interactive command-line interface for Quantum Canary.

Two modes of operation:

    Interactive (default — just run `quantum-canary`):
        Wizard-style flow that walks the user through credential entry,
        backend selection, and configuration choices. Best for first-time
        users.

    Direct (flag-based):
        `quantum-canary check   --backend ibm_kingston`
        `quantum-canary monitor --backend ibm_kingston --hours 24`
        Useful for scripts, automation, and CI pipelines.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

# ── BANNER ────────────────────────────────────────────────────────────────────
BANNER = r"""
+-----------------------------------------------------------+
|                                                           |
|              Welcome to Quantum Canary                    |
|    Real-time noise drift detection for NISQ-era qubits    |
|                                                           |
|      Author : Kanishka Singh                              |
|      Version: 0.1.0   |   License: MIT                    |
|                                                           |
|   Feedback welcome:                                       |
|   github.com/singh-kishu257/quantum-canary/issues         |
|                                                           |
+-----------------------------------------------------------+
"""


# ── CREDENTIAL HELPERS ────────────────────────────────────────────────────────
def _read_credential(path_or_value: str | None, env_var: str) -> str:
    """Resolve a credential from env var, file path, or raw value."""
    if path_or_value is None:
        val = os.environ.get(env_var)
        if val:
            return val.strip()
        raise SystemExit(
            f"No credential provided. Set ${env_var} or pass --token / --instance."
        )
    p = Path(path_or_value)
    if p.is_file():
        return p.read_text(encoding="utf-8").strip()
    return path_or_value.strip()


def _prompt(msg: str, default: str | None = None) -> str:
    """Prompt with an optional default. Returns the entered value."""
    suffix = f" [{default}]" if default else ""
    response = input(f"{msg}{suffix}: ").strip()
    return response if response else (default or "")


def _prompt_int(msg: str, default: int) -> int:
    """Prompt for an integer with a default."""
    while True:
        raw = _prompt(msg, str(default))
        try:
            return int(raw)
        except ValueError:
            print("  Please enter a valid integer.")


def _prompt_float(msg: str, default: float) -> float:
    """Prompt for a float with a default."""
    while True:
        raw = _prompt(msg, str(default))
        try:
            return float(raw)
        except ValueError:
            print("  Please enter a valid number.")


# ── INTERACTIVE WIZARD ────────────────────────────────────────────────────────
def _connect_and_select_backend() -> tuple[str, str, str]:
    """Run the credential + backend selection flow. Returns (token, instance, backend)."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    print("\n--- IBM Quantum Credentials ---")
    print("Your token will be hidden as you type. Paste it and press Enter.\n")
    token    = getpass.getpass("  IBM Quantum API token: ").strip()
    instance = getpass.getpass("  IBM Cloud CRN:         ").strip()

    if not token or not instance:
        print("\n  Token and CRN are both required. Exiting.")
        sys.exit(1)

    print("\nConnecting to IBM Quantum...")
    try:
        service = QiskitRuntimeService(
            channel="ibm_cloud", token=token, instance=instance
        )
        print("  Connected.\n")
    except Exception as e:
        print(f"  Connection failed: {e}")
        sys.exit(1)

    print("Fetching available backends...")
    try:
        backends = service.backends()
        operational = []
        for b in backends:
            try:
                status = b.status()
                if status.operational and not b.configuration().simulator:
                    operational.append(b)
            except Exception:
                continue
    except Exception as e:
        print(f"  Failed to fetch backends: {e}")
        sys.exit(1)

    if not operational:
        print("  No operational backends available. Exiting.")
        sys.exit(1)

    print(f"  Found {len(operational)} operational backend(s):\n")
    for i, b in enumerate(operational, start=1):
        try:
            n_qubits = b.configuration().n_qubits
            queue    = b.status().pending_jobs
            print(f"    [{i}] {b.name:<25s}  "
                  f"{n_qubits} qubits   queue: {queue} jobs")
        except Exception:
            print(f"    [{i}] {b.name}")

    print()
    while True:
        try:
            choice = int(input(f"  Select backend [1-{len(operational)}]: ").strip())
            if 1 <= choice <= len(operational):
                backend_name = operational[choice - 1].name
                break
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(operational)}.")

    print(f"\n  Selected: {backend_name}")
    return token, instance, backend_name


def _interactive_check() -> int:
    """Interactive: single drift check."""
    from quantum_canary import Canary

    token, instance, backend_name = _connect_and_select_backend()

    print("\n--- Configuration ---")
    shots = _prompt_int("  Shots per circuit", 1000)

    print("\nRunning a single canary check...\n")
    canary = Canary(token=token, instance=instance,
                    backend=backend_name, shots=shots)
    result = canary.check()

    print("Result")
    print("-" * 50)
    print(f"  Backend           : {backend_name}")
    print(f"  Timestamp         : {result['timestamp']}")
    print(f"  F_bell            : {result['F_bell']:.4f}")
    print(f"  F_gate            : {result['F_gate']:.4f}")
    print(f"  F_coherence       : {result['F_coherence']:.4f}")
    print(f"  Drift probability : {result['drift_probability']:.4f}")
    print(f"  Uncertainty       : {result['uncertainty']:.4f}")
    print(f"  Verdict           : {result['verdict']}")
    print()
    return 0


def _interactive_monitor() -> int:
    """Interactive: continuous monitoring."""
    from quantum_canary import Canary

    token, instance, backend_name = _connect_and_select_backend()

    print("\n--- Configuration ---")
    interval = _prompt_int(  "  Minutes between rounds", 15)
    hours    = _prompt_float("  Total monitoring duration (hours)", 24.0)
    shots    = _prompt_int(  "  Shots per circuit", 1000)
    log_path = _prompt(      "  Log file path", "logs/realtime_log.csv")

    n_rounds = int((hours * 3600) / (interval * 60))

    print("\n--- Confirm ---")
    print(f"  Backend  : {backend_name}")
    print(f"  Interval : {interval} minutes")
    print(f"  Duration : {hours} hours  ({n_rounds} rounds)")
    print(f"  Shots    : {shots}")
    print(f"  Log path : {log_path}")
    confirm = input("\n  Start monitoring? [y/N]: ").strip().lower()
    if confirm not in ("y", "yes"):
        print("  Cancelled.")
        return 0

    canary = Canary(token=token, instance=instance,
                    backend=backend_name, shots=shots)
    canary.monitor(interval_minutes=interval, hours=hours, log_path=log_path)
    return 0


def _interactive_classify() -> int:
    """Interactive: classify a fidelity triplet without IBM connection."""
    from quantum_canary import MLPEnsemble

    print("\n--- Classify Pre-Collected Fidelities ---")
    print("Enter the three fidelity values you've already measured.")
    print("No IBM connection required.\n")

    f_bell      = _prompt_float("  F_bell      (Bell state fidelity)", 0.95)
    f_gate      = _prompt_float("  F_gate      (gate-error fidelity)", 0.95)
    f_coherence = _prompt_float("  F_coherence (coherence fidelity)",  0.95)

    print("\nLoading MLP ensemble...")
    classifier = MLPEnsemble()

    import numpy as np
    result = classifier.predict(np.array([f_bell, f_gate, f_coherence]))

    print("\nResult")
    print("-" * 50)
    print(f"  F_bell            : {f_bell:.4f}")
    print(f"  F_gate            : {f_gate:.4f}")
    print(f"  F_coherence       : {f_coherence:.4f}")
    print(f"  Drift probability : {result['drift_probability']:.4f}")
    print(f"  Uncertainty       : {result['uncertainty']:.4f}")
    print(f"  Verdict           : {result['verdict']}")
    print()
    return 0


def _interactive_main() -> int:
    """Top-level interactive wizard."""
    print(BANNER)
    print("What would you like to do?")
    print("  [1] Run a single drift check")
    print("  [2] Continuous monitoring")
    print("  [3] Classify pre-collected fidelity data (no IBM connection)")
    print("  [4] Exit")
    print()

    choice = input("  > ").strip()
    if choice == "1":
        return _interactive_check()
    elif choice == "2":
        return _interactive_monitor()
    elif choice == "3":
        return _interactive_classify()
    else:
        print("\nGoodbye.")
        return 0


# ── DIRECT (FLAG-BASED) MODE ──────────────────────────────────────────────────
def cmd_check(args: argparse.Namespace) -> int:
    from quantum_canary import Canary

    token    = _read_credential(args.token,    "IBM_QUANTUM_TOKEN")
    instance = _read_credential(args.instance, "IBM_QUANTUM_CRN")

    canary = Canary(token=token, instance=instance, backend=args.backend)
    result = canary.check()

    print()
    print("Quantum Canary  --  single check")
    print("-" * 50)
    print(f"  Backend           : {args.backend}")
    print(f"  Timestamp         : {result['timestamp']}")
    print(f"  F_bell            : {result['F_bell']:.4f}")
    print(f"  F_gate            : {result['F_gate']:.4f}")
    print(f"  F_coherence       : {result['F_coherence']:.4f}")
    print(f"  Drift probability : {result['drift_probability']:.4f}")
    print(f"  Uncertainty       : {result['uncertainty']:.4f}")
    print(f"  Verdict           : {result['verdict']}")
    print()
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    from quantum_canary import Canary

    token    = _read_credential(args.token,    "IBM_QUANTUM_TOKEN")
    instance = _read_credential(args.instance, "IBM_QUANTUM_CRN")

    canary = Canary(token=token, instance=instance, backend=args.backend)
    canary.monitor(
        interval_minutes = args.interval,
        hours            = args.hours,
        log_path         = args.log,
    )
    return 0


# ── ARG PARSER ────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantum-canary",
        description="Real-time quantum noise drift detection. Run with no "
                    "arguments for an interactive wizard, or use subcommands "
                    "for direct invocation.",
    )
    parser.add_argument(
        "--token", default=None,
        help="IBM Quantum API token, OR a path to a file containing it. "
             "Falls back to the IBM_QUANTUM_TOKEN environment variable.",
    )
    parser.add_argument(
        "--instance", default=None,
        help="IBM Cloud CRN, OR a path to a file. "
             "Falls back to the IBM_QUANTUM_CRN environment variable.",
    )

    sub = parser.add_subparsers(dest="command")

    p_check = sub.add_parser("check", help="Run a single drift check.")
    p_check.add_argument("--backend", required=True,
                         help="IBM backend name (e.g. ibm_kingston).")
    p_check.set_defaults(func=cmd_check)

    p_mon = sub.add_parser("monitor",
                           help="Run continuous monitoring at fixed intervals.")
    p_mon.add_argument("--backend",  required=True,
                       help="IBM backend name (e.g. ibm_kingston).")
    p_mon.add_argument("--interval", type=int,   default=15,
                       help="Minutes between rounds (default 15).")
    p_mon.add_argument("--hours",    type=float, default=24.0,
                       help="Total monitoring duration in hours (default 24).")
    p_mon.add_argument("--log",      default="logs/realtime_log.csv",
                       help="Output CSV log path.")
    p_mon.set_defaults(func=cmd_monitor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args   = parser.parse_args(argv)

    # No subcommand --> launch interactive wizard
    if args.command is None:
        return _interactive_main()

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())