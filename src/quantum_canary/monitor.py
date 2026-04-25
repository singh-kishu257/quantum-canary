"""
Real-time canary monitoring on IBM Quantum hardware.

The `Canary` class wraps the entire workflow: connect to a backend,
transpile the canary circuits, run them on a schedule, classify each
round with the MLP ensemble, and persist results to disk.
"""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from quantum_canary.circuits   import (build_canary_circuits,
                                       fidelity_bell, fidelity_single)
from quantum_canary.classifier import MLPEnsemble


def _extract_counts(result_item) -> dict:
    """
    Extract counts from a SamplerV2 result item regardless of the
    classical register name. Different backends and transpilation
    presets produce different attribute names ('c', 'meas', 'creg', etc.).
    """
    data = result_item.data
    for attr in vars(data):
        try:
            return getattr(data, attr).get_counts()
        except Exception:
            continue
    raise ValueError(
        "Could not extract counts from result. "
        f"Available attributes: {list(vars(data))}"
    )


class Canary:
    """
    Real-time quantum drift monitor.

    Parameters
    ----------
    token : str
        IBM Quantum API token.
    instance : str
        IBM Cloud Resource Name (CRN) for your IBM Quantum instance.
    backend : str
        Backend name (e.g. 'ibm_kingston').
    shots : int, optional
        Shots per circuit (default 1000).
    classifier : MLPEnsemble, optional
        Pre-loaded MLP ensemble. Loaded automatically if not provided.

    Examples
    --------
    >>> from quantum_canary import Canary
    >>> canary = Canary(token="...", instance="...", backend="ibm_kingston")
    >>> result = canary.check()
    >>> print(result["verdict"])
    STABLE
    >>> canary.monitor(interval_minutes=15, hours=24)
    """

    def __init__(
        self,
        token:      str,
        instance:   str,
        backend:    str,
        shots:      int = 1000,
        classifier: Optional[MLPEnsemble] = None,
    ) -> None:
        from qiskit_ibm_runtime import QiskitRuntimeService
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        self.shots = shots
        self.service = QiskitRuntimeService(
            channel="ibm_cloud", token=token, instance=instance
        )
        self.backend = self.service.backend(backend)

        circuits = build_canary_circuits()
        pm = generate_preset_pass_manager(
            optimization_level=0, backend=self.backend
        )
        self.transpiled = [pm.run(qc) for qc in circuits]

        self.classifier = classifier or MLPEnsemble()

    def check(self) -> dict:
        """
        Run a single round of canary measurements.

        Returns
        -------
        dict
            {'timestamp': str (UTC),
             'F_bell':            float,
             'F_gate':            float,
             'F_coherence':       float,
             'drift_probability': float,
             'verdict':           'STABLE' or 'DRIFTED',
             'uncertainty':       float}
        """
        from qiskit_ibm_runtime import SamplerV2 as Sampler

        sampler = Sampler(self.backend)
        job     = sampler.run(self.transpiled, shots=self.shots)
        result  = job.result()

        counts_bell = _extract_counts(result[0])
        counts_coh  = _extract_counts(result[1])
        counts_gate = _extract_counts(result[2])

        f_bell       = fidelity_bell(counts_bell)
        f_coherence  = fidelity_single(counts_coh)
        f_gate       = fidelity_single(counts_gate)

        # MLP feature order: [F_bell, F_gate, F_coherence]
        prediction = self.classifier.predict(
            np.array([f_bell, f_gate, f_coherence])
        )

        return {
            "timestamp":         datetime.now(timezone.utc).strftime(
                                     "%Y-%m-%d %H:%M:%S UTC"),
            "F_bell":            round(f_bell,      6),
            "F_gate":            round(f_gate,      6),
            "F_coherence":       round(f_coherence, 6),
            "drift_probability": round(prediction["drift_probability"], 6),
            "verdict":           prediction["verdict"],
            "uncertainty":       round(prediction["uncertainty"], 6),
        }

    def classify(self,
                 f_bell: float,
                 f_gate: float,
                 f_coherence: float) -> dict:
        """
        Classify a fidelity triplet without running circuits.

        Useful when the user has already collected fidelity measurements
        from their own circuits and wants to run drift inference without
        any additional queue overhead.

        Parameters
        ----------
        f_bell : float
            Bell state fidelity (fraction of '00' + '11' counts).
        f_gate : float
            Gate-error circuit fidelity (fraction of '0' counts).
        f_coherence : float
            Coherence circuit fidelity (fraction of '0' counts).

        Returns
        -------
        dict
            {'drift_probability': float,
             'verdict': 'STABLE' or 'DRIFTED',
             'uncertainty': float}
        """
        return self.classifier.predict(
            np.array([f_bell, f_gate, f_coherence])
        )

    def monitor(
        self,
        interval_minutes: int  = 15,
        hours:            float = 24.0,
        log_path:         str  = "logs/realtime_log.csv",
        verbose:          bool = True,
    ) -> str:
        """
        Run continuous monitoring on a schedule.

        Each round runs `check()`, logs the result to disk, and waits
        `interval_minutes` before the next round.

        Parameters
        ----------
        interval_minutes : int, optional
            Minutes between measurements (default 15).
        hours : float, optional
            Total monitoring duration (default 24.0).
        log_path : str, optional
            Path to the CSV log file (created if missing).
        verbose : bool, optional
            Print each round to stdout (default True).

        Returns
        -------
        str
            Path to the CSV log file.
        """
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        n_rounds   = int((hours * 3600) / (interval_minutes * 60))
        interval_s = interval_minutes * 60

        if verbose:
            print(f"\n{'='*60}")
            print(f"  QUANTUM CANARY — REAL-TIME MONITOR")
            print(f"{'='*60}")
            print(f"  Backend  : {self.backend.name}")
            print(f"  Duration : {hours} hours")
            print(f"  Interval : {interval_minutes} minutes")
            print(f"  Rounds   : {n_rounds}")
            print(f"  Log      : {log_path}")
            print(f"{'='*60}\n")

        cols = [
            "timestamp", "F_bell", "F_gate", "F_coherence",
            "drift_probability", "verdict", "uncertainty",
        ]
        first_write = not Path(log_path).exists()

        for round_idx in range(n_rounds):
            try:
                row = self.check()
            except Exception as e:
                if verbose:
                    print(f"  [round {round_idx+1}] ✗ Error: {e}")
                if round_idx < n_rounds - 1:
                    time.sleep(interval_s)
                continue

            with open(log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=cols)
                if first_write:
                    writer.writeheader()
                    first_write = False
                writer.writerow(row)

            if verbose:
                icon = "[!]" if row["verdict"] == "DRIFTED" else "[ok]"
                print(f"  [{row['timestamp']}]  "
                      f"F_bell={row['F_bell']:.4f}  "
                      f"F_gate={row['F_gate']:.4f}  "
                      f"F_coh={row['F_coherence']:.4f}  "
                      f"prob={row['drift_probability']:.4f}  "
                      f"{icon} {row['verdict']}")

            if round_idx < n_rounds - 1:
                time.sleep(interval_s)

        if verbose:
            print(f"\n  Monitoring complete. Log saved to {log_path}\n")

        return log_path