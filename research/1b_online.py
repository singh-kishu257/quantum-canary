"""
1b_online.py — online, non-interrupting qubit health monitor.

Wraps 1_inversion.py's Lindblad inversion engine with an interleaving
layer that weaves Canary probe circuits into any running quantum
workflow's circuit list (VQE, QAOA, simulation, arbitrary circuits —
this file is not tied to any one algorithm). Canary probes run in the
same job submission, on the same qubits the algorithm already uses,
between batches of algorithm circuits. The algorithm's own circuits
and results are never touched or reordered.
"""
from __future__ import annotations
import csv
import datetime
import importlib.util
import pathlib
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "inversion", pathlib.Path(__file__).parent / "1_inversion.py")
inv = importlib.util.module_from_spec(_spec)
sys.modules["inversion"] = inv
_spec.loader.exec_module(inv)

__all__ = [
    "CanaryInterleaver",
    "OnlineInversionResult",
    "build_inline_probe_circuits",
    "interleave_circuits",
    "parse_inline_results",
    "online_inversion",
]


@dataclass
class OnlineInversionResult:
    """Per-round snapshot of every monitored qubit's inverted parameters.

    round_index identifies which Canary round produced this snapshot;
    qubit_results maps qubit_id -> InversionResult (from 1_inversion.py);
    algo_indices / canary_indices record which positions in the original
    (interleaved) job circuit list belonged to the algorithm vs. Canary
    for this round, for audit purposes.
    """
    round_index:    int
    timestamp:      str
    qubit_results:  dict
    algo_indices:   list
    canary_indices: list
    # 95th pctile chi2 for DOF=2 (3-point fits); 2.0 is too
    # tight and flags ~14% of valid fits as YELLOW by sampling variance alone.
    green_chi2_threshold:         float = 5.0
    yellow_t2_disagree_threshold: float = 0.20

    def health_summary(self) -> dict:
        """Return {qubit_id: "GREEN" | "YELLOW" | "RED"} for every qubit
        in this round, using chi2/dof and T2_ramsey/T2_echo agreement as
        the health signal.

        GREEN:  all probe chi2_dof <= threshold AND all sigmas finite
        YELLOW: any probe chi2_dof > threshold, OR T2_ramsey and T2_echo
                disagree by more than the configured fraction
        RED:    any fit sigma is inf/nan (a probe failed to converge)
        """
        out = {}
        for qid, r in self.qubit_results.items():
            sigmas = [r.T1_sigma_s, r.T2_sigma_s, r.delta_omega_sigma,
                      r.epsilon_sx_sigma, r.T2_ramsey_sigma_s, r.T2_echo_sigma_s]
            if any((not np.isfinite(s)) for s in sigmas):
                out[qid] = "RED"
                continue

            chi2s = [r.t1_chi2_dof, r.ramsey_chi2_dof, r.gate_chi2_dof, r.echo_chi2_dof]
            chi2_bad = any(np.isfinite(c) and c > self.green_chi2_threshold for c in chi2s)

            t2_disagree = False
            T2_ref = r.T2_s if r.T2_s > 0 else None
            if T2_ref and np.isfinite(r.T2_ramsey_s) and np.isfinite(r.T2_echo_s):
                frac = abs(r.T2_ramsey_s - r.T2_echo_s) / T2_ref
                t2_disagree = frac > self.yellow_t2_disagree_threshold

            if chi2_bad or t2_disagree:
                out[qid] = "YELLOW"
            else:
                out[qid] = "GREEN"
        return out

    def summary_table(self) -> str:
        """Human-readable multi-line table of every qubit's current
        parameters and health status for this round."""
        health = self.health_summary()
        lines = [f"Canary Round {self.round_index} | {self.timestamp}"]
        lines.append(f"{'Qubit':<6}{'T1':<14}{'T2':<14}{'Δω (kHz)':<12}"
                     f"{'ε_sx':<11}{'Health':<8}")
        for qid in sorted(self.qubit_results.keys()):
            r = self.qubit_results[qid]
            arch = inv.ARCH_DEFAULTS.get(r.architecture, inv.ARCH_DEFAULTS["unknown"])
            s, u = arch["time_scale"], arch["display_unit"]
            T1_str = f"{r.T1_s*s:.2f}±{r.T1_sigma_s*s:.2f}{u}"
            T2_str = f"{r.T2_s*s:.2f}±{r.T2_sigma_s*s:.2f}{u}"
            dw_khz = r.delta_omega / (2*np.pi*1e3)
            dw_sig_khz = r.delta_omega_sigma / (2*np.pi*1e3)
            dw_str = f"{dw_khz:.2f}±{dw_sig_khz:.2f}"
            eps_str = f"{r.epsilon_sx:.2e}"
            lines.append(f"{qid:<6}{T1_str:<14}{T2_str:<14}{dw_str:<12}"
                         f"{eps_str:<11}{health[qid]:<8}")
        return "\n".join(lines)


def build_inline_probe_circuits(profiles: dict, qubit_ids: list) -> tuple:
    """Build Canary probe circuits for multiple qubits in one flat list.

    1_inversion.py's build_probe_circuits() always targets qubit 0 of a
    fresh 1-qubit QuantumCircuit. Here, each qubit's probe set is built
    the same way (as an independent 1-qubit circuit named per-qubit) and
    relies on the backend transpiler's `initial_layout`/qubit mapping to
    place it on the correct physical qubit at submission time — the
    caller is responsible for passing the appropriate layout per circuit
    (e.g. transpile(circ, backend, initial_layout=[qubit_id])). This
    avoids building large, mostly-idle multi-qubit circuits just to
    address a single physical qubit.

    Returns (circuits, metadata) where metadata maps qubit_id -> dict
    with per-qubit probe layout info (see module docstring / spec).
    """
    circuits = []
    metadata = {}
    for qid in qubit_ids:
        profile = profiles[qid]
        qubit_circuits, meta = inv.build_probe_circuits(profile)
        start_idx = len(circuits)
        circuits.extend(qubit_circuits)
        meta = dict(meta)
        meta["circuit_start_idx"] = start_idx
        meta["n_circuits"] = len(qubit_circuits)
        metadata[qid] = meta
    return circuits, metadata


def interleave_circuits(algo_circuits: list, canary_circuits: list,
                        interval_n: int = 10) -> tuple:
    """Weave canary_circuits into algo_circuits every interval_n algorithm
    circuits, so a full Canary probe round runs after each batch.

    Returns (interleaved_circuit_list, round_map). round_map is a list
    of dicts, one per Canary round inserted, each recording the indices
    (in the interleaved list) of that round's algorithm circuits and
    where its Canary circuits start.

    If there are fewer algorithm circuits than interval_n, a single
    Canary round is appended once at the end.
    """
    n_algo = len(algo_circuits)
    n_canary = len(canary_circuits)

    if n_algo < interval_n:
        interleaved = list(algo_circuits) + list(canary_circuits)
        round_map = [{
            "round_index": 0,
            "algo_indices": list(range(n_algo)),
            "canary_start_idx": n_algo,
            "n_canary_circuits": n_canary,
        }]
        return interleaved, round_map

    interleaved = []
    round_map = []
    round_index = 0
    i = 0
    while i < n_algo:
        batch = algo_circuits[i:i+interval_n]
        batch_start = len(interleaved)
        interleaved.extend(batch)
        algo_indices = list(range(batch_start, batch_start + len(batch)))

        canary_start_idx = len(interleaved)
        interleaved.extend(canary_circuits)
        round_map.append({
            "round_index": round_index,
            "algo_indices": algo_indices,
            "canary_start_idx": canary_start_idx,
            "n_canary_circuits": n_canary,
        })
        round_index += 1
        i += interval_n

    return interleaved, round_map


def parse_inline_results(job_results: list, round_map: list,
                         canary_metadata: dict) -> tuple:
    """Split a flat job-result list (aligned with interleave_circuits()'s
    output order) back into per-round algorithm and Canary result lists.

    Returns (algo_results, canary_results): both are lists of lists,
    one inner list per Canary round in round_map order. algo_results
    contains the counts dicts for that round's algorithm circuits;
    canary_results contains the counts dicts for that round's Canary
    circuits, in the same flat order build_inline_probe_circuits()
    produced them.
    """
    algo_results = []
    canary_results = []
    for rmap in round_map:
        algo_slice = [job_results[i] for i in rmap["algo_indices"]]
        start = rmap["canary_start_idx"]
        n = rmap["n_canary_circuits"]
        canary_slice = job_results[start:start+n]
        algo_results.append(algo_slice)
        canary_results.append(canary_slice)
    return algo_results, canary_results


def online_inversion(canary_counts: list, canary_metadata: dict,
                     profiles: dict, round_index: int = 0,
                     shots_t1: int = 300, shots_ramsey: int = 1000,
                     shots_gate: int = 500, shots_echo: int = 500
                     ) -> OnlineInversionResult:
    """Run 1_inversion.py's lindblad_inversion() for every monitored qubit
    from a single Canary round's flat counts list, producing one
    OnlineInversionResult summarizing that round.
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    qubit_results = {}
    canary_indices = []

    for qid, meta in canary_metadata.items():
        start = meta["circuit_start_idx"]
        n = meta["n_circuits"]
        counts_slice = canary_counts[start:start+n]
        canary_indices.extend(range(start, start+n))

        profile = profiles[qid]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            result = inv.lindblad_inversion(
                counts_slice, meta, profile,
                shots_t1=shots_t1, shots_ramsey=shots_ramsey,
                shots_gate=shots_gate, shots_echo=shots_echo,
                qubit_id=qid, timestamp=timestamp)
        qubit_results[qid] = result

    return OnlineInversionResult(
        round_index=round_index, timestamp=timestamp,
        qubit_results=qubit_results, algo_indices=[],
        canary_indices=canary_indices)


class CanaryInterleaver:
    """High-level, stateful interface for online qubit monitoring.

    Handles Canary probe circuit construction, interleaving into an
    algorithm's circuit list, splitting job results back apart, running
    per-round Lindblad inversion, and keeping a rolling history of
    parameter estimates for every monitored qubit — usable with any
    quantum workflow (VQE, QAOA, simulation, arbitrary circuits).

    Usage:
        canary = CanaryInterleaver(
            qubit_ids=[0, 1, 2],
            profiles={i: inv.BackendProfile.from_architecture("superconducting")
                      for i in range(3)},
            interval_n=10,
        )
        full_circuit_list, round_map = canary.prepare(your_algo_circuits)
        # submit full_circuit_list, collect all_results (list[dict counts])
        algo_results, health_rounds = canary.process(all_results, round_map)
    """

    def __init__(self, qubit_ids: list, profiles: dict, interval_n: int = 10,
                 shots_t1: int = 300, shots_ramsey: int = 1000,
                 shots_gate: int = 500, shots_echo: int = 500,
                 # 95th pctile chi2 for DOF=2 (3-point fits); 2.0 is too
                 # tight and flags ~14% of valid fits as YELLOW by sampling
                 # variance alone.
                 green_chi2_threshold: float = 5.0,
                 yellow_t2_disagree_threshold: float = 0.20):
        """Store configuration and pre-build the Canary probe circuit set
        (shared across every round) once at construction time."""
        self.qubit_ids = list(qubit_ids)
        self.profiles = profiles
        self.interval_n = interval_n
        self.shots_t1 = shots_t1
        self.shots_ramsey = shots_ramsey
        self.shots_gate = shots_gate
        self.shots_echo = shots_echo
        self.green_chi2_threshold = green_chi2_threshold
        self.yellow_t2_disagree_threshold = yellow_t2_disagree_threshold

        self.canary_circuits, self.canary_metadata = build_inline_probe_circuits(
            profiles, self.qubit_ids)
        self._history: list = []

    def prepare(self, algo_circuits: list) -> tuple:
        """Interleave the cached Canary probe circuits into algo_circuits
        and return (interleaved_circuit_list, round_map), logging a
        one-line job-preparation summary including estimated overhead."""
        interleaved, round_map = interleave_circuits(
            algo_circuits, self.canary_circuits, self.interval_n)

        n_algo = len(algo_circuits)
        n_rounds = len(round_map)
        n_canary = len(self.canary_circuits)
        total = len(interleaved)
        overhead_pct = (n_rounds * n_canary) / n_algo * 100 if n_algo > 0 else 0.0

        print(f"[Canary] Prepared job: {n_algo} algo circuits + "
              f"{n_rounds} Canary rounds × {n_canary} circuits = "
              f"{total} circuits total. ~{overhead_pct:.1f}% overhead.")
        return interleaved, round_map

    def process(self, job_results: list, round_map: list) -> tuple:
        """Split job_results back into algorithm vs. Canary results, run
        Lindblad inversion for every Canary round, append each round's
        OnlineInversionResult to self.history, print a compact per-round
        health update, and return (algo_results, health_rounds)."""
        algo_results, canary_results_per_round = parse_inline_results(
            job_results, round_map, self.canary_metadata)

        health_rounds = []
        for rmap, canary_counts in zip(round_map, canary_results_per_round):
            round_index = rmap["round_index"]
            result = online_inversion(
                canary_counts, self.canary_metadata, self.profiles,
                round_index=round_index,
                shots_t1=self.shots_t1, shots_ramsey=self.shots_ramsey,
                shots_gate=self.shots_gate, shots_echo=self.shots_echo)
            result.algo_indices = rmap["algo_indices"]
            result.green_chi2_threshold = self.green_chi2_threshold
            result.yellow_t2_disagree_threshold = self.yellow_t2_disagree_threshold

            self._history.append(result)
            health_rounds.append(result)

            table_first_line = result.summary_table().splitlines()[0]
            print(f"[Canary] Round {round_index}: {table_first_line}")
            health = result.health_summary()
            for qid in sorted(result.qubit_results.keys()):
                r = result.qubit_results[qid]
                arch = inv.ARCH_DEFAULTS.get(r.architecture, inv.ARCH_DEFAULTS["unknown"])
                s, u = arch["time_scale"], arch["display_unit"]
                print(f"  Q{qid}: {health[qid]} "
                      f"T1={r.T1_s*s:.2f}{u} T2={r.T2_s*s:.2f}{u} "
                      f"ε={r.epsilon_sx:.2e}")

        return algo_results, health_rounds

    @property
    def history(self) -> list:
        """All OnlineInversionResult objects produced by process() calls
        so far, in chronological (round) order."""
        return self._history

    def trajectory(self, qubit_id: int, param: str) -> tuple:
        """Extract a time series for one qubit / one parameter across
        rounds processed so far.

        param must be one of: "T1_s", "T2_s", "delta_omega", "epsilon_sx".
        Returns (round_indices, values, sigmas) as parallel lists.
        """
        sigma_attr = {
            "T1_s": "T1_sigma_s",
            "T2_s": "T2_sigma_s",
            "delta_omega": "delta_omega_sigma",
            "epsilon_sx": "epsilon_sx_sigma",
        }
        if param not in sigma_attr:
            raise ValueError(f"Unknown param {param!r}; expected one of "
                             f"{sorted(sigma_attr.keys())}")

        round_indices, values, sigmas = [], [], []
        for result in self._history:
            if qubit_id not in result.qubit_results:
                continue
            r = result.qubit_results[qubit_id]
            round_indices.append(result.round_index)
            values.append(getattr(r, param))
            sigmas.append(getattr(r, sigma_attr[param]))
        return round_indices, values, sigmas

    def export_csv(self, path: str) -> None:
        """Write the full monitoring history to a CSV file, one row per
        (round, qubit) pair."""
        fieldnames = [
            "round_index", "timestamp", "qubit_id",
            "T1_s", "T1_sigma_s", "T2_s", "T2_sigma_s",
            "T2_ramsey_s", "T2_echo_s",
            "delta_omega", "delta_omega_sigma",
            "epsilon_sx", "epsilon_sx_sigma",
            "t1_chi2_dof", "ramsey_chi2_dof", "gate_chi2_dof", "echo_chi2_dof",
            "health", "spam_source",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self._history:
                health = result.health_summary()
                for qid in sorted(result.qubit_results.keys()):
                    r = result.qubit_results[qid]
                    spam_source = (r.calibration_source.spam_source
                                   if r.calibration_source is not None
                                   else "literature_fallback")
                    writer.writerow({
                        "round_index": result.round_index,
                        "timestamp": result.timestamp,
                        "qubit_id": qid,
                        "T1_s": r.T1_s, "T1_sigma_s": r.T1_sigma_s,
                        "T2_s": r.T2_s, "T2_sigma_s": r.T2_sigma_s,
                        "T2_ramsey_s": r.T2_ramsey_s, "T2_echo_s": r.T2_echo_s,
                        "delta_omega": r.delta_omega,
                        "delta_omega_sigma": r.delta_omega_sigma,
                        "epsilon_sx": r.epsilon_sx,
                        "epsilon_sx_sigma": r.epsilon_sx_sigma,
                        "t1_chi2_dof": r.t1_chi2_dof,
                        "ramsey_chi2_dof": r.ramsey_chi2_dof,
                        "gate_chi2_dof": r.gate_chi2_dof,
                        "echo_chi2_dof": r.echo_chi2_dof,
                        "health": health[qid],
                        "spam_source": spam_source,
                    })


if __name__ == "__main__":
    import tempfile

    print("=" * 70)
    print("[Canary] 1b_online.py self-test (simulation only)")
    print("=" * 70)

    failures = []

    # 1. Profiles for qubits 0-4
    N_QUBITS = 5
    profiles = {q: inv.BackendProfile.from_architecture("superconducting")
                for q in range(N_QUBITS)}

    # 2. 50 dummy algo circuits
    from qiskit import QuantumCircuit
    N_ALGO = 50
    dummy_algo_circuits = []
    for i in range(N_ALGO):
        qc = QuantumCircuit(1, 1, name=f"algo_{i}")
        qc.h(0)
        qc.measure(0, 0)
        dummy_algo_circuits.append(qc)

    # 3. CanaryInterleaver
    INTERVAL_N = 10
    canary = CanaryInterleaver(
        qubit_ids=list(range(N_QUBITS)), profiles=profiles,
        interval_n=INTERVAL_N)
    n_canary_per_round = len(canary.canary_circuits)

    # 4. prepare()
    interleaved, round_map = canary.prepare(dummy_algo_circuits)
    expected_rounds = N_ALGO // INTERVAL_N
    expected_total = N_ALGO + expected_rounds * n_canary_per_round
    print(f"\n[1] prepare(): {len(round_map)} rounds "
          f"(expected {expected_rounds}), "
          f"{len(interleaved)} total circuits (expected {expected_total})")
    if len(round_map) != expected_rounds or len(interleaved) != expected_total:
        failures.append("prepare() circuit counts did not match expectations")

    # 5. Simulate job results
    T1_TRUE = 150e-6
    T2_TRUE = 90e-6
    DW_TRUE = 2*np.pi*5e3
    EPS_TRUE = 3.5e-4
    arch = inv.ARCH_DEFAULTS["superconducting"]
    p0_given_1 = arch["p0_given_1"]
    p1_given_0 = arch["p1_given_0"]

    seed_rng = np.random.default_rng(7)

    def spam(p):
        return p*(1.0 - p0_given_1) + (1.0 - p)*p1_given_0

    def c1(rng_obj, p, sh):
        n1 = int(rng_obj.binomial(sh, float(np.clip(p, 0, 1))))
        return {"0": sh-n1, "1": n1}

    def c0(rng_obj, p, sh):
        n0 = int(rng_obj.binomial(sh, float(np.clip(p, 0, 1))))
        return {"0": n0, "1": sh-n0}

    SHOTS_T1, SHOTS_RAMSEY, SHOTS_GATE, SHOTS_ECHO = 300, 1000, 500, 500

    def simulate_qubit_counts(meta, rng_obj):
        t1_delays = meta["t1_delays_s"]
        ramsey_delays = meta["ramsey_delays_s"]
        echo_delays = meta["echo_delays_s"]
        Nv = np.array(meta["gate_rep_N"], dtype=float)

        t1_counts = [c1(rng_obj, spam(float(inv.forward_t1(d, T1_TRUE))), SHOTS_T1)
                     for d in t1_delays]

        ramsey_counts = []
        for t in ramsey_delays:
            px, py = inv.forward_ramsey_xy(t, T2_TRUE, DW_TRUE)
            ramsey_counts.append(c1(rng_obj, px, SHOTS_RAMSEY))
            ramsey_counts.append(c1(rng_obj, py, SHOTS_RAMSEY))

        p0_gate = inv.forward_gate(Nv, EPS_TRUE)
        gate_counts = [c0(rng_obj, spam(p), SHOTS_GATE) for p in p0_gate]

        T2_echo_true = min(T2_TRUE, 2.0*T1_TRUE)
        echo_counts = [c1(rng_obj, spam(float(inv.forward_echo(t, T2_echo_true))), SHOTS_ECHO)
                       for t in echo_delays]

        return t1_counts + ramsey_counts + gate_counts + echo_counts

    synthetic_results = [None] * len(interleaved)
    for rmap in round_map:
        for i in rmap["algo_indices"]:
            synthetic_results[i] = {"0": 500, "1": 500}

        start = rmap["canary_start_idx"]
        for qid, meta in canary.canary_metadata.items():
            qubit_counts = simulate_qubit_counts(meta, seed_rng)
            offset = start + meta["circuit_start_idx"]
            for j, cd in enumerate(qubit_counts):
                synthetic_results[offset + j] = cd

    if any(r is None for r in synthetic_results):
        failures.append("synthetic_results has unfilled slots")

    # 6. process()
    print("\n[2] process(): per-round health updates")
    algo_results, health_rounds = canary.process(synthetic_results, round_map)

    if len(health_rounds) != expected_rounds:
        failures.append("process() did not produce the expected number of rounds")
    if len(algo_results) != expected_rounds:
        failures.append("process() algo_results round count mismatch")

    # 7. trajectory()
    print("\n[3] trajectory(qubit_id=0, param='T1_s')")
    rounds, T1_vals, T1_sigs = canary.trajectory(qubit_id=0, param="T1_s")
    for ridx, val, sig in zip(rounds, T1_vals, T1_sigs):
        print(f"  round {ridx}: T1 = {val*1e6:.2f} ± {sig*1e6:.2f} µs")
    if len(rounds) != expected_rounds:
        failures.append("trajectory() did not return one point per round")

    # 8. export_csv()
    print("\n[4] export_csv()")
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = str(pathlib.Path(tmpdir) / "canary_online_test.csv")
        canary.export_csv(csv_path)
        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))
        n_data_rows = len(rows) - 1
        expected_rows = expected_rounds * N_QUBITS
        print(f"  wrote {n_data_rows} data rows (expected {expected_rows})")
        if n_data_rows != expected_rows:
            failures.append("export_csv() row count mismatch")

    # 9. health_summary() should be mostly GREEN on consistent synthetic data.
    # Each probe fits only 3 delay points (dof=2), so chi2/dof crossing the
    # 5.0 GREEN threshold (95th pctile for DOF=2) by pure sampling variance
    # is still possible some of the time even when the forward model matches
    # the truth exactly — a single unlucky round of YELLOW is not itself a
    # bug. We check that the large
    # majority of qubit-rounds come back GREEN and none come back RED (RED
    # would indicate an actual fit failure, which should not happen here).
    print("\n[5] health_summary() check "
          "(expect mostly GREEN, never RED, on consistent synthetic data)")
    n_total = 0
    n_green = 0
    for result in health_rounds:
        health = result.health_summary()
        for qid, status in health.items():
            n_total += 1
            if status == "GREEN":
                n_green += 1
            if status == "RED":
                failures.append(
                    f"round {result.round_index} qubit {qid} health=RED "
                    f"(fit failure) — unexpected on consistent synthetic data")
        print(f"  round {result.round_index}: {health}")
    green_frac = n_green / n_total if n_total else 0.0
    print(f"  {n_green}/{n_total} qubit-rounds GREEN ({green_frac*100:.0f}%)")
    if green_frac < 0.6:
        failures.append(
            f"only {green_frac*100:.0f}% of qubit-rounds were GREEN, expected >= 60%")

    print("\n" + "=" * 70)
    if not failures:
        print("[Canary] 1b_online.py self-test PASSED")
    else:
        print("[Canary] 1b_online.py self-test FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
