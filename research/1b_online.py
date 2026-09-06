"""
1b_online.py — online, non-interrupting qubit health monitor.

Wraps 1_inversion.py's Lindblad inversion engine with an interleaving
layer that weaves Canary probe circuits into any running quantum
workflow's circuit list (VQE, QAOA, simulation, arbitrary circuits —
this file is not tied to any one algorithm). Canary probes run in the
same job submission, on the same qubits the algorithm already uses,
between batches of algorithm circuits. The algorithm's own circuits
and results are never touched or reordered.

health_summary() (per round, from InversionResult's own chi2/dof) and
drift_report() (cross-round, comparing fitted parameters against a
baseline) answer different questions and are not interchangeable: the
former flags a round whose own data doesn't fit the assumed Markovian
model (e.g. coherent gate error, 1/f dephasing); the latter flags a
qubit whose fitted T1/T2/delta_omega/epsilon_sx has genuinely moved
since baseline, which a well-fit round will not by itself reveal.
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
    "chi2_scaled_z",
]


def chi2_scaled_z(delta: float, sigma_cur: float, sigma_base: float,
                   chi2_cur: float = 1.0, chi2_base: float = 1.0) -> float:
    """Two-sample z-score with each side's sigma inflated by
    sqrt(max(chi2/dof, 1)) before differencing -- the standard
    chi2-rescaling of a fit's reported uncertainty when its reduced
    chi-square exceeds 1 (Bevington & Robinson, Data Reduction and
    Error Analysis). This keeps a round whose own fit was already
    flagged as poorly modelled (non-Markovian noise, coherent gate
    error -- see InversionResult.t1_chi2_dof etc.) from masquerading
    as spurious parameter drift: an inflated chi2 on either side
    widens that side's effective sigma and shrinks |z| accordingly.
    Returns nan if either input is non-finite or both sigmas are zero.
    """
    if not np.isfinite(delta):
        return float("nan")
    sig_c = sigma_cur * np.sqrt(max(chi2_cur, 1.0)) if np.isfinite(chi2_cur) else sigma_cur
    sig_b = sigma_base * np.sqrt(max(chi2_base, 1.0)) if np.isfinite(chi2_base) else sigma_base
    denom = np.sqrt(sig_c**2 + sig_b**2)
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return float(delta / denom)


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
                 yellow_t2_disagree_threshold: float = 0.20,
                 # |z| > 2 matches the two-sample criterion used for the
                 # IBM Heron r2 A/B stability check (5_ibm_hardware.py's
                 # compare_runs()) -- roughly a 95% two-sided normal bound.
                 drift_z_threshold: float = 2.0):
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
        self.drift_z_threshold = drift_z_threshold

        self.canary_circuits, self.canary_metadata = build_inline_probe_circuits(
            profiles, self.qubit_ids)
        self._history: list = []
        self._baseline: dict = {}
        self._baseline_round: dict = {}

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
        health update (including drift vs. baseline, once a qubit has
        more than one round on record), and return
        (algo_results, health_rounds)."""
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

            for qid in sorted(result.qubit_results.keys()):
                if qid not in self._baseline:
                    self._baseline[qid] = result.qubit_results[qid]
                    self._baseline_round[qid] = round_index

            table_first_line = result.summary_table().splitlines()[0]
            print(f"[Canary] Round {round_index}: {table_first_line}")
            health = result.health_summary()
            for qid in sorted(result.qubit_results.keys()):
                r = result.qubit_results[qid]
                arch = inv.ARCH_DEFAULTS.get(r.architecture, inv.ARCH_DEFAULTS["unknown"])
                s, u = arch["time_scale"], arch["display_unit"]
                # Identity, not round_index, distinguishes "this round just
                # became the baseline" from "any other round": interleave_
                # circuits() restarts round_index at 0 on every independent
                # prepare() call, so two different rounds across separate
                # monitoring sessions can share the same round_index.
                drift_str = ""
                if self._baseline[qid] is not result.qubit_results[qid]:
                    flagged = [k for k, v in self.drift_report(qid)["drifted"].items() if v]
                    if flagged:
                        drift_str = f"  DRIFT[{','.join(flagged)}]"
                print(f"  Q{qid}: {health[qid]} "
                      f"T1={r.T1_s*s:.2f}{u} T2={r.T2_s*s:.2f}{u} "
                      f"ε={r.epsilon_sx:.2e}{drift_str}")

        return algo_results, health_rounds

    @property
    def history(self) -> list:
        """All OnlineInversionResult objects produced by process() calls
        so far, in chronological (round) order."""
        return self._history

    def set_baseline(self, qubit_id: Optional[int] = None) -> None:
        """Freeze the most recently processed round's result as the
        drift-detection reference point for qubit_id (or every currently
        monitored qubit, if qubit_id is None).

        CanaryInterleaver sets this automatically the first time each
        qubit appears in a process()ed round, so calling this explicitly
        is only needed to re-baseline after a known, intentional change
        (e.g. a provider recalibration) -- otherwise drift_report() would
        keep comparing against the pre-recalibration normal.
        """
        if not self._history:
            raise ValueError("set_baseline: no rounds processed yet")
        latest = self._history[-1]
        qids = [qubit_id] if qubit_id is not None else list(latest.qubit_results.keys())
        for qid in qids:
            if qid not in latest.qubit_results:
                raise ValueError(f"set_baseline: qubit {qid} not in the most recent round")
            self._baseline[qid] = latest.qubit_results[qid]
            self._baseline_round[qid] = latest.round_index

    def drift_report(self, qubit_id: int) -> dict:
        """Compare qubit_id's most recent InversionResult against its
        stored baseline, returning a two-sample z-score per parameter --
        the same z_j = (A-B)/sqrt(sigma_A^2+sigma_B^2) construction used
        for the IBM Heron r2 A/B stability check in 5_ibm_hardware.py's
        compare_runs(). |z| > self.drift_z_threshold flags that parameter
        as drifted.

        This answers a different question than health_summary(): a round
        can be perfectly well-fit (GREEN) by its own chi2/dof and still
        be reported here as drifted, if the qubit's true parameters have
        genuinely moved since baseline -- chi2/dof alone cannot see that,
        since each round is fit independently with no reference to any
        other round's data.

        T1, delta_omega, and epsilon_sx sigmas are chi2-scaled before
        differencing (see chi2_scaled_z) so a round contaminated by
        non-Markovian noise or coherent gate error doesn't masquerade as
        drift. T2's combined sigma already carries an equivalent chi2
        inflation from lindblad_inversion's Ramsey/echo fusion (Eq. 12)
        and is used as reported.

        Raises ValueError if qubit_id has no baseline yet (call
        process() at least once, or set_baseline() explicitly).
        """
        if qubit_id not in self._baseline:
            raise ValueError(
                f"drift_report: no baseline recorded for qubit {qubit_id}; "
                f"call process() at least once or set_baseline() first")
        if not self._history or qubit_id not in self._history[-1].qubit_results:
            raise ValueError(
                f"drift_report: qubit {qubit_id} not present in the most recent round")

        base = self._baseline[qubit_id]
        cur = self._history[-1].qubit_results[qubit_id]

        z_scores = {
            "T1_s": chi2_scaled_z(cur.T1_s - base.T1_s, cur.T1_sigma_s, base.T1_sigma_s,
                                   cur.t1_chi2_dof, base.t1_chi2_dof),
            "T2_s": chi2_scaled_z(cur.T2_s - base.T2_s, cur.T2_sigma_s, base.T2_sigma_s),
            "delta_omega": chi2_scaled_z(
                cur.delta_omega - base.delta_omega,
                cur.delta_omega_sigma, base.delta_omega_sigma,
                cur.ramsey_chi2_dof, base.ramsey_chi2_dof),
            "epsilon_sx": chi2_scaled_z(
                cur.epsilon_sx - base.epsilon_sx,
                cur.epsilon_sx_sigma, base.epsilon_sx_sigma,
                cur.gate_chi2_dof, base.gate_chi2_dof),
        }
        drifted = {k: bool(np.isfinite(v) and abs(v) > self.drift_z_threshold)
                   for k, v in z_scores.items()}
        return dict(qubit_id=qubit_id,
                    round_index=self._history[-1].round_index,
                    baseline_round_index=self._baseline_round[qubit_id],
                    z_scores=z_scores, drifted=drifted,
                    any_drift=any(drifted.values()))

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
        return simulate_qubit_counts_with(meta, rng_obj, T1_TRUE, T2_TRUE, DW_TRUE, EPS_TRUE)

    def simulate_qubit_counts_with(meta, rng_obj, T1_true, T2_true, dw_true, eps_true):
        t1_delays = meta["t1_delays_s"]
        ramsey_delays = meta["ramsey_delays_s"]
        echo_delays = meta["echo_delays_s"]
        Nv = np.array(meta["gate_rep_N"], dtype=float)

        t1_counts = [c1(rng_obj, spam(float(inv.forward_t1(d, T1_true))), SHOTS_T1)
                     for d in t1_delays]

        ramsey_counts = []
        for t in ramsey_delays:
            px, py = inv.forward_ramsey_xy(t, T2_true, dw_true)
            ramsey_counts.append(c1(rng_obj, px, SHOTS_RAMSEY))
            ramsey_counts.append(c1(rng_obj, py, SHOTS_RAMSEY))

        p0_gate = inv.forward_gate(Nv, eps_true)
        gate_counts = [c0(rng_obj, spam(p), SHOTS_GATE) for p in p0_gate]

        T2_echo_true = min(T2_true, 2.0*T1_true)
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

    # 10. drift_report() should mostly NOT flag drift across 5 rounds drawn
    # from the SAME true parameters (independent shot noise only). At
    # |z|>2 this is only a ~95% one-sided bound per test, so a rare
    # false positive by chance is expected and not itself a bug -- this
    # is a smoke test, not the statistical false-positive calibration
    # (that belongs to 3_null_nonm.py). We just check flags stay rare.
    print("\n[6] drift_report() false-positive check "
          "(same truth every round, shot noise only)")
    n_checked = 0
    n_flagged = 0
    for qid in range(N_QUBITS):
        dr = canary.drift_report(qid)
        n_checked += len(dr["drifted"])
        n_flagged += sum(dr["drifted"].values())
        print(f"  Q{qid}: z={ {k: round(v, 2) for k, v in dr['z_scores'].items()} }")
    print(f"  {n_flagged}/{n_checked} (parameter, qubit) pairs flagged as drifted")
    if n_flagged > n_checked * 0.5:
        failures.append(
            f"drift_report() flagged {n_flagged}/{n_checked} pairs on identical-truth "
            f"data -- expected this to stay rare")

    # 11. drift_report() SHOULD flag a genuine parameter shift: re-run
    # qubit 0 with T1 dropped 150us -> 90us (a -40% change, well outside
    # sampling noise at these shot counts) while every other qubit keeps
    # its original truth. Confirms both true-positive sensitivity and
    # that set_baseline() correctly resets the reference point.
    print("\n[7] drift_report() true-positive check "
          "(Q0 T1 shifted 150us -> 90us; other qubits unchanged)")
    interleaved_drift, round_map_drift = canary.prepare([])
    synthetic_drift = [None] * len(interleaved_drift)
    for qid, meta in canary.canary_metadata.items():
        T1_this = 90e-6 if qid == 0 else T1_TRUE
        qubit_counts = simulate_qubit_counts_with(
            meta, seed_rng, T1_this, T2_TRUE, DW_TRUE, EPS_TRUE)
        offset = meta["circuit_start_idx"]
        for j, cd in enumerate(qubit_counts):
            synthetic_drift[offset + j] = cd
    canary.process(synthetic_drift, round_map_drift)

    dr0 = canary.drift_report(0)
    print(f"  Q0 (drifted)  : T1_hat vs baseline z = {dr0['z_scores']['T1_s']:+.2f}  "
          f"drifted={dr0['drifted']}")
    if not dr0["drifted"]["T1_s"]:
        failures.append(
            f"drift_report(0) did not flag a real 40% T1 drop "
            f"(z={dr0['z_scores']['T1_s']:.2f}, threshold={canary.drift_z_threshold})")

    dr1 = canary.drift_report(1)
    print(f"  Q1 (unchanged): T1_hat vs baseline z = {dr1['z_scores']['T1_s']:+.2f}  "
          f"drifted={dr1['drifted']}")

    canary.set_baseline(0)
    dr0_reset = canary.drift_report(0)
    print(f"  Q0 after set_baseline(0): z = {dr0_reset['z_scores']}")
    if dr0_reset["drifted"]["T1_s"] or abs(dr0_reset["z_scores"]["T1_s"]) > 1e-9:
        failures.append(
            f"set_baseline(0) did not reset Q0's drift reference "
            f"(z={dr0_reset['z_scores']['T1_s']:.2e})")

    print("\n" + "=" * 70)
    if not failures:
        print("[Canary] 1b_online.py self-test PASSED")
    else:
        print("[Canary] 1b_online.py self-test FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
