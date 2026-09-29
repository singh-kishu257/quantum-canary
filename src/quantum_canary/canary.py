"""The Canary control loop."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, Union

import numpy as np
from qiskit import QuantumCircuit

from .compiler import compile_native
from .config import CanaryConfig
from .decision import Decision, SequentialTest, Status
from .design import BatchDesigner
from .estimator import HealthModel
from .executors import Executor
from .probes import return_probe_observations, return_probe_program, z2_flags
from .program import LogicalProgram
from .resources import Resource, ions_of, pair, to_error, to_key
from .results import (
    AssignmentRun,
    BatchPlan,
    BatchResult,
    ProbeRun,
    expectation,
    log_signal,
    merge_frequencies,
    normalize_counts,
)

CircuitLike = Union[QuantumCircuit, LogicalProgram]


class Canary:
    """Online ion-health learning and qubit-to-ion assignment for trapped-ion QPUs."""

    def __init__(
        self,
        executor: Executor,
        config: CanaryConfig | None = None,
        model: HealthModel | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.executor = executor
        self.config = config or CanaryConfig()
        self.config.validate()
        self.clock = clock
        self.rng = np.random.default_rng(self.config.seed)
        self.model = model if model is not None else HealthModel(self.config, clock)
        self.model.config = self.config
        self.tester = SequentialTest(self.config)
        self.designer = BatchDesigner(self.config, self.rng)
        self.decisions: dict[Resource, Decision] = self.tester.decide(self.model)
        self.flagged_ions: set[int] = set()
        self.probe_shots: dict[int, int] = {}

    @property
    def ions(self) -> tuple[int, ...]:
        """Ions available to Canary."""
        return self.executor.ions

    def prepare(self, circuit: CircuitLike, name: str | None = None) -> LogicalProgram:
        """Convert a circuit to a logical program (no-op for programs)."""
        if isinstance(circuit, LogicalProgram):
            return circuit
        return LogicalProgram.from_circuit(circuit, self.config.optimization_level, name)

    def recommend(self, circuit: CircuitLike) -> tuple[int, ...]:
        """Best current logical-to-ion assignment under pessimistic scores."""
        return self.designer.best(self.prepare(circuit), self.model, self.decisions, self.ions)

    def compile(self, circuit: CircuitLike, assignment: Sequence[int], twirl: bool = False) -> QuantumCircuit:
        """Compile a circuit onto the given ions in IonQ native gates."""
        return compile_native(self.prepare(circuit), assignment, rng=self.rng, twirl=twirl)

    def degraded(self) -> tuple[Resource, ...]:
        """Resources currently judged degraded."""
        return tuple(r for r, d in self.decisions.items() if d.status is Status.DEGRADED)

    def run(
        self,
        circuit: CircuitLike,
        shots: int,
        symmetry: str | None = None,
        observables: Iterable[Sequence[int]] = (),
    ) -> BatchResult:
        """Run one Canary batch for a single circuit."""
        return self.run_many([circuit], shots, [symmetry], [observables])[0]

    def _validate_batch(
        self,
        circuits: Sequence[CircuitLike],
        shots: int,
        symmetry: Sequence[str | None] | str | None,
        observables: Sequence[Iterable[Sequence[int]]] | None,
    ) -> tuple[list[LogicalProgram], list[str | None], list[tuple]]:
        programs = [self.prepare(c) for c in circuits]
        if not programs:
            raise ValueError("no circuits given")
        if shots < 1:
            raise ValueError("shots must be positive")
        nprog = len(programs)
        syms = [symmetry] * nprog if symmetry is None or isinstance(symmetry, str) else list(symmetry)
        if len(syms) != nprog or any(s not in (None, "z2") for s in syms):
            raise ValueError("symmetry must be None, 'z2', or one such entry per circuit")
        extra = [tuple(o) for o in observables] if observables is not None else [()] * nprog
        if len(extra) != nprog:
            raise ValueError("observables needs one entry per circuit")
        return programs, syms, extra

    def _build_batch(
        self, programs: Sequence[LogicalProgram], shots: int
    ) -> tuple[list, list[QuantumCircuit], list[tuple[Any, ...]], int, list[tuple[int, int]]]:
        cfg = self.config
        slots = cfg.assignments_per_batch * cfg.twirls_per_assignment
        shots_per = max(1, math.ceil(shots / slots))
        if self.executor.max_shots is not None and shots_per > self.executor.max_shots:
            raise ValueError(f"{shots_per} shots per circuit exceeds the backend limit")
        designs = [
            self.designer.design(p, self.model, self.decisions, self.ions, cfg.assignments_per_batch)
            for p in programs
        ]
        circuits_out: list[QuantumCircuit] = []
        tags: list[tuple[Any, ...]] = []
        for pi, (prog, design) in enumerate(zip(programs, designs)):
            reps = math.ceil(slots / len(design))
            for ai, assignment in enumerate(design):
                for t in range(reps):
                    name = f"{prog.name}_p{pi}_a{ai}_t{t}"
                    circuits_out.append(compile_native(prog, assignment, rng=self.rng, twirl=cfg.twirl, name=name))
                    tags.append(("alg", pi, ai))
        probe_pairs = self._select_probes(programs, designs, len(circuits_out), shots_per)
        for i, j in probe_pairs:
            for depth in cfg.probe_depths:
                name = f"canary_return_{i}_{j}_{depth}"
                circuits_out.append(
                    compile_native(return_probe_program(depth), (i, j), rng=self.rng, twirl=cfg.twirl, name=name)
                )
                tags.append(("probe", (i, j), depth))
        return designs, circuits_out, tags, shots_per, probe_pairs

    def plan(
        self,
        circuits: CircuitLike | Sequence[CircuitLike],
        shots: int,
    ) -> BatchPlan:
        """Preview the next batch (circuits, shots, gate-shot cost) without running it or changing state."""
        items = [circuits] if isinstance(circuits, (QuantumCircuit, LogicalProgram)) else list(circuits)
        programs, _, _ = self._validate_batch(items, shots, None, None)
        state = self.rng.bit_generator.state
        try:
            designs, circuits_out, tags, shots_per, probe_pairs = self._build_batch(programs, shots)
        finally:
            self.rng.bit_generator.state = state
        one_q = sum(1 for qc in circuits_out for inst in qc.data if inst.operation.name in ("gpi", "gpi2"))
        two_q = sum(1 for qc in circuits_out for inst in qc.data if inst.operation.name in ("zz", "ms"))
        n_probe = sum(1 for t in tags if t[0] == "probe")
        return BatchPlan(
            circuits=circuits_out,
            shots_per_circuit=shots_per,
            algorithm_circuits=len(circuits_out) - n_probe,
            probe_circuits=n_probe,
            probe_pairs=tuple(probe_pairs),
            assignments=tuple(tuple(d) for d in designs),
            single_qubit_gate_shots=one_q * shots_per,
            two_qubit_gate_shots=two_q * shots_per,
        )

    def run_many(
        self,
        circuits: Sequence[CircuitLike],
        shots: int,
        symmetry: Sequence[str | None] | str | None = None,
        observables: Sequence[Iterable[Sequence[int]]] | None = None,
    ) -> list[BatchResult]:
        """Run one co-scheduled batch (one job) for several circuits; shots are per circuit."""
        programs, syms, extra = self._validate_batch(circuits, shots, symmetry, observables)
        cfg = self.config
        designs, circuits_out, tags, shots_per, probe_pairs = self._build_batch(programs, shots)
        raw = self.executor.run(circuits_out, shots_per)
        if len(raw) != len(circuits_out):
            raise RuntimeError("executor returned a different number of results")
        now = float(self.clock())
        freqs = [normalize_counts(r) for r in raw]
        alg_parts: dict[tuple[int, int], list[dict[str, float]]] = {}
        probe_runs: list[ProbeRun] = []
        for tag, f in zip(tags, freqs):
            if tag[0] == "alg":
                alg_parts.setdefault((tag[1], tag[2]), []).append(f)
            else:
                probe_runs.append(ProbeRun(tag[1], tag[2], f, shots_per))
        merged: dict[tuple[int, int], tuple[dict[str, float], int]] = {}
        for key, parts in alg_parts.items():
            merged[key] = (merge_frequencies([(p, shots_per) for p in parts]), shots_per * len(parts))
        self._learn(programs, designs, merged, probe_runs, extra, now)
        flagged: set[int] = set()
        for pi, (prog, design) in enumerate(zip(programs, designs)):
            if syms[pi] == "z2":
                for ai, assignment in enumerate(design):
                    f, n = merged[(pi, ai)]
                    flagged |= z2_flags(prog, assignment, f, n, cfg.symmetry_z)
        self.flagged_ions = flagged
        for i, j in probe_pairs:
            cost = shots_per * len(cfg.probe_depths)
            self.probe_shots[i] = self.probe_shots.get(i, 0) + cost
            self.probe_shots[j] = self.probe_shots.get(j, 0) + cost
        degraded = set(self.degraded())
        job_ids = tuple(getattr(self.executor, "last_job_ids", ()))
        results = []
        for pi, (prog, design) in enumerate(zip(programs, designs)):
            runs = []
            for ai, assignment in enumerate(design):
                f, n = merged[(pi, ai)]
                trusted = not (prog.used_resources(assignment) & degraded)
                runs.append(AssignmentRun(assignment, f, n, trusted))
            results.append(
                BatchResult(
                    name=prog.name,
                    runs=runs,
                    recommended=self.recommend(prog),
                    probes=probe_runs,
                    flagged_ions=tuple(sorted(flagged)),
                    degraded=tuple(sorted(degraded, key=to_key)),
                    timestamp=now,
                    job_ids=job_ids,
                )
            )
        return results

    def _learn(
        self,
        programs: Sequence[LogicalProgram],
        designs: Sequence[Sequence[tuple[int, ...]]],
        merged: dict[tuple[int, int], tuple[dict[str, float], int]],
        probe_runs: Sequence[ProbeRun],
        extra: Sequence[Iterable[Sequence[int]]],
        now: float,
    ) -> None:
        cfg = self.config
        self.model.decay(now)
        for pi, (prog, design) in enumerate(zip(programs, designs)):
            if len(design) < 2:
                continue
            for obs in prog.observables(extra[pi]):
                group, signs = [], []
                for ai, assignment in enumerate(design):
                    f, n = merged[(pi, ai)]
                    m = expectation(f, obs.clbits)
                    sig = log_signal(m, n, cfg.model_sd, cfg.min_signal_z, cfg.min_abs_signal)
                    if sig is None:
                        continue
                    group.append((prog.exposure(obs, assignment), sig[0], sig[1]))
                    signs.append(m > 0)
                if len(group) >= 2 and (all(signs) or not any(signs)):
                    self.model.add_relative(group)
        absolute = []
        for run in probe_runs:
            absolute.extend(return_probe_observations(run.ions, run.depth, run.frequencies, run.shots, cfg.model_sd))
        self.model.add_absolute(absolute)
        self.decisions = self.tester.decide(self.model)

    def _select_probes(
        self,
        programs: Sequence[LogicalProgram],
        designs: Sequence[Sequence[tuple[int, ...]]],
        n_alg: int,
        shots_per: int,
    ) -> list[tuple[int, int]]:
        cfg = self.config
        budget = int(math.floor(cfg.probe_fraction * n_alg / len(cfg.probe_depths)))
        if budget <= 0:
            return []
        cost = shots_per * len(cfg.probe_depths)
        scores: dict[Resource, float] = {}
        for prog, design in zip(programs, designs):
            for rank, a in enumerate(design):
                share = 1.0 if rank == 0 else 1.0 / len(design)
                for (u, v), w in prog.interaction_weights.items():
                    r = pair(a[u], a[v])
                    d = self.decisions.get(r)
                    if d is not None and d.status is not Status.UNDECIDED:
                        continue
                    s = share * w * max(self.tester.probability_degraded(self.model, r), 1e-3)
                    if a[u] in self.flagged_ions or a[v] in self.flagged_ions:
                        s *= 10.0
                    scores[r] = scores.get(r, 0.0) + s
        chosen: list[tuple[int, int]] = []
        for r in sorted(scores, key=lambda r: (-scores[r], to_key(r))):
            i, j = r[1], r[2]
            if max(self.probe_shots.get(i, 0), self.probe_shots.get(j, 0)) + cost > cfg.max_probe_shots_per_ion:
                continue
            chosen.append((i, j))
            if len(chosen) >= budget:
                break
        return chosen

    def probe(self, pairs: Iterable[tuple[int, int]], shots: int) -> list[ProbeRun]:
        """Run ZZ return probes on explicit ion pairs (shots per circuit) and update health."""
        allowed = set(self.ions)
        todo = [(min(i, j), max(i, j)) for i, j in pairs]
        if not todo:
            return []
        if any(i == j or i not in allowed or j not in allowed for i, j in todo):
            raise ValueError("probe pairs must be distinct available ions")
        circuits, tags = [], []
        for i, j in todo:
            for depth in self.config.probe_depths:
                circuits.append(
                    compile_native(
                        return_probe_program(depth), (i, j), rng=self.rng, twirl=self.config.twirl, name=f"canary_return_{i}_{j}_{depth}"
                    )
                )
                tags.append(((i, j), depth))
        raw = self.executor.run(circuits, int(shots))
        now = float(self.clock())
        runs = [ProbeRun(t[0], t[1], normalize_counts(r), int(shots)) for t, r in zip(tags, raw)]
        self._learn([], [], {}, runs, [], now)
        for i, j in todo:
            cost = int(shots) * len(self.config.probe_depths)
            self.probe_shots[i] = self.probe_shots.get(i, 0) + cost
            self.probe_shots[j] = self.probe_shots.get(j, 0) + cost
        return runs

    def run_fixed(
        self, circuit: CircuitLike, assignment: Sequence[int], shots: int, twirls: int = 0
    ) -> BatchResult:
        """Baseline: run on a fixed assignment without learning (twirls=0 disables twirling)."""
        prog = self.prepare(circuit)
        assignment = tuple(int(x) for x in assignment)
        if not set(assignment) <= set(self.ions):
            raise ValueError("assignment uses unavailable ions")
        if twirls > 0:
            per = max(1, math.ceil(shots / twirls))
            circuits = [
                compile_native(prog, assignment, rng=self.rng, twirl=True, name=f"{prog.name}_fixed_t{t}")
                for t in range(twirls)
            ]
        else:
            per = int(shots)
            circuits = [compile_native(prog, assignment, twirl=False, name=f"{prog.name}_fixed")]
        raw = self.executor.run(circuits, per)
        f = merge_frequencies([(normalize_counts(r), per) for r in raw])
        run = AssignmentRun(assignment, f, per * len(circuits), True)
        return BatchResult(
            name=prog.name,
            runs=[run],
            recommended=assignment,
            timestamp=float(self.clock()),
            job_ids=tuple(getattr(self.executor, "last_job_ids", ())),
        )

    def report(self) -> list[dict[str, Any]]:
        """Per-resource health estimates, sorted from worst to best.

        'error' is process infidelity for ion pairs and symmetric readout-flip probability for ions.
        'probe_calibrated' is False when an estimate rests only on algorithm contrasts, whose
        lightcone exposure model can dilute damping; treat such values as effective lower bounds.
        """
        rows = []
        for r, (mean, sd) in self.model.estimates().items():
            d = self.decisions.get(r)
            rows.append(
                {
                    "resource": to_key(r),
                    "kind": r[0],
                    "ions": ions_of(r),
                    "error": to_error(r, mean),
                    "error_upper": to_error(r, mean + 2.0 * sd),
                    "damping": mean,
                    "damping_sd": sd,
                    "probe_calibrated": self.model.probe_information(r) > 0.0,
                    "status": (d.status if d else Status.UNDECIDED).value,
                    "llr": d.llr if d else 0.0,
                    "llr_probe": d.llr_probe if d else 0.0,
                }
            )
        return sorted(rows, key=lambda row: (-row["error"], row["resource"]))

    def save(self, path: str | Path) -> None:
        """Persist configuration and learned state to JSON."""
        data = {
            "version": 1,
            "config": self.config.to_dict(),
            "model": self.model.to_dict(),
            "probe_shots": {str(k): v for k, v in self.probe_shots.items()},
            "flagged_ions": sorted(self.flagged_ions),
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(
        cls, path: str | Path, executor: Executor, clock: Callable[[], float] = time.time
    ) -> "Canary":
        """Restore a Canary saved with save()."""
        data = json.loads(Path(path).read_text())
        config = CanaryConfig.from_dict(data["config"])
        model = HealthModel.from_dict(data["model"], config, clock)
        canary = cls(executor, config, model, clock)
        canary.probe_shots = {int(k): int(v) for k, v in data.get("probe_shots", {}).items()}
        canary.flagged_ions = set(int(i) for i in data.get("flagged_ions", []))
        return canary
