"""The Canary control loop.

Every batch is one job containing

* algorithm circuits: the user's circuit under ``assignments_per_batch`` logical-to-ion assignments (one pessimistic
  exploit choice, the rest Thompson-sampled), each compiled ``twirls_per_assignment`` times with independent
  randomized-compiling frames and paired complementary measurement twirls; and
* Tier-1 calibration circuits: cycle-benchmarking probes of the ion pairs whose calibration would most reduce the
  uncertainty of the batch's assignment costs, a SPAM probe of every available ion, and optional ZZ phase probes.

Only Tier-1 circuits update the health model and the verdicts. Algorithm results (Tier-2) are returned to the user,
checked for consistency against the calibrated rates, and used for a layout-diversity extrapolation.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, Union

import numpy as np
from qiskit import QuantumCircuit

from .compiler import AngleOffset, CompiledCircuit, compile_native, compile_twirled
from .config import CanaryConfig
from .decision import Decision, HealthTest, Status
from .design import BatchDesigner
from .estimator import CoherentModel, HealthModel
from .executors import Executor
from .probes import (
    BASIS_PAIRS,
    Observation,
    bonferroni_z,
    cb_estimate,
    cb_probe_program,
    cb_variance,
    phase_estimate,
    phase_probe_program,
    spam_observation,
    spam_probe_program,
    z2_flags,
)
from .program import LogicalProgram
from .resources import Resource, from_key, ion, ions_of, pair, to_error, to_key
from .results import (
    AssignmentRun,
    BatchPlan,
    BatchResult,
    ProbeRun,
    Tier2Fit,
    chi2_sf,
    expectation,
    merge_frequencies,
    normalize_counts,
)

CircuitLike = Union[QuantumCircuit, LogicalProgram]


@dataclass
class _Job:
    """One submitted circuit and what it is for."""

    kind: str
    compiled: CompiledCircuit
    program: int = -1
    assignment: int = -1
    ions: tuple[int, ...] = ()
    bases: tuple[str, ...] = ()
    depth: int = 0
    angle: float = 0.0
    setting: str = ""
    group: int = -1


def _flip_sets(clbits: Sequence[int], count: int, rng: np.random.Generator) -> list[tuple[int, ...]]:
    """Measurement-twirl flip sets; consecutive sets are complementary so each clbit is flipped in half of them."""
    bits_order = sorted(int(c) for c in clbits)
    out: list[tuple[int, ...]] = []
    while len(out) < count:
        bits = rng.integers(2, size=len(bits_order))
        out.append(tuple(c for c, b in zip(bits_order, bits) if b))
        if len(out) < count:
            out.append(tuple(c for c, b in zip(bits_order, bits) if not b))
    return out


def _proper_split(clbits: Sequence[int], rng: np.random.Generator) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Complementary flip sets that are both non-empty, so neither circuit is gate-free."""
    order = sorted(int(c) for c in clbits)
    if len(order) < 2:
        raise ValueError("a proper split needs at least two clbits")
    while True:
        bits = rng.integers(2, size=len(order))
        if 0 < int(bits.sum()) < len(order):
            return (
                tuple(c for c, b in zip(order, bits) if b),
                tuple(c for c, b in zip(order, bits) if not b),
            )


def _linear_fit(qubits: tuple[int, ...], xs: Sequence[float], ys: Sequence[float], vs: Sequence[float]) -> Tier2Fit:
    """Weighted straight-line fit y = a + b x; the intercept error is inflated by the Birge ratio when > 1."""
    fit = Tier2Fit(qubits, tuple(float(x) for x in xs), tuple(float(y) for y in ys), tuple(float(v) for v in vs))
    x, y, v = np.array(xs, dtype=float), np.array(ys, dtype=float), np.array(vs, dtype=float)
    k = len(x)
    if k < 2 or float(np.ptp(x)) < 1e-9:
        return fit
    w = 1.0 / v
    design = np.column_stack([np.ones(k), x])
    info = design.T @ (w[:, None] * design)
    cov = np.linalg.inv(info)
    beta = cov @ (design.T @ (w * y))
    resid = y - design @ beta
    fit.mitigated, fit.slope = float(beta[0]), float(beta[1])
    scale = 1.0
    if k > 2:
        chi2 = float(np.sum(w * resid**2))
        fit.chi2, fit.dof = chi2, k - 2
        fit.p_value = chi2_sf(chi2, k - 2)
        scale = max(1.0, chi2 / (k - 2))
    fit.mitigated_stderr = float(math.sqrt(cov[0, 0] * scale))
    return fit


class Canary:
    """Online ion-health learning and qubit-to-ion assignment for trapped-ion QPUs."""

    def __init__(
        self,
        executor: Executor,
        config: CanaryConfig | None = None,
        model: HealthModel | None = None,
        clock: Callable[[], float] = time.time,
        coherent: CoherentModel | None = None,
    ) -> None:
        self.executor = executor
        self.config = config or CanaryConfig()
        self.config.validate()
        self.clock = clock
        self.rng = np.random.default_rng(self.config.seed)
        self.model = model if model is not None else HealthModel(self.config, clock)
        self.model.config = self.config
        self.coherent = coherent if coherent is not None else CoherentModel(self.config, clock)
        self.coherent.config = self.config
        self.tester = HealthTest(self.config)
        self.designer = BatchDesigner(self.config, self.rng)
        self.flagged_ions: set[int] = set()
        self._tokens: dict[int, tuple[float, float]] = {}
        self._phase_last: dict[Resource, float] = {}
        self._group = 0
        self.decisions: dict[Resource, Decision] = self.tester.decide(self.model, self.family_size, self.clock())

    # ------------------------------------------------------------------ basics

    @property
    def ions(self) -> tuple[int, ...]:
        """Ions available to Canary."""
        return self.executor.ions

    @property
    def family_size(self) -> int:
        """Number of hypotheses (all ions and ion pairs) that the e-BH verdicts are corrected for."""
        n = len(self.ions)
        return n + n * (n - 1) // 2

    def prepare(self, circuit: CircuitLike, name: str | None = None) -> LogicalProgram:
        """Convert a circuit to a logical program (no-op for programs)."""
        if isinstance(circuit, LogicalProgram):
            return circuit
        return LogicalProgram.from_circuit(circuit, self.config.optimization_level, name)

    def recommend(self, circuit: CircuitLike) -> tuple[int, ...]:
        """Best current logical-to-ion assignment under pessimistic calibrated costs."""
        return self.designer.best(self.prepare(circuit), self.model, self.decisions, self.ions, now=self.clock())

    def compile(self, circuit: CircuitLike, assignment: Sequence[int], twirl: str | bool = False) -> QuantumCircuit:
        """Compile a circuit onto the given ions in IonQ native gates (angle corrections applied if enabled)."""
        offset = self._angle_offset(self.clock())
        return compile_native(self.prepare(circuit), assignment, rng=self.rng, twirl=twirl, angle_offset=offset)

    def degraded(self) -> tuple[Resource, ...]:
        """Resources currently judged degraded."""
        return tuple(r for r, d in self.decisions.items() if d.status is Status.DEGRADED)

    def _angle_offset(self, now: float) -> AngleOffset | None:
        if not self.config.angle_correction:
            return None
        coherent = self.coherent

        def offset(a: int, b: int, theta: float) -> float:
            return coherent.correction(pair(a, b), theta, now)

        return offset

    def _available(self, i: int, now: float) -> float:
        cap = float(self.config.max_probe_shots_per_ion)
        level, t = self._tokens.get(int(i), (cap, now))
        return min(cap, level + cap * max(now - t, 0.0) / self.config.half_life_s)

    def _consume(self, i: int, amount: float, now: float) -> None:
        self._tokens[int(i)] = (max(self._available(i, now) - float(amount), 0.0), now)

    def _next_group(self) -> int:
        self._group += 1
        return self._group

    # ------------------------------------------------------------------ circuits

    def _random_bases(self, k: int) -> list[tuple[str, str]]:
        picks = self.rng.choice(len(BASIS_PAIRS), size=min(int(k), len(BASIS_PAIRS)), replace=False)
        return [BASIS_PAIRS[int(p)] for p in picks]

    def _cb_jobs(self, ions_ij: tuple[int, int], bases_list: Iterable[tuple[str, str]], offset: AngleOffset | None) -> list[_Job]:
        jobs = []
        for bases in bases_list:
            group = self._next_group()
            flips = tuple(c for c in (0, 1) if self.rng.random() < 0.5)
            frame = (float(self.rng.uniform(0.0, 2.0 * math.pi)), float(self.rng.uniform(0.0, 2.0 * math.pi)))
            for depth in self.config.probe_depths:
                comp = compile_twirled(
                    cb_probe_program(bases, depth, frame),
                    ions_ij,
                    self.rng,
                    "rc",
                    flips,
                    offset,
                    f"canary_cb_{ions_ij[0]}_{ions_ij[1]}_{bases[0]}{bases[1]}_{depth}",
                )
                jobs.append(_Job("cb", comp, ions=ions_ij, bases=bases, depth=depth, group=group))
        return jobs

    def _spam_ions(self, used: Iterable[int]) -> tuple[int, ...]:
        ions_set = set(int(i) for i in used)
        for i in self.ions:
            if len(ions_set) >= 2:
                break
            ions_set.add(int(i))
        return tuple(sorted(ions_set))

    def _spam_jobs(self, ions_set: tuple[int, ...]) -> list[_Job]:
        group = self._next_group()
        program = spam_probe_program(len(ions_set))
        jobs = []
        for flips in _proper_split(range(len(ions_set)), self.rng):
            comp = compile_twirled(program, ions_set, self.rng, "none", flips, None, f"canary_spam_{len(ions_set)}")
            jobs.append(_Job("spam", comp, ions=ions_set, group=group))
        return jobs

    def _phase_jobs(self, ions_ij: tuple[int, int]) -> list[_Job]:
        jobs = []
        for angle in self.config.phase_angles:
            group = self._next_group()
            for setting in ("x", "yz"):
                program = phase_probe_program(angle, self.config.phase_repetitions, setting)
                for flips in _flip_sets((0, 1), 2, self.rng):
                    comp = compile_twirled(
                        program, ions_ij, self.rng, "commutant", flips, None,
                        f"canary_phase_{ions_ij[0]}_{ions_ij[1]}_{setting}",
                    )
                    jobs.append(_Job("phase", comp, ions=ions_ij, angle=angle, setting=setting, group=group))
        return jobs

    def _spam_factors(self, a: int, b: int, now: float) -> tuple[float, float, float]:
        """Predicted SPAM factors of <Z_a>, <Z_b> and <Z_a Z_b> from the current SPAM posteriors."""
        sa = max(self.model.estimate(ion(a), now)[0], 0.0)
        sb = max(self.model.estimate(ion(b), now)[0], 0.0)
        return math.exp(-sa), math.exp(-sb), math.exp(-sa - sb)

    def _select_probes(
        self,
        programs: Sequence[LogicalProgram],
        designs: Sequence[Sequence[tuple[int, ...]]],
        n_alg: int,
        shots_per: int,
        now: float,
    ) -> list[tuple[int, int]]:
        cfg = self.config
        budget = int(math.floor(cfg.probe_fraction * n_alg / len(cfg.probe_depths)))
        if budget <= 0:
            return []
        cost = shots_per * len(cfg.probe_depths)
        stake: dict[Resource, float] = {}
        for prog, design in zip(programs, designs):
            weights = prog.weighted_interactions(cfg.angle_exponent)
            for rank, a in enumerate(design):
                share = 1.0 if rank == 0 else 1.0 / len(design)
                for (u, v), w in weights.items():
                    r = pair(a[u], a[v])
                    stake[r] = stake.get(r, 0.0) + share * w
        scores: dict[Resource, float] = {}
        for r, s in stake.items():
            d = self.decisions.get(r)
            if d is not None and d.status is Status.DEGRADED:
                continue
            mean, sd = self.model.estimate(r, now)
            spam = self._spam_factors(r[1], r[2], now)
            noise = cb_variance(max(mean, 0.0), cfg.probe_depths, shots_per, spam, cfg.model_sd)
            gain = s * sd**4 / (sd**2 + noise)
            if r[1] in self.flagged_ions or r[2] in self.flagged_ions:
                gain *= 2.0
            scores[r] = gain
        chosen: list[tuple[int, int]] = []
        for r in sorted(scores, key=lambda r: (-scores[r], to_key(r))):
            i, j = int(r[1]), int(r[2])
            if min(self._available(i, now), self._available(j, now)) < cost:
                continue
            chosen.append((i, j))
            if len(chosen) >= budget:
                break
        return chosen

    def _select_phase_pairs(
        self, designs: Sequence[Sequence[tuple[int, ...]]], programs: Sequence[LogicalProgram], n_alg: int, shots_per: int, now: float
    ) -> list[tuple[int, int]]:
        cfg = self.config
        per_pair = 4 * len(cfg.phase_angles)
        budget = int(math.floor(cfg.phase_probe_fraction * n_alg / per_pair))
        if budget <= 0:
            return []
        candidates: set[Resource] = set()
        for prog, design in zip(programs, designs):
            if design:
                a = design[0]
                candidates.update(pair(a[u], a[v]) for u, v in prog.interaction_weights)
        cost = shots_per * per_pair
        order = sorted(candidates, key=lambda r: (self._phase_last.get(r, -math.inf), to_key(r)))
        chosen: list[tuple[int, int]] = []
        for r in order:
            i, j = int(r[1]), int(r[2])
            if min(self._available(i, now), self._available(j, now)) < cost:
                continue
            chosen.append((i, j))
            if len(chosen) >= budget:
                break
        return chosen

    def _build_batch(
        self, programs: Sequence[LogicalProgram], shots: int, now: float
    ) -> tuple[list[list[tuple[int, ...]]], list[_Job], int, list[tuple[int, int]], list[tuple[int, int]]]:
        cfg = self.config
        slots = cfg.assignments_per_batch * cfg.twirls_per_assignment
        shots_per = max(1, math.ceil(shots / slots))
        if self.executor.max_shots is not None and shots_per > self.executor.max_shots:
            raise ValueError(f"{shots_per} shots per circuit exceeds the backend limit")
        designs = [
            self.designer.design(p, self.model, self.decisions, self.ions, cfg.assignments_per_batch, now)
            for p in programs
        ]
        offset = self._angle_offset(now)
        jobs: list[_Job] = []
        for pi, (prog, design) in enumerate(zip(programs, designs)):
            reps = math.ceil(slots / len(design))
            clbits = [c for _, c in prog.measurements]
            for ai, assignment in enumerate(design):
                flip_sets = _flip_sets(clbits, reps, self.rng) if cfg.measure_twirl else [()] * reps
                for t in range(reps):
                    comp = compile_twirled(
                        prog, assignment, self.rng, cfg.twirl, flip_sets[t], offset, f"{prog.name}_p{pi}_a{ai}_t{t}"
                    )
                    jobs.append(_Job("alg", comp, program=pi, assignment=ai))
        n_alg = len(jobs)
        probe_pairs: list[tuple[int, int]] = []
        phase_pairs: list[tuple[int, int]] = []
        if cfg.probe_fraction > 0:
            probe_pairs = self._select_probes(programs, designs, n_alg, shots_per, now)
            for ions_ij in probe_pairs:
                jobs.extend(self._cb_jobs(ions_ij, self._random_bases(1), offset))
            if cfg.spam_probe:
                jobs.extend(self._spam_jobs(self._spam_ions(self.ions)))
            phase_pairs = self._select_phase_pairs(designs, programs, n_alg, shots_per, now)
            for ions_ij in phase_pairs:
                jobs.extend(self._phase_jobs(ions_ij))
        return designs, jobs, shots_per, probe_pairs, phase_pairs

    # ------------------------------------------------------------------ learning

    def _learn_calibration(self, jobs: Sequence[_Job], freqs: Sequence[dict[str, float]], shots: int, now: float) -> list[ProbeRun]:
        cfg = self.config
        runs: list[ProbeRun] = []
        cb: dict[int, tuple[tuple[int, ...], list[tuple[int, dict[str, float], float]]]] = {}
        spam: dict[int, tuple[tuple[int, ...], list[dict[str, float]]]] = {}
        phase: dict[int, tuple[tuple[int, ...], float, dict[str, list[dict[str, float]]]]] = {}
        for job, f in zip(jobs, freqs):
            if job.kind == "alg":
                continue
            runs.append(ProbeRun(job.ions, job.depth, f, shots, job.kind, job.bases, job.setting, job.angle))
            if job.kind == "cb":
                cb.setdefault(job.group, (job.ions, []))[1].append((job.depth, f, float(shots)))
            elif job.kind == "spam":
                spam.setdefault(job.group, (job.ions, []))[1].append(f)
            elif job.kind == "phase":
                phase.setdefault(job.group, (job.ions, job.angle, {}))[2].setdefault(job.setting, []).append(f)
        per_pair: dict[Resource, list[float]] = {}
        for ions_ij, parts in cb.values():
            est = cb_estimate(parts)
            if est is not None:
                per_pair.setdefault(pair(*ions_ij), []).append(est)
        for r, ests in per_pair.items():
            spam_factors = self._spam_factors(r[1], r[2], now)
            obs = Observation(
                now, float(np.mean(ests)), "cb", float(shots), tuple(cfg.probe_depths), spam_factors, len(ests),
                model_sd=cfg.model_sd,
            )
            self.model.record(r, obs)
        for ions_set, parts in spam.values():
            merged = merge_frequencies([(p, shots) for p in parts])
            n = shots * len(parts)
            for c, i in enumerate(ions_set):
                self.model.record(ion(i), spam_observation(merged, n, c, cfg.spam_model_sd, now))
        for ions_ij, angle, settings in phase.values():
            if "x" not in settings or "yz" not in settings:
                continue
            fx = merge_frequencies([(p, shots) for p in settings["x"]])
            fy = merge_frequencies([(p, shots) for p in settings["yz"]])
            est = phase_estimate(
                fx, shots * len(settings["x"]), fy, shots * len(settings["yz"]), angle, cfg.phase_repetitions
            )
            if est is not None:
                self.coherent.update(pair(*ions_ij), angle, est[0], est[1], now)
        return runs

    def _after_calibration(self, now: float) -> None:
        if self.config.empirical_bayes:
            self.model.refresh_population(now)
        self.model.prune(now)
        self.decisions = self.tester.decide(self.model, self.family_size, now)

    def _predicted_damping(self, exposure: dict[Resource, float], now: float) -> float:
        return float(sum(w * self.model.estimate(r, now)[0] for r, w in exposure.items()))

    def _tier2(
        self,
        prog: LogicalProgram,
        design: Sequence[tuple[int, ...]],
        merged: dict[tuple[int, int], tuple[dict[str, float], int]],
        pi: int,
        extra: Iterable[Sequence[int]],
        now: float,
    ) -> dict[tuple[int, ...], Tier2Fit]:
        fits: dict[tuple[int, ...], Tier2Fit] = {}
        if len(design) < 2:
            return fits
        gamma = self.config.angle_exponent
        for obs in prog.observables(extra):
            xs, ys, vs = [], [], []
            for ai, assignment in enumerate(design):
                f, n = merged[(pi, ai)]
                m = expectation(f, obs.clbits)
                xs.append(self._predicted_damping(prog.exposure(obs, assignment, gamma), now))
                ys.append(m)
                vs.append(max(1.0 - m * m, 1.0 / n) / n)
            fits[obs.qubits] = _linear_fit(obs.qubits, xs, ys, vs)
        return fits

    # ------------------------------------------------------------------ public API

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

    def plan(self, circuits: CircuitLike | Sequence[CircuitLike], shots: int) -> BatchPlan:
        """Preview the next batch (circuits, shots, gate-shot cost) without running it or changing state."""
        items = [circuits] if isinstance(circuits, (QuantumCircuit, LogicalProgram)) else list(circuits)
        programs, _, _ = self._validate_batch(items, shots, None, None)
        state = self.rng.bit_generator.state
        group = self._group
        try:
            designs, jobs, shots_per, probe_pairs, phase_pairs = self._build_batch(programs, shots, self.clock())
        finally:
            self.rng.bit_generator.state = state
            self._group = group
        circuits_out = [j.compiled.circuit for j in jobs]
        one_q = sum(1 for qc in circuits_out for inst in qc.data if inst.operation.name in ("gpi", "gpi2"))
        two_q = sum(1 for qc in circuits_out for inst in qc.data if inst.operation.name in ("zz", "ms"))
        count = {k: sum(1 for j in jobs if j.kind == k) for k in ("alg", "cb", "spam", "phase")}
        return BatchPlan(
            circuits=circuits_out,
            shots_per_circuit=shots_per,
            algorithm_circuits=count["alg"],
            probe_circuits=count["cb"],
            probe_pairs=tuple(probe_pairs),
            assignments=tuple(tuple(d) for d in designs),
            single_qubit_gate_shots=one_q * shots_per,
            two_qubit_gate_shots=two_q * shots_per,
            spam_circuits=count["spam"],
            phase_circuits=count["phase"],
            phase_pairs=tuple(phase_pairs),
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
        designs, jobs, shots_per, probe_pairs, phase_pairs = self._build_batch(programs, shots, float(self.clock()))
        raw = self.executor.run([j.compiled.circuit for j in jobs], shots_per)
        if len(raw) != len(jobs):
            raise RuntimeError("executor returned a different number of results")
        now = float(self.clock())
        freqs = [j.compiled.untwirl(normalize_counts(r)) for j, r in zip(jobs, raw)]
        probe_runs = self._learn_calibration(jobs, freqs, shots_per, now)
        self._after_calibration(now)
        for i, j in probe_pairs:
            cost = shots_per * len(cfg.probe_depths)
            self._consume(i, cost, now)
            self._consume(j, cost, now)
        for i, j in phase_pairs:
            cost = shots_per * 4 * len(cfg.phase_angles)
            self._consume(i, cost, now)
            self._consume(j, cost, now)
            self._phase_last[pair(i, j)] = now
        parts: dict[tuple[int, int], list[dict[str, float]]] = {}
        for job, f in zip(jobs, freqs):
            if job.kind == "alg":
                parts.setdefault((job.program, job.assignment), []).append(f)
        merged = {k: (merge_frequencies([(p, shots_per) for p in v]), shots_per * len(v)) for k, v in parts.items()}
        tests = sum(len(p.measurements) * len(d) for p, d, s in zip(programs, designs, syms) if s == "z2")
        flagged: set[int] = set()
        if tests:
            threshold = bonferroni_z(cfg.symmetry_alpha, tests)
            for pi, (prog, design) in enumerate(zip(programs, designs)):
                if syms[pi] == "z2":
                    for ai, assignment in enumerate(design):
                        f, n = merged[(pi, ai)]
                        flagged |= z2_flags(prog, assignment, f, n, threshold)
        self.flagged_ions = flagged
        degraded = set(self.degraded())
        job_ids = tuple(getattr(self.executor, "last_job_ids", ()))
        gamma = cfg.angle_exponent
        results = []
        for pi, (prog, design) in enumerate(zip(programs, designs)):
            runs = []
            for ai, assignment in enumerate(design):
                f, n = merged[(pi, ai)]
                trusted = not (prog.used_resources(assignment) & degraded)
                damping = self._predicted_damping(prog.total_exposure(assignment, gamma), now)
                runs.append(AssignmentRun(assignment, f, n, trusted, 1.0 - math.exp(-damping)))
            results.append(
                BatchResult(
                    name=prog.name,
                    runs=runs,
                    recommended=self.designer.best(prog, self.model, self.decisions, self.ions, now=now),
                    probes=probe_runs,
                    flagged_ions=tuple(sorted(flagged)),
                    degraded=tuple(sorted(degraded, key=to_key)),
                    timestamp=now,
                    job_ids=job_ids,
                    tier2=self._tier2(prog, design, merged, pi, extra[pi], now),
                )
            )
        return results

    def probe(self, pairs: Iterable[tuple[int, int]], shots: int, bases: int | None = None) -> list[ProbeRun]:
        """Cycle-benchmark explicit ion pairs (``bases`` random basis pairs each, shots per circuit)."""
        allowed = set(self.ions)
        todo = [(min(i, j), max(i, j)) for i, j in pairs]
        if not todo:
            return []
        if any(i == j or i not in allowed or j not in allowed for i, j in todo):
            raise ValueError("probe pairs must be distinct available ions")
        k = self.config.probe_bases if bases is None else int(bases)
        offset = self._angle_offset(self.clock())
        jobs: list[_Job] = []
        for p in todo:
            jobs.extend(self._cb_jobs(p, self._random_bases(k), offset))
        if self.config.spam_probe:
            jobs.extend(self._spam_jobs(self._spam_ions(i for p in todo for i in p)))
        raw = self.executor.run([j.compiled.circuit for j in jobs], int(shots))
        now = float(self.clock())
        freqs = [j.compiled.untwirl(normalize_counts(r)) for j, r in zip(jobs, raw)]
        runs = self._learn_calibration(jobs, freqs, int(shots), now)
        self._after_calibration(now)
        for i, j in todo:
            cost = int(shots) * len(self.config.probe_depths) * min(k, len(BASIS_PAIRS))
            self._consume(i, cost, now)
            self._consume(j, cost, now)
        return runs

    def phase_probe(self, pairs: Iterable[tuple[int, int]], shots: int) -> list[ProbeRun]:
        """Measure the coherent ZZ over-rotation of explicit ion pairs at the configured reference angles."""
        allowed = set(self.ions)
        todo = [(min(i, j), max(i, j)) for i, j in pairs]
        if not todo:
            return []
        if any(i == j or i not in allowed or j not in allowed for i, j in todo):
            raise ValueError("probe pairs must be distinct available ions")
        jobs: list[_Job] = []
        for p in todo:
            jobs.extend(self._phase_jobs(p))
        raw = self.executor.run([j.compiled.circuit for j in jobs], int(shots))
        now = float(self.clock())
        freqs = [j.compiled.untwirl(normalize_counts(r)) for j, r in zip(jobs, raw)]
        runs = self._learn_calibration(jobs, freqs, int(shots), now)
        for i, j in todo:
            cost = int(shots) * 4 * len(self.config.phase_angles)
            self._consume(i, cost, now)
            self._consume(j, cost, now)
            self._phase_last[pair(i, j)] = now
        return runs

    def run_fixed(
        self,
        circuit: CircuitLike,
        assignment: Sequence[int],
        shots: int,
        twirls: int = 0,
        measure_twirl: bool = False,
    ) -> BatchResult:
        """Baseline: run on a fixed assignment without learning.

        ``twirls=0`` disables Pauli-frame twirling. With ``measure_twirl`` the shots are split over an even number of
        circuits with complementary readout flips, so every clbit is flipped in exactly half of them.
        """
        prog = self.prepare(circuit)
        assignment = tuple(int(x) for x in assignment)
        if not set(assignment) <= set(self.ions):
            raise ValueError("assignment uses unavailable ions")
        count = max(int(twirls), 1)
        if measure_twirl and count % 2:
            count += 1
        per = max(1, math.ceil(shots / count)) if count > 1 else int(shots)
        clbits = [c for _, c in prog.measurements]
        flip_sets = _flip_sets(clbits, count, self.rng) if measure_twirl else [()] * count
        mode = self.config.twirl if twirls > 0 else "none"
        offset = self._angle_offset(self.clock())
        comps = [
            compile_twirled(prog, assignment, self.rng, mode, flip_sets[t], offset, f"{prog.name}_fixed_t{t}")
            for t in range(count)
        ]
        raw = self.executor.run([c.circuit for c in comps], per)
        f = merge_frequencies([(c.untwirl(normalize_counts(r)), per) for c, r in zip(comps, raw)])
        run = AssignmentRun(assignment, f, per * count, True)
        return BatchResult(
            name=prog.name,
            runs=[run],
            recommended=assignment,
            timestamp=float(self.clock()),
            job_ids=tuple(getattr(self.executor, "last_job_ids", ())),
        )

    def report(self, now: float | None = None) -> list[dict[str, Any]]:
        """Calibrated per-resource health, sorted from worst to best.

        'error' is the dressed-cycle process infidelity (cycle benchmarking) for ion pairs and the symmetrised SPAM
        flip probability for ions; 'error_upper' uses the posterior mean plus two standard deviations.
        """
        now = float(self.clock()) if now is None else float(now)
        rows = []
        for r, (mean, sd) in self.model.estimates(now).items():
            d = self.decisions.get(r)
            ev = self.model.evidence(r)
            rows.append(
                {
                    "resource": to_key(r),
                    "kind": r[0],
                    "ions": ions_of(r),
                    "error": to_error(r, mean),
                    "error_upper": to_error(r, mean + 2.0 * sd),
                    "damping": mean,
                    "damping_sd": sd,
                    "evidence": len(ev),
                    "last_update": ev[-1].t if ev else None,
                    "outlier_probability": self.model.outlier_probability(r, now),
                    "status": (d.status if d else Status.UNDECIDED).value,
                    "log_e_degraded": d.log_e_degraded if d else 0.0,
                    "log_e_healthy": d.log_e_healthy if d else 0.0,
                }
            )
        return sorted(rows, key=lambda row: (-row["error"], row["resource"]))

    def diagnostics(self, now: float | None = None) -> dict[str, Any]:
        """Calibration diagnostics.

        'innovation_z2' is the mean squared standardised innovation of the calibration evidence per resource kind,
        with the number of innovations; values well above 1 mean the measurement variances are too small (raise
        ``model_sd``), values well below 1 mean they are conservative. 'population' lists the (weight, mean, sd)
        mixture components of each resource kind.
        """
        now = float(self.clock()) if now is None else float(now)
        return {
            "innovation_z2": self.model.diagnostics(now),
            "population": {k: self.model.components(k) for k in ("pair", "ion")},
        }

    def coherent_report(self, now: float | None = None) -> list[dict[str, Any]]:
        """Coherent ZZ over-rotation (turns per gate) at each configured reference angle, per probed pair."""
        now = float(self.clock()) if now is None else float(now)
        rows = []
        for r in self.coherent.resources:
            for angle in self.config.phase_angles:
                mean, sd = self.coherent.estimate(r, angle, now)
                rows.append(
                    {
                        "resource": to_key(r),
                        "ions": ions_of(r),
                        "angle": angle,
                        "over_rotation": mean,
                        "over_rotation_sd": sd,
                        "correction": self.coherent.correction(r, angle, now),
                    }
                )
        return rows

    def save(self, path: str | Path) -> None:
        """Persist configuration and learned state to JSON."""
        data = {
            "version": 2,
            "config": self.config.to_dict(),
            "model": self.model.to_dict(),
            "coherent": self.coherent.to_dict(),
            "tokens": {str(k): list(v) for k, v in self._tokens.items()},
            "phase_last": {to_key(r): t for r, t in self._phase_last.items()},
            "flagged_ions": sorted(self.flagged_ions),
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path, executor: Executor, clock: Callable[[], float] = time.time) -> "Canary":
        """Restore a Canary saved with save()."""
        data = json.loads(Path(path).read_text())
        if data.get("version") != 2:
            raise ValueError("state files saved before the calibrated-evidence redesign cannot be restored")
        config = CanaryConfig.from_dict(data["config"])
        model = HealthModel.from_dict(data["model"], config, clock)
        coherent = CoherentModel.from_dict(data.get("coherent", {}), config, clock)
        canary = cls(executor, config, model, clock, coherent)
        canary._tokens = {int(k): (float(v[0]), float(v[1])) for k, v in data.get("tokens", {}).items()}
        canary._phase_last = {from_key(k): float(t) for k, t in data.get("phase_last", {}).items()}
        canary.flagged_ions = set(int(i) for i in data.get("flagged_ions", []))
        return canary
