"""High-level runners: run the Canary probe on IBM hardware, or in silico.

Nothing in this module changes the Canary method. It only orchestrates the
engine (quantum_canary.inversion):

  run_ibm()        live priors from the backend (BackendProfile.from_ibm_backend),
                   the engine's probe circuits (inversion.build_probe_circuits),
                   block-interleaved execution (inversion.interleave_probe_blocks),
                   one IBM Sampler job, and the engine's inversion
                   (inversion.lindblad_inversion) per qubit.
  run_simulation() the per-qubit flow of research/ideal_parity.py
                   (regime="ideal") or of research/benchmark.py with its
                   default settings (regime="nisq"), on the simulator in
                   quantum_canary.simulation,
                   with the true parameters kept for comparison.

IonQ trapped-ion hardware is run by quantum_canary.ionq.
"""
from __future__ import annotations

import contextlib
import dataclasses
import io
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import inversion as inv
from . import simulation as sim
from .online import OnlineInversionResult

__all__ = [
    "DEFAULT_SHOTS", "DEFAULT_PROBE_BLOCKS", "QubitReport", "CanaryReport",
    "run_ibm", "run_simulation",
]

# Shots per circuit of each probe family: the values every research benchmark
# (ideal_parity.py, benchmark.py) uses.
DEFAULT_SHOTS: Dict[str, int] = {"t1": 300, "ramsey": 600, "gate": 500, "echo": 500}
# Block-interleaved execution: benchmark.py's default (probe_blocks = 10).
DEFAULT_PROBE_BLOCKS = 10

# research/benchmark.py constants and default configuration (regime="nisq").
_NISQ_T_CALIB_S = 0.0
_NISQ_T_MEASURE_S = 3600.0
_NISQ_SPAM_SPREAD = 0.5      # per-qubit readout-error heterogeneity (benchmark --spam-spread)
_SPAM_CLIP_MAX = 0.499
# Probe family whose shot chunks define each parameter's comparison truth in
# a time-resolved run (benchmark.TRUTH_FAMILIES).
_TRUTH_FAMILIES = {"T1_s": ("t1_",), "T2_s": ("ramsey_X_", "ramsey_Y_", "echo_"),
                   "delta_omega": ("ramsey_X_", "ramsey_Y_"),
                   "epsilon_sx": ("gate_rep_N",)}
# research/ideal_parity.py and benchmark.py seed offsets per architecture.
_ARCH_SEED_OFFSET = {"superconducting": 0, "trapped_ion": 100_000}


@dataclass
class QubitReport:
    """One qubit's Canary result.

    result:       the engine's InversionResult (T1, T2 (echo), T2* (Ramsey),
                  delta_omega, epsilon_sx, their sigmas and chi2/dof).
    profile:      the BackendProfile the probe was built from (priors,
                  prior_confidence, calibration_source).
    health:       "GREEN" | "YELLOW" | "RED" from
                  online.OnlineInversionResult.health_summary().
    truth:        simulation only: the true parameters the result is compared
                  with (same definitions as the research benchmarks).
    engine_log:   lines the engine printed while building the profile and
                  inverting (live-prior summary, chi2 warnings).
    """
    qubit: int
    result: inv.InversionResult
    profile: inv.BackendProfile
    health: str
    truth: Optional[Dict[str, float]] = None
    engine_log: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        r = dataclasses.asdict(self.result)
        const = self.profile.constants
        return {"qubit": self.qubit, "health": self.health, "truth": self.truth,
                "prior_confidence": self.profile.prior_confidence,
                "T1_prior_s": self.profile.T1_prior_s,
                "T2_prior_s": self.profile.T2_prior_s,
                "eps_prior": const.get("eps_typical"),
                "p0_given_1": const.get("p0_given_1"),
                "p1_given_0": const.get("p1_given_0"),
                "gate_time_ns": const.get("gate_time_ns"),
                "dt_ns": self.profile.dt_ns,
                "result": r, "engine_log": list(self.engine_log)}


@dataclass
class CanaryReport:
    """Result of one Canary run over one or more qubits."""
    mode: str                     # "hardware" | "simulation"
    backend: str
    architecture: str
    shots: Dict[str, int]
    probe_blocks: int
    circuits_per_qubit: int
    qubits: List[QubitReport]
    timestamp: str
    regime: Optional[str] = None  # simulation only: "ideal" | "nisq"
    job_id: Optional[str] = None  # hardware only
    groups: Optional[List[List[int]]] = None  # hardware only: qubits probed together
    # hardware only: per qubit, every shot block's raw counts as
    # {"circuit": k, "block": b, "counts": {...}}
    block_counts: Optional[Dict[int, List[Dict[str, Any]]]] = field(default=None, repr=False)

    @property
    def shots_per_qubit(self) -> int:
        if not self.qubits:
            return 0
        meta_counts = _probe_counts(self.qubits[0].profile)
        return sum(meta_counts[k] * self.shots[k] for k in meta_counts)

    def to_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "backend": self.backend,
                "architecture": self.architecture, "regime": self.regime,
                "job_id": self.job_id, "shots": dict(self.shots),
                "probe_blocks": self.probe_blocks,
                "circuits_per_qubit": self.circuits_per_qubit,
                "groups": self.groups,
                "timestamp": self.timestamp,
                "qubits": [q.to_dict() for q in self.qubits],
                "block_counts": ({str(q): entries for q, entries in self.block_counts.items()}
                                 if self.block_counts else None)}

    def render(self, color: Optional[bool] = None) -> str:
        from .display import render_report
        return render_report(self, color=color)

    def print(self, color: Optional[bool] = None, file=None) -> None:
        """Print render() to `file` (default sys.stdout). Characters the
        stream's encoding cannot represent (e.g. on a legacy Windows code
        page) are replaced instead of raising UnicodeEncodeError."""
        import sys
        stream = sys.stdout if file is None else file
        text = self.render(color=color) + "\n"
        try:
            stream.write(text)
        except UnicodeEncodeError:
            enc = getattr(stream, "encoding", None) or "ascii"
            stream.write(text.encode(enc, errors="replace").decode(enc))
        stream.flush()


def _probe_counts(profile: inv.BackendProfile) -> Dict[str, int]:
    _, meta = inv.build_probe_circuits(profile)
    return {"t1": meta["n_t1"], "ramsey": meta["n_ramsey"],
            "gate": meta["n_gate"], "echo": meta["n_echo"]}


def _shots_list(meta: Dict[str, Any], shots: Dict[str, int]) -> List[int]:
    return ([shots["t1"]] * meta["n_t1"] + [shots["ramsey"]] * meta["n_ramsey"]
            + [shots["gate"]] * meta["n_gate"] + [shots["echo"]] * meta["n_echo"])


def _check_shots(shots: Optional[Dict[str, int]]) -> Dict[str, int]:
    out = dict(DEFAULT_SHOTS)
    if shots:
        unknown = set(shots) - set(DEFAULT_SHOTS)
        if unknown:
            raise ValueError(f"unknown shot keys {sorted(unknown)}; expected {sorted(DEFAULT_SHOTS)}")
        out.update({k: int(v) for k, v in shots.items()})
    if any(v < 1 for v in out.values()):
        raise ValueError(f"every shot count must be >= 1, got {out}")
    return out


def _health(qubit: int, result: inv.InversionResult) -> str:
    snapshot = OnlineInversionResult(
        round_index=0, timestamp=result.timestamp, qubit_results={qubit: result},
        algo_indices=[], canary_indices=[])
    return snapshot.health_summary()[qubit]


def _captured(fn, *args, **kwargs):
    """Call fn, returning (value, printed lines): the engine reports live
    priors and chi2 warnings with print(); the runner shows them in its
    report instead of interleaving them with its own output."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        value = fn(*args, **kwargs)
    return value, [ln for ln in buf.getvalue().splitlines() if ln.strip()]


# --------------------------------------------------------------------------
# Hardware (IBM, superconducting)
# --------------------------------------------------------------------------

def _counts_from_pub(pub_result, circuit, shots: int,
                     bit: Optional[int] = None) -> Dict[str, int]:
    """Counts of one probe qubit from a sampler PUB result: the circuit's
    only clbit, or clbit `bit` of a parallel probe circuit. Raises if the
    register is missing or the counts do not add up to the shots requested:
    the inversion must never see fabricated counts."""
    creg = circuit.cregs[0].name
    data = getattr(pub_result.data, creg, None)
    if data is None:
        raise RuntimeError(f"classical register {creg!r} missing from the sampler result")
    if bit is not None and data.num_bits > 1:
        data = data.slice_bits([int(bit)])
    raw = data.get_counts()
    counts = {str(k): int(v) for k, v in raw.items()}
    if not set(counts) <= {"0", "1"} or sum(counts.values()) != shots:
        raise RuntimeError(f"unexpected counts {raw!r} for {shots} shots")
    return counts


def probe_groups(backend, qubits: Sequence[int],
                 shot_plans: Optional[Dict[int, List[int]]] = None) -> List[List[int]]:
    """Split `qubits` into groups probed in parallel: no two qubits in a group
    are neighbours on the backend's coupling map, and (if shot_plans is given)
    all share the same per-circuit shot list. Qubits with the same shot list
    are two-coloured breadth-first along the coupling map, which gives at most
    two groups on bipartite layouts such as IBM's heavy-hex; a non-bipartite
    set falls back to greedy first-fit. Qubits keep their given order within
    a group. Without a coupling map every qubit gets its own group."""
    qubits = [int(q) for q in qubits]
    cmap = getattr(backend, "coupling_map", None)
    if cmap is None:
        return [[q] for q in qubits]
    members = set(qubits)
    nbr: Dict[int, set] = {q: set() for q in qubits}
    for a, b in cmap.get_edges():
        if a in members and b in members:
            nbr[a].add(b)
            nbr[b].add(a)
    parts: List[List[int]] = []
    for q in qubits:
        for part in parts:
            if shot_plans is None or shot_plans[part[0]] == shot_plans[q]:
                part.append(q)
                break
        else:
            parts.append([q])
    groups: List[List[int]] = []
    for part in parts:
        groups.extend(_two_colour(part, nbr) or _first_fit(part, nbr))
    return groups


def _two_colour(qubits: List[int], nbr: Dict[int, set]) -> Optional[List[List[int]]]:
    """The two colour classes of `qubits` along `nbr` (breadth-first, the first
    qubit of each connected piece coloured 0), or None if they are not bipartite."""
    members = set(qubits)
    colour: Dict[int, int] = {}
    for start in qubits:
        if start in colour:
            continue
        colour[start] = 0
        queue = [start]
        while queue:
            q = queue.pop(0)
            for n in nbr[q] & members:
                if n not in colour:
                    colour[n] = 1 - colour[q]
                    queue.append(n)
                elif colour[n] == colour[q]:
                    return None
    classes = [[q for q in qubits if colour[q] == c] for c in (0, 1)]
    return [c for c in classes if c]


def _first_fit(qubits: List[int], nbr: Dict[int, set]) -> List[List[int]]:
    """Greedy first-fit groups with no two neighbours, in the order given."""
    groups: List[List[int]] = []
    for q in qubits:
        for g in groups:
            if not nbr[q] & set(g):
                g.append(q)
                break
        else:
            groups.append([q])
    return groups


def _parallel_circuit(circuits: Sequence[Any], name: str):
    """One circuit holding each one-qubit probe circuit on its own qubit and
    clbit (qubit j of the result is circuits[j])."""
    from qiskit import QuantumCircuit
    out = QuantumCircuit(len(circuits), len(circuits), name=name)
    for j, qc in enumerate(circuits):
        out.compose(qc, qubits=[j], clbits=[j], inplace=True)
    return out


REPEAT_PARAMETER = "canary_repeat"


def uniform_pubs(circuits: Sequence[Any], shots_list: Sequence[int]) -> List[Any]:
    """Sampler PUBs that all ask for the same number of shots, as
    qiskit-ibm-runtime requires (mixed shots are deprecated in SamplerV2 and
    rejected by its client-side successor, ibm_sampler()).

    With s the gcd of shots_list, circuit i gets an inert rz(theta) before
    its first instruction and runs shots_list[i] // s bindings of theta = 0
    with s shots each: rz(0) is the identity and costs nothing on IBM
    hardware (a virtual frame change), so the executions are those of
    shots_list[i] shots of the original circuit. Counts summed over the
    bindings (BitArray.get_counts() unions them) are those counts; see
    flatten_result() for a result in the original shape. If every entry
    already has the same shots, the circuits are returned unchanged."""
    from qiskit.circuit import Parameter
    shots_list = [int(s) for s in shots_list]
    if len(set(shots_list)) <= 1:
        return [(qc, None, s) for qc, s in zip(circuits, shots_list)]
    s = 0
    for sh in shots_list:
        s = math.gcd(s, sh)
    theta = Parameter(REPEAT_PARAMETER)
    cache: Dict[int, Any] = {}
    pubs = []
    for qc, sh in zip(circuits, shots_list):
        if id(qc) not in cache:
            first = next(inst for inst in qc.data if inst.qubits)
            out = qc.copy_empty_like()
            out.rz(theta, first.qubits[0])
            out.compose(qc, inplace=True)
            cache[id(qc)] = out
        pubs.append((cache[id(qc)], np.zeros((sh // s, 1)), s))
    return pubs


def flatten_result(pub_result):
    """A uniform_pubs() result in the shape of an ordinary one-binding PUB:
    every BitArray's bindings concatenated into one array of all its shots."""
    from qiskit.primitives import BitArray, DataBin, SamplerPubResult
    data = pub_result.data
    if not data.shape:
        return pub_result
    fields = {name: BitArray.concatenate_shots([value[i] for i in range(value.shape[0])])
              for name, value in data.items() if isinstance(value, BitArray)}
    return SamplerPubResult(DataBin(**fields), metadata=pub_result.metadata)


def _is_local(backend) -> bool:
    """True for a qiskit_ibm_runtime fake backend or an AerSimulator."""
    try:
        from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2
        if isinstance(backend, FakeBackendV2):
            return True
    except ImportError:
        pass
    try:
        from qiskit_aer import AerSimulator
        return isinstance(backend, AerSimulator)
    except ImportError:
        return False


def sample(backend, circuits: Sequence[Any], shots_list: Sequence[int], sampler=None):
    """Run each circuit for its shots and return (results, job_id); results[i]
    is circuit i's SamplerPubResult, in order, and results.metadata holds
    the job's metadata (execution spans on IBM hardware).

    On real backends, or with a sampler given, this is one job of
    uniform_pubs(). On a fake backend or AerSimulator with no sampler given,
    the circuits are simulated locally in one Aer call per set of qubits
    they use, as plain circuits with their own shots: Aer's noisy
    simulation of a fake backend lets delay noise leak between circuits on
    different qubits within one call, and its handling of many parameter
    bindings is not reliable, so one batched call would not be simulated
    faithfully."""
    if sampler is None and _is_local(backend):
        from qiskit.primitives import PrimitiveResult
        from qiskit_aer.primitives import SamplerV2 as AerSampler
        local = AerSampler.from_backend(backend)
        by_set: Dict[frozenset, List[int]] = {}
        for i, qc in enumerate(circuits):
            used = frozenset(qc.find_bit(q).index for inst in qc.data for q in inst.qubits)
            by_set.setdefault(used, []).append(i)
        results: List[Any] = [None] * len(circuits)
        for idx in by_set.values():
            out = local.run([(circuits[i], None, int(shots_list[i])) for i in idx]).result()
            for i, r in zip(idx, out):
                results[i] = r
        return PrimitiveResult(results, metadata={}), None
    if sampler is None:
        sampler = ibm_sampler(backend)
    job = sampler.run(uniform_pubs(circuits, shots_list))
    return job.result(), (job.job_id() if hasattr(job, "job_id") else None)


def ibm_sampler(backend):
    """IBM's client-side Sampler (qiskit_ibm_runtime.executor_sampler), which
    replaces the deprecated SamplerV2; SamplerV2 on qiskit-ibm-runtime versions
    without it. Gate and measurement twirling and dynamical decoupling are off
    by default in both, as Canary's probe circuits require."""
    try:
        from qiskit_ibm_runtime.executor_sampler import Sampler
    except ImportError:
        from qiskit_ibm_runtime import SamplerV2 as Sampler
    return Sampler(mode=backend)


def pub_executions(pub) -> int:
    """Circuit executions a PUB asks for: bindings times shots."""
    values = pub[1] if len(pub) > 1 else None
    return int(pub[2]) * (int(np.asarray(values).shape[0]) if values is not None else 1)


def _block_numbers(shots_list: Sequence[int], n_blocks: int) -> List[int]:
    """Block number of every entry of inversion.interleave_probe_blocks()'s
    plan, in the same order."""
    out = []
    for b in range(int(n_blocks)):
        for sh in shots_list:
            base, rem = divmod(int(sh), int(n_blocks))
            if base + (1 if b < rem else 0) > 0:
                out.append(b)
    return out


def run_ibm(backend, qubits: Sequence[int], shots: Optional[Dict[str, int]] = None,
            probe_blocks: int = DEFAULT_PROBE_BLOCKS, sampler=None,
            parallel: bool = True) -> CanaryReport:
    """Run the Canary probe on physical `qubits` of an IBM (superconducting)
    backend and invert each qubit.

    backend:      a Qiskit backend, e.g. QiskitRuntimeService().backend("ibm_fez")
                  (or a qiskit_ibm_runtime fake backend for a local dry run).
    qubits:       physical qubit indices.
    shots:        per-circuit shots per probe family (default DEFAULT_SHOTS).
    probe_blocks: block-interleaved submission (default 10, as in the
                  benchmarks); 1 submits each circuit's shots consecutively.
                  Qiskit Runtime does not guarantee that the circuits of a
                  job execute in submission order, so on hardware this sets
                  the submitted order only.
    sampler:      optional sampler (SamplerV2 interface); default
                  ibm_sampler(backend).
    parallel:     probe several qubits in the same circuits (default). Qubits
                  are split by probe_groups() so that no two neighbours on
                  the coupling map are probed together; each group runs the
                  17 probe circuits once, so the shots executed scale with
                  the number of groups rather than qubits. False probes one
                  qubit at a time, in separate circuits.

    Priors come from the backend's own calibration
    (BackendProfile.from_ibm_backend, the profile get_live_profile builds for
    an IBM backend): prior_confidence is "live" when both T1 and T2 were read
    from it, otherwise "arch_default". All circuits are submitted as one job.
    Each qubit's counts are its own clbit only, and the engine inverts every
    qubit separately, exactly as for a one-qubit run.
    """
    from qiskit import transpile

    qubits = [int(q) for q in qubits]
    if not qubits:
        raise ValueError("qubits must not be empty")
    if len(set(qubits)) != len(qubits):
        raise ValueError(f"duplicate qubits in {qubits}")
    n_backend = getattr(backend, "num_qubits", None)
    if n_backend is not None and any(not 0 <= q < n_backend for q in qubits):
        raise ValueError(f"qubits {qubits} outside 0..{n_backend - 1} of {backend.name}")
    shots = _check_shots(shots)

    profiles, logs, metas, probes, plans = {}, {}, {}, {}, {}
    for q in qubits:
        # The call get_live_profile() makes for an IBM backend. Calling it
        # directly also accepts any Qiskit BackendV2 with properties() (e.g.
        # qiskit_ibm_runtime fake backends): get_live_profile's generic check
        # imports qiskit.providers.BackendV1, which Qiskit 2.x removed, so it
        # recognises only qiskit_ibm_runtime.IBMBackend there.
        profile, log = _captured(inv.BackendProfile.from_ibm_backend, backend, qubit=q)
        if profile.architecture != "superconducting":
            raise ValueError("run_ibm supports superconducting backends only "
                             f"(got architecture {profile.architecture!r})")
        qcs, meta = inv.build_probe_circuits(profile)
        profiles[q], logs[q], metas[q], probes[q] = profile, log, meta, qcs
        plans[q] = _shots_list(meta, shots)

    # Circuit k of every qubit in a group shares one circuit (same family and
    # shots for all members); units[i] is (group, k) of circuit i.
    groups = probe_groups(backend, qubits, plans) if parallel else [[q] for q in qubits]
    units, unit_circuits, unit_shots = [], [], []
    for g in groups:
        for k, sh in enumerate(plans[g[0]]):
            qc = (probes[g[0]][k] if len(g) == 1 else
                  _parallel_circuit([probes[q][k] for q in g], f"{probes[g[0]][k].name}_x{len(g)}"))
            unit_circuits.append(transpile(qc, backend=backend, initial_layout=g,
                                           optimization_level=0))
            unit_shots.append(sh)
            units.append((g, k))
    block_circuits, block_shots, block_index = inv.interleave_probe_blocks(
        unit_circuits, unit_shots, int(probe_blocks))
    block_no = _block_numbers(unit_shots, int(probe_blocks))

    pub_results, job_id = sample(backend, block_circuits, block_shots, sampler)

    blocks: Dict[int, List[Dict[str, Any]]] = {q: [] for q in qubits}
    for i, (u, b) in enumerate(zip(block_index, block_no)):
        g, k = units[u]
        for pos, q in enumerate(g):
            blocks[q].append({"circuit": k, "block": b, "counts": _counts_from_pub(
                pub_results[i], block_circuits[i], block_shots[i], bit=pos)})

    timestamp = datetime.now(timezone.utc).isoformat()
    reports = []
    for q in qubits:
        counts = inv.merge_block_counts([e["counts"] for e in blocks[q]],
                                        [e["circuit"] for e in blocks[q]], len(plans[q]))
        result, log = _captured(
            inv.lindblad_inversion, counts, metas[q], profiles[q],
            shots_t1=shots["t1"], shots_ramsey=shots["ramsey"],
            shots_gate=shots["gate"], shots_echo=shots["echo"],
            qubit_id=q, timestamp=timestamp)
        reports.append(QubitReport(qubit=q, result=result, profile=profiles[q],
                                   health=_health(q, result), engine_log=logs[q] + log))

    return CanaryReport(mode="hardware", backend=str(getattr(backend, "name", backend)),
                        architecture="superconducting", shots=shots,
                        probe_blocks=int(probe_blocks),
                        circuits_per_qubit=len(plans[qubits[0]]),
                        qubits=reports, timestamp=timestamp, job_id=job_id,
                        groups=[list(g) for g in groups], block_counts=blocks)


# --------------------------------------------------------------------------
# Simulation (superconducting and trapped ion)
# --------------------------------------------------------------------------

def _time_resolved_truth(truth, sim_result) -> Dict[str, float]:
    """research/benchmark._time_resolved_truth: per parameter, the
    shot-weighted mean over the chunks of the probe family that measures it."""
    out: Dict[str, float] = {}
    for param, prefixes in _TRUTH_FAMILIES.items():
        vals, weights = [], []
        for name, recs in zip(sim_result.circuit_names, sim_result.chunk_records):
            if not name.startswith(prefixes):
                continue
            for rec in recs:
                if param == "epsilon_sx":
                    vals.append(average_gate_infidelity(truth, rec["t_mid_s"]))
                else:
                    key = {"T1_s": "T1_s", "T2_s": "T2_s",
                           "delta_omega": "delta_omega_rad_s"}[param]
                    vals.append(float(rec["truth"][0][key]))
                weights.append(float(rec["shots"]))
        v, w = np.array(vals, dtype=float), np.array(weights, dtype=float)
        out[param] = float(v[0] + np.sum(w * (v - v[0])) / np.sum(w))
    return out


def average_gate_infidelity(truth: Any, t_s: float) -> float:
    """1 - F_avg of qubit 0's sx as the simulator plays it at t_s: relaxation and
    depolarizing after the pulse, the amplitude over-rotation, and the detuning
    during the pulse averaged over the quasi-static offset."""
    from qiskit.circuit.library import SXGate
    from qiskit.quantum_info import Operator, SuperOp, average_gate_fidelity
    err, f = sim._gate_noise_channel(truth, 0, t_s)
    g = Operator(SXGate())
    core = SuperOp(g).compose(SuperOp(err))
    if f > 0.0:
        core = core.compose(SuperOp(g.power(f)))
    during = bool(truth.architecture_specific.get("detuning_during_gates", False))
    dw = sim.evaluate_truth_at(truth, 0, t_s)["delta_omega_rad_s"] if during else 0.0
    qs = truth.architecture_specific.get("quasi_static_dw_std_rad_s")
    std = float(np.asarray(qs, dtype=float)[0]) if (during and qs is not None) else 0.0
    nodes, weights = np.polynomial.hermite_e.hermegauss(9)
    t_g = float(truth.gate_duration_ns) * 1e-9
    channel = None
    for x, w in zip(nodes, weights / weights.sum()):
        a = (dw + std * x) * t_g / 2.0
        rz = SuperOp(Operator(np.diag([np.exp(-0.5j * a), np.exp(0.5j * a)])))
        term = rz.compose(core).compose(rz) * w
        channel = term if channel is None else channel + term
    return 1.0 - float(average_gate_fidelity(channel, target=g))


def _simulate_one(architecture: str, regime: str, seed: int, qubit: int,
                  shots: Dict[str, int], probe_blocks: int):
    """One simulated qubit, exactly as one instance of research/ideal_parity.py
    (regime="ideal") or research/benchmark.py with default settings
    (regime="nisq"). Returns (InversionResult, profile, truth dict, engine log)."""
    if regime == "ideal":
        truth = sim.generate_device(architecture, n_qubits=1, seed=seed, regime="ideal",
                                    spam_spread_frac=0.0, perturbation_frac=None,
                                    dw_population="estimator_window")
        prior = sim.generate_calibration_prior(truth, qubit=0, t0_s=0.0,
                                               perturbation_frac=None)
        t_s = 0.0
    else:
        # Every simulator mechanism, as research/benchmark.py's defaults apply
        # them: 1/f detuning drift and T1 telegraph switching exist for
        # superconducting qubits only, the motional-heating gate error for
        # trapped ions only; crosstalk needs two qubits and cannot act here.
        sc = architecture == "superconducting"
        truth = sim.generate_device(architecture, n_qubits=1, seed=seed, regime="nisq",
                                    overrides=None, crosstalk_scale=1.0,
                                    detuning_1_over_f=sc,
                                    enable_heating_gate_error=not sc,
                                    spam_spread_frac=_NISQ_SPAM_SPREAD,
                                    perturbation_frac=None, dw_population="literature",
                                    t1_telegraph=sc, quasi_static_dephasing=True,
                                    detuning_during_gates=True)
        prior = sim.generate_calibration_prior(
            truth, qubit=0, t0_s=_NISQ_T_CALIB_S, perturbation_frac=None,
            spam_perturbation_frac=sim.resolve_calibration_perturbation(architecture, "nisq"))
        t_s = _NISQ_T_MEASURE_S

    p0_given_1 = float(np.clip(prior.spam_p0_given_1_prior, 0.0, _SPAM_CLIP_MAX))
    p1_given_0 = float(np.clip(prior.spam_p1_given_0_prior, 0.0, _SPAM_CLIP_MAX))
    constants = dict(inv.ARCH_DEFAULTS[architecture])
    constants.update(T1_s=prior.T1_prior_s, T2_s=prior.T2_prior_s,
                     eps_typical=prior.epsilon_prior,
                     p0_given_1=p0_given_1, p1_given_0=p1_given_0)
    profile = inv.BackendProfile(
        architecture=architecture, T1_prior_s=prior.T1_prior_s,
        T2_prior_s=prior.T2_prior_s, dt_ns=inv.ARCH_DEFAULTS[architecture]["dt_ns"],
        backend_name=f"simulated::{architecture}::{regime}", custom_arch=constants,
        prior_confidence="prior_informed")

    circuits, meta = inv.build_probe_circuits(profile)
    shots_list = _shots_list(meta, shots)
    timestamp = datetime.now(timezone.utc).isoformat()

    if regime == "ideal":
        sim_result = sim.simulate_probe(circuits, truth, shots=shots_list, t_s=0.0,
                                        dt_ns=profile.dt_ns, seed=int(seed) * 1000)
        counts_list = sim_result.counts_list
        tv = sim.evaluate_truth_at(truth, 0, 0.0)
        true_values = {"T1_s": float(tv["T1_s"]), "T2_s": float(tv["T2_s"]),
                       "delta_omega": float(tv["delta_omega_rad_s"]),
                       "epsilon_sx": sim.effective_gate_epsilon(truth, 0, 0.0)}
    else:
        block_circuits, block_shots, block_index = inv.interleave_probe_blocks(
            circuits, shots_list, int(probe_blocks))
        if len(block_circuits) > 1000:
            raise RuntimeError(f"{len(block_circuits)} block entries would reuse Aer seeds")
        sim_result = sim.simulate_probe(block_circuits, truth, shots=block_shots,
                                        t_s=t_s, dt_ns=profile.dt_ns,
                                        seed=int(seed) * 1000, time_resolved=True)
        counts_list = inv.merge_block_counts(sim_result.counts_list, block_index,
                                             len(circuits))
        true_values = _time_resolved_truth(truth, sim_result)

    result, log = _captured(
        inv.lindblad_inversion, counts_list, meta, profile,
        shots_t1=shots["t1"], shots_ramsey=shots["ramsey"],
        shots_gate=shots["gate"], shots_echo=shots["echo"],
        qubit_id=qubit, timestamp=timestamp)
    true_values.update(T1_prior_s=float(prior.T1_prior_s), T2_prior_s=float(prior.T2_prior_s),
                       epsilon_prior=float(prior.epsilon_prior), seed=int(seed))
    return result, profile, true_values, log


def run_simulation(architecture: str = "superconducting", n_qubits: int = 3,
                   regime: str = "nisq", seed: Optional[int] = None,
                   shots: Optional[Dict[str, int]] = None,
                   probe_blocks: int = DEFAULT_PROBE_BLOCKS) -> CanaryReport:
    """Run Canary on simulated qubits and keep the true parameters.

    Each qubit is an independent simulated device, generated and probed
    exactly as one instance of the research benchmarks:
      regime="nisq"  (default) research/benchmark.py with its default
                     settings: every simulator mechanism (crosstalk excepted,
                     as each qubit is a separate one-qubit device). T1 drift and
                     two-state T1 switching, 1/f frequency drift and
                     quasi-static dephasing (superconducting); OU frequency
                     drift and motional-heating gate error (trapped ion);
                     literature detuning population, detuning during gates,
                     coherent gate error, per-qubit readout heterogeneity,
                     stale calibration and SPAM priors (1 h old), and
                     time-resolved block-interleaved execution.
      regime="ideal" research/ideal_parity.py: correctly specified,
                     stationary Markovian noise.
    The true epsilon_sx is the engine's own epsilon (sim.effective_gate_epsilon)
    for regime="ideal" and the sx average gate infidelity
    (average_gate_infidelity) for regime="nisq", as in the two benchmarks.
    Qubit i uses seed (seed + i); seed defaults to the benchmarks' offset for
    the architecture (0 superconducting, 100000 trapped ion), so with default
    settings qubit i reproduces benchmark instance i.
    probe_blocks applies to regime="nisq" only (the ideal benchmark runs each
    circuit's shots consecutively).
    """
    if architecture not in ("superconducting", "trapped_ion"):
        raise ValueError(f"architecture must be 'superconducting' or 'trapped_ion', "
                         f"got {architecture!r}")
    if regime not in ("ideal", "nisq"):
        raise ValueError(f"regime must be 'ideal' or 'nisq', got {regime!r}")
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
    shots = _check_shots(shots)
    base = _ARCH_SEED_OFFSET[architecture] if seed is None else int(seed)

    reports = []
    for i in range(int(n_qubits)):
        result, profile, truth, log = _simulate_one(architecture, regime, base + i, i,
                                                    shots, probe_blocks)
        reports.append(QubitReport(qubit=i, result=result, profile=profile,
                                   health=_health(i, result), truth=truth,
                                   engine_log=log))
    circuits_per_qubit = len(inv.build_probe_circuits(reports[0].profile)[0])
    return CanaryReport(mode="simulation", backend="simulator",
                        architecture=architecture, shots=shots,
                        probe_blocks=int(probe_blocks) if regime == "nisq" else 1,
                        circuits_per_qubit=circuits_per_qubit, qubits=reports,
                        timestamp=datetime.now(timezone.utc).isoformat(), regime=regime)
