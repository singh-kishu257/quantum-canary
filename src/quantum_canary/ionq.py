"""IonQ trapped-ion hardware: Canary's gate-error probe on chosen ions.

On trapped ions T1 and T2 are seconds long, far beyond any circuit, so only
the gate family is run: randomized benchmarking, the engine's random Clifford
sequences in native GPI2/GPI pulses (inversion._build_rb_circuit) at the three
lengths of BackendProfile.rb_lengths, each ion with its own sequences. Each
ion's epsilon_sx (error per pulse) comes from the engine's own gate fit
(inversion._invert_rb), in which readout cancels.

    report = run_ionq(backend, ions=[3, 6, 14], eps_prior=2e-4, max_cost=1500)

Circuits are sent as IonQ native gates with optimization off and IonQ's
debiasing off (debiasing averages permuted copies of a circuit, which mixes
up the ions). By default the ions share circuits (batches of at most max_ops
gates), each running its own independent random sequences: simultaneous
randomized benchmarking, so epsilon_sx is the error per pulse while all the
ions are driven, crosstalk included. IonQ bills every circuit of a job at
least its minimum charge, so this is also far cheaper than probing ions one
at a time. parallel=False probes each ion in its own circuits (isolated
error; on Forte-1 in August 2026 the earlier gate-pair probe was disturbed
by crosstalk when ions shared circuits). probe_blocks=N (default 1) splits every
circuit's shots into N blocks, one job per block in order, so all sequence
lengths sample the same stretch of time.

Quantum Canary does not price runs. ionq_estimate() asks IonQ for its own
estimate of a job (cost and execution time, in IonQ's units), and max_cost
is checked against that estimate only; what you are actually charged is on
your IonQ account.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import chi2 as _chi2

from . import inversion as inv
from . import online, runner

__all__ = ["IonResult", "IonQReport", "ionq_circuits", "ionq_estimate", "ionq_priors", "run_ionq",
           "ion_health", "to_canary_report"]

DEFAULT_SHOTS = runner.DEFAULT_SHOTS["gate"]
DEFAULT_MAX_OPS = 24_000
# IonQ rejects a job larger than 10 MB, about 46 bytes per native gate: 150,000 gates is about 6.9 MB.
DEFAULT_MAX_JOB_GATES = 150_000
POOR_FIT_P = 0.01


@dataclass
class IonResult:
    """One ion: epsilon_sx with its 1-sigma error, the fit's chi2/dof, and
    the data it came from."""
    ion: int
    epsilon_sx: float
    epsilon_sx_sigma: float
    chi2_dof: float
    chi2_p: float
    rb_lengths: List[int]
    p0: List[float]
    shots: List[int]

    @property
    def poor_fit(self) -> bool:
        return bool(np.isfinite(self.chi2_p) and self.chi2_p < POOR_FIT_P)

    def to_dict(self) -> Dict[str, Any]:
        return {"ion": self.ion, "epsilon_sx": self.epsilon_sx,
                "epsilon_sx_sigma": self.epsilon_sx_sigma, "chi2_dof": self.chi2_dof,
                "chi2_p": self.chi2_p, "poor_fit": self.poor_fit,
                "rb_lengths": list(self.rb_lengths), "p0": list(self.p0),
                "shots": list(self.shots)}


@dataclass
class IonQReport:
    """Result of run_ionq()."""
    backend: str
    ions: List[int]
    shots: int
    rb_lengths: List[int]
    eps_prior: float
    spam: Tuple[float, float]
    batches: List[List[int]]
    job_ids: List[str]
    ionq_estimates: List[Optional[Dict[str, Any]]]
    results: List[IonResult]
    raw: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    rb_seed: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    probe_blocks: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {"backend": self.backend, "ions": list(self.ions), "shots": self.shots,
                "probe_blocks": self.probe_blocks,
                "rb_lengths": list(self.rb_lengths), "rb_seed": self.rb_seed,
                "eps_prior": self.eps_prior,
                "p0_given_1": self.spam[0], "p1_given_0": self.spam[1],
                "batches": [list(b) for b in self.batches], "job_ids": list(self.job_ids),
                "ionq_estimates": list(self.ionq_estimates), "timestamp": self.timestamp,
                "results": [r.to_dict() for r in self.results], "raw": self.raw}

    def render(self) -> str:
        lines = [f"IonQ {self.backend}: {len(self.ions)} ions, RB lengths {self.rb_lengths} x "
                 f"{list(inv._RB_SEQUENCES)} sequences, {self.shots} shots"
                 + (f" in {self.probe_blocks} blocks" if self.probe_blocks > 1 else "") + ", "
                 f"jobs {', '.join(str(j) for j in self.job_ids)}",
                 f"  {'ion':>4}  {'epsilon_sx':>12}  {'sigma':>9}  {'chi2/dof':>8}"]
        for r in self.results:
            lines.append(f"  {r.ion:>4}  {r.epsilon_sx:>12.3e}  {r.epsilon_sx_sigma:>9.1e}  "
                         f"{r.chi2_dof:>8.2f}" + ("  poor fit" if r.poor_fit else ""))
        eps = [r.epsilon_sx for r in self.results if np.isfinite(r.epsilon_sx)]
        if eps:
            lines.append(f"  median {np.median(eps):.3e}, range {min(eps):.3e} - {max(eps):.3e}")
        return "\n".join(lines)

    def print(self) -> None:
        print(self.render())


def _arch(eps_prior: Optional[float], spam: Optional[Tuple[float, float]]) -> Dict[str, Any]:
    arch = dict(inv.ARCH_DEFAULTS["trapped_ion"])
    if eps_prior is not None:
        arch["eps_typical"] = float(eps_prior)
    if spam is not None:
        arch["p0_given_1"], arch["p1_given_0"] = float(spam[0]), float(spam[1])
    return arch


def _profile(arch: Dict[str, Any]) -> inv.BackendProfile:
    return inv.BackendProfile(architecture="trapped_ion", T1_prior_s=arch["T1_s"],
                              T2_prior_s=arch["T2_s"], dt_ns=None, custom_arch=arch,
                              prior_confidence="prior_informed")


def ionq_circuits(ions: Sequence[int], rb_lengths: Sequence[int],
                  max_ops: int = DEFAULT_MAX_OPS, parallel: bool = True,
                  seed: Optional[int] = None) -> Tuple[List[List[int]], List[Any]]:
    """Batches of ions and, for each batch, the gate probe's circuits
    (inversion._RB_SEQUENCES per length of rb_lengths, in the engine's order):
    every ion of the batch runs its own random sequences on its own qubit and
    is measured into its own clbit (clbit j is batch[j]). The sequences are
    drawn in ion order from one generator seeded with `seed`. Without parallel
    every batch is a single ion."""
    from qiskit import QuantumCircuit
    ions = [int(q) for q in ions]
    batches = (inv.split_qubits_for_op_limit(ions, max(rb_lengths) + 1, max_ops)
               if parallel else [[q] for q in ions])
    rng = np.random.default_rng(seed)
    probes = {q: [inv._build_rb_circuit(m, j % 2, "trapped_ion", rng, f"rb_{s}_{j}_m{m}")
                  for s, (m, k) in enumerate(zip(rb_lengths, inv._RB_SEQUENCES)) for j in range(k)]
              for q in ions}
    circuits = []
    for batch in batches:
        for i, probe in enumerate(probes[batch[0]]):
            qc = QuantumCircuit(max(batch) + 1, len(batch), name=f"canary_{probe.name}_ions{'-'.join(map(str, batch))}")
            for j, q in enumerate(batch):
                qc.compose(probes[q][i], qubits=[q], clbits=[j], inplace=True)
            circuits.append(qc)
    return batches, circuits


def ionq_estimate(backend, circuits: Sequence[Any], shots: int) -> Optional[Dict[str, Any]]:
    """IonQ's own estimate for one job of `circuits` with `shots` each, from
    its /jobs/estimate endpoint (qiskit-ionq's client.estimate_job), asked once
    per circuit: IonQ bills each circuit of a job as its own child job, at
    least the rate card's job_cost_minimum, so the cost is the sum over
    circuits of max(IonQ's estimate, that minimum). Returns a dict with that
    cost, its unit, the per-circuit costs, the summed execution time, the
    queue time and the raw responses, or None if the backend gives no
    estimate (simulators, mock backends, API errors). The cost is read from
    estimated_cost or, in IonQ's current API, estimated_total_cost; times are
    in IonQ's units (milliseconds in October 2026)."""
    client = getattr(backend, "client", None)
    if client is None or not hasattr(client, "estimate_job"):
        return None
    costs, raws, unit, execution, queue = [], [], None, 0.0, None
    for qc in circuits:
        oneq = sum(n for name, n in qc.count_ops().items() if name.startswith("gpi"))
        qubits = len({qc.find_bit(i.qubits[0]).index for i in qc.data if i.operation.name == "measure"})
        try:
            est = client.estimate_job(backend=str(backend.name), oneq_gates=int(oneq), twoq_gates=0,
                                      qubits=int(qubits), shots=int(shots), error_mitigation=False)
        except Exception:
            return None
        raw = est.to_dict()
        cost = est.cost if est.cost is not None else raw.get("estimated_total_cost")
        minimum = max([float(r.get("job_cost_minimum") or 0.0) for r in raw.get("rate_card", {}).get("rates", [])] or [0.0])
        costs.append(None if cost is None else max(float(cost), minimum))
        raws.append(raw)
        unit, queue = unit or est.cost_unit or raw.get("estimated_unit"), est.queue_time
        execution += float(est.exec_time or 0.0)
    return {"cost": None if None in costs else sum(costs), "cost_unit": unit, "per_circuit_cost": costs,
            "execution_time": execution, "queue_time": queue, "raw": raws}


def _mixed_order(sizes: Sequence[int]) -> List[int]:
    """Indices of circuits in the engine's order (consecutive groups of `sizes`, one per sequence length) arranged
    so every group is spread evenly over the list: consecutive jobs then each hold the same mix of lengths, and a
    difference between jobs shifts all lengths alike instead of bending the decay."""
    keys, start = [], 0
    for s, n in enumerate(sizes):
        keys += [((j + 0.5) / n, s, start + j) for j in range(n)]
        start += n
    return [k for _, _, k in sorted(keys)]


def _job_chunks(circuits: Sequence[Any], max_gates: int) -> List[Tuple[int, List[Any]]]:
    """Consecutive (first index, circuits) runs of at most max_gates native gates each (one circuit if it alone
    exceeds max_gates): the jobs one batch is submitted as, in order, so each stays under IonQ's payload limit.
    IonQ bills each circuit separately, so the split does not change the cost."""
    out, start, count = [], 0, 0
    for i, qc in enumerate(circuits):
        gates = sum(n for name, n in qc.count_ops().items() if name.startswith("gpi"))
        if i > start and count + gates > max_gates:
            out.append((start, list(circuits[start:i])))
            start, count = i, 0
        count += gates
    out.append((start, list(circuits[start:])))
    return out


def _p0(counts: Dict[str, float], bit: int) -> Tuple[float, float]:
    """(weight of 0, total weight) on clbit `bit` of a counts or probability
    dict whose keys are bitstrings with clbit 0 rightmost."""
    zero = total = 0.0
    for key, v in counts.items():
        key = key.replace(" ", "")
        total += float(v)
        if len(key) <= bit or key[-1 - bit] == "0":
            zero += float(v)
    return zero, total


def _stat(entry: Any) -> Tuple[Optional[float], Optional[str]]:
    """(value, statistic) of an IonQ fidelity entry: the median if given, else the mean."""
    if isinstance(entry, dict):
        for key in ("median", "mean"):
            if entry.get(key) is not None:
                return float(entry[key]), key
        return None, None
    return (float(entry), "value") if entry is not None else (None, None)


def ionq_priors(backend) -> Optional[Dict[str, Any]]:
    """The device's published numbers from IonQ's current characterization
    endpoint, through qiskit-ionq's backend.calibration(): eps_prior = 1 - the
    single-qubit fidelity (median, else mean) and spam = (P(0|1), P(1|0)) from
    the SPAM fidelity, split as the engine splits IonQ's combined SPAM error
    (inversion._TI_SPAM_RATIO_P0_OVER_P1). Values outside (0, 0.5) are left out.
    None if the backend has no characterization (simulators, mocks, API errors)."""
    try:
        cal = backend.calibration() if hasattr(backend, "calibration") else None
    except Exception:
        cal = None
    if cal is None:
        return None
    fid = cal.fidelity or {}
    out: Dict[str, Any] = {"backend": getattr(cal, "backend", None), "date": str(getattr(cal, "date", "")),
                           "raw": dict(getattr(cal, "_data", {}) or {}), "eps_prior": None, "spam": None}
    f1, stat = _stat(fid.get("1q"))
    if f1 is not None and 0.0 < 1.0 - f1 < 0.5:
        out["eps_prior"], out["eps_stat"] = 1.0 - f1, stat
        if isinstance(fid.get("1q"), dict) and fid["1q"].get("stderr") is not None:
            out["eps_stderr"] = float(fid["1q"]["stderr"])
    fs, _ = _stat(fid.get("spam"))
    if fs is not None:
        e = 1.0 - fs if fs > 0.5 else fs
        if 0.0 < e < 0.5:
            ratio = inv._TI_SPAM_RATIO_P0_OVER_P1
            out["spam"] = (e * 2.0 * ratio / (1.0 + ratio), e * 2.0 / (1.0 + ratio))
    return out


def run_ionq(backend, ions: Sequence[int], shots: int = DEFAULT_SHOTS,
             eps_prior: Optional[float] = None, spam: Optional[Tuple[float, float]] = None,
             max_ops: int = DEFAULT_MAX_OPS, max_job_gates: int = DEFAULT_MAX_JOB_GATES,
             max_cost: Optional[float] = None,
             run_options: Optional[Dict[str, Any]] = None,
             parallel: bool = True, seed: Optional[int] = None,
             probe_blocks: int = 1) -> IonQReport:
    """Run Canary's gate-error probe on `ions` of a qiskit-ionq native-gate
    backend (provider.get_backend("qpu.forte-1", gateset="native"), or the
    "simulator" backend with a noise model) and fit epsilon_sx per ion.

    eps_prior sets the sequence lengths (BackendProfile.rb_lengths);
    spam = (P(0|1), P(1|0)) is recorded with the report (readout cancels in
    the fit). Each one not given comes from the device's published
    characterization (ionq_priors), or the engine's trapped-ion value if there
    is none. seed fixes the random sequences (None: a fresh seed, recorded as
    rb_seed). One job per batch of ions (all ions in shared circuits unless
    that exceeds max_ops; parallel=False: one ion per batch), all circuits
    with the same shots; probe_blocks (a divisor
    of shots, default 1) runs each batch as that many jobs in order, each with
    shots / probe_blocks per circuit, and sums their counts. Each of those is
    sent as consecutive jobs of at most max_job_gates gates (IonQ's 10 MB
    payload limit; the cost is unchanged, IonQ bills per circuit), the
    circuits ordered so every job holds the same mix of sequence lengths
    (_mixed_order). With max_cost,
    IonQ's own estimate (ionq_estimate) of all jobs is fetched first and the
    run is refused if it is higher, or if IonQ returns no estimate.
    """
    if isinstance(probe_blocks, bool) or int(probe_blocks) != probe_blocks or probe_blocks < 1 \
            or int(shots) % int(probe_blocks):
        raise ValueError(f"probe_blocks must be an int >= 1 dividing shots={shots}, got {probe_blocks!r}")
    blocks, per_block = int(probe_blocks), int(shots) // int(probe_blocks)
    if eps_prior is None or spam is None:
        published = ionq_priors(backend) or {}
        eps_prior = eps_prior if eps_prior is not None else published.get("eps_prior")
        spam = spam if spam is not None else published.get("spam")
    arch = _arch(eps_prior, spam)
    rb_lengths = _profile(arch).rb_lengths
    rb_seed = int(np.random.SeedSequence().generate_state(1)[0]) if seed is None else int(seed)
    batches, circuits = ionq_circuits(ions, rb_lengths, max_ops, parallel, rb_seed)
    n_per_batch = sum(inv._RB_SEQUENCES)
    estimates = [e for b in range(len(batches))
                 for e in [ionq_estimate(backend, circuits[b * n_per_batch:(b + 1) * n_per_batch], per_block)] * blocks]
    if max_cost is not None:
        if any(e is None or e.get("cost") is None for e in estimates):
            raise ValueError("IonQ returned no cost estimate, so max_cost cannot be checked")
        total = sum(float(e["cost"]) for e in estimates)
        if total > max_cost:
            raise ValueError(f"IonQ estimates {total:,.2f} {estimates[0]['cost_unit']}, "
                             f"above max_cost={max_cost:,.2f}")

    options = {"shots": per_block}
    try:
        from qiskit_ionq import ErrorMitigation
        options["error_mitigation"] = ErrorMitigation.NO_DEBIASING
    except ImportError:
        pass
    options.update(run_options or {})

    zeros = {q: np.zeros(n_per_batch) for q in ions}
    totals = {q: np.zeros(n_per_batch) for q in ions}
    job_ids, raw = [], []
    for b, batch in enumerate(batches):
        batch_circuits = circuits[b * n_per_batch:(b + 1) * n_per_batch]
        order = _mixed_order(inv._RB_SEQUENCES)
        for block in range(blocks):
            for start, chunk in _job_chunks([batch_circuits[k] for k in order], max_job_gates):
                job = backend.run(chunk, **options)
                job_id = job.job_id() if hasattr(job, "job_id") else None
                job_ids.append(job_id)
                result = job.result()
                for i in range(len(chunk)):
                    k = order[start + i]
                    probs = None
                    try:
                        probs = result.data(i).get("probabilities")
                    except Exception:
                        probs = None
                    counts = result.get_counts(i)
                    source = probs if probs else counts
                    raw.append({"job_id": job_id, "batch": list(batch), "block": block, "circuit": k,
                                "counts": dict(counts), "probabilities": dict(probs) if probs else None})
                    for j, q in enumerate(batch):
                        zero, total = _p0(source, j)
                        zeros[q][k] += zero
                        totals[q][k] += total

    results = []
    dof = len(rb_lengths) - 2
    for q in ions:
        p0 = np.where(totals[q] > 0, zeros[q] / np.where(totals[q] > 0, totals[q], 1.0), np.nan)
        eps, sigma, chi2_dof, _ = inv._invert_rb(p0, rb_lengths, inv._RB_SEQUENCES, arch)
        results.append(IonResult(ion=int(q), epsilon_sx=float(eps), epsilon_sx_sigma=float(sigma),
                                 chi2_dof=chi2_dof, chi2_p=float(_chi2.sf(chi2_dof * dof, dof)),
                                 rb_lengths=[int(m) for m in rb_lengths], p0=[float(x) for x in p0],
                                 shots=[int(shots)] * n_per_batch))
    return IonQReport(backend=str(getattr(backend, "name", backend)), ions=[int(q) for q in ions],
                      shots=int(shots), rb_lengths=[int(m) for m in rb_lengths],
                      eps_prior=float(arch["eps_typical"]),
                      spam=(float(arch["p0_given_1"]), float(arch["p1_given_0"])),
                      batches=[list(b) for b in batches], job_ids=job_ids, ionq_estimates=estimates,
                      results=results, raw=raw, rb_seed=rb_seed, probe_blocks=blocks)


def is_ionq(backend) -> bool:
    """True for a qiskit-ionq backend (QPU or cloud simulator)."""
    try:
        from qiskit_ionq.ionq_backend import IonQBackend
    except ImportError:
        return False
    return isinstance(backend, IonQBackend)


def ion_health(epsilon_sx_sigma: float, chi2_dof: float) -> str:
    """The IBM health rule restricted to the one probe run on ions: RED if the
    gate fit failed (inversion._fit_failed), YELLOW if its chi2/dof is over the
    gate probe's limit (online._green_chi2_threshold("gate"), the same limit as
    on IBM), otherwise GREEN. There is no T2*/T2 rule: T2 is not measured."""
    if inv._fit_failed(epsilon_sx_sigma):
        return "RED"
    if np.isfinite(chi2_dof) and chi2_dof > online._green_chi2_threshold("gate"):
        return "YELLOW"
    return "GREEN"


def to_canary_report(report: IonQReport) -> "runner.CanaryReport":
    """The same report type run_ibm returns, so Report, save/load, compare and
    Monitor work unchanged. Only epsilon_sx is measured: T1, T2, T2*, delta_omega
    and the other probes' chi2/dof are NaN (not measured). A failed gate fit is
    reported as NaN +/- inf, as on IBM. block_counts holds each ion's raw counts
    per gate circuit."""
    nan, arch = float("nan"), _arch(report.eps_prior, report.spam)
    profile = _profile(arch)
    raw_by_ion: Dict[int, List[Dict[str, Any]]] = {q: [] for q in report.ions}
    for entry in report.raw:
        for bit, ion in enumerate(entry["batch"]):
            raw_by_ion[int(ion)].append({"circuit": entry["circuit"], "block": entry.get("block", 0),
                                         "bit": bit, "counts": entry["counts"],
                                         "probabilities": entry["probabilities"]})
    qubits = []
    for r in report.results:
        failed = inv._fit_failed(r.epsilon_sx_sigma)
        result = inv.InversionResult(
            backend_name=report.backend, qubit_id=r.ion, timestamp=report.timestamp,
            architecture="trapped_ion", T1_s=nan, T1_sigma_s=nan, T2_s=nan, T2_sigma_s=nan,
            delta_omega=nan, delta_omega_sigma=nan,
            epsilon_sx=nan if failed else r.epsilon_sx,
            epsilon_sx_sigma=float("inf") if failed else r.epsilon_sx_sigma,
            t1_residual=nan, ramsey_residual=nan, gate_residual=nan,
            t1_chi2_dof=nan, ramsey_chi2_dof=nan, gate_chi2_dof=r.chi2_dof,
            T2_ramsey_s=nan, T2_ramsey_sigma_s=nan, T2_echo_s=nan, T2_echo_sigma_s=nan,
            echo_residual=nan, echo_chi2_dof=nan, rb_seed=report.rb_seed)
        qubits.append(runner.QubitReport(qubit=r.ion, result=result, profile=profile,
                                         health=ion_health(r.epsilon_sx_sigma, r.chi2_dof)))
    return runner.CanaryReport(
        mode="hardware", backend=report.backend, architecture="trapped_ion",
        shots={"t1": 0, "ramsey": 0, "gate": report.shots, "echo": 0}, probe_blocks=report.probe_blocks,
        circuits_per_qubit=sum(inv._RB_SEQUENCES), qubits=qubits, timestamp=report.timestamp,
        job_id=",".join(str(j) for j in report.job_ids if j) or None,
        groups=[list(b) for b in report.batches], block_counts=raw_by_ion)
