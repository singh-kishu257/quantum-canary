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
up the ions). Each ion is probed in its own circuits by default: on Forte-1
in August 2026, running several ions in the same circuit gave results
disturbed by crosstalk. parallel=True shares circuits between ions, in
batches of at most max_ops gates.

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

    def to_dict(self) -> Dict[str, Any]:
        return {"backend": self.backend, "ions": list(self.ions), "shots": self.shots,
                "rb_lengths": list(self.rb_lengths), "rb_seed": self.rb_seed,
                "eps_prior": self.eps_prior,
                "p0_given_1": self.spam[0], "p1_given_0": self.spam[1],
                "batches": [list(b) for b in self.batches], "job_ids": list(self.job_ids),
                "ionq_estimates": list(self.ionq_estimates), "timestamp": self.timestamp,
                "results": [r.to_dict() for r in self.results], "raw": self.raw}

    def render(self) -> str:
        lines = [f"IonQ {self.backend}: {len(self.ions)} ions, RB lengths {self.rb_lengths} x "
                 f"{list(inv._RB_SEQUENCES)} sequences, {self.shots} shots, "
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
                  max_ops: int = DEFAULT_MAX_OPS, parallel: bool = False,
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
    its /jobs/estimate endpoint (qiskit-ionq's client.estimate_job), given the
    job's native gate count and qubit count: a dict with IonQ's estimated
    cost, cost unit, execution and queue time and the raw response, or None
    if the backend gives no estimate (simulators, mock backends, API errors)."""
    client = getattr(backend, "client", None)
    if client is None or not hasattr(client, "estimate_job"):
        return None
    oneq = sum(n for qc in circuits for name, n in qc.count_ops().items() if name.startswith("gpi"))
    qubits = len({qc.find_bit(i.qubits[0]).index for qc in circuits for i in qc.data
                  if i.operation.name == "measure"})
    try:
        est = client.estimate_job(backend=str(backend.name), oneq_gates=int(oneq), twoq_gates=0,
                                  qubits=int(qubits), shots=int(shots), error_mitigation=False)
    except Exception:
        return None
    return {"cost": est.cost, "cost_unit": est.cost_unit, "execution_time_s": est.exec_time,
            "queue_time_s": est.queue_time, "raw": est.to_dict()}


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
             max_ops: int = DEFAULT_MAX_OPS, max_cost: Optional[float] = None,
             run_options: Optional[Dict[str, Any]] = None,
             parallel: bool = False, seed: Optional[int] = None) -> IonQReport:
    """Run Canary's gate-error probe on `ions` of a qiskit-ionq native-gate
    backend (provider.get_backend("qpu.forte-1", gateset="native"), or the
    "simulator" backend with a noise model) and fit epsilon_sx per ion.

    eps_prior sets the sequence lengths (BackendProfile.rb_lengths);
    spam = (P(0|1), P(1|0)) is recorded with the report (readout cancels in
    the fit). Each one not given comes from the device's published
    characterization (ionq_priors), or the engine's trapped-ion value if there
    is none. seed fixes the random sequences (None: a fresh seed, recorded as
    rb_seed). One job per batch of ions (one ion per batch unless
    parallel=True), all circuits with the same shots. With max_cost, IonQ's
    own estimate (ionq_estimate) of all jobs is fetched first and the run is
    refused if it is higher, or if IonQ returns no estimate.
    """
    if eps_prior is None or spam is None:
        published = ionq_priors(backend) or {}
        eps_prior = eps_prior if eps_prior is not None else published.get("eps_prior")
        spam = spam if spam is not None else published.get("spam")
    arch = _arch(eps_prior, spam)
    rb_lengths = _profile(arch).rb_lengths
    rb_seed = int(np.random.SeedSequence().generate_state(1)[0]) if seed is None else int(seed)
    batches, circuits = ionq_circuits(ions, rb_lengths, max_ops, parallel, rb_seed)
    n_per_batch = sum(inv._RB_SEQUENCES)
    estimates = [ionq_estimate(backend, circuits[b * n_per_batch:(b + 1) * n_per_batch], shots)
                 for b in range(len(batches))]
    if max_cost is not None:
        if any(e is None or e.get("cost") is None for e in estimates):
            raise ValueError("IonQ returned no cost estimate, so max_cost cannot be checked")
        total = sum(float(e["cost"]) for e in estimates)
        if total > max_cost:
            raise ValueError(f"IonQ estimates {total:,.2f} {estimates[0]['cost_unit']}, "
                             f"above max_cost={max_cost:,.2f}")

    options = {"shots": int(shots)}
    try:
        from qiskit_ionq import ErrorMitigation
        options["error_mitigation"] = ErrorMitigation.NO_DEBIASING
    except ImportError:
        pass
    options.update(run_options or {})

    per_ion: Dict[int, List[float]] = {q: [] for q in ions}
    job_ids, raw = [], []
    for b, batch in enumerate(batches):
        batch_circuits = circuits[b * n_per_batch:(b + 1) * n_per_batch]
        job = backend.run(batch_circuits, **options)
        job_id = job.job_id() if hasattr(job, "job_id") else None
        job_ids.append(job_id)
        result = job.result()
        for k in range(len(batch_circuits)):
            probs = None
            try:
                probs = result.data(k).get("probabilities")
            except Exception:
                probs = None
            counts = result.get_counts(k)
            source = probs if probs else counts
            raw.append({"job_id": job_id, "batch": list(batch), "circuit": k,
                        "counts": dict(counts), "probabilities": dict(probs) if probs else None})
            for j, q in enumerate(batch):
                zero, total = _p0(source, j)
                per_ion[q].append(zero / total if total else float("nan"))

    results = []
    dof = len(rb_lengths) - 2
    for q in ions:
        p0 = np.array(per_ion[q])
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
                      results=results, raw=raw, rb_seed=rb_seed)


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
            raw_by_ion[int(ion)].append({"circuit": entry["circuit"], "block": 0,
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
        shots={"t1": 0, "ramsey": 0, "gate": report.shots, "echo": 0}, probe_blocks=1,
        circuits_per_qubit=sum(inv._RB_SEQUENCES), qubits=qubits, timestamp=report.timestamp,
        job_id=",".join(str(j) for j in report.job_ids if j) or None,
        groups=[list(b) for b in report.batches], block_counts=raw_by_ion)
