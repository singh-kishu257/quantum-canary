"""IonQ trapped-ion hardware: Canary's gate-error probe on chosen ions.

On trapped ions T1 and T2 are seconds long, far beyond any circuit, so only
the gate-repetition family is run: N native (GPI2, GPI2^-1) pairs with a
random virtual-Z frame per pair, then a Z measurement, for the three N of
BackendProfile.gate_rep_n. Each ion's epsilon_sx comes from the engine's
own gate fit (inversion._invert_gate) with the readout errors given.

    report = run_ionq(backend, ions=[3, 6, 14], eps_prior=2e-4,
                      spam=(0.006, 0.004), max_cost=1500)

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

import contextlib
import io
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import chi2 as _chi2

from . import inversion as inv

__all__ = ["IonResult", "IonQReport", "ionq_circuits", "ionq_estimate", "run_ionq"]

DEFAULT_SHOTS = 500
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
    gate_rep_n: List[int]
    p0: List[float]
    shots: List[int]

    @property
    def poor_fit(self) -> bool:
        return bool(np.isfinite(self.chi2_p) and self.chi2_p < POOR_FIT_P)

    def to_dict(self) -> Dict[str, Any]:
        return {"ion": self.ion, "epsilon_sx": self.epsilon_sx,
                "epsilon_sx_sigma": self.epsilon_sx_sigma, "chi2_dof": self.chi2_dof,
                "chi2_p": self.chi2_p, "poor_fit": self.poor_fit,
                "gate_rep_n": list(self.gate_rep_n), "p0": list(self.p0),
                "shots": list(self.shots)}


@dataclass
class IonQReport:
    """Result of run_ionq()."""
    backend: str
    ions: List[int]
    shots: int
    gate_rep_n: List[int]
    eps_prior: float
    spam: Tuple[float, float]
    batches: List[List[int]]
    job_ids: List[str]
    ionq_estimates: List[Optional[Dict[str, Any]]]
    results: List[IonResult]
    raw: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {"backend": self.backend, "ions": list(self.ions), "shots": self.shots,
                "gate_rep_n": list(self.gate_rep_n), "eps_prior": self.eps_prior,
                "p0_given_1": self.spam[0], "p1_given_0": self.spam[1],
                "batches": [list(b) for b in self.batches], "job_ids": list(self.job_ids),
                "ionq_estimates": list(self.ionq_estimates), "timestamp": self.timestamp,
                "results": [r.to_dict() for r in self.results], "raw": self.raw}

    def render(self) -> str:
        lines = [f"IonQ {self.backend}: {len(self.ions)} ions, N = {self.gate_rep_n} pairs, "
                 f"{self.shots} shots, jobs {', '.join(str(j) for j in self.job_ids)}",
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


def ionq_circuits(ions: Sequence[int], gate_rep_n: Sequence[int],
                  max_ops: int = DEFAULT_MAX_OPS,
                  parallel: bool = False) -> Tuple[List[List[int]], List[Any]]:
    """Batches of ions and, for each batch, one circuit per N: every ion of the
    batch runs the engine's gate-repetition circuit on its own qubit and is
    measured into its own clbit (clbit j is batch[j]). Without parallel every
    batch is a single ion."""
    from qiskit import QuantumCircuit
    ions = [int(q) for q in ions]
    batches = (inv.split_qubits_for_op_limit(ions, 2 * max(gate_rep_n), max_ops)
               if parallel else [[q] for q in ions])
    circuits = []
    for batch in batches:
        for n in gate_rep_n:
            probe = inv._build_gate_rep_circuit(int(n), "trapped_ion")
            qc = QuantumCircuit(max(batch) + 1, len(batch), name=f"canary_N{n}_ions{'-'.join(map(str, batch))}")
            for j, q in enumerate(batch):
                qc.compose(probe, qubits=[q], clbits=[j], inplace=True)
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


def run_ionq(backend, ions: Sequence[int], shots: int = DEFAULT_SHOTS,
             eps_prior: Optional[float] = None, spam: Optional[Tuple[float, float]] = None,
             max_ops: int = DEFAULT_MAX_OPS, max_cost: Optional[float] = None,
             run_options: Optional[Dict[str, Any]] = None,
             parallel: bool = False) -> IonQReport:
    """Run Canary's gate-error probe on `ions` of a qiskit-ionq native-gate
    backend (provider.get_backend("qpu.forte-1", gateset="native"), or the
    "simulator" backend with a noise model) and fit epsilon_sx per ion.

    eps_prior sets the repetition counts (BackendProfile.gate_rep_n);
    spam = (P(0|1), P(1|0)) are the readout errors the fit corrects for. Both
    default to the engine's trapped-ion values; pass the device's published
    numbers when you have them. One job per batch of ions (one ion per batch
    unless parallel=True), all circuits with the same shots. With max_cost,
    IonQ's own estimate (ionq_estimate) of all jobs is fetched first and the
    run is refused if it is higher, or if IonQ returns no estimate.
    """
    arch = _arch(eps_prior, spam)
    gate_rep_n = _profile(arch).gate_rep_n
    batches, circuits = ionq_circuits(ions, gate_rep_n, max_ops, parallel)
    n_per_batch = len(gate_rep_n)
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

    per_ion: Dict[int, Dict[int, Tuple[float, float]]] = {q: {} for q in ions}
    job_ids, raw = [], []
    for b, batch in enumerate(batches):
        batch_circuits = circuits[b * n_per_batch:(b + 1) * n_per_batch]
        job = backend.run(batch_circuits, **options)
        job_id = job.job_id() if hasattr(job, "job_id") else None
        job_ids.append(job_id)
        result = job.result()
        for k, (qc, n) in enumerate(zip(batch_circuits, gate_rep_n)):
            probs = None
            try:
                probs = result.data(k).get("probabilities")
            except Exception:
                probs = None
            counts = result.get_counts(k)
            source = probs if probs else counts
            raw.append({"job_id": job_id, "batch": list(batch), "N": int(n),
                        "counts": dict(counts), "probabilities": dict(probs) if probs else None})
            for j, q in enumerate(batch):
                zero, total = _p0(source, j)
                per_ion[q][int(n)] = (zero / total if total else float("nan"), total)

    results = []
    for q in ions:
        n_vals = np.array(gate_rep_n, dtype=float)
        p0 = np.array([per_ion[q][int(n)][0] for n in gate_rep_n])
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            eps, sigma, _, _ = inv._invert_gate(p0, n_vals, arch, int(shots))
        model = inv.forward_gate(n_vals, eps, arch["p0_given_1"], arch["p1_given_0"])
        var = np.clip(model * (1 - model), 1e-12, None) / int(shots)
        dof = len(n_vals) - 1
        chi2_dof = float(np.sum((p0 - model) ** 2 / var) / dof)
        results.append(IonResult(ion=int(q), epsilon_sx=float(eps), epsilon_sx_sigma=float(sigma),
                                 chi2_dof=chi2_dof, chi2_p=float(_chi2.sf(chi2_dof * dof, dof)),
                                 gate_rep_n=[int(n) for n in gate_rep_n], p0=[float(x) for x in p0],
                                 shots=[int(shots)] * len(gate_rep_n)))
    return IonQReport(backend=str(getattr(backend, "name", backend)), ions=[int(q) for q in ions],
                      shots=int(shots), gate_rep_n=[int(n) for n in gate_rep_n],
                      eps_prior=float(arch["eps_typical"]),
                      spam=(float(arch["p0_given_1"]), float(arch["p1_given_0"])),
                      batches=[list(b) for b in batches], job_ids=job_ids, ionq_estimates=estimates,
                      results=results, raw=raw)
