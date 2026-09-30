"""Quantum Canary: online ion-health learning and qubit-to-ion assignment for trapped-ion QPUs."""

from .assignment import AssignmentSolver
from .canary import Canary
from .compiler import CompiledCircuit, compile_native, compile_twirled
from .config import CanaryConfig
from .decision import Decision, HealthTest, SequentialTest, Status, e_bh
from .design import BatchDesigner
from .estimator import CoherentModel, HealthModel
from .executors import BackendExecutor, Executor, IonQExecutor
from .probes import Observation, cb_probe_program, phase_probe_program, return_probe_program, spam_probe_program
from .program import Gate, LogicalProgram, Observable
from .resources import ion, pair
from .results import AssignmentRun, BatchPlan, BatchResult, ProbeRun, Tier2Fit

__version__ = "0.1.0"

__all__ = [
    "AssignmentRun",
    "AssignmentSolver",
    "BackendExecutor",
    "BatchDesigner",
    "BatchPlan",
    "BatchResult",
    "Canary",
    "CanaryConfig",
    "CoherentModel",
    "CompiledCircuit",
    "Decision",
    "Executor",
    "Gate",
    "HealthModel",
    "HealthTest",
    "IonQExecutor",
    "LogicalProgram",
    "Observable",
    "Observation",
    "ProbeRun",
    "SequentialTest",
    "Status",
    "Tier2Fit",
    "__version__",
    "cb_probe_program",
    "compile_native",
    "compile_twirled",
    "e_bh",
    "ion",
    "pair",
    "phase_probe_program",
    "return_probe_program",
    "spam_probe_program",
]
