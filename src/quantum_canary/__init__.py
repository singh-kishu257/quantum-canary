"""Quantum Canary: online ion-health learning and qubit-to-ion assignment for trapped-ion QPUs."""

from .assignment import AssignmentSolver
from .canary import Canary
from .compiler import compile_native
from .config import CanaryConfig
from .decision import Decision, SequentialTest, Status
from .design import BatchDesigner
from .estimator import HealthModel
from .executors import BackendExecutor, Executor, IonQExecutor
from .probes import return_probe_program
from .program import Gate, LogicalProgram, Observable
from .resources import ion, pair
from .results import AssignmentRun, BatchPlan, BatchResult, ProbeRun

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
    "Decision",
    "Executor",
    "Gate",
    "HealthModel",
    "IonQExecutor",
    "LogicalProgram",
    "Observable",
    "ProbeRun",
    "SequentialTest",
    "Status",
    "__version__",
    "compile_native",
    "return_probe_program",
    "ion",
    "pair",
]
