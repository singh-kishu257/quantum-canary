"""
Quantum Canary — Real-time noise drift detection for NISQ-era quantum processors.

Public API:
    Canary              — Main class for connecting to a backend and running
                          drift checks / continuous monitoring.
    MLPEnsemble         — Loads the bundled 10-seed MLP ensemble.
    build_canary_circuits — Construct the three canary circuits (Bell,
                            coherence, gate-error).

Example
-------
>>> from quantum_canary import Canary
>>> canary = Canary(token="...", instance="...", backend="ibm_kingston")
>>> result = canary.check()
>>> print(result["verdict"])
STABLE
"""

from quantum_canary.monitor    import Canary
from quantum_canary.classifier import MLPEnsemble
from quantum_canary.circuits   import build_canary_circuits

__version__ = "0.1.0"
__author__  = "Kanishka Singh"
__license__ = "MIT"

__all__ = [
    "Canary",
    "MLPEnsemble",
    "build_canary_circuits",
    "__version__",
]

