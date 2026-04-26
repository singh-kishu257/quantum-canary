# Quantum Canary
 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)]()
 
**Real-time noise drift detection for NISQ-era quantum processors.**
 
Quantum Canary is a physics-informed neural network that detects qubit hardware degradation between IBM Quantum's calibration snapshots, providing researchers with an early-warning signal hours before published calibration data updates.
 
---
 
## Why this exists
 
IBM Quantum refreshes published qubit health data on a daily cycle. Between those snapshots, environmental noise — thermal fluctuations, electromagnetic interference, TLS defects — silently degrades qubit fidelity. Computations run during this unmonitored window may return corrupted results without any indication. Quantum Canary closes this gap.
 
## Headline results
 
| Method | Test AUC | Notes |
|---|---|---|
| Per-feature majority-vote threshold | 0.7586 | Standard Statistical baseline |
| Hotelling's T² (multivariate SPC, 1947) | 0.7674 | Rigorous statistical baseline |
| **Quantum Canary MLP ensemble** | **0.9239** | **+20.4% over Hotelling T²** |
 
On IBM Kingston (h2 qpu) Quantum Canary anticipated **3 of 3 IBM recalibration events** with an average lead time of **3.34 hours**
 
---
 
## Install
 
```bash
pip install quantum-canary
```
 
---
 
## Run it (the easy way)
 
Just type one command and follow the prompts:
 
```bash
quantum-canary
```
 
You'll see:
 
```
+-----------------------------------------------------------+
|              Welcome to Quantum Canary                    |
|    Real-time noise drift detection for NISQ-era qubits    |
+-----------------------------------------------------------+
 
What would you like to do?
  [1] Run a single drift check
  [2] Continuous monitoring
  [3] Classify pre-collected fidelity data (no IBM connection)
  [4] Exit
> 1
 
  IBM Quantum API token: ************************
  IBM Cloud CRN:         ************************
 
Connecting to IBM Quantum... ✓
Fetching available backends... ✓
 
  [1] ibm_kingston   (156 qubits, 3 jobs queued)
  [2] ibm_fez        (156 qubits, 1 job queued)
 
  Select backend [1-2]: 1
  Shots per circuit [1000]:
```
 
The interactive wizard handles credentials, backend selection, intervals, and durations. Token input is hidden as you type.
 
---
 
## Run it from Python (3 lines)
 
```python
from quantum_canary import Canary
 
canary = Canary(token="YOUR_TOKEN", instance="YOUR_CRN", backend="IBM_YOURBACKEND")
canary.monitor(hours=24)
```
 
That's it. Three lines and you have continuous drift monitoring with logs and verdicts written to disk every 15 minutes.
 
For one-off checks instead of continuous monitoring:
 
```python
result = canary.check()
print(result)
# {'drift_probability': 0.034, 'verdict': 'STABLE', 'uncertainty': 0.001, ...}
```
 
For drop-in inference on fidelities you've already collected (no IBM connection):
 
```python
from quantum_canary import MLPEnsemble
import numpy as np
 
mlp = MLPEnsemble()
result = mlp.predict(np.array([0.94, 0.92, 0.95]))   # F_bell, F_gate, F_coherence
```
 
---
 
## Run it from the command line (for scripts and CI)
 
```bash
# Single check
quantum-canary check --backend IBM_YOURBACKEND
 
# Continuous monitoring
quantum-canary monitor --backend IBM_YOURBACKEND --hours 24 --interval 15
```
 
Provide credentials in any of these ways (in order of preference):
 
```bash
# Option A — environment variables (recommended for scripts)
export IBM_QUANTUM_TOKEN="your_token"
export IBM_QUANTUM_CRN="your_crn"
quantum-canary check --backend IBM_YOURBACKEND
 
# Option B — file paths
quantum-canary check --backend IBM_YOURBACKEND \
    --token api_token.txt \
    --instance crn_instance.txt
 
# Option C — direct flags (least secure, but works)
quantum-canary check --backend IBM_YOURBACKEND \
    --token "..." --instance "..."
```
 
---
 
## How it works
 
Three lightweight quantum circuits ("canary circuits") run on the target backend at regular intervals:
 
1. **Bell state circuit** — measures 2-qubit entanglement fidelity (F_bell)
2. **Coherence circuit** (20× H identity) — measures T2 dephasing (F_coherence)
3. **Gate-error circuit** (20× X identity) — measures single-qubit gate accumulation (F_gate)
The three measured fidelities feed into a 10-seed MLP ensemble trained on a 21,000-sample physics-grounded hybrid dataset (15,000 Qiskit Aer simulations + 6,000 real IBM calibration anchors). The ensemble outputs a drift probability and an uncertainty estimate from inter-seed disagreement.
 
---
 
## Repository structure
 
```
quantum-canary/
├── src/quantum_canary/      # The pip-installable package
├── research/                # Numbered scripts that reproduce the paper
│   └── README.md            # Research methodology and pipeline
└── results/                 # Live deployment logs and evaluation results
```
 
For reproducing the paper end-to-end, see [`research/README.md`](research/README.md).
 
---
 
## Tradeoffs and queue impact
 
Quantum Canary submits short canary circuits to your backend at the chosen interval. These submissions share the queue with your other jobs and consume shot budget. On IBM's free tier, continuous monitoring at 15-minute intervals over 24 hours uses approximately 288,000 shots. For paid tiers, queue impact is typically negligible.
 
If you want **zero queue overhead**, use the `MLPEnsemble.predict()` method to score fidelity measurements you've already collected from your own circuits.
 
---
 
## Citation
 
If you use Quantum Canary in your research, please cite:
 
```bibtex
@misc{singh2026quantumcanary,
  author       = {Singh, Kanishka},
  title        = {Quantum Canary: Real-Time Noise Drift Detection for
                  NISQ-Era Quantum Processors},
  year         = {2026},
  howpublished = {\url{https://github.com/singh-kishu257/quantum-canary}},
}
```
 
---
 
## License
 
MIT — see [LICENSE](LICENSE).
 
---
 
## Author
 
**Kanishka Singh** · 9th Grade · Urbana High School
 
Feedback and contributions welcome via [GitHub Issues](https://github.com/singh-kishu257/quantum-canary/issues).
