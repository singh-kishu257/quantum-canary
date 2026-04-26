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

Validated on held-out testing datasets & live deployment on `ibm_kingston`:

| Method                                  | Test AUC   | Notes                         |
| --------------------------------------- | ---------- | ----------------------------- |
| Per-feature majority-vote threshold     | 0.7586     | Naive baseline                |
| Hotelling's T² (multivariate SPC, 1947) | 0.7674     | Rigorous statistical baseline |
| **Quantum Canary MLP ensemble**         | **0.9239** | **+20.4% over Hotelling T²**  |

---

## Quickstart

### Install

```bash
pip install quantum-canary
```

### Interactive mode (recommended for first-time users)

```bash
quantum-canary
```

You'll be walked through credential entry, backend selection, and configuration choices.

### One-shot drift check

```python
from quantum_canary import Canary

canary = Canary(
    token="YOUR_IBM_QUANTUM_TOKEN",
    instance="YOUR_CRN",
    backend="ibm_kingston",
)

result = canary.check()
print(result)
# {'drift_probability': 0.034, 'verdict': 'STABLE',
#  'uncertainty': 0.001, 'F_bell': 0.987, ...}
```

### Continuous monitoring

```python
canary.monitor(interval_minutes=15, hours=24)
# Logs to logs/realtime_log.csv
```

### Drop-in inference (no IBM connection required)

If you've already collected fidelity measurements from your own circuits:

```python
result = canary.classify(f_bell=0.94, f_gate=0.92, f_coherence=0.95)
```

### Command-line

```bash
# Single check
quantum-canary check --backend ibm_kingston --token YOUR_TOKEN --instance YOUR_CRN

# Continuous monitoring
quantum-canary monitor --backend ibm_kingston --hours 24 --interval 15
```

---

## How it works

Three lightweight quantum circuits ("canary circuits") are run on the target backend at regular intervals:

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
└── results/                 # Live deployment logs and evaluation results
```

For reproducing the paper end-to-end, see [`research/`](research/).

---

## Tradeoffs and queue impact

Quantum Canary submits short canary circuits to your backend at the chosen interval. These submissions share the queue with your other jobs and consume shot budget. On IBM's free tier, continuous monitoring at 15-minute intervals over 24 hours uses approximately 288,000 shots. For paid tiers, queue impact is typically negligible.

If you want **zero queue overhead**, use the `classify()` method to score fidelity measurements you've already collected from your own circuits.

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
