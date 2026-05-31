"""
generate_figures.py — Quantum Canary research README figure generator

Generates four figures used in research/README.md:
    figures/circuit_bell.png         — Bell state canary circuit
    figures/circuit_coherence.png    — Coherence canary circuit
    figures/circuit_gate_error.png   — Gate-error canary circuit
    figures/mlp_architecture.png     — Neuron-level MLP ensemble diagram

Run from the research/ folder:
    python generate_figures.py

Requires: qiskit, matplotlib, numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

SCRIPT_DIR  = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"

FIGURES_DIR.mkdir(exist_ok=True)

# ── 1. BELL STATE CIRCUIT ─────────────────────────────────────────────────────
qc_bell = QuantumCircuit(2, 2, name="Bell canary")
qc_bell.h(0)
qc_bell.cx(0, 1)
qc_bell.measure([0, 1], [0, 1])

fig = circuit_drawer(qc_bell, output="mpl", style={"backgroundcolor": "#FFFFFF"})
fig.savefig(FIGURES_DIR / "circuit_bell.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved figures/circuit_bell.png")

# ── 2. COHERENCE CIRCUIT ──────────────────────────────────────────────────────
qc_coh = QuantumCircuit(1, 1, name="Coherence canary")
for _ in range(10):  # 10 H·H pairs = 20 H gates total
    qc_coh.h(0)
    qc_coh.h(0)
qc_coh.measure(0, 0)

fig = circuit_drawer(qc_coh, output="mpl", fold=20,
                     style={"backgroundcolor": "#FFFFFF"})
fig.savefig(FIGURES_DIR / "circuit_coherence.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved figures/circuit_coherence.png")

# ── 3. GATE-ERROR CIRCUIT ─────────────────────────────────────────────────────
qc_gate = QuantumCircuit(1, 1, name="Gate-error canary")
for _ in range(20):
    qc_gate.x(0)
qc_gate.measure(0, 0)

fig = circuit_drawer(qc_gate, output="mpl", fold=20,
                     style={"backgroundcolor": "#FFFFFF"})
fig.savefig(FIGURES_DIR / "circuit_gate_error.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved figures/circuit_gate_error.png")

# ── 4. MLP ARCHITECTURE — neuron-level diagram ────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis("off")
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# Layer sizes and x-positions
layer_sizes = [3, 64, 32, 16, 1]
layer_labels = [
    "Input\n(3 features)",
    "Dense(64)\nBN → ReLU → Dropout(0.3)",
    "Dense(32)\nBN → ReLU → Dropout(0.3)",
    "Dense(16)\nReLU",
    "Output\nSigmoid",
]
layer_x = [1.0, 4.0, 7.0, 10.0, 13.0]
y_center = 3.5

# Visual cap on neurons drawn — full layers shown for small ones,
# truncated dots for the 64- and 32-neuron layers.
display_max = 8
neuron_radius = 0.10

def neuron_y_positions(n_visual, span=4.5):
    """Evenly distribute n_visual neurons in a vertical span centered on y_center."""
    if n_visual == 1:
        return [y_center]
    return list(np.linspace(y_center - span / 2, y_center + span / 2, n_visual))

# For each layer, decide how many neurons to actually draw
visual_counts = []
for n in layer_sizes:
    visual_counts.append(min(n, display_max))

# Compute neuron y-positions for each layer
layer_positions = []
for n_visual in visual_counts:
    layer_positions.append(neuron_y_positions(n_visual))

# Draw connections (lighter weight for clarity)
for i in range(len(layer_sizes) - 1):
    src_positions = layer_positions[i]
    dst_positions = layer_positions[i + 1]
    for sy in src_positions:
        for dy in dst_positions:
            ax.plot([layer_x[i], layer_x[i + 1]], [sy, dy],
                    color="#cccccc", linewidth=0.4, zorder=1)

# Draw neurons
neuron_colors = ["#7fb069", "#3a86ff", "#3a86ff", "#3a86ff", "#e63946"]
for i, (x, positions, color) in enumerate(zip(layer_x, layer_positions, neuron_colors)):
    for y in positions:
        circle = patches.Circle((x, y), neuron_radius, facecolor=color,
                                edgecolor="black", linewidth=0.8, zorder=2)
        ax.add_patch(circle)

    # If layer is truncated, add ellipsis dots at top and bottom
    if layer_sizes[i] > display_max:
        ax.text(x, max(positions) + 0.35, "·\n·\n·",
                ha="center", va="center", fontsize=10, color="#444")
        ax.text(x, min(positions) - 0.55, "·\n·\n·",
                ha="center", va="center", fontsize=10, color="#444")

    # Layer count label below
    ax.text(x, 0.6, f"{layer_sizes[i]} neuron{'s' if layer_sizes[i] > 1 else ''}",
            ha="center", va="center", fontsize=10, fontweight="bold")

    # Layer description above
    ax.text(x, 6.4, layer_labels[i],
            ha="center", va="center", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f8f8",
                      edgecolor="#888", linewidth=0.6))

# Title
ax.text(7.0, 7.0,
        "Quantum Canary MLP Architecture  —  3 → 64 → 32 → 16 → 1",
        ha="center", va="center", fontsize=12, fontweight="bold")

# Bottom annotation
ax.text(7.0, 0.05,
        "Single MLP shown. Ensemble = 10 independently seeded copies of this network.",
        ha="center", va="center", fontsize=9, fontstyle="italic", color="#444")

plt.savefig(FIGURES_DIR / "mlp_architecture.png", dpi=200,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Saved figures/mlp_architecture.png")

print("\nAll figures generated. Reference them in research/README.md as:")
print("  ![Bell circuit](figures/circuit_bell.png)")
print("  ![Coherence circuit](figures/circuit_coherence.png)")
print("  ![Gate-error circuit](figures/circuit_gate_error.png)")
print("  ![MLP architecture](figures/mlp_architecture.png)")