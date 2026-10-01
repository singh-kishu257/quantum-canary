"""ionpack: pack small independent circuits onto disjoint ion blocks of one IonQ circuit."""

from .cost import AZURE_FORTE_RATES, BRAKET_FORTE, Pricing, circuit_cost, count_gates, packed_savings
from .ionq import packed_counts, run_packed  # qiskit_ionq is imported lazily inside these
from .layout import Layout, capacity, latin_rotation, make_layout
from .pack import NATIVE_GATES, PackedCircuit, check_circuit, is_sequential, pack
from .stats import BlockMoments, TostResult, block_moments, design_effect, indicator
from .stats import linear_estimate, welch_tost, z_parity
from .unpack import marginal_counts, sentinel_excitation, split_counts

__version__ = "0.1.0"

__all__ = [
    "AZURE_FORTE_RATES", "BRAKET_FORTE", "BlockMoments", "Layout", "NATIVE_GATES", "PackedCircuit",
    "Pricing", "TostResult", "block_moments", "capacity", "check_circuit", "circuit_cost",
    "count_gates", "design_effect", "indicator", "is_sequential", "latin_rotation",
    "linear_estimate", "make_layout", "marginal_counts", "pack", "packed_counts", "packed_savings",
    "run_packed", "sentinel_excitation", "split_counts", "welch_tost", "z_parity", "__version__",
]
