"""The packaged Canary must be the research Canary.

1. src/quantum_canary/inversion.py is byte-identical to research/1_inversion.py.
2. online.py / simulation.py differ from research/1b_online.py /
   research/1c_simulation_models.py only in how they import the engine.
3. quantum_canary.run_simulation() reproduces, value for value, one instance of
   research/2_ideal_parity.py (ideal) and research/3_benchmark.py (nisq,
   default settings) for both architectures: same estimates, sigmas and truth.

Run with: python -m pytest tests  (the research/ tree must be present).
"""
from __future__ import annotations

import difflib
import importlib.util
import math
import pathlib
import sys
import warnings

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
PKG = ROOT / "src" / "quantum_canary"

pytestmark = pytest.mark.skipif(not RESEARCH.is_dir(), reason="research/ tree not present")

PARAMS = ("T1_s", "T2_s", "delta_omega", "epsilon_sx")
RESULT_ATTR = {"T1_s": ("T1_s", "T1_sigma_s"), "T2_s": ("T2_s", "T2_sigma_s"),
               "delta_omega": ("delta_omega", "delta_omega_sigma"),
               "epsilon_sx": ("epsilon_sx", "epsilon_sx_sigma")}


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, RESEARCH / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _changed_lines(a: pathlib.Path, b: pathlib.Path):
    diff = difflib.unified_diff(a.read_text(encoding="utf-8").splitlines(),
                                b.read_text(encoding="utf-8").splitlines(), lineterm="", n=0)
    return [ln for ln in diff if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]


def test_engine_is_byte_identical():
    assert (PKG / "inversion.py").read_bytes() == (RESEARCH / "1_inversion.py").read_bytes()


@pytest.mark.parametrize("pkg_file,research_file", [
    ("online.py", "1b_online.py"), ("simulation.py", "1c_simulation_models.py")])
def test_modules_differ_only_in_engine_import(pkg_file, research_file):
    for line in _changed_lines(RESEARCH / research_file, PKG / pkg_file):
        body = line[1:].strip()
        if line.startswith("+"):
            assert body.startswith(("from . import inversion as", "# Package build")), line
        else:
            assert ("1_inversion.py" in body or "spec_from_file_location" in body
                    or "module_from_spec" in body or "exec_module" in body
                    or ".modules[" in body or body.startswith("import importlib.util")), line


def _same(a: float, b: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return a == b


@pytest.mark.parametrize("architecture", ["superconducting", "trapped_ion"])
@pytest.mark.parametrize("regime,script", [("ideal", "2_ideal_parity.py"),
                                           ("nisq", "3_benchmark.py")])
def test_run_simulation_reproduces_benchmark_instance(architecture, regime, script):
    import quantum_canary as qc
    warnings.filterwarnings("ignore")
    bench = _load(f"bench_{regime}", script)
    seed = bench.ARCH_SEED_OFFSET[architecture]
    rows = {r["parameter"]: r for r in bench.run_one_instance(architecture, 0, seed)}
    assert all(r["fit_status"] == "ok" for r in rows.values()), rows

    report = qc.run_simulation(architecture, n_qubits=1, regime=regime, seed=seed)
    q = report.qubits[0]
    for p in PARAMS:
        value_attr, sigma_attr = RESULT_ATTR[p]
        assert _same(float(getattr(q.result, value_attr)), rows[p]["canary_value"]), p
        assert _same(float(getattr(q.result, sigma_attr)), rows[p]["sigma_canary"]), p
        assert _same(float(q.truth[p]), rows[p]["true_value"]), p
