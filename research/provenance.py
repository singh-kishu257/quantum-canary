"""Records which code and packages produced a data file, in a .meta.json next to it."""
from __future__ import annotations

import json
import pathlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata

PACKAGES = ("quantum-canary", "qiskit", "qiskit-aer", "qiskit-ibm-runtime",
            "qiskit-experiments", "qiskit-ionq", "numpy", "scipy")
ROOT = pathlib.Path(__file__).resolve().parent.parent


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                             timeout=10, check=True)
    except Exception:
        return None
    return out.stdout.strip()


def provenance():
    """Commit, uncommitted code changes, Python and package versions, and the command."""
    import quantum_canary
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    status = _git("status", "--porcelain", "--", "src", "research/*.py")
    return {"written_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git("rev-parse", "HEAD"),
            "uncommitted_code_changes": None if status is None else bool(status),
            "quantum_canary_imported": quantum_canary.__version__,
            "python": platform.python_version(), "packages": versions,
            "command": sys.argv}


def write_meta(data_path) -> pathlib.Path:
    """Write provenance() to <data_path>.meta.json and return that path."""
    path = pathlib.Path(f"{data_path}.meta.json")
    path.write_text(json.dumps(provenance(), indent=2), encoding="utf-8")
    return path
