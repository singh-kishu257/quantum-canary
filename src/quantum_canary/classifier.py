"""
MLP ensemble loader and inference wrapper.

The shipped models were trained on a 21,000-sample physics-grounded
hybrid dataset (15,000 Qiskit Aer simulations + 6,000 IBM calibration
anchors) and stored as 10 separately-seeded Keras files.
"""

from __future__ import annotations

import importlib.resources as resources
import json
from pathlib import Path

import numpy as np


class MLPEnsemble:
    """
    10-seed MLP ensemble for quantum drift classification.

    Loads bundled pre-trained models and exposes a simple `predict()`
    method returning the ensemble mean and standard deviation.

    Parameters
    ----------
    models_dir : str or Path, optional
        Directory containing mlp_seed_{0..9}.keras and scaler.json.
        Defaults to the bundled `_data` directory inside the package.

    Attributes
    ----------
    n_models : int
        Number of MLP models loaded (10 by default).
    feature_names : list[str]
        Ordered feature names: ['F_bell', 'F_gate', 'F_coherence'].
    """

    FEATURE_NAMES = ["F_bell", "F_gate", "F_coherence"]

    def __init__(self, models_dir: str | Path | None = None) -> None:
        from tensorflow import keras  # imported lazily — heavy dependency

        if models_dir is None:
            with resources.as_file(
                resources.files("quantum_canary") / "_data"
            ) as p:
                models_dir = Path(p)
        else:
            models_dir = Path(models_dir)

        scaler_path = models_dir / "scaler.json"
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"scaler.json not found in {models_dir}. "
                "Run training first or check your models_dir argument."
            )
        with open(scaler_path, encoding="utf-8") as f:
            scaler_data = json.load(f)
        self._mean  = np.array(scaler_data["mean"])
        self._scale = np.array(scaler_data["scale"])

        self.models = []
        for seed in range(10):
            mp = models_dir / f"mlp_seed_{seed}.keras"
            if not mp.exists():
                raise FileNotFoundError(
                    f"Model file missing: {mp}. "
                    "All 10 seeded models are required for the ensemble."
                )
            self.models.append(keras.models.load_model(mp))

        self.n_models = len(self.models)

    def predict(self, features: np.ndarray) -> dict:
        """
        Run ensemble inference on a single fidelity triplet.

        Parameters
        ----------
        features : np.ndarray
            Shape (3,) or (n, 3). Order: [F_bell, F_gate, F_coherence].

        Returns
        -------
        dict
            Keys: 'drift_probability' (float in [0,1]),
                  'verdict' ('STABLE' or 'DRIFTED'),
                  'uncertainty' (ensemble standard deviation).
        """
        features = np.asarray(features, dtype=float)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.shape[1] != 3:
            raise ValueError(
                f"Expected 3 features, got {features.shape[1]}. "
                f"Order: {self.FEATURE_NAMES}"
            )

        X_scaled = (features - self._mean) / self._scale

        seed_preds = np.array([
            m.predict(X_scaled, verbose=0).flatten()
            for m in self.models
        ])

        prob_mean = float(seed_preds.mean(axis=0)[0])
        prob_std  = float(seed_preds.std(axis=0)[0])

        return {
            "drift_probability": prob_mean,
            "verdict":           "DRIFTED" if prob_mean >= 0.5 else "STABLE",
            "uncertainty":       prob_std,
        }