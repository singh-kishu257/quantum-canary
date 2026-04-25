"""
Statistical baselines for quantum drift detection.

Two non-ML methods are provided for comparison and reproducibility:

    StandardThreshold — Per-feature majority-vote threshold using
                        training-set means as cutoffs. 

    HotellingT2       — Multivariate statistical process control
                        (Hotelling 1947). Mahalanobis distance from
                        the healthy training centroid using a
                        regularised covariance matrix.

These classes exist to support reproducibility of the comparison
results published alongside Quantum Canary. They are not exposed in
the top-level public API but can be imported directly from
`quantum_canary.baselines` for verification.

Reference
---------
Hotelling, H. (1947). Multivariate Quality Control.
In Techniques of Statistical Analysis. McGraw-Hill.
"""

from __future__ import annotations
import numpy as np


class StandardThreshold:
    """
    Per-feature majority-vote threshold classifier.

    For each input feature, the class learns the training-set mean as
    a decision boundary. A test point votes "drifted" on a feature if
    it falls below that mean. Final prediction is the majority across
    the three votes.

    Notes
    -----
    Output scores are integer-valued (0, 1, 2, or 3 votes), so the
    resulting ROC curve has at most 4 distinct points.
    """

    def __init__(self) -> None:
        self.feature_means: np.ndarray | None = None

    def fit(self, X_train: np.ndarray) -> "StandardThreshold":
        """Compute per-feature means from the training distribution."""
        self.feature_means = np.asarray(X_train).mean(axis=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary drift labels: 1 = drifted, 0 = stable."""
        if self.feature_means is None:
            raise RuntimeError("Call fit() before predict().")
        votes = (np.asarray(X) < self.feature_means).astype(int)
        return (votes.sum(axis=1) >= 2).astype(int)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Continuous drift score = number of features voting drifted (0-3)."""
        if self.feature_means is None:
            raise RuntimeError("Call fit() before score().")
        votes = (np.asarray(X) < self.feature_means).astype(int)
        return votes.sum(axis=1)


class HotellingT2:
    """
    Hotelling's T² multivariate statistical process control.

    Models the healthy training distribution as a multivariate Gaussian
    with mean μ and covariance Σ. Each test point is scored by its
    squared Mahalanobis distance from μ:

        T²(x) = (x - μ)ᵀ · Σ⁻¹ · (x - μ)

    A small ridge regularisation (default 1e-6) is added to Σ before
    inversion to handle correlated features.

    Reference
    ---------
    Hotelling, H. (1947). Multivariate Quality Control.
    In Techniques of Statistical Analysis. McGraw-Hill.

    Parameters
    ----------
    threshold : float, optional
        T² decision cutoff. If None, must be set via `set_threshold()`
        after tuning on validation data.
    epsilon : float, optional
        Ridge regularisation added to the covariance diagonal before
        inversion (default 1e-6).
    """

    def __init__(self,
                 threshold: float | None = None,
                 epsilon:   float = 1e-6) -> None:
        self.threshold       = threshold
        self.epsilon         = epsilon
        self.mu_vec:    np.ndarray | None = None
        self.cov_inv:   np.ndarray | None = None
        self._fitted = False

    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray | None = None) -> "HotellingT2":
        """
        Fit the healthy distribution from stable training samples only.

        Parameters
        ----------
        X_train : np.ndarray
            Shape (n, n_features).
        y_train : np.ndarray, optional
            Binary labels. If provided, only y_train==0 (stable) samples
            are used to estimate μ and Σ. If None, all samples are used.
        """
        X = np.asarray(X_train, dtype=float)
        if y_train is not None:
            X = X[np.asarray(y_train) == 0]

        if len(X) < 2:
            raise ValueError("Need at least 2 stable samples to fit covariance.")

        self.mu_vec = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        cov_reg = cov + self.epsilon * np.eye(cov.shape[0])
        self.cov_inv = np.linalg.inv(cov_reg)
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return the T² distance for each sample."""
        if not self._fitted:
            raise RuntimeError("Call fit() before score().")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        d = X - self.mu_vec
        return np.einsum("ij,jk,ik->i", d, self.cov_inv, d)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary drift labels: 1 = drifted, 0 = stable."""
        if self.threshold is None:
            raise RuntimeError(
                "Decision threshold not set. Call set_threshold() "
                "after tuning on validation data."
            )
        return (self.score(X) > self.threshold).astype(int)

    def set_threshold(self, h: float) -> None:
        """Set the T² decision cutoff (typically tuned on validation F1)."""
        self.threshold = float(h)