"""
arx_model.py
------------
Defines the ReducedRankRidge multi-output ARX model used to train
and load the joint SAF furnace simulator.

ReducedRankRidge fits a Ridge regression coefficient matrix and then
truncates it to a low rank. All 10 output signals (R, kA, arc
reactance per electrode, plus transformer voltage) share an r-dimensional
latent space. This captures the common furnace physics that couples all
electrodes through the burden and electrical circuit.

Both the Ridge regularisation strength (alpha) and the latent rank (r) are
selected by cross-validation on the training data.

The interface is identical to sklearn's Ridge so the existing ARX bundle
loading and evaluation code works without any changes.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold


class ReducedRankRidge:
    """
    Multi-output reduced-rank ridge regression.

    Parameters
    ----------
    alphas : iterable
        Grid of Ridge regularisation strengths to search over.
    ranks  : iterable of int
        Candidate latent ranks to evaluate by cross-validation.
    cv     : int
        Number of cross-validation folds for rank selection.
    """

    def __init__(
        self,
        alphas=np.logspace(-4, 4, 20),
        ranks=(2, 3, 4, 5, 6, 8, 10),
        cv: int = 5,
    ):
        self.alphas = list(alphas)
        self.ranks  = list(ranks)
        self.cv     = cv
        self.coef_:       np.ndarray | None = None
        self.intercept_:  np.ndarray | None = None
        self.alpha_:      float | None      = None
        self.rank_:       int   | None      = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "ReducedRankRidge":
        """
        Fit the model on training data X (n samples x p features) and
        Y (n samples x q outputs).

        Step 1: Select alpha using a chronological 80/20 split
                X.T @ X is computed once (p x p) to keep cost low
        Step 2: Fit full-rank Ridge on all training data with best alpha.
        Step 3: Cross-validate to select the best latent rank r
        Step 4: Compute the final low-rank coefficient matrix
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        n, p = X.shape
        q    = Y.shape[1]
        I_p  = np.eye(p)

        # Step 1: alpha selection via 80/20 chronological split.
        # Computing X.T @ X once (p x p) avoids running per-output RidgeCV on n rows.
        n_tr = int(0.8 * n)
        X_tr, X_va = X[:n_tr], X[n_tr:]
        Y_tr, Y_va = Y[:n_tr], Y[n_tr:]
        G_tr = X_tr.T @ X_tr
        H_tr = X_tr.T @ Y_tr
        best_alpha, best_loss = self.alphas[0], np.inf
        for alpha in self.alphas:
            B    = np.linalg.solve(G_tr + alpha * I_p, H_tr)
            loss = np.mean(np.abs(Y_va - X_va @ B))
            if loss < best_loss:
                best_loss, best_alpha = loss, alpha
        self.alpha_ = float(best_alpha)
        del X_tr, X_va, Y_tr, Y_va, G_tr, H_tr

        # Step 2: full-rank Ridge on the complete training set
        G_full = X.T @ X
        H_full = X.T @ Y
        B_full = np.linalg.solve(G_full + self.alpha_ * I_p, H_full)

        # Step 3: cross-validate rank via KFold.
        kf = KFold(n_splits=self.cv, shuffle=False)
        rank_maes: dict[int, list] = {r: [] for r in self.ranks}
        for tr_idx, va_idx in kf.split(X):
            Xtr, Xva = X[tr_idx], X[va_idx]
            Ytr, Yva = Y[tr_idx], Y[va_idx]
            G_fold   = Xtr.T @ Xtr
            H_fold   = Xtr.T @ Ytr
            B_fold   = np.linalg.solve(G_fold + self.alpha_ * I_p, H_fold)
            U, s, Vt = np.linalg.svd(B_fold, full_matrices=False)
            XvaU     = Xva @ U
            for r in self.ranks:
                r_eff = min(r, p, q)
                Yhat  = (XvaU[:, :r_eff] * s[:r_eff]) @ Vt[:r_eff, :]
                rank_maes[r].append(np.mean(np.abs(Yva - Yhat)))
            del Xtr, Xva, Ytr, Yva, G_fold, H_fold, B_fold, U, s, Vt, XvaU

        rank_scores = {r: float(np.mean(v)) for r, v in rank_maes.items()}
        self.rank_  = min(rank_scores, key=rank_scores.get)
        print(f"    [RRR]  alpha={self.alpha_:.3g}   rank CV scores: "
              + "  ".join(f"r={r}:{v:.5f}" for r, v in rank_scores.items())
              + f"  ->  best_rank={self.rank_}")

        # Step 4: final coefficient matrix at best alpha + best rank
        U, s, Vt    = np.linalg.svd(B_full, full_matrices=False)
        r_use       = min(self.rank_, len(s))
        self.coef_      = (U[:, :r_use] * s[:r_use]) @ Vt[:r_use, :]
        self.intercept_ = np.zeros(q, dtype=np.float64)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted outputs for input matrix X (n x p)."""
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_
