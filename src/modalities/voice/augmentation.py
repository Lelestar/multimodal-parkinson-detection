"""Augmentation tabulaire pour la modalité voix (hors audio brut)."""

from __future__ import annotations

import numpy as np


def apply_smote_safe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Suréchantillonne la classe minoritaire (SMOTE) uniquement sur un jeu d'apprentissage.

    Retourne (X, y, ok). Si une seule classe, imblearn absent, ou échec SMOTE, renvoie l'entrée inchangée et ok=False.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        return x_train, y_train, False
    if len(np.unique(y_train)) < 2:
        return x_train, y_train, False
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    minority_count = min(n_pos, n_neg)
    if minority_count < 2:
        return x_train, y_train, False
    k_neighbors = max(1, min(5, minority_count - 1))
    try:
        x_res, y_res = SMOTE(random_state=random_state, k_neighbors=k_neighbors).fit_resample(x_train, y_train)
        return x_res.astype(np.float64), y_res.astype(np.int32), True
    except ValueError:
        return x_train, y_train, False
