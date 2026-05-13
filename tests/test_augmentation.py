"""Tests augmentation SMOTE pour la modalité voix."""

from __future__ import annotations

import numpy as np
import pytest

from src.modalities.voice.augmentation import apply_smote_safe


def test_smote_skipped_single_class():
    x = np.array([[0.0, 1.0], [0.1, 1.1]], dtype=np.float64)
    y = np.array([1, 1], dtype=np.int32)
    xr, yr, ok = apply_smote_safe(x, y)
    assert ok is False
    assert xr.shape == x.shape


def test_smote_balances_minority_when_imblearn_available():
    pytest.importorskip("imblearn")
    rng = np.random.default_rng(0)
    x = np.vstack([rng.standard_normal((20, 4)), rng.standard_normal((5, 4)) + 2.0])
    y = np.array([0] * 20 + [1] * 5, dtype=np.int32)
    xr, yr, ok = apply_smote_safe(x, y, random_state=0)
    assert ok is True
    assert len(xr) > len(x)
    assert np.sum(yr == 0) == np.sum(yr == 1)
