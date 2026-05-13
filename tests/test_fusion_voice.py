"""Tests de fusion avec résultat voix simulé."""

from __future__ import annotations

from src.common.fusion import late_fusion
from src.common.schemas import PredictionResult


def test_late_fusion_ignores_non_ok_voice():
    voice_bad = PredictionResult(modality="voice", status="error", confidence=0.0, warnings=["x"])
    kb = PredictionResult(modality="keyboard", status="ok", score=0.6, confidence=0.8, label="moderate")
    out = late_fusion([voice_bad, kb])
    assert out.status == "ok"
    assert "keyboard" in out.used_modalities
    assert "voice" in out.ignored_modalities


def test_late_fusion_keyboard_and_voice_ok():
    voice = PredictionResult(modality="voice", status="ok", score=0.7, confidence=0.85, label="elevated")
    kb = PredictionResult(modality="keyboard", status="ok", score=0.5, confidence=0.9, label="moderate")
    out = late_fusion([kb, voice])
    assert out.status == "ok"
    assert set(out.used_modalities) == {"keyboard", "voice"}
    assert 0.0 <= (out.score or 0) <= 1.0
