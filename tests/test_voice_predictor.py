"""Tests du prédicteur voix et du format PredictionResult."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.modalities.voice.predictor import VoicePredictor, DEFAULT_MODEL_PATH
from src.modalities.voice.features import VOICE_FEATURE_COLUMNS, build_voice_feature_row


def _full_payload() -> dict:
    return {
        "features": {
            "MDVP:Fo(Hz)": 197.076,
            "MDVP:Fhi(Hz)": 206.896,
            "MDVP:Flo(Hz)": 192.055,
            "MDVP:Jitter(%)": 0.00289,
            "MDVP:Jitter(Abs)": 0.00001,
            "MDVP:RAP": 0.00166,
            "MDVP:PPQ": 0.00168,
            "Jitter:DDP": 0.00498,
            "MDVP:Shimmer": 0.01098,
            "MDVP:Shimmer(dB)": 0.097,
            "Shimmer:APQ3": 0.00563,
            "Shimmer:APQ5": 0.0068,
            "MDVP:APQ": 0.00802,
            "Shimmer:DDA": 0.01689,
            "NHR": 0.00339,
            "HNR": 26.775,
            "RPDE": 0.422229,
            "DFA": 0.741367,
            "spread1": -7.3483,
            "spread2": 0.177551,
            "D2": 1.743867,
            "PPE": 0.085569,
        }
    }


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="Modèle voice absent")
def test_voice_model_file_exists():
    assert DEFAULT_MODEL_PATH.is_file()


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="Modèle voice absent")
def test_predict_full_payload_ok():
    pred = VoicePredictor()
    r = pred.predict(_full_payload())
    assert r.modality == "voice"
    assert r.status == "ok"
    assert r.score is not None
    assert 0.0 <= r.score <= 1.0
    assert r.confidence is not None
    assert r.confidence < 1.0
    assert r.label in ("low", "moderate", "elevated")
    d = r.to_dict()
    assert d["modality"] == "voice"
    assert "warnings" in d


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="Modèle voice absent")
def test_predict_incomplete_payload_error():
    pred = VoicePredictor()
    r = pred.predict({"features": {"HNR": 21.0}})
    assert r.status == "error"
    assert r.score is None
    assert r.label is None


def test_build_voice_feature_row_missing():
    row, missing, _ = build_voice_feature_row({}, expected_columns=list(VOICE_FEATURE_COLUMNS))
    assert row is None
    assert len(missing) == len(VOICE_FEATURE_COLUMNS)


def test_predict_empty_payload():
    p = VoicePredictor(model_path=Path("/nonexistent/voice.joblib"))
    r = p.predict({})
    assert r.status == "error"


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="Modèle voice absent")
def test_predict_payload_not_dict():
    pred = VoicePredictor()
    r = pred.predict("bad")  # type: ignore[arg-type]
    assert r.status == "error"
