"""Predictor for the voice modality."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import joblib

from src.common.schemas import PredictionResult, score_to_label
from src.modalities.voice.features import FEATURE_NAMES, extract_voice_features

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "voice_parkinson_xgb.joblib"

MIN_DURATION_SEC = 3.0


class VoicePredictor:
    """Load the voice pipeline and predict a risk score from a WAV recording."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH, threshold: float = 0.50) -> None:
        self.model_path = Path(model_path)
        self.threshold = threshold
        self._artifact: dict[str, Any] | None = None

    def _load_artifact(self) -> dict[str, Any]:
        if self._artifact is None:
            self._artifact = joblib.load(self.model_path)
            self.threshold = float(self._artifact.get("threshold", self.threshold))
        return self._artifact

    def predict(self, payload: dict[str, Any]) -> PredictionResult:
        """Accept {'audio': bytes} and return a standardised PredictionResult."""
        audio_bytes: bytes | None = payload.get("audio")
        if not isinstance(audio_bytes, (bytes, bytearray)) or len(audio_bytes) == 0:
            return PredictionResult(
                modality="voice",
                status="error",
                confidence=0.0,
                warnings=["Payload invalide : le champ 'audio' doit contenir des bytes WAV non vides."],
            )

        if not self.model_path.exists():
            return PredictionResult(
                modality="voice",
                status="error",
                confidence=0.0,
                details={"model_path": str(self.model_path)},
                warnings=["Modèle voix introuvable. Vérifiez que models/voice_parkinson_xgb.joblib est présent."],
            )

        try:
            artifact = self._load_artifact()
        except Exception as exc:
            return PredictionResult(
                modality="voice",
                status="error",
                confidence=0.0,
                warnings=[f"Erreur chargement modèle voix : {exc}"],
            )

        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                tmp_path = f.name

            try:
                features = extract_voice_features(tmp_path)
            except Exception as exc:
                return PredictionResult(
                    modality="voice",
                    status="error",
                    confidence=0.0,
                    warnings=[f"Erreur extraction features audio : {exc}. Vérifiez que l'audio est au format WAV."],
                )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # Check minimum voiced content
        n_nan = sum(1 for v in features.values() if np.isnan(v))
        if n_nan > len(FEATURE_NAMES) // 2:
            return PredictionResult(
                modality="voice",
                status="insufficient_data",
                confidence=0.0,
                details={"n_nan_features": n_nan, "total_features": len(FEATURE_NAMES)},
                warnings=[
                    f"Trop de features non calculables ({n_nan}/{len(FEATURE_NAMES)}). "
                    "Enregistrez une phonation soutenue /a/ d'au moins 5 secondes."
                ],
            )

        expected_features: list[str] = artifact.get("feature_names", artifact.get("features", FEATURE_NAMES))
        feature_medians: dict[str, float] = artifact.get("feature_medians", {})

        import pandas as pd
        row = {
            name: features.get(name, feature_medians.get(name, 0.0))
            for name in expected_features
        }
        # Replace remaining NaN with stored medians or 0
        for name, val in row.items():
            if np.isnan(val):
                row[name] = float(feature_medians.get(name, 0.0))

        X = pd.DataFrame([row], columns=expected_features)

        try:
            pipeline = artifact["pipeline"]
            if hasattr(pipeline, "predict_proba"):
                score = float(pipeline.predict_proba(X)[0, 1])
            elif hasattr(pipeline, "decision_function"):
                decision = float(pipeline.decision_function(X)[0])
                score = float(1.0 / (1.0 + np.exp(-decision)))
            else:
                score = float(pipeline.predict(X)[0])
        except Exception as exc:
            return PredictionResult(
                modality="voice",
                status="error",
                confidence=0.0,
                warnings=[f"Erreur prédiction voix : {exc}"],
            )

        confidence = max(0.0, min(1.0, 1.0 - n_nan / len(FEATURE_NAMES)))
        warnings: list[str] = []
        if n_nan > 0:
            warnings.append(f"{n_nan} feature(s) non calculable(s) — score moins fiable.")

        return PredictionResult(
            modality="voice",
            status="ok",
            score=score,
            confidence=confidence,
            label=score_to_label(score, high_threshold=self.threshold),
            details={
                "n_nan_features": n_nan,
                "threshold": self.threshold,
                "model_path": str(self.model_path),
            },
            warnings=warnings,
        )
