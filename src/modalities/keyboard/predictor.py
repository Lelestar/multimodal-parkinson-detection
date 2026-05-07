"""Predictor for the keyboard dynamics modality."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.common.schemas import PredictionResult, score_to_label
from src.modalities.keyboard.features import build_feature_table, events_to_keystrokes


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "keyboard_dynamics_neuroqwerty_v2_pipeline.joblib"


class KeyboardPredictor:
    """Load the keyboard v2 pipeline and predict an exploratory session risk."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        threshold: float = 0.58,
        min_segment_len: int = 300,
        window_size: int = 300,
        stride: int = 150,
    ) -> None:
        """Configure the model path and segmentation parameters."""
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.min_segment_len = min_segment_len
        self.window_size = window_size
        self.stride = stride
        self._artifact: dict[str, Any] | None = None

    def _load_artifact(self) -> dict[str, Any]:
        """Load the joblib artifact once and read its decision threshold."""
        if self._artifact is None:
            self._artifact = joblib.load(self.model_path)
            self.threshold = float(self._artifact.get("threshold", self.threshold))
        return self._artifact

    def predict(self, payload: dict[str, Any]) -> PredictionResult:
        """Convert a browser payload into a standard `PredictionResult`."""
        events = payload.get("events", [])
        if not isinstance(events, list):
            return PredictionResult(
                modality="keyboard",
                status="error",
                confidence=0.0,
                warnings=["Payload invalide : le champ events doit être une liste."],
            )

        keystrokes = events_to_keystrokes(events)
        features = build_feature_table(
            keystrokes,
            window_size=self.window_size,
            stride=self.stride,
            min_segment_len=self.min_segment_len,
        )
        valid_keystrokes = len(keystrokes)
        if features.empty:
            return PredictionResult(
                modality="keyboard",
                status="insufficient_data",
                confidence=0.0,
                details={
                    "valid_keystrokes": valid_keystrokes,
                    "required_keystrokes": self.min_segment_len,
                },
                warnings=[
                    f"Données insuffisantes : {valid_keystrokes} frappes valides, minimum {self.min_segment_len}."
                ],
            )

        if not self.model_path.exists():
            return PredictionResult(
                modality="keyboard",
                status="error",
                confidence=0.0,
                details={"model_path": str(self.model_path)},
                warnings=["Modèle clavier introuvable. Voir models/README.md."],
            )

        try:
            artifact = self._load_artifact()
            pipeline = artifact["pipeline"]
            expected_features = artifact.get("features", list(features.columns))
            model_input = features.reindex(columns=expected_features)
            if hasattr(pipeline, "predict_proba"):
                segment_scores = pipeline.predict_proba(model_input)[:, 1]
            elif hasattr(pipeline, "decision_function"):
                decision = pipeline.decision_function(model_input)
                segment_scores = 1.0 / (1.0 + np.exp(-decision))
            else:
                segment_scores = pipeline.predict(model_input).astype(float)
        except Exception as exc:
            return PredictionResult(
                modality="keyboard",
                status="error",
                confidence=0.0,
                details={"model_path": str(self.model_path), "valid_keystrokes": valid_keystrokes},
                warnings=[f"Erreur pendant la prédiction clavier : {exc}"],
            )

        score = float(np.mean(segment_scores))
        confidence = min(1.0, valid_keystrokes / float(self.window_size))
        warnings: list[str] = []
        if valid_keystrokes < self.window_size:
            warnings.append(
                f"Session courte : {valid_keystrokes} frappes valides. Le modèle est plus stable autour de {self.window_size} frappes."
            )

        return PredictionResult(
            modality="keyboard",
            status="ok",
            score=score,
            confidence=confidence,
            label=score_to_label(score, high_threshold=self.threshold),
            details={
                "valid_keystrokes": valid_keystrokes,
                "n_segments": int(len(features)),
                "segment_scores": [float(value) for value in segment_scores],
                "threshold": self.threshold,
                "model_path": str(self.model_path),
            },
            warnings=warnings,
        )
