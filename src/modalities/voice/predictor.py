"""Voice predictor placeholder that respects the shared contract."""

from __future__ import annotations

from typing import Any

from src.common.schemas import PredictionResult


class VoicePredictor:
    """Temporary predictor that reserves the voice modality interface."""

    def predict(self, payload: dict[str, Any]) -> PredictionResult:
        """Return an insufficient result until voice is implemented."""
        return PredictionResult(
            modality="voice",
            status="insufficient_data",
            confidence=0.0,
            warnings=["La modalité voix n’est pas encore implémentée."],
        )
