"""Drawing predictor placeholder that respects the shared contract."""

from __future__ import annotations

from typing import Any

from src.common.schemas import PredictionResult


class DrawingPredictor:
    """Temporary predictor that reserves the drawing modality interface."""

    def predict(self, payload: dict[str, Any]) -> PredictionResult:
        """Return an insufficient result until drawing is implemented."""
        return PredictionResult(
            modality="drawing",
            status="insufficient_data",
            confidence=0.0,
            warnings=["La modalité dessin n’est pas encore implémentée."],
        )
