"""Prédiction voix à partir de biomarqueurs tabulaires (dataset Oxford / UCI)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.common.schemas import PredictionResult, score_to_label
from src.modalities.voice.features import VOICE_FEATURE_COLUMNS, build_voice_feature_row


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "voice_parkinsons_tabular.joblib"


def _technical_confidence(*, completeness: float, extra_penalty: float = 1.0) -> float:
    """
    Confiance technique : jamais 1.0 (prudence pédagogique / non-certitude médicale).

    completeness vaut 1.0 lorsque toutes les features attendues sont présentes et valides.
    """
    base = 0.55 + 0.34 * float(np.clip(completeness, 0.0, 1.0))
    conf = base * float(np.clip(extra_penalty, 0.5, 1.0))
    return float(min(0.9, max(0.38, conf)))


class VoicePredictor:
    """Charge le pipeline voix tabulaire et renvoie un PredictionResult standard."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH, threshold: float = 0.5) -> None:
        self.model_path = Path(model_path)
        self.threshold = threshold
        self._artifact: dict[str, Any] | None = None

    def _load_artifact(self) -> dict[str, Any]:
        """Charge l'artefact joblib une seule fois (cache en mémoire)."""
        if self._artifact is None:
            self._artifact = joblib.load(self.model_path)
            self.threshold = float(self._artifact.get("threshold", self.threshold))
        return self._artifact

    def predict(self, payload: dict[str, Any]) -> PredictionResult:
        """Interprète un JSON de biomarqueurs et renvoie un résultat fusionnable."""
        if not isinstance(payload, dict):
            return PredictionResult(
                modality="voice",
                status="error",
                confidence=0.0,
                warnings=["Payload invalide : un objet JSON est attendu."],
            )

        if not payload or all(k in ("session_id", "name") for k in payload):
            return PredictionResult(
                modality="voice",
                status="error",
                confidence=0.0,
                details={"error": "Payload vide ou sans mesures vocales."},
                warnings=["La prédiction voix n'a pas pu être calculée : aucune mesure fournie."],
            )

        if not self.model_path.exists():
            return PredictionResult(
                modality="voice",
                status="error",
                confidence=0.0,
                details={"model_path": str(self.model_path)},
                warnings=[
                    "Modèle voix introuvable. Depuis Voice_Casin_Wayne_Oxford_AI : "
                    "python scripts/train_voice_tabular.py (voir README)."
                ],
            )

        try:
            artifact = self._load_artifact()
        except Exception as exc:
            return PredictionResult(
                modality="voice",
                status="error",
                confidence=0.0,
                details={"model_path": str(self.model_path)},
                warnings=[f"Erreur pendant le chargement du modèle voix : {exc}"],
            )

        expected: list[str] = list(artifact.get("features", list(VOICE_FEATURE_COLUMNS)))
        row, missing, parse_warnings = build_voice_feature_row(payload, expected_columns=expected)
        if row is None:
            if missing:
                return PredictionResult(
                    modality="voice",
                    status="error",
                    confidence=0.0,
                    details={"error": f"Features manquantes : {', '.join(missing)}"},
                    warnings=["La prédiction voix n'a pas pu être calculée."] + parse_warnings,
                )
            return PredictionResult(
                modality="voice",
                status="error",
                confidence=0.0,
                details={"error": " ".join(parse_warnings) if parse_warnings else "Données invalides."},
                warnings=["La prédiction voix n'a pas pu être calculée."] + parse_warnings,
            )

        try:
            pipeline = artifact["pipeline"]
            if hasattr(pipeline, "predict_proba"):
                proba = float(pipeline.predict_proba(row)[0, 1])
            elif hasattr(pipeline, "decision_function"):
                decision = float(pipeline.decision_function(row)[0])
                proba = float(1.0 / (1.0 + np.exp(-decision)))
            else:
                proba = float(np.clip(float(pipeline.predict(row)[0]), 0.0, 1.0))
            score = float(np.clip(proba, 0.0, 1.0))
        except Exception as exc:
            return PredictionResult(
                modality="voice",
                status="error",
                confidence=0.0,
                details={"model_path": str(self.model_path)},
                warnings=[f"Erreur pendant la prédiction voix : {exc}"],
            )

        warnings = list(parse_warnings)
        confidence = _technical_confidence(completeness=1.0)

        return PredictionResult(
            modality="voice",
            status="ok",
            score=score,
            confidence=confidence,
            label=score_to_label(score, high_threshold=self.threshold),
            details={
                "threshold": self.threshold,
                "model_path": str(self.model_path),
                "model_name": artifact.get("model"),
                "feature_count": len(expected),
            },
            warnings=warnings,
        )
