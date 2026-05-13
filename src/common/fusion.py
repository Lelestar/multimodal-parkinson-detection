"""Late fusion for scores produced by the available modalities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from src.common.schemas import FusionResult, PredictionResult, score_to_label


DEFAULT_WEIGHTS = {
    "keyboard": 1.0,
    "voice": 1.0,
    "drawing": 1.0,
}


def _as_prediction_result(raw: PredictionResult | Mapping) -> PredictionResult:
    """Normalize a dictionary or object into a `PredictionResult`."""
    if isinstance(raw, PredictionResult):
        return raw
    return PredictionResult(
        modality=str(raw.get("modality", "unknown")),
        status=raw.get("status", "error"),
        score=raw.get("score"),
        confidence=raw.get("confidence"),
        label=raw.get("label"),
        details=dict(raw.get("details", {})),
        warnings=list(raw.get("warnings", [])),
    )


def late_fusion(
    predictions: Sequence[PredictionResult | Mapping],
    weights: Mapping[str, float] | None = None,
) -> FusionResult:
    """Combine valid predictions with a confidence-weighted average."""
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    weighted_sum = 0.0
    weight_sum = 0.0
    used: list[str] = []
    ignored: list[str] = []
    warnings: list[str] = []
    per_modality: dict[str, dict] = {}

    for raw_prediction in predictions:
        prediction = _as_prediction_result(raw_prediction)
        per_modality[prediction.modality] = prediction.to_dict()
        if prediction.status != "ok" or prediction.score is None:
            ignored.append(prediction.modality)
            warnings.extend(prediction.warnings)
            continue

        confidence = prediction.confidence if prediction.confidence is not None else 1.0
        confidence = max(0.0, min(1.0, float(confidence)))
        modality_weight = float(weights.get(prediction.modality, 1.0))
        effective_weight = modality_weight * confidence
        if effective_weight <= 0:
            ignored.append(prediction.modality)
            warnings.append(f"{prediction.modality}: confiance ou poids nul.")
            continue

        weighted_sum += float(prediction.score) * effective_weight
        weight_sum += effective_weight
        used.append(prediction.modality)

    if not used or weight_sum <= 0:
        return FusionResult(
            status="insufficient_data",
            ignored_modalities=ignored,
            details={"modalities": per_modality},
            warnings=warnings or ["Aucune modalité exploitable pour la fusion."],
        )

    score = weighted_sum / weight_sum
    confidence = min(1.0, weight_sum / max(1.0, sum(weights.get(modality, 1.0) for modality in used)))
    return FusionResult(
        status="ok",
        score=score,
        confidence=confidence,
        label=score_to_label(score),
        used_modalities=used,
        ignored_modalities=ignored,
        details={"modalities": per_modality, "weights": weights},
        warnings=warnings,
    )
