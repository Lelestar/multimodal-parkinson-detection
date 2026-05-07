"""Shared schemas for modality predictions and multimodal fusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


PredictionStatus = Literal["ok", "insufficient_data", "error"]
RiskLabel = Literal["low", "moderate", "elevated"]


@dataclass
class PredictionResult:
    """Standard result returned by one prediction modality."""

    modality: str
    status: PredictionStatus
    score: float | None = None
    confidence: float | None = None
    label: RiskLabel | None = None
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class FusionResult:
    """Standard result returned by late modality fusion."""

    status: PredictionStatus
    score: float | None = None
    confidence: float | None = None
    label: RiskLabel | None = None
    used_modalities: list[str] = field(default_factory=list)
    ignored_modalities: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the fusion result to a JSON-serializable dictionary."""
        return asdict(self)


def score_to_label(score: float | None, low_threshold: float = 0.35, high_threshold: float = 0.58) -> RiskLabel | None:
    """Convert a continuous 0-1 score into a readable risk level."""
    if score is None:
        return None
    if score < low_threshold:
        return "low"
    if score < high_threshold:
        return "moderate"
    return "elevated"
