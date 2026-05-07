"""Registry for modalities available in the Flask application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


PredictorFactory = Callable[[], object]


@dataclass(frozen=True)
class ModalityRegistration:
    """Metadata needed to display or initialize one modality."""

    name: str
    display_name: str
    route: str
    predictor_factory: PredictorFactory | None = None


class ModalityRegistry:
    """Small in-memory registry that centralizes available modalities."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._modalities: dict[str, ModalityRegistration] = {}

    def register(self, registration: ModalityRegistration) -> None:
        """Add or replace a modality in the registry."""
        self._modalities[registration.name] = registration

    def list(self) -> list[ModalityRegistration]:
        """Return registered modalities in insertion order."""
        return list(self._modalities.values())

    def get(self, name: str) -> ModalityRegistration | None:
        """Return one modality by its technical name."""
        return self._modalities.get(name)


registry = ModalityRegistry()
