"""Predictor for the drawing modality."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog

from src.common.schemas import PredictionResult, score_to_label


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "drawing_spiral_v1_pipeline.joblib"


class DrawingPredictor:
    """Load the drawing HOG+GBC pipeline and predict from a spiral PNG image."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        threshold: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path)
        self.threshold = threshold
        self._artifact: dict[str, Any] | None = None

    def _load_artifact(self) -> dict[str, Any]:
        if self._artifact is None:
            self._artifact = joblib.load(self.model_path)
            self.threshold = float(self._artifact.get("threshold", self.threshold))
        return self._artifact

    @staticmethod
    def _preprocess(pil_img: Image.Image, img_size: tuple, noise_std: float = 0.03) -> np.ndarray:
        """Convert to grayscale, resize, invert, and add light training-like noise."""
        arr = np.array(pil_img.convert("L").resize(img_size, Image.LANCZOS), dtype=np.float32) / 255.0
        arr = 1.0 - arr
        # Light Gaussian noise reduces the domain shift between JPEG photos and clean canvas PNGs.
        rng = np.random.default_rng(seed=42)
        arr = arr + rng.normal(0.0, noise_std, arr.shape).astype(np.float32)
        return np.clip(arr, 0.0, 1.0)

    def predict(self, payload: dict[str, Any]) -> PredictionResult:
        """Convert a base64 PNG payload into a standard PredictionResult."""
        image_b64 = payload.get("image_b64", "")
        if not image_b64:
            return PredictionResult(
                modality="drawing",
                status="insufficient_data",
                confidence=0.0,
                warnings=["Aucune image reçue."],
            )

        # Strip the data-URL prefix.
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]

        try:
            png_bytes = base64.b64decode(image_b64)
            pil_img = Image.open(io.BytesIO(png_bytes))
        except Exception as exc:
            return PredictionResult(
                modality="drawing",
                status="error",
                confidence=0.0,
                warnings=[f"Impossible de décoder l'image : {exc}"],
            )

        if not self.model_path.exists():
            return PredictionResult(
                modality="drawing",
                status="error",
                confidence=0.0,
                details={"model_path": str(self.model_path)},
                warnings=["Modèle dessin introuvable. Exécutez notebooks/drawing/drawing_model_training.ipynb."],
            )

        try:
            artifact = self._load_artifact()
            pipeline = artifact["pipeline"]
            hog_params = artifact["hog_params"]
            img_size = tuple(hog_params["img_size"])

            arr = self._preprocess(pil_img, img_size)
            features = hog(
                arr,
                orientations=hog_params["orientations"],
                pixels_per_cell=tuple(hog_params["pixels_per_cell"]),
                cells_per_block=tuple(hog_params["cells_per_block"]),
                block_norm="L2-Hys",
                feature_vector=True,
            ).astype(np.float32).reshape(1, -1)

            score = float(pipeline.predict_proba(features)[0, 1])
        except Exception as exc:
            return PredictionResult(
                modality="drawing",
                status="error",
                confidence=0.0,
                warnings=[f"Erreur pendant la prédiction dessin : {exc}"],
            )

        return PredictionResult(
            modality="drawing",
            status="ok",
            score=score,
            confidence=1.0,
            label=score_to_label(score, high_threshold=self.threshold),
            details={"threshold": self.threshold, "model_path": str(self.model_path)},
        )
