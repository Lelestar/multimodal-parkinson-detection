"""Predictor for the drawing modality."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog, local_binary_pattern
from skimage.filters import threshold_otsu

from src.common.schemas import PredictionResult, score_to_label


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "drawing_spiral_v2_hog_lbp_pipeline.joblib"

# Décalage systématique mesuré entre canvas HTML et scans papier d'entraînement.
# Un canvas vide (blanc pur) obtient ~0.561 ; une spirale propre ~0.540.
# Les HC du dataset test obtiennent en moyenne 0.326.
# Offset 0.20 : ramène une spirale propre à ~0.34 (low) et préserve
# la sensibilité pour les tracés avec tremblements réels (raw ~0.70-0.85 → elevated).
_CANVAS_CALIBRATION_OFFSET: float = 0.20


class DrawingPredictor:
    """Load a drawing pipeline and predict from a spiral PNG image.

    Compatible with two artifact formats:
    - v1 (drawing_spiral_v1_pipeline.joblib): HOG only, grayscale inversion preprocessing.
    - v2 (drawing_spiral_v2_hog_lbp_pipeline.joblib): HOG + LBP, Otsu binarization.
      The preprocessing variant is detected from artifact['preprocessing'].
    """

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
    def _preprocess(pil_img: Image.Image, img_size: tuple, use_otsu: bool) -> np.ndarray:
        """Grayscale → resize → binarize (Otsu) or invert.

        - use_otsu=True  (v2): Otsu threshold → binary {0,1}, trait clair.
          Supprime la texture papier des scans pour correspondre au canvas HTML.
        - use_otsu=False (v1): simple inversion float32, rétro-compatible.
        """
        arr = np.array(pil_img.convert("L").resize(img_size, Image.LANCZOS), dtype=np.float32) / 255.0
        if use_otsu:
            thresh = threshold_otsu(arr)
            return (arr < thresh).astype(np.float32)  # trait sombre → 1.0
        return 1.0 - arr

    @staticmethod
    def _extract_features(arr: np.ndarray, hog_params: dict, lbp_params: dict | None) -> np.ndarray:
        """HOG (+ LBP si lbp_params fourni) → vecteur 1-D concaténé."""
        hog_features = hog(
            arr,
            orientations=hog_params["orientations"],
            pixels_per_cell=tuple(hog_params["pixels_per_cell"]),
            cells_per_block=tuple(hog_params["cells_per_block"]),
            block_norm="L2-Hys",
            feature_vector=True,
        ).astype(np.float32)

        if lbp_params is None:
            return hog_features

        lbp_input = np.rint(arr * 255).clip(0, 255).astype(np.uint8)
        lbp = local_binary_pattern(
            lbp_input,
            P=lbp_params["n_points"],
            R=lbp_params["radius"],
            method="uniform",
        )
        n_bins = lbp_params["n_bins"]
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-8)
        return np.concatenate([hog_features, hist])

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
                warnings=["Modèle dessin introuvable. Exécutez notebooks/drawing/drawing_model_spiral_dataset.ipynb."],
            )

        try:
            artifact = self._load_artifact()
            pipeline = artifact["pipeline"]
            hog_params = artifact["hog_params"]
            img_size = tuple(hog_params["img_size"])

            use_otsu = artifact.get("preprocessing") == "otsu_binarization"
            lbp_params = artifact.get("lbp_params")  # None for v1 artifacts

            arr = self._preprocess(pil_img, img_size, use_otsu=use_otsu)
            features = self._extract_features(arr, hog_params, lbp_params).reshape(1, -1)
            raw_score = float(pipeline.predict_proba(features)[0, 1])

            # Correction du domain gap canvas → papier :
            # Le modèle a été entraîné sur des scans papier. Un canvas blanc pur
            # score ~0.539 à cause de la différence de domaine. On soustrait
            # l'offset mesuré (0.27) pour ramener les scores à la plage du dataset.
            score = float(np.clip(raw_score - _CANVAS_CALIBRATION_OFFSET, 0.0, 1.0))
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
            details={
                "threshold": self.threshold,
                "model": artifact.get("model", "unknown"),
                "preprocessing": artifact.get("preprocessing", "inversion"),
                "raw_score": round(raw_score, 4),
                "canvas_calibration_offset": _CANVAS_CALIBRATION_OFFSET,
            },
        )
