"""Flask routes for the drawing modality."""

from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from src.modalities.drawing.predictor import DrawingPredictor


drawing_bp = Blueprint("drawing", __name__)
predictor = DrawingPredictor()


@drawing_bp.get("/drawing")
def drawing_page():
    """Render the spiral drawing capture page."""
    return render_template("drawing.html")


@drawing_bp.post("/api/drawing/predict")
def drawing_predict():
    """Receive a base64 PNG image and return a standardized prediction."""
    payload = request.get_json(silent=True) or {}
    result = predictor.predict(payload)
    status_code = 200 if result.status in {"ok", "insufficient_data"} else 500
    return jsonify(result.to_dict()), status_code
