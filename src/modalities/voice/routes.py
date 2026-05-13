"""Routes Flask pour la modalité voix (biomarqueurs tabulaires).

Ce blueprint peut être enregistré par n'importe quelle application Flask qui ajoute
``Voice_Casin_Wayne_Oxford_AI`` au ``sys.path`` (voir README du dossier).
"""

from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from src.modalities.voice.features import VOICE_FEATURE_COLUMNS
from src.modalities.voice.predictor import VoicePredictor


voice_bp = Blueprint("voice", __name__)
predictor = VoicePredictor()


@voice_bp.get("/voice")
def voice_page():
    """Page de saisie des biomarqueurs vocaux (pas d'audio brut dans cette version)."""
    return render_template("voice.html", feature_columns=list(VOICE_FEATURE_COLUMNS))


@voice_bp.post("/api/voice/predict")
def voice_predict():
    """Reçoit les mesures vocales et renvoie un PredictionResult JSON."""
    payload = request.get_json(silent=True) or {}
    result = predictor.predict(payload)
    status_code = 200 if result.status in {"ok", "insufficient_data"} else 500
    return jsonify(result.to_dict()), status_code
