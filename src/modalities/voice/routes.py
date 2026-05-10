"""Flask routes for the voice modality."""

from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from src.modalities.voice.predictor import VoicePredictor


voice_bp = Blueprint("voice", __name__)
predictor = VoicePredictor()


@voice_bp.get("/voice")
def voice_page():
    """Render the voice recording page."""
    return render_template("voice.html")


@voice_bp.post("/api/voice/predict")
def voice_predict():
    """Receive a WAV audio file and return a standardised prediction."""
    audio_file = request.files.get("audio")
    if audio_file is None:
        return jsonify({"status": "error", "warnings": ["Aucun fichier audio reçu."]}), 400

    audio_bytes = audio_file.read()
    result = predictor.predict({"audio": audio_bytes})
    status_code = 200 if result.status in {"ok", "insufficient_data"} else 500
    return jsonify(result.to_dict()), status_code
