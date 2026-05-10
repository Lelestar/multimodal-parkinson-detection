"""Flask application factory and shared routes."""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from src.common.fusion import late_fusion
from src.common.registry import ModalityRegistration, registry
from src.modalities.keyboard.routes import keyboard_bp
from src.modalities.voice.routes import voice_bp


def create_app() -> Flask:
    """Create the Flask application and register shared routes."""
    app = Flask(__name__)
    app.register_blueprint(keyboard_bp)
    app.register_blueprint(voice_bp)
    registry.register(ModalityRegistration(name="keyboard", display_name="Clavier", route="/keyboard"))
    registry.register(ModalityRegistration(name="voice", display_name="Voix", route="/voice"))
    registry.register(ModalityRegistration(name="drawing", display_name="Dessin", route="#"))

    @app.get("/")
    def index():
        """Render the home page with available modalities."""
        return render_template("index.html", modalities=registry.list())

    @app.get("/results")
    def results():
        """Render the global results page."""
        return render_template("results.html", modalities=registry.list())

    @app.post("/api/fusion")
    def fusion_api():
        """Expose late fusion through a JSON API."""
        payload = request.get_json(silent=True) or {}
        predictions = payload.get("predictions", [])
        weights = payload.get("weights")
        result = late_fusion(predictions, weights=weights)
        return jsonify(result.to_dict())

    @app.get("/health")
    def health():
        """Return the minimal application health status."""
        return jsonify({"status": "ok"})

    return app
