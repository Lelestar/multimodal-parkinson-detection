"""Flask routes for the keyboard dynamics modality."""

from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from src.modalities.keyboard.predictor import KeyboardPredictor


keyboard_bp = Blueprint("keyboard", __name__)
predictor = KeyboardPredictor()


@keyboard_bp.get("/keyboard")
def keyboard_page():
    """Render the keyboard event capture page."""
    prompt = (
        "Le soleil se lève lentement sur la ville, et les rues commencent à se remplir de bruits ordinaires. "
        "Les habitants ouvrent leurs fenêtres, préparent leur déjeuner, vérifient la météo et commencent les "
        "petites routines qui marquent le début de la journée. Dans un appartement voisin, une radio diffuse les "
        "nouvelles pendant qu'une cafetière termine son cycle. Plus bas, un livreur dépose des colis devant une "
        "porte bleue, puis repart vers la rue suivante. Au marché, les étals se couvrent de fruits, de pain frais "
        "et de fleurs de saison. Une personne traverse la place avec un sac en toile, s'arrête pour saluer un ami, "
        "puis reprend son chemin sans se presser. La matinée avance doucement, portée par des gestes simples et "
        "répétés, comme si toute la ville trouvait peu à peu son rythme."
    )
    return render_template("keyboard.html", prompt=prompt)


@keyboard_bp.post("/api/keyboard/predict")
def keyboard_predict():
    """Receive keyboard events and return a standardized prediction."""
    payload = request.get_json(silent=True) or {}
    result = predictor.predict(payload)
    status_code = 200 if result.status in {"ok", "insufficient_data"} else 500
    return jsonify(result.to_dict()), status_code
