"""Extraction et validation des biomarqueurs vocaux tabulaires pour la modalité voix."""

from __future__ import annotations

from typing import Any

import numpy as np

# Colonnes du fichier parkinsons.data (UCI / Oxford), hors identifiant et cible.
VOICE_FEATURE_COLUMNS: tuple[str, ...] = (
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE",
)

# Clés simplifiées optionnelles pour les payloads JSON.
VOICE_PAYLOAD_ALIASES: dict[str, str] = {
    "mdvp_fo_hz": "MDVP:Fo(Hz)",
    "mdvp_fhi_hz": "MDVP:Fhi(Hz)",
    "mdvp_flo_hz": "MDVP:Flo(Hz)",
    "mdvp_jitter_pct": "MDVP:Jitter(%)",
    "mdvp_jitter_abs": "MDVP:Jitter(Abs)",
    "mdvp_rap": "MDVP:RAP",
    "mdvp_ppq": "MDVP:PPQ",
    "jitter_ddp": "Jitter:DDP",
    "mdvp_shimmer": "MDVP:Shimmer",
    "mdvp_shimmer_db": "MDVP:Shimmer(dB)",
    "shimmer_apq3": "Shimmer:APQ3",
    "shimmer_apq5": "Shimmer:APQ5",
    "mdvp_apq": "MDVP:APQ",
    "shimmer_dda": "Shimmer:DDA",
    "nhr": "NHR",
    "hnr": "HNR",
    "rpde": "RPDE",
    "dfa": "DFA",
    "spread1": "spread1",
    "spread2": "spread2",
    "d2": "D2",
    "ppe": "PPE",
}


def _resolve_payload_keys(payload: dict[str, Any]) -> dict[str, Any]:
    """Aplatit le payload : accepte un bloc `features` ou des clés à la racine."""
    inner = payload.get("features")
    if isinstance(inner, dict):
        merged = {**{k: v for k, v in payload.items() if k != "features"}, **inner}
    else:
        merged = dict(payload)
    resolved: dict[str, Any] = {}
    for key, value in merged.items():
        if key in ("features", "name", "session_id"):
            continue
        canonical = VOICE_PAYLOAD_ALIASES.get(str(key).lower(), str(key))
        resolved[canonical] = value
    return resolved


def build_voice_feature_row(
    payload: dict[str, Any],
    expected_columns: list[str],
) -> tuple[np.ndarray | None, list[str], list[str]]:
    """
    Prépare une ligne de features dans le même ordre que celui utilisé à l'entraînement.

    Retourne (vecteur 1 x n_features, clés manquantes, avertissements non bloquants).
    """
    warnings: list[str] = []
    flat = _resolve_payload_keys(payload)
    missing: list[str] = []
    values: list[float] = []

    for col in expected_columns:
        if col not in flat:
            missing.append(col)
            continue
        raw = flat[col]
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            return None, [col], [f"Valeur non numérique pour la colonne {col!r}."]

    if missing:
        return None, missing, warnings

    row = np.array(values, dtype=np.float64).reshape(1, -1)
    if np.any(~np.isfinite(row)):
        return None, [], ["Certaines valeurs sont infinies ou NaN."]

    return row, [], warnings
