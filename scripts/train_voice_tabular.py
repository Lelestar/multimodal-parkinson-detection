#!/usr/bin/env python3
"""
Entraîne un modèle tabulaire voix et exporte models/voice_parkinsons_tabular.joblib.

À lancer depuis la racine du dossier Voice_Casin_Wayne_Oxford_AI (voir README).
Validation par patient (GroupKFold). XGBoost si installé.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

VOICE_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(VOICE_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(VOICE_PROJECT_ROOT))

from src.modalities.voice.augmentation import apply_smote_safe

DATA_PATH = VOICE_PROJECT_ROOT / "data" / "parkinsons.data"
MODEL_OUT = VOICE_PROJECT_ROOT / "models" / "voice_parkinsons_tabular.joblib"


def patient_group(name: str) -> str:
    """Regroupe les enregistrements d'un même sujet (suffixe numérique retiré)."""
    return re.sub(r"_\d+$", "", str(name))


def build_candidates() -> list[tuple[str, object]]:
    """Modèles candidats ; XGBoost ajouté seulement si importable."""
    candidates: list[tuple[str, object]] = [
        (
            "logistic_regression",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=8000, class_weight="balanced", random_state=42)),
                ]
            ),
        ),
        (
            "svc_rbf",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
                ]
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1),
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1),
        ),
        (
            "gradient_boosting",
            GradientBoostingClassifier(random_state=42),
        ),
    ]
    try:
        from xgboost import XGBClassifier

        candidates.append(
            (
                "xgboost",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            XGBClassifier(
                                n_estimators=120,
                                max_depth=3,
                                learning_rate=0.06,
                                subsample=0.9,
                                colsample_bytree=0.9,
                                reg_lambda=1.0,
                                objective="binary:logistic",
                                eval_metric="logloss",
                                random_state=42,
                                tree_method="hist",
                            ),
                        ),
                    ]
                ),
            )
        )
    except ImportError:
        pass
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Entraîne le modèle voix tabulaire (Oxford / UCI).")
    parser.add_argument(
        "--augment-cv",
        action="store_true",
        help="Applique SMOTE sur chaque jeu d'apprentissage de la validation croisée (test toujours brut).",
    )
    parser.add_argument(
        "--augment-final",
        action="store_true",
        help="Après la CV, entraîne le modèle final sur données SMOTE (tout le jeu mélangé synthétique+réel).",
    )
    args = parser.parse_args()

    if not DATA_PATH.exists():
        print(f"Dataset introuvable: {DATA_PATH}", file=sys.stderr)
        return 1

    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in ("name", "status")]
    x = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["status"].to_numpy(dtype=np.int32)
    groups = df["name"].map(patient_group).to_numpy()

    gkf = GroupKFold(n_splits=5)
    leaderboard: list[tuple[str, float, float]] = []

    for name, estimator in build_candidates():
        fold_scores: list[float] = []
        for train_idx, test_idx in gkf.split(x, y, groups):
            x_tr, y_tr = x[train_idx], y[train_idx]
            if args.augment_cv:
                x_tr, y_tr, _ = apply_smote_safe(x_tr, y_tr)
            fold_model = clone(estimator)
            fold_model.fit(x_tr, y_tr)
            if hasattr(fold_model, "predict_proba"):
                p_pos = fold_model.predict_proba(x[test_idx])[:, 1]
            else:
                raw = fold_model.decision_function(x[test_idx])
                p_pos = 1.0 / (1.0 + np.exp(-raw))
            fold_scores.append(float(roc_auc_score(y[test_idx], p_pos)))
        leaderboard.append((name, float(np.mean(fold_scores)), float(np.std(fold_scores))))

    best = max(leaderboard, key=lambda t: t[1] - 0.35 * t[2])
    best_name = best[0]
    best_estimator = dict(build_candidates())[best_name]
    final_model = clone(best_estimator)
    x_fit, y_fit = x, y
    if args.augment_final:
        x_fit, y_fit, _ = apply_smote_safe(x, y)
    final_model.fit(x_fit, y_fit)

    note_parts = [
        "Modèle tabulaire entraîné sur biomarqueurs Oxford (Voice_Casin_Wayne_Oxford_AI/data).",
    ]
    if args.augment_cv:
        note_parts.append("CV avec SMOTE sur les plis d'apprentissage uniquement.")
    if args.augment_final:
        note_parts.append("Fit final sur jeu suréchantillonné SMOTE.")
    artifact = {
        "pipeline": final_model,
        "features": feature_cols,
        "model": best_name,
        "threshold": 0.5,
        "note": " ".join(note_parts),
        "metrics": {
            "leaderboard": [
                {
                    "model": name,
                    "mean_roc_auc": mean,
                    "std_roc_auc": std,
                    "selection_score": mean - 0.35 * std,
                }
                for name, mean, std in sorted(leaderboard, key=lambda t: -(t[1] - 0.35 * t[2]))
            ],
            "chosen": best_name,
            "mean_roc_auc": best[1],
            "std_roc_auc": best[2],
            "augmentation": {
                "smote_cv": bool(args.augment_cv),
                "smote_final": bool(args.augment_final),
            },
        },
        "data_path_hint": str(DATA_PATH),
    }
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_OUT)
    print(json.dumps(artifact["metrics"], indent=2, ensure_ascii=False))
    print("Écrit:", MODEL_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
