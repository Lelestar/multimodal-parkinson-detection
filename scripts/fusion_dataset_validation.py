#!/usr/bin/env python3
"""Evaluate late fusion on a repeated cross-dataset pseudo-cohort.

This script does not create real multimodal patients. It builds repeated
label-consistent composites: one drawing sample, one keyboard subject/session
and one voice sample are randomly paired inside the same class.

The goal is to stress-test the late-fusion code and compare score separation
between modalities when no shared subject-level multimodal dataset is available.
The resulting metrics must be reported as exploratory only.
"""

from __future__ import annotations

import base64
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.fusion import late_fusion
from src.common.schemas import PredictionResult, score_to_label
from src.modalities.drawing.predictor import DrawingPredictor
from src.modalities.keyboard.features import build_agg_timing_xgb_feature_table


DRAWING_HC_DIR = ROOT / "data" / "spiral" / "testing" / "healthy"
DRAWING_PD_DIR = ROOT / "data" / "spiral" / "testing" / "parkinson"
KEYBOARD_ROOT = ROOT / "data" / "neuroqwerty-mit-csxpd-dataset-1.0.0"
VOICE_CSV = ROOT / "data" / "parkinson_disease_classification" / "pd_speech_features.csv"

DRAWING_MODEL = ROOT / "models" / "drawing_spiral_v2_hog_lbp_pipeline.joblib"
KEYBOARD_MODEL = ROOT / "models" / "keyboard_dynamics_neuroqwerty_agg_timing_xgb.joblib"
VOICE_MODEL = ROOT / "models" / "voice_parkinson_xgb.joblib"
OOF_SCORES = ROOT / "data" / "processed" / "multimodal_oof_scores.csv"

KEY_COLUMNS = ["key", "hold_time", "release_time", "press_time"]

VOICE_COLUMN_MAP: dict[str, str] = {
    "locPctJitter": "MDVP:Jitter(%)",
    "locAbsJitter": "MDVP:Jitter(Abs)",
    "rapJitter": "MDVP:RAP",
    "ppq5Jitter": "MDVP:PPQ",
    "ddpJitter": "Jitter:DDP",
    "locShimmer": "MDVP:Shimmer",
    "locDbShimmer": "MDVP:Shimmer(dB)",
    "apq3Shimmer": "Shimmer:APQ3",
    "apq5Shimmer": "Shimmer:APQ5",
    "apq11Shimmer": "MDVP:APQ",
    "ddaShimmer": "Shimmer:DDA",
    "meanNoiseToHarmHarmonicity": "NHR",
    "meanHarmToNoiseHarmonicity": "HNR",
    "RPDE": "RPDE",
    "DFA": "DFA",
    "PPE": "PPE",
}

CONFIGURATIONS: dict[str, tuple[str, ...]] = {
    "drawing": ("drawing",),
    "keyboard": ("keyboard",),
    "voice": ("voice",),
    "drawing+keyboard": ("drawing", "keyboard"),
    "drawing+voice": ("drawing", "voice"),
    "keyboard+voice": ("keyboard", "voice"),
    "drawing+keyboard+voice": ("drawing", "keyboard", "voice"),
}


@dataclass(frozen=True)
class ScorePool:
    """Positive-class scores for healthy controls and Parkinson samples."""

    hc: np.ndarray
    pd: np.ndarray


def image_to_base64(path: Path) -> str:
    """Convert an image file to a base64 PNG payload."""
    buffer = io.BytesIO()
    Image.open(path).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def load_neuroqwerty_session(file_path: Path) -> pd.DataFrame:
    """Load one NeuroQWERTY CSV session into the feature extractor format."""
    df = pd.read_csv(file_path, header=None, names=KEY_COLUMNS)
    df["key"] = df["key"].astype(str).str.strip().str.replace('"', "", regex=False)
    for column in ["hold_time", "release_time", "press_time"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["hold_time", "release_time", "press_time"])
    df = df[
        (df["press_time"] > 0)
        & (df["release_time"] > 0)
        & (df["hold_time"].between(0.0, 5.0))
    ]
    df = df.sort_values("press_time").reset_index(drop=True)
    df["flight_time"] = df["press_time"].diff()
    df.loc[df["flight_time"] < 0, "flight_time"] = np.nan
    return df


def score_keyboard_session(session_path: Path, artifact: dict) -> PredictionResult:
    """Score one NeuroQWERTY session with the exported keyboard model."""
    window = int(artifact.get("window", 300))
    stride = int(artifact.get("stride", 150))
    min_len = int(artifact.get("min_segment_len", 300))
    threshold = float(artifact.get("threshold", 0.5))
    pipeline = artifact.get("pipeline") or artifact.get("model")

    keystrokes = load_neuroqwerty_session(session_path)
    features = build_agg_timing_xgb_feature_table(
        keystrokes,
        window_size=window,
        stride=stride,
        min_segment_len=min_len,
    )
    if features.empty:
        return PredictionResult(
            modality="keyboard",
            status="insufficient_data",
            confidence=0.0,
            warnings=[f"Données insuffisantes : {len(keystrokes)} frappes valides."],
        )

    probabilities = pipeline.predict_proba(features)[:, 1]
    score = float(np.mean(probabilities))
    confidence = float(max(0.0, 1.0 - np.std(probabilities))) if len(probabilities) > 1 else 1.0
    return PredictionResult(
        modality="keyboard",
        status="ok",
        score=score,
        confidence=confidence,
        label=score_to_label(score, high_threshold=threshold),
        details={"session_file": session_path.name, "n_segments": len(probabilities)},
    )


def load_keyboard_session_paths(dataset: str = "MIT-CS1PD") -> tuple[list[Path], list[Path]]:
    """Return one available session path per subject for HC and PD classes."""
    dataset_root = KEYBOARD_ROOT / dataset
    gt_path = dataset_root / f"GT_DataPD_{dataset}.csv"
    raw_dir = dataset_root / f"data_{dataset}"
    gt = pd.read_csv(gt_path)
    gt["gt_bool"] = gt["gt"].astype(str).str.lower().isin(["true", "1"])

    hc_paths: list[Path] = []
    pd_paths: list[Path] = []
    file_columns = [column for column in gt.columns if column.startswith("file_")]

    for _, row in gt.iterrows():
        target = pd_paths if row["gt_bool"] else hc_paths
        for column in file_columns:
            value = row.get(column)
            if pd.isna(value):
                continue
            path = raw_dir / str(value)
            if path.exists():
                target.append(path)
                break

    return hc_paths, pd_paths


def load_voice_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Sakar-2019 voice features and split them into HC and PD rows."""
    df = pd.read_csv(VOICE_CSV, header=0)
    if "class" not in df.columns:
        df = pd.read_csv(VOICE_CSV, header=1)

    df["class"] = pd.to_numeric(df["class"], errors="coerce")
    df = df[df["class"].notna()].copy()
    df["class"] = df["class"].astype(int)

    feature_frame = df[list(VOICE_COLUMN_MAP.keys())].rename(columns=VOICE_COLUMN_MAP)
    feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce")
    feature_frame = feature_frame.fillna(feature_frame.median())
    feature_frame["class"] = df["class"].values

    hc = feature_frame[feature_frame["class"] == 0].drop(columns=["class"]).reset_index(drop=True)
    pd_ = feature_frame[feature_frame["class"] == 1].drop(columns=["class"]).reset_index(drop=True)
    return hc, pd_


def score_voice_row(row: pd.Series, artifact: dict) -> PredictionResult:
    """Score one voice feature row with the exported voice model."""
    feature_names = artifact.get("feature_names", artifact.get("features", list(VOICE_COLUMN_MAP.values())))
    medians = artifact.get("feature_medians", {})
    values = []
    for feature in feature_names:
        value = row.get(feature, medians.get(feature, 0.0))
        if pd.isna(value):
            value = medians.get(feature, 0.0)
        values.append(float(value))
    feature_vector = np.asarray(values, dtype=float).reshape(1, -1)
    pipeline = artifact.get("pipeline") or artifact.get("model")
    score = float(pipeline.predict_proba(feature_vector)[0, 1])
    threshold = float(artifact.get("threshold", 0.5))
    return PredictionResult(
        modality="voice",
        status="ok",
        score=score,
        confidence=1.0,
        label=score_to_label(score, high_threshold=threshold),
    )


def build_drawing_pool(predictor: DrawingPredictor) -> ScorePool:
    """Score all available drawing test images."""
    hc_scores = [
        predictor.predict({"image_b64": image_to_base64(path)}).score
        for path in sorted(DRAWING_HC_DIR.glob("*.png"))
    ]
    pd_scores = [
        predictor.predict({"image_b64": image_to_base64(path)}).score
        for path in sorted(DRAWING_PD_DIR.glob("*.png"))
    ]
    return ScorePool(
        hc=np.array([score for score in hc_scores if score is not None], dtype=float),
        pd=np.array([score for score in pd_scores if score is not None], dtype=float),
    )


def build_keyboard_pool(artifact: dict, dataset: str = "MIT-CS1PD") -> ScorePool:
    """Score one available NeuroQWERTY session per subject."""
    hc_paths, pd_paths = load_keyboard_session_paths(dataset)

    def score_paths(paths: list[Path]) -> np.ndarray:
        scores: list[float] = []
        for path in paths:
            result = score_keyboard_session(path, artifact)
            if result.status == "ok" and result.score is not None:
                scores.append(result.score)
        return np.array(scores, dtype=float)

    return ScorePool(hc=score_paths(hc_paths), pd=score_paths(pd_paths))


def build_voice_pool(artifact: dict) -> ScorePool:
    """Score all available Sakar-2019 voice rows."""
    hc_df, pd_df = load_voice_dataset()

    def score_rows(frame: pd.DataFrame) -> np.ndarray:
        scores = [score_voice_row(row, artifact).score for _, row in frame.iterrows()]
        return np.array([score for score in scores if score is not None], dtype=float)

    return ScorePool(hc=score_rows(hc_df), pd=score_rows(pd_df))


def check_required_paths() -> None:
    """Fail early with a clear message when OOF scores are missing."""
    if not OOF_SCORES.exists():
        print("Scores out-of-fold introuvables :")
        print(f"  - {OOF_SCORES.relative_to(ROOT)}")
        print("\nGénérez-les d'abord avec :")
        print("  python scripts/generate_unimodal_oof_scores.py")
        sys.exit(1)


def check_model_scoring_paths() -> None:
    """Fail early with a clear message when raw resources are missing."""
    required = [
        DRAWING_HC_DIR,
        DRAWING_PD_DIR,
        KEYBOARD_ROOT / "MIT-CS1PD" / "GT_DataPD_MIT-CS1PD.csv",
        KEYBOARD_ROOT / "MIT-CS1PD" / "data_MIT-CS1PD",
        VOICE_CSV,
        DRAWING_MODEL,
        KEYBOARD_MODEL,
        VOICE_MODEL,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        print("Ressources manquantes :")
        for path in missing:
            print(f"  - {path.relative_to(ROOT)}")
        print("\nVoir docs/multimodal/evaluation_pseudo_cohorte.md pour les chemins attendus.")
        sys.exit(1)


def load_oof_pools(path: Path = OOF_SCORES) -> dict[str, ScorePool]:
    """Load out-of-fold modality scores as HC/PD pools."""
    frame = pd.read_csv(path)
    required_columns = {"modality", "label", "score", "sample_id", "subject_id", "fold"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path.relative_to(ROOT)} : {sorted(missing)}")

    pools: dict[str, ScorePool] = {}
    for modality, group in frame.groupby("modality"):
        hc = group[group["label"].astype(int) == 0]["score"].to_numpy(dtype=float)
        pd_ = group[group["label"].astype(int) == 1]["score"].to_numpy(dtype=float)
        if len(hc) == 0 or len(pd_) == 0:
            raise ValueError(f"Scores insuffisants pour la modalité {modality}.")
        pools[str(modality)] = ScorePool(hc=hc, pd=pd_)
    return pools


def summarize_pool(name: str, pool: ScorePool) -> dict[str, float | int | str]:
    """Return basic score separation metrics for one modality."""
    y_true = np.array([0] * len(pool.hc) + [1] * len(pool.pd))
    y_score = np.concatenate([pool.hc, pool.pd])
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "configuration": name,
        "n_hc": len(pool.hc),
        "n_pd": len(pool.pd),
        "mean_hc": float(np.mean(pool.hc)) if len(pool.hc) else np.nan,
        "mean_pd": float(np.mean(pool.pd)) if len(pool.pd) else np.nan,
        "auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) == 2 else np.nan,
        "balanced_accuracy_at_0.5": float(balanced_accuracy_score(y_true, y_pred)),
    }


def sample_composite_scores(
    pools: dict[str, ScorePool],
    modalities: tuple[str, ...],
    rng: np.random.Generator,
    n_per_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample repeated label-consistent composites and return HC/PD fusion scores."""
    hc_scores: list[float] = []
    pd_scores: list[float] = []

    for label, output in [("hc", hc_scores), ("pd", pd_scores)]:
        for _ in range(n_per_class):
            predictions: list[PredictionResult] = []
            for modality in modalities:
                candidates = getattr(pools[modality], label)
                score = float(rng.choice(candidates))
                predictions.append(
                    PredictionResult(
                        modality=modality,
                        status="ok",
                        score=score,
                        confidence=1.0,
                        label=score_to_label(score),
                    )
                )
            fusion = late_fusion(predictions)
            if fusion.status == "ok" and fusion.score is not None:
                output.append(float(fusion.score))

    return np.array(hc_scores, dtype=float), np.array(pd_scores, dtype=float)


def evaluate_configurations(
    pools: dict[str, ScorePool],
    n_repeats: int = 50,
    n_per_class: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate individual modalities and late-fusion combinations repeatedly."""
    rows: list[dict[str, float | int | str]] = []
    master_rng = np.random.default_rng(seed)

    for configuration, modalities in CONFIGURATIONS.items():
        aucs: list[float] = []
        balanced_accuracies: list[float] = []
        mean_hc_values: list[float] = []
        mean_pd_values: list[float] = []

        for _ in range(n_repeats):
            rng = np.random.default_rng(int(master_rng.integers(0, 2**32 - 1)))
            hc_scores, pd_scores = sample_composite_scores(
                pools,
                modalities,
                rng=rng,
                n_per_class=n_per_class,
            )
            y_true = np.array([0] * len(hc_scores) + [1] * len(pd_scores))
            y_score = np.concatenate([hc_scores, pd_scores])
            y_pred = (y_score >= 0.5).astype(int)
            aucs.append(float(roc_auc_score(y_true, y_score)))
            balanced_accuracies.append(float(balanced_accuracy_score(y_true, y_pred)))
            mean_hc_values.append(float(np.mean(hc_scores)))
            mean_pd_values.append(float(np.mean(pd_scores)))

        rows.append(
            {
                "configuration": configuration,
                "modalities": "+".join(modalities),
                "repeats": n_repeats,
                "n_per_class_per_repeat": n_per_class,
                "auc_mean": np.mean(aucs),
                "auc_std": np.std(aucs, ddof=1),
                "balanced_accuracy_mean": np.mean(balanced_accuracies),
                "balanced_accuracy_std": np.std(balanced_accuracies, ddof=1),
                "mean_hc_score": np.mean(mean_hc_values),
                "mean_pd_score": np.mean(mean_pd_values),
            }
        )

    return pd.DataFrame(rows).sort_values("auc_mean", ascending=False).reset_index(drop=True)


def main() -> None:
    """Run the pseudo-cohort evaluation and print compact tables."""
    print("Evaluation multimodale par pseudo-cohorte cross-dataset avec scores OOF")
    print("=" * 64)
    print(
        "Attention : les composites n'associent pas les trois modalités d'une même personne.\n"
        "Les scores unimodaux sont out-of-fold : chaque score vient d'un modèle qui n'a pas vu cet exemple.\n"
        "Les résultats restent exploratoires et servent surtout à tester la fusion tardive.\n"
    )

    check_required_paths()
    pools = load_oof_pools()

    pool_summary = pd.DataFrame(
        summarize_pool(name, pool)
        for name, pool in pools.items()
    )
    print("\nScores disponibles par modalité")
    print(pool_summary.round(3).to_string(index=False))

    results = evaluate_configurations(pools)
    print("\nEvaluation répétée des fusions")
    print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
