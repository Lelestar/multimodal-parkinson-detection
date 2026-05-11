#!/usr/bin/env python3
"""Generate out-of-fold scores for the multimodal pseudo-cohort evaluation.

The exported application models are trained on all available data and should
not be used to estimate evaluation performance. This script retrains each
modality inside cross-validation folds and stores only predictions for held-out
samples. The fusion evaluation can then combine those scores without direct
train/test leakage at the unimodal scoring step.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fusion_dataset_validation import (  # noqa: E402
    KEYBOARD_ROOT,
    VOICE_COLUMN_MAP,
    VOICE_CSV,
    load_neuroqwerty_session,
)
from src.modalities.drawing.predictor import DrawingPredictor  # noqa: E402
from src.modalities.keyboard.features import build_agg_timing_xgb_feature_table  # noqa: E402


DRAWING_TRAIN_DIRS = [
    ROOT / "data" / "spiral" / "training" / "healthy",
    ROOT / "data" / "spiral" / "training" / "parkinson",
]
DRAWING_TEST_DIRS = [
    ROOT / "data" / "spiral" / "testing" / "healthy",
    ROOT / "data" / "spiral" / "testing" / "parkinson",
]

DRAWING_MODEL = ROOT / "models" / "drawing_spiral_v2_hog_lbp_pipeline.joblib"
KEYBOARD_MODEL = ROOT / "models" / "keyboard_dynamics_neuroqwerty_agg_timing_xgb.joblib"
VOICE_MODEL = ROOT / "models" / "voice_parkinson_xgb.joblib"
OUTPUT_PATH = ROOT / "data" / "processed" / "multimodal_oof_scores.csv"

RANDOM_STATE = 42
N_SPLITS = 5


@dataclass(frozen=True)
class OOFMetrics:
    """Compact metrics for one set of out-of-fold predictions."""

    modality: str
    n_samples: int
    n_subjects: int
    roc_auc: float
    balanced_accuracy_at_0_5: float


def _subject_group_count(y: np.ndarray, groups: np.ndarray) -> int:
    """Return the maximum safe number of stratified grouped folds."""
    per_class = []
    for label in np.unique(y):
        per_class.append(len(np.unique(groups[y == label])))
    return max(2, min(N_SPLITS, *per_class))


def _predict_positive(estimator, x):
    """Return positive-class probabilities for an estimator or pipeline."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x)[:, 1]
    decision = estimator.decision_function(x)
    return 1.0 / (1.0 + np.exp(-decision))


def _summarize(rows: list[dict[str, object]], modality: str) -> OOFMetrics:
    """Compute AUC and balanced accuracy for generated OOF rows."""
    frame = pd.DataFrame(rows)
    y_true = frame["label"].to_numpy(dtype=int)
    y_score = frame["score"].to_numpy(dtype=float)
    y_pred = (y_score >= 0.5).astype(int)
    return OOFMetrics(
        modality=modality,
        n_samples=len(frame),
        n_subjects=frame["subject_id"].nunique(),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        balanced_accuracy_at_0_5=float(balanced_accuracy_score(y_true, y_pred)),
    )


def _validate_oof_rows(rows: list[dict[str, object]], modality: str) -> None:
    """Validate that generated rows are complete and grouped by subject."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"Aucun score OOF généré pour la modalité {modality}.")
    if frame["score"].isna().any() or frame["fold"].isna().any():
        raise RuntimeError(f"Scores ou folds manquants pour la modalité {modality}.")
    if frame.duplicated(["modality", "sample_id"]).any():
        raise RuntimeError(f"Doublons sample_id détectés pour la modalité {modality}.")

    fold_count_by_subject = frame.groupby("subject_id")["fold"].nunique()
    leaking_subjects = fold_count_by_subject[fold_count_by_subject > 1]
    if not leaking_subjects.empty:
        raise RuntimeError(
            f"Fuite de groupe détectée pour {modality}: "
            f"{len(leaking_subjects)} sujet(s) présents dans plusieurs folds."
        )


def _drawing_subject_id(path: Path, label: int) -> str:
    """Extract a stable drawing subject id from a spiral file name."""
    match = re.match(r"^(V\d+(?:HE|PE))", path.stem, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return f"{label}:{path.stem}"


def _load_drawing_table(artifact: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """Extract drawing features from all available spiral images."""
    hog_params = artifact["hog_params"]
    lbp_params = artifact.get("lbp_params")
    img_size = tuple(hog_params["img_size"])
    use_otsu = artifact.get("preprocessing") == "otsu_binarization"

    rows: list[dict[str, object]] = []
    features: list[np.ndarray] = []
    for label, dirs in [(0, [DRAWING_TRAIN_DIRS[0], DRAWING_TEST_DIRS[0]]), (1, [DRAWING_TRAIN_DIRS[1], DRAWING_TEST_DIRS[1]])]:
        for directory in dirs:
            for path in sorted(directory.glob("*.png")):
                image = Image.open(path)
                arr = DrawingPredictor._preprocess(image, img_size, use_otsu=use_otsu)
                features.append(DrawingPredictor._extract_features(arr, hog_params, lbp_params))
                rows.append(
                    {
                        "modality": "drawing",
                        "sample_id": str(path.relative_to(ROOT)),
                        "subject_id": _drawing_subject_id(path, label),
                        "label": label,
                        "source_dataset": "spiral",
                    }
                )
    return np.vstack(features), pd.DataFrame(rows)


def generate_drawing_oof() -> tuple[list[dict[str, object]], OOFMetrics]:
    """Train drawing folds and return OOF scores."""
    artifact = joblib.load(DRAWING_MODEL)
    x, meta = _load_drawing_table(artifact)
    y = meta["label"].to_numpy(dtype=int)
    groups = meta["subject_id"].to_numpy()
    n_splits = _subject_group_count(y, groups)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    scores = np.full(len(meta), np.nan, dtype=float)
    for fold, (train_idx, test_idx) in enumerate(cv.split(x, y, groups), start=1):
        model = clone(artifact["pipeline"])
        model.fit(x[train_idx], y[train_idx])
        scores[test_idx] = _predict_positive(model, x[test_idx])
        meta.loc[test_idx, "fold"] = fold

    meta["score"] = scores
    rows = meta.to_dict("records")
    _validate_oof_rows(rows, "drawing")
    return rows, _summarize(rows, "drawing")


def _load_keyboard_segment_table(artifact: dict) -> pd.DataFrame:
    """Build one row per keyboard segment from all NeuroQWERTY sessions."""
    window = int(artifact.get("window", 300))
    stride = int(artifact.get("stride", 150))
    min_len = int(artifact.get("min_segment_len", 300))
    rows: list[pd.DataFrame] = []

    for dataset in ["MIT-CS1PD", "MIT-CS2PD"]:
        dataset_root = KEYBOARD_ROOT / dataset
        gt_path = dataset_root / f"GT_DataPD_{dataset}.csv"
        raw_dir = dataset_root / f"data_{dataset}"
        if not gt_path.exists() or not raw_dir.exists():
            continue

        gt = pd.read_csv(gt_path)
        gt["label"] = gt["gt"].astype(str).str.lower().isin(["true", "1"]).astype(int)
        file_columns = [column for column in gt.columns if column.startswith("file_")]

        for _, row in gt.iterrows():
            subject_id = f"{dataset}:{row['pID']}"
            for column in file_columns:
                value = row.get(column)
                if pd.isna(value):
                    continue
                session_path = raw_dir / str(value)
                if not session_path.exists():
                    continue
                features = build_agg_timing_xgb_feature_table(
                    load_neuroqwerty_session(session_path),
                    window_size=window,
                    stride=stride,
                    min_segment_len=min_len,
                )
                if features.empty:
                    continue
                features = features.copy()
                features["modality"] = "keyboard"
                features["sample_id"] = f"{dataset}:{session_path.name}"
                features["subject_id"] = subject_id
                features["label"] = int(row["label"])
                features["source_dataset"] = dataset
                rows.append(features)

    if not rows:
        raise RuntimeError("Aucune feature clavier générée. Vérifiez le dataset NeuroQWERTY.")
    return pd.concat(rows, ignore_index=True)


def generate_keyboard_oof() -> tuple[list[dict[str, object]], OOFMetrics]:
    """Train keyboard folds by subject and aggregate segment scores per session."""
    artifact = joblib.load(KEYBOARD_MODEL)
    segments = _load_keyboard_segment_table(artifact)
    feature_names = artifact.get("features")
    if not feature_names:
        feature_names = [
            column
            for column in segments.columns
            if column not in {"modality", "sample_id", "subject_id", "label", "source_dataset"}
        ]

    session_meta = segments[["sample_id", "subject_id", "label"]].drop_duplicates("sample_id")
    y_session = session_meta["label"].to_numpy(dtype=int)
    groups_session = session_meta["subject_id"].to_numpy()
    n_splits = _subject_group_count(y_session, groups_session)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    oof_rows: list[dict[str, object]] = []
    for fold, (train_session_idx, test_session_idx) in enumerate(
        cv.split(session_meta, y_session, groups_session),
        start=1,
    ):
        train_sessions = set(session_meta.iloc[train_session_idx]["sample_id"])
        test_sessions = set(session_meta.iloc[test_session_idx]["sample_id"])

        train_mask = segments["sample_id"].isin(train_sessions)
        test_mask = segments["sample_id"].isin(test_sessions)
        model = clone(artifact["pipeline"])
        model.fit(segments.loc[train_mask, feature_names], segments.loc[train_mask, "label"].astype(int))
        segment_scores = _predict_positive(model, segments.loc[test_mask, feature_names])

        scored = segments.loc[test_mask, ["sample_id", "subject_id", "label", "source_dataset"]].copy()
        scored["segment_score"] = segment_scores
        for sample_id, group in scored.groupby("sample_id"):
            oof_rows.append(
                {
                    "modality": "keyboard",
                    "sample_id": sample_id,
                    "subject_id": group["subject_id"].iloc[0],
                    "label": int(group["label"].iloc[0]),
                    "score": float(group["segment_score"].mean()),
                    "fold": fold,
                    "source_dataset": group["source_dataset"].iloc[0],
                }
            )

    _validate_oof_rows(oof_rows, "keyboard")
    return oof_rows, _summarize(oof_rows, "keyboard")


def _load_voice_table() -> pd.DataFrame:
    """Load voice features with one row per recording and subject ids."""
    df = pd.read_csv(VOICE_CSV, header=1)
    df["class"] = pd.to_numeric(df["class"], errors="coerce")
    df = df[df["class"].notna()].copy()
    df["class"] = df["class"].astype(int)

    feature_frame = df[list(VOICE_COLUMN_MAP.keys())].rename(columns=VOICE_COLUMN_MAP)
    feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce")
    feature_frame["modality"] = "voice"
    feature_frame["sample_id"] = [f"sakar2019:{idx}" for idx in df.index]
    feature_frame["subject_id"] = "sakar2019:" + df["id"].astype(str)
    feature_frame["label"] = df["class"].astype(int).values
    feature_frame["source_dataset"] = "sakar2019"
    return feature_frame.reset_index(drop=True)


def generate_voice_oof() -> tuple[list[dict[str, object]], OOFMetrics]:
    """Train voice folds by subject and return OOF scores."""
    artifact = joblib.load(VOICE_MODEL)
    frame = _load_voice_table()
    feature_names = artifact.get("feature_names", artifact.get("features", list(VOICE_COLUMN_MAP.values())))
    y = frame["label"].to_numpy(dtype=int)
    groups = frame["subject_id"].to_numpy()
    n_splits = _subject_group_count(y, groups)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    scores = np.full(len(frame), np.nan, dtype=float)
    for fold, (train_idx, test_idx) in enumerate(cv.split(frame[feature_names], y, groups), start=1):
        train_features = frame.iloc[train_idx][feature_names]
        test_features = frame.iloc[test_idx][feature_names]
        train_medians = train_features.median()
        x_train = train_features.fillna(train_medians).to_numpy(dtype=float)
        x_test = test_features.fillna(train_medians).to_numpy(dtype=float)

        model = clone(artifact["pipeline"])
        model.fit(x_train, y[train_idx])
        scores[test_idx] = _predict_positive(model, x_test)
        frame.loc[test_idx, "fold"] = fold

    frame["score"] = scores
    rows = frame[["modality", "sample_id", "subject_id", "label", "score", "fold", "source_dataset"]].to_dict("records")
    _validate_oof_rows(rows, "voice")
    return rows, _summarize(rows, "voice")


def main() -> None:
    """Generate the OOF score file consumed by the multimodal evaluation."""
    print("Génération des scores out-of-fold unimodaux")
    print("=" * 56)

    all_rows: list[dict[str, object]] = []
    metrics: list[OOFMetrics] = []
    for name, generator in [
        ("drawing", generate_drawing_oof),
        ("keyboard", generate_keyboard_oof),
        ("voice", generate_voice_oof),
    ]:
        print(f"Modalité {name}...")
        rows, summary = generator()
        all_rows.extend(rows)
        metrics.append(summary)
        print(
            f"  n={summary.n_samples}, sujets={summary.n_subjects}, "
            f"AUC={summary.roc_auc:.3f}, balanced_acc@0.5={summary.balanced_accuracy_at_0_5:.3f}"
        )

    output = pd.DataFrame(all_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print(f"\nScores sauvegardés dans {OUTPUT_PATH.relative_to(ROOT)}")
    print("\nRésumé OOF")
    print(pd.DataFrame([metric.__dict__ for metric in metrics]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
