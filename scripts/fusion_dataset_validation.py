#!/usr/bin/env python3
"""
Cross-dataset multimodal fusion validation.

Construit deux "patients composites" à partir de vrais échantillons de dataset :
  - Patient HC : 1 image spirale HC + 1 session clavier HC + 1 échantillon voix HC
  - Patient PD : 1 image spirale PD + 1 session clavier PD + 1 échantillon voix PD

Applique ensuite late_fusion() et affiche les scores par modalité + score fusionné.

DATASETS attendus :
  - Dessin  : data/spiral/testing/{healthy,parkinson}/*.png
  - Clavier : data/keyboard/MIT-CS1PD/{GT_DataPD_MIT-CS1PD.csv, data_MIT-CS1PD/*.csv}
  - Voix    : data/speech/pd_speech_features.csv  (Sakar-2019-UCI-470)

Usage (depuis la racine du projet, venv actif) :
    python scripts/fusion_dataset_validation.py
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── rendre le projet importable depuis scripts/ ───────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from src.common.fusion import late_fusion
from src.common.schemas import PredictionResult, score_to_label
from src.modalities.drawing.predictor import DrawingPredictor
from src.modalities.keyboard.features import build_agg_timing_xgb_feature_table

# ── chemins ───────────────────────────────────────────────────────────────────
SPIRAL_HC_DIR  = ROOT / "data" / "spiral" / "testing" / "healthy"
SPIRAL_PD_DIR  = ROOT / "data" / "spiral" / "testing" / "parkinson"
KEYBOARD_ROOT  = ROOT / "data" / "keyboard"
KEYBOARD_MODEL = ROOT / "models" / "keyboard_dynamics_neuroqwerty_agg_timing_xgb.joblib"
VOICE_CSV      = ROOT / "data" / "speech" / "pd_speech_features.csv"
VOICE_MODEL    = ROOT / "models" / "voice_parkinson_xgb.joblib"

# ── colonnes NeuroQWERTY ──────────────────────────────────────────────────────
KEY_COLUMNS = ["key", "hold_time", "release_time", "press_time"]

# ── correspondance Sakar-2019 → noms UCI-174 utilisés à l'entraînement ────────
VOICE_COLUMN_MAP: dict[str, str] = {
    "locPctJitter":               "MDVP:Jitter(%)",
    "locAbsJitter":               "MDVP:Jitter(Abs)",
    "rapJitter":                  "MDVP:RAP",
    "ppq5Jitter":                 "MDVP:PPQ",
    "ddpJitter":                  "Jitter:DDP",
    "locShimmer":                 "MDVP:Shimmer",
    "locDbShimmer":               "MDVP:Shimmer(dB)",
    "apq3Shimmer":                "Shimmer:APQ3",
    "apq5Shimmer":                "Shimmer:APQ5",
    "apq11Shimmer":               "MDVP:APQ",
    "ddaShimmer":                 "Shimmer:DDA",
    "meanNoiseToHarmHarmonicity": "NHR",
    "meanHarmToNoiseHarmonicity": "HNR",
    "RPDE":                       "RPDE",
    "DFA":                        "DFA",
    "PPE":                        "PPE",
}
VOICE_FEATURE_NAMES = list(VOICE_COLUMN_MAP.values())


# ─────────────────────────────────────────────────────────────────────────────
# Helpers dessin
# ─────────────────────────────────────────────────────────────────────────────

def _img_to_b64(path: Path) -> str:
    buf = io.BytesIO()
    Image.open(path).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def score_drawing(image_path: Path, predictor: DrawingPredictor) -> PredictionResult:
    return predictor.predict({"image_b64": _img_to_b64(image_path)})


def pick_spiral_images(n: int = 5) -> tuple[list[Path], list[Path]]:
    """Retourne jusqu'à n chemins d'images HC et PD."""
    hc = sorted(SPIRAL_HC_DIR.glob("*.png"))[:n]
    pd_ = sorted(SPIRAL_PD_DIR.glob("*.png"))[:n]
    return hc, pd_


# ─────────────────────────────────────────────────────────────────────────────
# Helpers clavier
# ─────────────────────────────────────────────────────────────────────────────

def load_neuroqwerty_session(file_path: Path) -> pd.DataFrame:
    """
    Charge un fichier CSV NeuroQWERTY et retourne un DataFrame compatible avec
    build_agg_timing_xgb_feature_table() (colonnes : hold_time, press_time,
    release_time, flight_time).
    """
    df = pd.read_csv(file_path, header=None, names=KEY_COLUMNS)
    df["key"] = df["key"].astype(str).str.strip().str.replace('"', "", regex=False)
    for col in ["hold_time", "release_time", "press_time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
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


def pick_keyboard_subjects(dataset: str = "MIT-CS1PD") -> tuple[Path, Path]:
    """
    Lit le fichier GT du dataset NeuroQWERTY et retourne un fichier de session
    pour un sujet HC et un sujet PD.
    """
    gt_path = KEYBOARD_ROOT / dataset / f"GT_DataPD_{dataset}.csv"
    raw_dir = KEYBOARD_ROOT / dataset / f"data_{dataset}"
    gt = pd.read_csv(gt_path)
    # gt peut être une chaîne "True"/"False" ou un booléen selon le CSV
    gt["gt_bool"] = gt["gt"].astype(str).str.lower().isin(["true", "1"])

    hc_rows = gt[gt["gt_bool"] == False]
    pd_rows = gt[gt["gt_bool"] == True]

    if hc_rows.empty or pd_rows.empty:
        raise RuntimeError(f"GT manquant pour HC ou PD dans {gt_path}")

    hc_file = raw_dir / str(hc_rows.iloc[0]["file_1"])
    pd_file = raw_dir / str(pd_rows.iloc[0]["file_1"])
    return hc_file, pd_file


def score_keyboard(session_path: Path, artifact: dict) -> PredictionResult:
    """Score une session NeuroQWERTY en utilisant l'artefact clavier."""
    window    = int(artifact.get("window", 300))
    stride    = int(artifact.get("stride", 150))
    min_len   = int(artifact.get("min_segment_len", 300))
    threshold = float(artifact.get("threshold", 0.5))
    model     = artifact.get("pipeline") or artifact.get("model")

    try:
        keystrokes = load_neuroqwerty_session(session_path)
    except Exception as exc:
        return PredictionResult(
            modality="keyboard", status="error", confidence=0.0,
            warnings=[f"Erreur lecture fichier clavier : {exc}"],
        )

    features = build_agg_timing_xgb_feature_table(
        keystrokes, window_size=window, stride=stride, min_segment_len=min_len,
    )

    if features.empty:
        return PredictionResult(
            modality="keyboard", status="insufficient_data", confidence=0.0,
            warnings=[f"Données insuffisantes : {len(keystrokes)} frappes valides."],
        )

    probs = model.predict_proba(features)[:, 1]
    score = float(np.mean(probs))
    confidence = float(max(0.0, 1.0 - np.std(probs))) if len(probs) > 1 else 1.0
    label = score_to_label(score, high_threshold=threshold)

    return PredictionResult(
        modality="keyboard",
        status="ok",
        score=score,
        label=label,
        confidence=confidence,
        details={
            "n_segments": len(probs),
            "session_file": session_path.name,
            "scores_per_segment": [round(float(p), 3) for p in probs],
        },
    )


def score_keyboard_subjects(
    dataset: str,
    n_per_class: int,
    artifact: dict,
) -> tuple[list[float], list[float]]:
    """Score n sujets HC et n sujets PD depuis un dataset NeuroQWERTY."""
    gt_path = KEYBOARD_ROOT / dataset / f"GT_DataPD_{dataset}.csv"
    raw_dir = KEYBOARD_ROOT / dataset / f"data_{dataset}"
    gt = pd.read_csv(gt_path)
    gt["gt_bool"] = gt["gt"].astype(str).str.lower().isin(["true", "1"])

    hc_scores: list[float] = []
    pd_scores: list[float] = []

    for label_bool, score_list in [(False, hc_scores), (True, pd_scores)]:
        rows = gt[gt["gt_bool"] == label_bool].head(n_per_class)
        for _, row in rows.iterrows():
            for file_col in [c for c in gt.columns if c.startswith("file_")]:
                fname = row.get(file_col)
                if pd.isna(fname):
                    continue
                fpath = raw_dir / str(fname)
                if not fpath.exists():
                    continue
                result = score_keyboard(fpath, artifact)
                if result.status == "ok" and result.score is not None:
                    score_list.append(result.score)
                break  # une session par sujet suffit

    return hc_scores, pd_scores


# ─────────────────────────────────────────────────────────────────────────────
# Helpers voix
# ─────────────────────────────────────────────────────────────────────────────

def load_voice_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge le CSV Sakar-2019-UCI-470, extrait les 16 features Praat-compatibles
    et retourne (hc_df, pd_df).

    Certaines versions du CSV ont deux lignes d'en-tête (header=1) ;
    d'autres n'en ont qu'une (header=0). On détecte automatiquement.
    """
    df = pd.read_csv(VOICE_CSV, header=0)
    if "class" not in df.columns:
        # Deuxième format : la vraie ligne d'en-tête est en position 1
        df = pd.read_csv(VOICE_CSV, header=1)
    df["class"] = pd.to_numeric(df["class"], errors="coerce")
    df = df[df["class"].notna()].copy()
    df["class"] = df["class"].astype(int)

    feat_df = df[list(VOICE_COLUMN_MAP.keys())].rename(columns=VOICE_COLUMN_MAP)
    feat_df = feat_df.apply(pd.to_numeric, errors="coerce")
    medians = feat_df.median()
    feat_df = feat_df.fillna(medians)
    feat_df["class"] = df["class"].values

    hc_df = feat_df[feat_df["class"] == 0].drop(columns=["class"]).reset_index(drop=True)
    pd_df = feat_df[feat_df["class"] == 1].drop(columns=["class"]).reset_index(drop=True)
    return hc_df, pd_df


def score_voice(row: pd.Series, artifact: dict) -> PredictionResult:
    """Score une ligne de features vocales avec l'artefact voix."""
    threshold = float(artifact.get("threshold", 0.5))
    model = artifact.get("pipeline") or artifact.get("model")
    X = np.array([[row[feat] for feat in VOICE_FEATURE_NAMES]])
    score = float(model.predict_proba(X)[0, 1])
    label = score_to_label(score, high_threshold=threshold)
    return PredictionResult(
        modality="voice", status="ok", score=score, label=label, confidence=1.0,
    )


def score_voice_population(
    voice_df: pd.DataFrame, artifact: dict, n: int,
) -> list[float]:
    """Score les n premiers échantillons d'un sous-ensemble voix."""
    scores = []
    for i in range(min(n, len(voice_df))):
        r = score_voice(voice_df.iloc[i], artifact)
        if r.score is not None:
            scores.append(r.score)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Affichage
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_score(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "N/A"


def print_composite_patient(
    title: str,
    preds: list[PredictionResult],
    fusion,
) -> None:
    W = 62
    print(f"\n{'='*W}")
    print(f"  {title}")
    print(f"{'='*W}")
    for p in preds:
        tag = f"[{p.modality:10s}]"
        info = (
            f"score={_fmt_score(p.score)}  "
            f"conf={_fmt_score(p.confidence)}  "
            f"label={p.label or p.status}"
        )
        print(f"  {tag}  {info}")
        if p.status == "keyboard" and p.details:
            segs = p.details.get("scores_per_segment", [])
            if segs:
                print(f"             segments: {segs}")
    print(f"  {'─'*W}")
    print(
        f"  [FUSION    ]  "
        f"score={_fmt_score(fusion.score)}  "
        f"conf={_fmt_score(fusion.confidence)}  "
        f"label={fusion.label or fusion.status}"
    )
    if fusion.ignored_modalities:
        print(f"  Ignorées : {fusion.ignored_modalities}")
    if fusion.warnings:
        for w in fusion.warnings:
            print(f"  ⚠  {w}")


def print_population_summary(
    spiral_hc: list[Path], spiral_pd: list[Path],
    drawing_pred: DrawingPredictor,
    hc_kbd_scores: list[float], pd_kbd_scores: list[float],
    hc_voice_df: pd.DataFrame, pd_voice_df: pd.DataFrame,
    voice_art: dict,
    n_voice: int = 20,
) -> None:
    W = 62
    print(f"\n\n{'='*W}")
    print("  Résumé population par modalité")
    print(f"{'='*W}")

    # Dessin
    print("\n  [ Dessin — dataset test spiral ]")
    for cls_name, imgs in [("healthy  ", spiral_hc), ("parkinson", spiral_pd)]:
        scores = [score_drawing(p, drawing_pred).score for p in imgs]
        scores = [s for s in scores if s is not None]
        if scores:
            print(
                f"    {cls_name}  n={len(scores):2d}  "
                f"mean={np.mean(scores):.3f}  "
                f"min={np.min(scores):.3f}  "
                f"max={np.max(scores):.3f}"
            )

    # Clavier
    print("\n  [ Clavier — NeuroQWERTY MIT-CS1PD ]")
    for cls_name, scores in [("HC", hc_kbd_scores), ("PD", pd_kbd_scores)]:
        if scores:
            print(
                f"    {cls_name}  n={len(scores):2d}  "
                f"mean={np.mean(scores):.3f}  "
                f"min={np.min(scores):.3f}  "
                f"max={np.max(scores):.3f}"
            )
        else:
            print(f"    {cls_name}  aucun score valide")

    # Voix
    print("\n  [ Voix — Sakar-2019-UCI-470 ]")
    for cls_name, vdf in [("HC", hc_voice_df), ("PD", pd_voice_df)]:
        scores = score_voice_population(vdf, voice_art, n=n_voice)
        if scores:
            print(
                f"    {cls_name}  n={len(scores):2d}  "
                f"mean={np.mean(scores):.3f}  "
                f"min={np.min(scores):.3f}  "
                f"max={np.max(scores):.3f}"
            )

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n  Validation fusion multimodale cross-dataset")
    print("  " + "─" * 58)

    # Vérification des chemins
    required = [
        SPIRAL_HC_DIR, SPIRAL_PD_DIR, KEYBOARD_ROOT,
        KEYBOARD_MODEL, VOICE_CSV, VOICE_MODEL,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("\nERREUR — ressources manquantes :")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    # Chargement des modèles
    print("\nChargement des modèles...", end=" ", flush=True)
    drawing_pred = DrawingPredictor()
    keyboard_art = joblib.load(KEYBOARD_MODEL)
    voice_art    = joblib.load(VOICE_MODEL)
    print("OK")

    # Sélection des échantillons
    hc_spirals, pd_spirals       = pick_spiral_images(n=5)
    hc_kbd_file, pd_kbd_file     = pick_keyboard_subjects("MIT-CS1PD")
    hc_voice_df, pd_voice_df     = load_voice_dataset()

    print("\nEchantillons sélectionnés :")
    print(f"  Dessin HC  : {hc_spirals[0].name}  |  PD : {pd_spirals[0].name}")
    print(f"  Clavier HC : {hc_kbd_file.name}")
    print(f"  Clavier PD : {pd_kbd_file.name}")
    print(f"  Voix HC    : {len(hc_voice_df)} enregistrements disponibles")
    print(f"  Voix PD    : {len(pd_voice_df)} enregistrements disponibles")

    # ── Patient HC ────────────────────────────────────────────────────────────
    print("\nScoring patient HC composite...", end=" ", flush=True)
    hc_drawing  = score_drawing(hc_spirals[0], drawing_pred)
    hc_keyboard = score_keyboard(hc_kbd_file, keyboard_art)
    hc_voice    = score_voice(hc_voice_df.iloc[0], voice_art)
    hc_fusion   = late_fusion([hc_drawing, hc_keyboard, hc_voice])
    print("OK")
    print_composite_patient(
        "Patient HC (composite cross-dataset)",
        [hc_drawing, hc_keyboard, hc_voice],
        hc_fusion,
    )

    # ── Patient PD ────────────────────────────────────────────────────────────
    print("\nScoring patient PD composite...", end=" ", flush=True)
    pd_drawing  = score_drawing(pd_spirals[0], drawing_pred)
    pd_keyboard = score_keyboard(pd_kbd_file, keyboard_art)
    pd_voice    = score_voice(pd_voice_df.iloc[0], voice_art)
    pd_fusion   = late_fusion([pd_drawing, pd_keyboard, pd_voice])
    print("OK")
    print_composite_patient(
        "Patient PD (composite cross-dataset)",
        [pd_drawing, pd_keyboard, pd_voice],
        pd_fusion,
    )

    # ── Résumé population ─────────────────────────────────────────────────────
    print("\nCalcul du résumé population (clavier : 5 sujets / classe)...", end=" ", flush=True)
    hc_kbd_scores, pd_kbd_scores = score_keyboard_subjects(
        "MIT-CS1PD", n_per_class=5, artifact=keyboard_art,
    )
    print("OK")
    print_population_summary(
        spiral_hc=hc_spirals,
        spiral_pd=pd_spirals,
        drawing_pred=drawing_pred,
        hc_kbd_scores=hc_kbd_scores,
        pd_kbd_scores=pd_kbd_scores,
        hc_voice_df=hc_voice_df,
        pd_voice_df=pd_voice_df,
        voice_art=voice_art,
        n_voice=20,
    )


if __name__ == "__main__":
    main()
