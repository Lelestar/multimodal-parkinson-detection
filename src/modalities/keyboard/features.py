"""Keyboard dynamics feature extraction from browser events."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "n_keystrokes",
    "duration_sec",
    "keys_per_min",
    "mean_hold",
    "std_hold",
    "median_hold",
    "iqr_hold",
    "q10_hold",
    "q90_hold",
    "skew_hold",
    "kurt_hold",
    "cv_hold",
    "mean_flight",
    "std_flight",
    "median_flight",
    "iqr_flight",
    "q10_flight",
    "q90_flight",
    "skew_flight",
    "kurt_flight",
    "cv_flight",
    "hold_to_flight",
    "long_hold_rate",
    "long_flight_rate",
    "space_punct_rate",
    "left_rate",
    "right_rate",
    "hand_switch_rate",
    "hand_entropy",
]

AGG_TIMING_XGB_FEATURE_COLUMNS = [
    "n_keystrokes",
    "duration_sec",
    "keys_per_min",
    "hold_to_flight",
    "long_hold_rate",
    "long_flight_rate",
    "hold_mean",
    "hold_std",
    "hold_median",
    "hold_iqr",
    "hold_q10",
    "hold_q90",
    "hold_cv",
    "hold_diff_mean",
    "hold_diff_std",
    "hold_diff_abs_mean",
    "hold_autocorr_lag1",
    "hold_autocorr_lag2",
    "hold_autocorr_lag5",
    "hold_autocorr_lag10",
    "flight_mean",
    "flight_std",
    "flight_median",
    "flight_iqr",
    "flight_q10",
    "flight_q90",
    "flight_cv",
    "flight_diff_mean",
    "flight_diff_std",
    "flight_diff_abs_mean",
    "flight_autocorr_lag1",
    "flight_autocorr_lag2",
    "flight_autocorr_lag5",
    "flight_autocorr_lag10",
]

LEFT_CODES = {
    "KeyQ",
    "KeyW",
    "KeyE",
    "KeyR",
    "KeyT",
    "KeyA",
    "KeyS",
    "KeyD",
    "KeyF",
    "KeyG",
    "KeyZ",
    "KeyX",
    "KeyC",
    "KeyV",
    "KeyB",
}
RIGHT_CODES = {
    "KeyY",
    "KeyU",
    "KeyI",
    "KeyO",
    "KeyP",
    "KeyH",
    "KeyJ",
    "KeyK",
    "KeyL",
    "KeyN",
    "KeyM",
}
PUNCT_OR_SPACE_CODES = {
    "Space",
    "Comma",
    "Period",
    "Semicolon",
    "Slash",
    "Minus",
    "Equal",
    "Quote",
    "Backquote",
    "BracketLeft",
    "BracketRight",
    "Backslash",
    "Enter",
}
EXCLUDED_CODES = {
    "Backspace",
    "Delete",
    "ShiftLeft",
    "ShiftRight",
    "ControlLeft",
    "ControlRight",
    "AltLeft",
    "AltRight",
    "MetaLeft",
    "MetaRight",
    "CapsLock",
    "Tab",
    "Escape",
}


def code_to_side(code: str | None) -> str:
    """Classify a browser key code by its approximate keyboard side."""
    if code in LEFT_CODES:
        return "left"
    if code in RIGHT_CODES:
        return "right"
    return "other"


def code_to_category(code: str | None) -> str:
    """Classify a browser key code as a letter, digit, punctuation/space, or other."""
    if code in PUNCT_OR_SPACE_CODES:
        return "space_punct"
    if code in LEFT_CODES or code in RIGHT_CODES:
        return "letter"
    if code and code.startswith("Digit"):
        return "digit"
    return "other"


def events_to_keystrokes(events: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Reconstruct complete keystrokes from keydown/keyup pairs."""
    active: dict[str, dict[str, Any]] = {}
    strokes: list[dict[str, Any]] = []

    normalized_events = sorted(
        (event for event in events if isinstance(event, dict)),
        key=lambda item: float(item.get("timestamp_ms", item.get("time", 0.0)) or 0.0),
    )
    for event in normalized_events:
        event_type = event.get("type")
        code = event.get("code") or event.get("key")
        if not code or code in EXCLUDED_CODES:
            continue
        timestamp_ms = event.get("timestamp_ms", event.get("time"))
        if timestamp_ms is None:
            continue
        try:
            timestamp_sec = float(timestamp_ms) / 1000.0
        except (TypeError, ValueError):
            continue

        if event_type == "keydown":
            if event.get("repeat") or code in active:
                continue
            active[code] = {
                "code": code,
                "key": event.get("key"),
                "press_time": timestamp_sec,
                "key_category": event.get("key_category") or code_to_category(code),
                "key_side": event.get("key_side") or code_to_side(code),
            }
        elif event_type == "keyup" and code in active:
            stroke = active.pop(code)
            release_time = timestamp_sec
            hold_time = release_time - float(stroke["press_time"])
            if 0.0 <= hold_time <= 5.0:
                stroke["release_time"] = release_time
                stroke["hold_time"] = hold_time
                strokes.append(stroke)

    if not strokes:
        return pd.DataFrame(columns=["code", "key", "press_time", "release_time", "hold_time", "key_category", "key_side"])

    frame = pd.DataFrame(strokes).sort_values("press_time").reset_index(drop=True)
    frame["flight_time"] = frame["press_time"].diff()
    frame.loc[frame["flight_time"] < 0, "flight_time"] = np.nan
    return frame


def clean_keystrokes(keystrokes: pd.DataFrame) -> pd.DataFrame:
    """Filter keystrokes with negative or implausible timings."""
    if keystrokes.empty:
        return keystrokes.copy()
    clean = keystrokes.copy()
    clean = clean[(clean["hold_time"].between(0.0, 5.0, inclusive="both"))]
    clean = clean[(clean["flight_time"].isna()) | (clean["flight_time"].between(0.0, 10.0, inclusive="both"))]
    clean = clean.sort_values("press_time").reset_index(drop=True)
    return clean


def _series_stats(values: pd.Series, prefix: str) -> dict[str, float]:
    """Compute descriptive statistics used by the model."""
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {
            f"mean_{prefix}": np.nan,
            f"std_{prefix}": np.nan,
            f"median_{prefix}": np.nan,
            f"iqr_{prefix}": np.nan,
            f"q10_{prefix}": np.nan,
            f"q90_{prefix}": np.nan,
            f"skew_{prefix}": np.nan,
            f"kurt_{prefix}": np.nan,
            f"cv_{prefix}": np.nan,
        }
    mean = float(values.mean())
    std = float(values.std())
    return {
        f"mean_{prefix}": mean,
        f"std_{prefix}": std,
        f"median_{prefix}": float(values.median()),
        f"iqr_{prefix}": float(values.quantile(0.75) - values.quantile(0.25)),
        f"q10_{prefix}": float(values.quantile(0.10)),
        f"q90_{prefix}": float(values.quantile(0.90)),
        f"skew_{prefix}": float(values.skew()) if len(values) >= 3 else 0.0,
        f"kurt_{prefix}": float(values.kurt()) if len(values) >= 4 else 0.0,
        f"cv_{prefix}": float(std / mean) if mean else 0.0,
    }


def _clean_signal(values: pd.Series, target_len: int) -> np.ndarray:
    """Return a fixed-length numeric signal with median imputation."""
    signal = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    median = signal.median()
    if pd.isna(median):
        median = 0.0
    array = signal.fillna(median).to_numpy(dtype=float)
    if len(array) == 0:
        array = np.zeros(target_len, dtype=float)
    if len(array) < target_len:
        array = np.pad(array, (0, target_len - len(array)), mode="edge")
    return array[:target_len]


def _agg_signal_stats(values: pd.Series, prefix: str, target_len: int = 300) -> dict[str, float]:
    """Compute the aggregate timing features used by the XGBoost model."""
    signal = _clean_signal(values, target_len=target_len)
    diff = np.diff(signal)
    mean = float(np.mean(signal))
    std = float(np.std(signal))
    features = {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_median": float(np.median(signal)),
        f"{prefix}_iqr": float(np.quantile(signal, 0.75) - np.quantile(signal, 0.25)),
        f"{prefix}_q10": float(np.quantile(signal, 0.10)),
        f"{prefix}_q90": float(np.quantile(signal, 0.90)),
        f"{prefix}_cv": float(std / mean) if mean else 0.0,
        f"{prefix}_diff_mean": float(np.mean(diff)),
        f"{prefix}_diff_std": float(np.std(diff)),
        f"{prefix}_diff_abs_mean": float(np.mean(np.abs(diff))),
    }
    for lag in [1, 2, 5, 10]:
        if len(signal) > lag and np.std(signal[:-lag]) > 0 and np.std(signal[lag:]) > 0:
            features[f"{prefix}_autocorr_lag{lag}"] = float(np.corrcoef(signal[:-lag], signal[lag:])[0, 1])
        else:
            features[f"{prefix}_autocorr_lag{lag}"] = 0.0
    return features


def extract_segment_features(segment: pd.DataFrame) -> dict[str, float]:
    """Extract model features for one keystroke segment."""
    if segment.empty:
        return {column: np.nan for column in FEATURE_COLUMNS}

    hold = pd.to_numeric(segment["hold_time"], errors="coerce").dropna()
    flight = pd.to_numeric(segment["flight_time"], errors="coerce").dropna()
    duration = float(segment["release_time"].max() - segment["press_time"].min())
    duration = max(duration, 0.0)

    features: dict[str, float] = {
        "n_keystrokes": float(len(segment)),
        "duration_sec": duration,
        "keys_per_min": float(len(segment) * 60.0 / duration) if duration > 0 else np.nan,
    }
    features.update(_series_stats(hold, "hold"))
    features.update(_series_stats(flight, "flight"))

    mean_hold = features["mean_hold"]
    mean_flight = features["mean_flight"]
    features["hold_to_flight"] = float(mean_hold / mean_flight) if mean_flight and not math.isnan(mean_flight) else np.nan
    features["long_hold_rate"] = float((hold > hold.quantile(0.90)).mean()) if len(hold) > 5 else np.nan
    features["long_flight_rate"] = float((flight > 1.0).mean()) if len(flight) > 5 else np.nan

    categories = segment.get("key_category", pd.Series(dtype=str)).fillna("other")
    sides = segment.get("key_side", pd.Series(dtype=str)).fillna("other")
    features["space_punct_rate"] = float((categories == "space_punct").mean())
    features["left_rate"] = float((sides == "left").mean())
    features["right_rate"] = float((sides == "right").mean())
    hand_sequence = sides[sides.isin(["left", "right"])].tolist()
    if len(hand_sequence) > 1:
        switches = sum(1 for previous, current in zip(hand_sequence, hand_sequence[1:]) if previous != current)
        features["hand_switch_rate"] = switches / (len(hand_sequence) - 1)
    else:
        features["hand_switch_rate"] = np.nan

    left_rate = features["left_rate"]
    right_rate = features["right_rate"]
    total_hand_rate = left_rate + right_rate
    if total_hand_rate > 0:
        p_left = left_rate / total_hand_rate
        p_right = right_rate / total_hand_rate
        entropy = 0.0
        for probability in (p_left, p_right):
            if probability > 0:
                entropy -= probability * math.log2(probability)
        features["hand_entropy"] = entropy
    else:
        features["hand_entropy"] = np.nan

    return {column: features.get(column, np.nan) for column in FEATURE_COLUMNS}


def extract_agg_timing_xgb_features(segment: pd.DataFrame, signal_len: int = 300) -> dict[str, float]:
    """Extract the aggregate timing features used by the final XGBoost candidate."""
    if segment.empty:
        return {column: np.nan for column in AGG_TIMING_XGB_FEATURE_COLUMNS}

    hold = pd.to_numeric(segment["hold_time"], errors="coerce")
    flight = pd.to_numeric(segment["flight_time"], errors="coerce")
    duration = float(segment["release_time"].max() - segment["press_time"].min())
    duration = max(duration, 0.0)
    hold_clean = hold.dropna()
    flight_clean = flight.dropna()

    features: dict[str, float] = {
        "n_keystrokes": float(len(segment)),
        "duration_sec": duration,
        "keys_per_min": float(len(segment) * 60.0 / duration) if duration > 0 else np.nan,
        "hold_to_flight": float(hold_clean.mean() / flight_clean.mean())
        if len(hold_clean) and len(flight_clean) and flight_clean.mean()
        else np.nan,
        "long_hold_rate": float((hold_clean > hold_clean.quantile(0.90)).mean()) if len(hold_clean) > 5 else np.nan,
        "long_flight_rate": float((flight_clean > 1.0).mean()) if len(flight_clean) > 5 else np.nan,
    }
    features.update(_agg_signal_stats(hold, "hold", target_len=signal_len))
    features.update(_agg_signal_stats(flight, "flight", target_len=signal_len))
    return {column: features.get(column, np.nan) for column in AGG_TIMING_XGB_FEATURE_COLUMNS}


def build_feature_table(
    keystrokes: pd.DataFrame,
    window_size: int = 300,
    stride: int = 150,
    min_segment_len: int = 120,
) -> pd.DataFrame:
    """Split one session into segments and return the feature table."""
    clean = clean_keystrokes(keystrokes)
    if len(clean) < min_segment_len:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    rows: list[dict[str, float]] = []
    start = 0
    while start < len(clean):
        segment = clean.iloc[start : start + window_size].copy()
        if len(segment) < min_segment_len:
            break
        rows.append(extract_segment_features(segment))
        if start + window_size >= len(clean):
            break
        start += stride

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def build_agg_timing_xgb_feature_table(
    keystrokes: pd.DataFrame,
    window_size: int = 300,
    stride: int = 150,
    min_segment_len: int = 300,
) -> pd.DataFrame:
    """Split one session into segments and return XGBoost aggregate timing features."""
    clean = clean_keystrokes(keystrokes)
    if len(clean) < min_segment_len:
        return pd.DataFrame(columns=AGG_TIMING_XGB_FEATURE_COLUMNS)

    rows: list[dict[str, float]] = []
    start = 0
    while start < len(clean):
        segment = clean.iloc[start : start + window_size].copy()
        if len(segment) < min_segment_len:
            break
        rows.append(extract_agg_timing_xgb_features(segment, signal_len=window_size))
        if start + window_size >= len(clean):
            break
        start += stride

    return pd.DataFrame(rows, columns=AGG_TIMING_XGB_FEATURE_COLUMNS)
