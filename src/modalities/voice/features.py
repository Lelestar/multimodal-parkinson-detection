"""Voice feature extraction matching UCI 174 (Max Little et al.) columns."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy.spatial.distance import pdist

FEATURE_NAMES: list[str] = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]

_F0_MIN = 75.0
_F0_MAX = 500.0
_PERIOD_FLOOR = 0.0001
_PERIOD_CEIL = 0.02
_MAX_PERIOD = 1.3
_MAX_AMP = 1.6


def _f0_features(pitch: parselmouth.Pitch) -> dict[str, float]:
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]
    if len(voiced) == 0:
        return {"MDVP:Fo(Hz)": float("nan"), "MDVP:Fhi(Hz)": float("nan"), "MDVP:Flo(Hz)": float("nan")}
    return {
        "MDVP:Fo(Hz)": float(np.mean(voiced)),
        "MDVP:Fhi(Hz)": float(np.max(voiced)),
        "MDVP:Flo(Hz)": float(np.min(voiced)),
    }


def _jitter_features(pp: parselmouth.Data) -> dict[str, float]:
    def _get(method: str) -> float:
        try:
            return float(call(pp, method, 0, 0, _PERIOD_FLOOR, _PERIOD_CEIL, _MAX_PERIOD))
        except Exception:
            return float("nan")

    rap = _get("Get jitter (rap)")
    return {
        "MDVP:Jitter(%)": _get("Get jitter (local)"),
        "MDVP:Jitter(Abs)": _get("Get jitter (local, absolute)"),
        "MDVP:RAP": rap,
        "MDVP:PPQ": _get("Get jitter (ppq5)"),
        "Jitter:DDP": 3.0 * rap if not np.isnan(rap) else float("nan"),
    }


def _shimmer_features(snd: parselmouth.Sound, pp: parselmouth.Data) -> dict[str, float]:
    def _get(method: str) -> float:
        try:
            return float(call([snd, pp], method, 0, 0, _PERIOD_FLOOR, _PERIOD_CEIL, _MAX_PERIOD, _MAX_AMP))
        except Exception:
            return float("nan")

    apq3 = _get("Get shimmer (apq3)")
    return {
        "MDVP:Shimmer": _get("Get shimmer (local)"),
        "MDVP:Shimmer(dB)": _get("Get shimmer (local_dB)"),
        "Shimmer:APQ3": apq3,
        "Shimmer:APQ5": _get("Get shimmer (apq5)"),
        "MDVP:APQ": _get("Get shimmer (apq11)"),
        "Shimmer:DDA": 3.0 * apq3 if not np.isnan(apq3) else float("nan"),
    }


def _hnr_features(snd: parselmouth.Sound) -> dict[str, float]:
    try:
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, _F0_MIN, 0.1, 1.0)
        hnr = float(call(harmonicity, "Get mean", 0, 0))
        nhr = float(10.0 ** (-hnr / 10.0)) if hnr > 0 else float("nan")
    except Exception:
        hnr, nhr = float("nan"), float("nan")
    return {"HNR": hnr, "NHR": nhr}


def _rpde(signal: np.ndarray, m: int = 4, tau: int = 1, epsilon_factor: float = 0.2, t_max: int = 80) -> float:
    """Recurrence Period Density Entropy (vectorised inner loop)."""
    std = float(np.std(signal))
    if std < 1e-10 or len(signal) < (m - 1) * tau + t_max + 2:
        return float("nan")
    epsilon = epsilon_factor * std
    n_pts = len(signal) - (m - 1) * tau
    embedded = np.column_stack([signal[d * tau: d * tau + n_pts] for d in range(m)])
    n_ref = min(n_pts - t_max, 500)
    periods: list[int] = []
    for i in range(n_ref):
        window = embedded[i + 1: i + t_max + 1] - embedded[i]
        dists = np.linalg.norm(window, axis=1)
        idx = np.argmax(dists < epsilon)
        if dists[idx] < epsilon:
            periods.append(int(idx + 1))
    if not periods:
        return float("nan")
    hist, _ = np.histogram(periods, bins=t_max, range=(1, t_max + 1))
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return float("nan")
    hist /= total
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)) / np.log2(t_max))


def _ppe(voiced_f0: np.ndarray, n_bins: int = 30) -> float:
    """Pitch Period Entropy."""
    if len(voiced_f0) < 4:
        return float("nan")
    median = float(np.median(voiced_f0))
    if median <= 0:
        return float("nan")
    log_f0 = np.log2(voiced_f0 / median)
    hist, _ = np.histogram(log_f0, bins=n_bins)
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return float("nan")
    hist /= total
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)) / np.log2(n_bins))


def _dfa(signal: np.ndarray, min_n: int = 4, n_scales: int = 10) -> float:
    """Detrended Fluctuation Analysis — returns Hurst-like exponent."""
    N = len(signal)
    max_n = N // 4
    if N < 2 * min_n or min_n >= max_n:
        return float("nan")
    y = np.cumsum(signal - np.mean(signal))
    scales = np.unique(np.round(np.logspace(np.log10(min_n), np.log10(max_n), n_scales)).astype(int))
    flucts, valid = [], []
    for n in scales:
        n = int(n)
        if n < 2 or N // n < 1:
            continue
        segs = [y[i * n: (i + 1) * n] for i in range(N // n)]
        rms_vals = []
        for seg in segs:
            x = np.arange(len(seg))
            trend = np.polyval(np.polyfit(x, seg, 1), x)
            rms_vals.append(float(np.sqrt(np.mean((seg - trend) ** 2))))
        flucts.append(float(np.mean(rms_vals)))
        valid.append(n)
    if len(valid) < 2:
        return float("nan")
    coeffs = np.polyfit(np.log(valid), np.log(flucts), 1)
    return float(coeffs[0])


def _corr_dim(signal: np.ndarray, m: int = 4, tau: int = 1, n_sample: int = 200) -> float:
    """Correlation dimension (Grassberger-Procaccia, simplified)."""
    N = len(signal)
    n_pts = N - (m - 1) * tau
    if n_pts < 20:
        return float("nan")
    embedded = np.column_stack([signal[d * tau: d * tau + n_pts] for d in range(m)])
    idx = np.random.default_rng(0).choice(n_pts, min(n_pts, n_sample), replace=False)
    dists = pdist(embedded[idx])
    dists = dists[dists > 0]
    if len(dists) == 0:
        return float("nan")
    eps_vals = np.logspace(np.log10(np.percentile(dists, 5)), np.log10(np.percentile(dists, 50)), 10)
    log_C, log_eps = [], []
    for eps in eps_vals:
        c = float(np.mean(dists < eps))
        if c > 0:
            log_C.append(np.log(c))
            log_eps.append(np.log(eps))
    if len(log_C) < 3:
        return float("nan")
    return float(np.polyfit(log_eps, log_C, 1)[0])


def _nonlinear_features(snd: parselmouth.Sound, voiced_f0: np.ndarray) -> dict[str, float]:
    signal = snd.values.flatten().astype(float)
    signal_ds = signal[::8] if len(signal) > 8000 else signal

    dfa_val = _dfa(signal_ds) if len(signal_ds) >= 100 else float("nan")
    d2_val = _corr_dim(voiced_f0) if len(voiced_f0) >= 20 else float("nan")

    rpde_val = _rpde(signal_ds)

    # spread1 approximates mean(log(1/F0)); spread2 approximates std(log(F0)).
    if len(voiced_f0) >= 4:
        log_f0 = np.log(voiced_f0)
        spread1 = float(-np.mean(log_f0))
        spread2 = float(np.std(log_f0))
    else:
        spread1, spread2 = float("nan"), float("nan")

    ppe_val = _ppe(voiced_f0)

    return {
        "RPDE": rpde_val,
        "DFA": dfa_val,
        "spread1": spread1,
        "spread2": spread2,
        "D2": d2_val,
        "PPE": ppe_val,
    }


def extract_voice_features(audio_path: str | Path) -> dict[str, float]:
    """Extract 22 voice features from a WAV file matching UCI 174 columns."""
    snd = parselmouth.Sound(str(audio_path))
    if snd.n_channels > 1:
        snd = snd.convert_to_mono()

    pitch = call(snd, "To Pitch", 0.0, _F0_MIN, _F0_MAX)
    voiced_f0 = pitch.selected_array["frequency"]
    voiced_f0 = voiced_f0[voiced_f0 > 0]

    pp = call(snd, "To PointProcess (periodic, cc)", _F0_MIN, _F0_MAX)

    features: dict[str, float] = {}
    features.update(_f0_features(pitch))
    features.update(_jitter_features(pp))
    features.update(_shimmer_features(snd, pp))
    features.update(_hnr_features(snd))
    features.update(_nonlinear_features(snd, voiced_f0))

    return {name: features.get(name, float("nan")) for name in FEATURE_NAMES}
