"""Wavelet-Decomposition für Multi-Scale-Analyse.

Anwendung
---------
Wavelet-Decomposition trennt eine Time-Series in Frequency-Bands:
- niedrige Frequenz = Trend
- mittlere Frequenz = Cyclical
- hohe Frequenz = Noise / Microstructure

Wavelet-Energy je Scale ist nützlich als Regime-Indikator (Boom/Bust haben
unterschiedliche Energy-Verteilungen).

Reference
---------
- Mallat, S. (1989). A theory for multiresolution signal decomposition.
- Gencay, R., Selcuk, F. & Whitcher, B. (2001). *An Introduction to Wavelets*.

Implementation
--------------
Erfordert PyWavelets (``pip install PyWavelets``). Falls nicht verfügbar,
fällt zurück auf einfache Haar-Transform (NumPy-only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _haar_dwt(signal: np.ndarray, level: int) -> list[np.ndarray]:
    """NumPy-only Haar discrete wavelet transform."""
    coeffs: list[np.ndarray] = []
    s = signal.astype(float).copy()
    for _ in range(level):
        if len(s) < 2:
            break
        avg = (s[::2] + s[1::2]) / np.sqrt(2)
        det = (s[::2] - s[1::2]) / np.sqrt(2)
        coeffs.insert(0, det)
        s = avg
    coeffs.insert(0, s)  # final approximation
    return coeffs


def wavelet_decompose(series: pd.Series, wavelet: str = "db4", level: int = 4) -> dict:
    """Multi-Level Discrete Wavelet Transform.

    Args:
        series: 1-D pandas Series.
        wavelet: PyWavelets wavelet name (z. B. 'db4', 'haar', 'sym8').
        level: decomposition level.

    Returns:
        Dict ``{'approximation': cA_n, 'details': [cD_1, cD_2, ..., cD_n]}``.
    """
    s = pd.Series(series).dropna().values
    if len(s) < 2**level:
        raise ValueError(f"need >= {2**level} samples")
    try:
        import pywt  # type: ignore

        coeffs = pywt.wavedec(s, wavelet, level=level)
        return {
            "approximation": coeffs[0],
            "details": list(coeffs[1:]),
        }
    except ImportError:
        haar = _haar_dwt(s, level)
        return {
            "approximation": haar[0],
            "details": list(haar[1:]),
        }


def wavelet_energy_per_scale(decomposition: dict) -> dict:
    """Energy = Σ coef²  je scale (Approximation + Details).

    Returns:
        Dict ``{'approx_energy': float, 'detail_energies': [float]}`` —
        und Anteil je Scale.
    """
    approx = decomposition["approximation"]
    details = decomposition["details"]
    e_approx = float(np.sum(approx**2))
    e_details = [float(np.sum(d**2)) for d in details]
    total = e_approx + sum(e_details)
    return {
        "approx_energy": e_approx,
        "detail_energies": e_details,
        "approx_share": e_approx / total if total > 0 else float("nan"),
        "detail_shares": [e / total if total > 0 else float("nan") for e in e_details],
        "total_energy": total,
    }


def reconstruct_from_band(
    decomposition: dict, level_to_keep: int, wavelet: str = "db4"
) -> np.ndarray:
    """Rekonstruiere Signal nur aus einem Frequency-Band.

    level_to_keep = -1 bedeutet nur Approximation (Trend).
    level_to_keep = 0 bedeutet nur höchste Detail-Komponente.
    """
    try:
        import pywt  # type: ignore

        coeffs = [decomposition["approximation"]] + list(decomposition["details"])
        new_coeffs = [np.zeros_like(c) for c in coeffs]
        if level_to_keep == -1:
            new_coeffs[0] = coeffs[0]
        else:
            idx = level_to_keep + 1
            if 0 < idx < len(coeffs):
                new_coeffs[idx] = coeffs[idx]
        return pywt.waverec(new_coeffs, wavelet)
    except ImportError:
        # Haar fallback: simply replicate approximation level
        return decomposition["approximation"].copy()


def rolling_wavelet_energy_ratio(
    series: pd.Series,
    window: int = 256,
    wavelet: str = "db4",
    level: int = 4,
    target_level: int = 0,
) -> pd.Series:
    """Rolling-Energy-Anteil eines bestimmten Wavelet-Levels.

    Niedrige hochfrequenz-Energie (target_level=0 = highest freq) =
    glattere Phase. Hohe HF-Energie = Stress / Whipsaw.
    """
    s = pd.Series(series).dropna()
    out = pd.Series(np.nan, index=s.index)
    for end in range(window, len(s) + 1):
        sub = s.iloc[end - window : end].values
        if len(sub) < 2**level:
            continue
        try:
            decomp = wavelet_decompose(pd.Series(sub), wavelet=wavelet, level=level)
            energy = wavelet_energy_per_scale(decomp)
            shares = energy["detail_shares"]
            if target_level < len(shares):
                out.iloc[end - 1] = shares[target_level]
        except Exception:  # noqa: BLE001
            continue
    return out


__all__ = [
    "wavelet_decompose",
    "wavelet_energy_per_scale",
    "reconstruct_from_band",
    "rolling_wavelet_energy_ratio",
]
