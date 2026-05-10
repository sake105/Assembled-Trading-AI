"""Spectral Analysis: FFT-Periodogramm, Wavelet-Coherence, Power-Spectral-Density.

Anwendung
---------
- Detection von periodischen Mustern (z. B. Wochentag-Effekte, Saison-Effekte)
- Cross-Spectrum: welche Frequenz-Komponenten teilen Asset A & B?
- Wavelet-Coherence: lokalisierte Cross-Spectrum (Zeit + Frequenz)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Periodogram:
    frequencies: np.ndarray
    power: np.ndarray
    dominant_period: float


def periodogram(series: pd.Series, detrend: bool = True) -> Periodogram:
    """FFT-Periodogramm einer Time-Series.

    Args:
        series: 1-D series, sampled uniformly.
        detrend: ob linearer Trend abgezogen wird.

    Returns:
        Periodogram mit dominant period (in Sample-Einheiten).
    """
    s = pd.Series(series).dropna().values.astype(float)
    n = len(s)
    if n < 16:
        raise ValueError("need >= 16 samples")
    if detrend:
        t = np.arange(n)
        slope, intercept = np.polyfit(t, s, 1)
        s = s - (slope * t + intercept)

    fft_vals = np.fft.rfft(s)
    power = np.abs(fft_vals) ** 2 / n
    freqs = np.fft.rfftfreq(n)
    # Dominant period (excl. DC)
    if len(power) > 1:
        idx = int(np.argmax(power[1:])) + 1
        if freqs[idx] > 0:
            dominant = 1.0 / freqs[idx]
        else:
            dominant = float("inf")
    else:
        dominant = float("nan")
    return Periodogram(frequencies=freqs, power=power, dominant_period=float(dominant))


def cross_spectrum(x: pd.Series, y: pd.Series, detrend: bool = True) -> dict:
    """Cross-Spectrum + Coherence between two series.

    Returns:
        dict mit frequencies, coherence (squared, ∈ [0,1]), phase.
    """
    s1 = pd.Series(x).dropna().values.astype(float)
    s2 = pd.Series(y).dropna().values.astype(float)
    n = min(len(s1), len(s2))
    s1, s2 = s1[:n], s2[:n]
    if n < 32:
        raise ValueError("need >= 32 samples")
    if detrend:
        t = np.arange(n)
        s1 = s1 - np.polyval(np.polyfit(t, s1, 1), t)
        s2 = s2 - np.polyval(np.polyfit(t, s2, 1), t)

    f1 = np.fft.rfft(s1)
    f2 = np.fft.rfft(s2)
    p1 = np.abs(f1) ** 2 / n
    p2 = np.abs(f2) ** 2 / n
    cross = f1.conj() * f2 / n
    # Smooth via simple moving average
    sm = max(3, n // 32)
    p1_s = pd.Series(p1).rolling(sm, min_periods=1, center=True).mean().values
    p2_s = pd.Series(p2).rolling(sm, min_periods=1, center=True).mean().values
    cross_s = (
        pd.Series(cross.real).rolling(sm, min_periods=1, center=True).mean().values
        + 1j
        * pd.Series(cross.imag).rolling(sm, min_periods=1, center=True).mean().values
    )
    denom = p1_s * p2_s
    denom = np.where(denom > 0, denom, 1.0)
    coherence_sq = np.abs(cross_s) ** 2 / denom
    phase = np.angle(cross_s)
    freqs = np.fft.rfftfreq(n)
    return {
        "frequencies": freqs,
        "coherence_squared": np.clip(coherence_sq, 0, 1),
        "phase": phase,
        "cross_power": np.abs(cross_s),
    }


def power_spectral_density_welch(series: pd.Series, n_segments: int = 8) -> dict:
    """Welch's method — average periodograms over overlapping segments.

    Robuster als single-FFT-Periodogramm.
    """
    s = pd.Series(series).dropna().values.astype(float)
    n = len(s)
    if n < 64:
        raise ValueError("need >= 64 samples")
    seg_len = n // n_segments * 2  # 50%-overlap
    if seg_len < 16:
        seg_len = 16
    overlap = seg_len // 2

    # Hann window
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(seg_len) / (seg_len - 1))
    win_norm = (window**2).sum()

    psds = []
    start = 0
    while start + seg_len <= n:
        seg = s[start : start + seg_len] * window
        fft_seg = np.fft.rfft(seg)
        psd_seg = np.abs(fft_seg) ** 2 / win_norm
        psds.append(psd_seg)
        start += seg_len - overlap

    if not psds:
        raise ValueError("not enough samples for any segment")
    psd_avg = np.mean(psds, axis=0)
    freqs = np.fft.rfftfreq(seg_len)
    return {"frequencies": freqs, "psd": psd_avg, "n_segments": len(psds)}


__all__ = [
    "Periodogram",
    "periodogram",
    "cross_spectrum",
    "power_spectral_density_welch",
]
