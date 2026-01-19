"""Minimal filtering helpers using SciPy SOS filters.

Ported behavior from python-acoustics (BSD-3-Clause) but implemented
with SciPy's public sosfilt/sosfiltfilt APIs.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt


def _butter_sos(order: int, cutoff: float, fs: float, btype: str):
    if fs <= 0:
        raise ValueError("fs must be positive")
    nyq = fs / 2.0
    if cutoff <= 0 or cutoff >= nyq:
        raise ValueError("cutoff must be between 0 and Nyquist")
    return butter(order, cutoff / nyq, btype=btype, output="sos")


def lowpass(signal, cutoff, fs, order: int = 4, zero_phase: bool = False):
    """Filter signal with a low-pass Butterworth filter.

    Args:
        signal: Input signal array.
        cutoff: Cutoff frequency (Hz).
        fs: Sampling rate (Hz).
        order: Filter order.
        zero_phase: If True, use forward-backward filtering.
    """
    sos = _butter_sos(order, cutoff, fs, btype="low")
    x = np.asarray(signal)
    if zero_phase:
        return sosfiltfilt(sos, x)
    return sosfilt(sos, x)


def highpass(signal, cutoff, fs, order: int = 4, zero_phase: bool = False):
    """Filter signal with a high-pass Butterworth filter.

    Args:
        signal: Input signal array.
        cutoff: Cutoff frequency (Hz).
        fs: Sampling rate (Hz).
        order: Filter order.
        zero_phase: If True, use forward-backward filtering.
    """
    sos = _butter_sos(order, cutoff, fs, btype="high")
    x = np.asarray(signal)
    if zero_phase:
        return sosfiltfilt(sos, x)
    return sosfilt(sos, x)
