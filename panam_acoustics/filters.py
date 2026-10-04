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


def _butter_filter(signal, cutoff, fs, order, zero_phase, btype):
    sos = _butter_sos(order, cutoff, fs, btype=btype)
    x = np.asarray(signal)
    if zero_phase:
        return sosfiltfilt(sos, x)
    return sosfilt(sos, x)


def lowpass(signal, cutoff, fs, order: int = 4, zero_phase: bool = False):
    """Filter signal with a low-pass Butterworth filter.

    Args:
        signal: Input signal array.
        cutoff: Cutoff frequency (Hz).
        fs: Sampling rate (Hz).
        order: Filter order.
        zero_phase: If True, use forward-backward filtering.
    """
    return _butter_filter(signal, cutoff, fs, order, zero_phase, btype="low")


def highpass(signal, cutoff, fs, order: int = 4, zero_phase: bool = False):
    """Filter signal with a high-pass Butterworth filter.

    Args:
        signal: Input signal array.
        cutoff: Cutoff frequency (Hz).
        fs: Sampling rate (Hz).
        order: Filter order.
        zero_phase: If True, use forward-backward filtering.
    """
    return _butter_filter(signal, cutoff, fs, order, zero_phase, btype="high")


def third_octave_filter_bank(signal, fs, band_centers, frame_centers, frame_length,
                             order: int = 3):
    """Mean-square output of a one-third octave filter bank, frame by frame.

    Each band is an order-``order`` Butterworth band-pass between
    ``fc * 2**(-1/6)`` and ``fc * 2**(1/6)``, the usual realization of an
    IEC 61260-1 class 1 filter and what analyzers implement.  It is applied
    causally, as an analyzer does, and each band's output is then advanced by
    its group delay at ``fc`` so that energy stays attached to the time it
    arrived; that delay is about 0.14 s at 20 Hz and 0.3 s at 10 Hz, which would
    otherwise shift low-band energy onto the wrong emission angles.

    Bands are filtered at a rate decimated octave by octave to about 12 times
    their upper edge, which keeps the low-band filters numerically sound.  The
    signal is reflected at both ends before filtering so start-up transients
    fall outside the record.

    The squared output is averaged with the same Hann-squared weighting a
    Hann-windowed PSD frame applies, so for a frame of ``frame_length``
    samples centerd at ``frame_centers`` the result is directly comparable
    with a PSD summed over the band: only the filter shape differs.

    Args:
        signal: 1-D pressure signal.
        fs: Sampling rate (Hz).
        band_centers: Band center frequencies (Hz).
        frame_centers: Frame center times (s) from the start of ``signal``.
        frame_length: Frame length (samples at ``fs``).
        order: Butterworth order of each band-pass.

    Returns:
        (Nbands, Nframes) mean-square band pressure, in the signal's units
        squared.  Bands reaching the Nyquist frequency are returned as NaN.
    """
    from scipy.signal import decimate, get_window, oaconvolve, sosfreqz

    x = np.asarray(signal, dtype=float)
    x = x - x.mean()
    band_centers = np.asarray(band_centers, dtype=float).ravel()
    frame_centers = np.asarray(frame_centers, dtype=float).ravel()
    out = np.full((band_centers.size, frame_centers.size), np.nan)

    decimated = {0: x}

    def at_rate(m):
        if m not in decimated:
            below = max(k for k in decimated if k < m)
            y = decimated[below]
            for _ in range(m - below):
                y = decimate(y, 2, ftype='fir', zero_phase=True)
            decimated[m] = y
        return decimated[m]

    for i, fc in enumerate(band_centers):
        f_lower, f_upper = fc / 2.0 ** (1.0 / 6.0), fc * 2.0 ** (1.0 / 6.0)
        if f_upper >= 0.98 * fs / 2.0:
            continue
        m = 0
        while fs / 2.0 ** (m + 1) >= 12.0 * f_upper and x.size / 2.0 ** (m + 1) > 64:
            m += 1
        y = at_rate(m)
        fs_m = fs / 2.0 ** m
        sos = butter(order, [f_lower, f_upper], btype='bandpass', fs=fs_m, output='sos')

        # Group delay at fc, from the phase slope.
        w = np.array([0.999, 1.001]) * fc
        _, h = sosfreqz(sos, worN=w, fs=fs_m)
        delay = -np.diff(np.unwrap(np.angle(h)))[0] / (2.0 * np.pi * (w[1] - w[0]))
        shift = int(round(delay * fs_m))

        # Reflect-pad by several filter time constants either side.
        pad = min(y.size - 1, int(np.ceil(6.0 * fs_m / (f_upper - f_lower))) + shift)
        padded = np.concatenate((2 * y[0] - y[pad:0:-1], y, 2 * y[-1] - y[-2:-pad - 2:-1]))
        squared = sosfilt(sos, padded)[pad + shift: pad + shift + y.size] ** 2
        if squared.size < y.size:
            squared = np.concatenate((squared, np.zeros(y.size - squared.size)))

        length = max(int(round(frame_length / 2.0 ** m)), 1)
        weights = get_window('hann', length) ** 2
        weights /= weights.sum()
        smoothed = oaconvolve(squared, weights[::-1], mode='same')
        index = np.clip(np.round(frame_centers * fs_m).astype(int), 0, y.size - 1)
        out[i] = smoothed[index]
    return out
