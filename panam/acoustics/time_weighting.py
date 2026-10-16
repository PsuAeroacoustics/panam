"""IEC 61672-1 frequency and time weighting of a pressure signal.

Frequency weightings A, C and Z are digital filters designed from the analog
poles of IEC 61672-1:2013, Annex E.  Time weightings F (Fast, 125 ms) and
S (Slow, 1 s) are the standard's exponential averages of the squared weighted
pressure; I (Impulse, 35 ms rise and 1.5 s decay) follows the withdrawn
IEC 60651 and is kept for comparison with older meters.

Pressures are in Pa, sample rates in Hz, times in s and levels in dB re
``p_ref`` (20 uPa by default).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import bilinear_zpk, lfilter, sosfilt, zpk2sos

# Pole frequencies of IEC 61672-1:2013, Annex E (Hz).
F1 = 20.598997
F2 = 107.65265
F3 = 737.86223
F4 = 12194.217

# Exponential time constants (s).
TIME_CONSTANTS = {'F': 0.125, 'S': 1.0}
# Impulse: rise and decay time constants of IEC 60651 (s).
IMPULSE_RISE = 0.035
IMPULSE_DECAY = 1.5


def _check_fs(fs):
    """Raise ValueError unless the sample rate ``fs`` (Hz) is positive and finite."""
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError('fs must be positive and finite')


def _analog_zpk(kind):
    """Analog zeros and poles (rad/s) of weighting ``kind``, unnormalized."""
    w = [2.0 * np.pi * f for f in (F1, F2, F3, F4)]
    if kind == 'A':
        zeros = np.zeros(4)
        poles = -np.array([w[0], w[0], w[1], w[2], w[3], w[3]])
    elif kind == 'C':
        zeros = np.zeros(2)
        poles = -np.array([w[0], w[0], w[3], w[3]])
    else:
        raise ValueError("kind must be 'A', 'C' or 'Z'")
    return zeros, poles


def frequency_weighting_sos(kind, fs):
    """Second-order sections of IEC 61672-1 frequency weighting A, C or Z.

    The analog zeros and poles of IEC 61672-1:2013, Annex E (poles at
    20.598997, 107.65265, 737.86223 and 12194.217 Hz) are mapped to the z
    plane by the bilinear transform without pre-warping, and the gain is set
    so that the digital response is exactly 0 dB at 1 kHz.  The response
    meets the class 1 acceptance limits of Table 3 for nominal frequencies up
    to ``fs/4``, and from 10 Hz to 20 kHz for ``fs`` of 44.1 kHz and above.
    Pre-warping the 12194 Hz poles raises the response 0.5 to 4 dB above the
    standard at 4 to 10 kHz for ``fs`` of 25 to 32 kHz (at 16 kHz the pole
    lies above the Nyquist frequency and cannot be pre-warped), and matched-z reads
    up to 4.5 dB high near the Nyquist frequency and fails at 12.5 kHz even
    at 48 kHz; the bilinear transform's droop toward the Nyquist frequency
    stays within the larger negative tolerances there.

    Args:
        kind: 'A', 'C' or 'Z' (Z is flat: one pass-through section).
        fs: sample rate, Hz.  A and C need ``fs`` above 2 kHz, so that the
            1 kHz normalization frequency lies below the Nyquist frequency.

    Returns:
        (n_sections, 6) array for ``scipy.signal.sosfilt``.

    Raises:
        ValueError: ``fs`` not positive and finite, or not above 2 kHz for
            A and C; unknown ``kind``.
    """
    _check_fs(fs)
    if kind == 'Z':
        return np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    zeros, poles = _analog_zpk(kind)
    if fs <= 2000.0:
        raise ValueError('fs must be above 2000 Hz for A and C weighting '
                         '(the 1 kHz normalization must lie below the Nyquist frequency)')
    z, p, _ = bilinear_zpk(zeros, poles, 1.0, fs)
    e = np.exp(2j * np.pi * 1000.0 / fs)
    gain = 1.0 / np.abs(np.prod(e - z) / np.prod(e - p))
    return zpk2sos(z, p, gain)


def apply_frequency_weighting(p, fs, kind='A'):
    """Frequency-weighted pressure, filtered causally along the last axis.

    Args:
        p: pressure, Pa; the last axis is time.
        fs: sample rate, Hz.
        kind: 'A' (default), 'C' or 'Z'.

    Returns:
        weighted pressure, Pa, same shape as ``p``.  The filter starts from
        rest, so the first tenth of a second or so carries its transient (the
        double pole at 20.6 Hz has a time constant of 7.7 ms).
    """
    p = np.asarray(p, dtype=float)
    if kind == 'Z':
        _check_fs(fs)
        return p.copy()
    return sosfilt(frequency_weighting_sos(kind, fs), p, axis=-1)


def _exponential(x, fs, tau):
    """Exponential average of ``x`` with time constant ``tau`` (s), from zero."""
    a = np.exp(-1.0 / (fs * tau))
    return lfilter([1.0 - a], [1.0, -a], x, axis=-1)


def time_weighted_square(p, fs, time='F'):
    """Time-weighted mean-square pressure of a (weighted) pressure signal.

    F and S are the exponential averages of IEC 61672-1, 3.9,
    (1/tau) * integral p^2(xi) exp(-(t - xi)/tau) dxi, with tau = 0.125 s and
    1 s, discretized as y[n] = a y[n-1] + (1 - a) p^2[n], a = exp(-1/(fs tau)).
    I is the IEC 60651 impulse detector: a 35 ms exponential average followed
    by a peak hold that decays with a 1.5 s time constant.  Every average
    starts from zero.

    Args:
        p: pressure, Pa; the last axis is time.
        fs: sample rate, Hz.
        time: 'F' (default), 'S' or 'I'.

    Returns:
        mean-square pressure, Pa^2, same shape as ``p``.
    """
    _check_fs(fs)
    p2 = np.asarray(p, dtype=float) ** 2
    if time in TIME_CONSTANTS:
        return _exponential(p2, fs, TIME_CONSTANTS[time])
    if time != 'I':
        raise ValueError("time must be 'F', 'S' or 'I'")
    rise = _exponential(p2, fs, IMPULSE_RISE)
    # Peak hold with exponential decay: y[n] = max(rise[n], y[n-1] d), d =
    # exp(-1/(fs IMPULSE_DECAY)).  In logs this is a running maximum of
    # ln rise[k] + k/(fs IMPULSE_DECAY), less n/(fs IMPULSE_DECAY).
    ramp = np.arange(p2.shape[-1]) / (fs * IMPULSE_DECAY)
    with np.errstate(divide='ignore'):
        held = np.maximum.accumulate(np.log(rise) + ramp, axis=-1) - ramp
    return np.exp(held)


def _level(mean_square, p_ref):
    if not p_ref > 0:
        raise ValueError('p_ref must be positive')
    with np.errstate(divide='ignore'):
        return 10.0 * np.log10(mean_square / p_ref ** 2)


def time_weighted_level(p, fs, frequency='A', time='F', p_ref=20e-6):
    """Frequency- and time-weighted sound level history and its maximum.

    The pressure is filtered by :func:`apply_frequency_weighting` and averaged
    by :func:`time_weighted_square`, both starting from rest, so the first
    second or so (5 tau) reads low.

    A NaN (no data) in ``p`` is carried by the recursive filters to every
    later sample, so ``level`` is NaN from there on.  ``lmax`` and ``t_max``
    are then NaN too: a maximum taken over the samples before the gap would
    not be the maximum of the record.

    Args:
        p: pressure history, Pa, one-dimensional at equal intervals 1/fs.
        fs: sample rate, Hz.
        frequency: frequency weighting 'A' (default), 'C' or 'Z'.
        time: time weighting 'F' (default, 125 ms), 'S' (1 s) or 'I'.
        p_ref: reference pressure, Pa (default 20e-6).

    Returns: dict with
        time: sample times, s, from 0 at the first sample.
        level: time-weighted level (e.g. LAF), dB re p_ref, at each sample;
            -inf where the averaged energy is zero.
        lmax: maximum of ``level`` (e.g. LAFmax), dB; NaN when ``p`` has NaN.
        t_max: time of the maximum, s (the first one when it repeats); NaN
            when ``p`` has NaN.
    """
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not p.size:
        raise ValueError('p must be a nonempty one-dimensional history')
    level = _level(time_weighted_square(apply_frequency_weighting(p, fs, frequency), fs, time),
                   p_ref)
    t = np.arange(p.size) / float(fs)
    if np.isnan(level).any():
        return {'time': t, 'level': level, 'lmax': np.nan, 't_max': np.nan}
    k = int(np.argmax(level))
    return {'time': t, 'level': level, 'lmax': float(level[k]), 't_max': float(t[k])}


def leq(p, fs, frequency='A', duration=None, p_ref=20e-6):
    """Equivalent continuous sound level (e.g. LAeq,T) of a pressure history.

    Leq,T = 10 lg( (1/T) sum p_w^2[n] / fs / p_ref^2 ), with p_w the
    frequency-weighted pressure (IEC 61672-1, 3.10).  Energy outside the
    record counts as zero, so a ``duration`` longer than the record gives the
    level of the event spread over T.

    Args:
        p: pressure history, Pa, one-dimensional at equal intervals 1/fs.
        fs: sample rate, Hz.
        frequency: frequency weighting 'A' (default), 'C' or 'Z'.
        duration: averaging time T, s; default (None) the record length
            len(p)/fs.  It may not be shorter than the record.
        p_ref: reference pressure, Pa (default 20e-6).

    Returns:
        Leq,T, dB re p_ref; -inf for a silent record, NaN when ``p`` has NaN.
    """
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not p.size:
        raise ValueError('p must be a nonempty one-dimensional history')
    _check_fs(fs)
    record = p.size / float(fs)
    T = record if duration is None else float(duration)
    if not np.isfinite(T) or T < record * (1.0 - 1e-12):
        raise ValueError('duration must be finite and at least the record length len(p)/fs')
    pw = apply_frequency_weighting(p, fs, frequency)
    return float(_level(np.sum(pw ** 2) / fs / T, p_ref))
