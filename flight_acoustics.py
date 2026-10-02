# coding=UTF-8
import os
import warnings
from configparser import ConfigParser
from glob import glob

import numpy as np
from panam_acoustics.atmosphere import Atmosphere
import h5py
from typing import Any, Optional, cast
import openpyxl
import scipy.signal
from scipy.interpolate import RegularGridInterpolator
from scipy.special import erf
import simplekml
# Colormap helper will import palettable lazily
import matplotlib
from matplotlib import tri
from matplotlib.pyplot import plot, subplots, colorbar, style, contourf, show
from netCDF4 import Dataset
from pyuff import UFF
from pymap3d import geodetic2enu, enu2geodetic

import unit_conversion
from panam_acoustics import filters as pa_filters

style.use('fivethirtyeight')
matplotlib.rcParams.update({'mathtext.fontset': 'dejavuserif'})
matplotlib.rcParams.update({'figure.autolayout': True})


def psd(signal, sampling_rate, cal=0.0):
    """
    Compute the acoustic power spectral density of a signal

    Args:
        signal: Array-like acoustic signal
        sampling_rate: Sampling rate of signal, Hz
        cal: Optional calibration factor to apply to signal (dB)
    Returns: tuple (frequency, psd_db, level)
           WHERE
           frequency is an array of band frequencies
           psd_db is the power spectral density in dB/Hz**2
           level is the integrated sound pressure level over all bands in dB
    """
    kcal = 10 ** (cal / 20)
    frequency, power_spectral_density = scipy.signal.periodogram(kcal * signal, sampling_rate)
    df = frequency[1] - frequency[0]
    pref = 2.0e-5
    psd_db = 10.0 * np.log10(power_spectral_density / (pref ** 2))
    level = 20.0 * np.log10(np.sqrt(np.sum(power_spectral_density * df)) / pref)
    return frequency, psd_db, level


def psd_welch(signal, sampling_rate, cal=0.0, window_time=1.0, window_type='hann', window_overlap=0.5, medfilter=None, passband = None):
    """
    Compute the acoustic power spectral density of a signal

    Args:
        signal: Array-like acoustic signal
        sampling_rate: Sampling rate of signal, Hz
        cal: Optional calibration factor to apply to signal (dB)
        window_time: duration of windows, s
        window_type: optional type of window to us, see scipy.signal.window, default="hann"
        window_overlap: optional proportion of overlap for windows, default=0.5
        medfilter: width (in Hz) of the median filter, if None, no filter applied (default)
        passband: pair of values (in Hz) defining the lower and upper regions of the passband.  Default None.
    Returns: tuple (frequency, psd_db, level)
           WHERE
           frequency is an array of band frequencies
           psd_db is the power spectral density in dB/Hz**2
           level is the integrated sound pressure level over all bands in dB
           level_A is the A-weighted integrated sound pressure level over all bands in dB
    """
    kcal = 10 ** (cal / 20)
    binwidth = int(2.0 ** nextpow2(window_time * sampling_rate))
    window = scipy.signal.get_window(window_type, binwidth)
    frequency, power_spectral_density = scipy.signal.welch(kcal * signal, sampling_rate,
                                                           window=window, noverlap=int(binwidth * window_overlap))
    if passband is not None:
        pass_indicies = np.logical_and(frequency >= passband[0], frequency <= passband[1])
        power_spectral_density = power_spectral_density[pass_indicies]
        frequency = frequency[pass_indicies]
    df = frequency[1] - frequency[0]
    if medfilter is not None:
        # medfilter is a width in Hz; medfilt wants an odd number of bins.
        medfilter_width = 2 * int(round(0.5 * float(medfilter) / df)) + 1
        power_spectral_density = scipy.signal.medfilt(power_spectral_density, medfilter_width)
    pref = 2.0e-5
    psd_db = 10.0 * np.log10(power_spectral_density / (pref ** 2))
    level = 20.0 * np.log10(np.sqrt(np.sum(power_spectral_density * df)) / pref)
    # dBAw is a level (dB), so the weight on a power spectral density is 10**(dB/10).
    weight = 10**(dBAw(frequency) / 10)
    level_A = 20.0 * np.log10(np.sqrt(np.sum(weight * power_spectral_density * df)) / pref)
    return frequency, psd_db, level, level_A


def overall_SPL(signal, sampling_rate):
    """
    Computed A-weighted and unweighted sound pressure levels
    Args:
        signal: pressure time history signal, Pa
        sampling_rate: sampling rate of signal, Hz

    Returns:
    tuple (A-weighted Level, Unweighted Level)
    """
    f, spl, levelZ = psd(signal, sampling_rate)
    weight = dBAw(f)
    df = f[1] - f[0]
    levelA = 10.0 * np.log10(df * np.sum(10.0 ** ((spl + weight) / 10.0)))
    return levelA, levelZ


def level_history(signal, sampling_rate, period=1.0):
    """
    Compute time history of SPL
    Args:
        signal: pressure time history signal, Pa
        sampling_rate: sampling rate of signal, Hz
        period: integration time for SPL calculations, sec

    Returns:
        tuple (time, A-weighted level, Unweighted level)
    """
    binwidth = period * sampling_rate
    # One edge per complete period, including the end of the last one.
    edges = np.arange(int(len(signal) // binwidth) + 1) * binwidth

    time = edges[0:-1] / sampling_rate
    level_a = np.zeros_like(time)
    level_z = np.zeros_like(time)
    for i in range(0, len(edges) - 1, 1):
        level_a[i], level_z[i] = overall_SPL(signal[int(edges[i]):int(edges[i + 1])], sampling_rate)
    return time, level_a, level_z


def nextpow2(x):
    """
    Compute the smallest integer N for x <= 2**N
    Args:
        x: number or array-like of values for which to compute N

    Returns:
        number or array like of integral N
    """
    return np.ceil(np.log2(np.abs(x)))

def third_octave_band_levels(signal, sampling_rate, cal=0.0, fmin=20.0, fmax=20000.0):
    """
    Compute third-octave band levels of a signal
    Args:
        signal: Array-like acoustic signal
        sampling_rate: Sampling rate of signal, Hz
        cal: Optional calibration factor to apply to signal (dB)
        fmin: minimum frequency for third-octave bands, Hz
        fmax: maximum frequency for third-octave bands, Hz

    Returns: tuple (band_centers, band_levels)
           WHERE
           band_centers is an array of third-octave band center frequencies
           band_levels is an array of third-octave band levels in dB
    """
    # Define third-octave band center frequencies.  fmin/fmax are compared
    # within a quarter band, so nominal limits keep their own bands: the exact
    # centres of the 20 Hz and 20 kHz bands are 19.69 and 20159 Hz.  Bands
    # reaching past Nyquist would be only partly filled, so they are dropped.
    k = np.arange(-50, 50)
    band_centers = 1000.0 * (2.0 ** (k / 3.0))
    tolerance = 2.0 ** (1.0 / 12.0)
    band_centers = band_centers[np.logical_and.reduce((
        band_centers >= fmin / tolerance,
        band_centers <= fmax * tolerance,
        band_centers * 2.0 ** (1.0 / 6.0) <= 0.5 * sampling_rate))]
    # Compute PSD
    frequency, psd_db, _ = psd(signal, sampling_rate, cal)
    pref = 2.0e-5
    psd_linear = (pref ** 2) * 10.0 ** (psd_db / 10.0)
    df = frequency[1] - frequency[0]
    band_levels = np.zeros_like(band_centers)
    for i, fc in enumerate(band_centers):
        f_lower = fc / (2.0 ** (1.0 / 6.0))
        f_upper = fc * (2.0 ** (1.0 / 6.0))
        band_indices = np.where(np.logical_and(frequency >= f_lower, frequency < f_upper))
        band_power = np.sum(psd_linear[band_indices] * df)
        band_levels[i] = 10.0 * np.log10(band_power / (pref ** 2))
    return band_centers, band_levels


def third_octave_band_edges(band_centers_hz):
    """Lower and upper edges of one-third octave bands that tile without gaps.

    Nominal centres (12.5, 1250, 1600 Hz, ...) are rounded, so edges taken as
    ``fc * 2**(+-1/6)`` straight from them overlap their neighbours or leave
    gaps -- up to 8 % of a band -- and a brick-wall band sum then drops or
    double-counts whatever lies there (a 1410 Hz tone falls in no band, an
    895 Hz one in two).  IEC 61260-1 defines each nominal band by its exact
    base-10 midband ``1000 * 10**(k/10)``, with edges a twentieth of a decade
    either side, so a nominal centre (within a quarter band of one) is given
    those edges.  When every centre is one of this module's own exact base-2
    centres ``1000 * 2**(k/3)``, they already tile and keep their
    ``fc * 2**(+-1/6)`` edges.  In a set that mixes the two, the base-2
    centres take their band's base-10 edges as well, because edges from the
    two grids do not meet: keeping ``fc * 2**(+-1/6)`` for 1259.9 Hz next to a
    nominal 1600 Hz band would overlap it by 1.7 Hz.  A centre near neither
    grid keeps ``fc * 2**(+-1/6)``.

    Args:
        band_centers_hz: band centre frequencies, Hz (exact or nominal)
    Returns: tuple (f_lower, f_upper) of arrays, Hz
    """
    fc = np.asarray(band_centers_hz, dtype=float)
    base2 = 1000.0 * 2.0 ** (np.round(3.0 * np.log2(fc / 1000.0)) / 3.0)
    if np.allclose(fc, base2, rtol=1e-9, atol=0.0):
        return fc / 2.0 ** (1.0 / 6.0), fc * 2.0 ** (1.0 / 6.0)
    base10 = 1000.0 * 10.0 ** (np.round(10.0 * np.log10(fc / 1000.0)) / 10.0)
    nominal = np.abs(np.log2(fc / base10)) < 1.0 / 12.0
    fm = np.where(nominal, base10, fc)
    half_band = np.where(nominal, 10.0 ** (1.0 / 20.0), 2.0 ** (1.0 / 6.0))
    return fm / half_band, fm * half_band


# --------------------------------------------------------------------------
# Perceived noise level metrics (PNL, PNLT, EPNL)
#
# Implements 14 CFR Part 36 Appendix A, section A36.4 (identical to ICAO
# Annex 16 Vol. I Appendix 2): noy conversion via Table A36-3, PNL, the
# ten-step spectral-irregularity (tone) correction with Table A36-2, the
# A36.4.4.2 band-sharing adjustment to PNLTM, and the duration correction.
# Cross-checked against the CFR text and the MATLAB reference implementation
# (tools.git/metrics/EPNLcalc.m); deviations from that MATLAB code are
# deliberate and follow the regulation:
#   * 500 Hz belongs to the 500<=f<=5000 tone-correction range (Table A36-2),
#   * the duration integral runs over the CONTIGUOUS t(1)..t(2) interval,
#     including any dips below PNLTM-10 within it (A36.4.5.2/A36.4.5.5),
#   * the A36.4.4.2 five-interval tone-suppression check is applied.
#
# Band convention: 24 one-third octave bands, 50 Hz to 10 kHz, ascending.

PNL_BAND_FREQUENCIES = np.array([
    50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0,
    500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0,
    4000.0, 5000.0, 6300.0, 8000.0, 10000.0])

_INF = np.inf
# Table A36-3: constants for mathematically formulated noy values.
_NOY_SPL_A = np.array([91.0, 85.9, 87.3, 79.9, 79.8, 76.0, 74.0, 74.9, 94.6,
                       _INF, _INF, _INF, _INF, _INF, _INF, _INF, _INF, _INF,
                       _INF, _INF, _INF, _INF, 44.3, 50.7])
_NOY_SPL_B = np.array([64.0, 60, 56, 53, 51, 48, 46, 44, 42, 40, 40, 40, 40,
                       40, 38, 34, 32, 30, 29, 29, 30, 31, 37, 41])
_NOY_SPL_C = np.array([52.0, 51, 49, 47, 46, 45, 43, 42, 41, 40, 40, 40, 40,
                       40, 38, 34, 32, 30, 29, 29, 30, 31, 34, 37])
_NOY_SPL_D = np.array([49.0, 44, 39, 34, 30, 27, 24, 21, 18, 16, 16, 16, 16,
                       16, 15, 12, 9, 5, 4, 5, 6, 10, 17, 21])
_NOY_SPL_E = np.array([55.0, 51, 46, 42, 39, 36, 33, 30, 27, 25, 25, 25, 25,
                       25, 23, 21, 18, 15, 14, 14, 15, 17, 23, 29])
_NOY_M_B = np.array([0.043478, 0.040570, 0.036831, 0.036831, 0.035336,
                     0.033333, 0.033333, 0.032051, 0.030675, 0.030103,
                     0.030103, 0.030103, 0.030103, 0.030103, 0.030103,
                     0.029960, 0.029960, 0.029960, 0.029960, 0.029960,
                     0.029960, 0.029960, 0.042285, 0.042285])
_NOY_M_C = np.array([0.030103, 0.030103, 0.030103, 0.030103, 0.030103,
                     0.030103, 0.030103, 0.030103, 0.030103, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.029960, 0.029960])           # 0.0 = "not applicable"
_NOY_M_D = np.array([0.079520, 0.068160, 0.068160, 0.059640, 0.053013,
                     0.053013, 0.053013, 0.053013, 0.053013, 0.053013,
                     0.053013, 0.053013, 0.053013, 0.053013, 0.059640,
                     0.053013, 0.053013, 0.047712, 0.047712, 0.053013,
                     0.053013, 0.068160, 0.079520, 0.059640])
_NOY_M_E = np.array([0.058098, 0.058098, 0.052288, 0.047534, 0.043573,
                     0.043573, 0.040221, 0.037349, 0.034859, 0.034859,
                     0.034859, 0.034859, 0.034859, 0.034859, 0.034859,
                     0.040221, 0.037349, 0.034859, 0.034859, 0.034859,
                     0.034859, 0.037349, 0.037349, 0.043573])


def noys(band_levels):
    """
    Convert one-third octave band SPLs to perceived noisiness in noys.

    Implements 14 CFR 36 Appendix A, A36.4.7.3 with Table A36-3.
    Args:
        band_levels: (..., 24) SPL in dB for the bands 50 Hz..10 kHz
    Returns: array of noys, same shape (0 below SPL(d))
    """
    spl = np.asarray(band_levels, dtype=float)
    if spl.shape[-1] != 24:
        raise ValueError("expected 24 one-third octave bands (50 Hz..10 kHz)")
    n = np.zeros_like(spl)
    region_a = spl >= _NOY_SPL_A
    region_b = (spl >= _NOY_SPL_B) & ~region_a
    region_e = (spl >= _NOY_SPL_E) & (spl < _NOY_SPL_B)
    region_d = (spl >= _NOY_SPL_D) & (spl < _NOY_SPL_E)
    n = np.where(region_a, 10.0 ** (_NOY_M_C * (spl - _NOY_SPL_C)), n)
    n = np.where(region_b, 10.0 ** (_NOY_M_B * (spl - _NOY_SPL_B)), n)
    n = np.where(region_e, 0.3 * 10.0 ** (_NOY_M_E * (spl - _NOY_SPL_E)), n)
    n = np.where(region_d, 0.1 * 10.0 ** (_NOY_M_D * (spl - _NOY_SPL_D)), n)
    return n


def perceived_noise_level(band_levels):
    """
    Perceived noise level PNL(k), 14 CFR 36 Appendix A, A36.4.2.

    Args:
        band_levels: (..., 24) SPL in dB for the bands 50 Hz..10 kHz
    Returns: PNL in PNdB (shape band_levels.shape[:-1]); -inf where the
        spectrum produces zero total noisiness
    """
    n = noys(band_levels)
    total = 0.85 * n.max(axis=-1) + 0.15 * n.sum(axis=-1)
    with np.errstate(divide="ignore"):
        return 40.0 + (10.0 / np.log10(2.0)) * np.log10(total)


def tone_correction(band_levels):
    """
    Tone correction factor C(k), 14 CFR 36 Appendix A, A36.4.3.1 steps 1-10.

    Args:
        band_levels: (n_times, 24) or (24,) SPL in dB, bands 50 Hz..10 kHz
    Returns: tuple (c_max, tone_band_index)
        c_max: (n_times,) largest tone correction factor, dB
        tone_band_index: (n_times,) band index (0-23) it occurred in
    """
    spl = np.atleast_2d(np.asarray(band_levels, dtype=float))
    nt = spl.shape[0]

    # Step 1: slopes; band 3 (index 2) has no value -- comparisons that need
    # it (step 2 at band 4) are therefore not evaluated, per the regulation.
    s = np.full((nt, 24), np.nan)
    s[:, 3:] = spl[:, 3:] - spl[:, 2:-1]

    # Step 2: |delta s| > 5, evaluable from band 5 (index 4) on
    encircle_slope = np.zeros((nt, 24), dtype=bool)
    encircle_slope[:, 4:] = np.abs(s[:, 4:] - s[:, 3:-1]) > 5.0

    # Step 3: encircle band SPLs
    enc = np.zeros((nt, 24), dtype=bool)
    for i in range(4, 24):
        e = encircle_slope[:, i]
        pos = e & (s[:, i] > 0) & (s[:, i] > s[:, i - 1])
        neg = e & (s[:, i] <= 0) & (s[:, i - 1] > 0)
        enc[:, i] |= pos
        enc[:, i - 1] |= neg

    # Step 4: adjusted levels SPL'
    spl1 = spl.copy()
    for i in range(3, 23):
        spl1[:, i] = np.where(enc[:, i],
                              0.5 * (spl[:, i - 1] + spl[:, i + 1]),
                              spl[:, i])
    spl1[:, 23] = np.where(enc[:, 23], spl[:, 22] + s[:, 22], spl[:, 23])

    # Step 5: new slopes s', with s'(3)=s'(4) and an imaginary 25th band
    s1 = np.empty((nt, 25))
    s1[:, 3:24] = spl1[:, 3:] - spl1[:, 2:-1]
    s1[:, 2] = s1[:, 3]
    s1[:, 24] = s1[:, 23]

    # Step 6: three-slope average, bands 3..23 (indices 2..22)
    sbar = np.empty((nt, 23))
    sbar[:, 2:] = (s1[:, 2:23] + s1[:, 3:24] + s1[:, 4:25]) / 3.0

    # Step 7: final background levels SPL''
    spl2 = np.empty_like(spl)
    spl2[:, 2] = spl[:, 2]
    for i in range(3, 24):
        spl2[:, i] = spl2[:, i - 1] + sbar[:, i - 1]

    # Step 8: differences F, bands 3..24; only F >= 1.5 matters
    f_diff = np.zeros_like(spl)
    f_diff[:, 2:] = spl[:, 2:] - spl2[:, 2:]

    # Step 9: Table A36-2. Note 500 Hz belongs to the middle range.
    freq = PNL_BAND_FREQUENCIES
    low = (freq < 500.0) | (freq > 5000.0)      # F/3-1/2, F/6, 10/3
    mid = ~low                                   # 2F/3-1, F/3, 20/3
    c = np.zeros_like(spl)
    F = f_diff
    c_low = np.where(F >= 20.0, 10.0 / 3.0,
                     np.where(F >= 3.0, F / 6.0,
                              np.where(F >= 1.5, F / 3.0 - 0.5, 0.0)))
    c_mid = np.where(F >= 20.0, 20.0 / 3.0,
                     np.where(F >= 3.0, F / 3.0,
                              np.where(F >= 1.5, 2.0 * F / 3.0 - 1.0, 0.0)))
    c[:, :] = np.where(low, c_low, c_mid)
    c[:, :2] = 0.0                               # bands 1-2 not eligible

    # Step 10
    c_max = c.max(axis=1)
    tone_band = np.argmax(c, axis=1)
    if np.ndim(band_levels) == 1:
        return float(c_max[0]), int(tone_band[0])
    return c_max, tone_band


def tone_corrected_perceived_noise_level(band_levels):
    """
    PNLT(k) = PNL(k) + C(k), 14 CFR 36 Appendix A, A36.4.3.

    Args:
        band_levels: (n_times, 24) or (24,) SPL in dB, bands 50 Hz..10 kHz
    Returns: tuple (pnlt, pnl, c_max, tone_band_index)
    """
    pnl = perceived_noise_level(band_levels)
    c_max, tone_band = tone_correction(band_levels)
    return pnl + c_max, pnl, c_max, tone_band


def effective_perceived_noise_level(band_level_history, dt=0.5,
                                    bandshare_adjustment=True):
    """
    EPNL of a noise event, 14 CFR 36 Appendix A, A36.4.

    Args:
        band_level_history: (n_times, 24) third-octave SPL history in dB
            (bands 50 Hz..10 kHz) at equal `dt` increments
        dt: time increment, s (the regulation prescribes 0.5 s)
        bandshare_adjustment: apply the A36.4.4.2 five-interval check for
            tone suppression by band sharing at PNLTM

    Returns: dict with
        epnl: EPNL in EPNdB
        pnltm: maximum tone-corrected perceived noise level, TPNdB
        duration_correction_db: D
        k1, k2: duration-interval sample limits (contiguous, PNLTM-10 down)
        pnlt, pnl, c_max, tone_band: per-sample histories
        clipped: True when the 10 dB-down interval hits the record edge

    The duration correction uses the exact 10*log10(dt/T) normalisation with
    T = 10 s; the regulation's specialised "-13" constant for dt = 0.5 s is a
    rounding of this (difference 0.01 dB).
    """
    spl = np.asarray(band_level_history, dtype=float)
    pnlt, pnl, c_max, tone_band = tone_corrected_perceived_noise_level(spl)
    if not np.isfinite(pnlt).any():
        raise ValueError("no finite PNLT values in the history")

    k_m = int(np.nanargmax(pnlt))
    pnltm = float(pnlt[k_m])
    if bandshare_adjustment:
        # A36.4.4.2: if C at PNLTM is below the average of the five
        # consecutive intervals centred there, tone suppression by band
        # sharing is suspected; recompute PNLTM with the average C.
        lo, hi = max(0, k_m - 2), min(len(pnlt), k_m + 3)
        c_avg = float(np.mean(c_max[lo:hi]))
        if c_max[k_m] < c_avg:
            pnltm = float(pnl[k_m] + c_avg)

    # Duration interval: contiguous from the first to the last sample at or
    # above PNLTM-10 (dips within the interval are included), A36.4.5.
    above = np.where(pnlt >= pnltm - 10.0)[0]
    k1, k2 = int(above[0]), int(above[-1])
    duration = 10.0 * np.log10(
        np.sum(10.0 ** (pnlt[k1:k2 + 1] / 10.0)) * dt / 10.0) - pnltm
    return {
        "epnl": pnltm + duration,
        "pnltm": pnltm,
        "duration_correction_db": duration,
        "k1": k1, "k2": k2,
        "pnlt": pnlt, "pnl": pnl, "c_max": c_max, "tone_band": tone_band,
        "clipped": bool(k1 == 0 or k2 == len(pnlt) - 1),
    }


def ten_db_down_interval(levels, down=10.0):
    """Sample limits of the 10 dB-down duration of a level history.

    ``(k1, k2)``: the first and last samples at or above ``max - down``.  Dips
    below the threshold between them are inside the interval, as in the EPNL
    duration of 14 CFR 36 A36.4.5 and the SEL convention that follows it --
    a second rise (e.g. a hover at the end of an approach heard from upstream)
    is part of the event, not cut off at the first dip.
    """
    levels = np.asarray(levels, dtype=float)
    finite = np.isfinite(levels)
    if not finite.any():
        raise ValueError('no finite levels in the history')
    peak = np.nanmax(np.where(finite, levels, -np.inf))
    above = np.where(finite & (levels >= peak - float(down)))[0]
    return int(above[0]), int(above[-1])


def sound_exposure_level(levels, dt, down=10.0, weighted_levels=None):
    """Sound exposure level of a noise event over its 10 dB-down duration.

    SEL = 10 log10( sum 10^(L/10) dt / T0 ), T0 = 1 s, summed over the interval
    from :func:`ten_db_down_interval` -- the standard duration for aircraft noise
    metrics.  Pass ``down=np.inf`` to integrate the whole record instead.

    Args:
        levels: level history (dB; normally A-weighted, i.e. LA) at equal ``dt``.
        dt: sample interval, s.
        down: how far below the maximum the duration extends, dB (default 10).
        weighted_levels: optional history to integrate over the interval chosen
            from ``levels`` (e.g. choose the interval on LA, integrate LC).

    Returns: dict with
        sel: sound exposure level, dB
        lmax: maximum level, dB
        k1, k2: interval sample limits
        duration_s: (k2 - k1 + 1) * dt
        clipped: True when the interval touches the record edge -- the event
            may extend beyond the data and the SEL is then a lower bound.
    """
    levels = np.asarray(levels, dtype=float)
    if np.isfinite(down):
        k1, k2 = ten_db_down_interval(levels, down)
    else:
        k1, k2 = 0, levels.size - 1
    integrand = levels if weighted_levels is None else np.asarray(weighted_levels, dtype=float)
    segment = integrand[k1:k2 + 1]
    energy = np.sum(10.0 ** (segment[np.isfinite(segment)] / 10.0)) * float(dt)
    return {
        'sel': 10.0 * np.log10(energy) if energy > 0 else -np.inf,
        'lmax': float(np.nanmax(levels)),
        'k1': k1, 'k2': k2,
        'duration_s': (k2 - k1 + 1) * float(dt),
        'clipped': bool(k1 == 0 or k2 == levels.size - 1),
    }


def load_mil_std_1474e_table_c1(filename):
    """
    Load MIL-STD-1474E Table C-1 data from CSV.

    Args:
        filename: CSV file path containing the converted Table C-1 data.

    Returns:
        dict with keys:
            band_freq_hz: third-octave center frequencies in the table
            distance_columns_m: nondetectability distance columns (m)
            measurement_distance_m: measurement distance used for each column (m)
            limits_db: table limits (rows=freq, cols=distance column), NaN for missing values
    """
    rows = np.genfromtxt(filename, delimiter=',', dtype=str)
    if rows.ndim != 2 or rows.shape[0] < 3:
        raise ValueError('MIL-STD table CSV appears malformed')

    distance_columns_m = np.array([float(v) for v in rows[1, 1:] if v != ''], dtype=float)

    measurement_row_idx = None
    for i in range(rows.shape[0]):
        if rows[i, 0].strip().lower() == 'measurement distance (meters)':
            measurement_row_idx = i
            break
    if measurement_row_idx is None:
        raise ValueError('Could not find "Measurement Distance (meters)" row in MIL-STD Table C-1 CSV')

    measurement_distance_m = np.array([float(v) for v in rows[measurement_row_idx, 1:] if v != ''], dtype=float)

    band_rows = rows[2:measurement_row_idx, :]
    band_freq_hz = np.array([float(r[0]) for r in band_rows], dtype=float)

    limits_db = np.zeros((band_rows.shape[0], distance_columns_m.size), dtype=float)
    limits_db[:] = np.nan
    for i, row in enumerate(band_rows):
        for j in range(distance_columns_m.size):
            value = row[j + 1].strip()
            if value == '' or value.upper() == 'NA':
                continue
            limits_db[i, j] = float(value)

    return {
        'band_freq_hz': band_freq_hz,
        'distance_columns_m': distance_columns_m,
        'measurement_distance_m': measurement_distance_m,
        'limits_db': limits_db,
    }


def _mil_std_interp_log_frequency(x_freq_hz, y_values, target_freq_hz):
    return np.interp(np.log10(target_freq_hz), np.log10(x_freq_hz), y_values)


def _mil_std_distance_within_group(measured_level_db, thresholds_db, distances_m):
    valid = np.isfinite(thresholds_db)
    if np.count_nonzero(valid) == 0:
        return np.nan, 'warning_insufficient_group_data', False

    thresholds = thresholds_db[valid]
    distances = distances_m[valid]
    order = np.argsort(distances)
    thresholds = thresholds[order]
    distances = distances[order]

    if measured_level_db <= thresholds[0]:
        return distances[0], 'warning_group_below_first', False

    for i in range(distances.size - 1):
        level_1 = thresholds[i]
        level_2 = thresholds[i + 1]
        distance_1 = distances[i]
        distance_2 = distances[i + 1]

        if (measured_level_db - level_1) * (measured_level_db - level_2) > 0.0:
            continue

        if level_2 == level_1:
            return distance_1, 'warning_group_flat_segment', False

        log_distance = np.interp(
            measured_level_db,
            [level_1, level_2],
            [np.log10(distance_1), np.log10(distance_2)]
        )
        return 10.0 ** log_distance, 'group_interpolated', True

    return np.nan, 'warning_group_above_last', False


def _mil_std_distance_from_table(measured_level_db_10m, threshold_levels_db, distances_m, measurement_distance_m):
    group_measurement_distances = np.unique(measurement_distance_m)
    candidates = []

    for group_distance_m in group_measurement_distances:
        group_mask = measurement_distance_m == group_distance_m
        if not np.any(group_mask):
            continue

        measured_level_group_db = measured_level_db_10m + 20.0 * np.log10(10.0 / group_distance_m)
        distance_m, status, is_valid = _mil_std_distance_within_group(
            measured_level_group_db,
            threshold_levels_db[group_mask],
            distances_m[group_mask],
        )

        if np.isfinite(distance_m):
            candidates.append((distance_m, f'{status}@{group_distance_m:.0f}m', is_valid))

    if len(candidates) == 0:
        return np.nan, 'warning_above_all_groups', False

    min_idx = int(np.argmin([c[0] for c in candidates]))
    return candidates[min_idx]


def mil_std_1474e_nondetectability_distance(
        band_centers_hz,
        band_levels_db,
        mil_std_table,
        spectrum_distance_m=10.0):
    """
    Compute MIL-STD-1474E Table C-1 nondetectability distance from a third-octave spectrum.

    Args:
        band_centers_hz: third-octave center frequencies for the spectrum.
        band_levels_db: third-octave levels (dB) at spectrum_distance_m.
        mil_std_table: dict from load_mil_std_1474e_table_c1(...) or CSV filename.
        spectrum_distance_m: distance corresponding to band_levels_db (m).

    Returns:
        dict with:
            overall_nondetectability_distance_m: overall distance using valid bands only
            trigger_frequency_hz: band frequency that sets the overall distance
            band_frequency_hz: MIL-STD band frequencies used in evaluation
            band_nondetectability_distance_m: per-band nondetectability distance
            band_status: per-band status strings
            band_valid: per-band boolean validity (True only for interpolated values)
            band_level_db_10m: input spectrum mapped to MIL-STD bands and normalized to 10 m
    """
    if isinstance(mil_std_table, (str, os.PathLike)):
        table = load_mil_std_1474e_table_c1(mil_std_table)
    else:
        table = mil_std_table

    band_centers_hz = np.asarray(band_centers_hz, dtype=float)
    band_levels_db = np.asarray(band_levels_db, dtype=float)

    table_band_freq_hz = np.asarray(table['band_freq_hz'], dtype=float)
    distance_columns_m = np.asarray(table['distance_columns_m'], dtype=float)
    measurement_distance_m = np.asarray(table['measurement_distance_m'], dtype=float)
    limits_db = np.asarray(table['limits_db'], dtype=float)

    if band_centers_hz.size == 0 or band_levels_db.size == 0:
        raise ValueError('Input spectrum is empty')

    band_levels_table = _mil_std_interp_log_frequency(
        band_centers_hz,
        band_levels_db,
        table_band_freq_hz,
    )
    band_levels_10m = band_levels_table + 20.0 * np.log10(spectrum_distance_m / 10.0)

    band_distances = np.zeros_like(table_band_freq_hz)
    band_status = np.empty(table_band_freq_hz.size, dtype=object)
    band_valid = np.zeros(table_band_freq_hz.size, dtype=bool)

    for i in range(table_band_freq_hz.size):
        distance_m, status, is_valid = _mil_std_distance_from_table(
            band_levels_10m[i],
            limits_db[i, :],
            distance_columns_m,
            measurement_distance_m,
        )
        band_distances[i] = distance_m
        band_status[i] = status
        band_valid[i] = is_valid

    valid_for_overall = np.logical_and(np.isfinite(band_distances), band_valid)
    if np.any(valid_for_overall):
        valid_indices = np.where(valid_for_overall)[0]
        best_local_idx = int(np.argmax(band_distances[valid_for_overall]))
        trigger_idx = valid_indices[best_local_idx]
        overall_distance = float(band_distances[trigger_idx])
        trigger_frequency = float(table_band_freq_hz[trigger_idx])
    else:
        overall_distance = np.nan
        trigger_frequency = np.nan

    return {
        'overall_nondetectability_distance_m': overall_distance,
        'trigger_frequency_hz': trigger_frequency,
        'band_frequency_hz': table_band_freq_hz,
        'band_nondetectability_distance_m': band_distances,
        'band_status': band_status,
        'band_valid': band_valid,
        'band_level_db_10m': band_levels_10m,
    }


def spectrogram(signal, sampling_rate, window_time=0.5, window_type="hann", window_overlap=7.0 / 8.0,
                detrend='constant', dbref=20e-6):
    """
    Computes the spectrogram of a signal
    Args:
        signal: array-like containing acoustic signal
        sampling_rate: sampling rate of signal, Hz
        window_time: duration of windows, s
        window_type: optional type of window to us, see scipy.signal.window, default="hann"
        window_overlap: optional proportion of overlap for windows, default=7.0/8.0
        detrend: optional detrending mode for signal, see scipy.signal.detrend
        dbref: optional reference value for calculating decibels, default 20e-6

    Returns: tuple (f, t, SPL)
    WHERE
    f is an array of frequencies
    t is an array of times
    SPL is a frequency x time power spectral density spectrogram, dB/Hz**2
    """
    # Pick next power of two that captures the window time, and generate the window
    binwidth = int(2.0 ** nextpow2(window_time * sampling_rate))
    window = scipy.signal.get_window(window_type, binwidth)
    # Calculate the spectrogram as a PSD
    f, t, Sxx = scipy.signal.spectrogram(signal, sampling_rate, window, noverlap=round(window_overlap * binwidth),
                                         detrend=detrend, mode='psd')
    # Convert to SPL (exact zeros in Sxx are legitimate; -inf, not a warning)
    with np.errstate(divide='ignore'):
        SPL = 10.0 * np.log10(Sxx / (dbref ** 2))
    return f, t, SPL


def plot_spectrogram(signal, sampling_rate, window_time=1.0, window_type="hann", window_overlap=7.0 / 8.0,
                     detrend='constant', dbref=20e-6, save_name=None, title=None, time0=0, clim=None, flim=None):
    """
    Plot a spectrogram of signal
    Args:
        signal: array-like containing acoustic signal
        sampling_rate: sampling rate of signal, Hz
        window_time: duration of windows, s
        window_type: optional type of window to us, see scipy.signal.window, default="hann"
        window_overlap: optional proportion of overlap for windows, default=7.0/8.0
        detrend: optional detrending mode for signal, see scipy.signal.detrend
        dbref: optional reference value for calculating decibels, default 20e-6
        save_name: optional path to image file to save figure, default None
        title: optional title for the plot, default None
        time0: optional start time for signal, default 0
        clim: optional level scale color limits, default None
        flim: optional frequency scale limits, default None

    Returns: tuple (fig, ax, cs)
    WHERE
    fig is a matplotlib handle to the figure
    ax is a matplotlib handle to the plot axis
    cs is a matplotlib handle to the contour quadmesh
    """
    f, t, SPL = spectrogram(signal, sampling_rate, window_time, window_type, window_overlap, detrend, dbref)
    if flim is not None:
        fidx = np.logical_and(f >= flim[0], f <= flim[1])
        f = f[fidx]
        SPL = SPL[fidx, :]

    fig, ax = subplots(facecolor='white')
    if clim is None:
        vmin = None
        vmax = None
    else:
        vmin = clim[0]
        vmax = clim[1]
    cs = ax.pcolormesh(t + time0, f, SPL, vmin=vmin, vmax=vmax)
    cb = colorbar(cs, format='%.0f')
    ax.set_ylabel('Frequency, Hz')
    ax.set_xlabel('Time, s')
    cb.set_label('Power Spectral Density, dB')
    if title is not None:
        ax.set_title(title)
    if save_name is not None:
        fig.savefig(os.path.abspath(os.path.expanduser(save_name)))
    return fig, ax, cs


def dedopplerize(time, pressure, speed_of_sound, track_time, position, observers,
                 radius=None, output_sample_rate=None, time_offset=0.0):
    """
    Time domain de-Dopplerization of an acoustic signal
    Args:
        time: observers x time points matrix of observer times
        pressure: observers x time point matrix of acoustic pressures
        speed_of_sound: speed of sound
        track_time: array of source times
        position: track times x 3 matrix of source positions
        observers: observers x 3 matrix of observer positions
        radius: optional virtual observer radius
        output_sample_rate: optional source time sample rate

    Returns: tuple (emission_times, dpres)
    WHERE
    emission_times is an array emission times
    dpres is an observer x emission times matrix of de-Dopplerized acoustic pressures
    """
    # If not set, infer sample rate from measured data
    if output_sample_rate is None:
        output_sample_rate = 1 / (time[0, 1] - time[0, 0])
    # Generate emission time array
    emission_times = np.arange(track_time[0], track_time[-1], 1 / output_sample_rate)
    number_of_microphones = observers.shape[0]
    number_of_times = emission_times.shape[0]
    dpres = np.zeros((number_of_microphones, number_of_times))
    # For each observer, de-Dopplerize
    for index, observer in enumerate(observers):
        rx = observer[0] - position[:, 0]
        ry = observer[1] - position[:, 1]
        rz = observer[2] - position[:, 2]
        r = np.interp(emission_times, track_time, np.sqrt(rx ** 2 + ry ** 2 + rz ** 2))
        dpres[index, :] = np.interp(emission_times + r / speed_of_sound, time[index, :], pressure[index, :])
        # Apply spherical spreading when radius is known
        if radius is not None:
            dpres[index, :] = r / radius * dpres[index, :]
    # Adjust time of emission to virtual observer observation time when radius is known
    if radius is not None:
        emission_times = emission_times + radius / speed_of_sound
    return emission_times, dpres


def linear_array_plan(nmics, altitude, min_elevation=10.0, target_elv=90.0):
    """
    Design a linear microphone array
    Args:
        nmics: number of microphones
        altitude: flight altitude of the vehicle
        min_elevation: desired sideline elevation angle relative to horizon
        target_elv: target elevation for equal angle sideline spacing

    Returns: array of microphone sideline locations
    """
    # Slant distance to target elevation center line position
    r = altitude / np.sin(np.radians(target_elv))
    # Now, determine spacing for equal angular resolution on target elv plane
    angles = np.linspace(-90. + min_elevation, 90.0 - min_elevation, nmics)
    return r * np.tan(np.radians(angles))


def hemigen(time, source, velocity, observers, speed_of_sound):
    """
    Compute the time of emission observer angles for a moving source
    Args:
        time: Array of source times
        source: time x 3 matrix of source positions
        velocity: time x 3 matrix of source velocities
        observers: number of mics x 3 matrix of observer positions
        speed_of_sound: speed of sound

    Returns: tuple (azimuth, elevation, r, t_observer, mach_r)
    WHERE
    azimith are the azimuth angles on the sphere for each emission time
    elevation are the elevation angles on the sphere for each emission time
    r are the propagation distances for each emission time
    t_observer are the observer times associated with each emission time
    mach_r is the Mach number of the source in the propagation direction
    """
    number_of_mics = observers.shape[0]
    number_of_times = time.shape[0]
    # Expand data into number_of_times x number_of_mics arrays
    t = (time * np.ones((number_of_mics, number_of_times))).transpose()
    sx = (source[:, 0] * np.ones((number_of_mics, number_of_times))).transpose()
    sy = (source[:, 1] * np.ones((number_of_mics, number_of_times))).transpose()
    sz = (source[:, 2] * np.ones((number_of_mics, number_of_times))).transpose()
    vx = (velocity[:, 0] * np.ones((number_of_mics, number_of_times))).transpose()
    vy = (velocity[:, 1] * np.ones((number_of_mics, number_of_times))).transpose()
    vz = (velocity[:, 2] * np.ones((number_of_mics, number_of_times))).transpose()
    ox = observers[:, 0] * np.ones((number_of_times, number_of_mics))
    oy = observers[:, 1] * np.ones((number_of_times, number_of_mics))
    oz = observers[:, 2] * np.ones((number_of_times, number_of_mics))
    # Propagation vectors
    rx = ox - sx
    ry = oy - sy
    rz = oz - sz
    r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
    # Observation time
    t_observer = t + r / speed_of_sound
    # Mach number along the radiation direction
    mach_r = (vx * rx / r + vy * ry / r + vz * rz / r) / speed_of_sound
    # Compute elevation relative to horizon
    ground_range = np.sqrt(rx ** 2 + ry ** 2)
    height = -rz
    elevation = np.degrees(np.arctan2(height, ground_range))
    # Compute azimuth as difference between aircraft heading and observer bearing
    bearing = np.degrees(np.arctan2(ry, rx))
    heading = np.degrees(np.arctan2(vy, vx))
    azimuth = np.remainder(bearing - (heading + 180), 360)
    # If the elevation exceeds 90 degrees, it should be flipped over to the other
    # side of the sphere
    flip = elevation > 90
    elevation[flip] = 180 - elevation[flip]
    # Then correct azimuths for these points
    right_side = np.logical_and(flip, azimuth <= 180)
    left_side = np.logical_and(flip, azimuth > 180)
    azimuth[right_side] = azimuth[right_side] + 180
    azimuth[left_side] = azimuth[left_side] - 180
    return azimuth, elevation, r, t_observer, mach_r


def depropagate_hemisphere(
        mic_locations,
        pressure,
        time,
        track_time,
        track_position,
        track_velocity=None,
        *,
        speed_of_sound=1135.0,
        length_units='ft',
        r_ref=100.0,
        freq_range=(0.0, 2000.0),
        window_time=0.5,
        window_overlap=0.5,
        point_stride=1,
        azi_step=10.0,
        elv_step=10.0,
        rmax=25.0,
        apply_absorption_deprop=False,
        atmosphere=Atmosphere(temperature=293.15, pressure=101.325, relative_humidity=20.0),
        flip_y_for_geometry=False,
        return_scattered=False,
        third_octave=False,
        third_octave_fmin=20.0,
        third_octave_band_centers_hz=None,
        third_octave_method='fft',
        narrowband=False,
        narrowband_stride=1,
        min_elevation_deg=0.0,
        max_range=None,
        ambient_time_range=None,
        ambient_pressure=None,
        ambient_time=None,
        ambient_percentile=None,
        band_snr_gate_db=3.0,
        max_absorption_correction_db=None,
        receiver_response_db=None,
        ray_model=None,
        interpolation=None,
        rim_range=None,
):
    """Generate an acoustic hemisphere from microphone time series and vehicle tracking data.

    Emission points whose elevation angle falls below ``min_elevation_deg``
    are excluded before gridding. Low-elevation points correspond to long
    ranges where the microphone signal is usually ambient-dominated;
    depropagating them multiplies the ambient floor by (r/r_ref)^2 and
    contaminates the hemisphere rim, so gate them out (and restrict
    track_time to the high-SNR portion of the pass) unless the data are
    known to be signal-dominated all the way down. Near-grazing incidence is
    also where ground impedance dominates what the microphone hears, which
    the depropagation does not model, so around 10 degrees is a sensible
    floor for a ground-board array rather than 0.

    ``max_range`` drops emission point / microphone pairs whose separation
    exceeds it, in ``length_units``. Depropagation assumes a straight ray
    through a homogeneous atmosphere; over long paths refraction by the wind
    and temperature profile makes that a poor model, quite apart from the
    absorption correction growing beyond what the measurement supports.

    Ambient gating (strongly recommended) removes the dominant failure mode
    of depropagation: a band that is ambient-limited at the microphone
    carries no source information, but spreading and especially absorption
    depropagation multiply it by (r/r_ref)^2 and 10^(alpha(f)(r-r_ref)/10).
    At the top third-octave bands alpha is tens of dB/km, so an ambient-
    limited 10 kHz band at a kilometre of range is amplified into a
    physically impossible source level -- the Be407 spheres reach 216 dB at
    10 kHz on the aft pole this way. A broadband SNR gate cannot catch it,
    because the broadband level is dominated by the low-frequency bands
    where the signal is strong.

    Supply the ambient reference in one of three ways:

    * ``ambient_pressure`` -- a separate signal-free recording per
      microphone (same channel order as ``mic_locations``), e.g. a dedicated
      ambient run from the same array and test day. Optionally pass
      ``ambient_time`` alongside; only the sample rate is used, so a bare
      pressure array is enough. This is the preferred form: the ambient is
      measured rather than carved out of the run.
    * ``ambient_time_range=(t0, t1)`` -- a signal-free window of the run
      recording itself, in the same absolute time base as ``time``.
    * ``ambient_percentile`` -- a low percentile of the run's own spectrogram,
      per microphone and per frequency bin. Use it when no ambient recording
      exists for that array layout. It assumes the quietest few percent of
      frames are signal-free: checked against B407's measured ambient runs,
      the 5th percentile agrees to -0.3 dB in the median, but with roughly
      +-8 dB of scatter per channel, so it is a fallback rather than an
      equivalent. Note that a *higher* percentile is not safer -- 25 already
      overestimates that ambient by 8 dB. The bias runs the other way on a
      record that is mostly quiet, where a low percentile samples the low tail
      of the noise fluctuation and gates too little.

    Either way each microphone's ambient PSD is the median spectrogram over
    the ambient frames, computed with the same window settings as the run so
    the frequency grids match. It is then applied per frequency bin *before*
    depropagation: bins whose measured PSD is within ``band_snr_gate_db`` of
    that mic's ambient are zeroed (they carry no usable signal), and the
    ambient power is subtracted from the bins that pass.

    An SNR gate alone is not sufficient, because it bounds the *relative*
    error of a bin but says nothing about how far the absorption correction
    extrapolates. At 10 kHz on a warm day alpha is around 90 dB/km, so an
    emission point 2.7 km away carries a correction near 250 dB: an ambient
    fluctuation that clears any plausible SNR gate still lands at a source
    level hundreds of dB too high, and because the hemisphere averages in
    linear power, one such point dominates its whole neighbourhood. Set
    ``max_absorption_correction_db`` to discard bins whose absorption
    correction exceeds what the measurement can support -- the band is simply
    not observable at that range, and a gap there is the honest result. It has
    no effect unless ``apply_absorption_deprop=True``.

    ``receiver_response_db``, if given, removes what the microphone's
    installation adds.  It is called as ``receiver_response_db(im, bands,
    source_offset)``, with ``bands`` the one-third octave centres (the output
    bands when ``third_octave``), ``source_offset`` the (Npts, 3) emission
    positions minus microphone ``im``'s position, and returns the band-averaged
    level re free field, dB, shaped (len(bands), Npts) -- for instance a
    ground plate's response (:func:`axisymmetric_bem.board_level`).  Each bin
    is divided by its band's value, after the ambient gate and before
    spreading, so a band's energy is divided by the band-averaged response.
    The pressures should then be supplied unscaled (no 0.5 pressure-doubling
    factor).

    ``ray_model``, if given, replaces the straight line in uniform air with a
    refracted ray (see :mod:`refracted_rays`, which defines the interface and
    wraps NICE-OPS's ray tracer).  For each sampled emission point and
    microphone it supplies where the sample is filed (the ray's launch
    depression in place of the straight line's elevation; the azimuth is
    unchanged), the reception time, the range spherical spreading is undone
    over (the ray tube's), the arc absorption is undone over, and the source
    offset ``receiver_response_db`` sees (the equivalent geometry at the ray's
    arrival angle).  Pairs it marks invalid (no ray arrives) are skipped.
    ``min_elevation_deg`` then applies to the filing angle; ``max_range``
    stays on the straight-line distance.  Without one the result is unchanged.

    ``rim_range``, if given as (elevation_deg, range), lets samples filed less than
    ``elevation_deg`` below the horizon come from as far as ``range`` instead of
    ``max_range``.  The range cap keeps a pass's samples to its steeper directions:
    at 2,000 ft a pass at 480 ft never files below 14 deg, so a sphere's rim came
    from low passes alone, and on the 2017 B407 data high passes of the same
    condition imply a forward rim 1-3.5 dB louder (and an aft rim ~1 dB quieter)
    than low passes do at the same angle.  With a ``ray_model`` the filing angle is
    the ray's launch angle and the far samples are depropagated along it; the
    absorption cap (``max_absorption_correction_db``) still drops the bands the
    range makes unobservable.

    ``interpolation``, if given (a dict; ``{}`` takes the defaults in
    :data:`ADAPTIVE_INTERPOLATION`), grids the samples with a radius chosen per
    node instead of the fixed ``rmax``: wide enough for ``k`` samples from
    ``min_mics`` microphones and for the samples' own angular resolution, capped
    at ``max_radius_deg`` (see :func:`adaptive_idw_weights`).  A sample's
    resolution is the arc its analysis window smears over, the source's motion
    across the line of sight in ``window_time`` plus ``source_extent`` (a length,
    e.g. the rotor diameter), over the range.  Nodes beyond the cap are left NaN
    and counted in ``out['interpolation']['gaps']``.  Without it the fixed
    ``rmax`` gridding is unchanged.

    This is the "normal" processing flow used by the demo scripts: use the vehicle kinematics
    to compute emission-time geometry (azimuth/elevation/range) via :func:`hemigen`, sample
    the measured PSD at the corresponding observer times, depropagate to a reference radius,
    then interpolate onto a regular azimuth/elevation grid using :func:`shepIDW`.

    Notes on conventions
    --------------------
    - Inputs must be in a *consistent* local Cartesian frame.
    - If the measured field is mirrored left/right against a reference hemisphere, the
      lateral axis of the supplied geometry runs opposite to the convention here
      (azimuth 180 ahead, 90 starboard). Set ``flip_y_for_geometry=True`` to apply
      Y -> -Y to geometry inputs *before* computing azimuth/elevation.
    - DECIDE THIS IN ANGLE SPACE, NOT BY EYE. Interpolate the reference onto the same
      azimuth/elevation grid and compare residuals against the reference and against
      the reference read backwards in azimuth; take whichever fits. Judging it from the
      rendered plate cannot separate a geometry sign error from a plotting one -- that
      is how the pre-2026-08-19 label/projection mismatch (see :func:`lambert_lon`)
      stayed hidden. ``as350_flip_check.py`` is a worked example. It once concluded the
      AS350 demo data needed the flip, but that was compensating for
      :func:`art2umapr` putting ART phi = +90 to port; with the AAM manual's sign
      (phi > 0 starboard) the same residuals favour no flip. A right-handed, z-up
      frame needs no flip.

    Args:
        mic_locations: (Nmics, 3) microphone locations in local Cartesian coordinates.
        pressure: microphone pressures, shape (Nmics, Nsamples) or list of 1D arrays.
        time: microphone sample times. Either:
            - 1D array (Nsamples,) applied to all microphones, or
            - 2D array (Nmics, Nsamples), or
            - list of 1D arrays.
        track_time: (Nt,) vehicle time array (seconds).
        track_position: (Nt, 3) vehicle position array in same frame as microphones.
        track_velocity: optional (Nt, 3) vehicle velocity array in same frame as microphones.
            If None, computed by finite differences.
        speed_of_sound: speed of sound in the length units per second (e.g., ft/s if length_units='ft').
        length_units: length units for mic/track geometry ('ft' or 'm' typically).
        r_ref: reference radius for depropagation in ``length_units``.
        freq_range: (fmin, fmax) in Hz used for OASPL integration.
        window_time: spectrogram window time (s).
        window_overlap: spectrogram window overlap fraction.
        point_stride: decimation factor applied along emission time samples for speed.
        azi_step: regular grid spacing in azimuth (deg). 360 deg is included to close the seam.
        elv_step: regular grid spacing in elevation (deg).
        rmax: Shepard/IDW neighborhood radius (deg).
        apply_absorption_deprop: if True, undo atmospheric absorption between range r and r_ref.
        atmosphere: Atmosphere instance used when apply_absorption_deprop is True.
        flip_y_for_geometry: if True, apply Y -> -Y to track_position/track_velocity/mic_locations.
        third_octave: if True, also compute third-octave band level hemispheres.
        third_octave_fmin: minimum band center (Hz) when third_octave=True.
        third_octave_method: how band levels are formed when third_octave=True.
            'fft' (default) sums the PSD bins between each band's edges, a
            brick-wall band.  'filter_bank' uses a true one-third octave
            filter bank (:func:`panam_acoustics.filters.third_octave_filter_bank`,
            order-3 Butterworth, as an analyser implements), averaged over the
            same frames.  The two agree within about 0.5 dB above 100 Hz; below
            it the filter skirts carry a strong rotor tone into the
            neighbouring bands, several dB for the bands between main-rotor
            harmonics, which is what analyser-based data such as NORAH2
            contain.  In 'filter_bank' mode the ambient gate, subtraction and
            absorption cap act per band rather than per bin, and absorption is
            taken at the band centre.  OASPL, A-weighted and narrowband output
            are FFT-based either way.

    Returns:
        dict with keys:
            'azi_grid_deg', 'elv_grid_deg',
            'oaspl_db' (2D grid, unweighted),
            'spl_a_db' (2D grid, A-weighted overall level in dBA),
            and optionally:
                        - when return_scattered=True: 'scattered' (dict of scattered az/el/power samples)
                            If third_octave/narrowband are enabled, the corresponding scattered data are
                            also included under 'scattered'.
            - when third_octave=True: 'third_octave' (band centers + band grids)
            - when narrowband=True: 'narrowband' (frequency + PSD grids)

        Grid cells with no sample within ``rmax`` are NaN: nothing was
        measured there.  Cells (and scattered samples) whose power is zero,
        e.g. gated out as ambient, are -inf: measured, with no energy left.
        Summing energy, a -inf contributes nothing and a NaN makes the sum
        unknown.
    """

    mic_locations = np.asarray(mic_locations, dtype=float)
    if mic_locations.ndim != 2 or mic_locations.shape[1] != 3:
        raise ValueError('mic_locations must be shape (Nmics, 3)')
    nmics = mic_locations.shape[0]

    track_time = np.asarray(track_time, dtype=float)
    track_position = np.asarray(track_position, dtype=float)
    if track_position.shape != (track_time.size, 3):
        raise ValueError('track_position must be shape (Nt, 3) matching track_time')

    if track_velocity is None:
        dt = np.gradient(track_time)
        track_velocity = np.gradient(track_position, axis=0) / dt[:, None]
    track_velocity = np.asarray(track_velocity, dtype=float)
    if track_velocity.shape != (track_time.size, 3):
        raise ValueError('track_velocity must be shape (Nt, 3) matching track_time')

    def _as_mic_list_1d(x, *, name):
        if isinstance(x, list):
            if len(x) != nmics:
                raise ValueError(f'{name} list must have length Nmics={nmics}')
            return [np.asarray(xi, dtype=float).ravel() for xi in x]
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            return [arr.ravel() for _ in range(nmics)]
        if arr.ndim == 2 and arr.shape[0] == nmics:
            return [arr[i, :].ravel() for i in range(nmics)]
        raise ValueError(f'{name} must be a list of 1D arrays, a 1D array, or a 2D array shaped (Nmics, Nsamples)')

    pressure_list = _as_mic_list_1d(pressure, name='pressure')
    time_list = _as_mic_list_1d(time, name='time')

    ambient_sources = [name for name, value in
                       (('ambient_pressure', ambient_pressure),
                        ('ambient_time_range', ambient_time_range),
                        ('ambient_percentile', ambient_percentile))
                       if value is not None]
    if len(ambient_sources) > 1:
        raise ValueError('Give only one ambient source, not both of: ' + ', '.join(ambient_sources))
    if ambient_percentile is not None and not 0.0 < float(ambient_percentile) < 100.0:
        raise ValueError('ambient_percentile must lie strictly between 0 and 100')
    if third_octave_method not in ('fft', 'filter_bank'):
        raise ValueError("third_octave_method must be 'fft' or 'filter_bank'")
    use_filter_bank = bool(third_octave) and third_octave_method == 'filter_bank'
    ambient_pressure_list = None
    ambient_time_list = None
    if ambient_pressure is not None:
        ambient_pressure_list = _as_mic_list_1d(ambient_pressure, name='ambient_pressure')
        if ambient_time is not None:
            ambient_time_list = _as_mic_list_1d(ambient_time, name='ambient_time')

    def _time_vec_for_mic(im):
        p = pressure_list[im]
        t = time_list[im]
        if p.size < 2:
            raise ValueError('pressure must have at least 2 samples per microphone')
        if t.size < 2:
            raise ValueError('time must have at least 2 samples per microphone (or provide a common 1D time vector)')
        if t.size == p.size:
            return t
        # Some files store a time vector that doesn't exactly match the signal length.
        # If dt is well-defined, reconstruct an evenly-sampled time vector aligned at t[0].
        dt = float(t[1] - t[0])
        if dt <= 0.0:
            raise ValueError('time must be strictly increasing')
        return float(t[0]) + dt * np.arange(p.size, dtype=float)

    # Optional geometry convention fix
    if flip_y_for_geometry:
        mic_geom = mic_locations.copy()
        mic_geom[:, 1] *= -1.0
        pos_geom = track_position.copy()
        pos_geom[:, 1] *= -1.0
        vel_geom = track_velocity.copy()
        vel_geom[:, 1] *= -1.0
    else:
        mic_geom = mic_locations
        pos_geom = track_position
        vel_geom = track_velocity

    # Emission-time geometry
    az_deg, el_deg, r_geom, t_obs, _ = hemigen(track_time, pos_geom, vel_geom, mic_geom, speed_of_sound)

    # Subsample emission-time points
    point_stride = int(point_stride)
    if point_stride < 1:
        raise ValueError('point_stride must be >= 1')
    tidx = np.arange(0, track_time.size, point_stride)

    # Refracted rays, at the sampled points only: the filing angle and the
    # reception time replace hemigen's; the spreading and absorption ranges and
    # the receiver's source offset are kept alongside.
    spread_geom = path_geom = ray_offset = ray_valid = None
    if ray_model is not None:
        rays = ray_model(np.asarray(track_position, dtype=float)[tidx], np.asarray(mic_locations, dtype=float))
        shape = (tidx.size, nmics)
        for key in ('depression_deg', 'travel_time', 'spreading_range', 'path_length', 'valid'):
            if np.shape(rays[key]) != shape:
                raise ValueError('ray_model {!r} must be shaped {}, got {}'.format(key, shape, np.shape(rays[key])))
        if np.shape(rays['offset']) != shape + (3,):
            raise ValueError('ray_model offset must be shaped {}'.format(shape + (3,)))
        el_deg = np.array(el_deg, dtype=float)
        t_obs = np.array(t_obs, dtype=float)
        el_deg[tidx] = rays['depression_deg']
        t_obs[tidx] = np.asarray(track_time, dtype=float)[tidx][:, None] + rays['travel_time']
        spread_geom = np.full(r_geom.shape, np.nan)
        path_geom = np.full(r_geom.shape, np.nan)
        spread_geom[tidx] = rays['spreading_range']
        path_geom[tidx] = rays['path_length']
        ray_offset = np.array(rays['offset'], dtype=float)
        if flip_y_for_geometry:
            ray_offset[..., 1] *= -1.0
        ray_valid = np.asarray(rays['valid'], dtype=bool) & np.all(np.isfinite(ray_offset), axis=-1)

    # Regular hemisphere grid (degrees) with seam closure at 360
    azi_grid_deg = np.arange(0.0, 360.0 + 1e-9, float(azi_step))
    elv_grid_deg = np.arange(0.0, 90.0 + 1e-9, float(elv_step))
    AZI_GRID, ELV_GRID = np.meshgrid(azi_grid_deg, elv_grid_deg)

    # Frequency selection based on the first mic's sampling
    # (rounding matches the demo scripts' convention)
    t0_vec = _time_vec_for_mic(0)
    fs0 = float(np.round(1.0 / (t0_vec[1] - t0_vec[0])))
    fmin, fmax = float(freq_range[0]), float(freq_range[1])

    # Narrowband output frequency decimation
    narrowband_stride = int(narrowband_stride)
    if narrowband_stride < 1:
        raise ValueError('narrowband_stride must be >= 1')

    # Accumulate scattered samples
    fazi_list = []
    felv_list = []
    fmic_list = []
    fres_list = []
    frange_list = []
    fheight_list = []
    interp_settings = None
    if interpolation is not None:
        unknown = set(interpolation) - set(ADAPTIVE_INTERPOLATION)
        if unknown:
            raise ValueError('unknown interpolation settings: {}'.format(sorted(unknown)))
        interp_settings = dict(ADAPTIVE_INTERPOLATION, **interpolation)
    oaspl_power_list = []
    spl_a_power_list = []
    if third_octave:
        if third_octave_band_centers_hz is not None:
            band_centers = np.asarray(third_octave_band_centers_hz, dtype=float).ravel()
            band_centers = band_centers[np.isfinite(band_centers)]
            if band_centers.size < 1:
                raise ValueError('third_octave_band_centers_hz must contain at least one finite frequency')
            band_centers = np.unique(band_centers)
            band_centers = band_centers[np.logical_and(band_centers >= float(third_octave_fmin), band_centers <= fmax)]
        else:
            # Precompute band centers (same definition as in third_octave_band_levels);
            # third_octave_fmin is compared within a quarter band so a nominal
            # limit keeps its own band (the 20 Hz band's exact centre is 19.69 Hz).
            k = np.arange(-50, 50)
            band_centers = 1000.0 * (2.0 ** (k / 3.0))
            band_centers = band_centers[np.logical_and(
                band_centers >= float(third_octave_fmin) / 2.0 ** (1.0 / 12.0), band_centers <= fmax)]
        band_power_lists = [list() for _ in range(band_centers.size)]
    else:
        band_centers = np.array([], dtype=float)
        band_power_lists = []

    # Bands the receiver response is evaluated in: the output bands, or the
    # standard centres over the frequency range when there are none.
    if third_octave:
        response_centers = band_centers
    else:
        k_bands = np.arange(-50, 50)
        response_centers = 1000.0 * 2.0 ** (k_bands / 3.0)
        response_centers = response_centers[(response_centers * 2 ** (1 / 6) >= fmin)
                                            & (response_centers / 2 ** (1 / 6) <= fmax)]

    psd_power_lists = []
    f_sel_master = None
    Aweight_db = None

    # Precompute absorption reference range (meters)
    r_ref_m = None
    if apply_absorption_deprop:
        r_ref_m = float(unit_conversion.len_conv(r_ref, from_units=length_units, to_units='m'))

    pref_sq = (20e-6) ** 2

    def _depropagate(lin, amb, r_v, alpha, response=None, r_path=None):
        """Gate against ambient, subtract it, then undo spreading and absorption.

        ``lin`` is (Nf, Npts) linear power (per bin or per band) at the
        microphone; ``amb`` the matching (Nf,) ambient or None; ``alpha`` the
        (Nf,) absorption coefficients in dB/m, used when absorption
        depropagation is on.  Spreading is undone over ``r_v`` and absorption
        over ``r_path`` (the same, without a ray model).
        """
        if r_path is None:
            r_path = r_v
        # Ambient gating and background subtraction have to happen before
        # spreading and absorption depropagation, or an ambient-limited band
        # gets amplified by both.
        if amb is not None:
            gate = lin >= amb[:, None] * 10.0 ** (band_snr_gate_db / 10.0)
            lin = np.where(gate, np.maximum(lin - amb[:, None], 0.0), 0.0)

        # The installation's response, as a power ratio to divide out.
        if response is not None:
            lin = lin * response

        # Spherical spreading depropagation to r_ref: multiply by (r/r_ref)^2
        lin = lin * ((r_v / float(r_ref)) ** 2)[None, :]

        # Optional absorption depropagation back to r_ref
        if apply_absorption_deprop:
            assert r_ref_m is not None
            r_m = unit_conversion.len_conv(r_path, from_units=length_units, to_units='m').astype(float)
            deltaL = np.asarray(alpha, dtype=float)[:, None] * (r_m[None, :] - r_ref_m)
            if max_absorption_correction_db is not None:
                # Drop bins the measurement cannot support before applying the
                # correction, not after: once multiplied they are indisting-
                # uishable from real high-frequency content.
                lin = np.where(deltaL <= float(max_absorption_correction_db), lin, 0.0)
            lin = lin * (10.0 ** (deltaL / 10.0))
        return lin

    for im in range(nmics):
        # PSD spectrogram on observer time axis
        t_vec = _time_vec_for_mic(im)
        fs = float(np.round(1.0 / (t_vec[1] - t_vec[0])))
        f, t_rel, psd_db = spectrogram(pressure_list[im], fs, window_time=window_time, window_overlap=window_overlap)
        t_abs = t_rel + t_vec[0]

        fmask = np.logical_and(f >= fmin, f <= fmax)
        f_sel = f[fmask]
        if f_sel.size < 2:
            raise ValueError('Selected frequency range does not contain enough bins')

        # OASPL, A-weighting and third-octave integration all happen per
        # microphone on that microphone's own frequency grid, so arrays may
        # differ from mic to mic.  Only the narrowband export, which stacks
        # every mic onto one frequency axis, needs a common grid -- arrays
        # recorded at different sample rates (the 2017 Noise Abatement test
        # mixes 25000 and 25600 Hz) are otherwise perfectly usable.
        if narrowband:
            if not np.isclose(fs, fs0, rtol=1e-3, atol=0.0):
                raise ValueError('narrowband=True requires all microphones at the same sample rate')
            if f_sel_master is None:
                f_sel_master = f_sel
            elif f_sel_master.shape != f_sel.shape or not np.allclose(f_sel_master, f_sel, rtol=0.0, atol=0.0):
                raise ValueError('narrowband=True requires identical frequency grids for all microphones')
        elif f_sel_master is None:
            f_sel_master = f_sel

        df = float(f_sel[1] - f_sel[0])

        Aweight_db = np.array([dBAw(fi) for fi in f_sel], dtype=float)
        Aweight_lin = 10.0 ** (Aweight_db / 10.0)

        alpha_db_per_m = None
        if apply_absorption_deprop:
            alpha_db_per_m = np.asarray(atmosphere.attenuation_coefficient(f_sel), dtype=float)

        # Observation times for selected emission points
        tobs_sub = t_obs[tidx, im]
        r_sub = r_geom[tidx, im]
        az_sub = az_deg[tidx, im]
        el_sub = el_deg[tidx, im]

        # Keep only points that can be interpolated in time and, when
        # requested, that lie above the elevation cutoff
        valid = np.logical_and(tobs_sub >= t_abs[0], tobs_sub <= t_abs[-1])
        if min_elevation_deg > 0.0:
            valid = np.logical_and(valid, el_sub >= float(min_elevation_deg))
        if max_range is not None:
            cap = np.full(r_sub.shape, float(max_range))
            if rim_range is not None:
                rim_elevation, rim_cap = float(rim_range[0]), float(rim_range[1])
                cap = np.where(el_sub < rim_elevation, max(rim_cap, float(max_range)), cap)
            valid = np.logical_and(valid, r_sub <= cap)
        if ray_valid is not None:
            valid = np.logical_and(valid, ray_valid[:, im])
        if not np.any(valid):
            continue

        tobs_v = tobs_sub[valid]
        r_v = r_sub[valid]
        spread_v = spread_geom[tidx, im][valid] if spread_geom is not None else r_v
        path_v = path_geom[tidx, im][valid] if path_geom is not None else r_v
        az_v = az_sub[valid]
        el_v = el_sub[valid]

        # Interpolate the PSD at tobs_v in linear power (relative to
        # pref^2/Hz).  Until 2026-09-24 this interpolated in dB, which between
        # frames takes a geometric mean of single-periodogram bins whose values
        # fluctuate strongly: broadband levels came out ~0.7 dB low in every
        # band, and in OASPL.
        psd_sel_db = psd_db[fmask, :]
        psd_sel_lin = 10.0 ** (psd_sel_db / 10.0)
        psd_v_lin = np.empty((f_sel.size, tobs_v.size), dtype=float)
        for fi in range(f_sel.size):
            psd_v_lin[fi, :] = np.interp(tobs_v, t_abs, psd_sel_lin[fi, :])

        # Filter-bank band levels on the spectrogram's own frames, so they
        # share its time base (and its ambient treatment below).
        frame_length = int(2.0 ** nextpow2(window_time * fs))
        band_frames = None
        if use_filter_bank:
            band_frames = pa_filters.third_octave_filter_bank(
                pressure_list[im], fs, band_centers, t_rel, frame_length) / pref_sq
            band_frames = np.where(np.isfinite(band_frames), band_frames, 0.0)

        # Per-bin (and per-band) ambient reference.
        amb_lin = None
        amb_band = None
        if ambient_pressure_list is not None:
            amb_p = ambient_pressure_list[im]
            amb_fs = fs
            if ambient_time_list is not None:
                amb_t = ambient_time_list[im]
                if amb_t.size >= 2:
                    amb_fs = float(np.round(1.0 / (amb_t[1] - amb_t[0])))
            if not np.isclose(amb_fs, fs, rtol=1e-6, atol=0.0):
                raise ValueError('ambient_pressure for mic {:d} is sampled at {:g} Hz but the run is at {:g} Hz; '
                                 'the ambient recording must come from the same channel and sample rate'
                                 .format(im, amb_fs, fs))
            # Same window settings as the run, so the ambient lands on the
            # identical frequency grid and can be applied bin by bin.
            f_amb, _, psd_amb_db = spectrogram(amb_p, amb_fs, window_time=window_time,
                                               window_overlap=window_overlap)
            if f_amb.shape != f.shape or psd_amb_db.shape[1] < 1:
                raise ValueError('ambient_pressure for mic {:d} did not yield a usable spectrogram on the run '
                                 'frequency grid (is the recording at least one window long?)'.format(im))
            amb_lin = 10.0 ** (np.median(psd_amb_db[fmask, :], axis=1) / 10.0)
            if use_filter_bank:
                _, t_amb, _ = scipy.signal.spectrogram(
                    amb_p, amb_fs, scipy.signal.get_window('hann', frame_length),
                    noverlap=round(window_overlap * frame_length), mode='psd')
                amb_band = np.nanmedian(pa_filters.third_octave_filter_bank(
                    amb_p, amb_fs, band_centers, t_amb, frame_length), axis=1) / pref_sq
        elif ambient_time_range is not None:
            amb_mask = np.logical_and(t_abs >= float(ambient_time_range[0]),
                                      t_abs <= float(ambient_time_range[1]))
            if not np.any(amb_mask):
                raise ValueError('ambient_time_range contains no spectrogram frames')
            amb_lin = 10.0 ** (np.median(psd_sel_db[:, amb_mask], axis=1) / 10.0)
            if use_filter_bank:
                amb_band = np.median(band_frames[:, amb_mask], axis=1)
        elif ambient_percentile is not None:
            amb_lin = 10.0 ** (np.percentile(psd_sel_db, float(ambient_percentile), axis=1) / 10.0)
            if use_filter_bank:
                amb_band = np.percentile(band_frames, float(ambient_percentile), axis=1)

        response_bins = response_bands = None
        if receiver_response_db is not None:
            offset = (ray_offset[:, im][valid] if ray_offset is not None
                      else pos_geom[tidx][valid] - mic_geom[im])
            gain_db = np.asarray(receiver_response_db(im, response_centers, offset), dtype=float)
            if gain_db.shape != (response_centers.size, tobs_v.size) or not np.all(np.isfinite(gain_db)):
                raise ValueError('receiver_response_db must return finite values shaped (bands, points)')
            inverse = 10.0 ** (-gain_db / 10.0)
            # Each bin takes the band the band sums below put it in: nominal
            # centres get their base-10 edges (see third_octave_band_edges)
            _, response_upper = third_octave_band_edges(response_centers)
            band_of_bin = np.clip(np.searchsorted(response_upper, f_sel, side='right'),
                                  0, response_centers.size - 1)
            response_bins = inverse[band_of_bin]
            response_bands = inverse

        psd_v_lin = _depropagate(psd_v_lin, amb_lin, spread_v, alpha_db_per_m, response_bins, path_v)

        # OASPL power over selected frequency range
        power_oaspl = np.sum(psd_v_lin * df, axis=0)
        power_spl_a = np.sum((psd_v_lin * Aweight_lin[:, None]) * df, axis=0)

        fazi_list.append(az_v)
        felv_list.append(el_v)
        fmic_list.append(np.full(az_v.size, im))
        frange_list.append(r_v)
        fheight_list.append(pos_geom[tidx][valid][:, 2] - mic_geom[im][2])
        if interpolation is not None:
            # The arc one analysis window smears over: the source's motion across
            # the line of sight in window_time, and its own size, over the range.
            line = pos_geom[tidx][valid] - mic_geom[im]
            unit_line = line / np.linalg.norm(line, axis=1)[:, None]
            v = np.asarray(vel_geom, dtype=float)[tidx][valid]
            across = np.linalg.norm(v - np.sum(v * unit_line, axis=1)[:, None] * unit_line, axis=1)
            extent = float(interp_settings['source_extent'])
            fres_list.append(np.degrees(np.hypot(across * float(window_time), extent) / r_v))
        oaspl_power_list.append(power_oaspl)
        spl_a_power_list.append(power_spl_a)

        if narrowband:
            psd_power_lists.append(psd_v_lin)

        if use_filter_bank:
            # Band power at the emission points' observer times, interpolated
            # in linear power as the PSD is, then depropagated band by band.
            band_v = np.empty((band_centers.size, tobs_v.size), dtype=float)
            for ib in range(band_centers.size):
                band_v[ib, :] = np.interp(tobs_v, t_abs, band_frames[ib, :])
            alpha_band = (np.asarray(atmosphere.attenuation_coefficient(band_centers), dtype=float)
                          if apply_absorption_deprop else None)
            band_v = _depropagate(band_v, amb_band, spread_v, alpha_band, response_bands, path_v)
            for ib in range(band_centers.size):
                band_power_lists[ib].append(band_v[ib, :])
        elif third_octave:
            # Integrate to third-octave bands in linear power, on edges that
            # tile even when the centres are nominal (see third_octave_band_edges)
            band_lower, band_upper = third_octave_band_edges(band_centers)
            for ib, (f_lower, f_upper) in enumerate(zip(band_lower, band_upper)):
                band_mask = np.logical_and(f_sel >= f_lower, f_sel < f_upper)
                if np.any(band_mask):
                    band_power = np.sum(psd_v_lin[band_mask, :] * df, axis=0)
                else:
                    # Keep shape/point counts consistent even if a band is empty at the
                    # selected frequency resolution/range. This band will evaluate to -inf dB.
                    band_power = np.zeros(tobs_v.size, dtype=float)
                band_power_lists[ib].append(band_power)

    if len(oaspl_power_list) == 0:
        raise ValueError('No valid hemisphere samples were generated (check time alignment and inputs)')

    fazi_pts = np.concatenate(fazi_list)
    felv_pts = np.concatenate(felv_list)
    P_oaspl_pts = np.concatenate(oaspl_power_list)
    P_spl_a_pts = np.concatenate(spl_a_power_list)

    # Enforce azimuth periodicity: duplicate scattered points at ±360 and close seam column
    fazi_ext = np.concatenate((fazi_pts, fazi_pts + 360.0, fazi_pts - 360.0))
    felv_ext = np.concatenate((felv_pts, felv_pts, felv_pts))

    # Cells with no sample within reach come out NaN (no data) and cells whose
    # samples carry no energy -inf, never a finite floor; see power_to_db.
    interpolation_info = None
    if interp_settings is not None:
        # One weight matrix for every quantity gridded: the radius is the node's,
        # not the band's.  Geodesic distances, so the samples once, not at +-360.
        weights, node_radius, node_gap = adaptive_idw_weights(
            ELV_GRID.ravel(), AZI_GRID.ravel(), felv_pts, fazi_pts, np.concatenate(fmic_list),
            np.concatenate(fres_list), k=interp_settings['k'], min_mics=interp_settings['min_mics'],
            kappa=interp_settings['kappa'], resolution_factor=interp_settings['resolution_factor'],
            max_radius_deg=interp_settings['max_radius_deg'])
        node_gap = node_gap.reshape(ELV_GRID.shape)

        def grid_power(power_pts):
            g = np.asarray(weights @ np.asarray(power_pts, dtype=float)).reshape(ELV_GRID.shape)
            return np.where(node_gap, np.nan, g)
        interpolation_info = dict(interp_settings, mode='adaptive',
                                  radius_deg=node_radius.reshape(ELV_GRID.shape),
                                  gaps=int(node_gap[:, :-1].sum()) if node_gap.shape[1] > 1 else int(node_gap.sum()))
    else:
        def grid_power(power_pts):
            ext = np.concatenate((power_pts, power_pts, power_pts))
            return shepIDW(ELV_GRID, AZI_GRID, felv_ext, fazi_ext, ext, rmax=float(rmax))

    P_oaspl_grid = grid_power(P_oaspl_pts)
    oaspl_db = power_to_db(P_oaspl_grid)
    if oaspl_db.shape[1] > 1:
        oaspl_db[:, -1] = oaspl_db[:, 0]

    P_spl_a_grid = grid_power(P_spl_a_pts)
    spl_a_db = power_to_db(P_spl_a_grid)
    if spl_a_db.shape[1] > 1:
        spl_a_db[:, -1] = spl_a_db[:, 0]

    out = {
        'azi_grid_deg': azi_grid_deg,
        'elv_grid_deg': elv_grid_deg,
        'oaspl_db': oaspl_db,
        'spl_a_db': spl_a_db,
        'metadata': {
            'r_ref': float(r_ref),
            'length_units': str(length_units),
            'freq_range_hz': (fmin, fmax),
            'window_time': float(window_time),
            'window_overlap': float(window_overlap),
            'point_stride': int(point_stride),
            'rmax_deg': float(rmax),
            'rim_range': None if rim_range is None else (float(rim_range[0]), float(rim_range[1])),
            'apply_absorption_deprop': bool(apply_absorption_deprop),
            'third_octave_method': str(third_octave_method),
            # Depropagation removes absorption over r - r_ref only, so the
            # sphere still carries absorption over r_ref in this atmosphere.
            # Exporters for formats with a different convention (NORAH2) need
            # it to take that back out.
            'atmosphere': {
                'temperature_k': float(atmosphere.temperature),
                'pressure_kpa': float(atmosphere.pressure),
                'relative_humidity': float(atmosphere.relative_humidity),
            },
            'flip_y_for_geometry': bool(flip_y_for_geometry),
            'return_scattered': bool(return_scattered),
            'narrowband': bool(narrowband),
            'narrowband_stride': int(narrowband_stride),
        }
    }
    if interpolation_info is not None:
        out['interpolation'] = interpolation_info

    scattered: Optional[dict[str, Any]] = None
    if return_scattered:
        scattered = cast(dict[str, Any], {
            'azi_deg': fazi_pts,
            'elv_deg': felv_pts,
            'oaspl_power': P_oaspl_pts,
            'spl_a_power': P_spl_a_pts,
            'mic': np.concatenate(fmic_list),
            # Straight-line slant range and the source's height above the microphone, in
            # length_units, at each sample's emission time.
            'range': np.concatenate(frange_list),
            'source_height': np.concatenate(fheight_list),
        })
        if fres_list:
            scattered['resolution_deg'] = np.concatenate(fres_list)
        out['scattered'] = scattered

    if third_octave:
        # Concatenate scattered band powers and interpolate per band
        band_grids_db = np.full((band_centers.size, ELV_GRID.shape[0], ELV_GRID.shape[1]), -np.inf, dtype=float)
        for ib in range(band_centers.size):
            if len(band_power_lists[ib]) == 0:
                continue
            P_band_pts = np.concatenate(band_power_lists[ib])
            if P_band_pts.size != fazi_pts.size:
                raise ValueError('Internal error: third-octave sample count does not match scattered angle count')
            P_band_grid = grid_power(P_band_pts)
            band_grids_db[ib, :, :] = power_to_db(P_band_grid)
            band_grids_db[ib, :, -1] = band_grids_db[ib, :, 0]

        out['third_octave'] = {
            'band_centers_hz': band_centers,
            'bands_db': band_grids_db,
        }

        if return_scattered:
            assert scattered is not None
            if band_centers.size > 0:
                band_power_pts = np.vstack([np.concatenate(band_power_lists[ib]) for ib in range(band_centers.size)])
                if band_power_pts.shape[1] != fazi_pts.size:
                    raise ValueError('Internal error: third-octave sample count does not match scattered angle count')
                scattered['third_octave'] = {
                    'band_centers_hz': band_centers,
                    'bands_db': power_to_db(band_power_pts),
                }
            else:
                scattered['third_octave'] = {
                    'band_centers_hz': band_centers,
                    'bands_db': np.empty((0, fazi_pts.size), dtype=float),
                }

    if narrowband:
        assert f_sel_master is not None
        # Concatenate scattered PSD power samples into (Nf, Np)
        if len(psd_power_lists) == 0:
            raise ValueError('No narrowband PSD samples were accumulated')

        psd_power_pts = np.concatenate(psd_power_lists, axis=1)
        if psd_power_pts.shape[1] != fazi_pts.size:
            raise ValueError('Internal error: PSD sample count does not match scattered angle count')

        f_nb = f_sel_master[::narrowband_stride]
        psd_grid_db = np.full((f_nb.size, ELV_GRID.shape[0], ELV_GRID.shape[1]), -np.inf, dtype=float)
        for i_f, fi in enumerate(range(0, f_sel_master.size, narrowband_stride)):
            P_f_grid = grid_power(psd_power_pts[fi, :])
            psd_grid_db[i_f, :, :] = power_to_db(P_f_grid)
            psd_grid_db[i_f, :, -1] = psd_grid_db[i_f, :, 0]

        out['narrowband'] = {
            'frequency_hz': f_nb,
            'psd_db': psd_grid_db,
        }

        if return_scattered:
            assert scattered is not None
            scattered['narrowband'] = {
                'frequency_hz': f_nb,
                'psd_db': power_to_db(psd_power_pts[::narrowband_stride, :]),
            }

    return out


def array_coverage(ymics, altitude, xmin=-1000.0, xmax=1000.0, speed=100.0, rate=0.1, speed_of_sound=1135.):
    """
    Calculate the spherical coverage for an overflight of a linear microphone array
    Args:
        ymics: sideline distances of observers (at X=0 where +X is direction the vehicle is traveling)
        altitude: altitude of the vehicle
        xmin: optional, start distance of vehicle trajectory (default -1000)
        xmax: optional, end distance of vehicle trajectory (default 1000)
        speed: optional, default 100
        rate: optional, sampling period of emission time points, default 0.1
        speed_of_sound: optional, speed of sound, default 1135

    Returns: tuple (azimuth, elevation, r)
    WHERE
    azimith are the azimuth angles on the sphere for each emission time
    elevation are the elevation angles on the sphere for each emission time
    r are the propagation distances for each emission time
    """
    observers = np.zeros((len(ymics), 3))
    observers[:, 1] = ymics
    tmin = 0
    tmax = (xmax - xmin) / speed
    time = np.arange(tmin, tmax, rate)
    source = np.zeros((len(time), 3))
    source[:, 0] = xmin + time * speed
    source[:, 2] = altitude

    velocity = np.zeros_like(source)
    velocity[:, 0] = speed

    azimuth, elevation, r, t_observer, mach_r = hemigen(time, source, velocity, observers, speed_of_sound)
    return azimuth, elevation, r


def hover_array_coverage(radial_positions, mic_azimuths, altitudes, headings, speed_of_sound=1135.0):
    """
    Calculate the spherical coverage for a hovering vehicle with microphones placed
    on a circle (or rings) around the source location.

    Args:
        radial_positions: radial distances of observers from the hover point (same length units as altitude)
        mic_azimuths: observer azimuths around the hover point, degrees (0 deg = +X, 90 deg = +Y)
        altitudes: one or more hover altitudes (same length units as radial_positions)
        headings: one or more vehicle headings, degrees (0 deg = +X, 90 deg = +Y)
        speed_of_sound: optional, speed of sound, default 1135

    Returns: tuple (azimuth, elevation, r)
    WHERE
    azimuth are the azimuth angles on the sphere for each hover condition (time) and mic
    elevation are the elevation angles on the sphere for each hover condition (time) and mic
    r are the propagation distances for each hover condition (time) and mic
    """
    radial_positions = np.atleast_1d(radial_positions).astype(float)
    mic_azimuths = np.atleast_1d(mic_azimuths).astype(float)
    altitudes = np.atleast_1d(altitudes).astype(float)
    headings = np.atleast_1d(headings).astype(float)

    if radial_positions.size == 0 or mic_azimuths.size == 0:
        raise ValueError('radial_positions and mic_azimuths must be non-empty')
    if altitudes.size == 0 or headings.size == 0:
        raise ValueError('altitudes and headings must be non-empty')

    if radial_positions.size != mic_azimuths.size:
        raise ValueError('radial_positions and mic_azimuths must have the same length')

    # Build mic locations in the ground plane centered on the hover point.
    az_rad = np.radians(mic_azimuths)
    obs_x = radial_positions * np.cos(az_rad)
    obs_y = radial_positions * np.sin(az_rad)
    observers = np.column_stack((obs_x, obs_y, np.zeros(obs_x.size)))

    # Enumerate hover conditions; time is arbitrary and only used to align rows.
    alt_grid, head_grid = np.meshgrid(altitudes, headings, indexing='ij')
    n_combo = alt_grid.size
    time = np.arange(n_combo, dtype=float)

    source = np.zeros((n_combo, 3), dtype=float)
    source[:, 2] = alt_grid.ravel()

    heading_rad = np.radians(head_grid.ravel())
    # Unit velocity vector defines vehicle heading for azimuth convention.
    velocity = np.zeros_like(source)
    velocity[:, 0] = np.cos(heading_rad)
    velocity[:, 1] = np.sin(heading_rad)

    azimuth, elevation, r, t_observer, mach_r = hemigen(time, source, velocity, observers, speed_of_sound)
    return azimuth, elevation, r


def takeoff_array_coverage(
        radial_positions,
        mic_azimuths,
        ground_distances,
        *,
        initial_altitude=0.0,
        flight_path_angle=8.0,
        heading=0.0,
    source_origin_xy=(0.0, 0.0),
        speed_of_sound=1135.0,
):
    """
    Calculate spherical coverage for a takeoff/climb trajectory over a circular/ring array.

    Args:
        radial_positions: radial distances of observers from the reference point.
        mic_azimuths: observer azimuths around the reference point, deg (0 deg = +X, 90 deg = +Y).
        ground_distances: source ground-track distances from reference point.
        initial_altitude: source altitude at zero ground distance.
        flight_path_angle: climb angle in deg above local horizon.
        heading: source heading in deg (0 deg = +X, 90 deg = +Y).
        source_origin_xy: (x0, y0) source location at zero ground distance in
            the same local coordinates as observers. Default is (0, 0).
        speed_of_sound: speed of sound.

    Returns: tuple (azimuth, elevation, r)
    WHERE
    azimuth are the observer azimuth angles on the source hemisphere
    elevation are the observer elevation angles on the source hemisphere
    r are source-observer propagation distances
    """
    radial_positions = np.atleast_1d(radial_positions).astype(float)
    mic_azimuths = np.atleast_1d(mic_azimuths).astype(float)
    ground_distances = np.atleast_1d(ground_distances).astype(float)

    if radial_positions.size == 0 or mic_azimuths.size == 0:
        raise ValueError('radial_positions and mic_azimuths must be non-empty')
    if ground_distances.size == 0:
        raise ValueError('ground_distances must be non-empty')
    if radial_positions.size != mic_azimuths.size:
        raise ValueError('radial_positions and mic_azimuths must have the same length')

    az_rad = np.radians(mic_azimuths)
    obs_x = radial_positions * np.cos(az_rad)
    obs_y = radial_positions * np.sin(az_rad)
    observers = np.column_stack((obs_x, obs_y, np.zeros(obs_x.size)))

    heading_rad = np.radians(float(heading))
    fpa_rad = np.radians(float(flight_path_angle))
    origin_xy = np.asarray(source_origin_xy, dtype=float).ravel()
    if origin_xy.size != 2:
        raise ValueError('source_origin_xy must have exactly 2 values: (x0, y0)')

    source = np.zeros((ground_distances.size, 3), dtype=float)
    source[:, 0] = origin_xy[0] + ground_distances * np.cos(heading_rad)
    source[:, 1] = origin_xy[1] + ground_distances * np.sin(heading_rad)
    source[:, 2] = float(initial_altitude) + ground_distances * np.tan(fpa_rad)

    velocity = np.zeros_like(source)
    velocity[:, 0] = np.cos(heading_rad)
    velocity[:, 1] = np.sin(heading_rad)
    velocity[:, 2] = np.tan(fpa_rad)

    time = np.arange(ground_distances.size, dtype=float)
    azimuth, elevation, r, t_observer, mach_r = hemigen(time, source, velocity, observers, speed_of_sound)
    return azimuth, elevation, r


def hover_array_coverage_plot(radial_positions, mic_azimuths, altitudes, headings, speed_of_sound=1135.0):
    """
    Plot the spherical coverage for hovering conditions with scattered microphones.

    Args:
        radial_positions: radial distances of observers from the hover point (same length units as altitude)
        mic_azimuths: observer azimuths around the hover point, degrees (0 deg = +X, 90 deg = +Y)
        altitudes: one or more hover altitudes (same length units as radial_positions)
        headings: one or more vehicle headings, degrees (0 deg = +X, 90 deg = +Y)
        speed_of_sound: optional, speed of sound, default 1135

    Returns: tuple (fig, ax, cs)
    WHERE
    fig is a matplotlib handle to the figure
    ax is a matplotlib handle to the plot axis
    cs is a matplotlib handle to the data points
    """
    azimuth, elevation, _ = hover_array_coverage(radial_positions, mic_azimuths, altitudes,
                                                 headings, speed_of_sound)
    fig, ax, cs = lambert_ea_points(np.radians(azimuth), np.radians(elevation))
    return fig, ax, cs


def array_coverage_plot(ymics, altitude, xmin=-1000.0, xmax=1000.0, speed=100.0, rate=0.1, speed_of_sound=1135.):
    """
    Plots the spherical coverage for an overflight of a linear microphone array
    Args:
        ymics: sideline distances of observers (at X=0 where +X is direction the vehicle is traveling)
        altitude: altitude of the vehicle
        xmin: optional, start distance of vehicle trajectory (default -1000)
        xmax: optional, end distance of vehicle trajectory (default 1000)
        speed: optional, default 100
        rate: optional, sampling period of emission time points, default 0.1
        speed_of_sound: optional, speed of sound, default 1135

    Returns: tuple (fig, ax, cs)
    WHERE
    fig is a matplotlib handle to the figure
    ax is a matplotlib handle to the plot axis
    cs is a matplotlib handle to the data points
    """
    azimuth, elevation, _ = array_coverage(ymics, altitude, xmin, xmax, speed, rate, speed_of_sound)
    fig, ax, cs = lambert_ea_points(np.radians(azimuth), np.radians(elevation))
    return fig, ax, cs


def load_nc_sphere(filename):
    """
    Load data from an AAM/RNM style acoustic hemisphere
    Args:
        filename: path to netCDF sphere file

    Returns: tuple (amplitude, phi, theta, frequency, radius, speed, flight_path_angle)
    WHERE
    amplitude is a phi x theta x frequency matrix of SPL amplitudes, dB
    phi is an array of lateral angles (degrees)
    theta is an array of longitudinal angles (degree)
    frequency is an array of frequencies, Hz
    radius is the sphere radius (feet)
    speed is the sphere speed (knots)
    flight_path_angle is the sphere flight path angle (degrees)

    Every array is a plain float array.  netCDF4 returns masked arrays, and
    their masked cells came through arithmetic as masked scalars that
    :func:`safe_log10` cannot test (409 RuntimeWarnings reducing one synthetic
    flyby sphere), while other results came out right only because a mask
    happened to propagate.  Anything netCDF4 masks (a _FillValue, missing_value or
    valid-range hit) is NaN instead, which :func:`mask_missing_levels` reads
    as missing along with the in-band sentinels.
    """
    def read(variable):
        return np.ma.filled(np.ma.asarray(variable[:], dtype=float), np.nan)

    with Dataset(filename, mode='r') as file_handle:
        # Noise spheres sometimes contain NaN, which netCDF4's masking compares
        # against; they are handled downstream, so do not warn about them.
        with np.errstate(invalid='ignore'):
            phi = read(file_handle.variables['PHI'])
            theta = read(file_handle.variables['THETA'])
            frequency = read(file_handle.variables['FREQUENCY'])
            amplitude = read(file_handle.variables['AMPLITUDE'])
            radius = read(file_handle.variables['RADIUS'])
            speed = read(file_handle.variables['SPEED'])
            flight_path_angle = read(file_handle.variables['FLIGHT_PATH_ANGLE'])
    return amplitude, phi, theta, frequency, radius, speed, flight_path_angle


#: Legacy AAM spheres blank-pad the ``unit`` attribute of every scalar variable
#: to this width (a Fortran CHARACTER(20) field written verbatim).
AAM_SCALAR_UNIT_WIDTH = 20

#: Level written into AMPLITUDE where a band carries no usable data.  The AAM
#: code masks known-bad values with -999, so that is what a sphere meant for it
#: should carry.  Note the shipped 2017 spheres do not use it -- they mask with
#: NaN -- and older panam output used a large positive sentinel, so
#: :func:`mask_missing_levels` accepts all three on read.
AAM_MISSING_LEVEL = -999.0

#: Anything at or below this reads as masked.  Loose enough to survive the
#: float32 round trip AMPLITUDE goes through, and far below any real source
#: level, so it cannot swallow a measurement.
AAM_MISSING_THRESHOLD = -998.0

#: Older panam spheres (and some AAM-style tooling) mask with a large positive
#: value instead.  Still accepted on read so existing files keep working.
AAM_LEGACY_MISSING_THRESHOLD = 1.0e34


def mask_missing_levels(amplitude):
    """Return ``amplitude`` with every masked band as -inf, i.e. no energy.

    Three conventions appear in the wild and all three are accepted: NaN (what
    the shipped 2017 Noise Abatement spheres use), -999 (what the AAM code
    masks with, and what this module now writes), and a large positive value
    (older panam output).  The array comes back as a plain float array, never a
    masked one, so downstream arithmetic behaves the same whichever it was.
    """
    amplitude = np.asarray(amplitude, dtype=float).copy()
    missing = (np.isnan(amplitude)
               | (amplitude <= AAM_MISSING_THRESHOLD)
               | (amplitude > AAM_LEGACY_MISSING_THRESHOLD))
    amplitude[missing] = -np.inf
    return amplitude


def _sample_hemisphere_levels(hemisphere, mode, azi_q_deg, elv_q_deg, minimum_level_db):
    """Band levels of a :func:`depropagate_hemisphere` result at arbitrary directions.

    The shared core of the sphere exporters: pick the spectrum ``mode`` selects,
    then interpolate each band over the UMAPR azimuth/elevation grid in linear
    power.  Directions off the grid, and levels below ``minimum_level_db``, come
    back as -inf; each writer turns that into its own missing-value convention.

    Args:
        hemisphere: Output dict from :func:`depropagate_hemisphere`.
        mode: 'auto', 'third_octave', or 'narrowband'. 'auto' prefers third-octave.
        azi_q_deg, elv_q_deg: UMAPR query directions (degrees), any matching shape.
        minimum_level_db: see :func:`write_aam_hemisphere_netcdf`.

    Returns:
        (frequency_hz, levels_db) with levels_db shaped ``azi_q_deg.shape + (Nf,)``.
    """
    if not isinstance(hemisphere, dict):
        raise TypeError('hemisphere must be a dict returned by depropagate_hemisphere')

    azi_grid_deg = np.asarray(hemisphere.get('azi_grid_deg', []), dtype=float)
    elv_grid_deg = np.asarray(hemisphere.get('elv_grid_deg', []), dtype=float)
    if azi_grid_deg.ndim != 1 or azi_grid_deg.size < 2:
        raise ValueError('hemisphere[\'azi_grid_deg\'] must be a 1D array with at least 2 elements')
    if elv_grid_deg.ndim != 1 or elv_grid_deg.size < 2:
        raise ValueError('hemisphere[\'elv_grid_deg\'] must be a 1D array with at least 2 elements')

    # Determine export spectrum (frequency + band levels) from hemisphere.
    selected_mode = str(mode).lower()
    if selected_mode == 'auto':
        if 'third_octave' in hemisphere:
            selected_mode = 'third_octave'
        elif 'narrowband' in hemisphere:
            selected_mode = 'narrowband'
        else:
            selected_mode = 'none'

    if selected_mode == 'third_octave':
        if 'third_octave' not in hemisphere:
            raise ValueError('mode=third_octave but hemisphere does not include third_octave data')
        band_centers_hz = np.asarray(hemisphere['third_octave']['band_centers_hz'], dtype=float)
        band_levels_db_umapr = np.asarray(hemisphere['third_octave']['bands_db'], dtype=float)
        if band_levels_db_umapr.ndim != 3:
            raise ValueError('hemisphere[\'third_octave\'][\'bands_db\'] must be (Nf, Nelv, Nazi)')
        frequency_hz = band_centers_hz
        levels_db_umapr = band_levels_db_umapr
    elif selected_mode == 'narrowband':
        if 'narrowband' not in hemisphere:
            raise ValueError('mode=narrowband but hemisphere does not include narrowband data')
        frequency_hz = np.asarray(hemisphere['narrowband']['frequency_hz'], dtype=float)
        psd_db_umapr = np.asarray(hemisphere['narrowband']['psd_db'], dtype=float)
        if psd_db_umapr.ndim != 3:
            raise ValueError('hemisphere[\'narrowband\'][\'psd_db\'] must be (Nf, Nelv, Nazi)')
        if frequency_hz.size < 2:
            raise ValueError('narrowband frequency_hz must have at least 2 elements')
        # Convert PSD (dB re pref^2/Hz) to approximate per-bin band level (dB re pref^2)
        # so that AAM-style OASPL integration (sum of 10^(SPL/10)) is meaningful.
        df = float(np.median(np.diff(frequency_hz)))
        psd_lin = np.power(10.0, psd_db_umapr / 10.0)
        band_power = psd_lin * df
        eps = np.finfo(float).tiny
        levels_db_umapr = 10.0 * np.log10(np.maximum(band_power, eps))
    else:
        raise ValueError('hemisphere must include third_octave or narrowband data; run depropagate_hemisphere with third_octave=True and/or narrowband=True')

    if levels_db_umapr.shape[1] != elv_grid_deg.size or levels_db_umapr.shape[2] != azi_grid_deg.size:
        raise ValueError('Spectrum grid shape does not match elv_grid_deg/azi_grid_deg')

    # Ensure azimuth periodicity is represented in the interpolation grid.
    # We need the seam column at (azi0 + 360 deg) with values matching azi0,
    # otherwise points near 360 deg can fall out-of-bounds and become missing.
    azi_axis = np.asarray(azi_grid_deg, dtype=float).ravel()
    if np.any(np.diff(azi_axis) <= 0.0) or np.any(np.diff(elv_grid_deg) <= 0.0):
        raise ValueError('azi_grid_deg and elv_grid_deg must be strictly increasing')

    azi0 = float(azi_axis[0])
    azi_end = azi0 + 360.0
    if np.isclose(azi_axis[-1], azi_end, rtol=0.0, atol=1e-6):
        # Seam column present; ensure the last column matches the first.
        levels_db_umapr = levels_db_umapr.copy()
        levels_db_umapr[:, :, -1] = levels_db_umapr[:, :, 0]
    else:
        # No seam column; append one.
        azi_axis = np.concatenate((azi_axis, np.array([azi_end], dtype=float)))
        seam_col = levels_db_umapr[:, :, 0:1]
        levels_db_umapr = np.concatenate((levels_db_umapr, seam_col), axis=2)

    query_shape = np.shape(azi_q_deg)
    # Wrap query azimuths into the interpolation axis range [azi0, azi0+360)
    azi_q = (np.asarray(azi_q_deg, dtype=float).reshape(-1) - azi0) % 360.0 + azi0
    elv_q = np.asarray(elv_q_deg, dtype=float).reshape(-1)
    # Query points are (elv, azi) pairs in degrees
    pts = np.column_stack((elv_q, azi_q))

    nfreq = int(np.asarray(frequency_hz).size)
    levels_db = np.full((pts.shape[0], nfreq), -np.inf, dtype=float)

    # Interpolate each band in linear power and convert to dB.
    eps = np.finfo(float).tiny
    for k in range(nfreq):
        Pk = np.power(10.0, levels_db_umapr[k, :, :] / 10.0)
        # Treat non-finite levels as zero power.
        Pk[~np.isfinite(Pk)] = 0.0
        interp = RegularGridInterpolator(
            (elv_grid_deg, azi_axis),
            Pk,
            bounds_error=False,
            fill_value=0.0,
        )
        Pq = interp(pts)
        Lq = np.full_like(Pq, -np.inf, dtype=float)
        pos = Pq > 0.0
        if np.any(pos):
            Lq[pos] = 10.0 * np.log10(np.maximum(Pq[pos], eps))
        if np.isfinite(minimum_level_db):
            Lq[Lq < float(minimum_level_db)] = -np.inf
        levels_db[:, k] = Lq

    return frequency_hz, levels_db.reshape(query_shape + (nfreq,))


def write_aam_hemisphere_netcdf(
        filename,
        hemisphere,
        *,
        mode: str = 'auto',
        phi_deg=None,
        theta_deg=None,
        speed_knots: float = 0.0,
        flight_path_angle_deg: float = 0.0,
        radius_ft: Optional[float] = None,
    title: Optional[str] = None,
    BB: float = 1.0,
    NB: float = 0.0,
    PT: float = 0.0,
    doppler_shift_removed: float = 0.0,
    empty_weight_lb: float = 0.0,
    fuel_weight_lb: float = 0.0,
    load_weight_lb: float = 0.0,
    pylon_angle_deg: float = 90.0,
    masttilt_deg: float = 0.0,
    xyz_ft=(0.0, 0.0, 0.0),
        minimum_level_db: float = -100.0,
        overwrite: bool = True,
):
    """Write a depropagated acoustic hemisphere to an AAM/RNM-style netCDF sphere.

    This exports a hemisphere in the same variable naming convention expected by
    :func:`load_nc_sphere` (and thus AAM-style tooling):

    - ``PHI`` (deg), ``THETA`` (deg), ``FREQUENCY`` (Hz)
    - ``AMPLITUDE`` (dB), shape (Nphi, Ntheta, Nfreq)
    - ``RADIUS`` (ft), ``SPEED`` (knots), ``FLIGHT_PATH_ANGLE`` (deg)

    The input ``hemisphere`` must be the dict returned by :func:`depropagate_hemisphere`
    with either ``third_octave=True`` or ``narrowband=True`` enabled.

    Notes
    -----
    ``depropagate_hemisphere`` produces a regular grid in UMAPR (azimuth/elevation).
    AAM spheres are stored on a regular grid in ART (phi/theta). This function
    interpolates the UMAPR grid onto a separable ART grid (``phi_deg`` x ``theta_deg``)
    in the *linear power* domain and converts back to dB.

    Args:
        filename: Path to the output netCDF file.
        hemisphere: Output dict from :func:`depropagate_hemisphere`.
        mode: 'auto', 'third_octave', or 'narrowband'. 'auto' prefers third-octave.
        phi_deg: Optional 1D ART phi grid (degrees). Default: -90..90 in 10 degree steps.
        theta_deg: Optional 1D ART theta grid (degrees). Default: 0..180 in 5 degree steps.
        speed_knots: Stored into ``SPEED`` (knots).
        flight_path_angle_deg: Stored into ``FLIGHT_PATH_ANGLE`` (deg).
        radius_ft: Optional override for ``RADIUS`` (ft). If None, derived from hemisphere metadata ``r_ref``.
        minimum_level_db: bands below this level are written as the AAM missing
            sentinel rather than as a number. Hemispheres built before
            2026-09-26 carry a -3076 dB floor where bands were gated to zero
            power (and where there was no data), which interpolates to levels
            near -3000 dB. Those are zero energy and consumers handle them, but
            "not measured" is what they mean and what AAM has a convention for.
            The default sits far below any real measurement at a 100 ft sphere
            radius, so it catches the gated bands and nothing else. Pass -inf
            to keep every finite level.
        overwrite: If False, raises when filename exists.

    Returns:
        None
    """

    if not overwrite and os.path.exists(filename):
        raise FileExistsError(f'Output file already exists: {filename}')

    if not isinstance(hemisphere, dict):
        raise TypeError('hemisphere must be a dict returned by depropagate_hemisphere')

    meta = hemisphere.get('metadata', {}) if isinstance(hemisphere.get('metadata', {}), dict) else {}

    # Default ART grids.
    # A common AAM/RNM convention (and the included example sphere) uses:
    #   PHI:   -90..90   (lateral)
    #   THETA: 0..180    (longitudinal)
    # The step sizes can vary by dataset; pass phi_deg/theta_deg explicitly if you
    # need an exact grid.
    if phi_deg is None:
        phi_deg = np.arange(-90.0, 90.0 + 1e-9, 10.0)
    if theta_deg is None:
        theta_deg = np.arange(0.0, 180.0 + 1e-9, 5.0)
    phi_deg = np.asarray(phi_deg, dtype=float).ravel()
    theta_deg = np.asarray(theta_deg, dtype=float).ravel()
    if phi_deg.size < 2 or theta_deg.size < 2:
        raise ValueError('phi_deg and theta_deg must each have at least 2 values')

    # Determine reference radius in feet.
    if radius_ft is None:
        if 'r_ref' not in meta or 'length_units' not in meta:
            raise ValueError('radius_ft not provided and hemisphere metadata does not include r_ref/length_units')
        r_ref = float(meta['r_ref'])
        length_units = str(meta['length_units'])
        radius_ft = float(unit_conversion.len_conv(r_ref, from_units=length_units, to_units='ft'))

    # ART grid points -> UMAPR for interpolation
    TH, PH = np.meshgrid(theta_deg, phi_deg)
    azi_rad, elv_rad = art2umapr(np.deg2rad(PH), np.deg2rad(TH))
    # Output amplitude array: (phi, theta, freq)
    frequency_hz, amplitude_db = _sample_hemisphere_levels(
        hemisphere, mode, np.degrees(azi_rad), np.degrees(elv_rad), minimum_level_db)
    nphi = phi_deg.size
    nth = theta_deg.size
    nfreq = int(np.asarray(frequency_hz).size)

    # Replace non-finite levels with the AAM missing mask
    missing_sentinel = np.float32(AAM_MISSING_LEVEL)
    amplitude_to_write = amplitude_db.astype(np.float32)
    amplitude_to_write[~np.isfinite(amplitude_to_write)] = missing_sentinel

    # Write netCDF (match AAM naming conventions).
    # Use NETCDF3_CLASSIC for broad interoperability with legacy AAM/RNM tooling
    # and to avoid HDF5 backend dependency issues on some CI platforms.
    ds = Dataset(filename, mode='w', format='NETCDF3_CLASSIC')
    try:
        # Dimensions (legacy AAM/RNM spheres commonly include these singleton dims)
        ds.createDimension('PHI', nphi)
        ds.createDimension('THETA', nth)
        ds.createDimension('FREQUENCY', nfreq)
        ds.createDimension('XYZ', 3)
        for d in [
            'BB', 'NB', 'PT', 'DOPPLER_SHIFT_REMOVED',
            'EMPTY_WEIGHT', 'FUEL_WEIGHT', 'LOAD_WEIGHT',
            'RADIUS', 'FLIGHT_PATH_ANGLE', 'PYLON_ANGLE', 'SPEED', 'MASTTILT'
        ]:
            if d not in ds.dimensions:
                ds.createDimension(d, 1)

        # Core AAM hemisphere variables
        vphi = ds.createVariable('PHI', 'f4', ('PHI',))
        vth = ds.createVariable('THETA', 'f4', ('THETA',))
        vf = ds.createVariable('FREQUENCY', 'f4', ('FREQUENCY',))
        # No fill_value here: the legacy AAM spheres carry no _FillValue on
        # AMPLITUDE, and setting one both adds an attribute they do not have and
        # makes netCDF4 return a masked array where they return a plain one.
        # The missing sentinel is written into the data instead, as they do.
        vamp = ds.createVariable('AMPLITUDE', 'f4', ('PHI', 'THETA', 'FREQUENCY'))

        # Scalar flight/condition variables (AAM example uses 0-D scalars)
        vr = ds.createVariable('RADIUS', 'f4')
        vs = ds.createVariable('SPEED', 'f4')
        vfpa = ds.createVariable('FLIGHT_PATH_ANGLE', 'f4')

        # Additional AAM metadata variables seen in example spheres
        vBB = ds.createVariable('BB', 'f4')
        vNB = ds.createVariable('NB', 'f4')
        vPT = ds.createVariable('PT', 'f4')
        vDSR = ds.createVariable('DOPPLER_SHIFT_REMOVED', 'f4')
        vEW = ds.createVariable('EMPTY_WEIGHT', 'f4')
        vFW = ds.createVariable('FUEL_WEIGHT', 'f4')
        vLW = ds.createVariable('LOAD_WEIGHT', 'f4')
        vPA = ds.createVariable('PYLON_ANGLE', 'f4')
        vMT = ds.createVariable('MASTTILT', 'f4')
        vXYZ = ds.createVariable('XYZ', 'f4', ('XYZ',))

        vphi[:] = np.asarray(phi_deg, dtype=np.float32)
        vth[:] = np.asarray(theta_deg, dtype=np.float32)
        vf[:] = np.asarray(frequency_hz, dtype=np.float32)
        vamp[:, :, :] = amplitude_to_write

        vr.assignValue(np.float32(radius_ft))
        vs.assignValue(np.float32(speed_knots))
        vfpa.assignValue(np.float32(flight_path_angle_deg))

        vBB.assignValue(np.float32(BB))
        vNB.assignValue(np.float32(NB))
        vPT.assignValue(np.float32(PT))
        vDSR.assignValue(np.float32(doppler_shift_removed))
        vEW.assignValue(np.float32(empty_weight_lb))
        vFW.assignValue(np.float32(fuel_weight_lb))
        vLW.assignValue(np.float32(load_weight_lb))
        vPA.assignValue(np.float32(pylon_angle_deg))
        vMT.assignValue(np.float32(masttilt_deg))

        xyz = np.asarray(xyz_ft, dtype=np.float32).ravel()
        if xyz.size != 3:
            raise ValueError('xyz_ft must be a 3-element iterable (x, y, z) in feet')
        vXYZ[:] = xyz

        # Match the example file's attribute naming: 'unit' (singular).
        # The legacy AAM spheres blank-pad the unit on every scalar variable to
        # AAM_SCALAR_UNIT_WIDTH characters and leave the array variables' units
        # unpadded -- the shape a Fortran CHARACTER(20) write produces.  Keep
        # both conventions so a reader that takes the attribute length at face
        # value sees what it saw in the shipped files.
        vphi.unit = 'DEGREE'
        vth.unit = 'DEGREE'
        vf.unit = 'HERTZ'
        vamp.unit = 'DECIBEL'
        vXYZ.unit = 'FEET'

        def _scalar_unit(text):
            return str(text).ljust(AAM_SCALAR_UNIT_WIDTH)

        vr.unit = _scalar_unit('FEET')
        vs.unit = _scalar_unit('KNOTS')
        vfpa.unit = _scalar_unit('DEGREE')
        vBB.unit = _scalar_unit('')
        vNB.unit = _scalar_unit('')
        vPT.unit = _scalar_unit('')
        vDSR.unit = _scalar_unit('')
        vEW.unit = _scalar_unit('POUNDS')
        vFW.unit = _scalar_unit('POUNDS')
        vLW.unit = _scalar_unit('POUNDS')
        vPA.unit = _scalar_unit('DEGREE')
        vMT.unit = _scalar_unit('DEGREE')

        ds.title = title if title is not None else 'AAM/RNM acoustic hemisphere'
    finally:
        ds.close()


# NORAH2 hemispheres (EASA's rotorcraft noise model) are ASCII "generic acoustic
# data" (.hem) files inherited from HELENA.  The layout is given in EASA NORAH2
# D1.5d Appendix A and D2.3 section 5.5; the data block, which neither shows,
# follows the files shipped with NORAH2 V2.0.74: one block per PHIOBSAC, one
# row per THETAOBSAC, the row starting with its theta and then one level per
# band.

#: Nominal one-third octave band centres of every NORAH2 hemisphere, 10 Hz to 10 kHz.
NORAH2_BAND_CENTERS_HZ = np.array([
    10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0,
    100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0,
    1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0,
    10000.0])

#: Polar angle: 0 at the nose, 90 straight down (at phi = 0), 180 at the tail.
NORAH2_THETA_DEG = np.arange(0.0, 180.0 + 1e-9, 10.0)
#: Azimuth: negative to port, positive to starboard, 0 in the vertical plane.
NORAH2_PHI_DEG = np.arange(-90.0, 90.0 + 1e-9, 10.0)

#: Every NORAH2 hemisphere is defined at 60 m, with absorption over those 60 m
#: included for the ICAO reference atmosphere (FREEFIELD = 2).  NORAH2 takes
#: that absorption back out using TAMB/RELHUM/PAMB before propagating, so the
#: header and the levels must agree.
NORAH2_REFERENCE_DISTANCE_M = 60.0
NORAH2_REFERENCE_TEMPERATURE_K = 298.15
NORAH2_REFERENCE_RELATIVE_HUMIDITY = 70.0
NORAH2_REFERENCE_PRESSURE_PA = 101325.0
NORAH2_MISSING_LEVEL = -999.0

#: Third table of every shipped triangulation file: level corrections NORAH2
#: applies for operations it has no hemisphere of their own for.
NORAH2_DEFAULT_CORRECTIONS = ((8.0, 'Outoffgroundhover'),
                              (-10.0, 'Reducedrpmidle'),
                              (-2.0, 'Fullrpmidle'))


def norah2umapr(phi, theta):
    """Convert NORAH2 (azimuth phi, polar theta) directions to UMAPR (azimuth, elevation).

    NORAH2 defines emission angles in the body axes (x forward, y starboard,
    z down) as x = cos(theta), y = sin(theta) sin(phi), z = sin(theta) cos(phi).
    This is the same convention as ART (see :func:`art2umapr`), so the two
    agree direction for direction.

    Args:
        phi: NORAH2 azimuth angles (radians), positive to starboard.
        theta: NORAH2 polar angles (radians), 0 at the nose.

    Returns: tuple (azimuth, elevation) angles (radians)
    """
    phi = np.asarray(phi, dtype=float)
    theta = np.asarray(theta, dtype=float)
    forward = np.cos(theta)
    starboard = np.sin(theta) * np.sin(phi)
    down = np.sin(theta) * np.cos(phi)
    elv = np.arctan2(down, np.hypot(forward, starboard))
    azi = np.mod(np.arctan2(starboard, -forward), 2.0 * np.pi)
    return azi, elv


def _norah2_band_index(frequency_hz):
    """Index into ``frequency_hz`` of each NORAH2 band, -1 where it has none.

    Matches within a twelfth of an octave, so exact (base-10) and nominal band
    centres both find their band.
    """
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    index = np.full(NORAH2_BAND_CENTERS_HZ.size, -1, dtype=int)
    if frequency_hz.size == 0:
        return index
    for i, nominal in enumerate(NORAH2_BAND_CENTERS_HZ):
        distance = np.abs(np.log2(frequency_hz / nominal))
        j = int(np.argmin(distance))
        if distance[j] < 1.0 / 12.0:
            index[i] = j
    return index


def write_norah2_hemisphere(
        filename,
        hemisphere,
        *,
        speed_knots: float,
        flight_path_angle_deg: float,
        title: Optional[str] = None,
        test_point: str = '',
        measurement_atmosphere: Optional[Atmosphere] = None,
        rotor_rpm_percent: float = 100.0,
        total_wind_knots: float = 0.0,
        cross_wind_knots: float = 0.0,
        pitch_deg: Optional[float] = None,
        roll_deg: Optional[float] = None,
        head_wind_knots: Optional[float] = None,
        minimum_level_db: float = -100.0,
        overwrite: bool = True,
):
    """Write a depropagated acoustic hemisphere as a NORAH2 ``.hem`` file.

    The levels are converted to NORAH2's convention on the way out:

    - moved by spherical spreading from the hemisphere's ``r_ref`` to 60 m;
    - the absorption depropagation leaves in over ``r_ref`` (in the
      measurement atmosphere) taken out, and absorption over 60 m in the ICAO
      reference atmosphere put in, as ``FREEFIELD = 2`` declares;
    - resampled in linear power onto the NORAH2 10 degree grid and its 31
      nominal one-third octave bands (10 Hz to 10 kHz).  A band the
      hemisphere does not have, a direction it did not cover, and a level
      below ``minimum_level_db`` are all written as -999, NORAH2's no-value.

    NORAH2 finds a hemisphere through its aircraft's triangulation file, not
    the file name; see :func:`write_norah2_triangulation`.  The shipped files
    are named ``[type]_[procedure]_[IAS]kts_[gamma]deg.hem``.

    Args:
        filename: Path to the output ``.hem`` file.
        hemisphere: Output dict from :func:`depropagate_hemisphere` with
            ``third_octave=True``.
        speed_knots: ``ACSPEED``, indicated airspeed (knots).
        flight_path_angle_deg: ``GAMM`` (deg), negative in descent.
        title: First line of the file. Default: the file name without extension.
        test_point: Written after the title, as the shipped files carry their
            test point numbers (e.g. 'TP01+02').
        measurement_atmosphere: The atmosphere the hemisphere was depropagated
            in. Default: the one recorded in its metadata. Written as Tm/RHm/Pm.
        rotor_rpm_percent: ``RmOmega``, main rotor speed (% of nominal).
        total_wind_knots, cross_wind_knots: ``TW``, ``CW``.
        pitch_deg, roll_deg, head_wind_knots: ``PITCH``, ``ROLL``, ``HW``. Some
            shipped files carry these and some do not; they are written only
            when given.
        minimum_level_db: Levels (at 60 m) below this are written as -999; see
            :func:`write_aam_hemisphere_netcdf`. Pass -inf to keep every finite level.
        overwrite: If False, raises when filename exists.

    Returns:
        levels_db: (phi, theta, band) levels as written, -inf where -999.
    """
    if not overwrite and os.path.exists(filename):
        raise FileExistsError(f'Output file already exists: {filename}')
    if not isinstance(hemisphere, dict):
        raise TypeError('hemisphere must be a dict returned by depropagate_hemisphere')
    if 'third_octave' not in hemisphere:
        raise ValueError('NORAH2 hemispheres are one-third octave band levels; run '
                         'depropagate_hemisphere with third_octave=True')

    meta = hemisphere.get('metadata', {}) if isinstance(hemisphere.get('metadata', {}), dict) else {}
    if 'r_ref' not in meta or 'length_units' not in meta:
        raise ValueError('hemisphere metadata does not include r_ref/length_units')
    r_ref_m = float(unit_conversion.len_conv(float(meta['r_ref']),
                                             from_units=str(meta['length_units']), to_units='m'))

    if measurement_atmosphere is None:
        recorded = meta.get('atmosphere')
        if recorded is None:
            raise ValueError('measurement_atmosphere not given and the hemisphere metadata '
                             'does not record one')
        measurement_atmosphere = Atmosphere(temperature=recorded['temperature_k'],
                                            pressure=recorded['pressure_kpa'],
                                            relative_humidity=recorded['relative_humidity'])
    reference_atmosphere = Atmosphere(temperature=NORAH2_REFERENCE_TEMPERATURE_K,
                                      pressure=NORAH2_REFERENCE_PRESSURE_PA / 1000.0,
                                      relative_humidity=NORAH2_REFERENCE_RELATIVE_HUMIDITY)

    PH, TH = np.meshgrid(NORAH2_PHI_DEG, NORAH2_THETA_DEG, indexing='ij')
    azi_rad, elv_rad = norah2umapr(np.deg2rad(PH), np.deg2rad(TH))
    # Resample before the unit conversion; minimum_level_db applies at 60 m.
    frequency_hz, sampled_db = _sample_hemisphere_levels(
        hemisphere, 'third_octave', np.degrees(azi_rad), np.degrees(elv_rad), -np.inf)

    band_index = _norah2_band_index(frequency_hz)
    levels_db = np.full(PH.shape + (NORAH2_BAND_CENTERS_HZ.size,), -np.inf, dtype=float)
    present = band_index >= 0
    levels_db[:, :, present] = sampled_db[:, :, band_index[present]]

    offset_db = 20.0 * np.log10(r_ref_m / NORAH2_REFERENCE_DISTANCE_M)
    if meta.get('apply_absorption_deprop', True):
        # Use the bands' own centres for the measured absorption, the nominal
        # ones for NORAH2's: those are what NORAH2 will remove it at.
        measured_centres = np.where(present, np.asarray(frequency_hz, dtype=float)[band_index],
                                    NORAH2_BAND_CENTERS_HZ)
        offset_db = (offset_db
                     + measurement_atmosphere.attenuation_coefficient(measured_centres) * r_ref_m
                     - reference_atmosphere.attenuation_coefficient(NORAH2_BAND_CENTERS_HZ)
                     * NORAH2_REFERENCE_DISTANCE_M)
    else:
        warnings.warn('hemisphere was built without absorption depropagation, so it carries '
                      'the absorption of each measured path; written with spreading only, '
                      'which NORAH2 will read as over-absorbed at high frequency')
    levels_db = levels_db + offset_db
    if np.isfinite(minimum_level_db):
        levels_db[levels_db < float(minimum_level_db)] = -np.inf

    constants = [
        ('POLDIST', f'{NORAH2_REFERENCE_DISTANCE_M:g}', 'Distance at which hemisphere is defined'),
        ('FREEFIELD', '2', 'Atmospheric absorption included in hemisphere'),
        ('NOVALUE', f'{NORAH2_MISSING_LEVEL:g}', 'no value indicator'),
        ('TAMB', f'{NORAH2_REFERENCE_TEMPERATURE_K:g}', 'Ambient temperature, deg Kelvin'),
        ('RELHUM', f'{NORAH2_REFERENCE_RELATIVE_HUMIDITY:g}', 'Relative humidity, %'),
        ('PAMB', f'{NORAH2_REFERENCE_PRESSURE_PA:g}', 'Ambient pressure, Pa'),
        ('Tm', f'{measurement_atmosphere.temperature - 273.15:g}',
         'Measurement ambient temperature at 10m, deg Celsius'),
        ('RHm', f'{measurement_atmosphere.relative_humidity:g}',
         'Measurement relative humidity at 10m, %'),
        ('Pm', f'{measurement_atmosphere.pressure * 1000.0:g}', 'Measurement ambient pressure at 10m, Pa'),
        ('RmOmega', f'{rotor_rpm_percent:g}', 'RotorRPM, rpm'),
        ('ACSPEED', f'{speed_knots:g}', 'Indicated airspeed, kts'),
        ('GAMM', f'{flight_path_angle_deg:g}', 'Path angle, deg'),
    ]
    if pitch_deg is not None:
        constants.append(('PITCH', f'{pitch_deg:g}', 'Pitch, deg'))
    if roll_deg is not None:
        constants.append(('ROLL', f'{roll_deg:g}', 'Roll, deg'))
    constants.append(('TW', f'{total_wind_knots:g}', 'Total wind, kts'))
    constants.append(('CW', f'{cross_wind_knots:g}', 'Cross wind, kts'))
    if head_wind_knots is not None:
        constants.append(('HW', f'{head_wind_knots:g}', 'Head wind, kts'))

    if title is None:
        title = os.path.splitext(os.path.basename(filename))[0]

    def _row(values):
        return '\t' + '\t'.join(values) + '\t'

    lines = [f'{title}\t{test_point}\t', f'{len(constants)}\t! # Table constants ']
    lines += [f'{name}\t{value}\t! {comment}' for name, value, comment in constants]
    lines += ['2\t ',
              f'THETAOBSAC\t{NORAH2_THETA_DEG.size}\t0\t3\t0',
              _row(f'{v:g}' for v in NORAH2_THETA_DEG),
              f'PHIOBSAC\t{NORAH2_PHI_DEG.size}\t0\t3\t0',
              _row(f'{v:g}' for v in NORAH2_PHI_DEG),
              '0\t! NPARAD: Additional point dependent parameters',
              f'NFREQ\t{NORAH2_BAND_CENTERS_HZ.size} ',
              '\t' + '\t'.join(f'{f:6.1f}' for f in NORAH2_BAND_CENTERS_HZ)]
    to_write = np.where(np.isfinite(levels_db), levels_db, NORAH2_MISSING_LEVEL)
    for i, phi in enumerate(NORAH2_PHI_DEG):
        lines.append(f'PHIOBSAC= {phi:.6f}')
        for j, theta in enumerate(NORAH2_THETA_DEG):
            lines.append(f'{theta:g}\t' + '\t'.join(f'{v:.1f}' for v in to_write[i, j, :]))

    # CRLF, as every shipped file has: NORAH2 is a Windows program.
    with open(filename, 'w', encoding='ascii', newline='\r\n') as handle:
        handle.write('\n'.join(lines) + '\n')
    return levels_db


def _norah2_numbers(tokens, count, name):
    values = []
    for token in tokens:
        values.append(float(token))
        if len(values) == count:
            return np.array(values)
    raise ValueError(f'NORAH2 hemisphere: {name} ended after {len(values)} of {count} values')


def load_norah2_hemisphere(filename):
    """Read a NORAH2 ``.hem`` hemisphere.

    Reads the two-axis (THETAOBSAC x PHIOBSAC) files every flight condition in
    the NORAH2 database uses.  Whitespace is free-form, as it is across the
    shipped files, and the table constants are taken by name, since their
    number varies (14 or 17).

    Returns: dict with
        title: first line of the file
        constants: {name: value} of the table constants
        theta_deg, phi_deg, frequency_hz: the axes
        levels_db: (phi, theta, frequency) levels, dB at POLDIST; NOVALUE as -inf
    """
    with open(filename, 'r', encoding='latin-1') as handle:
        lines = [line.rstrip('\r\n') for line in handle]

    def content(line):
        return line.split('!', 1)[0].split()

    title = lines[0].strip()
    n_constants = int(content(lines[1])[0])
    constants = {}
    for line in lines[2:2 + n_constants]:
        name, value = content(line)[:2]
        constants[name] = float(value)
    position = 2 + n_constants

    n_axes = int(content(lines[position])[0])
    position += 1
    if n_axes != 2:
        raise ValueError(f'{filename}: {n_axes}-axis hemispheres (special operations) are not supported')

    def read_axis(position):
        name, count = content(lines[position])[:2]
        count = int(count)
        tokens = []
        position += 1
        while len(tokens) < count:
            tokens += content(lines[position])
            position += 1
        return name, _norah2_numbers(tokens, count, name), position

    inner_name, inner, position = read_axis(position)
    outer_name, outer, position = read_axis(position)
    if int(content(lines[position])[0]) != 0:
        raise ValueError(f'{filename}: point dependent parameters (NPARAD > 0) are not supported')
    _, frequency_hz, position = read_axis(position + 1)

    axes = {inner_name: inner, outer_name: outer}
    if set(axes) != {'THETAOBSAC', 'PHIOBSAC'}:
        raise ValueError(f'{filename}: unexpected axes {inner_name}, {outer_name}')

    body = lines[position:]
    block_starts = [i for i, line in enumerate(body) if line.strip().startswith(outer_name)]
    if len(block_starts) != outer.size:
        raise ValueError(f'{filename}: {len(block_starts)} {outer_name} blocks, expected {outer.size}')
    levels = np.empty((outer.size, inner.size, frequency_hz.size), dtype=float)
    for k, start in enumerate(block_starts):
        stop = block_starts[k + 1] if k + 1 < len(block_starts) else len(body)
        tokens = [t for line in body[start + 1:stop] for t in content(line)]
        table = _norah2_numbers(tokens, inner.size * (1 + frequency_hz.size), outer_name)
        table = table.reshape(inner.size, 1 + frequency_hz.size)
        levels[k] = table[:, 1:]

    missing = constants.get('NOVALUE', NORAH2_MISSING_LEVEL)
    levels[np.isclose(levels, missing)] = -np.inf
    if outer_name == 'THETAOBSAC':
        levels = levels.transpose(1, 0, 2)
    return dict(title=title, constants=constants,
                theta_deg=axes['THETAOBSAC'], phi_deg=axes['PHIOBSAC'],
                frequency_hz=frequency_hz, levels_db=levels)


def write_norah2_triangulation(filename, hemispheres, *, corrections=NORAH2_DEFAULT_CORRECTIONS,
                               overwrite=True):
    """Write the ``[type]_Triangulation.int`` that lets NORAH2 use a set of hemispheres.

    NORAH2 interpolates between an aircraft's hemispheres over (airspeed, flight
    path angle) by triangles listed in this file.  They are the Delaunay
    triangulation of the raw (knots, degrees) conditions: that reproduces every
    triangulation file NORAH2 V2.0.74 ships, where the min-max normalised
    conditions the method report (D1.5d A.3) describes do not.

    Args:
        filename: Path to the output ``.int`` file; NORAH2 looks for it next to
            the hemispheres.
        hemispheres: iterable of (hem file name, speed_knots, flight_path_angle_deg).
            File names are written as given, relative to the hemisphere folder.
        corrections: (dB, operation) rows of the third table. The default is
            the table every shipped file carries.
        overwrite: If False, raises when filename exists.

    Returns:
        (Ntri, 3) array of zero-based hemisphere indices, one row per triangle.
    """
    from scipy.spatial import Delaunay

    if not overwrite and os.path.exists(filename):
        raise FileExistsError(f'Output file already exists: {filename}')
    hemispheres = [(str(name), float(speed), float(angle)) for name, speed, angle in hemispheres]
    if len(hemispheres) < 3:
        raise ValueError('a NORAH2 triangulation needs at least 3 hemispheres')
    points = np.array([(speed, angle) for _, speed, angle in hemispheres])
    if np.unique(np.round(points, 6), axis=0).shape[0] != points.shape[0]:
        raise ValueError('two hemispheres share a flight condition; NORAH2 merges repeat runs '
                         'into one hemisphere per condition, so average them first')
    triangles = Delaunay(points).simplices

    lines = ['&HEMISPHERES', f'\tNGAD = {len(hemispheres)}', '&END', '',
             'iHem\tHEMSpeed\tHEMAngle\t[Path\\]FileHem']
    lines += [f'{i}\t{speed:g}\t{angle:g}\t{name}'
              for i, (name, speed, angle) in enumerate(hemispheres, start=1)]
    lines += ['', '&TRIANGLES', f'\tNTRI = {len(triangles)}', '&END', '',
              'iTri\tiHem1\tiHem2\tiHem3']
    lines += [f'{i}\t' + '\t'.join(str(int(v) + 1) for v in triangle)
              for i, triangle in enumerate(triangles, start=1)]
    lines += ['', '&CORRECTIONS', f'NCOR\t= {len(corrections)}', '&END', '',
              'Corr_dB\tOperation']
    lines += [f'{dB:g}\t"{operation}"' for dB, operation in corrections]
    with open(filename, 'w', encoding='ascii', newline='\r\n') as handle:
        handle.write('\n'.join(lines) + '\n')
    return triangles


def OASPL(amplitudes):
    """
    Integrate an array of SPL amplitudes to compute the OASPL
    Args:
        amplitudes: array of SPL amplitudes
    Returns:
        OASPL, dB
    """
    return 10.0 * safe_log10(np.sum(np.power(10.0, amplitudes / 10.0)))


def power_to_db(power):
    """10 * log10(power), for power relative to its reference (p**2 / pref**2).

    Zero power comes back as -inf (no energy) and NaN stays NaN (no data), with
    none of numpy's divide-by-zero warnings.  Neither becomes a finite floor:
    the -3076 dB that ``np.maximum(power, np.finfo(float).tiny)`` gives reads as
    a level, pulling plot colour scales and statistics down with it.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return 10.0 * np.log10(np.asarray(power, dtype=float))


def safe_log10(x):
    """
    Safe version of log10, where values below zero are returned as NaN and values equal to zero are -inf
    Args:
        x: array of values

    Returns: checked log10 value
    """
    if x == 0:
        return -np.inf
    if x < 0:
        return np.nan
    else:
        return np.log10(x)


def dBAw(f):
    """
    A-weighting curve
    Args:
        f: frequencies to evaluate

    Returns: weighting (in dB) at each frequency
    """
    f = np.asarray(f, dtype=float)
    aweights = np.full_like(f, -300.0, dtype=float)
    pos = f > 0
    if np.any(pos):
        fp = f[pos]
        aweights[pos] = (
            10.0 * np.log10(
                1.562339
                * fp ** 4.0
                / ((fp ** 2.0 + 107.65265 ** 2.0) * (fp ** 2.0 + 737.86223 ** 2.0))
            )
            + 10.0
            * np.log10(
                2.242881e16
                * fp ** 4.0
                / ((fp ** 2.0 + 20.598997 ** 2.0) ** 2.0 * (fp ** 2.0 + 12194.22 ** 2.0) ** 2.0)
            )
        )
    return aweights


def load_nc_signal(filename):
    """
    Load NASA-formatted netCDF acoustic signal
    Args:
        filename: path to netCDF file

    Returns: tuple (pressure, time, location)
    WHERE
    pressure is an array of acoustic pressures
    time is an array of sampled times
    location is an array of the x,y,z location of the microphone
    """
    file_handle = Dataset(filename, mode='r')
    pressure = file_handle.variables['pressure'][:].astype(float).flatten()
    x = file_handle.X
    y = file_handle.Y
    z = file_handle.Z
    sample_rate = file_handle.sample_rate
    start_time = file_handle.start_time
    file_handle.close()
    # Not np.arange(start, stop, 1/fs): a float step can yield one sample too many.
    time = start_time + np.arange(pressure.size) / sample_rate
    location = np.array([x, y, z])
    return pressure, time, location


def load_h5_signal(filename, datasetname='Table1', signalname=None):
    """
    Load HDF5 files from BKConnect
    Args:
        filename: path to file
        datasetname: optional dataset name (default 'Table1')
        signalname: optional signal name (default None, loads all signals)

    Returns: h5py dataset for entire group or specific signal
    """
    file = h5py.File(filename, 'r')
    if signalname is None:
        grp = file[datasetname]
        # Return the group, type checker might complain but runtime is fine
        return cast(h5py.Group, grp)  # type: ignore[return-value]
    else:
        grp = file[datasetname]
        if isinstance(grp, h5py.Group):
            ds = grp[signalname]
            return cast(h5py.Dataset, ds)  # type: ignore[return-value]
        else:
            raise ValueError(f"Dataset {datasetname} is not a Group")


def load_UFF_signal(filename, sets = None):
    """
    Load UFF acoustic signal file
    Args:
        filename: path to UFF file
        sets: optional list of set numbers to load, default None (loads all sets)
    Returns: tuple (pressures, fs, channel_names, time)
    WHERE
    pressures is a channels x timepoints matrix of acoustic pressures
    fs is the sampling rate, Hz
    channel_names is a list of channel names
    time is an array of sampled times
    """

    file = UFF(filename)

    if sets is None:
        data = file.read_sets()
    else:
        data = file.read_sets(sets)
    # pyuff returns a bare dict, not a one-element list, when it reads one set
    if isinstance(data, dict):
        data = [data]

    channels = len(data)
    datasize = len(data[0]['x'])
    time = data[0]['x']
    fs = 1./(time[1]-time[0])
    pressures = np.zeros((channels, datasize))
    channel_names = []
    for i in range(channels):
        if len(data[i]['x']) != datasize:
            raise ValueError('Inconsistent recording lengths in UFF file.')
        pressures[i, :] = data[i]['data']
        channel_names.append(data[i]['id1'])
    return pressures, fs, channel_names, time


def highpass(x, fpass, fs, zero_phase=True):
    """
    Applies a high pass filter to a signal
    Args:
        x: signal array
        fpass: high pass frequency
        fs: sampling rate of x
        zero_phase: optional, use phase preserving filter, default True

    Returns: filtered signal
    """
    return np.array(pa_filters.highpass(x, fpass, fs, zero_phase=zero_phase))


def lowpass(x, fpass, fs, zero_phase=True):
    """
    Applies a low pass filter to a signal
    Args:
        x: signal array
        fpass: low pass frequency
        fs: sampling rate of x
        zero_phase: optional, use phase preserving filter, default True

    Returns: filtered signal
    """
    return np.array(pa_filters.lowpass(x, fpass, fs, zero_phase=zero_phase))


def art2umapr(phi, theta):
    """
    Convert between ART (AAM/RNM/ANOPP) coordinates and UMAPR coordinates

    ART follows the AAM Technical Reference (v3, sec. 2.4.1 and 2.5): theta is
    0 at the nose and 180 at the tail, phi is 0 directly below, positive to
    starboard and negative to port.  UMAPR azimuth is 180 ahead, 90 starboard.
    Before 2026-09-24 this used x = -sin(theta) sin(phi), putting phi = +90 to
    port; ``flip_y_for_geometry=True`` compensated for it in the AAM export,
    at the cost of mirroring every UMAPR hemisphere built that way.

    Args:
        phi: array of phi angles (radians)
        theta: array of theta angles (radians)

    Returns: tuple (azimuth, elevation) angles (radians)

    """
    x = np.sin(theta) * np.sin(phi)
    y = np.cos(theta)
    z = -np.sin(theta) * np.cos(phi)
    elv = -np.arctan2(z, np.sqrt(np.square(x) + np.square(y)))
    azi = np.arctan2(x, -y)
    azi[azi < 0.0] = azi[azi < 0.0] + 2.0 * np.pi
    return azi, elv


def lambert_ea(lat, lon):
    """
    Lambert equal-area projection of spherical angles
    Args:
        lat: array of lateral angles, radians
        lon: array of longitudinal angles, radians

    Returns: tuple, (x, y)

    """
    q = 2.0 * np.sin((np.pi / 2.0 - lat) / 2.0)
    x = q * np.sin(lon)
    y = q * np.cos(lon)
    return x, y


def lambert_lon(azimuth):
    """Page longitude for a UMAPR azimuth (radians in, radians out).

    The hemisphere is drawn as a VIEW FROM ABOVE, so it reads like the
    ground footprints and any other plan view:

        azimuth 180 (ahead)     -> top of the page
        azimuth  90 (starboard) -> RIGHT
        azimuth 270 (port)      -> left
        azimuth   0 (behind)    -> bottom

    i.e. azimuth increases counter-clockwise on the page, as
    :func:`lambert_ea_points` documents. Route every azimuth through here
    rather than open-coding the shift: before 2026-08-19 the data used
    ``lon = azimuth - pi`` (which mirrors the lateral axis, putting
    starboard on the LEFT) while the meridian labels printed
    ``360 - meridian``, so the rim numbers disagreed with the field.
    """
    return np.pi - np.asarray(azimuth, dtype=float)


# On-disk format version written into every database as "database_version".
# 0 / absent  -- built before this was introduced.  Hover spheres in those files
#                may have broken directivity (see mirror_phi_to_upper_surface).
# 1           -- directivity mirror correct; hover spheres fore-aft averaged on an
#                energy basis with EAA recomputed from the averaged spectrum.
DATABASE_FORMAT_VERSION = 1

# A condition is "near level flight" within this many degrees of zero flight
# path angle.  Used both to pick the source sphere for hover and to choose which
# conditions get re-emitted at extended flight path angles.
LEVEL_FLIGHT_TOLERANCE = 2.0


def average_fore_and_aft(spectrum):
    """Average a sphere fore-to-aft on an energy (p^2) basis.

    ``spectrum`` is dB per band with shape (phi, theta, frequency); theta runs
    nose (0) to tail (180), so band j = n-1-i is the fore/aft partner of band i.
    Each pair is replaced by their mean-square average::

        10 * log10( 0.5 * (10**(Li/10) + 10**(Lj/10)) )

    A plain arithmetic mean of decibels is not an energy average and biases the
    result low -- about 0.4 dB for a pair 5.8 dB apart, which is the spread seen
    between fore/aft partners in real spheres.  The middle band of an odd-length
    theta axis is its own partner and is left unchanged.
    """
    spectrum = np.array(spectrum, dtype=float, copy=True)
    n_theta = spectrum.shape[1]
    # -inf entries mean "no energy" and are expected; 10**(-inf/10) == 0.
    old = np.seterr(divide='ignore', invalid='ignore')
    try:
        for i in range(n_theta // 2):
            j = n_theta - 1 - i
            mean_square = 0.5 * (10.0 ** (spectrum[:, i, :] / 10.0)
                                 + 10.0 ** (spectrum[:, j, :] / 10.0))
            averaged = 10.0 * np.log10(mean_square)
            spectrum[:, i, :] = averaged
            spectrum[:, j, :] = averaged
    finally:
        np.seterr(**old)
    return spectrum


def spla_and_eaa_from_spectrum(spectrum, frequency, distance, atmosphere):
    """Derive A-weighted OASPL and excess atmospheric attenuation from a spectrum.

    Mirrors the reduction in :func:`extract_SPL`, so a caller that modifies the
    spectrum (e.g. fore/aft averaging for hover) can recompute both quantities
    consistently instead of carrying stale values forward.
    """
    a_weight = np.array([dBAw(f) for f in frequency])
    alpha = atmosphere.attenuation_coefficient(frequency)
    spla = np.apply_along_axis(OASPL, 2, spectrum + a_weight)
    spla_attenuated = np.apply_along_axis(OASPL, 2, spectrum + a_weight - distance * alpha)
    old = np.seterr(invalid='ignore')
    try:
        eaa = spla - spla_attenuated
    finally:
        np.seterr(**old)
    return spla, eaa


def _complete_sphere(phi_list, theta_list, spla, eaa, amplitude):
    """Extend a lower-hemisphere sphere onto its upper surface.

    Returns ``(phi, theta, spla, eaa, amplitude)`` with the reflected rows
    appended.  ``phi`` and ``theta`` come back as meshgrid arrays matching the
    data, so every array stays row-aligned with its azimuth -- the failure mode
    of the three-part slicing this replaces.
    """
    mirror_phi, source_index = mirror_phi_to_upper_surface(phi_list)
    phi_full_list = np.concatenate((np.asarray(phi_list, dtype=float), mirror_phi))
    # Rows sorted by phi, so the spectra -- written gridded, without angles of
    # their own -- are in the sorted order NICE-OPS reads them in.  Until
    # 2026-09-25 they were left in completion order (-90..90, then the mirrored
    # upper surface), which NICE-OPS read as sorted and so put every spectrum
    # in the wrong direction.  dBA/EAA carry their angles per channel and were
    # never affected.
    order = np.argsort(phi_full_list, kind='stable')
    rows = np.concatenate((np.arange(len(phi_list)), source_index))[order]
    theta, phi = np.meshgrid(theta_list, phi_full_list[order])
    return phi, theta, spla[rows], eaa[rows], amplitude[rows]


def mirror_phi_to_upper_surface(phi_list):
    """Complete a lower-hemisphere sphere onto its upper surface.

    Sphere data covers only the lower half of the roll circle, phi in
    [-90, 90] degrees with phi = 0 straight down.  The upper half is the
    reflection through the horizontal plane, phi -> 180 - phi wrapped into
    [-180, 180):  0 (down) -> 180 (up), and +/-90 map to themselves because the
    horizontal plane belongs to both halves.

    Returns (mirror_phi, source_index): the azimuths of the reflected rows
    and the indices of the source rows they came from, so that every data array
    can be extended consistently::

        mirror_phi, idx = mirror_phi_to_upper_surface(phi_list)
        phi_full  = np.concatenate((phi_list, mirror_phi))
        SPLA_full = np.concatenate((SPLA, SPLA[idx]))

    Rows whose reflection already exists in phi_list are dropped, so the
    shared +/-90 edges appear once rather than twice and the completed sphere
    has no duplicate (phi, theta) points.

    NOTE on two earlier forms, both of which produced bad databases:

    * np.concatenate((phi, -phi)) is a left/right flip *within* the lower
      half.  Because the source range is symmetric about zero it maps [-90, 90]
      onto itself, emitting a second copy of the lower hemisphere under the same
      labels and leaving the upper surface empty.  The hover spheres in
      S-76D_M3.nod were built this way: every (phi, theta) there carries two
      conflicting levels, 630 of 703 pairs disagreeing by up to 5.77 dB.
    * A three-part concatenation that sliced phi as [h+1:n], [:], [1:h]
      but the data arrays as [1:h], [:], [h+1:n].  Those slices have equal
      length only when len(phi_list) is even; the real sphere grids have 19
      azimuths, so the labels ran one row out of step with the acoustic data
      across most of the sphere -- corrupting forward-flight spheres too.
    """
    phi_list = np.asarray(phi_list, dtype=float)
    reflected = (180.0 - phi_list + 180.0) % 360.0 - 180.0
    # Drop reflections that coincide with a source azimuth (the +/-90 edges),
    # comparing on a rounded grid so exact float equality is not required.
    already_present = np.isin(np.round(reflected, 9), np.round(phi_list, 9))
    source_index = np.flatnonzero(~already_present)
    return reflected[source_index], source_index


def build_empirical_database(directory_name, database_filename, load_factors=np.linspace(0.7, 2.3, 5), infreqs=None,
                             distance=1000,
                             atmosphere=Atmosphere(temperature=293.15, pressure=101.325,
                                                   relative_humidity=20.0),
                             extended_flight_path_angles=None,
                             level_flight_tolerance=LEVEL_FLIGHT_TOLERANCE,
                             store_spectrum=True, clamp_empty_directions=True,
                             hover_correction=None):
    """Build a NICE-OPS sphere database from a directory of sphere files.

    load_factors scales thrust: each source condition is written once per load
    factor, with the sphere level offset by 20*log10(load_factor) and the
    thrust coefficient scaled to match.  Maneuver modelling needs this to span
    well beyond 1 g -- the shipped databases use
    [0, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0], finely spaced near 1 g and reaching
    a 5 g pull-up, rather than the uniform default here.

    Pass ``load_factors=None`` to write each condition once, at the LF=1
    reference only, and let NICE-OPS scale levels (and, for a spectral
    database, every band) to the queried load factor analytically instead of
    from stored samples.  This is exact, not an approximation: every load
    factor this function would otherwise materialise is the *same* spectrum
    offset by a uniform ``20*log10(load_factor)`` -- add_sphere_group does not
    scale ``amplitude``, only ``dBA`` and ``thrust_coefficient`` -- so writing
    eight copies of it (the shipped databases' load factor count) stores no
    information the reader could not derive from one. It shrinks a database
    roughly in proportion to the load factor count (measured on Be407: 547.9
    MB -> well under 100 MB). The database is written with a root
    ``fixed_load_factor`` flag so NICE-OPS knows the scaling is safe to apply
    -- it must NOT be inferred for a database built some other way, e.g. from
    genuinely separate measurements at different thrust conditions whose
    spectral *shape*, not just level, could differ with load factor.

    extended_flight_path_angles, when given, widens the flight-path-angle
    envelope past what was measured.  Measured spheres cluster near level
    flight, so a trajectory that climbs or descends steeply would otherwise
    interpolate against a clamped edge.  Every near-level condition (within
    level_flight_tolerance of zero, hover included) is re-emitted at each of
    these angles at its own airspeed, on the assumption that directivity at a
    given airspeed carries over to a steeper flight path.  The shipped
    databases use (-24.0, 35.0).

    hover_correction, when given, is added to the synthesised hover sphere's band levels
    after the fore-to-aft average: a callable taking phi and theta (degrees, arrays shaped
    like the sphere's (phi, theta) grid) and the band centres (Hz), returning dB with shape
    (phi, theta, frequency).  It is how measured hovers, which cover only part of the
    sphere, are blended into the synthesised one; the caller owns where it applies and how
    it tapers.  Its ``description`` attribute, if any, is written to the database.

    clamp_empty_directions replaces the NaN that EAA becomes where ambient
    gating emptied a direction (``-inf - -inf``) with zero.  A NaN there is not
    local: it put 886 NaN cells into a NICE-OPS footprint.  The dBA level is
    left as it falls out, -inf included; see :func:`_finite_sphere_levels`.

    store_spectrum keeps the source spectrum (frequency + amplitude) in each
    sphere group, so a database can be re-reduced -- different weighting, a
    different propagation distance -- without the original sphere files.  It
    accounts for roughly seven eighths of the file size and no consumer reads it
    today, so turn it off for databases that only need levels.
    """
    # TODO pack in redimensionalization data
    # TODO add reinterpolation flag

    # None means "write the LF=1 reference only, let the reader scale it" --
    # see the docstring.  Every load-factor loop below iterates this instead
    # of the load_factors parameter directly, so add_sphere_group need not
    # change: at load_factor=1.0 it already writes the unscaled level and
    # thrust_coefficient == weight_coefficient, which is exactly the reference.
    fixed_load_factor = load_factors is None
    load_factors_to_write = (1.0,) if fixed_load_factor else load_factors

    # Get vehicle attributes
    (main_rotor_radius, main_rotor_area, main_rotor_tip_speed,
     _, _, _, _, _, _, weight_coefficient, _, _, _, _) = read_vehicle_data(directory_name)

    # Set up database
    ncdatabase = Dataset(os.path.abspath(os.path.expanduser(database_filename)), 'w')

    # NICE-OPS requires this flag; it reads it unconditionally at load time.
    ncdatabase.createVariable("same_grid", 'b')
    ncdatabase['same_grid'][:] = True
    # Optional: absent (old files) or false means every stored condition is a
    # real, independent sample and NICE-OPS must not scale between them.  True
    # means every condition was synthesised from one LF=1 reference by a
    # uniform dB offset, so the reader may reproduce load factors this file
    # never stored by applying that same offset analytically.
    ncdatabase.createVariable("fixed_load_factor", 'b')
    ncdatabase['fixed_load_factor'][:] = fixed_load_factor
    # Format version, so consumers can tell a database built by this code from
    # the older ones whose hover spheres have broken directivity.  Bump this
    # whenever the on-disk meaning of the sphere data changes.
    ncdatabase.createVariable("database_version", 'i4')
    ncdatabase['database_version'][:] = DATABASE_FORMAT_VERSION

    # Root vehicle data, as carried by every shipped database (S-76D_M3.nod,
    # AW139_M1.nod, Be407_spectral.nod).  NICE-OPS reads them to redimensionalise
    # the sphere conditions and, for --export_aam, in preference to anything it
    # would otherwise infer, so leaving them out quietly changes what it does.
    # main_rotor_radius_meters restates each group's rotor_scale; tip speed and
    # weight appear nowhere else in the file.
    if main_rotor_radius is not None:
        ncdatabase.createVariable("main_rotor_radius_meters", 'f8')
        ncdatabase['main_rotor_radius_meters'][:] = float(main_rotor_radius)
    if main_rotor_tip_speed is not None:
        ncdatabase.createVariable("main_rotor_tip_speed_meters_per_sec", 'f8')
        ncdatabase['main_rotor_tip_speed_meters_per_sec'][:] = float(main_rotor_tip_speed)
    if hover_correction is not None:
        ncdatabase.hover_correction = str(getattr(hover_correction, 'description', 'applied'))
    vehicle_weight_newtons = read_vehicle_weight_newtons(directory_name)
    if vehicle_weight_newtons is not None:
        ncdatabase.createVariable("vehicle_weight_newtons", 'f8')
        ncdatabase['vehicle_weight_newtons'][:] = float(vehicle_weight_newtons)

    # Get list of full paths to netCDF files in directory
    local_glob = os.path.expanduser(directory_name) + '/*.nc'
    absolute_glob = os.path.abspath(local_glob)
    # Sorted, so group numbering and hover-sphere ties do not depend on the
    # order the filesystem happens to list the directory in.
    file_list = sorted(glob(absolute_glob))

    min_speed = np.inf
    min_speed_file = None
    slowest_speed = np.inf
    slowest_file = None
    sphere_index = 0
    level_conditions = []
    # Every condition to write, collected rather than written immediately, so
    # that whether phi/theta/frequency can be deduplicated (see
    # _grid_and_frequency_are_shared) is known before the first group is
    # written -- not discovered partway through and left to redo.
    pending_groups = []
    for filename in file_list:
        # Load the sphere data
        (_, _, phi_list, theta_list, radius, _, SPLA, EAA, speed, flight_path_angle,
         frequency, amplitude) = extract_SPL(filename, infreqs, distance, atmosphere)
        if clamp_empty_directions:
            SPLA, EAA = _finite_sphere_levels(SPLA, EAA)
        # Find the lowest speed file near level flight
        if speed < min_speed and np.abs(flight_path_angle) < level_flight_tolerance:
            min_speed = speed
            min_speed_file = filename
        # ...and the slowest overall, as a fallback when nothing is near level
        if speed < slowest_speed:
            slowest_speed = speed
            slowest_file = filename
        # Complete the sphere onto its upper surface
        phi_full, theta_full, SPLA_full, EAA_full, amplitude_full = _complete_sphere(
            phi_list, theta_list, SPLA, EAA, amplitude)
        # Remember near-level conditions; they are the ones re-emitted at the
        # extended flight path angles below.
        if extended_flight_path_angles is not None and np.abs(flight_path_angle) < level_flight_tolerance:
            level_conditions.append((phi_full, theta_full, radius, SPLA_full, EAA_full,
                                     speed, frequency, amplitude_full))
        # Augment load factor data
        for load_factor in load_factors_to_write:
            groupname = "sphere" + str(sphere_index)
            sphere_index = sphere_index + 1
            pending_groups.append((groupname, phi_full, theta_full, radius, SPLA_full, EAA_full,
                                   speed, flight_path_angle, load_factor,
                                   frequency if store_spectrum else None,
                                   amplitude_full if store_spectrum else None))

    if min_speed_file is None:
        # No sphere within level_flight_tolerance of level flight.  The hover
        # sphere is synthesised by averaging the slowest one fore-to-aft, so
        # fall back to the slowest available rather than failing with an
        # unbound name -- the further it is from level flight, the rougher the
        # hover approximation.
        if slowest_file is None:
            raise ValueError('No sphere files found in ' + str(directory_name))
        warnings.warn('No sphere within {:g} degrees of level flight; synthesising hover from the '
                      'slowest available at {:.1f} knots.'.format(level_flight_tolerance, slowest_speed))
        min_speed_file = slowest_file

    # Now, adapt the lowest speed sphere to a hover sphere by averaging from fore to aft
    (_, _, phi_list, theta_list, radius, _, _, _, _, _,
     frequency, amplitude) = extract_SPL(min_speed_file, infreqs, distance, atmosphere)
    # Average the SPECTRUM fore-to-aft on an energy basis, then derive SPLA and
    # EAA from the averaged spectrum.  Averaging the broadband dB levels instead
    # (a) is not an energy average, and (b) leaves EAA untouched, so level and
    # excess attenuation end up describing different spheres.
    amplitude = average_fore_and_aft(amplitude)
    if hover_correction is not None:
        theta_grid, phi_grid = np.meshgrid(np.asarray(theta_list, dtype=float),
                                           np.asarray(phi_list, dtype=float))
        correction = np.asarray(hover_correction(phi_grid, theta_grid, np.asarray(frequency, dtype=float)),
                                dtype=float)
        if correction.shape != amplitude.shape:
            raise ValueError('hover_correction returned shape {}, the sphere is {}'.format(
                correction.shape, amplitude.shape))
        amplitude = amplitude + correction
    SPLA, EAA = spla_and_eaa_from_spectrum(amplitude, frequency, distance, atmosphere)
    if clamp_empty_directions:
        SPLA, EAA = _finite_sphere_levels(SPLA, EAA)
    # Complete the sphere onto its upper surface
    phi_full, theta_full, SPLA_full, EAA_full, amplitude_full = _complete_sphere(
        phi_list, theta_list, SPLA, EAA, amplitude)
    # Set hover conditions.  Hover is a near-level condition, so it is written
    # at the extended angles as well as level flight.
    speed = 0
    if extended_flight_path_angles is not None:
        flight_path_angles = sorted({0.0, *extended_flight_path_angles})
    else:
        flight_path_angles = [-12, 0, 12]
    # Augment load factor data
    for load_factor in load_factors_to_write:
        for flight_path_angle in flight_path_angles:
            groupname = "sphere" + str(sphere_index)
            sphere_index = sphere_index + 1
            pending_groups.append((groupname, phi_full, theta_full, radius, SPLA_full, EAA_full,
                                   speed, flight_path_angle, load_factor,
                                   frequency if store_spectrum else None,
                                   amplitude_full if store_spectrum else None))

    # Widen the flight-path-angle envelope: re-emit each near-level condition at
    # the extended angles, keeping its own airspeed and directivity.  Hover is
    # already covered above, so skip any condition at zero airspeed.
    if extended_flight_path_angles is not None:
        for (phi_full, theta_full, radius, SPLA_full, EAA_full,
             level_speed, frequency, amplitude_full) in level_conditions:
            if level_speed == 0:
                continue
            for extended_angle in extended_flight_path_angles:
                for load_factor in load_factors_to_write:
                    groupname = "sphere" + str(sphere_index)
                    sphere_index = sphere_index + 1
                    pending_groups.append((groupname, phi_full, theta_full, radius, SPLA_full,
                                           EAA_full, level_speed, extended_angle, load_factor,
                                           frequency if store_spectrum else None,
                                           amplitude_full if store_spectrum else None))

    # phi/theta (and frequency, for a spectral database) are usually the same
    # on every condition -- panam completes every sphere onto a common grid
    # and reduces every condition to the same frequency bands -- but that is
    # checked here, not assumed: a mixed microphone layout or a per-file
    # frequency resolution would make it false, and writing the shared form
    # anyway would silently lose whatever a later condition's grid disagreed
    # on. See the root shared_grid_and_frequency flag this sets.
    shared_grid_and_frequency = _grid_and_frequency_are_shared(pending_groups)
    ncdatabase.createVariable("shared_grid_and_frequency", 'b')
    ncdatabase['shared_grid_and_frequency'][:] = shared_grid_and_frequency
    if shared_grid_and_frequency:
        (_, phi0, theta0, _, _, _, _, _, _, frequency0, _) = pending_groups[0]
        ncdatabase.createDimension("channels", phi0.size)
        ncdatabase.createVariable("phi", 'f8', ("channels",))
        ncdatabase['phi'][:] = phi0.flatten()
        ncdatabase.createVariable("theta", 'f8', ("channels",))
        ncdatabase['theta'][:] = theta0.flatten()
        if frequency0 is not None:
            ncdatabase.createDimension("frequency", np.size(frequency0))
            ncdatabase.createVariable("frequency", 'f8', ("frequency",))
            ncdatabase['frequency'][:] = np.asarray(frequency0).flatten()

    for (groupname, phi_full, theta_full, radius, SPLA_full, EAA_full, speed_or_level_speed,
         flight_path_angle, load_factor, frequency, amplitude_full) in pending_groups:
        add_sphere_group(ncdatabase, groupname, phi_full, theta_full, radius, SPLA_full, EAA_full,
                         speed_or_level_speed, flight_path_angle, load_factor, main_rotor_radius,
                         main_rotor_tip_speed, weight_coefficient, frequency, amplitude_full,
                         write_grid_and_frequency=not shared_grid_and_frequency)
    # Close explicitly: left to the garbage collector, a failed flush on close
    # is swallowed and the file can stay open (locked) while a traceback lives.
    ncdatabase.close()


def _grid_and_frequency_are_shared(pending_groups):
    """Whether every condition in pending_groups has the same phi, theta and
    frequency arrays, so build_empirical_database can write them once at the
    root instead of once per condition.  See its shared_grid_and_frequency
    flag.
    """
    if not pending_groups:
        return False
    _, phi0, theta0, *_, frequency0, _ = pending_groups[0]
    for _, phi, theta, *_, frequency, _ in pending_groups[1:]:
        if phi.shape != phi0.shape or not np.array_equal(phi, phi0):
            return False
        if theta.shape != theta0.shape or not np.array_equal(theta, theta0):
            return False
        if (frequency is None) != (frequency0 is None):
            return False
        if frequency is not None and not np.array_equal(frequency, frequency0):
            return False
    return True


def _finite_sphere_levels(spla, eaa):
    """Sanitise EAA where ambient gating emptied a direction.

    Ambient gating legitimately empties a direction -- a low-power descent can
    sit at the noise floor over most of the aft hemisphere -- and then
    ``SPLA = -inf``, so ``EAA = SPLA - SPLAa`` is ``-inf - -inf``, a NaN.

    The level itself is left alone, including when it is -inf. That was checked
    against NICE-OPS rather than assumed (its build/niceops, B205 track, Be407
    database, patching the directions a flyover actually uses so the test is
    demonstrably sensitive -- 26.9 dB of footprint change):

    * ``dBA = -inf`` and ``dBA = -3064`` give byte-identical footprints with no
      NaN cells. Its ``eval`` subtracts spreading and attenuation from the
      level and the exposure integral then sums energy, so a -inf direction
      contributes nothing -- which is exactly what an empty direction means.
    * ``EAA = NaN`` does not stay local: it put 886 NaN cells into a 120x120
      footprint and moved the mean level by 1.3 dB.

    So only EAA needs a value here, and zero is the right one: with no energy
    in the direction there is no excess attenuation to apply to it.
    """
    spla = np.asarray(spla, dtype=float)
    eaa = np.asarray(eaa, dtype=float).copy()
    eaa[~np.isfinite(eaa)] = 0.0
    return spla, eaa


def add_sphere_group(ncdatabase, groupname, phi, theta, radius, SPLA, EAA, speed, flight_path_angle, load_factor,
                     main_rotor_radius, main_rotor_tip_speed, weight_coefficient, frequency=None, amplitude=None,
                     write_grid_and_frequency=True):
    # Create a new group for this sphere
    this_group = ncdatabase.createGroup(groupname)
    # Define sphere nondimensional radius
    this_group.createDimension("radii", 1)
    this_group.createVariable("radius", 'f8', ("radii",))
    this_group.variables['radius'][:] = radius / main_rotor_radius
    # Define rotor scale
    this_group.createDimension("condition", 1)
    this_group.createVariable("rotor_scale", 'f8', ("condition",))
    this_group.variables['rotor_scale'][:] = main_rotor_radius
    # Define flight condition
    this_group.createVariable("advance_ratio", 'f8', ("condition",))
    knots_to_meters = 0.514444
    this_group.variables['advance_ratio'][:] = knots_to_meters * speed / main_rotor_tip_speed
    this_group.createVariable("flight_path_angle", 'f8', ("condition",))
    this_group.variables['flight_path_angle'][:] = flight_path_angle
    this_group.createVariable("thrust_coefficient", 'f8', ("condition",))
    this_group.variables['thrust_coefficient'][:] = load_factor * weight_coefficient
    # Define acoustic data.  "channels" is defined locally in every group
    # regardless of write_grid_and_frequency -- netCDF4/HDF5 charges real
    # space to *share* a dimension across many groups (each group's variable
    # needs its own dimension-scale attachment back to the shared one), enough
    # that on a 1432-group database it cost more than the phi/theta arrays it
    # was meant to save. A same-named local dimension in each group costs
    # nothing extra over what every database already paid before this existed;
    # only the phi/theta *values* -- the redundant part -- are skipped.
    this_group.createDimension("channels", phi.size)
    if write_grid_and_frequency:
        this_group.createVariable("phi", 'f8', ("channels",))
        this_group.variables['phi'][:] = phi.flatten()
        this_group.createVariable("theta", 'f8', ("channels",))
        this_group.variables['theta'][:] = theta.flatten()
    this_group.createVariable("dBA", 'f8', ("channels",))
    # Thrust scaling: a load factor of n raises the level by 20*log10(n).
    # A load factor of 0 is a special case -- it is not a physical condition but
    # a floor entry that extends the thrust-coefficient range down to CT = 0, so
    # trajectories below 1 g interpolate instead of clamping.  It carries the
    # 1 g levels unscaled; evaluating 20*log10(0) would write -inf into dBA, and
    # a zero barycentric weight against -inf yields NaN, poisoning every
    # interpolation that touches it.
    level_offset = 20 * np.log10(load_factor) if load_factor > 0 else 0.0
    this_group.variables['dBA'][:] = SPLA.flatten() + level_offset
    this_group.createVariable("EAA", 'f8', ("channels",))
    this_group.variables['EAA'][:] = EAA.flatten()
    # Store the source spectrum alongside the reduced levels, so a database can
    # be re-reduced (different weighting, different propagation distance)
    # without going back to the original sphere files.
    if frequency is not None and amplitude is not None:
        # Same reasoning as "channels" above: local dimensions in every group,
        # only the frequency values themselves are conditionally skipped.
        this_group.createDimension("frequency", np.size(frequency))
        this_group.createDimension("PHI", np.shape(amplitude)[0])
        this_group.createDimension("THETA", np.shape(amplitude)[1])
        if write_grid_and_frequency:
            this_group.createVariable("frequency", 'f8', ("frequency",))
            this_group.variables['frequency'][:] = np.asarray(frequency).flatten()
        this_group.createVariable("amplitude", 'f8', ("PHI", "THETA", "frequency"))
        this_group.variables['amplitude'][:] = amplitude


def project_sphere(filename, altitude, elv_cutoff, infreqs=None,
                   atmosphere=Atmosphere(temperature=293.15, pressure=101.325,
                                         relative_humidity=20.0)):
    distance = 1000  # reference distance for EAA
    azi, elv, phi, theta, radius, SPLO, SPLA, EAA, speed, flight_path_angle, _, _ = extract_SPL(filename, infreqs, distance,
                                                                                          atmosphere)
    # A direction with no energy (every band missing) has SPLA = -inf and an EAA
    # of -inf - -inf = NaN; give it EAA = 0, as the database does, so its ground
    # level is -inf (no energy) rather than NaN.
    SPLA, EAA = _finite_sphere_levels(SPLA, EAA)
    # Discard values outside of elevation cutoff
    included_angles = elv >= np.radians(elv_cutoff)
    azi = azi[included_angles]
    elv = elv[included_angles]
    SPLO = SPLO[included_angles]
    SPLA = SPLA[included_angles]
    EAA = EAA[included_angles]
    slant_range = altitude / np.sin(elv)
    ground_range = np.sqrt(slant_range ** 2 - altitude ** 2)
    # Plan view from above, as lambert_lon draws hemispheres: the flight
    # direction (azimuth 180) is +y and starboard (azimuth 90) is +x.  This was
    # -sin(azi) until 2026-09, which with art2umapr's old phi sign cancelled out;
    # once art2umapr put phi > 0 to starboard it mirrored every footprint.
    x = ground_range * np.sin(azi)
    y = -ground_range * np.cos(azi)
    absorption = EAA / distance * (slant_range - radius)
    spreading = 20 * np.log10(radius / slant_range)
    LA = SPLA - absorption + spreading
    return x, y, LA, speed, flight_path_angle


def read_vehicle_weight_newtons(directory_name):
    """Vehicle weight in newtons from a sphere directory's ``vehicle.cfg``.

    Separate from :func:`read_vehicle_data` so the database builder can write
    the root ``vehicle_weight_newtons`` without changing that function's
    fourteen-element return tuple.  Returns None when the config does not carry
    a weight.
    """
    config = ConfigParser()
    config.read(os.path.abspath(os.path.expanduser(directory_name) + '/vehicle.cfg'))
    if not config.has_section('Vehicle') or 'weight' not in config['Vehicle']:
        return None
    # Matches read_vehicle_data's constant, so the two agree to the digit.
    return 9.82 * float(config['Vehicle']['weight'])


def read_vehicle_data(directory_name, runs=None, speeds=None, flight_path_angles=None):
    # Process config file
    local_config = os.path.expanduser(directory_name) + '/vehicle.cfg'
    abs_config = os.path.abspath(local_config)
    config = ConfigParser()
    config.read(abs_config)
    number_of_main_rotor_blades = float(config['Main Rotor']['blades'])
    number_of_tail_rotor_blades = float(config['Tail Rotor']['blades'])
    if config.has_section('Option') and runs is not None:
        # read drag value and then open Excel reflist file for individual conditions
        nondimensional_flat_plate_drag = float(config['Option']['fbar'])
        reference_list = config['Option']['reflist']
        local_path_to_list = os.path.abspath(os.path.expanduser(directory_name) + '/' + reference_list)
        absolute_path_to_list = os.path.abspath(local_path_to_list)
        wb = openpyxl.load_workbook(absolute_path_to_list, read_only=True, data_only=True)
        ws = wb.active
        if ws is None:
            raise ValueError("Workbook has no active sheet")
        # Grab run values over the range in which they exist
        max_row = ws.max_row
        r_start = 'B2'
        r_end = 'B' + str(max_row)
        run_numbers = [i for i in np.array([[i.value for i in j] for j in ws[r_start:r_end]]).squeeze() if
                       i is not None]
        i_max = len(run_numbers)
        # Over the range where runs exist, grab the nondimensional condition values as an Nx3 matrix
        i_start = 'AH2'
        i_end = 'AJ' + str(i_max + 1)
        nondimensional_conditions = np.array([[i.value for i in j] for j in ws[i_start:i_end]]).squeeze()
        advance_ratios = []
        hover_tip_mach_numbers = []
        weight_coefficients = []
        for r in runs:
            index = run_numbers.index(r)
            advance_ratios.append(nondimensional_conditions[index][0])
            weight_coefficients.append(nondimensional_conditions[index][1])
            hover_tip_mach_numbers.append(nondimensional_conditions[index][2])
        advance_ratios = np.array(advance_ratios)
        hover_tip_mach_numbers = np.array(hover_tip_mach_numbers)
        weight_coefficients = np.array(weight_coefficients)
        # Grab dimensional condition indicated airspeeds in a similar way
        i_start = 'W2'
        i_end = 'W' + str(i_max + 1)
        dimensional_conditions = np.array([[i.value for i in j] for j in ws[i_start:i_end]]).squeeze()
        speeds = []
        for r in runs:
            index = run_numbers.index(r)
            speeds.append(dimensional_conditions[index])
        speeds = np.array(speeds)
        main_rotor_tip_speed = np.mean(0.514444 * speeds / advance_ratios)
        # Check on populating these?
        main_rotor_radius = None
        main_rotor_area = None
        tail_rotor_radius = None
        tail_rotor_area = None
        tail_rotor_tip_speed = None
        alphas = None
    else:
        main_rotor_radius = np.array(float(config['Main Rotor']['radius']))
        main_rotor_area = np.pi * main_rotor_radius ** 2
        main_rotor_tip_speed = np.array(float(config['Main Rotor']['tip speed']))
        tail_rotor_radius = np.array(float(config['Tail Rotor']['radius']))
        tail_rotor_area = np.pi * tail_rotor_radius ** 2
        tail_rotor_tip_speed = np.array(float(config['Tail Rotor']['tip speed']))
        ambient_temperature = np.array(float(config['Atmosphere']['temperature']))
        speed_of_sound = 20.05 * np.sqrt(ambient_temperature)
        ambient_density = np.array(float(config['Atmosphere']['density']))
        vehicle_weight_kg = np.array(float(config['Vehicle']['weight']))
        vehicle_weight_newtons = 9.82 * vehicle_weight_kg
        effective_flat_plate_drag_area = np.array(float(config['Vehicle']['drag']))
        nondimensional_flat_plate_drag = effective_flat_plate_drag_area / main_rotor_area
        hover_tip_mach_numbers = main_rotor_tip_speed / speed_of_sound
        weight_coefficients = vehicle_weight_newtons / (
                ambient_density * main_rotor_area * main_rotor_tip_speed ** 2)
        advance_ratios = None
        alphas = None
        if speeds is not None:
            ground_speed_meters_per_sec = 0.514444 * speeds
            advance_ratios = ground_speed_meters_per_sec / main_rotor_tip_speed
            hover_tip_mach_numbers = hover_tip_mach_numbers * np.ones_like(advance_ratios)
            drag_to_weight_ratio = 0.5 * nondimensional_flat_plate_drag * advance_ratios ** 2 / weight_coefficients
            if flight_path_angles is not None:
                alphas = -np.degrees(drag_to_weight_ratio) - flight_path_angles

    return (main_rotor_radius, main_rotor_area, main_rotor_tip_speed,
            number_of_main_rotor_blades, tail_rotor_radius, tail_rotor_area, tail_rotor_tip_speed,
            number_of_tail_rotor_blades, advance_ratios, weight_coefficients, hover_tip_mach_numbers, alphas,
            nondimensional_flat_plate_drag, speeds)


def project_directory(directory_name, altitude=500, cutoff=30, input_frequencies=None, fpa_climb_cutoff=5,
                      atmosphere=Atmosphere(temperature=293.15, pressure=101.325,
                                            relative_humidity=20.0), duration_correction=None):
    # Get list of full paths to netCDF files in directory
    local_glob = os.path.expanduser(directory_name) + '/*.nc'
    absolute_glob = os.path.abspath(local_glob)
    file_list = sorted(glob(absolute_glob))
    speeds = []
    flight_path_angles = []
    Lmax = []
    Lmean = []
    runs = []
    for filename in file_list:
        x, y, LA, speed, flight_path_angle = project_sphere(filename, altitude, cutoff, input_frequencies, atmosphere)
        if flight_path_angle <= fpa_climb_cutoff:
            # Directions with no energy (-inf) carry no level to take a max or
            # mean of; masked arrays used to hide them from these reductions.
            stencil = (np.sqrt(x ** 2 + y ** 2) > 0) & np.isfinite(LA)
            if not np.any(stencil):
                warnings.warn('{}: no direction below the aircraft has a level; skipping it'.format(filename))
                continue
            speeds.append(speed)
            flight_path_angles.append(flight_path_angle)
            Lmax.append(np.max(LA[stencil]))
            Lmean.append(np.mean(LA[stencil]))
            runs.append(int(filename[-6:-3]))
    speeds = np.array(speeds)
    flight_path_angles = np.array(flight_path_angles)
    Lmax = np.array(Lmax)
    Lmean = np.array(Lmean)
    # Apply duration correction if a reference speed is specified
    if duration_correction is not None:
        Lmax = Lmax + 10 * np.log10(duration_correction / speeds)
        Lmean = Lmean + 10 * np.log10(duration_correction / speeds)
    runs = np.array(runs)
    # Get nondimensional condition information
    (main_rotor_radius, main_rotor_area, main_rotor_tip_speed, number_of_main_rotor_blades,
     tail_rotor_radius, tail_rotor_area, tail_rotor_tip_speed, number_of_tail_rotor_blades,
     advance_ratios, weight_coefficients, hover_tip_mach_numbers, alphas, nondimensional_flat_plate_drag,
     speeds) = read_vehicle_data(directory_name, runs, speeds, flight_path_angles)

    return (speeds, flight_path_angles, Lmax, Lmean, advance_ratios, weight_coefficients, hover_tip_mach_numbers,
            alphas, runs, number_of_main_rotor_blades, number_of_tail_rotor_blades,
            nondimensional_flat_plate_drag, main_rotor_tip_speed)


def expand_if_single(x, r):
    """
    Expand a scalar or single-element array to match the shape of a reference array.

    This function takes a value (scalar or array) and expands it to match the shape
    of a reference array if it's a scalar or contains only one element. If the input
    already has multiple elements, it's returned unchanged.

    Args:
        x (scalar or array-like): The value to potentially expand. Can be a scalar
            or an array-like object.
        r (array-like): The reference array whose shape will be used for expansion.

    Returns:
        array-like: If x is a scalar or single-element array, returns an array of
            the same shape as r with all elements equal to x. Otherwise, returns x
            unchanged.
    """
    if np.isscalar(x) or len(x) == 1:
        x = x * np.ones_like(r)
    return x


def plot_fried_eggs(directory_names, metric='mean', dimensionless=False, altitude=500, cutoff=30,
                    input_frequencies=None, fpa_climb_cutoff=5,
                    atmosphere=Atmosphere(temperature=293.15, pressure=101.325,
                                          relative_humidity=20.0),
                    climb_rates=False, duration_correction=None, threshold=0.65, cull_noisy_fpa=None, xlim=(35, 140),
                    ylim=(-2000, 750), save_figures=False):
    for directory_name in directory_names:
        fig, ax, cs = fried_egg_plot(directory_name, metric, dimensionless, altitude, cutoff, input_frequencies,
                                     fpa_climb_cutoff, atmosphere, climb_rates, duration_correction, threshold,
                                     cull_noisy_fpa)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        mgr = getattr(fig.canvas, "manager", None)
        if mgr is not None and hasattr(mgr, "set_window_title"):
            mgr.set_window_title(directory_name)
        if save_figures:
            save_name = os.path.basename(directory_name) + '_fried_egg.pdf'
            fig.savefig(save_name)
    show(block=True)


def fried_egg_plot(directory_name, metric='mean', dimensionless=False, altitude=500, cutoff=30, input_frequencies=None,
                   fpa_climb_cutoff=5, atmosphere=Atmosphere(temperature=293.15, pressure=101.325,
                                                             relative_humidity=20.0),
                   climb_rates=False, duration_correction=None, threshold=0.65, cull_noisy_fpa=None,
                   suppress_classification=False):
    (speeds, flight_path_angles, Lmax, Lmean, advance_ratios, weight_coefficients, hover_tip_mach_numbers, alphas,
     runs, number_of_blades, number_of_tail_rotor_blades,
     effective_flat_plate_drags, tip_speeds) = project_directory(directory_name,
                                                                 altitude,
                                                                 cutoff,
                                                                 input_frequencies,
                                                                 fpa_climb_cutoff,
                                                                 atmosphere, duration_correction)
    samples = 1000
    if dimensionless:
        if advance_ratios is None or alphas is None:
            raise ValueError("Nondimensional data not available; set dimensionless=False or provide reflist.")
        x = advance_ratios
        y = alphas
    else:
        if climb_rates:
            assert speeds is not None and flight_path_angles is not None
            x = speeds
            y = 60 * 1.6878 * speeds * np.sin(np.radians(flight_path_angles))
        else:
            x = speeds
            y = flight_path_angles
    assert x is not None and y is not None
    xi = np.linspace(np.min(x), np.max(x), samples)
    yi = np.linspace(np.min(y), np.max(y), samples)
    if metric == 'mean':
        Lmetric = Lmean
    else:
        Lmetric = Lmax

    if cull_noisy_fpa is not None and threshold is not None:
        mask = data_filter(levels=Lmetric, flight_path_angles=flight_path_angles, threshold=threshold,
                           cull_noisy_fpa=cull_noisy_fpa)
        x = x[mask]
        y = y[mask]
        Lmetric = Lmetric[mask]

    triangles = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triangles, Lmetric)
    xim, yim = np.meshgrid(xi, yi)
    Li = interpolator(xim, yim)

    minSPL = np.min(Li)
    maxSPL = np.max(Li)
    num_levels = 9
    levels = np.linspace(minSPL, maxSPL, num_levels)
    color_map = get_ylorrd_cmap(num_levels)
    fig, ax = subplots(facecolor='white')
    cs = ax.contourf(xi, yi, Li, levels=levels, cmap=color_map)
    cb = colorbar(cs, format='%.0f')
    if metric == 'mean':
        if duration_correction is None:
            cb.set_label('Mean Ground Noise Level, dBA')
        else:
            cb.set_label('Ground Noise Exposure Level, dBA')
    else:
        if duration_correction is None:
            cb.set_label('Peak Ground Noise Level, dBA')
        else:
            cb.set_label('Ground Noise Exposure Level, dBA')
    if suppress_classification is False and threshold is not None:
        noisy_index = is_noisy(Lmetric, threshold)
        quiet_index = np.logical_not(noisy_index)
        ax.plot(x[quiet_index], y[quiet_index], 'ko', alpha=.8, markeredgecolor='w', markersize=10)
        ax.plot(x[noisy_index], y[noisy_index], 'ro', alpha=.8, markeredgecolor='k', markersize=10)
    if dimensionless:
        ax.set_xlabel('Advance Ratio - μ')
        ax.set_ylabel('Angle of Attack - α, °')
    else:
        if climb_rates:
            ax.set_xlabel('Speed, knots')
            ax.set_ylabel('Rate of Climb, fpm')
        else:
            ax.set_xlabel('Speed, knots')
            ax.set_ylabel('Flight Path Angle, °')
    return fig, ax, cs


def extract_SPL(filename, infreqs=None, distance=1000,
                atmosphere=Atmosphere(temperature=293.15, pressure=101.325,
                                      relative_humidity=20.0)):
    # Get data from netCDF file
    amplitude, phi, theta, frequency, radius, speed, flight_path_angle = load_nc_sphere(filename)
    # Convert radius to meters
    radius = radius * 0.3048
    # Replace masked values with no-energy SPL
    amplitude = mask_missing_levels(amplitude)
    # Check frequency range
    if infreqs is not None:
        frequency_index = (frequency >= infreqs[0]) & (frequency <= infreqs[1])
        amplitude = amplitude[:, :, frequency_index]
        frequency = frequency[frequency_index]
    SPLO = np.apply_along_axis(OASPL, 2, amplitude)
    Aweight = np.array([dBAw(f) for f in frequency])
    SPLA = np.apply_along_axis(OASPL, 2, amplitude + Aweight)
    # Convert from ART to UMAPR coordinates
    T, P = np.meshgrid(theta, phi)
    azi, elv = art2umapr(np.pi * P / 180.0, np.pi * theta / 180.0)
    # Compute excess atmospheric attenuation
    alpha = atmosphere.attenuation_coefficient(frequency)
    SPLAa = np.apply_along_axis(OASPL, 2, amplitude + Aweight - distance * alpha)
    # SPLs may contain -inf, which is expected, so suppress numpy warning
    np_error_settings = np.seterr()
    np.seterr(invalid='ignore')
    EAA = SPLA - SPLAa
    np.seterr(**np_error_settings)
    # frequency/amplitude are returned so callers can recompute band-based
    # quantities (e.g. EAA) after modifying the spectrum.
    return azi, elv, phi, theta, radius, SPLO, SPLA, EAA, speed, flight_path_angle, frequency, amplitude


def nc_unwrapped(filename, infreqs=None, weight=None):
    """
    Generate an unwrapped noise contour plot from a netCDF acoustic sphere.

    This function extracts sound pressure level (SPL) data from a file and creates
    a filled contour plot showing the acoustic field distribution in azimuth-elevation
    coordinates.

    Parameters
    ----------
    filename : str
        Path to the file containing netCDF formatted acoustic sphere data.
    infreqs : array-like, optional
        Input frequencies for SPL extraction. Default is None.
    weight : str, optional
        Frequency weighting to apply. If 'A', uses A-weighted SPL (SPLA),
        otherwise uses overall SPL (SPLO). Default is None.

    Returns
    -------
    None

    """
    azi, elv, phi, theta, radius, SPLO, SPLA, EAA, speed, flight_path_angle, _, _ = extract_SPL(filename, infreqs)
    if weight == 'A':
        SPL = SPLA
    else:
        SPL = SPLO
    # Directions with no energy (-inf) are left blank and do not set the scale.
    SPL = np.ma.masked_invalid(SPL)
    minSPL = np.min(SPL)
    maxSPL = np.max(SPL)
    num_levels = 9
    levels = np.round(np.linspace(minSPL, maxSPL, num_levels))
    color_map = get_ylorrd_cmap(num_levels)
    contourf(phi, theta, np.transpose(SPL), levels=levels, cmap=color_map)
    colorbar()


def plot_projection(filename, altitude=500, cutoff=30, infreqs=None, units='m'):
    """
    Generate a contour plot of sound pressure levels projected onto a ground plane.

    This function computes the acoustic footprint of an aircraft at a specified altitude
    by projecting sound pressure levels onto a horizontal plane and creating a contour plot.

    Args:
        filename (str): Path to the file containing netCDF formatted acoustic sphere data
        altitude (float, optional): Aircraft altitude in meters. Defaults to 500.
        cutoff (float, optional): Cutoff angle in degrees for projection calculations. 
            Defaults to 30.
        infreqs (array-like, optional): Input frequencies for analysis. Defaults to None.
        units (str, optional): Units for plot axes ('m' for meters, 'ft' for feet, etc.). 
            Defaults to 'm'.

    Returns:
        tuple: A tuple containing:
            - fig (matplotlib.figure.Figure): The figure object
            - ax (matplotlib.axes.Axes): The axes object
            - cs (matplotlib.contour.QuadContourSet): The contour set object

    Notes:
        - The function uses triangulation and linear interpolation to create smooth contours
        - Sound pressure levels are displayed in dBA (A-weighted decibels)
        - The plot uses a YlOrRd colormap with 9 levels
        - Coordinate system: cross-track (x-axis) and along-track (y-axis) directions
    """
    x, y, LA, speed, flight_path_angle = project_sphere(filename, altitude, cutoff, infreqs)
    fig, ax = subplots(facecolor='white')
    xi = np.linspace(np.min(x), np.max(x), 100)
    yi = np.linspace(np.min(y), np.max(y), 100)
    # Directions with no level (no energy) stay in the triangulation as NaN,
    # so the triangles touching them are left blank instead of interpolated
    # across, and they do not set the colour scale.
    LA = np.where(np.isfinite(LA), LA, np.nan)
    triangles = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triangles, LA)
    xim, yim = np.meshgrid(xi, yi)
    li = interpolator(xim, yim)
    num_levels = 9
    levels = np.round(10 * np.linspace(np.nanmin(LA), np.nanmax(LA), num_levels)) / 10
    color_map = get_ylorrd_cmap(num_levels)
    xi = unit_conversion.len_conv(xi, from_units='m', to_units=units)
    yi = unit_conversion.len_conv(yi, from_units='m', to_units=units)
    cs = ax.contourf(xi, yi, li, levels=levels, cmap=color_map)
    ax.axis('equal')
    ax.set_xlabel('Cross Track Direction, ' + units)
    ax.set_ylabel('Flight Track Direction, ' + units)
    cb = colorbar(cs, format='%.0f')
    cb.set_label('Sound Pressure Level, dBA')
    return fig, ax, cs

def _normalize_lambert_grid_convention(grid_convention):
    if grid_convention is None:
        return 'umapr'

    normalized = str(grid_convention).strip().lower()
    aliases = {
        'umapr': 'umapr',
        'art': 'art',
        'aam': 'art',
        'rnm': 'art',
    }
    if normalized not in aliases:
        raise ValueError("grid_convention must be one of 'umapr', 'art', 'aam', or 'rnm'")
    return aliases[normalized]


def _lambert_xy_to_umapr(x, y):
    azimuth = np.mod(np.pi / 2.0 - np.arctan2(-y, x), 2.0 * np.pi)
    q = np.sqrt(np.square(x) + np.square(y))
    elevation = np.pi / 2.0 - 2.0 * np.arcsin(np.clip(q / 2.0, 0.0, 1.0))
    return azimuth, elevation


def _umapr2art(azimuth, elevation):
    azimuth = np.asarray(azimuth, dtype=float)
    elevation = np.asarray(elevation, dtype=float)
    cos_elevation = np.cos(elevation)
    x = cos_elevation * np.sin(azimuth)
    y = -cos_elevation * np.cos(azimuth)
    z = -np.sin(elevation)
    theta = np.arccos(np.clip(y, -1.0, 1.0))
    phi = np.arctan2(x, -z)
    return phi, theta


def _configure_lambert_axes(ax):
    radius = np.sqrt(2.0)
    ax.set_aspect('equal', adjustable='box')
    if hasattr(ax, 'set_box_aspect'):
        ax.set_box_aspect(1.0)
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_lambert_grid(ax, grid_convention='umapr'):
    grid_convention = _normalize_lambert_grid_convention(grid_convention)

    if grid_convention == 'umapr':
        meridians = np.arange(0, 360, 45)
        for meridian in meridians:
            lats = np.linspace(0, 0.5 * np.pi, 1000)
            lons = lambert_lon(np.deg2rad(meridian)) * np.ones(len(lats))
            xm, ym = lambert_ea(lats, lons)
            (line,) = ax.plot(xm, ym, 'k--')
            line.set_gid(f'lambert-grid-umapr-meridian-{int(meridian)}')
            xl, yl = lambert_ea(np.deg2rad(-11.0), lambert_lon(np.deg2rad(meridian)))
            text = ax.text(
                xl,
                yl,
                "%d°" % int(meridian),
                horizontalalignment='center',
                verticalalignment='center',
            )
            text.set_gid(f'lambert-grid-umapr-meridian-label-{int(meridian)}')

        parallels = np.arange(0, 90, 30)
        for parallel in parallels:
            lons = np.linspace(-np.pi, np.pi, 1000)
            lats = np.deg2rad(parallel) * np.ones(len(lons))
            xm, ym = lambert_ea(lats, lons)
            (line,) = ax.plot(xm, ym, 'k--')
            line.set_gid(f'lambert-grid-umapr-parallel-{int(parallel)}')
            xl, yl = lambert_ea(np.deg2rad(parallel + 7.5), np.deg2rad(0.0))
            text = ax.text(
                xl + 0.025,
                yl - 0.025,
                "%d°" % parallel,
                horizontalalignment='left',
                verticalalignment='center',
            )
            text.set_gid(f'lambert-grid-umapr-parallel-label-{int(parallel)}')
    else:
        theta_curve = np.deg2rad(np.linspace(0.0, 180.0, 1000))
        phi_curve = np.deg2rad(np.linspace(-90.0, 90.0, 1000))
        label_bbox = dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85)

        for phi_deg in np.arange(-90, 91, 30):
            phi_line = np.deg2rad(phi_deg) * np.ones_like(theta_curve)
            azi_line, elv_line = art2umapr(phi_line, theta_curve)
            xm, ym = lambert_ea(elv_line, lambert_lon(azi_line))
            (line,) = ax.plot(xm, ym, 'k--')
            line.set_gid(f'lambert-grid-art-phi-{int(phi_deg)}')

            azi_label, elv_label = art2umapr(
                np.array([np.deg2rad(phi_deg)]),
                np.array([np.deg2rad(120.0)]),
            )
            xl, yl = lambert_ea(elv_label, lambert_lon(azi_label))
            if phi_deg > 0:
                label = f'+{int(phi_deg)}°'
            elif phi_deg < 0:
                label = f'{int(phi_deg)}°'
            else:
                label = '0°'

            x_label = float(xl[0])
            y_label = float(yl[0])
            r_label = np.hypot(x_label, y_label)
            if r_label > 0.0:
                x_label += 0.06 * x_label / r_label
                y_label += 0.06 * y_label / r_label

            text = ax.text(
                x_label,
                y_label,
                label,
                horizontalalignment='left' if phi_deg >= 0 else 'right',
                verticalalignment='center',
                bbox=label_bbox,
                clip_on=False,
                zorder=10,
            )
            text.set_gid(f'lambert-grid-art-phi-label-{int(phi_deg)}')

        for theta_deg in np.arange(30, 180, 30):
            theta_line = np.deg2rad(theta_deg) * np.ones_like(phi_curve)
            azi_line, elv_line = art2umapr(phi_curve, theta_line)
            xm, ym = lambert_ea(elv_line, lambert_lon(azi_line))
            (line,) = ax.plot(xm, ym, 'k--')
            line.set_gid(f'lambert-grid-art-theta-{int(theta_deg)}')

            azi_label, elv_label = art2umapr(
                np.array([np.deg2rad(80.0)]),
                np.array([np.deg2rad(theta_deg)]),
            )
            xl, yl = lambert_ea(elv_label, lambert_lon(azi_label))
            text = ax.text(
                float(xl[0]) + 0.08,
                float(yl[0]),
                f'{int(theta_deg)}°',
                horizontalalignment='left',
                verticalalignment='center',
                bbox=label_bbox,
                clip_on=False,
                zorder=10,
            )
            text.set_gid(f'lambert-grid-art-theta-label-{int(theta_deg)}')

        top_text = ax.text(
            0.0,
            np.sqrt(2.0) + 0.07,
            '0°',
            horizontalalignment='center',
            verticalalignment='bottom',
            bbox=label_bbox,
            clip_on=False,
            zorder=10,
        )
        top_text.set_gid('lambert-grid-art-theta-label-0')
        bottom_text = ax.text(
            0.0,
            -np.sqrt(2.0) - 0.07,
            '180°',
            horizontalalignment='center',
            verticalalignment='top',
            bbox=label_bbox,
            clip_on=False,
            zorder=10,
        )
        bottom_text.set_gid('lambert-grid-art-theta-label-180')

    def format_coord(x, y):
        cazi, celv = _lambert_xy_to_umapr(x, y)
        if grid_convention == 'art':
            cphi, ctheta = _umapr2art(cazi, celv)
            return 'φ = %0.1f, θ = %0.1f' % (np.degrees(cphi), np.degrees(ctheta))
        return 'ψ = %0.1f, θ = %0.1f' % (np.degrees(cazi), np.degrees(celv))

    ax.format_coord = format_coord


def nice_levels(vmin, vmax, target=9, step=None,
                steps=(0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 20.0, 25.0, 50.0)):
    """Contour levels on round values (0.5/1/2/2.5/5/10... dB) covering the data.

    Picks the smallest allowed step giving <= `target` intervals, then snaps
    the ends outward to multiples of it, so both the contour bands and the
    colourbar ticks land on values a reader can name. Pass `step` to force
    one (e.g. step=5 for even 5 dB bands).
    """
    vmin, vmax = float(vmin), float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    if step is None:
        raw = (vmax - vmin) / max(target - 1, 1)
        mag = 10.0 ** np.floor(np.log10(raw)) if raw > 0 else 1.0
        step = next((s * mag for s in steps if s * mag >= raw * 0.999),
                    steps[-1] * mag)
    lo = np.floor(vmin / step) * step
    hi = np.ceil(vmax / step) * step
    return np.arange(lo, hi + 0.5 * step, step)


def colorbar_ticks(levels, max_ticks=11):
    """Thin a level list down to at most `max_ticks` evenly spaced ticks."""
    levels = np.asarray(levels, dtype=float)
    if levels.size <= max_ticks:
        return levels
    return levels[:: int(np.ceil(levels.size / max_ticks))]


def plot_lambert_ea(azi,elv,SPL,SPL_range=None,weight=None,grid_convention='umapr',
                    levels=None,level_step=None):
    """
    Generate a Lambert equal-area azimuthal projection contour plot of sound pressure levels.

    This function creates a polar plot of acoustic data using Lambert equal-area projection,
    displaying sound pressure levels (SPL) as contours with azimuth and elevation coordinates.

    Parameters
    ----------
    azi : array-like
        Array of azimuth angles in radians.
    elv : array-like
        Array of elevation angles in radians.
    SPL : array-like
        Array of sound pressure levels corresponding to the azimuth and elevation angles.
    SPL_range : tuple of float, optional
        Tuple specifying (min_SPL, max_SPL) for the contour levels. If None, the range
        is automatically determined from the data. Default is None.
    weight : str, optional
        Units used for SPL label. Use 'A' for A-weighted SPL, otherwise
        overall SPL is used. Default is None.
    grid_convention : str, optional
        Grid overlay convention. Use 'umapr' for the existing azimuth/elevation grid or
        'art'/'aam'/'rnm' for RNM/AAM phi-theta grid lines. Default is 'umapr'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    cs : matplotlib.contour.QuadContourSet
        The contour set object from the contourf plot.

    """
    # Work on a copy, so the caller's -inf/NaN levels stay as they are; masked
    # cells (a masked array) count as missing too.
    SPL = np.ma.filled(np.ma.array(SPL, dtype=float, copy=True), np.nan)
    SPL[np.isinf(SPL)] = np.nan
    if SPL_range is None:
        minSPL = np.nanmin(SPL)
        maxSPL = np.nanmax(SPL)
    else:
        minSPL = SPL_range[0]
        maxSPL = SPL_range[1]
    SPL[np.isnan(SPL)] = 0.0
    # Contour bands on round dB values so the colourbar ticks are readable
    # (see `nice_levels`); pass `levels` or `level_step` to override.
    if levels is None:
        levels = nice_levels(minSPL, maxSPL, step=level_step)
    levels = np.asarray(levels, dtype=float)
    color_map = get_ylorrd_cmap(max(len(levels), 2))
    # Project to Cartesian
    lat = elv
    lon = lambert_lon(azi)
    x, y = lambert_ea(lat, lon)
    fig, ax = subplots(facecolor='white')
    ax.patch.set_visible(False)
    cs = ax.contourf(x, y, SPL, levels=levels, cmap=color_map)
    _configure_lambert_axes(ax)
    _draw_lambert_grid(ax, grid_convention=grid_convention)
    cb = colorbar(cs, pad=0.1, ticks=colorbar_ticks(levels))
    if weight == 'A':
        cb.set_label('Sound Pressure Level, dBA')
    else:
        cb.set_label('Sound Pressure Level, dB')
    return fig, ax, cs

def nc_lambert_ea(filename, input_frequencies=None, weight=None, SPL_range=None, grid_convention='umapr'):
    """
    Generate a Lambert equal-area azimuthal projection contour plot of sound pressure levels 
    on a netCDF formatted acoustic sphere.

    This function creates a polar plot of acoustic data using Lambert equal-area projection,
    displaying sound pressure levels (SPL) as contours with azimuth and elevation coordinates.

    Parameters
    ----------
    filename : str
        Path to the file containing acoustic data to be extracted and plotted.
    input_frequencies : array-like, optional
        Specific frequencies to extract from the data. If None, all frequencies are used.
    weight : str, optional
        Weighting scheme for SPL calculation. Use 'A' for A-weighted SPL, otherwise
        overall SPL is used. Default is None.
    SPL_range : tuple of float, optional
        Tuple specifying (min_SPL, max_SPL) for the contour levels. If None, the range
        is automatically determined from the data. Default is None.
    grid_convention : str, optional
        Grid overlay convention. Use 'umapr' for azimuth/elevation grid lines or
        'art'/'aam'/'rnm' for RNM/AAM phi-theta grid lines. Default is 'umapr'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    cs : matplotlib.contour.QuadContourSet
        The contour set object from the contourf plot.

    """
    azi, elv, phi, theta, radius, SPLO, SPLA, EAA, speed, flight_path_angle, _, _ = extract_SPL(filename, input_frequencies)
    if weight == 'A':
        SPL = SPLA
    else:
        SPL = SPLO
    fig, ax, cs = plot_lambert_ea(
        azi,
        elv,
        SPL,
        SPL_range,
        weight=weight,
        grid_convention=grid_convention,
    )
    return fig, ax, cs

def lambert_ea_points(azimuth, elevation, markers=None, colors=None, sizes=None, alpha=1.0, grid_convention='umapr'):
    """
    Plot levels on an acoustic sphere using the Lambert equal-area azimuthal projection.

    This function creates a polar plot using Lambert equal-area projection to visualize
    acoustic directivity patterns on a hemisphere.

    Parameters
    ----------
    azimuth : array_like
        Azimuth angles in radians. Convention: 0 at rear, increasing counterclockwise.
    elevation : array_like
        Elevation angles in radians. Convention: 0 at horizon, π/2 below the sphere.
    markers : array_like, optional
        Per-point marker styles. Must be the same size as azimuth/elevation when provided.
    colors : array_like or color, optional
        Per-point colors (same size as azimuth/elevation) or a single color spec.
    sizes : array_like or float, optional
        Per-point marker sizes (same size as azimuth/elevation) or a single size.
    alpha : float, optional
        Marker transparency, applied to all points. Default 1.0.
    grid_convention : str, optional
        Grid overlay convention. Use 'umapr' for azimuth/elevation grid lines or
        'art'/'aam'/'rnm' for RNM/AAM phi-theta grid lines. Default is 'umapr'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object with the Lambert projection plot.
    cs : list of matplotlib.artist.Artist
        Plot objects for the data points.

    """
    azimuth = np.asarray(azimuth)
    elevation = np.asarray(elevation)
    if azimuth.shape != elevation.shape:
        raise ValueError('azimuth and elevation must have the same shape')

    lat = elevation
    lon = lambert_lon(azimuth)
    x, y = lambert_ea(lat, lon)
    fig, ax = subplots(facecolor='white')
    ax.patch.set_visible(False)

    if markers is None and colors is None and sizes is None:
        cs = ax.plot(x, y, 'ro', markersize=4, alpha=alpha)
    else:
        x_flat = np.ravel(x)
        y_flat = np.ravel(y)
        npts = x_flat.size

        if markers is None:
            markers_arr = np.array(['o'] * npts, dtype=object)
        else:
            markers_arr = np.asarray(markers, dtype=object).ravel()
            if markers_arr.size != npts:
                raise ValueError('markers must match the number of points')

        color_scalar = 'r'
        colors_arr = None
        if colors is not None:
            c_arr = np.asarray(colors)
            if c_arr.ndim == 0 or c_arr.shape in [(3,), (4,)]:
                color_scalar = colors
            elif c_arr.ndim == 2 and c_arr.shape[0] == npts and c_arr.shape[1] in (3, 4):
                colors_arr = c_arr
            else:
                c_arr = c_arr.ravel()
                if c_arr.size != npts:
                    raise ValueError('colors must match the number of points')
                colors_arr = c_arr

        size_scalar = 16.0
        sizes_arr = None
        if sizes is not None:
            s_arr = np.asarray(sizes, dtype=float)
            if s_arr.ndim == 0:
                size_scalar = float(s_arr)
            else:
                s_arr = s_arr.ravel()
                if s_arr.size != npts:
                    raise ValueError('sizes must match the number of points')
                sizes_arr = s_arr

        cs = []
        for marker in dict.fromkeys(markers_arr.tolist()):
            mask = markers_arr == marker
            c_val = colors_arr[mask] if colors_arr is not None else color_scalar
            s_val = sizes_arr[mask] if sizes_arr is not None else size_scalar
            cs.append(
                ax.scatter(
                    x_flat[mask],
                    y_flat[mask],
                    c=c_val,
                    s=s_val,
                    marker=marker,
                    edgecolors='k',
                    linewidths=0.5,
                    alpha=alpha,
                )
            )
    _configure_lambert_axes(ax)
    _draw_lambert_grid(ax, grid_convention=grid_convention)
    return fig, ax, cs


def data_filter(levels, flight_path_angles, threshold=0.65, cull_noisy_fpa=-1.0):
    cull_index = np.logical_and(is_noisy(levels, threshold), flight_path_angles > cull_noisy_fpa)
    mask = np.logical_not(cull_index)
    return mask


def is_noisy(level, threshold=0.65):
    cutoff = (np.max(level) - np.min(level)) * threshold + np.min(level)
    noisy = level > cutoff
    return noisy


def geodetic2array(geodetic, reference, heading, units='ft'):
    """
    Convert geodetic coordinates to a local array coordinate system.

    This function transforms geodetic coordinates (latitude, longitude, altitude) into a local
    coordinate system centered at a reference point, rotated according to a specified heading.

    Args:
        geodetic (numpy.ndarray): An Nx3 array of geodetic coordinates where each row contains
            [latitude, longitude, altitude].
        reference (array-like): A 3-element array containing the reference point's geodetic
            coordinates [latitude, longitude, altitude].
        heading (float): The heading angle in degrees, measured clockwise from north (0-360).
            The local coordinate system is rotated so that the x-axis aligns with this heading.
        units (str, optional): The desired output length units. Defaults to 'ft' (feet).
            The function converts from meters to the specified units.

    Returns:
        numpy.ndarray: An Nx3 array of local coordinates where:
            - Column 0: x-coordinate (aligned with heading direction)
            - Column 1: y-coordinate (perpendicular to heading, positive left)
            - Column 2: z-coordinate (up, altitude above reference)

    Notes:
        - Input geodetic coordinates are first converted to ENU (East-North-Up) coordinates
          relative to the reference point.
        - The coordinate system is then rotated so that the x-axis points in the direction
          of the specified heading.
        - The rotation angle is computed as 90° - heading to align the x-axis with the heading.
    """
    east, north, up = geodetic2enu(geodetic[:, 0], geodetic[:, 1], geodetic[:, 2], reference[0], reference[1],
                                   reference[2])
    rotation = np.radians(90.0 - heading)
    local = np.zeros((len(east), 3))
    local[:, 0] = east * np.cos(rotation) + north * np.sin(rotation)
    local[:, 1] = -east * np.sin(rotation) + north * np.cos(rotation)
    local[:, 2] = up
    local = unit_conversion.len_conv(local, from_units='m', to_units=units)
    return local


def array2geodetic(local, reference, heading, units='ft'):
    """
    Convert local coordinate array to geodetic coordinates (latitude, longitude, height).

    This function transforms an array of points from a local coordinate system to geodetic 
    coordinates (WGS84) by first rotating the local coordinates based on a heading angle 
    to align with East-North-Up (ENU) convention, then converting to geodetic coordinates 
    relative to a reference point.

    Parameters
    ----------
    local : numpy.ndarray
        Array of shape (N, 3) containing local coordinates [x, y, z] in the
        frame :func:`geodetic2array` produces (this is its inverse):
        - x: along the heading (positive forward)
        - y: perpendicular to the heading (positive left)
        - z: vertical position (positive up)
    reference : array-like
        Reference point in geodetic coordinates [latitude, longitude, height] in degrees 
        and meters respectively, used as the origin for the ENU coordinate system.
    heading : float
        Heading angle in degrees (typically 0-360), measured clockwise from North.
        Used to rotate the local coordinates to align with true North.
    units : str, optional
        Units of the input local coordinates. Default is 'ft' (feet).
        Coordinates are converted to meters internally.

    Returns
    -------
    numpy.ndarray
        Array of shape (N, 3) containing geodetic coordinates where:
        - Column 0: latitude in degrees
        - Column 1: longitude in degrees
        - Column 2: height in meters

    Notes
    -----
    The conversion process involves:
    1. Converting input coordinates from specified units to meters
    2. Rotating coordinates by (heading - 90°) to convert from local to ENU frame
    3. Converting ENU coordinates to geodetic using the reference point
    """
    local = unit_conversion.len_conv(local, from_units=units, to_units='m')
    rotation = np.radians(heading - 90.0)
    # Convert local to ENU
    east = local[:, 0] * np.cos(rotation) + local[:, 1] * np.sin(rotation)
    north = -local[:, 0] * np.sin(rotation) + local[:, 1] * np.cos(rotation)
    up = local[:, 2]
    lat, lon, h = enu2geodetic(east, north, up, reference[0], reference[1], reference[2])
    geodetic = np.zeros_like(local)
    geodetic[:, 0] = lat
    geodetic[:, 1] = lon
    geodetic[:, 2] = h
    return geodetic


def write_kml(geodetic, savename, testname='Array', channel_prefix="M"):
    """
    Write geodetic coordinates to a KML/KMZ file for visualization in mapping applications.

    Parameters
    ----------
    geodetic : array-like
        An iterable of coordinate pairs where each row contains [latitude, longitude].
        Coordinates should be in decimal degrees.
    savename : str
        The output file path where the KMZ file will be saved.
    testname : str, optional
        The name of the KML document. Default is 'Array'.
    channel_prefix : str, optional
        The prefix to use for naming each point/placemark. Default is 'M'.
        Points will be named as '{channel_prefix}{number}' (e.g., 'M1', 'M2', etc.).

    Returns
    -------
    None
        The function saves the KMZ file to disk but does not return a value.

    Notes
    -----
    - Each point is displayed with a circular placemark icon from Google Maps.
    - Point numbering starts from 1 (i.e., channel_prefix + '1' for the first point).
    - The geodetic input should have longitude at index 1 and latitude at index 0.
    """
    kml = simplekml.Kml()
    kml.document.name = testname
    for (i, row) in enumerate(geodetic):
        pnt = kml.newpoint(name=channel_prefix + "{}".format(i + 1), coords=[(row[1], row[0])])
        pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
    kml.savekmz(savename)


def get_ylorrd_cmap(num_levels=9):
    """
    Return a YlOrRd colormap with approximately `num_levels` discrete levels.
    Uses palettable when an exact discrete palette exists (3..9),
    otherwise falls back to Matplotlib's continuous 'YlOrRd'.
    """
    try:
        import palettable.colorbrewer.sequential as cbseq
        attr = f"YlOrRd_{int(num_levels)}"
        if hasattr(cbseq, attr):
            return getattr(cbseq, attr).mpl_colormap
    except Exception:
        pass
    return matplotlib.colormaps['YlOrRd']

def atmosorb(freq, temp, humid, pstat):
    """
    Atmospheric absorption using acoustics.atmosphere module

    Args:
        freq: Array-like of frequencies in Hz.
        temp: Temperature in degrees Celsius.
        humid: Relative humidity in percent (e.g., 85% -> 85).
        pstat: Static pressure in mbar.

    Returns:
        numpy.ndarray of attenuation in dB per meter (same shape as `freq`).
    """
    # Convert inputs to Atmosphere() expected units
    tempK = np.asarray(temp, dtype=float) + 273.15  # C -> K
    pres_kpa = np.asarray(pstat, dtype=float) * 0.1  # mbar -> kPa
    rh_pct = np.asarray(humid, dtype=float)
    f = np.asarray(freq, dtype=float)

    # Scalar case: fast path
    if tempK.ndim == 0 and pres_kpa.ndim == 0 and rh_pct.ndim == 0:
        atm = Atmosphere(
            temperature=float(tempK),
            pressure=float(pres_kpa),
            relative_humidity=float(rh_pct),
        )
        alpha_db_per_m = atm.attenuation_coefficient(f)
        return alpha_db_per_m

    # Broadcast atmospheric inputs to common grid
    tempK_b, pres_kpa_b, rh_b = np.broadcast_arrays(tempK, pres_kpa, rh_pct)

    if f.ndim == 0:
        out = np.empty_like(tempK_b, dtype=float)
        it = np.nditer(tempK_b, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            atm = Atmosphere(
                temperature=float(tempK_b[idx]),
                pressure=float(pres_kpa_b[idx]),
                relative_humidity=float(rh_b[idx]),
            )
            out[idx] = float(atm.attenuation_coefficient(float(f)))
            it.iternext()
        return out
    else:
        # Frequency array: return array with shape (grid_shape + f.shape)
        out = np.empty(tempK_b.shape + f.shape, dtype=float)
        for idx in np.ndindex(tempK_b.shape):
            atm = Atmosphere(
                temperature=float(tempK_b[idx]),
                pressure=float(pres_kpa_b[idx]),
                relative_humidity=float(rh_b[idx]),
            )
            out[idx] = atm.attenuation_coefficient(f)
        return out

def geodist(elv1, azi1, elv2, azi2):
    """
    Compute geodesic arc length (degrees) on a sphere 

    Args:
        elv1, azi1, elv2, azi2: Scalars or array-like angles in degrees.

    Returns:
        numpy.ndarray (or scalar) of arc length in degrees.
    """
    elv1 = np.asarray(elv1, dtype=float)
    elv2 = np.asarray(elv2, dtype=float)
    azi1 = np.asarray(azi1, dtype=float)
    azi2 = np.asarray(azi2, dtype=float)

    # Convert degrees to radians
    r_elv1 = np.deg2rad(elv1)
    r_elv2 = np.deg2rad(elv2)
    r_azi1 = np.deg2rad(azi1)
    r_azi2 = np.deg2rad(azi2)

    # Spherical law of cosines
    arcs = np.sin(r_elv1) * np.sin(r_elv2) + np.cos(r_elv1) * np.cos(r_elv2) * np.cos(r_azi2 - r_azi1)

    # Clamp for numerical safety
    arcs = np.clip(arcs, -1.0, 1.0)

    # Arc length in degrees
    arclength = np.degrees(np.arccos(arcs))
    return arclength


def IDWweights(ielv, iazi, felv, fazi, rmax):
    """
    Inverse Distance Weights (Shepard) with Franke & Nielson adjustments.

    Mirrors the MATLAB implementation:
        function wi = IDWweights(ielv,iazi,felv,fazi,rmax)

    Args:
        ielv: Interpolation point elevation (deg)
        iazi: Interpolation point azimuth (deg)
        felv: Field elevations (deg), array-like
        fazi: Field azimuths (deg), array-like (same shape/broadcastable with felv)
        rmax: Maximum radius (deg) for weighting neighborhood

    Returns:
        numpy.ndarray of weights with the same broadcasted shape as felv/fazi.
    """
    felv = np.asarray(felv, dtype=float)
    fazi = np.asarray(fazi, dtype=float)
    # Broadcast to common shape if needed
    felv, fazi = np.broadcast_arrays(felv, fazi)

    # Build arrays for the interpolation point to match shape
    ielv_arr = np.full(felv.shape, float(ielv))
    iazi_arr = np.full(fazi.shape, float(iazi))

    # Geodesic distances (degrees)
    hi = geodist(ielv_arr, iazi_arr, felv, fazi)

    # Threshold exact data points to avoid division by zero
    eps = np.finfo(float).eps
    if np.any(hi <= 10.0 * eps):
        wi = np.zeros_like(hi, dtype=float)
        b = np.argmin(hi)
        wi.flat[b] = 1.0
        return wi

    # Neighborhood mask within rmax (up to 2D as in MATLAB)
    mask = hi <= float(rmax)
    m = np.zeros_like(hi, dtype=float)
    # Franke & Nielson measure
    m[mask] = ((float(rmax) - hi[mask]) / (float(rmax) * hi[mask])) ** 2

    # Normalization
    M = m.sum()
    if M > 0.0:
        wi = m / M
    else:
        # No neighbors within rmax — return zeros (matches safe behavior)
        wi = m
    return wi


#: Defaults for depropagate_hemisphere(interpolation={...}); see adaptive_idw_weights.
#: resolution_factor 0.5 takes half the window's arc, the sample being its centre.  On a
#: 2017 B407 sphere this gives about 3-5 deg at the rim, where samples crowd, 10-20 deg at
#: 10-40 deg below the horizon, and 30-50 deg overhead, where they are sparse and one window
#: spans a wide arc; 60 deg is the most a node may borrow from before it counts as a gap.
ADAPTIVE_INTERPOLATION = dict(k=8, min_mics=3, kappa=1.3, resolution_factor=0.5, max_radius_deg=60.0,
                              source_extent=0.0)


def adaptive_idw_weights(ielv, iazi, felv, fazi, mic, resolution_deg, *, k=8, min_mics=3, kappa=1.3,
                         resolution_factor=0.5, max_radius_deg=60.0, chunk=256):
    """Shepard weights with a radius chosen per node from the samples around it.

    A fixed radius has to be as large as the sparsest part of the sphere needs, or that part
    has holes, and is then far larger than the dense part can use.  On a 2017 B407 sphere the
    nearest sample is under a degree away at the rim but up to 18 deg away overhead, so the
    25 deg radius that keeps the overhead filled smears the rim -- by +1 dB at 2-4 deg below
    the horizon and -0.3 dB at 10-20 deg against the samples themselves.

    Here each node x gets
        R(x) = max(kappa d_k(x), kappa d_M(x), resolution_factor s(x)),
    with d_k the geodesic distance to the k-th nearest sample, d_M the distance within which
    samples from min_mics distinct microphones lie, and s the median angular resolution of
    its k nearest samples (``resolution_deg``: how much of the sphere one sample's window
    smears over).  Every node within reach of k samples from min_mics microphones is
    therefore filled -- no holes -- and each value brackets the node with several
    microphones, laterally and along track, rather than extrapolating off one microphone's
    line of samples.  kappa > 1 keeps the k-th sample inside the Franke and Nielson weight,
    which falls to zero at R.  A node that would need more than max_radius_deg is a genuine
    gap in the coverage, not a hole to smooth over: its row is empty and it is reported.

    Args:
        ielv, iazi: node elevations and azimuths (deg), 1-D, same length.
        felv, fazi: sample elevations and azimuths (deg), 1-D.  Geodesic distances, so no
            +-360 copies are needed (or wanted: they would be counted as extra samples).
        mic: (N,) the microphone each sample came from.
        resolution_deg: (N,) each sample's angular resolution, deg.

    Returns:
        (weights, radius_deg, gap): a scipy.sparse CSR matrix (nodes x samples) whose rows sum
        to one (empty for gaps), each node's radius (NaN for gaps), and the gap mask.
    """
    import scipy.sparse as sparse

    ielv = np.asarray(ielv, dtype=float).ravel()
    iazi = np.asarray(iazi, dtype=float).ravel()
    felv = np.asarray(felv, dtype=float).ravel()
    fazi = np.asarray(fazi, dtype=float).ravel()
    mic = np.asarray(mic).ravel()
    resolution_deg = np.asarray(resolution_deg, dtype=float).ravel()
    n = felv.size
    if not (fazi.size == mic.size == resolution_deg.size == n):
        raise ValueError('felv, fazi, mic and resolution_deg must have the same length')
    k = int(min(max(k, 1), n))

    def unit(elv, azi):
        e, a = np.deg2rad(elv), np.deg2rad(azi)
        return np.column_stack((np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)))

    samples = unit(felv, fazi)
    nodes = unit(ielv, iazi)
    mic_codes, mic_index = np.unique(mic, return_inverse=True)
    n_mics = mic_codes.size
    by_mic = np.argsort(mic_index, kind='stable')
    mic_starts = np.flatnonzero(np.r_[True, np.diff(mic_index[by_mic]) != 0])
    rows, cols, vals = [], [], []
    radius = np.full(ielv.size, np.nan)
    eps = 10.0 * np.finfo(float).eps
    for start in range(0, ielv.size, chunk):
        stop = min(start + chunk, ielv.size)
        h = np.degrees(np.arccos(np.clip(nodes[start:stop] @ samples.T, -1.0, 1.0)))
        order = np.argsort(h, axis=1)
        d_k = np.take_along_axis(h, order[:, k - 1:k], axis=1)[:, 0]
        # Nearest sample of each microphone, then the min_mics-th nearest microphone.
        nearest_per_mic = np.minimum.reduceat(h[:, by_mic], mic_starts, axis=1)
        if n_mics >= min_mics:
            d_m = np.sort(nearest_per_mic, axis=1)[:, min_mics - 1]
        else:
            d_m = np.full(stop - start, np.inf)
        s = np.median(resolution_deg[order[:, :k]], axis=1)
        r = np.maximum(np.maximum(kappa * d_k, kappa * d_m), resolution_factor * s)
        for i in range(stop - start):
            node = start + i
            hi = h[i]
            j_min = order[i, 0]
            if hi[j_min] <= eps:
                rows.append(np.array([node]))
                cols.append(np.array([j_min]))
                vals.append(np.array([1.0]))
                radius[node] = r[i]
                continue
            if not (r[i] <= max_radius_deg):
                continue
            near = np.flatnonzero(hi < r[i])
            m = ((r[i] - hi[near]) / (r[i] * hi[near])) ** 2
            total = m.sum()
            if not total > 0.0:
                continue
            rows.append(np.full(near.size, node))
            cols.append(near)
            vals.append(m / total)
            radius[node] = r[i]
    if rows:
        weights = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                    shape=(ielv.size, n))
    else:
        weights = sparse.csr_matrix((ielv.size, n))
    return weights, radius, ~np.isfinite(radius)


def shepIDW(ielv, iazi, felv, fazi, f, rmax):
    """
    Modified Shepard's Inverse Distance Weighting interpolation.

    Python version of the MATLAB function:
        function fi = shepIDW(ielv, iazi, felv, fazi, f, rmax)

    Args:
        ielv, iazi: Interpolant locations (deg). Scalars or array-like; must have same shape.
        felv, fazi: Data locations (deg). Arrays broadcastable to a common shape.
        f: Data values at (felv, fazi). Broadcastable to shape of felv/fazi.
        rmax: Maximum radius (deg) for neighborhood in weighting.

    Returns:
        numpy.ndarray of interpolated values with the same shape as `ielv` (or scalar if scalar inputs).
        NaN where no data point lies within `rmax`: there is no data there.
        (The MATLAB original gives 0, which reads as a measured zero -- zero
        energy, once the values are powers.)
    """
    # Ensure numpy arrays and broadcasting for data fields
    felv = np.asarray(felv, dtype=float)
    fazi = np.asarray(fazi, dtype=float)
    fvals = np.asarray(f, dtype=float)
    felv, fazi, fvals = np.broadcast_arrays(felv, fazi, fvals)

    # Prepare interpolant inputs and preserve original shape
    ielv_arr = np.asarray(ielv, dtype=float)
    iazi_arr = np.asarray(iazi, dtype=float)
    if ielv_arr.shape != iazi_arr.shape:
        ielv_arr, iazi_arr = np.broadcast_arrays(ielv_arr, iazi_arr)
    orig_shape = ielv_arr.shape
    ielv_flat = ielv_arr.ravel()
    iazi_flat = iazi_arr.ravel()

    fi_flat = np.empty_like(ielv_flat, dtype=float)
    for i in range(ielv_flat.size):
        wi = IDWweights(ielv_flat[i], iazi_flat[i], felv, fazi, rmax)
        # Only the neighbours: 0 * NaN is NaN, so one bad sample anywhere
        # would otherwise poison every node, however far away.
        near = wi > 0.0
        fi_flat[i] = np.sum(fvals[near] * wi[near]) if near.any() else np.nan

    fi = fi_flat.reshape(orig_shape)
    if fi.ndim == 0:
        return float(fi)
    return fi

def load_NASA_track(trackfile):
    """
    Load NASA track file and return data in a structured format.

    Args:
        trackfile: Path to NASA CSV format tracking data file.
    Returns:
        dict with keys:
            'time': numpy.ndarray of UTC time in seconds
            'latitude': numpy.ndarray of latitude in degrees
            'longitude': numpy.ndarray of longitude in degrees
            'altitude': numpy.ndarray of altitude in feet
            'heading': numpy.ndarray of heading in degrees  
            'pitch': numpy.ndarray of pitch in degrees
            'roll': numpy.ndarray of roll in degrees
            'x', 'y', 'z': numpy.ndarray of position in feet (local coordinates)
            'vx', 'vy', 'vz': numpy.ndarray of velocity in feet/second (local coordinates)
    """    
    data = np.genfromtxt(trackfile, delimiter=',', names=True)
    time = data['utcsec']
    latitude = data['lat']
    longitude = data['lon']
    altitude = data['alt']  
    heading = data['heading']
    pitch = data['pitch']
    roll = data['roll']
    x = data['x']
    y = data['y']
    z = data['z']
    vx = data['vx']
    vy = data['vy']
    vz = data['vz']   

    return {
        'time': time,
        'latitude': latitude,
        'longitude': longitude,
        'altitude': altitude,
        'heading': heading,
        'pitch': pitch,
        'roll': roll,
        'x': x,
        'y': y,
        'z': z,
        'vx': vx,
        'vy': vy,
        'vz': vz,
    }

def filter_track(track, xlims = None, ylims = None, zlims = None, decimate=1):
    """
    Filter track data based on specified limits for x, y, z coordinates.

    Args:
        track: dict containing track data with keys 'x', 'y', 'z'.
        xlims: tuple (xmin, xmax) for filtering x coordinates.
        ylims: tuple (ymin, ymax) for filtering y coordinates.
        zlims: tuple (zmin, zmax) for filtering z coordinates.
        decimate: integer factor to decimate the track data (default: 1, no decimation).
    Returns:
        dict containing filtered track data with the same keys as input.
    """    
    x = track['x']
    y = track['y']
    z = track['z']
    mask = np.ones_like(x, dtype=bool)

    if xlims is not None:
        mask = np.logical_and(mask, x >= xlims[0])
        mask = np.logical_and(mask, x <= xlims[1])
    if ylims is not None:
        mask = np.logical_and(mask, y >= ylims[0])
        mask = np.logical_and(mask, y <= ylims[1])
    if zlims is not None:
        mask = np.logical_and(mask, z >= zlims[0])
        mask = np.logical_and(mask, z <= zlims[1])

    filtered_track = {}
    for key in track:
        filtered_track[key] = track[key][mask]
    # Decimate track data
    if decimate > 1:
        for key in filtered_track:
            filtered_track[key] = filtered_track[key][::decimate]
    return filtered_track   

def spherical_reflection_coefficient(cos_grazing, image_range, f, a, flores,
                                     boundary_loss_correction=True, admittance=None):
    """Spherical-wave reflection coefficient Q of a locally reacting ground (e^{-i omega t}).

    Q = Rp + F(w) (1 - Rp): the plane-wave coefficient Rp with Chessell's
    boundary-loss factor F, over a Delany-Bazley ground of flow resistance
    ``flores`` (kPa s/m^2).  ``cos_grazing`` is the cosine of the angle from the
    surface normal at the specular point, ``image_range`` the image-source path
    length, ``a`` the sound speed (lengths and speed in matching units).  This is
    the coefficient :func:`ega` uses; it is exposed so that models needing the
    complex coefficient itself (impedance discontinuities, ground planes) use the
    same one.

    ``admittance``, when given, is the normalised surface admittance beta
    (same shape as, or broadcastable to, ``f``) and replaces the
    Delany-Bazley one from ``flores`` -- for other impedance models.
    """
    f = np.asarray(f, dtype=float)
    if admittance is None:
        flores = np.asarray(flores, dtype=float)
        freq_resistance_ratio = f / flores  # Normalized frequency-to-resistance ratio
        inv_freq_ratio = freq_resistance_ratio ** (-0.73)  # Inverse frequency ratio (Delany-Bazley)
        impedance_ratio = 1.0 / (1.0 + 9.08 * inv_freq_ratio / (freq_resistance_ratio ** 0.02) + 1j * 11.9 * inv_freq_ratio)
    else:
        impedance_ratio = np.asarray(admittance, dtype=complex)

    plane_wave_coeff = (cos_grazing - impedance_ratio) / (cos_grazing + impedance_ratio)  # Plane wave reflection coefficient
    if not boundary_loss_correction:
        return plane_wave_coeff

    # Compute numerical distance (simplified: 0.5*k1 = π*f/a)
    ground_effect_param = np.sqrt(
        1j * np.pi * f * image_range / a / (1.0 + impedance_ratio * cos_grazing)
    ) * (cos_grazing + impedance_ratio)
    w = ground_effect_param ** 2  # Numerical distance parameter

    # Compute boundary loss factor (ground surface effect)
    boundary_loss = np.zeros_like(w, dtype=complex)
    mask = np.abs(w) <= 500
    sqrt_boundary = np.sqrt(w[mask])
    boundary_loss[mask] = 1 + 1j * np.sqrt(np.pi * w[mask]) * np.exp(-w[mask]) * (1 - erf(-1j * sqrt_boundary))

    # Combined reflection + boundary loss
    return plane_wave_coeff + boundary_loss * (1.0 - plane_wave_coeff)


def ega(hs, hr, d2, f, a, flores, pt=True, cturb=0.0, boundary_loss_correction=True):
    """
    Calculate excess ground attenuation for a non-directional point source.
    
    Based on procedures in:
    - Chien and Soroka, "Sound Propagation Along an Impedance Plane",
      J. Sound. Vib., 43(1), 9-20, 1975 (corrected 1980)
    - Delany, Bazley, "Acoustical Properties of Fibrous Absorbent Materials",
      Appl. Acoust., 3, 105-116, 1970.
    - Chessell, "Propagation of noise along an impedance boundary",
      JASA 62(4), 825-834, 1977
    - Daigle, G.A., T.F.W. Embelton, and J.E. Piercy. 1979. "Some Comments on 
      the Literature of Propagation Near Boundaries of Finite Acoustical 
      Impedance," J. Acoust. Soc. Am. 66(3), 918-919.
    - Stusnick, Plotkin, Sutherland, "Short-Range Acoustic Propagation Model",
      Wyle Research Report WR 85-19, July 1985.
    
    Assumes level terrain, no wind, straight acoustic rays.
    
    Args:
        hs: Source height (ft or m, must match other length units)
        hr: Receiver height (ft or m)
        d2: 2D distance between source and receiver (ft or m)
        f: Frequency (Hz), scalar or array
        a: Speed of sound (ft/s or m/s, matching length units)
        flores: Specific flow resistance (kPa·s/m²)
            Typical values (from AAM technical reference):
                - Snow cover: 30
                - Grassy field: 225
                - Roadside soil: 650
                - Packed sand: 1650
                - Hard packed dirt: 3000
                - Exposed rock: 6000
                - Concrete: 10000
                - Asphalt: 50000
                - Water: 1e6 (effectively rigid)
        pt: True for pure tone (no third octave smearing), False for broadband (default: True)
        cturb: Turbulence parameter (rad·s·(m or ft)^-0.5); it multiplies
               f·sqrt(range), so its value depends on the length unit.
               Typical: 0 to 16e-4 rad·s·m^-0.5, i.e. 0 to 8.8e-4 rad·s·ft^-0.5.
               Used by the broadband mode only (pt=False).
         boundary_loss_correction: When True (default), includes the boundary-loss factor
             correction to the plane-wave reflection coefficient (Chessell), which is
             most relevant at grazing incidence. When False, uses only the plane-wave
             reflection coefficient.
    
    Returns:
        atten: Attenuation in dB
        phase: Phase (radians) of resultant wave relative to direct path
               (Pure tones only; NaN for broadband case)
    
    Notes:
        - Output "attenuation" is an attenuation factor in dB
        - Add to the SPL of the direct wave to compute SPL at receiver
        - Phase offset is meaningful only for pure tones
    """
    # Ensure float arrays
    hs = np.asarray(hs, dtype=float)
    hr = np.asarray(hr, dtype=float)
    d2 = np.asarray(d2, dtype=float)
    f = np.asarray(f, dtype=float)
    a = np.asarray(a, dtype=float)
    flores = np.asarray(flores, dtype=float)
    cturb = np.asarray(cturb, dtype=float)
    
    mu = 0.727477  # Spherical spreading coefficient
    eta = 6.325159  # Ground reflection coefficient
    
    # Calculate geometric values
    direct_range = np.sqrt(d2**2 + (hs - hr)**2)  # Direct acoustic path distance
    image_range = np.sqrt(d2**2 + (hs + hr)**2)  # Image source acoustic path distance
    grazing_angle = np.arccos((hs + hr) / image_range)  # Grazing angle (radians)
    path_delay = (image_range - direct_range) / a  # Time delay between direct and image paths
    range_ratio = image_range / direct_range  # Ratio of distances
    
    image_source_coeff = spherical_reflection_coefficient(
        np.cos(grazing_angle), image_range, f, a, flores, boundary_loss_correction)
    image_source_magnitude = np.abs(image_source_coeff)  # Magnitude of image source term
    image_source_phase = np.angle(image_source_coeff)  # Phase of image source term
    
    # Compute excess ground attenuation
    if pt:
        # Pure tone expression (Chessell's Equation 19)
        phase_delay = 1j * 2.0 * np.pi * f * path_delay  # Phase delay between paths
        resultant_amplitude = 1.0 + image_source_coeff * np.exp(phase_delay) / range_ratio  # Direct + image amplitude
        phase = np.angle(resultant_amplitude)
        attn = np.abs(resultant_amplitude)**2
    else:
        # Broadband mode (Chessell's Equations 20, 21)
        freq_path_delay = f * path_delay  # Normalized frequency-path delay product
        normalized_image_mag = image_source_magnitude / range_ratio  # Normalized image magnitude
        normalized_image_mag_sq = normalized_image_mag ** 2
        
        # Add turbulence factor (Chessell Equations 25, 27)
        turbulence_factor = np.exp(-(0.5 * cturb * f * np.sqrt(direct_range))**2) if np.any(cturb > 0.0) else 1.0
        
        # Combine attenuation computation
        ground_phase_term = eta * freq_path_delay + image_source_phase  # Ground reflection phase
        spherical_phase_term = mu * freq_path_delay  # Spherical spreading phase
        cosine_factor = np.cos(ground_phase_term) * turbulence_factor
        
        attn = 1.0 + normalized_image_mag_sq + 2.0 * normalized_image_mag * cosine_factor
        mask_positive_delay = path_delay > 0.0
        attn = np.where(
            mask_positive_delay,
            1.0 + normalized_image_mag_sq + 2.0 * normalized_image_mag * np.sin(spherical_phase_term) * cosine_factor / spherical_phase_term,
            attn
        )
        
        # Phase has no meaning for broadband, so set to NaN
        phase = np.full_like(attn, np.nan)
    
    # Convert magnitude to dB
    if np.all(attn > 0.0):
        attenuation_db = 10.0 * np.log10(attn)
    else:
        raise ValueError('Error in EGA: negative attenuation magnitude encountered')
    
    return attenuation_db, phase