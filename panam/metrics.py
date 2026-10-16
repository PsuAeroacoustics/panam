"""Noise metrics: perceived noise level (PNL, PNLT, EPNL), sound exposure level and
MIL-STD-1474E aural nondetectability.

Band levels are one-third octave levels in dB; times in s; distances in m.
``docs/THEORY.md`` section 9 gives the equations.
"""

import os

import numpy as np


# --------------------------------------------------------------------------
# Perceived noise level metrics (PNL, PNLT, EPNL)
#
# Implements 14 CFR Part 36 Appendix A, section A36.4 (identical to ICAO
# Annex 16 Vol. I Appendix 2): noy conversion via Table A36-3, PNL, the
# ten-step spectral-irregularity (tone) correction with Table A36-2, the
# A36.4.4.2 band-sharing check, applied as ICAO Annex 16 Vol. I App. 2 adds
# it (a separate Delta_B on the EPNL), and the duration correction.
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
    Returns: array of noys, same shape (0 below SPL(d), NaN for a NaN band)
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
    return np.where(np.isnan(spl), np.nan, n)


def perceived_noise_level(band_levels):
    """
    Perceived noise level PNL(k), 14 CFR 36 Appendix A, A36.4.2.

    Args:
        band_levels: (..., 24) SPL in dB for the bands 50 Hz..10 kHz
    Returns: PNL in PNdB (shape band_levels.shape[:-1]); -inf where the
        spectrum produces zero total noisiness, NaN where a band is NaN
    """
    n = noys(band_levels)
    total = 0.85 * n.max(axis=-1) + 0.15 * n.sum(axis=-1)
    with np.errstate(divide="ignore"):
        return 40.0 + (10.0 / np.log10(2.0)) * np.log10(total)


def tone_correction(band_levels, masked=False):
    """
    Tone correction factor C(k), 14 CFR 36 Appendix A, A36.4.3.1 steps 1-10.

    For finite band levels and ``masked=False`` this is the regulation's
    procedure as written.  A band of -inf (zero energy) has no slope, and
    taken literally it makes the background of step 7 infinite or NaN: the
    band next to it then reads as a full 20/3 dB tone, or, when the -inf band
    is band 3 (the background's anchor), every tone in the spectrum is lost.
    A -inf band therefore enters steps 1-7 at its noy threshold SPL(d) of
    Table A36-3 (the level below which it contributes no noys, so PNL is
    unchanged) and carries no tone correction itself.  With ``masked=True``
    every band below SPL(d) is treated so, which keeps an inaudible band
    rising out of a steep high-frequency fall from reading as a tone -- the
    prediction's counterpart of excluding tones in masked bands (FAA AC 36-4,
    Appendix 2) and NICEOPS's ToneMasking::below_noy_floor.

    Args:
        band_levels: (n_times, 24) or (24,) SPL in dB, bands 50 Hz..10 kHz
        masked: treat every band below SPL(d), not only -inf ones, as masked
    Returns: tuple (c_max, tone_band_index)
        c_max: (n_times,) largest tone correction factor, dB; NaN for a
            spectrum with a NaN band
        tone_band_index: (n_times,) band index (0-23) it occurred in (0 when
            there is no tone); -1 where c_max is NaN
    """
    levels = np.atleast_2d(np.asarray(band_levels, dtype=float))
    if np.isposinf(levels).any():
        raise ValueError('positive infinite band levels are invalid')
    missing = np.isnan(levels).any(axis=1)
    masked_bands = (levels < _NOY_SPL_D) if masked else np.isneginf(levels)
    spl = np.where(masked_bands, _NOY_SPL_D, levels)
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
    c[masked_bands] = 0.0                        # masked bands carry no tone

    # Step 10
    c_max = np.where(missing, np.nan, c.max(axis=1))
    tone_band = np.where(missing, -1, np.argmax(c, axis=1))
    if np.ndim(band_levels) == 1:
        return float(c_max[0]), int(tone_band[0])
    return c_max, tone_band


def tone_corrected_perceived_noise_level(band_levels, masked=False):
    """
    PNLT(k) = PNL(k) + C(k), 14 CFR 36 Appendix A, A36.4.3.

    Args:
        band_levels: (n_times, 24) or (24,) SPL in dB, bands 50 Hz..10 kHz
        masked: passed to :func:`tone_correction`
    Returns: tuple (pnlt, pnl, c_max, tone_band_index)
    """
    pnl = perceived_noise_level(band_levels)
    c_max, tone_band = tone_correction(band_levels, masked=masked)
    return pnl + c_max, pnl, c_max, tone_band


#: EPNL duration-correction reference time T, s (14 CFR 36 A36.4.5).
EPNL_REFERENCE_DURATION_S = 10.0


#: 10 lg(dt/T) for dt = 0.5 s as A36.4.5.4 writes it (exactly -13.0103).
EPNL_HALF_SECOND_NORMALIZATION_DB = -13.0


def effective_perceived_noise_level(band_level_history, dt=0.5,
                                    bandshare_adjustment=True, masked=False,
                                    normalization='regulatory'):
    """
    EPNL of a noise event, 14 CFR 36 Appendix A, A36.4.

    Args:
        band_level_history: (n_times, 24) third-octave SPL history in dB
            (bands 50 Hz..10 kHz) at equal `dt` increments
        dt: time increment, s (the regulation prescribes 0.5 s)
        bandshare_adjustment: apply the A36.4.4.2 five-interval check for
            tone suppression by band sharing at PNLTM
        masked: passed to :func:`tone_correction`
        normalization: 'regulatory' (default) takes the A36.4.5.4 constant
            -13 dB for 10 lg(dt/T) when dt = 0.5 s and the exact value for
            any other dt; 'exact' always takes 10 lg(dt/T), -13.0103 dB for
            0.5 s.  T = 10 s.

    Returns: dict with
        epnl: EPNL in EPNdB, PNLTM + D + delta_b
        pnltm: PNLTM after the band-sharing check (pnltm_unadjusted +
            delta_b), TPNdB
        pnltm_unadjusted: the maximum of PNLT, TPNdB
        delta_b: band-sharing adjustment, dB (0 when not applied)
        duration_correction_db: D = 10 lg(sum 10^(PNLT/10) dt / T) - PNLTM,
            from the unadjusted PNLTM, so epnl = pnltm + D
        k1, k2: duration-interval sample limits, inclusive
        pnlt, pnl, c_max, tone_band: per-sample histories
        clipped: True when the 10 dB-down interval hits the record edge

    The duration limits follow A36.4.5.5: the PNLT samples closest to
    PNLTM - 10 and, with several peaks, those giving the longest duration.
    Starting from the first and last samples at or above PNLTM - 10, each
    limit moves one sample outward when that sample is strictly closer to
    PNLTM - 10.  Every sample between the limits is summed, dips below
    PNLTM - 10 included.

    The band-sharing adjustment follows ICAO Annex 16 Vol. I, Appendix 2
    (and FAA AC 36-4): when C at PNLTM is below the mean C of the five
    records centered there, delta_b = PNL(kM) + C_avg - PNLTM is added to the
    EPNL as a separate term.  PNLTM, the 10 dB-down limits and D are those of
    the unadjusted PNLT history.  Read literally, 14 CFR 36 raises PNLTM and
    takes D from it, and the two cancel in EPNL = PNLTM + D.
    """
    if normalization not in ('regulatory', 'exact'):
        raise ValueError("normalization must be 'regulatory' or 'exact'")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be positive and finite')
    spl = np.asarray(band_level_history, dtype=float)
    if np.isnan(spl).any():
        raise ValueError("NaN band levels in the history")
    pnlt, pnl, c_max, tone_band = tone_corrected_perceived_noise_level(spl, masked=masked)
    if not np.isfinite(pnlt).any():
        raise ValueError("no finite PNLT values in the history")

    k_m = int(np.nanargmax(pnlt))
    pnltm = float(pnlt[k_m])
    delta_b = 0.0
    if bandshare_adjustment:
        # A36.4.4.2: if C at PNLTM is below the average of the five
        # consecutive intervals centered there, tone suppression by band
        # sharing is suspected; the difference is the adjustment Delta_B.
        lo, hi = max(0, k_m - 2), min(len(pnlt), k_m + 3)
        c_avg = float(np.mean(c_max[lo:hi]))
        if c_max[k_m] < c_avg:
            delta_b = float(pnl[k_m] + c_avg) - pnltm

    # Duration interval, A36.4.5.5: the outermost samples at or above
    # PNLTM-10, each moved one sample outward when that one is closer to it.
    threshold = pnltm - 10.0
    above = np.where(pnlt >= threshold)[0]
    k1, k2 = int(above[0]), int(above[-1])
    if k1 > 0 and abs(pnlt[k1 - 1] - threshold) < abs(pnlt[k1] - threshold):
        k1 -= 1
    if k2 < len(pnlt) - 1 and abs(pnlt[k2 + 1] - threshold) < abs(pnlt[k2] - threshold):
        k2 += 1
    if normalization == 'regulatory' and dt == 0.5:
        normalization_db = EPNL_HALF_SECOND_NORMALIZATION_DB
    else:
        normalization_db = 10.0 * np.log10(dt / EPNL_REFERENCE_DURATION_S)
    duration = 10.0 * np.log10(
        np.sum(10.0 ** (pnlt[k1:k2 + 1] / 10.0))) + normalization_db - pnltm
    return {
        "epnl": pnltm + duration + delta_b,
        "pnltm": pnltm + delta_b,
        "pnltm_unadjusted": pnltm,
        "delta_b": delta_b,
        "duration_correction_db": duration,
        "k1": k1, "k2": k2,
        "pnlt": pnlt, "pnl": pnl, "c_max": c_max, "tone_band": tone_band,
        "clipped": bool(k1 == 0 or k2 == len(pnlt) - 1),
    }


def ten_db_down_interval(levels, down=10.0):
    """Sample limits of the 10 dB-down duration of a level history.

    ``(k1, k2)``: the first and last samples at or above ``max - down``.  Dips
    below the threshold between them are inside the interval, so a second
    rise (e.g. a hover at the end of an approach heard from upstream) is part
    of the event, not cut off at the first dip.  Samples below the threshold
    are never limits, even when one is closer to ``max - down`` than the
    limit; the EPNL duration of 14 CFR 36 A36.4.5.5 instead takes the closest
    samples (:func:`effective_perceived_noise_level`).
    """
    levels = np.asarray(levels, dtype=float)
    if levels.ndim != 1 or not levels.size:
        raise ValueError('levels must be a nonempty one-dimensional history')
    if np.isnan(down) or down < 0:
        raise ValueError('down must be nonnegative or positive infinity')
    finite = np.isfinite(levels)
    if not finite.any():
        raise ValueError('no finite levels in the history')
    peak = np.nanmax(np.where(finite, levels, -np.inf))
    above = np.where(finite & (levels >= peak - float(down)))[0]
    return int(above[0]), int(above[-1])


def sound_exposure_level(levels, dt, down=10.0, weighted_levels=None, *, missing="raise"):
    """Sound exposure level of a noise event over its 10 dB-down duration.

    SEL = 10 log10( sum 10^(L/10) dt / T0 ), T0 = 1 s, summed over the interval
    from :func:`ten_db_down_interval`: the first to the last sample within
    ``down`` dB of the maximum, dips between them included.  Pass
    ``down=np.inf`` to integrate the whole record instead.

    Args:
        levels: level history (dB; normally A-weighted, i.e. LA) at equal ``dt``.
        dt: sample interval, s.
        down: how far below the maximum the duration extends, dB (default 10).
        missing: 'raise' (default) rejects NaN samples; 'omit' integrates
            available samples and reports missing_samples. -inf means zero energy.
        weighted_levels: optional history to integrate over the interval chosen
            from ``levels`` (e.g. choose the interval on LA, integrate LC).

    Returns: dict with
        sel: sound exposure level, dB
        lmax: maximum level, dB
        k1, k2: interval sample limits
        duration_s: (k2 - k1 + 1) * dt
        missing_samples: number of omitted integration samples
        clipped: True when the interval touches the record edge -- the event
            may extend beyond the data and the SEL is then a lower bound.
    """
    levels = np.asarray(levels, dtype=float)
    if levels.ndim != 1 or not levels.size:
        raise ValueError('levels must be a nonempty one-dimensional history')
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be positive and finite')
    if np.isnan(down) or down < 0:
        raise ValueError('down must be nonnegative or positive infinity')
    if missing not in ('raise', 'omit'):
        raise ValueError("missing must be 'raise' or 'omit'")
    integrand = levels if weighted_levels is None else np.asarray(weighted_levels, dtype=float)
    if integrand.shape != levels.shape:
        raise ValueError('weighted_levels must have the same shape as levels')
    if np.isposinf(levels).any() or np.isposinf(integrand).any():
        raise ValueError('positive infinite levels are invalid')
    if missing == 'raise' and (np.isnan(levels).any() or np.isnan(integrand).any()):
        raise ValueError("missing levels; use missing='omit' to integrate available samples")
    if np.isfinite(down):
        k1, k2 = ten_db_down_interval(levels, down)
    else:
        k1, k2 = 0, levels.size - 1
    segment = integrand[k1:k2 + 1]
    energy = np.sum(10.0 ** (segment[np.isfinite(segment)] / 10.0)) * float(dt)
    return {
        'sel': 10.0 * np.log10(energy) if energy > 0 else -np.inf,
        'lmax': float(np.nanmax(levels)),
        'k1': k1, 'k2': k2,
        'duration_s': (k2 - k1 + 1) * float(dt),
        'missing_samples': int(np.isnan(segment).sum()),
        'clipped': bool(k1 == 0 or k2 == levels.size - 1),
    }


#: MIL-STD-1474E Table C-1, converted to CSV, as PANAM ships it (package data).
MIL_STD_1474E_TABLE_C1_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                                           'mil_std_1474e_table_c1_full.csv')


def load_mil_std_1474e_table_c1(filename):
    """
    Load MIL-STD-1474E Table C-1 data from CSV.

    Args:
        filename: CSV file path containing the converted Table C-1 data;
            :data:`MIL_STD_1474E_TABLE_C1_FILE` is the one PANAM ships.

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


#: Input band centers within this factor (half a one-third octave) of the
#: input spectrum's ends still cover a MIL-STD-1474E table band.
_MIL_STD_BAND_TOLERANCE = 2.0 ** (1.0 / 6.0)


def _mil_std_band_distance(exceedance_db, distances_m):
    """Nondetectability distance of one band from its exceedances over the columns.

    exceedance_db[j] is the band level at column j's measurement distance
    minus that column's limit (-inf where the table lists no limit); columns
    ascend in distance.  The band is detectable at a column when its
    exceedance is positive.  Returns (distance, lower, upper, status): the
    distance beyond which it is nondetectable at every column, with the
    interval known to contain it.
    """
    detectable = np.where(exceedance_db > 0.0)[0]
    if detectable.size == 0:
        return distances_m[0], 0.0, distances_m[0], 'below_first'
    j = int(detectable[-1])
    if j == distances_m.size - 1:
        return distances_m[j], distances_m[j], np.inf, 'above_last'
    if not np.isfinite(exceedance_db[j + 1]):
        return distances_m[j + 1], distances_m[j], distances_m[j + 1], 'before_unlisted'
    e1, e2 = exceedance_db[j], exceedance_db[j + 1]
    lg1, lg2 = np.log10(distances_m[j]), np.log10(distances_m[j + 1])
    distance = 10.0 ** (lg1 + e1 / (e1 - e2) * (lg2 - lg1))
    return distance, distance, distance, 'interpolated'


def mil_std_1474e_nondetectability_distance(
        band_centers_hz,
        band_levels_db,
        mil_std_table,
        spectrum_distance_m=10.0):
    """
    Compute MIL-STD-1474E Table C-1 nondetectability distance from a third-octave spectrum.

    Table C-I gives, for each nondetectability distance (column), the band
    limits at that column's measurement distance (2, 10 or 30 m).  The
    spectrum is normalized to 10 m and taken to each column's measurement
    distance by spherical spreading, so its exceedance over a column is
    ``L10 + 20 lg(10 / m) - limit``.  C.5.1.2: a column is met when no band
    exceeds it.  A band's distance is where its exceedance, interpolated
    linearly in lg distance between adjacent columns (across the boundary
    between measurement-distance groups as well, the shift to a common
    distance being the same spreading), falls to zero for the last time --
    the distance beyond which the band meets every column.  The overall
    distance is the largest over the bands.

    Bounds: a band that meets every column is nondetectable from the first
    column (5 m) on, an upper bound ('below_first').  A band that exceeds the
    last column (6000 m) is detectable beyond it: its distance is reported as
    6000 m, a lower bound ('above_last'), and it sets the overall distance,
    which is then flagged as a lower bound.  Where the table lists no limit
    (NA) the band is taken to meet the column, so a band exceeding its last
    listed column lies between that column and the next ('before_unlisted';
    the next column is reported, an upper bound).  Table bands more than half
    a band outside the input spectrum, or whose level is NaN, are not
    evaluated ('outside_spectrum', 'no_level') -- they are not extrapolated.

    Args:
        band_centers_hz: third-octave center frequencies for the spectrum,
            any order (sorted here), no duplicates.
        band_levels_db: third-octave levels (dB) at spectrum_distance_m.
        mil_std_table: dict from load_mil_std_1474e_table_c1(...) or CSV filename.
        spectrum_distance_m: distance corresponding to band_levels_db (m).

    Returns:
        dict with:
            overall_nondetectability_distance_m: largest band distance
            overall_bound: 'lower', 'upper' or None (interpolated), from the
                trigger band
            overall_distance_lower_m, overall_distance_upper_m: interval
                containing the overall distance
            trigger_frequency_hz: band frequency that sets the overall distance
            band_frequency_hz: MIL-STD band frequencies used in evaluation
            band_nondetectability_distance_m: per-band distance (NaN when
                not evaluated)
            band_distance_lower_m, band_distance_upper_m: per-band interval
            band_status: per-band status strings ('interpolated',
                'below_first', 'above_last', 'before_unlisted',
                'outside_spectrum', 'no_level')
            band_valid: per-band boolean, True only for interpolated values
            band_level_db_10m: input spectrum mapped to MIL-STD bands and normalized to 10 m
    """
    if isinstance(mil_std_table, (str, os.PathLike)):
        table = load_mil_std_1474e_table_c1(mil_std_table)
    else:
        table = mil_std_table

    band_centers_hz = np.asarray(band_centers_hz, dtype=float).ravel()
    band_levels_db = np.asarray(band_levels_db, dtype=float).ravel()

    table_band_freq_hz = np.asarray(table['band_freq_hz'], dtype=float)
    distance_columns_m = np.asarray(table['distance_columns_m'], dtype=float)
    measurement_distance_m = np.asarray(table['measurement_distance_m'], dtype=float)
    limits_db = np.asarray(table['limits_db'], dtype=float)

    if band_centers_hz.size == 0 or band_levels_db.size == 0:
        raise ValueError('Input spectrum is empty')
    if band_centers_hz.size != band_levels_db.size:
        raise ValueError('band_centers_hz and band_levels_db differ in length')
    if not np.all(np.isfinite(band_centers_hz) & (band_centers_hz > 0)):
        raise ValueError('band centers must be positive and finite')
    order = np.argsort(band_centers_hz, kind='stable')
    band_centers_hz, band_levels_db = band_centers_hz[order], band_levels_db[order]
    if np.any(np.diff(band_centers_hz) == 0):
        raise ValueError('duplicate band centers')

    columns = np.argsort(distance_columns_m, kind='stable')
    distance_columns_m = distance_columns_m[columns]
    measurement_distance_m = measurement_distance_m[columns]
    limits_db = limits_db[:, columns]

    covered = ((table_band_freq_hz * _MIL_STD_BAND_TOLERANCE >= band_centers_hz[0])
               & (table_band_freq_hz <= band_centers_hz[-1] * _MIL_STD_BAND_TOLERANCE))
    band_levels_table = np.where(covered, _mil_std_interp_log_frequency(
        band_centers_hz,
        band_levels_db,
        table_band_freq_hz,
    ), np.nan)
    band_levels_10m = band_levels_table + 20.0 * np.log10(spectrum_distance_m / 10.0)

    n_bands = table_band_freq_hz.size
    band_distances = np.full(n_bands, np.nan)
    band_lower = np.full(n_bands, np.nan)
    band_upper = np.full(n_bands, np.nan)
    band_status = np.empty(n_bands, dtype=object)
    band_last_exceedance = np.full(n_bands, -np.inf)

    for i in range(n_bands):
        if not covered[i]:
            band_status[i] = 'outside_spectrum'
            continue
        if np.isnan(band_levels_10m[i]):
            band_status[i] = 'no_level'
            continue
        with np.errstate(invalid='ignore'):
            exceedance = band_levels_10m[i] + 20.0 * np.log10(10.0 / measurement_distance_m) - limits_db[i, :]
        exceedance = np.where(np.isnan(limits_db[i, :]), -np.inf, exceedance)
        band_last_exceedance[i] = exceedance[-1]
        band_distances[i], band_lower[i], band_upper[i], band_status[i] = _mil_std_band_distance(
            exceedance, distance_columns_m)

    band_valid = band_status == 'interpolated'
    evaluated = np.isfinite(band_distances)
    if np.any(evaluated):
        candidates = np.where(evaluated)[0]
        best = band_distances[candidates].max()
        tied = candidates[band_distances[candidates] == best]
        # Several bands beyond the last column: the one exceeding it most.
        trigger_idx = int(tied[np.argmax(band_last_exceedance[tied])]) if best == distance_columns_m[-1] \
            else int(tied[0])
        overall_distance = float(band_distances[trigger_idx])
        trigger_frequency = float(table_band_freq_hz[trigger_idx])
        overall_bound = {'interpolated': None, 'above_last': 'lower'}.get(band_status[trigger_idx], 'upper')
        overall_lower = float(np.max(band_lower[evaluated]))
        overall_upper = float(np.max(band_upper[evaluated]))
    else:
        overall_distance = trigger_frequency = overall_lower = overall_upper = np.nan
        overall_bound = None

    return {
        'overall_nondetectability_distance_m': overall_distance,
        'overall_bound': overall_bound,
        'overall_distance_lower_m': overall_lower,
        'overall_distance_upper_m': overall_upper,
        'trigger_frequency_hz': trigger_frequency,
        'band_frequency_hz': table_band_freq_hz,
        'band_nondetectability_distance_m': band_distances,
        'band_distance_lower_m': band_lower,
        'band_distance_upper_m': band_upper,
        'band_status': band_status,
        'band_valid': band_valid,
        'band_level_db_10m': band_levels_10m,
    }
