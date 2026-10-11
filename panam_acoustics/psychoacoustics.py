"""Stationary loudness (ISO 532-1:2017, Zwicker) and sharpness (DIN 45692:2009).

``loudness_zwicker`` implements the stationary method of ISO 532-1:2017 from
28 one-third octave band levels, 25 Hz to 12.5 kHz, as the standard's Annex A
program does (``f_loudness_from_levels`` with the stationary method).  The
tables are the standard's.  The program stores them as single-precision
literals and steps the critical-band rate in single-precision 0.1 Bark, so this
module does the same; that is what makes it reproduce the program's results
(Annex B.2) to their printed precision.

``sharpness_din`` implements DIN 45692:2009 from the specific loudness.
"""

import numpy as np

#: Nominal centers of the 28 one-third octave bands ISO 532-1 takes (Hz).
ISO_532_1_BAND_CENTERS_HZ = np.array([
    25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0,
    630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0,
    10000.0, 12500.0])

#: Critical-band rate (Bark) of each specific-loudness value: 0.1 to 24.0 in 0.1 steps.
BARK_AXIS = np.round(np.arange(1, 241) * 0.1, 1)


def _single(values):
    """The program's single-precision table literals, as doubles."""
    return np.asarray(values, dtype=np.float32).astype(float)


# ISO 532-1:2017, Annex A.  Upper limits of the level ranges (RAP) of the
# equal-loudness correction for the 11 bands up to 250 Hz.
_RAP = _single([45.0, 55.0, 65.0, 71.0, 80.0, 90.0, 100.0, 120.0])
# Level corrections in those ranges (DLL), dB.
_DLL = _single([
    [-32, -24, -16, -10, -5, 0, -7, -3, 0, -2, 0],
    [-29, -22, -15, -10, -4, 0, -7, -2, 0, -2, 0],
    [-27, -19, -14, -9, -4, 0, -6, -2, 0, -2, 0],
    [-25, -17, -12, -9, -3, 0, -5, -2, 0, -2, 0],
    [-23, -16, -11, -7, -3, 0, -4, -1, 0, -1, 0],
    [-20, -14, -10, -6, -3, 0, -4, -1, 0, -1, 0],
    [-18, -12, -9, -6, -2, 0, -3, -1, 0, -1, 0],
    [-15, -10, -8, -4, -2, 0, -3, -1, 0, -1, 0]])
# Critical-band level at the threshold in quiet (LTQ), dB.
_LTQ = _single([30, 18, 12, 8, 7, 6, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
# Transmission through the outer ear (A0), dB.
_A0 = _single([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, -1.6, -3.2, -5.4, -5.6, -4.0, -1.5, 2.0, 5.0, 12.0])
# Diffuse minus free field level (DDF), dB.
_DDF = _single([0, 0, 0.5, 0.9, 1.2, 1.6, 2.3, 2.8, 3.0, 2.0, 0, -1.4, -2.0, -1.9, -1.0, 0.5, 3.0,
                4.0, 4.3, 4.0])
# One-third octave to critical-band level (DCB), dB.
_DCB = _single([-0.25, -0.6, -0.8, -0.8, -0.5, 0, 0.5, 1.1, 1.5, 1.7, 1.8, 1.8, 1.7, 1.6, 1.4, 1.2,
                0.8, 0.5, 0, -0.5])
# Upper limits of the approximated critical bands (ZUP), Bark.
_ZUP = _single([0.9, 1.8, 2.8, 3.5, 4.4, 5.4, 6.6, 7.9, 9.2, 10.6, 12.3, 13.8, 15.2, 16.7, 18.1,
                19.3, 20.6, 21.8, 22.7, 23.6, 24.0])
# Specific-loudness ranges (RNS), sone/Bark, and the upper-slope steepness in
# each (USL), sone/Bark per Bark, by critical band.
_RNS = _single([21.5, 18.0, 15.1, 11.5, 9.0, 6.1, 4.4, 3.1, 2.13, 1.36, 0.82, 0.42, 0.30, 0.22,
                0.15, 0.10, 0.035, 0.0])
_USL = _single([
    [13.00, 8.20, 6.30, 5.50, 5.50, 5.50, 5.50, 5.50],
    [9.00, 7.50, 6.00, 5.10, 4.50, 4.50, 4.50, 4.50],
    [7.80, 6.70, 5.60, 4.90, 4.40, 3.90, 3.90, 3.90],
    [6.20, 5.40, 4.60, 4.00, 3.50, 3.20, 3.20, 3.20],
    [4.50, 3.80, 3.60, 3.20, 2.90, 2.70, 2.70, 2.70],
    [3.70, 3.00, 2.80, 2.35, 2.20, 2.20, 2.20, 2.20],
    [2.90, 2.30, 2.10, 1.90, 1.80, 1.70, 1.70, 1.70],
    [2.40, 1.70, 1.50, 1.35, 1.30, 1.30, 1.30, 1.30],
    [1.95, 1.45, 1.30, 1.15, 1.10, 1.10, 1.10, 1.10],
    [1.50, 1.20, 0.94, 0.86, 0.82, 0.82, 0.82, 0.82],
    [0.72, 0.67, 0.64, 0.63, 0.62, 0.62, 0.62, 0.62],
    [0.59, 0.53, 0.51, 0.50, 0.42, 0.42, 0.42, 0.42],
    [0.40, 0.33, 0.26, 0.24, 0.22, 0.22, 0.22, 0.22],
    [0.27, 0.21, 0.20, 0.18, 0.17, 0.17, 0.17, 0.17],
    [0.16, 0.15, 0.14, 0.12, 0.11, 0.11, 0.11, 0.11],
    [0.12, 0.11, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08],
    [0.09, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06, 0.05],
    [0.06, 0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02]])
_STEP = float(np.float32(0.1))
_ZUP_MARGIN = float(np.float32(0.0001))
_MP1_SCALE = float(np.float32(0.0635))
_KORRY = (float(np.float32(0.4)), float(np.float32(0.32)))


def _core_loudness(levels_db, diffuse):
    """Core loudness (sone/Bark) of the 20 critical bands plus a zero 21st."""
    intensities = np.empty(11)
    for i in range(11):
        j = 0
        while levels_db[i] > _RAP[j] - _DLL[j, i] and j < 7:
            j += 1
        intensities[i] = 10.0 ** ((levels_db[i] + _DLL[j, i]) / 10.0)
    lcb = np.array([intensities[0:6].sum(), intensities[6:9].sum(), intensities[9:11].sum()])
    with np.errstate(divide='ignore'):
        # The program leaves a zero intensity as a level of 0 dB, below every threshold.
        lcb = np.where(lcb > 0.0, 10.0 * np.log10(lcb), 0.0)

    core = np.zeros(21)
    for i in range(20):
        level = (lcb[i] if i < 3 else levels_db[i + 8]) - _A0[i]
        if diffuse:
            level += _DDF[i]
        if level > _LTQ[i]:
            level -= _DCB[i]
            mp1 = _MP1_SCALE * 10.0 ** (0.025 * _LTQ[i])
            mp2 = (0.75 + 0.25 * 10.0 ** (0.1 * (level - _LTQ[i]))) ** 0.25 - 1.0
            core[i] = max(mp1 * mp2, 0.0)
    # Run of the threshold in quiet within the lowest critical band.
    correction = _KORRY[0] + _KORRY[1] * core[0] ** 0.2
    if correction < 1.0:
        core[0] *= correction
    return core


def _slopes(core):
    """Total loudness (sone) and the 240 specific loudness values (sone/Bark)."""
    specific = np.zeros(BARK_AXIS.size)
    total = 0.0
    n1 = 0.0
    z = _STEP
    z1 = 0.0
    j = 0
    k = 0
    for i in range(21):
        core_i = core[i]
        zup = _ZUP[i] + _ZUP_MARGIN
        ig = min(i - 1, 7)
        next_band = False
        while not next_band:
            if n1 > core_i:
                usl = _USL[j, ig]
                n2 = max(_RNS[j], core_i)
                dz = (n1 - n2) / usl
                z2 = z1 + dz
                if z2 > zup:
                    next_band = True
                    z2 = zup
                    dz = z2 - z1
                    n2 = n1 - dz * usl
                total += dz * (n1 + n2) / 2.0
                zk = z
                while zk <= z2:
                    specific[k] = n1 - (zk - z1) * usl
                    k += 1
                    zk += _STEP
                z = zk
            else:
                if n1 < core_i:
                    j = 0
                    while j < 18 and _RNS[j] >= core_i:
                        j += 1
                next_band = True
                z2 = zup
                n2 = core_i
                total += n2 * (z2 - z1)
                zk = z
                while zk <= z2:
                    specific[k] = n2
                    k += 1
                    zk += _STEP
                z = zk
            while n2 <= _RNS[j] and j < 17:
                j += 1
            j = min(j, 17)
            z1 = z2
            n1 = n2
        total = max(total, 0.0)
    return total, specific


def loudness_level(loudness_sone):
    """Loudness level LN (phon) of a loudness N (sone), as ISO 532-1's program converts it.

    LN = 40 + 10 log2 N for N >= 1 sone, and 40 (N + 0.0005)^0.35 below, not
    less than 3 phon.  NaN stays NaN.
    """
    loudness_sone = np.asarray(loudness_sone, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        above = 40.0 + 10.0 * np.log2(loudness_sone)
        below = np.maximum(40.0 * np.power(loudness_sone + 0.0005, 0.35), 3.0)
    out = np.where(loudness_sone >= 1.0, above, below)
    out = np.where(np.isnan(loudness_sone), np.nan, out)
    return out[()] if out.ndim == 0 else out


def iso_532_1_levels(frequency_hz, levels_db, missing_12k5=-np.inf):
    """Pick the 28 ISO 532-1 bands out of a one-third octave spectrum.

    PANAM's spheres and hemispheres carry 31 bands from 10 Hz to 10 kHz, and
    :func:`loudness_zwicker` takes 28 from 25 Hz to 12.5 kHz.  Each ISO band is
    matched to the band of ``frequency_hz`` within 1/12 octave of its nominal
    center.  When the spectrum has no 12.5 kHz band, that band is filled with
    ``missing_12k5``.  The default, -inf (no energy), lets the loudness be
    computed but understates it by whatever the 12.5 kHz band holds.  NaN
    (no data) makes the loudness of every spectrum NaN.

    Args:
        frequency_hz: (Nf,) one-third octave band centers (Hz), exact or nominal.
        levels_db: (..., Nf) band levels (dB re 20 uPa).
        missing_12k5: level (dB) used for 12.5 kHz when ``frequency_hz`` lacks it.

    Returns:
        (..., 28) levels (dB) in the bands of :data:`ISO_532_1_BAND_CENTERS_HZ`.

    Raises:
        ValueError: a band from 25 Hz to 10 kHz is missing, or two bands of
            ``frequency_hz`` match one ISO band.
    """
    frequency_hz = np.asarray(frequency_hz, dtype=float).ravel()
    levels_db = np.asarray(levels_db, dtype=float)
    if levels_db.ndim < 1 or levels_db.shape[-1] != frequency_hz.size:
        raise ValueError(f'levels_db must end in the {frequency_hz.size} bands of frequency_hz, '
                         f'not shape {levels_db.shape}')
    with np.errstate(divide='ignore', invalid='ignore'):
        octaves = np.abs(np.log2(frequency_hz[None, :] / ISO_532_1_BAND_CENTERS_HZ[:, None]))
    match = octaves < 1.0 / 12.0
    out = np.empty(levels_db.shape[:-1] + (ISO_532_1_BAND_CENTERS_HZ.size,))
    for band, center in enumerate(ISO_532_1_BAND_CENTERS_HZ):
        index = np.flatnonzero(match[band])
        if index.size > 1:
            raise ValueError(f'two bands of frequency_hz match the {center:g} Hz band')
        if index.size == 1:
            out[..., band] = levels_db[..., index[0]]
        elif band == ISO_532_1_BAND_CENTERS_HZ.size - 1:
            out[..., band] = missing_12k5
        else:
            raise ValueError(f'frequency_hz has no {center:g} Hz band: ISO 532-1 needs every '
                             'band from 25 Hz to 10 kHz')
    return out


def loudness_zwicker(third_octave_levels_db, field='free'):
    """Stationary loudness by ISO 532-1:2017 (Zwicker) from one-third octave band levels.

    Args:
        third_octave_levels_db: (..., 28) unweighted one-third octave band levels
            (dB re 20 uPa) in the bands of :data:`ISO_532_1_BAND_CENTERS_HZ`, 25 Hz
            to 12.5 kHz, ascending.  -inf is a band with no energy.  A spectrum
            with a NaN band (no data) gives NaN results.
        field: 'free' (frontal incidence) or 'diffuse'.

    Returns:
        (N, N_specific, LN):
            N: (...) total loudness (sone).
            N_specific: (..., 240) specific loudness (sone/Bark) at :data:`BARK_AXIS`.
            LN: (...) loudness level (phon), from :func:`loudness_level`.
    """
    field = str(field).lower()
    if field not in ('free', 'diffuse'):
        raise ValueError(f"field must be 'free' or 'diffuse', not {field!r}")
    levels = np.asarray(third_octave_levels_db, dtype=float)
    if levels.ndim < 1 or levels.shape[-1] != ISO_532_1_BAND_CENTERS_HZ.size:
        raise ValueError('third_octave_levels_db must end in the 28 bands from 25 Hz to 12.5 kHz, '
                         f'not shape {levels.shape}')
    if np.any(levels == np.inf):
        raise ValueError('third_octave_levels_db must not hold +inf')
    flat = levels.reshape(-1, levels.shape[-1])
    total = np.full(flat.shape[0], np.nan)
    specific = np.full((flat.shape[0], BARK_AXIS.size), np.nan)
    for n, spectrum in enumerate(flat):
        if np.any(np.isnan(spectrum)):
            continue
        total[n], specific[n] = _slopes(_core_loudness(spectrum, field == 'diffuse'))
    total = total.reshape(levels.shape[:-1])
    specific = specific.reshape(levels.shape[:-1] + (BARK_AXIS.size,))
    return total[()] if total.ndim == 0 else total, specific, loudness_level(total)


def _din_45692_weighting(bark):
    """DIN 45692 weighting g(z): 1 up to 15.8 Bark, rising exponentially above."""
    bark = np.asarray(bark, dtype=float)
    return np.where(bark <= 15.8, 1.0, 0.85 + 0.15 * np.exp(0.42 * (bark - 15.8)))


def sharpness_din(loudness_sone, specific_loudness):
    """Sharpness S (acum) by DIN 45692:2009 from the loudness and specific loudness.

    S = 0.11 * sum(N'(z) g(z) z dz) / N over the 240 values of :data:`BARK_AXIS`
    (dz = 0.1 Bark), with N the total loudness of ISO 532-1 and g the DIN 45692
    weighting.  A 1 kHz critical-band-wide noise at 60 dB is 1 acum.

    Args:
        loudness_sone: (...) total loudness N (sone), from :func:`loudness_zwicker`.
        specific_loudness: (..., 240) specific loudness N' (sone/Bark), from
            :func:`loudness_zwicker`.

    Returns:
        (...) sharpness (acum); NaN where the loudness is zero (undefined) or NaN.
    """
    loudness_sone = np.asarray(loudness_sone, dtype=float)
    specific_loudness = np.asarray(specific_loudness, dtype=float)
    if specific_loudness.shape[-1] != BARK_AXIS.size:
        raise ValueError(f'specific_loudness must end in the {BARK_AXIS.size} values of BARK_AXIS')
    weighted = 0.1 * np.sum(specific_loudness * _din_45692_weighting(BARK_AXIS) * BARK_AXIS, axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(loudness_sone > 0.0, 0.11 * weighted / loudness_sone, np.nan)
    return out[()] if out.ndim == 0 else out
