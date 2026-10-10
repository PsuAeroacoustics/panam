import numpy as np
import pytest

import flight_acoustics as fa


def test_third_octave_band_levels_white_noise():
    fs = 40000  # Sampling frequency
    duration = 20.0  # seconds
    t = np.arange(0, duration, 1 / fs)
    rng = np.random.default_rng(12345)
    white_noise = rng.normal(0, 1, len(t))

    band_centers, band_levels = fa.third_octave_band_levels(white_noise, fs)

    assert band_centers.size > 0
    assert band_levels.size == band_centers.size
    assert np.all(np.isfinite(band_levels))

    # White noise should increase ~3 dB per octave band.
    # For third-octave bands, the slope vs log2(f) should be ~3 dB/octave.
    x = np.log2(band_centers)
    coeffs = np.polyfit(x, band_levels, 1)
    slope_db_per_octave = coeffs[0]
    assert 2.0 < slope_db_per_octave < 4.0


def test_third_octave_band_levels_1khz_calibrator_level():
    fs = 48000
    duration = 10.0
    f_tone = 1000.0
    pref = 2.0e-5
    target_spl_db = 94.1

    # Build a pure tone with known RMS pressure corresponding to target SPL.
    p_rms = pref * 10.0 ** (target_spl_db / 20.0)
    p_amp = np.sqrt(2.0) * p_rms
    t = np.arange(0, duration, 1 / fs)
    tone = p_amp * np.sin(2.0 * np.pi * f_tone * t)

    band_centers, band_levels = fa.third_octave_band_levels(tone, fs)

    center_index = np.argmin(np.abs(band_centers - f_tone))
    center_frequency = band_centers[center_index]
    measured_level_db = band_levels[center_index]

    # The nearest third-octave center should be 1000 Hz and match calibrator level.
    assert abs(center_frequency - 1000.0) < 1.0e-12
    assert np.isfinite(measured_level_db)
    assert np.isclose(measured_level_db, target_spl_db, atol=0.01)

    # Energy should be concentrated in the 1 kHz third-octave band.
    assert measured_level_db - np.max(np.delete(band_levels, center_index)) > 20.0


def test_third_octave_band_levels_nominal_limits_keep_their_bands():
    """The default 20 Hz..20 kHz limits give all 31 bands from 20 Hz to 20 kHz.

    The exact centers of those two bands are 19.69 and 20159 Hz, so comparing
    them directly against the nominal limits dropped both end bands.
    """
    fs = 48000
    noise = np.random.default_rng(3).standard_normal(4 * fs)
    band_centers, band_levels = fa.third_octave_band_levels(noise, fs)
    assert band_centers.size == 31
    assert band_centers[0] == pytest.approx(20.0, rel=0.02)
    assert band_centers[-1] == pytest.approx(20000.0, rel=0.02)
    assert np.all(np.isfinite(band_levels))


def test_third_octave_band_levels_drops_bands_past_nyquist():
    """A band reaching past Nyquist would be only partly filled, so it is left out."""
    fs = 40000
    noise = np.random.default_rng(4).standard_normal(4 * fs)
    band_centers, _ = fa.third_octave_band_levels(noise, fs)
    assert np.all(band_centers * 2.0 ** (1.0 / 6.0) <= fs / 2.0)
    assert band_centers[-1] == pytest.approx(16000.0)


def test_third_octave_band_levels_off_bin_tone_does_not_leak():
    """A 50.5 Hz tone (between 1 Hz bins) stays in its band.

    A rectangular window leaks it with a 1/k**2 skirt: the 1 kHz band read
    66 dB below the tone band, and the 10 kHz band 92 dB below it.
    """
    fs = 25600
    t = np.arange(fs) / fs
    tone = np.sqrt(2.0) * np.sin(2.0 * np.pi * 50.5 * t)    # 1 Pa rms, 93.98 dB
    band_centers, band_levels = fa.third_octave_band_levels(tone, fs)
    k = int(np.argmin(np.abs(band_centers - 50.0)))
    assert band_levels[k] == pytest.approx(10.0 * np.log10(np.mean(tone ** 2) / fa.P_REF ** 2), abs=0.05)
    assert np.all(band_levels[band_centers >= 400.0] < band_levels[k] - 120.0)
