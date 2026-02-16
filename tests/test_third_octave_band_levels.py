import numpy as np

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
