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
