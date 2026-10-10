"""hann_power and tone_aware_band_power against a directly computed Hann window."""

import numpy as np
import pytest
import scipy.signal

import flight_acoustics as fa


@pytest.mark.parametrize('offset', [0.0, 0.25, 0.5, 1.0, -1.0, 1.5, 2.0, 3.0])
def test_hann_power_matches_the_window_spectrum(offset):
    n, pad = 256, 64
    window = scipy.signal.get_window('hann', n)
    spectrum = np.abs(np.fft.fft(window, n * pad)) ** 2
    shift = int(round(abs(offset) * pad))
    # The formula is the continuous window's; 256 points differ by about 1e-9.
    assert fa.hann_power(offset) == pytest.approx(spectrum[shift] / spectrum[0], abs=1e-8)


def test_hann_power_is_continuous_at_one_bin():
    assert fa.hann_power(1.0) == pytest.approx(fa.hann_power(1.0 - 1e-6), rel=1e-5)


@pytest.mark.parametrize('bin_offset', [0.0, 0.3])
def test_a_tone_is_filed_whole_on_or_off_a_bin_center(bin_offset):
    """A 1 Pa^2 tone: the band holding it gets 1 Pa^2, nothing left over around it."""
    fs, n = 1024.0, 1024
    df = fs / n
    f0 = (100 + bin_offset) * df
    t = np.arange(n) / fs
    x = np.sqrt(2.0) * np.sin(2 * np.pi * f0 * t)
    f, psd = scipy.signal.periodogram(x, fs, window='hann', scaling='density')
    band = fa.tone_aware_band_power(psd[:, None], f, [90.0], [112.0])
    assert band[0, 0] == pytest.approx(1.0, rel=1e-3)
