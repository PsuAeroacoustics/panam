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


def test_a_gated_spectrum_keeps_its_energy():
    """Bins zeroed by the ambient gate leave a zero running-median floor; a peak
    above a zero floor is not a tone, and filing it as one invented energy."""
    rng = np.random.default_rng(3)
    f = np.arange(0.0, 400.0, 0.5)
    psd = rng.exponential(1.0, (f.size, 50))
    psd[rng.random(psd.shape) < 0.6] = 0.0
    lower, upper = fa.third_octave_band_edges(np.array([50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0]))
    tone_aware = fa.tone_aware_band_power(psd, f, lower, upper)
    brick = np.array([psd[(f >= lo) & (f < hi)].sum(axis=0) * 0.5 for lo, hi in zip(lower, upper)])
    # Noise peaks taken for tones read about +0.1 dB on ungated noise too; counting
    # the gated zeros in the floor made it +1 dB here.
    assert np.all(np.abs(10 * np.log10(tone_aware.sum(axis=1) / brick.sum(axis=1))) < 0.3)


def _tone_psd(f0, fs=1024.0, n=1024, noise=0.0, seed=0):
    t = np.arange(n) / fs
    x = np.sqrt(2.0) * np.sin(2 * np.pi * f0 * t)
    if noise:
        x = x + noise * np.random.default_rng(seed).standard_normal(n)
    return scipy.signal.periodogram(x, fs, window='hann', scaling='density')


@pytest.mark.parametrize('side', [-0.3, 0.3])
def test_a_tone_a_fraction_of_a_bin_from_a_band_edge_is_filed_in_its_own_band(side):
    """What the function is for: summing whole bins hands such a tone to the
    neighboring band."""
    lower, upper = fa.third_octave_band_edges(np.array([80.0, 100.0, 125.0]))
    edge = upper[0]                                     # 80/100 Hz edge, 1 Hz bins
    f, psd = _tone_psd(edge + side)
    band = fa.tone_aware_band_power(psd[:, None], f, lower, upper)[:, 0]
    expected = [1.0, 0.0, 0.0] if side < 0 else [0.0, 1.0, 0.0]
    np.testing.assert_allclose(band, expected, atol=2e-3)


def test_a_doppler_shifted_tone_is_filed_at_its_emitted_frequency():
    """A tone received at D f_e lands in f_e's band, whichever band D f_e is in."""
    lower, upper = fa.third_octave_band_edges(np.array([80.0, 100.0, 125.0]))
    doppler = 1.25
    f, psd = _tone_psd(100.3 * doppler)                 # received in the 125 Hz band
    band = fa.tone_aware_band_power(psd[:, None], f, lower, upper, doppler=np.array([doppler]))[:, 0]
    np.testing.assert_allclose(band, [0.0, 1.0, 0.0], atol=2e-3)


def test_white_noise_matches_a_brick_wall_sum():
    rng = np.random.default_rng(8)
    f = np.arange(0.0, 400.0, 0.5)
    psd = rng.exponential(1.0, (f.size, 200))
    lower, upper = fa.third_octave_band_edges(np.array([50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0]))
    tone_aware = fa.tone_aware_band_power(psd, f, lower, upper).sum(axis=1)
    brick = np.array([psd[(f >= lo) & (f < hi)].sum() * 0.5 for lo, hi in zip(lower, upper)])
    assert np.all(np.abs(10 * np.log10(tone_aware / brick)) < 0.3)
