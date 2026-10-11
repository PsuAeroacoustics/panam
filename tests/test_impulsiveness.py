import numpy as np
import pytest

from panam_acoustics.impulsiveness import (
    bvi_harmonic_ratio, crest_factor, frame_signal, kurtosis, pulse_peak_to_rms)

FS = 10000.0
BPF = 20.0                    # Hz; 500 samples per blade passage


def sine(freq=BPF, duration=2.0, amplitude=3.0, offset=0.0):
    # A cosine: over whole periods it is orthogonal to a line, so a linear
    # detrend leaves it unchanged.
    t = np.arange(int(duration * FS)) / FS
    return offset + amplitude * np.cos(2 * np.pi * freq * t)


def impulse_train(period=500, duration=2.0):
    p = np.zeros(int(duration * FS))
    p[period // 2::period] = 1.0
    return p


def test_frames_are_consecutive_and_centered():
    times, frames = frame_signal(np.arange(10.0), 10.0, 0.4)
    assert frames.shape == (2, 4)
    np.testing.assert_array_equal(frames[1], [4, 5, 6, 7])
    np.testing.assert_allclose(times, [0.15, 0.55])
    times, frames = frame_signal(np.arange(10.0), 10.0, 0.4, hop_time=0.2)
    assert frames.shape == (4, 4)


def test_sine_crest_factor_and_kurtosis():
    # Whole periods per 0.25 s frame; the mean is removed by detrending.
    _, crest = crest_factor(sine(offset=5.0), FS, 0.25)
    _, kurt = kurtosis(sine(offset=5.0), FS, 0.25)
    assert crest.shape == (8,)
    np.testing.assert_allclose(crest, np.sqrt(2.0), rtol=1e-6)
    np.testing.assert_allclose(kurt, 1.5, rtol=1e-6)


def test_gaussian_noise_kurtosis_is_three():
    rng = np.random.default_rng(1)
    _, kurt = kurtosis(rng.standard_normal((4, 400000)), FS, 10.0)
    np.testing.assert_allclose(kurt, 3.0, atol=0.05)


def test_impulse_train_is_far_more_impulsive_than_a_sine():
    T = 500
    _, crest = crest_factor(impulse_train(T), FS, 0.25)
    _, kurt = kurtosis(impulse_train(T), FS, 0.25)
    # One unit pulse per T samples, mean removed: exact moments of the train.
    peak, m2 = 1.0 - 1.0 / T, (1.0 - 1.0 / T) / T
    m4 = (1.0 / T) * (1.0 - 1.0 / T) ** 4 + (1.0 - 1.0 / T) * T ** -4.0
    np.testing.assert_allclose(crest, peak / np.sqrt(m2), rtol=1e-9)
    np.testing.assert_allclose(kurt, m4 / m2 ** 2, rtol=1e-9)
    assert np.all(crest > 20.0) and np.all(kurt > 400.0)


def test_linear_detrend_removes_a_ramp():
    ramp = np.linspace(0.0, 50.0, int(2 * FS))
    _, constant = crest_factor(sine() + ramp, FS, 0.25)
    _, linear = crest_factor(sine() + ramp, FS, 0.25, detrend='linear')
    np.testing.assert_allclose(linear, np.sqrt(2.0), rtol=2e-3)
    assert np.all(constant > 2.0)          # the ramp left in each frame


def test_nan_samples_from_dedopplerization_blank_their_frames():
    p = sine()
    p[:100] = np.nan                    # dedopplerize: received before the record
    for metric in (crest_factor, kurtosis):
        _, value = metric(p, FS, 0.25)
        assert np.isnan(value[0])
        np.testing.assert_allclose(value[1:], np.sqrt(2.0) if metric is crest_factor else 1.5, rtol=1e-6)
    _, ratio = pulse_peak_to_rms(p, FS, 0.25, band_hz=(5.5 * BPF, 40.5 * BPF))
    assert np.isnan(ratio[0]) and np.all(np.isfinite(ratio[1:]))


def test_zero_signal_is_nan_not_an_error():
    _, crest = crest_factor(np.zeros(1000), FS, 0.05)
    assert np.all(np.isnan(crest))


def test_pulse_peak_to_rms_sees_the_pulses_not_the_loading_tone():
    # A band wider than the harmonics, so that their gain is 1; the first
    # and last frames hold the filter's start-up transient.
    band = (2.5 * BPF, 60.0 * BPF)
    loading = sine(amplitude=1.0)
    _, tone_only = pulse_peak_to_rms(loading, FS, 0.25, band_hz=band)
    # A smooth pulse train confined to the BVI harmonics, added to the tone.
    t = np.arange(loading.size) / FS
    pulses = sum(0.1 * np.cos(2 * np.pi * k * BPF * t) for k in range(6, 41))
    _, with_pulses = pulse_peak_to_rms(loading + pulses, FS, 0.25, band_hz=band)
    assert np.all(tone_only[1:-1] < 0.01)
    rms = np.sqrt(np.mean((loading + pulses) ** 2))
    np.testing.assert_allclose(with_pulses[1:-1], 3.5 / rms, rtol=0.01)


def test_pulse_band_must_be_below_nyquist():
    with pytest.raises(ValueError):
        pulse_peak_to_rms(sine(), FS, 0.25, band_hz=(100.0, FS))


def flat_spectrum(df=1.0, fmax=1000.0, level=60.0):
    freqs = np.arange(df, fmax + df / 2, df)
    return freqs, np.full(freqs.size, level)


def test_harmonic_ratio_of_a_flat_spectrum_is_the_bandwidth_ratio():
    freqs, psd = flat_spectrum()
    # Harmonics 1-5: [10, 110) Hz; 6-40: [110, 810) Hz.
    ratio = bvi_harmonic_ratio(psd, freqs, BPF)
    assert ratio == pytest.approx(10 * np.log10(700.0 / 100.0), abs=1e-9)


def test_harmonic_ratio_of_tones_is_their_power_ratio_and_maps_over_a_grid():
    freqs, _ = flat_spectrum()
    psd = np.full((freqs.size, 2, 3), -np.inf)
    psd[freqs == 40.0] = 80.0           # 2nd harmonic
    psd[freqs == 200.0] = 70.0          # 10th harmonic
    psd[freqs == 400.0, 1, :] = 70.0    # 20th harmonic, second row only
    ratio = bvi_harmonic_ratio(psd, freqs, BPF)
    assert ratio.shape == (2, 3)
    np.testing.assert_allclose(ratio[0], -10.0)
    np.testing.assert_allclose(ratio[1], -10.0 + 10 * np.log10(2.0))


def test_harmonic_ratio_missing_and_empty_bins():
    freqs, psd = flat_spectrum()
    psd = np.stack([psd] * 4, axis=1)
    psd[(freqs >= 110) & (freqs < 810), 0] = -np.inf      # no BVI energy
    psd[(freqs >= 10) & (freqs < 110), 1] = -np.inf       # no low-harmonic energy
    psd[:, 2] = -np.inf                                    # no energy at all
    psd[300, 3] = np.nan                                   # no data in one bin
    ratio = bvi_harmonic_ratio(psd, freqs, BPF)
    assert ratio[0] == -np.inf and ratio[1] == np.inf
    assert np.isnan(ratio[2]) and np.isnan(ratio[3])


def test_harmonic_ratio_refuses_bands_outside_the_spectrum():
    freqs, psd = flat_spectrum(fmax=500.0)
    with pytest.raises(ValueError, match='outside'):
        bvi_harmonic_ratio(psd, freqs, BPF)
    assert np.isfinite(bvi_harmonic_ratio(psd, freqs, BPF, high_harmonics=(6, 20)))
