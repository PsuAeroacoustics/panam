"""Doppler-predicted order tracks and clock-offset estimation (order_tracking.py)."""
import numpy as np
import pytest
import scipy.signal as sig

from panam import order_tracking as ot
from panam.vold_kalman_filter import vold_kalman_filter

C = 1125.0          # ft/s
X0, V, H = -1500.0, 150.0, 300.0   # straight level pass: x = X0 + V t at height H, ft and ft/s


def _track(t0=0.0, t1=20.0, n=2001):
    t = np.linspace(t0, t1, n)
    return {'time': t, 'x': X0 + V * t, 'y': np.zeros_like(t), 'z': np.full_like(t, H)}


def _exact_emission(t_r, receiver=(0.0, 0.0, 0.0)):
    """Closed-form emission time of the straight pass, receiver on the y = 0 line."""
    xr, _, zr = receiver
    a = C ** 2 - V ** 2
    b = -2 * (C ** 2 * t_r + (X0 - xr) * V)
    c = C ** 2 * t_r ** 2 - (X0 - xr) ** 2 - (H - zr) ** 2
    return (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)


def _exact_doppler(t_r):
    t_e = _exact_emission(t_r)
    rx, rz = -(X0 + V * t_e), -H
    return 1.0 / (1.0 - V * rx / (np.hypot(rx, rz) * C))


def test_doppler_factor_matches_the_closed_form():
    t_r = np.linspace(2.0, 19.0, 50)
    t_e, factor = ot.doppler_factor(t_r, _track(), [0.0, 0.0, 0.0], C)
    np.testing.assert_allclose(t_e, _exact_emission(t_r), atol=1e-8)
    np.testing.assert_allclose(factor, _exact_doppler(t_r), rtol=1e-9)
    # D = dt_e/dt_r
    np.testing.assert_allclose(factor, np.gradient(_exact_emission(t_r), t_r), rtol=2e-3)
    assert factor[0] > 1.1 and factor[-1] < 0.9


def test_reception_before_the_track_can_reach_the_receiver_is_nan():
    t_e, factor = ot.doppler_factor(np.array([0.5, 10.0]), _track(), [0.0, 0.0, 0.0], C)
    assert np.isnan(t_e[0]) and np.isnan(factor[0])
    assert np.isfinite(t_e[1]) and np.isfinite(factor[1])


def test_observed_track_is_the_doppler_shifted_order_of_the_emitted_speed():
    t_r = np.linspace(2.0, 19.0, 40)
    # rotor speed rising 300 -> 400 rpm over the track: the observed order
    # follows the speed at the emission time, not at the reception time
    track = _track()
    rpm = 300.0 + 5.0 * track['time']
    f = ot.observed_order_track(t_r, track, [0.0, 0.0, 0.0], rpm, 4, C)
    t_e = _exact_emission(t_r)
    np.testing.assert_allclose(f, 4 * (300.0 + 5.0 * t_e) / 60.0 * _exact_doppler(t_r), rtol=1e-8)


def test_track_shapes_units_and_rotor_time():
    t_r = np.linspace(3.0, 18.0, 25)
    track = _track()
    rotor_time = np.linspace(0.0, 20.0, 41)
    speeds = np.column_stack([np.full(41, 20.0), np.full(41, 25.0), np.full(41, 30.0)])  # Hz
    one = ot.observed_order_track(t_r, track, [0, 0, 0], speeds[:, 0], 1, C, rotor_time=rotor_time,
                                  speed_unit='hz')
    assert one.shape == (25,)
    shafts = ot.observed_order_track(t_r, track, [0, 0, 0], speeds, 2, C, rotor_time=rotor_time, speed_unit='Hz')
    assert shafts.shape == (25, 3)
    np.testing.assert_allclose(shafts, 2 * one[:, None] * np.array([1.0, 1.25, 1.5]))
    both = ot.observed_order_track(t_r, track, [0, 0, 0], speeds, [1, 2, 3, 4], C, rotor_time=rotor_time,
                                   speed_unit='hz')
    assert both.shape == (25, 3, 4)
    radians = ot.observed_order_track(t_r, track, [0, 0, 0], 2 * np.pi * speeds, 2, C, rotor_time=rotor_time,
                                      speed_unit='rad/s')
    np.testing.assert_allclose(radians, shafts)
    called = ot.observed_order_track(t_r, track, [0, 0, 0], lambda t: np.full(np.shape(t), 1200.0), 1, C)
    np.testing.assert_allclose(called, one)
    with pytest.raises(ValueError, match='speed_unit'):
        ot.observed_order_track(t_r, track, [0, 0, 0], speeds, 1, C, rotor_time=rotor_time, speed_unit='kt')


# ------------------------------------------------- synthetic flyover tone

FS = 1024.0
F0 = 80.0       # emitted tone, Hz


def _flyover(t_r):
    """Received pressure of a steady F0 tone on the straight pass: phase at the emission time."""
    return np.cos(2 * np.pi * F0 * _exact_emission(t_r))


def test_ridge_of_a_flyover_tone_follows_the_predicted_track():
    t_r = np.arange(int(18 * FS)) / FS + 1.5
    times, peak = ot.ridge_track(_flyover(t_r), FS, (50.0, 110.0), window=0.5, hop=0.25)
    predicted = ot.observed_order_track(times + t_r[0], _track(), [0.0, 0.0, 0.0], F0 * 60.0, 1, C)
    assert np.max(np.abs(peak - predicted)) < 0.3          # 2 Hz bins, refined
    assert np.max(np.abs(peak - F0)) > 10.0                # the shift is not small


def test_ridge_frame_times_are_the_window_centers():
    """A linear chirp's ridge sits at the frequency of each Hann window's center sample."""
    fs, f0, rate = 32.0, 4.0, 2.0                  # Hz, Hz, Hz/s: 1/16 Hz per sample
    t = np.arange(int(4 * fs)) / fs
    x = np.cos(2 * np.pi * (f0 * t + rate * t ** 2 / 2))
    times, peak = ot.ridge_track(x, fs, (1.0, 15.0), window=1.0, hop=1.0 / fs)
    np.testing.assert_allclose(times[:2], [15.5 / fs, 16.5 / fs])
    lag_samples = (peak - (f0 + rate * times)) / rate * fs
    assert np.max(np.abs(lag_samples)) < 0.25


def test_vkf_on_the_predicted_track_recovers_the_received_tone():
    t_r = np.arange(int(6 * FS)) / FS + 7.0        # through the overhead pass
    x = _flyover(t_r)
    observed = ot.observed_order_track(t_r, _track(), [0.0, 0.0, 0.0], F0 * 60.0, 1, C)
    y, _, _ = vold_kalman_filter(x, observed, FS, 2.0, 2)
    y_source, _, _ = vold_kalman_filter(x, np.full_like(t_r, F0), FS, 2.0, 2)
    core = slice(int(FS), -int(FS))
    np.testing.assert_allclose(np.abs(y[core]), 1.0, atol=0.02)
    assert np.median(np.abs(y_source[core])) < 0.5


# ------------------------------------------------------------ clock offsets

def _noise(n, fs, seed=0, cutoff=40.0):
    rng = np.random.default_rng(seed)
    b, a = sig.butter(4, cutoff / (fs / 2))
    return sig.filtfilt(b, a, rng.standard_normal(n))


def _delayed(x, delay_samples):
    """``x`` delayed by a fractional number of samples (circular, by FFT phase)."""
    k = np.fft.rfftfreq(len(x))
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * k * delay_samples), len(x))


@pytest.mark.parametrize('delay', [7.0, -12.0, 3.37, -0.62])
def test_clock_offset_of_a_delayed_record(delay):
    fs = 500.0
    a = _noise(5000, fs)
    b = _delayed(a, delay) + 0.05 * _noise(5000, fs, seed=1, cutoff=200.0)
    lag, peak = ot.estimate_clock_offset(a, b, fs, max_lag=0.1)
    assert lag == pytest.approx(delay / fs, abs=0.05 / fs)
    assert peak > 0.95


def test_clock_offset_ignores_missing_samples_and_a_mean():
    fs = 200.0
    a = _noise(4000, fs, cutoff=20.0)
    b = _delayed(a, 25.0) + 3.0
    b[1000:1400] = np.nan
    lag, peak = ot.estimate_clock_offset(a, b, fs, max_lag=0.5)
    assert lag == pytest.approx(25.0 / fs, abs=0.02 / fs)
    assert peak > 0.99


def test_clock_offset_searches_only_within_max_lag():
    fs = 100.0
    a = _noise(3000, fs, cutoff=10.0)
    b = _delayed(a, 40.0)
    lag, _ = ot.estimate_clock_offset(a, b, fs, max_lag=0.2)
    assert abs(lag) <= 0.2 + 1 / fs
    assert ot.estimate_clock_offset(a, b, fs, max_lag=0.5)[0] == pytest.approx(0.4, abs=1e-3)


def test_offset_between_predicted_and_observed_order_tracks():
    # The recorder's clock runs 0.37 s fast: the track it hears, on its own
    # clock, is the prediction 0.37 s later.  A rotor-speed change gives the
    # tracks structure besides the Doppler sweep.
    track = _track(0.0, 40.0, 4001)
    rpm = 400.0 + 30.0 * np.sin(2 * np.pi * track['time'] / 9.0)
    times = np.arange(3.0, 36.0, 0.05)
    predicted = ot.observed_order_track(times, track, [100.0, 0.0, 0.0], rpm, 6, C)
    observed = ot.observed_order_track(times - 0.37, track, [100.0, 0.0, 0.0], rpm, 6, C)
    lag, peak = ot.order_track_offset(times, predicted, observed, max_lag=2.0)
    assert lag == pytest.approx(0.37, abs=0.005)
    assert peak > 0.99
    with pytest.raises(ValueError, match='equally spaced'):
        ot.order_track_offset(times ** 1.01, predicted, observed, max_lag=2.0)
