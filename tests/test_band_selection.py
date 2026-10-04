import numpy as np
import pytest

import flight_acoustics as fa


def _tone_hemisphere(**overrides):
    """A small fly-past with a broadband signal, through depropagate_hemisphere."""
    rng = np.random.default_rng(0)
    mics = np.array([[0.0, -50.0, 0.0], [0.0, 0.0, 0.0], [0.0, 50.0, 0.0]])
    track_time = np.linspace(0.0, 3.0, 60)
    position = np.column_stack([200.0 * track_time / 3.0 - 100.0, np.zeros(60), np.zeros(60)])
    velocity = np.tile([200.0 / 3.0, 0.0, 0.0], (60, 1))
    fs = 8000.0
    t = np.arange(0.0, 3.5, 1.0 / fs)
    pressure = 2e-5 * rng.standard_normal((3, t.size))
    kwargs = dict(mic_locations=mics, pressure=pressure, time=t, track_time=track_time,
                  track_position=position, track_velocity=velocity, speed_of_sound=1135.0,
                  length_units='ft', r_ref=100.0, freq_range=(0.0, 2000.0), window_time=0.1,
                  window_overlap=0.5, point_stride=3, azi_step=20.0, elv_step=15.0, rmax=35.0,
                  apply_absorption_deprop=False, third_octave=True, third_octave_fmin=100.0)
    kwargs.update(overrides)
    return fa.depropagate_hemisphere(**kwargs)


def test_bands_reaching_past_the_frequency_range_are_left_out():
    """The 2 kHz band (1782-2245 Hz) is not summed over 1782-2000 Hz only."""
    centers = _tone_hemisphere()['third_octave']['band_centers_hz']
    lower, upper = fa.third_octave_band_edges(centers)
    assert upper.max() <= 2000.0
    assert np.isclose(centers.max(), 1000.0 * 2.0 ** (2.0 / 3.0))      # the 1.6 kHz band
    assert np.isclose(centers.min(), 1000.0 * 2.0 ** (-10.0 / 3.0))    # the 100 Hz band


def test_requested_bands_outside_the_range_warn():
    with pytest.warns(UserWarning, match='extend past freq_range'):
        hemi = _tone_hemisphere(third_octave_band_centers_hz=[1000.0, 1250.0, 1600.0, 2000.0])
    np.testing.assert_array_equal(hemi['third_octave']['band_centers_hz'], [1000.0, 1250.0, 1600.0])


def test_third_octave_band_levels_keeps_nominal_limits_and_nyquist():
    fs = 8000.0
    signal = 2e-5 * np.random.default_rng(1).standard_normal(int(fs))
    centers, _ = fa.third_octave_band_levels(signal, fs, fmin=20.0, fmax=20000.0)
    assert np.isclose(centers.min(), 1000.0 * 2.0 ** (-17.0 / 3.0))    # 19.69 Hz, the 20 Hz band
    assert centers.max() * 2.0 ** (1.0 / 6.0) <= fs / 2.0
