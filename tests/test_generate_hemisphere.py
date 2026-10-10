import numpy as np
import pytest

import ground_plane as gp
from flight_acoustics import Atmosphere, depropagate_hemisphere


P_REF = 2.0e-5
ATMOSPHERE = Atmosphere(temperature=293.15, pressure=101.325, relative_humidity=20.0)


def test_depropagate_hemisphere_smoke():
    rng = np.random.default_rng(0)

    # Synthetic geometry: vehicle moves along +x, microphones off to the side
    nmics = 3
    mic_locations = np.array([
        [0.0, -50.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 50.0, 0.0],
    ])

    # Track (treated as emission-time samples)
    nt = 60
    track_time = np.linspace(0.0, 3.0, nt)
    track_position = np.column_stack([
        200.0 * (track_time / track_time[-1]) - 100.0,
        np.zeros(nt),
        np.zeros(nt),
    ])
    track_velocity = np.tile([200.0 / track_time[-1], 0.0, 0.0], (nt, 1))

    # Microphone signals
    fs = 2000.0
    duration = 3.5
    t = np.arange(0.0, duration, 1.0 / fs)
    base = (
        0.5 * P_REF * np.sin(2 * np.pi * 200.0 * t)
        + 0.3 * P_REF * np.sin(2 * np.pi * 500.0 * t)
    )
    pressure = np.vstack([
        base + 0.05 * P_REF * rng.standard_normal(t.size),
        base + 0.05 * P_REF * rng.standard_normal(t.size),
        base + 0.05 * P_REF * rng.standard_normal(t.size),
    ])

    hemi = depropagate_hemisphere(
        mic_locations=mic_locations,
        pressure=pressure,
        time=t,
        track_time=track_time,
        track_position=track_position,
        track_velocity=track_velocity,
        speed_of_sound=1135.0,
        length_units='ft',
        r_ref=100.0,
        freq_range=(0.0, 800.0),
        window_time=0.1,
        window_overlap=0.5,
        point_stride=3,
        azi_step=20.0,
        elv_step=15.0,
        rmax=35.0,
        apply_absorption_deprop=False,
        flip_y_for_geometry=False,
        third_octave=False,
    )

    assert 'azi_grid_deg' in hemi
    assert 'elv_grid_deg' in hemi
    assert 'oaspl_db' in hemi

    oaspl = hemi['oaspl_db']
    assert oaspl.ndim == 2
    assert oaspl.shape == (hemi['elv_grid_deg'].size, hemi['azi_grid_deg'].size)

    # Seam closure at 0/360
    assert np.allclose(oaspl[:, 0], oaspl[:, -1], equal_nan=True)

    # Should have at least some finite values
    assert np.isfinite(oaspl).any()


@pytest.mark.parametrize('height_ft, absorption', [(100.0, False), (400.0, False), (400.0, True)])
def test_broadband_levels_are_unbiased_between_frames(height_ft, absorption):
    """White noise heard from ``height_ft`` must depropagate to its band levels at
    r_ref: the mic's own levels plus the spreading, 20 lg(r/r_ref), and, with
    absorption, each bin's absorption over r - r_ref.

    Emission times fall between spectrogram frames.  Interpolating each PSD
    bin in dB there took a geometric mean of fluctuating periodogram bins and
    read about 0.7 dB low.  At r = r_ref neither the spreading nor the
    absorption term is exercised; 400 ft checks both.
    """

    rng = np.random.default_rng(3)
    fs, duration, r_ref_ft = 8000.0, 60.0, 100.0
    sigma = 1.0                                              # Pa rms, white
    t = np.arange(0.0, duration, 1.0 / fs)
    pressure = sigma * rng.standard_normal(t.size)

    # Hovering r_ref straight above the microphone, emission times on an
    # irregular grid so the observer times land between frames.
    track_time = np.sort(rng.uniform(2.0, duration - 3.0, 400))
    track_position = np.tile([0.0, 0.0, height_ft], (track_time.size, 1))
    track_velocity = np.tile([1e-6, 0.0, 0.0], (track_time.size, 1))

    hemi = depropagate_hemisphere(
        mic_locations=np.zeros((1, 3)), pressure=pressure[None, :], time=t,
        track_time=track_time, track_position=track_position, track_velocity=track_velocity,
        r_ref=r_ref_ft, freq_range=(0.0, 3500.0), window_time=0.5, window_overlap=0.5,
        azi_step=30.0, elv_step=10.0, rmax=15.0, third_octave=True, third_octave_fmin=200.0,
        apply_absorption_deprop=absorption, atmosphere=ATMOSPHERE, return_scattered=True)

    # Test the emission-point samples, where the interpolation happens: every
    # point sits on the same grid node, and Shepard weighting there returns a
    # single coincident sample rather than their mean.
    scattered = hemi['scattered']['third_octave']
    mean_power = np.mean(10.0 ** (scattered['bands_db'] / 10.0), axis=1)
    psd = sigma ** 2 / (fs / 2.0)                            # one-sided, Pa^2/Hz
    df = fs / 4096.0                                         # 0.5 s -> 4096-point frames
    for fc, power in zip(scattered['band_centers_hz'], mean_power):
        if fc > 3000.0:
            continue
        # The band's power is the sum of the FFT bins inside its edges, each
        # carried back from height_ft to r_ref.
        bins = np.arange(int(np.ceil(fc / 2 ** (1 / 6) / df)), int(np.ceil(fc * 2 ** (1 / 6) / df))) * df
        gain = np.ones(bins.size)
        if absorption:
            gain = 10.0 ** (ATMOSPHERE.attenuation_coefficient(bins) * (height_ft - r_ref_ft) * 0.3048 / 10.0)
        expected = (10.0 * np.log10(psd * df * np.sum(gain) / P_REF ** 2)
                    + 20.0 * np.log10(height_ft / r_ref_ft))
        assert 10.0 * np.log10(power) == pytest.approx(expected, abs=0.25), fc


@pytest.mark.parametrize('tone_aware', [False, True])
def test_remove_doppler_files_a_moving_tone_at_its_emitted_frequency(tone_aware):
    """A 1 kHz tone from a source passing at 300 ft/s is received 0.8-1.3 kHz.  With
    remove_doppler its band power belongs in the 1 kHz band, at the level its
    amplitude gives at r_ref."""
    c, speed, height, r_ref, f0 = 1125.0, 300.0, 100.0, 100.0, 1000.0
    fs = 8000.0
    t = np.arange(0.0, 4.0, 1.0 / fs)
    mics = np.array([[0.0, 0.0, 0.0], [0.0, 50.0, 0.0]])

    def position(time):
        time = np.asarray(time, dtype=float)
        return np.column_stack([speed * (time - 2.0), np.zeros(time.size), np.full(time.size, height)])

    pressure = []
    for mic in mics:
        t_e, x_e = gp.emission_times(position, t, mic, c)
        # 1 Pa amplitude at r_ref, spreading as 1/R
        pressure.append(r_ref / np.linalg.norm(x_e - mic, axis=1) * np.sin(2 * np.pi * f0 * t_e))
    track_time = np.linspace(0.3, 3.5, 200)
    hemi = depropagate_hemisphere(
        mic_locations=mics, pressure=np.array(pressure), time=t, track_time=track_time,
        track_position=position(track_time), track_velocity=np.tile([speed, 0.0, 0.0], (track_time.size, 1)),
        speed_of_sound=c, r_ref=r_ref, freq_range=(0.0, 3500.0), window_time=0.1, window_overlap=0.5,
        azi_step=30.0, elv_step=15.0, rmax=25.0, third_octave=True,
        third_octave_band_centers_hz=[500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0],
        remove_doppler=True, tone_aware=tone_aware, return_scattered=True)
    scattered = hemi['scattered']
    steep = scattered['elv_deg'] > 15.0
    centers = scattered['third_octave']['band_centers_hz']
    bands = 10.0 ** (scattered['third_octave']['bands_db'][:, steep] / 10.0)
    share = bands[np.argmin(np.abs(centers - f0))] / np.sum(bands, axis=0)
    assert steep.sum() > 50 and np.all(10.0 * np.log10(share) > -0.1)
    level = 10.0 * np.log10(bands[np.argmin(np.abs(centers - f0))] / 1.0)
    assert np.median(level) == pytest.approx(10.0 * np.log10(0.5 / P_REF ** 2), abs=0.3)
