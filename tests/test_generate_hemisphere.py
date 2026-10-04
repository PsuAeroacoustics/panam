import numpy as np

from flight_acoustics import depropagate_hemisphere


P_REF = 2.0e-5


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


def test_broadband_levels_are_unbiased_between_frames():
    """White noise heard at exactly r_ref must depropagate to its own band levels.

    Emission times fall between spectrogram frames.  Interpolating each PSD
    bin in dB there (the pre-2026-09-24 behavior) took a geometric mean of
    fluctuating periodogram bins and read about 0.7 dB low.
    """
    import pytest

    rng = np.random.default_rng(3)
    fs, duration, r_ref_ft = 8000.0, 60.0, 100.0
    sigma = 1.0                                              # Pa rms, white
    t = np.arange(0.0, duration, 1.0 / fs)
    pressure = sigma * rng.standard_normal(t.size)

    # Hovering r_ref straight above the microphone, emission times on an
    # irregular grid so the observer times land between frames.
    track_time = np.sort(rng.uniform(2.0, duration - 3.0, 400))
    track_position = np.tile([0.0, 0.0, r_ref_ft], (track_time.size, 1))
    track_velocity = np.tile([1e-6, 0.0, 0.0], (track_time.size, 1))

    hemi = depropagate_hemisphere(
        mic_locations=np.zeros((1, 3)), pressure=pressure[None, :], time=t,
        track_time=track_time, track_position=track_position, track_velocity=track_velocity,
        r_ref=r_ref_ft, freq_range=(0.0, 3500.0), window_time=0.5, window_overlap=0.5,
        azi_step=30.0, elv_step=10.0, rmax=15.0, third_octave=True, third_octave_fmin=200.0,
        return_scattered=True)

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
        # The band's power is the sum of the FFT bins inside its edges.
        nbins = int(np.ceil(fc * 2 ** (1 / 6) / df)) - int(np.ceil(fc / 2 ** (1 / 6) / df))
        expected = 10.0 * np.log10(psd * nbins * df / P_REF ** 2)
        assert 10.0 * np.log10(power) == pytest.approx(expected, abs=0.25), fc
