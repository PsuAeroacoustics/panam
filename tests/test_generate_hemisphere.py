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
