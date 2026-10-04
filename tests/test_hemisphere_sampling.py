import numpy as np

import flight_acoustics as fa


def _hemisphere(bands_db, azi, elv):
    return {'azi_grid_deg': azi, 'elv_grid_deg': elv,
            'third_octave': {'band_centers_hz': np.array([100.0]), 'bands_db': bands_db[None]}}


def test_levels_next_to_a_coverage_gap_are_not_pulled_toward_zero():
    """Unmeasured (NaN) cells carry no weight; they used to count as zero power."""
    azi = np.arange(0.0, 361.0, 5.0)
    elv = np.arange(-90.0, 91.0, 5.0)
    levels = np.full((elv.size, azi.size), 80.0)
    levels[(elv >= 0.0) & (elv <= 10.0), :] = np.nan
    _, sampled = fa._sample_hemisphere_levels(_hemisphere(levels, azi, elv), 'auto',
                                              np.full(5, 90.0), np.array([11.0, 12.5, 13.0, 15.0, 20.0]),
                                              -np.inf)
    # 11 deg is weighted mostly by the unmeasured 10 deg row: missing
    assert sampled[0, 0] == -np.inf
    np.testing.assert_allclose(sampled[1:, 0], 80.0)


def test_zero_energy_cells_still_count():
    """-inf (measured, no energy) is not missing: it pulls the interpolation down."""
    azi = np.arange(0.0, 361.0, 10.0)
    elv = np.arange(-90.0, 91.0, 10.0)
    levels = np.full((elv.size, azi.size), 80.0)
    levels[elv == 0.0, :] = -np.inf
    _, sampled = fa._sample_hemisphere_levels(_hemisphere(levels, azi, elv), 'auto',
                                              np.array([90.0]), np.array([5.0]), -np.inf)
    np.testing.assert_allclose(sampled[0, 0], 80.0 + 10.0 * np.log10(0.5))
