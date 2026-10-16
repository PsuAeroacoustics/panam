"""Per-node sample statistics of a gridded hemisphere (sphere_statistics.node_statistics)."""

import numpy as np
import pytest

from panam import flight_acoustics as fa
from panam import sphere_statistics as ss
from sphere_helpers import tone_flyover


def _db(p):
    return 10.0 * np.log10(p)


def test_two_equal_weights_by_hand():
    stats = ss.node_statistics(np.array([[1.0, 1.0]]), np.array([1e6, 1e7]), np.array([0, 1]))
    mean = _db(5.5e6)
    assert stats['n_samples'][0] == 2 and stats['n_mics'][0] == 2
    assert stats['n_eff'][0] == pytest.approx(2.0)
    assert stats['sd_db'][0] == pytest.approx(np.sqrt(((60.0 - mean) ** 2 + (70.0 - mean) ** 2) / 2.0))


def test_n_eff_of_unequal_weights_and_zero_sd_of_identical_samples():
    stats = ss.node_statistics(np.array([[3.0, 1.0]]), np.array([4e4, 4e4]), np.array([2, 2]))
    assert stats['n_eff'][0] == pytest.approx(16.0 / 10.0)
    assert stats['n_mics'][0] == 1
    assert stats['sd_db'][0] == pytest.approx(0.0, abs=1e-12)


def test_n_eff_equals_n_for_equal_weights():
    n = 7
    rng = np.random.default_rng(1)
    stats = ss.node_statistics(np.full((1, n), 0.3), rng.uniform(1.0, 10.0, n), np.arange(n) % 3)
    assert stats['n_samples'][0] == n and stats['n_mics'][0] == 3
    assert stats['n_eff'][0] == pytest.approx(n)


def test_missing_samples_are_left_out_and_the_weights_renormalized():
    # The NaN sample carries the largest weight; the other two are what the node has.
    stats = ss.node_statistics(np.array([[1.0, 2.0, 1.0]]), np.array([1.0, np.nan, 100.0]), np.array([0, 1, 0]))
    mean = _db(50.5)
    assert stats['n_samples'][0] == 2 and stats['n_mics'][0] == 1
    assert stats['n_eff'][0] == pytest.approx(2.0)
    assert stats['sd_db'][0] == pytest.approx(np.sqrt(((0.0 - mean) ** 2 + (20.0 - mean) ** 2) / 2.0))


def test_a_zero_power_sample_counts_but_has_no_level():
    # Mean energy 50 (17.0 dB); the SD is over the one sample with a level.
    stats = ss.node_statistics(np.array([[1.0, 1.0]]), np.array([0.0, 100.0]), np.array([0, 1]))
    assert stats['n_samples'][0] == 2
    assert stats['sd_db'][0] == pytest.approx(20.0 - _db(50.0))
    only_zero = ss.node_statistics(np.array([[1.0, 1.0]]), np.array([0.0, 0.0]), np.array([0, 1]))
    assert only_zero['n_samples'][0] == 2 and np.isnan(only_zero['sd_db'][0])


def test_a_node_with_no_samples_and_several_fields():
    weights = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    power = np.array([[1.0, 1.0, 5.0], [np.nan, np.nan, 5.0]])
    stats = ss.node_statistics(weights, power, np.array([0, 1, 2]))
    assert stats['n_samples'].shape == (2, 2)
    np.testing.assert_array_equal(stats['n_samples'], [[0, 2], [0, 0]])
    np.testing.assert_array_equal(stats['n_eff'], [[0.0, 2.0], [0.0, 0.0]])
    assert np.isnan(stats['sd_db'][:, 0]).all() and np.isnan(stats['sd_db'][1, 1])
    assert stats['sd_db'][0, 1] == pytest.approx(0.0, abs=1e-12)


def test_the_shepard_neighbor_list_becomes_the_same_weights():
    felv, fazi = np.array([10.0, 12.0, 40.0]), np.array([100.0, 104.0, 200.0])
    shape, neighbors = fa.shepIDW_weights(np.array([[11.0, 60.0]]), np.array([[101.0, 300.0]]), felv, fazi, 25.0)
    matrix = ss.idw_weight_matrix(neighbors, felv.size).toarray()
    assert matrix.shape == (2, 3)
    near, w = neighbors[0]
    np.testing.assert_array_equal(matrix[0, near], w)
    assert matrix[1].sum() == 0.0


def _pass():
    return dict(tone_flyover((-300.0, 0.0, 300.0, 600.0), seed=4), speed_of_sound=1135.0,
                r_ref=100.0, freq_range=(50.0, 1500.0), window_time=0.25, window_overlap=0.5, point_stride=2,
                azi_step=10.0, elv_step=10.0, rmax=25.0, third_octave=True, third_octave_fmin=100.0,
                narrowband=True, narrowband_stride=20, return_scattered=True)


@pytest.mark.parametrize('interpolation', [None, dict(min_mics=2, source_extent=35.0)])
def test_depropagate_hemisphere_reports_statistics_and_leaves_the_levels_alone(interpolation):
    common = _pass()
    plain = fa.depropagate_hemisphere(**common, interpolation=interpolation)
    out = fa.depropagate_hemisphere(**common, interpolation=interpolation, return_statistics=True)
    assert 'statistics' not in plain
    for key in ('oaspl_db', 'spl_a_db'):
        np.testing.assert_array_equal(out[key], plain[key])
    np.testing.assert_array_equal(out['third_octave']['bands_db'], plain['third_octave']['bands_db'])
    np.testing.assert_array_equal(out['narrowband']['psd_db'], plain['narrowband']['psd_db'])

    stats = out['statistics']
    assert stats['third_octave']['bands_db']['sd_db'].shape == out['third_octave']['bands_db'].shape
    assert stats['narrowband']['psd_db']['n_eff'].shape == out['narrowband']['psd_db'].shape
    overall = stats['oaspl_db']
    # A node has samples exactly where it has a level, and never more microphones than
    # samples or more effective samples than samples.
    np.testing.assert_array_equal(overall['n_samples'] > 0, ~np.isnan(out['oaspl_db']))
    assert np.all(overall['n_mics'] <= overall['n_samples'])
    assert np.all(overall['n_mics'] <= 4)
    assert np.all(overall['n_eff'] <= overall['n_samples'] + 1e-9)
    has = overall['n_samples'] > 1
    assert has.any() and np.all(overall['n_eff'][has] >= 1.0 - 1e-9)
    assert np.all(np.isfinite(overall['sd_db'][has])) and np.all(overall['sd_db'][has] >= 0.0)
    np.testing.assert_array_equal(overall['n_samples'][:, -1], overall['n_samples'][:, 0])


def test_the_fixed_radius_statistics_match_a_hand_computation_at_one_node():
    out = fa.depropagate_hemisphere(**_pass(), return_statistics=True)
    scattered = out['scattered']
    elv, azi = out['elv_grid_deg'], out['azi_grid_deg']
    stats = out['statistics']['oaspl_db']
    i, j = np.unravel_index(np.argmax(stats['n_samples']), stats['n_samples'].shape)
    _, neighbors = fa.shepIDW_weights(np.array([elv[i]]), np.array([azi[j]]), scattered['elv_deg'],
                                      scattered['azi_deg'], 25.0)
    near, w = neighbors[0]
    p = scattered['oaspl_power'][near]
    w = w / w.sum()
    mean = np.sum(w * p)
    assert 10.0 * np.log10(mean) == pytest.approx(out['oaspl_db'][i, j], abs=1e-9)
    assert stats['n_samples'][i, j] == near.size
    assert stats['n_mics'][i, j] == np.unique(scattered['mic'][near]).size
    assert stats['n_eff'][i, j] == pytest.approx(1.0 / np.sum(w * w))
    assert stats['sd_db'][i, j] == pytest.approx(np.sqrt(np.sum(w * (10.0 * np.log10(p / mean)) ** 2)))


def test_the_statistics_go_to_a_sphere_file_as_extra_variables(tmp_path):
    from netCDF4 import Dataset
    out = fa.depropagate_hemisphere(**_pass(), return_statistics=True)
    extra = ss.node_statistics_variables(out)
    n_bands = out['third_octave']['band_centers_hz'].size
    shape = (fa.AAM_DEFAULT_PHI_DEG.size, fa.AAM_DEFAULT_THETA_DEG.size, n_bands)
    assert extra['N_SAMPLES'][0].shape == extra['AMPLITUDE_STD'][0].shape == shape
    # Directly below the vehicle (phi 0, theta 90) is elevation 90 in either convention.
    below = ss.node_statistics_variables(out, [0.0], [90.0])['N_SAMPLES'][0][0, 0]
    np.testing.assert_array_equal(below, out['statistics']['third_octave']['bands_db']['n_samples'][:, -1, 0])
    path = tmp_path / 'sphere.nc'
    fa.write_aam_hemisphere_netcdf(str(path), out, mode='third_octave', extra_variables=extra)
    with Dataset(str(path)) as handle:
        np.testing.assert_array_equal(handle['N_SAMPLES'][:], extra['N_SAMPLES'][0])
        assert handle['AMPLITUDE_STD'].unit == 'DECIBEL'
    with pytest.raises(ValueError, match='return_statistics'):
        ss.node_statistics_variables(fa.depropagate_hemisphere(**_pass()))
