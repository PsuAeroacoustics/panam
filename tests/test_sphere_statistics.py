"""Energy averaging of repeat-run spheres and band-by-band sphere comparison."""

import numpy as np
import pytest
from netCDF4 import Dataset

from panam import flight_acoustics as fa
from panam import sphere_statistics as ss
from sphere_helpers import write_raw_sphere

PHI = np.array([-90.0, 0.0, 90.0])
THETA = np.array([0.0, 90.0, 180.0])
FREQUENCY = np.array([100.0, 125.0])


def _levels(value):
    return np.full((PHI.size, THETA.size, FREQUENCY.size), float(value))


def _raw(path, amplitude, **scalars):
    """A sphere written variable by variable, so that NaN stays NaN on disk."""
    values = dict(RADIUS=100.0, SPEED=60.0, FLIGHT_PATH_ANGLE=-6.0)
    values.update(scalars)
    write_raw_sphere(path, PHI, THETA, FREQUENCY, amplitude, **values)
    return path


def _sphere(path, levels, **kwargs):
    """A sphere written by write_aam_hemisphere_netcdf from ART levels."""
    options = dict(phi_deg=PHI, theta_deg=THETA, radius_ft=100.0, speed_knots=60.0,
                   flight_path_angle_deg=-6.0)
    options.update(kwargs)
    fa.write_aam_hemisphere_netcdf(str(path), None, art_levels=(FREQUENCY, levels), **options)
    return path


def test_runs_are_averaged_in_energy(tmp_path):
    a = _sphere(tmp_path / 'a.nc', _levels(80.0), speed_knots=58.0, empty_weight_lb=4000.0)
    b = _sphere(tmp_path / 'b.nc', _levels(90.0), speed_knots=62.0, empty_weight_lb=4200.0)
    average = ss.average_spheres([a, b], speed_tolerance_knots=5.0)
    np.testing.assert_allclose(average['levels_db'], 10.0 * np.log10((1e8 + 1e9) / 2.0))
    assert np.all(average['n_runs'] == 2)
    np.testing.assert_allclose(average['amplitude_sd_db'], np.sqrt(50.0))
    assert average['speed_knots'] == pytest.approx(60.0)
    assert average['weight_lb'] == pytest.approx(4100.0)


def test_a_cell_is_averaged_over_the_runs_that_cover_it(tmp_path):
    first, second = _levels(70.0), _levels(70.0)
    first[0, 0, 0] = np.nan                          # one run has no data
    first[1, 1, 0] = second[1, 1, 0] = np.nan        # neither run has data
    first[2, 2, 1] = -999.0                          # sentinel in one run
    first[0, 2, 1] = second[0, 2, 1] = -999.0        # sentinel in both
    a = _raw(tmp_path / 'a.nc', first)
    b = _raw(tmp_path / 'b.nc', second)

    average = ss.average_spheres([a, b])
    assert average['levels_db'][0, 0, 0] == pytest.approx(70.0)
    assert average['n_runs'][0, 0, 0] == 1
    assert np.isnan(average['levels_db'][1, 1, 0]) and average['n_runs'][1, 1, 0] == 0
    assert average['levels_db'][2, 2, 1] == pytest.approx(70.0)
    assert np.isnan(average['levels_db'][0, 2, 1])
    assert np.isnan(average['amplitude_sd_db'][0, 0, 0])

    zero = ss.average_spheres([a, b], missing='no_energy')
    assert zero['levels_db'][2, 2, 1] == pytest.approx(70.0 - 10.0 * np.log10(2.0))
    assert zero['n_runs'][2, 2, 1] == 2
    assert zero['levels_db'][0, 2, 1] == -np.inf     # every covering run has no energy
    assert np.isnan(zero['levels_db'][1, 1, 0])      # NaN on disk is still no data


def test_min_runs_leaves_thinly_covered_cells_without_a_level(tmp_path):
    first = _levels(70.0)
    first[0, 0, 0] = np.nan
    a = _raw(tmp_path / 'a.nc', first)
    b = _raw(tmp_path / 'b.nc', _levels(76.0))
    average = ss.average_spheres([a, b], min_runs=2)
    assert np.isnan(average['levels_db'][0, 0, 0])
    assert average['n_runs'][0, 0, 0] == 1
    assert np.isfinite(average['levels_db'][1, 1, 1])


def test_open_datasets_are_accepted(tmp_path):
    a = _sphere(tmp_path / 'a.nc', _levels(80.0))
    b = _sphere(tmp_path / 'b.nc', _levels(84.0))
    with Dataset(str(a)) as first, Dataset(str(b)) as second:
        average = ss.average_spheres([first, second])
    assert average['levels_db'][0, 0, 0] == pytest.approx(10.0 * np.log10((1e8 + 10 ** 8.4) / 2.0))


@pytest.mark.parametrize('change', [
    dict(phi_deg=PHI + 1.0), dict(radius_ft=150.0), dict(speed_knots=61.0),
    dict(azimuth_reference='heading'), dict(doppler_shift_removed=1.0)])
def test_mismatched_spheres_are_refused(tmp_path, change):
    a = _sphere(tmp_path / 'a.nc', _levels(80.0), azimuth_reference='track')
    b = _sphere(tmp_path / 'b.nc', _levels(80.0), **change)
    with pytest.raises(ValueError):
        ss.average_spheres([a, b])


def test_the_speed_tolerance_bounds_the_spread_of_all_runs(tmp_path):
    spheres = [_sphere(tmp_path / '{}.nc'.format(speed), _levels(80.0), speed_knots=speed)
               for speed in (60.0, 58.0, 62.0)]
    with pytest.raises(ValueError, match='SPEED'):
        ss.average_spheres(spheres, speed_tolerance_knots=2.0)
    assert ss.average_spheres(spheres, speed_tolerance_knots=4.0)['speed_knots'] == pytest.approx(60.0)


def test_the_flight_path_angle_tolerance_refuses_different_conditions(tmp_path):
    level = _sphere(tmp_path / 'level.nc', _levels(80.0), flight_path_angle_deg=0.0)
    descent = _sphere(tmp_path / 'descent.nc', _levels(80.0), flight_path_angle_deg=-9.0)
    with pytest.raises(ValueError, match='FLIGHT_PATH_ANGLE'):
        ss.average_spheres([level, descent], flight_path_angle_tolerance_deg=3.0)
    average = ss.average_spheres([level, descent])     # default: not checked
    assert average['flight_path_angle_deg'] == pytest.approx(-4.5)


def test_mismatched_bands_are_refused(tmp_path):
    a = _sphere(tmp_path / 'a.nc', _levels(80.0))
    b = tmp_path / 'b.nc'
    fa.write_aam_hemisphere_netcdf(str(b), None, art_levels=(FREQUENCY * 2.0, _levels(80.0)),
                                   phi_deg=PHI, theta_deg=THETA, radius_ft=100.0, speed_knots=60.0)
    with pytest.raises(ValueError, match='frequency'):
        ss.average_spheres([a, b])
    with pytest.raises(ValueError, match='frequency'):
        ss.compare_spheres(a, b)


def test_an_unlabeled_sphere_averages_with_a_track_one(tmp_path):
    a = _sphere(tmp_path / 'a.nc', _levels(80.0), azimuth_reference='track')
    b = _sphere(tmp_path / 'b.nc', _levels(80.0))
    assert ss.average_spheres([a, b])['azimuth_reference'] == 'track'


def test_the_average_is_written_with_its_run_counts(tmp_path):
    first = _levels(80.0)
    first[0, 0, 0] = np.nan
    a = _raw(tmp_path / 'a.nc', first)
    b = _raw(tmp_path / 'b.nc', _levels(82.0), DOPPLER_SHIFT_REMOVED=0.0)
    average = ss.average_spheres([a, b])
    out = tmp_path / 'average.nc'
    ss.write_averaged_sphere(str(out), average)

    amplitude, phi, theta, frequency, radius, speed, _ = fa.load_nc_sphere(str(out))
    np.testing.assert_allclose(amplitude[1:], average['levels_db'][1:], atol=1e-4)
    assert amplitude[0, 0, 0] == pytest.approx(82.0)
    with Dataset(str(out)) as handle:
        names = list(handle.variables)
        assert names[:len(fa.AAM_VARIABLE_ORDER)] == list(fa.AAM_VARIABLE_ORDER)
        assert names.index('N_RUNS') > names.index('AMPLITUDE')
        n_runs = handle.variables['N_RUNS'][:]
        assert n_runs.dtype == np.int32 and n_runs[0, 0, 0] == 1 and n_runs[1, 1, 1] == 2
        spread = np.asarray(handle.variables['AMPLITUDE_SD_DB'][:])
        assert spread[0, 0, 0] == fa.AAM_MISSING_LEVEL
        assert spread[1, 1, 1] == pytest.approx(np.sqrt(2.0))
        assert handle.variables['AMPLITUDE_SD_DB'].unit == 'DECIBEL'


def test_art_levels_are_written_without_resampling(tmp_path):
    levels = 60.0 + np.arange(_levels(0).size, dtype=float).reshape(_levels(0).shape)
    levels[1, 1, 1] = -np.inf
    path = _sphere(tmp_path / 'a.nc', levels)
    amplitude = fa.load_nc_sphere(str(path))[0]
    expected = levels.copy()
    expected[1, 1, 1] = fa.AAM_MISSING_LEVEL
    np.testing.assert_array_equal(amplitude, expected.astype(np.float32))


def test_writer_refuses_bad_art_levels_and_extra_variables(tmp_path):
    path = tmp_path / 'a.nc'
    with pytest.raises(ValueError, match='not both'):
        fa.write_aam_hemisphere_netcdf(str(path), {}, art_levels=(FREQUENCY, _levels(80.0)),
                                       phi_deg=PHI, theta_deg=THETA, radius_ft=100.0)
    with pytest.raises(ValueError, match='radius_ft'):
        fa.write_aam_hemisphere_netcdf(str(path), None, art_levels=(FREQUENCY, _levels(80.0)),
                                       phi_deg=PHI, theta_deg=THETA)
    with pytest.raises(ValueError, match='shaped'):
        fa.write_aam_hemisphere_netcdf(str(path), None, art_levels=(FREQUENCY[:1], _levels(80.0)),
                                       phi_deg=PHI, theta_deg=THETA, radius_ft=100.0)
    with pytest.raises(ValueError, match='shaped'):
        fa.write_aam_hemisphere_netcdf(str(path), None, art_levels=(FREQUENCY, _levels(80.0)),
                                       phi_deg=PHI, theta_deg=THETA, radius_ft=100.0,
                                       extra_variables={'N_RUNS': (np.zeros(3), '')})
    assert not path.exists()
    with pytest.raises(ValueError, match='replace'):
        fa.write_aam_hemisphere_netcdf(str(path), None, art_levels=(FREQUENCY, _levels(80.0)),
                                       phi_deg=PHI, theta_deg=THETA, radius_ft=100.0,
                                       extra_variables={'AMPLITUDE': (_levels(1.0), '')})
    assert not path.exists()


def test_cell_solid_angles_cover_the_lower_hemisphere():
    phi = np.arange(-90.0, 90.0 + 1e-9, 10.0)
    theta = np.arange(0.0, 180.0 + 1e-9, 5.0)
    weights = ss.cell_solid_angles(phi, theta)
    assert weights.shape == (phi.size, theta.size)
    assert weights.sum() == pytest.approx(2.0 * np.pi)
    assert np.all(weights > 0.0)


def test_compare_reports_per_band_statistics(tmp_path):
    reference = _levels(80.0)
    candidate = _levels(80.0)
    candidate[:, :, 0] += np.arange(9, dtype=float).reshape(3, 3)     # 0..8 dB
    candidate[:, :, 1] += 2.0
    reference[0, 0, 1] = -np.inf          # measured with no energy in A only
    candidate[2, 2, 1] = np.nan           # no data in B only
    a = _sphere(tmp_path / 'a.nc', reference)
    b = _sphere(tmp_path / 'b.nc', candidate)

    result = ss.compare_spheres(a, b)
    np.testing.assert_allclose(result['frequency_hz'], FREQUENCY)
    d = np.arange(9.0)
    assert result['n_cells'][0] == 9
    assert result['mean_db'][0] == pytest.approx(4.0)
    assert result['rms_db'][0] == pytest.approx(np.sqrt(np.mean(d ** 2)))
    assert result['p5_db'][0] == pytest.approx(np.percentile(d, 5))
    assert result['p95_db'][0] == pytest.approx(np.percentile(d, 95))
    weights = ss.cell_solid_angles(PHI, THETA).ravel()
    assert result['area_weighted_mean_db'][0] == pytest.approx(np.sum(weights * d) / weights.sum())
    assert result['n_cells'][1] == 7
    assert result['mean_db'][1] == pytest.approx(2.0)
    assert result['n_only_a'][1] == 1 and result['n_only_b'][1] == 1
    assert result['n_only_a'][0] == 0 and result['n_only_b'][0] == 0
    assert np.isnan(result['difference_db'][0, 0, 1])

    # The overall level sums the bands finite in both files: a single band at (0, 0).
    assert result['overall_difference_db'][0, 0] == pytest.approx(0.0)
    expected = 10.0 * np.log10((10 ** 8.7 + 10 ** 8.2) / (2 * 10 ** 8.0))
    assert result['overall_difference_db'][2, 1] == pytest.approx(expected)
    assert result['overall']['n_cells'] == 9

    one = ss.compare_spheres(a, b, bands=[125.0])
    np.testing.assert_allclose(one['frequency_hz'], [125.0])
    assert one['mean_db'][0] == pytest.approx(2.0)
    with pytest.raises(ValueError, match='Hz band'):
        ss.compare_spheres(a, b, bands=[1000.0])
    assert 'overall' in ss.format_comparison(result)


def test_area_weighting_discounts_the_crowded_nose(tmp_path):
    candidate = _levels(80.0)
    candidate[:, 0, :] += 3.0       # only theta = 0, where the phi points coincide
    result = ss.compare_spheres(_sphere(tmp_path / 'a.nc', _levels(80.0)),
                                _sphere(tmp_path / 'b.nc', candidate))
    assert result['mean_db'][0] == pytest.approx(1.0)
    assert result['area_weighted_mean_db'][0] < 0.5


def test_difference_map_uses_a_diverging_colormap(tmp_path):
    candidate = _levels(80.0)
    candidate[0] += 2.0
    result = ss.compare_spheres(_sphere(tmp_path / 'a.nc', _levels(80.0)),
                                _sphere(tmp_path / 'b.nc', candidate))
    fig = ss.plot_sphere_difference(result, band=100.0)
    contour = [c for c in fig.axes[0].collections if hasattr(c, 'get_cmap')][0]
    assert contour.get_cmap().name == 'RdBu_r'
    assert fig.axes[1].get_ylabel().startswith('100 Hz level difference')
    assert 'Sound Pressure Level' not in fig.axes[1].get_ylabel()
