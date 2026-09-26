"""Directions with no data, and directions with no energy, from grid to plot.

Two things used to come out as a finite -3076 dB (10 log10 of the smallest
float): hemisphere cells with no sample within rmax, and cells whose samples
carry no power.  Read as levels they set plot colour scales to -3500..500 dB and
entered residual statistics.  No data is now NaN and no energy -inf.  Sphere
files, for their part, came back from load_nc_sphere as masked arrays, whose
masked cells reached safe_log10 as masked scalars and whose masks were all that
kept NaN out of the footprint reductions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from netCDF4 import Dataset

import flight_acoustics as fa
from test_write_aam_hemisphere_netcdf import _minimal_hemisphere

P_REF = 2.0e-5
VEHICLE_CFG = ('[Main Rotor]\nradius = 5.334\ntip speed = 230.7\nblades = 4\n'
               '[Tail Rotor]\nradius = 0.8255\ntip speed = 216.1\nblades = 2\n'
               '[Atmosphere]\ndensity = 1.070\ntemperature = 280.37\n'
               '[Vehicle]\nweight = 2250\ndrag = 0.8175\n')


@pytest.fixture(scope='module')
def flyby():
    """A level flyby over three microphones: it covers a band of the
    hemisphere, leaving the rest with no sample within rmax.  With 10 Hz
    frequency bins the 12.5 and 16 Hz third-octave bands hold no bin, so every
    measured cell in them has zero power."""
    rng = np.random.default_rng(0)
    mics = np.array([[0.0, -50.0, 0.0], [0.0, 0.0, 0.0], [0.0, 50.0, 0.0]])
    track_time = np.linspace(0.0, 8.0, 60)
    position = np.column_stack([300.0 * track_time / track_time[-1] - 150.0,
                                np.zeros(track_time.size), np.full(track_time.size, 150.0)])
    velocity = np.tile([300.0 / track_time[-1], 0.0, 0.0], (track_time.size, 1))
    fs = 5120.0     # 512-sample frames at window_time=0.1: 10 Hz bins
    t = np.arange(0.0, 9.0, 1.0 / fs)
    tones = 1e3 * P_REF * (np.sin(2 * np.pi * 20.0 * t) + 0.3 * np.sin(2 * np.pi * 250.0 * t))
    pressure = np.vstack([tones + 50 * P_REF * rng.standard_normal(t.size) for _ in range(3)])
    return fa.depropagate_hemisphere(
        mics, pressure, t, track_time, position, velocity, speed_of_sound=1135.0, length_units='ft',
        r_ref=100.0, freq_range=(0.0, 1500.0), window_time=0.1, point_stride=2, azi_step=10.0,
        elv_step=10.0, rmax=25.0, third_octave=True, third_octave_fmin=10.0, return_scattered=True,
        apply_absorption_deprop=True)


def test_a_node_with_no_sample_within_rmax_is_nan_not_zero():
    """The MATLAB original returns 0 there, which as a power is a measured zero."""
    felv, fazi, values = np.array([10.0, 12.0]), np.array([0.0, 5.0]), np.array([1.0, 2.0])
    result = fa.shepIDW(np.array([11.0, 80.0]), np.array([2.0, 180.0]), felv, fazi, values, 25.0)
    assert np.isfinite(result[0])
    assert np.isnan(result[1])


def test_cells_with_no_sample_within_rmax_are_nan(flyby):
    elv, azi = np.meshgrid(flyby['elv_grid_deg'], flyby['azi_grid_deg'], indexing='ij')
    samples_elv, samples_azi = flyby['scattered']['elv_deg'], flyby['scattered']['azi_deg']
    nearest = np.min(fa.geodist(elv[..., None], azi[..., None], samples_elv, samples_azi), axis=-1)
    no_data = nearest >= 25.0
    assert no_data.any() and not no_data.all()
    for grid in (flyby['oaspl_db'], flyby['spl_a_db'], *flyby['third_octave']['bands_db']):
        np.testing.assert_array_equal(np.isnan(grid), no_data)


def test_nothing_is_floored_to_a_finite_level(flyby):
    for levels in (flyby['oaspl_db'], flyby['spl_a_db'], flyby['third_octave']['bands_db'],
                   flyby['scattered']['third_octave']['bands_db']):
        levels = np.asarray(levels)
        assert not np.any(np.isfinite(levels) & (levels < -1000.0))


def test_measured_cells_with_no_energy_are_minus_inf(flyby):
    centers = flyby['third_octave']['band_centers_hz']
    measured = ~np.isnan(flyby['oaspl_db'])
    for fc in (12.5, 16.0):
        band = flyby['third_octave']['bands_db'][int(np.argmin(np.abs(centers - fc)))]
        assert np.all(np.isneginf(band[measured]))
        assert np.all(np.isnan(band[~measured]))


def test_the_lambert_plot_scales_to_the_measured_levels(flyby):
    azi, elv = np.meshgrid(flyby['azi_grid_deg'], flyby['elv_grid_deg'])
    figure, _, contours = fa.plot_lambert_ea(np.deg2rad(azi), np.deg2rad(elv), flyby['oaspl_db'])
    plt.close(figure)
    measured = flyby['oaspl_db'][np.isfinite(flyby['oaspl_db'])]
    assert measured.min() - 5.0 < contours.levels[0] <= measured.min()
    assert measured.max() <= contours.levels[-1] < measured.max() + 5.0


def test_export_does_not_blend_cells_with_no_data_in_as_zero_energy(tmp_path):
    """Measured at 80 dB everywhere it was measured, the sphere must say 80 dB
    wherever it says anything.  Blended in as zero power, the cells with no data
    pulled the directions around them low."""
    hemisphere = _minimal_hemisphere()
    hemisphere['azi_grid_deg'] = np.arange(0.0, 360.0 + 1e-9, 10.0)
    hemisphere['elv_grid_deg'] = np.arange(0.0, 90.0 + 1e-9, 10.0)
    bands = np.full((3, hemisphere['elv_grid_deg'].size, hemisphere['azi_grid_deg'].size), 80.0)
    bands[:, hemisphere['elv_grid_deg'] < 45.0, :] = np.nan
    hemisphere['third_octave']['bands_db'] = bands
    path = tmp_path / 'edge.nc'
    fa.write_aam_hemisphere_netcdf(str(path), hemisphere, mode='third_octave', radius_ft=100.0, title='t')
    amplitude = fa.load_nc_sphere(str(path))[0]
    written = amplitude[amplitude > fa.AAM_MISSING_THRESHOLD]
    assert written.size and np.any(amplitude <= fa.AAM_MISSING_THRESHOLD)
    np.testing.assert_allclose(written, 80.0, atol=1e-4)


def test_load_nc_sphere_returns_plain_arrays_with_masked_cells_as_nan(tmp_path):
    path = tmp_path / 'filled.nc'
    with Dataset(str(path), 'w', format='NETCDF3_CLASSIC') as sphere:
        for name, size in (('PHI', 2), ('THETA', 3), ('FREQUENCY', 2)):
            sphere.createDimension(name, size)
        sphere.createVariable('PHI', 'f4', ('PHI',))[:] = [-10.0, 10.0]
        sphere.createVariable('THETA', 'f4', ('THETA',))[:] = [0.0, 90.0, 180.0]
        sphere.createVariable('FREQUENCY', 'f4', ('FREQUENCY',))[:] = [100.0, 125.0]
        amplitude = sphere.createVariable('AMPLITUDE', 'f4', ('PHI', 'THETA', 'FREQUENCY'), fill_value=-12345.0)
        values = np.full((2, 3, 2), 70.0, dtype=np.float32)
        values[0, 1, :] = -12345.0
        amplitude[:] = values
        for name in ('RADIUS', 'SPEED', 'FLIGHT_PATH_ANGLE'):
            sphere.createVariable(name, 'f4').assignValue(np.float32(1.0))
    loaded = fa.load_nc_sphere(str(path))
    assert not any(isinstance(value, np.ma.MaskedArray) for value in loaded)
    assert np.all(np.isnan(loaded[0][0, 1, :]))
    assert np.count_nonzero(np.isnan(loaded[0])) == 2
    assert np.all(np.isneginf(fa.mask_missing_levels(loaded[0])[0, 1, :]))


def test_a_failed_read_leaves_numpy_error_handling_as_it_was(tmp_path):
    path = tmp_path / 'incomplete.nc'
    with Dataset(str(path), 'w', format='NETCDF3_CLASSIC') as sphere:
        sphere.createDimension('PHI', 1)
        sphere.createVariable('PHI', 'f4', ('PHI',))[:] = [0.0]
    before = np.geterr()
    with pytest.raises(KeyError):
        fa.load_nc_sphere(str(path))
    assert np.geterr() == before


def _sphere_directory(tmp_path, hemispheres):
    directory = tmp_path / 'spheres'
    directory.mkdir()
    (directory / 'vehicle.cfg').write_text(VEHICLE_CFG)
    for i, (hemisphere, speed, angle) in enumerate(hemispheres):
        fa.write_aam_hemisphere_netcdf(str(directory / f'X{101 + i:03d}.nc'), hemisphere, mode='third_octave',
                                       phi_deg=np.arange(-90.0, 90.0 + 1e-9, 10.0),
                                       theta_deg=np.arange(0.0, 180.0 + 1e-9, 10.0), radius_ft=100.0,
                                       speed_knots=float(speed), flight_path_angle_deg=float(angle), title='t')
    return directory


def _gated_hemisphere(offset_db):
    """80 dB (plus offset), with everything within 30 degrees of straight down
    gated out: measured, and left with no energy."""
    hemisphere = _minimal_hemisphere()
    hemisphere['third_octave']['bands_db'] = hemisphere['third_octave']['bands_db'] + offset_db
    steep = hemisphere['elv_grid_deg'] >= 60.0
    hemisphere['third_octave']['bands_db'][:, steep, :] = -np.inf
    return hemisphere


@pytest.mark.filterwarnings('error::RuntimeWarning')
def test_reducing_a_sphere_with_missing_directions_is_warning_free(tmp_path):
    directory = _sphere_directory(tmp_path, [(_gated_hemisphere(0.0), 60, -3)])
    results = fa.extract_SPL(str(directory / 'X101.nc'))
    assert not any(isinstance(value, np.ma.MaskedArray) for value in results)
    assert np.isneginf(results[6]).any()


def test_footprint_levels_leave_out_directions_with_no_energy(tmp_path):
    conditions = [(40, -6), (60, -3), (80, 0), (100, -6), (60, 3), (80, -9)]
    directory = _sphere_directory(tmp_path, [(_gated_hemisphere(3.0 * i), speed, angle)
                                             for i, (speed, angle) in enumerate(conditions)])
    _, _, level_max, level_mean, *_ = fa.project_directory(str(directory), altitude=150, cutoff=20)
    assert np.all(np.isfinite(level_max)) and np.all(np.isfinite(level_mean))
    x, y, level_a, _, _ = fa.project_sphere(str(directory / 'X101.nc'), 150.0, 20.0)
    assert np.isneginf(level_a).any() and not np.isnan(level_a).any()
    kept = (np.hypot(x, y) > 0) & np.isfinite(level_a)
    assert level_max[0] == pytest.approx(np.max(level_a[kept]))
    assert level_mean[0] == pytest.approx(np.mean(level_a[kept]))
    figure, _, contours = fa.plot_projection(str(directory / 'X101.nc'), altitude=150, cutoff=20)
    plt.close(figure)
    assert np.all(np.isfinite(contours.levels))


def test_a_sphere_with_no_level_below_the_aircraft_is_skipped(tmp_path):
    empty = _minimal_hemisphere()
    empty['third_octave']['bands_db'][:] = -np.inf
    directory = _sphere_directory(tmp_path, [(_gated_hemisphere(0.0), 60, -3), (empty, 80, 0),
                                             (_gated_hemisphere(3.0), 100, -6)])
    with pytest.warns(UserWarning, match='X102.nc: no direction'):
        speeds, *_ = fa.project_directory(str(directory), altitude=150, cutoff=20)
    assert speeds.size == 2


def test_the_planner_reads_the_old_no_data_floor_as_missing(tmp_path):
    import array_planner

    levels = np.full((4, 7), 62.0)
    levels[0, :] = 10.0 * np.log10(np.finfo(float).tiny)
    path = tmp_path / 'old.npz'
    np.savez(path, azi_grid_deg=np.arange(0.0, 361.0, 60.0), elv_grid_deg=np.arange(0.0, 91.0, 30.0),
             hemisphere_oaspl_fullband_db=levels)
    grid = array_planner._load_depropagated_hemisphere_npz(str(path), field='oaspl_fullband', fc_hz=None)
    assert np.all(np.isnan(grid.spl_db[0])) and np.all(grid.spl_db[1:] == 62.0)
