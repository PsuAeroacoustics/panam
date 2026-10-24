"""Sphere frames: PANAM's level spheres and AAM's flight-path ones (THEORY 3.2).

AAM's sphere is fixed to the body, pitched by the deck's angle of attack plus the
flight path angle; with AOA 0 its nose is the velocity.  PANAM, NICEOPS and NORAH2
file directions level.  These pin the pitch's sign, the writer's default, and the
builder's rotation back to level.
"""

import warnings

import numpy as np
import pytest
from netCDF4 import Dataset

from panam import flight_acoustics as fa
from panam.geometry import pitch_art_directions
from panam.gridding import repitch_art_levels
from panam.sphere_io import read_sphere_frame
from panam import sphere_statistics

import sphere_helpers


def _down_level_hemisphere():
    """A hemisphere whose band levels are 60 + 10 sin(elevation) dB: louder toward straight down."""
    band_centers = np.array([100.0, 125.0])
    elv = np.arange(0.0, 90.0 + 1e-9, 2.5)
    azi = np.arange(0.0, 360.0 + 1e-9, 5.0)
    level = 60.0 + 10.0 * np.sin(np.radians(elv))[:, None] + 0.0 * azi[None, :]
    bands = np.repeat(level[None, :, :], band_centers.size, axis=0)
    return dict(azi_grid_deg=azi, elv_grid_deg=elv,
                third_octave=dict(band_centers_hz=band_centers, bands_db=bands))


def _level_down(phi_deg, theta_deg, gamma_deg):
    """The level frame's down component of a flight-path direction at flight path angle gamma."""
    phi, theta, gamma = np.radians(phi_deg), np.radians(theta_deg), np.radians(gamma_deg)
    return -np.sin(gamma) * np.cos(theta) + np.cos(gamma) * np.sin(theta) * np.cos(phi)


def test_a_climbing_nose_points_up_and_a_descending_one_down():
    # Theta 0 of a flight-path sphere is the velocity: gamma above the horizon in a
    # climb (phi 180, the upper half), gamma below it in a descent (phi 0).
    phi, theta = pitch_art_directions(0.0, 0.0, 10.0)
    assert theta == pytest.approx(10.0) and abs(phi) == pytest.approx(180.0)
    phi, theta = pitch_art_directions(0.0, 0.0, -10.0)
    assert theta == pytest.approx(10.0) and phi == pytest.approx(0.0)
    # Starboard is the pitch axis.
    phi, theta = pitch_art_directions(90.0, 90.0, 25.0)
    assert (phi, theta) == (pytest.approx(90.0), pytest.approx(90.0))
    # And -gamma undoes gamma.
    grid_phi, grid_theta = np.meshgrid(np.arange(-80.0, 81.0, 20.0), np.arange(5.0, 176.0, 10.0))
    back = pitch_art_directions(*pitch_art_directions(grid_phi, grid_theta, 7.0), -7.0)
    np.testing.assert_allclose(back[0], grid_phi, atol=1e-9)
    np.testing.assert_allclose(back[1], grid_theta, atol=1e-9)


def test_the_aam_writer_files_the_flight_path_frame_by_default(tmp_path):
    # The nose cell of a -8 deg sphere hears 8 deg below the horizon, and every
    # cell the direction AAM will pitch it to; a level sphere hears the horizon there.
    hemisphere = _down_level_hemisphere()
    gamma = -8.0
    path = tmp_path / 'X101.nc'
    fa.write_aam_hemisphere_netcdf(str(path), hemisphere, mode='third_octave', radius_ft=100.0,
                                   speed_knots=60.0, flight_path_angle_deg=gamma)
    assert read_sphere_frame(str(path)) == 'flight_path'
    amplitude, phi, theta, *_ = fa.load_nc_sphere(str(path))
    assert amplitude[0, 0, 0] == pytest.approx(60.0 + 10.0 * np.sin(np.radians(-gamma)), abs=0.05)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    expected = 60.0 + 10.0 * np.abs(_level_down(phi_grid, theta_grid, gamma))
    np.testing.assert_allclose(amplitude[:, :, 0], expected, atol=0.15)

    level = tmp_path / 'X102.nc'
    fa.write_aam_hemisphere_netcdf(str(level), hemisphere, mode='third_octave', radius_ft=100.0,
                                   speed_knots=60.0, flight_path_angle_deg=gamma, sphere_frame='level')
    assert read_sphere_frame(str(level)) == 'level'
    assert fa.load_nc_sphere(str(level))[0][0, 0, 0] == pytest.approx(60.0, abs=0.05)
    with pytest.raises(ValueError, match='sphere_frame'):
        fa.write_aam_hemisphere_netcdf(str(tmp_path / 'X103.nc'), hemisphere, mode='third_octave',
                                       radius_ft=100.0, sphere_frame='body')


def test_repitching_a_flight_path_sphere_gives_the_level_one():
    phi = np.arange(-90.0, 90.0 + 1e-9, 10.0)
    theta = np.arange(0.0, 180.0 + 1e-9, 5.0)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    gamma = 9.0
    filed = (60.0 + 10.0 * _level_down(phi_grid, theta_grid, gamma))[:, :, None]
    level, measured = repitch_art_levels(filed, phi, theta, gamma)
    truth = 60.0 + 10.0 * np.sin(np.radians(theta_grid)) * np.cos(np.radians(phi_grid))
    inside = measured[:, :, 0]
    np.testing.assert_allclose(level[:, :, 0][inside], truth[inside], atol=0.1)
    # The climbing sphere's equator rises ahead and falls aft, so the level cells it
    # does not reach (aft, within gamma below the horizon) take the level of its
    # nearest equator point and are not measured; the rest are, ahead near the
    # horizon included.  That fill is off by no more than the pitch's worth of pattern.
    assert not inside[phi == 0.0, theta == 175.0][0]
    assert inside[phi == 0.0, theta == 5.0][0]
    assert inside[phi == 0.0, theta == 90.0][0]
    assert inside.sum() > 0.8 * inside.size
    assert np.max(np.abs(level[:, :, 0] - truth)) < 10.0 * np.sin(np.radians(gamma)) + 0.1
    # A zero pitch is the identity, missing cells included.
    filed[0, 0, 0] = -np.inf
    same, same_measured = repitch_art_levels(filed, phi, theta, 0.0)
    np.testing.assert_array_equal(same, filed)
    assert not same_measured[0, 0, 0] and same_measured[1:].all()


def _build(directory, database, frame, **options):
    sphere_helpers.write_sphere_directory(directory, [(_down_level_hemisphere(), 60.0, 0.0),
                                                      (_down_level_hemisphere(), 80.0, -6.0)],
                                          sphere_frame=frame)
    fa.build_empirical_database(str(directory), str(database), load_factors=None, **options)
    with Dataset(str(database)) as handle:
        assert handle.sphere_frame == 'level'
        for name, group in handle.groups.items():
            if (group.variables['flight_path_angle'][0] == -6.0
                    and group.condition_origin == 'measured'):
                return (np.array(group.variables['amplitude'][:]), np.array(group.variables['coverage'][:]),
                        np.array(group.variables['dBA'][:]))
    raise AssertionError('no -6 deg group')


def test_the_builder_pitches_flight_path_spheres_to_level(tmp_path):
    # The same hemisphere filed level and in AAM's frame builds the same database,
    # but for one more resampling and the rim the flight-path sphere does not cover.
    # The resampling costs 0.04 dB on average and 0.7 at most, beside the horizon,
    # across the 10 deg phi rows; read level, the flight-path file would be
    # 10 sin 6 = 1.05 dB off at the nose and off on average too.
    level_amplitude, level_coverage, _ = _build(tmp_path / 'level', tmp_path / 'level.nod', 'level')
    path_amplitude, path_coverage, _ = _build(tmp_path / 'path', tmp_path / 'path.nod', 'flight_path')
    covered = path_coverage.astype(bool) & level_coverage.astype(bool)
    assert covered.sum() > 0.4 * covered.size     # the lower half of the completed sphere, nearly
    error = np.abs(path_amplitude[covered] - level_amplitude[covered])
    assert error.max() < 0.75 and error.mean() < 0.1
    assert not (path_coverage.astype(bool) & ~level_coverage.astype(bool)).any()
    assert (level_coverage.astype(bool) & ~path_coverage.astype(bool)).any()


def test_spheres_that_record_no_frame_are_taken_as_level_with_a_warning(tmp_path):
    directory = sphere_helpers.write_sphere_directory(tmp_path / 's', [(_down_level_hemisphere(), 60.0, 0.0)])
    for path in directory.glob('*.nc'):
        with Dataset(str(path), 'a') as handle:
            handle.delncattr('sphere_frame')
    with pytest.warns(UserWarning, match='carry no sphere_frame'):
        fa.build_empirical_database(str(directory), str(tmp_path / 'a.nod'), load_factors=None)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        fa.build_empirical_database(str(directory), str(tmp_path / 'b.nod'), load_factors=None,
                                    sphere_frame='level', azimuth_reference='track')
    # A sphere that records its frame must agree with the one the build is given.
    fa.write_aam_hemisphere_netcdf(str(directory / 'X102.nc'), _down_level_hemisphere(), mode='third_octave',
                                   radius_ft=100.0, speed_knots=80.0, flight_path_angle_deg=-6.0)
    with pytest.raises(ValueError, match="carries sphere_frame = 'flight_path'"):
        fa.build_empirical_database(str(directory), str(tmp_path / 'c.nod'), load_factors=None,
                                    sphere_frame='level')


def test_spheres_in_different_frames_are_not_averaged(tmp_path):
    paths = []
    for name, frame in (('X101.nc', 'level'), ('X102.nc', 'flight_path')):
        paths.append(str(tmp_path / name))
        fa.write_aam_hemisphere_netcdf(paths[-1], _down_level_hemisphere(), mode='third_octave', radius_ft=100.0,
                                       speed_knots=60.0, flight_path_angle_deg=-6.0, sphere_frame=frame)
    with pytest.raises(ValueError, match='sphere_frame differs'):
        sphere_statistics.average_spheres(paths)
    average = sphere_statistics.average_spheres([paths[1], paths[1]])
    assert average['sphere_frame'] == 'flight_path'
    sphere_statistics.write_averaged_sphere(str(tmp_path / 'mean.nc'), average)
    assert read_sphere_frame(str(tmp_path / 'mean.nc')) == 'flight_path'


def test_the_norah2_converter_pitches_a_flight_path_sphere_to_level(tmp_path):
    # NORAH2 is level: the hemisphere from AAM's frame matches the one from the level
    # file, where read level it would be 10 sin 8 = 1.4 dB loud at the nose.
    from panam.acoustics.atmosphere import Atmosphere
    from tools.sphere_to_norah2 import sphere_to_norah2
    atmosphere = Atmosphere(temperature=293.15, pressure=101.325, relative_humidity=50.0)
    levels = {}
    for frame in ('level', 'flight_path'):
        sphere = tmp_path / '{}.nc'.format(frame)
        fa.write_aam_hemisphere_netcdf(str(sphere), _down_level_hemisphere(), mode='third_octave',
                                       radius_ft=100.0, speed_knots=60.0, flight_path_angle_deg=-8.0,
                                       sphere_frame=frame)
        levels[frame] = sphere_to_norah2(str(sphere), str(tmp_path / '{}.hem'.format(frame)),
                                         measurement_atmosphere=atmosphere)
    both = np.isfinite(levels['level']) & np.isfinite(levels['flight_path'])
    assert both.any(axis=2).sum() > 0.8 * both.shape[0] * both.shape[1]     # two of its 31 bands
    error = np.abs(levels['flight_path'] - levels['level'])[both]
    assert error.max() < 0.75 and error.mean() < 0.1
