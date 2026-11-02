"""Full spheres (phi -180..180) whose seam row is stored twice, as PSU-WOPWOP writes them.

A full sphere stores phi = 180 as a copy of phi = -180, the same azimuth (the
upper meridian).  The database keeps phi in [-180, 180) (validate_nod C13), so
the builder drops the copy, and refuses a sphere whose two seam rows differ,
which would be two levels for one direction.  A full sphere is measured all
round, so nothing is mirrored onto its upper half.
"""

import numpy as np
import pytest
from netCDF4 import Dataset

from panam import flight_acoustics as fa

import sphere_helpers

PHI = np.arange(-180.0, 180.0 + 1e-9, 45.0)
THETA = np.arange(0.0, 180.0 + 1e-9, 30.0)
BANDS = np.array([100.0, 125.0, 160.0])


def _levels(theta_shift=0.0, phi=PHI):
    """A pattern that varies with both angles and is not symmetric in phi, the
    seam row a copy of phi = -180."""
    theta, phi = np.meshgrid(THETA, phi)
    level = (70.0 + 5.0 * np.cos(np.radians(theta + theta_shift))
             + 3.0 * np.sin(np.radians(theta)) * np.cos(np.radians(phi))
             + 2.0 * np.sin(np.radians(theta)) * np.sin(np.radians(phi)))
    levels = np.repeat(level[:, :, None], BANDS.size, axis=2) + np.arange(BANDS.size)
    levels[-1] = levels[0]          # sin(phi) rounds differently at -180 and 180
    return levels


def _write_full_sphere(path, speed, angle, frame, levels, phi=PHI):
    sphere_helpers.write_raw_sphere(path, phi, THETA, BANDS, levels, RADIUS=100.0, SPEED=speed,
                                    FLIGHT_PATH_ANGLE=angle, DOPPLER_SHIFT_REMOVED=0.0)
    with Dataset(str(path), 'a') as handle:
        handle.sphere_frame = frame
        handle.azimuth_reference = 'track'


def _full_sphere_directory(directory, frame, seam_differs=False, phi=PHI,
                           conditions=((60.0, 0.0), (80.0, -6.0))):
    directory.mkdir()
    (directory / 'vehicle.cfg').write_text(sphere_helpers.VEHICLE_CFG)
    for i, (speed, angle) in enumerate(conditions):
        levels = _levels(10.0 * i, phi)
        if seam_differs and i == 1:
            levels[-1] += 1.0
        _write_full_sphere(directory / f'X{101 + i:03d}.nc', speed, angle, frame, levels, phi)
    return directory


def _measured_groups(handle):
    return [g for g in handle.groups.values() if g.condition_origin == 'measured']


@pytest.mark.parametrize('frame', ['level', 'flight_path'])
def test_a_full_sphere_keeps_one_seam_row(tmp_path, frame):
    directory = _full_sphere_directory(tmp_path / 'spheres', frame)
    database = tmp_path / 'full.nod'
    fa.build_empirical_database(str(directory), str(database), load_factors=None)
    with Dataset(str(database)) as handle:
        # The database's grid holds every direction of the circle once.
        phi = np.unique(np.round(np.array(handle['phi'][:]), 6))
        np.testing.assert_allclose(phi, PHI[:-1])
        measured = _measured_groups(handle)
        assert len(measured) == 2
        for group in measured:
            assert group.variables['amplitude'].shape[:2] == (PHI.size - 1, THETA.size)
            # A full sphere is measured all round, the upper half too.
            assert np.all(np.array(group.variables['coverage'][:]) == 1)


def test_a_level_full_sphere_keeps_its_levels(tmp_path):
    # Dropping the copy loses nothing: the rows are the file's from phi = -180 on.
    directory = _full_sphere_directory(tmp_path / 'spheres', 'level')
    database = tmp_path / 'full.nod'
    fa.build_empirical_database(str(directory), str(database), load_factors=None)
    with Dataset(str(database)) as handle:
        group = next(g for g in _measured_groups(handle) if float(g['flight_path_angle'][0]) == 0.0)
        np.testing.assert_allclose(np.array(group.variables['amplitude'][:]), _levels(0.0)[:-1], atol=1e-4)


def test_a_full_sphere_on_a_grid_its_mirror_misses_is_not_mirrored(tmp_path):
    # At 40 degrees the reflection 180 - phi of most rows is not a row of the
    # grid; a full sphere still keeps its own rows only, all measured.
    phi = np.arange(-180.0, 180.0 + 1e-9, 40.0)
    directory = _full_sphere_directory(tmp_path / 'spheres', 'level', phi=phi)
    database = tmp_path / 'full.nod'
    fa.build_empirical_database(str(directory), str(database), load_factors=None)
    with Dataset(str(database)) as handle:
        np.testing.assert_allclose(np.unique(np.round(np.array(handle['phi'][:]), 6)), phi[:-1])
        for group in _measured_groups(handle):
            assert group.variables['amplitude'].shape[:2] == (phi.size - 1, THETA.size)
            assert np.all(np.array(group.variables['coverage'][:]) == 1)


def test_seam_rows_missing_in_different_conventions_are_one_row(tmp_path):
    # -999 at phi = -180 and NaN at phi = 180 are the same missing cell.
    directory = tmp_path / 'spheres'
    directory.mkdir()
    (directory / 'vehicle.cfg').write_text(sphere_helpers.VEHICLE_CFG)
    levels = _levels()
    levels[0, 2, 1] = -999.0
    levels[-1, 2, 1] = np.nan
    _write_full_sphere(directory / 'X101.nc', 60.0, 0.0, 'level', levels)
    database = tmp_path / 'full.nod'
    fa.build_empirical_database(str(directory), str(database), load_factors=None)
    with Dataset(str(database)) as handle:
        (group,) = _measured_groups(handle)
        assert group.variables['amplitude'].shape[:2] == (PHI.size - 1, THETA.size)
        assert not np.array(group.variables['coverage'][:])[0, 2, 1]


def test_full_spheres_averaged_at_one_condition_keep_one_seam_row(tmp_path):
    # Two runs of one condition are averaged before the seam is dropped.
    directory = _full_sphere_directory(tmp_path / 'spheres', 'level',
                                       conditions=((60.0, 0.0), (60.0, 0.0)))
    database = tmp_path / 'full.nod'
    fa.build_empirical_database(str(directory), str(database), load_factors=None, merge_repeats=True)
    with Dataset(str(database)) as handle:
        (group,) = _measured_groups(handle)
        assert group.variables['amplitude'].shape[:2] == (PHI.size - 1, THETA.size)
        assert np.all(np.array(group.variables['coverage'][:]) == 1)


def test_a_full_sphere_whose_seam_rows_differ_is_refused(tmp_path):
    directory = _full_sphere_directory(tmp_path / 'spheres', 'level', seam_differs=True)
    with pytest.raises(ValueError, match='phi = 180'):
        fa.build_empirical_database(str(directory), str(tmp_path / 'full.nod'), load_factors=None)
