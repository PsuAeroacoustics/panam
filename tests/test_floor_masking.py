"""Cells at a sphere's floor are no energy in the database.

PSU-WOPWOP raises a sphere's empty cells to one floor level per file and records
it (floor_level_dB).  Left as levels, NICEOPS would blend them in dB with real
ones; read as missing, it drops them and renormalises, as it does for a gated
direction.  So the builder masks them by default (floor_masking='auto'), can
take each file's lowest level as its floor ('minimum'), or keep them ('off').
"""

import numpy as np
import pytest
from netCDF4 import Dataset

from panam import flight_acoustics as fa
from tools import build_empirical_database as script

import sphere_helpers

PHI = np.arange(-90.0, 90.0 + 1e-9, 45.0)
THETA = np.arange(0.0, 180.0 + 1e-9, 30.0)
BANDS = np.array([100.0, 125.0, 160.0, 200.0])
FLOOR = -33.9748976


def _levels(speed, floor=FLOOR):
    theta, phi = np.meshgrid(THETA, PHI)
    level = 70.0 + 0.05 * speed + 4.0 * np.cos(np.radians(theta)) + np.sin(np.radians(phi))
    levels = np.repeat(level[:, :, None], BANDS.size, axis=2) - 3.0 * np.arange(BANDS.size)
    # The top band is empty ahead and the two lower bands at the poles: at the floor.
    levels[:, :2, -1] = floor
    levels[:, -1, :2] = floor
    return levels


def _folder(directory, floor=FLOOR, attribute=True, speeds=(40.0, 60.0)):
    """Spheres whose empty cells hold ``floor`` (-999: written as missing)."""
    directory.mkdir()
    (directory / 'vehicle.cfg').write_text(sphere_helpers.VEHICLE_CFG)
    for i, speed in enumerate(speeds):
        path = directory / f'X{101 + i:03d}.nc'
        sphere_helpers.write_raw_sphere(path, PHI, THETA, BANDS, _levels(speed, floor), RADIUS=100.0,
                                        SPEED=speed, FLIGHT_PATH_ANGLE=0.0, DOPPLER_SHIFT_REMOVED=0.0)
        with Dataset(str(path), 'a') as handle:
            handle.sphere_frame = 'level'
            if attribute:
                handle.floor_level_dB = '{:.7f}'.format(FLOOR)
    return directory


def _contents(database):
    with Dataset(str(database)) as handle:
        return [(np.array(g['dBA'][:]), np.array(g['EAA'][:]), np.array(g['amplitude'][:]),
                 np.array(g['coverage'][:])) for g in handle.groups.values()]


def _build(directory, database, **options):
    fa.build_empirical_database(str(directory), str(database), load_factors=None, **options)
    return _contents(database)


def _same(a, b):
    for group_a, group_b in zip(a, b, strict=True):
        for x, y in zip(group_a, group_b):
            np.testing.assert_array_equal(x, y)


def test_floor_cells_build_as_missing_ones(tmp_path):
    masked = _build(_folder(tmp_path / 'floor'), tmp_path / 'floor.nod')
    missing = _build(_folder(tmp_path / 'missing', floor=-999.0), tmp_path / 'missing.nod')
    _same(masked, missing)


def test_the_lowest_level_can_be_taken_as_the_floor(tmp_path):
    masked = _build(_folder(tmp_path / 'floor', attribute=False), tmp_path / 'floor.nod',
                    floor_masking='minimum')
    missing = _build(_folder(tmp_path / 'missing', floor=-999.0, attribute=False), tmp_path / 'missing.nod')
    _same(masked, missing)


def test_floor_cells_can_be_kept(tmp_path):
    kept = _build(_folder(tmp_path / 'floor'), tmp_path / 'floor.nod', floor_masking='off')
    unlabelled = _build(_folder(tmp_path / 'plain', attribute=False), tmp_path / 'plain.nod')
    _same(kept, unlabelled)
    assert np.isclose(kept[0][2], FLOOR, atol=1e-4).any()


def test_averaged_repeats_mask_their_floors(tmp_path):
    options = dict(merge_repeats=True)
    masked = _build(_folder(tmp_path / 'floor', speeds=(40.0, 40.0)), tmp_path / 'floor.nod', **options)
    missing = _build(_folder(tmp_path / 'missing', floor=-999.0, speeds=(40.0, 40.0)), tmp_path / 'missing.nod',
                     **options)
    _same(masked, missing)


def test_an_unknown_masking_is_refused(tmp_path):
    with pytest.raises(ValueError, match='floor_masking'):
        _build(_folder(tmp_path / 'floor'), tmp_path / 'x.nod', floor_masking='lowest')


@pytest.mark.parametrize('value', ['auto', 'minimum', 'off'])
def test_the_cli_takes_floor_masking(tmp_path, value):
    args = script.parse_arguments([str(tmp_path), str(tmp_path / 'db.nod'), '--floor-masking', value])
    assert script.build_options(args)['floor_masking'] == value
