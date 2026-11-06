"""A sphere at SPEED 0 is the hover: no hover is synthesized beside it.

A PSU-WOPWOP results folder can hold a real hover (a sphere predicted at zero
speed).  The builder takes it as the hover, written at the hover's flight path
angles as the synthesized one would be, rather than synthesizing a second hover
at the same condition from it; validate_nod refuses two groups at one
(advance ratio, flight path angle, thrust coefficient).
"""

import numpy as np
import pytest
from netCDF4 import Dataset

from panam import flight_acoustics as fa
from panam.nod_database import validate_nod

import sphere_helpers

PHI = np.arange(-90.0, 90.0 + 1e-9, 45.0)
THETA = np.arange(0.0, 180.0 + 1e-9, 30.0)
BANDS = np.array([100.0, 125.0, 160.0])


def _levels(speed):
    # Louder ahead than behind: a fore-aft average would change it.
    theta, _ = np.meshgrid(THETA, PHI)
    level = 70.0 + 0.05 * speed + 4.0 * np.cos(np.radians(theta))
    return np.repeat(level[:, :, None], BANDS.size, axis=2)


def _folder(directory, speeds=(0.0, 40.0, 60.0)):
    directory.mkdir()
    (directory / 'vehicle.cfg').write_text(sphere_helpers.VEHICLE_CFG)
    for i, speed in enumerate(speeds):
        path = directory / f'X{101 + i:03d}.nc'
        sphere_helpers.write_raw_sphere(path, PHI, THETA, BANDS, _levels(speed), RADIUS=100.0, SPEED=speed,
                                        FLIGHT_PATH_ANGLE=0.0, DOPPLER_SHIFT_REMOVED=0.0)
        with Dataset(str(path), 'a') as handle:
            handle.sphere_frame = 'level'
    return directory


def _groups(database):
    with Dataset(str(database)) as handle:
        return [dict(mu=float(g['advance_ratio'][0]), gamma=float(g['flight_path_angle'][0]),
                     origin=g.condition_origin, source=g.source_sphere,
                     amplitude=np.array(g['amplitude'][:])) for g in handle.groups.values()]


def _build(directory, database, **options):
    fa.build_empirical_database(str(directory), str(database), load_factors=None, **options)
    return database


def test_a_real_hover_is_the_hover(tmp_path):
    database = _build(_folder(tmp_path / 's'), tmp_path / 'x.nod')
    groups = _groups(database)
    assert not [g for g in groups if g['origin'] == 'synthesized_hover']
    hovers = [g for g in groups if g['mu'] == 0.0]
    assert sorted(g['gamma'] for g in hovers) == [-12.0, 0.0, 12.0]
    for g in hovers:
        assert g['source'] == 'X101.nc'
        # The file's own levels, not a fore-aft average of them (the lower
        # hemisphere completed onto the upper by the usual mirror).
        np.testing.assert_allclose(g['amplitude'][:PHI.size][np.argsort(PHI)].min(), _levels(0.0).min(),
                                   atol=1e-4)
        np.testing.assert_allclose(g['amplitude'].max(), _levels(0.0).max(), atol=1e-4)
    assert {g['origin'] for g in hovers} == {'measured', 'extended_flight_path_angle'}
    assert [g['origin'] for g in hovers if g['gamma'] == 0.0] == ['measured']
    assert validate_nod(str(database)) == []


def test_a_real_hover_takes_the_extended_angles(tmp_path):
    database = _build(_folder(tmp_path / 's'), tmp_path / 'x.nod', extended_flight_path_angles=(-24.0, 35.0))
    hovers = [g for g in _groups(database) if g['mu'] == 0.0]
    assert sorted(g['gamma'] for g in hovers) == [-24.0, 0.0, 35.0]
    assert validate_nod(str(database)) == []


def test_a_hover_source_beside_a_real_hover_is_refused(tmp_path):
    directory = _folder(tmp_path / 's')
    with pytest.raises(ValueError, match='hover'):
        _build(directory, tmp_path / 'x.nod', hover_source=str(directory / 'X102.nc'))


def test_without_a_real_hover_the_hover_is_synthesized(tmp_path):
    groups = _groups(_build(_folder(tmp_path / 's', (40.0, 60.0)), tmp_path / 'x.nod'))
    assert {g['origin'] for g in groups if g['mu'] == 0.0} == {'synthesized_hover'}


def test_validate_nod_refuses_two_groups_at_one_condition(tmp_path):
    database = _build(_folder(tmp_path / 's', (40.0, 60.0)), tmp_path / 'x.nod')
    with Dataset(str(database), 'a') as handle:
        first, second = list(handle.groups.values())[:2]
        for name in ('advance_ratio', 'flight_path_angle', 'thrust_coefficient'):
            second[name][:] = first[name][:]
    assert [v.rule for v in validate_nod(str(database))] == ['C14']
