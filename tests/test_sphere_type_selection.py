"""Sphere-type selection: a database is built from spheres of one type.

A PSU-WOPWOP results folder holds a sphere of each type it was asked for (BB,
OB12, NB, PT) at every condition, each file labeled with a ``sphere_type``
attribute as well as AAM's BB/NB/PT flags.  The builder takes the BB spheres by
default and leaves the others out; a database never mixes types.
"""

import numpy as np
import pytest
from netCDF4 import Dataset

from panam import flight_acoustics as fa
from panam.sphere_io import read_sphere_type

import sphere_helpers

PHI = np.arange(-90.0, 90.0 + 1e-9, 45.0)
THETA = np.arange(0.0, 180.0 + 1e-9, 30.0)
BANDS = np.array([100.0, 125.0, 160.0])
FLAGS = {'BB': (1.0, 0.0, 0.0), 'OB12': (0.0, 1.0, 0.0), 'NB': (0.0, 1.0, 0.0), 'PT': (0.0, 0.0, 1.0)}


def _write(path, kind=None, flags=None, speed=60.0):
    """A small sphere; kind is its sphere_type attribute (None: none), flags its
    (BB, NB, PT) variables (None: those of kind, or none)."""
    levels = np.full((PHI.size, THETA.size, BANDS.size), 70.0)
    scalars = dict(RADIUS=100.0, SPEED=speed, FLIGHT_PATH_ANGLE=0.0, DOPPLER_SHIFT_REMOVED=0.0)
    if flags is None and kind is not None:
        flags = FLAGS[kind]
    if flags is not None:
        scalars.update(BB=flags[0], NB=flags[1], PT=flags[2])
    sphere_helpers.write_raw_sphere(path, PHI, THETA, BANDS, levels, **scalars)
    if kind is not None:
        with Dataset(str(path), 'a') as handle:
            handle.sphere_type = kind
    return path


def _results_folder(directory, kinds=('BB', 'OB12', 'NB', 'PT')):
    """A WOPWOP-like folder: one sphere of each type at each of two speeds."""
    directory.mkdir()
    (directory / 'vehicle.cfg').write_text(sphere_helpers.VEHICLE_CFG)
    index = 0
    for speed in (60.0, 80.0):
        for kind in kinds:
            _write(directory / f'X{index:03d}.nc', kind, speed=speed)
            index += 1
    return directory


@pytest.mark.parametrize('kind', ['BB', 'OB12', 'NB', 'PT'])
def test_the_attribute_names_the_type(tmp_path, kind):
    assert read_sphere_type(str(_write(tmp_path / 'a.nc', kind))) == kind


@pytest.mark.parametrize('flags, kind', [((1.0, 0.0, 0.0), 'BB'), ((0.0, 1.0, 0.0), 'NB'),
                                         ((0.0, 0.0, 1.0), 'PT'), ((0.0, 0.0, 0.0), 'BB')])
def test_without_the_attribute_the_flags_name_the_type(tmp_path, flags, kind):
    assert read_sphere_type(str(_write(tmp_path / 'a.nc', flags=flags))) == kind


def test_a_sphere_without_type_or_flags_is_broadband(tmp_path):
    # Every sphere PANAM measured before the attribute existed.
    assert read_sphere_type(str(_write(tmp_path / 'a.nc'))) == 'BB'


def test_flags_that_contradict_the_attribute_are_refused(tmp_path):
    path = _write(tmp_path / 'a.nc', 'BB', flags=(0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match='sphere_type'):
        read_sphere_type(str(path))


def test_two_flags_set_are_refused(tmp_path):
    path = _write(tmp_path / 'a.nc', flags=(1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match='flags'):
        read_sphere_type(str(path))


def test_an_unknown_type_is_refused(tmp_path):
    path = _write(tmp_path / 'a.nc', 'OB3', flags=(1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match='OB3'):
        read_sphere_type(str(path))


def _source_spheres(database):
    with Dataset(str(database)) as handle:
        return str(handle.source_spheres).split()


def test_a_results_folder_builds_from_its_broadband_spheres(tmp_path):
    directory = _results_folder(tmp_path / 'spheres')
    database = tmp_path / 'bb.nod'
    fa.build_empirical_database(str(directory), str(database), load_factors=None)
    assert _source_spheres(database) == ['X000.nc', 'X004.nc']


def test_another_type_is_chosen_by_name(tmp_path):
    directory = _results_folder(tmp_path / 'spheres')
    database = tmp_path / 'ob12.nod'
    fa.build_empirical_database(str(directory), str(database), load_factors=None, sphere_type='OB12')
    assert _source_spheres(database) == ['X001.nc', 'X005.nc']


def test_no_sphere_of_the_type_is_an_error_naming_the_others(tmp_path):
    directory = _results_folder(tmp_path / 'spheres', kinds=('NB', 'PT'))
    with pytest.raises(ValueError, match='no BB sphere.*NB.*PT'):
        fa.build_empirical_database(str(directory), str(tmp_path / 'x.nod'), load_factors=None)


def test_any_type_refuses_a_mixed_folder(tmp_path):
    directory = _results_folder(tmp_path / 'spheres', kinds=('BB', 'PT'))
    with pytest.raises(ValueError, match='one sphere type'):
        fa.build_empirical_database(str(directory), str(tmp_path / 'x.nod'), load_factors=None,
                                    sphere_type=None)


def test_any_type_takes_a_folder_of_one_type(tmp_path):
    directory = _results_folder(tmp_path / 'spheres', kinds=('OB12',))
    database = tmp_path / 'any.nod'
    fa.build_empirical_database(str(directory), str(database), load_factors=None, sphere_type=None)
    assert _source_spheres(database) == ['X000.nc', 'X001.nc']


def test_a_named_sphere_of_another_type_is_refused(tmp_path):
    # A file the caller names is not silently dropped.
    directory = _results_folder(tmp_path / 'spheres')
    with pytest.raises(ValueError, match='X003.nc.*PT'):
        fa.build_empirical_database(str(directory), str(tmp_path / 'x.nod'), load_factors=None,
                                    spheres=['X000.nc', 'X003.nc', 'X004.nc'])


def test_an_unknown_type_to_build_is_refused(tmp_path):
    directory = _results_folder(tmp_path / 'spheres')
    with pytest.raises(ValueError, match='sphere_type'):
        fa.build_empirical_database(str(directory), str(tmp_path / 'x.nod'), load_factors=None,
                                    sphere_type='OB3')
