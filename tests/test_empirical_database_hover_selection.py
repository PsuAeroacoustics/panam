"""Hover-sphere selection in build_empirical_database.

The hover sphere is synthesized by averaging the slowest near-level sphere
fore-to-aft.  When no sphere falls within the level-flight tolerance there is
nothing to average, and the selection used to leave `min_speed_file` unbound --
the build then died with an UnboundLocalError from deep inside the writer
rather than saying what was wrong with the inputs.
"""

import os
import shutil
import warnings

import pytest

from flight_acoustics import build_empirical_database


SPHERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'example_data', 'AS350B3108.nc')

VEHICLE_CFG = """[Main Rotor]
blades = 3
radius = 5.345
tip speed = 218.0
[Tail Rotor]
blades = 2
radius = 0.93
tip speed = 202.0
[Atmosphere]
temperature = 288.15
density = 1.225
[Vehicle]
weight = 2250.0
drag = 1.0
"""


def _source_directory(tmp_path, spheres):
    directory = tmp_path / 'source'
    directory.mkdir()
    (directory / 'vehicle.cfg').write_text(VEHICLE_CFG)
    for index in range(spheres):
        shutil.copy(SPHERE, directory / 'sphere{}.nc'.format(index))
    return directory


def test_empty_source_directory_names_the_problem(tmp_path):
    directory = _source_directory(tmp_path, spheres=0)
    with pytest.raises(ValueError, match='No sphere files found'):
        build_empirical_database(str(directory), str(tmp_path / 'database.nc'))


def test_falls_back_to_the_slowest_sphere_when_none_is_level(tmp_path):
    directory = _source_directory(tmp_path, spheres=1)
    database = tmp_path / 'database.nc'
    # A zero tolerance admits nothing, which is the condition being guarded.
    with pytest.warns(UserWarning, match='level flight'):
        build_empirical_database(str(directory), str(database),
                                 load_factors=[1.0], level_flight_tolerance=0.0)
    assert database.exists()


def test_no_warning_when_a_level_sphere_exists(tmp_path):
    directory = _source_directory(tmp_path, spheres=1)
    database = tmp_path / 'database.nc'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        build_empirical_database(str(directory), str(database),
                                 load_factors=[1.0], level_flight_tolerance=90.0)
    assert not [w for w in caught if 'level flight' in str(w.message)]
    assert database.exists()
