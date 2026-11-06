"""A PSU-WOPWOP sphere's labels carried into the database, and the checks they allow.

WOPWOP labels each sphere: its type, content and source tag, the radius it was
predicted at (PREDICTION_RADIUS), the floor its empty cells were raised to, the
sound speed, the narrowband width, its center (XYZ) and that its SPEED is an
airspeed (speed_reference = "air").  The builder keeps them as group attributes
and root values, and refuses what a database cannot represent.  A sphere
without them (every one PANAM measured) builds as before.
"""

import warnings

import numpy as np
import pytest
from netCDF4 import Dataset

from panam import flight_acoustics as fa
from tools import build_empirical_database as script

import sphere_helpers

PHI = np.arange(-90.0, 90.0 + 1e-9, 45.0)
THETA = np.arange(0.0, 180.0 + 1e-9, 30.0)
BANDS = {'BB': np.array([100.0, 125.0, 160.0]), 'OB12': np.array([100.0, 105.9, 112.2]),
         'NB': np.array([10.0, 20.0, 30.0])}
# example_data/vehicle.cfg's main rotor radius, m.
ROTOR_RADIUS = 5.334

LABELS = dict(content='discrete frequency and broadband', source_tag='MR', floor_level_dB='-12.5',
              speed_of_sound_m_s='340.29', speed_reference='air', azimuth_reference='track')


def _write(path, kind='BB', speed=60.0, prediction_radius=300.0, xyz=(0.0, 0.0, 0.0), labels=True):
    bands = BANDS[kind]
    levels = np.full((PHI.size, THETA.size, bands.size), 70.0) - 0.1 * np.arange(THETA.size)[None, :, None]
    sphere_helpers.write_raw_sphere(path, PHI, THETA, bands, levels, RADIUS=100.0, SPEED=speed,
                                    FLIGHT_PATH_ANGLE=0.0, DOPPLER_SHIFT_REMOVED=0.0)
    with Dataset(str(path), 'a') as handle:
        handle.sphere_frame = 'level'
        handle.createDimension('XYZ', 3)
        handle.createVariable('XYZ', 'f4', ('XYZ',))[:] = xyz
        if labels:
            handle.sphere_type = kind
            for name, value in LABELS.items():
                handle.setncattr(name, value)
            if kind == 'NB':
                handle.nb_bandwidth_Hz = '10.0'
            variable = handle.createVariable('PREDICTION_RADIUS', 'f8')
            variable.assignValue(prediction_radius)
            variable.unit = 'METER'
    return path


def _folder(directory, **options):
    directory.mkdir()
    (directory / 'vehicle.cfg').write_text(sphere_helpers.VEHICLE_CFG)
    for i, speed in enumerate((40.0, 60.0)):
        _write(directory / f'X{101 + i:03d}.nc', speed=speed, **options)
    return directory


def _build(directory, database, **options):
    fa.build_empirical_database(str(directory), str(database), load_factors=None, **options)
    return database


def _measured(handle):
    return [g for g in handle.groups.values() if g.condition_origin == 'measured']


def test_the_labels_reach_every_group(tmp_path):
    database = _build(_folder(tmp_path / 'spheres'), tmp_path / 'x.nod')
    with Dataset(str(database)) as handle:
        for group in handle.groups.values():
            assert group.sphere_type == 'BB'
            assert group.content == 'discrete frequency and broadband'
            assert group.source_tag == 'MR'
            assert float(group.prediction_radius) == 300.0 and group.prediction_radius_units == 'm'
            assert float(group.floor_level) == -12.5 and group.floor_level_units == 'dB'
            assert float(group.speed_of_sound) == pytest.approx(340.29)
            assert group.speed_of_sound_units == 'm s-1'


def test_the_root_records_band_type_prediction_radius_and_air_speeds(tmp_path):
    database = _build(_folder(tmp_path / 'spheres'), tmp_path / 'x.nod')
    with Dataset(str(database)) as handle:
        assert handle.band_type == 'third_octave'
        assert float(handle['prediction_radius_meters'][...]) == 300.0
        # WOPWOP's SPEED is the airspeed: the database says so without being told.
        assert handle.speed_reference == 'air'


def test_a_narrowband_sphere_records_its_bandwidth(tmp_path):
    directory = tmp_path / 'spheres'
    directory.mkdir()
    (directory / 'vehicle.cfg').write_text(sphere_helpers.VEHICLE_CFG)
    for i, speed in enumerate((40.0, 60.0)):
        _write(directory / f'X{101 + i:03d}.nc', kind='NB', speed=speed)
    database = _build(directory, tmp_path / 'x.nod', sphere_type='NB')
    with Dataset(str(database)) as handle:
        assert handle.band_type == 'narrowband'
        for group in handle.groups.values():
            assert float(group.nb_bandwidth) == 10.0 and group.nb_bandwidth_units == 'Hz'


def test_unlabelled_spheres_build_as_before(tmp_path):
    database = _build(_folder(tmp_path / 'spheres', labels=False), tmp_path / 'x.nod')
    with Dataset(str(database)) as handle:
        assert handle.band_type == 'third_octave'
        assert handle.speed_reference == 'ground'
        assert 'prediction_radius_meters' not in handle.variables
        for group in handle.groups.values():
            assert not {'sphere_type', 'content', 'source_tag', 'prediction_radius', 'floor_level',
                        'speed_of_sound', 'nb_bandwidth', 'sphere_center_xyz_ft'} & set(group.ncattrs())


def test_a_speed_reference_the_spheres_contradict_is_refused(tmp_path):
    with pytest.raises(ValueError, match='speed_reference'):
        _build(_folder(tmp_path / 'spheres'), tmp_path / 'x.nod', speed_reference='ground')


def test_a_sphere_off_the_vehicle_origin_is_refused(tmp_path):
    # A database holds one source at the vehicle's origin.
    directory = _folder(tmp_path / 'spheres', xyz=(-12.0, 0.0, 7.0))
    with pytest.raises(ValueError, match='XYZ.*ignore_sphere_center'):
        _build(directory, tmp_path / 'x.nod')


def test_ignore_sphere_center_builds_and_records_it(tmp_path):
    directory = _folder(tmp_path / 'spheres', xyz=(-12.0, 0.0, 7.0))
    database = _build(directory, tmp_path / 'x.nod', ignore_sphere_center=True)
    with Dataset(str(database)) as handle:
        for group in _measured(handle):
            assert group.sphere_center_xyz_ft == '-12 0 7'


def test_a_band_method_for_one_third_octaves_is_refused_for_others(tmp_path):
    directory = tmp_path / 'spheres'
    directory.mkdir()
    (directory / 'vehicle.cfg').write_text(sphere_helpers.VEHICLE_CFG)
    for i, speed in enumerate((40.0, 60.0)):
        _write(directory / f'X{101 + i:03d}.nc', kind='OB12', speed=speed)
    with pytest.raises(ValueError, match='absorption_band_method'):
        _build(directory, tmp_path / 'x.nod', sphere_type='OB12', absorption_band_method='arp5534')
    _build(directory, tmp_path / 'center.nod', sphere_type='OB12')


def test_a_close_prediction_radius_warns(tmp_path):
    # 100 m is under 50 radii of a 5.334 m rotor: not yet the far field.
    with pytest.warns(UserWarning, match='PREDICTION_RADIUS'):
        _build(_folder(tmp_path / 'close', prediction_radius=100.0), tmp_path / 'close.nod')
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='.*PREDICTION_RADIUS.*')
        _build(_folder(tmp_path / 'far', prediction_radius=50.0 * ROTOR_RADIUS + 1.0), tmp_path / 'far.nod')


def test_the_cli_takes_ignore_sphere_center(tmp_path):
    args = script.parse_arguments([str(tmp_path), str(tmp_path / 'db.nod'), '--ignore-sphere-center'])
    assert script.build_options(args)['ignore_sphere_center'] is True


def test_wopwops_default_prediction_radius_does_not_warn(tmp_path):
    # WOPWOP's default is 50 rotor radii, its radius stored in single precision:
    # 55 m over a float32 1.1 m reads as 49.99999989 radii.
    directory = tmp_path / 'spheres'
    directory.mkdir()
    for i, speed in enumerate((40.0, 60.0)):
        path = _write(directory / f'X{101 + i:03d}.nc', speed=speed, prediction_radius=55.0)
        with Dataset(str(path), 'a') as handle:
            for name, value, unit in (('MAIN_ROTOR_RADIUS', 1.1, 'METER'), ('MAIN_ROTOR_TIP_SPEED', 33.05, 'M/S'),
                                      ('GROSS_WEIGHT', 1124.0, 'POUNDS'), ('AIR_DENSITY', 1.07, 'KG/M^3')):
                variable = handle.createVariable(name, 'f4')
                variable.assignValue(value)
                variable.unit = unit
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='.*PREDICTION_RADIUS.*')
        _build(directory, tmp_path / 'x.nod')
