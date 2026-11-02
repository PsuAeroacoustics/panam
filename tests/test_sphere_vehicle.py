"""The vehicle a database is built for, from vehicle.cfg, from the spheres, or both.

PSU-WOPWOP's spheres carry the main rotor and the run's weight and density
(MAIN_ROTOR_RADIUS, MAIN_ROTOR_TIP_SPEED, MAIN_ROTOR_BLADES, GROSS_WEIGHT,
AIR_DENSITY), each with a ``unit`` attribute.  A results folder then needs no
vehicle.cfg; where both exist they must agree on the rotor.
"""

import numpy as np
import pytest
from netCDF4 import Dataset

from panam import flight_acoustics as fa
from panam.sphere_io import POUND_FORCE_NEWTONS, read_sphere_vehicle

import sphere_helpers

PHI = np.arange(-90.0, 90.0 + 1e-9, 45.0)
THETA = np.arange(0.0, 180.0 + 1e-9, 30.0)
BANDS = np.array([100.0, 125.0, 160.0])
G = 9.80665
# example_data/vehicle.cfg: an AS350-like rotor, 2250 kg at 1.070 kg/m^3.
CFG = dict(radius=5.334, tip_speed=230.7, blades=4.0, weight_newtons=2250.0 * G, density=1.070)


def _vehicle_fields(radius=CFG['radius'], tip_speed=CFG['tip_speed'], blades=CFG['blades'],
                    weight_lb=CFG['weight_newtons'] / POUND_FORCE_NEWTONS, density=CFG['density']):
    """WOPWOP's vehicle fields and units: {name: (value, unit)}; a None value is left out."""
    fields = dict(MAIN_ROTOR_RADIUS=(radius, 'METER'), MAIN_ROTOR_TIP_SPEED=(tip_speed, 'M/S'),
                  MAIN_ROTOR_BLADES=(blades, ''), GROSS_WEIGHT=(weight_lb, 'POUNDS'),
                  AIR_DENSITY=(density, 'KG/M^3'))
    return {name: field for name, field in fields.items() if field[0] is not None}


def _write(path, fields, speed=60.0, angle=0.0):
    """A small level sphere carrying fields ({name: (value, unit or None)}) as f8 scalars."""
    levels = np.full((PHI.size, THETA.size, BANDS.size), 70.0) - 0.1 * np.arange(THETA.size)[None, :, None]
    sphere_helpers.write_raw_sphere(path, PHI, THETA, BANDS, levels, RADIUS=100.0, SPEED=speed,
                                    FLIGHT_PATH_ANGLE=angle, DOPPLER_SHIFT_REMOVED=0.0)
    with Dataset(str(path), 'a') as handle:
        handle.sphere_frame = 'level'
        for name, (value, unit) in fields.items():
            variable = handle.createVariable(name, 'f8')
            variable.assignValue(value)
            if unit is not None:
                variable.unit = unit
    return path


def _folder(directory, fields_per_sphere, cfg=True):
    directory.mkdir()
    if cfg:
        (directory / 'vehicle.cfg').write_text(sphere_helpers.VEHICLE_CFG)
    for i, fields in enumerate(fields_per_sphere):
        _write(directory / f'X{101 + i:03d}.nc', fields, speed=40.0 + 20.0 * i)
    return directory


def _build(directory, database, **options):
    fa.build_empirical_database(str(directory), str(database), load_factors=None, **options)
    return database


# ---- reading a sphere's vehicle fields ----

@pytest.mark.parametrize('name, value, unit, key, si', [
    ('MAIN_ROTOR_RADIUS', 5.334, 'METER', 'main_rotor_radius_m', 5.334),
    ('MAIN_ROTOR_RADIUS', 17.5, 'FEET', 'main_rotor_radius_m', 17.5 * 0.3048),
    ('MAIN_ROTOR_TIP_SPEED', 230.7, 'M/S', 'main_rotor_tip_speed_mps', 230.7),
    ('MAIN_ROTOR_TIP_SPEED', 700.0, 'FT/S', 'main_rotor_tip_speed_mps', 700.0 * 0.3048),
    ('MAIN_ROTOR_BLADES', 4.0, '', 'main_rotor_blades', 4.0),
    ('GROSS_WEIGHT', 5000.0, 'POUNDS', 'gross_weight_newtons', 5000.0 * POUND_FORCE_NEWTONS),
    ('GROSS_WEIGHT', 22000.0, 'NEWTON', 'gross_weight_newtons', 22000.0),
    ('AIR_DENSITY', 1.07, 'KG/M^3', 'air_density_kg_m3', 1.07),
])
def test_each_field_reads_in_si(tmp_path, name, value, unit, key, si):
    vehicle = read_sphere_vehicle(str(_write(tmp_path / 'a.nc', {name: (value, unit)})))
    assert vehicle[key] == pytest.approx(si, rel=1e-12)


def test_a_field_without_a_unit_takes_wopwops(tmp_path):
    vehicle = read_sphere_vehicle(str(_write(tmp_path / 'a.nc', {'GROSS_WEIGHT': (5000.0, None)})))
    assert vehicle['gross_weight_newtons'] == pytest.approx(5000.0 * POUND_FORCE_NEWTONS, rel=1e-12)


def test_an_absent_field_is_nan(tmp_path):
    vehicle = read_sphere_vehicle(str(_write(tmp_path / 'a.nc', {})))
    assert all(np.isnan(value) for value in vehicle.values())


def test_an_unknown_unit_is_refused(tmp_path):
    path = _write(tmp_path / 'a.nc', {'MAIN_ROTOR_RADIUS': (5.334, 'FURLONG')})
    with pytest.raises(ValueError, match='MAIN_ROTOR_RADIUS.*FURLONG'):
        read_sphere_vehicle(str(path))


# ---- building with and without vehicle.cfg ----

def _root(database):
    with Dataset(str(database)) as handle:
        return {name: float(handle[name][...]) for name in
                ('main_rotor_radius_meters', 'main_rotor_tip_speed_meters_per_sec', 'vehicle_weight_newtons')}


def _groups(database):
    with Dataset(str(database)) as handle:
        return [{name: np.array(group[name][:]) for name in ('advance_ratio', 'thrust_coefficient', 'rotor_scale')}
                for group in handle.groups.values()]


def test_spheres_alone_give_the_vehicle(tmp_path):
    # The same advance ratios and weight coefficients as vehicle.cfg gives.
    fields = [_vehicle_fields(), _vehicle_fields()]
    with_cfg = _build(_folder(tmp_path / 'cfg', fields), tmp_path / 'cfg.nod')
    without = _build(_folder(tmp_path / 'spheres', fields, cfg=False), tmp_path / 'spheres.nod')
    assert _root(without) == pytest.approx(_root(with_cfg), rel=1e-12)
    for a, b in zip(_groups(without), _groups(with_cfg), strict=True):
        for name in a:
            np.testing.assert_allclose(a[name], b[name], rtol=1e-12, err_msg=name)


def test_neither_cfg_nor_sphere_fields_is_a_clear_error(tmp_path):
    directory = _folder(tmp_path / 'spheres', [{}, {}], cfg=False)
    with pytest.raises(ValueError, match='vehicle.cfg.*MAIN_ROTOR_RADIUS'):
        _build(directory, tmp_path / 'x.nod')


def test_a_missing_field_without_cfg_is_named(tmp_path):
    fields = _vehicle_fields(density=None)
    directory = _folder(tmp_path / 'spheres', [fields, fields], cfg=False)
    with pytest.raises(ValueError, match='AIR_DENSITY'):
        _build(directory, tmp_path / 'x.nod')


def test_cfg_and_spheres_that_disagree_on_the_rotor_are_refused(tmp_path):
    fields = _vehicle_fields(radius=6.0)
    directory = _folder(tmp_path / 'spheres', [fields, fields])
    with pytest.raises(ValueError, match='MAIN_ROTOR_RADIUS.*vehicle.cfg'):
        _build(directory, tmp_path / 'x.nod')


def test_cfg_and_spheres_that_disagree_on_the_blades_are_refused(tmp_path):
    fields = _vehicle_fields(blades=5.0)
    directory = _folder(tmp_path / 'spheres', [fields, fields])
    with pytest.raises(ValueError, match='MAIN_ROTOR_BLADES.*vehicle.cfg'):
        _build(directory, tmp_path / 'x.nod')


def test_spheres_of_two_rotors_are_refused(tmp_path):
    directory = _folder(tmp_path / 'spheres', [_vehicle_fields(), _vehicle_fields(radius=6.0)],
                        cfg=False)
    with pytest.raises(ValueError, match='MAIN_ROTOR_RADIUS'):
        _build(directory, tmp_path / 'x.nod')


def test_a_weight_sweep_needs_cfg_for_its_nominal_weight(tmp_path):
    fields = [_vehicle_fields(), _vehicle_fields(weight_lb=6000.0)]
    with pytest.raises(ValueError, match='GROSS_WEIGHT.*vehicle.cfg'):
        _build(_folder(tmp_path / 'spheres', fields, cfg=False), tmp_path / 'x.nod')
    # With vehicle.cfg its weight is the nominal one, and each run keeps its own.
    database = _build(_folder(tmp_path / 'cfg', fields), tmp_path / 'cfg.nod')
    assert _root(database)['vehicle_weight_newtons'] == pytest.approx(CFG['weight_newtons'], rel=1e-12)


def test_spheres_without_vehicle_fields_beside_cfg_build_as_before(tmp_path):
    # Every sphere PANAM measured: the vehicle is vehicle.cfg's.
    database = _build(_folder(tmp_path / 'spheres', [{}, {}]), tmp_path / 'x.nod')
    assert _root(database) == pytest.approx(dict(
        main_rotor_radius_meters=CFG['radius'], main_rotor_tip_speed_meters_per_sec=CFG['tip_speed'],
        vehicle_weight_newtons=CFG['weight_newtons']), rel=1e-12)


def test_the_blade_count_is_not_needed(tmp_path):
    # The build uses the rotor's radius and tip speed; its blade count is only checked.
    fields = _vehicle_fields(blades=None)
    database = _build(_folder(tmp_path / 'spheres', [fields, fields], cfg=False), tmp_path / 'x.nod')
    assert _root(database)['main_rotor_radius_meters'] == pytest.approx(CFG['radius'], rel=1e-12)
