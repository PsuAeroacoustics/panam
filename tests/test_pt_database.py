"""pt_database: format 2 databases of pure-tone (PT) spheres, one source per rotor.

The spheres are written here in PSU-WOPWOP's layout (its per-rotor PT noise spheres: AMPLITUDE and PHASE over
(PHI, THETA, FREQUENCY), ROTOR_SHAFT_RATE, ROTOR_AZIMUTH, XYZ in AAM's aft/starboard/up feet, source_tag).
"""
import math

import numpy as np
import pytest
from netCDF4 import Dataset

from panam import nod_database as nd
from panam import pt_database as pt

SHAFT_RATE = 30.04796      # rad/s, the WOPWOP tandem regression rotor
SHAFT_HZ = SHAFT_RATE / (2 * math.pi)
FLOOR_DB = -23.4575


def write_pt_sphere(path, *, tag='Front', xyz_ft=(-19.66, 0.0, 0.98), azimuth_deg=344.3, phase_offset_deg=0.0,
                    speed_kt=77.85, fpa_deg=-2.86, phi=None, theta=None, orders=(1, 2, 3, 6, 9),
                    rate=SHAFT_RATE, rotation='counterclockwise', frame='flight_path', doppler_shift_removed=1,
                    levels=None, phase=None, units_speed='KNOTS'):
    """A PSU-WOPWOP-like PT sphere. Orders 3, 6 and 9 carry energy; 1 and 2 sit at the floor (WOPWOP lists every
    shaft order up to its tonal top and floors the empty ones)."""
    phi = np.arange(-90.0, 90.1, 15.0) if phi is None else np.asarray(phi, float)
    theta = np.arange(0.0, 180.1, 15.0) if theta is None else np.asarray(theta, float)
    frequency = np.asarray(orders, float) * abs(rate) / (2 * math.pi)
    P, T = np.meshgrid(phi, theta, indexing='ij')
    if levels is None:
        levels = np.full((phi.size, theta.size, len(orders)), FLOOR_DB)
        for k, m in enumerate(orders):
            if m % 3 == 0:
                levels[:, :, k] = 90.0 - 2.0 * k - 6.0 * np.cos(np.radians(T)) ** 2
    if phase is None:
        phase = np.zeros_like(levels)
        for k, m in enumerate(orders):
            phase[:, :, k] = (40.0 * np.sin(np.radians(P)) + 3.0 * m + phase_offset_deg) % 360.0
    with Dataset(str(path), 'w') as d:
        d.sphere_type = 'PT'
        d.sphere_frame = frame
        d.speed_reference = 'air'
        d.azimuth_reference = 'track'
        d.floor_level_dB = repr(FLOOR_DB)
        d.main_rotor_rotation = rotation
        d.phase_reference_time_s = '2.0000000298023224E-01'
        d.source_tag = tag
        d.center = 'rotor hub'
        d.speed_of_sound_m_s = '3.42350006E+02'

        def scalar(name, value, unit):
            v = d.createVariable(name, 'f4')
            v[...] = value
            v.unit = unit.ljust(20)
        for flag, value in (('BB', 0.0), ('NB', 0.0), ('PT', 1.0)):
            scalar(flag, value, '')
        scalar('DOPPLER_SHIFT_REMOVED', doppler_shift_removed, '')
        for name in ('EMPTY_WEIGHT', 'FUEL_WEIGHT', 'LOAD_WEIGHT'):
            scalar(name, 0.0, 'POUNDS')
        scalar('RADIUS', 100.0, 'FEET')
        scalar('FLIGHT_PATH_ANGLE', fpa_deg, 'DEGREE')
        scalar('PYLON_ANGLE', 90.0, 'DEGREE')
        scalar('SPEED', speed_kt, units_speed)
        d.createDimension('XYZ', 3)
        v = d.createVariable('XYZ', 'f4', ('XYZ',))
        v[:] = xyz_ft
        v.unit = 'FEET'
        for name, values, unit in (('PHI', phi, 'DEGREE'), ('THETA', theta, 'DEGREE'),
                                   ('FREQUENCY', frequency, 'HERTZ')):
            d.createDimension(name, values.size)
            v = d.createVariable(name, 'f4', (name,))
            v[:] = values
            v.unit = unit
        for name, values, unit in (('AMPLITUDE', levels, 'DECIBEL'), ('PHASE', phase, 'DEGREE')):
            v = d.createVariable(name, 'f4', ('PHI', 'THETA', 'FREQUENCY'))
            v[:] = values
            v.unit = unit
        scalar('AIR_DENSITY', 1.071778, 'KG/M^3')
        scalar('AIRSPEED', speed_kt, 'KNOTS')
        scalar('MAIN_ROTOR_RADIUS', 6.579, 'METER')
        scalar('MAIN_ROTOR_TIP_SPEED', 197.68553, 'M/S')
        scalar('MAIN_ROTOR_BLADES', 3.0, '')
        scalar('PREDICTION_RADIUS', 30.48, 'METER')
        scalar('ROTOR_SHAFT_RATE', rate, 'RAD/S')
        scalar('ROTOR_AZIMUTH', azimuth_deg, 'DEGREE')
        scalar('TONAL_MAX_FREQUENCY', 500.0, 'HERTZ')
    return path


def tandem(tmp_path, **rear):
    front = write_pt_sphere(tmp_path / 'TANDM000.nc')
    options = dict(tag='Rear', xyz_ft=(19.66, 0.0, 4.26), azimuth_deg=24.3, phase_offset_deg=120.0)
    options.update(rear)
    back = write_pt_sphere(tmp_path / 'TANDM001.nc', **options)
    return [front, back]


def build(tmp_path, files, **options):
    options.setdefault('weight_newtons', 1.0e5)
    path = tmp_path / 'pt.nod'
    pt.build_pt_database(files, path, **options)
    return path


def read_source(path, group='sphere0', tag='Front'):
    with Dataset(str(path)) as d:
        g = d.groups[group].groups[tag]
        out = {name: np.array(v[...]) for name, v in g.variables.items()}
        out.update({name: g.getncattr(name) for name in g.ncattrs()})
        return out


def test_root_declares_a_format_2_pure_tone_database(tmp_path):
    path = build(tmp_path, tandem(tmp_path))
    with Dataset(str(path)) as d:
        assert int(d['database_version'][...]) == 2
        assert d.sources == 'Front,Rear'
        assert d.sphere_frame == 'flight_path'
        assert float(d.sphere_levels_lossless) == 1.0
        assert d.band_content == 'broadband'
        assert d.condition_axes == 'helicopter'
        assert d.thrust_coefficient_density == 'run'
        assert d.speed_reference == 'air'
        assert list(d.groups) == ['sphere0']
        assert set(d.groups['sphere0'].groups) == {'Front', 'Rear'}
    assert nd.validate_nod(path) == []


def test_tones_are_the_shaft_orders_above_the_floor(tmp_path):
    path = build(tmp_path, tandem(tmp_path))
    s = read_source(path)
    assert list(s['harmonic']) == [3, 6, 9]
    np.testing.assert_allclose(s['frequency'], np.array([3, 6, 9]) * SHAFT_HZ, rtol=1e-6)
    assert float(s['shaft_rate']) == pytest.approx(SHAFT_RATE, rel=1e-6)


def test_phase_is_locked_to_the_rotor_azimuth(tmp_path):
    files = tandem(tmp_path)
    path = build(tmp_path, files)
    for tag, f in (('Front', files[0]), ('Rear', files[1])):
        s = read_source(path, tag=tag)
        with Dataset(str(f)) as d:
            psi = float(d['ROTOR_AZIMUTH'][...])
            raw = np.array(d['PHASE'][...], float)
            amp = np.array(d['AMPLITUDE'][...], float)
        for k, m in enumerate(s['harmonic']):
            j = {3: 2, 6: 3, 9: 4}[m]   # the column of order m in the test sphere (orders 1, 2, 3, 6, 9)
            expected = np.exp(1j * np.radians(raw[:, :, j] - m * psi))
            got = np.exp(1j * np.radians(s['phase'][:, :, k]))
            assert np.max(np.abs(got - expected)) < 1e-5
            np.testing.assert_allclose(s['amplitude'][:, :, k], amp[:, :, j], atol=1e-4)
        assert float(s['azimuth']) == pytest.approx(psi, abs=1e-4)
        assert s['phase_reference'] == 'rotor_azimuth'


def test_hub_is_written_forward_starboard_up_in_meters(tmp_path):
    path = build(tmp_path, tandem(tmp_path))
    np.testing.assert_allclose(read_source(path, tag='Front')['center'], np.array([19.66, 0.0, 0.98]) * 0.3048,
                               rtol=1e-6)
    np.testing.assert_allclose(read_source(path, tag='Rear')['center'], np.array([-19.66, 0.0, 4.26]) * 0.3048,
                               rtol=1e-6)


def test_rotors_on_one_shaft_rate_form_one_sync_group(tmp_path):
    path = build(tmp_path, tandem(tmp_path))
    assert read_source(path, tag='Front')['sync_group'] == read_source(path, tag='Rear')['sync_group'] != ''
    path = build(tmp_path, tandem(tmp_path, rate=SHAFT_RATE * 1.1))
    assert 'sync_group' not in read_source(path, tag='Front')
    path = build(tmp_path, tandem(tmp_path), sync_groups=None)
    assert 'sync_group' not in read_source(path, tag='Rear')
    path = build(tmp_path, tandem(tmp_path), sync_groups={'Front': 'a', 'Rear': 'b'})
    assert read_source(path, tag='Rear')['sync_group'] == 'b'


def test_clockwise_rotor_has_a_negative_shaft_rate(tmp_path):
    path = build(tmp_path, tandem(tmp_path, rotation='clockwise'))
    assert float(read_source(path, tag='Rear')['shaft_rate']) < 0
    assert float(read_source(path, tag='Front')['shaft_rate']) > 0


def test_floor_cells_stay_quiet_not_empty(tmp_path):
    # A kept tone's floor cells keep the floor: NICE-OPS reads -inf as missing and fills it from the neighbors.
    files = tandem(tmp_path)
    with Dataset(str(files[0]), 'a') as d:
        a = d['AMPLITUDE'][:]
        a[3, 4, 2] = FLOOR_DB
        d['AMPLITUDE'][:] = a
    s = read_source(build(tmp_path, files))
    assert s['amplitude'][3, 4, 0] == pytest.approx(FLOOR_DB, abs=1e-4)
    assert np.isfinite(s['phase'][3, 4, 0])
    assert np.all(np.isfinite(s['amplitude']))


def test_band_spectrum_is_empty_without_broadband_spheres(tmp_path):
    path = build(tmp_path, tandem(tmp_path))
    with Dataset(str(path)) as d:
        g = d.groups['sphere0']
        assert np.all(np.isneginf(g['amplitude'][:]))
        assert np.all(np.isneginf(g['dBA'][:]))
        assert np.all(np.asarray(g['EAA'][:]) == 0)
        frequency = np.asarray(d['frequency'][:] if 'frequency' in d.variables else g['frequency'][:])
        assert frequency.min() < 15.0 and frequency.max() > 9000.0


def test_conditions_are_grouped_and_each_carries_every_source(tmp_path):
    files = tandem(tmp_path)
    for name, speed in (('B', 40.0),):
        files.append(write_pt_sphere(tmp_path / f'{name}0.nc', speed_kt=speed))
        files.append(write_pt_sphere(tmp_path / f'{name}1.nc', tag='Rear', xyz_ft=(19.66, 0.0, 4.26),
                                     speed_kt=speed))
    path = build(tmp_path, files)
    with Dataset(str(path)) as d:
        assert list(d.groups) == ['sphere0', 'sphere1']
        mu = sorted(float(d.groups[g]['advance_ratio'][0]) for g in d.groups)
        np.testing.assert_allclose(mu, np.array([40.0, 77.85]) * 0.514444 / 197.68553, rtol=1e-6)
    assert nd.validate_nod(path) == []


def test_refusals(tmp_path):
    with pytest.raises(ValueError, match='one source per rotor'):
        build(tmp_path, [write_pt_sphere(tmp_path / 'a.nc'), write_pt_sphere(tmp_path / 'b.nc')])
    with pytest.raises(ValueError, match='grid'):
        build(tmp_path, tandem(tmp_path, theta=np.arange(0.0, 180.1, 10.0)))
    with pytest.raises(ValueError, match='flight_path'):
        build(tmp_path, [write_pt_sphere(tmp_path / 'c.nc', frame='level')])
    with pytest.raises(ValueError, match='emission'):
        build(tmp_path, [write_pt_sphere(tmp_path / 'd.nc', doppler_shift_removed=0)])
    with pytest.raises(ValueError, match='every condition'):
        files = tandem(tmp_path)
        files.append(write_pt_sphere(tmp_path / 'e.nc', speed_kt=40.0))
        build(tmp_path, files)
    with pytest.raises(ValueError, match='unit'):
        build(tmp_path, [write_pt_sphere(tmp_path / 'f.nc', units_speed='FURLONGS')])
    with pytest.raises(ValueError, match='weight'):
        pt.build_pt_database(tandem(tmp_path), tmp_path / 'g.nod')


def test_full_sphere_seam_drops_a_matching_row_and_refuses_one_that_differs(tmp_path):
    phi = np.arange(-180.0, 180.1, 15.0)
    front = write_pt_sphere(tmp_path / 'F.nc', phi=phi)
    with Dataset(str(front), 'a') as d:      # the +180 row is the -180 row
        for name in ('AMPLITUDE', 'PHASE'):
            v = d[name][:]
            v[-1] = v[0]
            d[name][:] = v
    path = build(tmp_path, [front])
    with Dataset(str(path)) as d:
        phis = np.unique(np.asarray(d['phi'][:]))
    assert phis.min() == -180.0 and phis.max() == 165.0
    with Dataset(str(front), 'a') as d:
        v = d['PHASE'][:]
        v[-1, 5, 2] += 90.0
        d['PHASE'][:] = v
    with pytest.raises(ValueError, match='seam'):
        build(tmp_path, [front])


def test_validate_nod_flags_a_tone_off_its_shaft_order(tmp_path):
    path = build(tmp_path, tandem(tmp_path))
    with Dataset(str(path), 'a') as d:
        f = d.groups['sphere0'].groups['Front']['frequency']
        f[0] = f[0] * 1.01
    rules = {v.rule for v in nd.validate_nod(path)}
    assert 'T1' in rules
    with Dataset(str(path), 'a') as d:
        d.groups['sphere0'].groups['Rear'].delncattr('phase_reference')
    rules = {v.rule for v in nd.validate_nod(path)}
    assert 'T3' in rules


def test_command_line_builds_and_refuses(tmp_path, capsys):
    from tools import build_pt_database
    files = [str(f) for f in tandem(tmp_path)]
    assert build_pt_database.main(files + ['-o', str(tmp_path / 'cli.nod'), '--weight-newtons', '1e5']) == 0
    assert nd.validate_nod(tmp_path / 'cli.nod') == []
    assert build_pt_database.main(files[:1] * 2 + ['-o', str(tmp_path / 'bad.nod'), '--weight-newtons', '1e5']) == 1
    assert 'one source per rotor' in capsys.readouterr().err
