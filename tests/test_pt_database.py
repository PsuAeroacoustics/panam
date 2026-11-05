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
                    levels=None, phase=None, units_speed='KNOTS', rotor_radius=6.579, air_density=1.071778,
                    speed_of_sound='3.42350006E+02', speed_reference='air',
                    phase_reference_time='2.0000000298023224E-01', tip_speed=197.68553):
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
        if speed_reference is not None:
            d.speed_reference = speed_reference
        d.azimuth_reference = 'track'
        d.floor_level_dB = repr(FLOOR_DB)
        d.main_rotor_rotation = rotation
        if phase_reference_time is not None:
            d.phase_reference_time_s = phase_reference_time
        d.source_tag = tag
        d.center = 'rotor hub'
        if speed_of_sound is not None:
            d.speed_of_sound_m_s = speed_of_sound

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
        scalar('AIR_DENSITY', air_density, 'KG/M^3')
        scalar('AIRSPEED', speed_kt, 'KNOTS')
        scalar('MAIN_ROTOR_RADIUS', rotor_radius, 'METER')
        scalar('MAIN_ROTOR_TIP_SPEED', tip_speed, 'M/S')
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
    # A kept tone's floor cells keep the floor: NICEOPS reads -inf as missing and fills it from the neighbors.
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


# ---------------------------------------------------------------------------------------- multirotor conditions

HEX_R = 0.3556          # m, a 28 in prop
HEX_WEIGHT = 151.2      # N
KNOT = 0.514444


def hexacopter(tmp_path, conditions=((0.0, 0.0), (8.6898, 0.0)), rates=(-200.0, 240.0), prefix='H'):
    """Two of a hexacopter's rotors (one each way) at each (speed kt, flight path angle deg) condition."""
    files = []
    for i, (speed, fpa) in enumerate(conditions):
        for tag, rate, x in (('M1', rates[0], (0.0, 2.6, 0.0)), ('M2', rates[1], (0.0, -2.6, 0.0))):
            files.append(write_pt_sphere(tmp_path / f'{prefix}{i}{tag}.nc', tag=tag, xyz_ft=x, speed_kt=speed,
                                         fpa_deg=fpa, rate=abs(rate), rotor_radius=HEX_R, air_density=1.15,
                                         rotation='counterclockwise' if rate > 0 else 'clockwise'))
    return files


def momentum(rho, weight=HEX_WEIGHT, n=6, r=HEX_R):
    return math.sqrt(weight / (2 * rho * n * math.pi * r ** 2))


def read_group(path, group):
    with Dataset(str(path)) as d:
        g = d.groups[group]
        return {name: np.array(v[...], float).reshape(-1)[0] for name, v in g.variables.items()
                if v.size == 1 and name not in ('DOPPLER_SHIFT_REMOVED',)}, set(g.variables)


def test_multirotor_conditions_are_momentum_similarity_axes(tmp_path):
    path = build(tmp_path, hexacopter(tmp_path), weight_newtons=HEX_WEIGHT, condition_axes='multirotor',
                 rotor_count=6)
    om_rms = math.sqrt((200.0 ** 2 + 240.0 ** 2) / 2)
    ct_h = HEX_WEIGHT / (6 * 1.15 * math.pi * HEX_R ** 2 * (om_rms * HEX_R) ** 2)
    with Dataset(str(path)) as d:
        assert d.condition_axes == 'multirotor'
        assert int(d.rotor_count) == 6
        assert float(d.rotor_radius_m) == pytest.approx(HEX_R, rel=1e-6)
        assert float(d.hover_thrust_coefficient) == pytest.approx(ct_h, rel=1e-6)
        assert d.thrust_coefficient_density == 'run'
        # NICEOPS refuses a multirotor database with fixed_load_factor set: its load-factor law is in C_T
        assert not bool(d['fixed_load_factor'][...])
    v_h = momentum(1.15)
    om_h = math.sqrt(HEX_WEIGHT / (6 * 1.15 * math.pi * HEX_R ** 2 * ct_h)) / HEX_R
    for group, speed in (('sphere0', 0.0), ('sphere1', 8.6898)):
        values, names = read_group(path, group)
        assert not names & {'advance_ratio', 'flight_path_angle', 'thrust_coefficient'}
        assert values['inplane_speed_ratio'] == pytest.approx(speed * KNOT / v_h, rel=1e-6, abs=1e-12)
        assert values['axial_speed_ratio'] == pytest.approx(0.0, abs=1e-12)
        assert values['hover_tip_mach'] == pytest.approx(om_h * HEX_R / 342.35, rel=1e-6)
        assert values['air_density'] == pytest.approx(1.15, rel=1e-6)
        assert values['weight_n'] == pytest.approx(HEX_WEIGHT)
        assert values['airspeed'] == pytest.approx(speed * KNOT, rel=1e-6, abs=1e-12)
        assert values['climb_rate'] == pytest.approx(0.0, abs=1e-12)
        for tag, rate in (('M1', 200.0), ('M2', 240.0)):
            assert float(read_source(path, group, tag)['speed_ratio']) == pytest.approx(rate / om_h, rel=1e-6)
    assert nd.validate_nod(path) == []


def test_multirotor_climb_is_positive_axial_speed(tmp_path):
    files = hexacopter(tmp_path, conditions=((0.0, 0.0), (8.6898, 6.0)))
    path = build(tmp_path, files, weight_newtons=HEX_WEIGHT, condition_axes='multirotor', rotor_count=6)
    values, _ = read_group(path, 'sphere1')
    u, v_h = 8.6898 * KNOT, momentum(1.15)
    assert values['axial_speed_ratio'] == pytest.approx(u * math.sin(math.radians(6.0)) / v_h, rel=1e-5)
    assert values['inplane_speed_ratio'] == pytest.approx(u * math.cos(math.radians(6.0)) / v_h, rel=1e-5)
    assert values['climb_rate'] == pytest.approx(u * math.sin(math.radians(6.0)), rel=1e-5)
    assert nd.validate_nod(path) == []


def test_multirotor_hover_thrust_coefficient_can_be_given(tmp_path):
    files = hexacopter(tmp_path, conditions=((8.6898, 0.0),))
    with pytest.raises(ValueError, match='hover'):
        build(tmp_path, files, weight_newtons=HEX_WEIGHT, condition_axes='multirotor', rotor_count=6)
    path = build(tmp_path, files, weight_newtons=HEX_WEIGHT, condition_axes='multirotor', rotor_count=6,
                 hover_thrust_coefficient=0.009)
    with Dataset(str(path)) as d:
        assert float(d.hover_thrust_coefficient) == pytest.approx(0.009)
    assert nd.validate_nod(path) == []


def test_multirotor_refusals(tmp_path):
    files = hexacopter(tmp_path)
    with pytest.raises(ValueError, match='rotor_count'):
        build(tmp_path, files, weight_newtons=HEX_WEIGHT, condition_axes='multirotor')
    with pytest.raises(ValueError, match='condition_axes'):
        build(tmp_path, files, weight_newtons=HEX_WEIGHT, condition_axes='quadcopter', rotor_count=6)
    odd = hexacopter(tmp_path, prefix='O')
    with Dataset(str(odd[1]), 'a') as d:
        d['MAIN_ROTOR_RADIUS'][...] = 0.4
    with pytest.raises(ValueError, match='radius'):
        build(tmp_path, odd, weight_newtons=HEX_WEIGHT, condition_axes='multirotor', rotor_count=6)
    quiet = [write_pt_sphere(tmp_path / f'S{k}.nc', tag=t, speed_kt=0.0, fpa_deg=0.0, rate=220.0, rotor_radius=HEX_R,
                             speed_of_sound=None) for k, t in enumerate(('M1', 'M2'))]
    with pytest.raises(ValueError, match='speed of sound'):
        build(tmp_path, quiet, weight_newtons=HEX_WEIGHT, condition_axes='multirotor', rotor_count=6)


def test_validate_nod_checks_the_multirotor_fields(tmp_path):
    path = build(tmp_path, hexacopter(tmp_path), weight_newtons=HEX_WEIGHT, condition_axes='multirotor',
                 rotor_count=6)
    with Dataset(str(path), 'a') as d:
        d.groups['sphere1'].groups['M2']['speed_ratio'][...] = 2.0
    assert 'M4' in {v.rule for v in nd.validate_nod(path)}
    path = build(tmp_path, hexacopter(tmp_path), weight_newtons=HEX_WEIGHT, condition_axes='multirotor',
                 rotor_count=6)
    with Dataset(str(path), 'a') as d:
        d.groups['sphere1']['inplane_speed_ratio'][:] = 9.0
    assert 'M3' in {v.rule for v in nd.validate_nod(path)}
    path = build(tmp_path, hexacopter(tmp_path), weight_newtons=HEX_WEIGHT, condition_axes='multirotor',
                 rotor_count=6)
    with Dataset(str(path), 'a') as d:
        d.delncattr('hover_thrust_coefficient')
    assert 'M1' in {v.rule for v in nd.validate_nod(path)}


def test_load_nod_reads_a_multirotor_database(tmp_path):
    files = hexacopter(tmp_path, conditions=((0.0, 0.0), (8.6898, 6.0)))
    path = build(tmp_path, files, weight_newtons=HEX_WEIGHT, condition_axes='multirotor', rotor_count=6)
    db = nd.load_nod(path)
    assert len(db) == 2
    hover, climb = db.conditions
    assert hover.speed_kt == pytest.approx(0.0, abs=1e-9)
    assert climb.speed_kt == pytest.approx(8.6898, rel=1e-6)
    assert climb.flight_path_angle_deg == pytest.approx(6.0, rel=1e-6)
    assert np.isnan(climb.advance_ratio) and np.isnan(climb.thrust_coefficient)
    x, z, m = climb.multirotor_axes
    u, v_h = 8.6898 * KNOT, momentum(1.15)
    assert z == pytest.approx(u * math.sin(math.radians(6.0)) / v_h, rel=1e-5)
    assert hover.key != climb.key and climb.key[0] == 'multirotor'


def test_command_line_builds_a_multirotor_database(tmp_path, capsys):
    from tools import build_pt_database
    files = [str(f) for f in hexacopter(tmp_path)]
    out = tmp_path / 'hex.nod'
    assert build_pt_database.main(files + ['-o', str(out), '--weight-newtons', str(HEX_WEIGHT),
                                           '--condition-axes', 'multirotor', '--rotor-count', '6']) == 0
    assert nd.validate_nod(out) == []
    assert build_pt_database.main(files + ['-o', str(tmp_path / 'bad.nod'), '--weight-newtons', '151',
                                           '--condition-axes', 'multirotor']) == 1
    assert 'rotor_count' in capsys.readouterr().err


def test_analysis_over_several_revolutions_keeps_the_whole_orders(tmp_path):
    # PSU-WOPWOP's spectrum over N revolutions has bins every 1/N shaft order: the tones are the whole orders;
    # the bins between (leakage, a loading's slow drift) are dropped and their share of the energy recorded.
    orders = np.arange(1, 37) / 4.0                        # quarter orders, as over 4 revolutions
    levels = np.full((13, 13, orders.size), 10.0)          # weak everywhere between the tones
    for k, m in enumerate(orders):
        if m in (3.0, 6.0, 9.0):
            levels[:, :, k] = 80.0
    sphere = write_pt_sphere(tmp_path / 'Q.nc', orders=orders, levels=levels)
    path = build(tmp_path, [sphere])
    s = read_source(path)
    assert list(s['harmonic']) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert 0 < float(s['dropped_energy_fraction']) < 1e-4
    off = write_pt_sphere(tmp_path / 'R.nc', orders=(1.0, 2.0, 2.37, 3.0))   # 237/100: no spectrum of <= 64 revolutions
    with pytest.raises(ValueError, match='shaft order'):
        build(tmp_path, [off])


def test_spheres_a_hair_apart_in_speed_form_one_condition_when_asked(tmp_path):
    # Per-rotor predictions of one pass, each over its own steadiest window, record speeds 1e-4 apart.
    files = [write_pt_sphere(tmp_path / 'a.nc', speed_kt=8.6775),
             write_pt_sphere(tmp_path / 'b.nc', tag='Rear', xyz_ft=(19.66, 0.0, 4.26), speed_kt=8.6785)]
    with pytest.raises(ValueError, match='every condition'):
        build(tmp_path, files)
    path = build(tmp_path, files, speed_tolerance_kt=0.01)
    with Dataset(str(path)) as d:
        assert list(d.groups) == ['sphere0']
    assert float(read_source(path, tag='Rear')['sphere_speed_kt']) == pytest.approx(8.6785, rel=1e-6)
    assert nd.validate_nod(path) == []


# ------------------------------------------------------------------------------------------------ review fixes

def test_conditions_at_different_air_densities_keep_the_thrust_coefficient_as_an_axis(tmp_path):
    # C_T = W / (rho pi R^2 V^2) at each condition's own density differs between conditions predicted in different
    # air: such a database is not fixed_load_factor (one C_T), or validate_nod's C8 and NICEOPS refuse it.
    files = []
    for i, (speed, fpa, rho) in enumerate(((40.0, -6.0, 1.07), (77.85, -2.86, 1.15), (60.0, 3.0, 1.10))):
        files += tandem_at(tmp_path, f'D{i}', speed_kt=speed, fpa_deg=fpa, air_density=rho)
    path = build(tmp_path, files)
    with Dataset(str(path)) as d:
        assert not bool(d['fixed_load_factor'][...])
        assert d.thrust_coefficient_density == 'run'
        ct = {round(float(d.groups[g]['air_density'][...] if 'air_density' in d.groups[g].variables
                          else d.groups[g].air_density), 6): float(d.groups[g]['thrust_coefficient'][:].ravel()[0])
              for g in d.groups}
    for rho, value in ct.items():
        assert value == pytest.approx(1.0e5 / (rho * math.pi * 6.579 ** 2 * 197.68553 ** 2), rel=1e-5)
    assert nd.validate_nod(path) == []
    # one density: one C_T, at load factor 1
    path = build(tmp_path, tandem(tmp_path))
    with Dataset(str(path)) as d:
        assert bool(d['fixed_load_factor'][...])


def test_niceops_loads_a_database_predicted_in_different_air(tmp_path):
    from sphere_helpers import executable
    import subprocess
    niceops = executable('niceops')
    if niceops is None:
        pytest.skip('niceops not configured (local_paths: niceops)')
    files = []
    for i, (speed, fpa, rho) in enumerate(((40.0, -6.0, 1.07), (77.85, -2.86, 1.15), (60.0, 3.0, 1.10))):
        files += tandem_at(tmp_path, f'N{i}', speed_kt=speed, fpa_deg=fpa, air_density=rho)
    path = build(tmp_path, files)
    result = subprocess.run([niceops, '-d', str(path), '--self_test', '-l', 'self_test.log'], cwd=tmp_path,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, (tmp_path / 'self_test.log').read_text()[-2000:] + result.stderr
    assert 'Self test passed' in (tmp_path / 'self_test.log').read_text()


def tandem_at(tmp_path, prefix, **condition):
    return [write_pt_sphere(tmp_path / f'{prefix}F.nc', **condition),
            write_pt_sphere(tmp_path / f'{prefix}R.nc', tag='Rear', xyz_ft=(19.66, 0.0, 4.26), azimuth_deg=24.3,
                            **condition)]


def test_spheres_a_hair_apart_in_speed_and_angle_form_one_condition_when_asked(tmp_path):
    files = [write_pt_sphere(tmp_path / 'a.nc', speed_kt=8.6775, fpa_deg=-2.8601),
             write_pt_sphere(tmp_path / 'b.nc', tag='Rear', xyz_ft=(19.66, 0.0, 4.26), speed_kt=8.6785,
                             fpa_deg=-2.8603)]
    path = build(tmp_path, files, speed_tolerance_kt=0.01)
    with Dataset(str(path)) as d:
        assert list(d.groups) == ['sphere0']
        assert float(d.groups['sphere0']['flight_path_angle'][:].ravel()[0]) == pytest.approx(-2.8602, abs=1e-6)
    assert nd.validate_nod(path) == []


def test_multirotor_conditions_may_record_different_tip_speeds(tmp_path):
    # A multirotor's rotor speed changes with the condition by design: only its radius must agree.
    files = hexacopter(tmp_path)
    for f in files[2:]:
        with Dataset(str(f), 'a') as d:
            d['MAIN_ROTOR_TIP_SPEED'][...] = 85.0
    path = build(tmp_path, files, weight_newtons=HEX_WEIGHT, condition_axes='multirotor', rotor_count=6)
    assert nd.validate_nod(path) == []
    # a helicopter's main rotor turns at one tip speed
    heli = tandem(tmp_path) + tandem_at(tmp_path, 'T', speed_kt=40.0, fpa_deg=-6.0, tip_speed=180.0)
    with pytest.raises(ValueError, match='tip speed'):
        build(tmp_path, heli)


def test_rotors_sharing_a_shaft_are_grouped_whichever_tag_sorts_first(tmp_path):
    files = tandem(tmp_path) + [write_pt_sphere(tmp_path / 'Aux.nc', tag='Aux', xyz_ft=(0.0, 3.0, 0.0),
                                                rate=SHAFT_RATE * 1.3)]
    path = build(tmp_path, files)
    front, rear, aux = (read_source(path, tag=t) for t in ('Front', 'Rear', 'Aux'))
    assert front['sync_group'] == rear['sync_group'] != ''
    assert 'sync_group' not in aux


def test_rotors_without_a_phase_reference_time_are_not_grouped(tmp_path):
    files = [write_pt_sphere(tmp_path / 'F.nc', phase_reference_time=None),
             write_pt_sphere(tmp_path / 'R.nc', tag='Rear', xyz_ft=(19.66, 0.0, 4.26), phase_reference_time=None)]
    with pytest.warns(UserWarning, match='phase_reference_time_s'):
        path = build(tmp_path, files)
    assert 'sync_group' not in read_source(path, tag='Front')
    assert 'sync_group' not in read_source(path, tag='Rear')


def test_an_unknown_main_rotor_is_refused_by_name(tmp_path, capsys):
    from tools import build_pt_database
    files = [str(f) for f in tandem(tmp_path)]
    with pytest.raises(ValueError, match="'front'.*Front"):
        build(tmp_path, files, main_rotor='front')
    assert build_pt_database.main(files + ['-o', str(tmp_path / 'x.nod'), '--weight-newtons', '1e5',
                                           '--main-rotor', 'front']) == 1
    err = capsys.readouterr().err
    assert 'build_pt_database:' in err and 'Front' in err and 'Traceback' not in err


def test_speed_reference_must_agree_with_what_the_spheres_record(tmp_path):
    mixed = [write_pt_sphere(tmp_path / 'F.nc'),
             write_pt_sphere(tmp_path / 'R.nc', tag='Rear', xyz_ft=(19.66, 0.0, 4.26), speed_reference='ground')]
    for argument in (None, 'air', 'ground'):
        with pytest.raises(ValueError, match='speed_reference'):
            build(tmp_path, mixed, speed_reference=argument)
    with pytest.raises(ValueError, match='speed_reference'):
        build(tmp_path, tandem(tmp_path), speed_reference='ground')       # the spheres record air
    assert build(tmp_path, tandem(tmp_path), speed_reference='air')
    bare = [write_pt_sphere(tmp_path / 'F0.nc', speed_reference=None),
            write_pt_sphere(tmp_path / 'R0.nc', tag='Rear', xyz_ft=(19.66, 0.0, 4.26), speed_reference=None)]
    with Dataset(str(build(tmp_path, bare, speed_reference='ground'))) as d:
        assert d.speed_reference == 'ground'
    with Dataset(str(build(tmp_path, bare))) as d:
        assert d.speed_reference == 'air'                                  # PSU-WOPWOP's convention
    with pytest.raises(ValueError, match='speed_reference'):
        build(tmp_path, bare, speed_reference='sideways')


def test_command_line_takes_the_speed_tolerance_and_reference(tmp_path):
    from tools import build_pt_database
    files = [str(write_pt_sphere(tmp_path / 'a.nc', speed_kt=8.6775, speed_reference=None)),
             str(write_pt_sphere(tmp_path / 'b.nc', tag='Rear', xyz_ft=(19.66, 0.0, 4.26), speed_kt=8.6785,
                                 speed_reference=None))]
    out = tmp_path / 'cli.nod'
    assert build_pt_database.main(files + ['-o', str(out), '--weight-newtons', '1e5', '--speed-tolerance-kt', '0.01',
                                           '--speed-reference', 'ground']) == 0
    with Dataset(str(out)) as d:
        assert list(d.groups) == ['sphere0'] and d.speed_reference == 'ground'


def test_measured_sources_write_a_valid_multirotor_database(tmp_path):
    """Measured tones (orders given, -inf where a cell has no data, per-rotor states) through write_pt_database."""
    rng = np.random.default_rng(4)
    phi, theta = np.arange(-90.0, 90.1, 15.0), np.arange(0.0, 180.1, 15.0)
    harmonics = np.array([1, 2, 4])
    hubs = {'M1': (0.0, 0.793, 0.0), 'M2': (0.0, -0.793, 0.0)}
    spheres = []
    for speed_kt, rates in ((0.0, (-220.0, 230.0)), (10.0, (-226.0, 224.0))):
        for (tag, hub), rate in zip(hubs.items(), rates):
            level = rng.uniform(40, 70, (phi.size, theta.size, harmonics.size))
            level[:, :2, :] = -np.inf                          # never seen: ahead of the nose
            level[..., 2] = -np.inf if tag == 'M2' and speed_kt else level[..., 2]
            phase = rng.uniform(-180, 180, level.shape)
            state = {'measured_shaft_rate': (rate, 'rad s-1', 'mean logged speed'),
                     'disk_normal': ((0.0, 0.0, 1.0), '1', 'thrust axis, sphere frame'),
                     'thrust_estimate_n': (27.0, 'N', 'C_Th rho pi R^2 (Omega R)^2')}
            spheres.append(pt.measured_pt_sphere(
                tag, phi, theta, harmonics, level, phase, radius_m=30.48, speed_kt=speed_kt,
                flight_path_angle_deg=0.0, shaft_rate=rate, center_m=hub, rotor_radius_m=0.3556, air_density=1.15,
                sound_speed_mps=338.0, state=state, name=f'{tag}_{speed_kt:g}kt'))
    path = tmp_path / 'measured.nod'
    pt.write_pt_database(spheres, path, weight_newtons=160.0, sync_groups=None, condition_axes='multirotor',
                         rotor_count=6, hover_thrust_coefficient=0.009, speed_reference='ground',
                         azimuth_label='relative_within_sense')
    assert nd.validate_nod(path) == []
    with Dataset(str(path)) as d:
        assert d.azimuth_label == 'relative_within_sense'
        m1 = d.groups['sphere0'].groups['M1']
        assert m1.phase_reference == 'rotor_azimuth' and 'sync_group' not in m1.ncattrs()
        assert np.isclose(float(m1['measured_shaft_rate'][...]), -220.0)
        assert np.allclose(m1['disk_normal'][:], (0.0, 0.0, 1.0))
        assert np.all(np.isneginf(m1['amplitude'][:, :2, :]))
        assert list(m1['harmonic'][:]) == [1, 2, 4]
        assert list(d.groups['sphere1'].groups['M2']['harmonic'][:]) == [1, 2]     # an order never heard is dropped
        assert float(m1['azimuth'][...]) == 0.0
    with pytest.raises(ValueError, match='finite wherever'):
        pt.measured_pt_sphere('M1', phi, theta, harmonics, np.zeros((phi.size, theta.size, 3)),
                              np.full((phi.size, theta.size, 3), np.nan), radius_m=30.48, speed_kt=0.0,
                              flight_path_angle_deg=0.0, shaft_rate=200.0, center_m=(0, 0, 0), rotor_radius_m=0.3,
                              air_density=1.2, sound_speed_mps=340.0)


def test_fitted_directivities_are_evaluated_on_the_grid_and_kept_as_provenance(tmp_path):
    """fitted_pt_sphere: the grid holds the expansion's level (rms) and phase, the coefficients ride along, and the
    root carries tone_speed_exponent."""
    from panam import rotor_tones as rt
    rng = np.random.default_rng(5)
    phi, theta = np.arange(-180.0, 180.0, 15.0), np.arange(0.0, 180.1, 15.0)
    degree = 6
    n = (degree + 1) ** 2
    ks = np.concatenate([np.arange(-l, l + 1) for l in range(degree + 1)])
    hubs = {'M1': ((0.0, 0.793, 0.0), 1), 'M2': ((0.0, -0.793, 0.0), -1)}
    knots = np.arange(0.0, 181.0, 30.0)
    knots_energy = np.array([4.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0])
    spheres, given = [], {}
    for tag, (hub, sense) in hubs.items():
        model = {m: [m * sense + j for j in (-1, 0, 1)] for m in (2, 4)}
        directivities = {m: np.where(np.isin(ks, model[m]), rng.normal(size=n) + 1j * rng.normal(size=n), 0.0)
                         for m in (2, 4)}
        given[tag] = directivities
        spheres.append(pt.fitted_pt_sphere(
            tag, phi, theta, directivities, degree=degree, shaft_rate=220.0 * sense, model=model,
            incoherent={4: knots_energy}, incoherent_knots_deg=knots, radius_m=1.0,
            speed_kt=0.0, flight_path_angle_deg=0.0, center_m=hub, rotor_radius_m=0.3556, air_density=1.15,
            sound_speed_mps=338.0))
    path = tmp_path / 'fitted.nod'
    pt.write_pt_database(spheres, path, weight_newtons=160.0, sync_groups=None, condition_axes='multirotor',
                         rotor_count=6, hover_thrust_coefficient=0.009, speed_reference='ground',
                         azimuth_label='relative_within_sense', tone_speed_exponent=3.0)
    assert nd.validate_nod(path) == []
    gp, gt = np.meshgrid(phi, theta, indexing='ij')
    with Dataset(str(path)) as d:
        assert float(d.tone_speed_exponent) == 3.0
        for tag in hubs:
            src = d.groups['sphere0'].groups[tag]
            assert list(src['harmonic'][:]) == [2, 4]
            c = src['directivity_real'][:] + 1j * src['directivity_imag'][:]
            assert int(src['directivity_real'].degree) == degree
            for i, m in enumerate((2, 4)):
                assert np.array_equal(c[i], given[tag][m])
                a = rt.evaluate_directivity(c[i], degree, gp, gt)
                stored = 10.0 ** (src['amplitude'][:, :, i] / 20.0) * np.sqrt(2.0) * rt.P_REF \
                    * np.exp(1j * np.radians(src['phase'][:, :, i]))
                assert np.allclose(stored, a, rtol=1e-9, atol=1e-12 * np.abs(a).max())
            assert src['directivity_real'].azimuthal_orders.startswith('2: ')
            inc = src['incoherent_amplitude'][:]
            assert np.all(np.isneginf(inc[:, :, 0]))                  # order 2 has none
            polar, _ = rt.axis_angles(gp, gt)
            want = 10 * np.log10(np.interp(polar, np.radians(knots), knots_energy) / 2 / rt.P_REF ** 2 + 1e-300)
            lit = np.isfinite(inc[:, :, 1])
            assert lit.any() and np.allclose(inc[:, :, 1][lit], want[lit], atol=1e-9)
            assert np.array_equal(src['incoherent_energy'][1], knots_energy)
    with pytest.raises(ValueError, match='tone_speed_exponent'):
        pt.write_pt_database(spheres, tmp_path / 'bad.nod', weight_newtons=160.0, sync_groups=None,
                             condition_axes='multirotor', rotor_count=6, hover_thrust_coefficient=0.009,
                             speed_reference='ground', tone_speed_exponent=-1.0)
    with pytest.raises(ValueError, match='coefficients'):
        pt.fitted_pt_sphere('M1', phi, theta, {2: np.zeros(n - 1)}, degree=degree, shaft_rate=220.0, radius_m=1.0,
                            speed_kt=0.0, flight_path_angle_deg=0.0, center_m=(0, 0, 0), rotor_radius_m=0.3,
                            air_density=1.2, sound_speed_mps=340.0)
