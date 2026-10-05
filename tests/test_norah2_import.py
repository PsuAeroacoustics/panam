"""NORAH2 .hem hemispheres into a NICE-OPS database (build_database_from_norah2, norah2_to_nod.py)."""
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

import flight_acoustics as fa
import local_paths
from panam_acoustics.atmosphere import Atmosphere

REPO = Path(__file__).resolve().parents[1]
_NORAH2 = local_paths.data_path('norah2', required=False)
NORAH2_HEMISPHERES = os.path.join(_NORAH2, 'Hemispheres') if _NORAH2 else ''
needs_norah2 = pytest.mark.skipif(not os.path.isdir(NORAH2_HEMISPHERES),
                                  reason='NORAH2 distribution not configured (local_paths: norah2)')

MEASURED = dict(temperature_k=288.15, pressure_kpa=98.0, relative_humidity=45.0)
ICAO = Atmosphere(temperature=298.15, pressure=101.325, relative_humidity=70.0)
#: An R44-like vehicle.
VEHICLE = dict(main_rotor_tip_speed_mps=214.88, main_rotor_radius_m=5.03,
               vehicle_weight_newtons=861.0 * fa.STANDARD_GRAVITY)
R_SPHERE = 30.48
#: Exact base-10 band centers 12.6 Hz - 5 kHz: the writer marks 10 Hz and 6.3-10 kHz missing.
BANDS = 1000.0 * 10.0 ** (np.arange(-19, 8) / 10.0)


def synthetic_hemisphere(level_db=70.0, starboard_boost_db=0.0, nose_boost_db=0.0):
    """A UMAPR hemisphere at 100 ft, optionally louder to starboard and toward the nose."""
    azi = np.arange(0.0, 360.0 + 1e-9, 5.0)
    elv = np.arange(-5.0, 90.0 + 1e-9, 5.0)
    AZI = np.broadcast_to(azi, (elv.size, azi.size))
    cos_elv = np.cos(np.radians(elv))[:, None]
    field = (level_db + starboard_boost_db * np.sin(np.radians(AZI)) * cos_elv
             - nose_boost_db * np.cos(np.radians(AZI)) * cos_elv)
    bands = np.broadcast_to(field, (BANDS.size,) + field.shape).copy()
    return {'azi_grid_deg': azi, 'elv_grid_deg': elv,
            'third_octave': {'band_centers_hz': BANDS, 'bands_db': bands},
            'metadata': {'r_ref': 100.0, 'length_units': 'ft', 'apply_absorption_deprop': True,
                         'atmosphere': dict(MEASURED)}}


def write_hem(path, speed_knots=80.0, angle_deg=0.0, minimum_level_db=-np.inf, **kwargs):
    fa.write_norah2_hemisphere(str(path), synthetic_hemisphere(**kwargs), speed_knots=speed_knots,
                               flight_path_angle_deg=angle_deg, minimum_level_db=minimum_level_db)
    return str(path)


def three_hems(tmp_path, **kwargs):
    return [write_hem(tmp_path / f'X_{i}.hem', speed, angle, **kwargs)
            for i, (speed, angle) in enumerate([(60.0, -6.0), (80.0, 0.0), (100.0, -3.0)])]


def build(hem_files, out, speed_mapping='ground_speed', **kwargs):
    arguments = dict(VEHICLE, speed_mapping=speed_mapping)
    arguments.update(kwargs)
    return fa.build_database_from_norah2(hem_files, str(out), **arguments)


def edit_constant(path, name, value):
    """Rewrite one table constant of a .hem in place, keeping its CRLF layout."""
    raw = Path(path).read_bytes().decode('ascii').split('\r\n')
    for i, line in enumerate(raw):
        if line.split() and line.split()[0] == name:
            raw[i] = f'{name}\t{value}\t! edited'
            break
    else:
        raise AssertionError(f'no {name} in {path}')
    Path(path).write_bytes('\r\n'.join(raw).encode('ascii'))


def group_grid(database, group='sphere0'):
    """(phi, theta) axes, (phi, theta, band) amplitude and coverage, dBA, EAA of one group."""
    with Dataset(str(database)) as nc:
        phi = np.array(nc['phi'][:])
        theta = np.array(nc['theta'][:])
        g = nc[group]
        amplitude = np.array(g['amplitude'][:])
        coverage = np.array(g['coverage'][:])
        dba = np.array(g['dBA'][:])
        eaa = np.array(g['EAA'][:])
    phis = np.unique(phi)
    thetas = np.unique(theta)
    shape = (phis.size, thetas.size)
    return phis, thetas, amplitude, coverage, dba.reshape(shape), eaa.reshape(shape)


def offset_db(frequency, poldist=60.0, radius=R_SPHERE, atmosphere=ICAO):
    return 20.0 * np.log10(poldist / radius) + atmosphere.attenuation_coefficient(frequency) * (poldist - radius)


def test_round_trip_levels_at_the_sphere_radius(tmp_path):
    """write_norah2_hemisphere -> import gives the analytic rescaling, cell for cell.

    Against the file: + 20 lg(60 / 30.48) + alpha_ICAO (60 - 30.48), exactly.  Against the
    original hemisphere: its first 30.48 m of absorption, measured atmosphere, becomes the
    ICAO atmosphere's, which is what a .hem can still say about it.
    """
    files = three_hems(tmp_path, starboard_boost_db=4.0, nose_boost_db=3.0)
    out = tmp_path / 'X.nod'
    build(files, out)
    hem = fa.load_norah2_hemisphere(files[0])
    phis, thetas, amplitude, coverage, _, _ = group_grid(out)
    lower = [list(phis).index(p) for p in hem['phi_deg']]
    stored = amplitude[lower]
    expected = hem['levels_db'] + offset_db(hem['frequency_hz'])
    present = np.isfinite(hem['levels_db'])
    assert present.sum() == 19 * 19 * 27
    assert np.allclose(stored[present], expected[present], rtol=0.0, atol=1e-9)

    measured = Atmosphere(temperature=MEASURED['temperature_k'], pressure=MEASURED['pressure_kpa'],
                          relative_humidity=MEASURED['relative_humidity'])
    nominal = fa.NORAH2_BAND_CENTERS_HZ[1:28]
    PH, TH = np.meshgrid(np.radians(hem['phi_deg']), np.radians(hem['theta_deg']), indexing='ij')
    azi, elv = fa.norah2umapr(PH, TH)
    field = 70.0 + 4.0 * np.sin(azi) * np.cos(elv) - 3.0 * np.cos(azi) * np.cos(elv)
    original = (field[:, :, None] + (measured.attenuation_coefficient(BANDS)
                                     - ICAO.attenuation_coefficient(nominal)) * R_SPHERE)
    # 0.05 dB is the .hem's one-decimal rounding; the field is resampled from a 5 deg grid.
    assert np.allclose(stored[:, :, 1:28], original, atol=0.08)


def test_the_files_reference_atmosphere_is_the_one_removed(tmp_path):
    """TAMB/RELHUM/PAMB come from the file, not from the ICAO constants."""
    files = three_hems(tmp_path)
    for f in files:
        edit_constant(f, 'TAMB', '283.15')
        edit_constant(f, 'RELHUM', '30')
        edit_constant(f, 'PAMB', '95000')
    out = tmp_path / 'X.nod'
    build(files, out)
    hem = fa.load_norah2_hemisphere(files[0])
    phis, _, amplitude, _, _, _ = group_grid(out)
    lower = [list(phis).index(p) for p in hem['phi_deg']]
    cold = Atmosphere(temperature=283.15, pressure=95.0, relative_humidity=30.0)
    expected = hem['levels_db'] + offset_db(hem['frequency_hz'], atmosphere=cold)
    present = np.isfinite(hem['levels_db'])
    assert np.allclose(amplitude[lower][present], expected[present], atol=1e-9)
    # A worthwhile difference at 5 kHz, so the test can tell the two apart.
    assert abs(offset_db(5000.0, atmosphere=cold) - offset_db(5000.0)) > 0.5


def test_a_radius_beyond_poldist_adds_the_absorption_between(tmp_path):
    files = three_hems(tmp_path)
    out = tmp_path / 'X.nod'
    build(files, out, radius_m=90.0)
    hem = fa.load_norah2_hemisphere(files[0])
    phis, _, amplitude, _, _, _ = group_grid(out)
    lower = [list(phis).index(p) for p in hem['phi_deg']]
    expected = hem['levels_db'] + offset_db(hem['frequency_hz'], radius=90.0)
    present = np.isfinite(hem['levels_db'])
    assert np.allclose(amplitude[lower][present], expected[present], atol=1e-9)
    with Dataset(str(out)) as nc:
        assert nc['sphere0']['radius'][0] * nc['sphere0']['rotor_scale'][0] == pytest.approx(90.0)


def test_upper_hemisphere_is_the_mirror_with_coverage_zero(tmp_path):
    files = three_hems(tmp_path, starboard_boost_db=6.0)
    out = tmp_path / 'X.nod'
    build(files, out)
    phis, thetas, amplitude, coverage, dba, _ = group_grid(out)
    assert phis.size == 36 and thetas.size == 19
    lower = (phis >= -90.0) & (phis <= 90.0)
    for i, phi in enumerate(phis):
        if lower[i]:
            continue
        source = (180.0 - phi + 180.0) % 360.0 - 180.0
        j = list(phis).index(source)
        assert np.array_equal(amplitude[i], amplitude[j])
        assert np.array_equal(dba[i], dba[j])
    assert not coverage[~lower].any()
    # Coverage 1 exactly where the file has a level, 0 where it has NOVALUE.
    hem = fa.load_norah2_hemisphere(files[0])
    assert np.array_equal(coverage[lower].astype(bool), np.isfinite(hem['levels_db']))


def test_starboard_stays_starboard_and_mirror_rotor_reverses_it(tmp_path):
    files = three_hems(tmp_path, starboard_boost_db=6.0)
    straight = tmp_path / 'straight.nod'
    mirrored = tmp_path / 'mirrored.nod'
    build(files, straight)
    build(files, mirrored, mirror_rotor=True)
    for path, sign in ((straight, 1.0), (mirrored, -1.0)):
        phis, thetas, _, _, dba, _ = group_grid(path)
        at_90 = dba[:, list(thetas).index(90.0)]
        starboard = at_90[list(phis).index(90.0)]
        port = at_90[list(phis).index(-90.0)]
        assert sign * (starboard - port) == pytest.approx(12.0, abs=0.3)
    a = group_grid(straight)
    b = group_grid(mirrored)
    phis = list(a[0])
    for phi in np.arange(-90.0, 91.0, 10.0):
        assert np.array_equal(a[2][phis.index(phi)], b[2][phis.index(-phi)])
    with Dataset(str(mirrored)) as nc:
        assert nc.norah2_mirror_rotor == 1



def drop_last_phi_row(path):
    """Cut a .hem's phi = 90 row, leaving an axis -90..80 that is not symmetric about 0."""
    raw = Path(path).read_bytes().decode('ascii').split('\r\n')
    start = raw.index('PHIOBSAC= 90.000000')
    end = next(i for i in range(start + 1, len(raw)) if not raw[i].strip())
    del raw[start:end]
    axis = raw.index('PHIOBSAC\t19\t0\t3\t0')
    raw[axis] = 'PHIOBSAC\t18\t0\t3\t0'
    assert raw[axis + 1].endswith('\t80\t90\t')
    raw[axis + 1] = raw[axis + 1][:-len('90\t')]
    Path(path).write_bytes('\r\n'.join(raw).encode('ascii'))


def test_mirror_rotor_relabels_an_uneven_azimuth_axis(tmp_path):
    """On -90..80, phi -> -phi gives -80..90: the labels move with the data, not only the rows.

    On the shipped, symmetric axis reversing the rows alone is indistinguishable from
    the mirror; this axis tells them apart.
    """
    files = three_hems(tmp_path, starboard_boost_db=6.0)
    for f in files:
        drop_last_phi_row(f)
    hem = fa.load_norah2_hemisphere(files[0])
    assert list(hem['phi_deg']) == list(np.arange(-90.0, 81.0, 10.0))
    out = tmp_path / 'mirrored.nod'
    build(files, out, mirror_rotor=True)
    phis, _, amplitude, coverage, _, _ = group_grid(out)
    expected = hem['levels_db'] + offset_db(hem['frequency_hz'])
    present = np.isfinite(expected)
    for k, phi in enumerate(hem['phi_deg']):
        row = list(phis).index(-phi)
        assert np.allclose(amplitude[row][present[k]], expected[k][present[k]], atol=1e-9)
        assert np.array_equal(coverage[row].astype(bool), present[k])


def test_novalue_cells_are_missing_and_uncovered(tmp_path):
    """-999 bands and whole directions read as no data: -inf, coverage 0, EAA 0 if empty."""
    files = three_hems(tmp_path, starboard_boost_db=20.0, minimum_level_db=60.0)
    out = tmp_path / 'X.nod'
    build(files, out)
    hem = fa.load_norah2_hemisphere(files[0])
    empty = ~np.isfinite(hem['levels_db']).any(axis=2)
    assert empty.any() and not empty.all()
    phis, _, amplitude, coverage, dba, eaa = group_grid(out)
    lower = [list(phis).index(p) for p in hem['phi_deg']]
    assert np.all(np.isneginf(amplitude[lower][~np.isfinite(hem['levels_db'])]))
    assert not coverage[lower][~np.isfinite(hem['levels_db'])].any()
    assert np.all(np.isneginf(dba[lower][empty]))
    assert np.all(eaa[lower][empty] == 0.0)
    assert np.all(np.isfinite(dba[lower][~empty])) and np.all(np.isfinite(eaa))


def test_fill_empty_takes_the_nearest_direction_and_keeps_coverage_zero(tmp_path):
    files = three_hems(tmp_path, starboard_boost_db=20.0, minimum_level_db=60.0)
    plain = tmp_path / 'plain.nod'
    filled = tmp_path / 'filled.nod'
    build(files, plain)
    build(files, filled, fill_empty=True)
    hem = fa.load_norah2_hemisphere(files[0])
    phis, thetas, a_plain, c_plain, _, _ = group_grid(plain)
    _, _, a_fill, c_fill, dba_fill, _ = group_grid(filled)
    assert np.array_equal(c_plain, c_fill)
    measured = c_plain.astype(bool)
    assert np.array_equal(a_plain[measured], a_fill[measured])
    lower = [list(phis).index(p) for p in hem['phi_deg']]
    # Bands 1..27 exist somewhere, so every lower cell has them after the fill.
    assert np.all(np.isfinite(a_fill[lower][:, :, 1:28]))
    assert np.all(np.isneginf(a_fill[:, :, 0]))       # no file has 10 Hz anywhere
    assert np.all(np.isfinite(dba_fill))
    # One empty cell against a brute-force nearest search.
    levels = hem['levels_db'] + offset_db(hem['frequency_hz'])
    i, j = np.argwhere(~np.isfinite(hem['levels_db'][:, :, 5]))[0]

    def vector(p, t):
        p, t = np.radians(p), np.radians(t)
        return np.array([np.cos(t), np.sin(t) * np.sin(p), np.sin(t) * np.cos(p)])
    target = vector(hem['phi_deg'][i], hem['theta_deg'][j])
    best = max(((target @ vector(hem['phi_deg'][k], hem['theta_deg'][m]), k, m)
                for k in range(19) for m in range(19) if np.isfinite(levels[k, m, 5])),
               key=lambda item: item[0])
    assert a_fill[lower[i], j, 5] == pytest.approx(levels[best[1], best[2], 5], abs=1e-9)
    with Dataset(str(filled)) as nc:
        assert nc.norah2_fill_empty == 'nearest'


def test_speed_mapping_and_speed_reference(tmp_path):
    files = three_hems(tmp_path)
    ground = tmp_path / 'ground.nod'
    air = tmp_path / 'air.nod'
    fixed = tmp_path / 'fixed.nod'
    build(files, ground, 'ground_speed')
    rows = build(files, air, 'ias_to_tas')
    build(files, fixed, 'ias_to_tas', tas_air_density=1.0)
    rho = fa.humid_air_density(MEASURED['temperature_k'], MEASURED['pressure_kpa'] * 1000.0,
                               MEASURED['relative_humidity'])
    for path, reference, factor in ((ground, 'ground', 1.0), (air, 'air', np.sqrt(1.225 / rho)),
                                    (fixed, 'air', np.sqrt(1.225))):
        with Dataset(str(path)) as nc:
            assert nc.speed_reference == reference
            mu = sorted(float(nc[g]['advance_ratio'][0]) for g in nc.groups)
        expected = sorted(0.514444 * v * factor / VEHICLE['main_rotor_tip_speed_mps']
                          for v in (60.0, 80.0, 100.0))
        assert np.allclose(mu, expected, rtol=1e-12)
    assert rows[1]['speed_knots'] == pytest.approx(80.0 * np.sqrt(1.225 / rho))
    with pytest.raises(ValueError, match='speed_mapping is required'):
        build(files, tmp_path / 'none.nod', None)
    with pytest.raises(ValueError, match='ias_to_tas'):
        build(files, tmp_path / 'g.nod', 'ground_speed', tas_air_density=1.0)


def test_ias_to_tas_needs_a_density(tmp_path):
    files = three_hems(tmp_path)
    raw = Path(files[0]).read_bytes().decode('ascii').split('\r\n')
    kept = [line for line in raw if not line.startswith('Pm\t')]
    kept[1] = kept[1].replace('14', '13', 1)
    Path(files[0]).write_bytes('\r\n'.join(kept).encode('ascii'))
    with pytest.raises(ValueError, match='no Pm'):
        build(files, tmp_path / 'x.nod', 'ias_to_tas')
    build(files, tmp_path / 'x.nod', 'ias_to_tas', tas_air_density=1.1)
    build(files, tmp_path / 'y.nod', 'ground_speed')


def test_humid_air_density():
    assert fa.humid_air_density(288.15, 101325.0, 0.0) == pytest.approx(1.2250, abs=1e-4)
    # Water vapor is lighter than dry air.
    assert fa.humid_air_density(298.15, 101325.0, 70.0) < fa.humid_air_density(298.15, 101325.0, 0.0)


def test_root_and_group_attributes(tmp_path):
    files = three_hems(tmp_path)
    out = tmp_path / 'X.nod'
    build(files, out, thrust_air_density=1.1,
          atmosphere=Atmosphere(temperature=290.0, pressure=99.0, relative_humidity=55.0))
    with Dataset(str(out)) as nc:
        assert int(nc['database_version'][...]) == 1
        assert bool(nc['same_grid'][...]) and bool(nc['fixed_load_factor'][...])
        assert bool(nc['shared_grid_and_frequency'][...])
        assert nc.azimuth_reference == 'track'
        assert nc.speed_reference == 'ground'
        assert (nc.build_temperature_K, nc.build_pressure_kPa, nc.build_relative_humidity_percent) \
            == (290.0, 99.0, 55.0)
        assert float(nc['main_rotor_radius_meters'][...]) == VEHICLE['main_rotor_radius_m']
        assert float(nc['main_rotor_tip_speed_meters_per_sec'][...]) == VEHICLE['main_rotor_tip_speed_mps']
        assert float(nc['vehicle_weight_newtons'][...]) == pytest.approx(VEHICLE['vehicle_weight_newtons'])
        ct = VEHICLE['vehicle_weight_newtons'] / (1.1 * np.pi * 5.03 ** 2 * 214.88 ** 2)
        assert len(nc.groups) == 3
        for name, group in nc.groups.items():
            assert int(group['DOPPLER_SHIFT_REMOVED'][...]) == 0
            assert float(group['thrust_coefficient'][0]) == pytest.approx(ct, rel=1e-12)
            assert group.norah2_POLDIST == 60.0 and group.norah2_FREEFIELD == 2.0
        assert nc['sphere1'].norah2_source == 'X_1.hem'
        assert float(nc['sphere1']['flight_path_angle'][0]) == 0.0
        assert float(nc['sphere0']['flight_path_angle'][0]) == -6.0


def test_vehicle_data_is_required(tmp_path):
    files = three_hems(tmp_path)
    for missing in ('main_rotor_tip_speed_mps', 'main_rotor_radius_m', 'vehicle_weight_newtons'):
        arguments = dict(VEHICLE, speed_mapping='ground_speed')
        arguments[missing] = None
        with pytest.raises(ValueError, match=missing):
            fa.build_database_from_norah2(files, str(tmp_path / 'x.nod'), **arguments)
    assert not (tmp_path / 'x.nod').exists()


def test_ground_included_tables_are_refused_unless_accepted(tmp_path):
    files = three_hems(tmp_path)
    edit_constant(files[1], 'FREEFIELD', '0')
    out = tmp_path / 'X.nod'
    with pytest.raises(ValueError, match='FREEFIELD = 0'):
        build(files, out)
    assert not out.exists()
    with pytest.warns(UserWarning, match='counted twice'):
        build(files, out, accept_ground_included=True)
    with Dataset(str(out)) as nc:
        assert nc.norah2_ground_included_accepted == 1
        assert nc['sphere1'].norah2_FREEFIELD == 0.0
    edit_constant(files[1], 'FREEFIELD', '1')
    with pytest.raises(ValueError, match='FREEFIELD = 1'):
        build(files, out, accept_ground_included=True)


def test_one_axis_hover_tables_are_refused(tmp_path):
    """NORAH2's in-ground hover table: one PHIOBSEC axis, no elevation to make a sphere of."""
    lines = ['B412_HIGE_SoftGrnd', '3\t! # Table constants', 'POLDIST\t70\t!', 'FREEFIELD\t0\t!',
             'NOVALUE\t-999\t!', '1', 'PHIOBSEC\t3\t0\t3\t0', '\t-90\t0\t90',
             '0\t! NPARAD', 'NFREQ\t2', '\t100\t200', 'PHIOBSEC', '-90\t70\t71', '0\t72\t73', '90\t74\t75']
    path = tmp_path / 'hover.hem'
    path.write_bytes('\r\n'.join(lines).encode('ascii'))
    with pytest.raises(ValueError, match='one-axis hover'):
        build([str(path)] + three_hems(tmp_path), tmp_path / 'x.nod', accept_ground_included=True)


def test_reference_atmosphere_units_are_checked(tmp_path):
    files = three_hems(tmp_path)
    edit_constant(files[2], 'TAMB', '25')
    with pytest.raises(ValueError, match='TAMB = 25 is not a temperature in kelvin'):
        build(files, tmp_path / 'x.nod')
    edit_constant(files[2], 'TAMB', '298.15')
    edit_constant(files[2], 'PAMB', '1013.25')
    with pytest.raises(ValueError, match='PAMB'):
        build(files, tmp_path / 'x.nod')


def test_conditions_must_be_distinct_and_span_both_axes(tmp_path):
    a = write_hem(tmp_path / 'a.hem', 60.0, -6.0)
    b = write_hem(tmp_path / 'b.hem', 60.0, -6.0)
    with pytest.raises(ValueError, match='same flight condition'):
        build([a, b], tmp_path / 'x.nod')
    level = [write_hem(tmp_path / f'l{i}.hem', speed, 0.0) for i, speed in enumerate((60.0, 80.0, 100.0))]
    with pytest.raises(ValueError, match='do not vary in both'):
        build(level, tmp_path / 'x.nod')
    # Two are enough for NICE-OPS to refuse a constant axis (it stops the run).
    with pytest.raises(ValueError, match='do not vary in both'):
        build(level[:2], tmp_path / 'x.nod')
    assert not (tmp_path / 'x.nod').exists()
    with pytest.warns(UserWarning, match='nearest condition'):
        build([a], tmp_path / 'one.nod')
    with pytest.warns(UserWarning, match='nearest condition'):
        build([a, level[1]], tmp_path / 'two.nod')


def test_collinear_conditions_are_warned_about(tmp_path):
    """Three conditions on one line: NICE-OPS triangulates no simplex and takes the nearest."""
    line = [write_hem(tmp_path / f'c{i}.hem', speed, angle)
            for i, (speed, angle) in enumerate([(60.0, -6.0), (80.0, -3.0), (100.0, 0.0)])]
    with pytest.warns(UserWarning, match='lie on one line'):
        build(line, tmp_path / 'line.nod')
    assert (tmp_path / 'line.nod').exists()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        build(three_hems(tmp_path), tmp_path / 'ok.nod')


def test_a_truncated_file_is_refused_by_name(tmp_path):
    files = three_hems(tmp_path)
    raw = Path(files[1]).read_bytes().split(b'\r\n')
    Path(files[1]).write_bytes(b'\r\n'.join(raw[:20]))
    with pytest.raises(ValueError, match=r'X_1\.hem: cannot be converted \(it ends early'):
        build(files, tmp_path / 'x.nod')
    assert not (tmp_path / 'x.nod').exists()


def test_build_atmosphere_units_are_checked(tmp_path):
    files = three_hems(tmp_path)
    with pytest.raises(ValueError, match='not in kPa'):
        build(files, tmp_path / 'x.nod',
              atmosphere=Atmosphere(temperature=293.15, pressure=101325.0, relative_humidity=20.0))
    with pytest.raises(ValueError, match='not in kelvin'):
        build(files, tmp_path / 'x.nod',
              atmosphere=Atmosphere(temperature=20.0, pressure=101.325, relative_humidity=20.0))


def test_a_failed_write_leaves_the_existing_output_alone(tmp_path, monkeypatch):
    files = three_hems(tmp_path)
    out = tmp_path / 'X.nod'
    out.write_text('existing')

    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(fa, 'add_sphere_group', fail)
    with pytest.raises(OSError, match='disk full'):
        build(files, out)
    assert out.read_text() == 'existing'
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [Path(f).name for f in files] + ['X.nod'])


def test_overwrite_false_refuses_and_inputs_are_not_outputs(tmp_path):
    files = three_hems(tmp_path)
    out = tmp_path / 'X.nod'
    out.write_text('existing')
    with pytest.raises(FileExistsError):
        build(files, out, overwrite=False)
    with pytest.raises(ValueError, match='one of the input files'):
        build(files, files[0])
    with pytest.raises(ValueError, match='more than once'):
        build(files + [files[0]], out)
    # The same file under another name is the same file.
    link = tmp_path / 'link.hem'
    link.symlink_to(files[0])
    with pytest.raises(ValueError, match='more than once'):
        build(files + [str(link)], out)
    with pytest.raises(ValueError, match='one of the input files'):
        build(files, link)


def test_read_vehicle_rotor_data(tmp_path):
    sections = {'Main Rotor': {'radius': '5.03', 'tip speed': '214.88', 'blades': '2'},
                'Atmosphere': {'density': '1.2', 'temperature': '300'},
                'Vehicle': {'weight': '861', 'drag': '0.3'}}
    json_path = tmp_path / 'R44.json'
    json_path.write_text(json.dumps(sections))
    cfg_path = tmp_path / 'vehicle.cfg'
    cfg_path.write_text(''.join(f'[{name}]\n' + ''.join(f'{k} = {v}\n' for k, v in items.items())
                                for name, items in sections.items()))
    for path in (json_path, cfg_path):
        vehicle = fa.read_vehicle_rotor_data(path)
        assert vehicle == dict(main_rotor_radius_m=5.03, main_rotor_tip_speed_mps=214.88,
                               vehicle_weight_newtons=pytest.approx(861.0 * 9.80665),
                               air_density_kg_m3=1.2)
    del sections['Vehicle']['weight']
    json_path.write_text(json.dumps(sections))
    with pytest.raises(ValueError, match='"weight"'):
        fa.read_vehicle_rotor_data(json_path)


def _run(*args, cwd):
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONPATH=str(REPO))
    return subprocess.run([sys.executable, str(REPO / 'norah2_to_nod.py'), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=600)


def test_command_line(tmp_path):
    files = three_hems(tmp_path)
    vehicle = tmp_path / 'R44.json'
    vehicle.write_text(json.dumps({'Main Rotor': {'radius': '5.03', 'tip speed': '214.88'},
                                   'Atmosphere': {'density': '1.225'}, 'Vehicle': {'weight': '861'}}))
    result = _run(*files, '-o', 'out.nod', '--vehicle', str(vehicle), '--speed', 'ias-to-tas', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'Wrote 3 condition(s)' in result.stdout
    with Dataset(str(tmp_path / 'out.nod')) as nc:
        assert nc.speed_reference == 'air' and len(nc.groups) == 3

    # Refusals: no --speed, no vehicle, an existing output, FREEFIELD = 0.
    result = _run(*files, '-o', 'b.nod', '--vehicle', str(vehicle), cwd=tmp_path)
    assert result.returncode == 2 and '--speed' in result.stderr
    result = _run(*files, '-o', 'b.nod', '--speed', 'ground-speed', cwd=tmp_path)
    assert result.returncode == 2 and 'vehicle is required' in result.stderr
    result = _run(*files, '-o', 'out.nod', '--vehicle', str(vehicle), '--speed', 'ground-speed', cwd=tmp_path)
    assert result.returncode == 2 and '--overwrite' in result.stderr
    edit_constant(files[0], 'FREEFIELD', '0')
    result = _run(*files, '-o', 'c.nod', '--tip-speed', '214.88', '--rotor-radius', '5.03',
                  '--weight-kg', '861', '--speed', 'ground-speed', cwd=tmp_path)
    assert result.returncode == 1 and 'FREEFIELD = 0' in result.stderr
    assert not (tmp_path / 'c.nod').exists()
    result = _run(*files, '-o', 'c.nod', '--tip-speed', '214.88', '--rotor-radius', '5.03',
                  '--weight-kg', '861', '--speed', 'ground-speed', '--accept-ground-included', cwd=tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.mark.data
@needs_norah2
def test_converts_a_shipped_set(tmp_path):
    """The public package's R44 set: every flight hemisphere converts, levels land where expected."""
    files = sorted(glob.glob(os.path.join(NORAH2_HEMISPHERES, 'R44_*.hem')))
    assert len(files) == 17
    out = tmp_path / 'R44.nod'
    rows = build(files, out, thrust_air_density=1.225)
    assert len(rows) == 17
    for k, path in enumerate(files):
        hem = fa.load_norah2_hemisphere(path)
        phis, _, amplitude, coverage, _, _ = group_grid(out, f'sphere{k}')
        lower = [list(phis).index(p) for p in hem['phi_deg']]
        present = np.isfinite(hem['levels_db'])
        reference = Atmosphere(temperature=hem['constants']['TAMB'],
                               pressure=hem['constants']['PAMB'] / 1000.0,
                               relative_humidity=hem['constants']['RELHUM'])
        expected = hem['levels_db'] + offset_db(hem['frequency_hz'], atmosphere=reference)
        assert np.allclose(amplitude[lower][present], expected[present], atol=1e-9)
        assert np.array_equal(coverage[lower].astype(bool), present)
