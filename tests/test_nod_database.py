"""nod_database: reading, checking and comparing NICEOPS databases.

conftest.py also runs validate_nod on every database the other tests build.
"""
import sys

import numpy as np
import pytest
from netCDF4 import Dataset

from panam import flight_acoustics as fa
from panam import nod_database as nd
from sphere_helpers import minimal_hemisphere, write_sphere_directory

# sphere_helpers.VEHICLE_CFG: 2250 kg, 5.334 m, 230.7 m/s, 1.070 kg/m^3.
WEIGHT_COEFFICIENT = 2250 * fa.STANDARD_GRAVITY / (1.070 * np.pi * 5.334 ** 2 * 230.7 ** 2)
LADDER = (0.7, 1.1, 1.5, 1.9, 2.3)


def _spheres(tmp_path, offset_db=0.0, name='spheres'):
    hemispheres = []
    for speed, angle in ((60.0, 0.0), (80.0, -6.0)):
        hemisphere = minimal_hemisphere()
        hemisphere['third_octave']['bands_db'] += offset_db
        # Gated cells, so that coverage and -inf vary.
        hemisphere['third_octave']['bands_db'][2, 2:, :] = -np.inf
        hemisphere['third_octave']['bands_db'][1, :, 3] = -np.inf
        hemispheres.append((hemisphere, speed, angle))
    return write_sphere_directory(tmp_path / name, hemispheres)


def _build(tmp_path, name='db.nod', offset_db=0.0, **options):
    options.setdefault('load_factors', None)
    path = tmp_path / name
    with pytest.warns(UserWarning):  # the spheres carry no azimuth_reference or Doppler flag
        fa.build_empirical_database(str(_spheres(tmp_path, offset_db, name + '.spheres')), str(path),
                                    **options)
    return path


@pytest.fixture(scope='module')
def shared(tmp_path_factory):
    return _build(tmp_path_factory.mktemp('shared'))


def _rules(path):
    return {violation.rule for violation in nd.validate_nod(path)}


def test_every_build_in_the_tests_is_validated():
    assert fa.build_empirical_database.__wrapped__ is not None
    assert fa.build_database_from_norah2.__wrapped__ is not None


def test_a_fixed_load_factor_database_reads_back_as_written(shared):
    database = nd.load_nod(shared)
    assert database.fixed_load_factor and database.shared_grid_and_frequency
    assert [c.name for c in database] == ['sphere{}'.format(i) for i in range(5)]
    np.testing.assert_allclose([c.speed_kt for c in database], [60.0, 80.0, 0.0, 0.0, 0.0])
    assert [c.flight_path_angle_deg for c in database] == [0.0, -6.0, -12.0, 0.0, 12.0]
    assert all(c.load_factor == 1.0 for c in database)
    np.testing.assert_allclose([c.thrust_coefficient for c in database], WEIGHT_COEFFICIENT, rtol=1e-12)
    assert [c.condition_origin for c in database] == ['measured'] * 2 + ['synthesized_hover'] * 3
    with Dataset(str(shared)) as nc:
        group = nc['sphere1']
        np.testing.assert_array_equal(database[1].phi_deg, nc['phi'][:])
        np.testing.assert_array_equal(database[1].frequency_hz, nc['frequency'][:])
        np.testing.assert_array_equal(database[1].amplitude, group['amplitude'][:])
        np.testing.assert_array_equal(database[1].dBA, group['dBA'][:])
        np.testing.assert_array_equal(database[1].coverage, group['coverage'][:] == 1)
    condition = database[1]
    assert condition.radius_m == pytest.approx(30.48)
    assert condition.rotor_scale_m == 5.334
    assert condition.doppler_shift_removed == 0
    assert condition.amplitude.shape == condition.coverage.shape == (36, 37, 3)
    assert condition.attributes['source_sphere'] == 'X102.nc'


def test_coverage_zero_reads_as_no_data(shared):
    """The database stores gated and unmeasured cells alike (-inf, coverage 0);
    measured_amplitude makes every coverage-0 cell NaN, filled ones included."""
    condition = nd.load_nod(shared)[0]
    measured = condition.measured_amplitude
    assert np.isneginf(condition.amplitude).any()
    assert not np.isneginf(measured).any()
    assert np.array_equal(np.isnan(measured), ~condition.coverage)
    # The mirrored upper surface holds levels but no measurement.
    assert (np.isfinite(condition.amplitude) & ~condition.coverage).any()
    np.testing.assert_array_equal(measured[condition.coverage], condition.amplitude[condition.coverage])


def test_per_group_grids_are_read_from_each_group(tmp_path):
    directory = tmp_path / 'mixed'
    write_sphere_directory(directory, [(minimal_hemisphere(), 60.0, 0.0)])
    fa.write_aam_hemisphere_netcdf(str(directory / 'X102.nc'), minimal_hemisphere(), mode='third_octave',
                                   radius_ft=100.0, speed_knots=80.0, flight_path_angle_deg=-6.0,
                                   title='t', theta_deg=np.arange(0.0, 180.0 + 1e-9, 15.0))
    path = tmp_path / 'mixed.nod'
    with pytest.warns(UserWarning):
        fa.build_empirical_database(str(directory), str(path), load_factors=None)
    database = nd.load_nod(path)
    assert not database.shared_grid_and_frequency
    assert database[0].phi_deg.size == 36 * 37 and database[1].phi_deg.size == 36 * 13
    assert database[1].amplitude.shape == (36, 13, 3)
    assert nd.validate_nod(path) == []


def test_a_ladder_needs_the_weight_coefficient_for_its_load_factors(tmp_path):
    path = _build(tmp_path, load_factors=LADDER)
    assert all(np.isnan(c.load_factor) for c in nd.load_nod(path))
    database = nd.load_nod(path, weight_coefficient=WEIGHT_COEFFICIENT)
    np.testing.assert_allclose([c.load_factor for c in database[:5]], LADDER, rtol=1e-12)
    # dBA carries 20 lg n; amplitude does not.
    np.testing.assert_allclose(database[4].dBA - database[0].dBA, 20 * np.log10(2.3 / 0.7))
    np.testing.assert_array_equal(database[4].amplitude, database[0].amplitude)


def test_a_hand_written_ladder_with_the_zero_floor_keeps_the_contract(tmp_path):
    """Groups written by add_sphere_group, as test_load_factor_scaling writes them,
    behind the root every database carries."""
    theta, phi = np.meshgrid([0.0, 90.0, 180.0], [-180.0, -90.0, 0.0, 90.0])
    path = tmp_path / 'ladder.nod'
    with Dataset(str(path), 'w') as db:
        fa._write_database_root(db, fixed_load_factor=False, speed_reference='ground',
                                atmosphere=fa.Atmosphere(293.15, 101.325, 20.0), main_rotor_radius=5.334,
                                main_rotor_tip_speed=230.7, vehicle_weight_newtons=None,
                                azimuth_reference='track')
        for i, lf in enumerate([0.0, 1.0, 2.0, 5.0]):
            fa.add_sphere_group(db, f'sphere{i}', phi, theta, radius=30.48, SPLA=np.full(phi.shape, 90.0),
                                EAA=np.zeros(phi.shape), speed=60.0, flight_path_angle=0.0, load_factor=lf,
                                main_rotor_radius=5.334, main_rotor_tip_speed=230.7,
                                weight_coefficient=0.004341)
    assert nd.validate_nod(path) == []
    database = nd.load_nod(path, weight_coefficient=0.004341)
    np.testing.assert_allclose([c.load_factor for c in database], [0.0, 1.0, 2.0, 5.0])
    assert database[0].amplitude is None and database[0].coverage is None


def test_a_database_without_spectra(tmp_path):
    path = _build(tmp_path, store_spectrum=False)
    condition = nd.load_nod(path)[0]
    assert condition.amplitude is None and condition.frequency_hz is None
    assert condition.measured_amplitude is None
    assert condition.coverage.shape == condition.dBA.shape


def test_a_norah2_database_reads_back(tmp_path):
    from test_norah2_import import build, three_hems
    path = tmp_path / 'norah2.nod'
    build(three_hems(tmp_path), path)
    database = nd.load_nod(path)
    np.testing.assert_allclose(sorted(c.speed_kt for c in database), [60.0, 80.0, 100.0])
    assert all(c.load_factor == 1.0 for c in database)
    assert all(c.condition_origin is None and 'norah2_source' in c.attributes for c in database)


# Each corruption and the rules it must break, no more and no fewer.
def _set(variable, value, index=Ellipsis, group='sphere1'):
    def corrupt(nc):
        target = nc if group is None else nc[group]
        values = np.array(target[variable][...])
        values[index] = value
        target[variable][...] = values
    return corrupt


def _rename_variable(old, group='sphere1'):
    return lambda nc: (nc if group is None else nc[group]).renameVariable(old, old + '_gone')


def _set_attribute(name, value, group=None):
    return lambda nc: setattr(nc if group is None else nc[group], name, value)


def _add_group_phi(nc):
    nc['sphere1'].createVariable('phi', 'f8', ('channels',))[:] = np.array(nc['phi'][:]) + 1.0


def _empty_direction(nc):
    amplitude = np.array(nc['sphere0']['amplitude'][:])
    amplitude[20, 3, :] = -np.inf
    nc['sphere0']['amplitude'][:] = amplitude


def _empty_direction_with_eaa(nc):
    """An emptied direction whose dBA is -inf but whose EAA is not 0."""
    _empty_direction(nc)
    channel = 20 * nc['sphere0'].dimensions['THETA'].size + 3
    dba = np.array(nc['sphere0']['dBA'][:])
    dba[channel] = -np.inf
    nc['sphere0']['dBA'][:] = dba
    eaa = np.array(nc['sphere0']['EAA'][:])
    eaa[channel] = 1.0
    nc['sphere0']['EAA'][:] = eaa


def _phi_in_umapr_range(nc):
    """The shared phi shifted by +180 deg, as a sign or convention mix-up leaves it."""
    nc['phi'][:] = np.array(nc['phi'][:]) + 180.0


CORRUPTIONS = {
    'same_grid false': (_set('same_grid', 0, group=None), {'R1'}),
    'no same_grid': (_rename_variable('same_grid', group=None), {'R1'}),
    'no rotor_scale': (_rename_variable('rotor_scale'), {'R2'}),
    'NaN advance ratio': (_set('advance_ratio', np.nan), {'R2'}),
    'no EAA': (_rename_variable('EAA'), {'R3'}),
    'no shared phi': (_rename_variable('phi', group=None), {'R4'}),
    'radius differs': (_set('radius', 2.0), {'R5'}),
    'NaN EAA': (_set('EAA', np.nan, 0), {'R6'}),
    'NaN dBA': (_set('dBA', np.nan, 0), {'R6'}),
    'future version': (_set('database_version', fa.DATABASE_FORMAT_VERSION + 1, group=None), {'C1'}),
    'unknown speed reference': (_set_attribute('speed_reference', 'sea'), {'C2'}),
    'group phi contradicts the root': (_add_group_phi, {'C3'}),
    'unsorted grid': (_set('phi', 170.0, 0, group=None), {'C4'}),
    'phi in UMAPR range': (_phi_in_umapr_range, {'C13'}),
    'theta past the tail': (_set('theta', 181.0, 1, group=None), {'C4', 'C13'}),
    'NaN band': (_set('amplitude', np.nan, (20, 3, 0)), {'C5', 'C6'}),
    'coverage 2': (_set('coverage', 2, (20, 3, 0)), {'C6'}),
    'coverage of a missing band': (_set('amplitude', -np.inf, (20, 3, 0)), {'C6'}),
    'energy in an empty direction': (_empty_direction, {'C6', 'C7'}),
    'EAA in an empty direction': (_empty_direction_with_eaa, {'C6', 'C7'}),
    'fixed load factor, two C_T': (_set('thrust_coefficient', 0.01), {'C8'}),
    'two Doppler flags': (_set('DOPPLER_SHIFT_REMOVED', 1), {'C9'}),
    'rotor radius differs': (_set('main_rotor_radius_meters', 5.0, group=None), {'C10'}),
    'group renamed': (lambda nc: nc.renameGroup('sphere1', 'extra'), {'C11'}),
    'unknown origin': (_set_attribute('condition_origin', 'guessed', group='sphere1'), {'C12'}),
}


def test_the_eaa_corruption_leaves_dba_alone(shared, tmp_path):
    """'EAA in an empty direction' breaks C7 through EAA only: with EAA restored to 0
    the emptied direction keeps C7."""
    path = tmp_path / 'corrupt.nod'
    path.write_bytes(shared.read_bytes())
    with Dataset(str(path), 'a') as nc:
        nc.set_auto_mask(False)
        _empty_direction_with_eaa(nc)
        channel = 20 * nc['sphere0'].dimensions['THETA'].size + 3
        eaa = np.array(nc['sphere0']['EAA'][:])
        eaa[channel] = 0.0
        nc['sphere0']['EAA'][:] = eaa
    assert {v.rule for v in nd.validate_nod(path)} == {'C6'}


@pytest.mark.parametrize('corruption', sorted(CORRUPTIONS))
def test_a_corrupted_database_breaks_exactly_its_rules(shared, tmp_path, corruption):
    corrupt, rules = CORRUPTIONS[corruption]
    path = tmp_path / 'corrupt.nod'
    path.write_bytes(shared.read_bytes())
    assert nd.validate_nod(path) == []
    with Dataset(str(path), 'a') as nc:
        nc.set_auto_mask(False)
        corrupt(nc)
    violations = nd.validate_nod(path)
    assert {v.rule for v in violations} == rules, [str(v) for v in violations]


def test_violations_name_their_group(shared, tmp_path):
    path = tmp_path / 'corrupt.nod'
    path.write_bytes(shared.read_bytes())
    with Dataset(str(path), 'a') as nc:
        nc['sphere3'].renameVariable('dBA', 'level')
    (violation,) = nd.validate_nod(path)
    assert violation.group == 'sphere3'
    assert str(violation) == 'R3 [sphere3]: no dBA'


def test_a_database_compares_equal_to_itself(shared):
    comparison = nd.compare_databases(shared, nd.load_nod(shared))
    assert not comparison.differs()
    assert len(comparison.conditions) == 5
    assert comparison.summary() == ''


def test_the_build_time_and_file_names_are_not_compared(tmp_path, shared):
    copy = tmp_path / 'copy.nod'
    copy.write_bytes(shared.read_bytes())
    with Dataset(str(copy), 'a') as nc:
        nc.panam_created_utc = '2001-01-01T00:00:00Z'
        nc.panam_inputs = '{"database_filename":"copy.nod"}'
    assert not nd.compare_databases(shared, copy).differs()
    with Dataset(str(copy), 'a') as nc:
        nc.panam_settings = '{"changed":true}'
    assert set(nd.compare_databases(shared, copy).root_differences) == {'panam_settings'}


def test_a_uniform_offset_shows_in_every_band(tmp_path, shared):
    louder = _build(tmp_path, 'louder.nod', offset_db=3.0)
    comparison = nd.compare_databases(shared, louder)
    assert comparison.root_differences == {}
    for condition in comparison.conditions:
        np.testing.assert_allclose(condition.band_difference_db, 3.0, atol=1e-9)
        assert condition.max_abs_band_difference_db == pytest.approx(3.0)
        assert condition.max_abs_dba_difference_db == pytest.approx(3.0)
        assert condition.coverage_gained == condition.coverage_lost == 0
    assert comparison.differs(tolerance_db=2.9) and not comparison.differs(tolerance_db=3.1)


def test_band_differences_are_summed_in_energy(tmp_path, shared):
    """One cell of one band +10 dB: the band's energy rises by 10 lg((N - 1 + 10) / N)."""
    changed = tmp_path / 'changed.nod'
    changed.write_bytes(shared.read_bytes())
    with Dataset(str(changed), 'a') as nc:
        amplitude = np.array(nc['sphere0']['amplitude'][:])
        assert np.all(amplitude[..., 0] == amplitude[20, 3, 0])  # a uniform band
        amplitude[20, 3, 0] += 10.0
        nc['sphere0']['amplitude'][:] = amplitude
    difference = nd.compare_databases(shared, changed).conditions[0]
    finite = amplitude[..., 0].size
    assert difference.band_difference_db[0] == pytest.approx(10 * np.log10((finite - 1 + 10) / finite))
    assert difference.band_difference_db[1:] == pytest.approx([0.0, 0.0])
    assert difference.max_abs_band_difference_db == pytest.approx(10.0)


def test_coverage_gained_and_lost_and_energy_on_one_side(tmp_path, shared):
    changed = tmp_path / 'changed.nod'
    changed.write_bytes(shared.read_bytes())
    with Dataset(str(changed), 'a') as nc:
        coverage = np.array(nc['sphere0']['coverage'][:])
        amplitude = np.array(nc['sphere0']['amplitude'][:])
        lost = np.argwhere(coverage == 1)[0]
        coverage[tuple(lost)] = 0
        amplitude[tuple(lost)] = -np.inf
        gained = np.argwhere((coverage == 0) & np.isfinite(amplitude))[0]
        coverage[tuple(gained)] = 1
        nc['sphere0']['coverage'][:] = coverage
        nc['sphere0']['amplitude'][:] = amplitude
    difference = nd.compare_databases(shared, changed).conditions[0]
    assert (difference.coverage_gained, difference.coverage_lost) == (1, 1)
    assert (difference.finite_only_in_a, difference.finite_only_in_b) == (1, 0)


def test_a_changed_reference_radius_or_doppler_convention_differs(tmp_path, shared):
    """Identical levels with another sphere radius or Doppler convention are a
    different database to NICEOPS."""
    changed = tmp_path / 'changed.nod'
    changed.write_bytes(shared.read_bytes())
    with Dataset(str(changed), 'a') as nc:
        group = nc['sphere0']
        group['radius'][...] = 2.0 * np.asarray(group['radius'][...])
        doppler = int(np.asarray(group['DOPPLER_SHIFT_REMOVED'][...]))
        group['DOPPLER_SHIFT_REMOVED'][...] = 1 - doppler
    comparison = nd.compare_databases(shared, changed)
    assert comparison.differs()
    first = comparison.conditions[0]
    assert set(first.scalar_differences) == {'radius_m', 'doppler_shift_removed'}
    assert first.scalar_differences['doppler_shift_removed'] == (doppler, 1 - doppler)
    assert not first.levels_differ()
    assert not any(c.differs() for c in comparison.conditions[1:])
    summary = comparison.summary()
    assert 'radius_m' in summary and 'doppler_shift_removed' in summary
    assert 'dBA up to' not in summary


def test_unmatched_conditions_and_root_differences(tmp_path, shared):
    ladder = _build(tmp_path, 'ladder.nod', load_factors=(1.0, 2.0), speed_reference='air')
    comparison = nd.compare_databases(shared, ladder)
    assert comparison.root_differences['fixed_load_factor'] == (1, 0)
    assert comparison.root_differences['speed_reference'] == ('ground', 'air')
    # The ladder's 1 g groups match; its 2 g groups do not.
    assert len(comparison.conditions) == 5 and comparison.only_in_a == []
    assert len(comparison.only_in_b) == 5
    assert all(key[2] == pytest.approx(2 * WEIGHT_COEFFICIENT) for _, key in comparison.only_in_b)
    assert not any(c.differs() for c in comparison.conditions)
    assert 'root speed_reference' in comparison.summary()


def test_to_frame_is_one_row_per_condition_channel_and_band(shared):
    pytest.importorskip('pandas')
    frame = nd.to_frame(shared)
    database = nd.load_nod(shared)
    assert len(frame) == sum(c.amplitude.size for c in database)
    row = frame[(frame.condition == 'sphere1')].iloc[3 * 100 + 2]
    condition = database[1]
    assert row.phi_deg == condition.phi_deg[100] and row.frequency_hz == condition.frequency_hz[2]
    assert row.amplitude_db == condition.amplitude.reshape(-1, 3)[100, 2]
    assert row.dBA == condition.dBA[100] and row.speed_kt == pytest.approx(80.0)


def test_to_frame_says_it_needs_pandas(shared, monkeypatch):
    monkeypatch.setitem(sys.modules, 'pandas', None)
    with pytest.raises(ImportError, match='needs pandas'):
        nd.to_frame(shared)

