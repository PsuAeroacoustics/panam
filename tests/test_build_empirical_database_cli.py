"""build_empirical_database.py's options, and the sphere selection they reach.

The parse tests check that every option maps to its build_empirical_database
keyword and that the defaults are the function's own.  The build tests write a
small synthetic sphere directory and check that each option reaches the file,
except --doppler-removed: the synthetic spheres carry their own
DOPPLER_SHIFT_REMOVED flag, which takes precedence, so the parse test covers it.
"""

import csv
import inspect
import warnings

import numpy as np
import pytest
from netCDF4 import Dataset

import flight_acoustics as fa
import noise_abatement_2017 as na
import build_empirical_database as script
from sphere_helpers import minimal_hemisphere, write_sphere_directory


def _directory(tmp_path, conditions=((60, -3), (80, 0), (100, 0))):
    hemispheres = []
    for i, (speed, angle) in enumerate(conditions):
        hemisphere = minimal_hemisphere()
        hemisphere['third_octave']['bands_db'] += 3.0 * i
        hemispheres.append((hemisphere, speed, angle))
    return write_sphere_directory(tmp_path / 'spheres', hemispheres)


def _build(tmp_path, *flags):
    """Run the CLI's main on the synthetic directory; return the database path."""
    directory = tmp_path / 'spheres'
    if not directory.exists():
        _directory(tmp_path)
    database = tmp_path / 'database.nod'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert script.main([str(directory), str(database), *flags]) == 0
    return database


# --- parsing ------------------------------------------------------------------------


def test_defaults_are_the_functions_defaults(tmp_path):
    options = script.build_options(script.parse_arguments([str(tmp_path), str(tmp_path / 'db.nod')]))
    defaults = {name: parameter.default for name, parameter in
                inspect.signature(fa.build_empirical_database).parameters.items()
                if parameter.default is not inspect.Parameter.empty}
    # load_factors is left to the function, so its default array is used as is.
    assert 'load_factors' not in options
    # Every other keyword but the Python-only callable has an option.
    assert set(options) == set(defaults) - {'load_factors', 'hover_correction'}
    for name, value in options.items():
        if name == 'atmosphere':
            for field in ('temperature', 'pressure', 'relative_humidity'):
                assert getattr(value, field) == getattr(defaults[name], field)
        else:
            assert value == defaults[name], name


def test_every_option_maps_to_its_keyword(tmp_path):
    args = script.parse_arguments([
        str(tmp_path), str(tmp_path / 'db.nod'), '--load-factors', '0,1,1.5', '--extended-fpa=-24:35',
        '--level-flight-tolerance', '3', '--hover-source', 'h.nc', '--speed-reference', 'air',
        '--azimuth-reference', 'heading', '--doppler-removed', '1', '--distance', '500',
        '--frequency-range', '50:5000', '--build-temperature', '280', '--build-pressure', '100',
        '--build-humidity', '50', '--no-spectrum', '--no-clamp-empty', '--sphere', 'a.nc',
        '--sphere', 'b.nc', '--exclude', 'c*.nc'])
    options = script.build_options(args)
    atmosphere = options.pop('atmosphere')
    assert (atmosphere.temperature, atmosphere.pressure, atmosphere.relative_humidity) == (280, 100, 50)
    assert options == dict(load_factors=[0.0, 1.0, 1.5], extended_flight_path_angles=(-24.0, 35.0),
                           level_flight_tolerance=3.0, hover_source='h.nc', speed_reference='air',
                           azimuth_reference='heading', doppler_shift_removed=1, distance=500.0,
                           infreqs=(50.0, 5000.0), store_spectrum=False, clamp_empty_directions=False,
                           spheres=['a.nc', 'b.nc'], exclude=['c*.nc'])


def test_fixed_load_factor_passes_none(tmp_path):
    args = script.parse_arguments([str(tmp_path), str(tmp_path / 'db.nod'), '--fixed-load-factor'])
    assert script.build_options(args)['load_factors'] is None


@pytest.mark.parametrize('flags', [
    ['--fixed-load-factor', '--load-factors', '1,2'],
    ['--skip-flagged'],
    ['--manifest', 'm.csv', '--sphere', 'a.nc'],
    ['--doppler-removed', '2'],
    ['--speed-reference', 'wind'],
    ['--extended-fpa', '-24'],
    ['--load-factors', ''],
    ['--load-factors', ','],
])
def test_bad_combinations_are_refused(tmp_path, flags, capsys):
    with pytest.raises(SystemExit):
        script.parse_arguments([str(tmp_path), str(tmp_path / 'db.nod'), *flags])


def test_existing_output_needs_overwrite(tmp_path):
    existing = tmp_path / 'db.nod'
    existing.write_bytes(b'')
    with pytest.raises(SystemExit):
        script.parse_arguments([str(tmp_path), str(existing)])
    assert script.parse_arguments([str(tmp_path), str(existing), '--overwrite']).overwrite


def test_help_says_hover_correction_is_python_only(capsys):
    with pytest.raises(SystemExit):
        script.parse_arguments(['--help'])
    assert 'hover_correction' in capsys.readouterr().out


# --- options reach the file ---------------------------------------------------------


def test_options_reach_the_database(tmp_path):
    database = _build(tmp_path, '--fixed-load-factor', '--extended-fpa=-24:35', '--speed-reference', 'air',
                      '--azimuth-reference', 'heading', '--build-temperature', '280',
                      '--build-pressure', '100', '--build-humidity', '50', '--no-spectrum',
                      '--hover-source', str(tmp_path / 'spheres' / 'X102.nc'))
    with Dataset(str(database)) as db:
        assert bool(db['fixed_load_factor'][()])
        assert db.speed_reference == 'air'
        assert db.azimuth_reference == 'heading'
        assert (db.build_temperature_K, db.build_pressure_kPa, db.build_relative_humidity_percent) == \
            (280.0, 100.0, 50.0)
        assert db.hover_source == 'X102.nc'
        groups = list(db.groups.values())
        assert all('amplitude' not in g.variables for g in groups)
        angles = {float(g['flight_path_angle'][0]) for g in groups}
        assert {-24.0, 35.0} <= angles
        # One thrust coefficient per condition: the 1 g reference only.
        assert len({float(g['thrust_coefficient'][0]) for g in groups}) == 1


def test_load_factors_reach_the_database(tmp_path):
    database = _build(tmp_path, '--load-factors', '1,2')
    with Dataset(str(database)) as db:
        # C_T = n C_W: the two load factors give two thrust coefficients, one twice the other.
        thrust = sorted({float(g['thrust_coefficient'][0]) for g in db.groups.values()})
        assert len(thrust) == 2 and thrust[1] == pytest.approx(2 * thrust[0])
        assert not bool(db['fixed_load_factor'][()])


def _groups(database):
    with Dataset(str(database)) as db:
        return db['frequency'][:].data.copy(), {name: dict(
            eaa=g['EAA'][:].filled(np.nan), advance_ratio=float(g['advance_ratio'][0]),
            source=g.source_sphere) for name, g in db.groups.items()}


def test_frequency_range_reaches_the_database(tmp_path):
    # The synthetic spheres have bands at 100, 125 and 160 Hz.
    frequency, _ = _groups(_build(tmp_path, '--fixed-load-factor', '--frequency-range', '110:200'))
    assert list(frequency) == [125.0, 160.0]


def test_distance_reaches_the_database(tmp_path):
    _, near = _groups(_build(tmp_path, '--fixed-load-factor'))
    (tmp_path / 'database.nod').unlink()
    _, far = _groups(_build(tmp_path, '--fixed-load-factor', '--distance', '500'))
    assert set(near) == set(far)
    for name in near:
        assert not np.allclose(near[name]['eaa'], far[name]['eaa']), name


def test_level_flight_tolerance_chooses_the_hover_source(tmp_path):
    # Conditions: X101 60 kt at -3 deg, X102 80 kt and X103 100 kt level.
    def hover_sources(*flags):
        database = _build(tmp_path, '--fixed-load-factor', *flags)
        _, groups = _groups(database)
        database.unlink()
        return {g['source'] for g in groups.values() if g['advance_ratio'] == 0}
    assert hover_sources() == {'X102.nc'}
    assert hover_sources('--level-flight-tolerance', '4') == {'X101.nc'}


def test_no_clamp_empty_reaches_the_database(tmp_path):
    # One sphere has a direction that ambient gating emptied in every band: -999 on disk.
    directory = _directory(tmp_path)
    with Dataset(str(directory / 'X103.nc'), 'a') as sphere:
        sphere['AMPLITUDE'][4, 6, :] = -999.0

    def eaa_of_x103(*flags):
        database = _build(tmp_path, '--fixed-load-factor', *flags)
        _, groups = _groups(database)
        database.unlink()
        return np.concatenate([g['eaa'].ravel() for g in groups.values() if g['source'] == 'X103.nc'])
    assert np.all(np.isfinite(eaa_of_x103()))
    assert np.any(np.isnan(eaa_of_x103('--no-clamp-empty')))


# --- sphere selection ---------------------------------------------------------------


def _source_spheres(database):
    with Dataset(str(database)) as db:
        return db.source_spheres.split('\n'), getattr(db, 'excluded_spheres', None), \
            {g.source_sphere for g in db.groups.values()}


def test_every_sphere_is_recorded_by_default(tmp_path):
    used, excluded, sources = _source_spheres(_build(tmp_path))
    assert used == ['X101.nc', 'X102.nc', 'X103.nc']
    assert excluded is None
    assert sources == set(used)


def test_exclude_leaves_spheres_out_and_records_them(tmp_path):
    used, excluded, sources = _source_spheres(_build(tmp_path, '--exclude', 'X103.nc'))
    assert used == ['X101.nc', 'X102.nc']
    assert excluded == 'X103.nc'
    assert 'X103.nc' not in sources


def test_explicit_spheres_and_glob_exclude(tmp_path):
    directory = _directory(tmp_path)
    database = tmp_path / 'explicit.nod'
    fa.build_empirical_database(str(directory), str(database), load_factors=None,
                                spheres=['X101.nc', str(directory / 'X102.nc'), 'X103.nc'],
                                exclude=['*3.nc'], azimuth_reference='track')
    used, excluded, sources = _source_spheres(database)
    assert used == ['X101.nc', 'X102.nc']
    assert excluded == 'X103.nc'
    # File names only: no directory reaches the file.
    with Dataset(str(database)) as db:
        assert str(tmp_path) not in db.source_spheres + db.excluded_spheres


def test_a_missing_explicit_sphere_is_refused_before_writing(tmp_path):
    directory = _directory(tmp_path)
    database = tmp_path / 'missing.nod'
    with pytest.raises(FileNotFoundError, match='X999.nc'):
        fa.build_empirical_database(str(directory), str(database), spheres=['X101.nc', 'X999.nc'])
    assert not database.exists()


def test_excluding_everything_is_refused_and_a_stray_pattern_warns(tmp_path):
    directory = _directory(tmp_path)
    with pytest.raises(ValueError, match='No sphere files'):
        fa.build_empirical_database(str(directory), str(tmp_path / 'none.nod'), exclude=['*.nc'])
    assert not (tmp_path / 'none.nod').exists()
    with pytest.warns(UserWarning, match='matches no sphere'):
        fa._select_sphere_files(str(directory), exclude=['Y*.nc'])


def _write_manifest(path, rows):
    fields = ['run', 'output', 'condition', 'track_flags', 'error']
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


MANIFEST_ROWS = [
    dict(run='101', output='/elsewhere/spheres/X101.nc', condition='LFO 60', track_flags=''),
    dict(run='102', output='C:\\build\\spheres\\X102.nc', condition='LFO 80',
         track_flags='its window is flown at -1.0 deg'),
    dict(run='103', output='/elsewhere/spheres/X103.nc', condition='LFO 100', track_flags='not checked'),
    dict(run='104', output='', condition='LFO 120', error='no ambient'),
]


def test_select_from_manifest(tmp_path):
    manifest = tmp_path / 'manifest.csv'
    _write_manifest(manifest, MANIFEST_ROWS)
    assert na.select_from_manifest(manifest) == ['X101.nc', 'X102.nc', 'X103.nc']
    assert na.select_from_manifest(manifest, skip_flagged=True) == ['X101.nc', 'X103.nc']
    assert na.select_from_manifest(manifest, conditions=['LFO 80']) == ['X102.nc']


def test_manifest_selects_the_spheres_built(tmp_path):
    """A sphere left in the directory that the manifest does not list is not used."""
    _directory(tmp_path, conditions=((60, -3), (80, 0), (100, 0), (120, 0)))
    manifest = tmp_path / 'manifest.csv'
    _write_manifest(manifest, MANIFEST_ROWS)
    used, _, _ = _source_spheres(_build(tmp_path, '--manifest', str(manifest)))
    assert used == ['X101.nc', 'X102.nc', 'X103.nc']
    (tmp_path / 'database.nod').unlink()
    used, _, _ = _source_spheres(_build(tmp_path, '--manifest', str(manifest), '--skip-flagged'))
    assert used == ['X101.nc', 'X103.nc']


def test_default_selection_leaves_levels_unchanged(tmp_path):
    """Selecting every sphere explicitly gives the same levels as the directory default."""
    directory = _directory(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fa.build_empirical_database(str(directory), str(tmp_path / 'a.nod'), load_factors=None)
        fa.build_empirical_database(str(directory), str(tmp_path / 'b.nod'), load_factors=None,
                                    spheres=['X103.nc', 'X101.nc', 'X102.nc'])
    with Dataset(str(tmp_path / 'a.nod')) as a, Dataset(str(tmp_path / 'b.nod')) as b:
        assert list(a.groups) == list(b.groups)
        for name in a.groups:
            np.testing.assert_array_equal(a[name]['dBA'][:], b[name]['dBA'][:])
