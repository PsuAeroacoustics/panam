"""Build recipes (build_recipe.py), --recipe on both command lines, and the 2017 release recipe."""

import inspect
import re
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest

import build_empirical_database as database_cli
import build_recipe
import flight_acoustics as fa
import noise_abatement_2017 as na
import nod_database
from sphere_helpers import minimal_hemisphere, write_sphere_directory

REPO = Path(__file__).resolve().parents[1]
RELEASE = REPO / 'docs' / 'recipes' / 'release_2017.toml'

CORRECTION_MODULE = '''
import numpy as np


def correction(phi, theta, frequency):
    """+2 dB ahead of the nose (theta < 90), 0 aft: dB, shaped (phi, theta, frequency)."""
    level = np.where(np.asarray(theta) < 90.0, 2.0, 0.0)
    return np.repeat(level[..., None], len(frequency), axis=-1)


correction.description = 'test correction'
'''


def _write(path, text):
    path.write_text(textwrap.dedent(text))
    return path


def _spheres(tmp_path):
    hemispheres = []
    for i, (speed, angle) in enumerate(((5, 0), (60, -3), (80, 0), (100, 0))):
        hemisphere = minimal_hemisphere()
        hemisphere['third_octave']['bands_db'] += 3.0 * i
        hemispheres.append((hemisphere, speed, angle))
    return write_sphere_directory(tmp_path / 'spheres', hemispheres)


# --- a database from a recipe -------------------------------------------------------


def test_a_database_from_a_recipe_equals_the_python_call(tmp_path, monkeypatch):
    directory = _spheres(tmp_path)
    (tmp_path / 'recipe_correction.py').write_text(CORRECTION_MODULE)
    recipe = _write(tmp_path / 'recipe.toml', '''
        [database]
        load_factors = [0.0, 1.0, 2.0]
        extended_flight_path_angles = [-24.0, 35.0]
        distance = 500.0
        store_spectrum = false
        atmosphere = {temperature = 283.15, pressure = 100.0, relative_humidity = 60.0}
        hover_correction = "recipe_correction:correction"
        ''')
    from_recipe, from_python = tmp_path / 'recipe.nod', tmp_path / 'python.nod'
    monkeypatch.syspath_prepend(str(tmp_path))
    import recipe_correction as module
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert database_cli.main([str(directory), str(from_recipe), '--recipe', str(recipe)]) == 0
        fa.build_empirical_database(
            str(directory), str(from_python), load_factors=[0.0, 1.0, 2.0],
            extended_flight_path_angles=(-24.0, 35.0), distance=500.0, store_spectrum=False,
            atmosphere=fa.Atmosphere(temperature=283.15, pressure=100.0, relative_humidity=60.0),
            hover_correction=module.correction)
    comparison = nod_database.compare_databases(str(from_recipe), str(from_python))
    assert not comparison.differs(), comparison.summary()
    # ... and the recipe's settings took effect: not the defaults.
    default = tmp_path / 'default.nod'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fa.build_empirical_database(str(directory), str(default))
    assert nod_database.compare_databases(str(from_recipe), str(default)).differs()
    assert nod_database.load_nod(str(from_recipe)).root_attributes['hover_correction'] == 'test correction'


def test_fixed_load_factor_in_a_recipe_is_load_factors_none(tmp_path):
    recipe = build_recipe.load_recipe(_write(tmp_path / 'r.toml', '[database]\nfixed_load_factor = true\n'))
    assert recipe.database_options() == dict(load_factors=None)
    with pytest.raises(ValueError, match='not both'):
        build_recipe.load_recipe(_write(tmp_path / 'r.toml', '''
            [database]
            fixed_load_factor = true
            load_factors = [1.0]
            '''))


# --- refusals -----------------------------------------------------------------------


@pytest.mark.parametrize('text, match', [
    ('[spheres]\nband_snr_gate = 10.0\n', r"\[spheres\] band_snr_gate: unknown key.*did you mean band_snr_gate_db"),
    ('[database]\nload_factor = [1.0]\n', r'\[database\] load_factor: unknown key.*load_factors'),
    ('[hover_source]\nremove_dopler = true\n', r'\[hover_source\] remove_dopler: unknown key'),
    ('[spheres.interpolation]\nkernal = "shepard"\n', r"unknown settings \['kernal'\]"),
    ('[sphere]\nrmax = 20.0\n', r"unknown sections \['sphere'\]"),
    ('[spheres]\nray_model = "x"\n', 'ray models come only from a profile directory'),
    ('[spheres]\noutput_directory = "out"\n', 'refused: an output'),
    ('[spheres]\naircraft = "B407"\n', 'refused: the dataset'),
    ('[database]\ndatabase_filename = "x.nod"\n', 'refused: an output'),
    ('[database]\nhover_correction = 3\n', "'module:callable'"),
    ('[spheres]\natmosphere = {temperature = 290.0, wind = 3.0}\n', 'temperature'),
])
def test_unknown_or_refused_keys_are_refused(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        build_recipe.load_recipe(_write(tmp_path / 'bad.toml', text))


def test_a_refused_recipe_stops_the_command_line(tmp_path, capsys):
    recipe = _write(tmp_path / 'bad.toml', '[spheres]\nband_snr_gate = 10.0\n')
    with pytest.raises(SystemExit):
        na.main(['B407', str(tmp_path / 'out'), '--recipe', str(recipe)])
    assert 'band_snr_gate_db' in capsys.readouterr().err


def test_every_function_keyword_is_a_recipe_key_or_refused():
    for section in ('spheres', 'database'):
        functions = ((fa.build_empirical_database,) if section == 'database'
                     else (na.build_all, na.build_sphere))
        refused = build_recipe.DATABASE_REFUSED if section == 'database' else build_recipe.SPHERE_REFUSED
        for function in functions:
            for name, parameter in inspect.signature(function).parameters.items():
                if parameter.kind is parameter.VAR_KEYWORD:
                    continue
                assert name in build_recipe.allowed_keys(section) or name in refused, (section, name)


# --- paths --------------------------------------------------------------------------


def test_paths_name_local_paths_entries(tmp_path, monkeypatch):
    monkeypatch.setenv('PANAM_RECIPE_TEST_DATA', str(tmp_path / 'data'))
    recipe = build_recipe.load_recipe(_write(tmp_path / 'r.toml', '''
        [spheres]
        reference_directory = "${recipe_test_data}/legacy"
        rmax = 20.0
        '''))
    assert recipe.sphere_options() == dict(reference_directory=str(tmp_path / 'data' / 'legacy'), rmax=20.0)
    # A relative path without ${name} starts at the recipe's directory, not the current one.
    (tmp_path / 'recipes').mkdir()
    relative = build_recipe.load_recipe(_write(tmp_path / 'recipes' / 'r.toml', '''
        [spheres]
        reference_directory = "legacy"
        ray_profiles = "../profiles"
        '''))
    (tmp_path / 'elsewhere').mkdir()
    monkeypatch.chdir(tmp_path / 'elsewhere')
    assert relative.sphere_options() == dict(reference_directory=str(tmp_path / 'recipes' / 'legacy'),
                                             ray_profiles=str(tmp_path / 'profiles'))
    monkeypatch.delenv('PANAM_RECIPE_TEST_DATA')
    monkeypatch.setattr(build_recipe.local_paths, 'CONFIG_FILE', str(tmp_path / 'absent.toml'))
    with pytest.raises(LookupError, match='recipe_test_data'):
        recipe.sphere_options()


# --- command-line precedence ---------------------------------------------------------


def _captured_build_all(monkeypatch):
    calls = []

    def build_all(aircraft, output_directory, **kwargs):
        calls.append(dict(kwargs, aircraft=aircraft, output_directory=output_directory))
        return [dict()], []
    monkeypatch.setattr(na, 'build_all', build_all)
    return calls


def test_explicit_options_override_the_recipe_and_the_recipe_overrides_defaults(tmp_path, monkeypatch):
    calls = _captured_build_all(monkeypatch)
    recipe = _write(tmp_path / 'r.toml', '''
        [spheres]
        band_snr_gate_db = 12.0
        point_stride = 3
        max_response_correction_db = 15.0
        tone_aware = true
        track_check = false

        [spheres.interpolation]
        kernel = "biweight"

        [hover_source]
        remove_doppler = true
        point_stride = 1
        ''')
    # --band-snr-gate-db 10 equals the default, and still overrides the recipe's 12.
    # --no-track-check sets what the recipe sets.  --rmax is not in the recipe.
    assert na.main(['B407', str(tmp_path / 'out'), '--recipe', str(recipe), '--band-snr-gate-db', '10',
                    '--rmax', '20']) == 0
    options = calls.pop()
    assert options['band_snr_gate_db'] == 10.0
    assert options['point_stride'] == 3
    assert options['max_response_correction_db'] == 15.0
    assert options['tone_aware'] is True and options['track_check'] is False
    assert options['interpolation'] == dict(kernel='biweight')
    assert options['rmax'] == 20.0
    assert options['remove_doppler'] is False
    assert options['min_elevation_deg'] == 10.0          # the command line's default
    # A flag that maps to another keyword overrides it too.
    assert na.main(['B407', str(tmp_path / 'out'), '--recipe', str(recipe), '--interpolation', 'aspect=5']) == 0
    assert calls.pop()['interpolation'] == dict(aspect=5.0)
    # --hover-source puts [hover_source] over [spheres].
    assert na.main(['B407', str(tmp_path / 'hover'), '--recipe', str(recipe), '--hover-source',
                    '--runs', '285101']) == 0
    options = calls.pop()
    assert options['remove_doppler'] is True and options['point_stride'] == 1
    assert options['band_snr_gate_db'] == 12.0 and options['runs'] == ['285101']


def test_tone_split_in_a_recipe_sets_tone_aware_as_the_flag_does(tmp_path, monkeypatch):
    calls = _captured_build_all(monkeypatch)
    recipe = _write(tmp_path / 'r.toml', '[spheres]\ntone_split = true\n')
    assert na.main(['B407', str(tmp_path / 'out'), '--recipe', str(recipe)]) == 0
    options = calls.pop()
    assert options['tone_split'] is True and options['tone_aware'] is True
    assert build_recipe.load_recipe(recipe).sphere_options() == dict(tone_split=True, tone_aware=True)
    # Only in [hover_source]: the hover source rebuild sets it, the flight build does not.
    recipe = _write(tmp_path / 'h.toml', '[hover_source]\ntone_split = true\n')
    assert build_recipe.load_recipe(recipe).sphere_options(hover_source=True)['tone_aware'] is True
    assert build_recipe.load_recipe(recipe).sphere_options() == {}
    # [hover_source] tone_split = false turns off the part spheres, not the banding.
    recipe = _write(tmp_path / 'o.toml', '[spheres]\ntone_split = true\n'
                                         '[hover_source]\nremove_doppler = true\ntone_split = false\n')
    assert build_recipe.load_recipe(recipe).sphere_options(hover_source=True) == dict(
        tone_split=False, tone_aware=True, remove_doppler=True)
    for text, section in (('[spheres]\ntone_split = true\ntone_aware = false\n', 'spheres'),
                          ('[spheres]\ntone_aware = false\n[hover_source]\ntone_split = true\n',
                           'hover_source')):
        with pytest.raises(ValueError, match=r'\[{}\] tone_split = true needs tone_aware'.format(section)):
            build_recipe.load_recipe(_write(tmp_path / 'bad.toml', text))


def test_a_key_the_recipe_omits_takes_the_functions_default_on_the_command_line(tmp_path, monkeypatch):
    calls = _captured_build_all(monkeypatch)
    recipe = _write(tmp_path / 'r.toml', '[spheres]\nrmax = 20.0\n')
    assert na.main(['B407', str(tmp_path / 'out')]) == 0
    assert calls.pop()['point_stride'] == 10              # the command line's default
    assert na.main(['B407', str(tmp_path / 'out'), '--recipe', str(recipe)]) == 0
    options = calls.pop()
    assert options['point_stride'] == 1                   # build_all's, as from Python
    assert options['rmax'] == 20.0
    assert na.main(['B407', str(tmp_path / 'out'), '--recipe', str(recipe), '--point-stride', '10']) == 0
    assert calls.pop()['point_stride'] == 10


def test_hover_source_needs_runs(tmp_path, monkeypatch, capsys):
    calls = _captured_build_all(monkeypatch)
    recipe = _write(tmp_path / 'r.toml', '[hover_source]\nremove_doppler = true\n')
    with pytest.raises(SystemExit):
        na.main(['B407', str(tmp_path / 'out'), '--recipe', str(recipe), '--hover-source'])
    assert '--hover-source needs --runs' in capsys.readouterr().err
    assert calls == []


def test_a_build_option_replaces_only_its_field_of_the_recipes_atmosphere(tmp_path):
    recipe = _write(tmp_path / 'r.toml', '''
        [database]
        atmosphere = {temperature = 283.15, pressure = 90.0, relative_humidity = 80.0}
        ''')
    parser = database_cli.make_parser()

    def atmosphere(*extra):
        argv = [str(tmp_path), str(tmp_path / 'db.nod'), '--recipe', str(recipe), *extra]
        value = database_cli.recipe_options(database_cli.parse_arguments(argv, parser), parser, argv)['atmosphere']
        return value.temperature, value.pressure, value.relative_humidity

    assert atmosphere() == (283.15, 90.0, 80.0)
    assert atmosphere('--build-temperature', '300') == (300.0, 90.0, 80.0)
    assert atmosphere('--build-pressure', '100', '--build-humidity', '20') == (283.15, 100.0, 20.0)


def test_the_database_command_line_overrides_its_recipe(tmp_path):
    recipe = _write(tmp_path / 'r.toml', '''
        [database]
        load_factors = [0.0, 1.0]
        distance = 500.0
        store_spectrum = false
        ''')
    parser = database_cli.make_parser()
    argv = [str(tmp_path), str(tmp_path / 'db.nod'), '--recipe', str(recipe), '--fixed-load-factor',
            '--distance', '1000']
    options = database_cli.recipe_options(database_cli.parse_arguments(argv, parser), parser, argv)
    assert options['load_factors'] is None
    assert options['distance'] == 1000.0                 # the default, given explicitly
    assert options['store_spectrum'] is False            # from the recipe
    argv = [str(tmp_path), str(tmp_path / 'db.nod'), '--recipe', str(recipe)]
    options = database_cli.recipe_options(database_cli.parse_arguments(argv, parser), parser, argv)
    assert options['load_factors'] == [0.0, 1.0] and options['distance'] == 500.0


# --- the new noise_abatement_2017 options ---------------------------------------------


def test_every_build_keyword_has_an_option_with_the_functions_default():
    options = na.build_options(na.make_parser().parse_args(['B407', 'out']))
    defaults = {name: parameter.default
                for function in (na.build_all, na.build_sphere)
                for name, parameter in inspect.signature(function).parameters.items()
                if parameter.default is not inspect.Parameter.empty}
    python_only = {'atmosphere', 'ground', 'absorption_profile', 'samples_path', 'ray_model', 'wind',
                   'reference_sphere'}
    assert set(defaults) - python_only - set(options) == set()
    # point_stride: the command line keeps the 2017 release's 10 (its help says so).
    for name, value in options.items():
        if name in defaults and name != 'point_stride':
            assert value == defaults[name], name


def test_the_new_options_reach_their_keywords():
    options = na.build_options(na.make_parser().parse_args([
        'B407', 'out', '--max-response-correction-db', '15', '--r-ref-ft', '50', '--window-time', '1',
        '--window-overlap', '0.75', '--azi-step', '2', '--elv-step', '3', '--rmax', '20',
        '--interpolation', 'kernel=shepard', '--interpolation', 'aspect=5', '--interpolation',
        'relax_min_mics=false', '--rim-elevation-deg', '12', '--max-rim-range-ft', '3000',
        '--remove-doppler', '--air-relative-doppler', '--nose-from-heading', '--speed-of-sound-ft-s', '1125',
        '--no-absorption-deprop', '--absorption-band-method', 'arp5534', '--flip-y-for-geometry',
        '--no-overwrite', '--sphere-prefix', 'X', '--ray-profiles', 'profiles']))
    assert options['interpolation'] == dict(kernel='shepard', aspect=5.0, relax_min_mics=False)
    expected = dict(max_response_correction_db=15.0, r_ref_ft=50.0, window_time=1.0, window_overlap=0.75,
                    azi_step=2.0, elv_step=3.0, rmax=20.0, rim_elevation_deg=12.0, max_rim_range_ft=3000.0,
                    remove_doppler=True, air_relative_doppler=True, nose_from_heading=True,
                    speed_of_sound_ft_s=1125.0, apply_absorption_deprop=False, absorption_band_method='arp5534',
                    flip_y_for_geometry=True, overwrite=False, sphere_prefix='X', ray_profiles='profiles')
    assert {name: options[name] for name in expected} == expected
    assert na.build_options(na.make_parser().parse_args(
        ['B407', 'out', '--adaptive-interpolation']))['interpolation'] == {}
    with pytest.raises(SystemExit):
        na.make_parser().parse_args(['B407', 'out', '--interpolation', 'radius=3'])


def test_profile_ray_models_trace_runs_with_a_profile_and_leave_the_rest_straight(tmp_path, monkeypatch):
    (tmp_path / '285101.csv').write_text('height,temperature\n')
    test = type('Test', (), {})()
    test.by_run = {'285101': dict(true_heading='270'), '285102': dict(true_heading='270')}
    test.track_path = lambda run: run
    monkeypatch.setattr(na, 'load_track', lambda path: {})
    seen = []
    monkeypatch.setattr(na.refracted_rays, 'external_ray_model',
                        lambda executable, path, **kwargs: seen.append((executable, path, kwargs)) or 'model')
    models = na.profile_ray_models(test, str(tmp_path), executable=str(tmp_path / 'tracer'))
    assert models('285102') is None
    assert models('285101') == 'model'
    assert seen == [(str(tmp_path / 'tracer'), str(tmp_path / '285101.csv'),
                     dict(frame_bearing_deg=270.0, refraction='stratified', receiver='plate', length_units='ft'))]
    with pytest.raises(FileNotFoundError):
        na.profile_ray_models(test, str(tmp_path / 'absent'))


def test_profile_ray_models_pickle_for_parallel_builds(tmp_path, monkeypatch):
    # build_all(jobs=N) sends ray_models to spawned workers, so the recipe's ray_profiles
    # (the 2017 release uses them) must survive a pickle round trip.
    import pickle
    (tmp_path / '285101.csv').write_text('height,temperature\n')
    test = type('Test', (), {})()
    test.aircraft, test.root = 'B407', str(tmp_path)
    test.by_run = {'285101': dict(true_heading='270')}
    test.track_path = lambda run: run
    opened = []
    monkeypatch.setattr(na, 'NoiseAbatementTest',
                        lambda aircraft, root=None: opened.append((aircraft, root)) or test)
    monkeypatch.setattr(na, 'load_track', lambda path: {})
    monkeypatch.setattr(na.refracted_rays, 'external_ray_model', lambda executable, path, **kwargs: 'model')
    models = pickle.loads(pickle.dumps(
        na.profile_ray_models(test, str(tmp_path), executable=str(tmp_path / 'tracer'))))
    assert models('285102') is None and opened == []
    assert models('285101') == 'model'
    assert opened == [('B407', str(tmp_path))]


def test_build_all_takes_ray_models_or_ray_profiles_not_both(tmp_path, monkeypatch):
    monkeypatch.setattr(na, 'NoiseAbatementTest', lambda aircraft, root=None: None)
    with pytest.raises(ValueError, match='not both'):
        na.build_all('B407', str(tmp_path), runs=[], ray_models=lambda run: None, ray_profiles=str(tmp_path))


# --- the 2017 release recipe ----------------------------------------------------------


def _release_row(setting):
    """The release value cell of the database_build.md settings-table row for ``setting``."""
    text = (REPO / 'docs' / 'database_build.md').read_text()
    rows = [line for line in text.splitlines() if line.startswith('|') and '`{}'.format(setting) in line]
    assert len(rows) == 1, setting
    return rows[0].strip('|').split('|')[-1]


def test_the_release_recipe_parses_and_matches_the_settings_table():
    recipe = build_recipe.load_recipe(RELEASE)
    spheres = recipe.spheres
    assert spheres == dict(
        band_snr_gate_db=10.0, max_response_correction_db=15.0, max_absorption_correction_db=30.0,
        max_propagation_range_ft=2000.0, min_elevation_deg=2.0, point_stride=10, board_correction='plate_bem',
        third_octave_method='fft', tone_aware=True, azi_step=2.0, elv_step=2.0, track_check=False,
        azimuth_reference='track', reference_directory='${b407_legacy_spheres}',
        ray_profiles='${b407_ray_profiles}', interpolation=dict(kernel='shepard', aspect=5.0, shepard_floor=1.0))
    assert recipe.hover_source == dict(remove_doppler=True)
    assert recipe.database == dict(load_factors=[0.0, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0],
                                   extended_flight_path_angles=[-24.0, 35.0], store_spectrum=False)
    # Each value appears in the table's release column.
    for setting in ('band_snr_gate_db', 'max_response_correction_db', 'max_absorption_correction_db',
                    'max_propagation_range_ft', 'min_elevation_deg'):
        assert re.search(r'\b{:g}\b'.format(spheres[setting]), _release_row(setting)), setting
    assert '(10)' in _release_row('point_stride')
    assert '`plate_bem`' in _release_row('board_correction')
    assert '`tone_aware=True`' in _release_row('third_octave_method')
    assert '2° × 2°' in _release_row('interpolation')
    for key, value in spheres['interpolation'].items():
        text = '`{}` `{}`'.format(key, value) if isinstance(value, str) else '`{}` {:g}'.format(key, value)
        assert text in _release_row('interpolation'), key
    assert 'not applied' in _release_row('track_check')
    assert '`track`' in _release_row('azimuth_reference')
    assert '0, 1, 1.1, 1.2, 1.5, 2, 3 and 5' in _release_row('load_factors')
    assert '-24° and +35°' in _release_row('extended_flight_path_angles')
    assert '`build_sphere(remove_doppler=True)`' in _release_row('hover_source')


def test_the_release_recipe_needs_its_data_only_for_the_spheres(monkeypatch, tmp_path):
    monkeypatch.setattr(build_recipe.local_paths, 'CONFIG_FILE', str(tmp_path / 'absent.toml'))
    for name in ('B407_LEGACY_SPHERES', 'B407_RAY_PROFILES'):
        monkeypatch.delenv('PANAM_' + name, raising=False)
    recipe = build_recipe.load_recipe(RELEASE)
    assert recipe.database_options()['store_spectrum'] is False
    with pytest.raises(LookupError, match='b407_legacy_spheres'):
        recipe.sphere_options()
    monkeypatch.setenv('PANAM_B407_LEGACY_SPHERES', str(tmp_path / 'legacy'))
    monkeypatch.setenv('PANAM_B407_RAY_PROFILES', str(tmp_path / 'profiles'))
    options = recipe.sphere_options(hover_source=True)
    assert options['reference_directory'] == str(tmp_path / 'legacy')
    assert options['ray_profiles'] == str(tmp_path / 'profiles')
    assert options['remove_doppler'] is True
    assert np.isclose(options['min_elevation_deg'], 2.0)


def test_a_recipes_merge_repeats_admits_a_command_line_merge_speed_tolerance(tmp_path):
    recipe = _write(tmp_path / 'r.toml', '''
        [database]
        merge_repeats = true
        ''')
    parser = database_cli.make_parser()
    argv = [str(tmp_path), str(tmp_path / 'db.nod'), '--recipe', str(recipe),
            '--merge-speed-tolerance', '1.5']
    options = database_cli.recipe_options(database_cli.parse_arguments(argv, parser), parser, argv)
    assert options['merge_repeats'] is True and options['merge_speed_tolerance'] == 1.5
    plain = _write(tmp_path / 'plain.toml', '[database]\ndistance = 500.0\n')
    with pytest.raises(SystemExit):
        database_cli.main([str(tmp_path), str(tmp_path / 'db2.nod'), '--recipe', str(plain),
                           '--merge-speed-tolerance', '1.5'])


def test_recipes_take_the_options_of_the_other_feature_branches(tmp_path):
    recipe = build_recipe.load_recipe(_write(tmp_path / 'r.toml', '''
        [spheres]
        node_statistics = true
        jobs = 2
        skip_unchanged = true

        [database]
        merge_repeats = true
        merge_speed_tolerance = 1.0
        '''))
    assert recipe.sphere_options() == dict(node_statistics=True, jobs=2, skip_unchanged=True)
    assert recipe.database_options() == dict(merge_repeats=True, merge_speed_tolerance=1.0)
