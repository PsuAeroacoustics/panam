"""The provenance stamp (provenance.py) and the writers that carry it."""

import hashlib
import json
import os
import pathlib
import subprocess

import numpy as np
from netCDF4 import Dataset

import flight_acoustics as fa
import provenance
import sphere_helpers
from panam_acoustics.atmosphere import Atmosphere
from test_norah2_import import VEHICLE, three_hems


def _every_attribute(dataset):
    """Every attribute value of a netCDF file, its groups and variables, as text."""
    values = [str(dataset.getncattr(name)) for name in dataset.ncattrs()]
    for variable in dataset.variables.values():
        values += [str(variable.getncattr(name)) for name in variable.ncattrs()]
    for group in dataset.groups.values():
        values += _every_attribute(group)
    return values


def _assert_no_machine_path(path, *directories):
    with Dataset(str(path)) as dataset:
        text = '\n'.join(_every_attribute(dataset))
    for directory in directories + (os.path.expanduser('~'),):
        for spelling in {str(directory), os.path.realpath(str(directory))}:
            assert spelling not in text, spelling


def _assert_stamp(path, tool):
    with Dataset(str(path)) as dataset:
        stamp = {name: dataset.getncattr(name) for name in provenance.PROVENANCE_ATTRIBUTES}
    assert stamp['panam_tool'] == tool
    assert stamp['panam_version']
    assert stamp['panam_created_utc'].endswith('Z')
    assert stamp['panam_settings_sha256'] == hashlib.sha256(stamp['panam_settings'].encode()).hexdigest()
    return json.loads(stamp['panam_settings']), json.loads(stamp['panam_inputs'])


def test_the_hash_is_the_same_for_equal_settings():
    first = dict(b=np.float64(2.5), a=np.array([1.0, 2.0]), c=dict(y=1, x=(3, 4)))
    second = dict(c=dict(x=[3, 4], y=1), a=[1.0, 2.0], b=2.5)
    one, two = provenance.build_provenance('t', first), provenance.build_provenance('t', second)
    assert one['panam_settings'] == two['panam_settings']
    assert one['panam_settings_sha256'] == two['panam_settings_sha256']
    assert one['panam_settings'] == '{"a":[1.0,2.0],"b":2.5,"c":{"x":[3,4],"y":1}}'
    changed = provenance.build_provenance('t', dict(second, b=2.6))
    assert changed['panam_settings_sha256'] != one['panam_settings_sha256']


def test_paths_callables_and_objects_are_reduced():
    def correction(phi, theta, frequency):
        return 0.0
    correction.description = 'blend'
    settings = provenance.canonical_settings(dict(
        absolute='/Users/someone/data/run.nc', windows='C:\\data\\spheres\\', home='~/x/y.hem',
        path=pathlib.Path('/tmp/z/out.nod'), units='m/s', name='AS350B3',
        function=correction, atmosphere=Atmosphere(293.15, 101.325, 20.0)))
    assert settings['absolute'] == 'run.nc'
    assert settings['windows'] == 'spheres'
    assert settings['home'] == 'y.hem'
    assert settings['path'] == 'out.nod'
    assert settings['units'] == 'm/s' and settings['name'] == 'AS350B3'
    assert settings['function'].endswith('correction: blend')
    assert settings['atmosphere']['type'] == 'Atmosphere'
    assert settings['atmosphere']['temperature'] == 293.15


def test_non_finite_numbers_are_written_as_strict_json():
    assert provenance.settings_json({'x': np.inf}) == '{"x":"inf"}'
    assert provenance.settings_json({'x': [float('nan'), -np.inf, np.float32(1.5)]}) == \
        '{"x":["nan","-inf",1.5]}'

    def refuse(constant):
        raise AssertionError('non-standard JSON constant ' + constant)
    stamp = provenance.build_provenance('t', dict(cap=np.inf, levels=np.array([np.nan, 1.0])))
    assert json.loads(stamp['panam_settings'], parse_constant=refuse) == \
        dict(cap='inf', levels=['nan', 1.0])


def test_inputs_are_recorded_apart_from_the_hashed_settings():
    one = provenance.build_provenance('t', dict(gate=10.0), inputs=dict(run='1', output='/a/b/one.nc'))
    two = provenance.build_provenance('t', dict(gate=10.0), inputs=dict(run='2', output='two.nc'))
    assert one['panam_settings_sha256'] == two['panam_settings_sha256']
    assert json.loads(one['panam_inputs']) == dict(run='1', output='one.nc')
    assert provenance.build_provenance('t', {})['panam_inputs'] == '{}'


def test_the_code_version_falls_back_to_unknown(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError('git')
    monkeypatch.setattr(subprocess, 'run', missing)
    assert provenance.code_version() == 'unknown'
    assert provenance.build_provenance('t', {})['panam_version'] == 'unknown'


def test_the_sphere_writer_stamps_its_arguments_without_paths(tmp_path):
    path = tmp_path / 'deep' / 'X101.nc'
    path.parent.mkdir()
    fa.write_aam_hemisphere_netcdf(str(path), sphere_helpers.minimal_hemisphere(), mode='third_octave',
                                   radius_ft=100.0, speed_knots=60.0, title='t')
    settings, inputs = _assert_stamp(path, 'flight_acoustics.write_aam_hemisphere_netcdf')
    assert inputs == dict(filename='X101.nc') and settings['speed_knots'] == 60.0
    assert 'filename' not in settings
    _assert_no_machine_path(path, tmp_path)

    stamp = provenance.build_provenance('caller', dict(band_snr_gate_db=10.0))
    fa.write_aam_hemisphere_netcdf(str(path), sphere_helpers.minimal_hemisphere(), mode='third_octave',
                                   radius_ft=100.0, provenance=stamp)
    assert _assert_stamp(path, 'caller') == (dict(band_snr_gate_db=10.0), {})


def test_the_empirical_database_records_its_build(tmp_path, monkeypatch):
    hemisphere = sphere_helpers.minimal_hemisphere()
    spheres = sphere_helpers.write_sphere_directory(tmp_path / 'spheres',
                                                    [(hemisphere, 60.0, 0.0), (hemisphere, 80.0, -6.0)])
    # A relative output path: its directory must not reach the stamp either.
    monkeypatch.chdir(tmp_path)
    os.makedirs('out')
    fa.build_empirical_database(str(spheres), os.path.join('out', 'db.nod'),
                                load_factors=[1.0, 2.0], extended_flight_path_angles=(-24.0, 35.0),
                                hover_source=str(spheres / 'X101.nc'))
    path = tmp_path / 'out' / 'db.nod'
    settings, inputs = _assert_stamp(path, 'flight_acoustics.build_empirical_database')
    assert inputs == dict(directory_name='spheres', database_filename='db.nod', hover_source='X101.nc')
    assert not {'directory_name', 'database_filename', 'hover_source'} & set(settings)
    assert settings['hover_source_given'] is True
    assert settings['load_factors'] == [1.0, 2.0]
    with Dataset(str(path)) as database:
        np.testing.assert_array_equal(database.load_factors, [1.0, 2.0])
        np.testing.assert_array_equal(database.extended_flight_path_angles_deg, [-24.0, 35.0])
    _assert_no_machine_path(path, tmp_path)

    fixed = tmp_path / 'fixed.nod'
    fa.build_empirical_database(str(spheres), str(fixed), load_factors=None)
    with Dataset(str(fixed)) as database:
        assert 'load_factors' not in database.ncattrs()
        assert 'extended_flight_path_angles_deg' not in database.ncattrs()

    # The same settings under another output name: the same settings hash.
    renamed = tmp_path / 'renamed.nod'
    fa.build_empirical_database(str(spheres), str(renamed), load_factors=None)
    with Dataset(str(fixed)) as one, Dataset(str(renamed)) as two:
        assert one.panam_settings_sha256 == two.panam_settings_sha256
        assert json.loads(two.panam_inputs)['database_filename'] == 'renamed.nod'


def test_the_norah2_database_records_its_build(tmp_path):
    files = three_hems(tmp_path)
    path = tmp_path / 'norah2.nod'
    fa.build_database_from_norah2(files, str(path), speed_mapping='ground_speed', **VEHICLE)
    settings, inputs = _assert_stamp(path, 'flight_acoustics.build_database_from_norah2')
    assert inputs == dict(hem_files=['X_0.hem', 'X_1.hem', 'X_2.hem'], database_filename='norah2.nod')
    assert 'hem_files' not in settings
    assert settings['speed_mapping'] == 'ground_speed'
    _assert_no_machine_path(path, tmp_path)


def test_an_averaged_sphere_keeps_its_levels_out_of_the_settings(tmp_path):
    import sphere_statistics as ss
    hemisphere = sphere_helpers.minimal_hemisphere()
    spheres = sphere_helpers.write_sphere_directory(tmp_path / 'spheres',
                                                    [(hemisphere, 60.0, 0.0), (hemisphere, 60.0, 0.0)])
    files = sorted(str(f) for f in spheres.glob('*.nc'))
    out = tmp_path / 'average.nc'
    ss.write_averaged_sphere(str(out), ss.average_spheres(files))
    settings, inputs = _assert_stamp(out, 'flight_acoustics.write_aam_hemisphere_netcdf')
    assert inputs == dict(filename='average.nc')
    assert not {'art_levels', 'extra_variables', 'hemisphere'} & set(settings)


def test_the_sphere_selection_is_an_input_not_a_setting(tmp_path):
    hemisphere = sphere_helpers.minimal_hemisphere()
    spheres = sphere_helpers.write_sphere_directory(tmp_path / 'spheres',
                                                    [(hemisphere, 60.0, 0.0), (hemisphere, 80.0, -6.0)])
    names = sorted(f.name for f in spheres.glob('*.nc'))
    every = tmp_path / 'every.nod'
    fa.build_empirical_database(str(spheres), str(every), load_factors=None)
    chosen = tmp_path / 'chosen.nod'
    fa.build_empirical_database(str(spheres), str(chosen), load_factors=None,
                                spheres=[str(spheres / name) for name in names])
    _, inputs = _assert_stamp(chosen, 'flight_acoustics.build_empirical_database')
    assert inputs['spheres'] == names
    with Dataset(str(every)) as one, Dataset(str(chosen)) as two:
        assert one.panam_settings_sha256 == two.panam_settings_sha256
    _assert_no_machine_path(chosen, tmp_path)


def test_settings_sha256_is_the_stamps_hash():
    settings = dict(b=2, a=np.array([1.0, np.inf]))
    assert provenance.settings_sha256(settings) == \
        provenance.build_provenance('t', settings)['panam_settings_sha256']


def test_the_input_signature_follows_size_and_time_and_names_no_directory(tmp_path):
    one, two = tmp_path / 'one.nc', tmp_path / 'two.nc'
    one.write_bytes(b'1234')
    two.write_bytes(b'12')
    signature = provenance.file_signature([str(two), one])
    assert signature == provenance.file_signature([one, str(two), one])        # order and repeats
    copy = tmp_path / 'elsewhere'
    copy.mkdir()
    for path in (one, two):
        target = copy / path.name
        target.write_bytes(path.read_bytes())
        os.utime(target, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns))
    assert provenance.file_signature([copy / 'one.nc', copy / 'two.nc']) == signature
    status = one.stat()
    os.utime(one, ns=(status.st_atime_ns, status.st_mtime_ns + 10 ** 9))
    assert provenance.file_signature([one, two]) != signature
    assert provenance.file_signature([one, tmp_path / 'missing.nc']) != provenance.file_signature([one])
    stamp = provenance.build_provenance('t', {}, input_signature=signature)
    assert stamp[provenance.INPUT_SIGNATURE_ATTRIBUTE] == signature
    assert provenance.INPUT_SIGNATURE_ATTRIBUTE not in provenance.build_provenance('t', {})


def test_a_build_record_reads_back_equal():
    """record_json keeps every float digit, NaN, tuples, dicts and numpy scalars,
    reduces machine paths to file names, and a stamp writes it when given."""
    record = dict(run='289108', speed=60.123456789012345, angle=np.float64(-6.000000000000001),
                  mics=np.int64(12), window=(1.5, 30.25), height=float('nan'), level=float('-inf'),
                  flags='', none=None, flag=True, nested={'a': [1, (2, 3)]}, path='/Users/me/a.nc')
    back = provenance.read_record_json(provenance.record_json(record))
    assert back.pop('path') == 'a.nc' and record.pop('path')
    assert np.isnan(back.pop('height')) and np.isnan(record.pop('height'))
    assert back == record
    assert type(back['window']) is tuple and type(back['nested']['a'][1]) is tuple
    assert type(back['mics']) is int
    stamp = provenance.build_provenance('t', {}, build_record={'a': 1.0})
    assert provenance.read_record_json(stamp[provenance.BUILD_RECORD_ATTRIBUTE]) == {'a': 1.0}
    assert provenance.BUILD_RECORD_ATTRIBUTE not in provenance.build_provenance('t', {})
