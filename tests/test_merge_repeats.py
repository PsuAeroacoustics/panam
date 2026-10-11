"""build_empirical_database's merge_repeats: repeat runs of one condition, energy-averaged.

Without it, two spheres at one speed and flight path angle are refused, as NICE-OPS
refuses them.  With it, they become one condition whose levels are the energy
average of the two, and the database names both files.
"""

import json
import warnings

import numpy as np
import pytest
from netCDF4 import Dataset

import flight_acoustics as fa
import build_empirical_database as script
from sphere_helpers import minimal_hemisphere, write_sphere_directory


def _repeat_directory(tmp_path, repeat_speed=80.0):
    """X101 (60 kt, -3 deg), X102 and X103 (80 kt and ``repeat_speed``, 0 deg), X104 (100 kt, 0 deg).

    X103 differs from X102 by a level offset that varies from cell to cell and band to band.
    """
    base = minimal_hemisphere()
    repeat = minimal_hemisphere()
    bands = repeat['third_octave']['bands_db']
    repeat['third_octave']['bands_db'] = bands + np.linspace(0.0, 9.0, bands.size).reshape(bands.shape)
    fast = minimal_hemisphere()
    fast['third_octave']['bands_db'] += 3.0
    return write_sphere_directory(tmp_path / 'spheres', [
        (minimal_hemisphere(), 60, -3), (base, 80, 0), (repeat, repeat_speed, 0), (fast, 100, 0)])


def _build(directory, database, **options):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fa.build_empirical_database(str(directory), str(database), load_factors=None, **options)
    return database


def _measured_group(database, source_sphere):
    """(amplitude dB, dBA, advance ratio) of the measured group whose source_sphere is given."""
    with Dataset(str(database)) as db:
        for group in db.groups.values():
            if group.condition_origin == 'measured' and group.source_sphere == source_sphere:
                return (np.asarray(group['amplitude'][:]), np.asarray(group['dBA'][:]),
                        float(group['advance_ratio'][0]))
    raise AssertionError('no measured group from {!r}'.format(source_sphere))


def test_repeats_are_refused_without_merge(tmp_path):
    directory = _repeat_directory(tmp_path)
    database = tmp_path / 'refused.nod'
    with pytest.raises(ValueError, match='X102.nc and X103.nc have the same flight condition'):
        _build(directory, database)
    assert not database.exists()


def test_repeats_merge_to_their_energy_average(tmp_path):
    directory = _repeat_directory(tmp_path)
    merged = _build(directory, tmp_path / 'merged.nod', merge_repeats=True)
    only_a = _build(directory, tmp_path / 'a.nod', exclude=['X103.nc'])
    only_b = _build(directory, tmp_path / 'b.nod', exclude=['X102.nc'])

    amplitude, dba, mu = _measured_group(merged, 'X102.nc\nX103.nc')
    amplitude_a, dba_a, mu_a = _measured_group(only_a, 'X102.nc')
    amplitude_b, dba_b, _ = _measured_group(only_b, 'X103.nc')
    expected = 10.0 * np.log10(0.5 * (10.0 ** (amplitude_a / 10.0) + 10.0 ** (amplitude_b / 10.0)))
    assert np.ptp(amplitude_b - amplitude_a) > 1.0     # the runs differ cell by cell
    np.testing.assert_allclose(amplitude, expected, atol=1e-4)
    # The broadband level comes from the averaged bands, so it lies between the runs'.
    assert np.all(dba >= np.minimum(dba_a, dba_b) - 1e-9)
    assert np.all(dba <= np.maximum(dba_a, dba_b) + 1e-9)
    assert mu == pytest.approx(mu_a)
    # One measured condition per speed: four spheres, three conditions.
    with Dataset(str(merged)) as db:
        measured = [g for g in db.groups.values() if g.condition_origin == 'measured']
        assert len(measured) == 3


def test_merged_inputs_are_recorded_by_file_name(tmp_path):
    directory = _repeat_directory(tmp_path)
    database = _build(directory, tmp_path / 'merged.nod', merge_repeats=True)
    with Dataset(str(database)) as db:
        inputs = json.loads(db.panam_inputs)
        settings = json.loads(db.panam_settings)
        hover_sources = {g.source_sphere for g in db.groups.values()
                         if g.condition_origin == 'synthesized_hover'}
        assert db.source_spheres.split('\n') == ['X101.nc', 'X102.nc', 'X103.nc', 'X104.nc']
    assert inputs['merged_spheres'] == [['X102.nc', 'X103.nc']]
    assert settings['merge_repeats'] is True
    # The hover is synthesized from the merged slowest level condition.
    assert hover_sources == {'X102.nc\nX103.nc'}
    # Nothing is merged without repeats, and nothing is recorded.
    plain = _build(directory, tmp_path / 'plain.nod', exclude=['X103.nc'])
    with Dataset(str(plain)) as db:
        assert 'merged_spheres' not in json.loads(db.panam_inputs)


def test_speed_tolerance_merges_nearby_speeds(tmp_path):
    directory = _repeat_directory(tmp_path, repeat_speed=81.0)
    # 1 kt apart: two conditions, unless the tolerance spans them.
    _build(directory, tmp_path / 'separate.nod', merge_repeats=True)
    with Dataset(str(tmp_path / 'separate.nod')) as db:
        assert 'merged_spheres' not in json.loads(db.panam_inputs)
    database = _build(directory, tmp_path / 'merged.nod', merge_repeats=True, merge_speed_tolerance=1.5)
    with Dataset(str(database)) as db:
        assert json.loads(db.panam_inputs)['merged_spheres'] == [['X102.nc', 'X103.nc']]
    # The merged condition sits at the runs' mean speed, 80.5 kt.
    _, _, mu = _measured_group(database, 'X102.nc\nX103.nc')
    _, _, mu_100 = _measured_group(database, 'X104.nc')
    assert mu == pytest.approx(mu_100 * 80.5 / 100.0)
    with pytest.raises(ValueError, match='at least 0'):
        _build(directory, tmp_path / 'bad.nod', merge_repeats=True, merge_speed_tolerance=-1.0)


def test_cli_merge_repeats(tmp_path):
    directory = _repeat_directory(tmp_path)
    args = script.parse_arguments([str(directory), str(tmp_path / 'db.nod'), '--merge-repeats',
                                   '--merge-speed-tolerance', '2'])
    options = script.build_options(args)
    assert options['merge_repeats'] is True and options['merge_speed_tolerance'] == 2.0
    with pytest.raises(SystemExit):
        script.parse_arguments([str(directory), str(tmp_path / 'db.nod'), '--merge-speed-tolerance', '2'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert script.main([str(directory), str(tmp_path / 'db.nod'), '--merge-repeats',
                            '--fixed-load-factor']) == 0
    with Dataset(str(tmp_path / 'db.nod')) as db:
        assert json.loads(db.panam_inputs)['merged_spheres'] == [['X102.nc', 'X103.nc']]
