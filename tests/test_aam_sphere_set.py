"""tools/aam_sphere_set.py: checking a set of AAM spheres before AAM reads it.

A vehicle's spheres (vvvvvnnn.nc, as PSU-WOPWOP writes them) must agree where
AAM interpolates between them, and a few things AAM refuses or mishandles are
easier to catch here than in a failed run.  Each test breaks one rule on an
otherwise clean synthetic set.
"""
import numpy as np
import pytest
from netCDF4 import Dataset

from tools import aam_sphere_set as tool

import sphere_helpers

PHI = np.arange(-180.0, 180.0 + 1e-9, 45.0)
THETA = np.arange(0.0, 180.0 + 1e-9, 30.0)
BANDS = {'BB': np.array([100.0, 125.0, 160.0]), 'OB12': np.array([100.0, 105.9, 112.2]),
         'NB': np.array([10.0, 20.0, 30.0]), 'PT': np.array([20.0, 40.0, 60.0])}


def _write(path, kind='BB', speed=60.0, angle=0.0, phi=PHI, frequency=None, **options):
    frequency = BANDS[kind] if frequency is None else frequency
    levels = np.full((phi.size, THETA.size, frequency.size), 70.0)
    phase = np.zeros_like(levels) if kind == 'PT' else None
    sphere_helpers.write_wopwop_sphere(path, levels, phi, THETA, frequency, kind=kind, speed=speed,
                                       flight_path_angle=angle, phase=phase, **options)
    if kind == 'PT':
        with Dataset(str(path), 'a') as handle:
            handle.phase_reference = 'cosine, emission from the center at phase_reference_time_s'
    return path


def _clean_set(directory, vehicle='VEHIC'):
    """BB at three conditions, an NB and a PT at one."""
    directory.mkdir(exist_ok=True)
    index = 0
    for speed, angle in ((40.0, 0.0), (60.0, 0.0), (60.0, -6.0)):
        _write(directory / f'{vehicle}{index:03d}.nc', speed=speed, angle=angle)
        index += 1
    _write(directory / f'{vehicle}{index:03d}.nc', 'NB')
    _write(directory / f'{vehicle}{index + 1:03d}.nc', 'PT')
    return directory


def _rules(findings, severity=None):
    return sorted({f.rule for f in findings if severity is None or f.severity == severity})


def test_a_clean_set_passes_and_gives_its_deck_values(tmp_path):
    spheres = tool.read_set([_clean_set(tmp_path / 'set')])
    assert len(spheres) == 5
    assert _rules(tool.check_set(spheres), 'error') == []
    table = tool.sorted_table(spheres)
    assert all(s.path.name in table for s in spheres)
    (vehicle,) = tool.deck_values(spheres)
    assert vehicle['vehicle'] == 'VEHIC'
    assert vehicle['centers_ft'] == {'BB': [(0, 0, 0)], 'NB': [(0, 0, 0)], 'PT': [(0, 0, 0)]}
    assert vehicle['audib_bandwidth_hz'] == pytest.approx(10.0)


def test_spheres_of_a_type_on_different_grids_are_an_error(tmp_path):
    directory = _clean_set(tmp_path / 'set')
    _write(directory / 'VEHIC010.nc', speed=80.0, frequency=np.array([100.0, 125.0, 160.0, 200.0]))
    assert 'grid' in _rules(tool.check_set(tool.read_set([directory])), 'error')


def test_a_center_off_whole_feet_is_an_error(tmp_path):
    directory = _clean_set(tmp_path / 'set')
    _write(directory / 'VEHIC010.nc', speed=80.0, xyz=(-12.4, 0.0, 7.0))
    assert 'center' in _rules(tool.check_set(tool.read_set([directory])), 'error')


def test_a_duplicate_condition_is_an_error_whatever_the_weight(tmp_path):
    directory = _clean_set(tmp_path / 'set')
    _write(directory / 'VEHIC010.nc', speed=60.0, angle=0.0, gross_weight_lb=6000.0)
    assert 'duplicate' in _rules(tool.check_set(tool.read_set([directory])), 'error')


def test_ob12_and_nb_at_one_center_are_an_error(tmp_path):
    directory = _clean_set(tmp_path / 'set')
    _write(directory / 'VEHIC010.nc', 'OB12')
    assert 'narrowband' in _rules(tool.check_set(tool.read_set([directory])), 'error')


def test_pt_spheres_of_two_phase_conventions_are_an_error(tmp_path):
    directory = _clean_set(tmp_path / 'set')
    path = _write(directory / 'VEHIC010.nc', 'PT', speed=40.0)
    with Dataset(str(path), 'a') as handle:
        handle.phase_reference = 'sine'
    assert 'phase' in _rules(tool.check_set(tool.read_set([directory])), 'error')


def test_pt_spheres_with_different_tone_counts_are_an_error(tmp_path):
    directory = _clean_set(tmp_path / 'set')
    _write(directory / 'VEHIC010.nc', 'PT', speed=40.0, frequency=np.array([20.0, 40.0, 60.0, 80.0]))
    assert 'grid' in _rules(tool.check_set(tool.read_set([directory])), 'error')


def test_two_nb_bandwidths_are_an_error(tmp_path):
    directory = _clean_set(tmp_path / 'set')
    _write(directory / 'VEHIC010.nc', 'NB', speed=40.0, frequency=np.array([5.0, 10.0, 15.0]))
    assert 'audib' in _rules(tool.check_set(tool.read_set([directory])), 'error')


def test_a_second_sound_speed_is_a_warning(tmp_path):
    directory = _clean_set(tmp_path / 'set')
    path = _write(directory / 'VEHIC010.nc', speed=80.0)
    with Dataset(str(path), 'a') as handle:
        handle.speed_of_sound_m_s = '3.30000000E+02'
    assert 'sound_speed' in _rules(tool.check_set(tool.read_set([directory])), 'warning')


def test_wide_spacing_is_a_warning(tmp_path):
    directory = _clean_set(tmp_path / 'set')
    _write(directory / 'VEHIC010.nc', speed=120.0)
    assert 'spacing' in _rules(tool.check_set(tool.read_set([directory])), 'warning')


def test_a_lower_hemisphere_off_level_flight_is_a_warning(tmp_path):
    directory = _clean_set(tmp_path / 'set')
    _write(directory / 'VEHIC010.nc', speed=40.0, angle=-6.0, phi=np.arange(-90.0, 90.0 + 1e-9, 45.0))
    findings = tool.check_set(tool.read_set([directory]))
    assert 'hemisphere' in _rules(findings, 'warning')


def test_more_than_ten_centers_of_a_type_are_an_error(tmp_path):
    directory = _clean_set(tmp_path / 'set')
    for k in range(10):
        _write(directory / f'VEHIC{20 + k:03d}.nc', speed=40.0, xyz=(float(k + 1), 0.0, 0.0))
    assert 'centers' in _rules(tool.check_set(tool.read_set([directory])), 'error')


def test_index_collisions_across_folders_are_an_error_and_merge_renumbers(tmp_path):
    first = _clean_set(tmp_path / 'first')
    second = tmp_path / 'second'
    second.mkdir()
    _write(second / 'VEHIC000.nc', speed=100.0)
    spheres = tool.read_set([first, second])
    assert 'index' in _rules(tool.check_set(spheres), 'error')
    merged = tool.merge(spheres, tmp_path / 'merged')
    names = sorted(p.name for p in (tmp_path / 'merged').glob('*.nc'))
    assert len(names) == 6 and len(set(names)) == 6 and 'VEHIC005.nc' in names
    assert 'index' not in _rules(tool.check_set(tool.read_set([tmp_path / 'merged'])), 'error')
    assert {m[1].name for m in merged} == set(names)


def test_the_command_line_fails_on_an_error(tmp_path, capsys):
    directory = _clean_set(tmp_path / 'set')
    assert tool.main([str(directory)]) == 0
    _write(directory / 'VEHIC010.nc', speed=80.0, xyz=(-12.4, 0.0, 7.0))
    assert tool.main([str(directory)]) == 1
    assert 'center' in capsys.readouterr().out
