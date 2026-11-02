"""PSU-WOPWOP results folders built into databases, end to end.

The first test needs no data: a results folder written as WOPWOP writes one
(sphere_helpers.write_wopwop_sphere) -- every type at each condition, full
spheres, a real hover, floor cells, the vehicle in the spheres and no
vehicle.cfg.  The data tests build WOPWOP's own regression spheres (ROTOR000,
ITRRX000), found through local_paths ``wopwop_noise_spheres``, the directory
WOPWOP's test/regression/NoiseSphere writes them to; neither is committed here.
Where NICEOPS is configured, it loads each database and runs its self test.
"""
import shutil
import subprocess
import warnings

import numpy as np
import pytest
from netCDF4 import Dataset

from panam import flight_acoustics as fa
from panam import local_paths
from panam.nod_database import validate_nod

import sphere_helpers

NICEOPS = sphere_helpers.executable('niceops')
FLOOR = -40.0
PHI = np.arange(-180.0, 180.0 + 1e-9, 45.0)
THETA = np.arange(0.0, 180.0 + 1e-9, 30.0)
BANDS = {'BB': np.array([100.0, 125.0, 160.0, 200.0]), 'NB': np.array([10.0, 20.0, 30.0, 40.0]),
         'PT': np.array([20.0, 40.0, 60.0, 80.0])}


def _levels(kind, speed):
    theta, phi = np.meshgrid(THETA, PHI)
    level = 70.0 + 0.05 * speed + 4.0 * np.cos(np.radians(theta)) + 2.0 * np.sin(np.radians(theta)) * np.cos(
        np.radians(phi))
    levels = np.repeat(level[:, :, None], BANDS[kind].size, axis=2) - 3.0 * np.arange(BANDS[kind].size)
    levels[:, :2, -1] = FLOOR                      # the top band empty ahead
    levels[-1] = levels[0]                         # the seam: phi = 180 is phi = -180
    return levels


def _results_folder(directory):
    directory.mkdir()
    index = 0
    for speed in (0.0, 40.0, 80.0):
        for kind in ('BB', 'NB', 'PT'):
            phase = np.zeros_like(_levels(kind, speed)) if kind == 'PT' else None
            sphere_helpers.write_wopwop_sphere(directory / f'WOPWP{index:03d}.nc', _levels(kind, speed), PHI, THETA,
                                               BANDS[kind], kind=kind, speed=speed, floor=FLOOR, phase=phase)
            index += 1
    return directory


def _self_test(database, cwd):
    if NICEOPS is None:
        return
    result = subprocess.run([NICEOPS, '-d', str(database), '--self_test', '-l', 'self_test.log'], cwd=cwd,
                            capture_output=True, text=True, timeout=300)
    log = (cwd / 'self_test.log').read_text()
    assert result.returncode == 0, log[-2000:] + result.stderr
    assert 'Self test passed' in log


def test_a_wopwop_results_folder_builds_a_valid_database(tmp_path):
    directory = _results_folder(tmp_path / 'results')
    database = tmp_path / 'wopwop.nod'
    fa.build_empirical_database(str(directory), str(database), load_factors=None)
    assert validate_nod(str(database)) == []
    with Dataset(str(database)) as handle:
        assert handle.source_spheres.split() == ['WOPWP000.nc', 'WOPWP003.nc', 'WOPWP006.nc']
        assert handle.band_type == 'third_octave' and handle.speed_reference == 'air'
        assert int(handle['sphere_levels_lossless'][...]) == 1
        assert float(handle['main_rotor_radius_meters'][...]) == pytest.approx(5.334, rel=1e-6)
        assert float(handle['prediction_radius_meters'][...]) == 300.0
        groups = list(handle.groups.values())
        assert {g.condition_origin for g in groups} == {'measured', 'extended_flight_path_angle'}
        for group in groups:
            amplitude = np.array(group['amplitude'][:])
            assert amplitude.shape[0] == PHI.size - 1                    # one seam row
            assert not np.any(np.isclose(amplitude, FLOOR))              # the floor read as missing
            assert np.all(np.array(group['coverage'][:]) == np.isfinite(amplitude))
    _self_test(database, tmp_path)


def _data_spheres(names):
    root = local_paths.data_path('wopwop_noise_spheres', required=False)
    if root is None:
        pytest.skip('WOPWOP noise spheres not configured (local_paths: wopwop_noise_spheres)')
    return [f'{root}/{name}' for name in names]


@pytest.mark.data
def test_wopwops_rotor_spheres_build_without_a_vehicle_file(tmp_path):
    directory = tmp_path / 'rotor'
    directory.mkdir()
    for path in _data_spheres(['ROTOR000.nc', 'ROTOR001.nc', 'ROTOR002.nc']):
        shutil.copy(path, directory)
    database = tmp_path / 'rotor.nod'
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='No sphere within')
        fa.build_empirical_database(str(directory), str(database), load_factors=None)
    assert validate_nod(str(database)) == []
    with Dataset(str(database)) as handle:
        assert handle.source_spheres.split() == ['ROTOR000.nc']
        assert int(handle['sphere_levels_lossless'][...]) == 1
        assert handle.speed_reference == 'air'
    _self_test(database, tmp_path)


@pytest.mark.data
def test_wopwops_far_field_spheres_build_with_a_vehicle_file(tmp_path):
    directory = tmp_path / 'far_field'
    directory.mkdir()
    (directory / 'vehicle.cfg').write_text(sphere_helpers.VEHICLE_CFG)
    for path in _data_spheres(['ITRRX000.nc', 'ITRRX001.nc']):
        shutil.copy(path, directory)
    database = tmp_path / 'far_field.nod'
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='No sphere within')
        fa.build_empirical_database(str(directory), str(database), load_factors=None)
    assert validate_nod(str(database)) == []
    with Dataset(str(database)) as handle:
        assert handle.source_spheres.split() == ['ITRRX000.nc']
    _self_test(database, tmp_path)
