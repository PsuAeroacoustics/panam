"""A PANAM-built .nod database, loaded by NICE-OPS itself.

panam's own tests read back the names and shapes it writes; only the consumer
checks the rest of the contract (rotor_scale present, amplitude rows matching
the channels, coverage lengths, equal bands and channel counts across groups).
These run NICE-OPS's self-test on a small database, and export a sphere back
to check its orientation.  They skip when the executable is not configured.
"""
import subprocess

import numpy as np
import pytest
from netCDF4 import Dataset

import flight_acoustics as fa
from sphere_helpers import executable, minimal_hemisphere, write_sphere_directory

NICEOPS = executable('niceops')

pytestmark = [
    pytest.mark.skipif(NICEOPS is None, reason='niceops not configured (local_paths: niceops)'),
    pytest.mark.filterwarnings('ignore:.*carry no azimuth_reference:UserWarning'),
]


def _patterned_hemisphere(offset_db):
    """Levels that differ ahead, behind, to either side and with elevation,
    so a mirrored or transposed sphere does not match."""
    hemisphere = minimal_hemisphere()
    azimuth = np.radians(hemisphere['azi_grid_deg'])[None, None, :]
    elevation = hemisphere['elv_grid_deg'][None, :, None]
    hemisphere['third_octave']['bands_db'] = (75.0 + offset_db + 8.0 * np.cos(azimuth) + 3.0 * np.sin(azimuth)
                                             + 0.1 * elevation + np.arange(3.0)[:, None, None])
    return hemisphere


@pytest.fixture
def spheres(tmp_path):
    return write_sphere_directory(
        tmp_path / 'spheres', [(_patterned_hemisphere(0.0), 70.0, 0.0), (_patterned_hemisphere(3.0), 90.0, -6.0)],
        phi_deg=np.arange(-90.0, 90.0 + 1e-9, 10.0), theta_deg=np.arange(0.0, 180.0 + 1e-9, 10.0))


def _niceops(*args, cwd):
    return subprocess.run([NICEOPS, *args], cwd=cwd, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize('options', [dict(), dict(load_factors=None), dict(store_spectrum=False)])
def test_niceops_self_test_passes_on_a_panam_database(spheres, tmp_path, options):
    database = tmp_path / 'database.nod'
    fa.build_empirical_database(str(spheres), str(database), **options)
    result = _niceops('-d', str(database), '--self_test', '-l', 'self_test.log', cwd=tmp_path)
    log = (tmp_path / 'self_test.log').read_text()
    assert result.returncode == 0, log[-2000:] + result.stderr
    assert 'Self test passed' in log


def test_niceops_refuses_a_database_without_rotor_scale(spheres, tmp_path):
    """The self-test above is a real check: NICE-OPS stops on a broken database."""
    database = tmp_path / 'database.nod'
    fa.build_empirical_database(str(spheres), str(database))
    with Dataset(str(database), 'a') as ds:
        ds.groups['sphere0'].renameVariable('rotor_scale', 'not_rotor_scale')
    result = _niceops('-d', str(database), '--self_test', cwd=tmp_path)
    assert result.returncode != 0
    assert 'rotor_scale' in result.stdout + result.stderr


def test_niceops_exports_the_source_sphere_back_unchanged(spheres, tmp_path):
    """At a stored condition NICE-OPS's AAM export is the source sphere: the
    same levels on the same PHI/THETA, so neither side is mirrored."""
    database = tmp_path / 'database.nod'
    fa.build_empirical_database(str(spheres), str(database), load_factors=None)
    with Dataset(str(database)) as ds:
        group = ds.groups['sphere0']
        condition = [float(group[name][:].ravel()[0])
                     for name in ('advance_ratio', 'flight_path_angle', 'thrust_coefficient')]
    assert condition[1] == 0.0                                       # sphere0 is X101.nc
    result = _niceops('-d', str(database), '--export_aam', 'export.nc',
                      '--sphere_condition', *('%.17g' % value for value in condition), cwd=tmp_path)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr
    exported, phi, theta, frequency, radius, speed, _ = fa.load_nc_sphere(str(tmp_path / 'export.nc'))
    source, source_phi, source_theta, source_frequency, *_ = fa.load_nc_sphere(str(spheres / 'X101.nc'))
    rows = np.searchsorted(phi, source_phi)                          # the export covers phi -180..170
    np.testing.assert_array_equal(phi[rows], source_phi)
    np.testing.assert_array_equal(theta, source_theta)
    np.testing.assert_array_equal(frequency, source_frequency)
    assert float(np.ravel(radius)[0]) == 100.0
    assert float(np.ravel(speed)[0]) == pytest.approx(70.0, abs=1e-3)
    np.testing.assert_allclose(exported[rows], source, atol=1e-4)
