"""Writers write a file in full or not at all (flight_acoustics._atomic_output)."""

import glob
import os

import numpy as np
import pytest

import flight_acoustics as fa
import provenance
from sphere_helpers import VEHICLE_CFG, minimal_hemisphere


class _Interrupted(Exception):
    pass


def _interrupt(*args, **kwargs):
    raise _Interrupted


def _leftovers(directory):
    """Every file in ``directory``, dotfiles included."""
    return sorted(os.listdir(directory))


def _write_sphere(path, speed_knots=70.0, flight_path_angle_deg=0.0):
    fa.write_aam_hemisphere_netcdf(
        str(path), minimal_hemisphere(), mode='third_octave',
        phi_deg=np.arange(-90.0, 90.0 + 1e-9, 30.0), theta_deg=np.arange(0.0, 180.0 + 1e-9, 30.0),
        radius_ft=100.0, speed_knots=speed_knots, flight_path_angle_deg=flight_path_angle_deg,
        title='t')


def test_the_temporary_name_is_not_matched_by_output_globs(tmp_path):
    target = tmp_path / 'A101.nc'
    with fa._atomic_output(target) as temporary:
        assert os.path.dirname(temporary) == str(tmp_path)
        assert os.path.basename(temporary) == '.A101.nc.{}.tmp'.format(os.getpid())
        open(temporary, 'w').close()
        assert glob.glob(str(tmp_path / '*.nc')) == [] and glob.glob(str(tmp_path / '*.nod')) == []
        assert not target.exists()
    assert target.exists() and _leftovers(tmp_path) == ['A101.nc']


def test_an_exception_leaves_an_existing_file_alone(tmp_path):
    target = tmp_path / 'A101.nc'
    target.write_text('old')
    with pytest.raises(_Interrupted):
        with fa._atomic_output(target) as temporary:
            with open(temporary, 'w') as handle:
                handle.write('new')
            raise _Interrupted
    assert target.read_text() == 'old' and _leftovers(tmp_path) == ['A101.nc']


def test_a_symlinked_output_is_written_through(tmp_path):
    """A sphere or database kept as a link into another folder updates the target,
    keeps the link, and keeps the target's permission bits."""
    shared = tmp_path / 'shared'
    shared.mkdir()
    target = shared / 'A101.nc'
    target.write_text('old')
    os.chmod(target, 0o640)
    link = tmp_path / 'A101.nc'
    link.symlink_to(target)
    _write_sphere(link)
    assert link.is_symlink() and os.readlink(link) == str(target)
    assert target.stat().st_size > 3 and target.read_bytes()[:3] != b'old'
    assert (target.stat().st_mode & 0o777) == 0o640
    assert _leftovers(tmp_path) == ['A101.nc', 'shared'] and _leftovers(shared) == ['A101.nc']


def test_a_sphere_interrupted_mid_write_leaves_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, 'write_provenance', _interrupt)
    with pytest.raises(_Interrupted):
        _write_sphere(tmp_path / 'A101.nc')
    assert _leftovers(tmp_path) == []


def test_a_database_interrupted_mid_write_leaves_no_file(tmp_path, monkeypatch):
    spheres = tmp_path / 'spheres'
    spheres.mkdir()
    (spheres / 'vehicle.cfg').write_text(VEHICLE_CFG)
    for name, speed, angle in (('A100.nc', 70.0, 0.0), ('A101.nc', 90.0, -6.0)):
        _write_sphere(spheres / name, speed, angle)
    output = tmp_path / 'out'
    output.mkdir()
    monkeypatch.setattr(fa, 'add_sphere_group', _interrupt)
    with pytest.raises(_Interrupted):
        fa.build_empirical_database(str(spheres), str(output / 'db.nod'),
                                    load_factors=np.array([1.0]), store_spectrum=False)
    assert _leftovers(output) == []
