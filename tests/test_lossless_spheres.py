"""The root flag sphere_levels_lossless: a sphere's levels carry no atmospheric absorption.

PSU-WOPWOP predicts its spheres in still, lossless air (air_density_source =
"wopwop"), so NICEOPS must absorb over the whole range from the center, where a
sphere PANAM depropagated through the run's atmosphere already holds what the
air took inside it.  The builder writes sphere_levels_lossless = 1 for the
former and leaves it out for the latter, so their databases are unchanged.
"""

import numpy as np
import pytest
from netCDF4 import Dataset

from panam import flight_acoustics as fa
from tools import build_empirical_database as script

import sphere_helpers

PHI = np.arange(-90.0, 90.0 + 1e-9, 45.0)
THETA = np.arange(0.0, 180.0 + 1e-9, 30.0)
BANDS = np.array([100.0, 125.0, 160.0])


def _write(path, speed, wopwop):
    levels = np.full((PHI.size, THETA.size, BANDS.size), 70.0) - 0.1 * np.arange(THETA.size)[None, :, None]
    sphere_helpers.write_raw_sphere(path, PHI, THETA, BANDS, levels, RADIUS=100.0, SPEED=speed,
                                    FLIGHT_PATH_ANGLE=0.0, DOPPLER_SHIFT_REMOVED=0.0)
    with Dataset(str(path), 'a') as handle:
        handle.sphere_frame = 'level'
        if wopwop:
            handle.air_density_source = 'wopwop'


def _folder(directory, wopwop=(True, True)):
    directory.mkdir()
    (directory / 'vehicle.cfg').write_text(sphere_helpers.VEHICLE_CFG)
    for i, flag in enumerate(wopwop):
        _write(directory / f'X{101 + i:03d}.nc', 40.0 + 20.0 * i, flag)
    return directory


def _flag(database):
    with Dataset(str(database)) as handle:
        if 'sphere_levels_lossless' not in handle.variables:
            return None
        return int(handle['sphere_levels_lossless'][...])


def _build(directory, database, **options):
    fa.build_empirical_database(str(directory), str(database), load_factors=None, **options)
    return database


def test_wopwop_spheres_are_flagged_lossless(tmp_path):
    assert _flag(_build(_folder(tmp_path / 's'), tmp_path / 'x.nod')) == 1


def test_measured_spheres_carry_no_flag(tmp_path):
    assert _flag(_build(_folder(tmp_path / 's', (False, False)), tmp_path / 'x.nod')) is None


def test_a_mix_is_refused(tmp_path):
    with pytest.raises(ValueError, match='lossless'):
        _build(_folder(tmp_path / 's', (True, False)), tmp_path / 'x.nod')


def test_lossless_spheres_of_another_source_can_be_declared(tmp_path):
    # AAM's own spheres, say: lossless, but not labeled so.
    database = _build(_folder(tmp_path / 's', (False, False)), tmp_path / 'x.nod', sphere_levels_lossless=True)
    assert _flag(database) == 1


def test_wopwop_spheres_declared_absorbed_are_refused(tmp_path):
    with pytest.raises(ValueError, match='lossless'):
        _build(_folder(tmp_path / 's'), tmp_path / 'x.nod', sphere_levels_lossless=False)


@pytest.mark.parametrize('flag, value', [('yes', True), ('no', False)])
def test_the_cli_takes_the_flag(tmp_path, flag, value):
    args = script.parse_arguments([str(tmp_path), str(tmp_path / 'db.nod'), '--sphere-levels-lossless', flag])
    assert script.build_options(args)['sphere_levels_lossless'] is value
