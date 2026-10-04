"""Thrust scaling of sphere levels, and the CT = 0 floor entry.

A load factor of n raises the sphere level by 20*log10(n) and scales the thrust
coefficient to match.  A load factor of 0 is NOT a physical condition: it is a
floor entry that extends the thrust-coefficient range down to CT = 0 so that
sub-1g trajectories interpolate instead of clamping at the edge.  It carries the
1 g levels unscaled.

Evaluating 20*log10(0) there would write -inf into dBA, and a zero barycentric
weight against -inf yields NaN -- which poisons every interpolation that touches
such a sphere.  That is not hypothetical: it made NICE-OPS's self test fail on
all 1,726,272 samples of a database built with the maneuver load factors.
"""

import numpy as np
import pytest

from netCDF4 import Dataset

from flight_acoustics import add_sphere_group


N_PHI, N_THETA = 4, 3
BASE_LEVEL = 90.0


@pytest.fixture
def written_spheres(tmp_path):
    """Write one sphere per load factor and read the stored dBA back."""
    phi = np.zeros((N_PHI, N_THETA))
    theta = np.zeros((N_PHI, N_THETA))
    spla = np.full((N_PHI, N_THETA), BASE_LEVEL)
    eaa = np.zeros((N_PHI, N_THETA))
    load_factors = [0.0, 1.0, 1.1, 2.0, 5.0]

    path = str(tmp_path / 'spheres.nc')
    out = {}
    with Dataset(path, 'w') as db:
        for i, lf in enumerate(load_factors):
            add_sphere_group(db, f'sphere{i}', phi, theta, radius=30.48, SPLA=spla, EAA=eaa,
                             speed=60.0, flight_path_angle=0.0, load_factor=lf,
                             main_rotor_radius=5.334, main_rotor_tip_speed=230.7,
                             weight_coefficient=0.004341)
    with Dataset(path, 'r') as db:
        for i, lf in enumerate(load_factors):
            g = db[f'sphere{i}']
            out[lf] = (np.array(g['dBA'][:]), float(np.array(g['thrust_coefficient'][:])[0]))
    return out


def test_unit_load_factor_is_unscaled(written_spheres):
    dba, _ = written_spheres[1.0]
    assert np.allclose(dba, BASE_LEVEL)


@pytest.mark.parametrize('load_factor', [1.1, 2.0, 5.0])
def test_positive_load_factors_scale_by_twenty_log10(written_spheres, load_factor):
    dba, ct = written_spheres[load_factor]
    assert np.allclose(dba, BASE_LEVEL + 20.0 * np.log10(load_factor))
    assert np.isclose(ct, load_factor * 0.004341)


def test_zero_load_factor_is_finite_and_unscaled(written_spheres):
    """The regression: 20*log10(0) would make every level -inf."""
    dba, ct = written_spheres[0.0]
    assert np.isfinite(dba).all(), 'CT = 0 sphere must not contain -inf'
    assert np.allclose(dba, BASE_LEVEL)
    assert ct == 0.0


def test_zero_load_factor_matches_one_g(written_spheres):
    """Matches the shipped databases, where the CT = 0 sphere is level-identical
    to the 1 g sphere at the same condition."""
    zero, _ = written_spheres[0.0]
    one_g, _ = written_spheres[1.0]
    assert np.array_equal(zero, one_g)
