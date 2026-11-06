"""thrust_from='sphere': a weight or density sweep's thrust coefficients from its spheres.

By default every condition is filed at the vehicle's nominal weight
coefficient (vehicle.cfg), so two spheres at one speed and flight path angle
are one condition, refused or merged.  With thrust_from='sphere' each sphere's
condition is its run's own C_T, GROSS_WEIGHT / (AIR_DENSITY pi R^2 V_tip^2), and
spheres at one speed and angle but different weights are separate conditions.
"""
import math

import numpy as np
import pytest
from netCDF4 import Dataset

from panam import flight_acoustics as fa
from panam.nod_database import validate_nod
from panam.sphere_io import POUND_FORCE_NEWTONS
from tools import build_empirical_database as script

import sphere_helpers

PHI = np.arange(-180.0, 180.0 + 1e-9, 45.0)
THETA = np.arange(0.0, 180.0 + 1e-9, 30.0)
BANDS = np.array([100.0, 125.0, 160.0])
# example_data/vehicle.cfg's main rotor
RADIUS, TIP_SPEED = 5.334, 230.7


def _run_ct(weight_lb, density=1.07):
    return weight_lb * POUND_FORCE_NEWTONS / (density * math.pi * RADIUS ** 2 * TIP_SPEED ** 2)


def _folder(directory, runs):
    """runs: (speed kt, weight lb or None) per sphere, all level flight."""
    directory.mkdir()
    (directory / 'vehicle.cfg').write_text(sphere_helpers.VEHICLE_CFG)
    for i, (speed, weight) in enumerate(runs):
        levels = np.full((PHI.size, THETA.size, BANDS.size), 70.0 + 0.05 * speed + (weight or 0) / 1000.0)
        sphere_helpers.write_wopwop_sphere(directory / f'SWEEP{i:03d}.nc', levels, PHI, THETA, BANDS, speed=speed,
                                           gross_weight_lb=weight, floor=-60.0)
    return directory


SWEEP = ((40.0, 4000.0), (40.0, 5500.0), (80.0, 4000.0), (80.0, 5500.0))


def _measured(database):
    with Dataset(str(database)) as handle:
        fixed = bool(handle['fixed_load_factor'][...])
        groups = [(float(g['advance_ratio'][0]), float(g['thrust_coefficient'][0]))
                  for g in handle.groups.values() if g.condition_origin == 'measured']
    return fixed, sorted(groups)


def test_by_default_a_weight_sweep_is_one_condition_twice(tmp_path):
    with pytest.raises(ValueError, match='same flight condition'):
        fa.build_empirical_database(str(_folder(tmp_path / 's', SWEEP)), str(tmp_path / 'x.nod'),
                                    load_factors=None)


def test_a_weight_sweep_files_each_run_at_its_thrust_coefficient(tmp_path):
    database = tmp_path / 'sweep.nod'
    fa.build_empirical_database(str(_folder(tmp_path / 's', SWEEP)), str(database), load_factors=None,
                                thrust_from='sphere')
    assert validate_nod(str(database)) == []
    fixed, groups = _measured(database)
    assert not fixed
    mu = [speed * 0.514444 / TIP_SPEED for speed, _ in SWEEP]
    expected = sorted(zip(mu, (_run_ct(weight) for _, weight in SWEEP)))
    np.testing.assert_allclose(groups, expected, rtol=1e-6)


def test_one_weight_from_the_spheres_is_still_a_fixed_load_factor(tmp_path):
    database = tmp_path / 'one.nod'
    fa.build_empirical_database(str(_folder(tmp_path / 's', ((40.0, 4000.0), (80.0, 4000.0)))), str(database),
                                load_factors=None, thrust_from='sphere')
    fixed, groups = _measured(database)
    assert fixed
    assert all(math.isclose(ct, _run_ct(4000.0), rel_tol=1e-6) for _, ct in groups)


def test_repeats_of_one_weight_merge_and_of_two_weights_do_not(tmp_path):
    runs = ((40.0, 4000.0), (40.0, 4000.0), (40.0, 5500.0), (80.0, 4000.0))
    database = tmp_path / 'merged.nod'
    fa.build_empirical_database(str(_folder(tmp_path / 's', runs)), str(database), load_factors=None,
                                thrust_from='sphere', merge_repeats=True)
    _, groups = _measured(database)
    assert len(groups) == 3


def test_a_sphere_without_its_weight_is_refused(tmp_path):
    with pytest.raises(ValueError, match='GROSS_WEIGHT'):
        fa.build_empirical_database(str(_folder(tmp_path / 's', ((40.0, 4000.0), (80.0, None)))),
                                    str(tmp_path / 'x.nod'), load_factors=None, thrust_from='sphere')


def test_the_cli_takes_thrust_from(tmp_path):
    args = script.parse_arguments([str(tmp_path), str(tmp_path / 'db.nod'), '--thrust-from', 'sphere'])
    assert script.build_options(args)['thrust_from'] == 'sphere'


def test_thrusts_that_agree_are_grouped_by_speed_from_the_slowest(tmp_path):
    """C_T that agree to 1e-6 are one thrust, and a group is then speeds from its slowest:
    a faster sphere at a C_T a hair below a slower one's is not merged into it."""
    from panam.database import _group_repeat_conditions
    folder = _folder(tmp_path / 's', ((100.0, 4000.0), (50.0, 4000.0)))
    fast, slow = str(folder / 'SWEEP000.nc'), str(folder / 'SWEEP001.nc')
    thrusts = {fast: 0.01, slow: 0.01 * (1.0 + 1e-9)}
    assert _group_repeat_conditions([fast, slow], 0.0, thrusts) == [(fast,), (slow,)]
    assert len(_group_repeat_conditions([fast, slow], 60.0, thrusts)) == 1


def test_a_hover_source_without_its_weight_is_refused(tmp_path):
    """The synthesized hovers take the hover source's own C_T under thrust_from='sphere', so a
    hover source that records no GROSS_WEIGHT is refused rather than written at NaN."""
    folder = _folder(tmp_path / 's', ((40.0, 4000.0), (80.0, 4000.0)))
    other = _folder(tmp_path / 'h', ((20.0, None),))
    with pytest.raises(ValueError, match='GROSS_WEIGHT'):
        fa.build_empirical_database(str(folder), str(tmp_path / 'x.nod'), load_factors=None,
                                    thrust_from='sphere', hover_source=str(other / 'SWEEP000.nc'))


def test_a_weight_sweep_holds_a_hover_at_each_weight(tmp_path):
    """A sweep with a hover at each weight files each at its own C_T, at every hover angle, and
    synthesizes none; two hovers at one weight are still two hovers of one condition."""
    runs = ((0.0, 4000.0), (0.0, 5500.0), (40.0, 4000.0), (40.0, 5500.0))
    database = tmp_path / 'hovers.nod'
    fa.build_empirical_database(str(_folder(tmp_path / 's', runs)), str(database), load_factors=None,
                                thrust_from='sphere')
    assert validate_nod(str(database)) == []
    with Dataset(str(database)) as handle:
        hovers = [(float(g['flight_path_angle'][0]), float(g['thrust_coefficient'][0]), g.condition_origin)
                  for g in handle.groups.values() if float(g['advance_ratio'][0]) == 0.0]
    assert 'synthesized_hover' not in {origin for _, _, origin in hovers}
    for weight in (4000.0, 5500.0):
        angles = sorted(gamma for gamma, ct, _ in hovers if math.isclose(ct, _run_ct(weight), rel_tol=1e-6))
        assert angles == [-12.0, 0.0, 12.0]
    # Two hovers at one weight filed at different flight path angles are not repeats of one
    # condition, but are still two hovers at one thrust.
    folder = _folder(tmp_path / 't', ((0.0, 4000.0), (40.0, 4000.0)))
    levels = np.full((PHI.size, THETA.size, BANDS.size), 70.0)
    sphere_helpers.write_wopwop_sphere(folder / 'SWEEP009.nc', levels, PHI, THETA, BANDS, speed=0.0,
                                       flight_path_angle=5.0, gross_weight_lb=4000.0, floor=-60.0)
    with pytest.raises(ValueError, match='one hover per thrust'):
        fa.build_empirical_database(str(folder), str(tmp_path / 'x.nod'), load_factors=None, thrust_from='sphere')
