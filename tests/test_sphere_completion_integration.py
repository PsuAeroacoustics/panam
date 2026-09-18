"""End-to-end checks of the sphere-completion path against a real sphere file.

These exercise what a NICE-OPS database actually contains.  The failure modes
being guarded against were all invisible to unit tests of the helpers alone:
duplicated (phi, theta) points, hover spheres laid out differently from
forward-flight spheres, and EAA describing a different sphere than dBA.
"""

import os

import numpy as np
import pytest

from flight_acoustics import (Atmosphere, _complete_sphere, average_fore_and_aft,
                              extract_SPL, spla_and_eaa_from_spectrum)


SPHERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'example_data', 'AS350B3108.nc')
DISTANCE = 1000
ATMOSPHERE = Atmosphere(temperature=293.15, pressure=101.325, relative_humidity=20.0)

pytestmark = pytest.mark.skipif(not os.path.exists(SPHERE),
                                reason='example sphere data not available')


@pytest.fixture(scope='module')
def sphere():
    (_, _, phi_list, theta_list, _, _, spla, eaa, _, _,
     frequency, amplitude) = extract_SPL(SPHERE, None, DISTANCE, ATMOSPHERE)
    return phi_list, theta_list, spla, eaa, frequency, amplitude


def _distinct_points(phi, theta):
    return set(zip(phi.flatten().tolist(), theta.flatten().tolist()))


def test_source_sphere_is_a_lower_hemisphere(sphere):
    phi_list, theta_list, _, _, _, amplitude = sphere
    # The premise of the whole mirror: source data is the lower half only.
    assert phi_list.min() == -90.0 and phi_list.max() == 90.0
    assert amplitude.shape[:2] == (phi_list.size, theta_list.size)


def test_forward_sphere_has_no_duplicate_points(sphere):
    phi_list, theta_list, spla, eaa, _, amplitude = sphere
    phi, theta, spla_f, _, _ = _complete_sphere(phi_list, theta_list, spla, eaa, amplitude)

    assert len(_distinct_points(phi, theta)) == phi.size, 'completed sphere has duplicates'
    assert np.unique(phi).size == 36
    assert phi.min() == -180.0 and phi.max() == 170.0
    assert np.isfinite(spla_f).all()


def test_hover_sphere_matches_forward_layout(sphere):
    """The legacy databases got this wrong: hover spheres covered only
    [-90, 90] while forward spheres covered the full circle."""
    phi_list, theta_list, spla, eaa, frequency, amplitude = sphere
    fwd_phi, fwd_theta, _, _, _ = _complete_sphere(phi_list, theta_list, spla, eaa, amplitude)

    averaged = average_fore_and_aft(amplitude)
    spla_h, eaa_h = spla_and_eaa_from_spectrum(averaged, frequency, DISTANCE, ATMOSPHERE)
    hov_phi, hov_theta, hov_spla, hov_eaa, _ = _complete_sphere(
        phi_list, theta_list, spla_h, eaa_h, averaged)

    assert np.array_equal(hov_phi, fwd_phi)
    assert np.array_equal(hov_theta, fwd_theta)
    assert len(_distinct_points(hov_phi, hov_theta)) == hov_phi.size
    assert np.isfinite(hov_spla).all() and np.isfinite(hov_eaa).all()


def test_hover_sphere_is_fore_aft_symmetric(sphere):
    phi_list, theta_list, _, _, frequency, amplitude = sphere
    averaged = average_fore_and_aft(amplitude)
    spla_h, _ = spla_and_eaa_from_spectrum(averaged, frequency, DISTANCE, ATMOSPHERE)
    assert np.allclose(spla_h, spla_h[:, ::-1])


def test_hover_eaa_is_recomputed_not_inherited(sphere):
    """EAA must describe the averaged sphere, not the unaveraged one."""
    phi_list, theta_list, _, eaa_original, frequency, amplitude = sphere
    averaged = average_fore_and_aft(amplitude)
    _, eaa_hover = spla_and_eaa_from_spectrum(averaged, frequency, DISTANCE, ATMOSPHERE)

    assert not np.allclose(eaa_hover, eaa_original)
    assert np.allclose(eaa_hover, eaa_hover[:, ::-1])   # symmetric like the levels


def test_energy_average_exceeds_decibel_mean(sphere):
    """Sanity: averaging in the energy domain is not the same as averaging dB,
    and never falls below it."""
    _, _, _, _, _, amplitude = sphere
    energy = average_fore_and_aft(amplitude)
    n_theta = amplitude.shape[1]
    finite = np.isfinite(amplitude)
    for i in range(n_theta // 2):
        j = n_theta - 1 - i
        both = finite[:, i, :] & finite[:, j, :]
        db_mean = 0.5 * (amplitude[:, i, :] + amplitude[:, j, :])
        assert np.all(energy[:, i, :][both] >= db_mean[both] - 1e-9)
