"""Exact properties of the hemisphere geometry and gridding: the Lambert projection and
its inverse, the ART <-> UMAPR conversions, and the Shepard weights."""

import numpy as np
import pytest

import flight_acoustics as fa


def _directions(n=400, seed=0):
    """UMAPR (azimuth, elevation), radians, uniform on the lower hemisphere, off the pole."""
    rng = np.random.default_rng(seed)
    azimuth = rng.uniform(0.0, 2.0 * np.pi, n)
    elevation = np.arcsin(rng.uniform(0.0, 0.999, n))
    return azimuth, elevation


def test_the_lambert_inverse_undoes_the_projection():
    azimuth, elevation = _directions()
    x, y = fa.lambert_ea(elevation, fa.lambert_lon(azimuth))
    azimuth_back, elevation_back = fa._lambert_xy_to_umapr(x, y)
    np.testing.assert_allclose(elevation_back, elevation, rtol=0, atol=1e-12)
    wrapped = np.angle(np.exp(1j * (azimuth_back - azimuth)))
    np.testing.assert_allclose(wrapped, 0.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize('lat', [0.0, 0.5, 1.0, 1.5])
@pytest.mark.parametrize('lon', [0.3, 2.0, -2.5])
def test_the_lambert_projection_is_equal_area(lat, lon):
    """Its Jacobian is the sphere's area element, cos(lat) dlat dlon."""
    step = 1e-6
    x, y = fa.lambert_ea(np.array([lat + step, lat - step]), np.full(2, lon))
    x_lat, y_lat = (x[0] - x[1]) / (2 * step), (y[0] - y[1]) / (2 * step)
    x, y = fa.lambert_ea(np.full(2, lat), np.array([lon + step, lon - step]))
    x_lon, y_lon = (x[0] - x[1]) / (2 * step), (y[0] - y[1]) / (2 * step)
    assert abs(x_lat * y_lon - x_lon * y_lat) == pytest.approx(np.cos(lat), rel=1e-8, abs=1e-9)


def test_uniform_directions_fill_equal_areas_of_the_page_equally():
    azimuth, elevation = _directions(200000, seed=1)
    x, y = fa.lambert_ea(elevation, fa.lambert_lon(azimuth))
    q = np.hypot(x, y)
    # Annuli of equal page area: equal counts (the rim, q = sqrt(2), is the horizon).
    edges = np.sqrt(np.linspace(0.0, 2.0, 5))
    counts = np.histogram(q, edges)[0]
    assert np.ptp(counts) / counts.mean() < 0.03
    sectors = np.histogram(np.arctan2(y, x), np.linspace(-np.pi, np.pi, 9))[0]
    assert np.ptp(sectors) / sectors.mean() < 0.03


def test_art_and_umapr_convert_back_and_forth():
    rng = np.random.default_rng(2)
    phi = rng.uniform(-np.pi / 2, np.pi / 2, 300)
    theta = rng.uniform(0.01, np.pi - 0.01, 300)
    azimuth, elevation = fa.art2umapr(phi, theta)
    phi_back, theta_back = fa._umapr2art(azimuth, elevation)
    np.testing.assert_allclose(phi_back, phi, rtol=0, atol=1e-12)
    np.testing.assert_allclose(theta_back, theta, rtol=0, atol=1e-12)
    # NORAH2's angles are ART's.
    azimuth_n, elevation_n = fa.norah2umapr(phi, theta)
    np.testing.assert_allclose(azimuth_n, azimuth, rtol=0, atol=1e-14)
    np.testing.assert_allclose(elevation_n, elevation, rtol=0, atol=1e-14)


def _samples(n=300, seed=3):
    rng = np.random.default_rng(seed)
    felv = np.degrees(np.arcsin(rng.uniform(0.0, 1.0, n)))
    fazi = rng.uniform(0.0, 360.0, n)
    values = rng.uniform(50.0, 90.0, n)
    return felv, fazi, values


def test_shepard_reproduces_a_sample_at_its_own_position():
    felv, fazi, values = _samples()
    at_samples = fa.shepIDW(felv[:40], fazi[:40], felv, fazi, values, 25.0)
    # To the arc cosine's roundoff: a node on a sample is a few 1e-7 deg from it, where
    # the weight, growing as 1/h^2, leaves the others about 1e-13 of the total.
    np.testing.assert_allclose(at_samples, values[:40], rtol=0, atol=1e-9)


def test_shepard_weights_are_positive_and_sum_to_one():
    felv, fazi, _ = _samples()
    ielv, iazi = np.meshgrid(np.arange(1.0, 90.0, 7.0), np.arange(3.0, 360.0, 11.0))
    _, neighbors = fa.shepIDW_weights(ielv, iazi, felv, fazi, 25.0)
    for node, (near, weights) in enumerate(neighbors):
        assert near.size and np.all(weights > 0.0)
        assert weights.sum() == pytest.approx(1.0, abs=1e-12)
        h = fa.geodist(ielv.ravel()[node], iazi.ravel()[node], felv[near], fazi[near])
        assert np.all(h <= 25.0)


def test_shepard_is_the_same_on_either_side_of_the_azimuth_seam_and_at_the_pole():
    felv, fazi, values = _samples()
    ielv = np.array([10.0, 45.0, 80.0])
    east = fa.shepIDW(ielv, np.full(3, 359.9), felv, fazi, values, 25.0)
    west = fa.shepIDW(ielv, np.full(3, -0.1), felv, fazi, values, 25.0)
    np.testing.assert_allclose(east, west, rtol=1e-12)
    pole = fa.shepIDW(np.full(36, 90.0), np.arange(0.0, 360.0, 10.0), felv, fazi, values, 25.0)
    assert np.ptp(pole) < 1e-9


@pytest.mark.parametrize('aspect', [1.0, 5.0])
def test_adaptive_weights_at_a_sample_and_across_the_seam(aspect):
    felv, fazi, values = _samples(600)
    mic = np.arange(felv.size) % 7
    resolution = np.full(felv.size, 1.0)
    w, _, gap = fa.adaptive_idw_weights(felv[:30], fazi[:30], felv, fazi, mic, resolution, aspect=aspect)
    assert not gap.any()
    np.testing.assert_allclose(np.asarray(w @ values).ravel(), values[:30], rtol=0, atol=1e-9)
    ielv = np.array([10.0, 45.0, 80.0])
    east, *_ = fa.adaptive_idw_weights(ielv, np.full(3, 359.9), felv, fazi, mic, resolution, aspect=aspect)
    west, *_ = fa.adaptive_idw_weights(ielv, np.full(3, -0.1), felv, fazi, mic, resolution, aspect=aspect)
    np.testing.assert_allclose(np.asarray(east @ values).ravel(), np.asarray(west @ values).ravel(),
                               rtol=1e-12)
    sums = np.asarray(east.sum(axis=1)).ravel()
    np.testing.assert_allclose(sums, 1.0, rtol=0, atol=1e-12)
    assert east.data.min() > 0.0
