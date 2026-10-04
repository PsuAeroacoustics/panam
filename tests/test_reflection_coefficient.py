import warnings

import numpy as np

import flight_acoustics as fa


def test_boundary_loss_is_continuous_at_large_numerical_distance():
    """F(w) used to drop to zero past |w| = 500, stepping Q there."""
    image_range = np.linspace(100.0, 20000.0, 4000)
    q = fa.spherical_reflection_coefficient(0.02, image_range, 2000.0, 1125.0, 200.0)
    assert np.all(np.isfinite(q))
    assert np.max(np.abs(np.diff(q))) < 1e-3


def test_broadband_ega_at_a_flush_receiver_is_warning_free():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        attenuation, _ = fa.ega(30.0, 0.0, np.array([50.0, 200.0]), np.array([500.0, 2000.0]),
                                343.0, 250.0, pt=False)
    assert np.all(np.isfinite(attenuation))


def test_a_mass_like_admittance_has_no_growing_surface_wave():
    """Im beta > 0 puts the numerical distance's root in the second quadrant.

    The principal root of w is then its negative, whose wofz carries a surface
    wave growing as exp(|Re w|): |Q| reached 1e7 at |w| = 50 and overflowed to
    NaN.  The root itself gives none, and Q stays near the plane coefficient.
    """
    beta = 0.1 + 0.8j
    for cos_grazing in (0.0, 0.02, 0.1):
        for w_abs in (50.0, 600.0, 2.0e3, 1.0e5, 1.0e9):
            # |w| = kappa |cos + beta|^2 / |1 + beta cos|, kappa = pi f R / a
            kappa = w_abs * abs(1.0 + beta * cos_grazing) / abs(cos_grazing + beta) ** 2
            image_range = kappa * 343.0 / (np.pi * 1000.0)
            q = fa.spherical_reflection_coefficient(cos_grazing, image_range, 1000.0, 343.0, None,
                                                    admittance=beta)
            assert np.isfinite(q)
            assert abs(q) < 1.1


def test_ordinary_grounds_are_unchanged_by_taking_the_root_itself():
    """Where the parameter's real part is positive it is the principal sqrt(w):
    Q is what squaring and re-rooting gave, to rounding."""
    from scipy.special import wofz
    rng = np.random.default_rng(3)
    cos_grazing = rng.uniform(0.0, 1.0, 2000)
    image_range = 10.0 ** rng.uniform(0.0, 4.5, 2000)
    f = 10.0 ** rng.uniform(1.0, 4.0, 2000)
    sigma = 10.0 ** rng.uniform(1.0, 5.0, 2000)
    q = fa.spherical_reflection_coefficient(cos_grazing, image_range, f, 343.0, sigma)
    ratio = f / sigma
    inverse = ratio ** -0.73
    beta = 1.0 / (1.0 + 9.08 * inverse / ratio ** 0.02 + 1j * 11.9 * inverse)
    z = np.sqrt(1j * np.pi * f * image_range / 343.0 / (1.0 + beta * cos_grazing)) * (cos_grazing + beta)
    assert np.all(z.real > 0.0)
    w = z ** 2
    plane = (cos_grazing - beta) / (cos_grazing + beta)
    old = plane + (1 + 1j * np.sqrt(np.pi * w) * wofz(np.sqrt(w))) * (1.0 - plane)
    np.testing.assert_allclose(q, old, rtol=1e-12, atol=1e-13)
