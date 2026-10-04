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
