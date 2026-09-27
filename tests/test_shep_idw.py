"""Modified Shepard (Franke-Nielson) interpolation over the hemisphere."""

import numpy as np

import flight_acoustics as fa


def test_a_bad_sample_only_reaches_its_own_neighbourhood():
    """0 * NaN is NaN: one bad sample used to turn every node NaN, however far away."""
    rng = np.random.default_rng(0)
    felv, fazi, values = rng.uniform(0, 90, 200), rng.uniform(0, 360, 200), rng.uniform(1, 2, 200)
    values[0] = np.nan
    ielv, iazi = np.meshgrid(np.arange(0.0, 91.0, 15.0), np.arange(0.0, 361.0, 30.0))
    result = fa.shepIDW(ielv, iazi, felv, fazi, values, 25.0)
    far = fa.geodist(ielv, iazi, felv[0], fazi[0]) > 25.0
    assert far.any() and np.all(np.isfinite(result[far]))


def test_query_shape_is_preserved():
    """A one-element array query comes back as an array, a scalar query as a float."""
    felv, fazi, values = np.array([10.0, 20.0, 30.0]), np.array([0.0, 10.0, 20.0]), np.array([1.0, 2.0, 3.0])
    assert np.shape(fa.shepIDW(np.array([15.0]), np.array([5.0]), felv, fazi, values, 25.0)) == (1,)
    assert isinstance(fa.shepIDW(15.0, 5.0, felv, fazi, values, 25.0), float)
