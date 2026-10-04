"""Fore/aft averaging used to synthesize a hover sphere from low-speed flight.

Two properties are pinned here:

* the average is taken on an energy (p^2) basis, not as an arithmetic mean of
  decibels, and
* it is applied to the spectrum, so SPLA and EAA can both be re-derived from the
  averaged data rather than EAA being carried over from the unaveraged sphere.
"""

import numpy as np

from flight_acoustics import average_fore_and_aft


def test_energy_average_of_a_pair():
    # theta axis of 2: the two bands are each other's fore/aft partner.
    spectrum = np.empty((1, 2, 1))
    spectrum[0, 0, 0] = 100.0
    spectrum[0, 1, 0] = 94.0
    out = average_fore_and_aft(spectrum)

    expected = 10.0 * np.log10(0.5 * (10.0 ** 10.0 + 10.0 ** 9.4))
    assert np.isclose(out[0, 0, 0], expected)
    assert np.isclose(out[0, 1, 0], expected)

    # An energy average sits above the arithmetic mean of the decibel values.
    assert expected > 0.5 * (100.0 + 94.0)


def test_equal_levels_are_unchanged():
    spectrum = np.full((2, 4, 3), 85.0)
    assert np.allclose(average_fore_and_aft(spectrum), 85.0)


def test_middle_band_of_odd_axis_is_its_own_partner():
    spectrum = np.zeros((1, 3, 1))
    spectrum[0, :, 0] = [100.0, 70.0, 94.0]
    out = average_fore_and_aft(spectrum)
    # Bands 0 and 2 pair up; band 1 is the middle and must be untouched.
    assert np.isclose(out[0, 1, 0], 70.0)
    assert np.isclose(out[0, 0, 0], out[0, 2, 0])


def test_result_is_symmetric_fore_to_aft():
    rng = np.random.default_rng(0)
    spectrum = rng.uniform(40.0, 110.0, size=(3, 7, 5))
    out = average_fore_and_aft(spectrum)
    assert np.allclose(out, out[:, ::-1, :])


def test_no_energy_bands_are_handled():
    """-inf marks 'no energy' and must not poison its partner."""
    spectrum = np.empty((1, 2, 1))
    spectrum[0, 0, 0] = -np.inf
    spectrum[0, 1, 0] = 100.0
    out = average_fore_and_aft(spectrum)
    # Half the energy of the live band: 100 dB - 3.01 dB.
    assert np.isclose(out[0, 0, 0], 100.0 - 10.0 * np.log10(2.0))
    assert np.isfinite(out).all()


def test_input_is_not_mutated():
    spectrum = np.full((1, 2, 1), 90.0)
    spectrum[0, 1, 0] = 80.0
    before = spectrum.copy()
    average_fore_and_aft(spectrum)
    assert np.array_equal(spectrum, before)
