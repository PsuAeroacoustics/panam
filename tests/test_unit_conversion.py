"""unit_conversion must never modify the array it is given.

Every converter used in-place arithmetic (``L /= 0.3048``) on its argument, so a
numpy array passed in came back rewritten in the pivot unit, and an integer
array raised.  depropagate_hemisphere converts its emission ranges this way;
with length_units='m' the filter-bank pass then reused the rewritten ranges and
came out about 10 dB high.
"""

import numpy as np
import pytest

import unit_conversion as uc

CASES = [
    (uc.len_conv, 'm', 'ft'), (uc.len_conv, 'in', 'm'), (uc.area_conv, 'm**2', 'in**2'),
    (uc.vol_conv, 'l', 'm**3'), (uc.speed_conv, 'km/h', 'm/s'), (uc.press_conv, 'mb', 'psi'),
    (uc.temp_conv, 'C', 'K'), (uc.density_conv, 'lb/ft**3', 'kg/m**3'), (uc.force_conv, 'N', 'lb'),
    (uc.wt_conv, 'kg', 'lb'), (uc.power_conv, 'W', 'hp'), (uc.avgas_conv, 'USG', 'kg'),
]


@pytest.mark.parametrize('convert,from_units,to_units', CASES)
def test_input_array_is_left_alone(convert, from_units, to_units):
    values = np.array([1.0, 20.0, 300.0])
    before = values.copy()
    result = convert(values, from_units=from_units, to_units=to_units)
    np.testing.assert_array_equal(values, before)
    assert result is not values


@pytest.mark.parametrize('convert,from_units,to_units', CASES)
def test_integer_input_matches_float_input(convert, from_units, to_units):
    np.testing.assert_allclose(convert(np.array([1, 20, 300]), from_units=from_units, to_units=to_units),
                               convert(np.array([1.0, 20.0, 300.0]), from_units=from_units, to_units=to_units))


def test_scalars_stay_scalars():
    assert isinstance(uc.len_conv(1.0, from_units='m', to_units='ft'), float)
    assert uc.len_conv(0.3048, from_units='m', to_units='ft') == pytest.approx(1.0)


def test_pounds_and_kilograms_convert_through_one_exact_constant():
    """lb -> kg used 0.453592 and kg -> lb 2.204622622, which are not
    reciprocals: lb -> lb came back 3.7e-8 low."""
    weights = np.array([1.0, 2250.0, 1e6])
    np.testing.assert_allclose(uc.wt_conv(weights, from_units='lb', to_units='lb'), weights, rtol=1e-15)
    assert uc.wt_conv(1.0, from_units='lb', to_units='kg') == 0.45359237
    np.testing.assert_allclose(uc.wt_conv(uc.wt_conv(weights, from_units='kg', to_units='lb'),
                                          from_units='lb', to_units='kg'), weights, rtol=1e-15)


def test_feet_per_second_does_not_follow_the_default_length_units(monkeypatch):
    """The ft/s branches converted through default_length_units, so changing
    that default to metres would have made 'ft/s' mean m/s."""
    expected = (uc.speed_conv(100.0, 'ft/s', 'kt'), uc.speed_conv(100.0, 'kt', 'ft/s'))
    monkeypatch.setattr(uc, 'default_length_units', 'm')
    assert (uc.speed_conv(100.0, 'ft/s', 'kt'), uc.speed_conv(100.0, 'kt', 'ft/s')) == expected
    assert uc.speed_conv(100.0, 'ft/s', 'm/s') == pytest.approx(30.48, rel=1e-12)


@pytest.mark.parametrize('units', [dict(from_units='BTU/hr'), dict(to_units='BTU/mn')])
def test_power_error_names_only_accepted_units(units):
    with pytest.raises(ValueError) as raised:
        uc.power_conv(1.0, **units)
    assert 'BTU' not in str(raised.value)


def test_avgas_grade_error_names_the_real_default():
    with pytest.raises(ValueError, match='default of "nominal"'):
        uc.avgas_conv(1.0, 'USG', 'lb', grade='91')
