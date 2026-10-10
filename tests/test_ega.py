import numpy as np
import pytest
from flight_acoustics import ega


from flight_acoustics import spherical_reflection_coefficient

# Three geometries (hs, hr, d2 in ft) and three grounds (kPa s/m^2).
GEOMETRIES = [(10.0, 5.0, 100.0), (5.0, 4.0, 1000.0), (300.0, 4.0, 150.0)]
FLOW_RESISTANCES = [50.0, 200.0, 500.0]


def _two_paths(hs, hr, d2, f, a, flores):
    """Q, the range ratio R1/R2 and the path delay of the direct and image paths."""
    r1, r2 = np.hypot(d2, hs - hr), np.hypot(d2, hs + hr)
    q = spherical_reflection_coefficient((hs + hr) / r2, r2, f, a, flores)
    return q, r1 / r2, (r2 - r1) / a


@pytest.mark.parametrize('flores', FLOW_RESISTANCES)
@pytest.mark.parametrize('hs, hr, d2', GEOMETRIES)
def test_ega_pure_tone_is_the_two_path_sum(hs, hr, d2, flores):
    # 20 lg |1 + Q e^{ik dR} R1/R2|, with the spherical-wave Q at the image
    # path's angle from the normal.
    f, a = np.array([63.0, 250.0, 500.0, 1000.0, 2000.0, 8000.0]), 1116.0
    atten, phase = ega(hs, hr, d2, f, a, flores, pt=True)
    q, ratio, delay = _two_paths(hs, hr, d2, f, a, flores)
    total = 1.0 + q * np.exp(2j * np.pi * f * delay) * ratio
    np.testing.assert_allclose(atten, 20.0 * np.log10(np.abs(total)), rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(phase, np.angle(total), rtol=0.0, atol=1e-9)


@pytest.mark.parametrize('cturb', [0.0, 8.8e-4])
@pytest.mark.parametrize('hs, hr, d2', GEOMETRIES)
def test_ega_band_average_is_chessells_form(hs, hr, d2, cturb):
    # Chessell's equations 20-21 (mu = 0.727477, eta = 6.325159) with the
    # turbulence factor exp(-(cturb f sqrt(R1) / 2)^2) on the cross term.
    f, a, flores = np.array([63.0, 250.0, 1000.0, 4000.0]), 1116.0, 200.0
    atten, phase = ega(hs, hr, d2, f, a, flores, pt=False, cturb=cturb)
    q, ratio, delay = _two_paths(hs, hr, d2, f, a, flores)
    m, x = np.abs(q) * ratio, f * delay
    turbulence = np.exp(-(0.5 * cturb * f * np.sqrt(np.hypot(d2, hs - hr))) ** 2)
    energy = 1.0 + m ** 2 + 2.0 * m * np.cos(6.325159 * x + np.angle(q)) * turbulence * \
        np.sin(0.727477 * x) / (0.727477 * x)
    np.testing.assert_allclose(atten, 10.0 * np.log10(energy), rtol=0.0, atol=1e-9)
    assert np.all(np.isnan(phase))


def test_ega_disable_boundary_loss_correction_matches_plane_wave():
    hs = 10.0
    hr = 5.0
    d2 = 100.0
    f = np.array([200.0, 500.0, 1000.0])
    a = 1116.0
    flores = 200.0

    atten, phase = ega(
        hs=hs,
        hr=hr,
        d2=d2,
        f=f,
        a=a,
        flores=flores,
        pt=True,
        cturb=0.0,
        boundary_loss_correction=False,
    )

    # Compute expected values using plane-wave reflection coefficient only
    direct_range = np.sqrt(d2**2 + (hs - hr)**2)
    image_range = np.sqrt(d2**2 + (hs + hr)**2)
    grazing_angle = np.arccos((hs + hr) / image_range)
    path_delay = (image_range - direct_range) / a
    range_ratio = image_range / direct_range

    freq_resistance_ratio = f / flores
    inv_freq_ratio = freq_resistance_ratio ** (-0.73)
    impedance_ratio = 1.0 / (
        1.0
        + 9.08 * inv_freq_ratio / (freq_resistance_ratio ** 0.02)
        + 1j * 11.9 * inv_freq_ratio
    )
    cos_grazing = np.cos(grazing_angle)
    plane_wave_coeff = (cos_grazing - impedance_ratio) / (cos_grazing + impedance_ratio)

    phase_delay = 1j * 2.0 * np.pi * f * path_delay
    resultant_amplitude = 1.0 + plane_wave_coeff * np.exp(phase_delay) / range_ratio
    expected_atten = 10.0 * np.log10(np.abs(resultant_amplitude) ** 2)
    expected_phase = np.angle(resultant_amplitude)

    assert np.allclose(atten, expected_atten)
    assert np.allclose(phase, expected_phase)
