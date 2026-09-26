"""Validation of the ground-effect and BEM models against independent references.

Where tests/test_ground_plane.py and tests/test_axisymmetric_bem.py check the
code against its own limits and against its other formulations, these compare
it with solutions obtained another way:

- the exact half-space Green's function (complex image) against the Sommerfeld
  (Hankel-transform) integral, computed directly by quadrature;
- the axisymmetric BEM against the exact series for a rigid sphere: a rigid
  hemisphere on a rigid ground is, by image symmetry, a sphere in free field
  struck by the direct and the reflected wave;
- the three plate formulations (thin impedance-jump disc, 3-D raised plate,
  axisymmetric raised plate), coded independently, against each other;
- the impedance, turbulence, microphone-response and geometry pieces against
  their published formulas and tables.
"""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import j0, spherical_jn, spherical_yn, eval_legendre

import axisymmetric_bem as ab
import ground_plane as gp

C = 1125.0


# --------------------------------------------------------------------------
# Exact Green's function vs the Sommerfeld integral
# --------------------------------------------------------------------------

def sommerfeld_reflected(k, beta, rho, z_sum):
    """(i/4 pi) int_0^inf J0(kappa rho) R(kappa) e^{i gamma Z} kappa/gamma dkappa, by quadrature.

    R = (gamma - k beta)/(gamma + k beta), gamma = sqrt(k^2 - kappa^2), Im >= 0
    (e^{-i omega t}).  kappa = k sin t below k and k cosh u above it remove the
    branch point; the second part decays as e^{-k Z sinh u}.
    """
    def below(t, part):
        g = k * np.cos(t)
        r = (g - k * beta) / (g + k * beta)
        v = 1j / (4 * np.pi) * j0(k * rho * np.sin(t)) * r * np.exp(1j * g * z_sum) * k * np.sin(t)
        return v.real if part == 0 else v.imag

    def above(u, part):
        g = 1j * k * np.sinh(u)
        r = (g - k * beta) / (g + k * beta)
        v = k * np.cosh(u) / (4 * np.pi) * j0(k * rho * np.cosh(u)) * r * np.exp(-k * z_sum * np.sinh(u))
        return v.real if part == 0 else v.imag

    u_max = np.arcsinh(40.0 / (k * z_sum))
    total = 0.0
    for part, unit in ((0, 1.0), (1, 1j)):
        a, _ = quad(below, 0.0, 0.5 * np.pi, args=(part,), limit=400, epsabs=1e-12, epsrel=1e-10)
        b, _ = quad(above, 0.0, u_max, args=(part,), limit=2000, epsabs=1e-12, epsrel=1e-10)
        total += unit * (a + b)
    return total


@pytest.mark.parametrize('f, ground', [
    (1000.0, dict(model='delany_bazley', sigma=100.0)),
    (4000.0, dict(model='delany_bazley', sigma=100.0)),
    (1000.0, dict(model='variable_porosity', sigma_e=200.0, alpha_e=1e-8)),
])
@pytest.mark.parametrize('rho, z_sum', [(0.05, 0.1), (0.3, 0.1), (1.5, 0.3), (0.3, 1.0), (3.0, 0.5)])
def test_exact_green_matches_the_sommerfeld_integral(f, ground, rho, z_sum):
    k = 2 * np.pi * f / C
    params = {n: v for n, v in ground.items() if n != 'model'}
    beta = complex(gp.surface_admittance(f, ground['model'], sound_speed_mps=C * 0.3048, **params))
    reference = sommerfeld_reflected(k, beta, rho, z_sum)
    r2 = np.hypot(rho, z_sum)
    big_i, _ = gp.image_integrals(k, beta, np.array([rho]), np.array([z_sum]))
    image = np.exp(1j * k * r2) / (4 * np.pi * r2) - 2 * k * beta * big_i[0]
    # Measured: agreement to 2e-9 relative (Weyl-van der Pol is off by up to 6% here).
    assert abs(image - reference) <= 1e-6 * abs(reference)


# --------------------------------------------------------------------------
# Axisymmetric BEM vs the rigid-sphere series (hemisphere on rigid ground)
# --------------------------------------------------------------------------

def sphere_surface_pressure(ka, cos_gamma, n_terms=60):
    """Total pressure on a rigid sphere for a unit plane wave, at angle gamma from its direction."""
    total = 0.0 + 0.0j
    for n in range(n_terms):
        dh = spherical_jn(n, ka, derivative=True) + 1j * spherical_yn(n, ka, derivative=True)
        total += (1j ** n) * (2 * n + 1) * eval_legendre(n, cos_gamma) * 1j / (ka ** 2 * dh)
    return total


def hemisphere(radius, n_segments):
    theta = np.linspace(0.0, 0.5 * np.pi, n_segments + 1)
    r, z = radius * np.sin(theta), radius * np.cos(theta)
    segs = np.column_stack((r[:-1], z[:-1], r[1:], z[1:]))
    return segs, np.zeros(n_segments, dtype=bool)


@pytest.mark.parametrize('ka', [0.5, 2.0, 5.0])
@pytest.mark.parametrize('el, az', [(20.0, 0.0), (60.0, 90.0), (5.0, 250.0)])
def test_axisymmetric_bem_matches_the_rigid_sphere(ka, el, az):
    radius = 0.3
    f = ka * C / (2 * np.pi * radius)
    segs, flat = hemisphere(radius, 90)
    for i_mic in (20, 60):                          # two microphone positions, at segment midpoints
        mic_rz = 0.5 * (segs[i_mic, :2] + segs[i_mic, 2:])
        pd, pr = ab.scattering([f], [el], [az], C, flow_resistance=1e14, generator=(segs, flat),
                               mic_rz=mic_rz, mic=(0.0, 1.0), radius=radius, thickness=radius,
                               extra_modes=12)
        bem = pd[0, 0, 0] + pr[0, 0, 0]              # rigid ground: Q = 1
        # Exact: the sphere's response to the direct and the image-reflected waves.
        k = 2 * np.pi * f / C
        x = np.array([0.0, mic_rz[0], mic_rz[1]]); xhat = x / np.linalg.norm(x)
        e, a = np.radians(el), np.radians(az)
        kd = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), -np.sin(e)])
        kr = kd * [1.0, 1.0, -1.0]
        # Both waves are referred to the sphere's centre, on the ground, where they are equal.
        exact = sphere_surface_pressure(ka, kd @ xhat) + sphere_surface_pressure(ka, kr @ xhat)
        direct = np.exp(1j * k * radius * (kd @ xhat))
        exact_ratio = exact / direct
        # Measured: 2.7e-4 at worst (0.002 dB), median 3e-5.
        assert abs(bem - exact_ratio) <= 2e-3 * abs(exact_ratio)


# --------------------------------------------------------------------------
# Plate formulations against each other
# --------------------------------------------------------------------------

THIN = dict(thickness=0.0003 / 0.3048, edge_thickness=0.0003 / 0.3048, taper_length=0.0)


@pytest.mark.parametrize('f', [250.0, 500.0, 1000.0])
def test_three_plate_formulations_agree_in_the_thin_limit(f):
    els, azs = [5.0, 30.0, 70.0], [90.0, 270.0]
    thin_disc = gp.disc_bem_scattered([f], els, azs, C, min_cells_across=36)
    axi = ab.scattering([f], els, azs, C, **THIN)
    for i, e in enumerate(els):
        q = gp.fa.spherical_reflection_coefficient(np.sin(np.radians(e)), 1500.0, f, C, gp.FLOW_RESISTANCE)
        a = 20 * np.log10(np.abs((1 + q) * (1 + thin_disc[0, i])))
        b = 20 * np.log10(np.abs(axi[0][0, i] + q * axi[1][0, i]))
        assert np.all(np.abs(a - b) < 0.25), (f, e, a, b)


def test_axisymmetric_and_3d_raised_plate_agree():
    for f in (500.0, 2000.0):
        els, azs = [10.0, 45.0], [90.0, 270.0]
        axi = ab.scattering([f], els, azs, C)
        s3 = gp.raised_plate_scattering([f], els, azs, C, min_cells_across=24)
        for i, e in enumerate(els):
            q = gp.fa.spherical_reflection_coefficient(np.sin(np.radians(e)), 1500.0, f, C, gp.FLOW_RESISTANCE)
            a = 20 * np.log10(np.abs(axi[0][0, i] + q * axi[1][0, i]))
            b = 20 * np.log10(np.abs(s3[0][0, i] + q * s3[1][0, i]))
            assert np.all(np.abs(a - b) < 0.3), (f, e, a, b)


# --------------------------------------------------------------------------
# Impedance, turbulence, microphone response, geometry
# --------------------------------------------------------------------------

def test_impedance_models_reproduce_their_formulas():
    # Delany-Bazley and Miki at X = f/sigma = 1.
    assert 1 / gp.surface_admittance(100.0, 'delany_bazley', sigma=100.0) == pytest.approx(1 + 9.08 + 11.9j)
    assert 1 / gp.surface_admittance(100.0, 'miki', sigma=100.0) == pytest.approx(1 + 5.50 + 8.43j)
    # Variable porosity, first term: the classic 0.436 (1 + i) sqrt(sigma_e/f) with sigma_e in Pa s/m^2.
    z = 1 / gp.surface_admittance(1000.0, 'variable_porosity', sigma_e=200.0, alpha_e=0.0)
    assert z == pytest.approx(0.436 * (1 + 1j) * np.sqrt(200e3 / 1000.0), rel=3e-3)


def test_hard_backed_layer_tends_to_the_semi_infinite_ground():
    f = np.array([100.0, 1000.0, 5000.0])
    deep = gp.surface_admittance(f, 'delany_bazley_layer', sigma=200.0, depth=5.0)
    semi = gp.surface_admittance(f, 'delany_bazley', sigma=200.0)
    assert np.allclose(deep, semi, rtol=0.02)


@pytest.mark.parametrize('model, params', [('delany_bazley', dict(sigma=50.0)), ('miki', dict(sigma=50.0)),
                                           ('variable_porosity', dict(sigma_e=100.0, alpha_e=30.0)),
                                           ('delany_bazley_layer', dict(sigma=250.0, depth=0.05)),
                                           ('delany_bazley_layer', dict(sigma=1300.0, depth=0.11))])
def test_impedance_models_are_passive(model, params):
    # Re(beta) > 0 is what the exact Green's function (and physics) requires.
    f = np.geomspace(10.0, 20000.0, 200)
    assert np.all(gp.surface_admittance(f, model, **params).real > 0.0)


def test_thin_delany_bazley_layer_is_not_passive_at_low_frequency_and_is_refused():
    # A known defect of the one-parameter model, made worse by a thin layer.
    f = np.geomspace(10.0, 20000.0, 400)
    beta = gp.surface_admittance(f, 'delany_bazley_layer', sigma=100.0, depth=0.03)
    assert np.all(beta[f > 100.0].real > 0.0) and np.any(beta[f < 50.0].real <= 0.0)
    bad = complex(beta[0])
    with pytest.raises(ValueError, match='passive'):
        gp.image_integrals(2 * np.pi * 10.0 / C, bad, np.array([0.1]), np.array([0.0]))


def test_harmonoise_coherence_by_hand():
    f, hs, hr, r, gamma_t, c = 2000.0, 100.0, 1.22, 300.0, 1e-6, 343.0
    k = 2 * np.pi * f / c
    gamma_p = hs * hr / (hs + hr)
    expected = np.exp(-3 / 8 * 0.364 * k ** 2 * gamma_p ** (5 / 3) * r * gamma_t)
    assert gp.turbulence_coherence(f, hs, hr, r, gamma_t, c) == pytest.approx(expected)
    assert gp.turbulence_coherence(f, hs, 0.0, r, gamma_t, c) == pytest.approx(1.0)


def test_microphone_response_reproduces_the_gras_table():
    t = gp.GRAS_40AE_FREE_FIELD_CORRECTIONS
    for row in t[::6]:
        for j, angle in enumerate([0, 30, 60, 90, 120, 150, 180]):
            assert gp.free_field_microphone_response(row[0], angle) == pytest.approx(row[1 + j] - row[1], abs=1e-9)
    assert gp.free_field_microphone_response(250.0, 90.0) == 0.0


def test_pole_incidence_geometry():
    # Source overhead.  Horizontal axis: both paths at 90 deg; vertical axis: 0 and 180 deg.
    d, r = gp.pole_incidence(0.0, 0.0, 500.0, 4.0, [0.0, 1.0, 0.0])
    assert float(d) == pytest.approx(90.0) and float(r) == pytest.approx(90.0)
    d, r = gp.pole_incidence(0.0, 0.0, 500.0, 4.0, [0.0, 0.0, 1.0])
    assert float(d) == pytest.approx(0.0, abs=1e-6) and float(r) == pytest.approx(180.0, abs=1e-6)
    # The certification axis is normal to the plane through the flight line and the pole.
    axis = gp.certification_axis(500.0, 500.0, 4.0)
    assert axis @ np.array([1.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert axis @ np.array([0.0, -500.0, 496.0]) == pytest.approx(0.0, abs=1e-9)


def test_roughness_factor_limits():
    bands = np.array([4000.0])
    smooth = gp.pole_level(bands, [500.0], [100.0], 4.0, C, 'delany_bazley', sigma=1e6)
    rough = gp.pole_level(bands, [500.0], [100.0], 4.0, C, 'delany_bazley', roughness=0.03, sigma=1e6)
    assert rough[0, 0] < smooth[0, 0]            # steep incidence: less coherent reflection
    grazing_s = gp.pole_level(bands, [1.0], [5000.0], 4.0, C, 'delany_bazley', sigma=1e6)
    grazing_r = gp.pole_level(bands, [1.0], [5000.0], 4.0, C, 'delany_bazley', roughness=0.03, sigma=1e6)
    assert grazing_r[0, 0] == pytest.approx(grazing_s[0, 0], abs=0.05)
