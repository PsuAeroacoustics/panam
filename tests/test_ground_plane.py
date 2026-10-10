import numpy as np
import pytest

import ground_plane as gp

C = 1125.0  # ft/s
BANDS = np.array([50.0, 200.0, 1000.0, 4000.0, 10000.0])


def test_emission_geometry_straight_pass():
    # Level pass at 500 ft, 100 ft/s along x, microphone at the origin.
    t = np.linspace(0.0, 40.0, 4001)
    track = {'time': t, 'x': -2000.0 + 100.0 * t, 'y': np.zeros_like(t), 'z': np.full_like(t, 500.0)}
    emitted = 20.0                                   # overhead
    reception = emitted + 500.0 / C
    g = gp.emission_geometry(track, [0.0, 0.0, 0.0], [reception, -5.0], C)
    assert g['elevation'][0] == pytest.approx(90.0, abs=0.05)
    assert g['source_height'][0] == pytest.approx(500.0)
    assert g['ground_distance'][0] == pytest.approx(0.0, abs=0.5)
    assert np.isnan(g['elevation'][1])               # before anything could arrive


def test_band_levels_from_psd_count_every_bin_once():
    # Nominal centers' own c * 2**(+-1/6) edges leave 1403-1425 Hz in no band
    # and 891-898 Hz in two; the bands must tile, so the band powers add up to
    # the PSD's total over them.
    frequency = np.arange(0.0, 12000.0, 1.0)
    psd = np.zeros((frequency.size, 3))
    psd[1410, 0] = 4e-10                            # a tone in the old gap
    psd[895, 1] = 4e-10                             # and one in the old overlap
    psd[:, 2] = 4e-10                               # white
    bands = gp.fa.PNL_BAND_FREQUENCIES
    levels = gp.band_levels_from_psd(frequency, psd, bands)
    power = 10.0 ** (levels / 10.0) * 4e-10
    for tone in (0, 1):
        assert np.sum(power[:, tone] > 1e-20) == 1
        assert power[:, tone].max() == pytest.approx(4e-10, rel=1e-12)
    lower, upper = gp.fa.third_octave_band_edges(bands)
    covered = (frequency >= lower[0]) & (frequency < upper[-1])
    assert power[:, 2].sum() == pytest.approx(psd[covered, 2].sum(), rel=1e-12)
    # 1410 Hz falls in the 1250 Hz band, whose base-10 upper edge is 1412.5 Hz.
    assert levels[list(bands).index(1250.0), 0] == pytest.approx(0.0, abs=1e-9)


def test_flat_is_pressure_doubling():
    assert np.allclose(gp.board_rigid_plane(BANDS, [100.0, 50.0], [10.0, 2000.0]), 6.0206, atol=1e-3)


def test_flush_plate_is_exact_doubling():
    # The 67AX diaphragm is flush in the plate: no height, so no interference
    # even at 10 kHz and normal incidence.
    level = gp.board_uniform(BANDS, [300.0, 50.0], [0.0, 1000.0], C)
    assert np.allclose(level, 6.02, atol=0.01)


def test_raised_microphone_interferes_at_high_frequency():
    # For contrast, a diaphragm 8 mm above a rigid plate (the geometry the
    # 67AX is not): at 1 kHz the 16 mm path difference is worth 0.1 dB.
    level = gp.board_uniform(np.array([1000.0]), [300.0], [0.0], C, mic_height=0.008 / 0.3048)[0, 0]
    assert level == pytest.approx(10 * np.log10(2 + 2 * np.cos(2 * np.pi * 1000 * 2 * 0.008 / 343)), abs=0.02)


def test_fresnel_strip_weight_limits():
    # Steep incidence, high frequency: the zone is on the plate.
    assert gp.fresnel_strip_weight([10000.0], [500.0], [10.0], C)[0, 0] == 0.0
    # Grazing, low frequency: the zone reaches far onto the soft ground.
    assert gp.fresnel_strip_weight([50.0], [20.0], [3000.0], C)[0, 0] > 0.99
    # A plate much larger than any zone is all plate; a vanishing one all ground.
    assert gp.fresnel_strip_weight([200.0], [100.0], [500.0], C, radius=1e5)[0, 0] == 0.0
    assert gp.fresnel_strip_weight([200.0], [100.0], [500.0], C, radius=1e-9)[0, 0] == pytest.approx(1.0)


def test_fresnel_strip_weight_grows_toward_grazing_and_low_frequency():
    elevations = np.radians([60.0, 20.0, 5.0, 1.0])
    r = gp.fresnel_strip_weight([500.0], 1000.0 * np.sin(elevations), 1000.0 * np.cos(elevations), C)[0]
    assert np.all(np.diff(r) >= 0.0)
    # Across frequency only broadly: once the zone reaches past the microphone
    # the single-boundary model's r stops being monotonic.
    by_band = gp.fresnel_strip_weight(BANDS, [100.0], [1000.0], C)[:, 0]
    assert by_band[0] > by_band[-1]


def test_fresnel_ellipse_matches_sampling():
    for hs, d2, hr, excess in ((40.0, 120.0, 4.0, 3.0), (500.0, 10.0, 0.026, 0.0375)):
        center, ax, ay = gp.fresnel_ellipse(hs, d2, hr, excess)
        # Brute force: ground points within the path-length limit.
        x = center + np.linspace(-1.2, 1.2, 1201) * ax
        y = np.linspace(-1.2, 1.2, 1201) * ay
        X, Y = np.meshgrid(x, y)
        inside = (np.sqrt((X + d2) ** 2 + Y ** 2 + hs ** 2) + np.sqrt(X ** 2 + Y ** 2 + hr ** 2)
                  <= np.hypot(d2, hs + hr) + excess)
        sampled = inside.sum() * (x[1] - x[0]) * (y[1] - y[0])
        assert np.pi * ax * ay == pytest.approx(sampled, rel=0.01)
        # And it is an ellipse: every point inside the fitted one is in the zone.
        fitted = ((X - center) / ax) ** 2 + (Y / ay) ** 2 <= 1.0
        assert np.mean(fitted == inside) > 0.999


def test_fresnel_disc_weight_limits_and_bounds():
    assert gp.fresnel_disc_weight([10000.0], [500.0], [10.0], C)[0, 0] == pytest.approx(1.0)
    # The two sampling branches agree where the ellipse and the plate are similar in size.
    w_small = gp.fresnel_disc_weight([2000.0], [60.0], [100.0], C, samples=(200, 400))[0, 0]
    assert gp.fresnel_disc_weight([2000.0], [60.0], [100.0], C)[0, 0] == pytest.approx(w_small, abs=0.02)
    assert gp.fresnel_disc_weight([50.0], [20.0], [3000.0], C)[0, 0] < 0.01
    w = gp.fresnel_disc_weight(BANDS, [100.0, 30.0], [300.0, 2000.0], C)
    assert np.all((w >= 0.0) & (w <= 1.0))


def test_models_lie_between_plate_and_soft_ground():
    hs, d2 = [150.0, 40.0, 15.0], [200.0, 1500.0, 3000.0]
    plate = gp.board_uniform(BANDS, hs, d2, C)
    soft = gp.board_uniform(BANDS, hs, d2, C, gp.FLOW_RESISTANCE, mic_height=gp.PLATE_THICKNESS_FT)
    low, high = np.minimum(plate, soft) - 1e-9, np.maximum(plate, soft) + 1e-9
    for model in (gp.board_fresnel_strip, gp.board_fresnel_disc):
        level = model(BANDS, hs, d2, C)
        assert np.all((level >= low) & (level <= high))


def test_energy_blend_limits_and_order():
    plate, soft = np.array([6.0]), np.array([-10.0])
    assert gp._blend(np.array([1.0]), plate, soft, 'energy') == pytest.approx(6.0)
    assert gp._blend(np.array([0.0]), plate, soft, 'energy') == pytest.approx(-10.0)
    # Energy weighting sits above dB weighting between the two (the plate dominates sooner).
    half = np.array([0.5])
    assert gp._blend(half, plate, soft, 'energy')[0] > gp._blend(half, plate, soft, 'db')[0]
    with pytest.raises(ValueError):
        gp._blend(half, plate, soft, 'average')


def test_nmid_large_plate_is_the_rigid_plane():
    level = gp.board_nmid(np.array([500.0, 2000.0]), [100.0], [300.0], C, radius=500.0)
    assert np.allclose(level, 6.02, atol=0.5)


def test_nmid_known_flush_receiver_limit():
    # Documented limitation: as the plate vanishes under a flush microphone,
    # nMID tends to 1 + 2 Q_soft - Q_plate, not to the soft ground's 1 + Q_soft.
    f, hs, d2 = 1000.0, 30.0, 300.0
    p = gp.nmid_pressure_ratio(f, hs, d2, C, radius=1e-7)
    r2 = np.hypot(d2, hs)
    q_soft = gp.fa.spherical_reflection_coefficient(hs / r2, r2, f, C, gp.FLOW_RESISTANCE)
    q_plate = gp.fa.spherical_reflection_coefficient(hs / r2, r2, f, C, gp.RIGID_FLOW_RESISTANCE)
    assert p == pytest.approx(1 + 2 * q_soft - q_plate, abs=5e-3)   # radius 1e-7, not 0


def test_nmid_elevated_receiver_vanishing_strip_is_nearly_soft_ground():
    # Away from the ground the defect above is small: a vanishing strip leaves
    # (almost) the soft-ground field.
    f, hs, d2, hr = 1000.0, 100.0, 330.0, 5.0
    p = gp.nmid_pressure_ratio(f, hs, d2, C, radius=1e-7, mic_height=hr)
    r1, r2 = np.hypot(d2, hs - hr), np.hypot(d2, hs + hr)
    q = gp.fa.spherical_reflection_coefficient((hs + hr) / r2, r2, f, C, gp.FLOW_RESISTANCE)
    soft = 1 + r1 / r2 * q * np.exp(1j * 2 * np.pi * f / C * (r2 - r1))
    assert abs(p) == pytest.approx(abs(soft), rel=0.05)


def test_disc_mesh_covers_the_disc():
    x, y, area, bounds = gp.disc_mesh(gp.PLATE_RADIUS_FT, 0.05)
    assert area.sum() == pytest.approx(np.pi * gp.PLATE_RADIUS_FT ** 2, rel=1e-12)
    assert x[0] == 0.0 and y[0] == 0.0                   # the microphone's cell
    assert np.all(np.hypot(x, y) < gp.PLATE_RADIUS_FT)


def test_inverse_distance_integral_of_a_disc_about_its_center():
    # int dA / rho over a disc of radius a about its center = 2 pi a.
    assert gp._inverse_distance_integral(0.0, 0.0, (0.0, 0.3, 0.0, 2 * np.pi)) == pytest.approx(
        2 * np.pi * 0.3, rel=1e-6)


def test_disc_bem_plate_on_rigid_ground_changes_nothing():
    s = gp.disc_bem_scattered([500.0, 2000.0], [5.0, 45.0], [0.0, 90.0], C, flow_resistance=1e9,
                              cells_per_wavelength=4, min_cells_across=8)
    assert np.allclose(np.abs(s), 0.0, atol=1e-4)


def test_disc_bem_large_plate_doubles_pressure():
    # A plate much larger than the Fresnel zone, steep incidence: |(1 + Q)(1 + S)| -> 2.
    f, el = 1000.0, 45.0
    s = gp.disc_bem_scattered([f], [el], [0.0], C, radius=3.0, mic=(0.0, 0.0),
                              cells_per_wavelength=5)[0, 0, 0]
    r = 1000.0
    q = gp.fa.spherical_reflection_coefficient(np.sin(np.radians(el)), r, f, C, gp.FLOW_RESISTANCE)
    assert abs((1 + q) * (1 + s)) == pytest.approx(2.0, rel=0.05)


def test_disc_bem_converges_with_the_mesh():
    coarse = gp.disc_bem_scattered([2000.0], [20.0], [90.0], C, cells_per_wavelength=8)[0, 0, 0]
    fine = gp.disc_bem_scattered([2000.0], [20.0], [90.0], C, cells_per_wavelength=12)[0, 0, 0]
    assert 20 * np.log10(abs(1 + fine) / abs(1 + coarse)) == pytest.approx(0.0, abs=0.3)


def test_disc_bem_offset_microphone_symmetry():
    # ARP 4055: 3/4 radius, normal to the track (along y).  Mirroring x leaves the
    # microphone in place, so propagation at azimuth a and 180 - a must agree.
    s = gp.disc_bem_scattered([1500.0], [10.0], [30.0, 150.0], C, cells_per_wavelength=6)[0, 0]
    assert abs(1 + s[0]) == pytest.approx(abs(1 + s[1]), rel=0.02)


THIN = dict(thickness=0.0003 / 0.3048, edge_thickness=0.0003 / 0.3048, taper_length=0.0)


def test_raised_plate_mesh_normals_point_into_the_air():
    panels = gp.raised_plate_mesh(gp.PLATE_RADIUS_FT, gp.PLATE_THICKNESS_FT, gp.PLATE_EDGE_THICKNESS_FT,
                                  gp.PLATE_TAPER_LENGTH_FT, 0.05)
    for panel in panels:
        pts, w, nrm = gp._panel_points(panel, 1, 1)
        r = np.hypot(pts[0, 0], pts[0, 1])
        assert nrm[0, 2] >= 0.0                                   # never into the ground
        if panel[1] == panel[3] and panel[1] > 0:                 # flat top
            assert nrm[0, 2] == pytest.approx(1.0)
        if panel[0] == panel[2]:                                  # rim
            assert nrm[0, 0] * pts[0, 0] + nrm[0, 1] * pts[0, 1] == pytest.approx(r)


def test_raised_thin_plate_on_rigid_ground_doubles_pressure():
    pd, pr = gp.raised_plate_scattering([1000.0], [45.0], [90.0], C, flow_resistance=1e9,
                                        cells_per_wavelength=4, min_cells_across=10, **THIN)
    assert abs(pd[0, 0, 0] + pr[0, 0, 0]) == pytest.approx(2.0, rel=0.03)


def test_raised_plate_center_microphone_is_axisymmetric():
    pd, pr = gp.raised_plate_scattering([1000.0], [30.0], [0.0, 90.0], C, mic=(0.0, 0.0),
                                        cells_per_wavelength=4, min_cells_across=10)
    q = gp.fa.spherical_reflection_coefficient(0.5, 1000.0, 1000.0, C, gp.FLOW_RESISTANCE)
    assert abs(pd[0, 0, 0] + q * pr[0, 0, 0]) == pytest.approx(abs(pd[0, 0, 1] + q * pr[0, 0, 1]), rel=0.01)


def test_raised_thin_plate_matches_the_thin_disc_at_low_frequency():
    f, el = 500.0, 45.0
    pd, pr = gp.raised_plate_scattering([f], [el], [90.0], C, cells_per_wavelength=4,
                                        min_cells_across=12, **THIN)
    s = gp.disc_bem_scattered([f], [el], [90.0], C, cells_per_wavelength=4, min_cells_across=12)
    q = gp.fa.spherical_reflection_coefficient(np.sin(np.radians(el)), 1000.0, f, C, gp.FLOW_RESISTANCE)
    raised = 20 * np.log10(abs(pd[0, 0, 0] + q * pr[0, 0, 0]))
    thin = 20 * np.log10(abs((1 + q) * (1 + s[0, 0, 0])))
    assert raised == pytest.approx(thin, abs=0.5)


def _beta(f, sigma=100.0):
    return gp.surface_admittance(f, 'delany_bazley', sigma=sigma)


POLE_GEOMETRIES = [(500.0, 100.0), (30.0, 1500.0), (4.0, 300.0)]      # (hs, d2), ft


def test_pole_level_matches_ega():
    # Delany-Bazley ground, no turbulence and no microphone response: the
    # same band-averaged two-path level as flight_acoustics.ega.
    bands = np.array([50.0, 100.0, 315.0, 630.0, 1250.0, 2500.0, 5000.0, 10000.0])
    hs, d2 = np.array(POLE_GEOMETRIES).T
    level = gp.pole_level(bands, hs, d2, 4.0, C, 'delany_bazley', sigma=200.0)
    np.testing.assert_allclose(level, gp.pole_ground_effect(bands, hs, d2, 4.0, C, 200.0), rtol=0.0, atol=1e-9)


def test_pole_level_weights_the_paths_and_their_coherence():
    # |A_d|^2 + |A_r Q R1/R2|^2 + 2 A_d A_r |Q| R1/R2 cos(eta x + arg Q) sinc(mu x) coherence,
    # x = f (R2 - R1) / c, with the microphone's response A_d, A_r on each path and the
    # HARMONOISE coherence (lengths in meters).
    bands = np.array([125.0, 1000.0, 4000.0])
    hs, d2 = np.array(POLE_GEOMETRIES).T
    hr, gamma_t, ft = 4.0, 3e-6, 0.3048
    level = gp.pole_level(bands, hs, d2, hr, C, 'delany_bazley', gamma_t=gamma_t, response_direct=1.5,
                          response_reflected=-2.0, sigma=200.0)
    f = bands[:, None]
    r1, r2 = np.hypot(d2, hs - hr), np.hypot(d2, hs + hr)
    q = gp.fa.spherical_reflection_coefficient((hs + hr) / r2, r2, f, C, 200.0)
    a_d, a_r = 10 ** (1.5 / 20), 10 ** (-2.0 / 20)
    m, x = a_r * np.abs(q) * r1 / r2, f * (r2 - r1) / C
    k = 2 * np.pi * f / (C * ft)
    coherence = np.exp(-0.375 * 0.364 * k ** 2 * (hs * hr / (hs + hr) * ft) ** (5 / 3) * r1 * ft * gamma_t)
    energy = a_d ** 2 + m ** 2 + 2 * a_d * m * np.cos(6.325159 * x + np.angle(q)) * \
        np.sin(0.727477 * x) / (0.727477 * x) * coherence
    np.testing.assert_allclose(level, 10 * np.log10(energy), rtol=0.0, atol=1e-9)
    assert np.all(coherence < 1.0) and coherence.min() < 0.9          # the turbulence term is exercised


@pytest.mark.parametrize('rho, z', [(0.02, 0.0), (0.1, 0.05), (1.0, 0.3), (30.0, 4.0)])
def test_exact_green_satisfies_the_impedance_condition(rho, z):
    # dG/dz = -i k beta G on the plane, exactly (not approximately, as for Weyl-van der Pol).
    f = 1000.0; k = 2 * np.pi * f / C; beta = _beta(f)
    g, dg, _ = gp.exact_half_space_green(k, beta, rho, 0.0, z)
    assert complex(dg / (-1j * k * beta * g)) == pytest.approx(1.0, abs=1e-6)


def test_exact_green_rigid_limit_is_the_image_source():
    f = 1000.0; k = 2 * np.pi * f / C
    g, _, _ = gp.exact_half_space_green(k, 1e-10, 0.3, 0.01, 0.02)
    r1, r2 = np.hypot(0.3, 0.01), np.hypot(0.3, 0.03)
    assert complex(g) == pytest.approx(np.exp(1j * k * r1) / (4 * np.pi * r1) + np.exp(1j * k * r2) / (4 * np.pi * r2), rel=1e-6)


@pytest.mark.parametrize('f', [100.0, 1000.0, 4000.0])
def test_exact_green_matches_weyl_van_der_pol_at_long_range(f):
    k = 2 * np.pi * f / C; beta = _beta(f)
    rho, zs, zt = 500.0, 50.0, 4.0
    g, _, _ = gp.exact_half_space_green(k, beta, rho, zs, zt)
    r1, r2 = np.hypot(rho, zs - zt), np.hypot(rho, zs + zt)
    q_exact = (g - np.exp(1j * k * r1) / (4 * np.pi * r1)) / (np.exp(1j * k * r2) / (4 * np.pi * r2))
    q_weyl = gp.fa.spherical_reflection_coefficient((zs + zt) / r2, r2, f, C, 100.0)
    assert abs(complex(q_exact) - complex(q_weyl)) < 0.005


def test_image_integrals_converge_on_the_surface():
    f = 1000.0; k = 2 * np.pi * f / C; beta = _beta(f)
    rho, z = np.array([0.05, 0.3, 1.2]), np.zeros(3)
    a, _ = gp.image_integrals(k, beta, rho, z, n=10)
    b, _ = gp.image_integrals(k, beta, rho, z, n=24)
    assert np.all(np.isfinite(a)) and np.allclose(a, b, rtol=1e-5)


def _image_integrals_by_brute_force(k, beta, rho, z):
    # Directly in q, on Gauss-Legendre segments growing by 15 % from Z / 1000:
    # fine on every scale for rho < Z, where R_q stays about Z from zero.
    edges = [0.0]
    while edges[-1] < rho + 60.0 / k:
        edges.append(max(1e-3 * z, 1.15 * edges[-1]))
    x, w = np.polynomial.legendre.leggauss(40)
    a, b = np.array(edges[:-1])[:, None], np.array(edges[1:])[:, None]
    q, wq = (0.5 * (b - a) * x + 0.5 * (a + b)).ravel(), (0.5 * (b - a) * w).ravel()
    r = np.sqrt(rho ** 2 + (z + 1j * q) ** 2)
    g = np.exp(-k * beta * q) * np.exp(1j * k * r) / (4 * np.pi * r) * wq
    return g.sum(), (g * (1j * k - 1.0 / r) / r).sum()


@pytest.mark.parametrize('rho_over_z', [0.0, 1e-4, 0.1, 0.9])
@pytest.mark.parametrize('f, z', [(200.0, 1e-5), (200.0, 0.003), (1000.0, 0.05), (8000.0, 0.3)])
def test_image_integrals_below_rho_equal_z(f, z, rho_over_z):
    # A source above (or nearly above) the receiver: not 0 at rho = 0, and as
    # accurate for rho << Z as near rho = Z.
    k = 2 * np.pi * f / C; beta = _beta(f)
    big_i, big_j = gp.image_integrals(k, beta, rho_over_z * z, z)
    ref_i, ref_j = _image_integrals_by_brute_force(k, beta, rho_over_z * z, z)
    assert complex(big_i[0]) == pytest.approx(ref_i, rel=1e-9)
    assert complex(big_j[0]) == pytest.approx(ref_j, rel=1e-9)


def test_image_integral_straight_above_is_an_exponential_integral():
    # At rho = 0, R_q = Z + i q and I = -i e^{-i k beta Z} E1(-i k (1 + beta) Z) / (4 pi).
    from scipy.special import exp1
    for f, z in ((200.0, 0.003), (1000.0, 0.05), (4000.0, 2.0)):
        k = 2 * np.pi * f / C; beta = _beta(f)
        exact = -1j * np.exp(-1j * k * beta * z) * exp1(-1j * k * (1 + beta) * z) / (4 * np.pi)
        assert complex(gp.image_integrals(k, beta, 0.0, z)[0][0]) == pytest.approx(exact, rel=1e-10)


def test_image_integral_table_interpolates():
    f = 2000.0; k = 2 * np.pi * f / C; beta = _beta(f)
    table = gp.ImageIntegralTable(k, beta, 1.4, 0.06)
    rho, z = np.array([0.003, 0.02, 0.1, 0.4, 1.2]), np.array([0.0, 0.0005, 0.02, 0.05, 0.0])
    direct, _ = gp.image_integrals(k, beta, rho, z)
    assert np.allclose(table(rho, z)[0], direct, rtol=5e-3)


def test_disc_bem_table_is_read_at_its_own_sub_frequencies():
    # On rigid ground the plate scatters nothing, so the flush microphone reads
    # pressure doubling -- provided the table's two sub-frequencies are the ones
    # evaluated.
    table = gp.disc_bem_table([500.0], C, flow_resistance=1e9, sub_bands=2, elevations=[5.0, 45.0],
                              azimuths=[0.0, 90.0], cells_per_wavelength=4, min_cells_across=8)
    level = gp.board_disc_bem([500.0], [100.0], [10.0], C, flow_resistance=1e9, table=table)
    np.testing.assert_allclose(level, 20 * np.log10(2.0), atol=0.01)
    with pytest.raises(ValueError, match='2 sub-frequencies per band, not 5'):
        gp.board_disc_bem([500.0], [100.0], [10.0], C, flow_resistance=1e9, sub_bands=5, table=table)


# ---------------------------------------------------------------- emission times

def _line(t0=-2000.0, v=150.0, h=500.0):
    return lambda t: np.column_stack((t0 + v * np.asarray(t), np.zeros_like(t), np.full_like(t, h)))


def test_emission_times_match_the_closed_form_for_a_straight_pass():
    # x(t) = x0 + v t at height h, receiver at the origin: (t_r - t_e) c = |x(t_e)|
    x0, v, h = -2000.0, 150.0, 500.0
    t_r = np.linspace(2.0, 30.0, 15)
    t_e, x = gp.emission_times(_line(x0, v, h), t_r, [0.0, 0.0, 0.0], C)
    # (c^2 - v^2) t_e^2 - 2 (c^2 t_r + x0 v) t_e + c^2 t_r^2 - x0^2 - h^2 = 0, smaller root
    a = C ** 2 - v ** 2
    b = -2 * (C ** 2 * t_r + x0 * v)
    c = C ** 2 * t_r ** 2 - x0 ** 2 - h ** 2
    exact = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    assert np.allclose(t_e, exact, atol=1e-8)
    assert np.allclose(np.linalg.norm(x, axis=1), C * (t_r - t_e), rtol=1e-9)


def test_emission_times_agree_with_emission_geometry():
    t = np.linspace(0.0, 40.0, 40001)
    xyz = _line()(t)
    track = {'time': t, 'x': xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2]}
    mic = np.array([300.0, 200.0, 0.0])
    t_r = np.linspace(5.0, 30.0, 11)
    g = gp.emission_geometry(track, mic, t_r, C)
    _, x = gp.emission_times(_line(), t_r, mic, C)
    assert np.allclose(x[:, 2] - mic[2], g['source_height'], atol=1e-6)
    assert np.allclose(x[:, 0] - mic[0], g['source_dx'], atol=1e-3)


def test_one_pass_emission_time_is_short_by_about_m2_r():
    # the error a single pass leaves: the source placed ~M^2 R cos(phi) toward the mic
    t_r = np.array([3.0])
    t_1 = t_r - np.linalg.norm(_line()(t_r), axis=1) / C          # one pass from t_e = t_r
    t_e, _ = gp.emission_times(_line(), t_r, [0.0, 0.0, 0.0], C)
    assert abs(t_1[0] - t_e[0]) > 1e-3


def test_emission_times_refuse_a_supersonic_approach():
    with pytest.raises(RuntimeError):
        gp.emission_times(_line(v=2 * C), np.array([1.0]), [0.0, 0.0, 0.0], C, max_iter=20)


# ---------------------------------------------------------------- pole interference nulls

def test_path_difference_and_its_inverse():
    hr, d2 = 4.0, np.array([50.0, 150.0, 400.0])
    hs = np.array([30.0, 60.0, 90.0])
    dR = gp.path_difference(hs, d2, hr)
    assert np.allclose(dR, 2 * hr * hs / np.hypot(d2, hs), rtol=0.01)       # far field: 2 hr sin(el)
    assert np.allclose(gp.height_from_path_difference(dR, d2, hr), hs, rtol=1e-9)
    assert np.all(np.isnan(gp.height_from_path_difference([0.0, 8.0], [100.0, 100.0], hr)))


def test_null_frequencies():
    f = gp.null_frequencies(0.5, C, 0.0, 3)
    assert np.allclose(f, [C / 1.0, 3 * C / 1.0, 5 * C / 1.0])
    assert gp.null_frequencies(0.5, C, 0.2, 1)[0] < f[0]          # a positive phase lowers the nulls


def test_fit_two_path_recovers_the_path_difference():
    rng = np.random.default_rng(0)
    f = np.arange(50.0, 6000.0, 4.0)
    true = dict(offset_db=-5.0, amplitude=0.85, amplitude_rolloff=4000.0, dR=0.62, phase=0.05)
    y = gp.two_path_db(f, true['offset_db'], true['amplitude'], true['amplitude_rolloff'], true['dR'],
                       true['phase'], C) + rng.normal(0.0, 0.8, f.size)
    fit = gp.fit_two_path(f, y, dR_guess=0.45, sound_speed=C, receiver_height=4.0)
    assert fit['dR'] == pytest.approx(true['dR'], rel=0.01)
    assert fit['phase'] == pytest.approx(true['phase'], abs=0.1)
    assert fit['f1'] == pytest.approx(gp.null_frequencies(true['dR'], C, true['phase'], 1)[0], rel=0.02)
    held = gp.fit_two_path(f, y, dR_guess=0.45, sound_speed=C, receiver_height=4.0, fix_phase=true['phase'])
    assert held['dR'] == pytest.approx(true['dR'], rel=0.01) and held['phase_se'] == 0.0


def test_fit_two_path_fits_an_absolute_level():
    # A pole spectrum in dB re 20 uPa: the same shape 75 dB up fits to the same
    # path difference, with the offset carried along.
    rng = np.random.default_rng(0)
    f = np.arange(50.0, 6000.0, 4.0)
    y = gp.two_path_db(f, -5.0, 0.85, 4000.0, 0.62, 0.05, C) + rng.normal(0.0, 0.8, f.size)
    for fix_phase in (None, 0.05):
        low = gp.fit_two_path(f, y, dR_guess=0.45, sound_speed=C, receiver_height=4.0, fix_phase=fix_phase)
        high = gp.fit_two_path(f, y + 75.0, dR_guess=0.45, sound_speed=C, receiver_height=4.0,
                               fix_phase=fix_phase)
        assert high['dR'] == pytest.approx(low['dR'], rel=1e-6)
        assert high['offset_db'] == pytest.approx(low['offset_db'] + 75.0, abs=1e-4)


def test_reflection_phase_is_zero_over_rigid_ground_and_small_over_stiff():
    rigid = gp.reflection_phase([500.0, 2000.0], 100.0, 300.0, 4.0, C, flow_resistance=gp.RIGID_FLOW_RESISTANCE)
    assert np.allclose(rigid, 0.0, atol=1e-6)
    stiff = gp.reflection_phase(2000.0, 100.0, 300.0, 4.0, C, flow_resistance=2e4)
    assert 0.0 < abs(float(stiff)) < 0.2


def _disc_table(bands, mic_height=0.0, sub_bands=2):
    """A disc_bem_table stand-in with no scattering (S = 0)."""
    frequencies = np.sort((np.asarray(bands)[:, None] * gp.sub_band_factors(sub_bands)[None, :]).ravel())
    return dict(frequencies=frequencies, elevations=np.array([0.0, 90.0]), azimuths=np.array([0.0, 180.0]),
                S=np.zeros((frequencies.size, 2, 2), dtype=complex), mic_height=mic_height, ground=None,
                flow_resistance=gp.FLOW_RESISTANCE, sub_bands=sub_bands)


def test_board_disc_bem_is_nan_for_bands_the_table_does_not_hold():
    table = _disc_table(np.array([500.0, 1000.0]))
    level = gp.board_disc_bem([400.0, 500.0, 630.0, 1000.0, 1250.0], [100.0], [200.0], C, table=table)[:, 0]
    assert np.all(np.isfinite(level[[1, 3]])) and np.all(np.isnan(level[[0, 2, 4]]))


def test_board_disc_bem_takes_q_at_the_microphone_height():
    # With S = 0 an inverted microphone h above the ground reads the two-path
    # field 1 + Q e^{2ikh sin(el)}, Q at the image geometry of a receiver at h.
    h, hs, d2, band = gp.INVERTED_MIC_HEIGHT_FT, 0.5, 20.0, 2000.0
    level = gp.board_disc_bem([band], [hs], [d2], C, table=_disc_table(np.array([band]), mic_height=h))[0, 0]
    r2 = np.hypot(d2, hs + h)
    energy = 0.0
    for f in band * gp.sub_band_factors(2):
        q = gp.fa.spherical_reflection_coefficient((hs + h) / r2, r2, f, C, gp.FLOW_RESISTANCE)
        energy += abs(1 + q * np.exp(2j * 2 * np.pi * f / C * h * np.sin(np.arctan2(hs, d2)))) ** 2
    assert level == pytest.approx(10 * np.log10(energy / 2), abs=1e-9)



def test_table_frames_interpolate_bilinearly_and_wrap_in_azimuth():
    # Values linear in elevation and, near 0 deg, in azimuth (350 deg -> 350,
    # 0 deg -> 360): the bilinear lookup is exact, across 350 -> 360 = 0 too.
    table = _disc_table(np.array([1000.0]))
    table['elevations'] = np.array([0.0, 10.0, 90.0])
    table['azimuths'] = np.arange(0.0, 360.0, 10.0)
    el, az = np.meshgrid(table['elevations'], table['azimuths'], indexing='ij')
    wrapped = np.where(az == 0.0, 360.0, np.where(az > 300.0, az, 0.0))
    table['S'] = np.broadcast_to(el + 1j * wrapped, (2,) + el.shape)
    hs, d2 = np.tan(np.radians(5.0)) * 100.0, 100.0
    toward = np.radians([355.0, 15.0, 5.0])                 # propagation azimuths
    f, _, _, elevation, offsets, values = gp.table_frames(
        table, ('S',), [1000.0, 2000.0], [hs] * 3, [d2] * 3, source_dx=-np.cos(toward),
        source_dy=-np.sin(toward))
    assert values.shape == (2, 1, 2, 3) and np.allclose(elevation, 5.0)
    np.testing.assert_allclose(values[:, 0, 0], np.broadcast_to(5.0 + 1j * np.array([355.0, 0.0, 180.0]),
                                                                (2, 3)), atol=1e-9)
    assert np.all(np.isnan(values[:, 0, 1]))               # no 2 kHz rows
    mirrored = gp.table_frames(table, ('S',), [1000.0], [hs] * 3, [d2] * 3, source_dx=-np.cos(toward),
                               source_dy=-np.sin(toward), mirror_y=True)[-1]
    np.testing.assert_allclose(mirrored[0, 0, 0].imag, [180.0, 345.0, 355.0], atol=1e-9)


def test_board_disc_bem_takes_its_ground_from_the_table():
    # Q from a ground other than the one S was computed over would mix two grounds.
    table = dict(_disc_table(np.array([500.0])), flow_resistance=1e9)
    level = gp.board_disc_bem([500.0], [100.0], [200.0], C, table=table)
    assert np.array_equal(level, gp.board_disc_bem([500.0], [100.0], [200.0], C, flow_resistance=1e9, table=table))
    assert not np.allclose(level, gp.board_disc_bem([500.0], [100.0], [200.0], C,
                                                    table=_disc_table(np.array([500.0]))))
    with pytest.raises(ValueError, match="differs from the table"):
        gp.board_disc_bem([500.0], [100.0], [200.0], C, flow_resistance=gp.FLOW_RESISTANCE, table=table)
    with pytest.raises(ValueError, match="needs a table"):
        gp.board_disc_bem([500.0], [100.0], [200.0], C)


def test_board_disc_bem_takes_a_ground_model_from_the_table():
    # A ground model sets Q whatever flow resistance the table carries (tables
    # built before it was None hold the unused default); no flow resistance,
    # the site's or that default, may be passed for it.
    ground = dict(model='variable_porosity', sigma_e=200.0, alpha_e=0.0)
    table = dict(_disc_table(np.array([500.0])), ground=ground)
    level = gp.board_disc_bem([500.0], [100.0], [200.0], C, table=table)
    r2 = np.hypot(200.0, 100.0)
    energy = 0.0
    for f in 500.0 * gp.sub_band_factors(2):
        beta = gp.surface_admittance(f, **ground, sound_speed_mps=C * 0.3048)
        energy += abs(1 + gp.fa.spherical_reflection_coefficient(100.0 / r2, r2, f, C, None, admittance=beta)) ** 2
    assert level[0, 0] == pytest.approx(10 * np.log10(energy / 2), abs=1e-9)
    for sigma in (200.0, gp.FLOW_RESISTANCE):
        with pytest.raises(ValueError, match="flow_resistance does not apply"):
            gp.board_disc_bem([500.0], [100.0], [200.0], C, flow_resistance=sigma, table=table)
    built = gp.disc_bem_table(np.array([200.0]), C, sub_bands=1, elevations=np.array([10.0]),
                              azimuths=np.array([0.0]), ground=ground)
    assert built['flow_resistance'] is None and built['ground'] == ground
