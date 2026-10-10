import numpy as np
import pytest

import axisymmetric_bem as ab
import ground_plane as gp

C = 1125.0
THIN = dict(thickness=0.0003 / 0.3048, edge_thickness=0.0003 / 0.3048, taper_length=0.0)


def test_generator_normals_point_into_the_air():
    segs, flat = ab.plate_generator(segment=0.02)
    dr, dz = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    nr, nz = -dz, dr
    assert np.all(nz >= 0.0) and np.all(nr[~flat] >= 0.0)
    assert np.allclose(segs[flat, 1], gp.PLATE_THICKNESS_FT)
    assert segs[-1, 3] == 0.0                          # the rim reaches the ground


def test_thin_plate_on_rigid_ground_doubles_pressure():
    pd, pr = ab.scattering([500.0, 2000.0], [10.0, 45.0], [90.0, 270.0], C, flow_resistance=1e9, **THIN)
    assert np.allclose(np.abs(pd + pr), 2.0, atol=0.02)


def test_center_microphone_is_independent_of_azimuth():
    pd, pr = ab.scattering([1500.0], [20.0], [0.0, 77.0, 200.0], C, mic=(0.0, 0.0))
    q = 0.3 - 0.2j
    p = np.abs(pd[0, 0] + q * pr[0, 0])
    assert np.allclose(p, p[0], rtol=1e-6)


def test_offset_microphone_mirror_symmetry():
    # Microphone on +y: azimuth a and 180 - a are mirror images in x.
    pd, pr = ab.scattering([2000.0], [15.0], [30.0, 150.0], C)
    assert np.allclose(pd[0, 0, 0], pd[0, 0, 1], rtol=1e-6) and np.allclose(pr[0, 0, 0], pr[0, 0, 1], rtol=1e-6)


def test_flush_microphone_must_be_on_the_flat_top():
    # Over the taper or beyond the rim a flush microphone would sit in the air
    # at the top's height and still be doubled as a surface point.
    r_taper = gp.PLATE_RADIUS_FT - gp.PLATE_TAPER_LENGTH_FT
    for r in (r_taper, gp.PLATE_RADIUS_FT - 0.5 * gp.PLATE_TAPER_LENGTH_FT, 1.2 * gp.PLATE_RADIUS_FT):
        with pytest.raises(ValueError, match='flat top'):
            ab.scattering([1000.0], [20.0], [0.0], C, mic=(0.0, r))
        with pytest.raises(ValueError, match='flat top'):
            ab.table(np.array([1000.0]), C, sub_bands=1, elevations=[20.0], azimuths=[0.0], mic=(r, 0.0))
    # Raised (inverted) above the plate it is in the air anywhere.
    pd, pr = ab.scattering([1000.0], [20.0], [0.0], C, mic=(0.0, r_taper), mic_height=gp.INVERTED_MIC_HEIGHT_FT)
    assert np.isfinite(pd).all() and np.isfinite(pr).all()


def test_matches_the_3d_surface_model():
    f, el = 500.0, 45.0
    pd, pr = ab.scattering([f], [el], [90.0], C)
    pd3, pr3 = gp.raised_plate_scattering([f], [el], [90.0], C, cells_per_wavelength=4, min_cells_across=16)
    q = gp.fa.spherical_reflection_coefficient(np.sin(np.radians(el)), 1000.0, f, C, gp.FLOW_RESISTANCE)
    a = 20 * np.log10(abs(pd[0, 0, 0] + q * pr[0, 0, 0]))
    b = 20 * np.log10(abs(pd3[0, 0, 0] + q * pr3[0, 0, 0]))
    assert a == pytest.approx(b, abs=0.25)


def test_converges_with_the_generator_mesh():
    kw = dict(frequencies=[2000.0], elevations=[10.0], azimuths=[90.0], sound_speed=C)
    coarse = ab.scattering(**kw, segments_per_wavelength=10, max_segment=0.02)
    fine = ab.scattering(**kw, segments_per_wavelength=30, max_segment=0.007)
    q = 0.2 + 0.1j
    assert abs(coarse[0] + q * coarse[1])[0, 0, 0] == pytest.approx(abs(fine[0] + q * fine[1])[0, 0, 0], rel=0.01)


def test_netcdf_export_round_trips(tmp_path):
    # The NICE-OPS --plate_table format: meters, the ground named, P_d and P_r
    # laid out band x sub x elevation x azimuth.
    from netCDF4 import Dataset
    ground = dict(model='variable_porosity', sigma_e=200.0, alpha_e=0.0)
    table = ab.table(np.array([500.0, 1000.0]), C, flow_resistance=200.0, ground=ground, sub_bands=2,
                     elevations=np.array([5.0, 30.0]), azimuths=np.array([0.0, 180.0]))
    path = tmp_path / 'plate.nc'
    ab.write_netcdf(str(path), table)
    with Dataset(path) as nc:
        assert nc.ground_model == 'variable_porosity' and nc.flow_resistance == 200.0
        assert nc.thickness_m == pytest.approx(0.008) and nc.radius_m == pytest.approx(0.2)
        assert np.hypot(nc.mic_x_m, nc.mic_y_m) == pytest.approx(0.15)
        assert nc.sound_speed_mps == pytest.approx(C * 0.3048)
        assert np.allclose(nc['frequency'][:].ravel(), table['frequencies'])
        p_r = nc['P_r_real'][:] + 1j * nc['P_r_imag'][:]
        assert p_r.shape == (2, 2, 2, 2)
        assert np.allclose(p_r.reshape(4, 2, 2), table['P_r'])


def test_inverted_microphone_over_a_vanishing_plate_is_two_paths():
    # 7 mm above a 0.01 mm plate on rigid ground: direct plus image about the
    # ground, including the 10 kHz null.  (A 0.3 mm plate is not thin enough
    # here: 2 k t is 0.11 rad at 10 kHz, visible near a null.)
    t = 0.00001 / 0.3048
    h = gp.INVERTED_MIC_HEIGHT_FT
    f = np.array([2500.0, 10000.0])
    el = np.array([30.0, 80.0])
    pd, pr = ab.scattering(f, el, [90.0], C, flow_resistance=1e9, mic_height=h,
                           thickness=t, edge_thickness=t, taper_length=0.0)
    k = 2 * np.pi * f / C
    exact = np.abs(1 + np.exp(2j * k[:, None] * (t + h) * np.sin(np.radians(el))[None, :]))
    assert np.allclose(20 * np.log10(np.abs(pd + pr)[:, :, 0]), 20 * np.log10(exact), atol=0.1)


def test_board_level_averages_over_the_tables_own_sub_frequencies():
    # One band at two sub-frequencies, |P_d|^2 = 1 at the first and 4 at the
    # second, P_r = 0: the band average is 2.5 whatever the ground.
    p_d = np.array([1.0, 2.0])[:, None, None] * np.ones((2, 2, 2), dtype=complex)
    table = dict(frequencies=1000.0 * 2.0 ** (np.array([-1.0, 1.0]) / 12.0),
                 elevations=np.array([0.0, 90.0]), azimuths=np.array([0.0, 180.0]), P_d=p_d,
                 P_r=np.zeros_like(p_d), thickness=gp.PLATE_THICKNESS_FT, flow_resistance=gp.FLOW_RESISTANCE,
                 ground=None, sub_bands=2)
    np.testing.assert_allclose(ab.board_level([1000.0], [100.0], [200.0], C, table), 10 * np.log10(2.5))
    with pytest.raises(ValueError, match='2 sub-frequencies per band, not 5'):
        ab.board_level([1000.0], [100.0], [200.0], C, table, sub_bands=5)


def test_board_level_takes_q_at_the_plates_top():
    # P_d = 0, P_r = 1: the level is |Q|^2, Q at the image geometry of the
    # plate's top even for an inverted microphone's table.
    bands = np.array([2000.0])
    frequencies = bands * gp.sub_band_factors(2)
    ones = np.ones((2, 2, 2), dtype=complex)
    t, hs, d2 = gp.PLATE_THICKNESS_FT, 0.5, 20.0
    table = dict(frequencies=frequencies, elevations=np.array([0.0, 90.0]), azimuths=np.array([0.0, 180.0]),
                 P_d=0.0 * ones, P_r=ones, thickness=t, mic_height=gp.INVERTED_MIC_HEIGHT_FT,
                 flow_resistance=gp.FLOW_RESISTANCE, ground=None, sub_bands=2)
    r2 = np.hypot(d2, hs + t)
    q = gp.fa.spherical_reflection_coefficient((hs + t) / r2, r2, frequencies, C, gp.FLOW_RESISTANCE)
    level = ab.board_level(bands, [hs], [d2], C, table)[0, 0]
    assert level == pytest.approx(10 * np.log10(np.mean(np.abs(q) ** 2)), abs=1e-9)


def _incident(f, el, az, r, z):
    """The incident plane wave at (x=0, y=r, z), for a source at elevation el, azimuth az."""
    k = 2 * np.pi * f / C
    return np.exp(1j * k * (np.cos(np.radians(el)) * r * np.sin(np.radians(az)) - np.sin(np.radians(el)) * z))


def test_field_matches_scattering_at_the_microphone():
    """field() off the surface reproduces scattering() at an inverted mic 7 mm above the plate."""
    f, el, az = 3000.0, 25.0, 40.0
    gap = gp.INVERTED_MIC_HEIGHT_FT
    rm = gp.PLATE_MIC_OFFSET_FT
    pd, pr = ab.scattering([f], [el], [az], C, mic=(0.0, rm), mic_height=gap)
    Pd, Pr = ab.field(f, el, az, C, [[0.0, rm, gp.PLATE_THICKNESS_FT + gap]])
    inc = _incident(f, el, az, rm, gp.PLATE_THICKNESS_FT + gap)
    assert Pd[0] / inc == pytest.approx(pd[0, 0, 0], rel=1e-5)
    assert Pr[0] / inc == pytest.approx(pr[0, 0, 0], rel=1e-5)


def test_field_on_the_surface_is_the_surface_value():
    # On the plate p = 2 (p_inc + K p), not the principal value p_inc + K p =
    # p / 2: field() on the top and on the taper reproduces scattering()'s
    # flush microphone there, and so does a point a nanometer off the top.
    f, el, az = 3000.0, 25.0, 40.0
    rm, t = gp.PLATE_MIC_OFFSET_FT, gp.PLATE_THICKNESS_FT
    segs, _ = ab.plate_generator()
    taper = segs[(segs[:, 1] > segs[:, 3]) & (segs[:, 0] < segs[:, 2])][3]
    r_taper, z_taper = 0.5 * (taper[0] + taper[2]), 0.5 * (taper[1] + taper[3])
    Pd, Pr = ab.field(f, el, az, C, [[0.0, rm, t], [0.0, rm, t + 3e-9], [0.0, r_taper, z_taper]])
    for i, (r, z, kw) in enumerate(((rm, t, {}), (rm, t, {}), (r_taper, z_taper, dict(mic_rz=(r_taper, z_taper))))):
        pd, pr = ab.scattering([f], [el], [az], C, mic=(0.0, r), **kw)
        inc = _incident(f, el, az, r, z)
        assert Pd[i] / inc == pytest.approx(pd[0, 0, 0], rel=1e-4)
        assert Pr[i] / inc == pytest.approx(pr[0, 0, 0], rel=1e-4)


def test_field_inside_the_plate_is_nan_and_far_field_tends_to_the_bare_ground():
    f, el = 2000.0, 30.0
    pts = [[0.0, 0.0, 0.5 * THIN["thickness"]], [0.0, 0.0, 30.0]]
    Pd, Pr = ab.field(f, el, 0.0, C, pts, **THIN)
    assert np.isnan(Pd[0]) and np.isnan(Pr[0])
    k = 2 * np.pi * f / C
    z = 30.0
    # 30 ft up, the plate's scattered wave is small next to the unit plane waves
    assert abs(Pd[1] - np.exp(-1j * k * np.sin(np.radians(el)) * z)) < 0.05
    assert abs(Pr[1] - np.exp(1j * k * np.sin(np.radians(el)) * z)) < 0.05


def test_board_level_is_nan_for_bands_the_table_does_not_hold():
    # A table for 500 and 1000 Hz: 400 (below), 630 and 800 (between) and
    # 1250 Hz (above) have no rows, and must not borrow the nearest one's.
    bands = np.array([500.0, 1000.0])
    frequencies = np.sort((bands[:, None] * gp.sub_band_factors(2)[None, :]).ravel())
    p_d = np.ones((frequencies.size, 2, 2), dtype=complex)
    table = dict(frequencies=frequencies, elevations=np.array([0.0, 90.0]), azimuths=np.array([0.0, 180.0]),
                 P_d=p_d, P_r=np.zeros_like(p_d), thickness=gp.PLATE_THICKNESS_FT,
                 flow_resistance=gp.FLOW_RESISTANCE, ground=None, sub_bands=2)
    level = ab.board_level([400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0], [100.0], [200.0], C, table)[:, 0]
    np.testing.assert_allclose(level[[1, 4]], 0.0, atol=1e-12)
    assert np.all(np.isnan(level[[0, 2, 3, 5]]))


def test_netcdf_export_orders_the_bands(tmp_path):
    # Bands given high to low: each band_center must still head its own
    # sub-frequencies and their P_d, P_r.
    from netCDF4 import Dataset
    table = ab.table(np.array([1000.0, 500.0]), C, sub_bands=2, elevations=np.array([5.0, 30.0]),
                     azimuths=np.array([0.0, 180.0]))
    path = tmp_path / 'plate.nc'
    ab.write_netcdf(str(path), table)
    with Dataset(path) as nc:
        centers, frequency = nc['band_center'][:], nc['frequency'][:]
        p_d = nc['P_d_real'][:] + 1j * nc['P_d_imag'][:]
    np.testing.assert_array_equal(centers, [500.0, 1000.0])
    np.testing.assert_allclose(frequency, centers[:, None] * gp.sub_band_factors(2)[None, :], rtol=1e-12)
    for i in range(2):
        for j in range(2):
            row = np.argmin(np.abs(table['frequencies'] - frequency[i, j]))
            np.testing.assert_array_equal(p_d[i, j], table['P_d'][row])
    table['bands'] = np.array([500.0, 900.0])
    with pytest.raises(ValueError, match="sub-frequencies"):
        ab.write_netcdf(str(tmp_path / 'bad.nc'), table)


def test_field_over_the_taper_is_air():
    # Halfway along the taper the surface is midway between the 8 mm top and
    # the 2.5 mm rim: a point 1 mm above it is in the air, 1 mm below it inside.
    r = gp.PLATE_RADIUS_FT - 0.5 * gp.PLATE_TAPER_LENGTH_FT
    surface = 0.5 * (gp.PLATE_THICKNESS_FT + gp.PLATE_EDGE_THICKNESS_FT)
    mm = 0.001 / 0.3048
    Pd, Pr = ab.field(2000.0, 30.0, 0.0, C, [[r, 0.0, surface + mm], [r, 0.0, surface - mm],
                                            [gp.PLATE_RADIUS_FT + mm, 0.0, gp.PLATE_THICKNESS_FT - mm]])
    assert np.isfinite(Pd[0]) and np.isfinite(Pr[0])
    assert np.isnan(Pd[1]) and np.isnan(Pr[1])
    assert np.isfinite(Pd[2])                        # beside the rim
    # Near-surface pressure over a rigid plate on soft ground: of order the doubled wave.
    assert 0.5 < abs(Pd[0]) < 3.0


def test_bessel_orders_match_scipy():
    # Miller's recurrence against scipy's jv over the incident fields' range
    # (orders to k a + 10 at 11 kHz, arguments to k a), tiny arguments included.
    from scipy.special import j0, j1, jv
    x = np.concatenate(([0.0, 1e-300, 1e-31, 1e-29, 1e-12], np.linspace(1e-3, 40.0, 4001)))
    ours = ab._bessel_j_orders(50, x, j0(x), j1(x))
    exact = jv(np.arange(51)[:, None], x[None, :])
    assert np.all(np.isfinite(ours))
    np.testing.assert_allclose(ours, exact, rtol=0.0, atol=5e-15)
    assert ours[2, 2] == pytest.approx(1e-31 ** 2 / 8.0, rel=1e-12)   # the series branch: J_2 = x^2/8
