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


def test_centre_microphone_is_independent_of_azimuth():
    pd, pr = ab.scattering([1500.0], [20.0], [0.0, 77.0, 200.0], C, mic=(0.0, 0.0))
    q = 0.3 - 0.2j
    p = np.abs(pd[0, 0] + q * pr[0, 0])
    assert np.allclose(p, p[0], rtol=1e-6)


def test_offset_microphone_mirror_symmetry():
    # Microphone on +y: azimuth a and 180 - a are mirror images in x.
    pd, pr = ab.scattering([2000.0], [15.0], [30.0, 150.0], C)
    assert np.allclose(pd[0, 0, 0], pd[0, 0, 1], rtol=1e-6) and np.allclose(pr[0, 0, 0], pr[0, 0, 1], rtol=1e-6)


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
