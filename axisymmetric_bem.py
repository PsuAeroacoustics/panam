"""Axisymmetric (body-of-revolution) BEM for a rigid plate lying on a locally reacting ground.

The plate and the ground's Green's function are both unchanged by rotation about
the plate's axis, so the surface pressure separates into azimuthal modes,
p(s, phi) = sum_m p_m(s) e^{i m phi}, each solved on the plate's generating curve
(the cross-section: flat top, taper, rim) alone.  A plane wave splits the same
way (Jacobi-Anger, e^{i kappa r cos(phi - az)} = sum_m i^m J_m(kappa r)
e^{i m (phi - az)}), so one solve per mode and elevation serves every azimuth,
and a microphone anywhere on the plate is a sum over modes.  This replaces the
3-D surface mesh of :func:`ground_plane.raised_plate_scattering` (same
equation, same exact Green's function) at a small fraction of the cost; the
numerics are compiled with Numba and run in parallel.

Units follow ground_plane: lengths in feet, speeds in ft/s.
"""
from types import SimpleNamespace

import numpy as np
from numba import njit, prange
from scipy.special import j0, j1

import ground_plane as gp


# --------------------------------------------------------------------------
# Compiled pieces
# --------------------------------------------------------------------------

@njit(cache=True, fastmath=False)
def _table_bilinear(rho, z, u0, du, nu, v0, dv, nv, offset, i_red, j_red):
    """Bilinear lookup of the reduced image integrals (see gp.ImageIntegralTable)."""
    u = np.log(rho + offset)
    v = np.log(z + offset)
    fu = (min(max(u, u0), u0 + du * (nu - 1)) - u0) / du
    fv = (min(max(v, v0), v0 + dv * (nv - 1)) - v0) / dv
    i0 = min(int(fu), nu - 2)
    j0 = min(int(fv), nv - 2)
    tu = fu - i0
    tv = fv - j0
    w00 = (1 - tu) * (1 - tv); w10 = tu * (1 - tv); w01 = (1 - tu) * tv; w11 = tu * tv
    ir = w00 * i_red[i0, j0] + w10 * i_red[i0 + 1, j0] + w01 * i_red[i0, j0 + 1] + w11 * i_red[i0 + 1, j0 + 1]
    jr = w00 * j_red[i0, j0] + w10 * j_red[i0 + 1, j0] + w01 * j_red[i0, j0 + 1] + w11 * j_red[i0 + 1, j0 + 1]
    return ir, jr


@njit(cache=True)
def _dgdn_cs(xr, xz, yr, yz, c, s, nr, nz, k, beta, tab, skip_direct):
    """dG/dn_y for target x = (xr, 0, xz) and source y = (yr c, yr s, yz), normal (nr, nz) at y; c, s = cos, sin phi."""
    u0, du, nu, v0, dv, nv, offset, i_red, j_red = tab
    dx = yr * c - xr
    dy = yr * s
    dz = yz - xz
    n_dot_h = nr * (yr - xr * c)                 # n_h . d_h
    out = 0.0 + 0.0j
    if not skip_direct:
        r1 = np.sqrt(dx * dx + dy * dy + dz * dz)
        g1 = np.exp(1j * k * r1) / (4 * np.pi * r1)
        out += g1 * (1j * k - 1.0 / r1) / r1 * (n_dot_h + nz * dz)
    rho = np.sqrt(dx * dx + dy * dy)
    zs = yz + xz
    r2 = np.sqrt(rho * rho + zs * zs)
    e2 = np.exp(1j * k * r2)
    g2 = e2 / (4 * np.pi * r2)
    # I and J by bilinear lookup in the table (gp.ImageIntegralTable), sharing e^{ikR2} unless R2 is below the table's offset.
    ir, jr = _table_bilinear(rho, zs, u0, du, nu, v0, dv, nv, offset, i_red, j_red)
    r2c = max(r2, offset)
    ph = e2 if r2c == r2 else np.exp(1j * k * r2c)
    big_i, big_j = ir * ph / (4 * np.pi * r2c), jr * ph / (4 * np.pi * r2c * r2c)
    dimg_z = g2 * (1j * k - 1.0 / r2) * zs / r2 - 2j * k * beta * g2 + 2j * k * k * beta * beta * big_i
    dimg_h = g2 * (1j * k - 1.0 / r2) / r2 - 2 * k * beta * big_j
    return out + nz * dimg_z + n_dot_h * dimg_h


@njit(cache=True)
def _graded_edges(center, scale, lo, hi, n_uniform, ratio):
    """Breakpoints on [lo, hi]: geometric from ``center`` (step ``scale`` x ratio^j), plus uniform ones."""
    edges = [lo, hi]
    for i in range(1, n_uniform):
        edges.append(lo + (hi - lo) * i / n_uniform)
    step = scale
    while step < (hi - lo):
        a = center - step
        b = center + step
        if lo < a < hi:
            edges.append(a)
        if lo < b < hi:
            edges.append(b)
        step *= ratio
    if lo < center < hi:
        edges.append(center)
    arr = np.array(edges)
    arr.sort()
    return arr


@njit(cache=True)
def _modal_integral(xr, xz, seg, m_max, k, beta, tab, gx, gw, n_phi_uniform, same_plane):
    """int over segment x [0, 2 pi] of dG/dn e^{i m phi} r ds, for m = 0..m_max (kernel even in phi)."""
    ra, za, rb, zb = seg[0], seg[1], seg[2], seg[3]
    length = np.sqrt((rb - ra) ** 2 + (zb - za) ** 2)
    nr = -(zb - za) / length
    nz = (rb - ra) / length
    # Distance scale of the near-singularity: to the target and to its image.
    tr, tz = rb - ra, zb - za
    t_star = ((xr - ra) * tr + (xz - za) * tz) / (length * length)
    t_star = min(max(t_star, 0.0), 1.0)
    pr, pz = ra + t_star * tr, za + t_star * tz
    d_direct = np.sqrt((pr - xr) ** 2 + (pz - xz) ** 2)
    t_img = ((xr - ra) * tr + (-xz - za) * tz) / (length * length)
    t_img = min(max(t_img, 0.0), 1.0)
    d_image = np.sqrt((ra + t_img * tr - xr) ** 2 + (za + t_img * tz + xz) ** 2)
    delta = min(d_direct, d_image)
    t_near = t_star if d_direct <= d_image else t_img
    near = delta < 3.0 * length
    if near:
        u_edges = _graded_edges(t_near, max(delta, 1e-6 * length) / length * 0.5, 0.0, 1.0, 2, 3.0)
        r_ref = max(xr, 1e-9)
        phi_edges = _graded_edges(0.0, max(delta, 1e-6 * length) / r_ref * 0.5, 0.0, np.pi, n_phi_uniform, 3.0)
    else:
        u_edges = np.array([0.0, 1.0])
        phi_edges = np.linspace(0.0, np.pi, n_phi_uniform + 1)
    # The phi nodes are the same at every node along the segment: sum over
    # the segment first, so the cos(m phi) recurrence runs once per phi node.
    ng = gx.size
    yr_n = np.empty((u_edges.size - 1) * ng)
    yz_n = np.empty_like(yr_n)
    jac_n = np.empty_like(yr_n)
    n_u = 0
    for a in range(u_edges.size - 1):
        u_lo, u_hi = u_edges[a], u_edges[a + 1]
        if u_hi <= u_lo:
            continue
        for i in range(ng):
            u = u_lo + (u_hi - u_lo) * gx[i]
            wu = (u_hi - u_lo) * gw[i]
            yr_n[n_u] = ra + u * tr
            yz_n[n_u] = za + u * tz
            jac_n[n_u] = yr_n[n_u] * length * wu
            n_u += 1
    out = np.zeros(m_max + 1, dtype=np.complex128)
    for b in range(phi_edges.size - 1):
        p_lo, p_hi = phi_edges[b], phi_edges[b + 1]
        if p_hi <= p_lo:
            continue
        for j in range(ng):
            phi = p_lo + (p_hi - p_lo) * gx[j]
            wp = (p_hi - p_lo) * gw[j]
            c1 = np.cos(phi)
            s1 = np.sin(phi)
            val = 0.0 + 0.0j
            for i in range(n_u):
                val += _dgdn_cs(xr, xz, yr_n[i], yz_n[i], c1, s1, nr, nz, k, beta, tab, same_plane) * jac_n[i]
            val *= wp * 2.0
            # cos(m phi) by recurrence
            cm_prev = 1.0
            cm = c1
            out[0] += val
            if m_max >= 1:
                out[1] += val * cm
            for m in range(2, m_max + 1):
                cm_next = 2.0 * c1 * cm - cm_prev
                cm_prev = cm
                cm = cm_next
                out[m] += val * cm
    return out


@njit(parallel=True, cache=True)
def _assemble(targets, segs, flat_top, m_max, k, beta, tab, gx, gw, n_phi_uniform):
    """Modal kernels K[m, i, j] for all collocation targets i and source segments j."""
    n_t = targets.shape[0]
    n_s = segs.shape[0]
    out = np.zeros((m_max + 1, n_t, n_s), dtype=np.complex128)
    # Over (target, segment) pairs, so a single target (the microphone) is
    # parallel too
    for p in prange(n_t * n_s):
        i = p // n_s
        j = p - i * n_s
        same_plane = flat_top[j] and abs(targets[i, 1] - segs[j, 1]) < 1e-12
        vals = _modal_integral(targets[i, 0], targets[i, 1], segs[j], m_max, k, beta, tab, gx, gw,
                               n_phi_uniform, same_plane)
        for m in range(m_max + 1):
            out[m, i, j] = vals[m]
    return out


@njit(parallel=True, cache=True)
def _bessel_j_orders(m_max, x, j0x, j1x):
    """J_m(x), m = 0..m_max, for each x >= 0: (m_max + 1, x.size).

    Miller's downward recurrence J_{m-1} = (2m/x) J_m - J_{m+1}, started well
    above both m_max and x (stable downward for J), rescaled before it
    overflows, and normalized to whichever of J_0 = ``j0x`` and J_1 = ``j1x``
    is the larger.  All orders cost about what scipy's jv takes for two.
    """
    out = np.zeros((m_max + 1, x.size))
    for p in prange(x.size):
        xv = x[p]
        if xv < 1e-30:
            # Leading-order series, exact to (x/2)^2 / (m + 1) (2 m / x would overflow).
            out[0, p] = j0x[p]
            for mm in range(1, m_max + 1):
                out[mm, p] = j1x[p] if mm == 1 else out[mm - 1, p] * xv / (2.0 * mm)
            continue
        top = max(m_max, int(np.ceil(xv)))
        b_next, b = 0.0, 1.0                    # unnormalized J_{j+1}, J_j
        for j in range(top + 30 + int(np.sqrt(160.0 * top)), 0, -1):
            b_next, b = b, 2.0 * j / xv * b - b_next
            if j - 1 <= m_max:
                out[j - 1, p] = b
            if abs(b) > 1e200:
                b *= 1e-200
                b_next *= 1e-200
                for mm in range(j - 1, m_max + 1):
                    out[mm, p] *= 1e-200
        scale = j0x[p] / b if abs(j0x[p]) >= abs(j1x[p]) else j1x[p] / b_next
        for mm in range(m_max + 1):
            out[mm, p] *= scale
    return out


# --------------------------------------------------------------------------
# Geometry and driver
# --------------------------------------------------------------------------

def plate_generator(radius=gp.PLATE_RADIUS_FT, thickness=gp.PLATE_THICKNESS_FT,
                    edge_thickness=gp.PLATE_EDGE_THICKNESS_FT, taper_length=gp.PLATE_TAPER_LENGTH_FT,
                    segment=0.02, edge_grading=4):
    """Segments (r0, z0, r1, z1) of the plate's generating curve, center outward and down.

    Normals (-dz, dr)/|.| point into the air.  Segments are about ``segment``
    long, and the ones meeting at each corner are graded ``edge_grading`` times
    finer, where the field varies fastest.  Returns (segments, flat_top flags).
    """
    r_taper = radius - taper_length
    pieces = [((0.0, thickness), (r_taper, thickness), True)]
    if taper_length > 0.0:
        pieces.append(((r_taper, thickness), (radius, edge_thickness), False))
    if edge_thickness > 0.0:
        pieces.append(((radius, edge_thickness), (radius, 0.0), False))
    segs, flat = [], []
    for (r0, z0), (r1, z1), is_top in pieces:
        length = np.hypot(r1 - r0, z1 - z0)
        n = max(2, int(np.ceil(length / segment)))
        u = np.linspace(0.0, 1.0, n + 1)
        # grade toward both ends (corners), keeping the center of the top coarse
        if n >= 4:
            fine = min(edge_grading, n // 2)
            u = np.concatenate((np.linspace(0, 1 / n, fine + 1)[:-1], np.linspace(1 / n, 1 - 1 / n, n - 1)[:-1],
                                np.linspace(1 - 1 / n, 1, fine + 1)))
            if is_top:            # the center r = 0 is not a corner
                u = np.concatenate((np.linspace(0, 1 - 1 / n, n)[:-1], np.linspace(1 - 1 / n, 1, fine + 1)))
        for a, b in zip(u[:-1], u[1:]):
            segs.append((r0 + a * (r1 - r0), z0 + a * (z1 - z0), r0 + b * (r1 - r0), z0 + b * (z1 - z0)))
            flat.append(is_top)
    return np.array(segs), np.array(flat)


def _surface_modes(f, sound_speed, flow_resistance, ground, geometry, segments_per_wavelength, max_segment,
                   extra_modes, n_phi_uniform, gx, gw, extent, generator=None):
    """What :func:`scattering` and :func:`field` share at one frequency: the generating curve,
    the mode count, the ground, the Green's function table and the surface's modal kernels.

    ``extent(segment)`` gives the table's (rho_max, z_max), which depend on each caller's
    targets.  Returns a namespace of k, segs, flat, m_max, n_phi, beta, tab, mids and kern.
    """
    radius = geometry.get('radius', gp.PLATE_RADIUS_FT)
    k = 2 * np.pi * f / sound_speed
    segment = min(max_segment, sound_speed / f / segments_per_wavelength)
    if generator is None:
        segs, flat = plate_generator(segment=segment, **geometry)
    else:
        segs, flat = np.asarray(generator[0], float), np.asarray(generator[1], bool)
    m_max = int(np.ceil(k * radius)) + extra_modes
    # Uniform azimuth panels (``gauss`` points each) over the half turn: one
    # per mode.  Against 4 per mode it is within 3e-5 dB at realistic levels
    # (100 Hz and 10 kHz bands), like the 2 per mode used before, and it
    # moves the production tables' band levels by at most 6e-5 dB.
    n_phi = n_phi_uniform or max(16, m_max)
    beta = gp._ground_admittance(f, sound_speed, flow_resistance, ground)
    table = gp.ImageIntegralTable(k, beta, *extent(segment))
    tab = (table.u[0], table.u[1] - table.u[0], table.u.size, table.v[0], table.v[1] - table.v[0],
           table.v.size, table.offset, np.ascontiguousarray(table.i_red), np.ascontiguousarray(table.j_red))
    mids = np.column_stack((0.5 * (segs[:, 0] + segs[:, 2]), 0.5 * (segs[:, 1] + segs[:, 3])))
    kern = _assemble(mids, segs, flat, m_max, k, complex(beta), tab, gx, gw, n_phi)
    return SimpleNamespace(k=k, segs=segs, flat=flat, m_max=m_max, n_phi=n_phi, beta=beta, tab=tab, mids=mids,
                           kern=kern)


def _solve_modes(modes, el):
    """Surface pressure per mode for direct and ground-reflected plane waves from elevations ``el``
    (radians): (m, segment, 2 n_el), the direct waves' columns first.

    The modal incident fields (Jacobi-Anger) exclude the e^{-i m az} factor.
    """
    kappa = modes.k * np.cos(el)                                  # (n_el,)
    kz = modes.k * np.sin(el)
    m = np.arange(modes.m_max + 1)
    mids = modes.mids
    x = (kappa[None, :] * mids[:, 0:1]).ravel()
    jm = _bessel_j_orders(modes.m_max, x, j0(x), j1(x)).reshape(m.size, mids.shape[0], el.size)   # (m, n, el)
    im = (1j ** m)[:, None, None]
    inc_d = im * jm * np.exp(-1j * kz[None, None, :] * mids[None, :, 1:2])
    inc_r = im * jm * np.exp(+1j * kz[None, None, :] * mids[None, :, 1:2])
    n = modes.segs.shape[0]
    sols = np.empty((modes.m_max + 1, n, 2 * el.size), dtype=complex)
    for mm in range(modes.m_max + 1):
        system = 0.5 * np.eye(n) - modes.kern[mm]
        sols[mm] = np.linalg.solve(system, np.hstack((inc_d[mm], inc_r[mm])))
    return sols


def scattering(frequencies, elevations, azimuths, sound_speed, flow_resistance=gp.FLOW_RESISTANCE,
               ground=None, mic=(0.0, gp.PLATE_MIC_OFFSET_FT), segments_per_wavelength=10,
               max_segment=0.02, extra_modes=10, gauss=6, n_phi_uniform=None, generator=None,
               mic_rz=None, mic_height=0.0, **geometry):
    """(P_d, P_r) at a flush microphone on a rigid plate lying on the ground, like
    :func:`ground_plane.raised_plate_scattering`, by azimuthal modes.

    P = P_d + Q P_r re the direct wave at the microphone; arrays
    (len(frequencies), len(elevations), len(azimuths)).  ``geometry`` goes to
    :func:`plate_generator` (radius, thickness, edge_thickness, taper_length).
    Modes |m| <= k a + ``extra_modes``.

    ``generator`` (segments, flat flags), when given, replaces the plate's
    generating curve -- any body of revolution resting on the ground, with
    segments ordered so that (-dz, dr) points into the air -- and ``mic_rz``
    places the microphone at (r, z) on it (it must be at a smooth point of the
    discretized surface, e.g. a segment midpoint); ``mic`` then gives only its
    azimuth.  ``radius`` and ``thickness`` in ``geometry`` still size the
    modes and the Green's function table.

    ``mic_height`` > 0 lifts the microphone that far above the plate's top, in
    the air: an inverted microphone over a board (SAE ARP 4055's 7 mm gap,
    :data:`ground_plane.INVERTED_MIC_HEIGHT_FT`).  Off the surface the field is
    p_inc + K p, not the surface value 2 (p_inc + K p).  The microphone's own
    body is not modeled.  A flush microphone on the plate (no ``generator``
    or ``mic_rz``) must be on the flat top, as in
    :func:`ground_plane.raised_plate_scattering`.
    """
    frequencies = np.atleast_1d(np.asarray(frequencies, float))
    el = np.radians(np.atleast_1d(np.asarray(elevations, float)))
    az = np.radians(np.atleast_1d(np.asarray(azimuths, float)))
    radius = geometry.get('radius', gp.PLATE_RADIUS_FT)
    thickness = geometry.get('thickness', gp.PLATE_THICKNESS_FT)
    r_mic = float(np.hypot(*mic)); phi_mic = float(np.arctan2(mic[1], mic[0]))
    if (generator is None and mic_rz is None and mic_height == 0.0
            and r_mic >= radius - geometry.get('taper_length', gp.PLATE_TAPER_LENGTH_FT)):
        raise ValueError('a flush microphone must be on the flat top')
    z_mic = thickness + float(mic_height)
    surface = mic_height == 0.0
    if mic_rz is not None:
        r_mic, z_mic = float(mic_rz[0]), float(mic_rz[1])
        surface = True
    field_factor = 2.0 if surface else 1.0
    gx, gw = np.polynomial.legendre.leggauss(gauss)
    gx, gw = 0.5 * (gx + 1.0), 0.5 * gw
    pd_out = np.empty((frequencies.size, el.size, az.size), dtype=complex)
    pr_out = np.empty_like(pd_out)
    # Vertical reach: source on the plate plus target, up to a raised microphone.
    z_reach = max(2.0 * thickness, thickness + z_mic)
    extent = lambda segment: (2.0 * radius * 1.02 + 2 * segment, z_reach * 1.05)
    for fi, f in enumerate(frequencies):
        modes = _surface_modes(f, sound_speed, flow_resistance, ground, geometry, segments_per_wavelength,
                               max_segment, extra_modes, n_phi_uniform, gx, gw, extent, generator)
        k, m_max, n_phi = modes.k, modes.m_max, modes.n_phi
        target = np.array([[r_mic, z_mic]])
        # Off the surface the ring kernel peaks within ~ the height of the ring
        # below, so resolve that in azimuth (a few points per height).
        n_phi_mic = max(n_phi, 64)
        if not surface:
            n_phi_mic = max(n_phi_mic, int(np.ceil(2 * np.pi * r_mic / (float(mic_height) / 4.0))))
        kmic = _assemble(target, modes.segs, modes.flat, m_max, k, complex(modes.beta), modes.tab, gx, gw,
                         n_phi_mic)[:, 0, :]
        kappa = k * np.cos(el)                                  # (n_el,)
        kz = k * np.sin(el)
        m = np.arange(m_max + 1)
        sols = _solve_modes(modes, el)
        c_d = np.empty((m_max + 1, el.size), dtype=complex)
        c_r = np.empty_like(c_d)
        for mm in range(m_max + 1):
            c_d[mm] = kmic[mm] @ sols[mm][:, :el.size]
            c_r[mm] = kmic[mm] @ sols[mm][:, el.size:]
        # Sum modes -M..M at the microphone: coefficients for +-m are equal.
        weights = np.where(m == 0, 1.0, 2.0)
        phase = np.cos(m[:, None] * (phi_mic - az[None, :]))    # (m, az)
        mic_inc_h = np.exp(1j * kappa[:, None] * r_mic * np.cos(phi_mic - az[None, :]))   # (el, az)
        inc_mic_d = mic_inc_h * np.exp(-1j * kz * z_mic)[:, None]
        inc_mic_r = mic_inc_h * np.exp(+1j * kz * z_mic)[:, None]
        sum_d = np.einsum('m,me,ma->ea', weights, c_d, phase)
        sum_r = np.einsum('m,me,ma->ea', weights, c_r, phase)
        pd_out[fi] = field_factor * (inc_mic_d + sum_d) / inc_mic_d
        pr_out[fi] = field_factor * (inc_mic_r + sum_r) / inc_mic_d
    return pd_out, pr_out


#: Points of :func:`field` this close to the plate's surface (ft, about 3 nm) are on it.
SURFACE_TOLERANCE_FT = 1e-8


def field(frequency, elevation, azimuth, sound_speed, points, flow_resistance=gp.FLOW_RESISTANCE, ground=None,
          segments_per_wavelength=10, max_segment=0.02, extra_modes=10, gauss=6, n_phi_uniform=None,
          **geometry):
    """(P_d, P_r) at arbitrary points in the air around the plate on the ground, for one
    frequency and one plane-wave direction: the same surface solution as :func:`scattering`,
    evaluated off the surface (p = p_inc + K p there).

    ``points`` (n, 3): x, y, z in feet, the plate centered on the z axis with its base on the
    ground (z = 0).  Unlike :func:`scattering`, P_d and P_r are NOT normalized by the direct
    wave at each point: they are the complex fields per unit direct plane wave of phase zero at
    the origin, so the total field over the ground is P_d + Q P_r and the free field is
    exp(i k . x).  Points inside the plate are NaN.  Points on its surface (within
    :data:`SURFACE_TOLERANCE_FT`) take the surface value 2 (p_inc + K p), as the flush
    microphone does in :func:`scattering`; at the profile's corners, where the surface is
    not smooth, that is approximate.  ``geometry`` as in :func:`scattering`.
    """
    pts = np.atleast_2d(np.asarray(points, float))
    radius = geometry.get('radius', gp.PLATE_RADIUS_FT)
    thickness = geometry.get('thickness', gp.PLATE_THICKNESS_FT)
    f = float(frequency)
    el, az = np.radians(float(elevation)), np.radians(float(azimuth))
    r_t = np.hypot(pts[:, 0], pts[:, 1])
    phi_t = np.arctan2(pts[:, 1], pts[:, 0])
    z_t = pts[:, 2].copy()
    # Below the plate's own profile (flat top, taper, rim), not its bounding
    # cylinder: the air over the taper is outside.
    edge_thickness = geometry.get('edge_thickness', gp.PLATE_EDGE_THICKNESS_FT)
    taper_length = geometry.get('taper_length', gp.PLATE_TAPER_LENGTH_FT)
    top = (np.interp(r_t, [radius - taper_length, radius], [thickness, edge_thickness])
           if taper_length > 0.0 else np.full(r_t.shape, thickness))
    # Points on the surface are put exactly on it, where the kernel's
    # principal value holds; just off it the quadrature cannot resolve them.
    on_top = (r_t <= radius) & (np.abs(z_t - top) <= SURFACE_TOLERANCE_FT)
    on_rim = ~on_top & (np.abs(r_t - radius) <= SURFACE_TOLERANCE_FT) & (z_t <= edge_thickness)
    z_t[on_top] = top[on_top]
    r_t = np.where(on_rim, radius, r_t)
    on_surface = on_top | on_rim
    inside = (r_t < radius) & (z_t < top)
    extent = lambda segment: (max(2.0 * radius, float(r_t.max()) + radius) * 1.02 + 2 * segment,
                              max(2.0 * thickness, float(z_t.max()) + thickness) * 1.05)
    gx, gw = np.polynomial.legendre.leggauss(gauss)
    gx, gw = 0.5 * (gx + 1.0), 0.5 * gw
    modes = _surface_modes(f, sound_speed, flow_resistance, ground, geometry, segments_per_wavelength,
                           max_segment, extra_modes, n_phi_uniform, gx, gw, extent)
    k, m_max = modes.k, modes.m_max
    kappa, kz = k * np.cos(el), k * np.sin(el)
    m = np.arange(m_max + 1)
    sols = _solve_modes(modes, np.array([el]))
    sol_d, sol_r = np.ascontiguousarray(sols[:, :, 0]), np.ascontiguousarray(sols[:, :, 1])
    ok = ~inside
    targets = np.column_stack((r_t[ok], z_t[ok]))
    ktar = _assemble(targets, modes.segs, modes.flat, m_max, k, complex(modes.beta), modes.tab, gx, gw,
                     modes.n_phi)                                                         # (m, t, s)
    weights = np.where(m == 0, 1.0, 2.0)
    phase = weights[:, None] * np.cos(m[:, None] * (phi_t[ok][None, :] - az))            # (m, t)
    scat_d = np.einsum('mt,mts,ms->t', phase, ktar, sol_d)
    scat_r = np.einsum('mt,mts,ms->t', phase, ktar, sol_r)
    h = np.exp(1j * kappa * r_t[ok] * np.cos(phi_t[ok] - az))
    P_d = np.full(pts.shape[0], np.nan + 0j)
    P_r = np.full(pts.shape[0], np.nan + 0j)
    surface_factor = np.where(on_surface[ok], 2.0, 1.0)
    P_d[ok] = surface_factor * (h * np.exp(-1j * kz * z_t[ok]) + scat_d)
    P_r[ok] = surface_factor * (h * np.exp(+1j * kz * z_t[ok]) + scat_r)
    return P_d, P_r


# --------------------------------------------------------------------------
# Tables for many frames
# --------------------------------------------------------------------------

def table(bands, sound_speed, flow_resistance=gp.FLOW_RESISTANCE, ground=None, sub_bands=5,
          elevations=gp.BEM_ELEVATIONS, azimuths=gp.BEM_AZIMUTHS, **options):
    """(P_d, P_r) over each band's sub-frequencies, elevations and azimuths, for :func:`board_level`."""
    bands = np.asarray(bands, dtype=float)
    offsets = gp.sub_band_factors(sub_bands)
    frequencies = np.sort((bands[:, None] * offsets[None, :]).ravel())
    p_d, p_r = scattering(frequencies, elevations, azimuths, sound_speed, flow_resistance, ground,
                          **options)
    return dict(frequencies=frequencies, elevations=np.asarray(elevations, float),
                azimuths=np.asarray(azimuths, float), P_d=p_d, P_r=p_r, ground=ground,
                flow_resistance=flow_resistance, bands=bands, sub_bands=sub_bands,
                sound_speed=sound_speed,
                thickness=options.get('thickness', gp.PLATE_THICKNESS_FT),
                radius=options.get('radius', gp.PLATE_RADIUS_FT),
                edge_thickness=options.get('edge_thickness', gp.PLATE_EDGE_THICKNESS_FT),
                taper_length=options.get('taper_length', gp.PLATE_TAPER_LENGTH_FT),
                mic=options.get('mic', (0.0, gp.PLATE_MIC_OFFSET_FT)),
                mic_height=options.get('mic_height', 0.0))


def board_level(bands, source_height, ground_distance, sound_speed, table, sub_bands=None,
                source_dx=None, source_dy=None, mirror_y=False):
    """The flush microphone on the plate lying on the ground, band averaged, dB re free field.

    |P_d + Q P_r|^2 averaged over the band's sub-frequencies, with P_d, P_r
    interpolated (real and imaginary parts, bilinear in elevation and azimuth)
    from ``table`` and Q from each frame's geometry, at the plate's top.
    Azimuth, ``mirror_y`` and NaN for bands not in the table as in
    :func:`ground_plane.table_frames`.  ``sub_bands`` is the table's own; a
    different count would pick frequencies the table does not hold.
    """
    f, hs, d2, elevation, offsets, values = gp.table_frames(
        table, ('P_d', 'P_r'), bands, source_height, ground_distance, source_dx, source_dy, mirror_y, sub_bands)
    height = table['thickness']
    image_range = np.hypot(d2, hs + height)
    cos_theta = (hs + height) / image_range
    energy = np.zeros(f.shape)
    for factor, (p_d, p_r) in zip(offsets, values):
        fj = f * factor
        q = gp.fa.spherical_reflection_coefficient(
            cos_theta, image_range, fj, sound_speed, table['flow_resistance'],
            admittance=gp._ground_admittance(fj, sound_speed, table['flow_resistance'], table.get('ground')))
        energy += np.abs(p_d + q * p_r) ** 2
    return 10.0 * np.log10(energy / offsets.size)


def write_netcdf(path, table, description=''):
    """Write a :func:`table` for NICE-OPS's ground-plane receiver (--plate_table).

    Dimensions band x sub x elevation x azimuth; P_d and P_r as real and
    imaginary parts.  Lengths in meters and the sound speed in m/s (NICE-OPS
    is metric inside).  Azimuth is that of the horizontal propagation
    direction, degrees from +x toward +y, in the plate's frame, where the
    microphone is offset along +y; elevation is the source's, above the ground.
    The ground the table was computed on is recorded, because P_d and P_r
    depend on it: NICE-OPS checks it against its own.
    """
    from netCDF4 import Dataset
    ft = 0.3048
    # The table's frequencies are sorted, so they fall into (band, sub) rows
    # only in ascending band order, and only if no two bands' sub-frequencies
    # interleave.
    bands = np.sort(np.asarray(table['bands'], float))
    n_sub = int(table['sub_bands'])
    frequency = np.asarray(table['frequencies'], float).reshape(bands.size, n_sub)
    if not np.allclose(frequency, bands[:, None] * gp.sub_band_factors(n_sub)[None, :], rtol=1e-9, atol=0.0):
        raise ValueError("the table's frequencies do not split into its bands' sub-frequencies")
    shape = (bands.size, n_sub, table['elevations'].size, table['azimuths'].size)
    ground = table.get('ground') or dict(model='delany_bazley', sigma=table['flow_resistance'])
    with Dataset(path, 'w') as nc:
        nc.createDimension('band', bands.size)
        nc.createDimension('sub', n_sub)
        nc.createDimension('elevation', table['elevations'].size)
        nc.createDimension('azimuth', table['azimuths'].size)
        nc.createVariable('band_center', 'f8', ('band',))[:] = bands
        nc.createVariable('frequency', 'f8', ('band', 'sub'))[:] = frequency
        nc.createVariable('elevation', 'f8', ('elevation',))[:] = table['elevations']
        nc.createVariable('azimuth', 'f8', ('azimuth',))[:] = table['azimuths']
        for key in ('P_d', 'P_r'):
            data = table[key].reshape(shape)
            nc.createVariable(key + '_real', 'f8', ('band', 'sub', 'elevation', 'azimuth'), zlib=True)[:] = data.real
            nc.createVariable(key + '_imag', 'f8', ('band', 'sub', 'elevation', 'azimuth'), zlib=True)[:] = data.imag
        nc.description = description or 'Rigid plate lying on the ground: P = P_d + Q P_r re the direct wave'
        nc.radius_m = table['radius'] * ft
        nc.thickness_m = table['thickness'] * ft
        nc.edge_thickness_m = table['edge_thickness'] * ft
        nc.taper_length_m = table['taper_length'] * ft
        nc.mic_x_m, nc.mic_y_m = (float(v) * ft for v in table['mic'])
        nc.mic_height_m = float(table.get('mic_height', 0.0)) * ft
        nc.sound_speed_mps = table['sound_speed'] * ft
        nc.ground_model = ground['model']
        nc.flow_resistance = float(ground.get('sigma', ground.get('sigma_e', table['flow_resistance'])))
        nc.porosity_rate = float(ground.get('alpha_e', 0.0))
        nc.layer_depth_m = float(ground.get('depth', 0.0))
