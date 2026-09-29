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
import numpy as np
from numba import njit, prange
from scipy.special import jv

import ground_plane as gp


# --------------------------------------------------------------------------
# Compiled pieces
# --------------------------------------------------------------------------

@njit(cache=True, fastmath=False)
def _table_lookup(rho, z, u0, du, nu, v0, dv, nv, offset, k, i_red, j_red):
    """Bilinear lookup of the exact image integrals I, J (see gp.ImageIntegralTable)."""
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
    r2 = max(np.sqrt(rho * rho + z * z), offset)
    ph = np.exp(1j * k * r2)
    return ir * ph / (4 * np.pi * r2), jr * ph / (4 * np.pi * r2 * r2)


@njit(cache=True)
def _dgdn(xr, xz, yr, yz, phi, nr, nz, k, beta, tab, skip_direct):
    """dG/dn_y for target x = (xr, 0, xz) and source y = (yr cos phi, yr sin phi, yz), normal (nr, nz) at y."""
    u0, du, nu, v0, dv, nv, offset, i_red, j_red = tab
    c = np.cos(phi)
    dx = yr * c - xr
    dy = yr * np.sin(phi)
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
    g2 = np.exp(1j * k * r2) / (4 * np.pi * r2)
    big_i, big_j = _table_lookup(rho, zs, u0, du, nu, v0, dv, nv, offset, k, i_red, j_red)
    dimg_z = g2 * (1j * k - 1.0 / r2) * zs / r2 - 2j * k * beta * g2 + 2j * k * k * beta * beta * big_i
    dimg_h = g2 * (1j * k - 1.0 / r2) / r2 - 2 * k * beta * big_j
    return out + nz * dimg_z + n_dot_h * dimg_h


@njit(cache=True)
def _graded_edges(centre, scale, lo, hi, n_uniform, ratio):
    """Breakpoints on [lo, hi]: geometric from ``centre`` (step ``scale`` x ratio^j), plus uniform ones."""
    edges = [lo, hi]
    for i in range(1, n_uniform):
        edges.append(lo + (hi - lo) * i / n_uniform)
    step = scale
    while step < (hi - lo):
        a = centre - step
        b = centre + step
        if lo < a < hi:
            edges.append(a)
        if lo < b < hi:
            edges.append(b)
        step *= ratio
    if lo < centre < hi:
        edges.append(centre)
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
    out = np.zeros(m_max + 1, dtype=np.complex128)
    ng = gx.size
    for a in range(u_edges.size - 1):
        u_lo, u_hi = u_edges[a], u_edges[a + 1]
        if u_hi <= u_lo:
            continue
        for i in range(ng):
            u = u_lo + (u_hi - u_lo) * gx[i]
            wu = (u_hi - u_lo) * gw[i]
            yr = ra + u * tr
            yz = za + u * tz
            jac = yr * length * wu
            for b in range(phi_edges.size - 1):
                p_lo, p_hi = phi_edges[b], phi_edges[b + 1]
                if p_hi <= p_lo:
                    continue
                for j in range(ng):
                    phi = p_lo + (p_hi - p_lo) * gx[j]
                    wp = (p_hi - p_lo) * gw[j]
                    val = _dgdn(xr, xz, yr, yz, phi, nr, nz, k, beta, tab, same_plane) * jac * wp * 2.0
                    # cos(m phi) by recurrence
                    c1 = np.cos(phi)
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
    for i in prange(n_t):
        for j in range(n_s):
            same_plane = flat_top[j] and abs(targets[i, 1] - segs[j, 1]) < 1e-12 and flat_top[j]
            vals = _modal_integral(targets[i, 0], targets[i, 1], segs[j], m_max, k, beta, tab, gx, gw,
                                   n_phi_uniform, same_plane)
            for m in range(m_max + 1):
                out[m, i, j] = vals[m]
    return out


# --------------------------------------------------------------------------
# Geometry and driver
# --------------------------------------------------------------------------

def plate_generator(radius=gp.PLATE_RADIUS_FT, thickness=gp.PLATE_THICKNESS_FT,
                    edge_thickness=gp.PLATE_EDGE_THICKNESS_FT, taper_length=gp.PLATE_TAPER_LENGTH_FT,
                    segment=0.02, edge_grading=4):
    """Segments (r0, z0, r1, z1) of the plate's generating curve, centre outward and down.

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
        # grade toward both ends (corners), keeping the centre of the top coarse
        if n >= 4:
            fine = min(edge_grading, n // 2)
            u = np.concatenate((np.linspace(0, 1 / n, fine + 1)[:-1], np.linspace(1 / n, 1 - 1 / n, n - 1)[:-1],
                                np.linspace(1 - 1 / n, 1, fine + 1)))
            if is_top:            # the centre r = 0 is not a corner
                u = np.concatenate((np.linspace(0, 1 - 1 / n, n)[:-1], np.linspace(1 - 1 / n, 1, fine + 1)))
        for a, b in zip(u[:-1], u[1:]):
            segs.append((r0 + a * (r1 - r0), z0 + a * (z1 - z0), r0 + b * (r1 - r0), z0 + b * (z1 - z0)))
            flat.append(is_top)
    return np.array(segs), np.array(flat)


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
    discretised surface, e.g. a segment midpoint); ``mic`` then gives only its
    azimuth.  ``radius`` and ``thickness`` in ``geometry`` still size the
    modes and the Green's function table.

    ``mic_height`` > 0 lifts the microphone that far above the plate's top, in
    the air: an inverted microphone over a board (SAE ARP 4055's 7 mm gap,
    :data:`ground_plane.INVERTED_MIC_HEIGHT_FT`).  Off the surface the field is
    p_inc + K p, not the surface value 2 (p_inc + K p).  The microphone's own
    body is not modelled.
    """
    frequencies = np.atleast_1d(np.asarray(frequencies, float))
    el = np.radians(np.atleast_1d(np.asarray(elevations, float)))
    az = np.radians(np.atleast_1d(np.asarray(azimuths, float)))
    radius = geometry.get('radius', gp.PLATE_RADIUS_FT)
    thickness = geometry.get('thickness', gp.PLATE_THICKNESS_FT)
    r_mic = float(np.hypot(*mic)); phi_mic = float(np.arctan2(mic[1], mic[0]))
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
    for fi, f in enumerate(frequencies):
        k = 2 * np.pi * f / sound_speed
        segment = min(max_segment, sound_speed / f / segments_per_wavelength)
        if generator is None:
            segs, flat = plate_generator(segment=segment, **geometry)
        else:
            segs, flat = np.asarray(generator[0], float), np.asarray(generator[1], bool)
        m_max = int(np.ceil(k * radius)) + extra_modes
        n_phi = n_phi_uniform or max(16, 2 * m_max)
        beta = gp._ground_admittance(f, sound_speed, flow_resistance, ground)
        # Vertical reach: source on the plate plus target, up to a raised microphone.
        z_reach = max(2.0 * thickness, thickness + z_mic)
        table = gp.ImageIntegralTable(k, beta, 2.0 * radius * 1.02 + 2 * segment, z_reach * 1.05)
        tab = (table.u[0], table.u[1] - table.u[0], table.u.size, table.v[0], table.v[1] - table.v[0],
               table.v.size, table.offset, np.ascontiguousarray(table.i_red), np.ascontiguousarray(table.j_red))
        mids = np.column_stack((0.5 * (segs[:, 0] + segs[:, 2]), 0.5 * (segs[:, 1] + segs[:, 3])))
        kern = _assemble(mids, segs, flat, m_max, k, complex(beta), tab, gx, gw, n_phi)
        target = np.array([[r_mic, z_mic]])
        # Off the surface the ring kernel peaks within ~ the height of the ring
        # below, so resolve that in azimuth (a few points per height).
        n_phi_mic = max(n_phi, 64)
        if not surface:
            n_phi_mic = max(n_phi_mic, int(np.ceil(2 * np.pi * r_mic / (float(mic_height) / 4.0))))
        kmic = _assemble(target, segs, flat, m_max, k, complex(beta), tab, gx, gw, n_phi_mic)[:, 0, :]
        n = segs.shape[0]
        kappa = k * np.cos(el)                                  # (n_el,)
        kz = k * np.sin(el)
        m = np.arange(m_max + 1)
        # Modal incident fields (per mode m, excluding the e^{-i m az} factor).
        jm = jv(m[:, None, None], kappa[None, None, :] * mids[None, :, 0:1])      # (m, n, el)
        im = (1j ** m)[:, None, None]
        inc_d = im * jm * np.exp(-1j * kz[None, None, :] * mids[None, :, 1:2])
        inc_r = im * jm * np.exp(+1j * kz[None, None, :] * mids[None, :, 1:2])
        c_d = np.empty((m_max + 1, el.size), dtype=complex)
        c_r = np.empty_like(c_d)
        for mm in range(m_max + 1):
            system = 0.5 * np.eye(n) - kern[mm]
            sol = np.linalg.solve(system, np.hstack((inc_d[mm], inc_r[mm])))
            c_d[mm] = kmic[mm] @ sol[:, :el.size]
            c_r[mm] = kmic[mm] @ sol[:, el.size:]
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
    exp(i k . x).  Points inside the plate are NaN.  ``geometry`` as in :func:`scattering`.
    """
    pts = np.atleast_2d(np.asarray(points, float))
    radius = geometry.get('radius', gp.PLATE_RADIUS_FT)
    thickness = geometry.get('thickness', gp.PLATE_THICKNESS_FT)
    f = float(frequency)
    el, az = np.radians(float(elevation)), np.radians(float(azimuth))
    k = 2 * np.pi * f / sound_speed
    segment = min(max_segment, sound_speed / f / segments_per_wavelength)
    segs, flat = plate_generator(segment=segment, **geometry)
    m_max = int(np.ceil(k * radius)) + extra_modes
    n_phi = n_phi_uniform or max(16, 2 * m_max)
    beta = gp._ground_admittance(f, sound_speed, flow_resistance, ground)
    r_t = np.hypot(pts[:, 0], pts[:, 1])
    phi_t = np.arctan2(pts[:, 1], pts[:, 0])
    z_t = pts[:, 2]
    inside = (r_t < radius) & (z_t < thickness)
    rho_max = max(2.0 * radius, float(r_t.max()) + radius) * 1.02 + 2 * segment
    z_max = max(2.0 * thickness, float(z_t.max()) + thickness) * 1.05
    tbl = gp.ImageIntegralTable(k, beta, rho_max, z_max)
    tab = (tbl.u[0], tbl.u[1] - tbl.u[0], tbl.u.size, tbl.v[0], tbl.v[1] - tbl.v[0], tbl.v.size, tbl.offset,
           np.ascontiguousarray(tbl.i_red), np.ascontiguousarray(tbl.j_red))
    gx, gw = np.polynomial.legendre.leggauss(gauss)
    gx, gw = 0.5 * (gx + 1.0), 0.5 * gw
    mids = np.column_stack((0.5 * (segs[:, 0] + segs[:, 2]), 0.5 * (segs[:, 1] + segs[:, 3])))
    kern = _assemble(mids, segs, flat, m_max, k, complex(beta), tab, gx, gw, n_phi)
    kappa, kz = k * np.cos(el), k * np.sin(el)
    m = np.arange(m_max + 1)
    jm = jv(m[:, None], kappa * mids[None, :, 0])
    im = (1j ** m)[:, None]
    inc_d = im * jm * np.exp(-1j * kz * mids[None, :, 1])
    inc_r = im * jm * np.exp(+1j * kz * mids[None, :, 1])
    n = segs.shape[0]
    sol_d = np.empty((m_max + 1, n), dtype=complex)
    sol_r = np.empty_like(sol_d)
    for mm in range(m_max + 1):
        system = 0.5 * np.eye(n) - kern[mm]
        sol = np.linalg.solve(system, np.column_stack((inc_d[mm], inc_r[mm])))
        sol_d[mm], sol_r[mm] = sol[:, 0], sol[:, 1]
    ok = ~inside
    targets = np.column_stack((r_t[ok], z_t[ok]))
    ktar = _assemble(targets, segs, flat, m_max, k, complex(beta), tab, gx, gw, n_phi)   # (m, t, s)
    weights = np.where(m == 0, 1.0, 2.0)
    phase = weights[:, None] * np.cos(m[:, None] * (phi_t[ok][None, :] - az))            # (m, t)
    scat_d = np.einsum('mt,mts,ms->t', phase, ktar, sol_d)
    scat_r = np.einsum('mt,mts,ms->t', phase, ktar, sol_r)
    h = np.exp(1j * kappa * r_t[ok] * np.cos(phi_t[ok] - az))
    P_d = np.full(pts.shape[0], np.nan + 0j)
    P_r = np.full(pts.shape[0], np.nan + 0j)
    P_d[ok] = h * np.exp(-1j * kz * z_t[ok]) + scat_d
    P_r[ok] = h * np.exp(+1j * kz * z_t[ok]) + scat_r
    return P_d, P_r


# --------------------------------------------------------------------------
# Tables for many frames
# --------------------------------------------------------------------------

def table(bands, sound_speed, flow_resistance=gp.FLOW_RESISTANCE, ground=None, sub_bands=5,
          elevations=gp.BEM_ELEVATIONS, azimuths=gp.BEM_AZIMUTHS, **options):
    """(P_d, P_r) over each band's sub-frequencies, elevations and azimuths, for :func:`board_level`."""
    bands = np.asarray(bands, dtype=float)
    offsets = 2.0 ** ((np.arange(sub_bands) + 0.5) / sub_bands / 3.0 - 1.0 / 6.0)
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
    Azimuth, ``mirror_y`` and NaN above the table as in
    :func:`ground_plane.board_disc_bem`.  ``sub_bands`` is the table's own; a
    different count would pick frequencies the table does not hold.
    """
    from scipy.interpolate import RegularGridInterpolator
    sub_bands = gp.table_sub_bands(table, sub_bands)
    f, hs, d2 = gp._broadcast(bands, source_height, ground_distance)
    elevation = np.degrees(np.arctan2(hs, d2))
    height = table['thickness']
    image_range = np.hypot(d2, hs + height)
    cos_theta = (hs + height) / image_range
    if source_dx is None:
        azimuth = np.zeros_like(hs)
    else:
        _, dx, dy = gp._broadcast(bands, source_dx, source_dy)
        azimuth = np.mod(np.degrees(np.arctan2(-dy, -dx)), 360.0)
    if mirror_y:
        azimuth = np.mod(-azimuth, 360.0)
    t_az = np.concatenate((table['azimuths'], [table['azimuths'][0] + 360.0]))
    offsets = 2.0 ** ((np.arange(sub_bands) + 0.5) / sub_bands / 3.0 - 1.0 / 6.0)
    top = table['frequencies'].max() * (1.0 + 1e-9)
    energy = np.zeros(f.shape)
    for factor in offsets:
        fj = f * factor
        q = gp.fa.spherical_reflection_coefficient(
            cos_theta, image_range, fj, sound_speed, table['flow_resistance'],
            admittance=gp._ground_admittance(fj, sound_speed, table['flow_resistance'], table.get('ground')))
        p = np.full(f.shape, np.nan, dtype=complex)
        for b in range(f.shape[0]):
            if fj[b, 0] > top:
                continue
            row = np.argmin(np.abs(table['frequencies'] - fj[b, 0]))
            points = np.column_stack((np.clip(elevation[b], table['elevations'][0], table['elevations'][-1]),
                                      azimuth[b]))
            vals = []
            for key in ('P_d', 'P_r'):
                grid = np.concatenate((table[key][row], table[key][row][:, :1]), axis=1)
                vals.append(RegularGridInterpolator((table['elevations'], t_az), grid.real)(points)
                            + 1j * RegularGridInterpolator((table['elevations'], t_az), grid.imag)(points))
            p[b] = vals[0] + q[b] * vals[1]
        energy += np.abs(p) ** 2
    return 10.0 * np.log10(energy / sub_bands)


def write_netcdf(path, table, description=''):
    """Write a :func:`table` for NICE-OPS's ground-plane receiver (--plate_table).

    Dimensions band x sub x elevation x azimuth; P_d and P_r as real and
    imaginary parts.  Lengths in metres and the sound speed in m/s (NICE-OPS
    is metric inside).  Azimuth is that of the horizontal propagation
    direction, degrees from +x toward +y, in the plate's frame, where the
    microphone is offset along +y; elevation is the source's, above the ground.
    The ground the table was computed on is recorded, because P_d and P_r
    depend on it: NICE-OPS checks it against its own.
    """
    from netCDF4 import Dataset
    ft = 0.3048
    bands = np.asarray(table['bands'], float)
    n_sub = int(table['sub_bands'])
    shape = (bands.size, n_sub, table['elevations'].size, table['azimuths'].size)
    ground = table.get('ground') or dict(model='delany_bazley', sigma=table['flow_resistance'])
    with Dataset(path, 'w') as nc:
        nc.createDimension('band', bands.size)
        nc.createDimension('sub', n_sub)
        nc.createDimension('elevation', table['elevations'].size)
        nc.createDimension('azimuth', table['azimuths'].size)
        nc.createVariable('band_centre', 'f8', ('band',))[:] = bands
        nc.createVariable('frequency', 'f8', ('band', 'sub'))[:] = table['frequencies'].reshape(bands.size, n_sub)
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
