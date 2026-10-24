"""Gridding: inverse-distance weights on the sphere, and hemisphere levels at any direction.

:func:`interpolate_hemisphere_power` samples a gridded UMAPR hemisphere at any
direction, bilinearly in energy over the measured cells; :func:`_sample_hemisphere_levels`
is the sphere and NORAH2 writers' front end to it (``docs/THEORY.md`` section 6.3).
:func:`repitch_art_levels` refiles an ART sphere's levels in a frame pitched from its own
(section 3.2: AAM's flight-path frame to the level one).

Scattered samples onto grid nodes: :func:`adaptive_idw_weights` chooses each node's
radius from the microphones around it; :func:`shepIDW`, :func:`shepIDW_weights` and
:func:`shepIDW_apply` use a fixed radius (sections 6.1-6.2).  Levels are interpolated
in energy.  Angles are in degrees.
"""

import numpy as np
import scipy.sparse as sparse
from scipy.interpolate import RegularGridInterpolator

from panam.geometry import pitch_art_directions


def interpolate_hemisphere_power(levels_db, elv_grid_deg, azi_grid_deg, azi_q_deg, elv_q_deg):
    """Band power of a gridded hemisphere at arbitrary directions, interpolated in energy.

    Each band is interpolated bilinearly over the UMAPR (elevation, azimuth)
    grid in linear power over the measured cells only.  Unmeasured (NaN) cells
    carry no weight: which cells are measured is interpolated alongside the
    power and the power renormalized by it, so a level next to a coverage gap
    is not pulled toward zero energy.  A direction weighted mostly by
    unmeasured cells, or off the grid, is not covered.  Azimuth is periodic:
    a seam column at the first azimuth + 360 deg is used if present (its
    values replaced by the first column's) and appended if not.

    Args:
        levels_db: (Nf, Nelv, Nazi) levels, dB (-inf no energy, NaN no data).
        elv_grid_deg: (Nelv,) strictly increasing UMAPR elevations, deg.
        azi_grid_deg: (Nazi,) strictly increasing UMAPR azimuths, deg.
        azi_q_deg, elv_q_deg: UMAPR query directions, deg, any matching shape.

    Returns:
        (power, covered), each shaped ``azi_q_deg.shape + (Nf,)``: the power
        relative to the reference of ``levels_db`` (0 where not covered), and
        True where the direction is covered (the measured cells' interpolated
        weight is at least 0.5).
    """
    levels_db = np.asarray(levels_db, dtype=float)
    azi_axis = np.asarray(azi_grid_deg, dtype=float).ravel()
    elv_grid_deg = np.asarray(elv_grid_deg, dtype=float).ravel()
    if levels_db.ndim != 3 or levels_db.shape[1] != elv_grid_deg.size or levels_db.shape[2] != azi_axis.size:
        raise ValueError('Spectrum grid shape does not match elv_grid_deg/azi_grid_deg')

    if np.any(np.diff(azi_axis) <= 0.0) or np.any(np.diff(elv_grid_deg) <= 0.0):
        raise ValueError('azi_grid_deg and elv_grid_deg must be strictly increasing')

    # Ensure azimuth periodicity is represented in the interpolation grid.
    # We need the seam column at (azi0 + 360 deg) with values matching azi0,
    # otherwise points near 360 deg can fall out-of-bounds and become missing.
    azi0 = float(azi_axis[0])
    azi_end = azi0 + 360.0
    if np.isclose(azi_axis[-1], azi_end, rtol=0.0, atol=1e-6):
        # Seam column present; ensure the last column matches the first.
        levels_db = levels_db.copy()
        levels_db[:, :, -1] = levels_db[:, :, 0]
    else:
        # No seam column; append one.
        azi_axis = np.concatenate((azi_axis, np.array([azi_end], dtype=float)))
        seam_col = levels_db[:, :, 0:1]
        levels_db = np.concatenate((levels_db, seam_col), axis=2)

    query_shape = np.shape(azi_q_deg)
    # Wrap query azimuths into the interpolation axis range [azi0, azi0+360)
    azi_q = (np.asarray(azi_q_deg, dtype=float).reshape(-1) - azi0) % 360.0 + azi0
    elv_q = np.asarray(elv_q_deg, dtype=float).reshape(-1)
    # Query points are (elv, azi) pairs in degrees
    pts = np.column_stack((elv_q, azi_q))

    nfreq = levels_db.shape[0]
    P = np.moveaxis(np.power(10.0, levels_db / 10.0), 0, -1)       # (Nelv, Nazi, Nf)
    Pq, covered = _interpolate_measured_power((elv_grid_deg, azi_axis), P, np.isfinite(P), pts)
    return Pq.reshape(query_shape + (nfreq,)), covered.reshape(query_shape + (nfreq,))


def _interpolate_measured_power(axes, power, measured, points):
    """Bilinear interpolation of band power over the measured grid cells only.

    Every band is interpolated at once in linear power.  Unmeasured cells carry
    no weight: which cells are measured is interpolated alongside the power and
    the power renormalized by it, so a level next to a coverage gap is not
    pulled toward zero energy.  A point weighted mostly by unmeasured cells (or
    off the grid) is not covered, which puts the edge of coverage halfway
    between measured and unmeasured cells.

    Args:
        axes: the grid's strictly increasing axes (deg), one per leading axis of
            ``power``.
        power: (..., Nf) band power on the grid, any power unit, bands last.
        measured: boolean, shaped like ``power``: True where a cell is measured.
            Unmeasured cells' power is ignored.
        points: (Npts, len(axes)) query points, in the axes' units.

    Returns:
        (power, covered), each (Npts, Nf): the power (0 where not covered) and
        True where the measured cells' interpolated weight is at least 0.5.
    """
    nfreq = power.shape[-1]
    interp = RegularGridInterpolator(
        axes,
        np.concatenate((np.where(measured, power, 0.0), measured.astype(float)), axis=-1),
        bounds_error=False,
        fill_value=0.0,
    )
    sampled = interp(points)
    Pq, weight = sampled[:, :nfreq], sampled[:, nfreq:]
    covered = weight >= 0.5
    Pq = np.divide(Pq, weight, out=np.zeros_like(Pq), where=covered)
    return Pq, covered


def repitch_art_levels(levels_db, phi_deg, theta_deg, pitch_deg):
    """An ART sphere's band levels refiled in a frame pitched from the one they are in.

    The levels are filed in a frame pitched nose-up by ``pitch_deg`` from the
    frame they are returned in, on the same (phi, theta) grid: a flight-path
    sphere at flight path angle gamma comes back level with ``pitch_deg=gamma``,
    and a level one in the flight-path frame with ``-gamma``.  Each grid direction
    is read off the source bilinearly in energy over its measured (finite) cells,
    as :func:`interpolate_hemisphere_power` reads a hemisphere.  A lower-hemisphere
    sphere (|phi| <= 90 deg) has no data above its own equator, where a pitched
    grid direction near the rim can land, up to the pitch away; it takes the level
    of the nearest point on the equator there (the direction with its down
    component dropped), and such a cell is never measured.  The mirror image, as the
    database completion fills the upper surface, reads twice as far from the
    direction: on a level pattern of 10 dB from horizon to nadir pitched 8 deg it
    filled the nose 2.8 dB loud, the nearest point 1.4.  A sphere storing the whole
    circle wraps in phi.

    Args:
        levels_db: (Nphi, Ntheta, Nf) band levels, dB, -inf or NaN where missing.
        phi_deg, theta_deg: the grid's strictly increasing ART axes, deg.
        pitch_deg: the source frame's pitch from the returned one, deg, nose up.

    Returns: (levels_db, measured): the levels on the same grid in the returned
        frame (-inf where not covered) and a bool array of the same shape, True
        where a cell is read from measured cells of the source's own half.  A zero
        pitch returns the levels as they are.
    """
    levels = np.asarray(levels_db, dtype=float)
    phi_axis = np.asarray(phi_deg, dtype=float).ravel()
    theta_axis = np.asarray(theta_deg, dtype=float).ravel()
    if levels.ndim != 3 or levels.shape[:2] != (phi_axis.size, theta_axis.size):
        raise ValueError('levels_db must be shaped (phi, theta, frequency) = ({}, {}, ...), not {}'.format(
            phi_axis.size, theta_axis.size, levels.shape))
    if np.any(np.diff(phi_axis) <= 0.0) or np.any(np.diff(theta_axis) <= 0.0):
        raise ValueError('phi_deg and theta_deg must be strictly increasing')
    if float(pitch_deg) == 0.0:
        return levels.copy(), np.isfinite(levels)
    theta_grid, phi_grid = np.meshgrid(theta_axis, phi_axis)
    query_phi, query_theta = pitch_art_directions(phi_grid, theta_grid, -float(pitch_deg))
    if np.max(np.abs(phi_axis)) > 90.0 + 1e-9:
        # The whole circle: wrap, with a seam column one period on, as
        # interpolate_hemisphere_power does in azimuth.
        mirrored = np.zeros(query_phi.shape, dtype=bool)
        phi0 = float(phi_axis[0])
        if not np.isclose(phi_axis[-1], phi0 + 360.0, rtol=0.0, atol=1e-6):
            phi_axis = np.append(phi_axis, phi0 + 360.0)
            levels = np.concatenate((levels, levels[:1]), axis=0)
        query_phi = (query_phi - phi0) % 360.0 + phi0
    else:
        # Above the equator: the nearest point on it, phi +/-90 at the theta of the
        # direction's (forward, starboard) part.
        mirrored = np.abs(query_phi) > 90.0 + 1e-9
        forward = np.cos(np.radians(query_theta))
        starboard = np.sin(np.radians(query_theta)) * np.sin(np.radians(query_phi))
        equator_theta = np.degrees(np.arctan2(np.abs(starboard), forward))
        query_theta = np.where(mirrored, equator_theta, query_theta)
        query_phi = np.where(mirrored, np.where(starboard < 0.0, -90.0, 90.0), query_phi)
    power = np.power(10.0, np.where(np.isnan(levels), -np.inf, levels) / 10.0)
    points = np.column_stack((query_phi.ravel(), query_theta.ravel()))
    Pq, covered = _interpolate_measured_power((phi_axis, theta_axis), power, np.isfinite(levels), points)
    shape = query_phi.shape + (levels.shape[2],)
    Pq, covered = Pq.reshape(shape), covered.reshape(shape)
    out = np.full(shape, -np.inf)
    positive = covered & (Pq > 0.0)
    out[positive] = 10.0 * np.log10(Pq[positive])
    return out, covered & ~mirrored[:, :, None]


def _sample_hemisphere_levels(hemisphere, mode, azi_q_deg, elv_q_deg, minimum_level_db):
    """Band levels of a :func:`depropagation.depropagate_hemisphere` result at arbitrary directions.

    The shared core of the sphere exporters (:func:`sphere_io.write_aam_hemisphere_netcdf`,
    :func:`norah2.write_norah2_hemisphere`): pick the spectrum ``mode`` selects, then
    interpolate each band over the UMAPR azimuth/elevation grid in linear power over the
    measured cells only.  Directions off the grid or mostly among unmeasured (NaN) cells,
    and levels below ``minimum_level_db``, come back as -inf; each writer turns that into
    its own missing-value convention.

    Args:
        hemisphere: Output dict from :func:`depropagation.depropagate_hemisphere`.
        mode: 'auto', 'third_octave', or 'narrowband'. 'auto' prefers third-octave.
        azi_q_deg, elv_q_deg: UMAPR query directions (degrees), any matching shape.
        minimum_level_db: see :func:`sphere_io.write_aam_hemisphere_netcdf`.

    Returns:
        (frequency_hz, levels_db) with levels_db shaped ``azi_q_deg.shape + (Nf,)``.
    """
    if not isinstance(hemisphere, dict):
        raise TypeError('hemisphere must be a dict returned by depropagate_hemisphere')

    azi_grid_deg = np.asarray(hemisphere.get('azi_grid_deg', []), dtype=float)
    elv_grid_deg = np.asarray(hemisphere.get('elv_grid_deg', []), dtype=float)
    if azi_grid_deg.ndim != 1 or azi_grid_deg.size < 2:
        raise ValueError('hemisphere[\'azi_grid_deg\'] must be a 1D array with at least 2 elements')
    if elv_grid_deg.ndim != 1 or elv_grid_deg.size < 2:
        raise ValueError('hemisphere[\'elv_grid_deg\'] must be a 1D array with at least 2 elements')

    # Determine export spectrum (frequency + band levels) from hemisphere.
    selected_mode = str(mode).lower()
    if selected_mode not in ('auto', 'third_octave', 'narrowband'):
        raise ValueError("mode must be 'auto', 'third_octave' or 'narrowband', not {!r}".format(mode))
    if selected_mode == 'auto':
        if 'third_octave' in hemisphere:
            selected_mode = 'third_octave'
        elif 'narrowband' in hemisphere:
            selected_mode = 'narrowband'
        else:
            selected_mode = 'none'

    eps = np.finfo(float).tiny
    if selected_mode == 'third_octave':
        if 'third_octave' not in hemisphere:
            raise ValueError('mode=third_octave but hemisphere does not include third_octave data')
        frequency_hz = np.asarray(hemisphere['third_octave']['band_centers_hz'], dtype=float)
        levels_db_umapr = np.asarray(hemisphere['third_octave']['bands_db'], dtype=float)
        if levels_db_umapr.ndim != 3:
            raise ValueError('hemisphere[\'third_octave\'][\'bands_db\'] must be (Nf, Nelv, Nazi)')
    elif selected_mode == 'narrowband':
        if 'narrowband' not in hemisphere:
            raise ValueError('mode=narrowband but hemisphere does not include narrowband data')
        frequency_hz = np.asarray(hemisphere['narrowband']['frequency_hz'], dtype=float)
        psd_db_umapr = np.asarray(hemisphere['narrowband']['psd_db'], dtype=float)
        if psd_db_umapr.ndim != 3:
            raise ValueError('hemisphere[\'narrowband\'][\'psd_db\'] must be (Nf, Nelv, Nazi)')
        if frequency_hz.size < 2:
            raise ValueError('narrowband frequency_hz must have at least 2 elements')
        # Convert PSD (dB re pref^2/Hz) to approximate per-bin band level (dB re pref^2)
        # so that AAM-style OASPL integration (sum of 10^(SPL/10)) is meaningful.
        df = float(np.median(np.diff(frequency_hz)))
        psd_lin = np.power(10.0, psd_db_umapr / 10.0)
        band_power = psd_lin * df
        levels_db_umapr = 10.0 * np.log10(np.maximum(band_power, eps))
    else:
        raise ValueError('hemisphere must include third_octave or narrowband data; run '
                         'depropagate_hemisphere with third_octave=True and/or narrowband=True')

    Pq, _ = interpolate_hemisphere_power(levels_db_umapr, elv_grid_deg, azi_grid_deg, azi_q_deg, elv_q_deg)
    levels_db = np.full_like(Pq, -np.inf)
    pos = Pq > 0.0
    levels_db[pos] = 10.0 * np.log10(np.maximum(Pq[pos], eps))
    if np.isfinite(minimum_level_db):
        levels_db[levels_db < float(minimum_level_db)] = -np.inf

    return frequency_hz, levels_db


# Scattered samples to grid nodes: inverse-distance weights.

#: Defaults for depropagate_hemisphere(interpolation={...}); see adaptive_idw_weights.
#: resolution_factor 0.5 takes half the window's arc, the sample being its center.  On a
#: 2017 B407 sphere this gives about 3-5 deg at the rim, where samples crowd, 10-20 deg at
#: 10-40 deg below the horizon, and 30-50 deg overhead, where they are sparse and one window
#: spans a wide arc; 60 deg is the most a node may borrow from before it counts as a gap.
ADAPTIVE_INTERPOLATION = dict(k=8, min_mics=3, kappa=1.3, resolution_factor=0.5, max_radius_deg=60.0,
                              relax_min_mics=True, kernel='shepard', aspect=1.0,
                              shepard_floor=0.0, source_extent=0.0)


def adaptive_idw_weights(ielv, iazi, felv, fazi, mic, resolution_deg, *, k=8, min_mics=3, kappa=1.3,
                         resolution_factor=0.5, max_radius_deg=60.0, chunk=256, relax_min_mics=True,
                         return_relaxed=False, aspect=1.0, kernel='shepard', shepard_floor=0.0):
    """Shepard weights with a radius chosen per node from the samples around it.

    A fixed radius has to be as large as the sparsest part of the sphere needs, or that part
    has holes, and is then far larger than the dense part can use.  On a 2017 B407 sphere the
    nearest sample is under a degree away at the rim but up to 18 deg away overhead, so the
    25 deg radius that keeps the overhead filled smears the rim -- by +1 dB at 2-4 deg below
    the horizon and -0.3 dB at 10-20 deg against the samples themselves.

    Here each node x gets
        R(x) = max(kappa d_k(x), kappa d_M(x), resolution_factor s(x)),
    with d_k the geodesic distance to the k-th nearest sample, d_M the distance within which
    samples from min_mics distinct microphones lie, and s the median angular resolution of
    its k nearest samples (``resolution_deg``: how much of the sphere one sample's window
    smears over).  Every node within reach of k samples from min_mics microphones is
    therefore filled -- no holes -- and each value brackets the node with several
    microphones, laterally and along track, rather than extrapolating off one microphone's
    line of samples.  kappa > 1 keeps the k-th sample inside the Franke and Nielson weight,
    which falls to zero at R.  A node that would need more than max_radius_deg is a genuine
    gap in the coverage, not a hole to smooth over: its row is empty and it is reported.

    ``aspect`` > 1 stretches the neighborhood in azimuth: the geodesic distance h from the
    node to a sample is split into the elevation difference de and the rest,
    sqrt(h^2 - de^2), which is the arc along the circle of latitude for nearby points, and
    the distance is taken as sqrt(de^2 + (h^2 - de^2) / aspect^2).  A node so reaches aspect
    times as far across the sphere in azimuth as in elevation.  A pass's microphones each
    trace a line of samples nose to tail, so at fixed elevation azimuth steps from one
    microphone's line to the next, while the level's steepest change near the horizon is in
    elevation: a wide reach in azimuth averages across the lines and a narrow one in
    elevation keeps that gradient.  The distance is continuous in aspect and in the node's
    position, the geodesic distance at aspect 1 (the default), and at a pole (elevation +-90
    deg), where every sample is due north or south, the geodesic distance for any aspect.

    ``kernel`` 'shepard' (the default) is Franke and Nielson's weight, ((R - h)/(R h))^2,
    which grows without bound at a sample and so nearly interpolates: a node on one
    microphone's line follows that microphone.  'biweight', (1 - (h/R)^2)^2, is finite
    everywhere and averages its neighborhood instead.  ``shepard_floor`` keeps Shepard's
    form but stops it interpolating: each distance h is taken as sqrt(h^2 + (f R)^2) in the
    weight, f the floor as a fraction of the node's radius R.

    With ``relax_min_mics`` (the default) such a node is first retried without the
    microphone count -- its k nearest samples, from however many microphones, within the
    cap -- and is a gap only if that fails too.  The nodes it fills are the ones only one
    or two microphones see (steep to the side of a pass over a single line of them, a
    descent's far end); left empty they were filled by the sphere writer's resampling,
    which reads past the grid's edge.  ``return_relaxed`` adds their mask to the result.

    Args:
        ielv, iazi: node elevations and azimuths (deg), 1-D, same length.
        felv, fazi: sample elevations and azimuths (deg), 1-D.  Geodesic distances, so no
            +-360 copies are needed (or wanted: they would be counted as extra samples).
        mic: (N,) the microphone each sample came from.
        resolution_deg: (N,) each sample's angular resolution, deg.

    Returns:
        (weights, radius_deg, gap): a scipy.sparse CSR matrix (nodes x samples) whose rows sum
        to one (empty for gaps), each node's radius (NaN for gaps), and the gap mask.
    """
    ielv = np.asarray(ielv, dtype=float).ravel()
    iazi = np.asarray(iazi, dtype=float).ravel()
    felv = np.asarray(felv, dtype=float).ravel()
    fazi = np.asarray(fazi, dtype=float).ravel()
    mic = np.asarray(mic).ravel()
    resolution_deg = np.asarray(resolution_deg, dtype=float).ravel()
    n = felv.size
    if not (fazi.size == mic.size == resolution_deg.size == n):
        raise ValueError('felv, fazi, mic and resolution_deg must have the same length')
    k = int(min(max(k, 1), n))
    if kernel not in ('shepard', 'biweight'):
        raise ValueError("kernel must be 'shepard' or 'biweight'")
    aspect = float(aspect)
    if not aspect > 0.0:
        raise ValueError('aspect must be positive')

    def unit(elv, azi):
        e, a = np.deg2rad(elv), np.deg2rad(azi)
        return np.column_stack((np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)))

    samples = unit(felv, fazi)
    nodes = unit(ielv, iazi)
    mic_codes, mic_index = np.unique(mic, return_inverse=True)
    n_mics = mic_codes.size
    by_mic = np.argsort(mic_index, kind='stable')
    mic_starts = np.flatnonzero(np.r_[True, np.diff(mic_index[by_mic]) != 0])
    rows, cols, vals = [], [], []
    radius = np.full(ielv.size, np.nan)
    relaxed = np.zeros(ielv.size, dtype=bool)
    eps = 10.0 * np.finfo(float).eps
    for start in range(0, ielv.size, chunk):
        stop = min(start + chunk, ielv.size)
        h = np.degrees(np.arccos(np.clip(nodes[start:stop] @ samples.T, -1.0, 1.0)))
        if aspect != 1.0:
            # The elevation difference is never more than the geodesic distance; the
            # clip is for the arc cosine's roundoff.
            de2 = (ielv[start:stop, None] - felv[None, :]) ** 2
            h = np.sqrt(de2 + np.maximum(h ** 2 - de2, 0.0) / aspect ** 2)
        order = np.argsort(h, axis=1)
        d_k = np.take_along_axis(h, order[:, k - 1:k], axis=1)[:, 0]
        # Nearest sample of each microphone, then the min_mics-th nearest microphone.
        nearest_per_mic = np.minimum.reduceat(h[:, by_mic], mic_starts, axis=1)
        if n_mics >= min_mics:
            d_m = np.sort(nearest_per_mic, axis=1)[:, min_mics - 1]
        else:
            d_m = np.full(stop - start, np.inf)
        s = np.median(resolution_deg[order[:, :k]], axis=1)
        r = np.maximum(np.maximum(kappa * d_k, kappa * d_m), resolution_factor * s)
        if relax_min_mics:
            r_relaxed = np.maximum(kappa * d_k, resolution_factor * s)
            retry = ~(r <= max_radius_deg) & (r_relaxed <= max_radius_deg)
            relaxed[start:stop] = retry
            r = np.where(retry, r_relaxed, r)
        for i in range(stop - start):
            node = start + i
            hi = h[i]
            j_min = order[i, 0]
            if kernel == 'shepard' and shepard_floor <= 0.0 and hi[j_min] <= eps:
                rows.append(np.array([node]))
                cols.append(np.array([j_min]))
                vals.append(np.array([1.0]))
                radius[node] = r[i]
                continue
            if not (r[i] <= max_radius_deg):
                continue
            near = np.flatnonzero(hi < r[i])
            if kernel == 'biweight':
                m = (1.0 - (hi[near] / r[i]) ** 2) ** 2
            elif shepard_floor > 0.0:
                h_eff = np.sqrt(hi[near] ** 2 + (shepard_floor * r[i]) ** 2)
                m = ((r[i] - hi[near]) / (r[i] * h_eff)) ** 2
            else:
                m = ((r[i] - hi[near]) / (r[i] * hi[near])) ** 2
            total = m.sum()
            if not total > 0.0:
                continue
            rows.append(np.full(near.size, node))
            cols.append(near)
            vals.append(m / total)
            radius[node] = r[i]
    if rows:
        weights = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                    shape=(ielv.size, n))
    else:
        weights = sparse.csr_matrix((ielv.size, n))
    gap = ~np.isfinite(radius)
    if return_relaxed:
        return weights, radius, gap, relaxed & ~gap
    return weights, radius, gap


def shepIDW(ielv, iazi, felv, fazi, f, rmax):
    """
    Modified Shepard's Inverse Distance Weighting interpolation.

    Python version of the MATLAB function:
        function fi = shepIDW(ielv, iazi, felv, fazi, f, rmax)

    Args:
        ielv, iazi: Interpolant locations (deg). Scalars or array-like; must have same shape.
        felv, fazi: Data locations (deg). Arrays broadcastable to a common shape.
        f: Data values at (felv, fazi). Broadcastable to shape of felv/fazi.
        rmax: Maximum radius (deg) for neighborhood in weighting.

    Returns:
        numpy.ndarray of interpolated values with the same shape as `ielv` (or scalar if scalar inputs).
        NaN where no data point lies within `rmax`: there is no data there.
        (The MATLAB original gives 0, which reads as a measured zero -- zero
        energy, once the values are powers.)
    """
    felv, fazi, fvals = np.broadcast_arrays(np.asarray(felv, dtype=float), np.asarray(fazi, dtype=float),
                                            np.asarray(f, dtype=float))
    return shepIDW_apply(shepIDW_weights(ielv, iazi, felv, fazi, rmax), fvals)


def shepIDW_weights(ielv, iazi, felv, fazi, rmax):
    """
    The neighbors and weights :func:`shepIDW` uses at each interpolant.

    They depend only on the geometry, so a caller interpolating several fields
    sampled at the same points computes them once and hands them to
    :func:`shepIDW_apply` for each field.

    Args:
        ielv, iazi, felv, fazi, rmax: as for :func:`shepIDW`.

    Returns:
        (shape, neighbors): the interpolant shape, and for each interpolant
        (flattened) a pair (flat indices into felv/fazi, their weights), both
        empty where no data point lies within ``rmax``.
    """
    felv = np.asarray(felv, dtype=float)
    fazi = np.asarray(fazi, dtype=float)
    felv, fazi = np.broadcast_arrays(felv, fazi)

    # Prepare interpolant inputs and preserve original shape
    ielv_arr = np.asarray(ielv, dtype=float)
    iazi_arr = np.asarray(iazi, dtype=float)
    if ielv_arr.shape != iazi_arr.shape:
        ielv_arr, iazi_arr = np.broadcast_arrays(ielv_arr, iazi_arr)
    ielv_flat = ielv_arr.ravel()
    iazi_flat = iazi_arr.ravel()

    # Franke & Nielson weights ((rmax - h) / (rmax h))^2 for a block of
    # interpolants at a time, by broadcasting.  The arithmetic is geodist's,
    # element by element, with the sines and cosines of each point's elevation
    # taken once rather than once per pair, and the arc cosine only where a
    # pair can be within rmax (the rest get no weight either way); so h is
    # geodist's, to the bit.
    rmax = float(rmax)
    eps = np.finfo(float).eps
    r_elv1, r_azi1 = np.deg2rad(ielv_flat), np.deg2rad(iazi_flat)
    r_elv2, r_azi2 = np.deg2rad(felv.ravel()), np.deg2rad(fazi.ravel())
    sin1, cos1 = np.sin(r_elv1)[:, None], np.cos(r_elv1)[:, None]
    sin2, cos2 = np.sin(r_elv2)[None, :], np.cos(r_elv2)[None, :]
    # Below this the arc is beyond rmax, with room for the arc cosine's roundoff
    arcs_floor = np.cos(np.radians(rmax)) - 1e-9
    block = max(1, 2 ** 22 // max(r_elv2.size, 1))
    neighbors = []
    for start in range(0, ielv_flat.size, block):
        rows = slice(start, start + block)
        arcs = sin1[rows] * sin2 + cos1[rows] * cos2 * np.cos(r_azi2[None, :] - r_azi1[rows, None])
        arcs = np.clip(arcs, -1.0, 1.0)
        candidate = arcs >= arcs_floor
        hi = np.full_like(arcs, np.inf)
        hi[candidate] = np.degrees(np.arccos(arcs[candidate]))
        m = np.zeros_like(hi)
        mask = hi <= rmax
        with np.errstate(divide='ignore'):
            m[mask] = ((rmax - hi[mask]) / (rmax * hi[mask])) ** 2
        exact = np.any(hi <= 10.0 * eps, axis=1)
        nearest = np.argmin(hi, axis=1)
        for row in range(hi.shape[0]):
            if exact[row]:
                # Threshold exact data points to avoid division by zero
                neighbors.append((nearest[row:row + 1], np.ones(1)))
                continue
            # Only the neighbors: 0 * NaN is NaN, so one bad sample anywhere
            # would otherwise poison every node, however far away.
            near = np.flatnonzero(m[row] > 0.0)
            neighbors.append((near, m[row, near] / m[row].sum()))
    return ielv_arr.shape, neighbors


def shepIDW_apply(weights, f):
    """
    :func:`shepIDW` of the field ``f`` with weights from :func:`shepIDW_weights`.

    Args:
        weights: what :func:`shepIDW_weights` returned.
        f: data values, the shape of felv/fazi given to :func:`shepIDW_weights`.

    Returns:
        as for :func:`shepIDW`.  A NaN value counts as missing: the node's
        weights are renormalized over its other neighbors, and a node whose
        neighbors are all missing is NaN.
    """
    shape, neighbors = weights
    fvals = np.asarray(f, dtype=float).ravel()
    if np.isnan(fvals).any():
        def node(near, w):
            kept = ~np.isnan(fvals[near])
            if kept.all():
                return np.sum(fvals[near] * w)
            return np.sum(fvals[near][kept] * w[kept]) / np.sum(w[kept]) if kept.any() else np.nan
    else:
        def node(near, w):
            return np.sum(fvals[near] * w)
    fi = np.array([node(near, w) if near.size else np.nan for near, w in neighbors],
                  dtype=float).reshape(shape)
    if fi.ndim == 0:
        return float(fi)
    return fi
