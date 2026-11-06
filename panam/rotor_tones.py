"""Rotor-locked tones from measured rotor angles: the measured side of format 2 (pure-tone) databases.

A multirotor's rotors turn at slightly different, wandering speeds, so their shaft orders sit a few hertz apart and
keep crossing.  Where each rotor's angle is logged, every shaft order of every rotor can be extracted with its phase
locked to that rotor's blades: the microphone signal is fitted as

    p(t) = sum over rotors r and orders h of Re{A_rh(t) exp(i h psi_r(t_e))} + residual,

with psi_r the rotor's azimuth in its own sense of rotation (increasing in time) at the emission time t_e of the
sound reaching the microphone at t from that rotor's hub, and A_rh a slowly varying complex amplitude.  The fit is
the Vold-Kalman filter (:func:`panam.vold_kalman_filter.vold_kalman_filter`) with the orders' phases given rather
than integrated from their frequencies, all rotors and orders at once, so orders of different rotors passing
through one frequency are shared out by least squares instead of leaking into each other.

The convention is e^{+i omega t}: A_rh's argument is the cosine phase of order h at emission when the rotor's
azimuth is zero, which is the format 2 ``phase`` with ``azimuth`` 0 (``docs/file_formats.md``, "Format 2:
pure-tone sources").  PANAM's ground models use e^{-i omega t}; :func:`plate_tone_response` returns their response
conjugated, in this module's convention.

Times in s, angles in rad unless named ``_deg``, lengths in m unless named ``_ft``.
"""
import numpy as np

from panam import axisymmetric_bem as ab
from panam import ground_plane as gp
from panam.geometry import art2umapr
from panam.gridding import adaptive_idw_weights
from panam.ground_effect import spherical_reflection_coefficient
from panam.vold_kalman_filter import vold_kalman_filter


def rotor_azimuth(time, angle, speed, *, aligned_angle=0.0, step_s=1e-3, window_s=0.25, min_coherence=0.3):
    """A rotor's continuous azimuth in its own sense of rotation, from a logged angle and speed.

    Args:
        time: the log's sample times, s, increasing.
        angle: the logged mechanical angle, rad, wrapped any way; it advances with the signed ``speed``.  Repeated
            values (a controller that updates the angle less often than the log's rows) are dropped.
        speed: the logged shaft speed, rad/s, signed by the sense of rotation the angle advances in.  Samples of the
            other sign are taken as they are if they are small (a stopped rotor's noise, under a tenth of the median
            speed).
        aligned_angle: the logged angle at which the blades are at the azimuth's zero (a prop set's alignment with
            its arm, say), rad.
        step_s: the output grid's spacing, s.
        window_s: the window over which the angle's offset from the speed integral is averaged, s.
        min_coherence: windows whose offsets scatter more than this mean resultant length allows are skipped.

    The azimuth is the integral of the speed plus a slow correction: the circular mean, in ``window_s`` windows,
    of the logged angle less that integral, unwrapped and interpolated.  Single logged angles of a fast rotor
    scatter tens of degrees about it; the speed carries the fast part and the angle only the drift.

    Returns (grid, azimuth, diagnostics): the grid's times (s, every ``step_s`` over the log), the azimuth there
    (rad, increasing whatever the sense: the signed angle less the aligned angle, times the sign of the median
    speed), and {'sense': +1 or -1 (the sign of the median speed), 'window_coherence': the median mean resultant
    length of the windows, 'scatter_deg': the rms of the samples about the result}.  Raises ValueError for a log
    with no usable window or a speed that turns the other way.
    """
    t = np.asarray(time, dtype=float)
    a = np.asarray(angle, dtype=float)
    w = np.asarray(speed, dtype=float)
    if t.ndim != 1 or a.shape != t.shape or w.shape != t.shape or t.size < 2:
        raise ValueError('time, angle and speed must be one-dimensional and of one length (at least 2)')
    if np.any(np.diff(t) <= 0):
        raise ValueError('time must increase')
    finite_w = np.isfinite(w)
    if not finite_w.any():
        raise ValueError('the speed has no finite sample')
    typical = float(np.median(w[finite_w]))
    sense = 1.0 if typical > 0 else -1.0
    if np.any(sense * w[finite_w] < -0.1 * abs(typical)):
        raise ValueError('the speed turns the other way (by more than a tenth of its median): a rotor turns one way')
    keep = np.isfinite(a) & np.r_[True, np.diff(a) != 0]
    ta, aa = t[keep], a[keep]
    grid = np.arange(t[0], t[-1], step_s)
    wg = np.interp(grid, t[finite_w], w[finite_w])
    integral = np.concatenate([[0.0], np.cumsum(0.5 * (wg[1:] + wg[:-1]) * np.diff(grid))])
    residual = np.exp(1j * (aa - np.interp(ta, grid, integral)))
    edges = np.arange(ta[0], ta[-1] + window_s, window_s)
    index = np.digitize(ta, edges) - 1
    sums = np.bincount(index, weights=residual.real, minlength=edges.size) \
        + 1j * np.bincount(index, weights=residual.imag, minlength=edges.size)
    counts = np.bincount(index, minlength=edges.size)
    with np.errstate(invalid='ignore', divide='ignore'):
        means = (sums / counts)[:edges.size - 1]
    centers = 0.5 * (edges[1:] + edges[:-1])
    ok = (counts[:edges.size - 1] > 0) & (np.abs(means) > min_coherence)
    if not ok.any():
        raise ValueError(f'no {window_s} s window of the angle is coherent with the speed integral '
                         f'(mean resultant length above {min_coherence})')
    correction = np.unwrap(np.angle(means[ok]))
    signed = integral + np.interp(grid, centers[ok], correction)
    scatter = np.angle(residual * np.exp(-1j * np.interp(ta, centers[ok], correction)))
    azimuth = sense * (signed - aligned_angle)
    return grid, azimuth, {'sense': int(sense), 'window_coherence': float(np.median(np.abs(means[ok]))),
                           'scatter_deg': float(np.degrees(np.sqrt(np.mean(scatter ** 2))))}


def emission_azimuth(reception_times, hub_position, receiver, sound_speed, azimuth_time, azimuth):
    """Each rotor's azimuth at the emission time of the sound reaching a fixed receiver.

    Args:
        reception_times: (n,) s.
        hub_position: the hub's position as a function of time: a callable taking (n,) times and returning (n, 3)
            positions, m.
        receiver: (3,) m, in the frame of ``hub_position``.
        sound_speed: m/s (straight rays in still air).
        azimuth_time, azimuth: the rotor's azimuth track (:func:`rotor_azimuth`), s and rad.

    Returns (t_e, azimuth_e, distance, direction): the emission times (s), the azimuth there (rad), the hub to
    receiver distance at emission (m) and the unit vector from the hub at emission toward the receiver, (n, 3).
    """
    t_e, x_e = gp.emission_times(hub_position, reception_times, receiver, sound_speed)
    d = np.asarray(receiver, dtype=float)[None, :] - x_e
    distance = np.linalg.norm(d, axis=1)
    return t_e, np.interp(t_e, azimuth_time, azimuth), distance, d / distance[:, None]


def rotor_locked_tones(pressure, fs, azimuths, harmonics, bandwidth_hz, *, p=2, method='vkf', group_gap_hz=None,
                       sweeps=3):
    """Every rotor's shaft orders, phase locked to its own azimuth, from one microphone.

    Args:
        pressure: (n,) Pa, sampled at ``fs`` (Hz); decimate first so that the highest order is below fs / 2 with
            room to spare (the filter's cost grows with n).
        azimuths: (n, n_rotors) rad, each rotor's azimuth at the emission time of each sample
            (:func:`emission_azimuth`), in its own sense of rotation.
        harmonics: the shaft orders to extract (positive integers), the same for every rotor.
        bandwidth_hz: the full -3 dB band of each order's envelope, Hz: narrower rejects more noise, wider follows a
            changing direction sooner (the envelope's 10-90 % rise takes about 0.7 / bandwidth s).
        p: the Vold-Kalman filter's structural order.
        method: ``'vkf'`` (the multi-order Vold-Kalman filter, every rotor and order at once) or ``'demodulate'``
            (each order demodulated by its own phasor and averaged over 1 / bandwidth_hz: no sharing out where two
            orders cross; a check).
        group_gap_hz: orders whose frequency ranges over the record come within this of each other are solved
            together (default 4 bandwidths).  One solve over every rotor and order costs about the cube of their
            number; the same order of six rotors, wandering a few percent apart, is one small group.
        sweeps: the groups are solved in turn, each on the pressure less the other groups' current fit, this many
            times over (block Gauss-Seidel on the joint least-squares problem).  One sweep leaves the other groups'
            tones in each group's data, which costs accuracy (a few hundredths of a pascal on tones of tenths
            beside them); three reach the joint solve's result.

    Returns (amplitude, residual): (n, n_rotors, n_harmonics) complex Pa with
    pressure ~ sum Re{amplitude exp(i h azimuth)}, and the pressure less that sum (Pa, (n,)): the broadband and the
    orders not extracted.
    """
    x = np.asarray(pressure, dtype=float)
    psi = np.asarray(azimuths, dtype=float)
    if psi.ndim == 1:
        psi = psi[:, None]
    h = np.atleast_1d(np.asarray(harmonics))
    if x.ndim != 1 or psi.shape[0] != x.size:
        raise ValueError('pressure must be (n,) and azimuths (n, n_rotors)')
    if h.ndim != 1 or h.size == 0 or not np.all(h == np.rint(h)) or np.any(h < 1):
        raise ValueError('harmonics must be positive whole shaft orders')
    n_rotors = psi.shape[1]
    phase = (psi[:, :, None] * h[None, None, :]).reshape(x.size, n_rotors * h.size)
    if method == 'vkf':
        gap = 4.0 * float(np.max(bandwidth_hz)) if group_gap_hz is None else float(group_gap_hz)
        y = np.zeros(phase.shape, dtype=complex)
        phasor = np.exp(1j * phase)
        groups = _frequency_groups(phase, fs, gap)
        fit = np.zeros((len(groups), x.size))
        for _ in range(sweeps if len(groups) > 1 else 1):
            for g, members in enumerate(groups):
                others = fit.sum(axis=0) - fit[g]
                y[:, members], _, _ = vold_kalman_filter(x - others, None, fs, bandwidth_hz, p,
                                                         phase=phase[:, members])
                fit[g] = np.real(y[:, members] * phasor[:, members]).sum(axis=1)
    elif method == 'demodulate':
        phasor = np.exp(1j * phase)
        n_avg = max(1, int(round(fs / bandwidth_hz)))
        kernel = np.ones(n_avg) / n_avg
        z = 2.0 * x[:, None] * np.conj(phasor)
        y = np.stack([np.convolve(z[:, k], kernel, mode='same') for k in range(z.shape[1])], axis=1)
    else:
        raise ValueError(f"method must be 'vkf' or 'demodulate', not {method!r}")
    residual = x - np.real(y * phasor).sum(axis=1)
    return y.reshape(x.size, n_rotors, h.size), residual


def _frequency_groups(phase, fs, gap_hz):
    """Orders (columns of ``phase``, rad) grouped so that orders whose frequency ranges (the 1st to 99th percentile
    of the phase's rate over the record, Hz) come within ``gap_hz`` of each other share a group."""
    rate = np.abs(np.diff(phase, axis=0)) * fs / (2.0 * np.pi)
    low, high = np.percentile(rate, 1, axis=0) - gap_hz / 2, np.percentile(rate, 99, axis=0) + gap_hz / 2
    order = np.argsort(low)
    groups, current, reach = [], [order[0]], high[order[0]]
    for k in order[1:]:
        if low[k] <= reach:
            current.append(k)
            reach = max(reach, high[k])
        else:
            groups.append(np.array(current))
            current, reach = [k], high[k]
    groups.append(np.array(current))
    return groups


def tone_noise(residual, fs, frequency_hz, bandwidth_hz, segment_s=4.0, percentile=75.0):
    """The noise an extracted order carries: the rms of |amplitude| (:func:`rotor_locked_tones`) that broadband
    alone would give at each frequency, Pa.

    sqrt(2 G(f) B), with B the envelope's bandwidth (the filter passes a band B wide about the order, and the
    amplitude is twice the envelope) and G the residual's one-sided power spectral density near f (Welch,
    ``segment_s`` segments).  The residual is notched at every extracted order, its own included, so G is taken as
    the ``percentile`` of the density over |f' - f| <= max(15 Hz, 0.15 f), where the notches are a minority.
    Compare it with |amplitude| to gate tones that are not above the noise.

    Args:
        residual: (n,) Pa, the pressure less the extracted orders.
        fs: Hz.
        frequency_hz: the received frequencies to evaluate at, any shape.
        bandwidth_hz: the extraction's bandwidth, Hz.
    """
    from scipy import signal as sg
    x = np.asarray(residual, dtype=float)
    f, g = sg.welch(x, fs, nperseg=min(x.size, int(round(segment_s * fs))))
    wanted = np.asarray(frequency_hz, dtype=float)
    centers = np.linspace(max(f[1], np.nanmin(wanted) * 0.9), np.nanmax(wanted) * 1.1, 200)
    level = np.empty(centers.size)
    for i, fc in enumerate(centers):
        near = np.abs(f - fc) <= max(15.0, 0.15 * fc)
        level[i] = np.percentile(g[near], percentile) if near.any() else np.nan
    return np.sqrt(2.0 * np.interp(wanted, centers, level) * float(bandwidth_hz))


def received_shaft_rate(azimuth_e, fs):
    """The shaft rate as heard, rad/s: the time derivative of the emission-time azimuth (n, ...) sampled at fs (Hz).
    Order h is received at h times this over 2 pi, Hz (the Doppler shift included)."""
    return np.gradient(np.asarray(azimuth_e, dtype=float), 1.0 / fs, axis=0)


def depropagate_tones(amplitude, frequency_hz, distance_m, sphere_radius_m, atmosphere=None, response=None,
                      lossless=True):
    """Tones heard at a receiver, as they would be at the sphere radius from the hub in free field.

    Args:
        amplitude: complex Pa, (n, ...): what :func:`rotor_locked_tones` returns, say.
        frequency_hz: each tone's received frequency, Hz, broadcastable to ``amplitude``.
        distance_m: (n,) the hub-to-receiver distance at emission, m.
        sphere_radius_m: the sphere's radius, m.
        atmosphere: a :class:`panam.acoustics.atmosphere.Atmosphere` for ISO 9613-1 absorption at each tone's
            frequency, or None for none.
        response: the receiver's complex response re the direct free-field wave (a plate on the ground,
            :func:`plate_tone_response`), broadcastable to ``amplitude``, or None for a free-field receiver.
        lossless: undo the absorption over the whole path (True: a lossless sphere, ``sphere_levels_lossless``,
            which NICEOPS absorbs from the hub) or only beyond the sphere radius (False).

    Returns amplitude x (distance / radius) x 10^(alpha L / 20) / response: spherical spreading and absorption
    undone, the receiver's response divided out (its phase too), and the emission phase kept.
    """
    a = np.asarray(amplitude)
    r = np.asarray(distance_m, dtype=float).reshape((-1,) + (1,) * (a.ndim - 1))
    gain = r / float(sphere_radius_m)
    if atmosphere is not None:
        path = r if lossless else r - float(sphere_radius_m)
        gain = gain * 10.0 ** (atmosphere.attenuation_coefficient(np.asarray(frequency_hz, dtype=float))
                               * path / 20.0)
    out = a * gain
    return out if response is None else out / response


def plate_tone_table(frequencies_hz, sound_speed_ft_s, flow_resistance=gp.FLOW_RESISTANCE, ground=None, chunk=8,
                     **options):
    """A plate-on-the-ground table for single frequencies (tones), not bands: :func:`axisymmetric_bem.table` with
    one sub-frequency per entry, so that ``frequencies`` are exactly ``frequencies_hz`` (sorted).  Lengths in ft,
    speed in ft/s, as there; ``options`` as there (the plate and the microphone).  Space the frequencies closely
    enough for linear interpolation between them (:func:`plate_tone_response`): the plate's response turns over a
    few hundred hertz for a plate a few tenths of a meter across.  Computed ``chunk`` frequencies at a time: the
    solver's working arrays grow with the frequencies it is given at once (all 240 of a 5 Hz grid to 1.2 kHz took
    over 15 GB)."""
    frequencies = np.unique(np.asarray(frequencies_hz, dtype=float))
    parts = [ab.table(frequencies[i:i + chunk], sound_speed_ft_s, flow_resistance, ground, sub_bands=1, **options)
             for i in range(0, frequencies.size, chunk)]
    table = dict(parts[0])
    for key in ('frequencies', 'P_d', 'P_r'):
        table[key] = np.concatenate([part[key] for part in parts], axis=0)
    table['bands'] = frequencies
    return table


def plate_tone_response(table, frequency_hz, source_height_ft, ground_distance_ft, sound_speed_ft_s,
                        source_dx_ft=None, source_dy_ft=None, mirror_y=False):
    """The complex response, re the direct free-field wave, of a microphone on a plate lying on the ground, at each
    frame's tone frequency and direction, in this module's e^{+i omega t} convention.

    P = P_d + Q P_r as :func:`axisymmetric_bem.board_level` forms it (Q, the spherical-wave reflection coefficient of
    the table's ground, at the plate's top from each frame's geometry; P_d and P_r bilinear in elevation and
    azimuth as :func:`ground_plane.table_frames` takes them, and linear in frequency between the table's), then
    conjugated: PANAM's ground models are e^{-i omega t}.  So |P|^2 is board_level's at a table frequency.

    Args:
        table: :func:`plate_tone_table`.
        frequency_hz, source_height_ft, ground_distance_ft: (n,) per frame; the source's height above the ground
            and horizontal distance from the microphone, ft.
        sound_speed_ft_s: ft/s.
        source_dx_ft, source_dy_ft, mirror_y: the source's offset from the microphone in the plate's frame, and the
            microphone on the plate's other side, as :func:`ground_plane.table_frames`.

    Returns (n,) complex; NaN outside the table's frequencies.
    """
    f = np.atleast_1d(np.asarray(frequency_hz, dtype=float))
    hs = np.broadcast_to(np.asarray(source_height_ft, dtype=float), f.shape)
    d2 = np.broadcast_to(np.asarray(ground_distance_ft, dtype=float), f.shape)
    elevation = np.degrees(np.arctan2(hs, d2))
    if source_dx_ft is None:
        azimuth = np.zeros_like(f)
    else:
        dx = np.broadcast_to(np.asarray(source_dx_ft, dtype=float), f.shape)
        dy = np.broadcast_to(np.asarray(source_dy_ft, dtype=float), f.shape)
        azimuth = np.mod(np.degrees(np.arctan2(-dy, -dx)), 360.0)
    if mirror_y:
        azimuth = np.mod(-azimuth, 360.0)
    t_el = np.asarray(table['elevations'], dtype=float)
    t_az = np.asarray(table['azimuths'], dtype=float)
    t_f = np.asarray(table['frequencies'], dtype=float)
    i, ti = gp._grid_interval(t_el, np.clip(elevation, t_el[0], t_el[-1]))
    j, tj = gp._grid_interval(np.append(t_az, t_az[0] + 360.0), t_az[0] + np.mod(azimuth - t_az[0], 360.0))
    j1 = (j + 1) % t_az.size
    inside = (f >= t_f[0]) & (f <= t_f[-1])
    k, tk = gp._grid_interval(t_f, np.clip(f, t_f[0], t_f[-1]))
    values = []
    for key in ('P_d', 'P_r'):
        grid = np.asarray(table[key])
        value = np.zeros(f.shape, dtype=complex)
        for wf, kk in ((1.0 - tk, k), (tk, k + 1)):
            for w, ii, jj in (((1.0 - ti) * (1.0 - tj), i, j), (ti * (1.0 - tj), i + 1, j),
                              ((1.0 - ti) * tj, i, j1), (ti * tj, i + 1, j1)):
                value += wf * w * grid[kk, ii, jj]
        values.append(value)
    height = table['thickness']
    image_range = np.hypot(d2, hs + height)
    q = spherical_reflection_coefficient(
        (hs + height) / image_range, image_range, f, sound_speed_ft_s, table['flow_resistance'],
        admittance=gp._ground_admittance(f, sound_speed_ft_s, table['flow_resistance'], table.get('ground')))
    response = np.conj(values[0] + q * values[1])
    return np.where(inside, response, np.nan + 0j)


P_REF = 2e-5


def grid_tones(phi_deg, theta_deg, amplitude, mic, resolution_deg, grid_phi_deg, grid_theta_deg, **idw):
    """Tones sampled over directions, put on a sphere's grid with their phase.

    Args:
        phi_deg, theta_deg: (n,) each sample's direction, ART angles (deg): phi 0 below, positive to starboard;
            theta 0 at the nose.
        amplitude: (n, n_tones) complex Pa (peak, e^{+i omega t}) at the sphere radius; NaN where a sample has no
            value for a tone.
        mic: (n,) the microphone each sample came from (the gridding brackets a node with several).
        resolution_deg: (n,) how far one sample smears over the sphere, deg.
        grid_phi_deg, grid_theta_deg: the grid's axes, ART, deg.
        idw: passed to :func:`panam.gridding.adaptive_idw_weights` (k, min_mics, kappa, max_radius_deg, ...).

    Each node takes the adaptive Shepard weights of the broadband build.  Its level is the weighted mean square
    pressure (an energy mean, so a scatter of phases between samples does not pull it down), and its phase is the
    argument of the weighted complex sum.  The coherence |sum w A| / sqrt(sum w |A|^2), 1 where the samples agree
    in phase, says how far that phase can be trusted.

    Returns a dict over (n_phi, n_theta, n_tones): 'level_db' (dB re 20 uPa, the tone's rms; -inf at gaps),
    'phase_deg' (NaN at gaps), 'coherence' (NaN at gaps), and 'gap' (n_phi, n_theta).
    """
    a = np.asarray(amplitude)
    if a.ndim == 1:
        a = a[:, None]
    gp_phi, gp_theta = np.meshgrid(np.asarray(grid_phi_deg, float), np.asarray(grid_theta_deg, float),
                                   indexing='ij')
    node_az, node_el = art2umapr(np.radians(gp_phi.ravel()), np.radians(gp_theta.ravel()))
    az, el = art2umapr(np.radians(np.asarray(phi_deg, float)), np.radians(np.asarray(theta_deg, float)))
    weights, _, gap = adaptive_idw_weights(np.degrees(node_el), np.degrees(node_az), np.degrees(el), np.degrees(az),
                                           mic, resolution_deg, **idw)
    valid = np.isfinite(a)
    filled = np.where(valid, a, 0.0)
    norm = weights @ valid.astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        power = (weights @ (np.abs(filled) ** 2 / 2.0)) / norm
        mean = (weights @ filled) / norm
        coherence = np.abs(mean) / np.sqrt(2.0 * power)
        level = 10.0 * np.log10(power / P_REF ** 2)
    empty = ~(norm > 0)
    level[empty] = -np.inf
    phase = np.degrees(np.angle(mean))
    phase[empty] = np.nan
    coherence[empty] = np.nan
    shape = gp_phi.shape + (a.shape[1],)
    return {'level_db': level.reshape(shape), 'phase_deg': phase.reshape(shape),
            'coherence': coherence.reshape(shape), 'gap': gap.reshape(gp_phi.shape)}


def clock_offsets(entries, *, max_offset_s=0.15, step_s=2e-5, sweeps=4, reference=None):
    """Each run's timing offset between its rotor angles and its sound, from where runs see the same directions.

    A run whose rotor angles are late by delta against its microphones hears every tone with its phase advanced by
    k delta, k the tone's own-sense angular rate (harmonic x |shaft rate|, rad/s).  Runs flown at one condition see
    the same directivity, so delta is what brings their tones into phase.

    Args:
        entries: a list of dicts, one per run and condition: 'run' (a label), 'condition' (a label: only entries of
            one condition are compared), 'amplitude' (complex (n_cells, n_tones) on a grid shared by the entries of
            that condition, NaN where the run has no value), 'rate' ((n_tones,) rad/s, each tone's k for that run).
        max_offset_s, step_s: the range and step searched, s.  A whole revolution of one rotor is told from another
            only by the rotors' different speeds, so the range may span several revolutions.
        sweeps: passes over the runs once all are placed (each run's offset refitted against all the others').
            Runs are first placed one at a time, each fitted against those already placed only.
        reference: the run held at 0 in each connected set of runs (runs linked by a shared condition); default
            the run with most entries.  The others join one at a time, the one sharing the most cells and tones with
            those placed first, each fitted against those placed; then every run is refitted against all.

    The objective is the real part of sum over pairs of entries of one condition, over cells and tones, of
    A_a conj(A_b) exp(-i k (delta_a - delta_b)): their tones' cross power once both are corrected.

    Returns (offsets, components): {run: delta (s)}, the reference's 0 in each set, and a list of the sets (lists
    of runs), whose offsets are relative to their own reference only.  A run that shares no cell with any other
    keeps 0 (it is not constrained).
    """
    runs = sorted({e['run'] for e in entries}, key=str)
    by_condition = {}
    for e in entries:
        by_condition.setdefault(e['condition'], []).append(e)
    # connected sets of runs
    parent = {r: r for r in runs}

    def find(r):
        while parent[r] != r:
            parent[r] = parent[parent[r]]
            r = parent[r]
        return r
    for group in by_condition.values():
        for e in group[1:]:
            parent[find(e['run'])] = find(group[0]['run'])
    components = {}
    for r in runs:
        components.setdefault(find(r), []).append(r)
    counts = {r: sum(e['run'] == r for e in entries) for r in runs}
    fixed = set()
    for members in components.values():
        if reference is not None and reference in members:
            fixed.add(reference)
        else:
            fixed.add(max(members, key=lambda r: (counts[r], str(r))))
    offsets = {r: 0.0 for r in runs}
    trial = np.arange(-max_offset_s, max_offset_s + step_s / 2, step_s)

    def fit(r, against):
        """The offset of run r that best matches the runs in `against` at their current offsets, or None."""
        terms_c, terms_k = [], []
        for group in by_condition.values():
            others = [e for e in group if e['run'] in against and e['run'] != r]
            for e in (e for e in group if e['run'] == r):
                a = np.nan_to_num(np.asarray(e['amplitude']))
                for o in others:
                    b = np.nan_to_num(np.asarray(o['amplitude'])) * np.exp(
                        -1j * np.asarray(o['rate'], float) * offsets[o['run']])
                    terms_c.append((a * np.conj(b)).sum(axis=0))
                    terms_k.append(np.asarray(e['rate'], float))
        if not terms_c:
            return None
        c, k = np.concatenate(terms_c), np.concatenate(terms_k)
        if not np.any(c):
            return None                                # nothing shared: the offset is not constrained
        objective = np.real(np.exp(-1j * np.outer(trial, k)) @ c)
        best = int(np.argmax(objective))
        delta = trial[best]
        if 0 < best < trial.size - 1:                  # parabolic refinement
            y0, y1, y2 = objective[best - 1:best + 2]
            denominator = y0 - 2 * y1 + y2
            if denominator < 0:
                delta += 0.5 * step_s * (y0 - y2) / denominator
        return float(delta)

    def overlap(r, against):
        """How much run r shares with the runs in `against`: cells and tones where both have a value."""
        total = 0
        for group in by_condition.values():
            for e in (e for e in group if e['run'] == r):
                have = np.isfinite(np.asarray(e['amplitude']))
                total += sum(int(np.sum(have & np.isfinite(np.asarray(o['amplitude'])))) for o in group
                             if o['run'] in against and o['run'] != r)
        return total

    # Runs join one at a time, each fitted against the runs already placed (the one sharing the most cells and tones
    # with them first, so that a run is placed through its strongest link),
    # so that no run is fitted against another's unknown offset; then sweeps refine every run against all others.
    placed = set(fixed)
    while len(placed) < len(runs):
        candidates = [r for r in runs if r not in placed]
        r = max(candidates, key=lambda r: (overlap(r, placed), str(r)))
        delta = fit(r, placed)
        offsets[r] = 0.0 if delta is None else delta
        placed.add(r)
    for _ in range(sweeps):
        for r in runs:
            if r not in fixed:
                delta = fit(r, set(runs))
                if delta is not None:
                    offsets[r] = delta
    return offsets, [sorted(m, key=str) for m in components.values()]


def axis_angles(phi_deg, theta_deg):
    """ART directions (phi from below toward starboard, theta from the nose; deg) as angles about a vertical shaft
    in the sphere frame: the polar angle from the down axis and the azimuth about it from the nose toward starboard,
    rad."""
    p, t = np.radians(np.asarray(phi_deg, float)), np.radians(np.asarray(theta_deg, float))
    forward, starboard, down = np.cos(t), np.sin(t) * np.sin(p), np.sin(t) * np.cos(p)
    return np.arccos(np.clip(down, -1.0, 1.0)), np.arctan2(starboard, forward)


def harmonic_basis(polar, azimuth, degree):
    """Spherical harmonics Y_l^k (orthonormal, complex), l <= degree, at (polar, azimuth) rad: (n, (degree + 1)^2),
    columns in (l, k) order, k from -l to l."""
    from scipy import special
    columns = []
    for l in range(degree + 1):
        for k in range(-l, l + 1):
            if hasattr(special, 'sph_harm_y'):
                columns.append(special.sph_harm_y(l, k, polar, azimuth))
            else:                                                       # scipy < 1.15
                columns.append(special.sph_harm(k, l, azimuth, polar))
    return np.stack(columns, axis=-1)


def fit_directivities(tones, *, degree=4, smoothing=1e-2, max_offset_s=0.01, step_s=2e-5, iterations=4,
                      reference=None, offsets=None, robust=4.0):
    """Smooth complex directivities of measured tones, with each run's timing offset fitted against them.

    Each tone (a rotor's shaft order) is a sum of spherical harmonics about the rotor's shaft (taken as the sphere
    frame's vertical) up to ``degree``, fitted by weighted least squares to its samples, with a penalty
    ``smoothing`` x l^2 (l + 1)^2 (relative to the mean diagonal of the normal equations) on each coefficient that
    keeps it smooth where the samples are sparse.  A run's timing offset delta advances every one of its tones by k delta (k the
    tone's own-sense angular rate, rad/s), so the runs' offsets are shared by all the tones: they are refitted, each
    run against the current directivities, between fits (``iterations`` times), the reference run held at its
    given offset.

    Args:
        tones: a list of dicts, one per tone: 'phi', 'theta' (deg, ART, (n,)), 'amplitude' (complex (n,), the
            depropagated samples), 'weight' ((n,), e.g. the samples' signal-to-noise power, capped), 'run' ((n,)
            labels), 'rate' ((n,) rad/s: harmonic x |shaft rate| at each sample).
        degree, smoothing: the expansion's degree and its smoothness penalty.
        max_offset_s, step_s: how far each run's offset may move from its starting value, and the search step, s.
        iterations: fit, refit offsets, fit again ... this many times.
        reference: the run held fixed (default the run with the most weight).
        offsets: starting offsets {run: s} (default 0): from :func:`clock_offsets`, say.
        robust: after each fit, a run's samples of a tone are weighted by its consistency with the fit to this
            power (0: not at all), so that one run whose tone does not agree with the others' (a different angle
            reference, a different source) does not pull the fit or the other runs' offsets.

    Returns a dict: 'coefficients' (list, one (n_basis,) complex array per tone), 'degree', 'offsets' {run: s},
    'reference', and 'consistency' {(tone index, run): Re(sum w A conj(M)) / sqrt(sum w |A|^2 sum w |M|^2)}, 1 where
    a run's samples of a tone agree in phase with the fit (corrected by its offset), low or negative where its phase
    is not the others' (a whole-run rotation included), and 'gain' {(tone, run):
    sum w |A|^2 / sum w |M|^2}, its level against the fit's.
    """
    runs = sorted({r for t in tones for r in np.asarray(t['run'])}, key=str)
    if reference is None:
        mass = {r: sum(float(np.sum(np.asarray(t['weight'])[np.asarray(t['run']) == r])) for t in tones) for r in runs}
        reference = max(runs, key=lambda r: (mass[r], str(r)))
    offsets = {r: float((offsets or {}).get(r, 0.0)) for r in runs}
    start = dict(offsets)
    prepared = []
    for t in tones:
        polar, azimuth = axis_angles(t['phi'], t['theta'])
        a = np.asarray(t['amplitude'], complex)
        w = np.asarray(t['weight'], float)
        ok = np.isfinite(a) & np.isfinite(w) & (w > 0)
        prepared.append({'basis': harmonic_basis(polar[ok], azimuth[ok], degree), 'a': a[ok], 'w': w[ok],
                         'run': np.asarray(t['run'])[ok], 'rate': np.asarray(t['rate'], float)[ok]})
    ls = np.concatenate([np.full(2 * l + 1, l) for l in range(degree + 1)])
    penalty = (ls * (ls + 1.0)) ** 2

    def corrected(p):
        shift = np.array([offsets[r] for r in p['run']])
        return p['a'] * np.exp(-1j * p['rate'] * shift)

    def agreement(i, p):
        """{run: consistency} of tone i's runs with its current fit."""
        model = p['basis'] @ coefficients[i]
        a = corrected(p)
        out = {}
        for r in np.unique(p['run']):
            m = p['run'] == r
            w = p['w'][m]
            den = np.sqrt(np.sum(w * np.abs(a[m]) ** 2) * np.sum(w * np.abs(model[m]) ** 2))
            out[r] = float(np.real(np.sum(w * a[m] * np.conj(model[m]))) / den) if den > 0 else float('nan')
        return out

    def solve(p, keep=None):
        """Tone p's coefficients from its samples (those in `keep` only, when given), weighted by w_fit."""
        b, w = p['basis'], p['w_fit']
        a = corrected(p)
        if keep is not None:
            b, w, a = b[keep], w[keep], a[keep]
        if not np.any(w > 0):
            return None
        normal = (b.conj().T * w) @ b
        normal += smoothing * np.mean(np.real(np.diag(normal))) * np.diag(penalty)
        return np.linalg.solve(normal, (b.conj().T * w) @ a)

    coefficients = [None] * len(prepared)
    for p in prepared:
        p['w_fit'] = p['w'].copy()
    for iteration in range(iterations + 1):
        for i, p in enumerate(prepared):
            coefficients[i] = solve(p)
        if iteration == iterations:
            break
        if robust:
            for i, p in enumerate(prepared):
                agree = agreement(i, p)
                factor = np.array([max(agree[r], 0.0) ** robust if np.isfinite(agree[r]) else 0.0 for r in p['run']])
                p['w_fit'] = p['w'] * factor
        # Each run against the fit of the others (its own samples, still misaligned, would hold the fit back).
        new_offsets = dict(offsets)
        for r in runs:
            if r == reference:
                continue
            trial = start[r] + np.arange(-max_offset_s, max_offset_s + step_s / 2, step_s)
            total = np.zeros(trial.size)
            for p in prepared:
                m = p['run'] == r
                if not m.any():
                    continue
                others = solve(p, ~m)
                if others is None:
                    continue
                cross = p['w_fit'][m] * p['a'][m] * np.conj(p['basis'][m] @ others)
                total += np.real(np.exp(-1j * np.outer(trial, p['rate'][m])) @ cross)
            if np.any(total):
                new_offsets[r] = float(trial[int(np.argmax(total))])
        offsets.update(new_offsets)
    consistency, gain = {}, {}
    for i, p in enumerate(prepared):
        model = p['basis'] @ coefficients[i]
        a = corrected(p)
        for r, value in agreement(i, p).items():
            m = p['run'] == r
            w = p['w'][m]
            consistency[(i, r)] = value
            gain[(i, r)] = float(np.sum(w * np.abs(a[m]) ** 2) / max(np.sum(w * np.abs(model[m]) ** 2), 1e-300))
    return {'coefficients': coefficients, 'degree': degree, 'offsets': offsets, 'reference': reference,
            'consistency': consistency, 'gain': gain}


def evaluate_directivity(coefficients, degree, phi_deg, theta_deg):
    """A fitted directivity (:func:`fit_directivities`) at ART directions (deg): complex, the shape of phi_deg."""
    polar, azimuth = axis_angles(phi_deg, theta_deg)
    shape = np.shape(polar)
    return (harmonic_basis(np.ravel(polar), np.ravel(azimuth), degree) @ coefficients).reshape(shape)


def fit_summed_directivities(problems, *, degree=4, smoothing=1e-2, max_offset_s=0.01, step_s=2e-5, iterations=4,
                             reference=None, offsets=None, robust=4.0):
    """Each source's smooth directivity of one shaft order, fitted to what the microphones hear of all the sources
    together, with each run's timing.

    Rotors turning within a fraction of a hertz of each other cannot be told apart at one microphone over the
    envelope's few tenths of a second, so their separately extracted tones (:func:`rotor_locked_tones`) share
    their energy out arbitrarily.  Their sum at the microphone is well determined.  So each source's directivity
    (spherical harmonics about its shaft, as :func:`fit_directivities`) is fitted to that sum, over every
    microphone and every second of every run at once:

        z = sum_r gain_r B(n_r) c_r exp(i (carrier_r + rate_r delta_run)),

    z the order's sum as heard, gain_r each source's propagation from its sphere to the microphone, n_r its
    direction, carrier_r = m psi_r its order's phase from the source's azimuth, rate_r = m |Omega_r| and delta_run
    the run's timing offset.  The sources' relative phases drift as their speeds differ, which is what tells them
    apart.

    Args:
        problems: a list of dicts, each with its own coefficients (one order at one condition, say) and the run
            offsets shared by all: 'sources' (R), 'blocks' (a list of dicts, one per microphone and segment: 'z'
            (n,) complex Pa; 'gain', 'carrier' (rad), 'rate' (rad/s), 'phi', 'theta' (deg, ART): (n, R);
            'weight' (n,): 1 / the noise variance, say; 'run': a label), and optionally 'azimuthal_orders': (R,)
            each source's azimuthal order k, or a set of them (or None for all).  A rotor in hover radiates a field
            that turns with it and does not otherwise change: its tone of shaft order m is a function of the polar
            angle about the shaft times exp(i k azimuth), k = +-m by its sense of rotation (Valente, Greenwood and
            Johnson's phase compensation).  Only the harmonics Y_l^k of those k are then fitted, and the fit
            predicts every azimuth from the ones measured.  A disturbance fixed in the vehicle under the rotor (an
            arm the blades pass over) adds the orders k +- 1, k +- 2, ... (rotor-stator interaction with one
            stator), weakening away from k: a set of a few orders about k.
        degree, smoothing, max_offset_s, step_s, iterations, reference, offsets, robust: as
            :func:`fit_directivities` (here each run is fitted against its problems' fits without it).

    Returns a dict: 'coefficients' (list, one (R, n_basis) array per problem), 'degree', 'offsets', 'reference',
    'agreement' {(problem, run): Re(sum w z conj(z_fit)) / sqrt(sum w |z|^2 sum w |z_fit|^2)} and 'explained'
    {(problem, run): 1 - sum w |z - z_fit|^2 / sum w |z|^2}, each run's z against the fit including it.
    """
    nb = (degree + 1) ** 2
    ls = np.concatenate([np.full(2 * l + 1, l) for l in range(degree + 1)])
    ks = np.concatenate([np.arange(-l, l + 1) for l in range(degree + 1)])
    for p in problems:
        orders = p.get('azimuthal_orders')
        if orders is None:
            p['_keep'] = np.ones(p['sources'] * nb, bool)
        else:
            sets = [None if k is None else np.atleast_1d(np.asarray(k, int)) for k in orders]
            if len(orders) != p['sources'] or any(k is not None and np.any(np.abs(k) > degree) for k in sets):
                raise ValueError(f'azimuthal_orders must give one order (or a set) per source, each within the '
                                 f'degree ({degree}): {orders}')
            p['_keep'] = np.concatenate([np.ones(nb, bool) if k is None else np.isin(ks, k) for k in sets])
    runs = sorted({b['run'] for p in problems for b in p['blocks']}, key=str)
    if reference is None:
        mass = {r: sum(float(np.sum(b['weight'])) for p in problems for b in p['blocks'] if b['run'] == r)
                for r in runs}
        reference = max(runs, key=lambda r: (mass[r], str(r)))
    offsets = {r: float((offsets or {}).get(r, 0.0)) for r in runs}
    start = dict(offsets)
    # each block's columns without the offset: gain B exp(i carrier), (n, R * nb)
    for p in problems:
        for b in p['blocks']:
            columns = []
            for r in range(p['sources']):
                polar, azimuth = axis_angles(b['phi'][:, r], b['theta'][:, r])
                columns.append(harmonic_basis(polar, azimuth, degree)
                               * (np.asarray(b['gain'])[:, r] * np.exp(1j * np.asarray(b['carrier'])[:, r]))[:, None])
            b['_columns'] = np.concatenate(columns, axis=1)
            b['_w'] = np.asarray(b['weight'], float).copy()
            ok = np.isfinite(b['z']) & np.all(np.isfinite(b['_columns']), axis=1) & (b['_w'] > 0)
            b['_w'][~ok] = 0.0
            b['_z'] = np.where(ok, b['z'], 0.0)
            b['_columns'][~ok] = 0.0

    def design(b, delta):
        shift = np.exp(1j * np.asarray(b['rate'], float) * delta)              # (n, R)
        return b['_columns'] * np.repeat(shift, nb, axis=1)

    def partial(p, robust_w):
        """{run: (N_run, y_run)}: each run's normal equations at the current offsets."""
        out = {}
        for b in p['blocks']:
            a = design(b, offsets[b['run']])
            w = b['_w'] * robust_w.get(b['run'], 1.0)
            n_, y_ = out.get(b['run'], (0.0, 0.0))
            out[b['run']] = (n_ + (a.conj().T * w) @ a, y_ + (a.conj().T * w) @ b['_z'])
        return out

    def solve(normal, rhs, keep):
        """The coefficients (zero outside `keep`, the harmonics fitted), or None where no sample carries weight
        (every run weighted out)."""
        if np.isscalar(normal):
            return None
        penalty = np.tile((ls * (ls + 1.0)) ** 2, normal.shape[0] // nb)[keep]
        sub = normal[np.ix_(keep, keep)]
        scale = np.mean(np.real(np.diag(sub)))
        if not scale > 0:
            return None
        out = np.zeros(normal.shape[0], dtype=complex)
        out[keep] = np.linalg.solve(sub + smoothing * scale * np.diag(penalty), rhs[keep])
        return out

    weights = [{} for _ in problems]
    coefficients = [None] * len(problems)
    for iteration in range(iterations + 1):
        parts = [partial(p, weights[i]) for i, p in enumerate(problems)]
        for i, (p, part) in enumerate(zip(problems, parts)):
            c = solve(sum(n_ for n_, _ in part.values()), sum(y_ for _, y_ in part.values()), p['_keep'])
            coefficients[i] = c if c is not None else np.zeros(p['sources'] * nb, dtype=complex)
        if iteration == iterations:
            break
        stats = _summed_agreement(problems, coefficients, design, offsets)
        if robust:
            for i in range(len(problems)):
                weights[i] = {r: max(stats[0].get((i, r), 0.0), 0.0) ** robust for r in runs}
        new_offsets = dict(offsets)
        for r in runs:
            if r == reference:
                continue
            trial = start[r] + np.arange(-max_offset_s, max_offset_s + step_s / 2, step_s)
            total = np.zeros(trial.size)
            for i, (p, part) in enumerate(zip(problems, parts)):
                if r not in part:
                    continue
                others = [run for run in part if run != r]
                if not others:
                    continue
                c = solve(sum(part[o][0] for o in others), sum(part[o][1] for o in others), p['_keep'])
                if c is None:
                    continue
                for b in (b for b in p['blocks'] if b['run'] == r):
                    # each source's term without its offset, and its rate: the fit at delta is their sum
                    terms = (b['_columns'] * c[None, :]).reshape(b['_z'].size, p['sources'], nb).sum(axis=2)
                    cross = (b['_w'] * weights[i].get(r, 1.0))[:, None] * np.conj(terms) * b['_z'][:, None]
                    for k in range(p['sources']):
                        total += np.real(np.exp(-1j * np.outer(trial, np.asarray(b['rate'])[:, k])) @ cross[:, k])
            if np.any(total):
                new_offsets[r] = float(trial[int(np.argmax(total))])
        offsets.update(new_offsets)
    agreement, explained = _summed_agreement(problems, coefficients, design, offsets)
    return {'coefficients': [c.reshape(p['sources'], nb) for c, p in zip(coefficients, problems)],
            'degree': degree, 'offsets': offsets, 'reference': reference, 'agreement': agreement,
            'explained': explained}


def energy_basis(polar, knots_rad):
    """Piecewise-linear hats in the polar angle (rad) about the knots (rad, increasing): (n, n_knots), a partition of
    unity, flat beyond the end knots.  A combination with coefficients >= 0 is >= 0 everywhere."""
    polar = np.clip(np.asarray(polar, float), knots_rad[0], knots_rad[-1])
    eye = np.eye(len(knots_rad))
    return np.stack([np.interp(polar, knots_rad, eye[k]) for k in range(len(knots_rad))], axis=-1)


def evaluate_energy_directivity(coefficients, knots_deg, phi_deg, theta_deg):
    """A fitted energy directivity (:func:`fit_incoherent_directivities`) at ART directions (deg): the mean square of
    the tone's incoherent part (peak amplitude squared, Pa^2, at the sphere radius), the shape of phi_deg."""
    polar, _ = axis_angles(phi_deg, theta_deg)
    shape = np.shape(polar)
    return (energy_basis(np.ravel(polar), np.radians(knots_deg)) @ np.asarray(coefficients, float)).reshape(shape)


def fit_incoherent_directivities(problems, fit, *, knots_deg=np.arange(0.0, 181.0, 15.0), shared=False,
                                 smoothing=1e-2, iterations=4):
    """Each source's energy directivity of what its order radiates without a stable phase: the part of the order sum
    that :func:`fit_summed_directivities` leaves.

    A tone whose phase wanders against the rotor's azimuth (unsteady loading, the rotor's own turbulence) averages
    out of the phase-locked fit, but its energy still reaches the microphones.  Taken as independent between
    sources, its mean square at a sample is

        E|z - z_fit|^2 = sum_r |gain_r|^2 I_r(n_r),

    I_r the source's mean-square peak amplitude at its sphere radius in direction n_r, axisymmetric about its
    shaft: piecewise linear in the polar angle (:func:`energy_basis`, the knots), each coefficient >= 0, so I_r >= 0.
    Fitted by non-negative least squares on the residual power, reweighted by the model's own mean (the residual
    power of a random phasor is exponentially distributed: its spread is its mean), with a second-difference
    penalty ``smoothing`` (relative to the normal equations' mean diagonal) along each source's knots.

    Args:
        problems: as :func:`fit_summed_directivities`.
        fit: its result for these problems (the coefficients, degree and offsets the residuals are taken from).
        knots_deg: the polar angles of the hats, deg from the down axis.
        shared: one directivity for every source of a problem (the rotors alike) instead of one each.
        smoothing, iterations: the penalty, and the reweighting passes.

    Returns a dict: 'coefficients' (list, one (R, n_knots) array per problem, Pa^2), 'knots_deg', and 'fraction'
    (list: the share of each problem's measured power that the incoherent part carries, sum I-model / sum |z|^2).
    """
    from scipy.optimize import nnls
    knots = np.radians(np.asarray(knots_deg, float))
    nk = knots.size
    out, fraction = [], []
    for i, p in enumerate(problems):
        R = p['sources']
        rows, targets, total = [], [], 0.0
        for b in p['blocks']:
            delta = fit['offsets'].get(b['run'], 0.0)
            z_fit = np.zeros(b['z'].size, complex)
            design = np.zeros((b['z'].size, R * nk))
            for r in range(R):
                polar, azimuth = axis_angles(b['phi'][:, r], b['theta'][:, r])
                z_fit += b['gain'][:, r] * (harmonic_basis(polar, azimuth, fit['degree']) @ fit['coefficients'][i][r]) \
                    * np.exp(1j * (b['carrier'][:, r] + b['rate'][:, r] * delta))
                design[:, r * nk:(r + 1) * nk] = np.abs(b['gain'][:, r])[:, None] ** 2 * energy_basis(polar, knots)
            rows.append(design)
            targets.append(np.abs(b['z'] - z_fit) ** 2)
            total += float(np.sum(np.abs(b['z']) ** 2))
        A, e = np.concatenate(rows), np.concatenate(targets)
        if shared:
            A = A.reshape(A.shape[0], R, nk).sum(axis=1)
        groups = 1 if shared else R
        d2 = np.diff(np.eye(nk), 2, axis=0)
        penalty = np.kron(np.eye(groups), d2)
        weight = np.full(e.size, 1.0 / max(np.mean(e), 1e-300) ** 2)
        for _ in range(iterations):
            sw = np.sqrt(weight)
            normal_scale = np.mean(np.sum((A * sw[:, None]) ** 2, axis=0))
            lhs = np.vstack([A * sw[:, None], np.sqrt(smoothing * normal_scale) * penalty])
            rhs = np.concatenate([e * sw, np.zeros(penalty.shape[0])])
            a, _ = nnls(lhs, rhs, maxiter=50 * lhs.shape[1])
            model = A @ a
            weight = 1.0 / np.maximum(model, 1e-3 * max(np.mean(e), 1e-300)) ** 2
        coefficients = np.tile(a, (R, 1)) if shared else a.reshape(R, nk)
        out.append(coefficients)
        fraction.append(float(np.sum(A @ a) / total) if total > 0 else 0.0)
    return {'coefficients': out, 'knots_deg': np.asarray(knots_deg, float), 'fraction': fraction}


def _summed_agreement(problems, coefficients, design, offsets):
    agreement, explained = {}, {}
    sums = {}
    for i, (p, c) in enumerate(zip(problems, coefficients)):
        for b in p['blocks']:
            fit = design(b, offsets[b['run']]) @ c
            w, z = b['_w'], b['_z']
            s = sums.setdefault((i, b['run']), np.zeros(4, dtype=complex))
            s += [np.sum(w * z * np.conj(fit)), np.sum(w * np.abs(z) ** 2), np.sum(w * np.abs(fit) ** 2),
                  np.sum(w * np.abs(z - fit) ** 2)]
    for key, (cross, zz, ff, rr) in sums.items():
        den = np.sqrt(np.real(zz) * np.real(ff))
        agreement[key] = float(np.real(cross) / den) if den > 0 else float('nan')
        explained[key] = float(1.0 - np.real(rr) / np.real(zz)) if np.real(zz) > 0 else float('nan')
    return agreement, explained
