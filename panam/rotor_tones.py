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
from panam.ground_effect import spherical_reflection_coefficient
from panam.vold_kalman_filter import vold_kalman_filter


def rotor_azimuth(time, angle, speed, *, aligned_angle=0.0, step_s=1e-3, window_s=0.25, min_coherence=0.3):
    """A rotor's continuous azimuth in its own sense of rotation, from a logged angle and speed.

    Args:
        time: the log's sample times, s, increasing.
        angle: the logged mechanical angle, rad, wrapped any way; it advances with the signed ``speed``.  Repeated
            values (a controller that updates the angle less often than the log's rows) are dropped.
        speed: the logged shaft speed, rad/s, signed by the sense of rotation the angle advances in.
        aligned_angle: the logged angle at which the blades are at the azimuth's zero (a prop set's alignment with
            its arm, say), rad.
        step_s: the output grid's spacing, s.
        window_s: the window over which the angle's offset from the speed integral is averaged, s.
        min_coherence: windows whose offsets scatter more than this mean resultant length allows are skipped.

    The azimuth is the integral of the speed plus a slow correction: the circular mean, in ``window_s`` windows,
    of the logged angle less that integral, unwrapped and interpolated.  Single logged angles of a fast rotor
    scatter tens of degrees about it; the speed carries the fast part and the angle only the drift.

    Returns (grid, azimuth, diagnostics): the grid's times (s, every ``step_s`` over the log), the azimuth there
    (rad, increasing whatever the sense: the signed angle times the sign of the mean speed, less the aligned
    angle), and {'sense': +1 or -1 (the sign of the mean speed), 'window_coherence': the median mean resultant
    length of the windows, 'scatter_deg': the rms of the samples about the result}.  Raises ValueError for a log
    with no usable window or a speed of either sign.
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
    sense = 1.0 if np.mean(w[finite_w]) > 0 else -1.0
    if np.any(np.sign(w[finite_w]) == -sense):
        raise ValueError('the speed changes sign: a rotor turns one way')
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


def rotor_locked_tones(pressure, fs, azimuths, harmonics, bandwidth_hz, *, p=2, method='vkf'):
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
        y, phasor, _ = vold_kalman_filter(x, None, fs, bandwidth_hz, p, phase=phase)
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


def plate_tone_table(frequencies_hz, sound_speed_ft_s, flow_resistance=gp.FLOW_RESISTANCE, ground=None,
                     **options):
    """A plate-on-the-ground table for single frequencies (tones), not bands: :func:`axisymmetric_bem.table` with
    one sub-frequency per entry, so that ``frequencies`` are exactly ``frequencies_hz`` (sorted).  Lengths in ft,
    speed in ft/s, as there; ``options`` as there (the plate and the microphone).  Space the frequencies closely
    enough for linear interpolation between them (:func:`plate_tone_response`): the plate's response turns over a
    few hundred hertz for a plate a few tenths of a meter across."""
    return ab.table(np.unique(np.asarray(frequencies_hz, dtype=float)), sound_speed_ft_s, flow_resistance, ground,
                    sub_bands=1, **options)


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
