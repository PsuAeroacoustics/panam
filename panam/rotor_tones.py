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

from panam import ground_plane as gp
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
