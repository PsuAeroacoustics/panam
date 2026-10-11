"""Doppler-predicted order tracks, and clock offsets between records.

An order of a rotor turning at shaft frequency f(t) is emitted at k f(t_e).  A
fixed microphone hears it at reception time t_r = t_e + R(t_e)/c, shifted by
the Doppler factor D = dt_e/dt_r = 1/(1 - M_r), M_r the source's Mach number
toward the microphone:

    f_obs(t_r) = D(t_r) k f(t_e(t_r))

:func:`observed_order_track` gives these tracks on a reception-time grid, in
the form :func:`vold_kalman_filter.vold_kalman_filter` takes as ``freq``, so the
filter follows the order the microphone hears rather than the one the rotor
makes.  :func:`ridge_track` measures the strongest tone in a band from the
signal itself.  :func:`estimate_clock_offset` cross-correlates two records, two
microphones' pressures or a predicted and a measured order track, and returns
their time offset to a fraction of a sample.  A clock offset between the
telemetry and the acoustic recorder shifts the whole predicted track in
reception time, so :func:`order_track_offset` measures it directly.

Lengths are in ft and speeds in ft/s, as in the 2017 pipeline; any length unit
works as long as the track, the receiver and the speed of sound agree.  Times
are in seconds.  The model is in ``docs/THEORY.md`` (auxiliary models).
"""
import numpy as np
import scipy.signal as sig

import ground_plane

#: Factors from each ``speed_unit`` to revolutions per second (Hz).
SPEED_UNITS = {'rpm': 1.0 / 60.0, 'hz': 1.0, 'rad/s': 1.0 / (2.0 * np.pi)}


def _track_arrays(track):
    """Time (s), position (n, 3) and velocity (n, 3) of a track dict."""
    time = np.asarray(track['time'], dtype=float)
    position = np.column_stack([np.asarray(track[k], dtype=float) for k in ('x', 'y', 'z')])
    if time.ndim != 1 or len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError('track time must be strictly increasing, with at least two samples')
    return time, position, np.gradient(position, time, axis=0)


def _interp_columns(t, time, values):
    """Linear interpolation of each column of ``values`` (n, k) at ``t``."""
    return np.column_stack([np.interp(t, time, values[:, k]) for k in range(values.shape[1])])


def doppler_factor(reception_times, track, receiver, sound_speed):
    """Emission time and Doppler factor of a moving source heard at a fixed receiver.

    ``track`` is a dict of ``time`` (s, increasing) and source position ``x``,
    ``y``, ``z`` (ft); positions between samples are interpolated linearly, and
    the velocity is the central difference of the samples, also interpolated
    linearly.  ``receiver`` is the (3,) receiver position (ft), ``sound_speed``
    in ft/s and ``reception_times`` in s.  Emission times are solved by
    :func:`ground_plane.emission_times`.

    Returns (t_e, D): emission times (s) and the Doppler factor
    D = dt_e/dt_r = 1/(1 - v.r/(|r| c)), r from the source to the receiver, so
    a received frequency is D times the emitted one.  Both are NaN where the
    sound would have left the source outside the track's time span.
    """
    time, position, velocity = _track_arrays(track)
    receiver = np.asarray(receiver, dtype=float)
    if receiver.shape != (3,):
        raise ValueError('receiver must be one (3,) position')
    t_r = np.asarray(reception_times, dtype=float)
    t_e, x_e = ground_plane.emission_times(lambda t: _interp_columns(t, time, position),
                                           t_r.ravel(), receiver, sound_speed)
    r = receiver - x_e
    v = _interp_columns(t_e, time, velocity)
    with np.errstate(invalid='ignore', divide='ignore'):
        mach_toward = np.sum(v * r, axis=1) / (np.linalg.norm(r, axis=1) * sound_speed)
        factor = 1.0 / (1.0 - mach_toward)
    outside = (t_e < time[0]) | (t_e > time[-1])
    t_e = np.where(outside, np.nan, t_e)
    factor = np.where(outside, np.nan, factor)
    return t_e.reshape(t_r.shape), factor.reshape(t_r.shape)


def observed_order_track(reception_times, track, receiver, rotor_speed, orders, sound_speed,
                         rotor_time=None, speed_unit='rpm'):
    """Frequencies (Hz) of rotor orders as heard by a fixed receiver, on its clock.

    Each order k of a shaft turning at f(t) revolutions per second is emitted
    at k f(t_e) and received at D(t_r) k f(t_e(t_r)) (:func:`doppler_factor`):
    the propagation delay and the Doppler shift are both applied, so the track
    lines up with the received signal.

    Parameters
    ----------
    reception_times : array_like
        Receiver times, s, on the same clock as the track (shape (n,)).
    track : dict
        ``time`` (s) and source position ``x``, ``y``, ``z`` (ft).
    receiver : array_like
        Receiver position (3,), ft, in the track's frame.
    rotor_speed : float, array_like or callable
        Shaft speed in ``speed_unit``: a constant, an array (m,) or
        (m, n_shafts) sampled at ``rotor_time``, or a callable that takes
        emission times (s) and returns (n,) or (n, n_shafts).
    orders : float or array_like
        Multiples of the shaft frequency to track (a blade-passage harmonic j
        of a b-bladed rotor is order j b).
    sound_speed : float
        ft/s.
    rotor_time : array_like, optional
        Times (s) of the ``rotor_speed`` samples; default the track's times.
        Rotor speeds are interpolated linearly and held at the ends.
    speed_unit : {'rpm', 'hz', 'rad/s'}
        Unit of ``rotor_speed``; default 'rpm'.

    Returns
    -------
    ndarray
        Observed frequency, Hz, of shape (n,) + shaft shape + orders shape: a
        single shaft and a list of orders give (n, n_orders) and several
        shafts with one order (n, n_shafts), each a ``freq`` for
        :func:`vold_kalman_filter.vold_kalman_filter`.  NaN where the sound
        left the source outside the track's time span; the filter needs
        finite frequencies, so trim those samples or fill them first.
    """
    try:
        to_hz = SPEED_UNITS[speed_unit.lower()]
    except KeyError:
        raise ValueError(f"speed_unit must be one of {sorted(SPEED_UNITS)}, not {speed_unit!r}") from None
    t_e, factor = doppler_factor(reception_times, track, receiver, sound_speed)
    known = np.isfinite(t_e)
    t_known = np.where(known, t_e, np.asarray(track['time'], dtype=float)[0])
    if callable(rotor_speed):
        speed = np.asarray(rotor_speed(t_known), dtype=float)
    elif np.ndim(rotor_speed) == 0:
        speed = np.full(t_known.shape, float(rotor_speed))
    else:
        values = np.asarray(rotor_speed, dtype=float)
        times = np.asarray(track['time'] if rotor_time is None else rotor_time, dtype=float)
        if values.shape[0] != len(times):
            raise ValueError(f'rotor_speed has {values.shape[0]} samples and its times {len(times)}')
        flat = values.reshape(len(times), -1)
        speed = _interp_columns(t_known, times, flat).reshape(t_known.shape + values.shape[1:])
    shaft_hz = speed * to_hz * np.where(known, factor, np.nan).reshape(t_e.shape + (1,) * (speed.ndim - t_e.ndim))
    orders = np.asarray(orders, dtype=float)
    return shaft_hz.reshape(shaft_hz.shape + (1,) * orders.ndim) * orders


def ridge_track(x, fs, band, window=1.0, hop=0.1):
    """Frequency (Hz) of the strongest spectral peak in ``band``, frame by frame.

    ``x`` is a signal sampled at ``fs`` (Hz); ``band`` is (low, high) in Hz;
    ``window`` and ``hop`` are the Hann frame length and step, s (defaults 1 s
    and 0.1 s).  The peak bin is refined by a parabola through the log power
    of it and its neighbors.  Returns (times, frequency): frame centers (s from
    the first sample) and peak frequencies (Hz).  It is meant for a single tone
    that dominates the band.
    """
    x = np.asarray(x, dtype=float)
    n, step = int(round(window * fs)), int(round(hop * fs))
    if n < 4 or step < 1 or len(x) < n:
        raise ValueError('the window must span at least 4 samples and fit in the signal')
    frequency = np.fft.rfftfreq(n, 1.0 / fs)
    inside = np.flatnonzero((frequency >= band[0]) & (frequency <= band[1]))
    if len(inside) == 0:
        raise ValueError(f'no frequency bin between {band[0]} and {band[1]} Hz')
    starts = np.arange(0, len(x) - n + 1, step)
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[starts] * np.hanning(n)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    peak = inside[np.argmax(power[:, inside], axis=1)]
    rows = np.arange(len(starts))
    refined = frequency[peak].copy()
    interior = (peak > 0) & (peak < len(frequency) - 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        lo, mid, hi = (np.log(power[rows[interior], peak[interior] + k]) for k in (-1, 0, 1))
        den = lo - 2.0 * mid + hi
        shift = np.where(np.isfinite(den) & (den < 0), 0.5 * (lo - hi) / den, 0.0)
    refined[interior] += shift * (frequency[1] - frequency[0])
    return (starts + n / 2.0) / fs, refined


def estimate_clock_offset(signal_a, signal_b, fs, max_lag, min_overlap=0.5):
    """Time offset between two records by normalized cross-correlation.

    ``signal_a`` and ``signal_b`` are sampled at the same rate ``fs`` (Hz) and
    start at the same nominal time; NaN samples are left out.  At each lag the
    correlation is the Pearson coefficient over the samples the two records
    share, so a record's mean and the length of the overlap do not bias the
    peak.  Lags are searched within +-``max_lag`` (s), and only where the
    overlap holds at least ``min_overlap`` (default 0.5) of the shorter
    record's finite samples.  The peak is refined by a parabola through it and
    its neighbors.

    Returns (lag, peak): lag in s, positive when ``signal_b`` lags
    ``signal_a`` (b(t) ~ a(t - lag), so an event at time t in a is at t + lag
    in b), and the correlation coefficient at the peak sample.
    """
    a = np.asarray(signal_a, dtype=float)
    b = np.asarray(signal_b, dtype=float)
    valid_a, valid_b = np.isfinite(a), np.isfinite(b)
    if valid_a.sum() < 3 or valid_b.sum() < 3:
        raise ValueError('each record needs at least three finite samples')
    a = np.where(valid_a, a - a[valid_a].mean(), 0.0)
    b = np.where(valid_b, b - b[valid_b].mean(), 0.0)
    ma, mb = valid_a.astype(float), valid_b.astype(float)

    def xc(u, v):  # sum over n of u[n + k] v[n], at sig.correlation_lags(len(u), len(v))
        return sig.correlate(u, v, mode='full')

    count = np.rint(xc(mb, ma))
    s_ab, s_a, s_b = xc(b, a), xc(mb, a), xc(b, ma)
    s_aa, s_bb = xc(mb, a * a), xc(b * b, ma)
    lags = sig.correlation_lags(len(b), len(a))
    needed = max(3.0, min_overlap * min(valid_a.sum(), valid_b.sum()))
    usable = (np.abs(lags) <= max_lag * fs) & (count >= needed)
    with np.errstate(divide='ignore', invalid='ignore'):
        n = np.where(usable, count, np.nan)
        cov = s_ab - s_a * s_b / n
        var = (s_aa - s_a ** 2 / n) * (s_bb - s_b ** 2 / n)
        corr = np.where(var > 0, cov / np.sqrt(var), np.nan)
    if not np.any(np.isfinite(corr)):
        raise ValueError('no lag within max_lag has enough overlapping samples')
    j = int(np.nanargmax(corr))
    lag = float(lags[j])
    if 0 < j < len(corr) - 1 and np.all(np.isfinite(corr[j - 1:j + 2])):
        c0, c1, c2 = corr[j - 1:j + 2]
        den = c0 - 2.0 * c1 + c2
        if den < 0:
            lag += 0.5 * (c0 - c2) / den
    return lag / fs, float(corr[j])


def order_track_offset(times, predicted, observed, max_lag, min_overlap=0.5):
    """Clock offset (s) between a predicted and a measured order track.

    ``times`` (s) are equally spaced; ``predicted`` (for example from
    :func:`observed_order_track`) and ``observed`` (for example from
    :func:`ridge_track`) are frequencies, Hz, at those times, NaN where
    unknown.  Returns (lag, peak) from :func:`estimate_clock_offset`: the
    measured track runs ``lag`` s late, observed(t) ~ predicted(t - lag), so
    subtracting ``lag`` from the recorder's times aligns it with the
    prediction's clock.
    """
    times = np.asarray(times, dtype=float)
    step = np.diff(times)
    if len(times) < 3 or not np.allclose(step, step[0], rtol=1e-6, atol=0.0) or step[0] <= 0:
        raise ValueError('times must be equally spaced and increasing')
    return estimate_clock_offset(predicted, observed, 1.0 / step[0], max_lag, min_overlap=min_overlap)
