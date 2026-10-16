"""Forward synthesis: microphone band-level histories from a source hemisphere.

``synthesize_band_histories`` is the inverse of
:func:`flight_acoustics.depropagate_hemisphere`.  For each microphone and
reception time it solves for the emission time, takes the direction the sound
left the source along, reads the hemisphere's band levels there, and carries
them to the microphone: the Doppler band shift (for a hemisphere filed at
emitted frequencies), the receiver's response, spherical spreading from the
hemisphere radius and atmospheric absorption beyond it.  ``leave_mics_out``
depropagates with some microphones held out, synthesizes the held-out ones and
reports the residuals band by band, so the gridding settings (``rmax``,
``interpolation``, ``rim_range``) can be compared without building a database.

Conventions are those of ``depropagate_hemisphere``: a right-handed, z-up
frame, lengths in its ``length_units`` (ft for the 2017 build) and speeds in
length units per second; hemisphere angles in UMAPR (azimuth 180 ahead, 90 to
starboard, elevation positive below the horizon); levels in dB re 20 uPa, with
-inf for no energy and NaN for no data, combined in energy.
"""

import inspect
import warnings

import numpy as np

from panam import flight_acoustics as fa
from panam import ground_plane as gp
from panam.acoustics import filters as pa_filters
from panam.acoustics.atmosphere import (Atmosphere, band_absorption_db, check_absorption_profile,
                                        energy_mean_attenuation_db)
from panam.unit_conversion import len_conv

__all__ = ['synthesize_band_histories', 'doppler_shift_band_power', 'measured_band_histories',
           'leave_mics_out', 'residual_statistics']


def _track_functions(track):
    """Position, velocity and nose of a track mapping as functions of time.

    ``track`` holds ``track_time`` (Nt,) s, ``track_position`` (Nt, 3) and
    optionally ``track_velocity`` (Nt, 3) (default: finite differences, as in
    ``depropagate_hemisphere``) and ``track_nose`` (Nt, 3).  Each is
    interpolated linearly and held at the ends.
    """
    time = np.asarray(track['track_time'], dtype=float)
    position = np.asarray(track['track_position'], dtype=float)
    if position.shape != (time.size, 3):
        raise ValueError('track_position must be shaped (Nt, 3) matching track_time')
    velocity = track.get('track_velocity')
    if velocity is None:
        velocity = fa.finite_difference_velocity(time, position)
    velocity = np.asarray(velocity, dtype=float)
    nose = track.get('track_nose')
    nose = None if nose is None else np.asarray(nose, dtype=float)

    def interpolate(values):
        return lambda t: gp.interp_columns(np.asarray(t, dtype=float), time, values)
    return time, interpolate(position), interpolate(velocity), None if nose is None else interpolate(nose)


def _emission_times(position, times, mic, speed_of_sound, ray_model, tol=1e-9, max_iter=50):
    """Emission times (s) and source positions of the sound reaching ``mic`` at ``times``.

    Straight lines: :func:`ground_plane.emission_times`.  With a ray model the
    straight solution starts the same fixed-point iteration on the ray's
    travel time, t_e = t - tau(x(t_e), mic).
    """
    t_e, x = gp.emission_times(position, times, mic, speed_of_sound, tol=tol, max_iter=max_iter)
    if ray_model is None:
        return t_e, x
    return gp.emission_times(position, times, mic, speed_of_sound, tol=tol, max_iter=max_iter, t0=t_e,
                             travel_time=lambda x: ray_model(x, mic[None, :])['travel_time'][:, 0])


def doppler_shift_band_power(power, band_centers_hz, doppler):
    """Band power at emitted frequencies carried to the bands it is received in.

    A band emitted over [f_l, f_u] arrives over [D f_l, D f_u].  Its power is
    taken as spread evenly in frequency over the band, so received band k gets
    the fraction of each shifted band that overlaps it.  Power is conserved:
    the shift moves frequency only (no convective amplification, as in
    depropagation).  This is the inverse of ``depropagate_hemisphere(
    remove_doppler=True)``, which integrates the received spectrum over
    [D f_l, D f_u]; a tone, which the even spread smears over the shifted
    band, returns to a single band only when D moves it whole.

    Args:
        power: (Nb, N) band power at emitted frequencies (any power unit, NaN no data).
        band_centers_hz: (Nb,) contiguous band centers, Hz (edges from
            :func:`flight_acoustics.third_octave_band_edges`).
        doppler: (N,) received over emitted frequency ratio D.

    Returns:
        (Nb, N) power in the same bands at received frequencies.  A received
        band is NaN where part of it comes from frequencies outside the given
        bands, or from a NaN band.
    """
    power = np.asarray(power, dtype=float)
    d = np.asarray(doppler, dtype=float)
    lower, upper = fa.third_octave_band_edges(band_centers_hz)
    lo = lower[None, :, None] * d[None, None, :]          # (1, Nb emitted, N)
    hi = upper[None, :, None] * d[None, None, :]
    overlap = np.clip(np.minimum(upper[:, None, None], hi) - np.maximum(lower[:, None, None], lo), 0.0, None)
    fraction = overlap / (hi - lo)                         # (Nb received, Nb emitted, N)
    with np.errstate(invalid='ignore'):
        out = np.sum(np.where(fraction > 0.0, fraction * power[None, :, :], 0.0), axis=1)
    slack = 1.0 + 1e-12
    covered = (lower[0] * d[None, :] <= lower[:, None] * slack) & (upper[:, None] <= upper[-1] * d[None, :] * slack)
    return np.where(covered, out, np.nan)


#: Frequencies per band at which :func:`_bin_band_absorption_db` averages the pure-tone
#: attenuation: the midpoints of equal slices of [f_l, f_u).
BIN_ABSORPTION_POINTS = 32


def _bin_band_absorption_db(tone_db, band_centers_hz):
    """Band attenuation, dB, (Nb,) + path shape, of a band whose FFT bins were
    corrected one by one: -10 lg of the mean over [f_l, f_u) of 10^(-delta_t(f) / 10).

    ``tone_db(f)`` returns the pure-tone attenuation over the path(s), dB, shaped
    ``f.shape + path shape``, for a 1-D array ``f`` in Hz.  The mean is over
    :data:`BIN_ABSORPTION_POINTS` frequencies spread evenly in frequency, as FFT bins
    are, so it is exact for a spectrum flat within the band.
    """
    lower, upper = fa.third_octave_band_edges(band_centers_hz)
    fraction = (np.arange(BIN_ABSORPTION_POINTS) + 0.5) / BIN_ABSORPTION_POINTS
    frequency = lower[:, None] + (upper - lower)[:, None] * fraction[None, :]      # (Nb, K)
    tone = np.asarray(tone_db(frequency.ravel()), dtype=float)
    return energy_mean_attenuation_db(tone.reshape(frequency.shape + tone.shape[1:]), axis=1)


def synthesize_band_histories(sphere, track, mics, *, times=None, speed_of_sound=1135.0, r_ref=None,
                              length_units=None, ray_model=None, apply_absorption=False, atmosphere=None,
                              absorption_band_method='center', absorption_profile=None, receiver_response_db=None,
                              doppler_shift=False, air_relative_doppler=False, flip_y_for_geometry=None,
                              absorption_per_bin=False):
    """Predict one-third octave band-level time histories at microphones from a source hemisphere.

    For each microphone and reception time t the emission time t_e solves
    t = t_e + tau, tau = |x(t_e) - x_m| / c (:func:`ground_plane.emission_times`)
    or the ray model's travel time.  The hemisphere is read in energy
    (:func:`flight_acoustics.interpolate_hemisphere_power`) along the direction
    the sound left: :func:`flight_acoustics.hemigen`'s azimuth and elevation at
    t_e, or with a ray model the ray's launch depression and, if it returns
    ``emission_direction``, its azimuth.  The band power P_b at ``r_ref`` is then

        P_b(mic) = S_b(D) * 10^(G_b / 10) * (r_ref / r_s)^2 * 10^(-[delta_b(l) - delta_b(r_ref)] / 10),

    in the order that undoes ``depropagate_hemisphere``'s: S the Doppler band
    shift (:func:`doppler_shift_band_power`, with ``doppler_shift``), G the
    receiver response in dB, r_s the straight distance or the ray's spreading
    range, and delta_b the band's absorption over the straight distance or the
    ray's arc length l (with ``apply_absorption``).  The hemisphere keeps the
    absorption over its first ``r_ref``, so only the excess is applied.

    Args:
        sphere: a ``depropagate_hemisphere`` result with ``third_octave``:
            ``azi_grid_deg``, ``elv_grid_deg`` (UMAPR, deg) and
            ``third_octave['bands_db']`` (Nb, Nelv, Nazi), dB re 20 uPa at r_ref.
        track: mapping with ``track_time`` (Nt,) s, ``track_position`` (Nt, 3),
            and optionally ``track_velocity`` (Nt, 3) length units per second
            and ``track_nose`` (Nt, 3), as ``depropagate_hemisphere`` takes
            them (a ``synthetic_flyover.synthesize_pass`` scenario works as is).
        mics: (Nmics, 3) microphone positions, length units.
        times: (N,) reception times, s, shared by every microphone.  Default
            ``track_time``.
        speed_of_sound: length units per second.
        r_ref: hemisphere radius, length units.  Default ``sphere['metadata']['r_ref']``.
        length_units: 'ft' or 'm', for absorption.  Default the sphere's metadata, else 'ft'.
        ray_model: None (straight lines) or a :mod:`refracted_rays` model.
        apply_absorption: apply atmospheric absorption beyond ``r_ref``.  Default False,
            matching ``depropagate_hemisphere(apply_absorption_deprop=False)``.
        atmosphere: :class:`panam.acoustics.atmosphere.Atmosphere` for absorption.
            Default the sphere's metadata atmosphere, else 293.15 K, 101.325 kPa, 20% RH.
        absorption_band_method: band absorption method
            (:func:`panam.acoustics.atmosphere.band_absorption_db`), default 'center'.
        absorption_profile: None (default) for the uniform ``atmosphere``, or a
            layered profile (:func:`panam.acoustics.atmosphere.check_absorption_profile`)
            as ``depropagate_hemisphere`` takes it: delta_b(l) is then the layered
            attenuation (:func:`panam.acoustics.atmosphere.layered_attenuation_db`)
            between the source and microphone heights, and delta_b(r_ref) stays
            the uniform ``atmosphere``'s.  Needs ``apply_absorption``.
        receiver_response_db: None, or ``f(im, band_centers_hz, source_offset)``
            returning the band response re free field, dB, (Nb, N), as
            ``depropagate_hemisphere`` takes it; ``source_offset`` (N, 3) is the
            emission position minus microphone ``im`` (or the ray model's offset).
        doppler_shift: the hemisphere is filed at emitted frequencies
            (``depropagate_hemisphere(remove_doppler=True)``): shift each band to
            the received frequencies.  Default False: the hemisphere is filed at
            received frequencies, as a flight sphere for NICEOPS is.
        air_relative_doppler: with ``doppler_shift`` and a ray model returning
            ``emission_direction`` and ``doppler_speed``, take D = 1 / (1 - v.n /
            (c + W.n)) instead of the straight line's 1 / (1 - v.u / c).
        flip_y_for_geometry: the hemisphere was filed with
            ``depropagate_hemisphere(flip_y_for_geometry=True)``, which mirrors Y
            on the microphones, track, nose and ray offsets before filing: read it
            on the same mirrored geometry (``mics`` and ``track`` stay as
            measured).  Default None: the sphere's
            ``metadata['flip_y_for_geometry']``, else False.
        absorption_per_bin: the hemisphere's absorption was corrected FFT bin by
            bin (``depropagate_hemisphere(third_octave_method='fft')``, which
            refers each bin to ``r_ref`` with alpha at its own frequency): delta_b
            is then -10 lg of the mean of 10^(-delta(f) / 10) over the band's
            frequencies, for both l and ``r_ref`` (:func:`_bin_band_absorption_db`),
            rather than the band-center value, which over-attenuates high bands
            on long paths.  Exact for a spectrum flat within the band.  Needs
            ``absorption_band_method='center'``.  Default False.

    Returns:
        dict of ``time`` (N,) s, ``band_centers_hz`` (Nb,) Hz, ``bands_db``
        (Nmics, Nb, N) dB re 20 uPa at the microphones, and per microphone and
        time (Nmics, N): ``emission_time`` s, ``azimuth_deg`` and
        ``elevation_deg`` (UMAPR, where the hemisphere was read), ``range``
        (straight distance at emission, length units), ``spreading_range``
        (length units) and ``doppler`` (received over emitted frequency).  A
        band is NaN where the emission time falls outside the track, where the
        ray model finds no ray, or where the hemisphere has no data in that
        direction.
    """
    third = sphere.get('third_octave')
    if third is None:
        raise ValueError('sphere must hold third_octave band levels (depropagate_hemisphere(third_octave=True))')
    metadata = sphere.get('metadata', {})
    centers = np.asarray(third['band_centers_hz'], dtype=float)
    levels = np.asarray(third['bands_db'], dtype=float)
    r_ref = float(metadata['r_ref'] if r_ref is None else r_ref)
    length_units = length_units or metadata.get('length_units', 'ft')
    if atmosphere is None:
        atmosphere = fa.metadata_atmosphere(
            metadata, default=Atmosphere(temperature=293.15, pressure=101.325, relative_humidity=20.0))
    if air_relative_doppler and not (doppler_shift and ray_model is not None):
        raise ValueError('air_relative_doppler needs doppler_shift=True and a ray model')
    if flip_y_for_geometry is None:
        flip_y_for_geometry = bool(metadata.get('flip_y_for_geometry', False))
    mirror = np.array([1.0, -1.0, 1.0]) if flip_y_for_geometry else np.ones(3)
    if absorption_profile is not None:
        if not apply_absorption:
            raise ValueError('absorption_profile needs apply_absorption=True')
        check_absorption_profile(absorption_profile)

    mics = np.asarray(mics, dtype=float)
    if mics.ndim != 2 or mics.shape[1] != 3:
        raise ValueError('mics must be shaped (Nmics, 3)')
    track_time, position, velocity, nose = _track_functions(track)
    times = track_time.copy() if times is None else np.asarray(times, dtype=float).ravel()
    c = float(speed_of_sound)

    if absorption_per_bin and absorption_band_method != 'center':
        raise ValueError("absorption_per_bin corrects FFT bins one by one: it needs absorption_band_method='center'")

    def band_absorption(tone_db):
        """delta_b from ``tone_db(frequencies, method)``: by bins or by ``absorption_band_method``."""
        if absorption_per_bin:
            return _bin_band_absorption_db(lambda f: tone_db(f, 'center'), centers)
        return tone_db(centers, absorption_band_method)

    absorption_reference = None
    if apply_absorption:
        r_ref_m = float(len_conv(r_ref, from_units=length_units, to_units='m'))
        absorption_reference = band_absorption(lambda f, method: band_absorption_db(
            r_ref_m, f, atmosphere.temperature, atmosphere.pressure, atmosphere.relative_humidity, method=method))

    fields = ('emission_time', 'azimuth_deg', 'elevation_deg', 'range', 'spreading_range', 'doppler')
    out = {key: np.full((mics.shape[0], times.size), np.nan) for key in fields}
    out['bands_db'] = np.full((mics.shape[0], centers.size, times.size), np.nan)
    for im, mic in enumerate(mics):
        # Travel times, and so emission times, are the same on the mirrored
        # geometry; the ray model, like depropagation's, sees the measured one.
        t_e, x = _emission_times(position, times, mic, c, ray_model)
        v = velocity(t_e)
        x_g, v_g, mic_g = x * mirror, v * mirror, mic * mirror
        az, el, r, _, _ = fa.hemigen(t_e, x_g, v_g, mic_g[None, :], c,
                                     nose=None if nose is None else nose(t_e) * mirror)
        az, el, r = az[:, 0], el[:, 0], r[:, 0]
        toward = (mic_g[None, :] - x_g) / r[:, None]
        doppler = 1.0 / (1.0 - np.sum(v_g * toward, axis=1) / c)
        spread, path, offset = r, r, x_g - mic_g[None, :]
        valid = (t_e >= track_time[0]) & (t_e <= track_time[-1])
        if ray_model is not None:
            rays = ray_model(x, mic[None, :])
            el = np.asarray(rays['depression_deg'], dtype=float)[:, 0]
            spread = np.asarray(rays['spreading_range'], dtype=float)[:, 0]
            path = np.asarray(rays['path_length'], dtype=float)[:, 0]
            offset = np.asarray(rays['offset'], dtype=float)[:, 0, :] * mirror
            valid &= np.asarray(rays['valid'], dtype=bool)[:, 0] & np.all(np.isfinite(offset), axis=1)
            if 'emission_direction' in rays:
                emitted = np.asarray(rays['emission_direction'], dtype=float)[:, 0, :] * mirror
                valid &= np.all(np.isfinite(emitted), axis=1)
                az = fa.ray_azimuth_deg(az, emitted, toward)
                if air_relative_doppler:
                    if 'doppler_speed' not in rays:
                        raise ValueError('air_relative_doppler needs a ray model that returns doppler_speed')
                    doppler = fa.air_relative_doppler_factor(
                        v_g, emitted, np.asarray(rays['doppler_speed'], dtype=float)[:, 0])
            elif air_relative_doppler:
                raise ValueError('air_relative_doppler needs a ray model that returns emission_direction')

        power, covered = fa.interpolate_hemisphere_power(levels, sphere['elv_grid_deg'], sphere['azi_grid_deg'],
                                                         az, el)
        power = np.where(covered, power, np.nan).T                       # (Nb, N) at r_ref
        if doppler_shift:
            power = doppler_shift_band_power(power, centers, doppler)
        if receiver_response_db is not None:
            gain = np.asarray(receiver_response_db(im, centers, offset), dtype=float)
            if gain.shape != power.shape:
                raise ValueError('receiver_response_db must return values shaped (bands, times)')
            power = power * 10.0 ** (gain / 10.0)
        power = power * (r_ref / spread)[None, :] ** 2
        if apply_absorption:
            delta = band_absorption(lambda f, method: fa.path_absorption_db(
                f, method, path, x[:, 2], mic[2], length_units=length_units, atmosphere=atmosphere,
                absorption_profile=absorption_profile))
            power = power * 10.0 ** (-(delta - absorption_reference[:, None]) / 10.0)
        power = np.where(valid[None, :], power, np.nan)
        out['bands_db'][im] = fa.power_to_db(power)
        for key, value in zip(fields, (t_e, az, el, r, spread, doppler)):
            out[key][im] = np.where(valid, value, np.nan)
    out['time'] = times
    out['band_centers_hz'] = centers
    return out


def measured_band_histories(pressure, time, band_centers_hz, *, window_time=0.5, window_overlap=0.5,
                            ambient_pressure=None, ambient_time_range=None, ambient_percentile=None,
                            band_snr_gate_db=3.0, third_octave_method='fft', tone_aware=False):
    """One microphone's one-third octave band levels per spectrogram frame.

    The bands are formed as ``depropagate_hemisphere`` forms them without
    Doppler removal.  With the FFT (default), the spectrogram
    (:func:`flight_acoustics.spectrogram`, Hann window) is summed over the bins
    in [f_l, f_u), or with ``tone_aware`` filed by
    :func:`flight_acoustics.tone_aware_band_power`; each bin is first gated
    and ambient-subtracted.  With ``third_octave_method='filter_bank'`` the
    bands are :func:`panam.acoustics.filters.third_octave_filter_bank` on the
    spectrogram's frames, gated and ambient-subtracted band by band.  The
    ambient is one of, in this order of precedence as in depropagation: the
    mean spectrum of ``ambient_pressure``, the mean over the frames in
    ``ambient_time_range``, or the ``ambient_percentile`` of the record's own
    frames; none gives no gating.

    Args:
        pressure: (Nsamples,) pressure, Pa.
        time: (Nsamples,) sample times, s (evenly spaced).
        band_centers_hz: (Nb,) band centers, Hz.
        window_time: spectrogram window, s.
        window_overlap: window overlap fraction.
        ambient_pressure: optional (Nsamples_a,) signal-free recording, Pa, at
            the same sample rate.
        ambient_time_range: optional (t0, t1) signal-free window of this
            record, s, in the time base of ``time``.
        ambient_percentile: optional percentile (0-100, exclusive) of the
            record's own frames taken as the ambient, per bin or band.
        band_snr_gate_db: bins (or bands) less than this far above the ambient are zeroed, dB.
        third_octave_method: 'fft' (default) or 'filter_bank'.
        tone_aware: with 'fft', file tones whole (default False).

    Returns:
        (frame_times, bands_db): (Nframes,) frame centers, s, and (Nb, Nframes)
        band levels, dB re 20 uPa (-inf where gating left no energy).
    """
    if third_octave_method not in ('fft', 'filter_bank'):
        raise ValueError("third_octave_method must be 'fft' or 'filter_bank'")
    if tone_aware and third_octave_method != 'fft':
        raise ValueError("tone_aware needs third_octave_method='fft'")
    if ambient_percentile is not None and not 0.0 < float(ambient_percentile) < 100.0:
        raise ValueError('ambient_percentile must lie strictly between 0 and 100')
    time = np.asarray(time, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    fs = float(np.round(1.0 / (time[1] - time[0])))
    f, t_rel, psd_db = fa.spectrogram(pressure, fs, window_time=window_time, window_overlap=window_overlap)
    frames = t_rel + time[0]
    centers = np.asarray(band_centers_hz, dtype=float)
    lower, upper = fa.third_octave_band_edges(centers)
    use_filter_bank = third_octave_method == 'filter_bank'
    frame_samples = fa.frame_length(window_time, fs)

    def filter_bank(signal, frame_times):
        return pa_filters.third_octave_filter_bank(signal, fs, centers, frame_times, frame_samples) / fa.P_REF ** 2

    # Power per bin (pref^2/Hz) or per band (pref^2), and the matching ambient,
    # selected and applied as depropagate_hemisphere does.
    power = filter_bank(pressure, t_rel) if use_filter_bank else 10.0 ** (psd_db / 10.0)
    if ambient_pressure is not None:
        quiet = np.asarray(ambient_pressure, dtype=float)
        _, t_amb, ambient_db = fa.spectrogram(quiet, fs, window_time=window_time, window_overlap=window_overlap)
        ambient = np.nanmean(filter_bank(quiet, t_amb) if use_filter_bank else 10.0 ** (ambient_db / 10.0), axis=1)
    else:
        ambient = fa.ambient_from_frames(power, frames, ambient_time_range, ambient_percentile,
                                         power_db=None if use_filter_bank else psd_db)
    if ambient is not None:
        power = fa.gate_ambient(power, ambient, band_snr_gate_db)
    if use_filter_bank:
        bands = power
    elif tone_aware:
        bands = fa.tone_aware_band_power(power, f, lower, upper)
    else:
        df = f[1] - f[0]
        bands = np.array([np.sum(power[(f >= lo) & (f < hi)], axis=0) * df for lo, hi in zip(lower, upper)])
    return frames, fa.power_to_db(np.reshape(bands, (len(lower), t_rel.size)))


def residual_statistics(residual_db):
    """Summary of residuals, dB, over the finite values of each band.

    Args:
        residual_db: (Nb, N) synthesized minus measured levels, dB (NaN or infinite left out).

    Returns:
        dict of (Nb,) arrays: ``count``, ``mean_db``, ``median_db``, ``rms_db``,
        ``std_db`` (sample standard deviation), ``p05_db`` and ``p95_db``; NaN
        for a band with no finite residual (``std_db`` with fewer than two).
    """
    residual_db = np.asarray(residual_db, dtype=float)
    finite = np.isfinite(residual_db)
    values = np.where(finite, residual_db, np.nan)
    count = finite.sum(axis=1)
    stats = dict(count=count)
    with np.errstate(invalid='ignore', divide='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)       # all-NaN bands
        stats['mean_db'] = np.nanmean(values, axis=1)
        stats['median_db'] = np.nanmedian(values, axis=1)
        stats['rms_db'] = np.sqrt(np.nanmean(values ** 2, axis=1))
        stats['std_db'] = np.where(count > 1, np.nanstd(values, axis=1, ddof=1), np.nan)
        stats['p05_db'] = np.nanpercentile(values, 5.0, axis=1)
        stats['p95_db'] = np.nanpercentile(values, 95.0, axis=1)
    return stats


def _pick(values, indices, nmics, name):
    """The given microphones' entries of a per-microphone argument (list, (Nmics, ...) array, or shared 1D)."""
    if values is None:
        return None
    if isinstance(values, (list, tuple)):
        if len(values) != nmics:
            raise ValueError('{} must have one entry per microphone'.format(name))
        return [values[i] for i in indices]
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array
    if array.shape[0] != nmics:
        raise ValueError('{} must be shaped (Nmics, Nsamples) or be one 1D array'.format(name))
    return array[list(indices)]


def _per_mic(values, index):
    """One microphone's 1D entry of a per-microphone argument picked by :func:`_pick`."""
    return values if isinstance(values, np.ndarray) and values.ndim == 1 else values[index]


def leave_mics_out(mic_locations, pressure, time, track_time, track_position, track_velocity=None, *,
                   held_out=None, ambient_pressure=None, **depropagation):
    """Depropagate without some microphones, synthesize them, and compare.

    For each group of held-out microphones, the others are depropagated with
    ``depropagate_hemisphere(**depropagation, third_octave=True)``, and each
    held-out microphone's band histories are synthesized from that hemisphere
    (:func:`synthesize_band_histories`) at its spectrogram frame times and
    compared with its measured bands (:func:`measured_band_histories`).  The
    synthesis takes the depropagation's speed of sound, ray model, receiver
    response, absorption (``apply_absorption_deprop``, ``atmosphere``,
    ``absorption_band_method``, ``absorption_profile``; with
    ``third_octave_method='fft'``, whose bins were corrected one by one, the
    band absorption is the bins' mean, ``absorption_per_bin``), Doppler convention
    (``remove_doppler``, ``air_relative_doppler``), ``flip_y_for_geometry``
    and ``track_nose``; frames the depropagation would have dropped
    (``min_elevation_deg``, ``max_range`` with ``rim_range``) are left out of
    the comparison.  The held-out microphones' bands are measured as the
    depropagation forms them (``third_octave_method``, ``tone_aware``) and
    gated against the same ambient (``ambient_pressure``,
    ``ambient_time_range`` or ``ambient_percentile``), at received
    frequencies.  Residuals are synthesized minus measured, dB, over the
    frames where both are finite: a band gated to no energy, or a direction
    the hemisphere does not cover, is not counted.

    Args:
        mic_locations, pressure, time, track_time, track_position, track_velocity:
            as for ``depropagate_hemisphere`` (lengths in its ``length_units``).
        held_out: sequence of groups of microphone indices to hold out in turn.
            Default: each microphone alone (leave one out).
        ambient_pressure: optional per-microphone ambient recordings, Pa, as for
            ``depropagate_hemisphere``; also used to gate the held-out bands.
        **depropagation: keyword arguments of ``depropagate_hemisphere``
            (``third_octave`` is always on).

    Returns:
        dict of ``band_centers_hz`` (Nb,) Hz; ``folds``, one dict per group with
        ``held_out`` and ``kept`` (index lists), ``hemisphere`` (the
        depropagation's result) and per held-out microphone ``time`` (Nframes,) s,
        ``measured_db``, ``synthesized_db`` and ``residual_db`` (Nb, Nframes) dB;
        ``statistics`` (:func:`residual_statistics` of every residual pooled,
        per band); and ``overall``, the same pooled over all bands, as scalars.
    """
    if depropagation.get('third_octave') is False:
        raise ValueError('leave_mics_out needs third_octave band levels')
    depropagation['third_octave'] = True
    mic_locations = np.asarray(mic_locations, dtype=float)
    nmics = mic_locations.shape[0]
    if held_out is None:
        held_out = [[i] for i in range(nmics)]
    # The depropagation's settings, its own defaults filled in, so the synthesis
    # and the measurement follow whatever the hemisphere was built with.
    setting = {key: parameter.default
               for key, parameter in inspect.signature(fa.depropagate_hemisphere).parameters.items()
               if parameter.default is not inspect.Parameter.empty}
    setting.update(depropagation)
    response = setting['receiver_response_db']
    synthesis = dict(
        speed_of_sound=setting['speed_of_sound'],
        length_units=setting['length_units'],
        ray_model=setting['ray_model'],
        apply_absorption=setting['apply_absorption_deprop'],
        atmosphere=setting['atmosphere'],
        absorption_band_method=setting['absorption_band_method'],
        absorption_profile=setting['absorption_profile'],
        doppler_shift=setting['remove_doppler'],
        air_relative_doppler=setting['air_relative_doppler'],
        flip_y_for_geometry=bool(setting['flip_y_for_geometry']),
        absorption_per_bin=setting['third_octave_method'] == 'fft')
    measurement = {key: setting[key] for key in ('window_time', 'window_overlap', 'ambient_time_range',
                                                 'ambient_percentile', 'band_snr_gate_db', 'third_octave_method')}
    measurement['tone_aware'] = bool(setting['tone_aware'])
    track = dict(track_time=track_time, track_position=track_position, track_velocity=track_velocity,
                 track_nose=setting['track_nose'])

    folds, residuals, band_centers = [], [], None
    for group in held_out:
        group = [int(i) for i in group]
        kept = [i for i in range(nmics) if i not in group]
        if not kept or not group:
            raise ValueError('each held-out group must leave at least one microphone in and take one out')
        kept_response = None if response is None else (lambda im, bands, offset, kept=kept:
                                                      response(kept[im], bands, offset))
        hemisphere = fa.depropagate_hemisphere(
            mic_locations[kept], _pick(pressure, kept, nmics, 'pressure'), _pick(time, kept, nmics, 'time'),
            track_time, track_position, track_velocity,
            **dict(depropagation, receiver_response_db=kept_response,
                   ambient_pressure=_pick(ambient_pressure, kept, nmics, 'ambient_pressure'),
                   ambient_time=_pick(depropagation.get('ambient_time'), kept, nmics, 'ambient_time')))
        band_centers = hemisphere['third_octave']['band_centers_hz']
        fold = dict(held_out=group, kept=kept, hemisphere=hemisphere, time=[], measured_db=[],
                    synthesized_db=[], residual_db=[])
        group_pressure = _pick(pressure, group, nmics, 'pressure')
        group_time = _pick(time, group, nmics, 'time')
        group_ambient = _pick(ambient_pressure, group, nmics, 'ambient_pressure')
        for k, im in enumerate(group):
            frames, measured = measured_band_histories(
                _per_mic(group_pressure, k), _per_mic(group_time, k), band_centers,
                ambient_pressure=None if group_ambient is None else _per_mic(group_ambient, k), **measurement)
            mic_response = None if response is None else (lambda _, bands, offset, im=im: response(im, bands, offset))
            synthesized = synthesize_band_histories(hemisphere, track, mic_locations[im][None, :], times=frames,
                                                    receiver_response_db=mic_response, **synthesis)
            keep = fa.range_cap_mask(synthesized['elevation_deg'][0], synthesized['range'][0],
                                     float(setting['min_elevation_deg']), setting['max_range'], setting['rim_range'])
            predicted = np.where(keep[None, :], synthesized['bands_db'][0], np.nan)
            with np.errstate(invalid='ignore'):
                residual = predicted - measured
            residual[~(np.isfinite(predicted) & np.isfinite(measured))] = np.nan
            fold['time'].append(frames)
            fold['measured_db'].append(measured)
            fold['synthesized_db'].append(predicted)
            fold['residual_db'].append(residual)
            residuals.append(residual)
        folds.append(fold)
    pooled = np.concatenate(residuals, axis=1)
    overall = {key: value[0] for key, value in residual_statistics(pooled.reshape(1, -1)).items()}
    return dict(band_centers_hz=band_centers, folds=folds, statistics=residual_statistics(pooled), overall=overall)
