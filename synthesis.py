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

import warnings

import numpy as np

import flight_acoustics as fa
import ground_plane as gp
from panam_acoustics.atmosphere import Atmosphere, band_absorption_db
from unit_conversion import len_conv

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
        velocity = np.gradient(position, axis=0) / np.gradient(time)[:, None]
    velocity = np.asarray(velocity, dtype=float)
    nose = track.get('track_nose')
    nose = None if nose is None else np.asarray(nose, dtype=float)

    def interpolate(values):
        def at(t):
            t = np.asarray(t, dtype=float)
            return np.column_stack([np.interp(t, time, values[:, k]) for k in range(values.shape[1])])
        return at
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
    for _ in range(max_iter):
        new = times - ray_model(x, mic[None, :])['travel_time'][:, 0]
        step = np.nanmax(np.abs(new - t_e), initial=0.0)
        t_e, x = new, np.asarray(position(new), dtype=float)
        if step < tol:
            return t_e, x
    raise RuntimeError('ray emission times did not converge in {} passes (last step {:.3g} s)'.format(max_iter, step))


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


def synthesize_band_histories(sphere, track, mics, *, times=None, speed_of_sound=1135.0, r_ref=None,
                              length_units=None, ray_model=None, apply_absorption=False, atmosphere=None,
                              absorption_band_method='center', receiver_response_db=None, doppler_shift=False,
                              air_relative_doppler=False):
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
        atmosphere: :class:`panam_acoustics.atmosphere.Atmosphere` for absorption.
            Default the sphere's metadata atmosphere, else 293.15 K, 101.325 kPa, 20% RH.
        absorption_band_method: band absorption method
            (:func:`panam_acoustics.atmosphere.band_absorption_db`), default 'center'.
        receiver_response_db: None, or ``f(im, band_centers_hz, source_offset)``
            returning the band response re free field, dB, (Nb, N), as
            ``depropagate_hemisphere`` takes it; ``source_offset`` (N, 3) is the
            emission position minus microphone ``im`` (or the ray model's offset).
        doppler_shift: the hemisphere is filed at emitted frequencies
            (``depropagate_hemisphere(remove_doppler=True)``): shift each band to
            the received frequencies.  Default False: the hemisphere is filed at
            received frequencies, as a flight sphere for NICE-OPS is.
        air_relative_doppler: with ``doppler_shift`` and a ray model returning
            ``emission_direction`` and ``doppler_speed``, take D = 1 / (1 - v.n /
            (c + W.n)) instead of the straight line's 1 / (1 - v.u / c).

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
        weather = metadata.get('atmosphere')
        atmosphere = (Atmosphere(temperature=293.15, pressure=101.325, relative_humidity=20.0) if weather is None
                      else Atmosphere(temperature=weather['temperature_k'], pressure=weather['pressure_kpa'],
                                      relative_humidity=weather['relative_humidity']))
    if air_relative_doppler and not (doppler_shift and ray_model is not None):
        raise ValueError('air_relative_doppler needs doppler_shift=True and a ray model')

    mics = np.asarray(mics, dtype=float)
    if mics.ndim != 2 or mics.shape[1] != 3:
        raise ValueError('mics must be shaped (Nmics, 3)')
    track_time, position, velocity, nose = _track_functions(track)
    times = track_time.copy() if times is None else np.asarray(times, dtype=float).ravel()
    c = float(speed_of_sound)

    absorption_reference = None
    if apply_absorption:
        weather = (atmosphere.temperature, atmosphere.pressure, atmosphere.relative_humidity)
        r_ref_m = float(len_conv(r_ref, from_units=length_units, to_units='m'))
        absorption_reference = band_absorption_db(r_ref_m, centers, *weather, method=absorption_band_method)

    fields = ('emission_time', 'azimuth_deg', 'elevation_deg', 'range', 'spreading_range', 'doppler')
    out = {key: np.full((mics.shape[0], times.size), np.nan) for key in fields}
    out['bands_db'] = np.full((mics.shape[0], centers.size, times.size), np.nan)
    for im, mic in enumerate(mics):
        t_e, x = _emission_times(position, times, mic, c, ray_model)
        v = velocity(t_e)
        az, el, r, _, _ = fa.hemigen(t_e, x, v, mic[None, :], c, nose=None if nose is None else nose(t_e))
        az, el, r = az[:, 0], el[:, 0], r[:, 0]
        toward = (mic[None, :] - x) / r[:, None]
        doppler = 1.0 / (1.0 - np.sum(v * toward, axis=1) / c)
        spread, path, offset = r, r, x - mic[None, :]
        valid = (t_e >= track_time[0]) & (t_e <= track_time[-1])
        if ray_model is not None:
            rays = ray_model(x, mic[None, :])
            el = np.asarray(rays['depression_deg'], dtype=float)[:, 0]
            spread = np.asarray(rays['spreading_range'], dtype=float)[:, 0]
            path = np.asarray(rays['path_length'], dtype=float)[:, 0]
            offset = np.asarray(rays['offset'], dtype=float)[:, 0, :]
            valid &= np.asarray(rays['valid'], dtype=bool)[:, 0] & np.all(np.isfinite(offset), axis=1)
            if 'emission_direction' in rays:
                emitted = np.asarray(rays['emission_direction'], dtype=float)[:, 0, :]
                turn = np.degrees(np.arctan2(emitted[:, 1], emitted[:, 0]) - np.arctan2(toward[:, 1], toward[:, 0]))
                az = np.remainder(az + turn, 360.0)
                if air_relative_doppler:
                    if 'doppler_speed' not in rays:
                        raise ValueError('air_relative_doppler needs a ray model that returns doppler_speed')
                    speed_along = np.sum(v * emitted, axis=1)
                    doppler = 1.0 / (1.0 - speed_along / np.asarray(rays['doppler_speed'], dtype=float)[:, 0])
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
            path_m = np.asarray(len_conv(path, from_units=length_units, to_units='m'), dtype=float)
            delta = band_absorption_db(path_m, centers, *weather, method=absorption_band_method)
            power = power * 10.0 ** (-(delta - absorption_reference[:, None]) / 10.0)
        power = np.where(valid[None, :], power, np.nan)
        out['bands_db'][im] = fa.power_to_db(power)
        for key, value in zip(fields, (t_e, az, el, r, spread, doppler)):
            out[key][im] = np.where(valid, value, np.nan)
    out['time'] = times
    out['band_centers_hz'] = centers
    return out


def measured_band_histories(pressure, time, band_centers_hz, *, window_time=0.5, window_overlap=0.5,
                            ambient_pressure=None, band_snr_gate_db=3.0):
    """One microphone's one-third octave band levels per spectrogram frame.

    The bands are formed as ``depropagate_hemisphere``'s FFT path forms them
    without Doppler removal: the spectrogram (:func:`flight_acoustics.spectrogram`,
    Hann window) summed over the bins in [f_l, f_u), each bin first gated and
    ambient-subtracted against the mean ambient spectrum when
    ``ambient_pressure`` is given.

    Args:
        pressure: (Nsamples,) pressure, Pa.
        time: (Nsamples,) sample times, s (evenly spaced).
        band_centers_hz: (Nb,) band centers, Hz.
        window_time: spectrogram window, s.
        window_overlap: window overlap fraction.
        ambient_pressure: optional (Nsamples_a,) signal-free recording, Pa, at
            the same sample rate.
        band_snr_gate_db: bins less than this far above the ambient are zeroed, dB.

    Returns:
        (frame_times, bands_db): (Nframes,) frame centers, s, and (Nb, Nframes)
        band levels, dB re 20 uPa (-inf where gating left no energy).
    """
    time = np.asarray(time, dtype=float)
    fs = float(np.round(1.0 / (time[1] - time[0])))
    f, t_rel, psd_db = fa.spectrogram(np.asarray(pressure, dtype=float), fs, window_time=window_time,
                                      window_overlap=window_overlap)
    psd = 10.0 ** (psd_db / 10.0)
    if ambient_pressure is not None:
        _, _, ambient_db = fa.spectrogram(np.asarray(ambient_pressure, dtype=float), fs, window_time=window_time,
                                          window_overlap=window_overlap)
        ambient = np.nanmean(10.0 ** (ambient_db / 10.0), axis=1)[:, None]
        psd = np.where(psd >= ambient * 10.0 ** (band_snr_gate_db / 10.0), np.maximum(psd - ambient, 0.0), 0.0)
    df = f[1] - f[0]
    lower, upper = fa.third_octave_band_edges(band_centers_hz)
    bands = np.array([np.sum(psd[(f >= lo) & (f < hi)], axis=0) * df for lo, hi in zip(lower, upper)])
    return t_rel + time[0], fa.power_to_db(bands.reshape(len(lower), t_rel.size))


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
    ``absorption_band_method``), Doppler convention (``remove_doppler``,
    ``air_relative_doppler``) and ``track_nose``; frames the depropagation
    would have dropped (``min_elevation_deg``, ``max_range`` with
    ``rim_range``) are left out of the comparison.  Residuals are synthesized
    minus measured, dB, over the frames where both are finite: a band gated
    to no energy, or a direction the hemisphere does not cover, is not counted.

    The measured bands are brick-wall sums of the spectrogram at received
    frequencies, so compare against a hemisphere built with the FFT bands;
    ``tone_aware`` and the filter bank form bands differently.

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
    response = depropagation.get('receiver_response_db')
    synthesis = dict(
        speed_of_sound=depropagation.get('speed_of_sound', 1135.0),
        length_units=depropagation.get('length_units', 'ft'),
        ray_model=depropagation.get('ray_model'),
        apply_absorption=depropagation.get('apply_absorption_deprop', False),
        atmosphere=depropagation.get('atmosphere'),
        absorption_band_method=depropagation.get('absorption_band_method', 'center'),
        doppler_shift=depropagation.get('remove_doppler', False),
        air_relative_doppler=depropagation.get('air_relative_doppler', False))
    window = dict(window_time=depropagation.get('window_time', 0.5),
                  window_overlap=depropagation.get('window_overlap', 0.5))
    min_elevation = float(depropagation.get('min_elevation_deg', 0.0))
    max_range = depropagation.get('max_range')
    rim_range = depropagation.get('rim_range')
    track = dict(track_time=track_time, track_position=track_position, track_velocity=track_velocity,
                 track_nose=depropagation.get('track_nose'))

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
                   ambient_pressure=_pick(ambient_pressure, kept, nmics, 'ambient_pressure')))
        band_centers = hemisphere['third_octave']['band_centers_hz']
        fold = dict(held_out=group, kept=kept, hemisphere=hemisphere, time=[], measured_db=[],
                    synthesized_db=[], residual_db=[])
        group_pressure = _pick(pressure, group, nmics, 'pressure')
        group_time = _pick(time, group, nmics, 'time')
        group_ambient = _pick(ambient_pressure, group, nmics, 'ambient_pressure')
        for k, im in enumerate(group):
            frames, measured = measured_band_histories(
                _per_mic(group_pressure, k), _per_mic(group_time, k), band_centers,
                ambient_pressure=None if group_ambient is None else _per_mic(group_ambient, k),
                band_snr_gate_db=depropagation.get('band_snr_gate_db', 3.0), **window)
            mic_response = None if response is None else (lambda _, bands, offset, im=im: response(im, bands, offset))
            synthesized = synthesize_band_histories(hemisphere, track, mic_locations[im][None, :], times=frames,
                                                    receiver_response_db=mic_response, **synthesis)
            keep = np.ones(frames.size, dtype=bool)
            elevation, distance = synthesized['elevation_deg'][0], synthesized['range'][0]
            if min_elevation > 0.0:
                keep &= elevation >= min_elevation
            if max_range is not None:
                cap = np.full(frames.size, float(max_range))
                if rim_range is not None:
                    cap = np.where(elevation < float(rim_range[0]), max(float(rim_range[1]), float(max_range)), cap)
                keep &= distance <= cap
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
