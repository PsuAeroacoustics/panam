"""Depropagation: microphone recordings to a source hemisphere at a reference radius.

:func:`depropagate_hemisphere` frames each channel, forms band power, gates it
against the ambient, divides out the receiver response, undoes spherical
spreading and atmospheric absorption back to ``r_ref`` and grids the samples
(``docs/THEORY.md`` section 5).  Lengths are in its ``length_units``.
"""

import warnings
from typing import Any, Optional, cast

import numpy as np

from panam import unit_conversion
from panam.acoustics import filters as pa_filters
from panam.acoustics.atmosphere import Atmosphere, band_absorption_db, check_absorption_profile
from panam.spectra import (BASE2_BAND_CENTERS, dBAw, frame_length, P_REF, power_to_db, spectrogram,
                           third_octave_band_edges, _band_integrals, _bands_missing, _bands_within,
                           _tone_aware_bands, _within_quarter_band)
from panam.geometry import finite_difference_velocity, hemigen
from panam.absorption import path_absorption_db, _check_band_method
from panam.gridding import adaptive_idw_weights, ADAPTIVE_INTERPOLATION, shepIDW_apply, shepIDW_weights


_AIR_RELATIVE_DOPPLER_RAYS = ('air_relative_doppler needs a ray model that returns emission_direction '
                             'and doppler_speed (refracted_rays.uniform_wind_ray_model)')


def ambient_from_frames(power, frames, ambient_time_range=None, ambient_percentile=None, power_db=None):
    """(Nf,) ambient power from a record's own frames, or None.

    The mean of ``power`` (Nf, Nframes) over the frames whose times ``frames``
    (Nframes,) s fall in ``ambient_time_range`` (t0, t1) s; else its
    ``ambient_percentile`` (0-100) per row, taken in dB over ``power_db`` when
    given (spectrogram bins) and in power otherwise (filter-bank bands); else None.
    """
    if ambient_time_range is not None:
        quiet = np.logical_and(frames >= float(ambient_time_range[0]), frames <= float(ambient_time_range[1]))
        if not np.any(quiet):
            raise ValueError('ambient_time_range contains no spectrogram frames')
        return np.nanmean(power[:, quiet], axis=1)
    if ambient_percentile is not None:
        if power_db is not None:
            return 10.0 ** (np.nanpercentile(power_db, float(ambient_percentile), axis=1) / 10.0)
        return np.nanpercentile(power, float(ambient_percentile), axis=1)
    return None


def gate_ambient(power, ambient, band_snr_gate_db):
    """Gate ``power`` (Nf, N) against ``ambient`` (Nf,) and subtract it: a value
    less than ``band_snr_gate_db`` dB above the ambient becomes 0 (measured, no
    energy left), any other loses the ambient (floored at 0)."""
    ambient = ambient[:, None]
    return np.where(power >= ambient * 10.0 ** (band_snr_gate_db / 10.0), np.maximum(power - ambient, 0.0), 0.0)


def range_cap_mask(elevation_deg, distance, min_elevation_deg=0.0, max_range=None, rim_range=None):
    """Samples ``depropagate_hemisphere`` keeps by elevation and range: elevation
    (deg, UMAPR) at least ``min_elevation_deg`` (when positive) and ``distance``
    within ``max_range``, raised to ``rim_range[1]`` below ``rim_range[0]`` deg.
    Lengths in one unit; returns a boolean array shaped like ``distance``."""
    keep = np.ones(np.shape(distance), dtype=bool)
    if min_elevation_deg > 0.0:
        keep &= elevation_deg >= float(min_elevation_deg)
    if max_range is not None:
        cap = np.full(np.shape(distance), float(max_range))
        if rim_range is not None:
            cap = np.where(elevation_deg < float(rim_range[0]), max(float(rim_range[1]), float(max_range)), cap)
        keep &= distance <= cap
    return keep


def ray_azimuth_deg(azimuth_deg, emitted, toward):
    """The straight line's azimuth (deg, UMAPR) turned by the horizontal angle
    from ``toward`` (the source-to-receiver line) to ``emitted`` (the direction
    a ray left the source along), both (..., 3): a crosswind's drift."""
    turn = np.degrees(np.arctan2(emitted[..., 1], emitted[..., 0]) - np.arctan2(toward[..., 1], toward[..., 0]))
    return np.remainder(azimuth_deg + turn, 360.0)


def air_relative_doppler_factor(velocity, emitted, doppler_speed):
    """D = 1 / (1 - v.n / doppler_speed): ``velocity`` and the emission
    direction ``emitted`` (..., 3), ``doppler_speed`` (...) the ray model's
    c + W.n, in matching length units per second."""
    return 1.0 / (1.0 - np.sum(velocity * emitted, axis=-1) / doppler_speed)


def depropagate_hemisphere(
        mic_locations,
        pressure,
        time,
        track_time,
        track_position,
        track_velocity=None,
        *,
        speed_of_sound=1135.0,
        length_units='ft',
        r_ref=100.0,
        freq_range=(0.0, 2000.0),
        window_time=0.5,
        window_overlap=0.5,
        point_stride=1,
        azi_step=10.0,
        elv_step=10.0,
        rmax=25.0,
        apply_absorption_deprop=False,
        atmosphere=Atmosphere(temperature=293.15, pressure=101.325, relative_humidity=20.0),
        flip_y_for_geometry=False,
        return_scattered=False,
        third_octave=False,
        third_octave_fmin=20.0,
        third_octave_band_centers_hz=None,
        third_octave_method='fft',
        narrowband=False,
        narrowband_stride=1,
        min_elevation_deg=0.0,
        max_range=None,
        ambient_time_range=None,
        ambient_pressure=None,
        ambient_time=None,
        ambient_percentile=None,
        band_snr_gate_db=3.0,
        max_absorption_correction_db=None,
        max_response_correction_db=None,
        receiver_response_db=None,
        ray_model=None,
        interpolation=None,
        rim_range=None,
        remove_doppler=False,
        tone_aware=False,
        decompose_tonal=False,
        track_nose=None,
        air_relative_doppler=False,
        absorption_band_method='center',
        absorption_profile=None,
        return_statistics=False,
):
    """Generate an acoustic hemisphere from microphone time series and vehicle tracking data.

    Emission points whose elevation angle falls below ``min_elevation_deg``
    are excluded before gridding. Low-elevation points correspond to long
    ranges where the microphone signal is usually ambient-dominated;
    depropagating them multiplies the ambient floor by (r/r_ref)^2 and
    contaminates the hemisphere rim, so gate them out (and restrict
    track_time to the high-SNR portion of the pass) unless the data are
    known to be signal-dominated all the way down. Near-grazing incidence is
    also where ground impedance dominates what the microphone hears, which
    the depropagation does not model, so around 10 degrees is a sensible
    floor for a ground-board array rather than 0.

    ``max_range`` drops emission point / microphone pairs whose separation
    exceeds it, in ``length_units``. Depropagation assumes a straight ray
    through a homogeneous atmosphere; over long paths refraction by the wind
    and temperature profile makes that a poor model, quite apart from the
    absorption correction growing beyond what the measurement supports.

    Ambient gating (strongly recommended) removes the dominant failure mode
    of depropagation: a band that is ambient-limited at the microphone
    carries no source information, but spreading and especially absorption
    depropagation multiply it by (r/r_ref)^2 and 10^(alpha(f)(r-r_ref)/10).
    At the top third-octave bands alpha is tens of dB/km, so an ambient-
    limited 10 kHz band at a kilometer of range is amplified into a
    physically impossible source level -- the Be407 spheres reach 216 dB at
    10 kHz on the aft pole this way. A broadband SNR gate cannot catch it,
    because the broadband level is dominated by the low-frequency bands
    where the signal is strong.

    Supply the ambient reference in one of three ways:

    * ``ambient_pressure`` -- a separate signal-free recording per
      microphone (same channel order as ``mic_locations``), e.g. a dedicated
      ambient run from the same array and test day. Optionally pass
      ``ambient_time`` alongside; only the sample rate is used, so a bare
      pressure array is enough. This is the preferred form: the ambient is
      measured rather than carved out of the run.
    * ``ambient_time_range=(t0, t1)`` -- a signal-free window of the run
      recording itself, in the same absolute time base as ``time``.
    * ``ambient_percentile`` -- a low percentile of the run's own spectrogram,
      per microphone and per frequency bin. Use it when no ambient recording
      exists for that array layout. It assumes the quietest few percent of
      frames are signal-free: checked against the median spectrogram of B407's
      measured ambient runs, the 5th percentile agrees to -0.3 dB in the
      median (so about 1.9 dB below their mean, which the measured ambient now
      uses), but with roughly +-8 dB of scatter per channel, so it is a
      fallback rather than an equivalent. Note that a *higher* percentile is not safer -- 25 already
      overestimates that ambient by 8 dB. The bias runs the other way on a
      record that is mostly quiet, where a low percentile samples the low tail
      of the noise fluctuation and gates too little.

    Each microphone's ambient PSD is the mean spectrogram over the ambient
    frames (in power), computed with the same window settings as the run so
    the frequency grids match; with ``ambient_percentile`` it is that
    percentile instead.  It is the mean, not the median, because a single
    frame's bin of noise is exponentially distributed and its median is ln 2
    of the mean (-1.6 dB), while the run's bins, from which it is subtracted,
    are single frames whose expectation is the mean.  It is then applied per
    frequency bin *before* depropagation: bins whose measured PSD is within
    ``band_snr_gate_db`` of that mic's ambient are zeroed (they carry no
    usable signal), and the ambient power is subtracted from the bins that
    pass.

    An SNR gate alone is not sufficient, because it bounds the *relative*
    error of a bin but says nothing about how far the absorption correction
    extrapolates. At 10 kHz on a warm day alpha is around 90 dB/km, so an
    emission point 2.7 km away carries a correction near 250 dB: an ambient
    fluctuation that clears any plausible SNR gate still lands at a source
    level hundreds of dB too high, and because the hemisphere averages in
    linear power, one such point dominates its whole neighborhood. Set
    ``max_absorption_correction_db`` to discard bins whose absorption
    correction exceeds what the measurement can support -- the band is simply
    not observable at that range, and a gap there is the honest result. It has
    no effect unless ``apply_absorption_deprop=True``.  ``max_response_correction_db``
    likewise discards bands whose receiver response (``receiver_response_db``) would have to
    be divided out by more than that many dB: near grazing a ground board's modeled
    response is a deep null, and dividing by it turns small measurement and model errors
    into tens of dB.  A discarded bin is missing, not zero: a band any of whose bins
    was discarded is missing for that sample, the overall levels sum the bins that
    remain, and the gridding leaves missing samples out of each node's weights.

    ``receiver_response_db``, if given, removes what the microphone's
    installation adds.  It is called as ``receiver_response_db(im, bands,
    source_offset)``, with ``bands`` the one-third octave centers (the output
    bands when ``third_octave``), ``source_offset`` the (Npts, 3) emission
    positions minus microphone ``im``'s position, and returns the band-averaged
    level re free field, dB, shaped (len(bands), Npts) -- for instance a
    ground plate's response (:func:`axisymmetric_bem.board_level`).  Each bin
    is divided by its band's value, after the ambient gate and before
    spreading, so a band's energy is divided by the band-averaged response.
    The pressures should then be supplied unscaled (no 0.5 pressure-doubling
    factor).

    ``ray_model``, if given, replaces the straight line in uniform air with a
    refracted ray (see :mod:`refracted_rays`, which defines the interface and
    wraps NICEOPS's ray tracer).  For each sampled emission point and
    microphone it supplies where the sample is filed (the ray's launch
    depression in place of the straight line's elevation; the azimuth is
    unchanged), the reception time, the range spherical spreading is undone
    over (the ray tube's), the arc absorption is undone over, and the source
    offset ``receiver_response_db`` sees (the equivalent geometry at the ray's
    arrival angle).  Pairs it marks invalid (no ray arrives) are skipped.
    ``min_elevation_deg`` then applies to the filing angle; ``max_range``
    stays on the straight-line distance.  Without one the result is unchanged.
    A ray model that also returns ``emission_direction`` (a uniform wind's,
    :func:`refracted_rays.uniform_wind_ray_model`) refiles the azimuth along it
    as well, turned from the straight line's by the horizontal angle between them.
    ``air_relative_doppler`` (with ``remove_doppler``) takes the Doppler factor
    along that direction in the moving air, D = 1 / (1 - v.n / (c + W.n)), from
    the model's ``emission_direction`` and ``doppler_speed`` (length units per
    second); default False, the straight line's factor in still air.

    ``rim_range``, if given as (elevation_deg, range), lets samples filed less than
    ``elevation_deg`` below the horizon come from as far as ``range`` instead of
    ``max_range``.  The range cap keeps a pass's samples to its steeper directions:
    at 2,000 ft a pass at 480 ft never files below 14 deg, so a sphere's rim came
    from low passes alone, and on the 2017 B407 data high passes of the same
    condition imply a forward rim 1-3.5 dB louder (and an aft rim ~1 dB quieter)
    than low passes do at the same angle.  With a ``ray_model`` the filing angle is
    the ray's launch angle and the far samples are depropagated along it; the
    absorption cap (``max_absorption_correction_db``) still drops the bands the
    range makes unobservable.

    ``interpolation``, if given (a dict; ``{}`` takes the defaults in
    :data:`ADAPTIVE_INTERPOLATION`), grids the samples with a radius chosen per
    node instead of the fixed ``rmax``: wide enough for ``k`` samples from
    ``min_mics`` microphones and for the samples' own angular resolution, capped
    at ``max_radius_deg`` (see :func:`adaptive_idw_weights`).  A sample's
    resolution is the arc its analysis window smears over, the source's motion
    across the line of sight in ``window_time`` plus ``source_extent`` (a length,
    e.g. the rotor diameter), over the range.  Nodes beyond the cap are left NaN
    and counted in ``out['interpolation']['gaps']``.  Without it the fixed
    ``rmax`` gridding is unchanged.

    This is the "normal" processing flow used by the demo scripts: use the vehicle kinematics
    to compute emission-time geometry (azimuth/elevation/range) via :func:`hemigen`, sample
    the measured PSD at the corresponding observer times, depropagate to a reference radius,
    then interpolate onto a regular azimuth/elevation grid using :func:`shepIDW`.

    Notes on conventions
    --------------------
    - Inputs must be in a *consistent* local Cartesian frame.
    - If the measured field is mirrored left/right against a reference hemisphere, the
      lateral axis of the supplied geometry runs opposite to the convention here
      (azimuth 180 ahead, 90 starboard). Set ``flip_y_for_geometry=True`` to apply
      Y -> -Y to geometry inputs *before* computing azimuth/elevation.
    - DECIDE THIS IN ANGLE SPACE, NOT BY EYE. Interpolate the reference onto the same
      azimuth/elevation grid and compare residuals against the reference and against
      the reference read backwards in azimuth; take whichever fits. Judging it from the
      rendered plate cannot separate a geometry sign error from a plotting one -- that
      is how the pre-2026-08-19 label/projection mismatch (see :func:`lambert_lon`)
      stayed hidden. Such a check on the AS350 demo data once concluded the data
      needed the flip, but that was compensating for
      :func:`art2umapr` putting ART phi = +90 to port; with the AAM manual's sign
      (phi > 0 starboard) the same residuals favor no flip. A right-handed, z-up
      frame needs no flip.

    Args:
        mic_locations: (Nmics, 3) microphone locations in local Cartesian coordinates.
        pressure: microphone pressures, shape (Nmics, Nsamples) or list of 1D arrays.
        time: microphone sample times. Either:
            - 1D array (Nsamples,) applied to all microphones, or
            - 2D array (Nmics, Nsamples), or
            - list of 1D arrays.
        track_time: (Nt,) vehicle time array (seconds).
        track_position: (Nt, 3) vehicle position array in same frame as microphones.
        track_velocity: optional (Nt, 3) vehicle velocity array in same frame as microphones.
            If None, computed by finite differences.
        track_nose: optional (Nt, 3) direction, in the same frame, that the sphere's azimuth
            is measured from (azimuth 180 along it): the airframe's heading, say, for a sphere
            filed in the heading frame.  Only its horizontal direction is used.  Default None:
            the velocity's (the ground track), as before.  Doppler, the convective Mach number
            and the samples' angular resolution follow ``track_velocity`` either way, since
            they are the motion, not the orientation.
        speed_of_sound: speed of sound in the length units per second (e.g., ft/s if length_units='ft').
        length_units: length units for mic/track geometry ('ft' or 'm' typically).
        r_ref: reference radius for depropagation in ``length_units``.
        freq_range: (fmin, fmax) in Hz used for OASPL integration.
        window_time: spectrogram window time (s).
        window_overlap: spectrogram window overlap fraction.
        point_stride: decimation factor applied along emission time samples for speed.
        azi_step: regular grid spacing in azimuth (deg). 360 deg is included to close the seam.
        elv_step: regular grid spacing in elevation (deg).
        rmax: Shepard/IDW neighborhood radius (deg).
        apply_absorption_deprop: if True, undo atmospheric absorption between range r and r_ref.
        atmosphere: Atmosphere instance used when apply_absorption_deprop is True.
        flip_y_for_geometry: if True, apply Y -> -Y to track_position/track_velocity/mic_locations.
        third_octave: if True, also compute third-octave band level hemispheres.
        third_octave_fmin: minimum band center (Hz) when third_octave=True.
            Bands are also limited to those ``freq_range`` covers from edge to
            edge (:func:`third_octave_band_edges`); a band only partly inside
            it is left out, not summed over the part inside.
        tone_aware: if True, form third-octave band power with :func:`tone_aware_band_power`:
            tones are located to a fraction of a bin, removed with the Hann window's own shape,
            and filed whole in the band of their own frequency, instead of summing whole FFT
            bins between band edges, which hands a tone near an edge to the wrong band (+5 to
            +19 dB in the B407's 31.5 and 63 Hz bands).  Combines with remove_doppler.  FFT
            method, Hann window.
        decompose_tonal: if True, also return the tonal and broadband parts of the third-octave
            bands (``tone_aware_band_power(..., return_parts=True)``), each depropagated and
            gridded with the same weights as the total, under 'tonal_db' and 'broadband_db' in
            'third_octave' (and in the scattered 'third_octave' when return_scattered=True).
            The two sum to 'bands_db' in energy.  Needs third_octave=True and tone_aware=True
            (so the FFT method).
        remove_doppler: if True, file each sample's third-octave band power at the frequency it
            was emitted at, not received at.  A sample's Doppler factor is
            D = 1 / (1 - v.u / c), v the source velocity at emission and u the unit vector
            from the source to the microphone (straight line), so a band emitted over
            [f_lower, f_upper] arrives over [D f_lower, D f_upper]; the narrowband spectrum is
            integrated over those edges (its running integral interpolated linearly, so a
            tone moves bands whole).  Levels are left as received.  It matters where a tone
            sits near a band edge: at 46 kt the main rotor's 55 Hz harmonic is received at
            58-60 Hz ahead, in the 63 Hz band, and 50-52 Hz behind.  Only the FFT method
            supports it.  Use it for a sphere a hover is synthesized from; a flight sphere
            NICEOPS reads must keep received frequencies, since NICEOPS applies no shift.
            A band whose scaled edges reach past ``freq_range`` is missing for that sample
            rather than integrated over part of its width.
        air_relative_doppler: if True (with remove_doppler and a ray model that returns
            ``emission_direction`` and ``doppler_speed``), the Doppler factor is the moving
            air's, D = 1 / (1 - v.n / (c + W.n)), n the wavefront normal the sound left
            along and W the wind, instead of the straight line's in still air.  Default False.
        third_octave_method: how band levels are formed when third_octave=True.
            'fft' (default) sums the PSD bins between each band's edges, a
            brick-wall band.  'filter_bank' uses a true one-third octave
            filter bank (:func:`panam.acoustics.filters.third_octave_filter_bank`,
            order-3 Butterworth, as an analyzer implements), averaged over the
            same frames.  The two agree within about 0.5 dB above 100 Hz; below
            it the filter skirts carry a strong rotor tone into the
            neighboring bands, several dB for the bands between main-rotor
            harmonics, which is what analyzer-based data such as NORAH2
            contain.  In 'filter_bank' mode the ambient gate, subtraction and
            absorption cap act per band rather than per bin, and absorption is
            taken at the band center unless ``absorption_band_method`` says
            otherwise.  OASPL, A-weighted and narrowband output are FFT-based
            either way.
        absorption_band_method: how the filter-bank bands' absorption is formed
            (:func:`panam.acoustics.atmosphere.band_absorption_db`): 'center'
            (default, alpha at the band center, as before), 'arp5534' (SAE ARP
            5534) or 'integrated' (ISO 9613-1 integrated over the band's filter,
            flat band levels).  The correction is the band attenuation over the
            path less that over ``r_ref``, so the sphere keeps the band
            attenuation over ``r_ref``.  It needs ``apply_absorption_deprop``
            and ``third_octave_method='filter_bank'``: FFT bins are corrected
            one by one and need no band method.
        absorption_profile: None (default) for the uniform ``atmosphere``, or a
            layered atmosphere to undo the absorption of each path through:
            a mapping of ``altitude_ft`` (ft, strictly increasing, in the frame
            of the track's z, converted from ``length_units``),
            ``temperature_k`` (K), ``relative_humidity`` (percent) and
            ``pressure_kpa`` (kPa).  Each path's absorption is integrated along
            the straight line from source to microphone altitude
            (:func:`panam.acoustics.atmosphere.layered_attenuation_db`), over
            the ray's arc length when a ray model is given.  What is put back
            is the absorption over ``r_ref`` in ``atmosphere``, so the sphere
            keeps the same convention.  Needs ``apply_absorption_deprop``.
        return_statistics: if True, also describe the samples behind every gridded
            node (:func:`sphere_statistics.node_statistics`), with the weights the
            gridding uses, renormalized over the samples a field has at that node.
            For each field: ``n_samples`` (samples with a positive weight and a
            power that is not NaN), ``n_mics`` (distinct microphones among them),
            ``n_eff`` = (sum w)^2 / sum w^2 (0 where none) and ``sd_db``, the
            weighted standard deviation, dB, of the sample levels 10 lg p_i about
            the node's energy-mean level 10 lg(sum w p / sum w), over the samples
            with positive power (NaN where none).  Default False: nothing extra
            is computed.

    Returns:
        dict with keys:
            'azi_grid_deg', 'elv_grid_deg',
            'oaspl_db' (2D grid, unweighted),
            'spl_a_db' (2D grid, A-weighted overall level in dBA),
            and optionally:
                        - when return_scattered=True: 'scattered' (dict of scattered az/el/power samples)
                            If third_octave/narrowband are enabled, the corresponding scattered data are
                            also included under 'scattered'.
            - when third_octave=True: 'third_octave' (band centers + band grids, dB re 20 uPa
              at r_ref; with decompose_tonal=True also 'tonal_db' and 'broadband_db')
            - when narrowband=True: 'narrowband' (frequency + PSD grids)
            - when return_statistics=True: 'statistics', with 'oaspl_db' and
              'spl_a_db', and 'third_octave' and 'narrowband' dicts keyed as the
              gridded fields there ('bands_db', 'tonal_db', 'broadband_db',
              'psd_db'); each entry a dict of 'n_samples', 'n_mics', 'n_eff' and
              'sd_db' arrays shaped as the field it describes

        Grid cells with no sample within ``rmax`` are NaN: nothing was
        measured there.  Scattered samples a cap discarded are NaN too, and a
        cell's weights are taken over the samples it has.  Cells (and scattered
        samples) whose power is zero, e.g. gated out as ambient, are -inf:
        measured, with no energy left.
        Summing energy, a -inf contributes nothing and a NaN makes the sum
        unknown.
    """

    mic_locations = np.asarray(mic_locations, dtype=float)
    if mic_locations.ndim != 2 or mic_locations.shape[1] != 3:
        raise ValueError('mic_locations must be shape (Nmics, 3)')
    nmics = mic_locations.shape[0]

    track_time = np.asarray(track_time, dtype=float)
    track_position = np.asarray(track_position, dtype=float)
    if track_position.shape != (track_time.size, 3):
        raise ValueError('track_position must be shape (Nt, 3) matching track_time')

    if track_velocity is None:
        track_velocity = finite_difference_velocity(track_time, track_position)
    track_velocity = np.asarray(track_velocity, dtype=float)
    if track_velocity.shape != (track_time.size, 3):
        raise ValueError('track_velocity must be shape (Nt, 3) matching track_time')
    if track_nose is not None:
        track_nose = np.asarray(track_nose, dtype=float)
        if track_nose.shape != (track_time.size, 3):
            raise ValueError('track_nose must be shape (Nt, 3) matching track_time')

    def _as_mic_list_1d(x, *, name):
        if isinstance(x, list):
            if len(x) != nmics:
                raise ValueError(f'{name} list must have length Nmics={nmics}')
            return [np.asarray(xi, dtype=float).ravel() for xi in x]
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            return [arr.ravel() for _ in range(nmics)]
        if arr.ndim == 2 and arr.shape[0] == nmics:
            return [arr[i, :].ravel() for i in range(nmics)]
        raise ValueError(f'{name} must be a list of 1D arrays, a 1D array, or a 2D array shaped (Nmics, Nsamples)')

    pressure_list = _as_mic_list_1d(pressure, name='pressure')
    time_list = _as_mic_list_1d(time, name='time')

    ambient_sources = [name for name, value in
                       (('ambient_pressure', ambient_pressure),
                        ('ambient_time_range', ambient_time_range),
                        ('ambient_percentile', ambient_percentile))
                       if value is not None]
    if len(ambient_sources) > 1:
        raise ValueError('Give only one ambient source, not both of: ' + ', '.join(ambient_sources))
    if ambient_percentile is not None and not 0.0 < float(ambient_percentile) < 100.0:
        raise ValueError('ambient_percentile must lie strictly between 0 and 100')
    if tone_aware and third_octave_method != 'fft':
        raise ValueError('tone_aware needs third_octave_method=\'fft\'')
    if decompose_tonal and not (tone_aware and third_octave):
        raise ValueError('decompose_tonal needs third_octave=True and tone_aware=True')
    if air_relative_doppler and not remove_doppler:
        raise ValueError('air_relative_doppler applies to remove_doppler, which is off')
    if air_relative_doppler and ray_model is None:
        raise ValueError(_AIR_RELATIVE_DOPPLER_RAYS)
    if remove_doppler and third_octave_method != 'fft':
        raise ValueError('remove_doppler needs third_octave_method=\'fft\'')
    if third_octave_method not in ('fft', 'filter_bank'):
        raise ValueError("third_octave_method must be 'fft' or 'filter_bank'")
    use_filter_bank = bool(third_octave) and third_octave_method == 'filter_bank'
    _check_band_method(absorption_band_method)
    if absorption_band_method != 'center':
        if not apply_absorption_deprop:
            raise ValueError('absorption_band_method needs apply_absorption_deprop=True')
        if not use_filter_bank:
            raise ValueError("absorption_band_method applies to third_octave_method='filter_bank' "
                             "bands; FFT bins are corrected one by one")
    if absorption_profile is not None:
        if not apply_absorption_deprop:
            raise ValueError('absorption_profile needs apply_absorption_deprop=True')
        check_absorption_profile(absorption_profile)
    ambient_pressure_list = None
    ambient_time_list = None
    if ambient_pressure is not None:
        ambient_pressure_list = _as_mic_list_1d(ambient_pressure, name='ambient_pressure')
        if ambient_time is not None:
            ambient_time_list = _as_mic_list_1d(ambient_time, name='ambient_time')

    def _time_vec_for_mic(im):
        p = pressure_list[im]
        t = time_list[im]
        if p.size < 2:
            raise ValueError('pressure must have at least 2 samples per microphone')
        if t.size < 2:
            raise ValueError('time must have at least 2 samples per microphone (or provide a common 1D time vector)')
        if t.size == p.size:
            return t
        # Some files store a time vector that doesn't exactly match the signal length.
        # If dt is well-defined, reconstruct an evenly-sampled time vector aligned at t[0].
        dt = float(t[1] - t[0])
        if dt <= 0.0:
            raise ValueError('time must be strictly increasing')
        return float(t[0]) + dt * np.arange(p.size, dtype=float)

    # Optional geometry convention fix
    if flip_y_for_geometry:
        mic_geom = mic_locations.copy()
        mic_geom[:, 1] *= -1.0
        pos_geom = track_position.copy()
        pos_geom[:, 1] *= -1.0
        vel_geom = track_velocity.copy()
        vel_geom[:, 1] *= -1.0
        nose_geom = None
        if track_nose is not None:
            nose_geom = track_nose.copy()
            nose_geom[:, 1] *= -1.0
    else:
        mic_geom = mic_locations
        pos_geom = track_position
        vel_geom = track_velocity
        nose_geom = track_nose

    # Emission-time geometry
    az_deg, el_deg, r_geom, t_obs, _ = hemigen(track_time, pos_geom, vel_geom, mic_geom, speed_of_sound,
                                               nose=nose_geom)
    doppler_geom = None
    if remove_doppler:
        toward = np.asarray(mic_geom, dtype=float)[None, :, :] - np.asarray(pos_geom, dtype=float)[:, None, :]
        toward = toward / np.linalg.norm(toward, axis=2, keepdims=True)
        mach_toward = np.sum(np.asarray(vel_geom, dtype=float)[:, None, :] * toward, axis=2) / float(speed_of_sound)
        doppler_geom = 1.0 / (1.0 - mach_toward)

    # Subsample emission-time points
    point_stride = int(point_stride)
    if point_stride < 1:
        raise ValueError('point_stride must be >= 1')
    tidx = np.arange(0, track_time.size, point_stride)

    # Refracted rays, at the sampled points only: the filing angle and the
    # reception time replace hemigen's; the spreading and absorption ranges and
    # the receiver's source offset are kept alongside.
    spread_geom = path_geom = ray_offset = ray_valid = None
    if ray_model is not None:
        rays = ray_model(np.asarray(track_position, dtype=float)[tidx], np.asarray(mic_locations, dtype=float))
        shape = (tidx.size, nmics)
        for key in ('depression_deg', 'travel_time', 'spreading_range', 'path_length', 'valid'):
            if np.shape(rays[key]) != shape:
                raise ValueError('ray_model {!r} must be shaped {}, got {}'.format(key, shape, np.shape(rays[key])))
        if np.shape(rays['offset']) != shape + (3,):
            raise ValueError('ray_model offset must be shaped {}'.format(shape + (3,)))
        el_deg = np.array(el_deg, dtype=float)
        t_obs = np.array(t_obs, dtype=float)
        el_deg[tidx] = rays['depression_deg']
        t_obs[tidx] = np.asarray(track_time, dtype=float)[tidx][:, None] + rays['travel_time']
        spread_geom = np.full(r_geom.shape, np.nan)
        path_geom = np.full(r_geom.shape, np.nan)
        spread_geom[tidx] = rays['spreading_range']
        path_geom[tidx] = rays['path_length']
        ray_offset = np.array(rays['offset'], dtype=float)
        if flip_y_for_geometry:
            ray_offset[..., 1] *= -1.0
        ray_valid = np.asarray(rays['valid'], dtype=bool) & np.all(np.isfinite(ray_offset), axis=-1)
        if 'emission_direction' in rays:
            # Turn the straight line's azimuth by the horizontal angle from it to the
            # direction the sound left along (a crosswind's drift).
            emitted = np.array(rays['emission_direction'], dtype=float)
            if emitted.shape != shape + (3,):
                raise ValueError('ray_model emission_direction must be shaped {}'.format(shape + (3,)))
            if flip_y_for_geometry:
                emitted[..., 1] *= -1.0
            toward = np.asarray(mic_geom, dtype=float)[None, :, :] - np.asarray(pos_geom, dtype=float)[tidx][:, None, :]
            az_deg = np.array(az_deg, dtype=float)
            az_deg[tidx] = ray_azimuth_deg(az_deg[tidx], emitted, toward)
            ray_valid &= np.all(np.isfinite(emitted), axis=-1)
        if air_relative_doppler:
            if 'emission_direction' not in rays or 'doppler_speed' not in rays:
                raise ValueError(_AIR_RELATIVE_DOPPLER_RAYS)
            doppler_geom[tidx] = air_relative_doppler_factor(np.asarray(vel_geom, dtype=float)[tidx][:, None, :],
                                                             emitted, np.asarray(rays['doppler_speed'], dtype=float))

    # Regular hemisphere grid (degrees) with seam closure at 360
    azi_grid_deg = np.arange(0.0, 360.0 + 1e-9, float(azi_step))
    elv_grid_deg = np.arange(0.0, 90.0 + 1e-9, float(elv_step))
    AZI_GRID, ELV_GRID = np.meshgrid(azi_grid_deg, elv_grid_deg)

    # Frequency selection based on the first mic's sampling
    # (rounding matches the demo scripts' convention)
    t0_vec = _time_vec_for_mic(0)
    fs0 = float(np.round(1.0 / (t0_vec[1] - t0_vec[0])))
    fmin, fmax = float(freq_range[0]), float(freq_range[1])

    # Narrowband output frequency decimation
    narrowband_stride = int(narrowband_stride)
    if narrowband_stride < 1:
        raise ValueError('narrowband_stride must be >= 1')

    # Accumulate scattered samples
    fazi_list = []
    felv_list = []
    fmic_list = []
    fres_list = []
    frange_list = []
    fheight_list = []
    interp_settings = None
    if interpolation is not None:
        unknown = set(interpolation) - set(ADAPTIVE_INTERPOLATION)
        if unknown:
            raise ValueError('unknown interpolation settings: {}'.format(sorted(unknown)))
        interp_settings = dict(ADAPTIVE_INTERPOLATION, **interpolation)
    oaspl_power_list = []
    spl_a_power_list = []
    if third_octave:
        if third_octave_band_centers_hz is not None:
            band_centers = np.asarray(third_octave_band_centers_hz, dtype=float).ravel()
            band_centers = band_centers[np.isfinite(band_centers)]
            if band_centers.size < 1:
                raise ValueError('third_octave_band_centers_hz must contain at least one finite frequency')
            band_centers = np.unique(band_centers)
            band_centers = band_centers[band_centers >= float(third_octave_fmin)]
        else:
            # third_octave_fmin is compared within a quarter band so a nominal
            # limit keeps its own band (the 20 Hz band's exact center is 19.69 Hz).
            band_centers = BASE2_BAND_CENTERS[_within_quarter_band(BASE2_BAND_CENTERS, float(third_octave_fmin),
                                                                   np.inf)]
        # Only bands freq_range covers edge to edge: the FFT band sum sees no
        # bins outside it, so a band reaching past fmax (the 2 kHz band,
        # 1782-2245 Hz, with freq_range (0, 2000)) would come out ~3 dB low,
        # and the filter bank would disagree with it there.
        covered = _bands_within(band_centers, fmin, fmax)
        if third_octave_band_centers_hz is not None and not np.all(covered):
            warnings.warn('third-octave bands {} Hz extend past freq_range {} and are left out'.format(
                np.round(band_centers[~covered], 1).tolist(), (fmin, fmax)))
        band_centers = band_centers[covered]
        band_power_lists = [list() for _ in range(band_centers.size)]
        tonal_power_lists = [list() for _ in range(band_centers.size)]
        broadband_power_lists = [list() for _ in range(band_centers.size)]
    else:
        band_centers = np.array([], dtype=float)
        band_power_lists = []

    # Bands the receiver response is evaluated in: the output bands, or the
    # standard centers over the frequency range when there are none.
    if third_octave:
        response_centers = band_centers
    else:
        response_centers = BASE2_BAND_CENTERS[(BASE2_BAND_CENTERS * 2 ** (1 / 6) >= fmin)
                                              & (BASE2_BAND_CENTERS / 2 ** (1 / 6) <= fmax)]

    psd_power_lists = []
    f_sel_master = None

    # Precompute absorption reference range (meters)
    r_ref_m = None
    if apply_absorption_deprop:
        r_ref_m = float(unit_conversion.len_conv(r_ref, from_units=length_units, to_units='m'))

    pref_sq = P_REF ** 2

    def _depropagate(lin, amb, r_v, alpha, response=None, r_path=None, delta_db=None):
        """Gate against ambient, subtract it, then undo spreading and absorption.

        ``lin`` is (Nf, Npts) linear power (per bin or per band) at the
        microphone; ``amb`` the matching (Nf,) ambient or None; ``alpha`` the
        (Nf,) absorption coefficients in dB/m, used when absorption
        depropagation is on.  Spreading is undone over ``r_v`` and absorption
        over ``r_path`` (the same, without a ray model).  ``delta_db``, if
        given, is the (Nf, Npts) absorption correction in dB used in place of
        ``alpha (r_path - r_ref)`` (a band method or a layered atmosphere).
        """
        if r_path is None:
            r_path = r_v
        # Ambient gating and background subtraction have to happen before
        # spreading and absorption depropagation, or an ambient-limited band
        # gets amplified by both.
        if amb is not None:
            lin = gate_ambient(lin, amb, band_snr_gate_db)

        # The installation's response, as a power ratio to divide out.
        if response is not None:
            lin = lin * response

        # Spherical spreading depropagation to r_ref: multiply by (r/r_ref)^2
        lin = lin * ((r_v / float(r_ref)) ** 2)[None, :]

        # Optional absorption depropagation back to r_ref
        if apply_absorption_deprop:
            assert r_ref_m is not None
            if delta_db is not None:
                deltaL = delta_db
            else:
                r_m = unit_conversion.len_conv(r_path, from_units=length_units, to_units='m').astype(float)
                deltaL = np.asarray(alpha, dtype=float)[:, None] * (r_m[None, :] - r_ref_m)
            if max_absorption_correction_db is not None:
                # Drop bins the measurement cannot support before applying the
                # correction, not after: once multiplied they are indisting-
                # uishable from real high-frequency content.  Dropped is NaN, not
                # zero: nothing was measured there, rather than no energy.
                lin = np.where(deltaL <= float(max_absorption_correction_db), lin, np.nan)
            lin = lin * (10.0 ** (deltaL / 10.0))
        return lin

    def _absorption_delta(frequency, method, r_path, source_z, receiver_z):
        """(Nf, Npts) absorption correction, dB: each path's attenuation (in the
        layered ``absorption_profile`` if given, else in ``atmosphere``) less the
        attenuation over r_ref in ``atmosphere``, by ``method``.  ``r_path``,
        ``source_z`` and ``receiver_z`` are in ``length_units``."""
        reference = band_absorption_db(r_ref_m, frequency, atmosphere.temperature, atmosphere.pressure,
                                       atmosphere.relative_humidity, method=method)
        path = path_absorption_db(frequency, method, r_path, source_z, receiver_z, length_units=length_units,
                                  atmosphere=atmosphere, absorption_profile=absorption_profile)
        return path - reference[:, None]

    for im in range(nmics):
        # PSD spectrogram on observer time axis
        t_vec = _time_vec_for_mic(im)
        fs = float(np.round(1.0 / (t_vec[1] - t_vec[0])))
        f, t_rel, psd_db = spectrogram(pressure_list[im], fs, window_time=window_time, window_overlap=window_overlap)
        t_abs = t_rel + t_vec[0]

        fmask = np.logical_and(f >= fmin, f <= fmax)
        f_sel = f[fmask]
        if f_sel.size < 2:
            raise ValueError('Selected frequency range does not contain enough bins')

        # OASPL, A-weighting and third-octave integration all happen per
        # microphone on that microphone's own frequency grid, so arrays may
        # differ from mic to mic.  Only the narrowband export, which stacks
        # every mic onto one frequency axis, needs a common grid -- arrays
        # recorded at different sample rates (the 2017 Noise Abatement test
        # mixes 25000 and 25600 Hz) are otherwise perfectly usable.
        if narrowband:
            if not np.isclose(fs, fs0, rtol=1e-3, atol=0.0):
                raise ValueError('narrowband=True requires all microphones at the same sample rate')
            if f_sel_master is None:
                f_sel_master = f_sel
            elif f_sel_master.shape != f_sel.shape or not np.allclose(f_sel_master, f_sel, rtol=0.0, atol=0.0):
                raise ValueError('narrowband=True requires identical frequency grids for all microphones')

        df = float(f_sel[1] - f_sel[0])

        Aweight_db = dBAw(f_sel)
        Aweight_lin = 10.0 ** (Aweight_db / 10.0)

        alpha_db_per_m = None
        if apply_absorption_deprop:
            alpha_db_per_m = np.asarray(atmosphere.attenuation_coefficient(f_sel), dtype=float)

        # Observation times for selected emission points
        tobs_sub = t_obs[tidx, im]
        r_sub = r_geom[tidx, im]
        az_sub = az_deg[tidx, im]
        el_sub = el_deg[tidx, im]

        # Keep only points that can be interpolated in time and, when
        # requested, that lie above the elevation cutoff
        valid = np.logical_and(tobs_sub >= t_abs[0], tobs_sub <= t_abs[-1])
        valid &= range_cap_mask(el_sub, r_sub, min_elevation_deg, max_range, rim_range)
        if ray_valid is not None:
            valid = np.logical_and(valid, ray_valid[:, im])
        if not np.any(valid):
            continue

        tobs_v = tobs_sub[valid]
        r_v = r_sub[valid]
        spread_v = spread_geom[tidx, im][valid] if spread_geom is not None else r_v
        path_v = path_geom[tidx, im][valid] if path_geom is not None else r_v
        az_v = az_sub[valid]
        el_v = el_sub[valid]

        # Interpolate the PSD at tobs_v in linear power (relative to
        # pref^2/Hz).  Until 2026-09-24 this interpolated in dB, which between
        # frames takes a geometric mean of single-periodogram bins whose values
        # fluctuate strongly: broadband levels came out ~0.7 dB low in every
        # band, and in OASPL.
        psd_sel_db = psd_db[fmask, :]
        psd_sel_lin = 10.0 ** (psd_sel_db / 10.0)
        psd_v_lin = np.empty((f_sel.size, tobs_v.size), dtype=float)
        for fi in range(f_sel.size):
            psd_v_lin[fi, :] = np.interp(tobs_v, t_abs, psd_sel_lin[fi, :])

        # Filter-bank band levels on the spectrogram's own frames, so they
        # share its time base (and its ambient treatment below).
        frame_samples = frame_length(window_time, fs)
        band_frames = None
        if use_filter_bank:
            # NaN where the record has a gap (or the band reaches Nyquist): no data
            band_frames = pa_filters.third_octave_filter_bank(
                pressure_list[im], fs, band_centers, t_rel, frame_samples) / pref_sq

        # Per-bin (and per-band) ambient reference.
        amb_lin = None
        amb_band = None
        if ambient_pressure_list is not None:
            amb_p = ambient_pressure_list[im]
            amb_fs = fs
            if ambient_time_list is not None:
                amb_t = ambient_time_list[im]
                if amb_t.size >= 2:
                    amb_fs = float(np.round(1.0 / (amb_t[1] - amb_t[0])))
            if not np.isclose(amb_fs, fs, rtol=1e-6, atol=0.0):
                raise ValueError('ambient_pressure for mic {:d} is sampled at {:g} Hz but the run is at {:g} Hz; '
                                 'the ambient recording must come from the same channel and sample rate'
                                 .format(im, amb_fs, fs))
            # Same window settings as the run, so the ambient lands on the
            # identical frequency grid and can be applied bin by bin.
            f_amb, t_amb, psd_amb_db = spectrogram(amb_p, amb_fs, window_time=window_time,
                                                   window_overlap=window_overlap)
            if f_amb.shape != f.shape or psd_amb_db.shape[1] < 1:
                raise ValueError('ambient_pressure for mic {:d} did not yield a usable spectrogram on the run '
                                 'frequency grid (is the recording at least one window long?)'.format(im))
            amb_lin = np.nanmean(10.0 ** (psd_amb_db[fmask, :] / 10.0), axis=1)
            if use_filter_bank:
                amb_band = np.nanmean(pa_filters.third_octave_filter_bank(
                    amb_p, amb_fs, band_centers, t_amb, frame_samples), axis=1) / pref_sq
        else:
            amb_lin = ambient_from_frames(psd_sel_lin, t_abs, ambient_time_range, ambient_percentile,
                                          power_db=psd_sel_db)
            if use_filter_bank:
                amb_band = ambient_from_frames(band_frames, t_abs, ambient_time_range, ambient_percentile)

        response_bins = response_bands = None
        if receiver_response_db is not None:
            offset = (ray_offset[:, im][valid] if ray_offset is not None
                      else pos_geom[tidx][valid] - mic_geom[im])
            gain_db = np.asarray(receiver_response_db(im, response_centers, offset), dtype=float)
            if gain_db.shape != (response_centers.size, tobs_v.size) or not np.all(np.isfinite(gain_db)):
                raise ValueError('receiver_response_db must return finite values shaped (bands, points)')
            inverse = 10.0 ** (-gain_db / 10.0)
            if max_response_correction_db is not None:
                # As the absorption cap: a band the installation all but nulls
                # cannot be divided back up.  A flush board's modeled response
                # falls to -25 to -40 dB as the arrival nears grazing, and a
                # curved ray arriving within half a degree of it turned a 90 dB
                # sample into 125.
                inverse = np.where(-gain_db <= float(max_response_correction_db), inverse, np.nan)
            # Each bin takes the band the band sums below put it in: nominal
            # centers get their base-10 edges (see third_octave_band_edges)
            _, response_upper = third_octave_band_edges(response_centers)
            band_of_bin = np.clip(np.searchsorted(response_upper, f_sel, side='right'),
                                  0, response_centers.size - 1)
            response_bins = inverse[band_of_bin]
            response_bands = inverse

        source_z_v = pos_geom[tidx][valid][:, 2]
        delta_bins = None
        if absorption_profile is not None:
            delta_bins = _absorption_delta(f_sel, 'center', path_v, source_z_v, mic_geom[im][2])
        psd_v_lin = _depropagate(psd_v_lin, amb_lin, spread_v, alpha_db_per_m, response_bins, path_v,
                                 delta_bins)

        # OASPL power over selected frequency range: the bins a cap dropped are
        # left out, and a sample with no bin left is missing
        dropped = np.isnan(psd_v_lin)
        if dropped.any():
            no_bins = np.all(dropped, axis=0)
            power_oaspl = np.where(no_bins, np.nan, np.nansum(psd_v_lin * df, axis=0))
            power_spl_a = np.where(no_bins, np.nan, np.nansum((psd_v_lin * Aweight_lin[:, None]) * df, axis=0))
        else:
            power_oaspl = np.sum(psd_v_lin * df, axis=0)
            power_spl_a = np.sum((psd_v_lin * Aweight_lin[:, None]) * df, axis=0)

        fazi_list.append(az_v)
        felv_list.append(el_v)
        fmic_list.append(np.full(az_v.size, im))
        frange_list.append(r_v)
        fheight_list.append(pos_geom[tidx][valid][:, 2] - mic_geom[im][2])
        if interpolation is not None:
            # The arc one analysis window smears over: the source's motion across
            # the line of sight in window_time, and its own size, over the range.
            line = pos_geom[tidx][valid] - mic_geom[im]
            unit_line = line / np.linalg.norm(line, axis=1)[:, None]
            v = np.asarray(vel_geom, dtype=float)[tidx][valid]
            across = np.linalg.norm(v - np.sum(v * unit_line, axis=1)[:, None] * unit_line, axis=1)
            extent = float(interp_settings['source_extent'])
            fres_list.append(np.degrees(np.hypot(across * float(window_time), extent) / r_v))
        oaspl_power_list.append(power_oaspl)
        spl_a_power_list.append(power_spl_a)

        if narrowband:
            psd_power_lists.append(psd_v_lin)

        if use_filter_bank:
            # Band power at the emission points' observer times, interpolated
            # in linear power as the PSD is, then depropagated band by band.
            band_v = np.empty((band_centers.size, tobs_v.size), dtype=float)
            for ib in range(band_centers.size):
                band_v[ib, :] = np.interp(tobs_v, t_abs, band_frames[ib, :])
            alpha_band = (np.asarray(atmosphere.attenuation_coefficient(band_centers), dtype=float)
                          if apply_absorption_deprop else None)
            delta_bands = None
            if absorption_profile is not None or absorption_band_method != 'center':
                delta_bands = _absorption_delta(band_centers, absorption_band_method, path_v, source_z_v,
                                                mic_geom[im][2])
            band_v = _depropagate(band_v, amb_band, spread_v, alpha_band, response_bands, path_v, delta_bands)
            for ib in range(band_centers.size):
                band_power_lists[ib].append(band_v[ib, :])
        elif third_octave:
            # Integrate to third-octave bands in linear power, on edges that
            # tile even when the centers are nominal (see third_octave_band_edges)
            band_lower, band_upper = third_octave_band_edges(band_centers)
            if tone_aware or doppler_geom is not None:
                d_v = doppler_geom[tidx, im][valid] if doppler_geom is not None else np.ones(tobs_v.size)
                # A band is missing where any bin it spans was dropped (NaN), or
                # where its Doppler-scaled edges leave the selected bins.
                psd_kept = np.where(dropped, 0.0, psd_v_lin) if dropped.any() else psd_v_lin
                missing = _bands_missing(dropped, f_sel, band_lower, band_upper, d_v)
                if tone_aware:
                    powers, tonal, broadband = _tone_aware_bands(psd_kept, f_sel, band_lower, band_upper,
                                                                 doppler=d_v, parts=decompose_tonal)
                    if decompose_tonal:
                        for ib in range(band_centers.size):
                            tonal_power_lists[ib].append(np.where(missing[ib], np.nan, tonal[ib]))
                            broadband_power_lists[ib].append(np.where(missing[ib], np.nan, broadband[ib]))
                else:
                    # Running integral of the spectrum at the bins' edges, read at each
                    # sample's Doppler-scaled band edges.
                    running = np.vstack((np.zeros((1, tobs_v.size)), np.cumsum(psd_kept * df, axis=0)))
                    powers = np.maximum(_band_integrals(running, f_sel[0] - 0.5 * df, df, band_lower, band_upper, d_v), 0.0)
                powers = np.where(missing, np.nan, powers)
                for ib in range(band_centers.size):
                    band_power_lists[ib].append(powers[ib])
            else:
                for ib, (f_lower, f_upper) in enumerate(zip(band_lower, band_upper)):
                    band_mask = np.logical_and(f_sel >= f_lower, f_sel < f_upper)
                    if np.any(band_mask):
                        band_power = np.sum(psd_v_lin[band_mask, :] * df, axis=0)
                    else:
                        # Keep shape/point counts consistent even if a band is empty at the
                        # selected frequency resolution/range. This band will evaluate to -inf dB.
                        band_power = np.zeros(tobs_v.size, dtype=float)
                    band_power_lists[ib].append(band_power)

    if len(oaspl_power_list) == 0:
        raise ValueError('No valid hemisphere samples were generated (check time alignment and inputs)')

    fazi_pts = np.concatenate(fazi_list)
    felv_pts = np.concatenate(felv_list)
    P_oaspl_pts = np.concatenate(oaspl_power_list)
    P_spl_a_pts = np.concatenate(spl_a_power_list)

    # The geodesic distance shepIDW weights by is already periodic in azimuth,
    # so points near 0 and 360 deg reach across the seam as they are; the seam
    # column is closed below.  (Copies of the points at +-360 deg, which this
    # once added, only tripled every weight, which the normalization cancels.)

    # Cells with no sample within reach come out NaN (no data) and cells whose
    # samples carry no energy -inf, never a finite floor; see power_to_db.
    interpolation_info = None
    if interp_settings is not None:
        # One weight matrix for every quantity gridded: the radius is the node's,
        # not the band's.  Geodesic distances, so the samples once, not at +-360.
        weights, node_radius, node_gap, node_relaxed = adaptive_idw_weights(
            ELV_GRID.ravel(), AZI_GRID.ravel(), felv_pts, fazi_pts, np.concatenate(fmic_list),
            np.concatenate(fres_list), k=interp_settings['k'], min_mics=interp_settings['min_mics'],
            kappa=interp_settings['kappa'], resolution_factor=interp_settings['resolution_factor'],
            max_radius_deg=interp_settings['max_radius_deg'],
            relax_min_mics=interp_settings.get('relax_min_mics', True), return_relaxed=True,
            kernel=interp_settings.get('kernel', 'shepard'), aspect=interp_settings.get('aspect', 1.0),
            shepard_floor=interp_settings.get('shepard_floor', 0.0))
        node_gap = node_gap.reshape(ELV_GRID.shape)
        node_relaxed = node_relaxed.reshape(ELV_GRID.shape)

        def grid_power(power_pts):
            # Missing (NaN) samples carry no weight: a node's weights are renormalized
            # over the samples it has, and a node with none left is a gap.
            power_pts = np.asarray(power_pts, dtype=float)
            missing = np.isnan(power_pts)
            g = np.asarray(weights @ np.where(missing, 0.0, power_pts))
            if missing.any():
                kept = np.asarray(weights @ (~missing).astype(float))
                reweight = np.asarray(weights @ missing.astype(float)) > 0.0
                with np.errstate(invalid='ignore', divide='ignore'):
                    g = np.where(reweight, np.where(kept > 0.0, g / kept, np.nan), g)
            return np.where(node_gap, np.nan, g.reshape(ELV_GRID.shape))
        interpolation_info = dict(interp_settings, mode='adaptive',
                                  radius_deg=node_radius.reshape(ELV_GRID.shape),
                                  gaps=int(node_gap[:, :-1].sum()) if node_gap.shape[1] > 1 else int(node_gap.sum()),
                                  relaxed=int(node_relaxed[:, :-1].sum()) if node_relaxed.shape[1] > 1
                                  else int(node_relaxed.sum()))
    else:
        # Every field is sampled at the same points, so the weights are shared
        idw_weights = shepIDW_weights(ELV_GRID, AZI_GRID, felv_pts, fazi_pts, rmax=float(rmax))

        def grid_power(power_pts):
            return shepIDW_apply(idw_weights, power_pts)

    grid_statistics = None
    if return_statistics:
        import scipy.sparse as sparse
        from panam import sphere_statistics     # here: only return_statistics needs it (no cycle)
        if interp_settings is not None:
            # A gap node is NaN whatever its weights: it has no samples.
            node_weights = sparse.diags((~node_gap.ravel()).astype(float)) @ weights
        else:
            node_weights = sphere_statistics.idw_weight_matrix(idw_weights[1], fazi_pts.size)
        mic_pts = np.concatenate(fmic_list)

        def grid_statistics(power_pts):
            # Statistics of one field (Np,) or several (Nf, Np), shaped as their grids,
            # with the seam column closed as the levels' is.
            power_pts = np.asarray(power_pts, dtype=float)
            stats = sphere_statistics.node_statistics(node_weights, power_pts, mic_pts)
            shape = power_pts.shape[:-1] + ELV_GRID.shape
            stats = {key: value.reshape(shape) for key, value in stats.items()}
            if ELV_GRID.shape[1] > 1:
                for value in stats.values():
                    value[..., -1] = value[..., 0]
            return stats

    P_oaspl_grid = grid_power(P_oaspl_pts)
    oaspl_db = power_to_db(P_oaspl_grid)
    if oaspl_db.shape[1] > 1:
        oaspl_db[:, -1] = oaspl_db[:, 0]

    P_spl_a_grid = grid_power(P_spl_a_pts)
    spl_a_db = power_to_db(P_spl_a_grid)
    if spl_a_db.shape[1] > 1:
        spl_a_db[:, -1] = spl_a_db[:, 0]

    out = {
        'azi_grid_deg': azi_grid_deg,
        'elv_grid_deg': elv_grid_deg,
        'oaspl_db': oaspl_db,
        'spl_a_db': spl_a_db,
        'metadata': {
            'r_ref': float(r_ref),
            'length_units': str(length_units),
            'freq_range_hz': (fmin, fmax),
            'window_time': float(window_time),
            'window_overlap': float(window_overlap),
            'point_stride': int(point_stride),
            'rmax_deg': float(rmax),
            'rim_range': None if rim_range is None else (float(rim_range[0]), float(rim_range[1])),
            'apply_absorption_deprop': bool(apply_absorption_deprop),
            'third_octave_method': str(third_octave_method),
            'absorption_band_method': str(absorption_band_method),
            'absorption_profile': absorption_profile is not None,
            # Depropagation removes absorption over r - r_ref only, so the
            # sphere still carries absorption over r_ref in this atmosphere.
            # Exporters for formats with a different convention (NORAH2) need
            # it to take that back out.
            'atmosphere': {
                'temperature_k': float(atmosphere.temperature),
                'pressure_kpa': float(atmosphere.pressure),
                'relative_humidity': float(atmosphere.relative_humidity),
            },
            'flip_y_for_geometry': bool(flip_y_for_geometry),
            # What azimuth 180 points along: the velocity, or the track_nose given.
            'azimuth_from': 'velocity' if track_nose is None else 'nose',
            'return_scattered': bool(return_scattered),
            'narrowband': bool(narrowband),
            'narrowband_stride': int(narrowband_stride),
        }
    }
    if interpolation_info is not None:
        out['interpolation'] = interpolation_info
    if grid_statistics is not None:
        out['statistics'] = {'oaspl_db': grid_statistics(P_oaspl_pts),
                             'spl_a_db': grid_statistics(P_spl_a_pts)}

    scattered: Optional[dict[str, Any]] = None
    if return_scattered:
        scattered = cast(dict[str, Any], {
            'azi_deg': fazi_pts,
            'elv_deg': felv_pts,
            'oaspl_power': P_oaspl_pts,
            'spl_a_power': P_spl_a_pts,
            'mic': np.concatenate(fmic_list),
            # Straight-line slant range and the source's height above the microphone, in
            # length_units, at each sample's emission time.
            'range': np.concatenate(frange_list),
            'source_height': np.concatenate(fheight_list),
        })
        if fres_list:
            scattered['resolution_deg'] = np.concatenate(fres_list)
        out['scattered'] = scattered

    if third_octave:
        # Concatenate scattered band powers and interpolate per band; the tonal
        # and broadband parts, when asked for, are gridded with the same weights
        fields = {'bands_db': band_power_lists}
        if decompose_tonal:
            fields.update(tonal_db=tonal_power_lists, broadband_db=broadband_power_lists)
        out['third_octave'] = {'band_centers_hz': band_centers}
        if return_scattered:
            assert scattered is not None
            scattered['third_octave'] = {'band_centers_hz': band_centers}
        for key, power_lists in fields.items():
            # Each band's samples, concatenated once; None for a band with none.
            band_rows = [np.concatenate(samples) if samples else None for samples in power_lists]
            band_grids_db = np.full((band_centers.size, ELV_GRID.shape[0], ELV_GRID.shape[1]), -np.inf, dtype=float)
            for ib, P_band_pts in enumerate(band_rows):
                if P_band_pts is None:
                    continue
                if P_band_pts.size != fazi_pts.size:
                    raise ValueError('Internal error: third-octave sample count does not match scattered angle count')
                P_band_grid = grid_power(P_band_pts)
                band_grids_db[ib, :, :] = power_to_db(P_band_grid)
                band_grids_db[ib, :, -1] = band_grids_db[ib, :, 0]
            out['third_octave'][key] = band_grids_db
            band_power_pts = (np.vstack(band_rows) if band_centers.size > 0
                              else np.empty((0, fazi_pts.size), dtype=float))
            if grid_statistics is not None and band_centers.size > 0:
                out['statistics'].setdefault('third_octave', {})[key] = grid_statistics(band_power_pts)
            if return_scattered:
                scattered['third_octave'][key] = (power_to_db(band_power_pts) if band_centers.size > 0
                                                  else band_power_pts)

    if narrowband:
        assert f_sel_master is not None
        # Concatenate scattered PSD power samples into (Nf, Np)
        if len(psd_power_lists) == 0:
            raise ValueError('No narrowband PSD samples were accumulated')

        psd_power_pts = np.concatenate(psd_power_lists, axis=1)
        if psd_power_pts.shape[1] != fazi_pts.size:
            raise ValueError('Internal error: PSD sample count does not match scattered angle count')

        f_nb = f_sel_master[::narrowband_stride]
        psd_grid_db = np.full((f_nb.size, ELV_GRID.shape[0], ELV_GRID.shape[1]), -np.inf, dtype=float)
        for i_f, fi in enumerate(range(0, f_sel_master.size, narrowband_stride)):
            P_f_grid = grid_power(psd_power_pts[fi, :])
            psd_grid_db[i_f, :, :] = power_to_db(P_f_grid)
            psd_grid_db[i_f, :, -1] = psd_grid_db[i_f, :, 0]

        out['narrowband'] = {
            'frequency_hz': f_nb,
            'psd_db': psd_grid_db,
        }
        if grid_statistics is not None:
            out['statistics']['narrowband'] = {'psd_db': grid_statistics(psd_power_pts[::narrowband_stride, :])}

        if return_scattered:
            assert scattered is not None
            scattered['narrowband'] = {
                'frequency_hz': f_nb,
                'psd_db': power_to_db(psd_power_pts[::narrowband_stride, :]),
            }

    return out
