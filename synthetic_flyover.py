"""Synthetic flyover recordings: a straight, level pass of a source of known directivity.

``synthesize_pass`` makes the microphone signals and track that
``flight_acoustics.depropagate_hemisphere`` takes, for a point source moving
along +x at constant speed and height over microphones in a right-handed,
z-up frame.  ``source_level_db`` gives the overall level the source radiates
at the reference radius in any direction, so a depropagated hemisphere can be
checked against the truth.  ``plot_level_difference`` maps recovered minus
true on the Lambert equal-area projection.

Lengths are in ft and speeds in ft/s throughout (any consistent length unit
works, provided ``speed_of_sound`` is in the same unit).  Pressures are in Pa.
Angles follow the UMAPR convention of hemisphere dicts: azimuth 180 deg ahead,
90 deg to starboard (here -y, since the source flies along +x with z up),
elevation positive below the horizon.
"""

import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.pyplot import colorbar, subplots

import flight_acoustics as fa
from panam_acoustics.plotting import acoustic_plot_style

#: Tones of the default source: (frequency Hz, amplitude), amplitude in units of ``source_strength``.
DEFAULT_TONES = ((110.0, 1.0), (440.0, 0.5))


def example_directivity_db(azimuth_deg, elevation_deg):
    """A smooth directivity for examples: louder ahead near the horizon, quieter overhead.

    Args:
        azimuth_deg: UMAPR azimuth, deg (180 ahead).
        elevation_deg: UMAPR elevation, deg (positive below the horizon).

    Returns:
        Level re the omnidirectional source, dB, broadcast to the inputs' shape.
        It spans -3 to +4 dB: 4 cos(azimuth - 180) cos(elevation) - 3 sin^2(elevation).
    """
    azimuth = np.radians(np.asarray(azimuth_deg, dtype=float))
    elevation = np.radians(np.asarray(elevation_deg, dtype=float))
    return 4.0 * np.cos(azimuth - np.pi) * np.cos(elevation) - 3.0 * np.sin(elevation) ** 2


def synthesize_pass(mic_locations, *, speed=150.0, height=300.0, duration=6.0, sampling_rate=4000.0,
                    speed_of_sound=1125.0, tones=DEFAULT_TONES, broadband=0.2, source_strength=100.0,
                    noise=0.002, directivity_db=None, track_time=None, seed=0):
    """Microphone signals and track for a straight, level pass of a point source.

    The source flies along +x at ``speed`` and ``height``, over x = 0 at
    ``duration / 2``.  It radiates ``tones`` plus white broadband noise of
    standard deviation ``broadband``, scaled by ``source_strength`` / r and by
    ``directivity_db`` taken at the emission angles.  Each microphone's signal
    is found at its own sample times by solving t = t_e + r(t_e) / c for the
    emission time t_e (30 fixed-point iterations), so it carries the
    propagation delay, the Doppler shift and spherical spreading.  No ground
    reflection, absorption or convective amplification is modeled.  Each
    channel then gets white noise of standard deviation ``noise``, and a
    separate ambient record of the same noise is returned.

    The broadband source noise is drawn per received sample, so it is white
    up to ``sampling_rate / 2`` at the microphone.  Random numbers are drawn
    in a fixed order from ``numpy.random.default_rng(seed)``: per microphone
    the source noise and then the microphone noise, then the ambient records.

    Args:
        mic_locations: (Nmics, 3) microphone positions, ft (z up).
        speed: source speed along +x, ft/s.
        height: source height, ft.
        duration: length of the recordings, s; they start at t = 0.
        sampling_rate: sample rate, Hz.
        speed_of_sound: ft/s.
        tones: sequence of (frequency Hz, amplitude) pairs; the amplitude is
            in units of ``source_strength`` (dimensionless).
        broadband: standard deviation of the broadband source noise, in units
            of ``source_strength`` (dimensionless); 0 for none.
        source_strength: pressure times distance, Pa ft: the pressure at
            1 ft of a unit-amplitude tone.
        noise: standard deviation of the microphone and ambient noise, Pa.
        directivity_db: None (omnidirectional), or a function of (azimuth
            deg, elevation deg), UMAPR, returning the level re the
            omnidirectional source in dB.  Evaluated at each sample's
            emission angles.
        track_time: track sample times, s.  Default 120 times from 0.5 s to
            ``duration - 0.5`` s.
        seed: seed of the random number generator.

    Returns:
        (scenario, ambient): ``scenario`` is a dict of the first arguments of
        ``depropagate_hemisphere``: ``mic_locations`` (Nmics, 3) ft,
        ``pressure`` (Nmics, Nsamples) Pa, ``time`` (Nsamples,) s,
        ``track_time`` (Nt,) s, ``track_position`` (Nt, 3) ft and
        ``track_velocity`` (Nt, 3) ft/s.  ``ambient`` is (Nmics, Nsamples)
        microphone noise alone, Pa, for ``ambient_pressure``.
    """
    rng = np.random.default_rng(seed)
    mics = np.asarray(mic_locations, dtype=float)
    if mics.ndim != 2 or mics.shape[1] != 3:
        raise ValueError('mic_locations must be shaped (Nmics, 3)')
    c = float(speed_of_sound)
    t = np.arange(0.0, duration, 1.0 / sampling_rate)
    overhead = duration / 2.0

    def position(time):
        time = np.asarray(time, dtype=float)
        return np.column_stack([speed * (time - overhead), np.zeros(time.size), np.full(time.size, height)])

    pressure = []
    for mic in mics:
        t_e = t.copy()
        for _ in range(30):
            t_e = t - np.linalg.norm(position(t_e) - mic, axis=1) / c
        source_position = position(t_e)
        r = np.linalg.norm(source_position - mic, axis=1)
        source = np.zeros(t.size)
        for frequency, amplitude in tones:
            source = source + amplitude * np.sin(2 * np.pi * frequency * t_e)
        if broadband:
            source = source + broadband * rng.standard_normal(t.size)
        scale = source_strength / r
        if directivity_db is not None:
            azimuth, elevation, _, _, _ = fa.hemigen(t_e, source_position, np.tile([speed, 0.0, 0.0], (t.size, 1)),
                                                     mic[None, :], c)
            scale = scale * 10.0 ** (np.asarray(directivity_db(azimuth[:, 0], elevation[:, 0])) / 20.0)
        pressure.append(scale * source + noise * rng.standard_normal(t.size))
    if track_time is None:
        track_time = np.linspace(0.5, duration - 0.5, 120)
    track_time = np.asarray(track_time, dtype=float)
    ambient = np.array([noise * rng.standard_normal(t.size) for _ in mics])
    return dict(mic_locations=mics, pressure=np.array(pressure), time=t, track_time=track_time,
                track_position=position(track_time),
                track_velocity=np.tile([speed, 0.0, 0.0], (track_time.size, 1))), ambient


def source_level_db(azimuth_deg, elevation_deg, r_ref, *, tones=DEFAULT_TONES, broadband=0.2,
                    source_strength=100.0, directivity_db=None):
    """The overall level ``synthesize_pass``'s source radiates at ``r_ref``.

    The mean square pressure is (S g / r_ref)^2 (sum of a^2 / 2 over the tones
    + broadband^2), with S = ``source_strength`` and g the directivity's
    amplitude factor.  It counts all the broadband power, which
    ``synthesize_pass`` spreads up to half the sample rate, so compare it
    with a depropagated overall level integrated to that frequency.

    Args:
        azimuth_deg, elevation_deg: UMAPR angles, deg (broadcast together).
        r_ref: reference radius, ft (the length unit of ``source_strength``).
        tones, broadband, source_strength, directivity_db: as for ``synthesize_pass``.

    Returns:
        Overall sound pressure level re 20 uPa, dB, shaped like the broadcast angles.
    """
    azimuth, elevation = np.broadcast_arrays(np.asarray(azimuth_deg, dtype=float),
                                             np.asarray(elevation_deg, dtype=float))
    mean_square = sum(amplitude ** 2 / 2.0 for _, amplitude in tones) + broadband ** 2
    level = 10.0 * np.log10((source_strength / r_ref) ** 2 * mean_square / fa.P_REF ** 2)
    if directivity_db is not None:
        level = level + np.asarray(directivity_db(azimuth, elevation), dtype=float)
    return np.broadcast_to(level, azimuth.shape).copy()


@acoustic_plot_style
def plot_level_difference(azi_grid_deg, elv_grid_deg, difference_db, limit_db=2.0, step_db=0.5,
                          label='Recovered minus true, dB'):
    """Map a level difference over a hemisphere grid on the Lambert equal-area projection.

    Args:
        azi_grid_deg: (Nazi,) UMAPR azimuths, deg.
        elv_grid_deg: (Nelv,) UMAPR elevations, deg.
        difference_db: (Nelv, Nazi) difference, dB.  NaN and infinite cells are left blank.
        limit_db: the color scale spans -limit_db to +limit_db, dB; larger differences
            take the end colors.
        step_db: contour interval, dB.
        label: colorbar label.

    Returns:
        The figure.
    """
    difference = np.array(difference_db, dtype=float)
    difference[~np.isfinite(difference)] = np.nan
    azimuth, elevation = np.meshgrid(np.radians(azi_grid_deg), np.radians(elv_grid_deg))
    x, y = fa.lambert_ea(elevation, fa.lambert_lon(azimuth))
    levels = np.arange(-limit_db, limit_db + step_db / 2.0, step_db)
    fig, ax = subplots(facecolor='white')
    ax.patch.set_visible(False)
    contours = ax.contourf(x, y, np.clip(difference, -limit_db, limit_db), levels=levels, cmap='RdBu_r',
                           norm=BoundaryNorm(levels, 256), extend='neither')
    fa._configure_lambert_axes(ax)
    fa._draw_lambert_grid(ax)
    colorbar(contours, pad=0.1, ticks=fa.colorbar_ticks(levels)).set_label(label)
    return fig
