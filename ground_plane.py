"""Ground-plane microphones: co-located pole comparisons and mixed-impedance corrections.

A ground-plane microphone is a small hard plate lying on softer ground, so its
flat pressure-doubling correction (+6.02 dB, the ``ground_board_scale=0.5`` in
:mod:`noise_abatement_2017`) holds only where the plate dominates the ground
reflection: high frequency and steep incidence.  At low frequency and grazing
incidence the reflection's Fresnel zone spreads onto the soft ground around the
plate.  See ``docs/ground_plane_corrections.md`` for the review this follows.

The 2017 NASA array has elevated (pole) microphones at the same surveyed points
as three of its ground-plane microphones (:data:`COLOCATED_PAIRS`).  With the
pole's own ground reflection modeled, a pair measures the ground plane's
transfer function directly::

    L_pole  = L_free + G_pole       (pole_ground_effect)
    L_board = L_free + T_board      (the unknown; board_* are models of it)
    T_board = L_board - L_pole + G_pole

Every ``board_*`` function returns T_board: the level at the ground-plane
microphone relative to free field, in dB, so the flat correction is
``board_rigid_plane() == 6.02``.  Lengths are in feet and speeds in ft/s, like the
rest of the 2017 pipeline; flow resistance is in kPa s/m^2, as in
:func:`flight_acoustics.ega`.
"""
import numpy as np
import scipy.signal as sig

import flight_acoustics as fa

#: (pole, ground-plane) microphone numbers at the same surveyed point, on every
#: aircraft of the 2017 test: x = -3760 ft, y = +500, 0 and -500 ft.
COLOCATED_PAIRS = ((50, 34), (51, 36), (52, 38))

#: Ground flow resistance, kPa s/m^2, one value for the whole site.  A joint fit
#: of the three poles' interference patterns (B407, 7 level runs, frames above
#: 20 deg) is flat from 60 to 225 with its minimum near 100.  It is held fixed
#: rather than fitted per pair: per-pair fits wandered from about 60 on the
#: centerline to 180 on the sidelines, which is more likely something else
#: (the ground plane itself, for one) being absorbed into the impedance.
FLOW_RESISTANCE = 100.0

#: Pole microphone heights, ft, fitted at :data:`FLOW_RESISTANCE`.  The
#: metadata does not record them (its heights are ground elevations); the
#: nominal is the 1.2 m (3.94 ft) certification height.
POLE_HEIGHT_FT = {50: 3.96, 51: 4.07, 52: 4.12}

#: GRAS 67AX: a flush-mounted 1/2" microphone (47AX) in a 400 mm diameter plate
#: (GR1425).  From the GRAS drawing (67AX data sheet, cutaway A-A): the plate is
#: 8.00 mm thick, its top surface tapering to 2.50 mm at the rim, and the
#: microphone sits flush in it 150 mm from the center -- 3/4 of the radius, the
#: ARP 4055 position.  The diaphragm is in the plate's top surface, 0 above the
#: reflecting plate.  In the 2017 test the plates lay on top of the ground, so
#: that top surface stands 8 mm above the soft ground and the tapered edge is
#: exposed.  Every ground microphone in that test was mounted this way, those
#: the dataset labels `invgb7` (inverted, 7 mm gap) included, per the test
#: team; the SAE/ICAO inverted microphone 7 mm above a plate is another layout.
PLATE_RADIUS_FT = 0.2 / 0.3048
PLATE_MIC_HEIGHT_FT = 0.0
#: Microphone position on the plate, per SAE ARP 4055 (and ICAO Annex 16 /
#: ETM, 0.15 m on a 0.4 m plate): 3/4 of the radius from the center, on a line
#: normal to the intended flight track, for both the inverted and the flush
#: types.  Here the offset is along +y of the test axes (x along the track); the
#: side (+y or -y) is not recorded for the 2017 array.
PLATE_MIC_OFFSET_FT = 0.75 * PLATE_RADIUS_FT
#: The ARP 4055 inverted microphone's diaphragm-to-plate gap.  Not the 2017
#: `invgb7` channels, which were flush despite the label (above).
INVERTED_MIC_HEIGHT_FT = 0.007 / 0.3048
PLATE_THICKNESS_FT = 0.008 / 0.3048
PLATE_EDGE_THICKNESS_FT = 0.0025 / 0.3048

#: Flow resistance standing in for the rigid plate.  It has to be far beyond
#: "rigid" ground values: any finite impedance departs from Q = 1 at grazing
#: incidence, and at 1e6 (water, in the AAM table) a flush microphone reads
#: 3.2 dB instead of 6.0 at 10 kHz with the source 5 ft up and 3000 ft away.
RIGID_FLOW_RESISTANCE = 1.0e12

#: Hothersall & Harriott's transition criterion: the zone where a path via the
#: ground exceeds the specular path by less than this fraction of a wavelength.
FRESNEL_ZONE_FRACTION = 1.0 / 3.0

PRESSURE_DOUBLING_DB = 20.0 * np.log10(2.0)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def emission_geometry(track, location, reception_times, sound_speed):
    """Source position relative to ``location`` at the emission time of each reception time.

    Returns a dict of ``elevation`` (deg above the horizon at the microphone),
    ``slant_range``, ``source_height`` (above the microphone's ground),
    ``ground_distance`` and the horizontal offsets ``source_dx``, ``source_dy``,
    each NaN where the reception time falls outside what
    the track can have emitted.
    """
    times = np.asarray(track['time'], dtype=float)
    offsets = np.column_stack((track['x'], track['y'], track['z'])) - np.asarray(location, float)
    arrival = times + np.linalg.norm(offsets, axis=1) / sound_speed
    reception_times = np.asarray(reception_times, dtype=float)
    inside = (reception_times >= arrival[0]) & (reception_times <= arrival[-1])
    emitted = np.interp(reception_times, arrival, times)
    source = np.column_stack([np.interp(emitted, times, offsets[:, k]) for k in range(3)])
    ground_distance = np.hypot(source[:, 0], source[:, 1])
    out = {
        'elevation': np.degrees(np.arctan2(source[:, 2], ground_distance)),
        'slant_range': np.linalg.norm(source, axis=1),
        'source_height': source[:, 2],
        'ground_distance': ground_distance,
        # Horizontal source offset from the microphone, in the test's x (along
        # track) and y axes: the incidence azimuth on an off-center microphone.
        'source_dx': source[:, 0],
        'source_dy': source[:, 1],
    }
    return {key: np.where(inside, value, np.nan) for key, value in out.items()}


def emission_times(position, reception_times, receiver, sound_speed, tol=1e-9, max_iter=50):
    """Emission times of the sound reaching ``receiver`` at ``reception_times``, for a
    source whose position is a function of time.

    Solves t_e = t - |x(t_e) - receiver| / c by fixed-point iteration from t_e = t.  Each
    pass shrinks the error by about the source's Mach number toward the receiver, so a
    single pass leaves the source about M^2 R cos(phi) short along its path (several
    meters at a few hundred meters' range and M ~ 0.15); the iteration runs to ``tol``
    (s).  Unlike :func:`emission_geometry`, which inverts a sampled track's arrival
    times, ``position`` can be any callable -- an interpolated, smoothed or shifted track
    -- and the receiver may move: ``receiver`` is (3,) or (n, 3), one row per reception
    time.

    ``position(t)`` takes an (n,) array of times and returns (n, 3) positions; lengths
    and ``sound_speed`` in matching units.  Returns (t_e, x(t_e)).  Raises
    RuntimeError if the iteration has not converged after ``max_iter`` passes (a
    supersonic approach has no unique solution).
    """
    t = np.asarray(reception_times, dtype=float)
    rec = np.asarray(receiver, dtype=float)
    t_e = t.copy()
    for _ in range(max_iter):
        x = np.asarray(position(t_e), dtype=float)
        new = t - np.linalg.norm(x - rec, axis=-1) / sound_speed
        step = np.nanmax(np.abs(new - t_e), initial=0.0)
        t_e = new
        if step < tol:
            return t_e, np.asarray(position(t_e), dtype=float)
    raise RuntimeError(f'emission_times did not converge in {max_iter} passes (last step {step:.3g} s)')



# --------------------------------------------------------------------------
# Measured pairs
# --------------------------------------------------------------------------

def band_levels_from_psd(frequency, psd, bands):
    """One-third octave band levels (dB re 20 uPa) from a one-sided PSD, brick-wall bands.

    ``psd`` is (frequency, frame) in Pa^2/Hz.  The same rectangular integration
    as :func:`flight_acoustics.third_octave_band_levels`.
    """
    df = frequency[1] - frequency[0]
    levels = np.empty((len(bands), psd.shape[1]))
    for i, center in enumerate(bands):
        inside = (frequency >= center / 2 ** (1 / 6)) & (frequency < center * 2 ** (1 / 6))
        levels[i] = 10.0 * np.log10(np.maximum(psd[inside].sum(axis=0) * df, 1e-30) / 4e-10)
    return levels


def pair_band_histories(test, run, pole, board, track, sound_speed, frame_s=0.5,
                        bands=fa.PNL_BAND_FREQUENCIES):
    """Band level histories of one co-located pair, with the emission geometry.

    Both channels are read as recorded -- no pressure-doubling factor on the
    ground plane -- and framed on a common time base.  A ground-plane channel
    at a different sample rate from its pole is resampled to the pole's.
    """
    import noise_abatement_2017 as na

    pole_p, pole_t, location, rate = na._read_signal(test.acoustic_files[run][pole])
    board_p, board_t, board_location, board_rate = na._read_signal(test.acoustic_files[run][board])
    if not np.allclose(location[:2], board_location[:2], atol=1.0):
        raise ValueError('Run {}: mics {} and {} are not co-located ({} vs {})'
                         .format(run, pole, board, location, board_location))
    if board_rate != rate:
        board_p = sig.resample_poly(board_p, int(round(rate)), int(round(board_rate)))
        board_t = board_t[0] + np.arange(board_p.size) / rate
    start, stop = max(pole_t[0], board_t[0]), min(pole_t[-1], board_t[-1])
    first_pole = int(np.ceil((start - pole_t[0]) * rate))
    first_board = int(np.ceil((start - board_t[0]) * rate))
    count = int(np.floor((stop - start) * rate))
    segment = int(round(frame_s * rate))
    frequency, frame_times, pole_psd = sig.spectrogram(
        pole_p[first_pole:first_pole + count], rate, window='hann', nperseg=segment, noverlap=0)
    _, _, board_psd = sig.spectrogram(
        board_p[first_board:first_board + count], rate, window='hann', nperseg=segment, noverlap=0)
    reception = start + frame_times
    out = emission_geometry(track, location, reception, sound_speed)
    out.update(time=reception, bands=np.asarray(bands, dtype=float),
               pole=band_levels_from_psd(frequency, pole_psd, bands),
               board=band_levels_from_psd(frequency, board_psd, bands))
    return out


def measured_board_transfer(test, run, pole, board, frame_s=0.5, snr_db=10.0,
                            flow_resistance=FLOW_RESISTANCE, pole_height=None):
    """The ground plane's transfer function measured against its co-located pole, (band, frame) dB.

    T_board = L_board - L_pole + G_pole, with G_pole the pole's modeled
    ground reflection at the site's flow resistance and fitted height.  Band
    frames are kept only where both channels are ``snr_db`` above that band's
    noise floor, taken as the 10th percentile of the run's frames (the quiet
    lead-in and tail of a pass); the rest are NaN.  Returns the
    :func:`pair_band_histories` dict with ``T_board`` and ``G_pole`` added.
    """
    import noise_abatement_2017 as na

    track = na.load_track(test.track_path(run))
    air = na.run_atmosphere(test, run, fallback=fa.Atmosphere(
        temperature=288.15, pressure=101.325, relative_humidity=50.0))
    sound_speed = float(fa.unit_conversion.len_conv(air.soundspeed, from_units='m', to_units='ft'))
    out = pair_band_histories(test, run, pole, board, track, sound_speed, frame_s=frame_s)
    height = POLE_HEIGHT_FT[pole] if pole_height is None else pole_height
    valid = np.isfinite(out['elevation']) & (out['source_height'] > 0.0)
    g_pole = np.full(out['pole'].shape, np.nan)
    if valid.any():
        g_pole[:, valid] = pole_ground_effect(out['bands'], out['source_height'][valid],
                                              out['ground_distance'][valid], height, sound_speed,
                                              flow_resistance)
    floor_pole = np.nanpercentile(out['pole'], 10, axis=1, keepdims=True)
    floor_board = np.nanpercentile(out['board'], 10, axis=1, keepdims=True)
    loud = (out['pole'] > floor_pole + snr_db) & (out['board'] > floor_board + snr_db)
    out['T_board'] = np.where(loud & valid[None, :], out['board'] - out['pole'] + g_pole, np.nan)
    out['G_pole'] = g_pole
    out['sound_speed'] = sound_speed
    return out


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def _broadcast(bands, source_height, ground_distance):
    bands = np.asarray(bands, dtype=float)
    hs = np.asarray(source_height, dtype=float)
    d2 = np.asarray(ground_distance, dtype=float)
    return np.broadcast_arrays(bands[:, None], hs[None, :], d2[None, :])


# --------------------------------------------------------------------------
# Ground impedance models and the pole
# --------------------------------------------------------------------------

#: Air properties for the impedance models (SI).
AIR_DENSITY = 1.2
GAMMA = 1.4


def surface_admittance(frequency, model='delany_bazley', **params):
    """Normalized surface admittance beta = 1/Z of a locally reacting ground (e^{-i omega t}).

    Models (Attenborough & Taherzadeh 2026, section 1.3, and references there):

    ``delany_bazley`` (sigma: flow resistivity, kPa s/m^2) -- one parameter,
        Z = 1 + 9.08 X^-0.75 + i 11.9 X^-0.73, X = f/sigma; what :func:`ega` uses.
    ``miki`` (sigma) -- Miki's (1990) physically better-behaved refit,
        Z = 1 + 5.50 X^-0.632 + i 8.43 X^-0.632.
    ``variable_porosity`` (sigma_e, kPa s/m^2; alpha_e, 1/m) -- their eq. (10),
        Z = (1 + i) sqrt(R_se/(pi gamma rho0 f)) + i c0 alpha_e/(8 pi gamma f),
        R_se = 1000 sigma_e.  Two parameters: an effective flow resistivity and
        the rate of porosity change with depth.
    ``delany_bazley_layer`` (sigma; depth, m) -- a hard-backed layer, eq. (5),
        Z = Z_c coth(-i k d), with Delany-Bazley's characteristic impedance and
        wavenumber (eqs. 4a, 4b).  Their pasture/meadow class is a 0.05 m layer.
    """
    f = np.asarray(frequency, dtype=float)
    if model == 'delany_bazley':
        x = f / params['sigma']
        z = 1.0 + 9.08 * x ** -0.75 + 11.9j * x ** -0.73
    elif model == 'miki':
        x = f / params['sigma']
        z = 1.0 + 5.50 * x ** -0.632 + 8.43j * x ** -0.632
    elif model == 'variable_porosity':
        speed = params.get('sound_speed_mps', 343.0)
        z = ((1.0 + 1j) * np.sqrt(1000.0 * params['sigma_e'] / (np.pi * GAMMA * AIR_DENSITY * f))
             + 1j * speed * params['alpha_e'] / (8.0 * np.pi * GAMMA * f))
    elif model == 'delany_bazley_layer':
        speed = params.get('sound_speed_mps', 343.0)
        x = f / params['sigma']
        zc = 1.0 + 9.08 * x ** -0.754 + 11.9j * x ** -0.732
        k = 2.0 * np.pi * f / speed * (1.0 + 10.8 * x ** -0.70 + 10.3j * x ** -0.595)
        z = zc / np.tanh(-1j * k * params['depth'])
    else:
        raise ValueError('unknown impedance model {!r}'.format(model))
    return 1.0 / z


#: GRAS free-field corrections for the 40AE capsule (the 46AE set), with the
#: protection grid, dB re the response at 250 Hz, by angle of incidence (0, 30,
#: 60, 90, 120, 150, 180 deg), from GRAS_Free-field_and_Random_Incidence_
#: Corrections.xlsx, sheet "Half-inch (Opt1)" (grasacoustics.com, Free-field
#: Correction Curves).  Below 500 Hz the corrections are negligible (0).
GRAS_40AE_FREE_FIELD_CORRECTIONS = np.array([
    # frequency, 0, 30, 60, 90, 120, 150, 180 deg
    (500.0, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04),
    (560.0, 0.03, 0.03, 0.01, 0.01, 0.01, 0.01, 0.01),
    (630.0, 0.04, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01),
    (710.0, 0.04, 0.05, 0.02, -0.00, -0.00, -0.00, -0.00),
    (800.0, 0.07, 0.06, 0.02, -0.02, -0.02, -0.02, -0.02),
    (900.0, 0.07, 0.06, 0.01, -0.05, -0.05, -0.05, -0.05),
    (1000.0, 0.09, 0.07, -0.02, -0.10, -0.10, -0.10, -0.10),
    (1120.0, 0.10, 0.05, -0.04, -0.15, -0.15, -0.15, -0.15),
    (1250.0, 0.13, 0.07, 0.00, -0.14, -0.14, -0.14, -0.14),
    (1400.0, 0.17, 0.07, 0.03, -0.15, -0.15, -0.15, -0.15),
    (1600.0, 0.21, 0.13, 0.06, -0.16, -0.17, -0.16, -0.16),
    (1800.0, 0.26, 0.17, 0.10, -0.17, -0.19, -0.19, -0.16),
    (2000.0, 0.32, 0.26, 0.15, -0.14, -0.17, -0.17, -0.12),
    (2240.0, 0.40, 0.32, 0.20, -0.13, -0.16, -0.16, -0.10),
    (2500.0, 0.49, 0.42, 0.26, -0.08, -0.13, -0.13, -0.04),
    (2800.0, 0.60, 0.50, 0.26, -0.10, -0.15, -0.15, -0.03),
    (3150.0, 0.72, 0.59, 0.37, -0.09, -0.19, -0.15, 0.05),
    (3550.0, 0.87, 0.67, 0.43, -0.07, -0.23, -0.15, 0.06),
    (4000.0, 1.06, 0.82, 0.48, -0.04, -0.30, -0.17, 0.08),
    (4500.0, 1.31, 1.03, 0.58, -0.02, -0.37, -0.26, 0.10),
    (5000.0, 1.61, 1.32, 0.77, 0.05, -0.39, -0.33, 0.16),
    (5600.0, 1.95, 1.61, 0.95, 0.07, -0.41, -0.56, 0.18),
    (6300.0, 2.32, 1.92, 1.18, 0.12, -0.42, -0.48, 0.24),
    (7100.0, 2.80, 2.34, 1.47, 0.26, -0.48, -0.41, 0.32),
    (8000.0, 3.38, 2.82, 1.80, 0.49, -0.53, -0.39, 0.39),
    (9000.0, 4.09, 3.40, 2.26, 0.84, -0.36, -0.17, 0.58),
    (10000.0, 4.94, 4.19, 2.90, 1.32, -0.05, 0.31, 1.05),
    (11200.0, 5.91, 5.09, 3.46, 1.41, -0.19, 0.32, 1.10),
    (12500.0, 6.49, 5.50, 3.60, 0.97, -0.92, -0.54, 0.30),
    (14000.0, 7.19, 6.18, 3.85, 0.93, -1.59, -1.30, -0.35),
    (16000.0, 7.90, 6.78, 3.98, 0.61, -2.32, -2.07, -0.79),
    (18000.0, 8.82, 7.59, 4.33, 0.28, -2.81, -2.61, -1.99),
    (20000.0, 9.26, 7.64, 4.09, -0.62, -4.03, -3.78, -3.29),
])

#: The 2017 pole microphones: GRAS 46AE on 4 ft tripods, placed "to emulate
#: certification placement at -45, 0 and 45 deg under the aircraft and
#: perpendicular to the flight path" (Watts et al., NASA TM 2019, section 5.1).
#: Certification (14 CFR 36 / ICAO Annex 16) orients the sensing element in the
#: plane containing the flight path and the station: grazing (90 deg)
#: incidence for sound from the flight line.
POLE_MICROPHONE = '46AE'


def free_field_microphone_response(frequency, incidence):
    """A 40AE/46AE free-field microphone's response re on-axis (0 deg) incidence, dB.

    The free-field corrections are what the microphone's actuator (pressure)
    response lacks of its free-field response at each angle; a free-field
    microphone is built flat at 0 deg, so its response at angle a relative to
    on axis is corr(a) - corr(0).  At 90 deg -- grazing incidence, the
    certification orientation -- that is -0.2 dB at 1 kHz, -1.1 at 4 kHz and
    -3.6 at 10 kHz.  Interpolated in log frequency and linearly in angle; 0
    below 500 Hz.  ``incidence`` in degrees, 0 to 180.
    """
    table = GRAS_40AE_FREE_FIELD_CORRECTIONS
    angles = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0])
    f = np.asarray(frequency, dtype=float)
    a = np.clip(np.asarray(incidence, dtype=float), 0.0, 180.0)
    f, a = np.broadcast_arrays(f, a)
    logf = np.log(np.clip(f, table[0, 0], table[-1, 0]))
    relative = table[:, 1:] - table[:, 1:2]                  # re on-axis
    by_angle = np.array([np.interp(logf.ravel(), np.log(table[:, 0]), relative[:, j])
                         for j in range(angles.size)])        # (angle, points)
    flat_a = a.ravel()
    lower = np.clip(np.searchsorted(angles, flat_a, side='right') - 1, 0, angles.size - 2)
    t = (flat_a - angles[lower]) / (angles[lower + 1] - angles[lower])
    points = np.arange(flat_a.size)
    out = ((1.0 - t) * by_angle[lower, points] + t * by_angle[lower + 1, points]).reshape(f.shape)
    return np.where(f < table[0, 0], 0.0, out)


def pole_incidence(source_dx, source_dy, source_height, pole_height, axis):
    """Angles of incidence (deg) of the direct and ground-reflected paths on a microphone axis.

    ``axis`` is the microphone's unit axis (x along track, y, z up) -- the
    direction its diaphragm faces.  The reflected path arrives from the image
    source below the ground.  Lengths in any one unit.
    """
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    def angle(vx, vy, vz):
        norm = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        return np.degrees(np.arccos(np.clip((axis[0] * vx + axis[1] * vy + axis[2] * vz) / norm,
                                            -1.0, 1.0)))
    dx, dy, hs = (np.asarray(v, dtype=float) for v in (source_dx, source_dy, source_height))
    return angle(dx, dy, hs - pole_height), angle(dx, dy, -hs - pole_height)


def certification_axis(pole_y, flight_height, pole_height=0.0, sign=1.0):
    """Microphone axis normal to the plane containing the (x-directed) flight line and the pole.

    The flight line is along x at y = 0 and height ``flight_height``; the pole
    is at y = ``pole_y``.  ``sign`` picks which way the diaphragm faces (the
    orientation fixes the plane, not the side).
    """
    to_line = np.array([0.0, -pole_y, flight_height - pole_height])
    normal = np.cross([1.0, 0.0, 0.0], to_line)
    return sign * normal / np.linalg.norm(normal)


#: HARMONOISE turbulence coherence constant (Attenborough & Taherzadeh eq. 18a).
HARMONOISE_B = 0.364


def turbulence_coherence(frequency, source_height, receiver_height, distance, gamma_t,
                         sound_speed):
    """HARMONOISE coherence of the direct and reflected paths, exp(-3/8 B k^2 gamma_p^(5/3) R gamma_T).

    gamma_p = h_s h_r / (h_s + h_r); ``gamma_t`` = C_T^2/T0^2 + 22 C_v^2/(3 c0^2),
    about 1e-6 for moderate turbulence (up to ~1e-5).  Lengths in meters.  Near
    1 for the flush ground plane (h_r ~ 0), not for a 1.2 m pole.
    """
    k = 2.0 * np.pi * np.asarray(frequency, float) / sound_speed
    gamma_p = source_height * receiver_height / (source_height + receiver_height)
    return np.exp(-0.375 * HARMONOISE_B * k ** 2 * gamma_p ** (5.0 / 3.0) * distance * gamma_t)


def pole_level(bands, source_height, ground_distance, pole_height, sound_speed,
               model='delany_bazley', gamma_t=0.0, response_direct=None,
               response_reflected=None, roughness=0.0, **ground):
    """Band-averaged level at a pole microphone re free field, (band, frame) dB, any ground model.

    The same third-octave form as :func:`flight_acoustics.ega` (Chessell's
    equations 20-21: direct plus reflected path with the band-averaging
    sin(x)/x), with the spherical-wave coefficient from ``model`` and the
    HARMONOISE coherence factor for turbulence.  Lengths in feet and speed in
    ft/s as elsewhere here; converted for the coherence factor.

    ``response_direct`` and ``response_reflected`` (dB, (band, frame)) weight
    the two paths by the microphone's response at each path's angle of
    incidence (:func:`free_field_microphone_response` with
    :func:`pole_incidence`); the result is then what the microphone reads re
    the free-field pressure.

    ``roughness`` is the ground's rms height (ft).  Surface roughness scatters
    part of the reflection out of the specular direction; the coherent part's
    amplitude falls by the Kirchhoff (Ament) factor exp(-2 (k sigma_h cos
    theta)^2), and the scattered part is taken as lost to the microphone.  It
    matters at a few kHz for centimeter roughness near normal incidence and
    vanishes toward grazing (cos theta -> 0).
    """
    f, hs, d2 = _broadcast(bands, source_height, ground_distance)
    hr = pole_height
    direct = np.hypot(d2, hs - hr)
    image = np.hypot(d2, hs + hr)
    cos_theta = (hs + hr) / image
    delay = (image - direct) / sound_speed
    beta = surface_admittance(f, model, sound_speed_mps=sound_speed * 0.3048, **ground)
    q = fa.spherical_reflection_coefficient(cos_theta, image, f, sound_speed, None,
                                            admittance=beta)
    ratio = direct / image
    coherence = 1.0 if gamma_t == 0.0 else turbulence_coherence(
        f, hs * 0.3048, hr * 0.3048, direct * 0.3048, gamma_t, sound_speed * 0.3048)
    fd = f * delay
    with np.errstate(divide='ignore', invalid='ignore'):
        sinc = np.where(fd > 0.0, np.sin(0.727477 * fd) / (0.727477 * fd), 1.0)
    a_d = 1.0 if response_direct is None else 10.0 ** (0.05 * np.asarray(response_direct))
    a_r = 1.0 if response_reflected is None else 10.0 ** (0.05 * np.asarray(response_reflected))
    if roughness > 0.0:
        k = 2.0 * np.pi * f / sound_speed
        a_r = a_r * np.exp(-2.0 * (k * roughness * cos_theta) ** 2)
    energy = (a_d ** 2 + (a_r * np.abs(q) * ratio) ** 2
              + 2.0 * a_d * a_r * np.abs(q) * ratio * np.cos(6.325159 * fd + np.angle(q)) * sinc
              * coherence)
    return 10.0 * np.log10(energy)


def pole_ground_effect(bands, source_height, ground_distance, pole_height, sound_speed,
                       flow_resistance=FLOW_RESISTANCE, turbulence=0.0):
    """Band-averaged excess attenuation at a pole microphone (Chien-Soroka), (band, frame) dB."""
    f, hs, d2 = _broadcast(bands, source_height, ground_distance)
    level, _ = fa.ega(hs, pole_height, d2, f, sound_speed, flow_resistance, pt=False,
                      cturb=turbulence)
    return level


def board_rigid_plane(bands, source_height, ground_distance, sound_speed=None):
    """The current correction: an infinite rigid plane, exact pressure doubling (+6.02 dB) at
    every frequency and angle."""
    f, _, _ = _broadcast(bands, source_height, ground_distance)
    return np.full(f.shape, PRESSURE_DOUBLING_DB)


def board_uniform(bands, source_height, ground_distance, sound_speed,
                  flow_resistance=RIGID_FLOW_RESISTANCE, mic_height=PLATE_MIC_HEIGHT_FT):
    """A microphone ``mic_height`` above a uniform ground: the plate (rigid) or the ground alone.

    With the defaults this is an infinite plate with the diaphragm flush in
    it: exact pressure doubling.  With the site's flow resistance and
    ``mic_height=PLATE_THICKNESS_FT`` it is the other bound, no plate at all:
    the diaphragm 8 mm above the soft ground.
    """
    f, hs, d2 = _broadcast(bands, source_height, ground_distance)
    # A flush microphone has no path delay; ega's np.where still evaluates
    # its sin(x)/x branch there, and discards it.
    with np.errstate(divide='ignore', invalid='ignore'):
        level, _ = fa.ega(hs, mic_height, d2, f, sound_speed, flow_resistance, pt=False)
    return level


def _path_via(x, source_height, ground_distance, mic_height):
    """Source to microphone path length via the ground point ``x`` from below the source."""
    return np.hypot(x, source_height) + np.hypot(ground_distance - x, mic_height)


def fresnel_strip_weight(bands, source_height, ground_distance, sound_speed,
                         radius=PLATE_RADIUS_FT, mic_height=PLATE_MIC_HEIGHT_FT,
                         zone_fraction=FRESNEL_ZONE_FRACTION, iterations=60):
    """Soft-ground weight r of Hothersall & Harriott's two-impedance model, (band, frame).

    In the vertical plane through source and microphone the plate is the chord
    [d - a, d + a] of the disc, d = ``ground_distance`` from the point below the
    source and a = ``radius``, with soft ground either side.  The transition
    zone is where the path via the ground exceeds the specular path by less
    than ``zone_fraction`` of a wavelength, between D1 and D2 (D2 > D1).  r is
    the share of the zone on soft ground: 1 - |[D1, D2] & [d - a, d + a]| /
    (D2 - D1).  With a single boundary D and plate beyond it -- Hothersall &
    Harriott's half-plane -- that is their r = (D - D1)/(D2 - D1).  The second
    edge matters because near grazing, and at low frequency, the zone runs well
    past the microphone, onto the ground behind the plate.
    """
    f, hs, d2 = _broadcast(bands, source_height, ground_distance)
    specular_x = d2 * hs / (hs + mic_height)
    specular = np.hypot(d2, hs + mic_height)
    target = specular + zone_fraction * sound_speed / f

    def path(x):
        return _path_via(x, hs, d2, mic_height)

    def solve(lo, hi):
        # Path length is monotonic either side of the specular point.
        return _bisect(path, target, lo, hi, path(hi) > path(lo), iterations)

    far = np.maximum(d2, 1.0) * 1e4 + target
    d1 = solve(-far, specular_x)           # source side of the specular point
    d2_zone = solve(specular_x, d2 + far)  # microphone side, and beyond it
    on_plate = np.clip(np.minimum(d2_zone, d2 + radius) - np.maximum(d1, d2 - radius), 0.0, None)
    return np.clip(1.0 - on_plate / (d2_zone - d1), 0.0, 1.0)


def _blend(plate_weight, plate, soft, blend):
    """Combine the plate and soft-ground results, in energy or in dB."""
    if blend == 'energy':
        return 10.0 * np.log10(plate_weight * 10.0 ** (0.1 * plate)
                               + (1.0 - plate_weight) * 10.0 ** (0.1 * soft))
    if blend == 'db':
        return plate_weight * plate + (1.0 - plate_weight) * soft
    raise ValueError("blend must be 'energy' or 'db', not {!r}".format(blend))


def _plate_and_soft(bands, source_height, ground_distance, sound_speed, flow_resistance, zone):
    """The two uniform-ground results the Fresnel models blend.

    On the plate the diaphragm is flush (``mic_height``, 0 by default); over
    the soft ground it is the plate's thickness above it.
    """
    plate = board_uniform(bands, source_height, ground_distance, sound_speed,
                          mic_height=zone.get('mic_height', PLATE_MIC_HEIGHT_FT))
    soft = board_uniform(bands, source_height, ground_distance, sound_speed, flow_resistance,
                         mic_height=PLATE_THICKNESS_FT)
    return plate, soft


def board_fresnel_strip(bands, source_height, ground_distance, sound_speed,
                        flow_resistance=FLOW_RESISTANCE, blend='energy', **zone):
    """Hothersall & Harriott's Fresnel-zone model on the plate's 2-D strip: |p/pf| = |A_soft|^r |A_plate|^(1-r).

    ``blend='db'`` weights the two results in dB, r G_soft + (1 - r) G_plate --
    Hothersall & Harriott's geometric interpolation, and the HARMONOISE form.
    ``blend='energy'`` (the default) weights the pressures squared instead,
    10 log10(r 10^(G_soft/10) + (1 - r) 10^(G_plate/10)), which Boulanger et al.
    (1997) and Attenborough & Taherzadeh (2026) found closer to BEM; weighting in
    dB can be several dB off where the two results differ widely.  Each is the
    band-averaged uniform
    ground result: the flush diaphragm on the plate, and the diaphragm the
    plate's thickness above the soft ground.  (Mikkelsen & Nickerson's eq. 4
    prints a sum; the geometric interpolation is Hothersall & Harriott's.)
    """
    r = fresnel_strip_weight(bands, source_height, ground_distance, sound_speed, **zone)
    plate, soft = _plate_and_soft(bands, source_height, ground_distance, sound_speed,
                                  flow_resistance, zone)
    return _blend(1.0 - r, plate, soft, blend)


def sub_band_factors(sub_bands):
    """Frequencies, as fractions of the band center, splitting a third-octave band
    into ``sub_bands`` equal log-width parts at their centers.  Plate tables are
    built and read on these, so both sides take them from here.
    """
    return 2.0 ** ((np.arange(sub_bands) + 0.5) / sub_bands / 3.0 - 1.0 / 6.0)


def _bisect(path, limit, lo, hi, increasing, iterations=60):
    """Where a path length monotonic on [lo, hi] reaches ``limit``, elementwise."""
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        move_hi = (path(mid) > limit) == increasing
        hi = np.where(move_hi, mid, hi)
        lo = np.where(move_hi, lo, mid)
    return 0.5 * (lo + hi)


def fresnel_ellipse(source_height, ground_distance, mic_height, excess):
    """The Fresnel ellipse on the ground: center x and semi-axes (along, across the line).

    It is the ground section of the ellipsoid with foci at the source and the
    microphone's image, and the specular path plus ``excess`` as its major
    axis -- the ground points whose path via the ground is within ``excess``
    of the specular one.  Coordinates are from the point below the microphone,
    negative toward the source.  The section of an ellipsoid by a plane is an
    ellipse, and this one is symmetric about the source-microphone line, so
    its two crossings of that line give the center and one semi-axis, and the
    crossing of the perpendicular through the center the other.  (The closed
    form for a plane section cancels catastrophically when the source is far
    above so small a zone.)
    """
    hs = np.asarray(source_height, dtype=float)
    d2 = np.asarray(ground_distance, dtype=float)
    limit = np.hypot(d2, hs + mic_height) + excess
    specular = -d2 * mic_height / (hs + mic_height)
    span = limit + d2 + 1.0
    along = lambda x: np.hypot(x + d2, hs) + np.hypot(x, mic_height)
    x_lo = _bisect(along, limit, specular - span, specular, False)
    x_hi = _bisect(along, limit, specular, specular + span, True)
    center = 0.5 * (x_lo + x_hi)
    across = lambda y: np.hypot(np.hypot(center + d2, y), hs) + np.sqrt(center ** 2 + y ** 2 + mic_height ** 2)
    semi_y = _bisect(across, limit, np.zeros_like(span), span, True)
    return center, 0.5 * (x_hi - x_lo), semi_y


def fresnel_disc_weight(bands, source_height, ground_distance, sound_speed,
                        radius=PLATE_RADIUS_FT, mic_height=PLATE_MIC_HEIGHT_FT,
                        zone_fraction=FRESNEL_ZONE_FRACTION, samples=(32, 64)):
    """Plate weight: the fraction of the Fresnel ellipse covered by the circular plate, (band, frame).

    The Nord2000 / Harmonoise generalization of Hothersall & Harriott: each
    surface counts in proportion to the share of the Fresnel ellipse it covers.
    The ellipse is the ground section of the zone where the path via the
    ground exceeds the specular path by less than ``zone_fraction`` of a
    wavelength; the plate is a disc of ``radius`` centerd under the
    microphone.

    The overlap is integrated on an area-uniform polar grid (``samples`` =
    radial, angular) over whichever of the two shapes is smaller, testing
    membership of the other.  Sampling the disc when the ellipse is far smaller
    (high frequency, steep incidence: a few hundredths of a foot across, against
    the plate's 0.66 ft radius) resolved almost none of it.
    """
    f, hs, d2 = _broadcast(bands, source_height, ground_distance)
    excess = zone_fraction * sound_speed / f
    limit = np.hypot(d2, hs + mic_height) + excess
    center, semi_x, semi_y = fresnel_ellipse(hs, d2, mic_height, excess)
    total = np.pi * semi_x * semi_y
    disc_area = np.pi * radius ** 2

    n_r, n_a = samples
    rho = np.sqrt((np.arange(n_r) + 0.5) / n_r)               # area-uniform radii
    phi = (np.arange(n_a) + 0.5) / n_a * 2.0 * np.pi
    unit_x = (rho[:, None] * np.cos(phi)[None, :]).ravel()
    unit_y = (rho[:, None] * np.sin(phi)[None, :]).ravel()

    weight = np.empty(f.shape)
    flat = weight.ravel()
    fields = [a.ravel() for a in (hs, d2, limit, total, center, semi_x, semi_y)]
    for i, (h, d, lim, area, cx, ax, ay) in enumerate(zip(*fields)):
        if area <= disc_area:
            # Sample the ellipse; count what lies on the plate.
            x = cx + ax * unit_x
            y = ay * unit_y
            flat[i] = np.mean(x ** 2 + y ** 2 <= radius ** 2)
        else:
            # Sample the plate; count what lies in the ellipse.
            x = radius * unit_x
            y = radius * unit_y
            inside = np.hypot(np.hypot(x + d, y), h) + np.sqrt(x ** 2 + y ** 2 + mic_height ** 2) <= lim
            flat[i] = np.mean(inside) * disc_area / area
    return np.clip(weight, 0.0, 1.0)


def board_fresnel_disc(bands, source_height, ground_distance, sound_speed,
                       flow_resistance=FLOW_RESISTANCE, blend='energy', **zone):
    """Fresnel-ellipse weighting over a circular plate, w on the plate and 1 - w on the soft ground.

    ``blend`` as in :func:`board_fresnel_strip`.
    """
    w = fresnel_disc_weight(bands, source_height, ground_distance, sound_speed, **zone)
    plate, soft = _plate_and_soft(bands, source_height, ground_distance, sound_speed,
                                  flow_resistance, zone)
    return _blend(w, plate, soft, blend)


def _fresnel_tail(x):
    """F2(x) = integral from x to infinity of exp(i w^2) dw (Lam & Monazzam eq. 2)."""
    from scipy.special import fresnel
    s_int, c_int = fresnel(np.asarray(x, dtype=float) * np.sqrt(2.0 / np.pi))
    return np.sqrt(np.pi / 2.0) * ((0.5 - c_int) + 1j * (0.5 - s_int))


def nmid_pressure_ratio(frequency, source_height, ground_distance, sound_speed,
                        flow_resistance=FLOW_RESISTANCE, radius=PLATE_RADIUS_FT,
                        mic_height=PLATE_MIC_HEIGHT_FT):
    """p/p_free at the ground-plane microphone from Lam & Monazzam's nMID model (eq. 12).

    The corrected (reciprocal) multi-discontinuity form of De Jong et al.
    (1983), in the vertical plane through source and microphone: soft ground,
    the plate chord [d - a, d + a] around the microphone at d, soft ground again.
    Two discontinuities, soft-to-plate (mu = -1) and plate-to-soft (mu = +1),
    each adding (Q_{j+1} - Q_j) e^{-i pi/4}/sqrt(pi) (r1/s_j)
    [mu_j F2(sqrt(k(s_j - r1))) + gamma_j F2(sqrt(k(s_j - r2))) e^{ik(r2 - r1)}],
    gamma_j = +1 when the specular point is on the source side of edge j.  Q's
    are :func:`flight_acoustics.spherical_reflection_coefficient` at the
    specular geometry; the plate is :data:`RIGID_FLOW_RESISTANCE`.

    Known limitations (see docs/ground_plane_corrections.md): De Jong is
    heuristic and stated invalid for kr < 10 (0.2 m to each edge: below about
    2.7 kHz); nMID loses accuracy toward grazing; and for a low receiver it does
    not reduce to the soft-ground result as the plate vanishes -- for a flush
    microphone it tends to 1 + 2 Q_soft - Q_plate instead of 1 + Q_soft.
    """
    f = np.asarray(frequency, dtype=float)
    hs = np.asarray(source_height, dtype=float)
    d = np.asarray(ground_distance, dtype=float)
    hr = mic_height
    k = 2.0 * np.pi * f / sound_speed
    r1 = np.hypot(d, hs - hr)
    r2 = np.hypot(d, hs + hr)
    cos_theta = (hs + hr) / r2
    q_soft = fa.spherical_reflection_coefficient(cos_theta, r2, f, sound_speed, flow_resistance)
    q_plate = fa.spherical_reflection_coefficient(cos_theta, r2, f, sound_speed,
                                                  RIGID_FLOW_RESISTANCE)
    specular = d * hs / (hs + hr)                   # from below the source
    on_plate = np.abs(specular - d) <= radius
    q_g = np.where(on_plate, q_plate, q_soft)
    phase = np.exp(1j * k * (r2 - r1))
    ratio = 1.0 + (r1 / r2) * q_g * phase
    for edge, dq, mu in ((d - radius, q_plate - q_soft, -1.0), (d + radius, q_soft - q_plate, 1.0)):
        s_j = np.hypot(edge, hs) + np.hypot(d - edge, hr)
        gamma = np.where(specular < edge, 1.0, -1.0)
        ratio = ratio + dq * np.exp(-0.25j * np.pi) / np.sqrt(np.pi) * (r1 / s_j) * (
            mu * _fresnel_tail(np.sqrt(np.maximum(k * (s_j - r1), 0.0)))
            + gamma * _fresnel_tail(np.sqrt(np.maximum(k * (s_j - r2), 0.0))) * phase)
    return ratio


def board_nmid(bands, source_height, ground_distance, sound_speed,
               flow_resistance=FLOW_RESISTANCE, sub_bands=9, **plate):
    """nMID (corrected De Jong) level re free field, averaged over each one-third octave band, dB.

    |p/p_free|^2 is averaged over ``sub_bands`` log-spaced frequencies across
    each band (De Jong et al. averaged three), which smooths the pure-tone
    interference the way a band measurement does.
    """
    f, hs, d2 = _broadcast(bands, source_height, ground_distance)
    offsets = sub_band_factors(sub_bands)
    energy = np.zeros(f.shape)
    for factor in offsets:
        energy += np.abs(nmid_pressure_ratio(f * factor, hs, d2, sound_speed, flow_resistance,
                                             **plate)) ** 2
    return 10.0 * np.log10(energy / sub_bands)


def _ground_admittance(frequency, sound_speed, flow_resistance, ground):
    """beta from a ground-model dict, or Delany-Bazley at ``flow_resistance`` when None."""
    if ground is None:
        return surface_admittance(frequency, 'delany_bazley', sigma=flow_resistance)
    params = {k: v for k, v in ground.items() if k != 'model'}
    return surface_admittance(frequency, ground['model'], sound_speed_mps=sound_speed * 0.3048,
                              **params)



# --------------------------------------------------------------------------
# Pole interference nulls (a phase-free check of the geometry)
# --------------------------------------------------------------------------
#
# A pole (elevated) microphone hears the direct and the ground-reflected path of the
# SAME emission, so the interference nulls in its spectrum -- or in its level difference
# from a co-located ground-plane microphone, where the source spectrum cancels -- depend
# only on the path difference dR and the reflection phase, not on the source's phase.
# Fitting the null pattern gives dR, and with the known pole height the arrival elevation
# or source height.  Lengths are in feet and speeds in ft/s as elsewhere here; the purely
# geometric functions work in any consistent unit.

def path_difference(source_height, ground_distance, receiver_height):
    """Image minus direct path length for a receiver ``receiver_height`` above the plane."""
    hs = np.asarray(source_height, dtype=float)
    d2 = np.asarray(ground_distance, dtype=float)
    return np.hypot(d2, hs + receiver_height) - np.hypot(d2, hs - receiver_height)


def height_from_path_difference(dR, ground_distance, receiver_height, iterations=40):
    """Source height giving path difference ``dR`` at ``ground_distance`` (inverse of
    :func:`path_difference`), vectorized.

    Newton's method from the small-angle guess h = dR d2 / (2 hr): dR(h) rises
    monotonically and concavely from 0 to 2 hr, so the iterates climb to the root.
    NaN where dR is outside (0, 2 hr).
    """
    hr = float(receiver_height)
    dR, d2 = np.broadcast_arrays(np.asarray(dR, float), np.asarray(ground_distance, float))
    ok = np.isfinite(dR) & np.isfinite(d2) & (dR > 0.0) & (dR < 2 * hr * 0.9999)
    h = np.where(ok, np.maximum(dR * d2 / (2 * hr), 1e-9), np.nan)
    for _ in range(iterations):
        up, dn = np.hypot(d2, h + hr), np.hypot(d2, h - hr)
        step = (up - dn - dR) / ((h + hr) / up - (h - hr) / dn)
        h = np.maximum(h - step, 1e-9)
        if np.nanmax(np.abs(step), initial=0.0) < 1e-12 * max(1.0, hr):
            break
    return np.where(ok, h, np.nan)


def null_frequencies(dR, sound_speed, phase=0.0, n=2):
    """The first ``n`` interference null frequencies, 2 pi f dR / c + phase = (2k - 1) pi.

    ``phase`` (rad) is arg Q of the ground reflection (0 for a rigid ground, small and
    positive for a stiff one).  Returns (..., n) for an array ``dR``.
    """
    k = np.arange(1, n + 1)
    return ((2 * k - 1) * np.pi - phase) * sound_speed / (2 * np.pi * np.asarray(dR, float)[..., None])


def two_path_db(frequency, offset_db, amplitude, amplitude_rolloff, dR, phase, sound_speed):
    """Two-path level re an arbitrary reference, dB: C + 10 log10 |1 + A e^{i(k dR + phase)}|^2.

    A = ``amplitude`` exp(-f / ``amplitude_rolloff``): the reflected path's relative
    amplitude, falling with frequency as coherence and reflection loss grow.  The pole
    minus ground-plane level difference has this form with C absorbing the plate's
    pressure doubling and any calibration offset.
    """
    f = np.asarray(frequency, dtype=float)
    a = amplitude * np.exp(-f / amplitude_rolloff)
    arg = 2 * np.pi * f * dR / sound_speed + phase
    return offset_db + 10 * np.log10(np.maximum(1 + a ** 2 + 2 * a * np.cos(arg), 1e-12))


def fit_two_path(frequency, level_db, dR_guess, sound_speed, receiver_height, fix_phase=None,
                 fmin=None, fmax=None, phase_bound=0.6):
    """Fit :func:`two_path_db` to a measured pole spectrum or pole-minus-board difference.

    The window runs from ``fmin`` (default 0.2 c / dR_guess) to ``fmax`` (default
    2.2 c / dR_guess, just below the guess's third null), so the first two nulls lie
    inside.  A grid over dR (0.4-2.2 times the guess, zero phase) finds the basin; a
    robust (soft-L1) least-squares fit then frees C, A, the roll-off, dR and the phase
    (bounded by +-``phase_bound`` rad, or held at ``fix_phase``).  Use narrowband data
    (a few Hz) averaged over frames at nearly the same geometry: third-octave bands smear
    the nulls.

    Returns dict(dR, phase, offset_db, amplitude, amplitude_rolloff, dR_se, phase_se,
    f1, f2, rms, n, fmin, fmax, at_bound).  ``dR_se`` and ``phase_se`` are the
    least-squares standard errors, which ignore correlation between frequency bins:
    use them for ranking, and the scatter over independent groups (runs) for
    intervals.
    """
    from scipy.optimize import least_squares
    f = np.asarray(frequency, dtype=float)
    y = np.asarray(level_db, dtype=float)
    hr = float(receiver_height)
    fmin = 0.2 * sound_speed / dR_guess if fmin is None else fmin
    fmax = min(2.2 * sound_speed / dR_guess, f.max()) if fmax is None else fmax
    sel = (f >= fmin) & (f <= fmax) & np.isfinite(y)
    f, y = f[sel], y[sel]
    grid = np.linspace(0.4, 2.2, 361) * dR_guess
    grid = grid[grid < 2 * hr]
    best = None
    for a0 in (0.6, 0.8, 0.92):
        shapes = np.array([two_path_db(f, 0.0, a0, 1e5, r, 0.0, sound_speed) for r in grid])
        offset = np.median(y[None, :] - shapes, axis=1)
        cost = np.mean(np.abs(y[None, :] - shapes - offset[:, None]), axis=1)
        k = int(np.argmin(cost))
        if best is None or cost[k] < best[0]:
            best = (cost[k], grid[k], a0, offset[k])
    _, r0, a00, c0 = best
    if fix_phase is None:
        p0 = [c0, a00, 3000.0, r0, 0.0]
        lo, hi = [-40, 0.05, 100.0, 0.02 * r0, -phase_bound], [40, 0.999, 1e6, 2 * hr, phase_bound]
        fun = lambda p: two_path_db(f, *p, sound_speed) - y
    else:
        p0 = [c0, a00, 3000.0, r0]
        lo, hi = [-40, 0.05, 100.0, 0.02 * r0], [40, 0.999, 1e6, 2 * hr]
        fun = lambda p: two_path_db(f, *p, fix_phase, sound_speed) - y
    res = least_squares(fun, p0, bounds=(lo, hi), loss='soft_l1', f_scale=2.0)
    p = res.x
    dof = max(1, f.size - p.size)
    s2 = float(np.sum(res.fun ** 2) / dof)
    try:
        se = np.sqrt(np.clip(np.diag(np.linalg.pinv(res.jac.T @ res.jac) * s2), 0.0, None))
    except np.linalg.LinAlgError:
        se = np.full(p.size, np.nan)
    phase = float(p[4]) if fix_phase is None else float(fix_phase)
    f12 = null_frequencies(p[3], sound_speed, phase, 2)
    return dict(dR=float(p[3]), phase=phase, offset_db=float(p[0]), amplitude=float(p[1]),
                amplitude_rolloff=float(p[2]), dR_se=float(se[3]),
                phase_se=float(se[4]) if fix_phase is None else 0.0, f1=float(f12[0]), f2=float(f12[1]),
                rms=float(np.sqrt(np.mean(res.fun ** 2))), n=int(f.size), fmin=float(fmin), fmax=float(fmax),
                at_bound=bool(fix_phase is None and abs(phase) > 0.98 * phase_bound))


def reflection_phase(frequency, source_height, ground_distance, receiver_height, sound_speed,
                     flow_resistance=FLOW_RESISTANCE, ground=None):
    """arg Q (rad) of the spherical-wave reflection coefficient at a pole microphone.

    The phase shifts the nulls (:func:`null_frequencies`); over a stiff ground it is
    small, so fitting it freely or holding it at this value are both reasonable.
    ``ground`` is a ground-model dict (:func:`surface_admittance`); without it,
    Delany-Bazley at ``flow_resistance``.
    """
    f, hs, d2 = np.broadcast_arrays(np.asarray(frequency, float), np.asarray(source_height, float),
                                    np.asarray(ground_distance, float))
    image = np.hypot(d2, hs + receiver_height)
    cos_theta = (hs + receiver_height) / image
    beta = _ground_admittance(f, sound_speed, flow_resistance, ground)
    q = fa.spherical_reflection_coefficient(cos_theta, image, f, sound_speed, flow_resistance, admittance=beta)
    return np.angle(q)


# --------------------------------------------------------------------------
# Exact Green's function of a locally reacting plane (complex image)
# --------------------------------------------------------------------------

def _gauss_segments(edges, n):
    """Gauss-Legendre nodes and weights over consecutive segments, broadcast over rows.

    ``edges`` is (m, s + 1); returns nodes and weights (m, s * n).
    """
    x, w = np.polynomial.legendre.leggauss(n)
    a, b = edges[:, :-1, None], edges[:, 1:, None]
    nodes = 0.5 * (b - a) * x[None, None, :] + 0.5 * (a + b)
    weights = 0.5 * (b - a) * w[None, None, :] * np.ones_like(nodes)
    return nodes.reshape(edges.shape[0], -1), weights.reshape(edges.shape[0], -1)


def image_integrals(k, beta, rho, z_sum, n=10):
    """I = int_0^inf e^{-k beta q} g(R_q) dq and J = int_0^inf e^{-k beta q} g'(R_q)/R_q dq.

    R_q = sqrt(rho^2 + (Z + i q)^2): the distance to an image at complex height
    -(Z + i q) below the plane, Z the sum of the two heights; g(R) = e^{ikR}/(4 pi R).
    The integrand peaks near q = rho, singularly so as Z -> 0, which the
    substitutions q = rho sin(phi) below rho and q = rho cosh(psi) above it take
    out (exactly, for Z = 0), with Gauss-Legendre segments graded toward the
    peak on a width sqrt(2 Z / rho).  Above rho, e^{ikR_q} decays like e^{-k q},
    so the upper limit is rho + 40/k.  Vectorized over ``rho`` and ``z_sum``.

    Needs a passive ground, Re(beta) > 0: otherwise e^{-k beta q} grows and the
    integral diverges.  Delany-Bazley-based layers are not passive at low
    frequency for thin, low-resistivity layers (e.g. 0.03 m at sigma 100 below
    about 60 Hz), a known defect of the one-parameter model (Attenborough &
    Taherzadeh 2026, sec. 1.2 (iv)); that is refused here rather than
    integrated into nonsense.
    """
    if not np.real(beta) > 0.0:
        raise ValueError('image_integrals needs a passive ground (Re beta > 0); got beta = {} '
                         '-- a Delany-Bazley layer below its passive range?'.format(beta))
    rho = np.atleast_1d(np.asarray(rho, dtype=float)).ravel()
    # Z < 0 (even by rounding) would put the root on the growing branch.
    z = np.maximum(np.broadcast_to(np.asarray(z_sum, dtype=float), rho.shape).ravel(), 0.0)
    rho = np.maximum(rho, 0.0)
    m = rho.size
    rho_s = np.maximum(rho, 1e-12)
    w = np.clip(np.sqrt(2.0 * np.maximum(z, 0.0) / rho_s), 1e-6, 0.25)
    half = 0.5 * np.pi
    # Breakpoints on the decay scale of e^{-k beta q} too: far from the plate
    # the integral is carried by q << rho.
    decay_q = np.array([0.3, 1.0, 3.0, 10.0, 30.0]) / (k * max(abs(beta), 1e-6))
    # Below rho: phi in [0, pi/2], q = rho sin(phi), rho^2 - q^2 = rho^2 cos^2(phi).
    ea = np.concatenate((np.zeros((m, 1)),
                         np.arcsin(np.minimum(decay_q[None, :] / rho_s[:, None], 1.0)),
                         np.stack((half - 4 * w, half - w, half - 0.25 * w, half * np.ones(m)), axis=1)),
                        axis=1)
    ea = np.sort(np.clip(ea, 0.0, half), axis=1)
    phi, wphi = _gauss_segments(ea, n)
    qa = rho[:, None] * np.sin(phi)
    ja = rho[:, None] * np.cos(phi) * wphi
    arg_a = (rho[:, None] * np.cos(phi)) ** 2 + z[:, None] ** 2 + 2j * z[:, None] * qa
    # Above rho: psi in [0, psi_max], q = rho cosh(psi), rho^2 - q^2 = -rho^2 sinh^2(psi).
    q_top = 40.0 / k
    psi_max = np.arccosh(1.0 + q_top / rho_s)
    eb = np.concatenate((np.zeros((m, 1)),
                         np.stack((0.25 * w, w, 4 * w), axis=1),
                         np.arccosh(1.0 + decay_q[None, :] / rho_s[:, None]),
                         psi_max[:, None]), axis=1)
    eb = np.sort(np.minimum(eb, psi_max[:, None]), axis=1)
    psi, wpsi = _gauss_segments(eb, n)
    qb = rho[:, None] * np.cosh(psi)
    jb = rho[:, None] * np.sinh(psi) * wpsi
    arg_b = -(rho[:, None] * np.sinh(psi)) ** 2 + z[:, None] ** 2 + 2j * z[:, None] * qb
    q = np.concatenate((qa, qb), axis=1)
    jac = np.concatenate((ja, jb), axis=1)
    arg = np.concatenate((arg_a, arg_b), axis=1)
    # On the branch cut (Z = 0, q > rho) take the +i root: e^{ikR} must decay.
    with np.errstate(invalid='ignore'):
        r = np.where((arg.imag == 0.0) & (arg.real < 0.0), 1j * np.sqrt(np.abs(arg.real)), np.sqrt(arg))
    decay = np.exp(-k * beta * q)
    # Zero-width segments (breakpoints that coincide) put nodes on the
    # singular point with zero weight; drop them rather than form 0 * inf.
    live = (jac != 0.0) & (np.abs(r) > 0.0)
    r = np.where(live, r, 1.0)
    g = np.where(live, np.exp(1j * k * r) / (4.0 * np.pi * r), 0.0)
    big_i = np.sum(decay * g * jac, axis=1)
    big_j = np.sum(decay * g * (1j * k - 1.0 / r) / r * jac, axis=1)
    return big_i, big_j


def exact_half_space_green(k, beta, rho, z_source, z_target, n=10):
    """Exact Green's function of a locally reacting plane and its gradient's pieces.

    G = g(R1) + g(R2) - 2 k beta I (e^{-i omega t}; dG/dz = -i k beta G on z = 0;
    Re beta > 0), and the derivative with respect to the source point's height,
    by parts in q, dG_image/dZ = dg(R2)/dZ - 2 i k beta g(R2) + 2 i k^2 beta^2 I.
    Returns (G, dG/dz_source, J) -- J for the horizontal derivative,
    d/dx_s G_image = dg(R2)/dx_s - 2 k beta J (x_s - x_t).
    """
    rho = np.asarray(rho, dtype=float)
    zs, zt = np.broadcast_arrays(np.asarray(z_source, float), np.asarray(z_target, float))
    shape = np.broadcast(rho, zs).shape
    rho_b = np.broadcast_to(rho, shape); zs = np.broadcast_to(zs, shape); zt = np.broadcast_to(zt, shape)
    r1 = np.hypot(rho_b, zs - zt)
    r2 = np.hypot(rho_b, zs + zt)
    g1 = np.exp(1j * k * r1) / (4 * np.pi * r1)
    g2 = np.exp(1j * k * r2) / (4 * np.pi * r2)
    big_i, big_j = image_integrals(k, beta, rho_b, zs + zt, n)
    big_i = big_i.reshape(shape); big_j = big_j.reshape(shape)
    green = g1 + g2 - 2 * k * beta * big_i
    dg2_dz = g2 * (1j * k - 1.0 / r2) * (zs + zt) / r2
    dgreen_dz = (g1 * (1j * k - 1.0 / r1) * (zs - zt) / r1 + dg2_dz
                 - 2j * k * beta * g2 + 2j * k ** 2 * beta ** 2 * big_i)
    return green, dgreen_dz, big_j


class ImageIntegralTable:
    """I(rho, Z) and J(rho, Z) of :func:`image_integrals`, tabulated for one frequency.

    Interpolated bilinearly in (log(rho + r0), log(Z + r0)) with the oscillation
    and the singular factor taken out: I 4 pi R2 e^{-ikR2} and
    J 4 pi R2^2 e^{-ikR2} vary slowly (R2 = sqrt(rho^2 + Z^2)).
    """

    def __init__(self, k, beta, rho_max, z_max, n_rho=240, n_z=28, offset=1e-5, n=10):
        self.k, self.offset = k, offset
        self.u = np.linspace(np.log(offset), np.log(rho_max + offset), n_rho)
        self.v = np.linspace(np.log(offset), np.log(max(z_max, offset) + offset), n_z)
        rho = np.maximum(np.exp(self.u) - offset, 0.0)
        z = np.maximum(np.exp(self.v) - offset, 0.0)
        R, Z = np.meshgrid(rho, z, indexing='ij')
        big_i, big_j = image_integrals(k, beta, R.ravel(), Z.ravel(), n)
        r2 = np.maximum(np.hypot(R, Z), offset).ravel()
        phase = np.exp(-1j * k * r2)
        self.i_red = (big_i * 4 * np.pi * r2 * phase).reshape(R.shape)
        self.j_red = (big_j * 4 * np.pi * r2 ** 2 * phase).reshape(R.shape)

    def __call__(self, rho, z):
        rho = np.asarray(rho, dtype=float); z = np.asarray(z, dtype=float)
        u = np.clip(np.log(rho + self.offset), self.u[0], self.u[-1])
        v = np.clip(np.log(z + self.offset), self.v[0], self.v[-1])
        fu = (u - self.u[0]) / (self.u[1] - self.u[0]); fv = (v - self.v[0]) / (self.v[1] - self.v[0])
        i0 = np.clip(fu.astype(int), 0, self.u.size - 2); j0 = np.clip(fv.astype(int), 0, self.v.size - 2)
        tu, tv = fu - i0, fv - j0
        def lerp(a):
            return ((1 - tu) * (1 - tv) * a[i0, j0] + tu * (1 - tv) * a[i0 + 1, j0]
                    + (1 - tu) * tv * a[i0, j0 + 1] + tu * tv * a[i0 + 1, j0 + 1])
        r2 = np.maximum(np.hypot(rho, z), self.offset)
        phase = np.exp(1j * self.k * r2)
        return lerp(self.i_red) * phase / (4 * np.pi * r2), lerp(self.j_red) * phase / (4 * np.pi * r2 ** 2)


def disc_mesh(radius, cell):
    """Polar mesh of a disc: centroids, areas and cell bounds, the first cell centerd at the origin.

    A central disc of diameter ``cell`` holds the microphone; rings of width
    about ``cell`` are split into annular sectors about ``cell`` long.  Returns
    (x, y, area, bounds) with bounds[:, :] = (r_in, r_out, phi_lo, phi_hi) per
    cell.  Areas sum to the disc's exactly.
    """
    r_center = min(0.5 * cell, radius)
    xs, ys, areas = [0.0], [0.0], [np.pi * r_center ** 2]
    bounds = [(0.0, r_center, 0.0, 2.0 * np.pi)]
    n_rings = max(1, int(np.ceil((radius - r_center) / cell)))
    edges = np.linspace(r_center, radius, n_rings + 1)
    for inner, outer in zip(edges[:-1], edges[1:]):
        mid = 0.5 * (inner + outer)
        n_cells = max(6, int(np.ceil(2.0 * np.pi * mid / cell)))
        width = 2.0 * np.pi / n_cells
        angles = (np.arange(n_cells) + 0.5) * width
        # Centroid radius of an annular sector.
        r_bar = (2.0 / 3.0) * (outer ** 3 - inner ** 3) / (outer ** 2 - inner ** 2) \
            * np.sinc(1.0 / n_cells)
        xs.extend(r_bar * np.cos(angles)); ys.extend(r_bar * np.sin(angles))
        areas.extend([np.pi * (outer ** 2 - inner ** 2) / n_cells] * n_cells)
        bounds.extend((inner, outer, a - 0.5 * width, a + 0.5 * width) for a in angles)
    return np.array(xs), np.array(ys), np.array(areas), np.array(bounds)


def _cell_subpoints(bound, n):
    """n x n area-weighted quadrature points (midpoint in r^2 and phi) of an annular sector."""
    r_in, r_out, lo, hi = bound
    u = (np.arange(n) + 0.5) / n
    r = np.sqrt(r_in ** 2 + u * (r_out ** 2 - r_in ** 2))
    phi = lo + u * (hi - lo)
    R, P = np.meshgrid(r, phi, indexing='ij')
    weight = np.full(R.size, 0.5 * (r_out ** 2 - r_in ** 2) * (hi - lo) / n ** 2)
    return (R * np.cos(P)).ravel(), (R * np.sin(P)).ravel(), weight


def _inverse_distance_integral(px, py, bound, n_angles=256):
    """Integral over an annular sector of dA / |x - p|, for p inside it: sum of ray lengths.

    In polar coordinates about p the integrand's 1/rho cancels the area
    element's rho, leaving the distance to the cell boundary along each ray.
    """
    r_in, r_out, lo, hi = bound
    phis = (np.arange(n_angles) + 0.5) * 2.0 * np.pi / n_angles
    dx, dy = np.cos(phis), np.sin(phis)

    def inside(t):
        qx, qy = px + t * dx, py + t * dy
        r = np.hypot(qx, qy)
        a = np.mod(np.arctan2(qy, qx) - lo, 2.0 * np.pi)
        full = (hi - lo) >= 2.0 * np.pi - 1e-12
        return (r >= r_in) & (r <= r_out) & (full | (a <= hi - lo))

    lo_t, hi_t = np.zeros(n_angles), np.full(n_angles, 2.0 * r_out + 1.0)
    for _ in range(50):              # the cells are star-shaped about points in them
        mid = 0.5 * (lo_t + hi_t)
        ok = inside(mid)
        lo_t = np.where(ok, mid, lo_t)
        hi_t = np.where(ok, hi_t, mid)
    return np.sum(0.5 * (lo_t + hi_t)) * 2.0 * np.pi / n_angles


def _cell_integral_near(bounds, px, py, height, green, sub, singular):
    """Integral of green(R) over one annular-sector cell, R the distance to (px, py, height).

    ``singular`` (height 0 and the point in the cell): the 1/R singularity is
    integrated exactly by rays and the smooth remainder on sub-points.
    """
    sx, sy, w = _cell_subpoints(bounds, sub)
    r = np.sqrt((sx - px) ** 2 + (sy - py) ** 2 + height ** 2)
    if not singular:
        return np.sum(green(r) * w)
    g0 = green.limit_numerator
    smooth = (green.numerator(r) - g0) / (4.0 * np.pi * r)
    return g0 / (4.0 * np.pi) * _inverse_distance_integral(px, py, bounds) + np.sum(smooth * w)


def _cell_containing(px, py, bounds):
    r = np.hypot(px, py)
    a = np.mod(np.arctan2(py, px), 2.0 * np.pi)
    for index, (r_in, r_out, lo, hi) in enumerate(bounds):
        if r_in <= r <= r_out and ((hi - lo) >= 2.0 * np.pi - 1e-12
                                   or np.mod(a - lo, 2.0 * np.pi) <= hi - lo):
            return index
    raise ValueError('point ({}, {}) is not on the disc'.format(px, py))


def disc_bem_scattered(frequencies, elevations, azimuths, sound_speed,
                       flow_resistance=FLOW_RESISTANCE, radius=PLATE_RADIUS_FT,
                       mic=(0.0, PLATE_MIC_OFFSET_FT), mic_height=PLATE_MIC_HEIGHT_FT,
                       cells_per_wavelength=8, min_cells_across=24, near=2.5, sub=6, ground=None,
                       green='exact'):
    """What a thin rigid disc in soft ground adds at a microphone on or above it, S(f, el, az).

    After Kingan et al. (2023): with the soft half-space's own Green's function
    G, only the disc needs discretizing.  For a locally reacting plane
    (dp/dz = -i k beta p, e^{-i omega t}), Green's second identity gives

        p(r) = p_inc(r) - i k beta_soft  integral over the disc of G(x, r) p(x) dS,

    first solved on the disc for its surface pressure, then evaluated at the
    microphone.  G from a surface point to a point at height h is
    e^{ikR}/(4 pi R) (1 + Q(h/R, R)), Q the soft ground's spherical-wave
    coefficient (:func:`flight_acoustics.spherical_reflection_coefficient`).

    A distant source's field on the soft ground's surface is (1 + Q_soft)
    times a plane wave, so the disc's surface pressure is (1 + Q_soft) times the
    solution u for a unit plane wave, and at the microphone

        p / p_free = 1 + Q_soft e^{2 i k h sin(el)} + (1 + Q_soft) S,

    S = -i k beta integral G u dS / (direct wave at the microphone), which is
    what this returns.  S depends on frequency, elevation and the azimuth of
    the horizontal propagation direction (radians-free: degrees, measured from
    +x toward +y), not on range, and the system matrix on frequency alone.
    For a flush microphone (h = 0) p/p_free = (1 + Q_soft)(1 + S).

    ``ground``, when given, is a dict {'model': ..., **params} for
    :func:`surface_admittance`, replacing Delany-Bazley at ``flow_resistance``.
    ``green='exact'`` (default) uses the exact complex-image Green's function
    (:func:`exact_half_space_green`) at the plate's centimeter ranges, where the
    Weyl-van der Pol coefficient (``green='weyl'``, the earlier behavior) is
    outside its long-range validity.

    ``mic`` is the microphone's (x, y) on the plate from its center (default:
    ARP 4055's 3/4 radius, normal to the track); ``mic_height`` its height
    above the plate (0: flush; :data:`INVERTED_MIC_HEIGHT_FT`: inverted).
    Collocation at cell centroids of a :func:`disc_mesh`; close pairs use
    sub-cell quadrature and self terms integrate the 1/R singularity exactly.
    Returns an array (len(frequencies), len(elevations), len(azimuths)).
    """
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
    elevations = np.radians(np.atleast_1d(np.asarray(elevations, dtype=float)))
    azimuths = np.radians(np.atleast_1d(np.asarray(azimuths, dtype=float)))
    mx, my = mic
    if np.hypot(mx, my) > radius:
        raise ValueError('the microphone must be over the plate')
    out = np.empty((frequencies.size, elevations.size, azimuths.size), dtype=complex)
    for i, f in enumerate(frequencies):
        k = 2.0 * np.pi * f / sound_speed
        cell = min(sound_speed / f / cells_per_wavelength, 2.0 * radius / min_cells_across)
        x, y, area, bounds = disc_mesh(radius, cell)
        beta = _ground_admittance(f, sound_speed, flow_resistance, ground)
        rho_grid = np.linspace(0.0, 2.0 * radius * 1.01 + cell, 1024)
        if green == 'exact':
            # (G_surface / g)(rho) = 2 - 2 k beta I(rho, 0) / g(rho)
            big_i, _ = image_integrals(k, beta, np.maximum(rho_grid, 1e-12), np.zeros_like(rho_grid))
            g_grid = np.exp(1j * k * np.maximum(rho_grid, 1e-12)) / (4 * np.pi * np.maximum(rho_grid, 1e-12))
            opq_grid = 2.0 - 2.0 * k * beta * big_i / g_grid
        else:
            opq_grid = 1.0 + fa.spherical_reflection_coefficient(
                0.0, np.maximum(rho_grid, 1e-9), f, sound_speed, flow_resistance, admittance=beta)

        def surface_green(rho):
            opq = np.interp(rho, rho_grid, opq_grid.real) + 1j * np.interp(rho, rho_grid, opq_grid.imag)
            return np.exp(1j * k * rho) / (4.0 * np.pi * rho) * opq

        surface_green.numerator = lambda rho: np.exp(1j * k * rho) * (
            np.interp(rho, rho_grid, opq_grid.real) + 1j * np.interp(rho, rho_grid, opq_grid.imag))
        surface_green.limit_numerator = opq_grid[0]

        rho = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
        np.fill_diagonal(rho, 1.0)
        kernel = surface_green(rho) * area[None, :]
        for obs, src in np.argwhere(rho < near * cell):
            kernel[obs, src] = _cell_integral_near(bounds[src], x[obs], y[obs], 0.0, surface_green,
                                                   sub, singular=(obs == src))
        system = np.eye(x.size) + 1j * k * beta * kernel

        # Unit plane waves: horizontal wavenumber k cos(el) along each azimuth.
        kx = (k * np.cos(elevations)[:, None] * np.cos(azimuths)[None, :]).ravel()
        ky = (k * np.cos(elevations)[:, None] * np.sin(azimuths)[None, :]).ravel()
        u = np.linalg.solve(system, np.exp(1j * (np.outer(x, kx) + np.outer(y, ky))))

        # Evaluate at the microphone.
        if mic_height == 0.0:
            green_mic = surface_green
        else:
            rho_mic = np.linspace(0.0, 2.0 * radius * 1.01 + cell, 1024)
            r_grid = np.sqrt(rho_mic ** 2 + mic_height ** 2)
            if green == 'exact':
                big_i, _ = image_integrals(k, beta, rho_mic, np.full_like(rho_mic, mic_height))
                opq_mic = 2.0 - 2.0 * k * beta * big_i / (np.exp(1j * k * r_grid) / (4 * np.pi * r_grid))
            else:
                opq_mic = 1.0 + fa.spherical_reflection_coefficient(
                    mic_height / r_grid, r_grid, f, sound_speed, flow_resistance, admittance=beta)

            def green_mic(r):
                opq = np.interp(r, r_grid, opq_mic.real) + 1j * np.interp(r, r_grid, opq_mic.imag)
                return np.exp(1j * k * r) / (4.0 * np.pi * r) * opq

        r_mic = np.sqrt((x - mx) ** 2 + (y - my) ** 2 + mic_height ** 2)
        weights = green_mic(np.maximum(r_mic, 1e-12)) * area
        home = _cell_containing(mx, my, bounds)
        for src in np.flatnonzero(np.hypot(x - mx, y - my) < near * cell + mic_height):
            fine = sub if src != home else max(sub, 24)
            weights[src] = _cell_integral_near(bounds[src], mx, my, mic_height, green_mic, fine,
                                               singular=(mic_height == 0.0 and src == home))
        scattered = -1j * k * beta * (weights @ u)
        direct = np.exp(1j * (kx * mx + ky * my - (k * np.sin(elevations)[:, None]
                                                   * np.ones_like(azimuths)[None, :]).ravel() * mic_height))
        out[i] = (scattered / direct).reshape(elevations.size, azimuths.size)
    return out


#: Radial length of the GR1425's edge taper (8 mm down to 2.5 mm).  Not given
#: on the GRAS drawing; an assumption, with its sensitivity checked separately.
PLATE_TAPER_LENGTH_FT = 0.020 / 0.3048


def raised_plate_mesh(radius, thickness, edge_thickness, taper_length, cell):
    """Panels of a plate lying on the ground: flat top, conical taper, vertical rim.

    Each panel is a patch of a surface of revolution: a segment in (r, z) from
    (r0, z0) to (r1, z1) swept over [phi_lo, phi_hi].  Segments run from the
    center outward and down -- top (0, t) -> (a - L, t), taper -> (a, t_e), rim ->
    (a, 0) -- so the normal (-dz, dr)/|.| points into the air.  Returns an array
    (n, 6) of (r0, z0, r1, z1, phi_lo, phi_hi).
    """
    r_taper = radius - taper_length
    panels = []
    # Top: a central disc, then rings.
    r_c = min(0.5 * cell, r_taper)
    panels.append((0.0, thickness, r_c, thickness, 0.0, 2 * np.pi))
    n_rings = max(1, int(np.ceil((r_taper - r_c) / cell)))
    for r0, r1 in zip(np.linspace(r_c, r_taper, n_rings + 1)[:-1], np.linspace(r_c, r_taper, n_rings + 1)[1:]):
        n = max(6, int(np.ceil(2 * np.pi * 0.5 * (r0 + r1) / cell)))
        w = 2 * np.pi / n
        panels += [(r0, thickness, r1, thickness, i * w, (i + 1) * w) for i in range(n)]
    segments = []
    if taper_length > 0.0:
        segments.append((r_taper, thickness, radius, edge_thickness))
    if edge_thickness > 0.0:
        segments.append((radius, edge_thickness, radius, 0.0))
    for r0, z0, r1, z1 in segments:
        length = np.hypot(r1 - r0, z1 - z0)
        n_seg = max(1, int(np.ceil(length / cell)))
        u = np.linspace(0.0, 1.0, n_seg + 1)
        for ua, ub in zip(u[:-1], u[1:]):
            ra, rb = r0 + ua * (r1 - r0), r0 + ub * (r1 - r0)
            za, zb = z0 + ua * (z1 - z0), z0 + ub * (z1 - z0)
            n = max(6, int(np.ceil(2 * np.pi * 0.5 * (ra + rb) / cell)))
            w = 2 * np.pi / n
            panels += [(ra, za, rb, zb, i * w, (i + 1) * w) for i in range(n)]
    return np.array(panels)


def _panel_points(panel, n_u, n_phi):
    """Quadrature points, weights (area) and the panel's unit normal (into the air)."""
    r0, z0, r1, z1, lo, hi = panel
    dr, dz = r1 - r0, z1 - z0
    length = np.hypot(dr, dz)
    u = (np.arange(n_u) + 0.5) / n_u
    phi = lo + (np.arange(n_phi) + 0.5) / n_phi * (hi - lo)
    if dz == 0.0 and r0 == 0.0:
        r = np.sqrt(u) * r1                                  # area-uniform on the central disc
        du_area = 0.5 * r1 ** 2 / n_u
        U, P = np.meshgrid(r, phi, indexing='ij')
        w = np.full(U.size, du_area * (hi - lo) / n_phi)
        z = np.full(U.size, z0)
        rr = U.ravel()
    else:
        rr_u = r0 + u * dr
        z_u = z0 + u * dz
        U, P = np.meshgrid(rr_u, phi, indexing='ij')
        Z, _ = np.meshgrid(z_u, phi, indexing='ij')
        rr = U.ravel(); z = Z.ravel()
        w = rr * length / n_u * (hi - lo) / n_phi
    pp = P.ravel()
    nr, nz = -dz / length, dr / length
    x, y = rr * np.cos(pp), rr * np.sin(pp)
    normal = np.stack((nr * np.cos(pp), nr * np.sin(pp), np.full(pp.size, nz)), axis=1)
    return np.stack((x, y, z), axis=1), w, normal


def _panel_geometry(panel, u, phi):
    """Points and area Jacobian of a surface-of-revolution panel at parameters (u, phi)."""
    r0, z0, r1, z1, lo, hi = panel
    dr, dz = r1 - r0, z1 - z0
    length = np.hypot(dr, dz)
    r = r0 + u * dr
    z = z0 + u * dz
    x, y = r * np.cos(phi), r * np.sin(phi)
    jac = r * length * (hi - lo)                  # dA = jac du dt, phi = lo + t (hi - lo)
    nr, nz = -dz / length, dr / length
    normal = np.stack((nr * np.cos(phi), nr * np.sin(phi), np.full(np.shape(phi), nz)), axis=-1)
    return np.stack((x, y, z), axis=-1), jac, normal


def _adaptive_points(panel, targets, ratio=0.35, max_depth=9, exclude=None):
    """Quadrature points on a panel, refined where it comes close to any of ``targets``.

    Geometry-driven: a parameter cell is split into four while its size exceeds
    ``ratio`` times its distance to the nearest target (the collocation point
    and its image below the ground, whose kernels are near-singular on the
    scale of the plate's height).  Leaves use a 2 x 2 Gauss rule.  Returns
    (points, weights, normals).  ``exclude`` drops leaf points coinciding with
    a singular target (principal value on a flat panel).
    """
    r0, z0, r1, z1, lo, hi = panel
    length = np.hypot(r1 - r0, z1 - z0)
    g = 0.5 * np.array([1 - 1 / np.sqrt(3), 1 + 1 / np.sqrt(3)])
    cells = [(0.0, 1.0, 0.0, 1.0, 0)]
    out_p, out_w, out_n = [], [], []
    while cells:
        u0, u1, t0, t1, depth = cells.pop()
        uc, tc = 0.5 * (u0 + u1), 0.5 * (t0 + t1)
        center, _, _ = _panel_geometry(panel, uc, lo + tc * (hi - lo))
        r_mid = r0 + uc * (r1 - r0)
        size = max((u1 - u0) * length, (t1 - t0) * (hi - lo) * max(r_mid, 1e-12))
        dist = np.min(np.linalg.norm(targets - center, axis=1))
        if depth < max_depth and size > ratio * dist:
            um, tm = uc, tc
            cells += [(u0, um, t0, tm, depth + 1), (um, u1, t0, tm, depth + 1),
                      (u0, um, tm, t1, depth + 1), (um, u1, tm, t1, depth + 1)]
            continue
        uu = u0 + g * (u1 - u0)
        tt = t0 + g * (t1 - t0)
        U, TT = np.meshgrid(uu, tt, indexing='ij')
        pts, jac, nrm = _panel_geometry(panel, U.ravel(), lo + TT.ravel() * (hi - lo))
        out_p.append(pts); out_n.append(nrm)
        out_w.append(jac * 0.25 * (u1 - u0) * (t1 - t0))
    pts, w, nrm = np.concatenate(out_p), np.concatenate(out_w), np.concatenate(out_n)
    if exclude is not None:
        keep = np.linalg.norm(pts - exclude, axis=1) > 1e-12
        pts, w, nrm = pts[keep], w[keep], nrm[keep]
    return pts, w, nrm


def _half_space_green_gradient_n(k, points, weights, normals, target, q_of, delta):
    """sum_w dG(y, x)/dn_y for G = g(|y - x|) + Q g(|y - x'|), x' the image of x.

    The image term's coefficient Q depends on the geometry (grazing cosine and
    image range); its normal derivative is taken by a central difference.
    """
    def g_and_grad(y, x):
        d = y - x
        r = np.linalg.norm(d, axis=1)
        g = np.exp(1j * k * r) / (4 * np.pi * r)
        dg = g * (1j * k - 1.0 / r) / r                      # dg/dr / r
        return g, dg[:, None] * d
    image = target * np.array([1.0, 1.0, -1.0])
    g_d, grad_d = g_and_grad(points, target)
    g_i, grad_i = g_and_grad(points, image)
    q = q_of(points, target)
    dq = (q_of(points + delta * normals, target) - q_of(points - delta * normals, target)) / (2 * delta)
    dgdn = np.sum(grad_d * normals, axis=1) + q * np.sum(grad_i * normals, axis=1) + dq * g_i
    return np.sum(dgdn * weights)


def raised_plate_scattering(frequencies, elevations, azimuths, sound_speed,
                            flow_resistance=FLOW_RESISTANCE, ground=None, radius=PLATE_RADIUS_FT,
                            thickness=PLATE_THICKNESS_FT, edge_thickness=PLATE_EDGE_THICKNESS_FT,
                            taper_length=PLATE_TAPER_LENGTH_FT, mic=(0.0, PLATE_MIC_OFFSET_FT),
                            cells_per_wavelength=8, min_cells_across=24, near=2.5, sub=5,
                            green='exact'):
    """Pressure at a flush microphone on a rigid plate lying on soft ground, per unit direct wave.

    The plate is a rigid body on the ground: flat top ``thickness`` up, a
    conical taper of ``taper_length`` down to ``edge_thickness``, and a vertical
    rim (:func:`raised_plate_mesh`).  With the soft half-space's Green's
    function G (direct plus image term with the spherical-wave coefficient),
    the ground outside the plate drops out exactly and, for the rigid body,

        1/2 p(x) = p_inc(x) + integral over the plate's exposed surface of p dG/dn dS

    on the surface (collocation at panel centroids; 1/2 away from edges).  The
    incident field is a direct plane wave plus Q times its ground reflection;
    by linearity the microphone's pressure is P = P_d + Q P_r, and this returns
    (P_d, P_r), each (len(frequencies), len(elevations), len(azimuths)),
    normalized by the direct wave at the microphone.  A frame then applies its
    own Q (range-dependent near grazing).  The microphone is on the top at
    ``mic`` = (x, y); azimuths as in :func:`disc_bem_scattered`.

    ``green='exact'`` (default) uses the exact complex-image Green's function,
    whose normal derivative follows from I and J without differentiating an
    approximation (:func:`exact_half_space_green`); ``'weyl'`` keeps the
    Weyl-van der Pol coefficient with a finite-difference derivative.
    """
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
    el = np.radians(np.atleast_1d(np.asarray(elevations, dtype=float)))
    az = np.radians(np.atleast_1d(np.asarray(azimuths, dtype=float)))
    mx, my = mic
    if np.hypot(mx, my) >= radius - taper_length:
        raise ValueError('the microphone must be on the flat top')
    pd_out = np.empty((frequencies.size, el.size, az.size), dtype=complex)
    pr_out = np.empty_like(pd_out)
    target_mic = np.array([mx, my, thickness])
    for fi, f in enumerate(frequencies):
        k = 2 * np.pi * f / sound_speed
        cell = min(sound_speed / f / cells_per_wavelength, 2 * radius / min_cells_across)
        panels = raised_plate_mesh(radius, thickness, edge_thickness, taper_length, cell)
        beta = _ground_admittance(f, sound_speed, flow_resistance, ground)

        def q_of(y, x):
            zsum = y[:, 2] + x[2]
            rho = np.hypot(y[:, 0] - x[0], y[:, 1] - x[1])
            r2 = np.hypot(rho, zsum)
            return fa.spherical_reflection_coefficient(zsum / r2, np.maximum(r2, 1e-9), f, sound_speed,
                                                       flow_resistance, admittance=beta)

        table = (ImageIntegralTable(k, beta, 2.0 * radius * 1.02 + 2 * cell, 2.0 * thickness * 1.05)
                 if green == 'exact' else None)

        def dgdn_exact(points, weights, normals, target):
            d = points - target
            r1 = np.linalg.norm(d, axis=1)
            g1 = np.exp(1j * k * r1) / (4 * np.pi * r1)
            direct = np.sum((g1 * (1j * k - 1.0 / r1) / r1)[:, None] * d * normals, axis=1)
            rho = np.hypot(d[:, 0], d[:, 1]); zs = points[:, 2] + target[2]
            r2 = np.hypot(rho, zs)
            g2 = np.exp(1j * k * r2) / (4 * np.pi * r2)
            big_i, big_j = table(rho, zs)
            dz = g2 * (1j * k - 1.0 / r2) * zs / r2 - 2j * k * beta * g2 + 2j * k ** 2 * beta ** 2 * big_i
            dh = g2 * (1j * k - 1.0 / r2) / r2 - 2 * k * beta * big_j
            image = normals[:, 2] * dz + (normals[:, 0] * d[:, 0] + normals[:, 1] * d[:, 1]) * dh
            return np.sum((direct + image) * weights)

        quad = [_panel_points(pn, 1, 1) for pn in panels]
        centers = np.array([q[0][0] for q in quad])
        centers[0] = (0.0, 0.0, thickness)            # the central disc's centroid is its center
        normals = np.array([q[2][0] for q in quad])
        areas = np.array([q[1][0] for q in quad])
        n = panels.shape[0]
        # Far pairs: one-point rule, vectorized over all pairs.
        d = centers[None, :, :] - centers[:, None, :]                      # y - x, (obs, src, 3)
        r = np.linalg.norm(d, axis=2)
        np.fill_diagonal(r, 1.0)
        g = np.exp(1j * k * r) / (4 * np.pi * r)
        dgd = np.sum((g * (1j * k - 1.0 / r) / r)[:, :, None] * d * normals[None, :, :], axis=2)
        img = centers[:, None, :] * np.array([1.0, 1.0, -1.0])
        di = centers[None, :, :] - img
        ri = np.linalg.norm(di, axis=2)
        gi = np.exp(1j * k * ri) / (4 * np.pi * ri)
        dgi = np.sum((gi * (1j * k - 1.0 / ri) / ri)[:, :, None] * di * normals[None, :, :], axis=2)
        delta = 1e-3 * cell
        if green == 'exact':
            zsum = centers[None, :, 2] + centers[:, None, 2]
            rho = np.hypot(d[:, :, 0], d[:, :, 1])
            big_i, big_j = table(rho, zsum)
            dz = gi * (1j * k - 1.0 / ri) * zsum / ri - 2j * k * beta * gi + 2j * k ** 2 * beta ** 2 * big_i
            dh = gi * (1j * k - 1.0 / ri) / ri - 2 * k * beta * big_j
            dimg = (normals[None, :, 2] * dz
                    + (normals[None, :, 0] * d[:, :, 0] + normals[None, :, 1] * d[:, :, 1]) * dh)
            kernel = (dgd + dimg) * areas[None, :]
        else:
            zsum = centers[None, :, 2] + centers[:, None, 2]
            rho = np.hypot(d[:, :, 0], d[:, :, 1])
            r2 = np.hypot(rho, zsum)
            qq = fa.spherical_reflection_coefficient((zsum / r2).ravel(), np.maximum(r2, 1e-9).ravel(), f,
                                                     sound_speed, flow_resistance, admittance=beta).reshape(r.shape)
            zp = zsum + delta * normals[None, :, 2]
            rhop = np.hypot(d[:, :, 0] + delta * normals[None, :, 0], d[:, :, 1] + delta * normals[None, :, 1])
            zm = zsum - delta * normals[None, :, 2]
            rhom = np.hypot(d[:, :, 0] - delta * normals[None, :, 0], d[:, :, 1] - delta * normals[None, :, 1])
            r2p, r2m = np.hypot(rhop, zp), np.hypot(rhom, zm)
            qp = fa.spherical_reflection_coefficient((zp / r2p).ravel(), r2p.ravel(), f, sound_speed,
                                                     flow_resistance, admittance=beta).reshape(r.shape)
            qm = fa.spherical_reflection_coefficient((zm / r2m).ravel(), r2m.ravel(), f, sound_speed,
                                                     flow_resistance, admittance=beta).reshape(r.shape)
            kernel = (dgd + qq * dgi + (qp - qm) / (2 * delta) * gi) * areas[None, :]
        # Near pairs and self terms: sub-panel quadrature (the direct term's
        # singular part vanishes on a flat panel: (y - x).n = 0 there).
        # A pair is "near" when either the collocation point or its image is
        # close to the source panel: the image of a point at height z sits 2 z
        # below it, so on a thin plate every close pair is near-singular.
        near_pairs = np.argwhere(np.minimum(r, ri) < near * cell)
        for obs, src in near_pairs:
            x = centers[obs]
            pts, w, nrm = _adaptive_points(panels[src], np.stack((x, x * [1, 1, -1])),
                                           exclude=x if obs == src else None)
            kernel[obs, src] = (dgdn_exact(pts, w, nrm, x) if green == 'exact' else
                                _half_space_green_gradient_n(k, pts, w, nrm, x, q_of, delta))
        system = 0.5 * np.eye(n) - kernel
        # Incident plane waves, direct (downward) and ground-reflected (upward).
        kx = (k * np.cos(el)[:, None] * np.cos(az)[None, :]).ravel()
        ky = (k * np.cos(el)[:, None] * np.sin(az)[None, :]).ravel()
        kz = (k * np.sin(el)[:, None] * np.ones_like(az)[None, :]).ravel()
        phase_h = np.outer(centers[:, 0], kx) + np.outer(centers[:, 1], ky)
        rhs = np.hstack((np.exp(1j * (phase_h - np.outer(centers[:, 2], kz))),
                         np.exp(1j * (phase_h + np.outer(centers[:, 2], kz)))))
        surface = np.linalg.solve(system, rhs)
        # At the microphone: 1/2 p = p_inc + integral, on the flat top.
        weights = np.empty(n, dtype=complex)
        mic_image = target_mic * np.array([1.0, 1.0, -1.0])
        dist = np.minimum(np.linalg.norm(centers - target_mic, axis=1),
                          np.linalg.norm(centers - mic_image, axis=1))
        for src in range(n):
            if dist[src] < near * cell:
                pts, w, nrm = _adaptive_points(panels[src], np.stack((target_mic, mic_image)),
                                               exclude=target_mic)
                weights[src] = (dgdn_exact(pts, w, nrm, target_mic) if green == 'exact' else
                                _half_space_green_gradient_n(k, pts, w, nrm, target_mic, q_of, delta))
            else:
                one = (centers[src:src + 1], areas[src:src + 1], normals[src:src + 1], target_mic)
                weights[src] = (dgdn_exact(*one) if green == 'exact' else
                                _half_space_green_gradient_n(k, *one[:3], target_mic, q_of, delta))
        inc_mic_h = kx * mx + ky * my
        inc_d = np.exp(1j * (inc_mic_h - kz * thickness))
        inc_r = np.exp(1j * (inc_mic_h + kz * thickness))
        m = el.size * az.size
        p_d = 2.0 * (inc_d + weights @ surface[:, :m])
        p_r = 2.0 * (inc_r + weights @ surface[:, m:])
        pd_out[fi] = (p_d / inc_d).reshape(el.size, az.size)
        pr_out[fi] = (p_r / inc_d).reshape(el.size, az.size)
    return pd_out, pr_out


#: Default grid for :func:`disc_bem_table`.
BEM_ELEVATIONS = np.concatenate((np.geomspace(0.05, 10.0, 40), np.linspace(11.0, 90.0, 40)))
BEM_AZIMUTHS = np.arange(0.0, 360.0, 10.0)


def disc_bem_table(bands, sound_speed, flow_resistance=FLOW_RESISTANCE, sub_bands=5,
                   elevations=BEM_ELEVATIONS, azimuths=BEM_AZIMUTHS, **options):
    """Precomputed S over each band's sub-frequencies, for :func:`board_disc_bem`.

    Returns a dict: frequencies, elevations, azimuths (deg), S (complex), and
    the microphone height used (``options`` go to :func:`disc_bem_scattered`).
    """
    bands = np.asarray(bands, dtype=float)
    offsets = sub_band_factors(sub_bands)
    frequencies = np.sort((bands[:, None] * offsets[None, :]).ravel())
    scattered = disc_bem_scattered(frequencies, elevations, azimuths, sound_speed, flow_resistance,
                                   **options)
    return dict(frequencies=frequencies, elevations=np.asarray(elevations, float),
                azimuths=np.asarray(azimuths, float), S=scattered,
                mic_height=options.get('mic_height', PLATE_MIC_HEIGHT_FT),
                ground=options.get('ground'), flow_resistance=flow_resistance, sub_bands=sub_bands)


def table_sub_bands(table, sub_bands=None):
    """The number of sub-frequencies per band ``table`` was computed with.

    Evaluating a table at a different count picks frequencies it does not
    hold, and averages over the wrong number, so a count that disagrees is
    refused.  Tables from before the count was recorded used the default, 5.
    """
    stored = int(table.get('sub_bands', 5))
    if sub_bands is not None and int(sub_bands) != stored:
        raise ValueError('the table holds {} sub-frequencies per band, not {}'.format(stored, sub_bands))
    return stored


def board_disc_bem(bands, source_height, ground_distance, sound_speed,
                   flow_resistance=FLOW_RESISTANCE, sub_bands=None, table=None,
                   source_dx=None, source_dy=None, mirror_y=False):
    """The ground-plane microphone on a thin rigid disc in soft ground, band averaged, dB re free field.

    |1 + Q e^{2ikh sin(el)} + (1 + Q) S|^2, averaged over ``sub_bands``
    frequencies per band, with S interpolated (real and imaginary parts,
    bilinearly in elevation and azimuth) from ``table`` (:func:`disc_bem_table`)
    and Q from each frame's own geometry.  The propagation azimuth is from the
    source toward the microphone, from ``source_dx``, ``source_dy`` (the
    source's offset from the microphone); without them the source is taken to
    be along -x.  ``mirror_y`` puts the microphone on the plate's other side
    (the -y offset), by reflecting the azimuth.  Bands above the table's range
    are NaN.  ``sub_bands`` is the table's own; a different count would pick
    frequencies the table does not hold.
    """
    from scipy.interpolate import RegularGridInterpolator

    sub_bands = table_sub_bands(table, sub_bands)
    f, hs, d2 = _broadcast(bands, source_height, ground_distance)
    elevation = np.degrees(np.arctan2(hs, d2))
    image_range = np.hypot(d2, hs)
    cos_theta = hs / image_range
    if source_dx is None:
        azimuth = np.zeros_like(hs)
    else:
        _, dx, dy = _broadcast(bands, source_dx, source_dy)
        azimuth = np.mod(np.degrees(np.arctan2(-dy, -dx)), 360.0)
    if mirror_y:
        azimuth = np.mod(-azimuth, 360.0)
    t_az = np.concatenate((table['azimuths'], [table['azimuths'][0] + 360.0]))
    height = table['mic_height']
    offsets = sub_band_factors(sub_bands)
    top = table['frequencies'].max() * (1.0 + 1e-9)
    energy = np.zeros(f.shape)
    for factor in offsets:
        fj = f * factor
        q = fa.spherical_reflection_coefficient(
            cos_theta, image_range, fj, sound_speed, flow_resistance,
            admittance=_ground_admittance(fj, sound_speed, flow_resistance, table.get('ground')))
        s_frame = np.full(f.shape, np.nan, dtype=complex)
        for b in range(f.shape[0]):
            if fj[b, 0] > top:
                continue
            row = np.argmin(np.abs(table['frequencies'] - fj[b, 0]))
            grid = table['S'][row]
            grid = np.concatenate((grid, grid[:, :1]), axis=1)          # periodic in azimuth
            points = np.column_stack((np.clip(elevation[b], table['elevations'][0],
                                              table['elevations'][-1]), azimuth[b]))
            s_frame[b] = (RegularGridInterpolator((table['elevations'], t_az), grid.real)(points)
                          + 1j * RegularGridInterpolator((table['elevations'], t_az), grid.imag)(points))
        k = 2.0 * np.pi * fj / sound_speed
        ratio = 1.0 + q * np.exp(2j * k * height * np.sin(np.radians(elevation))) + (1.0 + q) * s_frame
        energy += np.abs(ratio) ** 2
    return 10.0 * np.log10(energy / sub_bands)


def board_soft_ground(bands, source_height, ground_distance, sound_speed,
                      flow_resistance=FLOW_RESISTANCE):
    """No plate at all: the diaphragm the plate's thickness above the site's soft ground."""
    return board_uniform(bands, source_height, ground_distance, sound_speed, flow_resistance,
                         mic_height=PLATE_THICKNESS_FT)


#: The corrections compared, by name.
BOARD_MODELS = {
    'rigid_plane': board_rigid_plane,
    'soft_ground': board_soft_ground,
    'fresnel_strip': board_fresnel_strip,
    'fresnel_disc': board_fresnel_disc,
    'nmid': board_nmid,
}

#: What each correction assumes, for plots and reports.
MODEL_LABELS = {
    'rigid_plane': 'Ideal rigid plane: constant +6 dB (current correction)',
    'soft_ground': 'Soft ground, no plate',
    'fresnel_strip': 'Fresnel zone, 2-D strip across the plate (Hothersall & Harriott)',
    'fresnel_disc': 'Fresnel zone, 400 mm disc (Nord2000-style)',
    'nmid': 'Edge diffraction, 2-D strip (corrected De Jong: Lam & Monazzam nMID)',
}
