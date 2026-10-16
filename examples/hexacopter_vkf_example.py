#!/usr/bin/env python3
"""
Hexacopter Acoustic Signal Separation using Vold-Kalman Filter

The bundled record (example_data/F100_20_P#1[FA1FZC-3.h5].mat) is one 100 ft,
20 mph flyover of a reconfigurable hexacopter: the pressures of 16 microphones
as recorded, the aircraft track from its autopilot, the speeds of its six
rotors and the microphone positions.  The script resamples one microphone's
pressure onto emission time with flight_acoustics.dedopplerize
(docs/THEORY.md §10), which removes the propagation delay and the Doppler
shift, and then separates the blade-passage harmonics of every rotor with the
Vold-Kalman filter at the speeds the rotors turned.  The number of rotors is
taken from the speed record.  The settings at the top of the file keep the
first ten harmonics, which stand above the broadband floor on this record,
with bandwidths that follow how narrow the de-Dopplerized tones are: 5 Hz for
the fundamental, 3 Hz for harmonics 2-4 and 2 Hz above.

``--doppler track`` instead filters the signal as recorded at the speeds the
microphone hears, delayed and Doppler-shifted along the track
(order_tracking.observed_order_track), and ``--doppler none`` filters it as
recorded at the rotors' own speeds, which leaves each harmonic off by up to
2.5 % on this pass.

Microphones 1-12 and 16 are on ground boards; 13-15 are on 1.2 m poles beside
boards 8, 10 and 6, and their spectra carry ground-reflection interference
(info.micType in the file).

Translated from a MATLAB script developed by Joel Sundar Rachaprolu at Penn State
University. 
"""

import argparse
import functools
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, resample_poly, sosfiltfilt, spectrogram
import os
import time
import concurrent.futures as cf
from panam.cli import add_output_options, save_all_or_show
from panam.flight_acoustics import P_REF, dedopplerize, highpass, psd_welch
from panam.order_tracking import observed_order_track
from panam.vold_kalman_filter import vold_kalman_filter


# ----------------------------
# Analysis flags (edit here)
# ----------------------------
# Data
# The bundled record, in the repository's example_data/ whatever the current directory.
EXAMPLE_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'example_data')
MAT_FILE = os.path.join(EXAMPLE_DATA, 'F100_20_P#1[FA1FZC-3.h5].mat')
DOWNSAMPLE_FACTOR = 8
# Low enough to keep the blade-passage fundamental, 45-90 Hz on this record.
HIGHPASS_CUTOFF_HZ = 30

# VKF settings
# Microphone to separate, 1-based: 16 lies beneath the flight path.
MIC = 16
# Blade-passage-frequency harmonics k x BPF.  Above the 10th, the rotors'
# harmonics lie within about 3 dB of the broadband floor on this record, and
# their components would be mostly broadband noise.
VKF_ORDERS = list(range(1, 11))
VKF_P = 2
# Full -3 dB bandwidth of each harmonic, Hz.  De-Dopplerized, about 90 % of a
# tone's energy lies within 3 Hz of its track; the strong fundamental takes a
# wider band and the weaker harmonics narrower ones, which admit less noise.
VKF_BANDWIDTH_HZ = {k: 5.0 if k == 1 else 3.0 if k <= 4 else 2.0 for k in VKF_ORDERS}
# The bandwidth of a harmonic the table leaves out: VKF_BW_PERCENT of its
# frequency, clipped to [VKF_MIN_BW, VKF_MAX_BW] Hz.
VKF_MIN_BW = 3
VKF_MAX_BW = 9
VKF_BW_PERCENT = 0.05
VKF_SOLVER = "auto"  # banded Cholesky; sparse UMFPACK/SuperLU when needed
VKF_USE_COUPLING = True  # False is faster but less accurate
VKF_N_JOBS = None  # defaults to CPU count
# How the Doppler shift is handled, and the title of the signal's plots.
# 'dedopplerize': resample onto emission time and filter at the rotors' speeds;
# 'track': filter the recorded signal at the speeds the microphone hears;
# 'none': filter the recorded signal at the rotors' speeds.
DOPPLER = 'dedopplerize'
TITLES = {'dedopplerize': 'De-Dopplerized (emission time)', 'track': 'As recorded, Doppler-shifted tracks',
          'none': 'As recorded (reception time)'}
DOPPLER_MODES = tuple(TITLES)
SOUND_SPEED_M_S = 343.0
NO_BLADE = 2

# Plot settings
PLOT_INTERACTIVE = False
ZOOM_TIME_WINDOW = (108, 108.2)
# Bands of the residual-energy summary, Hz.
RESIDUAL_BANDS_HZ = ((40, 100), (100, 300), (300, 600), (600, 1000), (1000, 2000))
# Spectrogram figure: band (Hz) and rotor whose blade-passage harmonics are
# drawn.  R2's 4th and 5th harmonics are the strongest tones in this band.
SPECTROGRAM_BAND_HZ = (270, 400)
SPECTROGRAM_ROTOR = 2
SPECTROGRAM_WINDOW_S = 0.25
PSD_LABEL = 'PSD, dB re (20 µPa)²/Hz'

def load_hexacopter_data(mat_file_path):
    """Load hexacopter acoustic and aircraft data at the recorded rate.
    
    Args:
        mat_file_path: Path to .mat file containing hexacopter data.
        
    Returns:
        Dictionary with keys: acoustics_T (s), acoustics_P (n, n_mics), Pa, as
        stored (float32 in the bundled record), aircraft_time (s),
        aircraft_rpm (n_time, n_rotors), rad/s, and, when the file has them,
        track_time (s), track_position (n, 3), m, mic_location (n_mics, 3),
        m, in the track's frame, and mic_type (n_mics,), 'ground' or 'pole'.
    """
    print("Loading data from:", mat_file_path)
    
    mat_data = loadmat(mat_file_path, squeeze_me=True)
    acoustics = mat_data['acoustics']
    aircraft = mat_data['aircraft']
    
    acoustics_T = acoustics['T'].item()
    # The recorder's float32 samples; one microphone at a time is taken in float64.
    acoustics_P = acoustics['P'].item()
    aircraft_time = aircraft['time_onboard'].item()
    aircraft_rpm = np.asarray(aircraft['rpm'].item(), dtype=float)
    aircraft_rpm = aircraft_rpm.reshape(len(aircraft_rpm), -1)
    
    print(f"Original data shape: T={acoustics_T.shape}, P={acoustics_P.shape}")
    print(f"Aircraft RPM shape: {aircraft_rpm.shape}")
    
    print(f"Sampling frequency: {1.0 / (acoustics_T[1] - acoustics_T[0]):.0f} Hz")
    
    data = {
        'acoustics_T': acoustics_T,
        'acoustics_P': acoustics_P,
        'aircraft_time': aircraft_time,
        'aircraft_rpm': aircraft_rpm,
    }
    if 'T' in aircraft.dtype.names and 'position' in aircraft.dtype.names:
        data['track_time'] = aircraft['T'].item()
        data['track_position'] = np.asarray(aircraft['position'].item()).T
    info = mat_data.get('info')
    if info is not None and 'micLocation' in info.dtype.names:
        data['mic_location'] = np.asarray(info['micLocation'].item(), dtype=float).T
    if info is not None and 'micType' in info.dtype.names:
        data['mic_type'] = np.atleast_1d(info['micType'].item()).astype(str)
    return data


def apply_highpass_filter(acoustics_P, cutoff_freq, fs):
    """Apply highpass filter to acoustic pressure signals.
    
    Args:
        acoustics_P: Acoustic pressure array, shape (n_samples, n_mics).
        cutoff_freq: Highpass filter cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        
    Returns:
        Filtered acoustic pressure array with same shape as input.
    """
    print(f"Applying {cutoff_freq} Hz highpass filter...")
    
    if acoustics_P.ndim == 1:
        return highpass(acoustics_P, cutoff_freq, fs)
    else:
        # Apply to each column independently
        return np.column_stack([highpass(acoustics_P[:, i], cutoff_freq, fs) 
                                for i in range(acoustics_P.shape[1])])


def rotor_speed_function(aircraft_time, aircraft_rpm):
    """Rotor speeds at any times: cubic inside the speed record, its first or
    last speed outside it, since a cubic extrapolated past the record runs away.

    Args:
        aircraft_time: Times of the speed record, s.
        aircraft_rpm: Rotor speeds, rad/s, shape (n_time, n_rotors) or
            (n_time,) for a single rotor.

    Returns:
        A function of times, s, that returns speeds, rad/s, shape (n, n_rotors).
    """
    aircraft_time = np.asarray(aircraft_time, dtype=float)
    rpm = np.asarray(aircraft_rpm, dtype=float)
    cubic = interp1d(aircraft_time, rpm.reshape(len(rpm), -1), kind='cubic', axis=0)
    return lambda t: cubic(np.clip(t, aircraft_time[0], aircraft_time[-1]))


def interpolate_rpm_to_acoustic_time(aircraft_time, aircraft_rpm, acoustics_T):
    """Interpolate RPM data to acoustic sampling times.
    
    Args:
        aircraft_time: Aircraft data time vector.
        aircraft_rpm: Rotor speed array, rad/s, shape (n_time, n_rotors) or
            (n_time,) for a single rotor.
        acoustics_T: Acoustic time vector to interpolate to.
        
    Returns:
        Interpolated rotor speed array, rad/s, shape (len(acoustics_T), n_rotors),
        held at the ends of the speed record (:func:`rotor_speed_function`).
    """
    print(f"Interpolating RPM data...")
    rpmo = rotor_speed_function(aircraft_time, aircraft_rpm)(acoustics_T)
    print(f"RPM range: {rpmo.min():.1f} - {rpmo.max():.1f} rad/s")
    return rpmo


def microphone_position(data, mic, receiver=None):
    """Position (x, y, z), m, of microphone ``mic`` (1-based) in the track's
    frame: ``receiver`` when given, otherwise the file's.  None when the record
    has no aircraft track or no position for the microphone."""
    if 'track_time' not in data or 'track_position' not in data:
        return None
    if receiver is not None:
        return np.asarray(receiver, dtype=float)
    if 'mic_location' not in data:
        return None
    return np.asarray(data['mic_location'][mic - 1], dtype=float)


def dedopplerize_record(data, mic, sound_speed=SOUND_SPEED_M_S, receiver=None):
    """One microphone's pressure on the emission-time grid of the aircraft track.

    Args:
        data: Dictionary from load_hexacopter_data, with acoustics_T (s),
            acoustics_P (n, n_mics), Pa, track_time (s), track_position
            (m, 3), m, and, unless ``receiver`` is given, mic_location
            (n_mics, 3), m, in the track's frame.  The acoustic times must be
            on the track's clock.
        mic: Microphone, 1-based.
        sound_speed: Speed of sound, m/s.
        receiver: Microphone position (x, y, z), m, in place of row ``mic``
            of mic_location.

    Returns:
        Tuple (emission_times, pressure): emission times, s, spaced as the
        acoustic samples, and the pressure, Pa, received at each emission
        time plus the propagation time.  Emission times whose sound reached
        the microphone outside the record are left out.
    """
    observer = microphone_position(data, mic, receiver)
    if observer is None:
        raise ValueError(f"the record has no aircraft track or no position for microphone {mic}, "
                         "so it cannot be de-Dopplerized")
    time_ = np.asarray(data['acoustics_T'], dtype=float)
    pressure = np.asarray(data['acoustics_P'])[:, mic - 1].astype(float)
    emission_times, dedopplerized = dedopplerize(
        time_[np.newaxis, :], pressure[np.newaxis, :], sound_speed, np.asarray(data['track_time'], dtype=float),
        np.asarray(data['track_position'], dtype=float), observer[np.newaxis, :])
    received = np.flatnonzero(np.isfinite(dedopplerized[0]))
    if len(received) == 0:
        raise ValueError('no sound emitted along the track reaches the microphone inside the record')
    kept = slice(received[0], received[-1] + 1)
    return emission_times[kept], dedopplerized[0, kept]


def downsample_and_filter(time_, pressure, downsample_factor, highpass_cutoff):
    """Low-pass and decimate a signal, then high-pass it.

    Args:
        time_: Sample times, s, equally spaced.
        pressure: Pressure, Pa, one value per sample time.
        downsample_factor: Integer decimation factor.
        highpass_cutoff: High-pass cutoff, Hz.

    Returns:
        Tuple (time, pressure, fs): the decimated times (s), the filtered
        pressure (Pa) and the new sampling frequency (Hz).
    """
    # Low-pass before keeping every Nth sample: plain slicing folded everything
    # above the new Nyquist back into the band, +3 dB at 2-3 kHz and +6 dB at
    # 3-4 kHz on this record.
    if downsample_factor > 1:
        pressure = resample_poly(pressure, 1, downsample_factor)
        time_ = np.asarray(time_)[::downsample_factor][:len(pressure)]
    fs = int(np.round(1.0 / (time_[1] - time_[0])))
    return time_, apply_highpass_filter(pressure, highpass_cutoff, fs), fs


def setup_segmentation(n_samples, segment_fraction=0.01, overlap_factor=4, stride_factor=10):
    """Setup segment boundaries for processing long signals.
    
    Args:
        n_samples: Total number of samples in signal.
        segment_fraction: Fraction of signal for increment size (default: 0.01).
        overlap_factor: Multiplier for cutoff region size (default: 4).
        stride_factor: Multiplier for segment stride (default: 10).
        
    Returns:
        Tuple of (spacing, cutoff, increment) where spacing is list of segment boundaries.
    """
    increment = int(np.floor(segment_fraction * n_samples))
    cutoff = overlap_factor * increment
    spacing = list(range(cutoff, n_samples - cutoff, stride_factor * increment))
    spacing.append(n_samples - cutoff)  # Add final point
    
    total_samples = sum(spacing[i+1] - spacing[i] for i in range(len(spacing)-1))
    
    print(f"\nProcessing {len(spacing)-1} segments")
    print(f"Increment: {increment}, Cutoff: {cutoff}")
    print(f"Total output samples: {total_samples}")
    
    return spacing, cutoff, increment


def process_segment_serial(seg_idx, spacing, cutoff, xo, to, rpmo, orders, no_blade,
                          fs, bandwidth_of, p, solver, use_coupling,
                          total_spacing, total_start):
    """Process a single segment in serial mode.
    
    Args:
        seg_idx: Current segment index.
        spacing: List of segment boundary indices.
        cutoff: Number of samples to exclude at boundaries.
        xo: Original signal array, shape (n_samples, n_mics).
        to: Time vector.
        rpmo: Rotor speed array, rad/s, shape (n_samples, n_rotors).
        orders: Iterable of blade-passage-frequency harmonics (k x BPF) to extract.
        no_blade: Number of blades per rotor.
        fs: Sampling frequency in Hz.
        bandwidth_of: Function of (harmonic, frequency) giving its bandwidth,
            Hz (:func:`harmonic_bandwidth` with its rule fixed).
        p: VK filter order.
        solver: VKF solver (see vold_kalman_filter).
        use_coupling: Whether to include cross-order coupling.
        total_spacing: Total number of segments.
        total_start: Start time for progress tracking.
        
    Returns:
        Tuple of (separated_signals, time, rpm, original, seg_size).
    """
    seg_start = time.time()
    print(f"\nSegment {seg_idx+1}/{total_spacing}")

    # Extract segment with cutoff padding
    start_idx = spacing[seg_idx] - cutoff
    end_idx = spacing[seg_idx + 1] + cutoff

    orders = tuple(orders)
    print(f"  Processing {len(orders)} orders (all {rpmo.shape[1]} rotors per call)...")

    _, tmp_P_ordLoop, t_center, rpm_center, orig_center = _process_segment(
        seg_idx, xo[start_idx:end_idx, :], to[start_idx:end_idx], rpmo[start_idx:end_idx, :],
        orders, no_blade, fs, bandwidth_of, p, solver, use_coupling, cutoff)

    seg_time = time.time() - seg_start
    elapsed = time.time() - total_start
    avg_time = elapsed / (seg_idx + 1)
    remaining = avg_time * (total_spacing - 1 - seg_idx)
    print(f"  Segment time: {seg_time:.1f}s, Est. remaining: {remaining:.0f}s")

    return tmp_P_ordLoop, t_center, rpm_center, orig_center, tmp_P_ordLoop.shape[0]


def harmonic_bandwidth(order, freq, bandwidth, bw_percent, min_bw, max_bw):
    """Full -3 dB bandwidth, Hz, of one blade-passage harmonic.

    Args:
        order: The harmonic, k x BPF.
        freq: Its frequency, Hz, shape (n_samples, n_rotors).
        bandwidth: None, a number (Hz, every harmonic) or a mapping from
            harmonic to Hz; harmonics it leaves out take the rule below.
        bw_percent, min_bw, max_bw: The rule: bw_percent of the frequency,
            clipped to [min_bw, max_bw] Hz.

    Returns:
        A number, or an array shaped like ``freq``, Hz.
    """
    if isinstance(bandwidth, dict):
        if order in bandwidth:
            return float(bandwidth[order])
    elif bandwidth is not None:
        return float(bandwidth)
    return np.clip(freq * bw_percent, min_bw, max_bw)


def _process_segment(
    seg_idx,
    x_src,
    t,
    rpm,
    orders,
    no_blade,
    fs,
    bandwidth_of,
    p,
    solver,
    use_coupling,
    cutoff,
):
    """Process a single segment for parallel execution.
    
    Args:
        seg_idx: Segment index for result ordering.
        x_src: Source signal array, shape (n_samples, n_mics).
        t: Time vector for segment.
        rpm: Rotor speed array, rad/s, shape (n_samples, n_rotors).
        orders: Tuple of blade-passage-frequency harmonics (k x BPF) to extract.
        no_blade: Number of blades per rotor.
        fs: Sampling frequency in Hz.
        bandwidth_of: Function of (harmonic, frequency) giving its bandwidth, Hz.
        p: VK filter order.
        solver: VKF solver (see vold_kalman_filter).
        use_coupling: Whether to include cross-order coupling.
        cutoff: Number of samples to exclude at boundaries.
        
    Returns:
        Tuple of (seg_idx, separated_signals, time, rpm, original).
    """
    x = x_src.copy()
    seg_size = len(x) - 2 * cutoff
    n_mics = x.shape[1]

    n_rotors = rpm.shape[1]
    tmp_P_ordLoop = np.zeros((seg_size, n_rotors, n_mics))

    for order in orders:
        freq_all_rotors = rpm * order * no_blade / (2 * np.pi)
        bandwidth_all = bandwidth_of(order, freq_all_rotors)

        for mic in range(n_mics):
            y, ph, _ = vold_kalman_filter(
                x[:, mic],
                freq_all_rotors,
                fs,
                bandwidth_all,
                p,
                solver=solver,
                use_coupling=use_coupling,
            )

            components_full = np.real(y * ph)
            for rotor_idx in range(n_rotors):
                tmp_P_ordLoop[:, rotor_idx, mic] += components_full[cutoff:-cutoff, rotor_idx]
            x[:, mic] -= np.sum(components_full, axis=1)

    return (
        seg_idx,
        tmp_P_ordLoop,
        t[cutoff:-cutoff],
        rpm[cutoff:-cutoff, :],
        x_src[cutoff:-cutoff, :],
    )


def heard_rotor_speed(data, acoustics_T, receiver, sound_speed):
    """Rotor speeds as a microphone hears them, rad/s, at the acoustic times.

    Args:
        data: Dictionary from load_hexacopter_data, with track_time (s),
            track_position (n, 3) in m, aircraft_time (s) and aircraft_rpm (rad/s).
        acoustics_T: Acoustic time vector, s, on the track's clock.
        receiver: Microphone position (x, y, z), m, in the track's frame.
        sound_speed: Speed of sound, m/s.

    Returns:
        Array (len(acoustics_T), n_rotors), rad/s: each rotor's speed at the
        emission time times the Doppler factor.  Samples outside what the track
        can have emitted take the nearest predicted value, and emission times
        outside the RPM record take its first or last speed.
    """
    if 'track_position' not in data:
        raise ValueError('the record has no aircraft position, so the Doppler shift cannot be predicted')
    position = np.asarray(data['track_position'], dtype=float)
    track = {'time': data['track_time'], 'x': position[:, 0], 'y': position[:, 1], 'z': position[:, 2]}
    # At the emission times, held at the ends of the speed record: emission
    # precedes reception, and a cubic extrapolated over the delay runs away.
    speed = rotor_speed_function(data['aircraft_time'], data['aircraft_rpm'])
    heard = observed_order_track(acoustics_T, track, receiver, speed, 1, sound_speed,
                                 speed_unit='rad/s') * (2 * np.pi)
    known = np.isfinite(heard[:, 0])
    if not known.any():
        raise ValueError('no acoustic sample can have been emitted along the aircraft track')
    if not known.all():
        print(f"  {np.count_nonzero(~known)} samples lie outside the track; holding the nearest speed")
        for k in range(heard.shape[1]):
            heard[:, k] = np.interp(acoustics_T, acoustics_T[known], heard[known, k])
    return heard


def prepare_signal(data, mic, doppler=DOPPLER, receiver=None, sound_speed=SOUND_SPEED_M_S,
                   downsample_factor=DOWNSAMPLE_FACTOR, highpass_cutoff=HIGHPASS_CUTOFF_HZ):
    """One microphone's signal and the rotor speeds the filter follows in it.

    Args:
        data: Dictionary from load_hexacopter_data at the recorded rate.
        mic: Microphone, 1-based.
        doppler: 'dedopplerize' (resample onto emission time; rotor speeds at
            emission time), 'track' (the recorded signal; rotor speeds as the
            microphone hears them) or 'none' (the recorded signal; rotor speeds
            at the reception times).
        receiver: Microphone position (x, y, z), m, in place of the file's.
        sound_speed: Speed of sound, m/s.
        downsample_factor: Decimation applied before the separation.
        highpass_cutoff: High-pass cutoff, Hz.

    Returns:
        Tuple (time, pressure, speed, fs): times, s (emission time for
        'dedopplerize', reception time otherwise), pressure, Pa, rotor speeds,
        rad/s, shape (n, n_rotors), and the sampling frequency, Hz.
    """
    if doppler not in DOPPLER_MODES:
        raise ValueError(f"doppler must be one of {DOPPLER_MODES}, not {doppler!r}")
    n_mics = np.shape(data['acoustics_P'])[1]
    if not 1 <= mic <= n_mics:
        raise ValueError(f"mic must be between 1 and {n_mics}, not {mic}")
    if doppler == 'dedopplerize':
        time_, pressure = dedopplerize_record(data, mic, sound_speed, receiver)
    else:
        time_ = np.asarray(data['acoustics_T'], dtype=float)
        pressure = np.asarray(data['acoustics_P'])[:, mic - 1].astype(float)
    time_, pressure, fs = downsample_and_filter(time_, pressure, downsample_factor, highpass_cutoff)
    if doppler == 'track':
        position = microphone_position(data, mic, receiver)
        if position is None:
            raise ValueError(f"the record has no aircraft track or no position for microphone {mic}, "
                             "so the Doppler shift cannot be predicted")
        speed = heard_rotor_speed(data, time_, position, sound_speed)
    else:
        speed = interpolate_rpm_to_acoustic_time(data['aircraft_time'], data['aircraft_rpm'], time_)
    return time_, pressure, speed, fs


def separate_hexacopter_acoustics(mat_file_path, mic=MIC,
                                   orders=VKF_ORDERS, p=VKF_P,
                                   min_bw=VKF_MIN_BW, max_bw=VKF_MAX_BW, bw_percent=VKF_BW_PERCENT,
                                   solver=VKF_SOLVER, use_coupling=VKF_USE_COUPLING,
                                   n_jobs=None, downsample_factor=DOWNSAMPLE_FACTOR,
                                   highpass_cutoff=HIGHPASS_CUTOFF_HZ, doppler=DOPPLER, receiver=None,
                                   sound_speed=SOUND_SPEED_M_S, bandwidth=VKF_BANDWIDTH_HZ):
    """
    Separate hexacopter acoustic signals using Vold-Kalman filtering.
    
    Parameters
    ----------
    mat_file_path : str
        Path to .mat file containing hexacopter data
    mic : int, optional
        Microphone to process, 1-based, default 16
    orders : iterable, optional
        Blade-passage-frequency harmonics k x BPF to extract (order k is at
        k * no_blade times the shaft frequency), default 1-10 (VKF_ORDERS).
    p : int, optional
        VK filter order, default 2
    min_bw : float, optional
        Bandwidth floor in Hz, default 3, for harmonics ``bandwidth`` leaves out
    max_bw : float, optional
        Bandwidth ceiling in Hz, default 9, likewise
    bw_percent : float, optional
        Bandwidth as percentage of frequency, default 0.05 (5%), likewise
    solver : {"auto", "umfpack", "superlu"}, optional
        VKF solver (see vold_kalman_filter), default "auto": banded Cholesky,
        with the sparse solvers when it does not apply
    use_coupling : bool, optional
        Whether to include cross-order coupling (B_U) terms, default True
    n_jobs : int or None, optional
        Number of parallel workers for segment processing. Defaults to CPU count.
    downsample_factor : int, optional
        Decimation of the recorded signal before the separation, default 8.
    highpass_cutoff : float, optional
        High-pass cutoff, Hz, default 30.
    doppler : {'dedopplerize', 'track', 'none'}, optional
        How the Doppler shift is handled (see :func:`prepare_signal`), default
        'dedopplerize'.  The first two need the aircraft track, which must
        share the acoustic and RPM clock.
    receiver : array_like or None, optional
        Microphone position (x, y, z), m, in the frame of the file's aircraft
        position, in place of the file's position for ``mic``.
    sound_speed : float, optional
        Speed of sound, m/s, default 343.0.
    bandwidth : float, dict or None, optional
        Full -3 dB bandwidth of each harmonic, Hz: a number for all of them or
        a mapping from harmonic to Hz (default VKF_BANDWIDTH_HZ: 5 Hz for the
        fundamental, 3 Hz for harmonics 2-4, 2 Hz above).  Harmonics it leaves
        out, or all of them with None, take bw_percent of their frequency,
        clipped to [min_bw, max_bw].
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'P': separated pressures, shape (n_samples, n_rotors, 1)
        - 'T': time vector, s: emission time for 'dedopplerize', reception
          time otherwise
        - 'rpm': speed of each rotor, rad/s, shape (n_samples, n_rotors), as
          the filter followed it
        - 'original': signal before separation, shape (n_samples, 1)
        - 'fs': sampling frequency
    """
    data = load_hexacopter_data(mat_file_path)
    print(f"Processing microphone {mic} ({doppler})")
    to, xo, rpmo, fs = prepare_signal(data, mic, doppler=doppler, receiver=receiver, sound_speed=sound_speed,
                                      downsample_factor=downsample_factor, highpass_cutoff=highpass_cutoff)
    return separate_orders(to, xo[:, np.newaxis], rpmo, fs, orders=orders, p=p, min_bw=min_bw, max_bw=max_bw,
                           bw_percent=bw_percent, solver=solver, use_coupling=use_coupling,
                           n_jobs=n_jobs, bandwidth=bandwidth)


def separate_orders(to, xo, rpmo, fs, orders=VKF_ORDERS, p=VKF_P, min_bw=VKF_MIN_BW, max_bw=VKF_MAX_BW,
                    bw_percent=VKF_BW_PERCENT, solver=VKF_SOLVER, use_coupling=VKF_USE_COUPLING, n_jobs=None,
                    no_blade=NO_BLADE, bandwidth=VKF_BANDWIDTH_HZ):
    """Separate the rotor orders of signals already loaded and filtered.

    The segment loop of :func:`separate_hexacopter_acoustics`.

    Parameters
    ----------
    to : ndarray
        Time vector, s, shape (n_samples,).
    xo : ndarray
        Pressures, Pa, shape (n_samples, n_mics).
    rpmo : ndarray
        Speed of each rotor at ``to``, rad/s, shape (n_samples, n_rotors):
        the speeds whose orders the signals carry.
    fs : float
        Sampling frequency, Hz.
    orders, p, min_bw, max_bw, bw_percent, solver, use_coupling, n_jobs, bandwidth
        As for :func:`separate_hexacopter_acoustics`.
    no_blade : int, optional
        Blades per rotor, default 2.

    Returns
    -------
    dict
        As :func:`separate_hexacopter_acoustics` returns, with n_mics columns.
    """
    orders = tuple(orders)  # iterated once per segment, so not a one-shot generator
    # A partial of a module-level function, so it reaches the worker processes.
    bandwidth_of = functools.partial(harmonic_bandwidth, bandwidth=bandwidth, bw_percent=bw_percent,
                                     min_bw=min_bw, max_bw=max_bw)
    n_rotors = rpmo.shape[1]
    
    # Set up segmentation
    spacing, cutoff, increment = setup_segmentation(len(to))
    
    print(f"Extracting orders: {list(orders)}")
    
    # Pre-calculate total output size for efficiency
    total_samples = sum(spacing[i+1] - spacing[i] for i in range(len(spacing)-1))
    
    # Pre-allocate result arrays (much faster than appending lists)
    separated_P = np.zeros((total_samples, n_rotors, xo.shape[1]))
    separated_T = np.zeros(total_samples)
    separated_rpm = np.zeros((total_samples, n_rotors))
    separated_original = np.zeros((total_samples, xo.shape[1]))
    
    if n_jobs is None:
        n_jobs = os.cpu_count() or 1

    total_start = time.time()

    if n_jobs and n_jobs > 1:
        print(f"\nParallel processing with {n_jobs} workers")
        futures = []
        with cf.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for seg_idx in range(len(spacing) - 1):
                start_idx = spacing[seg_idx] - cutoff
                end_idx = spacing[seg_idx + 1] + cutoff

                x_src = xo[start_idx:end_idx, :]
                t = to[start_idx:end_idx]
                rpm = rpmo[start_idx:end_idx, :]

                futures.append(
                    executor.submit(
                        _process_segment,
                        seg_idx,
                        x_src,
                        t,
                        rpm,
                        orders,
                        no_blade,
                        fs,
                        bandwidth_of,
                        p,
                        solver,
                        use_coupling,
                        cutoff,
                    )
                )

            results = [None] * (len(spacing) - 1)
            for fut in cf.as_completed(futures):
                seg_idx, tmp_P, t_center, rpm_center, orig_center = fut.result()
                results[seg_idx] = (tmp_P, t_center, rpm_center, orig_center)

        # Assemble results in order
        output_pos = 0
        for seg_idx, (tmp_P, t_center, rpm_center, orig_center) in enumerate(results):
            seg_size = tmp_P.shape[0]
            separated_P[output_pos:output_pos+seg_size, :, :] = tmp_P
            separated_T[output_pos:output_pos+seg_size] = t_center
            separated_rpm[output_pos:output_pos+seg_size, :] = rpm_center
            separated_original[output_pos:output_pos+seg_size, :] = orig_center
            output_pos += seg_size
    else:
        # Serial processing
        output_pos = 0
        for seg_idx in range(len(spacing) - 1):
            tmp_P, t_center, rpm_center, orig_center, seg_size = process_segment_serial(
                seg_idx, spacing, cutoff, xo, to, rpmo, orders, no_blade,
                fs, bandwidth_of, p, solver, use_coupling,
                len(spacing) - 1, total_start
            )
            
            # Store results in pre-allocated arrays
            separated_P[output_pos:output_pos+seg_size, :, :] = tmp_P
            separated_T[output_pos:output_pos+seg_size] = t_center
            separated_rpm[output_pos:output_pos+seg_size, :] = rpm_center
            separated_original[output_pos:output_pos+seg_size, :] = orig_center
            output_pos += seg_size
    
    total_time = time.time() - total_start
    print(f"\nTotal processing time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # No concatenation needed - already stored in pre-allocated arrays!
    result = {
        'P': separated_P,
        'T': separated_T,
        'rpm': separated_rpm,
        'original': separated_original,
        'fs': fs
    }
    
    print(f"Final separated shape: {result['P'].shape}")
    print(f"Time vector shape: {result['T'].shape}")
    
    return result


def separation_residual(separated):
    """What the separation leaves of the first microphone's signal, Pa."""
    return separated['original'][:, 0] - separated['P'][:, :, 0].sum(axis=1)


def residual_levels(separated, bands=RESIDUAL_BANDS_HZ):
    """Energy the separation leaves behind, relative to the signal's, per band.

    Args:
        separated: Dictionary from separate_orders for one microphone.
        bands: (low, high) band edges, Hz.

    Returns:
        Array of residual-to-original energy ratios, dB, one per band; NaN
        for a band that reaches the Nyquist frequency.
    """
    original = separated['original'][:, 0]
    residual = separation_residual(separated)
    levels = []
    for low, high in bands:
        if high >= separated['fs'] / 2:
            levels.append(np.nan)
            continue
        sos = butter(4, [low, high], 'bandpass', fs=separated['fs'], output='sos')
        levels.append(10 * np.log10(np.sum(sosfiltfilt(sos, residual) ** 2)
                                    / np.sum(sosfiltfilt(sos, original) ** 2)))
    return np.array(levels)


def plot_results(
    separated,
    mic_plot=1,
    save_dir="demo_plots",
    filename_prefix="hexa",
    show_plots=True,
    zoom_time_window=(108, 108.2),
):
    """
    Plot the original, separated, and residual signals.
    
    Parameters
    ----------
    separated : dict
        Dictionary from separate_hexacopter_acoustics()
    mic_plot : int, optional
        Column of the separated result to plot (1-based), default 1
    save_dir : str or None, optional
        Directory for the four PNG files (300 dpi), created if needed;
        None writes nothing.  Default ``demo_plots``.
    filename_prefix : str, optional
        File name prefix, default ``hexa``.
    show_plots : bool, optional
        Also open the figures in windows.
    zoom_time_window : (float, float) or None, optional
        Time range of the zoomed figure, s.

    Returns
    -------
    list of str
        The paths written.
    """
    
    # Convert to 0-based index
    mic_idx = mic_plot - 1
    
    # Extract data
    T = separated['T']
    original = separated['original'][:, mic_idx]
    P_separated = separated['P'][:, :, mic_idx]  # (n_samples, n_rotors)
    rpm = separated['rpm']
    n_rotors = P_separated.shape[1]

    # Sum all rotor components
    summed_separated = np.sum(P_separated, axis=1)

    # Show time signals
    residual = original - summed_separated
    fig1 = plt.figure(figsize=(12, 6))
    plt.plot(T, original, linewidth=1.2, label='Original', alpha=0.7)
    plt.plot(T, summed_separated, linewidth=1.2, label='Separated (sum)', alpha=0.7)
    plt.plot(T, residual, linewidth=1.2, label='Residual', alpha=0.7)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.xlabel('Time, s')
    plt.ylabel('Acoustic Pressure, Pa')
    plt.tight_layout()

    # Zoomed time signals
    fig1_zoom = plt.figure(figsize=(12, 6))
    plt.plot(T, original, linewidth=1.2, label='Original', alpha=0.7)
    plt.plot(T, summed_separated, linewidth=1.2, label='Separated (sum)', alpha=0.7)
    plt.plot(T, residual, linewidth=1.2, label='Residual', alpha=0.7)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.xlabel('Time, s')
    plt.ylabel('Acoustic Pressure, Pa')
    if zoom_time_window is not None:
        plt.xlim(*zoom_time_window)
    plt.tight_layout()
    
    # Figure 2: Individual rotor components (offset for visibility)
    fig2 = plt.figure(figsize=(12, 8))
    offsets = [0.5 - 0.25 * k for k in range(n_rotors)]
    rotor_labels = [f'R{k + 1}' for k in range(n_rotors)]
    
    for rotor_idx in range(n_rotors):
        plt.plot(T, offsets[rotor_idx] + P_separated[:, rotor_idx], 
                linewidth=1.2, label=rotor_labels[rotor_idx], alpha=0.7)
    
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.xlabel('Time, s')
    plt.ylabel('Acoustic Pressure, Pa (offset)')
    plt.title('Separated Rotor Components')
    plt.tight_layout()
    
    # RPM Time History
    fig3 = plt.figure(figsize=(12, 6))
    for rotor_idx in range(n_rotors):
        plt.plot(T, 60.0*rpm[:, rotor_idx]/(2*np.pi), linewidth=1.2, 
                label=rotor_labels[rotor_idx], alpha=0.7)
    
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.xlabel('Time, s')
    plt.ylabel('Rotor Speed, RPM')
    plt.title('Rotor RPM Time History')
    plt.tight_layout()

    figures = {f"{filename_prefix}_original_separated": fig1,
               f"{filename_prefix}_original_separated_zoom": fig1_zoom,
               f"{filename_prefix}_rotor_components": fig2,
               f"{filename_prefix}_rpm_time_history": fig3}
    if save_dir or show_plots:
        return save_all_or_show(figures, save_dir, show_plots, dpi=300)
    return []


def plot_spectrograms(signals, rotor_time, rotor_speed, rotor=SPECTROGRAM_ROTOR, band=SPECTROGRAM_BAND_HZ):
    """Spectrograms with one rotor's blade-passage harmonics, at its own
    speed, drawn over each.

    Args:
        signals: List of (title, time (s), pressure (Pa), fs (Hz)).
        rotor_time: Rotor-speed times, s.
        rotor_speed: Rotor speeds, rad/s, shape (n, n_rotors).
        rotor: Rotor whose harmonics are drawn, 1-based.
        band: Frequency range shown, Hz.

    Returns:
        The figure.
    """
    # The harmonics drawn are averaged over the spectrogram's window.
    step = np.median(np.diff(rotor_time))
    width = max(1, int(round(SPECTROGRAM_WINDOW_S / step)))
    shaft_hz = uniform_filter1d(rotor_speed[:, rotor - 1], width, mode='nearest') / (2 * np.pi)
    harmonics = range(int(np.ceil(band[0] / (NO_BLADE * shaft_hz.max()))),
                      int(np.floor(band[1] / (NO_BLADE * shaft_hz.min()))) + 1)
    fig, axes = plt.subplots(len(signals), 1, figsize=(12, 4.5 * len(signals)), sharey=True, squeeze=False)
    for ax, (title, time_, pressure, fs) in zip(axes[:, 0], signals):
        window = int(SPECTROGRAM_WINDOW_S * fs)
        f, t, power = spectrogram(pressure, fs, nperseg=window, noverlap=window - int(0.05 * fs), nfft=int(fs))
        inside = (f >= band[0]) & (f <= band[1])
        level = 10 * np.log10(power[inside] / P_REF ** 2)
        mesh = ax.pcolormesh(time_[0] + t, f[inside], level, shading='auto', cmap='viridis',
                             vmin=np.percentile(level, 5), vmax=np.percentile(level, 99.5))
        for k in harmonics:
            ax.plot(rotor_time, k * NO_BLADE * shaft_hz, 'w--', linewidth=0.8, alpha=0.8)
        ax.set_xlim(time_[0], time_[-1])
        ax.set_ylim(*band)
        ax.set_title(f"{title}; dashed: R{rotor} blade-passage harmonics at its own speed")
        ax.set_xlabel('Time, s')
        ax.set_ylabel('Frequency, Hz')
        fig.colorbar(mesh, ax=ax, label=PSD_LABEL)
    fig.tight_layout()
    return fig


def plot_residual_spectrum(separated, title, band=(50, 1000)):
    """Power spectrum of the signal and of what the separation leaves behind.

    Args:
        separated: Dictionary from separate_orders for one microphone.
        title: Figure title.
        band: Frequency range shown, Hz.

    Returns:
        The figure.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    for signal, label in ((separated['original'][:, 0], 'Signal'),
                          (separation_residual(separated), 'Residual after separation')):
        f, psd_db, _, _ = psd_welch(signal, separated['fs'], window_time=1.0, passband=band)
        ax.plot(f, psd_db, linewidth=1.0, label=label)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('Frequency, Hz')
    ax.set_ylabel(PSD_LABEL)
    fig.tight_layout()
    return fig



def main(argv=None):
    """Run the separation with the settings at the top of this file, or the
    command line's, print the residual energy and plot the results."""
    parser = argparse.ArgumentParser(
        description='De-Dopplerize one microphone of the bundled hexacopter flyover and separate its rotor '
                    'harmonics with the Vold-Kalman filter; plot the original, separated and residual '
                    'signals, the rotor speeds, the spectrograms and the residual spectrum.')
    parser.add_argument('--mat-file', default=MAT_FILE,
                        help='MATLAB record (.mat) with acoustics, aircraft and info structs (default: the '
                             'bundled example_data/F100_20_P#1[FA1FZC-3.h5].mat of the repository)')
    parser.add_argument('--mic', '--mic-range', dest='mic', type=int, default=MIC,
                        help=f'microphone to separate, 1-based (default {MIC}, beneath the flight path)')
    parser.add_argument('--doppler', choices=DOPPLER_MODES, default=DOPPLER,
                        help="'dedopplerize' resamples the signal onto emission time; 'track' filters the "
                             "signal as recorded at the speeds the microphone hears; 'none' filters it as "
                             f"recorded at the rotors' speeds (default {DOPPLER})")
    parser.add_argument('--receiver', type=float, nargs=3, default=None, metavar=('X', 'Y', 'Z'),
                        help='microphone position, m, in the aircraft position\'s frame, in place of the '
                             'file\'s (default: the file\'s)')
    parser.add_argument('--jobs', type=int, default=VKF_N_JOBS,
                        help='worker processes for the segments (default: the CPU count)')
    # Figures go to demo_plots/ in the current directory, as hexa_*.png.
    add_output_options(parser, 'demo_plots', directory=True)
    args = parser.parse_args(argv)

    if not os.path.exists(args.mat_file):
        print(f"Error: Data file not found: {args.mat_file}")
        print("Please ensure the hexacopter data file is in the correct location.")
        return 1

    print("="*70)
    print("Hexacopter Acoustic Signal Separation")
    print("Using Vold-Kalman Filter")
    print("="*70)

    data = load_hexacopter_data(args.mat_file)
    receiver = None if args.receiver is None else tuple(args.receiver)
    prepare = functools.partial(prepare_signal, data, args.mic, receiver=receiver, sound_speed=SOUND_SPEED_M_S,
                                downsample_factor=DOWNSAMPLE_FACTOR, highpass_cutoff=HIGHPASS_CUTOFF_HZ)
    try:
        to, xo, rpmo, fs = prepare(doppler=args.doppler)
    except ValueError as error:
        print(f"Error: {error}")
        return 1
    if 'mic_type' in data and data['mic_type'][args.mic - 1] == 'pole':
        print(f"Note: microphone {args.mic} is an elevated pole microphone; its spectrum carries "
              "ground-reflection interference.")
    separated = separate_orders(to, xo[:, np.newaxis], rpmo, fs, orders=VKF_ORDERS, p=VKF_P,
                                min_bw=VKF_MIN_BW, max_bw=VKF_MAX_BW, bw_percent=VKF_BW_PERCENT,
                                solver=VKF_SOLVER, use_coupling=VKF_USE_COUPLING, n_jobs=args.jobs,
                                bandwidth=VKF_BANDWIDTH_HZ)

    print("\n" + "="*70)
    print("Processing complete!")
    print("="*70)
    print(f"\nEnergy left after the separation, relative to the signal, microphone {args.mic} ({args.doppler}):")
    for (low, high), level in zip(RESIDUAL_BANDS_HZ, residual_levels(separated)):
        print(f"{f'{low}-{high} Hz':>14s}  {level:6.1f} dB")

    show = args.plot or PLOT_INTERACTIVE
    plot_results(separated, mic_plot=1, save_dir=args.output_dir, show_plots=False,
                 zoom_time_window=ZOOM_TIME_WINDOW)
    figures = {'hexa_residual_spectrum': plot_residual_spectrum(separated, TITLES[args.doppler])}
    rotor_speed = data['aircraft_rpm']
    if microphone_position(data, args.mic, receiver) is not None and rotor_speed.shape[1] >= SPECTROGRAM_ROTOR:
        signals = []
        for mode in ('none', 'dedopplerize'):
            if mode == args.doppler:
                t, x, f = separated['T'], separated['original'][:, 0], fs
            else:
                t, x, _, f = prepare(doppler=mode)
            signals.append((TITLES[mode], t, x, f))
        figures['hexa_spectrograms'] = plot_spectrograms(signals, data['aircraft_time'], rotor_speed)
    save_all_or_show(figures, args.output_dir, show, dpi=300)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
