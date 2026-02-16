# coding=UTF-8
import os
from configparser import ConfigParser
from glob import glob

import numpy as np
from panam_acoustics.atmosphere import Atmosphere
import h5py
from typing import Any, Optional, cast
import openpyxl
import scipy.signal
from scipy.interpolate import RegularGridInterpolator
from scipy.special import erf
import simplekml
# Colormap helper will import palettable lazily
import matplotlib
from matplotlib import cm, tri
from matplotlib.pyplot import plot, subplots, colorbar, style, contourf, show
from netCDF4 import Dataset
from pyuff import UFF
from pymap3d import geodetic2enu, enu2geodetic

import unit_conversion
from panam_acoustics import filters as pa_filters

style.use('fivethirtyeight')
matplotlib.rcParams.update({'mathtext.fontset': 'dejavuserif'})
matplotlib.rcParams.update({'figure.autolayout': True})


def psd(signal, sampling_rate, cal=0.0):
    """
    Compute the acoustic power spectral density of a signal

    Args:
        signal: Array-like acoustic signal
        sampling_rate: Sampling rate of signal, Hz
        cal: Optional calibration factor to apply to signal (dB)
    Returns: tuple (frequency, psd_db, level)
           WHERE
           frequency is an array of band frequencies
           psd_db is the power spectral density in dB/Hz**2
           level is the integrated sound pressure level over all bands in dB
    """
    kcal = 10 ** (cal / 20)
    frequency, power_spectral_density = scipy.signal.periodogram(kcal * signal, sampling_rate)
    df = frequency[1] - frequency[0]
    pref = 2.0e-5
    psd_db = 10.0 * np.log10(power_spectral_density / (pref ** 2))
    level = 20.0 * np.log10(np.sqrt(np.sum(power_spectral_density * df)) / pref)
    return frequency, psd_db, level


def psd_welch(signal, sampling_rate, cal=0.0, window_time=1.0, window_type='hann', window_overlap=0.5, medfilter=None, passband = None):
    """
    Compute the acoustic power spectral density of a signal

    Args:
        signal: Array-like acoustic signal
        sampling_rate: Sampling rate of signal, Hz
        cal: Optional calibration factor to apply to signal (dB)
        window_time: duration of windows, s
        window_type: optional type of window to us, see scipy.signal.window, default="hann"
        window_overlap: optional proportion of overlap for windows, default=0.5
        medfilter: width (in Hz) of the median filter, if None, no filter applied (default)
        passband: pair of values (in Hz) defining the lower and upper regions of the passband.  Default None.
    Returns: tuple (frequency, psd_db, level)
           WHERE
           frequency is an array of band frequencies
           psd_db is the power spectral density in dB/Hz**2
           level is the integrated sound pressure level over all bands in dB
           level_A is the A-weighted integrated sound pressure level over all bands in dB
    """
    kcal = 10 ** (cal / 20)
    binwidth = int(2.0 ** nextpow2(window_time * sampling_rate))
    window = scipy.signal.get_window(window_type, binwidth)
    frequency, power_spectral_density = scipy.signal.welch(kcal * signal, sampling_rate,
                                                           window=window, noverlap=int(binwidth * window_overlap))
    if passband is not None:
        pass_indicies = np.logical_and(frequency >= passband[0], frequency <= passband[1])
        power_spectral_density = power_spectral_density[pass_indicies]
        frequency = frequency[pass_indicies]
    df = frequency[1] - frequency[0]
    if medfilter is not None:
        medfilter_width = int(np.ceil(medfilter) // 2 * 2 + 1)
        power_spectral_density = scipy.signal.medfilt(power_spectral_density, medfilter_width)
    pref = 2.0e-5
    psd_db = 10.0 * np.log10(power_spectral_density / (pref ** 2))
    level = 20.0 * np.log10(np.sqrt(np.sum(power_spectral_density * df)) / pref)
    weight = 10**(dBAw(frequency) / 2)
    level_A = 20.0 * np.log10(np.sqrt(np.sum(weight * power_spectral_density * df)) / pref)
    return frequency, psd_db, level, level_A


def overall_SPL(signal, sampling_rate):
    """
    Computed A-weighted and unweighted sound pressure levels
    Args:
        signal: pressure time history signal, Pa
        sampling_rate: sampling rate of signal, Hz

    Returns:
    tuple (A-weighted Level, Unweighted Level)
    """
    f, spl, levelZ = psd(signal, sampling_rate)
    weight = dBAw(f)
    df = f[1] - f[0]
    levelA = 10.0 * np.log10(df * np.sum(10.0 ** ((spl + weight) / 10.0)))
    return levelA, levelZ


def level_history(signal, sampling_rate, period=1.0):
    """
    Compute time history of SPL
    Args:
        signal: pressure time history signal, Pa
        sampling_rate: sampling rate of signal, Hz
        period: integration time for SPL calculations, sec

    Returns:
        tuple (time, A-weighted level, Unweighted level)
    """
    binwidth = period * sampling_rate
    edges = np.arange(0, len(signal), binwidth)

    time = edges[0:-1] / sampling_rate
    level_a = np.zeros_like(time)
    level_z = np.zeros_like(time)
    for i in range(0, len(edges) - 1, 1):
        level_a[i], level_z[i] = overall_SPL(signal[int(edges[i]):int(edges[i + 1])], sampling_rate)


def nextpow2(x):
    """
    Compute the smallest integer N for x <= 2**N
    Args:
        x: number or array-like of values for which to compute N

    Returns:
        number or array like of integral N
    """
    return np.ceil(np.log2(np.abs(x)))

def third_octave_band_levels(signal, sampling_rate, cal=0.0, fmin=20.0, fmax=20000.0):
    """
    Compute third-octave band levels of a signal
    Args:
        signal: Array-like acoustic signal
        sampling_rate: Sampling rate of signal, Hz
        cal: Optional calibration factor to apply to signal (dB)
        fmin: minimum frequency for third-octave bands, Hz
        fmax: maximum frequency for third-octave bands, Hz

    Returns: tuple (band_centers, band_levels)
           WHERE
           band_centers is an array of third-octave band center frequencies
           band_levels is an array of third-octave band levels in dB
    """
    # Define third-octave band center frequencies
    k = np.arange(-50, 50)
    band_centers = 1000.0 * (2.0 ** (k / 3.0))
    band_centers = band_centers[np.logical_and(band_centers >= fmin, band_centers <= fmax)]
    # Compute PSD
    frequency, psd_db, _ = psd(signal, sampling_rate, cal)
    pref = 2.0e-5
    psd_linear = (pref ** 2) * 10.0 ** (psd_db / 10.0)
    df = frequency[1] - frequency[0]
    band_levels = np.zeros_like(band_centers)
    for i, fc in enumerate(band_centers):
        f_lower = fc / (2.0 ** (1.0 / 6.0))
        f_upper = fc * (2.0 ** (1.0 / 6.0))
        band_indices = np.where(np.logical_and(frequency >= f_lower, frequency < f_upper))
        band_power = np.sum(psd_linear[band_indices] * df)
        band_levels[i] = 10.0 * np.log10(band_power / (pref ** 2))
    return band_centers, band_levels


def load_mil_std_1474e_table_c1(filename):
    """
    Load MIL-STD-1474E Table C-1 data from CSV.

    Args:
        filename: CSV file path containing the converted Table C-1 data.

    Returns:
        dict with keys:
            band_freq_hz: third-octave center frequencies in the table
            distance_columns_m: nondetectability distance columns (m)
            measurement_distance_m: measurement distance used for each column (m)
            limits_db: table limits (rows=freq, cols=distance column), NaN for missing values
    """
    rows = np.genfromtxt(filename, delimiter=',', dtype=str)
    if rows.ndim != 2 or rows.shape[0] < 3:
        raise ValueError('MIL-STD table CSV appears malformed')

    distance_columns_m = np.array([float(v) for v in rows[1, 1:] if v != ''], dtype=float)

    measurement_row_idx = None
    for i in range(rows.shape[0]):
        if rows[i, 0].strip().lower() == 'measurement distance (meters)':
            measurement_row_idx = i
            break
    if measurement_row_idx is None:
        raise ValueError('Could not find "Measurement Distance (meters)" row in MIL-STD Table C-1 CSV')

    measurement_distance_m = np.array([float(v) for v in rows[measurement_row_idx, 1:] if v != ''], dtype=float)

    band_rows = rows[2:measurement_row_idx, :]
    band_freq_hz = np.array([float(r[0]) for r in band_rows], dtype=float)

    limits_db = np.zeros((band_rows.shape[0], distance_columns_m.size), dtype=float)
    limits_db[:] = np.nan
    for i, row in enumerate(band_rows):
        for j in range(distance_columns_m.size):
            value = row[j + 1].strip()
            if value == '' or value.upper() == 'NA':
                continue
            limits_db[i, j] = float(value)

    return {
        'band_freq_hz': band_freq_hz,
        'distance_columns_m': distance_columns_m,
        'measurement_distance_m': measurement_distance_m,
        'limits_db': limits_db,
    }


def _mil_std_interp_log_frequency(x_freq_hz, y_values, target_freq_hz):
    return np.interp(np.log10(target_freq_hz), np.log10(x_freq_hz), y_values)


def _mil_std_distance_within_group(measured_level_db, thresholds_db, distances_m):
    valid = np.isfinite(thresholds_db)
    if np.count_nonzero(valid) == 0:
        return np.nan, 'warning_insufficient_group_data', False

    thresholds = thresholds_db[valid]
    distances = distances_m[valid]
    order = np.argsort(distances)
    thresholds = thresholds[order]
    distances = distances[order]

    if measured_level_db <= thresholds[0]:
        return distances[0], 'warning_group_below_first', False

    for i in range(distances.size - 1):
        level_1 = thresholds[i]
        level_2 = thresholds[i + 1]
        distance_1 = distances[i]
        distance_2 = distances[i + 1]

        if (measured_level_db - level_1) * (measured_level_db - level_2) > 0.0:
            continue

        if level_2 == level_1:
            return distance_1, 'warning_group_flat_segment', False

        log_distance = np.interp(
            measured_level_db,
            [level_1, level_2],
            [np.log10(distance_1), np.log10(distance_2)]
        )
        return 10.0 ** log_distance, 'group_interpolated', True

    return np.nan, 'warning_group_above_last', False


def _mil_std_distance_from_table(measured_level_db_10m, threshold_levels_db, distances_m, measurement_distance_m):
    group_measurement_distances = np.unique(measurement_distance_m)
    candidates = []

    for group_distance_m in group_measurement_distances:
        group_mask = measurement_distance_m == group_distance_m
        if not np.any(group_mask):
            continue

        measured_level_group_db = measured_level_db_10m + 20.0 * np.log10(10.0 / group_distance_m)
        distance_m, status, is_valid = _mil_std_distance_within_group(
            measured_level_group_db,
            threshold_levels_db[group_mask],
            distances_m[group_mask],
        )

        if np.isfinite(distance_m):
            candidates.append((distance_m, f'{status}@{group_distance_m:.0f}m', is_valid))

    if len(candidates) == 0:
        return np.nan, 'warning_above_all_groups', False

    min_idx = int(np.argmin([c[0] for c in candidates]))
    return candidates[min_idx]


def mil_std_1474e_nondetectability_distance(
        band_centers_hz,
        band_levels_db,
        mil_std_table,
        spectrum_distance_m=10.0):
    """
    Compute MIL-STD-1474E Table C-1 nondetectability distance from a third-octave spectrum.

    Args:
        band_centers_hz: third-octave center frequencies for the spectrum.
        band_levels_db: third-octave levels (dB) at spectrum_distance_m.
        mil_std_table: dict from load_mil_std_1474e_table_c1(...) or CSV filename.
        spectrum_distance_m: distance corresponding to band_levels_db (m).

    Returns:
        dict with:
            overall_nondetectability_distance_m: overall distance using valid bands only
            trigger_frequency_hz: band frequency that sets the overall distance
            band_frequency_hz: MIL-STD band frequencies used in evaluation
            band_nondetectability_distance_m: per-band nondetectability distance
            band_status: per-band status strings
            band_valid: per-band boolean validity (True only for interpolated values)
            band_level_db_10m: input spectrum mapped to MIL-STD bands and normalized to 10 m
    """
    if isinstance(mil_std_table, (str, os.PathLike)):
        table = load_mil_std_1474e_table_c1(mil_std_table)
    else:
        table = mil_std_table

    band_centers_hz = np.asarray(band_centers_hz, dtype=float)
    band_levels_db = np.asarray(band_levels_db, dtype=float)

    table_band_freq_hz = np.asarray(table['band_freq_hz'], dtype=float)
    distance_columns_m = np.asarray(table['distance_columns_m'], dtype=float)
    measurement_distance_m = np.asarray(table['measurement_distance_m'], dtype=float)
    limits_db = np.asarray(table['limits_db'], dtype=float)

    if band_centers_hz.size == 0 or band_levels_db.size == 0:
        raise ValueError('Input spectrum is empty')

    band_levels_table = _mil_std_interp_log_frequency(
        band_centers_hz,
        band_levels_db,
        table_band_freq_hz,
    )
    band_levels_10m = band_levels_table + 20.0 * np.log10(spectrum_distance_m / 10.0)

    band_distances = np.zeros_like(table_band_freq_hz)
    band_status = np.empty(table_band_freq_hz.size, dtype=object)
    band_valid = np.zeros(table_band_freq_hz.size, dtype=bool)

    for i in range(table_band_freq_hz.size):
        distance_m, status, is_valid = _mil_std_distance_from_table(
            band_levels_10m[i],
            limits_db[i, :],
            distance_columns_m,
            measurement_distance_m,
        )
        band_distances[i] = distance_m
        band_status[i] = status
        band_valid[i] = is_valid

    valid_for_overall = np.logical_and(np.isfinite(band_distances), band_valid)
    if np.any(valid_for_overall):
        valid_indices = np.where(valid_for_overall)[0]
        best_local_idx = int(np.argmax(band_distances[valid_for_overall]))
        trigger_idx = valid_indices[best_local_idx]
        overall_distance = float(band_distances[trigger_idx])
        trigger_frequency = float(table_band_freq_hz[trigger_idx])
    else:
        overall_distance = np.nan
        trigger_frequency = np.nan

    return {
        'overall_nondetectability_distance_m': overall_distance,
        'trigger_frequency_hz': trigger_frequency,
        'band_frequency_hz': table_band_freq_hz,
        'band_nondetectability_distance_m': band_distances,
        'band_status': band_status,
        'band_valid': band_valid,
        'band_level_db_10m': band_levels_10m,
    }

def spectrogram(signal, sampling_rate, window_time=0.5, window_type="hann", window_overlap=7.0 / 8.0,
                detrend='constant', dbref=20e-6):
    """
    Computes the spectrogram of a signal
    Args:
        signal: array-like containing acoustic signal
        sampling_rate: sampling rate of signal, Hz
        window_time: duration of windows, s
        window_type: optional type of window to us, see scipy.signal.window, default="hann"
        window_overlap: optional proportion of overlap for windows, default=7.0/8.0
        detrend: optional detrending mode for signal, see scipy.signal.detrend
        dbref: optional reference value for calculating decibels, default 20e-6

    Returns: tuple (f, t, SPL)
    WHERE
    f is an array of frequencies
    t is an array of times
    SPL is a frequency x time power spectral density spectrogram, dB/Hz**2
    """
    # Pick next power of two that captures the window time, and generate the window
    binwidth = int(2.0 ** nextpow2(window_time * sampling_rate))
    window = scipy.signal.get_window(window_type, binwidth)
    # Calculate the spectrogram as a PSD
    f, t, Sxx = scipy.signal.spectrogram(signal, sampling_rate, window, noverlap=round(window_overlap * binwidth),
                                         detrend=detrend, mode='psd')
    # Convert to SPL
    SPL = 10.0 * np.log10(Sxx / (dbref ** 2))
    return f, t, SPL


def plot_spectrogram(signal, sampling_rate, window_time=1.0, window_type="hann", window_overlap=7.0 / 8.0,
                     detrend='constant', dbref=20e-6, save_name=None, title=None, time0=0, clim=None, flim=None):
    """
    Plot a spectrogram of signal
    Args:
        signal: array-like containing acoustic signal
        sampling_rate: sampling rate of signal, Hz
        window_time: duration of windows, s
        window_type: optional type of window to us, see scipy.signal.window, default="hann"
        window_overlap: optional proportion of overlap for windows, default=7.0/8.0
        detrend: optional detrending mode for signal, see scipy.signal.detrend
        dbref: optional reference value for calculating decibels, default 20e-6
        save_name: optional path to image file to save figure, default None
        title: optional title for the plot, default None
        time0: optional start time for signal, default 0
        clim: optional level scale color limits, default None
        flim: optional frequency scale limits, default None

    Returns: tuple (fig, ax, cs)
    WHERE
    fig is a matplotlib handle to the figure
    ax is a matplotlib handle to the plot axis
    cs is a matplotlib handle to the contour quadmesh
    """
    f, t, SPL = spectrogram(signal, sampling_rate, window_time, window_type, window_overlap, detrend, dbref)
    if flim is not None:
        fidx = np.logical_and(f >= flim[0], f <= flim[1])
        f = f[fidx]
        SPL = SPL[fidx, :]

    fig, ax = subplots(facecolor='white')
    if clim is None:
        vmin = None
        vmax = None
    else:
        vmin = clim[0]
        vmax = clim[1]
    cs = ax.pcolormesh(t + time0, f, SPL, vmin=vmin, vmax=vmax)
    cb = colorbar(cs, format='%.0f')
    ax.set_ylabel('Frequency, Hz')
    ax.set_xlabel('Time, s')
    cb.set_label('Power Spectral Density, dB')
    if title is not None:
        ax.set_title(title)
    if save_name is not None:
        fig.savefig(os.path.abspath(os.path.expanduser(save_name)))
    return fig, ax, cs


def dedopplerize(time, pressure, speed_of_sound, track_time, position, observers,
                 radius=None, output_sample_rate=None, time_offset=0.0):
    """
    Time domain de-Dopplerization of an acoustic signal
    Args:
        time: observers x time points matrix of observer times
        pressure: observers x time point matrix of acoustic pressures
        speed_of_sound: speed of sound
        track_time: array of source times
        position: track times x 3 matrix of source positions
        observers: observers x 3 matrix of observer positions
        radius: optional virtual observer radius
        output_sample_rate: optional source time sample rate

    Returns: tuple (emission_times, dpres)
    WHERE
    emission_times is an array emission times
    dpres is an observer x emission times matrix of de-Dopplerized acoustic pressures
    """
    # If not set, infer sample rate from measured data
    if output_sample_rate is None:
        output_sample_rate = 1 / (time[0, 1] - time[0, 0])
    # Generate emission time array
    emission_times = np.arange(track_time[0], track_time[-1], 1 / output_sample_rate)
    number_of_microphones = observers.shape[0]
    number_of_times = emission_times.shape[0]
    dpres = np.zeros((number_of_microphones, number_of_times))
    # For each observer, de-Dopplerize
    for index, observer in enumerate(observers):
        rx = observer[0] - position[:, 0]
        ry = observer[1] - position[:, 1]
        rz = observer[2] - position[:, 2]
        r = np.interp(emission_times, track_time, np.sqrt(rx ** 2 + ry ** 2 + rz ** 2))
        dpres[index, :] = np.interp(emission_times + r / speed_of_sound, time[index, :], pressure[index, :])
        # Apply spherical spreading when radius is known
        if radius is not None:
            dpres[index, :] = r / radius * dpres[index, :]
    # Adjust time of emission to virtual observer observation time when radius is known
    if radius is not None:
        emission_times = emission_times + radius / speed_of_sound
    return emission_times, dpres


def linear_array_plan(nmics, altitude, min_elevation=10.0, target_elv=90.0):
    """
    Design a linear microphone array
    Args:
        nmics: number of microphones
        altitude: flight altitude of the vehicle
        min_elevation: desired sideline elevation angle relative to horizon
        target_elv: target elevation for equal angle sideline spacing

    Returns: array of microphone sideline locations
    """
    # Slant distance to target elevation center line position
    r = altitude / np.sin(np.radians(target_elv))
    # Now, determine spacing for equal angular resolution on target elv plane
    angles = np.linspace(-90. + min_elevation, 90.0 - min_elevation, nmics)
    return r * np.tan(np.radians(angles))


def hemigen(time, source, velocity, observers, speed_of_sound):
    """
    Compute the time of emission observer angles for a moving source
    Args:
        time: Array of source times
        source: time x 3 matrix of source positions
        velocity: time x 3 matrix of source velocities
        observers: number of mics x 3 matrix of observer positions
        speed_of_sound: speed of sound

    Returns: tuple (azimuth, elevation, r, t_observer, mach_r)
    WHERE
    azimith are the azimuth angles on the sphere for each emission time
    elevation are the elevation angles on the sphere for each emission time
    r are the propagation distances for each emission time
    t_observer are the observer times associated with each emission time
    mach_r is the Mach number of the source in the propagation direction
    """
    number_of_mics = observers.shape[0]
    number_of_times = time.shape[0]
    # Expand data into number_of_times x number_of_mics arrays
    t = (time * np.ones((number_of_mics, number_of_times))).transpose()
    sx = (source[:, 0] * np.ones((number_of_mics, number_of_times))).transpose()
    sy = (source[:, 1] * np.ones((number_of_mics, number_of_times))).transpose()
    sz = (source[:, 2] * np.ones((number_of_mics, number_of_times))).transpose()
    vx = (velocity[:, 0] * np.ones((number_of_mics, number_of_times))).transpose()
    vy = (velocity[:, 1] * np.ones((number_of_mics, number_of_times))).transpose()
    vz = (velocity[:, 2] * np.ones((number_of_mics, number_of_times))).transpose()
    ox = observers[:, 0] * np.ones((number_of_times, number_of_mics))
    oy = observers[:, 1] * np.ones((number_of_times, number_of_mics))
    oz = observers[:, 2] * np.ones((number_of_times, number_of_mics))
    # Propagation vectors
    rx = ox - sx
    ry = oy - sy
    rz = oz - sz
    r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
    # Observation time
    t_observer = t + r / speed_of_sound
    # Mach number along the radiation direction
    mach_r = (vx * rx / r + vy * ry / r + vz * rz / r) / speed_of_sound
    # Compute elevation relative to horizon
    ground_range = np.sqrt(rx ** 2 + ry ** 2)
    height = -rz
    elevation = np.degrees(np.arctan2(height, ground_range))
    # Compute azimuth as difference between aircraft heading and observer bearing
    bearing = np.degrees(np.arctan2(ry, rx))
    heading = np.degrees(np.arctan2(vy, vx))
    azimuth = np.remainder(bearing - (heading + 180), 360)
    # If the elevation exceeds 90 degrees, it should be flipped over to the other
    # side of the sphere
    flip = elevation > 90
    elevation[flip] = 180 - elevation[flip]
    # Then correct azimuths for these points
    right_side = np.logical_and(flip, azimuth <= 180)
    left_side = np.logical_and(flip, azimuth > 180)
    azimuth[right_side] = azimuth[right_side] + 180
    azimuth[left_side] = azimuth[left_side] - 180
    return azimuth, elevation, r, t_observer, mach_r


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
        narrowband=False,
        narrowband_stride=1,
):
    """Generate an acoustic hemisphere from microphone time series and vehicle tracking data.

    This is the "normal" processing flow used by the demo scripts: use the vehicle kinematics
    to compute emission-time geometry (azimuth/elevation/range) via :func:`hemigen`, sample
    the measured PSD at the corresponding observer times, depropagate to a reference radius,
    then interpolate onto a regular azimuth/elevation grid using :func:`shepIDW`.

    Notes on conventions
    --------------------
    - Inputs must be in a *consistent* local Cartesian frame.
    - If the result looks mirrored left/right relative to a reference hemisphere, that is
      usually a sign convention mismatch in the lateral axis. Set ``flip_y_for_geometry=True``
      to apply Y -> -Y to geometry inputs *before* computing azimuth/elevation.

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

    Returns:
        dict with keys:
            'azi_grid_deg', 'elv_grid_deg',
            'oaspl_db' (2D grid, unweighted),
            'spl_a_db' (2D grid, A-weighted overall level in dBA),
            and optionally:
                        - when return_scattered=True: 'scattered' (dict of scattered az/el/power samples)
                            If third_octave/narrowband are enabled, the corresponding scattered data are
                            also included under 'scattered'.
            - when third_octave=True: 'third_octave' (band centers + band grids)
            - when narrowband=True: 'narrowband' (frequency + PSD grids)
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
        dt = np.gradient(track_time)
        track_velocity = np.gradient(track_position, axis=0) / dt[:, None]
    track_velocity = np.asarray(track_velocity, dtype=float)
    if track_velocity.shape != (track_time.size, 3):
        raise ValueError('track_velocity must be shape (Nt, 3) matching track_time')

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
    else:
        mic_geom = mic_locations
        pos_geom = track_position
        vel_geom = track_velocity

    # Emission-time geometry
    az_deg, el_deg, r_geom, t_obs, _ = hemigen(track_time, pos_geom, vel_geom, mic_geom, speed_of_sound)

    # Subsample emission-time points
    point_stride = int(point_stride)
    if point_stride < 1:
        raise ValueError('point_stride must be >= 1')
    tidx = np.arange(0, track_time.size, point_stride)

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
    oaspl_power_list = []
    spl_a_power_list = []
    if third_octave:
        if third_octave_band_centers_hz is not None:
            band_centers = np.asarray(third_octave_band_centers_hz, dtype=float).ravel()
            band_centers = band_centers[np.isfinite(band_centers)]
            if band_centers.size < 1:
                raise ValueError('third_octave_band_centers_hz must contain at least one finite frequency')
            band_centers = np.unique(band_centers)
            band_centers = band_centers[np.logical_and(band_centers >= float(third_octave_fmin), band_centers <= fmax)]
        else:
            # Precompute band centers (same definition as in third_octave_band_levels)
            k = np.arange(-50, 50)
            band_centers = 1000.0 * (2.0 ** (k / 3.0))
            band_centers = band_centers[np.logical_and(band_centers >= float(third_octave_fmin), band_centers <= fmax)]
        band_power_lists = [list() for _ in range(band_centers.size)]
    else:
        band_centers = np.array([], dtype=float)
        band_power_lists = []

    psd_power_lists = []
    f_sel_master = None
    Aweight_db = None

    # Precompute absorption reference range (meters)
    r_ref_m = None
    if apply_absorption_deprop:
        r_ref_m = float(unit_conversion.len_conv(r_ref, from_units=length_units, to_units='m'))

    for im in range(nmics):
        # PSD spectrogram on observer time axis
        t_vec = _time_vec_for_mic(im)
        fs = float(np.round(1.0 / (t_vec[1] - t_vec[0])))
        if not np.isclose(fs, fs0, rtol=1e-3, atol=0.0):
            raise ValueError('All microphones must have the same sample rate for hemisphere generation')
        f, t_rel, psd_db = spectrogram(pressure_list[im], fs, window_time=window_time, window_overlap=window_overlap)
        t_abs = t_rel + t_vec[0]

        fmask = np.logical_and(f >= fmin, f <= fmax)
        f_sel = f[fmask]
        if f_sel.size < 2:
            raise ValueError('Selected frequency range does not contain enough bins')

        if f_sel_master is None:
            f_sel_master = f_sel
        else:
            if f_sel_master.shape != f_sel.shape or not np.allclose(f_sel_master, f_sel, rtol=0.0, atol=0.0):
                raise ValueError('All microphones must have identical frequency grids for hemisphere generation')

        df = float(f_sel[1] - f_sel[0])

        if Aweight_db is None:
            Aweight_db = np.array([dBAw(fi) for fi in f_sel], dtype=float)
        Aweight_lin = 10.0 ** (Aweight_db / 10.0)

        alpha_db_per_m = None
        if apply_absorption_deprop:
            alpha_db_per_m = np.asarray(atmosphere.attenuation_coefficient(f_sel), dtype=float)

        # Observation times for selected emission points
        tobs_sub = t_obs[tidx, im]
        r_sub = r_geom[tidx, im]
        az_sub = az_deg[tidx, im]
        el_sub = el_deg[tidx, im]

        # Keep only points that can be interpolated in time
        valid = np.logical_and(tobs_sub >= t_abs[0], tobs_sub <= t_abs[-1])
        if not np.any(valid):
            continue

        tobs_v = tobs_sub[valid]
        r_v = r_sub[valid]
        az_v = az_sub[valid]
        el_v = el_sub[valid]

        # Interpolate PSD at tobs_v for each selected frequency
        psd_sel_db = psd_db[fmask, :]
        psd_v_db = np.empty((f_sel.size, tobs_v.size), dtype=float)
        for fi in range(f_sel.size):
            psd_v_db[fi, :] = np.interp(tobs_v, t_abs, psd_sel_db[fi, :])

        # Convert to linear PSD relative to pref^2/Hz
        psd_v_lin = 10.0 ** (psd_v_db / 10.0)

        # Spherical spreading depropagation to r_ref: multiply by (r/r_ref)^2
        spread_scale = (r_v / float(r_ref)) ** 2
        psd_v_lin = psd_v_lin * spread_scale[None, :]

        # Optional absorption depropagation back to r_ref
        if apply_absorption_deprop:
            assert r_ref_m is not None
            assert alpha_db_per_m is not None
            r_m = unit_conversion.len_conv(r_v, from_units=length_units, to_units='m').astype(float)
            deltaL = alpha_db_per_m[:, None] * (r_m[None, :] - r_ref_m)
            psd_v_lin = psd_v_lin * (10.0 ** (deltaL / 10.0))

        # OASPL power over selected frequency range
        power_oaspl = np.sum(psd_v_lin * df, axis=0)
        power_spl_a = np.sum((psd_v_lin * Aweight_lin[:, None]) * df, axis=0)

        fazi_list.append(az_v)
        felv_list.append(el_v)
        oaspl_power_list.append(power_oaspl)
        spl_a_power_list.append(power_spl_a)

        if narrowband:
            psd_power_lists.append(psd_v_lin)

        if third_octave:
            # Integrate to third-octave bands in linear power
            for ib, fc in enumerate(band_centers):
                f_lower = fc / (2.0 ** (1.0 / 6.0))
                f_upper = fc * (2.0 ** (1.0 / 6.0))
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

    # Enforce azimuth periodicity: duplicate scattered points at ±360 and close seam column
    fazi_ext = np.concatenate((fazi_pts, fazi_pts + 360.0, fazi_pts - 360.0))
    felv_ext = np.concatenate((felv_pts, felv_pts, felv_pts))
    P_oaspl_ext = np.concatenate((P_oaspl_pts, P_oaspl_pts, P_oaspl_pts))

    P_oaspl_grid = shepIDW(ELV_GRID, AZI_GRID, felv_ext, fazi_ext, P_oaspl_ext, rmax=float(rmax))
    eps = np.finfo(float).tiny
    oaspl_db = 10.0 * np.log10(np.maximum(P_oaspl_grid, eps))
    if oaspl_db.shape[1] > 1:
        oaspl_db[:, -1] = oaspl_db[:, 0]

    P_spl_a_ext = np.concatenate((P_spl_a_pts, P_spl_a_pts, P_spl_a_pts))
    P_spl_a_grid = shepIDW(ELV_GRID, AZI_GRID, felv_ext, fazi_ext, P_spl_a_ext, rmax=float(rmax))
    spl_a_db = 10.0 * np.log10(np.maximum(P_spl_a_grid, eps))
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
            'apply_absorption_deprop': bool(apply_absorption_deprop),
            'flip_y_for_geometry': bool(flip_y_for_geometry),
            'return_scattered': bool(return_scattered),
            'narrowband': bool(narrowband),
            'narrowband_stride': int(narrowband_stride),
        }
    }

    scattered: Optional[dict[str, Any]] = None
    if return_scattered:
        scattered = cast(dict[str, Any], {
            'azi_deg': fazi_pts,
            'elv_deg': felv_pts,
            'oaspl_power': P_oaspl_pts,
            'spl_a_power': P_spl_a_pts,
        })
        out['scattered'] = scattered

    if third_octave:
        # Concatenate scattered band powers and interpolate per band
        band_grids_db = np.full((band_centers.size, ELV_GRID.shape[0], ELV_GRID.shape[1]), -np.inf, dtype=float)
        for ib in range(band_centers.size):
            if len(band_power_lists[ib]) == 0:
                continue
            P_band_pts = np.concatenate(band_power_lists[ib])
            if P_band_pts.size != fazi_pts.size:
                raise ValueError('Internal error: third-octave sample count does not match scattered angle count')
            P_band_ext = np.concatenate((P_band_pts, P_band_pts, P_band_pts))
            P_band_grid = shepIDW(ELV_GRID, AZI_GRID, felv_ext, fazi_ext, P_band_ext, rmax=float(rmax))
            band_grids_db[ib, :, :] = 10.0 * np.log10(np.maximum(P_band_grid, eps))
            band_grids_db[ib, :, -1] = band_grids_db[ib, :, 0]

        out['third_octave'] = {
            'band_centers_hz': band_centers,
            'bands_db': band_grids_db,
        }

        if return_scattered:
            assert scattered is not None
            if band_centers.size > 0:
                band_power_pts = np.vstack([np.concatenate(band_power_lists[ib]) for ib in range(band_centers.size)])
                if band_power_pts.shape[1] != fazi_pts.size:
                    raise ValueError('Internal error: third-octave sample count does not match scattered angle count')
                scattered['third_octave'] = {
                    'band_centers_hz': band_centers,
                    'bands_db': 10.0 * np.log10(np.maximum(band_power_pts, eps)),
                }
            else:
                scattered['third_octave'] = {
                    'band_centers_hz': band_centers,
                    'bands_db': np.empty((0, fazi_pts.size), dtype=float),
                }

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
            P_f_pts = psd_power_pts[fi, :]
            P_f_ext = np.concatenate((P_f_pts, P_f_pts, P_f_pts))
            P_f_grid = shepIDW(ELV_GRID, AZI_GRID, felv_ext, fazi_ext, P_f_ext, rmax=float(rmax))
            psd_grid_db[i_f, :, :] = 10.0 * np.log10(np.maximum(P_f_grid, eps))
            psd_grid_db[i_f, :, -1] = psd_grid_db[i_f, :, 0]

        out['narrowband'] = {
            'frequency_hz': f_nb,
            'psd_db': psd_grid_db,
        }

        if return_scattered:
            assert scattered is not None
            scattered['narrowband'] = {
                'frequency_hz': f_nb,
                'psd_db': 10.0 * np.log10(np.maximum(psd_power_pts[::narrowband_stride, :], eps)),
            }

    return out


def array_coverage(ymics, altitude, xmin=-1000, xmax=1000, speed=100, rate=0.1, speed_of_sound=1135.):
    """
    Calculate the spherical coverage for an overflight of a linear microphone array
    Args:
        ymics: sideline distances of observers (at X=0 where +X is direction the vehicle is traveling)
        altitude: altitude of the vehicle
        xmin: optional, start distance of vehicle trajectory (default -1000)
        xmax: optional, end distance of vehicle trajectory (default 1000)
        speed: optional, default 100
        rate: optional, sampling period of emission time points, default 0.1
        speed_of_sound: optional, speed of sound, default 1135

    Returns: tuple (azimuth, elevation, r)
    WHERE
    azimith are the azimuth angles on the sphere for each emission time
    elevation are the elevation angles on the sphere for each emission time
    r are the propagation distances for each emission time
    """
    observers = np.zeros((len(ymics), 3))
    observers[:, 1] = ymics
    tmin = 0
    tmax = (xmax - xmin) / speed
    time = np.arange(tmin, tmax, rate)
    source = np.zeros((len(time), 3))
    source[:, 0] = xmin + time * speed
    source[:, 2] = altitude

    velocity = np.zeros_like(source)
    velocity[:, 0] = speed

    azimuth, elevation, r, t_observer, mach_r = hemigen(time, source, velocity, observers, speed_of_sound)
    return azimuth, elevation, r


def hover_array_coverage(radial_positions, mic_azimuths, altitudes, headings, speed_of_sound=1135.0):
    """
    Calculate the spherical coverage for a hovering vehicle with microphones placed
    on a circle (or rings) around the source location.

    Args:
        radial_positions: radial distances of observers from the hover point (same length units as altitude)
        mic_azimuths: observer azimuths around the hover point, degrees (0 deg = +X, 90 deg = +Y)
        altitudes: one or more hover altitudes (same length units as radial_positions)
        headings: one or more vehicle headings, degrees (0 deg = +X, 90 deg = +Y)
        speed_of_sound: optional, speed of sound, default 1135

    Returns: tuple (azimuth, elevation, r)
    WHERE
    azimuth are the azimuth angles on the sphere for each hover condition (time) and mic
    elevation are the elevation angles on the sphere for each hover condition (time) and mic
    r are the propagation distances for each hover condition (time) and mic
    """
    radial_positions = np.atleast_1d(radial_positions).astype(float)
    mic_azimuths = np.atleast_1d(mic_azimuths).astype(float)
    altitudes = np.atleast_1d(altitudes).astype(float)
    headings = np.atleast_1d(headings).astype(float)

    if radial_positions.size == 0 or mic_azimuths.size == 0:
        raise ValueError('radial_positions and mic_azimuths must be non-empty')
    if altitudes.size == 0 or headings.size == 0:
        raise ValueError('altitudes and headings must be non-empty')

    if radial_positions.size != mic_azimuths.size:
        raise ValueError('radial_positions and mic_azimuths must have the same length')

    # Build mic locations in the ground plane centered on the hover point.
    az_rad = np.radians(mic_azimuths)
    obs_x = radial_positions * np.cos(az_rad)
    obs_y = radial_positions * np.sin(az_rad)
    observers = np.column_stack((obs_x, obs_y, np.zeros(obs_x.size)))

    # Enumerate hover conditions; time is arbitrary and only used to align rows.
    alt_grid, head_grid = np.meshgrid(altitudes, headings, indexing='ij')
    n_combo = alt_grid.size
    time = np.arange(n_combo, dtype=float)

    source = np.zeros((n_combo, 3), dtype=float)
    source[:, 2] = alt_grid.ravel()

    heading_rad = np.radians(head_grid.ravel())
    # Unit velocity vector defines vehicle heading for azimuth convention.
    velocity = np.zeros_like(source)
    velocity[:, 0] = np.cos(heading_rad)
    velocity[:, 1] = np.sin(heading_rad)

    azimuth, elevation, r, t_observer, mach_r = hemigen(time, source, velocity, observers, speed_of_sound)
    return azimuth, elevation, r


def hover_array_coverage_plot(radial_positions, mic_azimuths, altitudes, headings, speed_of_sound=1135.0):
    """
    Plot the spherical coverage for hovering conditions with scattered microphones.

    Args:
        radial_positions: radial distances of observers from the hover point (same length units as altitude)
        mic_azimuths: observer azimuths around the hover point, degrees (0 deg = +X, 90 deg = +Y)
        altitudes: one or more hover altitudes (same length units as radial_positions)
        headings: one or more vehicle headings, degrees (0 deg = +X, 90 deg = +Y)
        speed_of_sound: optional, speed of sound, default 1135

    Returns: tuple (fig, ax, cs)
    WHERE
    fig is a matplotlib handle to the figure
    ax is a matplotlib handle to the plot axis
    cs is a matplotlib handle to the data points
    """
    azimuth, elevation, _ = hover_array_coverage(radial_positions, mic_azimuths, altitudes,
                                                 headings, speed_of_sound)
    fig, ax, cs = lambert_ea_points(np.radians(azimuth), np.radians(elevation))
    return fig, ax, cs


def array_coverage_plot(ymics, altitude, xmin=-1000, xmax=1000, speed=100, rate=0.1, speed_of_sound=1135.):
    """
    Plots the spherical coverage for an overflight of a linear microphone array
    Args:
        ymics: sideline distances of observers (at X=0 where +X is direction the vehicle is traveling)
        altitude: altitude of the vehicle
        xmin: optional, start distance of vehicle trajectory (default -1000)
        xmax: optional, end distance of vehicle trajectory (default 1000)
        speed: optional, default 100
        rate: optional, sampling period of emission time points, default 0.1
        speed_of_sound: optional, speed of sound, default 1135

    Returns: tuple (fig, ax, cs)
    WHERE
    fig is a matplotlib handle to the figure
    ax is a matplotlib handle to the plot axis
    cs is a matplotlib handle to the data points
    """
    azimuth, elevation, _ = array_coverage(ymics, altitude, xmin, xmax, speed, rate, speed_of_sound)
    fig, ax, cs = lambert_ea_points(np.radians(azimuth), np.radians(elevation))
    return fig, ax, cs


def load_nc_sphere(filename):
    """
    Load data from an AAM/RNM style acoustic hemisphere
    Args:
        filename: path to netCDF sphere file

    Returns: tuple (amplitude, phi, theta, frequency, radius, speed, flight_path_angle)
    WHERE
    amplitude is a phi x theta x frequency matrix of SPL amplitudes, dB
    phi is an array of lateral angles (degrees)
    theta is an array of longitudinal angles (degree)
    frequency is an array of frequencies, Hz
    radius is the sphere radius (feet)
    speed is the sphere speed (knots)
    flight_path_angle is the sphere flight path angle (degrees)
    """
    file_handle = Dataset(filename, mode='r')
    # Noise spheres sometimes contain NaN; we'll handle them later, so suppress the warning
    np_error_settings = np.seterr()
    np.seterr(invalid='ignore')
    phi = file_handle.variables['PHI'][:].astype(float)
    theta = file_handle.variables['THETA'][:].astype(float)
    frequency = file_handle.variables['FREQUENCY'][:].astype(float)
    amplitude = file_handle.variables['AMPLITUDE'][:].astype(float)
    radius = file_handle.variables['RADIUS'][:].astype(float)
    speed = file_handle.variables['SPEED'][:].astype(float)
    flight_path_angle = file_handle.variables['FLIGHT_PATH_ANGLE'][:].astype(float)
    # Reset warning settings
    np.seterr(**np_error_settings)
    file_handle.close()
    return amplitude, phi, theta, frequency, radius, speed, flight_path_angle


def write_aam_hemisphere_netcdf(
        filename,
        hemisphere,
        *,
        mode: str = 'auto',
        phi_deg=None,
        theta_deg=None,
        speed_knots: float = 0.0,
        flight_path_angle_deg: float = 0.0,
        radius_ft: Optional[float] = None,
    title: Optional[str] = None,
    BB: float = 1.0,
    NB: float = 0.0,
    PT: float = 0.0,
    doppler_shift_removed: float = 0.0,
    empty_weight_lb: float = 0.0,
    fuel_weight_lb: float = 0.0,
    load_weight_lb: float = 0.0,
    pylon_angle_deg: float = 90.0,
    masttilt_deg: float = 0.0,
    xyz_ft=(0.0, 0.0, 0.0),
        overwrite: bool = True,
):
    """Write a depropagated acoustic hemisphere to an AAM/RNM-style netCDF sphere.

    This exports a hemisphere in the same variable naming convention expected by
    :func:`load_nc_sphere` (and thus AAM-style tooling):

    - ``PHI`` (deg), ``THETA`` (deg), ``FREQUENCY`` (Hz)
    - ``AMPLITUDE`` (dB), shape (Nphi, Ntheta, Nfreq)
    - ``RADIUS`` (ft), ``SPEED`` (knots), ``FLIGHT_PATH_ANGLE`` (deg)

    The input ``hemisphere`` must be the dict returned by :func:`depropagate_hemisphere`
    with either ``third_octave=True`` or ``narrowband=True`` enabled.

    Notes
    -----
    ``depropagate_hemisphere`` produces a regular grid in UMAPR (azimuth/elevation).
    AAM spheres are stored on a regular grid in ART (phi/theta). This function
    interpolates the UMAPR grid onto a separable ART grid (``phi_deg`` x ``theta_deg``)
    in the *linear power* domain and converts back to dB.

    Args:
        filename: Path to the output netCDF file.
        hemisphere: Output dict from :func:`depropagate_hemisphere`.
        mode: 'auto', 'third_octave', or 'narrowband'. 'auto' prefers third-octave.
        phi_deg: Optional 1D ART phi grid (degrees). Default: 0..360 at the UMAPR azimuth step.
        theta_deg: Optional 1D ART theta grid (degrees). Default: 0..180 at the UMAPR elevation step.
        speed_knots: Stored into ``SPEED`` (knots).
        flight_path_angle_deg: Stored into ``FLIGHT_PATH_ANGLE`` (deg).
        radius_ft: Optional override for ``RADIUS`` (ft). If None, derived from hemisphere metadata ``r_ref``.
        overwrite: If False, raises when filename exists.

    Returns:
        None
    """

    if not overwrite and os.path.exists(filename):
        raise FileExistsError(f'Output file already exists: {filename}')

    if not isinstance(hemisphere, dict):
        raise TypeError('hemisphere must be a dict returned by depropagate_hemisphere')

    azi_grid_deg = np.asarray(hemisphere.get('azi_grid_deg', []), dtype=float)
    elv_grid_deg = np.asarray(hemisphere.get('elv_grid_deg', []), dtype=float)
    if azi_grid_deg.ndim != 1 or azi_grid_deg.size < 2:
        raise ValueError('hemisphere[\'azi_grid_deg\'] must be a 1D array with at least 2 elements')
    if elv_grid_deg.ndim != 1 or elv_grid_deg.size < 2:
        raise ValueError('hemisphere[\'elv_grid_deg\'] must be a 1D array with at least 2 elements')

    meta = hemisphere.get('metadata', {}) if isinstance(hemisphere.get('metadata', {}), dict) else {}

    # Determine export spectrum (frequency + band levels) from hemisphere.
    selected_mode = str(mode).lower()
    if selected_mode == 'auto':
        if 'third_octave' in hemisphere:
            selected_mode = 'third_octave'
        elif 'narrowband' in hemisphere:
            selected_mode = 'narrowband'
        else:
            selected_mode = 'none'

    if selected_mode == 'third_octave':
        if 'third_octave' not in hemisphere:
            raise ValueError('mode=third_octave but hemisphere does not include third_octave data')
        band_centers_hz = np.asarray(hemisphere['third_octave']['band_centers_hz'], dtype=float)
        band_levels_db_umapr = np.asarray(hemisphere['third_octave']['bands_db'], dtype=float)
        if band_levels_db_umapr.ndim != 3:
            raise ValueError('hemisphere[\'third_octave\'][\'bands_db\'] must be (Nf, Nelv, Nazi)')
        frequency_hz = band_centers_hz
        levels_db_umapr = band_levels_db_umapr
    elif selected_mode == 'narrowband':
        if 'narrowband' not in hemisphere:
            raise ValueError('mode=narrowband but hemisphere does not include narrowband data')
        frequency_hz = np.asarray(hemisphere['narrowband']['frequency_hz'], dtype=float)
        psd_db_umapr = np.asarray(hemisphere['narrowband']['psd_db'], dtype=float)
        if psd_db_umapr.ndim != 3:
            raise ValueError('hemisphere[\'narrowband\'][\'psd_db\'] must be (Nf, Nelv, Nazi)')
        if frequency_hz.size < 2:
            raise ValueError('narrowband frequency_hz must have at least 2 elements')
        # Convert PSD (dB re pref^2/Hz) to approximate per-bin band level (dB re pref^2)
        # so that AAM-style OASPL integration (sum of 10^(SPL/10)) is meaningful.
        df = float(np.median(np.diff(frequency_hz)))
        psd_lin = np.power(10.0, psd_db_umapr / 10.0)
        band_power = psd_lin * df
        eps = np.finfo(float).tiny
        levels_db_umapr = 10.0 * np.log10(np.maximum(band_power, eps))
    else:
        raise ValueError('hemisphere must include third_octave or narrowband data; run depropagate_hemisphere with third_octave=True and/or narrowband=True')

    if levels_db_umapr.shape[1] != elv_grid_deg.size or levels_db_umapr.shape[2] != azi_grid_deg.size:
        raise ValueError('Spectrum grid shape does not match elv_grid_deg/azi_grid_deg')

    # Ensure azimuth periodicity is represented in the interpolation grid.
    # We need the seam column at (azi0 + 360 deg) with values matching azi0,
    # otherwise points near 360 deg can fall out-of-bounds and become missing.
    azi_axis = np.asarray(azi_grid_deg, dtype=float).ravel()
    if np.any(np.diff(azi_axis) <= 0.0) or np.any(np.diff(elv_grid_deg) <= 0.0):
        raise ValueError('azi_grid_deg and elv_grid_deg must be strictly increasing')

    azi0 = float(azi_axis[0])
    azi_end = azi0 + 360.0
    if np.isclose(azi_axis[-1], azi_end, rtol=0.0, atol=1e-6):
        # Seam column present; ensure the last column matches the first.
        levels_db_umapr = levels_db_umapr.copy()
        levels_db_umapr[:, :, -1] = levels_db_umapr[:, :, 0]
    else:
        # No seam column; append one.
        azi_axis = np.concatenate((azi_axis, np.array([azi_end], dtype=float)))
        seam_col = levels_db_umapr[:, :, 0:1]
        levels_db_umapr = np.concatenate((levels_db_umapr, seam_col), axis=2)

    # Default ART grids.
    # A common AAM/RNM convention (and the included example sphere) uses:
    #   PHI:   -90..90   (lateral)
    #   THETA: 0..180    (longitudinal)
    # The step sizes can vary by dataset; pass phi_deg/theta_deg explicitly if you
    # need an exact grid.
    if phi_deg is None:
        phi_deg = np.arange(-90.0, 90.0 + 1e-9, 10.0)
    if theta_deg is None:
        theta_deg = np.arange(0.0, 180.0 + 1e-9, 5.0)
    phi_deg = np.asarray(phi_deg, dtype=float).ravel()
    theta_deg = np.asarray(theta_deg, dtype=float).ravel()
    if phi_deg.size < 2 or theta_deg.size < 2:
        raise ValueError('phi_deg and theta_deg must each have at least 2 values')

    # Determine reference radius in feet.
    if radius_ft is None:
        if 'r_ref' not in meta or 'length_units' not in meta:
            raise ValueError('radius_ft not provided and hemisphere metadata does not include r_ref/length_units')
        r_ref = float(meta['r_ref'])
        length_units = str(meta['length_units'])
        radius_ft = float(unit_conversion.len_conv(r_ref, from_units=length_units, to_units='ft'))

    # ART grid points -> UMAPR for interpolation
    TH, PH = np.meshgrid(theta_deg, phi_deg)
    azi_rad, elv_rad = art2umapr(np.deg2rad(PH), np.deg2rad(TH))
    azi_q = np.degrees(azi_rad).reshape(-1)
    # Wrap query azimuths into the interpolation axis range [azi0, azi0+360)
    azi_q = (azi_q - azi0) % 360.0 + azi0
    elv_q = np.degrees(elv_rad).reshape(-1)

    # Query points are (elv, azi) pairs in degrees
    pts = np.column_stack((elv_q, azi_q))

    # Build output amplitude array: (phi, theta, freq)
    nphi = phi_deg.size
    nth = theta_deg.size
    nfreq = int(np.asarray(frequency_hz).size)
    amplitude_db = np.full((nphi, nth, nfreq), -np.inf, dtype=float)

    # Interpolate each band in linear power and convert to dB.
    eps = np.finfo(float).tiny
    for k in range(nfreq):
        Lk = levels_db_umapr[k, :, :]
        Pk = np.power(10.0, Lk / 10.0)
        # Treat non-finite levels as zero power.
        Pk[~np.isfinite(Pk)] = 0.0
        interp = RegularGridInterpolator(
            (elv_grid_deg, azi_axis),
            Pk,
            bounds_error=False,
            fill_value=0.0,
        )
        Pq = interp(pts).reshape(nphi, nth)
        Lq = np.full_like(Pq, -np.inf, dtype=float)
        pos = Pq > 0.0
        if np.any(pos):
            Lq[pos] = 10.0 * np.log10(np.maximum(Pq[pos], eps))
        amplitude_db[:, :, k] = Lq

    # Replace non-finite levels with AAM-style missing sentinel (>1e34)
    missing_sentinel = np.float32(1.0e35)
    amplitude_to_write = amplitude_db.astype(np.float32)
    amplitude_to_write[~np.isfinite(amplitude_to_write)] = missing_sentinel

    # Write netCDF (match AAM naming conventions).
    # Use NETCDF3_CLASSIC for broad interoperability with legacy AAM/RNM tooling
    # and to avoid HDF5 backend dependency issues on some CI platforms.
    ds = Dataset(filename, mode='w', format='NETCDF3_CLASSIC')
    try:
        # Dimensions (legacy AAM/RNM spheres commonly include these singleton dims)
        ds.createDimension('PHI', nphi)
        ds.createDimension('THETA', nth)
        ds.createDimension('FREQUENCY', nfreq)
        ds.createDimension('XYZ', 3)
        for d in [
            'BB', 'NB', 'PT', 'DOPPLER_SHIFT_REMOVED',
            'EMPTY_WEIGHT', 'FUEL_WEIGHT', 'LOAD_WEIGHT',
            'RADIUS', 'FLIGHT_PATH_ANGLE', 'PYLON_ANGLE', 'SPEED', 'MASTTILT'
        ]:
            if d not in ds.dimensions:
                ds.createDimension(d, 1)

        # Core AAM hemisphere variables
        vphi = ds.createVariable('PHI', 'f4', ('PHI',))
        vth = ds.createVariable('THETA', 'f4', ('THETA',))
        vf = ds.createVariable('FREQUENCY', 'f4', ('FREQUENCY',))
        vamp = ds.createVariable('AMPLITUDE', 'f4', ('PHI', 'THETA', 'FREQUENCY'), fill_value=missing_sentinel)

        # Scalar flight/condition variables (AAM example uses 0-D scalars)
        vr = ds.createVariable('RADIUS', 'f4')
        vs = ds.createVariable('SPEED', 'f4')
        vfpa = ds.createVariable('FLIGHT_PATH_ANGLE', 'f4')

        # Additional AAM metadata variables seen in example spheres
        vBB = ds.createVariable('BB', 'f4')
        vNB = ds.createVariable('NB', 'f4')
        vPT = ds.createVariable('PT', 'f4')
        vDSR = ds.createVariable('DOPPLER_SHIFT_REMOVED', 'f4')
        vEW = ds.createVariable('EMPTY_WEIGHT', 'f4')
        vFW = ds.createVariable('FUEL_WEIGHT', 'f4')
        vLW = ds.createVariable('LOAD_WEIGHT', 'f4')
        vPA = ds.createVariable('PYLON_ANGLE', 'f4')
        vMT = ds.createVariable('MASTTILT', 'f4')
        vXYZ = ds.createVariable('XYZ', 'f4', ('XYZ',))

        vphi[:] = np.asarray(phi_deg, dtype=np.float32)
        vth[:] = np.asarray(theta_deg, dtype=np.float32)
        vf[:] = np.asarray(frequency_hz, dtype=np.float32)
        vamp[:, :, :] = amplitude_to_write

        vr.assignValue(np.float32(radius_ft))
        vs.assignValue(np.float32(speed_knots))
        vfpa.assignValue(np.float32(flight_path_angle_deg))

        vBB.assignValue(np.float32(BB))
        vNB.assignValue(np.float32(NB))
        vPT.assignValue(np.float32(PT))
        vDSR.assignValue(np.float32(doppler_shift_removed))
        vEW.assignValue(np.float32(empty_weight_lb))
        vFW.assignValue(np.float32(fuel_weight_lb))
        vLW.assignValue(np.float32(load_weight_lb))
        vPA.assignValue(np.float32(pylon_angle_deg))
        vMT.assignValue(np.float32(masttilt_deg))

        xyz = np.asarray(xyz_ft, dtype=np.float32).ravel()
        if xyz.size != 3:
            raise ValueError('xyz_ft must be a 3-element iterable (x, y, z) in feet')
        vXYZ[:] = xyz

        # Match the example file's attribute naming: 'unit' (singular)
        vphi.unit = 'DEGREE'
        vth.unit = 'DEGREE'
        vf.unit = 'HERTZ'
        vamp.unit = 'DECIBEL'
        vr.unit = 'FEET'
        vs.unit = 'KNOTS'
        vfpa.unit = 'DEGREE'
        vXYZ.unit = 'FEET'

        # The example sets unit attrs (often blank) on the metadata scalars.
        vBB.unit = ''
        vNB.unit = ''
        vPT.unit = ''
        vDSR.unit = ''
        vEW.unit = 'POUNDS'
        vFW.unit = 'POUNDS'
        vLW.unit = 'POUNDS'
        vPA.unit = 'DEGREE'
        vMT.unit = 'DEGREE'

        ds.title = title if title is not None else 'AAM/RNM acoustic hemisphere'
    finally:
        ds.close()


def OASPL(amplitudes):
    """
    Integrate an array of SPL amplitudes to compute the OASPL
    Args:
        amplitudes: array of SPL amplitudes
    Returns:
        OASPL, dB
    """
    return 10.0 * safe_log10(np.sum(np.power(10.0, amplitudes / 10.0)))


def safe_log10(x):
    """
    Safe version of log10, where values below zero are returned as NaN and values equal to zero are -inf
    Args:
        x: array of values

    Returns: checked log10 value
    """
    if x == 0:
        return -np.inf
    if x < 0:
        return np.nan
    else:
        return np.log10(x)


def dBAw(f):
    """
    A-weighting curve
    Args:
        f: frequencies to evaluate

    Returns: weighting (in dB) at each frequency
    """
    f = np.asarray(f, dtype=float)
    aweights = np.full_like(f, -300.0, dtype=float)
    pos = f > 0
    if np.any(pos):
        fp = f[pos]
        aweights[pos] = (
            10.0 * np.log10(
                1.562339
                * fp ** 4.0
                / ((fp ** 2.0 + 107.65265 ** 2.0) * (fp ** 2.0 + 737.86223 ** 2.0))
            )
            + 10.0
            * np.log10(
                2.242881e16
                * fp ** 4.0
                / ((fp ** 2.0 + 20.598997 ** 2.0) ** 2.0 * (fp ** 2.0 + 12194.22 ** 2.0) ** 2.0)
            )
        )
    return aweights


def load_nc_signal(filename):
    """
    Load NASA-formatted netCDF acoustic signal
    Args:
        filename: path to netCDF file

    Returns: tuple (pressure, time, location)
    WHERE
    pressure is an array of acoustic pressures
    time is an array of sampled times
    location is an array of the x,y,z location of the microphone
    """
    file_handle = Dataset(filename, mode='r')
    pressure = file_handle.variables['pressure'][:].astype(float).flatten()
    x = file_handle.X
    y = file_handle.Y
    z = file_handle.Z
    sample_rate = file_handle.sample_rate
    start_time = file_handle.start_time
    file_handle.close()
    time = np.arange(start_time, pressure.size / sample_rate + start_time, 1 / sample_rate)
    location = np.array([x, y, z])
    return pressure, time, location


def load_h5_signal(filename, datasetname='Table1', signalname=None):
    """
    Load HDF5 files from BKConnect
    Args:
        filename: path to file
        datasetname: optional dataset name (default 'Table1')
        signalname: optional signal name (default None, loads all signals)

    Returns: h5py dataset for entire group or specific signal
    """
    file = h5py.File(filename, 'r')
    if signalname is None:
        grp = file[datasetname]
        # Return the group, type checker might complain but runtime is fine
        return cast(h5py.Group, grp)  # type: ignore[return-value]
    else:
        grp = file[datasetname]
        if isinstance(grp, h5py.Group):
            ds = grp[signalname]
            return cast(h5py.Dataset, ds)  # type: ignore[return-value]
        else:
            raise ValueError(f"Dataset {datasetname} is not a Group")


def load_UFF_signal(filename, sets = None):
    """
    Load UFF acoustic signal file
    Args:
        filename: path to UFF file
        sets: optional list of set numbers to load, default None (loads all sets)
    Returns: tuple (pressures, fs, channel_names, time)
    WHERE
    pressures is a channels x timepoints matrix of acoustic pressures
    fs is the sampling rate, Hz
    channel_names is a list of channel names
    time is an array of sampled times
    """

    file = UFF(filename)

    if sets is None:
        data = file.read_sets()
    else:
        data = file.read_sets(sets)
    
    channels = len(data)
    datasize = len(data[0]['x'])
    time = data[0]['x']
    fs = 1./(time[1]-time[0])
    pressures = np.zeros((channels, datasize))
    channel_names = []
    for i in range(channels):
        if len(data[i]['x']) != datasize:
            raise ValueError('Inconsistent recording lengths in UFF file.')
        pressures[i, :] = data[i]['data']
        channel_names.append(data[i]['id1'])
    return pressures, fs, channel_names, time


def highpass(x, fpass, fs, zero_phase=True):
    """
    Applies a high pass filter to a signal
    Args:
        x: signal array
        fpass: high pass frequency
        fs: sampling rate of x
        zero_phase: optional, use phase preserving filter, default True

    Returns: filtered signal
    """
    return np.array(pa_filters.highpass(x, fpass, fs, zero_phase=zero_phase))


def lowpass(x, fpass, fs, zero_phase=True):
    """
    Applies a low pass filter to a signal
    Args:
        x: signal array
        fpass: low pass frequency
        fs: sampling rate of x
        zero_phase: optional, use phase preserving filter, default True

    Returns: filtered signal
    """
    return np.array(pa_filters.lowpass(x, fpass, fs, zero_phase=zero_phase))


def art2umapr(phi, theta):
    """
    Convert between ART (AAM/RNM/ANOPP) coordinates and UMAPR coordinates
    Args:
        phi: array of phi angles (radians)
        theta: array of theta angles (radians)

    Returns: tuple (azimuth, elevation) angles (radians)

    """
    x = -np.sin(theta) * np.sin(phi)
    y = np.cos(theta)
    z = -np.sin(theta) * np.cos(phi)
    elv = -np.arctan2(z, np.sqrt(np.square(x) + np.square(y)))
    azi = np.arctan2(x, -y)
    azi[azi < 0.0] = azi[azi < 0.0] + 2.0 * np.pi
    return azi, elv


def lambert_ea(lat, lon):
    """
    Lambert equal-area projection of spherical angles
    Args:
        lat: array of lateral angles, radians
        lon: array of longitudinal angles, radians

    Returns: tuple, (x, y)

    """
    q = 2.0 * np.sin((np.pi / 2.0 - lat) / 2.0)
    x = q * np.sin(lon)
    y = q * np.cos(lon)
    return x, y


def build_empirical_database(directory_name, database_filename, load_factors=np.linspace(0.7, 2.3, 5), infreqs=None,
                             distance=1000,
                             atmosphere=Atmosphere(temperature=293.15, pressure=101.325,
                                                   relative_humidity=20.0)):
    # TODO extend FPA range
    # TODO pack in redimensionalization data
    # TODO add reinterpolation flag

    # Get vehicle attributes
    (main_rotor_radius, main_rotor_area, main_rotor_tip_speed,
     _, _, _, _, _, _, weight_coefficient, _, _, _, _) = read_vehicle_data(directory_name)

    # Set up database
    ncdatabase = Dataset(os.path.abspath(os.path.expanduser(database_filename)), 'w')

    # Get list of full paths to netCDF files in directory
    local_glob = os.path.expanduser(directory_name) + '/*.nc'
    absolute_glob = os.path.abspath(local_glob)
    file_list = glob(absolute_glob)

    min_speed = np.inf
    sphere_index = 0
    for filename in file_list:
        # Load the sphere data
        _, _, phi_list, theta_list, radius, _, SPLA, EAA, speed, flight_path_angle = extract_SPL(filename, infreqs,
                                                                                                 distance, atmosphere)
        # Find the lowest speed file near level flight
        if speed < min_speed and np.abs(flight_path_angle) < 2.0:
            min_speed = speed
            min_speed_file = filename
        # Mirror image sphere to get upper surface
        theta, phi = np.meshgrid(theta_list, phi_list)
        # Flatten and concatenate mirror points
        theta = np.concatenate((theta, theta))
        phi = np.concatenate((phi, -phi))
        SPLA = np.concatenate((SPLA, SPLA))
        EAA = np.concatenate((EAA, EAA))
        # Augment load factor data
        for load_factor in load_factors:
            groupname = "sphere" + str(sphere_index)
            sphere_index = sphere_index + 1
            add_sphere_group(ncdatabase, groupname, phi, theta, radius, SPLA, EAA, speed, flight_path_angle,
                             load_factor, main_rotor_radius, main_rotor_tip_speed, weight_coefficient)

    # Now, adapt the lowest speed sphere to a hover sphere by averaging from fore to aft
    _, _, phi_list, theta_list, radius, _, SPLA, EAA, _, _ = extract_SPL(min_speed_file, infreqs, distance, atmosphere)
    if theta_list.size % 2 == 1:
        averages = int((theta_list.size - 1) / 2)
    else:
        averages = int(theta_list.size / 2)
    for i in range(averages):
        j = theta_list.size - 1 - i
        average_values = 0.5 * (SPLA[:, i] + SPLA[:, j])
        SPLA[:, i] = average_values
        SPLA[:, j] = average_values
    # Mirror image sphere to get upper surface
    theta, phi = np.meshgrid(theta_list, phi_list)
    # Flatten and concatenate mirror points
    theta = np.concatenate((theta, theta))
    phi = np.concatenate((phi, -phi))
    SPLA = np.concatenate((SPLA, SPLA))
    EAA = np.concatenate((EAA, EAA))
    # Set hover conditions
    speed = 0
    flight_path_angles = [-12, 0, 12]
    # Augment load factor data
    for load_factor in load_factors:
        for flight_path_angle in flight_path_angles:
            groupname = "sphere" + str(sphere_index)
            sphere_index = sphere_index + 1
            add_sphere_group(ncdatabase, groupname, phi, theta, radius, SPLA, EAA, speed, flight_path_angle,
                             load_factor, main_rotor_radius, main_rotor_tip_speed, weight_coefficient)


def add_sphere_group(ncdatabase, groupname, phi, theta, radius, SPLA, EAA, speed, flight_path_angle, load_factor,
                     main_rotor_radius, main_rotor_tip_speed, weight_coefficient):
    # Create a new group for this sphere
    this_group = ncdatabase.createGroup(groupname)
    # Define sphere nondimensional radius
    this_group.createDimension("radii", 1)
    this_group.createVariable("radius", 'f8', ("radii",))
    this_group.variables['radius'][:] = radius / main_rotor_radius
    # Define rotor scale
    this_group.createDimension("condition", 1)
    this_group.createVariable("rotor_scale", 'f8', ("condition",))
    this_group.variables['rotor_scale'][:] = main_rotor_radius
    # Define flight condition
    this_group.createVariable("advance_ratio", 'f8', ("condition",))
    knots_to_meters = 0.514444
    this_group.variables['advance_ratio'][:] = knots_to_meters * speed / main_rotor_tip_speed
    this_group.createVariable("flight_path_angle", 'f8', ("condition",))
    this_group.variables['flight_path_angle'][:] = flight_path_angle
    this_group.createVariable("thrust_coefficient", 'f8', ("condition",))
    this_group.variables['thrust_coefficient'][:] = load_factor * weight_coefficient
    # Define acoustic data
    this_group.createDimension("channels", phi.size)
    this_group.createVariable("phi", 'f8', ("channels",))
    this_group.variables['phi'][:] = phi.flatten()
    this_group.createVariable("theta", 'f8', ("channels",))
    this_group.variables['theta'][:] = theta.flatten()
    this_group.createVariable("dBA", 'f8', ("channels",))
    this_group.variables['dBA'][:] = SPLA.flatten() + 20 * np.log10(load_factor)
    this_group.createVariable("EAA", 'f8', ("channels",))
    this_group.variables['EAA'][:] = EAA.flatten()


def project_sphere(filename, altitude, elv_cutoff, infreqs=None,
                   atmosphere=Atmosphere(temperature=293.15, pressure=101.325,
                                         relative_humidity=20.0)):
    distance = 1000  # reference distance for EAA
    azi, elv, phi, theta, radius, SPLO, SPLA, EAA, speed, flight_path_angle = extract_SPL(filename, infreqs, distance,
                                                                                          atmosphere)
    # Discard values outside of elevation cutoff
    included_angles = elv >= np.radians(elv_cutoff)
    azi = azi[included_angles]
    elv = elv[included_angles]
    SPLO = SPLO[included_angles]
    SPLA = SPLA[included_angles]
    EAA = EAA[included_angles]
    slant_range = altitude / np.sin(elv)
    ground_range = np.sqrt(slant_range ** 2 - altitude ** 2)
    x = -ground_range * np.sin(azi)
    y = -ground_range * np.cos(azi)
    absorption = EAA / distance * (slant_range - radius)
    spreading = 20 * np.log10(radius / slant_range)
    LA = SPLA - absorption + spreading
    return x, y, LA, speed, flight_path_angle


def read_vehicle_data(directory_name, runs=None, speeds=None, flight_path_angles=None):
    # Process config file
    local_config = os.path.expanduser(directory_name) + '/vehicle.cfg'
    abs_config = os.path.abspath(local_config)
    config = ConfigParser()
    config.read(abs_config)
    number_of_main_rotor_blades = float(config['Main Rotor']['blades'])
    number_of_tail_rotor_blades = float(config['Tail Rotor']['blades'])
    if config.has_section('Option') and runs is not None:
        # read drag value and then open Excel reflist file for individual conditions
        nondimensional_flat_plate_drag = float(config['Option']['fbar'])
        reference_list = config['Option']['reflist']
        local_path_to_list = os.path.abspath(os.path.expanduser(directory_name) + '/' + reference_list)
        absolute_path_to_list = os.path.abspath(local_path_to_list)
        wb = openpyxl.load_workbook(absolute_path_to_list, read_only=True, data_only=True)
        ws = wb.active
        if ws is None:
            raise ValueError("Workbook has no active sheet")
        # Grab run values over the range in which they exist
        max_row = ws.max_row
        r_start = 'B2'
        r_end = 'B' + str(max_row)
        run_numbers = [i for i in np.array([[i.value for i in j] for j in ws[r_start:r_end]]).squeeze() if
                       i is not None]
        i_max = len(run_numbers)
        # Over the range where runs exist, grab the nondimensional condition values as an Nx3 matrix
        i_start = 'AH2'
        i_end = 'AJ' + str(i_max + 1)
        nondimensional_conditions = np.array([[i.value for i in j] for j in ws[i_start:i_end]]).squeeze()
        advance_ratios = []
        hover_tip_mach_numbers = []
        weight_coefficients = []
        for r in runs:
            index = run_numbers.index(r)
            advance_ratios.append(nondimensional_conditions[index][0])
            weight_coefficients.append(nondimensional_conditions[index][1])
            hover_tip_mach_numbers.append(nondimensional_conditions[index][2])
        advance_ratios = np.array(advance_ratios)
        hover_tip_mach_numbers = np.array(hover_tip_mach_numbers)
        weight_coefficients = np.array(weight_coefficients)
        # Grab dimensional condition indicated airspeeds in a similar way
        i_start = 'W2'
        i_end = 'W' + str(i_max + 1)
        dimensional_conditions = np.array([[i.value for i in j] for j in ws[i_start:i_end]]).squeeze()
        speeds = []
        for r in runs:
            index = run_numbers.index(r)
            speeds.append(dimensional_conditions[index])
        speeds = np.array(speeds)
        main_rotor_tip_speed = np.mean(0.514444 * speeds / advance_ratios)
        # Check on populating these?
        main_rotor_radius = None
        main_rotor_area = None
        tail_rotor_radius = None
        tail_rotor_area = None
        tail_rotor_tip_speed = None
        alphas = None
    else:
        main_rotor_radius = np.array(float(config['Main Rotor']['radius']))
        main_rotor_area = np.pi * main_rotor_radius ** 2
        main_rotor_tip_speed = np.array(float(config['Main Rotor']['tip speed']))
        tail_rotor_radius = np.array(float(config['Tail Rotor']['radius']))
        tail_rotor_area = np.pi * tail_rotor_radius ** 2
        tail_rotor_tip_speed = np.array(float(config['Tail Rotor']['tip speed']))
        ambient_temperature = np.array(float(config['Atmosphere']['temperature']))
        speed_of_sound = 20.05 * np.sqrt(ambient_temperature)
        ambient_density = np.array(float(config['Atmosphere']['density']))
        vehicle_weight_kg = np.array(float(config['Vehicle']['weight']))
        vehicle_weight_newtons = 9.82 * vehicle_weight_kg
        effective_flat_plate_drag_area = np.array(float(config['Vehicle']['drag']))
        nondimensional_flat_plate_drag = effective_flat_plate_drag_area / main_rotor_area
        hover_tip_mach_numbers = main_rotor_tip_speed / speed_of_sound
        weight_coefficients = vehicle_weight_newtons / (
                ambient_density * main_rotor_area * main_rotor_tip_speed ** 2)
        advance_ratios = None
        alphas = None
        if speeds is not None:
            ground_speed_meters_per_sec = 0.514444 * speeds
            advance_ratios = ground_speed_meters_per_sec / main_rotor_tip_speed
            hover_tip_mach_numbers = hover_tip_mach_numbers * np.ones_like(advance_ratios)
            drag_to_weight_ratio = 0.5 * nondimensional_flat_plate_drag * advance_ratios ** 2 / weight_coefficients
            if flight_path_angles is not None:
                alphas = -np.degrees(drag_to_weight_ratio) - flight_path_angles

    return (main_rotor_radius, main_rotor_area, main_rotor_tip_speed,
            number_of_main_rotor_blades, tail_rotor_radius, tail_rotor_area, tail_rotor_tip_speed,
            number_of_tail_rotor_blades, advance_ratios, weight_coefficients, hover_tip_mach_numbers, alphas,
            nondimensional_flat_plate_drag, speeds)


def project_directory(directory_name, altitude=500, cutoff=30, input_frequencies=None, fpa_climb_cutoff=5,
                      atmosphere=Atmosphere(temperature=293.15, pressure=101.325,
                                            relative_humidity=20.0), duration_correction=None):
    # Get list of full paths to netCDF files in directory
    local_glob = os.path.expanduser(directory_name) + '/*.nc'
    absolute_glob = os.path.abspath(local_glob)
    file_list = glob(absolute_glob)
    speeds = []
    flight_path_angles = []
    Lmax = []
    Lmean = []
    runs = []
    for filename in file_list:
        x, y, LA, speed, flight_path_angle = project_sphere(filename, altitude, cutoff, input_frequencies, atmosphere)
        if flight_path_angle <= fpa_climb_cutoff:
            speeds.append(speed)
            flight_path_angles.append(flight_path_angle)
            stencil = np.sqrt(x ** 2 + y ** 2) > 0
            Lmax.append(np.max(LA[stencil]))
            Lmean.append(np.mean(LA[stencil]))
            runs.append(int(filename[-6:-3]))
    speeds = np.array(speeds)
    flight_path_angles = np.array(flight_path_angles)
    Lmax = np.array(Lmax)
    Lmean = np.array(Lmean)
    # Apply duration correction if a reference speed is specified
    if duration_correction is not None:
        Lmax = Lmax + 10 * np.log10(duration_correction / speeds)
        Lmean = Lmean + 10 * np.log10(duration_correction / speeds)
    runs = np.array(runs)
    # Get nondimensional condition information
    (main_rotor_radius, main_rotor_area, main_rotor_tip_speed, number_of_main_rotor_blades,
     tail_rotor_radius, tail_rotor_area, tail_rotor_tip_speed, number_of_tail_rotor_blades,
     advance_ratios, weight_coefficients, hover_tip_mach_numbers, alphas, nondimensional_flat_plate_drag,
     speeds) = read_vehicle_data(directory_name, runs, speeds, flight_path_angles)

    return (speeds, flight_path_angles, Lmax, Lmean, advance_ratios, weight_coefficients, hover_tip_mach_numbers,
            alphas, runs, number_of_main_rotor_blades, number_of_tail_rotor_blades,
            nondimensional_flat_plate_drag, main_rotor_tip_speed)


def expand_if_single(x, r):
    """
    Expand a scalar or single-element array to match the shape of a reference array.

    This function takes a value (scalar or array) and expands it to match the shape
    of a reference array if it's a scalar or contains only one element. If the input
    already has multiple elements, it's returned unchanged.

    Args:
        x (scalar or array-like): The value to potentially expand. Can be a scalar
            or an array-like object.
        r (array-like): The reference array whose shape will be used for expansion.

    Returns:
        array-like: If x is a scalar or single-element array, returns an array of
            the same shape as r with all elements equal to x. Otherwise, returns x
            unchanged.
    """
    if np.isscalar(x) or len(x) == 1:
        x = x * np.ones_like(r)
    return x


def plot_fried_eggs(directory_names, metric='mean', dimensionless=False, altitude=500, cutoff=30,
                    input_frequencies=None, fpa_climb_cutoff=5,
                    atmosphere=Atmosphere(temperature=293.15, pressure=101.325,
                                          relative_humidity=20.0),
                    climb_rates=False, duration_correction=None, threshold=0.65, cull_noisy_fpa=None, xlim=(35, 140),
                    ylim=(-2000, 750), save_figures=False):
    for directory_name in directory_names:
        fig, ax, cs = fried_egg_plot(directory_name, metric, dimensionless, altitude, cutoff, input_frequencies,
                                     fpa_climb_cutoff, atmosphere, climb_rates, duration_correction, threshold,
                                     cull_noisy_fpa)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        mgr = getattr(fig.canvas, "manager", None)
        if mgr is not None and hasattr(mgr, "set_window_title"):
            mgr.set_window_title(directory_name)
        if save_figures:
            save_name = os.path.basename(directory_name) + '_fried_egg.pdf'
            fig.savefig(save_name)
    show(block=True)


def fried_egg_plot(directory_name, metric='mean', dimensionless=False, altitude=500, cutoff=30, input_frequencies=None,
                   fpa_climb_cutoff=5, atmosphere=Atmosphere(temperature=293.15, pressure=101.325,
                                                             relative_humidity=20.0),
                   climb_rates=False, duration_correction=None, threshold=0.65, cull_noisy_fpa=None,
                   suppress_classification=False):
    (speeds, flight_path_angles, Lmax, Lmean, advance_ratios, weight_coefficients, hover_tip_mach_numbers, alphas,
     runs, number_of_blades, number_of_tail_rotor_blades,
     effective_flat_plate_drags, tip_speeds) = project_directory(directory_name,
                                                                 altitude,
                                                                 cutoff,
                                                                 input_frequencies,
                                                                 fpa_climb_cutoff,
                                                                 atmosphere, duration_correction)
    samples = 1000
    if dimensionless:
        if advance_ratios is None or alphas is None:
            raise ValueError("Nondimensional data not available; set dimensionless=False or provide reflist.")
        x = advance_ratios
        y = alphas
    else:
        if climb_rates:
            assert speeds is not None and flight_path_angles is not None
            x = speeds
            y = 60 * 1.6878 * speeds * np.sin(np.radians(flight_path_angles))
        else:
            x = speeds
            y = flight_path_angles
    assert x is not None and y is not None
    xi = np.linspace(np.min(x), np.max(x), samples)
    yi = np.linspace(np.min(y), np.max(y), samples)
    if metric == 'mean':
        Lmetric = Lmean
    else:
        Lmetric = Lmax

    if cull_noisy_fpa is not None and threshold is not None:
        mask = data_filter(levels=Lmetric, flight_path_angles=flight_path_angles, threshold=threshold,
                           cull_noisy_fpa=cull_noisy_fpa)
        x = x[mask]
        y = y[mask]
        Lmetric = Lmetric[mask]

    triangles = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triangles, Lmetric)
    xim, yim = np.meshgrid(xi, yi)
    Li = interpolator(xim, yim)

    minSPL = np.min(Li)
    maxSPL = np.max(Li)
    num_levels = 9
    levels = np.linspace(minSPL, maxSPL, num_levels)
    color_map = get_ylorrd_cmap(num_levels)
    fig, ax = subplots(facecolor='white')
    cs = ax.contourf(xi, yi, Li, levels=levels, cmap=color_map)
    cb = colorbar(cs, format='%.0f')
    if metric == 'mean':
        if duration_correction is None:
            cb.set_label('Mean Ground Noise Level, dBA')
        else:
            cb.set_label('Ground Noise Exposure Level, dBA')
    else:
        if duration_correction is None:
            cb.set_label('Peak Ground Noise Level, dBA')
        else:
            cb.set_label('Ground Noise Exposure Level, dBA')
    if suppress_classification is False and threshold is not None:
        noisy_index = is_noisy(Lmetric, threshold)
        quiet_index = np.logical_not(noisy_index)
        ax.plot(x[quiet_index], y[quiet_index], 'ko', alpha=.8, markeredgecolor='w', markersize=10)
        ax.plot(x[noisy_index], y[noisy_index], 'ro', alpha=.8, markeredgecolor='k', markersize=10)
    if dimensionless:
        ax.set_xlabel('Advance Ratio - μ')
        ax.set_ylabel('Angle of Attack - α, °')
    else:
        if climb_rates:
            ax.set_xlabel('Speed, knots')
            ax.set_ylabel('Rate of Climb, fpm')
        else:
            ax.set_xlabel('Speed, knots')
            ax.set_ylabel('Flight Path Angle, °')
    return fig, ax, cs


def extract_SPL(filename, infreqs=None, distance=1000,
                atmosphere=Atmosphere(temperature=293.15, pressure=101.325,
                                      relative_humidity=20.0)):
    # Get data from netCDF file
    amplitude, phi, theta, frequency, radius, speed, flight_path_angle = load_nc_sphere(filename)
    # Convert radius to meters
    radius = radius * 0.3048
    # Replace bad values with no-energy SPL
    amplitude[np.isnan(amplitude)] = -np.inf
    amplitude[amplitude > 1.0e34] = -np.inf
    # Check frequency range
    if infreqs is not None:
        frequency_index = (frequency >= infreqs[0]) & (frequency <= infreqs[1])
        amplitude = amplitude[:, :, frequency_index]
        frequency = frequency[frequency_index]
    SPLO = np.apply_along_axis(OASPL, 2, amplitude)
    Aweight = np.array([dBAw(f) for f in frequency])
    SPLA = np.apply_along_axis(OASPL, 2, amplitude + Aweight)
    # Convert from ART to UMAPR coordinates
    T, P = np.meshgrid(theta, phi)
    azi, elv = art2umapr(np.pi * P / 180.0, np.pi * theta / 180.0)
    # Compute excess atmospheric attenuation
    alpha = atmosphere.attenuation_coefficient(frequency)
    SPLAa = np.apply_along_axis(OASPL, 2, amplitude + Aweight - distance * alpha)
    # SPLs may contain -inf, which is expected, so suppress numpy warning
    np_error_settings = np.seterr()
    np.seterr(invalid='ignore')
    EAA = SPLA - SPLAa
    np.seterr(**np_error_settings)
    return azi, elv, phi, theta, radius, SPLO, SPLA, EAA, speed, flight_path_angle


def nc_unwrapped(filename, infreqs=None, weight=None):
    """
    Generate an unwrapped noise contour plot from a netCDF acoustic sphere.

    This function extracts sound pressure level (SPL) data from a file and creates
    a filled contour plot showing the acoustic field distribution in azimuth-elevation
    coordinates.

    Parameters
    ----------
    filename : str
        Path to the file containing netCDF formatted acoustic sphere data.
    infreqs : array-like, optional
        Input frequencies for SPL extraction. Default is None.
    weight : str, optional
        Frequency weighting to apply. If 'A', uses A-weighted SPL (SPLA),
        otherwise uses overall SPL (SPLO). Default is None.

    Returns
    -------
    None

    """
    azi, elv, phi, theta, radius, SPLO, SPLA, EAA, speed, flight_path_angle = extract_SPL(filename, infreqs)
    if weight == 'A':
        SPL = SPLA
    else:
        SPL = SPLO
    minSPL = np.min(SPL)
    maxSPL = np.max(SPL)
    num_levels = 9
    levels = np.round(np.linspace(minSPL, maxSPL, num_levels))
    color_map = get_ylorrd_cmap(num_levels)
    contourf(phi, theta, np.transpose(SPL), levels=levels, cmap=color_map)
    colorbar()


def plot_projection(filename, altitude=500, cutoff=30, infreqs=None, units='m'):
    """
    Generate a contour plot of sound pressure levels projected onto a ground plane.

    This function computes the acoustic footprint of an aircraft at a specified altitude
    by projecting sound pressure levels onto a horizontal plane and creating a contour plot.

    Args:
        filename (str): Path to the file containing netCDF formatted acoustic sphere data
        altitude (float, optional): Aircraft altitude in meters. Defaults to 500.
        cutoff (float, optional): Cutoff angle in degrees for projection calculations. 
            Defaults to 30.
        infreqs (array-like, optional): Input frequencies for analysis. Defaults to None.
        units (str, optional): Units for plot axes ('m' for meters, 'ft' for feet, etc.). 
            Defaults to 'm'.

    Returns:
        tuple: A tuple containing:
            - fig (matplotlib.figure.Figure): The figure object
            - ax (matplotlib.axes.Axes): The axes object
            - cs (matplotlib.contour.QuadContourSet): The contour set object

    Notes:
        - The function uses triangulation and linear interpolation to create smooth contours
        - Sound pressure levels are displayed in dBA (A-weighted decibels)
        - The plot uses a YlOrRd colormap with 9 levels
        - Coordinate system: cross-track (x-axis) and along-track (y-axis) directions
    """
    x, y, LA, speed, flight_path_angle = project_sphere(filename, altitude, cutoff, infreqs)
    fig, ax = subplots(facecolor='white')
    xi = np.linspace(np.min(x), np.max(x), 100)
    yi = np.linspace(np.min(y), np.max(y), 100)
    triangles = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triangles, LA)
    xim, yim = np.meshgrid(xi, yi)
    li = interpolator(xim, yim)
    num_levels = 9
    levels = np.round(10 * np.linspace(np.min(LA), np.max(LA), num_levels)) / 10
    color_map = get_ylorrd_cmap(num_levels)
    xi = unit_conversion.len_conv(xi, from_units='m', to_units=units)
    yi = unit_conversion.len_conv(yi, from_units='m', to_units=units)
    cs = ax.contourf(xi, yi, li, levels=levels, cmap=color_map)
    ax.axis('equal')
    ax.set_xlabel('Cross Track Direction, ' + units)
    ax.set_ylabel('Flight Track Direction, ' + units)
    cb = colorbar(cs, format='%.0f')
    cb.set_label('Sound Pressure Level, dBA')
    return fig, ax, cs

def plot_lambert_ea(azi,elv,SPL,SPL_range=None,weight=None):
    """
    Generate a Lambert equal-area azimuthal projection contour plot of sound pressure levels.

    This function creates a polar plot of acoustic data using Lambert equal-area projection,
    displaying sound pressure levels (SPL) as contours with azimuth and elevation coordinates.

    Parameters
    ----------
    azi : array-like
        Array of azimuth angles in radians.
    elv : array-like
        Array of elevation angles in radians.
    SPL : array-like
        Array of sound pressure levels corresponding to the azimuth and elevation angles.
    SPL_range : tuple of float, optional
        Tuple specifying (min_SPL, max_SPL) for the contour levels. If None, the range
        is automatically determined from the data. Default is None.
    weight : str, optional
        Units used for SPL label. Use 'A' for A-weighted SPL, otherwise
        overall SPL is used. Default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    cs : matplotlib.contour.QuadContourSet
        The contour set object from the contourf plot.

    """
    SPL[np.isinf(SPL)] = np.nan
    if SPL_range is None:
        minSPL = np.nanmin(SPL)
        maxSPL = np.nanmax(SPL)
    else:
        minSPL = SPL_range[0]
        maxSPL = SPL_range[1]
    SPL[np.isnan(SPL)] = 0.0
    num_levels = 9
    levels = np.round(np.linspace(minSPL, maxSPL, num_levels))
    color_map = get_ylorrd_cmap(num_levels)
    # Project to Cartesian
    lat = elv
    lon = azi - np.pi
    x, y = lambert_ea(lat, lon)
    fig, ax = subplots(facecolor='white')
    ax.patch.set_visible(False)
    cs = ax.contourf(x, y, SPL, levels=levels, cmap=color_map)
    # make sure aspect ratio preserved 
    ax.set_aspect('equal')
    # turn off rectangular frame. 
    ax.set_frame_on(False)
    # turn off axis ticks. 
    ax.set_xticks([])
    ax.set_yticks([])
    # Draw meridians
    meridians = np.arange(0, 360, 45)
    for meridian in meridians:
        lats = np.linspace(0, 0.5 * np.pi, 1000)
        lons = (np.deg2rad(meridian) - np.pi) * np.ones(len(lats))
        xm, ym = lambert_ea(lats, lons)
        ax.plot(xm, ym, 'k--')
        # Add label
        xl, yl = lambert_ea(np.deg2rad(-11.0), np.deg2rad(meridian) - np.pi)
        ax.text(xl, yl, "%d°" % np.fmod(360 - meridian, 360), horizontalalignment='center', verticalalignment='center')
    # Draw parallels
    parallels = np.arange(0, 90, 30)
    for parallel in parallels:
        lons = np.linspace(-np.pi, np.pi, 1000)
        lats = np.deg2rad(parallel) * np.ones(len(lons))
        xm, ym = lambert_ea(lats, lons)
        ax.plot(xm, ym, 'k--')
        # Add label
        xl, yl = lambert_ea(np.deg2rad(parallel + 7.5), np.deg2rad(0.0))
        xpad = 0.025
        ypad = -0.025
        ax.text(xl + xpad, yl + ypad, "%d°" % parallel, horizontalalignment='left', verticalalignment='center')

    # Change cursor to display polar coordinates
    def format_coord(x, y):
        cazi = np.mod(90 - np.degrees(np.arctan2(-y, x)), 360)
        cq = np.sqrt(np.square(x) + np.square(y))
        celv = np.degrees(np.pi / 2 - 2 * np.arcsin(cq / 2))
        return 'ψ = %0.1f, θ = %0.1f' % (cazi, celv)

    ax.format_coord = format_coord
    cb = colorbar(cs, pad=0.1)
    if weight == 'A':
        cb.set_label('Sound Pressure Level, dBA')
    else:
        cb.set_label('Sound Pressure Level, dB')
    return fig, ax, cs

def nc_lambert_ea(filename, input_frequencies=None, weight=None, SPL_range=None):
    """
    Generate a Lambert equal-area azimuthal projection contour plot of sound pressure levels 
    on a netCDF formatted acoustic sphere.

    This function creates a polar plot of acoustic data using Lambert equal-area projection,
    displaying sound pressure levels (SPL) as contours with azimuth and elevation coordinates.

    Parameters
    ----------
    filename : str
        Path to the file containing acoustic data to be extracted and plotted.
    input_frequencies : array-like, optional
        Specific frequencies to extract from the data. If None, all frequencies are used.
    weight : str, optional
        Weighting scheme for SPL calculation. Use 'A' for A-weighted SPL, otherwise
        overall SPL is used. Default is None.
    SPL_range : tuple of float, optional
        Tuple specifying (min_SPL, max_SPL) for the contour levels. If None, the range
        is automatically determined from the data. Default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    cs : matplotlib.contour.QuadContourSet
        The contour set object from the contourf plot.

    """
    azi, elv, phi, theta, radius, SPLO, SPLA, EAA, speed, flight_path_angle = extract_SPL(filename, input_frequencies)
    if weight == 'A':
        SPL = SPLA
    else:
        SPL = SPLO
    fig, ax, cs = plot_lambert_ea(azi, elv, SPL, SPL_range)
    return fig, ax, cs

def lambert_ea_points(azimuth, elevation, markers=None, colors=None, sizes=None, alpha=1.0):
    """
    Plot levels on an acoustic sphere using the Lambert equal-area azimuthal projection.

    This function creates a polar plot using Lambert equal-area projection to visualize
    acoustic directivity patterns on a hemisphere.

    Parameters
    ----------
    azimuth : array_like
        Azimuth angles in radians. Convention: 0 at rear, increasing counterclockwise.
    elevation : array_like
        Elevation angles in radians. Convention: 0 at horizon, π/2 below the sphere.
    markers : array_like, optional
        Per-point marker styles. Must be the same size as azimuth/elevation when provided.
    colors : array_like or color, optional
        Per-point colors (same size as azimuth/elevation) or a single color spec.
    sizes : array_like or float, optional
        Per-point marker sizes (same size as azimuth/elevation) or a single size.
    alpha : float, optional
        Marker transparency, applied to all points. Default 1.0.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object with the Lambert projection plot.
    cs : list of matplotlib.artist.Artist
        Plot objects for the data points.

    """
    azimuth = np.asarray(azimuth)
    elevation = np.asarray(elevation)
    if azimuth.shape != elevation.shape:
        raise ValueError('azimuth and elevation must have the same shape')

    lat = elevation
    lon = azimuth - np.pi
    x, y = lambert_ea(lat, lon)
    fig, ax = subplots(facecolor='white')
    ax.patch.set_visible(False)

    if markers is None and colors is None and sizes is None:
        cs = ax.plot(x, y, 'ro', markersize=4, alpha=alpha)
    else:
        x_flat = np.ravel(x)
        y_flat = np.ravel(y)
        npts = x_flat.size

        if markers is None:
            markers_arr = np.array(['o'] * npts, dtype=object)
        else:
            markers_arr = np.asarray(markers, dtype=object).ravel()
            if markers_arr.size != npts:
                raise ValueError('markers must match the number of points')

        color_scalar = 'r'
        colors_arr = None
        if colors is not None:
            c_arr = np.asarray(colors)
            if c_arr.ndim == 0 or c_arr.shape in [(3,), (4,)]:
                color_scalar = colors
            elif c_arr.ndim == 2 and c_arr.shape[0] == npts and c_arr.shape[1] in (3, 4):
                colors_arr = c_arr
            else:
                c_arr = c_arr.ravel()
                if c_arr.size != npts:
                    raise ValueError('colors must match the number of points')
                colors_arr = c_arr

        size_scalar = 16.0
        sizes_arr = None
        if sizes is not None:
            s_arr = np.asarray(sizes, dtype=float)
            if s_arr.ndim == 0:
                size_scalar = float(s_arr)
            else:
                s_arr = s_arr.ravel()
                if s_arr.size != npts:
                    raise ValueError('sizes must match the number of points')
                sizes_arr = s_arr

        cs = []
        for marker in dict.fromkeys(markers_arr.tolist()):
            mask = markers_arr == marker
            c_val = colors_arr[mask] if colors_arr is not None else color_scalar
            s_val = sizes_arr[mask] if sizes_arr is not None else size_scalar
            cs.append(
                ax.scatter(
                    x_flat[mask],
                    y_flat[mask],
                    c=c_val,
                    s=s_val,
                    marker=marker,
                    edgecolors='k',
                    linewidths=0.5,
                    alpha=alpha,
                )
            )
    # make sure aspect ratio preserved
    ax.set_aspect('equal')
    # turn off rectangular frame.
    ax.set_frame_on(False)
    # turn off axis ticks.
    ax.set_xticks([])
    ax.set_yticks([])
    # Draw meridians
    meridians = np.arange(0, 360, 45)
    for meridian in meridians:
        lats = np.linspace(0, 0.5 * np.pi, 1000)
        lons = (np.deg2rad(meridian) - np.pi) * np.ones(len(lats))
        xm, ym = lambert_ea(lats, lons)
        ax.plot(xm, ym, 'k--')
        # Add label
        xl, yl = lambert_ea(np.deg2rad(-11.0), np.deg2rad(meridian) - np.pi)
        ax.text(xl, yl, "%d°" % np.fmod(360 - meridian, 360), horizontalalignment='center', verticalalignment='center')
    # Draw parallels
    parallels = np.arange(0, 90, 30)
    for parallel in parallels:
        lons = np.linspace(-np.pi, np.pi, 1000)
        lats = np.deg2rad(parallel) * np.ones(len(lons))
        xm, ym = lambert_ea(lats, lons)
        ax.plot(xm, ym, 'k--')
        # Add label
        xl, yl = lambert_ea(np.deg2rad(parallel + 7.5), np.deg2rad(0.0))
        xpad = 0.025
        ypad = -0.025
        ax.text(xl + xpad, yl + ypad, "%d°" % parallel, horizontalalignment='left', verticalalignment='center')

    # Change cursor to display polar coordinates
    def format_coord(x, y):
        cazi = np.mod(90 - np.degrees(np.arctan2(-y, x)), 360)
        cq = np.sqrt(np.square(x) + np.square(y))
        celv = np.degrees(np.pi / 2 - 2 * np.arcsin(cq / 2))
        return 'ψ = %0.1f, θ = %0.1f' % (cazi, celv)

    ax.format_coord = format_coord
    return fig, ax, cs


def data_filter(levels, flight_path_angles, threshold=0.65, cull_noisy_fpa=-1.0):
    cull_index = np.logical_and(is_noisy(levels, threshold), flight_path_angles > cull_noisy_fpa)
    mask = np.logical_not(cull_index)
    return mask


def is_noisy(level, threshold=0.65):
    cutoff = (np.max(level) - np.min(level)) * threshold + np.min(level)
    noisy = level > cutoff
    return noisy


def geodetic2array(geodetic, reference, heading, units='ft'):
    """
    Convert geodetic coordinates to a local array coordinate system.

    This function transforms geodetic coordinates (latitude, longitude, altitude) into a local
    coordinate system centered at a reference point, rotated according to a specified heading.

    Args:
        geodetic (numpy.ndarray): An Nx3 array of geodetic coordinates where each row contains
            [latitude, longitude, altitude].
        reference (array-like): A 3-element array containing the reference point's geodetic
            coordinates [latitude, longitude, altitude].
        heading (float): The heading angle in degrees, measured clockwise from north (0-360).
            The local coordinate system is rotated so that the x-axis aligns with this heading.
        units (str, optional): The desired output length units. Defaults to 'ft' (feet).
            The function converts from meters to the specified units.

    Returns:
        numpy.ndarray: An Nx3 array of local coordinates where:
            - Column 0: x-coordinate (aligned with heading direction)
            - Column 1: y-coordinate (perpendicular to heading, positive left)
            - Column 2: z-coordinate (up, altitude above reference)

    Notes:
        - Input geodetic coordinates are first converted to ENU (East-North-Up) coordinates
          relative to the reference point.
        - The coordinate system is then rotated so that the x-axis points in the direction
          of the specified heading.
        - The rotation angle is computed as 90° - heading to align the x-axis with the heading.
    """
    east, north, up = geodetic2enu(geodetic[:, 0], geodetic[:, 1], geodetic[:, 2], reference[0], reference[1],
                                   reference[2])
    rotation = np.radians(90.0 - heading)
    local = np.zeros((len(east), 3))
    local[:, 0] = east * np.cos(rotation) + north * np.sin(rotation)
    local[:, 1] = -east * np.sin(rotation) + north * np.cos(rotation)
    local[:, 2] = up
    local = unit_conversion.len_conv(local, from_units='m', to_units=units)
    return local


def array2geodetic(local, reference, heading, units='ft'):
    """
    Convert local coordinate array to geodetic coordinates (latitude, longitude, height).

    This function transforms an array of points from a local coordinate system to geodetic 
    coordinates (WGS84) by first rotating the local coordinates based on a heading angle 
    to align with East-North-Up (ENU) convention, then converting to geodetic coordinates 
    relative to a reference point.

    Parameters
    ----------
    local : numpy.ndarray
        Array of shape (N, 3) containing local coordinates [x, y, z] where:
        - x: lateral position (positive right)
        - y: longitudinal position (positive forward)
        - z: vertical position (positive up)
    reference : array-like
        Reference point in geodetic coordinates [latitude, longitude, height] in degrees 
        and meters respectively, used as the origin for the ENU coordinate system.
    heading : float
        Heading angle in degrees (typically 0-360), measured clockwise from North.
        Used to rotate the local coordinates to align with true North.
    units : str, optional
        Units of the input local coordinates. Default is 'ft' (feet).
        Coordinates are converted to meters internally.

    Returns
    -------
    numpy.ndarray
        Array of shape (N, 3) containing geodetic coordinates where:
        - Column 0: latitude in degrees
        - Column 1: longitude in degrees
        - Column 2: height in meters

    Notes
    -----
    The conversion process involves:
    1. Converting input coordinates from specified units to meters
    2. Rotating coordinates by (heading - 90°) to convert from local to ENU frame
    3. Converting ENU coordinates to geodetic using the reference point
    """
    local = unit_conversion.len_conv(local, from_units=units, to_units='m')
    rotation = np.radians(heading - 90.0)
    # Convert local to ENU
    east = local[:, 0] * np.cos(rotation) + local[:, 1] * np.sin(rotation)
    north = -local[:, 0] * np.sin(rotation) + local[:, 1] * np.cos(rotation)
    up = local[:, 2]
    lat, lon, h = enu2geodetic(east, north, up, reference[0], reference[1], reference[2])
    geodetic = np.zeros_like(local)
    geodetic[:, 0] = lat
    geodetic[:, 1] = lon
    geodetic[:, 2] = h
    return geodetic


def write_kml(geodetic, savename, testname='Array', channel_prefix="M"):
    """
    Write geodetic coordinates to a KML/KMZ file for visualization in mapping applications.

    Parameters
    ----------
    geodetic : array-like
        An iterable of coordinate pairs where each row contains [latitude, longitude].
        Coordinates should be in decimal degrees.
    savename : str
        The output file path where the KMZ file will be saved.
    testname : str, optional
        The name of the KML document. Default is 'Array'.
    channel_prefix : str, optional
        The prefix to use for naming each point/placemark. Default is 'M'.
        Points will be named as '{channel_prefix}{number}' (e.g., 'M1', 'M2', etc.).

    Returns
    -------
    None
        The function saves the KMZ file to disk but does not return a value.

    Notes
    -----
    - Each point is displayed with a circular placemark icon from Google Maps.
    - Point numbering starts from 1 (i.e., channel_prefix + '1' for the first point).
    - The geodetic input should have longitude at index 1 and latitude at index 0.
    """
    kml = simplekml.Kml()
    kml.document.name = testname
    for (i, row) in enumerate(geodetic):
        pnt = kml.newpoint(name=channel_prefix + "{}".format(i + 1), coords=[(row[1], row[0])])
        pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
    kml.savekmz(savename)


def get_ylorrd_cmap(num_levels=9):
    """
    Return a YlOrRd colormap with approximately `num_levels` discrete levels.
    Uses palettable when an exact discrete palette exists (3..9),
    otherwise falls back to Matplotlib's continuous 'YlOrRd'.
    """
    try:
        import palettable.colorbrewer.sequential as cbseq
        attr = f"YlOrRd_{int(num_levels)}"
        if hasattr(cbseq, attr):
            return getattr(cbseq, attr).mpl_colormap
    except Exception:
        pass
    return cm.get_cmap('YlOrRd')

def atmosorb(freq, temp, humid, pstat):
    """
    Atmospheric absorption using acoustics.atmosphere module

    Args:
        freq: Array-like of frequencies in Hz.
        temp: Temperature in degrees Celsius.
        humid: Relative humidity in percent (e.g., 85% -> 85).
        pstat: Static pressure in mbar.

    Returns:
        numpy.ndarray of attenuation in dB per meter (same shape as `freq`).
    """
    # Convert inputs to Atmosphere() expected units
    tempK = np.asarray(temp, dtype=float) + 273.15  # C -> K
    pres_kpa = np.asarray(pstat, dtype=float) * 0.1  # mbar -> kPa
    rh_pct = np.asarray(humid, dtype=float)
    f = np.asarray(freq, dtype=float)

    # Scalar case: fast path
    if tempK.ndim == 0 and pres_kpa.ndim == 0 and rh_pct.ndim == 0:
        atm = Atmosphere(
            temperature=float(tempK),
            pressure=float(pres_kpa),
            relative_humidity=float(rh_pct),
        )
        alpha_db_per_m = atm.attenuation_coefficient(f)
        return alpha_db_per_m

    # Broadcast atmospheric inputs to common grid
    tempK_b, pres_kpa_b, rh_b = np.broadcast_arrays(tempK, pres_kpa, rh_pct)

    if f.ndim == 0:
        out = np.empty_like(tempK_b, dtype=float)
        it = np.nditer(tempK_b, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            atm = Atmosphere(
                temperature=float(tempK_b[idx]),
                pressure=float(pres_kpa_b[idx]),
                relative_humidity=float(rh_b[idx]),
            )
            out[idx] = float(atm.attenuation_coefficient(float(f)))
            it.iternext()
        return out
    else:
        # Frequency array: return array with shape (grid_shape + f.shape)
        out = np.empty(tempK_b.shape + f.shape, dtype=float)
        for idx in np.ndindex(tempK_b.shape):
            atm = Atmosphere(
                temperature=float(tempK_b[idx]),
                pressure=float(pres_kpa_b[idx]),
                relative_humidity=float(rh_b[idx]),
            )
            out[idx] = atm.attenuation_coefficient(f)
        return out

def geodist(elv1, azi1, elv2, azi2):
    """
    Compute geodesic arc length (degrees) on a sphere 

    Args:
        elv1, azi1, elv2, azi2: Scalars or array-like angles in degrees.

    Returns:
        numpy.ndarray (or scalar) of arc length in degrees.
    """
    elv1 = np.asarray(elv1, dtype=float)
    elv2 = np.asarray(elv2, dtype=float)
    azi1 = np.asarray(azi1, dtype=float)
    azi2 = np.asarray(azi2, dtype=float)

    # Convert degrees to radians
    r_elv1 = np.deg2rad(elv1)
    r_elv2 = np.deg2rad(elv2)
    r_azi1 = np.deg2rad(azi1)
    r_azi2 = np.deg2rad(azi2)

    # Spherical law of cosines
    arcs = np.sin(r_elv1) * np.sin(r_elv2) + np.cos(r_elv1) * np.cos(r_elv2) * np.cos(r_azi2 - r_azi1)

    # Clamp for numerical safety
    arcs = np.clip(arcs, -1.0, 1.0)

    # Arc length in degrees
    arclength = np.degrees(np.arccos(arcs))
    return arclength


def IDWweights(ielv, iazi, felv, fazi, rmax):
    """
    Inverse Distance Weights (Shepard) with Franke & Nielson adjustments.

    Mirrors the MATLAB implementation:
        function wi = IDWweights(ielv,iazi,felv,fazi,rmax)

    Args:
        ielv: Interpolation point elevation (deg)
        iazi: Interpolation point azimuth (deg)
        felv: Field elevations (deg), array-like
        fazi: Field azimuths (deg), array-like (same shape/broadcastable with felv)
        rmax: Maximum radius (deg) for weighting neighborhood

    Returns:
        numpy.ndarray of weights with the same broadcasted shape as felv/fazi.
    """
    felv = np.asarray(felv, dtype=float)
    fazi = np.asarray(fazi, dtype=float)
    # Broadcast to common shape if needed
    felv, fazi = np.broadcast_arrays(felv, fazi)

    # Build arrays for the interpolation point to match shape
    ielv_arr = np.full(felv.shape, float(ielv))
    iazi_arr = np.full(fazi.shape, float(iazi))

    # Geodesic distances (degrees)
    hi = geodist(ielv_arr, iazi_arr, felv, fazi)

    # Threshold exact data points to avoid division by zero
    eps = np.finfo(float).eps
    if np.any(hi <= 10.0 * eps):
        wi = np.zeros_like(hi, dtype=float)
        b = np.argmin(hi)
        wi.flat[b] = 1.0
        return wi

    # Neighborhood mask within rmax (up to 2D as in MATLAB)
    mask = hi <= float(rmax)
    m = np.zeros_like(hi, dtype=float)
    # Franke & Nielson measure
    m[mask] = ((float(rmax) - hi[mask]) / (float(rmax) * hi[mask])) ** 2

    # Normalization
    M = m.sum()
    if M > 0.0:
        wi = m / M
    else:
        # No neighbors within rmax — return zeros (matches safe behavior)
        wi = m
    return wi


def shepIDW(ielv, iazi, felv, fazi, f, rmax):
    """
    Modified Shepard's Inverse Distance Weighting interpolation.

    Python version of the MATLAB function:
        function fi = shepIDW(ielv, iazi, felv, fazi, f, rmax)

    Args:
        ielv, iazi: Interpolant locations (deg). Scalars or array-like; must have same shape.
        felv, fazi: Data locations (deg). Arrays broadcastable to a common shape.
        f: Data values at (felv, fazi). Broadcastable to shape of felv/fazi.
        rmax: Maximum radius (deg) for neighborhood in weighting.

    Returns:
        numpy.ndarray of interpolated values with the same shape as `ielv` (or scalar if scalar inputs).
    """
    # Ensure numpy arrays and broadcasting for data fields
    felv = np.asarray(felv, dtype=float)
    fazi = np.asarray(fazi, dtype=float)
    fvals = np.asarray(f, dtype=float)
    felv, fazi, fvals = np.broadcast_arrays(felv, fazi, fvals)

    # Prepare interpolant inputs and preserve original shape
    ielv_arr = np.asarray(ielv, dtype=float)
    iazi_arr = np.asarray(iazi, dtype=float)
    if ielv_arr.shape != iazi_arr.shape:
        ielv_arr, iazi_arr = np.broadcast_arrays(ielv_arr, iazi_arr)
    orig_shape = ielv_arr.shape
    ielv_flat = ielv_arr.ravel()
    iazi_flat = iazi_arr.ravel()

    fi_flat = np.empty_like(ielv_flat, dtype=float)
    for i in range(ielv_flat.size):
        wi = IDWweights(ielv_flat[i], iazi_flat[i], felv, fazi, rmax)
        fi_flat[i] = np.sum(fvals * wi)

    fi = fi_flat.reshape(orig_shape)
    if fi.size == 1:
        return float(fi)
    return fi

def load_NASA_track(trackfile):
    """
    Load NASA track file and return data in a structured format.

    Args:
        trackfile: Path to NASA CSV format tracking data file.
    Returns:
        dict with keys:
            'time': numpy.ndarray of UTC time in seconds
            'latitude': numpy.ndarray of latitude in degrees
            'longitude': numpy.ndarray of longitude in degrees
            'altitude': numpy.ndarray of altitude in feet
            'heading': numpy.ndarray of heading in degrees  
            'pitch': numpy.ndarray of pitch in degrees
            'roll': numpy.ndarray of roll in degrees
            'x', 'y', 'z': numpy.ndarray of position in feet (local coordinates)
            'vx', 'vy', 'vz': numpy.ndarray of velocity in feet/second (local coordinates)
    """    
    data = np.genfromtxt(trackfile, delimiter=',', names=True)
    time = data['utcsec']
    latitude = data['lat']
    longitude = data['lon']
    altitude = data['alt']  
    heading = data['heading']
    pitch = data['pitch']
    roll = data['roll']
    x = data['x']
    y = data['y']
    z = data['z']
    vx = data['vx']
    vy = data['vy']
    vz = data['vz']   

    return {
        'time': time,
        'latitude': latitude,
        'longitude': longitude,
        'altitude': altitude,
        'heading': heading,
        'pitch': pitch,
        'roll': roll,
        'x': x,
        'y': y,
        'z': z,
        'vx': vx,
        'vy': vy,
        'vz': vz,
    }

def filter_track(track, xlims = None, ylims = None, zlims = None, decimate=1):
    """
    Filter track data based on specified limits for x, y, z coordinates.

    Args:
        track: dict containing track data with keys 'x', 'y', 'z'.
        xlims: tuple (xmin, xmax) for filtering x coordinates.
        ylims: tuple (ymin, ymax) for filtering y coordinates.
        zlims: tuple (zmin, zmax) for filtering z coordinates.
        decimate: integer factor to decimate the track data (default: 1, no decimation).
    Returns:
        dict containing filtered track data with the same keys as input.
    """    
    x = track['x']
    y = track['y']
    z = track['z']
    mask = np.ones_like(x, dtype=bool)

    if xlims is not None:
        mask = np.logical_and(mask, x >= xlims[0])
        mask = np.logical_and(mask, x <= xlims[1])
    if ylims is not None:
        mask = np.logical_and(mask, y >= ylims[0])
        mask = np.logical_and(mask, y <= ylims[1])
    if zlims is not None:
        mask = np.logical_and(mask, z >= zlims[0])
        mask = np.logical_and(mask, z <= zlims[1])

    filtered_track = {}
    for key in track:
        filtered_track[key] = track[key][mask]
    # Decimate track data
    if decimate > 1:
        for key in filtered_track:
            filtered_track[key] = filtered_track[key][::decimate]
    return filtered_track   

def ega(hs, hr, d2, f, a, flores, pt=True, cturb=0.0):
    """
    Calculate excess ground attenuation for a non-directional point source.
    
    Based on procedures in:
    - Chien and Soroka, "Sound Propagation Along an Impedance Plane",
      J. Sound. Vib., 43(1), 9-20, 1975 (corrected 1980)
    - Delany, Bazley, "Acoustical Properties of Fibrous Absorbent Materials",
      Appl. Acoust., 3, 105-116, 1970.
    - Chessell, "Propagation of noise along an impedance boundary",
      JASA 62(4), 825-834, 1977
    - Daigle, G.A., T.F.W. Embelton, and J.E. Piercy. 1979. "Some Comments on 
      the Literature of Propagation Near Boundaries of Finite Acoustical 
      Impedance," J. Acoust. Soc. Am. 66(3), 918-919.
    - Stusnick, Plotkin, Sutherland, "Short-Range Acoustic Propagation Model",
      Wyle Research Report WR 85-19, July 1985.
    
    Assumes level terrain, no wind, straight acoustic rays.
    
    Args:
        hs: Source height (ft or m, must match other length units)
        hr: Receiver height (ft or m)
        d2: 2D distance between source and receiver (ft or m)
        f: Frequency (Hz), scalar or array
        a: Speed of sound (ft/s or m/s, matching length units)
        flores: Specific flow resistance (kPa·s/m²)
            Typical values (from AAM technical reference):
                - Snow cover: 30
                - Grassy field: 225
                - Roadside soil: 650
                - Packed sand: 1650
                - Hard packed dirt: 3000
                - Exposed rock: 6000
                - Concrete: 10000
                - Asphalt: 50000
                - Water: 1e6 (effectively rigid)
        pt: True for pure tone (no third octave smearing), False for broadband (default: True)
        cturb: Turbulence parameter (rad·s·(m or ft)^-0.5)
               Typical: 0 to 16e-4 (rad·s·√m) or 0 to 52.5e-4 (rad·s·√ft)
    
    Returns:
        atten: Attenuation in dB
        phase: Phase (radians) of resultant wave relative to direct path
               (Pure tones only; NaN for broadband case)
    
    Notes:
        - Output "attenuation" is an attenuation factor in dB
        - Add to the SPL of the direct wave to compute SPL at receiver
        - Phase offset is meaningful only for pure tones
    """
    # Ensure float arrays
    hs = np.asarray(hs, dtype=float)
    hr = np.asarray(hr, dtype=float)
    d2 = np.asarray(d2, dtype=float)
    f = np.asarray(f, dtype=float)
    a = np.asarray(a, dtype=float)
    flores = np.asarray(flores, dtype=float)
    cturb = np.asarray(cturb, dtype=float)
    
    mu = 0.727477  # Spherical spreading coefficient
    eta = 6.325159  # Ground reflection coefficient
    
    # Calculate geometric values
    direct_range = np.sqrt(d2**2 + (hs - hr)**2)  # Direct acoustic path distance
    image_range = np.sqrt(d2**2 + (hs + hr)**2)  # Image source acoustic path distance
    grazing_angle = np.arccos((hs + hr) / image_range)  # Grazing angle (radians)
    path_delay = (image_range - direct_range) / a  # Time delay between direct and image paths
    range_ratio = image_range / direct_range  # Ratio of distances
    
    # Compute ground impedance and reflection coefficient
    freq_resistance_ratio = f / flores  # Normalized frequency-to-resistance ratio
    inv_freq_ratio = freq_resistance_ratio ** (-0.73)  # Inverse frequency ratio (Delany-Bazley)
    impedance_ratio = 1.0 / (1.0 + 9.08 * inv_freq_ratio / (freq_resistance_ratio ** 0.02) + 1j * 11.9 * inv_freq_ratio)
    
    cos_grazing = np.cos(grazing_angle)  # Cosine of grazing angle
    plane_wave_coeff = (cos_grazing - impedance_ratio) / (cos_grazing + impedance_ratio)  # Plane wave reflection coefficient
    
    # Compute numerical distance (simplified: 0.5*k1 = π*f/a)
    ground_effect_param = np.sqrt(1j * np.pi * f * image_range / a / (1.0 + impedance_ratio * cos_grazing)) * (cos_grazing + impedance_ratio)
    w = ground_effect_param ** 2  # Numerical distance parameter
    
    # Compute boundary loss factor (ground surface effect)
    boundary_loss = np.zeros_like(w, dtype=complex)
    mask = np.abs(w) <= 500
    sqrt_boundary = np.sqrt(w[mask])
    boundary_loss[mask] = 1 + 1j * np.sqrt(np.pi * w[mask]) * np.exp(-w[mask]) * (1 - erf(-1j * sqrt_boundary))
    
    # Compute image source strength
    image_source_coeff = plane_wave_coeff + boundary_loss * (1.0 - plane_wave_coeff)  # Combined reflection + boundary loss
    image_source_magnitude = np.abs(image_source_coeff)  # Magnitude of image source term
    image_source_phase = np.angle(image_source_coeff)  # Phase of image source term
    
    # Compute excess ground attenuation
    if pt:
        # Pure tone expression (Chessell's Equation 19)
        phase_delay = 1j * 2.0 * np.pi * f * path_delay  # Phase delay between paths
        resultant_amplitude = 1.0 + image_source_coeff * np.exp(phase_delay) / range_ratio  # Direct + image amplitude
        phase = np.angle(resultant_amplitude)
        attn = np.abs(resultant_amplitude)**2
    else:
        # Broadband mode (Chessell's Equations 20, 21)
        freq_path_delay = f * path_delay  # Normalized frequency-path delay product
        normalized_image_mag = image_source_magnitude / range_ratio  # Normalized image magnitude
        normalized_image_mag_sq = normalized_image_mag ** 2
        
        # Add turbulence factor (Chessell Equations 25, 27)
        turbulence_factor = np.exp(-(0.5 * cturb * f * np.sqrt(direct_range))**2) if np.any(cturb > 0.0) else 1.0
        
        # Combine attenuation computation
        ground_phase_term = eta * freq_path_delay + image_source_phase  # Ground reflection phase
        spherical_phase_term = mu * freq_path_delay  # Spherical spreading phase
        cosine_factor = np.cos(ground_phase_term) * turbulence_factor
        
        attn = 1.0 + normalized_image_mag_sq + 2.0 * normalized_image_mag * cosine_factor
        mask_positive_delay = path_delay > 0.0
        attn = np.where(
            mask_positive_delay,
            1.0 + normalized_image_mag_sq + 2.0 * normalized_image_mag * np.sin(spherical_phase_term) * cosine_factor / spherical_phase_term,
            attn
        )
        
        # Phase has no meaning for broadband, so set to NaN
        phase = np.full_like(attn, np.nan)
    
    # Convert magnitude to dB
    if np.all(attn > 0.0):
        attenuation_db = 10.0 * np.log10(attn)
    else:
        raise ValueError('Error in EGA: negative attenuation magnitude encountered')
    
    return attenuation_db, phase