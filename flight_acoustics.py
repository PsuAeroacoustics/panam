# coding=UTF-8
import os
from configparser import ConfigParser
from glob import glob

import acoustics
import h5py
import openpyxl
import scipy.signal
import simplekml
from brewer2mpl import brewer2mpl
from matplotlib import tri
from matplotlib.pyplot import *
from netCDF4 import Dataset
from pymap3d import geodetic2enu, enu2geodetic

import unit_conversion

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


def spectrogram(signal, sampling_rate, window_time=1.0, window_type="hann", window_overlap=7.0 / 8.0,
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
                     detrend='constant', dbref=20e-6, save_name=None, time0=0, clim=None, flim=None):
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
    aweights = (10.0 * np.log10(1.562339 * f ** 4.0 / ((f ** 2.0 + 107.65265 ** 2.0) * (f ** 2.0 + 737.86223 ** 2.0)))
                + 10.0 * np.log10(2.242881E16 * f ** 4.0 /
                                  ((f ** 2.0 + 20.598997 ** 2.0) ** 2.0 * (f ** 2.0 + 12194.22 ** 2.0) ** 2.0)))
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
    pressure = file_handle.variables['pressure'][:].astype(float)
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
        return file[datasetname]
    else:
        return file[datasetname][signalname]


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
    return np.array(acoustics.Signal(x, fs).highpass(fpass, zero_phase=zero_phase))


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
    return np.array(acoustics.Signal(x, fs).lowpass(fpass, zero_phase=zero_phase))


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
                             atmosphere=acoustics.atmosphere.Atmosphere(temperature=293.15, pressure=101.325,
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

    min_speed = np.Inf
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
                   atmosphere=acoustics.atmosphere.Atmosphere(temperature=293.15, pressure=101.325,
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
                      atmosphere=acoustics.atmosphere.Atmosphere(temperature=293.15, pressure=101.325,
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
    if np.isscalar(x) or len(x) == 1:
        x = x * np.ones_like(r)
    return x


def plot_fried_eggs(directory_names, metric='mean', dimensionless=False, altitude=500, cutoff=30,
                    input_frequencies=None, fpa_climb_cutoff=5,
                    atmosphere=acoustics.atmosphere.Atmosphere(temperature=293.15, pressure=101.325,
                                                               relative_humidity=20.0),
                    climb_rates=False, duration_correction=None, threshold=0.65, cull_noisy_fpa=None, xlim=(35, 140),
                    ylim=(-2000, 750), save_figures=False):
    for directory_name in directory_names:
        fig, ax, cs = fried_egg_plot(directory_name, metric, dimensionless, altitude, cutoff, input_frequencies,
                                     fpa_climb_cutoff, atmosphere, climb_rates, duration_correction, threshold,
                                     cull_noisy_fpa)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        fig.canvas.set_window_title(directory_name)
        if save_figures:
            save_name = os.path.basename(directory_name) + '_fried_egg.pdf'
            fig.savefig(save_name)
    show(block=True)


def fried_egg_plot(directory_name, metric='mean', dimensionless=False, altitude=500, cutoff=30, input_frequencies=None,
                   fpa_climb_cutoff=5, atmosphere=acoustics.atmosphere.Atmosphere(temperature=293.15, pressure=101.325,
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
        x = advance_ratios
        y = alphas
    else:
        if climb_rates:
            x = speeds
            y = 60 * 1.6878 * speeds * np.sin(np.radians(flight_path_angles))
        else:
            x = speeds
            y = flight_path_angles
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
    color_map = brewer2mpl.get_map('YlOrRd', 'sequential', num_levels)
    fig, ax = subplots(facecolor='white')
    cs = ax.contourf(xi, yi, Li, levels=levels, colors=color_map.mpl_colors)
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
                atmosphere=acoustics.atmosphere.Atmosphere(temperature=293.15, pressure=101.325,
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
    azi, elv, phi, theta, radius, SPLO, SPLA, EAA, speed, flight_path_angle = extract_SPL(filename, infreqs)
    if weight == 'A':
        SPL = SPLA
    else:
        SPL = SPLO
    minSPL = np.min(SPL)
    maxSPL = np.max(SPL)
    num_levels = 9
    levels = np.round(np.linspace(minSPL, maxSPL, num_levels))
    color_map = brewer2mpl.get_map('YlOrRd', 'sequential', num_levels)
    contourf(phi, theta, np.transpose(SPL), levels=levels, colors=color_map.mpl_colors)
    colorbar()


def plot_projection(filename, altitude=500, cutoff=30, infreqs=None, units='m'):
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
    color_map = brewer2mpl.get_map('YlOrRd', 'sequential', num_levels)
    xi = unit_conversion.len_conv(xi, from_units='m', to_units=units)
    yi = unit_conversion.len_conv(yi, from_units='m', to_units=units)
    cs = ax.contourf(xi, yi, li, levels=levels, colors=color_map.mpl_colors)
    ax.axis('equal')
    ax.set_xlabel('Cross Track Direction, ' + units)
    ax.set_ylabel('Flight Track Direction, ' + units)
    cb = colorbar(cs, format='%.0f')
    cb.set_label('Sound Pressure Level, dBA')
    return fig, ax, cs


def nc_lambert_ea(filename, input_frequencies=None, weight=None, SPL_range=None):
    azi, elv, phi, theta, radius, SPLO, SPLA, EAA, speed, flight_path_angle = extract_SPL(filename, input_frequencies)
    if weight == 'A':
        SPL = SPLA
    else:
        SPL = SPLO
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
    color_map = brewer2mpl.get_map('YlOrRd', 'sequential', num_levels)
    # Project to Cartesian
    lat = elv
    lon = azi - np.pi
    x, y = lambert_ea(lat, lon)
    fig, ax = subplots(facecolor='white')
    ax.patch.set_visible(False)
    cs = ax.contourf(x, y, SPL, levels=levels, colors=color_map.mpl_colors)
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
    def format_coord(xx, yy):
        cazi = np.mod(90 - np.degrees(np.arctan2(-yy, xx)), 360)
        cq = np.sqrt(np.square(xx) + np.square(yy))
        celv = np.degrees(np.pi / 2 - 2 * np.arcsin(cq / 2))
        return 'ψ = %0.1f, θ = %0.1f' % (cazi, celv)

    ax.format_coord = format_coord
    cb = colorbar(cs, pad=0.1)
    if weight == 'A':
        cb.set_label('Sound Pressure Level, dBA')
    else:
        cb.set_label('Sound Pressure Level, dB')
    return fig, ax, cs


def lambert_ea_points(azimuth, elevation):
    lat = elevation
    lon = azimuth - np.pi
    x, y = lambert_ea(lat, lon)
    fig, ax = subplots(facecolor='white')
    ax.patch.set_visible(False)
    cs = ax.plot(x, y, 'ro', markersize=4)
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
    def format_coord(xx, yy):
        cazi = np.mod(90 - np.degrees(np.arctan2(-yy, xx)), 360)
        cq = np.sqrt(np.square(xx) + np.square(yy))
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
    kml = simplekml.Kml()
    kml.document.name = testname
    for (i, row) in enumerate(geodetic):
        pnt = kml.newpoint(name=channel_prefix + "{}".format(i + 1), coords=[(row[1], row[0])])
        pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
    kml.savekmz(savename)
