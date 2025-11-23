import os
import flight_acoustics as fa
import matplotlib.pyplot as plt
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define data file paths
basepath = os.path.expanduser('~/Desktop/AS350_demo/')
acoustics_path = os.path.join(basepath, 'Acoustic')
ambient_path = os.path.join(basepath, 'Ambient')

# Load ambient noise data
nc_files = []
for root, _, files in os.walk(ambient_path):
    for name in files:
        if name.lower().endswith('.nc'):
            nc_files.append(os.path.join(root, name))

# Compute ambient spectra for each microphone     
plt.figure()       
ambient_frequencies = []
ambient_spectra = []
for nc_file in nc_files:
    ambient_pressure, ambient_time, ambient_location = fa.load_nc_signal(nc_file)
    logging.debug(f'Loaded ambient noise data from {nc_file}:')
    logging.debug(f'  Location: {ambient_location}')
    logging.debug(f'  Time samples: {len(ambient_time)}')
    logging.debug(f'  Pressure samples: {len(ambient_pressure)}')
    fs = 1.0 / (ambient_time[1] - ambient_time[0])  # Sampling frequency
    # Compute ambient spectra with 100 Hz bandwidth
    freq_amb, psd_amb, _, _ = fa.psd_welch(ambient_pressure, fs, window_time=0.01)
    ambient_frequencies.append(freq_amb)
    ambient_spectra.append(psd_amb) 
    plt.semilogx(freq_amb, psd_amb)
logging.info(f'Found {len(nc_files)} ambient .nc files.')
plt.title('Ambient Noise Spectra')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (dB/Hz)')

# Gather all .nc files in the Acoustics directory
nc_files = []
for root, _, files in os.walk(acoustics_path):
    for name in files:
        if name.lower().endswith('.nc'):
            nc_files.append(os.path.join(root, name))
            logging.debug(f'Loaded acoustic data from {nc_file}:')
            logging.debug(f'  Location: {ambient_location}')
            logging.debug(f'  Time samples: {len(ambient_time)}')
            logging.debug(f'  Pressure samples: {len(ambient_pressure)}')
logging.info(f'Found {len(nc_files)} acoustic .nc files.')

# Plot microphone grid
miclocs = []
pressures = []
times = []
for nc_file in nc_files:
    pressure, time, location = fa.load_nc_signal(nc_file)
    miclocs.append(location)
    pressures.append(pressure)
    times.append(time)

xs = [loc[0] for loc in miclocs]
ys = [loc[1] for loc in miclocs]

# Select subset of microphones for analysis based on their X location
filtered = [(f, loc) for f, loc in zip(nc_files, miclocs) if -2500 <= loc[0] <= -1500]
filtered_nc_files = [f for f, _ in filtered]
filtered_miclocs = np.array([loc for _, loc in filtered])

# Build lists of pressures and times for the filtered microphones
idx_by_file = {f: i for i, f in enumerate(nc_files)}
filtered_pressures = [pressures[idx_by_file[f]] for f in filtered_nc_files]
filtered_times = [times[idx_by_file[f]] for f in filtered_nc_files]

xf = [loc[0] for loc in filtered_miclocs]
yf = [loc[1] for loc in filtered_miclocs]

# Load trajectory data
track = fa.load_NASA_track(os.path.join(basepath, 'Tracking','289108AC.csv'))
logging.info(f'Loaded trajectory with {len(track["time"])} points from 289108AC.csv')
plt.figure()
plt.plot(track['x'],track['z'],c='tab:green',lw=2)
plt.xlabel('X,ft')
plt.ylabel('Z,ft')
plt.title('2D Flight Trajectory')
plt.grid(True, ls=':')

plt.figure()
plt.scatter(xs, ys, c='tab:blue', edgecolors='k')
plt.scatter(xf, yf, c='tab:red', edgecolors='k')
plt.plot(track['x'], track['y'], c='tab:green', lw=2)
plt.xlabel('X,ft')
plt.ylabel('Y,ft')
plt.title('Microphone Array and Flight Trajectory')
plt.axis('equal')
plt.grid(True, ls=':')

# Plot array coverage
source = np.array([track['x'], track['y'], track['z']]).transpose()
velocity = np.array([track['vx'], track['vy'], track['vz']]).transpose()
azimuths, elevations, ranges, tobs, mach_rs = fa.hemigen(track['time'], source, velocity, filtered_miclocs, speed_of_sound=1135.0)
fa.lambert_ea_points(np.radians(azimuths),np.radians(elevations))

# Plot reference hemisphere
AAM_path = os.path.join(basepath, 'AAM')
AAM_sphere = 'AS350B3108.nc'
freqs = [30,10000]
fig, ax, cs = fa.nc_lambert_ea(os.path.join(AAM_path,AAM_sphere), freqs,SPL_range=[85,105])
plt.show()