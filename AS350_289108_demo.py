import os
import flight_acoustics as fa
import matplotlib.pyplot as plt
import numpy as np


basepath = '/Users/evg5332/Desktop/AS350_demo/'
acoustics_path = os.path.join(basepath, 'Acoustic')

# Gather all .nc files in the Acoustics directory
nc_files = []
for root, _, files in os.walk(acoustics_path):
    for name in files:
        if name.lower().endswith('.nc'):
            nc_files.append(os.path.join(root, name))


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
filtered_miclocs = [loc for _, loc in filtered]

# Build lists of pressures and times for the filtered microphones
idx_by_file = {f: i for i, f in enumerate(nc_files)}
filtered_pressures = [pressures[idx_by_file[f]] for f in filtered_nc_files]
filtered_times = [times[idx_by_file[f]] for f in filtered_nc_files]


xf = [loc[0] for loc in filtered_miclocs]
yf = [loc[1] for loc in filtered_miclocs]

# Load trajectory data
track = fa.load_NASA_track(os.path.join(basepath, 'Tracking','289108AC.csv'))
print(track)

plt.figure()
plt.scatter(xs, ys, c='tab:blue', edgecolors='k')
plt.scatter(xf, yf, c='tab:red', edgecolors='k')
plt.xlabel('X,ft')
plt.ylabel('Y,ft')
plt.title('Microphone Array Locations')
plt.axis('equal')
plt.grid(True, ls=':')
plt.show()

fa.array_coverage_plot(yf,250)
plt.show()

# Plot reference hemisphere
AAM_path = os.path.join(basepath, 'AAM')
AAM_sphere = 'AS350B3108.nc'
freqs = [30,10000]
fig, ax, cs = fa.nc_lambert_ea(os.path.join(AAM_path,AAM_sphere), freqs,SPL_range=[85,105])
plt.show()