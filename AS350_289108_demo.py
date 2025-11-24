import os
import argparse
import flight_acoustics as fa
import matplotlib.pyplot as plt
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_ambient_spectra(ambient_path):
    nc_files = []
    for root, _, files in os.walk(ambient_path):
        for name in files:
            if name.lower().endswith('.nc'):
                nc_files.append(os.path.join(root, name))
    plt.figure()
    ambient_frequencies = []
    ambient_spectra = []
    for nc_file in nc_files:
        ambient_pressure, ambient_time, ambient_location = fa.load_nc_signal(nc_file)
        fs = 1.0 / (ambient_time[1] - ambient_time[0])
        freq_amb, psd_amb, _, _ = fa.psd_welch(ambient_pressure, fs, window_time=0.01)
        ambient_frequencies.append(freq_amb)
        ambient_spectra.append(psd_amb)
        plt.semilogx(freq_amb, psd_amb)
    logging.info(f'Found {len(nc_files)} ambient .nc files.')
    plt.title('Ambient Noise Spectra')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (dB/Hz)')
    return plt.gcf()

def gather_acoustic_files(acoustics_path):
    nc_files = []
    for root, _, files in os.walk(acoustics_path):
        for name in files:
            if name.lower().endswith('.nc'):
                nc_files.append(os.path.join(root, name))
    logging.info(f'Found {len(nc_files)} acoustic .nc files.')
    return nc_files

def load_microphones(nc_files):
    miclocs, pressures, times = [], [], []
    for nc_file in nc_files:
        pressure, time, location = fa.load_nc_signal(nc_file)
        miclocs.append(location)
        pressures.append(pressure)
        times.append(time)
    return miclocs, pressures, times

def filter_microphones(nc_files, miclocs, pressures, times, x_range=(-2500,-1500)):
    filtered = [(f, loc) for f, loc in zip(nc_files, miclocs) if x_range[0] <= loc[0] <= x_range[1]]
    filtered_nc_files = [f for f, _ in filtered]
    filtered_miclocs = np.array([loc for _, loc in filtered])
    idx_by_file = {f: i for i, f in enumerate(nc_files)}
    filtered_pressures = [pressures[idx_by_file[f]] for f in filtered_nc_files]
    filtered_times = [times[idx_by_file[f]] for f in filtered_nc_files]
    return filtered_nc_files, filtered_miclocs, filtered_pressures, filtered_times

def plot_trajectory(track, filter_track=None):
    fig1 = plt.figure()
    plt.plot(track['x'],track['z'],c='tab:green',lw=2)
    if filter_track is not None:
        plt.plot(filter_track['x'],filter_track['z'],c='tab:red',lw=2)
    plt.xlabel('X,ft')
    plt.ylabel('Z,ft')
    plt.title('2D Flight Trajectory')
    plt.grid(True, ls=':')
    fig2 = plt.figure()
    plt.plot(track['x'], track['y'], c='tab:green', lw=2)
    if filter_track is not None:
        plt.plot(filter_track['x'], filter_track['y'], c='tab:red', lw=2)
    plt.xlabel('X,ft')
    plt.ylabel('Y,ft')
    plt.title('Flight Trajectory (XY)')
    plt.axis('equal')
    plt.grid(True, ls=':')
    return [fig1, fig2]

def plot_array(xs, ys, xf, yf, track, filter_track = None):
    fig = plt.figure()
    plt.scatter(xs, ys, c='tab:blue', edgecolors='k')
    plt.scatter(xf, yf, c='tab:red', edgecolors='k')
    plt.plot(track['x'], track['y'], c='tab:green', lw=2)
    if filter_track is not None:
        plt.plot(filter_track['x'], filter_track['y'], c='tab:red', lw=2)
    plt.xlabel('X,ft')
    plt.ylabel('Y,ft')
    plt.title('Microphone Array and Flight Trajectory')
    plt.axis('equal')
    plt.grid(True, ls=':')
    return fig

def plot_array_coverage(filter_track, filtered_miclocs):
    source = np.array([filter_track['x'], filter_track['y'], filter_track['z']]).transpose()
    velocity = np.array([filter_track['vx'], filter_track['vy'], filter_track['vz']]).transpose()
    azimuths, elevations, ranges, tobs, mach_rs = fa.hemigen(filter_track['time'], source, velocity, filtered_miclocs, speed_of_sound=1135.0)
    fig, ax, _ = fa.lambert_ea_points(np.radians(azimuths),np.radians(elevations))
    return fig

def plot_reference_hemisphere(AAM_path, AAM_sphere, freqs, SPL_range=None):
    fig, ax, cs = fa.nc_lambert_ea(os.path.join(AAM_path,AAM_sphere), freqs, SPL_range=SPL_range)
    return fig

def save_figures(figs, outdir, names):
    os.makedirs(outdir, exist_ok=True)
    for fig, name in zip(figs, names):
        path = os.path.join(outdir, f"{name}.png")
        fig.savefig(path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved {path}")

def main():
    parser = argparse.ArgumentParser(description='AS350 289108 Demo Plot Generator')
    parser.add_argument('--all', action='store_true', help='Generate all plots and exit (non-interactive).')
    parser.add_argument('--outdir', default='demo_plots', help='Directory to save plots when using --all.')
    parser.add_argument('--basepath', default=os.path.expanduser('~/Desktop/AS350_demo/'), help='Base path to demo data.')
    args = parser.parse_args()

    basepath = args.basepath
    acoustics_path = os.path.join(basepath, 'Acoustic')
    ambient_path = os.path.join(basepath, 'Ambient')
    AAM_path = os.path.join(basepath, 'AAM')
    AAM_sphere = 'AS350B3108.nc'
    freqs = [30,10000]

    if not os.path.isdir(basepath):
        logging.error(f'Base path {basepath} does not exist.')
        return

    figs = []
    names = []

    # Ambient spectra
    figs.append(plot_ambient_spectra(ambient_path))
    names.append('ambient_noise_spectra')

    # Acoustic files & microphones
    acoustic_files = gather_acoustic_files(acoustics_path)
    miclocs, pressures, times = load_microphones(acoustic_files)
    xs = [loc[0] for loc in miclocs]
    ys = [loc[1] for loc in miclocs]
    filtered_nc_files, filtered_miclocs, filtered_pressures, filtered_times = filter_microphones(acoustic_files, miclocs, pressures, times)
    xf = [loc[0] for loc in filtered_miclocs]
    yf = [loc[1] for loc in filtered_miclocs]

    # Trajectory
    track = fa.load_NASA_track(os.path.join(basepath, 'Tracking','289108AC.csv'))
    filter_track = fa.filter_track(track, xlims=(-4000,0), zlims=(50,1500))
    traj_figs = plot_trajectory(track, filter_track)
    figs.extend(traj_figs)
    names.extend(['trajectory_xz', 'trajectory_xy'])

    # Array plot
    figs.append(plot_array(xs, ys, xf, yf, track, filter_track))
    names.append('microphone_array')

    # Coverage plot
    figs.append(plot_array_coverage(filter_track, filtered_miclocs))
    names.append('array_coverage_lambert')

    # Reference hemisphere
    figs.append(plot_reference_hemisphere(AAM_path, AAM_sphere, freqs, SPL_range=[85,105]))
    names.append('reference_hemisphere')

    if args.all:
        save_figures(figs, args.outdir, names)
    else:
        plt.show()

if __name__ == '__main__':
    main()