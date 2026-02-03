#!/usr/bin/env python3

import os
import argparse
import flight_acoustics as fa
import matplotlib.pyplot as plt
import numpy as np
import logging

def plot_ambient_spectra(ambient_path):
    nc_files = []
    for root, _, files in os.walk(ambient_path):
        for name in files:
            if name.lower().endswith('.nc'):
                nc_files.append(os.path.join(root, name))
    plt.figure()
    for nc_file in nc_files:
        ambient_pressure, ambient_time, ambient_location = fa.load_nc_signal(nc_file)
        fs = np.round(1.0 / (ambient_time[1] - ambient_time[0]))
        freq_amb, psd_amb, _, _ = fa.psd_welch(ambient_pressure, fs, window_time=0.01)
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
    filtered_miclocs = np.array([loc for _, loc in filtered])
    idx_by_file = {f: i for i, f in enumerate(nc_files)}
    filtered_nc_files = [f for f, _ in filtered]
    filtered_pressures = [pressures[idx_by_file[f]] for f in filtered_nc_files]
    filtered_times = [times[idx_by_file[f]] for f in filtered_nc_files]
    return filtered_miclocs, filtered_pressures, filtered_times

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

def plot_array_coverage(azimuths, elevations):
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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



    parser = argparse.ArgumentParser(description='AS350 289108 Demo Plot Generator')
    parser.add_argument('--show', action='store_true', help='Display plots interactively instead of saving.')
    parser.add_argument('--outdir', default='demo_plots', help='Directory to save plots (default behavior).')
    parser.add_argument('--basepath', default=os.path.expanduser('~/Desktop/AS350_demo/'), help='Base path to demo data.')
    args = parser.parse_args()

    save_outputs = not args.show

    basepath = args.basepath
    acoustics_path = os.path.join(basepath, 'Acoustic')
    ambient_path = os.path.join(basepath, 'Ambient')
    AAM_path = os.path.join(basepath, 'AAM')
    AAM_sphere = 'AS350B3108.nc'

    # Hemisphere processing parameters
    build_hemisphere = True
    apply_absorption_deprop = True
    hemisphere_azi_step = 10.0  # deg
    hemisphere_elv_step = 10.0  # deg
    hemisphere_rmax = 25.0  # deg (Shepard neighborhood radius)
    hemisphere_point_stride = 1  # decimate emission-time samples for speed
    # Consistent color scaling for hemisphere contour plots
    hemisphere_spl_range = (80.0, 110.0)
    # Frequency range for hemisphere / reference comparisons
    hemisphere_freq_range_hz = (0.0, 2000.0)
    freqs = list(hemisphere_freq_range_hz)
    r_ref = 100.0  # Reference distance for depropagation in feet

    # Mirror Y-axis per track convention
    flip_y_for_geometry = True

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
    filtered_miclocs, filtered_pressures, filtered_times = filter_microphones(acoustic_files, miclocs, pressures, times)
    xf = [loc[0] for loc in filtered_miclocs]
    yf = [loc[1] for loc in filtered_miclocs]

    # Trajectory
    track = fa.load_NASA_track(os.path.join(basepath, 'Tracking','289108AC.csv'))
    filter_track = fa.filter_track(track, xlims=(-4000,0), zlims=(50,1500),decimate=10)
    traj_figs = plot_trajectory(track, filter_track)
    figs.extend(traj_figs)
    names.extend(['trajectory_xz', 'trajectory_xy'])

    # Array plot
    figs.append(plot_array(xs, ys, xf, yf, track, filter_track))
    names.append('microphone_array')

    # Kinematics inputs for hemisphere generation
    source = np.array([filter_track['x'], filter_track['y'], filter_track['z']]).transpose()
    velocity = np.array([filter_track['vx'], filter_track['vy'], filter_track['vz']]).transpose()

    # Build a hemisphere and interpolate onto a regular spherical grid
    if build_hemisphere:
        # Pressure correction: adjust for pressure doubling at the ground board.
        pressures_corrected = [np.asarray(p, dtype=float) * 0.5 for p in filtered_pressures]

        atm = fa.Atmosphere(temperature=293.15, pressure=101.325, relative_humidity=20.0)
        hemi = fa.depropagate_hemisphere(
            mic_locations=filtered_miclocs,
            pressure=pressures_corrected,
            time=filtered_times,
            track_time=filter_track['time'],
            track_position=source,
            track_velocity=velocity,
            speed_of_sound=1135.0,
            length_units='ft',
            r_ref=r_ref,
            freq_range=hemisphere_freq_range_hz,
            window_time=0.5,
            window_overlap=0.5,
            point_stride=hemisphere_point_stride,
            azi_step=hemisphere_azi_step,
            elv_step=hemisphere_elv_step,
            rmax=hemisphere_rmax,
            apply_absorption_deprop=apply_absorption_deprop,
            atmosphere=atm,
            flip_y_for_geometry=flip_y_for_geometry,
            return_scattered=True,
            third_octave=True,
            third_octave_fmin=max(20.0, float(hemisphere_freq_range_hz[0])),
        )

        band_centers = hemi['third_octave']['band_centers_hz']
        hemi_bands_db = hemi['third_octave']['bands_db']
        hemi_oaspl_db = hemi['oaspl_db']
        hemi_spl_a_db = hemi['spl_a_db']
        hemi_oaspl_full_db = hemi_oaspl_db

        azi_grid = hemi['azi_grid_deg']
        elv_grid = hemi['elv_grid_deg']
        AZI_GRID, ELV_GRID = np.meshgrid(azi_grid, elv_grid)

        # Coverage plot (uses scattered az/el from hemisphere samples)
        figs.append(plot_array_coverage(hemi['scattered']['azi_deg'], hemi['scattered']['elv_deg']))
        names.append('array_coverage_lambert')

        eps = np.finfo(float).tiny
        oaspl_fmax = float(hemisphere_freq_range_hz[1])
        oaspl_ref_fmax = float(hemisphere_freq_range_hz[1])

        # Plot OASPL integrated below 2 kHz using flight_acoustics plotting helper
        fig_oaspl, ax_oaspl, _ = fa.plot_lambert_ea(
            np.deg2rad(AZI_GRID),
            np.deg2rad(ELV_GRID),
            hemi_oaspl_db.copy(),
            SPL_range=hemisphere_spl_range,
        )
        figs.append(fig_oaspl)
        names.append('hemisphere_oaspl_lt2khz_lambert')

        # Plot full-band OASPL (kept for compatibility; same band as above in this demo)
        fig_oaspl_full, ax_oaspl_full, _ = fa.plot_lambert_ea(
            np.deg2rad(AZI_GRID),
            np.deg2rad(ELV_GRID),
            hemi_oaspl_full_db.copy(),
            SPL_range=hemisphere_spl_range,
        )
        figs.append(fig_oaspl_full)
        names.append('hemisphere_oaspl_fullband_lambert')

        # Debug: compare hemisphere vs AAM reference on the same regular grid
        ref_file = os.path.join(AAM_path, AAM_sphere)
        try:
            amp_ref, phi_ref, theta_ref, f_ref, radius_ref_ft, _, _ = fa.load_nc_sphere(ref_file)
            amp_ref = amp_ref.astype(float)
            amp_ref[np.isnan(amp_ref)] = -np.inf
            amp_ref[amp_ref > 1.0e34] = -np.inf

            radius_ref_ft = float(np.asarray(radius_ref_ft, dtype=float).ravel()[0])
            if radius_ref_ft <= 0.0:
                raise ValueError('Reference sphere radius is non-positive')

            # Match frequency band for comparison
            fmask_ref = np.logical_and(f_ref >= freqs[0], f_ref <= freqs[1])
            amp_ref = amp_ref[:, :, fmask_ref]

            # OASPL on the reference sphere grid (in ART coords)
            oaspl_ref_db = np.apply_along_axis(fa.OASPL, 2, amp_ref)

            # Convert ART(phi/theta) to UMAPR(azi/elv) for interpolation
            T_ref, P_ref = np.meshgrid(theta_ref.astype(float), phi_ref.astype(float))
            azi_ref_rad, elv_ref_rad = fa.art2umapr(np.deg2rad(P_ref), np.deg2rad(theta_ref.astype(float)))
            azi_ref_deg = np.degrees(azi_ref_rad)
            elv_ref_deg = np.degrees(elv_ref_rad)

            # Normalize reference levels to our r_ref (spherical spreading only)
            spread_power_scale = (radius_ref_ft / float(r_ref)) ** 2
            P_ref_pow = (10.0 ** (oaspl_ref_db / 10.0)) * spread_power_scale

            fazi_ref_pts = azi_ref_deg.reshape(-1)
            felv_ref_pts = elv_ref_deg.reshape(-1)
            P_ref_pts = P_ref_pow.reshape(-1)

            # Enforce azimuth periodicity for reference interpolation
            fazi_ref_ext = np.concatenate((fazi_ref_pts, fazi_ref_pts + 360.0, fazi_ref_pts - 360.0))
            felv_ref_ext = np.concatenate((felv_ref_pts, felv_ref_pts, felv_ref_pts))
            P_ref_ext = np.concatenate((P_ref_pts, P_ref_pts, P_ref_pts))

            P_ref_grid = fa.shepIDW(ELV_GRID, AZI_GRID, felv_ref_ext, fazi_ref_ext, P_ref_ext, rmax=float(hemisphere_rmax))
            ref_oaspl_full_db = 10.0 * np.log10(np.maximum(P_ref_grid, eps))
            if ref_oaspl_full_db.shape[1] > 1:
                ref_oaspl_full_db[:, -1] = ref_oaspl_full_db[:, 0]

            fig_ref_on_grid, ax_ref_on_grid, _ = fa.plot_lambert_ea(
                np.deg2rad(AZI_GRID),
                np.deg2rad(ELV_GRID),
                ref_oaspl_full_db.copy(),
                SPL_range=hemisphere_spl_range,
            )
            figs.append(fig_ref_on_grid)
            names.append('reference_hemisphere_on_grid')

            # Delta plot (custom, diverging colormap) and summary stats
            delta_db = hemi_oaspl_full_db - ref_oaspl_full_db
            finite = np.isfinite(delta_db)
            if np.any(finite):
                med = float(np.median(delta_db[finite]))
                mad = float(np.median(np.abs(delta_db[finite] - med)))
                logging.info(f'Hemisphere - reference (full-band) median={med:+.2f} dB, MAD={mad:.2f} dB (on regular grid)')

            lat = np.deg2rad(ELV_GRID)
            lon = np.deg2rad(AZI_GRID) - np.pi
            x, y = fa.lambert_ea(lat, lon)
            fig_delta = plt.figure()
            ax_delta = fig_delta.add_subplot(111)
            v = 10.0
            cs = ax_delta.contourf(x, y, np.nan_to_num(delta_db, nan=0.0), levels=np.linspace(-v, v, 21), cmap='seismic', extend='both')
            plt.colorbar(cs, ax=ax_delta, format='%.0f', label='Hemisphere - Reference (dB)')
            ax_delta.set_aspect('equal')
            ax_delta.set_xticks([])
            ax_delta.set_yticks([])
            figs.append(fig_delta)
            names.append('hemisphere_minus_reference_delta')

        except Exception as e:
            logging.warning(f'Could not compute reference comparison plots: {e}')

        # Save the gridded hemisphere product when not in interactive show mode
        if save_outputs:
            os.makedirs(args.outdir, exist_ok=True)
            out_npz = os.path.join(args.outdir, 'hemisphere_third_octave.npz')
            np.savez(
                out_npz,
                band_centers_hz=band_centers,
                azi_grid_deg=azi_grid,
                elv_grid_deg=elv_grid,
                hemisphere_bands_db=hemi_bands_db,
                hemisphere_oaspl_lt2khz_db=hemi_oaspl_db,
                hemisphere_splA_lt2khz_db=hemi_spl_a_db,
                hemisphere_oaspl_fullband_db=hemi_oaspl_full_db,
                r_ref_ft=r_ref,
                rmax_deg=hemisphere_rmax,
                point_stride=hemisphere_point_stride,
                absorption_deprop=apply_absorption_deprop,
                oaspl_fmax_hz=oaspl_fmax,
                oaspl_fullband_fmax_hz=oaspl_ref_fmax,
            )
            logging.info(f"Saved gridded hemisphere to {out_npz}")

    # Reference hemisphere
    figs.append(plot_reference_hemisphere(AAM_path, AAM_sphere, freqs, SPL_range=hemisphere_spl_range))
    names.append('reference_hemisphere')

    if save_outputs:
        save_figures(figs, args.outdir, names)
        logging.info(f"Done. Saved {len(figs)} figures to {args.outdir}")
    else:
        plt.show()

if __name__ == '__main__':
    main()