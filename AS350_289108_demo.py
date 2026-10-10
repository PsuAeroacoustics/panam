#!/usr/bin/env python3
"""AS350 run 289108 demo: depropagate one 2017 Noise Abatement run to a hemisphere and
compare it with the reference AAM sphere for the same run.

Needs the AS350 demo data (the ``as350_demo`` entry of local_paths, or --basepath:
the directory holding ``Acoustic/``, ``Ambient/``, ``AAM/`` and ``Tracking/``), which is not
distributed with PANAM.  Writes its figures, the gridded hemisphere
(``hemisphere_third_octave.npz``, which ``array_planner.py overlay`` reads) and an
AAM-style sphere to --outdir.
"""

import os
import argparse
import flight_acoustics as fa
import local_paths
import noise_abatement_2017 as na
import matplotlib.pyplot as plt
import numpy as np
import logging
from netCDF4 import Dataset


def compare_aam_netcdf_fields(reference_nc, candidate_nc):
    """Log a basic comparison of variable presence, shapes, and 'unit' attrs."""

    with Dataset(reference_nc, 'r') as ref, Dataset(candidate_nc, 'r') as cand:
        ref_vars = set(ref.variables.keys())
        cand_vars = set(cand.variables.keys())

        missing = sorted(ref_vars - cand_vars)
        extra = sorted(cand_vars - ref_vars)
        if missing:
            logging.warning(f'Candidate is missing variables present in reference: {missing}')
        if extra:
            logging.info(f'Candidate has extra variables not in reference: {extra}')

        common = sorted(ref_vars & cand_vars)
        unit_mismatch = []
        shape_mismatch = []
        dtype_mismatch = []
        for name in common:
            rv = ref.variables[name]
            cv = cand.variables[name]
            if rv.shape != cv.shape:
                shape_mismatch.append((name, rv.shape, cv.shape))
            if str(rv.dtype) != str(cv.dtype):
                dtype_mismatch.append((name, str(rv.dtype), str(cv.dtype)))
            ru = getattr(rv, 'unit', None)
            cu = getattr(cv, 'unit', None)
            if (ru is not None or cu is not None) and (str(ru).strip() != str(cu).strip()):
                unit_mismatch.append((name, ru, cu))

        if shape_mismatch:
            logging.info('Shape differences for common variables:')
            for name, rsh, csh in shape_mismatch:
                logging.info(f'  {name}: ref{rsh} vs cand{csh}')
        if dtype_mismatch:
            logging.info('Dtype differences for common variables:')
            for name, rd, cd in dtype_mismatch:
                logging.info(f'  {name}: ref {rd} vs cand {cd}')
        if unit_mismatch:
            logging.info("Unit attribute differences ('unit') for common variables:")
            for name, ru, cu in unit_mismatch:
                logging.info(f'  {name}: ref unit={ru!r} vs cand unit={cu!r}')

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

def load_track(basepath):
    """The run's track and its filtered, decimated segment, with ``vz_up``.

    The 2017 tracking files store ``vz`` positive DOWN while ``z`` is positive
    up (:func:`noise_abatement_2017.vz_sign` checks each file against dz/dt),
    so the velocity handed on is (vx, vy, vz_up).
    """
    track = fa.load_NASA_track(os.path.join(basepath, 'Tracking', '289108AC.csv'))
    track['vz_up'] = na.vz_sign(track) * track['vz']
    filter_track = fa.filter_track(track, xlims=(-4000, 0), zlims=(50, 1500), decimate=10)
    return track, filter_track


def track_kinematics(filter_track):
    """Source positions and up-positive velocities, (n, 3) each."""
    source = np.column_stack((filter_track['x'], filter_track['y'], filter_track['z']))
    velocity = np.column_stack((filter_track['vx'], filter_track['vy'], filter_track['vz_up']))
    return source, velocity


def demo_atmosphere():
    """The demo's atmosphere and its speed of sound in ft/s, from that atmosphere."""
    atmosphere = fa.Atmosphere(temperature=293.15, pressure=101.325, relative_humidity=20.0)
    return atmosphere, na.sound_speed_ft_s(atmosphere)


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

def oaspl_grids(band_centers, bands_db, freq_range):
    """OASPL and A-weighted SPL grids summed from third-octave band levels.

    Parameters
    ----------
    band_centers : array_like
        Band center frequencies (Hz), shape (n_bands,)
    bands_db : ndarray
        Band levels (dB), shape (n_bands, ...)
    freq_range : tuple of float
        (fmin, fmax) in Hz; the band-limited levels sum the bands whose
        centers lie in it.

    Returns
    -------
    dict
        ``oaspl_db`` and ``spla_db``: OASPL and A-weighted SPL of the bands in
        ``freq_range``; ``oaspl_fullband_db``: OASPL of every band;
        ``oaspl_fmax_hz``: ``freq_range[1]``; ``oaspl_fullband_fmax_hz``: the
        highest band center.  A -inf band (no energy) adds nothing to a sum,
        and a cell with no data (NaN) stays NaN.
    """
    band_centers = np.asarray(band_centers, dtype=float)
    band_mask = np.logical_and(band_centers >= float(freq_range[0]), band_centers <= float(freq_range[1]))
    if not np.any(band_mask):
        raise ValueError('No third-octave bands fall within hemisphere_freq_range_hz')

    P_plot = np.power(10.0, bands_db[band_mask, ...] / 10.0)
    Aweight = fa.dBAw(band_centers[band_mask])
    Aweight = Aweight.reshape((-1,) + (1,) * (np.ndim(bands_db) - 1))
    P_plot_A = np.power(10.0, (bands_db[band_mask, ...] + Aweight) / 10.0)
    P_full = np.power(10.0, bands_db / 10.0)
    return dict(
        oaspl_db=fa.power_to_db(np.sum(P_plot, axis=0)),
        spla_db=fa.power_to_db(np.sum(P_plot_A, axis=0)),
        oaspl_fullband_db=fa.power_to_db(np.sum(P_full, axis=0)),
        oaspl_fmax_hz=float(freq_range[1]),
        oaspl_fullband_fmax_hz=float(np.max(band_centers)),
    )


def save_figures(figs, outdir, names):
    os.makedirs(outdir, exist_ok=True)
    for fig, name in zip(figs, names):
        path = os.path.join(outdir, f"{name}.png")
        fig.savefig(path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved {path}")

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



    parser = argparse.ArgumentParser(
        description='AS350 run 289108 demo: depropagate one 2017 Noise Abatement run to a hemisphere '
                    'and compare it with the reference AAM sphere for the same run.')
    parser.add_argument('--show', action='store_true',
                        help='Open the plots in windows instead of saving them; nothing is written.')
    parser.add_argument('--outdir', default='demo_plots',
                        help='Directory for the figures, hemisphere_third_octave.npz and the AAM-style '
                             'sphere (default demo_plots).')
    parser.add_argument('--basepath', default=None,
                        help='Base path to demo data (default: the as350_demo entry of local_paths).')
    args = parser.parse_args()

    save_outputs = not args.show

    basepath = local_paths.data_path('as350_demo', args.basepath)
    acoustics_path = os.path.join(basepath, 'Acoustic')
    ambient_path = os.path.join(basepath, 'Ambient')
    AAM_path = os.path.join(basepath, 'AAM')
    AAM_sphere = 'AS350B3108.nc'
    ref_file = os.path.join(AAM_path, AAM_sphere)

    # Hemisphere processing parameters
    build_hemisphere = True
    apply_absorption_deprop = True
    hemisphere_azi_step = 10.0  # deg
    hemisphere_elv_step = 10.0  # deg
    hemisphere_rmax = 25.0  # deg (Shepard neighborhood radius)
    hemisphere_point_stride = 1  # decimate emission-time samples for speed
    # Consistent color scaling for hemisphere contour plots
    hemisphere_spl_range = (80.0, 110.0)
    # Frequency range used for plotting/metrics (but not necessarily for export)
    hemisphere_freq_range_hz = (0.0, 2000.0)
    freqs = list(hemisphere_freq_range_hz)

    # Export hemisphere over all frequency bands (match reference AAM sphere if available)
    export_freq_range_hz = None
    export_third_octave_fmin_hz = 20.0
    export_band_centers_hz = None
    try:
        _, _, _, f_ref, _, _, _ = fa.load_nc_sphere(ref_file)
        f_ref = np.asarray(f_ref, dtype=float)
        if f_ref.size >= 2 and np.all(np.isfinite(f_ref)):
            # The FFT range must reach the outer bands' edges, not their centers,
            # or the lowest and highest bands lose half their width (-3 dB).
            band_lower, band_upper = fa.third_octave_band_edges(f_ref)
            export_freq_range_hz = (float(np.min(band_lower)), float(np.max(band_upper)))
            export_third_octave_fmin_hz = float(np.min(f_ref))
            export_band_centers_hz = f_ref
            logging.info(f'Exporting hemisphere over reference band range: {export_freq_range_hz[0]:.1f}..{export_freq_range_hz[1]:.1f} Hz')
    except Exception as e:
        logging.warning(f'Could not read reference FREQUENCY grid; exporting only over plot range. Reason: {e}')
    r_ref = 100.0  # Reference distance for depropagation in feet

    # The track frame is right-handed and z-up, so no mirroring.  This was True
    # until 2026-09-24, compensating for art2umapr's reversed lateral sign.
    flip_y_for_geometry = False

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
    track, filter_track = load_track(basepath)
    traj_figs = plot_trajectory(track, filter_track)
    figs.extend(traj_figs)
    names.extend(['trajectory_xz', 'trajectory_xy'])

    # Array plot
    figs.append(plot_array(xs, ys, xf, yf, track, filter_track))
    names.append('microphone_array')

    # Kinematics inputs for hemisphere generation
    source, velocity = track_kinematics(filter_track)

    # Build a hemisphere and interpolate onto a regular spherical grid
    if build_hemisphere:
        # Pressure correction: adjust for pressure doubling at the ground board.
        pressures_corrected = [np.asarray(p, dtype=float) * 0.5 for p in filtered_pressures]

        atm, speed_of_sound_ft_s = demo_atmosphere()
        hemi = fa.depropagate_hemisphere(
            mic_locations=filtered_miclocs,
            pressure=pressures_corrected,
            time=filtered_times,
            track_time=filter_track['time'],
            track_position=source,
            track_velocity=velocity,
            speed_of_sound=speed_of_sound_ft_s,
            length_units='ft',
            r_ref=r_ref,
            freq_range=export_freq_range_hz if export_freq_range_hz is not None else hemisphere_freq_range_hz,
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
            third_octave_fmin=float(export_third_octave_fmin_hz),
            third_octave_band_centers_hz=export_band_centers_hz,
        )

        band_centers = np.asarray(hemi['third_octave']['band_centers_hz'], dtype=float)
        hemi_bands_db = hemi['third_octave']['bands_db']
        levels = oaspl_grids(band_centers, hemi_bands_db, hemisphere_freq_range_hz)
        hemi_oaspl_db = levels['oaspl_db']
        hemi_spl_a_db = levels['spla_db']
        hemi_oaspl_full_db = levels['oaspl_fullband_db']

        azi_grid = hemi['azi_grid_deg']
        elv_grid = hemi['elv_grid_deg']
        AZI_GRID, ELV_GRID = np.meshgrid(azi_grid, elv_grid)

        # Coverage plot (uses scattered az/el from hemisphere samples)
        figs.append(plot_array_coverage(hemi['scattered']['azi_deg'], hemi['scattered']['elv_deg']))
        names.append('array_coverage_lambert')

        # Plot OASPL integrated below 2 kHz using flight_acoustics plotting helper
        fig_oaspl, ax_oaspl, _ = fa.plot_lambert_ea(
            np.deg2rad(AZI_GRID),
            np.deg2rad(ELV_GRID),
            hemi_oaspl_db.copy(),
            SPL_range=hemisphere_spl_range,
            grid_convention='umapr',
        )
        figs.append(fig_oaspl)
        names.append('hemisphere_oaspl_lt2khz_lambert')

        fig_oaspl_art, ax_oaspl_art, _ = fa.plot_lambert_ea(
            np.deg2rad(AZI_GRID),
            np.deg2rad(ELV_GRID),
            hemi_oaspl_db.copy(),
            SPL_range=hemisphere_spl_range,
            grid_convention='art',
        )
        figs.append(fig_oaspl_art)
        names.append('hemisphere_oaspl_lt2khz_lambert_art_grid')

        # Plot full-band OASPL (every exported band)
        fig_oaspl_full, ax_oaspl_full, _ = fa.plot_lambert_ea(
            np.deg2rad(AZI_GRID),
            np.deg2rad(ELV_GRID),
            hemi_oaspl_full_db.copy(),
            SPL_range=hemisphere_spl_range,
        )
        figs.append(fig_oaspl_full)
        names.append('hemisphere_oaspl_fullband_lambert')

        # Debug: compare hemisphere vs AAM reference on the same regular grid
        try:
            amp_ref, phi_ref, theta_ref, f_ref, radius_ref_ft, _, _ = fa.load_nc_sphere(ref_file)
            amp_ref = amp_ref.astype(float)
            amp_ref = fa.mask_missing_levels(amp_ref)

            radius_ref_ft = float(np.asarray(radius_ref_ft, dtype=float).ravel()[0])
            if radius_ref_ft <= 0.0:
                raise ValueError('Reference sphere radius is non-positive')

            # Match frequency band for comparison (plot/metrics range, as hemi_oaspl_db)
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
            ref_oaspl_db = fa.power_to_db(P_ref_grid)
            if ref_oaspl_db.shape[1] > 1:
                ref_oaspl_db[:, -1] = ref_oaspl_db[:, 0]

            fig_ref_on_grid, ax_ref_on_grid, _ = fa.plot_lambert_ea(
                np.deg2rad(AZI_GRID),
                np.deg2rad(ELV_GRID),
                ref_oaspl_db.copy(),
                SPL_range=hemisphere_spl_range,
            )
            figs.append(fig_ref_on_grid)
            names.append('reference_hemisphere_on_grid')

            # Delta plot (custom, diverging colormap) and summary stats
            # Compare in the plotting band range.
            delta_db = hemi_oaspl_db - ref_oaspl_db
            finite = np.isfinite(delta_db)
            if np.any(finite):
                med = float(np.median(delta_db[finite]))
                mad = float(np.median(np.abs(delta_db[finite] - med)))
                logging.info(f'Hemisphere - reference (<{levels["oaspl_fmax_hz"] / 1000:g} kHz) median={med:+.2f} dB, MAD={mad:.2f} dB (on regular grid)')

            lat = np.deg2rad(ELV_GRID)
            lon = fa.lambert_lon(np.deg2rad(AZI_GRID))
            x, y = fa.lambert_ea(lat, lon)
            fig_delta = plt.figure()
            ax_delta = fig_delta.add_subplot(111)
            v = 10.0
            # Cells missing on either side are left blank, not drawn as no difference.
            cs = ax_delta.contourf(x, y, np.ma.masked_invalid(delta_db), levels=np.linspace(-v, v, 21), cmap='seismic', extend='both')
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
                oaspl_fmax_hz=levels['oaspl_fmax_hz'],
                oaspl_fullband_fmax_hz=levels['oaspl_fullband_fmax_hz'],
            )
            logging.info(f"Saved gridded hemisphere to {out_npz}")

            # Export AAM-style netCDF hemisphere using the reference PHI/THETA grids
            out_nc = os.path.join(args.outdir, 'hemisphere_third_octave_aam.nc')
            try:
                # Use reference ART grids when available
                try:
                    _, phi_ref, theta_ref, _, radius_ref_ft, speed_ref_knots, fpa_ref_deg = fa.load_nc_sphere(ref_file)
                    phi_export = np.asarray(phi_ref, dtype=float)
                    theta_export = np.asarray(theta_ref, dtype=float)
                    radius_export_ft = float(np.asarray(radius_ref_ft, dtype=float).ravel()[0])
                    speed_export_knots = float(np.asarray(speed_ref_knots, dtype=float).ravel()[0])
                    fpa_export_deg = float(np.asarray(fpa_ref_deg, dtype=float).ravel()[0])
                except Exception:
                    phi_export = None
                    theta_export = None
                    radius_export_ft = r_ref
                    # Estimate flight condition from track velocity (z up)
                    v = velocity.astype(float)
                    sp_ft_s = np.sqrt(np.sum(v ** 2, axis=1))
                    speed_export_knots = float(np.nanmedian(sp_ft_s) / 1.6878098571011957)
                    horiz = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2)
                    fpa_export_deg = float(np.degrees(np.nanmedian(np.arctan2(v[:, 2], horiz))))

                fa.write_aam_hemisphere_netcdf(
                    out_nc,
                    hemi,
                    mode='third_octave',
                    phi_deg=phi_export,
                    theta_deg=theta_export,
                    radius_ft=radius_export_ft,
                    speed_knots=speed_export_knots,
                    flight_path_angle_deg=fpa_export_deg,
                    title='AS350 depropagated hemisphere (third-octave)',
                    overwrite=True,
                )
                logging.info(f"Saved AAM netCDF hemisphere to {out_nc}")

                # Plot the exported hemisphere using the same method as the reference
                try:
                    fig_exported = plot_reference_hemisphere(args.outdir, os.path.basename(out_nc), freqs, SPL_range=hemisphere_spl_range)
                    figs.append(fig_exported)
                    names.append('exported_hemisphere')
                except Exception as e:
                    logging.warning(f'Could not plot exported AAM hemisphere: {e}')

                # Compare fields/attrs against the reference AAM sphere
                if os.path.exists(ref_file):
                    compare_aam_netcdf_fields(ref_file, out_nc)
            except Exception as e:
                logging.warning(f'Could not export AAM netCDF hemisphere: {e}')

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