"""Ground footprints: spheres projected onto flat ground.

:func:`project_sphere` carries a sphere's levels along straight rays to the
ground (spreading and absorption, no ground effect); :func:`plot_projection`
contours one footprint and :func:`fried_egg_plot` contours a directory of
them over flight condition.  :func:`project_directory` computes the per-run data behind it (each run's
maximum and mean ground level with its flight condition); :func:`is_noisy` and :func:`data_filter` mark and
cull the loudest runs.  Altitudes are in m.
"""

import os
import re
import warnings
from glob import glob

from matplotlib import tri
from matplotlib.pyplot import subplots, colorbar
import numpy as np

from panam import unit_conversion
from panam.acoustics.atmosphere import Atmosphere
from panam.acoustics.plotting import acoustic_plot_style
from panam.database import extract_SPL, read_vehicle_data, _finite_sphere_levels
from panam.lambert_plots import colorbar_ticks, get_ylorrd_cmap, nice_levels


def project_sphere(filename, altitude, elv_cutoff, infreqs=None,
                   atmosphere=Atmosphere(temperature=293.15, pressure=101.325,
                                         relative_humidity=20.0)):
    """A sphere's A-weighted levels projected onto flat ground ``altitude`` below it.

    Each direction at least ``elv_cutoff`` deg below the horizon is carried along a
    straight ray from the sphere's radius to the ground, with spherical spreading and
    the sphere's absorption.  ``elv_cutoff`` must lie strictly between 0 and 90 deg: a
    ray at or above the horizon never reaches the ground.

    ``altitude`` is in meters: :func:`extract_SPL` converts the sphere's radius
    from feet to meters, and the spreading and absorption compare the two.

    Returns:
        (x, y, LA, speed, flight_path_angle): ground positions in meters,
        starboard +x and the flight direction +y, and the level there (dBA; -inf
        where the direction has no energy).
    """
    if not 0.0 < elv_cutoff < 90.0:
        raise ValueError(f'elv_cutoff must be between 0 and 90 deg below the horizon, not {elv_cutoff}')
    distance = 1000  # reference distance for EAA
    azi, elv, _, _, radius, _, SPLA, EAA, speed, flight_path_angle, *_ = extract_SPL(
        filename, infreqs, distance, atmosphere)
    # A direction with no energy (every band missing) has SPLA = -inf and an EAA
    # of -inf - -inf = NaN; give it EAA = 0, as the database does, so its ground
    # level is -inf (no energy) rather than NaN.
    SPLA, EAA = _finite_sphere_levels(SPLA, EAA)
    # Discard values outside of elevation cutoff
    included_angles = elv >= np.radians(elv_cutoff)
    azi = azi[included_angles]
    elv = elv[included_angles]
    SPLA = SPLA[included_angles]
    EAA = EAA[included_angles]
    slant_range = altitude / np.sin(elv)
    ground_range = np.sqrt(slant_range ** 2 - altitude ** 2)
    # Plan view from above, as lambert_lon draws hemispheres: the flight
    # direction (azimuth 180) is +y and starboard (azimuth 90) is +x.  This was
    # -sin(azi) until 2026-09, which with art2umapr's old phi sign canceled out;
    # once art2umapr put phi > 0 to starboard it mirrored every footprint.
    x = ground_range * np.sin(azi)
    y = -ground_range * np.cos(azi)
    absorption = EAA / distance * (slant_range - radius)
    spreading = 20 * np.log10(radius / slant_range)
    LA = SPLA - absorption + spreading
    return x, y, LA, speed, flight_path_angle


@acoustic_plot_style
def plot_projection(filename, altitude=500, cutoff=30, infreqs=None, units='m'):
    """
    Generate a contour plot of sound pressure levels projected onto a ground plane.

    This function computes the acoustic footprint of an aircraft at a specified altitude
    by projecting sound pressure levels onto a horizontal plane and creating a contour plot.

    Args:
        filename (str): Path to the file containing netCDF formatted acoustic sphere data
        altitude (float, optional): Aircraft altitude in meters. Defaults to 500.
        cutoff (float, optional): Elevation cutoff (deg below the horizon, strictly
            between 0 and 90): only directions at least this far below the horizon are
            projected.  Defaults to 30.
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
        - The plot uses a YlOrRd colormap on round contour levels that cover the data
          (see :func:`nice_levels`)
        - Coordinate system: cross-track (x-axis) and along-track (y-axis) directions
    """
    x, y, LA, _, _ = project_sphere(filename, altitude, cutoff, infreqs)
    if x.size < 3:
        raise ValueError(f'{filename}: {x.size} directions at least {cutoff} deg below the horizon; '
                         'a footprint needs 3 (lower the cutoff)')
    if not np.any(np.isfinite(LA)):
        raise ValueError(f'{filename}: no direction at least {cutoff} deg below the horizon has a level')
    fig, ax = subplots(facecolor='white')
    xi = np.linspace(np.min(x), np.max(x), 100)
    yi = np.linspace(np.min(y), np.max(y), 100)
    # Directions with no level (no energy) stay in the triangulation as NaN,
    # so the triangles touching them are left blank instead of interpolated
    # across, and they do not set the color scale.
    LA = np.where(np.isfinite(LA), LA, np.nan)
    triangles = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triangles, LA)
    xim, yim = np.meshgrid(xi, yi)
    li = interpolator(xim, yim)
    # Round levels with the ends snapped outward, so neither the loudest spot nor the
    # quietest is left blank (contourf fills only between the first and last level).
    levels = nice_levels(np.nanmin(LA), np.nanmax(LA))
    color_map = get_ylorrd_cmap(max(len(levels), 2))
    xi = unit_conversion.len_conv(xi, from_units='m', to_units=units)
    yi = unit_conversion.len_conv(yi, from_units='m', to_units=units)
    cs = ax.contourf(xi, yi, li, levels=levels, cmap=color_map)
    ax.axis('equal')
    ax.set_xlabel('Cross Track Direction, ' + units)
    ax.set_ylabel('Flight Track Direction, ' + units)
    cb = colorbar(cs, ticks=colorbar_ticks(levels))
    cb.set_label('Sound Pressure Level, dBA')
    return fig, ax, cs


def project_directory(directory_name, altitude=500, cutoff=30, input_frequencies=None, fpa_climb_cutoff=5,
                      atmosphere=Atmosphere(temperature=293.15, pressure=101.325,
                                            relative_humidity=20.0), duration_correction=None):
    """Project every ``*.nc`` sphere in ``directory_name`` and reduce each run to its ground levels.

    Each sphere is projected with :func:`project_sphere`, and each run with a flight-path angle at most
    ``fpa_climb_cutoff`` deg is reduced to its maximum and mean ground level (dBA) over the directions that
    reach the ground with a level; runs that climb more steeply are left out.  A run with no such direction
    is skipped with a warning.  ``altitude`` is
    in m and ``cutoff`` in deg below the horizon.  ``duration_correction`` is a reference speed (kt) that adds
    10 lg(reference / speed) dB to both levels; None (the default) adds nothing.

    Returns:
        (speeds, flight_path_angles, Lmax, Lmean, advance_ratios, weight_coefficients,
        hover_tip_mach_numbers, alphas, runs, number_of_main_rotor_blades, number_of_tail_rotor_blades,
        nondimensional_flat_plate_drag, main_rotor_tip_speed): speeds in kt, flight-path angles in deg,
        levels in dBA, run numbers (the file name's last three digits, None when it has none), and the
        condition entries of :func:`database.read_vehicle_data` (angles of attack in deg, tip speed in
        m/s).  When ``vehicle.cfg`` has an ``[Option]`` section, the speeds are the reference list's
        airspeeds.
    """
    # Get list of full paths to netCDF files in directory
    local_glob = os.path.expanduser(directory_name) + '/*.nc'
    absolute_glob = os.path.abspath(local_glob)
    file_list = sorted(glob(absolute_glob))
    speeds = []
    flight_path_angles = []
    Lmax = []
    Lmean = []
    runs = []
    for filename in file_list:
        x, y, LA, speed, flight_path_angle = project_sphere(filename, altitude, cutoff, input_frequencies, atmosphere)
        if flight_path_angle <= fpa_climb_cutoff:
            # Directions with no energy (-inf) carry no level to take a max or
            # mean of; masked arrays used to hide them from these reductions.
            stencil = (np.sqrt(x ** 2 + y ** 2) > 0) & np.isfinite(LA)
            if not np.any(stencil):
                warnings.warn('{}: no direction below the aircraft has a level; skipping it'.format(filename))
                continue
            speeds.append(speed)
            flight_path_angles.append(flight_path_angle)
            Lmax.append(np.max(LA[stencil]))
            Lmean.append(np.mean(LA[stencil]))
            # The run number is the last three digits; any before them belong
            # to the vehicle's name (Be407100.nc is the Bell 407's run 100).
            run = re.search(r'(\d{3})\.nc$', os.path.basename(filename))
            runs.append(int(run.group(1)) if run else None)
    speeds = np.array(speeds)
    flight_path_angles = np.array(flight_path_angles)
    Lmax = np.array(Lmax)
    Lmean = np.array(Lmean)
    # Apply duration correction if a reference speed is specified
    if duration_correction is not None:
        Lmax = Lmax + 10 * np.log10(duration_correction / speeds)
        Lmean = Lmean + 10 * np.log10(duration_correction / speeds)
    runs = np.array(runs)
    # Get nondimensional condition information.  The rotor radii and areas and the
    # tail-rotor tip speed (entries 0, 1, 4, 5 and 6) are not returned.
    (_, _, main_rotor_tip_speed, number_of_main_rotor_blades,
     _, _, _, number_of_tail_rotor_blades,
     advance_ratios, weight_coefficients, hover_tip_mach_numbers, alphas, nondimensional_flat_plate_drag,
     speeds) = read_vehicle_data(directory_name, runs, speeds, flight_path_angles)

    return (speeds, flight_path_angles, Lmax, Lmean, advance_ratios, weight_coefficients,
            hover_tip_mach_numbers, alphas, runs, number_of_main_rotor_blades, number_of_tail_rotor_blades,
            nondimensional_flat_plate_drag, main_rotor_tip_speed)


def is_noisy(level, threshold=0.65):
    """True where ``level`` lies above ``threshold`` of the way from its minimum to its maximum.

    ``threshold`` is a fraction (0 to 1); ``level`` may be in any unit (dBA in :func:`fried_egg_plot`).
    """
    cutoff = (np.max(level) - np.min(level)) * threshold + np.min(level)
    noisy = level > cutoff
    return noisy


def data_filter(levels, flight_path_angles, threshold=0.65, cull_noisy_fpa=-1.0):
    """Mask (True = keep) dropping the runs that :func:`is_noisy` marks and that fly above ``cull_noisy_fpa``.

    ``flight_path_angles`` and ``cull_noisy_fpa`` are in deg; ``levels`` and ``threshold`` are as for
    :func:`is_noisy`.
    """
    cull_index = np.logical_and(is_noisy(levels, threshold), flight_path_angles > cull_noisy_fpa)
    mask = np.logical_not(cull_index)
    return mask


@acoustic_plot_style
def fried_egg_plot(directory_name, metric='mean', dimensionless=False, altitude=500, cutoff=30,
                   input_frequencies=None, fpa_climb_cutoff=5,
                   atmosphere=Atmosphere(temperature=293.15, pressure=101.325, relative_humidity=20.0),
                   climb_rates=False, duration_correction=None, threshold=0.65, cull_noisy_fpa=None,
                   suppress_classification=False):
    """Contour a directory's ground levels over flight condition.

    Contours the mean (``metric='mean'``) or else the maximum ground level (dBA) of
    :func:`project_directory` over speed (kt) and flight-path angle (deg), over speed and climb rate (fpm)
    with ``climb_rates``, or over advance ratio and angle of attack (deg) with ``dimensionless``.  Runs that
    :func:`is_noisy` marks by ``threshold`` are drawn red and the rest black, unless
    ``suppress_classification`` or ``threshold`` is None.  With ``cull_noisy_fpa`` (deg) and ``threshold``
    set, :func:`data_filter` first drops the noisy runs that fly above it.
    The other arguments are as for :func:`project_directory`.

    Returns:
        (fig, ax, cs): the figure, its axes and the contour set.
    """
    # The weight coefficients, hover tip Mach numbers and the entries after alphas (run numbers, blade
    # counts, flat-plate drag, tip speed) are not used here.
    (speeds, flight_path_angles, Lmax, Lmean, advance_ratios, _, _, alphas,
     *_) = project_directory(directory_name, altitude, cutoff, input_frequencies, fpa_climb_cutoff,
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

    # Classify on every run, before any are culled, so the points marked
    # noisy are judged by the same cutoff that did the culling.
    noisy_index = is_noisy(Lmetric, threshold) if threshold is not None else None
    if cull_noisy_fpa is not None and threshold is not None:
        mask = data_filter(levels=Lmetric, flight_path_angles=flight_path_angles, threshold=threshold,
                           cull_noisy_fpa=cull_noisy_fpa)
        x = x[mask]
        y = y[mask]
        Lmetric = Lmetric[mask]
        noisy_index = noisy_index[mask]

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
