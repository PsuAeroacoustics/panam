"""Lambert equal-area maps of sphere levels and of sampled directions.

:func:`plot_lambert_ea` maps levels over (azimuth, elevation),
:func:`nc_lambert_ea` maps a sphere file, and :func:`lambert_ea_points` draws
points on the same axes.  ``grid_convention`` labels the grid in UMAPR or ART
angles.

:func:`nice_levels`, :func:`colorbar_ticks` and :func:`get_ylorrd_cmap` choose the round dB contour levels,
colorbar ticks and yellow-orange-red colormap for these maps and for the footprint and ground-effect plots.
Importing the module loads neither pyplot nor netCDF4: the plots import pyplot when called, and
:func:`nc_lambert_ea` imports :func:`database.extract_SPL` when called.
"""

import matplotlib
import numpy as np

from panam.acoustics.plotting import acoustic_plot_style
from panam.geometry import art2umapr, lambert_ea, lambert_lon, _lambert_xy_to_umapr, _umapr2art


def _normalize_lambert_grid_convention(grid_convention):
    if grid_convention is None:
        return 'umapr'

    normalized = str(grid_convention).strip().lower()
    aliases = {
        'umapr': 'umapr',
        'art': 'art',
        'aam': 'art',
        'rnm': 'art',
    }
    if normalized not in aliases:
        raise ValueError("grid_convention must be one of 'umapr', 'art', 'aam', or 'rnm'")
    return aliases[normalized]


def _configure_lambert_axes(ax):
    radius = np.sqrt(2.0)
    ax.set_aspect('equal', adjustable='box')
    if hasattr(ax, 'set_box_aspect'):
        ax.set_box_aspect(1.0)
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_lambert_grid(ax, grid_convention='umapr'):
    grid_convention = _normalize_lambert_grid_convention(grid_convention)

    if grid_convention == 'umapr':
        meridians = np.arange(0, 360, 45)
        for meridian in meridians:
            lats = np.linspace(0, 0.5 * np.pi, 1000)
            lons = lambert_lon(np.deg2rad(meridian)) * np.ones(len(lats))
            xm, ym = lambert_ea(lats, lons)
            (line,) = ax.plot(xm, ym, 'k--')
            line.set_gid(f'lambert-grid-umapr-meridian-{int(meridian)}')
            xl, yl = lambert_ea(np.deg2rad(-11.0), lambert_lon(np.deg2rad(meridian)))
            text = ax.text(
                xl,
                yl,
                "%d°" % int(meridian),
                horizontalalignment='center',
                verticalalignment='center',
            )
            text.set_gid(f'lambert-grid-umapr-meridian-label-{int(meridian)}')

        parallels = np.arange(0, 90, 30)
        for parallel in parallels:
            lons = np.linspace(-np.pi, np.pi, 1000)
            lats = np.deg2rad(parallel) * np.ones(len(lons))
            xm, ym = lambert_ea(lats, lons)
            (line,) = ax.plot(xm, ym, 'k--')
            line.set_gid(f'lambert-grid-umapr-parallel-{int(parallel)}')
            xl, yl = lambert_ea(np.deg2rad(parallel + 7.5), np.deg2rad(0.0))
            text = ax.text(
                xl + 0.025,
                yl - 0.025,
                "%d°" % parallel,
                horizontalalignment='left',
                verticalalignment='center',
            )
            text.set_gid(f'lambert-grid-umapr-parallel-label-{int(parallel)}')
    else:
        theta_curve = np.deg2rad(np.linspace(0.0, 180.0, 1000))
        phi_curve = np.deg2rad(np.linspace(-90.0, 90.0, 1000))
        label_bbox = dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85)

        for phi_deg in np.arange(-90, 91, 30):
            phi_line = np.deg2rad(phi_deg) * np.ones_like(theta_curve)
            azi_line, elv_line = art2umapr(phi_line, theta_curve)
            xm, ym = lambert_ea(elv_line, lambert_lon(azi_line))
            (line,) = ax.plot(xm, ym, 'k--')
            line.set_gid(f'lambert-grid-art-phi-{int(phi_deg)}')

            azi_label, elv_label = art2umapr(
                np.array([np.deg2rad(phi_deg)]),
                np.array([np.deg2rad(120.0)]),
            )
            xl, yl = lambert_ea(elv_label, lambert_lon(azi_label))
            if phi_deg > 0:
                label = f'+{int(phi_deg)}°'
            elif phi_deg < 0:
                label = f'{int(phi_deg)}°'
            else:
                label = '0°'

            x_label = float(xl[0])
            y_label = float(yl[0])
            r_label = np.hypot(x_label, y_label)
            if r_label > 0.0:
                x_label += 0.06 * x_label / r_label
                y_label += 0.06 * y_label / r_label

            text = ax.text(
                x_label,
                y_label,
                label,
                horizontalalignment='left' if phi_deg >= 0 else 'right',
                verticalalignment='center',
                bbox=label_bbox,
                clip_on=False,
                zorder=10,
            )
            text.set_gid(f'lambert-grid-art-phi-label-{int(phi_deg)}')

        for theta_deg in np.arange(30, 180, 30):
            theta_line = np.deg2rad(theta_deg) * np.ones_like(phi_curve)
            azi_line, elv_line = art2umapr(phi_curve, theta_line)
            xm, ym = lambert_ea(elv_line, lambert_lon(azi_line))
            (line,) = ax.plot(xm, ym, 'k--')
            line.set_gid(f'lambert-grid-art-theta-{int(theta_deg)}')

            azi_label, elv_label = art2umapr(
                np.array([np.deg2rad(80.0)]),
                np.array([np.deg2rad(theta_deg)]),
            )
            xl, yl = lambert_ea(elv_label, lambert_lon(azi_label))
            text = ax.text(
                float(xl[0]) + 0.08,
                float(yl[0]),
                f'{int(theta_deg)}°',
                horizontalalignment='left',
                verticalalignment='center',
                bbox=label_bbox,
                clip_on=False,
                zorder=10,
            )
            text.set_gid(f'lambert-grid-art-theta-label-{int(theta_deg)}')

        top_text = ax.text(
            0.0,
            np.sqrt(2.0) + 0.07,
            '0°',
            horizontalalignment='center',
            verticalalignment='bottom',
            bbox=label_bbox,
            clip_on=False,
            zorder=10,
        )
        top_text.set_gid('lambert-grid-art-theta-label-0')
        bottom_text = ax.text(
            0.0,
            -np.sqrt(2.0) - 0.07,
            '180°',
            horizontalalignment='center',
            verticalalignment='top',
            bbox=label_bbox,
            clip_on=False,
            zorder=10,
        )
        bottom_text.set_gid('lambert-grid-art-theta-label-180')

    def format_coord(x, y):
        cazi, celv = _lambert_xy_to_umapr(x, y)
        if grid_convention == 'art':
            cphi, ctheta = _umapr2art(cazi, celv)
            return 'φ = %0.1f, θ = %0.1f' % (np.degrees(cphi), np.degrees(ctheta))
        return 'ψ = %0.1f, θ = %0.1f' % (np.degrees(cazi), np.degrees(celv))

    ax.format_coord = format_coord


def nice_levels(vmin, vmax, target=9, step=None,
                steps=(0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 20.0, 25.0, 50.0)):
    """Contour levels on round values (0.5/1/2/2.5/5/10... dB) covering the data.

    Picks the smallest allowed step giving <= `target` intervals, then snaps
    the ends outward to multiples of it, so both the contour bands and the
    colorbar ticks land on values a reader can name. Pass `step` to force
    one (e.g. step=5 for even 5 dB bands).
    """
    vmin, vmax = float(vmin), float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    if step is None:
        raw = (vmax - vmin) / max(target - 1, 1)
        mag = 10.0 ** np.floor(np.log10(raw)) if raw > 0 else 1.0
        step = next((s * mag for s in steps if s * mag >= raw * 0.999),
                    steps[-1] * mag)
    lo = np.floor(vmin / step) * step
    hi = np.ceil(vmax / step) * step
    return np.arange(lo, hi + 0.5 * step, step)


def colorbar_ticks(levels, max_ticks=11):
    """Thin a level list down to at most `max_ticks` evenly spaced ticks."""
    levels = np.asarray(levels, dtype=float)
    if levels.size <= max_ticks:
        return levels
    return levels[:: int(np.ceil(levels.size / max_ticks))]


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
    return matplotlib.colormaps['YlOrRd']


@acoustic_plot_style
def plot_lambert_ea(azi, elv, SPL, SPL_range=None, weight=None, grid_convention='umapr',
                    levels=None, level_step=None, cmap=None, colorbar_label=None):
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
    grid_convention : str, optional
        Grid overlay convention. Use 'umapr' for the existing azimuth/elevation grid or
        'art'/'aam'/'rnm' for RNM/AAM phi-theta grid lines. Default is 'umapr'.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap, e.g. a diverging one for a level difference. Default None, the
        yellow-orange-red map sized to the levels.
    colorbar_label : str, optional
        Colorbar label. Default None, a sound pressure level label set by `weight`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    cs : matplotlib.contour.QuadContourSet
        The contour set object from the contourf plot.

    """
    # Work on a copy, so the caller's -inf/NaN levels stay as they are; masked
    # cells (a masked array) count as missing too.
    SPL = np.ma.filled(np.ma.array(SPL, dtype=float, copy=True), np.nan)
    SPL[np.isinf(SPL)] = np.nan
    if SPL_range is None:
        minSPL = np.nanmin(SPL)
        maxSPL = np.nanmax(SPL)
    else:
        minSPL = SPL_range[0]
        maxSPL = SPL_range[1]
    # Missing cells stay NaN, which contourf leaves blank: filled with 0 dB,
    # each hole would be ringed by bands interpolated across the color scale.
    # Contour bands on round dB values so the colorbar ticks are readable
    # (see `nice_levels`); pass `levels` or `level_step` to override.
    if levels is None:
        levels = nice_levels(minSPL, maxSPL, step=level_step)
    levels = np.asarray(levels, dtype=float)
    color_map = get_ylorrd_cmap(max(len(levels), 2)) if cmap is None else cmap
    # Project to Cartesian
    lat = elv
    lon = lambert_lon(azi)
    x, y = lambert_ea(lat, lon)
    from matplotlib.pyplot import colorbar, subplots   # here: only the plot needs pyplot
    fig, ax = subplots(facecolor='white')
    ax.patch.set_visible(False)
    cs = ax.contourf(x, y, SPL, levels=levels, cmap=color_map)
    _configure_lambert_axes(ax)
    _draw_lambert_grid(ax, grid_convention=grid_convention)
    cb = colorbar(cs, pad=0.1, ticks=colorbar_ticks(levels))
    if colorbar_label is not None:
        cb.set_label(colorbar_label)
    elif weight == 'A':
        cb.set_label('Sound Pressure Level, dBA')
    else:
        cb.set_label('Sound Pressure Level, dB')
    return fig, ax, cs


def nc_lambert_ea(filename, input_frequencies=None, weight=None, SPL_range=None, grid_convention='umapr'):
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
    grid_convention : str, optional
        Grid overlay convention. Use 'umapr' for azimuth/elevation grid lines or
        'art'/'aam'/'rnm' for RNM/AAM phi-theta grid lines. Default is 'umapr'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    cs : matplotlib.contour.QuadContourSet
        The contour set object from the contourf plot.

    """
    from panam.database import extract_SPL     # here: keeps netCDF4 out of the plotting module's import
    azi, elv, _, _, _, SPLO, SPLA, *_ = extract_SPL(filename, input_frequencies)
    if weight == 'A':
        SPL = SPLA
    else:
        SPL = SPLO
    fig, ax, cs = plot_lambert_ea(
        azi,
        elv,
        SPL,
        SPL_range,
        weight=weight,
        grid_convention=grid_convention,
    )
    return fig, ax, cs


@acoustic_plot_style
def lambert_ea_points(azimuth, elevation, markers=None, colors=None, sizes=None, alpha=1.0,
                      grid_convention='umapr'):
    """
    Mark directions (for example the directions a microphone array samples) on a Lambert equal-area map of
    the lower hemisphere.

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
    grid_convention : str, optional
        Grid overlay convention. Use 'umapr' for azimuth/elevation grid lines or
        'art'/'aam'/'rnm' for RNM/AAM phi-theta grid lines. Default is 'umapr'.

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
    lon = lambert_lon(azimuth)
    x, y = lambert_ea(lat, lon)
    from matplotlib.pyplot import subplots   # here: only the plot needs pyplot
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
    _configure_lambert_axes(ax)
    _draw_lambert_grid(ax, grid_convention=grid_convention)
    return fig, ax, cs
