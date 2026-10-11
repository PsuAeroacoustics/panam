"""Energy-average repeat runs into one sphere, and compare two spheres band by band.

Both work on AAM sphere netCDF files (``docs/file_formats.md``) on one ART grid.
Levels are averaged in energy, never in dB.  A cell's level is in one of three
states, as in the gridded hemisphere: finite dB (measured), -inf (measured, no
energy) or NaN (no data).

A sphere file stores -999 both for a band that was measured with no energy and
for a direction that was not measured, so the file cannot tell them apart
(``docs/file_formats.md#how-a-bins-state-travels``).  Reading one, ``missing``
says which to take it as: ``'no_data'`` (the default) leaves that run out of the
cell's average, ``'no_energy'`` counts it as a run with zero power.  A NaN on
disk is always no data.
"""

import os

import numpy as np
from netCDF4 import Dataset

import flight_acoustics as fa
from panam_acoustics.plotting import acoustic_plot_style

#: How a sphere file's -999 (or >1e34) cell is read: left out, or zero power.
MISSING_MEANINGS = ('no_data', 'no_energy')


def _read_handle(handle, name, missing):
    def read(variable):
        return np.ma.filled(np.ma.asarray(variable[...], dtype=float), np.nan)

    variables = handle.variables
    with np.errstate(invalid='ignore'):
        levels = read(variables['AMPLITUDE'])
        sentinel = ((levels <= fa.AAM_MISSING_THRESHOLD)
                    | (levels > fa.AAM_LEGACY_MISSING_THRESHOLD))
    levels[sentinel] = np.nan if missing == 'no_data' else -np.inf

    def scalar(variable_name, default):
        if variable_name not in variables:
            return default
        return float(np.ravel(read(variables[variable_name]))[0])

    doppler = scalar('DOPPLER_SHIFT_REMOVED', None)
    if doppler is not None and doppler not in (0.0, 1.0):
        raise ValueError('{}: DOPPLER_SHIFT_REMOVED must be 0 or 1, found {}'.format(name, doppler))
    reference = (str(handle.getncattr('azimuth_reference'))
                 if 'azimuth_reference' in handle.ncattrs() else None)
    if reference is not None and reference not in fa.AZIMUTH_REFERENCES:
        raise ValueError('{}: azimuth_reference must be one of {}, found {!r}'.format(
            name, fa.AZIMUTH_REFERENCES, reference))
    weight = sum(scalar(n, 0.0) for n in ('EMPTY_WEIGHT', 'FUEL_WEIGHT', 'LOAD_WEIGHT'))
    return dict(
        name=name,
        phi_deg=read(variables['PHI']).ravel(),
        theta_deg=read(variables['THETA']).ravel(),
        frequency_hz=read(variables['FREQUENCY']).ravel(),
        levels_db=levels,
        radius_ft=scalar('RADIUS', float('nan')),
        speed_knots=scalar('SPEED', float('nan')),
        flight_path_angle_deg=scalar('FLIGHT_PATH_ANGLE', float('nan')),
        weight_lb=weight,
        doppler_shift_removed=None if doppler is None else int(doppler),
        azimuth_reference=reference,
    )


def read_sphere(source, missing='no_data'):
    """Read an AAM sphere into a dict of plain arrays.

    Args:
        source: a sphere file path, an open ``netCDF4.Dataset``, or a dict this
            function or :func:`average_spheres` returned (passed through).
        missing: how a -999 (or >1e34) cell is read, one of
            :data:`MISSING_MEANINGS`: ``'no_data'`` gives NaN, ``'no_energy'`` -inf.
            NaN on disk (and anything netCDF4 masks) is always NaN.

    Returns:
        dict with ``phi_deg``, ``theta_deg`` (deg, ART), ``frequency_hz`` (Hz),
        ``levels_db`` (dB, shaped (phi, theta, frequency)), ``radius_ft`` (ft),
        ``speed_knots`` (kt), ``flight_path_angle_deg`` (deg), ``weight_lb`` (lb, the
        sum of EMPTY_WEIGHT, FUEL_WEIGHT and LOAD_WEIGHT), ``doppler_shift_removed``
        (0, 1, or None when absent), ``azimuth_reference`` (str, or None when
        absent) and ``name``.
    """
    if missing not in MISSING_MEANINGS:
        raise ValueError('missing must be one of {}, not {!r}'.format(MISSING_MEANINGS, missing))
    if isinstance(source, dict):
        return source
    if isinstance(source, Dataset):
        return _read_handle(source, getattr(source, 'filepath', lambda: 'dataset')(), missing)
    with Dataset(os.fspath(source), mode='r') as handle:
        return _read_handle(handle, os.fspath(source), missing)


def _check_same_grid(spheres):
    first = spheres[0]
    for sphere in spheres[1:]:
        for key in ('phi_deg', 'theta_deg', 'frequency_hz'):
            if not np.array_equal(first[key], sphere[key]):
                raise ValueError('{} and {} have different {}'.format(first['name'], sphere['name'], key))


def average_spheres(paths_or_datasets, min_runs=1, missing='no_data', speed_tolerance_knots=0.0):
    """Energy-average AAM spheres of repeat runs, cell by cell.

    Each cell (direction and band) is averaged over the runs that cover it, those
    whose level there is not NaN: P = mean(10^(L/10)), L = 10 lg P.  A cell is -inf
    only when every covering run is -inf, and NaN when no run covers it or fewer than
    ``min_runs`` do.

    Every file must share PHI, THETA, FREQUENCY, RADIUS, the azimuth reference (a
    file without one is 'track') and DOPPLER_SHIFT_REMOVED (a file without it is 0);
    SPEED may differ by ``speed_tolerance_knots``.  The result carries the mean SPEED,
    FLIGHT_PATH_ANGLE and weight of the runs.

    Args:
        paths_or_datasets: sphere file paths, open ``netCDF4.Dataset`` objects, or
            :func:`read_sphere` dicts.  An averaged sphere given again counts as one run.
        min_runs: fewest covering runs a cell needs to get a level (default 1).
        missing: how a -999 cell reads; see :func:`read_sphere` (default 'no_data').
        speed_tolerance_knots: largest SPEED difference from the first file allowed,
            kt (default 0: the same to float32).

    Returns:
        A :func:`read_sphere`-style dict with the averaged ``levels_db`` (dB), plus
        ``n_runs`` (int, shaped as ``levels_db``: the runs covering each cell),
        ``amplitude_sd_db`` (dB: the sample standard deviation, ddof 1, of the
        finite levels of a cell; NaN where fewer than two are finite) and
        ``sources`` (the input names).
    """
    min_runs = int(min_runs)
    if min_runs < 1:
        raise ValueError('min_runs must be at least 1, not {}'.format(min_runs))
    spheres = [read_sphere(source, missing) for source in paths_or_datasets]
    if not spheres:
        raise ValueError('no spheres to average')
    _check_same_grid(spheres)
    first = spheres[0]
    for sphere in spheres[1:]:
        if sphere['radius_ft'] != first['radius_ft']:
            raise ValueError('{} has RADIUS {} ft, {} has {} ft'.format(
                sphere['name'], sphere['radius_ft'], first['name'], first['radius_ft']))
        if not abs(sphere['speed_knots'] - first['speed_knots']) <= float(speed_tolerance_knots):
            raise ValueError('{} has SPEED {} kt, {} has {} kt (tolerance {} kt)'.format(
                sphere['name'], sphere['speed_knots'], first['name'], first['speed_knots'],
                speed_tolerance_knots))
    references = [s['azimuth_reference'] for s in spheres]
    if len({'track' if r is None else r for r in references}) > 1:
        raise ValueError('azimuth_reference differs between the spheres: {}'.format(
            ', '.join('{} ({})'.format(s['name'], s['azimuth_reference']) for s in spheres)))
    dopplers = [s['doppler_shift_removed'] for s in spheres]
    if len({0 if d is None else d for d in dopplers}) > 1:
        raise ValueError('DOPPLER_SHIFT_REMOVED differs between the spheres: {}'.format(
            ', '.join('{} ({})'.format(s['name'], s['doppler_shift_removed']) for s in spheres)))

    levels = np.stack([s['levels_db'] for s in spheres])
    covering = ~np.isnan(levels)
    n_runs = covering.sum(axis=0)
    total = np.power(10.0, np.where(covering, levels, -np.inf) / 10.0).sum(axis=0)
    has_level = n_runs >= min_runs
    mean_power = np.divide(total, n_runs, out=np.zeros_like(total), where=has_level)
    averaged = np.full(total.shape, np.nan)
    averaged[has_level] = -np.inf
    positive = has_level & (mean_power > 0.0)
    averaged[positive] = 10.0 * np.log10(mean_power[positive])

    finite = np.isfinite(levels)
    n_finite = finite.sum(axis=0)
    finite_levels = np.where(finite, levels, 0.0)
    mean_db = np.divide(finite_levels.sum(axis=0), n_finite, out=np.zeros(total.shape),
                        where=n_finite > 0)
    squares = np.where(finite, np.square(finite_levels - mean_db), 0.0).sum(axis=0)
    spread = np.full(total.shape, np.nan)
    several = n_finite >= 2
    spread[several] = np.sqrt(squares[several] / (n_finite[several] - 1))

    labeled = [r for r in references if r is not None]
    flags = [d for d in dopplers if d is not None]
    return dict(
        name='average of {} spheres'.format(len(spheres)),
        phi_deg=first['phi_deg'], theta_deg=first['theta_deg'], frequency_hz=first['frequency_hz'],
        levels_db=averaged, n_runs=n_runs, amplitude_sd_db=spread,
        radius_ft=first['radius_ft'],
        speed_knots=float(np.mean([s['speed_knots'] for s in spheres])),
        flight_path_angle_deg=float(np.mean([s['flight_path_angle_deg'] for s in spheres])),
        weight_lb=float(np.mean([s['weight_lb'] for s in spheres])),
        doppler_shift_removed=flags[0] if flags else None,
        azimuth_reference=labeled[0] if labeled else None,
        sources=[s['name'] for s in spheres],
    )


def write_averaged_sphere(filename, average, title=None, overwrite=True):
    """Write an :func:`average_spheres` result as an AAM sphere file.

    Through :func:`flight_acoustics.write_aam_hemisphere_netcdf` (``art_levels``): the
    levels are written as they are, -999 where missing or below -100 dB, with the
    runs' mean weight as EMPTY_WEIGHT.  Two extra variables follow, which AAM and
    NICE-OPS do not read: ``N_RUNS`` (i4, the runs covering each cell) and
    ``AMPLITUDE_SD_DB`` (f4, dB, -999 where undefined).

    Args:
        filename: output path.
        average: dict from :func:`average_spheres`.
        title: the file's title (default names the source count).
        overwrite: if False, refuse an existing file.
    """
    fa.write_aam_hemisphere_netcdf(
        filename, None,
        art_levels=(average['frequency_hz'], average['levels_db']),
        phi_deg=average['phi_deg'], theta_deg=average['theta_deg'],
        radius_ft=average['radius_ft'], speed_knots=average['speed_knots'],
        flight_path_angle_deg=average['flight_path_angle_deg'],
        empty_weight_lb=average['weight_lb'], fuel_weight_lb=0.0, load_weight_lb=0.0,
        doppler_shift_removed=average['doppler_shift_removed'] or 0,
        azimuth_reference=average['azimuth_reference'],
        title=title if title is not None else 'Energy average of {} spheres'.format(
            len(average.get('sources', []))),
        overwrite=overwrite,
        extra_variables={'N_RUNS': (np.asarray(average['n_runs'], dtype=np.int32), ''),
                         'AMPLITUDE_SD_DB': (average['amplitude_sd_db'], 'DECIBEL')})


def cell_solid_angles(phi_deg, theta_deg):
    """Solid angle (sr) each ART grid point stands for, shaped (phi, theta).

    Each point owns the cell between the midpoints to its neighbors, clipped at
    the grid's ends: (cos theta_lo - cos theta_hi) times its phi width in radians.
    """
    def edges(values):
        values = np.asarray(values, dtype=float)
        middle = 0.5 * (values[1:] + values[:-1])
        return np.concatenate(([values[0]], middle, [values[-1]]))

    phi_edges = np.radians(edges(phi_deg))
    theta_edges = np.radians(edges(theta_deg))
    return np.outer(np.diff(phi_edges), -np.diff(np.cos(theta_edges)))


def _difference_statistics(difference, valid, weights):
    d = difference[valid]
    w = weights[valid]
    if d.size == 0:
        nan = float('nan')
        return dict(n_cells=0, mean_db=nan, rms_db=nan, p5_db=nan, p95_db=nan, area_weighted_mean_db=nan)
    return dict(n_cells=int(d.size), mean_db=float(d.mean()), rms_db=float(np.sqrt(np.mean(d * d))),
                p5_db=float(np.percentile(d, 5.0)), p95_db=float(np.percentile(d, 95.0)),
                area_weighted_mean_db=float(np.sum(w * d) / np.sum(w)) if np.sum(w) > 0 else float('nan'))


def compare_spheres(a, b, bands=None, missing='no_data'):
    """Band-by-band statistics of the level difference B - A between two spheres.

    Only cells finite in both files enter the statistics.  A cell finite in one file
    and -inf or NaN in the other is counted (``n_only_a``, ``n_only_b``), not
    differenced.  The area-weighted mean weights each cell by its solid angle
    (:func:`cell_solid_angles`), so that the crowded ART grid points near the nose
    and tail do not dominate it.  The overall difference of a cell sums, in energy,
    the selected bands finite in both files.

    Args:
        a, b: spheres as :func:`read_sphere` takes them, on the same PHI, THETA and
            FREQUENCY grid.
        bands: band centers to compare (Hz, matched to 0.1%); default every band.
        missing: how a -999 cell reads; see :func:`read_sphere`.

    Returns:
        dict with ``frequency_hz`` (Hz, the selected bands); per band, arrays
        ``n_cells``, ``mean_db``, ``rms_db`` (root mean square), ``p5_db``, ``p95_db``,
        ``area_weighted_mean_db`` (dB, NaN where no cell compares), ``n_only_a`` and
        ``n_only_b``; ``overall``, a dict of the same statistics for the overall level;
        ``difference_db`` (dB, (phi, theta, band), NaN where not compared),
        ``overall_difference_db`` (dB, (phi, theta)), ``phi_deg``, ``theta_deg`` and
        the two ``names``.
    """
    a = read_sphere(a, missing)
    b = read_sphere(b, missing)
    _check_same_grid([a, b])
    frequency = a['frequency_hz']
    if bands is None:
        index = np.arange(frequency.size)
    else:
        index = []
        for band in np.atleast_1d(np.asarray(bands, dtype=float)):
            match = np.flatnonzero(np.isclose(frequency, band, rtol=1e-3, atol=0.0))
            if match.size == 0:
                raise ValueError('no {} Hz band in the spheres'.format(band))
            index.append(int(match[0]))
        index = np.asarray(index)
    level_a = a['levels_db'][:, :, index]
    level_b = b['levels_db'][:, :, index]
    finite_a = np.isfinite(level_a)
    finite_b = np.isfinite(level_b)
    both = finite_a & finite_b
    difference = np.where(both, np.where(both, level_b, 0.0) - np.where(both, level_a, 0.0), np.nan)
    weights = cell_solid_angles(a['phi_deg'], a['theta_deg'])

    keys = ('n_cells', 'mean_db', 'rms_db', 'p5_db', 'p95_db', 'area_weighted_mean_db')
    rows = [_difference_statistics(difference[:, :, k], both[:, :, k], weights) for k in range(index.size)]
    result = {key: np.array([row[key] for row in rows]) for key in keys}
    result['n_only_a'] = np.sum(finite_a & ~finite_b, axis=(0, 1))
    result['n_only_b'] = np.sum(finite_b & ~finite_a, axis=(0, 1))

    def overall_power(levels):
        return np.where(both, np.power(10.0, np.where(both, levels, 0.0) / 10.0), 0.0).sum(axis=2)

    power_a, power_b = overall_power(level_a), overall_power(level_b)
    compared = both.any(axis=2) & (power_a > 0.0) & (power_b > 0.0)
    overall = np.full(compared.shape, np.nan)
    overall[compared] = 10.0 * np.log10(power_b[compared] / power_a[compared])
    result['overall'] = _difference_statistics(overall, compared, weights)
    result['overall']['n_only_a'] = int(np.sum(finite_a.any(axis=2) & ~finite_b.any(axis=2)))
    result['overall']['n_only_b'] = int(np.sum(finite_b.any(axis=2) & ~finite_a.any(axis=2)))
    result.update(frequency_hz=frequency[index], difference_db=difference, overall_difference_db=overall,
                  phi_deg=a['phi_deg'], theta_deg=a['theta_deg'], names=(a['name'], b['name']))
    return result


def format_comparison(result):
    """A text table of a :func:`compare_spheres` result, one row per band and an overall row."""
    header = '{:>10} {:>6} {:>8} {:>7} {:>7} {:>7} {:>8} {:>7} {:>7}'.format(
        'band Hz', 'cells', 'mean dB', 'rms dB', 'p5 dB', 'p95 dB', 'area dB', 'only A', 'only B')
    lines = ['B - A, B = {}, A = {}'.format(result['names'][1], result['names'][0]), header]
    row = '{:>10} {:>6d} {:>8.2f} {:>7.2f} {:>7.2f} {:>7.2f} {:>8.2f} {:>7d} {:>7d}'
    for k, frequency in enumerate(result['frequency_hz']):
        lines.append(row.format('{:g}'.format(frequency), int(result['n_cells'][k]), result['mean_db'][k],
                                result['rms_db'][k], result['p5_db'][k], result['p95_db'][k],
                                result['area_weighted_mean_db'][k], int(result['n_only_a'][k]),
                                int(result['n_only_b'][k])))
    o = result['overall']
    lines.append(row.format('overall', o['n_cells'], o['mean_db'], o['rms_db'], o['p5_db'], o['p95_db'],
                            o['area_weighted_mean_db'], o['n_only_a'], o['n_only_b']))
    return '\n'.join(lines)


@acoustic_plot_style
def plot_sphere_difference(result, band=None, level_range=None, grid_convention='art'):
    """Map a :func:`compare_spheres` difference on a Lambert equal-area projection.

    Args:
        result: dict from :func:`compare_spheres`.
        band: band center to map (Hz); default the overall difference.
        level_range: (low, high) color range, dB; default symmetric about 0 over
            the largest finite difference.
        grid_convention: grid lines, as :func:`flight_acoustics.plot_lambert_ea`
            (default 'art').

    Returns:
        matplotlib Figure, drawn with a diverging colormap, red where B is louder.
    """
    if band is None:
        field = result['overall_difference_db']
        label = 'Overall level difference B - A, dB'
    else:
        match = np.flatnonzero(np.isclose(result['frequency_hz'], band, rtol=1e-3, atol=0.0))
        if match.size == 0:
            raise ValueError('no {} Hz band in the comparison'.format(band))
        field = result['difference_db'][:, :, match[0]]
        label = '{:g} Hz level difference B - A, dB'.format(result['frequency_hz'][match[0]])
    if not np.any(np.isfinite(field)):
        raise ValueError('no cell is finite in both spheres')
    if level_range is None:
        largest = float(np.nanmax(np.abs(field)))
        largest = max(np.ceil(largest), 1.0)
        level_range = (-largest, largest)
    theta, phi = np.meshgrid(result['theta_deg'], result['phi_deg'])
    azi, elv = fa.art2umapr(np.radians(phi), np.radians(theta))
    fig, _, _ = fa.plot_lambert_ea(azi, elv, field, level_range, grid_convention=grid_convention,
                                   cmap='RdBu_r', colorbar_label=label)
    return fig
