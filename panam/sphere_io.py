"""AAM sphere files: reading, writing and the run metadata they carry.

:func:`write_aam_hemisphere_netcdf` resamples a hemisphere onto the AAM (phi,
theta) grid and writes the sphere netCDF; :func:`load_nc_sphere` reads one back.
The file contract, with ``RADIUS`` in ft and ``SPEED`` in knots, is
``docs/file_formats.md``.  :func:`mask_missing_levels` maps the on-disk missing
markers (NaN, -999, >1e34) to -inf.  :func:`read_run_metadata` reads a sphere's
per-run metadata back; :func:`band_index` finds a band center on a frequency axis.
:func:`_atomic_output`, used by the sphere writer, both database writers and the ``.hem``
writers (not :func:`norah2.write_norah2_triangulation`), writes a file in full or not at
all.
"""

import contextlib
import os
import shutil
from typing import Optional

from netCDF4 import Dataset
import numpy as np

from panam import provenance as _provenance
from panam import unit_conversion
from panam.geometry import art2umapr
from panam.gridding import _sample_hemisphere_levels


#: One knot in m/s.
KNOT_MPS = 0.514444


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

    Every array is a plain float array.  netCDF4 returns masked arrays, and
    their masked cells came through arithmetic as masked scalars that
    :func:`spectra.safe_log10` cannot test (409 RuntimeWarnings reducing one synthetic
    flyby sphere), while other results came out right only because a mask
    happened to propagate.  Anything netCDF4 masks (a _FillValue, missing_value or
    valid-range hit) is NaN instead, which :func:`mask_missing_levels` reads
    as missing along with the in-band sentinels.
    """
    def read(variable):
        return np.ma.filled(np.ma.asarray(variable[:], dtype=float), np.nan)

    with Dataset(filename, mode='r') as file_handle:
        # Noise spheres sometimes contain NaN, which netCDF4's masking compares
        # against; they are handled downstream, so do not warn about them.
        with np.errstate(invalid='ignore'):
            phi = read(file_handle.variables['PHI'])
            theta = read(file_handle.variables['THETA'])
            frequency = read(file_handle.variables['FREQUENCY'])
            amplitude = read(file_handle.variables['AMPLITUDE'])
            radius = read(file_handle.variables['RADIUS'])
            speed = read(file_handle.variables['SPEED'])
            flight_path_angle = read(file_handle.variables['FLIGHT_PATH_ANGLE'])
    return amplitude, phi, theta, frequency, radius, speed, flight_path_angle


#: What a sphere's azimuth is measured from (azimuth 180 along it): ``'track'``, the
#: ground-velocity direction, or ``'heading'``, the airframe's heading.  Written as the
#: text attribute ``azimuth_reference`` of a sphere file and of a database.
AZIMUTH_REFERENCES = ('track', 'heading')


#: Wind speed units a caller may declare for :func:`write_aam_hemisphere_netcdf`'s run
#: metadata, and their size in m/s.  The unit is stored as declared; it is converted only
#: to form the airspeed.  No unit is assumed, even where one is known (the 2017 LIDAR
#: winds are in knots).
WIND_SPEED_UNITS = {'kt': KNOT_MPS, 'm/s': 1.0, 'ft/s': 0.3048, 'mph': 0.44704}


#: Where a run's wind at the aircraft came from.  Several may be joined with '+', e.g.
#: 'lidar+balloon' for a profile pieced together from both.
WIND_SOURCES = ('lidar', 'balloon', 'station', 'none')


#: Pounds-force to newtons.
POUND_FORCE_NEWTONS = 4.4482216152605


#: Per-run metadata a sphere file may carry (:func:`write_aam_hemisphere_netcdf`'s
#: ``run_metadata``): sphere variable name, unit, and the key it is read back under by
#: :func:`read_run_metadata`.  NaN where unknown.  WIND_* take the unit the caller declared.
RUN_METADATA_VARIABLES = (
    ('GROSS_WEIGHT', 'POUNDS', 'gross_weight_lb'),
    ('AIR_DENSITY', 'KG/M^3', 'air_density_kg_m3'),
    ('WIND_ALONG_TRACK', None, 'wind_along_track'),
    ('WIND_CROSS_TRACK', None, 'wind_cross_track'),
    ('AIRSPEED', 'KNOTS', 'airspeed_knots'),
)


#: Text attributes of a sphere file carrying the rest of the run metadata, '' if unknown.
RUN_METADATA_ATTRIBUTES = ('wind_source', 'wind_units', 'wind_reference_direction', 'air_density_source')


def normalize_wind_source(source):
    """``source`` as stored: lower case, each '+'-joined part one of :data:`WIND_SOURCES`."""
    text = str(source).strip().lower()
    parts = [part.strip() for part in text.split('+')]
    if not text or any(part not in WIND_SOURCES for part in parts) or ('none' in parts and len(parts) > 1):
        raise ValueError('wind source must be one of {} (several joined with +), not {!r}'.format(
            WIND_SOURCES, source))
    return '+'.join(parts)


def _empty_run_metadata():
    out = {key: float('nan') for _, _, key in RUN_METADATA_VARIABLES}
    out.update({name: '' for name in RUN_METADATA_ATTRIBUTES})
    return out


def read_run_metadata(filename):
    """The per-run metadata a sphere file carries (see :data:`RUN_METADATA_VARIABLES`).

    Returns a dict: the numbers (NaN when the file has none, as every sphere written before
    these existed), the text attributes ('' when absent) and ``azimuth_reference`` (None
    when absent).
    """
    out = _empty_run_metadata()
    out['azimuth_reference'] = None
    with Dataset(filename, mode='r') as handle:
        for name, _, key in RUN_METADATA_VARIABLES:
            if name in handle.variables:
                value = _filled(handle.variables[name])
                out[key] = float(np.ravel(value)[0])
        for name in RUN_METADATA_ATTRIBUTES:
            if name in handle.ncattrs():
                out[name] = str(handle.getncattr(name))
        if 'azimuth_reference' in handle.ncattrs():
            out['azimuth_reference'] = str(handle.getncattr('azimuth_reference'))
    return out


def _read_azimuth_reference(filename):
    """The azimuth_reference a sphere file carries, or None if it has none."""
    with Dataset(filename, mode='r') as handle:
        return _azimuth_reference_of(handle, filename)


def _azimuth_reference_of(handle, name):
    """:func:`_read_azimuth_reference` of an open sphere ``handle``; ``name`` labels errors."""
    if 'azimuth_reference' not in handle.ncattrs():
        return None
    value = str(handle.getncattr('azimuth_reference'))
    if value not in AZIMUTH_REFERENCES:
        raise ValueError('{}: azimuth_reference must be one of {}, found {!r}'.format(
            name, AZIMUTH_REFERENCES, value))
    return value


def _read_doppler_shift_removed(filename):
    """The DOPPLER_SHIFT_REMOVED flag a sphere file carries, or None if it has none.

    0 means received-frame spheres (the normal case), 1 de-Dopplerized
    (stationary-frame) ones.  :func:`write_aam_hemisphere_netcdf` writes it.
    """
    with Dataset(filename, mode='r') as file_handle:
        return _doppler_shift_removed_of(file_handle, filename)


def _doppler_shift_removed_of(handle, name):
    """:func:`_read_doppler_shift_removed` of an open sphere ``handle``; ``name`` labels errors."""
    if 'DOPPLER_SHIFT_REMOVED' not in handle.variables:
        return None
    raw = _filled(handle.variables['DOPPLER_SHIFT_REMOVED'])
    flag = raw.ravel()
    if flag.size != 1 or flag[0] not in (0.0, 1.0):
        raise ValueError('{}: DOPPLER_SHIFT_REMOVED must be 0 or 1, found {}'.format(name, raw))
    return int(flag[0])


def _filled(variable):
    """All of a netCDF variable's values as a float array, NaN where masked."""
    return np.ma.filled(np.ma.asarray(variable[...], dtype=float), np.nan)


def _read_scalar(handle, name, *default):
    """A netCDF variable's first value as a float, NaN where masked.

    Without ``default`` a missing variable raises KeyError; with it, the default
    is returned instead.
    """
    if default and name not in handle.variables:
        return default[0]
    with np.errstate(invalid='ignore'):
        return float(np.ravel(_filled(handle.variables[name]))[0])


#: Level written into AMPLITUDE where a band carries no usable data.  The AAM
#: code masks known-bad values with -999, so that is what a sphere meant for it
#: should carry.  Note the shipped 2017 spheres do not use it -- they mask with
#: NaN -- and older panam output used a large positive sentinel, so
#: :func:`mask_missing_levels` accepts all three on read.
AAM_MISSING_LEVEL = -999.0


#: Anything at or below this reads as masked.  Loose enough to survive the
#: float32 round trip AMPLITUDE goes through, and far below any real source
#: level, so it cannot swallow a measurement.
AAM_MISSING_THRESHOLD = -998.0


#: Older panam spheres (and some AAM-style tooling) mask with a large positive
#: value instead.  Still accepted on read so existing files keep working.
AAM_LEGACY_MISSING_THRESHOLD = 1.0e34


def mask_missing_levels(amplitude):
    """Return ``amplitude`` with every masked band as -inf, i.e. no energy.

    Three conventions appear in the wild and all three are accepted: NaN (what
    the shipped 2017 Noise Abatement spheres use), -999 (what the AAM code
    masks with, and what this module now writes), and a large positive value
    (older panam output).  The array comes back as a plain float array, never a
    masked one, so downstream arithmetic behaves the same whichever it was.
    """
    amplitude = np.asarray(amplitude, dtype=float).copy()
    amplitude[np.isnan(amplitude) | missing_level_sentinel(amplitude)] = -np.inf
    return amplitude


def missing_level_sentinel(levels_db):
    """True where a sphere level on disk is a missing-level sentinel: -999 or below, or > 1e34.

    NaN, the third convention :func:`mask_missing_levels` accepts, is not a
    sentinel and gives False.
    """
    return (levels_db <= AAM_MISSING_THRESHOLD) | (levels_db > AAM_LEGACY_MISSING_THRESHOLD)


def band_index(frequency_hz, band_hz):
    """Index of the first band center in ``frequency_hz`` (Hz) within 0.1 % of
    ``band_hz`` (Hz), or None when there is none."""
    match = np.flatnonzero(np.isclose(frequency_hz, band_hz, rtol=1e-3, atol=0.0))
    return int(match[0]) if match.size else None


@contextlib.contextmanager
def _atomic_output(path):
    """Write a file in full or not at all.

    Yields a temporary path beside ``path``, named ``.<name>.<pid>.tmp`` so
    that a ``*.nc`` or ``*.nod`` glob (and a sphere directory listing) does not
    match it, and moves it onto ``path`` with :func:`os.replace` when the
    ``with`` block finishes.  If the block raises, the temporary file is
    removed and ``path`` is left as it was: an existing file is kept, and no
    partial file appears.  The writer must close the temporary file inside the
    block.  The process id keeps parallel workers writing the same name apart.
    (Not mkstemp: its file is private, 0600, and os.replace would keep that.)
    A symbolic link is written through, as ``open(path, 'w')`` would: the
    temporary file sits beside the link's target and replaces the target, and the
    link stays.  An existing file's permission bits are kept.
    """
    path = os.path.realpath(os.path.expanduser(os.fspath(path)))
    temporary = os.path.join(os.path.dirname(path), '.{}.{}.tmp'.format(
        os.path.basename(path), os.getpid()))
    try:
        yield temporary
        if os.path.exists(path):
            shutil.copymode(path, temporary)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(temporary)
        raise


#: Legacy AAM spheres blank-pad the ``unit`` attribute of every scalar variable
#: to this width (a Fortran CHARACTER(20) field written verbatim).
AAM_SCALAR_UNIT_WIDTH = 20


#: The order AAM's own spheres declare their dimensions and variables in.  AAM 3.1.0 reads
#: them by position, not by name: a sphere declaring PHI, THETA, FREQUENCY and AMPLITUDE
#: first is refused ("RMNETCDF2 ... Variable ID mismatch!!"), as panam's were until the
#: 2026-10-05 cross-code run found it.  Anything else a sphere carries comes after AMPLITUDE,
#: which AAM accepts.
AAM_VARIABLE_ORDER = ('BB', 'NB', 'PT', 'DOPPLER_SHIFT_REMOVED', 'EMPTY_WEIGHT', 'FUEL_WEIGHT',
                      'LOAD_WEIGHT', 'RADIUS', 'FLIGHT_PATH_ANGLE', 'PYLON_ANGLE', 'SPEED', 'XYZ',
                      'PHI', 'THETA', 'FREQUENCY', 'AMPLITUDE')


#: The ART phi grid :func:`write_aam_hemisphere_netcdf` writes when given none: -90..90 deg
#: (lateral) in 10 deg steps, a common AAM/RNM range, as in the example sphere; step sizes
#: vary by dataset.
AAM_DEFAULT_PHI_DEG = np.arange(-90.0, 90.0 + 1e-9, 10.0)


#: The ART theta grid :func:`write_aam_hemisphere_netcdf` writes when given none: 0..180 deg in
#: 5 deg steps (longitudinal).
AAM_DEFAULT_THETA_DEG = np.arange(0.0, 180.0 + 1e-9, 5.0)


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
        empty_weight_lb: Optional[float] = None,
        fuel_weight_lb: Optional[float] = None,
        load_weight_lb: Optional[float] = None,
        pylon_angle_deg: float = 90.0,
        masttilt_deg: float = 0.0,
        xyz_ft=(0.0, 0.0, 0.0),
        minimum_level_db: float = -100.0,
        overwrite: bool = True,
        azimuth_reference: Optional[str] = None,
        run_metadata: Optional[dict] = None,
        art_levels=None,
        extra_variables: Optional[dict] = None,
        provenance: Optional[dict] = None,
):
    """Write a depropagated acoustic hemisphere to an AAM/RNM-style netCDF sphere.

    This exports a hemisphere in the same variable naming convention expected by
    :func:`load_nc_sphere` (and thus AAM-style tooling):

    - ``PHI`` (deg), ``THETA`` (deg), ``FREQUENCY`` (Hz)
    - ``AMPLITUDE`` (dB), shape (Nphi, Ntheta, Nfreq)
    - ``RADIUS`` (ft), ``SPEED`` (knots), ``FLIGHT_PATH_ANGLE`` (deg)

    The input ``hemisphere`` must be the dict returned by :func:`depropagation.depropagate_hemisphere`
    with either ``third_octave=True`` or ``narrowband=True`` enabled.

    Notes
    -----
    ``depropagate_hemisphere`` produces a regular grid in UMAPR (azimuth/elevation).
    AAM spheres are stored on a regular grid in ART (phi/theta). This function
    interpolates the UMAPR grid onto a separable ART grid (``phi_deg`` x ``theta_deg``)
    in the *linear power* domain and converts back to dB.

    Args:
        filename: Path to the output netCDF file.
        hemisphere: Output dict from :func:`depropagation.depropagate_hemisphere`.
        mode: 'auto', 'third_octave', or 'narrowband'. 'auto' prefers third-octave.
        phi_deg: Optional 1D ART phi grid (degrees). Default: -90..90 in 10 degree steps.
        theta_deg: Optional 1D ART theta grid (degrees). Default: 0..180 in 5 degree steps.
        speed_knots: Stored into ``SPEED`` (knots).
        flight_path_angle_deg: Stored into ``FLIGHT_PATH_ANGLE`` (deg).
        radius_ft: Optional override for ``RADIUS`` (ft). If None, derived from hemisphere metadata ``r_ref``.
        minimum_level_db: bands below this level are written as the AAM missing
            sentinel rather than as a number. Hemispheres built before
            2026-09-26 carry a -3076 dB floor where bands were gated to zero
            power (and where there was no data), which interpolates to levels
            near -3000 dB. Those are zero energy and consumers handle them, but
            "not measured" is what they mean and what AAM has a convention for.
            The default sits far below any real measurement at a 100 ft sphere
            radius, so it catches the gated bands and nothing else. Pass -inf
            to keep every finite level.
        overwrite: If False, raises when filename exists.
        empty_weight_lb, fuel_weight_lb, load_weight_lb: ``EMPTY_WEIGHT``,
            ``FUEL_WEIGHT`` and ``LOAD_WEIGHT`` (lb).  AAM uses only their sum, as
            the vehicle weight its effective-weight lookup (QSAM2, FRAME) compares.
            Left all three None, a finite ``run_metadata`` ``gross_weight_lb`` is
            written as EMPTY_WEIGHT with the other two zero, so the sum is the
            run's weight (a run records no breakdown); without one all three are
            zero.  Any one given, the others None are zero.
        azimuth_reference: what the hemisphere's azimuth was measured from,
            one of :data:`AZIMUTH_REFERENCES`; written as the text attribute
            ``azimuth_reference``.  None writes nothing (a reader then assumes
            'track', as every sphere before it was).
        run_metadata: optional dict describing the run the sphere was built
            from, written as the scalar variables of :data:`RUN_METADATA_VARIABLES`
            (left out where a key is absent, None or NaN, which
            :func:`read_run_metadata` reads back as NaN) and the text attributes of
            :data:`RUN_METADATA_ATTRIBUTES`.  ``wind_units`` must be one of
            :data:`WIND_SPEED_UNITS` when a wind component is finite, and is
            written as the WIND_* variables' unit; ``wind_source`` is checked
            against :data:`WIND_SOURCES`.  :func:`read_run_metadata` reads it back.
        art_levels: optional ``(frequency_hz, levels_db)``: band levels (dB) already on
            the ART grid, shaped (len(phi_deg), len(theta_deg), len(frequency_hz)), with
            -inf or NaN where missing.  They are written as they are, without
            resampling (``minimum_level_db`` still applies); ``hemisphere`` must then be
            None and ``phi_deg``, ``theta_deg`` and ``radius_ft`` must be given.
            :func:`sphere_statistics.write_averaged_sphere` uses it.
        extra_variables: optional dict ``{name: (values, unit)}`` of further
            (PHI, THETA, FREQUENCY) variables, written after everything else, which
            AAM does not read.  Integer arrays are written as i4; anything else as f4,
            with non-finite values as -999.
        provenance: the provenance stamp (:func:`provenance.build_provenance`) to
            write as the ``panam_*`` global attributes.  None stamps the file with
            this writer's own arguments as the settings (the level data in
            hemisphere, art_levels and extra_variables, run_metadata and filename
            left out) and the file name as its
            ``panam_inputs``.  AAM reads none of them.

    Returns:
        None
    """
    if provenance is None:
        settings = {key: value for key, value in locals().items()
                    if key not in ('hemisphere', 'run_metadata', 'provenance', 'filename',
                                   'art_levels', 'extra_variables')}
        # A fixed panam_tool value (docs/file_formats.md): it kept its name when the function moved here.
        provenance = _provenance.build_provenance('flight_acoustics.write_aam_hemisphere_netcdf',
                                                  settings, inputs={'filename': _provenance.path_name(filename)})

    if azimuth_reference is not None and azimuth_reference not in AZIMUTH_REFERENCES:
        raise ValueError('azimuth_reference must be one of {}, not {!r}'.format(
            AZIMUTH_REFERENCES, azimuth_reference))
    if float(doppler_shift_removed) not in (0.0, 1.0):
        raise ValueError('doppler_shift_removed must be 0 or 1, not {!r}'.format(doppler_shift_removed))
    xyz = np.asarray(xyz_ft, dtype=np.float32).ravel()
    if xyz.size != 3:
        raise ValueError('xyz_ft must be a 3-element iterable (x, y, z) in feet')
    metadata = None
    if run_metadata is not None:
        unknown = set(run_metadata) - {key for _, _, key in RUN_METADATA_VARIABLES} - set(RUN_METADATA_ATTRIBUTES)
        if unknown:
            raise ValueError('unknown run_metadata keys: {}'.format(sorted(unknown)))
        metadata = _empty_run_metadata()
        for key, value in run_metadata.items():
            if key in RUN_METADATA_ATTRIBUTES:
                metadata[key] = '' if value is None else str(value)
            else:
                metadata[key] = float('nan') if value is None else float(value)
        has_wind = np.isfinite(metadata['wind_along_track']) or np.isfinite(metadata['wind_cross_track'])
        if has_wind and metadata['wind_units'] not in WIND_SPEED_UNITS:
            raise ValueError('run_metadata wind_units must be one of {} when a wind is given, not {!r}'.format(
                sorted(WIND_SPEED_UNITS), metadata['wind_units']))
        if metadata['wind_source']:
            metadata['wind_source'] = normalize_wind_source(metadata['wind_source'])
        if has_wind and metadata['wind_source'] in ('', 'none'):
            raise ValueError('run_metadata gives a wind but no wind_source')

    weights = (empty_weight_lb, fuel_weight_lb, load_weight_lb)
    if all(weight is None for weight in weights):
        gross = metadata['gross_weight_lb'] if metadata is not None else float('nan')
        weights = (gross if np.isfinite(gross) else 0.0, 0.0, 0.0)
    weights = tuple(0.0 if weight is None else float(weight) for weight in weights)

    if not overwrite and os.path.exists(filename):
        raise FileExistsError(f'Output file already exists: {filename}')

    if art_levels is not None:
        if hemisphere is not None:
            raise ValueError('pass hemisphere or art_levels, not both')
        if phi_deg is None or theta_deg is None or radius_ft is None:
            raise ValueError('art_levels needs phi_deg, theta_deg and radius_ft')
        meta = {}
    elif not isinstance(hemisphere, dict):
        raise TypeError('hemisphere must be a dict returned by depropagate_hemisphere')
    else:
        meta = hemisphere.get('metadata', {}) if isinstance(hemisphere.get('metadata', {}), dict) else {}

    # Default ART grids; pass phi_deg/theta_deg for an exact grid.
    if phi_deg is None:
        phi_deg = AAM_DEFAULT_PHI_DEG
    if theta_deg is None:
        theta_deg = AAM_DEFAULT_THETA_DEG
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

    if art_levels is not None:
        frequency_hz = np.asarray(art_levels[0], dtype=float).ravel()
        amplitude_db = np.array(art_levels[1], dtype=float)
        if amplitude_db.shape != (phi_deg.size, theta_deg.size, frequency_hz.size):
            raise ValueError('art_levels levels must be shaped (phi, theta, frequency) = {}, not {}'
                             .format((phi_deg.size, theta_deg.size, frequency_hz.size), amplitude_db.shape))
        with np.errstate(invalid='ignore'):
            amplitude_db[amplitude_db < float(minimum_level_db)] = -np.inf
    else:
        # ART grid points -> UMAPR for interpolation
        TH, PH = np.meshgrid(theta_deg, phi_deg)
        azi_rad, elv_rad = art2umapr(np.deg2rad(PH), np.deg2rad(TH))
        # Output amplitude array: (phi, theta, freq)
        frequency_hz, amplitude_db = _sample_hemisphere_levels(
            hemisphere, mode, np.degrees(azi_rad), np.degrees(elv_rad), minimum_level_db)
    nphi = phi_deg.size
    nth = theta_deg.size
    nfreq = int(np.asarray(frequency_hz).size)

    # Replace non-finite levels with the AAM missing mask
    missing_sentinel = np.float32(AAM_MISSING_LEVEL)
    amplitude_to_write = amplitude_db.astype(np.float32)
    amplitude_to_write[~np.isfinite(amplitude_to_write)] = missing_sentinel

    extra = {}
    reserved = set(AAM_VARIABLE_ORDER) | {'MASTTILT'} | {name for name, _, _ in RUN_METADATA_VARIABLES}
    for name, (values, unit) in (extra_variables or {}).items():
        if name in reserved:
            raise ValueError('extra variable {!r} would replace a sphere variable'.format(name))
        values = np.asarray(values)
        if values.shape != (nphi, nth, nfreq):
            raise ValueError('extra variable {!r} must be shaped {}, not {}'.format(
                name, (nphi, nth, nfreq), values.shape))
        extra[name] = (values, unit)

    # Write netCDF (match AAM naming conventions).
    # Use NETCDF3_CLASSIC for broad interoperability with legacy AAM/RNM tooling
    # and to avoid HDF5 backend dependency issues on some CI platforms.
    with _atomic_output(filename) as temporary, \
            Dataset(temporary, mode='w', format='NETCDF3_CLASSIC') as ds:
        # Dimensions and variables in AAM_VARIABLE_ORDER, which AAM checks by
        # position; legacy AAM/RNM spheres give every scalar a singleton dimension
        # of its own name, declared in the same order.  MASTTILT, which AAM does
        # not read, comes after AMPLITUDE.
        sizes = dict(XYZ=3, PHI=nphi, THETA=nth, FREQUENCY=nfreq)
        for name in AAM_VARIABLE_ORDER + ('MASTTILT',):
            if name != 'AMPLITUDE':
                ds.createDimension(name, sizes.get(name, 1))

        vBB = ds.createVariable('BB', 'f4')
        vNB = ds.createVariable('NB', 'f4')
        vPT = ds.createVariable('PT', 'f4')
        vDSR = ds.createVariable('DOPPLER_SHIFT_REMOVED', 'f4')
        vEW = ds.createVariable('EMPTY_WEIGHT', 'f4')
        vFW = ds.createVariable('FUEL_WEIGHT', 'f4')
        vLW = ds.createVariable('LOAD_WEIGHT', 'f4')
        vr = ds.createVariable('RADIUS', 'f4')
        vfpa = ds.createVariable('FLIGHT_PATH_ANGLE', 'f4')
        vPA = ds.createVariable('PYLON_ANGLE', 'f4')
        vs = ds.createVariable('SPEED', 'f4')
        vXYZ = ds.createVariable('XYZ', 'f4', ('XYZ',))
        vphi = ds.createVariable('PHI', 'f4', ('PHI',))
        vth = ds.createVariable('THETA', 'f4', ('THETA',))
        vf = ds.createVariable('FREQUENCY', 'f4', ('FREQUENCY',))
        # No fill_value here: the legacy AAM spheres carry no _FillValue on
        # AMPLITUDE, and setting one both adds an attribute they do not have and
        # makes netCDF4 return a masked array where they return a plain one.
        # The missing sentinel is written into the data instead, as they do.
        vamp = ds.createVariable('AMPLITUDE', 'f4', ('PHI', 'THETA', 'FREQUENCY'))
        vMT = ds.createVariable('MASTTILT', 'f4')

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
        vEW.assignValue(np.float32(weights[0]))
        vFW.assignValue(np.float32(weights[1]))
        vLW.assignValue(np.float32(weights[2]))
        vPA.assignValue(np.float32(pylon_angle_deg))
        vMT.assignValue(np.float32(masttilt_deg))

        vXYZ[:] = xyz

        # Match the example file's attribute naming: 'unit' (singular).
        # The legacy AAM spheres blank-pad the unit on every scalar variable to
        # AAM_SCALAR_UNIT_WIDTH characters and leave the array variables' units
        # unpadded -- the shape a Fortran CHARACTER(20) write produces.  Keep
        # both conventions so a reader that takes the attribute length at face
        # value sees what it saw in the shipped files.
        vphi.unit = 'DEGREE'
        vth.unit = 'DEGREE'
        vf.unit = 'HERTZ'
        vamp.unit = 'DECIBEL'
        vXYZ.unit = 'FEET'

        def _scalar_unit(text):
            return str(text).ljust(AAM_SCALAR_UNIT_WIDTH)

        vr.unit = _scalar_unit('FEET')
        vs.unit = _scalar_unit('KNOTS')
        vfpa.unit = _scalar_unit('DEGREE')
        vBB.unit = _scalar_unit('')
        vNB.unit = _scalar_unit('')
        vPT.unit = _scalar_unit('')
        vDSR.unit = _scalar_unit('')
        vEW.unit = _scalar_unit('POUNDS')
        vFW.unit = _scalar_unit('POUNDS')
        vLW.unit = _scalar_unit('POUNDS')
        vPA.unit = _scalar_unit('DEGREE')
        vMT.unit = _scalar_unit('DEGREE')

        ds.title = title if title is not None else 'AAM/RNM acoustic hemisphere'
        if azimuth_reference is not None:
            ds.azimuth_reference = azimuth_reference
        if metadata is not None:
            # Doubles, not the legacy scalars' float: a density or a weight is worth its digits.
            # An unknown is left out rather than written as NaN, which read_run_metadata
            # reads back the same way and which no AAM sphere carries.
            for name, unit, key in RUN_METADATA_VARIABLES:
                if not np.isfinite(metadata[key]):
                    continue
                variable = ds.createVariable(name, 'f8')
                variable.assignValue(metadata[key])
                variable.unit = _scalar_unit(metadata['wind_units'] if unit is None else unit)
            for name in RUN_METADATA_ATTRIBUTES:
                ds.setncattr(name, metadata[name])
        for name, (values, unit) in extra.items():
            if np.issubdtype(values.dtype, np.integer):
                variable = ds.createVariable(name, 'i4', ('PHI', 'THETA', 'FREQUENCY'))
                variable[:, :, :] = values.astype(np.int32)
            else:
                variable = ds.createVariable(name, 'f4', ('PHI', 'THETA', 'FREQUENCY'))
                values = values.astype(np.float32)
                values[~np.isfinite(values)] = missing_sentinel
                variable[:, :, :] = values
            variable.unit = str(unit)
        _provenance.write_provenance(ds, provenance)
