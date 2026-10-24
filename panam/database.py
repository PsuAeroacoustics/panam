"""The NICEOPS sphere database (``.nod``) built from a directory of AAM spheres.

:func:`build_empirical_database` reads the spheres and their ``vehicle.cfg``,
adds the load-factor ladder, the extended flight path angles and the
synthesized hover, and writes the database through :func:`add_sphere_group`
(``docs/THEORY.md`` section 7; the contract is ``docs/file_formats.md``).
:func:`extract_SPL` reduces one sphere file to the levels a group carries, and
:func:`read_vehicle_data` and :func:`read_vehicle_rotor_data` read the vehicle file.
:func:`norah2.build_database_from_norah2` writes through the same helpers: the root
writer (``_write_database_root``), :func:`add_sphere_group`, the sphere completion and
the hover synthesis (``_synthesized_hover_sphere``).
Databases are metric; sphere files keep ft and knots.
"""

import fnmatch
import json
import os
import warnings
from collections import namedtuple
from configparser import ConfigParser
from glob import glob

from netCDF4 import Dataset
import numpy as np

from panam import provenance as _provenance
from panam.acoustics.atmosphere import Atmosphere
from panam.spectra import dBAw, OASPL, power_to_db
from panam.geometry import art2umapr, SPHERE_FRAMES
from panam.gridding import repitch_art_levels
from panam.absorption import _band_attenuation_db
from panam import sphere_statistics
from panam.sphere_io import (AZIMUTH_REFERENCES, KNOT_MPS, load_nc_sphere, mask_missing_levels,
                             POUND_FORCE_NEWTONS, read_run_metadata, read_sphere_frame, _atomic_output,
                             _read_azimuth_reference, _read_doppler_shift_removed, _read_scalar)


#: On-disk format version written into every database as "database_version".
#: 0 / absent  -- built before this was introduced.  Hover spheres in those files
#:                may have broken directivity (see mirror_phi_to_upper_surface).
#: 1           -- directivity mirror correct; hover spheres fore-aft averaged on an
#:                energy basis with EAA recomputed from the averaged spectrum.
DATABASE_FORMAT_VERSION = 1


#: Values of the root attribute speed_reference: whether a condition's advance
#: ratio and flight path angle are taken against the ground or the air.
SPEED_REFERENCES = ('ground', 'air')


#: A condition is "near level flight" within this many degrees of zero flight
#: path angle.  Used both to pick the source sphere for hover and to choose which
#: conditions get re-emitted at extended flight path angles.
LEVEL_FLIGHT_TOLERANCE = 2.0


#: One condition to write, ready for add_sphere_group.  ``frequency`` and
#: ``amplitude`` are None when the database carries no spectra; ``coverage`` is
#: (n_phi, n_theta, n_bands) bool, see _complete_sphere_with_coverage.
_SphereCondition = namedtuple('_SphereCondition', [
    'name', 'phi', 'theta', 'radius', 'SPLA', 'EAA', 'speed', 'flight_path_angle',
    'load_factor', 'frequency', 'amplitude', 'coverage', 'run_metadata'], defaults=(None,))


#: Per-group run metadata in a database (:func:`add_sphere_group`'s ``run_metadata``):
#: the name of each numeric group attribute (f8, NaN when unknown) and its units, written
#: beside it as the text attribute ``<name>_units`` (None: the wind units the caller
#: declared).  They are group attributes, not variables: NICEOPS's nc_open reads every
#: dataset's metadata in every group eagerly, and six one-element variables per group
#: made a 1432-group database (B407_ambient_gated) load 0.23 s (31%) slower in every
#: NICEOPS run, where the same values as attributes, with their units, cost nothing
#: measurable (0.72 s either way).
GROUP_RUN_METADATA_NUMBERS = (
    ('gross_weight', 'N'),
    ('air_density', 'kg m-3'),
    ('wind_along_track', None),
    ('wind_cross_track', None),
    ('advance_ratio_air', '1'),
    ('thrust_coefficient_run', '1'),
)


#: Per-group text attributes alongside them.
GROUP_RUN_METADATA_TEXT = ('wind_source', 'wind_units', 'wind_reference_direction',
                           'air_density_source', 'source_sphere', 'condition_origin')


#: Values of a group's condition_origin.
CONDITION_ORIGINS = ('measured', 'extended_flight_path_angle', 'synthesized_hover')


# ---- Vehicle files ----

#: Standard gravity, m/s^2: vehicle weights in the configs are masses in kg.
STANDARD_GRAVITY = 9.80665


def read_vehicle_rotor_data(filename):
    """Main rotor radius (m), tip speed (m/s), weight (N) and density (kg/m^3) of a vehicle file.

    Takes either the NICEOPS vehicle JSON or panam's ``vehicle.cfg``; both have
    sections ``Main Rotor`` (radius, tip speed), ``Vehicle`` (weight, a mass in
    kg) and ``Atmosphere`` (density, the one the thrust coefficient is formed
    with).  Refuses a file that lacks any of them.
    """
    path = os.path.abspath(os.path.expanduser(str(filename)))
    if not os.path.isfile(path):
        raise FileNotFoundError(f'vehicle file not found: {filename}')
    if path.lower().endswith('.json'):
        with open(path, 'r', encoding='utf-8') as handle:
            sections = json.load(handle)
    else:
        config = ConfigParser()
        config.read(path)
        sections = {name: dict(config[name]) for name in config.sections()}
    wanted = (('Main Rotor', 'radius'), ('Main Rotor', 'tip speed'), ('Vehicle', 'weight'),
              ('Atmosphere', 'density'))
    values = {}
    for section, key in wanted:
        try:
            values[(section, key)] = float(sections[section][key])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f'{filename}: no usable [{section}] "{key}"') from None
    return dict(main_rotor_radius_m=values[('Main Rotor', 'radius')],
                main_rotor_tip_speed_mps=values[('Main Rotor', 'tip speed')],
                vehicle_weight_newtons=STANDARD_GRAVITY * values[('Vehicle', 'weight')],
                air_density_kg_m3=values[('Atmosphere', 'density')])


def read_vehicle_weight_newtons(directory_name):
    """Vehicle weight in newtons from a sphere directory's ``vehicle.cfg``.

    Separate from :func:`read_vehicle_data` so the database builder can write
    the root ``vehicle_weight_newtons`` without changing that function's
    fourteen-element return tuple.  Returns None when the config does not carry
    a weight.
    """
    config = ConfigParser()
    config.read(os.path.abspath(os.path.expanduser(directory_name) + '/vehicle.cfg'))
    if not config.has_section('Vehicle') or 'weight' not in config['Vehicle']:
        return None
    return STANDARD_GRAVITY * float(config['Vehicle']['weight'])


def read_vehicle_data(directory_name, runs=None, speeds=None, flight_path_angles=None):
    """Rotor and vehicle data from a sphere directory's ``vehicle.cfg``.

    Args:
        directory_name: the directory holding ``vehicle.cfg``.
        runs: run numbers to look up in the ``[Option]`` reflist workbook; used only when the
            config has an ``[Option]`` section.
        speeds: flight speeds, kt (replaced by the reflist's indicated airspeeds when it is used).
        flight_path_angles: flight path angles, deg.

    Returns: a 14-tuple (main rotor radius m, main rotor area m^2, main rotor tip speed m/s,
        main rotor blade count, tail rotor radius m, tail rotor area m^2, tail rotor tip speed
        m/s, tail rotor blade count, advance ratios, weight coefficients, hover tip Mach numbers,
        alphas deg, nondimensional flat-plate drag, speeds kt).  With an ``[Option]`` reflist
        and ``runs``, the rotor radii, areas and tail tip speed are None and the main rotor tip
        speed is the mean of the reflist's airspeeds over its advance ratios.  Otherwise the
        advance ratios and alphas are None without ``speeds``.  Alphas are None without
        ``flight_path_angles``.
    """
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
        absolute_path_to_list = os.path.abspath(os.path.expanduser(directory_name) + '/' + reference_list)
        import openpyxl                         # here: only the [Option] reference list needs it
        wb = openpyxl.load_workbook(absolute_path_to_list, read_only=True, data_only=True)
        try:
            ws = wb.active
            if ws is None:
                raise ValueError("Workbook has no active sheet")
            # One row per run: the run number in column B, the indicated
            # airspeed in W, and advance ratio, weight coefficient and hover
            # tip Mach number in AH:AJ.  Read whole rows, so a blank run cell
            # drops its own row rather than shifting every later run onto the
            # conditions of the row below.
            rows = [row for row in ws.iter_rows(min_row=2, max_col=36, values_only=True)
                    if row[1] is not None]
        finally:
            # A read-only workbook holds its file open until closed
            wb.close()
        run_numbers = [row[1] for row in rows]
        selected = [rows[run_numbers.index(r)] for r in runs]
        advance_ratios = np.array([row[33] for row in selected])
        weight_coefficients = np.array([row[34] for row in selected])
        hover_tip_mach_numbers = np.array([row[35] for row in selected])
        # The reflist's indicated airspeeds replace the caller's speeds.
        speeds = np.array([row[22] for row in selected])
        main_rotor_tip_speed = np.mean(KNOT_MPS * speeds / advance_ratios)
        # The reflist gives no rotor geometry.
        main_rotor_radius = None
        main_rotor_area = None
        tail_rotor_radius = None
        tail_rotor_area = None
        tail_rotor_tip_speed = None
        alphas = None
        if flight_path_angles is not None:
            drag_to_weight_ratio = 0.5 * nondimensional_flat_plate_drag * advance_ratios ** 2 / weight_coefficients
            alphas = -np.degrees(drag_to_weight_ratio) - flight_path_angles
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
        vehicle_weight_newtons = STANDARD_GRAVITY * vehicle_weight_kg
        effective_flat_plate_drag_area = np.array(float(config['Vehicle']['drag']))
        nondimensional_flat_plate_drag = effective_flat_plate_drag_area / main_rotor_area
        hover_tip_mach_numbers = main_rotor_tip_speed / speed_of_sound
        weight_coefficients = vehicle_weight_newtons / (
                ambient_density * main_rotor_area * main_rotor_tip_speed ** 2)
        advance_ratios = None
        alphas = None
        if speeds is not None:
            ground_speed_meters_per_sec = KNOT_MPS * speeds
            advance_ratios = ground_speed_meters_per_sec / main_rotor_tip_speed
            hover_tip_mach_numbers = hover_tip_mach_numbers * np.ones_like(advance_ratios)
            drag_to_weight_ratio = 0.5 * nondimensional_flat_plate_drag * advance_ratios ** 2 / weight_coefficients
            if flight_path_angles is not None:
                alphas = -np.degrees(drag_to_weight_ratio) - flight_path_angles

    return (main_rotor_radius, main_rotor_area, main_rotor_tip_speed,
            number_of_main_rotor_blades, tail_rotor_radius, tail_rotor_area, tail_rotor_tip_speed,
            number_of_tail_rotor_blades, advance_ratios, weight_coefficients, hover_tip_mach_numbers, alphas,
            nondimensional_flat_plate_drag, speeds)


# ---- Reducing a sphere to group levels ----

def extract_SPL(filename, infreqs=None, distance=1000,
                atmosphere=Atmosphere(temperature=293.15, pressure=101.325,
                                      relative_humidity=20.0), absorption_band_method='center',
                sphere_frame=None):
    """Read a sphere file and reduce it to the levels a database group carries, in the level frame.

    ``distance`` (m) and ``atmosphere`` set the EAA, with each band's attenuation
    formed by ``absorption_band_method`` (see :func:`spla_and_eaa_from_spectrum`).
    ``sphere_frame`` is the frame the file's directions are in
    (:data:`geometry.SPHERE_FRAMES`); None takes the file's own ``sphere_frame``, or
    'level' when it records none.  A flight-path sphere is pitched to level
    (:func:`gridding.repitch_art_levels`).
    """
    if sphere_frame is None:
        sphere_frame = read_sphere_frame(filename) or 'level'
    return _reduce_sphere_levels(*load_nc_sphere(filename), infreqs, distance, atmosphere,
                                 absorption_band_method, sphere_frame)


def _reduce_sphere_levels(amplitude, phi, theta, frequency, radius, speed, flight_path_angle,
                          infreqs, distance, atmosphere, absorption_band_method, sphere_frame='level',
                          with_measured=False):
    """:func:`extract_SPL` from levels already read, as :func:`load_nc_sphere` returns them.

    amplitude is dB (phi, theta, frequency), any sentinel or NaN read as no energy;
    radius is ft, speed kt, angles deg; sphere_frame is the frame amplitude is filed
    in.  Returns what :func:`extract_SPL` returns: (azimuth, elevation, phi, theta,
    radius m, SPLO, SPLA, EAA, speed, flight path angle, frequency, amplitude), the
    levels level.  with_measured appends ``measured``, (phi, theta, band) bool, True
    where a band is read from measured cells of the file's own hemisphere (every
    finite one of a level sphere), for the builder's coverage.
    """
    if sphere_frame not in SPHERE_FRAMES:
        raise ValueError('sphere_frame must be one of {}, not {!r}'.format(SPHERE_FRAMES, sphere_frame))
    # Convert radius to meters
    radius = radius * 0.3048
    # Replace masked values with no-energy SPL
    amplitude = mask_missing_levels(amplitude)
    # Check frequency range
    if infreqs is not None:
        frequency_index = (frequency >= infreqs[0]) & (frequency <= infreqs[1])
        amplitude = amplitude[:, :, frequency_index]
        frequency = frequency[frequency_index]
    if sphere_frame == 'flight_path':
        # AAM's frame: pitched nose-up by the flight path angle (THEORY 3.2).
        amplitude, measured = repitch_art_levels(amplitude, phi, theta, float(np.ravel(flight_path_angle)[0]))
    else:
        measured = np.isfinite(amplitude)
    SPLO = power_to_db(np.sum(np.power(10.0, amplitude / 10.0), axis=2))
    # A-weighted OASPL and excess atmospheric attenuation
    SPLA, EAA = spla_and_eaa_from_spectrum(amplitude, frequency, distance, atmosphere, absorption_band_method)
    # Convert from ART to UMAPR coordinates
    _, P = np.meshgrid(theta, phi)
    azi, elv = art2umapr(np.pi * P / 180.0, np.pi * theta / 180.0)
    # frequency/amplitude are returned so callers can recompute band-based
    # quantities (e.g. EAA) after modifying the spectrum.
    out = (azi, elv, phi, theta, radius, SPLO, SPLA, EAA, speed, flight_path_angle, frequency, amplitude)
    return out + (measured,) if with_measured else out


def spla_and_eaa_from_spectrum(spectrum, frequency, distance, atmosphere, absorption_band_method='center'):
    """Derive A-weighted OASPL and excess atmospheric attenuation from a spectrum.

    Mirrors the reduction in :func:`extract_SPL`, so a caller that modifies the
    spectrum (e.g. fore/aft averaging for hover) can recompute both quantities
    consistently instead of carrying stale values forward.

    Args:
        spectrum: (phi, theta, frequency) band levels, dB.
        frequency: band centers, Hz.
        distance: the distance the EAA is formed over, m.
        atmosphere: the Atmosphere of the absorption (K, kPa, percent).
        absorption_band_method: how each band's attenuation over ``distance`` is
            formed (:func:`panam.acoustics.atmosphere.band_absorption_db`):
            'center' (default, alpha at the band center times the distance),
            'arp5534' or 'integrated' (flat band levels).

    Returns: (SPLA, dBA; EAA, dB over ``distance``), each (phi, theta).
    """
    a_weight = np.array([dBAw(f) for f in frequency])
    attenuation = _band_attenuation_db(atmosphere, frequency, float(distance), absorption_band_method)
    spla = np.apply_along_axis(OASPL, 2, spectrum + a_weight)
    spla_attenuated = np.apply_along_axis(OASPL, 2, spectrum + a_weight - attenuation)
    with np.errstate(invalid='ignore'):
        eaa = spla - spla_attenuated
    return spla, eaa


def _finite_sphere_levels(spla, eaa):
    """Sanitize EAA where ambient gating emptied a direction.

    Ambient gating legitimately empties a direction -- a low-power descent can
    sit at the noise floor over most of the aft hemisphere -- and then
    ``SPLA = -inf``, so ``EAA = SPLA - SPLAa`` is ``-inf - -inf``, a NaN.

    The level itself is left alone, including when it is -inf. That was checked
    against NICEOPS rather than assumed (its build/niceops, B205 track, Be407
    database, patching the directions a flyover actually uses so the test is
    demonstrably sensitive -- 26.9 dB of footprint change):

    * ``dBA = -inf`` and ``dBA = -3064`` give byte-identical footprints with no
      NaN cells. Its ``eval`` subtracts spreading and attenuation from the
      level and the exposure integral then sums energy, so a -inf direction
      contributes nothing -- which is exactly what an empty direction means.
    * ``EAA = NaN`` does not stay local: it put 886 NaN cells into a 120x120
      footprint and moved the mean level by 1.3 dB.

    So only EAA needs a value here, and zero is the right one: with no energy
    in the direction there is no excess attenuation to apply to it.
    """
    spla = np.asarray(spla, dtype=float)
    eaa = np.asarray(eaa, dtype=float).copy()
    eaa[~np.isfinite(eaa)] = 0.0
    return spla, eaa


def _measured_bins(amplitude, eaa):
    """(n_phi, n_theta, n_bands) bool: True where a bin carries a measurement.

    A bin is measured when its level is finite after :func:`mask_missing_levels`,
    which turns the masked, gated and empty bins into -inf.  A direction whose
    EAA is not finite is not measured either: :func:`_finite_sphere_levels`
    zeroes that EAA, so the value is a fill, not an observation.
    """
    return np.isfinite(amplitude) & np.isfinite(eaa)[:, :, None]


# ---- Completion and hover ----

def _complete_sphere(phi_list, theta_list, spla, eaa, amplitude):
    """Extend a lower-hemisphere sphere onto its upper surface.

    Returns ``(phi, theta, spla, eaa, amplitude)`` with the reflected rows
    appended.  ``phi`` and ``theta`` come back as meshgrid arrays matching the
    data, so every array stays row-aligned with its azimuth -- the failure mode
    of the three-part slicing this replaces.
    """
    phi, theta, spla, eaa, amplitude, _ = _complete_sphere_with_coverage(
        phi_list, theta_list, spla, eaa, amplitude, measured=None)
    return phi, theta, spla, eaa, amplitude


def _complete_sphere_with_coverage(phi_list, theta_list, spla, eaa, amplitude, measured):
    """:func:`_complete_sphere`, and the coverage of the completed rows.

    ``measured`` is (n_phi, n_theta, n_bands) bool over the source rows, True
    where a bin carries a measurement (see :func:`_measured_bins`), or None.
    The returned coverage has the completed shape and is True only on source
    rows, where the bin is measured.  The reflected upper surface is never
    measured: it is a symmetry fill, so it is recorded as unmeasured.  Returns
    None for coverage when ``measured`` is None.
    """
    mirror_phi, source_index = mirror_phi_to_upper_surface(phi_list)
    n_source = len(phi_list)
    phi_full_list = np.concatenate((np.asarray(phi_list, dtype=float), mirror_phi))
    # Rows sorted by phi, so the spectra -- written gridded, without angles of
    # their own -- are in the sorted order NICEOPS reads them in.  Until
    # 2026-09-25 they were left in completion order (-90..90, then the mirrored
    # upper surface), which NICEOPS read as sorted and so put every spectrum
    # in the wrong direction.  dBA/EAA carry their angles per channel and were
    # never affected.
    order = np.argsort(phi_full_list, kind='stable')
    rows = np.concatenate((np.arange(n_source), source_index))[order]
    # Which completed rows are source rows.  Mirror rows index source data too
    # (source_index), so the row index alone cannot say which is which.
    is_source = np.concatenate((np.ones(n_source, dtype=bool),
                                np.zeros(source_index.size, dtype=bool)))[order]
    theta, phi = np.meshgrid(theta_list, phi_full_list[order])
    coverage = None
    if measured is not None:
        coverage = measured[rows] & is_source[:, None, None]
    return phi, theta, spla[rows], eaa[rows], amplitude[rows], coverage


def mirror_phi_to_upper_surface(phi_list):
    """Complete a lower-hemisphere sphere onto its upper surface.

    Sphere data covers only the lower half of the roll circle, phi in
    [-90, 90] degrees with phi = 0 straight down.  The upper half is the
    reflection through the horizontal plane, phi -> 180 - phi wrapped into
    [-180, 180):  0 (down) -> -180 (up; the same direction as 180), and +/-90
    map to themselves because the horizontal plane belongs to both halves.

    Returns (mirror_phi, source_index): the azimuths of the reflected rows
    and the indices of the source rows they came from, so that every data array
    can be extended consistently::

        mirror_phi, idx = mirror_phi_to_upper_surface(phi_list)
        phi_full  = np.concatenate((phi_list, mirror_phi))
        SPLA_full = np.concatenate((SPLA, SPLA[idx]))

    Rows whose reflection already exists in phi_list are dropped, so the
    shared +/-90 edges appear once rather than twice and the completed sphere
    has no duplicate (phi, theta) points.

    NOTE on two earlier forms, both of which produced bad databases:

    * np.concatenate((phi, -phi)) is a left/right flip *within* the lower
      half.  Because the source range is symmetric about zero it maps [-90, 90]
      onto itself, emitting a second copy of the lower hemisphere under the same
      labels and leaving the upper surface empty.  The hover spheres in
      S-76D_M3.nod were built this way: every (phi, theta) there carries two
      conflicting levels, 630 of 703 pairs disagreeing by up to 5.77 dB.
    * A three-part concatenation that sliced phi as [h+1:n], [:], [1:h]
      but the data arrays as [1:h], [:], [h+1:n].  Those slices have equal
      length only when len(phi_list) is even; the real sphere grids have 19
      azimuths, so the labels ran one row out of step with the acoustic data
      across most of the sphere -- corrupting forward-flight spheres too.
    """
    phi_list = np.asarray(phi_list, dtype=float)
    reflected = (180.0 - phi_list + 180.0) % 360.0 - 180.0
    # Drop reflections that coincide with a source azimuth (the +/-90 edges),
    # comparing on a rounded grid so exact float equality is not required.
    already_present = np.isin(np.round(reflected, 9), np.round(phi_list, 9))
    source_index = np.flatnonzero(~already_present)
    return reflected[source_index], source_index


def average_fore_and_aft(spectrum):
    """Average a sphere fore-to-aft on an energy (p^2) basis.

    ``spectrum`` is dB per band with shape (phi, theta, frequency); theta runs
    nose (0) to tail (180), so band j = n-1-i is the fore/aft partner of band i.
    Each pair is replaced by their mean-square average::

        10 * log10( 0.5 * (10**(Li/10) + 10**(Lj/10)) )

    A plain arithmetic mean of decibels is not an energy average and biases the
    result low -- about 0.4 dB for a pair 5.8 dB apart, which is the spread seen
    between fore/aft partners in real spheres.  The middle band of an odd-length
    theta axis is its own partner and is left unchanged.
    """
    spectrum = np.array(spectrum, dtype=float, copy=True)
    n_theta = spectrum.shape[1]
    # -inf entries mean "no energy" and are expected; 10**(-inf/10) == 0.
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(n_theta // 2):
            j = n_theta - 1 - i
            mean_square = 0.5 * (10.0 ** (spectrum[:, i, :] / 10.0)
                                 + 10.0 ** (spectrum[:, j, :] / 10.0))
            averaged = 10.0 * np.log10(mean_square)
            spectrum[:, i, :] = averaged
            spectrum[:, j, :] = averaged
    return spectrum


def _fore_aft_measured(measured):
    """Coverage of a fore-to-aft averaged spectrum (see :func:`average_fore_and_aft`).

    An averaged pair is measured only where both partners are.  Averaging a
    measured bin with a gated partner gives both cells a level taken from the
    partner, which is a fill, not a measurement.  The middle theta of an
    odd-length axis is its own partner and is left as it is.
    """
    averaged = np.array(measured, dtype=bool, copy=True)
    n_theta = averaged.shape[1]
    for i in range(n_theta // 2):
        j = n_theta - 1 - i
        both = measured[:, i, :] & measured[:, j, :]
        averaged[:, i, :] = both
        averaged[:, j, :] = both
    return averaged


#: Flight path angles, degrees, a synthesized hover is written at when no
#: extended angles are given (see :func:`_hover_flight_path_angles`).
HOVER_FLIGHT_PATH_ANGLES = (-12, 0, 12)


def _hover_flight_path_angles(extended_flight_path_angles=None):
    """Flight path angles, degrees, a synthesized hover is written at.

    0 and each of ``extended_flight_path_angles`` (degrees) when given, sorted;
    otherwise :data:`HOVER_FLIGHT_PATH_ANGLES`.  A hover has no flight path, so
    the copies only keep the condition hull from clamping a steep, slow state.
    """
    if extended_flight_path_angles is not None:
        return sorted({0.0, *extended_flight_path_angles})
    return list(HOVER_FLIGHT_PATH_ANGLES)


def _hover_source_index(conditions, level_flight_tolerance=LEVEL_FLIGHT_TOLERANCE):
    """Index of the condition a hover is synthesized from.

    ``conditions`` is a sequence of (speed in knots, flight path angle in
    degrees).  The slowest condition within ``level_flight_tolerance`` degrees
    of level flight is picked, the first on a tie; if none is that close, the
    slowest overall, with a warning (the further from level flight, the rougher
    the hover approximation).
    """
    if len(conditions) == 0:
        raise ValueError('no conditions to synthesize a hover from')
    min_speed, min_index = np.inf, None
    slowest_speed, slowest_index = np.inf, None
    for index, (speed, flight_path_angle) in enumerate(conditions):
        if speed < min_speed and np.abs(flight_path_angle) < level_flight_tolerance:
            min_speed, min_index = speed, index
        if speed < slowest_speed:
            slowest_speed, slowest_index = speed, index
    if min_index is None:
        warnings.warn('No sphere within {:g} degrees of level flight; synthesizing hover from the '
                      'slowest available at {:.1f} knots.'.format(level_flight_tolerance, slowest_speed))
        return slowest_index
    return min_index


def _synthesized_hover_sphere(phi_list, theta_list, amplitude, measured, frequency, distance,
                              atmosphere, hover_correction=None, clamp_empty_directions=True,
                              absorption_band_method='center'):
    """Synthesize a completed hover sphere from a slow flight sphere's band levels (THEORY 7.4).

    Shared by :func:`build_empirical_database` and :func:`norah2.build_database_from_norah2`.
    The spectrum is averaged fore to aft in energy (:func:`average_fore_and_aft`),
    a cell is measured only where both partners were (:func:`_fore_aft_measured`),
    ``hover_correction`` (dB; see :func:`build_empirical_database`) is added,
    SPLA and EAA are recomputed from the result (:func:`spla_and_eaa_from_spectrum`),
    and the sphere is completed onto its upper surface.

    Args:
        phi_list, theta_list: the lower-hemisphere axes, degrees.
        amplitude: (phi, theta, band) levels at the sphere radius, dB, -inf where missing.
        measured: (phi, theta, band) bool, True where the source bin is measured.
        frequency: band centers, Hz.
        distance: the EAA distance, m.
        atmosphere: the atmosphere of the EAA.
        hover_correction: None, or a callable as in :func:`build_empirical_database`.
        clamp_empty_directions: zero the EAA of a direction with no energy
            (:func:`_finite_sphere_levels`).
        absorption_band_method: how each band's attenuation over ``distance``
            enters the EAA (:func:`spla_and_eaa_from_spectrum`).

    Returns: ``(phi, theta, SPLA, EAA, amplitude, coverage)`` of the completed sphere,
        as :func:`_complete_sphere_with_coverage` returns them (dB, dB per ``distance``).
    """
    # The averaging below gives a gated bin the level of its partner, so the
    # hover's coverage is measured only where both partners were.
    measured = _fore_aft_measured(measured)
    # Average the SPECTRUM fore-to-aft on an energy basis, then derive SPLA and
    # EAA from the averaged spectrum.  Averaging the broadband dB levels instead
    # (a) is not an energy average, and (b) leaves EAA untouched, so level and
    # excess attenuation end up describing different spheres.
    amplitude = average_fore_and_aft(amplitude)
    if hover_correction is not None:
        theta_grid, phi_grid = np.meshgrid(np.asarray(theta_list, dtype=float),
                                           np.asarray(phi_list, dtype=float))
        correction = np.asarray(hover_correction(phi_grid, theta_grid, np.asarray(frequency, dtype=float)),
                                dtype=float)
        if correction.shape != amplitude.shape:
            raise ValueError('hover_correction returned shape {}, the sphere is {}'.format(
                correction.shape, amplitude.shape))
        amplitude = amplitude + correction
    SPLA, EAA = spla_and_eaa_from_spectrum(amplitude, frequency, distance, atmosphere,
                                           absorption_band_method)
    if clamp_empty_directions:
        SPLA, EAA = _finite_sphere_levels(SPLA, EAA)
    # Complete the sphere onto its upper surface
    return _complete_sphere_with_coverage(phi_list, theta_list, SPLA, EAA, amplitude, measured)


# ---- Source spheres and run metadata ----

def _select_sphere_files(directory_name, spheres=None, exclude=None):
    """The sphere files :func:`build_empirical_database` uses, and those ``exclude`` drops.

    Returns ``(used, excluded)``, two sorted lists of absolute paths.  Without
    ``spheres``, the candidates are every ``*.nc`` in directory_name; with it,
    the files it names, a relative path taken relative to directory_name.  A
    named file that does not exist raises FileNotFoundError.  ``exclude`` holds
    file names or ``fnmatch`` patterns matched against each file name; one that
    matches no candidate is warned about, since it is most likely a typo.
    """
    directory = os.path.abspath(os.path.expanduser(directory_name))
    if spheres is None:
        candidates = glob(os.path.join(directory, '*.nc'))
    else:
        if isinstance(spheres, (str, os.PathLike)):
            raise TypeError('spheres must be a list of files, not one file')
        candidates = []
        for sphere in spheres:
            path = os.path.join(directory, os.path.expanduser(os.fspath(sphere)))
            if not os.path.isfile(path):
                raise FileNotFoundError('sphere file not found: {}'.format(path))
            candidates.append(os.path.abspath(path))
    candidates = sorted(set(candidates))
    if exclude is None:
        return candidates, []
    if isinstance(exclude, (str, os.PathLike)):
        exclude = [exclude]
    patterns = [os.path.basename(os.fspath(pattern)) for pattern in exclude]
    for pattern in patterns:
        if not any(fnmatch.fnmatchcase(os.path.basename(f), pattern) for f in candidates):
            warnings.warn('exclude pattern {!r} matches no sphere file'.format(pattern))
    excluded = [f for f in candidates
                if any(fnmatch.fnmatchcase(os.path.basename(f), p) for p in patterns)]
    return [f for f in candidates if f not in excluded], excluded


def _sphere_condition(filename):
    """The (SPEED kt, FLIGHT_PATH_ANGLE deg) a sphere file records, NaN where unreadable."""
    with Dataset(filename, mode='r') as handle:
        return _read_scalar(handle, 'SPEED'), _read_scalar(handle, 'FLIGHT_PATH_ANGLE')


def _group_repeat_conditions(filenames, speed_tolerance_knots=0.0):
    """Group sphere files by flight condition, for :func:`build_empirical_database`.

    Two files share a condition when their FLIGHT_PATH_ANGLE agrees to 1e-9 deg
    and their SPEED agrees to 1e-9 kt, or lies within speed_tolerance_knots (kt)
    of the slowest file of the group.  Returns a list of tuples of file names,
    each in the order of ``filenames``, ordered by its first file, so that
    files with no repeat give one 1-tuple each, in their own order.
    """
    conditions = [_sphere_condition(f) for f in filenames]
    order = sorted(range(len(filenames)),
                   key=lambda i: (round(conditions[i][1], 9), conditions[i][0], i))
    groups = []
    for i in order:
        speed, angle = conditions[i]
        if groups:
            first_speed, first_angle = conditions[groups[-1][0]]
            if round(angle, 9) == round(first_angle, 9) and (
                    round(speed, 9) == round(first_speed, 9)
                    or speed - first_speed <= speed_tolerance_knots):
                groups[-1].append(i)
                continue
        groups.append([i])
    return [tuple(filenames[i] for i in sorted(group)) for group in sorted(groups, key=min)]


def _extract_condition_SPL(filenames, infreqs, distance, atmosphere, absorption_band_method,
                           speed_tolerance_knots=0.0, sphere_frame='level'):
    """:func:`extract_SPL` for one condition: one sphere file, or the energy average of several.

    Several files are averaged by :func:`sphere_statistics.average_spheres`, a -999
    cell read as no data, after which a cell no run covers is no energy, as
    :func:`extract_SPL` reads a -999.  speed_tolerance_knots (kt) is the SPEED
    spread allowed between them.  The speed returned is the runs' mean, kt.
    sphere_frame is the frame the files are filed in; they are averaged in it and
    the average pitched to level.  Returns :func:`extract_SPL`'s tuple and the
    ``measured`` mask of :func:`_reduce_sphere_levels`.
    """
    if len(filenames) == 1:
        return _reduce_sphere_levels(*load_nc_sphere(filenames[0]), infreqs, distance, atmosphere,
                                     absorption_band_method, sphere_frame, with_measured=True)
    average = sphere_statistics.average_spheres(
        filenames, speed_tolerance_knots=max(float(speed_tolerance_knots), 1e-9),
        flight_path_angle_tolerance_deg=1e-9)
    return _reduce_sphere_levels(
        average['levels_db'], average['phi_deg'], average['theta_deg'], average['frequency_hz'],
        np.asarray(average['radius_ft']), np.asarray(average['speed_knots']),
        np.asarray(average['flight_path_angle_deg']), infreqs, distance, atmosphere,
        absorption_band_method, sphere_frame, with_measured=True)


def group_run_metadata(filename, main_rotor_area=None, main_rotor_tip_speed=None,
                       condition_origin='measured'):
    """The per-group run metadata for a condition whose levels came from sphere ``filename``.

    Reads the sphere's :func:`read_run_metadata` and forms the two dimensionless labels a
    database can be relabeled by later:

    * ``advance_ratio_air``: the run's horizontal airspeed (ground velocity minus the wind
      at the aircraft) over the tip speed, as ``advance_ratio`` is its horizontal ground
      speed over the tip speed;
    * ``thrust_coefficient_run``: the run's own gross weight over rho A V_tip^2 at the run's
      air density, where ``thrust_coefficient`` uses the vehicle file's nominal weight and
      density.  It is the run's, not scaled by the group's load factor.

    NaN wherever an input is unknown.  A synthesized hover keeps the source run's weight,
    density and C_T (its levels are that run's), but no wind or advance ratio, which
    describe a flight condition the hover is not.
    """
    if condition_origin not in CONDITION_ORIGINS:
        raise ValueError('condition_origin must be one of {}, not {!r}'.format(CONDITION_ORIGINS, condition_origin))
    run = read_run_metadata(filename)
    nan = float('nan')
    area = nan if main_rotor_area is None else float(main_rotor_area)
    tip_speed = nan if main_rotor_tip_speed is None else float(main_rotor_tip_speed)
    weight_newtons = run['gross_weight_lb'] * POUND_FORCE_NEWTONS
    density = run['air_density_kg_m3']
    out = dict(gross_weight=weight_newtons, air_density=density,
               wind_along_track=run['wind_along_track'], wind_cross_track=run['wind_cross_track'],
               advance_ratio_air=KNOT_MPS * run['airspeed_knots'] / tip_speed,
               thrust_coefficient_run=weight_newtons / (density * area * tip_speed ** 2),
               wind_source=run['wind_source'] or 'none', wind_units=run['wind_units'],
               wind_reference_direction=run['wind_reference_direction'],
               air_density_source=run['air_density_source'] or 'none',
               source_sphere=os.path.basename(str(filename)), condition_origin=condition_origin)
    if condition_origin == 'synthesized_hover':
        out.update(wind_along_track=nan, wind_cross_track=nan, advance_ratio_air=nan,
                   wind_source='none', wind_units='', wind_reference_direction='')
    return out


def _merged_run_metadata(rows):
    """One condition's run metadata from the :func:`group_run_metadata` of each run averaged into it.

    One row is returned as it is.  Several give the mean of each numeric value (NaN
    if any run's is unknown, and for the winds if the runs declare different wind
    units or wind reference directions, which make their winds incommensurable),
    the first run's text values, except wind_source, which names every source the
    runs used, '+'-joined in run order ('none' only if no run had one), and every
    run's source_sphere, one per line.
    """
    if len(rows) == 1:
        return rows[0]
    merged = dict(rows[0])
    for name, _ in GROUP_RUN_METADATA_NUMBERS:
        merged[name] = float(np.mean([row[name] for row in rows]))
    if any(len({row[key] for row in rows}) > 1
           for key in ('wind_units', 'wind_reference_direction')):
        merged.update(wind_along_track=float('nan'), wind_cross_track=float('nan'))
    sources = [part for row in rows for part in row['wind_source'].split('+')
               if part not in ('', 'none')]
    merged['wind_source'] = '+'.join(dict.fromkeys(sources)) or 'none'
    merged['source_sphere'] = '\n'.join(row['source_sphere'] for row in rows)
    return merged


def _resolve_doppler_shift_removed(filenames, argument=None):
    """The single DOPPLER_SHIFT_REMOVED flag every source sphere shares.

    A sphere that carries the flag is taken at its word, and an ``argument``
    given alongside must agree with it.  A sphere without the flag takes the
    argument, or 0 (received frame) when there is none, with a warning.  Sources
    that disagree would be combined into one database, so they are refused.
    Returns the shared flag, 0 when there are no sources.
    """
    if argument is not None and argument not in (0, 1):
        raise ValueError('doppler_shift_removed must be 0, 1 or None, not {!r}'.format(argument))
    fallback = 0 if argument is None else int(argument)
    resolved = {}
    unlabeled = []
    for filename in filenames:
        flag = _read_doppler_shift_removed(filename)
        if flag is None:
            unlabeled.append(filename)
            resolved[filename] = fallback
            continue
        if argument is not None and flag != argument:
            raise ValueError('{} carries DOPPLER_SHIFT_REMOVED = {}, but the build was asked for {}'
                             .format(os.path.basename(filename), flag, argument))
        resolved[filename] = flag
    if unlabeled and argument is None:
        warnings.warn('{} source sphere(s) carry no DOPPLER_SHIFT_REMOVED; assuming 0 '
                      '(received-frame spheres)'.format(len(unlabeled)))
    values = set(resolved.values())
    if len(values) > 1:
        groups = ['{}: {}'.format(value, ', '.join(os.path.basename(f) for f, v in resolved.items()
                                                    if v == value))
                  for value in sorted(values)]
        raise ValueError('DOPPLER_SHIFT_REMOVED differs between the sources of one database '
                         '(' + '; '.join(groups) + '); combine spheres of one convention only')
    return values.pop() if values else fallback


def _resolve_sphere_frames(filenames, argument=None):
    """The frame each source sphere is filed in, ``{filename: frame}``.

    As :func:`_resolve_doppler_shift_removed`: a sphere that records one is taken at its
    word and an ``argument`` given must agree with it; a sphere without one takes the
    argument, or 'level' (every sphere PANAM wrote before the attribute existed), with a
    warning, since AAM's own spheres are in the flight-path frame.  Unlike the other two,
    frames may differ between the sources: each sphere is pitched to level as it is read.
    """
    if argument is not None and argument not in SPHERE_FRAMES:
        raise ValueError('sphere_frame must be one of {} or None, not {!r}'.format(SPHERE_FRAMES, argument))
    fallback = 'level' if argument is None else argument
    resolved = {}
    unlabeled = []
    for filename in filenames:
        value = read_sphere_frame(filename)
        if value is None:
            unlabeled.append(filename)
            resolved[filename] = fallback
            continue
        if argument is not None and value != argument:
            raise ValueError('{} carries sphere_frame = {!r}, but the build was asked for {!r}'
                             .format(os.path.basename(filename), value, argument))
        resolved[filename] = value
    if unlabeled and argument is None:
        warnings.warn("{} source sphere(s) carry no sphere_frame; assuming 'level' (PANAM's). AAM's own "
                      "spheres are in the flight-path frame: pass sphere_frame='flight_path' for "
                      "them".format(len(unlabeled)))
    return resolved


def _resolve_azimuth_reference(filenames, argument=None):
    """The single azimuth_reference every source sphere shares.

    As :func:`_resolve_doppler_shift_removed`: a sphere that carries one is taken at its
    word and an ``argument`` given must agree with it; a sphere without one takes the
    argument, or 'track' (what every sphere was built in before the attribute existed),
    with a warning.  A database filed in two frames at once is refused.
    """
    if argument is not None and argument not in AZIMUTH_REFERENCES:
        raise ValueError('azimuth_reference must be one of {} or None, not {!r}'.format(
            AZIMUTH_REFERENCES, argument))
    fallback = 'track' if argument is None else argument
    resolved = {}
    unlabeled = []
    for filename in filenames:
        value = _read_azimuth_reference(filename)
        if value is None:
            unlabeled.append(filename)
            resolved[filename] = fallback
            continue
        if argument is not None and value != argument:
            raise ValueError('{} carries azimuth_reference = {!r}, but the build was asked for {!r}'
                             .format(os.path.basename(filename), value, argument))
        resolved[filename] = value
    if unlabeled and argument is None:
        warnings.warn("{} source sphere(s) carry no azimuth_reference; assuming 'track' "
                      "(azimuth from the ground track)".format(len(unlabeled)))
    values = set(resolved.values())
    if len(values) > 1:
        groups = ['{}: {}'.format(value, ', '.join(os.path.basename(f) for f, v in resolved.items()
                                                    if v == value))
                  for value in sorted(values)]
        raise ValueError('azimuth_reference differs between the sources of one database '
                         '(' + '; '.join(groups) + '); build every sphere in one frame')
    return values.pop() if values else fallback


# ---- Building a database ----

def build_empirical_database(directory_name, database_filename, load_factors=np.linspace(0.7, 2.3, 5), infreqs=None,
                             distance=1000,
                             atmosphere=Atmosphere(temperature=293.15, pressure=101.325,
                                                   relative_humidity=20.0),
                             extended_flight_path_angles=None,
                             level_flight_tolerance=LEVEL_FLIGHT_TOLERANCE,
                             store_spectrum=True, clamp_empty_directions=True,
                             hover_correction=None, hover_source=None,
                             speed_reference='ground', doppler_shift_removed=None,
                             azimuth_reference=None, spheres=None, exclude=None,
                             absorption_band_method='center', merge_repeats=False,
                             merge_speed_tolerance=0.0, sphere_frame=None):
    """Build a NICEOPS sphere database from a directory of sphere files.

    directory_name holds vehicle.cfg and, unless ``spheres`` says otherwise,
    the sphere files: every ``*.nc`` in it.  database_filename is the output,
    replaced if it exists.  distance is the propagation distance of the
    broadband EAA, m; infreqs, when given, is the (low, high) band range kept,
    Hz; level_flight_tolerance is in degrees of flight path angle.

    absorption_band_method sets how each band's attenuation over ``distance``
    (m) enters the EAA (:func:`spla_and_eaa_from_spectrum`): 'center' (default,
    the ISO 9613-1 coefficient at the band center), 'arp5534' (SAE ARP 5534) or
    'integrated' (ISO 9613-1 integrated over the band, flat band levels).  The
    database records it only in the provenance settings.

    spheres, when given, is the explicit list of sphere files to use in place
    of the directory's ``*.nc``; a relative path is taken relative to
    directory_name, so the names :func:`noise_abatement_2017.select_from_manifest`
    returns can be passed as they are.  exclude, when given, is a list of file
    names or shell-style patterns (``fnmatch``, matched against the file name)
    whose spheres are left out.  The root text attribute source_spheres always
    records the file names used, one per line, sorted; with exclude,
    excluded_spheres records those left out.  Neither holds a directory.

    load_factors scales thrust: each source condition is written once per load
    factor, with the sphere level offset by 20*log10(load_factor) and the
    thrust coefficient scaled to match.  Maneuver modeling needs this to span
    well beyond 1 g -- the shipped databases use
    [0, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0], finely spaced near 1 g and reaching
    a 5 g pull-up, rather than the uniform default here.

    Pass ``load_factors=None`` to write each condition once, at the LF=1
    reference only, and let NICEOPS scale levels (and, for a spectral
    database, every band) to the queried load factor analytically instead of
    from stored samples.  It carries the same information: every load factor
    this function would otherwise materialize is the *same* spectrum offset by
    a uniform ``20*log10(load_factor)`` -- add_sphere_group does not scale
    ``amplitude``, only ``dBA`` and ``thrust_coefficient`` -- so eight copies
    of it (the shipped databases' load factor count) store nothing the reader
    could not derive from one.  The results are not identical, though.  The
    fixed database gets the 20*log10 law exactly at every load factor above
    1 g and leaves lower ones unscaled; a ladder reproduces the law only at
    its stored load factors, NICEOPS interpolating linearly in C_T between
    them (at 4 g between stored 3 and 5 g, 11.76 dB rather than 12.04), and
    scales the entries below 1 g it stores (0.7 g: -3.1 dB) except LF=0,
    which is stored unscaled.  It shrinks a database
    roughly in proportion to the load factor count (measured on Be407: 547.9
    MB -> well under 100 MB). The database is written with a root
    ``fixed_load_factor`` flag so NICEOPS knows the scaling is safe to apply
    -- it must NOT be inferred for a database built some other way, e.g. from
    genuinely separate measurements at different thrust conditions whose
    spectral *shape*, not just level, could differ with load factor.

    extended_flight_path_angles, when given, widens the flight-path-angle
    envelope past what was measured.  Measured spheres cluster near level
    flight, so a trajectory that climbs or descends steeply would otherwise
    interpolate against a clamped edge.  Every near-level condition (within
    level_flight_tolerance of zero, hover included) is re-emitted at each of
    these angles at its own airspeed, on the assumption that directivity at a
    given airspeed carries over to a steeper flight path.  The shipped
    databases use (-24.0, 35.0).

    hover_source, when given, is the sphere file the hover is synthesized from in place of
    the slowest near-level one in the directory -- typically that same run rebuilt with
    ``noise_abatement_2017.build_sphere(remove_doppler=True)``, so that the fore-to-aft average combines
    spectra at their emitted frequencies.  The flight spheres in the directory keep
    their received ones.  hover_source is a path resolved against the current
    directory, not against sphere_directory as the names in ``spheres`` are.

    hover_correction, when given, is added to the synthesized hover sphere's band levels
    after the fore-to-aft average: a callable taking phi and theta (degrees, arrays shaped
    like the sphere's (phi, theta) grid) and the band centers (Hz), returning dB with shape
    (phi, theta, frequency).  It is how measured hovers, which cover only part of the
    sphere, are blended into the synthesized one; the caller owns where it applies and how
    it tapers.  Its ``description`` attribute, if any, is written to the database.

    clamp_empty_directions replaces the NaN that EAA becomes where ambient
    gating emptied a direction (``-inf - -inf``) with zero.  A NaN there is not
    local: it put 886 NaN cells into a NICEOPS footprint.  The dBA level is
    left as it falls out, -inf included; see :func:`_finite_sphere_levels`.

    store_spectrum keeps the source spectrum (frequency + amplitude) in each
    sphere group.  NICEOPS's spectral evaluation reads it, with the per-band
    coverage; without it only the broadband dBA/EAA path is available.  It also
    lets a database be re-reduced -- different weighting, a different
    propagation distance -- without the original sphere files.  It accounts for
    roughly seven eighths of the file size, so turn it off only for databases
    that will be used for broadband levels alone.

    The database records what it was built against, so NICEOPS can refuse a
    mismatch instead of passing it silently.  Root attributes: speed_reference
    ("ground" or "air"; the reference the conditions' advance ratio and flight
    path angle are taken in), and build_temperature_K, build_pressure_kPa and
    build_relative_humidity_percent, the atmosphere the broadband EAA was
    computed in (taken from ``atmosphere``).  Each condition group carries
    DOPPLER_SHIFT_REMOVED (0 received-frame, 1 de-Dopplerized) and a coverage
    mask: 1 where a cell was measured, 0 where it was gated, masked, mirrored
    from the other half of the sphere, or averaged from a gated partner.  It is
    one int8 value per channel, in the same group and order as dBA; with
    spectra it is per band, over the spectrum's own (PHI, THETA, frequency)
    dimensions, and without, a channel is measured only when all its bands are.

    speed_reference is written as given.  doppler_shift_removed, when given,
    is the flag assumed for any source sphere that does not carry one; a sphere
    that carries one must agree with it.  Sources with different flags are
    refused, because they would be combined into one database.

    The root text attribute azimuth_reference ('track' or 'heading', see
    :data:`AZIMUTH_REFERENCES`) is always written: what the flight spheres'
    azimuth is measured from, the ground track or the airframe's heading, for
    NICEOPS to orient its sphere by.  It is the value every source sphere in
    the directory carries (``noise_abatement_2017.build_sphere(azimuth_reference=)``); azimuth_reference,
    when given, is assumed for spheres that carry none and must agree with those
    that do, and spheres filed in different frames are refused.  Without either,
    'track', as every sphere was built before the attribute existed.  A hover is
    oriented by its heading whatever this says, on both sides (``hover_source``
    is not consulted).

    The database is level (``docs/THEORY.md`` section 3.2), and its root attribute
    sphere_frame says so.  Each source sphere is read in the frame it records
    (:data:`geometry.SPHERE_FRAMES`): a 'flight_path' sphere, AAM's, is pitched to level
    by its FLIGHT_PATH_ANGLE (:func:`gridding.repitch_art_levels`); cells beyond its own
    hemisphere, filled from its equator, or read mostly from unmeasured cells have
    coverage 0.
    sphere_frame, when given, is assumed for spheres that record none and must agree
    with those that do; without it they are taken as level, with a warning.  Repeat
    runs merged into one condition must share a frame.

    Every condition group also carries the metadata of the measured run its
    levels came from (see :func:`group_run_metadata` and
    :data:`GROUP_RUN_METADATA_NUMBERS`) as group attributes, NaN where the
    sphere does not say: gross_weight (N), air_density (kg m-3),
    wind_along_track and wind_cross_track (in the units the build declared,
    the group's wind_units), advance_ratio_air and thrust_coefficient_run,
    each with its units in ``<name>_units``, and the text attributes
    wind_source, wind_units, wind_reference_direction,
    air_density_source, source_sphere (the sphere file's name) and
    condition_origin ('measured', 'extended_flight_path_angle' or
    'synthesized_hover').  NICEOPS reads none of them; they are recorded so
    that air-referenced or per-run C_T labels can be tested without a rebuild.

    The root also records the build itself: the text attributes of
    :data:`provenance.PROVENANCE_ATTRIBUTES` (code version, library versions,
    time, the processing arguments of this call as canonical JSON with its
    SHA-256, and the directory, database, hover source, ``spheres`` and
    ``exclude`` by file name in ``panam_inputs``), and, as float attributes, ``load_factors``
    (omitted with ``load_factors=None``; ``fixed_load_factor`` says so) and
    ``extended_flight_path_angles_deg`` (omitted when not given).

    NICEOPS takes one sphere per flight condition, so two spheres with the
    same SPEED and FLIGHT_PATH_ANGLE (equal to 1e-9 kt and deg) are refused.
    merge_repeats=True energy-averages them instead, with
    :func:`sphere_statistics.average_spheres` (a -999 cell is no data, left out
    of that cell's average), into one condition at the runs' mean speed; with
    it, merge_speed_tolerance (kt, default 0) also merges spheres at one flight
    path angle whose speeds lie within it of the slowest of the group.  A merged
    condition's groups carry the runs' mean numeric run metadata (NaN winds
    if the runs' wind units or reference directions differ), the first run's
    text metadata but every run's wind_source, and every run's file name in
    source_sphere, one per line; ``panam_inputs`` lists each merged set as ``merged_spheres``.  The
    hover is synthesized from a merged condition's averaged levels when that
    condition is the slowest near-level one.
    """
    # File identity apart from the hashed settings; paths by file name only (a relative
    # path's directories name this machine's layout too).  Whether a hover source was
    # given changes the build, so that much stays in the settings.
    settings = {key: value for key, value in locals().items()
                if key not in ('directory_name', 'database_filename', 'hover_source',
                               'spheres', 'exclude')}
    settings['hover_source_given'] = hover_source is not None
    inputs = dict(directory_name=_provenance.path_name(directory_name),
                  database_filename=_provenance.path_name(database_filename),
                  hover_source=None if hover_source is None else _provenance.path_name(hover_source))
    # Which spheres a build uses is its input, not a processing setting.
    if spheres is not None and not isinstance(spheres, (str, os.PathLike)):
        inputs['spheres'] = sorted(_provenance.path_name(sphere) for sphere in spheres)
    if exclude is not None:
        inputs['exclude'] = sorted(_provenance.path_name(pattern) for pattern in (
            [exclude] if isinstance(exclude, (str, os.PathLike)) else exclude))
    # TODO add reinterpolation flag

    if speed_reference not in SPEED_REFERENCES:
        raise ValueError('speed_reference must be one of {}, not {!r}'.format(
            SPEED_REFERENCES, speed_reference))

    # None means "write the LF=1 reference only, let the reader scale it" --
    # see the docstring.  Every load-factor loop below iterates this instead
    # of the load_factors parameter directly, so add_sphere_group need not
    # change: at load_factor=1.0 it already writes the unscaled level and
    # thrust_coefficient == weight_coefficient, which is exactly the reference.
    fixed_load_factor = load_factors is None
    load_factors_to_write = (1.0,) if fixed_load_factor else load_factors

    # Get vehicle attributes
    (main_rotor_radius, main_rotor_area, main_rotor_tip_speed,
     _, _, _, _, _, _, weight_coefficient, _, _, _, _) = read_vehicle_data(directory_name)

    def run_metadata(filenames, origin='measured'):
        return _merged_run_metadata([group_run_metadata(filename, main_rotor_area,
                                                        main_rotor_tip_speed, origin)
                                     for filename in filenames])

    def condition_levels(filenames, frames=None):
        frames = sphere_frames if frames is None else frames
        names = {frames[f] for f in filenames}
        if len(names) > 1:
            raise ValueError('{} are filed in different frames ({}); merge spheres of one frame only'.format(
                ' and '.join(os.path.basename(f) for f in filenames), ', '.join(sorted(names))))
        return _extract_condition_SPL(filenames, infreqs, distance, atmosphere,
                                      absorption_band_method, merge_speed_tolerance, names.pop())

    # Sorted, so group numbering and hover-sphere ties do not depend on the
    # order the filesystem happens to list the directory in.
    file_list, excluded = _select_sphere_files(directory_name, spheres, exclude)
    if not file_list:
        raise ValueError('No sphere files found in ' + str(directory_name))
    if not merge_speed_tolerance >= 0.0:
        raise ValueError('merge_speed_tolerance must be at least 0 kt, not {!r}'.format(
            merge_speed_tolerance))
    if merge_speed_tolerance != 0.0 and not merge_repeats:
        raise ValueError('merge_speed_tolerance applies to merge_repeats=True only')
    # One entry per condition to write: a tuple of the sphere files averaged into it.
    condition_files = _group_repeat_conditions(
        file_list, merge_speed_tolerance if merge_repeats else 0.0)
    repeats = [files for files in condition_files if len(files) > 1]
    if repeats and not merge_repeats:
        speed, angle = _sphere_condition(repeats[0][0])
        raise ValueError('{} have the same flight condition ({:.4g} kt, {:.4g} deg); NICEOPS takes '
                         'one sphere per condition, so merge or drop one (merge_repeats=True, '
                         'or --merge-repeats, energy-averages them)'.format(
                             ' and '.join(os.path.basename(f) for f in repeats[0]), speed, angle))
    if repeats:
        inputs['merged_spheres'] = [[os.path.basename(f) for f in files] for files in repeats]
    # Settled before the output exists, so a refused mix of sources leaves no
    # half-written database behind.
    doppler = _resolve_doppler_shift_removed(file_list, doppler_shift_removed)
    azimuth = _resolve_azimuth_reference(file_list, azimuth_reference)
    sphere_frames = _resolve_sphere_frames(file_list, sphere_frame)

    # Set up the database.  Written to a temporary file and moved into place
    # only when complete (_atomic_output); the with block closes it explicitly,
    # since a failed flush on a close left to the garbage collector is swallowed.
    with _atomic_output(database_filename) as temporary, Dataset(temporary, 'w') as ncdatabase:
        _write_database_root(ncdatabase, fixed_load_factor=fixed_load_factor,
                             speed_reference=speed_reference, atmosphere=atmosphere,
                             main_rotor_radius=main_rotor_radius,
                             main_rotor_tip_speed=main_rotor_tip_speed,
                             vehicle_weight_newtons=read_vehicle_weight_newtons(directory_name),
                             azimuth_reference=azimuth,
                             # A fixed panam_tool value (docs/file_formats.md): it kept its name when the
                             # function moved here.
                             provenance=_provenance.build_provenance(
                                 'flight_acoustics.build_empirical_database', settings,
                                 inputs=inputs))
        if not fixed_load_factor:
            ncdatabase.load_factors = np.asarray(load_factors, dtype=np.float64).ravel()
        if extended_flight_path_angles is not None:
            ncdatabase.extended_flight_path_angles_deg = np.asarray(
                extended_flight_path_angles, dtype=np.float64).ravel()
        if hover_correction is not None:
            ncdatabase.hover_correction = str(getattr(hover_correction, 'description', 'applied'))
        if hover_source is not None:
            ncdatabase.hover_source = os.path.basename(str(hover_source))
        ncdatabase.source_spheres = '\n'.join(os.path.basename(f) for f in file_list)
        if exclude is not None:
            ncdatabase.excluded_spheres = '\n'.join(os.path.basename(f) for f in excluded)

        source_conditions = []
        level_conditions = []
        # Each condition's levels, kept for the hover synthesis below.
        condition_inputs = {}
        # Every condition to write, collected rather than written immediately, so
        # that whether phi/theta/frequency can be deduplicated (see
        # _grid_and_frequency_are_shared) is known before the first group is
        # written -- not discovered partway through and left to redo.
        pending_groups = []

        def add(phi, theta, radius, spla, eaa, speed, flight_path_angle, load_factor, frequency,
                amplitude, coverage, metadata):
            pending_groups.append(_SphereCondition(
                'sphere' + str(len(pending_groups)), phi, theta, radius, spla, eaa,
                speed, flight_path_angle, load_factor,
                frequency if store_spectrum else None,
                amplitude if store_spectrum else None, coverage, metadata))

        for filenames in condition_files:
            # Load the sphere data (the energy average of repeat runs, when merged)
            (_, _, phi_list, theta_list, radius, _, SPLA, EAA, speed, flight_path_angle,
             frequency, amplitude, in_hemisphere) = condition_levels(filenames)
            # Which bins carry a measurement.  Taken before the clamp, and before the
            # completion fills the upper surface, so a gated bin is recorded as such;
            # a flight-path sphere's cells read from beyond its hemisphere are not.
            measured = in_hemisphere & _measured_bins(amplitude, EAA)
            condition_inputs[filenames] = (phi_list, theta_list, radius, frequency, amplitude, measured)
            if clamp_empty_directions:
                SPLA, EAA = _finite_sphere_levels(SPLA, EAA)
            # Remembered for picking the hover source (_hover_source_index).
            source_conditions.append((speed, flight_path_angle))
            # Complete the sphere onto its upper surface
            phi_full, theta_full, SPLA_full, EAA_full, amplitude_full, coverage_full = (
                _complete_sphere_with_coverage(phi_list, theta_list, SPLA, EAA, amplitude, measured))
            # Remember near-level conditions; they are the ones re-emitted at the
            # extended flight path angles below.
            if extended_flight_path_angles is not None and np.abs(flight_path_angle) < level_flight_tolerance:
                level_conditions.append((phi_full, theta_full, radius, SPLA_full, EAA_full,
                                         speed, frequency, amplitude_full, coverage_full, filenames))
            # Augment load factor data
            measured_metadata = run_metadata(filenames)
            for load_factor in load_factors_to_write:
                add(phi_full, theta_full, radius, SPLA_full, EAA_full, speed, flight_path_angle,
                    load_factor, frequency, amplitude_full, coverage_full, measured_metadata)

        min_speed_files = condition_files[_hover_source_index(source_conditions, level_flight_tolerance)]

        if hover_source is not None:
            # Its DOPPLER_SHIFT_REMOVED does not enter the database's flag: a hover
            # has no Doppler shift, so its emitted and received frequencies are one
            # and the group is received-frame like the flight spheres around it.
            min_speed_files = (os.path.abspath(os.path.expanduser(hover_source)),)
            (_, _, phi_list, theta_list, radius, _, _, eaa_source, _, _,
             frequency, amplitude, in_hemisphere) = condition_levels(
                min_speed_files, _resolve_sphere_frames(min_speed_files, sphere_frame))
            measured = in_hemisphere & _measured_bins(amplitude, eaa_source)
        else:
            phi_list, theta_list, radius, frequency, amplitude, measured = condition_inputs[min_speed_files]
        # Now, adapt the lowest speed sphere to a hover sphere by averaging from fore to aft
        phi_full, theta_full, SPLA_full, EAA_full, amplitude_full, coverage_full = _synthesized_hover_sphere(
            phi_list, theta_list, amplitude, measured, frequency,
            distance, atmosphere, hover_correction=hover_correction,
            clamp_empty_directions=clamp_empty_directions,
            absorption_band_method=absorption_band_method)
        # Set hover conditions.  Hover is a near-level condition, so it is written
        # at the extended angles as well as level flight.
        speed = 0
        flight_path_angles = _hover_flight_path_angles(extended_flight_path_angles)
        # Augment load factor data
        hover_metadata = run_metadata(min_speed_files, 'synthesized_hover')
        for load_factor in load_factors_to_write:
            for flight_path_angle in flight_path_angles:
                add(phi_full, theta_full, radius, SPLA_full, EAA_full, speed, flight_path_angle,
                    load_factor, frequency, amplitude_full, coverage_full, hover_metadata)

        # Widen the flight-path-angle envelope: re-emit each near-level condition at
        # the extended angles, keeping its own airspeed and directivity.  Hover is
        # already covered above, so skip any condition at zero airspeed.
        if extended_flight_path_angles is not None:
            for (phi_full, theta_full, radius, SPLA_full, EAA_full,
                 level_speed, frequency, amplitude_full, coverage_full, source_files) in level_conditions:
                if level_speed == 0:
                    continue
                extended_metadata = run_metadata(source_files, 'extended_flight_path_angle')
                for extended_angle in extended_flight_path_angles:
                    for load_factor in load_factors_to_write:
                        add(phi_full, theta_full, radius, SPLA_full, EAA_full, level_speed,
                            extended_angle, load_factor, frequency, amplitude_full, coverage_full,
                            extended_metadata)

        # phi/theta (and frequency, for a spectral database) are usually the same
        # on every condition -- panam completes every sphere onto a common grid
        # and reduces every condition to the same frequency bands -- but that is
        # checked here, not assumed: a mixed microphone layout or a per-file
        # frequency resolution would make it false, and writing the shared form
        # anyway would silently lose whatever a later condition's grid disagreed
        # on. See the root shared_grid_and_frequency flag this sets.
        shared_grid_and_frequency = _write_shared_grid_and_frequency(ncdatabase, pending_groups)

        for condition in pending_groups:
            add_sphere_group(ncdatabase, condition.name, condition.phi, condition.theta,
                             condition.radius, condition.SPLA, condition.EAA,
                             condition.speed, condition.flight_path_angle, condition.load_factor,
                             main_rotor_radius, main_rotor_tip_speed, weight_coefficient,
                             condition.frequency, condition.amplitude,
                             write_grid_and_frequency=not shared_grid_and_frequency,
                             doppler_shift_removed=doppler, coverage=condition.coverage,
                             run_metadata=condition.run_metadata)


# ---- Writing a database ----

def _write_database_root(ncdatabase, *, fixed_load_factor, speed_reference, atmosphere,
                         main_rotor_radius, main_rotor_tip_speed, vehicle_weight_newtons,
                         azimuth_reference, provenance=None):
    """Write the root flags, build atmosphere and vehicle data every database carries.

    Shared by :func:`build_empirical_database` and :func:`norah2.build_database_from_norah2`.
    The vehicle values are skipped when None.  azimuth_reference (one of
    :data:`AZIMUTH_REFERENCES`) is required, so that no database is written
    without saying which frame its spheres' azimuth is measured in.
    provenance, a stamp from :func:`provenance.build_provenance`, is written
    as the root ``panam_*`` text attributes; None writes none.
    """
    if speed_reference not in SPEED_REFERENCES:
        raise ValueError('speed_reference must be one of {}, not {!r}'.format(
            SPEED_REFERENCES, speed_reference))
    if azimuth_reference not in AZIMUTH_REFERENCES:
        raise ValueError('azimuth_reference must be one of {}, not {!r}'.format(
            AZIMUTH_REFERENCES, azimuth_reference))
    # NICEOPS requires this flag; it reads it unconditionally at load time.
    ncdatabase.createVariable("same_grid", 'b')
    ncdatabase['same_grid'][:] = True
    # Optional: absent (old files) or false means every stored condition is a
    # real, independent sample and NICEOPS must not scale between them.  True
    # means every condition was synthesized from one LF=1 reference by a
    # uniform dB offset, so the reader may reproduce load factors this file
    # never stored by applying that same offset analytically.
    ncdatabase.createVariable("fixed_load_factor", 'b')
    ncdatabase['fixed_load_factor'][:] = fixed_load_factor
    # Format version, so consumers can tell a database built by this code from
    # the older ones whose hover spheres have broken directivity.  Bump this
    # whenever the on-disk meaning of the sphere data changes.
    ncdatabase.createVariable("database_version", 'i4')
    ncdatabase['database_version'][:] = DATABASE_FORMAT_VERSION
    # Optional root attributes recording what the database was built against.
    # The EAA is computed with ``atmosphere`` (extract_SPL, spla_and_eaa_from_spectrum),
    # so these are that atmosphere: temperature in K, pressure in kPa (the ISO 9613-1
    # reference unit, as Atmosphere is written), relative humidity in percent.
    ncdatabase.speed_reference = speed_reference
    ncdatabase.azimuth_reference = azimuth_reference
    # Every database is level: AAM's flight-path spheres are pitched to it as they are read.
    ncdatabase.sphere_frame = 'level'
    ncdatabase.build_temperature_K = float(atmosphere.temperature)
    ncdatabase.build_pressure_kPa = float(atmosphere.pressure)
    ncdatabase.build_relative_humidity_percent = float(atmosphere.relative_humidity)

    # Root vehicle data, as carried by every shipped database (S-76D_M3.nod,
    # AW139_M1.nod, Be407_spectral.nod).  NICEOPS reads them to redimensionalize
    # the sphere conditions and, for --export_aam, in preference to anything it
    # would otherwise infer, so leaving them out quietly changes what it does.
    # main_rotor_radius_meters restates each group's rotor_scale; tip speed and
    # weight appear nowhere else in the file.
    if main_rotor_radius is not None:
        ncdatabase.createVariable("main_rotor_radius_meters", 'f8')
        ncdatabase['main_rotor_radius_meters'][:] = float(main_rotor_radius)
    if main_rotor_tip_speed is not None:
        ncdatabase.createVariable("main_rotor_tip_speed_meters_per_sec", 'f8')
        ncdatabase['main_rotor_tip_speed_meters_per_sec'][:] = float(main_rotor_tip_speed)
    if vehicle_weight_newtons is not None:
        ncdatabase.createVariable("vehicle_weight_newtons", 'f8')
        ncdatabase['vehicle_weight_newtons'][:] = float(vehicle_weight_newtons)
    _provenance.write_provenance(ncdatabase, provenance)


def _write_shared_grid_and_frequency(ncdatabase, pending_groups):
    """Write the root ``shared_grid_and_frequency`` flag, and the shared arrays when it is set.

    Returns the flag: whether the groups may leave out their own phi, theta and
    frequency.  See :func:`_grid_and_frequency_are_shared`.
    """
    shared_grid_and_frequency = _grid_and_frequency_are_shared(pending_groups)
    ncdatabase.createVariable("shared_grid_and_frequency", 'b')
    ncdatabase['shared_grid_and_frequency'][:] = shared_grid_and_frequency
    if shared_grid_and_frequency:
        first = pending_groups[0]
        ncdatabase.createDimension("channels", first.phi.size)
        ncdatabase.createVariable("phi", 'f8', ("channels",))
        ncdatabase['phi'][:] = first.phi.flatten()
        ncdatabase.createVariable("theta", 'f8', ("channels",))
        ncdatabase['theta'][:] = first.theta.flatten()
        if first.frequency is not None:
            ncdatabase.createDimension("frequency", np.size(first.frequency))
            ncdatabase.createVariable("frequency", 'f8', ("frequency",))
            ncdatabase['frequency'][:] = np.asarray(first.frequency).flatten()
    return shared_grid_and_frequency


def _grid_and_frequency_are_shared(pending_groups):
    """Whether every condition in pending_groups has the same phi, theta and
    frequency arrays, so build_empirical_database can write them once at the
    root instead of once per condition.  See its shared_grid_and_frequency
    flag.
    """
    if not pending_groups:
        return False
    first = pending_groups[0]
    for condition in pending_groups[1:]:
        phi, phi0 = condition.phi, first.phi
        theta, theta0 = condition.theta, first.theta
        frequency, frequency0 = condition.frequency, first.frequency
        if phi.shape != phi0.shape or not np.array_equal(phi, phi0):
            return False
        if theta.shape != theta0.shape or not np.array_equal(theta, theta0):
            return False
        if (frequency is None) != (frequency0 is None):
            return False
        if frequency is not None and not np.array_equal(frequency, frequency0):
            return False
    return True


def add_sphere_group(ncdatabase, groupname, phi, theta, radius, SPLA, EAA, speed, flight_path_angle, load_factor,
                     main_rotor_radius, main_rotor_tip_speed, weight_coefficient, frequency=None, amplitude=None,
                     write_grid_and_frequency=True, doppler_shift_removed=None, coverage=None,
                     run_metadata=None):
    """Write one condition group.

    run_metadata, when given, is a dict as :func:`group_run_metadata` returns: each
    of :data:`GROUP_RUN_METADATA_NUMBERS` is written as an f8 attribute of the group
    (NaN where unknown) with its units in the text attribute ``<name>_units``, and
    each of :data:`GROUP_RUN_METADATA_TEXT` as a text attribute.  Attributes, not
    variables, for NICEOPS's load time (see :data:`GROUP_RUN_METADATA_NUMBERS`).

    doppler_shift_removed, when given, is written as the group's scalar int
    DOPPLER_SHIFT_REMOVED (0 received-frame, 1 de-Dopplerized).  coverage, when
    given, is the (n_phi, n_theta, n_bands) bool measured mask of the completed
    sphere.  It is written as int8 ``coverage`` over the amplitude's dimensions
    ("PHI", "THETA", "frequency") when the group has spectra, and otherwise over
    ("channels",), one value per channel in dBA's order, where a channel is
    measured only if every band of it is.  Both are optional so that a caller
    building a group by hand is unaffected.
    """
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
    this_group.variables['advance_ratio'][:] = KNOT_MPS * speed / main_rotor_tip_speed
    this_group.createVariable("flight_path_angle", 'f8', ("condition",))
    this_group.variables['flight_path_angle'][:] = flight_path_angle
    this_group.createVariable("thrust_coefficient", 'f8', ("condition",))
    this_group.variables['thrust_coefficient'][:] = load_factor * weight_coefficient
    # Define acoustic data.  "channels" is defined locally in every group
    # regardless of write_grid_and_frequency -- netCDF4/HDF5 charges real
    # space to *share* a dimension across many groups (each group's variable
    # needs its own dimension-scale attachment back to the shared one), enough
    # that on a 1432-group database it cost more than the phi/theta arrays it
    # was meant to save. A same-named local dimension in each group costs
    # nothing extra over what every database already paid before this existed;
    # only the phi/theta *values* -- the redundant part -- are skipped.
    this_group.createDimension("channels", phi.size)
    if write_grid_and_frequency:
        this_group.createVariable("phi", 'f8', ("channels",))
        this_group.variables['phi'][:] = phi.flatten()
        this_group.createVariable("theta", 'f8', ("channels",))
        this_group.variables['theta'][:] = theta.flatten()
    this_group.createVariable("dBA", 'f8', ("channels",))
    # Thrust scaling: a load factor of n raises the level by 20*log10(n).
    # A load factor of 0 is a special case -- it is not a physical condition but
    # a floor entry that extends the thrust-coefficient range down to CT = 0, so
    # trajectories below 1 g interpolate instead of clamping.  It carries the
    # 1 g levels unscaled; evaluating 20*log10(0) would write -inf into dBA, and
    # a zero barycentric weight against -inf yields NaN, poisoning every
    # interpolation that touches it.
    level_offset = 20 * np.log10(load_factor) if load_factor > 0 else 0.0
    this_group.variables['dBA'][:] = SPLA.flatten() + level_offset
    this_group.createVariable("EAA", 'f8', ("channels",))
    this_group.variables['EAA'][:] = EAA.flatten()
    has_spectrum = frequency is not None and amplitude is not None
    # Store the source spectrum alongside the reduced levels, so a database can
    # be re-reduced (different weighting, different propagation distance)
    # without going back to the original sphere files.
    if has_spectrum:
        # Same reasoning as "channels" above: local dimensions in every group,
        # only the frequency values themselves are conditionally skipped.
        this_group.createDimension("frequency", np.size(frequency))
        this_group.createDimension("PHI", np.shape(amplitude)[0])
        this_group.createDimension("THETA", np.shape(amplitude)[1])
        if write_grid_and_frequency:
            this_group.createVariable("frequency", 'f8', ("frequency",))
            this_group.variables['frequency'][:] = np.asarray(frequency).flatten()
        this_group.createVariable("amplitude", 'f8', ("PHI", "THETA", "frequency"))
        this_group.variables['amplitude'][:] = amplitude

    if doppler_shift_removed is not None:
        if doppler_shift_removed not in (0, 1):
            raise ValueError('doppler_shift_removed must be 0 or 1, not {!r}'.format(doppler_shift_removed))
        this_group.createVariable("DOPPLER_SHIFT_REMOVED", 'i4')
        this_group.variables['DOPPLER_SHIFT_REMOVED'][:] = int(doppler_shift_removed)

    if coverage is not None:
        measured = np.asarray(coverage, dtype=bool)
        if measured.ndim != 3 or measured.shape[:2] != phi.shape:
            raise ValueError('coverage must be (n_phi, n_theta, n_bands) to match phi, not {}'.format(
                measured.shape))
        if has_spectrum:
            # The same dimensions as the amplitude it masks: gridded (PHI, THETA, frequency).
            this_group.createVariable("coverage", 'i1', ("PHI", "THETA", "frequency"))
            this_group.variables['coverage'][:] = measured.astype(np.int8)
        else:
            # One value per channel, in the same order as dBA and EAA.  No spectrum to
            # keep the bands in: a cell is measured only if all its bands are.
            this_group.createVariable("coverage", 'i1', ("channels",))
            this_group.variables['coverage'][:] = measured.all(axis=2).reshape(-1).astype(np.int8)
        this_group.variables['coverage'].description = (
            '1 = measured cell (every band measured, when there is no spectrum); 0 = unmeasured: '
            'gated, masked, mirrored from the other half of the sphere, or averaged from a gated partner')

    if run_metadata is not None:
        missing = ({name for name, _ in GROUP_RUN_METADATA_NUMBERS} | set(GROUP_RUN_METADATA_TEXT)) \
            - set(run_metadata)
        if missing:
            raise ValueError('run_metadata lacks {}'.format(sorted(missing)))
        for name, units in GROUP_RUN_METADATA_NUMBERS:
            this_group.setncattr(name, np.float64(run_metadata[name]))
            this_group.setncattr(name + '_units', str(run_metadata['wind_units']) if units is None else units)
        for name in GROUP_RUN_METADATA_TEXT:
            this_group.setncattr(name, str(run_metadata[name]))
