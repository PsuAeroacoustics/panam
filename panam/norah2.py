"""NORAH2 (HELENA, ECAC Doc 32) hemispheres: reading, writing, and a NICEOPS database from them.

:func:`write_norah2_hemisphere` and :func:`write_norah2_levels` write ``.hem``
files, :func:`load_norah2_hemisphere` reads them, and
:func:`build_database_from_norah2` imports them into a NICEOPS database
(``docs/THEORY.md`` section 8 and ``docs/norah2_import.md``).
"""

import os
import warnings
from typing import Optional

from netCDF4 import Dataset
import numpy as np

from panam import provenance as _provenance
from panam import unit_conversion
from panam.acoustics.atmosphere import Atmosphere
from panam.geometry import norah2umapr
from panam.absorption import metadata_atmosphere, _band_attenuation_db, _check_band_method
from panam.gridding import _sample_hemisphere_levels
from panam.sphere_io import KNOT_MPS, mask_missing_levels, _atomic_output
from panam.database import (add_sphere_group, spla_and_eaa_from_spectrum, _complete_sphere_with_coverage,
                            _finite_sphere_levels, _hover_flight_path_angles, _hover_source_index,
                            _measured_bins, _SphereCondition, _synthesized_hover_sphere, _write_database_root,
                            _write_shared_grid_and_frequency)


# NORAH2 hemispheres (EASA's rotorcraft noise model) are ASCII "generic acoustic
# data" (.hem) files inherited from HELENA.  The layout is given in EASA NORAH2
# D1.5d Appendix A and D2.3 section 5.5; the data block, which neither shows,
# follows the files shipped with NORAH2 V2.0.74: one block per PHIOBSAC, one
# row per THETAOBSAC, the row starting with its theta and then one level per
# band.

#: Nominal one-third octave band centers of every NORAH2 hemisphere, 10 Hz to 10 kHz.
NORAH2_BAND_CENTERS_HZ = np.array([
    10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0,
    100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0,
    1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0,
    10000.0])


#: Polar angle: 0 at the nose, 90 straight down (at phi = 0), 180 at the tail.
NORAH2_THETA_DEG = np.arange(0.0, 180.0 + 1e-9, 10.0)


#: Azimuth: negative to port, positive to starboard, 0 in the vertical plane.
NORAH2_PHI_DEG = np.arange(-90.0, 90.0 + 1e-9, 10.0)


#: Every NORAH2 hemisphere is defined at 60 m, with absorption over those 60 m
#: included for the ICAO reference atmosphere (FREEFIELD = 2).  NORAH2 takes
#: that absorption back out using TAMB/RELHUM/PAMB before propagating, so the
#: header and the levels must agree.
NORAH2_REFERENCE_DISTANCE_M = 60.0


NORAH2_REFERENCE_TEMPERATURE_K = 298.15


NORAH2_REFERENCE_RELATIVE_HUMIDITY = 70.0


NORAH2_REFERENCE_PRESSURE_PA = 101325.0


NORAH2_MISSING_LEVEL = -999.0


#: Third table of every shipped triangulation file: level corrections NORAH2
#: applies for operations it has no hemisphere of their own for.
NORAH2_DEFAULT_CORRECTIONS = ((8.0, 'Outoffgroundhover'),
                              (-10.0, 'Reducedrpmidle'),
                              (-2.0, 'Fullrpmidle'))


def _norah2_band_index(frequency_hz):
    """Index into ``frequency_hz`` of each NORAH2 band, -1 where it has none.

    Matches within a twelfth of an octave, so exact (base-10) and nominal band
    centers both find their band.
    """
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    index = np.full(NORAH2_BAND_CENTERS_HZ.size, -1, dtype=int)
    if frequency_hz.size == 0:
        return index
    for i, nominal in enumerate(NORAH2_BAND_CENTERS_HZ):
        distance = np.abs(np.log2(frequency_hz / nominal))
        j = int(np.argmin(distance))
        if distance[j] < 1.0 / 12.0:
            index[i] = j
    return index


def write_norah2_hemisphere(
        filename,
        hemisphere,
        *,
        speed_knots: float,
        flight_path_angle_deg: float,
        title: Optional[str] = None,
        test_point: str = '',
        measurement_atmosphere: Optional[Atmosphere] = None,
        rotor_rpm_percent: float = 100.0,
        total_wind_knots: float = 0.0,
        cross_wind_knots: float = 0.0,
        pitch_deg: Optional[float] = None,
        roll_deg: Optional[float] = None,
        head_wind_knots: Optional[float] = None,
        minimum_level_db: float = -100.0,
        overwrite: bool = True,
):
    """Write a depropagated acoustic hemisphere as a NORAH2 ``.hem`` file.

    The levels are converted to NORAH2's convention on the way out:

    - moved by spherical spreading from the hemisphere's ``r_ref`` to 60 m;
    - the absorption depropagation leaves in over ``r_ref`` (in the
      measurement atmosphere, by the band method in its metadata's
      ``absorption_band_method``) taken out, and absorption over 60 m in the
      ICAO reference atmosphere put in, as ``FREEFIELD = 2`` declares;
    - resampled in linear power onto the NORAH2 10 degree grid and its 31
      nominal one-third octave bands (10 Hz to 10 kHz).  A band the
      hemisphere does not have, a direction it did not cover, and a level
      below ``minimum_level_db`` are all written as -999, NORAH2's no-value.

    NORAH2 finds a hemisphere through its aircraft's triangulation file, not
    the file name; see :func:`write_norah2_triangulation`.  The shipped files
    are named ``[type]_[procedure]_[IAS]kts_[gamma]deg.hem``.

    Args:
        filename: Path to the output ``.hem`` file.
        hemisphere: Output dict from :func:`depropagate_hemisphere` with
            ``third_octave=True``.
        speed_knots: ``ACSPEED`` (knots).  The format labels it indicated airspeed,
            but NORAH2 looks hemispheres up by ground speed, which is what panam's
            tracks provide and what :func:`noise_abatement_2017.build_sphere` passes.
        flight_path_angle_deg: ``GAMM`` (deg), negative in descent.
        title: First line of the file. Default: the file name without extension.
        test_point: Written after the title, as the shipped files carry their
            test point numbers (e.g. 'TP01+02').
        measurement_atmosphere: The atmosphere the hemisphere was depropagated
            in. Default: the one recorded in its metadata. Written as Tm/RHm/Pm.
        rotor_rpm_percent: ``RmOmega``, main rotor speed (% of nominal).
        total_wind_knots, cross_wind_knots: ``TW``, ``CW``.
        pitch_deg, roll_deg, head_wind_knots: ``PITCH``, ``ROLL``, ``HW``. Some
            shipped files carry these and some do not; they are written only
            when given.
        minimum_level_db: Levels (at 60 m) below this are written as -999; see
            :func:`write_aam_hemisphere_netcdf`. Pass -inf to keep every finite level.
        overwrite: If False, raises when filename exists.

    Returns:
        levels_db: (phi, theta, band) levels as written, -inf where -999.
    """
    if not overwrite and os.path.exists(filename):
        raise FileExistsError(f'Output file already exists: {filename}')
    if not isinstance(hemisphere, dict):
        raise TypeError('hemisphere must be a dict returned by depropagate_hemisphere')
    if 'third_octave' not in hemisphere:
        raise ValueError('NORAH2 hemispheres are one-third octave band levels; run '
                         'depropagate_hemisphere with third_octave=True')

    meta = hemisphere.get('metadata', {}) if isinstance(hemisphere.get('metadata', {}), dict) else {}
    if 'r_ref' not in meta or 'length_units' not in meta:
        raise ValueError('hemisphere metadata does not include r_ref/length_units')
    r_ref_m = float(unit_conversion.len_conv(float(meta['r_ref']),
                                             from_units=str(meta['length_units']), to_units='m'))

    if measurement_atmosphere is None:
        measurement_atmosphere = metadata_atmosphere(meta)
        if measurement_atmosphere is None:
            raise ValueError('measurement_atmosphere not given and the hemisphere metadata '
                             'does not record one')

    PH, TH = np.meshgrid(NORAH2_PHI_DEG, NORAH2_THETA_DEG, indexing='ij')
    azi_rad, elv_rad = norah2umapr(np.deg2rad(PH), np.deg2rad(TH))
    # Resample before the unit conversion; minimum_level_db applies at 60 m.
    frequency_hz, sampled_db = _sample_hemisphere_levels(
        hemisphere, 'third_octave', np.degrees(azi_rad), np.degrees(elv_rad), -np.inf)
    return write_norah2_levels(
        filename, frequency_hz, sampled_db, r_ref_m=r_ref_m,
        measurement_atmosphere=measurement_atmosphere,
        apply_absorption_deprop=meta.get('apply_absorption_deprop', True),
        absorption_band_method=str(meta.get('absorption_band_method', 'center')),
        speed_knots=speed_knots, flight_path_angle_deg=flight_path_angle_deg, title=title,
        test_point=test_point, rotor_rpm_percent=rotor_rpm_percent,
        total_wind_knots=total_wind_knots, cross_wind_knots=cross_wind_knots,
        pitch_deg=pitch_deg, roll_deg=roll_deg, head_wind_knots=head_wind_knots,
        minimum_level_db=minimum_level_db)


def write_norah2_levels(
        filename,
        frequency_hz,
        levels_db,
        *,
        r_ref_m: float,
        measurement_atmosphere: Atmosphere,
        apply_absorption_deprop: bool = True,
        absorption_band_method: str = 'center',
        speed_knots: float,
        flight_path_angle_deg: float,
        title: Optional[str] = None,
        test_point: str = '',
        rotor_rpm_percent: float = 100.0,
        total_wind_knots: float = 0.0,
        cross_wind_knots: float = 0.0,
        pitch_deg: Optional[float] = None,
        roll_deg: Optional[float] = None,
        head_wind_knots: Optional[float] = None,
        minimum_level_db: float = -100.0,
        overwrite: bool = True,
):
    """Write band levels already on the NORAH2 grid as a NORAH2 ``.hem`` file.

    The writer behind :func:`write_norah2_hemisphere`, for levels that are on the
    NORAH2 (phi, theta) grid already, such as an AAM sphere sampled there
    (``tools/sphere_to_norah2.py``).  It does the unit conversion that function
    describes: spreading from ``r_ref_m`` to 60 m, the measured absorption over
    ``r_ref_m`` out and the ICAO reference absorption over 60 m in, the match to
    NORAH2's 31 nominal bands, and -999 for every missing level.

    Args:
        filename: Path to the output ``.hem`` file.
        frequency_hz: (Nf,) band centers of ``levels_db`` (Hz).
        levels_db: (19, 19, Nf) unweighted band levels (dB re 20 uPa) at
            ``r_ref_m``, indexed (``NORAH2_PHI_DEG``, ``NORAH2_THETA_DEG``, band);
            -inf or NaN where missing.
        r_ref_m: Distance the levels are referred to (m).
        measurement_atmosphere: The atmosphere whose absorption over ``r_ref_m``
            the levels keep. Written as Tm/RHm/Pm.
        apply_absorption_deprop: False if the levels keep the absorption of each
            measured path instead; then only spreading is applied, with a warning.
        absorption_band_method: how the absorption over ``r_ref_m`` that the levels
            keep was formed (:func:`panam.acoustics.atmosphere.band_absorption_db`):
            'center' (default, the ISO 9613-1 coefficient at the band center),
            'arp5534' or 'integrated'.  That amount is taken out.
        Others: as :func:`write_norah2_hemisphere`.

    Returns:
        levels_db: (phi, theta, band) levels as written at 60 m (dB), -inf where -999.
    """
    if not overwrite and os.path.exists(filename):
        raise FileExistsError(f'Output file already exists: {filename}')
    sampled_db = np.asarray(levels_db, dtype=float)
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    expected_shape = (NORAH2_PHI_DEG.size, NORAH2_THETA_DEG.size, frequency_hz.size)
    if sampled_db.shape != expected_shape:
        raise ValueError(f'levels_db must be {expected_shape} (phi, theta, band), not {sampled_db.shape}')
    sampled_db = np.where(np.isnan(sampled_db), -np.inf, sampled_db)
    reference_atmosphere = Atmosphere(temperature=NORAH2_REFERENCE_TEMPERATURE_K,
                                      pressure=NORAH2_REFERENCE_PRESSURE_PA / 1000.0,
                                      relative_humidity=NORAH2_REFERENCE_RELATIVE_HUMIDITY)

    band_index = _norah2_band_index(frequency_hz)
    levels_db = np.full(expected_shape[:2] + (NORAH2_BAND_CENTERS_HZ.size,), -np.inf, dtype=float)
    present = band_index >= 0
    levels_db[:, :, present] = sampled_db[:, :, band_index[present]]

    offset_db = 20.0 * np.log10(r_ref_m / NORAH2_REFERENCE_DISTANCE_M)
    if apply_absorption_deprop:
        # Use the bands' own centers for the measured absorption, the nominal
        # ones for NORAH2's: those are what NORAH2 will remove it at.
        measured_centers = np.where(present, frequency_hz[band_index],
                                    NORAH2_BAND_CENTERS_HZ)
        # The sphere keeps the band attenuation over r_ref as its band method
        # formed it (see depropagate_hemisphere), so take out that amount.
        measured_db = _band_attenuation_db(measurement_atmosphere, measured_centers, r_ref_m,
                                           str(absorption_band_method))
        offset_db = (offset_db
                     + measured_db
                     - reference_atmosphere.attenuation_coefficient(NORAH2_BAND_CENTERS_HZ)
                     * NORAH2_REFERENCE_DISTANCE_M)
    else:
        warnings.warn('hemisphere was built without absorption depropagation, so it carries '
                      'the absorption of each measured path; written with spreading only, '
                      'which NORAH2 will read as over-absorbed at high frequency')
    levels_db = levels_db + offset_db
    if np.isfinite(minimum_level_db):
        levels_db[levels_db < float(minimum_level_db)] = -np.inf

    constants = [
        ('POLDIST', f'{NORAH2_REFERENCE_DISTANCE_M:g}', 'Distance at which hemisphere is defined'),
        ('FREEFIELD', '2', 'Atmospheric absorption included in hemisphere'),
        ('NOVALUE', f'{NORAH2_MISSING_LEVEL:g}', 'no value indicator'),
        ('TAMB', f'{NORAH2_REFERENCE_TEMPERATURE_K:g}', 'Ambient temperature, deg Kelvin'),
        ('RELHUM', f'{NORAH2_REFERENCE_RELATIVE_HUMIDITY:g}', 'Relative humidity, %'),
        ('PAMB', f'{NORAH2_REFERENCE_PRESSURE_PA:g}', 'Ambient pressure, Pa'),
        ('Tm', f'{measurement_atmosphere.temperature - 273.15:g}',
         'Measurement ambient temperature at 10m, deg Celsius'),
        ('RHm', f'{measurement_atmosphere.relative_humidity:g}',
         'Measurement relative humidity at 10m, %'),
        ('Pm', f'{measurement_atmosphere.pressure * 1000.0:g}', 'Measurement ambient pressure at 10m, Pa'),
        ('RmOmega', f'{rotor_rpm_percent:g}', 'RotorRPM, rpm'),
        ('ACSPEED', f'{speed_knots:g}', 'Indicated airspeed, kts'),
        ('GAMM', f'{flight_path_angle_deg:g}', 'Path angle, deg'),
    ]
    if pitch_deg is not None:
        constants.append(('PITCH', f'{pitch_deg:g}', 'Pitch, deg'))
    if roll_deg is not None:
        constants.append(('ROLL', f'{roll_deg:g}', 'Roll, deg'))
    constants.append(('TW', f'{total_wind_knots:g}', 'Total wind, kts'))
    constants.append(('CW', f'{cross_wind_knots:g}', 'Cross wind, kts'))
    if head_wind_knots is not None:
        constants.append(('HW', f'{head_wind_knots:g}', 'Head wind, kts'))

    if title is None:
        title = os.path.splitext(os.path.basename(filename))[0]

    def _row(values):
        return '\t' + '\t'.join(values) + '\t'

    lines = [f'{title}\t{test_point}\t', f'{len(constants)}\t! # Table constants ']
    lines += [f'{name}\t{value}\t! {comment}' for name, value, comment in constants]
    lines += ['2\t ',
              f'THETAOBSAC\t{NORAH2_THETA_DEG.size}\t0\t3\t0',
              _row(f'{v:g}' for v in NORAH2_THETA_DEG),
              f'PHIOBSAC\t{NORAH2_PHI_DEG.size}\t0\t3\t0',
              _row(f'{v:g}' for v in NORAH2_PHI_DEG),
              '0\t! NPARAD: Additional point dependent parameters',
              f'NFREQ\t{NORAH2_BAND_CENTERS_HZ.size} ',
              '\t' + '\t'.join(f'{f:6.1f}' for f in NORAH2_BAND_CENTERS_HZ)]
    to_write = np.where(np.isfinite(levels_db), levels_db, NORAH2_MISSING_LEVEL)
    for i, phi in enumerate(NORAH2_PHI_DEG):
        lines.append(f'PHIOBSAC= {phi:.6f}')
        for j, theta in enumerate(NORAH2_THETA_DEG):
            lines.append(f'{theta:g}\t' + '\t'.join(f'{v:.1f}' for v in to_write[i, j, :]))

    # CRLF, as every shipped file has: NORAH2 is a Windows program.  Written in full or
    # not at all (_atomic_output): a build resumed with skip_existing takes any .hem it
    # finds as complete.
    with _atomic_output(filename) as temporary:
        with open(temporary, 'w', encoding='ascii', newline='\r\n') as handle:
            handle.write('\n'.join(lines) + '\n')
    return levels_db


def _norah2_numbers(tokens, count, name):
    values = []
    for token in tokens:
        values.append(float(token))
        if len(values) == count:
            return np.array(values)
    raise ValueError(f'NORAH2 hemisphere: {name} ended after {len(values)} of {count} values')


def load_norah2_hemisphere(filename):
    """Read a NORAH2 ``.hem`` hemisphere.

    Reads the two-axis (THETAOBSAC x PHIOBSAC) files every flight condition in
    the NORAH2 database uses.  Whitespace is free-form, as it is across the
    shipped files, and the table constants are taken by name, since their
    number varies (14 or 17).

    Returns: dict with
        title: first line of the file
        constants: {name: value} of the table constants
        theta_deg, phi_deg, frequency_hz: the axes
        levels_db: (phi, theta, frequency) levels, dB at POLDIST; NOVALUE as -inf
    """
    with open(filename, 'r', encoding='latin-1') as handle:
        lines = [line.rstrip('\r\n') for line in handle]

    def content(line):
        return line.split('!', 1)[0].split()

    title = lines[0].strip()
    n_constants = int(content(lines[1])[0])
    constants = {}
    for line in lines[2:2 + n_constants]:
        name, value = content(line)[:2]
        constants[name] = float(value)
    position = 2 + n_constants

    n_axes = int(content(lines[position])[0])
    position += 1
    if n_axes != 2:
        raise ValueError(f'{filename}: {n_axes}-axis hemispheres (special operations) are not supported')

    def read_axis(position):
        name, count = content(lines[position])[:2]
        count = int(count)
        tokens = []
        position += 1
        while len(tokens) < count:
            tokens += content(lines[position])
            position += 1
        return name, _norah2_numbers(tokens, count, name), position

    inner_name, inner, position = read_axis(position)
    outer_name, outer, position = read_axis(position)
    if int(content(lines[position])[0]) != 0:
        raise ValueError(f'{filename}: point dependent parameters (NPARAD > 0) are not supported')
    _, frequency_hz, position = read_axis(position + 1)

    axes = {inner_name: inner, outer_name: outer}
    if set(axes) != {'THETAOBSAC', 'PHIOBSAC'}:
        raise ValueError(f'{filename}: unexpected axes {inner_name}, {outer_name}')

    body = lines[position:]
    block_starts = [i for i, line in enumerate(body) if line.strip().startswith(outer_name)]
    if len(block_starts) != outer.size:
        raise ValueError(f'{filename}: {len(block_starts)} {outer_name} blocks, expected {outer.size}')
    levels = np.empty((outer.size, inner.size, frequency_hz.size), dtype=float)
    for k, start in enumerate(block_starts):
        stop = block_starts[k + 1] if k + 1 < len(block_starts) else len(body)
        tokens = [t for line in body[start + 1:stop] for t in content(line)]
        table = _norah2_numbers(tokens, inner.size * (1 + frequency_hz.size), outer_name)
        table = table.reshape(inner.size, 1 + frequency_hz.size)
        levels[k] = table[:, 1:]

    missing = constants.get('NOVALUE', NORAH2_MISSING_LEVEL)
    levels[np.isclose(levels, missing)] = -np.inf
    if outer_name == 'THETAOBSAC':
        levels = levels.transpose(1, 0, 2)
    return dict(title=title, constants=constants,
                theta_deg=axes['THETAOBSAC'], phi_deg=axes['PHIOBSAC'],
                frequency_hz=frequency_hz, levels_db=levels)


def write_norah2_triangulation(filename, hemispheres, *, corrections=NORAH2_DEFAULT_CORRECTIONS,
                               overwrite=True):
    """Write the ``[type]_Triangulation.int`` that lets NORAH2 use a set of hemispheres.

    NORAH2 interpolates between an aircraft's hemispheres over (airspeed, flight
    path angle) by triangles listed in this file.  They are the Delaunay
    triangulation of the raw (knots, degrees) conditions: that reproduces every
    triangulation file NORAH2 V2.0.74 ships, where the min-max normalized
    conditions the method report (D1.5d A.3) describes do not.

    Args:
        filename: Path to the output ``.int`` file; NORAH2 looks for it next to
            the hemispheres.
        hemispheres: iterable of (hem file name, speed_knots, flight_path_angle_deg),
            at least three conditions not all on one line.  File names are written as
            given, relative to the hemisphere folder.
        corrections: (dB, operation) rows of the third table. The default is
            the table every shipped file carries.
        overwrite: If False, raises when filename exists.

    Returns:
        (Ntri, 3) array of zero-based hemisphere indices, one row per triangle.
    """
    from scipy.spatial import Delaunay

    if not overwrite and os.path.exists(filename):
        raise FileExistsError(f'Output file already exists: {filename}')
    hemispheres = [(str(name), float(speed), float(angle)) for name, speed, angle in hemispheres]
    if len(hemispheres) < 3:
        raise ValueError('a NORAH2 triangulation needs at least 3 hemispheres')
    points = np.array([(speed, angle) for _, speed, angle in hemispheres])
    if np.unique(np.round(points, 6), axis=0).shape[0] != points.shape[0]:
        raise ValueError('two hemispheres share a flight condition; NORAH2 merges repeat runs '
                         'into one hemisphere per condition, so average them first')
    if _conditions_are_collinear(points[:, 0], points[:, 1]):
        raise ValueError('the conditions lie on one line in (speed, flight path angle); '
                         'NORAH2 needs at least one triangle')
    triangles = Delaunay(points).simplices

    lines = ['&HEMISPHERES', f'\tNGAD = {len(hemispheres)}', '&END', '',
             'iHem\tHEMSpeed\tHEMAngle\t[Path\\]FileHem']
    lines += [f'{i}\t{speed:g}\t{angle:g}\t{name}'
              for i, (name, speed, angle) in enumerate(hemispheres, start=1)]
    lines += ['', '&TRIANGLES', f'\tNTRI = {len(triangles)}', '&END', '',
              'iTri\tiHem1\tiHem2\tiHem3']
    lines += [f'{i}\t' + '\t'.join(str(int(v) + 1) for v in triangle)
              for i, triangle in enumerate(triangles, start=1)]
    lines += ['', '&CORRECTIONS', f'NCOR\t= {len(corrections)}', '&END', '',
              'Corr_dB\tOperation']
    lines += [f'{dB:g}\t"{operation}"' for dB, operation in corrections]
    with open(filename, 'w', encoding='ascii', newline='\r\n') as handle:
        handle.write('\n'.join(lines) + '\n')
    return triangles


# --- Importing NORAH2 hemispheres into a NICEOPS database -------------------
#
# The two formats disagree on four things, each worth a decibel or more if it
# is carried across unconverted (measured; see docs/norah2_import.md):
#
# - Levels.  A .hem stores levels at POLDIST (60 m) with the absorption over
#   those 60 m included, in the reference atmosphere TAMB/RELHUM/PAMB (ICAO,
#   25 C and 70 %).  A NICEOPS sphere stores them at its own radius with the
#   absorption of that first radius kept and nothing beyond it, which NICEOPS
#   adds back from the surface outward.  Read as lossless, the 60 m of ICAO
#   absorption is 0.33 dB of LA at the median (0.77 at worst), and 4-6 dB in
#   the 8-10 kHz bands.
# - Speed.  ACSPEED is labeled indicated airspeed, but NORAH2 looks hemispheres
#   up by the trajectory's ground speed.  Which one a file really holds depends
#   on who wrote it (panam writes ground speed: its tracks carry no airspeed),
#   so the importer will not guess.
# - Ground.  FREEFIELD = 2 hemispheres are free field.  FREEFIELD = 0 ones
#   (NORAH2's hover tables) include the ground reflection, which NICEOPS's
#   ground model would then add a second time.
# - Rotor sense.  NORAH2 represents a type whose main rotor turns the other way
#   by its class's hemispheres with the azimuth reversed (phi -> -phi, D1.5d
#   Annex C); that is a property of the use, not of the file.

#: Values of ``speed_mapping`` for :func:`build_database_from_norah2`, and the
#: database speed_reference each one produces.
NORAH2_SPEED_MAPPINGS = {'ias_to_tas': 'air', 'ground_speed': 'ground'}


#: ICAO standard sea-level density, kg/m^3: indicated airspeed is true airspeed
#: in air of this density.
STANDARD_SEA_LEVEL_DENSITY = 1.225


#: Specific gas constants of dry air and water vapor, J/(kg K).
DRY_AIR_GAS_CONSTANT = 287.058


WATER_VAPOR_GAS_CONSTANT = 461.495


#: Radius of every sphere in a panam-built NICEOPS database: 100 ft.
DATABASE_SPHERE_RADIUS_M = 30.48


#: The table constants a .hem must carry for it to be converted.
_NORAH2_REQUIRED_CONSTANTS = ('POLDIST', 'FREEFIELD', 'TAMB', 'RELHUM', 'PAMB', 'ACSPEED', 'GAMM')


def humid_air_density(temperature_k, pressure_pa, relative_humidity):
    """Density of moist air, kg/m^3, as the sum of its dry-air and vapor partial densities.

    The vapor pressure is ``relative_humidity`` percent of the ISO 9613-1
    saturation pressure, the same formula the absorption uses.
    """
    from panam.acoustics.iso_9613_1_1993 import saturation_pressure
    temperature_k = float(temperature_k)
    vapor_pa = float(relative_humidity) / 100.0 * saturation_pressure(temperature_k) * 1000.0
    return ((float(pressure_pa) - vapor_pa) / (DRY_AIR_GAS_CONSTANT * temperature_k)
            + vapor_pa / (WATER_VAPOR_GAS_CONSTANT * temperature_k))


def _norah2_reference_atmosphere(constants, filename):
    """The atmosphere a .hem's absorption to POLDIST was computed in, range-checked.

    TAMB is in kelvin and PAMB in pascals; a file written in Celsius or
    hectopascals would otherwise be read as a wildly different atmosphere and
    move every high band by decibels without complaint.
    """
    temperature = float(constants['TAMB'])
    humidity = float(constants['RELHUM'])
    pressure_pa = float(constants['PAMB'])
    if not 200.0 <= temperature <= 350.0:
        raise ValueError(f'{filename}: TAMB = {temperature:g} is not a temperature in kelvin '
                         '(expected 200-350 K)')
    if not 0.0 <= humidity <= 100.0:
        raise ValueError(f'{filename}: RELHUM = {humidity:g} is not a relative humidity in percent')
    if not 30000.0 <= pressure_pa <= 110000.0:
        raise ValueError(f'{filename}: PAMB = {pressure_pa:g} is not a pressure in pascals '
                         '(expected 30000-110000 Pa)')
    return Atmosphere(temperature=temperature, pressure=pressure_pa / 1000.0,
                      relative_humidity=humidity)


def norah2_levels_at_radius(hemisphere, radius_m=DATABASE_SPHERE_RADIUS_M, absorption_band_method='center'):
    """Band levels of a loaded .hem moved from POLDIST to a NICEOPS sphere's radius.

    The level at POLDIST includes absorption from the center out to POLDIST in
    the file's reference atmosphere (TAMB/RELHUM/PAMB).  A NICEOPS sphere keeps
    the absorption over its own first ``radius_m`` and is otherwise lossless, so
    the absorption between ``radius_m`` and POLDIST is taken back out and the
    spreading undone::

        L(radius) = L(POLDIST) + 20 log10(POLDIST / radius) + alpha (POLDIST - radius)

    alpha being the ISO 9613-1 coefficient at each band's center in the
    reference atmosphere.  For a radius beyond POLDIST the same expression adds
    the absorption between the two instead.  Missing cells stay -inf.

    With ``absorption_band_method`` other than 'center' the absorption term is
    the band attenuation over POLDIST less that over ``radius_m``
    (:func:`panam.acoustics.atmosphere.band_absorption_db`): 'arp5534', SAE ARP
    5534 as NORAH2 itself applies it, or 'integrated' (flat band levels).

    Args:
        hemisphere: a :func:`load_norah2_hemisphere` result.
        radius_m: the sphere radius, meters.
        absorption_band_method: 'center' (default), 'arp5534' or 'integrated'.

    Returns: (phi, theta, frequency) levels at ``radius_m``, dB.
    """
    constants = hemisphere['constants']
    for name in ('POLDIST', 'TAMB', 'RELHUM', 'PAMB'):
        if name not in constants:
            raise ValueError(f'{hemisphere.get("title", "hemisphere")}: no {name} in the table constants')
    poldist = float(constants['POLDIST'])
    radius_m = float(radius_m)
    if not (np.isfinite(poldist) and poldist > 0.0):
        raise ValueError(f'POLDIST must be a positive distance in meters, not {poldist:g}')
    if not (np.isfinite(radius_m) and radius_m > 0.0):
        raise ValueError(f'radius_m must be a positive distance in meters, not {radius_m:g}')
    atmosphere = _norah2_reference_atmosphere(constants, hemisphere.get('title', 'hemisphere'))
    frequency = np.asarray(hemisphere['frequency_hz'], dtype=float)
    if absorption_band_method == 'center':
        absorption_db = _band_attenuation_db(atmosphere, frequency, poldist - radius_m, 'center')
    else:
        absorption_db = np.diff(_band_attenuation_db(atmosphere, frequency, np.array([radius_m, poldist]),
                                                     absorption_band_method), axis=1)[:, 0]
    offset_db = 20.0 * np.log10(poldist / radius_m) + absorption_db
    return mask_missing_levels(hemisphere['levels_db']) + offset_db[None, None, :]


def norah2_speed_knots(constants, speed_mapping, filename='', air_density=None):
    """The speed a .hem's condition is filed at in the database, knots.

    ``speed_mapping`` says what ACSPEED is taken to be; there is no default,
    because the format labels it indicated airspeed while NORAH2 itself looks it
    up by ground speed, and files differ in which they really hold.

    - ``'ias_to_tas'``: indicated airspeed, converted to true airspeed as
      ``IAS sqrt(1.225 / rho)``.  rho is ``air_density`` when given, and
      otherwise the moist-air density of the file's measurement atmosphere
      (Tm in Celsius, Pm in pascals, RHm in percent); a file without them is
      refused.  The database is then air-referenced.
    - ``'ground_speed'``: ACSPEED taken as the ground speed, unchanged, as
      NORAH2 uses it.  The database is ground-referenced.

    Instrument and position error, and compressibility (0.1 % at 150 kt), are
    not modeled: IAS is taken as equivalent airspeed.
    """
    if speed_mapping not in NORAH2_SPEED_MAPPINGS:
        raise ValueError(f'speed_mapping must be one of {sorted(NORAH2_SPEED_MAPPINGS)}, '
                         f'not {speed_mapping!r}')
    acspeed = float(constants['ACSPEED'])
    if not (np.isfinite(acspeed) and acspeed >= 0.0):
        raise ValueError(f'{filename}: ACSPEED = {acspeed:g} is not a speed')
    if speed_mapping == 'ground_speed':
        return acspeed
    if air_density is None:
        missing = [name for name in ('Tm', 'Pm', 'RHm') if name not in constants]
        if missing:
            raise ValueError(f'{filename}: converting indicated to true airspeed needs the air '
                             f'density, and the file has no {", ".join(missing)}; give air_density')
        temperature = float(constants['Tm']) + 273.15
        pressure_pa = float(constants['Pm'])
        humidity = float(constants['RHm'])
        if not (200.0 <= temperature <= 350.0 and 30000.0 <= pressure_pa <= 110000.0
                and 0.0 <= humidity <= 100.0):
            raise ValueError(f'{filename}: measurement atmosphere Tm = {constants["Tm"]:g} C, '
                             f'Pm = {pressure_pa:g} Pa, RHm = {humidity:g} % is out of range; '
                             'give air_density')
        air_density = humid_air_density(temperature, pressure_pa, humidity)
    air_density = float(air_density)
    if not (np.isfinite(air_density) and 0.3 <= air_density <= 1.6):
        raise ValueError(f'air density {air_density:g} kg/m^3 is out of range (0.3-1.6)')
    return acspeed * np.sqrt(STANDARD_SEA_LEVEL_DENSITY / air_density)


def fill_norah2_empty_cells(levels_db, phi_deg, theta_deg):
    """Fill every missing (-inf) cell of a hemisphere from its nearest filled direction.

    NORAH2 does this at use time: a .hem leaves 33-62 % of its lower-hemisphere
    bins at NOVALUE, and its predictor takes an empty bin's level from the
    nearest bin that has one.  NICEOPS instead reads an unmeasured cell as no
    data, which drops out of a blend where neighbors have data and is silence
    where none do.  This reproduces NORAH2's behavior in the database, band by
    band.  "Nearest" is the smallest angle between the two directions (body
    axes x = cos theta, y = sin theta sin phi, z = sin theta cos phi); a tie
    goes to the first cell in (phi, theta) order.

    Args:
        levels_db: (phi, theta, band) levels, -inf where missing.
        phi_deg, theta_deg: the axes.

    Returns: the filled copy.  A band with no level anywhere stays empty.
    """
    levels_db = np.array(levels_db, dtype=float, copy=True)
    PH, TH = np.meshgrid(np.radians(np.asarray(phi_deg, dtype=float)),
                         np.radians(np.asarray(theta_deg, dtype=float)), indexing='ij')
    directions = np.stack((np.cos(TH), np.sin(TH) * np.sin(PH), np.sin(TH) * np.cos(PH)),
                          axis=-1).reshape(-1, 3)
    closeness = directions @ directions.T
    flat = levels_db.reshape(-1, levels_db.shape[-1])
    for band in range(flat.shape[1]):
        present = np.isfinite(flat[:, band])
        if present.all() or not present.any():
            continue
        missing = np.flatnonzero(~present)
        candidates = np.flatnonzero(present)
        nearest = candidates[np.argmax(closeness[np.ix_(missing, candidates)], axis=1)]
        flat[missing, band] = flat[nearest, band]
    return flat.reshape(levels_db.shape)


def build_database_from_norah2(hem_files, database_filename, *, main_rotor_tip_speed_mps,
                               main_rotor_radius_m, vehicle_weight_newtons, speed_mapping,
                               thrust_air_density=STANDARD_SEA_LEVEL_DENSITY, tas_air_density=None,
                               radius_m=DATABASE_SPHERE_RADIUS_M,
                               atmosphere=Atmosphere(temperature=293.15, pressure=101.325,
                                                     relative_humidity=20.0),
                               accept_ground_included=False, mirror_rotor=False,
                               fill_empty=False, synthesize_hover=False,
                               scale_tip_speed_by_rmomega=False, store_spectrum=True, overwrite=True,
                               absorption_band_method='center'):
    """Build a NICEOPS database from NORAH2 (HELENA, ECAC Doc 32) ``.hem`` hemispheres.

    One condition per file, at load factor 1 with ``fixed_load_factor`` set, so
    NICEOPS scales the level to the trajectory's thrust coefficient itself.
    Each file is converted as follows:

    - levels moved from POLDIST to ``radius_m`` (see :func:`norah2_levels_at_radius`):
      spherical spreading, and the ISO 9613-1 absorption between ``radius_m``
      and POLDIST in the file's reference atmosphere taken back out, which
      leaves the first ``radius_m`` of absorption in the sphere as panam's own
      databases keep it;
    - NOVALUE (-999) cells stored as missing (-inf, which NICEOPS reads as
      unmeasured) with coverage 0; a direction with no band left has dBA -inf
      and EAA 0.  With ``fill_empty`` they take their nearest filled
      direction's level instead, as NORAH2 does at use time, still with
      coverage 0;
    - the upper hemisphere completed by panam's mirror, phi -> 180 - phi, with
      coverage 0 (:func:`mirror_phi_to_upper_surface`);
    - the speed mapped by ``speed_mapping`` (:func:`norah2_speed_knots`), and the
      flight path angle taken from GAMM (negative in descent) as stored.

    With ``synthesize_hover``, a hover is added as :func:`build_empirical_database`
    makes one (THEORY 7.4): the slowest converted condition within
    :data:`LEVEL_FLIGHT_TOLERANCE` degrees of level flight (the slowest overall,
    with a warning, if none is) is averaged fore to aft in energy, a cell keeps
    coverage 1 only where both partners were measured, and the result is written
    at advance ratio 0 and flight path angles -12, 0 and +12 deg
    (:data:`HOVER_FLIGHT_PATH_ANGLES`).  Its groups carry ``condition_origin =
    'synthesized_hover'`` and the source file's ``norah2_source`` and constants.

    The root records ``speed_reference`` from the mapping,
    ``azimuth_reference = 'track'`` (NORAH2 hemispheres are filed against the
    ground track, attitude not considered), ``database_version`` 1, the build
    atmosphere of the EAA, and the vehicle's rotor and weight; every group
    carries ``DOPPLER_SHIFT_REMOVED = 0``.  NORAH2 describes no step that
    removes the Doppler shift, and its predictor applies none, so its spheres
    are taken to keep the shift the microphones received -- inferred, not
    stated by EASA.

    Args:
        hem_files: .hem paths, one per flight condition.  Every file must share
            one angle grid and band set.
        database_filename: output .nod path.
        main_rotor_tip_speed_mps, main_rotor_radius_m, vehicle_weight_newtons:
            the vehicle.  They set the advance ratio and thrust coefficient of
            every condition and are written at the root, where NICEOPS
            compares them with its vehicle file.  Required: a .hem carries none
            of them.
        speed_mapping: ``'ias_to_tas'`` or ``'ground_speed'``; required, see
            :func:`norah2_speed_knots`.
        thrust_air_density: the density the thrust coefficient W / (rho A V_tip^2)
            is formed with, kg/m^3.  Use the vehicle file's.
        tas_air_density: ``'ias_to_tas'`` only: one density for every file in
            place of each file's measurement atmosphere.
        radius_m: sphere radius, meters.
        atmosphere: the atmosphere the broadband EAA is computed in (over
            1000 m, as :func:`build_empirical_database` does).
        accept_ground_included: convert FREEFIELD = 0 hemispheres, whose levels
            include the ground reflection, with a warning instead of refusing
            them.  Their levels are taken as free field, so the ground is
            counted twice wherever NICEOPS applies its own ground model.
        mirror_rotor: reverse the azimuth (phi -> -phi), NORAH2's way of
            using a class's hemispheres for a type whose main rotor turns the
            other way.
        fill_empty: fill each NOVALUE cell from its nearest filled direction
            (:func:`fill_norah2_empty_cells`), as NORAH2 does at use time.  The
            filled cells keep coverage 0.  Off by default: NICEOPS then reads
            them as unmeasured, silent where no neighbor has data.
        synthesize_hover: add a hover synthesized from the slowest level
            condition (above).  Off by default: NICEOPS then clamps every state
            slower than the slowest file to it.  The theta grid must be
            symmetric about 90 deg, so that theta and 180 - theta are both on it.
        scale_tip_speed_by_rmomega: form each file's advance ratio with the tip
            speed its rotor ran at, ``main_rotor_tip_speed_mps * RmOmega / 100``
            (RmOmega in percent of nominal), instead of the nominal tip speed.
            A file without RmOmega, or with one outside 50-130 %, is refused.
            The thrust coefficient and the root's tip speed stay nominal.  Off
            by default.
        store_spectrum: keep the band levels in each group, as NICEOPS's
            spectral path needs.  Without them only dBA and EAA are written.
        overwrite: if False, refuse an existing ``database_filename``.
        absorption_band_method: how band absorption is formed, both in the move
            from POLDIST to ``radius_m`` and in the EAA
            (:func:`panam.acoustics.atmosphere.band_absorption_db`): 'center'
            (default, the coefficient at the band center), 'arp5534' (SAE ARP
            5534, NORAH2's own) or 'integrated'.  The database records it only in
            the provenance settings.

    The root carries the provenance stamp of :mod:`provenance`: the processing
    arguments of this call as canonical JSON, and the .hem files and the
    database by name in ``panam_inputs``.

    Returns: list of dicts, one per written condition, in group order: source
        file, speed_knots, flight_path_angle_deg, advance_ratio,
        main_rotor_tip_speed_mps (the tip speed the advance ratio is formed
        with, m/s) and condition_origin ('measured' or 'synthesized_hover').
    """
    # File identity apart from the hashed settings; paths by file name only (a relative
    # path's directories name this machine's layout too).
    settings = {key: value for key, value in locals().items()
                if key not in ('hem_files', 'database_filename', 'overwrite')}
    inputs = dict(hem_files=[_provenance.path_name(f) for f in hem_files],
                  database_filename=_provenance.path_name(database_filename))
    if speed_mapping not in NORAH2_SPEED_MAPPINGS:
        raise ValueError('speed_mapping is required and must be one of {}: whether ACSPEED is an '
                         'indicated airspeed to convert to true airspeed or a ground speed. Got {!r}'
                         .format(sorted(NORAH2_SPEED_MAPPINGS), speed_mapping))
    vehicle = dict(main_rotor_tip_speed_mps=main_rotor_tip_speed_mps,
                   main_rotor_radius_m=main_rotor_radius_m,
                   vehicle_weight_newtons=vehicle_weight_newtons,
                   thrust_air_density=thrust_air_density)
    for name, value in vehicle.items():
        if value is None or not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f'{name} must be a positive number (a .hem carries no vehicle data), '
                             f'not {value!r}')
    main_rotor_tip_speed = float(main_rotor_tip_speed_mps)
    main_rotor_radius = float(main_rotor_radius_m)
    weight_coefficient = float(vehicle_weight_newtons) / (
        float(thrust_air_density) * np.pi * main_rotor_radius ** 2 * main_rotor_tip_speed ** 2)
    if tas_air_density is not None and speed_mapping != 'ias_to_tas':
        raise ValueError("tas_air_density applies to speed_mapping='ias_to_tas' only")

    _check_build_atmosphere(atmosphere)
    _check_band_method(absorption_band_method)

    # realpath, so a symbolic link or a second spelling of one file is caught too.
    hem_files = [os.path.realpath(os.path.expanduser(str(f))) for f in hem_files]
    if not hem_files:
        raise ValueError('no .hem files given')
    if len(set(hem_files)) != len(hem_files):
        raise ValueError('a .hem file is given more than once')
    database_path = os.path.realpath(os.path.expanduser(str(database_filename)))
    if database_path in hem_files:
        raise ValueError('the output database is one of the input files')
    if not overwrite and os.path.exists(database_path):
        raise FileExistsError(f'Output file already exists: {database_filename}')

    # Everything is read and checked before the output is opened, so a refused
    # file leaves no half-written database behind.
    pending_groups = []
    conditions = []
    hover_sources = []
    grid = None
    for index, filename in enumerate(hem_files):
        name = os.path.basename(filename)
        try:
            hemisphere = load_norah2_hemisphere(filename)
        except (ValueError, IndexError) as error:
            # IndexError: a file that ends before its header says it should.
            reason = str(error).replace(filename + ': ', '') or type(error).__name__
            if isinstance(error, IndexError):
                reason = f'it ends early ({reason})'
            message = f'{name}: cannot be converted ({reason}).'
            if 'axis' in reason or 'axes' in reason:
                message += (' Only two-axis (THETAOBSAC x PHIOBSAC) hemispheres become spheres; '
                            "NORAH2's one-axis hover and idle tables have no elevation axis.")
            raise ValueError(message) from None
        constants = hemisphere['constants']
        missing = [c for c in _NORAH2_REQUIRED_CONSTANTS if c not in constants]
        if missing:
            raise ValueError(f'{name}: no {", ".join(missing)} in the table constants')
        freefield = float(constants['FREEFIELD'])
        if freefield == 0.0:
            if not accept_ground_included:
                raise ValueError(f'{name}: FREEFIELD = 0, so its levels include the ground '
                                 'reflection (NORAH2 writes its hover tables so). NICEOPS adds '
                                 'its own ground; pass accept_ground_included to convert it anyway.')
            warnings.warn(f'{name}: FREEFIELD = 0 (ground reflection included) converted as free '
                          'field; the ground is counted twice wherever NICEOPS applies its own')
        elif freefield != 2.0:
            raise ValueError(f'{name}: FREEFIELD = {freefield:g} is not a convention this importer '
                             'knows (2 = free field with absorption to POLDIST, 0 = ground included)')

        phi_deg = np.asarray(hemisphere['phi_deg'], dtype=float)
        theta_deg = np.asarray(hemisphere['theta_deg'], dtype=float)
        frequency = np.asarray(hemisphere['frequency_hz'], dtype=float)
        if np.any(np.abs(phi_deg) > 90.0 + 1e-9) or np.any(np.diff(phi_deg) <= 0.0):
            raise ValueError(f'{name}: PHIOBSAC must increase within [-90, 90] deg')
        if np.any(theta_deg < -1e-9) or np.any(theta_deg > 180.0 + 1e-9) or np.any(np.diff(theta_deg) <= 0.0):
            raise ValueError(f'{name}: THETAOBSAC must increase within [0, 180] deg')
        if synthesize_hover and not np.allclose(theta_deg, 180.0 - theta_deg[::-1]):
            # average_fore_and_aft pairs theta index i with n-1-i, a fore-aft
            # pair only on a grid symmetric about 90 deg.
            raise ValueError(f'{name}: THETAOBSAC is not symmetric about 90 deg, so a hover '
                             'cannot be averaged fore and aft from it; leave out synthesize_hover')
        if np.any(frequency <= 0.0) or np.any(np.diff(frequency) <= 0.0):
            raise ValueError(f'{name}: band centers must be positive and increasing')
        if grid is None:
            grid = (phi_deg, theta_deg, frequency, name)
        elif not (np.array_equal(grid[0], phi_deg) and np.array_equal(grid[1], theta_deg)
                  and np.array_equal(grid[2], frequency)):
            raise ValueError(f'{name} and {grid[3]} have different angle grids or bands; one '
                             'database takes one layout')

        amplitude = norah2_levels_at_radius(hemisphere, radius_m, absorption_band_method)
        if mirror_rotor:
            # phi -> -phi reverses the row order; keep the rows ascending.
            amplitude = amplitude[::-1]
            phi_list = -phi_deg[::-1]
        else:
            phi_list = phi_deg
        if not np.any(np.isfinite(amplitude)):
            raise ValueError(f'{name}: every cell is NOVALUE')

        speed_knots = norah2_speed_knots(constants, speed_mapping, name, tas_air_density)
        flight_path_angle = float(constants['GAMM'])
        if not (np.isfinite(flight_path_angle) and abs(flight_path_angle) < 90.0):
            raise ValueError(f'{name}: GAMM = {flight_path_angle:g} is not a flight path angle')
        tip_speed = main_rotor_tip_speed
        if scale_tip_speed_by_rmomega:
            tip_speed = main_rotor_tip_speed * _norah2_rotor_speed_fraction(constants, name)

        spla, eaa = spla_and_eaa_from_spectrum(amplitude, frequency, 1000.0, atmosphere,
                                               absorption_band_method)
        # Coverage is what the file measured, taken before any fill.
        measured = _measured_bins(amplitude, eaa)
        if fill_empty:
            amplitude = fill_norah2_empty_cells(amplitude, phi_list, theta_deg)
            spla, eaa = spla_and_eaa_from_spectrum(amplitude, frequency, 1000.0, atmosphere,
                                                   absorption_band_method)
        spla, eaa = _finite_sphere_levels(spla, eaa)
        phi_full, theta_full, spla_full, eaa_full, amplitude_full, coverage_full = (
            _complete_sphere_with_coverage(phi_list, theta_deg, spla, eaa, amplitude, measured))
        pending_groups.append(_SphereCondition(
            f'sphere{index}', phi_full, theta_full, float(radius_m), spla_full, eaa_full,
            speed_knots, flight_path_angle, 1.0,
            frequency if store_spectrum else None,
            amplitude_full if store_spectrum else None, coverage_full))
        conditions.append(dict(source=name, speed_knots=float(speed_knots),
                               flight_path_angle_deg=flight_path_angle,
                               advance_ratio=KNOT_MPS * float(speed_knots) / tip_speed,
                               main_rotor_tip_speed_mps=tip_speed,
                               condition_origin='measured', constants=constants))
        if synthesize_hover:
            # What the hover is synthesized from: the levels as converted (and
            # filled), and the coverage taken before any fill.
            hover_sources.append((phi_list, theta_deg, frequency, amplitude, measured))

    if synthesize_hover:
        source = _hover_source_index([(c['speed_knots'], c['flight_path_angle_deg']) for c in conditions])
        phi_list, theta_deg, frequency, amplitude, measured = hover_sources[source]
        phi_full, theta_full, spla_full, eaa_full, amplitude_full, coverage_full = (
            _synthesized_hover_sphere(phi_list, theta_deg, amplitude, measured, frequency,
                                      1000.0, atmosphere,
                                      absorption_band_method=absorption_band_method))
        source_condition = conditions[source]
        for flight_path_angle in _hover_flight_path_angles():
            pending_groups.append(_SphereCondition(
                f'sphere{len(pending_groups)}', phi_full, theta_full, float(radius_m), spla_full,
                eaa_full, 0.0, float(flight_path_angle), 1.0,
                frequency if store_spectrum else None,
                amplitude_full if store_spectrum else None, coverage_full))
            conditions.append(dict(source=source_condition['source'], speed_knots=0.0,
                                   flight_path_angle_deg=float(flight_path_angle), advance_ratio=0.0,
                                   main_rotor_tip_speed_mps=source_condition['main_rotor_tip_speed_mps'],
                                   condition_origin='synthesized_hover',
                                   constants=source_condition['constants']))

    # NICEOPS refuses two spheres at one condition, and a set of two or more
    # that does not span both axes it triangulates over; say so here, naming
    # the files.
    keys = {}
    for condition in conditions:
        key = (round(condition['advance_ratio'], 9), round(condition['flight_path_angle_deg'], 9))
        label = condition['source']
        if condition['condition_origin'] == 'synthesized_hover':
            label = f'the hover synthesized from {label}'
        if key in keys:
            raise ValueError('{} and {} have the same flight condition ({:.4g} kt, {:.4g} deg); '
                             'NICEOPS takes one sphere per condition, so merge or drop one'
                             .format(keys[key], label, condition['speed_knots'],
                                     condition['flight_path_angle_deg']))
        keys[key] = label
    if len(conditions) >= 2:
        mus = np.array([c['advance_ratio'] for c in conditions])
        gammas = np.array([c['flight_path_angle_deg'] for c in conditions])
        # NICEOPS refuses a database of two or more spheres in which either
        # axis is constant, not only one of three or more (measured on the
        # pinned binary: two level-flight spheres stop the run).
        if np.ptp(mus) == 0.0 or np.ptp(gammas) == 0.0:
            raise ValueError('the hemispheres do not vary in both speed and flight path angle '
                             f'({", ".join(c["source"] for c in conditions)}); NICEOPS refuses a '
                             'database of two or more spheres that does not')
    if len(conditions) < 3:
        warnings.warn(f'{len(conditions)} hemisphere(s): NICEOPS takes the nearest condition '
                      'rather than interpolating between fewer than three')
    elif _conditions_are_collinear(mus, gammas):
        # NICEOPS loads these, but its triangulation has no simplex (it logs
        # "0 simplices") and every query falls back to the nearest condition.
        warnings.warn(f'the {len(conditions)} hemispheres lie on one line in (speed, flight path '
                      'angle), so NICEOPS finds no triangle among them and takes the nearest '
                      'condition rather than interpolating')

    # Written to a temporary file beside the output and moved into place only
    # when complete, so a failure while writing neither leaves a partial
    # database nor destroys an existing one (_atomic_output).
    with _atomic_output(database_path) as temporary:
        _write_norah2_database(temporary, pending_groups, conditions,
                               speed_mapping=speed_mapping, atmosphere=atmosphere,
                               main_rotor_radius=main_rotor_radius,
                               main_rotor_tip_speed=main_rotor_tip_speed,
                               vehicle_weight_newtons=float(vehicle_weight_newtons),
                               weight_coefficient=weight_coefficient,
                               thrust_air_density=float(thrust_air_density),
                               mirror_rotor=mirror_rotor,
                               accept_ground_included=accept_ground_included,
                               fill_empty=fill_empty,
                               scale_tip_speed_by_rmomega=scale_tip_speed_by_rmomega,
                               provenance=_provenance.build_provenance(
                                   'flight_acoustics.build_database_from_norah2', settings,
                                   inputs=inputs))
    return [{k: v for k, v in c.items() if k != 'constants'} for c in conditions]


def _norah2_rotor_speed_fraction(constants, filename):
    """A .hem's main rotor speed as a fraction of nominal, from RmOmega (percent).

    Refuses a file without RmOmega, or with one outside 50-130 %: shipped A109,
    EC120, EC135 and R66 files carry values such as -999, -479, 44 or (most
    EC120 files) 337-410, which look like rpm.
    """
    if 'RmOmega' not in constants:
        raise ValueError(f'{filename}: no RmOmega in the table constants, so its tip speed cannot '
                         'be scaled; leave out scale_tip_speed_by_rmomega or the file')
    percent = float(constants['RmOmega'])
    if not (np.isfinite(percent) and 50.0 <= percent <= 130.0):
        raise ValueError(f'{filename}: RmOmega = {percent:g} is not a rotor speed in percent of '
                         'nominal (expected 50-130; a value near 400 may be rpm)')
    return percent / 100.0


def _conditions_are_collinear(mus, gammas):
    """Whether (advance ratio, flight path angle) points all lie on one line.

    Taken as NICEOPS takes them, each axis normalized onto [0, 1] (a
    collinear set stays collinear under that, and the tolerance then has a
    scale).  A constant axis is collinear.
    """
    if np.ptp(mus) == 0.0 or np.ptp(gammas) == 0.0:
        return True
    points = np.column_stack(((mus - mus.min()) / np.ptp(mus),
                              (gammas - gammas.min()) / np.ptp(gammas)))
    offsets = points[1:] - points[0]
    return bool(np.linalg.matrix_rank(offsets, tol=1e-9) < 2)


def _check_build_atmosphere(atmosphere):
    """Refuse a build atmosphere (the EAA's) given in the wrong units.

    The same bounds as a .hem's reference atmosphere, in the units Atmosphere
    takes: kelvin, kPa, percent.
    """
    temperature = float(atmosphere.temperature)
    pressure = float(atmosphere.pressure)
    humidity = float(atmosphere.relative_humidity)
    if not 200.0 <= temperature <= 350.0:
        raise ValueError(f'build atmosphere: temperature {temperature:g} is not in kelvin '
                         '(expected 200-350 K)')
    if not 30.0 <= pressure <= 110.0:
        raise ValueError(f'build atmosphere: pressure {pressure:g} is not in kPa '
                         '(expected 30-110 kPa)')
    if not 0.0 <= humidity <= 100.0:
        raise ValueError(f'build atmosphere: relative humidity {humidity:g} is not a percentage')


def _write_norah2_database(path, pending_groups, conditions, *, speed_mapping, atmosphere,
                           main_rotor_radius, main_rotor_tip_speed, vehicle_weight_newtons,
                           weight_coefficient, thrust_air_density, mirror_rotor,
                           accept_ground_included, fill_empty, scale_tip_speed_by_rmomega,
                           provenance=None):
    """Write the checked conditions of :func:`build_database_from_norah2` to ``path``."""
    ncdatabase = Dataset(path, 'w')
    try:
        _write_database_root(ncdatabase, fixed_load_factor=True,
                             speed_reference=NORAH2_SPEED_MAPPINGS[speed_mapping],
                             atmosphere=atmosphere, main_rotor_radius=main_rotor_radius,
                             main_rotor_tip_speed=main_rotor_tip_speed,
                             vehicle_weight_newtons=vehicle_weight_newtons,
                             azimuth_reference='track', provenance=provenance)
        ncdatabase.source_format = 'NORAH2 hemisphere (.hem)'
        ncdatabase.norah2_speed_mapping = speed_mapping
        ncdatabase.norah2_mirror_rotor = int(bool(mirror_rotor))
        ncdatabase.norah2_ground_included_accepted = int(bool(accept_ground_included))
        ncdatabase.norah2_fill_empty = 'nearest' if fill_empty else 'none'
        ncdatabase.norah2_thrust_air_density_kg_m3 = thrust_air_density
        hovers = [c['source'] for c in conditions if c['condition_origin'] == 'synthesized_hover']
        ncdatabase.norah2_hover_source = hovers[0] if hovers else 'none'
        ncdatabase.norah2_scale_tip_speed_by_rmomega = int(bool(scale_tip_speed_by_rmomega))
        shared = _write_shared_grid_and_frequency(ncdatabase, pending_groups)
        for group, condition in zip(pending_groups, conditions):
            # The advance ratio is formed with the condition's own tip speed
            # (the nominal one unless scaled by RmOmega); C_T keeps the nominal.
            add_sphere_group(ncdatabase, group.name, group.phi, group.theta, group.radius,
                             group.SPLA, group.EAA, group.speed, group.flight_path_angle,
                             group.load_factor, main_rotor_radius,
                             condition['main_rotor_tip_speed_mps'],
                             weight_coefficient, group.frequency, group.amplitude,
                             write_grid_and_frequency=not shared, doppler_shift_removed=0,
                             coverage=group.coverage)
            # Provenance: what the file said, before any mapping.
            written = ncdatabase[group.name]
            written.norah2_source = condition['source']
            if condition['condition_origin'] != 'measured':
                written.condition_origin = condition['condition_origin']
            for constant in ('POLDIST', 'FREEFIELD', 'TAMB', 'RELHUM', 'PAMB', 'ACSPEED', 'GAMM',
                             'RmOmega', 'Tm', 'RHm', 'Pm', 'PITCH', 'ROLL', 'TW', 'CW', 'HW'):
                if constant in condition['constants']:
                    setattr(written, 'norah2_' + constant, float(condition['constants'][constant]))
    finally:
        # Closed explicitly, so a failed flush raises here rather than being
        # lost to the garbage collector.
        ncdatabase.close()
