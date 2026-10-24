"""Convert an AAM sphere netCDF to a NORAH2 (HELENA) .hem hemisphere.

Usage:  python tools/sphere_to_norah2.py SPHERE.nc -o HEMISPHERE.hem
            --temperature K --pressure KPA --humidity PERCENT

A sphere file keeps the absorption over its RADIUS in the atmosphere of its
measurement, but does not record that atmosphere, so it is required.  The
levels are moved to NORAH2's 60 m and ICAO reference atmosphere by
flight_acoustics.write_norah2_levels; see docs/THEORY.md section 8.  NORAH2's
hemisphere is level; a sphere in AAM's flight-path frame is pitched to level first
(docs/THEORY.md section 3.2).
"""
import argparse
import sys

import numpy as np

from panam.acoustics.atmosphere import Atmosphere, BAND_ABSORPTION_METHODS

from panam import flight_acoustics as fa
from panam.geometry import SPHERE_FRAMES
from panam.gridding import repitch_art_levels
from panam.sphere_io import read_sphere_frame


def sphere_levels_on_norah2_grid(amplitude_db, phi_deg, theta_deg):
    """Sample AAM sphere band levels on the NORAH2 10 degree grid.

    Sphere files and NORAH2 share the ART angles (phi 0 below, positive to
    starboard; theta 0 at the nose), so the levels are interpolated over (phi,
    theta) directly, bilinearly in linear power.  A sphere on the default
    build grid (phi every 10 deg, theta every 5 deg) holds every NORAH2
    direction, so the levels pass through unchanged.

    The sphere's missing levels (NaN, -999 or > 1e34; see
    :func:`flight_acoustics.mask_missing_levels`) mean either unmeasured or
    no energy, and the file does not say which.  They are taken as unmeasured:
    they carry no interpolation weight, and a direction weighted mostly by them,
    or outside the sphere's grid, is missing, as in
    :func:`flight_acoustics._sample_hemisphere_levels`.

    Args:
        amplitude_db: (Nphi, Ntheta, Nf) band levels (dB), as
            :func:`flight_acoustics.load_nc_sphere` returns them.
        phi_deg: (Nphi,) sphere PHI (deg), strictly increasing.
        theta_deg: (Ntheta,) sphere THETA (deg), strictly increasing.

    Returns:
        (19, 19, Nf) levels (dB) at (``NORAH2_PHI_DEG``, ``NORAH2_THETA_DEG``),
        -inf where missing.
    """
    phi_deg = np.asarray(phi_deg, dtype=float)
    theta_deg = np.asarray(theta_deg, dtype=float)
    levels_db = fa.mask_missing_levels(amplitude_db)
    if levels_db.shape[:2] != (phi_deg.size, theta_deg.size):
        raise ValueError('AMPLITUDE must be (PHI, THETA, FREQUENCY)')
    if phi_deg.size < 2 or theta_deg.size < 2 or np.any(np.diff(phi_deg) <= 0.0) \
            or np.any(np.diff(theta_deg) <= 0.0):
        raise ValueError('sphere PHI and THETA must be strictly increasing, with at least 2 values each')
    nfreq = levels_db.shape[2]
    measured = np.isfinite(levels_db)
    power = np.power(10.0, np.where(measured, levels_db, 0.0) / 10.0)
    PH, TH = np.meshgrid(fa.NORAH2_PHI_DEG, fa.NORAH2_THETA_DEG, indexing='ij')
    sampled_power, _ = fa._interpolate_measured_power(
        (phi_deg, theta_deg), power, measured, np.column_stack((PH.ravel(), TH.ravel())))
    out = np.full_like(sampled_power, -np.inf)
    positive = sampled_power > 0.0
    out[positive] = 10.0 * np.log10(sampled_power[positive])
    return out.reshape(PH.shape + (nfreq,))


def check_measurement_atmosphere(temperature_k, pressure_kpa, humidity_percent):
    """Reject a measurement atmosphere outside physical bounds, most likely a unit slip.

    Args:
        temperature_k: temperature (K), 200 to 350.
        pressure_kpa: static pressure (kPa), 30 to 120.
        humidity_percent: relative humidity (percent), 0 to 100.

    Raises:
        ValueError: naming the expected unit, e.g. for a pressure in Pa or a
            temperature in deg C.
    """
    checks = (('temperature', temperature_k, 200.0, 350.0, 'K'),
              ('pressure', pressure_kpa, 30.0, 120.0, 'kPa'),
              ('relative humidity', humidity_percent, 0.0, 100.0, 'percent'))
    for name, value, low, high, unit in checks:
        value = float(value)
        if not low <= value <= high:
            raise ValueError(f'measurement {name} {value:g} is outside {low:g} to {high:g} {unit}: '
                             f'give it in {unit}')


def sphere_to_norah2(sphere_filename, hem_filename, *, measurement_atmosphere,
                     speed_knots=None, flight_path_angle_deg=None, apply_absorption_deprop=True,
                     sphere_frame=None, **norah2_options):
    """Write an AAM sphere netCDF as a NORAH2 ``.hem`` hemisphere.

    Args:
        sphere_filename: AAM sphere (``RADIUS`` ft, ``SPEED`` kt, band levels in
            dB at ``RADIUS``).
        hem_filename: Output ``.hem`` file.
        measurement_atmosphere: :class:`Atmosphere` (K, kPa, % RH) of the
            measurement, whose absorption over ``RADIUS`` the sphere keeps.
            Required: the sphere file does not record it.
        speed_knots: ``ACSPEED`` (kt). Default: the sphere's ``SPEED``.
        flight_path_angle_deg: ``GAMM`` (deg). Default: the sphere's
            ``FLIGHT_PATH_ANGLE``.
        apply_absorption_deprop: False for a sphere built without absorption
            depropagation; written with spreading only, with a warning.
        sphere_frame: the frame the sphere is filed in (:data:`geometry.SPHERE_FRAMES`).
            Default: the file's own ``sphere_frame``, or 'level' when it records none.
            A 'flight_path' sphere is pitched to level by its own FLIGHT_PATH_ANGLE
            (not by ``flight_path_angle_deg``, which only labels the output).
        norah2_options: passed on to :func:`flight_acoustics.write_norah2_levels`
            (title, test_point, rotor_rpm_percent, minimum_level_db, overwrite,
            absorption_band_method, ...).

    Returns:
        (phi, theta, band) levels as written at 60 m (dB), -inf where -999.
    """
    if not isinstance(measurement_atmosphere, Atmosphere):
        raise TypeError('measurement_atmosphere must be an Atmosphere: the sphere file does not '
                        'record the atmosphere of its measurement')
    check_measurement_atmosphere(measurement_atmosphere.temperature,
                                 measurement_atmosphere.pressure,
                                 measurement_atmosphere.relative_humidity)
    amplitude, phi, theta, frequency, radius, speed, angle = fa.load_nc_sphere(sphere_filename)
    if np.all(fa._norah2_band_index(frequency) < 0):
        raise ValueError(f'{sphere_filename}: no FREQUENCY is a one-third octave band of NORAH2 '
                         '(10 Hz to 10 kHz)')
    radius_ft = float(np.ravel(radius)[0])
    if not np.isfinite(radius_ft) or radius_ft <= 0.0:
        raise ValueError(f'{sphere_filename}: RADIUS must be a positive number of feet')
    if sphere_frame is None:
        sphere_frame = read_sphere_frame(sphere_filename) or 'level'
    if sphere_frame not in SPHERE_FRAMES:
        raise ValueError('sphere_frame must be one of {}, not {!r}'.format(SPHERE_FRAMES, sphere_frame))
    if sphere_frame == 'flight_path':
        # The rim beyond the sphere's own hemisphere is missing: a .hem has no coverage
        # to mark a fill with.
        amplitude, measured = repitch_art_levels(fa.mask_missing_levels(amplitude), phi, theta,
                                                 float(np.ravel(angle)[0]))
        amplitude = np.where(measured, amplitude, -np.inf)
    levels_db = sphere_levels_on_norah2_grid(amplitude, phi, theta)
    return fa.write_norah2_levels(
        hem_filename, frequency, levels_db, r_ref_m=radius_ft * 0.3048,
        measurement_atmosphere=measurement_atmosphere,
        apply_absorption_deprop=apply_absorption_deprop,
        speed_knots=float(np.ravel(speed)[0]) if speed_knots is None else float(speed_knots),
        flight_path_angle_deg=(float(np.ravel(angle)[0]) if flight_path_angle_deg is None
                               else float(flight_path_angle_deg)),
        **norah2_options)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('sphere', help='AAM sphere netCDF (.nc)')
    parser.add_argument('-o', '--output', required=True, help='output NORAH2 hemisphere (.hem)')
    atmosphere = parser.add_argument_group('measurement atmosphere (required: the sphere file '
                                           'does not record it)')
    atmosphere.add_argument('--temperature', type=float, required=True, help='temperature, K (200 to 350)')
    atmosphere.add_argument('--pressure', type=float, required=True, help='pressure, kPa, not Pa (30 to 120)')
    atmosphere.add_argument('--humidity', type=float, required=True, help='relative humidity, percent (0 to 100)')
    parser.add_argument('--speed', type=float, default=None,
                        help="ACSPEED, kt (default: the sphere's SPEED, the ground speed for the "
                             '2017 build, which is what NORAH2 looks hemispheres up by)')
    parser.add_argument('--flight-path-angle', type=float, default=None,
                        help="GAMM, deg, negative in descent (default: the sphere's FLIGHT_PATH_ANGLE)")
    parser.add_argument('--sphere-frame', choices=SPHERE_FRAMES, default=None,
                        help="the frame the sphere is filed in; flight_path (AAM's) is pitched to level "
                             "by its FLIGHT_PATH_ANGLE (default: the file's sphere_frame, else level)")
    parser.add_argument('--title', default=None,
                        help='first line of the file (default: the output name without extension)')
    parser.add_argument('--test-point', default='', help="test point label after the title (default: none)")
    parser.add_argument('--rotor-rpm', type=float, default=100.0,
                        help='RmOmega, main rotor speed, percent of nominal (default %(default)g)')
    parser.add_argument('--minimum-level', type=float, default=-100.0,
                        help='levels at 60 m below this, dB, are written as -999 (default %(default)g)')
    parser.add_argument('--spreading-only', action='store_true',
                        help='the sphere was built without absorption depropagation: apply '
                             'spreading only (default: take out the absorption over RADIUS)')
    parser.add_argument('--absorption-band-method', choices=BAND_ABSORPTION_METHODS,
                        default='center',
                        help='how the sphere build formed the band absorption over RADIUS that is '
                             'taken out: center (ISO 9613-1 at the band center), arp5534 (SAE ARP '
                             '5534) or integrated (ISO 9613-1 over the band) (default %(default)s)')
    parser.add_argument('--overwrite', action='store_true', help='replace an existing output')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    atmosphere = Atmosphere(temperature=args.temperature, pressure=args.pressure,
                            relative_humidity=args.humidity)
    levels_db = sphere_to_norah2(
        args.sphere, args.output, measurement_atmosphere=atmosphere, speed_knots=args.speed,
        flight_path_angle_deg=args.flight_path_angle,
        apply_absorption_deprop=not args.spreading_only, sphere_frame=args.sphere_frame, title=args.title,
        test_point=args.test_point, rotor_rpm_percent=args.rotor_rpm,
        minimum_level_db=args.minimum_level, overwrite=args.overwrite,
        absorption_band_method=args.absorption_band_method)
    written = np.isfinite(levels_db)
    print(f'Wrote {args.output}: {int(written.sum())} of {written.size} band levels '
          f'(the rest -999)')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, TypeError, FileNotFoundError, FileExistsError) as error:
        print(f'sphere_to_norah2.py: error: {error}', file=sys.stderr)
        sys.exit(1)
