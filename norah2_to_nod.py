"""Build a NICE-OPS sphere database from NORAH2 (HELENA, ECAC Doc 32) .hem hemispheres.

Usage:  python norah2_to_nod.py HEM [HEM ...] -o DATABASE.nod
            --vehicle VEHICLE.json --speed {ias-to-tas,ground-speed}

A .hem carries no rotor or weight, so the vehicle is required: a NICE-OPS
vehicle JSON or a panam vehicle.cfg (--vehicle), or all of --tip-speed,
--rotor-radius and --weight-kg.  What ACSPEED holds must be stated (--speed):
the format labels it indicated airspeed, NORAH2 looks it up as ground speed,
and panam's own .hem exports write ground speed into it.  See
flight_acoustics.build_database_from_norah2 and docs/norah2_import.md.
"""
import argparse
import os
import sys

from panam_acoustics.atmosphere import Atmosphere

import flight_acoustics

SPEED_CHOICES = {'ias-to-tas': 'ias_to_tas', 'ground-speed': 'ground_speed'}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('hem_files', nargs='+', help='.hem hemispheres, one per flight condition')
    parser.add_argument('-o', '--output', required=True, help='output NICE-OPS database (.nod)')
    parser.add_argument('--speed', required=True, choices=sorted(SPEED_CHOICES),
                        help='what ACSPEED is: ias-to-tas reads it as indicated airspeed and '
                             'converts it to true airspeed (an air-referenced database); '
                             'ground-speed takes it as the ground speed, as NORAH2 does '
                             '(a ground-referenced database)')
    parser.add_argument('--tas-density', type=float, default=None,
                        help='ias-to-tas: air density for every file, kg/m^3, in place of each '
                             "file's measurement atmosphere (Tm, Pm, RHm)")
    vehicle = parser.add_argument_group('vehicle (required: --vehicle, or all three of '
                                        '--tip-speed, --rotor-radius, --weight-kg)')
    vehicle.add_argument('--vehicle', help='NICE-OPS vehicle JSON or panam vehicle.cfg')
    vehicle.add_argument('--tip-speed', type=float, help='main rotor tip speed, m/s')
    vehicle.add_argument('--rotor-radius', type=float, help='main rotor radius, m')
    vehicle.add_argument('--weight-kg', type=float, help='vehicle mass, kg')
    vehicle.add_argument('--thrust-density', type=float, default=None,
                         help='density the thrust coefficient is formed with, kg/m^3 (default: '
                              "the vehicle file's, or 1.225 without one)")
    parser.add_argument('--radius', type=float, default=flight_acoustics.DATABASE_SPHERE_RADIUS_M,
                        help='sphere radius, m (default %(default)g)')
    parser.add_argument('--build-temperature', type=float, default=293.15,
                        help='atmosphere of the broadband EAA: temperature, K (default %(default)g)')
    parser.add_argument('--build-pressure', type=float, default=101.325,
                        help='... pressure, kPa (default %(default)g)')
    parser.add_argument('--build-humidity', type=float, default=20.0,
                        help='... relative humidity, percent (default %(default)g)')
    parser.add_argument('--accept-ground-included', action='store_true',
                        help='convert FREEFIELD = 0 hemispheres (ground reflection included) as '
                             'free field, with a warning, instead of refusing them')
    parser.add_argument('--mirror-rotor', action='store_true',
                        help='reverse the azimuth (phi -> -phi), for a type whose main rotor turns '
                             "the other way from the hemispheres' (NORAH2's bracketed types)")
    parser.add_argument('--fill-empty', action='store_true',
                        help="fill each NOVALUE cell from its nearest filled direction, as NORAH2's "
                             'own predictor does (coverage stays 0); without it NICE-OPS reads them '
                             'as unmeasured')
    parser.add_argument('--no-spectrum', action='store_true',
                        help='write dBA and EAA only, without the band levels')
    parser.add_argument('--overwrite', action='store_true', help='replace an existing output')
    args = parser.parse_args(argv)

    explicit = (args.tip_speed, args.rotor_radius, args.weight_kg)
    if args.vehicle is not None and any(v is not None for v in explicit):
        parser.error('give --vehicle or --tip-speed/--rotor-radius/--weight-kg, not both')
    if args.vehicle is None and any(v is None for v in explicit):
        parser.error('the vehicle is required: a .hem carries no rotor or weight; give --vehicle, '
                     'or all of --tip-speed, --rotor-radius and --weight-kg')
    if args.tas_density is not None and args.speed != 'ias-to-tas':
        parser.error('--tas-density applies to --speed ias-to-tas only')
    if os.path.exists(args.output) and not args.overwrite:
        parser.error(f'{args.output} exists; pass --overwrite to replace it')
    return args


def main(argv=None):
    args = parse_arguments(argv)
    if args.vehicle is not None:
        vehicle = flight_acoustics.read_vehicle_rotor_data(args.vehicle)
    else:
        vehicle = dict(main_rotor_radius_m=args.rotor_radius, main_rotor_tip_speed_mps=args.tip_speed,
                       vehicle_weight_newtons=flight_acoustics.STANDARD_GRAVITY * args.weight_kg,
                       air_density_kg_m3=flight_acoustics.STANDARD_SEA_LEVEL_DENSITY)
    thrust_density = args.thrust_density if args.thrust_density is not None else vehicle['air_density_kg_m3']
    conditions = flight_acoustics.build_database_from_norah2(
        sorted(args.hem_files), args.output,
        main_rotor_tip_speed_mps=vehicle['main_rotor_tip_speed_mps'],
        main_rotor_radius_m=vehicle['main_rotor_radius_m'],
        vehicle_weight_newtons=vehicle['vehicle_weight_newtons'],
        speed_mapping=SPEED_CHOICES[args.speed],
        thrust_air_density=thrust_density,
        tas_air_density=args.tas_density,
        radius_m=args.radius,
        atmosphere=Atmosphere(temperature=args.build_temperature, pressure=args.build_pressure,
                              relative_humidity=args.build_humidity),
        accept_ground_included=args.accept_ground_included,
        mirror_rotor=args.mirror_rotor,
        fill_empty=args.fill_empty,
        store_spectrum=not args.no_spectrum,
        overwrite=args.overwrite)
    for condition in conditions:
        print('{source}: {speed_knots:.2f} kt, {flight_path_angle_deg:+.2f} deg, mu {advance_ratio:.4f}'
              .format(**condition))
    print(f'Wrote {len(conditions)} condition(s) to {args.output}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(f'norah2_to_nod.py: error: {error}', file=sys.stderr)
        sys.exit(1)
