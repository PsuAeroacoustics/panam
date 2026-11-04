"""Build a format 2 NICEOPS database of pure tones from PSU-WOPWOP's per-rotor PT spheres.

Usage:  python build_pt_database.py SPHERE [SPHERE ...] -o DATABASE.nod --weight-newtons W
            [--sync-groups shared_rate|none] [--main-rotor TAG] [--speed-reference air|ground]
            [--speed-tolerance-kt KT]
            [--condition-axes helicopter|multirotor --rotor-count N [--hover-thrust-coefficient CT]
             [--sound-speed C]]

One PT sphere per rotor (its source_tag) per flight condition, in the flight-path frame. The spheres carry no
weight, so the vehicle's weight (N) is required for the thrust-coefficient axis. Rotors predicted at one shaft rate
form one coherent group unless --sync-groups none. A multirotor database (--condition-axes multirotor) is indexed
by momentum similarity and needs the vehicle's rotor count. See pt_database.build_pt_database and docs/file_formats.md,
"Format 2: pure-tone sources".
"""
import argparse
import sys

from panam import pt_database


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('spheres', nargs='+', help='PSU-WOPWOP PT sphere files (.nc), one per rotor and condition')
    parser.add_argument('-o', '--output', required=True, help='output NICEOPS database (.nod); replaced if it exists')
    parser.add_argument('--weight-newtons', type=float, required=True, help="the vehicle's weight, N (required)")
    parser.add_argument('--sync-groups', choices=('shared_rate', 'none'), default='shared_rate',
                        help='shared_rate (default): rotors on one shaft rate add coherently; none: all free-running')
    parser.add_argument('--main-rotor', default=None,
                        help='the source tag whose rotor sets the advance ratio and C_T (default: the largest rotor '
                             'radius, the first tag on a tie)')
    parser.add_argument('--speed-reference', choices=pt_database.SPEED_REFERENCES, default=None,
                        help='air or ground, for spheres that record none (default: what the spheres record, else '
                             'air); refused if the spheres record another')
    parser.add_argument('--speed-tolerance-kt', type=float, default=None,
                        help='kt: spheres whose speeds lie within this (and flight path angles within 1e-3 deg) are '
                             'one condition, at their mean (default: exact, to 1e-6 kt)')
    parser.add_argument('--condition-axes', choices=('helicopter', 'multirotor'), default='helicopter',
                        help='helicopter (advance ratio, flight path angle, C_T; default) or multirotor (momentum '
                             'similarity: in-plane and axial speed over v_h, hover tip Mach number)')
    parser.add_argument('--rotor-count', type=int, default=None,
                        help="the vehicle's number of rotors (multirotor: required; default none)")
    parser.add_argument('--hover-thrust-coefficient', type=float, default=None,
                        help='C_Th per rotor, dimensionless (multirotor; default from the hover condition)')
    parser.add_argument('--sound-speed', type=float, default=None,
                        help='m/s, for the hover tip Mach number where a sphere records none (multirotor; default '
                             'none: such a sphere is refused)')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    try:
        pt_database.build_pt_database(args.spheres, args.output, weight_newtons=args.weight_newtons,
                                      sync_groups=None if args.sync_groups == 'none' else 'shared_rate',
                                      main_rotor=args.main_rotor, speed_reference=args.speed_reference,
                                      condition_axes=args.condition_axes, rotor_count=args.rotor_count,
                                      hover_thrust_coefficient=args.hover_thrust_coefficient,
                                      sound_speed_mps=args.sound_speed,
                                      speed_tolerance_kt=args.speed_tolerance_kt)
    except ValueError as error:
        print(f'build_pt_database: {error}', file=sys.stderr)
        return 1
    print(f'wrote {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
