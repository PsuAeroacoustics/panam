"""Build a format 2 NICE-OPS database of pure tones from PSU-WOPWOP's per-rotor PT spheres.

Usage:  python build_pt_database.py SPHERE [SPHERE ...] -o DATABASE.nod --weight-newtons W
            [--sync-groups shared_rate|none] [--main-rotor TAG]

One PT sphere per rotor (its source_tag) per flight condition, in the flight-path frame. The spheres carry no
weight, so the vehicle's weight (N) is required for the thrust-coefficient axis. Rotors predicted at one shaft rate
form one coherent group unless --sync-groups none. See pt_database.build_pt_database and docs/file_formats.md,
"Format 2: pure-tone sources".
"""
import argparse
import sys

from panam import pt_database


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('spheres', nargs='+', help='PSU-WOPWOP PT sphere files (.nc), one per rotor and condition')
    parser.add_argument('-o', '--output', required=True, help='output NICE-OPS database (.nod)')
    parser.add_argument('--weight-newtons', type=float, required=True, help="the vehicle's weight, N")
    parser.add_argument('--sync-groups', choices=('shared_rate', 'none'), default='shared_rate',
                        help='shared_rate (default): rotors on one shaft rate add coherently; none: all free-running')
    parser.add_argument('--main-rotor', default=None, help="the tag whose rotor sets the advance ratio and C_T")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    try:
        pt_database.build_pt_database(args.spheres, args.output, weight_newtons=args.weight_newtons,
                                      sync_groups=None if args.sync_groups == 'none' else 'shared_rate',
                                      main_rotor=args.main_rotor)
    except ValueError as error:
        print(f'build_pt_database: {error}', file=sys.stderr)
        return 1
    print(f'wrote {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
