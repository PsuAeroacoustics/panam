#!/usr/bin/env python
"""Inspect a NICE-OPS database (.nod): list its conditions, map one, plot its envelope, check it."""
import argparse
import math
import sys

import nod_database as nd
from cli import colon_pair, save_or_show


def build_parser():
    parser = argparse.ArgumentParser(
        description='Inspect a NICE-OPS database (.nod): list its conditions, map one quantity of '
                    'a condition on a Lambert equal-area projection, plot the flight envelope it '
                    'covers, or check it against docs/file_formats.md.')
    parser.add_argument('--weight-coefficient', type=float, default=None,
                        help='C_W = W / (rho pi R^2 V_tip^2), 1, for the load factor n = C_T / C_W of a '
                             'database that is not fixed_load_factor (default: none; n is then unknown)')
    commands = parser.add_subparsers(dest='command', required=True)

    listing = commands.add_parser(
        'list', description='Print one row per condition: group, speed (kt), flight path angle (deg), '
                            'load factor, condition origin, fraction of coverage cells set and the '
                            'largest channel dBA (dB, at the sphere radius).',
        help='list the conditions')
    listing.add_argument('database', help='NICE-OPS database (.nod)')

    plot = commands.add_parser(
        'plot', description='Map one quantity of one condition on a Lambert equal-area projection '
                            'with ART grid lines.',
        help='map one condition')
    plot.add_argument('database', help='NICE-OPS database (.nod)')
    which = plot.add_mutually_exclusive_group(required=True)
    which.add_argument('-g', '--group', help='The condition by group name, e.g. sphere0')
    which.add_argument('-c', '--condition', type=colon_pair,
                       help='The condition by "speed:flight_path_angle", kt and deg (within 0.5 kt '
                            'and 0.05 deg), e.g. -c=80:-6')
    plot.add_argument('-n', '--load-factor', type=float, default=None,
                      help='Load factor, 1, that picks one of several conditions at the speed and angle '
                           'of --condition (default: none; the match must be unique)')
    plot.add_argument('-q', '--quantity', choices=nd.PLOT_QUANTITIES, default='dBA',
                      help='dBA (dB at the sphere radius), EAA (dB/1000 m), band (band level, dB, '
                           'needs --band) or coverage (1 where measured; the fraction of bands '
                           'without --band) (default dBA)')
    plot.add_argument('-b', '--band', type=float, default=None,
                      help='Band center, Hz, for band and coverage (default: none)')
    plot.add_argument('-r', '--range', type=colon_pair, default=None,
                      help='Color range "low:high" in the quantity\'s unit (default: the data range; '
                           '0:1 for coverage)')
    plot.add_argument('-o', '--output', default=None,
                      help='Image file (format from its extension); default: open a window')
    plot.add_argument('-p', '--plot', action='store_true', help='Also open the figure in a window')

    envelope = commands.add_parser(
        'envelope', description='Scatter the conditions in speed (kt) and flight path angle (deg), '
                                'colored by condition_origin.',
        help='plot the conditions in speed and flight path angle')
    envelope.add_argument('database', help='NICE-OPS database (.nod)')
    envelope.add_argument('-o', '--output', default=None,
                          help='Image file (format from its extension); default: open a window')
    envelope.add_argument('-p', '--plot', action='store_true', help='Also open the figure in a window')

    check = commands.add_parser(
        'check', description='Check the database against docs/file_formats.md and print each '
                             'violation; the exit status is 1 when there is one, else 0.',
        help='check the file contract')
    check.add_argument('database', help='NICE-OPS database (.nod)')
    return parser


def _number(value, spec):
    return '-' if value is None or (isinstance(value, float) and math.isnan(value)) else format(value, spec)


def format_listing(database):
    """The table that ``list`` prints, one row per condition."""
    row = '{:<12} {:>8} {:>8} {:>6} {:<27} {:>8} {:>8}'
    lines = [row.format('group', 'speed kt', 'fpa deg', 'n', 'origin', 'coverage', 'dBA max')]
    for c in database:
        s = nd.summarize_condition(c)
        lines.append(row.format(s['group'], _number(s['speed_kt'], '.1f'),
                                _number(s['flight_path_angle_deg'], '.2f'), _number(s['load_factor'], '.3g'),
                                s['condition_origin'] or '-', _number(s['coverage_fraction'], '.3f'),
                                _number(s['dBA_max'], '.1f')))
    return '\n'.join(lines)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == 'check':
        violations = nd.validate_nod(args.database)
        for violation in violations:
            print(violation)
        if not violations:
            print(f'{args.database}: no violations')
        return int(bool(violations))
    database = nd.load_nod(args.database, weight_coefficient=args.weight_coefficient)
    if args.command == 'list':
        print(format_listing(database))
        return 0
    if args.command == 'envelope':
        fig = nd.plot_envelope(database)
    else:
        try:
            if args.group is not None:
                matches = [c for c in database if c.name == args.group]
                if not matches:
                    raise ValueError(f'no group {args.group}')
                condition = matches[0]
            else:
                condition = nd.select_condition(database, *args.condition, load_factor=args.load_factor)
            fig = nd.plot_condition(condition, args.quantity, band_hz=args.band, level_range=args.range)
        except ValueError as error:
            raise SystemExit(f'nod_tool.py plot: {error}')
    save_or_show(fig, args.output, args.plot, bbox_inches='tight')
    return 0


if __name__ == '__main__':
    sys.exit(main())
