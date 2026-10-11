#!/usr/bin/env python
"""Energy-average repeat-run spheres, or compare two spheres band by band."""
import argparse

import numpy as np

import sphere_statistics as ss
from cli import colon_pair, save_or_show

MISSING_HELP = ("How a sphere file's -999 cell reads: no_data leaves the run out of the cell, "
                "no_energy counts it as zero power (default no_data)")


def build_parser():
    parser = argparse.ArgumentParser(
        description='Energy-average AAM spheres of repeat runs into one sphere, or print the '
                    'band-by-band level difference between two spheres and map it.')
    commands = parser.add_subparsers(dest='command', required=True)

    average = commands.add_parser(
        'average', description='Energy-average spheres on one grid, cell by cell, over the runs '
                               'that cover each cell, and write the result with N_RUNS and AMPLITUDE_SD_DB.',
        help='energy-average spheres of repeat runs')
    average.add_argument('output', help='Sphere file (.nc) to write')
    average.add_argument('inputs', nargs='+', help='Sphere files (.nc) to average')
    average.add_argument('-m', '--min-runs', type=int, default=1,
                         help='Fewest covering runs a cell needs to get a level (default 1)')
    average.add_argument('--missing', choices=ss.MISSING_MEANINGS, default='no_data', help=MISSING_HELP)
    average.add_argument('--speed-tolerance', type=float, default=0.0,
                         help='Largest spread (max - min) of SPEED allowed between the inputs, kt (default 0)')
    average.add_argument('--fpa-tolerance', type=float, default=float('inf'),
                         help='Largest spread (max - min) of FLIGHT_PATH_ANGLE allowed between the '
                              'inputs, deg (default inf: not checked)')
    average.add_argument('-t', '--title', default=None,
                         help='Title of the written file (default names the number of inputs)')

    compare = commands.add_parser(
        'compare', description='Print per-band statistics of the level difference B - A over the '
                               'cells finite in both spheres, and optionally map it.',
        help='compare two spheres band by band')
    compare.add_argument('a', help='Sphere file A (.nc), the reference')
    compare.add_argument('b', help='Sphere file B (.nc), compared with A')
    compare.add_argument('-f', '--frequency', type=colon_pair, default=None,
                         help='Compare only the bands whose center lies in "low:high", Hz (default: every band)')
    compare.add_argument('-b', '--band', type=float, default=None,
                         help='Map this band center, Hz (default: the overall level of the compared bands)')
    compare.add_argument('-r', '--range', type=colon_pair, default=None,
                         help='Map color range "low:high", dB (default: symmetric about 0 over the data)')
    compare.add_argument('-o', '--output', default=None,
                         help='Image file for the map (format from its extension); without it and -p, no map')
    compare.add_argument('-p', '--plot', action='store_true', help='Open the map in a window')
    compare.add_argument('--missing', choices=ss.MISSING_MEANINGS, default='no_data', help=MISSING_HELP)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == 'average':
        average = ss.average_spheres(args.inputs, min_runs=args.min_runs, missing=args.missing,
                                     speed_tolerance_knots=args.speed_tolerance,
                                     flight_path_angle_tolerance_deg=args.fpa_tolerance)
        ss.write_averaged_sphere(args.output, average, title=args.title)
        print('Wrote {} from {} spheres; {} of {} cells have a level'.format(
            args.output, len(args.inputs), int(np.sum(~np.isnan(average['levels_db']))),
            average['levels_db'].size))
        return
    a = ss.read_sphere(args.a, args.missing)
    b = ss.read_sphere(args.b, args.missing)
    bands = None
    if args.frequency is not None:
        frequency = a['frequency_hz']
        bands = frequency[(frequency >= args.frequency[0]) & (frequency <= args.frequency[1])]
        if bands.size == 0:
            raise SystemExit('no band center lies in {}:{} Hz'.format(*args.frequency))
    result = ss.compare_spheres(a, b, bands=bands)
    print(ss.format_comparison(result))
    if args.output or args.plot:
        fig = ss.plot_sphere_difference(result, band=args.band, level_range=args.range)
        save_or_show(fig, args.output, args.plot, bbox_inches='tight', facecolor='none')


if __name__ == '__main__':
    main()
