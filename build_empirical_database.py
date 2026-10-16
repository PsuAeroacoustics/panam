"""Build a NICEOPS sphere database from a directory of AAM sphere files.

Usage:  python build_empirical_database.py SPHERE_DIRECTORY DATABASE_FILE [options]

SPHERE_DIRECTORY holds the .nc spheres and a vehicle.cfg (rotors,
atmosphere, weight and drag; see docs/file_formats.md).  Every option maps to
a keyword of flight_acoustics.build_empirical_database and defaults to that
function's default.  --recipe FILE.toml takes the [database] section of a build
recipe (build_recipe.py); options given on the command line override it.
"""
import argparse
import os
import sys

from panam_acoustics.atmosphere import Atmosphere, BAND_ABSORPTION_METHODS

import cli
import flight_acoustics

EPILOG = """\
hover_correction, a callable that adds dB to the synthesized hover per direction
and band, has no option: give it in a recipe's [database] section as
"module:callable" (--recipe; see build_recipe.py), or call
flight_acoustics.build_empirical_database from Python.  Options given on the
command line override the recipe.
"""

#: Destinations whose build_empirical_database keyword has another name.  Every other
#: destination is its keyword.  A recipe value gives way when its destination is given.
CLI_KEYWORDS = dict(
    manifest=('spheres',), skip_flagged=('spheres',), fixed_load_factor=('load_factors',),
    extended_fpa=('extended_flight_path_angles',), doppler_removed=('doppler_shift_removed',),
    frequency_range=('infreqs',), build_temperature=('atmosphere',), build_pressure=('atmosphere',),
    build_humidity=('atmosphere',), no_spectrum=('store_spectrum',),
    no_clamp_empty=('clamp_empty_directions',))

#: The --build-* destinations and the Atmosphere field each sets (K, kPa, percent).
ATMOSPHERE_FIELDS = dict(build_temperature='temperature', build_pressure='pressure',
                         build_humidity='relative_humidity')


def _float_list(value):
    """argparse type for a comma-separated list of floats."""
    try:
        values = [float(item) for item in value.split(',') if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error))
    if not values:
        raise argparse.ArgumentTypeError('at least one load factor is required')
    return values


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sphere_directory', help='directory of .nc spheres and vehicle.cfg')
    parser.add_argument('database_file', help='output NICEOPS database (.nc/.nod)')
    parser.add_argument('--recipe', default=None, metavar='FILE.toml',
                        help="take the settings of this recipe's [database] section; options given "
                             'here override it (default: none)')

    selection = parser.add_argument_group('sphere selection (default: every .nc in SPHERE_DIRECTORY)')
    selection.add_argument('--sphere', dest='spheres', action='append', default=None, metavar='FILE',
                           help='use this sphere file (relative to SPHERE_DIRECTORY unless absolute); '
                                'repeat for each one')
    selection.add_argument('--manifest', default=None,
                           help="use the spheres a noise_abatement_2017.py build manifest lists as built "
                                '(their file names, taken in SPHERE_DIRECTORY)')
    selection.add_argument('--skip-flagged', action='store_true',
                           help='with --manifest: leave out runs whose track check raised a flag')
    selection.add_argument('--exclude', action='append', default=None, metavar='PATTERN',
                           help='leave out spheres whose file name matches this name or shell pattern '
                                '(e.g. "Be407285236.nc" or "*_hover*.nc"); repeat for each one')
    selection.add_argument('--merge-repeats', action='store_true',
                           help='energy-average spheres at the same speed and flight path angle into one '
                                'condition (default: refuse them, since NICEOPS takes one sphere per '
                                'condition)')
    selection.add_argument('--merge-speed-tolerance', type=float, default=0.0, metavar='KT',
                           help='with --merge-repeats: also merge spheres at one flight path angle whose '
                                'speeds lie within this of the slowest, kt (default %(default)g)')

    load = parser.add_mutually_exclusive_group()
    load.add_argument('--load-factors', type=_float_list, default=None, metavar='N1,N2,...',
                      help='load factors to write each condition at, g, comma-separated '
                           '(default 0.7,1.1,1.5,1.9,2.3)')
    load.add_argument('--fixed-load-factor', action='store_true',
                      help='write each condition once, at 1 g, and let NICEOPS scale it '
                           '(load_factors=None)')

    parser.add_argument('--extended-fpa', type=cli.colon_pair, default=None, metavar='LO:HI',
                        help='re-emit every near-level condition at these two flight path angles, deg '
                             '(e.g. --extended-fpa=-24:35; default none: the hover is written at -12, 0 and +12)')
    parser.add_argument('--level-flight-tolerance', type=float,
                        default=flight_acoustics.LEVEL_FLIGHT_TOLERANCE,
                        help='flight path angle within which a sphere counts as level flight, deg '
                             '(default %(default)g)')
    parser.add_argument('--hover-source', default=None,
                        help='path of the sphere file the hover is synthesized from, relative to the '
                             'current directory, not SPHERE_DIRECTORY (default: the slowest sphere '
                             'within the level-flight tolerance)')
    parser.add_argument('--speed-reference', choices=flight_acoustics.SPEED_REFERENCES, default='ground',
                        help='what the advance ratio and flight path angle are taken against '
                             '(default %(default)s)')
    parser.add_argument('--azimuth-reference', choices=flight_acoustics.AZIMUTH_REFERENCES, default=None,
                        help="what the spheres' azimuth is measured from, for spheres that carry no "
                             "azimuth_reference (default: the spheres' own, else track)")
    parser.add_argument('--doppler-removed', type=int, choices=(0, 1), default=None,
                        help='DOPPLER_SHIFT_REMOVED for spheres that carry no flag: 0 received '
                             "frequencies, 1 emitted (default: the spheres' own, else 0 with a warning)")
    parser.add_argument('--distance', type=float, default=1000.0,
                        help='propagation distance of the broadband EAA, m (default %(default)g)')
    parser.add_argument('--frequency-range', type=cli.colon_pair, default=None, metavar='LO:HI',
                        help='keep only bands from LO to HI, Hz (default: every band)')
    cli.add_build_atmosphere_options(parser)
    parser.add_argument('--absorption-band-method', choices=BAND_ABSORPTION_METHODS,
                        default='center',
                        help='how band absorption over --distance enters the EAA: center (ISO 9613-1 '
                             'at the band center), arp5534 (SAE ARP 5534) or integrated (ISO 9613-1 '
                             'integrated over the band, flat band levels) (default %(default)s)')
    parser.add_argument('--no-spectrum', action='store_true',
                        help='write dBA and EAA only, without the band levels')
    parser.add_argument('--no-clamp-empty', action='store_true',
                        help='leave EAA NaN where ambient gating emptied a direction, instead of 0')
    parser.add_argument('--overwrite', action='store_true', help='replace an existing output')
    return parser


def parse_arguments(argv=None, parser=None):
    parser = parser or make_parser()
    args = parser.parse_args(argv)

    if args.skip_flagged and args.manifest is None:
        parser.error('--skip-flagged applies to --manifest only')
    # With a recipe, merge_repeats may come from its [database]: main checks the merged options.
    if args.merge_speed_tolerance != 0.0 and not args.merge_repeats and args.recipe is None:
        parser.error('--merge-speed-tolerance applies to --merge-repeats only')
    if not args.merge_speed_tolerance >= 0.0:
        parser.error('--merge-speed-tolerance must be at least 0 kt')
    if args.manifest is not None and args.spheres is not None:
        parser.error('give --manifest or --sphere, not both')
    if os.path.exists(args.database_file) and not args.overwrite:
        parser.error(f'{args.database_file} exists; pass --overwrite to replace it')
    return args


def build_options(args):
    """The keyword arguments of flight_acoustics.build_empirical_database that ``args`` give.

    load_factors is left out unless --load-factors or --fixed-load-factor is given, so
    that the function's own default array is used unchanged.
    """
    options = dict(
        distance=args.distance,
        infreqs=args.frequency_range,
        atmosphere=cli.build_atmosphere(args),
        extended_flight_path_angles=args.extended_fpa,
        level_flight_tolerance=args.level_flight_tolerance,
        store_spectrum=not args.no_spectrum,
        clamp_empty_directions=not args.no_clamp_empty,
        hover_source=args.hover_source,
        speed_reference=args.speed_reference,
        doppler_shift_removed=args.doppler_removed,
        azimuth_reference=args.azimuth_reference,
        spheres=args.spheres,
        exclude=args.exclude,
        absorption_band_method=args.absorption_band_method,
        merge_repeats=args.merge_repeats,
        merge_speed_tolerance=args.merge_speed_tolerance,
    )
    if args.fixed_load_factor:
        options['load_factors'] = None
    elif args.load_factors is not None:
        options['load_factors'] = args.load_factors
    if args.manifest is not None:
        import noise_abatement_2017      # only here: it is the 2017 build's module
        options['spheres'] = noise_abatement_2017.select_from_manifest(
            args.manifest, skip_flagged=args.skip_flagged)
    return options


def recipe_options(args, parser, argv=None):
    """build_options(args) with the recipe's [database] settings where no option was given.

    A --build-temperature, --build-pressure or --build-humidity option replaces only its
    own field of a recipe's ``atmosphere``; the recipe's other fields stay.
    """
    options = build_options(args)
    if args.recipe is None:
        return options
    import build_recipe
    recipe = build_recipe.load_recipe(args.recipe).database_options()
    merged, _ = build_recipe.apply(parser, argv, recipe, options, CLI_KEYWORDS)
    explicit = build_recipe.explicit_destinations(parser, argv)
    given = {field: getattr(args, destination) for destination, field in ATMOSPHERE_FIELDS.items()
             if destination in explicit}
    if 'atmosphere' in recipe and given:
        fields = {field: getattr(recipe['atmosphere'], field) for field in ATMOSPHERE_FIELDS.values()}
        fields.update(given)
        merged['atmosphere'] = Atmosphere(**fields)
    return merged


def main(argv=None):
    parser = make_parser()
    args = parse_arguments(argv, parser)
    options = recipe_options(args, parser, argv)
    if options.get('merge_speed_tolerance', 0.0) != 0.0 and not options.get('merge_repeats', False):
        parser.error('--merge-speed-tolerance (or merge_speed_tolerance) applies to merge_repeats only')
    flight_acoustics.build_empirical_database(args.sphere_directory, args.database_file, **options)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, LookupError, OSError) as error:
        print(f'build_empirical_database.py: error: {error}', file=sys.stderr)
        sys.exit(1)
