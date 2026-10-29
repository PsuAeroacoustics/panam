#!/usr/bin/env python3
"""Build source spheres from the 2017 FAA/NASA Noise Abatement flight test.

Usage:  python tools/noise_abatement_2017.py AIRCRAFT OUTPUT_DIRECTORY [options]

Every option maps to a keyword of :func:`panam.noise_abatement_2017.build_all` and
:func:`panam.noise_abatement_2017.build_sphere`.  ``--check-tracks`` checks the tracks
without building, ``--plate-cache list|clear`` inspects or empties the plate-table
cache, and ``--recipe FILE.toml`` takes the ``[spheres]`` section of a build recipe
(:mod:`panam.build_recipe`); options given on the command line override it.  The
dataset root is the ``noise_abatement_2017`` entry of local_paths, or ``--root``.
See ``docs/database_build.md``.
"""
import argparse
import logging
import os
from collections import defaultdict

from panam import build_recipe
from panam import flight_acoustics as fa
from panam import noise_abatement_2017 as na


def _interpolation_setting(value):
    """argparse type for one adaptive gridding setting, KEY=VALUE (see fa.ADAPTIVE_INTERPOLATION)."""
    key, separator, text = value.partition('=')
    key = key.strip()
    if not separator or key not in fa.ADAPTIVE_INTERPOLATION:
        raise argparse.ArgumentTypeError('expected KEY=VALUE with KEY one of {}, got {!r}'.format(
            ', '.join(sorted(fa.ADAPTIVE_INTERPOLATION)), value))
    default = fa.ADAPTIVE_INTERPOLATION[key]
    text = text.strip()
    if isinstance(default, bool):
        if text.lower() not in ('true', 'false'):
            raise argparse.ArgumentTypeError('{} takes true or false, not {!r}'.format(key, text))
        return key, text.lower() == 'true'
    try:
        return key, type(default)(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError('{}: {}'.format(key, error))


#: Command-line destinations whose build_all/build_sphere keyword has another name, or that
#: set more than one keyword.  Every other destination is its keyword.  A recipe value of
#: one of these keywords gives way when the destination is given on the command line.
CLI_KEYWORDS = dict(
    include_maneuvers=('steady_only',), manifest=('manifest_path',), no_prefetch=('prefetch',),
    no_ambient_gate=('gate_ambient',), no_track_check=('track_check',),
    ray_model=('ray_models', 'ray_profiles'), ray_profiles=('ray_profiles', 'ray_models'),
    tone_split=('tone_split', 'tone_aware'), adaptive_interpolation=('interpolation',),
    interpolation=('interpolation',), no_absorption_deprop=('apply_absorption_deprop',),
    no_overwrite=('overwrite',))


def make_parser():
    """The command line's parser: every build_all and build_sphere option it exposes."""
    parser = argparse.ArgumentParser(
        description='Rebuild 2017 Noise Abatement source spheres with ambient gating.',
        epilog='Python-only build_sphere keywords (atmosphere, ground, absorption_profile, '
               'samples_path, a ray model other than the profiles or the wind) have no option. '
               '--recipe reads settings from a TOML recipe (see panam/build_recipe.py); an option given '
               'on the command line overrides the recipe.')
    parser.add_argument('aircraft', nargs='?', default=None,
                        help='dataset directory name under --root: AS350B3, B206L3, B407, '
                             'EC130B4, R44 or R66 (2017), S76D (2019) (not needed with --plate-cache)')
    parser.add_argument('output_directory', nargs='?', default=None,
                        help='where the spheres go (not needed with --check-tracks or --plate-cache)')
    parser.add_argument('--recipe', default=None, metavar='FILE.toml',
                        help="build with the [spheres] settings of this recipe; options given here "
                             'override it (default: none)')
    parser.add_argument('--hover-source', action='store_true',
                        help="with --recipe: build with the recipe's [hover_source] settings over its "
                             '[spheres] ones, the rebuild the database synthesizes its hover from; '
                             'needs --runs (default off)')
    parser.add_argument('--plate-table-directory', default=None,
                        help='where computed ground-plate tables are cached (default: '
                             '~/.cache/panam/plate_tables)')
    parser.add_argument('--plate-cache', choices=('list', 'clear'), default=None,
                        help='do not build: list the cached plate tables (size in MB and whether '
                             'each is current), or delete them; default: neither')
    parser.add_argument('--check-tracks', action='store_true',
                        help='do not build: check every steady run\'s track as the build would '
                             '(heights above the ground boards, altitude consistency, decelerations, '
                             'level-offs, card flight path angle), print what is refused and flagged, '
                             'and write the table to --manifest if given; needs only the tracks')
    parser.add_argument('--hovers', action='store_true',
                        help='with --check-tracks, check the hover runs too (their whole record)')
    parser.add_argument('--azimuth-reference', choices=('track', 'heading'), default=None,
                        help="'track' (the default for flight) measures each sphere's azimuth from the ground track; "
                             "'heading' from the tracked INS heading, with Doppler and depropagation "
                             'still on the ground velocity.  Written to every sphere and so to the database')
    parser.add_argument('--sphere-frame', choices=('level', 'flight_path'), default='level',
                        help="the frame each sphere is filed in: level (default; PANAM's and the "
                             "database's) or flight_path (AAM's, pitched by the run's flight path angle, "
                             'for a deck flown at AOA 0)')
    parser.add_argument('--nose-from-heading', action='store_true',
                        help='orient each sphere by the tracked heading, for hovers: no Doppler or '
                             'convective term, and the whole record instead of a steady window '
                             '(default off)')
    parser.add_argument('--winds', default=None,
                        help='CSV of the wind at the aircraft per run (run, units, source, and east/north '
                             'or speed/direction_from_deg; see read_winds), recorded in each sphere; '
                             'nothing is assumed about its units')
    parser.add_argument('--ray-model', choices=('straight', 'wind'), default='straight',
                        help="'straight' (default): straight lines in still air; 'wind': straight rays "
                             "carried by each run's wind from --winds (required), uniform over the path; "
                             'a run with no wind there fails')
    parser.add_argument('--ray-profiles', default=None, metavar='DIRECTORY',
                        help="refracted rays through each run's NICEOPS atmosphere CSV, <run>.csv in "
                             'this directory (heights in ft), traced by niceops_ray_geometry from '
                             'local_paths; straight lines for a run without one (default: none)')
    parser.add_argument('--air-relative-doppler', action='store_true',
                        help='with --remove-doppler and --ray-model wind, take the Doppler factor in '
                             'the moving air (default off)')
    parser.add_argument('--no-track-check', action='store_true',
                        help='build runs whose track fails the sanity check (below or within '
                             '--min-height-above-array-ft of the ground boards, z not following altitude)')
    parser.add_argument('--min-height-above-array-ft', type=float, default=na.MIN_HEIGHT_ABOVE_ARRAY_FT,
                        help='refuse a run whose track comes closer than this to the ground boards '
                             '(default %(default)s)')
    parser.add_argument('--root', default=None,
                        help='dataset root (default: the noise_abatement_2017 entry of local_paths)')
    parser.add_argument('--track-datum', choices=na.TRACK_DATUMS, default='track',
                        help="what the tracks' z is measured from: track (default), the file's z; reference, "
                             'altitude less the reference list ref_elips_ft, the microphones\' datum (2019)')
    parser.add_argument('--runs', nargs='*', default=None,
                        help='specific run ids (default: every steady, non-maneuvering run)')
    parser.add_argument('--include-maneuvers', action='store_true',
                        help='also build spheres from turns and accelerating passes; off '
                             'by default, since they smear directivity across azimuth in a '
                             'way depropagation does not correct for')
    parser.add_argument('--reference-directory', default=None,
                        help='legacy spheres, used only for their PHI/THETA/FREQUENCY grids')
    parser.add_argument('--sphere-prefix', default=None,
                        help="sphere file name prefix (default: the aircraft's legacy prefix, e.g. Be407)")
    parser.add_argument('--manifest', default=None,
                        help="CSV recording each run's inputs and outcome (with --check-tracks: "
                             'the track-check table); default: not written')
    parser.add_argument('--norah2-directory', default=None,
                        help='also write each sphere as a NORAH2 .hem file here, with the '
                             'triangulation file NORAH2 needs to interpolate between them')
    parser.add_argument('--no-overwrite', action='store_true',
                        help='fail on a run whose sphere exists instead of replacing it (default: replace)')
    parser.add_argument('--board-correction', choices=('plate_bem', 'flat'), default='plate_bem',
                        help="'plate_bem' (default) divides out the ground plate's modeled "
                             "response per band and frame; 'flat' is the old constant -6 dB")
    parser.add_argument('--third-octave-method', choices=('fft', 'filter_bank'), default='fft',
                        help="'filter_bank' forms bands with a true one-third octave filter "
                             "bank, as an analyzer does; it differs from the default FFT band "
                             "sum only below ~100 Hz, between strong rotor tones")
    parser.add_argument('--band-snr-gate-db', type=float, default=10.0,
                        help="zero frequency bins less than this many dB above the mic's ambient "
                             '(default %(default)g)')
    parser.add_argument('--max-absorption-correction-db', type=float, default=30.0,
                        help='discard bins needing more absorption correction than this, dB; '
                             '30 dB is what keeps source spectra rolling off physically '
                             '(default %(default)g)')
    parser.add_argument('--max-response-correction-db', type=float, default=None,
                        help='discard bins whose receiver (ground-plate) correction exceeds this, dB '
                             '(default: no cap)')
    parser.add_argument('--no-absorption-deprop', action='store_true',
                        help='do not undo atmospheric absorption (default: undo it)')
    parser.add_argument('--absorption-band-method', choices=fa.BAND_ABSORPTION_METHODS,
                        default='center',
                        help="band absorption for --third-octave-method filter_bank: 'center' (ISO "
                             "9613-1 at the band center), 'arp5534' or 'integrated' (default %(default)s)")
    parser.add_argument('--speed-of-sound-ft-s', type=float, default=None,
                        help="speed of sound, ft/s (default: from the run's atmosphere)")
    parser.add_argument('--r-ref-ft', type=float, default=na.DEFAULT_R_REF_FT,
                        help='radius the sphere is referred to, ft (default %(default)g)')
    parser.add_argument('--window-time', type=float, default=0.5,
                        help='analysis window length, s (default %(default)g)')
    parser.add_argument('--window-overlap', type=float, default=0.5,
                        help='fraction of a window the next one overlaps (default %(default)g)')
    parser.add_argument('--point-stride', type=int, default=10,
                        help='depropagate every Nth sample of the steady track (default %(default)s, '
                             "the 2017 release's setting; build_sphere and build_all default to 1, "
                             'every sample, which is about N times slower and does not reproduce it; '
                             'with --recipe, a recipe without point_stride takes 1)')
    parser.add_argument('--min-elevation-deg', type=float, default=10.0,
                        help='drop emission points below this elevation, where ground '
                             'impedance dominates (default 10)')
    parser.add_argument('--max-array-range-ft', type=float, default=None,
                        help='additionally restrict the track to within this distance of '
                             "the run's reference point; off by default, since capping the "
                             'propagation path bounds the ray directly and this also '
                             'drops whole runs and biases descents toward the flare')
    parser.add_argument('--max-propagation-range-ft', type=float, default=2000.0,
                        help='drop emission-point/microphone pairs farther apart than '
                             'this, beyond which a straight ray through a homogeneous '
                             'atmosphere is a poor model (default 2000)')
    parser.add_argument('--rim-elevation-deg', type=float, default=None,
                        help='with --max-rim-range-ft: samples filed shallower than this, deg below the '
                             'horizon, may come from that range (default {:g})'.format(na.DEFAULT_RIM_ELEVATION_DEG))
    parser.add_argument('--max-rim-range-ft', type=float, default=None,
                        help='propagation range allowed for samples filed at the rim, ft '
                             '(default: --max-propagation-range-ft everywhere)')
    parser.add_argument('--min-steady-duration-s', type=float, default=8.0,
                        help='shortest steady segment worth building a sphere from, s; '
                             'the binding constraint on how many runs survive the gates '
                             '(default %(default)g)')
    parser.add_argument('--ambient-fallback-percentile', type=float, default=None,
                        help="percentile of the run's own spectrogram to use as ambient (e.g. 5) "
                             'where the array layout has no ambient run; less reliable than a '
                             'measured one')
    parser.add_argument('--azi-step', type=float, default=10.0,
                        help='azimuth step of the gridding, deg (default %(default)g)')
    parser.add_argument('--elv-step', type=float, default=10.0,
                        help='elevation step of the gridding, deg (default %(default)g)')
    parser.add_argument('--rmax', type=float, default=25.0,
                        help='fixed gridding radius, deg (default %(default)g; unused with adaptive gridding)')
    parser.add_argument('--adaptive-interpolation', action='store_true',
                        help='grid with a radius chosen per node (flight_acoustics.adaptive_idw_weights) '
                             'instead of the fixed --rmax (default off)')
    parser.add_argument('--interpolation', dest='interpolation', action='append', default=None,
                        type=_interpolation_setting, metavar='KEY=VALUE',
                        help='an adaptive gridding setting (implies --adaptive-interpolation), e.g. '
                             'kernel=shepard, aspect=5 or shepard_floor=1; repeat for each one; keys: '
                             + ', '.join(sorted(fa.ADAPTIVE_INTERPOLATION)) + ' (default: their defaults)')
    parser.add_argument('--flip-y-for-geometry', action='store_true',
                        help='mirror the track and microphones across y = 0 (a diagnostic; default off)')
    parser.add_argument('--no-prefetch', action='store_true',
                        help='do not warm the cloud-storage cache ahead of each run')
    parser.add_argument('--skip-existing', action='store_true',
                        help='do not rebuild a run whose sphere (and NORAH2 file and tonal and broadband '
                             'parts, when asked for) already exists in output_directory; default: rebuild')
    parser.add_argument('--skip-unchanged', action='store_true',
                        help='do not rebuild a run whose existing sphere was built with the same settings '
                             '(panam_settings_sha256) from the same input files (panam_input_signature: '
                             'names, sizes and modification times); default: rebuild.  A change to '
                             'the PANAM code does not count: rebuild without it after updating PANAM')
    parser.add_argument('--jobs', type=int, default=1,
                        help='build this many runs at once in worker processes (default %(default)s); '
                             'the output is the same as with 1')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='log more: DEBUG messages too (default: INFO and above)')
    parser.add_argument('-q', '--quiet', action='count', default=0,
                        help='log less: -q WARNING and above, -qq ERROR and above (default: INFO and above)')
    parser.add_argument('--log-file', default=None,
                        help='also append the log to this file (default: stderr only)')
    parser.add_argument('--no-ambient-gate', action='store_true',
                        help='reproduce the uncorrected legacy behavior')
    parser.add_argument('--remove-doppler', action='store_true',
                        help='file band power at the emitted frequency, for a hover source sphere; '
                             'never for spheres NICEOPS reads as flight (default off)')
    parser.add_argument('--tone-aware', action='store_true',
                        help='file each tone whole in the band of its own frequency instead of summing '
                             'whole FFT bins (FFT method only; default off)')
    parser.add_argument('--tone-split', action='store_true',
                        help='also write tonal and broadband part spheres, <name>_tonal.nc and '
                             '<name>_broadband.nc, in tonal/ and broadband/ under output_directory; '
                             'implies --tone-aware (default off)')
    parser.add_argument('--node-statistics', action='store_true',
                        help='also write N_SAMPLES and AMPLITUDE_STD (dB), the samples behind each '
                             'gridded node and their weighted level spread, to every sphere (default off)')
    return parser


def build_options(args):
    """The keyword arguments of :func:`panam.noise_abatement_2017.build_all` (and so
    :func:`build_sphere`) that ``args`` give.

    ``aircraft`` is included; ``output_directory`` is not.
    """
    interpolation = None
    if args.adaptive_interpolation or args.interpolation:
        interpolation = dict(args.interpolation or ())
    return dict(
        aircraft=args.aircraft, root=args.root, runs=args.runs,
        steady_only=not args.include_maneuvers, sphere_prefix=args.sphere_prefix,
        reference_directory=args.reference_directory, manifest_path=args.manifest,
        norah2_directory=args.norah2_directory, prefetch=not args.no_prefetch,
        winds=na.read_winds(args.winds) if args.winds else None,
        ray_models=na.wind_ray_models if args.ray_model == 'wind' else None,
        ray_profiles=args.ray_profiles,
        third_octave_method=args.third_octave_method,
        board_correction=args.board_correction,
        plate_table_directory=args.plate_table_directory,
        band_snr_gate_db=args.band_snr_gate_db,
        max_absorption_correction_db=args.max_absorption_correction_db,
        max_response_correction_db=args.max_response_correction_db,
        apply_absorption_deprop=not args.no_absorption_deprop,
        absorption_band_method=args.absorption_band_method,
        speed_of_sound_ft_s=args.speed_of_sound_ft_s,
        r_ref_ft=args.r_ref_ft, window_time=args.window_time, window_overlap=args.window_overlap,
        point_stride=args.point_stride, gate_ambient=not args.no_ambient_gate,
        min_elevation_deg=args.min_elevation_deg,
        max_array_range_ft=args.max_array_range_ft,
        max_propagation_range_ft=args.max_propagation_range_ft,
        rim_elevation_deg=args.rim_elevation_deg, max_rim_range_ft=args.max_rim_range_ft,
        min_steady_duration_s=args.min_steady_duration_s,
        ambient_fallback_percentile=args.ambient_fallback_percentile,
        azi_step=args.azi_step, elv_step=args.elv_step, rmax=args.rmax, interpolation=interpolation,
        flip_y_for_geometry=args.flip_y_for_geometry, overwrite=not args.no_overwrite,
        azimuth_reference=args.azimuth_reference, nose_from_heading=args.nose_from_heading,
        remove_doppler=args.remove_doppler, air_relative_doppler=args.air_relative_doppler,
        track_check=not args.no_track_check,
        min_height_above_array_ft=args.min_height_above_array_ft,
        tone_aware=args.tone_aware or args.tone_split,
        tone_split=args.tone_split,
        node_statistics=args.node_statistics,
        sphere_frame=args.sphere_frame,
        track_datum=args.track_datum,
        skip_existing=args.skip_existing, skip_unchanged=args.skip_unchanged, jobs=args.jobs)


def main(argv=None):
    parser = make_parser()
    # Intermixed, so that the optional positionals may follow options on Python 3.11 too.
    args = parser.parse_intermixed_args(argv)
    options = build_options(args)
    if args.hover_source and not args.recipe:
        parser.error('--hover-source applies to --recipe only')
    if args.recipe:
        try:
            recipe = build_recipe.load_recipe(args.recipe)
            recipe_options = recipe.sphere_options(hover_source=args.hover_source)
        except (ValueError, LookupError, OSError) as error:
            parser.error(str(error))
        options, explicit = build_recipe.apply(parser, argv, recipe_options, options, CLI_KEYWORDS)
        # A keyword neither in the recipe nor on the command line takes the function's
        # default, so that a recipe builds the same spheres here as from Python.  Only
        # point_stride has a command-line default (10) other than the function's (1).
        for key, value in build_recipe.function_defaults('spheres').items():
            if key in options and key not in explicit and key not in recipe_options:
                options[key] = value
        if args.hover_source and not options['runs']:
            parser.error('--hover-source needs --runs: the hover source run (a sphere built with '
                         'remove_doppler must not be read as flight)')
    if options['jobs'] < 1:
        parser.error('--jobs must be at least 1')

    root = logging.getLogger()
    previous_level = root.level
    handler = na.configure_logging(args.verbose - args.quiet, args.log_file)
    try:
        return _run_command(parser, args, options)
    finally:
        root.setLevel(previous_level)
        if handler is not None:
            root.removeHandler(handler)
            handler.close()


def _run_command(parser, args, options):
    """Everything :func:`main` does once its arguments are parsed and logging is set up."""
    if args.plate_cache:
        directory = na.resolve_plate_table_directory(options['plate_table_directory'])
        if args.plate_cache == 'clear':
            print('Deleted {} plate tables from {}'.format(na.clear_plate_table_cache(directory), directory))
            return 0
        records = na.plate_table_cache(directory)
        for record in records:
            print('{:8.1f} MB  {:<14s} {}'.format(record['size'] / 1e6, record['status'],
                                                   os.path.basename(record['path'])))
        print('{} plate tables in {}'.format(len(records), directory))
        return 0
    aircraft = options.pop('aircraft')
    if aircraft is None:
        parser.error('aircraft is required unless --plate-cache is given')
    if args.check_tracks:
        checks = na.check_tracks(aircraft, root=options['root'], runs=options['runs'],
                                 steady_only=options['steady_only'], hovers=args.hovers,
                                 max_array_range_ft=options['max_array_range_ft'],
                                 min_steady_duration_s=options['min_steady_duration_s'],
                                 min_height_ft=options['min_height_above_array_ft'],
                                 track_datum=options['track_datum'])
        for record in checks:
            record['aircraft'] = aircraft
            if record['status'] != 'ok':
                print('{} {} {}: {}'.format(record['run'], record['condition'], record['status'],
                                            '; '.join(r for r in (record['reasons'], record['flags']) if r)))
        counts = defaultdict(int)
        for record in checks:
            counts[record['status']] += 1
        print('{}: {} runs checked, {}'.format(aircraft, len(checks), ', '.join(
            '{} {}'.format(n, status) for status, n in sorted(counts.items()))))
        if options['manifest_path']:
            na.write_track_checks(options['manifest_path'], checks)
        return 0
    if args.output_directory is None:
        parser.error('output_directory is required unless --check-tracks is given')
    if args.ray_model == 'wind' and not options['winds']:
        parser.error('--ray-model wind needs --winds')
    if options['ray_models'] is not None and options['ray_profiles'] is not None:
        parser.error('give --ray-model wind or --ray-profiles, not both')
    records, failures = na.build_all(aircraft, args.output_directory, **options)
    skipped = sum(1 for record in records if record.get('skipped'))
    logging.info('Built %d spheres, skipped %d, %d failures', len(records) - skipped, skipped, len(failures))
    return 0 if records else 1


if __name__ == '__main__':
    raise SystemExit(main())
