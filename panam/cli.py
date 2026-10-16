"""Argument types and figure output shared by the command-line scripts.

Figure output follows one convention.  A script that draws one figure takes
``-o/--output FILE`` and ``-p/--plot``: it writes FILE when given, opens a
window when not, and does both with ``-o`` and ``-p``.  A script that writes
several files takes ``-o/--output-dir DIR`` instead, and ``-p/--plot`` to also
open the figures.  :func:`add_output_options` adds these options and
:func:`save_or_show` and :func:`save_all_or_show` act on them.
"""
import argparse
import os

import matplotlib.pyplot as plt

from panam.acoustics.atmosphere import Atmosphere

#: The default atmosphere of a database's broadband EAA (Atmosphere arguments:
#: K, kPa, percent), as flight_acoustics' database builders default to it.
BUILD_ATMOSPHERE = dict(temperature=293.15, pressure=101.325, relative_humidity=20.0)


def colon_pair(value):
    """argparse type for a "low:high" pair of floats, quoted or not."""
    parts = value.strip().strip('"').strip("'").split(':')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError('expected "low:high", got {!r}'.format(value))
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error))


def colon_pair_or_single(value):
    """argparse type for "low:high", or a single value v given as (v, v)."""
    if ':' in value:
        return colon_pair(value)
    try:
        single = float(value.strip().strip('"').strip("'"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error))
    return single, single


def add_output_options(parser, default=None, *, directory=False, aliases=(), plot_flags=('-p', '--plot')):
    """Add the figure output options to ``parser``.

    Args:
        parser: The ``argparse.ArgumentParser`` (or argument group).
        default: Default output file (or directory); None opens a window
            instead of writing a file.
        directory: If True, add ``-o/--output-dir`` (``args.output_dir``), the
            directory that receives several files; if False (default), add
            ``-o/--output`` (``args.output``), one file.
        aliases: Older spellings of the output option, kept working.
        plot_flags: Flags of the "also open a window" option
            (``args.plot``); ``('--plot',)`` where ``-p`` means something else.
    """
    if directory:
        parser.add_argument('-o', '--output-dir', *aliases, dest='output_dir', default=default,
                            help=f'Directory to write the figures and files to, created if needed '
                                 f'(default {default})')
        parser.add_argument(*plot_flags, dest='plot', action='store_true',
                            help='Also open the figures in windows (default: only write them)')
    else:
        where = default if default else 'none: open the figure in a window'
        parser.add_argument('-o', '--output', *aliases, dest='output', default=default,
                            help=f'Image file to write; the format follows the extension (default {where})')
        parser.add_argument(*plot_flags, dest='plot', action='store_true',
                            help='With -o, also open the figure in a window')


def add_build_atmosphere_options(parser):
    """Add ``--build-temperature``, ``--build-pressure`` and ``--build-humidity``.

    They set the atmosphere of a database's broadband EAA (K, kPa, percent),
    defaulting to ``BUILD_ATMOSPHERE``; :func:`build_atmosphere` forms it.
    """
    parser.add_argument('--build-temperature', type=float, default=BUILD_ATMOSPHERE['temperature'],
                        help='atmosphere of the broadband EAA: temperature, K (default %(default)g)')
    parser.add_argument('--build-pressure', type=float, default=BUILD_ATMOSPHERE['pressure'],
                        help='... pressure, kPa (default %(default)g)')
    parser.add_argument('--build-humidity', type=float, default=BUILD_ATMOSPHERE['relative_humidity'],
                        help='... relative humidity, percent (default %(default)g)')


def build_atmosphere(args):
    """The :class:`Atmosphere` (K, kPa, percent) of :func:`add_build_atmosphere_options`' options."""
    return Atmosphere(temperature=args.build_temperature, pressure=args.build_pressure,
                      relative_humidity=args.build_humidity)


def save_or_show(fig, output=None, plot=False, **savefig_kwargs):
    """Save ``fig`` to ``output`` when given, and show it when not (or when ``plot``).

    The directory of ``output`` is created if needed.
    """
    if output:
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        fig.savefig(output, **savefig_kwargs)
    if not output or plot:
        plt.show(block=True)


def save_all_or_show(figures, output_dir=None, plot=False, extension='png', **savefig_kwargs):
    """Save each figure as ``output_dir/<name>.<extension>`` when ``output_dir`` is
    given, and show them all at once when not (or when ``plot``).

    Args:
        figures: Mapping of file name (without extension) to figure.
        output_dir: Directory to write to, created if needed; None writes nothing.
        plot: With ``output_dir``, also show the figures.
        extension: File extension, which sets the format (default ``'png'``).
        **savefig_kwargs: Passed to ``Figure.savefig`` (e.g. ``dpi``).

    Returns:
        The list of paths written, in the order of ``figures``.
    """
    paths = []
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for name, fig in figures.items():
            path = os.path.join(output_dir, f'{name}.{extension}')
            fig.savefig(path, **savefig_kwargs)
            paths.append(path)
    if not output_dir or plot:
        plt.show(block=True)
    return paths
