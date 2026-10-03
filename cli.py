"""Argument types and figure output shared by the command-line scripts."""
import argparse

import matplotlib.pyplot as plt


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


def save_or_show(fig, output=None, plot=False, **savefig_kwargs):
    """Save ``fig`` to ``output`` when given, and show it when not (or when ``plot``)."""
    if output:
        fig.savefig(output, **savefig_kwargs)
    if not output or plot:
        plt.show(block=True)
