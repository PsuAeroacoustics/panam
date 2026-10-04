"""Scoped defaults for acoustic figures; importing does not alter plot settings."""
from functools import wraps

import matplotlib


def acoustic_plot_style(function):
    """Apply historical figure defaults only while the plotting function runs."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        from matplotlib import style
        with style.context('fivethirtyeight'), matplotlib.rc_context({
                'mathtext.fontset': 'dejavuserif', 'figure.autolayout': True}):
            return function(*args, **kwargs)
    return wrapped
