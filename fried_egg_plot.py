#!/usr/bin/env python
import argparse

import flight_acoustics
from cli import colon_pair, save_or_show


parser = argparse.ArgumentParser(
    description="Fried-egg plot: project every sphere in a directory onto flat ground and contour each "
                "run's mean or peak ground level (dBA) over its flight condition. The directory needs "
                "a vehicle.cfg (see docs/file_formats.md).")
parser.add_argument("-f", "--frequency", type=colon_pair, default=None,
                    help="Sum only the bands whose center lies in \"low:high\", Hz (default: every band)")
parser.add_argument("-o", "--output", type=str, default=None,
                    help="Image file to write (format from its extension); without it the plot opens in a window")
parser.add_argument("-c", "--cutoff", type=float, default=30.0,
                    help="Elevation cutoff (deg below the horizon, between 0 and 90): "
                         "directions at least this far below are projected (default 30)")
parser.add_argument("-l", "--climb-cutoff", type=float, default=5.0,
                    help="Leave out runs whose flight path angle is above this, deg (default 5)")
parser.add_argument("-a", "--altitude", type=float, default=500.0, help="Altitude above ground, m (default 500)")
parser.add_argument("-p", "--plot", action="store_true", help="With -o, also open the plot in a window")
parser.add_argument("-m", "--metric", default="mean", choices=["mean", "max"],
                    help="Ground level of each run: the mean or the max over its footprint (default mean)")
parser.add_argument("-t", "--speed-duration-correction", type=float, default=None,
                    help="Reference speed, kt: add 10 lg(V_ref/V) to each run's level, a duration "
                         "correction (default: none)")
parser.add_argument("-n", "--dimensionless", dest='dimensionless', action='store_true',
                    help="Plot over advance ratio and rotor angle of attack (needs the [Option] fbar and "
                         "reflist entries in vehicle.cfg)")
parser.add_argument("-d", "--dimensional", dest='dimensionless', action='store_false',
                    help="Plot over speed (kt) and flight path angle (deg); the default")
parser.add_argument("-r", "--climb-rates", dest='climb_rates', action='store_true',
                    help="Plot over speed (kt) and rate of climb (ft/min)")
parser.add_argument("-x", "--x-limits", dest='x_limits', type=colon_pair, default=None,
                    help="Plot x-axis limits \"low:high\", in the axis' units")
parser.add_argument("-y", "--y-limits", dest='y_limits', type=colon_pair, default=None,
                    help="Plot y-axis limits \"low:high\", in the axis' units")
parser.add_argument("indir", help="Directory of AAM sphere files (.nc) and their vehicle.cfg")

args = parser.parse_args()
fig, ax, cs = flight_acoustics.fried_egg_plot(args.indir, metric=args.metric, dimensionless=args.dimensionless,
                                              altitude=args.altitude, cutoff=args.cutoff,
                                              input_frequencies=args.frequency, fpa_climb_cutoff=args.climb_cutoff,
                                              climb_rates=args.climb_rates,
                                              duration_correction=args.speed_duration_correction)

if args.x_limits is not None:
    ax.set_xlim(args.x_limits)
if args.y_limits is not None:
    ax.set_ylim(args.y_limits)
save_or_show(fig, args.output, args.plot)
