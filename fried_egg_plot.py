#!/usr/bin/env python
import argparse

import matplotlib.pyplot as plt

import flight_acoustics


def two_floats(value):
    values = value.strip("'").strip('"').split(':')
    if len(values) != 2:
        raise argparse.ArgumentTypeError('expected "low:high"')
    values = list(map(float, values))
    return values


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--frequency", type=two_floats, default=None, help="Frequency range \"low:high\"")
parser.add_argument("-o", "--output", type=str, default=None, help="Output image file name")
parser.add_argument("-c", "--cutoff", type=float, default=30.0, help="Elevation cutoff angle")
parser.add_argument("-l", "--climb-cutoff", type=float, default=5.0, help="Climb angle cutoff")
parser.add_argument("-a", "--altitude", type=float, default=500.0, help="Altitude above ground (m)")
parser.add_argument("-p", "--plot", action="store_true", help="Also plot image when outputting to file")
parser.add_argument("-m", "--metric", type=str, default="mean", help="Mean or max projected ground level")
parser.add_argument("-t", "--speed-duration-correction", type=float, default=None,
                    help="Reference speed (kts) for duration correction.")
parser.add_argument("-n", "--dimensionless", dest='dimensionless', action='store_true',
                    help="Use nondimensional flight conditions")
parser.add_argument("-d", "--dimensional", dest='dimensionless', action='store_false',
                    help="Use dimensional flight conditions")
parser.add_argument("-r", "--climb-rates", dest='climb_rates', action='store_true',
                    help="Use dimensional flight conditions with climb rates")
parser.add_argument("-x", "--x-limits", dest='x_limits', action='store', type=two_floats, default=None,
                    help="Set plot x limits \"low:high\"")
parser.add_argument("-y", "--y-limits", dest='y_limits', type=two_floats, default=None,
                    help="Set plot y limits \"low:high\"")
parser.set_defaults(dimensionless=False)
parser.set_defaults(climb_rates=False)
parser.add_argument("indir", help="Input directory containing netCDF files")

args = parser.parse_args()
fig, ax, cs = flight_acoustics.fried_egg_plot(args.indir, metric=args.metric, dimensionless=args.dimensionless,
                                              altitude=args.altitude, cutoff=args.cutoff,
                                              input_frequencies=args.frequency, fpa_climb_cutoff=args.climb_cutoff,
                                              climb_rates=args.climb_rates,
                                              duration_correction=args.speed_duration_correction, threshold=.65)

if args.x_limits is not None:
    ax.set_xlim(args.x_limits)
if args.y_limits is not None:
    ax.set_ylim(args.y_limits)
if args.output:
    fig.savefig(args.output)
    if args.plot:
        plt.show(block=True)
else:
    plt.show(block=True)
