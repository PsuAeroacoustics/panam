#!/usr/bin/env python
import argparse

import flight_acoustics
from cli import colon_pair, save_or_show

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--frequency", type=colon_pair, default=None, help="Frequency range \"low:high\"")
parser.add_argument("-o", "--output", type=str, default=None, help="Output image file name")
parser.add_argument("-c", "--cutoff", type=float, default=30.0,
                    help="Elevation cutoff (deg below the horizon, between 0 and 90): "
                         "directions at least this far below are projected")
parser.add_argument("-a", "--altitude", type=float, default=500.0, help="Altitude above ground (m)")
parser.add_argument("-p", "--plot", action="store_true", help="Also plot image when outputting to file")
parser.add_argument("-u", "--units", type=str, default='m', help="Distance units to plot")
parser.add_argument("infile", help="Input netCDF file")

args = parser.parse_args()
try:
    fig, ax, cs = flight_acoustics.plot_projection(args.infile, args.altitude, args.cutoff, args.frequency,
                                                   args.units)
except ValueError as error:
    parser.error(str(error))
save_or_show(fig, args.output, args.plot, facecolor='none')
