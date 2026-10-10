#!/usr/bin/env python
import argparse

import flight_acoustics
from cli import colon_pair, save_or_show

parser = argparse.ArgumentParser(
    description="Project a sphere file's A-weighted levels onto flat ground below it and contour the "
                "footprint: straight rays with spreading and atmospheric absorption, no ground effect.")
parser.add_argument("-f", "--frequency", type=colon_pair, default=None,
                    help="Sum only the bands whose center lies in \"low:high\", Hz (default: every band)")
parser.add_argument("-o", "--output", type=str, default=None,
                    help="Image file to write (format from its extension); without it the plot opens in a window")
parser.add_argument("-c", "--cutoff", type=float, default=30.0,
                    help="Elevation cutoff (deg below the horizon, between 0 and 90): "
                         "directions at least this far below are projected (default 30)")
parser.add_argument("-a", "--altitude", type=float, default=500.0, help="Altitude above ground, m (default 500)")
parser.add_argument("-p", "--plot", action="store_true", help="With -o, also open the plot in a window")
parser.add_argument("-u", "--units", type=str, default='m',
                    help="Units of the plot axes: ft, in, m, km, sm or nm (default m)")
parser.add_argument("infile", help="AAM sphere file (.nc), e.g. example_data/AS350B3108.nc")

args = parser.parse_args()
try:
    fig, ax, cs = flight_acoustics.plot_projection(args.infile, args.altitude, args.cutoff, args.frequency,
                                                   args.units)
except ValueError as error:
    parser.error(str(error))
save_or_show(fig, args.output, args.plot, facecolor='none')
