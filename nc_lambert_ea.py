#!/usr/bin/env python
import argparse

import flight_acoustics
from cli import colon_pair, save_or_show

parser = argparse.ArgumentParser(
    description="Plot a sphere file's overall or A-weighted levels on a Lambert equal-area projection.")
parser.add_argument("-f", "--frequency", type=colon_pair, default=None,
                    help="Sum only the bands whose center lies in \"low:high\", Hz (default: every band)")
parser.add_argument("-r", "--range", type=colon_pair, default=None,
                    help="Contour level range \"low:high\", dB (default: from the data)")
parser.add_argument("-o", "--output", type=str, default=None,
                    help="Image file to write (format from its extension); without it the plot opens in a window")
parser.add_argument("-p", "--plot", action="store_true", help="With -o, also open the plot in a window")
parser.add_argument("-w", "--weight", type=str, default='none', choices=["A", "none"],
                    help="Frequency weighting: A, or none for unweighted levels (default none)")
parser.add_argument(
    "-g",
    "--grid-convention",
    type=str,
    default='umapr',
    choices=["umapr", "art", "aam", "rnm"],
    help="Grid lines to draw: umapr for azimuth/elevation; art, aam or rnm for phi/theta (default umapr)",
)
parser.add_argument("infile", help="AAM sphere file (.nc), e.g. example_data/AS350B3108.nc")

args = parser.parse_args()
fig, ax, cs = flight_acoustics.nc_lambert_ea(
    args.infile,
    args.frequency,
    args.weight,
    args.range,
    grid_convention=args.grid_convention,
)
save_or_show(fig, args.output, args.plot, bbox_inches='tight', facecolor='none')
