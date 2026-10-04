#!/usr/bin/env python
import argparse

import flight_acoustics
from cli import colon_pair, save_or_show

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--frequency", type=colon_pair, default=None, help="Frequency range \"low:high\"")
parser.add_argument("-r", "--range", type=colon_pair, default=None, help="SPL contour range \"low:high\"")
parser.add_argument("-o", "--output", type=str, default=None, help="Output image file name")
parser.add_argument("-p", "--plot", action="store_true", help="Also plot image when outputting to file")
parser.add_argument("-w", "--weight", type=str, default='none', choices=["A", "none"], help="Frequency weighting")
parser.add_argument(
    "-g",
    "--grid-convention",
    type=str,
    default='umapr',
    choices=["umapr", "art", "aam", "rnm"],
    help="Lambert grid overlay convention",
)
parser.add_argument("infile", help="Input netCDF file")

args = parser.parse_args()
fig, ax, cs = flight_acoustics.nc_lambert_ea(
    args.infile,
    args.frequency,
    args.weight,
    args.range,
    grid_convention=args.grid_convention,
)
save_or_show(fig, args.output, args.plot, bbox_inches='tight', facecolor='none')
