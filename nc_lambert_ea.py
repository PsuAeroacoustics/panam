#!/usr/bin/env python
import argparse

import matplotlib.pyplot as plt

import flight_acoustics

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--frequency", type=str, default=None, help="Frequency range \"low:high\"")
parser.add_argument("-r", "--range", type=str, default=None, help="SPL contour range \"low:high\"")
parser.add_argument("-o", "--output", type=str, default=None, help="Output image file name")
parser.add_argument("-p", "--plot", action="store_true", help="Also plot image when outputting to file")
parser.add_argument("-w", "--weight", type=str, default='none', choices=["A", "none"], help="Frequency weighting")
parser.add_argument("infile", help="Input netCDF file")

args = parser.parse_args()
freqs = None
if args.frequency is not None:
    freq_str = args.frequency.split(':')
    freqs = list(map(float, freq_str))
SPL_ranges = None
if args.range is not None:
    SPL_range_str = args.range.split(':')
    SPL_ranges = list(map(float, SPL_range_str))
fig, ax, cs = flight_acoustics.nc_lambert_ea(args.infile, freqs, args.weight, SPL_ranges)
if args.output:
    fig.savefig(args.output, bbox_inches='tight', facecolor='none')
    if args.plot:
        plt.show(block=True)
else:
    plt.show(block=True)
