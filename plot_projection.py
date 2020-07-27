#!/usr/bin/env python
import argparse

import matplotlib.pyplot as plt

import flight_acoustics

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--frequency", type=str, default=None, help="Frequency range \"low:high\"")
parser.add_argument("-o", "--output", type=str, default=None, help="Output image file name")
parser.add_argument("-c", "--cutoff", type=float, default=30.0, help="Elevation cutoff angle")
parser.add_argument("-a", "--altitude", type=float, default=500.0, help="Altitude above ground (m)")
parser.add_argument("-p", "--plot", action="store_true", help="Also plot image when outputting to file")
parser.add_argument("-u", "--units", type=str, default='m', help="Distance units to plot")
parser.add_argument("infile", help="Input netCDF file")

args = parser.parse_args()
freqs = None
if args.frequency is not None:
    freq_str = args.frequency.split(':')
    freqs = list(map(float, freq_str))
fig, ax, cs = flight_acoustics.plot_projection(args.infile, args.altitude, args.cutoff, freqs, args.units)
if args.output:
    fig.savefig(args.output, facecolor='none')
    if args.plot:
        plt.show(block=True)
else:
    plt.show(block=True)
