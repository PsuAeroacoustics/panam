#!/usr/bin/env python
import argparse
import os.path

import matplotlib.pyplot as plt

import flight_acoustics as fa


def two_floats(value):
    values = value.strip("'").strip('"').split(':')
    if len(values) != 2:
        raise argparse.ArgumentError
    values = list(map(float, values))
    return values


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--frequency", type=two_floats, default=None, help="Frequency range \"low:high\"")
parser.add_argument("-o", "--output", type=str, default=None, help="Output image file name")
parser.add_argument("-p", "--plot", action="store_true", help="Also plot image when outputting to file")
parser.add_argument("-x", "--x-limits", dest='x_limits', action='store', type=two_floats, default=None,
                    help="Set plot x limits \"low:high\"", nargs=1)
parser.add_argument("-y", "--y-limits", dest='y_limits', type=two_floats, default=None,
                    help="Set plot y limits \"low:high\"")
parser.add_argument("-c", "--c-limits", dest='c_limits', type=two_floats, default=None,
                    help="Set plot color level limits \"low:high\"")
parser.add_argument("-d", "--data-format", default=None, help="Force input file format (hdf5 or netcdf)",
                    choices=['hdf5', 'netcdf'])
parser.add_argument("-w", "--window-time", default=0.1, type=float, help="Time length of window")
parser.add_argument("-l", "--overlap", default=7.0 / 8.0, type=float, help="Proportion of window to overlap")
parser.add_argument("-s", "--signal", default=None, help="Name of signal in HDF5 dataset")
parser.add_argument("infile", help="Input filename.")
args = parser.parse_args()

if args.data_format is None:
    filename, file_extension = os.path.splitext(args.infile)
    if file_extension.lower() in ['.h5', '.hdf5']:
        data_format = 'hdf5'
    elif file_extension.lower() in ['.nc']:
        data_format = 'netcdf'
    else:
        print(file_extension)
        raise RuntimeError("No data-format supplied and cannot infer type from extension.")
else:
    data_format = args.data_format

if data_format == 'hdf5':
    if args.signal is None:
        ds = fa.load_h5_signal(args.infile)
        print("No signal selected.  Specify one of: ", list(ds.keys()))
        exit(0)
    signal = fa.load_h5_signal(args.infile, signalname=args.signal)
    fs = 1.0 / signal.attrs['ChannelInformationSamplingPeriod'][0]
else:
    signal, time, _ = fa.load_nc_signal(args.infile)
    fs = 1.0 / (time[1] - time[0])

if args.x_limits is not None:
    tlim = args.x_limits[0]
    i1 = int(tlim[0] * fs)
    i2 = int(tlim[1] * fs)
    signal = signal[i1:i2]
    time0 = tlim[0]
else:
    time0 = 0.0

if args.c_limits is not None:
    clim = args.c_limits
else:
    clim = None
if args.y_limits is not None:
    flim = args.y_limits
else:
    flim = None

fig, ax, cs = fa.plot_spectrogram(signal, fs, window_time=args.window_time, window_overlap=args.overlap, time0=time0,
                                  clim=clim, flim=flim)

if args.output:
    fig.savefig(args.output)
    if args.plot:
        plt.show(block=True)
else:
    plt.show(block=True)
