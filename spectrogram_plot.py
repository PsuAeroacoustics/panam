#!/usr/bin/env python
import argparse
import os.path
import sys

import numpy as np

import flight_acoustics as fa
from cli import colon_pair, save_or_show


parser = argparse.ArgumentParser(
    description="Plot the spectrogram of one microphone signal from an HDF5 (.h5) or netCDF (.nc) file.")
parser.add_argument("-f", "--frequency", type=colon_pair, default=None,
                    help="Displayed frequency range \"low:high\", Hz (default: 0 to half the sampling rate)")
parser.add_argument("-o", "--output", type=str, default=None,
                    help="Image file to write (format from its extension); without it the plot opens in a window")
parser.add_argument("-p", "--plot", action="store_true", help="With -o, also open the plot in a window")
parser.add_argument("-x", "--x-limits", dest='x_limits', type=colon_pair, default=None,
                    help="Time range to analyze and plot \"low:high\", s, in the file's own times (a "
                         "netCDF file's start_time onward; an HDF5 record starts at 0); the signal is trimmed "
                         "to it")
parser.add_argument("-y", "--y-limits", dest='y_limits', type=colon_pair, default=None,
                    help="Displayed frequency range \"low:high\", Hz, as -f; takes precedence over -f "
                         "when both are given")
parser.add_argument("-c", "--c-limits", dest='c_limits', type=colon_pair, default=None,
                    help="Color scale limits \"low:high\", dB (default: from the data)")
parser.add_argument("-d", "--data-format", default=None, choices=['hdf5', 'netcdf'],
                    help="Input file format (default: from the extension, .h5/.hdf5 or .nc)")
parser.add_argument("-w", "--window-time", default=0.1, type=float, help="Window length, s (default 0.1)")
parser.add_argument("-l", "--overlap", default=7.0 / 8.0, type=float,
                    help="Fraction of each window that overlaps the next, 0 to 1 (default 0.875)")
parser.add_argument("-s", "--signal", default=None,
                    help="Signal (dataset) name in an HDF5 file; without it the names are listed and "
                         "nothing is plotted")
parser.add_argument("infile", help="Recording file (.h5, .hdf5 or .nc)")
args = parser.parse_args()

if args.data_format is None:
    file_extension = os.path.splitext(args.infile)[1]
    if file_extension.lower() in ['.h5', '.hdf5']:
        data_format = 'hdf5'
    elif file_extension.lower() in ['.nc']:
        data_format = 'netcdf'
    else:
        raise RuntimeError("No data-format supplied and cannot infer type from extension {!r}.".format(file_extension))
else:
    data_format = args.data_format

if data_format == 'hdf5':
    if args.signal is None:
        with fa.open_h5_signal(args.infile) as ds:
            print("No signal selected.  Specify one of: ", list(ds.keys()))
        sys.exit(0)
    with fa.open_h5_signal(args.infile, signalname=args.signal) as dataset:
        fs = 1.0 / dataset.attrs['ChannelInformationSamplingPeriod'][0]
        signal = dataset[:]
    time = np.arange(signal.size) / fs
else:
    signal, time, _ = fa.load_nc_signal(args.infile)
    fs = 1.0 / (time[1] - time[0])

if args.x_limits is not None:
    low, high = args.x_limits
    time_range = (time[0], time[-1])
    selection = (time >= low) & (time <= high)
    signal = signal[selection]
    time = time[selection]
    window_samples = fa.frame_length(args.window_time, fs)
    if signal.size < window_samples:
        parser.error(f"-x {low:g}:{high:g} selects {signal.size} samples, fewer than one {window_samples}-sample "
                     f"window; the record spans {time_range[0]:g} to {time_range[1]:g} s")
time0 = time[0]

flim = args.y_limits if args.y_limits is not None else args.frequency

fig, ax, cs = fa.plot_spectrogram(signal, fs, window_time=args.window_time, window_overlap=args.overlap, time0=time0,
                                  clim=args.c_limits, flim=flim)

save_or_show(fig, args.output, args.plot)
