"""Acoustic recording loaders with explicit file lifetimes."""

from contextlib import contextmanager

import h5py
import numpy as np
from netCDF4 import Dataset
from pyuff import UFF


def load_nc_signal(filename):
    """
    Load NASA-formatted netCDF acoustic signal
    Args:
        filename: path to netCDF file

    Returns: tuple (pressure, time, location)
    WHERE
    pressure is an array of acoustic pressures
    time is an array of sampled times
    location is an array of the x,y,z location of the microphone
    """
    with Dataset(filename, mode='r') as file_handle:
        # Masked (fill-value) samples are missing, not pressures
        pressure = np.ma.filled(file_handle.variables['pressure'][:].astype(float), np.nan).flatten()
        x = file_handle.X
        y = file_handle.Y
        z = file_handle.Z
        sample_rate = file_handle.sample_rate
        start_time = file_handle.start_time
    # Not np.arange(start, stop, 1/fs): a float step can yield one sample too many.
    time = start_time + np.arange(pressure.size) / sample_rate
    location = np.array([x, y, z])
    return pressure, time, location


def load_h5_signal(filename, datasetname='Table1', signalname=None):
    """Return a live HDF5 group or dataset.

    The caller owns the open file and must close ``result.file``. Prefer
    ``open_h5_signal`` for scoped access. Selection errors close the file.
    """
    handle = h5py.File(filename, 'r')
    try:
        group = handle[datasetname]
        if not isinstance(group, h5py.Group):
            raise ValueError(f"Dataset {datasetname} is not a Group")
        return group if signalname is None else group[signalname]
    except BaseException:
        handle.close()
        raise


@contextmanager
def open_h5_signal(filename, datasetname='Table1', signalname=None):
    """Yield an HDF5 group or dataset and close its file on exit.

    Copy samples and attributes inside the context to retain them afterward.
    """
    signal = load_h5_signal(filename, datasetname, signalname)
    handle = signal.file
    try:
        yield signal
    finally:
        handle.close()


def load_UFF_signal(filename, sets = None):
    """
    Load UFF acoustic signal file
    Args:
        filename: path to UFF file
        sets: optional list of set numbers to load, default None (loads all sets);
            only type-58 (function at nodal DOF) sets are read as channels, so
            header (151), units (164) and other sets are skipped
    Returns: tuple (pressures, fs, channel_names, time)
    WHERE
    pressures is a channels x timepoints matrix of acoustic pressures
    fs is the sampling rate, Hz
    channel_names is a list of channel names
    time is an array of sampled times
    """

    file = UFF(filename)

    if sets is None:
        data = file.read_sets()
    else:
        data = file.read_sets(sets)
    # pyuff returns a bare dict, not a one-element list, when it reads one set
    if isinstance(data, dict):
        data = [data]
    data = [d for d in data if d.get('type') == 58]

    if not data:
        raise ValueError('No type-58 signal sets in UFF file.')
    channels = len(data)
    datasize = len(data[0]['x'])
    time = data[0]['x']
    time = np.asarray(time, dtype=float)
    if time.ndim != 1 or time.size < 2 or not np.isfinite(time).all():
        raise ValueError('UFF time grid must contain at least two finite samples.')
    spacing = np.diff(time)
    if spacing[0] <= 0 or not np.allclose(spacing, spacing[0], rtol=1e-7, atol=0):
        raise ValueError('UFF time grid must be increasing and uniformly spaced.')
    fs = 1. / spacing[0]
    pressures = np.zeros((channels, datasize))
    channel_names = []
    for i in range(channels):
        if len(data[i]['x']) != datasize:
            raise ValueError('Inconsistent recording lengths in UFF file.')
        channel_time = np.asarray(data[i]['x'], dtype=float)
        if channel_time.shape != time.shape or not np.allclose(
                channel_time, time, rtol=0, atol=spacing[0] * 1e-7):
            raise ValueError('Inconsistent channel time grids in UFF file.')
        pressures[i, :] = data[i]['data']
        channel_names.append(data[i]['id1'])
    return pressures, fs, channel_names, time


