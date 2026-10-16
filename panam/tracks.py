"""NASA flight-test track files (comma-separated text): reading and trimming."""

import numpy as np


def load_NASA_track(trackfile):
    """
    Load NASA track file and return data in a structured format.

    Args:
        trackfile: Path to NASA CSV format tracking data file.
    Returns:
        dict with keys:
            'time': numpy.ndarray of UTC time in seconds
            'latitude': numpy.ndarray of latitude in degrees
            'longitude': numpy.ndarray of longitude in degrees
            'altitude': numpy.ndarray of altitude in feet
            'heading': numpy.ndarray of heading in degrees  
            'pitch': numpy.ndarray of pitch in degrees
            'roll': numpy.ndarray of roll in degrees
            'x', 'y', 'z': numpy.ndarray of position in feet (local coordinates,
                z positive up)
            'vx', 'vy', 'vz': numpy.ndarray of velocity in feet/second, as
                stored.  vz is not in z's sense in most files: the 2017 tracks
                store it positive DOWN, except EC130B4 test day 298, which
                stores it positive up.  Take its sign from
                :func:`noise_abatement_2017.vz_sign`.
    """    
    data = np.genfromtxt(trackfile, delimiter=',', names=True)
    time = data['utcsec']
    latitude = data['lat']
    longitude = data['lon']
    altitude = data['alt']  
    heading = data['heading']
    pitch = data['pitch']
    roll = data['roll']
    x = data['x']
    y = data['y']
    z = data['z']
    vx = data['vx']
    vy = data['vy']
    vz = data['vz']   

    return {
        'time': time,
        'latitude': latitude,
        'longitude': longitude,
        'altitude': altitude,
        'heading': heading,
        'pitch': pitch,
        'roll': roll,
        'x': x,
        'y': y,
        'z': z,
        'vx': vx,
        'vy': vy,
        'vz': vz,
    }


def filter_track(track, xlims = None, ylims = None, zlims = None, decimate=1):
    """
    Filter track data based on specified limits for x, y, z coordinates.

    Args:
        track: dict containing track data with keys 'x', 'y', 'z'.
        xlims: tuple (xmin, xmax) for filtering x coordinates.
        ylims: tuple (ymin, ymax) for filtering y coordinates.
        zlims: tuple (zmin, zmax) for filtering z coordinates.
        decimate: integer factor to decimate the track data (default: 1, no decimation).
    Returns:
        dict containing filtered track data with the same keys as input.
    """    
    x = track['x']
    y = track['y']
    z = track['z']
    mask = np.ones_like(x, dtype=bool)

    if xlims is not None:
        mask = np.logical_and(mask, x >= xlims[0])
        mask = np.logical_and(mask, x <= xlims[1])
    if ylims is not None:
        mask = np.logical_and(mask, y >= ylims[0])
        mask = np.logical_and(mask, y <= ylims[1])
    if zlims is not None:
        mask = np.logical_and(mask, z >= zlims[0])
        mask = np.logical_and(mask, z <= zlims[1])

    filtered_track = {}
    for key in track:
        filtered_track[key] = track[key][mask]
    # Decimate track data
    if decimate > 1:
        for key in filtered_track:
            filtered_track[key] = filtered_track[key][::decimate]
    return filtered_track   
