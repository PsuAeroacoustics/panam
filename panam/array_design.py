"""Microphone array design: positions, coverage of the hemisphere, and KML export.

:func:`linear_array_plan` places microphones along a sideline; the coverage
functions give the directions each microphone samples during a pass, a hover
or a takeoff (``docs/THEORY.md`` section 10).  :func:`geodetic2array` and
:func:`array2geodetic` convert between latitude/longitude and the array frame.
"""

import numpy as np
from pymap3d import geodetic2enu, enu2geodetic
import simplekml

from panam import unit_conversion
from panam.geometry import hemigen
from panam.lambert_plots import lambert_ea_points


def linear_array_plan(nmics, altitude, min_elevation=10.0, target_elv=90.0):
    """
    Design a linear microphone array
    Args:
        nmics: number of microphones
        altitude: flight altitude of the vehicle
        min_elevation: desired sideline elevation angle relative to horizon
        target_elv: target elevation for equal angle sideline spacing

    Returns: array of microphone sideline locations
    """
    # Slant distance to target elevation center line position
    r = altitude / np.sin(np.radians(target_elv))
    # Now, determine spacing for equal angular resolution on target elv plane
    angles = np.linspace(-90. + min_elevation, 90.0 - min_elevation, nmics)
    return r * np.tan(np.radians(angles))


def array_coverage(ymics, altitude, xmin=-1000.0, xmax=1000.0, speed=100.0, rate=0.1, speed_of_sound=1135.):
    """
    Calculate the spherical coverage for an overflight of a linear microphone array
    Args:
        ymics: sideline distances of observers (at X=0 where +X is direction the vehicle is traveling)
        altitude: altitude of the vehicle
        xmin: optional, start distance of vehicle trajectory (default -1000)
        xmax: optional, end distance of vehicle trajectory (default 1000)
        speed: optional, default 100
        rate: optional, sampling period of emission time points, default 0.1
        speed_of_sound: optional, speed of sound, default 1135

    Returns: tuple (azimuth, elevation, r)
    WHERE
    azimuth are the azimuth angles on the sphere for each emission time
    elevation are the elevation angles on the sphere for each emission time
    r are the propagation distances for each emission time
    """
    observers = np.zeros((len(ymics), 3))
    observers[:, 1] = ymics
    tmin = 0
    tmax = (xmax - xmin) / speed
    time = np.arange(tmin, tmax, rate)
    source = np.zeros((len(time), 3))
    source[:, 0] = xmin + time * speed
    source[:, 2] = altitude

    velocity = np.zeros_like(source)
    velocity[:, 0] = speed

    azimuth, elevation, r, t_observer, mach_r = hemigen(time, source, velocity, observers, speed_of_sound)
    return azimuth, elevation, r


def hover_array_coverage(radial_positions, mic_azimuths, altitudes, headings, speed_of_sound=1135.0):
    """
    Calculate the spherical coverage for a hovering vehicle with microphones placed
    on a circle (or rings) around the source location.

    Args:
        radial_positions: radial distances of observers from the hover point (same length units as altitude)
        mic_azimuths: observer azimuths around the hover point, degrees (0 deg = +X, 90 deg = +Y)
        altitudes: one or more hover altitudes (same length units as radial_positions)
        headings: one or more vehicle headings, degrees (0 deg = +X, 90 deg = +Y)
        speed_of_sound: optional, speed of sound, default 1135

    Returns: tuple (azimuth, elevation, r)
    WHERE
    azimuth are the azimuth angles on the sphere for each hover condition (time) and mic
    elevation are the elevation angles on the sphere for each hover condition (time) and mic
    r are the propagation distances for each hover condition (time) and mic
    """
    radial_positions = np.atleast_1d(radial_positions).astype(float)
    mic_azimuths = np.atleast_1d(mic_azimuths).astype(float)
    altitudes = np.atleast_1d(altitudes).astype(float)
    headings = np.atleast_1d(headings).astype(float)

    if radial_positions.size == 0 or mic_azimuths.size == 0:
        raise ValueError('radial_positions and mic_azimuths must be non-empty')
    if altitudes.size == 0 or headings.size == 0:
        raise ValueError('altitudes and headings must be non-empty')

    if radial_positions.size != mic_azimuths.size:
        raise ValueError('radial_positions and mic_azimuths must have the same length')

    # Build mic locations in the ground plane centered on the hover point.
    az_rad = np.radians(mic_azimuths)
    obs_x = radial_positions * np.cos(az_rad)
    obs_y = radial_positions * np.sin(az_rad)
    observers = np.column_stack((obs_x, obs_y, np.zeros(obs_x.size)))

    # Enumerate hover conditions; time is arbitrary and only used to align rows.
    alt_grid, head_grid = np.meshgrid(altitudes, headings, indexing='ij')
    n_combo = alt_grid.size
    time = np.arange(n_combo, dtype=float)

    source = np.zeros((n_combo, 3), dtype=float)
    source[:, 2] = alt_grid.ravel()

    heading_rad = np.radians(head_grid.ravel())
    # Unit velocity vector defines vehicle heading for azimuth convention.
    velocity = np.zeros_like(source)
    velocity[:, 0] = np.cos(heading_rad)
    velocity[:, 1] = np.sin(heading_rad)

    azimuth, elevation, r, t_observer, mach_r = hemigen(time, source, velocity, observers, speed_of_sound)
    return azimuth, elevation, r


def takeoff_array_coverage(
        radial_positions,
        mic_azimuths,
        ground_distances,
        *,
        initial_altitude=0.0,
        flight_path_angle=8.0,
        heading=0.0,
    source_origin_xy=(0.0, 0.0),
        speed_of_sound=1135.0,
):
    """
    Calculate spherical coverage for a takeoff/climb trajectory over a circular/ring array.

    Args:
        radial_positions: radial distances of observers from the reference point.
        mic_azimuths: observer azimuths around the reference point, deg (0 deg = +X, 90 deg = +Y).
        ground_distances: source ground-track distances from reference point.
        initial_altitude: source altitude at zero ground distance.
        flight_path_angle: climb angle in deg above local horizon.
        heading: source heading in deg (0 deg = +X, 90 deg = +Y).
        source_origin_xy: (x0, y0) source location at zero ground distance in
            the same local coordinates as observers. Default is (0, 0).
        speed_of_sound: speed of sound.

    Returns: tuple (azimuth, elevation, r)
    WHERE
    azimuth are the observer azimuth angles on the source hemisphere
    elevation are the observer elevation angles on the source hemisphere
    r are source-observer propagation distances
    """
    radial_positions = np.atleast_1d(radial_positions).astype(float)
    mic_azimuths = np.atleast_1d(mic_azimuths).astype(float)
    ground_distances = np.atleast_1d(ground_distances).astype(float)

    if radial_positions.size == 0 or mic_azimuths.size == 0:
        raise ValueError('radial_positions and mic_azimuths must be non-empty')
    if ground_distances.size == 0:
        raise ValueError('ground_distances must be non-empty')
    if radial_positions.size != mic_azimuths.size:
        raise ValueError('radial_positions and mic_azimuths must have the same length')

    az_rad = np.radians(mic_azimuths)
    obs_x = radial_positions * np.cos(az_rad)
    obs_y = radial_positions * np.sin(az_rad)
    observers = np.column_stack((obs_x, obs_y, np.zeros(obs_x.size)))

    heading_rad = np.radians(float(heading))
    fpa_rad = np.radians(float(flight_path_angle))
    origin_xy = np.asarray(source_origin_xy, dtype=float).ravel()
    if origin_xy.size != 2:
        raise ValueError('source_origin_xy must have exactly 2 values: (x0, y0)')

    source = np.zeros((ground_distances.size, 3), dtype=float)
    source[:, 0] = origin_xy[0] + ground_distances * np.cos(heading_rad)
    source[:, 1] = origin_xy[1] + ground_distances * np.sin(heading_rad)
    source[:, 2] = float(initial_altitude) + ground_distances * np.tan(fpa_rad)

    velocity = np.zeros_like(source)
    velocity[:, 0] = np.cos(heading_rad)
    velocity[:, 1] = np.sin(heading_rad)
    velocity[:, 2] = np.tan(fpa_rad)

    time = np.arange(ground_distances.size, dtype=float)
    azimuth, elevation, r, t_observer, mach_r = hemigen(time, source, velocity, observers, speed_of_sound)
    return azimuth, elevation, r


def hover_array_coverage_plot(radial_positions, mic_azimuths, altitudes, headings, speed_of_sound=1135.0):
    """
    Plot the spherical coverage for hovering conditions with scattered microphones.

    Args:
        radial_positions: radial distances of observers from the hover point (same length units as altitude)
        mic_azimuths: observer azimuths around the hover point, degrees (0 deg = +X, 90 deg = +Y)
        altitudes: one or more hover altitudes (same length units as radial_positions)
        headings: one or more vehicle headings, degrees (0 deg = +X, 90 deg = +Y)
        speed_of_sound: optional, speed of sound, default 1135

    Returns: tuple (fig, ax, cs)
    WHERE
    fig is a matplotlib handle to the figure
    ax is a matplotlib handle to the plot axis
    cs is a matplotlib handle to the data points
    """
    azimuth, elevation, _ = hover_array_coverage(radial_positions, mic_azimuths, altitudes,
                                                 headings, speed_of_sound)
    fig, ax, cs = lambert_ea_points(np.radians(azimuth), np.radians(elevation))
    return fig, ax, cs


def array_coverage_plot(ymics, altitude, xmin=-1000.0, xmax=1000.0, speed=100.0, rate=0.1, speed_of_sound=1135.):
    """
    Plots the spherical coverage for an overflight of a linear microphone array
    Args:
        ymics: sideline distances of observers (at X=0 where +X is direction the vehicle is traveling)
        altitude: altitude of the vehicle
        xmin: optional, start distance of vehicle trajectory (default -1000)
        xmax: optional, end distance of vehicle trajectory (default 1000)
        speed: optional, default 100
        rate: optional, sampling period of emission time points, default 0.1
        speed_of_sound: optional, speed of sound, default 1135

    Returns: tuple (fig, ax, cs)
    WHERE
    fig is a matplotlib handle to the figure
    ax is a matplotlib handle to the plot axis
    cs is a matplotlib handle to the data points
    """
    azimuth, elevation, _ = array_coverage(ymics, altitude, xmin, xmax, speed, rate, speed_of_sound)
    fig, ax, cs = lambert_ea_points(np.radians(azimuth), np.radians(elevation))
    return fig, ax, cs


def geodetic2array(geodetic, reference, heading, units='ft'):
    """
    Convert geodetic coordinates to a local array coordinate system.

    This function transforms geodetic coordinates (latitude, longitude, altitude) into a local
    coordinate system centered at a reference point, rotated according to a specified heading.

    Args:
        geodetic (numpy.ndarray): An Nx3 array of geodetic coordinates where each row contains
            [latitude, longitude, altitude].
        reference (array-like): A 3-element array containing the reference point's geodetic
            coordinates [latitude, longitude, altitude].
        heading (float): The heading angle in degrees, measured clockwise from north (0-360).
            The local coordinate system is rotated so that the x-axis aligns with this heading.
        units (str, optional): The desired output length units. Defaults to 'ft' (feet).
            The function converts from meters to the specified units.

    Returns:
        numpy.ndarray: An Nx3 array of local coordinates where:
            - Column 0: x-coordinate (aligned with heading direction)
            - Column 1: y-coordinate (perpendicular to heading, positive left)
            - Column 2: z-coordinate (up, altitude above reference)

    Notes:
        - Input geodetic coordinates are first converted to ENU (East-North-Up) coordinates
          relative to the reference point.
        - The coordinate system is then rotated so that the x-axis points in the direction
          of the specified heading.
        - The rotation angle is computed as 90° - heading to align the x-axis with the heading.
    """
    east, north, up = geodetic2enu(geodetic[:, 0], geodetic[:, 1], geodetic[:, 2], reference[0], reference[1],
                                   reference[2])
    rotation = np.radians(90.0 - heading)
    local = np.zeros((len(east), 3))
    local[:, 0] = east * np.cos(rotation) + north * np.sin(rotation)
    local[:, 1] = -east * np.sin(rotation) + north * np.cos(rotation)
    local[:, 2] = up
    local = unit_conversion.len_conv(local, from_units='m', to_units=units)
    return local


def array2geodetic(local, reference, heading, units='ft'):
    """
    Convert local coordinate array to geodetic coordinates (latitude, longitude, height).

    This function transforms an array of points from a local coordinate system to geodetic 
    coordinates (WGS84) by first rotating the local coordinates based on a heading angle 
    to align with East-North-Up (ENU) convention, then converting to geodetic coordinates 
    relative to a reference point.

    Parameters
    ----------
    local : numpy.ndarray
        Array of shape (N, 3) containing local coordinates [x, y, z] in the
        frame :func:`geodetic2array` produces (this is its inverse):
        - x: along the heading (positive forward)
        - y: perpendicular to the heading (positive left)
        - z: vertical position (positive up)
    reference : array-like
        Reference point in geodetic coordinates [latitude, longitude, height] in degrees 
        and meters respectively, used as the origin for the ENU coordinate system.
    heading : float
        Heading angle in degrees (typically 0-360), measured clockwise from North.
        Used to rotate the local coordinates to align with true North.
    units : str, optional
        Units of the input local coordinates. Default is 'ft' (feet).
        Coordinates are converted to meters internally.

    Returns
    -------
    numpy.ndarray
        Array of shape (N, 3) containing geodetic coordinates where:
        - Column 0: latitude in degrees
        - Column 1: longitude in degrees
        - Column 2: height in meters

    Notes
    -----
    The conversion process involves:
    1. Converting input coordinates from specified units to meters
    2. Rotating coordinates by (heading - 90°) to convert from local to ENU frame
    3. Converting ENU coordinates to geodetic using the reference point
    """
    local = unit_conversion.len_conv(local, from_units=units, to_units='m')
    rotation = np.radians(heading - 90.0)
    # Convert local to ENU
    east = local[:, 0] * np.cos(rotation) + local[:, 1] * np.sin(rotation)
    north = -local[:, 0] * np.sin(rotation) + local[:, 1] * np.cos(rotation)
    up = local[:, 2]
    lat, lon, h = enu2geodetic(east, north, up, reference[0], reference[1], reference[2])
    geodetic = np.zeros_like(local)
    geodetic[:, 0] = lat
    geodetic[:, 1] = lon
    geodetic[:, 2] = h
    return geodetic


def write_kml(geodetic, savename, testname='Array', channel_prefix="M"):
    """
    Write geodetic coordinates to a KML/KMZ file for visualization in mapping applications.

    Parameters
    ----------
    geodetic : array-like
        An iterable of coordinate pairs where each row contains [latitude, longitude].
        Coordinates should be in decimal degrees.
    savename : str
        The output file path where the KMZ file will be saved.
    testname : str, optional
        The name of the KML document. Default is 'Array'.
    channel_prefix : str, optional
        The prefix to use for naming each point/placemark. Default is 'M'.
        Points will be named as '{channel_prefix}{number}' (e.g., 'M1', 'M2', etc.).

    Returns
    -------
    None
        The function saves the KMZ file to disk but does not return a value.

    Notes
    -----
    - Each point is displayed with a circular placemark icon from Google Maps.
    - Point numbering starts from 1 (i.e., channel_prefix + '1' for the first point).
    - The geodetic input should have longitude at index 1 and latitude at index 0.
    """
    kml = simplekml.Kml()
    kml.document.name = testname
    for (i, row) in enumerate(geodetic):
        pnt = kml.newpoint(name=channel_prefix + "{}".format(i + 1), coords=[(row[1], row[0])])
        pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
    kml.savekmz(savename)
