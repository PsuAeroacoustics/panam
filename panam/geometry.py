"""Emission geometry and angle frames.

:func:`hemigen` gives the direction each emission point is heard in from a
microphone and :func:`dedopplerize` resamples a pressure onto emission time
(``docs/THEORY.md`` sections 3 and 10).  The angle frames: ART/AAM (sphere
files, ``.nod``, NORAH2) has phi 0 below, positive to starboard, and theta 0 at
the nose; UMAPR (hemisphere dicts) has azimuth 180 ahead, 90 starboard, and
elevation positive below the horizon.  :func:`art2umapr` and :func:`norah2umapr`
convert; :func:`lambert_ea` projects onto the Lambert equal-area plane.
Lengths are in the caller's units, speeds in those units per second.
"""

import numpy as np


def dedopplerize(time, pressure, speed_of_sound, track_time, position, observers,
                 radius=None, output_sample_rate=None):
    """
    Time domain de-Dopplerization of an acoustic signal
    Args:
        time: observers x time points matrix of observer times
        pressure: observers x time point matrix of acoustic pressures
        speed_of_sound: speed of sound
        track_time: array of source times
        position: track times x 3 matrix of source positions
        observers: observers x 3 matrix of observer positions
        radius: optional virtual observer radius
        output_sample_rate: optional source time sample rate

    Returns: tuple (emission_times, dpres)
    WHERE
    emission_times is an array emission times
    dpres is an observer x emission times matrix of de-Dopplerized acoustic pressures,
    NaN where the emission would be received outside that observer's record
    """
    # If not set, infer sample rate from measured data
    if output_sample_rate is None:
        output_sample_rate = 1 / (time[0, 1] - time[0, 0])
    # Generate emission time array
    emission_times = np.arange(track_time[0], track_time[-1], 1 / output_sample_rate)
    number_of_microphones = observers.shape[0]
    number_of_times = emission_times.shape[0]
    dpres = np.zeros((number_of_microphones, number_of_times))
    # For each observer, de-Dopplerize
    for index, observer in enumerate(observers):
        rx = observer[0] - position[:, 0]
        ry = observer[1] - position[:, 1]
        rz = observer[2] - position[:, 2]
        r = np.interp(emission_times, track_time, np.sqrt(rx ** 2 + ry ** 2 + rz ** 2))
        dpres[index, :] = np.interp(emission_times + r / speed_of_sound, time[index, :], pressure[index, :],
                                    left=np.nan, right=np.nan)
        # Apply spherical spreading when radius is known
        if radius is not None:
            dpres[index, :] = r / radius * dpres[index, :]
    # Adjust time of emission to virtual observer observation time when radius is known
    if radius is not None:
        emission_times = emission_times + radius / speed_of_sound
    return emission_times, dpres


def hemigen(time, source, velocity, observers, speed_of_sound, nose=None):
    """
    Compute the time of emission observer angles for a moving source
    Args:
        time: Array of source times
        source: time x 3 matrix of source positions
        velocity: time x 3 matrix of source velocities
        observers: number of mics x 3 matrix of observer positions
        speed_of_sound: speed of sound
        nose: optional time x 3 (or time x 2) matrix of directions the azimuth is
            measured from, e.g. the airframe's heading.  Only its horizontal
            direction is used.  Default None: the velocity's, as before.  The
            Mach number is the velocity's either way.

    Returns: tuple (azimuth, elevation, r, t_observer, mach_r)
    WHERE
    azimuth are the azimuth angles on the sphere for each emission time
    elevation are the elevation angles on the sphere for each emission time
    r are the propagation distances for each emission time
    t_observer are the observer times associated with each emission time
    mach_r is the Mach number of the source in the propagation direction
    """
    # Rows are emission times, columns microphones
    t = time[:, None]
    sx, sy, sz = (source[:, i, None] for i in range(3))
    vx, vy, vz = (velocity[:, i, None] for i in range(3))
    ox, oy, oz = (observers[None, :, i] for i in range(3))
    # Propagation vectors
    rx = ox - sx
    ry = oy - sy
    rz = oz - sz
    r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
    # Observation time
    t_observer = t + r / speed_of_sound
    # Mach number along the radiation direction
    mach_r = (vx * rx / r + vy * ry / r + vz * rz / r) / speed_of_sound
    # Compute elevation relative to horizon; ground_range >= 0 keeps it in
    # [-90, 90], so it never runs over the pole
    ground_range = np.sqrt(rx ** 2 + ry ** 2)
    height = -rz
    elevation = np.degrees(np.arctan2(height, ground_range))
    # Compute azimuth as difference between aircraft heading and observer bearing
    bearing = np.degrees(np.arctan2(ry, rx))
    if nose is None:
        heading = np.degrees(np.arctan2(vy, vx))
    else:
        nose = np.asarray(nose, dtype=float)
        if nose.ndim != 2 or nose.shape[0] != np.size(time) or nose.shape[1] not in (2, 3):
            raise ValueError('nose must be shaped (Nt, 2) or (Nt, 3) to match time, not {}'.format(nose.shape))
        if not np.all(np.isfinite(nose[:, :2])) or np.any(np.hypot(nose[:, 0], nose[:, 1]) == 0.0):
            raise ValueError('nose must give a finite, nonzero horizontal direction at every time')
        heading = np.degrees(np.arctan2(nose[:, 1, None], nose[:, 0, None]))
    azimuth = np.remainder(bearing - (heading + 180), 360)
    return azimuth, elevation, r, t_observer, mach_r


def finite_difference_velocity(time, position):
    """Velocity (Nt, 3), length units per second, of a sampled track: central
    differences of ``position`` (Nt, 3) over ``time`` (Nt,) s (one-sided at the
    ends), as ``depropagate_hemisphere`` takes it when none is given."""
    return np.gradient(position, axis=0) / np.gradient(time)[:, None]


def norah2umapr(phi, theta):
    """Convert NORAH2 (azimuth phi, polar theta) directions to UMAPR (azimuth, elevation).

    NORAH2 defines emission angles in the body axes (x forward, y starboard,
    z down) as x = cos(theta), y = sin(theta) sin(phi), z = sin(theta) cos(phi).
    This is the same convention as ART (see :func:`art2umapr`), so the two
    agree direction for direction.

    Args:
        phi: NORAH2 azimuth angles (radians), positive to starboard.
        theta: NORAH2 polar angles (radians), 0 at the nose.

    Returns: tuple (azimuth, elevation) angles (radians)
    """
    phi = np.asarray(phi, dtype=float)
    theta = np.asarray(theta, dtype=float)
    forward = np.cos(theta)
    starboard = np.sin(theta) * np.sin(phi)
    down = np.sin(theta) * np.cos(phi)
    elv = np.arctan2(down, np.hypot(forward, starboard))
    azi = np.mod(np.arctan2(starboard, -forward), 2.0 * np.pi)
    return azi, elv


def art2umapr(phi, theta):
    """
    Convert between ART (AAM/RNM/ANOPP) coordinates and UMAPR coordinates

    ART follows the AAM Technical Reference (v3, sec. 2.4.1 and 2.5): theta is
    0 at the nose and 180 at the tail, phi is 0 directly below, positive to
    starboard and negative to port.  UMAPR azimuth is 180 ahead, 90 starboard.
    Before 2026-09-24 this used x = -sin(theta) sin(phi), putting phi = +90 to
    port; ``flip_y_for_geometry=True`` compensated for it in the AAM export,
    at the cost of mirroring every UMAPR hemisphere built that way.

    Args:
        phi: array of phi angles (radians)
        theta: array of theta angles (radians)

    Returns: tuple (azimuth, elevation) angles (radians)

    """
    x = np.sin(theta) * np.sin(phi)
    y = np.cos(theta)
    z = -np.sin(theta) * np.cos(phi)
    elv = -np.arctan2(z, np.sqrt(np.square(x) + np.square(y)))
    azi = np.arctan2(x, -y)
    azi[azi < 0.0] = azi[azi < 0.0] + 2.0 * np.pi
    return azi, elv


def lambert_ea(lat, lon):
    """
    Lambert equal-area projection of spherical angles
    Args:
        lat: array of lateral angles, radians
        lon: array of longitudinal angles, radians

    Returns: tuple, (x, y)

    """
    q = 2.0 * np.sin((np.pi / 2.0 - lat) / 2.0)
    x = q * np.sin(lon)
    y = q * np.cos(lon)
    return x, y


def lambert_lon(azimuth):
    """Page longitude for a UMAPR azimuth (radians in, radians out).

    The hemisphere is drawn as a VIEW FROM ABOVE, so it reads like the
    ground footprints and any other plan view:

        azimuth 180 (ahead)     -> top of the page
        azimuth  90 (starboard) -> RIGHT
        azimuth 270 (port)      -> left
        azimuth   0 (behind)    -> bottom

    i.e. azimuth increases counter-clockwise on the page, as
    :func:`lambert_ea_points` documents. Route every azimuth through here
    rather than open-coding the shift: before 2026-08-19 the data used
    ``lon = azimuth - pi`` (which mirrors the lateral axis, putting
    starboard on the LEFT) while the meridian labels printed
    ``360 - meridian``, so the rim numbers disagreed with the field.
    """
    return np.pi - np.asarray(azimuth, dtype=float)


def _lambert_xy_to_umapr(x, y):
    azimuth = np.mod(np.pi / 2.0 - np.arctan2(-y, x), 2.0 * np.pi)
    q = np.sqrt(np.square(x) + np.square(y))
    elevation = np.pi / 2.0 - 2.0 * np.arcsin(np.clip(q / 2.0, 0.0, 1.0))
    return azimuth, elevation


def _umapr2art(azimuth, elevation):
    azimuth = np.asarray(azimuth, dtype=float)
    elevation = np.asarray(elevation, dtype=float)
    cos_elevation = np.cos(elevation)
    x = cos_elevation * np.sin(azimuth)
    y = -cos_elevation * np.cos(azimuth)
    z = -np.sin(elevation)
    theta = np.arccos(np.clip(y, -1.0, 1.0))
    phi = np.arctan2(x, -z)
    return phi, theta


def geodist(elv1, azi1, elv2, azi2):
    """
    Compute geodesic arc length (degrees) on a sphere 

    Args:
        elv1, azi1, elv2, azi2: Scalars or array-like angles in degrees.

    Returns:
        numpy.ndarray (or scalar) of arc length in degrees.
    """
    elv1 = np.asarray(elv1, dtype=float)
    elv2 = np.asarray(elv2, dtype=float)
    azi1 = np.asarray(azi1, dtype=float)
    azi2 = np.asarray(azi2, dtype=float)

    # Convert degrees to radians
    r_elv1 = np.deg2rad(elv1)
    r_elv2 = np.deg2rad(elv2)
    r_azi1 = np.deg2rad(azi1)
    r_azi2 = np.deg2rad(azi2)

    # Spherical law of cosines
    arcs = np.sin(r_elv1) * np.sin(r_elv2) + np.cos(r_elv1) * np.cos(r_elv2) * np.cos(r_azi2 - r_azi1)

    # Clamp for numerical safety
    arcs = np.clip(arcs, -1.0, 1.0)

    # Arc length in degrees
    arclength = np.degrees(np.arccos(arcs))
    return arclength
