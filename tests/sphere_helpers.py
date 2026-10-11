"""Sphere fixtures shared by several test modules.

Imported as ``sphere_helpers``: pytest.ini puts ``tests`` on sys.path, so this
works under every pytest import mode, unlike importing one test module from
another.
"""

import os
import shutil

import numpy as np
from netCDF4 import Dataset

import flight_acoustics as fa
import local_paths


def executable(name):
    """A NICE-OPS executable, or None: local_paths (PANAM_<NAME> or
    local_paths.toml), then the variable <NAME>, then PATH."""
    return (local_paths.data_path(name, required=False) or os.environ.get(name.upper())
            or shutil.which(name))

#: A vehicle.cfg for directories of spheres (AS350-like rotors): example_data/vehicle.cfg.
with open(os.path.join(os.path.dirname(__file__), os.pardir, 'example_data', 'vehicle.cfg')) as _cfg:
    VEHICLE_CFG = _cfg.read()


def minimal_hemisphere():
    """A uniform 80 dB depropagate_hemisphere result: three bands on a 30 x 60 deg grid."""
    band_centers = np.array([100.0, 125.0, 160.0])
    elv = np.arange(0.0, 90.0 + 1e-9, 30.0)
    azi = np.arange(0.0, 360.0 + 1e-9, 60.0)
    bands = np.full((band_centers.size, elv.size, azi.size), 80.0)
    return dict(azi_grid_deg=azi, elv_grid_deg=elv,
                third_octave=dict(band_centers_hz=band_centers, bands_db=bands))


def write_sphere_directory(directory, hemispheres, **grid):
    """Write ``(hemisphere, speed_knots, flight_path_angle_deg)`` triples as
    X101.nc, X102.nc, ... beside a vehicle.cfg; ``grid`` passes phi_deg and
    theta_deg on to the writer."""
    directory.mkdir(exist_ok=True)
    (directory / 'vehicle.cfg').write_text(VEHICLE_CFG)
    for i, (hemisphere, speed, angle) in enumerate(hemispheres):
        fa.write_aam_hemisphere_netcdf(str(directory / f'X{101 + i:03d}.nc'), hemisphere,
                                       mode='third_octave', radius_ft=100.0, speed_knots=float(speed),
                                       flight_path_angle_deg=float(angle), title='t', **grid)
    return directory


def stepped_sphere_directory(directory, conditions, step_db=3.0):
    """:func:`write_sphere_directory` of :func:`minimal_hemisphere` at each
    ``(speed_knots, flight_path_angle_deg)`` condition, the i-th raised by
    ``i * step_db`` dB so that the spheres can be told apart."""
    hemispheres = []
    for i, (speed, angle) in enumerate(conditions):
        hemisphere = minimal_hemisphere()
        hemisphere['third_octave']['bands_db'] += step_db * i
        hemispheres.append((hemisphere, speed, angle))
    return write_sphere_directory(directory, hemispheres)


def tone_flyover(mic_y=(-300.0, 0.0, 300.0), seed=0):
    """A level pass for depropagation tests: ``depropagate_hemisphere``'s first
    arguments, ft and s.  The source crosses x = -400 to 400 ft at 250 ft over
    6 s (120 track samples); microphones on the ground at x = 0 and ``mic_y``
    (ft) record 7 s at 4 kHz of 2e-5 (3 sin(2 pi 250 t) + white noise) Pa,
    drawn from ``numpy.random.default_rng(seed)``."""
    rng = np.random.default_rng(seed)
    mic_locations = np.array([[0.0, float(y), 0.0] for y in mic_y])
    nt = 120
    track_time = np.linspace(0.0, 6.0, nt)
    track_position = np.column_stack([800.0 * track_time / track_time[-1] - 400.0, np.zeros(nt),
                                      np.full(nt, 250.0)])
    track_velocity = np.tile([800.0 / track_time[-1], 0.0, 0.0], (nt, 1))
    fs = 4000.0
    t = np.arange(0.0, 7.0, 1.0 / fs)
    pressure = np.vstack([2.0e-5 * (3.0 * np.sin(2 * np.pi * 250.0 * t) + rng.standard_normal(t.size))
                          for _ in mic_y])
    return dict(mic_locations=mic_locations, pressure=pressure, time=t, track_time=track_time,
                track_position=track_position, track_velocity=track_velocity)


def write_raw_sphere(path, phi=None, theta=None, frequency=None, amplitude=None, fill_value=None,
                     **scalars):
    """Write an AAM sphere variable by variable, bypassing write_aam_hemisphere_netcdf.

    For files that writer would not produce: a declared fill value, or
    variables left out (any argument left None).  ``scalars`` are scalar
    variables, e.g. RADIUS=100.0.
    """
    with Dataset(str(path), 'w', format='NETCDF3_CLASSIC') as sphere:
        for name, values in (('PHI', phi), ('THETA', theta), ('FREQUENCY', frequency)):
            if values is not None:
                sphere.createDimension(name, len(values))
                sphere.createVariable(name, 'f4', (name,))[:] = values
        if amplitude is not None:
            sphere.createVariable('AMPLITUDE', 'f4', ('PHI', 'THETA', 'FREQUENCY'),
                                  fill_value=fill_value)[:] = amplitude
        for name, value in scalars.items():
            sphere.createVariable(name, 'f4').assignValue(np.float32(value))
