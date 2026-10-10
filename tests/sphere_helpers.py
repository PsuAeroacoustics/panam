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

#: A vehicle.cfg for directories of spheres (AS350-like rotors).
VEHICLE_CFG = ('[Main Rotor]\nradius = 5.334\ntip speed = 230.7\nblades = 4\n'
               '[Tail Rotor]\nradius = 0.8255\ntip speed = 216.1\nblades = 2\n'
               '[Atmosphere]\ndensity = 1.070\ntemperature = 280.37\n'
               '[Vehicle]\nweight = 2250\ndrag = 0.8175\n')


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
