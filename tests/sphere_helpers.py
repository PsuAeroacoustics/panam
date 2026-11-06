"""Sphere fixtures shared by several test modules.

Imported as ``sphere_helpers``: pytest.ini puts ``tests`` on sys.path, so this
works under every pytest import mode, unlike importing one test module from
another.
"""

import os
import shutil
import subprocess

import numpy as np
from netCDF4 import Dataset

from panam import flight_acoustics as fa
from panam import local_paths


def executable(name):
    """A NICEOPS executable, or None: local_paths (PANAM_<NAME> or
    local_paths.toml), then the variable <NAME>, then PATH."""
    return (local_paths.data_path(name, required=False) or os.environ.get(name.upper())
            or shutil.which(name))

#: The NICEOPS commit these tests were last run against: one that reads what PANAM writes
#: (tests/niceops_pin.txt; re-pinned as NICEOPS changes).
with open(os.path.join(os.path.dirname(__file__), 'niceops_pin.txt')) as _pin:
    NICEOPS_PIN = _pin.read().strip()


def niceops():
    """``(path, reason)``: the configured niceops when it reports the pinned commit
    (``--version``; a dirty build of it too), else ``(None, why)`` for a test's skip.
    PANAM_NICEOPS_ANY_VERSION=1 takes any version."""
    path = executable('niceops')
    if path is None:
        return None, 'niceops not configured (local_paths: niceops)'
    try:
        version = subprocess.run([path, '--version'], capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, 'niceops {} does not run: {}'.format(path, error)
    if os.environ.get('PANAM_NICEOPS_ANY_VERSION') or version.split('-')[0].startswith(NICEOPS_PIN[:7]):
        return path, ''
    return None, ('niceops {} is NICEOPS {}, not the pinned {} (tests/niceops_pin.txt); '
                  'PANAM_NICEOPS_ANY_VERSION=1 runs it anyway'.format(path, version or '(no version)', NICEOPS_PIN))


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
    theta_deg on to the writer.  Level spheres, as the 2017 build writes them, unless
    ``grid`` says otherwise."""
    grid.setdefault('sphere_frame', 'level')
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


#: PSU-WOPWOP's AAM sphere scalars, in the order AAM reads them, and their units.
_WOPWOP_SCALARS = (('BB', ''), ('NB', ''), ('PT', ''), ('DOPPLER_SHIFT_REMOVED', ''), ('EMPTY_WEIGHT', 'POUNDS'),
                   ('FUEL_WEIGHT', 'POUNDS'), ('LOAD_WEIGHT', 'POUNDS'), ('RADIUS', 'FEET'),
                   ('FLIGHT_PATH_ANGLE', 'DEGREE'), ('PYLON_ANGLE', 'DEGREE'), ('SPEED', 'KNOTS'))
_WOPWOP_FLAG = {'BB': 'BB', 'OB12': 'NB', 'NB': 'NB', 'PT': 'PT'}


def write_wopwop_sphere(path, levels, phi, theta, frequency, kind='BB', speed=60.0, flight_path_angle=0.0,
                        floor=-40.0, rotor=(5.334, 230.7, 4), gross_weight_lb=4960.0, air_density=1.07,
                        prediction_radius=300.0, xyz=(0.0, 0.0, 0.0), phase=None):
    """A sphere laid out as PSU-WOPWOP writes one (its noiseSphere.f90 WriteSphereFile).

    netCDF classic; a singleton dimension per AAM scalar, the scalars in AAM's order
    with their ``unit``s; XYZ, PHI, THETA, FREQUENCY, AMPLITUDE (and PHASE for PT);
    then the metadata PANAM reads (GROSS_WEIGHT, AIR_DENSITY, AIRSPEED, MAIN_ROTOR_*,
    PREDICTION_RADIUS, TONAL_MAX_FREQUENCY) and WOPWOP's text attributes.  ``levels``
    (phi, theta, frequency) are written as given, so cells meant empty should hold
    ``floor``.  ``rotor`` is (radius m, tip speed m/s, blades), or None for none.  The
    frame is WOPWOP's, flight_path.
    """
    scalars = dict(BB=0.0, NB=0.0, PT=0.0, DOPPLER_SHIFT_REMOVED=0.0, EMPTY_WEIGHT=gross_weight_lb or 0.0,
                   FUEL_WEIGHT=0.0, LOAD_WEIGHT=0.0, RADIUS=100.0, FLIGHT_PATH_ANGLE=flight_path_angle,
                   PYLON_ANGLE=90.0, SPEED=speed)
    scalars[_WOPWOP_FLAG[kind]] = 1.0
    with Dataset(str(path), 'w', format='NETCDF3_CLASSIC') as sphere:
        for name, _ in _WOPWOP_SCALARS:
            sphere.createDimension(name, 1)
        for name, size in (('XYZ', 3), ('PHI', len(phi)), ('THETA', len(theta)), ('FREQUENCY', len(frequency))):
            sphere.createDimension(name, size)
        for name, unit in _WOPWOP_SCALARS:
            variable = sphere.createVariable(name, 'f4')
            variable.assignValue(np.float32(scalars[name]))
            variable.unit = unit
        for name, dims, values, unit in (('XYZ', ('XYZ',), xyz, 'FEET'), ('PHI', ('PHI',), phi, 'DEGREE'),
                                         ('THETA', ('THETA',), theta, 'DEGREE'),
                                         ('FREQUENCY', ('FREQUENCY',), frequency, 'HERTZ'),
                                         ('AMPLITUDE', ('PHI', 'THETA', 'FREQUENCY'), levels, 'DECIBEL')):
            variable = sphere.createVariable(name, 'f4', dims)
            variable[:] = values
            variable.unit = unit
        if phase is not None:
            variable = sphere.createVariable('PHASE', 'f4', ('PHI', 'THETA', 'FREQUENCY'))
            variable[:] = phase
            variable.unit = 'DEGREE'
        metadata = [('GROSS_WEIGHT', gross_weight_lb, 'POUNDS'), ('AIR_DENSITY', air_density, 'KG/M^3'),
                    ('AIRSPEED', speed, 'KNOTS')]
        if rotor is not None:
            metadata += [('MAIN_ROTOR_RADIUS', rotor[0], 'METER'), ('MAIN_ROTOR_TIP_SPEED', rotor[1], 'M/S'),
                         ('MAIN_ROTOR_BLADES', rotor[2], '')]
        metadata += [('PREDICTION_RADIUS', prediction_radius, 'METER'),
                     ('TONAL_MAX_FREQUENCY', float(np.max(frequency)), 'HERTZ')]
        for name, value, unit in metadata:
            if value is None:
                continue
            variable = sphere.createVariable(name, 'f4')
            variable.assignValue(np.float32(value))
            variable.unit = unit
        sphere.title = 'WOPWOP-like test sphere'
        sphere.sphere_frame = 'flight_path'
        sphere.sphere_type = kind
        sphere.content = 'discrete frequency'
        sphere.wind_source = 'none'
        sphere.air_density_source = 'wopwop'
        sphere.center_xyz_ft = '   '.join('{:.8E}'.format(v) for v in xyz)
        sphere.speed_reference = 'air'
        sphere.azimuth_reference = 'track'
        sphere.speed_of_sound_m_s = '{:.8E}'.format(340.0)
        sphere.floor_level_dB = '{:.8E}'.format(floor)
        if kind == 'NB':
            sphere.nb_bandwidth_Hz = '{:.6f}'.format(float(frequency[1] - frequency[0]))
        sphere.source = 'test'
        sphere.case = 'test'
    return path
