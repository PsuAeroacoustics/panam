"""Format 2 NICEOPS databases of pure-tone (PT) spheres: one source per rotor, with its phase.

PSU-WOPWOP writes one PT noise sphere per rotor (``sourceTag``): each shaft order's level and cosine phase over the
sphere, about the rotor's own hub, in the flight-path frame, with the rotor's shaft rate and its azimuth at the
phase reference time.  :func:`build_pt_database` turns a set of them into a database NICEOPS evaluates tone by
tone (``docs/file_formats.md``, "Format 2: pure-tone sources"):

- every condition group (one per flight condition: SPEED, FLIGHT_PATH_ANGLE) keeps the format 1 fields NICEOPS
  requires, with a band spectrum that holds the broadband only (``band_content = broadband``): empty (no energy)
  unless broadband spheres of the same conditions are given;
- each condition group holds one subgroup per source (rotor), named by its tag: ``center`` (m, sphere frame:
  forward, starboard, up), ``shaft_rate`` (rad/s, positive counterclockwise seen from above), ``harmonic`` (shaft
  orders), ``frequency`` (Hz, emission frame), ``amplitude`` (dB at the sphere radius, over PHI, THETA, tone; PSU-WOPWOP's floor level where a tone is quieter
  than it),
  ``phase`` (deg), ``azimuth`` (deg: the rotor's azimuth the phase was locked to), and the text attributes
  ``phase_reference = rotor_azimuth`` and, for rotors that turn together, ``sync_group``.

Nothing that would damage phase is done: no pitch to level (``sphere_frame = flight_path``), no mirroring of the
upper half, no regridding, no synthesized hover, no load-factor ladder.  The spheres are lossless (PSU-WOPWOP
applies no absorption), so ``sphere_levels_lossless = 1``.

Lengths in m unless named otherwise; angles in degrees; speeds as named.
"""
import math
import os
import warnings
from collections import OrderedDict

import numpy as np
from netCDF4 import Dataset

from panam.acoustics.atmosphere import Atmosphere
from panam.database import (_SphereCondition, _write_database_root, _write_shared_grid_and_frequency,
                            add_sphere_group)
from panam.sphere_io import KNOT_MPS, _atomic_output
from panam import provenance as _provenance

#: Root ``database_version`` of a database with pure-tone sources.
PT_DATABASE_FORMAT_VERSION = 2

#: A multirotor condition's momentum-similarity axes, in NICEOPS's order (x, z, M_h), and the fields they follow from.
MULTIROTOR_AXES = ('inplane_speed_ratio', 'axial_speed_ratio', 'hover_tip_mach')
MULTIROTOR_FIELDS = ('air_density', 'weight_n', 'airspeed', 'climb_rate')

#: The speed references a sphere or a database may give (``speed_reference``).
SPEED_REFERENCES = ('air', 'ground')


def hover_induced_velocity(weight_n, air_density, rotor_count, rotor_radius_m):
    """Momentum theory's hover induced velocity v_h = sqrt(W / (2 rho N pi R^2)), m/s, for a vehicle of weight W (N)
    on N rotors of radius R (m) in air of density rho (kg/m^3)."""
    return math.sqrt(weight_n / (2.0 * air_density * rotor_count * math.pi * rotor_radius_m ** 2))


def hover_shaft_rate(weight_n, air_density, rotor_count, rotor_radius_m, hover_thrust_coefficient):
    """The shaft rate Omega_h (rad/s) at which each of N rotors carries W / N at the hover thrust coefficient C_Th
    (per rotor, T / (rho pi R^2 (Omega R)^2)): Omega_h R = sqrt(W / (N rho pi R^2 C_Th))."""
    return math.sqrt(weight_n / (rotor_count * air_density * math.pi * rotor_radius_m ** 2
                                 * hover_thrust_coefficient)) / rotor_radius_m

#: Units a PSU-WOPWOP (AAM-style) sphere writes in each field's ``unit`` attribute, blank-padded; anything else is
#: refused rather than guessed.
_UNITS = {'RADIUS': 'FEET', 'SPEED': 'KNOTS', 'FLIGHT_PATH_ANGLE': 'DEGREE', 'XYZ': 'FEET', 'PHI': 'DEGREE',
          'THETA': 'DEGREE', 'FREQUENCY': 'HERTZ', 'AMPLITUDE': 'DECIBEL', 'PHASE': 'DEGREE',
          'MAIN_ROTOR_RADIUS': 'METER', 'MAIN_ROTOR_TIP_SPEED': 'M/S', 'ROTOR_SHAFT_RATE': 'RAD/S',
          'ROTOR_AZIMUTH': 'DEGREE', 'AIR_DENSITY': 'KG/M^3'}

FOOT_M = 0.3048

#: Third-octave band centers (Hz, base 10: 10^(n/10)) of the band spectrum written beside the tones, 10 Hz-10 kHz:
#: wide enough to hold a slow rotor's blade passage (the PSU-WOPWOP tandem rotor's is 14.3 Hz).
BAND_CENTERS_HZ = 10.0 ** (np.arange(10, 41) / 10.0)


class PTSphere:
    """One PT sphere file, read and checked (see :func:`read_pt_sphere`)."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def _unit(variable):
    return str(getattr(variable, 'unit', getattr(variable, 'units', ''))).strip().upper()


def _value(handle, name, path):
    if name not in handle.variables:
        raise ValueError(f'{path}: no {name}')
    variable = handle.variables[name]
    if name in _UNITS and _unit(variable) != _UNITS[name]:
        raise ValueError(f'{path}: {name} has unit {_unit(variable)!r}, not {_UNITS[name]!r}; '
                         'the reader takes only the units PSU-WOPWOP writes')
    return np.ma.filled(np.ma.asarray(variable[...], dtype=float), np.nan)


def read_pt_sphere(path):
    """Read a PSU-WOPWOP PT sphere: levels, phases and the rotor's kinematics.

    Returns a :class:`PTSphere` with phi, theta (deg), frequency (Hz, emission frame), amplitude (dB, (phi, theta,
    frequency)), phase (deg), radius_m, speed_kt, flight_path_angle_deg, center_m (forward, starboard, up),
    shaft_rate (rad/s, signed: positive counterclockwise), azimuth_deg, tag, rotor_radius_m, tip_speed_mps,
    air_density, floor_db, phase_reference_time_s, azimuth_reference.  Refuses a sphere that is not PT, not in the
    flight-path frame, not in the emission frame (DOPPLER_SHIFT_REMOVED 1), or in units it does not know.
    """
    path = str(path)
    with Dataset(path) as handle:
        attributes = {name: handle.getncattr(name) for name in handle.ncattrs()}
        kind = attributes.get('sphere_type')
        if kind is None:
            kind = 'PT' if 'PT' in handle.variables and float(handle['PT'][...]) == 1.0 else None
        if kind != 'PT':
            raise ValueError(f'{path}: not a pure-tone (PT) sphere (sphere_type {kind!r})')
        if attributes.get('sphere_frame') != 'flight_path':
            raise ValueError(f'{path}: sphere_frame is {attributes.get("sphere_frame")!r}; a PT database keeps '
                             'the spheres in their flight_path frame (pitching to level would re-interpolate phase)')
        if 'DOPPLER_SHIFT_REMOVED' not in handle.variables or float(handle['DOPPLER_SHIFT_REMOVED'][...]) != 1.0:
            raise ValueError(f'{path}: the tones must be in the emission frame (DOPPLER_SHIFT_REMOVED 1): NICEOPS '
                             'Doppler-shifts them itself')
        tag = attributes.get('source_tag')
        if not tag:
            raise ValueError(f'{path}: no source_tag: a PT database holds one sphere per rotor, named by its tag')
        rotation = str(attributes.get('main_rotor_rotation', 'counterclockwise')).strip().lower()
        if rotation not in ('counterclockwise', 'clockwise'):
            raise ValueError(f'{path}: main_rotor_rotation {rotation!r} is neither counterclockwise nor clockwise')
        rate = float(_value(handle, 'ROTOR_SHAFT_RATE', path))
        sphere = PTSphere(
            path=path, name=os.path.basename(path), tag=str(tag),
            phi=_value(handle, 'PHI', path), theta=_value(handle, 'THETA', path),
            frequency=_value(handle, 'FREQUENCY', path), amplitude=_value(handle, 'AMPLITUDE', path),
            phase=_value(handle, 'PHASE', path), radius_m=float(_value(handle, 'RADIUS', path)) * FOOT_M,
            speed_kt=float(_value(handle, 'SPEED', path)),
            flight_path_angle_deg=float(_value(handle, 'FLIGHT_PATH_ANGLE', path)),
            shaft_rate=abs(rate) if rotation == 'counterclockwise' else -abs(rate),
            azimuth_deg=float(_value(handle, 'ROTOR_AZIMUTH', path)),
            rotor_radius_m=float(_value(handle, 'MAIN_ROTOR_RADIUS', path)),
            tip_speed_mps=float(_value(handle, 'MAIN_ROTOR_TIP_SPEED', path)),
            air_density=float(_value(handle, 'AIR_DENSITY', path)),
            phase_reference_time_s=float(attributes.get('phase_reference_time_s', 'nan')),
            azimuth_reference=str(attributes.get('azimuth_reference', 'track')),
            speed_reference=str(attributes['speed_reference']) if 'speed_reference' in attributes else None,
            sound_speed_mps=float(attributes['speed_of_sound_m_s']) if 'speed_of_sound_m_s' in attributes
            else float('nan'))
        xyz = _value(handle, 'XYZ', path).reshape(-1)
        # AAM body axes of the flight-path frame (aft, starboard, up; ft) to the database's forward, starboard, up (m).
        sphere.center_m = np.array([-xyz[0], xyz[1], xyz[2]]) * FOOT_M
        floor = attributes.get('floor_level_dB')
        sphere.floor_db = float(floor) if floor is not None else float(np.nanmin(sphere.amplitude))
    if sphere.amplitude.shape != (sphere.phi.size, sphere.theta.size, sphere.frequency.size) \
            or sphere.phase.shape != sphere.amplitude.shape:
        raise ValueError(f'{path}: AMPLITUDE and PHASE must be (PHI, THETA, FREQUENCY)')
    return sphere


def measured_pt_sphere(tag, phi_deg, theta_deg, harmonics, level_db, phase_deg, *, radius_m, speed_kt,
                       flight_path_angle_deg, shaft_rate, center_m, rotor_radius_m, air_density, sound_speed_mps,
                       speed_reference='ground', phase_reference='rotor_azimuth', name=None, state=None):
    """A :class:`PTSphere` from measured tones (:func:`panam.rotor_tones.grid_tones`), for :func:`write_pt_database`.

    Args:
        tag: the source's name (one rotor).
        phi_deg, theta_deg: the grid, ART angles, deg; phi in [-180, 180), theta in [0, 180].
        harmonics: the shaft orders, (n_tones,).
        level_db: (n_phi, n_theta, n_tones) dB re 20 uPa (the tone's rms) at the sphere radius; -inf where the
            measurement has no value (NICEOPS fills such cells from their neighbors).
        phase_deg: the same shape: the cosine phase at emission with the rotor's azimuth zero (its blades at their
            reference); ignored where the level is -inf.
        radius_m: the sphere radius, m.  speed_kt, flight_path_angle_deg: the condition.
        shaft_rate: rad/s, positive counterclockwise seen from above (the mean over the condition's runs).
        center_m: the hub in the sphere frame (forward, starboard, up), m.
        rotor_radius_m, air_density (kg m-3), sound_speed_mps: the rotor and the air of the condition.
        speed_reference: 'ground' or 'air': what speed_kt is.
        phase_reference: 'rotor_azimuth' (the phases are locked to the rotor's measured angle), 'relative' or
            'none'.
        name: the source's label in the database (``source_sphere``).
        state: {variable: (value, units, description)} written beside the source: the rotor's own measured state
            (a scalar, or a vector over the sphere frame's xyz).

    The azimuth the phases are locked to is 0, and the source records no phase reference time: measured rotors turn
    free, so no two share a sync group unless one is declared.
    """
    level = np.asarray(level_db, dtype=float)
    phase = np.asarray(phase_deg, dtype=float)
    phi = np.asarray(phi_deg, dtype=float)
    theta = np.asarray(theta_deg, dtype=float)
    harmonics = np.asarray(harmonics, dtype=int)
    if level.shape != (phi.size, theta.size, harmonics.size) or phase.shape != level.shape:
        raise ValueError(f'level_db and phase_deg must be (n_phi, n_theta, n_tones) = '
                         f'({phi.size}, {theta.size}, {harmonics.size}), not {level.shape} and {phase.shape}')
    if np.any(np.isnan(level)) or np.any(level == np.inf):
        raise ValueError('level_db must be finite or -inf (no data)')
    if np.any(~np.isfinite(phase) & np.isfinite(level)):
        raise ValueError('phase_deg must be finite wherever level_db is')
    if phase_reference not in ('rotor_azimuth', 'relative', 'none'):
        raise ValueError(f"phase_reference must be 'rotor_azimuth', 'relative' or 'none', not {phase_reference!r}")
    shaft_hz = abs(shaft_rate) / (2.0 * math.pi)
    return PTSphere(
        path=name or tag, name=name or tag, tag=str(tag), phi=phi, theta=theta,
        frequency=harmonics * shaft_hz, harmonics=harmonics, amplitude=level,
        phase=np.where(np.isfinite(level), phase, 0.0), radius_m=float(radius_m), speed_kt=float(speed_kt),
        flight_path_angle_deg=float(flight_path_angle_deg), shaft_rate=float(shaft_rate), azimuth_deg=0.0,
        rotor_radius_m=float(rotor_radius_m), tip_speed_mps=abs(float(shaft_rate)) * float(rotor_radius_m),
        air_density=float(air_density), phase_reference_time_s=float('nan'), azimuth_reference='track',
        speed_reference=speed_reference, sound_speed_mps=float(sound_speed_mps),
        center_m=np.asarray(center_m, dtype=float), floor_db=-np.inf, phase_reference=phase_reference,
        state=dict(state or {}))


#: How far below its peak a tone's incoherent part is written where its fit is zero, dB.
INCOHERENT_FLOOR_DB = -100.0


def fitted_pt_sphere(tag, phi_deg, theta_deg, directivities, *, degree, shaft_rate, model=None, incoherent=None,
                     incoherent_knots_deg=None, **kwargs):
    """A :class:`PTSphere` from fitted spherical-harmonic directivities (:func:`panam.rotor_tones.fit_directivities`
    or :func:`~panam.rotor_tones.fit_summed_directivities`), evaluated on the grid, for :func:`write_pt_database`.

    Args:
        tag: the source's name (one rotor).
        phi_deg, theta_deg: the grid, ART angles, deg.
        directivities: {shaft order m: ((degree + 1)^2,) complex coefficients}: the tone's complex pressure
            amplitude (peak, Pa) at the sphere radius, in :func:`~panam.rotor_tones.harmonic_basis` about the
            sphere frame's vertical (:func:`~panam.rotor_tones.axis_angles`), its phase locked to the rotor's azimuth.
        degree: the expansion's degree.
        shaft_rate: rad/s, positive counterclockwise seen from above: the speed the directivities refer to.  A
            database written with ``tone_speed_exponent`` scales each tone with the run's speed over this one.
        model: {shaft order m: azimuthal orders k fitted (a list)}, or a description (str), kept beside the
            coefficients; optional.
        incoherent, incoherent_knots_deg: {shaft order m: (n_knots,) energy coefficients, Pa^2} and their knots
            (:func:`panam.rotor_tones.fit_incoherent_directivities`): each tone's part without a stable phase,
            written as ``incoherent_amplitude`` (dB rms; INCOHERENT_FLOOR_DB below its peak where the fit is zero,
            quiet rather than missing; -inf throughout for an order without one) with the coefficients beside it
            (``incoherent_energy``).  Optional; an order it does not name has none.
        kwargs: as :func:`measured_pt_sphere` (radius_m, speed_kt, flight_path_angle_deg, center_m,
            rotor_radius_m, air_density, sound_speed_mps, ...).

    The coefficients are written beside the grid (``directivity_real``, ``directivity_imag``) as the source's
    provenance: the grid is what NICEOPS evaluates.  A direction where the expansion is exactly zero is -inf dB.
    """
    from panam import rotor_tones as rt
    harmonics = np.array(sorted(directivities), dtype=int)
    n = (int(degree) + 1) ** 2
    coefficients = np.array([np.asarray(directivities[m], dtype=complex) for m in harmonics])
    if coefficients.shape != (harmonics.size, n):
        raise ValueError(f'each directivity must hold (degree + 1)^2 = {n} coefficients, not {coefficients.shape[1:]}')
    if not np.all(np.isfinite(coefficients)):
        raise ValueError('the directivity coefficients must be finite')
    gp, gt = np.meshgrid(np.asarray(phi_deg, float), np.asarray(theta_deg, float), indexing='ij')
    values = np.stack([rt.evaluate_directivity(c, degree, gp, gt) for c in coefficients], axis=-1)
    with np.errstate(divide='ignore'):
        level = 20.0 * np.log10(np.abs(values) / math.sqrt(2.0) / rt.P_REF)
    phase = np.degrees(np.angle(values))
    sphere = measured_pt_sphere(tag, phi_deg, theta_deg, harmonics, level, phase, shaft_rate=shaft_rate, **kwargs)
    sphere.directivity = {'harmonics': harmonics, 'coefficients': coefficients, 'degree': int(degree),
                          'model': model}
    if incoherent:
        if incoherent_knots_deg is None:
            raise ValueError('incoherent needs incoherent_knots_deg')
        knots = np.asarray(incoherent_knots_deg, dtype=float)
        if set(incoherent) - set(int(m) for m in harmonics):
            raise ValueError(f'incoherent names orders {sorted(set(incoherent) - set(harmonics))} that the '
                             'directivities do not')
        energy = np.zeros((harmonics.size, knots.size))
        for i, m in enumerate(harmonics):
            if m in incoherent:
                energy[i] = np.asarray(incoherent[m], dtype=float)
        if energy.shape[1] != knots.size or np.any(~np.isfinite(energy)) or np.any(energy < 0):
            raise ValueError('each incoherent energy directivity must hold one finite value >= 0 per knot')
        mean_square = np.stack([rt.evaluate_energy_directivity(e, knots, gp, gt) for e in energy], axis=-1)
        # Where the fit is exactly zero (a non-negative fit sets whole ranges of knots to 0) the part is quiet, not
        # missing: NICEOPS fills a -inf cell from its neighbors, which would put energy where the fit has none.
        # Such cells take INCOHERENT_FLOOR_DB below the tone's peak; a tone with none at all stays -inf.
        peak = mean_square.max(axis=(0, 1), keepdims=True)
        floored = np.where(peak > 0, np.maximum(mean_square, peak * 10.0 ** (INCOHERENT_FLOOR_DB / 10.0)), 0.0)
        with np.errstate(divide='ignore'):
            sphere.incoherent_db = 10.0 * np.log10(floored / 2.0 / rt.P_REF ** 2)
        sphere.incoherent = {'harmonics': harmonics, 'energy': energy, 'knots_deg': knots}
    return sphere


def _close_seam(sphere):
    """Drop a full sphere's +180 phi row when it repeats the -180 row in level and phase (the database keeps phi in
    [-180, 180)); refuse a +180 row that differs."""
    if sphere.phi.size < 2 or not (sphere.phi[0] == -180.0 and sphere.phi[-1] == 180.0):
        return sphere
    a0, a1 = sphere.amplitude[0], sphere.amplitude[-1]
    lit = (a0 > sphere.floor_db) | (a1 > sphere.floor_db)
    same_level = np.allclose(a0[lit], a1[lit], atol=1e-3)
    turn = np.exp(1j * np.radians(sphere.phase[0][lit] - sphere.phase[-1][lit]))
    same_phase = not lit.any() or np.max(np.abs(turn - 1.0)) < np.radians(0.05)
    if not (same_level and same_phase):
        raise ValueError(f'{sphere.path}: the full sphere\'s +180 and -180 phi rows differ at the seam; they are '
                         'one direction, so the sphere is inconsistent there')
    sphere.phi, sphere.amplitude, sphere.phase = sphere.phi[:-1], sphere.amplitude[:-1], sphere.phase[:-1]
    return sphere


def _revolutions(sphere, shaft_hz, harmonic_tolerance, most=64):
    """N when every FREQUENCY is a whole multiple of shaft_hz / N (PSU-WOPWOP's spectrum over N revolutions), the
    smallest such N up to `most`; None when there is none."""
    for n in range(1, most + 1):
        bins = np.rint(sphere.frequency * n / shaft_hz)
        if np.all(np.abs(sphere.frequency - bins * shaft_hz / n) <= harmonic_tolerance * np.maximum(bins * shaft_hz / n,
                                                                                                     1e-300)):
            return n
    return None


def _tones(sphere, harmonic_tolerance=1e-5):
    """The sphere's shaft orders that carry energy above its floor: (harmonics, frequencies Hz, amplitude dB,
    phase deg locked to the rotor azimuth).  A kept tone's cells at the floor keep the floor level (quiet, not
    missing).  Every listed frequency must be a shaft order, or (a spectrum over N revolutions) a whole multiple of
    1/N of one: then only the whole orders are tones, and the bins between them (leakage, a loading's slow drift)
    are dropped; their share of the energy is set on the sphere as ``dropped_energy_fraction`` (and the number of
    revolutions as ``revolutions``)."""
    shaft_hz = abs(sphere.shaft_rate) / (2.0 * math.pi)
    if getattr(sphere, 'harmonics', None) is not None:
        # Shaft orders given (a measured source): no floor, -inf where a cell has no data; every given order with a
        # value anywhere is kept.
        harmonics = np.asarray(sphere.harmonics, dtype=int)
        keep = np.isfinite(sphere.amplitude).any(axis=(0, 1))
        harmonics = harmonics[keep]
        sphere.dropped_energy_fraction = 0.0
        locked = sphere.phase[:, :, keep] - harmonics[None, None, :] * sphere.azimuth_deg
        locked = np.where(np.isfinite(locked), (locked + 180.0) % 360.0 - 180.0, np.nan)
        return harmonics, harmonics * shaft_hz, sphere.amplitude[:, :, keep], locked
    n_rev = _revolutions(sphere, shaft_hz, harmonic_tolerance)
    if n_rev is None:
        orders = np.rint(sphere.frequency / shaft_hz).astype(int)
        off = np.abs(sphere.frequency - orders * shaft_hz) > harmonic_tolerance * np.maximum(orders * shaft_hz, 1e-300)
        raise ValueError(f'{sphere.path}: FREQUENCY {sphere.frequency[off][:3]} Hz is not a whole shaft order of '
                         f'ROTOR_SHAFT_RATE ({shaft_hz:.6g} Hz), nor a whole fraction of one')
    bins = np.rint(sphere.frequency * n_rev / shaft_hz).astype(int)
    whole = (bins % n_rev == 0) & (bins > 0)
    energy = 10.0 ** (np.where(sphere.amplitude > sphere.floor_db + 1e-3, sphere.amplitude, -np.inf) / 10.0)
    total = energy.sum()
    dropped = float(energy[:, :, ~whole].sum() / total) if total > 0 else 0.0
    sphere.dropped_energy_fraction = dropped
    sphere.revolutions = n_rev
    sphere.frequency = sphere.frequency[whole]
    sphere.amplitude = sphere.amplitude[:, :, whole]
    sphere.phase = sphere.phase[:, :, whole]
    orders = bins[whole] // n_rev
    # A tone at PSU-WOPWOP's floor (floor_level_dB) everywhere has no energy and is left out.  Where a kept tone is at
    # the floor in some cells, the floor stays: a predicted cell is quiet, not missing, and NICEOPS drops a -inf
    # entry from its interpolation and renormalizes the rest, which would fill a directivity null with its
    # neighbors' level (measured on the tandem: +15 to +33 dB in bands made of such tones).
    lit = sphere.amplitude > sphere.floor_db + 1e-3
    keep = lit.any(axis=(0, 1))
    amplitude = sphere.amplitude[:, :, keep]
    harmonics = orders[keep]
    # Locked to the rotor: the stored phase is the tone's less m times the azimuth it was predicted at, so a rotor
    # turned dpsi ahead (e^{+i m dpsi}) is the same stored sphere.
    locked = sphere.phase[:, :, keep] - harmonics[None, None, :] * sphere.azimuth_deg
    locked = (locked + 180.0) % 360.0 - 180.0
    return harmonics, harmonics * shaft_hz, amplitude, locked


def _merge_close_speeds(spheres, tolerance_kt):
    """Put spheres whose speeds are within tolerance_kt of a cluster's first (sorted by speed) and whose flight path
    angles are within 1e-3 deg of its first, at the cluster's mean speed and mean flight path angle: one
    condition."""
    clusters = []
    for sphere in sorted(spheres, key=lambda s: (round(s.flight_path_angle_deg, 3), s.speed_kt)):
        c = clusters[-1] if clusters else None
        if c and abs(sphere.flight_path_angle_deg - c[0].flight_path_angle_deg) <= 1e-3 \
                and sphere.speed_kt - c[0].speed_kt <= tolerance_kt:
            c.append(sphere)
        else:
            clusters.append([sphere])
    for c in clusters:
        speed = float(np.mean([s.speed_kt for s in c]))
        angle = float(np.mean([s.flight_path_angle_deg for s in c]))
        for s in c:
            s.speed_kt, s.flight_path_angle_deg = speed, angle


def _condition_key(sphere):
    return round(sphere.speed_kt, 6), round(sphere.flight_path_angle_deg, 6)


def build_pt_database(sphere_files, database_filename, *, weight_newtons=None, sync_groups='shared_rate',
                      main_rotor=None, speed_reference=None, condition_axes='helicopter', rotor_count=None,
                      hover_thrust_coefficient=None, sound_speed_mps=None, speed_tolerance_kt=None):
    """Build a format 2 database from PSU-WOPWOP PT spheres, one per rotor per flight condition.

    Args:
        sphere_files: the PT sphere files (one per rotor and condition).
        database_filename: the .nod written (replaced if it exists).
        weight_newtons: the vehicle's weight, N, for the thrust coefficient axis (C_T = W / (rho pi R^2 V_tip^2) with
            the sphere's own AIR_DENSITY, so ``thrust_coefficient_density = run``).  Required: PSU-WOPWOP's spheres
            carry no weight.
        sync_groups: ``'shared_rate'`` (default): rotors predicted at one shaft rate and one phase reference time
            turn together and form one coherent group (``sync_group = shaft``); None: every rotor free-running; or a
            {tag: group name} dict ('' for free-running).  Measured rotors are never grouped by equal speed alone;
            a prediction's are locked by construction.
        main_rotor: the tag whose MAIN_ROTOR_* define the database's rotor (advance ratio, C_T, sphere radius
            scale); default the largest radius (the first tag on a tie).  A helicopter's main rotor must have one
            radius and tip speed at every condition; a multirotor's one radius (its speed varies by design).
        speed_reference: 'air' or 'ground', for spheres that do not record one; default what the spheres record,
            and 'air' (PSU-WOPWOP's convention) where none does.  Spheres that record different references, or one
            other than this argument, are refused.
        condition_axes: ``'helicopter'`` (advance ratio, flight path angle, thrust coefficient) or ``'multirotor'``:
            momentum similarity, with v_h = sqrt(W / (2 rho N pi R^2)) and
            Omega_h R = sqrt(W / (N rho pi R^2 C_Th)), each condition at in-plane and axial air speed over v_h
            (``inplane_speed_ratio``, ``axial_speed_ratio``, the thrust axis taken as vertical) and the hover tip
            Mach number Omega_h R / c (``hover_tip_mach``), with ``air_density``, ``weight_n``, ``airspeed`` and
            ``climb_rate``; each source's ``speed_ratio`` = |shaft_rate| / Omega_h.  Every rotor's sphere must give
            one rotor radius.
        rotor_count: N, the vehicle's rotors (multirotor; the database may hold fewer sources).
        hover_thrust_coefficient: C_Th per rotor (multirotor).  Default: from the hover condition (speed 0), with
            each rotor carrying W / N at the rms of the sources' shaft rates.
        sound_speed_mps: c for hover_tip_mach where a sphere does not record ``speed_of_sound_m_s``.
        speed_tolerance_kt: spheres whose speeds lie within this of each other (and flight path angles within 1e-3 deg)
            are one condition, at their mean speed and mean angle; each source records its own sphere's speed
            (``sphere_speed_kt``).
            Default: exact (to 1e-6 kt).  For per-rotor predictions of one flight made over different windows.

    The database holds every flight condition once, at load factor 1.  A helicopter database whose spheres share one
    air density has one C_T and is ``fixed_load_factor``: NICEOPS scales the levels to other load factors.  Spheres
    predicted in different air give each condition the C_T of its own density, so C_T is an axis there and the
    database is not ``fixed_load_factor`` (one C_T is what that flag promises).  Raises ValueError, before writing anything, for spheres that are not PT, not in
    the flight-path frame, on different grids within a condition, two spheres of one tag at one condition, a
    condition missing a source another has, or tones off their shaft orders.
    """
    if weight_newtons is None or not np.isfinite(weight_newtons) or weight_newtons <= 0:
        raise ValueError('weight_newtons is required (N, > 0): PSU-WOPWOP spheres carry no vehicle weight, and the '
                         'thrust coefficient axis needs one')
    if condition_axes not in ('helicopter', 'multirotor'):
        raise ValueError(f"condition_axes must be 'helicopter' or 'multirotor', not {condition_axes!r}")
    if condition_axes == 'multirotor' and (rotor_count is None or int(rotor_count) != rotor_count or rotor_count < 1):
        raise ValueError('a multirotor database needs rotor_count (the vehicle\'s number of rotors, >= 1)')
    spheres = [_close_seam(read_pt_sphere(path)) for path in sphere_files]
    if not spheres:
        raise ValueError('no PT spheres given')
    return write_pt_database(
        spheres, database_filename, weight_newtons=weight_newtons, sync_groups=sync_groups, main_rotor=main_rotor,
        speed_reference=speed_reference, condition_axes=condition_axes, rotor_count=rotor_count,
        hover_thrust_coefficient=hover_thrust_coefficient, sound_speed_mps=sound_speed_mps,
        speed_tolerance_kt=speed_tolerance_kt, tool='pt_database.build_pt_database',
        input_names=[os.path.basename(str(p)) for p in sphere_files])


def write_pt_database(spheres, database_filename, *, weight_newtons, sync_groups='shared_rate', main_rotor=None,
                      speed_reference=None, condition_axes='helicopter', rotor_count=None,
                      hover_thrust_coefficient=None, sound_speed_mps=None, speed_tolerance_kt=None,
                      azimuth_label=None, tool='pt_database.write_pt_database', input_names=None,
                      tone_speed_exponent=None):
    """Write a format 2 database from :class:`PTSphere` sources, one per rotor per condition: what
    :func:`build_pt_database` does once it has read PSU-WOPWOP's files, open to sources built otherwise (measured
    ones: :func:`measured_pt_sphere`).  Arguments as :func:`build_pt_database`, and:

        azimuth_label: written as the root's ``azimuth_label`` when given: what the sources' azimuths are relative to
            (a measured database's ``relative_within_sense``: its common timing offset is unknown, which relabels
            every rotor's blade angle by the same amount in its own sense of rotation).
        tool, input_names: the provenance stamp's tool and input names.
        tone_speed_exponent: written as the root's ``tone_speed_exponent`` when given (>= 0): NICEOPS scales each
            tone's pressure by the run's shaft rate over the source's ``shaft_rate`` to this power (3: the level as
            rpm to the sixth), where a run turns a rotor at another speed (a replayed history, a sync group).
    """
    if weight_newtons is None or not np.isfinite(weight_newtons) or weight_newtons <= 0:
        raise ValueError('weight_newtons is required (N, > 0) for the thrust coefficient axis')
    if tone_speed_exponent is not None and not (np.isfinite(tone_speed_exponent) and tone_speed_exponent >= 0):
        raise ValueError(f'tone_speed_exponent must be finite and >= 0, not {tone_speed_exponent}')
    if condition_axes not in ('helicopter', 'multirotor'):
        raise ValueError(f"condition_axes must be 'helicopter' or 'multirotor', not {condition_axes!r}")
    if condition_axes == 'multirotor' and (rotor_count is None or int(rotor_count) != rotor_count or rotor_count < 1):
        raise ValueError('a multirotor database needs rotor_count (the vehicle\'s number of rotors, >= 1)')
    spheres = list(spheres)
    if not spheres:
        raise ValueError('no PT spheres given')
    sphere_files = input_names if input_names is not None else [s.name for s in spheres]
    for sphere in spheres:
        sphere.own_speed_kt = sphere.speed_kt
    if speed_tolerance_kt is not None:
        _merge_close_speeds(spheres, float(speed_tolerance_kt))
    conditions = OrderedDict()
    for sphere in sorted(spheres, key=lambda s: (_condition_key(s), s.tag)):
        by_tag = conditions.setdefault(_condition_key(sphere), OrderedDict())
        if sphere.tag in by_tag:
            raise ValueError(f'{by_tag[sphere.tag].name} and {sphere.name} are both source {sphere.tag!r} at '
                             f'{sphere.speed_kt} kt, {sphere.flight_path_angle_deg} deg: one source per rotor per '
                             'condition')
        by_tag[sphere.tag] = sphere
    tags = list(next(iter(conditions.values())).keys())
    for key, by_tag in conditions.items():
        if list(by_tag) != tags:
            raise ValueError(f'the condition at {key[0]} kt, {key[1]} deg has sources {list(by_tag)}, another has '
                             f'{tags}: every condition must carry every source')
    first = spheres[0]
    for s in spheres:
        if s.phi.shape != first.phi.shape or not np.array_equal(s.phi, first.phi) \
                or s.theta.shape != first.theta.shape or not np.array_equal(s.theta, first.theta):
            raise ValueError(f'{s.name} is on a different (PHI, THETA) grid from {first.name}: the database shares '
                             'one grid and regridding would damage the phase')
        if not np.isclose(s.radius_m, first.radius_m, rtol=1e-9):
            raise ValueError(f'{s.name} has RADIUS {s.radius_m} m, {first.name} {first.radius_m} m: one sphere radius')
    if np.any(first.phi < -180.0) or np.any(first.phi >= 180.0) or np.any(first.theta < 0) \
            or np.any(first.theta > 180.0):
        raise ValueError(f'{first.name}: PHI must lie in [-180, 180) and THETA in [0, 180] deg')
    if main_rotor is not None and main_rotor not in tags:
        raise ValueError(f'main_rotor {main_rotor!r} is not one of the sources {tags}')
    main = main_rotor or max(tags, key=lambda t: (conditions[next(iter(conditions))][t].rotor_radius_m,
                                                  -tags.index(t)))
    rotor_radius = conditions[next(iter(conditions))][main].rotor_radius_m
    tip_speed = conditions[next(iter(conditions))][main].tip_speed_mps
    for by_tag in conditions.values():
        s = by_tag[main]
        if not np.isclose(s.rotor_radius_m, rotor_radius, rtol=1e-6):
            raise ValueError(f'{s.name}: the main rotor ({main}) has radius {s.rotor_radius_m} m, other conditions '
                             f'{rotor_radius} m')
        # A multirotor's rotor speed changes with the condition by design (speed_ratio); a helicopter's does not.
        if condition_axes == 'helicopter' and not np.isclose(s.tip_speed_mps, tip_speed, rtol=1e-6):
            raise ValueError(f'{s.name}: the main rotor ({main}) has tip speed {s.tip_speed_mps} m/s, other '
                             f'conditions {tip_speed} m/s: a helicopter database has one')
    speed_reference = _speed_reference(spheres, speed_reference)
    azimuth_references = {s.azimuth_reference for s in spheres}
    if len(azimuth_references) != 1:
        raise ValueError(f'the spheres record azimuth_reference {sorted(azimuth_references)}; they must agree')

    # One air density: one C_T, and the database is fixed_load_factor.  Several: each condition's C_T at its own
    # density, an axis NICEOPS interpolates over; fixed_load_factor would promise one C_T (validate_nod's C8).
    densities = [by_tag[main].air_density for by_tag in conditions.values()]
    one_density = all(np.isclose(rho, densities[0], rtol=1e-9, atol=0.0) for rho in densities)
    multirotor = None
    if condition_axes == 'multirotor':
        multirotor = _multirotor_axes(conditions, tags, main, float(weight_newtons), int(rotor_count),
                                      hover_thrust_coefficient, sound_speed_mps)
    group_of = _sync_groups(conditions, tags, sync_groups)
    phi_grid, theta_grid = np.meshgrid(first.phi, first.theta, indexing='ij')
    n_bands = BAND_CENTERS_HZ.size
    empty_bands = np.full(phi_grid.shape + (n_bands,), -np.inf)
    settings = {'weight_newtons': float(weight_newtons), 'sync_groups': sync_groups if not isinstance(
        sync_groups, dict) else dict(sync_groups), 'main_rotor': main, 'speed_reference': speed_reference,
        'condition_axes': condition_axes, 'rotor_count': rotor_count,
        'hover_thrust_coefficient': hover_thrust_coefficient, 'sound_speed_mps': sound_speed_mps,
        'speed_tolerance_kt': speed_tolerance_kt, 'tone_speed_exponent': tone_speed_exponent}
    inputs = {'sphere_files': sorted(str(p) for p in sphere_files),
              'database_filename': _provenance.path_name(database_filename)}
    with _atomic_output(database_filename) as temporary, Dataset(temporary, 'w') as database:
        _write_database_root(
            # Multirotor: not fixed_load_factor (NICEOPS's load-factor law is in C_T, which multirotor axes do not
            # carry; it refuses the flag there).
            database, fixed_load_factor=multirotor is None and one_density, speed_reference=speed_reference,
            atmosphere=Atmosphere(temperature=293.15, pressure=101.325, relative_humidity=20.0),
            main_rotor_radius=rotor_radius, main_rotor_tip_speed=tip_speed, vehicle_weight_newtons=weight_newtons,
            azimuth_reference=azimuth_references.pop(),
            provenance=_provenance.build_provenance(tool, settings, inputs=inputs))
        database['database_version'][:] = PT_DATABASE_FORMAT_VERSION
        database.sphere_frame = 'flight_path'
        database.sphere_levels_lossless = np.float64(1.0)
        database.band_content = 'broadband'
        database.condition_axes = condition_axes
        if multirotor is not None:
            database.rotor_count = np.int32(rotor_count)
            database.rotor_radius_m = np.float64(multirotor['radius'])
            database.hover_thrust_coefficient = np.float64(multirotor['ct_h'])
        database.thrust_coefficient_density = 'run'
        if azimuth_label is not None:
            database.azimuth_label = str(azimuth_label)
        if tone_speed_exponent is not None:
            database.tone_speed_exponent = np.float64(tone_speed_exponent)
        database.sources = ','.join(tags)
        database.main_rotor_source = main
        pending, key_of = [], {}
        for i, (key, by_tag) in enumerate(conditions.items()):
            key_of['sphere' + str(i)] = key
            s = by_tag[main]
            coverage = np.zeros(empty_bands.shape, dtype=bool)
            pending.append(_SphereCondition(
                'sphere' + str(i), phi_grid, theta_grid, s.radius_m, np.full(phi_grid.shape, -np.inf),
                np.zeros(phi_grid.shape), s.speed_kt, s.flight_path_angle_deg, 1.0, BAND_CENTERS_HZ, empty_bands,
                coverage, None))
        shared = _write_shared_grid_and_frequency(database, pending)
        for condition, by_tag in zip(pending, conditions.values()):
            s = by_tag[main]
            # C_T at the condition's own density (the prediction's), the database's weight: thrust_coefficient_density
            # run (one value when the spheres share one density).  add_sphere_group writes load_factor x
            # weight_coefficient.
            weight_coefficient = weight_newtons / (s.air_density * math.pi * rotor_radius ** 2 * tip_speed ** 2)
            axes = multirotor['conditions'][key_of[condition.name]] if multirotor is not None else None
            add_sphere_group(database, condition.name, condition.phi, condition.theta, condition.radius,
                                condition.SPLA, condition.EAA, condition.speed, condition.flight_path_angle, 1.0,
                                rotor_radius, tip_speed, weight_coefficient, condition.frequency,
                                condition.amplitude, write_grid_and_frequency=not shared, doppler_shift_removed=0,
                                coverage=condition.coverage,
                                condition_variables=axes['variables'] if axes is not None else None)
            # DOPPLER_SHIFT_REMOVED applies to the band spectrum, which is empty here (received frame by
            # convention); the tones are emission-frame shaft orders by definition, and NICEOPS shifts them.
            group = database.groups[condition.name]
            # No condition_origin: a predicted condition is a stored condition of its own, as a measured one is
            # (NICEOPS reads a group without one so); the root's panam_tool names the prediction build.
            group.air_density = np.float64(s.air_density)
            group.air_density_units = 'kg m-3'
            for tag in tags:
                _write_source(group, by_tag[tag], group_of.get(tag, ''),
                              speed_ratio=axes['speed_ratio'][tag] if axes is not None else None)


def _speed_reference(spheres, requested):
    """The database's speed reference: what the spheres record ('air' or 'ground'), or the requested one for spheres
    that record none ('air', PSU-WOPWOP's convention, by default).  Spheres that record different references, or one
    other than the request, are refused: relabeling air speeds as ground speeds would misplace every condition."""
    if requested is not None and requested not in SPEED_REFERENCES:
        raise ValueError(f'speed_reference {requested!r} is not one of {SPEED_REFERENCES}')
    recorded = {s.speed_reference for s in spheres if s.speed_reference is not None}
    unknown = recorded - set(SPEED_REFERENCES)
    if unknown:
        raise ValueError(f'the spheres record speed_reference {sorted(unknown)}, not one of {SPEED_REFERENCES}')
    if len(recorded) > 1:
        raise ValueError(f'the spheres record speed_reference {sorted(recorded)}: one database has one reference')
    if recorded:
        found = recorded.pop()
        if requested is not None and requested != found:
            raise ValueError(f'speed_reference {requested!r} was given, but the spheres record {found!r}')
        return found
    return requested or 'air'


def _sync_groups(conditions, tags, sync_groups):
    """{tag: group name} for the coherent groups (absent or '' = free-running).

    'shared_rate': every set of two or more rotors with the same |shaft rate| and phase reference time at every
    condition is one group, named ``shaft`` when there is one such set, else ``shaft_<lowest tag>``.  A rotor whose
    sphere records no phase reference time cannot be shown to be locked to another, so it is left free-running
    (with a warning)."""
    if sync_groups is None:
        return {}
    if isinstance(sync_groups, dict):
        unknown = set(sync_groups) - set(tags)
        if unknown:
            raise ValueError(f'sync_groups names {sorted(unknown)}, which are not sources ({tags})')
        return {tag: str(name) for tag, name in sync_groups.items() if name}
    if sync_groups != 'shared_rate':
        raise ValueError(f"sync_groups must be 'shared_rate', None or a {{tag: group}} dict, not {sync_groups!r}")
    untimed = [t for t in tags if any(not np.isfinite(by_tag[t].phase_reference_time_s)
                                      for by_tag in conditions.values())]
    if untimed:
        warnings.warn(f'sources {untimed} record no phase_reference_time_s: they cannot be shown to turn locked to '
                      'another rotor, so they are left free-running', UserWarning, stacklevel=3)
    # Rotors of one prediction at one |shaft rate| and one phase reference time, at every condition.
    signatures = OrderedDict()
    for t in tags:
        if t in untimed:
            continue
        key = tuple((round(abs(by_tag[t].shaft_rate), 9), round(by_tag[t].phase_reference_time_s, 9))
                    for by_tag in conditions.values())
        signatures.setdefault(key, []).append(t)
    sets = [sorted(members) for members in signatures.values() if len(members) >= 2]
    if len(sets) == 1:
        return {tag: 'shaft' for tag in sets[0]}
    return {tag: f'shaft_{members[0]}' for members in sets for tag in members}


def _multirotor_axes(conditions, tags, main, weight, rotor_count, hover_thrust_coefficient, sound_speed_mps):
    """Momentum-similarity axes of every condition (see :func:`build_pt_database`): {'radius', 'ct_h',
    'conditions': {key: {'variables': {name: (value, units)}, 'speed_ratio': {tag: |shaft_rate| / Omega_h}}}}."""
    radii = {round(by_tag[t].rotor_radius_m, 9) for by_tag in conditions.values() for t in tags}
    if len(radii) != 1:
        raise ValueError(f'the rotors\' spheres give MAIN_ROTOR_RADIUS {sorted(radii)} m: a multirotor database has one '
                         'rotor radius')
    radius = radii.pop()
    if hover_thrust_coefficient is None:
        hovers = [by_tag for key, by_tag in conditions.items() if abs(key[0]) < 1e-9]
        if not hovers:
            raise ValueError('no hover condition (speed 0) to take the hover thrust coefficient from; give '
                             'hover_thrust_coefficient')
        by_tag = hovers[0]
        rms = math.sqrt(np.mean([by_tag[t].shaft_rate ** 2 for t in tags]))
        rho = by_tag[main].air_density
        hover_thrust_coefficient = weight / (rotor_count * rho * math.pi * radius ** 2 * (rms * radius) ** 2)
    if not np.isfinite(hover_thrust_coefficient) or hover_thrust_coefficient <= 0:
        raise ValueError(f'hover_thrust_coefficient {hover_thrust_coefficient} is not positive')
    out = {'radius': radius, 'ct_h': float(hover_thrust_coefficient), 'conditions': {}}
    for key, by_tag in conditions.items():
        s = by_tag[main]
        c = s.sound_speed_mps if np.isfinite(s.sound_speed_mps) else sound_speed_mps
        if c is None or not np.isfinite(c) or c <= 0:
            raise ValueError(f'{s.name} records no speed of sound (speed_of_sound_m_s) for the hover tip Mach number; '
                             'give sound_speed_mps')
        rho = s.air_density
        v_h = hover_induced_velocity(weight, rho, rotor_count, radius)
        omega_h = hover_shaft_rate(weight, rho, rotor_count, radius, hover_thrust_coefficient)
        u = s.speed_kt * KNOT_MPS
        gamma = math.radians(s.flight_path_angle_deg)
        inplane, climb = u * math.cos(gamma), u * math.sin(gamma)
        out['conditions'][key] = {
            'variables': OrderedDict(zip(MULTIROTOR_AXES + MULTIROTOR_FIELDS, (
                (inplane / v_h, '1'), (climb / v_h, '1'), (omega_h * radius / c, '1'),
                (rho, 'kg m-3'), (weight, 'N'), (u, 'm s-1'), (climb, 'm s-1')))),
            'speed_ratio': {t: abs(by_tag[t].shaft_rate) / omega_h for t in tags}}
    return out


def _write_source(group, sphere, sync_group, speed_ratio=None):
    harmonics, frequencies, amplitude, phase = _tones(sphere)
    sub = group.createGroup(sphere.tag)
    sub.createDimension('tone', harmonics.size)
    sub.createDimension('xyz', 3)
    v = sub.createVariable('center', 'f8', ('xyz',))
    v[:] = sphere.center_m
    v.units = 'm'
    v.description = 'hub position in the sphere frame: forward, starboard, up'
    v = sub.createVariable('shaft_rate', 'f8')
    v[...] = sphere.shaft_rate
    v.units = 'rad s-1'
    v.description = 'positive counterclockwise seen from above'
    v = sub.createVariable('harmonic', 'i4', ('tone',))
    v[:] = harmonics
    v = sub.createVariable('frequency', 'f8', ('tone',))
    v[:] = frequencies
    v.units = 'Hz'
    v.description = 'emission frame: harmonic x |shaft_rate| / 2 pi'
    for name, values, units in (('amplitude', amplitude, 'dB re 20 uPa at the sphere radius'),
                                ('phase', phase, 'degree')):
        v = sub.createVariable(name, 'f8', ('PHI', 'THETA', 'tone'))
        v[:] = values
        v.units = units
    sub['phase'].description = 'cosine phase at emission from the hub, less harmonic x azimuth (rotor-locked)'
    v = sub.createVariable('dropped_energy_fraction', 'f8')
    v[...] = getattr(sphere, 'dropped_energy_fraction', 0.0)
    v.description = ('the share of the sphere\'s energy in the bins between whole shaft orders, dropped (a '
                     'spectrum over several revolutions)')
    v = sub.createVariable('azimuth', 'f8')
    v[...] = sphere.azimuth_deg
    v.units = 'degree'
    v.description = 'the rotor azimuth (its own sense of rotation) the phase was locked to'
    if speed_ratio is not None:
        v = sub.createVariable('speed_ratio', 'f8')
        v[...] = speed_ratio
        v.description = '|shaft_rate| over the condition\'s momentum hover speed Omega_h'
    sub.phase_reference = getattr(sphere, 'phase_reference', 'rotor_azimuth')
    for name, (value, units, description) in getattr(sphere, 'state', {}).items():
        value = np.asarray(value, dtype=float)
        if value.ndim == 0:
            v = sub.createVariable(name, 'f8')
            v[...] = value
        else:
            v = sub.createVariable(name, 'f8', ('xyz',))
            v[:] = value
        v.units = units
        v.description = description
    incoherent = getattr(sphere, 'incoherent', None)
    if incoherent is not None:
        rows = [int(np.flatnonzero(incoherent['harmonics'] == m)[0]) for m in harmonics]
        v = sub.createVariable('incoherent_amplitude', 'f8', ('PHI', 'THETA', 'tone'))
        v[:] = sphere.incoherent_db[:, :, rows]
        v.units = 'dB re 20 uPa at the sphere radius'
        v.description = ('rms level of what the rotor radiates at the shaft order without a stable phase; added in '
                         'energy, each source alone; -inf where none')
        sub.createDimension('knot', incoherent['knots_deg'].size)
        v = sub.createVariable('incoherent_energy', 'f8', ('tone', 'knot'))
        v[:] = incoherent['energy'][rows]
        v.units = 'Pa2'
        v.description = ('the fitted energy directivity incoherent_amplitude was evaluated from (provenance): mean '
                         'square peak pressure at the sphere radius, piecewise linear in the polar angle from the down '
                         'axis between the knots (knots_deg), flat beyond them')
        v.knots_deg = incoherent['knots_deg']
    directivity = getattr(sphere, 'directivity', None)
    if directivity is not None:
        rows = [int(np.flatnonzero(directivity['harmonics'] == m)[0]) for m in harmonics]
        coefficients = directivity['coefficients'][rows]
        sub.createDimension('coefficient', coefficients.shape[1])
        for part, values in (('real', coefficients.real), ('imag', coefficients.imag)):
            v = sub.createVariable('directivity_' + part, 'f8', ('tone', 'coefficient'))
            v[:] = values
            v.units = 'Pa'
        sub['directivity_real'].description = (
            'the fitted directivity the amplitude and phase were evaluated from (provenance): the complex peak '
            'pressure at the sphere radius, sum over l <= degree, k = -l..l (in that order) of c_lk Y_l^k(polar, '
            'azimuth), Y orthonormal complex spherical harmonics, polar from the down axis, azimuth from the nose '
            'toward starboard')
        sub['directivity_real'].degree = np.int32(directivity['degree'])
        model = directivity.get('model')
        if isinstance(model, dict):
            sub['directivity_real'].azimuthal_orders = '; '.join(
                f'{m}: ' + ','.join(str(int(k)) for k in model[m]) for m in harmonics if m in model)
        elif model:
            sub['directivity_real'].model = str(model)
    if sync_group:
        sub.sync_group = sync_group
    sub.source_sphere = sphere.name
    v = sub.createVariable('sphere_speed_kt', 'f8')
    v[...] = getattr(sphere, 'own_speed_kt', sphere.speed_kt)
    v.description = 'the speed its own sphere was predicted at (the condition may be the mean of several)'
    sub.phase_reference_time_s = np.float64(sphere.phase_reference_time_s)
