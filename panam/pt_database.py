"""Format 2 NICE-OPS databases of pure-tone (PT) spheres: one source per rotor, with its phase.

PSU-WOPWOP writes one PT noise sphere per rotor (``sourceTag``): each shaft order's level and cosine phase over the
sphere, about the rotor's own hub, in the flight-path frame, with the rotor's shaft rate and its azimuth at the
phase reference time.  :func:`build_pt_database` turns a set of them into a database NICE-OPS evaluates tone by
tone (``docs/file_formats.md``, "Format 2: pure-tone sources"):

- every condition group (one per flight condition: SPEED, FLIGHT_PATH_ANGLE) keeps the format 1 fields NICE-OPS
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
from collections import OrderedDict

import numpy as np
from netCDF4 import Dataset

from panam import flight_acoustics as fa
from panam import provenance as _provenance

#: Root ``database_version`` of a database with pure-tone sources.
PT_DATABASE_FORMAT_VERSION = 2

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
            raise ValueError(f'{path}: the tones must be in the emission frame (DOPPLER_SHIFT_REMOVED 1): NICE-OPS '
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
            speed_reference=str(attributes.get('speed_reference', 'air')),
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
    """The sphere's shaft orders that carry energy above its floor: (harmonics, frequencies Hz, amplitude dB with
    floor cells at -inf, phase deg locked to the rotor azimuth, the share of the energy dropped).  Every listed
    frequency must be a shaft order, or (a spectrum over N revolutions) a whole multiple of 1/N of one: then only
    the whole orders are tones, and the bins between them (leakage, a loading's slow drift) are dropped."""
    shaft_hz = abs(sphere.shaft_rate) / (2.0 * math.pi)
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
    # the floor in some cells, the floor stays: a predicted cell is quiet, not missing, and NICE-OPS drops a -inf
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


def _condition_key(sphere):
    return round(sphere.speed_kt, 6), round(sphere.flight_path_angle_deg, 6)


def build_pt_database(sphere_files, database_filename, *, weight_newtons=None, sync_groups='shared_rate',
                      main_rotor=None, speed_reference=None, condition_axes='helicopter', rotor_count=None,
                      hover_thrust_coefficient=None, sound_speed_mps=None):
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
            scale); default the largest radius (the first tag on a tie).
        speed_reference: 'air' or 'ground'; default what the spheres record (PSU-WOPWOP: air).
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

    The database holds every flight condition once, at load factor 1 (``fixed_load_factor``): NICE-OPS scales the
    levels to other load factors.  Raises ValueError, before writing anything, for spheres that are not PT, not in
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
    main = main_rotor or max(tags, key=lambda t: (conditions[next(iter(conditions))][t].rotor_radius_m,
                                                  -tags.index(t)))
    rotor_radius = conditions[next(iter(conditions))][main].rotor_radius_m
    tip_speed = conditions[next(iter(conditions))][main].tip_speed_mps
    for by_tag in conditions.values():
        s = by_tag[main]
        if not (np.isclose(s.rotor_radius_m, rotor_radius, rtol=1e-6) and np.isclose(s.tip_speed_mps, tip_speed,
                                                                                    rtol=1e-6)):
            raise ValueError(f'{s.name}: the main rotor ({main}) has radius {s.rotor_radius_m} m and tip speed '
                             f'{s.tip_speed_mps} m/s, other conditions {rotor_radius} m and {tip_speed} m/s')
    references = {s.speed_reference for s in spheres}
    if speed_reference is None:
        if len(references) != 1:
            raise ValueError(f'the spheres record speed_reference {sorted(references)}; give one')
        speed_reference = references.pop()
    azimuth_references = {s.azimuth_reference for s in spheres}
    if len(azimuth_references) != 1:
        raise ValueError(f'the spheres record azimuth_reference {sorted(azimuth_references)}; they must agree')

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
        'hover_thrust_coefficient': hover_thrust_coefficient, 'sound_speed_mps': sound_speed_mps}
    inputs = {'sphere_files': sorted(os.path.basename(str(p)) for p in sphere_files),
              'database_filename': _provenance.path_name(database_filename)}
    with fa._atomic_output(database_filename) as temporary, Dataset(temporary, 'w') as database:
        fa._write_database_root(
            # Multirotor: not fixed_load_factor (NICEOPS's load-factor law is in C_T, which multirotor axes do not
            # carry; it refuses the flag there).
            database, fixed_load_factor=multirotor is None, speed_reference=speed_reference,
            atmosphere=fa.Atmosphere(temperature=293.15, pressure=101.325, relative_humidity=20.0),
            main_rotor_radius=rotor_radius, main_rotor_tip_speed=tip_speed, vehicle_weight_newtons=weight_newtons,
            azimuth_reference=azimuth_references.pop(),
            provenance=_provenance.build_provenance('pt_database.build_pt_database', settings, inputs=inputs))
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
        database.sources = ','.join(tags)
        database.main_rotor_source = main
        pending, key_of = [], {}
        for i, (key, by_tag) in enumerate(conditions.items()):
            key_of['sphere' + str(i)] = key
            s = by_tag[main]
            coverage = np.zeros(empty_bands.shape, dtype=bool)
            pending.append(fa._SphereCondition(
                'sphere' + str(i), phi_grid, theta_grid, s.radius_m, np.full(phi_grid.shape, -np.inf),
                np.zeros(phi_grid.shape), s.speed_kt, s.flight_path_angle_deg, 1.0, BAND_CENTERS_HZ, empty_bands,
                coverage, None))
        shared = fa._write_shared_grid_and_frequency(database, pending)
        for condition, by_tag in zip(pending, conditions.values()):
            s = by_tag[main]
            # C_T at the condition's own density (the prediction's), the database's weight: thrust_coefficient_density
            # run.  add_sphere_group writes load_factor x weight_coefficient.
            weight_coefficient = weight_newtons / (s.air_density * math.pi * rotor_radius ** 2 * tip_speed ** 2)
            axes = multirotor['conditions'][key_of[condition.name]] if multirotor is not None else None
            fa.add_sphere_group(database, condition.name, condition.phi, condition.theta, condition.radius,
                                condition.SPLA, condition.EAA, condition.speed, condition.flight_path_angle, 1.0,
                                rotor_radius, tip_speed, weight_coefficient, condition.frequency,
                                condition.amplitude, write_grid_and_frequency=not shared, doppler_shift_removed=0,
                                coverage=condition.coverage,
                                condition_variables=axes['variables'] if axes is not None else None)
            # DOPPLER_SHIFT_REMOVED applies to the band spectrum, which is empty here (received frame by
            # convention); the tones are emission-frame shaft orders by definition, and NICE-OPS shifts them.
            group = database.groups[condition.name]
            # No condition_origin: a predicted condition is a stored condition of its own, as a measured one is
            # (NICE-OPS reads a group without one so); the root's panam_tool names the prediction build.
            group.air_density = np.float64(s.air_density)
            group.air_density_units = 'kg m-3'
            for tag in tags:
                _write_source(group, by_tag[tag], group_of.get(tag, ''),
                              speed_ratio=axes['speed_ratio'][tag] if axes is not None else None)


def _sync_groups(conditions, tags, sync_groups):
    """{tag: group name} for the coherent groups (absent or '' = free-running)."""
    if sync_groups is None:
        return {}
    if isinstance(sync_groups, dict):
        unknown = set(sync_groups) - set(tags)
        if unknown:
            raise ValueError(f'sync_groups names {sorted(unknown)}, which are not sources ({tags})')
        return {tag: str(name) for tag, name in sync_groups.items() if name}
    if sync_groups != 'shared_rate':
        raise ValueError(f"sync_groups must be 'shared_rate', None or a {{tag: group}} dict, not {sync_groups!r}")
    # Rotors of one prediction at one |shaft rate| and one phase reference time, at every condition.
    def signature(s):
        return round(abs(s.shaft_rate), 9), round(s.phase_reference_time_s, 9)
    together = [t for t in tags
                if all(signature(by_tag[t]) == signature(by_tag[tags[0]]) for by_tag in conditions.values())]
    if len(together) < 2:
        return {}
    return {tag: 'shaft' for tag in together}


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
        v_h = math.sqrt(weight / (2.0 * rho * rotor_count * math.pi * radius ** 2))
        omega_h = math.sqrt(weight / (rotor_count * rho * math.pi * radius ** 2 * hover_thrust_coefficient)) / radius
        u = s.speed_kt * fa.KNOT_MPS
        gamma = math.radians(s.flight_path_angle_deg)
        inplane, climb = u * math.cos(gamma), u * math.sin(gamma)
        out['conditions'][key] = {
            'variables': OrderedDict((
                ('inplane_speed_ratio', (inplane / v_h, '1')),
                ('axial_speed_ratio', (climb / v_h, '1')),
                ('hover_tip_mach', (omega_h * radius / c, '1')),
                ('air_density', (rho, 'kg m-3')),
                ('weight_n', (weight, 'N')),
                ('airspeed', (u, 'm s-1')),
                ('climb_rate', (climb, 'm s-1')))),
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
    sub.phase_reference = 'rotor_azimuth'
    if sync_group:
        sub.sync_group = sync_group
    sub.source_sphere = sphere.name
    sub.phase_reference_time_s = np.float64(sphere.phase_reference_time_s)
