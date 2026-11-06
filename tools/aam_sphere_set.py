#!/usr/bin/env python3
"""Check a vehicle's AAM sphere set, and give the values its AAM deck needs.

AAM reads a vehicle's spheres (vvvvvnnn.nc: a five-character vehicle ID and a
three-digit index, as PSU-WOPWOP writes them) from one folder and interpolates
between those of one type.  It needs them consistent where it interpolates (AAM
Technical Reference p.134-138), takes a sphere's center as whole feet (Table
3-25, p.75) and at most ten of a type (the same table), and refuses two spheres
it would take as the same.  This reads the spheres of one or more results
folders and reports:

errors (exit status 1):
  grid        spheres of one AAM type that differ in RADIUS, PHI, THETA or FREQUENCY
              (band limits, narrowband frequencies, the PT tone count)
  center      an XYZ that is not whole feet
  centers     more than ten centers of one type
  duplicate   two spheres of one AAM type at one center, pylon angle, speed and flight
              path angle (0.5 deg), whatever their weights (a weight sweep needs AAM's
              weight lookup, not two spheres at one condition)
  narrowband  OB12 and NB spheres at one center: AAM reads both as NB
  phase       PT spheres of different phase conventions (phase_reference)
  audib       NB spheres of different bandwidths: a deck has one AUDIB bandwidth
  index       one file name in two folders (merge them with --merge, which renumbers)

warnings:
  spacing     neighboring speeds more than --speed-gap kt apart at one flight path
              angle, or flight path angles more than --angle-gap deg apart at one speed
              (the spacing AAM interpolates across, Table 2-7, p.33)
  hemisphere  a lower hemisphere off level flight, where part of the ground lies above it
  sound_speed spheres predicted at sound speeds more than 0.5 m/s apart

It prints the set as a table sorted by vehicle, type, speed and flight path
angle, the findings, and per vehicle the values its deck needs: each type's
centers in whole feet, the narrowband (AUDIB) bandwidth, and that the track is
flown at AOA 0 (WOPWOP's spheres are in the flight-path frame).  It writes no
deck itself.

Usage: aam_sphere_set.py FOLDER [FOLDER ...] [--merge OUT] [--speed-gap KT] [--angle-gap DEG]
"""
import argparse
import os
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from panam.sphere_io import read_sphere_type

#: The AAM sphere type each type is read as: OB12 and NB are both narrowband to AAM.
AAM_TYPE = {'BB': 'BB', 'OB12': 'NB', 'NB': 'NB', 'PT': 'PT'}
#: AAM's limit on the centers of one type (Table 3-25, p.75).
MAX_CENTERS = 10
#: Flight path angles within this of each other are one condition (degrees).
ANGLE_TOLERANCE_DEG = 0.5
#: Speeds within this of each other are one condition (knots).
SPEED_TOLERANCE_KT = 0.05
SOUND_SPEED_TOLERANCE = 0.5
SPHERE_NAME = re.compile(r'^(?P<vehicle>.{5})(?P<index>\d{3})\.nc$')


@dataclass
class Sphere:
    path: Path
    vehicle: str
    index: int
    kind: str
    speed: float
    angle: float
    pylon: float
    xyz: np.ndarray
    weight: float
    radius: float
    phi: np.ndarray
    theta: np.ndarray
    frequency: np.ndarray
    phase_reference: str = ''
    sound_speed: float = float('nan')
    nb_bandwidth: float = float('nan')

    @property
    def aam_type(self):
        return AAM_TYPE[self.kind]

    @property
    def full(self):
        return self.phi.min() <= -180.0 + 1e-6 and self.phi.max() >= 180.0 - 1e-6


@dataclass
class Finding:
    severity: str
    rule: str
    message: str
    paths: list = field(default_factory=list)


def _scalar(handle, name, default=float('nan')):
    if name not in handle.variables:
        return default
    return float(np.ravel(np.ma.filled(np.ma.asarray(handle[name][...], dtype=float), np.nan))[0])


def _array(handle, name):
    return np.asarray(np.ma.filled(np.ma.asarray(handle[name][...], dtype=float), np.nan)).ravel()


def read_sphere(path):
    """One sphere file as a :class:`Sphere`; None if its name is not vvvvvnnn.nc."""
    path = Path(path)
    match = SPHERE_NAME.match(path.name)
    if match is None:
        return None
    kind = read_sphere_type(str(path))
    with Dataset(str(path)) as handle:
        attributes = handle.ncattrs()
        nb = float(handle.getncattr('nb_bandwidth_Hz')) if 'nb_bandwidth_Hz' in attributes else float('nan')
        frequency = _array(handle, 'FREQUENCY')
        if kind == 'NB' and not np.isfinite(nb) and frequency.size > 1:
            nb = float(frequency[1] - frequency[0])
        return Sphere(
            path=path, vehicle=match['vehicle'], index=int(match['index']), kind=kind,
            speed=_scalar(handle, 'SPEED'), angle=_scalar(handle, 'FLIGHT_PATH_ANGLE'),
            pylon=_scalar(handle, 'PYLON_ANGLE', 90.0), xyz=_array(handle, 'XYZ'),
            weight=sum(_scalar(handle, n, 0.0) for n in ('EMPTY_WEIGHT', 'FUEL_WEIGHT', 'LOAD_WEIGHT')),
            radius=_scalar(handle, 'RADIUS'), phi=_array(handle, 'PHI'), theta=_array(handle, 'THETA'),
            frequency=frequency,
            phase_reference=str(handle.getncattr('phase_reference')) if 'phase_reference' in attributes else '',
            sound_speed=float(handle.getncattr('speed_of_sound_m_s')) if 'speed_of_sound_m_s' in attributes
            else float('nan'),
            nb_bandwidth=nb)


def read_set(folders):
    """Every vvvvvnnn.nc sphere of ``folders``, in folder and name order."""
    spheres = []
    for folder in folders:
        for path in sorted(Path(folder).glob('*.nc')):
            sphere = read_sphere(path)
            if sphere is not None:
                spheres.append(sphere)
    return spheres


def _center(sphere):
    return tuple(int(round(v)) for v in sphere.xyz)


def check_set(spheres, speed_gap_kt=20.0, angle_gap_deg=5.0):
    """The :class:`Finding` s of a set (see the module docstring)."""
    findings = []

    def add(severity, rule, message, *group):
        findings.append(Finding(severity, rule, message, [s.path for s in group]))

    names = defaultdict(list)
    for sphere in spheres:
        names[sphere.path.name].append(sphere)
    for name, group in names.items():
        if len(group) > 1:
            add('error', 'index', '{} is in {} folders: {}'.format(
                name, len(group), ', '.join(str(s.path.parent) for s in group)), *group)
    by_vehicle = defaultdict(list)
    for sphere in spheres:
        by_vehicle[sphere.vehicle].append(sphere)
    for vehicle, members in by_vehicle.items():
        for sphere in members:
            if not np.allclose(sphere.xyz, np.round(sphere.xyz), rtol=0.0, atol=1e-4):
                add('error', 'center', '{}: XYZ {} ft is not whole feet; AAM takes the center as whole '
                    'feet (give sphereXYZ)'.format(sphere.path.name, sphere.xyz.tolist()), sphere)
            if not sphere.full and abs(sphere.angle) > ANGLE_TOLERANCE_DEG:
                add('warning', 'hemisphere', '{}: a lower hemisphere at a flight path angle of {:g} deg; '
                    'part of the ground lies above it (hemisphere = \'full\')'.format(sphere.path.name, sphere.angle),
                    sphere)
        types = defaultdict(list)
        for sphere in members:
            types[sphere.aam_type].append(sphere)
        for aam_type, group in types.items():
            first = group[0]
            for sphere in group[1:]:
                for what in ('radius', 'phi', 'theta', 'frequency'):
                    a, b = np.atleast_1d(getattr(first, what)), np.atleast_1d(getattr(sphere, what))
                    if a.shape != b.shape or not np.allclose(a, b, rtol=1e-6, atol=1e-6):
                        add('error', 'grid', '{} and {}: {} {} spheres differ in {}'.format(
                            first.path.name, sphere.path.name, vehicle, aam_type, what.upper()), first, sphere)
                        break
            centers = sorted({_center(s) for s in group})
            if len(centers) > MAX_CENTERS:
                add('error', 'centers', '{} {}: {} centers, more than the {} AAM takes'.format(
                    vehicle, aam_type, len(centers), MAX_CENTERS), *group)
            seen = {}
            for sphere in group:
                key = (_center(sphere), round(sphere.pylon, 3))
                for other in seen.get(key, []):
                    if abs(other.speed - sphere.speed) <= SPEED_TOLERANCE_KT and \
                            abs(other.angle - sphere.angle) <= ANGLE_TOLERANCE_DEG:
                        add('error', 'duplicate', '{} and {}: two {} spheres at {:g} kt, {:g} deg and one center '
                            '(weights {:g} and {:g} lb)'.format(other.path.name, sphere.path.name, aam_type,
                                                                 sphere.speed, sphere.angle, other.weight,
                                                                 sphere.weight), other, sphere)
                seen.setdefault(key, []).append(sphere)
            if aam_type == 'NB':
                for center in centers:
                    kinds = {s.kind for s in group if _center(s) == center}
                    if kinds == {'OB12', 'NB'}:
                        add('error', 'narrowband', '{}: OB12 and NB spheres at center {}; AAM reads both as NB'.format(
                            vehicle, center), *[s for s in group if _center(s) == center])
                widths = sorted({round(s.nb_bandwidth, 6) for s in group
                                 if s.kind == 'NB' and np.isfinite(s.nb_bandwidth)})
                if len(widths) > 1:
                    add('error', 'audib', '{}: NB spheres of bandwidths {} Hz; a deck has one AUDIB bandwidth'.format(
                        vehicle, widths), *[s for s in group if s.kind == 'NB'])
            if aam_type == 'PT':
                conventions = sorted({s.phase_reference for s in group})
                if len(conventions) > 1:
                    add('error', 'phase', '{}: PT spheres of {} phase conventions: {}'.format(
                        vehicle, len(conventions), conventions), *group)
            for center in centers:
                at = [s for s in group if _center(s) == center]
                _spacing(at, vehicle, aam_type, speed_gap_kt, angle_gap_deg, add)
        speeds = [s.sound_speed for s in members if np.isfinite(s.sound_speed)]
        if speeds and max(speeds) - min(speeds) > SOUND_SPEED_TOLERANCE:
            add('warning', 'sound_speed', '{}: spheres predicted at sound speeds from {:g} to {:g} m/s'.format(
                vehicle, min(speeds), max(speeds)), *members)
    return findings


def _spacing(spheres, vehicle, aam_type, speed_gap, angle_gap, add):
    """Neighboring conditions further apart than AAM interpolates across."""
    def groups(key, tolerance):
        out = []
        for sphere in sorted(spheres, key=key):
            if out and abs(key(sphere) - key(out[-1][0])) <= tolerance:
                out[-1].append(sphere)
            else:
                out.append([sphere])
        return out

    for row in groups(lambda s: s.angle, ANGLE_TOLERANCE_DEG):
        values = sorted({round(s.speed, 3) for s in row})
        for low, high in zip(values, values[1:]):
            if high - low > speed_gap:
                add('warning', 'spacing', '{} {}: speeds {:g} and {:g} kt at {:g} deg are more than {:g} kt '
                    'apart'.format(vehicle, aam_type, low, high, row[0].angle, speed_gap), *row)
    for column in groups(lambda s: s.speed, SPEED_TOLERANCE_KT):
        values = sorted({round(s.angle, 3) for s in column})
        for low, high in zip(values, values[1:]):
            if high - low > angle_gap:
                add('warning', 'spacing', '{} {}: flight path angles {:g} and {:g} deg at {:g} kt are more than '
                    '{:g} deg apart'.format(vehicle, aam_type, low, high, column[0].speed, angle_gap), *column)


def sorted_table(spheres):
    """The set as text, one line a sphere, by vehicle, type, center, speed and angle."""
    lines = ['{:<14} {:<5} {:>9} {:>9} {:>7} {:>16} {:>9} {:>8} {:>6} {:<6}'.format(
        'file', 'type', 'speed_kt', 'fpa_deg', 'pylon', 'xyz_ft', 'weight_lb', 'radius', 'nfreq', 'sphere')]
    for s in sorted(spheres, key=lambda s: (s.vehicle, s.aam_type, _center(s), s.speed, s.angle, s.kind)):
        lines.append('{:<14} {:<5} {:9.2f} {:9.2f} {:7.1f} {:>16} {:9.0f} {:8.1f} {:6d} {:<6}'.format(
            s.path.name, s.kind, s.speed, s.angle, s.pylon, ' '.join(str(v) for v in _center(s)), s.weight,
            s.radius, s.frequency.size, 'full' if s.full else 'lower'))
    return '\n'.join(lines)


def deck_values(spheres):
    """Per vehicle: each AAM type's centers in whole feet, the NB (AUDIB) bandwidth, and the frame."""
    out = []
    for vehicle in sorted({s.vehicle for s in spheres}):
        members = [s for s in spheres if s.vehicle == vehicle]
        centers = {}
        for aam_type in ('BB', 'NB', 'PT'):
            found = sorted({_center(s) for s in members if s.aam_type == aam_type})
            if found:
                centers[aam_type] = found
        widths = sorted({s.nb_bandwidth for s in members if s.kind == 'NB' and np.isfinite(s.nb_bandwidth)})
        out.append(dict(vehicle=vehicle, centers_ft=centers,
                        audib_bandwidth_hz=widths[0] if len(widths) == 1 else None,
                        track='fly at AOA 0, roll 0 and yaw 0 (the spheres are in the flight-path frame)'))
    return out


def merge(spheres, out):
    """Copy ``spheres`` into the new folder ``out``, giving a file whose name is taken the
    vehicle's next free index.  Returns [(source, destination)]."""
    out = Path(out)
    if out.exists():
        raise ValueError('{} exists'.format(out))
    out.mkdir(parents=True)
    used = defaultdict(set)
    for sphere in spheres:
        used[sphere.vehicle].add(sphere.index)
    taken, copies = set(), []
    for sphere in spheres:
        name = sphere.path.name
        if name in taken:
            index = max(used[sphere.vehicle]) + 1
            if index > 999:
                raise ValueError('{}: no free index left'.format(sphere.vehicle))
            used[sphere.vehicle].add(index)
            name = '{}{:03d}.nc'.format(sphere.vehicle, index)
        taken.add(name)
        shutil.copy2(sphere.path, out / name)
        copies.append((sphere.path, out / name))
    return copies


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('folders', nargs='+', help='results folders holding vvvvvnnn.nc spheres')
    parser.add_argument('--merge', default=None, metavar='OUT',
                        help='copy the spheres into the new folder OUT, renumbering any whose name is taken')
    parser.add_argument('--speed-gap', type=float, default=20.0, metavar='KT',
                        help='warn where neighboring speeds are further apart (default %(default)g)')
    parser.add_argument('--angle-gap', type=float, default=5.0, metavar='DEG',
                        help='warn where neighboring flight path angles are further apart (default %(default)g)')
    args = parser.parse_args(argv)
    spheres = read_set(args.folders)
    if not spheres:
        print('no vvvvvnnn.nc spheres in ' + ', '.join(args.folders))
        return 1
    if args.merge:
        for source, destination in merge(spheres, args.merge):
            if source.name != destination.name:
                print('{} -> {}'.format(source, destination.name))
        spheres = read_set([args.merge])
    print(sorted_table(spheres))
    findings = check_set(spheres, args.speed_gap, args.angle_gap)
    for finding in findings:
        print('{}: [{}] {}'.format(finding.severity.upper(), finding.rule, finding.message))
    for vehicle in deck_values(spheres):
        print('{}: centers (ft) {}; AUDIB bandwidth {} Hz; {}'.format(
            vehicle['vehicle'], vehicle['centers_ft'],
            'none' if vehicle['audib_bandwidth_hz'] is None else '{:g}'.format(vehicle['audib_bandwidth_hz']),
            vehicle['track']))
    errors = sum(f.severity == 'error' for f in findings)
    print('{} sphere(s), {} error(s), {} warning(s)'.format(len(spheres), errors, len(findings) - errors))
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
