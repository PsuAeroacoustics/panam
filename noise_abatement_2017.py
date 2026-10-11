"""Regenerate AAM source spheres from the 2017 FAA/NASA Noise Abatement flight test.

The spheres shipped in ``Sphere_Database`` were depropagated without any
ambient reference.  Depropagation multiplies the measured spectrum by
``(r/r_ref)^2`` and, when absorption is undone, by ``10^(alpha(f)(r-r_ref)/10)``.
A band that is ambient-limited at the microphone carries no source information,
so both factors amplify pure noise floor -- and because ``alpha`` reaches tens
of dB/km in the top third-octave bands, the result is physically impossible
source data.  ``Be407220.nc`` reaches 216 dB at 10 kHz on the aft pole this way,
against ~90 dB at theta=60 (air goes nonlinear around 194 dB).

This module rebuilds the spheres with :func:`flight_acoustics.depropagate_hemisphere`
driven by a *measured* ambient reference: the test flew dedicated ambient (AMB)
runs on each array layout and test day, so every flyover can be gated per
microphone, per frequency bin against real ambient from the same channel.

Layout of the dataset this module expects (``root``)::

    <aircraft>/<aircraft>FullRefList.csv       run index: condition, layout, times
    <aircraft>/<aircraft>_AC_Data/<run>AC.csv  tracking, local array frame
    <aircraft>/<aircraft>_Acoustic_Data/...    <run>_<mic>_pascal.nc, per channel
    <aircraft>/*MicFullList.csv, *StaticList.csv   instrument type per channel

Conventions established by checking regenerated labels against the legacy
spheres (see :func:`steady_window`):

* ``z`` is positive *up*, and the tracking file's ``vz`` is positive *down* in
  every file except EC130B4 day 298's, where it is positive up; :func:`vz_sign`
  takes the sign from dz/dt, and the flight path angle is
  ``atan2(vz_up, hypot(vx, vy))``;
* the sphere's ``SPEED`` is mean ground speed in knots (``VGk``);
* both labels are means over the steady segment, and any steady sub-window
  reproduces them, so the segment need not match the legacy one exactly;
* the track frame's +x runs along the reference list's ``true_heading``, which
  differs between layouts (270 and 279 deg at Amedee, 140 and 92.3 at Eglin), and
  +y 90 deg to its left; every compass direction goes through
  :func:`frame_bearing_deg`.

Before depropagating, :func:`build_sphere` refuses a run whose track is impossible
over its window (:func:`check_track`: below or within 10 ft of the ground boards, or
a z that does not follow its altitude) and flags one worth a second look; it records
the run's gross weight, air density and, when the caller gives it, the wind at the
aircraft; and it can file azimuth by the INS heading instead of the ground track
(``azimuth_reference``).  ``docs/database_build.md`` describes all three and how to
run a build; ``docs/notes/2017_build_findings.md`` records what the track check found
across the six aircraft; ``docs/file_formats.md`` gives the sphere and database
variables.
"""

import contextlib
import functools
import os
import csv
import glob
import inspect
import logging
import multiprocessing
import pickle
import re
import warnings
from collections import defaultdict

import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from netCDF4 import Dataset

import flight_acoustics as fa
import local_paths
import provenance
import refracted_rays
import sphere_statistics

#: Name under which :mod:`local_paths` finds the dataset root.
DATA_ROOT_NAME = 'noise_abatement_2017'

#: Dataset directory name -> sphere-file prefix used by the legacy database.
#: These are not the dataset directory names: the shipped spheres are
#: ``Be407220.nc``, ``RO-66539.nc``, ``AS350368.nc`` and so on.
AIRCRAFT_SPHERE_PREFIX = {
    'AS350B3': 'AS350',
    'B206L3': 'Be206',
    'B407': 'Be407',
    'EC130B4': 'EC130',
    'R44': 'RO-44',
    'R66': 'RO-66',
}

#: Total width of the legacy spheres' ``title`` attribute, blank padded.
AAM_TITLE_WIDTH = 77

#: Year the test was flown; run ids carry the day of year, not the date.
TEST_YEAR = 2017

#: Instrument types that sit on (or flush with) the ground and therefore see
#: pressure doubling.  Elevated microphones are excluded instead of corrected:
#: they are co-located with ground boards on this array, so they add no
#: directions, and their ground-reflection comb filtering would corrupt the
#: high-frequency bands this work is trying to clean up.
GROUND_BOARD_INSTRUMENTS = ('gdbdfl', 'invgb7')

#: The test site's ground (Amedee Army Airfield), fitted to the co-located
#: pole and ground-plate microphone pairs: variable porosity, effective flow
#: resistivity 200 kPa s/m^2 (docs/notes/ground_plane_corrections.md, path 4 and
#: the run-holdout profile).
SITE_GROUND = dict(model='variable_porosity', sigma_e=200.0, alpha_e=0.0)

#: Microphone height above its plate's top, feet, per ground-board instrument
#: type.  Every ground microphone in the NASA test was a flush GRAS 67AX in a
#: 400 mm GR1425 plate, at 3/4 of the radius, outboard of the track -- the
#: dataset's ``invgb7`` label ("inverted over a ground board with a 7 mm gap")
#: included, per the test team.  A true inverted layout (``axisymmetric_bem``'s
#: ``mic_height``, plus the microphone body over the gap, which is not yet
#: modeled) is for other tests.
BOARD_MIC_HEIGHT_FT = {'gdbdfl': 0.0, 'invgb7': 0.0}

#: Plate tables are computed at the run's sound speed rounded to this
#: relative step (they scale with frequency / sound speed).
PLATE_TABLE_SOUND_SPEED_STEP = 0.005

#: Legacy sphere reference radius, feet.
DEFAULT_R_REF_FT = 100.0

#: build_sphere's rim elevation, deg below the horizon, when max_rim_range_ft
#: is given without rim_elevation_deg.
DEFAULT_RIM_ELEVATION_DEG = 14.0

#: Threads used to warm the cloud-storage cache ahead of each run.
PREFETCH_WORKERS = 24

#: The atmosphere a run without usable ground weather is depropagated in:
#: 20 C, 101.325 kPa, 20 % relative humidity (fa.Atmosphere arguments).
STANDARD_DAY = dict(temperature=293.15, pressure=101.325, relative_humidity=20.0)

#: The command line's log format.
LOG_FORMAT = '%(asctime)s %(levelname)s %(message)s'

#: Third-octave band centers carried by the legacy spheres.
LEGACY_BAND_CENTERS_HZ = np.array([
    10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0,
    100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0,
    1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0,
    8000.0, 10000.0,
])


# --------------------------------------------------------------------------
# Dataset indexing
# --------------------------------------------------------------------------

def is_steady_flight_card(row):
    """Whether a reference-list row describes a non-maneuvering pass.

    A run is treated as steady when the flight card's own ``bank_ang`` and
    ``accel_rate`` fields both read as an explicit zero.  An *unspecified*
    field (blank) is not the same as zero and is treated as maneuvering --
    checked empirically for the B407 "A" (approach) family, whose accel_rate
    is blank throughout: only 4 of its 25 runs even contain an 8 s window
    that :func:`steady_window` accepts, and those four still drift a median
    5.8 kt across it, against 2.9 kt for the "D" (descent) family and 1.3 kt
    for "L" (level).  They are decelerating approaches, not steady runs that
    the flight card simply forgot to annotate.

    Verified against the Be407 sphere set actually shipped in
    ``Sphere_Database``: this predicate reproduces all 113 of its source runs
    exactly (no omissions), and additionally admits 6 runs at repeated
    conditions (extra L4/L9/D4 passes, one flagged as an aborted approach in
    its comments) that the legacy build happened to skip.

    The approach family is refused by name as well, whatever its fields say.
    Not every aircraft's cards leave them blank: all 23 of the B206L3's and 2
    of the R66's carry an explicit zero bank and acceleration, which let
    them through as steady and built validation runs into those spheres.
    """
    condition = (row.get('test_cond') or '').strip()
    if condition == 'AMB' or re.match(r'^[AH]\d', condition):
        return False

    def explicit_zero(value):
        text = (value or '').strip()
        if text == '':
            return False
        try:
            return float(text) == 0.0
        except ValueError:
            return False

    return explicit_zero(row.get('bank_ang')) and explicit_zero(row.get('accel_rate'))


class NoiseAbatementTest:
    """Index one aircraft's directory of the 2017 Noise Abatement test."""

    def __init__(self, aircraft, root=None):
        """``root`` defaults to the ``noise_abatement_2017`` entry of :mod:`local_paths`."""
        self.aircraft = aircraft
        self.root = local_paths.data_path(DATA_ROOT_NAME, root)
        self.base = os.path.join(self.root, aircraft)
        if not os.path.isdir(self.base):
            raise FileNotFoundError('No such aircraft directory: ' + self.base)

        self.reference = self._load_reference_list()
        self.by_run = {row['combined']: row for row in self.reference}
        self.acoustic_files = self._index_acoustic_files()
        self.instrument_types = self._load_instrument_types()

    # -- loading -----------------------------------------------------------

    def _load_reference_list(self):
        pattern = os.path.join(self.base, '*FullRefList.csv')
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError('No FullRefList.csv under ' + self.base)
        rows = []
        with _open_csv(matches[0]) as handle:
            for row in csv.DictReader(handle):
                clean = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
                if clean.get('combined'):
                    rows.append(clean)
        return rows

    def _index_acoustic_files(self):
        """Map run id -> {mic number: path}, searching the day subdirectories."""
        index = defaultdict(dict)
        acoustic_root = os.path.join(self.base, self.aircraft + '_Acoustic_Data')
        for path in glob.glob(os.path.join(acoustic_root, '**', '*_pascal.nc'), recursive=True):
            name = os.path.basename(path)
            parts = name.split('_')
            if len(parts) < 3:
                continue
            run, mic = parts[0], parts[1]
            try:
                mic_number = int(mic)
            except ValueError:
                continue
            # Some runs appear both loose in the top level and inside a day
            # directory; either copy is fine, so first one wins.
            index[run].setdefault(mic_number, path)
        return dict(index)

    def _load_instrument_types(self):
        """Map microphone number -> instrument type, merged over the layout lists."""
        types = {}
        for pattern in ('*MicFullList.csv', '*StaticList.csv'):
            for path in glob.glob(os.path.join(self.base, pattern)):
                with _open_csv(path) as handle:
                    for row in csv.DictReader(handle):
                        try:
                            mic_number = int(str(row.get('M', '')).strip())
                        except (TypeError, ValueError):
                            continue
                        instrument = str(row.get('insttype', '')).strip()
                        # A microphone number can appear on more than one
                        # layout; keep the ground-board reading if any layout
                        # has one, since that is the channel we process.
                        if mic_number not in types or instrument in GROUND_BOARD_INSTRUMENTS:
                            types[mic_number] = instrument
        return types

    # -- queries -----------------------------------------------------------

    def runs(self, conditions=None, exclude_conditions=('AMB',), steady_only=False):
        """Run ids that have both acoustic data and a tracking file.

        ``steady_only=True`` additionally requires the flight card's own bank
        angle and acceleration-rate fields to read as zero -- see
        :func:`is_steady_flight_card`.  This is what determines which runs get
        a source sphere at all: turns and accelerating/decelerating passes
        smear directivity across azimuth in a way depropagation does not
        correct for.  Against the legacy Be407 sphere set, see
        :func:`is_steady_flight_card`.
        """
        out = []
        for row in self.reference:
            run = row['combined']
            if conditions is not None and row.get('test_cond') not in conditions:
                continue
            if exclude_conditions and row.get('test_cond') in exclude_conditions:
                continue
            if steady_only and not is_steady_flight_card(row):
                continue
            if run not in self.acoustic_files:
                continue
            if not os.path.exists(self.track_path(run)):
                continue
            out.append(run)
        return sorted(set(out))

    def track_path(self, run):
        return os.path.join(self.base, self.aircraft + '_AC_Data', run + 'AC.csv')

    def ambient_run(self, run):
        """The ambient run to gate ``run`` against.

        Prefers the same array layout and the same test day, then falls back to
        the same layout on any day, choosing whichever ambient run is closest in
        time.  Gating against a different layout would compare a channel with a
        microphone that was somewhere else, so that is refused.
        """
        row = self.by_run.get(run)
        if row is None:
            raise KeyError('Unknown run ' + str(run))
        layout = row.get('layout')
        candidates = [r for r in self.reference
                      if r.get('test_cond') == 'AMB'
                      and r.get('layout') == layout
                      and r['combined'] in self.acoustic_files]
        if not candidates:
            return None

        def day(candidate):
            return candidate['combined'][:3]

        def seconds(candidate):
            try:
                return float(candidate.get('utc_secs_from_mid_start') or 'nan')
            except ValueError:
                return float('nan')

        same_day = [c for c in candidates if day(c) == day(row)]
        pool = same_day or candidates
        run_seconds = seconds(row)
        if np.isfinite(run_seconds):
            finite = [c for c in pool if np.isfinite(seconds(c))]
            if finite:
                return min(finite, key=lambda c: abs(seconds(c) - run_seconds))['combined']
        return pool[0]['combined']

    def ground_board_mics(self, run):
        """Microphone numbers for ``run`` that sit on a ground board."""
        available = self.acoustic_files.get(run, {})
        mics = []
        for mic_number in sorted(available):
            instrument = self.instrument_types.get(mic_number)
            if instrument is None:
                warnings.warn('Run {}: no instrument type for mic {}; skipping it'
                              .format(run, mic_number))
                continue
            if instrument in GROUND_BOARD_INSTRUMENTS:
                mics.append(mic_number)
        return mics


# --------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------

def _open_csv(path):
    """Open a dataset CSV tolerantly.

    ``R66FullRefList.csv`` carries a stray non-UTF-8 byte (0x89 at offset
    36283) in a comment field.  It decodes as U+FFFD (``errors='replace'``),
    so the run index still reads rather than the whole aircraft failing to load.
    """
    return open(path, encoding='utf-8-sig', errors='replace', newline='')


def vz_sign(track):
    """+1 if the track's ``vz`` is positive up, -1 if positive down, from ``z`` itself.

    The files do not agree: ``z`` is positive up throughout, and ``vz`` is
    positive DOWN in every file (agreement with ``d/dt`` of ``z`` around -0.99)
    except EC130B4 test day 298, whose 49 files have it positive UP (+1.000).
    Assuming down everywhere labeled those days' descents climbs.

    A track with too little vertical motion to judge -- level passes and hovers,
    about 40 files across the dataset, all leaning negative -- falls back to
    down, the form every day but one uses; there the sign hardly matters.
    """
    dz = np.gradient(track['z'], track['time'])
    vz = track['vz']
    # Uncentered, so a steady descent -- constant dz/dt, no variance to
    # correlate -- still decides: vz . dz/dt is +|dz|^2 when they agree.
    rms = lambda x: np.sqrt(np.mean(np.square(x)))
    if rms(vz) < 0.5 or rms(dz) < 0.5:                 # ft/s
        return -1.0
    agreement = np.mean(vz * dz) / (rms(vz) * rms(dz))
    if agreement > 0.5:
        return 1.0
    if agreement < -0.5:
        return -1.0
    warnings.warn('vz and dz/dt barely agree (cosine {:.2f}); taking vz as positive down, '
                  'as every day but EC130B4 day 298 stores it'.format(agreement))
    return -1.0


def load_track(path):
    """Load a tracking CSV, making the vertical velocity positive up.

    ``z`` is positive up, but whether ``vz`` is depends on the file (test day); see
    :func:`vz_sign`.  :func:`flight_acoustics.hemigen` only takes heading from
    the horizontal components, so the sign does not corrupt the hemisphere
    geometry, but it does set the flight path angle -- which is how a descent
    gets labeled as a climb.
    """
    data = np.genfromtxt(path, delimiter=',', names=True)
    if data.size < 2:
        raise ValueError('Tracking file has too few samples: ' + path)
    track = {name: np.asarray(data[name], dtype=float) for name in data.dtype.names}
    track['time'] = track['utcsec']
    track['vz_up'] = vz_sign(track) * track['vz']
    track['fpa_deg'] = np.degrees(np.arctan2(track['vz_up'],
                                             np.hypot(track['vx'], track['vy'])))
    track['ground_speed_knots'] = track['VGk']
    return track


def _true_runs(mask):
    """Half-open ``(starts, stops)`` index arrays of the runs of True in ``mask``."""
    edges = np.flatnonzero(np.diff(np.concatenate(([False], mask, [False]))))
    return edges[::2], edges[1::2]


def _close_short_gaps(ok, time, max_gap_s):
    """Fill interior runs of False shorter than ``max_gap_s``.

    Steadiness is judged sample by sample at 50 Hz, so one gust-induced roll
    spike marks a single sample unsteady and splits an otherwise good 40 s
    window into two 20 s halves.  Closing brief gaps keeps the window whole
    while still rejecting a real turn, which lasts seconds rather than
    hundredths.

    2.0 s (not the 0.5 s this was first tuned to) is what generalizes: tuned
    against Be407 alone, 0.5 s recovered every Be407 run but left every other
    aircraft with a 17-50% "steady segment too short" failure rate, because a
    lighter, twitchier airframe's gust response is a wider, longer-lived
    excursion, not sensor noise -- R44's median roll and turn rate sit
    comfortably inside the default tolerances (1.3 deg, 1.17 deg/s on one
    representative run) while a single gust spikes to 16 deg / 8 deg/s for a
    couple of seconds. Surveyed across every run that failed on duration in
    the six-aircraft rebuild (100 runs): 0% recovered at 0.5 s (all had
    already failed there), 48% at 1.0 s, 57% at 1.5 s, 62% at 2.0 s, 65% at
    3.0 s -- past 2 s the extra recovery is small, and a longer bridge starts
    rejoining what could be a genuine multi-second deviation rather than a
    gust. The steadiness bounds themselves (roll, turn rate, speed, FPA) are
    untouched by this -- only how long a brief excursion can be before it
    counts as a real break.

    Only a gap with steady flight on both sides is a gap: an unsteady stretch
    at the start or end of the record bridges nothing, so it is left alone,
    however short.
    """
    ok = np.asarray(ok, dtype=bool).copy()
    if max_gap_s <= 0.0:
        return ok
    for start, stop in zip(*_true_runs(~ok)):
        if start > 0 and stop < ok.size and time[stop] - time[start] <= max_gap_s:
            ok[start:stop] = True
    return ok


def steady_window(track, speed_tolerance_knots=4.0, fpa_tolerance_deg=2.0,
                  roll_tolerance_deg=5.0, turn_rate_tolerance_deg_s=1.5,
                  max_array_range=None, min_duration_s=8.0, max_gap_s=2.0):
    """Longest contiguous stretch of steady flight in ``track``.

    Returns ``(index_start, index_stop)`` as a half-open slice.  Steadiness is
    judged against the median condition over the half of the record closest to
    the array, which is the part the microphones actually hear; taking the
    median over the whole record would be pulled by the turn-in and the
    climb-away at the ends.

    Speed and flight path angle alone do not exclude the turn onto the run:
    a level turn holds both.  Bank angle and turn rate do, so both are
    required as well -- a sphere built through a turn smears directivity
    across azimuth, since the airframe is rotating relative to the array.

    The speed and flight path angle tests are a band around the run's own
    median, so they bound scatter directly and a monotonic drift only through
    the band width.  Measured over the 113 Be407 conditions, the defaults
    leave a median end-to-end drift of 2.3 kt and 0.8 deg, with a worst case
    of 7.7 kt and 3.4 deg.  Tightening to +-3 kt / +-1.5 deg takes the worst
    case to 6.2 kt / 3.1 deg but costs 9 of the 113 runs, and +-2 / +-1 costs
    29 -- the tolerances are the knob for that trade.

    ``max_array_range``, when given, additionally restricts the window to
    where the vehicle is within that distance of the run's reference point,
    the track frame's origin (same units as the track).  That bounds the
    vehicle, not the propagation path -- this array spans over 6000 ft, so a
    microphone at the far end is still 5000+ ft away from a vehicle sitting
    over the reference point.  Bounding the path is what straight-ray validity actually asks for, and
    ``depropagate_hemisphere(max_range=)`` does it per emission point and
    microphone; this is the blunter instrument, off by default because it
    also throws away whole runs and, on a steep descent, selects the flare.
    """
    speed = track['ground_speed_knots']
    fpa = track['fpa_deg']
    slant = np.sqrt(track['x'] ** 2 + track['y'] ** 2 + track['z'] ** 2)
    near = slant <= np.median(slant)
    if not np.any(near):
        near = np.ones_like(slant, dtype=bool)
    speed_reference = float(np.median(speed[near]))
    fpa_reference = float(np.median(fpa[near]))

    ok = np.logical_and(np.abs(speed - speed_reference) <= speed_tolerance_knots,
                        np.abs(fpa - fpa_reference) <= fpa_tolerance_deg)
    if 'roll' in track:
        ok = np.logical_and(ok, np.abs(track['roll']) <= float(roll_tolerance_deg))
    if 'heading' in track and track['heading'].size > 1:
        # Unwrap before differentiating, or every pass through +/-180 reads as
        # a turn of several hundred degrees per second.
        heading = np.degrees(np.unwrap(np.radians(track['heading'])))
        turn_rate = np.gradient(heading, track['time'])
        ok = np.logical_and(ok, np.abs(turn_rate) <= float(turn_rate_tolerance_deg_s))
    ok = _close_short_gaps(ok, track['time'], float(max_gap_s))
    if max_array_range is not None:
        # Applied after closing: this one is a geometric window, not a
        # transient, so a gap in it is real.
        ok = np.logical_and(ok, slant <= float(max_array_range))
    if not np.any(ok):
        raise ValueError('No steady flight segment found')

    # Longest run of True; argmax takes the first of equally long ones.
    starts, stops = _true_runs(ok)
    longest = int(np.argmax(stops - starts))
    best_start, best_stop = int(starts[longest]), int(stops[longest])

    duration = track['time'][best_stop - 1] - track['time'][best_start]
    if duration < min_duration_s:
        raise ValueError('Steady segment is only {:.1f} s long (need {:.1f} s)'
                         .format(duration, min_duration_s))
    return best_start, best_stop


def flight_condition(track, index_start, index_stop):
    """Mean ground speed (knots) and flight path angle (deg) over a window."""
    speed = float(np.mean(track['ground_speed_knots'][index_start:index_stop]))
    fpa = float(np.mean(track['fpa_deg'][index_start:index_stop]))
    return speed, fpa


# --------------------------------------------------------------------------
# The track frame
# --------------------------------------------------------------------------

#: WGS84 semi-major axis (ft) and first eccentricity squared, for the local radii of
#: curvature that turn latitude and longitude into feet about a run's reference point.
WGS84_A_FT = 6378137.0 / 0.3048
WGS84_E2 = 6.69437999014e-3

#: A track frame's +x bearing fitted from the track's own latitude and longitude must agree
#: with the reference list's ``true_heading`` to this many degrees.  Over every track of the
#: six aircraft they agree to 0.4 deg (Amedee's flight layout 270, its hover layout 279;
#: Eglin's 140 and 92.3).
FRAME_BEARING_TOLERANCE_DEG = 1.0

#: The fit is attempted only over a track spanning at least this much ground (ft); a hover's
#: few feet of drift do not fix a direction.
FRAME_FIT_MIN_SPAN_FT = 500.0


def _float_or_nan(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return float('nan')


def local_east_north_ft(lat_deg, lon_deg, lat0_deg, lon0_deg):
    """East and north (ft) of points from a reference point, on the WGS84 ellipsoid's
    local radii of curvature at the reference: good to a fraction of a foot over the few
    kilometers of an array."""
    lat0 = np.radians(float(lat0_deg))
    w = 1.0 - WGS84_E2 * np.sin(lat0) ** 2
    meridional = WGS84_A_FT * (1.0 - WGS84_E2) / w ** 1.5
    normal = WGS84_A_FT / np.sqrt(w)
    east = np.radians(np.asarray(lon_deg, dtype=float) - float(lon0_deg)) * normal * np.cos(lat0)
    north = np.radians(np.asarray(lat_deg, dtype=float) - float(lat0_deg)) * meridional
    return east, north


def east_north_to_frame(bearing_deg, east, north):
    """East/north components to the track frame, whose +x lies along compass ``bearing_deg``
    and +y along ``bearing_deg - 90`` (to the left of +x), z up: the frame of every 2017 AC
    track, fitted from their latitude and longitude."""
    b = np.radians(float(bearing_deg))
    east, north = np.asarray(east, dtype=float), np.asarray(north, dtype=float)
    return east * np.sin(b) + north * np.cos(b), -east * np.cos(b) + north * np.sin(b)


def frame_to_east_north(bearing_deg, x, y):
    """The inverse of :func:`east_north_to_frame`."""
    b = np.radians(float(bearing_deg))
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return x * np.sin(b) - y * np.cos(b), x * np.cos(b) + y * np.sin(b)


def heading_to_frame(heading_deg, bearing_deg):
    """Unit (x, y) of compass headings in the track frame of :func:`east_north_to_frame`."""
    h = np.radians(np.asarray(heading_deg, dtype=float))
    return east_north_to_frame(bearing_deg, np.sin(h), np.cos(h))


def frame_bearing_deg(row, track=None, tolerance_deg=FRAME_BEARING_TOLERANCE_DEG):
    """Compass bearing of a run's track-frame +x axis.

    The reference list's ``true_heading`` is that bearing: 270 on Amedee's flight layout,
    279 on its hover (static) layout, 140 and 92.3 at Eglin.  It is not one value for the
    whole test, which is what the heading frame and the hovers need it for: a compass
    heading means nothing in the track frame until it is known.  When the track carries
    latitude and longitude and spans enough ground, the bearing is also fitted from them
    and must agree, so a wrong reference-list entry is refused rather than turning every
    heading by the error.  Without ``true_heading`` the
    fit alone is used.  Raises ValueError when neither is available.
    """
    listed = _float_or_nan(row.get('true_heading'))
    fitted = float('nan')
    lat0, lon0 = _float_or_nan(row.get('ref_lat')), _float_or_nan(row.get('ref_lon'))
    if track is not None and {'lat', 'lon', 'x'} <= set(track) and np.isfinite(lat0) and np.isfinite(lon0):
        east, north = local_east_north_ft(track['lat'], track['lon'], lat0, lon0)
        good = np.isfinite(east) & np.isfinite(north) & np.isfinite(track['x'])
        if good.sum() > 2 and np.ptp(east[good]) ** 2 + np.ptp(north[good]) ** 2 >= FRAME_FIT_MIN_SPAN_FT ** 2:
            a, b = np.linalg.lstsq(np.column_stack((east[good], north[good])), track['x'][good], rcond=None)[0]
            fitted = float(np.degrees(np.arctan2(a, b)) % 360.0)
    if np.isfinite(listed) and np.isfinite(fitted):
        difference = (fitted - listed + 180.0) % 360.0 - 180.0
        if abs(difference) > tolerance_deg:
            raise ValueError('the track frame fitted from latitude and longitude runs along {:.1f} deg, but the '
                             'reference list gives true_heading {:g}'.format(fitted, listed))
        return listed % 360.0
    if np.isfinite(listed):
        return listed % 360.0
    if np.isfinite(fitted):
        return fitted
    raise ValueError('no true_heading in the reference list and no latitude/longitude in the track to fit it '
                     'from, so a compass direction cannot be put into the track frame')


def ground_microphone_positions(test, run):
    """The run's layout's ground-board microphones in its track frame (ft), from the
    layout's microphone list (the reference list's ``mic_loc_file``).

    The list gives each microphone's latitude, longitude and ellipsoidal height in meters;
    the track's z is ellipsoidal height less ``ref_elips_ft`` (alt - z is that constant in
    every track).  Checked on run 285236: microphones 1, 3, 10, 17 and 36 land within
    0.6 ft in plan and 0.5 ft in height of the X, Y and Z their acoustic files carry.  Returns ``(positions, None)``,
    or ``(None, reason)`` when the list, the reference point or a column is missing.
    """
    row = test.by_run.get(run) or {}
    name = (row.get('mic_loc_file') or '').strip()
    if not name:
        return None, 'the reference list names no mic_loc_file'
    path = os.path.join(test.base, name)
    if not os.path.exists(path):
        return None, 'no microphone list {}'.format(path)
    lat0, lon0 = _float_or_nan(row.get('ref_lat')), _float_or_nan(row.get('ref_lon'))
    elips0 = _float_or_nan(row.get('ref_elips_ft'))
    if not (np.isfinite(lat0) and np.isfinite(lon0) and np.isfinite(elips0)):
        return None, 'the reference list gives no ref_lat/ref_lon/ref_elips_ft'
    try:
        bearing = frame_bearing_deg(row)
    except ValueError as error:
        return None, str(error)
    lat, lon, height_m = [], [], []
    with _open_csv(path) as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []

        def column(prefix):
            return next((f for f in fields if f and f.strip().lower().startswith(prefix)), None)
        lat_key, lon_key, hgt_key = column('latitude'), column('longitude'), column('hgt')
        if None in (lat_key, lon_key, hgt_key):
            return None, '{} has no Latitude/Longitude/Hgt columns'.format(name)
        for entry in reader:
            if str(entry.get('insttype', '')).strip() not in GROUND_BOARD_INSTRUMENTS:
                continue
            values = [_float_or_nan(entry.get(key)) for key in (lat_key, lon_key, hgt_key)]
            if all(np.isfinite(values)):
                lat.append(values[0])
                lon.append(values[1])
                height_m.append(values[2])
    if not lat:
        return None, '{} lists no ground-board microphone with a position'.format(name)
    east, north = local_east_north_ft(lat, lon, lat0, lon0)
    x, y = east_north_to_frame(bearing, east, north)
    z = np.asarray(height_m) / 0.3048 - elips0
    return np.column_stack((x, y, z)), None


# --------------------------------------------------------------------------
# Track checks
# --------------------------------------------------------------------------

#: A run is refused when its tracked position comes closer than this (ft) to the ground
#: under it, the nearest ground board's height, anywhere in the window it is depropagated
#: from.  Below the microphones is impossible, and so is within 10 ft: the B407's antenna
#: reads 9-10 ft with the aircraft on the ground (runs 283421-2).  Every steady window of
#: the six aircraft but one stays above 27 ft; 285236 descends to 66 ft below the boards.
MIN_HEIGHT_ABOVE_ARRAY_FT = 10.0

#: A run is refused when its z departs from its own altitude (``alt - ref_elips_ft``) by
#: more than this (ft) in the window: z would then not be the height the frame says.  The
#: largest departure in any steady window of the six aircraft is 3.5 ft.
MAX_ALTITUDE_MISMATCH_FT = 10.0

#: A window is flagged when its along-track acceleration, averaged over
#: :data:`TRACK_CHECK_AVERAGING_S`, falls below minus this (ft/s^2, 1.8 kt/s) anywhere in it.
DECELERATION_FLAG_FT_S2 = 3.0

#: A window is flagged as ending at a decelerating level-off when, within
#: :data:`LEVEL_OFF_LOOKAHEAD_S` of its end, the aircraft both decelerates along track by
#: more than LEVEL_OFF_DECELERATION_FT_S2 and raises its flight path angle by more than
#: LEVEL_OFF_FPA_RISE_DEG over the window's mean (both averaged over
#: :data:`TRACK_CHECK_AVERAGING_S`).  Run 283101 (card D4) is the case: a level segment at
#: 75 ft whose window ends 2 s before an aborted pull-up.
LEVEL_OFF_DECELERATION_FT_S2 = 2.5
LEVEL_OFF_FPA_RISE_DEG = 1.5
LEVEL_OFF_LOOKAHEAD_S = 3.0

#: A window is flagged when its mean flight path angle is this far (deg) from the flight
#: card's ``fpa``: the window was not flown at the card's condition (283101, card -3 deg,
#: is a level segment).  The sphere is labeled with the window's own condition either way.
CARD_FPA_FLAG_DEG = 2.0

#: Running mean the acceleration and flight path tests take, s.
TRACK_CHECK_AVERAGING_S = 1.0


class TrackRefused(ValueError):
    """A run whose track fails :func:`check_track`; the message says why."""


class TrackCheck:
    """What :func:`check_track` found: ``refusals`` (reasons the run must not be
    depropagated), ``flags`` (worth knowing, not disqualifying) and ``metrics``."""

    def __init__(self, refusals, flags, metrics):
        self.refusals = list(refusals)
        self.flags = list(flags)
        self.metrics = dict(metrics)

    @property
    def refused(self):
        return bool(self.refusals)

    def __repr__(self):
        return 'TrackCheck(refusals={!r}, flags={!r})'.format(self.refusals, self.flags)


def _running_mean(values, samples):
    values = np.asarray(values, dtype=float)
    samples = int(max(1, min(samples, values.size)))
    if values.size == 0:
        return values
    return np.convolve(values, np.ones(samples) / samples, mode='valid')


def height_above_array(x, y, z, ground_mics=None):
    """Height (ft) of track points above the ground under them, and what that ground was.

    The ground under a point is the height of the ground board nearest it in plan
    (``ground_mics``, (n, 3) in the track frame); without boards, the frame's z = 0.
    """
    x, y, z = (np.asarray(v, dtype=float) for v in (x, y, z))
    if ground_mics is None:
        return z.copy(), 'frame z = 0'
    mics = np.asarray(ground_mics, dtype=float).reshape(-1, 3)
    nearest = np.argmin((x[:, None] - mics[None, :, 0]) ** 2 + (y[:, None] - mics[None, :, 1]) ** 2, axis=1)
    return z - mics[nearest, 2], 'nearest of {} ground boards'.format(mics.shape[0])


def check_track(track, index_start, index_stop, *, ground_mics=None, ref_elips_ft=None, card_fpa_deg=None,
                min_height_ft=MIN_HEIGHT_ABOVE_ARRAY_FT, max_altitude_mismatch_ft=MAX_ALTITUDE_MISMATCH_FT,
                deceleration_flag_ft_s2=DECELERATION_FLAG_FT_S2,
                level_off_deceleration_ft_s2=LEVEL_OFF_DECELERATION_FT_S2,
                level_off_fpa_rise_deg=LEVEL_OFF_FPA_RISE_DEG, level_off_lookahead_s=LEVEL_OFF_LOOKAHEAD_S,
                card_fpa_flag_deg=CARD_FPA_FLAG_DEG, averaging_s=TRACK_CHECK_AVERAGING_S, motion_flags=True):
    """Sanity of a loaded track (:func:`load_track`) over the window ``[index_start, index_stop)``
    it is to be depropagated from.

    Refusals, each a reason the run's depropagation geometry cannot be right:

    * the aircraft comes within ``min_height_ft`` of the ground under it, or goes below it,
      anywhere in the window.  The ground under each sample is the height of the nearest
      ground board in plan (``ground_mics``, (n, 3) in the track frame, from
      :func:`ground_microphone_positions`), which follows a sloping array (Eglin's boards
      span 25 ft); without them it is the frame's z = 0, the reference point's height,
      which on the Amedee lakebed is within 2 ft of every board.  None turns the test off.
      Known case: B407 run 285236 (card D19), whose steady window reaches 66 ft below the
      nearest boards where its twin 285235 stays 40 ft above them: the tracking altitude, not the frame,
      is wrong (its z follows its alt exactly).
    * z does not follow the track's own altitude: ``|alt - ref_elips_ft - z|`` exceeds
      ``max_altitude_mismatch_ft`` (needs ``alt`` in the track and ``ref_elips_ft``; None
      turns the test off), or the altitude is not finite.  A caller that corrects a track's
      height (the R66's altitude reads about 30 ft low) must shift ``alt`` with ``z``, or
      this refuses it.
    * z is not finite somewhere in the window.

    Flags (``motion_flags``; off for a hover, whose window is the whole record):

    * along-track deceleration (d|v_h|/dt, averaged over ``averaging_s``) beyond
      ``deceleration_flag_ft_s2`` inside the window;
    * the window ends at a decelerating level-off: within ``level_off_lookahead_s`` after
      it, deceleration beyond ``level_off_deceleration_ft_s2`` together with a flight path
      angle more than ``level_off_fpa_rise_deg`` above the window's mean.  The sphere holds
      only the window's emission points, so this is no fault in the sphere; it says the
      condition was left in a maneuver right after, as a descent's flare usually is.
    * the window's mean flight path angle is more than ``card_fpa_flag_deg`` from the
      card's ``card_fpa_deg``.

    Deceleration is not refused: :func:`steady_window` holds speed within a few knots of
    the window's median, which bounds any sustained deceleration inside it.

    Returns a :class:`TrackCheck`.
    """
    refusals, flags = [], []
    window = slice(int(index_start), int(index_stop))
    time = np.asarray(track['time'], dtype=float)
    t0 = float(time[0])
    x = np.asarray(track['x'], dtype=float)[window]
    y = np.asarray(track['y'], dtype=float)[window]
    z = np.asarray(track['z'], dtype=float)[window]
    t = time[window]
    metrics = dict(min_height_above_array_ft=float('nan'), mean_height_above_array_ft=float('nan'),
                   max_altitude_mismatch_ft=float('nan'), min_along_track_acceleration_ft_s2=float('nan'),
                   plane='frame z = 0')
    if z.size == 0:
        return TrackCheck(['the window is empty'], [], metrics)
    if not np.all(np.isfinite(z) & np.isfinite(x) & np.isfinite(y)):
        refusals.append('its position is not finite in the window')
        return TrackCheck(refusals, flags, metrics)

    height, metrics['plane'] = height_above_array(x, y, z, ground_mics)
    lowest = int(np.argmin(height))
    metrics['min_height_above_array_ft'] = float(height[lowest])
    metrics['mean_height_above_array_ft'] = float(np.mean(height))
    if min_height_ft is not None and height[lowest] < float(min_height_ft):
        where = 't = {:.1f} s, x = {:.0f} ft'.format(t[lowest] - t0, x[lowest])
        if height[lowest] < 0.0:
            refusals.append('the aircraft goes {:.0f} ft below the microphones at {} ({})'.format(
                -height[lowest], where, metrics['plane']))
        else:
            refusals.append('the aircraft comes within {:.1f} ft of the microphones at {}, under the {:g} ft '
                            'minimum ({})'.format(height[lowest], where, float(min_height_ft), metrics['plane']))

    elips = float('nan') if ref_elips_ft is None else float(ref_elips_ft)
    if 'alt' in track and np.isfinite(elips):
        mismatch = np.abs(np.asarray(track['alt'], dtype=float)[window] - elips - z)
        worst = float(np.max(mismatch))
        metrics['max_altitude_mismatch_ft'] = worst
        if max_altitude_mismatch_ft is not None:
            if not np.isfinite(worst):
                refusals.append('its altitude is not finite in the window, so z cannot be checked against it')
            elif worst > float(max_altitude_mismatch_ft):
                refusals.append('its z departs from its altitude less ref_elips_ft by up to {:.1f} ft (at most '
                                '{:g}; a caller that corrects z must correct alt by the same amount)'.format(
                                    worst, float(max_altitude_mismatch_ft)))

    if motion_flags and 'vx' in track and 'vy' in track and time.size > 2:
        step = float(np.median(np.diff(time)))
        samples = max(1, int(round(float(averaging_s) / step))) if step > 0 else 1
        speed = np.hypot(np.asarray(track['vx'], dtype=float), np.asarray(track['vy'], dtype=float))
        along = np.gradient(speed, time)
        inside = _running_mean(along[window], samples)
        if inside.size:
            metrics['min_along_track_acceleration_ft_s2'] = float(np.min(inside))
            if deceleration_flag_ft_s2 is not None and np.min(inside) < -float(deceleration_flag_ft_s2):
                flags.append('it decelerates at {:.1f} ft/s^2 inside the window'.format(-np.min(inside)))
        stop = int(index_stop)
        after = slice(stop, int(np.searchsorted(time, time[stop - 1] + float(level_off_lookahead_s), side='right')))
        if (level_off_deceleration_ft_s2 is not None and level_off_fpa_rise_deg is not None
                and 'fpa_deg' in track and after.stop - after.start >= samples):
            fpa = np.asarray(track['fpa_deg'], dtype=float)
            deceleration = -float(np.min(_running_mean(along[after], samples)))
            rise = float(np.max(_running_mean(fpa[after], samples)) - np.mean(fpa[window]))
            if deceleration > float(level_off_deceleration_ft_s2) and rise > float(level_off_fpa_rise_deg):
                flags.append('the window ends at a decelerating level-off: within {:g} s of its end the '
                             'aircraft decelerates at {:.1f} ft/s^2 and its flight path angle rises {:.1f} deg'
                             .format(float(level_off_lookahead_s), deceleration, rise))
        card = float('nan') if card_fpa_deg is None else float(card_fpa_deg)
        if card_fpa_flag_deg is not None and np.isfinite(card) and 'fpa_deg' in track:
            flown = float(np.mean(np.asarray(track['fpa_deg'], dtype=float)[window]))
            if abs(flown - card) > float(card_fpa_flag_deg):
                flags.append('its window is flown at {:+.1f} deg, not the card\'s {:+g} deg'.format(flown, card))
    return TrackCheck(refusals, flags, metrics)


def run_track_check(test, run, track, index_start, index_stop, hover=False, **thresholds):
    """:func:`check_track` for one run of ``test``, with the run's reference-list row
    (``ref_elips_ft``, the card's ``fpa``) and its layout's ground boards
    (:func:`ground_microphone_positions`); the frame's z = 0 stands in for them, with a
    warning, when the microphone list cannot be read.  A hover's motion flags are off."""
    row = test.by_run.get(run) or {}
    mics, reason = ground_microphone_positions(test, run)
    if mics is None:
        logging.warning('Run %s: checking heights against the frame\'s z = 0, not the ground boards: %s',
                        run, reason)
    return check_track(track, index_start, index_stop, ground_mics=mics,
                       ref_elips_ft=_float_or_nan(row.get('ref_elips_ft')),
                       card_fpa_deg=None if hover else _float_or_nan(row.get('fpa')),
                       motion_flags=not hover, **thresholds)


# --------------------------------------------------------------------------
# Run metadata
# --------------------------------------------------------------------------

#: Specific gas constant of dry air, J/(kg K), and standard gravity, m/s^2.
DRY_AIR_GAS_CONSTANT = 287.058
STANDARD_GRAVITY = 9.80665


def run_gross_weight_lb(row):
    """The run's gross weight from its reference-list row (``gross_weight``, pounds), or NaN.

    The B407's read 3173-3824 lb, against the 2250 kg (4960 lb) its vehicle file's nominal
    C_T is taken at."""
    weight = _float_or_nan((row or {}).get('gross_weight'))
    return weight if np.isfinite(weight) and weight > 0.0 else float('nan')


def air_density_kg_m3(atmosphere, height_ft=0.0):
    """Moist-air density (kg/m^3) at ``height_ft`` above where ``atmosphere`` was measured.

    rho = (p - 0.378 e) / (R_d T), e the vapor pressure from the relative humidity and
    ISO 9613-1's saturation pressure (the one absorption uses), carried up hydrostatically
    at the measured temperature: p(h) = p exp(-g h / (R_d T_v)).  Over the few hundred feet
    of these runs the temperature's lapse moves it by well under 0.1 %; the height itself
    by about 3 % per 1000 ft, which is why it is applied.
    """
    temperature = float(atmosphere.temperature)
    pressure_pa = 1000.0 * float(atmosphere.pressure)
    if not (temperature > 0.0 and pressure_pa > 0.0):
        return float('nan')
    vapor_pa = 1000.0 * float(atmosphere.relative_humidity) / 100.0 * float(atmosphere.saturation_pressure)
    if not 0.0 <= vapor_pa < pressure_pa:
        return float('nan')
    virtual_temperature = temperature / (1.0 - 0.378 * vapor_pa / pressure_pa)
    scale = np.exp(-STANDARD_GRAVITY * float(height_ft) * 0.3048 / (DRY_AIR_GAS_CONSTANT * virtual_temperature))
    return float(scale * (pressure_pa - 0.378 * vapor_pa) / (DRY_AIR_GAS_CONSTANT * temperature))


def normalize_wind(wind):
    """A caller's wind at the aircraft, as ``dict(east, north, units, source)``: the
    components it blows toward, in the units declared.

    ``wind`` gives either ``east`` and ``north`` (the direction the air moves toward), or
    ``speed`` and ``direction_from_deg`` (meteorological: the compass direction, true north,
    it blows from), plus ``units`` (one of :data:`flight_acoustics.WIND_SPEED_UNITS`) and
    ``source`` (:data:`flight_acoustics.WIND_SOURCES`, '+'-joined for a combination).
    Nothing is assumed: the units are the caller's, and a magnetic direction must be turned
    to true north before it is given.
    """
    if not isinstance(wind, dict):
        raise ValueError('wind must be a dict, not {!r}'.format(type(wind).__name__))
    units = str(wind.get('units', '')).strip()
    if units not in fa.WIND_SPEED_UNITS:
        raise ValueError('wind units must be declared, one of {}, not {!r}'.format(
            sorted(fa.WIND_SPEED_UNITS), wind.get('units')))
    source = fa.normalize_wind_source(wind.get('source', ''))
    if source == 'none':
        raise ValueError("a wind with source 'none' is no wind; pass wind=None")
    vector = {'east', 'north'} <= set(wind)
    polar = {'speed', 'direction_from_deg'} <= set(wind)
    if vector == polar:
        raise ValueError('give the wind as east and north, or as speed and direction_from_deg, '
                         'not {}'.format('both' if vector else 'neither'))
    if vector:
        east, north = float(wind['east']), float(wind['north'])
    else:
        speed, direction = float(wind['speed']), np.radians(float(wind['direction_from_deg']))
        if speed < 0.0:
            raise ValueError('wind speed must not be negative, got {}'.format(speed))
        # Blowing from a bearing is moving toward the opposite one.
        east, north = -speed * np.sin(direction), -speed * np.cos(direction)
    if not (np.isfinite(east) and np.isfinite(north)):
        raise ValueError('wind components must be finite')
    return dict(east=float(east), north=float(north), units=units, source=source)


def wind_components(wind, reference_east_north, ground_velocity_east_north_ft_s):
    """Along- and cross-reference components of a :func:`normalize_wind` wind, in its own
    units, and the mean horizontal airspeed (knots).

    ``reference_east_north`` is the direction the components are taken along (the mean
    ground track, or the heading for a hover); along is positive toward it (a tailwind),
    cross positive toward its starboard side.  ``ground_velocity_east_north_ft_s`` is the
    (n, 2) horizontal ground velocity over the window; the airspeed is the mean of
    |v_ground - wind|, as the sphere's SPEED is the mean of |v_ground|.
    """
    reference = np.asarray(reference_east_north, dtype=float)
    reference = reference / np.hypot(*reference)
    starboard = np.array([reference[1], -reference[0]])
    w = np.array([wind['east'], wind['north']])
    along, cross = float(w @ reference), float(w @ starboard)
    w_ft_s = w * fa.WIND_SPEED_UNITS[wind['units']] / 0.3048
    ground = np.asarray(ground_velocity_east_north_ft_s, dtype=float).reshape(-1, 2)
    airspeed_knots = float(np.mean(np.hypot(*(ground - w_ft_s).T))) * 0.3048 / fa.KNOT_MPS
    return along, cross, airspeed_knots


def wind_in_frame_ft_s(wind, bearing_deg):
    """A :func:`normalize_wind` wind as (Wx, Wy, 0) in ft/s in the track frame of
    :func:`east_north_to_frame` (+x along compass ``bearing_deg``, deg): the air's horizontal
    velocity, for :func:`refracted_rays.uniform_wind_ray_model`."""
    scale = fa.WIND_SPEED_UNITS[wind['units']] / 0.3048
    x, y = east_north_to_frame(bearing_deg, wind['east'] * scale, wind['north'] * scale)
    return np.array([float(x), float(y), 0.0])


# --------------------------------------------------------------------------
# Acoustic loading
# --------------------------------------------------------------------------

def _signal_header(handle):
    """``(sample_rate, start_time, location)`` of one open channel."""
    return (float(handle.sample_rate), float(handle.start_time),
            np.array([float(handle.X), float(handle.Y), float(handle.Z)]))


def _read_window(handle, time_range=None, scale=1.0):
    """Read an open channel, optionally trimming to an absolute time range."""
    sample_rate, start_time, location = _signal_header(handle)
    n_samples = handle.variables['pressure'].shape[0]
    if time_range is None:
        first, last = 0, n_samples
    else:
        first = int(np.floor((float(time_range[0]) - start_time) * sample_rate))
        last = int(np.ceil((float(time_range[1]) - start_time) * sample_rate)) + 1
        first = max(first, 0)
        last = min(max(last, first + 2), n_samples)
    pressure = handle.variables['pressure'][first:last].astype(float).ravel()
    time = start_time + (first + np.arange(pressure.size, dtype=float)) / sample_rate
    return pressure * float(scale), time, location, sample_rate


def _read_signal(path, time_range=None, scale=1.0):
    """Read one channel, optionally trimming to an absolute time range."""
    with Dataset(path, mode='r') as handle:
        return _read_window(handle, time_range, scale)


def load_run_channels(test, run, mics, time_range=None, ground_board_scale=0.5):
    """Load one run's microphone channels.

    ``ground_board_scale`` accounts for pressure doubling at the ground board;
    the demo scripts apply the same 0.5 factor.

    ``time_range`` may also be a function of the channels' locations (an
    (n, 3) array, in ``mics`` order, of those that could be opened) returning
    the range, for a range that depends on where the microphones are: each
    file is then still opened only once.
    """
    with contextlib.ExitStack() as stack:
        opened = []
        for mic_number in mics:
            path = test.acoustic_files[run][mic_number]
            try:
                handle = stack.enter_context(Dataset(path, mode='r'))
                location = _signal_header(handle)[2]
            except Exception as error:                  # noqa: BLE001 - one bad channel must not kill the run
                warnings.warn('Run {}: could not read mic {} ({}); skipping it'
                              .format(run, mic_number, error))
                continue
            opened.append((mic_number, handle, location))
        if callable(time_range):
            time_range = (time_range(np.array([location for _, _, location in opened]))
                          if opened else None)

        locations, pressures, times, kept = [], [], [], []
        for mic_number, handle, location in opened:
            try:
                pressure, time, _, _ = _read_window(handle, time_range, ground_board_scale)
            except Exception as error:                  # noqa: BLE001 - one bad channel must not kill the run
                warnings.warn('Run {}: could not read mic {} ({}); skipping it'
                              .format(run, mic_number, error))
                continue
            if pressure.size < 2:
                continue
            locations.append(location)
            pressures.append(pressure)
            times.append(time)
            kept.append(mic_number)
    if not kept:
        raise ValueError('Run {}: no usable microphone channels'.format(run))
    return np.array(locations), pressures, times, kept


def load_ambient_channels(test, ambient_run, mics, max_duration_s=30.0,
                          ground_board_scale=0.5):
    """Load the ambient recording for ``mics``, in that order.

    Every requested microphone must appear in the ambient run: gating a channel
    against a different channel's noise floor compares two positions, and
    silently dropping one would misalign the ambient list against the pressure
    list that :func:`flight_acoustics.depropagate_hemisphere` pairs it with.
    Callers filter ``mics`` to the channels both runs have.

    A median PSD converges well inside ``max_duration_s``, so only that much of
    the recording is read.
    """
    available = test.acoustic_files.get(ambient_run, {})
    missing = [m for m in mics if m not in available]
    if missing:
        raise ValueError('Ambient run {} has no recording for mics {}'
                         .format(ambient_run, missing))
    pressures, times = [], []
    for mic_number in mics:
        with Dataset(available[mic_number], mode='r') as handle:
            start_time = _signal_header(handle)[1]
            pressure, time, _, _ = _read_window(
                handle, (start_time, start_time + float(max_duration_s)), ground_board_scale)
        pressures.append(pressure)
        times.append(time)
    return pressures, times, list(mics)


# --------------------------------------------------------------------------
# Ground-plate response
# --------------------------------------------------------------------------

_PLATE_TABLES = {}


def default_plate_table_directory():
    """Where computed plate tables are kept between runs."""
    return os.path.join(os.path.expanduser('~'), '.cache', 'panam', 'plate_tables')


def _plate_table_directory(directory=None):
    """``directory`` as an absolute path, '~' expanded; None gives
    :func:`default_plate_table_directory`."""
    return os.path.abspath(os.path.expanduser(directory or default_plate_table_directory()))


def _plate_table_speed(sound_speed_ft_s):
    """The sound speed (ft/s) a plate table is computed at: ``sound_speed_ft_s``
    rounded to a step of :data:`PLATE_TABLE_SOUND_SPEED_STEP`."""
    step = np.log1p(PLATE_TABLE_SOUND_SPEED_STEP)
    return float(np.exp(np.round(np.log(sound_speed_ft_s) / step) * step))


def plate_table(instrument, sound_speed_ft_s, bands, ground=None, directory=None, **options):
    """The axisymmetric BEM table (:func:`axisymmetric_bem.table`) for one board type.

    ``sound_speed_ft_s`` in ft/s, ``bands`` the band center frequencies in Hz.
    Computed at the sound speed rounded to :data:`PLATE_TABLE_SOUND_SPEED_STEP`
    and cached in memory and in ``directory`` (default
    :func:`default_plate_table_directory`); a table takes 15-45 s.
    ``options`` (mesh, quadrature, directions; see
    :func:`axisymmetric_bem.table_settings`) go to :func:`axisymmetric_bem.table`.

    The cache key holds :data:`axisymmetric_bem.PLATE_TABLE_VERSION`, the
    microphone height, the ground, the rounded sound speed, every band and
    every setting of :func:`axisymmetric_bem.table_settings`.  Each table is an
    .npz file (no pickle) that stores its key; one whose key differs, or that
    cannot be read, is recomputed and replaced.  Files from before the version
    (.pkl) are ignored.
    """
    import axisymmetric_bem as ab
    import hashlib
    import json
    ground = dict(SITE_GROUND if ground is None else ground)
    speed = _plate_table_speed(sound_speed_ft_s)
    bands = np.asarray(bands, dtype=float)
    mic_height = BOARD_MIC_HEIGHT_FT[instrument]
    # Keyed on what the table depends on, not the instrument's name.
    key = json.dumps(dict(version=ab.PLATE_TABLE_VERSION, ground=ground, sound_speed_ft_s=speed,
                          bands=bands.tolist(), settings=ab.table_settings(**options, mic_height=mic_height)),
                     sort_keys=True, default=lambda value: value.tolist())     # numpy values
    table = _PLATE_TABLES.get(key)
    if table is not None:
        return table
    # The name is for people; the hash makes it unique and the stored key decides.
    ground_name = '_'.join('{}{}'.format(k, ground[k]) for k in sorted(ground))
    name = 'h{:.2f}mm_{}_c{:.2f}_b{}-{:g}-{:g}_v{}_{}.npz'.format(
        mic_height * 304.8, ground_name, speed, bands.size, bands[0], bands[-1], ab.PLATE_TABLE_VERSION,
        hashlib.sha256(key.encode()).hexdigest()[:12])
    directory = _plate_table_directory(directory)
    path = os.path.join(directory, name)
    table = None
    if os.path.exists(path):
        try:
            table, stored_key = ab.read_table_npz(path)
        except (OSError, ValueError) as error:
            logging.warning('Ignoring the unreadable plate table %s: %s', path, error)
            stored_key = None
        if stored_key != key:
            table = None
    if table is None:
        logging.info('Computing the plate table %s', name)
        flow = ground.get('sigma', ground.get('sigma_e'))
        table = ab.table(bands, speed, flow_resistance=flow, ground=ground, mic_height=mic_height, **options)
        os.makedirs(directory, exist_ok=True)
        # Parallel builds may race; either copy is right.
        with fa._atomic_output(path) as temporary, open(temporary, 'wb') as handle:
            ab.write_table_npz(handle, table, key)
    _PLATE_TABLES[key] = table
    return table


def plate_table_cache(directory=None):
    """The files in the plate-table cache ``directory`` (default
    :func:`default_plate_table_directory`): one dict per file, sorted by name,
    with ``path``, ``size`` (bytes) and ``status``: 'current' (an .npz of
    this :data:`axisymmetric_bem.PLATE_TABLE_VERSION`), 'old version N',
    'old format' (a .pkl from before versioning) or 'unreadable'."""
    import axisymmetric_bem as ab
    import json
    directory = _plate_table_directory(directory)
    records = []
    for path in sorted(glob.glob(os.path.join(directory, '*.npz')) + glob.glob(os.path.join(directory, '*.pkl'))):
        if path.endswith('.pkl'):
            status = 'old format'
        else:
            try:
                version = json.loads(ab.read_table_key(path))['version']
                status = 'current' if version == ab.PLATE_TABLE_VERSION else 'old version {}'.format(version)
            except (OSError, ValueError, KeyError, TypeError):
                status = 'unreadable'
        records.append(dict(path=path, size=os.path.getsize(path), status=status))
    return records


def clear_plate_table_cache(directory=None):
    """Delete the plate tables (.npz, old .pkl and leftover .tmp files) in
    ``directory`` (default :func:`default_plate_table_directory`), but not the
    directory or any other file.  Returns the number of files deleted."""
    directory = _plate_table_directory(directory)
    paths = [path for pattern in ('*.npz', '*.pkl', '.*.npz.*.tmp', '*.npz.*.tmp', '*.pkl.*.tmp')
             for path in glob.glob(os.path.join(directory, pattern))]
    for path in paths:
        os.remove(path)
    return len(paths)


def plate_response(tables, mirror, sound_speed_ft_s):
    """A ``receiver_response_db`` for :func:`flight_acoustics.depropagate_hemisphere`.

    ``tables[im]`` is microphone ``im``'s plate table and ``mirror[im]`` puts
    its offset on -y (outboard for a microphone on the -y side of the track).
    """
    import axisymmetric_bem as ab

    def response(im, bands, offset):
        offset = np.asarray(offset, dtype=float)
        return ab.board_level(bands, offset[:, 2], np.hypot(offset[:, 0], offset[:, 1]),
                              sound_speed_ft_s, tables[im], source_dx=offset[:, 0],
                              source_dy=offset[:, 1], mirror_y=bool(mirror[im]))
    return response


# --------------------------------------------------------------------------
# Atmosphere
# --------------------------------------------------------------------------

def sound_speed_ft_s(atmosphere):
    """``atmosphere``'s speed of sound in ft/s, the dataset's length unit."""
    return float(fa.unit_conversion.len_conv(atmosphere.soundspeed, from_units='m', to_units='ft'))


def _weather_station_files(test, run):
    """The ground weather station files of ``run``'s day, sorted (see :func:`run_atmosphere`)."""
    return sorted(glob.glob(os.path.join(test.base, test.aircraft + '_Weather',
                                         test.aircraft + '_Ground[_ ]Stations',
                                         '{}_{}_*.csv'.format(test.aircraft, run[:3]))))


def run_atmosphere(test, run, fallback=None):
    """Atmosphere measured by the ground weather stations at the time of ``run``.

    Depropagation *undoes* absorption, so the atmosphere used here sets how much
    the high-frequency bands get multiplied by.  Assuming a dry standard day
    when the test day was humid would inflate them: at 20 C absorption at
    3.15 kHz is about 49 dB/km at 20 % relative humidity but roughly a third of
    that at 70 %, and that difference is applied over kilometers of slant range.

    The stations report ``airtemp`` in degrees Fahrenheit, pressure in kPa and
    humidity in percent.  The file does not say so; the balloon sondes, which
    use the same columns, also log air density, and p / (R rho) reproduces
    their airtemp as Fahrenheit (e.g. AS350B3 day 289: 18.1 against 18.7 F)
    for every aircraft in the dataset.

    The stations folder is ``<aircraft>_Ground_Stations`` for every aircraft
    but EC130B4, whose folder is ``EC130B4_Ground Stations``; both spellings
    are read.
    """
    row = test.by_run.get(run)
    try:
        run_seconds = float(row.get('utc_secs_from_mid_start'))
    except (TypeError, ValueError, AttributeError):
        run_seconds = None

    samples = []
    for path in _weather_station_files(test, run):
        try:
            data = np.genfromtxt(path, delimiter=',', names=True)
        except Exception:                               # noqa: BLE001
            continue
        if data.size < 1:
            continue
        names = data.dtype.names
        if not {'utcsec', 'airtemp', 'humidity', 'pressure'} <= set(names):
            continue
        seconds = np.atleast_1d(data['utcsec']).astype(float)
        index = (int(np.argmin(np.abs(seconds - run_seconds)))
                 if run_seconds is not None else seconds.size // 2)
        temperature = float(np.atleast_1d(data['airtemp'])[index])
        humidity = float(np.atleast_1d(data['humidity'])[index])
        pressure = float(np.atleast_1d(data['pressure'])[index])
        # Stations report -1000 for a dead channel.
        if temperature < -100 or humidity < 0 or pressure <= 0:
            continue
        samples.append((temperature, humidity, pressure))

    if not samples:
        if fallback is not None:
            return fallback
        raise ValueError('No usable ground weather for run ' + str(run))

    temperature, humidity, pressure = np.mean(np.array(samples, dtype=float), axis=0)
    return fa.Atmosphere(temperature=(temperature - 32.0) / 1.8 + 273.15,
                         pressure=pressure,
                         relative_humidity=humidity)


# --------------------------------------------------------------------------
# Sphere generation
# --------------------------------------------------------------------------

def sphere_title(sphere_prefix, run_number, day_of_year, year=TEST_YEAR):
    """Reproduce the legacy spheres' title, padding included.

    They read ``'Be407  Run 220 10/12/2017 Noise Abatement Flight Test 1/3
    Octave Band'`` blank padded to 77 characters -- the shape a Fortran
    CHARACTER write leaves behind.  Run ids in this dataset carry the day of
    year (285 -> 12 October), not the date.
    """
    import datetime
    date = datetime.date(int(year), 1, 1) + datetime.timedelta(days=int(day_of_year) - 1)
    text = '{}  Run {} {} Noise Abatement Flight Test 1/3 Octave Band'.format(
        sphere_prefix, run_number, date.strftime('%m/%d/%Y'))
    return text.ljust(AAM_TITLE_WIDTH)


def legacy_sphere_grid(path):
    """PHI/THETA/FREQUENCY grids and radius from an existing sphere, if present."""
    if not path or not os.path.exists(path):
        return None
    with Dataset(path, mode='r') as handle:
        return dict(phi_deg=np.array(handle['PHI'][:], dtype=float),
                    theta_deg=np.array(handle['THETA'][:], dtype=float),
                    band_centers_hz=np.array(handle['FREQUENCY'][:], dtype=float),
                    radius_ft=float(np.ravel(handle['RADIUS'][:])[0]))


def _sphere_grid(reference):
    """(band centers Hz, PHI deg, THETA deg) of a build: a :func:`legacy_sphere_grid`
    ``reference``'s, or :data:`LEGACY_BAND_CENTERS_HZ` and None, None (the writer's
    default grid) without one."""
    if reference is None:
        return LEGACY_BAND_CENTERS_HZ, None, None
    return reference['band_centers_hz'], reference['phi_deg'], reference['theta_deg']


def _run_atmosphere_or_standard_day(test, run):
    """(atmosphere, error): :func:`run_atmosphere` and None, or, without usable
    ground weather, :data:`STANDARD_DAY` and the ValueError that said so."""
    try:
        return run_atmosphere(test, run), None
    except ValueError as error:
        return fa.Atmosphere(**STANDARD_DAY), error


def _mics_with_ambient(test, mics, ambient_run):
    """The microphones of ``mics`` that have a recording in ``ambient_run``."""
    return [m for m in mics if m in test.acoustic_files.get(ambient_run, {})]


def tone_split_paths(output_path):
    """Where ``build_sphere(tone_split=True)`` writes the parts of the sphere at ``output_path``:
    ``{'tonal': <dir>/tonal/<stem>_tonal.nc, 'broadband': <dir>/broadband/<stem>_broadband.nc}``."""
    directory, name = os.path.split(os.path.abspath(output_path))
    stem, extension = os.path.splitext(name)
    return {part: os.path.join(directory, part, '{}_{}{}'.format(stem, part, extension))
            for part in ('tonal', 'broadband')}


def norah2_file_name(sphere_prefix, speed_knots, fpa_deg, run_number):
    """``[type]_[procedure]_[IAS]kts_[gamma]deg_[run].hem``, as NORAH2 names its files.

    The run number is added because a NORAH2 file is one merged condition and
    these are single runs, several of which share a nominal condition.
    """
    if fpa_deg < -fa.LEVEL_FLIGHT_TOLERANCE:
        procedure = 'Approach'
    elif fpa_deg > fa.LEVEL_FLIGHT_TOLERANCE:
        procedure = 'Takeoff'
    else:
        procedure = 'Flyover'
    return '{}_{}_{:.0f}kts_{:g}deg_{}.hem'.format(
        sphere_prefix, procedure, speed_knots, round(abs(fpa_deg), 1), run_number)


#: build_sphere settings copied, as given, into its record and the manifest; the
#: rest are summarized there already (interpolation, rim_range, board_correction,
#: min_elevation_deg, max_array_range_ft, azimuth_reference) or are in the sphere's
#: provenance stamp only.
RECORDED_SETTINGS = (
    'band_snr_gate_db', 'gate_ambient', 'ambient_fallback_percentile',
    'max_absorption_correction_db', 'max_response_correction_db', 'third_octave_method',
    'tone_aware', 'remove_doppler', 'point_stride', 'r_ref_ft', 'window_time', 'window_overlap',
    'azi_step', 'elv_step', 'rmax', 'apply_absorption_deprop', 'max_propagation_range_ft',
    'min_steady_duration_s', 'nose_from_heading', 'flip_y_for_geometry', 'track_check',
)


def _sphere_settings(arguments):
    """The provenance settings and inputs of a :func:`build_sphere` call.

    ``arguments`` maps every parameter of :func:`build_sphere` to its value,
    defaults included (its ``locals()`` on entry, or a bound signature).
    Returns ``(settings, inputs)`` for :func:`provenance.build_provenance`:
    the processing arguments, and what identifies the sphere, with paths
    reduced to file names.
    """
    settings = dict(arguments)
    test, run = settings.pop('test'), settings.pop('run')
    # The node statistics add diagnostic variables and change no level: left off, they stay
    # out of the settings so that a default build keeps the hash it had before the option.
    if not settings.get('node_statistics', False):
        settings.pop('node_statistics', None)
    # Whether an existing file may be replaced changes no level: a resumed build with
    # overwrite=False must still match the sphere a default build wrote.
    settings.pop('overwrite', None)
    inputs = dict(test=test.aircraft, run=str(run))
    for name in ('output_path', 'reference_sphere', 'norah2_directory', 'plate_table_directory',
                 'samples_path'):
        value = settings.pop(name)
        inputs[name] = None if value is None else provenance.path_name(value)
    # A reference sphere sets the grid and bands, so whether one was given is a setting.
    settings['reference_sphere_given'] = arguments['reference_sphere'] is not None
    return settings, inputs


def build_sphere(test, run, output_path, *, reference_sphere=None,
                 band_snr_gate_db=10.0, gate_ambient=True,
                 max_absorption_correction_db=30.0, max_response_correction_db=None,
                 ambient_fallback_percentile=None,
                 r_ref_ft=DEFAULT_R_REF_FT, window_time=0.5, window_overlap=0.5,
                 azi_step=10.0, elv_step=10.0, rmax=25.0, point_stride=1,
                 min_elevation_deg=10.0, max_array_range_ft=None,
                 max_propagation_range_ft=2000.0, min_steady_duration_s=8.0,
                 flip_y_for_geometry=False,
                 atmosphere=None, speed_of_sound_ft_s=None,
                 apply_absorption_deprop=True, overwrite=True, norah2_directory=None,
                 third_octave_method='fft', board_correction='plate_bem', ground=None,
                 plate_table_directory=None, ray_model=None, interpolation=None,
                 rim_elevation_deg=None, max_rim_range_ft=None, remove_doppler=False,
                 nose_from_heading=False, samples_path=None, tone_aware=False, tone_split=False,
                 azimuth_reference=None, wind=None, track_check=True,
                 min_height_above_array_ft=MIN_HEIGHT_ABOVE_ARRAY_FT, air_relative_doppler=False,
                 absorption_profile=None, absorption_band_method='center', node_statistics=False):
    """Depropagate one run into an AAM-style source sphere.

    ``track_check`` (default True) refuses, with :class:`TrackRefused` and the reason, a run
    whose track :func:`check_track` finds impossible over the window: within
    ``min_height_above_array_ft`` of the ground boards or below them (None turns that test
    off), or with a z that does not follow its altitude.  Its flags (a deceleration, a
    window ending at a decelerating level-off, a window flown away from the card's flight
    path angle) are logged and returned as ``track_flags``.

    ``azimuth_reference`` is what the sphere's azimuth is measured from: ``'track'``, the
    ground-velocity direction (the default for flight, as every sphere before), or
    ``'heading'``, the tracking file's INS heading put into the track frame by the run's
    :func:`frame_bearing_deg`.  Doppler, the convective Mach number, spreading and every
    other part of the depropagation still follow the ground velocity; only the filing
    azimuth turns.  On windy day 284 the B407 crabbed 10.6 deg (median) against 4.2 on
    calm day 286, so a track-filed sphere carries that day's crab in its azimuths.  A hover
    (``nose_from_heading``) is always heading-filed, and None picks 'heading' for it;
    'track' with ``nose_from_heading`` is refused.  Written to the sphere as the attribute
    ``azimuth_reference``, which ``build_empirical_database`` carries to the database.

    ``wind``, if given, is the wind at the aircraft over the window, as
    :func:`normalize_wind` takes it: components or speed and direction, with the ``units``
    and ``source`` the caller declares (nothing is assumed about either).  The sphere then
    records its along- and cross-track components (along the heading for a hover) in those
    units, and the run's horizontal airspeed.  Every sphere also records the run's gross
    weight (reference list) and its air density at the aircraft (the ground stations'
    T, p and RH, :func:`air_density_kg_m3`), NaN when unknown; see
    :func:`flight_acoustics.write_aam_hemisphere_netcdf`'s ``run_metadata``.

    ``remove_doppler`` files band power at the emitted frequency, not the received one
    (:func:`flight_acoustics.depropagate_hemisphere`), and sets the sphere's
    DOPPLER_SHIFT_REMOVED.  For a sphere a hover is synthesized from
    (``build_empirical_database(hover_source=)``), never for one NICE-OPS reads as flight.

    ``nose_from_heading`` orients the sphere by the tracked heading instead of the velocity:
    for a hover, whose velocity is a few tenths of a knot of drift in any direction.  The
    velocity handed to depropagation is then the heading's unit vector at 1e-3 ft/s, which
    sets the azimuth reference and leaves no Doppler or convective term.  The heading is put
    into the track frame by the run's own frame bearing (:func:`frame_bearing_deg`), not by
    the flight layout's 270 deg this once assumed: the hover layouts' frames run along
    279 deg (Amedee) and 92.3 deg (Eglin), so hovers built before 2026-10-05 have their
    azimuths turned by 9 deg (Amedee) or 178 deg (Eglin).

    ``tone_aware`` files each tone whole in the band of its own frequency
    (:func:`flight_acoustics.tone_aware_band_power`) instead of summing whole FFT bins.

    ``tone_split`` (needs ``tone_aware``) also writes the run's tonal and broadband parts
    (:func:`flight_acoustics.depropagate_hemisphere`'s ``decompose_tonal``) as spheres of their
    own, ``<name>_tonal.nc`` in a ``tonal`` directory and ``<name>_broadband.nc`` in a
    ``broadband`` directory beside ``output_path``.  They sum to the total sphere in energy.  They
    go in directories of their own because a database is built from every ``.nc`` in a
    directory, and the parts beside the total would count the run three times.  The returned
    dict names them under ``tonal_output`` and ``broadband_output``; ``samples_path`` then also
    holds ``tonal_db`` and ``broadband_db``.

    ``samples_path``, if given, also saves the scattered samples (before gridding) to that
    .npz: azimuth and elevation (deg, panam's convention: 180 ahead, elevation positive
    below the horizon), band centers, band levels (dB at ``r_ref_ft``, bands x samples),
    microphone, slant range and source height (ft).

    ``ray_model`` (see :mod:`refracted_rays`) depropagates along refracted rays
    through the run's atmosphere instead of straight lines in uniform air: the
    samples are filed at the rays' launch angles, spreading is undone over the
    ray tubes and absorption over the arcs, and the plate correction is taken at
    the rays' arrival angles.  ``min_elevation_deg`` then bounds the launch
    angle.  Default None: straight lines, as before.  ``ray_model='wind'`` depropagates
    along straight rays carried by the run's ``wind`` (required), uniform over the whole
    path, in the run's uniform atmosphere (:func:`refracted_rays.uniform_wind_ray_model`):
    a downwind sample travels faster and spreads over a shorter range, and a crosswind
    turns the direction it is filed at.  ``air_relative_doppler`` (with ``remove_doppler``
    and ``ray_model='wind'``) also takes the Doppler factor in the moving air; default False.

    ``max_rim_range_ft`` with ``rim_elevation_deg`` lets samples filed shallower than
    that come from as far as that range instead of ``max_propagation_range_ft``
    (:func:`flight_acoustics.depropagate_hemisphere`'s ``rim_range``), so that every
    pass contributes to the rim, not only the low ones.

    ``interpolation`` grids the samples with a radius chosen per node (see
    :func:`flight_acoustics.adaptive_idw_weights`): ``{}`` for the defaults,
    or settings to override them.  Default None: the fixed ``rmax``.

    ``board_correction`` removes the ground board's effect: ``'plate_bem'``
    (default) divides each band by the plate's modeled response for that
    frame's geometry -- the axisymmetric BEM of the plate on the site's
    ``ground`` (default :data:`SITE_GROUND`), per instrument type
    (:data:`BOARD_MIC_HEIGHT_FT`), microphone outboard of the track -- and
    ``'flat'`` applies the constant pressure-doubling factor 0.5 (-6 dB) the
    spheres were built with before 2026-09-26.  Against the co-located pole
    microphones the flat factor reads 2-4 dB high at mid frequencies at
    10-40 deg elevation, where the plate's response falls short of +6 dB.

    ``norah2_directory``, if given, also writes the same hemisphere there as a
    NORAH2 ``.hem`` file named by :func:`norah2_file_name` (see
    :func:`flight_acoustics.write_norah2_hemisphere`).  Its ACSPEED is the
    ground speed the AAM sphere is labeled with; the tracking data carries no
    airspeed.

    ``absorption_profile`` undoes each path's absorption through a layered atmosphere instead
    of the uniform ``atmosphere``: a mapping of ``altitude_ft`` (ft, in the track frame's z,
    the same datum as the track and microphone heights), ``temperature_k`` (K),
    ``relative_humidity`` (percent) and ``pressure_kpa`` (kPa), integrated along each straight
    path (:func:`panam_acoustics.atmosphere.layered_attenuation_db`).  The sphere still keeps
    the absorption over ``r_ref_ft`` in ``atmosphere``.  Default None: the uniform atmosphere.

    ``absorption_band_method`` (``'center'``, the default, ``'arp5534'`` or ``'integrated'``)
    forms the filter-bank bands' absorption (``third_octave_method='filter_bank'`` only; see
    :func:`flight_acoustics.depropagate_hemisphere`).

    ``node_statistics`` (default False) also writes, per ART point and band, the samples
    behind the gridded node nearest it (:func:`sphere_statistics.node_statistics_variables`):
    ``N_SAMPLES`` and ``AMPLITUDE_STD`` (dB, the weighted standard deviation of the sample
    levels about the node's energy mean), into every sphere this call writes, the tonal and
    broadband parts each with their own.  AAM and NICE-OPS do not read them.

    ``atmosphere``, if not given, is the ground stations' at the time of the run
    (:func:`run_atmosphere`).  A run whose day has no usable station record (AS350B3 day 292,
    R66 day 231) is depropagated in a 20 C, 20 % RH, 101.325 kPa standard day with a logged
    warning, and its sphere records the air density as NaN with source ``'none'``.

    The sphere carries the provenance stamp of :mod:`provenance`: the processing
    arguments of this call as canonical JSON, and what identifies the sphere (the
    aircraft, the run, and the output, reference, NORAH2, plate-table and samples
    paths by file name) in ``panam_inputs``.  Equal processing settings give an
    equal ``settings_sha256`` across runs.

    Returns a dict describing what was processed, so a batch caller can log and
    audit it without re-opening the output.  It includes the settings of
    :data:`RECORDED_SETTINGS` as given, the code version (``panam_version``) and the
    stamp's ``settings_sha256``.
    """
    settings, inputs = _sphere_settings(locals())
    if rim_elevation_deg is None:
        rim_elevation_deg = DEFAULT_RIM_ELEVATION_DEG
    if tone_split and not tone_aware:
        raise ValueError('tone_split needs tone_aware=True')
    row = test.by_run[run]
    input_signature = provenance.file_signature(run_input_files(test, run, gate_ambient, reference_sphere,
                                                                 ray_model))
    track = load_track(test.track_path(run))
    if nose_from_heading:
        # A hover's flight path angle is the direction of its drift, erratic at a few tenths
        # of a knot, so the steady-flight window would reject it: take the whole record.
        index_start, index_stop = 0, int(track['time'].size)
    else:
        index_start, index_stop = steady_window(track, max_array_range=max_array_range_ft,
                                                min_duration_s=min_steady_duration_s)
    speed_knots, fpa_deg = flight_condition(track, index_start, index_stop)

    if azimuth_reference is None:
        azimuth_reference = 'heading' if nose_from_heading else 'track'
    if azimuth_reference not in fa.AZIMUTH_REFERENCES:
        raise ValueError('azimuth_reference must be one of {}, not {!r}'.format(
            fa.AZIMUTH_REFERENCES, azimuth_reference))
    if nose_from_heading and azimuth_reference != 'heading':
        raise ValueError("Run {}: nose_from_heading files a hover by its heading, so azimuth_reference "
                         "'{}' contradicts it".format(run, azimuth_reference))

    check = None
    if track_check:
        check = run_track_check(test, run, track, index_start, index_stop, hover=nose_from_heading,
                                min_height_ft=min_height_above_array_ft)
        if check.refused:
            raise TrackRefused('Run {} (card {}) refused: {}'.format(run, row.get('test_cond'),
                                                                   '; '.join(check.refusals)))
        for flag in check.flags:
            logging.warning('Run %s (card %s): %s', run, row.get('test_cond'), flag)

    segment = {key: value[index_start:index_stop] for key, value in track.items()
               if isinstance(value, np.ndarray) and value.size == track['time'].size}
    position = np.column_stack((segment['x'], segment['y'], segment['z']))
    velocity = np.column_stack((segment['vx'], segment['vy'], segment['vz_up']))
    bearing = None
    if azimuth_reference == 'heading' or wind is not None:
        try:
            bearing = frame_bearing_deg(row, track)
        except ValueError as error:
            raise ValueError('Run {}: {}'.format(run, error)) from None
    nose = None
    if azimuth_reference == 'heading':
        heading = np.asarray(segment.get('heading', np.full(segment['time'].size, np.nan)), dtype=float)
        if not np.all(np.isfinite(heading)):
            raise ValueError('Run {}: the track carries no finite heading over the window, so the sphere '
                             'cannot be filed by heading'.format(run))
        nose_x, nose_y = heading_to_frame(heading, bearing)
        nose = np.column_stack((nose_x, nose_y, np.zeros_like(nose_x)))
    if nose_from_heading:
        # No Doppler or convective term, and the azimuth along the heading.
        velocity = 1e-3 * nose
        nose = None

    density_source = 'caller'
    if atmosphere is None:
        atmosphere, error = _run_atmosphere_or_standard_day(test, run)
        density_source = 'ground stations'
        if error is not None:
            logging.warning('Run %s: %s; depropagating in a 20 C, 20 %% RH, 101.325 kPa standard day',
                            run, error)
            density_source = 'none'
    if check is not None:
        mean_height_ft = check.metrics['mean_height_above_array_ft']
    else:
        mics, _ = ground_microphone_positions(test, run)
        mean_height_ft = float(np.mean(height_above_array(segment['x'], segment['y'], segment['z'], mics)[0]))
    air_density = (float('nan') if density_source == 'none'
                   else air_density_kg_m3(atmosphere, max(mean_height_ft, 0.0)))

    wind_along = wind_cross = airspeed_knots = float('nan')
    wind_units, wind_source, wind_reference = '', 'none', ''
    if wind is not None:
        try:
            normalized = normalize_wind(wind)
        except ValueError as error:
            raise ValueError('Run {}: {}'.format(run, error)) from None
        ground_east, ground_north = frame_to_east_north(bearing, segment['vx'], segment['vy'])
        if nose_from_heading:
            heading = np.radians(np.asarray(segment['heading'], dtype=float))
            reference = (np.mean(np.sin(heading)), np.mean(np.cos(heading)))
            wind_reference = 'heading'
        else:
            reference = (np.mean(ground_east), np.mean(ground_north))
            wind_reference = 'ground track'
        if not np.hypot(*reference) > 0.0:
            raise ValueError('Run {}: no mean {} to take the wind along'.format(run, wind_reference))
        wind_along, wind_cross, airspeed_knots = wind_components(
            normalized, reference, np.column_stack((ground_east, ground_north)))
        wind_units, wind_source = normalized['units'], normalized['source']
    run_metadata = dict(gross_weight_lb=run_gross_weight_lb(row), air_density_kg_m3=air_density,
                        air_density_source=density_source, wind_along_track=wind_along,
                        wind_cross_track=wind_cross, wind_units=wind_units, wind_source=wind_source,
                        wind_reference_direction=wind_reference, airspeed_knots=airspeed_knots)

    if speed_of_sound_ft_s is None:
        speed_of_sound_ft_s = sound_speed_ft_s(atmosphere)
    if isinstance(ray_model, str):
        if ray_model != 'wind':
            raise ValueError("ray_model must be a ray model, None or 'wind', not {!r}".format(ray_model))
        if wind is None:
            raise ValueError("Run {}: ray_model='wind' needs the run's wind".format(run))
        ray_model = refracted_rays.uniform_wind_ray_model(speed_of_sound_ft_s,
                                                          wind_in_frame_ft_s(normalized, bearing))

    mics = test.ground_board_mics(run)
    if not mics:
        raise ValueError('Run {}: no ground-board microphones'.format(run))

    ambient_run = test.ambient_run(run) if gate_ambient else None
    ambient_source = 'measured'
    if gate_ambient and ambient_run is None:
        if ambient_fallback_percentile is None:
            raise ValueError('Run {}: no ambient run for layout {!r}. Set '
                             'ambient_fallback_percentile to estimate ambient from the run '
                             'itself, or gate_ambient=False only if you accept '
                             'ambient-amplified high-frequency bands'
                             .format(run, row.get('layout')))
        # Checked against B407's measured ambient runs: a 5th-percentile
        # estimate agrees to -0.3 dB in the median, with about +-8 dB of scatter
        # per channel (flight_acoustics.depropagate_hemisphere), so it is a
        # fallback, not an equivalent.  Reported in the manifest.
        ambient_source = 'percentile-{:g}'.format(ambient_fallback_percentile)
    elif not gate_ambient:
        ambient_source = 'none'
    if ambient_run is not None:
        mics = _mics_with_ambient(test, mics, ambient_run)
        if not mics:
            raise ValueError('Run {}: no microphone has both run and ambient data'.format(run))

    # Trim the recordings to the observer times that the steady segment can
    # reach.  A full run is ~50 channels x 90 s x 25 kHz; loading only what is
    # used keeps a run inside a few hundred MB instead of a couple of GB.
    # flip_y_for_geometry flips the track and the microphones together, which
    # changes no distance, so the unflipped geometry gives the same range.
    def time_range(locations):
        ranges = np.sqrt(((position[:, None, :] - locations[None, :, :]) ** 2).sum(axis=2))
        return (float(segment['time'][0]),
                float(segment['time'][-1] + ranges.max() / speed_of_sound_ft_s + 2.0 * window_time))

    if board_correction not in ('plate_bem', 'flat'):
        raise ValueError("board_correction must be 'plate_bem' or 'flat'")
    board_scale = 0.5 if board_correction == 'flat' else 1.0
    locations, pressures, times, mics = load_run_channels(test, run, mics, time_range,
                                                          ground_board_scale=board_scale)

    ambient_pressure = None
    ambient_time = None
    ambient_percentile = None
    if ambient_run is not None:
        # Keep the times: depropagate_hemisphere checks the ambient sample rate
        # against the run's from them, and assumes they match without them.
        ambient_pressure, ambient_time, _ = load_ambient_channels(test, ambient_run, mics,
                                                                  ground_board_scale=board_scale)
    elif ambient_source.startswith('percentile'):
        ambient_percentile = float(ambient_fallback_percentile)

    band_centers, phi_deg, theta_deg = _sphere_grid(legacy_sphere_grid(reference_sphere))

    receiver_response = None
    if board_correction == 'plate_bem':
        tables = [plate_table(test.instrument_types[m], speed_of_sound_ft_s, band_centers,
                              ground=ground, directory=plate_table_directory) for m in mics]
        side_y = locations[:, 1] * (-1.0 if flip_y_for_geometry else 1.0)
        receiver_response = plate_response(tables, side_y < 0.0, speed_of_sound_ft_s)

    band_low = float(band_centers.min()) / 2.0 ** (1.0 / 6.0)
    band_high = float(band_centers.max()) * 2.0 ** (1.0 / 6.0)

    hemisphere = fa.depropagate_hemisphere(
        mic_locations=locations,
        pressure=pressures,
        time=times,
        track_time=segment['time'],
        track_position=position,
        track_velocity=velocity,
        speed_of_sound=speed_of_sound_ft_s,
        length_units='ft',
        r_ref=r_ref_ft,
        freq_range=(band_low, band_high),
        window_time=window_time,
        window_overlap=window_overlap,
        point_stride=point_stride,
        azi_step=azi_step,
        elv_step=elv_step,
        rmax=rmax,
        apply_absorption_deprop=apply_absorption_deprop,
        atmosphere=atmosphere,
        flip_y_for_geometry=flip_y_for_geometry,
        third_octave=True,
        third_octave_fmin=float(band_centers.min()),
        third_octave_band_centers_hz=band_centers,
        third_octave_method=third_octave_method,
        min_elevation_deg=min_elevation_deg,
        max_range=max_propagation_range_ft,
        ambient_pressure=ambient_pressure,
        ambient_time=ambient_time,
        ambient_percentile=ambient_percentile,
        band_snr_gate_db=band_snr_gate_db,
        max_absorption_correction_db=max_absorption_correction_db,
        max_response_correction_db=max_response_correction_db,
        receiver_response_db=receiver_response,
        ray_model=ray_model,
        interpolation=interpolation,
        rim_range=(None if max_rim_range_ft is None else
                   (rim_elevation_deg, max_rim_range_ft)),
        remove_doppler=remove_doppler,
        tone_aware=tone_aware,
        decompose_tonal=tone_split,
        return_scattered=samples_path is not None,
        track_nose=nose,
        air_relative_doppler=air_relative_doppler,
        absorption_profile=absorption_profile,
        absorption_band_method=absorption_band_method,
        return_statistics=node_statistics,
    )
    if samples_path is not None:
        scattered = hemisphere['scattered']
        os.makedirs(os.path.dirname(os.path.abspath(samples_path)) or '.', exist_ok=True)
        parts = {key: scattered['third_octave'][key] for key in ('tonal_db', 'broadband_db') if tone_split}
        np.savez(samples_path, azimuth_deg=scattered['azi_deg'], elevation_deg=scattered['elv_deg'],
                 band_centers_hz=scattered['third_octave']['band_centers_hz'],
                 bands_db=scattered['third_octave']['bands_db'], mic=scattered['mic'],
                 range_ft=scattered['range'], source_height_ft=scattered['source_height'], run=str(run),
                 **parts)

    # The record, without its paths and stamp: written into each sphere as its build
    # record, so that build_all can list a run it skips as this build listed it.
    details = dict(condition=row.get('test_cond'),
                   layout=row.get('layout'), ambient_run=ambient_run,
                   ambient_source=ambient_source,
                   mics=len(mics), speed_knots=speed_knots, flight_path_angle_deg=fpa_deg,
                   window_s=(float(segment['time'][0] - track['time'][0]),
                             float(segment['time'][-1] - track['time'][0])),
                   window_points=int(index_stop - index_start),
                   min_elevation_deg=min_elevation_deg,
                   max_array_range_ft=max_array_range_ft,
                   temperature_k=atmosphere.temperature,
                   relative_humidity=atmosphere.relative_humidity,
                   pressure_kpa=atmosphere.pressure,
                   speed_of_sound_ft_s=speed_of_sound_ft_s,
                   azimuth_reference=azimuth_reference,
                   track_flags='; '.join(check.flags) if check is not None else 'not checked',
                   min_height_above_array_ft=(check.metrics['min_height_above_array_ft']
                                              if check is not None else float('nan')),
                   gross_weight_lb=run_metadata['gross_weight_lb'],
                   air_density_kg_m3=air_density, air_density_source=density_source,
                   wind_source=wind_source, wind_units=wind_units, wind_along_track=wind_along,
                   wind_cross_track=wind_cross, airspeed_knots=airspeed_knots,
                   rays='straight' if ray_model is None else getattr(ray_model, 'description', 'custom'),
                   interpolation=('fixed rmax {:g}'.format(rmax) if interpolation is None else
                                  'adaptive ' + ' '.join('{}={}'.format(k, v) for k, v in sorted(
                                      dict(fa.ADAPTIVE_INTERPOLATION, **interpolation).items()))),
                   gaps=(hemisphere.get('interpolation', {}).get('gaps', 0)),
                   rim_range='' if max_rim_range_ft is None else '{:g} ft below {:g} deg'.format(
                       max_rim_range_ft, rim_elevation_deg),
                   board_correction=board_correction if board_correction == 'flat' else
                   'plate_bem ' + ' '.join('{}={}'.format(k, v) for k, v in
                                            sorted((ground or SITE_GROUND).items())),
                   **{name: settings[name] for name in RECORDED_SETTINGS})

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    title = sphere_title(AIRCRAFT_SPHERE_PREFIX.get(test.aircraft, test.aircraft), row.get('run_num', run), run[:3])
    spheres = [(output_path, hemisphere, title, 'bands_db')]
    part_paths = {}
    if tone_split:
        part_paths = tone_split_paths(output_path)
        for part in ('tonal', 'broadband'):
            os.makedirs(os.path.dirname(part_paths[part]), exist_ok=True)
            bands = dict(hemisphere['third_octave'], bands_db=hemisphere['third_octave'][part + '_db'])
            spheres.append((part_paths[part], dict(hemisphere, third_octave=bands), '{} {}'.format(title, part),
                            part + '_db'))
    for path, sphere, sphere_name, field in spheres:
        # Each file is stamped with its own name; the settings, and their hash, are shared.
        stamp = provenance.build_provenance(
            'noise_abatement_2017.build_sphere', settings,
            inputs=dict(inputs, output_path=os.path.basename(path)), input_signature=input_signature,
            build_record=details)
        fa.write_aam_hemisphere_netcdf(
            path,
            sphere,
            mode='third_octave',
            phi_deg=phi_deg,
            theta_deg=theta_deg,
            radius_ft=r_ref_ft,
            speed_knots=speed_knots,
            flight_path_angle_deg=fpa_deg,
            title=sphere_name,
            overwrite=overwrite,
            doppler_shift_removed=1.0 if remove_doppler else 0.0,
            azimuth_reference=azimuth_reference,
            run_metadata=run_metadata,
            provenance=stamp,
            extra_variables=(sphere_statistics.node_statistics_variables(hemisphere, phi_deg, theta_deg, field)
                             if node_statistics else None),
        )
    norah2_output_path = None
    if norah2_directory is not None:
        norah2_directory = os.path.abspath(os.path.expanduser(norah2_directory))
        os.makedirs(norah2_directory, exist_ok=True)
        norah2_output_path = os.path.join(norah2_directory, norah2_file_name(
            AIRCRAFT_SPHERE_PREFIX.get(test.aircraft, test.aircraft),
            speed_knots, fpa_deg, row.get('run_num', run)))
        fa.write_norah2_hemisphere(
            norah2_output_path,
            hemisphere,
            speed_knots=speed_knots,
            flight_path_angle_deg=fpa_deg,
            test_point='Run {}'.format(row.get('run_num', run)),
            measurement_atmosphere=atmosphere,
            overwrite=overwrite,
        )

    return dict(run=run, output=output_path, norah2_output=norah2_output_path,
                tonal_output=part_paths.get('tonal'), broadband_output=part_paths.get('broadband'),
                **details, panam_version=stamp['panam_version'],
                settings_sha256=stamp['panam_settings_sha256'], input_signature=input_signature)


#: build_sphere's parameters, for working out the settings of a call without making it.
_BUILD_SPHERE_SIGNATURE = inspect.signature(build_sphere)


# --------------------------------------------------------------------------
# Batch
# --------------------------------------------------------------------------

def _run_file_paths(test, run, gate_ambient=True):
    """Every acoustic file one run needs, its ambient recording included
    unless ``gate_ambient`` is False (:func:`build_sphere` then reads none)."""
    mics = test.ground_board_mics(run)
    paths = [test.acoustic_files[run][m] for m in mics]
    ambient = test.ambient_run(run) if gate_ambient else None
    if ambient:
        paths += [path for mic, path in test.acoustic_files.get(ambient, {}).items()
                  if mic in mics]
    return paths


def _prefetch(paths, workers=PREFETCH_WORKERS):
    """Warm the cloud-storage cache for a run's files.

    When the dataset lives on cloud storage such as OneDrive, files are
    placeholders until read.
    Serial reads run at a few MB/s and dominate the batch -- a run is ~200 MB
    against ~17 s of computation -- so pull the next run's files in parallel
    while the current one is processed.
    """
    def touch(path):
        try:
            with open(path, 'rb') as handle:
                while handle.read(1 << 22):
                    pass
        except OSError as error:
            logging.debug('prefetch skipped %s (%s)', path, error)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(touch, paths))


def _prefetch_run(test, run, gate_ambient=True, done=None):
    """:func:`_prefetch` one run's files, off the main thread.

    A run whose files can't be listed (an unknown run number, say) is skipped
    here: :func:`build_sphere` meets the same error and reports the run as failed.
    Paths in the set ``done`` are skipped, and the rest added to it: an
    ambient run serves every run on its layout and day, and need be pulled
    only once.
    """
    try:
        paths = _run_file_paths(test, run, gate_ambient)
    except Exception as error:                          # noqa: BLE001
        logging.debug('prefetch skipped run %s (%s)', run, error)
        return
    if done is not None:
        paths = [path for path in paths if path not in done]
        done.update(paths)
    _prefetch(paths)


def run_input_files(test, run, gate_ambient=True, reference_sphere=None, ray_model=None):
    """Every file :func:`build_sphere` reads for ``run``, sorted: the track, the
    run's and its ambient run's recordings (:func:`_run_file_paths`), the reference
    and microphone lists, the day's weather station files, ``reference_sphere`` and
    the files ``ray_model`` names in its ``source_files`` (the atmospheric profile of
    :func:`refracted_rays.external_ray_model`).
    :func:`provenance.file_signature` of them is the sphere's input signature."""
    paths = [test.track_path(run)] + _run_file_paths(test, run, gate_ambient)
    for pattern in ('*FullRefList.csv', '*MicFullList.csv', '*StaticList.csv'):
        paths += glob.glob(os.path.join(test.base, pattern))
    paths += _weather_station_files(test, run)
    if reference_sphere is not None:
        paths.append(os.fspath(reference_sphere))
    paths += [os.fspath(path) for path in getattr(ray_model, 'source_files', ())]
    return sorted(set(paths))


def _per_run(source, run):
    """``source(run)`` for a callable, ``source.get(run)`` for a mapping, None for None."""
    if source is None:
        return None
    return source(run) if callable(source) else source.get(run)


def _build_run(test, run, output_path, reference, norah2_directory, ray_models, winds, kwargs):
    """:func:`build_sphere` for one run of :func:`build_all`: ``(record, None)``,
    or ``(None, error text)`` if it fails."""
    try:
        record = build_sphere(test, run, output_path, reference_sphere=reference,
                              norah2_directory=norah2_directory, ray_model=_per_run(ray_models, run),
                              wind=_per_run(winds, run), **kwargs)
    except Exception as error:                          # noqa: BLE001
        return None, str(error)
    return record, None


def _norah2_output(test, run, norah2_directory, details):
    """The NORAH2 .hem file the build that wrote ``details`` (its record: speed and
    flight path angle) wrote for ``run`` in ``norah2_directory``, or None if it is not there.

    The name is worked out with :func:`norah2_file_name`, not searched for: a run
    rebuilt at another speed label leaves its older .hem beside the new one.
    """
    prefix = AIRCRAFT_SPHERE_PREFIX.get(test.aircraft, test.aircraft)
    number = test.by_run.get(run, {}).get('run_num', run)
    path = os.path.join(os.path.abspath(os.path.expanduser(norah2_directory)), norah2_file_name(
        prefix, details['speed_knots'], details['flight_path_angle_deg'], number))
    return path if os.path.exists(path) else None


def _sphere_record(test, run, output_path):
    """``(stamp, details)`` of the sphere at ``output_path``: its global attributes, and
    the record its build wrote (``panam_build_record``).

    A sphere without a build record gives only the run's identity, its speed and
    flight path angle from the sphere's float32 ``SPEED`` and ``FLIGHT_PATH_ANGLE``,
    and ``track_flags`` :data:`UNKNOWN_TRACK_FLAGS`, so that a run whose flags are
    not known is never taken as unflagged.
    """
    row = test.by_run.get(run, {})
    with Dataset(output_path) as sphere:
        stamp = {name: sphere.getncattr(name) for name in sphere.ncattrs()}
        if provenance.BUILD_RECORD_ATTRIBUTE in stamp:
            details = provenance.read_record_json(stamp[provenance.BUILD_RECORD_ATTRIBUTE])
        else:
            details = dict(condition=row.get('test_cond'), layout=row.get('layout'),
                           speed_knots=fa._read_scalar(sphere, 'SPEED'),
                           flight_path_angle_deg=fa._read_scalar(sphere, 'FLIGHT_PATH_ANGLE'),
                           track_flags=UNKNOWN_TRACK_FLAGS)
    return stamp, details


def _skip_reason(test, run, output_path, call, skip_existing, skip_unchanged, ray_models):
    """Why :func:`build_all` need not build ``run`` ('exists', 'unchanged' or None),
    and the sphere's :func:`_sphere_record` (None when it was not read).

    A run counts as built only when every file its build writes exists: the
    sphere, its NORAH2 file when ``call`` has a ``norah2_directory`` and its
    parts when it asks for ``tone_split``.  'unchanged' also needs the sphere's
    ``panam_settings_sha256`` to equal this call's and its
    ``panam_input_signature`` to equal :func:`run_input_files`' signature now.
    """
    if not (skip_existing or skip_unchanged):
        return None, None
    outputs = [output_path]
    if call.get('tone_split'):
        outputs += list(tone_split_paths(output_path).values())
    if not all(os.path.exists(path) for path in outputs):
        return None, None
    try:
        stamp, details = _sphere_record(test, run, output_path)
    except (OSError, KeyError, IndexError, ValueError):
        return None, None
    if call.get('norah2_directory') is not None and _norah2_output(
            test, run, call['norah2_directory'], details) is None:
        return None, (stamp, details)
    if skip_existing:
        return 'exists', (stamp, details)
    arguments = dict(call, ray_model=_per_run(ray_models, run))
    bound = _BUILD_SPHERE_SIGNATURE.bind(test, run, output_path, **arguments)
    bound.apply_defaults()
    settings, _ = _sphere_settings(bound.arguments)
    signature = provenance.file_signature(run_input_files(
        test, run, arguments.get('gate_ambient', True), arguments.get('reference_sphere'),
        arguments['ray_model']))
    if (stamp.get('panam_settings_sha256') == provenance.settings_sha256(settings)
            and stamp.get(provenance.INPUT_SIGNATURE_ATTRIBUTE) == signature):
        return 'unchanged', (stamp, details)
    return None, (stamp, details)


#: track_flags of a skipped run whose sphere has no build record (written before
#: build records were): not blank, so select_from_manifest(skip_flagged=True) leaves it out.
UNKNOWN_TRACK_FLAGS = 'unknown: the sphere has no build record'


def _skipped_record(test, run, output_path, norah2_directory, reason, sphere_record=None):
    """The record of a run :func:`build_all` skipped, read from its existing sphere.

    The sphere's build record (``panam_build_record``, the record
    :func:`build_sphere` returned, without its paths and stamp) gives every
    manifest column as the original build wrote it, the float64 speed and flight
    path angle among them, so the manifest row and the NORAH2 triangulation are
    those of a full build.  The paths are this build's, the stamp columns come
    from the sphere's provenance attributes, and ``skipped`` is ``reason``.

    A sphere without a build record gives only the run's identity, its speed and
    flight path angle from the sphere's float32 ``SPEED`` and ``FLIGHT_PATH_ANGLE``,
    and ``track_flags`` :data:`UNKNOWN_TRACK_FLAGS`, so that a run whose flags are
    not known is never taken as unflagged.  ``sphere_record``, the sphere's
    :func:`_sphere_record` when already read, saves reading it again.
    """
    stamp, details = sphere_record or _sphere_record(test, run, output_path)
    if provenance.BUILD_RECORD_ATTRIBUTE not in stamp:
        logging.warning('run %s: %s has no build record; its manifest row has no track flags',
                        run, os.path.basename(output_path))
    parts = tone_split_paths(output_path)
    return dict(run=run, output=output_path,
                norah2_output=None if norah2_directory is None else _norah2_output(test, run, norah2_directory,
                                                                                    details),
                tonal_output=parts['tonal'] if os.path.exists(parts['tonal']) else None,
                broadband_output=parts['broadband'] if os.path.exists(parts['broadband']) else None,
                **details,
                panam_version=stamp.get('panam_version', ''),
                settings_sha256=stamp.get('panam_settings_sha256', ''),
                input_signature=stamp.get(provenance.INPUT_SIGNATURE_ATTRIBUTE, ''),
                skipped=reason)


def _plate_table_jobs(test, plans, kwargs):
    """The distinct :func:`plate_table` arguments the runs of ``plans`` need.

    Picks the sound speed, the bands and the microphones as :func:`build_sphere`
    does.  A run whose tables cannot be worked out here is left out: its build
    meets the same problem and reports it.  Returns a list of
    ``(instrument, sound_speed_ft_s, bands, ground, directory)``.
    """
    if kwargs.get('board_correction', 'plate_bem') != 'plate_bem':
        return []
    bands_of = {}
    jobs = {}
    for plan in plans:
        run = plan['run']
        try:
            speed = kwargs.get('speed_of_sound_ft_s')
            if speed is None:
                atmosphere = kwargs.get('atmosphere')
                if atmosphere is None:
                    atmosphere, _ = _run_atmosphere_or_standard_day(test, run)
                speed = sound_speed_ft_s(atmosphere)
            if plan['reference'] not in bands_of:
                bands_of[plan['reference']] = _sphere_grid(legacy_sphere_grid(plan['reference']))[0]
            bands = bands_of[plan['reference']]
            mics = test.ground_board_mics(run)
            ambient = test.ambient_run(run) if kwargs.get('gate_ambient', True) else None
            if ambient is not None:
                mics = _mics_with_ambient(test, mics, ambient)
            for instrument in sorted({test.instrument_types[m] for m in mics}):
                key = (instrument, _plate_table_speed(speed), tuple(np.asarray(bands, dtype=float)))
                jobs.setdefault(key, (instrument, speed, bands, kwargs.get('ground'),
                                      kwargs.get('plate_table_directory')))
        except Exception as error:                      # noqa: BLE001
            logging.debug('no plate tables warmed for run %s (%s)', run, error)
    return list(jobs.values())


def _warm_plate_tables(test, plans, kwargs):
    """Compute, in this process, the plate tables the runs of ``plans`` need
    (:func:`_plate_table_jobs`), so that parallel workers read them from the cache
    rather than each computing them.  A table that fails here is left to its builds."""
    for instrument, speed, bands, ground, directory in _plate_table_jobs(test, plans, kwargs):
        try:
            plate_table(instrument, speed, bands, ground=ground, directory=directory)
        except Exception as error:                      # noqa: BLE001
            logging.debug('plate table for %s not warmed (%s)', instrument, error)


#: State of a build_all worker process, set once by _start_worker.
_WORKER = {}


@functools.lru_cache(maxsize=None)
def _open_test(aircraft, root):
    """The :class:`NoiseAbatementTest` of ``aircraft`` at ``root``, opened once per
    process: a :func:`build_all` worker and the ray models it unpickles share it."""
    return NoiseAbatementTest(aircraft, root=root)


def _start_worker(aircraft, root, norah2_directory, ray_models, winds, kwargs, log_level, log_files):
    """Initialize a :func:`build_all` worker process: its logging and its own
    :class:`NoiseAbatementTest`."""
    configure_logging(log_level=log_level)
    for path in log_files:
        _add_log_file(path)
    _WORKER.update(test=_open_test(aircraft, root), norah2_directory=norah2_directory,
                   ray_models=ray_models, winds=winds, kwargs=kwargs)


def _worker_build(run, output_path, reference):
    """:func:`_build_run` in a worker process."""
    return _build_run(_WORKER['test'], run, output_path, reference, _WORKER['norah2_directory'],
                      _WORKER['ray_models'], _WORKER['winds'], _WORKER['kwargs'])


def _check_picklable(jobs, **values):
    """Refuse, before anything is built, a value a worker process cannot receive."""
    for name, value in values.items():
        try:
            pickle.dumps(value)
        except Exception as error:                      # noqa: BLE001
            raise TypeError('build_all(jobs={}) sends {} to worker processes, so it must be None, a dict '
                            'or a module-level function, and every build_sphere argument must be '
                            'picklable: {}'.format(jobs, name, error)) from None


def build_all(aircraft, output_directory, *, root=None, runs=None,
              steady_only=True, reference_directory=None, sphere_prefix=None,
              manifest_path=None, prefetch=True, norah2_directory=None, ray_models=None, winds=None,
              ray_profiles=None, skip_existing=False, skip_unchanged=False, jobs=1, **kwargs):
    """Rebuild every usable run for one aircraft.

    ``winds``, if given, is called with each run id and returns that run's ``wind`` for
    :func:`build_sphere` (see :func:`normalize_wind`), or None where the wind at the
    aircraft is unknown; a mapping of run id to wind works too (:func:`read_winds` reads
    one from a CSV file).

    A run whose track :func:`check_track` refuses fails like any other, with the reason
    in its failure record; pass ``track_check=False`` to build it anyway.

    ``ray_models``, if given, is called with each run id and returns that run's
    ``ray_model`` for :func:`build_sphere` (its own atmosphere, or ``'wind'`` for its
    wind from ``winds``), or None to build that run along straight lines; a mapping
    of run id to ray model works too.  The manifest's ``rays`` column says which.

    ``ray_profiles``, if given, is a directory of per-run NICE-OPS ``--atmosphere`` CSV
    files (heights in ft) named by run id, ``<run>.csv``: :func:`profile_ray_models` turns
    it into ``ray_models``.  A run with a profile is depropagated along refracted rays
    through it, and a run without one along straight lines.  Give ``ray_models`` or
    ``ray_profiles``, not both.

    ``norah2_directory``, if given, also receives each sphere as a NORAH2
    ``.hem`` file and, once the batch is done, the ``[prefix]_Triangulation.int``
    NORAH2 needs to interpolate between them.

    A run that fails is logged and skipped rather than aborting the batch --
    across 1400 runs there are always a few with a truncated tracking file or a
    missing ambient layout, and they should not cost the other 1399.

    ``steady_only`` (default True, ignored when ``runs`` is given explicitly)
    selects only non-maneuvering passes via :func:`is_steady_flight_card` --
    see there for what that means and how it was checked.  A sphere built
    through a turn or an accelerating approach smears directivity across
    azimuth in a way depropagation does not correct for, so this is what the
    legacy Be407 database restricts to as well, though nothing there enforced
    it explicitly.

    ``skip_existing`` (default False) does not rebuild a run whose outputs all
    exist: its sphere, and its NORAH2 file and tonal and broadband parts when
    asked for.  ``skip_unchanged`` (default False) does not rebuild one whose
    outputs exist and whose sphere's ``panam_settings_sha256`` and
    ``panam_input_signature`` (:func:`run_input_files`) match this build's; the
    code version is not compared, so rebuild without it after updating PANAM.  A
    skipped run keeps its files; its record, rebuilt from the sphere's build
    record (:func:`_skipped_record`), is the one its original build returned
    (track flags, rays, window and settings included) plus ``skipped`` ('exists'
    or 'unchanged'), and goes into the manifest and the NORAH2 triangulation like
    a built one.

    ``jobs`` (default 1) builds that many runs at once, in worker processes
    started with the spawn method.  Each worker opens the dataset itself.  The
    plate tables are computed first, in this process, so the workers read them
    from the cache.  ``ray_models``, ``winds`` and every :func:`build_sphere`
    argument go to the workers, so they must be picklable: None, a dict or a
    module-level function, not a lambda or a nested function (TypeError
    otherwise).  The records come back in run order, so the manifest and the
    triangulation are the same as with ``jobs=1``.  This process prefetches each
    run before it hands it to a worker.

    Returns ``(records, failures)``, each in run order.
    """
    jobs = int(jobs)
    if jobs < 1:
        raise ValueError('jobs must be at least 1, not {}'.format(jobs))
    if jobs > 1:
        _check_picklable(jobs, ray_models=ray_models, winds=winds, kwargs=kwargs)
    test = NoiseAbatementTest(aircraft, root=root)
    if ray_profiles is not None:
        if ray_models is not None:
            raise ValueError('give ray_models or ray_profiles, not both')
        ray_models = profile_ray_models(test, ray_profiles)
    if runs is None:
        runs = test.runs(steady_only=steady_only)
    prefix = sphere_prefix or AIRCRAFT_SPHERE_PREFIX.get(aircraft, aircraft)
    output_directory = os.path.abspath(os.path.expanduser(output_directory))
    os.makedirs(output_directory, exist_ok=True)

    # Each run: (record, None) once built or skipped, (None, error) once failed.
    outcomes = {}
    plans = []
    for number, run in enumerate(runs, start=1):
        row = test.by_run.get(run, {})
        name = '{}{}.nc'.format(prefix, row.get('run_num', run))
        reference = None
        if reference_directory:
            candidate = os.path.join(os.path.expanduser(reference_directory), name)
            reference = candidate if os.path.exists(candidate) else None
        plan = dict(number=number, run=run, name=name, output=os.path.join(output_directory, name),
                    reference=reference)
        if skip_existing or skip_unchanged:
            try:
                call = dict(kwargs, reference_sphere=reference, norah2_directory=norah2_directory,
                            wind=_per_run(winds, run))
                reason, sphere_record = _skip_reason(test, run, plan['output'], call, skip_existing,
                                                     skip_unchanged, ray_models)
                if reason is not None:
                    outcomes[number] = (_skipped_record(test, run, plan['output'], norah2_directory,
                                                        reason, sphere_record), None)
                    logging.info('[%d/%d] %s skipped: %s %s', number, len(runs), run, name, reason)
                    continue
            except Exception as error:                  # noqa: BLE001
                logging.debug('run %s: not skipped (%s)', run, error)
        plans.append(plan)

    def report(plan, outcome):
        outcomes[plan['number']] = outcome
        record, error = outcome
        if record is None:
            logging.warning('[%d/%d] %s failed: %s', plan['number'], len(runs), plan['run'], error)
        else:
            logging.info('[%d/%d] %s -> %s  %.1f kt  %+.1f deg  %d mics  ambient %s',
                         plan['number'], len(runs), plan['run'], plan['name'], record['speed_knots'],
                         record['flight_path_angle_deg'], record['mics'], record['ambient_run'])

    gate_ambient = kwargs.get('gate_ambient', True)
    prefetched = set()
    if jobs == 1:
        pool = ThreadPoolExecutor(max_workers=1) if prefetch and plans else None

        def prefetch_next(index):
            return (pool.submit(_prefetch_run, test, plans[index]['run'], gate_ambient, prefetched)
                    if index < len(plans) else None)

        try:
            pending = prefetch_next(0) if pool else None
            for index, plan in enumerate(plans, start=1):
                if pending is not None:
                    pending.result()
                    pending = prefetch_next(index)
                report(plan, _build_run(test, plan['run'], plan['output'], plan['reference'],
                                        norah2_directory, ray_models, winds, kwargs))
        finally:
            if pool is not None:
                pool.shutdown(cancel_futures=True)
    elif plans:
        _warm_plate_tables(test, plans, kwargs)
        log_files = [handler.baseFilename for handler in logging.getLogger().handlers
                     if isinstance(handler, logging.FileHandler)]
        # test.root, resolved, so that a worker and the ray models it unpickles share
        # one NoiseAbatementTest (_open_test).
        executor = ProcessPoolExecutor(
            max_workers=min(jobs, len(plans)), mp_context=multiprocessing.get_context('spawn'),
            initializer=_start_worker,
            initargs=(aircraft, test.root, norah2_directory, ray_models, winds, kwargs,
                      logging.getLogger().getEffectiveLevel(), log_files))
        pool = ThreadPoolExecutor(max_workers=1) if prefetch else None
        try:
            futures = {executor.submit(_worker_build, plan['run'], plan['output'], plan['reference']): plan
                       for plan in plans}
            # Pulled in plan order, ahead of the workers, without holding them back.
            if pool is not None:
                for plan in plans:
                    pool.submit(_prefetch_run, test, plan['run'], gate_ambient, prefetched)
            for future in as_completed(futures):
                report(futures[future], future.result())
        finally:
            if pool is not None:
                pool.shutdown(cancel_futures=True)
            executor.shutdown(cancel_futures=True)

    records = [outcomes[n][0] for n in sorted(outcomes) if outcomes[n][0] is not None]
    failures = []
    for number in sorted(outcomes):
        if outcomes[number][0] is None:
            run = runs[number - 1]
            row = test.by_run.get(run, {})
            failures.append(dict(run=run, error=outcomes[number][1], condition=row.get('test_cond'),
                                 layout=row.get('layout')))
    if norah2_directory is not None and len(records) >= 3:
        triangulation = os.path.join(os.path.abspath(os.path.expanduser(norah2_directory)),
                                     '{}_Triangulation.int'.format(prefix))
        try:
            fa.write_norah2_triangulation(
                triangulation,
                [(os.path.basename(r['norah2_output']), r['speed_knots'], r['flight_path_angle_deg'])
                 for r in records])
            logging.info('Wrote NORAH2 triangulation %s', triangulation)
        except ValueError as error:
            logging.warning('No NORAH2 triangulation written: %s', error)
    if manifest_path:
        write_manifest(manifest_path, records, failures)
    return records, failures


def profile_ray_models(test, directory, executable=None):
    """``build_all``'s ``ray_models`` from a directory of per-run atmospheric profiles.

    ``directory`` holds NICE-OPS ``--atmosphere`` CSV files (heights in ft, above the
    ground boards) named by run id, ``<run>.csv``.  The returned callable gives, for a
    run with a profile, :func:`refracted_rays.external_ray_model` through it: stratified
    refraction, flush plate receivers, and the run's own :func:`frame_bearing_deg` to place
    the profile's winds.  For a run without one it gives None, straight lines.
    ``executable`` is NICE-OPS's ``niceops_ray_geometry`` (default: the
    ``niceops_ray_geometry`` entry of :mod:`local_paths`), looked up only when a run has a
    profile.  The callable pickles, so that ``build_all(jobs=N)`` can send it to its
    worker processes: it carries the dataset's aircraft and root, and a worker opens
    the dataset again (or uses its own) on its first call.
    """
    directory = os.path.abspath(os.path.expanduser(directory))
    if not os.path.isdir(directory):
        raise FileNotFoundError('ray profile directory {} does not exist'.format(directory))
    return _ProfileRayModels(test, directory, executable)


class _ProfileRayModels:
    """The callable :func:`profile_ray_models` returns; module-level, so it pickles."""

    def __init__(self, test, directory, executable):
        self.test = test
        self.directory = directory
        self.executable = executable

    def __getstate__(self):
        state = dict(self.__dict__)
        test = state.pop('test')
        state['dataset'] = (test.aircraft, test.root)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.test = None

    def _dataset(self):
        if self.test is None:
            self.test = _open_test(*self.dataset)
        return self.test

    def __call__(self, run):
        path = os.path.join(self.directory, '{}.csv'.format(run))
        if not os.path.isfile(path):
            return None
        tracer = local_paths.data_path('niceops_ray_geometry', self.executable)
        test = self._dataset()
        bearing = frame_bearing_deg(test.by_run[run], load_track(test.track_path(run)))
        return refracted_rays.external_ray_model(tracer, path, frame_bearing_deg=bearing,
                                                 refraction='stratified', receiver='plate',
                                                 length_units='ft')


def write_manifest(path, records, failures):
    """Record what went into the spheres, so a database can be audited later.

    A ``skipped`` column, added when a record has one (``build_all(skip_existing=True)``
    or ``skip_unchanged=True``), says which spheres were kept from an earlier build."""
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fields = ['run', 'output', 'norah2_output', 'condition', 'layout', 'ambient_run',
              'ambient_source', 'mics',
              'speed_knots', 'flight_path_angle_deg', 'window_s', 'window_points',
              'min_elevation_deg', 'max_array_range_ft', 'temperature_k',
              'relative_humidity', 'pressure_kpa', 'speed_of_sound_ft_s', 'board_correction',
              'rays', 'interpolation', 'gaps', 'rim_range', 'azimuth_reference', 'track_flags',
              'min_height_above_array_ft', 'gross_weight_lb', 'air_density_kg_m3', 'air_density_source',
              'wind_source', 'wind_units', 'wind_along_track', 'wind_cross_track', 'airspeed_knots',
              *RECORDED_SETTINGS, 'panam_version', 'settings_sha256', 'error']
    if any(record.get('skipped') for record in records):
        fields.insert(fields.index('error'), 'skipped')
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for record in records:
            writer.writerow(record)
        for failure in failures:
            writer.writerow(failure)


def select_from_manifest(manifest_path, *, skip_flagged=False, conditions=None):
    """The sphere file names a build manifest (:func:`write_manifest`) lists as built.

    Returns the file names (no directory), in manifest order, of every run that
    produced a sphere; failed runs are left out.  The result is meant for
    ``flight_acoustics.build_empirical_database(spheres=)``, which takes each
    name relative to the sphere directory, so the database is built from what
    this build wrote and from nothing else left in the directory.

    skip_flagged leaves out runs whose track check raised a flag (a non-empty
    ``track_flags`` other than ``not checked``).  conditions, when given, keeps
    only runs whose ``condition`` (the test card) is in it.  By default every
    built sphere is returned.
    """
    path = os.path.abspath(os.path.expanduser(manifest_path))
    names = []
    with open(path, encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if 'output' not in (reader.fieldnames or []):
            raise ValueError('{}: a build manifest needs an output column'.format(manifest_path))
        wanted = None if conditions is None else {str(c) for c in conditions}
        for entry in reader:
            entry = {k: (v or '').strip() for k, v in entry.items() if k is not None}
            if not entry['output'] or entry.get('error'):
                continue
            if skip_flagged and entry.get('track_flags', '') not in ('', 'not checked'):
                continue
            if wanted is not None and entry.get('condition', '') not in wanted:
                continue
            # Basename on either separator: the manifest may come from another machine.
            name = entry['output'].replace('\\', '/').rsplit('/', 1)[-1]
            if name not in names:
                names.append(name)
    return names


def read_winds(path):
    """Per-run winds at the aircraft from a CSV file, as ``{run: wind}`` for :func:`build_all`.

    Columns: ``run``, ``units`` and ``source``, and either ``east`` and ``north`` or
    ``speed`` and ``direction_from_deg`` (see :func:`normalize_wind`).  A row whose wind
    cells are blank is a run with no known wind and is left out.  Every row is checked as
    it is read, so a bad unit or source fails before any sphere is built.
    """
    winds = {}
    with open(os.path.abspath(os.path.expanduser(path)), encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if not {'run', 'units', 'source'} <= fields:
            raise ValueError('{}: a winds file needs run, units and source columns'.format(path))
        for line, entry in enumerate(reader, start=2):
            entry = {k: (v or '').strip() for k, v in entry.items() if k is not None}
            run = entry['run']
            values = {k: entry[k] for k in ('east', 'north', 'speed', 'direction_from_deg') if entry.get(k)}
            if not run or not values:
                continue
            if run in winds:
                raise ValueError('{}: run {} appears twice'.format(path, run))
            try:
                winds[run] = normalize_wind(dict({k: float(v) for k, v in values.items()},
                                                 units=entry['units'], source=entry['source']))
            except ValueError as error:
                raise ValueError('{} line {}: {}'.format(path, line, error)) from None
    return winds


def check_tracks(aircraft, *, root=None, runs=None, steady_only=True, hovers=False,
                 max_array_range_ft=None, min_steady_duration_s=8.0, **thresholds):
    """Run :func:`check_track` over one aircraft's tracks without building anything.

    A dry pass: every run with a tracking file whose card is steady (:func:`is_steady_flight_card`,
    unless ``steady_only`` is False) or, with ``hovers``, a hover card, takes the window
    :func:`build_sphere` would (the steady window, or a hover's whole record) and is checked
    as the build would check it.  Acoustic data are not needed.  ``thresholds`` go to
    :func:`check_track`.

    Returns one dict per run: run, condition, status ('refused', 'flagged', 'ok', or
    'no window' when :func:`steady_window` finds none), reasons (refusals, or the window's
    error), flags, and the check's metrics.
    """
    test = NoiseAbatementTest(aircraft, root=root)
    out = []
    for row in test.reference:
        run = row['combined']
        condition = (row.get('test_cond') or '').strip()
        hover = condition.startswith('H')
        if runs is not None:
            if run not in runs:
                continue
        elif condition == 'AMB' or (hover and not hovers) or (
                not hover and steady_only and not is_steady_flight_card(row)):
            continue
        path = test.track_path(run)
        if not os.path.exists(path):
            continue
        record = dict(run=run, condition=condition, status='ok', reasons='', flags='')
        try:
            track = load_track(path)
            if hover:
                index_start, index_stop = 0, int(track['time'].size)
            else:
                index_start, index_stop = steady_window(track, max_array_range=max_array_range_ft,
                                                        min_duration_s=min_steady_duration_s)
        except ValueError as error:
            record.update(status='no window', reasons=str(error))
            out.append(record)
            continue
        check = run_track_check(test, run, track, index_start, index_stop, hover=hover, **thresholds)
        record.update(check.metrics)
        record.update(reasons='; '.join(check.refusals), flags='; '.join(check.flags),
                      window_s='{:.1f}-{:.1f}'.format(track['time'][index_start] - track['time'][0],
                                                      track['time'][index_stop - 1] - track['time'][0]))
        if check.refused:
            record['status'] = 'refused'
        elif check.flags:
            record['status'] = 'flagged'
        out.append(record)
    return out


def write_track_checks(path, records):
    """Write :func:`check_tracks` records as CSV."""
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fields = ['aircraft', 'run', 'condition', 'status', 'window_s', 'min_height_above_array_ft',
              'mean_height_above_array_ft', 'max_altitude_mismatch_ft', 'min_along_track_acceleration_ft_s2',
              'plane', 'reasons', 'flags']
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def configure_logging(verbosity=0, log_file=None, log_level=None):
    """Set up logging for the command line and return the file handler it adds, or None.

    The level is INFO, one step quieter per unit of negative ``verbosity``
    (WARNING, ERROR, CRITICAL) and DEBUG for a positive one; ``log_level``, if
    given, sets it instead.  Messages go to stderr (unless the root logger
    already has a handler) and, with ``log_file``, are appended to that file too.
    """
    if log_level is None:
        log_level = min(max(logging.INFO - 10 * int(verbosity), logging.DEBUG), logging.CRITICAL)
    logging.basicConfig(format=LOG_FORMAT)
    root = logging.getLogger()
    root.setLevel(log_level)
    if not log_file:
        return None
    path = os.path.abspath(os.path.expanduser(log_file))
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    return _add_log_file(path)


def _add_log_file(path):
    """Append the root logger's messages to the file ``path`` in :data:`LOG_FORMAT`;
    returns the handler."""
    handler = logging.FileHandler(path, encoding='utf-8')
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(handler)
    return handler


def _wind_rays(run):
    """``ray_models`` for ``--ray-model wind``: every run along its wind's rays.
    Module-level, so that ``--jobs`` can send it to worker processes."""
    return 'wind'


def _interpolation_setting(value):
    """argparse type for one adaptive gridding setting, KEY=VALUE (see fa.ADAPTIVE_INTERPOLATION)."""
    import argparse
    key, separator, text = value.partition('=')
    key = key.strip()
    if not separator or key not in fa.ADAPTIVE_INTERPOLATION:
        raise argparse.ArgumentTypeError('expected KEY=VALUE with KEY one of {}, got {!r}'.format(
            ', '.join(sorted(fa.ADAPTIVE_INTERPOLATION)), value))
    default = fa.ADAPTIVE_INTERPOLATION[key]
    text = text.strip()
    if isinstance(default, bool):
        if text.lower() not in ('true', 'false'):
            raise argparse.ArgumentTypeError('{} takes true or false, not {!r}'.format(key, text))
        return key, text.lower() == 'true'
    try:
        return key, type(default)(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError('{}: {}'.format(key, error))


#: Command-line destinations whose build_all/build_sphere keyword has another name, or that
#: set more than one keyword.  Every other destination is its keyword.  A recipe value of
#: one of these keywords gives way when the destination is given on the command line.
CLI_KEYWORDS = dict(
    include_maneuvers=('steady_only',), manifest=('manifest_path',), no_prefetch=('prefetch',),
    no_ambient_gate=('gate_ambient',), no_track_check=('track_check',),
    ray_model=('ray_models', 'ray_profiles'), ray_profiles=('ray_profiles', 'ray_models'),
    tone_split=('tone_split', 'tone_aware'), adaptive_interpolation=('interpolation',),
    interpolation=('interpolation',), no_absorption_deprop=('apply_absorption_deprop',),
    no_overwrite=('overwrite',))


def make_parser():
    """The command line's parser: every build_all and build_sphere option it exposes."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Rebuild 2017 Noise Abatement source spheres with ambient gating.',
        epilog='Python-only build_sphere keywords (atmosphere, ground, absorption_profile, '
               'samples_path, a ray model other than the profiles or the wind) have no option. '
               '--recipe reads settings from a TOML recipe (see build_recipe.py); an option given '
               'on the command line overrides the recipe.')
    parser.add_argument('aircraft', nargs='?', default=None,
                        help='dataset directory name under --root: AS350B3, B206L3, B407, '
                             'EC130B4, R44 or R66 (not needed with --plate-cache)')
    parser.add_argument('output_directory', nargs='?', default=None,
                        help='where the spheres go (not needed with --check-tracks or --plate-cache)')
    parser.add_argument('--recipe', default=None, metavar='FILE.toml',
                        help="build with the [spheres] settings of this recipe; options given here "
                             'override it (default: none)')
    parser.add_argument('--hover-source', action='store_true',
                        help="with --recipe: build with the recipe's [hover_source] settings over its "
                             '[spheres] ones, the rebuild the database synthesizes its hover from; '
                             'needs --runs (default off)')
    parser.add_argument('--plate-table-directory', default=None,
                        help='where computed ground-plate tables are cached (default: '
                             '~/.cache/panam/plate_tables)')
    parser.add_argument('--plate-cache', choices=('list', 'clear'), default=None,
                        help='do not build: list the cached plate tables (size in MB and whether '
                             'each is current), or delete them; default: neither')
    parser.add_argument('--check-tracks', action='store_true',
                        help='do not build: check every steady run\'s track as the build would '
                             '(heights above the ground boards, altitude consistency, decelerations, '
                             'level-offs, card flight path angle), print what is refused and flagged, '
                             'and write the table to --manifest if given; needs only the tracks')
    parser.add_argument('--hovers', action='store_true',
                        help='with --check-tracks, check the hover runs too (their whole record)')
    parser.add_argument('--azimuth-reference', choices=('track', 'heading'), default=None,
                        help="'track' (the default for flight) measures each sphere's azimuth from the ground track; "
                             "'heading' from the tracked INS heading, with Doppler and depropagation "
                             'still on the ground velocity.  Written to every sphere and so to the database')
    parser.add_argument('--nose-from-heading', action='store_true',
                        help='orient each sphere by the tracked heading, for hovers: no Doppler or '
                             'convective term, and the whole record instead of a steady window '
                             '(default off)')
    parser.add_argument('--winds', default=None,
                        help='CSV of the wind at the aircraft per run (run, units, source, and east/north '
                             'or speed/direction_from_deg; see read_winds), recorded in each sphere; '
                             'nothing is assumed about its units')
    parser.add_argument('--ray-model', choices=('straight', 'wind'), default='straight',
                        help="'straight' (default): straight lines in still air; 'wind': straight rays "
                             "carried by each run's wind from --winds (required), uniform over the path; "
                             'a run with no wind there fails')
    parser.add_argument('--ray-profiles', default=None, metavar='DIRECTORY',
                        help="refracted rays through each run's NICE-OPS atmosphere CSV, <run>.csv in "
                             'this directory (heights in ft), traced by niceops_ray_geometry from '
                             'local_paths; straight lines for a run without one (default: none)')
    parser.add_argument('--air-relative-doppler', action='store_true',
                        help='with --remove-doppler and --ray-model wind, take the Doppler factor in '
                             'the moving air (default off)')
    parser.add_argument('--no-track-check', action='store_true',
                        help='build runs whose track fails the sanity check (below or within '
                             '--min-height-above-array-ft of the ground boards, z not following altitude)')
    parser.add_argument('--min-height-above-array-ft', type=float, default=MIN_HEIGHT_ABOVE_ARRAY_FT,
                        help='refuse a run whose track comes closer than this to the ground boards '
                             '(default %(default)s)')
    parser.add_argument('--root', default=None,
                        help='dataset root (default: the noise_abatement_2017 entry of local_paths)')
    parser.add_argument('--runs', nargs='*', default=None,
                        help='specific run ids (default: every steady, non-maneuvering run)')
    parser.add_argument('--include-maneuvers', action='store_true',
                        help='also build spheres from turns and accelerating passes; off '
                             'by default, since they smear directivity across azimuth in a '
                             'way depropagation does not correct for')
    parser.add_argument('--reference-directory', default=None,
                        help='legacy spheres, used only for their PHI/THETA/FREQUENCY grids')
    parser.add_argument('--sphere-prefix', default=None,
                        help="sphere file name prefix (default: the aircraft's legacy prefix, e.g. Be407)")
    parser.add_argument('--manifest', default=None,
                        help="CSV recording each run's inputs and outcome (with --check-tracks: "
                             'the track-check table); default: not written')
    parser.add_argument('--norah2-directory', default=None,
                        help='also write each sphere as a NORAH2 .hem file here, with the '
                             'triangulation file NORAH2 needs to interpolate between them')
    parser.add_argument('--no-overwrite', action='store_true',
                        help='fail on a run whose sphere exists instead of replacing it (default: replace)')
    parser.add_argument('--board-correction', choices=('plate_bem', 'flat'), default='plate_bem',
                        help="'plate_bem' (default) divides out the ground plate's modeled "
                             "response per band and frame; 'flat' is the old constant -6 dB")
    parser.add_argument('--third-octave-method', choices=('fft', 'filter_bank'), default='fft',
                        help="'filter_bank' forms bands with a true one-third octave filter "
                             "bank, as an analyzer does; it differs from the default FFT band "
                             "sum only below ~100 Hz, between strong rotor tones")
    parser.add_argument('--band-snr-gate-db', type=float, default=10.0,
                        help="zero frequency bins less than this many dB above the mic's ambient "
                             '(default %(default)g)')
    parser.add_argument('--max-absorption-correction-db', type=float, default=30.0,
                        help='discard bins needing more absorption correction than this, dB; '
                             '30 dB is what keeps source spectra rolling off physically '
                             '(default %(default)g)')
    parser.add_argument('--max-response-correction-db', type=float, default=None,
                        help='discard bins whose receiver (ground-plate) correction exceeds this, dB '
                             '(default: no cap)')
    parser.add_argument('--no-absorption-deprop', action='store_true',
                        help='do not undo atmospheric absorption (default: undo it)')
    parser.add_argument('--absorption-band-method', choices=fa.BAND_ABSORPTION_METHODS,
                        default='center',
                        help="band absorption for --third-octave-method filter_bank: 'center' (ISO "
                             "9613-1 at the band center), 'arp5534' or 'integrated' (default %(default)s)")
    parser.add_argument('--speed-of-sound-ft-s', type=float, default=None,
                        help="speed of sound, ft/s (default: from the run's atmosphere)")
    parser.add_argument('--r-ref-ft', type=float, default=DEFAULT_R_REF_FT,
                        help='radius the sphere is referred to, ft (default %(default)g)')
    parser.add_argument('--window-time', type=float, default=0.5,
                        help='analysis window length, s (default %(default)g)')
    parser.add_argument('--window-overlap', type=float, default=0.5,
                        help='fraction of a window the next one overlaps (default %(default)g)')
    parser.add_argument('--point-stride', type=int, default=10,
                        help='depropagate every Nth sample of the steady track (default %(default)s, '
                             "the 2017 release's setting; build_sphere and build_all default to 1, "
                             'every sample, which is about N times slower and does not reproduce it; '
                             'with --recipe, a recipe without point_stride takes 1)')
    parser.add_argument('--min-elevation-deg', type=float, default=10.0,
                        help='drop emission points below this elevation, where ground '
                             'impedance dominates (default 10)')
    parser.add_argument('--max-array-range-ft', type=float, default=None,
                        help='additionally restrict the track to within this distance of '
                             "the run's reference point; off by default, since capping the "
                             'propagation path bounds the ray directly and this also '
                             'drops whole runs and biases descents toward the flare')
    parser.add_argument('--max-propagation-range-ft', type=float, default=2000.0,
                        help='drop emission-point/microphone pairs farther apart than '
                             'this, beyond which a straight ray through a homogeneous '
                             'atmosphere is a poor model (default 2000)')
    parser.add_argument('--rim-elevation-deg', type=float, default=None,
                        help='with --max-rim-range-ft: samples filed shallower than this, deg below the '
                             'horizon, may come from that range (default {:g})'.format(DEFAULT_RIM_ELEVATION_DEG))
    parser.add_argument('--max-rim-range-ft', type=float, default=None,
                        help='propagation range allowed for samples filed at the rim, ft '
                             '(default: --max-propagation-range-ft everywhere)')
    parser.add_argument('--min-steady-duration-s', type=float, default=8.0,
                        help='shortest steady segment worth building a sphere from, s; '
                             'the binding constraint on how many runs survive the gates '
                             '(default %(default)g)')
    parser.add_argument('--ambient-fallback-percentile', type=float, default=None,
                        help="percentile of the run's own spectrogram to use as ambient (e.g. 5) "
                             'where the array layout has no ambient run; less reliable than a '
                             'measured one')
    parser.add_argument('--azi-step', type=float, default=10.0,
                        help='azimuth step of the gridding, deg (default %(default)g)')
    parser.add_argument('--elv-step', type=float, default=10.0,
                        help='elevation step of the gridding, deg (default %(default)g)')
    parser.add_argument('--rmax', type=float, default=25.0,
                        help='fixed gridding radius, deg (default %(default)g; unused with adaptive gridding)')
    parser.add_argument('--adaptive-interpolation', action='store_true',
                        help='grid with a radius chosen per node (flight_acoustics.adaptive_idw_weights) '
                             'instead of the fixed --rmax (default off)')
    parser.add_argument('--interpolation', dest='interpolation', action='append', default=None,
                        type=_interpolation_setting, metavar='KEY=VALUE',
                        help='an adaptive gridding setting (implies --adaptive-interpolation), e.g. '
                             'kernel=shepard, aspect=5 or shepard_floor=1; repeat for each one; keys: '
                             + ', '.join(sorted(fa.ADAPTIVE_INTERPOLATION)) + ' (default: their defaults)')
    parser.add_argument('--flip-y-for-geometry', action='store_true',
                        help='mirror the track and microphones across y = 0 (a diagnostic; default off)')
    parser.add_argument('--no-prefetch', action='store_true',
                        help='do not warm the cloud-storage cache ahead of each run')
    parser.add_argument('--skip-existing', action='store_true',
                        help='do not rebuild a run whose sphere (and NORAH2 file and tonal and broadband '
                             'parts, when asked for) already exists in output_directory; default: rebuild')
    parser.add_argument('--skip-unchanged', action='store_true',
                        help='do not rebuild a run whose existing sphere was built with the same settings '
                             '(panam_settings_sha256) from the same input files (panam_input_signature: '
                             'names, sizes and modification times); default: rebuild.  A change to '
                             'the PANAM code does not count: rebuild without it after updating PANAM')
    parser.add_argument('--jobs', type=int, default=1,
                        help='build this many runs at once in worker processes (default %(default)s); '
                             'the output is the same as with 1')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='log more: DEBUG messages too (default: INFO and above)')
    parser.add_argument('-q', '--quiet', action='count', default=0,
                        help='log less: -q WARNING and above, -qq ERROR and above (default: INFO and above)')
    parser.add_argument('--log-file', default=None,
                        help='also append the log to this file (default: stderr only)')
    parser.add_argument('--no-ambient-gate', action='store_true',
                        help='reproduce the uncorrected legacy behavior')
    parser.add_argument('--remove-doppler', action='store_true',
                        help='file band power at the emitted frequency, for a hover source sphere; '
                             'never for spheres NICE-OPS reads as flight (default off)')
    parser.add_argument('--tone-aware', action='store_true',
                        help='file each tone whole in the band of its own frequency instead of summing '
                             'whole FFT bins (FFT method only; default off)')
    parser.add_argument('--tone-split', action='store_true',
                        help='also write tonal and broadband part spheres, <name>_tonal.nc and '
                             '<name>_broadband.nc, in tonal/ and broadband/ under output_directory; '
                             'implies --tone-aware (default off)')
    parser.add_argument('--node-statistics', action='store_true',
                        help='also write N_SAMPLES and AMPLITUDE_STD (dB), the samples behind each '
                             'gridded node and their weighted level spread, to every sphere (default off)')
    return parser


def build_options(args):
    """The keyword arguments of :func:`build_all` (and so :func:`build_sphere`) that ``args`` give.

    ``aircraft`` is included; ``output_directory`` is not.
    """
    interpolation = None
    if args.adaptive_interpolation or args.interpolation:
        interpolation = dict(args.interpolation or ())
    return dict(
        aircraft=args.aircraft, root=args.root, runs=args.runs,
        steady_only=not args.include_maneuvers, sphere_prefix=args.sphere_prefix,
        reference_directory=args.reference_directory, manifest_path=args.manifest,
        norah2_directory=args.norah2_directory, prefetch=not args.no_prefetch,
        winds=read_winds(args.winds) if args.winds else None,
        ray_models=_wind_rays if args.ray_model == 'wind' else None,
        ray_profiles=args.ray_profiles,
        third_octave_method=args.third_octave_method,
        board_correction=args.board_correction,
        plate_table_directory=args.plate_table_directory,
        band_snr_gate_db=args.band_snr_gate_db,
        max_absorption_correction_db=args.max_absorption_correction_db,
        max_response_correction_db=args.max_response_correction_db,
        apply_absorption_deprop=not args.no_absorption_deprop,
        absorption_band_method=args.absorption_band_method,
        speed_of_sound_ft_s=args.speed_of_sound_ft_s,
        r_ref_ft=args.r_ref_ft, window_time=args.window_time, window_overlap=args.window_overlap,
        point_stride=args.point_stride, gate_ambient=not args.no_ambient_gate,
        min_elevation_deg=args.min_elevation_deg,
        max_array_range_ft=args.max_array_range_ft,
        max_propagation_range_ft=args.max_propagation_range_ft,
        rim_elevation_deg=args.rim_elevation_deg, max_rim_range_ft=args.max_rim_range_ft,
        min_steady_duration_s=args.min_steady_duration_s,
        ambient_fallback_percentile=args.ambient_fallback_percentile,
        azi_step=args.azi_step, elv_step=args.elv_step, rmax=args.rmax, interpolation=interpolation,
        flip_y_for_geometry=args.flip_y_for_geometry, overwrite=not args.no_overwrite,
        azimuth_reference=args.azimuth_reference, nose_from_heading=args.nose_from_heading,
        remove_doppler=args.remove_doppler, air_relative_doppler=args.air_relative_doppler,
        track_check=not args.no_track_check,
        min_height_above_array_ft=args.min_height_above_array_ft,
        tone_aware=args.tone_aware or args.tone_split,
        tone_split=args.tone_split,
        node_statistics=args.node_statistics,
        skip_existing=args.skip_existing, skip_unchanged=args.skip_unchanged, jobs=args.jobs)


def main(argv=None):
    parser = make_parser()
    # Intermixed, so that the optional positionals may follow options on Python 3.11 too.
    args = parser.parse_intermixed_args(argv)
    options = build_options(args)
    if args.hover_source and not args.recipe:
        parser.error('--hover-source applies to --recipe only')
    if args.recipe:
        import build_recipe          # here: build_recipe imports this module
        try:
            recipe = build_recipe.load_recipe(args.recipe)
            recipe_options = recipe.sphere_options(hover_source=args.hover_source)
        except (ValueError, LookupError, OSError) as error:
            parser.error(str(error))
        options, explicit = build_recipe.apply(parser, argv, recipe_options, options, CLI_KEYWORDS)
        # A keyword neither in the recipe nor on the command line takes the function's
        # default, so that a recipe builds the same spheres here as from Python.  Only
        # point_stride has a command-line default (10) other than the function's (1).
        for key, value in build_recipe.function_defaults('spheres').items():
            if key in options and key not in explicit and key not in recipe_options:
                options[key] = value
        if args.hover_source and not options['runs']:
            parser.error('--hover-source needs --runs: the hover source run (a sphere built with '
                         'remove_doppler must not be read as flight)')
    if options['jobs'] < 1:
        parser.error('--jobs must be at least 1')

    root = logging.getLogger()
    previous_level = root.level
    handler = configure_logging(args.verbose - args.quiet, args.log_file)
    try:
        return _run_command(parser, args, options)
    finally:
        root.setLevel(previous_level)
        if handler is not None:
            root.removeHandler(handler)
            handler.close()


def _run_command(parser, args, options):
    """Everything :func:`main` does once its arguments are parsed and logging is set up."""
    if args.plate_cache:
        directory = _plate_table_directory(options['plate_table_directory'])
        if args.plate_cache == 'clear':
            print('Deleted {} plate tables from {}'.format(clear_plate_table_cache(directory), directory))
            return 0
        records = plate_table_cache(directory)
        for record in records:
            print('{:8.1f} MB  {:<14s} {}'.format(record['size'] / 1e6, record['status'],
                                                   os.path.basename(record['path'])))
        print('{} plate tables in {}'.format(len(records), directory))
        return 0
    aircraft = options.pop('aircraft')
    if aircraft is None:
        parser.error('aircraft is required unless --plate-cache is given')
    if args.check_tracks:
        checks = check_tracks(aircraft, root=options['root'], runs=options['runs'],
                              steady_only=options['steady_only'], hovers=args.hovers,
                              max_array_range_ft=options['max_array_range_ft'],
                              min_steady_duration_s=options['min_steady_duration_s'],
                              min_height_ft=options['min_height_above_array_ft'])
        for record in checks:
            record['aircraft'] = aircraft
            if record['status'] != 'ok':
                print('{} {} {}: {}'.format(record['run'], record['condition'], record['status'],
                                            '; '.join(r for r in (record['reasons'], record['flags']) if r)))
        counts = defaultdict(int)
        for record in checks:
            counts[record['status']] += 1
        print('{}: {} runs checked, {}'.format(aircraft, len(checks), ', '.join(
            '{} {}'.format(n, status) for status, n in sorted(counts.items()))))
        if options['manifest_path']:
            write_track_checks(options['manifest_path'], checks)
        return 0
    if args.output_directory is None:
        parser.error('output_directory is required unless --check-tracks is given')
    if args.ray_model == 'wind' and not options['winds']:
        parser.error('--ray-model wind needs --winds')
    if options['ray_models'] is not None and options['ray_profiles'] is not None:
        parser.error('give --ray-model wind or --ray-profiles, not both')
    records, failures = build_all(aircraft, args.output_directory, **options)
    skipped = sum(1 for record in records if record.get('skipped'))
    logging.info('Built %d spheres, skipped %d, %d failures', len(records) - skipped, skipped, len(failures))
    return 0 if records else 1


if __name__ == '__main__':
    raise SystemExit(main())
