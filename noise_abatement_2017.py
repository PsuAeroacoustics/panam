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

* the tracking file's ``vz`` is positive *down* while ``z`` is positive *up*,
  so flight path angle is ``atan2(-vz, hypot(vx, vy))``;
* the sphere's ``SPEED`` is mean ground speed in knots (``VGk``);
* both labels are means over the steady segment, and any steady sub-window
  reproduces them, so the segment need not match the legacy one exactly.
"""

import os
import csv
import glob
import logging
import re
import warnings
from collections import defaultdict

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from netCDF4 import Dataset

import flight_acoustics as fa
import local_paths

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
#: resistivity 200 kPa s/m^2 (docs/ground_plane_corrections.md, path 4 and
#: the run-holdout profile).
SITE_GROUND = dict(model='variable_porosity', sigma_e=200.0, alpha_e=0.0)

#: Microphone height above its plate's top, feet, per ground-board instrument
#: type.  Every ground microphone in the NASA test was a flush GRAS 67AX in a
#: 400 mm GR1425 plate, at 3/4 of the radius, outboard of the track -- the
#: dataset's ``invgb7`` label ("inverted over a ground board with a 7 mm gap")
#: included, per the test team.  A true inverted layout (``axisymmetric_bem``'s
#: ``mic_height``, plus the microphone body over the gap, which is not yet
#: modelled) is for other tests.
BOARD_MIC_HEIGHT_FT = {'gdbdfl': 0.0, 'invgb7': 0.0}

#: Plate tables are computed at the run's sound speed rounded to this
#: relative step (they scale with frequency / sound speed).
PLATE_TABLE_SOUND_SPEED_STEP = 0.005

#: Legacy sphere reference radius, feet.
DEFAULT_R_REF_FT = 100.0

#: Threads used to warm the cloud-storage cache ahead of each run.
PREFETCH_WORKERS = 24

#: Third-octave band centres carried by the legacy spheres.
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
        correct for.  It reproduces the legacy Be407 sphere set exactly (113
        of 113, no extras, no omissions) and adds a handful of duplicate-speed
        runs the legacy build happened to skip.
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
    36283) in a comment field.  Latin-1 decodes every byte, so the run index
    still reads rather than the whole aircraft failing to load.
    """
    return open(path, encoding='utf-8-sig', errors='replace', newline='')


def vz_sign(track):
    """+1 if the track's ``vz`` is positive up, -1 if positive down, from ``z`` itself.

    The files do not agree: ``z`` is positive up throughout, and ``vz`` is
    positive DOWN in every file (agreement with ``d/dt`` of ``z`` around -0.99)
    except EC130B4 test day 298, whose 49 files have it positive UP (+1.000).
    Assuming down everywhere labelled those days' descents climbs.

    A track with too little vertical motion to judge -- level passes and hovers,
    about 40 files across the dataset, all leaning negative -- falls back to
    down, the form every day but one uses; there the sign hardly matters.
    """
    dz = np.gradient(track['z'], track['time'])
    vz = track['vz']
    # Uncentred, so a steady descent -- constant dz/dt, no variance to
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

    ``z`` is positive up, but whether ``vz`` is depends on the aircraft; see
    :func:`vz_sign`.  :func:`flight_acoustics.hemigen` only takes heading from
    the horizontal components, so the sign does not corrupt the hemisphere
    geometry, but it does set the flight path angle -- which is how a descent
    gets labelled as a climb.
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


def _close_short_gaps(ok, time, max_gap_s):
    """Fill runs of False shorter than ``max_gap_s``.

    Steadiness is judged sample by sample at 50 Hz, so one gust-induced roll
    spike marks a single sample unsteady and splits an otherwise good 40 s
    window into two 20 s halves.  Closing brief gaps keeps the window whole
    while still rejecting a real turn, which lasts seconds rather than
    hundredths.

    2.0 s (not the 0.5 s this was first tuned to) is what generalises: tuned
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
    """
    ok = np.asarray(ok, dtype=bool).copy()
    if max_gap_s <= 0.0:
        return ok
    edges = np.flatnonzero(np.diff(np.concatenate(([True], ok, [True]))))
    for start, stop in zip(edges[::2], edges[1::2]):
        if time[min(stop, time.size - 1)] - time[start] <= max_gap_s:
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
    where the vehicle is within that distance of the array centroid (same
    units as the track).  That bounds the vehicle, not the propagation path --
    this array spans over 6000 ft, so a microphone at the far end is still
    5000+ ft away from a vehicle sitting on top of the centroid.  Bounding the
    path is what straight-ray validity actually asks for, and
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

    # Longest run of True
    best_start = best_stop = start = None
    for index, value in enumerate(ok):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if best_start is None or index - start > best_stop - best_start:
                best_start, best_stop = start, index
            start = None
    if start is not None and (best_start is None or ok.size - start > best_stop - best_start):
        best_start, best_stop = start, ok.size

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
# Acoustic loading
# --------------------------------------------------------------------------

def _read_signal(path, time_range=None, scale=1.0):
    """Read one channel, optionally trimming to an absolute time range."""
    with Dataset(path, mode='r') as handle:
        sample_rate = float(handle.sample_rate)
        start_time = float(handle.start_time)
        location = np.array([float(handle.X), float(handle.Y), float(handle.Z)])
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


def load_run_channels(test, run, mics, time_range=None, ground_board_scale=0.5):
    """Load one run's microphone channels.

    ``ground_board_scale`` accounts for pressure doubling at the ground board;
    the demo scripts apply the same 0.5 factor.
    """
    locations, pressures, times, kept = [], [], [], []
    for mic_number in mics:
        path = test.acoustic_files[run][mic_number]
        try:
            pressure, time, location, _ = _read_signal(path, time_range, ground_board_scale)
        except Exception as error:                      # noqa: BLE001 - one bad channel must not kill the run
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
        path = available[mic_number]
        with Dataset(path, mode='r') as handle:
            start_time = float(handle.start_time)
        pressure, time, _, _ = _read_signal(
            path, (start_time, start_time + float(max_duration_s)), ground_board_scale)
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


def plate_table(instrument, sound_speed_ft_s, bands, ground=None, directory=None):
    """The axisymmetric BEM table (:func:`axisymmetric_bem.table`) for one board type.

    Computed at ``sound_speed_ft_s`` rounded to :data:`PLATE_TABLE_SOUND_SPEED_STEP`
    and cached in memory and in ``directory`` (default
    :func:`default_plate_table_directory`); a table takes 15-45 s.
    """
    import pickle
    import axisymmetric_bem as ab
    ground = dict(SITE_GROUND if ground is None else ground)
    step = np.log1p(PLATE_TABLE_SOUND_SPEED_STEP)
    speed = float(np.exp(np.round(np.log(sound_speed_ft_s) / step) * step))
    bands = np.asarray(bands, dtype=float)
    ground_key = '_'.join('{}{}'.format(k, ground[k]) for k in sorted(ground))
    # Keyed on what the table depends on, not the instrument's name.
    key = 'h{:.2f}mm_{}_c{:.2f}_b{}-{:g}-{:g}'.format(BOARD_MIC_HEIGHT_FT[instrument] * 304.8, ground_key,
                                                      speed, bands.size, bands[0], bands[-1])
    # The key holds only the ends of the band set, so check the bands, as for
    # a table read from disk.
    table = _PLATE_TABLES.get(key)
    if table is not None and np.array_equal(table.get('bands'), bands):
        return table
    directory = os.path.abspath(os.path.expanduser(directory or default_plate_table_directory()))
    path = os.path.join(directory, key + '.pkl')
    table = None
    if os.path.exists(path):
        with open(path, 'rb') as handle:
            table = pickle.load(handle)
        if not np.array_equal(table.get('bands'), bands):
            table = None
    if table is None:
        logging.info('Computing the plate table %s', key)
        flow = ground.get('sigma', ground.get('sigma_e'))
        table = ab.table(bands, speed, flow_resistance=flow, ground=ground,
                         mic_height=BOARD_MIC_HEIGHT_FT[instrument])
        os.makedirs(directory, exist_ok=True)
        temporary = path + '.{}.tmp'.format(os.getpid())
        with open(temporary, 'wb') as handle:
            pickle.dump(table, handle)
        os.replace(temporary, path)             # parallel builds may race; either copy is right
    _PLATE_TABLES[key] = table
    return table


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

def run_atmosphere(test, run, fallback=None):
    """Atmosphere measured by the ground weather stations at the time of ``run``.

    Depropagation *undoes* absorption, so the atmosphere used here sets how much
    the high-frequency bands get multiplied by.  Assuming a dry standard day
    when the test day was humid would inflate them: at 20 C absorption at
    3.15 kHz is about 49 dB/km at 20 % relative humidity but roughly a third of
    that at 70 %, and that difference is applied over kilometres of slant range.

    The stations report ``airtemp`` in degrees Fahrenheit, pressure in kPa and
    humidity in percent.  The file does not say so; the balloon sondes, which
    use the same columns, also log air density, and p / (R rho) reproduces
    their airtemp as Fahrenheit (e.g. AS350B3 day 289: 18.1 against 18.7 F)
    for every aircraft in the dataset.
    """
    row = test.by_run.get(run)
    try:
        run_seconds = float(row.get('utc_secs_from_mid_start'))
    except (TypeError, ValueError, AttributeError):
        run_seconds = None

    day = run[:3]
    samples = []
    pattern = os.path.join(test.base, test.aircraft + '_Weather',
                           test.aircraft + '_Ground_Stations',
                           '{}_{}_*.csv'.format(test.aircraft, day))
    for path in sorted(glob.glob(pattern)):
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
                 nose_from_heading=False, samples_path=None, tone_aware=False):
    """Depropagate one run into an AAM-style source sphere.

    ``remove_doppler`` files band power at the emitted frequency, not the received one
    (:func:`flight_acoustics.depropagate_hemisphere`), and sets the sphere's
    DOPPLER_SHIFT_REMOVED.  For a sphere a hover is synthesised from
    (``build_empirical_database(hover_source=)``), never for one NICE-OPS reads as flight.

    ``nose_from_heading`` orients the sphere by the tracked heading instead of the velocity:
    for a hover, whose velocity is a few tenths of a knot of drift in any direction.  The
    velocity handed to depropagation is then the heading's unit vector at 1e-3 ft/s, which
    sets the azimuth reference and leaves no Doppler or convective term.

    ``tone_aware`` files each tone whole in the band of its own frequency
    (:func:`flight_acoustics.tone_aware_band_power`) instead of summing whole FFT bins.

    ``samples_path``, if given, also saves the scattered samples (before gridding) to that
    .npz: azimuth and elevation (deg, panam's convention: 180 ahead, elevation positive
    below the horizon), band centres, band levels (dB at ``r_ref_ft``, bands x samples),
    microphone, slant range and source height (ft).

    ``ray_model`` (see :mod:`refracted_rays`) depropagates along refracted rays
    through the run's atmosphere instead of straight lines in uniform air: the
    samples are filed at the rays' launch angles, spreading is undone over the
    ray tubes and absorption over the arcs, and the plate correction is taken at
    the rays' arrival angles.  ``min_elevation_deg`` then bounds the launch
    angle.  Default None: straight lines, as before.

    ``max_rim_range_ft`` with ``rim_elevation_deg`` lets samples filed shallower than
    that come from as far as that range instead of ``max_propagation_range_ft``
    (:func:`flight_acoustics.depropagate_hemisphere`'s ``rim_range``), so that every
    pass contributes to the rim, not only the low ones.

    ``interpolation`` grids the samples with a radius chosen per node (see
    :func:`flight_acoustics.adaptive_idw_weights`): ``{}`` for the defaults,
    or settings to override them.  Default None: the fixed ``rmax``.

    ``board_correction`` removes the ground board's effect: ``'plate_bem'``
    (default) divides each band by the plate's modelled response for that
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
    ground speed the AAM sphere is labelled with; the tracking data carries no
    airspeed.

    Returns a dict describing what was processed, so a batch caller can log and
    audit it without re-opening the output.
    """
    row = test.by_run[run]
    track = load_track(test.track_path(run))
    if nose_from_heading:
        # A hover's flight path angle is the direction of its drift, erratic at a few tenths
        # of a knot, so the steady-flight window would reject it: take the whole record.
        index_start, index_stop = 0, int(track['time'].size)
    else:
        index_start, index_stop = steady_window(track, max_array_range=max_array_range_ft,
                                                min_duration_s=min_steady_duration_s)
    speed_knots, fpa_deg = flight_condition(track, index_start, index_stop)

    segment = {key: value[index_start:index_stop] for key, value in track.items()
               if isinstance(value, np.ndarray) and value.size == track['time'].size}
    position = np.column_stack((segment['x'], segment['y'], segment['z']))
    velocity = np.column_stack((segment['vx'], segment['vy'], segment['vz_up']))
    if nose_from_heading:
        # Compass heading to the track frame (+x along true bearing 270, +y south):
        # east = -x, north = -y.
        heading = np.radians(np.asarray(segment['heading'], dtype=float))
        velocity = 1e-3 * np.column_stack((-np.sin(heading), -np.cos(heading), np.zeros_like(heading)))

    if atmosphere is None:
        atmosphere = run_atmosphere(test, run, fallback=fa.Atmosphere(
            temperature=293.15, pressure=101.325, relative_humidity=20.0))
    if speed_of_sound_ft_s is None:
        speed_of_sound_ft_s = float(fa.unit_conversion.len_conv(
            atmosphere.soundspeed, from_units='m', to_units='ft'))

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
        # estimate lands within about +-10 dB of the real thing per channel,
        # so it is a fallback, not an equivalent.  Reported in the manifest.
        ambient_source = 'percentile-{:g}'.format(ambient_fallback_percentile)
    elif not gate_ambient:
        ambient_source = 'none'
    if ambient_run is not None:
        mics = [m for m in mics if m in test.acoustic_files.get(ambient_run, {})]
        if not mics:
            raise ValueError('Run {}: no microphone has both run and ambient data'.format(run))

    # Trim the recordings to the observer times that the steady segment can
    # reach.  A full run is ~50 channels x 90 s x 25 kHz; loading only what is
    # used keeps a run inside a few hundred MB instead of a couple of GB.
    locations = np.array([_read_signal(test.acoustic_files[run][m], (0.0, 0.0))[2]
                          for m in mics])
    geometry_locations = locations.copy()
    if flip_y_for_geometry:
        geometry_locations[:, 1] *= -1.0
    ranges = np.sqrt(((position[:, None, :] - geometry_locations[None, :, :]) ** 2).sum(axis=2))
    time_range = (float(segment['time'][0]),
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

    reference = legacy_sphere_grid(reference_sphere)
    if reference is not None:
        band_centers = reference['band_centers_hz']
        phi_deg, theta_deg = reference['phi_deg'], reference['theta_deg']
    else:
        band_centers = LEGACY_BAND_CENTERS_HZ
        phi_deg = theta_deg = None

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
                   (14.0 if rim_elevation_deg is None else rim_elevation_deg, max_rim_range_ft)),
        remove_doppler=remove_doppler,
        tone_aware=tone_aware,
        return_scattered=samples_path is not None,
    )
    if samples_path is not None:
        scattered = hemisphere['scattered']
        os.makedirs(os.path.dirname(os.path.abspath(samples_path)) or '.', exist_ok=True)
        np.savez(samples_path, azimuth_deg=scattered['azi_deg'], elevation_deg=scattered['elv_deg'],
                 band_centers_hz=scattered['third_octave']['band_centers_hz'],
                 bands_db=scattered['third_octave']['bands_db'], mic=scattered['mic'],
                 range_ft=scattered['range'], source_height_ft=scattered['source_height'], run=str(run))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    fa.write_aam_hemisphere_netcdf(
        output_path,
        hemisphere,
        mode='third_octave',
        phi_deg=phi_deg,
        theta_deg=theta_deg,
        radius_ft=r_ref_ft,
        speed_knots=speed_knots,
        flight_path_angle_deg=fpa_deg,
        title=sphere_title(AIRCRAFT_SPHERE_PREFIX.get(test.aircraft, test.aircraft),
                           row.get('run_num', run), run[:3]),
        overwrite=overwrite,
        doppler_shift_removed=1.0 if remove_doppler else 0.0,
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

    return dict(run=run, output=output_path, norah2_output=norah2_output_path, condition=row.get('test_cond'),
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
                rays='straight' if ray_model is None else getattr(ray_model, 'description', 'custom'),
                interpolation=('fixed rmax {:g}'.format(rmax) if interpolation is None else
                               'adaptive ' + ' '.join('{}={}'.format(k, v) for k, v in sorted(
                                   dict(fa.ADAPTIVE_INTERPOLATION, **interpolation).items()))),
                gaps=(hemisphere.get('interpolation', {}).get('gaps', 0)),
                rim_range='' if max_rim_range_ft is None else '{:g} ft below {:g} deg'.format(
                    max_rim_range_ft, 14.0 if rim_elevation_deg is None else rim_elevation_deg),
                board_correction=board_correction if board_correction == 'flat' else
                'plate_bem ' + ' '.join('{}={}'.format(k, v) for k, v in
                                         sorted((ground or SITE_GROUND).items())))


# --------------------------------------------------------------------------
# Batch
# --------------------------------------------------------------------------

def _run_file_paths(test, run):
    """Every acoustic file one run needs, its ambient recording included."""
    mics = test.ground_board_mics(run)
    paths = [test.acoustic_files[run][m] for m in mics if m in test.acoustic_files.get(run, {})]
    ambient = test.ambient_run(run)
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


def build_all(aircraft, output_directory, *, root=None, runs=None,
              steady_only=True, reference_directory=None, sphere_prefix=None,
              manifest_path=None, prefetch=True, norah2_directory=None, ray_models=None, **kwargs):
    """Rebuild every usable run for one aircraft.

    ``ray_models``, if given, is called with each run id and returns that run's
    ``ray_model`` for :func:`build_sphere` (its own atmosphere), or None to
    build that run along straight lines; the manifest's ``rays`` column says
    which.

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

    Returns ``(records, failures)``.
    """
    test = NoiseAbatementTest(aircraft, root=root)
    if runs is None:
        runs = test.runs(steady_only=steady_only)
    prefix = sphere_prefix or AIRCRAFT_SPHERE_PREFIX.get(aircraft, aircraft)
    output_directory = os.path.abspath(os.path.expanduser(output_directory))
    os.makedirs(output_directory, exist_ok=True)

    records, failures = [], []
    pool = ThreadPoolExecutor(max_workers=1) if prefetch and runs else None
    pending = pool.submit(_prefetch, _run_file_paths(test, runs[0])) if pool else None
    for number, run in enumerate(runs, start=1):
        if pending is not None:
            pending.result()
            pending = (pool.submit(_prefetch, _run_file_paths(test, runs[number]))
                       if number < len(runs) else None)
        row = test.by_run.get(run, {})
        name = '{}{}.nc'.format(prefix, row.get('run_num', run))
        reference = None
        if reference_directory:
            candidate = os.path.join(os.path.expanduser(reference_directory), name)
            reference = candidate if os.path.exists(candidate) else None
        try:
            record = build_sphere(test, run, os.path.join(output_directory, name),
                                  reference_sphere=reference, norah2_directory=norah2_directory,
                                  ray_model=ray_models(run) if ray_models is not None else None,
                                  **kwargs)
        except Exception as error:                      # noqa: BLE001
            logging.warning('[%d/%d] %s failed: %s', number, len(runs), run, error)
            failures.append(dict(run=run, error=str(error),
                                 condition=row.get('test_cond'),
                                 layout=row.get('layout')))
            continue
        records.append(record)
        logging.info('[%d/%d] %s -> %s  %.1f kt  %+.1f deg  %d mics  ambient %s',
                     number, len(runs), run, name, record['speed_knots'],
                     record['flight_path_angle_deg'], record['mics'],
                     record['ambient_run'])

    if pool is not None:
        pool.shutdown()
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


def write_manifest(path, records, failures):
    """Record what went into the spheres, so a database can be audited later."""
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fields = ['run', 'output', 'norah2_output', 'condition', 'layout', 'ambient_run',
              'ambient_source', 'mics',
              'speed_knots', 'flight_path_angle_deg', 'window_s', 'window_points',
              'min_elevation_deg', 'max_array_range_ft', 'temperature_k',
              'relative_humidity', 'pressure_kpa', 'speed_of_sound_ft_s', 'board_correction',
              'rays', 'interpolation', 'gaps', 'rim_range', 'error']
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for record in records:
            writer.writerow(record)
        for failure in failures:
            writer.writerow(failure)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description='Rebuild 2017 Noise Abatement source spheres with ambient gating.')
    parser.add_argument('aircraft', help='dataset directory name, e.g. B407')
    parser.add_argument('output_directory')
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
    parser.add_argument('--manifest', default=None)
    parser.add_argument('--norah2-directory', default=None,
                        help='also write each sphere as a NORAH2 .hem file here, with the '
                             'triangulation file NORAH2 needs to interpolate between them')
    parser.add_argument('--board-correction', choices=('plate_bem', 'flat'), default='plate_bem',
                        help="'plate_bem' (default) divides out the ground plate's modelled "
                             "response per band and frame; 'flat' is the old constant -6 dB")
    parser.add_argument('--third-octave-method', choices=('fft', 'filter_bank'), default='fft',
                        help="'filter_bank' forms bands with a true one-third octave filter "
                             "bank, as an analyser does; it differs from the default FFT band "
                             "sum only below ~100 Hz, between strong rotor tones")
    parser.add_argument('--band-snr-gate-db', type=float, default=10.0)
    parser.add_argument('--max-absorption-correction-db', type=float, default=30.0,
                        help='discard bins needing more absorption correction than this; '
                             '30 dB is what keeps source spectra rolling off physically')
    parser.add_argument('--point-stride', type=int, default=10)
    parser.add_argument('--min-elevation-deg', type=float, default=10.0,
                        help='drop emission points below this elevation, where ground '
                             'impedance dominates (default 10)')
    parser.add_argument('--max-array-range-ft', type=float, default=None,
                        help='additionally restrict the track to within this distance of '
                             'the array centroid; off by default, since capping the '
                             'propagation path bounds the ray directly and this also '
                             'drops whole runs and biases descents toward the flare')
    parser.add_argument('--max-propagation-range-ft', type=float, default=2000.0,
                        help='drop emission-point/microphone pairs farther apart than '
                             'this, beyond which a straight ray through a homogeneous '
                             'atmosphere is a poor model (default 2000)')
    parser.add_argument('--min-steady-duration-s', type=float, default=8.0,
                        help='shortest steady segment worth building a sphere from; '
                             'the binding constraint on how many runs survive the gates')
    parser.add_argument('--ambient-fallback-percentile', type=float, default=None,
                        help='estimate ambient from the run itself (e.g. 5) where the array '
                             'layout has no ambient run; less reliable than a measured one')
    parser.add_argument('--no-prefetch', action='store_true',
                        help='do not warm the cloud-storage cache ahead of each run')
    parser.add_argument('--no-ambient-gate', action='store_true',
                        help='reproduce the uncorrected legacy behaviour')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    records, failures = build_all(
        args.aircraft, args.output_directory, root=args.root, runs=args.runs,
        steady_only=not args.include_maneuvers,
        reference_directory=args.reference_directory, manifest_path=args.manifest,
        norah2_directory=args.norah2_directory,
        third_octave_method=args.third_octave_method,
        board_correction=args.board_correction,
        band_snr_gate_db=args.band_snr_gate_db,
        max_absorption_correction_db=args.max_absorption_correction_db,
        point_stride=args.point_stride, gate_ambient=not args.no_ambient_gate,
        min_elevation_deg=args.min_elevation_deg,
        max_array_range_ft=args.max_array_range_ft,
        max_propagation_range_ft=args.max_propagation_range_ft,
        min_steady_duration_s=args.min_steady_duration_s,
        ambient_fallback_percentile=args.ambient_fallback_percentile,
        prefetch=not args.no_prefetch)
    logging.info('Built %d spheres, %d failures', len(records), len(failures))
    return 0 if records else 1


if __name__ == '__main__':
    raise SystemExit(main())
