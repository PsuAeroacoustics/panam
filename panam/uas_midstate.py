"""Read the PSU Mid-State UAS flight-test dataset: one NetCDF-4 run file per recording.

Each run file holds one microphone-array recording and everything measured with it, every time axis in seconds
after the run's ``time_origin`` (UTC, with the recorder's clock corrected):

``/`` (attributes)
    run id, vehicle, ``status`` (``flight``, ``flight_no_log`` with ``status_reason``, ``ambient``), the run log's
    row (text as written, and ``runlog_heights_ft``, ``runlog_speeds`` as numbers), clock corrections, the site
    origin (``site_origin_lat``, ``site_origin_lon``, ``site_origin_height``) and its datums, ACDD discovery metadata.
``/acoustic``
    ``counts`` (mic, sample) int32 and ``step`` (mic) in Pa per count: **pressure = counts x step**. The counts
    are the recorder's own integers. Sample ``i`` is at ``i / sample_rate`` s. Per microphone: ``mic_id``
    (physical microphone), ``channel`` (recorder channel), ``mic_east``, ``mic_north``, ``mic_up`` (m, site
    frame), ``mic_x``, ``mic_y``, ``mic_z`` (m, array frame), ``mic_ground_up``, ``mic_height_above_ground``,
    ``mic_lat``, ``mic_lon``, ``mic_pole_offset_east``, ``mic_pole_offset_north``, ``mic_pole_offset_uncertainty``
    (m; a pole's measured offset from its plate, already included in its position; NaN otherwise), ``mic_mount``
    (inverted ground plane or pole), ``survey_point``, ``notes``, and ``mic_quality`` (flags: clipped, dropout,
    in_downwash, position_uncertain, runlog_note; :meth:`RunFile.quality`).
``/vehicle/<topic>``
    PX4 state at its own rates: ``position`` (east, north, up and array-frame x, y, z in m; lat, lon, alt AMSL),
    ``velocity`` (both frames, m/s), ``attitude`` (``q_w`` .. ``q_z``; roll, pitch, yaw in rad), ``angular_rate``
    (``roll_rate``, ``pitch_rate``, ``yaw_rate``), ``motor_command`` (``command`` (motor, time)),
    ``thrust_setpoint`` (``thrust_x``, ``thrust_y``, ``thrust_z``), ``battery``, ``barometer`` (the autopilot's),
    ``status`` (``arming_state``, ``nav_state``: integer flags with ``flag_meanings``), ``land_detector``
    (``landed``). Absent when no flight log matched the recording.
``/rotors/motor<k>``
    onboard rotor ``speed`` (rad/s, negative clockwise seen from above) and, where logged, ``angle`` (rad, the
    motor controller's own zero). The ``/rotors`` group's attributes give the angle definition and
    ``prop_aligned_angle`` (per motor, in motor order: the reported angle with the prop along its arm, where that
    was measured); :meth:`RunFile.rotors` hands each motor its own entry.
``/air_data``
    the aircraft-mounted air data system. Its instrument's own record (``wx150_<n>``, ``trisonica_<n>``) measures
    the air moving past the aircraft: ``airspeed_measured`` (uncalibrated), ``airflow_angle``, ``air_pressure``.
    ``airspeed`` holds the calibrated airspeed and the wind derived from it, with the calibration in its attributes.
``/weather/<instrument>_<n>``
    ground weather stations (``wind_speed``, ``wind_from_direction``, ``station_pressure``, ...), each group with
    its instrument; each variable's units are in its ``units`` attribute (:meth:`RunFile.units`).
``/segments``
    the flight's steady segments: passes, climbs, descents and hovers, with their conditions, the run log's nominal
    height and speed (numbers, NaN where none matched), and each microphone's closest approach, kept to the
    recording (vehicle times from ``time_origin`` to its last sample).

Every group with a time axis, and ``/acoustic``, gives its clock: ``clock`` (words), ``clock_offset_s`` (added to
the instrument's own time stamps to put them on UTC) and ``clock_offset_uncertainty_s`` (NaN where not assessed).

Positions and velocities are given in the site frame (east, north, up from ``site_origin``) and in the array frame
(``x`` along runway 16/34, ``y`` lateral to the left, ``z`` up; root attribute ``array_frame``).

Each group's ``time`` is CF (``seconds since <time_origin>``), so xarray decodes it to datetimes unless opened
with ``decode_times=False``; this module returns the seconds, and refuses a time axis on any other origin. Integer
(flag) variables come back as integers, other numbers as float64 with unwritten cells NaN, strings as object
arrays, a group's attributes as ``'@name'`` entries. Lengths here are in m and speeds in m/s; :meth:`RunFile.track` and :meth:`RunFile.receiver` convert to
ft for :mod:`panam.order_tracking` and the 2017 pipeline. A missing optional group (``/rotors``, ``/air_data``,
``/weather``, ``/segments``) reads as ``{}``; asking for a missing ``/vehicle`` topic raises KeyError. The reader
reads format 0.4 (:data:`FORMAT_VERSIONS`) and refuses other versions.
"""
import datetime as dt
import os
import re

import netCDF4
import numpy as np

M_TO_FT = 1.0 / 0.3048
#: The run-file format versions this module reads (root attribute ``format_version``).
FORMAT_VERSIONS = ('0.4',)
_SCALES = {'m': 1.0, 'ft': M_TO_FT}
_FRAMES = {'site': ('east', 'north', 'up'), 'array': ('x', 'y', 'z')}


def _values(var):
    """A variable's values: strings as an object array; integers (flags) as integers; other numbers as float64.
    Unwritten or fill cells are NaN (an integer variable with any becomes float64)."""
    if var.dtype is str:
        return np.array(var[:], dtype=object)
    data = var[:]
    if np.issubdtype(var.dtype, np.integer) and not np.ma.is_masked(data):
        return np.asarray(data)
    return np.ma.filled(np.ma.asarray(data, dtype=float), np.nan)


def _choice(table, key, what):
    if key not in table:
        raise ValueError(f'{what} {key!r}: use one of {", ".join(map(repr, table))}')
    return table[key]


class RunFile:
    """One run file, opened read-only. Use as a context manager, or call :meth:`close`.

    Nothing large is read until asked for: :meth:`pressure` reads only the requested microphones and samples.
    A file whose ``format_version`` is not in :data:`FORMAT_VERSIONS` is refused (ValueError).
    """

    def __init__(self, path):
        self.path = os.fspath(path)
        self._nc = netCDF4.Dataset(self.path, 'r')
        version = self._nc.format_version if 'format_version' in self._nc.ncattrs() else None
        if version not in FORMAT_VERSIONS:
            self._nc.close()
            raise ValueError(f'{self.path}: format_version {version!r}; this reader reads '
                             f'{", ".join(FORMAT_VERSIONS)}. Rebuild the file, or use the reader of its version')
        # The counts are the recorder's integers: no fill value or scaling applies to them.
        self._nc['acoustic']['counts'].set_auto_maskandscale(False)

    def close(self):
        """Close the file; closing it again does nothing."""
        if self._nc.isopen():
            self._nc.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ----------------------------------------------------------------------------------------- metadata

    @property
    def attrs(self):
        """The run's metadata (root attributes): run id, vehicle, run-log fields, clock corrections, origin."""
        return {k: self._nc.getncattr(k) for k in self._nc.ncattrs()}

    @property
    def groups(self):
        """Paths of every group in the file."""
        out = []

        def walk(g, prefix):
            for name, sub in g.groups.items():
                out.append(f'{prefix}/{name}')
                walk(sub, f'{prefix}/{name}')
        walk(self._nc, '')
        return out

    def _group(self, path):
        """The group at ``path`` ('vehicle/position'); KeyError naming the groups there if it is missing."""
        g = self._nc
        for part in path.strip('/').split('/'):
            if part not in g.groups:
                here = ', '.join(sorted(g.groups)) or 'none'
                raise KeyError(f'{self.path} has no group /{path} (run status {self.attrs.get("status", "")!r}; '
                               f'groups in /{g.path.strip("/")}: {here})')
            g = g.groups[part]
        return g

    @property
    def time_origin(self):
        """UTC of acoustic sample 0 and of time 0 on every group's axis, as numpy datetime64 (ns)."""
        t = dt.datetime.fromisoformat(self._nc.time_origin)
        if t.tzinfo is not None:
            t = t.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return np.datetime64(t, 'ns')

    def utc(self, seconds):
        """UTC (datetime64 ns) of times given in seconds after time_origin."""
        return self.time_origin + (np.asarray(seconds, dtype=float) * 1e9).astype('timedelta64[ns]')

    @property
    def sample_rate(self):
        """Acoustic sample rate (Hz)."""
        return float(self._nc['acoustic'].sample_rate)

    @property
    def n_samples(self):
        """Number of acoustic samples per microphone."""
        return len(self._nc['acoustic'].dimensions['sample'])

    def microphones(self):
        """dict of per-microphone arrays, in file order: ``mic_id``, ``channel``, ``mic_quality`` (flags), ``mic_east``, ``mic_north``,
        ``mic_up`` (m, site frame), ``mic_x``, ``mic_y``, ``mic_z`` (m, array frame), ``mic_mount``,
        ``mic_height_above_ground``, ``mic_lat``, ``mic_lon``, ``mic_pole_offset_*``, ``step`` (Pa per count), ...
        Strings as object arrays, numbers as float64 (NaN where not given)."""
        a = self._nc['acoustic']
        return {name: _values(var) for name, var in a.variables.items() if var.dimensions == ('mic',)}

    def mic_index(self, mic):
        """Index of a microphone given its id (``mic16``), channel (``M8``) or index (0 to n_mics - 1)."""
        a = self._nc['acoustic']
        if isinstance(mic, (int, np.integer)):
            n = len(a.dimensions['mic'])
            if not 0 <= mic < n:
                raise IndexError(f'microphone index {mic} outside 0..{n - 1} in {self.path}')
            return int(mic)
        for key in ('mic_id', 'channel'):
            names = list(a[key][:])
            if mic in names:
                return names.index(mic)
        raise KeyError(f'{mic!r} is neither a microphone id nor a channel of {self.path}')

    def quality(self, mic):
        """Names of the quality flags set for a microphone in this run ([] when none), from ``mic_quality``."""
        q = self._nc['acoustic']['mic_quality']
        value = int(q[self.mic_index(mic)])
        masks, names = np.atleast_1d(q.flag_masks), q.flag_meanings.split()
        return [n for m, n in zip(masks, names) if value & int(m)]

    # ------------------------------------------------------------------------------------------ acoustics

    def _range(self, start, stop):
        """Sample indices [start, stop) clipped to the recording; negative indices are refused."""
        n = self.n_samples
        start = 0 if start is None else int(start)
        stop = n if stop is None else int(stop)
        if start < 0 or stop < 0:
            raise ValueError(f'sample indices count from sample 0 at time_origin; got negative start {start} or '
                             f'stop {stop} (use samples() to convert times, which clips to the recording)')
        stop = min(stop, n)
        return min(start, stop), stop

    def pressure(self, mics=None, start=None, stop=None):
        """Acoustic pressure (Pa, float64), shape (mic, sample), for samples [start, stop).

        ``mics``: None (all), one id, channel or index, or a list of them (one row each, in that order).
        ``start``/``stop`` are sample indices from 0, clipped to the end of the recording; negative ones are
        refused. :meth:`samples` converts from seconds. The result is the only large allocation (8 bytes per
        sample), filled one microphone at a time.
        """
        a = self._nc['acoustic']
        if mics is None:
            idx = range(len(a.dimensions['mic']))
        elif isinstance(mics, (str, int, np.integer)):
            idx = [self.mic_index(mics)]
        else:
            idx = [self.mic_index(m) for m in mics]
        start, stop = self._range(start, stop)
        step = np.asarray(a['step'][:], dtype=float)
        counts = a['counts']
        out = np.empty((len(idx), stop - start))
        for j, i in enumerate(idx):
            out[j] = counts[i, start:stop]
            out[j] *= step[i]
        return out

    def samples(self, t_start, t_stop):
        """Sample index range [start, stop) covering t_start..t_stop (s after time_origin), clipped to the
        recording: a window wholly before or after it gives an empty range (start == stop)."""
        fs, n = self.sample_rate, self.n_samples
        # rounded to a millionth of a sample first, so that 0.29 s at 100 Hz is sample 29, not 28
        start = int(np.floor(np.round(t_start * fs, 6)))
        stop = int(np.ceil(np.round(t_stop * fs, 6)))
        start = min(max(start, 0), n)
        return start, min(max(stop, start), n)

    def time(self, start=None, stop=None):
        """Time (s after time_origin) of acoustic samples [start, stop), clipped as :meth:`pressure` clips."""
        start, stop = self._range(start, stop)
        return np.arange(start, stop) / self.sample_rate

    # ------------------------------------------------------------------------------------- other streams

    def _series(self, path):
        """A group's variables (see :func:`_values`) and its attributes as '@name'. Every time axis must be in
        seconds since the run's time_origin (ValueError otherwise)."""
        g = self._group(path)
        origin = f'seconds since {self._nc.time_origin}'
        out = {}
        for name, var in g.variables.items():
            units = var.units if 'units' in var.ncattrs() else ''
            if units.startswith('seconds since') and units != origin:
                raise ValueError(f'{self.path}: /{path}/{name} is in {units!r}, not seconds since the run\'s '
                                 f'time_origin {self._nc.time_origin!r}')
            out[name] = _values(var)
        out.update({f'@{k}': g.getncattr(k) for k in g.ncattrs()})
        return out

    def units(self, path):
        """{variable: its units attribute ('' if none)} of a group, e.g. ``units('vehicle/motor_command')``."""
        return {name: (var.units if 'units' in var.ncattrs() else '')
                for name, var in self._group(path).variables.items()}

    def vehicle(self, topic='position'):
        """One PX4 topic as a dict of arrays with its own ``time`` (s after time_origin); attributes as '@name'.
        KeyError when the run has no flight log (no /vehicle) or no such topic."""
        return self._series(f'vehicle/{topic}')

    def rotors(self):
        """{motor number k (PX4 numbering): dict(time, speed (rad/s, negative clockwise seen from above), angle
        (rad) if logged)}, or {} without onboard rotor data. Each motor's dict also carries the /rotors group's
        attributes as '@name' (``@clock``, ``@angle_definition``, ...), with ``@prop_aligned_angle`` reduced to
        motor k's entry (rad), where it was measured."""
        if 'rotors' not in self._nc.groups:
            return {}
        r = self._nc['rotors']
        common = {f'@{k}': r.getncattr(k) for k in r.ncattrs()}
        aligned = np.atleast_1d(common.pop('@prop_aligned_angle', []))
        out = {}
        for name in r.groups:
            m = re.fullmatch(r'motor(\d+)', name)
            if not m:
                continue
            k = int(m.group(1))
            d = {**common, **self._series(f'rotors/{name}')}
            if 1 <= k <= len(aligned):
                d['@prop_aligned_angle'] = float(aligned[k - 1])
            out[k] = d
        return dict(sorted(out.items()))

    def weather(self):
        """Ground weather stations: {group name: dict of arrays with its own time}; attributes (location,
        instrument, clock) as '@name'. {} when the run had none. Units: :meth:`units`."""
        if 'weather' not in self._nc.groups:
            return {}
        return {name: self._series(f'weather/{name}') for name in self._nc['weather'].groups}

    def air_data(self):
        """The aircraft-mounted air data: the instrument's own record (``wx150_1``, ...) and the derived
        ``airspeed`` group (calibrated airspeed, its body components, the air-relative velocity and the wind, in
        m/s; attributes give the calibration). {} when the run had none."""
        if 'air_data' not in self._nc.groups:
            return {}
        return {name: self._series(f'air_data/{name}') for name in self._nc['air_data'].groups}

    def segments(self):
        """The steady segments of the flight (passes, climbs, descents, hovers): dict of per-segment arrays (``kind``,
        ``start``/``end`` s after time_origin, ``direction``, ``ground_speed``, ``height``, ``airspeed``,
        ``nominal_height_ft`` (a string, '' when none matched), ...) and per segment and microphone
        (``closest_distance``, ``closest_time``, ``closest_arrival``, columns in ``channel`` order). The writer
        keeps segments to the recording; :meth:`segment_samples` still checks. {} when the run has no
        /segments group; a run without a flight log has the group with no segments."""
        if 'segments' not in self._nc.groups:
            return {}
        return self._series('segments')

    def segment_samples(self, index):
        """Acoustic sample range [start, stop) of segment `index`'s steady portion (vehicle time; add the
        propagation delay, e.g. ``closest_arrival - closest_time``, for what a microphone heard). A segment that
        runs past either end of the recording is clipped to it; one wholly outside it raises ValueError."""
        s = self.segments()
        if not s:
            raise KeyError(f'{self.path} has no /segments group')
        start, stop = self.samples(s['start'][index], s['end'][index])
        if start >= stop:
            raise ValueError(f'segment {index} ({s["kind"][index]}, {s["start"][index]:.2f} to {s["end"][index]:.2f} s) '
                             f'lies outside the recording (0 to {self.n_samples / self.sample_rate:.2f} s)')
        return start, stop

    def track(self, units='m', frame='site'):
        """The vehicle's position as a track dict (``time`` s, sorted, repeated times dropped; ``x``, ``y``, ``z``)
        in m or ft, the form :func:`panam.order_tracking.doppler_factor` takes. ``frame='site'``: east, north, up
        from the site origin; ``frame='array'``: along runway 16/34, lateral (left), up (the root attribute
        ``array_frame``). KeyError when the run has no flight log."""
        scale = _choice(_SCALES, units, 'units')
        cx, cy, cz = _choice(_FRAMES, frame, 'frame')
        p = self.vehicle('position')
        t, keep = np.unique(p['time'], return_index=True)
        return {'time': t, 'x': p[cx][keep] * scale, 'y': p[cy][keep] * scale, 'z': p[cz][keep] * scale}

    def receiver(self, mic, units='m', frame='site'):
        """A microphone's (3,) position in m or ft, in the site frame (east, north, up) or the array frame. A
        pole's position includes its measured offset from its plate (``mic_pole_offset_*``)."""
        scale = _choice(_SCALES, units, 'units')
        cx, cy, cz = _choice(_FRAMES, frame, 'frame')
        i = self.mic_index(mic)
        a = self._nc['acoustic']
        return np.array([a[f'mic_{cx}'][i], a[f'mic_{cy}'][i], a[f'mic_{cz}'][i]], dtype=float) * scale


def open_run(path):
    """Open a run file: ``with open_run(p) as run: p = run.pressure(['M8'])``."""
    return RunFile(path)


def list_runs(root):
    """Every run file under a dataset root (``runs/<date>/<recording>.nc``), sorted. FileNotFoundError if the
    root has no ``runs`` directory."""
    runs = os.path.join(os.fspath(root), 'runs')
    if not os.path.isdir(runs):
        raise FileNotFoundError(f'no runs directory under {os.fspath(root)!r}')
    out = []
    for dirpath, _, files in os.walk(runs):
        out += [os.path.join(dirpath, f) for f in files if f.endswith('.nc') and not f.startswith('._')]
    return sorted(out)
