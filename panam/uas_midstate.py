"""Read the PSU Mid-State UAS flight-test dataset: one NetCDF-4 run file per recording.

Each run file holds one microphone-array recording and everything measured with it, every time axis in seconds
after the run's ``time_origin`` (UTC, with the recorder's clock corrected):

``/acoustic``
    ``counts`` (mic, sample) int32 and ``step`` (mic) in Pa per count: **pressure = counts x step**. The counts
    are the recorder's own integers. Sample ``i`` is at ``i / sample_rate`` s. Per microphone: ``mic``
    (physical microphone id), ``channel`` (recorder channel), ``mic_east``, ``mic_north``, ``mic_up`` (m, site
    frame), ``mic_ground_up``, ``mic_height_above_ground``, ``mic_lat``, ``mic_lon``, ``mic_mount`` (ground board
    or pole), ``survey_point``, ``notes``.
``/vehicle/<topic>``
    PX4 state at its own rates: ``position`` (east, north, up in m; lat, lon, alt AMSL), ``velocity``,
    ``attitude`` (quaternion, roll, pitch, yaw in rad), ``angular_rate``, ``motor_command``,
    ``thrust_setpoint``, ``battery``, ``air_data``.
``/rotors/motor<k>``
    onboard rotor ``speed`` (rad/s, negative clockwise seen from above) and, where logged, ``angle`` (rad).
``/weather/<location>_<n>``
    ground and aircraft-mounted ("onboard") weather, SI units, each group with its instrument and clock.

The site frame is east, north, up (m) from the root attribute ``site_origin``. Lengths here are in m and speeds
in m/s; :meth:`RunFile.track` converts to ft for :mod:`panam.order_tracking` and the 2017 pipeline.
"""
import os

import netCDF4
import numpy as np

M_TO_FT = 1.0 / 0.3048


class RunFile:
    """One run file, opened read-only. Use as a context manager, or call :meth:`close`.

    Nothing large is read until asked for: :meth:`pressure` reads only the requested microphones and samples.
    """

    def __init__(self, path):
        self.path = os.fspath(path)
        self._nc = netCDF4.Dataset(self.path, 'r')
        self._nc.set_auto_maskandscale(False)

    def close(self):
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

    @property
    def sample_rate(self):
        """Acoustic sample rate (Hz)."""
        return float(self._nc['acoustic'].sample_rate)

    @property
    def n_samples(self):
        return len(self._nc['acoustic'].dimensions['sample'])

    def microphones(self):
        """dict of per-microphone arrays: id, channel, position (east, north, up m), mount, height, lat, lon, ..."""
        a = self._nc['acoustic']
        out = {}
        for name, var in a.variables.items():
            if var.dimensions == ('mic',) and name != 'step':
                out[name] = np.array(var[:])
        out['step'] = np.array(a['step'][:], dtype=float)
        return out

    def mic_index(self, mic):
        """Index of a microphone given its id (``mic16``), channel (``M8``) or index."""
        if isinstance(mic, (int, np.integer)):
            return int(mic)
        a = self._nc['acoustic']
        for key in ('mic', 'channel'):
            names = list(a[key][:])
            if mic in names:
                return names.index(mic)
        raise KeyError(f'{mic!r} is neither a microphone id nor a channel of {self.path}')

    # ------------------------------------------------------------------------------------------ acoustics

    def pressure(self, mics=None, start=None, stop=None):
        """Acoustic pressure (Pa, float64), shape (mic, sample), for samples [start, stop).

        ``mics``: None (all), or a list of ids, channels or indices. ``start``/``stop`` are sample indices; see
        :meth:`samples` to convert from seconds.
        """
        a = self._nc['acoustic']
        idx = range(len(a.dimensions['mic'])) if mics is None else [self.mic_index(m) for m in mics]
        start = 0 if start is None else int(start)
        stop = self.n_samples if stop is None else int(stop)
        step = np.asarray(a['step'][:], dtype=float)
        return np.stack([np.asarray(a['counts'][i, start:stop], dtype=np.float64) * step[i] for i in idx])

    def samples(self, t_start, t_stop):
        """Sample index range [start, stop) covering t_start..t_stop (s after time_origin)."""
        fs = self.sample_rate
        return max(0, int(np.floor(t_start * fs))), min(self.n_samples, int(np.ceil(t_stop * fs)))

    def time(self, start=None, stop=None):
        """Time (s after time_origin) of acoustic samples [start, stop)."""
        start = 0 if start is None else int(start)
        stop = self.n_samples if stop is None else int(stop)
        return np.arange(start, stop) / self.sample_rate

    # ------------------------------------------------------------------------------------- other streams

    def _series(self, path):
        g = self._nc[path]
        out = {name: np.array(var[:], dtype=float) for name, var in g.variables.items()}
        out.update({f'@{k}': g.getncattr(k) for k in g.ncattrs()})
        return out

    def vehicle(self, topic='position'):
        """One PX4 topic as a dict of arrays with its own ``time`` (s after time_origin); attributes as '@name'."""
        return self._series(f'vehicle/{topic}')

    def rotors(self):
        """{motor number: dict(time, speed (rad/s), angle (rad) if logged)}, or {} without onboard rotor data."""
        if 'rotors' not in self._nc.groups:
            return {}
        return {int(name[5:]): self._series(f'rotors/{name}') for name in self._nc['rotors'].groups}

    def weather(self):
        """{group name: dict of arrays with its own time}; group attributes (location, instrument, clock) as '@name'."""
        if 'weather' not in self._nc.groups:
            return {}
        return {name: self._series(f'weather/{name}') for name in self._nc['weather'].groups}

    def track(self, units='m'):
        """The vehicle's position as a track dict (``time`` s; ``x`` east, ``y`` north, ``z`` up) in m or ft, the
        form :func:`panam.order_tracking.doppler_factor` takes."""
        p = self.vehicle('position')
        scale = {'m': 1.0, 'ft': M_TO_FT}[units]
        t, keep = np.unique(p['time'], return_index=True)
        return {'time': t, 'x': p['east'][keep] * scale, 'y': p['north'][keep] * scale, 'z': p['up'][keep] * scale}

    def receiver(self, mic, units='m'):
        """A microphone's (3,) position (east, north, up) in m or ft."""
        i = self.mic_index(mic)
        a = self._nc['acoustic']
        scale = {'m': 1.0, 'ft': M_TO_FT}[units]
        return np.array([a['mic_east'][i], a['mic_north'][i], a['mic_up'][i]], dtype=float) * scale


def open_run(path):
    """Open a run file: ``with open_run(p) as run: p = run.pressure(['M8'])``."""
    return RunFile(path)


def list_runs(root):
    """Every run file under a dataset root (``runs/<date>/<recording>.nc``), sorted."""
    out = []
    for dirpath, _, files in os.walk(os.path.join(os.fspath(root), 'runs')):
        out += [os.path.join(dirpath, f) for f in files if f.endswith('.nc') and not f.startswith('._')]
    return sorted(out)
