"""Coherence of a rotorcraft's sound between emission directions, for NICEOPS's ray sums.

A sphere database holds levels, not phases. Where NICEOPS adds rays that leave the
source in different directions (its point-source urban model, ``niceops --buildings``), it
weights each pair's cross term by how coherent the source is between their two launch
directions. For two directions theta degrees apart, in a band at f Hz:

- the mean coherency is g = exp(-(f theta / scale)^exponent), times, in bands centered up to
  ``rotation_limit``, the delay a pattern turning with the main rotor puts between them:
  sense x (their azimuth difference about the shaft, wrapped to +-pi) / Omega;
- the coherence is gamma^2 = T + (1 - T) exp(-(f theta / spread_scale)^spread_exponent),
  T the band's coherence between distant directions (its tonal fraction);
- the spread about the mean, gamma^2 - g^2, sets NICEOPS's phase interval.

This module measures that from a flight test (:func:`band_coherency`, on microphone
signals already de-Dopplerized onto emission time with :func:`panam.geometry.dedopplerize`),
fits the parameters (:func:`fit_coherence`), and reads and writes them as a database's root
attributes (:func:`write_coherence_record`, :func:`read_coherence_record`;
``docs/file_formats.md`` has the names). NICEOPS takes a database that records any of them
with this model, filling what it leaves out from :data:`DEFAULTS`, the fits to the 2017
Bell 407 passes and hovers (niceops-validation, ``coherence/``). A database without them
assumes nothing of the source's phase: NICEOPS adds its rays in energy, except those leaving
within a small cone of each other. The rotor's delay comes only from a record.

The band coherency is the cross power over the band divided by the root of the two auto
powers, not a coherence per frequency bin: a ray sum adds band energies, so that is what it
needs, and it weights a band's tones and broadband by their power.
"""
from dataclasses import asdict, dataclass, field
import json
import math

import numpy as np
from scipy.optimize import minimize

#: What NICEOPS fills a record's gaps with (its source_coherence.hpp): the mean coherency's
#: decay with the rotor's delay removed, and without it; the coherence's decay; the tonal
#: fraction's law 1 / (1 + (f / corner)^power); and how many blade-passage frequencies the
#: delay holds to.
DEFAULTS = dict(scale_hz_deg=850.0, exponent=1.6, scale_without_rotation_hz_deg=475.0,
                exponent_without_rotation=2.3, spread_scale_hz_deg=2120.0, spread_exponent=1.0,
                tonal_corner_hz=58.0, tonal_power=1.67, rotation_harmonics=4.5)

#: What the root attribute ``main_rotor_rotation`` holds, seen from above, and the sense
#: NICEOPS takes from it.
ROTATIONS = {'counterclockwise': 1, 'clockwise': -1}


@dataclass
class CoherenceRecord:
    """What a database records of its vehicle's coherence between emission directions.

    Every field is optional (None, or empty). ``blades`` and ``rotation``
    ('counterclockwise' or 'clockwise', seen from above) give NICEOPS the main rotor's
    delay, at the database's rotor speed; ``rotation_limit_hz`` the band center up to which
    it holds. The fits are the ones the record's own rotor implies: with the delay removed
    where it gives the blades and the rotation, without it otherwise; NICEOPS applies the
    rotor's delay from a record only. ``tonal_fraction`` is T per
    band at ``tonal_fraction_bands`` (Hz); the writer interpolates it onto the database's
    own bands. ``source`` is free text saying what the fits were made from.
    """
    blades: float = None
    rotation: str = None
    rotation_limit_hz: float = None
    scale_hz_deg: float = None
    exponent: float = None
    spread_scale_hz_deg: float = None
    spread_exponent: float = None
    tonal_fraction: np.ndarray = field(default_factory=lambda: np.array([]))
    tonal_fraction_bands: np.ndarray = field(default_factory=lambda: np.array([]))
    source: str = ''


def _check(record):
    for name in ('blades', 'rotation_limit_hz', 'scale_hz_deg', 'exponent', 'spread_scale_hz_deg',
                 'spread_exponent'):
        value = getattr(record, name)
        if value is not None and not (math.isfinite(value) and value > 0):
            raise ValueError(f'{name} must be positive, not {value!r}')
    if record.rotation is not None and record.rotation not in ROTATIONS:
        raise ValueError(f'rotation must be one of {tuple(ROTATIONS)}, not {record.rotation!r}')
    for a, b in (('scale_hz_deg', 'exponent'), ('spread_scale_hz_deg', 'spread_exponent')):
        if (getattr(record, a) is None) != (getattr(record, b) is None):
            raise ValueError(f'{a} and {b} come together')
    fraction = np.asarray(record.tonal_fraction, dtype=float)
    bands = np.asarray(record.tonal_fraction_bands, dtype=float)
    if fraction.shape != bands.shape:
        raise ValueError('tonal_fraction needs one band center per value in tonal_fraction_bands')
    if fraction.size and not np.all((fraction >= 0) & (fraction <= 1)):
        raise ValueError('tonal_fraction is a fraction, within [0, 1]')
    if bands.size and not (np.all(bands > 0) and np.all(np.diff(bands) > 0)):
        raise ValueError('tonal_fraction_bands must be positive and ascending')


def write_coherence_record(ncdatabase, record, band_centers=None):
    """Write ``record`` as root attributes of an open netCDF4 Dataset.

    The tonal fraction is interpolated in log frequency onto ``band_centers`` (the
    database's bands, Hz), held at its ends; without them (a database that stores no
    spectra) it is left out, since NICEOPS reads it one per band.
    """
    _check(record)
    for name, attribute in (('blades', 'main_rotor_blades'), ('rotation_limit_hz', 'coherence_rotation_limit_Hz'),
                            ('scale_hz_deg', 'coherence_scale_Hz_deg'), ('exponent', 'coherence_exponent'),
                            ('spread_scale_hz_deg', 'coherence_spread_scale_Hz_deg'),
                            ('spread_exponent', 'coherence_spread_exponent')):
        value = getattr(record, name)
        if value is not None:
            ncdatabase.setncattr(attribute, float(value))
    if record.rotation is not None:
        ncdatabase.setncattr('main_rotor_rotation', record.rotation)
    fraction = np.asarray(record.tonal_fraction, dtype=float)
    if fraction.size and band_centers is not None and np.size(band_centers):
        at = np.interp(np.log(np.asarray(band_centers, dtype=float)),
                       np.log(np.asarray(record.tonal_fraction_bands, dtype=float)), fraction)
        ncdatabase.setncattr('coherence_tonal_fraction', np.clip(at, 0.0, 1.0).astype(np.float64))
    if record.source:
        ncdatabase.setncattr('coherence_source', record.source)


def read_coherence_record(ncdatabase, band_centers=None):
    """The record an open netCDF4 Dataset carries at its root (empty fields where it has none).

    ``band_centers``, when given, labels the tonal fraction (which the file stores one per
    database band); otherwise ``tonal_fraction_bands`` is the index of each value.
    """
    attributes = set(ncdatabase.ncattrs())

    def number(name):
        return float(ncdatabase.getncattr(name)) if name in attributes else None

    fraction = (np.asarray(ncdatabase.getncattr('coherence_tonal_fraction'), dtype=float).ravel()
                if 'coherence_tonal_fraction' in attributes else np.array([]))
    bands = (np.asarray(band_centers, dtype=float) if band_centers is not None
             else np.arange(1.0, fraction.size + 1.0))
    return CoherenceRecord(
        blades=number('main_rotor_blades'),
        rotation=str(ncdatabase.getncattr('main_rotor_rotation')) if 'main_rotor_rotation' in attributes else None,
        rotation_limit_hz=number('coherence_rotation_limit_Hz'), scale_hz_deg=number('coherence_scale_Hz_deg'),
        exponent=number('coherence_exponent'), spread_scale_hz_deg=number('coherence_spread_scale_Hz_deg'),
        spread_exponent=number('coherence_spread_exponent'), tonal_fraction=fraction,
        tonal_fraction_bands=bands[:fraction.size] if fraction.size else np.array([]),
        source=str(ncdatabase.getncattr('coherence_source')) if 'coherence_source' in attributes else '')


def save_coherence_record(record, path):
    """Write ``record`` to ``path`` as JSON: what ``build_empirical_database.py --coherence``
    reads, so a fit made once can be built into several databases."""
    _check(record)
    fields = {name: (value.tolist() if isinstance(value, np.ndarray) else value)
              for name, value in asdict(record).items()}
    with open(path, 'w') as handle:
        json.dump(fields, handle, indent=1)


def load_coherence_record(path):
    """The :class:`CoherenceRecord` :func:`save_coherence_record` wrote (or any JSON object of
    its fields), checked as the writer checks it."""
    with open(path) as handle:
        fields = json.load(handle)
    for name in ('tonal_fraction', 'tonal_fraction_bands'):
        if name in fields:
            fields[name] = np.asarray(fields[name], dtype=float)
    record = CoherenceRecord(**fields)
    _check(record)
    return record


def flat_delay(tau, f1, f2):
    """The average of exp(-2 pi i f tau) over f in [f1, f2]: a delay's coherency over a band
    of flat spectrum, as NICEOPS's sub-bands average it."""
    tau = np.asarray(tau, dtype=float)
    x = -2j * np.pi * tau
    small = np.abs(tau) * (f2 - f1) < 1e-9
    safe = np.where(small, 1.0, x)
    value = (np.exp(safe * f2) - np.exp(safe * f1)) / (safe * (f2 - f1))
    return np.where(small, np.exp(x * 0.5 * (f1 + f2)), value)


def _segments(signals, fs, nseg):
    hop = nseg // 2
    count = (signals.shape[1] - nseg) // hop + 1
    index = np.arange(count)[:, None] * hop + np.arange(nseg)[None, :]
    frames = signals[:, index]
    finite = np.all(np.isfinite(frames), axis=2)
    window = np.hanning(nseg + 1)[:-1]
    spectra = np.fft.rfft(np.where(np.isfinite(frames), frames, 0.0) * window, axis=2)
    return spectra, np.arange(count) * hop + nseg // 2, finite


def band_coherency(signals, emission_time, fs, track_time, track_position, microphones, bands, window=1.0,
                   max_range=np.inf, rotation=None, shaft=(0.0, 0.0, 1.0)):
    """Every pair of microphones' band coherency, window by window.

    Args:
        signals: (microphones, samples) pressures de-Dopplerized onto ``emission_time``
            (:func:`panam.geometry.dedopplerize`), NaN where a microphone heard nothing.
        emission_time: (samples,) s; fs: its sample rate, Hz.
        track_time, track_position: the source's track, s and (n, 3) in the microphones'
            frame (z up).
        microphones: (microphones, 3) positions.
        bands: third-octave band centers, Hz.
        window: s of emission time per estimate; max_range: microphones farther from the
            source at a window's middle are left out of it (the frame's length unit).
        rotation: (Omega rad/s, sense), sense +1 counterclockwise seen from above along
            ``shaft``; None for no rotor's delay.
        shaft: the rotor's shaft, up (the tip-path plane's normal), in the frame.

    Each band is cut into Hann segments of 16 / f s (between 1/16 and 1/2 s) at half
    overlap, so it holds a few bins and a segment's geometry stays nearly fixed while the
    source moves; the geometry is taken at each segment's middle. Per window and pair:

    - ``rho``: sum over segments and the band's bins of conj(X_a) X_b, over the root of
      the two auto powers so summed (scipy's csd convention);
    - ``rho_turned``: the same with each bin's cross term times exp(+2 pi i f tau), tau the
      rotor's delay of b behind a at that segment (= ``rho`` without a rotation);
    - ``delay``: the delay's prediction, the segments' sum of sqrt(P_aa P_bb) D(tau) over
      the same root, D = :func:`flat_delay` over the band; ``coherent``: the same with D = 1;
    - ``separation``: the angle between the two emission directions, deg, and ``dpsi``,
      their azimuth difference about the shaft, rad, both weighted by sqrt(P_aa P_bb);
    - ``n_eff``: segments times bins, for the estimate's noise (about 1 / n_eff in |rho|^2).

    Returns a dict of those arrays and ``band`` (index into ``bands``).
    """
    signals = np.asarray(signals, dtype=float)
    track_position = np.asarray(track_position, dtype=float)
    microphones = np.asarray(microphones, dtype=float)
    up = np.asarray(shaft, dtype=float) / np.linalg.norm(shaft)
    # A basis in the plane of the rotor: east-ish first, then up x it (counterclockwise from above).
    seed = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = seed - seed.dot(up) * up
    first /= np.linalg.norm(first)
    second = np.cross(up, first)
    out = {name: [] for name in ('band', 'separation', 'dpsi', 'n_eff', 'rho', 'rho_turned', 'delay', 'coherent')}
    t0 = emission_time[0]
    for b, fc in enumerate(bands):
        f1, f2 = fc / 2 ** (1 / 6), fc * 2 ** (1 / 6)
        nseg = 2 * int(round(0.5 * fs * np.clip(16.0 / fc, 1 / 16, 0.5)))
        if nseg >= signals.shape[1]:
            continue
        spectra, centers, finite = _segments(signals, fs, nseg)
        freqs = np.fft.rfftfreq(nseg, 1.0 / fs)
        inband = (freqs >= f1) & (freqs < f2)
        if not inband.any():
            continue
        X, fb = spectra[:, :, inband], freqs[inband]
        times = emission_time[centers]
        source = np.column_stack([np.interp(times, track_time, track_position[:, k]) for k in range(3)])
        d = microphones[:, None, :] - source[None, :, :]
        distance = np.linalg.norm(d, axis=2)
        u = d / distance[:, :, None]
        psi = np.arctan2(u @ second, u @ first)
        auto = np.sum(np.abs(X) ** 2, axis=2)
        index = np.floor((times - t0) / window).astype(int)
        for w in np.unique(index):
            segs = np.flatnonzero(index == w)
            if len(segs) < 2:
                continue
            middle = segs[len(segs) // 2]
            keep = np.flatnonzero(np.all(finite[:, segs], axis=1) & (distance[:, middle] < max_range))
            if len(keep) < 2:
                continue
            a, c = np.triu_indices(len(keep), 1)
            a, c = keep[a], keep[c]
            cross = np.conj(X[a][:, segs, :]) * X[c][:, segs, :]
            pa, pc = auto[a][:, segs], auto[c][:, segs]
            norm = np.sqrt(pa.sum(axis=1) * pc.sum(axis=1))
            scale = np.sqrt(pa * pc)
            weight = scale / np.maximum(scale.sum(axis=1, keepdims=True), 1e-300)
            separation = np.degrees(np.arccos(np.clip(np.sum(u[a][:, segs] * u[c][:, segs], axis=2), -1, 1)))
            dpsi = np.angle(np.exp(1j * (psi[c][:, segs] - psi[a][:, segs])))
            if rotation is not None:
                omega, sense = rotation
                tau = sense * dpsi / omega
                turned = cross * np.exp(2j * np.pi * fb[None, None, :] * tau[:, :, None])
                delay = np.sum(scale * flat_delay(tau, f1, f2), axis=1) / norm
            else:
                turned = cross
                delay = scale.sum(axis=1) / norm
            out['band'].append(np.full(len(a), b))
            out['separation'].append(np.sum(weight * separation, axis=1))
            out['dpsi'].append(np.sum(weight * dpsi, axis=1))
            out['n_eff'].append(np.full(len(a), len(segs) * len(fb)))
            out['rho'].append(cross.sum(axis=(1, 2)) / norm)
            out['rho_turned'].append(turned.sum(axis=(1, 2)) / norm)
            out['delay'].append(delay)
            out['coherent'].append(scale.sum(axis=1) / norm)
    return {name: np.concatenate(values) if values else np.array([]) for name, values in out.items()}


def _decay(x, scale, exponent):
    return np.exp(-(np.maximum(x, 0.0) / scale) ** exponent)


def fit_coherence(records, bands, rotation_rate=None, blades=None, rotation=None,
                  rotation_harmonics=DEFAULTS['rotation_harmonics'], far=40.0, source=''):
    """The model's parameters fitted to :func:`band_coherency`'s records.

    ``records`` is one set of records, or a list of sets (cases: a vehicle's passes and its
    hovers, say), each weighted equally however many pair-windows it holds; all made with
    the same ``bands`` and rotation. :data:`DEFAULTS` is this fit to the Bell 407's
    passes and hovers as two cases.

    With ``rotation_rate`` (rad/s), ``blades`` and ``rotation``, the records must have been
    made with that rotation; the delay is then applied in the bands centered up to
    ``rotation_harmonics`` blade-passage frequencies, and the record carries the rotor.
    Without them the fit is the one without the delay.

    - ``scale_hz_deg``, ``exponent``: least squares of rho - g(f theta) x (delay or zero
      phase), f the band's center: each case's mean over its pair-windows, averaged;
    - ``tonal_fraction``: per band and case, the mean of |rho|^2 - 1 / n_eff over the pairs
      at least ``far`` degrees apart, held to [0, 1], averaged over the cases with 20 such
      pairs or more; a band none has takes its neighbors' by interpolation (or NICEOPS's
      default law where no band has any);
    - ``spread_scale_hz_deg``, ``spread_exponent``: least squares of T + (1 - T)
      exp(-(f theta / scale)^exponent) against the bins' mean |rho|^2 - 1 / n_eff, bins of
      case, band and separation (16 per band, geometric from 1 deg; 30 pair-windows or
      more) weighted equally, each with its case's T, so the few small separations count as
      much as the many large ones.

    Returns a :class:`CoherenceRecord`.
    """
    cases = list(records) if isinstance(records, (list, tuple)) else [records]
    bands = np.asarray(bands, dtype=float)
    rotates = rotation_rate is not None and blades is not None and rotation is not None
    limit = rotation_harmonics * blades * rotation_rate / (2 * np.pi) if rotates else None
    prepared = []
    for case in cases:
        band = np.asarray(case['band'], dtype=int)
        f = bands[band]
        separation = np.asarray(case['separation'], dtype=float)
        rho = np.asarray(case['rho'])
        base = (np.where(f <= limit, case['delay'], case['coherent']) if rotates
                else np.asarray(case['coherent']))
        power = np.abs(rho) ** 2 - 1.0 / np.asarray(case['n_eff'], dtype=float)
        prepared.append((band, f * separation, separation, rho, base, power))

    def mean_cost(v):
        scale, exponent = np.exp(v)
        return np.mean([np.mean(np.abs(rho - _decay(x, scale, exponent) * base) ** 2)
                        for _, x, _, rho, base, _ in prepared])

    start = (DEFAULTS['scale_hz_deg'], DEFAULTS['exponent']) if rotates else (
        DEFAULTS['scale_without_rotation_hz_deg'], DEFAULTS['exponent_without_rotation'])
    scale, exponent = np.exp(minimize(mean_cost, np.log(start), method='Nelder-Mead',
                                      options=dict(xatol=1e-4, fatol=1e-10)).x)

    def tonal_of(band, separation, power):
        tonal = np.full(len(bands), np.nan)
        for b in range(len(bands)):
            m = (band == b) & (separation >= far)
            if m.sum() >= 20:
                tonal[b] = np.clip(np.mean(power[m]), 0.0, 1.0)
        return tonal

    per_case = [tonal_of(band, separation, power) for band, _, separation, _, _, power in prepared]
    stacked = np.array(per_case)
    counted = np.isfinite(stacked).sum(axis=0)
    tonal = np.where(counted > 0, np.nansum(stacked, axis=0) / np.maximum(counted, 1), np.nan)
    known = np.isfinite(tonal)
    if known.any():
        tonal[~known] = np.interp(np.log(bands[~known]), np.log(bands[known]), tonal[known])
    else:
        tonal = 1.0 / (1.0 + (bands / DEFAULTS['tonal_corner_hz']) ** DEFAULTS['tonal_power'])

    edges = np.concatenate([[0.0], np.geomspace(1.0, 180.0, 16)])
    bins = []
    for (band, x, separation, _, _, power), own in zip(prepared, per_case):
        own = np.where(np.isfinite(own), own, tonal)
        which = np.searchsorted(edges, separation)
        for b in range(len(bands)):
            for k in np.unique(which[band == b]):
                m = (band == b) & (which == k)
                if m.sum() >= 30:
                    bins.append((x[m].mean(), own[b], np.mean(power[m])))
    bins = np.array(bins)

    def spread_cost(v):
        s, p = np.exp(v)
        return np.mean((bins[:, 1] + (1 - bins[:, 1]) * _decay(bins[:, 0], s, p) - bins[:, 2]) ** 2)

    spread_scale, spread_exponent = (np.exp(minimize(spread_cost, np.log([2000.0, 1.0]),
                                                     method='Nelder-Mead').x)
                                     if len(bins) >= 2 else (DEFAULTS['spread_scale_hz_deg'],
                                                             DEFAULTS['spread_exponent']))
    return CoherenceRecord(
        blades=float(blades) if rotates else None, rotation=rotation if rotates else None,
        rotation_limit_hz=float(limit) if rotates else None, scale_hz_deg=float(scale), exponent=float(exponent),
        spread_scale_hz_deg=float(spread_scale), spread_exponent=float(spread_exponent),
        tonal_fraction=tonal, tonal_fraction_bands=bands.copy(), source=source)
