"""Impulsiveness metrics for blade-vortex interaction (BVI) noise.

Three frame-wise waveform statistics of a pressure time history and one
spectral ratio:

* ``crest_factor``: peak absolute pressure over the frame RMS.
* ``kurtosis``: the fourth central moment over the squared variance
  (Pearson's kurtosis, not the excess): 1.5 for a sine, 3 for Gaussian noise,
  large for a sparse train of pulses.
* ``pulse_peak_to_rms``: the peak of the pressure band-passed to the pulse
  band ``band_hz`` (for BVI, the 6th to 40th blade-passage harmonics,
  ``(5.5 * bpf, 40.5 * bpf)``) over the RMS of the full-band frame.  It measures how large the impulsive part of the
  waveform is against everything else the microphone hears.
* ``bvi_harmonic_ratio``: the energy in mid-frequency blade-passage harmonics
  over the energy in the low harmonics, in dB, from a narrowband PSD such as
  ``depropagate_hemisphere(..., narrowband=True)['narrowband']``.  That
  spectrum is in received frequency, so the blade-passage frequency passed
  for each direction is the received one, the Doppler factor times the
  source's.

The waveform statistics take a microphone pressure history in Pa (any
consistent pressure unit; the results are ratios) at ``sampling_rate`` Hz,
with frames ``frame_time`` seconds long.  The history may first be
de-Dopplerized with ``flight_acoustics.dedopplerize``, whose output is NaN
where the emission would be received outside the record; a frame holding
any NaN sample gives NaN.  Each frame is de-trended (``detrend='constant'``
removes its mean, ``'linear'`` a straight line, ``None`` nothing) before the
statistic is taken.

These statistics are not gridded onto the spheres, and PANAM has no
sphere-file or database field for them.  The sphere is a field of energy:
depropagation scales a band's power by spreading and absorption, and
gridding averages power across nearby emission directions.  Crest factor
and kurtosis are neither.  They depend on the relative phase of the
harmonics, which a power spectrum discards, so they cannot be rebuilt from
the sphere's spectra; an average or interpolation of them across
microphones and emission times has no physical meaning, since the
statistic of a sum or a mix of waveforms is not the mean of their
statistics; and propagation changes them in a way a level correction does
not undo, because absorption removes the high harmonics that make a pulse
sharp, and ambient noise pulls kurtosis toward 3 and crest factor toward
the Gaussian value as range grows.  Evaluate them on the microphone
records, and use ``bvi_harmonic_ratio``, an energy ratio, on the
narrowband sphere.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, detrend as _detrend, sosfiltfilt

__all__ = [
    "frame_signal",
    "crest_factor",
    "kurtosis",
    "pulse_peak_to_rms",
    "bvi_harmonic_ratio",
]


def frame_signal(pressure, sampling_rate, frame_time, hop_time=None):
    """Cut a pressure history into frames along its last axis.

    Args:
        pressure: (..., Nt) pressure, any unit (Pa in PANAM).
        sampling_rate: samples per second, Hz.
        frame_time: frame length, s; rounded to a whole number of samples
            (at least 2).
        hop_time: start-to-start spacing of the frames, s; ``None`` (the
            default) gives consecutive, non-overlapping frames.  A partial
            last frame is dropped.

    Returns:
        tuple ``(times, frames)``: ``times`` (Nframes,) the frame centers in
        s from the first sample, and ``frames`` (..., Nframes, Nsamples) in
        the unit of ``pressure``.
    """
    pressure = np.asarray(pressure, dtype=float)
    n = int(round(float(frame_time) * float(sampling_rate)))
    if n < 2:
        raise ValueError("frame_time must span at least 2 samples")
    hop = n if hop_time is None else int(round(float(hop_time) * float(sampling_rate)))
    if hop < 1:
        raise ValueError("hop_time must span at least 1 sample")
    if pressure.shape[-1] < n:
        raise ValueError("the record is shorter than one frame")
    starts = np.arange(0, pressure.shape[-1] - n + 1, hop)
    frames = pressure[..., starts[:, None] + np.arange(n)[None, :]]
    times = (starts + 0.5 * (n - 1)) / float(sampling_rate)
    return times, frames


def _prepared_frames(pressure, sampling_rate, frame_time, hop_time, detrend):
    times, frames = frame_signal(pressure, sampling_rate, frame_time, hop_time)
    missing = np.isnan(frames).any(axis=-1)
    if detrend is not None:
        if detrend not in ("constant", "linear"):
            raise ValueError("detrend must be 'constant', 'linear' or None, not {!r}".format(detrend))
        frames = _detrend(np.where(np.isnan(frames), 0.0, frames), axis=-1, type=detrend)
    return times, frames, missing


def _rms(frames):
    return np.sqrt(np.mean(frames ** 2, axis=-1))


def crest_factor(pressure, sampling_rate, frame_time, *, hop_time=None, detrend="constant"):
    """Frame-wise crest factor, peak absolute pressure over RMS.

    A sine gives sqrt(2); an impulse train of one pulse per period ``T``
    samples gives sqrt(T).

    Args:
        pressure: (..., Nt) pressure history, Pa (any unit), optionally the
            output of ``flight_acoustics.dedopplerize``.
        sampling_rate: Hz.
        frame_time: frame length, s.
        hop_time: frame spacing, s; default ``None``, non-overlapping frames.
        detrend: ``'constant'`` (default), ``'linear'`` or ``None``, applied
            to each frame first.

    Returns:
        tuple ``(times, crest)``: frame centers, s, and the dimensionless
        crest factor (..., Nframes).  It is NaN for a frame with a NaN
        sample or with zero RMS.
    """
    times, frames, missing = _prepared_frames(pressure, sampling_rate, frame_time, hop_time, detrend)
    with np.errstate(divide="ignore", invalid="ignore"):
        crest = np.max(np.abs(frames), axis=-1) / _rms(frames)
    crest = np.where(missing | ~np.isfinite(crest), np.nan, crest)
    return times, crest


def kurtosis(pressure, sampling_rate, frame_time, *, hop_time=None, detrend="constant"):
    """Frame-wise kurtosis, E[(p - mean)^4] / E[(p - mean)^2]^2.

    This is Pearson's kurtosis, not the excess: 1.5 for a sine, 3 for
    Gaussian noise, about ``T`` for one pulse per ``T`` samples.  The mean
    is removed whatever ``detrend`` is, so ``detrend`` matters only as
    ``'linear'``.

    Args:
        pressure: (..., Nt) pressure history, Pa (any unit), optionally the
            output of ``flight_acoustics.dedopplerize``.
        sampling_rate: Hz.
        frame_time: frame length, s.
        hop_time: frame spacing, s; default ``None``, non-overlapping frames.
        detrend: ``'constant'`` (default), ``'linear'`` or ``None``.

    Returns:
        tuple ``(times, kurt)``: frame centers, s, and the dimensionless
        kurtosis (..., Nframes).  It is NaN for a frame with a NaN sample
        or with zero variance.
    """
    times, frames, missing = _prepared_frames(pressure, sampling_rate, frame_time, hop_time, detrend)
    frames = frames - frames.mean(axis=-1, keepdims=True)
    m2 = np.mean(frames ** 2, axis=-1)
    m4 = np.mean(frames ** 4, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        kurt = m4 / m2 ** 2
    kurt = np.where(missing | ~np.isfinite(kurt), np.nan, kurt)
    return times, kurt


def pulse_peak_to_rms(pressure, sampling_rate, frame_time, *, band_hz, hop_time=None,
                      detrend="constant", order=4):
    """Frame-wise peak of the pulse band over the full-band RMS.

    The record is band-passed to ``band_hz`` with a zero-phase Butterworth
    filter (``order`` per pass, applied forward and backward), and each
    frame's peak absolute band-passed pressure is divided by the RMS of
    the same frame of the de-trended, unfiltered pressure.  For BVI take
    the band as the 6th to 40th blade-passage harmonics,
    ``(5.5 * bpf, 40.5 * bpf)``.  A sine outside the band gives about 0; a
    pressure history that is all pulse band gives its crest factor.

    The filter's start-up transient, a few periods of the lower band edge,
    reaches into the first and last frames of the record.  NaN samples, as
    ``flight_acoustics.dedopplerize`` returns outside the record, are
    filtered as zeros and their frames reported as NaN; the transient at
    each NaN boundary reaches into the neighboring frames in the same way.

    Args:
        pressure: (..., Nt) pressure history, Pa (any unit).
        sampling_rate: Hz.
        frame_time: frame length, s.
        band_hz: ``(f_low, f_high)`` pulse band, Hz; ``f_high`` must be
            below the Nyquist frequency.
        hop_time: frame spacing, s; default ``None``, non-overlapping frames.
        detrend: ``'constant'`` (default), ``'linear'`` or ``None``, applied
            to the full-band frames.
        order: Butterworth order of each pass; default 4.

    Returns:
        tuple ``(times, ratio)``: frame centers, s, and the dimensionless
        ratio (..., Nframes), NaN for a frame with a NaN sample or zero RMS.
    """
    f_low, f_high = (float(f) for f in band_hz)
    if not 0.0 < f_low < f_high < 0.5 * float(sampling_rate):
        raise ValueError("band_hz must satisfy 0 < f_low < f_high < sampling_rate / 2")
    pressure = np.asarray(pressure, dtype=float)
    sos = butter(int(order), [f_low, f_high], btype="bandpass", fs=float(sampling_rate), output="sos")
    pulse = sosfiltfilt(sos, np.where(np.isnan(pressure), 0.0, pressure), axis=-1)
    times, frames, missing = _prepared_frames(pressure, sampling_rate, frame_time, hop_time, detrend)
    _, pulse_frames = frame_signal(pulse, sampling_rate, frame_time, hop_time)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.max(np.abs(pulse_frames), axis=-1) / _rms(frames)
    ratio = np.where(missing | ~np.isfinite(ratio), np.nan, ratio)
    return times, ratio


def _harmonic_band_energy(power, freqs, width, lower, upper):
    """Sum of power * bin width over bins with lower <= f < upper, per direction.

    ``lower`` and ``upper`` have the shape of ``power.shape[1:]``.  A NaN bin
    in the band gives NaN; NaN bins outside it are ignored.
    """
    f = freqs.reshape((-1,) + (1,) * (power.ndim - 1))
    inside = (f >= lower[None]) & (f < upper[None])
    if not inside.any(axis=0).all():
        raise ValueError("no frequency bin falls in a harmonic band "
                         "[{:g}, {:g}) Hz".format(float(np.min(lower)), float(np.max(upper))))
    w = width.reshape(f.shape)
    with np.errstate(invalid="ignore"):
        return np.sum(np.where(inside, power * w, 0.0), axis=0)


def bvi_harmonic_ratio(spectrum, freqs, bpf, low_harmonics=(1, 5), high_harmonics=(6, 40)):
    """Energy in mid-frequency blade-passage harmonics over the low harmonics, dB.

    Harmonic ``k`` owns the band ``[(k - 1/2) bpf, (k + 1/2) bpf)``, so a
    harmonic range ``(k1, k2)`` is the band ``[(k1 - 1/2) bpf,
    (k2 + 1/2) bpf)``, its broadband content included, as a band-limited
    BVI SPL is.  The energy is the sum of PSD power times bin width over
    the bins whose centers lie in the band, and the result is

        10 lg(E_high / E_low).

    It is an energy ratio, so it can be evaluated on gridded narrowband
    sphere output, unlike the waveform statistics of this module.

    The narrowband sphere of ``depropagate_hemisphere`` is in received
    frequency: ``remove_doppler`` refiles only the third-octave bands.  A
    harmonic emitted at ``k * bpf`` arrives at ``D * k * bpf``, with the
    Doppler factor ``D = 1 / (1 - M_r)``, about 1.14 ahead of and 0.88
    behind an aircraft at 80 kt.  The band edges must move with it, or ahead
    of the aircraft the 5th harmonic lands in the BVI band and behind it the
    6th lands in the low band.  Pass ``bpf`` as an array of the received
    blade-passage frequency per direction, ``D`` times the source value; a
    scalar ``bpf`` is valid only where ``D`` is near 1, as in hover or
    overhead.

    Args:
        spectrum: (Nf, ...) PSD in dB re any reference (the reference
            cancels), such as ``hemisphere['narrowband']['psd_db']``
            (Nf, Nelv, Nazi).  ``-inf`` is a bin with no energy; NaN is no
            data.
        freqs: (Nf,) ascending bin center frequencies, Hz.
        bpf: received blade-passage frequency, Hz (main-rotor blades times
            rotor rotation frequency, times the Doppler factor): a scalar,
            or an array that broadcasts against ``spectrum.shape[1:]``,
            one value per direction.
        low_harmonics: ``(k1, k2)`` inclusive range of low harmonics;
            default ``(1, 5)``.
        high_harmonics: ``(k1, k2)`` inclusive range of BVI harmonics;
            default ``(6, 40)``.

    Returns:
        ndarray (...) of the ratio in dB, of shape ``spectrum.shape[1:]``:
        NaN where either band holds a NaN bin or both have no energy,
        ``-inf`` where only the high band has no energy, ``+inf`` where only
        the low band has none.

    Raises:
        ValueError: when a band reaches outside ``freqs``, or contains no bin,
            in any direction.
    """
    freqs = np.asarray(freqs, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    if freqs.ndim != 1 or freqs.size < 2 or np.any(np.diff(freqs) <= 0.0):
        raise ValueError("freqs must be a 1-D ascending array of at least 2 frequencies")
    if spectrum.shape[0] != freqs.size:
        raise ValueError("spectrum's first axis must match freqs")
    try:
        bpf = np.broadcast_to(np.asarray(bpf, dtype=float), spectrum.shape[1:])
    except ValueError:
        raise ValueError("bpf must broadcast against spectrum.shape[1:] = {}".format(
            spectrum.shape[1:])) from None
    if not np.all(bpf > 0.0):
        raise ValueError("bpf must be positive")
    width = np.gradient(freqs)
    edge0, edge1 = freqs[0] - 0.5 * width[0], freqs[-1] + 0.5 * width[-1]
    with np.errstate(over="ignore"):
        power = 10.0 ** (spectrum / 10.0)
    energies = []
    for k1, k2 in (low_harmonics, high_harmonics):
        if not 1 <= int(k1) <= int(k2):
            raise ValueError("harmonic ranges must satisfy 1 <= k1 <= k2")
        lower, upper = (int(k1) - 0.5) * bpf, (int(k2) + 0.5) * bpf
        if np.min(lower) < edge0 or np.max(upper) > edge1:
            raise ValueError("harmonics {}-{} ({:g}-{:g} Hz) reach outside the spectrum "
                             "({:g}-{:g} Hz)".format(k1, k2, float(np.min(lower)),
                                                     float(np.max(upper)), edge0, edge1))
        energies.append(_harmonic_band_energy(power, freqs, width, lower, upper))
    low, high = energies
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(high / low)
