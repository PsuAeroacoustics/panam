"""Spectra and levels: PSDs, one-third octave bands, spectrograms, A-weighting and filters.

Pressures are in Pa and levels in dB re :data:`P_REF` (20 uPa); frequencies in Hz.
:func:`tone_aware_band_power` files each tone whole in the band of its own
frequency (``docs/THEORY.md`` section 2).
"""

import os

import numpy as np
from scipy.ndimage import maximum_filter1d
import scipy.signal

from panam.acoustics import filters as pa_filters
from panam.acoustics.plotting import acoustic_plot_style


#: Reference sound pressure, Pa (20 uPa).
P_REF = 2.0e-5


def psd(signal, sampling_rate, cal=0.0, window='hann'):
    """
    Compute the acoustic power spectral density of a signal

    A single periodogram of the whole record, density-scaled (divided by
    ``sampling_rate * sum(w**2)``), so a broadband level and the integral over
    all bands are unbiased whatever the window.  The default Hann window
    keeps a tone's leakage within a few bins (sidelobes falling 18 dB per
    octave); a tone's power is spread over its main lobe, four bins wide.
    ``window='boxcar'`` gives the exact energy of the record (Parseval) but
    leaks an off-bin tone across the whole spectrum with a 1/k**2 skirt.

    Args:
        signal: Array-like acoustic signal
        sampling_rate: Sampling rate of signal, Hz
        cal: Optional calibration factor to apply to signal (dB)
        window: window name or array, see scipy.signal.get_window, default 'hann'
    Returns: tuple (frequency, psd_db, level)
           WHERE
           frequency is an array of band frequencies
           psd_db is the power spectral density in dB re (20 uPa)^2/Hz
           level is the integrated sound pressure level over all bands in dB
    """
    kcal = 10 ** (cal / 20)
    frequency, power_spectral_density = scipy.signal.periodogram(kcal * signal, sampling_rate, window=window)
    df = frequency[1] - frequency[0]
    with np.errstate(divide='ignore'):
        psd_db = 10.0 * np.log10(power_spectral_density / (P_REF ** 2))
        level = 20.0 * np.log10(np.sqrt(np.sum(power_spectral_density * df)) / P_REF)
    return frequency, psd_db, level


def psd_welch(signal, sampling_rate, cal=0.0, window_time=1.0, window_type='hann', window_overlap=0.5, medfilter=None, passband = None):
    """
    Compute the acoustic power spectral density of a signal

    Args:
        signal: Array-like acoustic signal
        sampling_rate: Sampling rate of signal, Hz
        cal: Optional calibration factor to apply to signal (dB)
        window_time: duration of windows, s
        window_type: optional type of window to us, see scipy.signal.window, default="hann"
        window_overlap: optional proportion of overlap for windows, default=0.5
        medfilter: width (in Hz) of the median filter, if None, no filter applied (default)
        passband: pair of values (in Hz) defining the lower and upper regions of the passband.  Default None.
    Returns: tuple (frequency, psd_db, level, level_A)
           WHERE
           frequency is an array of band frequencies
           psd_db is the power spectral density in dB re (20 uPa)^2/Hz
           level is the integrated sound pressure level over all bands in dB
           level_A is the A-weighted integrated sound pressure level over all bands in dB
    """
    kcal = 10 ** (cal / 20)
    binwidth = frame_length(window_time, sampling_rate)
    window = scipy.signal.get_window(window_type, binwidth)
    frequency, power_spectral_density = scipy.signal.welch(kcal * signal, sampling_rate,
                                                           window=window, noverlap=int(binwidth * window_overlap))
    df = frequency[1] - frequency[0]
    if passband is not None:
        if len(passband) != 2 or not np.isfinite(passband).all() or passband[0] > passband[1]:
            raise ValueError('passband must be a finite ordered pair')
        pass_indices = np.logical_and(frequency >= passband[0], frequency <= passband[1])
        power_spectral_density = power_spectral_density[pass_indices]
        frequency = frequency[pass_indices]
        if frequency.size == 0:
            raise ValueError('passband contains no frequency bins')
    if medfilter is not None:
        # medfilter is a width in Hz; medfilt wants an odd number of bins.
        medfilter_width = 2 * int(round(0.5 * float(medfilter) / df)) + 1
        power_spectral_density = scipy.signal.medfilt(power_spectral_density, medfilter_width)
    psd_db = 10.0 * np.log10(power_spectral_density / (P_REF ** 2))
    level = 20.0 * np.log10(np.sqrt(np.sum(power_spectral_density * df)) / P_REF)
    # dBAw is a level (dB), so the weight on a power spectral density is 10**(dB/10).
    weight = 10**(dBAw(frequency) / 10)
    level_A = 20.0 * np.log10(np.sqrt(np.sum(weight * power_spectral_density * df)) / P_REF)
    return frequency, psd_db, level, level_A


def overall_SPL(signal, sampling_rate, window='hann'):
    """
    Computed A-weighted and unweighted sound pressure levels

    The unweighted level is the mean-square pressure of the record (mean
    removed), exact for any signal.  The A-weighted level scales it by the
    A-weighted fraction of the record's windowed spectrum (:func:`psd`), so a
    strong low-frequency tone does not leak into the bands where the
    A-weighting is near 0 dB.

    Args:
        signal: pressure time history signal, Pa
        sampling_rate: sampling rate of signal, Hz
        window: window of the spectrum the A-weighting is applied to, see
            :func:`psd`, default 'hann'

    Returns:
    tuple (A-weighted Level, Unweighted Level)
    """
    signal = np.asarray(signal, dtype=float)
    mean_square = np.mean((signal - np.mean(signal)) ** 2)
    f, spl, _ = psd(signal, sampling_rate, window=window)
    power = 10.0 ** (spl / 10.0)
    total = np.sum(power)
    with np.errstate(divide='ignore'):
        levelZ = 10.0 * np.log10(mean_square / P_REF ** 2)
        if total > 0:
            levelA = levelZ + 10.0 * np.log10(np.sum(10.0 ** (dBAw(f) / 10.0) * power) / total)
        else:
            levelA = -np.inf
    return levelA, levelZ


def level_history(signal, sampling_rate, period=1.0, window='hann'):
    """
    Compute time history of SPL
    Args:
        signal: pressure time history signal, Pa
        sampling_rate: sampling rate of signal, Hz
        period: integration time for SPL calculations, sec
        window: window of each period's spectrum, see :func:`overall_SPL`

    Returns:
        tuple (time, A-weighted level, Unweighted level)
    """
    binwidth = period * sampling_rate
    # One edge per complete period, including the end of the last one.
    edges = np.arange(int(len(signal) // binwidth) + 1) * binwidth

    time = edges[0:-1] / sampling_rate
    level_a = np.zeros_like(time)
    level_z = np.zeros_like(time)
    for i in range(0, len(edges) - 1, 1):
        level_a[i], level_z[i] = overall_SPL(signal[int(edges[i]):int(edges[i + 1])], sampling_rate, window)
    return time, level_a, level_z


def frame_length(window_time, sampling_rate):
    """Samples in an analysis frame of ``window_time`` seconds: the next power of two."""
    return int(2.0 ** nextpow2(window_time * sampling_rate))


def nextpow2(x):
    """
    Compute the smallest integer N for x <= 2**N
    Args:
        x: number or array-like of values for which to compute N

    Returns:
        number or array like of integral N
    """
    return np.ceil(np.log2(np.abs(x)))


# Exact base-2 one-third octave band centers, 1000 * 2**(k/3), 0.01 Hz to 100 kHz
BASE2_BAND_CENTERS = 1000.0 * 2.0 ** (np.arange(-50, 50) / 3.0)


def _within_quarter_band(band_centers, fmin, fmax):
    """Centers between fmin and fmax, each limit widened by a quarter band so a
    nominal limit (20 Hz, 20 kHz) keeps its own band whichever side of it the
    exact center falls."""
    tolerance = 2.0 ** (1.0 / 12.0)
    return np.logical_and(band_centers >= fmin / tolerance, band_centers <= fmax * tolerance)


def _bands_within(band_centers, fmin, fmax):
    """Bands lying wholly between fmin and fmax, on the edges they are summed
    over (:func:`third_octave_band_edges`): a band reaching past either limit
    is only partly filled."""
    lower, upper = third_octave_band_edges(band_centers)
    slack = 1.0 + 1e-9
    return np.logical_and(lower * slack >= fmin, upper <= fmax * slack)


def third_octave_band_levels(signal, sampling_rate, cal=0.0, fmin=20.0, fmax=20000.0, window='hann'):
    """
    Compute third-octave band levels of a signal
    Args:
        signal: Array-like acoustic signal
        sampling_rate: Sampling rate of signal, Hz
        cal: Optional calibration factor to apply to signal (dB)
        fmin: minimum frequency for third-octave bands, Hz
        fmax: maximum frequency for third-octave bands, Hz
        window: window of the spectrum summed into bands, see :func:`psd`,
            default 'hann'

    Returns: tuple (band_centers, band_levels)
           WHERE
           band_centers is an array of third-octave band center frequencies
           band_levels is an array of third-octave band levels in dB
    """
    # fmin/fmax are compared within a quarter band, so nominal limits keep
    # their own bands: the exact centers of the 20 Hz and 20 kHz bands are
    # 19.69 and 20159 Hz.  Bands reaching past Nyquist would be only partly
    # filled, so they are dropped.
    band_centers = BASE2_BAND_CENTERS[np.logical_and(
        _within_quarter_band(BASE2_BAND_CENTERS, fmin, fmax),
        _bands_within(BASE2_BAND_CENTERS, 0.0, 0.5 * sampling_rate))]
    # Compute PSD
    frequency, psd_db, _ = psd(signal, sampling_rate, cal, window)
    psd_linear = (P_REF ** 2) * 10.0 ** (psd_db / 10.0)
    df = frequency[1] - frequency[0]
    band_levels = np.zeros_like(band_centers)
    for i, (f_lower, f_upper) in enumerate(zip(*third_octave_band_edges(band_centers))):
        band_indices = np.where(np.logical_and(frequency >= f_lower, frequency < f_upper))
        band_power = np.sum(psd_linear[band_indices] * df)
        band_levels[i] = 10.0 * np.log10(band_power / (P_REF ** 2))
    return band_centers, band_levels


def third_octave_band_edges(band_centers_hz):
    """Lower and upper edges of one-third octave bands that tile without gaps.

    Nominal centers (12.5, 1250, 1600 Hz, ...) are rounded, so edges taken as
    ``fc * 2**(+-1/6)`` straight from them overlap their neighbors or leave
    gaps -- up to 8 % of a band -- and a brick-wall band sum then drops or
    double-counts whatever lies there (a 1410 Hz tone falls in no band, an
    895 Hz one in two).  IEC 61260-1 defines each nominal band by its exact
    base-10 midband ``1000 * 10**(k/10)``, with edges a twentieth of a decade
    either side, so a nominal center (within a quarter band of one) is given
    those edges.  When every center is one of this module's own exact base-2
    centers ``1000 * 2**(k/3)``, they already tile and keep their
    ``fc * 2**(+-1/6)`` edges.  In a set that mixes the two, the base-2
    centers take their band's base-10 edges as well, because edges from the
    two grids do not meet: keeping ``fc * 2**(+-1/6)`` for 1259.9 Hz next to a
    nominal 1600 Hz band would overlap it by 1.7 Hz.  A center near neither
    grid keeps ``fc * 2**(+-1/6)``.

    Args:
        band_centers_hz: band center frequencies, Hz (exact or nominal)
    Returns: tuple (f_lower, f_upper) of arrays, Hz
    """
    fc = np.asarray(band_centers_hz, dtype=float)
    base2 = 1000.0 * 2.0 ** (np.round(3.0 * np.log2(fc / 1000.0)) / 3.0)
    if np.allclose(fc, base2, rtol=1e-9, atol=0.0):
        return fc / 2.0 ** (1.0 / 6.0), fc * 2.0 ** (1.0 / 6.0)
    base10 = 1000.0 * 10.0 ** (np.round(10.0 * np.log10(fc / 1000.0)) / 10.0)
    nominal = np.abs(np.log2(fc / base10)) < 1.0 / 12.0
    fm = np.where(nominal, base10, fc)
    half_band = np.where(nominal, 10.0 ** (1.0 / 20.0), 2.0 ** (1.0 / 6.0))
    return fm / half_band, fm * half_band


def spectrogram(signal, sampling_rate, window_time=0.5, window_type="hann", window_overlap=7.0 / 8.0,
                detrend='constant', dbref=20e-6):
    """
    Computes the spectrogram of a signal
    Args:
        signal: array-like containing acoustic signal
        sampling_rate: sampling rate of signal, Hz
        window_time: duration of windows, s
        window_type: optional type of window to us, see scipy.signal.window, default="hann"
        window_overlap: optional proportion of overlap for windows, default=7.0/8.0
        detrend: optional detrending mode for signal, see scipy.signal.detrend
        dbref: optional reference value for calculating decibels, default 20e-6

    Returns: tuple (f, t, SPL)
    WHERE
    f is an array of frequencies
    t is an array of times
    SPL is a frequency x time power spectral density spectrogram, dB re (20 uPa)^2/Hz
    """
    # Pick next power of two that captures the window time, and generate the window
    binwidth = frame_length(window_time, sampling_rate)
    window = scipy.signal.get_window(window_type, binwidth)
    # Calculate the spectrogram as a PSD
    f, t, Sxx = scipy.signal.spectrogram(signal, sampling_rate, window, noverlap=round(window_overlap * binwidth),
                                         detrend=detrend, mode='psd')
    # Convert to SPL (exact zeros in Sxx are legitimate; -inf, not a warning)
    with np.errstate(divide='ignore'):
        SPL = 10.0 * np.log10(Sxx / (dbref ** 2))
    return f, t, SPL


@acoustic_plot_style
def plot_spectrogram(signal, sampling_rate, window_time=1.0, window_type="hann", window_overlap=7.0 / 8.0,
                     detrend='constant', dbref=20e-6, save_name=None, title=None, time0=0, clim=None, flim=None):
    """
    Plot a spectrogram of signal
    Args:
        signal: array-like containing acoustic signal
        sampling_rate: sampling rate of signal, Hz
        window_time: duration of windows, s
        window_type: optional type of window to us, see scipy.signal.window, default="hann"
        window_overlap: optional proportion of overlap for windows, default=7.0/8.0
        detrend: optional detrending mode for signal, see scipy.signal.detrend
        dbref: optional reference value for calculating decibels, default 20e-6
        save_name: optional path to image file to save figure, default None
        title: optional title for the plot, default None
        time0: optional start time for signal, default 0
        clim: optional level scale color limits, default None
        flim: optional frequency scale limits, default None

    Returns: tuple (fig, ax, cs)
    WHERE
    fig is a matplotlib handle to the figure
    ax is a matplotlib handle to the plot axis
    cs is a matplotlib handle to the contour quadmesh
    """
    f, t, SPL = spectrogram(signal, sampling_rate, window_time, window_type, window_overlap, detrend, dbref)
    if flim is not None:
        fidx = np.logical_and(f >= flim[0], f <= flim[1])
        f = f[fidx]
        SPL = SPL[fidx, :]

    from matplotlib.pyplot import colorbar, subplots   # here: only the plot needs pyplot
    fig, ax = subplots(facecolor='white')
    if clim is None:
        vmin = None
        vmax = None
    else:
        vmin = clim[0]
        vmax = clim[1]
    cs = ax.pcolormesh(t + time0, f, SPL, vmin=vmin, vmax=vmax)
    cb = colorbar(cs, format='%.0f')
    ax.set_ylabel('Frequency, Hz')
    ax.set_xlabel('Time, s')
    cb.set_label('Power Spectral Density, dB')
    if title is not None:
        ax.set_title(title)
    if save_name is not None:
        fig.savefig(os.path.abspath(os.path.expanduser(save_name)))
    return fig, ax, cs


HANN_ENBW = 1.5          # equivalent noise bandwidth of a Hann window, bins


def hann_power(offset):
    """|W(d)|^2 of a Hann window, normalized to 1 at d = 0, d in bins."""
    d = np.asarray(offset, dtype=float)
    s = np.sinc(d)
    out = np.where(np.abs(np.abs(d) - 1.0) < 1e-9, 0.5, s / np.where(np.abs(1 - d ** 2) < 1e-12, 1.0, 1 - d ** 2))
    return out ** 2


def _running_integral_at(running, edge0, df, freq):
    """``running`` (Nbins + 1, Npts), a cumulative sum over bins of width ``df``
    whose first edge is ``edge0``, read at ``freq`` (Npts,) per column, linearly
    between bin edges and held at the ends."""
    x = np.clip((freq - edge0) / df, 0.0, running.shape[0] - 1.0)
    k = np.minimum(np.floor(x).astype(int), running.shape[0] - 2)
    cols = np.arange(running.shape[1])
    return running[k, cols] + (x - k) * (running[k + 1, cols] - running[k, cols])


def _band_integrals(running, edge0, df, lower, upper, scale):
    """(Nbands, Npts): ``running`` (see :func:`_running_integral_at`) between each
    band's edges, the edges scaled by each column's ``scale``."""
    return np.array([_running_integral_at(running, edge0, df, hi * scale)
                     - _running_integral_at(running, edge0, df, lo * scale)
                     for lo, hi in zip(lower, upper)]).reshape(len(lower), running.shape[1])


def _running_floor(psd, size, chunk=64):
    """Running median over ``size`` bins of each column of ``psd`` (bins x samples),
    edges held, taken over the bins that are not zero.

    Bins gated out as ambient are exactly zero; counted, they pull the median to
    zero wherever they are the majority, and every surviving bin then stands out
    as a tone.  Where none is zero this is ``median_filter(psd, (size, 1),
    mode='nearest')``, element for element.
    """
    half = size // 2
    padded = np.pad(psd, ((half, size - 1 - half), (0, 0)), mode='edge')
    floor = np.zeros_like(psd)
    for c0 in range(0, psd.shape[1], chunk):
        windows = np.lib.stride_tricks.sliding_window_view(padded[:, c0:c0 + chunk], size, axis=0)
        ordered = np.sort(windows, axis=-1)
        nonzero = np.count_nonzero(windows, axis=-1)
        rank = np.minimum(size - nonzero + nonzero // 2, size - 1)
        floor[:, c0:c0 + chunk] = np.where(nonzero > 0, np.take_along_axis(ordered, rank[..., None], axis=-1)[..., 0], 0.0)
    return floor


def tone_aware_band_power(psd, f, lower, upper, doppler=None, floor_bins=15, threshold_db=6.0, lobe=2, reach=4,
                          return_parts=False):
    """Band power (bands x samples) from Hann-windowed PSD columns (freq x samples), tones filed whole.

    Summing whole FFT bins between band edges (brick-wall) gives a tone near an edge to the wrong
    band: with 0.64 s Hann windows (1.56 Hz bins) the B407 main rotor's 27.6 and 55.2 Hz harmonics,
    0.6 and 1.0 Hz below the 25/31.5 and 50/63 Hz edges, put the 31.5 and 63 Hz bands of a hover
    5-19 dB high against a 4 s reference, and the 50 Hz band 1.5-2 dB low.  Here instead:

    each tone (a local maximum over +-lobe bins standing threshold_db above the running-median floor)
    is located to a fraction of a bin by the Hann ratio formula, its power inferred from the peak bin
    and the window's response, and that window-shaped contribution removed from the bins around it;
    the tone is filed whole in the band of its own frequency (divided by doppler when given), the
    remainder band-summed over (Doppler-scaled) edges.

    psd is in power per Hz (any power unit), f the bin centers (Hz, evenly spaced), lower and upper
    the band edges (Hz), doppler the (samples,) received-over-emitted frequency ratio, floor_bins
    the running-median width (bins), threshold_db the tone threshold above that floor (dB), lobe
    and reach half-widths (bins).  Returns band power (bands x samples, power units of psd x Hz).

    With return_parts=True, returns (tonal, broadband) instead, each bands x samples in the same
    units: the located tones' power filed by band, and the band-summed remainder.  Their sum is
    the total to rounding, and exactly where a band holds at most one tone in a sample.  The split
    is set by threshold_db, lobe and floor_bins: a tone weaker than threshold_db above the floor
    stays in the broadband part."""
    out, tonal_bands, broadband_bands = _tone_aware_bands(psd, f, lower, upper, doppler, floor_bins=floor_bins,
                                                          threshold_db=threshold_db, lobe=lobe, reach=reach,
                                                          parts=return_parts)
    return (tonal_bands, broadband_bands) if return_parts else out


def _tone_aware_bands(psd, f, lower, upper, doppler=None, *, floor_bins=15, threshold_db=6.0, lobe=2, reach=4,
                      parts=False):
    """:func:`tone_aware_band_power` as (total, tonal, broadband); the last two None unless ``parts``."""
    df = f[1] - f[0]
    floor = _running_floor(psd, floor_bins)
    peak = (psd == maximum_filter1d(psd, size=2 * lobe + 1, axis=0, mode='nearest')) & (psd > floor * 10 ** (threshold_db / 10))
    peak[:reach] = peak[-reach:] = False
    I, J = np.nonzero(peak)
    p0 = np.maximum(psd[I, J] - floor[I, J], 1e-300)
    pl = np.maximum(psd[I - 1, J] - floor[I - 1, J], 0.0)
    pr = np.maximum(psd[I + 1, J] - floor[I + 1, J], 0.0)
    right = pr >= pl
    alpha = np.sqrt(np.where(right, pr, pl) / p0)
    delta = np.clip((2 * alpha - 1) / (alpha + 1), 0.0, 0.5) * np.where(right, 1.0, -1.0)
    # PSD of a tone of power P at bin offset x: P |W(x)|^2 / (ENBW df).
    power = p0 * HANN_ENBW * df / np.maximum(hann_power(delta), 1e-6)
    tonal = np.zeros_like(psd)
    for o in range(-reach, reach + 1):
        np.add.at(tonal, (I + o, J), power * hann_power(o - delta) / (HANN_ENBW * df))
    broadband = np.clip(psd - tonal, 0.0, None)
    f_peak = f[I] + delta * df
    if doppler is not None:
        f_peak = f_peak / doppler[J]
    run = np.vstack((np.zeros((1, psd.shape[1])), np.cumsum(broadband * df, axis=0)))
    e0 = f[0] - 0.5 * df
    d = np.ones(psd.shape[1]) if doppler is None else doppler
    out = np.maximum(_band_integrals(run, e0, df, lower, upper, d), 0.0)
    tonal_bands = np.zeros_like(out) if parts else None
    broadband_bands = out.copy() if parts else None
    for ib, (lo, hi) in enumerate(zip(lower, upper)):
        inside = (f_peak >= lo) & (f_peak < hi)
        np.add.at(out[ib], J[inside], power[inside])
        if parts:
            np.add.at(tonal_bands[ib], J[inside], power[inside])
    return out, tonal_bands, broadband_bands


def _bands_missing(dropped, f, lower, upper, doppler):
    """(Nbands, Npts) True where a band, its edges scaled by each column's
    ``doppler``, spans a ``dropped`` bin or reaches more than half a bin past
    the bins ``f`` holds (the most an unscaled band within the selected range
    can, since the bins stop within a bin of either end of it)."""
    df = f[1] - f[0]
    edge0, edge1 = f[0] - 0.5 * df, f[-1] + 0.5 * df
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    out = ((lower[:, None] * doppler[None, :] < edge0 - 0.5 * df)
           | (upper[:, None] * doppler[None, :] > edge1 + 0.5 * df))
    if dropped.any():
        count = np.zeros((dropped.shape[0] + 1, dropped.shape[1]), dtype=np.int32)
        np.cumsum(dropped, axis=0, dtype=np.int32, out=count[1:])
        out |= _band_integrals(count, edge0, df, lower, upper, doppler) > 0.0
    return out


def OASPL(amplitudes):
    """
    Integrate an array of SPL amplitudes to compute the OASPL
    Args:
        amplitudes: array of SPL amplitudes
    Returns:
        OASPL, dB
    """
    return 10.0 * safe_log10(np.sum(np.power(10.0, amplitudes / 10.0)))


def power_to_db(power):
    """10 * log10(power), for power relative to its reference (p**2 / pref**2).

    Zero power comes back as -inf (no energy) and NaN stays NaN (no data), with
    none of numpy's divide-by-zero warnings.  Neither becomes a finite floor:
    the -3076 dB that ``np.maximum(power, np.finfo(float).tiny)`` gives reads as
    a level, pulling plot color scales and statistics down with it.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return 10.0 * np.log10(np.asarray(power, dtype=float))


def safe_log10(x):
    """
    Safe version of log10, where values below zero are returned as NaN and values equal to zero are -inf
    Args:
        x: array of values

    Returns: checked log10 value
    """
    if x == 0:
        return -np.inf
    if x < 0:
        return np.nan
    else:
        return np.log10(x)


def dBAw(f):
    """
    A-weighting curve
    Args:
        f: frequencies to evaluate

    Returns: weighting (in dB) at each frequency
    """
    f = np.asarray(f, dtype=float)
    aweights = np.full_like(f, -300.0, dtype=float)
    pos = f > 0
    if np.any(pos):
        fp = f[pos]
        aweights[pos] = (
            10.0 * np.log10(
                1.562339
                * fp ** 4.0
                / ((fp ** 2.0 + 107.65265 ** 2.0) * (fp ** 2.0 + 737.86223 ** 2.0))
            )
            + 10.0
            * np.log10(
                2.242881e16
                * fp ** 4.0
                / ((fp ** 2.0 + 20.598997 ** 2.0) ** 2.0 * (fp ** 2.0 + 12194.22 ** 2.0) ** 2.0)
            )
        )
    return aweights


def highpass(x, fpass, fs, zero_phase=True):
    """
    Applies a high pass filter to a signal
    Args:
        x: signal array
        fpass: high pass frequency
        fs: sampling rate of x
        zero_phase: optional, use phase preserving filter, default True.  Note that
            panam.acoustics.filters.highpass, which this wraps, defaults to False (causal).

    Returns: filtered signal
    """
    return np.array(pa_filters.highpass(x, fpass, fs, zero_phase=zero_phase))


def lowpass(x, fpass, fs, zero_phase=True):
    """
    Applies a low pass filter to a signal
    Args:
        x: signal array
        fpass: low pass frequency
        fs: sampling rate of x
        zero_phase: optional, use phase preserving filter, default True.  Note that
            panam.acoustics.filters.lowpass, which this wraps, defaults to False (causal).

    Returns: filtered signal
    """
    return np.array(pa_filters.lowpass(x, fpass, fs, zero_phase=zero_phase))
