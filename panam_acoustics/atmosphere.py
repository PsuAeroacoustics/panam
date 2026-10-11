"""Atmospheric absorption: the ISO 9613-1 Atmosphere, band levels and layered paths.

The Atmosphere class is derived from python-acoustics (BSD-3-Clause).
:func:`band_absorption_db` forms one-third octave band absorption from the
pure-tone coefficient (at the band center, by SAE ARP 5534, or integrated over
the band), and :func:`layered_attenuation_db` integrates it along straight rays
through a layered atmosphere.
"""

from __future__ import annotations

import numpy as np

from .iso_9613_1_1993 import (
    REFERENCE_PRESSURE,
    REFERENCE_TEMPERATURE,
    TRIPLE_TEMPERATURE,
    attenuation_coefficient,
    molar_concentration_water_vapor,
    relaxation_frequency_nitrogen,
    relaxation_frequency_oxygen,
    saturation_pressure,
    soundspeed,
)


class Atmosphere:
    """Class describing atmospheric conditions for ISO 9613-1 absorption."""

    def __init__(
        self,
        temperature: float = REFERENCE_TEMPERATURE,
        pressure: float = REFERENCE_PRESSURE,
        relative_humidity: float = 0.0,
        reference_temperature: float = REFERENCE_TEMPERATURE,
        reference_pressure: float = REFERENCE_PRESSURE,
        triple_temperature: float = TRIPLE_TEMPERATURE,
    ) -> None:
        self.temperature = temperature
        self.pressure = pressure
        self.relative_humidity = relative_humidity
        self.reference_temperature = reference_temperature
        self.reference_pressure = reference_pressure
        self.triple_temperature = triple_temperature

    @property
    def soundspeed(self):
        return soundspeed(self.temperature, self.reference_temperature)

    @property
    def saturation_pressure(self):
        return saturation_pressure(
            self.temperature,
            self.reference_pressure,
            self.triple_temperature,
        )

    @property
    def molar_concentration_water_vapor(self):
        return molar_concentration_water_vapor(
            self.relative_humidity,
            self.saturation_pressure,
            self.pressure,
        )

    @property
    def relaxation_frequency_nitrogen(self):
        return relaxation_frequency_nitrogen(
            self.pressure,
            self.temperature,
            self.molar_concentration_water_vapor,
            self.reference_pressure,
            self.reference_temperature,
        )

    @property
    def relaxation_frequency_oxygen(self):
        return relaxation_frequency_oxygen(
            self.pressure,
            self.molar_concentration_water_vapor,
            self.reference_pressure,
        )

    def attenuation_coefficient(self, frequency):
        return attenuation_coefficient(
            self.pressure,
            self.temperature,
            self.reference_pressure,
            self.reference_temperature,
            self.relaxation_frequency_nitrogen,
            self.relaxation_frequency_oxygen,
            frequency,
        )


# --------------------------------------------------------------------------
# Band-level and layered absorption
# --------------------------------------------------------------------------

#: How a band's absorption is formed from ISO 9613-1's pure-tone coefficient:
#: 'center' (at the band center, PANAM's default), 'arp5534' (SAE ARP 5534) and
#: 'integrated' (ISO 9613-1 integrated over the band's filter and a spectrum).
BAND_ABSORPTION_METHODS = ('center', 'arp5534', 'integrated')

#: SAE ARP 5534 constants A-G (Rickley et al., DOT-VNTSC-FAA-12-14, 2012, Eqs. 16-17).
ARP5534_COEFFICIENTS = dict(A=0.867942, B=0.111761, C=0.95824, D=0.008191, E=1.6, F=9.2, G=0.765)
#: The mid-band (pure-tone) attenuation, dB, at which ARP 5534 changes from its
#: power law (Eq. 16) to its straight line (Eq. 17).
ARP5534_BREAK_DB = 150.0

#: Points per band, and half-width in octaves, of the 'integrated' quadrature.
_INTEGRATION_POINTS = 241
_INTEGRATION_HALF_WIDTH_OCTAVES = 1.0
#: Quality factor of an order-3 Butterworth band-pass with -3 dB edges at
#: fc 2^-1/6 and fc 2^1/6, the filter panam_acoustics.filters uses.
_FILTER_Q = 1.0 / (2.0 ** (1.0 / 6.0) - 2.0 ** (-1.0 / 6.0))
_FILTER_ORDER = 3


def exact_midband_frequency(band_centers):
    """The exact base-10 mid-band frequency 10^(i/10) Hz nearest each band center.

    A nominal center (1250 Hz) and an exact base-2 one (1259.9 Hz) both map to
    the exact base-10 frequency (1258.9 Hz), at which SAE ARP 5534 evaluates the
    pure-tone attenuation.

    Args:
        band_centers: band center frequencies, Hz (positive).

    Returns: exact mid-band frequencies, Hz, shaped like ``band_centers``.
    """
    centers = np.asarray(band_centers, dtype=float)
    if np.any(~np.isfinite(centers)) or np.any(centers <= 0.0):
        raise ValueError('band centers must be positive frequencies in Hz')
    return 10.0 ** (np.round(10.0 * np.log10(centers)) / 10.0)


def arp5534_band_attenuation(tone_attenuation_db):
    """SAE ARP 5534 band attenuation from the mid-band pure-tone attenuation.

    The method of SAE ARP 5534 (2013), developed in Rickley, E. J., Rosenbaum,
    J. E., Fleming, G. G., Roof, C. J., and Boeker, E. R., "Development of
    Simplified Procedure for Computing the Absorption of Sound by the Atmosphere
    and Applicability to Aircraft Noise Certification: Proposed SAE Method,"
    DOT-VNTSC-FAA-12-14, Volpe Center, 2012, Eqs. 16-17::

        delta_B = A delta_t (1 + B (C - D delta_t))^E,   delta_t < 150 dB
        delta_B = F + G delta_t,                         delta_t >= 150 dB

    with ``delta_t = alpha(f_m) s`` the pure-tone attenuation over the path at
    the exact mid-band frequency f_m and the constants of
    :data:`ARP5534_COEFFICIENTS`.  It is a least-squares fit to the ISO 9613-1 /
    ANSI S1.26 spectrum-integration method for broadband spectra sloping -5 to
    +5 dB per band through base-10 order-3 Butterworth one-third octave filters,
    normalized to a fixed slope of 3 dB per band, and extends the ANSI S1.26
    Annex E approximation past its 50 dB limit.  The branches meet at 150 dB
    within 0.05 dB.  It is reciprocal: the same value applies from source to
    receiver and back.

    Args:
        tone_attenuation_db: mid-band pure-tone attenuation over the path, dB
            (non-negative; any shape).

    Returns: band attenuation, dB, shaped like ``tone_attenuation_db``.
    """
    c = ARP5534_COEFFICIENTS
    delta_t = np.asarray(tone_attenuation_db, dtype=float)
    low = c['A'] * delta_t * (1.0 + c['B'] * (c['C'] - c['D'] * np.minimum(delta_t, ARP5534_BREAK_DB))) ** c['E']
    return np.where(delta_t < ARP5534_BREAK_DB, low, c['F'] + c['G'] * delta_t)


def _band_attenuation(tone_db, band_centers, method, spectrum_slope_db):
    """Band attenuation, dB, shaped (bands,) + path shape, from a pure-tone one.

    ``tone_db(f)`` returns the pure-tone attenuation over the path(s), dB, shaped
    ``f.shape + path shape``, for a 1-D array of frequencies ``f`` in Hz.
    """
    if method not in BAND_ABSORPTION_METHODS:
        raise ValueError('absorption band method must be one of {}, not {!r}'.format(
            BAND_ABSORPTION_METHODS, method))
    centers = np.atleast_1d(np.asarray(band_centers, dtype=float))
    if centers.ndim != 1:
        raise ValueError('band centers must be one-dimensional')
    if method == 'center':
        return np.asarray(tone_db(centers), dtype=float)
    midband = exact_midband_frequency(centers)
    if method == 'arp5534':
        return arp5534_band_attenuation(tone_db(midband))
    # 'integrated': the band's power after the path over its power before, for a
    # spectrum whose band levels slope ``spectrum_slope_db`` per one-third
    # octave (density proportional to f^(slope - 1), so f^slope per unit log
    # frequency) seen through the band's filter, by the trapezoidal rule on a
    # uniform grid in log frequency.
    slope = np.broadcast_to(np.asarray(spectrum_slope_db, dtype=float), centers.shape)
    octaves = np.linspace(-_INTEGRATION_HALF_WIDTH_OCTAVES, _INTEGRATION_HALF_WIDTH_OCTAVES,
                          _INTEGRATION_POINTS)
    ratio = 2.0 ** octaves
    trapezoid = np.ones(octaves.size)
    trapezoid[[0, -1]] = 0.5
    response = trapezoid / (1.0 + (_FILTER_Q * (ratio - 1.0 / ratio)) ** (2 * _FILTER_ORDER))
    out = None
    for ib, fc in enumerate(centers):
        weight = response * ratio ** slope[ib]
        weight = weight / np.sum(weight)
        tone = np.asarray(tone_db(fc * ratio), dtype=float)          # (K,) + path shape
        weight = weight.reshape((-1,) + (1,) * (tone.ndim - 1))
        # Relative to the least-attenuated frequency, so a long path does not
        # underflow every term to zero.
        least = np.min(tone, axis=0)
        kept = np.sum(weight * 10.0 ** (-(tone - least) / 10.0), axis=0)
        band = least - 10.0 * np.log10(kept)
        if out is None:
            out = np.empty((centers.size,) + band.shape, dtype=float)
        out[ib] = band
    return out


def band_absorption_db(distance_m, band_centers, temperature, pressure, relative_humidity,
                       method='center', spectrum_slope_db=0.0):
    """Atmospheric absorption of one-third octave bands over a uniform path.

    ``method`` says how the ISO 9613-1 pure-tone coefficient alpha(f) (dB/m)
    becomes a band's attenuation over ``distance_m``:

    - ``'center'``: alpha at the given band center times the distance, what PANAM
      has always done.  It serves for FFT bins too.
    - ``'arp5534'``: SAE ARP 5534 (:func:`arp5534_band_attenuation`): the
      pure-tone attenuation at the exact base-10 mid-band frequency
      (:func:`exact_midband_frequency`), corrected for how absorption varies
      across the band.  It takes no spectrum: it was fitted over spectra of
      several slopes and normalized to 3 dB per band.
    - ``'integrated'``: ISO 9613-1 integrated over the band (the spectrum
      integration of ISO 9613-1 and ANSI S1.26 Annex D), the band's power after
      the path over its power before::

          delta_B = -10 lg [ int W(f) 10^(-alpha(f) s / 10) df / int W(f) df ],
          W(f) = |H(f)|^2 (f / f_c)^(k - 1),
          |H(f)|^2 = 1 / (1 + (Q (f/f_c - f_c/f))^6),  Q = 1 / (2^(1/6) - 2^(-1/6)),

      with H the order-3 Butterworth filter of
      :func:`panam_acoustics.filters.third_octave_filter_bank` about the given
      center and a spectrum whose band levels rise ``k = spectrum_slope_db`` dB
      per one-third octave (0, the default, is flat band levels).  It is
      integrated over one octave either side of the center (241 points); past
      about 50 dB of mid-band attenuation the result is set by the filter skirt
      and so depends on that range.

    Args:
        distance_m: path length(s), m (scalar or array).
        band_centers: (n,) band center frequencies, Hz (nominal or exact).
        temperature: air temperature, K.
        pressure: air pressure, kPa.
        relative_humidity: relative humidity, percent.
        method: one of :data:`BAND_ABSORPTION_METHODS`.
        spectrum_slope_db: ``'integrated'`` only: the spectrum's band-level
            slope, dB per one-third octave, a scalar or one per band.

    Returns: band attenuation, dB, shaped ``(n,) + np.shape(distance_m)``.
    """
    atmosphere = Atmosphere(temperature=temperature, pressure=pressure,
                            relative_humidity=relative_humidity)
    distance = np.asarray(distance_m, dtype=float)

    def tone_db(frequency):
        alpha = np.asarray(atmosphere.attenuation_coefficient(np.asarray(frequency, dtype=float)), dtype=float)
        return alpha.reshape(alpha.shape + (1,) * distance.ndim) * distance

    return _band_attenuation(tone_db, band_centers, method, spectrum_slope_db)


#: Keys of an absorption profile (:func:`layered_attenuation_db`).
ABSORPTION_PROFILE_KEYS = ('altitude_ft', 'temperature_k', 'relative_humidity', 'pressure_kpa')
#: Keys of a straight ray (:func:`layered_attenuation_db`).
RAY_KEYS = ('source_altitude_ft', 'receiver_altitude_ft', 'path_length_ft')

_FT_TO_M = 0.3048


def check_absorption_profile(profile):
    """Validate an absorption profile and return its columns as float arrays.

    Args:
        profile: mapping of ``altitude_ft`` (ft, strictly increasing),
            ``temperature_k`` (K), ``relative_humidity`` (percent) and
            ``pressure_kpa`` (kPa), 1-D arrays of one length.

    Returns: (altitude ft, temperature K, relative humidity %, pressure kPa).
    """
    missing = [key for key in ABSORPTION_PROFILE_KEYS if key not in profile]
    if missing:
        raise ValueError('absorption profile needs {}'.format(', '.join(missing)))
    columns = [np.atleast_1d(np.asarray(profile[key], dtype=float)) for key in ABSORPTION_PROFILE_KEYS]
    if any(c.ndim != 1 or c.size != columns[0].size for c in columns) or columns[0].size < 1:
        raise ValueError('absorption profile columns must be one-dimensional and of one length')
    if not all(np.all(np.isfinite(c)) for c in columns):
        raise ValueError('absorption profile values must be finite')
    altitude, temperature, humidity, pressure = columns
    if np.any(np.diff(altitude) <= 0.0):
        raise ValueError('absorption profile altitude_ft must increase strictly')
    if np.any(temperature < 150.0) or np.any(temperature > 400.0):
        raise ValueError('absorption profile temperature_k must be in kelvin (150-400 K)')
    if np.any(pressure < 1.0) or np.any(pressure > 120.0):
        raise ValueError('absorption profile pressure_kpa must be in kPa (1-120 kPa)')
    if np.any(humidity < 0.0) or np.any(humidity > 100.0):
        raise ValueError('absorption profile relative_humidity must be a percentage')
    return altitude, temperature, humidity, pressure


def layered_attenuation_db(profile, ray, frequency_hz, method='center', spectrum_slope_db=0.0):
    """Atmospheric absorption along straight rays through a layered atmosphere.

    The ISO 9613-1 coefficient alpha (dB/m) is evaluated at each altitude of the
    profile, taken to vary linearly with altitude between them, and held at its
    end values beyond them.  Along a straight ray from altitude z_r to z_s, of
    length l, the pure-tone attenuation is the mean of alpha over that altitude
    range times the length::

        delta_t = l / (z_s - z_r) int_{z_r}^{z_s} alpha(z) dz,

    integrated exactly for the piecewise-linear alpha; a level ray takes
    l alpha(z_r).  ``method`` then forms band values from delta_t as
    :func:`band_absorption_db` does: ``'center'`` is the pure-tone value at
    each frequency, ``'arp5534'`` applies SAE ARP 5534 to delta_t at the exact
    mid-band frequency, and ``'integrated'`` integrates delta_t(f) over the
    band.  With one profile level, or every level alike, the result is
    :func:`band_absorption_db`'s.

    Args:
        profile: see :func:`check_absorption_profile`.  Altitude is measured in
            the same frame, and from the same datum, as the ray's.
        ray: mapping of ``source_altitude_ft``, ``receiver_altitude_ft`` and
            ``path_length_ft`` (ft), arrays that broadcast together.
        frequency_hz: (n,) frequencies or band centers, Hz.
        method: one of :data:`BAND_ABSORPTION_METHODS`.
        spectrum_slope_db: ``'integrated'`` only: dB per one-third octave.

    Returns: attenuation, dB, shaped ``(n,) +`` the rays' broadcast shape.
    """
    altitude, temperature, humidity, pressure = check_absorption_profile(profile)
    missing = [key for key in RAY_KEYS if key not in ray]
    if missing:
        raise ValueError('ray needs {}'.format(', '.join(missing)))
    z_source, z_receiver, length = np.broadcast_arrays(
        *(np.asarray(ray[key], dtype=float) for key in RAY_KEYS))
    shape = z_source.shape
    z_s = z_source.ravel() * _FT_TO_M
    z_r = z_receiver.ravel() * _FT_TO_M
    length_m = length.ravel() * _FT_TO_M
    z_nodes = altitude * _FT_TO_M
    nodes = Atmosphere(temperature=temperature, pressure=pressure, relative_humidity=humidity)
    n = z_nodes.size
    level = np.abs(z_s - z_r) < 1e-9
    if n > 1:
        span = np.diff(z_nodes)
        # The segment each ray end lies in (the end segments beyond the profile).
        k_s = np.clip(np.searchsorted(z_nodes, z_s, side='right') - 1, 0, n - 2)
        k_r = np.clip(np.searchsorted(z_nodes, z_r, side='right') - 1, 0, n - 2)

    def tone_db(frequency):
        f = np.asarray(frequency, dtype=float)
        alpha = np.asarray(nodes.attenuation_coefficient(f[:, None]), dtype=float)   # (nf, n)
        if n == 1:
            return alpha[:, :1] * length_m
        # Running integral of alpha over altitude at the nodes, dB.
        cumulative = np.concatenate((np.zeros((f.size, 1)),
                                     np.cumsum(0.5 * (alpha[:, 1:] + alpha[:, :-1]) * span, axis=1)), axis=1)

        def integral(z, k):
            a0, a1 = alpha[:, k], alpha[:, k + 1]
            dz = np.clip(z, z_nodes[0], z_nodes[-1]) - z_nodes[k]
            inside = cumulative[:, k] + a0 * dz + 0.5 * (a1 - a0) / span[k] * dz ** 2
            below = np.minimum(z - z_nodes[0], 0.0) * alpha[:, :1]
            above = np.maximum(z - z_nodes[-1], 0.0) * alpha[:, -1:]
            return inside + below + above

        a0, a1 = alpha[:, k_r], alpha[:, k_r + 1]
        local = a0 + (a1 - a0) * np.clip((z_r - z_nodes[k_r]) / span[k_r], 0.0, 1.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = (integral(z_s, k_s) - integral(z_r, k_r)) / (z_s - z_r)
        return np.where(level, local, mean) * length_m

    result = _band_attenuation(tone_db, frequency_hz, method, spectrum_slope_db)
    return result.reshape(result.shape[:1] + shape)
