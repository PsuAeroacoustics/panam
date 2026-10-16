"""Atmospheric absorption along a path and over a band.

:func:`atmosorb` is the ISO 9613-1 coefficient in dB/m (deg C, % RH, mbar);
:func:`path_absorption_db` and :func:`_band_attenuation_db` apply the band
absorption methods of :func:`panam.acoustics.atmosphere.band_absorption_db`
(``docs/THEORY.md`` section 5.4).
"""

import numpy as np

from panam import unit_conversion
from panam.acoustics.atmosphere import (Atmosphere, BAND_ABSORPTION_METHODS, band_absorption_db,
                                        layered_attenuation_db)


def _check_band_method(method):
    """Raise ValueError unless ``method`` is one of ``BAND_ABSORPTION_METHODS``."""
    if method not in BAND_ABSORPTION_METHODS:
        raise ValueError('absorption_band_method must be one of {}, not {!r}'.format(
            BAND_ABSORPTION_METHODS, method))


def _band_attenuation_db(atmosphere, frequency_hz, distance_m, method):
    """Band attenuation (dB) over ``distance_m`` (m) in ``atmosphere`` (K, kPa, % RH).

    'center' is the ISO 9613-1 coefficient at ``frequency_hz`` times the
    distance, computed directly so its results stay bit-identical; any other
    method goes through :func:`panam.acoustics.atmosphere.band_absorption_db`.
    """
    if method == 'center':
        return atmosphere.attenuation_coefficient(frequency_hz) * distance_m
    return band_absorption_db(distance_m, frequency_hz, atmosphere.temperature, atmosphere.pressure,
                              atmosphere.relative_humidity, method=method)


def metadata_atmosphere(metadata, default=None):
    """The :class:`Atmosphere` a hemisphere's ``metadata['atmosphere']`` records
    (``temperature_k`` K, ``pressure_kpa`` kPa, ``relative_humidity`` %), or
    ``default`` when it records none."""
    recorded = metadata.get('atmosphere')
    if recorded is None:
        return default
    return Atmosphere(temperature=recorded['temperature_k'], pressure=recorded['pressure_kpa'],
                      relative_humidity=recorded['relative_humidity'])


def path_absorption_db(frequency, method, r_path, source_z, receiver_z, *, length_units, atmosphere,
                       absorption_profile=None):
    """(Nf, Npts) band attenuation, dB, of each path by ``method``: in the
    layered ``absorption_profile`` (:func:`panam.acoustics.atmosphere.layered_attenuation_db`)
    if given, else in the uniform ``atmosphere``
    (:func:`panam.acoustics.atmosphere.band_absorption_db`).

    ``frequency`` (Nf,) Hz; ``r_path`` (Npts,) path lengths and ``source_z``,
    ``receiver_z`` heights, all in ``length_units`` ('ft' or 'm').
    Depropagation and :mod:`synthesis` both form absorption with it.
    """
    def to(value, units):
        return np.asarray(unit_conversion.len_conv(np.asarray(value, dtype=float), from_units=length_units,
                                                   to_units=units), dtype=float)
    if absorption_profile is None:
        return band_absorption_db(to(r_path, 'm'), frequency, atmosphere.temperature, atmosphere.pressure,
                                  atmosphere.relative_humidity, method=method)
    ray = dict(source_altitude_ft=to(source_z, 'ft'), receiver_altitude_ft=to(receiver_z, 'ft'),
               path_length_ft=to(r_path, 'ft'))
    return layered_attenuation_db(absorption_profile, ray, frequency, method=method)


def atmosorb(freq, temp, humid, pstat):
    """
    Atmospheric absorption, ISO 9613-1, via panam.acoustics.atmosphere.Atmosphere.

    Args:
        freq: Array-like of frequencies in Hz.
        temp: Temperature in degrees Celsius.
        humid: Relative humidity in percent (e.g., 85% -> 85).
        pstat: Static pressure in mbar.

    Returns:
        numpy.ndarray of attenuation in dB per meter (same shape as `freq`).
    """
    # Convert inputs to Atmosphere() expected units
    tempK = np.asarray(temp, dtype=float) + 273.15  # C -> K
    pres_kpa = np.asarray(pstat, dtype=float) * 0.1  # mbar -> kPa
    rh_pct = np.asarray(humid, dtype=float)
    f = np.asarray(freq, dtype=float)

    # Scalar atmosphere: plain floats, result shaped like freq
    if tempK.ndim == 0 and pres_kpa.ndim == 0 and rh_pct.ndim == 0:
        atm = Atmosphere(
            temperature=float(tempK),
            pressure=float(pres_kpa),
            relative_humidity=float(rh_pct),
        )
        alpha_db_per_m = atm.attenuation_coefficient(f)
        return alpha_db_per_m

    # Broadcast atmospheric inputs to common grid.  The ISO 9613-1 formulas
    # are elementwise, so one Atmosphere covers it, with trailing axes for
    # frequency: the result is grid_shape + f.shape.
    tempK_b, pres_kpa_b, rh_b = np.broadcast_arrays(tempK, pres_kpa, rh_pct)
    expand = (Ellipsis,) + (None,) * f.ndim
    atm = Atmosphere(
        temperature=tempK_b[expand],
        pressure=pres_kpa_b[expand],
        relative_humidity=rh_b[expand],
    )
    return atm.attenuation_coefficient(f)
