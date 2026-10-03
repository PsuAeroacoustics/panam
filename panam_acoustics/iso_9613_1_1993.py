"""ISO 9613-1:1993 atmospheric absorption (minimal subset).

Derived from python-acoustics (BSD-3-Clause).
"""

from __future__ import annotations

import numpy as np

SOUNDSPEED = 343.2
REFERENCE_TEMPERATURE = 293.15
REFERENCE_PRESSURE = 101.325
TRIPLE_TEMPERATURE = 273.16


def soundspeed(temperature, reference_temperature: float = REFERENCE_TEMPERATURE):
    return 343.2 * np.sqrt(temperature / reference_temperature)


def saturation_pressure(
    temperature,
    reference_pressure: float = REFERENCE_PRESSURE,
    triple_temperature: float = TRIPLE_TEMPERATURE,
):
    return reference_pressure * 10.0 ** (
        -6.8346 * (triple_temperature / temperature) ** 1.261 + 4.6151
    )


def molar_concentration_water_vapor(relative_humidity, saturation_pressure, pressure):
    return relative_humidity * saturation_pressure / pressure


def relaxation_frequency_oxygen(
    pressure, h, reference_pressure: float = REFERENCE_PRESSURE
):
    return pressure / reference_pressure * (
        24.0 + 4.04 * 10.0**4.0 * h * (0.02 + h) / (0.391 + h)
    )


def relaxation_frequency_nitrogen(
    pressure,
    temperature,
    h,
    reference_pressure: float = REFERENCE_PRESSURE,
    reference_temperature: float = REFERENCE_TEMPERATURE,
):
    return pressure / reference_pressure * (temperature / reference_temperature) ** (-0.5) * (
        9.0
        + 280.0
        * h
        * np.exp(
            -4.170 * ((temperature / reference_temperature) ** (-1.0 / 3.0) - 1.0)
        )
    )


def attenuation_coefficient(
    pressure,
    temperature,
    reference_pressure,
    reference_temperature,
    relaxation_frequency_nitrogen,
    relaxation_frequency_oxygen,
    frequency,
):
    return 8.686 * frequency**2.0 * (
        (1.84 * 10.0 ** (-11.0) * (reference_pressure / pressure) * (temperature / reference_temperature) ** 0.5)
        + (temperature / reference_temperature) ** (-2.5)
        * (
            0.01275
            * np.exp(-2239.1 / temperature)
            * (relaxation_frequency_oxygen + (frequency**2.0 / relaxation_frequency_oxygen)) ** (-1.0)
            + 0.1068
            * np.exp(-3352.0 / temperature)
            * (relaxation_frequency_nitrogen + (frequency**2.0 / relaxation_frequency_nitrogen)) ** (-1.0)
        )
    )
