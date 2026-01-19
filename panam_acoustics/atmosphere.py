"""Minimal Atmosphere class for atmospheric absorption.

Derived from python-acoustics (BSD-3-Clause).
"""

from __future__ import annotations

from .iso_9613_1_1993 import (
    REFERENCE_PRESSURE,
    REFERENCE_TEMPERATURE,
    TRIPLE_TEMPERATURE,
    attenuation_coefficient,
    molar_concentration_water_vapour,
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
    def molar_concentration_water_vapour(self):
        return molar_concentration_water_vapour(
            self.relative_humidity,
            self.saturation_pressure,
            self.pressure,
        )

    @property
    def relaxation_frequency_nitrogen(self):
        return relaxation_frequency_nitrogen(
            self.pressure,
            self.temperature,
            self.molar_concentration_water_vapour,
            self.reference_pressure,
            self.reference_temperature,
        )

    @property
    def relaxation_frequency_oxygen(self):
        return relaxation_frequency_oxygen(
            self.pressure,
            self.molar_concentration_water_vapour,
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
