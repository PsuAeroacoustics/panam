"""The dBA and EAA every database group carries, against their formula:
SPLA = 10 lg sum 10^((L + A)/10), EAA = SPLA - 10 lg sum 10^((L + A - d alpha)/10),
d the EAA distance (m) and alpha the attenuation coefficient (dB/m)."""

import numpy as np
import pytest
from netCDF4 import Dataset

import flight_acoustics as fa
from panam_acoustics.atmosphere import Atmosphere
from sphere_helpers import minimal_hemisphere, write_sphere_directory

ATMOSPHERE = Atmosphere(temperature=293.15, pressure=101.325, relative_humidity=20.0)


def _formula(levels, frequency, distance):
    a_weight = fa.dBAw(frequency)
    alpha = ATMOSPHERE.attenuation_coefficient(np.asarray(frequency, dtype=float))
    spla = 10.0 * np.log10(np.sum(10.0 ** (0.1 * (levels + a_weight)), axis=-1))
    attenuated = 10.0 * np.log10(np.sum(10.0 ** (0.1 * (levels + a_weight - distance * alpha)), axis=-1))
    return spla, spla - attenuated


@pytest.mark.parametrize('frequency', [1000.0, 125.0, 8000.0])
def test_one_band_is_the_band_level_plus_its_a_weight_and_eaa_its_absorption(frequency):
    levels = np.array([[[70.0], [80.0], [91.5]]])
    spla, eaa = fa.spla_and_eaa_from_spectrum(levels, np.array([frequency]), 1000.0, ATMOSPHERE)
    np.testing.assert_allclose(spla, levels[..., 0] + fa.dBAw(frequency), rtol=0, atol=1e-12)
    alpha = float(np.squeeze(ATMOSPHERE.attenuation_coefficient(np.array([frequency]))))
    np.testing.assert_allclose(eaa, 1000.0 * alpha, rtol=1e-12)
    # dB per metre: at 1 kHz, 20 C and 20 % RH about 9 dB over the 1000 m.
    if frequency == 1000.0:
        assert 5.0 < eaa[0, 0] < 15.0


@pytest.mark.parametrize('distance', [1000.0, 300.0])
def test_two_bands_follow_the_formula(distance):
    rng = np.random.default_rng(3)
    frequency = np.array([500.0, 4000.0])
    levels = rng.uniform(50.0, 90.0, (4, 5, 2))
    spla, eaa = fa.spla_and_eaa_from_spectrum(levels, frequency, distance, ATMOSPHERE)
    expected_spla, expected_eaa = _formula(levels, frequency, distance)
    np.testing.assert_allclose(spla, expected_spla, rtol=0, atol=1e-10)
    np.testing.assert_allclose(eaa, expected_eaa, rtol=0, atol=1e-10)
    # The attenuated sum is weighted toward the band that loses less.
    assert np.all(eaa > 0.0)
    assert np.all(eaa < distance * ATMOSPHERE.attenuation_coefficient(frequency).max())


def test_the_database_carries_them_for_every_group(tmp_path):
    rng = np.random.default_rng(4)
    hemispheres = []
    for speed, angle in [(40, -6), (60, 0), (80, -3)]:
        hemisphere = minimal_hemisphere()
        hemisphere['third_octave']['band_centers_hz'] = np.array([250.0, 1000.0, 4000.0])
        bands = hemisphere['third_octave']['bands_db']
        hemisphere['third_octave']['bands_db'] = bands + rng.uniform(-10.0, 10.0, bands.shape)
        hemispheres.append((hemisphere, speed, angle))
    directory = write_sphere_directory(tmp_path / 'spheres', hemispheres)
    database = tmp_path / 'database.nod'
    fa.build_empirical_database(str(directory), str(database), load_factors=None,
                                atmosphere=ATMOSPHERE)
    with Dataset(str(database)) as db:
        frequency = np.asarray(db['frequency'][:], dtype=float)
        np.testing.assert_array_equal(frequency, [250.0, 1000.0, 4000.0])
        assert len(db.groups) >= 3
        for name, group in db.groups.items():
            amplitude = np.asarray(group['amplitude'][:], dtype=float)
            spla, eaa = _formula(amplitude, frequency, 1000.0)
            np.testing.assert_allclose(np.asarray(group['dBA'][:]), spla.ravel(), rtol=0, atol=1e-9,
                                       err_msg=name)
            np.testing.assert_allclose(np.asarray(group['EAA'][:]), eaa.ravel(), rtol=0, atol=1e-9,
                                       err_msg=name)
