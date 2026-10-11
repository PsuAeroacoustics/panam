"""AAM sphere netCDF to NORAH2 .hem (sphere_to_norah2.py)."""

import numpy as np
import pytest

import flight_acoustics as fa
from panam_acoustics.atmosphere import Atmosphere
from sphere_helpers import write_raw_sphere
from sphere_to_norah2 import sphere_levels_on_norah2_grid, sphere_to_norah2

MEASURED = Atmosphere(temperature=283.15, pressure=97.5, relative_humidity=55.0)
REFERENCE = Atmosphere(temperature=fa.NORAH2_REFERENCE_TEMPERATURE_K,
                       pressure=fa.NORAH2_REFERENCE_PRESSURE_PA / 1000.0,
                       relative_humidity=fa.NORAH2_REFERENCE_RELATIVE_HUMIDITY)
RADIUS_FT = 100.0
# The build's default grid: PHI every 10 deg, THETA every 5 deg, 31 bands 10 Hz to 10 kHz.
PHI = np.arange(-90.0, 90.0 + 1e-9, 10.0)
THETA = np.arange(0.0, 180.0 + 1e-9, 5.0)
BANDS = fa.NORAH2_BAND_CENTERS_HZ


def directional_levels(phi, theta, frequency):
    """A field that varies with every axis: (phi, theta, band) dB."""
    P, T, F = np.meshgrid(phi, theta, frequency, indexing='ij')
    return 80.0 + 6.0 * np.sin(np.radians(P)) + 4.0 * np.cos(np.radians(T)) - 3.0 * np.log10(F / 1000.0)


def write_sphere(path, phi=PHI, theta=THETA, frequency=BANDS, amplitude=None):
    if amplitude is None:
        amplitude = directional_levels(phi, theta, frequency)
    write_raw_sphere(path, phi=phi, theta=theta, frequency=frequency, amplitude=amplitude,
                     RADIUS=RADIUS_FT, SPEED=65.0, FLIGHT_PATH_ANGLE=-6.0)
    return amplitude


def back_to_sphere(levels_60m_db, frequency_hz):
    """Undo the documented NORAH2 conversion (THEORY section 8) to the sphere's radius."""
    r_m = RADIUS_FT * 0.3048
    return (levels_60m_db + 20.0 * np.log10(fa.NORAH2_REFERENCE_DISTANCE_M / r_m)
            - MEASURED.attenuation_coefficient(frequency_hz) * r_m
            + REFERENCE.attenuation_coefficient(fa.NORAH2_BAND_CENTERS_HZ) * fa.NORAH2_REFERENCE_DISTANCE_M)


def test_round_trip_sphere_to_norah2_and_back(tmp_path):
    """Every NORAH2 direction is on the build grid, so the levels pass through unchanged,
    missing cells stay missing, and the header carries the sphere's condition."""
    amplitude = directional_levels(PHI, THETA, BANDS)
    amplitude[2, 4, :] = fa.AAM_MISSING_LEVEL           # phi -70, theta 20: a NORAH2 node
    amplitude[10, 7, 5] = np.nan                         # phi 10, theta 35: between nodes
    write_sphere(tmp_path / 'X101.nc', amplitude=amplitude)
    written = sphere_to_norah2(str(tmp_path / 'X101.nc'), str(tmp_path / 'X101.hem'),
                               measurement_atmosphere=MEASURED, minimum_level_db=-np.inf)

    hem = fa.load_norah2_hemisphere(str(tmp_path / 'X101.hem'))
    assert np.array_equal(hem['phi_deg'], fa.NORAH2_PHI_DEG)
    assert np.array_equal(hem['theta_deg'], fa.NORAH2_THETA_DEG)
    assert hem['constants']['ACSPEED'] == 65.0 and hem['constants']['GAMM'] == -6.0
    assert hem['constants']['Tm'] == pytest.approx(10.0)
    assert hem['constants']['Pm'] == pytest.approx(97500.0)
    assert hem['constants']['RHm'] == pytest.approx(55.0)

    recovered = back_to_sphere(hem['levels_db'], BANDS)
    expected = amplitude[:, ::2, :].astype(np.float32).astype(float)    # theta every 10 deg
    missing = np.zeros(expected.shape, dtype=bool)
    missing[2, 2, :] = True
    assert np.all(np.isneginf(hem['levels_db'][missing]))
    assert np.all(np.isneginf(written[missing]))
    # The file holds levels to 0.1 dB.
    assert np.allclose(recovered[~missing], expected[~missing], rtol=0.0, atol=0.05 + 1e-9)


def test_interpolates_in_linear_power_between_sphere_nodes():
    """Off-node directions are interpolated in power; outside the sphere's grid is missing."""
    phi = np.array([-90.0, -45.0, 0.0, 45.0, 90.0])
    theta = np.arange(0.0, 90.0 + 1e-9, 15.0)            # forward quarter only
    levels = np.empty((phi.size, theta.size, 1))
    levels[:, :, 0] = np.where(np.arange(phi.size) % 2 == 0, 70.0, 80.0)[:, None]
    sampled = sphere_levels_on_norah2_grid(levels, phi, theta)

    i10 = int(np.flatnonzero(fa.NORAH2_PHI_DEG == -80.0)[0])   # 10/45 of the way to -45
    j30 = int(np.flatnonzero(fa.NORAH2_THETA_DEG == 30.0)[0])
    w = 10.0 / 45.0
    expected = 10.0 * np.log10((1.0 - w) * 1e7 + w * 1e8)
    assert sampled[i10, j30, 0] == pytest.approx(expected, abs=1e-9)
    assert np.all(np.isneginf(sampled[:, fa.NORAH2_THETA_DEG > 90.0, 0]))
    assert np.all(np.isfinite(sampled[:, fa.NORAH2_THETA_DEG <= 90.0, 0]))


def test_missing_sphere_cells_carry_no_weight():
    """-999 next to a measured cell does not pull the level toward zero energy;
    a direction weighted mostly by missing cells is missing."""
    phi = np.array([-90.0, -60.0, 90.0])
    theta = np.array([0.0, 180.0])
    levels = np.full((3, 2, 1), 75.0)
    levels[1, :, 0] = fa.AAM_MISSING_LEVEL
    sampled = sphere_levels_on_norah2_grid(levels, phi, theta)[:, :, 0]
    row = {phi_deg: sampled[i] for i, phi_deg in enumerate(fa.NORAH2_PHI_DEG)}
    assert np.allclose(row[-90.0], 75.0)
    assert np.allclose(row[-80.0], 75.0)                # 2/3 measured: not pulled down
    assert np.all(np.isneginf(row[-70.0]))              # 1/3 measured: missing
    assert np.all(np.isneginf(row[-50.0]))
    assert np.allclose(row[80.0], 75.0)


def test_measurement_atmosphere_is_required(tmp_path):
    write_sphere(tmp_path / 'X101.nc')
    with pytest.raises(TypeError, match='atmosphere'):
        sphere_to_norah2(str(tmp_path / 'X101.nc'), str(tmp_path / 'X101.hem'),
                         measurement_atmosphere=None)


def test_condition_overrides_and_spreading_only(tmp_path):
    write_sphere(tmp_path / 'X101.nc')
    with pytest.warns(UserWarning, match='spreading only'):
        written = sphere_to_norah2(str(tmp_path / 'X101.nc'), str(tmp_path / 'X101.hem'),
                                   measurement_atmosphere=MEASURED, speed_knots=70.0,
                                   flight_path_angle_deg=3.0, apply_absorption_deprop=False)
    hem = fa.load_norah2_hemisphere(str(tmp_path / 'X101.hem'))
    assert hem['constants']['ACSPEED'] == 70.0 and hem['constants']['GAMM'] == 3.0
    spreading = 20.0 * np.log10(RADIUS_FT * 0.3048 / fa.NORAH2_REFERENCE_DISTANCE_M)
    source = directional_levels(PHI, THETA[::2], BANDS).astype(np.float32).astype(float)
    assert np.allclose(written, source + spreading, atol=1e-4)


def test_refuses_spheres_without_norah2_bands(tmp_path):
    write_sphere(tmp_path / 'X101.nc', frequency=np.array([20000.0, 25000.0]))
    with pytest.raises(ValueError, match='NORAH2'):
        sphere_to_norah2(str(tmp_path / 'X101.nc'), str(tmp_path / 'X101.hem'),
                         measurement_atmosphere=MEASURED)


def test_write_norah2_levels_checks_the_grid(tmp_path):
    with pytest.raises(ValueError, match='phi, theta, band'):
        fa.write_norah2_levels(str(tmp_path / 'x.hem'), BANDS, np.zeros((19, 37, 31)),
                               r_ref_m=30.48, measurement_atmosphere=MEASURED,
                               speed_knots=60.0, flight_path_angle_deg=0.0)
