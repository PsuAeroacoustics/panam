"""Band-level and layered atmospheric absorption (panam.acoustics.atmosphere) and the
options that use them: absorption_band_method on the filter-bank depropagation, the
EAA and the NORAH2 import, and absorption_profile on the depropagation and build_sphere."""

from pathlib import Path

import numpy as np
import pytest

from panam import flight_acoustics as fa
from panam.acoustics.atmosphere import (ARP5534_COEFFICIENTS, Atmosphere, arp5534_band_attenuation,
                                        band_absorption_db, exact_midband_frequency,
                                        layered_attenuation_db)

T, P, RH = 298.15, 101.325, 70.0
ATMOSPHERE = Atmosphere(temperature=T, pressure=P, relative_humidity=RH)
CENTERS = np.array([500.0, 1000.0, 2000.0, 4000.0, 8000.0])
P_REF = 2.0e-5


def uniform_profile(temperature=T, humidity=RH, pressure=P, altitudes=(0.0, 500.0, 2000.0)):
    n = len(altitudes)
    return dict(altitude_ft=np.array(altitudes, dtype=float), temperature_k=np.full(n, temperature),
                relative_humidity=np.full(n, humidity), pressure_kpa=np.full(n, pressure))


# --- band_absorption_db ------------------------------------------------------------------

def test_center_is_the_coefficient_at_the_center_times_the_distance():
    distance = np.array([10.0, 300.0, 1000.0])
    result = band_absorption_db(distance, CENTERS, T, P, RH)
    assert result.shape == (CENTERS.size, distance.size)
    expected = ATMOSPHERE.attenuation_coefficient(CENTERS)[:, None] * distance
    np.testing.assert_allclose(result, expected, rtol=1e-14)
    assert band_absorption_db(1000.0, CENTERS, T, P, RH).shape == (CENTERS.size,)


def test_exact_midband_frequency_is_the_base_ten_frequency_of_the_band():
    nominal = np.array([50.0, 1000.0, 1250.0, 8000.0])
    np.testing.assert_allclose(exact_midband_frequency(nominal), 10.0 ** (np.array([17, 30, 31, 39]) / 10.0))
    base2 = 1000.0 * 2.0 ** (np.arange(-3, 4) / 3.0)
    np.testing.assert_allclose(exact_midband_frequency(base2), 10.0 ** (np.arange(27, 34) / 10.0))
    with pytest.raises(ValueError, match='positive'):
        exact_midband_frequency([0.0, 100.0])


def test_arp5534_is_the_published_two_branch_fit():
    """Rickley et al. (2012), Eqs. 16-17: A dt (1 + B (C - D dt))^E below 150 dB,
    F + G dt above; the branches meet at 150 dB within 0.05 dB."""
    c = ARP5534_COEFFICIENTS
    for dt in (0.5, 10.0, 50.0, 149.0):
        expected = c['A'] * dt * (1.0 + c['B'] * (c['C'] - c['D'] * dt)) ** c['E']
        assert arp5534_band_attenuation(dt) == pytest.approx(expected, rel=1e-14)
    for dt in (150.0, 400.0):
        assert arp5534_band_attenuation(dt) == pytest.approx(c['F'] + c['G'] * dt, rel=1e-14)
    below, at = arp5534_band_attenuation([150.0 - 1e-9, 150.0])
    assert abs(below - at) < 0.05
    assert arp5534_band_attenuation(0.0) == 0.0
    # Small attenuation: about 2 % over the mid-band pure-tone value.
    assert arp5534_band_attenuation(1e-6) / 1e-6 == pytest.approx(1.0215, abs=1e-3)


def test_arp5534_band_absorption_evaluates_the_tone_at_the_exact_midband():
    distance = 800.0
    result = band_absorption_db(distance, CENTERS, T, P, RH, method='arp5534')
    tone = ATMOSPHERE.attenuation_coefficient(exact_midband_frequency(CENTERS)) * distance
    np.testing.assert_allclose(result, arp5534_band_attenuation(tone), rtol=1e-14)
    assert not np.allclose(result, band_absorption_db(distance, CENTERS, T, P, RH))


def test_integrated_agrees_with_arp5534_over_its_fitted_range():
    """ARP 5534 is a fit to the spectrum-integration method for spectra normalized to
    3 dB per band.  Over 263-303 K, 10-95 % RH and exact centers 100 Hz to 10 kHz,
    the integration with that slope agrees within 0.3 dB where the ARP attenuation is
    at most 20 dB and within 3 dB where it is at most 50 dB."""
    centers = 10.0 ** (np.arange(20, 41) / 10.0)
    worst_20 = worst_50 = 0.0
    for temperature in (263.15, 283.15, 303.15):
        for humidity in (10.0, 30.0, 70.0, 95.0):
            for distance in (50.0, 300.0, 1000.0, 2000.0, 3000.0):
                arp = band_absorption_db(distance, centers, temperature, P, humidity, method='arp5534')
                integrated = band_absorption_db(distance, centers, temperature, P, humidity,
                                                method='integrated', spectrum_slope_db=3.0)
                error = np.abs(integrated - arp)
                worst_20 = max(worst_20, error[arp <= 20.0].max(initial=0.0))
                worst_50 = max(worst_50, error[arp <= 50.0].max(initial=0.0))
    assert worst_20 < 0.3
    assert worst_50 < 3.0


def test_integrated_tends_to_the_band_mean_at_short_range_and_below_the_tone_at_long():
    short = band_absorption_db(1.0, CENTERS, T, P, RH, method='integrated')
    center = band_absorption_db(1.0, CENTERS, T, P, RH)
    np.testing.assert_allclose(short, center, rtol=0.05)
    # Far out, the band's least-attenuated (low) side carries the band: less than the
    # pure tone at the center, and increasingly so.
    far = band_absorption_db(3000.0, CENTERS, T, P, RH, method='integrated')
    far_center = band_absorption_db(3000.0, CENTERS, T, P, RH)
    assert far[-1] < 0.7 * far_center[-1]
    assert np.all(band_absorption_db(0.0, CENTERS, T, P, RH, method='integrated') == pytest.approx(0.0, abs=1e-12))


def test_integrated_weights_the_band_by_the_spectrum_slope():
    """A spectrum rising with frequency puts more weight on the band's upper, more
    attenuated side; a per-band slope array works as a scalar does."""
    flat = band_absorption_db(500.0, CENTERS, T, P, RH, method='integrated')
    rising = band_absorption_db(500.0, CENTERS, T, P, RH, method='integrated', spectrum_slope_db=5.0)
    falling = band_absorption_db(500.0, CENTERS, T, P, RH, method='integrated', spectrum_slope_db=-5.0)
    assert np.all(rising > flat) and np.all(falling < flat)
    mixed = band_absorption_db(500.0, CENTERS, T, P, RH, method='integrated',
                               spectrum_slope_db=np.array([5.0, 0.0, -5.0, 0.0, 0.0]))
    np.testing.assert_allclose(mixed[[0, 1, 2]], [rising[0], flat[1], falling[2]], rtol=1e-12)


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError, match='band method'):
        band_absorption_db(100.0, CENTERS, T, P, RH, method='edge')


# --- layered_attenuation_db --------------------------------------------------------------

RAYS = dict(source_altitude_ft=np.array([400.0, 1500.0, 300.0, 2500.0]),
            receiver_altitude_ft=np.array([0.0, 0.0, 300.0, 50.0]),
            path_length_ft=np.array([500.0, 1800.0, 600.0, 2600.0]))


@pytest.mark.parametrize('method', ['center', 'arp5534', 'integrated'])
def test_a_uniform_profile_is_the_uniform_atmosphere(method):
    layered = layered_attenuation_db(uniform_profile(), RAYS, CENTERS, method=method)
    uniform = band_absorption_db(RAYS['path_length_ft'] * 0.3048, CENTERS, T, P, RH, method=method)
    assert layered.shape == (CENTERS.size, 4)
    np.testing.assert_allclose(layered, uniform, rtol=1e-10)
    one_level = uniform_profile(altitudes=(100.0,))
    np.testing.assert_allclose(layered_attenuation_db(one_level, RAYS, CENTERS, method=method), uniform,
                               rtol=1e-10)


def test_a_layered_path_takes_the_altitude_mean_of_the_coefficient():
    """alpha is linear in altitude between profile levels, so a ray between them takes
    the trapezoid mean; a level ray the local value; beyond the profile, the end value."""
    profile = dict(altitude_ft=np.array([0.0, 1000.0]), temperature_k=np.array([298.15, 288.15]),
                   relative_humidity=np.array([20.0, 80.0]), pressure_kpa=np.array([101.325, 97.7]))
    bottom = Atmosphere(298.15, 101.325, 20.0).attenuation_coefficient(CENTERS)
    top = Atmosphere(288.15, 97.7, 80.0).attenuation_coefficient(CENTERS)
    ray = dict(source_altitude_ft=np.array([1000.0, 500.0, 3000.0, 1000.0]),
               receiver_altitude_ft=np.array([0.0, 500.0, 1000.0, 2000.0]),
               path_length_ft=np.array([1200.0, 700.0, 2100.0, 1100.0]))
    result = layered_attenuation_db(profile, ray, CENTERS)
    meters = ray['path_length_ft'] * 0.3048
    np.testing.assert_allclose(result[:, 0], 0.5 * (bottom + top) * meters[0], rtol=1e-12)
    np.testing.assert_allclose(result[:, 1], 0.5 * (bottom + top) * meters[1], rtol=1e-12)
    np.testing.assert_allclose(result[:, 2], top * meters[2], rtol=1e-12)
    np.testing.assert_allclose(result[:, 3], top * meters[3], rtol=1e-12)
    # Partway: from 0 to 250 ft the mean is alpha at 125 ft.
    part = layered_attenuation_db(profile, dict(source_altitude_ft=250.0, receiver_altitude_ft=0.0,
                                                path_length_ft=250.0), CENTERS)
    np.testing.assert_allclose(part, (bottom + 0.125 * (top - bottom)) * 250.0 * 0.3048, rtol=1e-12)
    # Reciprocal: swapping the ends changes nothing.
    swapped = dict(ray, source_altitude_ft=ray['receiver_altitude_ft'], receiver_altitude_ft=ray['source_altitude_ft'])
    np.testing.assert_allclose(layered_attenuation_db(profile, swapped, CENTERS), result, rtol=1e-12)


def test_arp5534_through_a_profile_applies_to_the_layered_tone():
    profile = dict(altitude_ft=np.array([0.0, 1000.0]), temperature_k=np.array([298.15, 288.15]),
                   relative_humidity=np.array([20.0, 80.0]), pressure_kpa=np.array([101.325, 97.7]))
    tone = layered_attenuation_db(profile, RAYS, exact_midband_frequency(CENTERS))
    np.testing.assert_allclose(layered_attenuation_db(profile, RAYS, CENTERS, method='arp5534'),
                               arp5534_band_attenuation(tone), rtol=1e-12)


def test_a_bad_profile_is_refused():
    with pytest.raises(ValueError, match='needs'):
        layered_attenuation_db({'altitude_ft': [0.0]}, RAYS, CENTERS)
    with pytest.raises(ValueError, match='increase'):
        layered_attenuation_db(uniform_profile(altitudes=(0.0, 0.0, 10.0)), RAYS, CENTERS)
    with pytest.raises(ValueError, match='kelvin'):
        layered_attenuation_db(uniform_profile(temperature=25.0), RAYS, CENTERS)
    with pytest.raises(ValueError, match='kPa'):
        layered_attenuation_db(uniform_profile(pressure=101325.0), RAYS, CENTERS)


# --- depropagate_hemisphere ------------------------------------------------------------

def _hover(height_ft=400.0, **options):
    """White noise heard from a source hovering ``height_ft`` above a microphone."""
    rng = np.random.default_rng(3)
    fs, duration = 8000.0, 20.0
    t = np.arange(0.0, duration, 1.0 / fs)
    pressure = rng.standard_normal(t.size)
    track_time = np.linspace(2.0, duration - 3.0, 60)
    arguments = dict(
        mic_locations=np.zeros((1, 3)), pressure=pressure[None, :], time=t,
        track_time=track_time, track_position=np.tile([0.0, 0.0, height_ft], (track_time.size, 1)),
        track_velocity=np.tile([1e-6, 0.0, 0.0], (track_time.size, 1)),
        r_ref=100.0, freq_range=(0.0, 3500.0), window_time=0.5, window_overlap=0.5,
        azi_step=30.0, elv_step=10.0, rmax=15.0, third_octave=True, third_octave_fmin=200.0,
        apply_absorption_deprop=True, atmosphere=ATMOSPHERE, return_scattered=True)
    arguments.update(options)
    return fa.depropagate_hemisphere(**arguments)


@pytest.mark.parametrize('method', ['arp5534', 'integrated'])
def test_filter_bank_bands_take_the_band_method(method):
    """The correction is the band attenuation over the path less that over r_ref, in
    place of alpha(fc) (r - r_ref); everything else is unchanged."""
    center = _hover(third_octave_method='filter_bank')
    banded = _hover(third_octave_method='filter_bank', absorption_band_method=method)
    centers = center['scattered']['third_octave']['band_centers_hz']
    r_m = center['scattered']['range'] * 0.3048
    r_ref_m = 100.0 * 0.3048
    expected = (band_absorption_db(r_m, centers, T, P, RH, method=method)
                - band_absorption_db(r_ref_m, centers, T, P, RH, method=method)[:, None]
                - ATMOSPHERE.attenuation_coefficient(centers)[:, None] * (r_m - r_ref_m))
    difference = banded['scattered']['third_octave']['bands_db'] - center['scattered']['third_octave']['bands_db']
    np.testing.assert_allclose(difference, expected, atol=1e-9)
    assert np.abs(expected).max() > 1e-3
    np.testing.assert_array_equal(banded['oaspl_db'], center['oaspl_db'])
    assert banded['metadata']['absorption_band_method'] == method


def test_band_method_needs_absorption_and_the_filter_bank():
    with pytest.raises(ValueError, match='filter_bank'):
        _hover(absorption_band_method='arp5534')
    with pytest.raises(ValueError, match='apply_absorption_deprop'):
        _hover(third_octave_method='filter_bank', absorption_band_method='arp5534',
               apply_absorption_deprop=False)
    with pytest.raises(ValueError, match='absorption_band_method must be'):
        _hover(third_octave_method='filter_bank', absorption_band_method='edge')


@pytest.mark.parametrize('third_octave_method', ['fft', 'filter_bank'])
def test_a_uniform_profile_reproduces_the_uniform_atmosphere(third_octave_method):
    plain = _hover(third_octave_method=third_octave_method)
    layered = _hover(third_octave_method=third_octave_method, absorption_profile=uniform_profile())
    np.testing.assert_allclose(layered['scattered']['third_octave']['bands_db'],
                               plain['scattered']['third_octave']['bands_db'], atol=1e-9)
    np.testing.assert_allclose(layered['oaspl_db'], plain['oaspl_db'], atol=1e-9, equal_nan=True)
    assert layered['metadata']['absorption_profile'] is True
    assert plain['metadata']['absorption_profile'] is False


def test_a_layered_profile_undoes_each_paths_own_absorption():
    """Filter-bank bands: the correction is the path's layered attenuation less alpha(fc)
    r_ref in the sphere's atmosphere, so against the uniform build each band differs by
    the layered attenuation less the uniform one over the path."""
    profile = dict(altitude_ft=np.array([0.0, 1000.0]), temperature_k=np.array([298.15, 288.15]),
                   relative_humidity=np.array([20.0, 80.0]), pressure_kpa=np.array([101.325, 97.7]))
    plain = _hover(third_octave_method='filter_bank')
    layered = _hover(third_octave_method='filter_bank', absorption_profile=profile)
    centers = plain['scattered']['third_octave']['band_centers_hz']
    range_ft = plain['scattered']['range']
    height_ft = plain['scattered']['source_height']
    expected = (layered_attenuation_db(profile, dict(source_altitude_ft=height_ft, receiver_altitude_ft=0.0,
                                                     path_length_ft=range_ft), centers)
                - ATMOSPHERE.attenuation_coefficient(centers)[:, None] * range_ft * 0.3048)
    difference = layered['scattered']['third_octave']['bands_db'] - plain['scattered']['third_octave']['bands_db']
    np.testing.assert_allclose(difference, expected, atol=1e-9)
    assert np.abs(expected).max() > 0.1
    # The FFT path corrects every bin through the profile; OASPL moves with it.
    fft = _hover(absorption_profile=profile)
    assert not np.allclose(fft['oaspl_db'], _hover()['oaspl_db'], equal_nan=True)


def test_a_profile_needs_absorption_and_is_checked():
    with pytest.raises(ValueError, match='apply_absorption_deprop'):
        _hover(absorption_profile=uniform_profile(), apply_absorption_deprop=False)
    with pytest.raises(ValueError, match='kelvin'):
        _hover(absorption_profile=uniform_profile(temperature=25.0))


# --- build_sphere -------------------------------------------------------------------------

def test_build_sphere_passes_the_profile_through(tmp_path):
    from panam import noise_abatement_2017 as na
    from test_noise_abatement_2017 import _archive
    _archive(tmp_path)
    test = na.NoiseAbatementTest('AS350B3', root=str(tmp_path))
    atmosphere = Atmosphere(293.15, 101.325, 20.0)
    levels = {}
    for name, profile in (('plain', None),
                          ('uniform', uniform_profile(293.15, 20.0, 101.325)),
                          ('humid', uniform_profile(293.15, 90.0, 101.325))):
        record = na.build_sphere(test, '289108', str(tmp_path / (name + '.nc')), board_correction='flat',
                                 atmosphere=atmosphere, absorption_profile=profile)
        levels[name] = fa.mask_missing_levels(fa.load_nc_sphere(record['output'])[0])
    np.testing.assert_allclose(levels['uniform'], levels['plain'], atol=1e-6)
    assert not np.allclose(levels['humid'], levels['plain'], atol=1e-3)


# --- the EAA and the NORAH2 import ----------------------------------------------------

@pytest.mark.parametrize('method', ['arp5534', 'integrated'])
def test_eaa_uses_the_band_method(method):
    rng = np.random.default_rng(5)
    frequency = np.array([250.0, 1000.0, 4000.0])
    levels = rng.uniform(50.0, 90.0, (3, 4, 3))
    spla, eaa = fa.spla_and_eaa_from_spectrum(levels, frequency, 1000.0, ATMOSPHERE,
                                              absorption_band_method=method)
    a_weight = fa.dBAw(frequency)
    attenuation = band_absorption_db(1000.0, frequency, T, P, RH, method=method)
    expected_spla = 10.0 * np.log10(np.sum(10.0 ** (0.1 * (levels + a_weight)), axis=-1))
    attenuated = 10.0 * np.log10(np.sum(10.0 ** (0.1 * (levels + a_weight - attenuation)), axis=-1))
    np.testing.assert_allclose(spla, expected_spla, atol=1e-10)
    np.testing.assert_allclose(eaa, expected_spla - attenuated, atol=1e-10)
    _, eaa_center = fa.spla_and_eaa_from_spectrum(levels, frequency, 1000.0, ATMOSPHERE)
    assert not np.allclose(eaa, eaa_center)


def test_norah2_levels_take_the_band_method():
    hemisphere = {'title': 'x', 'frequency_hz': CENTERS, 'levels_db': np.full((2, 3, CENTERS.size), 70.0),
                  'constants': {'POLDIST': 60.0, 'TAMB': T, 'RELHUM': RH, 'PAMB': 101325.0}}
    center = fa.norah2_levels_at_radius(hemisphere, 30.48)
    arp = fa.norah2_levels_at_radius(hemisphere, 30.48, absorption_band_method='arp5534')
    absorption = np.diff(band_absorption_db(np.array([30.48, 60.0]), CENTERS, T, P, RH, method='arp5534'),
                         axis=1)[:, 0]
    np.testing.assert_allclose(arp[0, 0], 70.0 + 20.0 * np.log10(60.0 / 30.48) + absorption, atol=1e-12)
    np.testing.assert_allclose(center[0, 0], 70.0 + 20.0 * np.log10(60.0 / 30.48)
                               + ATMOSPHERE.attenuation_coefficient(CENTERS) * (60.0 - 30.48), atol=1e-12)


# --- numpy error state ----------------------------------------------------------------

def test_numpy_warnings_are_suppressed_locally():
    """AGENTS.md: np.errstate, never np.seterr, which changes the process-wide state."""
    for path in Path(fa.__file__).parent.rglob('*.py'):
        assert 'np.seterr' not in path.read_text(), path.name
    before = np.geterr()
    spectrum = np.full((1, 3, 2), -np.inf)
    fa.average_fore_and_aft(spectrum)
    fa.spla_and_eaa_from_spectrum(spectrum, np.array([500.0, 1000.0]), 1000.0, ATMOSPHERE)
    assert np.geterr() == before
