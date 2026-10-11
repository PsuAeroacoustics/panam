"""synthesis: forward synthesis of band histories and the leave-microphones-out check."""

import numpy as np
import pytest

import flight_acoustics as fa
import refracted_rays as rr
import synthesis as sy
import synthetic_flyover as sf
from panam_acoustics.atmosphere import Atmosphere

C = 1125.0                      # ft/s
FS = 4000.0                     # Hz
WINDOW = 0.25                   # s
LINE = np.column_stack([np.zeros(7), np.linspace(-600.0, 600.0, 7), np.zeros(7)])
CENTERS = fa.BASE2_BAND_CENTERS[(fa.BASE2_BAND_CENTERS >= 90.0) & (fa.BASE2_BAND_CENTERS <= 1600.0)]
DEPROPAGATION = dict(speed_of_sound=C, r_ref=100.0, freq_range=(0.0, 2000.0), window_time=WINDOW, point_stride=2,
                     azi_step=15.0, elv_step=10.0, rmax=20.0, third_octave=True, third_octave_fmin=80.0)


def _sphere(levels_db, centers=CENTERS, step=5.0, r_ref=100.0):
    """A hemisphere dict on a regular UMAPR grid; ``levels_db(azimuth, elevation)`` -> (Nb, ...) dB."""
    azimuth, elevation = np.arange(0.0, 360.0 + 1e-9, step), np.arange(0.0, 90.0 + 1e-9, step)
    a, e = np.meshgrid(azimuth, elevation)
    return dict(azi_grid_deg=azimuth, elv_grid_deg=elevation, metadata=dict(r_ref=r_ref, length_units='ft'),
                third_octave=dict(band_centers_hz=np.asarray(centers, dtype=float),
                                  bands_db=np.broadcast_to(levels_db(a, e), (len(centers),) + a.shape).copy()))


def _uniform(level_db=80.0, **kw):
    return _sphere(lambda a, e: np.full(a.shape, level_db), **kw)


def _hover(height=300.0, duration=4.0):
    """A source fixed at (0, 0, height) ft; its nose points along +x."""
    t = np.linspace(0.0, duration, 41)
    return dict(track_time=t, track_position=np.tile([0.0, 0.0, height], (t.size, 1)),
                track_velocity=np.zeros((t.size, 3)), track_nose=np.tile([1.0, 0.0, 0.0], (t.size, 1)))


def _energy_ratio_db(a_db, b_db):
    """10 lg of the summed power of a over that of b, over the cells finite in both."""
    ok = np.isfinite(a_db) & np.isfinite(b_db)
    return 10.0 * np.log10(np.sum(10.0 ** (a_db[ok] / 10.0)) / np.sum(10.0 ** (b_db[ok] / 10.0)))


# --- the forward model, term by term ---------------------------------------------------------------------------

def test_spreading_from_the_sphere_radius():
    mic = np.array([[400.0, 0.0, 0.0]])
    out = sy.synthesize_band_histories(_uniform(80.0), _hover(), mic, times=[2.0, 3.0], speed_of_sound=C)
    r = np.hypot(400.0, 300.0)
    np.testing.assert_allclose(out['bands_db'], 80.0 - 20.0 * np.log10(r / 100.0), atol=1e-9)
    np.testing.assert_allclose(out['range'], r, rtol=1e-12)
    np.testing.assert_allclose(out['emission_time'], np.array([[2.0, 3.0]]) - r / C, atol=1e-9)
    np.testing.assert_allclose(out['elevation_deg'], np.degrees(np.arctan2(300.0, 400.0)), atol=1e-9)


def test_absorption_beyond_the_sphere_radius():
    weather = Atmosphere(temperature=288.15, pressure=101.325, relative_humidity=70.0)
    mic = np.array([[0.0, 0.0, -700.0]])                    # 1000 ft straight below
    plain = sy.synthesize_band_histories(_uniform(), _hover(), mic, times=[3.0], speed_of_sound=C)
    absorbed = sy.synthesize_band_histories(_uniform(), _hover(), mic, times=[3.0], speed_of_sound=C,
                                            apply_absorption=True, atmosphere=weather)
    alpha = np.asarray(weather.attenuation_coefficient(CENTERS), dtype=float)       # dB/m
    expected = alpha * (1000.0 - 100.0) * 0.3048
    np.testing.assert_allclose(plain['bands_db'][0, :, 0] - absorbed['bands_db'][0, :, 0], expected, rtol=1e-9)


def test_receiver_response_is_added():
    mic = np.array([[100.0, 0.0, 0.0]])
    seen = []

    def response(im, bands, offset):
        seen.append((im, offset.copy()))
        return np.full((bands.size, offset.shape[0]), 6.0)
    plain = sy.synthesize_band_histories(_uniform(), _hover(), mic, times=[2.0], speed_of_sound=C)
    board = sy.synthesize_band_histories(_uniform(), _hover(), mic, times=[2.0], speed_of_sound=C,
                                         receiver_response_db=response)
    np.testing.assert_allclose(board['bands_db'] - plain['bands_db'], 6.0, atol=1e-12)
    assert seen[0][0] == 0
    np.testing.assert_allclose(seen[0][1], [[-100.0, 0.0, 300.0]])     # source minus microphone, ft


def test_directions_follow_umapr():
    """Azimuth 180 is ahead and 90 starboard (-y for a source flying along +x, z up)."""
    def starboard_ahead(a, e):
        return 80.0 + 10.0 * (np.sin(np.radians(a)) > 0.5) + 20.0 * (np.cos(np.radians(a - 180.0)) > 0.5)
    sphere = _sphere(starboard_ahead)
    mics = np.array([[0.0, -400.0, 0.0], [0.0, 400.0, 0.0], [400.0, 0.0, 0.0], [-400.0, 0.0, 0.0]])
    out = sy.synthesize_band_histories(sphere, _hover(height=50.0), mics, times=[3.0], speed_of_sound=C)
    level = out['bands_db'][:, 0, 0] + 20.0 * np.log10(np.hypot(400.0, 50.0) / 100.0)
    np.testing.assert_allclose(level, [90.0, 80.0, 100.0, 80.0], atol=1e-6)


def test_missing_outside_the_track_and_the_coverage():
    sphere = _uniform()
    sphere['third_octave']['bands_db'][:, sphere['elv_grid_deg'] > 45.0, :] = np.nan
    mics = np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]])
    out = sy.synthesize_band_histories(sphere, _hover(), mics, times=[0.1, 2.0], speed_of_sound=C)
    assert np.all(np.isnan(out['bands_db'][:, :, 0]))       # emitted before the track starts
    assert np.all(np.isnan(out['bands_db'][0, :, 1]))       # overhead: no data there
    assert np.all(np.isfinite(out['bands_db'][1, :, 1]))    # 17 deg below the horizon


def test_doppler_band_shift():
    power = np.vstack([np.arange(1.0, CENTERS.size + 1.0)] * 3).T                # (Nb, 3)
    same = sy.doppler_shift_band_power(power, CENTERS, np.ones(3))
    np.testing.assert_allclose(same, power, rtol=1e-12)
    # A third of an octave up moves every band whole to the next; the lowest has no source.
    up = sy.doppler_shift_band_power(power, CENTERS, np.full(3, 2.0 ** (1.0 / 3.0)))
    assert np.all(np.isnan(up[0]))
    np.testing.assert_allclose(up[1:], power[:-1], rtol=1e-9)
    # Any shift conserves the power carried into fully covered bands.
    d = np.array([1.07])
    shifted = sy.doppler_shift_band_power(power[:, :1], CENTERS, d)
    lower, upper = fa.third_octave_band_edges(CENTERS)
    density = power[:, 0] / ((upper - lower) * d)                                 # per Hz, received
    np.testing.assert_allclose(shifted[1:, 0],
                               [np.sum(density * np.clip(np.minimum(hi, upper * d) - np.maximum(lo, lower * d),
                                                         0.0, None)) for lo, hi in zip(lower[1:], upper[1:])],
                               rtol=1e-12)
    assert np.isnan(shifted[0, 0])          # its lower part was emitted below the lowest band


def test_synthesis_applies_the_doppler_shift_when_asked():
    track = dict(track_time=np.linspace(0.0, 6.0, 61), track_position=np.column_stack(
        [150.0 * (np.linspace(0.0, 6.0, 61) - 3.0), np.zeros(61), np.full(61, 300.0)]),
        track_velocity=np.tile([150.0, 0.0, 0.0], (61, 1)))
    sphere = _sphere(lambda a, e: 60.0 + 3.0 * np.arange(CENTERS.size)[:, None, None])
    mic = np.zeros((1, 3))
    plain = sy.synthesize_band_histories(sphere, track, mic, times=[2.5], speed_of_sound=C)
    shifted = sy.synthesize_band_histories(sphere, track, mic, times=[2.5], speed_of_sound=C, doppler_shift=True)
    d = shifted['doppler'][0, 0]
    assert d > 1.0                                                                # approaching
    power = 10.0 ** (plain['bands_db'][0] / 10.0)
    np.testing.assert_allclose(shifted['bands_db'][0], fa.power_to_db(
        sy.doppler_shift_band_power(power, CENTERS, [d])), atol=1e-9)


def test_still_wind_ray_model_matches_straight_lines():
    scenario = sf.synthesize_pass(LINE[:2], duration=6.0, noise=0.0)[0]
    times = np.linspace(1.0, 5.0, 9)
    straight = sy.synthesize_band_histories(_uniform(), scenario, LINE[:2], times=times, speed_of_sound=C)
    still = sy.synthesize_band_histories(_uniform(), scenario, LINE[:2], times=times, speed_of_sound=C,
                                         ray_model=rr.uniform_wind_ray_model(C, (0.0, 0.0)))
    for key in ('bands_db', 'emission_time', 'azimuth_deg', 'elevation_deg'):
        np.testing.assert_allclose(still[key], straight[key], atol=1e-7)


def test_wind_ray_emission_time_solves_the_travel_time():
    wind = rr.uniform_wind_ray_model(C, (30.0, 20.0))
    scenario = sf.synthesize_pass(LINE[:1], duration=6.0, noise=0.0)[0]
    times = np.linspace(1.5, 5.0, 8)
    out = sy.synthesize_band_histories(_uniform(), scenario, LINE[:1], times=times, speed_of_sound=C, ray_model=wind)
    source = np.column_stack([np.interp(out['emission_time'][0], scenario['track_time'],
                                        scenario['track_position'][:, k]) for k in range(3)])
    tau = wind(source, LINE[:1])['travel_time'][:, 0]
    np.testing.assert_allclose(out['emission_time'][0] + tau, times, atol=1e-8)
    np.testing.assert_allclose(out['spreading_range'][0], C * tau, rtol=1e-9)


# --- against the synthetic flyover ----------------------------------------------------------------------------

def test_true_directivity_reproduces_the_synthetic_band_histories():
    """The synthetic source's broadband noise is white at the microphone, at
    (S g / r)^2 broadband^2 / (fs / 2) per Hz, so its hemisphere holds that
    density times each band's brick-wall width at r_ref = 100 ft."""
    scenario = sf.synthesize_pass(LINE, duration=12.0, tones=(), noise=0.0, directivity_db=sf.example_directivity_db,
                                  seed=1, sampling_rate=FS, speed_of_sound=C)[0]
    f = np.fft.rfftfreq(fa.frame_length(WINDOW, FS), 1.0 / FS)
    lower, upper = fa.third_octave_band_edges(CENTERS)
    width = np.array([np.sum((f >= lo) & (f < hi)) for lo, hi in zip(lower, upper)]) * (f[1] - f[0])
    base = 10.0 * np.log10((100.0 / 100.0) ** 2 * 0.2 ** 2 / (FS / 2.0) * width / fa.P_REF ** 2)

    def ratios(directivity):
        sphere = _sphere(lambda a, e: base[:, None, None] + directivity(a, e)[None])
        out = []
        for im in range(LINE.shape[0]):
            frames, measured = sy.measured_band_histories(scenario['pressure'][im], scenario['time'], CENTERS,
                                                          window_time=WINDOW, window_overlap=0.5)
            synthesized = sy.synthesize_band_histories(sphere, scenario, LINE[im:im + 1], times=frames,
                                                       speed_of_sound=C)['bands_db'][0]
            wide = CENTERS >= 250.0                     # enough bins per band to average the noise
            out.append(_energy_ratio_db(synthesized[wide], measured[wide]))
        return np.array(out)

    np.testing.assert_allclose(ratios(sf.example_directivity_db), 0.0, atol=0.15)
    # The check sees a mirrored source: port and starboard swap by 2-3 dB at the ends of the line.
    mirrored = ratios(lambda a, e: sf.example_directivity_db(360.0 - a, e))
    assert mirrored[0] < -1.5 and mirrored[-1] > 1.5


def test_depropagate_then_synthesize_round_trip():
    scenario, ambient = sf.synthesize_pass(LINE, duration=12.0, directivity_db=sf.example_directivity_db, seed=1)
    hemisphere = fa.depropagate_hemisphere(**scenario, **DEPROPAGATION, ambient_pressure=ambient)
    centers = hemisphere['third_octave']['band_centers_hz']
    for im in range(LINE.shape[0]):
        frames, measured = sy.measured_band_histories(scenario['pressure'][im], scenario['time'], centers,
                                                      window_time=WINDOW, window_overlap=0.5,
                                                      ambient_pressure=ambient[im])
        synthesized = sy.synthesize_band_histories(hemisphere, scenario, LINE[im:im + 1], times=frames,
                                                   speed_of_sound=C)['bands_db'][0]
        assert np.isfinite(synthesized).sum() > 0.8 * synthesized.size
        assert abs(_energy_ratio_db(synthesized, measured)) < 0.3


# --- leaving microphones out ----------------------------------------------------------------------------------

def test_leave_mics_out_ranks_the_gridding_radius():
    scenario, ambient = sf.synthesize_pass(LINE, duration=12.0, directivity_db=sf.example_directivity_db, seed=1)
    narrow = sy.leave_mics_out(**scenario, ambient_pressure=ambient, held_out=[[1], [3], [5]], **DEPROPAGATION)
    wide = sy.leave_mics_out(**scenario, ambient_pressure=ambient, held_out=[[1], [3], [5]],
                             **dict(DEPROPAGATION, rmax=60.0))
    assert [fold['kept'] for fold in narrow['folds']][0] == [0, 2, 3, 4, 5, 6]
    assert narrow['statistics']['count'].shape == narrow['band_centers_hz'].shape
    assert abs(narrow['overall']['median_db']) < 0.5
    assert narrow['overall']['rms_db'] < 2.5
    assert wide['overall']['rms_db'] > narrow['overall']['rms_db'] + 0.5


def test_leave_mics_out_passes_each_microphone_its_own_index():
    scenario, ambient = sf.synthesize_pass(LINE[:3], duration=8.0, directivity_db=sf.example_directivity_db, seed=2)
    seen = set()

    def response(im, bands, offset):
        seen.add(im)
        return np.zeros((bands.size, offset.shape[0]))
    sy.leave_mics_out(**scenario, ambient_pressure=ambient, held_out=[[0]], receiver_response_db=response,
                      **DEPROPAGATION)
    assert seen == {0, 1, 2}


def test_leave_mics_out_drops_frames_the_depropagation_drops():
    scenario, ambient = sf.synthesize_pass(LINE, duration=12.0, directivity_db=sf.example_directivity_db, seed=1)
    every = sy.leave_mics_out(**scenario, ambient_pressure=ambient, held_out=[[0]], **DEPROPAGATION)
    steep = sy.leave_mics_out(**scenario, ambient_pressure=ambient, held_out=[[0]],
                              **dict(DEPROPAGATION, min_elevation_deg=20.0))
    assert 0 < steep['overall']['count'] < every['overall']['count']


def test_residual_statistics():
    stats = sy.residual_statistics(np.array([[1.0, -1.0, np.nan, 3.0], [np.nan, np.nan, -np.inf, np.nan]]))
    assert stats['count'].tolist() == [3, 0]
    assert stats['mean_db'][0] == pytest.approx(1.0)
    assert stats['median_db'][0] == pytest.approx(1.0)
    assert stats['rms_db'][0] == pytest.approx(np.sqrt(11.0 / 3.0))
    assert stats['std_db'][0] == pytest.approx(2.0)
    assert np.isnan(stats['mean_db'][1]) and np.isnan(stats['std_db'][1])
