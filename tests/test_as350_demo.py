"""The AS350 demo's inputs to depropagation, on a synthetic track file."""

import numpy as np
import pytest

import AS350_289108_demo as demo
import flight_acoustics as fa
import noise_abatement_2017 as na


def _write_descent(basepath, rate_fps=15.0, n=2000):
    """A 2017-format track descending at ``rate_fps``, its vz stored positive down."""
    tracking = basepath / 'Tracking'
    tracking.mkdir(parents=True)
    t = np.arange(n) * 0.02
    x = -3500.0 + 100.0 * t
    z = 1000.0 - rate_fps * t
    columns = dict(utcsec=1000.0 + t, lat=np.zeros(n), lon=np.zeros(n), alt=z, heading=np.zeros(n),
                   pitch=np.zeros(n), roll=np.zeros(n), x=x, y=np.zeros(n), z=z,
                   vx=np.full(n, 100.0), vy=np.zeros(n), vz=np.full(n, rate_fps))
    rows = np.column_stack(list(columns.values()))
    np.savetxt(tracking / '289108AC.csv', rows, delimiter=',', header=','.join(columns), comments='')


def test_the_velocity_handed_to_depropagation_is_positive_up(tmp_path):
    """load_NASA_track returns the file's vz, which is positive DOWN; read raw,
    a descent's flight path angle came out as a climb."""
    _write_descent(tmp_path)
    _, filtered = demo.load_track(tmp_path)
    source, velocity = demo.track_kinematics(filtered)
    assert source.shape == velocity.shape == (filtered['x'].size, 3)
    np.testing.assert_allclose(velocity[:, 2], -15.0)
    fpa = np.degrees(np.arctan2(velocity[:, 2], np.hypot(velocity[:, 0], velocity[:, 1])))
    assert np.median(fpa) == pytest.approx(-np.degrees(np.arctan(0.15)))


def test_the_speed_of_sound_comes_from_the_atmosphere():
    """It was a fixed 1135 ft/s beside a 20 C atmosphere whose own is ~1126."""
    atmosphere, speed_of_sound = demo.demo_atmosphere()
    assert speed_of_sound == pytest.approx(atmosphere.soundspeed / 0.3048)
    assert speed_of_sound == na.sound_speed_ft_s(atmosphere)
    assert 1120.0 < speed_of_sound < 1130.0


def test_the_saved_band_limits_match_the_bands_summed():
    """The full-band OASPL sums every band (to 10 kHz with the reference
    sphere), but its saved upper limit was the 2 kHz plotting range."""
    centers = np.array([10.0, 100.0, 1000.0, 2000.0, 10000.0])
    bands_db = np.arange(centers.size * 6, dtype=float).reshape(centers.size, 2, 3) + 60.0
    bands_db[0, 0, 0] = -np.inf
    levels = demo.oaspl_grids(centers, bands_db, (0.0, 2000.0))
    assert levels['oaspl_fmax_hz'] == 2000.0
    assert levels['oaspl_fullband_fmax_hz'] == 10000.0
    power = 10.0 ** (bands_db / 10.0)
    np.testing.assert_allclose(levels['oaspl_fullband_db'], 10.0 * np.log10(power.sum(axis=0)), rtol=1e-12)
    np.testing.assert_allclose(levels['oaspl_db'], 10.0 * np.log10(power[:4].sum(axis=0)), rtol=1e-12)
    a_weights = fa.dBAw(centers[:4])[:, None, None]
    np.testing.assert_allclose(levels['spla_db'], 10.0 * np.log10((10.0 ** ((bands_db[:4] + a_weights) / 10.0)).sum(axis=0)),
                               rtol=1e-12)
