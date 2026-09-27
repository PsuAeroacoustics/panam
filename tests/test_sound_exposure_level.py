import numpy as np
import pytest

import flight_acoustics as fa


def test_constant_level_over_one_second_is_that_level():
    out = fa.sound_exposure_level(np.full(10, 80.0), dt=0.1)
    assert out['sel'] == pytest.approx(80.0)
    assert out['clipped']


def test_interval_is_first_to_last_sample_within_10_db_of_the_peak():
    # rise, peak, dip more than 10 dB down, second rise within 10 dB, fall
    levels = np.array([60, 72, 85, 90, 84, 75, 83, 70, 60], dtype=float)
    k1, k2 = fa.ten_db_down_interval(levels)
    assert (k1, k2) == (2, 6)              # the dip at index 5 stays inside
    out = fa.sound_exposure_level(levels, dt=0.5)
    expected = 10 * np.log10(np.sum(10 ** (levels[2:7] / 10)) * 0.5)
    assert out['sel'] == pytest.approx(expected)
    assert out['duration_s'] == pytest.approx(2.5)
    assert not out['clipped']


def test_ten_db_down_versus_whole_record():
    t = np.arange(-30, 30, 0.1)
    levels = 90 - 20 * np.log10(np.hypot(t * 40, 150) / 150)   # a straight flyover
    down = fa.sound_exposure_level(levels, dt=0.1)['sel']
    whole = fa.sound_exposure_level(levels, dt=0.1, down=np.inf)['sel']
    assert whole > down                    # the tails carry energy
    assert whole - down < 1.0              # but for a flyover not much


def test_interval_chosen_on_one_history_can_integrate_another():
    la = np.array([60, 80, 90, 80, 60], dtype=float)
    lc = la + 5
    out = fa.sound_exposure_level(la, dt=1.0, weighted_levels=lc)
    assert (out['k1'], out['k2']) == (1, 3)
    assert out['sel'] == pytest.approx(10 * np.log10(np.sum(10 ** (lc[1:4] / 10))))
