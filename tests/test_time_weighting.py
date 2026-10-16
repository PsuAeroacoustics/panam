import numpy as np
import pytest
import scipy.signal as sig

from panam.acoustics.time_weighting import (
    apply_frequency_weighting, frequency_weighting_sos, leq, time_weighted_level,
    time_weighted_square)

# IEC 61672-1:2013, Table 3: nominal frequencies (evaluated at the exact
# base-ten frequencies 1000 * 10^(k/10)), A and C weightings (dB) and the
# class 1 acceptance limits (dB; inf where the limit is minus infinity).
EXACT_F = 1000.0 * 10.0 ** (np.arange(-20, 14) / 10.0)          # 10 Hz .. 20 kHz
TABLE_A = [-70.4, -63.4, -56.7, -50.5, -44.7, -39.4, -34.6, -30.2, -26.2, -22.5, -19.1,
           -16.1, -13.4, -10.9, -8.6, -6.6, -4.8, -3.2, -1.9, -0.8, 0.0, 0.6, 1.0, 1.2,
           1.3, 1.2, 1.0, 0.5, -0.1, -1.1, -2.5, -4.3, -6.6, -9.3]
TABLE_C = [-14.3, -11.2, -8.5, -6.2, -4.4, -3.0, -2.0, -1.3, -0.8, -0.5, -0.3, -0.2,
           -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, -0.2, -0.3, -0.5,
           -0.8, -1.3, -2.0, -3.0, -4.4, -6.2, -8.5, -11.2]
UPPER = np.array([3.5, 3.0, 2.5, 2.5, 2.5, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5,
                  1.5, 1.5, 2.0, 2.0, 2.5, 3.0])
LOWER = np.array([np.inf, np.inf, 4.5, 2.5, 2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.5, 2.0, 2.5, 3.0, 5.0, 16.0, np.inf])


def response_db(kind, fs, f):
    _, h = sig.sosfreqz(frequency_weighting_sos(kind, fs), worN=np.asarray(f), fs=fs)
    with np.errstate(divide='ignore'):
        return 20.0 * np.log10(np.abs(h))


@pytest.mark.parametrize('fs', [16000.0, 25000.0, 32000.0, 44100.0, 48000.0, 51200.0])
@pytest.mark.parametrize('kind, table', [('A', TABLE_A), ('C', TABLE_C)])
def test_weighting_meets_class_1_limits_where_the_sample_rate_supports(kind, table, fs):
    supported = EXACT_F <= (EXACT_F[-1] if fs >= 44100.0 else fs / 4.0)
    error = response_db(kind, fs, EXACT_F[supported]) - np.asarray(table)[supported]
    assert np.all(error <= UPPER[supported])
    assert np.all(error >= -LOWER[supported])
    # Well inside the band the design is close to the analog curve.
    mid = supported & (EXACT_F >= 20.0) & (EXACT_F <= fs / 10.0)
    assert np.allclose(response_db(kind, fs, EXACT_F[mid]), np.asarray(table)[mid], atol=0.1)


@pytest.mark.parametrize('kind', ['A', 'C', 'Z'])
def test_weighting_is_0_db_at_1_khz(kind):
    assert response_db(kind, 25000.0, [1000.0])[0] == pytest.approx(0.0, abs=1e-9)


def test_z_weighting_passes_the_signal_unchanged():
    x = np.random.default_rng(0).standard_normal(1000)
    assert np.array_equal(apply_frequency_weighting(x, 25000.0, 'Z'), x)


def test_a_weighted_tone_through_the_filter():
    fs = 48000.0
    t = np.arange(int(2 * fs)) / fs
    x = np.sin(2 * np.pi * 100.0 * t)
    y = apply_frequency_weighting(x, fs, 'A')[int(fs):]
    gain_db = 10 * np.log10(np.mean(y ** 2) / np.mean(x[int(fs):] ** 2))
    assert gain_db == pytest.approx(-19.1, abs=0.1)


def tone(fs, seconds, f=4000.0, rms_pa=1.0):
    t = np.arange(int(round(seconds * fs))) / fs
    return np.sqrt(2.0) * rms_pa * np.sin(2 * np.pi * f * t)


# IEC 61672-1:2013, Table 4: maximum time-weighted level of a 4 kHz tone
# burst relative to the steady level, dB, by burst duration in ms.
BURSTS = {
    'F': {1000: 0.0, 500: -0.1, 200: -1.0, 100: -2.6, 50: -4.8, 20: -8.3, 10: -11.1,
          5: -14.1, 2: -18.0, 1: -21.0, 0.5: -24.0, 0.25: -27.0},
    'S': {1000: -2.0, 500: -4.1, 200: -7.4, 100: -10.2, 50: -13.1, 20: -17.0, 10: -20.0,
          5: -23.0, 2: -27.0},
}


@pytest.mark.parametrize('time, table', BURSTS.items())
def test_tone_burst_response(time, table):
    fs = 48000.0
    steady = time_weighted_level(tone(fs, 10.0), fs, 'A', time)['level'][-1]
    tau = {'F': 0.125, 'S': 1.0}[time]
    for ms, expected in table.items():
        # A burst of whole cycles cut from the steady tone, between silences.
        x = np.concatenate([np.zeros(int(fs)), tone(fs, ms / 1000.0), np.zeros(int(3 * fs))])
        out = time_weighted_level(x, fs, 'A', time)
        delta = out['lmax'] - steady
        assert delta == pytest.approx(expected, abs=0.5 if ms >= 1 else 1.0), ms
        # The response is the exact exponential one to well within a tenth of a dB.
        assert delta == pytest.approx(10 * np.log10(1 - np.exp(-ms / 1000.0 / tau)), abs=0.1 if ms >= 1 else 0.2), ms
        assert 1.0 + ms / 1000.0 - 0.01 < out['t_max'] < 1.0 + ms / 1000.0 + 0.01


def test_steady_tone_level_and_time_vector():
    fs = 25000.0
    rms = 20e-6 * 10 ** (94.0 / 20.0)
    out = time_weighted_level(tone(fs, 3.0, 1000.0, rms), fs, 'A', 'F')
    assert out['time'][1] == pytest.approx(1 / fs)
    assert out['level'][-1] == pytest.approx(94.0, abs=0.05)
    assert out['lmax'] == pytest.approx(94.0, abs=0.05)
    assert out['level'][0] == -np.inf          # silence before the first sample
    assert np.isneginf(time_weighted_level(np.zeros(100), fs)['lmax'])


def test_slow_decays_at_4_34_db_per_second():
    fs = 8000.0
    x = np.concatenate([tone(fs, 5.0, 1000.0), np.zeros(int(3 * fs))])
    level = time_weighted_level(x, fs, 'Z', 'S')['level']
    rate = (level[int(6 * fs)] - level[int(7 * fs)])
    assert rate == pytest.approx(10 * np.log10(np.e), abs=0.01)


def test_impulse_rises_with_35_ms_and_decays_with_1_5_s():
    fs = 8000.0
    x = np.concatenate([np.zeros(int(fs)), tone(fs, 0.035, 1000.0), np.zeros(int(4 * fs))])
    ms = time_weighted_square(x, fs, 'I')
    k = int(np.argmax(ms))
    peak_db = 10 * np.log10(ms[k])
    # Unit mean square for 35 ms through a 35 ms exponential reads 1 - 1/e.
    assert peak_db == pytest.approx(10 * np.log10(1 - np.exp(-1.0)), abs=0.1)
    # Held peak then decays at 10 lg(e) / 1.5 dB/s, slower than the 35 ms average.
    decay = peak_db - 10 * np.log10(ms[k + int(fs)])
    assert decay == pytest.approx(10 * np.log10(np.e) / 1.5, abs=0.01)
    assert np.all(np.diff(ms[k:]) <= 0)
    # A steady signal reads its mean square, as F and S do.
    steady = time_weighted_square(tone(fs, 2.0, 1000.0), fs, 'I')
    assert 10 * np.log10(steady[-1]) == pytest.approx(0.0, abs=0.05)


def test_leq_of_a_steady_tone_and_over_a_longer_period():
    fs = 25000.0
    rms = 20e-6 * 10 ** (94.0 / 20.0)
    x = tone(fs, 10.0, 1000.0, rms)
    assert leq(x, fs) == pytest.approx(94.0, abs=0.02)
    assert leq(x, fs, duration=20.0) == pytest.approx(94.0 - 10 * np.log10(2.0), abs=0.02)
    assert leq(tone(fs, 10.0, 100.0, rms), fs, 'A') == pytest.approx(94.0 - 19.1, abs=0.1)
    assert leq(tone(fs, 10.0, 100.0, rms), fs, 'Z') == pytest.approx(94.0, abs=0.02)
    assert leq(np.zeros(10), fs) == -np.inf
    with pytest.raises(ValueError):
        leq(x, fs, duration=5.0)


def test_leq_equals_energy_mean_of_the_time_history():
    fs = 10000.0
    x = np.random.default_rng(1).standard_normal(int(5 * fs)) * 0.1
    pw = apply_frequency_weighting(x, fs, 'C')
    assert leq(x, fs, 'C') == pytest.approx(10 * np.log10(np.mean(pw ** 2) / 4e-10), abs=1e-9)


def test_bad_arguments():
    with pytest.raises(ValueError):
        frequency_weighting_sos('B', 25000.0)
    with pytest.raises(ValueError):
        time_weighted_square(np.ones(10), 25000.0, 'X')
    with pytest.raises(ValueError):
        time_weighted_level(np.ones((2, 10)), 25000.0)
    for kind in ('A', 'C'):
        with pytest.raises(ValueError):
            frequency_weighting_sos(kind, 2000.0)
        with pytest.raises(ValueError):
            frequency_weighting_sos(kind, 1000.0)
    frequency_weighting_sos('Z', 1000.0)
    with pytest.raises(ValueError):
        leq(np.ones(10), 0.0)
    with pytest.raises(ValueError):
        leq(np.ones(10), np.nan, frequency='Z')


def test_nan_sample_makes_the_maximum_nan():
    fs = 48000.0
    p = np.sqrt(2.0) * np.sin(2.0 * np.pi * 1000.0 * np.arange(int(fs)) / fs)
    p[100] = np.nan
    out = time_weighted_level(p, fs)
    assert np.isnan(out['level'][100:]).all()
    assert np.isnan(out['lmax'])
    assert np.isnan(out['t_max'])
