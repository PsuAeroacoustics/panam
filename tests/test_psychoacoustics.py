"""ISO 532-1:2017 stationary loudness and DIN 45692 sharpness.

References are the standard's own: Annex B.2's third-octave test case with its
published results, and Annex B.3's synthetic signals.  For the tones and the
narrow-band noise, the third-octave levels are those the Annex A program's
filter bank gives for the signal (a 1 kHz tone also reaches the neighboring
bands, at about 20 dB below), rounded to 0.01 dB.
"""

import numpy as np
import pytest

from panam_acoustics import psychoacoustics as pa

# ISO 532-1:2017 Annex B.2, "Test signal 1": 28 bands, 25 Hz to 12.5 kHz.
TEST_SIGNAL_1_DB = np.array([-60, -60, 78, 79, 89, 72, 80, 89, 75, 87, 85, 79, 86, 80, 71, 70, 72, 71,
                             72, 74, 69, 65, 67, 77, 68, 58, 45, 30], dtype=float)
# Its published free-field result, N' at 0.1 to 24.0 Bark (sone/Bark).
TEST_SIGNAL_1_SPECIFIC = np.array([
    5.3272, 5.3272, 5.3272, 5.3272, 5.3272, 5.3272, 5.3272, 5.3272, 5.3272, 7.3877, 7.3877,
    7.3877, 7.3877, 7.3877, 7.3877, 7.3877, 7.3877, 7.3877, 7.5599, 7.5599, 7.5599, 7.5599,
    7.5599, 7.5599, 7.5599, 7.5599, 7.5599, 7.5599, 7.2802, 7.0002, 6.7202, 6.4402, 6.1602,
    5.9351, 5.7251, 6.4327, 6.4327, 6.4327, 6.4327, 6.4327, 6.4327, 6.4327, 6.4327, 6.4327,
    6.2129, 6.0124, 5.8324, 5.6524, 5.4724, 5.2924, 5.1124, 4.9324, 4.7524, 4.5724, 4.4024,
    4.2718, 4.1418, 4.0118, 3.8818, 3.7518, 3.6218, 3.4918, 3.3618, 3.2318, 3.1018, 2.9915,
    2.8815, 2.7715, 2.6615, 2.5515, 2.4415, 2.3315, 2.2901, 2.2901, 2.2901, 2.2901, 2.2901,
    2.2901, 2.2901, 2.5232, 2.5232, 2.5232, 2.5232, 2.5232, 2.5232, 2.5232, 2.5232, 2.5232,
    2.5232, 2.5232, 2.5232, 2.5232, 2.4134, 2.3498, 2.3498, 2.3498, 2.3498, 2.3498, 2.3498,
    2.3498, 2.3498, 2.3498, 2.3498, 2.3498, 2.3498, 2.3498, 2.5533, 2.5533, 2.5533, 2.5533,
    2.5533, 2.5533, 2.5533, 2.5533, 2.5533, 2.5533, 2.5533, 2.5533, 2.5533, 2.5533, 2.5533,
    2.5533, 2.5533, 3.0669, 3.0669, 3.0669, 3.0669, 3.0669, 3.0669, 3.0669, 3.0669, 3.0669,
    3.0669, 3.0669, 3.0669, 3.0669, 3.0669, 3.0669, 2.957, 2.847, 2.737, 2.627, 2.5232, 2.5232,
    2.5232, 2.5232, 2.5232, 2.5232, 2.5232, 2.5232, 2.5232, 2.5232, 2.4134, 2.3034, 2.281,
    2.281, 2.281, 2.281, 2.281, 2.281, 2.281, 2.281, 2.281, 2.281, 2.281, 2.281, 2.281, 2.6301,
    2.6301, 2.6301, 2.6301, 2.6301, 2.6301, 2.6301, 2.6301, 2.6301, 2.6301, 2.6301, 2.6301,
    2.6301, 2.6301, 4.3633, 4.3633, 4.3633, 4.3633, 4.3633, 4.3633, 4.3633, 4.3633, 4.3633,
    4.3633, 4.3633, 4.3633, 4.2334, 4.1034, 3.9734, 3.8434, 3.7134, 3.5834, 3.4534, 3.3234,
    3.1934, 3.069, 2.959, 2.849, 2.739, 2.629, 2.519, 2.409, 2.299, 2.189, 2.092, 2.01, 1.928,
    1.846, 1.764, 1.682, 1.6, 1.518, 1.436, 1.3555, 1.2935, 1.2315, 1.1695, 1.1075, 1.0455,
    0.9835, 0.9215, 0.8595, 0.8047, 0.7627, 0.7207, 0.6787, 0.6367, 0.5947, 0.5527, 0.5107,
    0.4687, 0.4267, 0.4015])

# Third-octave levels (dB) of signals through the Annex A filter bank.
TONE_1KHZ_40DB = np.array([
    -26.02, -26.02, -26.02, -26.02, -26.02, -26.02, -26.02, -26.01, -25.99, -25.89, -25.46, -23.70,
    -18.85, -10.59, 1.03, 20.00, 40.00, 20.00, 1.07, -10.52, -18.69, -23.56, -25.38, -25.86, -25.98,
    -26.01, -26.02, -26.02])
TONE_250HZ_80DB = np.array([
    -4.76, -10.61, -7.25, -1.05, 5.58, 12.64, 20.42, 29.51, 41.33, 60.56, 80.00, 59.45, 40.76, 29.11,
    20.09, 12.36, 5.34, -1.27, -7.60, -13.62, -18.96, -22.93, -24.98, -25.72, -25.93, -25.99, -26.01,
    -26.02])
TONE_1KHZ_60DB = np.array([
    -25.69, -25.98, -26.01, -26.02, -26.01, -25.98, -25.85, -25.36, -23.74, -19.99, -14.29, -7.47,
    0.24, 9.29, 21.02, 40.00, 60.00, 40.00, 21.07, 9.36, 0.43, -7.13, -13.76, -19.30, -23.07, -24.94,
    -25.63, -25.88])
TONE_4KHZ_40DB = np.array([
    -26.02, -26.02, -26.02, -26.02, -26.02, -26.02, -26.02, -26.02, -26.02, -26.02, -26.02, -26.02,
    -26.02, -26.01, -26.00, -25.91, -25.53, -23.96, -19.38, -11.20, 0.47, 19.44, 40.00, 20.55, 2.03,
    -8.79, -16.04, -21.12])
# Noise from 920 Hz to 1080 Hz (the critical band at 1 kHz), 60 dB.
CRITICAL_BAND_NOISE_1KHZ_60DB = np.array([
    -26.02, -26.02, -26.02, -26.02, -26.01, -25.98, -25.84, -25.32, -23.63, -19.77, -14.00, -7.12,
    0.68, 9.92, 22.22, 44.22, 60.01, 43.62, 22.04, 9.85, 0.76, -6.88, -13.56, -19.15, -22.98, -24.90,
    -25.61, -25.88])


def test_annex_b2_test_signal_1_free_field():
    """The standard's published result: N, LN and every N' to their printed 4 decimals."""
    N, specific, LN = pa.loudness_zwicker(TEST_SIGNAL_1_DB, field='free')
    assert N == pytest.approx(83.2957, abs=5e-5)
    assert LN == pytest.approx(103.8017, abs=5e-5)
    assert specific.shape == (240,)
    assert np.allclose(specific, TEST_SIGNAL_1_SPECIFIC, rtol=0.0, atol=5e-5)


def test_annex_b2_test_signal_1_diffuse_field():
    """The Annex A program's diffuse-field result for the same levels."""
    N, _, _ = pa.loudness_zwicker(TEST_SIGNAL_1_DB, field='diffuse')
    assert N == pytest.approx(85.57456, abs=1e-5)


def test_1khz_tone_at_40db_is_one_sone_and_40_phon():
    """The definition of the sone and the phon, through the standard's filter bank."""
    N, _, LN = pa.loudness_zwicker(TONE_1KHZ_40DB)
    assert N == pytest.approx(1.0, abs=1e-3)
    assert LN == pytest.approx(40.0, abs=0.01)


@pytest.mark.parametrize('levels, expected_sone, expected_phon', [
    (TONE_250HZ_80DB, 14.6545, 78.7327),   # Annex B.3, test signal 2
    (TONE_1KHZ_60DB, 4.0192, 60.0689),     # test signal 3
    (TONE_4KHZ_40DB, 1.5494, 46.3169),     # test signal 4
])
def test_annex_b3_synthetic_tones(levels, expected_sone, expected_phon):
    """Within 1 % of the published results, well inside the standard's 5 % tolerance.

    The published values come from the signal files, the levels here from a
    synthesized tone, so they do not agree to the last digit.
    """
    N, _, LN = pa.loudness_zwicker(levels)
    assert N == pytest.approx(expected_sone, rel=0.01)
    assert LN == pytest.approx(expected_phon, abs=0.15)


def test_din_45692_reference_sound_is_one_acum():
    """A critical-band noise at 1 kHz, 60 dB, is 1 acum."""
    N, specific, _ = pa.loudness_zwicker(CRITICAL_BAND_NOISE_1KHZ_60DB)
    assert pa.sharpness_din(N, specific) == pytest.approx(1.0, abs=0.01)


def test_sharpness_rises_with_high_frequency_content():
    low = np.full(28, -np.inf)
    low[16] = 60.0
    high = low.copy()
    high[25] = 60.0                                     # add 8 kHz
    s_low = pa.sharpness_din(*pa.loudness_zwicker(low)[:2])
    s_high = pa.sharpness_din(*pa.loudness_zwicker(high)[:2])
    assert s_high > s_low + 0.5


def test_loudness_level_conversion():
    assert pa.loudness_level(1.0) == pytest.approx(40.0)
    assert pa.loudness_level(2.0) == pytest.approx(50.0)
    assert pa.loudness_level(0.0) == pytest.approx(3.0)
    assert pa.loudness_level(0.5) == pytest.approx(40.0 * 0.5005 ** 0.35)
    assert np.isnan(pa.loudness_level(np.nan))


def test_no_energy_and_no_data():
    """-inf is a band with no energy; a NaN band makes the spectrum's results NaN."""
    quiet = TONE_1KHZ_40DB.copy()
    quiet[:5] = -np.inf
    very_low = TONE_1KHZ_40DB.copy()
    very_low[:5] = -300.0
    assert pa.loudness_zwicker(quiet)[0] == pa.loudness_zwicker(very_low)[0]

    silent = np.full(28, -np.inf)
    N, specific, LN = pa.loudness_zwicker(silent)
    assert N == 0.0 and np.all(specific == 0.0) and LN == 3.0
    assert np.isnan(pa.sharpness_din(N, specific))

    gap = TONE_1KHZ_40DB.copy()
    gap[3] = np.nan
    N, specific, LN = pa.loudness_zwicker(np.stack([TONE_1KHZ_40DB, gap]))
    assert N.shape == (2,) and specific.shape == (2, 240) and LN.shape == (2,)
    assert np.isfinite(N[0]) and np.isnan(N[1]) and np.all(np.isnan(specific[1])) and np.isnan(LN[1])
    assert np.isnan(pa.sharpness_din(N, specific)[1])


def test_batches_match_single_spectra():
    levels = np.stack([TEST_SIGNAL_1_DB, TONE_1KHZ_40DB, TONE_250HZ_80DB]).reshape(3, 1, 28)
    N, specific, _ = pa.loudness_zwicker(levels, field='diffuse')
    assert N.shape == (3, 1) and specific.shape == (3, 1, 240)
    for i in range(3):
        single = pa.loudness_zwicker(levels[i, 0], field='diffuse')
        assert N[i, 0] == single[0]
        assert np.array_equal(specific[i, 0], single[1])


def test_rejects_bad_input():
    with pytest.raises(ValueError, match='28 bands'):
        pa.loudness_zwicker(np.zeros(27))
    with pytest.raises(ValueError, match='field'):
        pa.loudness_zwicker(TEST_SIGNAL_1_DB, field='pressure')
    with pytest.raises(ValueError, match=r'\+inf'):
        pa.loudness_zwicker(np.full(28, np.inf))
