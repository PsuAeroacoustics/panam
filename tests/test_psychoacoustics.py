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
# Its published free-field N' (sone/Bark), every 20th value: 2.0, 4.0, ..., 24.0 Bark.
# Only these spot values are kept from the Annex B.2 table; with N and LN they
# still catch any error in the tables or the slopes.
TEST_SIGNAL_1_SPECIFIC_INDEX = np.arange(19, 240, 20)
TEST_SIGNAL_1_SPECIFIC = np.array([7.5599, 6.4327, 3.7518, 2.5232, 2.3498, 2.5533, 2.847, 2.281,
                                   2.6301, 3.4534, 1.436, 0.4015])

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
    """The standard's published result: N, LN and spot N' values to their printed 4 decimals."""
    N, specific, LN = pa.loudness_zwicker(TEST_SIGNAL_1_DB, field='free')
    assert N == pytest.approx(83.2957, abs=5e-5)
    assert LN == pytest.approx(103.8017, abs=5e-5)
    assert specific.shape == (240,)
    assert np.allclose(specific[TEST_SIGNAL_1_SPECIFIC_INDEX], TEST_SIGNAL_1_SPECIFIC, rtol=0.0, atol=5e-5)


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


def test_iso_532_1_levels_from_a_31_band_spectrum():
    """A PANAM 10 Hz-10 kHz spectrum gives the same loudness as the hand-built 28 bands."""
    import flight_acoustics as fa
    frequency = fa.NORAH2_BAND_CENTERS_HZ
    assert frequency.size == 31
    spectrum = np.concatenate([[50.0, 45.0, 40.0, 35.0], TEST_SIGNAL_1_DB[:-1]])
    by_hand = TEST_SIGNAL_1_DB.copy()
    by_hand[-1] = -np.inf
    picked = pa.iso_532_1_levels(frequency, spectrum)
    assert np.array_equal(picked, by_hand)
    assert pa.loudness_zwicker(picked)[0] == pa.loudness_zwicker(by_hand)[0]
    exact = 1000.0 * 10.0 ** (np.arange(-20, 11) / 10.0)
    batch = pa.iso_532_1_levels(exact, np.stack([spectrum, spectrum]), missing_12k5=30.0)
    assert batch.shape == (2, 28) and np.array_equal(batch[1], TEST_SIGNAL_1_DB)
    assert np.all(np.isnan(pa.iso_532_1_levels(frequency, spectrum, missing_12k5=np.nan)[-1]))
    with pytest.raises(ValueError, match='no 25 Hz band'):
        pa.iso_532_1_levels(frequency[5:], spectrum[5:])
    with pytest.raises(ValueError, match='bands of frequency_hz'):
        pa.iso_532_1_levels(frequency, spectrum[1:])
