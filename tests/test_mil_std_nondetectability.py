from pathlib import Path

import numpy as np
import pytest

import flight_acoustics as fa


def _table_path():
    return Path(__file__).resolve().parents[1] / 'mil_std_1474e_table_c1_full.csv'


def test_mil_std_loader_parses_table_c1_csv():
    table = fa.load_mil_std_1474e_table_c1(_table_path())

    assert table['band_freq_hz'].size > 0
    assert table['distance_columns_m'].size > 0
    assert table['measurement_distance_m'].size == table['distance_columns_m'].size
    assert table['limits_db'].shape == (
        table['band_freq_hz'].size,
        table['distance_columns_m'].size,
    )


def _table():
    return fa.load_mil_std_1474e_table_c1(_table_path())


def _quiet(band_freq):
    return np.full_like(band_freq, -100.0, dtype=float)


def test_mil_std_nondetectability_status_and_trigger_selection():
    table = _table()
    band_freq = table['band_freq_hz']
    levels_10m = _quiet(band_freq)
    idx_80 = int(np.where(band_freq == 80.0)[0][0])
    idx_100 = int(np.where(band_freq == 100.0)[0][0])
    idx_125 = int(np.where(band_freq == 125.0)[0][0])

    # 100 Hz at 51 dB: at 2 m 64.98 dB, over every 2 m column (30 m: 56 dB,
    # exceedance 8.98); at 10 m under the 100 m column's 59 dB (-8).  The
    # exceedance falls to zero between 30 and 100 m, across the group
    # boundary: lg d = lg 30 + 8.98 / 16.98 * lg(100 / 30), d = 56.71 m.
    # (It was 'below first' of the 10 m group and left out.)
    levels_10m[idx_100] = 51.0
    # 125 Hz at 61 dB: +3 over the 100 m column (58), -3 under the 200 m
    # column (64): d = sqrt(100 * 200) = 141.42 m.
    levels_10m[idx_125] = 61.0

    result = fa.mil_std_1474e_nondetectability_distance(band_freq, levels_10m, table, spectrum_distance_m=10.0)

    assert result['band_status'][idx_100] == 'interpolated'
    assert bool(result['band_valid'][idx_100])
    d_100 = 30.0 * (100.0 / 30.0) ** ((51.0 + 20 * np.log10(5.0) - 56.0) / (51.0 + 20 * np.log10(5.0) - 56.0 + 8.0))
    assert result['band_nondetectability_distance_m'][idx_100] == pytest.approx(d_100, rel=1e-12)
    assert result['band_nondetectability_distance_m'][idx_100] == pytest.approx(56.71, abs=0.01)

    assert result['band_status'][idx_125] == 'interpolated'
    assert result['band_nondetectability_distance_m'][idx_125] == pytest.approx(np.sqrt(2.0e4), rel=1e-12)

    # A band under every column is nondetectable from the first column on.
    assert result['band_status'][idx_80] == 'below_first'
    assert not bool(result['band_valid'][idx_80])
    assert result['band_nondetectability_distance_m'][idx_80] == 5.0
    assert (result['band_distance_lower_m'][idx_80], result['band_distance_upper_m'][idx_80]) == (0.0, 5.0)

    assert result['trigger_frequency_hz'] == 125.0
    assert result['overall_nondetectability_distance_m'] == pytest.approx(np.sqrt(2.0e4), rel=1e-12)
    assert result['overall_bound'] is None


def test_band_above_every_column_sets_a_lower_bound():
    """50 Hz at 110 dB at 10 m is 100.46 dB at 30 m, over the 6000 m column's
    92 dB: detectable beyond the table.  It was NaN and left out, so the
    overall distance came from a quieter band (141 m)."""
    table = _table()
    band_freq = table['band_freq_hz']
    levels_10m = _quiet(band_freq)
    levels_10m[band_freq == 50.0] = 110.0
    levels_10m[band_freq == 125.0] = 61.0
    result = fa.mil_std_1474e_nondetectability_distance(band_freq, levels_10m, table)
    assert result['band_status'][0] == 'above_last'
    assert result['band_nondetectability_distance_m'][0] == 6000.0
    assert result['band_distance_upper_m'][0] == np.inf
    assert result['overall_nondetectability_distance_m'] == 6000.0
    assert result['trigger_frequency_hz'] == 50.0
    assert result['overall_bound'] == 'lower'
    assert result['overall_distance_lower_m'] == 6000.0
    assert result['overall_distance_upper_m'] == np.inf


def test_several_bands_above_every_column_trigger_on_the_largest_exceedance():
    table = _table()
    band_freq = table['band_freq_hz']
    levels_10m = _quiet(band_freq)
    levels_10m[band_freq == 50.0] = 105.0       # 3.5 dB over 92 at 30 m
    levels_10m[band_freq == 63.0] = 110.0       # 6.5 dB over 94
    result = fa.mil_std_1474e_nondetectability_distance(band_freq, levels_10m, table)
    assert result['trigger_frequency_hz'] == 63.0


def test_level_between_the_10_m_and_30_m_groups():
    """100 Hz at 73.5 dB: 1.5 dB over the 400 m column (72 dB at 10 m), and
    at 30 m 63.96 dB, 1.04 dB under the 500 m column (65): d = 400 * 1.25 **
    (1.5 / 2.54) = 456.3 m.  It was 'above last' of the 10 m group (NaN) and
    'below first' of the 30 m group (invalid), and left out."""
    table = _table()
    band_freq = table['band_freq_hz']
    levels_10m = _quiet(band_freq)
    levels_10m[band_freq == 100.0] = 73.5
    result = fa.mil_std_1474e_nondetectability_distance(band_freq, levels_10m, table)
    e400 = 73.5 - 72.0
    e500 = 73.5 - 20 * np.log10(3.0) - 65.0
    expected = 400.0 * 1.25 ** (e400 / (e400 - e500))
    assert expected == pytest.approx(456.3, abs=0.05)
    assert result['band_status'][3] == 'interpolated'
    assert result['overall_nondetectability_distance_m'] == pytest.approx(expected, rel=1e-12)
    assert result['trigger_frequency_hz'] == 100.0


def test_band_over_its_last_listed_limit_is_bracketed():
    """10 kHz has no limit past 400 m (NA): at 97 dB, 1 dB over 400 m's 96,
    it lies between 400 and 500 m; 500 m is reported, an upper bound."""
    table = _table()
    band_freq = table['band_freq_hz']
    levels_10m = _quiet(band_freq)
    levels_10m[-1] = 97.0
    result = fa.mil_std_1474e_nondetectability_distance(band_freq, levels_10m, table)
    assert result['band_status'][-1] == 'before_unlisted'
    assert result['band_nondetectability_distance_m'][-1] == 500.0
    assert (result['band_distance_lower_m'][-1], result['band_distance_upper_m'][-1]) == (400.0, 500.0)
    assert result['overall_bound'] == 'upper'
    assert (result['overall_distance_lower_m'], result['overall_distance_upper_m']) == (400.0, 500.0)


def test_truncated_spectrum_does_not_extrapolate():
    """A spectrum from 50 Hz to 1 kHz says nothing above 1 kHz.  Holding the
    1 kHz level flat to 10 kHz made the 3.15 kHz band, at 60 dB, the trigger
    (885 m) instead of the 1 kHz band (221 m)."""
    table = _table()
    band_freq = table['band_freq_hz']
    centers = band_freq[band_freq <= 1000.0]
    result = fa.mil_std_1474e_nondetectability_distance(centers, np.full(centers.size, 60.0), table)
    above = band_freq > 1000.0
    assert np.all(result['band_status'][above] == 'outside_spectrum')
    assert np.all(np.isnan(result['band_level_db_10m'][above]))
    assert np.all(np.isnan(result['band_nondetectability_distance_m'][above]))
    assert result['trigger_frequency_hz'] == 1000.0
    assert result['overall_nondetectability_distance_m'] == pytest.approx(221.3, abs=0.05)
    # Exact base-2 centers within half a band of the table's still cover it.
    exact = 1000.0 * 2.0 ** (np.round(3.0 * np.log2(centers / 1000.0)) / 3.0)
    again = fa.mil_std_1474e_nondetectability_distance(exact, np.full(centers.size, 60.0), table)
    np.testing.assert_array_equal(again['band_status'], result['band_status'])


def test_input_order_does_not_matter():
    table = _table()
    band_freq = table['band_freq_hz']
    levels = np.linspace(70.0, 30.0, band_freq.size)
    forward = fa.mil_std_1474e_nondetectability_distance(band_freq, levels, table)
    backward = fa.mil_std_1474e_nondetectability_distance(band_freq[::-1], levels[::-1], table)
    np.testing.assert_array_equal(forward['band_nondetectability_distance_m'],
                                  backward['band_nondetectability_distance_m'])
    np.testing.assert_array_equal(forward['band_level_db_10m'], backward['band_level_db_10m'])
    with pytest.raises(ValueError, match='duplicate'):
        fa.mil_std_1474e_nondetectability_distance([100.0, 100.0], [50.0, 50.0], table)
    with pytest.raises(ValueError, match='length'):
        fa.mil_std_1474e_nondetectability_distance([100.0, 125.0], [50.0], table)


def test_silent_band_is_below_first():
    table = _table()
    band_freq = table['band_freq_hz']
    levels = _quiet(band_freq)
    levels[5] = -np.inf
    levels[7] = np.nan
    result = fa.mil_std_1474e_nondetectability_distance(band_freq, levels, table)
    assert result['band_status'][5] == 'below_first'
    assert result['band_status'][7] == 'no_level'
    assert result['overall_nondetectability_distance_m'] == 5.0
    assert result['overall_bound'] == 'upper'
