from pathlib import Path

import numpy as np

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


def test_mil_std_nondetectability_status_and_trigger_selection():
    table = fa.load_mil_std_1474e_table_c1(_table_path())
    band_freq = table['band_freq_hz']

    # Build a synthetic 10 m spectrum directly on table bands.
    # Most bands are intentionally low to produce warning statuses and invalid flags.
    levels_10m = np.full_like(band_freq, -100.0, dtype=float)

    idx_100 = int(np.where(band_freq == 100.0)[0][0])
    idx_125 = int(np.where(band_freq == 125.0)[0][0])

    # 100 Hz below first 10 m threshold => warning_group_below_first@10m, invalid.
    levels_10m[idx_100] = 51.0

    # 125 Hz between 10 m group thresholds at 100 m (58 dB) and 200 m (64 dB)
    # => interpolated valid result near 141 m.
    levels_10m[idx_125] = 61.0

    result = fa.mil_std_1474e_nondetectability_distance(
        band_freq,
        levels_10m,
        table,
        spectrum_distance_m=10.0,
    )

    assert result['band_status'][idx_100] == 'warning_group_below_first@10m'
    assert not bool(result['band_valid'][idx_100])
    assert np.isclose(result['band_nondetectability_distance_m'][idx_100], 100.0, atol=1e-9)

    assert result['band_status'][idx_125] == 'group_interpolated@10m'
    assert bool(result['band_valid'][idx_125])
    assert 100.0 < result['band_nondetectability_distance_m'][idx_125] < 200.0

    assert np.isclose(result['trigger_frequency_hz'], 125.0, atol=1e-12)
    assert np.isclose(
        result['overall_nondetectability_distance_m'],
        result['band_nondetectability_distance_m'][idx_125],
        atol=1e-12,
    )
