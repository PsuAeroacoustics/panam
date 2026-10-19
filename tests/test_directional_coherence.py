"""directional_coherence: the band coherency between emission directions, its fit, and the
database's record of it."""
import numpy as np
import pytest
from netCDF4 import Dataset

from panam import directional_coherence as dc
from panam import flight_acoustics as fa
from panam import nod_database as nd
from sphere_helpers import minimal_hemisphere, write_sphere_directory


def _record(**overrides):
    fields = dict(blades=4.0, rotation='counterclockwise', rotation_limit_hz=124.0, scale_hz_deg=850.0,
                  exponent=1.6, spread_scale_hz_deg=2120.0, spread_exponent=1.0,
                  tonal_fraction=np.array([0.9, 0.5, 0.1]), tonal_fraction_bands=np.array([50.0, 100.0, 200.0]),
                  source='synthetic')
    fields.update(overrides)
    return dc.CoherenceRecord(**fields)


def test_a_record_survives_the_database_root(tmp_path):
    path = tmp_path / 'root.nc'
    with Dataset(path, 'w') as nc:
        dc.write_coherence_record(nc, _record(), band_centers=[50.0, 70.7, 100.0, 400.0])
    with Dataset(path) as nc:
        back = dc.read_coherence_record(nc, band_centers=[50.0, 70.7, 100.0, 400.0])
    assert back.blades == 4.0 and back.rotation == 'counterclockwise'
    assert back.rotation_limit_hz == 124.0
    assert (back.scale_hz_deg, back.exponent) == (850.0, 1.6)
    assert (back.spread_scale_hz_deg, back.spread_exponent) == (2120.0, 1.0)
    # Interpolated in log frequency onto the database's bands, held at the ends.
    np.testing.assert_allclose(back.tonal_fraction, [0.9, 0.7, 0.5, 0.1], atol=1e-3)
    assert back.source == 'synthetic'


def test_without_spectra_the_tonal_fraction_is_left_out(tmp_path):
    path = tmp_path / 'root.nc'
    with Dataset(path, 'w') as nc:
        dc.write_coherence_record(nc, _record())
    with Dataset(path) as nc:
        assert 'coherence_tonal_fraction' not in nc.ncattrs()
        assert dc.read_coherence_record(nc).tonal_fraction.size == 0


@pytest.mark.parametrize('bad', [dict(scale_hz_deg=-1.0), dict(exponent=None), dict(rotation='sunwise'),
                                 dict(tonal_fraction=np.array([0.5, 1.5, 0.0])),
                                 dict(tonal_fraction_bands=np.array([100.0, 50.0, 200.0]))])
def test_a_record_niceops_would_refuse_is_not_written(tmp_path, bad):
    with Dataset(tmp_path / 'root.nc', 'w') as nc, pytest.raises(ValueError):
        dc.write_coherence_record(nc, _record(**bad))


def _turning_pattern(sense, omega=400.0, fs=8000.0, seconds=4.0, count=12, seed=1):
    """Microphones on an arc round a hovering source, each hearing one noise delayed by its
    azimuth over Omega: a pattern turning with the rotor (counterclockwise seen from above
    for sense +1), already on emission time.  The arc spans less than half a turn, so no
    pair's azimuth difference wraps (noise does not repeat each turn as a rotor's pattern
    does), and Omega is fast enough that the delays (6.5 ms at most) sit well inside a
    segment."""
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    noise = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    azimuth = np.linspace(0.0, np.radians(150.0), count)
    microphones = np.column_stack([100.0 * np.cos(azimuth), 100.0 * np.sin(azimuth), np.zeros(count)])
    delay = sense * azimuth / omega
    signals = np.array([np.fft.irfft(noise * np.exp(-2j * np.pi * f * tau), n) for tau in delay])
    time = np.arange(n) / fs
    track = np.array([[0.0, 0.0, 60.0], [0.0, 0.0, 60.0]])
    return signals, time, fs, np.array([0.0, seconds]), track, microphones


def test_the_rotors_delay_is_removed_with_the_right_sense():
    signals, time, fs, track_time, track, microphones = _turning_pattern(sense=1)
    bands = [1000.0, 2000.0]
    right = dc.band_coherency(signals, time, fs, track_time, track, microphones, bands, window=2.0,
                              rotation=(400.0, 1))
    wrong = dc.band_coherency(signals, time, fs, track_time, track, microphones, bands, window=2.0,
                              rotation=(400.0, -1))
    apart = right['separation'] > 20.0
    # The raw coherency is the delay's, band-averaged away; with the delay removed the
    # directions are one waveform (short of 1 by what of a segment the delay shifts out of
    # it, up to a tenth); turned the wrong way, twice the delay.
    assert np.mean(np.real(right['rho_turned'][apart])) > 0.97
    assert np.all(np.real(right['rho_turned'][apart]) > 0.9)
    assert np.mean(np.abs(right['rho'][apart])) < 0.5
    assert np.mean(np.real(wrong['rho_turned'][apart])) < 0.1
    # The delay model predicts the raw coherency.
    np.testing.assert_allclose(right['delay'], right['rho'], atol=0.08)


def test_a_fit_recovers_the_decay_and_the_tonal_fraction():
    rng = np.random.default_rng(3)
    bands = np.array([50.0, 100.0, 200.0, 400.0, 800.0])
    count = 100000
    band = rng.integers(0, len(bands), count)
    separation = rng.uniform(0.5, 120.0, count)
    x = bands[band] * separation
    mean = np.exp(-(x / 700.0) ** 1.8)
    tonal = np.array([0.4, 0.25, 0.1, 0.05, 0.0])[band]
    # A deviation at a random phase with the spread T (1 - g^2): the mean is g.
    spread = np.sqrt(tonal * (1.0 - mean ** 2))
    rho = mean + spread * np.exp(2j * np.pi * rng.uniform(size=count))
    records = dict(rho=rho, band=band, separation=separation, n_eff=np.full(count, 1e12),
                   coherent=np.ones(count), delay=np.ones(count))
    fit = dc.fit_coherence(records, bands, source='synthetic')
    assert fit.scale_hz_deg == pytest.approx(700.0, rel=0.02)
    assert fit.exponent == pytest.approx(1.8, rel=0.03)
    np.testing.assert_allclose(fit.tonal_fraction, [0.4, 0.25, 0.1, 0.05, 0.0], atol=0.02)
    assert fit.rotation is None and fit.blades is None
    # With the rotor, the record carries it and the delay's band.
    turning = dc.fit_coherence(records, bands, rotation_rate=43.25, blades=4, rotation='counterclockwise')
    assert turning.rotation_limit_hz == pytest.approx(4.5 * 4 * 43.25 / (2 * np.pi))


def test_a_database_carries_the_record_it_was_built_with(tmp_path):
    spheres = write_sphere_directory(tmp_path / 'spheres', [(minimal_hemisphere(), 60.0, 0.0),
                                                            (minimal_hemisphere(), 80.0, -6.0)])
    database = tmp_path / 'database.nod'
    record = _record(tonal_fraction=np.array([0.6, 0.3]), tonal_fraction_bands=np.array([100.0, 160.0]))
    fa.build_empirical_database(str(spheres), str(database), coherence=record)
    assert nd.validate_nod(database) == []
    with Dataset(database) as nc:
        back = dc.read_coherence_record(nc)
        assert back.scale_hz_deg == 850.0 and back.rotation == 'counterclockwise'
        np.testing.assert_allclose(back.tonal_fraction, [0.6, np.interp(np.log(125.0), np.log([100, 160]),
                                                                        [0.6, 0.3]), 0.3])
    # A recipe's table gives the fields as a mapping.
    fa.build_empirical_database(str(spheres), str(tmp_path / 'mapping.nod'),
                                coherence=dict(scale_hz_deg=500.0, exponent=2.0))
    with Dataset(tmp_path / 'mapping.nod') as nc:
        assert nc.getncattr('coherence_scale_Hz_deg') == 500.0


def test_validate_nod_holds_the_record_to_what_niceops_reads(tmp_path):
    spheres = write_sphere_directory(tmp_path / 'spheres', [(minimal_hemisphere(), 60.0, 0.0),
                                                            (minimal_hemisphere(), 80.0, -6.0)])
    database = tmp_path / 'database.nod'
    fa.build_empirical_database(str(spheres), str(database))
    with Dataset(database, 'a') as nc:
        nc.coherence_scale_Hz_deg = 800.0
        nc.coherence_tonal_fraction = np.array([0.5, 0.2])
        nc.main_rotor_rotation = 'sunwise'
    rules = [v.rule for v in nd.validate_nod(database)]
    assert rules.count('R7') == 3


def test_a_record_saved_as_json_loads_as_it_was(tmp_path):
    record = _record()
    dc.save_coherence_record(record, tmp_path / 'record.json')
    back = dc.load_coherence_record(tmp_path / 'record.json')
    assert back.scale_hz_deg == record.scale_hz_deg and back.rotation == record.rotation
    np.testing.assert_array_equal(back.tonal_fraction, record.tonal_fraction)
    np.testing.assert_array_equal(back.tonal_fraction_bands, record.tonal_fraction_bands)


def test_cases_count_equally_however_many_pairs_they_hold():
    rng = np.random.default_rng(5)
    bands = np.array([100.0, 200.0, 400.0])

    def case(scale, count):
        band = rng.integers(0, len(bands), count)
        separation = rng.uniform(0.5, 90.0, count)
        rho = np.exp(-(bands[band] * separation / scale) ** 2) + 0j
        return dict(rho=rho, band=band, separation=separation, n_eff=np.full(count, 1e12),
                    coherent=np.ones(count), delay=np.ones(count))

    small, large = case(1200.0, 2000), case(600.0, 20000)
    doubled = {k: np.concatenate([v, v]) for k, v in large.items()}
    pooled = {k: np.concatenate([small[k], large[k]]) for k in small}
    by_case = dc.fit_coherence([small, large], bands)
    assert dc.fit_coherence([small, doubled], bands).scale_hz_deg == pytest.approx(by_case.scale_hz_deg, rel=1e-6)
    assert dc.fit_coherence([large, small], bands).scale_hz_deg == pytest.approx(by_case.scale_hz_deg, rel=1e-6)
    # Pooled, the large case outweighs the small one.
    assert abs(dc.fit_coherence(pooled, bands).scale_hz_deg - 600.0) < abs(by_case.scale_hz_deg - 600.0)
