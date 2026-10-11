"""Dataset-independent pieces of the 2017 Noise Abatement sphere rebuild."""

import os

import numpy as np
import pytest

import noise_abatement_2017 as na


def _track(duration=60.0, rate=50.0, speed=80.0, fpa=0.0, roll=0.0,
           heading=-84.0, x0=-4000.0, altitude=600.0):
    """A straight, steady pass; callers perturb one field to break steadiness."""
    n = int(duration * rate)
    time = np.arange(n) / rate
    return dict(
        time=time,
        ground_speed_knots=np.full(n, speed),
        fpa_deg=np.full(n, fpa),
        roll=np.full(n, roll),
        heading=np.full(n, heading),
        x=x0 + speed * 1.68781 * time,
        y=np.zeros(n),
        z=np.full(n, altitude),
    )


def test_steady_window_keeps_a_steady_pass_whole():
    track = _track()
    start, stop = na.steady_window(track)
    assert (start, stop) == (0, track['time'].size)


def test_steady_window_excludes_the_turn_onto_the_run():
    """A level turn holds speed and flight path angle, so those cannot see it.

    Bank angle and turn rate can, and must: through a turn the airframe is
    rotating relative to the array, which smears directivity across azimuth.
    """
    track = _track()
    n = track['time'].size
    turning = slice(0, n // 3)
    track['roll'][turning] = 20.0
    track['heading'][turning] = -84.0 + 6.0 * track['time'][turning]

    start, stop = na.steady_window(track)
    assert start >= n // 3 - 2
    assert stop == n


def test_steady_window_ignores_a_heading_wrap():
    """Unwrap first, or a pass through +/-180 reads as a huge turn rate.

    The heading creeps by 0.02 deg/s and happens to cross the branch cut, so
    every sample is steady; only the wrap makes it look otherwise.
    """
    track = _track()
    heading = 179.5 + 0.02 * track['time']
    track['heading'] = (heading + 180.0) % 360.0 - 180.0
    assert track['heading'].min() < -179.0 < track['heading'].max()   # it really wraps

    start, stop = na.steady_window(track)
    assert (start, stop) == (0, track['time'].size)


def test_steady_window_can_bound_the_range_to_the_array():
    """Straight-ray propagation is only a fair model over a limited path."""
    track = _track(x0=-8000.0, altitude=400.0)
    unbounded = na.steady_window(track)
    bounded = na.steady_window(track, max_array_range=2000.0)
    assert bounded[1] - bounded[0] < unbounded[1] - unbounded[0]
    kept = np.sqrt(track['x'] ** 2 + track['y'] ** 2 + track['z'] ** 2)[bounded[0]:bounded[1]]
    assert kept.max() <= 2000.0


def test_steady_window_refuses_a_segment_that_is_too_short():
    track = _track(duration=60.0)
    with pytest.raises(ValueError, match='only'):
        na.steady_window(track, min_duration_s=120.0)


def test_sphere_title_matches_the_legacy_format():
    """Byte-for-byte, padding included; AAM readers take the length as given."""
    assert (na.sphere_title('Be407', '220', '285')
            == 'Be407  Run 220 10/12/2017 Noise Abatement Flight Test 1/3 Octave Band'.ljust(77))
    assert na.sphere_title('RO-66', '500', '234').startswith('RO-66  Run 500 08/22/2017')
    assert na.sphere_title('AS350', '269', '291').startswith('AS350  Run 269 10/18/2017')


def test_norah2_file_name_follows_the_shipped_convention():
    """[type]_[procedure]_[IAS]kts_[gamma]deg, with the run to keep repeats apart."""
    assert na.norah2_file_name('Be407', 66.15, -8.34, '220') == 'Be407_Approach_66kts_8.3deg_220.hem'
    assert na.norah2_file_name('RO-44', 109.8, 0.4, '12') == 'RO-44_Flyover_110kts_0.4deg_12.hem'
    assert na.norah2_file_name('AS350', 69.3, 9.0, '7') == 'AS350_Takeoff_69kts_9deg_7.hem'


def test_flight_condition_averages_over_the_window_only():
    track = _track(speed=80.0)
    track['ground_speed_knots'][:100] = 200.0
    speed, fpa = na.flight_condition(track, 100, track['time'].size)
    assert speed == pytest.approx(80.0)
    assert fpa == pytest.approx(0.0)


def test_steady_window_checks_speed_and_flight_path_angle():
    """Both are tested as a band around the run's own median."""
    track = _track()
    n = track['time'].size
    track['ground_speed_knots'][: n // 4] += 30.0        # accelerating entry
    track['fpa_deg'][-n // 4:] -= 20.0                   # pushover at the end
    start, stop = na.steady_window(track)
    kept = slice(start, stop)
    assert np.ptp(track['ground_speed_knots'][kept]) <= 2 * 4.0
    assert np.ptp(track['fpa_deg'][kept]) <= 2 * 2.0
    assert start >= n // 4 - 2 and stop <= n - n // 4 + 2


def test_steady_window_survives_a_single_sample_excursion():
    """One gust-induced roll spike must not halve an otherwise good window.

    Steadiness is judged per sample at 50 Hz, so without closing brief gaps a
    single bad sample splits the window in two -- that cost 15 of 113 Be407
    runs their 8 s minimum.
    """
    track = _track()
    n = track['time'].size
    track['roll'][n // 2] = 25.0            # one sample, 0.02 s

    start, stop = na.steady_window(track)
    assert stop - start == n

    # A real turn lasts seconds and must still split it.
    track['roll'][n // 2: n // 2 + 200] = 25.0
    start, stop = na.steady_window(track)
    assert stop - start < n


def _row(test_cond='D14', bank_ang='0', accel_rate='0'):
    return dict(test_cond=test_cond, bank_ang=bank_ang, accel_rate=accel_rate)


def test_is_steady_flight_card_accepts_explicit_zero():
    assert na.is_steady_flight_card(_row())


def test_is_steady_flight_card_rejects_a_maneuver():
    assert not na.is_steady_flight_card(_row(bank_ang='15'))
    assert not na.is_steady_flight_card(_row(accel_rate='-0.3'))


def test_is_steady_flight_card_rejects_ambient_and_hover():
    assert not na.is_steady_flight_card(_row(test_cond='AMB'))
    assert not na.is_steady_flight_card(_row(test_cond='H3'))


def test_is_steady_flight_card_rejects_approaches_even_with_explicit_zeros():
    # The B206L3's approach cards read bank 0 and acceleration 0.
    assert not na.is_steady_flight_card(_row(test_cond='A7'))
    assert na.is_steady_flight_card(_row(test_cond='L7'))


def test_is_steady_flight_card_treats_blank_as_maneuvering():
    """Blank is not the same as zero: the B407 'A' family leaves accel_rate
    blank on decelerating approaches, not on runs that just forgot to log 0."""
    assert not na.is_steady_flight_card(_row(accel_rate=''))
    assert not na.is_steady_flight_card(_row(bank_ang=''))


def test_run_atmosphere_reads_station_temperature_as_fahrenheit(tmp_path):
    """The ground stations log Fahrenheit; 32 F must come back as 273.15 K."""
    stations = tmp_path / 'X_Weather' / 'X_Ground_Stations'
    stations.mkdir(parents=True)
    (stations / 'X_289_SWS1.csv').write_text(
        '    utcsec,    time, airtemp, humidity, pressure\n'
        '     39909, 11:05:09,       32.0,       71.3,    88.8300\n'
        '     39919, 11:05:19,     -1000.0,       71.5,    88.8300\n')

    class FakeTest:
        base = str(tmp_path)
        aircraft = 'X'
        by_run = {'289100': {'utc_secs_from_mid_start': '39910'}}

    atmosphere = na.run_atmosphere(FakeTest(), '289100')
    assert atmosphere.temperature == pytest.approx(273.15)
    assert atmosphere.relative_humidity == pytest.approx(71.3)
    assert atmosphere.pressure == pytest.approx(88.83)


def test_run_atmosphere_finds_stations_spelled_with_a_space(tmp_path):
    """EC130B4's folder is 'EC130B4_Ground Stations'; its runs must not miss their weather."""
    stations = tmp_path / 'X_Weather' / 'X_Ground Stations'
    stations.mkdir(parents=True)
    (stations / 'X_296_SWS1.csv').write_text(
        '    utcsec,    time, airtemp, humidity, pressure\n'
        '     39909, 11:05:09,       50.0,       80.0,    89.1000\n')

    class FakeTest:
        base = str(tmp_path)
        aircraft = 'X'
        by_run = {'296100': {'utc_secs_from_mid_start': '39910'}}

    atmosphere = na.run_atmosphere(FakeTest(), '296100')
    assert atmosphere.temperature == pytest.approx(283.15)
    assert atmosphere.relative_humidity == pytest.approx(80.0)
    assert atmosphere.pressure == pytest.approx(89.1)


def _descending(vz_positive_up, rate_fps=10.0, n=500):
    time = np.arange(n) * 0.02
    z = 1000.0 - rate_fps * time
    vz = -rate_fps if vz_positive_up else rate_fps
    return dict(time=time, z=z, vz=np.full(n, vz) + 0.8 * np.sin(time))


def test_vz_sign_follows_each_file_not_the_dataset():
    """Every 2017 file stores vz positive down except EC130B4 day 298's, which store it up."""
    assert na.vz_sign(_descending(vz_positive_up=False)) == -1.0
    assert na.vz_sign(_descending(vz_positive_up=True)) == 1.0


@pytest.mark.parametrize('vz_positive_up', [False, True])
def test_load_track_gives_a_descent_a_negative_flight_path_angle(tmp_path, vz_positive_up):
    """fpa = atan2(vz_up, hypot(vx, vy)) whichever way the file stores vz."""
    track = _descending(vz_positive_up)
    path = tmp_path / 'track.csv'
    _write_csv(path, ['utcsec', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'VGk'],
               [[t, 100.0 * t, 0.0, z, 100.0, 0.0, vz, 59.2]
                for t, z, vz in zip(track['time'], track['z'], track['vz'])])
    loaded = na.load_track(str(path))
    vz_up = track['vz'] if vz_positive_up else -track['vz']
    expected = np.degrees(np.arctan2(vz_up, 100.0))
    np.testing.assert_allclose(loaded['fpa_deg'], expected, atol=1e-9)
    assert np.median(loaded['fpa_deg']) < -5.0


def test_vz_sign_on_a_level_track_falls_back_to_down():
    time = np.arange(500) * 0.02
    level = dict(time=time, z=np.full(500, 500.0), vz=np.zeros(500))
    assert na.vz_sign(level) == -1.0


def test_vz_sign_falls_back_to_down_with_a_warning_when_ambiguous():
    """Level passes and hovers (about 40 files) barely correlate; they must still
    load, as before the sign was checked, rather than fail the run."""
    rng = np.random.default_rng(0)
    time = np.arange(2000) * 0.02
    track = dict(time=time, z=500.0 + np.cumsum(rng.normal(0, 0.1, time.size)),
                 vz=rng.normal(0, 5.0, time.size))
    with pytest.warns(UserWarning, match='barely agree'):
        assert na.vz_sign(track) == -1.0


def _plate_table_stub(gain_db, calls=None):
    """Stands in for ``axisymmetric_bem.table``: a plate that reads ``gain_db``
    over free field in every direction (P_d constant, P_r zero)."""
    def table(bands, sound_speed, flow_resistance=None, ground=None, sub_bands=5, **options):
        bands = np.asarray(bands, dtype=float)
        if calls is not None:
            calls.append(bands)
        offsets = 2.0 ** ((np.arange(sub_bands) + 0.5) / sub_bands / 3.0 - 1.0 / 6.0)
        frequencies = np.sort((bands[:, None] * offsets[None, :]).ravel())
        p_d = np.full((frequencies.size, 2, 4), 10.0 ** (gain_db / 20.0), dtype=complex)
        return dict(frequencies=frequencies, elevations=np.array([0.0, 90.0]),
                    azimuths=np.array([0.0, 90.0, 180.0, 270.0]), P_d=p_d, P_r=np.zeros_like(p_d),
                    ground=ground, flow_resistance=flow_resistance, bands=bands, sub_bands=sub_bands,
                    thickness=0.008 / 0.3048, mic_height=options.get('mic_height', 0.0))
    return table


def test_plate_table_checks_the_bands_of_a_table_held_in_memory(monkeypatch, tmp_path):
    """The cache key holds only the band count and ends, so two band sets that
    share them must not share a table."""
    import axisymmetric_bem as ab
    calls = []
    monkeypatch.setattr(ab, 'table', _plate_table_stub(0.0, calls))
    monkeypatch.setattr(na, '_PLATE_TABLES', {})
    first = na.plate_table('gdbdfl', 1125.0, [630.0, 800.0, 1000.0], directory=str(tmp_path))
    second = na.plate_table('gdbdfl', 1125.0, [630.0, 793.7, 1000.0], directory=str(tmp_path))
    np.testing.assert_array_equal(first['bands'], [630.0, 800.0, 1000.0])
    np.testing.assert_array_equal(second['bands'], [630.0, 793.7, 1000.0])
    assert na.plate_table('gdbdfl', 1125.0, [630.0, 793.7, 1000.0], directory=str(tmp_path)) is second
    assert len(calls) == 2


BANDS = [630.0, 800.0, 1000.0]


def test_plate_table_reads_back_from_disk_without_recomputing(monkeypatch, tmp_path):
    import axisymmetric_bem as ab
    calls = []
    monkeypatch.setattr(ab, 'table', _plate_table_stub(2.0, calls))
    monkeypatch.setattr(na, '_PLATE_TABLES', {})
    first = na.plate_table('gdbdfl', 1125.0, BANDS, directory=str(tmp_path))
    files = list(tmp_path.glob('*.npz'))
    assert len(files) == 1 and not list(tmp_path.glob('*.tmp'))
    monkeypatch.setattr(na, '_PLATE_TABLES', {})
    second = na.plate_table('gdbdfl', 1125.0, BANDS, directory=str(tmp_path))
    assert second is not first and len(calls) == 1
    for name in ('frequencies', 'elevations', 'azimuths', 'P_d', 'P_r', 'bands'):
        np.testing.assert_array_equal(second[name], first[name])
    assert second['ground'] == first['ground'] and second['sub_bands'] == first['sub_bands']


@pytest.mark.parametrize('change', ['version', 'option', 'ground', 'sound_speed'])
def test_plate_table_recomputes_when_its_key_changes(monkeypatch, tmp_path, change):
    """Each part of the key, changed alone, makes a new table rather than
    reading the cached one, from memory or from disk."""
    import axisymmetric_bem as ab
    calls = []
    monkeypatch.setattr(ab, 'table', _plate_table_stub(0.0, calls))
    monkeypatch.setattr(na, '_PLATE_TABLES', {})
    na.plate_table('gdbdfl', 1125.0, BANDS, directory=str(tmp_path))
    arguments = dict(directory=str(tmp_path))
    speed = 1125.0
    if change == 'version':
        monkeypatch.setattr(ab, 'PLATE_TABLE_VERSION', ab.PLATE_TABLE_VERSION + 1)
    elif change == 'option':
        arguments['gauss'] = 8
    elif change == 'ground':
        arguments['ground'] = dict(na.SITE_GROUND, sigma_e=150.0)
    else:
        speed = 1150.0
    na.plate_table('gdbdfl', speed, BANDS, **arguments)
    assert len(calls) == 2
    monkeypatch.setattr(na, '_PLATE_TABLES', {})
    na.plate_table('gdbdfl', speed, BANDS, **arguments)
    assert len(calls) == 2                      # the new one is on disk too
    assert len(list(tmp_path.glob('*.npz'))) == 2


def test_plate_table_passes_its_options_to_the_bem(monkeypatch, tmp_path):
    import axisymmetric_bem as ab
    seen = {}

    def table(bands, sound_speed, **options):
        seen.update(options)
        return _plate_table_stub(0.0)(bands, sound_speed, **options)
    monkeypatch.setattr(ab, 'table', table)
    monkeypatch.setattr(na, '_PLATE_TABLES', {})
    na.plate_table('gdbdfl', 1125.0, BANDS, directory=str(tmp_path), segments_per_wavelength=6)
    assert seen['segments_per_wavelength'] == 6 and seen['mic_height'] == 0.0


def test_plate_table_ignores_old_and_mismatched_files(monkeypatch, tmp_path):
    """A pickle from before the versioned cache is never read, and an .npz
    whose stored key differs from the one asked for, or that holds a pickled
    object, is recomputed and replaced."""
    import pickle
    import axisymmetric_bem as ab
    calls = []
    monkeypatch.setattr(ab, 'table', _plate_table_stub(0.0, calls))
    monkeypatch.setattr(na, '_PLATE_TABLES', {})
    # The old cache's name for this table, with a table of the wrong gain in it.
    stale = _plate_table_stub(40.0)(np.array(BANDS), 1125.0)
    old = tmp_path / 'h0.00mm_alpha_e0.0_modelvariable_porosity_sigma_e200.0_c1125.84_b3-630-1000.pkl'
    old.write_bytes(pickle.dumps(stale))
    table = na.plate_table('gdbdfl', 1125.0, BANDS, directory=str(tmp_path))
    assert len(calls) == 1 and np.allclose(np.abs(table['P_d']), 1.0)
    assert old.exists()
    (path,) = tmp_path.glob('*.npz')
    # Same file name, another key inside: recomputed.
    ab.write_table_npz(str(path), stale, key='{"version": 0}')
    monkeypatch.setattr(na, '_PLATE_TABLES', {})
    table = na.plate_table('gdbdfl', 1125.0, BANDS, directory=str(tmp_path))
    assert len(calls) == 2 and np.allclose(np.abs(table['P_d']), 1.0)
    # A pickled object in the .npz is refused, not loaded.
    with open(path, 'wb') as handle:
        np.savez(handle, P_d=np.array([stale], dtype=object))
    monkeypatch.setattr(na, '_PLATE_TABLES', {})
    na.plate_table('gdbdfl', 1125.0, BANDS, directory=str(tmp_path))
    assert len(calls) == 3
    monkeypatch.setattr(na, '_PLATE_TABLES', {})
    na.plate_table('gdbdfl', 1125.0, BANDS, directory=str(tmp_path))
    assert len(calls) == 3                      # and the replacement reads back


@pytest.mark.parametrize('damage', ['truncated', 'empty'])
def test_plate_table_recomputes_a_truncated_or_empty_file(monkeypatch, tmp_path, damage):
    """A cut-off .npz (BadZipFile) or a 0-byte one (EOFError) at the cache name
    is listed as unreadable, then recomputed and replaced, not a crash."""
    import axisymmetric_bem as ab
    calls = []
    monkeypatch.setattr(ab, 'table', _plate_table_stub(0.0, calls))
    monkeypatch.setattr(na, '_PLATE_TABLES', {})
    na.plate_table('gdbdfl', 1125.0, BANDS, directory=str(tmp_path))
    (path,) = tmp_path.glob('*.npz')
    contents = path.read_bytes()
    path.write_bytes(contents[:len(contents) // 2] if damage == 'truncated' else b'')
    with pytest.raises(ValueError):
        ab.read_table_npz(str(path))
    assert [record['status'] for record in na.plate_table_cache(str(tmp_path))] == ['unreadable']
    monkeypatch.setattr(na, '_PLATE_TABLES', {})
    table = na.plate_table('gdbdfl', 1125.0, BANDS, directory=str(tmp_path))
    assert len(calls) == 2 and np.allclose(np.abs(table['P_d']), 1.0)
    assert [record['status'] for record in na.plate_table_cache(str(tmp_path))] == ['current']


def test_plate_table_cache_lists_and_clears(monkeypatch, tmp_path):
    import axisymmetric_bem as ab
    monkeypatch.setattr(ab, 'table', _plate_table_stub(0.0))
    monkeypatch.setattr(na, '_PLATE_TABLES', {})
    na.plate_table('gdbdfl', 1125.0, BANDS, directory=str(tmp_path))
    monkeypatch.setattr(ab, 'PLATE_TABLE_VERSION', ab.PLATE_TABLE_VERSION + 1)
    na.plate_table('gdbdfl', 1125.0, BANDS, directory=str(tmp_path))
    (tmp_path / 'old.pkl').write_bytes(b'')
    (tmp_path / 'notes.txt').write_text('kept')
    statuses = sorted(record['status'] for record in na.plate_table_cache(str(tmp_path)))
    assert statuses == ['current', 'old format', 'old version {}'.format(ab.PLATE_TABLE_VERSION - 1)]
    assert na.clear_plate_table_cache(str(tmp_path)) == 3
    assert [p.name for p in tmp_path.iterdir()] == ['notes.txt']


#: Microphone number -> (instrument type, location in ft).  Mic 4 has no
#: ambient recording, so the build drops it; the pole microphone is not a
#: ground board.
ARCHIVE_MICS = {1: ('gdbdfl', (0.0, -150.0, 0.0)), 2: ('gdbdfl', (0.0, 0.0, 0.0)),
                3: ('invgb7', (0.0, 150.0, 0.0)), 4: ('gdbdfl', (300.0, -80.0, 0.0)),
                50: ('elevtd', (0.0, 0.0, 4.0))}


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(','.join(str(value) for value in row) + '\n' for row in [header] + rows))


def _write_signal(path, pressure, fs, start_time, location):
    from netCDF4 import Dataset
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(str(path), 'w') as handle:
        handle.createDimension('samples', pressure.size)
        handle.createVariable('pressure', 'f4', ('samples',))[:] = pressure
        handle.sample_rate = fs
        handle.start_time = start_time
        handle.X, handle.Y, handle.Z = location


def _write_track(base, y=0.0):
    """Run 289108's track: level at 300 ft, 150 ft/s along x, ``y`` ft off the centerline,
    10 s at 50 Hz."""
    speed = 150.0
    _write_csv(base / 'AS350B3_AC_Data' / '289108AC.csv',
               ['utcsec', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'VGk', 'roll', 'heading'],
               [[t, speed * (t - 105.0), y, 300.0, speed, 0.0, 0.0, speed * 0.3048 / na.fa.KNOT_MPS, 0.0, 90.0]
                for t in 100.0 + np.arange(500) / 50.0])


def _archive(root):
    """A level pass and its ambient run over ARCHIVE_MICS, laid out as the 2017
    archive.  The pass is white noise 20 dB over the ambient, so the ambient
    subtraction shows if the two are scaled differently."""
    rng = np.random.default_rng(0)
    base = root / 'AS350B3'
    _write_csv(base / 'AS350B3FullRefList.csv',
               ['combined', 'test_cond', 'layout', 'bank_ang', 'accel_rate', 'utc_secs_from_mid_start', 'run_num'],
               [['289101', 'AMB', 'A', '0', '0', '60.0', '101'], ['289108', 'L1', 'A', '0', '0', '100.0', '108']])
    _write_csv(base / 'AS350B3MicFullList.csv', ['M', 'insttype'],
               [[mic, kind] for mic, (kind, _) in ARCHIVE_MICS.items()])
    _write_track(base)
    fs = 25600.0
    acoustic = base / 'AS350B3_Acoustic_Data' / '289'
    for mic, (_, location) in ARCHIVE_MICS.items():
        _write_signal(acoustic / '289108_{}_pascal.nc'.format(mic), 0.2 * rng.standard_normal(int(13 * fs)),
                      fs, 99.0, location)
        if mic != 4:
            _write_signal(acoustic / '289101_{}_pascal.nc'.format(mic), 0.02 * rng.standard_normal(int(4 * fs)),
                          fs, 60.0, location)


def test_the_plate_correction_runs_through_the_sphere_build(monkeypatch, tmp_path):
    """build_all with the default plate correction, against the flat factor.

    With a plate that reads 3 dB over free field everywhere, the plate spheres
    come out 20 log10(2) - 3 dB above the flat ones -- which holds only if the
    run and its ambient both go in unscaled -- and each retained microphone
    gets its own plate, outboard of the track.
    """
    import csv
    import axisymmetric_bem as ab
    import flight_acoustics as fa
    _archive(tmp_path)
    monkeypatch.setattr(ab, 'table', _plate_table_stub(3.0))
    monkeypatch.setattr(na, '_PLATE_TABLES', {})
    seen = []
    plate_response = na.plate_response

    def spy(tables, mirror, sound_speed):
        seen.append((len(tables), list(mirror)))
        return plate_response(tables, mirror, sound_speed)
    monkeypatch.setattr(na, 'plate_response', spy)

    spheres = {}
    for correction in ('plate_bem', 'flat'):
        manifest = tmp_path / (correction + '.csv')
        records, failures = na.build_all('AS350B3', str(tmp_path / correction), root=str(tmp_path),
                                         prefetch=False, manifest_path=str(manifest),
                                         board_correction=correction,
                                         plate_table_directory=str(tmp_path / 'tables'))
        assert failures == [] and [r['run'] for r in records] == ['289108']
        assert records[0]['mics'] == 3
        with open(manifest, newline='') as handle:
            spheres[correction] = (next(csv.DictReader(handle))['board_correction'],
                                   fa.mask_missing_levels(fa.load_nc_sphere(records[0]['output'])[0]))

    assert seen == [(3, [True, False, False])]
    assert spheres['flat'][0] == 'flat'
    assert spheres['plate_bem'][0] == 'plate_bem alpha_e=0.0 model=variable_porosity sigma_e=200.0'
    plate, flat = spheres['plate_bem'][1], spheres['flat'][1]
    expected = 20.0 * np.log10(2.0) - 3.0
    # Cells clear of the writer's -100 dB floor (minimum_level_db) in both
    # spheres; a cell kept in one and gated out of the other fails as inf.
    clear = (flat > -90.0) | (plate > -90.0 + expected)
    assert clear.sum() > 1000
    np.testing.assert_allclose(plate[clear] - flat[clear], expected, atol=1e-4)


class _StopAfterAtmosphere(Exception):
    pass


def test_the_sphere_build_warns_when_it_falls_back_to_a_standard_day(monkeypatch, tmp_path, caplog):
    """A run with no ground weather is depropagated in a standard day; that must not be silent."""
    import logging
    _archive(tmp_path)                                      # no <ac>_Weather folder at all

    def stop(test, run, mics, time_range, **kwargs):
        raise _StopAfterAtmosphere
    monkeypatch.setattr(na, 'load_run_channels', stop)
    test = na.NoiseAbatementTest('AS350B3', root=str(tmp_path))
    with caplog.at_level(logging.WARNING), pytest.raises(_StopAfterAtmosphere):
        na.build_sphere(test, '289108', str(tmp_path / 'out.nc'), board_correction='flat')
    messages = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert any('289108' in message and 'No usable ground weather' in message and 'standard day' in message
               for message in messages), messages


def test_the_recording_trim_does_not_depend_on_flip_y_for_geometry(monkeypatch, tmp_path):
    """depropagate_hemisphere flips the track and the microphones together, which changes no
    distance, so the trim range must be the same flipped or not.  It used to flip the
    microphones alone; with the track off the centerline that cut the recordings short."""
    _archive(tmp_path)
    _write_track(tmp_path / 'AS350B3', y=100.0)
    locations = np.array([location for _, location in ARCHIVE_MICS.values()], dtype=float)
    ranges = {}

    def stop(test, run, mics, time_range, **kwargs):
        ranges[flip] = time_range(locations)
        raise _StopAfterAtmosphere
    monkeypatch.setattr(na, 'load_run_channels', stop)
    test = na.NoiseAbatementTest('AS350B3', root=str(tmp_path))
    for flip in (False, True):
        with pytest.raises(_StopAfterAtmosphere):
            na.build_sphere(test, '289108', str(tmp_path / 'out.nc'), board_correction='flat',
                            flip_y_for_geometry=flip, speed_of_sound_ft_s=1125.0)
    assert ranges[True] == ranges[False]


@pytest.mark.parametrize('edge', ['start', 'end'])
def test_steady_window_leaves_an_unsteady_edge_out(edge):
    """A brief excursion is closed only between two steady stretches.

    An unsteady second at either end of the record bridges nothing; it used
    to be filled in like an interior gap, so the window began (or ended) in
    the turn-in it was meant to exclude.
    """
    track = _track()
    n = track['time'].size
    unsteady = slice(0, 50) if edge == 'start' else slice(n - 50, n)    # 1 s, under max_gap_s
    track['roll'][unsteady] = 25.0
    start, stop = na.steady_window(track)
    assert (start, stop) == ((50, n) if edge == 'start' else (0, n - 50))


def test_steady_window_takes_the_first_of_equally_long_segments():
    track = _track(duration=30.0)
    n = track['time'].size
    track['roll'][n // 2 - 150: n // 2 + 150] = 25.0         # 6 s turn in the middle
    start, stop = na.steady_window(track)
    assert (start, stop) == (0, n // 2 - 150)


def test_the_sphere_build_opens_each_channel_once(monkeypatch, tmp_path):
    """It used to open every run channel once just for its location, and
    every ambient channel once just for its start time."""
    from collections import Counter
    from netCDF4 import Dataset
    _archive(tmp_path)
    opened = Counter()

    def counting(path, *args, **kwargs):
        opened[os.path.basename(str(path))] += 1
        return Dataset(path, *args, **kwargs)
    monkeypatch.setattr(na, 'Dataset', counting)
    records, failures = na.build_all('AS350B3', str(tmp_path / 'out'), root=str(tmp_path),
                                     prefetch=False, board_correction='flat')
    assert failures == [] and records[0]['mics'] == 3
    channels = {name: count for name, count in opened.items() if name.endswith('_pascal.nc')}
    assert channels == {'{}_{}_pascal.nc'.format(run, mic): 1
                        for run in ('289108', '289101') for mic in (1, 2, 3)}


@pytest.mark.parametrize('gate_ambient', [True, False])
def test_the_prefetch_pulls_ambient_only_when_it_is_used(monkeypatch, tmp_path, gate_ambient):
    """With gate_ambient=False build_sphere reads no ambient recording, so the
    prefetch must not pull the ambient run's files either.  A run id with no
    data must not stop the batch."""
    _archive(tmp_path)
    fetched = []
    monkeypatch.setattr(na, '_prefetch', lambda paths, workers=None: fetched.extend(paths))
    records, failures = na.build_all('AS350B3', str(tmp_path / 'out'), root=str(tmp_path),
                                     runs=['289108', '999999'], board_correction='flat',
                                     gate_ambient=gate_ambient)
    assert [r['run'] for r in records] == ['289108'] and [f['run'] for f in failures] == ['999999']
    names = sorted(os.path.basename(path) for path in fetched)
    run_files = ['289108_{}_pascal.nc'.format(mic) for mic in (1, 2, 3, 4)]
    ambient_files = ['289101_{}_pascal.nc'.format(mic) for mic in (1, 2, 3)]
    assert names == sorted(run_files + (ambient_files if gate_ambient else []))


def test_the_prefetch_pulls_each_file_once_per_batch(monkeypatch, tmp_path):
    """An ambient run serves every run on its layout and day."""
    _archive(tmp_path)
    fetched = []
    monkeypatch.setattr(na, '_prefetch', lambda paths, workers=None: fetched.extend(paths))
    test = na.NoiseAbatementTest('AS350B3', root=str(tmp_path))
    done = set()
    na._prefetch_run(test, '289108', done=done)
    na._prefetch_run(test, '289108', done=done)
    assert len(fetched) == 7 and set(fetched) == done


def test_the_prefetch_pool_is_shut_down_when_the_batch_is_interrupted(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    _archive(tmp_path)
    pools = []

    class Recording(ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.closed = False
            pools.append(self)

        def shutdown(self, *args, **kwargs):
            self.closed = True
            super().shutdown(*args, **kwargs)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(na, 'ThreadPoolExecutor', Recording)
    monkeypatch.setattr(na, '_prefetch', lambda paths, workers=None: None)
    monkeypatch.setattr(na, 'build_sphere', interrupt)
    with pytest.raises(KeyboardInterrupt):
        na.build_all('AS350B3', str(tmp_path / 'out'), root=str(tmp_path))
    assert len(pools) == 1 and pools[0].closed


def test_the_cli_help_states_the_point_stride_defaults(capsys):
    """The CLI builds every 10th track sample, the API every one; the help must say so."""
    with pytest.raises(SystemExit):
        na.main(['--help'])
    help_text = ' '.join(capsys.readouterr().out.split())
    assert '--point-stride' in help_text
    assert 'default 10' in help_text and 'default to 1' in help_text



def test_the_cli_tone_split_writes_part_spheres_that_sum_to_the_total(tmp_path):
    """--tone-split writes each run's tonal and broadband spheres in directories of their
    own (a database takes every .nc in a directory), and in energy they sum to the total."""
    import flight_acoustics as fa
    _archive(tmp_path)
    out = tmp_path / 'out'
    assert na.main(['AS350B3', str(out), '--root', str(tmp_path), '--no-prefetch', '--board-correction', 'flat',
                    '--tone-split']) == 0
    spheres = list(out.glob('*.nc'))
    assert len(spheres) == 1
    total_path = spheres[0]
    stem = total_path.stem
    levels = {}
    for part, path in (('total', total_path), ('tonal', out / 'tonal' / (stem + '_tonal.nc')),
                       ('broadband', out / 'broadband' / (stem + '_broadband.nc'))):
        levels[part] = fa.mask_missing_levels(fa.load_nc_sphere(str(path))[0])
    energy = {part: 10.0 ** (level / 10.0) for part, level in levels.items()}
    # Cells clear of the writer's -100 dB floor (minimum_level_db), which clamps each part
    # alone; levels are stored to single precision
    clear = levels['total'] > -90.0
    assert clear.sum() > 1000
    np.testing.assert_allclose(energy['tonal'][clear] + energy['broadband'][clear], energy['total'][clear],
                               rtol=1e-5)
    # Every part carries the build's provenance under its own file name.
    import json
    from netCDF4 import Dataset
    stamps = {}
    for path in (total_path, out / 'tonal' / (stem + '_tonal.nc'), out / 'broadband' / (stem + '_broadband.nc')):
        with Dataset(str(path)) as sphere:
            stamps[path.name] = (sphere.panam_tool, sphere.panam_settings_sha256,
                                 json.loads(sphere.panam_inputs)['output_path'])
    assert {tool for tool, _, _ in stamps.values()} == {'noise_abatement_2017.build_sphere'}
    assert len({sha for _, sha, _ in stamps.values()}) == 1
    assert all(name == output for name, (_, _, output) in stamps.items())


def test_the_cli_node_statistics_adds_variables_and_leaves_the_levels_alone(tmp_path):
    """--node-statistics writes N_SAMPLES and AMPLITUDE_STD into each sphere, the parts
    included; without it the spheres carry neither, and the levels are the same either way."""
    from netCDF4 import Dataset
    _archive(tmp_path)
    common = ['AS350B3', '--root', str(tmp_path), '--no-prefetch', '--board-correction', 'flat', '--tone-split']
    assert na.main([common[0], str(tmp_path / 'plain')] + common[1:]) == 0
    assert na.main([common[0], str(tmp_path / 'stats')] + common[1:] + ['--node-statistics']) == 0
    name = next((tmp_path / 'plain').glob('*.nc')).name
    stem = name[:-3]
    for relative in (name, 'tonal/{}_tonal.nc'.format(stem), 'broadband/{}_broadband.nc'.format(stem)):
        with Dataset(str(tmp_path / 'plain' / relative)) as plain, \
                Dataset(str(tmp_path / 'stats' / relative)) as stats:
            assert 'N_SAMPLES' not in plain.variables and 'AMPLITUDE_STD' not in plain.variables
            np.testing.assert_array_equal(plain['AMPLITUDE'][:], stats['AMPLITUDE'][:])
            n_samples = stats['N_SAMPLES'][:]
            spread = np.ma.filled(stats['AMPLITUDE_STD'][:], np.nan)
            assert n_samples.shape == stats['AMPLITUDE'].shape and n_samples.max() > 1
            assert stats['AMPLITUDE_STD'].unit == 'DECIBEL'
            assert np.all((spread >= 0.0) | (spread == -999.0))
            assert np.any(spread > 0.0)


def test_node_statistics_left_off_keep_the_default_settings_hash(tmp_path):
    """node_statistics=False stays out of the hashed settings, so a default build keeps the
    hash of the same settings before the option existed; turned on, it is recorded."""
    import json
    from netCDF4 import Dataset
    _archive(tmp_path)
    test = na.NoiseAbatementTest('AS350B3', root=str(tmp_path))
    plain = na.build_sphere(test, '289108', str(tmp_path / 'plain.nc'), board_correction='flat')
    stats = na.build_sphere(test, '289108', str(tmp_path / 'stats.nc'), board_correction='flat',
                            node_statistics=True)
    with Dataset(plain['output']) as one, Dataset(stats['output']) as two:
        plain_settings, stats_settings = json.loads(one.panam_settings), json.loads(two.panam_settings)
    assert 'node_statistics' not in plain_settings
    assert stats_settings.pop('node_statistics') is True
    assert stats_settings == plain_settings
    assert stats['settings_sha256'] != plain['settings_sha256']


def test_tone_split_needs_tone_aware(tmp_path):
    _archive(tmp_path)
    test = na.NoiseAbatementTest('AS350B3', root=str(tmp_path))
    with pytest.raises(ValueError, match='tone_split needs tone_aware'):
        na.build_sphere(test, '289108', str(tmp_path / 'out.nc'), board_correction='flat', tone_split=True)


def _reference_index(rows, without_acoustics=()):
    """A NoiseAbatementTest over ``rows`` (run, condition, layout, seconds) with no files behind it."""
    test = object.__new__(na.NoiseAbatementTest)
    test.reference = [dict(combined=run, test_cond=condition, layout=layout, utc_secs_from_mid_start=seconds)
                      for run, condition, layout, seconds in rows]
    test.by_run = {row['combined']: row for row in test.reference}
    test.acoustic_files = {row['combined']: {1: 'x'} for row in test.reference
                           if row['combined'] not in without_acoustics}
    return test


AMBIENT_ROWS = [
    ('289101', 'AMB', 'A', '30000'),
    ('289150', 'AMB', 'A', '40000'),
    ('289160', 'AMB', 'A', '41000'),       # nearest to 289120, but has no acoustic files
    ('289170', 'AMB', 'B', '40500'),
    ('290101', 'AMB', 'A', '40900'),
    ('290102', 'AMB', 'C', '40000'),
    ('289120', 'L1', 'A', '40800'),
    ('289121', 'L1', 'B', '30000'),
    ('290120', 'L1', 'C', ''),
    ('291120', 'L1', 'A', '30100'),
    ('291121', 'L1', 'D', '30100'),
]


def test_ambient_run_prefers_the_same_day_and_layout_nearest_in_time():
    test = _reference_index(AMBIENT_ROWS, without_acoustics={'289160'})
    # 290101 is nearer in time-of-day, and 289160 nearer on the same day, but
    # 290101 is another day and 289160 has nothing to gate against.
    assert test.ambient_run('289120') == '289150'


def test_ambient_run_falls_back_to_another_day_on_the_same_layout():
    test = _reference_index(AMBIENT_ROWS, without_acoustics={'289160'})
    assert test.ambient_run('291120') == '289101'                # nearest time, any day
    assert test.ambient_run('290120') == '290102'                # no time: the only candidate


def test_ambient_run_does_not_cross_layouts():
    test = _reference_index(AMBIENT_ROWS)
    assert test.ambient_run('289121') == '289170'                # its own layout, not A's 289101
    assert test.ambient_run('291121') is None
    assert _reference_index(AMBIENT_ROWS, without_acoustics={'289170'}).ambient_run('289121') is None


def test_ambient_run_refuses_an_unknown_run():
    with pytest.raises(KeyError):
        _reference_index(AMBIENT_ROWS).ambient_run('999999')


def test_the_sphere_build_records_its_settings(tmp_path):
    """The record, the manifest and the sphere's provenance stamp carry the settings,
    and the stamp names no directory of this machine."""
    import csv
    import hashlib
    import json
    from netCDF4 import Dataset
    _archive(tmp_path)
    manifest = tmp_path / 'manifest.csv'
    records, failures = na.build_all('AS350B3', str(tmp_path / 'out'), root=str(tmp_path),
                                     prefetch=False, manifest_path=str(manifest),
                                     board_correction='flat', band_snr_gate_db=6.0,
                                     third_octave_method='filter_bank', point_stride=2)
    assert failures == []
    record = records[0]
    assert (record['band_snr_gate_db'], record['third_octave_method'], record['point_stride']) == \
        (6.0, 'filter_bank', 2)
    assert record['max_absorption_correction_db'] == 30.0 and record['tone_aware'] is False
    with open(manifest, newline='') as handle:
        row = next(csv.DictReader(handle))
    assert set(na.RECORDED_SETTINGS) | {'panam_version', 'settings_sha256'} <= set(row)
    assert (row['band_snr_gate_db'], row['gate_ambient'], row['remove_doppler']) == ('6.0', 'True', 'False')
    with Dataset(record['output']) as sphere:
        text = sphere.panam_settings
        assert sphere.panam_tool == 'noise_abatement_2017.build_sphere'
        assert sphere.panam_settings_sha256 == row['settings_sha256'] == record['settings_sha256']
        inputs = sphere.panam_inputs
    assert hashlib.sha256(text.encode()).hexdigest() == record['settings_sha256']
    settings = json.loads(text)
    assert settings['band_snr_gate_db'] == 6.0 and settings['reference_sphere_given'] is False
    assert not {'test', 'run', 'output_path', 'plate_table_directory'} & set(settings)
    identity = json.loads(inputs)
    assert identity['test'] == 'AS350B3' and identity['run'] == '289108'
    assert identity['output_path'] == os.path.basename(record['output'])
    assert identity['plate_table_directory'] is None
    for spelling in (text, inputs):
        assert str(tmp_path) not in spelling and os.path.realpath(str(tmp_path)) not in spelling


def test_spheres_built_with_equal_settings_share_the_settings_hash(tmp_path):
    """The hash covers the processing settings, not the file's name: two spheres
    written under different names with the same settings record the same hash."""
    from netCDF4 import Dataset
    _archive(tmp_path)
    test = na.NoiseAbatementTest('AS350B3', root=str(tmp_path))
    first = na.build_sphere(test, '289108', str(tmp_path / 'a' / 'one.nc'), board_correction='flat')
    second = na.build_sphere(test, '289108', str(tmp_path / 'b' / 'two.nc'), board_correction='flat')
    changed = na.build_sphere(test, '289108', str(tmp_path / 'c' / 'one.nc'), board_correction='flat',
                              band_snr_gate_db=6.0)
    assert first['settings_sha256'] == second['settings_sha256']
    assert changed['settings_sha256'] != first['settings_sha256']
    with Dataset(first['output']) as one, Dataset(second['output']) as two:
        assert one.panam_settings == two.panam_settings
        assert one.panam_inputs != two.panam_inputs


def _add_copied_run(root, run='289110', number='110'):
    """Add run ``run`` to the _archive, a copy of run 289108 (track and recordings)."""
    import shutil
    base = root / 'AS350B3'
    with open(base / 'AS350B3FullRefList.csv', 'a') as handle:
        handle.write('{},L1,A,0,0,100.0,{}\n'.format(run, number))
    shutil.copy(base / 'AS350B3_AC_Data' / '289108AC.csv', base / 'AS350B3_AC_Data' / (run + 'AC.csv'))
    acoustic = base / 'AS350B3_Acoustic_Data' / '289'
    for mic in ARCHIVE_MICS:
        shutil.copy(acoustic / '289108_{}_pascal.nc'.format(mic), acoustic / '{}_{}_pascal.nc'.format(run, mic))


def _manifest_rows(path):
    import csv
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def _no_build(*args, **kwargs):
    raise AssertionError('build_sphere was called')


def test_skip_existing_keeps_built_spheres_and_lists_them(monkeypatch, tmp_path):
    """A second build with skip_existing builds nothing, keeps the files, and still
    lists every sphere in the manifest, so the database step takes them."""
    _archive(tmp_path)
    out = tmp_path / 'out'
    first, failures = na.build_all('AS350B3', str(out), root=str(tmp_path), prefetch=False,
                                   board_correction='flat', manifest_path=str(tmp_path / 'first.csv'))
    assert failures == [] and 'skipped' not in first[0]
    assert 'skipped' not in _manifest_rows(tmp_path / 'first.csv')[0]
    written = os.path.getmtime(first[0]['output'])
    monkeypatch.setattr(na, 'build_sphere', _no_build)
    manifest = tmp_path / 'manifest.csv'
    records, failures = na.build_all('AS350B3', str(out), root=str(tmp_path), prefetch=False,
                                     board_correction='flat', band_snr_gate_db=3.0, skip_existing=True,
                                     manifest_path=str(manifest))
    assert failures == [] and [r['run'] for r in records] == ['289108']
    record = records[0]
    assert record['skipped'] == 'exists' and record['output'] == first[0]['output']
    assert record['settings_sha256'] == first[0]['settings_sha256']
    np.testing.assert_allclose(record['speed_knots'], first[0]['speed_knots'], rtol=1e-6)
    assert os.path.getmtime(record['output']) == written
    assert _manifest_rows(manifest)[0]['skipped'] == 'exists'
    assert na.select_from_manifest(str(manifest)) == [os.path.basename(record['output'])]


def test_skip_existing_rebuilds_a_run_missing_one_of_its_outputs(tmp_path):
    """With tone_split, a run whose part sphere is gone is built again."""
    _archive(tmp_path)
    out = tmp_path / 'out'
    first, _ = na.build_all('AS350B3', str(out), root=str(tmp_path), prefetch=False,
                            board_correction='flat', tone_aware=True, tone_split=True)
    os.remove(first[0]['tonal_output'])
    records, failures = na.build_all('AS350B3', str(out), root=str(tmp_path), prefetch=False,
                                     board_correction='flat', tone_aware=True, tone_split=True,
                                     skip_existing=True)
    assert failures == [] and 'skipped' not in records[0]
    assert os.path.exists(first[0]['tonal_output'])


def test_skip_unchanged_rebuilds_only_on_new_settings_or_inputs(monkeypatch, tmp_path):
    from netCDF4 import Dataset
    import provenance
    _archive(tmp_path)
    out = tmp_path / 'out'
    arguments = dict(root=str(tmp_path), prefetch=False, board_correction='flat', point_stride=2)
    first, _ = na.build_all('AS350B3', str(out), **arguments)
    test = na.NoiseAbatementTest('AS350B3', root=str(tmp_path))
    signature = provenance.file_signature(na.run_input_files(test, '289108'))
    with Dataset(first[0]['output']) as sphere:
        assert sphere.panam_input_signature == signature == first[0]['input_signature']

    built = []
    build_sphere = na.build_sphere

    def counting(test, run, *args, **kwargs):
        built.append(run)
        return build_sphere(test, run, *args, **kwargs)
    monkeypatch.setattr(na, 'build_sphere', counting)

    records, _ = na.build_all('AS350B3', str(out), skip_unchanged=True, **arguments)
    assert built == [] and records[0]['skipped'] == 'unchanged'
    records, _ = na.build_all('AS350B3', str(out), skip_unchanged=True, **dict(arguments, point_stride=3))
    assert built == ['289108'] and 'skipped' not in records[0]
    records, _ = na.build_all('AS350B3', str(out), skip_unchanged=True, **dict(arguments, point_stride=3))
    assert built == ['289108'] and records[0]['skipped'] == 'unchanged'
    # A recording replaced since: same name, a new modification time.
    recording = tmp_path / 'AS350B3' / 'AS350B3_Acoustic_Data' / '289' / '289101_2_pascal.nc'
    stamp = os.stat(recording)
    os.utime(recording, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 10 ** 9))
    records, _ = na.build_all('AS350B3', str(out), skip_unchanged=True, **dict(arguments, point_stride=3))
    assert built == ['289108', '289108'] and 'skipped' not in records[0]


def test_skip_unchanged_with_no_overwrite_skips_a_default_build(monkeypatch, tmp_path):
    """overwrite changes no level, so a resumed --skip-unchanged --no-overwrite build
    keeps the sphere a default (overwrite=True) build wrote instead of failing on it."""
    _archive(tmp_path)
    out = tmp_path / 'out'
    arguments = dict(root=str(tmp_path), prefetch=False, board_correction='flat', point_stride=2)
    first, failures = na.build_all('AS350B3', str(out), **arguments)
    assert failures == []
    monkeypatch.setattr(na, 'build_sphere', _no_build)
    records, failures = na.build_all('AS350B3', str(out), skip_unchanged=True, overwrite=False,
                                     **arguments)
    assert failures == [] and [r['skipped'] for r in records] == ['unchanged']
    assert records[0]['settings_sha256'] == first[0]['settings_sha256']


def test_a_skipped_run_keeps_the_manifest_row_of_its_build(monkeypatch, tmp_path):
    """A resumed build lists a skipped run as its original build did: its track flags
    (so --skip-flagged still leaves a flagged run out), its rays, window and settings,
    and its float64 speed and flight path angle in the NORAH2 triangulation."""
    _archive(tmp_path)
    out, norah2 = tmp_path / 'out', tmp_path / 'norah2'
    check = na.run_track_check

    def flagging(*args, **kwargs):
        result = check(*args, **kwargs)
        result.flags.append('a flag for the test')
        return result
    monkeypatch.setattr(na, 'run_track_check', flagging)
    arguments = dict(root=str(tmp_path), prefetch=False, board_correction='flat', point_stride=2,
                     runs=['289108'], norah2_directory=str(norah2))
    first, failures = na.build_all('AS350B3', str(out), manifest_path=str(tmp_path / 'first.csv'),
                                   **arguments)
    assert failures == [] and 'a flag for the test' in first[0]['track_flags']
    assert na.select_from_manifest(str(tmp_path / 'first.csv'), skip_flagged=True) == []
    monkeypatch.setattr(na, 'build_sphere', _no_build)
    triangulation = []
    monkeypatch.setattr(na.fa, 'write_norah2_triangulation',
                        lambda path, entries: triangulation.append(entries))
    for option in ('skip_existing', 'skip_unchanged'):
        manifest = tmp_path / '{}.csv'.format(option)
        records, failures = na.build_all('AS350B3', str(out), manifest_path=str(manifest),
                                         **{option: True}, **arguments)
        assert failures == [] and records[0]['skipped'] in ('exists', 'unchanged')
        record = dict(records[0])
        del record['skipped']
        assert _comparable(record) == _comparable(first[0])
        assert type(record['window_s']) is tuple
        assert na.select_from_manifest(str(manifest), skip_flagged=True) == []
        rows, original = _manifest_rows(manifest)[0], _manifest_rows(tmp_path / 'first.csv')[0]
        assert {key: rows[key] for key in original} == original
    # Bypass build_all (three records are needed): the triangulation carries the float64 values.
    assert triangulation == []
    record = na._skipped_record(na.NoiseAbatementTest('AS350B3', root=str(tmp_path)), '289108',
                                first[0]['output'], str(norah2), 'exists')
    assert record['speed_knots'] == first[0]['speed_knots']
    assert record['flight_path_angle_deg'] == first[0]['flight_path_angle_deg']
    assert record['norah2_output'] == first[0]['norah2_output']


def test_a_skipped_sphere_without_a_build_record_counts_as_flagged(tmp_path):
    """A sphere written before build records were gives no track flags, so the
    manifest marks them unknown and --skip-flagged leaves the run out."""
    from netCDF4 import Dataset
    import provenance
    _archive(tmp_path)
    out = tmp_path / 'out'
    first, _ = na.build_all('AS350B3', str(out), root=str(tmp_path), prefetch=False,
                            board_correction='flat', point_stride=2)
    with Dataset(first[0]['output'], 'a') as sphere:
        sphere.delncattr(provenance.BUILD_RECORD_ATTRIBUTE)
    manifest = tmp_path / 'manifest.csv'
    records, _ = na.build_all('AS350B3', str(out), root=str(tmp_path), prefetch=False,
                              board_correction='flat', point_stride=2, skip_existing=True,
                              manifest_path=str(manifest))
    assert records[0]['track_flags'] == na.UNKNOWN_TRACK_FLAGS
    np.testing.assert_allclose(records[0]['speed_knots'], first[0]['speed_knots'], rtol=1e-6)
    assert na.select_from_manifest(str(manifest)) == [os.path.basename(first[0]['output'])]
    assert na.select_from_manifest(str(manifest), skip_flagged=True) == []


def _comparable(record):
    """A build record with its output directories left out, and NaN as text (NaN != NaN)."""
    def plain(key, value):
        if key.endswith('output') and value:
            return os.path.basename(value)
        return 'nan' if isinstance(value, float) and np.isnan(value) else value
    return {key: plain(key, value) for key, value in record.items()}


def test_parallel_build_matches_the_serial_one(monkeypatch, tmp_path):
    """jobs=2 gives the records, failures, manifest and spheres of jobs=1, in run
    order.  The plate tables are computed in this process (here by a stub the
    workers do not see), so the workers read them from the cache."""
    import functools
    import axisymmetric_bem as ab
    import flight_acoustics as fa
    _archive(tmp_path)
    _add_copied_run(tmp_path)
    # With the real table's signature, so that the cache key is the one the workers compute.
    monkeypatch.setattr(ab, 'table', functools.wraps(ab.table)(_plate_table_stub(3.0)))
    results = {}
    for jobs in (1, 2):
        monkeypatch.setattr(na, '_PLATE_TABLES', {})
        out = tmp_path / 'jobs{}'.format(jobs)
        records, failures = na.build_all('AS350B3', str(out), root=str(tmp_path), prefetch=False,
                                         runs=['289110', '999999', '289108'], jobs=jobs, point_stride=2,
                                         plate_table_directory=str(tmp_path / 'tables{}'.format(jobs)),
                                         manifest_path=str(out / 'manifest.csv'))
        rows = _manifest_rows(out / 'manifest.csv')
        levels = [fa.load_nc_sphere(r['output'])[0] for r in records]
        results[jobs] = ([_comparable(r) for r in records], failures,
                         [_comparable(row) for row in rows], levels)
    serial, parallel = results[1], results[2]
    assert [r['run'] for r in parallel[0]] == ['289110', '289108']
    assert [f['run'] for f in parallel[1]] == ['999999']
    assert parallel[0] == serial[0] and parallel[1] == serial[1] and parallel[2] == serial[2]
    for one, two in zip(serial[3], parallel[3]):
        np.testing.assert_array_equal(one, two)


def test_parallel_build_refuses_what_a_worker_cannot_receive(tmp_path):
    _archive(tmp_path)
    with pytest.raises(TypeError, match='ray_models'):
        na.build_all('AS350B3', str(tmp_path / 'out'), root=str(tmp_path), jobs=2,
                     ray_models=lambda run: None)
    with pytest.raises(ValueError, match='jobs'):
        na.build_all('AS350B3', str(tmp_path / 'out'), root=str(tmp_path), jobs=0)


@pytest.mark.parametrize('flags, info, debug', [((), True, False), (('-q',), False, False),
                                                 (('-v',), True, True)])
def test_the_cli_log_level_and_log_file(tmp_path, flags, info, debug):
    import logging
    _archive(tmp_path)
    log = tmp_path / 'logs' / 'build.log'
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    # Run 999999 has no data: its prefetch logs a DEBUG message.
    assert na.main(['AS350B3', str(tmp_path / 'out'), '--root', str(tmp_path), '--board-correction', 'flat',
                    '--runs', '289108', '999999', '--log-file', str(log), *flags]) == 0
    text = log.read_text()
    assert ('INFO [1/2] 289108 -> ' in text) is info
    assert ('DEBUG prefetch skipped run 999999' in text) is debug
    assert 'WARNING [2/2] 999999 failed' in text
    assert root.level == level and root.handlers == handlers
