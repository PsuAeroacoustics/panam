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
