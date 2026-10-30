"""The 2019 Noise Abatement test (Coyle Field, the S-76D) through the 2017 pipeline: its reference
list's run time, whole-test station file, microphone list heights, track datum and year."""

import numpy as np
import pytest
from netCDF4 import Dataset

from panam import noise_abatement_2017 as na


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(','.join(str(value) for value in row) + '\n' for row in [header] + rows))


def _dataset(tmp_path, weather_folder=False):
    """A one-run S76D test in the 2019 layout: the reference list with utc_secs_from_mid, a
    track whose alt - z is 54.27 ft against ref_elips_ft 62.8, one recording dated 2019, and
    the whole-test station file."""
    base = tmp_path / 'S76D'
    _write_csv(base / 'S76DFullRefList.csv',
               ['flt_num', 'run_num', 'combined', 'test_cond', 'utc_secs_from_mid', 'layout', 'ref_lat',
                'ref_lon', 'ref_elips_ft', 'true_heading', 'mic_loc_file'],
               [[197, 421, '197421', 'A24', 50737, 'Turn', 39.81042423, -74.42630869, 62.8, 210,
                 'CoyleNoiseAbatMicFullList.csv'],
                [197, 931, '197931', 'AMB', 50000, 'Turn', 39.81042423, -74.42630869, 62.8, 210,
                 'CoyleNoiseAbatMicFullList.csv']])
    t = 50737.0 + np.arange(200) / 50.0
    _write_csv(base / 'S76D_AC_Data' / '197421AC.csv',
               ['utcsec', 'x', 'y', 'z', 'alt', 'vx', 'vy', 'vz', 'VGk', 'roll', 'heading'],
               [[s, 100.0 * (s - t[0]), 0.0, 500.0 - (s - t[0]), 554.27 - (s - t[0]), 100.0, 0.0, -1.0, 59.2, 0.0,
                 210.0] for s in t])
    for run in ('197421', '197931'):
        path = base / 'S76D_Acoustic_Data' / 'S76D_197_pascal' / f'{run}_01_pascal.nc'
        path.parent.mkdir(parents=True, exist_ok=True)
        with Dataset(str(path), 'w') as handle:
            handle.createDimension('samples', 10)
            handle.createVariable('pressure', 'f4', ('samples',))[:] = np.zeros(10)
            handle.sample_rate, handle.start_time = 25000.0, 50737.0
            handle.X, handle.Y, handle.Z = 0.0, 0.0, 10.58
            handle.date = 'June/July 2019     '
    _write_csv(base / 'CoyleNoiseAbatMicFullList.csv', ['M', 'Latitude(utm-decdeg)', 'Longitude(utm-decdeg)',
                                                       'Elips(m)', 'insttype'],
               [[1, 39.81495932, -74.43084095, 22.40, 'gdbdfl']])
    _write_csv(base / 'S76D_all_SWS_weather.csv',
               ['fltnum', ' run', ' comb', ' WS00 avgt', ' WS00 avgws', ' WS00 avgwd', ' WS00 avgpress',
                ' WS00 avghumid', ' WS11 avgt', ' WS11 avgws', ' WS11 avgwd', ' WS11 avgpress', ' WS11 avghumid'],
               [[197, 421, 197421, 77.0, 3.0, 240, 100.4, 70.0, 81.0, 2.0, 250, 100.6, 60.0],
                [197, 931, 197931, 60.0, 3.0, 240, 100.0, 50.0, '', 2.0, 250, '', '']])
    if weather_folder:
        (base / 'S76D_Weather').mkdir()
    return tmp_path


def test_reference_seconds_reads_either_column_name():
    assert na.reference_seconds({'utc_secs_from_mid_start': '39910'}) == 39910.0
    assert na.reference_seconds({'utc_secs_from_mid': '50737'}) == 50737.0
    assert np.isnan(na.reference_seconds({}))
    assert np.isnan(na.reference_seconds(None))


def test_the_whole_test_station_file_gives_the_runs_atmosphere(tmp_path):
    """Its stations' means, F and kPa as the per-day files; a blank station is left out."""
    test = na.NoiseAbatementTest('S76D', root=str(_dataset(tmp_path)))
    atmosphere = na.run_atmosphere(test, '197421')
    assert atmosphere.temperature == pytest.approx((79.0 - 32.0) / 1.8 + 273.15)
    assert atmosphere.relative_humidity == pytest.approx(65.0)
    assert atmosphere.pressure == pytest.approx(100.5)
    ambient = na.run_atmosphere(test, '197931')
    assert ambient.temperature == pytest.approx((60.0 - 32.0) / 1.8 + 273.15)
    assert str(tmp_path / 'S76D' / 'S76D_all_SWS_weather.csv') in na.run_input_files(test, '197421')


def test_a_test_with_a_weather_folder_keeps_its_per_day_stations(tmp_path):
    """2017's tests carry their own copy of the whole-test file, and it goes unread: their
    builds do not move."""
    test = na.NoiseAbatementTest('S76D', root=str(_dataset(tmp_path, weather_folder=True)))
    with pytest.raises(ValueError, match='No usable ground weather'):
        na.run_atmosphere(test, '197421')


def test_the_ambient_run_is_the_nearest_by_the_2019_run_time(tmp_path):
    test = na.NoiseAbatementTest('S76D', root=str(_dataset(tmp_path)))
    assert test.ambient_run('197421') == '197931'


def test_the_year_comes_from_the_recordings(tmp_path):
    test = na.NoiseAbatementTest('S76D', root=str(_dataset(tmp_path)))
    assert test.year == 2019
    assert '06/25/2019' in na.sphere_title('S-76D', 106, 176, year=test.year)


def test_the_reference_datum_takes_z_from_the_altitude(tmp_path):
    """Coyle Field's tracks have alt - z = 54.27 ft, the microphones' datum ref_elips_ft 62.8:
    'reference' puts the aircraft 8.53 ft lower; 'track' keeps the file's z."""
    root = str(_dataset(tmp_path))
    track = na.NoiseAbatementTest('S76D', root=root).load_track('197421')
    reference = na.NoiseAbatementTest('S76D', root=root, track_datum='reference').load_track('197421')
    np.testing.assert_allclose(track['z'], na.load_track(str(tmp_path / 'S76D' / 'S76D_AC_Data' / '197421AC.csv'))['z'])
    np.testing.assert_allclose(reference['z'], track['z'] - 8.53, atol=1e-9)
    with pytest.raises(ValueError, match='track_datum'):
        na.NoiseAbatementTest('S76D', root=root, track_datum='geoid')


def test_the_2019_microphone_list_gives_ellipsoidal_heights(tmp_path):
    """'Elips(m)', where 2017's lists say 'Hgt(m)': microphone 1 at 22.40 m is 10.69 ft above
    the reference point's 62.8 ft, as its recording's Z (10.58 ft) has it to 0.1 ft."""
    test = na.NoiseAbatementTest('S76D', root=str(_dataset(tmp_path)))
    positions, reason = na.ground_microphone_positions(test, '197421')
    assert reason is None
    assert positions[0, 2] == pytest.approx(22.40 / 0.3048 - 62.8)


def test_the_s76d_spheres_take_a_five_character_prefix():
    assert na.AIRCRAFT_SPHERE_PREFIX['S76D'] == 'S-76D'


def test_a_channel_the_logs_call_bad_is_left_out_of_its_days_runs(tmp_path, monkeypatch):
    """BAD_CHANNELS drops a microphone from the runs its prefix matches (day 176's mic 33 at
    Coyle Field), and from no others."""
    test = na.NoiseAbatementTest('S76D', root=str(_dataset(tmp_path)))
    assert test.ground_board_mics('197421') == [1]
    monkeypatch.setitem(na.BAD_CHANNELS, ('S76D', '197'), {1: 'test'})
    assert test.ground_board_mics('197421') == []
    assert test.bad_channel('197421', 1) == 'test' and test.bad_channel('198421', 1) is None
    assert 'BAD THROUGHOUT' in test.bad_channel('176108', 33)


def _climb_card_track(climb_fpa=(16.5, 19.5), climb_s=10.0, rate=50.0):
    """A 2019 climb card: a 25 s shallow descent at 75 kt, then a climb at 70 kt whose path
    angle drifts from climb_fpa[0] to climb_fpa[1] over climb_s, then 10 s level."""
    parts = [(25.0, 75.0, np.full(int(25.0 * rate), -3.0)),
             (climb_s, 70.0, np.linspace(*climb_fpa, int(climb_s * rate))),
             (10.0, 72.0, np.full(int(10.0 * rate), 0.5))]
    fpa = np.concatenate([p[2] for p in parts])
    speed = np.concatenate([np.full(p[2].size, p[1]) for p in parts])
    n = fpa.size
    time = np.arange(n) / rate
    return dict(time=time, ground_speed_knots=speed, fpa_deg=fpa, roll=np.zeros(n), heading=np.full(n, 30.0),
                x=-3000.0 + np.cumsum(speed) * 1.68781 / rate, y=np.zeros(n), z=np.full(n, 200.0))


def test_the_card_names_the_sign_of_its_path():
    assert na.card_path_sign({'fpa': 'BROC'}) == 1
    assert na.card_path_sign({'fpa': '6'}) == 1
    assert na.card_path_sign({'fpa': '-7.5'}) == -1
    assert na.card_path_sign({'fpa': '0'}) == 0
    assert na.card_path_sign({'fpa': 'Fly Neighborly'}) is None
    assert na.card_path_sign({'fpa': ''}) is None


def test_the_own_condition_window_takes_the_cards_climb_not_the_run_in():
    """Without the card's sign the floating band takes the longest steady stretch, the run-in;
    with it, the climb, whose path angle drifts 3 deg, inside the band's 4."""
    track = _climb_card_track()
    start, stop = na.local_steady_window(track)
    assert np.median(track['fpa_deg'][start:stop]) == pytest.approx(-3.0)
    start, stop = na.local_steady_window(track, path_sign=1)
    climb = track['fpa_deg'][start:stop]
    assert climb.min() > 16.0 and track['time'][stop - 1] - track['time'][start] == pytest.approx(10.0, abs=0.05)


def test_the_own_condition_window_holds_the_band_width():
    """A climb whose path angle drifts 6 deg over 10 s holds the 4 deg band for under 8 s."""
    track = _climb_card_track(climb_fpa=(15.0, 21.0))
    with pytest.raises(ValueError, match='about its own condition'):
        na.local_steady_window(track, path_sign=1)


def test_the_fallback_is_taken_only_where_the_usual_window_fails(monkeypatch):
    track = _climb_card_track()
    usual = na.steady_window(track)
    assert na.find_steady_window(track, local_fallback=True, path_sign=1) == (*usual, 'near half')

    def no_window(*args, **kwargs):
        raise ValueError('No steady flight segment found')
    monkeypatch.setattr(na, 'steady_window', no_window)
    with pytest.raises(ValueError, match='No steady'):
        na.find_steady_window(track, path_sign=1)
    start, stop, reference = na.find_steady_window(track, local_fallback=True, path_sign=1)
    assert reference == 'own' and track['fpa_deg'][start:stop].min() > 16.0
