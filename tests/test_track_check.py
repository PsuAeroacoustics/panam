"""The build's track sanity check and the track frame it rests on (noise_abatement_2017).

A run whose tracked position is impossible over its depropagation window is refused
with the reason; a window worth a second look is flagged.  The synthetic tracks here
stand in for the cases found in the 2017 data (B407 285236 below the boards, its twin
285235 above them, 283101's window ending at an aborted pull-up); the tests marked
``data`` read those runs themselves when the dataset is configured.
"""

import csv
import os

import numpy as np
import pytest

import local_paths
import noise_abatement_2017 as na


RATE = 50.0


def _descent(bottom_ft, fpa_deg=-7.5, speed_kt=80.0, duration=40.0, after=None):
    """A loaded track (load_track's fields) descending at ``fpa_deg`` along +x to
    ``bottom_ft`` at x = 0, where it ends; ``after``, if given, is a dict of
    'deceleration' (ft/s^2) and 'climb_deg' flown for its last 5 s instead."""
    n = int(duration * RATE)
    time = np.arange(n) / RATE
    speed = np.full(n, speed_kt * 1.68781)
    fpa = np.full(n, float(fpa_deg))
    if after is not None:
        tail = time >= duration - 5.0
        speed[tail] -= after.get('deceleration', 0.0) * (time[tail] - (duration - 5.0))
        fpa[tail] = after.get('climb_deg', fpa_deg)
    vx = speed
    vz_up = speed * np.tan(np.radians(fpa))
    x = np.concatenate(([0.0], np.cumsum(vx[1:]) / RATE))
    z = np.concatenate(([0.0], np.cumsum(vz_up[1:]) / RATE))
    x -= x[-1]
    z += bottom_ft - z[-1] if after is None else bottom_ft - np.min(z)
    return dict(time=time, x=x, y=np.zeros(n), z=z, vx=vx, vy=np.zeros(n), vz_up=vz_up,
                fpa_deg=np.degrees(np.arctan2(vz_up, vx)), ground_speed_knots=speed / 1.68781,
                heading=np.full(n, 270.0), roll=np.zeros(n))


def _whole(track, stop=None):
    return 0, (track['time'].size if stop is None else stop)


# --- heights --------------------------------------------------------------------------

def test_a_descent_below_the_microphones_is_refused_and_its_twin_is_not():
    """285236 reached 79 ft below the boards over the array; 285235, the same card,
    47 ft above them."""
    low = na.check_track(_descent(-79.0), *_whole(_descent(-79.0)))
    high = na.check_track(_descent(47.0), *_whole(_descent(47.0)))
    assert low.refused and 'below the microphones' in low.refusals[0]
    assert low.metrics['min_height_above_array_ft'] == pytest.approx(-79.0, abs=1.0)
    assert not high.refused and high.refusals == []


def test_within_the_minimum_height_is_refused_and_none_turns_the_test_off():
    track = _descent(6.0)
    check = na.check_track(track, *_whole(track))
    assert check.refused and 'within 6.0 ft' in check.refusals[0]
    assert not na.check_track(track, *_whole(track), min_height_ft=5.0).refused
    assert not na.check_track(track, *_whole(track), min_height_ft=None).refused
    # Exactly at the minimum is allowed: only closer is refused.
    level = _descent(10.0, fpa_deg=0.0)
    assert not na.check_track(level, *_whole(level)).refused


def test_only_the_window_is_checked():
    """A track that goes low after its window is not refused for it."""
    track = _descent(-50.0)
    below = int(np.argmax(track['z'] < 10.0))
    assert not na.check_track(track, 0, below).refused
    assert na.check_track(track, 0, below + 1).refused


def test_the_ground_under_the_aircraft_is_the_nearest_board():
    """On a sloping array the board under the aircraft, not the frame's z = 0, is the
    ground: a pass 25 ft above the frame is 5 ft above a board standing 20 ft up."""
    track = _descent(25.0, fpa_deg=0.0)
    boards = np.array([[0.0, 0.0, 20.0], [-3000.0, 0.0, 0.0], [3000.0, 300.0, 0.0]])
    assert not na.check_track(track, *_whole(track)).refused
    check = na.check_track(track, *_whole(track), ground_mics=boards)
    assert check.refused and 'within 5.0 ft' in check.refusals[0]
    assert check.metrics['plane'] == 'nearest of 3 ground boards'
    # Far from the high board the same height is fine.
    far = na.check_track(track, 0, int(0.3 * track['time'].size), ground_mics=boards)
    assert not far.refused


def test_z_that_does_not_follow_the_altitude_is_refused():
    track = _descent(200.0)
    track['alt'] = track['z'] + 3926.51
    assert not na.check_track(track, *_whole(track), ref_elips_ft=3926.51).refused
    track['alt'][100:200] += 25.0
    check = na.check_track(track, *_whole(track), ref_elips_ft=3926.51)
    assert check.refused and 'departs from its altitude' in check.refusals[0]
    assert check.metrics['max_altitude_mismatch_ft'] == pytest.approx(25.0)
    # Without the reference height there is nothing to compare.
    assert not na.check_track(track, *_whole(track)).refused
    # None turns the test off, and an altitude that is not finite is refused with that reason.
    assert not na.check_track(track, *_whole(track), ref_elips_ft=3926.51, max_altitude_mismatch_ft=None).refused
    track['alt'][300] = np.nan
    check = na.check_track(track, *_whole(track), ref_elips_ft=3926.51)
    assert check.refused and 'altitude is not finite' in check.refusals[-1]
    assert not na.check_track(track, *_whole(track), ref_elips_ft=3926.51, max_altitude_mismatch_ft=None).refused


def test_a_height_correction_must_move_the_altitude_with_z():
    """The R66's altitude reads ~30 ft low and the harness raises z before building; raising
    z alone leaves it 30 ft from its own altitude, which the check refuses."""
    track = _descent(-20.0, fpa_deg=0.0)
    track['alt'] = track['z'] + 3926.51
    track['z'] = track['z'] + 30.0
    check = na.check_track(track, *_whole(track), ref_elips_ft=3926.51)
    assert check.refused and 'correct alt by the same amount' in check.refusals[0]
    track['alt'] = track['alt'] + 30.0
    assert not na.check_track(track, *_whole(track), ref_elips_ft=3926.51).refused


def test_a_position_that_is_not_finite_is_refused():
    track = _descent(200.0)
    track['z'][10] = np.nan
    assert na.check_track(track, *_whole(track)).refused


# --- flags ----------------------------------------------------------------------------

def test_a_window_ending_at_a_decelerating_level_off_is_flagged_but_not_refused():
    """283101: the steady window ends 2 s before a decelerating pull-up.  Both the
    deceleration and the rise in flight path angle are needed for the flag."""
    window_stop = int(35.0 * RATE)
    both = _descent(300.0, fpa_deg=-3.0, after=dict(deceleration=6.0, climb_deg=3.0))
    check = na.check_track(both, 0, window_stop)
    assert not check.refused
    assert len(check.flags) == 1 and 'decelerating level-off' in check.flags[0]
    decelerating = _descent(300.0, fpa_deg=-3.0, after=dict(deceleration=6.0))
    climbing = _descent(300.0, fpa_deg=-3.0, after=dict(climb_deg=3.0))
    assert na.check_track(decelerating, 0, window_stop).flags == []
    assert na.check_track(climbing, 0, window_stop).flags == []
    # Past the lookahead it is no longer the window's end.
    assert na.check_track(both, 0, int(30.0 * RATE)).flags == []


def test_a_deceleration_inside_the_window_is_flagged():
    track = _descent(300.0, fpa_deg=-3.0, after=dict(deceleration=6.0))
    check = na.check_track(track, *_whole(track))
    assert any('decelerates at' in flag and 'inside the window' in flag for flag in check.flags)
    assert check.metrics['min_along_track_acceleration_ft_s2'] < -3.0
    assert not check.refused
    gentle = _descent(300.0, fpa_deg=-3.0, after=dict(deceleration=2.0))
    assert na.check_track(gentle, *_whole(gentle)).flags == []


def test_a_window_away_from_the_cards_flight_path_angle_is_flagged():
    track = _descent(300.0, fpa_deg=0.5)
    assert 'not the card' in na.check_track(track, *_whole(track), card_fpa_deg=-3.0).flags[0]
    assert na.check_track(track, *_whole(track), card_fpa_deg=-1.0).flags == []
    assert na.check_track(track, *_whole(track), card_fpa_deg=float('nan')).flags == []


def test_a_hover_takes_no_motion_flags():
    track = _descent(300.0, fpa_deg=-3.0, after=dict(deceleration=6.0))
    check = na.check_track(track, *_whole(track), card_fpa_deg=20.0, motion_flags=False)
    assert check.flags == []


# --- the track frame ------------------------------------------------------------------

def test_compass_directions_land_where_the_2017_frames_put_them():
    """+x along the reference list's true_heading, +y 90 deg to its left (z up)."""
    np.testing.assert_allclose(na.heading_to_frame(270.0, 270.0), (1.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(na.heading_to_frame(90.0, 270.0), (-1.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(na.heading_to_frame(180.0, 270.0), (0.0, 1.0), atol=1e-12)    # +y is south
    np.testing.assert_allclose(na.heading_to_frame(50.0, 140.0), (0.0, 1.0), atol=1e-12)     # Eglin: +y along 50
    np.testing.assert_allclose(na.heading_to_frame(279.0, 279.0), (1.0, 0.0), atol=1e-12)
    east, north = np.array([3.0, -1.0, 0.5]), np.array([2.0, 4.0, -7.0])
    for bearing in (270.0, 279.0, 140.0, 92.3):
        x, y = na.east_north_to_frame(bearing, east, north)
        np.testing.assert_allclose(na.frame_to_east_north(bearing, x, y), (east, north), atol=1e-12)


def _geodetic_track(bearing, lat0=40.25, lon0=-120.15, n=200):
    x = np.linspace(-3000.0, 1000.0, n)
    y = np.linspace(-20.0, 30.0, n)
    east, north = na.frame_to_east_north(bearing, x, y)
    lat0r = np.radians(lat0)
    w = 1.0 - na.WGS84_E2 * np.sin(lat0r) ** 2
    lat = lat0 + np.degrees(north / (na.WGS84_A_FT * (1.0 - na.WGS84_E2) / w ** 1.5))
    lon = lon0 + np.degrees(east / (na.WGS84_A_FT / np.sqrt(w) * np.cos(lat0r)))
    return dict(x=x, y=y, lat=lat, lon=lon), dict(ref_lat=str(lat0), ref_lon=str(lon0))


def test_the_frame_bearing_is_the_reference_lists_and_must_match_the_track():
    track, row = _geodetic_track(279.0)
    assert na.frame_bearing_deg(dict(row, true_heading='279'), track) == pytest.approx(279.0)
    assert na.frame_bearing_deg(row, track) == pytest.approx(279.0, abs=1e-6)        # fitted alone
    assert na.frame_bearing_deg(dict(true_heading='279')) == 279.0                   # listed alone
    with pytest.raises(ValueError, match='true_heading 270'):
        na.frame_bearing_deg(dict(row, true_heading='270'), track)
    with pytest.raises(ValueError, match='no true_heading'):
        na.frame_bearing_deg({})
    # A hover's few feet of drift fix no direction: the listed value stands.
    hover = {key: value[:2] for key, value in track.items()}
    assert na.frame_bearing_deg(dict(row, true_heading='270'), hover) == 270.0


def test_ground_microphones_come_from_the_layouts_list(tmp_path):
    """Latitude, longitude and ellipsoidal height in meters to the track frame, ground
    boards only, heights relative to ref_elips_ft."""
    base = tmp_path / 'B407'
    base.mkdir()
    bearing, lat0, lon0, elips0 = 140.0, 30.64, -86.22, 99.04
    boards = {1: (100.0, -50.0, 3.0), 2: (-2500.0, 400.0, -7.0), 3: (0.0, 0.0, 12.5)}
    rows = []
    for mic, (x, y, z) in boards.items():
        east, north = na.frame_to_east_north(bearing, x, y)
        lat0r = np.radians(lat0)
        w = 1.0 - na.WGS84_E2 * np.sin(lat0r) ** 2
        lat = lat0 + np.degrees(north / (na.WGS84_A_FT * (1.0 - na.WGS84_E2) / w ** 1.5))
        lon = lon0 + np.degrees(east / (na.WGS84_A_FT / np.sqrt(w) * np.cos(lat0r)))
        rows.append([mic, '{:.12f}'.format(lat), '{:.12f}'.format(lon), '{:.9f}'.format((z + elips0) * 0.3048),
                     'gdbdfl'])
    rows.append([50, '30.64', '-86.22', '40.0', 'elevtd'])
    with open(base / 'EglinMics.csv', 'w', newline='') as handle:
        handle.write('﻿M,Latitude(utm-decdeg),Longitude(utm-decdeg),Hgt(m),insttype\r')
        for row in rows:
            handle.write(','.join(str(v) for v in row) + '\r')     # Eglin's lists end lines in CR alone

    class Test:
        pass
    test = Test()
    test.base = str(base)
    test.by_run = {'230000': dict(mic_loc_file='EglinMics.csv', ref_lat=str(lat0), ref_lon=str(lon0),
                                  ref_elips_ft=str(elips0), true_heading=str(bearing))}
    positions, reason = na.ground_microphone_positions(test, '230000')
    assert reason is None
    np.testing.assert_allclose(positions, np.array(list(boards.values())), atol=1e-6)
    test.by_run['230000']['mic_loc_file'] = 'missing.csv'
    positions, reason = na.ground_microphone_positions(test, '230000')
    assert positions is None and 'no microphone list' in reason


# --- through the build ----------------------------------------------------------------

def _archive_with_track(root, z_ft):
    """test_noise_abatement_2017's synthetic archive with its pass flown at ``z_ft``."""
    from test_noise_abatement_2017 import _archive, _write_csv
    _archive(root)
    speed = 150.0
    _write_csv(root / 'AS350B3' / 'AS350B3_AC_Data' / '289108AC.csv',
               ['utcsec', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'VGk', 'roll', 'heading'],
               [[t, speed * (t - 105.0), 0.0, z_ft, speed, 0.0, 0.0, speed * 0.3048 / 0.514444, 0.0, 90.0]
                for t in 100.0 + np.arange(500) / 50.0])


def test_the_build_refuses_a_track_below_the_microphones_before_reading_any_audio(monkeypatch, tmp_path):
    from collections import Counter
    from netCDF4 import Dataset
    _archive_with_track(tmp_path, -40.0)
    opened = Counter()

    def counting(path, *args, **kwargs):
        opened[os.path.basename(str(path))] += 1
        return Dataset(path, *args, **kwargs)
    monkeypatch.setattr(na, 'Dataset', counting)
    records, failures = na.build_all('AS350B3', str(tmp_path / 'out'), root=str(tmp_path),
                                     prefetch=False, board_correction='flat')
    assert records == [] and len(failures) == 1
    assert failures[0]['error'].startswith('Run 289108 (card L1) refused: the aircraft goes 40 ft below')
    assert not any(name.endswith('_pascal.nc') for name in opened)
    test = na.NoiseAbatementTest('AS350B3', root=str(tmp_path))
    with pytest.raises(na.TrackRefused):
        na.build_sphere(test, '289108', str(tmp_path / 'one.nc'), board_correction='flat')


def test_the_build_can_be_told_to_skip_the_check(tmp_path):
    """A pass 5 ft over the boards is refused, and built when told to."""
    _archive_with_track(tmp_path, 5.0)
    records, failures = na.build_all('AS350B3', str(tmp_path / 'out0'), root=str(tmp_path), prefetch=False,
                                     board_correction='flat')
    assert records == [] and 'within 5.0 ft' in failures[0]['error']
    manifest = tmp_path / 'manifest.csv'
    records, failures = na.build_all('AS350B3', str(tmp_path / 'out'), root=str(tmp_path), prefetch=False,
                                     board_correction='flat', track_check=False, manifest_path=str(manifest))
    assert failures == [] and records[0]['track_flags'] == 'not checked'
    records, failures = na.build_all('AS350B3', str(tmp_path / 'out2'), root=str(tmp_path), prefetch=False,
                                     board_correction='flat', min_height_above_array_ft=None)
    assert failures == [] and records[0]['min_height_above_array_ft'] == pytest.approx(5.0)


def test_a_checked_build_records_the_lowest_height(tmp_path):
    _archive_with_track(tmp_path, 300.0)
    manifest = tmp_path / 'manifest.csv'
    records, failures = na.build_all('AS350B3', str(tmp_path / 'out'), root=str(tmp_path), prefetch=False,
                                     board_correction='flat', manifest_path=str(manifest))
    assert failures == [] and records[0]['min_height_above_array_ft'] == pytest.approx(300.0)
    with open(manifest, newline='') as handle:
        row = next(csv.DictReader(handle))
    assert row['track_flags'] == '' and float(row['min_height_above_array_ft']) == pytest.approx(300.0)


def test_the_dry_pass_reports_without_building(tmp_path, capsys):
    _archive_with_track(tmp_path, -40.0)
    out = tmp_path / 'checks.csv'
    assert na.main(['AS350B3', '--check-tracks', '--root', str(tmp_path), '--manifest', str(out)]) == 0
    printed = capsys.readouterr().out
    assert '289108 L1 refused: the aircraft goes 40 ft below' in printed
    assert 'AS350B3: 1 runs checked, 1 refused' in printed
    with open(out, newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert [(r['run'], r['status']) for r in rows] == [('289108', 'refused')]
    assert not (tmp_path / 'out').exists()


# --- the 2017 data --------------------------------------------------------------------

_DATA = local_paths.data_path(na.DATA_ROOT_NAME, required=False)
_B407_TRACKS = _DATA is not None and os.path.exists(os.path.join(_DATA, 'B407', 'B407_AC_Data', '285236AC.csv'))
_needs_b407 = pytest.mark.skipif(not _B407_TRACKS, reason='the 2017 B407 tracks are not configured')


@pytest.mark.data
@_needs_b407
def test_b407_285236_is_refused_and_its_twin_285235_is_not():
    checks = {r['run']: r for r in na.check_tracks('B407', root=_DATA, runs={'285235', '285236', '283101'})}
    assert checks['285236']['status'] == 'refused'
    assert 'below the microphones' in checks['285236']['reasons']
    assert checks['285236']['min_height_above_array_ft'] < -50.0
    assert checks['285235']['status'] in ('ok', 'flagged') and checks['285235']['reasons'] == ''
    assert checks['285235']['min_height_above_array_ft'] > 20.0
    assert checks['283101']['status'] == 'flagged'
    assert 'decelerating level-off' in checks['283101']['flags']
    assert "not the card's -3 deg" in checks['283101']['flags']


@pytest.mark.data
@_needs_b407
def test_b407_285236_is_refused_by_the_build_itself(tmp_path):
    test = na.NoiseAbatementTest('B407', root=_DATA)
    with pytest.raises(na.TrackRefused, match='Run 285236 \\(card D19\\) refused'):
        na.build_sphere(test, '285236', str(tmp_path / 'Be407236.nc'))
