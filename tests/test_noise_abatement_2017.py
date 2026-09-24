"""Dataset-independent pieces of the 2017 Noise Abatement sphere rebuild."""

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
