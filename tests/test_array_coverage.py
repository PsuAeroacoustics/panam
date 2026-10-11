"""array_coverage, hover_array_coverage and takeoff_array_coverage on geometry
whose sphere directions are known in closed form.

Local frame: +x forward, +y left, z up. Sphere directions are UMAPR: azimuth
180 deg ahead of the nose, 90 deg starboard, 270 deg port, 0 deg behind;
elevation positive below the horizon.
"""

import numpy as np
import pytest

import flight_acoustics as fa


def test_level_pass_is_overhead_when_the_source_crosses_a_track_microphone():
    azimuth, elevation, r = fa.array_coverage([0.0, 150.0], 150.0, xmin=-100.0, xmax=100.0, speed=10.0, rate=1.0)
    overhead = 10  # x = -100 + 10 * 10 = 0
    assert elevation[overhead, 0] == pytest.approx(90.0)
    assert r[overhead, 0] == pytest.approx(150.0)
    # 150 to the left at 150 below: 45 deg below the horizon, to port.
    assert elevation[overhead, 1] == pytest.approx(45.0)
    assert azimuth[overhead, 1] == pytest.approx(270.0)


def test_a_microphone_directly_below_the_hover_point_is_at_elevation_90():
    _, elevation, r = fa.hover_array_coverage([0.0], [0.0], [30.0], [0.0])
    assert elevation.shape == (1, 1)
    assert elevation[0, 0] == pytest.approx(90.0)
    assert r[0, 0] == pytest.approx(30.0)


def test_ring_microphones_sit_at_atan_altitude_over_radius():
    altitude, radius = 50.0, 120.0
    mic_azimuths = np.array([0.0, 90.0, 180.0, 270.0])  # ahead, left, behind, right of +x
    azimuth, elevation, r = fa.hover_array_coverage(np.full(4, radius), mic_azimuths, [altitude], [0.0])
    np.testing.assert_allclose(elevation[0], np.degrees(np.arctan2(altitude, radius)))
    np.testing.assert_allclose(r[0], np.hypot(altitude, radius))
    np.testing.assert_allclose(azimuth[0], [180.0, 270.0, 0.0, 90.0], atol=1e-9)


def test_hover_heading_turns_the_azimuths_and_rows_run_altitude_then_heading():
    altitudes, headings = [20.0, 40.0], [0.0, 90.0, 180.0]
    azimuth, elevation, _ = fa.hover_array_coverage([100.0], [90.0], altitudes, headings)
    assert azimuth.shape == (len(altitudes) * len(headings), 1)
    # The mic on +y: port at heading 0, ahead at heading 90, starboard at 180.
    np.testing.assert_allclose(azimuth[:, 0], [270.0, 180.0, 90.0] * 2, atol=1e-9)
    np.testing.assert_allclose(elevation[:3, 0], np.degrees(np.arctan2(20.0, 100.0)))
    np.testing.assert_allclose(elevation[3:, 0], np.degrees(np.arctan2(40.0, 100.0)))


def test_hover_refuses_mismatched_microphone_lists():
    with pytest.raises(ValueError, match='same length'):
        fa.hover_array_coverage([10.0, 20.0], [0.0], [30.0], [0.0])


@pytest.mark.parametrize('angle', [3.0, 8.0, 15.0])
def test_a_microphone_at_the_liftoff_point_sees_the_climb_angle_behind(angle):
    distances = np.array([10.0, 100.0, 1000.0])
    azimuth, elevation, r = fa.takeoff_array_coverage([0.0], [0.0], distances, flight_path_angle=angle)
    np.testing.assert_allclose(elevation[:, 0], angle)
    np.testing.assert_allclose(azimuth[:, 0] % 360.0, 0.0, atol=1e-9)
    np.testing.assert_allclose(r[:, 0], distances / np.cos(np.radians(angle)))


def test_takeoff_origin_heading_and_initial_altitude_place_the_track():
    # Liftoff at (100, 50) heading +y (90 deg) from 20 up, climbing at 10 deg.
    # 200 along the track the source is at (100, 250, 20 + 200 tan 10).
    # A mic at radius 250 on +y is at (0, 250): 100 to the right of the source.
    height = 20.0 + 200.0 * np.tan(np.radians(10.0))
    azimuth, elevation, r = fa.takeoff_array_coverage(
        [250.0], [90.0], [200.0], initial_altitude=20.0, flight_path_angle=10.0, heading=90.0,
        source_origin_xy=(100.0, 50.0))
    # Heading +y, the mic is toward -x of the source: left of the track (port).
    assert azimuth[0, 0] == pytest.approx(270.0)
    assert elevation[0, 0] == pytest.approx(np.degrees(np.arctan2(height, 100.0)))
    assert r[0, 0] == pytest.approx(np.hypot(height, 100.0))


def test_takeoff_refuses_a_bad_origin():
    with pytest.raises(ValueError, match='source_origin_xy'):
        fa.takeoff_array_coverage([0.0], [0.0], [10.0], source_origin_xy=(0.0, 0.0, 0.0))
