"""synthetic_flyover: the synthetic pass and the depropagation's known answer."""

import os

import numpy as np
import pytest
from scipy.ndimage import binary_erosion

import flight_acoustics as fa
import synthetic_flyover as sf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINE = np.column_stack([np.zeros(7), np.linspace(-600.0, 600.0, 7), np.zeros(7)])


def _quiet_pass(directivity_db=None):
    """One microphone under the track, tones only, no noise."""
    return sf.synthesize_pass(np.zeros((1, 3)), broadband=0.0, noise=0.0, directivity_db=directivity_db)[0]


def test_directivity_scales_the_pressure_by_its_gain():
    omni = _quiet_pass()['pressure'][0]
    louder = _quiet_pass(lambda azimuth, elevation: np.full(np.shape(azimuth), 6.0))['pressure'][0]
    np.testing.assert_allclose(louder, 10.0 ** (6.0 / 20.0) * omni, rtol=1e-12, atol=0.0)


def test_directivity_is_taken_at_umapr_angles():
    """Azimuth 180 is ahead: a source loud only ahead is loud while it approaches."""
    def ahead_db(azimuth, elevation):
        return np.where(np.cos(np.radians(azimuth - 180.0)) > 0.0, 6.0, 0.0)
    scenario = _quiet_pass(ahead_db)
    ratio = scenario['pressure'][0] / _quiet_pass()['pressure'][0]
    t = scenario['time']
    arrival = 3.0 + 300.0 / 1125.0      # sound emitted overhead arrives here, s
    approaching = (t < arrival - 0.1) & np.isfinite(ratio)
    receding = (t > arrival + 0.1) & np.isfinite(ratio)
    np.testing.assert_allclose(ratio[approaching], 10.0 ** 0.3, rtol=1e-9)
    np.testing.assert_allclose(ratio[receding], 1.0, rtol=1e-9)


def test_source_level_closed_form():
    # (100 Pa ft / 100 ft)^2 (1/2 + 0.5^2/2 + 0.2^2) = 0.665 Pa^2
    expected = 10.0 * np.log10(0.665 / fa.P_REF ** 2)
    assert sf.source_level_db(180.0, 45.0, 100.0) == pytest.approx(expected, abs=1e-12)
    level = sf.source_level_db([180.0, 0.0], 0.0, 100.0, directivity_db=sf.example_directivity_db)
    np.testing.assert_allclose(level, expected + np.array([4.0, -4.0]), atol=1e-12)


def _recovered_minus(truth_directivity):
    scenario, ambient = sf.synthesize_pass(LINE, duration=12.0, directivity_db=sf.example_directivity_db, seed=1)
    hemisphere = fa.depropagate_hemisphere(
        **scenario, speed_of_sound=1125.0, r_ref=100.0, freq_range=(0.0, 2000.0), window_time=0.25,
        point_stride=2, azi_step=15.0, elv_step=10.0, rmax=20.0, ambient_pressure=ambient)
    azimuth, elevation = np.meshgrid(hemisphere['azi_grid_deg'], hemisphere['elv_grid_deg'])
    return hemisphere['oaspl_db'] - sf.source_level_db(azimuth, elevation, 100.0, directivity_db=truth_directivity)


def test_depropagation_recovers_the_known_directivity():
    difference = _recovered_minus(sf.example_directivity_db)
    measured = np.isfinite(difference)
    # Away from the edges of the measured region, where the gridding has samples on every side.
    interior = binary_erosion(measured, border_value=0)
    assert interior.sum() > 0.6 * difference.size
    assert np.abs(difference[interior]).max() < 1.0
    assert abs(np.median(difference[interior])) < 0.2
    # The check can fail: against an omnidirectional truth the same hemisphere is off by up to 4 dB.
    assert np.nanmax(np.abs(_recovered_minus(None))) > 3.0


def test_example_vehicle_cfg_reads():
    rotor = fa.read_vehicle_rotor_data(os.path.join(REPO, 'example_data', 'vehicle.cfg'))
    assert rotor['main_rotor_radius_m'] == 5.334 and rotor['main_rotor_tip_speed_mps'] == 230.7
    assert rotor['air_density_kg_m3'] == 1.070
    assert rotor['vehicle_weight_newtons'] == pytest.approx(2250 * 9.80665)


def test_plot_level_difference_returns_a_figure():
    import matplotlib.pyplot as plt
    difference = np.zeros((10, 25))
    difference[0, :3] = np.nan
    difference[1, 0] = -np.inf
    fig = sf.plot_level_difference(np.arange(0.0, 361.0, 15.0), np.arange(0.0, 91.0, 10.0), difference)
    try:
        assert fig.axes and fig.axes[-1].get_ylabel() == 'Recovered minus true, dB'
    finally:
        plt.close(fig)
