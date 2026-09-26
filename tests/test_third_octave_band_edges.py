"""One-third octave FFT bands must tile the frequency axis.

The NORAH2 export and the 2017 batch hand depropagate_hemisphere *nominal*
band centres (12.5, 1250, 1600 Hz, ...).  Taking each band's edges as
fc * 2**(+-1/6) straight from a rounded centre leaves gaps and overlaps of up
to 8 % of a band between neighbours, so the brick-wall band sum dropped
whatever fell in a gap (a 1410 Hz tone reached no band at all) and counted
whatever fell in an overlap twice (895 Hz landed in both 800 and 1000 Hz).
"""

import numpy as np
import pytest

import flight_acoustics as fa


def test_nominal_centres_tile_without_gaps_or_overlaps():
    lower, upper = fa.third_octave_band_edges(fa.NORAH2_BAND_CENTERS_HZ)
    np.testing.assert_allclose(upper[:-1], lower[1:], rtol=1e-12, atol=0.0)


def test_exact_centres_keep_their_edges_bit_for_bit():
    centres = 1000.0 * 2.0 ** (np.arange(-20, 11) / 3.0)
    lower, upper = fa.third_octave_band_edges(centres)
    np.testing.assert_array_equal(lower, centres / 2.0 ** (1.0 / 6.0))
    np.testing.assert_array_equal(upper, centres * 2.0 ** (1.0 / 6.0))


def test_nominal_centres_get_the_iec_base10_edges():
    """IEC 61260-1: nominal band k is the exact base-10 band 1000 * 10**(k/10)."""
    lower, upper = fa.third_octave_band_edges(fa.NORAH2_BAND_CENTERS_HZ)
    exact = 1000.0 * 10.0 ** (np.arange(-20, 11) / 10.0)
    np.testing.assert_allclose(lower, exact / 10.0 ** 0.05, rtol=1e-12)
    np.testing.assert_allclose(upper, exact * 10.0 ** 0.05, rtol=1e-12)


def test_a_set_mixing_exact_and_nominal_centres_still_tiles():
    """Edges from the base-2 and base-10 grids do not meet, so in a mixed set the
    exact base-2 centres take their band's base-10 edges too."""
    mixed = np.array([1000.0 * 2.0 ** (1.0 / 3.0), 1600.0, 2000.0])   # base-2, nominal, both
    lower, upper = fa.third_octave_band_edges(mixed)
    np.testing.assert_allclose(upper[:-1], lower[1:], rtol=1e-12, atol=0.0)
    exact = 1000.0 * 10.0 ** (np.arange(1, 4) / 10.0)
    np.testing.assert_allclose(lower, exact / 10.0 ** 0.05, rtol=1e-12)


@pytest.mark.parametrize('tone_hz,band_hz', [(1410.0, 1250.0), (895.0, 1000.0), (14.2, 16.0)])
def test_every_frequency_falls_in_exactly_one_band(tone_hz, band_hz):
    lower, upper = fa.third_octave_band_edges(fa.NORAH2_BAND_CENTERS_HZ)
    inside = (lower <= tone_hz) & (tone_hz < upper)
    assert fa.NORAH2_BAND_CENTERS_HZ[inside].tolist() == [band_hz]


def test_depropagated_tone_lands_in_its_nominal_band():
    """End to end: a 1410 Hz tone must carry the 1250 Hz band, not vanish."""
    p_ref = 2e-5
    fs = 8000.0
    t = np.arange(0.0, 9.0, 1.0 / fs)
    rng = np.random.default_rng(0)
    tone = 1e3 * p_ref * np.sin(2 * np.pi * 1410.0 * t)
    pressure = np.vstack([tone + p_ref * rng.standard_normal(t.size) for _ in range(3)])
    nt = 60
    track_time = np.linspace(0.0, 8.0, nt)
    hemi = fa.depropagate_hemisphere(
        mic_locations=np.array([[0.0, -50.0, 0.0], [0.0, 0.0, 0.0], [0.0, 50.0, 0.0]]),
        pressure=pressure, time=t, track_time=track_time,
        track_position=np.column_stack([300.0 * track_time / track_time[-1] - 150.0,
                                        np.zeros(nt), np.full(nt, 150.0)]),
        track_velocity=np.tile([300.0 / track_time[-1], 0.0, 0.0], (nt, 1)),
        r_ref=100.0, freq_range=(0.0, 3000.0), window_time=0.5, window_overlap=0.5,
        point_stride=2, azi_step=20.0, elv_step=15.0, rmax=40.0,
        third_octave=True, third_octave_fmin=10.0,
        third_octave_band_centers_hz=fa.NORAH2_BAND_CENTERS_HZ)
    centres = hemi['third_octave']['band_centers_hz']
    bands = hemi['third_octave']['bands_db']
    k = int(np.argmin(np.abs(centres - 1250.0)))
    covered = np.isfinite(hemi['oaspl_db']) & (hemi['oaspl_db'] > -100.0)
    # The tone dominates, so the 1250 Hz band carries essentially all of OASPL.
    assert np.all(np.abs(bands[k][covered] - hemi['oaspl_db'][covered]) < 0.5)
