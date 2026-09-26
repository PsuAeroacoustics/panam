import numpy as np
import pytest
import matplotlib.pyplot as plt

from flight_acoustics import psd, psd_welch, atmosorb, art2umapr, geodetic2array, array2geodetic, lambert_ea_points
from flight_acoustics import dBAw, overall_SPL, level_history

P_REF = 2.0e-5


def test_psd_sine_level():
    fs = 2048
    duration = 1.0
    t = np.arange(0, duration, 1/fs)
    # Sine with peak amplitude = p_ref so RMS = p_ref / sqrt(2)
    signal = P_REF * np.sin(2*np.pi*200*t)
    f, psd_db, level = psd(signal, fs)
    # Expected level ~ -3.01 dB re 20 uPa
    assert -4.5 < level < -1.5, f"Level outside expected range: {level}"  # loose tolerance for windowing
    assert f[0] >= 0
    assert psd_db.shape == f.shape


def test_psd_calibration_offset():
    rng = np.random.default_rng(123)
    fs = 2000
    n = 4096
    signal = rng.normal(0.0, 1.0e-3, size=n)  # non-zero random signal in Pa
    f0, psd_db0, level0 = psd(signal, fs, cal=0.0)
    f1, psd_db1, level1 = psd(signal, fs, cal=10.0)  # +10 dB calibration
    # Frequency bins should be identical
    assert np.array_equal(f0, f1)
    assert psd_db0.shape == psd_db1.shape
    # Level should increase by approximately the calibration amount
    assert np.isfinite(level0)
    assert np.isfinite(level1)
    assert abs((level1 - level0) - 10.0) < 0.05


def test_psd_zero_signal():
    fs = 1024
    n = 2048
    signal = np.zeros(n)
    f, psd_db, level = psd(signal, fs)
    # PSD should be -inf across all frequencies and level -inf
    assert np.all(np.isneginf(psd_db))
    assert np.isneginf(level)


def test_psd_welch_sine_level():
    fs = 4096
    duration = 2.0
    t = np.arange(0, duration, 1/fs)
    signal = P_REF * np.sin(2*np.pi*500*t)
    f, psd_db, level, level_A = psd_welch(signal, fs, window_time=0.5)
    assert -4.5 < level < -1.5
    # At 500 Hz A-weighting is negative; ensure it is lower than unweighted
    assert level_A <= level


@pytest.mark.parametrize('f0', [100.0, 1000.0, 2000.0, 4000.0])
def test_psd_welch_a_weighted_level_of_a_tone(f0):
    """A tone's A-weighted level is its level plus the A-weighting at its frequency.

    psd_welch once weighted the PSD by 10**(dBA/2) instead of 10**(dBA/10):
    right at 1 kHz, where the weighting is 0 dB, but 77 dB low at 100 Hz and
    5 dB high at 2 kHz -- which ``level_A <= level`` above cannot see.
    """
    fs = 48000
    t = np.arange(10 * fs) / fs
    signal = np.sqrt(2.0) * np.sin(2 * np.pi * f0 * t)      # 1 Pa rms
    _, _, level, level_A = psd_welch(signal, fs)
    assert level_A == pytest.approx(level + float(dBAw(f0)), abs=0.05)
    assert level_A == pytest.approx(overall_SPL(signal, fs)[0], abs=0.05)


@pytest.mark.parametrize('width_hz', [10.0, 40.0])
def test_psd_welch_medfilter_width_is_in_hz(width_hz):
    """medfilter is a width in Hz, as documented, not a number of bins."""
    import scipy.signal
    fs = 8192
    signal = np.random.default_rng(7).normal(0.0, 1.0, 20 * fs)
    f, raw_db, _, _ = psd_welch(signal, fs, window_time=0.25)         # df = 4 Hz
    df = f[1] - f[0]
    _, filtered_db, _, _ = psd_welch(signal, fs, window_time=0.25, medfilter=width_hz)
    bins = 2 * int(round(0.5 * width_hz / df)) + 1
    expected = 10.0 * np.log10(scipy.signal.medfilt(10.0 ** (raw_db / 10.0), bins))
    assert df == 4.0
    np.testing.assert_allclose(filtered_db, expected, atol=1e-9)


def test_level_history_keeps_the_last_full_period():
    """A 10 s record in 1 s periods has 10 levels; the last one was dropped."""
    fs = 1000
    signal = np.random.default_rng(0).standard_normal(10 * fs)
    time, level_a, level_z = level_history(signal, fs, period=1.0)
    np.testing.assert_array_equal(time, np.arange(10.0))
    assert np.all(np.isfinite(level_a)) and np.all(np.isfinite(level_z))


def test_atmosorb_shapes_monotonic():
    freqs = np.array([100.0, 1000.0, 5000.0])
    alpha = atmosorb(freqs, 20.0, 50.0, 1013.25)  # 20 C, 50% RH, 1 atm
    assert alpha.shape == freqs.shape
    assert np.all(alpha >= 0)
    # Expect higher absorption at higher frequencies
    assert alpha[-1] >= alpha[0]


def test_atmosorb_broadcast():
    freqs = np.array([500.0, 1000.0])
    temps = np.array([0.0, 10.0, 20.0])
    pressures = 1013.25 * np.ones_like(temps)
    humid = 60.0
    alpha = atmosorb(freqs, temps, humid, pressures)
    assert alpha.shape == temps.shape + freqs.shape


def test_art2umapr_basic():
    # Simple angles: phi=0, theta=0 should map to some azimuth/elevation
    phi = np.array([0.0])
    theta = np.array([0.0])
    azi, elv = art2umapr(phi, theta)
    assert azi.shape == phi.shape
    assert elv.shape == phi.shape
    # Elevation should be near 0 for theta=0
    assert abs(elv[0]) < 1e-12


def test_umapr_starboard_is_azimuth_90():
    """hemigen in a right-handed, z-up frame: ahead 180, starboard 90, port 270, behind 0."""
    from flight_acoustics import hemigen
    # Flying east (+x) at 100 ft; starboard of an eastbound aircraft is south (-y).
    observers = np.array([[1000.0, 0.0, 0.0], [0.0, -1000.0, 0.0], [0.0, 1000.0, 0.0], [-1000.0, 0.0, 0.0]])
    azimuth, _, _, _, _ = hemigen(np.array([0.0]), np.array([[0.0, 0.0, 100.0]]),
                                  np.array([[100.0, 0.0, 0.0]]), observers, 1100.0)
    assert np.allclose(azimuth[0], [180.0, 90.0, 270.0, 0.0])


def test_art2umapr_follows_the_aam_manual():
    """AAM v3 sec. 2.4.1: theta 0 at the nose, phi 0 below and positive to starboard."""
    from flight_acoustics import _umapr2art
    phi = np.radians([90.0, -90.0, 0.0, 0.0, 0.0])
    theta = np.radians([90.0, 90.0, 90.0, 0.0, 180.0])
    azi, elv = art2umapr(phi, theta)
    assert np.allclose(np.degrees(azi[[0, 1, 3, 4]]), [90.0, 270.0, 180.0, 0.0], atol=1e-9)
    assert np.allclose(np.degrees(elv), [0.0, 0.0, 90.0, 0.0, 0.0], atol=1e-9)

    P, T = np.meshgrid(np.radians(np.arange(-80.0, 81.0, 20.0)), np.radians(np.arange(10.0, 171.0, 20.0)))
    back_phi, back_theta = _umapr2art(*art2umapr(P, T))
    assert np.allclose(back_phi, P, atol=1e-9) and np.allclose(back_theta, T, atol=1e-9)


def test_lambert_ea_points_default_umapr_grid():
    fig, ax, _ = lambert_ea_points(
        np.array([], dtype=float),
        np.array([], dtype=float),
        markers=np.array([], dtype=object),
    )
    try:
        meridian_zero = next(line for line in ax.lines if line.get_gid() == 'lambert-grid-umapr-meridian-0')
        xdata = np.asarray(meridian_zero.get_xdata(), dtype=float)
        assert np.max(np.abs(xdata)) < 1.0e-12
        assert all('lambert-grid-art-' not in (line.get_gid() or '') for line in ax.lines)
    finally:
        plt.close(fig)


def test_lambert_ea_points_art_grid_overlay():
    fig, ax, _ = lambert_ea_points(
        np.array([], dtype=float),
        np.array([], dtype=float),
        markers=np.array([], dtype=object),
        grid_convention='rnm',
    )
    try:
        theta_ninety = next(line for line in ax.lines if line.get_gid() == 'lambert-grid-art-theta-90')
        ydata = np.asarray(theta_ninety.get_ydata(), dtype=float)
        assert np.max(np.abs(ydata)) < 1.0e-12
        labels = {text.get_text() for text in ax.texts}
        assert '90°' in labels
        assert '0°' in labels
    finally:
        plt.close(fig)


def test_geodetic_local_roundtrip():
    ref = np.array([40.0, -77.0, 300.0])  # lat, lon, height(m)
    local = np.array([[0.0, 0.0, 0.0],
                      [10.0, 0.0, 0.0],
                      [0.0, 10.0, 5.0]])  # ft inputs
    heading = 45.0
    geod = array2geodetic(local, ref, heading, units='ft')
    local_rt = geodetic2array(geod, ref, heading, units='ft')
    # Roundtrip within a small tolerance
    assert np.allclose(local, local_rt, atol=1e-2), f"Roundtrip mismatch: {local_rt - local}"
