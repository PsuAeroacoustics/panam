import numpy as np
import pytest
import matplotlib.pyplot as plt

from flight_acoustics import psd, psd_welch, atmosorb, art2umapr, geodetic2array, array2geodetic, lambert_ea_points
from flight_acoustics import dBAw, overall_SPL, level_history, dedopplerize

P_REF = 2.0e-5


def test_psd_sine_level():
    fs = 2048
    duration = 1.0
    t = np.arange(0, duration, 1/fs)
    # Sine with peak amplitude = p_ref so RMS = p_ref / sqrt(2)
    signal = P_REF * np.sin(2*np.pi*200*t)
    f, psd_db, level = psd(signal, fs)
    # RMS p_ref / sqrt(2): -3.0103 dB re 20 uPa, exact for a whole number of cycles
    assert level == pytest.approx(-10.0 * np.log10(2.0), abs=1e-6)
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
    assert level == pytest.approx(-10.0 * np.log10(2.0), abs=1e-6)
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


@pytest.mark.parametrize('f0', [20.5, 25.5, 50.5])
def test_overall_spl_a_weighted_level_of_an_off_bin_low_tone(f0):
    """A tone between bins: its A-weighted level is its level plus the A-weighting.

    A rectangular window leaks an off-bin tone across the spectrum with a
    1/k**2 skirt, and the leakage reaching 1-4 kHz, where the A-weighting is
    near 0 dB, read 1.8, 0.9 and 0.1 dB high at these frequencies.
    """
    fs = 25600
    t = np.arange(fs) / fs
    signal = np.sqrt(2.0) * np.sin(2 * np.pi * f0 * t)      # 1 Pa rms
    level_A, level_Z = overall_SPL(signal, fs)
    assert level_Z == pytest.approx(10.0 * np.log10(np.mean((signal - signal.mean()) ** 2) / P_REF ** 2),
                                    abs=1e-9)
    assert level_A == pytest.approx(level_Z + float(dBAw(f0)), abs=0.05)


def test_level_history_a_weighted_level_of_an_off_bin_low_tone():
    """Quarter-second periods (4 Hz bins): a rectangular window read 2.2 dB high.

    Each period holds 12.6 cycles, so its own mean square is within 0.06 dB
    of the tone's 1 Pa rms.
    """
    fs = 25600
    t = np.arange(4 * fs) / fs
    signal = np.sqrt(2.0) * np.sin(2 * np.pi * 50.5 * t)    # 1 Pa rms
    _, level_a, level_z = level_history(signal, fs, period=0.25)
    tone_level = 10.0 * np.log10(1.0 / P_REF ** 2)
    np.testing.assert_allclose(level_z, tone_level, atol=0.07)
    np.testing.assert_allclose(level_a, tone_level + float(dBAw(50.5)), atol=0.1)


def test_overall_spl_of_silence_is_minus_inf():
    level_A, level_Z = overall_SPL(np.zeros(1024), 1024)
    assert level_A == -np.inf and level_Z == -np.inf


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



def test_array2geodetic_in_meters_leaves_its_input_alone():
    """units='m' used to rewrite the caller's coordinates into feet in place,
    so converting the same array twice put the microphones 3.28x too far out."""
    reference = np.array([40.0, -77.0, 300.0])
    local = np.array([[100.0, 0.0, 0.0], [0.0, 50.0, 1.5]])
    before = local.copy()
    first = array2geodetic(local, reference, heading=30.0, units='m')
    second = array2geodetic(local, reference, heading=30.0, units='m')
    np.testing.assert_array_equal(local, before)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(geodetic2array(first, reference, 30.0, units='m'), local, atol=1e-6)


# IEC 61672-1:2013 Table 3, at the exact base-10 frequencies the standard uses.
IEC_61672_A_WEIGHTING_DB = {10: -70.4, 31.5: -39.4, 63: -26.2, 125: -16.1, 250: -8.6, 500: -3.2,
                            1000: 0.0, 2000: 1.2, 4000: 1.0, 8000: -1.1, 16000: -6.6, 20000: -9.3}


@pytest.mark.parametrize('nominal_hz,table_db', IEC_61672_A_WEIGHTING_DB.items())
def test_a_weighting_matches_iec_61672(nominal_hz, table_db):
    exact_hz = 1000.0 * 10.0 ** (np.round(10.0 * np.log10(nominal_hz / 1000.0)) / 10.0)
    assert float(dBAw(exact_hz)) == pytest.approx(table_db, abs=0.05)


def test_atmosorb_matches_iso_9613_2_table():
    """ISO 9613-2 Table 2, 20 C and 70 % RH, octave bands 63 Hz..8 kHz (dB/km)."""
    table = [0.1, 0.3, 1.1, 2.8, 5.0, 9.0, 22.9, 76.6]
    frequencies = 1000.0 * 10.0 ** (0.3 * np.arange(-4, 4))
    alpha_db_per_km = 1000.0 * atmosorb(frequencies, 20.0, 70.0, 1013.25)
    np.testing.assert_array_equal(np.round(alpha_db_per_km, 1), table)


def test_welch_single_bin_passband_retains_correct_bin_width():
    x = np.sin(2 * np.pi * 100 * np.arange(4096) / 1024)
    frequency, _, level, _ = psd_welch(x, 1024, passband=(100, 100))
    assert frequency.tolist() == [100]
    assert np.isfinite(level)


def test_welch_empty_passband_reports_input_error():
    with pytest.raises(ValueError, match='no frequency bins'):
        psd_welch(np.ones(4096), 1024, passband=(100.1, 100.2))


def test_plot_style_is_restored():
    import matplotlib
    from flight_acoustics import plot_spectrogram
    before = {key: matplotlib.rcParams[key] for key in ('axes.facecolor', 'mathtext.fontset', 'figure.autolayout')}
    fig, *_ = plot_spectrogram(np.ones(4096), 1024)
    plt.close(fig)
    assert before == {key: matplotlib.rcParams[key] for key in before}


def test_import_preserves_global_plot_defaults():
    import os
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-c', '''
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['axes.facecolor'] = 'pink'
matplotlib.rcParams['figure.autolayout'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'
import flight_acoustics
assert matplotlib.rcParams['axes.facecolor'] == 'pink'
assert matplotlib.rcParams['figure.autolayout'] is False
assert matplotlib.rcParams['mathtext.fontset'] == 'cm'
'''], capture_output=True, text=True, env=dict(os.environ, MPLBACKEND='Agg'))
    assert result.returncode == 0, result.stderr


def test_dedopplerize_leaves_unrecorded_emissions_missing():
    """np.interp held the first and last samples flat for reception times outside
    the record, a step that contaminated the de-Dopplerized spectrum."""
    c = 1000.0
    track_time = np.linspace(0.0, 2.0, 21)
    position = np.column_stack([np.zeros(21), np.zeros(21), np.full(21, 100.0)])
    observers = np.zeros((1, 3))
    t = np.arange(0.2, 2.0, 1e-3)[None, :]          # starts after the first arrival (0.1 s)
    pressure = np.sin(2 * np.pi * 50.0 * t)
    emission, dpres = dedopplerize(t, pressure, c, track_time, position, observers)
    received = emission + 0.1
    outside = (received < t[0, 0]) | (received > t[0, -1])
    assert outside.any() and np.all(np.isnan(dpres[0, outside]))
    np.testing.assert_allclose(dpres[0, ~outside], np.sin(2 * np.pi * 50.0 * received[~outside]), atol=0.02)
