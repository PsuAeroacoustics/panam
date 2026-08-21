"""Tests for the d' (d-prime) aural detection model.

Implements Sickenberger, Gopalan & Schmitz (AHS Forum 67, 2011), Eqs. 2-5.
Anchor values are chosen so failures localise the broken step: ERB and
threshold tables, ro-ex filtering, the EASN calibration, the composite
root-sum-square, and the detection distance search.
"""

import numpy as np
import pytest

import flight_acoustics as fa


def third_octave_centers(fmin=25.0, fmax=4000.0):
    k = np.arange(-50, 50)
    centers = 1000.0 * (2.0 ** (k / 3.0))
    return centers[np.logical_and(centers >= fmin, centers <= fmax)]


# ---------------------------------------------------------------- ERB, ro-ex

def test_erb_glasberg_moore_anchors():
    assert fa.erb_bandwidth(1000.0) == pytest.approx(24.7 * 5.37)
    assert fa.erb_bandwidth(0.0) == pytest.approx(24.7)
    # Paper context: ERB at 80 Hz is wider than the third-octave bandwidth
    assert fa.erb_bandwidth(80.0) == pytest.approx(33.33, rel=1e-3)


def test_roex_weight_shape():
    fc = 500.0
    assert fa.roex_filter_weight(fc, fc) == pytest.approx(1.0)
    # symmetric in |f - fc| and monotonically decreasing away from fc
    assert fa.roex_filter_weight(400.0, fc) == pytest.approx(
        fa.roex_filter_weight(600.0, fc))
    w = fa.roex_filter_weight(np.array([500.0, 550.0, 700.0, 1200.0]), fc)
    assert np.all(np.diff(w) < 0.0)


def test_roex_integral_equals_erb():
    """The ro-ex filter's equivalent rectangular bandwidth is ERB(fc)."""
    fc = 250.0
    f = np.linspace(1.0e-3, 20.0 * fc, 400001)
    integral = np.trapezoid(fa.roex_filter_weight(f, fc), f)
    assert integral == pytest.approx(fa.erb_bandwidth(fc), rel=1e-4)


def test_critical_band_level_of_white_noise():
    """Flat PSD in -> critical band level = spectrum level + 10 log10(ERB)."""
    centers = third_octave_centers(20.0, 20000.0)
    bandwidths = centers * (2.0 ** (1.0 / 6.0) - 2.0 ** (-1.0 / 6.0))
    spectrum_level = 40.0  # dB re 20 uPa per Hz
    band_levels = spectrum_level + 10.0 * np.log10(bandwidths)
    analysis, critical = fa.critical_band_levels(
        centers, band_levels, analysis_centers=np.array([250.0]))
    expected = spectrum_level + 10.0 * np.log10(fa.erb_bandwidth(250.0))
    assert critical[0] == pytest.approx(expected, abs=0.1)


# ------------------------------------------------------------------- dprime

def test_dprime_threshold_in_quiet():
    """A band at the hearing threshold in quiet gives d' near d'_T = 1.5.

    Exactly 1.5 holds for a pure tone at the filter center; representing the
    tone as a uniform-PSD third-octave band spreads a little energy outside
    the critical band, so d' falls slightly below 1.5.
    """
    centers = third_octave_centers()
    idx = int(np.argmin(np.abs(centers - 100.0)))
    signal = np.full(centers.size, -200.0)
    signal[idx] = fa.hearing_threshold_spl(centers[idx])
    ambient = np.full(centers.size, -200.0)
    result = fa.dprime(centers, signal, ambient)
    band_idx = int(np.argmin(np.abs(result['band_frequency_hz'] - centers[idx])))
    assert 1.2 < result['dprime_band'][band_idx] <= 1.5


def test_dprime_linear_in_signal_power():
    centers = third_octave_centers()
    ambient = fa.dprime_artificial_ambient(centers, 'low')
    quiet = fa.dprime(centers, ambient + 5.0, ambient)
    loud = fa.dprime(centers, ambient + 15.0, ambient)
    assert loud['dprime_composite'] == pytest.approx(
        10.0 * quiet['dprime_composite'], rel=1e-9)


def test_dprime_composite_is_root_sum_square():
    centers = third_octave_centers()
    ambient = fa.dprime_artificial_ambient(centers, 'moderate')
    result = fa.dprime(centers, ambient + 10.0, ambient)
    assert result['dprime_composite'] == pytest.approx(
        np.sqrt(np.sum(result['dprime_band'] ** 2)))


def test_dprime_default_efficiency_is_inverse_erb():
    """detector_efficiency=1/ERB must reproduce the default exactly."""
    centers = third_octave_centers()
    ambient = fa.dprime_artificial_ambient(centers, 'low')
    signal = ambient + 10.0
    default = fa.dprime(centers, signal, ambient)
    eta = 1.0 / fa.erb_bandwidth(default['band_frequency_hz'])
    explicit = fa.dprime(centers, signal, ambient, detector_efficiency=eta)
    assert explicit['dprime_band'] == pytest.approx(default['dprime_band'])


def test_dprime_analysis_range():
    centers = third_octave_centers(20.0, 20000.0)
    ambient = fa.dprime_artificial_ambient(centers, 'low')
    result = fa.dprime(centers, ambient + 10.0, ambient)
    assert result['band_frequency_hz'][0] >= 45.0
    assert result['band_frequency_hz'][-1] <= 2240.0
    # 50 Hz .. 2 kHz nominal third-octave centers -> 17 bands
    assert result['band_frequency_hz'].size == 17


def test_dprime_vectorized_over_leading_dims():
    centers = third_octave_centers()
    ambient = fa.dprime_artificial_ambient(centers, 'low')
    spectra = np.stack([ambient + 5.0, ambient + 15.0])
    result = fa.dprime(centers, spectra, ambient)
    assert result['dprime_band'].shape == (2, 17)
    assert result['dprime_composite'].shape == (2,)
    single = fa.dprime(centers, ambient + 5.0, ambient)
    assert result['dprime_composite'][0] == pytest.approx(
        single['dprime_composite'])


def test_inm_detector_efficiency_reproduces_inm_band_dprime():
    """eta_INM * sqrt(BW) at 1 kHz: 10log(eta) = -3.56 dB, BW = 231.6 Hz."""
    eff = fa.inm_detector_efficiency(np.array([1000.0]))
    eta_bw = eff * fa.erb_bandwidth(1000.0)   # eta(i)*ERB(i) weight in Eq. 2
    expected = 10.0 ** (-3.56 / 10.0) * np.sqrt(1000.0 * (2 ** (1 / 6) - 2 ** (-1 / 6)))
    assert eta_bw[0] == pytest.approx(expected, rel=1e-6)
    # INM weight is stronger than the default (eta*ERB = 1), so INM-scale
    # criteria (d' ~= 5) are consistently reachable
    assert eta_bw[0] > 1.0


# -------------------------------------------------------- artificial ambient

def test_artificial_ambient_anchors():
    f = np.array([25.0, 1600.0, 5000.0])
    low = fa.dprime_artificial_ambient(f, 'low')
    assert low == pytest.approx([33.0, 20.0, 20.0])
    f = np.array([25.0, 100.0, 400.0, 2000.0])
    moderate = fa.dprime_artificial_ambient(f, 'moderate')
    assert moderate == pytest.approx([60.0, 60.0, 40.0, 40.0])
    with pytest.raises(ValueError):
        fa.dprime_artificial_ambient(f, 'extreme')


# ---------------------------------------------------------- detection range

def helicopter_like_spectrum(centers):
    """Low-frequency-dominated source spectrum, dB at the reference distance."""
    return 80.0 - 12.0 * np.log10(centers / 50.0)


def test_detection_distance_basic():
    centers = third_octave_centers()
    levels = helicopter_like_spectrum(centers)
    ambient = fa.dprime_artificial_ambient(centers, 'low')
    result = fa.dprime_detection_distance(centers, levels, ambient, 150.0)
    assert result['status'] == 'interpolated'
    d = result['detection_distance_m']
    assert 150.0 < d < 1.0e5
    # the composite d' brackets the critical value at the crossing
    idx = int(np.searchsorted(result['distances_m'], d))
    assert result['dprime_composite'][idx - 1] >= 1.0
    assert result['dprime_composite'][idx] < 1.0
    assert result['trigger_frequency_hz'] in result['band_frequency_hz']


def test_detection_distance_orderings():
    centers = third_octave_centers()
    levels = helicopter_like_spectrum(centers)
    low = fa.dprime_artificial_ambient(centers, 'low')
    moderate = fa.dprime_artificial_ambient(centers, 'moderate')
    base = fa.dprime_detection_distance(centers, levels, low, 150.0)
    louder = fa.dprime_detection_distance(centers, levels + 10.0, low, 150.0)
    masked = fa.dprime_detection_distance(centers, levels, moderate, 150.0)
    stricter = fa.dprime_detection_distance(centers, levels, low, 150.0,
                                            critical_dprime=2.33)
    assert louder['detection_distance_m'] > base['detection_distance_m']
    assert masked['detection_distance_m'] < base['detection_distance_m']
    assert stricter['detection_distance_m'] < base['detection_distance_m']


def test_detection_distance_inaudible():
    centers = third_octave_centers()
    levels = np.full(centers.size, -30.0)
    ambient = fa.dprime_artificial_ambient(centers, 'moderate')
    result = fa.dprime_detection_distance(centers, levels, ambient, 150.0)
    assert result['status'] == 'below_critical_everywhere'
    assert np.isnan(result['detection_distance_m'])
    assert np.isnan(result['trigger_frequency_hz'])


def test_detection_distance_with_ega():
    centers = third_octave_centers()
    levels = helicopter_like_spectrum(centers)
    ambient = fa.dprime_artificial_ambient(centers, 'low')
    free_field = fa.dprime_detection_distance(centers, levels, ambient, 150.0)
    over_ground = fa.dprime_detection_distance(
        centers, levels, ambient, 150.0,
        source_height_m=45.0, receiver_height_m=1.2, flow_resistance=225.0)
    assert np.isfinite(over_ground['detection_distance_m'])
    assert (over_ground['detection_distance_m']
            != pytest.approx(free_field['detection_distance_m']))
