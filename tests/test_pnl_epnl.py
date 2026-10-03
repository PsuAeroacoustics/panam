"""Tests for the PNL / PNLT / EPNL implementation (14 CFR 36 Appendix A36.4).

Anchor values verified against the CFR text (Table A36-3 constants, Table
A36-2 tone factors) and chosen so failures localize the broken step.
"""

import numpy as np
import pytest

import flight_acoustics as fa

FLAT = 70.0 * np.ones(24)
K1000 = 13          # index of the 1000 Hz band
K500 = 10           # index of the 500 Hz band


def test_noy_anchors():
    spl = np.zeros(24)
    spl[K1000] = 40.0
    assert fa.noys(spl)[K1000] == pytest.approx(1.0)      # 1 noy at 40 dB
    spl[K1000] = 70.0
    assert fa.noys(spl)[K1000] == pytest.approx(8.0, rel=1e-3)  # x2 per 10 dB
    # 50 Hz: SPL(b)=64 -> 1 noy; SPL(d)=49 -> 0.1 noy; below -> 0
    spl = np.zeros(24)
    spl[0] = 64.0
    assert fa.noys(spl)[0] == pytest.approx(1.0)
    spl[0] = 49.0
    assert fa.noys(spl)[0] == pytest.approx(0.1)
    spl[0] = 48.9
    assert fa.noys(spl)[0] == 0.0
    # continuity at the SPL(a) discontinuity coordinate (50 Hz: 91.0 dB)
    spl[0] = 91.0
    above = fa.noys(spl)[0]
    spl[0] = 90.999
    assert above == pytest.approx(fa.noys(spl)[0], rel=1e-3)


def test_pnl_single_band_tracks_level():
    """A lone 1 kHz band above 40 dB: N = n so PNL = SPL."""
    spl = np.zeros(24)
    for level in (40.0, 50.0, 60.0, 70.0):
        spl[K1000] = level
        assert fa.perceived_noise_level(spl) == pytest.approx(level, abs=0.01)


def test_pnl_silence_is_minus_inf():
    assert fa.perceived_noise_level(np.zeros(24)) == -np.inf


def test_tone_correction_flat_spectrum_is_zero():
    c, _ = fa.tone_correction(FLAT)
    assert c == 0.0


def test_tone_correction_single_tone():
    """+10 dB tone on a flat background: F=10 -> C=F/3 in the middle range."""
    spl = FLAT.copy()
    spl[K1000] += 10.0
    c, band = fa.tone_correction(spl)
    assert c == pytest.approx(10.0 / 3.0, abs=1e-6)
    assert band == K1000
    pnlt, pnl, c_max, _ = fa.tone_corrected_perceived_noise_level(spl)
    assert pnlt - pnl == pytest.approx(10.0 / 3.0, abs=1e-6)


def test_tone_correction_500hz_uses_middle_range():
    """Table A36-2 puts 500 Hz in the 500<=f<=5000 range: C=F/3, not F/6."""
    spl = FLAT.copy()
    spl[K500] += 10.0
    c, band = fa.tone_correction(spl)
    assert band == K500
    assert c == pytest.approx(10.0 / 3.0, abs=1e-6)


def test_tone_correction_low_band_range():
    """A 250 Hz tone (f<500) gets the F/6 factor."""
    spl = FLAT.copy()
    spl[7] += 10.0                                   # 250 Hz
    c, band = fa.tone_correction(spl)
    assert band == 7
    assert c == pytest.approx(10.0 / 6.0, abs=1e-6)


def test_epnl_constant_history():
    """Constant PNLT for 20 s: EPNL = PNLTM + 10 log10(duration/10)."""
    hist = np.tile(FLAT, (40, 1))
    res = fa.effective_perceived_noise_level(hist, dt=0.5)
    assert res["duration_correction_db"] == pytest.approx(
        10.0 * np.log10(40 * 0.5 / 10.0), abs=1e-6)
    assert res["epnl"] == pytest.approx(res["pnltm"]
                                        + res["duration_correction_db"])
    assert res["clipped"]                            # window spans the record


def test_epnl_contiguous_duration_window():
    """Dips below PNLTM-10 inside t1..t2 are included in the integral."""
    levels = np.array([40.0, 80.0, 68.0, 80.0, 40.0])
    hist = np.stack([np.full(24, lv) for lv in levels])
    res = fa.effective_perceived_noise_level(hist, dt=0.5)
    pnlt = res["pnlt"]
    assert res["k1"] == 1 and res["k2"] == 3
    # the k=2 dip is more than 10 dB down yet still integrated
    assert pnlt[2] < res["pnltm"] - 10.0
    expected = 10.0 * np.log10(
        np.sum(10.0 ** (pnlt[1:4] / 10.0)) * 0.5 / 10.0) - res["pnltm"]
    assert res["duration_correction_db"] == pytest.approx(expected)


def test_epnl_bandshare_adjustment():
    """A36.4.4.2: suppressed C at the PNLT peak raises PNLTM to PNL+avg(C)."""
    hist = np.tile(FLAT, (7, 1))
    hist[:, K1000] += 10.0                           # tone in every frame...
    hist[3, K1000] -= 10.0                           # ...except the peak one
    hist[3, :] += 4.0                                # peak PNL frame, no tone
    res = fa.effective_perceived_noise_level(hist, bandshare_adjustment=True)
    res0 = fa.effective_perceived_noise_level(hist, bandshare_adjustment=False)
    assert np.argmax(res["pnlt"]) == 3
    assert res["c_max"][3] == pytest.approx(0.0, abs=1e-9)
    assert res["pnltm"] > res0["pnltm"]
    lo_hi_avg = np.mean(res["c_max"][1:6])
    assert res["pnltm"] == pytest.approx(res["pnl"][3] + lo_hi_avg)


def test_nice_levels():
    ap = fa                      # helpers live in flight_acoustics itself

    lv = ap.nice_levels(66.0, 73.8)                 # ~8 dB span -> 1 dB steps
    assert lv[0] == pytest.approx(66.0) and lv[-1] == pytest.approx(74.0)
    assert np.allclose(np.diff(lv), 1.0)
    # levels must bracket the data and sit on multiples of the step
    for lo, hi in [(61.4, 79.8), (40.0, 95.0), (84.2, 90.1), (0.3, 0.4)]:
        lv = ap.nice_levels(lo, hi)
        step = lv[1] - lv[0]
        assert lv[0] <= lo and lv[-1] >= hi
        assert np.allclose(np.remainder(lv + step / 2, step) - step / 2, 0,
                           atol=1e-9)
    # forced step
    assert np.allclose(ap.nice_levels(61.4, 79.8, step=5.0),
                       [60, 65, 70, 75, 80])
    # degenerate input must not explode
    assert len(ap.nice_levels(70.0, 70.0)) >= 2
    assert len(ap.colorbar_ticks(np.arange(40), max_ticks=11)) <= 11


def test_vectorized_matches_single():
    rng = np.random.default_rng(7)
    hist = 60.0 + 10.0 * rng.random((5, 24))
    pnlt_v, pnl_v, c_v, _ = fa.tone_corrected_perceived_noise_level(hist)
    for k in range(5):
        pnlt_1, pnl_1, c_1, _ = fa.tone_corrected_perceived_noise_level(hist[k])
        assert pnlt_v[k] == pytest.approx(pnlt_1)
        assert pnl_v[k] == pytest.approx(pnl_1)
        assert c_v[k] == pytest.approx(c_1)
