import numpy as np
import pytest
import scipy.signal as sig

from flight_acoustics import depropagate_hemisphere
from panam_acoustics.filters import third_octave_filter_bank

FS = 25000.0
CENTERS = 1000.0 * 2.0 ** (np.arange(-20, 11) / 3.0)      # 10 Hz .. 10 kHz
FRAME = 16384                                              # build_sphere's 0.5 s window at 25 kHz


def frames(duration, fs=FS, n=FRAME):
    """Frame centers (s) of a 50 %-overlap spectrogram, as depropagate_hemisphere uses."""
    return (np.arange(0, int(duration * fs) - n + 1, n // 2) + n / 2) / fs


def db(x):
    return 10.0 * np.log10(x)


@pytest.mark.parametrize('tone_hz', [19.3, 1000.0])
def test_tone_level_and_skirts_follow_the_butterworth_response(tone_hz):
    t = np.arange(0.0, 20.0, 1.0 / FS)
    x = np.sqrt(2.0) * np.sin(2 * np.pi * tone_hz * t)            # unit mean square
    centers = frames(20.0)
    keep = (centers > 3.0) & (centers < 17.0)
    P = third_octave_filter_bank(x, FS, CENTERS, centers, FRAME)[:, keep].mean(axis=1)

    own = int(np.argmin(np.abs(np.log2(CENTERS / tone_hz))))
    assert db(P[own]) == pytest.approx(0.0, abs=0.3)
    # The neighboring bands see the tone through their filter's skirt.
    for k in (own - 1, own + 1):
        fc = CENTERS[k]
        sos = sig.butter(3, [fc / 2 ** (1 / 6), fc * 2 ** (1 / 6)], btype='bandpass', fs=FS, output='sos')
        _, h = sig.sosfreqz(sos, worN=[tone_hz], fs=FS)
        assert db(P[k]) == pytest.approx(db(np.abs(h[0]) ** 2), abs=1.0)


def test_group_delay_is_compensated():
    """A 20 Hz burst centerd at 10 s must come out centerd at 10 s, not ~0.14 s later."""
    t = np.arange(0.0, 20.0, 1.0 / FS)
    burst = np.exp(-0.5 * ((t - 10.0) / 1.0) ** 2) * np.sin(2 * np.pi * 20.0 * t)
    n = 2048
    centers = (np.arange(0, t.size - n + 1, n // 4) + n / 2) / FS
    P = third_octave_filter_bank(burst, FS, [20.0], centers, n)[0]
    assert np.sum(P * centers) / np.sum(P) == pytest.approx(10.0, abs=0.05)


def test_broadband_noise_matches_the_psd_band_sum():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(int(30.0 * FS))
    f, tf, S = sig.spectrogram(x, FS, sig.get_window('hann', FRAME), noverlap=FRAME // 2, mode='psd')
    P = third_octave_filter_bank(x, FS, CENTERS, tf, FRAME)
    for i, fc in enumerate(CENTERS[CENTERS >= 100.0], start=int(np.sum(CENTERS < 100.0))):
        band = (f >= fc / 2 ** (1 / 6)) & (f < fc * 2 ** (1 / 6))
        fft_band = np.mean(S[band].sum(axis=0) * (f[1] - f[0]))
        # An order-3 Butterworth passes ~5 % more noise than a brick wall (+0.2 dB).
        assert db(np.mean(P[i]) / fft_band) == pytest.approx(0.2, abs=0.35)


def test_bands_above_nyquist_are_nan():
    x = np.random.default_rng(2).standard_normal(8000)
    P = third_octave_filter_bank(x, 2000.0, [100.0, 1000.0], [2.0], 1024)
    assert np.isfinite(P[0, 0]) and np.isnan(P[1, 0])


def _flyby(third_octave_method, **kwargs):
    rng = np.random.default_rng(0)
    p_ref = 2e-5
    mic_locations = np.array([[0.0, -50.0, 0.0], [0.0, 0.0, 0.0], [0.0, 50.0, 0.0]])
    nt = 60
    track_time = np.linspace(0.0, 8.0, nt)
    track_position = np.column_stack([300.0 * track_time / track_time[-1] - 150.0,
                                      np.zeros(nt), np.full(nt, 150.0)])
    track_velocity = np.tile([300.0 / track_time[-1], 0.0, 0.0], (nt, 1))
    fs = 4000.0
    t = np.arange(0.0, 9.0, 1.0 / fs)
    base = 1e3 * p_ref * (np.sin(2 * np.pi * 20.0 * t) + 0.3 * np.sin(2 * np.pi * 250.0 * t))
    pressure = np.vstack([base + 50 * p_ref * rng.standard_normal(t.size) for _ in range(3)])
    return depropagate_hemisphere(
        mic_locations=mic_locations, pressure=pressure, time=t,
        track_time=track_time, track_position=track_position, track_velocity=track_velocity,
        r_ref=100.0, freq_range=(0.0, 1500.0), window_time=0.5, window_overlap=0.5,
        point_stride=2, azi_step=20.0, elv_step=15.0, rmax=40.0,
        third_octave=True, third_octave_fmin=10.0, third_octave_method=third_octave_method,
        **kwargs)


def test_depropagate_hemisphere_filter_bank_option():
    fft = _flyby('fft')
    bank = _flyby('filter_bank')
    assert bank['metadata']['third_octave_method'] == 'filter_bank'
    assert fft['metadata']['third_octave_method'] == 'fft'
    centers = bank['third_octave']['band_centers_hz']
    assert np.allclose(centers, fft['third_octave']['band_centers_hz'])

    def band_mean(h, fc):
        k = int(np.argmin(np.abs(centers - fc)))
        return 10 * np.log10(np.nanmean(10 ** (h['third_octave']['bands_db'][k] / 10)))

    # Where the tone sits, and in broadband noise well away from it, they agree...
    for fc in (20.0, 250.0, 800.0):
        assert band_mean(bank, fc) == pytest.approx(band_mean(fft, fc), abs=1.0)
    # ...but the filter skirts carry the 20 Hz tone into 31.5 Hz, the FFT does not.
    assert band_mean(bank, 31.5) - band_mean(fft, 31.5) > 3.0

    # The ambient gate works per band in filter-bank mode.
    gated = _flyby('filter_bank', ambient_percentile=5.0, band_snr_gate_db=3.0)
    assert np.isfinite(gated['third_octave']['bands_db']).any()


def test_unknown_third_octave_method_is_rejected():
    with pytest.raises(ValueError, match='third_octave_method'):
        _flyby('octave')


@pytest.mark.parametrize('third_octave_method', ['fft', 'filter_bank'])
def test_depropagation_does_not_depend_on_the_length_unit(third_octave_method):
    """The same flyby in meters and in feet must give the same hemisphere.

    unit_conversion used to rewrite the emission ranges in place, so with
    length_units='m' the filter-bank pass reused ranges already converted to
    feet: +10 dB of spreading plus over-applied absorption in every band.
    """
    def run(length_units, scale):
        rng = np.random.default_rng(0)
        p_ref = 2e-5
        nt = 60
        track_time = np.linspace(0.0, 8.0, nt)
        fs = 4000.0
        t = np.arange(0.0, 9.0, 1.0 / fs)
        base = 1e3 * p_ref * (np.sin(2 * np.pi * 20.0 * t) + 0.3 * np.sin(2 * np.pi * 250.0 * t))
        return depropagate_hemisphere(
            mic_locations=scale * np.array([[0.0, -50.0, 0.0], [0.0, 0.0, 0.0], [0.0, 50.0, 0.0]]),
            pressure=np.vstack([base + 50 * p_ref * rng.standard_normal(t.size) for _ in range(3)]),
            time=t, track_time=track_time,
            track_position=scale * np.column_stack([300.0 * track_time / track_time[-1] - 150.0,
                                                    np.zeros(nt), np.full(nt, 150.0)]),
            track_velocity=np.tile([scale * 300.0 / track_time[-1], 0.0, 0.0], (nt, 1)),
            speed_of_sound=scale * 1135.0, length_units=length_units, r_ref=scale * 100.0,
            freq_range=(0.0, 1500.0), window_time=0.5, window_overlap=0.5,
            point_stride=2, azi_step=20.0, elv_step=15.0, rmax=40.0,
            third_octave=True, third_octave_fmin=10.0, third_octave_method=third_octave_method,
            apply_absorption_deprop=True)

    feet = run('ft', 1.0)
    meters = run('m', 0.3048)
    covered = feet['oaspl_db'] > -100.0
    np.testing.assert_allclose(meters['oaspl_db'][covered], feet['oaspl_db'][covered], atol=1e-6)
    for f_db, m_db in zip(feet['third_octave']['bands_db'], meters['third_octave']['bands_db']):
        np.testing.assert_allclose(m_db[covered], f_db[covered], atol=1e-6)
