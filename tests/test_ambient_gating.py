"""Ambient gating and absorption limiting in depropagate_hemisphere.

These cover the failure that produced 216 dB at 10 kHz in the shipped Be407
source spheres: an ambient-limited band, amplified by spreading and by an
absorption correction of hundreds of dB.
"""

import numpy as np
import pytest

from flight_acoustics import Atmosphere, depropagate_hemisphere


P_REF = 2.0e-5


def _scenario(noise_only=False, seed=0, nsamples=None, duty=None):
    """A short pass over three microphones; tone pair plus broadband noise.

    ``duty`` restricts the tones to that central fraction of the record, the
    way a real flyover leaves quiet lead-in and run-out; without it the source
    is on for every frame.
    """
    rng = np.random.default_rng(seed)
    mic_locations = np.array([
        [0.0, -50.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 50.0, 0.0],
    ])
    nt = 60
    track_time = np.linspace(0.0, 3.0, nt)
    track_position = np.column_stack([
        200.0 * (track_time / track_time[-1]) - 100.0,
        np.zeros(nt),
        300.0 * np.ones(nt),
    ])
    track_velocity = np.tile([200.0 / track_time[-1], 0.0, 0.0], (nt, 1))

    fs = 2000.0
    t = np.arange(0.0, 3.5, 1.0 / fs) if nsamples is None else np.arange(nsamples) / fs
    noise = lambda: 0.05 * P_REF * rng.standard_normal(t.size)
    if noise_only:
        pressure = np.vstack([noise() for _ in range(3)])
    else:
        tone = (0.5 * P_REF * np.sin(2 * np.pi * 200.0 * t)
                + 0.3 * P_REF * np.sin(2 * np.pi * 500.0 * t))
        if duty is not None:
            middle = np.abs(t - 0.5 * (t[0] + t[-1])) <= 0.5 * float(duty) * (t[-1] - t[0])
            tone = tone * middle
        pressure = np.vstack([tone + noise() for _ in range(3)])
    return dict(mic_locations=mic_locations, pressure=pressure, time=t,
                track_time=track_time, track_position=track_position,
                track_velocity=track_velocity)


def _run(scenario, **kwargs):
    defaults = dict(speed_of_sound=1135.0, length_units='ft', r_ref=100.0,
                    freq_range=(0.0, 800.0), window_time=0.1, window_overlap=0.5,
                    point_stride=3, azi_step=20.0, elv_step=15.0, rmax=35.0,
                    third_octave=True, third_octave_fmin=100.0)
    defaults.update(kwargs)
    return depropagate_hemisphere(**scenario, **defaults)


def _peak_level(hemisphere):
    levels = hemisphere['oaspl_db']
    finite = levels[np.isfinite(levels)]
    assert finite.size > 0
    return float(finite.max())


def test_ambient_gate_suppresses_a_noise_only_run():
    """A run that is nothing but ambient must not produce a source level.

    The suppression has to grow with the threshold, and a 3 dB gate is
    demonstrably too weak to rely on: a spectrogram bin of pure noise clears
    3 dB often enough that the gate only takes a few dB off. This is why the
    2017 spheres are rebuilt at 10 dB rather than the documented default.
    """
    scenario = _scenario(noise_only=True, seed=1)
    ambient = _scenario(noise_only=True, seed=2)['pressure']

    ungated = _peak_level(_run(scenario))
    suppression = {}
    for gate in (3.0, 6.0, 10.0):
        gated = _run(scenario, ambient_pressure=ambient, band_snr_gate_db=gate)
        suppression[gate] = ungated - _peak_level(gated)

    assert suppression[3.0] < suppression[6.0] < suppression[10.0]
    assert suppression[3.0] < 10.0
    assert suppression[10.0] > 20.0


def test_ambient_gate_preserves_signal_above_the_floor():
    """Tones well above ambient survive gating."""
    scenario = _scenario(seed=3)
    ambient = _scenario(noise_only=True, seed=4)['pressure']

    ungated = _run(scenario)
    gated = _run(scenario, ambient_pressure=ambient, band_snr_gate_db=3.0)

    mask = np.isfinite(ungated['oaspl_db']) & np.isfinite(gated['oaspl_db'])
    assert mask.sum() > 0
    # Subtracting the ambient power lowers the level slightly but must not gut it.
    assert np.allclose(gated['oaspl_db'][mask], ungated['oaspl_db'][mask], atol=3.0)


def test_ambient_gate_does_not_touch_a_loud_source():
    """A strict gate must leave a signal that is well above ambient alone."""
    scenario = _scenario(seed=3)
    ambient = _scenario(noise_only=True, seed=4)['pressure']

    ungated = _run(scenario)
    gated = _run(scenario, ambient_pressure=ambient, band_snr_gate_db=10.0)
    mask = np.isfinite(ungated['oaspl_db']) & np.isfinite(gated['oaspl_db'])
    assert np.allclose(gated['oaspl_db'][mask], ungated['oaspl_db'][mask], atol=1.0)


def test_max_absorption_correction_caps_high_frequency_amplification():
    """The absorption cap must bound what depropagation can invent."""
    scenario = _scenario(seed=5)
    atmosphere = Atmosphere(temperature=300.0, pressure=90.0, relative_humidity=60.0)

    uncapped = _run(scenario, apply_absorption_deprop=True, atmosphere=atmosphere)
    capped = _run(scenario, apply_absorption_deprop=True, atmosphere=atmosphere,
                  max_absorption_correction_db=5.0)

    bands_uncapped = uncapped['third_octave']['bands_db']
    bands_capped = capped['third_octave']['bands_db']
    finite = np.isfinite(bands_uncapped) & np.isfinite(bands_capped)
    assert finite.sum() > 0
    assert np.nanmax(bands_capped[finite]) <= np.nanmax(bands_uncapped[finite]) + 1e-9
    # The cap only ever removes energy, never adds it.
    assert np.all(bands_capped[finite] <= bands_uncapped[finite] + 1e-6)


def test_ambient_sources_are_mutually_exclusive():
    scenario = _scenario(seed=6)
    with pytest.raises(ValueError, match='not both'):
        _run(scenario, ambient_pressure=scenario['pressure'], ambient_time_range=(0.0, 0.5))


def test_mixed_sample_rates_allowed_without_narrowband():
    """The 2017 test mixes 25000 and 25600 Hz channels across one array."""
    scenario = _scenario(seed=7)
    fs_a, fs_b = 2000.0, 2048.0
    n = scenario['pressure'].shape[1]
    times = [np.arange(n) / fs_a, np.arange(n) / fs_b, np.arange(n) / fs_a]
    scenario['time'] = times

    hemi = _run(scenario)
    assert np.isfinite(hemi['oaspl_db']).any()

    with pytest.raises(ValueError, match='same sample rate'):
        _run(scenario, narrowband=True)


def test_ambient_sample_rate_must_match_the_run():
    scenario = _scenario(seed=8)
    ambient = _scenario(noise_only=True, seed=9)['pressure']
    n = ambient.shape[1]
    with pytest.raises(ValueError, match='same channel and sample rate'):
        _run(scenario, ambient_pressure=ambient,
             ambient_time=[np.arange(n) / 1000.0] * 3)


def test_ambient_percentile_tracks_a_measured_ambient_recording():
    """Fallback for array layouts with no ambient recording of their own.

    The claim is that it lands close to what a measured ambient gives, not
    that it suppresses everything: it estimates the floor from the quietest
    frames of the run, so it needs the run to contain signal. On a record that
    is pure ambient a low percentile samples the low tail of the noise
    fluctuation and deliberately gates very little.
    """
    scenario = _scenario(seed=10, duty=0.4)
    ambient = _scenario(noise_only=True, seed=11)['pressure']

    measured = _run(scenario, ambient_pressure=ambient, band_snr_gate_db=10.0)
    estimated = _run(scenario, ambient_percentile=5.0, band_snr_gate_db=10.0)

    # Compare only where both kept something: a cell one method gates away
    # entirely and the other keeps a trace of differs by thousands of dB and
    # says nothing about how well the floor was estimated.
    a, b = measured['oaspl_db'], estimated['oaspl_db']
    mask = np.isfinite(a) & np.isfinite(b) & (a > -100.0) & (b > -100.0)
    assert mask.sum() > 0.5 * a.size
    assert abs(float(np.median(b[mask] - a[mask]))) < 1.0


def test_only_one_ambient_source_is_accepted():
    scenario = _scenario(seed=11)
    with pytest.raises(ValueError, match='only one ambient source'):
        _run(scenario, ambient_percentile=5.0, ambient_time_range=(0.0, 0.5))
    with pytest.raises(ValueError, match='strictly between 0 and 100'):
        _run(scenario, ambient_percentile=0.0)


def test_max_range_drops_distant_emission_points():
    """Straight-ray depropagation is only a fair model over a limited path."""
    scenario = _scenario(seed=12)
    # Ranges in this geometry run 300-320 ft, so the limit has to bite inside that.
    near = _run(scenario, max_range=310.0, return_scattered=True)
    far = _run(scenario, return_scattered=True)

    assert near['scattered']['azi_deg'].size < far['scattered']['azi_deg'].size
    assert near['scattered']['azi_deg'].size > 0


def test_min_elevation_drops_grazing_incidence():
    """Near the horizon the microphone hears ground impedance, not the source."""
    scenario = _scenario(seed=13)
    # Elevations here run 70-90 deg; a 30 deg floor would not bite.
    gated = _run(scenario, min_elevation_deg=78.0, return_scattered=True)
    ungated = _run(scenario, return_scattered=True)

    assert gated['scattered']['elv_deg'].min() >= 78.0
    assert gated['scattered']['elv_deg'].size < ungated['scattered']['elv_deg'].size


@pytest.mark.parametrize('method', ['fft', 'filter_bank'])
def test_constant_receiver_response_matches_the_flat_scale(method):
    """A constant +6.02 dB receiver response is the old 0.5 pressure scale.

    The response is divided out after the ambient gate, so the ambient has to
    be scaled the same way as the run in both forms for them to agree.
    """
    scenario = _scenario(seed=4)
    ambient = _scenario(noise_only=True, seed=5)['pressure']
    doubling_db = 20.0 * np.log10(2.0)
    calls = []

    def response(im, bands, offset):
        calls.append((im, offset.shape))
        return np.full((bands.size, offset.shape[0]), doubling_db)

    flat = _run(dict(scenario, pressure=0.5 * scenario['pressure']), ambient_pressure=0.5 * ambient,
                third_octave_method=method)
    modelled = _run(scenario, ambient_pressure=ambient, receiver_response_db=response,
                    third_octave_method=method)
    assert {im for im, _ in calls} == {0, 1, 2} and all(shape[1] == 3 for _, shape in calls)
    np.testing.assert_allclose(modelled['oaspl_db'], flat['oaspl_db'], atol=1e-9)
    np.testing.assert_allclose(modelled['third_octave']['bands_db'], flat['third_octave']['bands_db'],
                               atol=1e-9)


def test_receiver_response_is_applied_per_band():
    """Each band is divided by its own response, and the offset is source minus mic."""
    scenario = _scenario(seed=6)

    def response(im, bands, offset):
        # 10 dB in the band holding 200 Hz, 0 elsewhere; check the geometry.
        assert np.all(offset[:, 2] == 300.0)
        gain = np.zeros((bands.size, offset.shape[0]))
        gain[np.argmin(np.abs(bands - 200.0))] = 10.0
        return gain

    plain = _run(scenario)
    modelled = _run(scenario, receiver_response_db=response)
    bands = np.asarray(plain['third_octave']['band_centers_hz'])
    grids_plain = plain['third_octave']['bands_db']
    grids_model = modelled['third_octave']['bands_db']
    for i, fc in enumerate(bands):
        a, b = np.asarray(grids_plain[i]), np.asarray(grids_model[i])
        finite = np.isfinite(a) & np.isfinite(b) & (a > -200.0)   # not the empty-cell floor
        expected = -10.0 if i == np.argmin(np.abs(bands - 200.0)) else 0.0
        np.testing.assert_allclose(b[finite] - a[finite], expected, atol=1e-9)
