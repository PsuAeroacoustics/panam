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
    modeled = _run(scenario, ambient_pressure=ambient, receiver_response_db=response,
                    third_octave_method=method)
    assert {im for im, _ in calls} == {0, 1, 2} and all(shape[1] == 3 for _, shape in calls)
    np.testing.assert_allclose(modeled['oaspl_db'], flat['oaspl_db'], atol=1e-9)
    np.testing.assert_allclose(modeled['third_octave']['bands_db'], flat['third_octave']['bands_db'],
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
    modeled = _run(scenario, receiver_response_db=response)
    bands = np.asarray(plain['third_octave']['band_centers_hz'])
    grids_plain = plain['third_octave']['bands_db']
    grids_model = modeled['third_octave']['bands_db']
    for i, fc in enumerate(bands):
        a, b = np.asarray(grids_plain[i]), np.asarray(grids_model[i])
        finite = np.isfinite(a) & np.isfinite(b) & (a > -200.0)   # not the empty-cell floor
        expected = -10.0 if i == np.argmin(np.abs(bands - 200.0)) else 0.0
        np.testing.assert_allclose(b[finite] - a[finite], expected, atol=1e-9)


def test_a_response_beyond_the_cap_drops_the_band_instead_of_dividing_it_up():
    """max_response_correction_db: a band whose response is a deep null (-30 dB here) is
    dropped, not raised 30 dB; a band within the cap is still divided out."""
    scenario = _scenario(seed=7)

    def response(im, bands, offset):
        gain = np.zeros((bands.size, offset.shape[0]))
        gain[np.argmin(np.abs(bands - 200.0))] = -30.0
        gain[np.argmin(np.abs(bands - 400.0))] = -10.0
        return gain

    plain = _run(scenario)
    capped = _run(scenario, receiver_response_db=response, max_response_correction_db=15.0)
    uncapped = _run(scenario, receiver_response_db=response)
    bands = np.asarray(plain['third_octave']['band_centers_hz'])
    i200, i400 = np.argmin(np.abs(bands - 200.0)), np.argmin(np.abs(bands - 400.0))
    a, b, c = (np.asarray(x['third_octave']['bands_db'][i200]) for x in (plain, capped, uncapped))
    finite = np.isfinite(a) & (a > -200.0)
    np.testing.assert_allclose(c[finite] - a[finite], 30.0, atol=1e-9)
    assert not np.any(np.isfinite(b) & (b > -200.0))
    a, b = (np.asarray(x['third_octave']['bands_db'][i400]) for x in (plain, capped))
    finite = np.isfinite(a) & np.isfinite(b) & (a > -200.0)
    np.testing.assert_allclose(b[finite] - a[finite], 10.0, atol=1e-9)


def test_receiver_response_follows_the_band_sums_edges_for_nominal_centers():
    """Each bin is divided by the response of the band it is summed into.

    Nominal centers take their IEC base-10 edges in the band sums, so the 800
    Hz band ends at 891.25 Hz, not at 800 * 2**(1/6) = 897.97 Hz.  The 894.53
    Hz bin (fs 4000 Hz, 1024-point frames) lies between the two: summed into
    the 1000 Hz band, it has to be divided by that band's response.
    """
    rng = np.random.default_rng(14)
    fs = 4000.0
    t = np.arange(0.0, 3.5, 1.0 / fs)
    scenario = dict(_scenario(noise_only=True), time=t,
                    pressure=np.vstack([0.05 * P_REF * rng.standard_normal(t.size) for _ in range(3)]))
    centers = np.array([630.0, 800.0, 1000.0])
    gains_db = {630.0: 0.0, 800.0: 6.0, 1000.0: 12.0}

    def response(im, bands, offset):
        return np.repeat([[gains_db[fc]] for fc in bands], offset.shape[0], axis=1)

    options = dict(freq_range=(0.0, 1200.0), window_time=0.25, third_octave_fmin=630.0,
                   third_octave_band_centers_hz=centers)
    plain = _run(scenario, **options)
    modeled = _run(scenario, receiver_response_db=response, **options)
    np.testing.assert_array_equal(plain['third_octave']['band_centers_hz'], centers)
    for fc, a, b in zip(centers, plain['third_octave']['bands_db'], modeled['third_octave']['bands_db']):
        finite = np.isfinite(a)
        assert finite.any()
        np.testing.assert_allclose(b[finite] - a[finite], -gains_db[fc], atol=1e-9)


def test_a_capped_sample_is_a_gap_not_silence():
    """A band dropped by a cap was not measured; averaged in as zero power it pulled
    every node it shared with measured samples down (-3 dB at half the weight)."""
    scenario = _scenario(noise_only=True, seed=12)

    def flat(im, bands, offset):
        return np.zeros((bands.size, offset.shape[0]))

    def null_on_mic_0(im, bands, offset):
        return np.full((bands.size, offset.shape[0]), -30.0 if im == 0 else 0.0)

    kept = _run(scenario, receiver_response_db=flat, return_scattered=True)
    dropped = _run(scenario, receiver_response_db=null_on_mic_0, max_response_correction_db=15.0,
                   return_scattered=True)

    # Microphone 0's samples are missing, not silent ...
    mic0 = dropped['scattered']['mic'] == 0
    assert np.all(np.isnan(dropped['scattered']['oaspl_power'][mic0]))
    assert np.all(np.isnan(dropped['scattered']['third_octave']['bands_db'][:, mic0]))
    # ... so the nodes the other microphones reach keep their level.
    for key in ('oaspl_db', 'spl_a_db'):
        both = np.isfinite(kept[key]) & np.isfinite(dropped[key])
        assert both.sum() > 0.5 * np.isfinite(kept[key]).sum()
        assert abs(float(np.median(dropped[key][both] - kept[key][both]))) < 0.5
    bands_kept, bands_dropped = kept['third_octave']['bands_db'], dropped['third_octave']['bands_db']
    both = np.isfinite(bands_kept) & np.isfinite(bands_dropped)
    assert abs(float(np.median(bands_dropped[both] - bands_kept[both]))) < 0.5


def test_absorption_cap_leaves_capped_bands_missing():
    scenario = _scenario(seed=5)
    atmosphere = Atmosphere(temperature=300.0, pressure=90.0, relative_humidity=60.0)
    capped = _run(scenario, apply_absorption_deprop=True, atmosphere=atmosphere,
                  max_absorption_correction_db=0.05, return_scattered=True)
    bands = capped['scattered']['third_octave']['bands_db']
    # The highest band is beyond the cap for the farther samples: those are NaN,
    # never -inf, and the overall level of every sample is still finite.
    assert np.isnan(bands[-1]).any() and not np.isneginf(bands[-1]).any()
    assert np.all(np.isfinite(capped['scattered']['oaspl_power']))


def test_a_doppler_scaled_band_past_the_selected_range_is_missing():
    """remove_doppler reads each band over [D f_lower, D f_upper]; past the selected
    bins the band used to be integrated over part of its width without a word."""
    import flight_acoustics as fa
    f = np.arange(0.0, 2000.0 + 1e-9, 2.0)
    lower, upper = fa.third_octave_band_edges(np.array([1000.0, 1600.0]))
    dropped = np.zeros((f.size, 3), dtype=bool)
    dropped[500, 2] = True          # a capped bin (1000 Hz) inside the 1 kHz band
    missing = fa._bands_missing(dropped, f, lower, upper, np.array([1.0, 1.18, 1.0]))
    assert missing.tolist() == [[False, False, True], [False, True, False]]
