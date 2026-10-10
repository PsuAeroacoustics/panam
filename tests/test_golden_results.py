"""Golden results: depropagation and a database built from it, over its main options.

Each case runs depropagate_hemisphere on the same seeded synthetic pass and
compares every gridded level with tests/data/golden_results.npz.  A refactor
must leave them unchanged; a commit that changes results on purpose
regenerates the file and says so:

    PANAM_REGENERATE_GOLDEN=1 python -m pytest tests/test_golden_results.py

The tolerance (1e-6 dB) allows for platform differences in FFTs and sums,
not for any change of method.
"""

import os

import numpy as np
import pytest

import flight_acoustics as fa
import sphere_helpers

GOLDEN = os.path.join(os.path.dirname(__file__), 'data', 'golden_results.npz')
REGENERATE = bool(os.environ.get('PANAM_REGENERATE_GOLDEN'))
P_REF = 2.0e-5


def _pass(seed=0):
    """Two tones and broadband noise from a source passing at 150 ft/s, 300 ft up,
    received at three microphones with propagation delay and spreading."""
    rng = np.random.default_rng(seed)
    c, speed, height = 1125.0, 150.0, 300.0
    mics = np.array([[0.0, -100.0, 0.0], [0.0, 0.0, 0.0], [0.0, 100.0, 0.0]])
    fs = 4000.0
    t = np.arange(0.0, 6.0, 1.0 / fs)

    def position(time):
        time = np.asarray(time, dtype=float)
        return np.column_stack([speed * (time - 3.0), np.zeros(time.size), np.full(time.size, height)])

    pressure = []
    for mic in mics:
        t_e = t.copy()
        for _ in range(30):
            t_e = t - np.linalg.norm(position(t_e) - mic, axis=1) / c
        r = np.linalg.norm(position(t_e) - mic, axis=1)
        source = (np.sin(2 * np.pi * 110.0 * t_e) + 0.5 * np.sin(2 * np.pi * 440.0 * t_e)
                  + 0.2 * rng.standard_normal(t.size))
        pressure.append(100.0 / r * source + 0.002 * rng.standard_normal(t.size))
    track_time = np.linspace(0.5, 5.5, 120)
    ambient = np.array([0.002 * rng.standard_normal(t.size) for _ in mics])
    return dict(mic_locations=mics, pressure=np.array(pressure), time=t, track_time=track_time,
                track_position=position(track_time),
                track_velocity=np.tile([speed, 0.0, 0.0], (track_time.size, 1))), ambient


def _flat_board(im, bands, offset):
    # A response that nulls the steepest arrivals, so the response cap acts.
    elevation = np.degrees(np.arctan2(offset[:, 2], np.hypot(offset[:, 0], offset[:, 1])))
    return np.tile(np.where(elevation > 80.0, -30.0, -3.0), (bands.size, 1))


ATMOSPHERE = fa.Atmosphere(temperature=288.15, pressure=101.325, relative_humidity=70.0)
CASES = {
    'fft': {},
    'filter_bank': dict(third_octave_method='filter_bank'),
    'ambient_gate': dict(ambient=True, band_snr_gate_db=10.0),
    'percentile_absorption_cap': dict(ambient_percentile=5.0, band_snr_gate_db=6.0,
                                      apply_absorption_deprop=True, atmosphere=ATMOSPHERE,
                                      max_absorption_correction_db=0.5),
    'response_cap': dict(receiver_response_db=_flat_board, max_response_correction_db=15.0),
    'tone_aware_doppler': dict(tone_aware=True, remove_doppler=True),
    'narrowband': dict(narrowband=True, narrowband_stride=8),
    'adaptive_aspect': dict(interpolation=dict(aspect=5.0, shepard_floor=1.0)),
}


def _results():
    scenario, ambient = _pass()
    out = {}
    for name, options in CASES.items():
        options = dict(options)
        if options.pop('ambient', False):
            options['ambient_pressure'] = ambient
        hemisphere = fa.depropagate_hemisphere(
            **scenario, speed_of_sound=1125.0, length_units='ft', r_ref=100.0, freq_range=(0.0, 1500.0),
            window_time=0.25, window_overlap=0.5, point_stride=2, azi_step=30.0, elv_step=15.0,
            rmax=30.0, third_octave=True, third_octave_fmin=80.0, **options)
        out[name + '.oaspl_db'] = hemisphere['oaspl_db']
        out[name + '.spl_a_db'] = hemisphere['spl_a_db']
        out[name + '.bands_db'] = hemisphere['third_octave']['bands_db']
        if 'narrowband' in hemisphere:
            out[name + '.psd_db'] = hemisphere['narrowband']['psd_db']
        if name == 'fft':
            out['database.dBA'], out['database.EAA'] = _database(hemisphere)
    return out


def _database(hemisphere):
    import tempfile
    import pathlib
    from netCDF4 import Dataset

    with tempfile.TemporaryDirectory() as scratch:
        spheres = sphere_helpers.write_sphere_directory(
            pathlib.Path(scratch) / 'spheres', [(hemisphere, 60.0, 0.0), (hemisphere, 80.0, -6.0)])
        path = os.path.join(scratch, 'db.nod')
        fa.build_empirical_database(str(spheres), path, load_factors=None)
        with Dataset(path) as database:
            group = database.groups['sphere0']
            return np.array(group.variables['dBA'][:]), np.array(group.variables['EAA'][:])


@pytest.fixture(scope='module')
def results():
    return _results()


def test_results_match_the_golden_file(results):
    if REGENERATE:
        np.savez_compressed(GOLDEN, **results)
        pytest.skip('regenerated ' + GOLDEN)
    golden = np.load(GOLDEN)
    assert sorted(golden.files) == sorted(results)
    for key in golden.files:
        expected, actual = golden[key], results[key]
        assert actual.shape == expected.shape, key
        # Missing (NaN) and empty (-inf) cells must stay where they were.
        assert np.array_equal(np.isnan(actual), np.isnan(expected)), key
        assert np.array_equal(np.isneginf(actual), np.isneginf(expected)), key
        finite = np.isfinite(expected)
        np.testing.assert_allclose(actual[finite], expected[finite], rtol=0.0, atol=1e-6, err_msg=key)


def test_every_case_exercises_its_option(results):
    """Guard the golden file against cases that quietly test nothing."""
    fft = results['fft.bands_db']
    assert np.isnan(results['percentile_absorption_cap.bands_db']).sum() > np.isnan(fft).sum()  # caps leave gaps
    for case in ('filter_bank', 'ambient_gate', 'response_cap', 'tone_aware_doppler', 'adaptive_aspect'):
        assert not np.array_equal(results[case + '.bands_db'], fft, equal_nan=True), case
    assert 'narrowband.psd_db' in results
