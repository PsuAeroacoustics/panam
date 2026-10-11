"""The hexacopter separation script's segment loop, on a short synthetic record."""

import numpy as np
import pytest

import hexacopter_vkf_example as hexa

FS = 2048


@pytest.fixture
def synthetic(monkeypatch):
    """Six rotors near 200 rad/s with two blades each, two microphones."""
    n = 4000
    t = np.arange(n) / FS
    rpm = 200.0 + 5.0 * np.sin(2 * np.pi * 0.5 * t[:, None] + np.arange(6))
    phase = np.cumsum(rpm * 2 / (2 * np.pi), axis=0) / FS
    rng = np.random.default_rng(0)
    tones = sum(np.cos(2 * np.pi * order * phase).sum(axis=1) / order for order in (2, 3))
    pressure = np.column_stack([tones + 0.05 * rng.standard_normal(n) for _ in range(2)])
    data = dict(acoustics_T=t, acoustics_P=pressure, aircraft_time=t[::10], aircraft_rpm=rpm[::10], fs=FS)
    monkeypatch.setattr(hexa, 'load_hexacopter_data', lambda path, downsample_factor=8: dict(data))


def _separate(orders, n_jobs):
    return hexa.separate_hexacopter_acoustics('unused.mat', mic_range=2, orders=orders, n_jobs=n_jobs,
                                              highpass_cutoff=100)


def test_orders_may_be_a_generator(synthetic):
    """The orders were listed for a progress message before the segment loop
    iterated them, so a generator left every segment with nothing to extract."""
    from_list = _separate([2, 3], n_jobs=1)
    from_generator = _separate((order for order in (2, 3)), n_jobs=1)
    assert np.abs(from_list['P']).max() > 0.1
    np.testing.assert_array_equal(from_generator['P'], from_list['P'])


def test_serial_and_parallel_processing_agree(synthetic):
    serial = _separate([2, 3], n_jobs=1)
    parallel = _separate([2, 3], n_jobs=2)
    for key in ('P', 'T', 'rpm', 'original'):
        np.testing.assert_array_equal(serial[key], parallel[key])


def _record(monkeypatch, n_rotors, track=None):
    """Steady rotors at 200-215 rad/s, two blades, two microphones; optional track (m)."""
    n = 4000
    t = np.arange(n) / FS
    rpm = np.tile(200.0 + 5.0 * np.arange(n_rotors), (n, 1))
    phase = np.cumsum(rpm * 2 / (2 * np.pi), axis=0) / FS
    tones = np.cos(2 * np.pi * 2 * phase).sum(axis=1)
    data = dict(acoustics_T=t, acoustics_P=np.column_stack([tones, tones]), aircraft_time=t[::10],
                aircraft_rpm=rpm[::10], fs=FS)
    if track is not None:
        data.update(track)
    monkeypatch.setattr(hexa, 'load_hexacopter_data', lambda path, downsample_factor=8: dict(data))
    return rpm


def test_the_number_of_rotors_comes_from_the_rpm_record(monkeypatch):
    _record(monkeypatch, 4)
    result = _separate([2], n_jobs=1)
    assert result['P'].shape[1:] == (4, 1)
    assert result['rpm'].shape[1] == 4
    assert np.abs(result['P']).max() > 0.1


def test_a_receiver_filters_at_the_doppler_shifted_speeds(monkeypatch):
    # The aircraft flies straight away from the microphone at 20 m/s: every
    # rotor is heard at 1 / (1 + 20/343) of its speed.
    t = np.linspace(-1.0, 3.0, 41)
    position = np.column_stack([100.0 + 20.0 * t, np.zeros_like(t), np.full_like(t, 30.0)])
    rpm = _record(monkeypatch, 3, dict(track_time=t, track_position=position))
    receiver = (0.0, 0.0, 30.0)
    result = hexa.separate_hexacopter_acoustics('unused.mat', mic_range=2, orders=[2], n_jobs=1,
                                                highpass_cutoff=100, receiver=receiver, sound_speed=343.0)
    np.testing.assert_allclose(result['rpm'], rpm[:len(result['rpm'])] / (1 + 20.0 / 343.0), rtol=1e-9)
    default = _separate([2], n_jobs=1)
    np.testing.assert_allclose(default['rpm'], rpm[:len(default['rpm'])], rtol=1e-9)


def test_a_one_dimensional_rpm_record_is_one_rotor():
    from scipy.interpolate import interp1d
    aircraft_time = np.linspace(0.0, 2.0, 21)
    rpm = 200.0 + 10.0 * np.sin(2 * np.pi * 0.7 * aircraft_time)
    acoustics_T = np.linspace(0.05, 1.95, 97)
    out = hexa.interpolate_rpm_to_acoustic_time(aircraft_time, rpm, acoustics_T)
    assert out.shape == (len(acoustics_T), 1)
    np.testing.assert_allclose(out[:, 0], interp1d(aircraft_time, rpm, kind='cubic')(acoustics_T))


def test_the_bundled_record_is_found_beside_the_script():
    """MAT_FILE was relative to the current directory: the script ran only from the repository root."""
    import os
    assert os.path.isabs(hexa.MAT_FILE) and os.path.isfile(hexa.MAT_FILE)


def test_main_writes_the_figures_to_demo_plots_by_default(synthetic, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert hexa.main(['--mic-range', '2', '--jobs', '1']) == 0
    names = sorted(path.name for path in (tmp_path / 'demo_plots').iterdir())
    assert names == ['hexa_original_separated.png', 'hexa_original_separated_zoom.png',
                     'hexa_rotor_components.png', 'hexa_rpm_time_history.png']
    assert hexa.main(['--mic-range', '2', '--jobs', '1', '-o', 'elsewhere']) == 0
    assert len(list((tmp_path / 'elsewhere').glob('hexa_*.png'))) == 4
    assert hexa.main(['--mat-file', 'missing.mat']) == 1
