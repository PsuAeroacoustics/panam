"""The hexacopter separation script: its segment loop on short synthetic records,
de-Dopplerization on a synthetic overhead pass, and the bundled record."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import ground_plane
import hexacopter_vkf_example as hexa
from flyover_record import SOUND_SPEED, flyover_record, write_flyover_mat

FS = 2048


def _serve(monkeypatch, data):
    """Make the script's loader return ``data`` whatever file it is given."""
    monkeypatch.setattr(hexa, 'load_hexacopter_data', lambda path: dict(data))


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
    _serve(monkeypatch, dict(acoustics_T=t, acoustics_P=pressure, aircraft_time=t[::10], aircraft_rpm=rpm[::10]))


def _separate(orders, n_jobs, doppler='none', **options):
    return hexa.separate_hexacopter_acoustics('unused.mat', mic=2, orders=orders, n_jobs=n_jobs, doppler=doppler,
                                              downsample_factor=1, highpass_cutoff=100, **options)


def test_orders_may_be_a_generator(synthetic):
    """The orders were listed for a progress message before the segment loop
    iterated them, so a generator left every segment with nothing to extract."""
    from_list = _separate([2, 3], n_jobs=1)
    from_generator = _separate((order for order in (2, 3)), n_jobs=1)
    assert np.abs(from_list['P']).max() > 0.1
    np.testing.assert_array_equal(from_generator['P'], from_list['P'])


def test_serial_and_parallel_processing_agree(synthetic):
    # A bandwidth table that differs from the default reaches the workers too.
    serial = _separate([2, 3], n_jobs=1, bandwidth={2: 1.0, 3: 3.0})
    parallel = _separate([2, 3], n_jobs=2, bandwidth={2: 1.0, 3: 3.0})
    for key in ('P', 'T', 'rpm', 'original'):
        np.testing.assert_array_equal(serial[key], parallel[key])


def test_a_bandwidth_per_harmonic_overrides_the_percentage_rule(synthetic):
    rule = _separate([2, 3], n_jobs=1, bandwidth=None, min_bw=3, max_bw=3)
    np.testing.assert_allclose(_separate([2, 3], n_jobs=1, bandwidth={2: 3.0, 3: 3.0})['P'], rule['P'],
                               rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(_separate([2, 3], n_jobs=1, bandwidth=3.0)['P'], rule['P'], rtol=1e-9, atol=1e-12)
    # A harmonic the table leaves out takes the rule.
    np.testing.assert_allclose(_separate([2, 3], n_jobs=1, bandwidth={3: 3.0}, min_bw=3, max_bw=3)['P'],
                               rule['P'], rtol=1e-9, atol=1e-12)
    narrow = _separate([2, 3], n_jobs=1, bandwidth={2: 1.0, 3: 3.0})
    assert np.max(np.abs(narrow['P'] - rule['P'])) > 1e-3


def test_the_harmonic_bandwidth_rule():
    freq = np.array([[40.0, 100.0, 300.0]])
    np.testing.assert_allclose(hexa.harmonic_bandwidth(1, freq, None, 0.05, 3, 9), [[3.0, 5.0, 9.0]])
    assert hexa.harmonic_bandwidth(1, freq, {1: 5.0}, 0.05, 3, 9) == 5.0
    assert hexa.harmonic_bandwidth(2, freq, 2.5, 0.05, 3, 9) == 2.5
    assert hexa.VKF_BANDWIDTH_HZ[1] == 5.0 and set(hexa.VKF_BANDWIDTH_HZ) == set(hexa.VKF_ORDERS)


def _record(monkeypatch, n_rotors, track=None):
    """Steady rotors at 200-215 rad/s, two blades, two microphones; optional track (m)."""
    n = 4000
    t = np.arange(n) / FS
    rpm = np.tile(200.0 + 5.0 * np.arange(n_rotors), (n, 1))
    phase = np.cumsum(rpm * 2 / (2 * np.pi), axis=0) / FS
    tones = np.cos(2 * np.pi * 2 * phase).sum(axis=1)
    data = dict(acoustics_T=t, acoustics_P=np.column_stack([tones, tones]), aircraft_time=t[::10],
                aircraft_rpm=rpm[::10])
    if track is not None:
        data.update(track)
    _serve(monkeypatch, data)
    return rpm


def test_the_number_of_rotors_comes_from_the_rpm_record(monkeypatch):
    _record(monkeypatch, 4)
    result = _separate([2], n_jobs=1)
    assert result['P'].shape[1:] == (4, 1)
    assert result['rpm'].shape[1] == 4
    assert np.abs(result['P']).max() > 0.1


def test_doppler_track_filters_at_the_doppler_shifted_speeds(monkeypatch):
    # The aircraft flies straight away from the microphone at 20 m/s: every
    # rotor is heard at 1 / (1 + 20/343) of its speed.
    t = np.linspace(-1.0, 3.0, 41)
    position = np.column_stack([100.0 + 20.0 * t, np.zeros_like(t), np.full_like(t, 30.0)])
    rpm = _record(monkeypatch, 3, dict(track_time=t, track_position=position))
    result = _separate([2], n_jobs=1, doppler='track', receiver=(0.0, 0.0, 30.0), sound_speed=343.0)
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


def test_rotor_speeds_are_held_at_the_ends_of_their_record():
    """A cubic extrapolated before the speed record ran away: de-Dopplerized
    sound is emitted before the onboard log of the old record starts."""
    aircraft_time = np.linspace(1.0, 2.0, 21)
    rpm = 200.0 + 5.0 * aircraft_time
    out = hexa.interpolate_rpm_to_acoustic_time(aircraft_time, rpm, np.array([0.0, 0.5, 1.5, 2.5]))
    np.testing.assert_allclose(out[:, 0], [205.0, 205.0, 207.5, 210.0], rtol=1e-12)


def test_dedopplerize_record_restores_the_emitted_signal():
    # dedopplerize interpolates the range linearly between track samples: 0.1 s
    # apart overhead of this pass, that alone errs by 17 mm (0.06 rad at 205 Hz).
    data, emitted = flyover_record(track_step=0.01)
    time, pressure = hexa.dedopplerize_record(data, 1, sound_speed=SOUND_SPEED)
    # As recorded, the tones arrive late and shifted: nothing like the emission.
    assert np.max(np.abs(data['acoustics_P'][:, 0] - emitted(data['acoustics_T']))) > 0.5
    np.testing.assert_allclose(pressure, emitted(time), atol=0.01)
    # Only emission times whose sound reached the microphone inside the record.
    source = ground_plane.interp_columns(time, data['track_time'], data['track_position'])
    arrival = time + np.linalg.norm(data['mic_location'][0] - source, axis=1) / SOUND_SPEED
    assert arrival[0] >= data['acoustics_T'][0] - 1e-9 and arrival[-1] <= data['acoustics_T'][-1] + 1e-9


def test_dedopplerization_lets_the_filter_follow_the_emitted_speeds(monkeypatch):
    _serve(monkeypatch, flyover_record(fs=2048)[0])
    residual = {}
    for doppler in ('dedopplerize', 'track', 'none'):
        result = hexa.separate_hexacopter_acoustics('unused.mat', mic=1, orders=(2, 3), n_jobs=1, downsample_factor=1,
                                                    doppler=doppler, sound_speed=SOUND_SPEED)
        residual[doppler] = hexa.residual_levels(result, bands=((100, 300),))[0]
    assert residual['dedopplerize'] < -20 and residual['track'] < -20
    assert residual['none'] > residual['dedopplerize'] + 10


def test_a_record_without_a_track_cannot_be_de_dopplerized():
    data, _ = flyover_record()
    del data['track_position']
    with pytest.raises(ValueError, match='cannot be de-Dopplerized'):
        hexa.dedopplerize_record(data, 1)
    with pytest.raises(ValueError, match='Doppler shift cannot be predicted'):
        hexa.prepare_signal(data, 1, doppler='track')
    with pytest.raises(ValueError, match='doppler must be one of'):
        hexa.prepare_signal(data, 1, doppler='remove')


def test_the_loader_reads_the_microphone_positions_and_types(tmp_path):
    path = write_flyover_mat(tmp_path / 'pass.mat', duration=1.0)
    data = hexa.load_hexacopter_data(str(path))
    np.testing.assert_array_equal(data['mic_location'], [[0.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
    assert list(data['mic_type']) == ['ground', 'pole']


def test_the_bundled_record_is_found_beside_the_script():
    """MAT_FILE was relative to the current directory: the script ran only from the repository root."""
    import os
    assert os.path.isabs(hexa.MAT_FILE) and os.path.isfile(hexa.MAT_FILE)


def test_the_bundled_record_covers_every_emission_it_hears():
    """The rebuilt record: one position and type per channel, and a track and
    rotor-speed record that start before the first sound heard was emitted."""
    data = hexa.load_hexacopter_data(hexa.MAT_FILE)
    n_mics = data['acoustics_P'].shape[1]
    assert data['mic_location'].shape == (n_mics, 3)
    assert [k + 1 for k, kind in enumerate(data['mic_type']) if kind == 'pole'] == [13, 14, 15]
    farthest = max(np.linalg.norm(data['track_position'] - mic, axis=1).max() for mic in data['mic_location'])
    first, last = data['acoustics_T'][0] - farthest / SOUND_SPEED, data['acoustics_T'][-1]
    assert data['track_time'][0] <= first and data['aircraft_time'][0] <= first
    assert data['track_time'][-1] >= last - 0.1 and data['aircraft_time'][-1] >= last


def test_main_writes_the_figures_to_demo_plots_by_default(synthetic, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hexa, 'DOWNSAMPLE_FACTOR', 1)  # the synthetic record is at 2048 Hz already
    monkeypatch.setattr(hexa, 'VKF_ORDERS', [2, 3])     # the only harmonics it has
    assert hexa.main(['--mic', '2', '--jobs', '1', '--doppler', 'none']) == 0
    names = sorted(path.name for path in (tmp_path / 'demo_plots').iterdir())
    assert names == ['hexa_original_separated.png', 'hexa_original_separated_zoom.png',
                     'hexa_residual_spectrum.png', 'hexa_rotor_components.png', 'hexa_rpm_time_history.png']
    assert hexa.main(['--mic', '2', '--jobs', '1', '--doppler', 'none', '-o', 'elsewhere']) == 0
    assert len(list((tmp_path / 'elsewhere').glob('hexa_*.png'))) == 5
    # The default de-Dopplerizes, which this record (no track) cannot.
    assert hexa.main(['--mic', '2', '--jobs', '1']) == 1
    assert hexa.main(['--mic', '3', '--doppler', 'none']) == 1
    assert hexa.main(['--mat-file', 'missing.mat']) == 1
    plt.close('all')


def test_main_notes_a_pole_microphone(monkeypatch, tmp_path, capsys):
    data, _ = flyover_record(fs=2048, duration=2.0)
    _serve(monkeypatch, dict(data, mic_type=np.array(['ground', 'pole'])))
    monkeypatch.setattr(hexa, 'DOWNSAMPLE_FACTOR', 1)
    monkeypatch.setattr(hexa, 'VKF_ORDERS', [2, 3])
    monkeypatch.setattr(hexa, 'save_all_or_show', lambda *args, **kwargs: [])
    assert hexa.main(['--mic', '2', '--jobs', '1', '--doppler', 'track']) == 0
    assert 'microphone 2 is an elevated pole microphone' in capsys.readouterr().out
    assert hexa.main(['--mic', '1', '--jobs', '1', '--doppler', 'track']) == 0
    assert 'pole microphone' not in capsys.readouterr().out
    plt.close('all')
