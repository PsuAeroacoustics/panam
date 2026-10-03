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
