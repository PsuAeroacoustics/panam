"""panam.uas_midstate reads the Mid-State UAS run files: pressure from counts and steps, microphones by id or
channel, the time axes, and the track in m or ft. The run file is written here from the format's description, so
the test needs no dataset."""
import netCDF4
import numpy as np
import pytest

from panam import uas_midstate as um


def write_run(path, fs=1000.0, n=50):
    rng = np.random.default_rng(1)
    counts = rng.integers(-2000, 2000, size=(2, n), dtype=np.int32)
    step = np.array([2.5e-5, 3.0e-5])
    with netCDF4.Dataset(path, 'w') as nc:
        nc.run_id = '2023-06-08_TEST-1'
        nc.time_origin = '2023-06-08T14:00:00+00:00'
        a = nc.createGroup('acoustic')
        a.createDimension('mic', 2)
        a.createDimension('sample', n)
        a.sample_rate = fs
        for name, vals in (('mic', ['mic16', 'mic13']), ('channel', ['M8', 'M13']),
                           ('mic_mount', ['ground board', 'pole'])):
            v = a.createVariable(name, str, ('mic',))
            for i, s in enumerate(vals):
                v[i] = s
        for name, vals in (('mic_east', [12.0, 12.0]), ('mic_north', [-28.0, -28.0]), ('mic_up', [-0.8, 0.6]),
                           ('mic_x', [30.0, 30.0]), ('mic_y', [1.0, 1.0]), ('mic_z', [-0.8, 0.6])):
            a.createVariable(name, 'f8', ('mic',))[:] = vals
        a.createVariable('step', 'f8', ('mic',))[:] = step
        a.createVariable('counts', 'i4', ('mic', 'sample'))[:] = counts
        v = nc.createGroup('vehicle')
        p = v.createGroup('position')
        p.createDimension('time', 3)
        p.createVariable('time', 'f8', ('time',))[:] = [0.0, 0.5, 1.0]
        for name, vals in (('east', [0.0, 3.048, 6.096]), ('north', [0.0, 0.0, 0.0]), ('up', [10.0, 10.0, 10.0]),
                           ('x', [0.0, -1.0, -2.0]), ('y', [0.0, 2.0, 4.0]), ('z', [10.0, 10.0, 10.0])):
            p.createVariable(name, 'f8', ('time',))[:] = vals
    return counts, step


def test_pressure_is_counts_times_step(tmp_path):
    path = tmp_path / 'run.nc'
    counts, step = write_run(path)
    with um.open_run(path) as run:
        p = run.pressure()
        assert p.shape == counts.shape
        np.testing.assert_array_equal(p, counts * step[:, None])
        np.testing.assert_array_equal(run.pressure(['M13'], 5, 10)[0], counts[1, 5:10] * step[1])
        np.testing.assert_array_equal(run.pressure(['mic16'])[0], counts[0] * step[0])


def test_microphones_and_receiver(tmp_path):
    path = tmp_path / 'run.nc'
    write_run(path)
    with um.open_run(path) as run:
        mics = run.microphones()
        assert list(mics['channel']) == ['M8', 'M13']
        assert list(mics['mic_mount']) == ['ground board', 'pole']
        assert run.mic_index('M13') == run.mic_index('mic13') == 1
        np.testing.assert_allclose(run.receiver('M8'), [12.0, -28.0, -0.8])
        np.testing.assert_allclose(run.receiver('M8', units='ft'), np.array([12.0, -28.0, -0.8]) / 0.3048)
        with pytest.raises(KeyError):
            run.mic_index('M99')


def test_time_samples_and_track(tmp_path):
    path = tmp_path / 'run.nc'
    write_run(path, fs=1000.0, n=50)
    with um.open_run(path) as run:
        assert run.sample_rate == 1000.0
        np.testing.assert_allclose(run.time(0, 3), [0.0, 0.001, 0.002])
        assert run.samples(0.0105, 0.020) == (10, 20)
        tr = run.track('ft')
        np.testing.assert_allclose(tr['x'], [0.0, 10.0, 20.0])
        assert run.rotors() == {} and run.weather() == {} and run.air_data() == {} and run.segments() == {}
        np.testing.assert_allclose(run.track(frame='array')['y'], [0.0, 2.0, 4.0])
        np.testing.assert_allclose(run.receiver('M13', frame='array'), [30.0, 1.0, 0.6])
        assert run.attrs['run_id'] == '2023-06-08_TEST-1'
        assert '/vehicle/position' in run.groups
        assert run.time_origin == np.datetime64('2023-06-08T14:00:00', 'ns')
        assert run.utc(1.5) == np.datetime64('2023-06-08T14:00:01.5', 'ns')


def test_list_runs(tmp_path):
    d = tmp_path / 'runs' / '2023-06-08'
    d.mkdir(parents=True)
    write_run(d / 'B.nc')
    write_run(d / 'A.nc')
    (d / '._A.nc').write_bytes(b'')
    assert [p.split('/')[-1] for p in um.list_runs(tmp_path)] == ['A.nc', 'B.nc']


def test_segments(tmp_path):
    path = tmp_path / 'run.nc'
    write_run(path)
    with netCDF4.Dataset(path, 'a') as nc:
        g = nc.createGroup('segments')
        g.createDimension('segment', 2)
        g.createDimension('mic', 2)
        k = g.createVariable('kind', str, ('segment',))
        k[0], k[1] = 'pass', 'hover'
        g.createVariable('start', 'f8', ('segment',))[:] = [0.010, 0.020]
        g.createVariable('end', 'f8', ('segment',))[:] = [0.020, 0.040]
        g.createVariable('closest_distance', 'f8', ('segment', 'mic'))[:] = [[2.0, 3.0], [5.0, 6.0]]
    with um.open_run(path) as run:
        s = run.segments()
        assert list(s['kind']) == ['pass', 'hover']
        assert s['closest_distance'].shape == (2, 2)
        assert run.segment_samples(1) == (20, 40)
