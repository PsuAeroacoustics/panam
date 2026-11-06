"""panam.uas_midstate reads the Mid-State UAS run files: pressure from counts and steps, microphones by id or
channel, the time axes, and the track in m or ft. The run file is written here from the format's description, so
the test needs no dataset."""
import warnings

import netCDF4
import numpy as np
import pytest

from panam import uas_midstate as um


ORIGIN = '2023-06-08T14:00:00+00:00'


def add_series(parent, name, t, fields, attrs=None, time_units=f'seconds since {ORIGIN}'):
    """A group with its own time axis, as the run files hold them; fields: {name: values (floats or strings)}."""
    g = parent.createGroup(name)
    g.createDimension('time', len(t))
    v = g.createVariable('time', 'f8', ('time',))
    v[:] = t
    v.units = time_units
    for key, vals in fields.items():
        if isinstance(vals[0], str):
            var = g.createVariable(key, str, ('time',))
            for i, x in enumerate(vals):
                var[i] = x
        else:
            g.createVariable(key, 'f8', ('time',))[:] = vals
    for k, val in (attrs or {}).items():
        setattr(g, k, val)
    return g


def write_run(path, fs=1000.0, n=50, version='0.4', vehicle=True):
    rng = np.random.default_rng(1)
    counts = rng.integers(-2000, 2000, size=(2, n), dtype=np.int32)
    step = np.array([2.5e-5, 3.0e-5])
    with netCDF4.Dataset(path, 'w') as nc:
        nc.run_id = '2023-06-08_TEST-1'
        nc.status = 'flight' if vehicle else 'ambient'
        if version is not None:
            nc.format_version = version
        nc.time_origin = ORIGIN
        a = nc.createGroup('acoustic')
        a.createDimension('mic', 2)
        a.createDimension('sample', n)
        a.sample_rate = fs
        for name, vals in (('mic_id', ['mic16', 'mic13']), ('channel', ['M8', 'M13']),
                           ('mic_mount', ['inverted ground plane', 'pole (4 ft, Part 36)'])):
            v = a.createVariable(name, str, ('mic',))
            for i, s in enumerate(vals):
                v[i] = s
        for name, vals in (('mic_east', [10.0, 10.0]), ('mic_north', [-20.0, -20.0]), ('mic_up', [-0.8, 0.6]),
                           ('mic_x', [30.0, 30.0]), ('mic_y', [1.0, 1.0]), ('mic_z', [-0.8, 0.6])):
            a.createVariable(name, 'f8', ('mic',))[:] = vals
        a.createVariable('step', 'f8', ('mic',))[:] = step
        a.createVariable('counts', 'i4', ('mic', 'sample'))[:] = counts
        q = a.createVariable('mic_quality', 'i2', ('mic',))
        q[:] = [0, 4 | 8]
        q.flag_masks = np.array([1, 2, 4, 8, 16], dtype='i2')
        q.flag_meanings = 'clipped dropout in_downwash position_uncertain runlog_note'
        if vehicle:
            v = nc.createGroup('vehicle')
            p = add_series(v, 'position', [0.0, 0.5, 1.0], {
                'east': [0.0, 3.048, 6.096], 'north': [0.0, 0.0, 0.0], 'up': [10.0, 10.0, 10.0],
                'x': [0.0, -1.0, -2.0], 'y': [0.0, 2.0, 4.0], 'z': [10.0, 10.0, 10.0]})
            p['east'].units = 'm'
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
        assert list(mics['mic_mount']) == ['inverted ground plane', 'pole (4 ft, Part 36)']
        assert list(mics['mic_id']) == ['mic16', 'mic13']
        assert run.mic_index('M13') == run.mic_index('mic13') == 1
        np.testing.assert_allclose(run.receiver('M8'), [10.0, -20.0, -0.8])
        np.testing.assert_allclose(run.receiver('M8', units='ft'), np.array([10.0, -20.0, -0.8]) / 0.3048)
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


def add_segments(path, start, end):
    with netCDF4.Dataset(path, 'a') as nc:
        g = nc.createGroup('segments')
        g.createDimension('segment', len(start))
        g.createDimension('mic', 2)
        k = g.createVariable('kind', str, ('segment',))
        for i, x in enumerate(['pass', 'hover', 'climb', 'descent'][:len(start)]):
            k[i] = x
        for key, vals in (('start', start), ('end', end)):
            var = g.createVariable(key, 'f8', ('segment',))
            var[:] = vals
            var.units = f'seconds since {ORIGIN}'
        g.createVariable('closest_distance', 'f8', ('segment', 'mic'))[:] = [[2.0 + i, 3.0 + i] for i in range(len(start))]


def test_segments(tmp_path):
    path = tmp_path / 'run.nc'
    write_run(path)
    add_segments(path, [0.010, 0.020], [0.020, 0.040])
    with um.open_run(path) as run:
        s = run.segments()
        assert list(s['kind']) == ['pass', 'hover']
        assert s['closest_distance'].shape == (2, 2)
        assert run.segment_samples(1) == (20, 40)


def test_refuses_other_format_versions(tmp_path):
    for version in ('0.3', '0.5', None):
        path = tmp_path / f'run_{version}.nc'
        write_run(path, version=version)
        with pytest.raises(ValueError, match=f'format_version {version!r}.*0.4'):
            um.open_run(path)


def test_sample_ranges_stay_in_the_recording(tmp_path):
    """A window before or after the recording is empty, never wrapped to the end the way a Python slice is."""
    path = tmp_path / 'run.nc'
    counts, step = write_run(path, fs=100.0, n=50)
    with um.open_run(path) as run:
        assert run.samples(-2.0, -1.0) == (0, 0)
        assert run.samples(0.6, 0.9) == (50, 50)
        assert run.samples(-1.0, 0.1) == (0, 10)
        assert run.samples(0.3, 0.2) == (30, 30)
        assert run.samples(0.29, 0.07) == (29, 29)  # 0.29 * 100 = 28.999999999999996
        assert run.samples(0.0, 0.07) == (0, 7)     # 0.07 * 100 = 7.000000000000001
        for s0, s1 in ((0, 60), (45, None), (10, 5), (60, 70)):
            p, t = run.pressure(None, s0, s1), run.time(s0, s1)
            assert p.shape[1] == len(t)
            np.testing.assert_array_equal(p, counts[:, s0:s1] * step[:, None])
            np.testing.assert_allclose(t, np.arange(50)[s0:s1] / 100.0)
        for s0, s1 in ((0, -10), (-5, None)):
            with pytest.raises(ValueError, match='negative'):
                run.pressure(None, s0, s1)
            with pytest.raises(ValueError, match='negative'):
                run.time(s0, s1)


def test_segment_outside_the_recording(tmp_path):
    """Segments come from the flight log, which can begin before the recording and end after it."""
    path = tmp_path / 'run.nc'
    write_run(path, fs=1000.0, n=50)
    add_segments(path, [-0.010, 0.030, 0.060, -0.030], [0.020, 0.070, 0.090, -0.020])
    with um.open_run(path) as run:
        assert run.segment_samples(0) == (0, 20)
        assert run.segment_samples(1) == (30, 50)
        for i in (2, 3):
            with pytest.raises(ValueError, match='outside the recording'):
                run.segment_samples(i)


def test_pressure_takes_one_microphone_or_none(tmp_path):
    path = tmp_path / 'run.nc'
    counts, step = write_run(path)
    with um.open_run(path) as run:
        np.testing.assert_array_equal(run.pressure('M13'), counts[1:2] * step[1])
        np.testing.assert_array_equal(run.pressure(np.int64(0), 3, 7), counts[0:1, 3:7] * step[0])
        assert run.pressure([]).shape == (0, 50)
        with pytest.raises(IndexError):
            run.pressure([2])
        with pytest.raises(IndexError):
            run.mic_index(-1)


def test_missing_groups_are_named(tmp_path):
    path = tmp_path / 'run.nc'
    write_run(path, vehicle=False)
    with um.open_run(path) as run:
        with pytest.raises(KeyError, match="no group /vehicle/position.*'ambient'"):
            run.track()
    path = tmp_path / 'run2.nc'
    write_run(path)
    with um.open_run(path) as run:
        with pytest.raises(KeyError, match='no group /vehicle/air_data.*position'):
            run.vehicle('air_data')
        with pytest.raises(ValueError, match='units'):
            run.track('yd')
        with pytest.raises(ValueError, match='frame'):
            run.track(frame='ned')


def test_rotors_weather_and_air_data(tmp_path):
    path = tmp_path / 'run.nc'
    write_run(path)
    with netCDF4.Dataset(path, 'a') as nc:
        r = nc.createGroup('rotors')
        r.clock = 'onboard clock + 0.1 s'
        r.prop_aligned_angle = np.array([0.1, 0.2, 0.3])
        r.prop_aligned_angle_source = 'a test set'
        add_series(r, 'motor1', [0.0, 0.1], {'speed': [-300.0, -301.0], 'angle': [0.5, -3.0]})
        add_series(r, 'motor2', [0.05, 0.15], {'speed': [310.0, 311.0]})
        add_series(r, 'summary', [0.0], {'speed': [1.0]})
        w = nc.createGroup('weather')
        g = add_series(w, 'kestrel_1', [0.0, 60.0, 120.0], {'temperature': [13.0, 13.5, 14.0],
                                                             'note': ['a', 'b', 'c']}, {'location': 'ground'})
        g['temperature'].units = 'degC'
        g.createVariable('station_pressure', 'f8', ('time',))[0] = 94070.0  # the other cells unwritten
        ad = nc.createGroup('air_data')
        add_series(ad, 'wx150_1', [0.0, 0.5], {'wind_speed_apparent': [4.0, 4.5]}, {'mounting': 'mast'})
        add_series(ad, 'airspeed', [0.0, 0.5], {'airspeed': [3.0, 3.5]}, {'calibration_C': 1.25})
    with um.open_run(path) as run:
        ro = run.rotors()
        assert sorted(ro) == [1, 2]
        np.testing.assert_array_equal(ro[1]['speed'], [-300.0, -301.0])
        np.testing.assert_array_equal(ro[1]['angle'], [0.5, -3.0])
        np.testing.assert_array_equal(ro[2]['time'], [0.05, 0.15])
        assert 'angle' not in ro[2]
        assert ro[2]['@prop_aligned_angle'] == 0.2 and ro[1]['@prop_aligned_angle'] == 0.1
        assert ro[1]['@clock'] == 'onboard clock + 0.1 s' and ro[2]['@prop_aligned_angle_source'] == 'a test set'
        wx = run.weather()['kestrel_1']
        np.testing.assert_array_equal(wx['temperature'], [13.0, 13.5, 14.0])
        assert list(wx['note']) == ['a', 'b', 'c'] and wx['@location'] == 'ground'
        assert wx['station_pressure'][0] == 94070.0 and np.isnan(wx['station_pressure'][1:]).all()
        assert run.units('weather/kestrel_1')['temperature'] == 'degC'
        assert run.units('vehicle/position')['east'] == 'm'
        air = run.air_data()
        assert sorted(air) == ['airspeed', 'wx150_1'] and air['airspeed']['@calibration_C'] == 1.25
        np.testing.assert_array_equal(air['wx150_1']['wind_speed_apparent'], [4.0, 4.5])


def test_time_axis_on_another_origin_is_refused(tmp_path):
    path = tmp_path / 'run.nc'
    write_run(path)
    with netCDF4.Dataset(path, 'a') as nc:
        add_series(nc.createGroup('weather'), 'kestrel_1', [0.0], {'temperature': [13.0]},
                   time_units='seconds since 2023-06-08T13:00:00+00:00')
    with um.open_run(path) as run:
        with pytest.raises(ValueError, match='time_origin'):
            run.weather()


def test_time_origin_in_z_form(tmp_path):
    path = tmp_path / 'run.nc'
    write_run(path)
    with netCDF4.Dataset(path, 'a') as nc:
        nc.time_origin = '2023-06-08T14:00:00.250000Z'
    with um.open_run(path) as run, warnings.catch_warnings():
        warnings.simplefilter('error')
        assert run.time_origin == np.datetime64('2023-06-08T14:00:00.25', 'ns')


def test_close_is_idempotent(tmp_path):
    path = tmp_path / 'run.nc'
    write_run(path)
    with um.open_run(path) as run:
        run.close()
    run.close()
    with pytest.raises(FileNotFoundError):
        um.list_runs(tmp_path / 'nowhere')


def test_flags_keep_their_integers(tmp_path):
    """Flag variables (PX4 states, microphone quality) come back as integers, and quality() decodes them."""
    path = tmp_path / 'run.nc'
    write_run(path)
    with netCDF4.Dataset(path, 'a') as nc:
        st = add_series(nc['vehicle'], 'status', [0.0, 0.5, 1.0], {'nav_state': [2.0, 3.0, 3.0]})
        st.renameVariable('nav_state', 'nav_float')
        v = st.createVariable('nav_state', 'i1', ('time',))
        v[:] = [2, 3, 3]
        v.flag_values = np.array([2, 3], dtype='i1')
        v.flag_meanings = 'posctl auto_mission'
    with um.open_run(path) as run:
        nav = run.vehicle('status')['nav_state']
        assert nav.dtype.kind == 'i' and list(nav) == [2, 3, 3]
        q = run.microphones()['mic_quality']
        assert q.dtype.kind == 'i' and list(q) == [0, 12]
        assert run.quality('M8') == [] and run.quality('mic13') == ['in_downwash', 'position_uncertain']
