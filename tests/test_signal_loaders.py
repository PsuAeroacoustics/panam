import numpy as np
import pytest

import flight_acoustics as fa


def test_load_nc_signal_time_matches_the_samples(tmp_path):
    """np.arange(start, stop, 1/fs) can give one sample too many (here 12346 for 12345)."""
    from netCDF4 import Dataset
    path = tmp_path / 'signal.nc'
    n, fs = 12345, 48000.0
    with Dataset(str(path), 'w') as ds:
        ds.createDimension('n', n)
        ds.createVariable('pressure', 'f8', ('n',))[:] = np.sin(np.arange(n) / 10.0)
        ds.X, ds.Y, ds.Z = 1.0, 2.0, 3.0
        ds.sample_rate = fs
        ds.start_time = 12.5
    pressure, time, location = fa.load_nc_signal(str(path))
    assert time.shape == pressure.shape == (n,)
    assert time[0] == 12.5
    np.testing.assert_allclose(np.diff(time), 1.0 / fs, rtol=1e-9)
    np.testing.assert_array_equal(location, [1.0, 2.0, 3.0])


def test_load_nc_signal_reads_fill_values_as_missing(tmp_path):
    """Fill-value samples come back NaN, in a plain array, not as masked pressures."""
    from netCDF4 import Dataset
    path = tmp_path / 'signal.nc'
    with Dataset(str(path), 'w') as ds:
        ds.createDimension('n', 5)
        ds.createVariable('pressure', 'f8', ('n',), fill_value=-9999.0)[:] = [0.1, -9999.0, 0.3, 0.4, 0.5]
        ds.X, ds.Y, ds.Z = 0.0, 0.0, 0.0
        ds.sample_rate = 10.0
        ds.start_time = 0.0
    pressure, _, _ = fa.load_nc_signal(str(path))
    assert not np.ma.isMaskedArray(pressure)
    np.testing.assert_array_equal(pressure, [0.1, np.nan, 0.3, 0.4, 0.5])


@pytest.mark.parametrize('channels', [1, 2])
def test_load_uff_signal_single_and_multi_channel(tmp_path, channels):
    """pyuff returns a bare dict for a single set, which used to raise KeyError: 0."""
    pyuff = pytest.importorskip('pyuff')
    path = tmp_path / 'signal.uff'
    t = np.arange(1000) / 1000.0
    pyuff.UFF(str(path)).write_sets([
        pyuff.prepare_58(func_type=1, rsp_node=ch, rsp_dir=1, ref_node=1, ref_dir=1,
                         data=np.sin(2 * np.pi * 5.0 * ch * t), x=t, id1='mic%d' % ch, binary=0,
                         abscissa_spacing=1, abscissa_min=0.0, abscissa_inc=1e-3,
                         orddenom_spec_data_type=0, abscissa_spec_data_type=17,
                         ordinate_spec_data_type=0)
        for ch in range(1, channels + 1)], mode='overwrite')
    pressures, fs, names, time = fa.load_UFF_signal(str(path))
    assert pressures.shape == (channels, t.size)
    assert fs == pytest.approx(1000.0)
    assert names == ['mic%d' % ch for ch in range(1, channels + 1)]
    single, *_ = fa.load_UFF_signal(str(path), sets=0)
    assert single.shape == (1, t.size)
