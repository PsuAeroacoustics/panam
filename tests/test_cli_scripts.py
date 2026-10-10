"""Smoke tests for the command-line scripts, run as a user would run them."""

import os
import subprocess
import sys
from pathlib import Path

from sphere_helpers import minimal_hemisphere, write_sphere_directory

REPO = Path(__file__).resolve().parents[1]


def _run(script, *args, cwd):
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONPATH=str(REPO))
    return subprocess.run([sys.executable, str(REPO / script), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=600)


def test_fried_egg_plot_runs(tmp_path):
    """It passed threshhold= (sic) to fried_egg_plot and so failed on every run."""
    hemispheres = []
    for i, (speed, angle) in enumerate([(40, -6), (60, -3), (80, 0), (100, -6), (60, 3), (80, -9)]):
        hemisphere = minimal_hemisphere()
        hemisphere['third_octave']['bands_db'] += 3.0 * i
        hemispheres.append((hemisphere, speed, angle))
    write_sphere_directory(tmp_path / 'spheres', hemispheres)
    result = _run('fried_egg_plot.py', 'spheres', '-o', 'eggs.png', '-x', '30:110', '-y=-10:5', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'eggs.png').stat().st_size > 0


def test_default_outputs_work_in_a_fresh_checkout(tmp_path):
    """Both default to demo_plots/, which is git-ignored and was never created."""
    for script, output in (('atmomap.py', 'atmomap.pdf'), ('ega_plot.py', 'ega_plot.pdf')):
        result = _run(script, cwd=tmp_path)
        assert result.returncode == 0, result.stderr
        assert (tmp_path / 'demo_plots' / output).is_file()


def test_board_field_plot_runs(tmp_path):
    result = _run('board_field_plot.py', '--frequency', '1000', '--elevation', '30', '--grid', '21x9',
                  '-o', 'field.png', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'field.png').stat().st_size > 0


def test_spectrogram_frequency_option_sets_display_range(tmp_path):
    import h5py
    import numpy as np
    import runpy
    from unittest.mock import patch
    import matplotlib.pyplot as plt
    import flight_acoustics as fa
    path = tmp_path / 'signal.h5'
    with h5py.File(path, 'w') as handle:
        dataset = handle.create_dataset('Table1/mic', data=np.sin(np.arange(4096)))
        dataset.attrs['ChannelInformationSamplingPeriod'] = [.001]
    plot = fa.plot_spectrogram
    with patch.object(sys, 'argv', ['spectrogram_plot.py', str(path), '-s', 'mic', '-f', '100:200']), \
            patch.object(fa, 'plot_spectrogram', wraps=plot) as mocked, \
            patch('cli.save_or_show'):
        runpy.run_path(str(REPO / 'spectrogram_plot.py'), run_name='__main__')
    assert mocked.call_args.kwargs['flim'] == (100, 200)
    assert np.array_equal(mocked.call_args.args[0], np.sin(np.arange(4096)))
    plt.close('all')


def test_vold_kalman_filter_demo_runs(tmp_path):
    """The demo's crossing chirp fell to negative frequency over its 5 s record."""
    result = _run('vold_kalman_filter.py', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'Spectrogram generation skipped' not in result.stdout
    plots = tmp_path / 'demo_plots'
    assert (plots / 'vkf_example.png').stat().st_size > 0
    assert len(list(plots.glob('vkf_spectrogram_*.png'))) == 10


def _spectrogram_call(path, *options):
    """Run spectrogram_plot.py in-process and return plot_spectrogram's call."""
    import runpy
    from unittest.mock import patch
    import matplotlib.pyplot as plt
    import flight_acoustics as fa
    plot = fa.plot_spectrogram
    try:
        with patch.object(sys, 'argv', ['spectrogram_plot.py', str(path), *options]), \
                patch.object(fa, 'plot_spectrogram', wraps=plot) as mocked, \
                patch('cli.save_or_show'):
            runpy.run_path(str(REPO / 'spectrogram_plot.py'), run_name='__main__')
    finally:
        plt.close('all')
    return mocked.call_args


def _write_nc_signal(path, pressure, sample_rate, start_time):
    from netCDF4 import Dataset
    with Dataset(str(path), 'w') as ds:
        ds.createDimension('n', pressure.size)
        ds.createVariable('pressure', 'f8', ('n',))[:] = pressure
        ds.X, ds.Y, ds.Z = 0.0, 0.0, 0.0
        ds.sample_rate = sample_rate
        ds.start_time = start_time


def test_spectrogram_netcdf_uses_the_file_start_time(tmp_path, capsys):
    """The netCDF start_time was dropped: the axis began at 0, and -x in the
    file's times sliced past the end of the record and crashed in scipy."""
    import numpy as np
    import pytest
    pressure = np.sin(np.arange(20000) / 3.0)
    path = tmp_path / 'signal.nc'
    _write_nc_signal(path, pressure, 1000.0, 50000.0)

    call = _spectrogram_call(path)
    assert call.kwargs['time0'] == 50000.0
    assert np.array_equal(call.args[0], pressure)

    call = _spectrogram_call(path, '-x', '50005:50010')
    assert call.kwargs['time0'] == pytest.approx(50005.0)
    np.testing.assert_array_equal(call.args[0], pressure[5000:10001])

    with pytest.raises(SystemExit) as raised:
        _spectrogram_call(path, '-x', '0:5')
    assert raised.value.code == 2
    assert 'the record spans 50000 to 50020 s' in capsys.readouterr().err
