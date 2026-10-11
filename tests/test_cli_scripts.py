"""Smoke tests for the command-line scripts, run as a user would run them."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sphere_helpers import minimal_hemisphere, write_sphere_directory

REPO = Path(__file__).resolve().parents[1]


def _run(script, *args, cwd):
    """Run ``script`` in a fresh interpreter, as a user would."""
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONPATH=str(REPO))
    return subprocess.run([sys.executable, str(REPO / script), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=600)


def run_here(script, *args, cwd):
    """Run ``script`` as __main__ in this interpreter, which has PANAM imported
    already (a fresh one spends ~2 s importing it); same result fields as _run."""
    import contextlib
    import io
    import runpy
    from types import SimpleNamespace
    from unittest.mock import patch
    import matplotlib.pyplot as plt
    stdout, stderr, code = io.StringIO(), io.StringIO(), 0
    here = os.getcwd()
    try:
        os.chdir(cwd)
        with patch.object(sys, 'argv', [script, *args]), contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(stderr):
            runpy.run_path(str(REPO / script), run_name='__main__')
    except SystemExit as exit_:
        code = exit_.code if isinstance(exit_.code, int) else (0 if exit_.code is None else 1)
    finally:
        os.chdir(here)
        plt.close('all')
    return SimpleNamespace(returncode=code, stdout=stdout.getvalue(), stderr=stderr.getvalue())


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
    path = tmp_path / 'signal.h5'
    with h5py.File(path, 'w') as handle:
        dataset = handle.create_dataset('Table1/mic', data=np.sin(np.arange(4096)))
        dataset.attrs['ChannelInformationSamplingPeriod'] = [.001]
    call = _spectrogram_call(path, '-s', 'mic', '-f', '100:200')
    assert call.kwargs['flim'] == (100, 200)
    assert np.array_equal(call.args[0], np.sin(np.arange(4096)))


def test_vold_kalman_filter_demo_runs(tmp_path):
    """The demo's crossing chirp fell to negative frequency over its 5 s record."""
    result = run_here('vold_kalman_filter.py', cwd=tmp_path)
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


def test_ega_plot_help_examples_run(tmp_path):
    """The epilog's second example used a -t flag the parser does not have."""
    import shlex
    help_text = run_here('ega_plot.py', '--help', cwd=tmp_path).stdout
    examples = [shlex.split(line)[1:] for line in help_text.splitlines() if line.strip().startswith('ega_plot.py ')]
    assert len(examples) == 3
    for i, example in enumerate(examples):
        if '-o' not in example:
            example += ['-o', f'example{i}.pdf']
        result = run_here('ega_plot.py', *example, cwd=tmp_path)
        assert result.returncode == 0, (example, result.stderr)
        assert (tmp_path / example[example.index('-o') + 1]).stat().st_size > 0


EXAMPLE_SPHERE = REPO / 'example_data' / 'AS350B3108.nc'


def test_plot_projection_runs(tmp_path):
    result = run_here('plot_projection.py', str(EXAMPLE_SPHERE), '-a', '150', '-c', '20', '-o', 'projection.png',
                  cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'projection.png').stat().st_size > 0


def test_nc_lambert_ea_runs(tmp_path):
    result = run_here('nc_lambert_ea.py', str(EXAMPLE_SPHERE), '-f', '50:2000', '-w', 'A', '-g', 'art',
                  '-o', 'lambert.png', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'lambert.png').stat().st_size > 0


def test_build_empirical_database_runs(tmp_path):
    from netCDF4 import Dataset
    hemispheres = []
    for i, (speed, angle) in enumerate([(60, -3), (80, 0)]):
        hemisphere = minimal_hemisphere()
        hemisphere['third_octave']['bands_db'] += 3.0 * i
        hemispheres.append((hemisphere, speed, angle))
    write_sphere_directory(tmp_path / 'spheres', hemispheres)
    result = run_here('build_empirical_database.py', 'spheres', 'database.nod', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    with Dataset(str(tmp_path / 'database.nod')) as ds:
        assert len(ds.groups) >= 2
    # The output exists now: it is replaced only with --overwrite.
    result = run_here('build_empirical_database.py', 'spheres', 'database.nod', '--fixed-load-factor',
                      '--exclude', 'X102.nc', cwd=tmp_path)
    assert result.returncode == 2 and '--overwrite' in result.stderr
    result = run_here('build_empirical_database.py', 'spheres', 'database.nod', '--fixed-load-factor',
                      '--exclude', 'X102.nc', '--extended-fpa=-24:35', '--overwrite', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    with Dataset(str(tmp_path / 'database.nod')) as ds:
        assert ds.source_spheres == 'X101.nc' and ds.excluded_spheres == 'X102.nc'


def test_compare_databases_runs(tmp_path):
    import flight_acoustics as fa
    for name, offset in (('a', 0.0), ('b', 1.0)):
        hemisphere = minimal_hemisphere()
        hemisphere['third_octave']['bands_db'] += offset
        write_sphere_directory(tmp_path / name, [(hemisphere, 60, -3), (hemisphere, 80, 0)])
        fa.build_empirical_database(str(tmp_path / name), str(tmp_path / f'{name}.nod'), load_factors=None)
    same = run_here('compare_databases.py', 'a.nod', 'a.nod', '--validate', cwd=tmp_path)
    assert same.returncode == 0, same.stderr
    assert 'no differences' in same.stdout
    different = run_here('compare_databases.py', 'a.nod', 'b.nod', cwd=tmp_path)
    assert different.returncode == 1, different.stderr
    assert 'band energy +1 dB' in different.stdout
    tolerated = run_here('compare_databases.py', 'a.nod', 'b.nod', '--tolerance', '1.5', cwd=tmp_path)
    assert tolerated.returncode == 0, tolerated.stdout


def test_noise_abatement_plate_cache_lists_and_clears(tmp_path):
    (tmp_path / 'tables').mkdir()
    (tmp_path / 'tables' / 'old.pkl').write_bytes(b'')
    result = run_here('noise_abatement_2017.py', '--plate-cache', 'list', '--plate-table-directory', 'tables',
                      cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'old format' in result.stdout and '1 plate tables' in result.stdout
    result = run_here('noise_abatement_2017.py', '--plate-cache', 'clear', '--plate-table-directory', 'tables',
                      cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert not list((tmp_path / 'tables').iterdir())
    assert run_here('noise_abatement_2017.py', cwd=tmp_path).returncode != 0     # aircraft still required


def test_array_planner_subcommands_run(tmp_path):
    design = ['--nmics', '6', '--altitude', '150']
    result = run_here('array_planner.py', 'design', *design, '--format', 'json', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)['ymics']) == 6
    result = run_here('array_planner.py', 'coverage', *design, '--x-offsets=-200,0,200', '-o', 'coverage.png',
                  cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'coverage.png').stat().st_size > 0
    result = run_here('array_planner.py', 'kml', *design, '--ref-lat', '40.8', '--ref-lon', '-77.9', '--heading', '90',
                  '--kmz', 'array.kmz', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'array.kmz').stat().st_size > 0


def test_sphere_diff_averages_and_compares(tmp_path):
    import numpy as np
    louder = minimal_hemisphere()
    louder['third_octave']['bands_db'] += 6.0
    write_sphere_directory(tmp_path / 'spheres', [(minimal_hemisphere(), 60, -6), (louder, 60, -6)])
    result = run_here('sphere_diff.py', 'average', 'mean.nc', 'spheres/X101.nc', 'spheres/X102.nc',
                      cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    expected = 10.0 * np.log10((10 ** 8.0 + 10 ** 8.6) / 2.0)
    result = run_here('sphere_diff.py', 'compare', 'spheres/X101.nc', 'mean.nc', '-f', '100:125',
                      '-o', 'diff.png', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    rows = [line.split() for line in result.stdout.splitlines()]
    assert [row[0] for row in rows[2:]] == ['100', '125', 'overall']
    assert float(rows[2][2]) == pytest.approx(expected - 80.0, abs=0.01)
    assert (tmp_path / 'diff.png').stat().st_size > 0


def test_depropagation_tutorial_runs(tmp_path):
    """By default it writes only into demo_plots/ in the current directory."""
    from netCDF4 import Dataset
    result = run_here('depropagation_tutorial.py', '--mics', '5', '--duration', '8', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'Recovered minus true' in result.stdout
    assert [path.name for path in tmp_path.iterdir()] == ['demo_plots']
    plots = tmp_path / 'demo_plots'
    assert (plots / 'tutorial_difference.png').stat().st_size > 0
    assert sorted(path.name for path in (plots / 'tutorial_spheres').iterdir()) == ['X101.nc', 'vehicle.cfg']
    with Dataset(str(plots / 'tutorial.nod')) as ds:
        assert len(ds.groups) >= 1
    result = run_here('depropagation_tutorial.py', '--mics', '3', '--duration', '6', '-o', 'elsewhere',
                      cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'elsewhere' / 'tutorial.nod').is_file()


def test_array_planner_hover_and_takeoff_run(tmp_path):
    result = run_here('array_planner.py', 'hover', '--radii', '50,100', '--nmics', '8', '--altitudes', '30,60',
                      '--vehicle-headings', '0,90', '-o', 'hover.png', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'hover.png').stat().st_size > 0
    (tmp_path / 'takeoff.toml').write_text(
        'radii = [150]\nnmics = 12\norigin = [-100, 0]\nflight_path_angle = 8\n'
        'ref_lat = 40.8\nref_lon = -77.9\nheading = 90\nkmz = "takeoff.kmz"\n')
    result = run_here('array_planner.py', 'takeoff', '--case', 'takeoff.toml', '-o', 'takeoff.png', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'takeoff.png').stat().st_size > 0
    assert (tmp_path / 'takeoff.kmz').stat().st_size > 0
