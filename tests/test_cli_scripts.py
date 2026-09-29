"""Smoke tests for the command-line scripts, run as a user would run them."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import flight_acoustics as fa

REPO = Path(__file__).resolve().parents[1]


def _run(script, *args, cwd):
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONPATH=str(REPO))
    return subprocess.run([sys.executable, str(REPO / script), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=600)


def test_fried_egg_plot_runs(tmp_path):
    """It passed threshhold= (sic) to fried_egg_plot and so failed on every run."""
    from test_write_aam_hemisphere_netcdf import _minimal_hemisphere
    spheres = tmp_path / 'spheres'
    spheres.mkdir()
    (spheres / 'vehicle.cfg').write_text(
        '[Main Rotor]\nradius = 5.334\ntip speed = 230.7\nblades = 4\n'
        '[Tail Rotor]\nradius = 0.8255\ntip speed = 216.1\nblades = 2\n'
        '[Atmosphere]\ndensity = 1.070\ntemperature = 280.37\n'
        '[Vehicle]\nweight = 2250\ndrag = 0.8175\n')
    for i, (speed, angle) in enumerate([(40, -6), (60, -3), (80, 0), (100, -6), (60, 3), (80, -9)]):
        hemisphere = _minimal_hemisphere()
        hemisphere['third_octave']['bands_db'] += 3.0 * i
        fa.write_aam_hemisphere_netcdf(str(spheres / f'X{101 + i:03d}.nc'), hemisphere,
                                       mode='third_octave', radius_ft=100.0, speed_knots=float(speed),
                                       flight_path_angle_deg=float(angle), title='t')
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
