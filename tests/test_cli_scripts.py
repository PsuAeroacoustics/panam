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
