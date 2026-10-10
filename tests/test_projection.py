"""Ground footprints: project_sphere, plot_projection and project_directory."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pytest

import flight_acoustics as fa
from sphere_helpers import minimal_hemisphere

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = str(REPO / 'example_data' / 'AS350B3108.nc')


@pytest.mark.parametrize('cutoff', [-10.0, 0.0, 90.0, 95.0])
def test_project_sphere_refuses_a_cutoff_whose_rays_miss_the_ground(cutoff):
    with pytest.raises(ValueError, match='elv_cutoff'):
        fa.project_sphere(EXAMPLE, 150.0, cutoff)


def test_plot_projection_needs_three_directions():
    with pytest.raises(ValueError, match='needs 3'):
        fa.plot_projection(EXAMPLE, altitude=150, cutoff=89.0)


def test_plot_projection_reports_a_bad_cutoff_without_a_traceback(tmp_path):
    from test_cli_scripts import run_here
    result = run_here('plot_projection.py', EXAMPLE, '-c', '0', '-o', 'out.png', cwd=tmp_path)
    assert result.returncode == 2
    assert 'between 0 and 90' in result.stderr and 'Traceback' not in result.stderr


@pytest.mark.parametrize('cutoff', [20.0, 30.0, 45.0])
def test_plot_projection_levels_cover_the_footprint(cutoff):
    """contourf fills only between the first and last level; at 45 deg the old levels,
    rounded to 0.1 dB either way, stopped at 79.1 dB with the peak at 79.13."""
    _, _, level_a, _, _ = fa.project_sphere(EXAMPLE, 150.0, cutoff)
    level_a = level_a[np.isfinite(level_a)]
    figure, _, contours = fa.plot_projection(EXAMPLE, altitude=150, cutoff=cutoff)
    plt.close(figure)
    assert contours.levels[0] <= level_a.min()
    assert contours.levels[-1] >= level_a.max()
    np.testing.assert_allclose(np.diff(contours.levels), np.diff(contours.levels)[0])


def _reflist_directory(directory, runs):
    """Spheres named as the 2017 sets name them, the vehicle's name ending in digits
    (Be407100.nc is the Bell 407's run 100), with a reflist giving each run's conditions."""
    directory.mkdir()
    (directory / 'vehicle.cfg').write_text('[Main Rotor]\nblades = 4\n[Tail Rotor]\nblades = 2\n'
                                           '[Option]\nfbar = 0.01\nreflist = runs.xlsx\n')
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet['B1'] = 'Run'
    for row, (run, speed) in enumerate(runs, start=2):
        sheet[f'B{row}'] = run
        sheet[f'W{row}'] = speed
        sheet[f'AH{row}'] = 0.1 * speed / 60.0
        sheet[f'AI{row}'] = 0.006
        sheet[f'AJ{row}'] = 0.6
    workbook.save(directory / 'runs.xlsx')
    for run, speed in runs:
        fa.write_aam_hemisphere_netcdf(str(directory / f'Be407{run:03d}.nc'), minimal_hemisphere(),
                                       mode='third_octave', radius_ft=100.0, speed_knots=float(speed),
                                       flight_path_angle_deg=-3.0, title='t')
    return directory


def test_project_directory_takes_the_run_number_from_the_last_three_digits(tmp_path):
    directory = _reflist_directory(tmp_path / 'Be407', [(100, 60.0), (101, 80.0)])
    out = fa.project_directory(str(directory), altitude=150, cutoff=20)
    speeds, runs = out[0], out[8]
    np.testing.assert_array_equal(runs, [100, 101])
    np.testing.assert_array_equal(speeds, [60.0, 80.0])          # the reflist's, by run
