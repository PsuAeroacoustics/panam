"""nod_database's inspection functions (summaries, selection, maps, envelope),
which nod_tool.py prints and draws, on a database built in the test."""
import matplotlib.pyplot as plt
import numpy as np
import pytest

from panam import flight_acoustics as fa
from panam import nod_database as nd
from tools import nod_tool
from sphere_helpers import minimal_hemisphere, write_sphere_directory


@pytest.fixture(scope='module')
def database(tmp_path_factory):
    """Two measured conditions (60 kt level; 80 kt, -6 deg, 6 dB louder), each
    with a gated band, and the three synthesized hover conditions of a
    fixed_load_factor build."""
    directory = tmp_path_factory.mktemp('nod_tool')
    hemispheres = []
    for speed, angle, offset in ((60.0, 0.0, 0.0), (80.0, -6.0, 6.0)):
        hemisphere = minimal_hemisphere()
        hemisphere['third_octave']['bands_db'] += offset
        hemisphere['third_octave']['bands_db'][2, 2:, :] = -np.inf
        hemispheres.append((hemisphere, speed, angle))
    write_sphere_directory(directory / 'spheres', hemispheres)
    path = directory / 'db.nod'
    with pytest.warns(UserWarning):  # the spheres carry no azimuth_reference or Doppler flag
        fa.build_empirical_database(str(directory / 'spheres'), str(path), load_factors=None)
    return nd.load_nod(path)


def _replace(condition, **changes):
    return nd.NodCondition(**{**vars(condition), **changes})


def test_summary_gives_one_row_per_condition(database):
    rows = [nd.summarize_condition(c) for c in database]
    assert [r['group'] for r in rows] == ['sphere{}'.format(i) for i in range(5)]
    assert [r['condition_origin'] for r in rows] == ['measured'] * 2 + ['synthesized_hover'] * 3
    np.testing.assert_allclose([r['speed_kt'] for r in rows[:2]], [60.0, 80.0])
    assert [r['flight_path_angle_deg'] for r in rows[:2]] == [0.0, -6.0]
    assert [r['load_factor'] for r in rows] == [1.0] * 5
    assert rows[1]['dBA_max'] == pytest.approx(np.max(database[1].dBA))
    assert rows[1]['dBA_max'] - rows[0]['dBA_max'] == pytest.approx(6.0, abs=1e-6)
    # The fraction is of the lower hemisphere: the mirrored upper surface
    # (coverage 0) does not halve a fully measured sphere.
    # (coverage 0) does not count; the gated band leaves some lower cells unset.
    lower = np.abs(database[1].phi_deg.reshape(database[1].coverage.shape[:2])[:, 0]) <= 90.0
    assert rows[1]['coverage_fraction'] == pytest.approx(np.mean(database[1].coverage[lower]))
    full = np.zeros_like(database[1].coverage)
    full[lower] = 1.0
    assert np.mean(full) < 0.6
    assert nd.summarize_condition(_replace(database[1], coverage=full))['coverage_fraction'] == 1.0


def test_summary_marks_what_a_group_lacks(database):
    row = nd.summarize_condition(_replace(database[0], coverage=None, load_factor=np.nan))
    assert np.isnan(row['coverage_fraction']) and np.isnan(row['load_factor'])
    assert nd.summarize_condition(_replace(database[0], dBA=np.full(3, -np.inf)))['dBA_max'] == -np.inf


def test_select_condition_matches_speed_and_angle(database):
    assert nd.select_condition(database, 80.2, -6.0).name == 'sphere1'
    with pytest.raises(ValueError, match='no condition'):
        nd.select_condition(database, 70.0, -6.0)
    with pytest.raises(ValueError, match='no condition'):
        nd.select_condition(database, 80.0, -3.0)


def test_select_condition_asks_for_the_load_factor_of_a_ladder(database):
    ladder = nd.NodDatabase('x', [database[0], _replace(database[0], name='other', load_factor=1.5)], {}, {})
    with pytest.raises(ValueError, match='2 conditions'):
        nd.select_condition(ladder, 60.0, 0.0)
    assert nd.select_condition(ladder, 60.0, 0.0, load_factor=1.5).name == 'other'


def test_select_condition_checks_the_load_factor_of_a_single_match(database):
    with pytest.raises(ValueError, match='load factor 1.5'):
        nd.select_condition(database, 80.0, -6.0, load_factor=1.5)
    assert nd.select_condition(database, 80.0, -6.0, load_factor=1.0).name == 'sphere1'
    unknown = nd.NodDatabase('x', [_replace(database[1], load_factor=np.nan)], {}, {})
    with pytest.raises(ValueError, match='weight coefficient'):
        nd.select_condition(unknown, 80.0, -6.0, load_factor=1.0)


@pytest.mark.parametrize('quantity, band, label', [
    ('dBA', None, 'A-weighted level'), ('EAA', None, 'Excess atmospheric attenuation'),
    ('band', 125.0, '125 Hz band level'), ('coverage', None, 'Fraction of bands measured'),
    ('coverage', 160.0, 'Measured, 160 Hz')])
def test_plot_condition_maps_each_quantity(database, quantity, band, label):
    fig = nd.plot_condition(database[1], quantity, band_hz=band)
    assert label in fig.axes[-1].get_ylabel()
    plt.close(fig)


def test_plot_condition_puts_starboard_where_art2umapr_does(database):
    """The channels go through art2umapr in radians: a field loud only at
    phi > 0 (starboard, ART) covers the projected starboard point, not its mirror."""
    c = database[0]
    fig = nd.plot_condition(_replace(c, dBA=np.where(c.phi_deg > 0, 90.0, 60.0)), 'dBA',
                            level_range=(60.0, 90.0))
    azi, elv = fa.art2umapr(np.radians([90.0]), np.radians([90.0]))
    x, y = fa.lambert_ea(elv, fa.lambert_lon(azi))
    assert abs(x[0]) > 0.5
    contours = fig.axes[0].collections[0]
    loud = contours.get_paths()[-1]
    assert loud.contains_point((0.9 * x[0], 0.9 * y[0]))
    assert not loud.contains_point((-0.9 * x[0], 0.9 * y[0]))
    plt.close(fig)


def test_plot_condition_maps_the_lower_hemisphere_only(database):
    """The rows with |phi| > 90 deg (the mirrored upper surface) would project
    outside the Lambert circle, radius sqrt(2), and fill the corners."""
    c = database[0]
    assert np.max(np.abs(c.phi_deg)) > 90.0
    with pytest.raises(ValueError, match='no finite'):
        nd.plot_condition(_replace(c, dBA=np.where(np.abs(c.phi_deg) > 90.0, 90.0, -np.inf)), 'dBA')
    fig = nd.plot_condition(_replace(c, dBA=60.0 + np.abs(c.phi_deg) / 6.0), 'dBA')
    radius = max(np.max(np.hypot(*path.vertices.T))
                 for path in fig.axes[0].collections[0].get_paths() if len(path.vertices))
    assert radius <= np.sqrt(2.0) + 1e-6
    assert fig.axes[-1].get_ylim()[1] < 80.0  # the data stop at 75 dB, |phi| = 90 deg; 90 dB above
    plt.close(fig)


def test_plot_condition_rejects_what_the_group_lacks(database):
    with pytest.raises(ValueError, match='needs a band'):
        nd.plot_condition(database[0], 'band')
    with pytest.raises(ValueError, match='no 1000 Hz band'):
        nd.plot_condition(database[0], 'band', band_hz=1000.0)
    with pytest.raises(ValueError, match='not one of'):
        nd.plot_condition(database[0], 'SEL')
    with pytest.raises(ValueError, match='no coverage'):
        nd.plot_condition(_replace(database[0], coverage=None), 'coverage')


def test_plot_envelope_colors_by_origin(database):
    fig = nd.plot_envelope(database)
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ['measured', 'synthesized hover']
    np.testing.assert_allclose(np.asarray(ax.collections[0].get_offsets()), [[60.0, 0.0], [80.0, -6.0]])
    assert np.asarray(ax.collections[1].get_offsets()).shape == (3, 2)
    assert ax.get_xlabel() == 'Speed, kt'
    plt.close(fig)


def test_nod_tool_rejects_a_load_factor_with_a_group(tmp_path):
    with pytest.raises(SystemExit, match='--condition only'):
        nod_tool.main(['plot', str(tmp_path / 'absent.nod'), '-g', 'sphere0', '-n', '1.5'])
