"""array_planner.py: the design inputs it accepts and the designs it prints."""

import numpy as np
import pytest

import array_planner


@pytest.mark.parametrize('options, named', [
    (['--nmics', '1'], '--nmics'),
    (['--nmics', '0'], '--nmics'),
    (['--nmics', '12', '--min-elevation', '0'], '--min-elevation'),
    (['--nmics', '12', '--min-elevation', '90'], '--min-elevation'),
    (['--nmics', '12', '--target-elv', '0'], '--target-elv'),
    (['--nmics', '12', '--target-elv', '120'], '--target-elv'),
    (['--nmics', '12', '--altitude', '-150'], '--altitude'),
    (['--nmics', '12', '--speed', '0'], '--speed'),
    (['--nmics', '12', '--rate', '0'], '--rate'),
])
def test_meaningless_inputs_are_refused_by_name(options, named, capsys):
    """These printed or plotted nonsense with exit 0 (a single mic at -850 m,
    mics at +/-2.4e18 m, an empty design, a mirrored array), or failed with a
    bare 'division by zero'."""
    argv = ['design', '--altitude', '150', *options]
    with pytest.raises(SystemExit) as raised:
        array_planner.main(argv)
    assert raised.value.code == 2
    assert named in capsys.readouterr().err


def test_the_outer_microphone_sits_at_min_elevation_overhead_at_target_elv_90(capsys):
    assert array_planner.main(['design', '--nmics', '12', '--altitude', '150', '--min-elevation', '10']) == 0
    ymics = np.array(capsys.readouterr().out.strip().split(','), dtype=float)
    assert ymics.size == 12
    np.testing.assert_allclose(np.degrees(np.arctan2(150.0, np.abs(ymics[[0, -1]]))), 10.0, rtol=1e-5)


def test_min_elevation_is_measured_in_the_target_elv_plane(capsys):
    """Seen from overhead the outer mic is at atan(sin(target_elv) * tan(min_elevation))."""
    assert array_planner.main(['design', '--nmics', '12', '--altitude', '150', '--target-elv', '30']) == 0
    ymics = np.array(capsys.readouterr().out.strip().split(','), dtype=float)
    expected = np.degrees(np.arctan(np.sin(np.radians(30.0)) * np.tan(np.radians(10.0))))
    np.testing.assert_allclose(np.degrees(np.arctan2(150.0, ymics[-1])), expected, rtol=1e-5)
    with pytest.raises(SystemExit):
        array_planner.main(['design', '--help'])
    assert 'measured in the --target-elv plane' in ' '.join(capsys.readouterr().out.split())


def test_overlay_field_choices_come_from_the_level_table(monkeypatch):
    """--field repeated the table's keys by hand, so a field added to the table
    was refused on the command line."""
    fields = dict(array_planner._NPZ_LEVEL_FIELDS, oaspl_lt1khz=('hemisphere_oaspl_lt1khz_db', 'OASPL (<1 kHz), dB'))
    monkeypatch.setattr(array_planner, '_NPZ_LEVEL_FIELDS', fields)
    args = array_planner.build_parser().parse_args(['overlay', '--altitude', '150', '--field', 'oaspl_lt1khz'])
    assert args.field == 'oaspl_lt1khz'
    for field in ('oaspl_fullband', 'oaspl_lt2khz', 'splA_lt2khz', 'third_octave'):
        assert array_planner.build_parser().parse_args(['overlay', '--altitude', '1', '--field', field]).field == field


@pytest.mark.parametrize('command', ['design', 'coverage', 'overlay', 'kml'])
@pytest.mark.parametrize('altitude', ['0', '-1'])
def test_every_level_pass_subcommand_refuses_a_meaningless_altitude(command, altitude, tmp_path, capsys):
    """overlay ran the level pass from a non-positive --altitude with exit 0
    once the overflight check became a per-subcommand default."""
    extra = {
        'coverage': ['-o', str(tmp_path / 'coverage.png')],
        'overlay': ['-o', str(tmp_path / 'overlay.png')],
        'kml': ['--kmz', str(tmp_path / 'array.kmz'), '--ref-lat', '40.8', '--ref-lon', '-77.9',
                '--heading', '0'],
    }.get(command, [])
    with pytest.raises(SystemExit) as raised:
        array_planner.main([command, '--nmics', '4', f'--altitude={altitude}', *extra])
    assert raised.value.code == 2
    assert '--altitude' in capsys.readouterr().err


def _lambert(azimuth_deg, elevation_deg):
    """Lambert axes coordinates of UMAPR directions (deg), flattened."""
    az = np.radians(np.ravel(azimuth_deg))
    el = np.radians(np.ravel(elevation_deg))
    return np.array(array_planner.fa.lambert_ea(el, array_planner.fa.lambert_lon(az)))


@pytest.fixture
def ring_run(monkeypatch):
    """Run array_planner.main; return the Lambert (x, y) of the points it drew
    (None when it drew no figure) and what it would write to a KMZ (or None)."""
    import matplotlib.pyplot as plt
    seen = {}

    def save_or_show(fig, output=None, plot=False, **kwargs):
        seen['points'] = np.array(fig.axes[0].lines[-1].get_data())
        plt.close(fig)

    def write_kml(geodetic, savename, **kwargs):
        seen['kmz'] = (geodetic, savename, kwargs)

    monkeypatch.setattr(array_planner, 'save_or_show', save_or_show)
    monkeypatch.setattr(array_planner.fa, 'write_kml', write_kml)

    def run(argv):
        seen.clear()
        assert array_planner.main(argv) == 0
        return seen.get('points'), seen.get('kmz')
    return run


def test_hover_draws_hover_array_coverage_of_the_rings(ring_run):
    points, kmz = ring_run(['hover', '--radii', '50,100', '--nmics', '4', '--altitudes', '30,60',
                            '--vehicle-headings', '0,90'])
    assert kmz is None
    azimuth, elevation, _ = array_planner.fa.hover_array_coverage(
        [50.0] * 4 + [100.0] * 4, [0.0, 90.0, 180.0, 270.0] * 2, [30.0, 60.0], [0.0, 90.0])
    np.testing.assert_allclose(points, _lambert(azimuth, elevation), atol=1e-12)


def test_hover_heights_are_above_the_microphones(ring_run):
    """With --mic-height h a mic ahead at radius R is atan((altitude - h) / R) below the horizon."""
    points, _ = ring_run(['hover', '--radii', '40', '--mic-azimuths', '0', '--altitudes', '31.2',
                          '--mic-height', '1.2'])
    np.testing.assert_allclose(points, _lambert([180.0], [np.degrees(np.arctan2(30.0, 40.0))]), atol=1e-12)


def test_a_case_file_gives_the_options_and_the_command_line_overrides_it(ring_run, tmp_path):
    case = tmp_path / 'hover.toml'
    case.write_text('radii = [50, 100]\nmic-azimuths = [0, 90]\naltitudes = [30]\nvehicle_headings = [0, 45]\n')
    from_file, _ = ring_run(['hover', '--case', str(case)])
    by_hand, _ = ring_run(['hover', '--radii', '50,100', '--mic-azimuths', '0,90', '--altitudes', '30',
                           '--vehicle-headings', '0,45'])
    np.testing.assert_array_equal(from_file, by_hand)
    overridden, _ = ring_run(['hover', '--case', str(case), '--altitudes', '60'])
    elevation = np.degrees(np.arctan2(60.0, [50.0, 100.0]))
    np.testing.assert_allclose(overridden[:, :2], _lambert([180.0, 270.0], elevation), atol=1e-12)


def test_nmics_on_the_command_line_replaces_the_case_file_mic_azimuths(ring_run, tmp_path):
    """--nmics was ignored when the case file set mic_azimuths."""
    case = tmp_path / 'hover.toml'
    case.write_text('radii = [50]\nmic_azimuths = [0, 90]\naltitudes = [30]\n')
    from_case, _ = ring_run(['hover', '--case', str(case), '--nmics', '4'])
    by_hand, _ = ring_run(['hover', '--radii', '50', '--nmics', '4', '--altitudes', '30'])
    assert by_hand.shape[1] == 4
    np.testing.assert_array_equal(from_case, by_hand)


def test_mic_azimuths_on_the_command_line_replaces_the_case_file_nmics(ring_run, tmp_path):
    case = tmp_path / 'hover.toml'
    case.write_text('radii = [50]\nnmics = 8\naltitudes = [30]\n')
    from_case, _ = ring_run(['hover', '--case', str(case), '--mic-azimuths', '0'])
    np.testing.assert_allclose(from_case, _lambert([180.0], [np.degrees(np.arctan2(30.0, 50.0))]), atol=1e-12)


@pytest.mark.parametrize('command', ['hover', 'takeoff'])
def test_mic_azimuths_and_nmics_together_are_refused_by_name(command, tmp_path, capsys):
    """One of the two was silently ignored."""
    argv = [command, '--radii', '50', '--mic-azimuths', '0,90', '--nmics', '4']
    if command == 'hover':
        argv += ['--altitudes', '30']
    with pytest.raises(SystemExit) as raised:
        array_planner.main(argv)
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert '--mic-azimuths' in err and '--nmics' in err

    case = tmp_path / 'case.toml'
    case.write_text('radii = [50]\nmic_azimuths = [0, 90]\nnmics = 4\n'
                    + ('altitudes = [30]\n' if command == 'hover' else ''))
    with pytest.raises(SystemExit) as raised:
        array_planner.main([command, '--case', str(case)])
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert '--mic-azimuths' in err and '--nmics' in err


def test_an_unknown_case_key_is_refused_by_name(tmp_path, capsys):
    case = tmp_path / 'hover.toml'
    case.write_text('radii = [50]\nnmics = 4\naltitudes = [30]\nflight_path_angle = 8\n')
    with pytest.raises(SystemExit) as raised:
        array_planner.main(['hover', '--case', str(case)])
    assert raised.value.code == 2
    assert 'flight_path_angle' in capsys.readouterr().err


def test_hover_kmz_places_the_ring_about_the_reference_point(ring_run):
    points, (geodetic, savename, kwargs) = ring_run([
        'hover', '--radii', '100', '--mic-azimuths', '0,90', '--altitudes', '30',
        '--ref-lat', '40.8', '--ref-lon', '-77.9', '--heading', '0', '--kmz', 'ring.kmz', '--prefix', 'R'])
    assert points is None  # a KMZ alone draws no figure
    assert savename == 'ring.kmz' and kwargs['channel_prefix'] == 'R'
    # +x is north (heading 0), +y is left of it: west.
    expected = array_planner.fa.array2geodetic(np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]]),
                                               np.array([40.8, -77.9, 0.0]), 0.0, units='m')
    np.testing.assert_allclose(geodetic, expected)
    assert geodetic[0, 0] > 40.8 and geodetic[1, 1] < -77.9


def test_kmz_needs_the_reference_point(capsys):
    with pytest.raises(SystemExit):
        array_planner.main(['hover', '--radii', '100', '--nmics', '4', '--altitudes', '30', '--kmz', 'ring.kmz'])
    assert '--ref-lat' in capsys.readouterr().err


def test_takeoff_draws_takeoff_array_coverage_from_the_origin(ring_run):
    points, _ = ring_run(['takeoff', '--radii', '150', '--nmics', '6', '--origin=-100:20', '--fpa', '6',
                          '--initial-altitude', '5', '--ground-distances', '0:500', '--ground-step', '50',
                          '--vehicle-heading', '30'])
    azimuth, elevation, _ = array_planner.fa.takeoff_array_coverage(
        np.full(6, 150.0), 60.0 * np.arange(6), np.arange(0.0, 501.0, 50.0), initial_altitude=5.0,
        flight_path_angle=6.0, heading=30.0, source_origin_xy=(-100.0, 20.0))
    assert azimuth.shape == (11, 6)
    np.testing.assert_allclose(points, _lambert(azimuth, elevation), atol=1e-12)


@pytest.mark.parametrize('options, named', [
    (['--ground-distances', '0:100'], '--ground-distances'),
    (['--fpa', '90'], '--flight-path-angle'),
    (['--ground-step', '0'], '--ground-step'),
    (['--ground-distances', '100:20'], '--ground-distances'),
])
def test_takeoff_refuses_meaningless_inputs_by_name(options, named, capsys):
    """Ground distance 0 at the default initial altitude 0 puts the source on the ground."""
    with pytest.raises(SystemExit) as raised:
        array_planner.main(['takeoff', '--radii', '150', '--nmics', '6', *options])
    assert raised.value.code == 2
    assert named in capsys.readouterr().err


@pytest.mark.parametrize('options, named', [
    (['--altitudes', '30'], '--radii'),
    (['--radii', '50', '--altitudes', '30'], '--nmics'),
    (['--radii', '50,60', '--mic-azimuths', '0,90,180', '--altitudes', '30'], '--mic-azimuths'),
    (['--radii', '50', '--nmics', '4'], '--altitudes'),
    (['--radii', '50', '--nmics', '4', '--altitudes', '0'], '--altitudes'),
])
def test_hover_refuses_incomplete_rings_by_name(options, named, capsys):
    """Hover and takeoff do not take the level pass's required --altitude."""
    with pytest.raises(SystemExit) as raised:
        array_planner.main(['hover', *options])
    assert raised.value.code == 2
    assert named in capsys.readouterr().err
