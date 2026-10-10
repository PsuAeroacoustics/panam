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
