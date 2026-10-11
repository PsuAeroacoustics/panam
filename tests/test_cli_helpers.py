import argparse

import pytest

import cli


@pytest.mark.parametrize('text', ['1:2', '"1:2"', "'1:2'", ' 1:2 '])
def test_colon_pair_parses_quoted_or_not(text):
    assert cli.colon_pair(text) == (1.0, 2.0)


@pytest.mark.parametrize('text', ['5', '1:2:3', 'a:b'])
def test_colon_pair_rejects_anything_else(text):
    with pytest.raises(argparse.ArgumentTypeError):
        cli.colon_pair(text)


def test_colon_pair_or_single_takes_one_value_as_a_pair():
    assert cli.colon_pair_or_single('5') == (5.0, 5.0)
    assert cli.colon_pair_or_single('1:2') == (1.0, 2.0)


def _figure():
    from matplotlib.figure import Figure
    fig = Figure()
    fig.subplots().plot([0, 1], [0, 1])
    return fig


def test_save_or_show_creates_the_output_directory(tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(cli.plt, 'show', lambda **kwargs: shown.append(kwargs))
    cli.save_or_show(_figure(), str(tmp_path / 'new' / 'figure.png'))
    assert (tmp_path / 'new' / 'figure.png').stat().st_size > 0 and not shown
    cli.save_or_show(_figure(), str(tmp_path / 'new' / 'figure.png'), plot=True)
    assert len(shown) == 1


def test_save_all_or_show_writes_each_figure_or_shows_them_once(tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(cli.plt, 'show', lambda **kwargs: shown.append(kwargs))
    paths = cli.save_all_or_show({'a': _figure(), 'b': _figure()}, str(tmp_path / 'out'), dpi=50)
    assert [p.rsplit('/', 1)[-1] for p in paths] == ['a.png', 'b.png'] and not shown
    assert all((tmp_path / 'out' / name).stat().st_size > 0 for name in ('a.png', 'b.png'))
    assert cli.save_all_or_show({'c': _figure()}, None) == [] and len(shown) == 1
    cli.save_all_or_show({'d': _figure()}, str(tmp_path / 'out'), plot=True, extension='pdf')
    assert (tmp_path / 'out' / 'd.pdf').is_file() and len(shown) == 2


def test_add_output_options_keeps_aliases_and_frees_p():
    parser = argparse.ArgumentParser()
    cli.add_output_options(parser, 'demo_plots', directory=True, aliases=('--outdir',))
    assert parser.parse_args([]).output_dir == 'demo_plots'
    assert parser.parse_args(['--outdir', 'x', '-p']).output_dir == 'x'
    assert parser.parse_args(['-o', 'y']).plot is False
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pressure')
    cli.add_output_options(parser, 'a.pdf', plot_flags=('--plot',))
    args = parser.parse_args(['-p', '3', '--plot'])
    assert args.pressure == '3' and args.plot and args.output == 'a.pdf'
