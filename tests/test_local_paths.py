import os

import pytest

import local_paths


@pytest.fixture(autouse=True)
def no_panam_environment(monkeypatch):
    """The tests set PANAM_* themselves; a developer's own must not leak in."""
    for name in list(os.environ):
        if name.startswith('PANAM_'):
            monkeypatch.delenv(name)


@pytest.fixture
def config(tmp_path, monkeypatch):
    path = tmp_path / 'local_paths.toml'
    path.write_text('[paths]\nnorah2 = "~/from_config"\nempty = ""\n')
    monkeypatch.setattr(local_paths, 'CONFIG_FILE', str(path))
    return path


def test_config_file_is_read_and_expanded(config):
    assert local_paths.data_path('norah2') == os.path.expanduser('~/from_config')


def test_environment_beats_config_and_argument_beats_both(config, monkeypatch, tmp_path):
    monkeypatch.setenv('PANAM_NORAH2', str(tmp_path / 'from_env'))
    assert local_paths.data_path('norah2') == str(tmp_path / 'from_env')
    assert local_paths.data_path('norah2', str(tmp_path / 'explicit')) == str(tmp_path / 'explicit')


def test_missing_name_raises_or_returns_none(config):
    with pytest.raises(LookupError, match='PANAM_AS350_DEMO'):
        local_paths.data_path('as350_demo')
    assert local_paths.data_path('as350_demo', required=False) is None
    assert local_paths.data_path('empty', required=False) is None


def test_no_config_file_at_all(tmp_path, monkeypatch):
    monkeypatch.setattr(local_paths, 'CONFIG_FILE', str(tmp_path / 'absent.toml'))
    assert local_paths.data_path('norah2', required=False) is None
