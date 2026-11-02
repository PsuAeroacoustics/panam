"""The NICEOPS executable PANAM's tests run is the pinned one.

tests/niceops_pin.txt holds the NICEOPS commit these tests were last checked
against; a configured niceops reporting another commit (--version) is not run,
so that a footprint or self-test result is never read against a NICEOPS that
does not read what PANAM now writes.  PANAM_NICEOPS_ANY_VERSION=1 runs it anyway.
"""
import os
import re
import stat

import sphere_helpers


def _fake_niceops(tmp_path, version):
    path = tmp_path / 'niceops'
    path.write_text('#!/bin/sh\necho {}\n'.format(version))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_the_pin_is_a_commit():
    assert re.fullmatch(r'[0-9a-f]{7,40}', sphere_helpers.NICEOPS_PIN)


def test_another_commit_is_not_run(tmp_path, monkeypatch):
    monkeypatch.setenv('PANAM_NICEOPS', str(_fake_niceops(tmp_path, '0000000')))
    monkeypatch.delenv('PANAM_NICEOPS_ANY_VERSION', raising=False)
    path, reason = sphere_helpers.niceops()
    assert path is None and sphere_helpers.NICEOPS_PIN in reason and '0000000' in reason


def test_the_pinned_commit_runs_and_the_override_runs_any(tmp_path, monkeypatch):
    monkeypatch.delenv('PANAM_NICEOPS_ANY_VERSION', raising=False)
    pinned = _fake_niceops(tmp_path, sphere_helpers.NICEOPS_PIN + '-dirty')
    monkeypatch.setenv('PANAM_NICEOPS', str(pinned))
    assert sphere_helpers.niceops()[0] == str(pinned)
    other = tmp_path / 'other'
    other.mkdir()
    monkeypatch.setenv('PANAM_NICEOPS', str(_fake_niceops(other, 'abcdef0')))
    monkeypatch.setenv('PANAM_NICEOPS_ANY_VERSION', '1')
    assert sphere_helpers.niceops()[0] is not None


def test_no_niceops_says_so(monkeypatch):
    monkeypatch.setenv('PANAM_NICEOPS', os.devnull + '-missing')
    monkeypatch.setattr(sphere_helpers, 'executable', lambda name: None)
    path, reason = sphere_helpers.niceops()
    assert path is None and 'not configured' in reason
