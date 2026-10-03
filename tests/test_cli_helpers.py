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
