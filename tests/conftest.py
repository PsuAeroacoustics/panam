"""Use a noninteractive backend for automated plotting tests, and check every
database a test builds against the file contract."""
import functools
import os

import pytest

os.environ['MPLBACKEND'] = 'Agg'


def _validated(build, path_argument):
    """``build``, followed by nod_database.validate_nod on the database it wrote."""
    import nod_database

    @functools.wraps(build)
    def wrapper(*args, **kwargs):
        result = build(*args, **kwargs)
        path = kwargs.get('database_filename', args[path_argument] if len(args) > path_argument else None)
        violations = nod_database.validate_nod(path)
        if kwargs.get('clamp_empty_directions') is False:
            # Asked for: an empty direction keeps the NaN EAA that R6 and C7 rule out.
            violations = [v for v in violations if v.rule not in ('R6', 'C7')]
        assert not violations, 'the database breaks the contract:\n' + '\n'.join(map(str, violations))
        return result
    return wrapper


@pytest.fixture(scope='session', autouse=True)
def every_built_database_keeps_the_contract():
    """Run validate_nod on every database that build_empirical_database or
    build_database_from_norah2 writes during the tests (called through the
    module, ``fa.build_...``, or by a script run in-process)."""
    import flight_acoustics as fa
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fa, 'build_empirical_database', _validated(fa.build_empirical_database, 1))
        patch.setattr(fa, 'build_database_from_norah2', _validated(fa.build_database_from_norah2, 1))
        yield
