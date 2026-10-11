"""pyproject.toml, requirements.txt and the CI workflow agree."""

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _requirements():
    lines = (line.split('#', 1)[0].strip() for line in (REPO / 'requirements.txt').read_text().splitlines())
    return [line for line in lines if line]


def test_pyproject_dependencies_mirror_requirements():
    project = tomllib.loads((REPO / 'pyproject.toml').read_text())['project']
    runtime = [name for name in _requirements() if name != 'pytest']
    assert project['dependencies'] == runtime
    assert 'pytest' in project['optional-dependencies']['test']


def test_pyproject_builds_nothing_and_leaves_pytest_to_pytest_ini():
    pyproject = tomllib.loads((REPO / 'pyproject.toml').read_text())
    assert pyproject['project']['requires-python'] == '>=3.11'
    assert 'build-system' not in pyproject and 'scripts' not in pyproject['project']
    assert 'pytest' not in pyproject.get('tool', {})
    assert (REPO / 'pytest.ini').is_file()


def test_ci_tests_the_supported_pythons_on_linux_and_macos():
    workflow = (REPO / '.github' / 'workflows' / 'tests.yml').read_text()
    assert re.search(r'python-version: \["3\.11", "3\.14"\]', workflow)
    assert 'macos-latest' in workflow
    assert '--cov' in workflow and '--durations=20' in workflow and 'NUMBA_CACHE_DIR' in workflow
