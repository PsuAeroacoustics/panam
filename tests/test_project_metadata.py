"""pyproject.toml, requirements.txt and the CI workflow agree."""

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _requirements():
    lines = (line.split('#', 1)[0].strip() for line in (REPO / 'requirements.txt').read_text().splitlines())
    return [line for line in lines if line]


def _pyproject():
    return tomllib.loads((REPO / 'pyproject.toml').read_text())


def _workflow():
    return (REPO / '.github' / 'workflows' / 'tests.yml').read_text()


def test_pyproject_dependencies_mirror_requirements():
    project = _pyproject()['project']
    assert project['dependencies'] == [name for name in _requirements() if name != 'pytest']
    assert 'pytest' in project['optional-dependencies']['test']


def test_requirements_name_packages_only():
    """A `-e .` there installed the current directory, so a project installing PANAM's
    dependencies with `-r path/to/panam/requirements.txt` from its own root failed."""
    assert all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', line) for line in _requirements())


def test_ci_installs_panam_in_editable_mode():
    assert 'pip install -e .' in _workflow()


def test_pyproject_installs_only_the_panam_package_and_leaves_pytest_to_pytest_ini():
    pyproject = _pyproject()
    assert pyproject['project']['requires-python'] == '>=3.11'
    assert pyproject['tool']['setuptools']['packages']['find']['include'] == ['panam', 'panam.*']
    assert 'scripts' not in pyproject['project']
    assert 'pytest' not in pyproject['tool']
    assert (REPO / 'pytest.ini').is_file()


def test_the_package_data_is_inside_the_package():
    for pattern in _pyproject()['tool']['setuptools']['package-data']['panam']:
        assert list((REPO / 'panam').glob(pattern)), pattern


def test_ci_tests_the_supported_pythons_on_linux_and_macos():
    workflow = _workflow()
    assert re.search(r'python-version: \["3\.11", "3\.14"\]', workflow)
    assert 'macos-latest' in workflow
    assert '--cov' in workflow and '--durations=20' in workflow and 'NUMBA_CACHE_DIR' in workflow


def test_ci_pins_and_hashes_every_numba_cached_source():
    """The cache step names the modules with Numba-cached functions by path; a moved
    file or a new one left out would leave the cache keyed on the wrong sources."""
    cached = sorted(str(path.relative_to(REPO)) for path in (REPO / 'panam').rglob('*.py')
                    if re.search(r'@(numba\.)?njit\([^)]*cache=True', path.read_text()))
    workflow = _workflow()
    touched = re.search(r'touch -t \d+ (.+)', workflow).group(1).split()
    hashed = re.findall(r"'([^']+\.py)'", re.search(r'hashFiles\(([^)]*)\)', workflow).group(1))
    assert cached and sorted(touched) == sorted(hashed) == cached
