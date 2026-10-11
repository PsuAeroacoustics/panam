"""Provenance stamps: which code, libraries and settings wrote a file.

:func:`build_provenance` returns the stamp as a dict of text attributes, and
:func:`write_provenance` writes it as global attributes of an open netCDF
file.  The sphere writer (:func:`flight_acoustics.write_aam_hemisphere_netcdf`)
and both database writers (:func:`flight_acoustics.build_empirical_database`,
:func:`flight_acoustics.build_database_from_norah2`) write one.  The attributes
are optional: no reader requires them.  See ``docs/file_formats.md``.

The settings are written as canonical JSON (sorted keys, no whitespace), so
equal settings give the same text and the same SHA-256 whatever order they
were given in.  A stamp must not carry a machine path: a path is reduced to
its file name (see :func:`canonical_settings`).
"""

import dataclasses
import datetime
import hashlib
import importlib.metadata
import json
import ntpath
import os
import platform
import posixpath
import re
import subprocess

import numpy as np

__all__ = ['PROVENANCE_ATTRIBUTES', 'build_provenance', 'canonical_settings', 'code_version',
           'settings_json', 'write_provenance']

#: Every attribute a stamp writes, in the order it writes them.
PROVENANCE_ATTRIBUTES = (
    'panam_tool', 'panam_version', 'panam_created_utc', 'panam_python', 'panam_numpy',
    'panam_scipy', 'panam_netcdf4', 'panam_numba', 'panam_settings', 'panam_settings_sha256',
)

_SOURCE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def code_version():
    """``git describe --always --dirty`` of the PANAM source directory, or 'unknown'.

    'unknown' when git is not installed, the source is not a git checkout, or
    git fails.  A ``-dirty`` suffix marks uncommitted changes.
    """
    try:
        result = subprocess.run(['git', 'describe', '--always', '--dirty'], cwd=_SOURCE_DIRECTORY,
                                capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return 'unknown'
    version = result.stdout.strip()
    return version if result.returncode == 0 and version else 'unknown'


def _package_version(name):
    """The installed version of distribution ``name``, or 'not installed'."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return 'not installed'


def _is_path_text(text):
    """Whether a string is a path that would name a place on this machine:
    absolute (POSIX or Windows) or starting with '~'."""
    return text.startswith('~') or posixpath.isabs(text) or ntpath.isabs(text)


def _basename(text):
    """The last component of a POSIX or Windows path; a trailing separator is ignored."""
    return re.split(r'[\\/]', text.rstrip('/\\'))[-1]


def _callable_name(value):
    """``module.qualname`` of a function, class or callable object, with its
    ``description`` attribute (the repository's label for ray models and hover
    corrections) appended when it has one."""
    target = value if hasattr(value, '__qualname__') else type(value)
    module = getattr(target, '__module__', None)
    name = target.__qualname__ if not module or module == 'builtins' else module + '.' + target.__qualname__
    description = getattr(value, 'description', None)
    if isinstance(description, str):
        name += ': ' + canonical_settings(description)
    return name


def canonical_settings(value):
    """``value`` reduced to plain JSON types, the same for equal settings.

    - dicts: keys as text (JSON sorts them when written);
    - lists, tuples and sets: lists (a set sorted by its JSON text);
    - numpy arrays and scalars: lists and Python numbers;
    - ``os.PathLike`` values, and strings that are absolute paths or start
      with '~': the file name only, so no machine path is recorded;
    - callables: their qualified name (:func:`_callable_name`);
    - dataclasses and other objects with attributes (an ``Atmosphere``):
      ``{'type': name, ...public attributes}``;
    - anything else: its ``str``.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return canonical_settings(value.item())
    if isinstance(value, np.ndarray):
        return canonical_settings(value.tolist())
    if isinstance(value, os.PathLike):
        return _basename(os.fspath(value))
    if isinstance(value, str):
        return _basename(value) if _is_path_text(value) else value
    if isinstance(value, dict):
        return {str(key): canonical_settings(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical_settings(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [canonical_settings(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if callable(value):
        return _callable_name(value)
    if dataclasses.is_dataclass(value):
        fields = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    elif hasattr(value, '__dict__'):
        fields = {key: item for key, item in vars(value).items() if not key.startswith('_')}
    else:
        return str(value)
    return dict(canonical_settings(fields), type=type(value).__qualname__)


def settings_json(settings):
    """Canonical JSON text of ``settings`` (:func:`canonical_settings`), keys sorted,
    no whitespace, ASCII only (a NETCDF3_CLASSIC attribute takes nothing else)."""
    return json.dumps(canonical_settings(settings), sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True)


def build_provenance(tool, settings):
    """The provenance stamp of a file written by ``tool`` with ``settings``.

    Args:
        tool: name of the function or script that writes the file.
        settings: dict of the settings the file was built with, in any units
            (they are recorded, not interpreted).

    Returns:
        dict of text, keyed by the attribute names of
        :data:`PROVENANCE_ATTRIBUTES`: the tool, the code version
        (:func:`code_version`), the creation time (UTC, ISO 8601, whole
        seconds), the Python, numpy, scipy, netCDF4 and numba versions, the
        canonical settings JSON (:func:`settings_json`) and its SHA-256 (hex).
    """
    text = settings_json(settings)
    return {
        'panam_tool': str(tool),
        'panam_version': code_version(),
        'panam_created_utc': datetime.datetime.now(datetime.timezone.utc).replace(
            microsecond=0).isoformat().replace('+00:00', 'Z'),
        'panam_python': platform.python_version(),
        'panam_numpy': _package_version('numpy'),
        'panam_scipy': _package_version('scipy'),
        'panam_netcdf4': _package_version('netCDF4'),
        'panam_numba': _package_version('numba'),
        'panam_settings': text,
        'panam_settings_sha256': hashlib.sha256(text.encode('ascii')).hexdigest(),
    }


def write_provenance(dataset, provenance):
    """Write a stamp from :func:`build_provenance` as global text attributes of
    the open netCDF ``dataset``.  None writes nothing."""
    if provenance is None:
        return
    for name in PROVENANCE_ATTRIBUTES:
        if name in provenance:
            dataset.setncattr(name, str(provenance[name]))
