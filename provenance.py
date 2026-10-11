"""Provenance stamps: which code, libraries and settings wrote a file.

:func:`build_provenance` returns the stamp as a dict of text attributes, and
:func:`write_provenance` writes it as global attributes of an open netCDF
file.  The sphere writer (:func:`flight_acoustics.write_aam_hemisphere_netcdf`)
and both database writers (:func:`flight_acoustics.build_empirical_database`,
:func:`flight_acoustics.build_database_from_norah2`) write one.  The attributes
are optional: no reader requires them.  See ``docs/file_formats.md``.

The settings are written as canonical JSON (sorted keys, no whitespace), so
equal settings give the same text and the same SHA-256 whatever order they
were given in.  The file's identity (its own name, the run, the input files)
is kept out of the settings and written separately as ``panam_inputs``, so
that files built with the same processing settings share the hash.  A stamp
must not carry a machine path: a path is reduced to its file name (see
:func:`canonical_settings`).

A stamp can also carry an input signature (:func:`file_signature`): a hash of
the names, sizes and modification times of the files a build read, which
``noise_abatement_2017.build_all(skip_unchanged=True)`` compares, and a build
record (:func:`record_json`): what ``noise_abatement_2017.build_sphere`` returned,
from which ``build_all`` rebuilds the manifest row of a run it skips.
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

__all__ = ['BUILD_RECORD_ATTRIBUTE', 'INPUT_SIGNATURE_ATTRIBUTE', 'PROVENANCE_ATTRIBUTES',
           'build_provenance', 'canonical_settings', 'code_version', 'file_signature',
           'read_record_json', 'record_json', 'settings_json', 'settings_sha256',
           'write_provenance']

#: Every attribute a stamp writes, in the order it writes them.
PROVENANCE_ATTRIBUTES = (
    'panam_tool', 'panam_version', 'panam_created_utc', 'panam_python', 'panam_numpy',
    'panam_scipy', 'panam_netcdf4', 'panam_numba', 'panam_settings', 'panam_settings_sha256',
    'panam_inputs',
)

#: The optional attribute that holds a stamp's input signature (:func:`file_signature`),
#: written after :data:`PROVENANCE_ATTRIBUTES` when the writer gives one.
INPUT_SIGNATURE_ATTRIBUTE = 'panam_input_signature'

#: The optional attribute that holds a stamp's build record (:func:`record_json`),
#: written after the input signature when the writer gives one.
BUILD_RECORD_ATTRIBUTE = 'panam_build_record'

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

    - non-finite floats: the text 'nan', 'inf' or '-inf' (JSON has no
      number for them);
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
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if np.isnan(value):
            return 'nan'
        return value if np.isfinite(value) else ('inf' if value > 0 else '-inf')
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
    no whitespace, ASCII only (a NETCDF3_CLASSIC attribute takes nothing else).
    Strict JSON: a non-finite number is written as text, never as ``NaN``."""
    return json.dumps(canonical_settings(settings), sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False)


def settings_sha256(settings):
    """SHA-256 (hex) of :func:`settings_json` of ``settings``: a stamp's
    ``panam_settings_sha256``, computed without building the stamp."""
    return hashlib.sha256(settings_json(settings).encode('ascii')).hexdigest()


def file_signature(paths):
    """SHA-256 (hex) of the file name, size (bytes) and modification time (ns) of each of ``paths``.

    A cheap test of whether a build's input files changed: it reads no file
    content, so it does not pull a cloud-storage placeholder down.  The paths
    are sorted, and each is recorded by its file name only, so the signature
    names no directory.  A missing file is recorded as missing.  Copying the
    files elsewhere keeps the signature only if the copy keeps their times.
    """
    entries = []
    for path in sorted(set(os.fspath(path) for path in paths)):
        try:
            status = os.stat(path)
            entries.append([_basename(path), status.st_size, status.st_mtime_ns])
        except OSError:
            entries.append([_basename(path), None, None])
    return hashlib.sha256(json.dumps(entries, separators=(',', ':')).encode('utf-8')).hexdigest()


def _record_value(value):
    """One value of a build record as JSON that :func:`_restore_value` turns back into it."""
    if value is None or isinstance(value, (bool, int, str)):
        return _basename(value) if isinstance(value, str) and _is_path_text(value) else value
    if isinstance(value, np.generic):
        return _record_value(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else {'float': repr(value)}
    if isinstance(value, tuple):
        return {'tuple': [_record_value(item) for item in value]}
    if isinstance(value, (list, np.ndarray)):
        return [_record_value(item) for item in list(value)]
    if isinstance(value, dict):
        return {'dict': {str(key): _record_value(item) for key, item in value.items()}}
    raise TypeError('a build record holds plain values, not {}'.format(type(value).__name__))


def _restore_value(value):
    """The value :func:`_record_value` encoded."""
    if isinstance(value, list):
        return [_restore_value(item) for item in value]
    if isinstance(value, dict):
        (kind, item), = value.items()
        if kind == 'float':
            return float(item)
        if kind == 'tuple':
            return tuple(_restore_value(entry) for entry in item)
        return {key: _restore_value(entry) for key, entry in item.items()}
    return value


def record_json(record):
    """A build record (a dict of plain values) as JSON text that :func:`read_record_json`
    turns back into an equal dict.

    Unlike :func:`settings_json`, nothing is reduced: floats keep every digit (a
    non-finite one is tagged ``{"float": "nan"}``), tuples and dicts are tagged so
    they come back as tuples and dicts, numpy scalars become Python numbers.  A
    string that is a machine path (absolute or starting with '~') is reduced to
    its file name, since a stamp names no directory; leave the paths out.  Keys
    sorted, no whitespace, ASCII only, strict JSON.
    """
    return json.dumps({str(key): _record_value(value) for key, value in record.items()},
                      sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def read_record_json(text):
    """The dict :func:`record_json` wrote as ``text``."""
    return {key: _restore_value(value) for key, value in json.loads(text).items()}


def build_provenance(tool, settings, inputs=None, input_signature=None, build_record=None):
    """The provenance stamp of a file written by ``tool`` with ``settings``.

    Args:
        tool: name of the function or script that writes the file.
        settings: dict of the processing settings the file was built with, in
            any units (they are recorded, not interpreted).  Leave out what
            names the file or its inputs, so that equal settings give an equal
            hash across files.
        inputs: dict of what identifies this file: its own name, the run, the
            input files.  Recorded as canonical JSON in ``panam_inputs`` (paths
            reduced to file names), not hashed.  None records ``{}``.
        input_signature: text that changes when the input files change, such as
            :func:`file_signature` of them, recorded as
            :data:`INPUT_SIGNATURE_ATTRIBUTE`.  None (the default) records none.
        build_record: dict of plain values describing the build, recorded as
            :func:`record_json` in :data:`BUILD_RECORD_ATTRIBUTE`.  None (the
            default) records none.

    Returns:
        dict of text, keyed by the attribute names of
        :data:`PROVENANCE_ATTRIBUTES`: the tool, the code version
        (:func:`code_version`), the creation time (UTC, ISO 8601, whole
        seconds), the Python, numpy, scipy, netCDF4 and numba versions, the
        canonical settings JSON (:func:`settings_json`), its SHA-256 (hex)
        and the canonical JSON of ``inputs``; and, when given, the input signature
        and the build record.
    """
    text = settings_json(settings)
    stamp = {
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
        'panam_inputs': settings_json({} if inputs is None else inputs),
    }
    if input_signature is not None:
        stamp[INPUT_SIGNATURE_ATTRIBUTE] = str(input_signature)
    if build_record is not None:
        stamp[BUILD_RECORD_ATTRIBUTE] = record_json(build_record)
    return stamp


def write_provenance(dataset, provenance):
    """Write a stamp from :func:`build_provenance` as global text attributes of
    the open netCDF ``dataset``.  None writes nothing."""
    if provenance is None:
        return
    for name in PROVENANCE_ATTRIBUTES + (INPUT_SIGNATURE_ATTRIBUTE, BUILD_RECORD_ATTRIBUTE):
        if name in provenance:
            dataset.setncattr(name, str(provenance[name]))
