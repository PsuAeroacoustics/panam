"""Build recipes: the settings of a sphere build and of a database build in one TOML file.

A recipe is a TOML file with up to three sections, each optional:

``[spheres]``
    Keywords of :func:`noise_abatement_2017.build_all` and
    :func:`noise_abatement_2017.build_sphere`: how every run is built.  The subtable
    ``[spheres.interpolation]`` is ``build_sphere``'s ``interpolation``, the adaptive
    gridding's settings (keys of :data:`flight_acoustics.ADAPTIVE_INTERPOLATION`); an
    empty one selects the adaptive gridding with its defaults.
``[hover_source]``
    The same keywords, overriding ``[spheres]`` for the rebuild a database synthesizes its
    hover from (typically ``remove_doppler = true``).
``[database]``
    Keywords of :func:`flight_acoustics.build_empirical_database`.

Keys are checked against the signatures of those functions (:func:`inspect.signature`),
and a key that is not one of their keywords is refused with the nearest names.  Units are
the functions' own.  A few keywords take a recipe form, because TOML holds no Python
objects or None:

- ``atmosphere`` is a table of ``temperature`` (K), ``pressure`` (kPa) and
  ``relative_humidity`` (percent);
- ``hover_correction`` is a ``"module:callable"`` string, imported from the recipe's own
  directory or from ``sys.path``;
- ``winds`` is the path of a CSV file :func:`noise_abatement_2017.read_winds` reads;
- ray models come only from a directory of per-run atmospheric profiles,
  ``ray_profiles`` (:func:`noise_abatement_2017.profile_ray_models`);
- ``fixed_load_factor = true`` in ``[database]`` is ``load_factors=None``;
- ``tone_split = true`` sets ``tone_aware = true`` too, as ``--tone-split`` does, in its
  own section, so ``[hover_source] tone_split = false`` keeps the tone-aware banding
  ``[spheres] tone_split = true`` set; a recipe that gives ``tone_split = true`` with
  ``tone_aware = false`` is refused.

The dataset (``aircraft``), outputs (``output_directory``, ``manifest_path``,
``norah2_directory``, ``samples_path``, the database file), per-run arguments and
Python-only objects (``ground``, a ray model callable) are refused: they belong to the
caller.  An omitted key keeps the function's default, from Python and on the command
line alike; a recipe cannot set a keyword back to None.

Path values (``root``, ``reference_directory``, ``plate_table_directory``,
``ray_profiles``, ``winds``, ``hover_source``) may name a :mod:`local_paths` entry as
``${name}``, e.g. ``"${b407_legacy_spheres}"``.  After ``~`` is expanded, a relative path
without a ``${name}`` is relative to the recipe's own directory, the directory a
``hover_correction`` module is imported from.  Names are looked up only when a section's
options are formed, so a ``[database]`` build does not need the sphere build's data.

On the command line (``--recipe FILE.toml`` of ``noise_abatement_2017.py`` and
``build_empirical_database.py``) an option given explicitly overrides the recipe, and the
recipe overrides the command line's defaults.  A keyword neither in the recipe nor on the
command line takes the function's default; of the sphere build's options only
``--point-stride`` has another default (10, the function's 1).  A ``--build-temperature``,
``--build-pressure`` or ``--build-humidity`` option replaces only its own field of a
recipe's ``atmosphere``.
"""

import argparse
import difflib
import importlib
import inspect
import os
import re
import sys
import tomllib
from dataclasses import dataclass, field

import flight_acoustics as fa
import local_paths
import noise_abatement_2017 as na

SECTIONS = ('spheres', 'hover_source', 'database')

_OUTPUT = 'an output: give it on the command line or in the call, not in a recipe'

#: Keywords of the sphere build's functions a recipe refuses, with the reason.
SPHERE_REFUSED = dict(
    aircraft='the dataset: give it on the command line or in the call',
    output_directory=_OUTPUT, manifest_path=_OUTPUT, norah2_directory=_OUTPUT, samples_path=_OUTPUT,
    output_path=_OUTPUT, test='a per-run argument of build_sphere', run='a per-run argument of build_sphere',
    reference_sphere='a per-run argument: give reference_directory',
    ray_model='ray models come only from a profile directory: give ray_profiles',
    ray_models='ray models come only from a profile directory: give ray_profiles',
    wind='a per-run argument: give winds, the path of a winds CSV',
    ground='a Python object with no recipe form: call build_sphere from Python')

#: Keywords of build_empirical_database a recipe refuses, with the reason.
DATABASE_REFUSED = dict(directory_name='an input: give it on the command line or in the call',
                        database_filename=_OUTPUT)

#: Keys that are paths, resolved through local_paths names.
PATH_KEYS = ('root', 'reference_directory', 'plate_table_directory', 'ray_profiles', 'winds',
             'hover_source')

#: Recipe keys that are not keywords of the functions, per section.
EXTRA_KEYS = dict(spheres=(), hover_source=(), database=('fixed_load_factor',))

#: Per kind of section: the functions it feeds (a later one's defaults win), the
#: keys it refuses and how an error names the functions.
_SPHERE_SECTION = ((na.build_sphere, na.build_all), SPHERE_REFUSED,
                   'noise_abatement_2017.build_all or build_sphere')
_DATABASE_SECTION = ((fa.build_empirical_database,), DATABASE_REFUSED,
                     'flight_acoustics.build_empirical_database')


def _section(section):
    """(functions, refused, target) of ``section``: [database] or a sphere section."""
    return _DATABASE_SECTION if section == 'database' else _SPHERE_SECTION


def _keywords(*functions):
    return {name for function in functions
            for name, parameter in inspect.signature(function).parameters.items()
            if parameter.kind not in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL)}


def function_defaults(section):
    """The defaults of the keywords a recipe section feeds, from the function signatures."""
    return {name: parameter.default for function in _section(section)[0]
            for name, parameter in inspect.signature(function).parameters.items()
            if parameter.default is not inspect.Parameter.empty}


def allowed_keys(section):
    """The keys a recipe section takes, from the signatures of the functions it feeds."""
    functions, refused, _ = _section(section)
    return (_keywords(*functions) - set(refused)) | set(EXTRA_KEYS[section])


def _check_section(path, section, values):
    if not isinstance(values, dict):
        raise ValueError('{}: [{}] must be a table'.format(path, section))
    _, refused, targets = _section(section)
    allowed = allowed_keys(section)
    for key, value in values.items():
        where = '{}: [{}] {}'.format(path, section, key)
        if key in refused:
            raise ValueError('{} is refused: {}'.format(where, refused[key]))
        if key not in allowed:
            close = difflib.get_close_matches(key, sorted(allowed), n=3)
            raise ValueError('{}: unknown key, not a keyword of {}{}'.format(
                where, targets, '; did you mean {}?'.format(', '.join(close)) if close else ''))
        if key == 'interpolation':
            if not isinstance(value, dict):
                raise ValueError('{} must be a table ([{}.interpolation])'.format(where, section))
            unknown = sorted(set(value) - set(fa.ADAPTIVE_INTERPOLATION))
            if unknown:
                raise ValueError('{}: unknown settings {}; known: {}'.format(
                    where, unknown, ', '.join(sorted(fa.ADAPTIVE_INTERPOLATION))))
        elif key == 'atmosphere':
            if not isinstance(value, dict) or not set(value) <= {'temperature', 'pressure', 'relative_humidity'}:
                raise ValueError('{} must be a table of temperature (K), pressure (kPa) and '
                                 'relative_humidity (percent)'.format(where))
        elif key == 'hover_correction':
            if not isinstance(value, str) or value.count(':') != 1:
                raise ValueError("{} must be a 'module:callable' string".format(where))
        elif key in PATH_KEYS and not isinstance(value, str):
            raise ValueError('{} must be a path string'.format(where))
    if section == 'database' and 'fixed_load_factor' in values:
        if not isinstance(values['fixed_load_factor'], bool):
            raise ValueError('{}: [database] fixed_load_factor must be true or false'.format(path))
        if values['fixed_load_factor'] and 'load_factors' in values:
            raise ValueError('{}: [database] give load_factors or fixed_load_factor, not both'.format(path))


def resolve_path(value, directory=None):
    """A recipe path: ``${name}`` replaced by the :mod:`local_paths` entry, made absolute.

    A relative path without a ``${name}`` is taken relative to ``directory`` (the
    recipe's), or to the current directory when ``directory`` is None.
    """
    named = re.search(r'\$\{[A-Za-z0-9_]+\}', value) is not None
    expanded = os.path.expanduser(
        re.sub(r'\$\{([A-Za-z0-9_]+)\}', lambda match: local_paths.data_path(match.group(1)), value))
    if directory is not None and not named:
        expanded = os.path.join(directory, expanded)
    return os.path.abspath(expanded)


def resolve_callable(text, directory=None):
    """The object a ``"module:callable"`` string names (attributes may be dotted).

    The module is imported with ``directory`` (the recipe's) first on ``sys.path``.
    """
    module_name, _, attribute = text.partition(':')
    if not module_name or not attribute:
        raise ValueError("expected 'module:callable', got {!r}".format(text))
    added = directory is not None and directory not in sys.path
    if added:
        sys.path.insert(0, directory)
    try:
        value = importlib.import_module(module_name)
    finally:
        if added:
            sys.path.remove(directory)
    for part in attribute.split('.'):
        value = getattr(value, part)
    if not callable(value):
        raise ValueError('{} is not callable'.format(text))
    return value


def _tone_split_implies_tone_aware(section):
    """A copy of one recipe section with ``tone_aware = true`` added where its
    ``tone_split = true`` implies it, as ``--tone-split`` does.  Applied to each
    section on its own, so that ``[hover_source] tone_split = false`` turns off the
    part spheres only, not the tone-aware banding ``[spheres] tone_split`` set."""
    values = dict(section)
    if values.get('tone_split') and 'tone_aware' not in values:
        values['tone_aware'] = True
    return values


@dataclass
class Recipe:
    """A loaded, checked recipe: its sections as read (paths unresolved)."""
    path: str
    spheres: dict = field(default_factory=dict)
    hover_source: dict = field(default_factory=dict)
    database: dict = field(default_factory=dict)

    @property
    def directory(self):
        """The recipe's directory: relative paths and hover_correction modules start here."""
        return os.path.dirname(os.path.abspath(self.path))

    def _convert(self, values):
        options = {}
        for key, value in values.items():
            if key in PATH_KEYS:
                value = resolve_path(value, self.directory)
            if key == 'atmosphere':
                value = fa.Atmosphere(**value)
            elif key == 'winds':
                value = na.read_winds(value)
            elif key == 'hover_correction':
                value = resolve_callable(value, self.directory)
            elif key == 'interpolation':
                value = dict(value)
            options[key] = value
        return options

    def sphere_options(self, hover_source=False):
        """Keywords of :func:`noise_abatement_2017.build_all`: ``[spheres]``, with
        ``[hover_source]`` over it when ``hover_source``."""
        values = _tone_split_implies_tone_aware(self.spheres)
        if hover_source:
            values.update(_tone_split_implies_tone_aware(self.hover_source))
        return self._convert(values)

    def database_options(self):
        """Keywords of :func:`flight_acoustics.build_empirical_database`: ``[database]``."""
        values = dict(self.database)
        if values.pop('fixed_load_factor', False):
            values['load_factors'] = None
        return self._convert(values)


def load_recipe(path):
    """Read and check a recipe (see the module docstring); returns a :class:`Recipe`.

    Raises ValueError, naming the file, section and key, for an unknown section or key, a
    refused key or a value of the wrong form.
    """
    with open(path, 'rb') as handle:
        try:
            content = tomllib.load(handle)
        except tomllib.TOMLDecodeError as error:
            raise ValueError('{}: {}'.format(path, error)) from None
    unknown = sorted(set(content) - set(SECTIONS))
    if unknown:
        raise ValueError('{}: unknown sections {}; a recipe has [{}]'.format(
            path, unknown, '], ['.join(SECTIONS)))
    for section in SECTIONS:
        _check_section(path, section, content.get(section, {}))
    spheres = content.get('spheres', {})
    for section, values in (('spheres', spheres),
                            ('hover_source', {**spheres, **content.get('hover_source', {})})):
        if values.get('tone_split') and values.get('tone_aware') is False:
            raise ValueError('{}: [{}] tone_split = true needs tone_aware = true (or no tone_aware: '
                             'tone_split sets it)'.format(path, section))
    return Recipe(path=str(path), **{section: content.get(section, {}) for section in SECTIONS})


def explicit_destinations(parser, argv=None):
    """The destinations of ``parser`` that ``argv`` sets explicitly.

    It parses ``argv`` again with every default set to ``argparse.SUPPRESS``, so that only
    what the command line gives lands in the namespace.
    """
    saved = [(action, action.default) for action in parser._actions]
    try:
        for action, _ in saved:
            action.default = argparse.SUPPRESS
        namespace = parser.parse_intermixed_args(argv)
    finally:
        for action, default in saved:
            action.default = default
    return set(vars(namespace))


def explicit_keywords(parser, argv, destination_keywords):
    """The function keywords the command line sets explicitly.

    ``destination_keywords`` maps a destination to the keywords it sets, where they are
    not the destination itself.
    """
    keywords = set()
    for destination in explicit_destinations(parser, argv):
        keywords.update(destination_keywords.get(destination, (destination,)))
    return keywords


def apply(parser, argv, recipe_options, cli_options, destination_keywords):
    """``cli_options`` with ``recipe_options`` for every keyword ``argv`` does not set.

    Returns ``(merged, explicit)``: the :func:`merge` of the two and the keywords the
    command line set (:func:`explicit_keywords`).
    """
    explicit = explicit_keywords(parser, argv, destination_keywords)
    return merge(recipe_options, cli_options, explicit), explicit


def merge(recipe_options, cli_options, explicit):
    """``cli_options`` with the recipe's values in place of every keyword not in ``explicit``."""
    merged = dict(cli_options)
    merged.update({key: value for key, value in recipe_options.items() if key not in explicit})
    return merged
