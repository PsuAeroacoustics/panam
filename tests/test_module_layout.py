"""flight_acoustics is a namespace over the topic modules; library modules do not import it.

The namespace imports every topic module, so a library module that imports it closes an
import cycle as soon as a topic module imports that library module.
"""

import ast
from pathlib import Path

PANAM = Path(__file__).resolve().parents[1] / 'panam'

#: Library modules that still reach the library through the namespace (``fa.<name>``),
#: from before it was split.  Remove a module from here when it imports the topic modules.
STILL_USE_THE_NAMESPACE = {'build_recipe', 'ground_plane', 'noise_abatement_2017', 'synthesis',
                           'synthetic_flyover'}


def _imported_modules(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # ``from . import flight_acoustics`` has no module, only the name.
            prefix = ''
            if node.module:
                yield node.module
                prefix = node.module + '.'
            yield from (prefix + alias.name for alias in node.names)


def _imports_the_namespace(path):
    return bool({'panam.flight_acoustics', 'flight_acoustics'} & set(_imported_modules(path)))


def test_library_modules_import_the_topic_modules_not_the_namespace():
    offenders = []
    for path in sorted(PANAM.rglob('*.py')):
        name = path.relative_to(PANAM).with_suffix('').as_posix()
        if name == 'flight_acoustics' or name in STILL_USE_THE_NAMESPACE:
            continue
        if _imports_the_namespace(path):
            offenders.append(name)
    assert offenders == []


def test_the_list_of_modules_still_using_the_namespace_is_current():
    stale = [name for name in sorted(STILL_USE_THE_NAMESPACE)
             if not _imports_the_namespace(PANAM / (name + '.py'))]
    assert stale == [], 'no longer import flight_acoustics: remove from STILL_USE_THE_NAMESPACE'
