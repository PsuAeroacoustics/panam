"""Machine-specific data locations, kept out of the repository.

Scripts and tests find external data (flight-test archives, reference
distributions) by name through :func:`data_path` instead of hard-coding where
it lives on one machine.  A name resolves, first match wins, from

1. the environment variable ``PANAM_<NAME>`` (upper case), e.g.
   ``PANAM_NORAH2=/data/NORAH2_V2.0.74_public``;
2. ``local_paths.toml`` at the repository root, which is git-ignored so each
   machine keeps its own; ``local_paths.example.toml`` lists the names.

Command-line arguments, where a script has one, take precedence over both.

Known names:

``noise_abatement_2017``
    Root of the 2017 FAA/NASA Noise Abatement flight test data (the directory
    holding ``AS350B3/``, ``B407/``, ...).
``as350_demo``
    AS350 demo data used by ``AS350_289108_demo.py`` and ``as350_flip_check.py``
    (the directory holding ``AAM/`` and the run's acoustic files).
``norah2``
    The EASA NORAH2 distribution (the directory holding ``Hemispheres/``).
``niceops``
    The NICE-OPS executable (a file, e.g. ``.../NICEOPS/build/niceops``), which
    the tests use to load the databases PANAM writes.
``niceops_ray_geometry``
    NICE-OPS's ray-geometry executable, for
    :func:`refracted_rays.external_ray_model` and its tests.
"""

from __future__ import annotations

import os
import tomllib
from typing import Optional

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local_paths.toml')


def _from_config(name: str) -> Optional[str]:
    try:
        with open(CONFIG_FILE, 'rb') as handle:
            config = tomllib.load(handle)
    except FileNotFoundError:
        return None
    value = config.get('paths', {}).get(name)
    return None if value in (None, '') else str(value)


def data_path(name: str, override: Optional[str] = None, *, required: bool = True) -> Optional[str]:
    """Absolute path configured for ``name``.

    Args:
        name: One of the names listed in this module's docstring.
        override: A path given explicitly (e.g. on the command line); used as
            is when not None.
        required: If True (default) raise when ``name`` is not configured,
            with a message saying how to configure it; if False return None,
            which is what tests that skip without the data want.
    """
    value = override
    if value is None:
        value = os.environ.get('PANAM_' + name.upper()) or _from_config(name)
    if value is None:
        if not required:
            return None
        raise LookupError(
            f"No path configured for {name!r}. Pass it explicitly, set PANAM_{name.upper()}, "
            f"or add it under [paths] in {CONFIG_FILE} (see local_paths.example.toml).")
    return os.path.abspath(os.path.expanduser(value))
