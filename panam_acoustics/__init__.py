"""PANAM's support package: signal loaders, filters, absorption and plot style.

``signal_io`` loads HDF5, netCDF and UFF recordings; ``filters`` holds the
Butterworth low/high-pass helpers and the one-third-octave filter bank;
``atmosphere`` and ``iso_9613_1_1993`` give ISO 9613-1 absorption; ``plotting``
holds the scoped figure style.  ``atmosphere.py``, ``iso_9613_1_1993.py`` and
the low/high-pass helpers in ``filters.py`` derive from python-acoustics
(BSD-3-Clause; see THIRD_PARTY_NOTICES.md).  The rest is PANAM's own code.
"""

from .atmosphere import Atmosphere
from .filters import highpass, lowpass

__all__ = [
    "Atmosphere",
    "highpass",
    "lowpass",
]
