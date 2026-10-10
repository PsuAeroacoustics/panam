# Third-Party Notices

This directory includes code derived from the python-acoustics project:
https://github.com/python-acoustics/python-acoustics

The derived files are:

- `atmosphere.py`: the `Atmosphere` class (ISO 9613-1 absorption).
- `iso_9613_1_1993.py`: the ISO 9613-1:1993 formulas and constants.
- `filters.py`: the Butterworth `lowpass` and `highpass` helpers, whose
  behavior is ported from python-acoustics and reimplemented with SciPy's SOS
  filters. `tests/data/python_acoustics_reference.npz` holds the
  python-acoustics output they are tested against.

The rest of this package is PANAM's own code under the repository's MIT
license: `signal_io.py`, `plotting.py`, `__init__.py` and
`filters.third_octave_filter_bank`.

Outside this directory, `unit_conversion.py` and `default_units.py` at the
repository root are Kevin Horton's (2008, BSD-style license) and carry that
license in their own headers.

## python-acoustics (BSD 3-Clause License)

Copyright (c) 2013, Python Acoustics
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

* Neither the name of the {organization} nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
