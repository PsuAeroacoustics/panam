# Installation Steps

Python 3.11 or newer is required (`local_paths.py` uses `tomllib`).

1. Create a Python virtual environment (venv)
`python3 -m venv .venv`
2. Activate the environment   
     `source .venv/bin/activate` (Unix)   
     `.venv\Scripts\Activate.ps1` (Windows PowerShell)
3. Install the dependencies:
`pip install -r requirements.txt`

# Running the tests

From the repository root, run `pytest` (or `python -m pytest`).

# Data locations

Scripts and tests that need external data (the 2017 Noise Abatement flight
test archive, the AS350 demo data, the EASA NORAH2 distribution) look it up by
name instead of using hard-coded paths. Copy `local_paths.example.toml` to
`local_paths.toml` (git-ignored) and point each entry at your copy, or set the
matching `PANAM_<NAME>` environment variable (e.g. `PANAM_NORAH2`). Tests
whose data are not configured are skipped. See `local_paths.py` for the names.

# Signal loading and plotting

Signal loaders are available in `panam_acoustics.signal_io` and remain
importable from `flight_acoustics`. Use `open_h5_signal` as a context manager
and copy samples or attributes inside the context. The legacy
`load_h5_signal` returns a live HDF5 object; callers must close its `.file`.
UFF channels must have matching, uniformly spaced time grids.

`sound_exposure_level` requires matching level-history shapes and a positive
finite sample interval. NaN samples raise an error by default. Pass
`missing="omit"` to integrate available samples; the returned
`missing_samples` counts omitted integration samples, while `duration_s`
remains the selected interval length. Negative infinity represents zero energy.

Plotting functions apply their style within a temporary context; importing
`flight_acoustics` leaves global Matplotlib settings unchanged. Tests use the
noninteractive `Agg` backend automatically.

For `spectrogram_plot.py`, `--frequency low:high` selects the displayed
frequency range. `--y-limits` takes precedence when both are supplied.

# NORAH2 hemispheres

`norah2_to_nod.py` builds a NICE-OPS database from NORAH2 `.hem` hemispheres
(`flight_acoustics.build_database_from_norah2`). The vehicle and the meaning of
ACSPEED (`--speed ias-to-tas` or `ground-speed`) must be given. See
`docs/norah2_import.md` for the conversion and its hazards.
