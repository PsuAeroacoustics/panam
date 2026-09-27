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
