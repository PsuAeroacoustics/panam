# AGENTS.md

Instructions for AI coding tools working on PANAM. Human developers: see
`docs/DEVELOPING.md`, which this file summarizes. Read it before a large change.

## Setup and commands

- Python 3.11 or newer. Use the repository's virtual environment:
  `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.
- Run everything from the repository root. Modules are imported by name
  (`import flight_acoustics as fa`); there is no installed package.
- Tests: `pytest`, or `pytest -q -m "not data"` to skip the external-data tests.
  One file: `pytest tests/test_ega.py`.
- `pytest.ini` puts `.` and `tests/` on `sys.path`. Scripts find their own
  directory. No command needs `PYTHONPATH`.
- External data and the NICE-OPS executables are found by name through
  `local_paths.data_path(name)`: a CLI argument, then `PANAM_<NAME>`, then the
  git-ignored `local_paths.toml`. Missing data makes tests skip. Two older
  lookups remain and are documented, not removed: `as350_flip_check.py` reads
  `AS350_DEMO_PATH` ahead of `as350_demo`, and `tests/sphere_helpers.executable`
  tries `local_paths`, then `NICEOPS` / `NICEOPS_RAY_GEOMETRY`, then `PATH`.

## Module map (put new code in the module that owns the topic)

- `flight_acoustics.py`: spectra and bands, metrics, emission geometry,
  `depropagate_hemisphere`, gridding, sphere and NORAH2 I/O, database writers
  (`build_empirical_database`, `build_database_from_norah2`), projection, plots,
  `ega`. Over 6,000 lines: add a new module rather than growing it.
- `noise_abatement_2017.py`: the 2017 flight-test build (`build_sphere`,
  `build_all`, track check, plate tables) and its CLI.
- `ground_plane.py`, `axisymmetric_bem.py`: ground, pole and ground-plate
  receiver models. Numba-compiled; ft and ft/s.
- `refracted_rays.py`: ray models for depropagation; wraps NICE-OPS's
  `niceops_ray_geometry`.
- `vold_kalman_filter.py`: Vold-Kalman order tracking.
- `panam_acoustics/`: `signal_io` (loaders), `filters`, `atmosphere` and
  `iso_9613_1_1993` (ISO 9613-1), `plotting` (`acoustic_plot_style`).
- `cli.py`: `colon_pair`, `colon_pair_or_single`, `save_or_show`.
- `local_paths.py`: data lookup. `unit_conversion.py`, `default_units.py`:
  third-party unit conversions.
- Tools: `noise_abatement_2017.py`, `build_empirical_database.py`,
  `norah2_to_nod.py`, `array_planner.py`, plotting scripts. Examples and
  diagnostics: `AS350_289108_demo.py`, `hexacopter_vkf_example.py`,
  `as350_flip_check.py`.

## Conventions

- State the units of every argument and return value in the docstring.
  2017 build and ground models: ft, ft/s. Sphere files: `RADIUS` ft, `SPEED`
  kt. NICE-OPS databases: metric. Flow resistance: kPa s/m^2.
- Average, interpolate and sum levels in energy, not dB.
- `-inf` = measured, no energy. `NaN` = no data. Sphere files write `-999`
  for both; `.nod` uses `coverage` 0. `mask_missing_levels` maps `NaN`,
  `-999` and `>1e34` on disk to `-inf`. Never clamp power to a tiny value.
  Full rules: `docs/file_formats.md#how-a-bins-state-travels`.
- Suppress numpy warnings locally with `np.errstate`, never `np.seterr`.
- Load factor n: `dBA` gets +20 lg n (n > 0) and C_T = n C_W; band `amplitude` is not scaled.
- Angles: ART/AAM (files, `.nod`, NORAH2) phi 0 below, positive to starboard,
  theta 0 at the nose. UMAPR (hemisphere dicts) azimuth 180 ahead, 90
  starboard, elevation positive below the horizon. Convert with `art2umapr`.
- Plotting functions use `@acoustic_plot_style`; never change global
  Matplotlib style or `rcParams`. Return the figure; do not call `plt.show()`
  in library code.
- `open_h5_signal` is a context manager; `load_h5_signal` leaves a file open.
- `sound_exposure_level(..., missing='omit')` integrates around NaN samples.
- New CLI: argparse with `description=`, units and defaults in every help
  string, `cli.colon_pair` for ranges, `cli.save_or_show` for figures, a
  smoke test in `tests/test_cli_scripts.py` (`run_here`), and a row in the
  README's tools table.
- File contracts (sphere netCDF, `.nod`, `vehicle.cfg`): `docs/file_formats.md`.
  A `.nod` change that NICE-OPS must know about bumps
  `DATABASE_FORMAT_VERSION` and needs a matching NICE-OPS change.

## Tests and results

- Every behavior change comes with a test that fails without it.
- `tests/data/golden_results.npz` pins depropagation and a built database to
  1e-6 dB. A refactor must pass unchanged. A commit that changes results on
  purpose regenerates it and says so:
  `PANAM_REGENERATE_GOLDEN=1 python -m pytest tests/test_golden_results.py`.
- Never regenerate `tests/data/python_acoustics_reference.npz` from PANAM.
- Shared fixtures: `tests/sphere_helpers.py`. Tests use the `Agg` backend.
- Skips: `@pytest.mark.data` (external datasets), NICE-OPS executables not
  configured, example data missing. CI (`.github/workflows/tests.yml`, Python
  3.11 and 3.14) has no data, so those skip there: run them locally when you
  touch the build, the database writers or the NORAH2 import.
- Numba compiles on first call and caches in `__pycache__`. Plate tables are
  cached in `~/.cache/panam/plate_tables` (15-45 s each to compute).
- The 2017 archive and the AS350 demo data are not distributed with PANAM.
  Do not invent a download location.
- Running an example or a build writes files that `.gitignore` does not cover
  (`.png`, `.nc`, `.nod`). Write them to `demo_plots/` (ignored) or outside the
  checkout, and check `git status` before staging.

## Documentation

- Equations: `docs/THEORY.md`. A model change updates it in the same commit,
  including the row of its appendix table, "Where each model lives"
  (`docs/THEORY.md#appendix-where-each-model-lives`).
- User how-tos: `README.md`, `docs/database_build.md` (with the 2017 release
  settings table), `docs/norah2_import.md`. File contracts:
  `docs/file_formats.md`.
- `docs/notes/` is dated research history. Do not edit it to match the code;
  write current behavior in THEORY or DEVELOPING instead.
- Style: plain declarative sentences, US spelling, no marketing, no emojis.

## Workflow

- Default branch `master`. Work on a branch and open a pull request to
  `PsuAeroacoustics/panam`. CI must pass.
- Commit subjects start with the area (`THEORY:`, `docs:`, `Tests:`,
  `Simplify:`, a module name). Bodies are short bulleted lists.
- Say whether results change: "Results are bit-identical: the golden results
  match exactly", or what changed and that the goldens were regenerated.
- `F0xx:` prefixes in the history refer to a past review list. Do not invent
  new ones.

## Never do

- Never commit data, results or machine paths: no `local_paths.toml`, no flight-test recordings, spheres or databases, no absolute paths from your machine, no credentials. Never `git add -f` an ignored file (`example_data/*.mat` and `mil_std_1474e_table_c1_full.csv` are tracked on purpose; add no others).
- Never commit flight-test data or results from the private validation repository, or cite its file paths. PANAM is public: restate the fact inline.
- Never prefix commands with `PYTHONPATH=.`; run from the repository root.
- Never change behavior or results without a test; never regenerate the golden results to make a refactor pass.
- Never push to `master`; use a branch and a pull request.
