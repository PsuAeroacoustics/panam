# AGENTS.md

Instructions for AI coding tools working on PANAM. Human developers: see
`docs/DEVELOPING.md`, which this file summarizes. Read it before a large change.

## Setup and commands

- Python 3.11 or newer. Use the repository's virtual environment:
  `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .`,
  from the repository root. `pip install -e .` installs the `panam` package in
  editable mode; `pyproject.toml` (setuptools) installs `panam/` only, and its
  dependencies mirror `requirements.txt`, so change both together. Keep
  `requirements.txt` to package names: pip reads a path there against the
  current directory, which breaks a project that installs it with `-r`.
- Layout: the library is the package `panam/` (`from panam import
  flight_acoustics as fa`), the command-line tools are scripts in `tools/`, the
  examples are scripts in `examples/`. The library never imports from
  `tools/` or `examples/`; a tool holds its argument parsing and calls into
  `panam`.
- Tests: `pytest`, or `pytest -q -m "not data"` to skip the external-data tests.
  One file: `pytest tests/test_ega.py`.
- `pytest.ini` puts `.` and `tests/` on `sys.path`, so the tests import `panam`
  without the install and the scripts as `from tools import nod_tool`. Scripts
  find `example_data/` from their own location. No command needs `PYTHONPATH`.
- External data and the NICEOPS executables are found by name through
  `local_paths.data_path(name)`: a CLI argument, then `PANAM_<NAME>`, then the
  git-ignored `local_paths.toml` at the repository root. Missing data makes
  tests skip. One older lookup remains and is documented, not removed:
  `tests/sphere_helpers.executable` tries `local_paths`, then `NICEOPS` /
  `NICEOPS_RAY_GEOMETRY`, then `PATH`.

## Module map (put new code in the module that owns the topic)

Library modules, in `panam/`:

- `flight_acoustics.py`: re-exports every name it had before it was split into
  the topic modules below, so `fa.<name>` keeps working. Add nothing to it, and
  import the topic modules, not it, in library code (`tests/test_module_layout.py`
  lists the older modules that still use it).
- Topic modules: `spectra.py` (PSDs, bands, spectrograms, `dBAw`), `metrics.py`
  (PNL/EPNL, SEL, MIL-STD-1474E), `geometry.py` (`hemigen`, `dedopplerize`,
  `art2umapr`, Lambert angles), `absorption.py`, `gridding.py` (IDW weights,
  `interpolate_hemisphere_power`), `depropagation.py`
  (`depropagate_hemisphere`), `sphere_io.py` (sphere netCDF, run metadata),
  `database.py` (`build_empirical_database`, `add_sphere_group`), `norah2.py`
  (`.hem` files, `build_database_from_norah2`), `lambert_plots.py`,
  `projection.py`, `array_design.py`, `tracks.py`, `ground_effect.py` (`ega`).
- `noise_abatement_2017.py`: the 2017 flight-test build (`build_sphere`,
  `build_all`, track check, plate tables); its CLI is
  `tools/noise_abatement_2017.py`.
- `ground_plane.py`, `axisymmetric_bem.py`: ground, pole and ground-plate
  receiver models. Numba-compiled; ft and ft/s.
- `refracted_rays.py`: ray models for depropagation; wraps NICEOPS's
  `niceops_ray_geometry`.
- `vold_kalman_filter.py`: Vold-Kalman order tracking.
- `order_tracking.py`: Doppler-predicted order tracks at a fixed microphone,
  ridge tracking, clock offsets.
- `sphere_statistics.py`: energy averaging of repeat-run spheres,
  band-by-band sphere comparison (CLI `tools/sphere_diff.py`) and per-node sample
  statistics (`node_statistics`).
- `nod_database.py`: reads `.nod` databases: `load_nod`, `validate_nod`,
  `compare_databases`, `to_frame`, and the maps and summaries of
  `tools/nod_tool.py` (`plot_condition`, `plot_envelope`).
- `synthetic_flyover.py`: synthetic straight-line passes of a source of known
  directivity (`synthesize_pass`, `source_level_db`); the golden results' input.
- `synthesis.py`: forward synthesis of microphone band histories from a
  hemisphere (`synthesize_band_histories`) and the leave-microphones-out check
  (`leave_mics_out`).
- `acoustics/` (`panam.acoustics`): `signal_io` (loaders), `filters`, `atmosphere` and
  `iso_9613_1_1993` (ISO 9613-1), `time_weighting` (IEC 61672-1 frequency
  and time weighting, Lmax, Leq), `plotting` (`acoustic_plot_style`),
  `impulsiveness` (BVI crest factor, kurtosis, harmonic ratio),
  `psychoacoustics` (ISO 532-1 loudness, DIN 45692 sharpness).
- `cli.py`: `colon_pair`, `colon_pair_or_single`, `add_output_options`,
  `save_or_show`, `save_all_or_show`.
- `provenance.py`: the `panam_*` provenance stamp on written files.
- `build_recipe.py`: TOML build recipes (`load_recipe`) for `--recipe` on
  `tools/noise_abatement_2017.py` and `tools/build_empirical_database.py`;
  `docs/recipes/release_2017.toml` is the 2017 release.
- `local_paths.py`: data lookup. `unit_conversion.py`, `default_units.py`:
  third-party unit conversions. `data/`: the MIL-STD-1474E table.

Tools are in `tools/` and examples in `examples/`; the README's tools and
examples tables list them.

## Conventions

- State the units of every argument and return value in the docstring.
  2017 build and ground models: ft, ft/s. Sphere files: `RADIUS` ft, `SPEED`
  kt. NICEOPS databases: metric. Flow resistance: kPa s/m^2.
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
- New CLI: a script in `tools/` with a `main(argv=None)`; argparse with
  `description=`, units and defaults in every help string, `cli.colon_pair` for ranges, `cli.add_output_options` and
  `cli.save_or_show` (`-o/--output`, `-p/--plot`) for figures, a
  smoke test in `tests/test_cli_scripts.py` (`run_here`), and a row in the
  README's tools table.
- File contracts (sphere netCDF, `.nod`, `vehicle.cfg`): `docs/file_formats.md`.
  A `.nod` change that NICEOPS must know about bumps
  `DATABASE_FORMAT_VERSION` and needs a matching NICEOPS change.

## Tests and results

- Every behavior change comes with a test that fails without it.
- `tests/data/golden_results.npz` pins depropagation and a built database to
  1e-6 dB. A refactor must pass unchanged. A commit that changes results on
  purpose regenerates it and says so:
  `PANAM_REGENERATE_GOLDEN=1 python -m pytest tests/test_golden_results.py`.
- Never regenerate `tests/data/python_acoustics_reference.npz` from PANAM.
- Shared fixtures: `tests/sphere_helpers.py`. Tests use the `Agg` backend.
  `tests/conftest.py` validates every database built through `fa.build_...`.
  A monkeypatch must target the module that makes the call (`depropagation.hemigen`),
  not `fa`.
- Skips: `@pytest.mark.data` (external datasets), NICEOPS executables not
  configured, example data missing. CI (`.github/workflows/tests.yml`: Ubuntu
  with Python 3.11 and 3.14, macOS with 3.14; coverage and `--durations=20`
  reported, no threshold) has no data, so those skip there: run them locally when you
  touch the build, the database writers or the NORAH2 import.
- Numba compiles on first call and caches in `__pycache__`. Plate tables are
  cached as `.npz` in `~/.cache/panam/plate_tables` (15-45 s each to
  compute), keyed on `axisymmetric_bem.PLATE_TABLE_VERSION` and every
  setting. Bump the version in a commit that changes plate-table numbers.
  `tools/noise_abatement_2017.py --plate-cache list|clear` inspects or empties it.
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

- Never commit data, results or machine paths: no `local_paths.toml`, no flight-test recordings, spheres or databases, no absolute paths from your machine, no credentials. Never `git add -f` an ignored file (`example_data/*.mat`, `example_data/vehicle.cfg` and `panam/data/mil_std_1474e_table_c1_full.csv` are tracked on purpose; add no others).
- Never commit flight-test data or results from the private validation repository, or cite its file paths. PANAM is public: restate the fact inline.
- Never prefix commands with `PYTHONPATH=.`; install with `requirements.txt` and run from the repository root.
- Never change behavior or results without a test; never regenerate the golden results to make a refactor pass.
- Never push to `master`; use a branch and a pull request.
