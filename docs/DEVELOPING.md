# Developing PANAM

This guide is for people changing PANAM's code. Users start at the
[README](../README.md); the equations are in [THEORY.md](THEORY.md); file
contracts are in [file_formats.md](file_formats.md). AI coding tools read
[AGENTS.md](../AGENTS.md), which summarizes this guide.

## Architecture

PANAM is a set of modules at the repository root, not an installed package.
Scripts and tests import them by name (`import flight_acoustics as fa`), so
everything runs from the repository root.

### Data flow

A source noise sphere is built from flight-test recordings, and a database is
built from a directory of spheres:

1. **Recordings.** Pressure time series per microphone, with the aircraft's
   track. `noise_abatement_2017.load_run_channels` reads the 2017 test's
   netCDF channels; `panam_acoustics.signal_io` reads netCDF, HDF5 and UFF
   recordings in general.
2. **Spectra.** `flight_acoustics.depropagate_hemisphere` cuts each channel
   into frames and forms one-third-octave band power from Hann-windowed PSDs
   (`third_octave_method='fft'`, optionally `tone_aware_band_power`) or from
   `panam_acoustics.filters.third_octave_filter_bank` (`'filter_bank'`).
3. **Depropagation.** In the same call: emission geometry from `hemigen`
   (or a ray model from `refracted_rays`), the ambient gate, the receiver
   response (`receiver_response_db`, the ground-plate model), spherical
   spreading and atmospheric absorption back to `r_ref`.
4. **Gridding.** Samples are interpolated in energy onto an azimuth/elevation
   grid by `adaptive_idw_weights` (with `interpolation=`) or the fixed-radius
   `shepIDW_weights`. The result is a hemisphere dict in UMAPR angles.
5. **Sphere file.** `write_aam_hemisphere_netcdf` resamples the hemisphere
   onto the AAM (phi, theta) grid and writes the sphere netCDF;
   `write_norah2_hemisphere` writes the NORAH2 `.hem` form.
6. **Database.** `build_empirical_database` reads a directory of spheres and
   its `vehicle.cfg` and writes the NICE-OPS `.nod` database through
   `add_sphere_group`. `build_database_from_norah2` does the same from NORAH2
   hemispheres.

`noise_abatement_2017.build_sphere` drives steps 1 to 5 for one run of the
2017 test; `build_all` loops over an aircraft's runs. The plotting and
projection functions (`nc_lambert_ea`, `project_sphere`, `fried_egg_plot`)
read the sphere files back.

### Module map

| Module | Owns |
| --- | --- |
| `flight_acoustics.py` | Spectra and bands, metrics (SEL, PNL/EPNL, MIL-STD-1474E), emission geometry (`hemigen`, `dedopplerize`), `depropagate_hemisphere`, gridding, sphere and NORAH2 I/O, the database writers, ground projection, Lambert plots, `ega` and the reflection coefficient, array geometry helpers |
| `noise_abatement_2017.py` | The 2017 Noise Abatement build: dataset layout, tracks and the track check, steady windows, atmosphere per run, plate tables, `build_sphere`, `build_all`, and its CLI |
| `ground_plane.py` | Ground impedance models, pole microphones, co-located pole and board pairs, ground-plate models, the exact half-space Green's function, the two-path fit, `emission_times`. Numba-compiled |
| `axisymmetric_bem.py` | The axisymmetric BEM of the plate on the ground (`scattering`, `table`, `board_level`, `field`, `write_netcdf` for NICE-OPS). Numba-compiled |
| `refracted_rays.py` | The ray-model interface for depropagation: `straight_ray_model`, and `external_ray_model`, which runs NICE-OPS's `niceops_ray_geometry` |
| `vold_kalman_filter.py` | The multi-shaft Vold-Kalman order-tracking filter |
| `panam_acoustics/` | `signal_io` (recording loaders), `filters` (Butterworth helpers, the one-third-octave filter bank), `atmosphere` and `iso_9613_1_1993` (ISO 9613-1 absorption), `plotting` (`acoustic_plot_style`). See its `THIRD_PARTY_NOTICES.md` for the parts derived from python-acoustics |
| `cli.py` | Argument types and figure output shared by the scripts: `colon_pair`, `colon_pair_or_single`, `save_or_show` |
| `local_paths.py` | `data_path(name)`: where this machine keeps external data and the NICE-OPS executables |
| `unit_conversion.py`, `default_units.py` | Aerospace unit conversions (third-party, BSD-style license in the file headers) |

Scripts with a command line: `noise_abatement_2017.py` (the main pipeline),
`build_empirical_database.py`, `norah2_to_nod.py`, `array_planner.py`, and the
plotting scripts `nc_lambert_ea.py`, `plot_projection.py`, `fried_egg_plot.py`,
`spectrogram_plot.py`, `atmomap.py`, `ega_plot.py` and `board_field_plot.py`.
`AS350_289108_demo.py`, `hexacopter_vkf_example.py`, `as350_flip_check.py` and
the `__main__` block of `vold_kalman_filter.py` are examples and diagnostics,
not tools.

### Where new code goes

Put new code in the module that owns the topic. When no module does, add a new
module rather than growing `flight_acoustics.py`, which is already over 6,000
lines. Receiver and ground models go in `ground_plane.py` or
`axisymmetric_bem.py`; anything specific to one flight test goes in that
test's module, as `noise_abatement_2017.py` is for 2017; recording formats go
in `panam_acoustics/signal_io.py`. A new model's equation goes in
`docs/THEORY.md` in the same commit, with a row in its appendix table
([Where each model lives](THEORY.md#appendix-where-each-model-lives)).

## Library conventions

### Units

Every function states the units of its arguments and return values in its
docstring. Keep to that, and name a unit in a parameter name when it is not
obvious (`radius_ft`, `speed_knots`). The common systems are:

- The 2017 build, `ground_plane` and `axisymmetric_bem` work in ft and ft/s.
  `depropagate_hemisphere` works in its `length_units` (default `'ft'`), with
  `speed_of_sound` in those units per second.
- Sphere netCDF files carry `RADIUS` in ft and `SPEED` in knots.
- NICE-OPS databases and the plate tables for NICE-OPS are metric.
- `atmosorb(freq, temp, humid, pstat)` takes deg C, % RH and mbar and returns
  dB/m; `panam_acoustics.atmosphere.Atmosphere` takes K, kPa and % RH.
- Flow resistance is in kPa s/m^2. The reference pressure is
  `flight_acoustics.P_REF` (20 uPa).

### Levels, energy and missing values

Average, interpolate and sum levels in energy (10^(L/10)), not in dB. In memory
a level is finite when measured, `-inf` when measured with no energy (gated,
zero power) and `NaN` when there is no data (an unmeasured cell, a gap). How
these states are written to a sphere file and a database is in
[file_formats.md](file_formats.md#how-a-bins-state-travels).

- `power_to_db` returns `-inf` for zero power and keeps `NaN`; never clamp
  power to a tiny value, which turns "no energy" into a finite level.
- Suppress the resulting numpy warnings locally with
  `with np.errstate(...)`, never with the global `np.seterr`.
- Load factor n (`build_empirical_database`): `dBA` gets +20 lg n (n > 0) and
  C_T = n C_W; the stored band `amplitude` is not scaled.
- `dBAw` is vectorized: `fa.dBAw(frequency)`.

### Angle frames

- **ART/AAM** (sphere files, `.nod`, NORAH2): phi is 0 directly below,
  positive to starboard; theta is 0 at the nose and 180 at the tail (AAM v3
  Technical Reference, sec. 2.4.1).
- **UMAPR** (hemisphere dicts, `hemigen`, depropagation): azimuth 180 ahead,
  90 starboard, 270 port, 0 behind; elevation positive below the horizon.
  Convert with `art2umapr(phi, theta)` (radians) or `norah2umapr`.
- **Track frame** (the 2017 build): +x along the run's frame bearing
  (`frame_bearing_deg`), +y 90 deg to its left, z up. Every compass direction
  goes through `frame_bearing_deg`.
- **Azimuth reference**: a sphere's azimuth is measured from the ground track
  (`'track'`) or the INS heading (`'heading'`); see `AZIMUTH_REFERENCES` and
  [database_build.md](database_build.md).
- Before 2026-09-24, `art2umapr` put phi = +90 to port, and
  `flip_y_for_geometry=True` compensated for it. Both are history: the flag
  now defaults to False, and `as350_flip_check.py` records the check.

### Plots

Plotting functions are decorated with
`panam_acoustics.plotting.acoustic_plot_style`, which applies the
`fivethirtyeight` style and a few rc settings only while the function runs.
Importing `flight_acoustics` leaves Matplotlib's global settings unchanged;
never change the global style or `rcParams`. Return `(fig, ax, ...)` rather
than showing the figure. `tests/conftest.py` sets the `Agg` backend for tests.

### Signal loading

- Use `open_h5_signal` as a context manager, and copy samples and attributes
  inside the context. The legacy `load_h5_signal` returns a live HDF5 object;
  the caller must close its `.file`.
- `load_UFF_signal` reads only type-58 sets, and its channels must share one
  uniformly spaced time grid.
- The loaders live in `panam_acoustics.signal_io` and are re-exported by
  `flight_acoustics`.

### Sound exposure level

`sound_exposure_level` requires matching level-history shapes and a positive,
finite sample interval. NaN samples raise an error by default. Pass
`missing='omit'` to integrate the available samples; the result's
`missing_samples` counts the omitted ones, while `duration_s` stays the length
of the selected interval. `-inf` means zero energy.

## Adding things

### A command-line script

- Use `argparse` with a `description=`, and give every option a help string
  that states its units and default.
- Use `cli.colon_pair` (or `colon_pair_or_single`) for `low:high` ranges and
  `cli.save_or_show(fig, args.output, args.plot)` for figure output, so the
  script writes a file with `-o` and opens a window otherwise. `atmomap.py`,
  `ega_plot.py` and `board_field_plot.py` predate this and always write a
  default file.
- Find external data with `local_paths.data_path(name, args.something)`, so
  that a command-line path overrides `PANAM_<NAME>` and `local_paths.toml`.
- Add a smoke test to `tests/test_cli_scripts.py`. `run_here` runs the script
  as `__main__` in the test process, which is fast; `_run` starts a fresh
  interpreter, for checks that need one.
- Add the script to the README's tools table.

### A vehicle or aircraft

`build_empirical_database` needs a `vehicle.cfg` in the sphere directory: rotor
blades, radii and tip speeds, the atmosphere, weight and drag, in SI units. Its
format, including the optional `[Option]` section, is in
[file_formats.md](file_formats.md). `tests/sphere_helpers.VEHICLE_CFG` is a
minimal example.

For another aircraft of the 2017 test, the dataset directory name is the
`noise_abatement_2017.py` argument. Add it to `AIRCRAFT_SPHERE_PREFIX` only
if its spheres need a legacy file prefix. Then build the spheres, write the
`vehicle.cfg`, run `build_empirical_database.py`, and load the database in
NICE-OPS ([database_build.md](database_build.md)).

### A model

Write the equation in `docs/THEORY.md`, and add or update its row in the
appendix table [Where each model lives](THEORY.md#appendix-where-each-model-lives)
(section, functions, tests). Implement it with its units stated, and test it
against a closed form, a limit or an independent reference. If it
changes results, regenerate the golden results (below) in the same commit.

### A change to the `.nod` format

The `.nod` database is a contract with NICE-OPS
([file_formats.md](file_formats.md#the-nice-ops-database-nod); the version rule is in
[DATABASE_FORMAT_VERSION](file_formats.md#database_format_version) and the tests in
[How the contract is tested](file_formats.md#how-the-contract-is-tested)). A change that NICE-OPS must know about
bumps `flight_acoustics.DATABASE_FORMAT_VERSION` and needs a matching NICE-OPS
change. Three tests hold the contract: `tests/test_niceops_loads_databases.py`
runs NICE-OPS's self-test on PANAM-built databases,
`tests/test_database_provenance.py` pins names and attributes, and
`tests/test_golden_results.py` pins built values.

## Testing

Run the tests from the repository root:

```bash
pytest                    # everything configured on this machine
pytest -m "not data"      # skip the tests that read external datasets
pytest tests/test_ega.py  # one file
```

`pytest.ini` puts the repository root and `tests/` on `sys.path`, so neither
`PYTHONPATH=.` nor an install is needed. There are about 750 tests;
`pytest -m "not data"` takes a few minutes on a laptop.

### What skips, and why

Tests that need something this machine may not have skip instead of failing:

- `@pytest.mark.data`: reads an external dataset found through `local_paths`
  (the NORAH2 package, the 2017 archive). These can take minutes on a cold
  cloud drive. Deselect them with `-m "not data"`.
- `skipif` on `sphere_helpers.executable(...)`: the NICE-OPS executables
  `niceops` and `niceops_ray_geometry`. `executable` looks in `local_paths`
  (`PANAM_NICEOPS`, `local_paths.toml`), then in the unprefixed environment
  variable (`NICEOPS`, `NICEOPS_RAY_GEOMETRY`), then on `PATH`.
- `skipif` on `example_data/AS350B3108.nc`: the bundled example sphere.
- `importorskip('pyuff')` in the UFF loader tests.
- `vold_kalman_filter` uses the optional `scikits.umfpack` in its sparse
  fallback and for `solver='umfpack'`; the test that `solver='umfpack'` fails
  without the package skips when it is installed.

The 2017 Noise Abatement archive and the AS350 demo data are not distributed
with PANAM. Without them, their tests skip, and the scripts that need them
stop with a `LookupError` that says how to configure the path.

Shared fixtures (a minimal hemisphere, a sphere directory with its
`vehicle.cfg`, raw sphere files) are in `tests/sphere_helpers.py`.

### Golden results

`tests/data/golden_results.npz` pins depropagation and a database built from
it, over the main options, to 1e-6 dB. A refactor must leave them unchanged.
A commit that changes results on purpose regenerates them and says so:

```bash
PANAM_REGENERATE_GOLDEN=1 python -m pytest tests/test_golden_results.py
```

`tests/data/python_acoustics_reference.npz` pins the Butterworth helpers in
`panam_acoustics.filters` against python-acoustics' output. Do not regenerate
it from PANAM.

### Continuous integration

`.github/workflows/tests.yml` runs `python -m pytest -q` on Ubuntu with
Python 3.11 and 3.14, on every push and pull request to `master`. CI has no
external data and no NICE-OPS executables, so the data and NICE-OPS tests
skip there; run them locally before merging a change that touches the build,
the database writers or the NORAH2 import.

## Environment and caches

- Python 3.11 or newer (`local_paths.py` uses `tomllib`). Install with
  `pip install -r requirements.txt` in a virtual environment at the
  repository root.
- `local_paths.toml` holds this machine's data paths and is git-ignored.
  `local_paths.example.toml` lists the names; `PANAM_<NAME>` overrides a
  file entry, and a command-line argument overrides both.
- `ground_plane.py` and `axisymmetric_bem.py` are compiled with Numba
  (`cache=True`). The first call after a change compiles, and the compiled
  code is cached in `__pycache__`.
- `noise_abatement_2017.plate_table` caches plate tables in memory and in
  `~/.cache/panam/plate_tables`, one pickle per microphone height, ground,
  sound speed (rounded to 0.5% steps) and band set. A table takes 15-45 s to
  compute. Delete the directory after changing the plate model; the cache key
  does not include the code version.
- A 2017 build reads the archive from wherever `local_paths` points; on a
  cloud drive, `build_all` prefetches each run's files in threads.

## The ground-plate correction in the sphere builds

`noise_abatement_2017.build_sphere(board_correction='plate_bem')` is the
default (`--board-correction` on the command line); `'flat'` keeps the older
flat -6 dB pressure-doubling factor. The model is in
[THEORY §4.5](THEORY.md#45-ground-plate-microphones), and the research that
led to it is in [notes/ground_plane_corrections.md](notes/ground_plane_corrections.md).
The code contract:

- **The hook.** `depropagate_hemisphere(receiver_response_db=...)` takes a
  callable giving the installation's band-averaged response for each emission
  point. It divides each band's energy by that response after the ambient
  gate and before spreading. `noise_abatement_2017.plate_response` builds the
  callable from the plate tables, and the pressures then go in unscaled.
- **The cache.** `plate_table` caches tables in `~/.cache/panam/plate_tables`
  (see above). Delete it after changing the plate model.
- **NICE-OPS.** `axisymmetric_bem.write_netcdf` exports a plate table for
  NICE-OPS's `--plate_table` option: band x sub-frequency x elevation x
  azimuth, metric, with the ground it was computed on, which NICE-OPS checks
  against its own ground model.

## Workflow

- The default branch is `master`. Work on a branch (dated names such as
  `cleanup-2026-10-09` are the habit) and merge through a pull request to
  `PsuAeroacoustics/panam`. CI must pass.
- Commit subjects start with the area: `THEORY:`, `docs:`, `Tests:`,
  `Simplify:`, or a module or file name. The bodies are short bulleted lists.
  The `F0xx:` prefixes in the history refer to a past review list; do not
  start new ones.
- Say in the commit message whether results change. A refactor states that the
  golden results match ("Results are bit-identical: the golden results match
  exactly"); a commit that changes results says what changed and regenerates
  the goldens.
- A model change updates `docs/THEORY.md` in the same commit, including its
  appendix table, "Where each model lives".
- The repository is public. Do not commit flight-test data, results built
  from it, `local_paths.toml`, absolute paths from your machine, or
  credentials. Data file types (`*.csv`, `*.mat`, `*.kmz`, `*.pdf`) are
  git-ignored; do not force-add them. The bundled example
  (`example_data/*.mat`) and the MIL-STD-1474E table
  (`mil_std_1474e_table_c1_full.csv`) are the deliberate exceptions; adding
  another needs the maintainer's agreement. Results from the private companion
  validation repository stay there: restate a needed fact in the text instead
  of citing that repository's files.
