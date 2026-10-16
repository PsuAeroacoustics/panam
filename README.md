# PANAM

PANAM is a Python toolkit for rotorcraft aeroacoustics. It turns flight-test
microphone recordings into source noise spheres, builds the sphere databases
(`.nod`) that the NICEOPS footprint model reads, converts NORAH2 hemispheres
to the same format, and plots spheres, footprints and spectrograms.

## Install

Python 3.11 or newer is required (CI tests 3.11 and 3.14 on Linux, and 3.14
on macOS).

```
python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` covers every import and includes pytest. PANAM is not an
installed package: run the scripts from the checkout. `pyproject.toml` records
the project's metadata, the supported Python versions and the same dependency
list, for tools that read it; it builds nothing, so `pip install .` is not
supported. The optional
`scikits.umfpack` package is used by `vold_kalman_filter`'s sparse fallback solve
(narrow bands, or a system that is not positive definite) and by
`solver='umfpack'`; without it the fallback uses SciPy's SuperLU.
The ground-plate models are compiled by numba on first use, so the first run
that needs them is slower.

## Quickstart

These run on the data bundled in `example_data/`. Run them from the repository
root. They write into `demo_plots/`, which is git-ignored.

```
mkdir -p demo_plots
python nc_lambert_ea.py example_data/AS350B3108.nc -w A -o demo_plots/sphere.png
python plot_projection.py example_data/AS350B3108.nc -a 150 -u ft -o demo_plots/footprint.png  # -a is in m; -u sets the axes
python array_planner.py coverage --nmics 12 --altitude 150 -o demo_plots/coverage.png
python hexacopter_vkf_example.py   # writes demo_plots/hexa_*.png
python sphere_to_norah2.py example_data/AS350B3108.nc -o demo_plots/AS350B3108.hem \
    --temperature 288.15 --pressure 101.325 --humidity 70   # K, kPa, % of the measurement
```

To build a database, a directory needs spheres and a `vehicle.cfg`. One sphere
makes a toy database:

```
mkdir -p demo_plots/spheres
cp example_data/AS350B3108.nc demo_plots/spheres/
cp example_data/vehicle.cfg demo_plots/spheres/
python build_empirical_database.py demo_plots/spheres demo_plots/toy.nod
```

It warns that the sphere has no `azimuth_reference` and that the hover was
synthesized from the 78.7 kt sphere. Spheres from the 2017 flight test are built
as in [`docs/database_build.md`](docs/database_build.md). Spheres from other
recordings are built with the Python API (`depropagate_hemisphere`, then
`write_aam_hemisphere_netcdf`; see
[`docs/DEVELOPING.md`](docs/DEVELOPING.md#data-flow) and
[THEORY §5](docs/THEORY.md#5-depropagation)).

## Try it without flight-test data

`depropagation_tutorial.py` runs the whole pipeline on a synthetic flyover. It
synthesizes a straight, level pass of a source with a known directivity over a
line of microphones (`synthetic_flyover.synthesize_pass`), depropagates it to a
hemisphere at 100 ft, maps the recovered overall level minus the true one on a
Lambert equal-area projection, and builds a toy database from the result:

```
python depropagation_tutorial.py            # writes into demo_plots/
python depropagation_tutorial.py -o /tmp/tutorial --mics 11 --mic-y=-900:900
```

It prints the median and largest recovered-minus-true difference and writes
`tutorial_difference.png`, a sphere directory `tutorial_spheres/` (with a copy
of `example_data/vehicle.cfg`) and `tutorial.nod`. With the defaults the
difference stays within 1 dB. Read the script to see each step's API call.

Every script, tool, plotting script and example, takes `-h`. Run with no
options, each example does what it did before it had options.

## Tools

| Script | Purpose | Needs external data? |
|---|---|---|
| `noise_abatement_2017.py` | Build source spheres from the 2017 Noise Abatement flight test; `--check-tracks` checks the tracks without building; `--tone-split` also writes tonal and broadband spheres; `--node-statistics` adds per-cell sample counts and level spread; `--skip-existing` or `--skip-unchanged` resumes a build; `--jobs N` builds N runs at once; `--plate-cache list` or `clear` inspects or empties the plate-table cache; `--recipe` reads the settings from a build recipe (`docs/recipes/release_2017.toml`) | Yes: the 2017 archive (not for `--plate-cache`) |
| `build_empirical_database.py` | Build a NICEOPS database from a directory of spheres and its `vehicle.cfg`; `--manifest` and `--exclude` select the spheres; `--merge-repeats` energy-averages repeat runs; `--recipe` reads the settings from a build recipe | Your spheres |
| `norah2_to_nod.py` | Build a NICEOPS database from NORAH2 `.hem` hemispheres | `.hem` files |
| `nod_tool.py` | Inspect a NICEOPS database: `list` its conditions (speed, flight path angle, load factor, origin, coverage, maximum dBA), `plot` a Lambert map of one condition's dBA, EAA, band level or coverage, plot its `envelope` in speed and flight path angle, `check` the file contract | Your database |
| `compare_databases.py` | Compare two NICEOPS databases: per-band energy, coverage, unmatched conditions, root attributes; `--validate` checks the file contract | Your databases |
| `sphere_to_norah2.py` | Convert an AAM sphere to a NORAH2 `.hem` hemisphere; the measurement atmosphere is required | No (bundled sphere) |
| `nc_lambert_ea.py` | Plot a sphere's levels on a Lambert equal-area projection | No (bundled sphere) |
| `sphere_diff.py` | Energy-average repeat-run spheres (`average`), or print and map the band-by-band level difference of two spheres (`compare`) | Your spheres |
| `plot_projection.py` | Project a sphere onto flat ground and contour the footprint | No (bundled sphere) |
| `fried_egg_plot.py` | Contour ground levels of a directory of spheres over flight condition | Your spheres and `vehicle.cfg` |
| `spectrogram_plot.py` | Spectrogram of one HDF5 or netCDF recording | Your recording |
| `array_planner.py` | Design a linear microphone array and plot its level-pass coverage; `hover` and `takeoff` plot the coverage of ring microphones (optional TOML case file); export KMZ | No |
| `atmomap.py` | Map atmospheric absorption over temperature and pressure | No |
| `ega_plot.py` | Plot excess ground attenuation | No |
| `board_field_plot.py` | Map the sound field around a ground-plane microphone plate | No |
| `depropagation_tutorial.py` | Tutorial: depropagate a synthetic flyover, map recovered minus true directivity, build a toy database | No |
| `AS350_289108_demo.py` | Example: depropagate one AS350 run and compare it with its reference sphere | Yes: AS350 demo data |
| `hexacopter_vkf_example.py` | Example: de-Dopplerize one microphone of a hexacopter flyover and separate its rotor harmonics with the Vold-Kalman filter | No (bundled `.mat`) |
| `vold_kalman_filter.py` | Run as a script: Vold-Kalman filter examples on synthetic signals | No |
| `as350_flip_check.py` | Diagnostic: check the AS350 demo's lateral sign against its reference | Yes: AS350 demo data |

The plotting scripts write the file named by `-o/--output`, open a window
without it, and with `-o` and `-p/--plot` do both. `atmomap.py`, `ega_plot.py`
and `board_field_plot.py` always write a file (defaults
`demo_plots/atmomap.pdf`, `demo_plots/ega_plot.pdf` and `board_field.png` in
the current directory) and open it too with `--plot`; in the first two `-p`
means something else (`--pressure`, `--plot-type`). The examples that write
several files take `-o/--output-dir` (default `demo_plots/` in the current
directory; `AS350_289108_demo.py` still accepts `--outdir`) and `-p/--plot`
to also open the figures. The examples find `example_data/` beside the script, so
they run from any directory.

Loudness and sharpness of a one-third-octave spectrum are library functions, not a
script: `panam_acoustics.psychoacoustics.loudness_zwicker` (ISO 532-1:2017, stationary,
25 Hz to 12.5 kHz) and `sharpness_din` (DIN 45692). `iso_532_1_levels` picks those bands out of
a PANAM 10 Hz to 10 kHz spectrum and sets 12.5 kHz to no energy, which understates the loudness
by the 12.5 kHz content; see [THEORY §9](docs/THEORY.md#9-metrics).

## Data locations

External data and executables are looked up by name. Copy
`local_paths.example.toml` to `local_paths.toml` (git-ignored) and set the
entries you have. A path given on the command line wins, then the environment
variable `PANAM_<NAME>` (for example `PANAM_NORAH2`), then `local_paths.toml`.

| Name | What it is | Used by |
|---|---|---|
| `noise_abatement_2017` | 2017 FAA/NASA Noise Abatement flight test archive | `noise_abatement_2017.py` (or `--root`), track-check tests |
| `as350_demo` | AS350 demo data (`Acoustic/`, `Ambient/`, `AAM/`, `Tracking/`) | `AS350_289108_demo.py` and `as350_flip_check.py` (or `--basepath`) |
| `norah2` | EASA NORAH2 public distribution (`Hemispheres/`) | NORAH2 tests |
| `niceops` | NICEOPS executable | tests that load PANAM databases in NICEOPS |
| `niceops_ray_geometry` | NICEOPS ray-geometry executable | refracted-ray tests |
| `b407_legacy_spheres` | legacy B407 sphere directory, the recipe's `reference_directory` | `docs/recipes/release_2017.toml` |
| `b407_ray_profiles` | per-run NICEOPS atmosphere CSVs (`<run>.csv`, heights in ft), the recipe's `ray_profiles` | `docs/recipes/release_2017.toml` |

The 2017 archive and the AS350 demo data are not distributed with PANAM. The
NICEOPS executables come from a NICEOPS build.

`as350_flip_check.py` also reads the older variable `AS350_DEMO_PATH`, ahead of
`as350_demo`.

## Running the tests

```
pytest                  # from the repository root
pytest -m "not data"    # skip the tests that read the external datasets
```

Tests whose data or NICEOPS executables are not configured are skipped, so a
fresh checkout passes with skips. The `data` tests can take minutes on a cold
cloud drive.

## Documentation

- [`docs/THEORY.md`](docs/THEORY.md): the theory manual. Every model as
  equations, from the spectra through depropagation and gridding to the
  database and the metrics, with the code's defaults.
- [`docs/database_build.md`](docs/database_build.md): building spheres and a
  database from the 2017 flight test, step by step, and the table of the 2017
  release settings.
- [`docs/file_formats.md`](docs/file_formats.md): the sphere netCDF file, the
  `.nod` database contract and `vehicle.cfg`.
- [`docs/norah2_import.md`](docs/norah2_import.md): converting NORAH2
  hemispheres to a NICEOPS database.
- [`docs/DEVELOPING.md`](docs/DEVELOPING.md): for developers: code layout,
  conventions, testing.
- [`AGENTS.md`](AGENTS.md): instructions for AI coding tools.

Background: [`docs/notes/README.md`](docs/notes/README.md) indexes the dated
research notes (the ground-plate corrections, the 2017 build findings, the
NORAH2 import checks). They are kept as history and do not track the code.

PANAM is MIT licensed (`LICENSE`). Parts of `panam_acoustics` derive from
python-acoustics (`panam_acoustics/THIRD_PARTY_NOTICES.md`).
