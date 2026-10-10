# PANAM

PANAM is a Python toolkit for rotorcraft aeroacoustics. It turns flight-test
microphone recordings into source noise spheres, builds the sphere databases
(`.nod`) that the NICE-OPS footprint model reads, converts NORAH2 hemispheres
to the same format, and plots spheres, footprints and spectrograms.

## Install

Python 3.11 or newer is required (CI tests 3.11 and 3.14).

```
python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` covers every import and includes pytest. The optional
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
```

To build a database, a directory needs spheres and a `vehicle.cfg`. One sphere
makes a toy database:

```
mkdir -p demo_plots/spheres
cp example_data/AS350B3108.nc demo_plots/spheres/
# write demo_plots/spheres/vehicle.cfg from the example in docs/file_formats.md
python build_empirical_database.py demo_plots/spheres demo_plots/toy.nod
```

It warns that the sphere has no `azimuth_reference` and that the hover was
synthesized from the 78.7 kt sphere. Spheres from the 2017 flight test are built
as in [`docs/database_build.md`](docs/database_build.md). Spheres from other
recordings are built with the Python API (`depropagate_hemisphere`, then
`write_aam_hemisphere_netcdf`; see
[`docs/DEVELOPING.md`](docs/DEVELOPING.md#data-flow) and
[THEORY §5](docs/THEORY.md#5-depropagation)).

The tools and plotting scripts, and `AS350_289108_demo.py`, take `-h`. Three
scripts take no options and run as soon as they start, `-h` included:
`hexacopter_vkf_example.py`, `vold_kalman_filter.py` and `as350_flip_check.py`.

## Tools

| Script | Purpose | Needs external data? |
|---|---|---|
| `noise_abatement_2017.py` | Build source spheres from the 2017 Noise Abatement flight test; `--check-tracks` checks the tracks without building | Yes: the 2017 archive |
| `build_empirical_database.py` | Build a NICE-OPS database from a directory of spheres and its `vehicle.cfg` | Your spheres |
| `norah2_to_nod.py` | Build a NICE-OPS database from NORAH2 `.hem` hemispheres | `.hem` files |
| `nc_lambert_ea.py` | Plot a sphere's levels on a Lambert equal-area projection | No (bundled sphere) |
| `plot_projection.py` | Project a sphere onto flat ground and contour the footprint | No (bundled sphere) |
| `fried_egg_plot.py` | Contour ground levels of a directory of spheres over flight condition | Your spheres and `vehicle.cfg` |
| `spectrogram_plot.py` | Spectrogram of one HDF5 or netCDF recording | Your recording |
| `array_planner.py` | Design a linear microphone array, plot its coverage, export KMZ | No |
| `atmomap.py` | Map atmospheric absorption over temperature and pressure | No |
| `ega_plot.py` | Plot excess ground attenuation | No |
| `board_field_plot.py` | Map the sound field around a ground-plane microphone plate | No |
| `AS350_289108_demo.py` | Example: depropagate one AS350 run and compare it with its reference sphere | Yes: AS350 demo data |
| `hexacopter_vkf_example.py` | Example: separate rotor harmonics with the Vold-Kalman filter | No (bundled `.mat`) |
| `vold_kalman_filter.py` | Run as a script: Vold-Kalman filter examples on synthetic signals | No |
| `as350_flip_check.py` | Diagnostic: check the AS350 demo's lateral sign against its reference | Yes: AS350 demo data |

`atmomap.py`, `ega_plot.py` and `board_field_plot.py` always write a file
(defaults `demo_plots/atmomap.pdf`, `demo_plots/ega_plot.pdf` and
`board_field.png` in the current directory); in the first two `-p` means
something else (`--pressure`, `--plot-type`). The other plotting scripts write
the file named by `-o`, open a window without it, and with `-o` and `-p/--plot`
do both. The examples write to `demo_plots/` in the current directory;
`hexacopter_vkf_example.py` must be run from the repository root.

## Data locations

External data and executables are looked up by name. Copy
`local_paths.example.toml` to `local_paths.toml` (git-ignored) and set the
entries you have. A path given on the command line wins, then the environment
variable `PANAM_<NAME>` (for example `PANAM_NORAH2`), then `local_paths.toml`.

| Name | What it is | Used by |
|---|---|---|
| `noise_abatement_2017` | 2017 FAA/NASA Noise Abatement flight test archive | `noise_abatement_2017.py` (or `--root`), track-check tests |
| `as350_demo` | AS350 demo data (`Acoustic/`, `Ambient/`, `AAM/`, `Tracking/`) | `AS350_289108_demo.py` (or `--basepath`), `as350_flip_check.py` |
| `norah2` | EASA NORAH2 public distribution (`Hemispheres/`) | NORAH2 tests |
| `niceops` | NICE-OPS executable | tests that load PANAM databases in NICE-OPS |
| `niceops_ray_geometry` | NICE-OPS ray-geometry executable | refracted-ray tests |

The 2017 archive and the AS350 demo data are not distributed with PANAM. The
NICE-OPS executables come from a NICE-OPS build.

`as350_flip_check.py` also reads the older variable `AS350_DEMO_PATH`, ahead of
`as350_demo`.

## Running the tests

```
pytest                  # from the repository root
pytest -m "not data"    # skip the tests that read the external datasets
```

Tests whose data or NICE-OPS executables are not configured are skipped, so a
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
  hemispheres to a NICE-OPS database.
- [`docs/DEVELOPING.md`](docs/DEVELOPING.md): for developers: code layout,
  conventions, testing.
- [`AGENTS.md`](AGENTS.md): instructions for AI coding tools.

Background: [`docs/notes/README.md`](docs/notes/README.md) indexes the dated
research notes (the ground-plate corrections, the 2017 build findings, the
NORAH2 import checks). They are kept as history and do not track the code.

PANAM is MIT licensed (`LICENSE`). Parts of `panam_acoustics` derive from
python-acoustics (`panam_acoustics/THIRD_PARTY_NOTICES.md`).
