# Building spheres and databases from the 2017 flight test

This is how to turn the 2017 FAA/NASA Noise Abatement flight test into AAM source spheres
(`.nc`) and a NICE-OPS database (`.nod`). It is also the reference for the build's track
frame, track check, per-run metadata and azimuth frame. What the files contain is in
[`file_formats.md`](file_formats.md), and the models are in [`THEORY.md`](THEORY.md). The
results of checking the 2017 data, and the heading-frame evaluation, are kept as a record in
[`notes/2017_build_findings.md`](notes/2017_build_findings.md).

## Building a database, step by step

The 2017 data are not distributed with PANAM. The steps below assume you have a copy of the
archive. Run every command from the repository root. The examples write to `~/panam_out`,
outside the checkout, so that no sphere or database lands in the repository.

### 1. Point PANAM at the data

`noise_abatement_2017.py` reads the archive from `--root`. Without it, the script uses the
`noise_abatement_2017` entry of `local_paths.toml`, or the variable
`PANAM_NOISE_ABATEMENT_2017`. Under the root, each aircraft has its own directory:

```text
<root>/
  B407/                                    one directory per aircraft
    B407FullRefList.csv                    run index (*FullRefList.csv): card, layout, times,
                                           true_heading, ref_lat/ref_lon/ref_elips_ft,
                                           mic_loc_file, gross_weight
    *MicFullList.csv, *StaticList.csv      microphone lists: insttype, Latitude, Longitude, Hgt
    B407_AC_Data/<run>AC.csv               tracking, in the track frame (below)
    B407_Acoustic_Data/**/<run>_<mic>_pascal.nc   one file per channel, in any subdirectory
    B407_Weather/B407_Ground_Stations/B407_<day>_*.csv   ground weather stations
                                           (EC130B4: "EC130B4_Ground Stations")
```

The six aircraft directories are `AS350B3`, `B206L3`, `B407`, `EC130B4`, `R44` and `R66`.
Their spheres are named with the legacy file prefix (`AIRCRAFT_SPHERE_PREFIX`: `AS350`,
`Be206`, `Be407`, `EC130`, `RO-44`, `RO-66`) and the run number, for example
`Be407220.nc`. Any other directory name is its own prefix (`build_all(sphere_prefix=...)`
overrides it).

### 2. Check the tracks

```sh
python noise_abatement_2017.py B407 --check-tracks --hovers --manifest ~/panam_out/b407_tracks.csv
```

This step reads only the tracks. It runs the [track sanity check](#track-sanity-check) over
the window of every steady run, and with `--hovers` over the whole record of every hover. It
prints each run that is refused, flagged or has no steady window. With `--manifest` it also
writes the table.

### 3. Build the spheres

```sh
python noise_abatement_2017.py B407 ~/panam_out/spheres/Be407 --manifest ~/panam_out/spheres/Be407/manifest.csv
```

This builds one sphere per steady, non-maneuvering flight run (`is_steady_flight_card`). A
run qualifies when its card's `bank_ang` and `accel_rate` both read as an explicit zero, and
approach (`A`) and hover (`H`) cards are always excluded. `--include-maneuvers` drops the
bank and acceleration test. Each run is gated against a measured ambient run on its own
layout, preferably from the same day. A run that fails is logged and skipped, including one
the track check refuses. `--manifest` records every run's inputs and outcome. The main
options:

| option | default | effect |
| --- | --- | --- |
| `--runs RUN ...` | every steady run | build only these run ids (the reference list's `combined`) |
| `--reference-directory DIR` | none | take each sphere's PHI, THETA and FREQUENCY grid from the legacy sphere of the same name |
| `--norah2-directory DIR` | none | also write each sphere as a NORAH2 `.hem`, and `<prefix>_Triangulation.int` |
| `--winds FILE` | none | the wind at the aircraft per run, recorded only (see [per-run metadata](#per-run-metadata)) |
| `--azimuth-reference` | `track` | `track` or `heading` (see [the heading frame](#the-heading-frame)) |
| `--no-track-check`, `--min-height-above-array-ft` | check on, 10 ft | see [options](#options) |

`python noise_abatement_2017.py -h` lists the depropagation settings, among them
`--band-snr-gate-db`, `--max-absorption-correction-db`, `--min-elevation-deg`,
`--max-propagation-range-ft`, `--point-stride`, `--board-correction` and
`--third-octave-method`. Some of the 2017 release's settings have no command-line option;
see [the 2017 release settings](#the-2017-release-settings).

The command line builds only flight spheres. To build a hover sphere, call
`build_sphere(..., nose_from_heading=True)` from Python.

### 4. Write vehicle.cfg

`build_empirical_database` reads `vehicle.cfg` from the sphere directory, but the 2017
build does not write one. Put one beside the spheres yourself. Its format is in
[`file_formats.md`](file_formats.md#vehiclecfg). The database takes the main rotor's radius
and tip speed from it, along with the weight and density used to form the weight
coefficient.

### 5. Build the database

```sh
python build_empirical_database.py ~/panam_out/spheres/Be407 ~/panam_out/Be407.nod
```

The script takes every `.nc` file in the directory, so the directory must hold only this
build's spheres. It calls `build_empirical_database` with that function's defaults:

- load factors of 0.7, 1.1, 1.5, 1.9 and 2.3 g;
- no extended flight path angles;
- a hover synthesized from the slowest sphere within 2° of level flight, as built;
- spectra stored.

To use the release's choices, call the function from Python (see
[the 2017 release settings](#the-2017-release-settings)).

### 6. Load it in NICE-OPS

```sh
niceops -d ~/panam_out/Be407.nod --self_test -l ~/panam_out/self_test.log
```

If NICE-OPS accepts the database, the log ends with "Self test passed".
`tests/test_niceops_loads_databases.py` runs this same check. Use the same vehicle data in
NICE-OPS as in `vehicle.cfg`.

### Time and disk

The first build at a new sound speed computes the ground-plate tables, which take 15-45 s
each. They are cached under `~/.cache/panam/plate_tables`. The command line's default,
`--point-stride 10`, depropagates every tenth track sample. `build_sphere` and `build_all`
default to 1, which is about ten times slower. `build_all` warms a cloud-synced archive's
cache ahead of each run, and `--no-prefetch` turns that off. Database size depends on the
load factors and the spectra:

- B407 with the eight-rung load-factor ladder: 548 MB.
- With `load_factors=None`: well under 100 MB.
- Spectra take about seven eighths of a database's size.

## The track frame

Every 2017 AC track is in a local frame. Its +x axis runs along a compass bearing, +y is
90 degrees to the left of it, and z is up. z equals the ellipsoidal height `alt` less the
reference list's `ref_elips_ft`. The bearing is the reference list's `true_heading`, and it
is not the same for every layout:

| site and layout | `true_heading` |
| --- | --- |
| Amedee, flight (`AmedeeNoiseAbatMicFullList.csv`) | 270 |
| Amedee, hover (`AmedeeNoiseAbatStaticList.csv`) | 279 |
| Eglin, flight (`EglinNoiseAbatMicFullList.csv`) | 140 |
| Eglin, hover (`EglinNoiseAbatStaticList.csv`) | 92.3 |

`frame_bearing_deg` takes `true_heading`, and where the track allows it also fits the
bearing from the track's latitude and longitude and refuses a disagreement of more than
1 degree; the equations and the conditions are in
[THEORY §3.2](THEORY.md#32-directions). Without `true_heading` it uses the fit alone, and
with neither it refuses. Every compass direction the build uses, such as the INS heading or
a wind direction, goes through it. The bearings fitted from every 2017 track are in
[`notes/2017_build_findings.md`](notes/2017_build_findings.md#evidence-behind-the-build-reference).

`ground_microphone_positions` puts the microphones into the same frame. It reads them from
the layout's microphone list (the reference list's `mic_loc_file`: latitude, longitude and
ellipsoidal height in meters) and uses the WGS84 local radii of curvature at the reference
point.

## Steady windows

A flight run contributes its longest steady window (`steady_window`), and the sphere is
labeled with the window's mean ground speed and mean flight path angle. The criteria and
their defaults are in [THEORY §3.4](THEORY.md#34-steady-segments); the minimum duration,
8 s by default, is set with `--min-steady-duration-s`. A hover uses its whole record.

## Track sanity check

Before it reads any audio, `build_sphere` checks the track over the window it is about to
depropagate from (the steady window, or a hover's whole record). `check_track` returns
refusals and flags. A refusal raises `TrackRefused` (a `ValueError`), and `build_all`
records it as that run's failure, with the reason.

### Refusals

| test | default | why |
| --- | --- | --- |
| height above the ground under the aircraft | at least 10 ft (`min_height_above_array_ft`) | the depropagation geometry needs the aircraft above the microphones; lower is below what a parked aircraft's tracking reads |
| `\|alt - ref_elips_ft - z\|` | at most 10 ft, and finite | z has to be the height the frame says |
| position finite | | |

The ground under each sample is the height of the ground board nearest it in plan, so the
check follows a sloping array: Eglin's boards span 25 ft. When the microphone list cannot
be read, the frame's z = 0 stands in, with a warning. On the Amedee lakebed, z = 0 is
within 2 ft of every board.

### Flags

Flags are logged and written to the manifest's `track_flags` column. They do not stop the
build.

| flag | default | meaning |
| --- | --- | --- |
| deceleration inside the window | d\|v_h\|/dt below -3 ft/s^2 (1.8 kt/s), 1 s running mean | the window holds a speed change |
| the window ends at a decelerating level-off | within 3 s after the window, deceleration beyond 2.5 ft/s^2 and a flight path angle more than 1.5 deg above the window's mean | the condition was left in a flare or pull-up right after the window |
| window away from the card | mean flight path angle more than 2 deg from the card's `fpa` | the window was not flown at the card's condition; the sphere is labeled with the window's own condition either way |

Deceleration is not refused, because `steady_window` holds the speed within 4 kt of the
window's median, which bounds any sustained deceleration inside it. The level-off flag
describes what follows the window; the sphere holds only the window's emission points.
Hovers take no flags: their window is the whole record, and their flight path angle is the
direction of their drift.

### Options

- `build_sphere(track_check=False)`, `--no-track-check`: build without checking.
- `build_sphere(min_height_above_array_ft=...)`, `--min-height-above-array-ft`: the height
  minimum; None turns the height test off.
- `check_track`'s keyword arguments set every other threshold. The module constants
  (`MIN_HEIGHT_ABOVE_ARRAY_FT` and the rest) hold the defaults and their evidence, which is
  also recorded in
  [`notes/2017_build_findings.md`](notes/2017_build_findings.md#evidence-behind-the-build-reference).
- A caller can correct a track's altitude, for example by patching `load_track`. The check
  then runs on the corrected track, so the caller must shift `alt` by the same amount as
  `z`, or the altitude test refuses the run. A caller that accepts the altitude as it is
  passes `track_check=False`.

The manifest gains `track_flags` (or `not checked`) and `min_height_above_array_ft`.

## Per-run metadata

Each sphere records three things about its run:

- its gross weight (the reference list's `gross_weight`);
- its air density at the aircraft;
- when the caller gives them, the wind at the aircraft and the airspeed.

`build_empirical_database` copies them into every condition group the sphere's levels go
to, and adds two labels derived from them (`advance_ratio_air`,
`thrust_coefficient_run`). Nothing uses them. A sphere is labeled by its ground speed and
the nominal C_T of `vehicle.cfg` (THEORY §7.1). The metadata is recorded so that
air-referenced or per-run labels can be tested without a rebuild.
[`file_formats.md`](file_formats.md) lists the variables and attributes.

**Air density.** The build computes moist-air density from the ground stations'
temperature, pressure and humidity (`run_atmosphere`, `air_density_kg_m3`). It carries the
density up hydrostatically to the aircraft's mean height above the boards over the window,
which changes it by about 3% per 1000 ft. Two days have no station data (AS350B3 day 292,
R66 day 231). For their runs the build uses a standard day (20 C, 20% RH, 101.325 kPa) for
absorption and logs a warning naming the run. The recorded density is then NaN, not the
standard day's.

**Wind.** The wind is the caller's, given through `build_sphere(wind=...)`,
`build_all(winds=...)` (a callable or a mapping by run) or `--winds FILE`. `read_winds`
reads the file's columns `run`, `units` and `source`, plus either `east`/`north` or
`speed`/`direction_from_deg`. A wind is given in one of two forms:

- the components the air moves toward;
- a meteorological speed and direction, the direction being the compass bearing (true
  north) it blows from.

Every wind declares its units (`kt`, `m/s`, `ft/s` or `mph`) and its source (`lidar`,
`balloon`, `station`, several joined with `+`, or `none`). Nothing is assumed: a wind
without declared units is refused, and so is an unknown source. The components are stored
in the declared unit; only the airspeed calculation converts it.

For the 2017 data, declare a LIDAR wind in `kt`; its direction is taken as true north,
which is a working reading, not documented (the evidence is in
[`notes/2017_build_findings.md`](notes/2017_build_findings.md#why-the-per-run-metadata-is-recorded-and-not-used)).
If you are not sure of a wind, leave it out rather than guess. Turn a magnetic direction
(the ground stations') to true north before giving it.

## The heading frame

A sphere's azimuth is measured from a direction: 180 is ahead along it, and 90 is to
starboard. `hemigen` uses the ground velocity, so a sphere built from a crabbed run carries
the crab in its azimuths. `build_sphere(azimuth_reference=...)` and `--azimuth-reference`
choose the direction:

- `track` (the default for flight, and the frame of every sphere built before the option
  existed): azimuth from the ground-velocity direction.
- `heading`: azimuth from the tracking file's INS heading, put into the track frame by
  `frame_bearing_deg`. Doppler, the convective Mach number, the spreading and absorption
  ranges and the samples' angular resolution all stay on the ground velocity.
  `depropagate_hemisphere` takes the heading as a separate `track_nose`, which only
  `hemigen`'s azimuth uses.
- A hover (`nose_from_heading=True`) is always heading-filed: None picks `heading` for it,
  and `track` is refused.

Every sphere `build_sphere` writes carries the text attribute `azimuth_reference`.
`build_empirical_database` always writes a root text attribute `azimuth_reference`, set to
the value every source sphere carries. The rules for mixed sources:

- spheres without the attribute are taken as `track`, with a warning;
- the `azimuth_reference` argument, when given, is assumed for those spheres and must agree
  with the ones that carry it;
- spheres filed in different frames are refused, as mixed `DOPPLER_SHIFT_REMOVED` is.

NICE-OPS (`--nose`) reads the attribute to choose its nose. For `heading` it uses the
heading at all speeds, which then needs the run's `--frame_bearing` for an AC track. The
hover groups are heading-oriented on both sides whatever the attribute says.

The root writer every database shares (`_write_database_root`) writes the attribute and
refuses to write a database without one. A NORAH2 import
([`norah2_import.md`](norah2_import.md)) therefore carries it too, as `track`.

## The 2017 release settings

The 2017 release is the B407 database built for the NICE-OPS validation study. It was built
with PANAM's functions, but not with all of their defaults. The table gives both. "PANAM
default" is the default of `build_sphere` (or `build_empirical_database`), with the command
line's default added where it differs.

| setting | PANAM default | release value |
| --- | --- | --- |
| ambient gate (`band_snr_gate_db`), against the measured ambient run | 10 dB (`depropagate_hemisphere` alone: 3 dB, and only when an ambient is given) | 10 dB |
| receiver-response cap (`max_response_correction_db`) | none | 15 dB |
| absorption-correction cap (`max_absorption_correction_db`) | 30 dB (`depropagate_hemisphere` alone: none) | 30 dB |
| range limit (`max_propagation_range_ft`) | 2000 ft (`depropagate_hemisphere` alone: none) | 2000 ft |
| minimum emission elevation (`min_elevation_deg`) | 10° (`depropagate_hemisphere` alone: 0°) | 2° |
| track samples used (`point_stride`) | every sample (1); command line 10 | every tenth (10) |
| board correction (`board_correction`) | `plate_bem` on `SITE_GROUND` | `plate_bem` on `SITE_GROUND` |
| banding | FFT, whole bins (`third_octave_method='fft'`, `tone_aware=False`) | FFT, tone-aware (`tone_aware=True`) |
| propagation (`ray_model`) | straight lines | refracted rays through each run's measured profile (`refracted_rays.external_ray_model`, stratified); straight lines for a run without a profile |
| gridding (`interpolation`, `azi_step`, `elv_step`) | fixed radius `rmax` = 25° on a 10° × 10° grid | adaptive radius (`adaptive_idw_weights` defaults) with the Shepard kernel, `aspect` 5 and `shepard_floor` 1, on a 2° × 2° grid |
| output sphere grid | PHI every 10°, THETA every 5°, 31 bands 10 Hz-10 kHz | the legacy sphere's grid (`reference_directory`) |
| track check (`track_check`) | on | not applied: the shipped database predates the check, and B407 285236, which the check refuses, is one of its spheres |
| azimuth frame (`azimuth_reference`) | `track` | `track` |
| load factors (`load_factors`) | 0.7, 1.1, 1.5, 1.9, 2.3 | 0, 1, 1.1, 1.2, 1.5, 2, 3 and 5 (`ambient_gated`, without spectra; `ambient_gated_spectral`, with), or None (`fixed_load_factor`) |
| extended flight path angles (`extended_flight_path_angles`) | none (the hover is written at -12°, 0° and +12°) | -24° and +35° |
| hover source (`hover_source`) | the slowest sphere within 2° of level flight, as built (received frequencies) | that run rebuilt with `build_sphere(remove_doppler=True)` |
| hover correction (`hover_correction`) | none | supplied by the caller for the B407: a Fourier series in azimuth from the heading, Σ_m (a_m cos mψ + b_m sin mψ) per band, fitted to each ring of measured hovers, blended linearly in elevation between the rings from 3° to 17.5° below the horizon, with term m fading toward the nadir as (cos e / cos e_ring)^m |

The command line does not expose these settings:

- the response cap;
- tone-aware banding;
- the ray model;
- the adaptive gridding;
- the hover source;
- the database options.

A release build therefore calls the functions from Python:

```python
import flight_acoustics as fa
import noise_abatement_2017 as na

settings = dict(band_snr_gate_db=10.0, max_response_correction_db=15.0,
                max_absorption_correction_db=30.0, max_propagation_range_ft=2000.0,
                min_elevation_deg=2.0, point_stride=10, board_correction='plate_bem',
                tone_aware=True, azi_step=2.0, elv_step=2.0,
                interpolation=dict(kernel='shepard', aspect=5.0, shepard_floor=1.0))
na.build_all('B407', f'{OUT}/spheres/Be407', reference_directory=LEGACY_SPHERES,
             manifest_path=f'{OUT}/spheres/Be407/manifest.csv', ray_models=RAY_MODELS, **settings)
# The hover source: the run build_empirical_database would pick, without its Doppler shift.
na.build_all('B407', f'{OUT}/hover_source', runs=[HOVER_RUN], remove_doppler=True,
             reference_directory=LEGACY_SPHERES, ray_models=RAY_MODELS, **settings)
fa.build_empirical_database(f'{OUT}/spheres/Be407', f'{OUT}/Be407_ambient_gated.nod',
                            load_factors=[0.0, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0],
                            extended_flight_path_angles=(-24.0, 35.0),
                            hover_source=HOVER_SOURCE_SPHERE, hover_correction=HOVER_CORRECTION,
                            store_spectrum=False)
```

The names in capitals are the caller's:

- `OUT`: an output directory outside the checkout.
- `LEGACY_SPHERES`: the directory of the shipped legacy spheres, whose grids the new spheres
  take.
- `RAY_MODELS`: maps a run id to its ray model, or to None (`build_all`'s `ray_models`).
  PANAM does not build the per-run atmospheric profiles the rays need.
- `HOVER_RUN`: the run id of the slowest sphere within 2° of level flight, the one
  `build_empirical_database` would pick as the hover source.
- `HOVER_SOURCE_SPHERE`: the path of that run's sphere in `{OUT}/hover_source`.
- `HOVER_CORRECTION`: a callable taking phi and theta (degrees, arrays shaped like the
  sphere's grid) and the band centers (Hz), and returning dB of shape (phi, theta,
  frequency); an optional `description` attribute is recorded in the database. The B407's
  is fitted outside PANAM.

## Code and tests

The code is in `noise_abatement_2017.py` (`build_all`, `build_sphere`, `check_tracks`,
`check_track`, `steady_window`, `frame_bearing_deg`, `ground_microphone_positions`,
`read_winds`) and `flight_acoustics.py` (`build_empirical_database`,
`write_aam_hemisphere_netcdf`, `read_vehicle_data`). The tests are
`tests/test_noise_abatement_2017.py`, `tests/test_track_check.py`,
`tests/test_run_metadata.py` and `tests/test_azimuth_reference.py`.
