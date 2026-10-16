# Building spheres and databases from the 2017 flight test

This is how to turn the 2017 FAA/NASA Noise Abatement flight test into AAM source spheres
(`.nc`) and a NICEOPS database (`.nod`). It is also the reference for the build's track
frame, track check, per-run metadata and azimuth frame. What the files contain is in
[`file_formats.md`](file_formats.md), and the models are in [`THEORY.md`](THEORY.md). The
results of checking the 2017 data, and the heading-frame evaluation, are kept as a record in
[`notes/2017_build_findings.md`](notes/2017_build_findings.md).

## Building a database, step by step

The 2017 data are not distributed with PANAM. The steps below assume you have a copy of the
archive. Run every command from the repository root. The examples write to `~/panam_out`,
outside the checkout, so that no sphere or database lands in the repository.

### 1. Point PANAM at the data

`tools/noise_abatement_2017.py` reads the archive from `--root`. Without it, the script uses the
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
python tools/noise_abatement_2017.py B407 --check-tracks --hovers --manifest ~/panam_out/b407_tracks.csv
```

This step reads only the tracks. It runs the [track sanity check](#track-sanity-check) over
the window of every steady run, and with `--hovers` over the whole record of every hover. It
prints each run that is refused, flagged or has no steady window. With `--manifest` it also
writes the table.

### 3. Build the spheres

```sh
python tools/noise_abatement_2017.py B407 ~/panam_out/spheres/Be407 --manifest ~/panam_out/spheres/Be407/manifest.csv
```

This builds one sphere per steady, non-maneuvering flight run (`is_steady_flight_card`). A
run qualifies when its card's `bank_ang` and `accel_rate` both read as an explicit zero, and
approach (`A`) and hover (`H`) cards are always excluded. `--include-maneuvers` drops the
bank and acceleration test. Each run is gated against a measured ambient run on its own
layout, preferably from the same day. A run that fails is logged and skipped, including one
the track check refuses. `--manifest` records every run's inputs, settings and outcome, and
each sphere carries its settings in its [provenance stamp](file_formats.md#provenance-stamp).
The main options:

| option | default | effect |
| --- | --- | --- |
| `--runs RUN ...` | every steady run | build only these run ids (the reference list's `combined`) |
| `--reference-directory DIR` | none | take each sphere's PHI, THETA and FREQUENCY grid from the legacy sphere of the same name |
| `--norah2-directory DIR` | none | also write each sphere as a NORAH2 `.hem`, and `<prefix>_Triangulation.int` |
| `--winds FILE` | none | the wind at the aircraft per run, recorded (see [per-run metadata](#per-run-metadata)) |
| `--ray-model` | `straight` | `straight`, or `wind`: depropagate along straight rays carried by each run's wind from `--winds`, uniform over the path; a run without a wind fails |
| `--ray-profiles DIR` | none | depropagate along refracted rays through each run's NICEOPS atmosphere CSV, `<run>.csv` (heights in ft), traced by `niceops_ray_geometry`; straight lines for a run without one |
| `--recipe FILE.toml` | none | take settings from a [build recipe](#build-recipes); options given override it |
| `--azimuth-reference` | `track` | `track` or `heading` (see [the heading frame](#the-heading-frame)) |
| `--no-track-check`, `--min-height-above-array-ft` | check on, 10 ft | see [options](#options) |
| `--tone-aware` | off | file each tone whole in the band of its own frequency |
| `--tone-split` | off | also write each run's tonal and broadband parts (implies `--tone-aware`; see below) |
| `--node-statistics` | off | also write `N_SAMPLES` and `AMPLITUDE_STD` per cell ([file formats](file_formats.md#aam-sphere-netcdf-nc)) |
| `--skip-existing` | off | keep a run whose outputs (sphere, NORAH2 file and parts when asked for) all exist |
| `--skip-unchanged` | off | keep a run whose existing sphere has this build's settings hash and input signature |
| `--jobs N` | 1 | build N runs at once in worker processes; the output is the same as with 1 |
| `-v`, `-q`, `--log-file FILE` | INFO, stderr | log DEBUG too, or only WARNING (`-qq` ERROR); also append the log to FILE |

`python tools/noise_abatement_2017.py -h` lists the depropagation settings, among them
`--band-snr-gate-db`, `--max-absorption-correction-db`, `--min-elevation-deg`,
`--max-propagation-range-ft`, `--point-stride`, `--board-correction` and
`--third-octave-method`. Every keyword of `build_sphere` that has a command-line form has an
option, with the function's default except `--point-stride` (10, the function's 1). `--recipe FILE.toml` reads
settings from a [build recipe](#build-recipes); options given on the command line override
it. [The 2017 release settings](#the-2017-release-settings) are a recipe.

`--skip-existing` and `--skip-unchanged` resume or refresh a build. A run counts as built
when its sphere exists, with its NORAH2 file under `--norah2-directory` and its parts under
`--tone-split`. The NORAH2 file is the one named from the speed and flight path angle in the
sphere's build record; an older `.hem` of the run at another speed label does not count. `--skip-unchanged` also needs the sphere's `panam_settings_sha256` to equal this
build's and its `panam_input_signature` to equal the signature of the run's input files now:
the track, the run and ambient recordings, the reference and microphone lists, the day's
weather station files, the reference sphere and the run's `--ray-profiles` CSV, by name, size
and modification time. A
skipped run keeps its files and is listed in the manifest, with `skipped` set to `exists` or
`unchanged`, and in the NORAH2 triangulation. Its manifest row is the one its original build
wrote, track flags included, read from the sphere's `panam_build_record`, so `--skip-flagged`
still leaves out a run that build flagged. A sphere written before build records were has
`track_flags` set to `unknown: the sphere has no build record`, which `--skip-flagged` also
leaves out. A change to the PANAM code does not count as a change: after updating PANAM,
rebuild without `--skip-unchanged`. Every sphere, NORAH2 file and database is written to a
temporary file and moved into place when complete, so an interrupted build leaves no partial
sphere for either option to keep.

`--jobs N` (`build_all(jobs=N)`) builds N runs at once in processes started with the spawn
method. The plate tables are computed first, in the parent, so the workers read them from
the cache. The parent prefetches each run before it hands it to a worker. Records come back
in run order, so the manifest and the triangulation are those of `--jobs 1`. From Python,
`ray_models` and `winds` must be None, a dict or a module-level function, since they are sent
to the workers. `ray_profiles` and `--ray-profiles`, from the command line or a
recipe, work with any `--jobs`.

The command line selects only flight runs by itself. To build a hover sphere, give its run
with `--runs` and add `--nose-from-heading`.

`--tone-split` (`build_sphere(tone_split=True)`) writes `<name>_tonal.nc` in `tonal/` and
`<name>_broadband.nc` in `broadband/` under the output directory, beside each total sphere.
In energy the two sum to the total. They are ordinary sphere files. They go in directories of
their own because `build_empirical_database` reads every `.nc` in its directory, and parts
beside the total would count each run three times. Where the split falls depends on the tone
detector's `threshold_db`, `lobe` and `floor_bins` ([THEORY §2.2](THEORY.md#22-one-third-octave-bands)). The `.nod` database
format is unchanged.

### 4. Write vehicle.cfg

`build_empirical_database` reads `vehicle.cfg` from the sphere directory, but the 2017
build does not write one. Put one beside the spheres yourself. Its format is in
[`file_formats.md`](file_formats.md#vehiclecfg). The database takes the main rotor's radius
and tip speed from it, along with the weight and density used to form the weight
coefficient.

### 5. Build the database

```sh
python tools/build_empirical_database.py ~/panam_out/spheres/Be407 ~/panam_out/Be407.nod
```

By default the script takes every `.nc` file in the directory. To build from exactly the
spheres a build wrote, pass its manifest; `--exclude` then leaves out named spheres:

```sh
python tools/build_empirical_database.py ~/panam_out/spheres/Be407 ~/panam_out/Be407.nod \
    --manifest ~/panam_out/spheres/Be407/manifest.csv --exclude Be407285236.nc
```

`--manifest` takes the file names of the runs the manifest lists as built and looks for them
in the sphere directory (`noise_abatement_2017.select_from_manifest`); `--skip-flagged` also
leaves out runs whose track check raised a flag. `--sphere FILE` names spheres one at a time
instead, and `--exclude` takes shell patterns as well as names. The database's root attribute
`source_spheres` lists the file names used, and `excluded_spheres` those `--exclude` left out.

NICEOPS takes one sphere per flight condition, so two spheres with the same speed and flight
path angle are refused. `--merge-repeats` energy-averages such repeat runs into one condition
instead (`sphere_statistics.average_spheres`), and `--merge-speed-tolerance KT` also merges
spheres at one flight path angle whose speeds lie within KT of the slowest. The merged
condition's `source_sphere` names every run, and `panam_inputs` lists each merged set as
`merged_spheres`.

Every other option of `build_empirical_database` that takes a plain value has a flag, and
each flag defaults to the function's default (`python tools/build_empirical_database.py -h` lists
them with their units):

- load factors of 0.7, 1.1, 1.5, 1.9 and 2.3 g (`--load-factors`, or `--fixed-load-factor`
  for one 1 g reference that NICEOPS scales);
- no extended flight path angles (`--extended-fpa LO:HI`, deg);
- a hover synthesized from the slowest sphere within 2° of level flight, as built
  (`--hover-source`, `--level-flight-tolerance`; the `--hover-source` path is taken
  relative to the current directory, not the sphere directory);
- spectra stored (`--no-spectrum` leaves them out);
- the EAA computed at 1000 m in 293.15 K, 101.325 kPa and 20% relative humidity
  (`--distance`, `--build-temperature`, `--build-pressure`, `--build-humidity`), with each
  band's absorption taken at its center frequency (`--absorption-band-method arp5534` or
  `integrated`, THEORY §5.4).

The script refuses to replace an existing database without `--overwrite`. The hover
correction is a Python callable and has no flag; a [build recipe](#build-recipes) names it as
`"module:callable"`. `--recipe FILE.toml` takes the recipe's `[database]` settings, and
options given on the command line override them.

### 6. Load it in NICEOPS

```sh
niceops -d ~/panam_out/Be407.nod --self_test -l ~/panam_out/self_test.log
```

If NICEOPS accepts the database, the log ends with "Self test passed".
`tests/test_niceops_loads_databases.py` runs this same check. Use the same vehicle data in
NICEOPS as in `vehicle.cfg`.

### Time and disk

The first build at a new sound speed computes the ground-plate tables, which take 15-45 s
each. They are cached under `~/.cache/panam/plate_tables`, or `--plate-table-directory`;
`--plate-cache list` lists the cached tables and `--plate-cache clear` deletes them. The
command line's default,
`--point-stride 10`, depropagates every tenth track sample. `build_sphere` and `build_all`
default to 1, which is about ten times slower. `build_all` warms a cloud-synced archive's
cache ahead of each run, and `--no-prefetch` turns that off. `--jobs` divides the computation,
not the download: with a cloud-synced archive that is not yet on the disk, the prefetch can
take as long as the build. Database size depends on the
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
in the declared unit; only the airspeed calculation converts it. By default the wind
changes no level. With `--ray-model wind` (`build_sphere(ray_model='wind')`) it also carries
the rays (`refracted_rays.uniform_wind_ray_model`, [THEORY §5.3](THEORY.md#53-refracted-rays)),
turned into the track frame by the run's frame bearing.

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

NICEOPS (`--nose`) reads the attribute to choose its nose. For `heading` it uses the
heading at all speeds, which then needs the run's `--frame_bearing` for an AC track. The
hover groups are heading-oriented on both sides whatever the attribute says.

The root writer every database shares (`_write_database_root`) writes the attribute and
refuses to write a database without one. A NORAH2 import
([`norah2_import.md`](norah2_import.md)) therefore carries it too, as `track`.

## The 2017 release settings

The 2017 release is the B407 database built for the NICEOPS validation study. It was built
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
| gridding (`interpolation`, `azi_step`, `elv_step`) | fixed radius `rmax` = 25° on a 10° × 10° grid | adaptive radius (`adaptive_idw_weights` defaults) with `kernel` `shepard`, `aspect` 5 and `shepard_floor` 1, on a 2° × 2° grid |
| output sphere grid | PHI every 10°, THETA every 5°, 31 bands 10 Hz-10 kHz | the legacy sphere's grid (`reference_directory`) |
| track check (`track_check`) | on | not applied: the shipped database predates the check, and B407 285236, which the check refuses, is one of its spheres |
| azimuth frame (`azimuth_reference`) | `track` | `track` |
| load factors (`load_factors`) | 0.7, 1.1, 1.5, 1.9, 2.3 | 0, 1, 1.1, 1.2, 1.5, 2, 3 and 5 (`ambient_gated`, without spectra; `ambient_gated_spectral`, with), or None (`fixed_load_factor`) |
| extended flight path angles (`extended_flight_path_angles`) | none (the hover is written at -12°, 0° and +12°) | -24° and +35° |
| hover source (`hover_source`) | the slowest sphere within 2° of level flight, as built (received frequencies) | that run rebuilt with `build_sphere(remove_doppler=True)` |
| hover correction (`hover_correction`) | none | supplied by the caller for the B407: a Fourier series in azimuth from the heading, Σ_m (a_m cos mψ + b_m sin mψ) per band, fitted to each ring of measured hovers, blended linearly in elevation between the rings from 3° to 17.5° below the horizon, with term m fading toward the nadir as (cos e / cos e_ring)^m |

The recipe [`docs/recipes/release_2017.toml`](recipes/release_2017.toml) holds these release
values. A recipe is a TOML file of settings that `build_recipe.load_recipe` reads and checks
(see [build recipes](#build-recipes)). Three commands build the release database from it:

```sh
python tools/noise_abatement_2017.py B407 $OUT/spheres/Be407 --recipe docs/recipes/release_2017.toml \
    --manifest $OUT/spheres/Be407/manifest.csv
python tools/noise_abatement_2017.py B407 $OUT/hover_source --recipe docs/recipes/release_2017.toml \
    --hover-source --runs $HOVER_RUN
python tools/build_empirical_database.py $OUT/spheres/Be407 $OUT/Be407_ambient_gated.nod \
    --recipe docs/recipes/release_2017.toml --manifest $OUT/spheres/Be407/manifest.csv \
    --hover-source $HOVER_SOURCE_SPHERE
```

The first builds the flight spheres from the recipe's `[spheres]` section. The second rebuilds
the hover source run with `[hover_source]` (`remove_doppler = true`) over `[spheres]`. The
third writes the `ambient_gated` database from `[database]`. For `ambient_gated_spectral`
set `store_spectrum = true` in a copy of the recipe. For `fixed_load_factor`, replace
`load_factors` with `fixed_load_factor = true`, or add `--fixed-load-factor` to the command.

The recipe leaves out the hover correction, which is fitted outside PANAM. Without it, these
commands build the release database's spheres and settings with an uncorrected hover. To
apply it, put a module that defines it beside a copy of the recipe and name it in
`[database]` as `hover_correction = "module:callable"`.

The names in capitals are the caller's:

- `OUT`: an output directory outside the checkout.
- `HOVER_RUN`: the run id of the slowest sphere within 2° of level flight, the one
  `build_empirical_database` would pick as the hover source.
- `HOVER_SOURCE_SPHERE`: the path of that run's sphere in `$OUT/hover_source`.

Two [`local_paths`](../local_paths.example.toml) entries give the recipe's inputs:

- `b407_legacy_spheres`: the directory of the shipped legacy B407 spheres, whose grids the
  new spheres take (`reference_directory`).
- `b407_ray_profiles`: the directory of per-run NICEOPS atmosphere CSVs, `<run>.csv` with
  heights in ft (`ray_profiles`). PANAM does not build these profiles.

The same build from Python:

```python
from panam import build_recipe
from panam import flight_acoustics as fa
from panam import noise_abatement_2017 as na

recipe = build_recipe.load_recipe('docs/recipes/release_2017.toml')
na.build_all('B407', f'{OUT}/spheres/Be407', manifest_path=f'{OUT}/spheres/Be407/manifest.csv',
             **recipe.sphere_options())
na.build_all('B407', f'{OUT}/hover_source', runs=[HOVER_RUN], **recipe.sphere_options(hover_source=True))
fa.build_empirical_database(f'{OUT}/spheres/Be407', f'{OUT}/Be407_ambient_gated.nod',
                            hover_source=HOVER_SOURCE_SPHERE, hover_correction=HOVER_CORRECTION,
                            **recipe.database_options())
```

`HOVER_CORRECTION` is a callable taking phi and theta (degrees, arrays shaped like the
sphere's grid) and the band centers (Hz), and returning dB of shape (phi, theta, frequency).
An optional `description` attribute is recorded in the database.

## Build recipes

A recipe has up to three sections, each optional:

- `[spheres]`: keywords of `build_all` and `build_sphere`, with the subtable
  `[spheres.interpolation]` for the adaptive gridding's settings;
- `[hover_source]`: the same keywords, over `[spheres]` for the hover source rebuild
  (`tools/noise_abatement_2017.py --hover-source`);
- `[database]`: keywords of `build_empirical_database`.

`load_recipe` checks every key against the signatures of those functions and refuses an
unknown one, naming the file, section and key and suggesting the nearest keyword. Values are
in the functions' units. Some keywords take a recipe form, because TOML holds no Python
objects or None:

- `atmosphere`: a table of `temperature` (K), `pressure` (kPa) and `relative_humidity`
  (percent);
- `hover_correction`: a `"module:callable"` string; the module is imported from the
  recipe's directory or from `sys.path`;
- `winds`: the path of a winds CSV (`read_winds`);
- ray models: only `ray_profiles`, a directory of per-run atmosphere CSVs
  (`profile_ray_models`; stratified refraction, plate receivers, each run's frame bearing);
- `fixed_load_factor = true` in `[database]`: `load_factors=None`;
- `tone_split = true` also sets `tone_aware = true`, as `--tone-split` does, in its own
  section: `[hover_source] tone_split = false` turns off the part spheres of the hover
  source and keeps the tone-aware banding `[spheres] tone_split = true` set; a recipe with
  `tone_split = true` and `tone_aware = false` is refused.

A recipe refuses the dataset (`aircraft`), the outputs (output directory, `manifest_path`,
`norah2_directory`, `samples_path`, the database file), per-run arguments and Python-only
objects (`ground`, a ray model callable). An omitted key keeps the function's default, from
Python and on the command line alike.

A path value may name a `local_paths` entry as `${name}`, for example
`"${b407_legacy_spheres}"`. A relative path without a `${name}` is taken relative to the
recipe's own directory, the directory a `hover_correction` module is imported from, so a
recipe works from any current directory. Each name is looked up only when its section is
used, so a database build does not need the sphere build's data.

With `--recipe`, an option given on the command line overrides the recipe, and the recipe
overrides the command line's defaults. This holds even for an option given at its default
value: `--band-snr-gate-db 10` overrides a recipe's 12. A keyword neither in the recipe nor
on the command line takes the function's default, so a recipe builds the same spheres from
the command line as from Python. The only sphere option this changes is `--point-stride`:
10 without a recipe, 1 with a recipe that omits `point_stride`. `--build-temperature`,
`--build-pressure` and `--build-humidity` each replace only their own field of a recipe's
`atmosphere`. `--hover-source` needs `--runs`, so that a sphere built at emitted
frequencies is never built for every flight run.

## Code and tests

The code is in `panam/build_recipe.py` (`load_recipe`), `panam/noise_abatement_2017.py` (`build_all`,
`build_sphere`, `profile_ray_models`, `check_tracks`, `check_track`, `steady_window`,
`frame_bearing_deg`, `ground_microphone_positions`, `read_winds`, `select_from_manifest`), `panam/database.py` (`build_empirical_database`,
`read_vehicle_data`) and `panam/sphere_io.py` (`write_aam_hemisphere_netcdf`). The tests are
`tests/test_noise_abatement_2017.py`, `tests/test_track_check.py`,
`tests/test_run_metadata.py`, `tests/test_azimuth_reference.py`,
`tests/test_build_empirical_database_cli.py` and `tests/test_build_recipe.py`.
