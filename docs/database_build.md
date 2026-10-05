# Building spheres and databases from the 2017 flight test

Status: implemented (2026-10-05). This note covers three build-side features
added together, and it is the reference for what a sphere file (`.nc`) and a
NICE-OPS database (`.nod`) carry:

1. a track sanity check that refuses impossible runs before they are
   depropagated, and flags windows worth a second look;
2. per-run metadata (gross weight, air density, wind at the aircraft,
   air-referenced advance ratio, the run's own C_T), recorded and not used;
3. a heading-frame build option, and the `azimuth_reference` attribute that
   says which frame a database's spheres are in.

None of them changes a level in a database built the default way. The track
check does change *which* runs are built: it refuses B407 285236, a sphere
source in the shipped database.

The code is in `noise_abatement_2017.py` (`check_track`, `check_tracks`,
`frame_bearing_deg`, `ground_microphone_positions`, `normalize_wind`,
`build_sphere`) and `flight_acoustics.py` (`hemigen`'s `nose`,
`depropagate_hemisphere`'s `track_nose`, `write_aam_hemisphere_netcdf`'s
`run_metadata` and `azimuth_reference`, `build_empirical_database`). The tests
are `tests/test_track_check.py`, `tests/test_run_metadata.py` and
`tests/test_azimuth_reference.py`.

## The track frame

Every 2017 AC track is in a local frame whose +x runs along a compass bearing,
+y 90 degrees to the left of it, and z up, with z equal to the ellipsoidal
height `alt` less the reference list's `ref_elips_ft`. The bearing is the
reference list's `true_heading`, and it is not the same for every layout.
Fitted from each track's own latitude and longitude against its x (as the
harness's `frame_bearing` does), over every track of the six aircraft:

| site and layout | `true_heading` | fitted +x | fitted +y |
| --- | --- | --- | --- |
| Amedee, flight (`AmedeeNoiseAbatMicFullList.csv`) | 270 | 269.7-270.4 | 180.0 |
| Amedee, hover (`AmedeeNoiseAbatStaticList.csv`) | 279 | 279.0 | 189.0 |
| Eglin, flight (`EglinNoiseAbatMicFullList.csv`) | 140 | 139.7-140.0 | 50.1 |
| Eglin, hover (`EglinNoiseAbatStaticList.csv`) | 92.3 | 92.3 | 2.3 |

`frame_bearing_deg` takes `true_heading`, and when the track carries latitude
and longitude over more than 500 ft of ground it fits the bearing as well and
refuses a disagreement of more than 1 degree. Without `true_heading` it uses the
fit alone; with neither it refuses. Every compass direction the build uses (the
INS heading, a wind direction) goes through it.

**This fixes a bug in the hover builds.** `build_sphere(nose_from_heading=True)`
used to convert the heading as if every frame ran along 270 degrees. Hovers fly
on the hover layouts, so every hover sphere built before 2026-10-05 has its
azimuths turned: by 9 degrees for the Amedee aircraft (B407, B206L3, EC130B4),
by 178 degrees, nose to tail, for the Eglin ones (R44, R66). The observation that
the B407 hovers' AC heading reads about 9 degrees clockwise of the flight cards'
relative directions may be the same 9 degrees. The measured-hover analyses in the
harness (`hover/depropagate_hovers.py`, `hover/r66_hovers.py`) and any database
whose `hover_correction` came from them were built with the turned azimuths and
want rebuilding.

The microphones are put into the same frame from the layout's microphone list
(the reference list's `mic_loc_file`: latitude, longitude and ellipsoidal height
in meters) by `ground_microphone_positions`, on the WGS84 local radii of
curvature at the reference point. On run 285236 the list puts microphones 1, 3,
10, 17 and 36 within 0.6 ft in plan and 0.5 ft in height of the X, Y and Z their
acoustic files carry.

## Track sanity check

`build_sphere` checks the track over the window it is about to depropagate
from (the steady window, or a hover's whole record) before it reads any audio.
`check_track` returns refusals and flags; a refusal raises `TrackRefused` (a
`ValueError`), which `build_all` records as that run's failure with the reason.

### Refusals

| test | default | why |
| --- | --- | --- |
| height above the ground under the aircraft | at least 10 ft (`min_height_above_array_ft`) | the depropagation geometry needs the aircraft above the microphones; the B407's GPS antenna reads 9-10 ft with the aircraft on the ground (runs 283421-2), so anything lower is impossible |
| `|alt - ref_elips_ft - z|` | at most 10 ft | z has to be the height the frame says; the largest departure in any steady window of the six aircraft is 3.5 ft |
| position finite | | |

The ground under each sample is the height of the ground board nearest it in
plan, so a sloping array is followed: Eglin's boards span 25 ft. When the
microphone list cannot be read the frame's z = 0 stands in, with a warning; on
the Amedee lakebed it is within 2 ft of every board.

### Flags

Flags are logged, written to the manifest's `track_flags` column, and do not
stop the build.

| flag | default | meaning |
| --- | --- | --- |
| deceleration inside the window | d\|v_h\|/dt below -3 ft/s^2 (1.8 kt/s), 1 s running mean | the window holds a speed change |
| the window ends at a decelerating level-off | within 3 s after the window, deceleration beyond 2.5 ft/s^2 and a flight path angle more than 1.5 deg above the window's mean | the condition was left in a flare or pull-up right after the window |
| window away from the card | mean flight path angle more than 2 deg from the card's `fpa` | the window was not flown at the card's condition; the sphere is labeled with the window's own condition either way |

Deceleration is not refused: `steady_window` holds the speed within 4 kt of the
window's median, which bounds any sustained deceleration inside it, and the
dry pass below found no window with more than 4.9 ft/s^2. The level-off flag
describes what follows the window; the sphere holds only the window's emission
points. Hovers take no flags (their window is the whole record, their flight
path angle the direction of their drift).

### What the check finds in the 2017 data

`python noise_abatement_2017.py AIRCRAFT --check-tracks [--hovers] [--manifest
out.csv]` runs the check over every steady run's track (and with `--hovers`
the hover runs) without building anything and without the acoustic data
(`check_tracks` in code). Run over all six aircraft on 2026-10-05 (B407 and
AS350B3 from the OneDrive copy, the other four from the USB copy):

| aircraft | steady runs | refused | no steady window | ends at a level-off | decelerates in window | away from card | hovers | hovers refused |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AS350B3 | 107 | 1 | 6 | 10 | 1 | 1 | 0 | 0 |
| B206L3 | 117 | 0 | 6 | 21 | 0 | 5 | 20 | 0 |
| B407 | 119 | 1 | 0 | 6 | 0 | 1 | 20 | 2 |
| EC130B4 | 132 | 0 | 19 | 5 | 2 | 3 | 20 | 12 |
| R44 | 86 | 0 | 3 | 28 | 0 | 0 | 22 | 14 |
| R66 | 98 | 0 | 7 | 9 | 1 | 1 | 22 | 12 |

The refusals among the flights:

- **B407 285236** (card D19, -7.5 deg; a sphere source in the shipped database,
  with blend weight up to 0.19 on the day-286 approaches): its steady window
  reaches 66 ft below the boards, and the track goes on to 96 ft below them past
  the array. The steady window of its twin 285235 (the same card) stays 40 ft
  above them. The
  track's z follows its `alt` exactly, so the tracking altitude, not the frame,
  is wrong. Depropagated, the array saw it from above the horizon, and in
  prediction 91% of microphone 1's SEL came from the upper half.
- **AS350B3 291260** (card L4): the "track" is the aircraft parked on the ground
  5 km from the array; the window's flight path angle is +64 deg, the direction
  of a few hundredths of a knot of drift.

The hover refusals are altitudes at or below the boards: EC130B4's 12 come
within 2-5 ft of them, R44's 14 range from 4 ft above to 24 ft below, R66's 12
from 17 to 33 ft below (the harness's `r66_hovers.py` already found the R66's
altitude about 30 ft low and raises it before building), and two of the B407's
in-ground-effect hovers come within 4.7 ft (283424) and go 11 ft below (283435). B206L3's hovers pass. A caller that corrects a hover's altitude (by
patching `load_track`, as `r66_hovers.py` does) is checked on the corrected
track; one that accepts the altitude as it is passes `track_check=False`.

The decelerating level-off flag fires on 79 of the 659 steady runs, almost all
descents whose window ends where the flare begins, which is why it is a flag.
B407 283101 (card D4, -3 deg; "ABORT; Couldn't see lamps") is the case that
motivated it: its window is a level segment at 75 ft, flown at +0.7 deg, that
ends 2 s before a decelerating pull-up and turn (2.8 ft/s^2, the flight path
angle rising 2.9 deg). It is flagged twice, for the level-off and for the card.

### Options

- `build_sphere(track_check=False)`, `--no-track-check`: build without checking.
- `build_sphere(min_height_above_array_ft=...)`, `--min-height-above-array-ft`:
  the height minimum; None turns the height test off.
- `check_track`'s keyword arguments set every other threshold; the module
  constants (`MIN_HEIGHT_ABOVE_ARRAY_FT` and the rest) hold the defaults and
  their evidence.

The manifest gains `track_flags` (or `not checked`) and
`min_height_above_array_ft`.

## Per-run metadata

The flight condition a sphere is labeled with is ground-referenced and its C_T
nominal: `advance_ratio` is the mean horizontal ground speed over the tip speed,
and `thrust_coefficient` is the load factor times one weight coefficient from
`vehicle.cfg`. The flown runs differ from both: the B407's gross weights are
3173-3824 lb against the 2250 kg (4960 lb) of its vehicle file, so its per-run
C_T is 0.65-0.74 of the label (median -3.08 dB, sd 0.40 dB in 20 lg C_T), and
on windy day 284 the headwind was 8.4 kt (median). Relabeling by either did not
improve held-out SEL in the evaluation that motivated this, so nothing uses
them; they are recorded so that air-referenced or per-run labels can be tested
without a rebuild.

### In the sphere file

`build_sphere` passes `write_aam_hemisphere_netcdf` a `run_metadata` dict,
written as f8 scalar variables (NaN where unknown) and text attributes, and
read back by `flight_acoustics.read_run_metadata`:

| variable | unit | content |
| --- | --- | --- |
| `GROSS_WEIGHT` | POUNDS | the reference list's `gross_weight` |
| `AIR_DENSITY` | KG/M^3 | moist-air density at the aircraft (below) |
| `WIND_ALONG_TRACK` | as declared | wind at the aircraft along the reference direction, positive toward it (a tailwind) |
| `WIND_CROSS_TRACK` | as declared | across it, positive toward its starboard side |
| `AIRSPEED` | KNOTS | mean horizontal airspeed, mean of \|v_ground - wind\| over the window |

| attribute | content |
| --- | --- |
| `wind_source` | `lidar`, `balloon`, `station`, several joined with `+`, or `none` |
| `wind_units` | `kt`, `m/s`, `ft/s` or `mph`, as the caller declared |
| `wind_reference_direction` | `ground track` (flights) or `heading` (hovers, whose ground track is drift) |
| `air_density_source` | `ground stations`, `caller` (an `atmosphere` passed in), or `none` |
| `azimuth_reference` | `track` or `heading` (see below) |

The **air density** is rho = (p - 0.378 e) / (R_d T) from the ground stations'
temperature, pressure and humidity (`run_atmosphere`; e from ISO 9613-1's
saturation pressure), carried up to the aircraft's mean height above the boards
over the window, hydrostatically at the measured temperature (about 3% per
1000 ft). When no station data exist the build still falls back to a standard
day for absorption, as before, but the density is NaN, not the standard day's.

The **wind** is the caller's. `build_sphere(wind=...)`, `build_all(winds=...)`
(a callable or a mapping by run) and `--winds FILE` (`read_winds`: columns
`run`, `units`, `source`, and `east`/`north` or `speed`/`direction_from_deg`)
take either the components the air moves toward or a meteorological speed and
direction (the compass bearing, true north, it blows from), with the units and
the source declared. Nothing is assumed: a wind without declared units is
refused, and so is an unknown source. The 2017 LIDAR's speed unit (knots or
m/s) is still in question in the harness, and its direction reference (true or
magnetic, 13.7 deg apart at Amedee) is not documented, so a caller that is not
sure should leave the wind out rather than guess; a magnetic direction must be
turned to true north before it is given. The components are stored in the
declared unit; only the airspeed converts it.

### In the database

`build_empirical_database` reads each sphere's metadata and writes it into
every condition group its levels go to, as f8 variables over `("condition",)`
with a `units` attribute, NaN when unknown:

| variable | units | content |
| --- | --- | --- |
| `gross_weight` | N | the run's gross weight |
| `air_density` | kg m-3 | its air density at the aircraft |
| `wind_along_track`, `wind_cross_track` | the group's `wind_units` | as in the sphere |
| `advance_ratio_air` | 1 | 0.514444 `AIRSPEED` / V_tip: the horizontal airspeed over the tip speed, as `advance_ratio` is the horizontal ground speed |
| `thrust_coefficient_run` | 1 | `gross_weight` / (`air_density` pi R^2 V_tip^2): the run's own C_T, **not** scaled by the group's load factor |

and the group text attributes `wind_source`, `wind_units`,
`wind_reference_direction`, `air_density_source`, `source_sphere` (the sphere
file's name) and `condition_origin`:

- `measured`: one of the sphere's own conditions (at any load factor);
- `extended_flight_path_angle`: a level sphere re-emitted at an extended flight
  path angle; it carries its source's metadata;
- `synthesized_hover`: the hover synthesized from the slowest level sphere (or
  `hover_source`); it keeps that run's weight, density and C_T, which describe
  its levels, but no wind or `advance_ratio_air`, which describe a flight
  condition the hover is not.

R and V_tip come from `vehicle.cfg`; a configuration read from a reference
workbook (the `[Option]` path) gives no radius, and the two derived labels are
then NaN. NICE-OPS reads none of these variables.

## The heading frame

A sphere's azimuth is measured from a direction: 180 is ahead along it, 90 to
starboard. `hemigen` used the ground velocity, so a sphere built from a crabbed
run carries the crab in its azimuths. The B407 crabbed 10.6 deg (median, max
18.6) on windy day 284 and 4.2 deg on calm day 286; the INS heading has no
offset of its own (crab = -0.08 deg + k asin(crosswind / V)).

`build_sphere(azimuth_reference=...)` and `--azimuth-reference`:

- `track` (the default for flight, and every sphere before): azimuth from the
  ground-velocity direction.
- `heading`: azimuth from the tracking file's INS heading, put into the track
  frame by `frame_bearing_deg`. Doppler, the convective Mach number, the
  spreading and absorption ranges and the samples' angular resolution all
  stay on the ground velocity: `depropagate_hemisphere`
  takes the heading as a separate `track_nose`, which only `hemigen`'s azimuth
  uses. On a synthetic pass flown 10 deg crabbed, the heading-filed samples
  are the track-filed ones turned by exactly 10 deg, at the same elevations and
  the same levels.
- A hover (`nose_from_heading=True`) is always heading-filed; None picks
  `heading` for it and `track` is refused.

Every sphere `build_sphere` writes carries the text attribute
`azimuth_reference`. `build_empirical_database` always writes the root text
attribute `azimuth_reference`: the value every source sphere carries. Spheres
without one are taken as `track`, with a warning; the `azimuth_reference`
argument, when given, is assumed for them and must agree with those that carry
one; and spheres filed in different frames are refused, as mixed
`DOPPLER_SHIFT_REMOVED` is. NICE-OPS reads the attribute to choose its nose (the
heading at all speeds for `heading`). The hover groups are heading-oriented on
both sides whatever it says.

In the evaluation, moving only NICE-OPS's nose to the heading took the windy
day's held-out SEL from 1.69 to 1.53 dB rms; rebuilding the spheres in the
heading frame as well gave 1.50, and on the calm day the both-sides version cost
+0.025 dB. The rebuild is therefore optional: today's track-filed spheres carry
the build days' crab (about -4 deg on the held-out database), which costs well
under 0.1 dB against a correct nose.

## The `.nod` format

What `build_empirical_database` writes. Items NICE-OPS requires are marked R;
the rest are optional to a reader.

Root variables: `same_grid` (R, byte), `fixed_load_factor` (byte),
`database_version` (i4, `DATABASE_FORMAT_VERSION`, 1), `main_rotor_radius_meters`,
`main_rotor_tip_speed_meters_per_sec`, `vehicle_weight_newtons` (f8, from
`vehicle.cfg`), `shared_grid_and_frequency` (byte) and, when it is true, `phi`,
`theta` (over `channels`) and `frequency`.

Root attributes: `speed_reference` (`ground` or `air`), `azimuth_reference`
(`track` or `heading`), `build_temperature_K`, `build_pressure_kPa`,
`build_relative_humidity_percent` (the atmosphere the EAA was computed in), and
when used `hover_correction` and `hover_source`.

One group per condition (`sphere0`, `sphere1`, ...): `radius` (R, over `radii`,
in rotor radii), `rotor_scale`, `advance_ratio` (R), `flight_path_angle` (R),
`thrust_coefficient` (R) over `condition`; `dBA` (R) and `EAA` (R) over
`channels`, with `phi` and `theta` unless shared; `amplitude` over
`(PHI, THETA, frequency)` and `frequency` unless shared, when spectra are kept;
`DOPPLER_SHIFT_REMOVED` (i4); `coverage` (i1); the per-run metadata variables
above, and the group attributes `wind_source`, `wind_units`,
`wind_reference_direction`, `air_density_source`, `source_sphere` and
`condition_origin`. The additions are new names only, so `database_version`
stays 1.
