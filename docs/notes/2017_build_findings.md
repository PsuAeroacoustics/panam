# 2017 build findings

> **Status: research record, 2026-10-05.** This note was moved from `docs/database_build.md`
> on 2026-10-10. It is not maintained to match the code. The current how-to and reference
> are [`../database_build.md`](../database_build.md) and
> [`../file_formats.md`](../file_formats.md). Where the original cited files in the companion
> validation repository, the fact is stated here instead.

The track check, the per-run metadata and the heading-frame option were added together on
2026-10-05. None of them changes a level in a database built the default way. The track
check does change *which* runs are built: it refuses B407 285236, a sphere source in the
shipped database.

## The hover-azimuth bug (fixed 2026-10-05)

**This fixes a bug in the hover builds.** `build_sphere(nose_from_heading=True)` used to
convert the heading as if every frame ran along 270 degrees. Hovers fly on the hover
layouts, so every hover sphere built before 2026-10-05 has its azimuths turned: by 9 degrees
for the Amedee aircraft (B407, B206L3, EC130B4), by 178 degrees, nose to tail, for the Eglin
ones (R44, R66). The observation that the B407 hovers' AC heading reads about 9 degrees
clockwise of the flight cards' relative directions may be the same 9 degrees. The
measured-hover analyses in the companion validation study (its B407 and R66 hover
depropagations), and any database whose `hover_correction` came from them, were built with
the turned azimuths and want rebuilding. The B407 hover correction that study applies is not
one of them: it is fitted to NICE-OPS predictions against the measured microphones, not to
these spheres.

On run 285236 the microphone list puts microphones 1, 3, 10, 17 and 36 within 0.6 ft in plan
and 0.5 ft in height of the X, Y and Z their acoustic files carry.

## What the check finds in the 2017 data

`python noise_abatement_2017.py AIRCRAFT --check-tracks [--hovers] [--manifest out.csv]`
runs the check over every steady run's track (and with `--hovers` the hover runs) without
building anything and without the acoustic data (`check_tracks` in code). Run over all six
aircraft on 2026-10-05, from two copies of the archive:

| aircraft | steady runs | refused | no steady window | ends at a level-off | decelerates in window | away from card | hovers | hovers refused |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AS350B3 | 107 | 1 | 6 | 10 | 1 | 1 | 0 | 0 |
| B206L3 | 117 | 0 | 6 | 21 | 0 | 5 | 20 | 0 |
| B407 | 119 | 1 | 0 | 6 | 0 | 1 | 20 | 2 |
| EC130B4 | 132 | 0 | 19 | 5 | 2 | 3 | 20 | 12 |
| R44 | 86 | 0 | 3 | 28 | 0 | 0 | 22 | 14 |
| R66 | 98 | 0 | 7 | 9 | 1 | 1 | 22 | 12 |

The refusals among the flights:

- **B407 285236** (card D19, -7.5 deg; a sphere source in the shipped database, with blend
  weight up to 0.19 on the day-286 approaches): its steady window reaches 66 ft below the
  boards, and the track goes on to 96 ft below them past the array. The steady window of its
  twin 285235 (the same card) stays 40 ft above them. The track's z follows its `alt`
  exactly, so the tracking altitude, not the frame, is wrong. Depropagated, the array saw it
  from above the horizon, and in prediction 91% of microphone 1's SEL came from the upper
  half.
- **AS350B3 291260** (card L4): the "track" is the aircraft parked on the ground 5 km from
  the array; the window's flight path angle is +64 deg, the direction of a few hundredths of
  a knot of drift.

The hover refusals are altitudes at or below the boards: EC130B4's 12 come within 2-5 ft of
them, R44's 14 range from 4 ft above to 24 ft below, R66's 12 from 17 to 33 ft below (the
companion validation study had already found the R66's altitude about 30 ft low, and raises
it before building), and two of the B407's in-ground-effect hovers come within 4.7 ft
(283424) and go 11 ft below (283435). B206L3's hovers pass. A caller that corrects a hover's
altitude by patching `load_track` is checked on the corrected track, but it must shift `alt`
by the same amount as `z`, or the altitude test refuses the run for z departing from its
altitude: that study's R66 hover build raised `z` alone on 2026-10-05, and needs to raise
`alt` with it (or pass `track_check=False`) to rebuild under this check. One that accepts the
altitude as it is passes `track_check=False`.

The decelerating level-off flag fires on 79 of the 659 steady runs, almost all descents whose
window ends where the flare begins, which is why it is a flag. B407 283101 (card D4, -3 deg;
"ABORT; Couldn't see lamps") is the case that motivated it: its window is a level segment at
75 ft, flown at +0.7 deg, that ends 2 s before a decelerating pull-up and turn (2.8 ft/s^2,
the flight path angle rising 2.9 deg). It is flagged twice, for the level-off and for the
card. The dry pass found no window with more than 4.9 ft/s^2 of deceleration.

## Evidence behind the build reference

These were moved from `database_build.md` on 2026-10-10; the thresholds themselves stay
there.

**Track-frame bearings.** The bearings fitted from each track's own latitude and longitude
against its x, over every track of the six aircraft, beside each layout's `true_heading`:

| site and layout | `true_heading` | fitted +x | fitted +y |
| --- | --- | --- | --- |
| Amedee, flight (`AmedeeNoiseAbatMicFullList.csv`) | 270 | 269.7-270.4 | 180.0 |
| Amedee, hover (`AmedeeNoiseAbatStaticList.csv`) | 279 | 279.0 | 189.0 |
| Eglin, flight (`EglinNoiseAbatMicFullList.csv`) | 140 | 139.7-140.0 | 50.1 |
| Eglin, hover (`EglinNoiseAbatStaticList.csv`) | 92.3 | 92.3 | 2.3 |

**Track-check refusal thresholds.** The 10 ft height minimum: the B407's GPS antenna reads
9-10 ft with the aircraft on the ground (runs 283421-2), so anything lower is impossible.
The 10 ft limit on `|alt - ref_elips_ft - z|`: the largest departure in any steady window of
the six aircraft is 3.5 ft.

## Why the per-run metadata is recorded and not used

The flight condition a sphere is labeled with is ground-referenced and its C_T nominal:
`advance_ratio` is the mean horizontal ground speed over the tip speed, and
`thrust_coefficient` is the load factor times one weight coefficient from `vehicle.cfg`. The
flown runs differ from both: the B407's gross weights are 3173-3824 lb against the 2250 kg
(4960 lb) of its vehicle file, so its per-run C_T is 0.65-0.74 of the label (median
-3.08 dB, sd 0.40 dB in 20 lg C_T), and on windy day 284 the headwind was 8.4 kt (median).
Relabeling by either did not improve held-out SEL in the evaluation that motivated this, so
nothing uses them; they are recorded so that air-referenced or per-run labels can be tested
without a rebuild.

The group metadata are attributes rather than variables for NICE-OPS's sake. netCDF-4 opens
every dataset in every group when a file is opened, and NICE-OPS's database load is mostly
that. Six one-element variables per group took the 1432-group `B407_ambient_gated.nod` from
0.72 to 0.95 s per NICE-OPS run (31%, every run, footprint unchanged); the same values as
group attributes, with their units, cost nothing measurable.

The 2017 LIDAR's speed unit was settled as knots on 2026-10-05, as the data report says: the
balloon, the station cups and the aircraft's own turns agree. Its direction reference is not
documented; the companion validation study reads it as true north, which the windy day's
turns support (their wind lies -9 deg [-19, +2] from the LIDAR's, where uncorrected magnetic
directions would put it about 13.5 deg clockwise) without proving it.

## The heading frame: crab and evaluation

The B407 crabbed 10.6 deg (median, max 18.6) on windy day 284 and 4.2 deg on calm day 286. A
pooled fit put the INS heading's own offset near zero (crab = -0.08 deg + k asin(crosswind /
V)), but a later check in the companion validation study (the same 2026-10-05 LIDAR-units
check) found that the offset of the heading box from the airframe changes from day to day,
by 3-4 deg on the B407 and 15 deg on the EC130B4, and that the pilots do not fly zero
sideslip; a crab offset has to be fitted per day, never per aircraft.

On a synthetic pass flown 10 deg crabbed, the heading-filed samples are the track-filed ones
turned by exactly 10 deg, at the same elevations and the same levels.

In the evaluation, moving only NICE-OPS's nose to the heading took the windy day's held-out
SEL from 1.69 to 1.53 dB rms (1.54 with NICE-OPS's `--nose heading` as merged, which the
validation study's paper scores); the heading frame on the build side as well, emulated by
turning each held-out sphere by its own build crab rather than by a rebuild, gave 1.50, and on
the calm day the both-sides version cost +0.025 dB. No heading-frame rebuild had been scored
by 2026-10-05. The rebuild is therefore optional: the track-filed spheres carry the build
days' crab (about -4 deg on the held-out database), which costs well under 0.1 dB against a
correct nose.
