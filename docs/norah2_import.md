# Importing NORAH2 hemispheres into a NICE-OPS database

Status: implemented (2026-10-05). `flight_acoustics.build_database_from_norah2`
and the script `norah2_to_nod.py` turn a set of NORAH2 (HELENA, ECAC Doc 32)
`.hem` hemispheres into a NICE-OPS `.nod` database, one condition per file.
panam already wrote `.hem` files (`write_norah2_hemisphere`) and read them
(`load_norah2_hemisphere`); this is the path from them to NICE-OPS.

```sh
PYTHONPATH=. python norah2_to_nod.py Hemispheres/R44_*.hem -o R44_norah2.nod \
    --vehicle R44.json --speed ground-speed
```

A `.hem` carries no rotor or weight, so the vehicle is required: a NICE-OPS
vehicle JSON or a panam `vehicle.cfg` (`--vehicle`), or all of `--tip-speed`
(m/s), `--rotor-radius` (m) and `--weight-kg`. `--speed` is required too; see
below. Use the same vehicle file in NICE-OPS: it compares tip speed, radius,
weight and C_T with the database's and refuses a mismatch over 2 percent.

## The format, briefly

EASA NORAH2 D1.5d Appendix A and ECAC Doc 32 Appendix A. A header of named
table constants, then levels on a THETAOBSAC x PHIOBSAC grid (0-180 deg x
-90..90 deg in 10 deg steps in every shipped file) in 31 one-third octave bands,
10 Hz to 10 kHz. The constants that matter here:

| Constant | Meaning | Shipped files |
| --- | --- | --- |
| POLDIST | distance the levels are given at, m | 60 (the hover table: 70) |
| FREEFIELD | 2: free field, with absorption from the center to POLDIST included; 0: ground reflection and absorption included | 2 (the hover table: 0) |
| TAMB, RELHUM, PAMB | atmosphere of that absorption, K, %, Pa | 298.1 K, 70 %, 101325 Pa (ICAO) |
| NOVALUE | no-data marker | -999 |
| ACSPEED | labeled "indicated airspeed", kt | |
| GAMM | flight path angle, deg, negative in descent | |
| Tm, RHm, Pm | measurement atmosphere at 10 m, deg C, %, Pa | Pm 89.9-102.4 kPa |
| RmOmega, PITCH, ROLL, TW, CW, HW | rotor speed, attitude, winds | |

The angles are the AAM/ART ones, which NICE-OPS uses: theta 0 at the nose, phi
0 straight down and positive to starboard (`test_norah2_and_art_angles_agree`).
So no angle conversion is needed.

## What the conversion does, and why

Four things differ between the formats. Each is worth a decibel or more if it
is carried across unconverted, so none is left to a default.

### 1. Levels: spreading and the absorption to POLDIST

A `.hem` level at POLDIST includes the absorption from the center out to
POLDIST, in its reference atmosphere (FREEFIELD = 2). NORAH2 itself takes it
back out before propagating (`SPLrRef` in its manual, section 5.4.2). A NICE-OPS
sphere keeps the absorption over its own first radius (30.48 m, as panam's
databases do) and is otherwise lossless. NICE-OPS adds absorption from the
surface outward. So each band moves by

    L(r) = L(POLDIST) + 20 lg(POLDIST / r) + alpha (POLDIST - r)

alpha being the ISO 9613-1 coefficient (`panam_acoustics.iso_9613_1_1993`) at the
band's center in the file's own TAMB/RELHUM/PAMB, not assumed ICAO constants.
At r = 30.48 m that is +5.88 dB of spreading plus 0.18, 0.31, 0.65, 1.96 and
2.93 dB of absorption at 1, 2, 4, 8 and 10 kHz.

**The hazard.** Read as lossless, the ICAO absorption over 60 m stays in the
levels. On B407 spheres exported by NICE-OPS at 80 kt and -6 deg it is worth a
median 0.33 dB of LA (p5 0.22, p95 0.48, max 0.77). Per band it is 0.37 dB at
1 kHz, 1.32 at 4 kHz and 5.94 at 10 kHz (the hazards evaluation of 2026-10-04).

Two smaller points:

- The first 30.48 m of absorption kept in the sphere is the reference
  atmosphere's (ICAO), not the measurement's. That is all a `.hem` can still
  say about it. A panam hemisphere sent through `write_norah2_hemisphere` and
  back therefore comes out shifted by (alpha_measured - alpha_ICAO) x 30.48 m.
  The round-trip test checks exactly that.
- NORAH2 removes its absorption by SAE ARP 5534 with a band correction.
  The importer uses ISO 9613-1 at the nominal band center, as the panam writer
  does when it adds that absorption. The band correction is second order in
  the band's attenuation, which is at most 2.9 dB over these 29.5 m. It is not
  quantified here.

`radius_m` (`--radius`) changes the sphere radius. Beyond POLDIST, the same
expression adds the absorption between the two instead of removing it.

### 2. Speed: ACSPEED is labeled IAS, but is used as ground speed

The format calls ACSPEED indicated airspeed. NORAH2 looks hemispheres up by the
trajectory's ground speed. Which one a file really holds depends on who wrote
it: panam's own `.hem` exports write the ground speed (its tracking data carries
no airspeed). The importer therefore will not guess. `speed_mapping`
(`--speed`) is required:

- `ias_to_tas` (`ias-to-tas`): ACSPEED is taken as indicated airspeed and
  converted to true airspeed, TAS = IAS sqrt(1.225 / rho). rho is the moist-air
  density of the file's measurement atmosphere (Tm, Pm, RHm), or one density
  for every file (`--tas-density`). A file without Tm/Pm/RHm, or with values
  out of range, is refused unless a density is given. The database is
  written with `speed_reference = 'air'`.
- `ground_speed` (`ground-speed`): ACSPEED is taken as the ground speed,
  unchanged, as NORAH2 uses it. The database is written with
  `speed_reference = 'ground'`.

Instrument and position error are not modeled, nor is compressibility (0.1 %
at 150 kt): IAS is taken as equivalent airspeed. The density is the station's,
at 10 m. A flight 150 m higher sees about 1.5 % less, which is 0.75 % in speed.
GAMM is kept as stored under either mapping, though under `ias_to_tas` NICE-OPS
reads the path angle against the air too. The two angles' tangents differ by
the factor V_ground / V_air. The shipped files' total winds (TW) are a median
5 kt and at most 18 kt. A 10 kt headwind at 60 kt turns a -6 deg ground path
angle into about -5.1 deg against the air.

**The hazard.** At the shipped files' station pressures (89.9-102.4 kPa) the
IAS-to-TAS factor is 1.00-1.07; at the R44 site it is 1.054, about +4 kt at
80 kt. A 10 kt error at 80 kt is a change of 0.023 in advance ratio. Between B407
spheres 10 kt apart that moves LA per cell by a median +0.5 to +1.7 dB, p95
3-7 dB and at most 6.8-13.5 dB (same evaluation).

An air-referenced database needs the wind at the aircraft in NICE-OPS
(`--atmosphere` with wind columns). Without it, NICE-OPS refuses the run unless
told `--speed_reference ground`.

### 3. Ground: FREEFIELD = 0 tables

NORAH2's hover tables are written with FREEFIELD = 0: the ground reflection is
in the levels. NICE-OPS applies its own ground model (on by default), so a
FREEFIELD = 0 table read as free field would count the ground twice. Such files
are refused unless `accept_ground_included` (`--accept-ground-included`) is
given. Then they are converted as free field, with a warning, and the root
records `norah2_ground_included_accepted = 1`. Any other FREEFIELD value is
refused as unknown.

The one shipped hover table (`R22_Ingroundhover_0kts_0deg_test.hem`) also has
only one axis (PHIOBSEC around the vehicle at 70 m). It has no elevation axis to
make a sphere from, so it is refused whatever the option. Leave it out of the
file list. The shipped `R22_Fullrpmidle_0kts_0deg_test.hem` is a two-axis,
FREEFIELD = 2 table. It converts, but it is an idle on the ground, not a hover,
and would be filed at advance ratio 0. Leave it out too unless that is wanted.

### 4. Rotor sense

NORAH2 represents a helicopter type with no hemispheres of its own by its
class's set. For the types its manual brackets as "mirrored" (the main rotor
turns the other way), the azimuth is reversed: phi -> -phi (NORAH2 User Manual,
Annex C). `mirror_rotor` (`--mirror-rotor`) applies that, before the upper
hemisphere is completed. It is a property of how the set is used, so it is
off by default and recorded at the root (`norah2_mirror_rotor`).

## Missing cells

NOVALUE cells are written as missing (-inf), with coverage 0. NICE-OPS reads that
as unmeasured, as it reads panam's gated cells: such a cell drops out of a blend
where its neighbors have data, and is silence where none do. A direction
with no band left gets dBA -inf and EAA 0 (`_finite_sphere_levels`).

That matters more here than for panam's spheres. Across the 203 two-axis files
of the NORAH2 V2.0.74 public package, **45.5 %** of the lower-hemisphere cells are
NOVALUE: whole directions, never single bands (40-53 % per aircraft). NORAH2's
own predictor fills an empty bin from its nearest neighbor at use time.
`fill_empty` (`--fill-empty`) does the same in the database
(`fill_norah2_empty_cells`). Each band takes its level from the nearest
direction that has one (largest dot product of body-axis unit vectors; ties
go to the first cell in (phi, theta) order). The filled cells keep coverage 0.
It is off by default, because a fill is an assumption about directions nobody
measured.

On the 2017 R44 approaches (below), filling raised the predicted SEL at the
array microphones by 2.7 dB on average. That is what the choice is worth, not
a verdict on which is right.

## What the database carries

The importer follows `build_empirical_database` (it shares its root writer).
Each condition is written once, at load factor 1, with `fixed_load_factor` set,
so NICE-OPS scales the level by 20 lg(C_T / C_T,ref) itself. C_T is
W / (rho pi R^2 V_tip^2), with rho the vehicle file's density
(`thrust_air_density`, default 1.225).

- Root: `same_grid`, `fixed_load_factor` = 1, `database_version` = 1,
  `shared_grid_and_frequency`, and `speed_reference` from the mapping.
  `azimuth_reference = 'track'`: NORAH2 hemispheres are filed against the
  ground track, attitude not considered. Also `build_temperature_K`,
  `build_pressure_kPa`, `build_relative_humidity_percent` (the atmosphere of the
  broadband EAA, over 1000 m, default 293.15 K / 101.325 kPa / 20 %, set with
  `--build-temperature/-pressure/-humidity`), and `main_rotor_radius_meters`,
  `main_rotor_tip_speed_meters_per_sec`, `vehicle_weight_newtons`. Provenance:
  `source_format`, `norah2_speed_mapping`, `norah2_mirror_rotor`,
  `norah2_ground_included_accepted`, `norah2_fill_empty`,
  `norah2_thrust_air_density_kg_m3`.
- Each group: dBA, EAA, the band spectrum (`--no-spectrum` leaves it out),
  `coverage`, and `DOPPLER_SHIFT_REMOVED = 0`. The group also carries
  `norah2_source` and the file's constants as `norah2_<NAME>` attributes, as
  read (ACSPEED before the mapping; RmOmega as found, which includes -999 and
  -479 in some A109 files).
- The upper hemisphere is panam's mirror, phi -> 180 - phi, with coverage 0
  (`mirror_phi_to_upper_surface`).

**DOPPLER_SHIFT_REMOVED = 0 is inferred.** NORAH2's documentation describes no
step that removes the Doppler shift, and its predictor applies none. So its
spheres are taken to keep the shift the microphones received, as panam's do.
EASA does not state it.

`azimuth_reference` has no reader in NICE-OPS yet. It records the frame, so a
reader that learns attitude-referenced spheres can tell these apart.

## What it refuses

Each of these stops the conversion before the output is opened, with the file
and the reason named:

- a missing vehicle value (tip speed, radius, weight);
- no `speed_mapping`, or `tas_air_density` with `ground_speed`;
- a missing required constant (POLDIST, FREEFIELD, TAMB, RELHUM, PAMB, ACSPEED, GAMM);
- FREEFIELD = 0 without the option, and any FREEFIELD other than 0 or 2;
- one-axis tables, or point-dependent parameters (NPARAD > 0);
- TAMB outside 200-350 K, RELHUM outside 0-100 %, PAMB outside 30-110 kPa
  (a file in deg C or hPa);
- `ias_to_tas` without a usable measurement atmosphere or density;
- files with different angle grids or bands, a file that is all NOVALUE, the
  same file twice, or the output among the inputs;
- two files at one flight condition (NICE-OPS takes one sphere per condition);
- three or more conditions that do not vary in both speed and path angle,
  which NICE-OPS cannot triangulate. One or two conditions convert, with a
  warning that NICE-OPS will take the nearest one.

## Not done

- **No hover is synthesized.** `build_empirical_database` makes one from the
  slowest level sphere; this importer writes only what the files hold. NICE-OPS
  clamps below the slowest hemisphere (R44: 36 kt). On R44 approach run 229354,
  2,075 of 6,728 states were clamped on advance ratio.
- **Rotor speed.** RmOmega is recorded but does not change the tip speed or
  the advance ratio.
- **The receiver correction cannot be undone.** NORAH builds with a constant
  pressure-doubling G on its plate microphones, where panam divides out a
  boundary-element model of the plate on the site ground. The hazards
  evaluation measured the difference at about 0.2 dB rms, with a -1 dB median
  bias. Its rim behavior is inherited from the source and not correctable here
  (see the harness's `paper/model_review.md`, section 1.2).

## How it was checked

`tests/test_norah2_import.py`:

- **Round trip.** A synthetic hemisphere is written by `write_norah2_hemisphere`
  and converted. The stored levels equal the file's plus the analytic offset to
  1e-9 dB. They equal the original hemisphere plus (alpha_measured - alpha_ICAO)
  x 30.48 m to the file's 0.1 dB rounding.
- **The reference atmosphere is the file's.** Edited TAMB/RELHUM/PAMB move the
  levels accordingly. A radius beyond POLDIST adds the absorption between.
- **Geometry.** The upper hemisphere is the mirror with coverage 0. Starboard
  stays starboard (a +/-6 dB field reads 12 dB between phi = +90 and -90), and
  `mirror_rotor` reverses it.
- **Missing cells.** They come out -inf with coverage 0, EAA 0. `fill_empty`
  agrees with a brute-force nearest search and leaves coverage alone.
- **Speed and attributes.** Both speed mappings give the advance ratio and
  `speed_reference` they should, as do the density override and the moist-air
  density. The root and group attributes are checked, including C_T.
- **Refusals.** Every refusal above has a test, and so does the command line
  (argument errors exit 2, conversion errors exit 1 and leave no output).
- **The shipped R44 set** (`-m data`, with the NORAH2 package configured as
  `norah2` in `local_paths`) converts, level for level and coverage for coverage.

Each of these mutants fails at least one test:

- dropping the kept first radius of absorption;
- the ICAO constants in place of the file's atmosphere;
- `argmin` for `argmax` in the fill;
- `mirror_rotor` as a no-op;
- the IAS-to-TAS ratio inverted.

Every aircraft set in the public package converts under both mappings. The R22
set converts once its one-axis hover table is left out.

**Smoke test through NICE-OPS** (pinned binary 6bbd6bb). The R44 set,
converted with `--speed ground-speed` and the harness's `RO-44.json`, was run on
the 14 R44 approach runs of 2017 at their array microphones. The runs used
`--humidity 50 --no-ground_effect`, against the harness's flat-corrected measured
SEL (`approach/<run>_sel.csv`). This is a plumbing check, not a validation: the
receiver and atmosphere are not the paper's. Every level was finite.

| Database | Mean | RMS | Median | Within 2 dB |
| --- | --- | --- | --- | --- |
| NORAH2 R44, imported | +0.81 | 4.37 | +1.24 | 45 % |
| NORAH2 R44, imported, `--fill-empty` | +3.46 | 5.26 | +2.44 | 42 % |
| shipped `RO-44_fixed_load_factor.nod` | +3.11 | 4.23 | +2.71 | 40 % |

All values are in dB, over 587 microphone-runs. The `ias-to-tas` database
loads too. It is refused without wind, as it should be, and runs under
`--speed_reference ground`.
