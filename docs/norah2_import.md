# Importing NORAH2 hemispheres into a NICEOPS database

`flight_acoustics.build_database_from_norah2` and the script `norah2_to_nod.py` turn a set of
NORAH2 (HELENA, ECAC Doc 32) `.hem` hemispheres into a NICEOPS `.nod` database, with one
condition per file. PANAM already writes `.hem` files (`write_norah2_hemisphere`, and
`sphere_to_norah2.py` from a sphere file already written) and reads them
(`load_norah2_hemisphere`). The importer is the path from them to NICEOPS. The
equations it applies are in [THEORY §8](THEORY.md#8-norah2-and-aam-interchange). The
database it writes is described in [`file_formats.md`](file_formats.md).

```sh
python norah2_to_nod.py Hemispheres/R44_*.hem -o ~/panam_out/R44_norah2.nod \
    --vehicle R44.json --speed ground-speed
```

Write the database outside the checkout (here `~/panam_out`), so that it is never committed.
A `.hem` carries no rotor or weight, so you must supply the vehicle in one of two ways:

- a NICEOPS vehicle JSON or a PANAM `vehicle.cfg` (`--vehicle`);
- all three of `--tip-speed` (m/s), `--rotor-radius` (m) and `--weight-kg`.

`--speed` is required too; see below. Use the same vehicle file in NICEOPS. NICEOPS
compares tip speed, radius, weight and C_T with the database's and refuses a mismatch over
2 percent.

| option | default | effect |
| --- | --- | --- |
| `--speed {ias-to-tas,ground-speed}` | required | what ACSPEED holds (see [speed](#2-speed-acspeed-is-labeled-ias-but-is-used-as-ground-speed)) |
| `--tas-density` | each file's Tm/Pm/RHm | `ias-to-tas` only: one air density for every file, kg/m³ |
| `--thrust-density` | the vehicle file's density, or 1.225 | the density C_T is formed with, kg/m³ |
| `--radius` | 30.48 | sphere radius, m |
| `--build-temperature`, `--build-pressure`, `--build-humidity` | 293.15 K, 101.325 kPa, 20% | the atmosphere of the broadband EAA |
| `--absorption-band-method {center,arp5534,integrated}` | center | how band absorption is formed in the move to the sphere radius and in the EAA ([THEORY §5.4](THEORY.md#54-the-atmosphere)) |
| `--accept-ground-included` | off | convert FREEFIELD = 0 tables (see [ground](#3-ground-freefield--0-tables)) |
| `--mirror-rotor` | off | reverse the azimuth (see [rotor sense](#4-rotor-sense)) |
| `--fill-empty` | off | fill NOVALUE cells from the nearest direction (see [missing cells](#missing-cells)) |
| `--synthesize-hover` | off | add a hover made from the slowest level file (see [hover](#5-hover)) |
| `--scale-tip-speed-by-rmomega` | off | form each file's advance ratio with its own rotor speed (see [rotor speed](#6-rotor-speed-rmomega)) |
| `--no-spectrum` | off | write dBA and EAA only |
| `--overwrite` | off | replace an existing output |

An argument error exits with status 2, and a conversion error with status 1, leaving no
output.

## The format, briefly

The format is defined in EASA NORAH2 D1.5d Appendix A and ECAC Doc 32 Appendix A. A file
starts with a header of named table constants. Then come levels on a THETAOBSAC x PHIOBSAC
grid (0-180 deg x -90..90 deg in 10 deg steps in every shipped file), in 31 one-third octave
bands from 10 Hz to 10 kHz. The constants that matter here:

| Constant | Meaning | Shipped files |
| --- | --- | --- |
| POLDIST | distance the levels are given at, m | 60 (the hover table: 70) |
| FREEFIELD | 2: free field, with absorption from the center to POLDIST included; 0: ground reflection and absorption included | 2 (the hover table: 0) |
| TAMB, RELHUM, PAMB | atmosphere of that absorption, K, %, Pa | 298.1 K, 70 %, 101325 Pa (ICAO; the importer uses each file's own values, and `write_norah2_hemisphere` writes ICAO's 25 °C as 298.15 K) |
| NOVALUE | no-data marker | -999 |
| ACSPEED | labeled "indicated airspeed", kt | |
| GAMM | flight path angle, deg, negative in descent | |
| Tm, RHm, Pm | measurement atmosphere at 10 m, deg C, %, Pa | Pm 89.9-102.4 kPa |
| RmOmega, PITCH, ROLL, TW, CW, HW | rotor speed, attitude, winds | |

The angles are the AAM/ART ones, which NICEOPS also uses: theta is 0 at the nose, and phi
is 0 straight down and positive to starboard (`test_norah2_and_art_angles_agree`). So no
angle conversion is needed.

## What the conversion does, and why

Four things differ between the formats. Each is worth a decibel or more if it is carried
across unconverted, so none is left to a default.

### 1. Levels: spreading and the absorption to POLDIST

A `.hem` level at POLDIST includes the absorption from the center out to POLDIST, in the
file's reference atmosphere (FREEFIELD = 2). NORAH2 itself takes that absorption back out
before propagating (`SPLrRef` in its manual, section 5.4.2). A NICEOPS sphere keeps the
absorption over its own first radius (30.48 m, as PANAM's databases do) and is otherwise
lossless, and NICEOPS adds absorption from the surface outward. So each band is moved from
POLDIST to the sphere radius with spherical spreading and with the ISO 9613-1 absorption
between the two radii taken back out (THEORY §8). The absorption is evaluated at the band's
center in the file's own TAMB/RELHUM/PAMB, not in assumed ICAO constants. At r = 30.48 m
this is +5.88 dB of spreading, plus 0.18, 0.31, 0.65, 1.96 and 2.92 dB of absorption at 1,
2, 4, 8 and 10 kHz.

**The hazard.** If the file is read as lossless, the ICAO absorption over 60 m stays in the
levels: a few tenths of a dB of LA, and several dB in the top bands (about 6 dB at 10 kHz).
The measured sizes of this and the other hazards below are recorded in
[`notes/norah2_import_checks.md`](notes/norah2_import_checks.md#hazard-sizes).

Two smaller points:

- The first 30.48 m of absorption kept in the sphere is the reference atmosphere's (ICAO),
  not the measurement's, because that is all a `.hem` can still say about it. A PANAM
  hemisphere sent through `write_norah2_hemisphere` and back therefore comes out shifted by
  (alpha_measured - alpha_ICAO) x 30.48 m. The round-trip test checks exactly that.
- NORAH2 removes its absorption by SAE ARP 5534 with a band correction. By default the
  importer uses ISO 9613-1 at the nominal band center, as the PANAM writer does when it adds
  that absorption. `--absorption-band-method arp5534` (`absorption_band_method='arp5534'`)
  uses ARP 5534 instead, in the move to the sphere radius and in the EAA;
  `--absorption-band-method integrated` integrates ISO 9613-1 over each band (THEORY §5.4).
  The band correction is second order in the band's attenuation, which is at most 2.9 dB
  over these 29.5 m.

`radius_m` (`--radius`) changes the sphere radius. Beyond POLDIST, the same expression adds
the absorption between the two radii instead of removing it.

### 2. Speed: ACSPEED is labeled IAS, but is used as ground speed

The format calls ACSPEED indicated airspeed, but NORAH2 looks hemispheres up by the
trajectory's ground speed. Which one a file really holds depends on who wrote it. PANAM's
own `.hem` exports write the ground speed, because its tracking data carries no airspeed.
The importer therefore will not guess, and `speed_mapping` (`--speed`) is required:

- `ias_to_tas` (`ias-to-tas`): ACSPEED is taken as indicated airspeed and converted to true
  airspeed with the density ratio (THEORY §8). The density is the moist-air density of the
  file's measurement atmosphere (Tm, Pm, RHm), or one density for every file
  (`--tas-density`). A file without Tm/Pm/RHm, or with values out of range, is refused
  unless a density is given. The database is written with `speed_reference = 'air'`.
- `ground_speed` (`ground-speed`): ACSPEED is taken as the ground speed, unchanged, as
  NORAH2 uses it. The database is written with `speed_reference = 'ground'`.

Instrument and position error are not modeled, nor is compressibility (0.1 % at 150 kt), so
IAS is taken as equivalent airspeed. The density is the station's, at 10 m. A flight 150 m
higher sees about 1.5 % less, which is 0.75 % in speed. GAMM is kept as stored under either
mapping, though under `ias_to_tas` NICEOPS reads the path angle against the air too. The two
angles' tangents differ by the factor V_ground / V_air. The shipped files' total winds (TW)
are a median 5 kt and at most 18 kt. A 10 kt headwind at 60 kt turns a -6 deg ground path
angle into about -5.1 deg against the air.

**The hazard.** At the shipped files' station pressures (89.9-102.4 kPa), the IAS-to-TAS
factor is 1.00-1.07. At the R44 site it is 1.054, about +4 kt at 80 kt. A 10 kt error at
80 kt changes the advance ratio by 0.023, which moves LA per cell by about a decibel, and by
several in some cells.

An air-referenced database needs the wind at the aircraft in NICEOPS (`--atmosphere` with
wind columns, or `--wind`). Without it, NICEOPS refuses the run unless told
`--speed_reference ground`.

### 3. Ground: FREEFIELD = 0 tables

NORAH2's hover tables are written with FREEFIELD = 0: the ground reflection is in the
levels. NICEOPS applies its own ground model (on by default), so a FREEFIELD = 0 table read
as free field would count the ground twice. Such files are refused unless
`accept_ground_included` (`--accept-ground-included`) is given. With it, they are converted
as free field, with a warning, and the root records `norah2_ground_included_accepted = 1`.
Any other FREEFIELD value is refused as unknown.

The one shipped hover table (`R22_Ingroundhover_0kts_0deg_test.hem`) also has only one axis
(PHIOBSEC around the vehicle at 70 m). It has no elevation axis to make a sphere from, so it
is refused whatever the option; leave it out of the file list. The shipped
`R22_Fullrpmidle_0kts_0deg_test.hem` is a two-axis, FREEFIELD = 2 table. It converts, but it
is an idle on the ground, not a hover, and would be filed at advance ratio 0. Leave it out
too unless that is wanted.

### 4. Rotor sense

NORAH2 represents a helicopter type that has no hemispheres of its own by its class's set.
For the types its manual brackets as "mirrored" (the main rotor turns the other way), the
azimuth is reversed: phi -> -phi (NORAH2 User Manual, Annex C). `mirror_rotor`
(`--mirror-rotor`) applies that reversal before the upper hemisphere is completed. It is a
property of how the set is used, so it is off by default, and it is recorded at the root
(`norah2_mirror_rotor`).

### 5. Hover

NORAH2 has no two-axis hover hemisphere: its hover tables have one axis and are refused
(above). Without a hover, NICEOPS clamps every state slower than the slowest file to that
file. For the R44 set that is 36 kt, so a slow approach spends part of its states clamped
on advance ratio.

`synthesize_hover` (`--synthesize-hover`) adds the hover `build_empirical_database` makes
(THEORY §7.4), from the converted levels. The source is the slowest file within 2 deg of
level flight (`LEVEL_FLIGHT_TOLERANCE`), or the slowest overall, with a warning, if none is.
Its band levels at the sphere radius are averaged fore and aft in energy, and a cell keeps
coverage 1 only where both partners were measured. With `fill_empty`, the filled levels are
averaged, and the coverage is still taken before the fill. SPLA and EAA are recomputed from
the averaged spectrum. The hover is written at advance ratio 0 and at -12, 0 and +12 deg.

Its groups carry `condition_origin = 'synthesized_hover'`, and the source file's
`norah2_source` and constants, so `norah2_ACSPEED` is the source's speed, not 0. The root
records the source as `norah2_hover_source` (`none` without the option). In the R44 set the
source is `R44_Flyover_55kts_0deg.hem`: the 36 kt approaches are not level. A file at 0 kt
and 0 or ±12 deg duplicates a hover condition and is refused with the hover named.
The fore-aft average pairs theta with 180 - theta by index, so the option refuses a theta
grid that is not symmetric about 90 deg. Every shipped file has the symmetric 0-180 deg
grid.

The option is off by default, because the hover is made from a forward-flight condition
(for the R44, a 55 kt flyover), not measured. The average keeps none of the source's fore-aft
directivity.

### 6. Rotor speed: RmOmega

RmOmega is the main rotor speed in percent of nominal. By default it is recorded and does
not change the condition: the advance ratio is formed with the vehicle's nominal tip speed.
`scale_tip_speed_by_rmomega` (`--scale-tip-speed-by-rmomega`) forms each file's advance
ratio with the tip speed its rotor ran at (THEORY §8):

    mu = 0.514444 V / (V_tip RmOmega / 100)

The thrust coefficient stays C_W at the nominal tip speed, and so does the root's
`main_rotor_tip_speed_meters_per_sec`. NICEOPS forms the trajectory's C_T with its vehicle
file's tip speed, and its load-factor scaling would otherwise read the lower rotor speed as
extra thrust. A file without RmOmega, or with one outside 50-130 %, is refused. In the NORAH2
V2.0.74 public package that refuses files in four sets:

| Set | Files refused | RmOmega found |
| --- | --- | --- |
| A109 | 6 of 21 | -999, -479, -335.8, -254.4, -116.1 |
| EC120 | 26 of 28 | mostly 337-410, which looks like rpm (the EC120 rotor turns at about 406-414 rpm); also 143, negative values and -999 |
| EC135 | 1 of 19 | -926.5 |
| R66 | 4 of 20 | 44.0 and negative values |

The option cannot be used on these sets without editing the files' RmOmega. The root records `norah2_scale_tip_speed_by_rmomega` (0 or 1).
The hover is at advance ratio 0 either way.

In the R44 set, RmOmega is 97.5-100 %, so the option raises the advance ratios by up to
2.6 %. The option is off by default, because it moves files within the condition hull:
NICEOPS forms the trajectory's advance ratio with the nominal tip speed, so a file flown at
a lower rotor speed is then found at the advance ratio it flew rather than at its airspeed.

## Missing cells

NOVALUE cells are written as missing (-inf), with coverage 0. NICEOPS reads that as
unmeasured, as it reads PANAM's gated cells: such a cell drops out of a blend where its
neighbors have data, and is silence where none do. A direction with no band left gets dBA
-inf and EAA 0 (`_finite_sphere_levels`).

That matters more here than for PANAM's spheres. Across the 203 two-axis files of the NORAH2
V2.0.74 public package, **45.5 %** of the lower-hemisphere cells are NOVALUE. They are whole
directions, never single bands (40-53 % per aircraft, 33-62 % per file). NORAH2's own
predictor fills an empty bin from its nearest neighbor at use time. `fill_empty`
(`--fill-empty`) does the same in the database (`fill_norah2_empty_cells`). Each band takes
its level from the nearest direction that has one: the largest dot product of body-axis unit
vectors, with ties going to the first cell in (phi, theta) order. The filled cells keep
coverage 0, which is taken before the fill. The option is off by default, because a fill is
an assumption about directions nobody measured.

On the 2017 R44 approaches, filling raised the predicted SEL at the array microphones by
2.7 dB on average ([`notes/norah2_import_checks.md`](notes/norah2_import_checks.md)). That
is what the choice is worth, not a verdict on which is right.

## What the database carries

The importer follows `build_empirical_database` and shares its root writer. Each condition is
written once, at load factor 1, with `fixed_load_factor` set, so NICEOPS scales the level by
20 lg(C_T / C_T,ref) itself. C_T is formed from the vehicle's weight, rotor radius and tip
speed, with the vehicle file's density (`thrust_air_density`, default 1.225 kg/m³;
THEORY §7.1).

- Root: the variables and attributes every database carries
  ([`file_formats.md`](file_formats.md#root)), plus the NORAH2 provenance attributes listed
  there. What is NORAH2-specific: `fixed_load_factor` = 1 with every condition at 1 g;
  `speed_reference` from the mapping; and `azimuth_reference = 'track'`, because NORAH2
  hemispheres are filed against the ground track, with attitude not considered. The build
  atmosphere of the broadband EAA is set with `--build-temperature/-pressure/-humidity`.
- Each group: dBA, EAA, the band spectrum (`--no-spectrum` leaves it out), `coverage`, and
  `DOPPLER_SHIFT_REMOVED = 0`. The group also carries `norah2_source` and the file's
  constants as `norah2_<NAME>` attributes, as read: ACSPEED before the mapping, and RmOmega
  as found, which includes values that are not percentages (see [rotor speed](#6-rotor-speed-rmomega)).
- The upper hemisphere is PANAM's mirror, phi -> 180 - phi, with coverage 0
  (`mirror_phi_to_upper_surface`).

**DOPPLER_SHIFT_REMOVED = 0 is inferred.** NORAH2's documentation describes no step that
removes the Doppler shift, and its predictor applies none. So its spheres are taken to keep
the shift the microphones received, as PANAM's do. EASA does not state it.

NICEOPS reads `azimuth_reference` to choose the sphere's nose (`--nose`). `track` keeps the
nose on the velocity, as these hemispheres were filed.

## What it refuses

Each of these stops the conversion before the output is opened, and the message names the
file and the reason. The database is written to a temporary file beside the output and moved
into place only when it is complete. A failure while writing therefore leaves neither a
partial database nor a damaged earlier one.

- a missing vehicle value (tip speed, radius, weight);
- no `speed_mapping`, or `tas_air_density` with `ground_speed`;
- a missing required constant (POLDIST, FREEFIELD, TAMB, RELHUM, PAMB, ACSPEED, GAMM);
- FREEFIELD = 0 without the option, and any FREEFIELD other than 0 or 2;
- one-axis tables, point-dependent parameters (NPARAD > 0), and a file that ends before its
  header says it should;
- TAMB outside 200-350 K, RELHUM outside 0-100 % or PAMB outside 30-110 kPa (a file in deg C
  or hPa), and a build atmosphere outside the same bounds (given in deg C or Pa);
- `ias_to_tas` without a usable measurement atmosphere or density;
- files with different angle grids or bands, a file that is all NOVALUE, the same file twice
  (also through a symbolic link), or the output among the inputs;
- with `scale_tip_speed_by_rmomega`, a file without RmOmega or with one outside 50-130 %;
- with `synthesize_hover`, a file at the hover's condition (0 kt at 0 or ±12 deg), and a
  THETAOBSAC grid that is not symmetric about 90 deg;
- an existing output, from the command line without `--overwrite` (the function's
  `overwrite` defaults to True);
- two files at one flight condition (NICEOPS takes one sphere per condition);
- two or more conditions that do not vary in both speed and path angle. NICEOPS refuses
  such a database at load, two spheres as well as more (checked with NICEOPS build
  6bbd6bb: two level-flight spheres stop the run).

Two cases convert with a warning instead, because NICEOPS loads them but takes the nearest
condition rather than interpolating:

- one or two conditions (fewer than a triangle needs);
- three or more that lie on one line in (speed, path angle), for example 60 kt / -6 deg,
  80 kt / -3 deg, 100 kt / 0 deg. NICEOPS's triangulation of them has no simplex (it logs
  "0 simplices"), and every query falls back to the nearest condition.

## Not done

- **No hover by default.** Without `synthesize_hover` the importer writes only what the
  files hold, and NICEOPS clamps below the slowest hemisphere (R44: 36 kt).
- **Rotor speed changes only the advance ratio.** With `scale_tip_speed_by_rmomega`, the
  thrust coefficient stays at the nominal tip speed, and the levels are not adjusted for
  rotor speed.
- **The receiver correction cannot be undone.** NORAH builds with a constant
  pressure-doubling G on its plate microphones, where PANAM divides out a boundary-element
  model of the plate on the site ground. The difference is a fraction of a dB rms, with a
  bias of about a decibel. Its rim behavior is inherited from the source
  and cannot be corrected here.

## Tests

`tests/test_norah2_import.py` covers the conversion: a round trip through
`write_norah2_hemisphere`, the file's own reference atmosphere, the geometry and
`mirror_rotor`, missing cells and `fill_empty`, both speed mappings, the synthesized hover
(its source, the energy average, its coverage before a fill, the fallback and the duplicate),
the RmOmega scaling and its refusals, every refusal and warning above, and the command line. With the NORAH2 package configured as `norah2` in
`local_paths`, it also converts the shipped R44 set (`-m data`). The list of checks, the
mutants they catch, and a smoke test through NICEOPS are kept in
[`notes/norah2_import_checks.md`](notes/norah2_import_checks.md).
