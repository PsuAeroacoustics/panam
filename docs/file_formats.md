# File formats

This page describes the files PANAM writes and reads at its boundaries:

- the AAM source sphere (`.nc`), which `build_sphere` writes and `build_empirical_database`
  reads;
- the NICE-OPS database (`.nod`), which `build_empirical_database` and
  `build_database_from_norah2` write;
- `vehicle.cfg`, which the user writes.

How the files are made is in [`database_build.md`](database_build.md) and
[`norah2_import.md`](norah2_import.md). The equations behind the values are in
[`THEORY.md`](THEORY.md).

## AAM sphere netCDF (`.nc`)

`flight_acoustics.write_aam_hemisphere_netcdf` writes one flight condition per file, in
`NETCDF3_CLASSIC`, following the shipped AAM/RNM spheres. AAM 3.1 reads the variables by
position, not by name, so they are declared in the order of `AAM_VARIABLE_ORDER`. Each
scalar also has a singleton dimension of its own name, declared in the same order.

| variable | type, dimensions | `unit` | content |
| --- | --- | --- | --- |
| `BB`, `NB`, `PT` | f4 scalar | | 1, 0, 0 by default |
| `DOPPLER_SHIFT_REMOVED` | f4 scalar | | 0: received frequencies; 1: emitted (`remove_doppler`) |
| `EMPTY_WEIGHT`, `FUEL_WEIGHT`, `LOAD_WEIGHT` | f4 scalar | POUNDS | AAM uses only their sum; by default the run's gross weight as `EMPTY_WEIGHT`, the others 0 |
| `RADIUS` | f4 scalar | FEET | the sphere radius (100 ft for the 2017 build) |
| `FLIGHT_PATH_ANGLE` | f4 scalar | DEGREE | negative in descent |
| `PYLON_ANGLE` | f4 scalar | DEGREE | 90 by default |
| `SPEED` | f4 scalar | KNOTS | the mean ground speed for the 2017 build |
| `XYZ` | f4 (`XYZ` = 3) | FEET | 0, 0, 0 by default |
| `PHI` | f4 (`PHI`) | DEGREE | 0 straight down, positive to starboard, -90 to 90 |
| `THETA` | f4 (`THETA`) | DEGREE | 0 at the nose, 180 at the tail |
| `FREQUENCY` | f4 (`FREQUENCY`) | HERTZ | band centers |
| `AMPLITUDE` | f4 (`PHI`, `THETA`, `FREQUENCY`) | DECIBEL | unweighted band level at `RADIUS`; -999 where missing (below) |
| `MASTTILT` | f4 scalar | DEGREE | after `AMPLITUDE`; AAM does not read it |

The scalar units are blank-padded to 20 characters, as in the shipped spheres. The default
grid is PHI every 10° and THETA every 5°. `build_sphere` uses that grid with the 31 bands
from 10 Hz to 10 kHz. When it is given a `reference_sphere`, it uses the legacy sphere's
grid and bands instead. `AMPLITUDE` carries no `_FillValue`.

Attributes: `title`, and when known `azimuth_reference` (`track` or `heading`; a reader takes
a sphere without it as `track`).

**Per-run metadata.** When `build_sphere` passes `run_metadata`, the file also carries
these f8 scalar variables after `AMPLITUDE`. A variable is left out when its value is
unknown, and `read_run_metadata` reads a missing one back as NaN.

| variable | `unit` | content |
| --- | --- | --- |
| `GROSS_WEIGHT` | POUNDS | the reference list's `gross_weight` |
| `AIR_DENSITY` | KG/M^3 | moist-air density at the aircraft |
| `WIND_ALONG_TRACK` | the declared `wind_units` | wind at the aircraft along the reference direction, positive toward it (a tailwind) |
| `WIND_CROSS_TRACK` | the declared `wind_units` | across it, positive toward its starboard side |
| `AIRSPEED` | KNOTS | mean horizontal airspeed, the mean of \|v_ground - wind\| over the window |

The file also carries these text attributes, each '' when unknown:

| attribute | content |
| --- | --- |
| `wind_source` | `lidar`, `balloon`, `station`, several joined with `+`, or `none` |
| `wind_units` | `kt`, `m/s`, `ft/s` or `mph`, as the caller declared |
| `wind_reference_direction` | `ground track` (flights) or `heading` (hovers, whose ground track is drift) |
| `air_density_source` | `ground stations`, `caller` (an `atmosphere` passed in), or `none` |

**Missing values.** A missing band is stored as -999 (`AAM_MISSING_LEVEL`); see
[how a bin's state travels](#how-a-bins-state-travels).

## The NICE-OPS database (`.nod`)

`build_empirical_database` writes a netCDF-4 file with one group per flight condition and
load factor. `build_database_from_norah2` writes the same layout. Items NICE-OPS requires
are marked R; the rest are optional to a reader.

### Root

Variables:

| variable | type | content |
| --- | --- | --- |
| `same_grid` (R) | byte | true |
| `fixed_load_factor` | byte | true: each condition is stored once, at 1 g, and the reader scales it (`load_factors=None`) |
| `database_version` | i4 | `DATABASE_FORMAT_VERSION` (1) |
| `main_rotor_radius_meters` | f8 | from `vehicle.cfg`, m |
| `main_rotor_tip_speed_meters_per_sec` | f8 | from `vehicle.cfg`, m/s |
| `vehicle_weight_newtons` | f8 | `vehicle.cfg`'s weight × 9.80665, N |
| `shared_grid_and_frequency` | byte | true when every group has the same `phi`, `theta` and `frequency` |
| `phi`, `theta` | f8 (`channels`) | only when shared |
| `frequency` | f8 (`frequency`) | only when shared and spectra are stored |

Attributes:

| attribute | content |
| --- | --- |
| `speed_reference` | `ground` or `air`: what the advance ratio and flight path angle are taken against |
| `azimuth_reference` | `track` or `heading` (see [the heading frame](database_build.md#the-heading-frame)); always written |
| `build_temperature_K`, `build_pressure_kPa`, `build_relative_humidity_percent` | the atmosphere the EAA was computed in (default 293.15 K, 101.325 kPa, 20%) |
| `hover_correction` | when one was applied: its `description`, or `applied` |
| `hover_source` | when given: the hover source sphere's file name |

A NORAH2 import adds the attributes `source_format`, `norah2_speed_mapping`,
`norah2_mirror_rotor`, `norah2_ground_included_accepted`, `norah2_fill_empty` and
`norah2_thrust_air_density_kg_m3`.

### Condition groups

The groups are named `sphere0`, `sphere1`, and so on.

| variable | type, dimensions | units | content |
| --- | --- | --- | --- |
| `radius` (R) | f8 (`radii` = 1) | rotor radii | sphere radius over the rotor radius |
| `rotor_scale` (R) | f8 (`condition` = 1) | m | rotor radius |
| `advance_ratio` (R) | f8 (`condition`) | 1 | 0.514444 V / V_tip, V in knots |
| `flight_path_angle` (R) | f8 (`condition`) | deg | |
| `thrust_coefficient` (R) | f8 (`condition`) | 1 | load factor × the weight coefficient |
| `phi`, `theta` | f8 (`channels`) | deg | unless shared at the root |
| `dBA` (R) | f8 (`channels`) | dB | A-weighted level at the sphere radius, plus 20 lg n for load factor n > 0; -inf for a direction with no energy |
| `EAA` (R) | f8 (`channels`) | dB per 1000 m | excess atmospheric attenuation (THEORY §7.2); 0 where the direction has no energy (`clamp_empty_directions`, default on) |
| `frequency` | f8 (`frequency`) | Hz | with spectra, unless shared |
| `amplitude` | f8 (`PHI`, `THETA`, `frequency`) | dB | unweighted band levels at the sphere radius, not scaled by load factor; -inf where missing |
| `DOPPLER_SHIFT_REMOVED` | i4 scalar | | 0: received frequencies; 1: emitted. The same in every group |
| `coverage` | i1 (`PHI`, `THETA`, `frequency`) with spectra, else (`channels`) | | 1 measured, 0 not (below) |

The variables carry no units attributes. NICE-OPS refuses a database whose `radius` ×
`rotor_scale` differs between groups. Angles are ART angles: `phi` covers [-180, 180) once
the upper surface is mirrored in, and `theta` covers [0, 180]. In a database without
spectra (`store_spectrum=False`), the groups have no `frequency` or `amplitude` variable and
no `PHI`, `THETA` or `frequency` dimension.

**Channel order.** The channels are the row-major flatten of the (`PHI`, `THETA`) grid of
`amplitude`. Channel r·n_theta + j holds the direction of `amplitude[r, j, :]`, and the rows
are sorted by `phi` (`_complete_sphere_with_coverage`). When the channels form such a grid,
NICE-OPS maps each channel's (`phi`, `theta`) to its grid cell. Otherwise it reads
`amplitude` as gridded over the sorted unique angles.

**Coverage.** 1 marks a measured cell, 0 any other; the rules are in
[how a bin's state travels](#how-a-bins-state-travels).

**Per-run metadata.** `build_empirical_database` writes the metadata of each group's source
run as group attributes, not variables, so they add nothing to the time NICE-OPS takes to
open the file; NICE-OPS reads none of them. The numeric attributes are f8, NaN when unknown,
and each has its units in a text attribute `<name>_units`:

| attribute | units | content |
| --- | --- | --- |
| `gross_weight` | N | the run's gross weight |
| `air_density` | kg m-3 | its air density at the aircraft |
| `wind_along_track`, `wind_cross_track` | the group's `wind_units` | as in the sphere |
| `advance_ratio_air` | 1 | 0.514444 `AIRSPEED` / V_tip: the horizontal airspeed over the tip speed |
| `thrust_coefficient_run` | 1 | `gross_weight` / (`air_density` π R² V_tip²): the run's own C_T, not scaled by the group's load factor |

The text attributes are `wind_source`, `wind_units`, `wind_reference_direction`,
`air_density_source`, `source_sphere` (the sphere file's name) and `condition_origin`.
`condition_origin` takes one of three values:

- `measured`: one of the sphere's own conditions, at any load factor;
- `extended_flight_path_angle`: a level sphere re-emitted at an extended flight path angle,
  carrying its source's metadata;
- `synthesized_hover`: the hover synthesized from the slowest level sphere (or
  `hover_source`). It keeps that run's weight, density and C_T, which describe its levels,
  but not the wind or `advance_ratio_air`, which describe a flight condition the hover is
  not.

R and V_tip come from `vehicle.cfg`. A NORAH2 import writes none of these attributes.
Instead, each of its groups carries `norah2_source` and the file's constants as
`norah2_<NAME>`.

### DATABASE_FORMAT_VERSION

`database_version` is `flight_acoustics.DATABASE_FORMAT_VERSION`:

- 0, or absent: written before the version existed. Hover spheres in those files may have
  broken directivity.
- 1: the corrected upper-surface mirror, and the hover averaged fore and aft in energy with
  EAA recomputed from the averaged spectrum.

Adding names leaves the version at 1. Bump it when the on-disk meaning of existing data
changes, together with a matching change in NICE-OPS.

## How a bin's state travels

This is the one full statement of the missing-value rules. A band level is in one of three
states:

| stage | measured | gated (measured, no energy) | not measured |
| --- | --- | --- | --- |
| gridded hemisphere (`depropagate_hemisphere`) | finite dB | -inf | NaN |
| AAM sphere file | finite dB (-999 if below -100 dB) | -999 | -999 |
| after reading (`mask_missing_levels`) | finite dB | -inf | -inf |
| database `amplitude` / `coverage` | finite / 1 | -inf / 0 | -inf / 0 |

**Sphere file.** The writer stores -999 (`AAM_MISSING_LEVEL`) in a band when the direction
is unmeasured (under half of its interpolation weight comes from measured cells), when the
band was gated to zero energy, or when the level is below -100 dB (`minimum_level_db`). The
file does not tell these cases apart, so the distinction between gated and unmeasured holds
only up to the gridded hemisphere.

**Reading.** `mask_missing_levels` treats three conventions as missing and turns each into
-inf (no energy): NaN, which the shipped 2017 spheres use; any value at or below -998; and
any value above 1e34, which older PANAM output used.

**Database.** A `coverage` cell is 1 only where its band level and its direction's EAA were
finite in the source sphere. It is 0 for cells that were gated, masked or below the -100 dB
floor (all -999 in the sphere file), for the mirrored upper surface, for hover cells whose
fore/aft partner was unmeasured, and for NORAH2 NOVALUE cells, including filled ones.
Without spectra, a channel is 1 only if every band is measured; the variable's
`description` attribute says this. A direction with no band left has `dBA` = -inf and
`EAA` = 0.

## vehicle.cfg

`build_empirical_database` (through `read_vehicle_data`) and `fried_egg_plot` read
`vehicle.cfg` from the sphere directory. `norah2_to_nod.py --vehicle` also accepts one
(`read_vehicle_rotor_data`). It is an INI file read by Python's `ConfigParser`. Units:

- radius in m, tip speed in m/s;
- temperature in K, density in kg/m³;
- weight as a mass in kg;
- drag as a flat-plate area in m².

```ini
[Main Rotor]
blades = 4
radius = 5.33
tip speed = 216

[Tail Rotor]
blades = 2
radius = 0.83
tip speed = 216

[Atmosphere]
temperature = 293.15
density = 1.225

[Vehicle]
weight = 2250
drag = 0.82
```

`ConfigParser` does not strip inline comments, so keep values bare. Comments go on lines of
their own. A database build needs every key above. It forms the weight coefficient as
weight × 9.80665 / (density π radius² tip speed²). `norah2_to_nod.py` needs only
`[Main Rotor]` radius and tip speed, `[Vehicle]` weight and `[Atmosphere]` density.

**Optional `[Option]` section.** It names a per-run reference list for the fried-egg plots;
`fried_egg_plot.py -n` (over advance ratio and rotor angle of attack) needs it:

```ini
[Option]
fbar = 0.01
reflist = runs.xlsx
```

`fbar` is the nondimensional flat-plate drag, f/A. `reflist` is an Excel workbook (read with
`openpyxl`), with its path relative to the sphere directory. The workbook has one header
row, then one row per run:

| column | content |
| --- | --- |
| B | run number |
| W | indicated airspeed, kt |
| AH | advance ratio |
| AI | weight coefficient |
| AJ | hover tip Mach number |

`read_vehicle_data` uses this section only when it is given the runs (`project_directory`,
`fried_egg_plot`). It then takes the conditions from the workbook and not from
`[Main Rotor]`. `build_empirical_database` ignores the section.

## How the contract is tested

- `tests/test_niceops_loads_databases.py` runs NICE-OPS's self-test on PANAM-built
  databases, with the default load factors, `load_factors=None` and
  `store_spectrum=False`. It also checks that NICE-OPS's AAM export of a stored condition
  is the source sphere. It skips when the `niceops` executable is not configured.
- `tests/test_database_provenance.py` pins the root attributes, the group variables, their
  types and dimensions, and the coverage of gated and fore/aft-averaged cells. It also
  checks that a default build differs from the older format only by the added items.
- `tests/test_run_metadata.py` pins the per-run metadata in the sphere and the database, and
  that the database's metadata are attributes, not variables.
- `tests/test_write_aam_hemisphere_netcdf.py` pins the AAM variable order, the -999
  sentinel, the masking conventions read back, and the root vehicle fields.
- `tests/test_golden_results.py` pins depropagated levels and the database built from them
  to 1e-6 dB.
