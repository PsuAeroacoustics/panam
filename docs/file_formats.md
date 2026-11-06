# File formats

This page describes the files PANAM writes and reads at its boundaries:

- the AAM source sphere (`.nc`), which `build_sphere` writes and `build_empirical_database`
  reads;
- the NICEOPS database (`.nod`), which `build_empirical_database` and
  `build_database_from_norah2` write (format 1), and `pt_database.build_pt_database` and
  `write_pt_database` (format 2, pure tones);
- `vehicle.cfg`, which the user writes (`example_data/vehicle.cfg` is an example);
- the Mid-State UAS run file (`.nc`), which `uas_midstate` reads.

PANAM writes each sphere and database to a temporary file beside the output, named
`.<name>.<pid>.tmp`, and moves it into place only when it is complete. A failed or interrupted
write leaves neither a partial file nor a damaged earlier one, and a `*.nc` or `*.nod` glob
never matches the temporary file.

How the files are made is in [`database_build.md`](database_build.md) and
[`norah2_import.md`](norah2_import.md). The equations behind the values are in
[`THEORY.md`](THEORY.md).

## AAM sphere netCDF (`.nc`)

`sphere_io.write_aam_hemisphere_netcdf` writes one flight condition per file, in
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

Attributes: `title`; `sphere_frame`, the frame `PHI` and `THETA` are filed in: `flight_path`
(the writer's default; AAM's body frame with AOA 0, pitched nose-up by `FLIGHT_PATH_ANGLE`) or
`level` (`build_sphere`'s; PANAM's, NICEOPS's and NORAH2's). A reader takes a sphere without it
as `level`, with a warning ([THEORY §3.2](THEORY.md#32-directions)). And when known
`azimuth_reference` (`track` or `heading`; a reader takes a sphere without it as `track`).
Every sphere also carries the [provenance stamp](#provenance-stamp).

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

**Averaged spheres.** `sphere_statistics.write_averaged_sphere` (`tools/sphere_diff.py average`)
writes an energy average of repeat runs in the same layout, with the runs' mean `SPEED`,
`FLIGHT_PATH_ANGLE` and weight (as `EMPTY_WEIGHT`), in the runs' frame; runs filed in different
frames are refused. It adds two optional variables after
everything else. AAM and NICEOPS do not read them, and a reader may ignore them.

| variable | type, dimensions | `unit` | content |
| --- | --- | --- | --- |
| `N_RUNS` | i4 (`PHI`, `THETA`, `FREQUENCY`) | | the runs covering the cell (below) |
| `AMPLITUDE_SD_DB` | f4 (`PHI`, `THETA`, `FREQUENCY`) | DECIBEL | sample standard deviation of the runs' finite levels; -999 where fewer than two are finite |

A run covers a cell when its level there is not NaN. A -999 in a source run is read as no
data by default (`missing='no_data'`), or as zero power (`missing='no_energy'`), since the
file does not say which it is. The provenance stamp hashes the averaging settings (`min_runs`,
`missing` and the speed and flight path angle tolerances) and names the averaged runs in
`panam_inputs`. `write_aam_hemisphere_netcdf` writes such variables through
its `extra_variables` argument and levels already on the ART grid through `art_levels`.

**Node statistics.** `noise_abatement_2017.build_sphere(node_statistics=True)` (CLI
`--node-statistics`; default off) adds two optional variables after everything else, in the
total sphere and in each tonal and broadband part sphere. AAM and NICEOPS do not read them,
and a reader may ignore them.

| variable | type, dimensions | `unit` | content |
| --- | --- | --- | --- |
| `N_SAMPLES` | i4 (`PHI`, `THETA`, `FREQUENCY`) | | depropagated samples behind the gridded node: positive interpolation weight, not missing |
| `AMPLITUDE_STD` | f4 (`PHI`, `THETA`, `FREQUENCY`) | DECIBEL | weighted standard deviation of those samples' levels about the node's energy-mean level; -999 where no sample has positive power |

Each ART point takes the values of the gridded node nearest it in azimuth and in elevation, so
they describe the neighborhood its `AMPLITUDE` was interpolated from, not that value exactly.
The definitions are in [THEORY §6.4](THEORY.md#64-node-statistics).

**Missing values.** A missing band is stored as -999 (`AAM_MISSING_LEVEL`); see
[how a bin's state travels](#how-a-bins-state-travels).

## The NICEOPS database (`.nod`)

`build_empirical_database` writes a netCDF-4 file with one group per flight condition and
load factor. `build_database_from_norah2` writes the same layout. Items NICEOPS requires
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
| `sphere_frame` | `level`, always: flight-path spheres are pitched to level as they are read. NICEOPS refuses any other value |
| `build_temperature_K`, `build_pressure_kPa`, `build_relative_humidity_percent` | the atmosphere the EAA was computed in (default 293.15 K, 101.325 kPa, 20%) |
| `hover_correction` | when one was applied: its `description`, or `applied` |
| `hover_source` | when given: the hover source sphere's file name |
| `source_spheres` | the file names of the spheres used, one per line, sorted (no directory); always written by `build_empirical_database` |
| `excluded_spheres` | when `exclude` was given: the file names it left out, one per line |

`build_empirical_database` also records `load_factors` (f8 array; left out with
`load_factors=None`, which `fixed_load_factor` records) and `extended_flight_path_angles_deg`
(f8 array, degrees; left out when not given). Both builders write the
[provenance stamp](#provenance-stamp).

A NORAH2 import adds the attributes `source_format`, `norah2_speed_mapping`,
`norah2_mirror_rotor`, `norah2_ground_included_accepted`, `norah2_fill_empty`,
`norah2_thrust_air_density_kg_m3`, `norah2_hover_source` (the file a synthesized hover was
made from, or `none`) and `norah2_scale_tip_speed_by_rmomega` (1 when each advance ratio
was formed with the file's RmOmega).

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

The variables carry no units attributes. NICEOPS refuses a database whose `radius` ×
`rotor_scale` differs between groups. Angles are ART angles: `phi` covers [-180, 180) once
the upper surface is mirrored in, and `theta` covers [0, 180]. In a database without
spectra (`store_spectrum=False`), the groups have no `frequency` or `amplitude` variable and
no `PHI`, `THETA` or `frequency` dimension.

**Channel order.** The channels are the row-major flatten of the (`PHI`, `THETA`) grid of
`amplitude`. Channel r·n_theta + j holds the direction of `amplitude[r, j, :]`, and the rows
are sorted by `phi` (`_complete_sphere_with_coverage`). When the channels form such a grid,
NICEOPS maps each channel's (`phi`, `theta`) to its grid cell. Otherwise it reads
`amplitude` as gridded over the sorted unique angles.

**Coverage.** 1 marks a measured cell, 0 any other; the rules are in
[how a bin's state travels](#how-a-bins-state-travels).

**Per-run metadata.** `build_empirical_database` writes the metadata of each group's source
run as group attributes, not variables, so they add nothing to the time NICEOPS takes to
open the file; NICEOPS reads none of them. The numeric attributes are f8, NaN when unknown,
and each has its units in a text attribute `<name>_units`:

| attribute | units | content |
| --- | --- | --- |
| `gross_weight` | N | the run's gross weight |
| `air_density` | kg m-3 | its air density at the aircraft |
| `wind_along_track`, `wind_cross_track` | the group's `wind_units` | as in the sphere |
| `advance_ratio_air` | 1 | 0.514444 `AIRSPEED` / V_tip: the horizontal airspeed over the tip speed |
| `thrust_coefficient_run` | 1 | `gross_weight` / (`air_density` π R² V_tip²): the run's own C_T, not scaled by the group's load factor |

The text attributes are `wind_source`, `wind_units`, `wind_reference_direction`,
`air_density_source`, `source_sphere` (the sphere file's name; for a condition that
`merge_repeats` averaged from repeat runs, every run's file name, one per line, with the runs'
mean numeric values (NaN winds if their wind units or reference directions differ), every
source any run named in `wind_source`, and the first run's other text values) and
`condition_origin`.
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
`norah2_<NAME>`. Its synthesized hover groups (`synthesize_hover`) also carry
`condition_origin = 'synthesized_hover'`, with the source file's `norah2_source` and
constants.

### DATABASE_FORMAT_VERSION

`database_version` is `database.DATABASE_FORMAT_VERSION`:

- 0, or absent: written before the version existed. Hover spheres in those files may have
  broken directivity.
- 1: the corrected upper-surface mirror, and the hover averaged fore and aft in energy with
  EAA recomputed from the averaged spectrum.

- 2: a database with pure-tone sources (`pt_database.PT_DATABASE_FORMAT_VERSION`; below).
  One without them stays at 1.

Adding names leaves the version at 1. Bump it when the on-disk meaning of existing data
changes, together with a matching change in NICEOPS.

### Format 2: pure-tone sources

`pt_database.build_pt_database` (CLI `build_pt_database.py`) writes one from PSU-WOPWOP's
per-rotor PT spheres, one per rotor (`source_tag`) per flight condition. NICEOPS evaluates
every tone at every arrival, with its phase.

Root, beside the format 1 fields:

- `database_version` 2; `sources`, the rotors' tags, comma separated; `main_rotor_source`, the
  tag whose rotor sets the advance ratio, C_T and `rotor_scale`.
- `sphere_frame = flight_path`: the spheres are not pitched to level, which would re-interpolate
  the phase.
- `sphere_levels_lossless = 1`: PSU-WOPWOP applies no absorption, so NICEOPS absorbs from the hub.
- `band_content = broadband`: the band spectrum leaves the tones out. PANAM writes it empty
  (-inf, `dBA` -inf, `EAA` 0, coverage 0), so the tones are the whole level.
- `condition_axes = helicopter` (or `multirotor`, below); `thrust_coefficient_density = run`: C_T
  is W / (rho pi R^2 V_tip^2) at each condition's own `AIR_DENSITY`, the weight given to the build.

Each condition group keeps its format 1 fields and holds one subgroup per source, named by its
tag:

| Variable or attribute | Meaning |
|---|---|
| `center(xyz)` | The hub, m, in the sphere frame: forward, starboard, up (PSU-WOPWOP's XYZ is aft, starboard, up in ft) |
| `shaft_rate` | rad/s, positive counterclockwise seen from above (`main_rotor_rotation`) |
| `harmonic(tone)`, `frequency(tone)` | Shaft orders, and their emission-frame frequencies, Hz. A spectrum over N revolutions (bins every 1/N order, N <= 64) keeps its whole orders; the bins between are dropped |
| `dropped_energy_fraction` | The share of the sphere's energy in those dropped bins (leakage, a loading's slow drift) |
| `amplitude(PHI, THETA, tone)` | dB re 20 µPa at the sphere radius. Shaft orders at PSU-WOPWOP's floor everywhere are left out; a kept tone's floor cells keep the floor level, since NICEOPS reads -inf as missing and fills it from the neighbors |
| `phase(PHI, THETA, tone)` | deg: the cosine phase at emission from the hub, less harmonic × `azimuth` (locked to the rotor) |
| `azimuth` | deg: the rotor azimuth, in its own sense of rotation, the phase was locked to (ROTOR_AZIMUTH); NICEOPS's default `--rotor_azimuth` |
| `phase_reference` | `rotor_azimuth` |
| `sync_group` | Rotors predicted at one shaft rate and one phase reference time share one (`shaft`, or `shaft_<lowest tag>` when there are several such sets); absent for a free-running rotor, and for one whose sphere records no phase reference time |
| `source_sphere`, `phase_reference_time_s` | The sphere file, and its phase reference time, s |
| `sphere_speed_kt` | The speed its own sphere was predicted at; with `speed_tolerance_kt`, spheres a hair apart in speed and flight path angle are one condition at their mean |

No upper half is mirrored, no grid is changed, no hover is synthesized and no load-factor
ladder is written. Spheres that share one air density give one C_T and a `fixed_load_factor`
database; spheres predicted in different air give each condition the C_T of its own density, and
the database is not `fixed_load_factor` (C8 would fail). `speed_reference` is what the spheres
record (they must agree), or the builder's argument for spheres that record none (default `air`). At a full sphere's seam the +180° phi row is dropped
only when its level and phase repeat the -180° row; one that differs is refused. The groups
carry no `condition_origin`: a predicted condition is a stored one of its own.

#### Measured sources

`pt_database.write_pt_database` writes the same layout from sources built in memory;
`pt_database.measured_pt_sphere` builds one from measured tones (`rotor_tones.grid_tones`, the
tones of each rotor demodulated by its logged angle). A measured source differs from a predicted
one in these ways:

- its shaft orders are given, and a cell the measurement never reached is -inf (NICEOPS fills it
  from its neighbors), not a floor level;
- `azimuth` is 0: the phase is the tone's with the rotor's blades at their reference (a prop along
  its arm, say); `phase_reference_time_s` is NaN, and there is no `sync_group` unless one is given;
- `phase_reference` is `rotor_azimuth` where the rotor's angle was logged, else `relative` or `none`;
- an order with no value anywhere is left out, and `speed_reference` defaults to `ground`;
- the root's `azimuth_label`, when present, says what the azimuths are relative to.
  `relative_within_sense` means that one timing offset common to every run is unknown. It turns
  every rotor's blade angle by the same amount in that rotor's own sense of rotation. For rotors at
  one speed that is a common phase, so relative azimuths in these labels (what a controller reading
  the same angles commands) predict the field as measured; only the physical angle between blades
  of opposite senses is uncertain;
- the rotor's own measured state may be written beside it, each variable with `units` and a
  `description` (a scalar, or a vector over `xyz` in the sphere frame): its logged speed, its
  hub's air velocity, its disk's normal, an estimate of its thrust.

`pt_database.fitted_pt_sphere` builds a measured source from fitted directivities instead
(`rotor_tones.fit_directivities` or `fit_summed_directivities`: spherical harmonics about the
shaft, spinning modes of azimuthal order k). The grid holds the expansion's level and phase, and
the coefficients are kept beside it as provenance:

- `directivity_real`, `directivity_imag` (`tone`, `coefficient`), Pa: the complex peak pressure at
  the sphere radius is the sum over l <= `degree` and k = -l..l, in that order, of c_lk Y_l^k
  (orthonormal complex spherical harmonics; polar angle from the down axis, azimuth from the nose
  toward starboard). `directivity_real` carries `degree` and, when the fit restricted them, the
  azimuthal orders fitted for each shaft order (`azimuthal_orders`, `"2: 1,2,3; 4: 3,4,5"`), or
  `model`, a description, when the fit's model was given as text;
- `shaft_rate` is the speed the fit refers to. With the root's `tone_speed_exponent` (nu, given to
  `write_pt_database`), NICEOPS scales each tone's pressure by the run's shaft rate over it to the
  power nu where a run turns the rotor at another speed (nu = 3: the level as rpm to the sixth).
- `incoherent_amplitude` (`PHI`, `THETA`, `tone`), dB re 20 µPa rms at the sphere radius: what
  the rotor radiates at that shaft order without a stable phase. Anywhere more than 100 dB below
  the tone's peak (where the fit is zero, say) it is 100 dB below it: quiet, which NICEOPS keeps,
  not -inf, which it fills from the neighboring cells. An order without one is -inf throughout
  (`rotor_tones.fit_incoherent_directivities`, fitted to what the phase-locked fit leaves). NICEOPS
  adds it in energy, each source alone. The fit beside it is `incoherent_energy` (`tone`, `knot`),
  Pa^2: the mean-square peak pressure, piecewise linear in the polar angle from the down axis
  between `knots_deg`.

NICEOPS reads the grids only.

#### Multirotor conditions

With `condition_axes = multirotor` (`--condition-axes multirotor`) the conditions are indexed by
momentum similarity instead of advance ratio, flight path angle and C_T. With N rotors of radius R,
weight W, density rho at the condition and C_Th the hover thrust coefficient per rotor:

    v_h = sqrt(W / (2 rho N pi R^2)),   Omega_h R = sqrt(W / (N rho pi R^2 C_Th))

Root attributes: `rotor_count` (N), `rotor_radius_m` (R, every rotor's sphere must give the same
MAIN_ROTOR_RADIUS) and `hover_thrust_coefficient` (C_Th, given to the build or taken from its hover
condition, speed 0, with each rotor carrying W / N at the rms of the sources' shaft rates).

Each condition group carries these over `condition`, in place of `advance_ratio`,
`flight_path_angle` and `thrust_coefficient`:

| Variable | Meaning |
|---|---|
| `inplane_speed_ratio` | x: the air speed's part perpendicular to the thrust axis, over v_h |
| `axial_speed_ratio` | z: the part along it, over v_h, positive in climb |
| `hover_tip_mach` | M_h = Omega_h R / c, c the sphere's `speed_of_sound_m_s` |
| `air_density`, `weight_n` | rho (kg m-3) and W (N) of the condition |
| `airspeed`, `climb_rate` | m s-1 |

`fixed_load_factor` is 0: NICEOPS refuses it in multirotor axes, whose load factor enters
through v_h and Omega_h rather than a C_T law. The thrust axis is taken as vertical (load factor 1,
no attitude) when the axes are formed. Each source also carries
`speed_ratio`, |`shaft_rate`| / Omega_h: its share of the trim at that condition.

### Checking a database

`nod_database.validate_nod(path)` returns one violation per broken rule, with the group it
is in. The R rules are what NICEOPS requires. The C rules are the rest of this contract,
which the writers keep; a reader may rely on them.

1. **R1.** The root has `same_grid`, and it is true.
2. **R2.** There is at least one group, and every group has `radius`, `rotor_scale`,
   `advance_ratio`, `flight_path_angle` and `thrust_coefficient`, each one finite value.
3. **R3.** Every group has `dBA` and `EAA` over its `channels` dimension.
4. **R4.** Every group's channels have directions: `phi` and `theta`, one value per channel,
   in the group or, when `shared_grid_and_frequency` is true, at the root.
5. **R5.** `radius` × `rotor_scale` is the same in every group.
6. **R6.** `EAA` is finite. `dBA` is finite or -inf, never NaN or +inf.
7. **C1.** `database_version`, when present, is an integer from 0 to
   `DATABASE_FORMAT_VERSION`.
8. **C2.** `speed_reference` and `azimuth_reference`, when present, take the values listed
   above.
9. **C3.** A group with `amplitude` has its band centers, in the group or shared at the
   root. A group's own `phi`, `theta` or `frequency` equals the shared one at the root.
10. **C4.** `amplitude` is over (`PHI`, `THETA`, `frequency`), with PHI × THETA = channels and
    one band center per band, and the channels are its row-major flatten, rows sorted by
    `phi` (channel order, above).
11. **C5.** `amplitude` is finite or -inf, never NaN or +inf.
12. **C6.** `coverage` is 0 or 1, over the dimensions given above, and is 1 only where the
    level it marks (the band, or `dBA` without spectra) is finite.
13. **C7.** A direction with no finite band has `dBA` = -inf and `EAA` = 0.
14. **C8.** With `fixed_load_factor` set, `thrust_coefficient` is the same in every group.
15. **C9.** `DOPPLER_SHIFT_REMOVED`, when present, is 0 or 1 and the same in every group.
16. **C10.** Every `rotor_scale` equals `main_rotor_radius_meters`, when present.
17. **C11.** The groups are named `sphere0` to `sphere<N-1>`.
18. **C12.** `condition_origin`, when present, is one of the three values above.
19. **C13.** `phi` is in [-180, 180) deg and `theta` is in [0, 180] deg (ART angles, above).

A database listing `sources` (format 2) also keeps the T rules:

20. **T1.** Every tone is a whole shaft order of its source's `shaft_rate` (1e-5 relative).
21. **T2.** Every condition holds exactly the sources the root lists.
22. **T3.** Every source declares `phase_reference` (`rotor_azimuth`, `relative` or `none`),
    and has a finite `phase` wherever its `amplitude` is finite (unless `none`).
23. **T4.** `database_version` is 2 and `band_content` is `broadband` or `total`.
24. **T5.** `amplitude` and `phase` are over (`PHI`, `THETA`, `tone`), on the condition's grid.

C1 then allows `database_version` up to 2.

A database with `condition_axes` also keeps the M rules:

25. **M1.** `condition_axes` is `helicopter` or `multirotor`. A multirotor root gives
    `rotor_count` (a whole number >= 1), `rotor_radius_m` and `hover_thrust_coefficient`,
    all positive.
26. **M2.** Every multirotor condition carries `inplane_speed_ratio`, `axial_speed_ratio`,
    `hover_tip_mach`, `air_density`, `weight_n`, `airspeed` and `climb_rate` (and R2's `radius`
    and `rotor_scale`; the helicopter variables are not required).
27. **M3.** x v_h is the in-plane part of `airspeed` and z v_h is `climb_rate`, v_h from the
    group's own weight and density (1e-6 relative).
28. **M4.** Every source has a positive `speed_ratio`, and |`shaft_rate`| / `speed_ratio` is the
    condition's Omega_h (1e-6 relative).

`load_nod` reads a multirotor condition with `advance_ratio` and `thrust_coefficient` NaN,
`speed_kt` from `airspeed`, `flight_path_angle_deg` its climb angle, and `multirotor_axes`
(x, z, M_h), which is also its `key`.

### Reading a database

`nod_database.load_nod(path)` returns the conditions in group order, each with its speed in
knots (from `advance_ratio` and `main_rotor_tip_speed_meters_per_sec`), flight path angle,
thrust coefficient, sphere radius, channel directions, `dBA`, `EAA`, band centers,
`amplitude` and `coverage`, whichever layout the file uses. The load factor is 1 in a
`fixed_load_factor` database. Otherwise it is C_T / C_W, and the caller passes C_W, since
the file stores no density to form it. Levels are returned as stored, -inf where missing.
`measured_amplitude` sets every coverage-0 cell to NaN. The file stores a gated cell and an
unmeasured one alike ([how a bin's state travels](#how-a-bins-state-travels)), so a reader
cannot restore the gated cell's -inf.

`nod_database.compare_databases(a, b)` matches conditions by advance ratio, flight path
angle and thrust coefficient. It reports the root variables and attributes that differ (not
`panam_created_utc` and `panam_inputs`, which name the build rather than its content), the
conditions in only one file, and for each matched condition the group scalars that differ
(`radius_m`, `rotor_scale_m`, `doppler_shift_removed`, `condition_origin`), the per-band
difference of the energy summed over the cells finite in both, the largest level differences, and the
coverage gained and lost. `tools/compare_databases.py` prints the same report.

`tools/nod_tool.py` inspects one database: `list` prints a row per condition (its coverage
fraction is of the lower hemisphere, `|phi| <= 90`, the part a measurement can cover),
`plot` maps the lower hemisphere of a condition's `dBA`, `EAA`, one band of `amplitude`
or `coverage` on a Lambert projection,
`envelope` plots the conditions in speed and flight path angle by `condition_origin`, and
`check` prints the `validate_nod` violations and exits with status 1 when there is one.

## Provenance stamp

The AAM sphere and the database root carry these optional text attributes, written by
`provenance.build_provenance` and `provenance.write_provenance`. No reader requires them, and
adding them left `DATABASE_FORMAT_VERSION` at 1. A file written before them has none.

| attribute | content |
| --- | --- |
| `panam_tool` | the writer: `noise_abatement_2017.build_sphere`, `sphere_statistics.write_averaged_sphere`, `flight_acoustics.write_aam_hemisphere_netcdf` (called directly), `flight_acoustics.build_empirical_database` or `flight_acoustics.build_database_from_norah2`. These are fixed names: the last three stayed `flight_acoustics.<name>` when the functions moved to `sphere_io`, `database` and `norah2` |
| `panam_version` | `git describe --always --dirty` of the PANAM source directory; `unknown` without git |
| `panam_created_utc` | when the file was written, UTC, ISO 8601 to the second, e.g. `2026-10-11T14:03:22Z` |
| `panam_python`, `panam_numpy`, `panam_scipy`, `panam_netcdf4`, `panam_numba` | library versions; `not installed` when absent |
| `panam_settings` | the writer's processing arguments as canonical JSON: keys sorted, no whitespace, ASCII |
| `panam_settings_sha256` | SHA-256 (hex) of `panam_settings` |
| `panam_inputs` | what identifies the file, as canonical JSON, not hashed: its own name and its inputs by file name (see below) |
| `panam_input_signature` | spheres from `build_sphere` only: SHA-256 (hex) of the name, size and modification time of each input file (`provenance.file_signature` of `noise_abatement_2017.run_input_files`); `build_all(skip_unchanged=True)` compares it |
| `panam_build_record` | spheres from `build_sphere` only: the record `build_sphere` returned (the manifest columns) without its paths and stamp columns, as JSON with keys sorted and tuples, dicts and non-finite floats tagged (`{"tuple": [...]}`, `{"dict": {...}}`, `{"float": "nan"}`; `provenance.read_record_json` reads it); `build_all` rebuilds a skipped run's manifest row from it |

The settings are reduced to plain JSON before they are written: numpy arrays and scalars to
lists and numbers, a non-finite number to the text `nan`, `inf` or `-inf` (the JSON is
strict, with no `NaN` or `Infinity`), a callable to its qualified name (with its
`description`, if it has one), an object such as an `Atmosphere` to its public attributes
and type name. A path is reduced to its file name: the builders' path arguments always, and
any other string that is an absolute path or starts with `~`. No stamp names a directory of
the machine that wrote it.

The file's identity is kept out of the hashed settings and written to `panam_inputs`
instead:

| writer | `panam_inputs` |
| --- | --- |
| `build_sphere` | `test` (the aircraft), `run`, `output_path`, `reference_sphere`, `norah2_directory`, `plate_table_directory`, `samples_path` |
| `write_averaged_sphere` | `filename`, `sources` (the averaged runs) |
| `write_aam_hemisphere_netcdf` (called directly) | `filename` |
| `build_empirical_database` | `directory_name`, `database_filename`, `hover_source`, `spheres` and `exclude` when given, and `merged_spheres` (a list of file-name lists, one per condition `merge_repeats` averaged) when any were merged |
| `build_database_from_norah2` | `hem_files`, `database_filename` |

Where an identity changes the processing, the settings record that much: `build_sphere`
records `reference_sphere_given` (a reference sphere sets the grid and bands) and
`build_empirical_database` records `hover_source_given`. Equal processing settings give
equal text and an equal hash, so `panam_settings_sha256` alone says whether two spheres, or
two databases, were built with the same settings, whatever their names or runs.
`build_sphere` records `node_statistics` only when it is true: the node statistics add
variables and change no level, so a default build keeps the hash it had before the option.
`overwrite` is a file-handling choice, not a processing setting: `build_sphere` and
`build_database_from_norah2` leave it out of the settings.

`build_sphere`'s record, and the `--manifest` CSV, also carry the settings in
`noise_abatement_2017.RECORDED_SETTINGS` as given, `panam_version` and `settings_sha256`.

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

**Averaging spheres.** `sphere_statistics.read_sphere` (used by `average_spheres`,
`compare_spheres` and `tools/sphere_diff.py`) reads the same three conventions differently: by
default (`missing='no_data'`) a -999 or >1e34 cell is NaN, no data, so the run is left out of
that cell's average; with `missing='no_energy'` it is -inf, a run with zero power, as
`mask_missing_levels` reads it. NaN on disk is always no data.

**Database.** A `coverage` cell is 1 only where its band level and its direction's EAA were
finite in the source sphere. It is 0 for cells that were gated, masked or below the -100 dB
floor (all -999 in the sphere file), for the mirrored upper surface, for hover cells whose
fore/aft partner was unmeasured, and for NORAH2 NOVALUE cells, including filled ones.
Without spectra, a channel is 1 only if every band is measured; the variable's
`description` attribute says this. A direction with no band left has `dBA` = -inf and
`EAA` = 0.

## vehicle.cfg

`build_empirical_database` (through `read_vehicle_data`) and `fried_egg_plot` read
`vehicle.cfg` from the sphere directory. `tools/norah2_to_nod.py --vehicle` also accepts one
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
weight × 9.80665 / (density π radius² tip speed²). `tools/norah2_to_nod.py` needs only
`[Main Rotor]` radius and tip speed, `[Vehicle]` weight and `[Atmosphere]` density.

**Optional `[Option]` section.** It names a per-run reference list for the fried-egg plots;
`tools/fried_egg_plot.py -n` (over advance ratio and rotor angle of attack) needs it:

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

## Mid-State UAS run file (`.nc`)

One NetCDF-4 file per microphone-array recording of the PSU Mid-State multirotor flight tests (a Tarot X8
octocopter in 2021, a hexacopter from 2022), at `runs/<date>/<recording>.nc` under the dataset root.
`panam.uas_midstate` reads it (`open_run`, `list_runs`) and refuses a file whose root attribute `format_version`
is not one it knows (`FORMAT_VERSIONS`: 0.4). Every time axis is in seconds after the root attribute
`time_origin` (ISO 8601, UTC, with the recorder's clock corrected), with CF units `seconds since <time_origin>`.
Positions are east, north, up (m) in a local tangent plane at the site origin (root attributes `site_origin_lat`,
`site_origin_lon`, `site_origin_height`; latitude and longitude WGS 84 as ITRF2014, heights above the EGM96 geoid,
`geoid_offset` below the ellipsoid: `horizontal_datum`, `vertical_datum`), and in the array frame (`x`
along runway 16/34 toward 156 deg true, `y` lateral to the left, `z` up; root attributes `array_frame`,
`array_frame_x_bearing_deg`). The files follow CF-1.10 within each group and carry ACDD-1.3 discovery attributes.

| Group | Variables | Notes |
|---|---|---|
| `/` (attributes) | `Conventions`, `dataset`, `format_version`, `title`, `id`, `run_id`, `summary`, `source`, `processing_level`, `date`, `date_created`, `vehicle`, `array_configuration`, `array_table`, `status`, `status_reason`, `time_origin`, `time_coverage_start`, `time_coverage_end`, `time_coverage_duration`, `time_coverage_resolution`, `geospatial_*`, `site_origin_lat`, `site_origin_lon`, `site_origin_height`, `site_origin_source`, `horizontal_datum`, `vertical_datum`, `geoid_offset`, `frame`, `array_frame`, `array_frame_x_bearing_deg`, `runlog_*`, `daq_hour_offset`, `daq_hour_offset_source`, `daq_offset_s`, `daq_offset_uncertainty_s`, `daq_offset_method`, `clock_check_*`, `clock_note` (where a run's clocks disagree), `history`, `institution`, `project`, `creator_name`, `license`, `acknowledgment`, `references`, `keywords` | `status`: `flight`, `flight_no_log` (a flight recording with no flight log; `status_reason` says why) or `ambient`. The run log's row: `runlog_matched` (1 if the run log has a row for the recording), its text as written (`runlog_type`, `runlog_altitude_text`, `runlog_speed_text`, `runlog_comments`, `runlog_onboard`, `runlog_ulog`; '' where blank) and the values it lists as numbers (`runlog_heights_ft`, `runlog_speeds` in `runlog_speed_unit`) |
| `/acoustic` | attributes `sample_rate`, `source_file`, `source_sha256`, `source_start_time` (the recorder's own label); `counts` (mic, sample) int32; `step` (mic) Pa; `mic_id`, `channel`, `mic_mount`, `survey_point`, `notes` (mic) strings; `mic_quality` (mic) flags; `mic_east`, `mic_north`, `mic_up`, `mic_ground_up`, `mic_height_above_ground`, `mic_lat`, `mic_lon`, `mic_x`, `mic_y`, `mic_z`, `mic_pole_offset_east`, `mic_pole_offset_north`, `mic_pole_offset_uncertainty` (mic) | **pressure (Pa) = counts × step**; sample `i` at `i / sample_rate` s. A pole's position already includes its measured offset from its plate (`mic_pole_offset_*`, NaN for plates and undetermined offsets). The counts are the recorder's integers, so the original float32 export is reproduced to within its own rounding. `mic_id` is the physical microphone (from its calibration step), `channel` the recorder's channel name. `mic_quality` (CF `flag_masks`): `clipped` (1), `dropout` (2), `in_downwash` (4), `position_uncertain` (8), `runlog_note` (16); `RunFile.quality` names the flags set |
| `/vehicle/<topic>` | `time` and fields | absent when no flight log matched the recording. PX4 at native rates: `position` (east, north, up m; lat, lon, alt AMSL), `velocity` (m/s), `attitude` (`q_w`, `q_x`, `q_y`, `q_z`: body FRD to NED; roll, pitch, yaw rad), `angular_rate` (`roll_rate`, `pitch_rate`, `yaw_rate` about the FRD axes), `motor_command` (`command` (motor, time): 0-1, or PWM µs on the X8), `thrust_setpoint` (`thrust_x`, `thrust_y`, `thrust_z`), `battery`, `barometer`, `status` (`arming_state`, `nav_state`: CF flags with the PX4 release's meanings), `land_detector` (`landed`: flag, `airborne` 0, `landed` 1); positions and velocities also in the array frame (`x`, `y`, `z`) |
| `/rotors/motor<k>` | `time`, `speed` (rad/s, negative clockwise seen from above), `angle` (rad, where logged) | onboard logger, PX4 motor numbering; attributes of `/rotors`: `clock` (its correction), `source`, `motor_order`, `blades`, `angle_definition`, `correction` (where logs were relabeled), `prop_aligned_angle` (in motor order, the reported angle with the prop along its arm, where measured) and `prop_aligned_angle_source`. `RunFile.rotors` gives each motor these attributes, with its own `prop_aligned_angle` entry |
| `/air_data/<instrument>_<n>` | `time` and the aircraft-mounted instrument's fields, which measure the air moving past the aircraft: `airspeed_measured` (m/s, uncalibrated), `airflow_angle` (deg, the direction the air comes from relative to the instrument's bow, positive to starboard), `temperature`, `relative_humidity`, `air_pressure`, `heading_magnetic`, `pitch`, `roll`; a 3-D sonic's `air_velocity_u/v/w`, `airspeed_3d`, `airspeed_2d`, `airflow_direction` | attributes give the instrument, mounting and source file; a `comment` counts rows merged at repeated times |
| `/air_data/airspeed` | `time`, `airspeed_measured`, `airspeed` (calibrated), `airspeed_forward`, `airspeed_right`, `air_velocity_east/north`, `wind_east/north`, `wind_x/y`, `wind_speed`, `wind_from_direction` | derived: the wind is the ground velocity less the air velocity; attributes give the calibration C (measured = C x true), its flights, validation and caveats |
| `/segments` | per segment: `kind` (pass, climb, descent, hover, vertical climb, vertical descent), `start`, `end`, `duration`, `direction` (+x/-x), `heading_deg`, `ground_speed`, `vertical_speed`, `flight_path_angle_deg`, `height` (above takeoff), `x_start`, `x_end`, `y_mean`, `yaw_mean_deg`, `yaw_rate_deg_s`, `airspeed`, `wind_along`, `wind_cross`, `nominal_height_ft`, `nominal_speed` (in the run log's unit, its `units`; both NaN where none matched); per segment and microphone: `closest_distance`, `closest_time`, `closest_arrival` (`channel` and `mic_id` name the microphones) | steady portions only (attribute `method`); also tabulated in the dataset's `segments.csv`. Kept to the recording (vehicle times from `time_origin` to its last sample); `RunFile.segment_samples` still clips a segment that runs past either end and refuses one wholly outside. A run without a flight log has the group with no segments |
| `/weather/<instrument>_<n>` | `time` and any of `temperature` (degC), `relative_humidity` (%), `station_pressure` (Pa), `wind_speed` (m/s), `wind_from_direction` (deg true) | ground stations; attributes give the instrument and source file |

Every group with a time axis, and `/acoustic`, gives its clock: `clock` (in words), `clock_offset_s` (added to the
instrument's own time stamps to put them on UTC) and `clock_offset_uncertainty_s` (NaN where not assessed). Missing
groups mean the stream was not recorded for that run. Every variable carries its `units` (`RunFile.units`); the
reader returns integer (flag) variables as integers, other numbers as float64 with unwritten cells NaN, and refuses
a time axis whose units are not `seconds since <time_origin>`.

## How the contract is tested

- `tests/test_niceops_loads_databases.py` runs NICEOPS's self-test on PANAM-built
  databases, with the default load factors, `load_factors=None` and
  `store_spectrum=False`. It also checks that NICEOPS's AAM export of a stored condition
  is the source sphere. It skips when the `niceops` executable is not configured.
- `tests/test_database_provenance.py` pins the root attributes, the group variables, their
  types and dimensions, and the coverage of gated and fore/aft-averaged cells. It also
  checks that a default build differs from the older format only by the added items.
- `tests/test_run_metadata.py` pins the per-run metadata in the sphere and the database, and
  that the database's metadata are attributes, not variables.
- `tests/test_sphere_statistics.py` pins the averaged sphere's `N_RUNS` and
  `AMPLITUDE_SD_DB`, and that they follow the AAM variables.
- `tests/test_node_statistics.py` and `tests/test_noise_abatement_2017.py` pin `N_SAMPLES` and
  `AMPLITUDE_STD`, and that a sphere built without them is unchanged.
- `tests/test_write_aam_hemisphere_netcdf.py` pins the AAM variable order, the -999
  sentinel, the masking conventions read back, and the root vehicle fields.
- `tests/test_provenance.py` pins the provenance stamp: a stable hash for equal settings,
  the reduction of paths, and that no written attribute names a directory of the machine.
- `tests/test_golden_results.py` pins depropagated levels and the database built from them
  to 1e-6 dB.
- `tests/conftest.py` runs `validate_nod` on every database the tests build, and
  `tests/test_nod_database.py` corrupts a database one way at a time and checks that
  exactly the matching rules break.
