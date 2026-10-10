# How the NORAH2 import was checked

> **Status: verification record, 2026-10-04 to 2026-10-05.** This note was moved from
> `docs/norah2_import.md` on 2026-10-10. It is not maintained to match the code: test names
> and results are as they were then. The current import is described in
> [`../norah2_import.md`](../norah2_import.md), and the tests are
> `tests/test_norah2_import.py`. Where the original cited files in the companion validation
> repository, the fact is stated here instead.

`tests/test_norah2_import.py`:

- **Round trip.** A synthetic hemisphere is written by `write_norah2_hemisphere` and
  converted. The stored levels equal the file's plus the analytic offset to 1e-9 dB. They
  equal the original hemisphere plus (alpha_measured - alpha_ICAO) x 30.48 m to the file's
  0.1 dB rounding.
- **The reference atmosphere is the file's.** Edited TAMB/RELHUM/PAMB move the levels
  accordingly. A radius beyond POLDIST adds the absorption between.
- **Geometry.** The upper hemisphere is the mirror with coverage 0. Starboard stays starboard
  (a +/-6 dB field reads 12 dB between phi = +90 and -90), and `mirror_rotor` reverses it.
- **Missing cells.** They come out -inf with coverage 0, EAA 0. `fill_empty` agrees with a
  brute-force nearest search and leaves coverage alone.
- **Speed and attributes.** Both speed mappings give the advance ratio and `speed_reference`
  they should, as do the density override and the moist-air density. The root and group
  attributes are checked, including C_T.
- **Refusals.** Every refusal and warning has a test, and so does the command line (argument
  errors exit 2, conversion errors exit 1 and leave no output). A write that fails partway
  leaves an existing output untouched.
- **An uneven azimuth axis.** On the shipped -90..90 axis, reversing the rows alone is
  indistinguishable from `mirror_rotor`'s phi -> -phi. A file cut to -90..80 tells them
  apart: the labels have to move with the data.
- **The shipped R44 set** (`-m data`, with the NORAH2 package configured as `norah2` in
  `local_paths`) converts, level for level and coverage for coverage.

Each of these mutants fails at least one test:

- dropping the kept first radius of absorption;
- the ICAO constants in place of the file's atmosphere;
- `argmin` for `argmax` in the fill;
- `mirror_rotor` as a no-op, and `mirror_rotor` reversing the rows without relabeling them;
- the condition-span check applied only from three conditions up;
- coverage taken after the fill instead of before;
- the IAS-to-TAS ratio inverted.

Every aircraft set in the public package converts under both mappings. The R22 set converts
once its one-axis hover table is left out.

**Smoke test through NICE-OPS** (pinned binary 6bbd6bb). The R44 set, converted with
`--speed ground-speed` and the NICE-OPS vehicle file for the R44 used in the companion
validation study, was run on the 14 R44 approach runs of 2017 at their array microphones.
The runs used `--humidity 50 --no-ground_effect`, against that study's measured SEL per run
and microphone, corrected for the ground board with the flat pressure-doubling factor. This
is a plumbing check, not a validation: the receiver and atmosphere are not the paper's. Every
level was finite.

| Database | Mean | RMS | Median | Within 2 dB |
| --- | --- | --- | --- | --- |
| NORAH2 R44, imported | +0.81 | 4.37 | +1.24 | 45 % |
| NORAH2 R44, imported, `--fill-empty` | +3.46 | 5.26 | +2.44 | 42 % |
| shipped `RO-44_fixed_load_factor.nod` | +3.11 | 4.23 | +2.71 | 40 % |

All values are in dB, over 587 microphone-runs. The `ias-to-tas` database loads too. It is
refused without wind, as it should be, and runs under `--speed_reference ground`.

On these approaches, filling raised the predicted SEL at the array microphones by 2.7 dB on
average. That is what the choice is worth, not a verdict on which is right.

## Hazard sizes

These were measured on 2026-10-04 in an evaluation in the companion validation repository,
with NICE-OPS build 6bbd6bb, and were moved here from `norah2_import.md` on 2026-10-10.

- **Absorption to POLDIST left in.** On B407 spheres exported by NICE-OPS at 80 kt and
  -6 deg, reading a `.hem` as lossless is worth a median 0.33 dB of LA (p5 0.22, p95 0.48,
  max 0.77). Per band it is 0.37 dB at 1 kHz, 1.32 at 4 kHz and 5.94 at 10 kHz.
- **Speed.** Between B407 spheres 10 kt apart (the size of a 10 kt speed error at 80 kt),
  LA per cell moves by a median +0.5 to +1.7 dB, p95 3-7 dB and at most 6.8-13.5 dB.
- **No hover.** On R44 approach run 229354, 2,075 of 6,728 states were clamped on advance
  ratio below the slowest R44 hemisphere (36 kt).
- **Receiver correction.** NORAH's constant pressure-doubling G on its plate microphones
  and PANAM's boundary-element plate model differ by about 0.2 dB rms, with a -1 dB median
  bias.
