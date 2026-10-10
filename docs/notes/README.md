# Research notes

These notes are dated records of how parts of PANAM were investigated, fitted
and checked. They are kept for the reasoning and the numbers behind the
current models, and they are not maintained to match the code: function names,
defaults and test counts are as they were on the dates given. For current
behavior, read [THEORY.md](../THEORY.md), the how-to guides and
[DEVELOPING.md](../DEVELOPING.md).

| Note | Dates | Topic | Status / superseded by |
| --- | --- | --- | --- |
| [ground_plane_corrections.md](ground_plane_corrections.md) | 2026-09-25 to 2026-09-28 | Ground-plane microphone corrections: co-located pole and board pairs, ground impedance and turbulence fits, plate models from Fresnel zones to the axisymmetric BEM, the plate correction in the sphere builds and in NICE-OPS | Research log; later sections revise earlier ones. Current model: [THEORY §4.5-4.6](../THEORY.md#45-ground-plate-microphones). Operational notes: [DEVELOPING.md](../DEVELOPING.md#the-ground-plate-correction-in-the-sphere-builds) |
| [2017_build_findings.md](2017_build_findings.md) | 2026-10-05 | The 2017 sphere build: the hover-azimuth bug fix, what the track check found per aircraft, the evidence behind the build's bearings and track-check thresholds, and the heading-frame evaluation | Findings record. Current build: [database_build.md](../database_build.md) |
| [norah2_import_checks.md](norah2_import_checks.md) | 2026-10-04 to 2026-10-05 | How the NORAH2 import was checked: the mutants the tests catch, a smoke test through a pinned NICE-OPS build, and the measured size of each conversion hazard | Verification record. Current import: [norah2_import.md](../norah2_import.md); tests: `tests/test_norah2_import.py` |

`ground_plane_corrections.md` was moved here from `docs/` on 2026-10-10; a citation of
`docs/ground_plane_corrections.md` refers to this file. The `.nod` format that
`database_build.md` used to describe is now in [`../file_formats.md`](../file_formats.md).
