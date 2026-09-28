# Ground-plane microphone corrections: mixed impedance and grazing incidence

Status: implemented (2026-09-26). The corrections are in `ground_plane.py` and
`axisymmetric_bem.py`, the 2017 sphere builds divide out the plate's response
by default ("In the sphere builds"), and NICE-OPS reads the exported plate
table ("In NICE-OPS"). The notes are a research log in the order the work was
done: later sections revise earlier ones, and the Plan at the end is the plan
the work started from.

## Why this matters

Flight-test acoustics in this repository come mostly from ground-plane
microphones, and panam reduces them to free field with a flat pressure-doubling
correction (amplitude x0.5, i.e. -6.02 dB; `ground_board_scale=0.5` in
`noise_abatement_2017.load_run_channels`). That correction is exact only for an
infinite, rigid, flat ground. A real ground plane is a small hard plate lying on
softer ground, so the boundary seen by the microphone is **mixed impedance**:

- at high frequency, or near normal incidence, the Fresnel zone of the ground
  reflection shrinks onto the plate and the flat -6 dB is close to right;
- at low frequency, and especially at **grazing incidence**, the zone spreads far
  beyond the plate onto the soft ground. The sound reaching the plate has then
  travelled over, and interacted with, that soft ground, and the plate edge
  diffracts.

Evidence that this is not negligible here: comparing NICE-OPS against the 2017
NASA approach measurements, the prediction is +4.8 dB high (median) below 5 deg
elevation above the microphone, and +2.9 dB high at 5-10 deg, but within about
1 dB above 10 deg (6,118 mic-runs, standard 10 dB-down SEL). Mikkelsen &
Nickerson (below) see the same signature from the other side: pole-microphone
levels simulated from ground-plane data read low early and late in each event,
at the low angles. A ground-plane microphone that reads low at grazing
incidence, corrected by a flat -6 dB, makes a free-field prediction look high
at exactly those angles.

NICE-OPS's own ground model (`--ground_effect`: Chien-Soroka reflection over a
uniform Delany-Bazley ground) cannot represent this. It assumes a single
impedance everywhere, and its flush-receiver case gave a -3 to -4 dB offset
against the data, which says the ground-plane receiver is being mis-modelled
rather than that the ground effect is small.

## The 2017 NASA hardware

- **GRAS 67AX ground array microphone**, product page:
  <https://www.grasacoustics.com/products/special-microphone/product/197-67ax#specifications>.
  - It is a **flush-mounted** 1/2" CCP microphone set (47AX) in a GRAS GR1425
    circular ground plate, diameter 400 mm.
  - The microphone is only 8 mm tall to its diaphragm, which is what lets it sit
    flush. So the diaphragm is in the plate's top surface, 0 above the
    reflector, and that surface stands about 8 mm above the surrounding ground.
    It is not an inverted microphone above a plate.
  - The microphone sits asymmetrically in its mounting.
  - Frequency response is +/-1 dB from 5 Hz to 12.5 kHz, and +/-2 dB from
    3.15 Hz to 20 kHz.
  - The AM0375 windscreen is included.
  - I take the dataset's instrument type `gdbdfl` ("ground board, flush") to be
    this microphone. That is not confirmed.
- `invgb7` is an inverted microphone about 7 mm above a ground board. That is
  the SAE / ICAO arrangement, which Mikkelsen & Nickerson use, and a different
  geometry from the flush 67AX: a 40 cm plate with the mic
  7 mm above it. B407, AS350B3, B206L3 and R66 have 13 of these; EC130B4 and R44
  have none.
  - Since corrected: the test team confirmed that these channels were flush
    67AXs as well, despite the label, and the sphere builds treat them so
    ("In the sphere builds").
- `elevtd`: elevated (pole) microphones, mics 50, 51 and 52 on every aircraft.
  - Each is **co-located** with a ground-plane microphone at the same surveyed
    point (`*MicFullList.csv`: identical latitude, longitude and height).
  - The pairs are 50/34 (y = +500 ft), 51/36 (y = 0) and 52/38 (y = -500 ft),
    all at x = -3760 ft.
  - Between them they span overhead to low-elevation sideline geometry on every
    pass.
  - **The pole height is not recorded** in the metadata: the files' `Z` and the
    list's `Hgt(m)` are the ground elevation, identical within each pair. It
    has to be recovered from the data, from the ground-reflection interference
    pattern of the elevated microphone, whose notch frequencies are set by
    height and elevation angle. The certification standard is 1.2 m, so that
    is the first guess.
- The site is Amedee Army Airfield, CA, a high-desert site. Its ground impedance
  is unknown. The same elevated-microphone interference pattern constrains it,
  as in a two-microphone level-difference measurement (cf. ANSI S1.18).

## First look at the co-located pairs (2026-09-25)

These are scratch scripts on the B407 level passes, not yet in panam.
Method:
- Narrowband spectra in 0.25 s frames, with dL = L_pole - (L_board - 6.02).
- The emission-time elevation and range of the aircraft from each pair.

Findings:

- **The pole's comb filter collapses in f sin(elevation)** for all three pairs.
  - The first notch is near 55-60 Hz, with about 130 Hz between notches.
  - At high frequency the centreline pole sits about +3 dB above the board,
    which is the incoherent direct-plus-reflected sum.
  - The sideline poles trend toward 0 to -3 dB there at low elevation. That is
    the behaviour to separate into pole ground reflection and board deviation.
- **Pole height, fitted:**
  - Method: fit pure-tone Chien-Soroka (`flight_acoustics.ega`) to dL over 40
    Hz-2.5 kHz, on frames above 20 deg elevation, with the board taken as an
    ideal +6 dB there.
  - Result, from 7 level runs: h = 4.0 ft (1.22 m) for poles 50 and 51, and
    4.1-4.3 ft for pole 52. So the poles are the 1.2 m certification height.
- **Effective flow resistance:** 120-180 kPa s/m^2 at the sideline pairs and
  about 60 at the centreline pair.
  - The mean narrowband residual is 3.4-4.1 dB at the sideline pairs and about
    5.6 dB at the centreline pair.
  - The centreline difference may be real ground variation, or a sign that its
    board is not ideal even at 20-60 deg. That is to be resolved with band-level
    fits.

### Fixed flow resistance (2026-09-25)

Per-pair fits let the flow resistance absorb whatever else differs between the
centreline and the sidelines, so it is held at one site value.
- **The joint fit** (one shared flow resistance, a height per pole) is flat from
  60 to 225 kPa s/m^2, within about 0.08 dB of mean residual, with its minimum
  near 100. So `ground_plane.FLOW_RESISTANCE = 100`.
- **Heights at that value:** 3.96, 4.07 and 4.12 ft for poles 50, 51 and 52
  (`POLE_HEIGHT_FT`).
- **The centreline pair's residual stays the worst** at any flow resistance:
  about 5.7 dB, against 3.6-4.2 dB at the sidelines. That is consistent with
  something other than ground impedance at the centreline.

### Implementation

`ground_plane.py` in panam:
- `COLOCATED_PAIRS`, `emission_geometry`, `pair_band_histories` and
  `measured_board_transfer`. The last gives T_board = L_board - L_pole +
  G_pole per band and frame, with an SNR gate against each band's quiet-frame
  floor.
- Board models, each returning the level re free field. The names are in
  `BOARD_MODELS`, the plot labels in `MODEL_LABELS`:
  - `board_rigid_plane`: an ideal infinite rigid plane, a constant +6.02 dB
    (the current correction).
  - `board_uniform`: the flush diaphragm on an ideal rigid plate (exact +6.02
    dB), or 8 mm (the plate's thickness) above the site ground with no plate.
  - `board_soft_ground`: no plate at all.
  - `board_fresnel_strip`: Hothersall & Harriott, generalised to the plate's two
    edges. r is the share of the zone off the plate chord, which reduces to
    their r = (D - D1)/(D2 - D1) for a half-plane. The second edge matters
    because near grazing the zone runs well past the microphone.
  - `board_fresnel_disc`: the Fresnel-ellipse share on a 400 mm disc.
- Tests are in `tests/test_ground_plane.py`.
- De Jong is not implemented yet. Its formula needs to come from the 1983 paper
  or the 2006 JASA extension, not from memory.
- R44 and R66 have no pole recordings (mics 50-52 are listed but have no
  files). The pairs exist on B407, AS350B3, B206L3 and EC130B4.

### First comparison, all runs (2026-09-25)

Data: 771 runs on B407, AS350B3, B206L3 and EC130B4, three pairs each, giving
261,374 band frames (0.5 s, 50 Hz-10 kHz, SNR gate 10 dB). The pole's ground
reflection is modelled at flow resistance 100 with the fitted heights.

Median of measured T_board minus model, in dB, pooled over all bands, with the
median |residual| in brackets. The pooled figures hide band-by-band errors;
see below.



| elevation | rigid plane (+6 dB) | soft ground, no plate | Fresnel zone, 2-D strip | Fresnel zone, 400 mm disc |
|---|---|---|---|---|
| 0-2 deg | -10.6 [10.6] | +0.2 [1.0] | +0.2 [1.0] | +0.2 [1.0] |
| 2-5 | -6.6 [6.6] | +0.4 [1.5] | +0.4 [1.5] | +0.4 [1.5] |
| 5-10 | -3.7 [3.7] | +0.6 [1.5] | +0.5 [1.5] | +0.6 [1.5] |
| 10-20 | -2.0 [2.2] | +0.7 [1.6] | +0.6 [1.5] | +0.7 [1.6] |
| 20-40 | -1.4 [1.8] | +0.7 [1.5] | +0.1 [1.4] | +0.3 [1.4] |
| 40-90 | -0.9 [1.6] | +0.7 [1.7] | -0.4 [1.4] | -0.3 [1.4] |

What is solid:

- **The flat pressure-doubling correction is badly wrong at low elevation.**
  The ground plane reads 10.6, 6.6, 3.7 and 2.0 dB below +6 at 0-2, 2-5, 5-10
  and 10-20 deg. This has the sign and shape of the NICE-OPS excess against
  the 2017 approach data (+4.8 dB below 5 deg, +2.9 at 5-10).
- **The all-band medians in the table are misleading at grazing. Read the
  per-band figure instead** (`ground_plane_measured.png`, `ground_plane_model_errors.png`).
  - Below 10 deg the SNR gate leaves few frames above 1 kHz: hundreds, against
    thousands below 300 Hz. The pooled median is therefore dominated by the low
    bands.
  - Band by band, the soft-ground and Fresnel models match only at low
    frequency: below about 300 Hz at 0-2 deg and below about 500 Hz at 2-5 deg.
  - Above that they predict -10 to -25 dB, while the measurement bottoms out
    near -10 dB around 300-600 Hz and rises again. The residuals reach +10 to
    +15 dB by 1-2 kHz.
  - So at grazing the plate does take over at high frequency, far lower in
    angle than a lambda/3 Fresnel zone allows. That zone never fits on a
    400 mm plate near grazing. This is the regime a diffraction model (De Jong)
    or a smaller effective zone must explain.
  - One caveat: the pole's modelled ground reflection at grazing and high
    frequency enters the measured T directly.

Not yet trustworthy (figures: `ground_plane_measured.png` and `ground_plane_model_errors.png`
in the session scratchpad):

- **The high bands (above about 3 kHz) read well above +6 dB**, up to +10-17 dB
  at grazing. A flush plate cannot do that. Candidates:
  - The pole microphone's own response with incidence angle. A 1/2"
    microphone pointing up rolls off at grazing incidence at high frequency, so
    the pole reads low, which pushes T high. **The pole microphones' type and
    orientation are needed.**
  - A noise floor in the high bands at long range.
  - The pole ground model in the incoherent regime.
- **The mid-band level at steep incidence is about 3.5-4 dB, not 6**, for pairs
  51 and 52. The absolute level is confounded between the pole ground model
  (its |Q| at flow resistance 100) and the plate. The variation with elevation
  is the robust part.
- **Pair 50/34 sits about 1.5 dB above the other two**, broadband, on AS350B3,
  B407 and EC130B4, but only about 0.4 dB on B206L3. That looks like a
  calibration or gain difference in pole 50 or board 34, not acoustics.
- **Between 10 and 40 deg the Fresnel models predict more soft-ground
  influence at 1-3 kHz than is measured.** The measured curve sits between the
  plate and the Fresnel curves, so the lambda/3 zone may be too large for this
  plate, or the blend too simple. That is where De Jong, or a smaller zone
  fraction, should be tested.

## Literature review of the downloaded papers (2026-09-25, ~/Desktop/papers)

- **De Jong, Moerkerken & van der Toorn 1983** (JSV 86:23), section 3.4, eqs.
  36-43. At one admittance step the field is the uniform-ground field on the
  specular side plus diffraction terms D'_i = (Q1 - Q2) D_i and D'_r =
  (Q1 - Q2) D_r. D_i and D_r are the rigid semi-infinite-screen coefficients
  (26a,b), (R/L) e^{-i pi/4}/sqrt(pi) F(sqrt(k(L-R))), with
  F(x) = int_x^inf e^{it^2} dt. It is heuristic, derived for a hard-to-soft
  step, and the paper states it "loses its validity for kr < 10".
  - For our plate the microphone is 0.2 m from both edges, so kr < 10 below
    about 2.7 kHz.
- **Lam & Monazzam 2006** (JASA 120:686). The original De Jong violates
  reciprocity for a soft-to-hard step. The fixes:
  - eq. (11), the modified De Jong, adds a sign mu on the direct-path Fresnel
    term;
  - eq. (12), nMID, sums that over n discontinuities;
  - eq. (13) is the strip form.
  - Validated against BEM, but "the accuracy decreases as the source or
    receiver height decreases (nearer grazing)". At heights of 0.01 m it shows
    a constant error of several dB (their Fig. 8b).
  - **Checked numerically here: nMID does not reduce to the soft-ground result
    as the plate shrinks to nothing when the receiver is low.** The two edges'
    direct-path terms add instead of cancelling. With Q_soft = -0.3+0.4i at
    1 kHz, a vanishing strip gives |p/p1| = 1.41 against 1.38 (correct) at
    1.5 m, 0.96 against 0.51 at 0.1 m, and 0.99 against 0.81 for a flush
    receiver. The flush limit is exactly 1 + 2 Q_soft - Q_plate. A flush
    microphone on a 400 mm plate is the worst case for this model family.
- **Attenborough & Taherzadeh 2026** (Applied Acoustics 242:111067), a review
  of outdoor ground-effect prediction.
  - (a) Fresnel-zone weighting should be applied to **pressure squared, not to
    excess attenuation in dB**, following Boulanger et al. 1997. That is
    closer to BEM, and HARMONOISE's dB weighting can be off by about 8 dB
    (their Figs. 10-14).
  - (b) A lambda/3 (or /4) zone is closer to BEM than HARMONOISE's /16.
  - (c) Delany-Bazley fits grassland short-range data poorly. Slit-pore,
    Wilson, variable-porosity or hard-backed-layer models are better. Their
    pasture/meadow class is 250 +/- 100 kPa s/m^2, porosity 0.25, layer
    0.05 m.
  - (d) A turbulence coherence factor (HARMONOISE; Ostashev) scales with
    h_s h_r/(h_s + h_r). It is about 1 for the flush board but not for the
    1.2 m pole.
- **Tinkham** (Sheffield Hallam thesis): numerical Kirchhoff-Fresnel
  integration over surface elements for finite barriers. It is a quadrature
  technique for area diffraction integrals, not an impedance-plane model.
- **Parry** (thesis): forward and inverse uncertainty in outdoor propagation,
  with Bayesian (MAP) inference of ground parameters from level differences.
  Relevant to fitting the pole's ground model with uncertainty instead of one
  fixed value.

Implications:
1. The Fresnel models here blend in dB. They should blend pressure squared,
   and try lambda/3 against lambda/4.
2. nMID (De Jong) is structurally unsuited to a flush microphone on a small
   plate. It is still worth running as a comparison, with that caveat.
3. The first-principles route that handles a flush receiver, both edges, the
   circular shape and grazing incidence is the tailored-Green's-function
   boundary integral on the disc (Kingan et al. 2023): the soft half-space
   Green's function, with only the plate discretised. It can be tabulated over
   frequency and elevation once, and it is also the reference for everything
   else.
4. At grazing the measured transfer function rests on the pole's modelled
   ground reflection (-10 to -14 dB at 125-500 Hz below 2 deg). So the pole
   model deserves the same care: impedance model, turbulence coherence, and a
   fit with uncertainty.

### Paths 1 and 2 evaluated (2026-09-25)

The Fresnel models now blend in energy by default (`blend='energy'`; `'db'`
is still available), and `board_nmid` implements Lam & Monazzam's eq. (12).
- nMID is averaged over nine sub-frequencies per band.
- Its complex Q comes from `flight_acoustics.spherical_reflection_coefficient`.
  That function was split out of `ega`, and `ega` is bit-identical afterwards.

Score: RMS over 50 Hz-2.5 kHz of each band's median error (measured - model),
in dB, skipping bands with fewer than 300 frames. 771 runs.

| variant | 0-2 | 2-5 | 5-10 | 10-20 | 20-40 | 40-90 deg |
|---|---|---|---|---|---|---|
| rigid plane (current +6 dB) | 13.13 | 7.19 | 4.37 | 2.58 | 1.81 | 1.39 |
| soft ground, no plate | 6.12 | 4.85 | 4.27 | 3.06 | 2.06 | 1.88 |
| Fresnel strip, dB, lambda/3 | 6.12 | 4.79 | 4.05 | 2.50 | 0.82 | 0.84 |
| Fresnel strip, energy, lambda/3 | 5.77 | 4.34 | 3.35 | 1.89 | 0.57 | 0.90 |
| Fresnel strip, energy, lambda/4 | 5.75 | 4.23 | 3.17 | 1.67 | 0.51 | 1.03 |
| Fresnel disc, dB, lambda/3 | 6.12 | 4.85 | 4.21 | 2.75 | 0.90 | 0.76 |
| Fresnel disc, energy, lambda/3 | 6.11 | 4.78 | 3.95 | 2.34 | 0.62 | 0.82 |
| Fresnel disc, energy, lambda/4 | 6.11 | 4.74 | 3.81 | 2.10 | 0.45 | 0.98 |
| nMID (corrected De Jong) | 5.27 | 5.21 | 3.14 | 1.36 | 0.75 | 1.12 |

- Energy blending helps at every elevation below 40 deg, as Attenborough &
  Taherzadeh found. lambda/4 helps a little more below 40 deg and costs a
  little above.
- nMID is best at 5-20 deg. Near grazing it is the only model that brings the
  plate back in at high frequency, but it oscillates: +2 to +6 dB at low
  frequency, -9 dB at 300-500 Hz and +13 dB near 800 Hz at 2-5 deg. That is
  consistent with its flush-receiver defect.
- Nothing gets below about 4-5 dB at 0-5 deg. That is the regime for the disc
  boundary integral (path 3), and for the pole reference (path 4).

### Path 3: boundary integral on the disc (2026-09-25)

`disc_bem_factor`, `disc_bem_table` and `board_disc_bem` in `ground_plane.py`.

Formulation:
- For a locally reacting plane with dp/dz = -i k beta p (e^{-i omega t}),
  Green's second identity with the uniform soft ground's own Green's function
  gives, on the plane, p(r) = p_inc(r) - i k beta_soft integral over the disc
  of G(x, r) p(x) dS. This is the tailored-Green's-function approach of Kingan
  et al. 2023: only the rigid disc is discretised.
- G between surface points is e^{ik rho}/(4 pi rho) (1 + Q(rho)), with Q from
  `spherical_reflection_coefficient` at cos = 0.
- The incident field from a distant source is (1 + Q_soft) times a plane wave.
  So the plate's effect is a factor M(f, elevation), and the microphone reads
  |1 + Q_soft|^2 |M|^2 re free field, averaged over five sub-frequencies per
  band.
- The system matrix depends only on frequency. One solve per frequency serves
  every elevation, and the whole table (105 frequencies up to 5 kHz by 80
  elevations) takes 15 s.

Numerics:
- Collocation on a polar mesh.
- Close cell pairs use 6 x 6 sub-cell quadrature. Each cell's own 1/rho
  singularity is integrated exactly, by rays.
- Converged to about 0.1-0.3 dB at 8 cells per wavelength with at least 24
  cells across.
- Checks:
  - rigid soft ground gives |M| = 1;
  - a 6 ft plate at 45 deg doubles the pressure;
  - mesh refinement converges.

Score (same measure as above):

| | 0-2 | 2-5 | 5-10 | 10-20 | 20-40 | 40-90 deg |
|---|---|---|---|---|---|---|
| disc boundary integral | 5.59 | 3.56 | 2.23 | 1.10 | 0.82 | 1.25 |

It is the best model from 2 to 20 deg. It follows the rise back toward the
plate at high frequency, and it is the only one to do so without nMID's
oscillation. Still open:

- **0-2 deg.** It recovers from -17 to about -13 dB at 1-2 kHz where the data
  reach about -2 dB. Candidates:
  - the plate's 8 mm thickness, i.e. the diaphragm 8 mm above the soft ground
    (a raised edge, as Kingan et al. and Blandeau et al. found matters);
  - the soft ground's impedance model (Delany-Bazley is poor for grassland);
  - the pole reference, whose modelled reflection is largest exactly here.
- **Steep angles (20-90 deg).** It is slightly worse than the Fresnel models,
  with +1 to +2 dB ripple at 1.5-2.5 kHz. A microphone at the exact centre of
  a circular plate receives edge diffraction from the whole rim in phase. The
  67AX microphone sits deliberately off-centre "to attain a more uniform
  frequency response", so the model should put it where the 67AX does. The
  offset is not yet known.

### Microphone position: SAE ARP 4055 (2026-09-25)

Confirmed from the ARP 4055 figure (reproduced in the AERSP511 lecture notes,
"3 Experimental Rotorcraft Aeroacoustics", slide 73):
- a 0.4 m plate, at least 2.5 mm thick, painted white;
- its **surface flush with the surrounding ground**;
- the microphone at **3/4 of the radius** (0.15 m), on a line **normal to the
  intended flight track**.

The ICAO configuration quoted in arXiv:2409.10957 gives the same: a 40 cm
plate, a 7 mm inverted gap, and the microphone axis 0.15 m from the centre. It
states the 6 dB correction is valid up to 10 kHz and for incidence under 60
deg from the normal, i.e. elevations above 30 deg. NASA's WAMS II (ERF48, 2022)
put a GRAS 67AX in a 381 mm (15 in) board, offset from the centre "based on"
ARP 4055. **Whether the 2017 boards were 400 or 381 mm, and embedded flush or
laid on the ground, is not known.**

Model changes:
- `disc_bem_scattered` takes the microphone position (default 3/4 R along +y,
  `PLATE_MIC_OFFSET_FT`) and height (0 flush;
  `INVERTED_MIC_HEIGHT_FT` = 7 mm for the inverted type).
- It returns the scattered field S(f, elevation, azimuth), so p/p_free =
  1 + Q e^{2ikh sin el} + (1 + Q) S.
- The collected frames now carry the source's horizontal offset (dx, dy) for
  the azimuth.

**Which side of the plate.** ARP 4055 fixes the line, not the side, and the
2017 records do not say. The sideline pairs decide it: pair 50/34
(y = +500 ft) fits best with the microphone on +y, and pair 52/38 (-500 ft) on
-y. That means **outboard**, on the half of the plate away from the flight
track, so sound crosses the plate before reaching the microphone. For the
centreline pair the two sides are equivalent.

| | 0-2 | 2-5 | 5-10 | 10-20 | 20-40 | 40-90 deg |
|---|---|---|---|---|---|---|
| boundary integral, mic at centre | 5.67 | 3.66 | 2.34 | 1.18 | 0.73 | 1.15 |
| boundary integral, 3/4 R outboard (ARP 4055) | 5.83 | 3.92 | 2.59 | 1.33 | 0.66 | 0.95 |

The ARP 4055 position removes the centred microphone's rim-focusing ripple.
It is the best model of all above 20 deg (0.43 dB at 20-40 deg for the +y
table pooled), and 0.15-0.26 dB worse than the centred one below 20 deg,
mostly at the centreline pair. The low-angle mismatch is the next target:
- the plate's thickness, if it lay on the ground rather than flush;
- the ground impedance model;
- the pole reference (path 4).

The centred result below 20 deg is probably a compensating error, not
evidence for a centred microphone.

### The 2017 plates (2026-09-25)

Confirmed by the user: the boards are as specified by GRAS (GR1425, 400 mm)
and lay on top of the ground, so part of the rounded edge is exposed. The GRAS
67AX data sheet's cutaway drawing (page 4) gives:
- 8.00 mm thickness;
- the top surface tapering to 2.50 mm at the rim;
- the microphone flush, 150 mm from the centre, which is exactly 3/4 R.

So the ARP 4055 position is also the 67AX's.

How large is the step likely to be? The thin-disc model puts the plate in the
ground plane. The real top surface is 8 mm above the soft ground, with a 2.5 mm
rim and a short taper.
- The electrical size of the step is small: k t = 0.15 at 1 kHz, 0.29 at
  2 kHz and 0.73 at 5 kHz for t = 8 mm (0.05-0.23 for the 2.5 mm rim).
- Raising the receiver 8 mm hardly changes the soft ground's own field near
  grazing. The phase term 2 k t sin(el) is about 0.01 at 1 kHz and 2 deg, and
  the numerical distance is set by beta, not by the small change in cos.
- So the step is unlikely to explain the 5-10 dB gap at 0-2 deg and
  0.5-2 kHz, but it could matter near the top of the band.

Settling it properly needs a raised-plate BEM: the plate's top, taper and rim
as a rigid body above the soft half-space, with that half-space's Green's
function. It is a bigger build than the thin-disc model and is deferred until
the pole reference has been dealt with.

### Path 4: the pole reference and the site's ground (2026-09-25)

New in `ground_plane.py`:
- `surface_admittance`, with four models: Delany-Bazley; Miki; variable
  porosity, from Attenborough & Taherzadeh eq. (10); and a hard-backed
  Delany-Bazley layer, eq. (5).
- `turbulence_coherence`, the HARMONOISE factor, eq. (18).
- `pole_level`, the `ega` band-averaged form with any ground and that
  coherence. With Delany-Bazley and no turbulence it equals `ega` to 2e-11 dB.
- `disc_bem_*` and `board_disc_bem` take the same ground dict.
- `flight_acoustics.spherical_reflection_coefficient` accepts an admittance;
  `ega` is still bit-identical.

**The fit** (scratchpad `fit_ground.py`) uses only frames at or above 30 deg,
where the ICAO 6 dB correction holds and the boundary-integral plate model is
good.
- Observable: the raw board - pole difference.
- Model: board (boundary integral, 3/4 R outboard) minus G_pole.
- Fitted: the ground parameters, the three pole heights and gamma_T.
- Per-pair gain offsets (calibration) are solved exactly as L1 medians.
- 6,000 frames; 10 bootstrap resamples over runs for the best model.

| model | mean abs residual (dB) | fitted ground | gamma_T |
|---|---|---|---|
| current (DB 100, no turbulence) | 1.51 | sigma 100 | 0 |
| Delany-Bazley | 1.39 | sigma 1300 | 3.2e-5 |
| Miki | 1.39 | sigma 1700 | 3.2e-5 |
| variable porosity | 1.39 | sigma_e 380, alpha_e -> 0 | 3.0e-5 |
| hard-backed DB layer | 1.39 | sigma 1300, depth 0.11 m (effectively semi-infinite) | 3.2e-5 |

- Pole heights: 4.08, 4.05 and 4.08 ft (+/- 0.005 by bootstrap).
- Variable porosity: log10 sigma_e = 2.61 +/- 0.05, log10 gamma_T = -4.57
  +/- 0.04.
- **The site is much harder than the pole-notch fit suggested.** sigma of
  about 1300 is hard-packed soil, not grass, which is plausible for Amedee.
- **The turbulence is strong.** 3e-5 is above HARMONOISE's typical 1e-6 to
  1e-5; desert daytime convection is plausible.
- The two are partly traded: both fill in the pole's interference dips.
- Steep incidence cannot tell the impedance models apart. They all fit alike.

**Out-of-sample test below 30 deg** (`eval_ground.py`). Each model rebuilds
the measured transfer function (with its own pole correction and pair
offsets) and the boundary-integral plate model on the same ground. RMS of the
per-band median error, dB:

| ground | 0-2 | 2-5 | 5-10 | 10-20 | 20-30 | 30-40 | 40-90 deg |
|---|---|---|---|---|---|---|---|
| current (DB 100, no turbulence) | 6.18 | 4.21 | 2.89 | 1.59 | 0.78 | 0.79 | 0.75 |
| Delany-Bazley 1300 + turbulence | 2.91 | 2.66 | 2.11 | 0.73 | 0.54 | 0.74 | 0.88 |
| Miki 1700 + turbulence | 3.86 | 2.22 | 1.91 | 0.64 | 0.48 | 0.70 | 0.83 |
| variable porosity 380 + turbulence | 4.69 | **1.66** | **1.61** | **0.54** | **0.40** | **0.62** | 0.79 |
| DB layer 0.11 m + turbulence | **2.90** | 2.67 | 2.11 | 0.73 | 0.54 | 0.74 | 0.88 |

- **Much of the apparent grazing deficit came from the pole reference, not the
  plate.** With the refitted pole model the measured ground-plane level near
  grazing rises by up to about 8 dB (figure `ground_refit.png`). The ideal
  rigid plane's error at 0-2 deg falls from 12.7 to 3.5-5 dB.
- The flat +6 dB correction is still several dB high at grazing.
- **The boundary-integral plate model on the same ground now follows the
  measurement to within a few dB at all elevations,** on data the fit never
  used.
- Variable porosity is best from 2 to 40 deg. Delany-Bazley (or the layer) is
  best at 0-2 deg.
- Still unexplained: the measured level above about 3 kHz exceeds +6 dB, up to
  +10 dB. A flush plate cannot do that, so the pole microphone's own response
  is the leading suspect (type and orientation still needed).

### The pole microphones' response (2026-09-25)

From the test report (Watts et al., NASA TM 2019, `Watts.TM2019.pdf` in the
data folder), section 5.1:
- "Three positions in the array used G.R.A.S. **46AE** microphones mounted on
  4 foot tripods to emulate certification placement at -45, 0, and 45 deg under
  the aircraft and perpendicular to the flight path."
- The primary microphone is the 67AX, "flush mounted in a **15 inch** ground
  board", offset from the centre, "based on ... ARP4055, but adapted from
  inverted microphones to flush mounted".
- Section 8.1: `gdbdfl` is "flush mounted in a ground board with the diaphragm
  pointed up", `elevtd` is "elevated 4 feet on a tripod", and `invgb7` is
  "inverted over a ground board with a 7 mm gap".
- The report applies "a constant installation correction of -6 dB for 0 to
  10,000 Hz" to `gdbdfl`.
- **The 15 inch board (381 mm) in the report differs from the GRAS GR1425
  (400 mm) confirmed by the user. It is flagged here, not resolved.**

The 46AE is a free-field microphone (40AE capsule).
- "Emulate certification placement" means the diaphragm lies in the plane of
  the flight path and the station (14 CFR 36 / Annex 16), so sound from the
  flight line arrives at 90 deg incidence.
- GRAS's free-field corrections for the 40AE (`GRAS_40AE_FREE_FIELD_CORRECTIONS`,
  from GRAS_Free-field_and_Random_Incidence_Corrections.xlsx, with protection
  grid) give the response re on-axis. At 90 deg it is -0.2 dB at 1 kHz, -0.5 at
  2 kHz, -1.1 at 4 kHz, -2.9 at 8 kHz and -3.6 at 10 kHz.
- Code: `free_field_microphone_response`; `pole_incidence`, for the direct and
  ground-reflected angles on a microphone axis; `certification_axis`, normal
  to the plane of the flight line and the pole; and `pole_level(...,
  response_direct, response_reflected)`.
- On the centreline both paths arrive at 90 deg. At the sideline poles the
  reflected path arrives at about 23 or 157 deg, depending on which way the
  diaphragm faces. The two signs fit identically.

**Result.**
- Including the response lowers the measured ground-plane level at steep
  angles by up to about 3 dB at 8-10 kHz, in the right direction.
- **About +2 dB of excess remains at 2-5 kHz.** At 40-90 deg the measurement is
  7.6-8.4 dB where the plate model gives 6.2-6.4, which a flush plate cannot
  exceed. Still-unmodelled candidates:
  - rough-ground scattering weakening the coherent reflection, which would put
    too much reflected energy in the pole model;
  - the pole's windscreen or tripod;
  - an orientation other than certification practice.
- **The flow resistivity is not identified by the steep-angle fit.** It moved
  from 1300 to about 5300 (Delany-Bazley), and from 380 to 1900 (variable
  porosity), when this sub-dB high-frequency term was added. The loss did not
  change (1.39 to 1.40 dB). It then costs grazing accuracy (0-2 deg: 2.9 to
  4.5 dB).
- The ground has to be fitted where it matters, near grazing. That can still
  be done out of sample by holding out runs instead of elevations.

### Orientation, windscreen, roughness, and pinning the ground (2026-09-25)

- **Orientation, confirmed by the user.** Per Part 36 the diaphragm's plane
  lies along the flight track, so the source stays nominally in that plane
  throughout. That is what `certification_axis` does: 90 deg incidence for the
  direct path.
- **Windscreen.** The 46AEs had a standard spherical foam windscreen, size not
  recorded (typically about 90 mm). Its insertion effect is a few tenths of a
  dB below about 5 kHz. It is not modelled, because no curve is to hand, and
  it is too small to be the 2 dB residual.
- **Roughness.** `pole_level(..., roughness=)` applies the Kirchhoff (Ament)
  coherent-reflection factor exp(-2 (k sigma_h cos theta)^2), which vanishes
  toward grazing.
  - Fitted at 30 deg and above over 50 Hz-5 kHz: sigma_h = 8 mm.
  - 2.5-5 kHz error at 40-90 deg falls from 1.67 to 0.58 dB, and at 30-40 deg
    from 2.15 to 1.54 dB.
  - About +1.5-2 dB of excess remains at 3-5 kHz at 20-40 deg.

**Pinning the ground** (`profile_ground.py`):
- The runs are split in half at random.
- For each flow resistivity on a grid, the plate table is rebuilt on that
  ground, and the pole's heights, turbulence and roughness are fitted on the
  training runs at 30 deg and above.
- Every elevation is then scored on training runs (for selection) and on
  held-out test runs (reported).

Held-out-run RMS error in dB (50 Hz-2.5 kHz):

| ground | 0-2 | 2-5 | 5-10 | 10-20 | 20-30 | 30-40 | 40-90 deg |
|---|---|---|---|---|---|---|---|
| before (DB 100, no turbulence) | 6.18 | 4.21 | 2.89 | 1.59 | 0.78 | 0.79 | 0.75 |
| DB 200 | 8.29 | 2.74 | 1.82 | 0.81 | 0.44 | 0.51 | 0.55 |
| DB 400 | 6.49 | 1.66 | 1.47 | 0.50 | 0.32 | 0.41 | 0.53 |
| DB 800 | 4.62 | 2.06 | 1.74 | 0.58 | 0.40 | 0.53 | 0.66 |
| DB 3200 | 1.89 | 3.32 | 2.24 | 0.77 | 0.54 | 0.74 | 0.86 |
| VP sigma_e 100 | 7.76 | 2.06 | 1.37 | 0.61 | 0.30 | 0.38 | 0.55 |
| **VP sigma_e 200** | 6.50 | **1.28** | **1.17** | **0.37** | **0.27** | **0.43** | 0.60 |
| VP sigma_e 400 | 5.18 | 1.71 | 1.52 | 0.47 | 0.35 | 0.53 | 0.69 |
| VP sigma_e 1600 | 2.75 | 2.98 | 2.10 | 0.70 | 0.48 | 0.69 | 0.83 |

- **From 2 to 40 deg the ground is pinned.** Variable porosity with sigma_e
  about 200 (about 400 for Delany-Bazley) with turbulence gamma_T about 3e-5.
  The held-out error is 1.3, 1.2, 0.4 and 0.3 dB at 2-5, 5-10, 10-20 and
  20-30 deg, three to four times better than where this started.
- **0-2 deg wants a far harder ground** (sigma 3200-6400: 1.6-1.9 dB) than
  every other elevation, so it conflicts with them. Those frames are the
  lowest and longest-range: the aircraft far away and near the horizon.
  - Near grazing, atmospheric refraction bends the direct and reflected rays
    differently at a 1.2 m pole and a flush board. That is not modelled, and
    a harder "effective" ground is how a refraction-free model would try to
    imitate it.
  - A ground that changes along the path is the other candidate.
  - The test recorded temperature profiles (the balloon's temperature string,
    every 10 ft) and LIDAR winds to 900 ft (Watts et al. section 5.2), so
    refraction can be modelled rather than fitted around.

### Plate thickness: the raised-plate model (2026-09-25)

The user confirmed the boards are the GRAS GR1425, 400 mm (the report's
"15 inch" is wrong), lying on the ground.

**Model** (`raised_plate_mesh`, `raised_plate_scattering`):
- The plate is a rigid body on the soft ground: the flat top 8 mm up with the
  microphone flush in it at 3/4 R, a conical taper down to 2.5 mm, and a
  vertical rim. The taper's radial length is not on the GRAS drawing; it is
  assumed to be 20 mm (`PLATE_TAPER_LENGTH_FT`) and checked at 10 and 30 mm.
- An exterior Neumann boundary integral over the exposed surfaces uses the
  soft half-space's Green's function. The ground outside the plate then drops
  out exactly: 1/2 p = p_inc + integral of p dG/dn dS.
- By linearity the microphone's pressure is P = P_d + Q P_r, from the direct
  and ground-reflected incident plane waves solved separately. Each frame
  applies its own Q.
- A point's image sits 2 z below the ground, so on a thin plate every close
  pair is near-singular on the scale of the plate's height. Near pairs use
  geometry-driven adaptive quadrature (`_adaptive_points`).

**Checks.**
- Axisymmetric for a centred microphone. A bug that put the central panel's
  collocation point off-centre was found by this test and fixed.
- A thin plate on rigid ground gives 2.00-2.09, against an exact 2.
- In the thin limit it matches the thin-disc model to 0.2-0.4 dB at 500 Hz
  and at 10 deg and 2 kHz.
- **Unresolved: up to 1.5 dB difference at 2 kHz and 45 deg.** The same
  thin-limit difference appears on a 1.5 ft plate, where it is 2.28 against
  1.82.
  - Both models use the Weyl-van der Pol (Chien-Soroka) coefficient for the
    ground's Green's function, a long-range approximation used here at
    centimetre ranges (kR about 1).
  - The thin-disc model uses only its values, through the exact
    dG/dz = -i k beta G. The raised model needs its normal derivative, which
    is the least reliable part of the approximation.
  - The definitive fix is the exact complex-image Green's function,
    G = g(R1) + g(R2) - 2 k beta integral from 0 to infinity of
    e^{-k beta q} g(R(q)) dq, with R(q) the distance to an image at complex
    height -z - iq (Ochmann; Taraldsen; valid for Re beta > 0). It is not yet
    implemented.

**Thickness effect** (`thickness.py`). This is the increment from the real
plate over a 0.3 mm plate, within the same model, so most of the kernel's
approximation cancels. Ground: variable porosity sigma_e 200. Values in dB,
for sound travelling toward / away from the microphone's side of the plate
(20 mm taper):

| elevation | 250 Hz | 500 Hz | 1 kHz | 2 kHz | 3.15 kHz | 5 kHz |
|---|---|---|---|---|---|---|
| 1-10 deg | 0.0/+0.1 | -0.2/+0.3 | -0.3/+0.5 | +1.0/+0.3 | +0.2/-0.5 | +0.9/+0.1 |
| 20 deg | -0.1/+0.1 | -0.3/+0.3 | -0.5/+0.4 | +0.9/+0.3 | 0.0/-0.5 | +0.4/-0.4 |
| 45 deg | -0.1/0.0 | -0.4/+0.2 | -0.6/+0.2 | +1.2/-0.4 | -0.8/-1.0 | -0.2/-0.8 |
| 80 deg | -0.1/-0.1 | -0.3/-0.1 | +0.1/-0.2 | +0.3/-0.5 | +0.6/+0.3 | -0.5/+0.2 |

- Within 0.6 dB below 1 kHz; up to about 1-1.5 dB at 2-5 kHz, with a sign
  that depends on direction.
- The taper length (10-30 mm) changes it by up to about 0.6 dB at high
  frequency.
- It is nearly independent of elevation below 10 deg.
- **The thickness is a second-order effect.** It cannot explain the multi-dB
  0-2 deg conflict, which asked for a much harder ground. The high-frequency
  values carry the kernel uncertainty above.

### Exact ground Green's function (2026-09-25)

`image_integrals`, `exact_half_space_green` and `ImageIntegralTable`:
- G = g(R1) + g(R2) - 2 k beta I, with I = int_0^inf e^{-k beta q} g(R_q) dq
  and R_q the distance to an image at complex height (Ochmann; Taraldsen).
  e^{-i omega t}, dp/dz = -i k beta p on the plane, Re beta > 0.
- The normal derivative needs no differentiation of an approximation. By
  parts in q, dG_img/dZ = dg(R2)/dZ - 2 i k beta g(R2) + 2 i k^2 beta^2 I. The
  horizontal derivative uses J = int e^{-k beta q} g'(R_q)/R_q dq.
- Quadrature: q = rho sin(phi) below rho and q = rho cosh(psi) above remove
  the q = rho singularity (exactly on the surface). Gauss-Legendre segments are
  graded toward it and on the decay length 1/(k |beta|).
- I and J are tabulated per frequency in (log rho, log Z) with the phase and
  1/R factors divided out. Interpolation error is at most 0.15% (J 1.5% on
  the surface at the far edge).

Checks:
- dG/dz = -i k beta G to 1e-6 at all ranges, including on the surface.
- The rigid limit is the image source (1e-8).
- It agrees with Weyl-van der Pol (Chien-Soroka) to 0.005 in Q at long range,
  100 Hz-4 kHz.
- It converges on the surface in 10 nodes per segment.
- Bugs found and fixed along the way: 0 * inf at zero-width segments, and a
  rounding-negative Z on the table's first row that put the root on the
  growing branch.

Both plate models now use it by default (`green='exact'`; `'weyl'` keeps the
old kernel).
- In the thin limit they agree to 0.04 dB at 500 Hz (0.2 before).
- At 2 kHz the gap shrinks with mesh refinement: at 10 deg 0.94, 0.96 and
  0.58 dB, and at 45 deg 0.52, 0.57 and 0.41 dB, for 24, 36 and 48 cells
  across. That is discretisation error, not a formulation error.
- **Correction to an earlier claim:** below about 2.5 kHz the mesh is set by
  the cells-across floor, not by cells per wavelength, so the earlier
  thin-disc convergence check (which varied the latter) did not test it. At
  2 kHz and 10 deg the thin disc moves 0.7 dB from 24 to 48 cells across. The
  scoring tables built at 24 carry roughly that much mesh error near grazing
  at 2 kHz.

**Thickness effect with the exact kernel** (36 cells across). This is the real
plate re a 0.3 mm plate, in dB, toward / away from the microphone's side:

| elevation | 250 Hz | 500 Hz | 1 kHz | 2 kHz | 3.15 kHz | 5 kHz |
|---|---|---|---|---|---|---|
| 1-10 deg | -0.1/+0.1 | -0.3/+0.2 | -0.6/+0.3 | +0.8/+0.4 | +0.3/+0.2 | -0.1/-0.1 |
| 20 deg | -0.1/+0.1 | -0.4/+0.2 | -0.8/+0.3 | +0.7/+0.3 | +0.3/0.0 | -0.2/-0.5 |
| 45 deg | -0.1/0.0 | -0.5/+0.1 | -0.8/0.0 | +1.0/-0.5 | -0.4/-0.7 | -0.3/-0.6 |
| 80 deg | -0.2/-0.1 | -0.4/-0.2 | -0.2/-0.5 | 0.0/-0.8 | +0.7/+0.2 | +0.3/+0.7 |

The effect stays within about +/-1 dB everywhere, so it is second order.

### Axisymmetric, compiled BEM (2026-09-25)

`axisymmetric_bem.py` solves the same equation as `raised_plate_scattering`,
with the same exact Green's function, by azimuthal modes.
- The plate and the half-space Green's function are both invariant under
  rotation about the plate's axis. The surface pressure separates into modes
  e^{i m phi}, each solved on the generating curve (top, taper, rim): about 60
  segments, instead of about 2000 surface panels.
- A plane wave splits by Jacobi-Anger into i^m J_m(kappa r) e^{i m (phi - az)}.
  One solve per mode and elevation serves every azimuth. An off-centre
  microphone is a sum over modes of that mode's on-surface representation;
  only the geometry and Green's function must be axisymmetric.
- Modes |m| <= k a + 10.
- The modal kernels integrate dG/dn cos(m phi) for all m at once. Near pairs
  use breakpoints graded geometrically toward the singular point in s (the
  target or its image) and in phi toward 0.
- Compiled with Numba (`numba` added to `requirements.txt`), parallel over
  collocation points on 14 cores.
- Apple's GPU backends (JAX-Metal, PyTorch MPS) lack reliable double-precision
  complex arithmetic, which these near-singular integrals need. A GPU path
  (JAX on CUDA) is possible later and would also give gradients.

**Speed.** About 0.1 s per frequency, against 45 s for the 3-D surface model
at 36 cells across. The full real-plate table (105 frequencies x 80
elevations x 36 azimuths) takes 14 s, where the 3-D model would take over an
hour.

**Checks** (`tests/test_axisymmetric_bem.py`):
- A thin plate on rigid ground gives 2 to 0.3-0.8%.
- A centred microphone is independent of azimuth (1e-6).
- There is mirror symmetry in x for the +y microphone.
- It agrees with the 3-D surface model to 0.02-0.12 dB.
- Refining the generator mesh fourfold changes results in the third digit.

**The real plate on rigid ground is not exactly 2.** That is a converged,
physical O(kt) effect of the 8 mm step and taper 30 mm from the microphone:
-0.35/+0.2 dB at 500 Hz and about +/-1.2 dB at 2 kHz, with opposite signs for
sound travelling up and down the step.

**The ground profile redone with the real plate** (`PLATE=raised
profile_ground.py`). Held-out-run RMS error in dB:

| plate model, ground | 0-2 | 2-5 | 5-10 | 10-20 | 20-30 | 30-40 | 40-90 deg |
|---|---|---|---|---|---|---|---|
| thin disc (24 cells), VP 200 | 6.50 | 1.28 | 1.17 | 0.37 | 0.27 | 0.43 | 0.60 |
| **real plate (axisymmetric), VP 200** | 6.26 | 1.33 | 1.25 | 0.41 | 0.23 | 0.35 | 0.37 |
| real plate, VP 150 | 6.85 | 1.45 | 1.21 | 0.45 | 0.21 | 0.33 | 0.38 |
| real plate, DB 400 | 6.36 | 1.66 | 1.53 | 0.51 | 0.28 | 0.40 | 0.43 |

- The fitted roughness is now 10-15 mm (VP 150-300).
- The physical plate is as good below 20 deg and better above: 0.35-0.37 dB
  at 30-90 deg, against 0.43-0.60.
- The 0-2 deg conflict (it wants a much harder ground) is unchanged. That
  points past the plate, to refraction or a range-dependent ground.

### Validation (2026-09-25)

`tests/test_ground_validation.py` compares the models with references
obtained independently. The internal checks (limits, symmetry, formulation
agreement) are in `tests/test_ground_plane.py` and
`tests/test_axisymmetric_bem.py`. The full suite is 238 tests.

| component | independent reference | result |
|---|---|---|
| exact half-space Green's function (`image_integrals`, `exact_half_space_green`) | the Sommerfeld (Hankel-transform) integral, int J0(kappa rho) R(kappa) e^{i gamma Z} kappa/gamma dkappa, by direct quadrature; 15 cases over 3 grounds, kR 0.3-30 | agreement to **2e-9** relative. Weyl-van der Pol (Chien-Soroka) is off by up to 6% at these ranges and 2% at kR > 10, which is why it was replaced inside the plate |
| same | the impedance boundary condition, the rigid limit, and Weyl-van der Pol at long range | dG/dz = -i k beta G to 1e-6; image source to 1e-8; Q to 0.005 at 500 ft |
| axisymmetric BEM (`axisymmetric_bem.scattering`), including modes, Jacobi-Anger, near-singular kernels, image term and off-axis microphone | exact series for a rigid sphere. A rigid hemisphere on rigid ground is a sphere in free field struck by the direct and reflected waves. ka 0.5, 2, 5; three incidences; two microphone positions | max **2.7e-4** relative error (0.002 dB), median 3e-5 |
| three plate formulations: thin impedance-jump disc (`disc_bem_scattered`), 3-D raised plate (`raised_plate_scattering`), axisymmetric raised plate | each other, in the thin limit (0.3 mm) at 250, 500 and 1000 Hz, 5-70 deg, both azimuths | within 0.25 dB; the gap shrinks with mesh refinement at 2 kHz |
| real plate, axisymmetric vs 3-D surface | each other at 500 Hz and 2 kHz, 10 and 45 deg | within 0.3 dB (0.02-0.12 dB at 36 cells across) |
| plates on rigid ground; large plate | exact pressure doubling | thin plate 2 to 0.3-0.8%; a large plate doubles at 45 deg. The real 8 mm plate departs by a converged, physical O(kt) amount |
| impedance models (`surface_admittance`) | their published formulas: DB and Miki at X = 1; variable porosity against the classic 0.436 (1+i) sqrt(sigma_e/f); a deep layer tends to the half-space; passivity 10 Hz-20 kHz | all pass, except that **a thin, low-resistivity DB layer is not passive at low frequency** (0.03 m at sigma 100 below about 60 Hz). That is a known defect of the one-parameter model. The exact Green's function now refuses a non-passive ground rather than diverging. The fitted grounds are passive throughout |
| HARMONOISE coherence | hand calculation of eq. (18); a receiver on the ground gives 1 | exact |
| 46AE response | the GRAS table at its nodes; 0 below 500 Hz | exact |
| incidence geometry, certification axis | analytic cases (source overhead; axis normal to the flight-line plane) | exact |
| roughness factor | reduces the coherent reflection at steep incidence and vanishes at grazing | pass |

Not independently validated, and why:
- **The combined soft-ground plate problem has no closed form.** Its
  validation is the independently validated pieces (Green's function and
  solver), the agreement of the three independent formulations, and the
  comparison with the 2017 co-located pairs (the held-out-run scores above).
- **Published BEM correction curves** (Kingan et al. 2023/2024, Acoustics
  Australia 2026, Blandeau et al.) would be a further check. Their numbers
  were not accessible.

### In NICE-OPS (2026-09-25)

The ground models and the plate are now in NICE-OPS as well, on its spectral
path with `--ground_effect`:

- **Ground impedance, turbulence and roughness.** `--impedance_model`
  (`delany_bazley`, `miki`, `variable_porosity`, `delany_bazley_layer`),
  `--turbulence_model harmonoise --gamma_t`, and `--roughness`. They are the
  formulas of `surface_admittance`, `turbulence_coherence` and the Ament
  factor. NICE-OPS's tests check them against 15 `pole_level` cases to
  1e-6 dB.
- **The plate.** `axisymmetric_bem.write_netcdf` exports a `table()` as
  netCDF: band x sub x elevation x azimuth, in metres, with the ground named.
  NICE-OPS reads it with `--plate_table` and applies |P_d + Q P_r|^2,
  band-averaged, with Q from its own ground model, and refuses a table built
  for a different ground. NICE-OPS ships two tables at the databases' 31
  bands (10 Hz-10 kHz): Delany-Bazley 225 and variable porosity σe 200.
  - The exact form matches `board_level` to 1e-6 dB.
  - The run path takes Q as quadratic in log f through each band's edges and
    centre. That is 63 reflection coefficients per evaluation instead of
    155, within 0.07 dB wherever the level is above -15 dB.
  - A linear Q was tried and rejected. It was off by up to 0.8 dB above
    -15 dB (2.9 dB in the grazing nulls), because P_d - P_r nearly cancels
    there and any error in Q shows.
  - Interpolating band-averaged quadratic moments of P_d and P_r, instead of
    P_d and P_r themselves, was accurate to 0.065 dB but saves nothing once Q
    is quadratic.

### In the sphere builds (2026-09-26)

The 2017 spheres were built with the dataset's constant installation
correction: board pressures times 0.5 (-6 dB). That correction is now the
plate model, per band and per frame:

- **The hook.** `flight_acoustics.depropagate_hemisphere(...,
  receiver_response_db=...)` takes the installation's band-averaged response
  for each emission point (the source position minus the mic's). It divides
  each bin by its band's value after the ambient gate and before spreading,
  so each band's energy is divided by the plate's band-averaged response.
  Pressures then go in unscaled.
- **The builder.** `noise_abatement_2017.build_sphere(board_correction=
  'plate_bem')` is now the default; `'flat'` keeps the old -6 dB. Each mic's
  response is `axisymmetric_bem.board_level`:
  - on the GR1425 plate over `SITE_GROUND` (variable porosity, σe 200);
  - mic outboard of the track (on the -y side for a mic at y < 0);
  - Q at the run's sound speed.
- **Table cache.** Tables are cached per sound speed (0.5% steps) in
  `~/.cache/panam/plate_tables`, about 30 s each.
- **Inverted mics.** Every ground microphone in the NASA test was a flush
  67AX, including those the dataset labels `invgb7` (inverted, 7 mm gap), so
  they all use the flush model.
  - A true inverted layout is left for later tests. It needs the mic body
    over the gap. A bare point 7 mm up (the validated `mic_height` option of
    `axisymmetric_bem.scattering`) sits near the two-path null, 11 dB under
    doubling at 10 kHz overhead, which the real arrangement doesn't show.
- **Effect** on B407 run 283115 (level, 47 mics), plate minus flat, median
  over azimuth:
  - +0.2 to +0.4 dB below 100 Hz;
  - +0.8 to +2.6 dB at 0.5-2 kHz, largest toward the sidelines (the lowest
    elevations the 10° floor admits), where the plate's response is furthest
    below +6 dB;
  - within ±0.2 dB above 5 kHz.

  The co-located pole microphones implied this size and direction: the
  plate reads 2-4 dB under +6 dB at mid frequencies below 40°.

### Field maps, pole nulls and emission times (2026-09-28)

- **`axisymmetric_bem.field`** evaluates the BEM's surface solution anywhere in the air
  around the plate (p = p_inc + K p off the surface), for one frequency and one plane-wave
  direction. At a microphone point it reproduces `scattering` to about 1e-6.
  `board_field_plot.py` maps the level re free field around a plate on soft ground, together
  with the level along the microphone height.
  - At 4 kHz and 30 deg over grass (225 kPa s/m^2), the plate lifts the level near its top
    to about the +6 dB of a rigid plane. It rings at the edges and throws a diffraction
    shadow downwind.
  - The two possible positions of a 3/4-radius microphone differ by about 3 dB.
- **Pole interference nulls** (`ground_plane.path_difference`,
  `height_from_path_difference`, `null_frequencies`, `two_path_db`, `fit_two_path`,
  `reflection_phase`). The direct and reflected paths at a pole microphone come from the
  same emission, so their nulls fix the path difference whatever the source's phase.
  - Fitting the pole-minus-board narrowband difference gives dR. With the pole height,
    dR gives the arrival elevation or the source height.
  - This checks the flight geometry and the pole height independently of any level model.
  - Use run-to-run scatter for intervals: the least-squares standard errors ignore
    correlation between neighboring bins.
- **`ground_plane.emission_times`** solves t_e = t - |x(t_e) - receiver| / c to
  convergence for a trajectory given as a callable, with the receiver fixed or moving.
  A single pass leaves a moving source about M^2 R cos(phi) short along its path: about
  10 m at 600 m range and M 0.13.

## Candidate models

| Model | Mixed boundary | Grazing | Edge diffraction (ripple) | Cost per evaluation |
|---|---|---|---|---|
| Flat -6 dB (current) | no | no | no | none |
| Chien-Soroka over uniform ground (NICE-OPS now) | no | yes (soft only) | no | ~31 Faddeeva calls |
| Fresnel-zone weighting | yes | partly, via zone growth | no | closed form |
| De Jong semi-empirical diffraction | yes | yes | yes | one Fresnel integral per band and edge |
| Rasmussen impedance-jump integral | yes | yes | yes | numerical quadrature |
| BEM with tailored Green's function | yes, exact | yes | yes, plus board thickness | heavy, offline |
| Parabolic equation / FDTD | yes | yes, plus refraction | yes | far too heavy per sample |

### Fresnel-zone weighting

Hothersall & Harriott (1995) model two half-planes of impedance Z1 and Z2.
Away from the boundary the pressure is the uniform-ground result for whichever
side the specular point is on:

    A = 1 + Q (R1/R2) exp(ik(R2 - R1))

Q is the spherical-wave reflection coefficient for that side. In a transition
zone, where the path through the boundary exceeds the specular path by less
than lambda/3, the two are blended geometrically:

    |p/pf| = |A1|^r |A2|^(1-r),   with r = (D - D1)/(D2 - D1)

- This is what Mikkelsen & Nickerson use, with hard Q2 and soft Q1.
- Nord2000 and Harmonoise generalise it to a Fresnel ellipse on the ground.
  Each surface is weighted by the fraction of the ellipse it covers, which for
  a circular plate is an ellipse-circle overlap.
- It handles grazing better than its use by Mikkelsen & Nickerson shows. At low
  elevation the Fresnel ellipse stretches far past a 40 cm plate, so the soft
  ground takes over automatically at exactly the angles in question.
- Limitations:
  - the lambda/3 criterion is empirical;
  - it has no diffraction ripple from the plate edge;
  - its surface-wave behaviour is only what the soft side's Q carries.

### De Jong, Moerkerken & van der Toorn (1983)

- The model is the uniform-ground reflection plus (Q2 - Q1) times a
  Fresnel-integral diffraction term at the impedance edge.
- It was developed for grass-to-hard transitions at low heights, so grazing is
  its home ground.
- A reciprocity-corrected, multi-discontinuity extension exists (JASA 120:686,
  2006). A plate is two edges in the vertical plane of the ray, near and far.
- It captures the low-frequency edge ripple that a 40 cm plate produces
  (Mikkelsen & Nickerson and Giannakis & Nesbitt put edge effects below about
  500 Hz).

### Rasmussen (1982)

A more exact integral over the impedance jump, evaluated by quadrature. It is
more useful as an accuracy reference than as a production model.

### BEM

- Kingan, Go, Piscoya & Ochmann (JSV 2023) developed a BEM with a tailored
  Green's function, so only the board and microphone are meshed over an
  impedance half-space. Go, Kingan, Schmid & Hall (JSV 2024) apply it.
- The 2026 Acoustics Australia paper measures correction curves against
  frequency and incidence angle, for each microphone configuration and ground.
- Blandeau et al. (Airbus, ACTIPOLE BEM; AIAA 2018-3295, 2021-2141, 2023-4165)
  add porous ground under the plate (up to about 2 dB at 200-800 Hz) and gaps
  around its edge.
- Giannakis & Nesbitt (AIAA 2020-2612) derive a plate transfer function
  numerically and apply it with a short-time Fourier transform.
- Findings common to these: deviations from +6 dB grow at high frequency, at
  grazing incidence, and on soft ground.
- The BEM is exact, including grazing and plate thickness. It costs a lot but
  only once, offline, so it is the reference to judge the cheap models against.

### Parabolic equation / FDTD

Only worth it if atmospheric refraction joins the problem.

## Plan

1. **Measure it (panam).** Use the co-located pairs (50/34, 51/36, 52/38) on
   every aircraft and run.
   - Compute third-octave band time histories for both microphones, along with
     the elevation angle and range at emission.
   - Form board - pole level differences against frequency and elevation.
   - Recover the pole height and an effective ground flow resistance from the
     pole's interference pattern.
   - This gives an empirical ground-plane transfer function that needs no model.
2. **Compare corrections against each other and the data.** Candidates:
   - ideal rigid plane (the current constant -6 dB);
   - Fresnel-zone, both the Hothersall & Harriott line form and the
     Nord2000-style ellipse overlap for a 400 mm disc;
   - De Jong, two-edge;
   - where available, published BEM correction curves.

   Each is applied to the ground-plane record and judged against the pole
   record, itself de-reflected with Chien-Soroka on the fitted ground. The same
   comparison in the other direction (pole predicted from ground plane)
   reproduces Mikkelsen & Nickerson's test.
3. **Carry the winner into the pipelines.**
   - Replace `ground_board_scale=0.5` with the band- and angle-dependent
     correction when building spheres (it changes the high-frequency and
     low-elevation content of every sphere).
   - Offer it in NICE-OPS as a receiver transfer function, tabulated over band,
     elevation and, if grazing needs it, range, so a prediction can be made
     "as a ground-plane microphone would measure it".
   - A table lookup is far cheaper than the Chien-Soroka evaluation that
     dominates `--ground_effect` runs today. Its per-band maximum also gives
     `--db_cutoff` a much tighter bound than the current +9.6 dB worst case
     (`ground_gain_bounds`).
   - Chien-Soroka stays for community receivers at 1.2 m over grass.

## References

- D. Mikkelsen & M. Nickerson, "Pole Microphone Predictions Using Ground Plane
  Microphones", AIAA 2026-4033, AIAA AVIATION 2026.
- D. C. Hothersall & J. N. B. Harriott, "Approximate models for sound propagation
  above multi-impedance plane boundaries", JASA 97(2):918-926, 1995.
  <https://pubs.aip.org/asa/jasa/article/97/2/918/947659/Approximate-models-for-sound-propagation-above>
- B. A. De Jong, A. Moerkerken & J. D. van der Toorn, "Propagation of sound over
  grassland and over an earth barrier", JSV 86(1):23-46, 1983.
  <https://doi.org/10.1016/0022-460X(83)90941-0>
- "On the modeling of sound propagation over multi-impedance discontinuities
  using a semiempirical diffraction formulation", JASA 120(2):686, 2006.
  <https://pubs.aip.org/asa/jasa/article-abstract/120/2/686/893445/On-the-modeling-of-sound-propagation-over-multi>
- K. B. Rasmussen, "A note on the calculation of sound propagation over impedance
  jumps and screens", JSV 84(4):598-602, 1982.
- P. Boulanger, T. Waters-Fuller, K. Attenborough & K. M. Li, "Models and
  measurements of sound propagation from a point source over mixed impedance
  ground", JASA 102:1432-1442, 1997.
- Nord2000 comprehensive propagation model (FORCE Technology).
  <https://forcetechnology.com/-/media/force-technology-media/pdf-files/projects/nord2000/nord2000-nordtestproposal-rev4.pdf>
- D. van Maercke & J. Defrance, Harmonoise propagation model, Acta Acustica
  united with Acustica 93:201-212, 2007.
- M. J. Kingan, S. T. Go, R. Piscoya & M. Ochmann, "On the modelling of
  ground-board mounted microphones for outdoor noise measurements", JSV, 2023.
  <https://www.sciencedirect.com/science/article/pii/S0022460X23003437>
- S. T. Go, M. J. Kingan, G. Schmid & A. Hall, "On the use of ground-board
  mounted microphones for outdoor noise measurements", JSV 584:118432, 2024.
  <https://www.sciencedirect.com/science/article/pii/S0022460X24001950>
- "Correction Curves for Ground-Board Mounted Microphone Measurements",
  Acoustics Australia, 2026. <https://link.springer.com/article/10.1007/s40857-026-00407-0>
- J. M. Giannakis & E. H. Nesbitt, "Evaluation of a Correction Factor for
  Flyover-Noise Ground Plane Microphones", AIAA 2020-2612.
  <https://doi.org/10.2514/6.2020-2612>
- V. P. Blandeau et al., AIAA 2018-3295, 2021-2141 and 2023-4165 (ground plates,
  infinite impedance plane modelling, propagation around and under plate edges).
- M. Albert, P. Bousquet & D. Lizarazu, "Ground Effects For Aircraft Noise
  Certification", AIAA 2017-3845.
- B. N. Shivashankara & G. W. Stubbs, "Ground plane microphone for measurement of
  aircraft flyover noise", J. Aircraft 24(11):751-758, 1987.
- C. F. Chien & W. W. Soroka, "Sound propagation along an impedance plane",
  JSV 43(1):9-20, 1975.
