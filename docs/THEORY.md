# PANAM theory

This document states the models PANAM implements, as equations, with the references they come
from and the defaults the code uses. It describes the code as it is. Where PANAM departs from a
cited reference, the departure is stated. The settings a particular build chose are not given
here: those of the 2017 release databases are in
[`database_build.md`](database_build.md#the-2017-release-settings).

Related documents: [`../README.md`](../README.md) (installation and tools),
[`database_build.md`](database_build.md) (building the 2017 databases),
[`file_formats.md`](file_formats.md) (the sphere netCDF file, the `.nod` database and
`vehicle.cfg`), [`norah2_import.md`](norah2_import.md) (converting NORAH2 hemispheres) and
[`DEVELOPING.md`](DEVELOPING.md) (contributing). How the ground-plate model was fitted is
recorded in [`notes/ground_plane_corrections.md`](notes/ground_plane_corrections.md), a dated
research log.

PANAM's main product is the database of noise spheres that the NICE-OPS footprint model reads.
The NICE-OPS theory document covers what happens to a sphere after that: blending, propagation to
the ground and the metrics. Several of PANAM's models are the reference NICE-OPS's ports are pinned
to (ISO 9613-1, the ground effect, A-weighting, EPNL).

References are numbered in order of first citation and listed at the end. The appendix
[Where each model lives](#appendix-where-each-model-lives) maps each section to its functions and
tests.

## Contents

1. [Overview and notation](#1-overview-and-notation)
2. [Signals and spectra](#2-signals-and-spectra)
3. [Emission geometry and timing](#3-emission-geometry-and-timing)
4. [Ground and receiver models](#4-ground-and-receiver-models)
5. [Depropagation](#5-depropagation)
6. [Gridding onto the sphere](#6-gridding-onto-the-sphere)
7. [The database](#7-the-database)
8. [NORAH2 and AAM interchange](#8-norah2-and-aam-interchange)
9. [Metrics](#9-metrics)
10. [Auxiliary models](#10-auxiliary-models)
11. [What is not modeled](#11-what-is-not-modeled)
12. [Appendix: Where each model lives](#appendix-where-each-model-lives)
13. [References](#references)

## 1. Overview and notation

PANAM turns flight-test microphone recordings into source noise spheres. The approach is the
fixed-radius source hemisphere of the Fundamental Rotorcraft Acoustic Modeling from Experiments
(FRAME) method [1, 2, 3]. Each steady flight condition gets
one sphere: the band levels the aircraft radiates in every direction, referred to a fixed radius.
The shipped databases come from the 2017 Noise Abatement flight test [4, 5].

For each run, PANAM:

1. computes short-time spectra at each ground microphone (§2);
2. maps each spectral frame to the direction and distance at which it was emitted (§3);
3. removes the ambient noise (§5.1) and the microphone installation's response (§4), and undoes
   the spreading and absorption back to the reference radius (§5);
4. grids the scattered samples onto a regular sphere, in energy (§6).

A database then collects the spheres by flight condition (§7).

The symbols used in more than one section are listed below. A symbol used in one section only is
defined where it appears. Where a letter would otherwise carry two meanings, one of them has a
subscript or a different letter, as the last rows note.

| Symbol | Meaning |
| --- | --- |
| $f$, $k = 2\pi f/c$ | frequency; acoustic wavenumber |
| $c$, $c_0$ | speed of sound (the run's, §5.4); the same in the ground models (§4.1) |
| $S(f) = S_{xx}(f)/p_\mathrm{ref}^2$ | linear power spectral density, re $(20\ \mu\mathrm{Pa})^2$/Hz |
| $\mathrm{PSD}(f) = 10\lg S(f)$ | the same in dB |
| $P$ | mean-square pressure (power) in a bin or band, re $(20\ \mu\mathrm{Pa})^2$ |
| $L = 10\lg P$ | level, dB re 20 µPa |
| $\Delta f$, $T_w$ | FFT bin width; requested analysis window duration (`window_time`); the frame is the next power of two of $T_w f_s$ samples (§2.1) |
| $r$, $r_\mathrm{ref}$ | source–microphone distance; sphere reference radius, 100 ft (30.48 m) |
| $e$, $\psi$ | emission elevation (depression below the horizontal) and azimuth from the ground track |
| $\theta$, $\phi$ | AAM/ART sphere angles: polar angle from the nose, roll angle from straight down |
| $\mathbf{v}$, $M_r$, $D$ | source velocity; Mach number toward the microphone; Doppler factor $1/(1 - M_r)$ |
| $h_s$, $h_r$, $d$ | source height, receiver height, horizontal distance |
| $R_1$, $R_2$, $\tau$ | direct and image path lengths; path delay $(R_2 - R_1)/c$ |
| $\beta = 1/Z$ | normalized ground admittance ($Z$ the normalized impedance) |
| $Q$, $\vartheta$ | spherical-wave reflection coefficient; image path's angle from the ground normal |
| $T$, $p_a$, $H$ | air temperature (K), pressure (kPa), relative humidity (%) |
| $\alpha(f)$ | ISO 9613-1 absorption coefficient, dB/m |
| $T_\mathrm{board}$, $P_d$, $P_r$ | plate-microphone response re free field, dB; its direct and reflected transfer functions (§4.5) |
| $W_A(f)$ | A-weighting, dB |
| $\mu$, $\gamma$, $C_T$, $C_W$, $n_z$ | advance ratio, flight path angle, thrust and weight coefficients, load factor (§7) |
| $R_\mathrm{mr}$, $V_\mathrm{tip}$, $W$ | main-rotor radius, tip speed, vehicle weight |
| $d_g$, $R_n$ | geodesic (angular) distance on the sphere; a node's gridding radius (§6) |
| $\eta_C$, $\mu_C$ | Chessell's band-average constants (§4.3), not the advance ratio |
| $\chi$, $\chi_p$ | azimuth about the plate's axis; plate-frame propagation azimuth (§4.5), not $\psi$ or $\phi$ |
| $\mathcal{C}_T^2$, $\mathcal{C}_v^2$ | temperature and wind structure parameters (§4.4), not the thrust coefficient |
| $c_t$ | Chessell's turbulence parameter (§4.3) |
| $\gamma_\mathrm{air}$ | ratio of specific heats, 1.4 (§4.1), not the flight path angle |
| $p_v$ | water vapor pressure (§5.4, §8), not the elevation $e$ |

The time dependence is $e^{-i\omega t}$, so a passive ground has $\mathrm{Re}\,\beta > 0$. Lengths
are in feet in the 2017 build and the ground models, and in meters in the database; each function
states its units.

## 2. Signals and spectra

### 2.1 Short-time spectra

Microphone pressure (Pa, as stored; PANAM applies no calibration) is analyzed in Hann-windowed
frames [6], with the mean removed from each frame (`spectrogram`'s default
`detrend='constant'`). The frame length is the power of two at or above $T_w f_s$ samples
(`frame_length`). The linear spectral density is $S(f) = S_{xx}(f)/p_\mathrm{ref}^2$,
$p_\mathrm{ref} = 20\ \mu\mathrm{Pa}$, and its level is

$$
\mathrm{PSD}(f) = 10\lg S(f)\quad \text{dB re } (20\ \mu\mathrm{Pa})^2/\mathrm{Hz}.
$$

`depropagate_hemisphere` and `build_sphere` default to $T_w = 0.5$ s with 50% overlap, so a frame
is 16,384 samples: 0.64 s and $\Delta f = 1.5625$ Hz at 25.6 kHz, and 0.655 s and
$\Delta f = 1.526$ Hz at 25.0 kHz (the 2017 data mix the two rates; each microphone is analyzed on
its own frequency grid). Each frame is a single periodogram. The averaging happens later, over the
samples that fall in a sphere cell (§6). The spectrum at an emission point's reception time is
interpolated between frame centers **in power**, not in dB.

Welch averaging [7] is available for stationary signals (`psd_welch`). The whole-record
spectrum (`psd`, and through it `third_octave_band_levels`, `overall_SPL` and `level_history`) is
one Hann-windowed, density-scaled periodogram; the Hann window keeps an off-bin rotor tone's
leakage near the tone, where a rectangular window spreads it across the spectrum. `overall_SPL`'s
unweighted level is the record's mean-square pressure (mean removed); its A-weighted level scales
that by the A-weighted fraction of the Hann spectrum.

### 2.2 One-third-octave bands

Band edges follow IEC 61260-1 [8] (`third_octave_band_edges`). A band with a nominal
center (within 1/12 octave of $1000\cdot10^{n/10}$ Hz, $n$ an integer) takes the base-10 edges
$f_m 10^{\pm 1/20}$ around its exact midband frequency $f_m$. A set of exact base-2 centers
$1000\cdot2^{n/3}$ takes $f_c 2^{\pm1/6}$; `third_octave_band_levels` uses such centers. The
sphere build uses the 31 nominal centers from 10 Hz to 10 kHz, so its edges are base 10.

**Allocation.** By default a band's power is the sum of whole FFT bins inside its edges:
$P_b = \sum_{f_l \le f < f_u} S(f)\,\Delta f$. With a Doppler factor (§3.3) or tone-aware
banding, the running integral $R(f) = \sum S\,\Delta f$ is interpolated linearly within bins, and
$P_b = R(f_u) - R(f_l)$ counts fractional bins.

**Tone-aware banding** (`tone_aware=True`; off by default). A rotor harmonic near a band edge
splits its window main lobe between two bands. `tone_aware_band_power` files each tone whole:

1. The floor is the running median of $S$ over 15 bins, taken over the bins the ambient gate left
   nonzero. A tone is a local maximum over ±2 bins standing more than 6 dB above it.
2. Its offset from the peak bin follows from the Hann window's two-bin interpolation
   [9], with $S_0$ and $S_{l,r}$ the floor-subtracted peak and neighbors:
   $u = \sqrt{\max(S_l, S_r)/S_0}$, $\delta = \pm\,\mathrm{clip}\!\left(\frac{2u - 1}{u + 1}, 0, \tfrac12\right)$, toward the larger neighbor.
3. Its power corrects the peak bin for the window's scalloping, with $\mathrm{ENBW} = 1.5$ bins:

$$
P_\mathrm{tone} = \frac{S_0\, \mathrm{ENBW}\, \Delta f}{\lvert H(\delta)\rvert^2}, \qquad
\lvert H(x)\rvert^2 = \left(\frac{\operatorname{sinc} x}{1 - x^2}\right)^2 .
$$

4. Its contribution $P_\mathrm{tone}\lvert H(o - \delta)\rvert^2/(\mathrm{ENBW}\,\Delta f)$ is
   removed from bins $o = -4 \dots 4$ (clipped at zero). The remainder is banded by the running
   integral, and the tone is added whole to the band containing $f_I + \delta\Delta f$ (divided by
   $D$ when the Doppler shift is removed).

**Filter bank** (`third_octave_method='filter_bank'`; the default is the FFT). Each band is an
order-3 Butterworth band-pass between $f_c 2^{\mp1/6}$ about the band's given center (nominal or
exact), so its edges are not the base-10 edges the FFT path uses
(`panam_acoustics.filters.third_octave_filter_bank`). The record has its mean removed and is
reflected at both ends; each band is filtered causally at a rate decimated octave by octave to
about 12 times its upper edge. Each band's output is then advanced by the filter's group delay at
$f_c$ (about 0.14 s at 20 Hz), so that low-band energy stays at the emission angle it arrived from.
The squared output is averaged with Hann² weights over the FFT frame length, centered on the FFT
frames, so it is comparable with a PSD band sum. In this mode the ambient gate, the subtraction and
the absorption cap act per band, and absorption is taken at the band center.

The banding the 2017 release used is in [`database_build.md`](database_build.md#the-2017-release-settings).

### 2.3 A-weighting

$$
W_A(f) = 10\lg\frac{1.562339\, f^4}{(f^2 + 107.65265^2)(f^2 + 737.86223^2)}
       + 10\lg\frac{2.242881\times10^{16}\, f^4}{(f^2 + 20.598997^2)^2 (f^2 + 12194.22^2)^2},
$$

the pole frequencies of IEC 61672-1 [10] (12194.22 Hz where the standard gives 12194.217),
normalized to 0 dB at 1 kHz (`dBAw`). It is applied per FFT bin in depropagation and per band
center in the EAA (§7.2).

## 3. Emission geometry and timing

### 3.1 Reception time

The track gives the source position $\mathbf{x}_s(t_e)$ and velocity $\mathbf{v}$ at emission
times $t_e$. For each microphone at $\mathbf{x}_m$, the emission maps forward to reception
(`hemigen`):

$$
\mathbf{r} = \mathbf{x}_m - \mathbf{x}_s(t_e), \qquad
t_\mathrm{obs} = t_e + \frac{\lvert\mathbf{r}\rvert}{c}, \qquad
M_r = \frac{\mathbf{v}\cdot\mathbf{r}}{\lvert\mathbf{r}\rvert c},
$$

with $c = 343.2\sqrt{T/293.15}$ m/s at the run's station temperature (§5.4) and no wind. No
retarded-time equation has to be solved: the track is sampled in emission time. With a ray model
(§5.3), $t_\mathrm{obs} = t_e + t_\mathrm{ray}$. Emission samples can be decimated
(`point_stride`, default 1).

### 3.2 Directions

PANAM files a sphere in a **level, horizontal frame**. The elevation is the depression of the
microphone below the horizontal at the source, and the azimuth is measured from the aircraft's
horizontal ground track:

$$
e = \operatorname{atan2}\!\left(z_s - z_m,\ \sqrt{r_x^2 + r_y^2}\right), \qquad
\psi = \left(\operatorname{atan2}(r_y, r_x) - \Psi - 180^\circ\right) \bmod 360^\circ,
$$

with $\Psi = \operatorname{atan2}(v_y, v_x)$ the track direction, or the inertial heading for a
hover and under `azimuth_reference='heading'`. This "UMAPR" convention puts 180° ahead, 90° to
starboard and 0° aft. There is no tip-path-plane tilt. In steady flight NICE-OPS's sphere frame
coincides with it, because its tip-path-plane normal is the load-factor direction, which is
vertical when the aircraft does not accelerate.

The AAM/NORAH2 ("ART") sphere angles are $\theta$ from the nose and $\phi$ from straight down,
positive to starboard [11] (`art2umapr`):

$$
\hat{\mathbf{d}} = (\cos\theta,\ \sin\theta\sin\phi,\ \sin\theta\cos\phi)\quad(\text{forward, starboard, down}).
$$

**Track frame.** The 2017 tracks are in each layout's local frame: $+x$ along the layout's true
bearing $b$, $+y$ to the left, $z$ up. Latitude and longitude map to local east and north on the
WGS84 radii of curvature at the run's reference point [12] (`local_east_north_ft`), then
rotate by $b$ (`east_north_to_frame`): $x = E\sin b + N\cos b$, $y = -E\cos b + N\sin b$.
`frame_bearing_deg` takes $b$ from the reference list's `true_heading`. When the track carries
latitude and longitude, the run has a reference point, and the track spans at least 500 ft, it
also fits $b$ from the track and refuses a disagreement over 1°. Otherwise the listed bearing is
used as given, so a hover's drift is never checked. The bearings of the 2017 layouts are listed in
[`database_build.md`](database_build.md#the-track-frame).

### 3.3 Doppler

The received frequency is $f = D f_e$, with $D = 1/(1 - M_r)$ along the straight line. By default
(`DOPPLER_SHIFT_REMOVED = 0`) the spheres keep the received, Doppler-shifted spectra, and the
bands are the received bands. `remove_doppler=True` instead integrates each emitted band
$[f_l, f_u]$ over $[D f_l, D f_u]$ of the received spectrum (with `tone_aware` also set, tones
are located and filed whole at $f_\mathrm{peak}/D$); a band whose scaled edges leave the analyzed range is missing for that
sample. Only frequencies are rescaled: no convective amplification factor $D^n$ is applied to the
levels. A sphere built this way is meant as the source of a synthesized hover (§7.4), not as a
flight sphere, since NICE-OPS applies no shift.

### 3.4 Steady segments

`steady_window` gives each run its longest steady segment, and `flight_condition` labels it with
the segment's mean ground speed and mean flight path angle. The reference values are the medians
of ground speed and flight path angle over the half of the record nearest (in slant distance) the
run's reference point, the track-frame origin. With the defaults, a sample is steady when it lies
within 4 kt and 2° of them, its roll is within 5°, and its heading changes by less than 1.5°/s;
gaps of 2 s or less are closed, and the segment must last at least 8 s (`steady_window`'s
keyword arguments, and `--min-steady-duration-s` for the last). A hover
(`nose_from_heading=True`) uses the whole record, with the nose along the heading and no Doppler or
convective term.

## 4. Ground and receiver models

### 4.1 Ground impedance

`ground_plane.surface_admittance` provides four locally reacting models, with $X = f/\sigma$
($\sigma$ in kPa·s/m²), $\rho_0 = 1.2$ kg/m³ and $\gamma_\mathrm{air} = 1.4$:

| Model | Impedance |
| --- | --- |
| Delany–Bazley [13] | $Z = 1 + 9.08X^{-0.75} + 11.9iX^{-0.73}$ |
| Miki [14] | $Z = 1 + 5.50X^{-0.632} + 8.43iX^{-0.632}$ |
| Variable porosity [15] | $Z = (1+i)\sqrt{\dfrac{1000\,\sigma_e}{\pi\gamma_\mathrm{air}\rho_0 f}} + \dfrac{i c_0 \alpha_e}{8\pi\gamma_\mathrm{air} f}$ |
| Hard-backed Delany–Bazley layer, depth $d_l$ [15] | $Z = Z_c\coth(-ik_l d_l)$ |

For the layer, $Z_c = 1 + 9.08X^{-0.754} + 11.9iX^{-0.732}$ and
$k_l = (2\pi f/c_0)(1 + 10.8X^{-0.70} + 10.3iX^{-0.595})$. $c_0$ is the sound speed the caller
passes: the run's in the plate tables (§4.5) and the pole model (§4.4), 343 m/s when none is given.
The first term of the variable-porosity form is the familiar $0.436(1+i)\sqrt{\sigma_e/f}$ with
$\sigma_e$ in Pa·s/m².

**The $\alpha_e$ convention.** The second term's coefficient, $c_0/(8\pi\gamma_\mathrm{air})
\approx 9.75$ m/s, is what a first-order (WKB) solution gives for porosity $\Omega_0 e^{-\alpha z}$
with isothermal compressibility and $\alpha_e = \alpha/\Omega_0$. The form usually quoted,
$0.436(1+i)\sqrt{\sigma_e/f} + 19.48\,i\alpha_e/f$, has twice the coefficient,
$c_0/(4\pi\gamma_\mathrm{air})$, so its $\alpha_e$ is defined differently: an $\alpha_e$ fitted in
that form is halved here. The printed eq. (10) of [15] has not been checked against
this. The 2017 site ground uses $\alpha_e = 0$ (§4.6), where the two agree.

A thin layer of low resistivity is not passive at low frequency ($\mathrm{Re}\,\beta < 0$). The
exact Green's function used by the plate BEM (§4.5) refuses it; the impedance,
reflection-coefficient and pole models use it unchecked.

### 4.2 The spherical-wave reflection coefficient

For a source at height $h_s$ and a receiver at $h_r$, a horizontal distance $d$ apart, the
direct and image paths are $R_1 = \sqrt{d^2 + (h_s - h_r)^2}$ and
$R_2 = \sqrt{d^2 + (h_s + h_r)^2}$. With $\cos\vartheta = (h_s + h_r)/R_2$, measured from the
normal (`spherical_reflection_coefficient`'s argument `cos_grazing` is $\cos\vartheta$, despite its
name; `ega()` calls $\vartheta$ `incidence_angle`):

$$
R_p = \frac{\cos\vartheta - \beta}{\cos\vartheta + \beta}, \qquad
Q = R_p + (1 - R_p)\left[1 + i\sqrt{\pi}\,p_w\,\mathrm{w}(p_w)\right], \qquad
p_w = \sqrt{\frac{i\pi f R_2}{c\,(1 + \beta\cos\vartheta)}}\,(\cos\vartheta + \beta).
$$

This is the corrected Chien–Soroka form [16, 17]. $\mathrm{w}(z) = e^{-z^2}\mathrm{erfc}(-iz)$
is the Faddeeva function (SciPy's `wofz`). The code keeps $p_w$ itself rather than the principal
root of $w = p_w^2$. The two differ only for a mass-like ground ($\mathrm{Im}\,\beta > 0$), where the
principal root would admit a growing surface wave. There is no cutoff in $\lvert w\rvert$. Without
an admittance given, $\beta$ is Delany–Bazley's at the flow resistance passed.

### 4.3 Excess ground attenuation: `ega()`

`ega()` is the ground-effect model of the Wyle short-range propagation model
[18], after Chien and Soroka and Chessell [19], with the corrections noted
by Daigle, Embleton and Piercy [20]. It uses the Delany–Bazley impedance and the path
delay $\tau = (R_2 - R_1)/c$. For a pure tone (`pt=True`, the default):

$$
\Delta L = 20\lg\left\lvert 1 + Q\,\frac{R_1}{R_2}\, e^{2\pi i f\tau}\right\rvert .
$$

For a one-third-octave band (`pt=False`), Chessell's band average, with $q_r = \lvert Q\rvert R_1/R_2$:

$$
\Delta L = 10\lg\left[1 + q_r^2 + 2q_r\cos(\eta_C f\tau + \arg Q)\,\frac{\sin(\mu_C f\tau)}{\mu_C f\tau}\, C_\mathrm{turb}\right],
\qquad C_\mathrm{turb} = \exp\!\left[-\left(\tfrac12 c_t f \sqrt{R_1}\right)^2\right],
$$

where $\eta_C = \pi(2^{1/6} + 2^{-1/6}) = 6.325159$ and $\mu_C = \pi(2^{1/6} - 2^{-1/6}) = 0.727477$.
They come from averaging $\cos(2\pi f'\tau + \arg Q)$ uniformly in $f'$ over the band
$[f2^{-1/6}, f2^{1/6}]$. $C_\mathrm{turb}$ is Chessell's turbulent decoherence, with $c_t$
(`cturb`, default 0) in rad·s per square root of the length unit (typically 0–16×10⁻⁴ rad·s/√m).
NICE-OPS's ground model is ported from this function and pinned to it at $10^{-9}$ dB.

### 4.4 Pole microphones

`pole_level` is the band-averaged level of a point microphone above any of the four grounds, re
free field. It generalizes `ega()` (to which it reduces, within $10^{-9}$ dB, with Delany–Bazley and
no turbulence, roughness or microphone response):

$$
E_p = a_d^2 + \left(a_r q_r\right)^2 + 2a_d a_r q_r\cos(\eta_C f\tau + \arg Q)\,\frac{\sin(\mu_C f\tau)}{\mu_C f\tau}\,C_\mathrm{turb},
\qquad \Delta L = 10\lg E_p,
$$

with $a_d$ and $a_r$ the microphone's free-field amplitude response at each path's angle of
incidence (1 when not given). `free_field_microphone_response` gives that response for the 2017
pole microphones (GRAS 46AE, a 40AE capsule) from the manufacturer's free-field corrections
[21], re on-axis incidence, interpolated linearly in angle and in log frequency, and 0 below
500 Hz. `pole_incidence` gives the two paths' incidence angles on the microphone's axis, and
`certification_axis` the certification orientation: the axis normal to the plane containing the
flight line and the pole, so that sound from the flight line arrives at grazing (90°) incidence.
Two refinements act on the reflected path:

- **Turbulence** reduces its coherence with the HARMONOISE form [22, 15],
  $C_\mathrm{turb} = \exp\left[-\tfrac38 B k^2 \gamma_p^{5/3} R_1 \gamma_T\right]$, with $B = 0.364$,
  $\gamma_p = h_s h_r/(h_s + h_r)$, and the strength
  $\gamma_T = \mathcal{C}_T^2/T_\mathrm{mean}^2 + 22\mathcal{C}_v^2/(3c_0^2)$ (`gamma_t`, default 0)
  from the temperature and wind structure parameters [23] and the mean air temperature
  $T_\mathrm{mean}$ (K), lengths in meters. The caller forms $\gamma_T$; `pole_level` takes it as given.
- **Roughness** of rms height $\sigma_h$ (`roughness`, default 0) reduces its amplitude by the
  Kirchhoff (Ament) factor [24]: $a_r \leftarrow a_r\exp[-2(k\sigma_h\cos\vartheta)^2]$.

### 4.5 Ground-plate microphones

The 2017 ground microphones are GRAS 67AX microphones flush in GR1425 plates lying on the
ground: radius $a = 0.2$ m, 8 mm thick, tapering to a 2.5 mm edge (a 20 mm taper length is
assumed), with the microphone at $0.75a$ from the center, outboard of the track
[4, 25]. This holds for the channels the dataset labels `invgb7` ("inverted over a ground
board with a 7 mm gap") too: they were flush as well, so `noise_abatement_2017.BOARD_MIC_HEIGHT_FT`
puts both instrument types at height 0. A plate is not an ideal pressure-doubling surface: it is finite and
raised, and the ground around it is not rigid [26]. Kingan, Go and co-workers model
ground-board microphones with a boundary element method over impedance ground [27, 28].
PANAM computes the response with an axisymmetric (body-of-revolution) boundary element method
[29] of the plate's own profile (`axisymmetric_bem.py`).

**The Green's function** is the exact one for a point source over a locally reacting plane
[30, 31] (`ground_plane.exact_half_space_green`):

$$
G = g(R_1) + g(R_2) - 2k\beta\int_0^\infty e^{-k\beta q}\, g(R_q)\, dq, \qquad
g(R) = \frac{e^{ikR}}{4\pi R}, \qquad R_q = \sqrt{\varrho^2 + (Z_s + iq)^2},
$$

with $\varrho$ the horizontal distance and $Z_s$ the summed heights of the two points. The
integral is evaluated by graded Gauss–Legendre quadrature (`image_integrals`) and tabulated per
frequency (`ImageIntegralTable`). It agrees with the Sommerfeld integral to $2\times10^{-9}$
relative.

**The integral equation.** With $G$ satisfying the ground's boundary condition, only the rigid
plate's surface $\Sigma$ remains, and the exterior Neumann problem is

$$
\tfrac12\, p(\mathbf{x}) = p_\mathrm{inc}(\mathbf{x}) + \int_\Sigma p(\mathbf{y})\,\frac{\partial G}{\partial n_y}\, d\Sigma_y .
$$

The plate is a body of revolution. Expanding $p = \sum_m p_m(\xi)e^{im\chi}$ in the azimuth $\chi$
about its axis, with $\xi$ the arc length along the generating curve (flat top, taper, rim),
decouples the modes, $|m| \le \lceil ka\rceil + 10$. Each mode solves a one-dimensional equation on
the generating curve, collocated at segment midpoints (segments of about $\min(0.02\ \mathrm{ft},
\lambda/10)$, graded finer toward the corners). A plane wave from elevation $e$ whose horizontal
propagation direction has plate-frame azimuth $\chi_p$ expands by the Jacobi–Anger identity:

$$
p_\mathrm{inc} = \sum_m i^m J_m(k r_p\cos e)\, e^{im(\chi - \chi_p)}\, e^{\mp ik z\sin e},
$$

with $r_p$ and $z$ the radial and vertical coordinates on the plate, the upper sign for the direct
(downward) wave and the lower for the ground-reflected (upward) one. $\chi_p$ is measured from
$+x$ toward $+y$ in the plate's frame, where the microphone is offset along $+y$; for a microphone
on the $-y$ side of the track the azimuth is mirrored (`mirror_y`), which puts its offset on $-y$.
A flush microphone takes the surface value $2(p_\mathrm{inc} + \int_\Sigma p\,\partial G/\partial n\,d\Sigma)$.

**The response.** The direct and ground-reflected incident waves are solved separately, giving
transfer functions $P_d$ and $P_r$ at the microphone, both normalized by the direct wave there
(`axisymmetric_bem.scattering`). The field at the microphone is $P_d + Q P_r$, with $Q$ from §4.2
evaluated at the plate's top. The band response averages five sub-frequencies
$f_j = f_c 2^{(j + 0.5)/15 - 1/6}$ about the band's given center (`ground_plane.sub_band_factors`):

$$
T_\mathrm{board} = 10\lg\left[\frac15\sum_{j=0}^{4}\left\lvert P_d(f_j) + Q(f_j)P_r(f_j)\right\rvert^2\right]\quad \text{dB re free field}
$$

(`axisymmetric_bem.board_level`). $P_d$ and $P_r$ are tabulated (`axisymmetric_bem.table`) on 80
elevations (40 geometrically spaced from 0.05° to 10°, 40 evenly from 11° to 90°) and 36 azimuths
(every 10°), and interpolated bilinearly. `noise_abatement_2017.plate_table` computes a table at
the run's sound speed rounded to a 0.5% step in $\ln c$ and caches it in memory and on disk. The
tables can be written for NICE-OPS's `--plate_table` (`axisymmetric_bem.write_netcdf`). Checks: a
rigid sphere against its exact series [32] to $2.7\times10^{-4}$, a thin plate on rigid
ground within 1% of pressure doubling, and agreement with the 3-D surface-mesh BEM of §10.

`build_sphere`'s `board_correction='plate_bem'` (the default) divides each band by this response;
`'flat'` instead applies a uniform pressure-doubling correction, amplitude × 0.5 (−6.02 dB).

### 4.6 The site ground

The 2017 site ground is `noise_abatement_2017.SITE_GROUND`: variable porosity with
$\sigma_e = 200$ kPa·s/m², $\alpha_e = 0$, and no roughness. How it was fitted to the co-located
pole and plate microphones is recorded in
[`notes/ground_plane_corrections.md`](notes/ground_plane_corrections.md). Functions in
`ground_plane.py` called with a flow resistance and no `ground` model use Delany–Bazley at their
default `FLOW_RESISTANCE = 100` kPa·s/m², an earlier fit, not the site ground.

## 5. Depropagation

### 5.1 The sequence

Each emission sample carries a received power spectrum $P(f)$ at a microphone. Depropagation
(`depropagate_hemisphere`) refers it to the sphere radius $r_\mathrm{ref}$ (default 100 ft) in four
steps, in this order:

1. **Ambient gate.** With $A(f)$ the ambient spectrum, a bin is kept only if it stands $L_G$ dB
   above the ambient, and the ambient is then subtracted:
   $P \leftarrow \max(P - A, 0)$ where $P \ge A\,10^{L_G/10}$, and 0 elsewhere. $A(f)$ is one of:
   the mean power over the frames of a separate ambient recording (`ambient_pressure`; the 2017
   build uses the measured ambient run on the same layout), the mean over a quiet window of the
   run itself (`ambient_time_range`), or a low percentile of the run's own spectrogram
   (`ambient_percentile`).
2. **Receiver response.** $P \leftarrow P\cdot10^{-T_\mathrm{board}/10}$ (§4.5), each bin divided
   by its band's value. A cap (`max_response_correction_db`) discards bands whose correction would
   exceed it.
3. **Spreading.** $P \leftarrow P\,(r_s/r_\mathrm{ref})^2$, with $r_s$ the straight distance or the
   ray tube's equivalent range.
4. **Absorption** (when `apply_absorption_deprop=True`).
   $P \leftarrow P\cdot10^{\alpha(f)(\ell - r_\mathrm{ref})/10}$, with $\alpha$ the ISO 9613-1
   coefficient at the received frequency (§5.4) and $\ell$ the straight distance or the ray's arc
   length. A cap (`max_absorption_correction_db`) discards bins whose correction would exceed it.

The gate comes first: an ambient-limited bin would otherwise be amplified by both corrections. The
ambient is a mean because each $P$ is a single frame, whose expectation is the mean; a single
frame's bin of noise is exponentially distributed, so its median sits $\ln 2$ of the mean (1.6 dB)
low. A gated bin carries zero power (measured, no energy); a capped bin is missing (not measured):
a band spanning one is missing for that sample, the overall levels sum the bins that remain, and
the gridding (§6) leaves missing samples out of each node's weights. This distinction holds up to
the gridded hemisphere; the sphere file and the database merge the two (§6.3). **The sphere keeps
the absorption over its first $r_\mathrm{ref}$** in the run's own atmosphere; the database's EAA
(§7.2) accounts for absorption beyond it.

**Defaults.** `depropagate_hemisphere` on its own undoes spreading only: absorption is off
(`apply_absorption_deprop=False`), and there is no gate unless an ambient is supplied (then
$L_G = 3$ dB). It has no caps, no range limit and a 0° elevation floor. `build_sphere` turns
absorption on, gates against the layout's measured ambient run (and refuses a run without one
unless `ambient_fallback_percentile` is given), and defaults to $L_G = 10$ dB, a 30 dB absorption
cap, no response cap, a 2000 ft range limit and a 10° elevation floor. The 2017 release's values
are in [`database_build.md`](database_build.md#the-2017-release-settings).

### 5.2 Range and elevation limits

A sample is dropped if its filing elevation is below `min_elevation_deg`, if its straight-line
distance exceeds `max_range` (with a longer `rim_range` allowed for samples filed less than a given
elevation below the horizon; `build_sphere` takes 14° when only the rim range is given), if its ray
is in a shadow, or if its reception time falls outside the recording. With a ray model the
elevation floor and the rim test apply to the ray's launch angle, while `max_range` and the rim
range still apply to the straight-line distance.

### 5.3 Refracted rays

PANAM does not trace rays itself. `refracted_rays.external_ray_model` calls NICE-OPS's
`niceops_ray_geometry` through the run's measured atmosphere, and takes from it the launch angle,
the travel time, the ray-tube spreading range, the arc length, and the geometry the plate sees at
the ray's arrival. The ray physics is in the NICE-OPS theory document. A sample is then filed at
its launch angle rather than the straight-line elevation, at the straight line's azimuth. The
absorption coefficient stays uniform along the ray. `refracted_rays.straight_ray_model` gives the
straight-line answer in the same form. Without a ray model (`ray_model=None`, the default) the
paths are straight lines. The 2017 release's ray model is in
[`database_build.md`](database_build.md#the-2017-release-settings).

### 5.4 The atmosphere

**Absorption** is the ISO 9613-1 coefficient [33] (`panam_acoustics.iso_9613_1_1993`). With
$T_0 = 293.15$ K, $T_{01} = 273.16$ K, $p_{a0} = 101.325$ kPa, temperature $T$, pressure $p_a$ and
relative humidity $H$ (%):

$$
h_w = H\,10^{\,-6.8346(T_{01}/T)^{1.261} + 4.6151}\,\frac{p_{a0}}{p_a}, \qquad
f_{rO} = \frac{p_a}{p_{a0}}\left(24 + 4.04\times10^4 h_w\frac{0.02 + h_w}{0.391 + h_w}\right),
$$

$$
f_{rN} = \frac{p_a}{p_{a0}}\left(\frac{T}{T_0}\right)^{-1/2}\left(9 + 280h_w\,e^{-4.170[(T/T_0)^{-1/3} - 1]}\right),
$$

$$
\alpha = 8.686f^2\left[1.84\times10^{-11}\frac{p_{a0}}{p_a}\left(\frac{T}{T_0}\right)^{1/2} + \left(\frac{T}{T_0}\right)^{-5/2}\left(\frac{0.01275\,e^{-2239.1/T}}{f_{rO} + f^2/f_{rO}} + \frac{0.1068\,e^{-3352.0/T}}{f_{rN} + f^2/f_{rN}}\right)\right]\ \mathrm{dB/m},
$$

with $h_w$ the molar concentration of water vapor (%). It is evaluated at single frequencies (a
bin or a band center), with no integration over a band. The speed of sound is
$c = 343.2\sqrt{T/T_0}$ m/s.

**The run's atmosphere** is uniform: the mean over the site's ground weather stations of the
samples nearest the run's start time (`noise_abatement_2017.run_atmosphere`). A run with no usable
station record is depropagated in 293.15 K, 101.325 kPa and 20% RH, which is also
`depropagate_hemisphere`'s default. A stratified profile (balloon temperature, LIDAR wind) enters
only through the ray model of §5.3. The air density, recorded as run metadata, is that of moist air
carried up hydrostatically at the measured temperature (`air_density_kg_m3`),

$$
\rho(z) = \frac{p - 0.378p_v}{R_d T}\,\exp\!\left(-\frac{g z}{R_d T_v}\right), \qquad
p = 1000\,p_a, \qquad p_v = \frac{H}{100}\,p_\mathrm{sat}(T), \qquad T_v = \frac{T}{1 - 0.378p_v/p},
$$

with $p$ the station pressure and $p_v$ the vapor pressure, both in Pa, $R_d = 287.058$ J/(kg·K),
$p_\mathrm{sat}$ the ISO 9613-1 saturation pressure, and $z$ the aircraft's mean height above the
array. The hydrostatic factor multiplies the whole density, vapor term included.

## 6. Gridding onto the sphere

### 6.1 Inverse-distance weighting in energy

The scattered depropagated samples are gridded onto a regular grid of azimuth and elevation (10°
by default, `azi_step`, `elv_step`). Every quantity is interpolated **in power**, then converted
to dB. Distances are great-circle arcs, with elevation as latitude:

$$
\cos d_g = \sin e_1\sin e_2 + \cos e_1\cos e_2\cos(\psi_1 - \psi_2).
$$

By default (`interpolation=None`) the weights are Franke and Nielson's modified Shepard weights
[34, 35] within a fixed radius $R_n$ (`rmax`, default 25°; `shepIDW_weights`):

$$
w_i = \left(\frac{R_n - d_{g,i}}{R_n\,d_{g,i}}\right)^2\ (d_{g,i} \le R_n), \qquad
P(\text{node}) = \frac{\sum_i w_i P_i}{\sum_i w_i}.
$$

A node on a sample takes that sample's value. A node with no sample in reach is empty (NaN), not
zero, and a missing (NaN) sample is left out of its node's sums. The 2017 release's grid and
interpolation settings are in [`database_build.md`](database_build.md#the-2017-release-settings).

### 6.2 The adaptive radius

With `interpolation` given (`{}` for the defaults in `ADAPTIVE_INTERPOLATION`),
`adaptive_idw_weights` sets each node's radius from the data around it:

$$
R_n = \max\left(\kappa\, d_k,\ \kappa\, d_M,\ \tfrac12\tilde{s}\right),
$$

with $\kappa = 1.3$, $d_k$ the distance to the 8th nearest sample (the `k` setting, default 8), $d_M$ the distance to the 3rd
nearest distinct microphone, and $\tilde{s}$ the median angular resolution of the 8 nearest
samples. A sample's resolution is the arc its analysis window smears over, converted to degrees:

$$
s = \frac{\sqrt{\left(\lvert\mathbf{v}_\perp\rvert T_w\right)^2 + \ell_s^2}}{r},
$$

with $T_w$ the requested `window_time` (0.5 s by default), not the longer power-of-two frame,
$\mathbf{v}_\perp$ the source velocity across the line of sight and $\ell_s$ the source's own
size (`source_extent`, default 0; for example the rotor diameter). A node whose radius exceeds 60°
is retried without the $d_M$ term, i.e. with its 8 nearest samples (the `k` setting) from any
number of microphones; it is a gap only if that radius also exceeds 60°. The nodes filled this way are seen
by only one or two microphones, and their count is reported as `relaxed` in the result's
`interpolation` record. Three options shape the kernel:

- **Aspect** $\Lambda$ (`aspect`, default 1) stretches the distance in azimuth,
  $d_{g,\Lambda} = \sqrt{\Delta e^2 + (d_g^2 - \Delta e^2)/\Lambda^2}$ from the geodesic distance
  $d_g$ and the elevation difference $\Delta e$, so that samples along a flight pass (which spread
  in azimuth) share more than samples across it.
- **Floor** $c_f$ (`shepard_floor`, default 0) softens the Shepard singularity,
  $d_g \to \sqrt{d_g^2 + (c_f R_n)^2}$ in the denominator, so that one sample does not dominate a
  node it nearly hits.
- **Kernel** (`kernel`, default `'shepard'`): `'biweight'` replaces the Shepard weight with
  $w_i = \left(1 - (d_{g,i}/R_n)^2\right)^2$, finite everywhere, which averages its neighborhood
  instead of nearly interpolating.

The 2017 release's kernel, aspect and floor are in
[`database_build.md`](database_build.md#the-2017-release-settings).

### 6.3 Output grids and coverage

The gridded sphere is resampled onto the AAM grid ($\phi$ every 10°, $\theta$ every 5°, or a
reference sphere's own grid) bilinearly in power, renormalized by the interpolated weight of
measured cells (`write_aam_hemisphere_netcdf`, through `_sample_hemisphere_levels`). Cells with
zero power count as measured; NaN cells do not.

**Missing values.** In the gridded hemisphere, $-\infty$ is measured with no energy (every
sample in reach gated) and NaN is unmeasured (no sample in reach, or only capped ones). How
these states are written to the sphere file (−999) and to the database (`coverage` 0) is in
[`file_formats.md`](file_formats.md#how-a-bins-state-travels).

For plots, spheres are drawn in the azimuthal equal-area projection centered on the nadir
[36] (`lambert_ea`, `lambert_lon`): $x = q\sin\lambda$, $y = q\cos\lambda$, with
$q = 2\sin[(90^\circ - e)/2]$ and $\lambda = 180^\circ - \psi$, a view from above with the nose up
and starboard to the right.

## 7. The database

`build_empirical_database` collects a directory of sphere files and its `vehicle.cfg`
([`file_formats.md`](file_formats.md)) into a NICE-OPS database.

### 7.1 Flight condition

Each sphere is labeled by advance ratio, flight path angle and thrust coefficient:

$$
\mu = \frac{0.514444\,V}{V_\mathrm{tip}}, \qquad \gamma = \bar{\gamma}_\mathrm{track}, \qquad
C_T = n_z\,C_W, \qquad C_W = \frac{W}{\rho\,\pi R_\mathrm{mr}^2 V_\mathrm{tip}^2},
$$

with $V$ the segment's mean **ground** speed in knots, $\gamma$ its mean flight path angle, and
$W$ ($g$ times the vehicle file's mass), $\rho$, $R_\mathrm{mr}$ and $V_\mathrm{tip}$ the vehicle
file's nominal values. The run's own thrust coefficient, from its weight and air density, and its
air-referenced advance ratio, when a wind was given, are recorded as metadata but do not label the
sphere.

### 7.2 Excess atmospheric attenuation

A NICE-OPS broadband run carries the A-weighted level to the ground with one attenuation
coefficient per direction, the EAA. PANAM computes it from the sphere's band levels $L_i$
(`spla_and_eaa_from_spectrum`):

$$
\mathrm{SPLA} = 10\lg\sum_i 10^{(L_i + W_A(f_i))/10}, \qquad
\mathrm{EAA} = \mathrm{SPLA} - 10\lg\sum_i 10^{(L_i + W_A(f_i) - \alpha(f_i)\,d_E)/10},
$$

with $d_E$ the `distance` argument (default 1000 m) and $\alpha$ in the build atmosphere
(`atmosphere`, default 293.15 K, 101.325 kPa and 20% RH), recorded in the file. NICE-OPS consumes
it as $\mathrm{EAA}\,(r - R_s)/1000$ m, so $d_E$ must stay 1000 m. Where the gate emptied a
direction, SPLA is $-\infty$ and EAA would be NaN; with `clamp_empty_directions=True` (the default)
EAA is set to 0 there and dBA stays $-\infty$ (`_finite_sphere_levels`).

### 7.3 Load factor

With `load_factors=None`, the database stores only the 1 g spheres with `fixed_load_factor` set,
$C_{T,\mathrm{ref}} = C_W$, and the reader applies $20\lg(C_T/C_{T,\mathrm{ref}})$ above 1 g.
Otherwise each condition is stored at a ladder of load factors $n_z$ (default five, evenly from
0.7 to 2.3), with $\mathrm{dBA} \leftarrow \mathrm{dBA} + 20\lg n_z$ for $n_z > 0$; the band
spectrum is not scaled. A load factor of 0 is stored unscaled, as a floor entry at $C_T = 0$. The
law is the same in both forms, but a ladder holds it exactly only at its rungs and NICE-OPS is
linear in $C_T$ between them (−0.3 dB at 4 g between rungs at 3 g and 5 g). A ladder whose rungs
below 1 g are only 0 and 1 leaves a level below 1 g at its 1 g value, as the fixed form does; a
ladder with rungs between 0 and 1 g scales them. The 2017 release's ladders are in
[`database_build.md`](database_build.md#the-2017-release-settings).

### 7.4 Hover

A hover passes no microphones, so the arrays see it only along a few fixed rings of elevation, and
the rest of its sphere has to be synthesized. The source is the slowest sphere within 2° of level
flight (`level_flight_tolerance`), or the slowest overall, with a warning, if none is; or the sphere
passed as `hover_source`, typically that run rebuilt with `build_sphere(remove_doppler=True)`
(§3.3). Its spectrum is averaged fore and aft in energy so that it has no direction of flight
(`average_fore_and_aft`):

$$
L_i = L_{N_\theta-1-i} = 10\lg\left[\tfrac12\left(10^{L_i/10} + 10^{L_{N_\theta-1-i}/10}\right)\right]
$$

over the $N_\theta$ polar rows. A cell's coverage is 1 only where both partners were measured
(`_fore_aft_measured`). PANAM then adds an optional correction in dB as a function of direction and
band (`hover_correction`), and recomputes SPLA and EAA (§7.2) from the averaged and corrected
spectrum. The hover is written at speed 0 and every load factor, at 0° and at each extended path
angle (§7.5), or at −12°, 0° and +12° when no extended angles are given. The 2017 release's hover
source and correction are in [`database_build.md`](database_build.md#the-2017-release-settings).

### 7.5 Padding and completion

With `extended_flight_path_angles` (off by default), every non-hover flight condition within 2° of
level is copied to those angles at its own speed and every load factor, so that the condition hull
covers steep states; the 2017 release's angles are in [`database_build.md`](database_build.md#the-2017-release-settings). The upper hemisphere, which no ground microphone sees, is completed by
mirroring through the sphere's horizontal plane (the level frame of §3.2), $\phi' = 180^\circ - \phi$
wrapped into $[-180^\circ, 180^\circ)$ (`mirror_phi_to_upper_surface`), and carries coverage 0.
Database format version 1 marks files with the corrected mirror (an earlier `-phi` mirror reflected
the lower hemisphere onto itself) and the energy-averaged hover with recomputed EAA.

## 8. NORAH2 and AAM interchange

**NORAH2 import** (`build_database_from_norah2`, `norah2_to_nod.py`). A NORAH2 hemisphere (`.hem`)
[37, 38, 39] gives one-third-octave levels on a 10° grid at the distance POLDIST
(60 m in the shipped forward-flight files), free field with absorption to that distance in the
file's own atmosphere (TAMB, RELHUM, PAMB). PANAM refers it to the NICE-OPS sphere radius $r$
(`radius_m`, `--radius`, default 30.48 m = 100 ft; `norah2_levels_at_radius`):

$$
L(r) = L(\mathrm{POLDIST}) + 20\lg\frac{\mathrm{POLDIST}}{r} + \alpha_\mathrm{file}(f_c)\,(\mathrm{POLDIST} - r),
$$

with POLDIST read from each file and $\alpha_\mathrm{file}$ the ISO 9613-1 coefficient at the
band centers in the file's atmosphere. For $r > \mathrm{POLDIST}$ the same expression adds the
absorption between the two. **Departure:** NORAH2 itself applies SAE ARP 5534's band correction
[40], which is not reproduced. The angle conventions agree with NICE-OPS's (`norah2umapr`),
so no rotation is needed.

NORAH2's speed field (ACSPEED) is labeled indicated airspeed but used as ground speed, so the
importer requires the user to say which (`norah2_speed_knots`). `ground_speed` keeps it as the
ground speed. `ias_to_tas` takes it as equivalent airspeed (no instrument, position or
compressibility error) and converts it,

$$
V_\mathrm{TAS} = V_\mathrm{IAS}\sqrt{\frac{1.225}{\rho}}, \qquad
\rho = \frac{p - p_v}{R_d T} + \frac{p_v}{R_v T}, \qquad p_v = \frac{H}{100}\,p_\mathrm{sat}(T),
$$

with $R_d = 287.058$ and $R_v = 461.495$ J/(kg·K), $p_\mathrm{sat}$ the ISO 9613-1 saturation
pressure, and $T$, $p$, $H$ the file's measurement atmosphere (Tm in °C, Pm in Pa, RHm in %;
`humid_air_density`), or one density for every file (`tas_air_density`, `--tas-density`). A file
without Tm, Pm and RHm, or with values out of range, is refused unless a density is given, and
$\rho$ must lie within 0.3–1.6 kg/m³.

The conversion makes these further choices; [`norah2_import.md`](norah2_import.md) gives the
reasons:

- A file whose levels include the ground reflection (FREEFIELD = 0) is refused unless accepted
  explicitly (`accept_ground_included`), and is then converted as free field.
- `mirror_rotor` reverses the azimuth, $\phi \to -\phi$, before the upper hemisphere is completed.
  It is off by default.
- NOVALUE cells are missing ($-\infty$, coverage 0). `fill_empty` (off by default) gives each band
  of such a cell the level of the nearest direction that has one (largest dot product of the
  body-axis unit vectors; `fill_norah2_empty_cells`); coverage is taken before the fill, so filled
  cells keep coverage 0.
- Each file is one condition, written once at load factor 1 with `fixed_load_factor` set (§7.3),
  with $C_W$ formed at `thrust_air_density` (default 1.225 kg/m³).
- SPLA and EAA follow §7.2, over 1000 m in the build atmosphere (default 293.15 K, 101.325 kPa,
  20% RH).
- Every group carries `DOPPLER_SHIFT_REMOVED = 0`, an inference: NORAH2 describes no step that
  removes the shift. The root carries `azimuth_reference = 'track'`.
- The upper hemisphere is completed as in §7.5, with coverage 0.

**Export.** Spheres can be written in the AAM netCDF format [11], in the variable order AAM 3.1
reads by position, and as NORAH2 hemispheres (`write_norah2_hemisphere`). The NORAH2 export
resamples the third-octave hemisphere in linear power onto the NORAH2 10° grid and its 31 nominal
bands from 10 Hz to 10 kHz, as in §6.3. It moves the levels to 60 m, takes out the absorption the
sphere keeps over $r_\mathrm{ref}$ in the measurement atmosphere, and puts in absorption over the
whole 60 m in the ICAO reference atmosphere (298.15 K, 70% RH, 101.325 kPa):

$$
L(60\ \mathrm{m}) = L(r_\mathrm{ref}) - 20\lg\frac{60\ \mathrm{m}}{r_\mathrm{ref}}
+ \alpha_\mathrm{meas}(f)\,r_\mathrm{ref} - \alpha_\mathrm{ICAO}(f_c)\cdot 60\ \mathrm{m},
$$

with $f$ the hemisphere's own band center and $f_c$ the nominal one. A band the hemisphere lacks,
a direction it did not cover, and a level below −100 dB at 60 m (`minimum_level_db`) are written
as −999. This is not the inverse of the import: a sphere exported and imported again comes back
shifted by $(\alpha_\mathrm{meas} - \alpha_\mathrm{ICAO})\,r_\mathrm{ref}$. A sphere built without
absorption depropagation is written with spreading only, with a warning.

`write_norah2_triangulation` writes the `.int` file through which NORAH2 interpolates between an
aircraft's hemispheres: the Delaunay triangulation of the raw (knots, degrees) conditions, and by
default the correction table every shipped file carries (+8 dB out-of-ground hover, −10 dB reduced
rpm idle, −2 dB full rpm idle).

## 9. Metrics

**Sound exposure level** (`sound_exposure_level`). Over the 10 dB-down window (the first to the
last sample within 10 dB of the maximum, dips included; unlike the EPNL limits, a sample below the
threshold is never one), $\mathrm{SEL} = 10\lg\left(\sum_k 10^{L_k/10}\Delta t/1\ \mathrm{s}\right)$
[41]. A $-\infty$ sample is zero energy; a NaN sample is refused unless `missing='omit'`. The
result is flagged `clipped` when the window touches either end of the record. Unlike NICE-OPS,
PANAM's samples are already in reception time.

**Maximum levels.** PANAM has no exponentially time-weighted level (no $L_{AF\max}$ or
$L_{AS\max}$). `level_history` gives levels of consecutive, non-overlapping blocks (`period`,
default 1 s), each from `overall_SPL`, and `sound_exposure_level`'s `lmax` is the maximum of the
samples passed in. Neither is directly comparable with NICE-OPS's $L_{AF\max}$.

**EPNL** (`effective_perceived_noise_level`) follows 14 CFR 36, Appendix A, §A36.4 [41],
identical to ICAO Annex 16 [42]. Its input is a history of 24 one-third-octave band levels,
50 Hz to 10 kHz in ascending order (`PNL_BAND_FREQUENCIES`), so a 31-band sphere spectrum has to be
cut to those bands first. It uses noy values $N$ from Table A36-3;
$\mathrm{PNL} = 40 + \frac{10}{\lg 2}\lg(0.85N_\max + 0.15\sum N)$; the ten-step tone correction
with Table A36-2; the band-sharing check of §A36.4.4.2; and the duration correction
$D_E = 10\lg\sum_{k_1}^{k_2} 10^{\mathrm{PNLT}(k)/10} + 10\lg(\Delta t/t_\mathrm{ref}) - \mathrm{PNLTM}$,
$t_\mathrm{ref} = 10$ s, taking the regulation's −13 dB for $10\lg(0.5/10) = -13.0103$ dB when
$\Delta t = 0.5$ s (§A36.4.5.4; `normalization='exact'` keeps −13.0103). The limits $k_1, k_2$ are
the PNLT samples closest to PNLTM − 10 (§A36.4.5.5): the first and last samples at or above it,
each moved one sample outward when that sample is closer, and every sample between them is summed,
dips included, which gives the longest duration when there are several peaks. The result is
flagged `clipped` when $k_1$ or $k_2$ is a record end. The defaults are $\Delta t = 0.5$ s,
`bandshare_adjustment=True`, `masked=False` and `normalization='regulatory'`. Where readings of
the regulation differ, PANAM puts 500 Hz in the middle range of Table A36-2. NICE-OPS's EPNL is
ported from this implementation.

**Band sharing.** When the tone correction $C$ at PNLTM is below the mean $\bar C$ of the five
records centered there (§A36.4.4.2), or of those that exist when PNLTM is within two samples of a
record end (this truncation is not flagged), the adjustment $\Delta_B = \bar C - C(k_M)$ is added to
the EPNL as a separate term, as ICAO Annex 16 Vol. I, Appendix 2 and FAA AC 36-4 apply it:
$\mathrm{EPNL} = \mathrm{PNLTM} + D_E + \Delta_B$, with PNLTM, the 10 dB-down limits and $D_E$ all
from the unadjusted PNLT history. Read literally, 14 CFR 36 raises PNLTM and takes $D_E$ from it,
so the adjustment cancels in $\mathrm{PNLTM} + D_E$ and acts only by narrowing the duration, which
lowers the EPNL. `effective_perceived_noise_level` returns `pnltm` (adjusted),
`pnltm_unadjusted` and `delta_b`.

The regulation's tone correction is defined for finite band levels. A band of $-\infty$ (zero
energy) would make the step-7 background infinite or undefined, so it enters steps 1–7 at its
noy threshold SPL(d) of Table A36-3 and carries no tone itself; PNL is unchanged, since the band
has no noys either way. With `masked=True` every band below SPL(d) is treated so, as in
NICE-OPS's `below_noy_floor` masking. A NaN band is a missing level: noys, PNL, $C$ and PNLT are
NaN, and EPNL refuses the history.

**Aural nondetectability** (`mil_std_1474e_nondetectability_distance`). MIL-STD-1474E Table C-I
[43] gives, per one-third-octave band, the limit at each nondetectability distance $x_j$
(5 m to 6 km), measured at a distance $x^\mathrm{m}_j$; a distance is met when no band exceeds its
limit (§C.5.1.2). PANAM ships the table as `mil_std_1474e_table_c1_full.csv`
(`load_mil_std_1474e_table_c1`): 24 bands from 50 Hz to 10 kHz, 20 distance columns, measured at
2 m for the 5–30 m columns, 10 m for 100–400 m and 30 m for 500–6000 m, with NA where no limit is
listed. The input spectrum, given at `spectrum_distance_m` $d$ (default 10 m), is first
interpolated linearly in $\lg f$ onto the table's band centers; a table band up to half a
one-third octave past either end of the input takes that end's level, and bands further out are
not evaluated (`outside_spectrum`). It is then normalized to 10 m, $L_{10} = L + 20\lg(d/10)$, and
taken to each column's measurement distance by spherical spreading, giving the exceedance
$\Delta L_j = L_{10} + 20\lg(10/x^\mathrm{m}_j) - L_{\lim,j}$. A band's distance is where
$\Delta L$, interpolated linearly in $\lg x$ between adjacent columns, falls to zero for the last
time. Within a measurement-distance group this is the reading of Figures C-1 to C-5; across the
30–100 m and 400–500 m group boundaries it rests on the same spreading that shifts the spectrum.
The nondetectability distance is the largest over the bands, every band counting. A band under
every column gives 5 m, an upper bound; one over the 6 km column gives 6 km, a lower bound that
wins and flags the result; an NA entry counts as met, so a band over its last listed column is
bracketed by that column and the next.

## 10. Auxiliary models

These are in PANAM but not used to build the databases.

- **Vold–Kalman order tracking** (`vold_kalman_filter.py`) [44, 45]. Each order $j$ is a
  slowly varying complex envelope $a_j$ on a known phasor
  $\Theta_j(n) = \exp\left(2\pi i\,\Delta t\sum_{m \le n} f_j(m)\right)$, $n$ the sample
  index. The data equation $x = \sum_j a_j\Theta_j + \nu$ and a structural equation of order $p$,
  $\sum_{s=0}^{p}(-1)^s\binom{p}{s}a_j(n+s) = \varepsilon_j(n)$, are solved together by least
  squares, minimizing $\lVert x - \sum_j a_j\Theta_j\rVert^2 + \sum_j\lVert w\,\varepsilon_j\rVert^2$.
  A component $\Omega$ rad/sample from the order passes with gain
  $1/\left[1 + w^2\left(2\sin(\Omega/2)\right)^{2p}\right]$ [45]. Setting that to $1/\sqrt2$ at
  half the full −3 dB bandwidth $B_w$ (Hz) gives the weight, unless `r` is passed:

  $$
  w = \sqrt{\frac{\sqrt2 - 1}{\left[2\sin\left(\pi B_w/(2f_s)\right)\right]^{2p}}} .
  $$

  The function returns $y_j = 2a_j$, so $\lvert y_j\rvert$ is the order's amplitude and its
  waveform is $\mathrm{Re}(y_j\Theta_j)$. `use_coupling=False` drops the cross-order terms.
- **Ground-board models** in `ground_plane.py`, from the plate study: Fresnel-zone weighting of a
  finite plate (strip [46] and disc [47]), blended in energy [48] or in dB;
  an impedance-discontinuity model after De Jong and Lam and Monazzam [49, 50]; a thin-disc
  BEM [27] (`disc_bem_scattered`); and a 3-D surface-mesh BEM of the raised, tapered plate
  (`raised_plate_scattering`), the same integral equation and Green's function as §4.5, which the
  axisymmetric BEM replaced at a fraction of the cost and is checked against. See
  [`notes/ground_plane_corrections.md`](notes/ground_plane_corrections.md).
- **Two-path pole fit** (`fit_two_path`, `two_path_db`, `height_from_path_difference`). The
  pole-minus-plate narrowband level difference is fitted with
  $C + 10\lg\lvert 1 + A e^{i(2\pi f\Delta R/c + \varphi_Q)}\rvert^2$, $A = A_0e^{-f/f_A}$, for
  the path difference $\Delta R = R_2 - R_1$; with the pole height, $\Delta R$ gives the arrival
  elevation or the source height. It checks the flight geometry and the pole height independently
  of any level model (see the ground-plane notes).
- **Array planning** (`linear_array_plan`, `array_planner.py`): microphones along a line,
  $y_i = (h/\sin e_t)\tan\theta_i$, with $\theta_i$ equally spaced in
  $[-(90^\circ - e_\min),\ 90^\circ - e_\min]$ (defaults $e_\min = 10^\circ$, $e_t = 90^\circ$),
  so that the outermost microphone is seen from overhead at $\arctan(\sin e_t\tan e_\min)$ below
  the horizon. `array_coverage`, `hover_array_coverage` and `takeoff_array_coverage` give the
  sphere directions a planned level pass, hover or takeoff covers (§3.1–3.2).
- **Footprint preview** (`project_sphere`, `plot_projection`): each direction at least $e_c$ below
  the horizon (`elv_cutoff`, no default; `project_directory` and `plot_projection` use 30°)
  follows a straight slant ray from the sphere radius $r_\mathrm{ref}$ to flat ground $h_a$ below
  (meters), with no ground effect:
  $R = h_a/\sin e$, $R_g = \sqrt{R^2 - h_a^2}$, $x = R_g\sin\psi$ (starboard), $y = -R_g\cos\psi$
  (ahead), $L_A = \mathrm{SPLA} + 20\lg(r_\mathrm{ref}/R) - \mathrm{EAA}\,(R - r_\mathrm{ref})/1000\ \mathrm{m}$,
  with EAA from §7.2 in the given atmosphere (default 293.15 K, 101.325 kPa, 20% RH).
- **Fried-egg plot** (`fried_egg_plot`, `project_directory`). Each sphere in a directory, except
  climbs steeper than `fpa_climb_cutoff` (default 5°), is reduced to the maximum or the mean of its
  footprint preview level, optionally duration-corrected by $10\lg(V_\mathrm{ref}/V)$, and contoured
  over speed and flight path angle. A run is classed noisy when
  $L > L_\min + 0.65\,(L_\max - L_\min)$ over the set (`is_noisy`). The nondimensional axes are
  $\mu$ and a quasi-static tip-path-plane angle of attack,
  $\alpha_\mathrm{TPP} = -\deg\!\left(\tfrac12\bar f\mu^2/C_W\right) - \gamma$, with
  $\bar f = f_e/(\pi R_\mathrm{mr}^2)$ the flat-plate drag area $f_e$ over the rotor disc area,
  from `vehicle.cfg`; with an `[Option]` section, $\bar f$ is its `fbar` and $\mu$ and $C_W$ come from
  its reference list. This angle is a plot coordinate only;
  the spheres themselves are not tilted (§11).
- **Time-domain de-Dopplerization** (`dedopplerize`): each microphone's pressure is resampled at
  $t_e + r(t_e)/c$ on an emission-time grid, and, with a virtual observer radius $R_v$, scaled by
  $r(t_e)/R_v$ and labeled with time $t_e + R_v/c$; samples received outside the record are NaN.

## 11. What is not modeled

- **Tip-path-plane orientation.** Spheres are filed in a level frame (§3.2).
- **Convective amplification.** Doppler removal rescales frequency only (§3.3).
- **Wind on straight rays.** Depropagation along straight lines ignores wind; only a ray model
  (§5.3) sees it. A wind given to `build_sphere` is recorded as metadata only.
- **A stratified atmosphere without rays.** Absorption and timing use one uniform atmosphere; only
  the optional ray model sees the profile (§5.3).
- **Band-integrated absorption.** ISO 9613-1 is evaluated at single frequencies.
- **Ground roughness in the sphere build.** Only `pole_level` has a roughness term; the plate BEM
  has none (§4.6).

## Appendix: Where each model lives

Functions are in `flight_acoustics.py` unless another module is named. Tests are in `tests/`.

| Section | Functions | Tests |
| --- | --- | --- |
| §2.1 Short-time spectra | `spectrogram`, `frame_length`, `psd`, `psd_welch`, `overall_SPL`, `level_history` | `test_acoustics.py`, `test_generate_hemisphere.py` |
| §2.2 Bands | `third_octave_band_edges`, `third_octave_band_levels`, `tone_aware_band_power`, `hann_power`, `panam_acoustics.filters.third_octave_filter_bank` | `test_third_octave_band_edges.py`, `test_third_octave_band_levels.py`, `test_band_selection.py`, `test_tone_aware_band_power.py`, `test_third_octave_filter_bank.py` |
| §2.3 A-weighting | `dBAw` | `test_acoustics.py` |
| §3.1–3.2 Geometry | `hemigen`, `art2umapr`; `noise_abatement_2017.local_east_north_ft`, `east_north_to_frame`, `frame_bearing_deg` | `test_lambert_geometry.py`, `test_hemisphere_geometry_accuracy.py`, `test_azimuth_reference.py`, `test_track_check.py` |
| §3.3 Doppler | `depropagate_hemisphere` (`remove_doppler`) | `test_generate_hemisphere.py`, `test_tone_aware_band_power.py` |
| §3.4 Steady segments | `noise_abatement_2017.steady_window`, `flight_condition` | `test_noise_abatement_2017.py` |
| §4.1 Impedance | `ground_plane.surface_admittance` | `test_ground_validation.py` |
| §4.2 Reflection coefficient | `spherical_reflection_coefficient` | `test_reflection_coefficient.py` |
| §4.3 `ega()` | `ega` | `test_ega.py` |
| §4.4 Pole microphones | `ground_plane.pole_level`, `turbulence_coherence`, `free_field_microphone_response`, `pole_incidence`, `certification_axis` | `test_ground_plane.py`, `test_ground_validation.py` |
| §4.5 Plate BEM | `axisymmetric_bem.scattering`, `table`, `board_level`, `write_netcdf`; `ground_plane.exact_half_space_green`, `image_integrals`, `ImageIntegralTable`, `sub_band_factors`, `table_frames`; `noise_abatement_2017.plate_table`, `plate_response` | `test_axisymmetric_bem.py`, `test_ground_validation.py` |
| §4.6 Site ground | `noise_abatement_2017.SITE_GROUND` | (none) |
| §5.1–5.2 Depropagation | `depropagate_hemisphere`; `noise_abatement_2017.build_sphere` | `test_ambient_gating.py`, `test_generate_hemisphere.py`, `test_golden_results.py` |
| §5.3 Rays | `refracted_rays.external_ray_model`, `straight_ray_model` | `test_refracted_rays.py` |
| §5.4 Atmosphere | `panam_acoustics.iso_9613_1_1993`, `panam_acoustics.atmosphere.Atmosphere`, `atmosorb`; `noise_abatement_2017.run_atmosphere`, `air_density_kg_m3` | `test_acoustics.py`, `test_python_acoustics_equivalence.py`, `test_run_metadata.py`, `test_noise_abatement_2017.py` |
| §6.1 Fixed radius | `shepIDW`, `shepIDW_weights`, `shepIDW_apply`, `geodist` | `test_shep_idw.py`, `test_no_data.py`, `test_hemisphere_geometry_accuracy.py` |
| §6.2 Adaptive radius | `adaptive_idw_weights`, `ADAPTIVE_INTERPOLATION` | `test_adaptive_interpolation.py`, `test_hemisphere_geometry_accuracy.py` |
| §6.3 Output grids | `write_aam_hemisphere_netcdf`, `_sample_hemisphere_levels`, `mask_missing_levels`, `lambert_ea`, `lambert_lon` | `test_write_aam_hemisphere_netcdf.py`, `test_hemisphere_sampling.py`, `test_no_data.py`, `test_lambert_geometry.py` |
| §7.1–7.2 Condition, EAA | `build_empirical_database`, `read_vehicle_data`, `add_sphere_group`, `extract_SPL`, `spla_and_eaa_from_spectrum`, `_finite_sphere_levels`, `_measured_bins` | `test_spla_and_eaa.py`, `test_database_provenance.py`, `test_vehicle_data.py` |
| §7.3 Load factor | `add_sphere_group` | `test_load_factor_scaling.py`, `test_write_aam_hemisphere_netcdf.py` |
| §7.4 Hover | `average_fore_and_aft`, `_fore_aft_measured` | `test_hover_averaging.py`, `test_sphere_completion_integration.py`, `test_empirical_database_hover_selection.py`, `test_database_provenance.py` |
| §7.5 Completion | `mirror_phi_to_upper_surface`, `_complete_sphere_with_coverage` | `test_mirror_phi.py`, `test_sphere_completion_integration.py` |
| §8 NORAH2 | `build_database_from_norah2`, `norah2_levels_at_radius`, `norah2_speed_knots`, `humid_air_density`, `fill_norah2_empty_cells`, `write_norah2_hemisphere`, `write_norah2_triangulation`, `norah2umapr` | `test_norah2_import.py`, `test_write_norah2_hemisphere.py` |
| §9 Metrics | `sound_exposure_level`, `ten_db_down_interval`, `noys`, `perceived_noise_level`, `tone_correction`, `effective_perceived_noise_level`, `mil_std_1474e_nondetectability_distance` | `test_sound_exposure_level.py`, `test_pnl_epnl.py`, `test_mil_std_nondetectability.py` |
| §10 VKF | `vold_kalman_filter.vold_kalman_filter` | `test_vold_kalman_filter.py` |
| §10 Ground-board models, two-path fit | `ground_plane.board_*`, `disc_bem_scattered`, `raised_plate_scattering`, `fit_two_path`, `height_from_path_difference` | `test_ground_plane.py`, `test_ground_validation.py`, `test_axisymmetric_bem.py` |
| §10 Array planning | `linear_array_plan` (through the `array_planner.py design` CLI); `array_coverage`, `hover_array_coverage`, `takeoff_array_coverage` | `test_array_planner.py` (the CLI's spacing and options); the coverage functions: none |
| §10 Footprint, fried egg | `project_sphere`, `project_directory`, `fried_egg_plot`, `is_noisy` | `test_projection.py`, `test_no_data.py`, `test_cli_scripts.py` (smoke) |
| §10 De-Dopplerization | `dedopplerize` | `test_acoustics.py` |

## References

1. Greenwood, E., "Fundamental Rotorcraft Acoustic Modeling from Experiments (FRAME)," Ph.D. Dissertation, Dept. of Aerospace Engineering, University of Maryland, College Park, MD, 2011. http://hdl.handle.net/1903/11518
2. Schmitz, F. H., Greenwood, E., Sickenberger, R. D., Gopalan, G., Sim, B. W.-C., Conner, D., Moralez, E., III, and Decker, W. A., "Measurement and Characterization of Helicopter Noise in Steady-State and Maneuvering Flight," *Proceedings of the American Helicopter Society 63rd Annual Forum*, Virginia Beach, VA, May 2007. https://doi.org/10.4050/VFS-F63-000258
3. Greenwood, E., Schmitz, F. H., and Sickenberger, R. D., "A Semiempirical Noise Modeling Method for Helicopter Maneuvering Flight Operations," *Journal of the American Helicopter Society*, Vol. 60, No. 2, 2015, Paper 022007. https://doi.org/10.4050/JAHS.60.022007
4. Watts, M. E., Greenwood, E., Smith, C. D., and Stephenson, J. H., "Noise Abatement Flight Test Data Report," NASA TM-2019-220264, NASA Langley Research Center, Hampton, VA, March 2019.
5. Pascioni, K. A., Greenwood, E., Watts, M. E., Smith, C. D., and Stephenson, J. H., "Medium-Sized Helicopter Noise Abatement Flight Test," *Proceedings of the Vertical Flight Society 76th Annual Forum*, Virtual, October 2020. https://doi.org/10.4050/F-0076-2020-16497
6. Harris, F. J., "On the Use of Windows for Harmonic Analysis with the Discrete Fourier Transform," *Proceedings of the IEEE*, Vol. 66, No. 1, 1978, pp. 51–83. https://doi.org/10.1109/PROC.1978.10837
7. Welch, P. D., "The Use of Fast Fourier Transform for the Estimation of Power Spectra: A Method Based on Time Averaging over Short, Modified Periodograms," *IEEE Transactions on Audio and Electroacoustics*, Vol. AU-15, No. 2, 1967, pp. 70–73. https://doi.org/10.1109/TAU.1967.1161901
8. "Electroacoustics — Octave-Band and Fractional-Octave-Band Filters — Part 1: Specifications," IEC 61260-1:2014, International Electrotechnical Commission, Geneva, 2014.
9. Grandke, T., "Interpolation Algorithms for Discrete Fourier Transforms of Weighted Signals," *IEEE Transactions on Instrumentation and Measurement*, Vol. IM-32, No. 2, 1983, pp. 350–355. https://doi.org/10.1109/TIM.1983.4315077
10. "Electroacoustics — Sound Level Meters — Part 1: Specifications," IEC 61672-1:2013, Ed. 2.0, International Electrotechnical Commission, Geneva, 2013.
11. Page, J. A., Rapoza, A., Oberg, A., Hastings, A., Baker, G., and Shumway, M., "Advanced Acoustic Model (AAM) Technical Reference and User's Guide," Software Manual, Version 3.1, U.S. Department of Transportation, Volpe National Transportation Systems Center, Cambridge, MA, May 2021.
12. "Department of Defense World Geodetic System 1984: Its Definition and Relationships with Local Geodetic Systems," NIMA TR8350.2, 3rd ed., Amendment 1, National Imagery and Mapping Agency, July 1997 (amended January 2000).
13. Delany, M. E., and Bazley, E. N., "Acoustical Properties of Fibrous Absorbent Materials," *Applied Acoustics*, Vol. 3, No. 2, 1970, pp. 105–116. https://doi.org/10.1016/0003-682X(70)90031-9
14. Miki, Y., "Acoustical Properties of Porous Materials — Modifications of Delany-Bazley Models," *Journal of the Acoustical Society of Japan (E)*, Vol. 11, No. 1, 1990, pp. 19–24. https://doi.org/10.1250/ast.11.19
15. Attenborough, K., and Taherzadeh, S., "Prediction of Outdoor Ground Effect," *Applied Acoustics*, Vol. 242, 2026, Paper 111067. https://doi.org/10.1016/j.apacoust.2025.111067
16. Chien, C. F., and Soroka, W. W., "Sound Propagation Along an Impedance Plane," *Journal of Sound and Vibration*, Vol. 43, No. 1, 1975, pp. 9–20. https://doi.org/10.1016/0022-460X(75)90200-X
17. Chien, C. F., and Soroka, W. W., "A Note on the Calculation of Sound Propagation Along an Impedance Surface," *Journal of Sound and Vibration*, Vol. 69, No. 2, 1980, pp. 340–343. https://doi.org/10.1016/0022-460X(80)90618-5
18. Stusnick, E., Plotkin, K. J., and Sutherland, L. C., "Short-Range Acoustic Propagation Model," Wyle Laboratories Research Report WR 85-19, Wyle Laboratories, Arlington, VA, July 1985.
19. Chessell, C. I., "Propagation of Noise Along a Finite Impedance Boundary," *Journal of the Acoustical Society of America*, Vol. 62, No. 4, 1977, pp. 825–834. https://doi.org/10.1121/1.381603
20. Daigle, G. A., Embleton, T. F. W., and Piercy, J. E., "Some Comments on the Literature of Propagation near Boundaries of Finite Acoustical Impedance," *Journal of the Acoustical Society of America*, Vol. 66, No. 3, 1979, pp. 918–919. https://doi.org/10.1121/1.383207
21. "Free-Field and Random Incidence Corrections," Microsoft Excel workbook GRAS_Free-field_and_Random_Incidence_Corrections.xlsx, sheet "Half-inch (Opt1)," GRAS Sound & Vibration, Holte, Denmark. URL: https://www.grasacoustics.com
22. van Maercke, D., and Defrance, J., "Development of an Analytical Model for Outdoor Sound Propagation Within the Harmonoise Project," *Acta Acustica united with Acustica*, Vol. 93, No. 2, 2007, pp. 201–212.
23. Ostashev, V. E., and Wilson, D. K., *Acoustics in Moving Inhomogeneous Media*, 2nd ed., CRC Press, Boca Raton, FL, 2015. https://doi.org/10.1201/b18922
24. Ament, W. S., "Toward a Theory of Reflection by a Rough Surface," *Proceedings of the IRE*, Vol. 41, No. 1, 1953, pp. 142–146. https://doi.org/10.1109/JRPROC.1953.274171
25. "Ground-Plane Microphone Configuration for Propeller-Driven Light-Aircraft Noise Measurement," SAE ARP4055A, SAE International, Warrendale, PA, 2020.
26. Shivashankara, B. N., and Stubbs, G. W., "Ground Plane Microphone for Measurement of Aircraft Flyover Noise," *Journal of Aircraft*, Vol. 24, No. 11, 1987, pp. 751–758. https://doi.org/10.2514/3.45517
27. Kingan, M. J., Go, S. T., Piscoya, R., and Ochmann, M., "On the Modelling of Ground-Board Mounted Microphones for Outdoor Noise Measurements," *Journal of Sound and Vibration*, Vol. 565, 2023, Paper 117894. https://doi.org/10.1016/j.jsv.2023.117894
28. Go, S. T., Kingan, M. J., Schmid, G., and Hall, A., "On the Use of Ground-Board Mounted Microphones for Outdoor Noise Measurements," *Journal of Sound and Vibration*, Vol. 584, 2024, Paper 118432. https://doi.org/10.1016/j.jsv.2024.118432
29. Seybert, A. F., Soenarko, B., Rizzo, F. J., and Shippy, D. J., "A Special Integral Equation Formulation for Acoustic Radiation and Scattering for Axisymmetric Bodies and Boundary Conditions," *Journal of the Acoustical Society of America*, Vol. 80, No. 4, 1986, pp. 1241–1247.
30. Ochmann, M., "The Complex Equivalent Source Method for Sound Propagation over an Impedance Plane," *Journal of the Acoustical Society of America*, Vol. 116, No. 6, 2004, pp. 3304–3311. https://doi.org/10.1121/1.1819504
31. Taraldsen, G., "The Complex Image Method," *Wave Motion*, Vol. 43, No. 1, 2005, pp. 91–97. https://doi.org/10.1016/j.wavemoti.2005.07.001
32. Morse, P. M., and Ingard, K. U., *Theoretical Acoustics*, McGraw–Hill, New York, 1968, Chap. 8.
33. "Acoustics — Attenuation of Sound during Propagation Outdoors — Part 1: Calculation of the Absorption of Sound by the Atmosphere," ISO 9613-1:1993, International Organization for Standardization, Geneva, 1993.
34. Shepard, D., "A Two-Dimensional Interpolation Function for Irregularly-Spaced Data," *Proceedings of the 1968 23rd ACM National Conference*, ACM, New York, 1968, pp. 517–524. https://doi.org/10.1145/800186.810616
35. Franke, R., and Nielson, G., "Smooth Interpolation of Large Sets of Scattered Data," *International Journal for Numerical Methods in Engineering*, Vol. 15, No. 11, 1980, pp. 1691–1704. https://doi.org/10.1002/nme.1620151110
36. Snyder, J. P., "Map Projections — A Working Manual," U.S. Geological Survey Professional Paper 1395, U.S. Government Printing Office, Washington, DC, 1987. https://doi.org/10.3133/pp1395
37. Olsen, H., Tuinstra, M., and van Oosten, N., "D1.5d Rotorcraft Noise Modelling Guidance," Research Project NOISE, EASA.2020.FC.06, Royal Netherlands Aerospace Centre (NLR), for the European Union Aviation Safety Agency, Cologne, January 2024.
38. van Oosten, N., Ionescu, S. E., Meliveo, L., Konovalova, O., van der Meulen, J. M., and Tuinstra, M., "D2.3 NORAH Hemisphere Database Extension," Research Project NOISE, EASA.2020.FC.06, European Union Aviation Safety Agency, Cologne, December 2023.
39. "Report on Standard Method of Computing Rotorcraft Noise Contours," ECAC.CEAC Doc 32, 1st ed., European Civil Aviation Conference, Neuilly-sur-Seine, France, May 2026.
40. "Application of Pure-Tone Atmospheric Absorption Losses to One-Third Octave-Band Data," SAE ARP5534, SAE International, Warrendale, PA, 2013.
41. "Noise Standards: Aircraft Type and Airworthiness Certification," Title 14, Code of Federal Regulations, Part 36, Appendix A, "Aircraft Noise Measurement and Evaluation Under § 36.101." URL: https://www.ecfr.gov/current/title-14/part-36 [retrieved 8 October 2026].
42. "Environmental Protection, Volume I — Aircraft Noise," Annex 16 to the Convention on International Civil Aviation, 8th ed., International Civil Aviation Organization, Montréal, July 2017.
43. "Design Criteria Standard: Noise Limits," MIL-STD-1474E, U.S. Department of Defense, Washington, DC, 15 April 2015.
44. Vold, H., and Leuridan, J., "High Resolution Order Tracking at Extreme Slew Rates, Using Kalman Tracking Filters," SAE Technical Paper 931288, May 1993. https://doi.org/10.4271/931288
45. Tůma, J., "The Passband Width of the Vold-Kalman Order Tracking Filter," *Transactions of the VŠB – Technical University of Ostrava, Mechanical Series*, Vol. 51, No. 2, 2005, pp. 149–154.
46. Hothersall, D. C., and Harriott, J. N. B., "Approximate Models for Sound Propagation Above Multi-Impedance Plane Boundaries," *Journal of the Acoustical Society of America*, Vol. 97, No. 2, 1995, pp. 918–926. https://doi.org/10.1121/1.412136
47. Plovsing, B., "Nord2000. Comprehensive Outdoor Sound Propagation Model. Part 1: Propagation in an Atmosphere without Significant Refraction," DELTA Acoustics & Vibration Report AV 1849/00 (revised), Hørsholm, Denmark, March 2006.
48. Boulanger, P., Waters-Fuller, T., Attenborough, K., and Li, K. M., "Models and Measurements of Sound Propagation from a Point Source over Mixed Impedance Ground," *Journal of the Acoustical Society of America*, Vol. 102, No. 3, 1997, pp. 1432–1442. https://doi.org/10.1121/1.420101
49. de Jong, B. A., Moerkerken, A., and van der Toorn, J. D., "Propagation of Sound over Grassland and over an Earth Barrier," *Journal of Sound and Vibration*, Vol. 86, No. 1, 1983, pp. 23–46. https://doi.org/10.1016/0022-460X(83)90941-0
50. Lam, Y. W., and Monazzam, M. R., "On the Modeling of Sound Propagation over Multi-Impedance Discontinuities Using a Semiempirical Diffraction Formulation," *Journal of the Acoustical Society of America*, Vol. 120, No. 2, 2006, pp. 686–698. https://doi.org/10.1121/1.2216905
