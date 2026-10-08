# PANAM theory

This document states the models PANAM implements, as equations, with the references they come
from. It describes the code as it is. Where PANAM departs from a cited reference, the departure is
stated, and so is the line between what PANAM provides and what a particular build chose. The
shorter notes in this directory cover the details: [`database_build.md`](database_build.md)
(the 2017 sphere build and the `.nod` format), [`ground_plane_corrections.md`](ground_plane_corrections.md)
(the ground-plate corrections and how they were fitted) and [`norah2_import.md`](norah2_import.md).

PANAM's main product is the database of noise spheres that the NICE-OPS footprint model reads.
The NICE-OPS theory document covers what happens to a sphere after that: blending, propagation to
the ground and the metrics. Several of PANAM's models are the reference NICE-OPS's ports are pinned
to (ISO 9613-1, the ground effect, A-weighting, EPNL).

References are numbered in order of first citation and listed at the end.

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
12. [References](#references)

## 1. Overview and notation

PANAM turns flight-test microphone recordings into source noise spheres. The approach is the
fixed-radius source hemisphere of the Fundamental Rotorcraft Acoustic Modeling from Experiments
(FRAME) method [1, 2, 3]. Each steady flight condition gets
one sphere: the band levels the aircraft radiates in every direction, referred to a fixed radius.
The shipped databases come from the 2017 Noise Abatement flight test [4, 5].

For each run, PANAM:

1. computes short-time spectra at each ground microphone (§2);
2. maps each spectral frame to the direction and distance at which it was emitted (§3);
3. removes the ambient noise and the microphone installation's response (§4), and undoes the
   spreading and absorption back to the reference radius (§5);
4. grids the scattered samples onto a regular sphere, in energy (§6).

A database then collects the spheres by flight condition (§7).

| Symbol | Meaning |
| --- | --- |
| $f$, $k = 2\pi f/c$ | frequency, wavenumber |
| $P$ | mean-square pressure (power) in a bin or band, re $(20\ \mu\mathrm{Pa})^2$ |
| $L = 10\lg P$ | level, dB re 20 µPa |
| $r$, $r_\mathrm{ref}$ | source–microphone distance; reference radius, 100 ft (30.48 m) |
| $e$, $\psi$ | emission elevation (depression below the horizontal) and azimuth |
| $\beta = 1/Z$ | normalized ground admittance ($Z$ the normalized impedance) |

The time dependence is $e^{-i\omega t}$, so a passive ground has $\mathrm{Re}\,\beta > 0$. Lengths
are in feet in the 2017 build and in meters in the database; each function states its units.

## 2. Signals and spectra

### 2.1 Short-time spectra

Microphone pressure (Pa, as stored; PANAM applies no calibration) is analyzed in Hann-windowed
frames [6]. The frame length is the power of two at or above $T_w f_s$ samples. The
spectral density is

$$
\mathrm{PSD}(f) = 10\lg\frac{S_{xx}(f)}{p_\mathrm{ref}^2}, \qquad p_\mathrm{ref} = 20\ \mu\mathrm{Pa},
$$

in dB re $(20\ \mu\mathrm{Pa})^2$/Hz. For depropagation, $T_w = 0.5$ s with 50% overlap, so a frame is 16,384
samples (0.64 s at 25.6 kHz, $\Delta f = 1.5625$ Hz). Each frame is a single periodogram. The
averaging happens later, over the samples that fall in a sphere cell (§6). The spectrum at an
emission point's reception time is interpolated between frame centers **in power**, not in dB.
Welch averaging [7] is available for stationary signals (`psd_welch`).

### 2.2 One-third-octave bands

Band edges follow IEC 61260-1 [8]. A band with a nominal center (within 1/12 octave of
$1000\cdot10^{n/10}$ Hz) takes the base-10 edges $f_m 10^{\pm 1/20}$ around its exact midband
frequency $f_m$. A set of exact base-2 centers $1000\cdot2^{k/3}$ takes $f_c 2^{\pm1/6}$. The
production bands are the 31 nominal centers from 10 Hz to 10 kHz, so production edges are base 10.

**Allocation.** By default a band's power is the sum of whole FFT bins inside its edges:
$P_b = \sum_{f_l \le f < f_u} \mathrm{PSD}\,\Delta f$. With a Doppler factor (§3.3) or tone-aware
banding, the running integral $R(f) = \sum \mathrm{PSD}\,\Delta f$ is interpolated linearly within
bins, and $P_b = R(f_u) - R(f_l)$ counts fractional bins.

**Tone-aware banding.** A rotor harmonic near a band edge splits its window main lobe between
two bands. On the B407 the 27.6 and 55.2 Hz harmonics, 0.6 and 1.0 Hz from band edges, read
5–19 dB high in the 31.5 and 63 Hz bands. `tone_aware_band_power` files each tone whole:

1. The floor is the running median of the PSD over 15 bins. A tone is a local maximum over ±2 bins
   standing more than 6 dB above it.
2. Its offset from the peak bin follows from the Hann window's two-bin interpolation
   [9], with $p_0$ and $p_{l,r}$ the floor-subtracted peak and neighbors:
   $\alpha = \sqrt{\max(p_l, p_r)/p_0}$, $\delta = \pm\,\mathrm{clip}\!\left(\frac{2\alpha - 1}{\alpha + 1}, 0, \tfrac12\right)$, toward the larger neighbor.
3. Its power corrects the peak bin for the window's scalloping, with $\mathrm{ENBW} = 1.5$ bins:

$$
P_\mathrm{tone} = \frac{p_0\, \mathrm{ENBW}\, \Delta f}{\lvert W(\delta)\rvert^2}, \qquad
\lvert W(d)\rvert^2 = \left(\frac{\operatorname{sinc} d}{1 - d^2}\right)^2 .
$$

4. Its contribution $P_\mathrm{tone}\lvert W(o - \delta)\rvert^2/(\mathrm{ENBW}\,\Delta f)$ is
   removed from bins $o = -4 \dots 4$ (clipped at zero). The remainder is banded by the running
   integral, and the tone is added whole to the band containing $f_I + \delta\Delta f$.

A causal Butterworth filter bank (order 3, $f_c 2^{\mp1/6}$ edges, octave decimation) is
available as an alternative (`third_octave_method='filter_bank'`); production uses the FFT.

### 2.3 A-weighting

$$
W_A(f) = 10\lg\frac{1.562339\, f^4}{(f^2 + 107.65265^2)(f^2 + 737.86223^2)}
       + 10\lg\frac{2.242881\times10^{16}\, f^4}{(f^2 + 20.598997^2)^2 (f^2 + 12194.22^2)^2},
$$

the pole frequencies of IEC 61672-1 [10] (12194.22 Hz where the standard gives 12194.217),
normalized to 0 dB at 1 kHz. It is applied per FFT bin in depropagation and per band center in the
EAA (§7.2).

## 3. Emission geometry and timing

### 3.1 Reception time

The track gives the source position $\mathbf{x}_s(t_e)$ and velocity $\mathbf{v}$ at emission
times $t_e$. For each microphone at $\mathbf{x}_m$, the emission maps forward to reception:

$$
\mathbf{r} = \mathbf{x}_m - \mathbf{x}_s(t_e), \qquad
t_\mathrm{obs} = t_e + \frac{\lvert\mathbf{r}\rvert}{c}, \qquad
M_r = \frac{\mathbf{v}\cdot\mathbf{r}}{\lvert\mathbf{r}\rvert c},
$$

with $c = 343.2\sqrt{T/293.15}$ m/s at the run's station temperature (§5.4) and no wind. No
retarded-time equation has to be solved: the track is sampled in emission time. With a ray model
(§5.3), $t_\mathrm{obs} = t_e + t_\mathrm{ray}$. Emission samples are decimated (`point_stride`; 10
in the 2017 release, so 50 Hz tracking becomes 5 Hz).

### 3.2 Directions

PANAM files a sphere in a **level, horizontal frame**. The elevation is the depression of the
microphone below the horizontal at the source, and the azimuth is measured from the aircraft's
horizontal ground track:

$$
e = \operatorname{atan2}\!\left(z_s - z_m,\ \sqrt{r_x^2 + r_y^2}\right), \qquad
\psi = \left(\operatorname{atan2}(r_y, r_x) - h - 180^\circ\right) \bmod 360^\circ,
$$

with $h = \operatorname{atan2}(v_y, v_x)$, or the inertial heading for hover and under
`azimuth_reference='heading'`. This "UMAPR" convention puts 180° ahead, 90° to starboard and 0°
aft. There is no tip-path-plane tilt. In steady flight NICE-OPS's sphere frame coincides with it,
because its tip-path-plane normal is the load-factor direction, which is vertical when the
aircraft does not accelerate.

The AAM/NORAH2 ("ART") sphere angles are $\theta$ from the nose and $\phi$ from straight down,
positive to starboard [11]:

$$
\hat{\mathbf{d}} = (\cos\theta,\ \sin\theta\sin\phi,\ \sin\theta\cos\phi)\quad(\text{forward, starboard, down}).
$$

**Track frame.** Positions are in the test's local frame: $+x$ along the layout's true bearing $b$
(270° for the Amedee flight layout, 279° for its hover layout, 140° and 92.3° at Eglin), $+y$ to
the left, $z$ up. Latitude and longitude map to local east and north on the WGS84 radii of
curvature at the reference point [12], then rotate by $b$:
$x = E\sin b + N\cos b$, $y = -E\cos b + N\sin b$. PANAM checks $b$ against a fit of the track and
refuses a disagreement over 1°.

### 3.3 Doppler

The received frequency is $f = D f_e$, with $D = 1/(1 - M_r)$. By default (`DOPPLER_SHIFT_REMOVED = 0`)
the spheres keep the received, Doppler-shifted spectra, and the bands are the received bands.
`remove_doppler` instead integrates each emitted band $[f_l, f_u]$ over $[D f_l, D f_u]$ of the
received spectrum, and files tones at $f_\mathrm{peak}/D$. Only frequencies are rescaled: no
convective amplification factor $D^n$ is applied to the levels. In the 2017 release it is used
only for the run the hover is synthesized from (§7.4).

### 3.4 Steady segments

A run contributes its longest steady segment. Taking reference values as the medians of ground
speed and flight path angle over the half of the record nearest the array, a sample is steady
when it lies within 4 kt and 2° of them, its roll is within 5°, and its heading changes by less
than 1.5°/s. Gaps of 2 s or less are closed, and the segment must last at least 8 s. Its flight
condition is the segment's mean ground speed and mean flight path angle. Hovers use the whole
record, with the nose along the heading.

## 4. Ground and receiver models

### 4.1 Ground impedance

`ground_plane.surface_admittance` provides four locally reacting models, with $X = f/\sigma$
($\sigma$ in kPa·s/m²), $\rho_0 = 1.2$ kg/m³ and $\gamma = 1.4$:

| Model | Impedance |
| --- | --- |
| Delany–Bazley [13] | $Z = 1 + 9.08X^{-0.75} + 11.9iX^{-0.73}$ |
| Miki [14] | $Z = 1 + 5.50X^{-0.632} + 8.43iX^{-0.632}$ |
| Variable porosity [15] | $Z = (1+i)\sqrt{\dfrac{1000\,\sigma_e}{\pi\gamma\rho_0 f}} + \dfrac{i c_0 \alpha_e}{8\pi\gamma f}$ |
| Hard-backed Delany–Bazley layer, depth $d$ [15] | $Z = Z_c\coth(-ikd)$ |

For the layer, $Z_c = 1 + 9.08X^{-0.754} + 11.9iX^{-0.732}$ and
$k = (2\pi f/c_0)(1 + 10.8X^{-0.70} + 10.3iX^{-0.595})$. A thin layer of low resistivity is not
passive at low frequency, and is refused. The 2017 site ground is variable porosity with
$\sigma_e = 200$ kPa·s/m² and $\alpha_e = 0$ (§4.6). The first term of the variable-porosity form is
the familiar $0.436(1+i)\sqrt{\sigma_e/f}$ with $\sigma_e$ in Pa·s/m².

### 4.2 The spherical-wave reflection coefficient

For a source at height $h_s$ and a receiver at $h_r$, a horizontal distance $d$ apart, the
direct and image paths are $R_1 = \sqrt{d^2 + (h_s - h_r)^2}$ and
$R_2 = \sqrt{d^2 + (h_s + h_r)^2}$. With $\cos\vartheta = (h_s + h_r)/R_2$, measured from the normal
(the code's `grazing_angle` is this angle):

$$
R_p = \frac{\cos\vartheta - \beta}{\cos\vartheta + \beta}, \qquad
Q = R_p + (1 - R_p)\left[1 + i\sqrt{\pi}\,p_w\,\mathrm{w}(p_w)\right], \qquad
p_w = \sqrt{\frac{i\pi f R_2}{c\,(1 + \beta\cos\vartheta)}}\,(\cos\vartheta + \beta).
$$

This is the corrected Chien–Soroka form [16, 17]. $\mathrm{w}(z) = e^{-z^2}\mathrm{erfc}(-iz)$
is the Faddeeva function (SciPy's `wofz`). The code keeps $p_w$ itself rather than the principal
root of $w = p_w^2$. The two differ only for a mass-like ground ($\mathrm{Im}\,\beta > 0$), where the
principal root would admit a growing surface wave. There is no cutoff in $\lvert w\rvert$.

### 4.3 Excess ground attenuation: `ega()`

`ega()` is the ground-effect model of the Wyle short-range propagation model
[18], after Chien and Soroka and Chessell [19], with the corrections noted
by Daigle, Embleton and Piercy [20]. It uses the Delany–Bazley impedance and the path
delay $\tau = (R_2 - R_1)/c$. For a pure tone:

$$
\Delta L = 20\lg\left\lvert 1 + Q\,\frac{R_1}{R_2}\, e^{2\pi i f\tau}\right\rvert .
$$

For a one-third-octave band, Chessell's band average, with $q = \lvert Q\rvert R_1/R_2$:

$$
\Delta L = 10\lg\left[1 + q^2 + 2q\cos(\eta f\tau + \arg Q)\,\frac{\sin(\mu f\tau)}{\mu f\tau}\, T\right],
\qquad T = \exp\!\left[-\left(\tfrac12 C f \sqrt{R_1}\right)^2\right],
$$

where $\eta = \pi(2^{1/6} + 2^{-1/6}) = 6.325159$ and $\mu = \pi(2^{1/6} - 2^{-1/6}) = 0.727477$.
They come from averaging $\cos(2\pi f'\tau + \arg Q)$ uniformly in $f'$ over the band
$[f2^{-1/6}, f2^{1/6}]$. $T$ is Chessell's turbulent decoherence, with $C$ in rad·s/√length
(typically 0–16×10⁻⁴ rad·s/√m). The code comments call $\eta$ and $\mu$ "ground reflection" and
"spherical spreading" coefficients; they are the band-average constants above. NICE-OPS's ground
model is ported from this function and pinned to it at $10^{-9}$ dB.

### 4.4 Pole microphones

`pole_level` is the band-averaged level of a point microphone above any of the four grounds, re
free field. It generalizes `ega()` (to which it reduces, to $2\times10^{-11}$ dB, with
Delany–Bazley and no turbulence or roughness):

$$
E = a_d^2 + \left(a_r q\right)^2 + 2a_d a_r q\cos(\eta f\tau + \arg Q)\,\frac{\sin(\mu f\tau)}{\mu f\tau}\,C_\mathrm{turb},
\qquad \Delta L = 10\lg E,
$$

with $a_d$ and $a_r$ the microphone's free-field response at each path's incidence. Two
refinements act on the reflected path:

- **Turbulence** reduces its coherence with the HARMONOISE form [21, 15],
  $C_\mathrm{turb} = \exp\left[-\tfrac38 B k^2 \gamma_p^{5/3} R_1 \gamma_T\right]$, with $B = 0.364$,
  $\gamma_p = h_s h_r/(h_s + h_r)$, and the strength $\gamma_T = C_T^2/T_0^2 + 22C_v^2/(3c_0^2)$
  from the temperature and wind structure parameters [22].
- **Roughness** of rms height $\sigma_h$ reduces its amplitude by the Kirchhoff (Ament) factor
  [23]: $a_r \leftarrow a_r\exp[-2(k\sigma_h\cos\vartheta)^2]$.

### 4.5 Ground-plate microphones

The 2017 ground microphones are GRAS 67AX microphones flush in GR1425 plates lying on the
ground: radius $a = 0.2$ m, 8 mm thick, tapering to a 2.5 mm edge (a 20 mm taper length is
assumed), with the microphone at $0.75a$ from the center, outboard of the track
[4, 24]. A plate is not an ideal pressure-doubling surface: it is finite and
raised, and the ground around it is not rigid [25]. Kingan, Go and co-workers model
ground-board microphones with a boundary element method over impedance ground [26, 27].
PANAM computes the response with an axisymmetric boundary element method of the plate's own
profile (`axisymmetric_bem.py`).

**The Green's function** is the exact one for a point source over a locally reacting plane
[28, 29]:

$$
G = g(R_1) + g(R_2) - 2k\beta\int_0^\infty e^{-k\beta q}\, g(R_q)\, dq, \qquad
g(R) = \frac{e^{ikR}}{4\pi R}, \qquad R_q = \sqrt{\rho^2 + (Z + iq)^2},
$$

with $\rho$ the horizontal and $Z$ the summed vertical distance. The integral is evaluated by
graded Gauss–Legendre quadrature and tabulated. It agrees with the Sommerfeld integral to
$2\times10^{-9}$.

**The integral equation.** With $G$ satisfying the ground's boundary condition, only the rigid
plate's surface $S$ remains, and the exterior Neumann problem is

$$
\tfrac12\, p(\mathbf{x}) = p_\mathrm{inc}(\mathbf{x}) + \int_S p(\mathbf{y})\,\frac{\partial G}{\partial n_y}\, dS_y .
$$

The plate is a body of revolution. Expanding $p = \sum_m p_m(s)e^{im\varphi}$ in azimuth decouples
the modes, $|m| \le \lceil ka\rceil + 10$. Each mode solves a one-dimensional equation on the
generating curve, collocated at segment midpoints. Plane waves arriving at elevation $e$ and
azimuth $\psi$ expand by the Jacobi–Anger identity, $i^m J_m(k r\cos e)\,e^{\mp ik z\sin e}$.

**The response.** The direct (downward) and ground-reflected (upward) incident waves are solved
separately, giving transfer functions $P_d$ and $P_r$ at the microphone, both normalized by the
direct wave there. The field at the microphone is $P_d + Q P_r$, with $Q$ from §4.2 evaluated at
the plate's top. The band response averages five sub-frequencies
$f_j = f_c 2^{(j + 0.5)/15 - 1/6}$ across the band:

$$
T_\mathrm{board} = 10\lg\left[\frac15\sum_{j=0}^{4}\left\lvert P_d(f_j) + Q(f_j)P_r(f_j)\right\rvert^2\right]\quad \text{dB re free field}.
$$

$P_d$ and $P_r$ are tabulated on 80 elevations (0.05°–90°) and 36 azimuths, and interpolated
bilinearly. The tables can be written for NICE-OPS's `--plate_table`. Checks: a rigid sphere
against its exact series to $2.7\times10^{-4}$, and a thin plate on rigid ground to within
0.3–0.8% of pressure doubling. The simpler `'flat'` correction, a uniform −6.02 dB, is the
alternative.

### 4.6 What was fitted

The site ground was fitted to the 2017 measurements (details in
[`ground_plane_corrections.md`](ground_plane_corrections.md)). Pole–board pairs at elevations above
30° constrain the turbulence strength, $\gamma_T \approx 3\times10^{-5}\ \mathrm{m}^{-2/3}$. The
resistivity is poorly identified at steep angles. A held-out profile over resistivity selected
variable porosity with $\sigma_e \approx 200$ kPa·s/m² and roughness of 10–15 mm, which is the
adopted `SITE_GROUND`.

## 5. Depropagation

### 5.1 The sequence

Each emission sample carries a received power spectrum $P(f)$ at a microphone. Depropagation
refers it to the sphere radius $r_\mathrm{ref} = 100$ ft in four steps, in this order:

1. **Ambient gate.** With $A(f)$ the ambient spectrum (the median over frames of a measured
   ambient run on the same layout, or a low percentile of the run's own spectrogram), a bin is kept
   only if it stands $G$ dB above the ambient, and the ambient is then subtracted:
   $P \leftarrow \max(P - A, 0)$ where $P \ge A\,10^{G/10}$, and 0 elsewhere.
2. **Receiver response.** $P \leftarrow P\cdot10^{-T_\mathrm{board}/10}$ (§4.5), per band. A cap
   discards bins whose correction would exceed a set number of dB.
3. **Spreading.** $P \leftarrow P\,(r_s/r_\mathrm{ref})^2$, with $r_s$ the straight distance or the
   ray tube's equivalent range.
4. **Absorption.** $P \leftarrow P\cdot10^{\alpha(f)(\ell - r_\mathrm{ref})/10}$, with $\alpha$ the
   ISO 9613-1 coefficient at the received frequency (§5.4) and $\ell$ the straight distance or the
   ray's arc length. Bins whose correction would exceed a cap are discarded.

The gate comes first: an ambient-limited bin would otherwise be amplified by both corrections. A
discarded bin carries zero power. **The sphere keeps the absorption over its first
$r_\mathrm{ref}$** in the run's own atmosphere; the database's EAA (§7.2) accounts for absorption
beyond it.

The 2017 release used a 10 dB gate, a 15 dB response cap, a 30 dB absorption cap, a 2000 ft
range limit and a 2° floor on the emission elevation. These are settings of the build script in the
validation harness; PANAM's own defaults are looser.

### 5.2 Range and elevation limits

A sample is dropped if its emission elevation is below `min_elevation_deg`, if it is farther than
`max_range` (with a longer `rim_range` allowed near the horizon), if its ray is in a shadow, or if
its reception time falls outside the recording.

### 5.3 Refracted rays

PANAM does not trace rays itself. `refracted_rays.external_ray_model` calls NICE-OPS's
`niceops_ray_geometry` through the run's measured atmosphere, and takes from it the launch angle,
the travel time, the ray-tube spreading range, the arc length, and the geometry the plate sees at
the ray's arrival. The ray physics is in the NICE-OPS theory document. A sample is then filed at
its launch angle rather than the straight-line elevation. The absorption coefficient stays uniform
along the ray.

### 5.4 The atmosphere

**Absorption** is the ISO 9613-1 coefficient [30]. With $T_0 = 293.15$ K,
$T_{01} = 273.16$ K, $p_r = 101.325$ kPa, temperature $T$, pressure $p_a$ and relative humidity
$h_r$:

$$
h = h_r\,10^{\,-6.8346(T_{01}/T)^{1.261} + 4.6151}\,\frac{p_r}{p_a}, \qquad
f_{rO} = \frac{p_a}{p_r}\left(24 + 4.04\times10^4 h\frac{0.02 + h}{0.391 + h}\right),
$$

$$
f_{rN} = \frac{p_a}{p_r}\left(\frac{T}{T_0}\right)^{-1/2}\left(9 + 280h\,e^{-4.170[(T/T_0)^{-1/3} - 1]}\right),
$$

$$
\alpha = 8.686f^2\left[1.84\times10^{-11}\frac{p_r}{p_a}\left(\frac{T}{T_0}\right)^{1/2} + \left(\frac{T}{T_0}\right)^{-5/2}\left(\frac{0.01275\,e^{-2239.1/T}}{f_{rO} + f^2/f_{rO}} + \frac{0.1068\,e^{-3352.0/T}}{f_{rN} + f^2/f_{rN}}\right)\right]\ \mathrm{dB/m}.
$$

It is evaluated at single frequencies (a bin or a band center), with no integration over a band.

**The run's atmosphere** is uniform: the mean over the site's ground weather stations of the
samples nearest the run's start time. A stratified profile (balloon temperature, LIDAR wind) enters
only through the ray model of §5.3. The air density, used for the run's thrust coefficient, is that
of moist air,
$\rho = (p - 0.378e)/(R_d T)$ with $e$ the vapor pressure and $R_d = 287.058$ J/(kg·K), reduced
hydrostatically to the aircraft's height.

## 6. Gridding onto the sphere

### 6.1 Inverse-distance weighting in energy

The scattered depropagated samples are gridded onto a regular grid of azimuth and elevation (10°
by default; 2° in the 2017 release). Every quantity is interpolated **in power**, then converted
to dB. Distances are great-circle arcs, with elevation as latitude:

$$
\cos h = \sin e_1\sin e_2 + \cos e_1\cos e_2\cos(\psi_1 - \psi_2).
$$

The weights are Franke and Nielson's modified Shepard weights [31, 32] within a
radius $R$:

$$
w_i = \left(\frac{R - h_i}{R\,h_i}\right)^2\ (h_i \le R), \qquad
P(\text{node}) = \frac{\sum_i w_i P_i}{\sum_i w_i}.
$$

A node with no sample in reach is empty, not zero.

### 6.2 The adaptive radius

`adaptive_idw_weights` sets each node's radius from the data around it:

$$
R = \max\left(\kappa\, d_k,\ \kappa\, d_M,\ \tfrac12\tilde{s}\right),
$$

with $\kappa = 1.3$, $d_k$ the distance to the 8th nearest sample, $d_M$ the distance to the 3rd
nearest distinct microphone, and $\tilde{s}$ the median angular resolution of the nearest samples,
$s = \lvert\mathbf{v}_\perp\rvert T_w/r$ (the angle the source sweeps across the line of sight in
one frame). A radius above 60° makes the node a gap. Two options shape the kernel:

- **Aspect** stretches the distance in azimuth, $h = \sqrt{\Delta e^2 + (\Delta\psi\cos\bar{e}/A)^2}$, so that
  samples along a flight pass (which spread in azimuth) share more than samples across it. The
  2017 release used $A = 5$.
- **Floor** softens the Shepard singularity, $h \to \sqrt{h^2 + (fR)^2}$ in the denominator, so
  that one sample does not dominate a node it nearly hits. The 2017 release used $f = 1$.

### 6.3 Output grids and coverage

The gridded sphere is resampled onto the AAM grid ($\phi$ every 10°, $\theta$ every 5°, or a
reference sphere's own grid) bilinearly in power, renormalized by the interpolated weight of
measured cells. A direction whose measured weight is below 0.5 is missing (−999 in the file). Each
database group carries a `coverage` flag per cell, set where the cell was measured. For plots,
spheres are drawn in the azimuthal equal-area projection centered on the nadir [33]:
$x = q\sin\lambda$, $y = q\cos\lambda$, with $q = 2\sin[(90^\circ - e)/2]$ and $\lambda = 180^\circ - \psi$.

## 7. The database

### 7.1 Flight condition

Each sphere is labeled by advance ratio, flight path angle and thrust coefficient:

$$
\mu = \frac{0.514444\,V}{V_\mathrm{tip}}, \qquad \gamma = \bar{\gamma}_\mathrm{track}, \qquad
C_T = n\,C_W, \qquad C_W = \frac{W}{\rho\,\pi R^2 V_\mathrm{tip}^2},
$$

with $V$ the segment's mean **ground** speed in knots, $\gamma$ its mean flight path angle, and
$W$, $\rho$, $R$ and $V_\mathrm{tip}$ the vehicle file's nominal values. The run's own thrust
coefficient, from its weight and air density, is recorded as metadata but does not label the
sphere.

### 7.2 Excess atmospheric attenuation

A NICE-OPS broadband run carries the A-weighted level to the ground with one attenuation
coefficient per direction, the EAA. PANAM computes it from the sphere's band spectrum $S_i$:

$$
\mathrm{SPLA} = 10\lg\sum_i 10^{(S_i + W_A(f_i))/10}, \qquad
\mathrm{EAA} = \mathrm{SPLA} - 10\lg\sum_i 10^{(S_i + W_A(f_i) - 1000\,\alpha(f_i))/10},
$$

with $\alpha$ in the build atmosphere (293.15 K, 101.325 kPa and 20% RH unless set), recorded in
the file. NICE-OPS consumes it as $\mathrm{EAA}\,(r - R_s)/1000$ m.

### 7.3 Load factor

A `fixed_load_factor` database stores only the 1 g spheres, with $C_{T,\mathrm{ref}} = C_W$, and
the reader applies $20\lg(C_T/C_{T,\mathrm{ref}})$ above 1 g. A database whose load factors are spheres
stores each condition at a ladder of load factors $n$ (the shipped `ambient_gated` files use
0, 1, 1.1, 1.2, 1.5, 2, 3 and 5), with $\mathrm{dBA} \leftarrow \mathrm{dBA} + 20\lg n$ for $n > 0$.
Either way the load-factor law is the same; only where it is applied differs.

### 7.4 Hover

A hover passes no microphones, so the arrays see it only along fixed rings of elevation (for the
B407, 0.5–3° and 18–25° below the horizon), and the rest of its sphere has to be synthesized. The source is the slowest level pass, rebuilt with the Doppler shift
removed (§3.3), averaged fore and aft in energy so that it has no direction of flight:

$$
L_i = L_{n-1-i} = 10\lg\left[\tfrac12\left(10^{L_i/10} + 10^{L_{n-1-i}/10}\right)\right]
$$

over the $n$ polar rows. PANAM accepts a further correction in dB as a function of direction and
band. For the B407 the validation harness supplies one fitted to the measured hover rings: a
Fourier series in azimuth from the heading, $\sum_m (a_m\cos m\psi + b_m\sin m\psi)$, fitted per
ring, blended linearly in elevation from 3° to 17.5°, with term $m$ fading toward the nadir as
$(\cos e/\cos e_\mathrm{ring})^m$. The hover is written at speed 0 and at each path angle the
database spans.

### 7.5 Padding and completion

Level conditions are copied to path angles of −24° and +35°, so that the condition hull covers
steep states. The upper hemisphere, which no ground microphone sees, is completed by mirroring
through the tip-path plane, $\phi' = 180^\circ - \phi$, and carries coverage 0. Database format
version 1 marks files with the corrected mirror (an earlier `-phi` mirror reflected the lower
hemisphere onto itself).

## 8. NORAH2 and AAM interchange

**NORAH2 import.** A NORAH2 hemisphere (`.hem`) [34, 35, 36] gives one-third-octave levels on
a 10° grid at POLDIST = 60 m, free field with absorption to that distance in the file's own
atmosphere. PANAM refers it to the NICE-OPS sphere radius $r = 30.48$ m:

$$
L(r) = L(60\ \mathrm{m}) + 20\lg\frac{60\ \mathrm{m}}{r} + \alpha_\mathrm{file}(f_c)\,(60\ \mathrm{m} - r),
$$

with ISO 9613-1 at the band centers. **Departure:** NORAH2 itself applies SAE ARP 5534's band
correction [37], which is not reproduced. The angle conventions agree with NICE-OPS's, so no
rotation is needed. NORAH2's speed field is labeled indicated airspeed but used as ground speed,
so the importer requires the user to say which (`ground_speed`, or `ias_to_tas` with
$V_\mathrm{TAS} = V_\mathrm{IAS}\sqrt{1.225/\rho}$). A file whose levels include the ground
reflection (FREEFIELD = 0) is refused unless accepted explicitly.

**Export.** Spheres can be written in the AAM netCDF format [11], in the variable order AAM
3.1 reads by position, and as NORAH2 hemispheres, with the inverse of the conversion above against
the ICAO reference atmosphere (298.15 K, 70% RH).

## 9. Metrics

**Sound exposure level.** Over the 10 dB-down window (the first to the last sample within 10 dB of
the maximum, dips included), $\mathrm{SEL} = 10\lg\left(\sum_k 10^{L_k/10}\Delta t/1\ \mathrm{s}\right)$
[38]. Unlike NICE-OPS, PANAM's samples are already in reception time.

**EPNL** follows 14 CFR 36, Appendix A, §A36.4 [38], identical to ICAO Annex 16 [39]:
noy values from Table A36-3; $\mathrm{PNL} = 40 + \frac{10}{\lg 2}\lg(0.85n_\max + 0.15\sum n)$;
the ten-step tone correction with Table A36-2; the band-sharing check of §A36.4.4.2; and the
duration correction over the records within 10 dB of PNLTM, with $T_0 = 10$ s exactly (the
regulation's −13 for 0.5 s records is that value rounded). Where readings of the regulation
differ, PANAM puts 500 Hz in the middle range of Table A36-2 and takes the duration as one
contiguous interval. NICE-OPS's EPNL is ported from this implementation.

**Aural nondetectability.** MIL-STD-1474E Table C-1 [40] gives, per one-third-octave
band, the level at which a sound is undetectable at each distance. A spectrum normalized to 10 m,
$L_{10} = L + 20\lg(d/10)$, is shifted to each of the table's measurement distances and
interpolated in $\lg$ distance. The nondetectability distance is the largest over the bands.

## 10. Auxiliary models

These are in PANAM but not used to build the shipped databases.

- **Vold–Kalman order tracking** (`vold_kalman_filter.py`) [41, 42]. Each order $k$
  is a slowly varying complex envelope $a_k$ on a known phasor
  $\Theta_k(n) = \exp(2\pi i\,\Delta t\sum_{m \le n} f_k(m))$. The data equation
  $x = \sum_k a_k\Theta_k + \eta$ and a structural equation of order $p$,
  $\sum_{s=0}^{p}(-1)^s\binom{p}{s}a_k(n+s) = \varepsilon_k(n)$, are solved together by least
  squares, with the weight $r$ set from the −3 dB bandwidth.
- **Ground-board models** in `ground_plane.py`, from the plate study: Fresnel-zone weighting of a
  finite plate (strip [43] and disc [44]), blended in energy
  [45] or in dB; an impedance-discontinuity model after De Jong and Lam and Monazzam
  [46, 47]; and a thin-disc BEM [26]. The axisymmetric BEM of §4.5
  superseded them.
- **Array planning** (`array_planner.py`): equal angular spacing of microphones along a line,
  $y_i = (h/\sin e_t)\tan\theta_i$, and the coverage of a planned trajectory on the sphere.
- **Footprint preview** (`project_sphere`): a sphere carried straight down to the ground with
  spreading and EAA, for plots.

## 11. What is not modeled

- **Tip-path-plane orientation.** Spheres are filed in a level frame (§3.2).
- **Convective amplification.** Doppler removal rescales frequency only (§3.3).
- **A stratified atmosphere without rays.** Absorption and timing use one uniform atmosphere; only
  the optional ray model sees the profile (§5.3).
- **Band-integrated absorption.** ISO 9613-1 is evaluated at single frequencies.
- **The variable-porosity model's second term** has not been checked against its source. Its
  coefficient, $c_0/(8\pi\gamma) \approx 9.75$ m/s, is half the value usually quoted for that model,
  $c_0/(4\pi\gamma)$. It does not affect the shipped spheres, which use $\alpha_e = 0$.

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
21. van Maercke, D., and Defrance, J., "Development of an Analytical Model for Outdoor Sound Propagation Within the Harmonoise Project," *Acta Acustica united with Acustica*, Vol. 93, No. 2, 2007, pp. 201–212.
22. Ostashev, V. E., and Wilson, D. K., *Acoustics in Moving Inhomogeneous Media*, 2nd ed., CRC Press, Boca Raton, FL, 2015. https://doi.org/10.1201/b18922
23. Ament, W. S., "Toward a Theory of Reflection by a Rough Surface," *Proceedings of the IRE*, Vol. 41, No. 1, 1953, pp. 142–146. https://doi.org/10.1109/JRPROC.1953.274171
24. "Ground-Plane Microphone Configuration for Propeller-Driven Light-Aircraft Noise Measurement," SAE ARP4055A, SAE International, Warrendale, PA, 2020.
25. Shivashankara, B. N., and Stubbs, G. W., "Ground Plane Microphone for Measurement of Aircraft Flyover Noise," *Journal of Aircraft*, Vol. 24, No. 11, 1987, pp. 751–758. https://doi.org/10.2514/3.45517
26. Kingan, M. J., Go, S. T., Piscoya, R., and Ochmann, M., "On the Modelling of Ground-Board Mounted Microphones for Outdoor Noise Measurements," *Journal of Sound and Vibration*, Vol. 565, 2023, Paper 117894. https://doi.org/10.1016/j.jsv.2023.117894
27. Go, S. T., Kingan, M. J., Schmid, G., and Hall, A., "On the Use of Ground-Board Mounted Microphones for Outdoor Noise Measurements," *Journal of Sound and Vibration*, Vol. 584, 2024, Paper 118432. https://doi.org/10.1016/j.jsv.2024.118432
28. Ochmann, M., "The Complex Equivalent Source Method for Sound Propagation over an Impedance Plane," *Journal of the Acoustical Society of America*, Vol. 116, No. 6, 2004, pp. 3304–3311. https://doi.org/10.1121/1.1819504
29. Taraldsen, G., "The Complex Image Method," *Wave Motion*, Vol. 43, No. 1, 2005, pp. 91–97. https://doi.org/10.1016/j.wavemoti.2005.07.001
30. "Acoustics — Attenuation of Sound during Propagation Outdoors — Part 1: Calculation of the Absorption of Sound by the Atmosphere," ISO 9613-1:1993, International Organization for Standardization, Geneva, 1993.
31. Shepard, D., "A Two-Dimensional Interpolation Function for Irregularly-Spaced Data," *Proceedings of the 1968 23rd ACM National Conference*, ACM, New York, 1968, pp. 517–524. https://doi.org/10.1145/800186.810616
32. Franke, R., and Nielson, G., "Smooth Interpolation of Large Sets of Scattered Data," *International Journal for Numerical Methods in Engineering*, Vol. 15, No. 11, 1980, pp. 1691–1704. https://doi.org/10.1002/nme.1620151110
33. Snyder, J. P., "Map Projections — A Working Manual," U.S. Geological Survey Professional Paper 1395, U.S. Government Printing Office, Washington, DC, 1987. https://doi.org/10.3133/pp1395
34. Olsen, H., Tuinstra, M., and van Oosten, N., "D1.5d Rotorcraft Noise Modelling Guidance," Research Project NOISE, EASA.2020.FC.06, Royal Netherlands Aerospace Centre (NLR), for the European Union Aviation Safety Agency, Cologne, January 2024.
35. van Oosten, N., Ionescu, S. E., Meliveo, L., Konovalova, O., van der Meulen, J. M., and Tuinstra, M., "D2.3 NORAH Hemisphere Database Extension," Research Project NOISE, EASA.2020.FC.06, European Union Aviation Safety Agency, Cologne, December 2023.
36. "Report on Standard Method of Computing Rotorcraft Noise Contours," ECAC.CEAC Doc 32, 1st ed., European Civil Aviation Conference, Neuilly-sur-Seine, France, May 2026.
37. "Application of Pure-Tone Atmospheric Absorption Losses to One-Third Octave-Band Data," SAE ARP5534, SAE International, Warrendale, PA, 2013.
38. "Noise Standards: Aircraft Type and Airworthiness Certification," Title 14, Code of Federal Regulations, Part 36, Appendix A, "Aircraft Noise Measurement and Evaluation Under § 36.101." URL: https://www.ecfr.gov/current/title-14/part-36 [retrieved 8 October 2026].
39. "Environmental Protection, Volume I — Aircraft Noise," Annex 16 to the Convention on International Civil Aviation, 8th ed., International Civil Aviation Organization, Montréal, July 2017.
40. "Design Criteria Standard: Noise Limits," MIL-STD-1474E, U.S. Department of Defense, Washington, DC, 15 April 2015.
41. Vold, H., and Leuridan, J., "High Resolution Order Tracking at Extreme Slew Rates, Using Kalman Tracking Filters," SAE Technical Paper 931288, May 1993. https://doi.org/10.4271/931288
42. Tůma, J., "The Passband Width of the Vold-Kalman Order Tracking Filter," *Transactions of the VŠB – Technical University of Ostrava, Mechanical Series*, Vol. 51, No. 2, 2005, pp. 149–154.
43. Hothersall, D. C., and Harriott, J. N. B., "Approximate Models for Sound Propagation Above Multi-Impedance Plane Boundaries," *Journal of the Acoustical Society of America*, Vol. 97, No. 2, 1995, pp. 918–926. https://doi.org/10.1121/1.412136
44. Plovsing, B., "Nord2000. Comprehensive Outdoor Sound Propagation Model. Part 1: Propagation in an Atmosphere without Significant Refraction," DELTA Acoustics & Vibration Report AV 1849/00 (revised), Hørsholm, Denmark, March 2006.
45. Boulanger, P., Waters-Fuller, T., Attenborough, K., and Li, K. M., "Models and Measurements of Sound Propagation from a Point Source over Mixed Impedance Ground," *Journal of the Acoustical Society of America*, Vol. 102, No. 3, 1997, pp. 1432–1442. https://doi.org/10.1121/1.420101
46. de Jong, B. A., Moerkerken, A., and van der Toorn, J. D., "Propagation of Sound over Grassland and over an Earth Barrier," *Journal of Sound and Vibration*, Vol. 86, No. 1, 1983, pp. 23–46. https://doi.org/10.1016/0022-460X(83)90941-0
47. Lam, Y. W., and Monazzam, M. R., "On the Modeling of Sound Propagation over Multi-Impedance Discontinuities Using a Semiempirical Diffraction Formulation," *Journal of the Acoustical Society of America*, Vol. 120, No. 2, 2006, pp. 686–698. https://doi.org/10.1121/1.2216905
