"""Excess ground attenuation over an impedance ground.

:func:`spherical_reflection_coefficient` is the spherical-wave reflection
coefficient and :func:`ega` the excess ground attenuation, with the Chessell
form of the boundary loss (``docs/THEORY.md`` section 4).  Flow resistance is in
kPa s/m^2.  Lengths and the sound speed are in any one consistent unit (ft and ft/s in the
2017 build, or m and m/s); the value of :func:`ega`'s ``cturb`` depends on the length unit.
"""

import numpy as np
from scipy.special import wofz


def spherical_reflection_coefficient(cos_grazing, image_range, f, a, flores,
                                     boundary_loss_correction=True, admittance=None):
    """Spherical-wave reflection coefficient Q of a locally reacting ground (e^{-i omega t}).

    Q = Rp + F(w) (1 - Rp): the plane-wave coefficient Rp with Chessell's
    boundary-loss factor F, over a Delany-Bazley ground of flow resistance
    ``flores`` (kPa s/m^2).  ``cos_grazing`` is, despite its name, the cosine of
    the angle from the surface normal at the specular point, ``image_range`` the image-source path
    length, ``a`` the sound speed (lengths and speed in matching units).  This is
    the coefficient :func:`ega` uses; it is exposed so that models needing the
    complex coefficient itself (impedance discontinuities, ground planes) use the
    same one.

    ``admittance``, when given, is the normalized surface admittance beta
    (same shape as, or broadcastable to, ``f``) and replaces the
    Delany-Bazley one from ``flores`` -- for other impedance models.
    """
    f = np.asarray(f, dtype=float)
    if admittance is None:
        flores = np.asarray(flores, dtype=float)
        # Delany-Bazley admittance beta = 1/Z in X = f / flores; X^-0.73 / X^0.02
        # is ground_plane.surface_admittance's X^-0.75, kept in this form.
        x = f / flores
        x_073 = x ** (-0.73)
        beta = 1.0 / (1.0 + 9.08 * x_073 / (x ** 0.02) + 1j * 11.9 * x_073)
    else:
        beta = np.asarray(admittance, dtype=complex)

    plane_wave_coeff = (cos_grazing - beta) / (cos_grazing + beta)  # Plane wave reflection coefficient
    if not boundary_loss_correction:
        return plane_wave_coeff

    # Compute numerical distance (simplified: 0.5*k1 = π*f/a).  This is the
    # root z of the numerical distance w = z^2.
    ground_effect_param = np.sqrt(
        1j * np.pi * f * image_range / a / (1.0 + beta * cos_grazing)
    ) * (cos_grazing + beta)

    # Boundary loss factor F = 1 + i sqrt(pi w) exp(-w) erfc(-i sqrt(w)), with
    # exp(-z^2) erfc(-i z) as the Faddeeva function: accurate at any |w|, where
    # exp(-w) and erfc evaluated apart overflow, which once forced a cutoff at
    # |w| = 500 that stepped Q by up to 2e-3.
    #
    # sqrt(w) is the parameter itself, not the principal root of its square.
    # The two agree wherever its real part is positive, which is every passive
    # ground but a mass-like one (Im beta > 0, as a thin hard-backed layer
    # gives in some bands).  There the parameter is in the second quadrant and
    # the principal root is its negative, whose wofz carries a surface wave
    # growing as exp(|Re w|) with range: |Q| of 1e7 at |w| = 50, an overflow
    # by |w| ~ 1e3.  Such a surface has no surface wave, and the parameter
    # gives none.
    boundary_loss = 1 + 1j * np.sqrt(np.pi) * ground_effect_param * wofz(ground_effect_param)

    # Combined reflection + boundary loss
    return plane_wave_coeff + boundary_loss * (1.0 - plane_wave_coeff)


#: Chessell's band-average constants: averaging cos(2 pi f' tau + arg Q) uniformly over a
#: one-third-octave band [f 2^-1/6, f 2^1/6] gives cos(eta f tau + arg Q) sin(mu f tau)/(mu f tau),
#: with eta = pi (2^1/6 + 2^-1/6) (the band's arithmetic center) and mu = pi (2^1/6 - 2^-1/6)
#: (its width).  They are not a spreading or a reflection coefficient.
CHESSELL_MU = 0.727477
CHESSELL_ETA = 6.325159


def ega(hs, hr, d2, f, a, flores, pt=True, cturb=0.0, boundary_loss_correction=True):
    """
    Calculate excess ground attenuation for a non-directional point source.

    Based on procedures in:
    - Chien and Soroka, "Sound Propagation Along an Impedance Plane",
      J. Sound. Vib., 43(1), 9-20, 1975 (corrected 1980)
    - Delany, Bazley, "Acoustical Properties of Fibrous Absorbent Materials",
      Appl. Acoust., 3, 105-116, 1970.
    - Chessell, "Propagation of noise along an impedance boundary",
      JASA 62(4), 825-834, 1977
    - Daigle, G.A., T.F.W. Embelton, and J.E. Piercy. 1979. "Some Comments on
      the Literature of Propagation Near Boundaries of Finite Acoustical
      Impedance," J. Acoust. Soc. Am. 66(3), 918-919.
    - Stusnick, Plotkin, Sutherland, "Short-Range Acoustic Propagation Model",
      Wyle Research Report WR 85-19, July 1985.

    Assumes level terrain, no wind, straight acoustic rays.

    Args:
        hs: Source height (ft or m, must match other length units)
        hr: Receiver height (ft or m)
        d2: 2D distance between source and receiver (ft or m)
        f: Frequency (Hz), scalar or array
        a: Speed of sound (ft/s or m/s, matching length units)
        flores: Specific flow resistance (kPa·s/m²)
            Typical values (from AAM technical reference):
                - Snow cover: 30
                - Grassy field: 225
                - Roadside soil: 650
                - Packed sand: 1650
                - Hard packed dirt: 3000
                - Exposed rock: 6000
                - Concrete: 10000
                - Asphalt: 50000
                - Water: 1e6 (effectively rigid)
        pt: True for pure tone (no third octave smearing), False for broadband (default: True)
        cturb: Turbulence parameter (rad·s·(m or ft)^-0.5); it multiplies
               f·sqrt(range), so its value depends on the length unit.
               Typical: 0 to 16e-4 rad·s·m^-0.5, i.e. 0 to 8.8e-4 rad·s·ft^-0.5.
               Used by the broadband mode only (pt=False).
        boundary_loss_correction: When True (default), includes the boundary-loss factor
            correction to the plane-wave reflection coefficient (Chessell), which is
            most relevant at grazing incidence. When False, uses only the plane-wave
            reflection coefficient.

    Returns:
        atten: Attenuation in dB
        phase: Phase (radians) of resultant wave relative to direct path
               (Pure tones only; NaN for broadband case)

    Notes:
        - Output "attenuation" is an attenuation factor in dB
        - Add to the SPL of the direct wave to compute SPL at receiver
        - Phase offset is meaningful only for pure tones
    """
    # Ensure float arrays
    hs = np.asarray(hs, dtype=float)
    hr = np.asarray(hr, dtype=float)
    d2 = np.asarray(d2, dtype=float)
    f = np.asarray(f, dtype=float)
    a = np.asarray(a, dtype=float)
    flores = np.asarray(flores, dtype=float)
    cturb = np.asarray(cturb, dtype=float)

    # Calculate geometric values
    direct_range = np.sqrt(d2**2 + (hs - hr)**2)  # Direct acoustic path distance
    image_range = np.sqrt(d2**2 + (hs + hr)**2)  # Image source acoustic path distance
    # The image path's angle from the ground normal (cos = (hs + hr)/R2), i.e. 90 deg less the
    # grazing angle: the angle whose cosine spherical_reflection_coefficient's cos_grazing wants.
    incidence_angle = np.arccos((hs + hr) / image_range)
    path_delay = (image_range - direct_range) / a  # Time delay between direct and image paths
    range_ratio = image_range / direct_range  # Ratio of distances

    image_source_coeff = spherical_reflection_coefficient(
        np.cos(incidence_angle), image_range, f, a, flores, boundary_loss_correction)
    image_source_magnitude = np.abs(image_source_coeff)  # Magnitude of image source term
    image_source_phase = np.angle(image_source_coeff)  # Phase of image source term

    # Compute excess ground attenuation
    if pt:
        # Pure tone expression (Chessell's Equation 19)
        phase_delay = 1j * 2.0 * np.pi * f * path_delay  # Phase delay between paths
        # Direct + image amplitude
        resultant_amplitude = 1.0 + image_source_coeff * np.exp(phase_delay) / range_ratio
        phase = np.angle(resultant_amplitude)
        attn = np.abs(resultant_amplitude)**2
    else:
        # Broadband mode (Chessell's Equations 20, 21)
        freq_path_delay = f * path_delay  # Normalized frequency-path delay product
        normalized_image_mag = image_source_magnitude / range_ratio  # Normalized image magnitude
        normalized_image_mag_sq = normalized_image_mag ** 2

        # Add turbulence factor (Chessell Equations 25, 27)
        turbulence_factor = (np.exp(-(0.5 * cturb * f * np.sqrt(direct_range))**2)
                             if np.any(cturb > 0.0) else 1.0)

        # Combine attenuation computation
        # Interference phase at the band center
        ground_phase_term = CHESSELL_ETA * freq_path_delay + image_source_phase
        spherical_phase_term = CHESSELL_MU * freq_path_delay  # the band-width sinc's argument
        cosine_factor = np.cos(ground_phase_term) * turbulence_factor

        attn = 1.0 + normalized_image_mag_sq + 2.0 * normalized_image_mag * cosine_factor
        mask_positive_delay = path_delay > 0.0
        # np.where evaluates both branches, so sin(x)/x is also taken at the
        # zero delays (flush receivers) it then discards: silence that 0/0.
        with np.errstate(divide='ignore', invalid='ignore'):
            attn = np.where(
                mask_positive_delay,
                1.0 + normalized_image_mag_sq
                + 2.0 * normalized_image_mag * np.sin(spherical_phase_term)
                * cosine_factor / spherical_phase_term,
                attn
            )

        # Phase has no meaning for broadband, so set to NaN
        phase = np.full_like(attn, np.nan)

    # Convert magnitude to dB
    if not np.all(np.isfinite(attn)):
        raise ValueError('Error in EGA: non-finite attenuation magnitude encountered (check for f = 0)')
    if np.all(attn > 0.0):
        attenuation_db = 10.0 * np.log10(attn)
    else:
        raise ValueError('Error in EGA: non-positive attenuation magnitude encountered')

    return attenuation_db, phase
