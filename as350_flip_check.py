"""
Is the AS350 demo's `flip_y_for_geometry = True` real, or was it papering
over the Lambert projection bug?

The demo carries a reference AAM sphere for the same aircraft and
condition, and compares against it on a regular azimuth/elevation grid --
in ANGLE space, not page space -- so the comparison is independent of how
either sphere is drawn. That makes it a clean discriminator: build the
measured hemisphere both ways and see which one agrees with the
reference. A lateral sign error shows up as the measured sphere matching
the reference's MIRROR instead of the reference.

Result (47 acoustic files, 2026-08-21): the flip appeared GENUINE, matching
the reference to MAD 0.72 dB with it and its mirror without it.  It was in
fact compensating for :func:`flight_acoustics.art2umapr`, which until
2026-09-24 put ART phi = +90 to port.  The AAM Technical Reference (v3,
sec. 2.4.1) puts it to starboard, and the shipped AAM spheres agree: every
counter-clockwise rotor in them is louder at phi > 0 in descent, the
clockwise EC130 and Mi-8 at phi < 0.  With the sign corrected the same
residuals favor no flip, which is now the demo's setting.

Run:  python as350_flip_check.py [--basepath DIR]

The demo data are found from --basepath, then the older variable
``AS350_DEMO_PATH``, then the ``as350_demo`` entry of local_paths.  The check
prints a table and writes nothing.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import flight_acoustics as fa
import local_paths
import AS350_289108_demo as demo

REFERENCE_SPHERE = os.path.join("AAM", "AS350B3108.nc")   # relative to the demo data
FREQS = (0.0, 2000.0)
R_REF_FT = 100.0
AZI_STEP = ELV_STEP = 10.0
RMAX = 25.0


def demo_data_path(basepath=None):
    """The AS350 demo data directory: ``basepath``, else ``AS350_DEMO_PATH``,
    else the ``as350_demo`` entry of local_paths."""
    return local_paths.data_path("as350_demo", basepath or os.environ.get("AS350_DEMO_PATH"))


def _reference_on_grid(azi_grid, elv_grid, base):
    """Reference AAM sphere OASPL interpolated onto our grid (dB at r_ref);
    ``base`` is the demo data directory."""
    amp, phi, theta, f, radius_ft, _, _ = fa.load_nc_sphere(os.path.join(base, REFERENCE_SPHERE))
    amp = amp.astype(float)
    amp = fa.mask_missing_levels(amp)
    radius_ft = float(np.asarray(radius_ft, dtype=float).ravel()[0])
    amp = amp[:, :, np.logical_and(f >= FREQS[0], f <= FREQS[1])]
    oaspl = np.apply_along_axis(fa.OASPL, 2, amp)

    _, P = np.meshgrid(theta.astype(float), phi.astype(float))
    azi_r, elv_r = fa.art2umapr(np.deg2rad(P), np.deg2rad(theta.astype(float)))
    azi_d, elv_d = np.degrees(azi_r).ravel(), np.degrees(elv_r).ravel()
    pw = ((10.0 ** (oaspl / 10.0)) * (radius_ft / R_REF_FT) ** 2).ravel()
    # periodic in azimuth
    a = np.concatenate([azi_d, azi_d + 360.0, azi_d - 360.0])
    e = np.concatenate([elv_d, elv_d, elv_d])
    p = np.concatenate([pw, pw, pw])
    # NaN where the reference has no data within RMAX, -inf where it has no
    # energy: np.isfinite in main() then leaves both out of the statistics.
    g = fa.shepIDW(elv_grid, azi_grid, e, a, p, rmax=RMAX)
    out = fa.power_to_db(g)
    if out.shape[1] > 1:
        out[:, -1] = out[:, 0]
    return out


def measured(flip, base):
    """Measured hemisphere from the demo's own inputs, one flip setting;
    ``base`` is the demo data directory."""
    files = demo.gather_acoustic_files(os.path.join(base, "Acoustic"))
    miclocs, pressures, times = demo.load_microphones(files)
    miclocs, pressures, times = demo.filter_microphones(
        files, miclocs, pressures, times)
    _, ft = demo.load_track(base)
    src, vel = demo.track_kinematics(ft)
    atmosphere, speed_of_sound_ft_s = demo.demo_atmosphere()
    return fa.depropagate_hemisphere(
        mic_locations=miclocs,
        pressure=[np.asarray(p, dtype=float) * 0.5 for p in pressures],
        time=times, track_time=ft["time"], track_position=src,
        track_velocity=vel, speed_of_sound=speed_of_sound_ft_s, length_units="ft",
        r_ref=R_REF_FT, freq_range=FREQS, window_time=0.5,
        window_overlap=0.5, point_stride=1, azi_step=AZI_STEP,
        elv_step=ELV_STEP, rmax=RMAX, apply_absorption_deprop=True,
        atmosphere=atmosphere, flip_y_for_geometry=flip)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check the AS350 demo's lateral sign: depropagate the demo run with and "
                    "without flip_y_for_geometry and compare each with the reference AAM sphere "
                    "and its mirror (overall level, 0 to 2 kHz, at 100 ft). Prints a table; "
                    "writes nothing.")
    parser.add_argument("--basepath", default=None,
                        help="AS350 demo data directory, holding Acoustic/, AAM/ and Tracking/ "
                             "(default: AS350_DEMO_PATH, then the as350_demo entry of local_paths)")
    args = parser.parse_args(argv)
    base = demo_data_path(args.basepath)
    out = {}
    for flip in (False, True):
        h = measured(flip, base)
        azi, elv = h["azi_grid_deg"], h["elv_grid_deg"]
        AZ, EL = np.meshgrid(azi, elv)
        ref = _reference_on_grid(AZ, EL, base)
        # the same reference read backwards in azimuth: what a lateral sign
        # error would make the measurement look like
        ref_mirror = _reference_on_grid(np.remainder(-AZ, 360.0), EL, base)
        m = h["oaspl_db"]
        out[flip] = {}
        for name, r in (("reference", ref), ("mirrored reference", ref_mirror)):
            d = m - r
            ok = np.isfinite(d)
            out[flip][name] = (float(np.median(d[ok])),
                               float(np.median(np.abs(d[ok] - np.median(d[ok])))))
    print(f"{'flip_y_for_geometry':>22} | {'vs reference':>22} | "
          f"{'vs mirrored reference':>24}")
    print("-" * 74)
    for flip in (False, True):
        a, b = out[flip]["reference"], out[flip]["mirrored reference"]
        print(f"{str(flip):>22} | median {a[0]:+6.2f}  MAD {a[1]:5.2f} | "
              f"median {b[0]:+6.2f}  MAD {b[1]:5.2f}")
    print()
    # the verdict is which flip setting matches the REFERENCE (not its
    # mirror); matching the mirror is the failure mode, not a second answer
    want = min(out, key=lambda f: out[f]["reference"][1])
    a = out[want]["reference"][1]
    b = out[want]["mirrored reference"][1]
    print(f"VERDICT: flip_y_for_geometry={want} — matches the reference at "
          f"MAD {a:.2f} dB, its mirror only at {b:.2f} dB.")
    print("         " + ("the flip is GENUINE and must stay: this data's "
                         "lateral axis runs\n         opposite to the "
                         "convention hemigen assumes."
                         if want else
                         "no flip needed: the data already matches the "
                         "hemigen convention."))
    return out


if __name__ == "__main__":
    main()
