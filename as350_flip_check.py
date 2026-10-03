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
residuals favour no flip, which is now the demo's setting.

Run:  PYTHONPATH=. python as350_flip_check.py

The demo data are found through the ``as350_demo`` entry of local_paths
(``AS350_DEMO_PATH`` is still honored and takes precedence).
"""

from __future__ import annotations

import os

import numpy as np

import flight_acoustics as fa
import local_paths
import AS350_289108_demo as demo

BASE = local_paths.data_path("as350_demo", os.environ.get("AS350_DEMO_PATH"))
REF = os.path.join(BASE, "AAM", "AS350B3108.nc")
FREQS = (0.0, 2000.0)
R_REF_FT = 100.0
AZI_STEP = ELV_STEP = 10.0
RMAX = 25.0


def _reference_on_grid(azi_grid, elv_grid):
    """Reference AAM sphere OASPL interpolated onto our grid (dB at r_ref)."""
    amp, phi, theta, f, radius_ft, _, _ = fa.load_nc_sphere(REF)
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


def measured(flip):
    """Measured hemisphere from the demo's own inputs, one flip setting."""
    files = demo.gather_acoustic_files(os.path.join(BASE, "Acoustic"))
    miclocs, pressures, times = demo.load_microphones(files)
    miclocs, pressures, times = demo.filter_microphones(
        files, miclocs, pressures, times)
    track = fa.load_NASA_track(os.path.join(BASE, "Tracking", "289108AC.csv"))
    ft = fa.filter_track(track, xlims=(-4000, 0), zlims=(50, 1500),
                         decimate=10)
    src = np.array([ft["x"], ft["y"], ft["z"]]).T
    vel = np.array([ft["vx"], ft["vy"], ft["vz"]]).T
    return fa.depropagate_hemisphere(
        mic_locations=miclocs,
        pressure=[np.asarray(p, dtype=float) * 0.5 for p in pressures],
        time=times, track_time=ft["time"], track_position=src,
        track_velocity=vel, speed_of_sound=1135.0, length_units="ft",
        r_ref=R_REF_FT, freq_range=FREQS, window_time=0.5,
        window_overlap=0.5, point_stride=1, azi_step=AZI_STEP,
        elv_step=ELV_STEP, rmax=RMAX, apply_absorption_deprop=True,
        atmosphere=fa.Atmosphere(temperature=293.15, pressure=101.325,
                                 relative_humidity=20.0),
        flip_y_for_geometry=flip)


def main():
    out = {}
    for flip in (False, True):
        h = measured(flip)
        azi, elv = h["azi_grid_deg"], h["elv_grid_deg"]
        AZ, EL = np.meshgrid(azi, elv)
        ref = _reference_on_grid(AZ, EL)
        # the same reference read backwards in azimuth: what a lateral sign
        # error would make the measurement look like
        ref_mirror = _reference_on_grid(np.remainder(-AZ, 360.0), EL)
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
