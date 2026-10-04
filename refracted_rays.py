"""Refracted-ray geometry for depropagation (:func:`flight_acoustics.depropagate_hemisphere`).

Depropagation along straight lines in uniform air files each sample at the straight
line's depression angle and spreads it over the straight-line distance.  Through a
stratified atmosphere the sound left the source at the ray's launch angle, spread
over the ray tube, traveled the arc and met the microphone at the ray's arrival
angle.  ``depropagate_hemisphere(ray_model=...)`` takes those from a *ray model*:

    ray_model(source_positions, mic_locations) -> dict of arrays shaped (Npts, Nmics)

with ``source_positions`` (Npts, 3) and ``mic_locations`` (Nmics, 3) in the
depropagation's own frame and length units (before any ``flip_y_for_geometry``),
z up.  The dict holds:

    depression_deg   where the sample is filed: degrees below the horizontal at
                     the source (the straight line's elevation in hemigen's sense)
    travel_time      seconds from emission to reception
    spreading_range  the distance spherical spreading is equivalent to (the ray
                     tube's), length units: replaces r in (r / r_ref)^2
    path_length      arc length, length units, for absorption
    offset           (Npts, Nmics, 3): the source offset the receiver response
                     sees -- the equivalent straight geometry whose elevation is
                     the ray's arrival angle at the microphone
    valid            bool: False where no ray reaches the microphone (a shadow),
                     which depropagation then skips

The azimuth is left as hemigen gives it: a ray in a stratified atmosphere stays in
the vertical plane of its endpoints.

:func:`straight_ray_model` gives the straight-line answer in this form, and
:func:`external_ray_model` runs NICE-OPS's ``niceops_ray_geometry`` on a profile
file.  Any other ray tracer can be wrapped the same way.
"""
import csv
import os
import subprocess
import tempfile

import numpy as np

__all__ = ['straight_ray_model', 'external_ray_model']


def straight_ray_model(speed_of_sound):
    """Straight lines in uniform air: what depropagation does without a ray model."""
    def model(source_positions, mic_locations):
        source = np.asarray(source_positions, dtype=float)[:, None, :]
        mic = np.asarray(mic_locations, dtype=float)[None, :, :]
        offset = source - mic
        ground = np.hypot(offset[..., 0], offset[..., 1])
        r = np.sqrt(ground ** 2 + offset[..., 2] ** 2)
        return dict(depression_deg=np.degrees(np.arctan2(offset[..., 2], ground)),
                    travel_time=r / float(speed_of_sound), spreading_range=r, path_length=r,
                    offset=offset, valid=np.ones(r.shape, dtype=bool))
    model.description = 'straight'
    return model


def external_ray_model(executable, atmosphere_path, *, frame_bearing_deg=90.0, refraction='stratified',
                       receiver='plate', receiver_heights=None, length_units='ft', threads=0, cwd=None):
    """A ray model that runs NICE-OPS's ``niceops_ray_geometry`` through a profile.

    ``atmosphere_path`` is a NICE-OPS ``--atmosphere`` CSV (heights in
    ``length_units``); ``frame_bearing_deg`` the compass bearing of the frame's +x
    axis, which places the profile's winds.  ``refraction`` is ``'stratified'``
    (the profile's own layers) or ``'linear'``.  ``receiver`` sets which
    equivalent geometry the offset carries: ``'plate'``, the direct ray's arrival
    (a ground board), or ``'point'``, the ground-reflected ray's (an elevated
    microphone).  ``receiver_heights`` (Nmics,) are the microphones' heights above
    the ground (default 0, flush); the sources' heights are measured from the
    same ground under each microphone.  Lengths are in ``length_units`` ('ft' or 'm').
    """
    if length_units not in ('ft', 'm'):
        raise ValueError("length_units must be 'ft' or 'm'")
    if receiver not in ('plate', 'point'):
        raise ValueError("receiver must be 'plate' or 'point'")

    def model(source_positions, mic_locations):
        source = np.asarray(source_positions, dtype=float)
        mic = np.asarray(mic_locations, dtype=float)
        npts, nmics = source.shape[0], mic.shape[0]
        heights = np.zeros(nmics) if receiver_heights is None else np.asarray(receiver_heights, dtype=float)
        offset = source[:, None, :] - mic[None, :, :]
        with tempfile.TemporaryDirectory() as workdir:
            paths = os.path.join(workdir, 'paths.csv')
            with open(paths, 'w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['id', 'dx', 'dy', 'source_height', 'receiver_height'])
                k = 0
                for i in range(npts):
                    for j in range(nmics):
                        # '%.17g': repr() of a numpy float is "np.float64(...)" under numpy 2.
                        writer.writerow([k] + ['%.17g' % v for v in (offset[i, j, 0], offset[i, j, 1],
                                                                       offset[i, j, 2] + heights[j], heights[j])])
                        k += 1
            cmd = [executable, '--atmosphere', os.path.abspath(atmosphere_path), '--paths', paths,
                   '--frame_bearing', '%.17g' % float(frame_bearing_deg), '--refraction', refraction,
                   '--ground', receiver, '--units', length_units]
            if threads:
                cmd += ['-j', str(int(threads))]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or workdir)
        if result.returncode != 0:
            raise RuntimeError('{} failed ({}): {}'.format(os.path.basename(executable), result.returncode,
                                                           result.stderr[-500:]))
        table = np.genfromtxt(result.stdout.splitlines(), delimiter=',', names=True)
        table = np.atleast_1d(table)
        if table.size != npts * nmics:
            raise RuntimeError('ray model returned {} rows for {} pairs'.format(table.size, npts * nmics))
        order = np.argsort(table['id'])
        table = table[order]

        def grid(name):
            return np.asarray(table[name], dtype=float).reshape(npts, nmics)

        # The equivalent geometry keeps the pair's azimuth: scale the horizontal
        # offset to the equivalent ground distance.
        ground = np.hypot(offset[..., 0], offset[..., 1])
        scale = np.divide(grid('ground_distance'), ground, out=np.ones_like(ground), where=ground > 0.0)
        equivalent = np.stack((offset[..., 0] * scale, offset[..., 1] * scale,
                               grid('source_height') - (heights[None, :] if receiver == 'point' else 0.0)),
                              axis=-1)
        launch = grid('launch_deg')
        out = dict(depression_deg=-launch, travel_time=grid('travel_time'),
                   spreading_range=grid('range'), path_length=grid('path_length'), offset=equivalent,
                   valid=(grid('lit') == 1) & np.isfinite(launch) & (grid('range') > 0.0))
        return out
    model.description = 'niceops_ray_geometry {} {} {}'.format(refraction, receiver, os.path.basename(atmosphere_path))
    return model
