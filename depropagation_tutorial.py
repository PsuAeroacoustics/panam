#!/usr/bin/env python
"""Depropagation tutorial: recover a known source directivity from a synthetic flyover.

The script runs PANAM's pipeline end to end without flight-test data:

1. synthesize a straight, level pass of a source with a known directivity
   over a line of microphones (synthetic_flyover.synthesize_pass);
2. depropagate it to a hemisphere at the reference radius
   (flight_acoustics.depropagate_hemisphere);
3. map the recovered overall level minus the true one on a Lambert
   equal-area projection;
4. write the hemisphere as a sphere file beside example_data/vehicle.cfg and
   build a toy NICE-OPS database from it (build_empirical_database).

Everything is written to the output directory (default demo_plots/, which is
git-ignored): tutorial_difference.png, tutorial_spheres/ and tutorial.nod.
"""
import argparse
import os
import shutil

import numpy as np

import flight_acoustics as fa
import synthetic_flyover as sf
from cli import colon_pair, save_or_show

HERE = os.path.dirname(os.path.abspath(__file__))
#: ft/s in one knot.
FT_PER_S_PER_KNOT = fa.KNOT_MPS / 0.3048


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('-o', '--output-dir', default='demo_plots',
                        help='directory for the figure, the sphere directory and the database (default demo_plots)')
    parser.add_argument('-p', '--plot', action='store_true', help='also open the figure in a window')
    parser.add_argument('--mics', type=int, default=7, help='number of microphones (default 7)')
    parser.add_argument('--mic-y', type=colon_pair, default=(-600.0, 600.0),
                        help='lateral extent of the microphone line "low:high", ft (default -600:600)')
    parser.add_argument('--speed', type=float, default=150.0, help='source speed, ft/s (default 150)')
    parser.add_argument('--height', type=float, default=300.0, help='source height, ft (default 300)')
    parser.add_argument('--duration', type=float, default=12.0,
                        help='record length, s; the source is overhead halfway through (default 12)')
    parser.add_argument('--seed', type=int, default=1, help='random seed of the synthetic noise (default 1)')
    parser.add_argument('--r-ref', type=float, default=100.0, help='reference radius, ft (default 100)')
    parser.add_argument('--limit', type=float, default=2.0,
                        help='color scale of the difference map, +-dB (default 2)')
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    sampling_rate, speed_of_sound = 4000.0, 1125.0
    mics = np.column_stack([np.zeros(args.mics), np.linspace(*args.mic_y, args.mics), np.zeros(args.mics)])
    scenario, ambient = sf.synthesize_pass(
        mics, speed=args.speed, height=args.height, duration=args.duration, sampling_rate=sampling_rate,
        speed_of_sound=speed_of_sound, directivity_db=sf.example_directivity_db, seed=args.seed)
    print('Synthesized {} microphones, {:.1f} s at {:.0f} Hz'.format(args.mics, args.duration, sampling_rate))

    # The overall level integrates to half the sample rate, where the
    # synthetic broadband noise ends, so it compares with source_level_db.
    hemisphere = fa.depropagate_hemisphere(
        **scenario, speed_of_sound=speed_of_sound, length_units='ft', r_ref=args.r_ref,
        freq_range=(0.0, sampling_rate / 2.0), window_time=0.25, window_overlap=0.5, point_stride=2,
        azi_step=15.0, elv_step=10.0, rmax=20.0, third_octave=True, third_octave_fmin=80.0,
        ambient_pressure=ambient)
    azimuth, elevation = np.meshgrid(hemisphere['azi_grid_deg'], hemisphere['elv_grid_deg'])
    true = sf.source_level_db(azimuth, elevation, args.r_ref, directivity_db=sf.example_directivity_db)
    difference = hemisphere['oaspl_db'] - true
    measured = np.isfinite(difference)
    print('Recovered minus true overall level over {} of {} grid cells: median {:+.2f} dB, '
          'largest magnitude {:.2f} dB'.format(measured.sum(), difference.size, np.median(difference[measured]),
                                              np.abs(difference[measured]).max()))

    fig = sf.plot_level_difference(hemisphere['azi_grid_deg'], hemisphere['elv_grid_deg'], difference,
                                   limit_db=args.limit)
    figure = os.path.join(args.output_dir, 'tutorial_difference.png')
    save_or_show(fig, figure, args.plot, bbox_inches='tight', facecolor='white')
    print('Wrote ' + figure)

    spheres = os.path.join(args.output_dir, 'tutorial_spheres')
    os.makedirs(spheres, exist_ok=True)
    shutil.copy(os.path.join(HERE, 'example_data', 'vehicle.cfg'), spheres)
    fa.write_aam_hemisphere_netcdf(os.path.join(spheres, 'X101.nc'), hemisphere, mode='third_octave',
                                   speed_knots=args.speed / FT_PER_S_PER_KNOT, flight_path_angle_deg=0.0,
                                   title='synthetic flyover', azimuth_reference='track')
    database = os.path.join(args.output_dir, 'tutorial.nod')
    fa.build_empirical_database(spheres, database, load_factors=None)
    print('Wrote {} and {}'.format(spheres, database))


if __name__ == '__main__':
    main()
