#!/usr/bin/env python3
"""
Command-line tool to generate excess ground attenuation (EGA) contour plots.

Visualizes how sound attenuation varies with frequency, propagation distance,
source/receiver height, and ground characteristics.
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from cli import colon_pair, colon_pair_or_single
from flight_acoustics import ega, nice_levels


def get_ground_resistance(ground_type):
    """
    Return specific flow resistance for common ground types (kPa·s/m²).
    
    Based on AAM technical references.
    """
    grounds = {
        'snow': 30,
        'grass': 225,
        'soil': 650,
        'sand': 1650,
        'dirt': 3000,
        'rock': 6000,
        'concrete': 10000,
        'asphalt': 50000,
        'water': 1e6,
    }
    if ground_type.lower() not in grounds:
        valid = ', '.join(grounds.keys())
        raise ValueError(f'Unknown ground type: {ground_type}. Valid: {valid}')
    return grounds[ground_type.lower()]


def main():
    parser = argparse.ArgumentParser(
        description='Generate excess ground attenuation (EGA) contour plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Frequency vs distance for grass, source at 10 m, receiver at 1.5 m
  %(prog)s -g grass -f 100:5000 -d 50:500

  # Distance vs height for asphalt over 1 kHz
  %(prog)s -g asphalt -p distance_height -f 1000 -d 10:300

  # Broadband attenuation (third-octave) over hard ground
  %(prog)s -g rock -b -f 100:5000 -d 50:500 -o ega_rock_broadband.pdf
        """
    )
    
    # Geometry and ground
    parser.add_argument('-g', '--ground', type=str, default=None,
                        choices=['snow', 'grass', 'soil', 'sand', 'dirt', 'rock', 
                                'concrete', 'asphalt', 'water'],
                        help='Ground type (default: grass, or use --resistivity for custom value)')
    parser.add_argument('--resistivity', type=float, default=None,
                        help='Specific flow resistivity in kPa·s/m² (overrides --ground if provided)')
    parser.add_argument('-s', '--source-height', type=float, default=10.0,
                        help='Source height in meters (default: 10)')
    parser.add_argument('-r', '--receiver-height', type=float, default=1.5,
                        help='Receiver height in meters (default: 1.5)')
    parser.add_argument('-c', '--speed-of-sound', type=float, default=343.0,
                        help='Speed of sound in m/s (default: 343.0)')
    # Plot configuration
    parser.add_argument('-p', '--plot-type', type=str, default='frequency_distance',
                        choices=['frequency_distance', 'distance_height','frequency'],
                        help='Plot type (default: frequency_distance)')
    parser.add_argument('-f', '--frequency', type=colon_pair_or_single, default='100:5000',
                        help='Frequency range in Hz as "min:max" or single value (default: 100:5000)')
    parser.add_argument('-d', '--distance', type=colon_pair_or_single, default='50:500',
                        help='2D distance in meters as "min:max" or single value (default: 50:500)')
    
    # Mode and resolution
    parser.add_argument('-b', '--broadband', action='store_true',
                        help='Use broadband (third-octave) mode instead of pure tone')
    parser.add_argument(
        '--no-boundary-loss-correction',
        dest='boundary_loss_correction',
        action='store_false',
        help='Disable the grazing-incidence boundary-loss correction; use plane-wave reflection coefficient only',
    )
    parser.add_argument('--turbulence', type=float, default=0.0,
                        help='Turbulence parameter, rad·s·m^-0.5; broadband mode only (default: 0.0)')
    parser.add_argument('--levels', type=int, default=50,
                        help='Number of contour levels (default: 50)')
    parser.add_argument('--resolution', type=int, default=100,
                        help='Grid resolution per axis (default: 100)')
    parser.add_argument('--clim', type=colon_pair, default=None,
                        help='Colorbar range as "min:max" in dB (default: automatic)')
    
    # Output
    parser.add_argument('-o', '--output', type=str, default='demo_plots/ega_plot.pdf',
                        help='Output filename (default: demo_plots/ega_plot.pdf)')
    
    args = parser.parse_args()
    
    freq_min, freq_max = args.frequency
    frequencies = (np.linspace(freq_min, freq_max, args.resolution) if freq_max != freq_min
                   else np.array([freq_min]))
    dist_min, dist_max = args.distance
    distances = np.linspace(dist_min, dist_max, args.resolution)
    if args.plot_type == 'frequency_distance' and (frequencies.size < 2 or dist_min == dist_max):
        parser.error('a frequency_distance plot needs "min:max" ranges for both --frequency and --distance')
    
    flores = get_ground_resistance(args.ground or 'grass') if args.resistivity is None else args.resistivity
    ground_label = (args.ground or 'grass').capitalize() if args.resistivity is None else f'Resistivity {flores:.0f}'
    hs = args.source_height
    hr = args.receiver_height
    pt = not args.broadband
    cturb = args.turbulence
    c = args.speed_of_sound
    boundary_loss_correction = args.boundary_loss_correction
    
    clim_min, clim_max = args.clim if args.clim else (None, None)
    
    # Create meshgrid and compute EGA
    if args.plot_type == 'frequency_distance':
        X, Y = np.meshgrid(frequencies, distances)
        Z, _ = ega(hs, hr, Y, X, c, flores, pt=pt, cturb=cturb, boundary_loss_correction=boundary_loss_correction)
        xlabel = 'Frequency (Hz)'
        ylabel = '2D Distance (m)'
        title_extra = f'{ground_label} (hs={hs}m, hr={hr}m)'
    elif args.plot_type == 'distance_height':
        heights_src = np.linspace(0.5, 50.0, args.resolution)
        X, Y = np.meshgrid(distances, heights_src)
        # Varying source height, fixed receiver
        f_single = frequencies[0]
        Z, _ = ega(Y, hr, X, f_single, c, flores, pt=pt, cturb=cturb,
                   boundary_loss_correction=boundary_loss_correction)
        xlabel = '2D Distance (m)'
        ylabel = 'Source Height (m)'
        title_extra = f'{ground_label} @ {f_single:.0f}Hz (hr={hr}m)'
    else:  # frequency plot
        X = frequencies
        d_single = dist_min
        Z, _ = ega(hs, hr, d_single, X, c, flores, pt=pt, cturb=cturb,
                   boundary_loss_correction=boundary_loss_correction)
        xlabel = 'Frequency (Hz)'
        ylabel = 'Excess Attenuation (dB)'
        title_extra = f'{ground_label} (hs={hs}m, hr={hr}m, d={d_single}m)'

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if args.plot_type in ['frequency_distance', 'distance_height']:
        # Contour plot with optional colorbar limits
        contour_kwargs = {'cmap': 'RdYlGn_r'}
        if clim_min is not None and clim_max is not None:
            # Generate nice rounded levels when limits are specified
            contour_kwargs['levels'] = nice_levels(clim_min, clim_max, target=args.levels)
            contour_kwargs['extend'] = 'both'  # Show arrows for out-of-range values
        else:
            # Generate nice levels from data range
            data_min, data_max = np.min(Z), np.max(Z)
            contour_kwargs['levels'] = nice_levels(data_min, data_max, target=args.levels)
        
        cf = ax.contourf(X, Y, Z, **contour_kwargs)
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label('Excess Attenuation (dB)')
    
        # Contour lines
        lines = ax.contour(X, Y, Z, colors='k', linewidths=0.5, alpha=0.4)
        ax.clabel(lines, fontsize=9, fmt='%.1f')
    else:
        ax.plot(X, Z, '-b')
        ax.grid(True)


    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    mode_str = 'Broadband (1/3 octave)' if args.broadband else 'Pure Tone'
    ax.set_title(f'Excess Ground Attenuation\n{mode_str} - {title_extra}')
    
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f'Plot saved to {args.output}')


if __name__ == '__main__':
    main()
