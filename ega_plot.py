#!/usr/bin/env python3
"""
Command-line tool to generate excess ground attenuation (EGA) contour plots.

Visualizes how sound attenuation varies with frequency, propagation distance,
source/receiver height, and ground characteristics.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

from flight_acoustics import ega


def parse_range(range_str):
    """Parse 'min:max' string into (min, max) tuple."""
    parts = range_str.split(':')
    if len(parts) != 2:
        raise ValueError(f'Range must be in format "min:max", got "{range_str}"')
    return float(parts[0]), float(parts[1])


def parse_range_or_single(value_str):
    """Parse 'min:max' string or a single value.
    
    Returns:
        tuple: (min, max) if range format, or (value, value) if single value
    """
    parts = value_str.split(':')
    if len(parts) == 2:
        return float(parts[0]), float(parts[1])
    elif len(parts) == 1:
        val = float(parts[0])
        return val, val
    else:
        raise ValueError(f'Invalid format: "{value_str}"')


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
  # Frequency vs distance for grass at 10m height difference
  %(prog)s -g grass -f 100:5000 -d 50:500

  # Distance vs height for asphalt over 1 kHz
  %(prog)s -g asphalt -t distance_height -f 1000 -d 10:300

  # Broadband attenuation (third-octave) over hard ground
  %(prog)s -g rock -b -f 100:5000 -d 50:500 -o ega_rock_broadband.pdf
        """
    )
    
    # Geometry and ground
    parser.add_argument('-g', '--ground', type=str, default='grass',
                        choices=['snow', 'grass', 'soil', 'sand', 'dirt', 'rock', 
                                'concrete', 'asphalt', 'water'],
                        help='Ground type (default: grass)')
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
    parser.add_argument('-f', '--frequency', type=str, default='100:5000',
                        help='Frequency range in Hz as "min:max" or single value (default: 100:5000)')
    parser.add_argument('-d', '--distance', type=str, default='50:500',
                        help='2D distance range in meters as "min:max" (default: 50:500)')
    
    # Mode and resolution
    parser.add_argument('-b', '--broadband', action='store_true',
                        help='Use broadband (third-octave) mode instead of pure tone')
    parser.add_argument('--turbulence', type=float, default=0.0,
                        help='Turbulence parameter (rad·s·√m) (default: 0.0)')
    parser.add_argument('--levels', type=int, default=50,
                        help='Number of contour levels (default: 50)')
    parser.add_argument('--resolution', type=int, default=100,
                        help='Grid resolution per axis (default: 100)')
    
    # Output
    parser.add_argument('-o', '--output', type=str, default='demo_plots/ega_plot.pdf',
                        help='Output filename (default: ega_plot.pdf)')
    
    args = parser.parse_args()
    
    # Parse ranges
    try:
        freq_parts = args.frequency.split(':')
        if len(freq_parts) == 2:
            freq_min, freq_max = map(float, freq_parts)
            frequencies = np.linspace(freq_min, freq_max, args.resolution)
        else:
            frequencies = np.array([float(args.frequency)])
    except ValueError as e:
        parser.error(f'Invalid frequency argument: {e}')
    
    dist_min, dist_max = parse_range(args.distance)
    distances = np.linspace(dist_min, dist_max, args.resolution)
    
    flores = get_ground_resistance(args.ground)
    hs = args.source_height
    hr = args.receiver_height
    pt = not args.broadband
    cturb = args.turbulence
    c = args.speed_of_sound
    
    # Create meshgrid and compute EGA
    if args.plot_type == 'frequency_distance':
        X, Y = np.meshgrid(frequencies, distances)
        Z, _ = ega(hs, hr, Y, X, c, flores, pt=pt, cturb=cturb)
        xlabel = 'Frequency (Hz)'
        ylabel = '2D Distance (m)'
        title_extra = f'{args.ground.capitalize()} (hs={hs}m, hr={hr}m)'
    elif args.plot_type == 'distance_height':
        heights_src = np.linspace(0.5, 50.0, args.resolution)
        X, Y = np.meshgrid(distances, heights_src)
        # Compute with varying source height, fixed receiver at 1.5m
        Z = np.zeros_like(X)
        f_single = frequencies[0] if len(frequencies) > 0 else 1000.0
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j], _ = ega(X[i, j], hr, Y[i, j], f_single, c, flores, pt=pt, cturb=cturb)
        xlabel = '2D Distance (m)'
        ylabel = 'Source Height (m)'
        title_extra = f'{args.ground.capitalize()} @ {f_single:.0f}Hz (hr={hr}m)'
    else:  # frequency plot
        X = frequencies
        Z = np.zeros_like(X)
        d_single = distances[0] if len(distances) > 0 else distances
        for i in range(X.shape[0]):
            Z[i], _ = ega(hs, hr, d_single, X[i], c, flores, pt=pt, cturb=cturb)
        xlabel = 'Frequency (Hz)'
        ylabel = 'Excess Attenuation (dB)'
        title_extra = f'{args.ground.capitalize()} (hs={hs}m, hr={hr}m, d={d_single}m)'

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if args.plot_type in ['frequency_distance', 'distance_height']:
        # Contour plot
        cf = ax.contourf(X, Y, Z, levels=args.levels, cmap='RdYlGn_r')
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label('Excess Attenuation (dB)')
    
        # Contour lines
        c = ax.contour(X, Y, Z, colors='k', linewidths=0.5, alpha=0.4)
        ax.clabel(c, fontsize=9, fmt='%.1f')
    else:
        ax.plot(X, Z, '-b')
        ax.grid(True)


    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    mode_str = 'Broadband (1/3 octave)' if args.broadband else 'Pure Tone'
    ax.set_title(f'Excess Ground Attenuation\n{mode_str} - {title_extra}')
    
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f'Plot saved to {args.output}')


if __name__ == '__main__':
    main()
