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

from flight_acoustics import ega


def nice_levels(vmin, vmax, max_levels=50):
    """
    Generate nice rounded contour levels between vmin and vmax.
    
    Args:
        vmin: Minimum value
        vmax: Maximum value
        max_levels: Maximum number of levels to generate
    
    Returns:
        numpy array of nicely rounded level values
    """
    range_val = vmax - vmin
    
    # Determine nice step sizes based on range
    if range_val <= 0:
        return np.array([vmin])
    
    # Calculate order of magnitude
    magnitude = 10 ** np.floor(np.log10(range_val))
    
    # Try nice fractions of the magnitude
    nice_steps = [0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0]
    
    for base_step in nice_steps:
        step = base_step * magnitude
        n_steps = int(np.ceil(range_val / step))
        if n_steps <= max_levels:
            # Round start to nearest step
            start = np.floor(vmin / step) * step
            end = np.ceil(vmax / step) * step
            levels = np.arange(start, end + step/2, step)
            # Filter to actual range
            levels = levels[(levels >= vmin - step/10) & (levels <= vmax + step/10)]
            return levels
    
    # Fallback to linear spacing if no nice step found
    return np.linspace(vmin, vmax, max_levels)


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
    parser.add_argument('-f', '--frequency', type=str, default='100:5000',
                        help='Frequency range in Hz as "min:max" or single value (default: 100:5000)')
    parser.add_argument('-d', '--distance', type=str, default='50:500',
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
    parser.add_argument('--clim', type=str, default=None,
                        help='Colorbar range as "min:max" in dB (default: automatic)')
    
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
    
    dist_min, dist_max = parse_range_or_single(args.distance)
    distances = np.linspace(dist_min, dist_max, args.resolution)
    # If single distance value, keep it as-is
    is_single_distance = (dist_min == dist_max)
    
    flores = get_ground_resistance(args.ground or 'grass') if args.resistivity is None else args.resistivity
    ground_label = (args.ground or 'grass').capitalize() if args.resistivity is None else f'Resistivity {flores:.0f}'
    hs = args.source_height
    hr = args.receiver_height
    pt = not args.broadband
    cturb = args.turbulence
    c = args.speed_of_sound
    boundary_loss_correction = args.boundary_loss_correction
    
    # Parse colorbar limits if provided
    if args.clim:
        try:
            clim_min, clim_max = parse_range(args.clim)
        except ValueError as e:
            parser.error(f'Invalid --clim argument: {e}')
    else:
        clim_min, clim_max = None, None
    
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
        # Compute with varying source height, fixed receiver at 1.5m
        Z = np.zeros_like(X)
        f_single = frequencies[0] if len(frequencies) > 0 else 1000.0
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j], _ = ega(
                    Y[i, j],    # source height
                    hr,
                    X[i, j],    # 2D distance
                    f_single,
                    c,
                    flores,
                    pt=pt,
                    cturb=cturb,
                    boundary_loss_correction=boundary_loss_correction,
                )
        xlabel = '2D Distance (m)'
        ylabel = 'Source Height (m)'
        title_extra = f'{ground_label} @ {f_single:.0f}Hz (hr={hr}m)'
    else:  # frequency plot
        X = frequencies
        Z = np.zeros_like(X)
        d_single = distances[0] if not is_single_distance else dist_min
        for i in range(X.shape[0]):
            Z[i], _ = ega(
                hs,
                hr,
                d_single,
                X[i],
                c,
                flores,
                pt=pt,
                cturb=cturb,
                boundary_loss_correction=boundary_loss_correction,
            )
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
            contour_kwargs['levels'] = nice_levels(clim_min, clim_max, args.levels)
            contour_kwargs['extend'] = 'both'  # Show arrows for out-of-range values
        else:
            # Generate nice levels from data range
            data_min, data_max = np.min(Z), np.max(Z)
            contour_kwargs['levels'] = nice_levels(data_min, data_max, args.levels)
        
        cf = ax.contourf(X, Y, Z, **contour_kwargs)
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
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f'Plot saved to {args.output}')


if __name__ == '__main__':
    main()
