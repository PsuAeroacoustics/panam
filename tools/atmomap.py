#!/usr/bin/env python
"""Map the atmospheric absorption at one frequency over temperature and pressure."""
import argparse
import numpy as np
import matplotlib.pyplot as plt

from panam.cli import add_output_options, colon_pair, save_or_show
from panam.absorption import atmosorb


def main():
    parser = argparse.ArgumentParser(
        description='Map the atmospheric absorption (dB/m, ISO 9613-1) at one frequency over '
                    'temperature and pressure. Always writes a file (default demo_plots/atmomap.pdf '
                    'in the current directory).')
    parser.add_argument('-t', '--temperature', type=colon_pair, default='0:40',
                        help='Temperature range in °C as "min:max" (default: 0:40)')
    parser.add_argument('-p', '--pressure', type=colon_pair, default='1:30',
                        help='Pressure range in atm as "min:max" (default: 1:30)')
    parser.add_argument('-f', '--frequency', type=float, default=10000.0,
                        help='Frequency in Hz (default: 10000.0)')
    parser.add_argument('-H', '--humidity', type=float, default=10.0,
                        help='Relative humidity in percent (default: 10.0)')
    # -p is --pressure here, so the window option is --plot alone.
    add_output_options(parser, 'demo_plots/atmomap.pdf', plot_flags=('--plot',))
    
    args = parser.parse_args()
    
    temp_min, temp_max = args.temperature
    press_min, press_max = args.pressure
    
    # Parameters
    pressures_atm = np.linspace(press_min, press_max, 1000)  # atm
    pstat_mbar = pressures_atm * 1013.25  # mbar
    humidity = args.humidity  # % RH (percent, not fraction)
    temperatures_C = np.linspace(temp_min, temp_max, 100)  # deg C

    freq_hz = args.frequency  # Hz

    TT, PP = np.meshgrid(temperatures_C, pstat_mbar)

    dBpm = atmosorb(freq_hz, TT, humidity, PP)

    # Build filled contours and labeled contour lines (similar to MATLAB's contourf + clabel)
    fig, ax = plt.subplots(figsize=(7, 5))

    # Convert pressure to atm for y-axis, as in MATLAB (PP./1013.25)
    y_atm = PP / 1013.25

    # Filled contours
    cf = ax.contourf(TT, y_atm, dBpm, levels=50, cmap='viridis')
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label('Absorption (dB/m)')

    # Overlay contour lines in black, thicker lines, and label them
    c = ax.contour(TT, y_atm, dBpm, colors='k', linewidths=2)
    ax.clabel(c, fontsize=14)

    ax.set_xlabel('Temperature, °C')
    ax.set_ylabel('Pressure, atm')

    title_line1 = f"Absorption at {freq_hz/1000.0:g} kHz dB/m"
    title_line2 = f"for {humidity:g} % relative humidity"
    ax.set_title(f"{title_line1}\n{title_line2}")

    fig.tight_layout()
    save_or_show(fig, args.output, args.plot, bbox_inches='tight')

if __name__ == '__main__':
    main()
