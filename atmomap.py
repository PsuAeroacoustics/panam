#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt

from flight_acoustics import atmosorb


def main():
    parser = argparse.ArgumentParser(description='Generate atmospheric absorption map')
    parser.add_argument('-t', '--temperature', type=str, default='0:40',
                        help='Temperature range in °C as "min:max" (default: 0:40)')
    parser.add_argument('-p', '--pressure', type=str, default='1:30',
                        help='Pressure range in atm as "min:max" (default: 1:30)')
    parser.add_argument('-f', '--frequency', type=float, default=10000.0,
                        help='Frequency in Hz (default: 10000.0)')
    parser.add_argument('-H', '--humidity', type=float, default=10.0,
                        help='Relative humidity in percent (default: 10.0)')
    parser.add_argument('-o', '--output', type=str, default='atmomap.pdf',
                        help='Output filename (default: atmomap.pdf)')
    
    args = parser.parse_args()
    
    # Parse temperature range
    temp_values = args.temperature.split(':')
    if len(temp_values) != 2:
        raise ValueError('Temperature range must be in format "min:max"')
    temp_min, temp_max = map(float, temp_values)
    
    # Parse pressure range
    press_values = args.pressure.split(':')
    if len(press_values) != 2:
        raise ValueError('Pressure range must be in format "min:max"')
    press_min, press_max = map(float, press_values)
    
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

    #ax.set_ylim(0.0, 30.0)

    ax.set_xlabel('Temperature, °C')
    ax.set_ylabel('Pressure, atm')

    title_line1 = f"Absorption at {freq_hz/1000.0:g} kHz dB/m"
    title_line2 = f"for {humidity:g} % relative humidity"
    ax.set_title(f"{title_line1}\n{title_line2}")

    fig.tight_layout()
    fig.savefig(args.output, bbox_inches='tight')

if __name__ == '__main__':
    main()
