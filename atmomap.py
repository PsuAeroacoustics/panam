import numpy as np
import matplotlib.pyplot as plt

from flight_acoustics import atmosorb


def main():
    # Parameters (mirroring the MATLAB script)
    pressures_atm = np.linspace(1.0, 30.0, 1000)  # atm
    pstat_mbar = pressures_atm * 1013.25  # mbar
    humidity = 10.0  # % RH (percent, not fraction)
    temperatures_F = np.linspace(50.0, 100.0, 100)  # deg F

    freq_hz = 10e3  # Hz

    TT, PP = np.meshgrid(temperatures_F, pstat_mbar)

    dBpft = atmosorb(freq_hz, TT, humidity, PP)

    # Build filled contours and labeled contour lines (similar to MATLAB's contourf + clabel)
    fig, ax = plt.subplots(figsize=(7, 5))

    # Convert pressure to atm for y-axis, as in MATLAB (PP./1013.25)
    y_atm = PP / 1013.25

    # Filled contours
    cf = ax.contourf(TT, y_atm, dBpft, levels=50, cmap='viridis')
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label('Absorption (dB/ft)')

    # Overlay contour lines in black, thicker lines, and label them
    c = ax.contour(TT, y_atm, dBpft, colors='k', linewidths=2)
    ax.clabel(c, fontsize=14)

    ax.set_ylim(0.0, 30.0)

    ax.set_xlabel('Temperature, °F')
    ax.set_ylabel('Pressure, atm')

    title_line1 = f"Absorption at {freq_hz/1000.0:g} kHz dB/ft"
    title_line2 = f"for {humidity:g} % relative humidity"
    ax.set_title(f"{title_line1}\n{title_line2}")

    fig.tight_layout()
    fig.savefig('atmomap.pdf', bbox_inches='tight')
    print('Saved atmomap.pdf')


if __name__ == '__main__':
    main()
