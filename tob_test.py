#!/usr/bin/env python3
import flight_acoustics as fa
import numpy as np
import matplotlib.pyplot as plt

#Generate white noise signal
fs = 40000  # Sampling frequency
duration = 20.0  # seconds
t = np.arange(0, duration, 1/fs)
white_noise = np.random.normal(0, 1, len(t))
band_centers, band_levels = fa.third_octave_band_levels(white_noise, fs)
plt.semilogx(band_centers, band_levels)
plt.title('Third-Octave Band Levels of White Noise')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Level (dB)')
plt.grid(True)
plt.show()