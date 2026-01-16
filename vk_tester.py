import numpy as np
from vold_kalman_filter import vold_kalman_filter

fs = 1000
t = np.arange(0, 1, 1/fs)
signal = np.sin(2*np.pi*100*t)
freq_vec = 100 * np.ones_like(signal)
p = [2]

# Get the complex envelope before 2× scaling
y, phasor, cost = vold_kalman_filter(signal, freq_vec, fs, 5, p)

# Divide by 2 to get y_R (before the 2× scaling)
y_R = y / 2.0

print("Python y_R (first 5):")
for i in range(5):
    print(f"y_R[{i}] = {y_R[i,0]:.5f}")
print(f"y_R max = {np.max(np.abs(y_R[:,0])):.5f}")

print("\nMATLAB y_R (first 5):")
print("y_R(1:5): -0.288+0.40451i  -0.288+0.40451i  -0.28801+0.40451i  -0.28802+0.40451i  -0.28802+0.40451i")
print("y_R max: 0.50348")