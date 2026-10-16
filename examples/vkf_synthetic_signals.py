#!/usr/bin/env python3
"""Vold-Kalman filter examples on synthetic signals.

Three cases show what :func:`panam.vold_kalman_filter.vold_kalman_filter` does:
a constant 100 Hz tone in noise with a 400 Hz interferer; a 50-150 Hz chirp
with a crossing chirp, filtered with a fixed bandwidth and with one
proportional to frequency; and three orders of a rotor whose speed varies,
with a crossing interferer.  It prints the extracted RMS and error of each and
writes a comparison plot and spectrograms (PNG) to --output-dir (default
demo_plots in the current directory).
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from panam import cli
from panam.flight_acoustics import plot_spectrogram
from panam.vold_kalman_filter import vold_kalman_filter


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Vold-Kalman filter examples on synthetic signals: a constant tone, a chirp '
                    'with a crossing interferer, and three orders at once. Writes a comparison '
                    'plot and spectrograms (PNG).')
    cli.add_output_options(parser, 'demo_plots', directory=True)
    args = parser.parse_args(argv)
    figures = {}
    fs = 5000  # Sampling frequency
    duration = 5  # seconds
    t = np.arange(0, duration, 1 / fs)

    print("="*70)
    print("Vold-Kalman Filter Examples")
    print("="*70)

    # Case 1: CONSTANT frequency (100 Hz)
    freq1_const = 100. + 0*t  # Hz (constant)
    x_clean_const = 0.8 * np.sin(2 * np.pi * freq1_const * t)

    # Add noise and interference
    np.random.seed(42)
    x_noisy_const = x_clean_const + 0.3 * np.random.randn(len(t))
    x_noisy_const += 0.4 * np.sin(2 * np.pi * 400 * t)

    # For CONSTANT frequency, fixed bandwidth works fine
    bandwidth_const = 20  # Hz
    p = 1  # Filter order

    y_const, phasor_const, _ = vold_kalman_filter(x_noisy_const, freq1_const, fs, bandwidth_const, p)
    x_filtered_const = np.real(y_const[:, 0] * phasor_const[:, 0])

    error_const = np.mean((x_filtered_const - x_clean_const)**2)
    print(f"\nCase 1 - Constant Frequency (100 Hz):")
    print(f"  Extracted signal: RMS = {np.sqrt(np.mean(x_filtered_const**2)):.3f} (target: {0.8 / np.sqrt(2):.3f})")
    print(f"  MSE = {error_const:.2e}")

    # ========================================================================

    # Case 2: TIME-VARYING frequency (50-150 Hz chirp over the record)
    freq1_chirp = 50. + 100*t/duration  # Hz (sweeps from 50 to 150)
    phase_chirp = 2 * np.pi * np.cumsum(freq1_chirp) / fs
    x_clean_chirp = 0.8 * np.sin(phase_chirp)

    # Add noise and crossing interference chirp (sweeps opposite direction: 150 to 50 Hz)
    np.random.seed(42)
    x_noisy_chirp = x_clean_chirp + 0.05 * np.random.randn(len(t))
    freq_interference_chirp = 150. - 100*t/duration  # Hz (sweeps from 150 to 50)
    phase_interference = 2 * np.pi * np.cumsum(freq_interference_chirp) / fs
    x_noisy_chirp += 0.3 * np.sin(phase_interference)

    # Track the main chirp alone with a fixed bandwidth (for comparison)
    bandwidth_fixed = 20  # Hz
    y_wrong, phasor_wrong, _ = vold_kalman_filter(x_noisy_chirp, freq1_chirp, fs, bandwidth_fixed, p)
    x_filtered_wrong = np.real(y_wrong[:, 0] * phasor_wrong[:, 0])

    # Track both the main chirp and the crossing interference with adaptive bandwidth
    freq_chirp_multi = np.zeros((len(t), 2))
    freq_chirp_multi[:, 0] = freq1_chirp  # Main chirp: 50-150 Hz
    freq_chirp_multi[:, 1] = freq_interference_chirp  # Crossing interference: 150-50 Hz

    bandwidth_adaptive = 0.10 * freq_chirp_multi  # 10% of instantaneous frequency

    y_correct, phasor_correct, _ = vold_kalman_filter(x_noisy_chirp, freq_chirp_multi, fs, bandwidth_adaptive, p)
    x_filtered_correct = np.real(y_correct[:, 0] * phasor_correct[:, 0])
    x_filtered_interference = np.real(y_correct[:, 1] * phasor_correct[:, 1])

    error_correct = np.mean((x_filtered_correct - x_clean_chirp)**2)
    print(f"\nCase 2 - Time-Varying Frequencies (50-150 Hz chirp + crossing interference):")
    print(f"  Main chirp extracted: RMS = {np.sqrt(np.mean(x_filtered_correct**2)):.3f} (target: {0.8 / np.sqrt(2):.3f}), MSE = {error_correct:.2e}")
    print(f"  Interference extracted: RMS = {np.sqrt(np.mean(x_filtered_interference**2)):.3f} (target: {0.3 / np.sqrt(2):.3f})")

    # ========================================================================

    # Case 3: Track multiple harmonic orders simultaneously
    # Simulate a rotor at varying RPM with multiple harmonics (1x, 2x, 3x)

    rotor_rpm = 1200 + 200 * np.sin(2 * np.pi * 0.5 * t)  # RPM varying between 1000-1400
    rotor_freq = rotor_rpm / 60  # Convert to Hz (base frequency)

    # Create freq array with shape (n_samples, n_orders) for multi-order tracking
    # Now tracking 4 orders: 1x, 2x, 3x, and the interference order
    freq_multi = np.zeros((len(t), 4))
    freq_multi[:, 0] = rotor_freq         # 1x order
    freq_multi[:, 1] = 2 * rotor_freq     # 2x order  
    freq_multi[:, 2] = 3 * rotor_freq     # 3x order
    freq_multi[:, 3] = 70 - 40 * (t / duration)  # Interference order (70-30 Hz)

    # Generate signal with three harmonic components
    phase_1x = 2 * np.pi * np.cumsum(freq_multi[:, 0]) / fs
    phase_2x = 2 * np.pi * np.cumsum(freq_multi[:, 1]) / fs
    phase_3x = 2 * np.pi * np.cumsum(freq_multi[:, 2]) / fs

    amp_1x = 1.0
    amp_2x = 0.5
    amp_3x = 0.3

    x_1x = amp_1x * np.sin(phase_1x)
    x_2x = amp_2x * np.sin(phase_2x)
    x_3x = amp_3x * np.sin(phase_3x)

    x_clean_multi = x_1x + x_2x + x_3x

    # Add noise and interference
    np.random.seed(42)
    x_noisy_multi = x_clean_multi + 0.2 * np.random.randn(len(t))
    # Add interference from a 4th order that crosses the tracked orders
    # This interference order sweeps from 70 Hz down to 30 Hz (opposite to main rotor)
    interference_freq = 70 - 40 * (t / duration)  # Hz, decreases over time
    phase_interference = 2 * np.pi * np.cumsum(interference_freq) / fs
    x_noisy_multi += 0.3 * np.sin(phase_interference)  # Interference order

    # Apply VKF with multi-order tracking
    # Use adaptive bandwidth as a percentage of each order's frequency
    bandwidth_multi = 0.15 * freq_multi  # 15% of each order's frequency

    y_multi, phasor_multi, _ = vold_kalman_filter(x_noisy_multi, freq_multi, fs, bandwidth_multi, p)

    # Extract each order
    x_filtered_1x = np.real(y_multi[:, 0] * phasor_multi[:, 0])
    x_filtered_2x = np.real(y_multi[:, 1] * phasor_multi[:, 1])
    x_filtered_3x = np.real(y_multi[:, 2] * phasor_multi[:, 2])
    x_filtered_interference_multi = np.real(y_multi[:, 3] * phasor_multi[:, 3])

    # Compute errors
    error_1x = np.mean((x_filtered_1x - x_1x)**2)
    error_2x = np.mean((x_filtered_2x - x_2x)**2)
    error_3x = np.mean((x_filtered_3x - x_3x)**2)

    print(f"\nCase 3 - Multi-Order Tracking (rotor harmonics + crossing interference):")
    print(f"  Rotor frequency: {rotor_freq.min():.1f} - {rotor_freq.max():.1f} Hz")
    print(f"  1x order: RMS = {np.sqrt(np.mean(x_filtered_1x**2)):.3f} (target: {amp_1x / np.sqrt(2):.3f}), MSE = {error_1x:.2e}")
    print(f"  2x order: RMS = {np.sqrt(np.mean(x_filtered_2x**2)):.3f} (target: {amp_2x / np.sqrt(2):.3f}), MSE = {error_2x:.2e}")
    print(f"  3x order: RMS = {np.sqrt(np.mean(x_filtered_3x**2)):.3f} (target: {amp_3x / np.sqrt(2):.3f}), MSE = {error_3x:.2e}")
    print(f"  Interference (70-30 Hz): RMS = {np.sqrt(np.mean(x_filtered_interference_multi**2)):.3f} (target: {0.3 / np.sqrt(2):.3f})")


    # ========================================================================
    # Plot comparison
    print("\n" + "="*70)
    print("Generating plots...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Case 1: Constant frequency
    ax = axes[0]
    ax.plot(t, x_clean_const, 'g-', label='Clean Signal (100 Hz constant)', linewidth=2, alpha=0.7)
    ax.plot(t, x_noisy_const, 'gray', label='Noisy Signal', linewidth=0.5, alpha=0.5)
    ax.plot(t, x_filtered_const, 'r-', label='Filtered (BW=20Hz)', linewidth=2, alpha=0.8)
    ax.set_ylabel('Amplitude')
    ax.set_title('Case 1: Constant Frequency - Works Well with Fixed Bandwidth')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration)

    # Case 2: Time-varying frequency
    ax = axes[1]
    ax.plot(t, x_clean_chirp, 'g-', label='Clean Signal (50-150 Hz chirp)', linewidth=2, alpha=0.7)
    ax.plot(t, x_noisy_chirp, 'gray', label='Noisy Signal', linewidth=0.5, alpha=0.5)
    ax.plot(t, x_filtered_wrong, 'r--', label='Fixed BW=20 Hz, main chirp tracked alone', linewidth=1.5, alpha=0.8)
    ax.plot(t, x_filtered_correct, 'b-', label='BW=10%*freq, both chirps tracked', linewidth=2, alpha=0.8)
    ax.set_ylabel('Amplitude')
    ax.set_title('Case 2: Time-Varying Frequency, Bandwidth Proportional to Frequency')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration)

    # Case 3: Multi-order tracking
    ax = axes[2]
    # Show first 1 second for clarity
    t_zoom = t[:int(1.0*fs)]
    ax.plot(t_zoom, x_noisy_multi[:len(t_zoom)], 'gray', label='Noisy Signal (1x+2x+3x+interference+noise)', linewidth=0.5, alpha=0.5)
    ax.plot(t_zoom, x_filtered_1x[:len(t_zoom)], 'r-', label=f'1x order (RMS={np.sqrt(np.mean(x_filtered_1x**2)):.2f})', linewidth=1.5, alpha=0.8)
    ax.plot(t_zoom, x_filtered_2x[:len(t_zoom)], 'b-', label=f'2x order (RMS={np.sqrt(np.mean(x_filtered_2x**2)):.2f})', linewidth=1.5, alpha=0.8)
    ax.plot(t_zoom, x_filtered_3x[:len(t_zoom)], 'g-', label=f'3x order (RMS={np.sqrt(np.mean(x_filtered_3x**2)):.2f})', linewidth=1.5, alpha=0.8)
    ax.plot(t_zoom, x_filtered_interference_multi[:len(t_zoom)], 'm--', label=f'Interference (RMS={np.sqrt(np.mean(x_filtered_interference_multi**2)):.2f})', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Case 3: Multi-Order Tracking - Extract Multiple Harmonics Simultaneously')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.0)

    plt.tight_layout()
    figures['vkf_example'] = fig

    # ========================================================================
    # Spectrogram plots
    # ========================================================================

    print("\nGenerating spectrograms...")

    # Each signal's figure name and title, all on one display scale.
    x_reconstructed_multi = x_filtered_1x + x_filtered_2x + x_filtered_3x
    spectrograms = [
        ('clean', x_clean_chirp, 'Spectrogram: Clean Chirp Signal (50-150 Hz)'),
        ('noisy', x_noisy_chirp, 'Spectrogram: Noisy Signal'),
        ('filtered', x_filtered_correct, 'Spectrogram: Filtered Signal (Adaptive BW=10%*freq)'),
        ('multiorder_clean', x_clean_multi, 'Clean Multi-Order Signal (1x+2x+3x)'),
        ('multiorder_noisy', x_noisy_multi, 'Noisy Multi-Order Signal (1x+2x+3x+noise+interference)'),
        ('multiorder_filtered', x_reconstructed_multi, 'Reconstructed Multi-Order Signal (filtered 1x+2x+3x)'),
        ('multiorder_1x', x_filtered_1x, 'Extracted: 1x Order'),
        ('multiorder_2x', x_filtered_2x, 'Extracted: 2x Order'),
        ('multiorder_3x', x_filtered_3x, 'Extracted: 3x Order'),
        ('multiorder_interference', x_filtered_interference_multi, 'Extracted: Interference Order (70-30 Hz)'),
    ]
    for name, signal, title in spectrograms:
        figures[f'vkf_spectrogram_{name}'] = plot_spectrogram(signal, fs, window_time=0.2, title=title,
                                                              clim=[20, 80], flim=[0, 400])[0]

    paths = cli.save_all_or_show(figures, args.output_dir, args.plot, dpi=150)
    if paths:
        print(f"\n{len(paths)} plots saved to {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
