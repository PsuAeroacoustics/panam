#!/usr/bin/env python3
"""
Hexacopter Acoustic Signal Separation using Vold-Kalman Filter

This script applies the Vold-Kalman filter to separate harmonic components
of acoustic data from a reconfigurable hexacopter with 6 rotors.

Based on MATLAB script: ssp_hexacopterExample/exampleHexacopter.m
Author: Joel Rachaprolu (original MATLAB)
Python port: GitHub Copilot
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import os
import time
import concurrent.futures as cf
from vold_kalman_filter import vold_kalman_filter


def highpass(data, cutoff_freq, fs, order=5):
    """
    Apply highpass Butterworth filter to data.
    
    Parameters
    ----------
    data : ndarray
        Input signal(s), shape (n_samples,) or (n_samples, n_channels)
    cutoff_freq : float
        Cutoff frequency in Hz
    fs : float
        Sampling frequency in Hz
    order : int, optional
        Filter order, default 5
    
    Returns
    -------
    ndarray
        Filtered signal with same shape as input
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    
    if data.ndim == 1:
        return filtfilt(b, a, data)
    else:
        # Apply to each column
        return np.apply_along_axis(lambda x: filtfilt(b, a, x), 0, data)


def _process_segment(
    seg_idx,
    x_src,
    t,
    rpm,
    orders,
    no_blade,
    fs,
    bw_percent,
    min_bw,
    max_bw,
    p,
    solver,
    use_coupling,
    cutoff,
):
    """Process a single segment for parallel execution."""
    x = x_src.copy()
    seg_size = len(x) - 2 * cutoff
    n_mics = x.shape[1]

    tmp_P_ordLoop = np.zeros((seg_size, 6, n_mics))

    for order in orders:
        freq_all_rotors = rpm * order * no_blade / (2 * np.pi)
        bandwidth_all = np.clip(freq_all_rotors * bw_percent, min_bw, max_bw)

        for mic in range(n_mics):
            y, ph, _ = vold_kalman_filter(
                x[:, mic],
                freq_all_rotors,
                fs,
                bandwidth_all,
                p,
                solver=solver,
                use_coupling=use_coupling,
            )

            components_full = np.real(y * ph)
            for rotor_idx in range(6):
                tmp_P_ordLoop[:, rotor_idx, mic] += components_full[cutoff:-cutoff, rotor_idx]
            x[:, mic] -= np.sum(components_full, axis=1)

    return (
        seg_idx,
        tmp_P_ordLoop,
        t[cutoff:-cutoff],
        rpm[cutoff:-cutoff, :],
        x_src[cutoff:-cutoff, :],
    )


def separate_hexacopter_acoustics(mat_file_path, mic_range=16, 
                                   orders=range(2, 31), p=1,
                                   min_bw=3, max_bw=9, bw_percent=0.05,
                                   solver="auto", use_coupling=True,
                                   n_jobs=None):
    """
    Separate hexacopter acoustic signals using Vold-Kalman filtering.
    
    Parameters
    ----------
    mat_file_path : str
        Path to .mat file containing hexacopter data
    mic_range : int, optional
        Microphone index to process (1-based to match MATLAB), default 16
    orders : iterable, optional
        Shaft orders to extract, default range(2, 31)
    p : int, optional
        VK filter order, default 1
    min_bw : float, optional
        Bandwidth floor in Hz, default 3
    max_bw : float, optional
        Bandwidth ceiling in Hz, default 9
    bw_percent : float, optional
        Bandwidth as percentage of frequency, default 0.05 (5%)
    solver : {"auto", "pardiso", "umfpack", "superlu"}, optional
        Sparse solver backend for VKF, default "auto"
    use_coupling : bool, optional
        Whether to include cross-order coupling (B_U) terms, default True
    n_jobs : int or None, optional
        Number of parallel workers for segment processing. Defaults to CPU count.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'P': separated pressures, shape (n_samples, 6_rotors, n_mics)
        - 'T': time vector
        - 'rpm': RPM for each rotor, shape (n_samples, 6)
        - 'original': original signal before separation
        - 'fs': sampling frequency
    """
    
    print("Loading data from:", mat_file_path)
    
    # Load the .mat file
    mat_data = loadmat(mat_file_path, squeeze_me=True)
    
    # Extract acoustics and aircraft data
    acoustics = mat_data['acoustics']
    aircraft = mat_data['aircraft']
    
    # Get fields from structured arrays
    acoustics_T = acoustics['T'].item()
    acoustics_P = acoustics['P'].item()
    aircraft_time = aircraft['time_onboard'].item()
    aircraft_rpm = aircraft['rpm'].item()
    
    print(f"Original data shape: T={acoustics_T.shape}, P={acoustics_P.shape}")
    print(f"Aircraft RPM shape: {aircraft_rpm.shape}")
    
    # Downsample the acoustic data (every 8th sample)
    acoustics_T = acoustics_T[::8]
    acoustics_P = acoustics_P[::8, :]
    
    print(f"Downsampled data shape: T={acoustics_T.shape}, P={acoustics_P.shape}")
    
    # Calculate sampling frequency
    fs = int(np.round(1.0 / (acoustics_T[1] - acoustics_T[0])))
    print(f"Sampling frequency: {fs} Hz")
    
    # Apply highpass filter at 100 Hz
    print("Applying 100 Hz highpass filter...")
    acoustics_P = highpass(acoustics_P, 100, fs)
    
    # Convert mic_range from 1-based (MATLAB) to 0-based (Python)
    mic_idx = mic_range - 1
    
    # Extract time, pressure, and interpolate RPM
    to = acoustics_T
    xo = acoustics_P[:, mic_idx:mic_idx+1]  # Keep 2D for consistency
    
    print(f"Processing microphone {mic_range} (index {mic_idx})")
    print(f"Signal shape: {xo.shape}")
    
    # Interpolate RPM to acoustic time samples
    # aircraft_rpm should be shape (n_time, 6_rotors)
    if aircraft_rpm.ndim == 1:
        # Single rotor case, replicate to 6 rotors
        aircraft_rpm = np.tile(aircraft_rpm[:, np.newaxis], (1, 6))
    
    print(f"Interpolating RPM data...")
    rpmo = np.zeros((len(to), 6))
    for rotor_idx in range(6):
        interp_func = interp1d(aircraft_time, aircraft_rpm[:, rotor_idx], 
                               kind='cubic', fill_value='extrapolate')
        rpmo[:, rotor_idx] = interp_func(to)
    
    print(f"RPM range: {rpmo.min():.1f} - {rpmo.max():.1f} rad/s")
    
    # Set up segmentation
    increment = int(np.floor(0.01 * len(to)))
    cutoff = 4 * increment
    spacing = list(range(cutoff, len(to) - cutoff, 10 * increment))
    spacing.append(len(to) - cutoff)  # Add final point
    
    print(f"\nProcessing {len(spacing)-1} segments")
    print(f"Increment: {increment}, Cutoff: {cutoff}")
    print(f"Extracting orders: {list(orders)}")
    
    # Number of blades per rotor
    no_blade = 2
    
    # Pre-calculate total output size for efficiency
    total_samples = sum(spacing[i+1] - spacing[i] for i in range(len(spacing)-1))
    print(f"Total output samples: {total_samples}")
    
    # Pre-allocate result arrays (much faster than appending lists)
    separated_P = np.zeros((total_samples, 6, xo.shape[1]))
    separated_T = np.zeros(total_samples)
    separated_rpm = np.zeros((total_samples, 6))
    separated_original = np.zeros((total_samples, xo.shape[1]))
    
    if n_jobs is None:
        n_jobs = os.cpu_count() or 1

    total_start = time.time()

    if n_jobs and n_jobs > 1:
        print(f"\nParallel processing with {n_jobs} workers")
        futures = []
        with cf.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for seg_idx in range(len(spacing) - 1):
                start_idx = spacing[seg_idx] - cutoff
                end_idx = spacing[seg_idx + 1] + cutoff

                x_src = xo[start_idx:end_idx, :]
                t = to[start_idx:end_idx]
                rpm = rpmo[start_idx:end_idx, :]

                futures.append(
                    executor.submit(
                        _process_segment,
                        seg_idx,
                        x_src,
                        t,
                        rpm,
                        tuple(orders),
                        no_blade,
                        fs,
                        bw_percent,
                        min_bw,
                        max_bw,
                        p,
                        solver,
                        use_coupling,
                        cutoff,
                    )
                )

            results = [None] * (len(spacing) - 1)
            for fut in cf.as_completed(futures):
                seg_idx, tmp_P, t_center, rpm_center, orig_center = fut.result()
                results[seg_idx] = (tmp_P, t_center, rpm_center, orig_center)

        # Assemble results in order
        output_pos = 0
        for seg_idx, (tmp_P, t_center, rpm_center, orig_center) in enumerate(results):
            seg_size = tmp_P.shape[0]
            separated_P[output_pos:output_pos+seg_size, :, :] = tmp_P
            separated_T[output_pos:output_pos+seg_size] = t_center
            separated_rpm[output_pos:output_pos+seg_size, :] = rpm_center
            separated_original[output_pos:output_pos+seg_size, :] = orig_center
            output_pos += seg_size
    else:
        # Serial processing
        output_pos = 0
        for seg_idx in range(len(spacing) - 1):
            seg_start = time.time()
            print(f"\nSegment {seg_idx+1}/{len(spacing)-1}")

            # Extract segment with cutoff padding
            start_idx = spacing[seg_idx] - cutoff
            end_idx = spacing[seg_idx + 1] + cutoff

            # IMPORTANT: NumPy slicing returns views. We subtract extracted components
            # from `x` in-place during VKF, so `x` must be a copy to avoid mutating
            # `xo` (the original signal) and corrupting later segments.
            x_src = xo[start_idx:end_idx, :]
            x = x_src.copy()

            t = to[start_idx:end_idx]
            rpm = rpmo[start_idx:end_idx, :]

            # Calculate segment size
            seg_size = len(x) - 2*cutoff

            # Save original signal to pre-allocated array
            separated_original[output_pos:output_pos+seg_size, :] = x_src[cutoff:-cutoff, :]

            # Number of microphones
            n_mics = x.shape[1]

            # Initialize array for separated signals (6 rotors)
            tmp_P_ordLoop = np.zeros((seg_size, 6, n_mics))

            print(f"  Processing {len(orders)} orders (all 6 rotors per call)...")

            # Process each order - process all 6 rotors at once per order
            for order in orders:
                # Calculate frequency for all 6 rotors: shape (n_samples, 6)
                # rpm is in rad/s, so divide by 2*pi to convert to Hz
                freq_all_rotors = rpm * order * no_blade / (2 * np.pi)

                # Set bandwidth as percentage of frequency
                bandwidth_all = freq_all_rotors * bw_percent

                # Apply floor and ceiling to bandwidth matrix
                bandwidth_all = np.clip(bandwidth_all, min_bw, max_bw)

                # Apply VK filter for each microphone
                # Process all 6 rotors at once!
                for mic in range(n_mics):
                    y, ph, _ = vold_kalman_filter(
                        x[:, mic],
                        freq_all_rotors,
                        fs,
                        bandwidth_all,
                        p,
                        solver=solver,
                        use_coupling=use_coupling,
                    )

                    # y and ph have shape (n_samples, 6) - one column per rotor
                    # Reconstruct the signal for all rotors
                    components_full = np.real(y * ph)  # shape (n_samples, 6)

                    # Extract the center portion (excluding cutoff) and add to accumulated signal
                    for rotor_idx in range(6):
                        tmp_P_ordLoop[:, rotor_idx, mic] += components_full[cutoff:-cutoff, rotor_idx]

                    # Remove all rotor components from the signal
                    x[:, mic] -= np.sum(components_full, axis=1)

            # Store results in pre-allocated arrays
            separated_P[output_pos:output_pos+seg_size, :, :] = tmp_P_ordLoop
            separated_T[output_pos:output_pos+seg_size] = t[cutoff:-cutoff]
            separated_rpm[output_pos:output_pos+seg_size, :] = rpm[cutoff:-cutoff, :]

            # Update position
            output_pos += seg_size

            seg_time = time.time() - seg_start
            elapsed = time.time() - total_start
            avg_time = elapsed / (seg_idx + 1)
            remaining = avg_time * (len(spacing) - 2 - seg_idx)
            print(f"  Segment time: {seg_time:.1f}s, Est. remaining: {remaining:.0f}s")
    
    total_time = time.time() - total_start
    print(f"\nTotal processing time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # No concatenation needed - already stored in pre-allocated arrays!
    result = {
        'P': separated_P,
        'T': separated_T,
        'rpm': separated_rpm,
        'original': separated_original,
        'fs': fs
    }
    
    print(f"Final separated shape: {result['P'].shape}")
    print(f"Time vector shape: {result['T'].shape}")
    
    return result


def plot_results(separated, mic_plot=1):
    """
    Plot the original, separated, and residual signals.
    
    Parameters
    ----------
    separated : dict
        Dictionary from separate_hexacopter_acoustics()
    mic_plot : int, optional
        Microphone index to plot (1-based), default 1
    """
    
    # Convert to 0-based index
    mic_idx = mic_plot - 1
    
    # Extract data
    T = separated['T']
    original = separated['original'][:, mic_idx]
    P_separated = separated['P'][:, :, mic_idx]  # (n_samples, 6_rotors)
    rpm = separated['rpm']

    # Sum all rotor components
    summed_separated = np.sum(P_separated, axis=1)

    # Residual
    residual = original - summed_separated
    plt.figure(figsize=(12, 6))
    plt.plot(T, original, linewidth=1.2, label='Original', alpha=0.7)
    plt.plot(T, summed_separated, linewidth=1.2, label='Separated (sum)', alpha=0.7)
    plt.plot(T, residual, linewidth=1.2, label='Residual', alpha=0.7)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.xlabel('Time, s')
    plt.ylabel('Acoustic Pressure, Pa')
    plt.tight_layout()
    
    # Figure 2: Individual rotor components (offset for visibility)
    plt.figure(figsize=(12, 8))
    offsets = [0.5, 0.25, 0, -0.25, -0.5, -0.75]
    rotor_labels = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']
    
    for rotor_idx in range(6):
        plt.plot(T, offsets[rotor_idx] + P_separated[:, rotor_idx], 
                linewidth=1.2, label=rotor_labels[rotor_idx], alpha=0.7)
    
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.xlabel('Time, s')
    plt.ylabel('Acoustic Pressure, Pa (offset)')
    plt.title(f'Separated Rotor Components - Microphone {mic_plot}')
    plt.tight_layout()
    
    # Figure 3: RPM Time History
    plt.figure(figsize=(12, 6))
    for rotor_idx in range(6):
        plt.plot(T, rpm[:, rotor_idx], linewidth=1.2, 
                label=rotor_labels[rotor_idx], alpha=0.7)
    
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.xlabel('Time, s')
    plt.ylabel('Rotor Speed, RPM')
    plt.title('Rotor RPM Time History')
    plt.tight_layout()
    
    plt.show()


if __name__ == "__main__":
    # Path to the data file
    mat_file = 'example_data/F100_20_P#1[FA1FZC-3.h5].mat'
    
    # Check if file exists
    if not os.path.exists(mat_file):
        print(f"Error: Data file not found: {mat_file}")
        print("Please ensure the hexacopter data file is in the correct location.")
        exit(1)
    
    print("="*70)
    print("Hexacopter Acoustic Signal Separation")
    print("Using Vold-Kalman Filter")
    print("="*70)
    
    # Process the hexacopter data
    # Parameters match MATLAB script:
    # - mic_range=16 (microphone 16)
    # - orders 2-30
    # - p=1 (filter order)
    # - min_bw=3 Hz, max_bw=9 Hz
    # - 5% bandwidth
    
    solver = "auto"  # prefers Pardiso/UMFPACK if available
    use_coupling = True  # set False for faster but less accurate results
    n_jobs = None  # defaults to CPU count

    separated = separate_hexacopter_acoustics(
        mat_file,
        mic_range=16,
        orders=range(2, 31),
        p=1,
        min_bw=3,
        max_bw=9,
        bw_percent=0.05,
        solver=solver,
        use_coupling=use_coupling,
        n_jobs=n_jobs
    )
    
    print("\n" + "="*70)
    print("Processing complete!")
    print("="*70)
    
    # Plot results
    plot_results(separated, mic_plot=1)
