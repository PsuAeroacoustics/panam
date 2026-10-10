#!/usr/bin/env python3
"""
Hexacopter Acoustic Signal Separation using Vold-Kalman Filter

This script applies the Vold-Kalman filter to separate harmonic components
of acoustic data from a reconfigurable hexacopter with 6 rotors.

Translated from a MATLAB script developed by Joel Sundar Rachaprolu at Penn State
University. 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.signal import resample_poly
import os
import time
import concurrent.futures as cf
from vold_kalman_filter import vold_kalman_filter
from flight_acoustics import highpass


# ----------------------------
# Analysis flags (edit here)
# ----------------------------
# Data
MAT_FILE = 'example_data/F100_20_P#1[FA1FZC-3.h5].mat'
DOWNSAMPLE_FACTOR = 8
HIGHPASS_CUTOFF_HZ = 100

# VKF settings
MIC_RANGE = 16
VKF_ORDERS = list(range(2, 31))
VKF_P = 1
VKF_MIN_BW = 3
VKF_MAX_BW = 9
VKF_BW_PERCENT = 0.05
VKF_SOLVER = "auto"  # UMFPACK if scikits.umfpack is installed, else SuperLU
VKF_USE_COUPLING = True  # False is faster but less accurate
VKF_N_JOBS = None  # defaults to CPU count

# Plot settings
PLOT_INTERACTIVE = False
PLOT_MIC = 1
ZOOM_TIME_WINDOW = (108, 108.2)

def load_hexacopter_data(mat_file_path, downsample_factor=8):
    """Load and downsample hexacopter acoustic and aircraft data.
    
    Args:
        mat_file_path: Path to .mat file containing hexacopter data.
        downsample_factor: Factor by which to downsample acoustic data (default: 8).
        
    Returns:
        Dictionary with keys: acoustics_T, acoustics_P, aircraft_time, aircraft_rpm, fs.
    """
    print("Loading data from:", mat_file_path)
    
    mat_data = loadmat(mat_file_path, squeeze_me=True)
    acoustics = mat_data['acoustics']
    aircraft = mat_data['aircraft']
    
    acoustics_T = acoustics['T'].item()
    acoustics_P = acoustics['P'].item()
    aircraft_time = aircraft['time_onboard'].item()
    aircraft_rpm = aircraft['rpm'].item()
    
    print(f"Original data shape: T={acoustics_T.shape}, P={acoustics_P.shape}")
    print(f"Aircraft RPM shape: {aircraft_rpm.shape}")
    
    # Downsample the acoustic data.  Low-pass before keeping every Nth sample:
    # plain slicing folded everything above the new Nyquist back into the band,
    # +3 dB at 2-3 kHz and +6 dB at 3-4 kHz on this record.
    acoustics_T = acoustics_T[::downsample_factor]
    acoustics_P = resample_poly(acoustics_P, 1, downsample_factor, axis=0)
    
    print(f"Downsampled data shape: T={acoustics_T.shape}, P={acoustics_P.shape}")
    
    # Calculate sampling frequency
    fs = int(np.round(1.0 / (acoustics_T[1] - acoustics_T[0])))
    print(f"Sampling frequency: {fs} Hz")
    
    return {
        'acoustics_T': acoustics_T,
        'acoustics_P': acoustics_P,
        'aircraft_time': aircraft_time,
        'aircraft_rpm': aircraft_rpm,
        'fs': fs
    }


def apply_highpass_filter(acoustics_P, cutoff_freq, fs):
    """Apply highpass filter to acoustic pressure signals.
    
    Args:
        acoustics_P: Acoustic pressure array, shape (n_samples, n_mics).
        cutoff_freq: Highpass filter cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        
    Returns:
        Filtered acoustic pressure array with same shape as input.
    """
    print(f"Applying {cutoff_freq} Hz highpass filter...")
    
    if acoustics_P.ndim == 1:
        return highpass(acoustics_P, cutoff_freq, fs)
    else:
        # Apply to each column independently
        return np.column_stack([highpass(acoustics_P[:, i], cutoff_freq, fs) 
                                for i in range(acoustics_P.shape[1])])


def interpolate_rpm_to_acoustic_time(aircraft_time, aircraft_rpm, acoustics_T):
    """Interpolate RPM data to acoustic sampling times.
    
    Args:
        aircraft_time: Aircraft data time vector.
        aircraft_rpm: RPM array, shape (n_time, n_rotors) or (n_time,).
        acoustics_T: Acoustic time vector to interpolate to.
        
    Returns:
        Interpolated RPM array, shape (len(acoustics_T), 6).
    """
    print(f"Interpolating RPM data...")
    
    # Handle single rotor case
    if aircraft_rpm.ndim == 1:
        aircraft_rpm = np.tile(aircraft_rpm[:, np.newaxis], (1, 6))
    
    rpmo = np.zeros((len(acoustics_T), 6))
    for rotor_idx in range(6):
        interp_func = interp1d(aircraft_time, aircraft_rpm[:, rotor_idx], 
                               kind='cubic', fill_value='extrapolate')
        rpmo[:, rotor_idx] = interp_func(acoustics_T)
    
    print(f"RPM range: {rpmo.min():.1f} - {rpmo.max():.1f} rad/s")
    return rpmo


def setup_segmentation(n_samples, segment_fraction=0.01, overlap_factor=4, stride_factor=10):
    """Setup segment boundaries for processing long signals.
    
    Args:
        n_samples: Total number of samples in signal.
        segment_fraction: Fraction of signal for increment size (default: 0.01).
        overlap_factor: Multiplier for cutoff region size (default: 4).
        stride_factor: Multiplier for segment stride (default: 10).
        
    Returns:
        Tuple of (spacing, cutoff, increment) where spacing is list of segment boundaries.
    """
    increment = int(np.floor(segment_fraction * n_samples))
    cutoff = overlap_factor * increment
    spacing = list(range(cutoff, n_samples - cutoff, stride_factor * increment))
    spacing.append(n_samples - cutoff)  # Add final point
    
    total_samples = sum(spacing[i+1] - spacing[i] for i in range(len(spacing)-1))
    
    print(f"\nProcessing {len(spacing)-1} segments")
    print(f"Increment: {increment}, Cutoff: {cutoff}")
    print(f"Total output samples: {total_samples}")
    
    return spacing, cutoff, increment


def process_segment_serial(seg_idx, spacing, cutoff, xo, to, rpmo, orders, no_blade,
                          fs, bw_percent, min_bw, max_bw, p, solver, use_coupling,
                          total_spacing, total_start):
    """Process a single segment in serial mode.
    
    Args:
        seg_idx: Current segment index.
        spacing: List of segment boundary indices.
        cutoff: Number of samples to exclude at boundaries.
        xo: Original signal array, shape (n_samples, n_mics).
        to: Time vector.
        rpmo: RPM array, shape (n_samples, 6_rotors).
        orders: Iterable of shaft orders to extract.
        no_blade: Number of blades per rotor.
        fs: Sampling frequency in Hz.
        bw_percent: Bandwidth as percentage of frequency.
        min_bw: Minimum bandwidth in Hz.
        max_bw: Maximum bandwidth in Hz.
        p: VK filter order.
        solver: Sparse solver backend.
        use_coupling: Whether to include cross-order coupling.
        total_spacing: Total number of segments.
        total_start: Start time for progress tracking.
        
    Returns:
        Tuple of (separated_signals, time, rpm, original, seg_size).
    """
    seg_start = time.time()
    print(f"\nSegment {seg_idx+1}/{total_spacing}")

    # Extract segment with cutoff padding
    start_idx = spacing[seg_idx] - cutoff
    end_idx = spacing[seg_idx + 1] + cutoff

    orders = tuple(orders)
    print(f"  Processing {len(orders)} orders (all 6 rotors per call)...")

    _, tmp_P_ordLoop, t_center, rpm_center, orig_center = _process_segment(
        seg_idx, xo[start_idx:end_idx, :], to[start_idx:end_idx], rpmo[start_idx:end_idx, :],
        orders, no_blade, fs, bw_percent, min_bw, max_bw, p, solver, use_coupling, cutoff)

    seg_time = time.time() - seg_start
    elapsed = time.time() - total_start
    avg_time = elapsed / (seg_idx + 1)
    remaining = avg_time * (total_spacing - 1 - seg_idx)
    print(f"  Segment time: {seg_time:.1f}s, Est. remaining: {remaining:.0f}s")

    return tmp_P_ordLoop, t_center, rpm_center, orig_center, tmp_P_ordLoop.shape[0]


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
    """Process a single segment for parallel execution.
    
    Args:
        seg_idx: Segment index for result ordering.
        x_src: Source signal array, shape (n_samples, n_mics).
        t: Time vector for segment.
        rpm: RPM array, shape (n_samples, 6_rotors).
        orders: Tuple of shaft orders to extract.
        no_blade: Number of blades per rotor.
        fs: Sampling frequency in Hz.
        bw_percent: Bandwidth as percentage of frequency.
        min_bw: Minimum bandwidth in Hz.
        max_bw: Maximum bandwidth in Hz.
        p: VK filter order.
        solver: Sparse solver backend.
        use_coupling: Whether to include cross-order coupling.
        cutoff: Number of samples to exclude at boundaries.
        
    Returns:
        Tuple of (seg_idx, separated_signals, time, rpm, original).
    """
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
                                   n_jobs=None, downsample_factor=8,
                                   highpass_cutoff=100):
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
    solver : {"auto", "umfpack", "superlu"}, optional
        Sparse solver backend for VKF (see vold_kalman_filter), default "auto"
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
    orders = tuple(orders)  # iterated once per segment, so not a one-shot generator

    # Load and preprocess data
    data = load_hexacopter_data(mat_file_path, downsample_factor=downsample_factor)
    acoustics_T = data['acoustics_T']
    acoustics_P = data['acoustics_P']
    aircraft_time = data['aircraft_time']
    aircraft_rpm = data['aircraft_rpm']
    fs = data['fs']
    
    # Apply highpass filter
    acoustics_P = apply_highpass_filter(acoustics_P, highpass_cutoff, fs)
    
    # Convert mic_range from 1-based (MATLAB) to 0-based (Python)
    mic_idx = mic_range - 1
    
    # Extract time and pressure for selected microphone
    to = acoustics_T
    xo = acoustics_P[:, mic_idx:mic_idx+1]  # Keep 2D for consistency
    
    print(f"Processing microphone {mic_range} (index {mic_idx})")
    print(f"Signal shape: {xo.shape}")
    
    # Interpolate RPM to acoustic time samples
    rpmo = interpolate_rpm_to_acoustic_time(aircraft_time, aircraft_rpm, to)
    
    # Set up segmentation
    spacing, cutoff, increment = setup_segmentation(len(to))
    
    print(f"Extracting orders: {list(orders)}")
    
    # Number of blades per rotor
    no_blade = 2
    
    # Pre-calculate total output size for efficiency
    total_samples = sum(spacing[i+1] - spacing[i] for i in range(len(spacing)-1))
    
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
            tmp_P, t_center, rpm_center, orig_center, seg_size = process_segment_serial(
                seg_idx, spacing, cutoff, xo, to, rpmo, orders, no_blade,
                fs, bw_percent, min_bw, max_bw, p, solver, use_coupling,
                len(spacing) - 1, total_start
            )
            
            # Store results in pre-allocated arrays
            separated_P[output_pos:output_pos+seg_size, :, :] = tmp_P
            separated_T[output_pos:output_pos+seg_size] = t_center
            separated_rpm[output_pos:output_pos+seg_size, :] = rpm_center
            separated_original[output_pos:output_pos+seg_size, :] = orig_center
            output_pos += seg_size
    
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


def plot_results(
    separated,
    mic_plot=1,
    save_dir="demo_plots",
    filename_prefix="hexa",
    show_plots=True,
    zoom_time_window=(108, 108.2),
):
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

    # Show time signals
    residual = original - summed_separated
    fig1 = plt.figure(figsize=(12, 6))
    plt.plot(T, original, linewidth=1.2, label='Original', alpha=0.7)
    plt.plot(T, summed_separated, linewidth=1.2, label='Separated (sum)', alpha=0.7)
    plt.plot(T, residual, linewidth=1.2, label='Residual', alpha=0.7)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.xlabel('Time, s')
    plt.ylabel('Acoustic Pressure, Pa')
    plt.tight_layout()

    # Zoomed time signals
    fig1_zoom = plt.figure(figsize=(12, 6))
    plt.plot(T, original, linewidth=1.2, label='Original', alpha=0.7)
    plt.plot(T, summed_separated, linewidth=1.2, label='Separated (sum)', alpha=0.7)
    plt.plot(T, residual, linewidth=1.2, label='Residual', alpha=0.7)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.xlabel('Time, s')
    plt.ylabel('Acoustic Pressure, Pa')
    if zoom_time_window is not None:
        plt.xlim(*zoom_time_window)
    plt.tight_layout()
    
    # Figure 2: Individual rotor components (offset for visibility)
    fig2 = plt.figure(figsize=(12, 8))
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
    
    # RPM Time History
    fig3 = plt.figure(figsize=(12, 6))
    for rotor_idx in range(6):
        plt.plot(T, 60.0*rpm[:, rotor_idx]/(2*np.pi), linewidth=1.2, 
                label=rotor_labels[rotor_idx], alpha=0.7)
    
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.xlabel('Time, s')
    plt.ylabel('Rotor Speed, RPM')
    plt.title('Rotor RPM Time History')
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig1.savefig(os.path.join(save_dir, f"{filename_prefix}_original_separated.png"), dpi=300)
        fig1_zoom.savefig(os.path.join(save_dir, f"{filename_prefix}_original_separated_zoom.png"), dpi=300)
        fig2.savefig(os.path.join(save_dir, f"{filename_prefix}_rotor_components.png"), dpi=300)
        fig3.savefig(os.path.join(save_dir, f"{filename_prefix}_rpm_time_history.png"), dpi=300)
    
    if show_plots:
        plt.show()


if __name__ == "__main__":
    # Path to the data file
    mat_file = MAT_FILE
    
    # Check if file exists
    if not os.path.exists(mat_file):
        print(f"Error: Data file not found: {mat_file}")
        print("Please ensure the hexacopter data file is in the correct location.")
        exit(1)
    
    print("="*70)
    print("Hexacopter Acoustic Signal Separation")
    print("Using Vold-Kalman Filter")
    print("="*70)
    
    # Perform separation with the settings at the top of this file
    separated = separate_hexacopter_acoustics(
        mat_file,
        mic_range=MIC_RANGE,
        orders=VKF_ORDERS,
        p=VKF_P,
        min_bw=VKF_MIN_BW,
        max_bw=VKF_MAX_BW,
        bw_percent=VKF_BW_PERCENT,
        solver=VKF_SOLVER,
        use_coupling=VKF_USE_COUPLING,
        n_jobs=VKF_N_JOBS,
        downsample_factor=DOWNSAMPLE_FACTOR,
        highpass_cutoff=HIGHPASS_CUTOFF_HZ,
    )
    
    print("\n" + "="*70)
    print("Processing complete!")
    print("="*70)
    
    # Plot results
    plot_results(
        separated,
        mic_plot=PLOT_MIC,
        show_plots=PLOT_INTERACTIVE,
        zoom_time_window=ZOOM_TIME_WINDOW,
    )
