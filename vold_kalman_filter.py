"""
Multi-shaft Vold-Kalman filter implementation.

This module implements the multi-shaft Vold-Kalman filter for extracting
acoustic signal components at specific frequencies/orders.

This module is adapted from a MATLAB implementation by Joel Sundar Rachaprolu at Penn State University.

Functions
---------
vold_kalman_filter(x, freq, fs, bandwidth, p, r=None)
    Extract complex envelopes and phasors from acoustic signal using Vold-Kalman filtering.
"""

import numpy as np
from scipy.sparse import diags, kron, spdiags, eye as speye
from scipy.sparse.linalg import spsolve
from scipy.special import comb

# Cache for expensive, size-dependent matrices/indices
_AA_CACHE = {}
_BU_INDEX_CACHE = {}

# Optional faster sparse solvers (if installed)
try:
    from pypardiso import spsolve as _pardiso_spsolve  # type: ignore
    _HAVE_PARDISO = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_PARDISO = False

try:
    import scikits.umfpack  # type: ignore
    _HAVE_UMFPACK = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_UMFPACK = False


def _get_bu_index_cache(n_x, n_ord):
    """Precompute and cache row/col indices and pair indices for B_U."""
    key = (n_x, n_ord)
    cached = _BU_INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    n_pairs = n_ord * (n_ord - 1) // 2
    total_nnz = n_x * n_pairs

    row_indices = np.empty(total_nnz, dtype=int)
    col_indices = np.empty(total_nnz, dtype=int)
    pair_i = np.empty(n_pairs, dtype=int)
    pair_j = np.empty(n_pairs, dtype=int)

    idx = 0
    pair_idx = 0
    for i in range(n_ord):
        rows = np.arange(i * n_x, (i + 1) * n_x)
        for j in range(i + 1, n_ord):
            cols = np.arange(j * n_x, (j + 1) * n_x)
            row_indices[idx:idx + n_x] = rows
            col_indices[idx:idx + n_x] = cols
            pair_i[pair_idx] = i
            pair_j[pair_idx] = j
            pair_idx += 1
            idx += n_x

    _BU_INDEX_CACHE[key] = (row_indices, col_indices, pair_i, pair_j)
    return row_indices, col_indices, pair_i, pair_j


def vold_kalman_filter(x, freq, fs, bandwidth, p, r=None, solver="auto", use_coupling=True):
    r"""
    Multi-shaft Vold-Kalman filter for extracting acoustic signal components.

    This function extracts specific frequency/order components from an acoustic
    time-history signal using the Vold-Kalman filtering algorithm.

    Parameters
    ----------
    x : ndarray
        Acoustic time-history signal, shape (n_samples,)
    freq : ndarray
        Frequency vector with frequencies/orders of interest, Hz.
        Must have same length as x, shape (n_samples,) or (n_samples, n_orders)
        For order tracking: freq = order × shaft_speed(t)
        Can track multiple orders simultaneously with shape (n_samples, n_orders)
    fs : float
        Sampling frequency of the acoustic signal and frequency vector, Hz
    bandwidth : float or ndarray
        Bandwidth for weighting factor formulation, Hz.
        Can be:
        - scalar: single bandwidth for all orders
        - vector with same length as x: time-varying bandwidth
        - vector with length equal to number of orders: order-wise bandwidth
        - array same shape as freq: time and order-varying bandwidth
    p : int
        Structural filter order (order of difference operator for regularization).
        NOT the harmonic order to track (those are specified by freq).
        p=1: first-order differences (velocity constraint) - recommended for time-varying freq
        p=2: second-order differences (acceleration constraint) - can cause envelope decay
        p=3: third-order differences (jerk constraint) - typically too strong
    r : float or ndarray, optional
        Weighting factor for the filter. If not provided, computed from bandwidth.
        Default is None (compute from bandwidth).
    solver : {"auto", "pardiso", "umfpack", "superlu"}, optional
        Sparse solver backend. "auto" prefers Pardiso (if installed), then UMFPACK,
        and falls back to SuperLU. Default is "auto".
    use_coupling : bool, optional
        Whether to include cross-order coupling (B_U) terms. Default True.
        Setting False can speed up solves but changes results.

    Returns
    -------
    y : ndarray
        Extracted complex envelopes of each order, shape (n_samples, n_orders)
    phasor : ndarray
        Complex phasor corresponding to each order, shape (n_samples, n_orders)
    cost_Mat : ndarray
        Cost matrix of the iterative solver

    Notes
    -----
    Waveforms can be obtained by multiplying complex envelopes and phasors:
        waveform = y * phasor

    This algorithm tracks acoustic signals produced by shafts/orders given in the
    reference frequency vector.

    If weighting factor is not specified, bandwidth is used to compute a desired
    weighting factor for the signal.
    
    **Bandwidth Selection for Different Scenarios:**
    
    1. **Constant frequency signals**: Use fixed bandwidth
       - bandwidth = 20  # Hz (absolute value)
       - p = 1           # Filter order
       - Works well because relative bandwidth is constant
       
    2. **Time-varying frequency signals (frequency sweeps)**: Use FREQUENCY-ADAPTIVE bandwidth
       - bandwidth = 0.10 * freq  # 10% of instantaneous frequency
       - p = 1                     # Keep filter order at 1 (higher orders degrade envelope)
       - This maintains constant relative bandwidth throughout the sweep
    
    The key insight: **relative bandwidth** (bandwidth / frequency) must remain constant.
    For time-varying frequencies, use adaptive bandwidth to ensure this constancy.
    Higher filter orders can degrade envelope preservation with adaptive bandwidth,
    so keep p=1 for frequency-adaptive cases.

    Examples
    --------
    >>> # Constant frequency - use fixed bandwidth
    >>> x = np.sin(2*np.pi*100*np.arange(1000)/1000)
    >>> freq = 100 * np.ones(1000)
    >>> y, phasor = vold_kalman_filter(x, freq, 1000, 20, 1)  # BW=20 Hz
    
    >>> # Time-varying frequency - use adaptive bandwidth  
    >>> freq_chirp = 50 + 100*np.arange(1000)/1000  # 50-150 Hz sweep
    >>> phase = 2 * np.pi * np.cumsum(freq_chirp) / 1000
    >>> x_chirp = np.sin(phase)
    >>> y, phasor = vold_kalman_filter(x_chirp, freq_chirp, 1000, 0.10*freq_chirp, 1)
    """

    # Input validation
    use_weight_factor = r is None

    # Initialize variables
    dt = 1.0 / fs
    n_x = len(x)
    x = np.asarray(x)

    # Handle freq as 1D or 2D array
    freq = np.atleast_1d(freq)
    if freq.ndim == 1:
        freq = freq[:, np.newaxis]

    n_f, n_ord = freq.shape

    if n_f != n_x:
        raise ValueError("The length of frequency vector must be the same as the signal")

    n_tot = n_x * n_ord

    # Creating the Phasor Vector
    phasor = np.exp(2j * np.pi * np.cumsum(freq, axis=0) * dt)

    # Initializing the bandwidth vector
    bandwidth = np.atleast_1d(bandwidth)
    
    if bandwidth.size == 1:
        bw_band = bandwidth[0] * np.ones((n_x, 1))
    elif bandwidth.shape == x.shape:
        bw_band = bandwidth.reshape(-1, 1)
    elif bandwidth.size == n_ord:
        bw_band = np.tile(bandwidth, (n_x, 1))
    elif bandwidth.shape == freq.shape:
        bw_band = bandwidth
    else:
        raise ValueError(
            "Bandwidth not chosen properly. Check bandwidth scalar/vector dimensions"
        )

    bw_rad = bw_band * np.pi / fs

    # Setting the filter order & coefficients
    p_arr = np.atleast_1d(p).astype(int)
    p_p = int(np.max(p_arr))
    if p_p < 1 or n_x <= p_p:
        raise ValueError("p must be at least 1 and smaller than the signal length")

    def _diff_coeff(order: int) -> np.ndarray:
        """Binomial finite-difference coefficients with alternating sign."""
        return np.array([((-1) ** k) * comb(order, k) for k in range(order + 1)], dtype=float)

    # Main difference coefficients for order p_p
    diff_main = _diff_coeff(p_p)

    # Smoothness operator: the p-th difference of each order's envelope, n_x - p
    # rows per order.  Every row is a whole stencil, so it sums to zero and
    # penalises only changes in the envelope, never its level, and the record
    # ends are left free.  It used to be padded out to n_x rows per order with
    # truncated (single order, p >= 2) or misaligned (several orders) boundary
    # rows; those penalised the level itself and drove the envelope to zero at
    # the ends of the record.  It was also built through dense n_x x n_x
    # arrays, so memory grew as n_x**2.
    aa_cache_key = (n_x, n_ord, p_p)
    AA_sparse = _AA_CACHE.get(aa_cache_key)
    if AA_sparse is None:
        D = diags([np.full(n_x - p_p, c) for c in diff_main], list(range(p_p + 1)),
                  shape=(n_x - p_p, n_x), format='csr')
        AA_sparse = kron(speye(n_ord, format='csr'), D, format='csr')
        _AA_CACHE[aa_cache_key] = AA_sparse

    # DEBUG: Verify AA boundaries
    if False:  # Set to True for debugging
        AA_check = AA_sparse.toarray()
        print(f"AA[0, :10] = {AA_check[0, :10]}")
        print(f"AA[-1, -10:] = {AA_check[-1, -10:]}")

    # Solving for r-weighting factor
    if use_weight_factor:
        # Compute weighting factor from bandwidth
        if bw_rad.ndim == 1 or bw_rad.shape[1] == 1:
            # Scalar or time-varying bandwidth
            weig_r0 = _compute_weighting_factor(bw_rad.ravel(), p_p)
            weig_r = np.tile(weig_r0, n_ord)
        else:
            # Order-wise bandwidth
            weig_r_list = []
            for j in range(n_ord):
                weig_r0_j = _compute_weighting_factor(bw_rad[:, j], p_p)
                weig_r_list.extend(weig_r0_j)
            weig_r = np.array(weig_r_list)
            weig_r = weig_r[:n_tot]
    else:
        # Use provided weighting factor
        if np.isscalar(r):
            weig_r = r * np.ones(n_tot)
        else:
            weig_r = np.asarray(r).ravel()
            weig_r = np.tile(weig_r, n_ord) if len(weig_r) < n_tot else weig_r[:n_tot]
    
    # Compute B0 = AA' * R^2 * AA + I, the regularised least-squares matrix.
    # Each difference row takes the weight of the sample it starts at.
    n_rows = n_ord * (n_x - p_p)
    weig_r2 = (np.asarray(weig_r, dtype=float).reshape(n_x, n_ord, order="F")[:n_x - p_p] ** 2).ravel(order="F")
    RR_squared = spdiags([weig_r2], [0], n_rows, n_rows, format='csr')
    B0 = AA_sparse.T @ RR_squared @ AA_sparse + speye(n_tot, format='csr')
    
    # Precompute conjugate phasor in (n_x, n_ord) form
    conj_phasor = np.conj(phasor)
    
    # Build off-diagonal coupling matrix B_U (MATLAB lines 249-261)
    # This captures cross-order interactions via phasor products
    if n_ord > 1 and use_coupling:
        from scipy.sparse import coo_matrix
        # Build B_U using cached structure (row/col indices) and update only values
        row_indices, col_indices, pair_i, pair_j = _get_bu_index_cache(n_x, n_ord)

        # Vectorized pairwise products: shape (n_x, n_pairs)
        values_matrix = conj_phasor[:, pair_i] * phasor[:, pair_j]
        # Flatten column-wise to match row/col block ordering
        values = values_matrix.reshape(-1, order="F")
        
        B_U = coo_matrix((values, (row_indices, col_indices)), shape=(n_tot, n_tot), dtype=complex)
        B_U = B_U.tocsr()  # Convert to CSR for efficient matrix operations
        
        # Form full B matrix: B = B0 + B_U + B_U' (MATLAB line 264)
        # Keep in sparse format for more efficient solve
        B_mat = B0.astype(complex) + B_U + B_U.conj().T
    else:
        # Single order: no cross-coupling needed
        B_mat = B0.astype(complex)
    # Calculate Right-hand side of the linear differential equations
    # Avoid creating large tiled vectors by using broadcasting
    cH_x = (conj_phasor * x[:, None]).ravel(order="F")

    # DEBUG: Print solve inputs
    if False:  # Set to True for debugging
        print(f"B_mat[0,0] = {B_mat[0,0]:.4f}")
        print(f"B_mat is symmetric: {np.allclose(B_mat, B_mat.conj().T)}")
        print(f"cH_x[0:5] = {cH_x[0:5]}")

    # Solve the linear equations using sparse solver.  Bad arguments raise here,
    # outside the fallback below, instead of being swallowed by it.
    solver_choice = (solver or "auto").lower()
    if solver_choice not in ("auto", "pardiso", "umfpack", "superlu"):
        raise ValueError(f"Unknown solver '{solver}'")
    if solver_choice == "pardiso" and not _HAVE_PARDISO:
        raise RuntimeError("pypardiso is not installed")
    # Convert to CSC for faster factorization in spsolve
    B_mat = B_mat.tocsc()
    try:
        if solver_choice in ("auto", "pardiso") and _HAVE_PARDISO:
            y_R = _pardiso_spsolve(B_mat, cH_x)
        else:
            try:
                y_R = spsolve(B_mat, cH_x, use_umfpack=_HAVE_UMFPACK and solver_choice != "superlu")
            except TypeError:
                y_R = spsolve(B_mat, cH_x)
    except Exception:
        if solver_choice != "auto":
            raise
        # pypardiso rejects complex matrices, and this system is complex:
        # fall back to SuperLU, never to a dense solve (n_tot**2 memory).
        y_R = spsolve(B_mat, cH_x, use_umfpack=False)

    # DEBUG: Check residual
    if False:  # Enable for debugging
        residual = B_mat @ y_R - cH_x
        print(f"Residual norm: {np.linalg.norm(residual):.6e}")
        print(f"Relative residual: {np.linalg.norm(residual) / np.linalg.norm(cH_x):.6e}")

    # Get cost matrix (residual) - keep sparse if B_mat is sparse
    if hasattr(B_mat, 'dot'):
        # B_mat is sparse, use sparse multiplication
        cost_mat = cH_x - B_mat.dot(y_R)
    else:
        # B_mat is dense
        cost_mat = cH_x - B_mat @ y_R


    # Reorder the complex envelope from a column vector to a
    # (n_samples, n_orders) matrix
    y = y_R.reshape((n_x, n_ord), order="F")

    # Apply 2x scaling factor (standard in VKF implementations)
    y = 2.0 * y

    return y, phasor, cost_mat


def _pascal_coefficients(n):
    """
    Generate Pascal's triangle coefficients.

    Parameters
    ----------
    n : int
        Number of rows in Pascal's triangle

    Returns
    -------
    ndarray
        Pascal's triangle matrix, shape (n, n)
    """
    pascal = np.zeros((n, n))
    for i in range(n):
        pascal[i, 0] = 1
        for j in range(1, i + 1):
            pascal[i, j] = pascal[i - 1, j - 1] + pascal[i - 1, j]
    return pascal


# Module-level cache for Vandermonde coefficients (keyed by filter order)
_vandermonde_cache = {}

def _compute_weighting_factor(bw_rad, p_p):
    """
    Compute weighting factor from bandwidth using Vandermonde system.

    Parameters
    ----------
    bw_rad : ndarray
        Bandwidth in radians, shape (n_samples,)
    p_p : int
        Filter order

    Returns
    -------
    ndarray
        Weighting factor array, shape (n_samples,)
    """
    # MATLAB Vandermonde system approach with caching
    # Coefficients depend only on filter order p_p, so cache them
    if p_p not in _vandermonde_cache:
        # Build coefficient matrix (weigF.coecos in MATLAB)
        n = np.arange(p_p + 1)
        sign = (-1.0) ** n
        coecos = np.ones((p_p + 1, p_p + 1))
        
        for i in range(1, p_p + 1):
            coecos[i, :] = (n ** (2 * (i - 1))) * sign
        
        # Right-hand side (weigF.coe in MATLAB)
        coe = np.zeros(p_p + 1)
        coe[0] = 2.0 ** (2 * p_p)
        
        # Solve Vandermonde system: coecos \ coe
        coeff = np.linalg.solve(coecos, coe) * sign
        _vandermonde_cache[p_p] = coeff
    else:
        coeff = _vandermonde_cache[p_p]
    
    # Compute denominator as sum of coeff[i] * cos(bw_rad * i) using vectorized operations
    # Build cosine terms: cos(bw_rad * 0), cos(bw_rad * 1), ..., cos(bw_rad * (p_p))
    cos_indices = np.arange(len(coeff))
    cos_terms = np.cos(bw_rad[:, np.newaxis] * cos_indices[np.newaxis, :])
    denominator = cos_terms @ coeff
    
    # Numerator is sqrt(2) - 1
    numerator = np.sqrt(2.0) - 1.0
    
    # Avoid division by zero
    denominator = np.maximum(denominator, 1e-12)
    r0 = np.sqrt(np.maximum(numerator / denominator, 1e-12))
    
    return r0


if __name__ == "__main__":
    # Example usage demonstrating filter with constant vs. time-varying frequencies
    import matplotlib.pyplot as plt    
    import os
    
    # Create demo_plots directory if it doesn't exist
    os.makedirs('demo_plots', exist_ok=True)    
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
    print(f"  Extracted signal: RMS = {np.sqrt(np.mean(x_filtered_const**2)):.3f} (target: 0.8)")
    print(f"  MSE = {error_const:.2e}")
    
    # ========================================================================
    
    # Case 2: TIME-VARYING frequency (50-150 Hz chirp)
    freq1_chirp = 50. + 100*t  # Hz (sweeps from 50 to 150)
    phase_chirp = 2 * np.pi * np.cumsum(freq1_chirp) / fs
    x_clean_chirp = 0.8 * np.sin(phase_chirp)
    
    # Add noise and crossing interference chirp (sweeps opposite direction: 150 to 50 Hz)
    np.random.seed(42)
    x_noisy_chirp = x_clean_chirp + 0.05 * np.random.randn(len(t))
    phase_interference = 2 * np.pi * np.cumsum(150. - 100*t) / fs
    x_noisy_chirp += 0.3 * np.sin(phase_interference)

    # Track both orders with fixed bandwidth (for comparison)
    bandwidth_fixed = 20  # Hz
    y_wrong, phasor_wrong, _ = vold_kalman_filter(x_noisy_chirp, freq1_chirp, fs, bandwidth_fixed, p)
    x_filtered_wrong = np.real(y_wrong[:, 0] * phasor_wrong[:, 0])

    # Track both the main chirp and the crossing interference with adaptive bandwidth
    freq_chirp_multi = np.zeros((len(t), 2))
    freq_chirp_multi[:, 0] = freq1_chirp  # Main chirp: 50-150 Hz
    freq_chirp_multi[:, 1] = 150. - 100*t  # Crossing interference: 150-50 Hz
    
    bandwidth_adaptive = 0.10 * freq_chirp_multi  # 10% of instantaneous frequency
    
    y_correct, phasor_correct, _ = vold_kalman_filter(x_noisy_chirp, freq_chirp_multi, fs, bandwidth_adaptive, p)
    x_filtered_correct = np.real(y_correct[:, 0] * phasor_correct[:, 0])
    x_filtered_interference = np.real(y_correct[:, 1] * phasor_correct[:, 1])
    
    error_correct = np.mean((x_filtered_correct - x_clean_chirp)**2)
    print(f"\nCase 2 - Time-Varying Frequencies (50-150 Hz chirp + crossing interference):")
    print(f"  Main chirp extracted: RMS = {np.sqrt(np.mean(x_filtered_correct**2)):.3f} (target: 0.8), MSE = {error_correct:.2e}")
    print(f"  Interference extracted: RMS = {np.sqrt(np.mean(x_filtered_interference**2)):.3f} (target: 0.3)")
    
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
    print(f"  1x order: RMS = {np.sqrt(np.mean(x_filtered_1x**2)):.3f} (target: {amp_1x:.1f}), MSE = {error_1x:.2e}")
    print(f"  2x order: RMS = {np.sqrt(np.mean(x_filtered_2x**2)):.3f} (target: {amp_2x:.1f}), MSE = {error_2x:.2e}")
    print(f"  3x order: RMS = {np.sqrt(np.mean(x_filtered_3x**2)):.3f} (target: {amp_3x:.1f}), MSE = {error_3x:.2e}")
    print(f"  Interference (70-30 Hz): RMS = {np.sqrt(np.mean(x_filtered_interference_multi**2)):.3f} (target: 0.3)")
    
    
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
    ax.set_xlim(0, 5)
    
    # Case 2: Time-varying frequency
    ax = axes[1]
    ax.plot(t, x_clean_chirp, 'g-', label='Clean Signal (50-150 Hz chirp)', linewidth=2, alpha=0.7)
    ax.plot(t, x_noisy_chirp, 'gray', label='Noisy Signal', linewidth=0.5, alpha=0.5)
    ax.plot(t, x_filtered_wrong, 'r--', label='Problem: Fixed BW=20Hz (narrows to 13%)', linewidth=1.5, alpha=0.8)
    ax.plot(t, x_filtered_correct, 'b-', label='Solution: Adaptive BW=10%*freq (constant 10%)', linewidth=2, alpha=0.8)
    ax.set_ylabel('Amplitude')
    ax.set_title('Case 2: Time-Varying Frequency - Use Frequency-Adaptive Bandwidth')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)
    
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
    plt.savefig('demo_plots/vkf_example.png', dpi=150)
    print("  Comparison plot: demo_plots/vkf_example.png")
    
    # ========================================================================
    # Spectrogram plots
    # ========================================================================
    
    try:
        from flight_acoustics import plot_spectrogram
        
        print("\nGenerating spectrograms...")
        
        # Spectrogram of clean signal
        plot_spectrogram(x_clean_chirp, fs, window_time=0.2, title='Spectrogram: Clean Chirp Signal (50-150 Hz)', clim=[20, 80], flim=[0, 400])
        plt.savefig('demo_plots/vkf_spectrogram_clean.png', dpi=150)
        plt.close()
        
        # Spectrogram of noisy signal
        plot_spectrogram(x_noisy_chirp, fs, window_time=0.2, title='Spectrogram: Noisy Signal', clim=[20, 80], flim=[0, 400])
        plt.savefig('demo_plots/vkf_spectrogram_noisy.png', dpi=150)
        plt.close()
        
        # Spectrogram of filtered signal (with correct adaptive bandwidth)
        plot_spectrogram(x_filtered_correct, fs, window_time=0.2, title='Spectrogram: Filtered Signal (Adaptive BW=10%*freq)', clim=[20, 80], flim=[0, 400])
        plt.savefig('demo_plots/vkf_spectrogram_filtered.png', dpi=150)
        plt.close()
        
        # Clean multi-order signal (sum of all three orders)
        plot_spectrogram(x_clean_multi, fs, window_time=0.2, title='Clean Multi-Order Signal (1x+2x+3x)', clim=[20, 80], flim=[0, 400])
        plt.savefig('demo_plots/vkf_spectrogram_multiorder_clean.png', dpi=150)
        plt.close()
        
        # Noisy input signal with all orders plus noise and interference
        plot_spectrogram(x_noisy_multi, fs, window_time=0.2, title='Noisy Multi-Order Signal (1x+2x+3x+noise+interference)', clim=[20, 80], flim=[0, 400])
        plt.savefig('demo_plots/vkf_spectrogram_multiorder_noisy.png', dpi=150)
        plt.close()
        
        # Reconstructed signal (sum of all filtered orders)
        x_reconstructed_multi = x_filtered_1x + x_filtered_2x + x_filtered_3x
        plot_spectrogram(x_reconstructed_multi, fs, window_time=0.2, title='Reconstructed Multi-Order Signal (filtered 1x+2x+3x)', clim=[20, 80], flim=[0, 400])
        plt.savefig('demo_plots/vkf_spectrogram_multiorder_filtered.png', dpi=150)
        plt.close()
        
        # Individual extracted orders
        plot_spectrogram(x_filtered_1x, fs, window_time=0.2, title='Extracted: 1x Order', clim=[20, 80], flim=[0, 400])
        plt.savefig('demo_plots/vkf_spectrogram_multiorder_1x.png', dpi=150)
        plt.close()
        
        plot_spectrogram(x_filtered_2x, fs, window_time=0.2, title='Extracted: 2x Order', clim=[20, 80], flim=[0, 400])
        plt.savefig('demo_plots/vkf_spectrogram_multiorder_2x.png', dpi=150)
        plt.close()
        
        plot_spectrogram(x_filtered_3x, fs, window_time=0.2, title='Extracted: 3x Order', clim=[20, 80], flim=[0, 400])
        plt.savefig('demo_plots/vkf_spectrogram_multiorder_3x.png', dpi=150)
        plt.close()
        
        # Extracted interference order
        plot_spectrogram(x_filtered_interference_multi, fs, window_time=0.2, title='Extracted: Interference Order (70-30 Hz)', clim=[20, 80], flim=[0, 400])
        plt.savefig('demo_plots/vkf_spectrogram_multiorder_interference.png', dpi=150)
        plt.close()
        
        print("  Spectrograms: demo_plots/vkf_spectrogram_*.png")
        
    except Exception as e:
        print(f"  Spectrogram generation skipped: {e}")
    
    print("\nAll plots saved to demo_plots/")
    print("="*70)

