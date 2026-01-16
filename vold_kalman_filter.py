"""
Multi-shaft Vold-Kalman filter implementation.

This module implements the multi-shaft Vold-Kalman filter for extracting
acoustic signal components at specific frequencies/orders.

Functions
---------
vold_kalman_filter(x, freq, fs, bandwidth, p, r=None)
    Extract complex envelopes and phasors from acoustic signal using Vold-Kalman filtering.
"""

import numpy as np
from scipy.sparse import spdiags, eye as speye, block_diag, csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.special import comb


def vold_kalman_filter(x, freq, fs, bandwidth, p, r=None):
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
    fs : float
        Sampling frequency of the acoustic signal and frequency vector, Hz
    bandwidth : float or ndarray
        Bandwidth for weighting factor formulation, Hz.
        Can be:
        - scalar: single bandwidth for all orders
        - vector with same length as x: time-varying bandwidth
        - vector with length equal to number of orders: order-wise bandwidth
        - array same shape as freq: time and order-varying bandwidth
    p : int or array-like
        Filter order(s). If array, can contain multiple orders (e.g., [1, 2])
    r : float or ndarray, optional
        Weighting factor for the filter. If not provided, computed from bandwidth.
        Default is None (compute from bandwidth).

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

    Examples
    --------
    >>> x = np.sin(2*np.pi*100*np.arange(1000)/1000)
    >>> freq = 100 * np.ones(1000)
    >>> y, phasor = vold_kalman_filter(x, freq, 1000, 10, 1)
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
    p_p_orig = int(np.max(p_arr))
    p_p = p_p_orig

    # Defining the matrix of linear equations
    df = np.setdiff1d(p_arr, p_p)
    nr = len(df)

    def _diff_coeff(order: int) -> np.ndarray:
        """Binomial finite-difference coefficients with alternating sign."""
        return np.array([((-1) ** k) * comb(order, k) for k in range(order + 1)], dtype=float)

    # Main difference coefficients for order p_p
    diff_main = _diff_coeff(p_p)

    # Boundary coefficients use lower-order padding (df). If no lower order,
    # fall back to first-order difference.
    if nr > 0:
        boundary_order = int(df[0])
    else:
        boundary_order = 1
    boundary_coeff = _diff_coeff(boundary_order)

    # A0: rows built from boundary coefficients (no sign flip beyond diff definition)
    A0_dense = np.zeros((1 if boundary_coeff.size > 0 else 0, n_x))
    if A0_dense.shape[0] > 0:
        A0_dense[0, :boundary_coeff.size] = boundary_coeff
    
    # A0_end: boundary condition for the end (shifted to last columns with reversed sign)
    # MATLAB shows last row as [0...0, -1, 1] which is [-1, 1] at the end
    # This is the reverse of A0's [1, -1]
    A0_end = np.zeros((1 if boundary_coeff.size > 0 else 0, n_x))
    if A0_end.shape[0] > 0:
        # Reverse the boundary coefficients for the end
        A0_end[0, -boundary_coeff.size:] = boundary_coeff[::-1]

    # Build A as n_x x n_x with constant diagonals then fix first row to boundary coeffs
    diag_offsets = np.arange(0, p_p + 1)
    A_diags = np.zeros((p_p + 1, n_x))
    for i in range(p_p + 1):
        A_diags[i, :] = diff_main[i]

    A_sparse = spdiags(A_diags, diag_offsets, n_x, n_x, format='csr')

    # Do NOT override first row here - let it stay as the natural second-difference row

    # Combine A0 (start), A (middle), and A0_end (end boundary)
    # Convert A_sparse only when needed for vstack
    A_dense = A_sparse.toarray()
    A_combined = np.vstack([A0_dense, A_dense, A0_end])
    
    # Build AA matrix for single or multi-order case
    if n_ord == 1:
        # Single order: simplify by selecting n_x rows from A_combined
        # A_combined shape: (n_x+2, n_x) with rows [A0, A (n_x rows), A0_end]
        # Use first boundary row, middle n_x-2 rows from A, last boundary row
        # Skip rows 1 and n_x from A to get exactly n_x rows total
        row_indices = [0] + list(range(2, n_x)) + [n_x+1]
        AA_dense = A_combined[row_indices, :]
        AA_sparse = csr_matrix(AA_dense)
    else:
        # Multi-order: extract diagonals and replicate
        diag_offsets = list(range(-p_p, p_p + 1))
        
        # Extract diagonals from A_combined more efficiently
        diagonals_list = []
        for offset in diag_offsets:
            diag = np.diagonal(A_combined, offset=offset)
            # Pad to n_x
            padded = np.zeros(n_x)
            padded[:len(diag)] = diag
            diagonals_list.append(padded)
        
        # Build sparse matrix from diagonals for one order
        diagonal_matrix = np.array(diagonals_list).T  # (n_x, n_diags)
        diagonal_rep = np.tile(diagonal_matrix, (n_ord, 1))  # (n_tot, n_diags)
        
        # Build square AA matrix using scipy spdiags with the extracted diagonals
        AA_sparse = spdiags(diagonal_rep.T, diag_offsets, n_tot, n_tot, format='csr')

        # Enforce boundary rows using lil_matrix (more efficient than dense conversion)
        if boundary_coeff.size > 0:
            AA_sparse = AA_sparse.tolil()  # Convert to lil for efficient row modification
            for ord_idx in range(n_ord):
                row_base = ord_idx * n_x
                # Enforce first boundary row: [1, -1, 0, ...] using slice assignment
                AA_sparse[row_base, :] = 0.0
                AA_sparse[row_base, row_base:row_base + len(boundary_coeff)] = boundary_coeff
                # Enforce last boundary row: [0, ..., -1, 1] using slice assignment
                last_row = row_base + n_x - 1
                AA_sparse[last_row, :] = 0.0
                AA_sparse[last_row, last_row - len(boundary_coeff) + 1:last_row + 1] = boundary_coeff[::-1]
            AA_sparse = AA_sparse.tocsr()  # Convert back to csr for efficient arithmetic
    
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
    
    # Create RR matrix (square, matching AA)
    RR = spdiags([weig_r], [0], n_tot, n_tot, format='csr')
    
    # Compute B0 = AA' * (RR²) * AA + I
    # This is the main regularized least squares matrix
    # Optimize: RR² is diagonal so compute directly instead of RR @ RR
    RR_squared = spdiags([weig_r**2], [0], n_tot, n_tot, format='csr')
    B0 = AA_sparse.T @ RR_squared @ AA_sparse + speye(n_tot, format='csr')
    
    # Reshape phasor for cross-coupling computation
    phasor_rs = phasor.reshape(n_tot, 1, order="F").ravel()
    conj_phasor = np.conj(phasor_rs)
    
    # Build off-diagonal coupling matrix B_U (MATLAB lines 249-261)
    # This captures cross-order interactions via phasor products
    if n_ord > 1:
        from scipy.sparse import coo_matrix
        # Build B_U more efficiently using COO format with pre-allocated arrays
        # Calculate total non-zeros: n_x * (n_ord choose 2)
        n_pairs = n_ord * (n_ord - 1) // 2
        total_nnz = n_x * n_pairs
        
        row_indices = np.empty(total_nnz, dtype=int)
        col_indices = np.empty(total_nnz, dtype=int)
        values = np.empty(total_nnz, dtype=complex)
        
        idx = 0
        for i in range(n_ord):
            for j in range(i + 1, n_ord):
                rows = np.arange(i * n_x, (i + 1) * n_x)
                cols = np.arange(j * n_x, (j + 1) * n_x)
                vals = np.conj(phasor[:, i]) * phasor[:, j]
                
                # Direct array assignment (O(1) vs O(n) for extend)
                row_indices[idx:idx + n_x] = rows
                col_indices[idx:idx + n_x] = cols
                values[idx:idx + n_x] = vals
                idx += n_x
        
        B_U = coo_matrix((values, (row_indices, col_indices)), shape=(n_tot, n_tot), dtype=complex)
        B_U = B_U.tocsr()  # Convert to CSR for efficient matrix operations
        
        # Form full B matrix: B = B0 + B_U + B_U' (MATLAB line 264)
        # Keep in sparse format for more efficient solve
        B_mat = B0.astype(complex) + B_U + B_U.conj().T
    else:
        # Single order: no cross-coupling needed
        B_mat = B0.astype(complex)
    x_rv = np.tile(x, n_ord)

    # Calculate Right-hand side of the linear differential equations
    cH_x = conj_phasor * x_rv

    # DEBUG: Print solve inputs
    if False:  # Set to True for debugging
        print(f"B_mat[0,0] = {B_mat[0,0]:.4f}")
        print(f"B_mat is symmetric: {np.allclose(B_mat, B_mat.conj().T)}")
        print(f"cH_x[0:5] = {cH_x[0:5]}")

    # Solve the linear equations using sparse solver
    # spsolve handles both sparse and dense inputs efficiently
    try:
        y_R = spsolve(B_mat, cH_x)
    except Exception:
        # Fallback to dense solve if sparse solver fails
        if hasattr(B_mat, 'toarray'):
            B_mat_dense = B_mat.toarray()
        else:
            B_mat_dense = B_mat
        y_R = np.linalg.solve(B_mat_dense, cH_x)
    
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
    # Example usage with improved test
    import matplotlib.pyplot as plt
    
    fs = 5000  # Sampling frequency
    duration = 1  # seconds
    t = np.arange(0, duration, 1 / fs)
    
    # Create test signal with two time varying frequency components
    freq1 = 50.  + 100*t  # Hz
    freq2 = 500. - 250*t  # Hz
    x_clean = np.sin(2 * np.pi * freq1 * t) + 0.0 * np.sin(2 * np.pi * freq2 * t)
    
    # Add noise
    np.random.seed(42)
    x_noisy = x_clean + 0.3 * np.random.randn(len(t))
    
    # Add a signal with a different frequency to test filter selectivity
    x_noisy += 0.6 * np.sin(2 * np.pi * 400 * t)

    # Set up filter parameters to extract the 100 Hz component
    freq_vec = freq1
    bandwidth = 20  # Hz
    filter_order = 1

    # Apply Vold-Kalman filter
    y, phasor, cost = vold_kalman_filter(x_noisy, freq_vec, fs, bandwidth, filter_order)

    # Reconstruct the filtered signal
    x_filtered = np.real(y[:, 0] * phasor[:, 0])
    
    # Compare with original clean signal component
    x_target = np.sin(2 * np.pi * freq1 * t)
    
    # Plot signals
    plt.figure()
    plt.plot(t, x_noisy, label='Noisy Signal', alpha=0.5)
    plt.plot(t, x_filtered, label='Filtered Signal', linewidth=2)
    plt.plot(t, x_target, label='Target Signal (100 Hz)', linestyle='--', linewidth=2)
    plt.xlim(0, 1)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Vold-Kalman Filter Performance')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Plot PSDs using flight_acoustics module
    from flight_acoustics import psd
    
    freq_clean, psd_clean, level_clean = psd(x_target, fs)
    freq_noisy, psd_noisy, level_noisy = psd(x_noisy, fs)
    freq_filtered, psd_filtered, level_filtered = psd(x_filtered, fs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(freq_clean, psd_clean, label=f'Clean Signal ({level_clean:.1f} dB)', linewidth=2)
    plt.plot(freq_noisy, psd_noisy, label=f'Noisy Signal ({level_noisy:.1f} dB)', alpha=0.7)
    plt.plot(freq_filtered, psd_filtered, label=f'Filtered Signal ({level_filtered:.1f} dB)', linewidth=2, linestyle='--')
    plt.xlim(0, 500)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (dB re 20 µPa)')
    plt.title('Power Spectral Density Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot the spectrogram
    from flight_acoustics import plot_spectrogram
    
    plot_spectrogram(x_noisy, fs, window_time=0.05,title='Spectrogram of Noisy Signal')
    plot_spectrogram(x_filtered, fs, window_time=0.05,title='Spectrogram of Filtered Signal')
    plot_spectrogram(x_target, fs, window_time=0.05,title='Spectrogram of Target Signal')
    plt.show() 


    # Calculate SNR improvement
    noise_power_before = np.mean((x_noisy - x_target)**2)
    noise_power_after = np.mean((x_filtered - x_target)**2)
    snr_improvement_db = 10 * np.log10(noise_power_before / noise_power_after)
    
    print("Complex envelope shape:", y.shape)
    print("Phasor shape:", phasor.shape)
    print("Cost matrix shape:", cost.shape)
    print(f"\nSNR improvement: {snr_improvement_db:.2f} dB")
    print(f"RMS error before filtering: {np.sqrt(noise_power_before):.4f}")
    print(f"RMS error after filtering: {np.sqrt(noise_power_after):.4f}")
    
    # Simple amplitude check - the envelope magnitude should be ~A/2 for signal A*sin(...)
    expected_amplitude = 1.0
    actual_amplitude = np.mean(np.abs(y[len(y)//4:3*len(y)//4, 0]))  # Average over middle half
    print(f"\nExpected envelope amplitude: ~{expected_amplitude/2:.4f}")
    print(f"Actual envelope amplitude: {actual_amplitude:.4f}")
    print(f"Amplitude error: {abs(actual_amplitude - expected_amplitude/2):.4f}")
    
    if snr_improvement_db > 5:
        print("\n✓ Filter is working correctly - significant noise reduction achieved")
    else:
        print("\n✗ Warning: Filter may not be working properly - low SNR improvement")
