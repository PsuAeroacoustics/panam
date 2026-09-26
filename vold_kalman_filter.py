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

import warnings

import numpy as np
from scipy.sparse import bmat, csr_matrix, diags, kron, spdiags, vstack, eye as speye
from scipy.sparse.linalg import spsolve
from scipy.special import comb

# Cache for expensive, size-dependent matrices/indices
_AA_CACHE = {}
_BU_INDEX_CACHE = {}

#: The regularised normal equations  (I + AA' R^2 AA) a = C^H x  have a
#: condition number of about 1 + r**2 * 4**p.  Up to this value they are solved
#: as they are; beyond it rounding erodes their accuracy (by 1e16 the identity
#: is lost next to the r**2 terms and the result is garbage), so the equivalent
#: augmented system, conditioned like the square root of that, is solved instead.
_NORMAL_EQUATIONS_MAX_CONDITION = 1e10

#: Warn when even the augmented system (condition ~ sqrt(1 + r**2 * 4**p)) is
#: past this: the requested band is too narrow for this sample rate and p.
#: Measured band-edge gain errors: 0.15 % at 5e13; for p = 4, 6 and 8,
#: 0.2-1.1 % at 1e14, 0.2-2.4 % at 1e15 and 0.2-5.6 % at 3e15.
_AUGMENTED_MAX_CONDITION = 1e14

#: Refuse past this, where the augmented system is singular to double precision:
#: the measured errors are 8-37 % at 1e16 and erratic, up to 100 %, beyond.
#: Far enough out r overflows, which gave NaN output or an OverflowError.
_AUGMENTED_SINGULAR_CONDITION = 1.0 / np.finfo(float).eps

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
        Full -3 dB width of the passband around each tracked order, Hz: a
        component bandwidth/2 away from the order comes through at 1/sqrt(2).
        Must be positive and below fs.  Can be:
        - scalar: single bandwidth for all orders
        - vector with same length as x: time-varying bandwidth
        - vector with length equal to number of orders: order-wise bandwidth
        - array same shape as freq: time and order-varying bandwidth
        An (n_samples, 1) or (1, n_orders) array broadcasts against freq; any
        other vector as long as both x and the orders is taken as time-varying.
    p : int
        Structural filter order (order of difference operator for regularization).
        NOT the harmonic order to track (those are specified by freq).
        It sets how sharply the passband falls off: a component Omega
        rad/sample from the order comes through with gain
        1 / (1 + r**2 (2 sin(Omega/2))**(2p)), which outside the band drops
        by about 12p dB per octave.  With a 5 Hz band, a steady tone 1.5
        bandwidths away is rejected by 14 dB at p=1, 31 dB at p=2 and 50 dB at
        p=3, so higher p separates close orders better, at the cost of
        slightly larger errors near the ends of the record.  Narrow bands at
        high sample rates with p >= 2 are solved through a better-conditioned
        formulation, automatically, at 2-5 times the cost.  Bands narrower
        still lose accuracy (a RuntimeWarning says so) and then cannot be
        resolved at all (a ValueError): at p=3, below about 1.2e-5 and 3.3e-6
        times fs, or 0.6 and 0.16 Hz at 48 kHz.  Decimate the signal first,
        or use a lower p.
    r : float or ndarray, optional
        Weighting factor for the filter, used instead of bandwidth when given;
        same shapes as bandwidth.  Default is None (compute from bandwidth).
    solver : {"auto", "pardiso", "umfpack", "superlu"}, optional
        Sparse solver backend. "auto" prefers Pardiso (if installed), then UMFPACK,
        and falls back to SuperLU. Default is "auto".
    use_coupling : bool, optional
        Whether to include cross-order coupling (B_U) terms. Default True.
        Setting False can speed up solves but changes results.

    Returns
    -------
    y : ndarray
        Complex amplitude of each order, shape (n_samples, n_orders): twice
        the complex envelope, so abs(y) is the order's amplitude
    phasor : ndarray
        Complex phasor corresponding to each order, shape (n_samples, n_orders)
    cost_Mat : ndarray
        Residual of the solved normal equations, shape (n_samples * n_orders,)

    Notes
    -----
    Waveforms can be obtained by multiplying complex envelopes and phasors:
        waveform = np.real(y * phasor)

    This algorithm tracks acoustic signals produced by shafts/orders given in the
    reference frequency vector.

    If weighting factor is not specified, bandwidth is used to compute a desired
    weighting factor for the signal.
    
    **Choosing the bandwidth**

    The bandwidth trades rejection against tracking speed.  A narrower band
    passes less noise and less of any neighbouring component; a wider one
    follows changes in the order's amplitude sooner.  After a step in
    amplitude the envelope takes about 0.7 / bandwidth seconds to rise from
    10 % to 90 % of the change (130 ms at 5 Hz), with 3-6 % overshoot at
    p=2 and 3.

    - Constant frequency: a fixed bandwidth, e.g. ``bandwidth = 20`` (Hz).
    - Sweeps: given an accurate frequency track, the phasor follows the
      frequency, so the band does not need to widen with it.  A bandwidth
      proportional to frequency, e.g. ``bandwidth = 0.10 * freq``, suits
      run-ups whose neighbouring components are other orders of the same
      shaft: their spacing grows with shaft speed, so they stay the same
      number of bandwidths away throughout.  A fixed bandwidth narrow enough
      for the closest spacing works as well; it rejects more, but follows
      amplitude changes more slowly.
    - A component too close to reject is better tracked as another order
      (another column of ``freq``, with ``use_coupling=True``) than removed
      by narrowing the band.  With a tone half as strong 4 Hz away, tracking
      both at 5 Hz gave a 0.3 % amplitude error, against 2.8 % tracking
      alone at 1.5 Hz and 17 % alone at 5 Hz.

    Examples
    --------
    >>> # Constant frequency - use fixed bandwidth
    >>> x = np.sin(2*np.pi*100*np.arange(1000)/1000)
    >>> freq = 100 * np.ones(1000)
    >>> y, phasor, _ = vold_kalman_filter(x, freq, 1000, 20, 1)  # BW=20 Hz
    
    >>> # Time-varying frequency, with a bandwidth proportional to it
    >>> freq_chirp = 50 + 100*np.arange(1000)/1000  # 50-150 Hz sweep
    >>> phase = 2 * np.pi * np.cumsum(freq_chirp) / 1000
    >>> x_chirp = np.sin(phase)
    >>> y, phasor, _ = vold_kalman_filter(x_chirp, freq_chirp, 1000, 0.10*freq_chirp, 1)
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

    # Weighting factor r for every sample and order, given or from the bandwidth
    if use_weight_factor:
        bw = _per_sample_and_order(bandwidth, n_x, n_ord, 'bandwidth')
        if not np.all(np.isfinite(bw)) or np.any(bw <= 0.0) or np.any(bw >= fs):
            raise ValueError('bandwidth (the full -3 dB width, Hz) must be positive and below the sample rate')
        weight = _compute_weighting_factor(bw * np.pi / fs, p_p)
    else:
        weight = _per_sample_and_order(r, n_x, n_ord, 'r')
        if not np.all(np.isfinite(weight)) or np.any(weight < 0.0):
            raise ValueError('r must be finite and non-negative')
    # Each difference row takes the weight of the sample it starts at.
    row_weight = weight[:n_x - p_p].ravel(order="F")
    n_rows = row_weight.size

    solver_choice = (solver or "auto").lower()
    if solver_choice not in ("auto", "pardiso", "umfpack", "superlu"):
        raise ValueError(f"Unknown solver '{solver}'")
    if solver_choice == "pardiso" and not _HAVE_PARDISO:
        raise RuntimeError("pypardiso is not installed")

    # sqrt(1 + r**2 * 4**p), the augmented system's condition number, computed
    # without overflow; r itself is inf for a band far too narrow to resolve.
    with np.errstate(over='ignore'):
        root_condition = float(np.hypot(1.0, np.ldexp(np.max(row_weight, initial=0.0), p_p)))
    if not root_condition <= _AUGMENTED_SINGULAR_CONDITION:
        raise ValueError(
            'the requested bandwidth is too narrow to resolve at this sample rate with p={:d} '
            '(condition number ~{:.0e}); decimate the signal first, or use a lower p{}'
            .format(p_p, root_condition, '' if use_weight_factor else ' or a smaller r'))
    condition = root_condition ** 2
    if condition > _NORMAL_EQUATIONS_MAX_CONDITION:
        if root_condition > _AUGMENTED_MAX_CONDITION:
            warnings.warn(
                'the requested bandwidth is too narrow for this sample rate with p={:d} '
                '(condition number ~{:.0e}), so the envelopes may be inaccurate; decimate '
                'the signal first, or use a lower p'
                .format(p_p, root_condition), RuntimeWarning, stacklevel=2)
        y_R, cost_mat = _solve_augmented(x, phasor, AA_sparse, row_weight,
                                         n_ord > 1 and use_coupling, solver_choice)
        return 2.0 * y_R.reshape((n_x, n_ord), order="F"), phasor, cost_mat

    # Compute B0 = AA' * R^2 * AA + I, the regularised least-squares matrix.
    RR_squared = spdiags([row_weight ** 2], [0], n_rows, n_rows, format='csr')
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

    # Solve the linear equations using sparse solver
    B_mat = B_mat.tocsc()
    y_R = _solve_sparse(B_mat, cH_x, solver_choice)

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


def _per_sample_and_order(value, n_x, n_ord, name):
    """``value`` broadcast to (n_x, n_ord): a scalar, one value per sample,
    one per order, or one per sample and order.  An (n_x, 1) column or a
    (1, n_ord) row broadcasts as numpy would, even with as many samples as
    orders; any other vector as long as both is taken per sample."""
    v = np.asarray(value, dtype=float)
    if v.size == 1:
        return np.full((n_x, n_ord), v.item())
    if v.shape in ((n_x, n_ord), (n_x, 1), (1, n_ord)):
        return np.broadcast_to(v, (n_x, n_ord)).copy()
    if v.ndim == 1 or (v.ndim == 2 and 1 in v.shape):
        flat = v.ravel()
        if flat.size == n_x:
            return np.repeat(flat[:, None], n_ord, axis=1)
        if flat.size == n_ord:
            return np.tile(flat[None, :], (n_x, 1))
    raise ValueError(f"{name} must be a scalar, or have n_samples ({n_x}), n_orders ({n_ord}) "
                     f"or (n_samples, n_orders) values; got shape {v.shape}")


def _solve_sparse(A, b, solver_choice):
    """Solve A z = b with the selected sparse backend (A in CSC)."""
    try:
        if solver_choice in ("auto", "pardiso") and _HAVE_PARDISO:
            return _pardiso_spsolve(A, b)
        try:
            return spsolve(A, b, use_umfpack=_HAVE_UMFPACK and solver_choice != "superlu")
        except TypeError:
            return spsolve(A, b)
    except Exception:
        if solver_choice != "auto":
            raise
        # pypardiso rejects complex matrices, and these systems are complex:
        # fall back to SuperLU, never to a dense solve (n_tot**2 memory).
        return spsolve(A, b, use_umfpack=False)


def _solve_augmented(x, phasor, AA, row_weight, coupled, solver_choice):
    """Solve the Vold-Kalman least-squares problem through its augmented system.

    With M = [C; R AA] and y = [x; 0] the envelopes a minimise |y - M a|**2.
    The normal equations M^H M a = M^H y add the identity to entries of order
    r**2, which rounding swamps for narrow bands at high sample rates.  The
    augmented system

        [[I, M], [M^H, 0]] [s; a] = [y; 0],    s = y - M a,

    has the same solution, never forms R**2 AA' AA, and is conditioned like
    the square root of the normal equations.

    Returns (a, M^H (y - M a)), the latter being the normal-equations residual.
    """
    n_x, n_ord = phasor.shape
    n_tot = n_x * n_ord
    unknowns = np.arange(n_tot)
    values = phasor.ravel(order="F")
    if coupled:
        # Every order fits the one signal together (the cross-order coupling).
        C = csr_matrix((values, (np.tile(np.arange(n_x), n_ord), unknowns)), shape=(n_x, n_tot))
        data = np.asarray(x)
    else:
        # Each order fits the signal on its own.
        C = csr_matrix((values, (unknowns, unknowns)), shape=(n_tot, n_tot))
        data = np.tile(np.asarray(x), n_ord)
    M = vstack((C, spdiags([row_weight], [0], row_weight.size, row_weight.size) @ AA), format="csr")
    y = np.concatenate((data, np.zeros(row_weight.size))).astype(complex)
    m = M.shape[0]
    augmented = bmat([[speye(m), M], [M.conj().T, None]], format="csc")
    solution = _solve_sparse(augmented, np.concatenate((y, np.zeros(n_tot, dtype=complex))), solver_choice)
    a = solution[m:]
    return a, M.conj().T @ (y - M @ a)


def _compute_weighting_factor(bw_rad, p_p):
    """Weighting factor r that puts the filter's half-power points at +-bw_rad.

    A component Omega rad/sample away from the tracked order reaches the
    envelope with gain 1 / (1 + r**2 (2 sin(Omega/2))**(2p)) (Tuma 2005), so r
    follows from setting that to 1/sqrt(2) at Omega = bw_rad, half the full
    bandwidth.  This used to evaluate (2 - 2 cos Omega)**p as a cosine sum:
    for a narrow band that is ~Omega**(2p), far below the rounding error of
    its O(10) terms, and its 1e-12 clamp capped r at 6.4e5, silently widening
    the band (p = 2 at 48 kHz could not go below 15 Hz, p = 3 at 8 kHz below
    26 Hz).

    Parameters
    ----------
    bw_rad : ndarray
        Half the bandwidth in radians per sample (pi * bandwidth_Hz / fs)
    p_p : int
        Filter order

    Returns
    -------
    ndarray
        Weighting factor, same shape as bw_rad; inf where it overflows, for
        bands far too narrow to resolve
    """
    bw_rad = np.asarray(bw_rad, dtype=float)
    with np.errstate(divide='ignore', over='ignore'):
        return np.sqrt((np.sqrt(2.0) - 1.0) / (2.0 * np.sin(0.5 * bw_rad)) ** (2 * p_p))


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
    ax.set_title('Case 2: Time-Varying Frequency, Bandwidth Proportional to Frequency')
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

