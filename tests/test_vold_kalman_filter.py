"""Vold-Kalman order tracking: amplitude recovery over the whole record.

The smoothness operator used to be padded to n_x rows per order with
truncated (one order, p >= 2) or misaligned (several orders) rows that did not
sum to zero.  Weighted by r**2 they penalized the envelope's *level* rather
than its changes, and drove it to zero over the last ~1.5 time constants of
the record (and, for several orders at p >= 2, the first).  It was also
assembled through dense n_x x n_x arrays, so a 10 s record at 2 kHz needed
about 10 GB.
"""

import numpy as np
import pytest

import vold_kalman_filter as vk
from vold_kalman_filter import _compute_weighting_factor, vold_kalman_filter

FS = 1000.0
N = 4000
T = np.arange(N) / FS
F1 = 50.0 + 25.0 * T          # chirp, 50 -> 150 Hz
F2 = 1.2 * F1                 # a second order, 10..30 Hz away


def _signal():
    rng = np.random.default_rng(0)
    return (np.cos(2 * np.pi * np.cumsum(F1) / FS) + 0.5 * np.cos(2 * np.pi * np.cumsum(F2) / FS)
            + 0.2 * rng.standard_normal(N))


@pytest.mark.parametrize('p', [1, 2, 3])
def test_single_order_amplitude_holds_to_the_end_of_the_record(p):
    y, phasor, _ = vold_kalman_filter(_signal(), F1, FS, 4.0, p)
    assert y.shape == phasor.shape == (N, 1)
    assert np.max(np.abs(np.abs(y[:, 0]) - 1.0)) < 0.15


@pytest.mark.parametrize('p', [1, 2, 3])
def test_two_orders_hold_their_amplitudes_at_both_ends(p):
    y, _, _ = vold_kalman_filter(_signal(), np.column_stack([F1, F2]), FS, 4.0, p)
    assert np.max(np.abs(np.abs(y[:, 0]) - 1.0)) < 0.15
    assert np.max(np.abs(np.abs(y[:, 1]) - 0.5)) < 0.15


@pytest.mark.parametrize('p', [1, 2])
def test_uncoupled_orders_do_not_depend_on_each_other(p):
    """Without coupling each order's solution must equal its single-order solution."""
    x = _signal()
    single, _, _ = vold_kalman_filter(x, F1, FS, 4.0, p)
    pair, _, _ = vold_kalman_filter(x, np.column_stack([F1, F2]), FS, 4.0, p, use_coupling=False)
    np.testing.assert_allclose(pair[:, 0], single[:, 0], atol=1e-6)


def test_long_record_stays_sparse():
    """30 000 samples would need a 7 GB dense matrix; sparse it takes milliseconds."""
    n = 30000
    freq = np.full(n, 120.0)
    x = np.cos(2 * np.pi * np.cumsum(freq) / FS)
    y, _, _ = vold_kalman_filter(x, freq, FS, 4.0, 1)
    assert np.max(np.abs(np.abs(y[:, 0]) - 1.0)) < 0.02


def test_unknown_solver_is_an_error_not_a_silent_fallback():
    freq = np.full(200, 120.0)
    with pytest.raises(ValueError, match='solver'):
        vold_kalman_filter(np.cos(2 * np.pi * np.cumsum(freq) / FS), freq, FS, 4.0, 1, solver='bogus')


@pytest.mark.filterwarnings('error::RuntimeWarning')
@pytest.mark.parametrize('fs,p,bandwidth', [(8192.0, 1, 3.0), (25600.0, 2, 5.0), (48000.0, 2, 3.0),
                                            (8192.0, 3, 3.0), (48000.0, 3, 50.0)])
def test_the_band_edge_is_where_the_bandwidth_says(fs, p, bandwidth):
    """A tone half a bandwidth off the tracked order comes through at -3 dB.

    The weighting factor was evaluated through a cosine sum that rounding
    destroys for narrow bands, and a clamp then capped r at 6.4e5: the band
    could not be narrower than 15 Hz for p = 2 at 48 kHz or 26 Hz for p = 3 at
    8 kHz, and a tone at the requested band edge came through almost
    unattenuated (gain 0.94-0.9998).  The exact r needs the better-conditioned
    solve past a condition number of about 1e10.
    """
    n = int(min(max(30 * fs / (np.pi * bandwidth), 4000), 200000))
    t = np.arange(n) / fs
    x = np.cos(2 * np.pi * (1000.0 + bandwidth / 2) * t)
    y, _, _ = vold_kalman_filter(x, np.full(n, 1000.0), fs, bandwidth, p)
    assert np.abs(y[n // 3: 2 * n // 3, 0]).mean() == pytest.approx(1 / np.sqrt(2), abs=0.002)


def test_a_band_too_narrow_to_resolve_says_so():
    freq = np.full(5000, 1000.0)
    x = np.cos(2 * np.pi * np.cumsum(freq) / 8192.0)
    with pytest.warns(RuntimeWarning, match='too narrow'):
        vold_kalman_filter(x, freq, 8192.0, 15.0, 6)


@pytest.mark.filterwarnings('error::RuntimeWarning')
@pytest.mark.parametrize('fs,p,bandwidth,r', [(8192.0, 6, 3.0, None), (48000.0, 2, 1e-73, None),
                                              (48000.0, 1, None, 1e200)])
def test_a_band_too_narrow_to_solve_is_refused(fs, p, bandwidth, r):
    """Past a condition number of 1/eps the solve is singular to double
    precision and its output is noise (3 Hz at p = 6); further out r
    overflows, to NaN output (1e-73 Hz at p = 2) or an OverflowError
    (r = 1e200).  Such bands are refused rather than answered."""
    freq = np.full(5000, 1000.0)
    x = np.cos(2 * np.pi * np.cumsum(freq) / fs)
    with pytest.raises(ValueError, match='too narrow'):
        vold_kalman_filter(x, freq, fs, bandwidth, p, r=r)


def test_equivalent_bandwidth_and_r_shapes_agree():
    """Per-order values given as (n_orders,), (n_orders, 1), (1, n_orders) or
    (n_samples, n_orders), as bandwidths or as the matching r, describe the same
    filter.  Several of these used to be mis-ordered or to give all-NaN output."""
    n, fs = 3000, 1000.0
    t = np.arange(n) / fs
    freq = np.column_stack([np.full(n, 100.0), np.full(n, 160.0)])
    x = np.cos(2 * np.pi * 100.0 * t) + 0.5 * np.cos(2 * np.pi * 160.0 * t)
    reference, _, _ = vold_kalman_filter(x, freq, fs, np.array([4.0, 40.0]), 1)
    r_pair = _compute_weighting_factor(np.pi * np.array([4.0, 40.0]) / fs, 1)
    for kwargs in ({'bandwidth': np.array([[4.0], [40.0]])},
                   {'bandwidth': np.array([[4.0, 40.0]])},
                   {'bandwidth': np.tile([4.0, 40.0], (n, 1))},
                   {'bandwidth': None, 'r': r_pair},
                   {'bandwidth': None, 'r': np.tile(r_pair, (n, 1))}):
        y, _, _ = vold_kalman_filter(x, freq, fs, p=1, **kwargs)
        np.testing.assert_allclose(y, reference, atol=1e-12)
    scalar, _, _ = vold_kalman_filter(x, freq[:, 0], fs, None, 1, r=50.0)
    zero_d, _, _ = vold_kalman_filter(x, freq[:, 0], fs, None, 1, r=np.array(50.0))
    np.testing.assert_allclose(zero_d, scalar, atol=1e-12)


def test_a_per_order_row_stays_per_order_with_as_many_samples_as_orders():
    """A (1, n_orders) row is one value per order even when there are as many
    samples as orders; it used to be flattened and taken per sample.  The
    orders are fitted separately, which keeps each fit well posed with so few
    samples."""
    n, fs = 6, 1000.0
    freq = np.tile([100.0, 160.0, 220.0, 280.0, 340.0, 400.0], (n, 1))
    x = np.cos(2 * np.pi * 100.0 * np.arange(n) / fs)
    per_order = np.array([1.0, 450.0, 1.0, 450.0, 1.0, 450.0])
    reference, _, _ = vold_kalman_filter(x, freq, fs, np.tile(per_order, (n, 1)), 1, use_coupling=False)
    as_row, _, _ = vold_kalman_filter(x, freq, fs, per_order[None, :], 1, use_coupling=False)
    np.testing.assert_allclose(as_row, reference, atol=1e-12)
    per_sample, _, _ = vold_kalman_filter(x, freq, fs, per_order[:, None], 1, use_coupling=False)
    assert np.max(np.abs(per_sample - reference)) > 0.1


@pytest.mark.parametrize('bandwidth', [0.0, -3.0, np.nan, 1000.0, [4.0, 5.0, 6.0]])
def test_invalid_bandwidth_is_rejected(bandwidth):
    n = 3000
    freq = np.column_stack([np.full(n, 100.0), np.full(n, 160.0)])
    x = np.cos(2 * np.pi * 100.0 * np.arange(n) / 1000.0)
    with pytest.raises(ValueError, match='bandwidth'):
        vold_kalman_filter(x, freq, 1000.0, bandwidth, 1)


@pytest.mark.skipif(vk._HAVE_UMFPACK, reason='scikits.umfpack is installed')
def test_umfpack_without_scikits_umfpack_is_an_error():
    """SciPy quietly solves with SuperLU when asked for UMFPACK it does not have."""
    freq = np.full(200, 120.0)
    with pytest.raises(RuntimeError, match='umfpack'):
        vold_kalman_filter(np.cos(2 * np.pi * np.cumsum(freq) / FS), freq, FS, 4.0, 1, solver='umfpack')


def test_pardiso_is_never_handed_a_complex_system(monkeypatch):
    """pypardiso solves only real systems and these are complex: "auto" used to
    try it first anyway, and "pardiso" passed it the complex matrix."""
    calls = []
    monkeypatch.setattr(vk, '_HAVE_PARDISO', True)
    monkeypatch.setattr(vk, '_pardiso_spsolve', lambda A, b: calls.append(A), raising=False)
    x = _signal()[:1000]
    freq = np.column_stack([F1, F2])[:1000]
    expected, _, _ = vold_kalman_filter(x, freq, FS, 4.0, 2, solver='superlu')
    y, _, _ = vold_kalman_filter(x, freq, FS, 4.0, 2, solver='auto')
    np.testing.assert_allclose(y, expected, rtol=0, atol=1e-7)     # banded Cholesky vs sparse LU
    with pytest.raises(TypeError, match='only real'):
        vold_kalman_filter(x, freq, FS, 4.0, 2, solver='pardiso')
    assert calls == []


@pytest.mark.parametrize('p,coupled,n_ord', [(1, True, 2), (2, True, 2), (3, True, 2), (1, False, 2),
                                             (2, False, 1), (1, True, 6)])
def test_the_banded_solve_matches_the_sparse_lu(monkeypatch, p, coupled, n_ord):
    """"auto" solves the normal equations by a banded Cholesky factorization;
    it must agree with the assembled sparse system to within its conditioning,
    and its residual must be as small."""
    calls = []
    banded_solve = vk._solve_normal_banded
    monkeypatch.setattr(vk, '_solve_normal_banded', lambda *args: calls.append(1) or banded_solve(*args))
    x = _signal()[:1500]
    freq = np.column_stack([F1 * (1.0 + 0.2 * i) for i in range(n_ord)])[:1500]
    bandwidth = 4.0 if p < 3 else 20.0         # keeps p = 3 off the augmented formulation
    banded, _, banded_residual = vold_kalman_filter(x, freq, FS, bandwidth, p, use_coupling=coupled)
    assert calls == [1]
    sparse, _, sparse_residual = vold_kalman_filter(x, freq, FS, bandwidth, p, use_coupling=coupled,
                                                    solver='superlu')
    # Both are off the exact solution by up to ~4e-7 of the peak at p = 3
    # (checked against an iteratively refined solve), the banded one less.
    np.testing.assert_allclose(banded, sparse, rtol=0, atol=1e-6 * np.abs(sparse).max())
    assert np.linalg.norm(banded_residual) < 10.0 * np.linalg.norm(sparse_residual) + 1e-12


def test_a_matrix_the_banded_solve_rejects_goes_to_the_sparse_lu(monkeypatch):
    def not_positive_definite(*args, **kwargs):
        raise np.linalg.LinAlgError('2th leading minor not positive definite')
    x = _signal()[:1000]
    freq = np.column_stack([F1, F2])[:1000]
    expected, _, _ = vold_kalman_filter(x, freq, FS, 4.0, 1, solver='superlu')
    monkeypatch.setattr(vk, 'solveh_banded', not_positive_definite)
    y, _, _ = vold_kalman_filter(x, freq, FS, 4.0, 1)
    np.testing.assert_array_equal(y, expected)


def test_the_size_caches_are_bounded():
    """One entry per record length, ~24 MB each at 1e5 samples, used to
    accumulate without limit."""
    for n in range(300, 1300, 100):
        freq = np.column_stack([np.full(n, 100.0), np.full(n, 160.0)])
        vold_kalman_filter(np.cos(2 * np.pi * 100.0 * np.arange(n) / FS), freq, FS, 4.0, 1,
                           solver='superlu')
    for cached in (vk._smoothness_operator, vk._get_bu_index_cache):
        assert 0 < cached.cache_info().currsize <= vk._CACHE_SIZES
