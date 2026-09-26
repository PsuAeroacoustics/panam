"""Vold-Kalman order tracking: amplitude recovery over the whole record.

The smoothness operator used to be padded to n_x rows per order with
truncated (one order, p >= 2) or misaligned (several orders) rows that did not
sum to zero.  Weighted by r**2 they penalised the envelope's *level* rather
than its changes, and drove it to zero over the last ~1.5 time constants of
the record (and, for several orders at p >= 2, the first).  It was also
assembled through dense n_x x n_x arrays, so a 10 s record at 2 kHz needed
about 10 GB.
"""

import numpy as np
import pytest

from vold_kalman_filter import vold_kalman_filter

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
