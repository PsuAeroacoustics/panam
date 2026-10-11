"""ground_plane.fit_ground_impedance: recover a variable-porosity ground from synthetic pole levels."""
import numpy as np
import pytest

import ground_plane as gp

C = 1120.0                                              # ft/s
BANDS = np.asarray(gp.fa.PNL_BAND_FREQUENCIES)          # 50 Hz to 10 kHz
HS = np.array([30.0, 60.0, 100.0, 200.0, 400.0])        # source height, ft
D2 = np.array([600.0, 1200.0, 300.0, 900.0, 500.0])     # ground distance, ft
POLE = 4.0                                              # ft


def synthetic(sigma_e, alpha_e, noise_db, seed):
    level = gp.pole_level(BANDS, HS, D2, POLE, C, 'variable_porosity', sigma_e=sigma_e, alpha_e=alpha_e)
    return level + np.random.default_rng(seed).normal(0.0, noise_db, level.shape)


@pytest.mark.parametrize('sigma_e, alpha_e', [(50.0, 10.0), (150.0, 30.0), (400.0, 80.0)])
def test_recovers_both_parameters_from_noisy_levels(sigma_e, alpha_e):
    measured = synthetic(sigma_e, alpha_e, 0.5, seed=0)
    fit = gp.fit_ground_impedance(BANDS, HS, D2, POLE, C, measured)
    se = fit['standard_error']
    assert fit['success'] and not fit['at_bound'] and fit['fitted'] == ('sigma_e', 'alpha_e')
    assert abs(fit['sigma_e'] - sigma_e) < 3.0 * se[0] and abs(fit['sigma_e'] / sigma_e - 1.0) < 0.1
    assert abs(fit['alpha_e'] - alpha_e) < 3.0 * se[1] and abs(fit['alpha_e'] / alpha_e - 1.0) < 0.25
    assert fit['covariance'].shape == (2, 2) and np.allclose(fit['covariance'], fit['covariance'].T)
    # The residual is the fitted prediction minus the data, and its rms is the noise.
    np.testing.assert_allclose(fit['residual'], fit['prediction'] - measured, atol=1e-12)
    assert fit['band_rms'].shape == BANDS.shape and 0.35 < fit['rms'] < 0.65


def test_noise_free_levels_give_the_parameters_exactly():
    fit = gp.fit_ground_impedance(BANDS, HS, D2, POLE, C, synthetic(150.0, 30.0, 0.0, 0), sigma_e=1000.0)
    assert fit['sigma_e'] == pytest.approx(150.0, rel=1e-4)
    assert fit['alpha_e'] == pytest.approx(30.0, rel=1e-4)
    assert fit['rms'] < 1e-4


def test_standard_errors_match_the_scatter_over_noise_draws():
    fits = [gp.fit_ground_impedance(BANDS, HS, D2, POLE, C, synthetic(150.0, 30.0, 0.5, seed))
            for seed in range(24)]
    values = np.array([[f['sigma_e'], f['alpha_e']] for f in fits])
    se = np.mean([f['standard_error'] for f in fits], axis=0)
    ratio = values.std(axis=0, ddof=1) / se
    assert np.all((ratio > 0.6) & (ratio < 1.6))


def test_held_alpha_fits_sigma_alone_including_the_site_ground():
    fit = gp.fit_ground_impedance(BANDS, HS, D2, POLE, C, synthetic(200.0, 0.0, 0.3, 1), alpha_e=0.0)
    assert fit['fitted'] == ('sigma_e',) and fit['alpha_e'] == 0.0
    assert fit['covariance'].shape == (1, 1)
    assert abs(fit['sigma_e'] - 200.0) < 3.0 * fit['standard_error'][0]


def test_soft_l1_resists_outliers():
    measured = synthetic(150.0, 30.0, 0.3, 2)
    rng = np.random.default_rng(3)
    bad = rng.choice(measured.size, size=8, replace=False)
    measured.flat[bad] += rng.choice([-15.0, 15.0], size=bad.size)
    fit = gp.fit_ground_impedance(BANDS, HS, D2, POLE, C, measured)
    assert fit['sigma_e'] == pytest.approx(150.0, rel=0.1)
    assert fit['alpha_e'] == pytest.approx(30.0, rel=0.25)


def test_nan_and_zero_weight_points_are_left_out():
    clean = synthetic(150.0, 30.0, 0.0, 0)
    measured = clean.copy()
    measured[3, 1] = np.nan
    measured[7, 2] += 40.0
    weights = np.ones_like(measured)
    weights[7, 2] = 0.0
    fit = gp.fit_ground_impedance(BANDS, HS, D2, POLE, C, measured, weights=weights)
    assert fit['n'] == measured.size - 2
    assert np.isnan(fit['residual'][3, 1]) and np.isnan(fit['residual'][7, 2])
    assert fit['sigma_e'] == pytest.approx(150.0, rel=1e-4)
    assert np.all(np.isfinite(fit['band_rms']))


def test_unknown_target_is_rejected():
    with pytest.raises(ValueError, match='target'):
        gp.fit_ground_impedance(BANDS, HS, D2, POLE, C, synthetic(150.0, 30.0, 0.0, 0), target='plate')
