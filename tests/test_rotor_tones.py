"""Rotor-locked tones from measured rotor angles (panam.rotor_tones)."""
import numpy as np
import pytest

from panam import rotor_tones as rt
from panam.vold_kalman_filter import vold_kalman_filter

FS = 1024.0
DURATION = 8.0


def _wandering_speeds(rng, n_rotors, base=220.0, spread=(1.00, 1.045, 0.965, 1.07, 0.95, 1.02)):
    """(time, |speed| rad/s per rotor): each rotor's mean ratio to ``base`` plus a slow wander of about 1 %."""
    t = np.arange(0.0, DURATION, 1.0 / FS)
    speeds = []
    for r in range(n_rotors):
        wander = sum(0.004 * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f in (0.13, 0.31, 0.7))
        speeds.append(base * spread[r] * (1.0 + wander))
    return t, np.stack(speeds, axis=1)


def _synthetic(rng, n_rotors=6, harmonics=(1, 2, 3, 4), noise_pa=0.05):
    t, speeds = _wandering_speeds(rng, n_rotors)
    psi = np.cumsum(speeds, axis=0) / FS + rng.uniform(0, 2 * np.pi, n_rotors)
    h = np.asarray(harmonics)
    amplitude = rng.uniform(0.1, 1.0, (n_rotors, h.size)) * np.exp(1j * rng.uniform(-np.pi, np.pi, (n_rotors, h.size)))
    x = np.real(amplitude[None, :, :] * np.exp(1j * psi[:, :, None] * h[None, None, :])).sum(axis=(1, 2))
    return t, psi, h, amplitude, x + noise_pa * rng.standard_normal(t.size)


def test_vkf_phase_track_is_the_frequency_integral_when_they_agree():
    t = np.arange(0.0, 2.0, 1.0 / FS)
    freq = 40.0 + 5.0 * t
    phase = 2 * np.pi * np.cumsum(freq) / FS
    x = np.cos(phase + 0.3)
    y_f, phasor_f, _ = vold_kalman_filter(x, freq, FS, 4.0, 2)
    y_p, phasor_p, _ = vold_kalman_filter(x, None, FS, 4.0, 2, phase=phase)
    assert np.allclose(phasor_f, phasor_p) and np.allclose(y_f, y_p)
    with pytest.raises(ValueError, match='freq is required'):
        vold_kalman_filter(x, None, FS, 4.0, 2)
    with pytest.raises(ValueError, match='phase vector'):
        vold_kalman_filter(x, None, FS, 4.0, 2, phase=phase[:-1])


def test_rotor_azimuth_from_a_scattered_quantized_log():
    rng = np.random.default_rng(3)
    t, speeds = _wandering_speeds(rng, 1)
    psi_true = np.cumsum(speeds[:, 0]) / FS + 1.0
    aligned = 0.4
    log_t = np.arange(0.05, DURATION - 0.05, 1 / 90.0)
    logged = -np.interp(log_t, t, psi_true) + aligned + np.radians(30.0) * rng.standard_normal(log_t.size)
    logged = np.radians(np.round(np.degrees(np.angle(np.exp(1j * logged)))))      # 1 deg steps, wrapped
    logged[1::4] = logged[0::4][:logged[1::4].size]                                # repeated rows
    speed = -np.interp(log_t, t, speeds[:, 0]) * (1 + 0.002 * rng.standard_normal(log_t.size))
    grid, azimuth, diag = rt.rotor_azimuth(log_t, logged, speed, aligned_angle=aligned)
    assert diag['sense'] == -1
    inner = (grid > 0.5) & (grid < DURATION - 0.5)
    # psi_true is the own-sense azimuth from the aligned position: logged = -psi_true + aligned
    error = np.angle(np.exp(1j * (azimuth[inner] - np.interp(grid[inner], t, psi_true))))
    assert np.degrees(np.sqrt(np.mean(error ** 2))) < 8.0
    assert 20.0 < diag['scatter_deg'] < 45.0
    stopped = speed.copy()
    stopped[:5] = 0.5                                   # a stopped rotor's noise: accepted
    assert rt.rotor_azimuth(log_t, logged, stopped)[2]['sense'] == -1
    with pytest.raises(ValueError, match='turns the other way'):
        rt.rotor_azimuth(log_t, logged, np.where(log_t > 4, -speed, speed))


def test_rotor_locked_tones_recover_every_rotor_and_order():
    """Six rotors 3-7 % apart, four orders each: the joint filter recovers every amplitude to 0.3 dB and 2 deg; each
    order demodulated alone is several dB off where other rotors' orders are near."""
    rng = np.random.default_rng(7)
    t, psi, h, amplitude, x = _synthetic(rng)
    inner = (t > 1.0) & (t < DURATION - 1.0)
    worst = {}
    for method in ('vkf', 'demodulate'):
        y, residual = rt.rotor_locked_tones(x, FS, psi, h, 3.0, method=method)
        estimate = np.median(y[inner].real, axis=0) + 1j * np.median(y[inner].imag, axis=0)
        level_error = 20 * np.log10(np.abs(estimate) / np.abs(amplitude))
        phase_error = np.degrees(np.angle(estimate / amplitude))
        worst[method] = np.max(np.abs(level_error))
        if method == 'vkf':
            assert worst[method] < 0.3
            assert np.max(np.abs(phase_error)) < 2.0
            assert np.std(residual[inner]) < 0.07          # the noise (0.05 Pa) is what is left
    assert worst['demodulate'] > 1.0


def test_vkf_shares_out_two_rotors_crossing_one_frequency():
    """Two rotors whose speeds cross: the joint filter keeps each rotor's own tone; demodulating each alone lets the
    other leak in as the frequencies meet."""
    t = np.arange(0.0, DURATION, 1.0 / FS)
    speed_a = 220.0 * (1 + 0.01 * np.sin(2 * np.pi * t / DURATION))
    speed_b = 220.0 * (1 - 0.01 * np.sin(2 * np.pi * t / DURATION))
    psi = np.stack([np.cumsum(speed_a) / FS, np.cumsum(speed_b) / FS + 1.3], axis=1)
    a, b = 1.0 * np.exp(0.5j), 0.6 * np.exp(-2.0j)
    x = np.real(a * np.exp(2j * psi[:, 0]) + b * np.exp(2j * psi[:, 1]))
    errors = {}
    for method in ('vkf', 'demodulate'):
        y, _ = rt.rotor_locked_tones(x, FS, psi, [2], 2.0, method=method)
        inner = (t > 1.0) & (t < DURATION - 1.0)
        errors[method] = np.max(np.abs(y[inner, 0, 0] - a))
    assert errors['vkf'] < 0.05 < errors['demodulate']


def test_emission_azimuth_and_received_rate():
    """A hub passing a microphone: the received shaft rate is the emitted one Doppler shifted, 1 / (1 - M cos)."""
    c, v, omega = 340.0, 10.0, 220.0
    t_rx = np.arange(0.0, 4.0, 1.0 / FS)
    grid = np.arange(-1.0, 5.0, 1e-3)

    def hub(times):
        times = np.asarray(times, dtype=float)
        return np.stack([v * (times - 2.0), np.zeros_like(times), np.full_like(times, 15.0)], axis=1)

    t_e, psi_e, distance, direction = rt.emission_azimuth(t_rx, hub, np.zeros(3), c, grid, omega * grid)
    assert np.allclose(np.linalg.norm(hub(t_e) - 0.0, axis=1), distance)
    assert np.allclose(t_rx - t_e, distance / c, rtol=0, atol=1e-9)
    received = rt.received_shaft_rate(psi_e, FS)
    mach_toward = (v / c) * direction[:, 0]           # the hub's velocity toward the microphone over c
    assert np.allclose(received[5:-5], omega / (1.0 - mach_toward[5:-5]), rtol=1e-4)


def test_depropagate_tones_undoes_spreading_absorption_and_the_receiver():
    from panam.acoustics.atmosphere import Atmosphere
    air = Atmosphere(temperature=293.15, pressure=101.325, relative_humidity=50.0)
    rng = np.random.default_rng(1)
    source = rng.uniform(0.5, 2.0, (5, 2)) * np.exp(1j * rng.uniform(-np.pi, np.pi, (5, 2)))
    frequency = np.array([[70.0, 140.0]] * 5) * rng.uniform(0.95, 1.05, (5, 1))
    distance = rng.uniform(10.0, 200.0, 5)
    response = rng.uniform(1.0, 2.0, (5, 1)) * np.exp(1j * rng.uniform(-1, 1, (5, 1)))
    alpha = air.attenuation_coefficient(frequency)
    heard = source * (30.48 / distance[:, None]) * 10 ** (-alpha * distance[:, None] / 20) * response
    back = rt.depropagate_tones(heard, frequency, distance, 30.48, air, response)
    assert np.allclose(back, source, rtol=1e-12)
    beyond = rt.depropagate_tones(heard, frequency, distance, 30.48, air, response, lossless=False)
    assert np.allclose(np.abs(beyond / back), 10 ** (-alpha * 30.48 / 20), rtol=1e-12)
    assert np.allclose(np.angle(beyond / back), 0.0)


@pytest.fixture(scope='module')
def small_plate_table():
    """A coarse plate table (three frequencies, a few directions): enough to pin the conventions."""
    return rt.plate_tone_table([100.0, 400.0, 1000.0], 1125.0, ground=dict(model='delany_bazley', sigma=75.0),
                               elevations=np.array([2.0, 10.0, 30.0, 60.0, 90.0]),
                               azimuths=np.arange(0.0, 360.0, 90.0), radius=8.0 / 12.0,
                               thickness=0.125 / 12.0, mic=(0.0, 0.75 * 8.0 / 12.0),
                               mic_height=0.007 / 0.3048)


def test_plate_tone_response_is_the_band_model_at_a_table_frequency(small_plate_table):
    from panam import axisymmetric_bem as ab
    table = small_plate_table
    hs = np.array([40.0, 15.0, 120.0, 50.0])
    d2 = np.array([30.0, 150.0, 5.0, 60.0])
    dx = np.array([-30.0, 100.0, 3.0, 0.0])
    dy = np.array([0.0, -111.8, 4.0, 60.0])
    for f in (100.0, 400.0, 1000.0):
        p = rt.plate_tone_response(table, np.full(4, f), hs, d2, 1125.0, dx, dy)
        band = ab.board_level(np.array([f]), hs, d2, 1125.0, table, sub_bands=1, source_dx=dx, source_dy=dy)
        assert np.allclose(10 * np.log10(np.abs(p) ** 2), band[0], atol=1e-9)
    # between table frequencies: linear in P_d and P_r, so the response lies between its neighbors' (and is
    # NaN outside the table)
    p = rt.plate_tone_response(table, np.array([250.0, 50.0, 2000.0]), 40.0, 30.0, 1125.0)
    assert np.isfinite(p[0]) and np.isnan(p[1]) and np.isnan(p[2])


def test_orders_solved_in_frequency_groups_are_as_accurate_as_one_joint_solve():
    """Each order of the six rotors in its own group, solved in turn on the pressure less the others' fit: three
    sweeps are as close to the truth as one solve over every order (which costs the cube of the order count); one
    sweep, leaving the other groups' tones in each group's data, is not."""
    rng = np.random.default_rng(11)
    t, psi, h, amplitude, x = _synthetic(rng, harmonics=(1, 2, 3))
    groups = rt._frequency_groups((psi[:, :, None] * h[None, None, :]).reshape(t.size, -1), FS, 12.0)
    assert len(groups) == 3
    inner = (t > 1.0) & (t < DURATION - 1.0)

    def error(**options):
        y, _ = rt.rotor_locked_tones(x, FS, psi, h, 3.0, **options)
        return np.median(np.abs(y[inner] - amplitude[None]))
    joint = error(group_gap_hz=1e6)
    assert error() < 1.1 * joint
    assert error(sweeps=1) > 1.05 * error()


def _directivity(phi_deg, theta_deg):
    """A smooth tone directivity (complex Pa): level falling toward the nose and tail, phase turning with phi."""
    phi, theta = np.radians(phi_deg), np.radians(theta_deg)
    return (0.2 + np.sin(theta) ** 2) * np.exp(1j * (2.0 * phi + 0.5 * theta))


def _passes(rng, n_mics=7):
    """Samples along straight passes: each microphone a line of theta at its own phi."""
    phi = np.repeat(np.linspace(-60.0, 60.0, n_mics), 120) + rng.normal(0, 0.5, n_mics * 120)
    theta = np.tile(np.linspace(20.0, 160.0, 120), n_mics)
    return phi, theta, np.repeat(np.arange(n_mics), 120)


def test_grid_tones_keeps_level_and_phase():
    rng = np.random.default_rng(5)
    phi, theta, mic = _passes(rng)
    a = _directivity(phi, theta)[:, None]
    grid_phi, grid_theta = np.arange(-50.0, 51.0, 10.0), np.arange(30.0, 151.0, 10.0)
    g = rt.grid_tones(phi, theta, a, mic, np.full(phi.size, 2.0), grid_phi, grid_theta)
    gphi, gtheta = np.meshgrid(grid_phi, grid_theta, indexing='ij')
    truth = _directivity(gphi, gtheta)
    level = 20 * np.log10(np.abs(truth) / np.sqrt(2) / rt.P_REF)
    assert not g['gap'].any()
    assert np.max(np.abs(g['level_db'][..., 0] - level)) < 0.5
    assert np.max(np.abs(np.angle(np.exp(1j * np.radians(g['phase_deg'][..., 0])) / truth * np.abs(truth),
                                  deg=True))) < 6.0
    assert np.min(g['coherence']) > 0.9               # the phase turns across the kernel
    # phases scattered at random: the energy mean keeps the level; the coherence says the phase is lost
    scattered = np.abs(a) * np.exp(1j * rng.uniform(-np.pi, np.pi, a.shape))
    s = rt.grid_tones(phi, theta, scattered, mic, np.full(phi.size, 2.0), grid_phi, grid_theta)
    assert np.max(np.abs(s['level_db'] - g['level_db'])) < 1e-9
    assert np.median(s['coherence']) < 0.5
    # a sample with no value for a tone is left out of that tone only
    two = np.concatenate([a, a], axis=1)
    two[::3, 1] = np.nan
    t = rt.grid_tones(phi, theta, two, mic, np.full(phi.size, 2.0), grid_phi, grid_theta)
    assert np.allclose(t['level_db'][..., 0], g['level_db'][..., 0])
    assert np.max(np.abs(t['level_db'][..., 1] - level)) < 0.6


def test_clock_offsets_recover_each_runs_timing():
    """Four runs of one condition and one of another, linked by a run flown at both: each run's tones advanced by
    k delta; the fit recovers every delta (to 50 us) relative to the reference, whole revolutions apart included."""
    rng = np.random.default_rng(9)
    n_cells = 150
    rates = np.array([h * w for w in (205.0, 214.0, 221.0, 199.0, 230.0, 210.0) for h in (1, 2, 3)])
    truth = {'a': 0.0, 'b': 0.0031, 'c': -0.0074, 'd': 0.041, 'e': -0.012}
    directivity = {cond: rng.normal(size=(n_cells, rates.size)) + 1j * rng.normal(size=(n_cells, rates.size))
                   for cond in ('pass', 'hover')}
    entries = []
    for run, cond in (('a', 'pass'), ('b', 'pass'), ('c', 'pass'), ('d', 'pass'), ('d', 'hover'), ('e', 'hover')):
        a = directivity[cond] * np.exp(1j * rates * truth[run])
        a = a + 0.3 * (rng.normal(size=a.shape) + 1j * rng.normal(size=a.shape))
        a[rng.random(n_cells) < 0.3] = np.nan               # cells the run did not reach
        entries.append({'run': run, 'condition': cond, 'amplitude': a, 'rate': rates})
    offsets, components = rt.clock_offsets(entries, reference='a')
    assert components == [['a', 'b', 'c', 'd', 'e']]
    for run, delta in truth.items():
        assert abs(offsets[run] - delta) < 5e-5, (run, offsets[run], delta)


def test_clock_offsets_leave_an_unconstrained_run_at_zero():
    """A run whose cells never meet another's is not constrained: it keeps 0, not the edge of the search."""
    a = np.full((10, 2), np.nan, dtype=complex)
    b = a.copy()
    a[:5] = 1.0
    b[5:] = 1.0
    entries = [{'run': 'a', 'condition': 'x', 'amplitude': a, 'rate': np.array([200.0, 400.0])},
               {'run': 'b', 'condition': 'x', 'amplitude': b, 'rate': np.array([200.0, 400.0])}]
    offsets, _ = rt.clock_offsets(entries, reference='a')
    assert offsets == {'a': 0.0, 'b': 0.0}


def test_tone_noise_is_what_broadband_alone_gives():
    """White noise through the extraction: the rms |amplitude| it leaves on each order is about tone_noise's estimate,
    which errs high where neighboring orders share the noise out (a conservative gate), never low by much."""
    rng = np.random.default_rng(13)
    t, psi, h, amplitude, _ = _synthetic(rng, harmonics=(2, 4))
    noise = 0.3 * rng.standard_normal(t.size)
    y, residual = rt.rotor_locked_tones(noise, FS, psi, h, 3.0)
    inner = (t > 1.0) & (t < DURATION - 1.0)
    frequency = rt.received_shaft_rate(psi, FS)[:, :, None] * h[None, None, :] / (2 * np.pi)
    estimate = rt.tone_noise(residual, FS, frequency[inner], 3.0)
    measured = np.sqrt(np.mean(np.abs(y[inner]) ** 2, axis=0))
    ratio = measured / np.mean(estimate, axis=0)
    assert np.all((ratio > 0.4) & (ratio < 1.3))
    assert 0.7 < np.median(ratio) < 1.1


def _harmonic_truth(degree=3, seed=21):
    rng = np.random.default_rng(seed)
    n = (degree + 1) ** 2
    c = (rng.normal(size=n) + 1j * rng.normal(size=n)) / (1 + np.repeat(np.arange(degree + 1), 2 * np.arange(degree + 1) + 1))
    return c


def test_fit_directivities_recovers_the_field_and_each_runs_timing():
    """Three rotors' orders 2 and 4 seen on three runs, each run late by its own delta (shared by its six tones): the
    fit gives each run's delta (to 50 us) and each tone's field (to a few degrees of phase where it is loud).  One
    run's tone with a phase of its own does not move that run's delta, and shows in its consistency."""
    rng = np.random.default_rng(17)
    deltas = {'a': 0.0, 'b': 0.0013, 'c': -0.0021}
    keys = [(rotor, order) for rotor in range(3) for order in (2, 4)]
    truth = {key: _harmonic_truth(3, 30 + i) for i, key in enumerate(keys)}
    speeds = {0: 215.0, 1: 224.0, 2: 207.0}
    tones = []
    for rotor, order in keys:
        phi, theta, run, a, rate = [], [], [], [], []
        for r, delta in deltas.items():
            p, t, _ = _passes(rng, n_mics=5)
            field = rt.evaluate_directivity(truth[(rotor, order)], 3, p, t)
            k = np.full(p.size, order * speeds[rotor])
            heard = field * np.exp(1j * k * delta)
            if (r, rotor, order) == ('c', 2, 4):
                heard = heard * np.exp(1j * np.radians(120.0))         # this run's tone is not the others'
            noise = 0.03 * np.mean(np.abs(field)) * (rng.normal(size=p.size) + 1j * rng.normal(size=p.size))
            phi.append(p); theta.append(t); run.append(np.full(p.size, r)); a.append(heard + noise); rate.append(k)
        tones.append({'phi': np.concatenate(phi), 'theta': np.concatenate(theta), 'run': np.concatenate(run),
                      'amplitude': np.concatenate(a), 'rate': np.concatenate(rate),
                      'weight': np.ones(sum(x.size for x in phi))})
    fit = rt.fit_directivities(tones, degree=3, smoothing=1e-6, reference='a', max_offset_s=0.004, iterations=6)
    for r, delta in deltas.items():
        assert abs(fit['offsets'][r] - delta) < 5e-5, (r, fit['offsets'][r], delta)
    gp, gt = np.meshgrid(np.arange(-50.0, 51.0, 10.0), np.arange(30.0, 151.0, 10.0), indexing='ij')
    for i, key in enumerate(keys):
        model = rt.evaluate_directivity(fit['coefficients'][i], 3, gp, gt)
        true = rt.evaluate_directivity(truth[key], 3, gp, gt)
        loud = np.abs(true) > 0.3 * np.abs(true).max()
        assert np.max(np.abs(np.angle(model[loud] / true[loud], deg=True))) < 5.0, key
    bad = keys.index((2, 4))
    assert fit['consistency'][(bad, 'c')] < 0.6
    assert min(v for (i, r), v in fit['consistency'].items() if (i, r) != (bad, 'c')) > 0.97


def test_summed_directivities_separate_rotors_a_fraction_of_a_hertz_apart():
    """Three rotors, two of them 0.5 % apart in speed (0.3 Hz at order 2), heard only as their sum: the fit gives
    each rotor's directivity (to a few degrees of phase where it is loud) and each run's timing, because their
    relative phases drift over a pass."""
    rng = np.random.default_rng(23)
    speeds = np.array([215.0, 216.1, 231.0])
    truth = [_harmonic_truth(3, 40 + r) for r in range(3)]
    deltas = {'a': 0.0, 'b': 0.0011, 'c': -0.0017}
    order = 2
    blocks = []
    for run, delta in deltas.items():
        p, t, mic = _passes(rng, n_mics=5)
        for m in np.unique(mic):
            sel = mic == m
            n = sel.sum()
            time = np.linspace(0.0, 30.0, n)
            psi = speeds[None, :] * time[:, None] + rng.uniform(0, 2 * np.pi, 3)
            phi, theta = np.repeat(p[sel, None], 3, axis=1), np.repeat(t[sel, None], 3, axis=1)
            rate = np.repeat(order * speeds[None, :], n, axis=0)
            z = sum(rt.evaluate_directivity(truth[r], 3, phi[:, r], theta[:, r])
                    * np.exp(1j * (order * psi[:, r] + rate[:, r] * delta)) for r in range(3))
            z = z + 0.02 * (rng.normal(size=n) + 1j * rng.normal(size=n))
            blocks.append({'z': z, 'gain': np.ones((n, 3)), 'carrier': order * psi, 'rate': rate, 'phi': phi,
                           'theta': theta, 'weight': np.ones(n), 'run': run})
    fit = rt.fit_summed_directivities([{'sources': 3, 'blocks': blocks}], degree=3, smoothing=1e-6, reference='a',
                                      max_offset_s=0.004, iterations=6)
    for run, delta in deltas.items():
        assert abs(fit['offsets'][run] - delta) < 5e-5, (run, fit['offsets'][run], delta)
    gp, gt = np.meshgrid(np.arange(-50.0, 51.0, 10.0), np.arange(30.0, 151.0, 10.0), indexing='ij')
    for r in range(3):
        model = rt.evaluate_directivity(fit['coefficients'][0][r], 3, gp, gt)
        true = rt.evaluate_directivity(truth[r], 3, gp, gt)
        loud = np.abs(true) > 0.3 * np.abs(true).max()
        assert np.max(np.abs(np.angle(model[loud] / true[loud], deg=True))) < 5.0, r
        assert np.max(np.abs(20 * np.log10(np.abs(model[loud] / true[loud])))) < 0.5, r
    assert min(fit['explained'].values()) > 0.99


def _rotating_field(order, k, degree, seed):
    """A hover rotor's tone: Y_l^k only (a field that turns with the rotor), random polar shape."""
    rng = np.random.default_rng(seed)
    nb = (degree + 1) ** 2
    ks = np.concatenate([np.arange(-l, l + 1) for l in range(degree + 1)])
    c = np.zeros(nb, dtype=complex)
    c[ks == k] = rng.normal(size=(ks == k).sum()) + 1j * rng.normal(size=(ks == k).sum())
    return c


def test_rotating_fields_predict_azimuths_never_measured():
    """Three hover rotors (two 0.5 % apart, opposite senses), heard as one order-2 sum at microphones on one side
    only (phi 10 to 60 deg) at several polar angles: with each rotor's one azimuthal order the fit predicts the other
    side (phi -60 to -10 deg) and ahead and behind, where the free fit cannot."""
    rng = np.random.default_rng(29)
    degree, order = 6, 2
    senses = (1, -1, 1)
    speeds = np.array([215.0, 216.1, 231.0])
    truth = [_rotating_field(order, order * s, degree, 50 + r) for r, s in enumerate(senses)]

    def blocks(phi_range, theta_range, n_mics, run):
        out = []
        for phi in np.linspace(*phi_range, n_mics):
            theta = np.linspace(*theta_range, 150)
            n = theta.size
            ph = np.full((n, 3), phi)
            th = np.repeat(theta[:, None], 3, axis=1)
            psi = speeds[None, :] * np.linspace(0.0, 30.0, n)[:, None] + rng.uniform(0, 2 * np.pi, 3)
            z = sum(rt.evaluate_directivity(truth[r], degree, ph[:, r], th[:, r]) * np.exp(1j * order * psi[:, r])
                    for r in range(3))
            out.append({'z': z + 0.02 * (rng.normal(size=n) + 1j * rng.normal(size=n)), 'gain': np.ones((n, 3)),
                        'carrier': order * psi, 'rate': np.repeat(order * speeds[None, :], n, axis=0), 'phi': ph,
                        'theta': th, 'weight': np.ones(n), 'run': run})
        return out
    train = blocks((10.0, 60.0), (40.0, 140.0), 6, 'a')
    fits = {}
    for label, orders in (('rotating', [order * s for s in senses]), ('free', None)):
        problem = {'sources': 3, 'blocks': [dict(b) for b in train]}
        if orders is not None:
            problem['azimuthal_orders'] = orders
        fits[label] = rt.fit_summed_directivities([problem], degree=degree, smoothing=1e-6, iterations=0, robust=0)
    gp, gt = np.meshgrid(np.arange(-60.0, -9.0, 10.0), np.arange(50.0, 131.0, 10.0), indexing='ij')
    for r in range(3):
        true = rt.evaluate_directivity(truth[r], degree, gp, gt)
        rot = rt.evaluate_directivity(fits['rotating']['coefficients'][0][r], degree, gp, gt)
        free = rt.evaluate_directivity(fits['free']['coefficients'][0][r], degree, gp, gt)
        loud = np.abs(true) > 0.3 * np.abs(true).max()
        assert np.max(np.abs(np.angle(rot[loud] / true[loud], deg=True))) < 5.0, r
        assert np.max(np.abs(20 * np.log10(np.abs(rot[loud] / true[loud])))) < 0.5, r
        assert np.median(np.abs(free[loud] - true[loud]) / np.abs(true[loud])) > 0.3, r
    with pytest.raises(ValueError, match='azimuthal_orders'):
        rt.fit_summed_directivities([{'sources': 3, 'blocks': train, 'azimuthal_orders': [2, -2]}], degree=degree)


def test_incoherent_directivities_recover_random_phase_energy():
    """Sources radiating at random phase (each sample's phase drawn afresh) over a known axisymmetric energy
    directivity: the fit to what the phase-locked fit leaves recovers each source's I(polar), and a source without
    one comes back with none."""
    rng = np.random.default_rng(11)
    R, n = 3, 40000
    knots = np.arange(0.0, 181.0, 15.0)
    truth = [lambda p: 2.0 + np.cos(p) ** 2, lambda p: 1.0 + 0.0 * p, lambda p: 0.0 * p]
    polar = rng.uniform(0.0, np.radians(80.0), (n, R))
    azimuth = rng.uniform(-np.pi, np.pi, (n, R))
    # the ART angles of each sample's direction from each source (axis_angles inverted)
    forward, starboard, down = np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)
    theta = np.degrees(np.arccos(forward))
    phi = np.degrees(np.arctan2(starboard, down))
    gain = rng.uniform(0.5, 2.0, (n, R)) * np.exp(1j * rng.uniform(0, 2 * np.pi, (n, R)))
    z = np.zeros(n, complex)
    for r in range(R):
        z += gain[:, r] * np.sqrt(truth[r](polar[:, r])) * np.exp(1j * rng.uniform(0, 2 * np.pi, n))
    blocks = [{'z': z[i::4], 'gain': gain[i::4], 'carrier': np.zeros((n // 4, R)), 'rate': np.ones((n // 4, R)),
               'phi': phi[i::4], 'theta': theta[i::4], 'weight': np.ones(n // 4), 'run': 'a'} for i in range(4)]
    problems = [{'sources': R, 'blocks': blocks}]
    degree = 2
    fit = {'coefficients': [np.zeros((R, (degree + 1) ** 2), complex)], 'degree': degree, 'offsets': {'a': 0.0}}
    result = rt.fit_incoherent_directivities(problems, fit, knots_deg=knots)
    sample = np.radians(np.arange(5.0, 80.0, 10.0))
    for r in range(R):
        got = rt.energy_basis(sample, np.radians(knots)) @ result['coefficients'][0][r]
        want = truth[r](sample)
        assert np.all(got >= 0.0)
        assert np.max(np.abs(got - want)) < 0.25 * max(want.max(), 1.0), (r, got, want)
    assert result['fraction'][0] == pytest.approx(1.0, abs=0.05)
    level = rt.evaluate_energy_directivity(result['coefficients'][0][0], knots, phi[:5, 0], theta[:5, 0])
    assert np.allclose(level, rt.energy_basis(polar[:5, 0], np.radians(knots)) @ result['coefficients'][0][0])
