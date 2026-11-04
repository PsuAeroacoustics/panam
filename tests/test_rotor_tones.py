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
    with pytest.raises(ValueError, match='changes sign'):
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
