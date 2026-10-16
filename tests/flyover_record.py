"""A synthetic hexacopter-style record: rotor tones heard by fixed microphones
as the aircraft flies straight overhead, with the propagation delay and the
Doppler shift."""

import numpy as np
from scipy.io import savemat

SOUND_SPEED = 343.0
MICROPHONES = np.array([[0.0, 0.0, 0.0], [0.0, 20.0, 0.0]])


def flyover_record(fs=8192, duration=4.0, speed=20.0, altitude=30.0, rotor_speeds=(200.0, 215.0),
                   rotor_slope=0.0, orders=(2, 3), no_blade=2, microphones=MICROPHONES, track_step=0.1):
    """Rotors fly along x at ``speed`` (m/s) and ``altitude`` (m), overhead of
    the origin at ``duration / 2`` s; each microphone records ``duration`` s.

    Rotor j turns at ``rotor_speeds[j] + rotor_slope * t`` rad/s and emits
    blade-passage harmonics ``orders`` of amplitude 1/k Pa.

    Returns:
        Tuple (data, emitted): ``data`` as hexacopter_vkf_example.
        load_hexacopter_data returns it at the recorded rate ``fs`` (Hz), with
        the track (every ``track_step`` s) and the rotor speeds (200 Hz)
        covering 1 s more at each end; ``emitted(t)`` the pressure, Pa, the
        rotors emit at times ``t`` (s).
    """
    rotor_speeds = np.asarray(rotor_speeds, dtype=float)

    def phase(t):  # shaft angle, rad, (n, n_rotors)
        t = np.asarray(t, dtype=float)[:, np.newaxis]
        return rotor_speeds * t + 0.5 * rotor_slope * t ** 2

    def emitted(t):
        angle = phase(t)
        return sum(np.cos(k * no_blade * angle).sum(axis=1) / k for k in orders)

    def position(t):
        t = np.asarray(t, dtype=float)
        return np.column_stack([speed * (t - duration / 2), np.zeros_like(t), np.full_like(t, altitude)])

    reception = np.arange(int(round(duration * fs))) / fs
    pressure = np.empty((len(reception), len(microphones)))
    for m, microphone in enumerate(microphones):
        emission = reception - altitude / SOUND_SPEED
        # t_e = t_r - |x_m - x_s(t_e)| / c, solved here independently of
        # ground_plane.emission_times, which --doppler track itself uses.
        for _ in range(50):  # contracts by the Mach number
            emission = reception - np.linalg.norm(microphone - position(emission), axis=1) / SOUND_SPEED
        pressure[:, m] = emitted(emission)
    track_time = np.arange(-1.0, duration + 1.0 + 1e-9, track_step)
    rotor_time = np.arange(-1.0, duration + 1.0 + 1e-9, 0.005)
    data = dict(acoustics_T=reception, acoustics_P=pressure, aircraft_time=rotor_time,
                aircraft_rpm=rotor_speeds + rotor_slope * rotor_time[:, np.newaxis],
                track_time=track_time, track_position=position(track_time),
                mic_location=np.asarray(microphones, dtype=float))
    return data, emitted


def write_flyover_mat(path, mic_types=('ground', 'pole'), **options):
    """Write ``flyover_record(**options)`` as a .mat file laid out like the
    bundled hexacopter record, one ``mic_types`` entry per microphone, and
    return ``path``."""
    data, _ = flyover_record(**options)
    savemat(path, {
        'acoustics': {'T': data['acoustics_T'], 'P': data['acoustics_P']},
        'aircraft': {'T': data['track_time'], 'position': data['track_position'].T,
                     'time_onboard': data['aircraft_time'], 'rpm': data['aircraft_rpm']},
        'info': {'micLocation': data['mic_location'].T, 'micType': np.array(mic_types, dtype=object)},
    })
    return path
