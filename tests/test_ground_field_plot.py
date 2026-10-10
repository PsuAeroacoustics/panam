import numpy as np

import board_field_plot as bfp
import ground_plane as gp


def test_plot_passes_the_ground_and_the_plate_geometry(monkeypatch, tmp_path):
    seen = []

    def level(frequency, elevation, x_m, z_m, sound_speed, flow_resistance, ground=None, **geometry):
        seen.append((ground, geometry))
        zeros = np.zeros(np.shape(x_m))
        return zeros, zeros
    monkeypatch.setattr(bfp, 'level_re_free_field', level)
    ground = dict(model='variable_porosity', sigma_e=200.0, alpha_e=0.0)
    output = bfp.plot(1000.0, 30.0, 225.0, str(tmp_path / 'field.png'), grid=(11, 5), ground=ground,
                      radius=0.5, taper_length=0.0)
    assert (tmp_path / 'field.png').stat().st_size > 0 and output.endswith('field.png')
    assert len(seen) == 2
    assert all(g == ground and geometry == dict(radius=0.5, taper_length=0.0) for g, geometry in seen)


def test_level_re_free_field_uses_the_ground_model():
    # Far above the plate the map is the bare ground's two-wave field, which
    # depends on the ground model given.
    x, z = np.array([0.0]), np.array([3.0])
    soft = bfp.level_re_free_field(2000.0, 30.0, x, z, flow_resistance=225.0)[1]
    ground = dict(model='delany_bazley', sigma=gp.RIGID_FLOW_RESISTANCE)
    hard = bfp.level_re_free_field(2000.0, 30.0, x, z, flow_resistance=225.0, ground=ground)[1]
    assert abs(hard[0] - soft[0]) > 0.1


def test_plot_leaves_the_matplotlib_backend_alone(monkeypatch, tmp_path):
    """plot() called matplotlib.use('Agg'), switching an interactive session's backend."""
    import matplotlib
    import matplotlib.pyplot as plt

    def use(*args, **kwargs):
        raise AssertionError('plot() switched the Matplotlib backend')
    monkeypatch.setattr(matplotlib, 'use', use)
    monkeypatch.setattr(bfp, 'level_re_free_field', lambda *a, **k: (np.zeros(np.shape(a[2])),) * 2)
    figures = plt.get_fignums()
    bfp.plot(1000.0, 30.0, 225.0, str(tmp_path / 'field.png'), grid=(11, 5))
    assert (tmp_path / 'field.png').stat().st_size > 0
    assert plt.get_fignums() == figures
