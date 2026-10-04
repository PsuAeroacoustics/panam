"""Map the sound field around a ground-plane microphone plate lying on soft ground.

Uses the axisymmetric BEM (:func:`axisymmetric_bem.field`): a rigid plate on a locally
reacting ground, a plane wave from a given elevation, and the total field P_d + Rp P_r
(Rp the ground's plane-wave reflection coefficient: a distant source).  The top panel
maps the level re the free-field wave in the vertical plane through the plate's center
along the incidence direction; the bottom panel is the level along the microphone's
height, with the plate and over the bare ground.

Example::

    python board_field_plot.py --frequency 4000 --elevation 30 --flow-resistance 225 -o field.png

Lengths on the command line are in meters; the plate defaults are
:mod:`ground_plane`'s (0.2 m radius, 8 mm thick, 20 mm taper to a 2.5 mm edge).
"""
import argparse

import numpy as np

import axisymmetric_bem as ab
import ground_plane as gp

FT = 0.3048


def plane_wave_reflection(frequency, elevation, sound_speed, flow_resistance, ground=None):
    """Rp = (sin(el) - beta) / (sin(el) + beta) of the ground."""
    beta = gp._ground_admittance(frequency, sound_speed, flow_resistance, ground)
    s = np.sin(np.radians(elevation))
    return (s - beta) / (s + beta)


def level_re_free_field(frequency, elevation, x_m, z_m, sound_speed=1125.0, flow_resistance=225.0,
                        ground=None, **geometry):
    """Level (dB re the free-field wave) at points (x_m, z_m) in the plane y = 0, with the
    plate (BEM) and over the bare ground.  The wave travels toward +x, down at ``elevation``."""
    x = np.asarray(x_m, float) / FT
    z = np.asarray(z_m, float) / FT
    pts = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
    pd, pr = ab.field(frequency, elevation, 0.0, sound_speed, pts, flow_resistance=flow_resistance,
                      ground=ground, **geometry)
    rp = plane_wave_reflection(frequency, elevation, sound_speed, flow_resistance, ground)
    k = 2 * np.pi * frequency / sound_speed
    el = np.radians(elevation)
    bare = np.exp(1j * k * np.cos(el) * pts[:, 0]) * (np.exp(-1j * k * np.sin(el) * pts[:, 2])
                                                      + rp * np.exp(1j * k * np.sin(el) * pts[:, 2]))
    lev = lambda p: (20 * np.log10(np.abs(p))).reshape(x.shape)
    return lev(pd + rp * pr), lev(bare)


def plot(frequency, elevation, flow_resistance, output, x_range=(-0.4, 0.4), z_top=0.2, grid=(201, 101),
         mic_height_m=gp.PLATE_MIC_HEIGHT_FT * FT, sound_speed=1125.0, levels=(-15.0, 10.0), ground=None,
         **geometry):
    """Draw the map and the level at the microphone's height to ``output``.

    ``ground`` (a :func:`ground_plane.surface_admittance` model dict) replaces
    Delany-Bazley at ``flow_resistance``; ``geometry`` goes to
    :func:`axisymmetric_bem.field` (radius, thickness, edge_thickness,
    taper_length, in feet as there).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    radius = geometry.get('radius', gp.PLATE_RADIUS_FT) * FT
    thickness = geometry.get('thickness', gp.PLATE_THICKNESS_FT) * FT
    edge_thickness = geometry.get('edge_thickness', gp.PLATE_EDGE_THICKNESS_FT) * FT
    taper_length = geometry.get('taper_length', gp.PLATE_TAPER_LENGTH_FT) * FT
    x = np.linspace(*x_range, grid[0])
    z = np.linspace(5e-4, z_top, grid[1])
    X, Z = np.meshgrid(x, z)
    L, _ = level_re_free_field(frequency, elevation, X, Z, sound_speed, flow_resistance, ground, **geometry)
    zm = thickness + mic_height_m + 1e-4           # just clear of the plate's top
    xl = np.linspace(*x_range, 4 * grid[0])
    Ll, Ll0 = level_re_free_field(frequency, elevation, xl, np.full_like(xl, zm), sound_speed, flow_resistance,
                                  ground, **geometry)
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(12.5, 5.6), sharex=True, height_ratios=(2.2, 1.4),
                                 layout='constrained')
    im = ax.pcolormesh(x, z * 100, np.clip(L, *levels), cmap='magma', vmin=levels[0], vmax=levels[1],
                       shading='auto', rasterized=True)
    # The plate's section: flat top, taper, rim.
    r_taper = radius - taper_length
    ax.fill_between([-radius, -r_taper, r_taper, radius],
                    0, np.array([edge_thickness, thickness, thickness, edge_thickness]) * 100, color='0.8', lw=0)
    rm = gp.PLATE_MIC_OFFSET_FT / gp.PLATE_RADIUS_FT * radius      # ARP 4055: 3/4 of the radius
    ax.plot([-rm, rm], [zm * 100] * 2, 'o', ms=7, mfc='white', mec='k', mew=1.5)
    ax.set_ylim(0, z_top * 100)
    ax.set_ylabel('Height (cm)')
    ground_label = (f'{flow_resistance:g} kPa s/m²' if ground is None else
                    ' '.join(f'{k}={v:g}' if k != 'model' else v for k, v in ground.items()))
    ax.set_title(f'{frequency:.0f} Hz, plane wave from {elevation:.0f}° elevation, ground '
                 f'{ground_label}', loc='left')
    fig.colorbar(im, ax=ax, pad=0.01, aspect=12).set_label('Level re free field (dB)')
    bx.axvspan(-radius, radius, color='0.92', lw=0)
    bx.plot(xl, Ll0, color='0.45', lw=2, ls='--', label='Ground alone')
    bx.plot(xl, Ll, color='tab:blue', lw=2.5, label='Plate on the ground (BEM)')
    bx.axhline(gp.PRESSURE_DOUBLING_DB, color='k', lw=1)
    bx.set_ylabel('At the mic height\n(dB re free field)')
    bx.set_xlim(*x_range)
    bx.set_xlabel('Horizontal distance from the plate center, along the incidence direction (m)')
    bx.legend(loc='lower left', frameon=False)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--frequency', type=float, default=4000.0, help='Hz (default 4000)')
    ap.add_argument('--elevation', type=float, default=30.0, help='incidence elevation, deg (default 30)')
    ap.add_argument('--flow-resistance', type=float, default=225.0,
                    help='Delany-Bazley flow resistivity, kPa s/m^2 (default 225, grass)')
    ap.add_argument('--mic-height', type=float, default=gp.PLATE_MIC_HEIGHT_FT * FT,
                    help='microphone height above the plate top, m (default flush; 0.007 for an inverted mic)')
    ap.add_argument('--grid', default='201x101', help='map points, NXxNZ (default 201x101)')
    ap.add_argument('-o', '--output', default='board_field.png')
    a = ap.parse_args(argv)
    nx, nz = (int(v) for v in a.grid.lower().split('x'))
    print(plot(a.frequency, a.elevation, a.flow_resistance, a.output, grid=(nx, nz), mic_height_m=a.mic_height))


if __name__ == '__main__':
    main()
