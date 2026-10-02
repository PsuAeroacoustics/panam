"""adaptive_idw_weights: a radius per node that leaves no holes, brackets every node with several
microphones, and keeps a gradient a fixed radius smears."""
import numpy as np
import pytest

from flight_acoustics import adaptive_idw_weights, depropagate_hemisphere, geodist, shepIDW


def _lines(rng=None, n_mics=11, height=400.0, stride_ft=8.0):
    """A flyover's samples: a level pass at `height` over microphones across the track,
    through hemigen, kept within the build's 2,000 ft and above 2 deg -- each microphone a
    line on the sphere from fore to aft, the lines densest at the rim."""
    from flight_acoustics import hemigen
    x = np.arange(-2000.0, 2000.0, stride_ft)
    time = x / 169.0
    source = np.column_stack((x, np.zeros_like(x), np.full_like(x, height)))
    velocity = np.tile([169.0, 0.0, 0.0], (x.size, 1))
    mics = np.column_stack((np.zeros(n_mics), np.linspace(-1500.0, 1500.0, n_mics), np.zeros(n_mics)))
    az, el, r, _, _ = hemigen(time, source, velocity, mics, 1125.0)
    keep = (r <= 2000.0) & (el >= 2.0)
    mic = np.broadcast_to(np.arange(n_mics), el.shape)
    return el[keep], az[keep], mic[keep]


def _nodes(step=2.0):
    e, a = np.meshgrid(np.arange(2.0, 90.0 + 1e-9, step), np.arange(0.0, 360.0, step), indexing='ij')
    return e.ravel(), a.ravel()


def test_rows_sum_to_one_and_every_radius_brackets_with_several_microphones():
    rng = np.random.default_rng(0)
    felv, fazi, mic = _lines(rng)
    res = np.full(felv.size, 1.0)
    ielv, iazi = _nodes(4.0)
    w, radius, gap = adaptive_idw_weights(ielv, iazi, felv, fazi, mic, res, k=8, min_mics=3,
                                          max_radius_deg=60.0)
    sums = np.asarray(w.sum(axis=1)).ravel()
    assert np.allclose(sums[~gap], 1.0)
    assert np.all(sums[gap] == 0.0)
    for node in np.flatnonzero(~gap)[::7]:
        cols = w[node].indices
        h = geodist(ielv[node], iazi[node], felv, fazi)
        if h.min() <= 1e-12:
            continue                                   # a node on a sample takes that sample alone
        assert np.unique(mic[cols]).size >= 3
        assert np.sum(h < radius[node]) >= 8


def test_no_holes_where_a_fixed_radius_leaves_them_and_gaps_are_reported():
    rng = np.random.default_rng(1)
    felv, fazi, mic = _lines(rng)
    res = np.full(felv.size, 0.5)
    ielv, iazi = _nodes()
    # A fixed radius small enough for the rim leaves holes overhead and between lines.
    fixed = shepIDW(ielv, iazi, felv, fazi, np.ones(felv.size), rmax=5.0)
    assert np.isnan(fixed).sum() > 0
    w, radius, gap = adaptive_idw_weights(ielv, iazi, felv, fazi, mic, res, max_radius_deg=60.0)
    # Far fewer gaps than the fixed radius has holes, and each one a node that cannot be
    # bracketed by three microphones within the cap: steep to the side, where only the
    # centreline microphone sees the aircraft.
    assert gap.sum() < 0.05 * np.isnan(fixed).sum()
    kappa, cap = 1.3, 60.0
    for node in np.flatnonzero(gap):
        h = geodist(ielv[node], iazi[node], felv, fazi)
        assert np.unique(mic[h < cap / kappa]).size < 3 or np.sort(h)[7] > cap / kappa
    # With the cap below what some nodes need, those are gaps, and only those.
    w2, radius2, gap2 = adaptive_idw_weights(ielv, iazi, felv, fazi, mic, res, max_radius_deg=10.0)
    assert 0 < gap2.sum() < gap2.size
    assert np.all(np.isnan(radius2[gap2]))


def test_radius_follows_sample_density_and_resolution():
    rng = np.random.default_rng(2)
    felv, fazi, mic = _lines(rng)
    ielv, iazi = _nodes(4.0)
    w, radius, gap = adaptive_idw_weights(ielv, iazi, felv, fazi, mic, np.full(felv.size, 0.1),
                                          max_radius_deg=90.0)
    # Samples crowd at shallow angles and thin out overhead.
    shallow, overhead = (ielv >= 10) & (ielv < 30), ielv > 60
    assert np.nanmedian(radius[shallow]) < np.nanmedian(radius[overhead])
    # A coarser measurement widens the radius to its resolution.
    w2, radius2, _ = adaptive_idw_weights(ielv, iazi, felv, fazi, mic, np.full(felv.size, 20.0),
                                          resolution_factor=1.0, max_radius_deg=90.0)
    assert np.nanmin(radius2) >= 20.0


def test_dense_rim_is_no_worse_than_a_fixed_radius():
    rng = np.random.default_rng(3)
    felv, fazi, mic = _lines(height=150.0, stride_ft=4.0)       # a low pass: data to 4 deg
    level = lambda e: 0.25 * e                                            # dB, rising off the rim  # noqa: E731
    power = 10.0 ** (0.1 * level(felv))
    ielv, iazi = _nodes()
    w, _, gap = adaptive_idw_weights(ielv, iazi, felv, fazi, mic, np.full(felv.size, 0.5),
                                     max_radius_deg=60.0)
    adaptive = 10.0 * np.log10(np.asarray(w @ power)).ravel()
    fixed = 10.0 * np.log10(shepIDW(ielv, iazi, np.r_[felv, felv, felv], np.r_[fazi, fazi + 360, fazi - 360],
                                    np.r_[power, power, power], rmax=25.0))
    # Where the samples are dense the radius is small, so the grid is no worse than the fixed
    # 25 deg one.  (A synthetic gradient this steep, averaged as power, is biased by any
    # radius; spheres/rim_samples.py in the validation harness checks real spheres.)
    rim = (ielv >= 4) & (ielv < 14) & ~gap & np.isfinite(fixed)
    err_adaptive = np.mean(np.abs(adaptive[rim] - level(ielv[rim])))
    err_fixed = np.mean(np.abs(fixed[rim] - level(ielv[rim])))
    assert err_adaptive <= err_fixed


def test_depropagate_hemisphere_adaptive_runs_and_reports():
    rng = np.random.default_rng(4)
    mic_locations = np.array([[0.0, -300.0, 0.0], [0.0, 0.0, 0.0], [0.0, 300.0, 0.0], [0.0, 600.0, 0.0]])
    nt = 120
    track_time = np.linspace(0.0, 6.0, nt)
    track_position = np.column_stack([800.0 * track_time / track_time[-1] - 400.0, np.zeros(nt), np.full(nt, 250.0)])
    track_velocity = np.tile([800.0 / track_time[-1], 0.0, 0.0], (nt, 1))
    fs = 4000.0
    t = np.arange(0.0, 7.0, 1.0 / fs)
    pressure = np.vstack([2e-5 * (3.0 * np.sin(2 * np.pi * 250.0 * t) + rng.standard_normal(t.size))
                          for _ in range(4)])
    common = dict(mic_locations=mic_locations, pressure=pressure, time=t, track_time=track_time,
                  track_position=track_position, track_velocity=track_velocity, speed_of_sound=1135.0,
                  r_ref=100.0, freq_range=(50.0, 1500.0), window_time=0.25, window_overlap=0.5, point_stride=2,
                  azi_step=10.0, elv_step=5.0, rmax=25.0, third_octave=True, third_octave_fmin=100.0,
                  return_scattered=True)
    out = depropagate_hemisphere(**common, interpolation=dict(min_mics=2, source_extent=35.0))
    info = out['interpolation']
    assert info['mode'] == 'adaptive' and isinstance(info['gaps'], int)
    assert info['radius_deg'].shape == out['oaspl_db'].shape
    scattered = out['scattered']
    assert scattered['mic'].size == scattered['elv_deg'].size == scattered['resolution_deg'].size
    assert np.all(scattered['resolution_deg'] > 0)
    finite = np.isfinite(out['oaspl_db'])
    assert finite.any()
    assert np.array_equal(np.isnan(out['oaspl_db'][:, :-1]), np.isnan(info['radius_deg'][:, :-1]))
    with pytest.raises(ValueError, match='unknown interpolation'):
        depropagate_hemisphere(**common, interpolation=dict(radius=3))
