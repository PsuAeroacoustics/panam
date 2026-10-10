"""depropagate_hemisphere's ray_model: the straight-ray model reproduces the default,
and a refracted model refiles, respreads and skips samples as it says."""
import os
import shutil

import numpy as np
import pytest

from flight_acoustics import depropagate_hemisphere
import refracted_rays as rr

P_REF = 2.0e-5
SPEED = 1135.0


def _flyover(seed=0):
    rng = np.random.default_rng(seed)
    mic_locations = np.array([[0.0, -300.0, 0.0], [0.0, 0.0, 0.0], [0.0, 300.0, 0.0]])
    nt = 120
    track_time = np.linspace(0.0, 6.0, nt)
    track_position = np.column_stack([800.0 * track_time / track_time[-1] - 400.0, np.zeros(nt),
                                      np.full(nt, 250.0)])
    track_velocity = np.tile([800.0 / track_time[-1], 0.0, 0.0], (nt, 1))
    fs = 4000.0
    t = np.arange(0.0, 7.0, 1.0 / fs)
    pressure = np.vstack([P_REF * (3.0 * np.sin(2 * np.pi * 250.0 * t) + rng.standard_normal(t.size))
                          for _ in range(3)])
    return dict(mic_locations=mic_locations, pressure=pressure, time=t, track_time=track_time,
                track_position=track_position, track_velocity=track_velocity)


def _hemisphere(ray_model=None, **extra):
    return depropagate_hemisphere(**_flyover(), speed_of_sound=SPEED, length_units='ft', r_ref=100.0,
                                  freq_range=(50.0, 1500.0), window_time=0.25, window_overlap=0.5,
                                  point_stride=4, azi_step=20.0, elv_step=10.0, rmax=30.0,
                                  third_octave=True, third_octave_fmin=100.0, return_scattered=True,
                                  ray_model=ray_model, **extra)


def _response(im, bands, offset):
    # A response that depends on the arrival elevation, so the offset is exercised.
    elevation = np.degrees(np.arctan2(offset[:, 2], np.hypot(offset[:, 0], offset[:, 1])))
    return np.broadcast_to(3.0 + 0.05 * elevation, (len(bands), offset.shape[0])).copy()


def test_straight_ray_model_reproduces_the_default():
    for extra in ({}, dict(apply_absorption_deprop=True, receiver_response_db=_response, min_elevation_deg=10.0)):
        plain = _hemisphere(**extra)['scattered']
        rayed = _hemisphere(rr.straight_ray_model(SPEED), **extra)['scattered']
        assert np.allclose(plain['elv_deg'], rayed['elv_deg'], rtol=0.0, atol=1e-9)
        assert np.allclose(plain['azi_deg'], rayed['azi_deg'], rtol=0.0, atol=1e-9)
        assert np.allclose(plain['third_octave']['bands_db'], rayed['third_octave']['bands_db'],
                           rtol=0.0, atol=1e-9, equal_nan=True)


def test_a_refracted_model_refiles_respreads_and_skips():
    straight = rr.straight_ray_model(SPEED)

    def bent(source, mics):
        out = straight(source, mics)
        out['depression_deg'] = out['depression_deg'] - 2.0        # launched 2 deg shallower
        out['spreading_range'] = out['spreading_range'] * 2.0      # a ray tube twice as wide: +6 dB
        out['valid'] = out['valid'].copy()
        out['valid'][:, 0] = False                                  # mic 0 in a shadow
        return out

    plain = _hemisphere()['scattered']
    rayed = _hemisphere(bent)['scattered']
    # Mic 0's samples are gone; the rest are filed 2 deg shallower and 6 dB louder.
    n_plain, n_rayed = plain['elv_deg'].size, rayed['elv_deg'].size
    assert n_rayed == pytest.approx(n_plain * 2 / 3, abs=2)
    keep = slice(n_plain - n_rayed, None)                           # mics are appended in order
    assert np.allclose(rayed['elv_deg'], plain['elv_deg'][keep] - 2.0, atol=1e-9)
    gain = rayed['third_octave']['bands_db'] - plain['third_octave']['bands_db'][:, keep]
    finite = np.isfinite(gain)
    assert finite.any()
    assert np.allclose(gain[finite], 20.0 * np.log10(2.0), atol=1e-9)


def test_ray_model_shapes_are_checked():
    def wrong(source, mics):
        out = rr.straight_ray_model(SPEED)(source, mics)
        out['travel_time'] = out['travel_time'][:, :1]
        return out
    with pytest.raises(ValueError, match='travel_time'):
        _hemisphere(wrong)


RAYS = os.environ.get('NICEOPS_RAY_GEOMETRY') or shutil.which('niceops_ray_geometry')


@pytest.mark.skipif(RAYS is None, reason='set NICEOPS_RAY_GEOMETRY to a niceops_ray_geometry build')
def test_external_model_in_uniform_air_is_straight(tmp_path):
    atmosphere = tmp_path / 'uniform.csv'
    atmosphere.write_text('z_ft,T_C,RH\n0,15,50\n')
    model = rr.external_ray_model(RAYS, str(atmosphere))
    flight = _flyover()
    source, mics = flight['track_position'][::10], flight['mic_locations']
    got, want = model(source, mics), rr.straight_ray_model(340.294 / 0.3048)(source, mics)
    assert got['valid'].all()
    for key in ('depression_deg', 'spreading_range', 'path_length'):
        assert np.allclose(got[key], want[key], rtol=1e-9, atol=1e-9), key
    assert np.allclose(got['offset'], want['offset'], rtol=1e-9, atol=1e-6)
    # Its sound speed is the profile's (15 C), within the reference's rounding.
    assert np.allclose(got['travel_time'], want['travel_time'], rtol=1e-4)


@pytest.mark.skipif(RAYS is None, reason='set NICEOPS_RAY_GEOMETRY to a niceops_ray_geometry build')
def test_external_model_under_an_inversion_launches_shallower(tmp_path):
    atmosphere = tmp_path / 'inversion.csv'
    atmosphere.write_text('z_ft,T_C,RH\n0,0,50\n150,8,50\n2000,8,50\n')
    model = rr.external_ray_model(RAYS, str(atmosphere))
    # Inside the reach of the rays the 150 ft cap turns back (about 2,800 ft from
    # a 100 ft source); beyond it ray theory has a gap until the bounced rays.
    source = np.array([[-1000.0, 0.0, 100.0], [-1500.0, 0.0, 100.0], [-2500.0, 0.0, 100.0]])
    got = model(source, np.zeros((1, 3)))
    straight = rr.straight_ray_model(SPEED)(source, np.zeros((1, 3)))
    assert got['valid'].all()
    # Bent down by the inversion: it leaves shallower and arrives steeper.
    assert (got['depression_deg'] < straight['depression_deg']).all()
    arrival = np.degrees(np.arctan2(got['offset'][..., 2], np.hypot(got['offset'][..., 0], got['offset'][..., 1])))
    assert (arrival > straight['depression_deg']).all()


def test_rim_range_admits_far_samples_only_at_the_rim():
    # A long, high pass: within the 600 ft cap it never files below ~24 deg; with the
    # rim range the shallow directions come in from further out, and nothing steeper
    # changes.
    flight = _flyover()
    flight['track_position'] = flight['track_position'] * np.array([3.0, 1.0, 1.0])
    flight['track_velocity'] = flight['track_velocity'] * 3.0
    common = dict(**flight, speed_of_sound=SPEED, length_units='ft', r_ref=100.0, freq_range=(50.0, 1500.0),
                  window_time=0.25, window_overlap=0.5, point_stride=2, azi_step=20.0, elv_step=10.0,
                  rmax=30.0, third_octave=True, third_octave_fmin=100.0, return_scattered=True,
                  max_range=600.0)
    plain = depropagate_hemisphere(**common)['scattered']
    rim = depropagate_hemisphere(**common, rim_range=(25.0, 1300.0))['scattered']
    assert rim['elv_deg'].size > plain['elv_deg'].size
    # Every extra sample is filed below the rim elevation.
    steep_plain = np.sort(plain['elv_deg'][plain['elv_deg'] >= 25.0])
    steep_rim = np.sort(rim['elv_deg'][rim['elv_deg'] >= 25.0])
    assert np.allclose(steep_plain, steep_rim)
    assert (rim['elv_deg'] < 25.0).sum() > (plain['elv_deg'] < 25.0).sum()


# A stand-in for niceops_ray_geometry: straight lines at FAKE_SPEED, its rows in
# reverse order, the pair with id FAKE_DARK_ID unlit, and exit 1 with FAKE_FAIL set.
FAKE_TRACER = r'''#!{python}
import argparse, csv, math, os, sys
parser = argparse.ArgumentParser()
for option in ('--atmosphere', '--paths', '--frame_bearing', '--refraction', '--ground', '--units', '-j'):
    parser.add_argument(option)
args = parser.parse_args()
if os.environ.get('FAKE_FAIL'):
    sys.stderr.write('fake tracer: profile rejected\n')
    sys.exit(1)
assert os.path.isfile(args.atmosphere)
with open(args.paths) as handle:
    rows = list(csv.DictReader(handle))
print('id,lit,launch_deg,arrival_deg,spreading_db,shadow,boundary,from_fit,bounces,extras,extra_db,'
      'source_height,ground_distance,range,path_length,travel_time')
for row in reversed(rows):
    dx, dy, hs, hr = (float(row[k]) for k in ('dx', 'dy', 'source_height', 'receiver_height'))
    ground = math.hypot(dx, dy)
    r = math.hypot(ground, hs - hr)
    launch = -math.degrees(math.atan2(hs - hr, ground))
    lit = 0 if row['id'] == os.environ.get('FAKE_DARK_ID') else 1
    print(','.join(str(v) for v in (row['id'], lit, repr(launch), repr(launch), 0, 1 - lit, 'inf', 0, 0, 0, '-inf',
                                    repr(hs), repr(ground), repr(r), repr(r), repr(r / {speed!r}))))
'''


def _fake_tracer(directory):
    import sys
    path = directory / 'fake_ray_geometry'
    path.write_text(FAKE_TRACER.format(python=sys.executable, speed=SPEED))
    path.chmod(0o755)
    return path


def test_external_model_parses_a_tracer_and_rebuilds_the_geometry(tmp_path, monkeypatch):
    """The CSV round trip, the id reordering, the scaling to ground_distance,
    the 'point' receiver height and the lit mask, against straight lines."""
    tracer = _fake_tracer(tmp_path)
    atmosphere = tmp_path / 'profile.csv'
    atmosphere.write_text('z_ft,T_C,RH\n0,15,50\n')
    monkeypatch.setenv('FAKE_DARK_ID', '4')                      # source 1, mic 1
    flight = _flyover()
    source, mics = flight['track_position'][::30], flight['mic_locations']
    model = rr.external_ray_model(str(tracer), str(atmosphere), receiver='point',
                                  receiver_heights=np.array([5.0, 1.0, 0.0]))
    got, want = model(source, mics), rr.straight_ray_model(SPEED)(source, mics)
    expected_valid = np.ones((source.shape[0], mics.shape[0]), dtype=bool)
    expected_valid[1, 1] = False
    assert np.array_equal(got['valid'], expected_valid)
    for key in ('depression_deg', 'travel_time', 'spreading_range', 'path_length'):
        np.testing.assert_allclose(got[key], want[key], rtol=1e-12, atol=1e-9, err_msg=key)
    np.testing.assert_allclose(got['offset'], want['offset'], rtol=1e-12, atol=1e-9)
    assert model.description == 'niceops_ray_geometry stratified point profile.csv'


def test_external_model_runs_a_tracer_given_by_relative_path(tmp_path, monkeypatch):
    """The tracer runs in a scratch directory, so a relative path to it must
    not be resolved from there."""
    (tmp_path / 'bin').mkdir()
    _fake_tracer(tmp_path / 'bin')
    (tmp_path / 'profile.csv').write_text('z_ft,T_C,RH\n0,15,50\n')
    monkeypatch.chdir(tmp_path)
    model = rr.external_ray_model(os.path.join('bin', 'fake_ray_geometry'), 'profile.csv')
    got = model(np.array([[-500.0, 0.0, 300.0]]), np.zeros((1, 3)))
    assert got['valid'].all()


def test_external_model_reports_a_failed_tracer(tmp_path, monkeypatch):
    tracer = _fake_tracer(tmp_path)
    atmosphere = tmp_path / 'profile.csv'
    atmosphere.write_text('z_ft,T_C,RH\n0,15,50\n')
    monkeypatch.setenv('FAKE_FAIL', '1')
    model = rr.external_ray_model(str(tracer), str(atmosphere))
    with pytest.raises(RuntimeError, match=r'fake_ray_geometry failed \(1\): fake tracer: profile rejected'):
        model(np.array([[-500.0, 0.0, 300.0]]), np.zeros((1, 3)))
