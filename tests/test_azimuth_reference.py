"""Filing a sphere's azimuth by the ground track or by the airframe's heading.

build_sphere(azimuth_reference='heading') measures azimuth from the INS heading while
Doppler, the convective Mach number and the depropagation keep the ground velocity; the
database's root azimuth_reference says which frame its spheres are in.
"""

import warnings

import numpy as np
import pytest
from netCDF4 import Dataset

import flight_acoustics as fa
import noise_abatement_2017 as na

from test_database_provenance import _source_directory


# --- hemigen and depropagate_hemisphere -----------------------------------------------------

def test_a_nose_turns_the_azimuth_and_nothing_else():
    """A nose turned 10 deg clockwise (seen from above) of the velocity puts every
    observer 10 deg further round in azimuth; elevation, range, reception time and the
    Mach number stay the velocity's."""
    time = np.linspace(0.0, 2.0, 5)
    source = np.column_stack((100.0 * time - 100.0, np.zeros(5), np.full(5, 300.0)))
    velocity = np.tile([100.0, 0.0, -5.0], (5, 1))
    observers = np.array([[0.0, -150.0, 0.0], [50.0, 200.0, 2.0], [-400.0, 0.0, 0.0]])
    turn = np.radians(-10.0)                                   # clockwise in a z-up frame
    nose = np.tile([np.cos(turn), np.sin(turn), 0.0], (5, 1))
    plain = fa.hemigen(time, source, velocity, observers, 1100.0)
    turned = fa.hemigen(time, source, velocity, observers, 1100.0, nose=nose)
    difference = (turned[0] - plain[0] + 180.0) % 360.0 - 180.0
    np.testing.assert_allclose(difference, 10.0, atol=1e-9)
    for a, b in zip(plain[1:], turned[1:]):
        np.testing.assert_array_equal(a, b)
    # Only the nose's horizontal direction counts, and (Nt, 2) works as well.
    np.testing.assert_allclose(fa.hemigen(time, source, velocity, observers, 1100.0, nose=5.0 * nose)[0],
                               turned[0], atol=1e-12)
    np.testing.assert_allclose(fa.hemigen(time, source, velocity, observers, 1100.0, nose=nose[:, :2])[0],
                               turned[0], atol=1e-12)


def test_a_nose_must_be_a_direction():
    time = np.linspace(0.0, 1.0, 3)
    source = np.zeros((3, 3))
    velocity = np.tile([1.0, 0.0, 0.0], (3, 1))
    observers = np.array([[10.0, 0.0, -100.0]])
    with pytest.raises(ValueError, match='shaped'):
        fa.hemigen(time, source, velocity, observers, 1100.0, nose=np.ones((2, 3)))
    with pytest.raises(ValueError, match='nonzero horizontal'):
        fa.hemigen(time, source, velocity, observers, 1100.0, nose=np.tile([0.0, 0.0, 1.0], (3, 1)))


def test_depropagation_flips_the_nose_with_the_geometry(monkeypatch):
    seen = {}

    def capture(time, source, velocity, observers, speed_of_sound, nose=None):
        seen.update(velocity=velocity, nose=nose)
        raise RuntimeError('captured')
    monkeypatch.setattr(fa, 'hemigen', capture)
    time = np.linspace(0.0, 1.0, 4)
    position = np.zeros((4, 3))
    velocity = np.tile([100.0, 3.0, 0.0], (4, 1))
    nose = np.tile([0.9, 0.3, 0.0], (4, 1))
    with pytest.raises(RuntimeError, match='captured'):
        fa.depropagate_hemisphere(np.zeros((1, 3)), np.zeros((1, 10)), np.arange(10.0), time, position,
                                  velocity, track_nose=nose, flip_y_for_geometry=True)
    np.testing.assert_array_equal(seen['nose'][:, 1], -nose[:, 1])
    np.testing.assert_array_equal(seen['velocity'][:, 1], -velocity[:, 1])
    monkeypatch.undo()
    with pytest.raises(ValueError, match='track_nose must be shape'):
        fa.depropagate_hemisphere(np.zeros((1, 3)), np.zeros((1, 10)), np.arange(10.0), time, position,
                                  velocity, track_nose=nose[:3])


# --- the sphere build ---------------------------------------------------------------------

def _archive_with_heading(root, heading_deg, true_heading='270', missing=()):
    """The synthetic archive, its reference list carrying true_heading and its track the
    INS heading ``heading_deg`` (NaN at the sample indices ``missing``); the pass flies
    along +x, compass 270 on a 270 frame."""
    from test_noise_abatement_2017 import _archive, _write_csv
    _archive(root)
    base = root / 'AS350B3'
    _write_csv(base / 'AS350B3FullRefList.csv',
               ['combined', 'test_cond', 'layout', 'bank_ang', 'accel_rate', 'utc_secs_from_mid_start',
                'run_num', 'true_heading'],
               [['289101', 'AMB', 'A', '0', '0', '60.0', '101', true_heading],
                ['289108', 'L1', 'A', '0', '0', '100.0', '108', true_heading]])
    speed = 150.0
    _write_csv(base / 'AS350B3_AC_Data' / '289108AC.csv',
               ['utcsec', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'VGk', 'roll', 'heading'],
               [[t, speed * (t - 105.0), 0.0, 300.0, speed, 0.0, 0.0, speed * 0.3048 / 0.514444, 0.0,
                 float('nan') if i in missing else heading_deg]
                for i, t in enumerate(100.0 + np.arange(500) / 50.0)])
    return na.NoiseAbatementTest('AS350B3', root=str(root))


def test_a_heading_filed_sphere_turns_by_the_crab_and_keeps_its_levels(tmp_path):
    """Nose 10 deg right of the ground track (compass 280 on a 270 track): every sample
    is filed 10 deg further round, at the same elevation and the same levels, because
    the depropagation itself still follows the ground velocity."""
    test = _archive_with_heading(tmp_path, 280.0)
    samples = {}
    for reference in ('track', 'heading'):
        record = na.build_sphere(test, '289108', str(tmp_path / (reference + '.nc')), board_correction='flat',
                                 azimuth_reference=reference, samples_path=str(tmp_path / (reference + '.npz')))
        assert record['azimuth_reference'] == reference
        assert fa.read_run_metadata(record['output'])['azimuth_reference'] == reference
        samples[reference] = np.load(str(tmp_path / (reference + '.npz')))
    track, heading = samples['track'], samples['heading']
    assert track['azimuth_deg'].size > 100
    turn = (heading['azimuth_deg'] - track['azimuth_deg'] + 180.0) % 360.0 - 180.0
    np.testing.assert_allclose(turn, 10.0, atol=1e-6)
    for name in ('elevation_deg', 'bands_db', 'range_ft', 'mic'):
        np.testing.assert_array_equal(heading[name], track[name])


def _capture_depropagation(monkeypatch):
    seen = {}

    def capture(**kwargs):
        seen.update(kwargs)
        raise RuntimeError('captured')
    monkeypatch.setattr(na.fa, 'depropagate_hemisphere', capture)
    return seen


def test_the_heading_frame_hands_depropagation_the_ground_velocity_and_the_nose(monkeypatch, tmp_path):
    test = _archive_with_heading(tmp_path, 280.0)
    seen = _capture_depropagation(monkeypatch)
    with pytest.raises(RuntimeError, match='captured'):
        na.build_sphere(test, '289108', str(tmp_path / 'x.nc'), board_correction='flat', azimuth_reference='heading')
    np.testing.assert_allclose(seen['track_velocity'], np.tile([150.0, 0.0, 0.0], (seen['track_time'].size, 1)))
    turn = np.radians(-10.0)
    np.testing.assert_allclose(seen['track_nose'], np.tile([np.cos(turn), np.sin(turn), 0.0],
                                                           (seen['track_time'].size, 1)), atol=1e-12)
    with pytest.raises(RuntimeError, match='captured'):
        na.build_sphere(test, '289108', str(tmp_path / 'x.nc'), board_correction='flat')
    assert seen['track_nose'] is None


def test_a_hover_is_oriented_by_its_heading_in_its_own_frame(monkeypatch, tmp_path):
    """A hover layout's frame is not the flight layout's: Amedee's runs along 279 deg.
    The nose (compass 300) is 21 deg clockwise of +x there, where the 270 deg this once
    assumed put it 30 deg round."""
    test = _archive_with_heading(tmp_path, 300.0, true_heading='279')
    seen = _capture_depropagation(monkeypatch)
    with pytest.raises(RuntimeError, match='captured'):
        na.build_sphere(test, '289108', str(tmp_path / 'x.nc'), board_correction='flat', nose_from_heading=True)
    direction = seen['track_velocity'] / np.linalg.norm(seen['track_velocity'], axis=1)[:, None]
    turn = np.radians(-21.0)
    np.testing.assert_allclose(direction, np.tile([np.cos(turn), np.sin(turn), 0.0], (direction.shape[0], 1)),
                               atol=1e-12)
    assert np.allclose(np.linalg.norm(seen['track_velocity'], axis=1), 1e-3)
    assert seen['track_nose'] is None


@pytest.mark.parametrize('options, message', [
    (dict(nose_from_heading=True, azimuth_reference='track'), 'contradicts'),
    (dict(azimuth_reference='nose'), 'must be one of'),
])
def test_contradictory_frames_are_refused(tmp_path, options, message):
    test = _archive_with_heading(tmp_path, 280.0)
    with pytest.raises(ValueError, match=message):
        na.build_sphere(test, '289108', str(tmp_path / 'x.nc'), board_correction='flat', **options)


def test_a_heading_frame_needs_a_heading_and_a_frame_bearing(tmp_path):
    from test_noise_abatement_2017 import _archive
    _archive(tmp_path)                                   # no true_heading in its reference list
    test = na.NoiseAbatementTest('AS350B3', root=str(tmp_path))
    with pytest.raises(ValueError, match='no true_heading'):
        na.build_sphere(test, '289108', str(tmp_path / 'x.nc'), board_correction='flat', azimuth_reference='heading')
    # A hover's window is its whole record, dropouts included.
    test = _archive_with_heading(tmp_path, 280.0, missing=range(250, 255))
    with pytest.raises(ValueError, match='no finite heading'):
        na.build_sphere(test, '289108', str(tmp_path / 'x.nc'), board_correction='flat', nose_from_heading=True)


# --- the database ---------------------------------------------------------------------------

def _label(directory, name, reference):
    with Dataset(str(directory / name), 'a') as ds:
        ds.azimuth_reference = reference


def _build(directory, path, **options):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fa.build_empirical_database(str(directory), str(path), load_factors=None, **options)
    with Dataset(str(path)) as db:
        return db.getncattr('azimuth_reference'), [str(w.message) for w in caught]


def test_the_database_says_which_frame_its_spheres_are_in(tmp_path):
    spheres = {'A.nc': dict(speed=60.0, doppler=0.0), 'B.nc': dict(speed=90.0, doppler=0.0)}
    directory = _source_directory(tmp_path, spheres)
    reference, messages = _build(directory, tmp_path / 'unlabeled.nod')
    assert reference == 'track' and any('carry no azimuth_reference' in m for m in messages)
    reference, messages = _build(directory, tmp_path / 'told.nod', azimuth_reference='heading')
    assert reference == 'heading' and not any('azimuth_reference' in m for m in messages)
    _label(directory, 'A.nc', 'heading')
    _label(directory, 'B.nc', 'heading')
    reference, messages = _build(directory, tmp_path / 'labeled.nod')
    assert reference == 'heading' and not any('azimuth_reference' in m for m in messages)
    with pytest.raises(ValueError, match="asked for 'track'"):
        _build(directory, tmp_path / 'conflict.nod', azimuth_reference='track')


def test_a_database_filed_in_two_frames_is_refused(tmp_path):
    directory = _source_directory(tmp_path, {'A.nc': dict(speed=60.0, doppler=0.0),
                                             'B.nc': dict(speed=90.0, doppler=0.0)})
    _label(directory, 'A.nc', 'heading')
    _label(directory, 'B.nc', 'track')
    with pytest.raises(ValueError, match='differs between the sources'):
        _build(directory, tmp_path / 'mixed.nod')
    _label(directory, 'B.nc', 'nose')
    with pytest.raises(ValueError, match='must be one of'):
        _build(directory, tmp_path / 'bad.nod')
