"""Flight-path databases: spheres kept in AAM's frame instead of pitched to level.

database_frame='flight_path' builds a database whose spheres are as PSU-WOPWOP
(or AAM) files them -- theta from the velocity, phi about it -- and records
sphere_frame = flight_path at the root, which NICEOPS reads in that frame.  The
default, 'level', pitches each sphere by its flight path angle as before.  Where
NICEOPS is configured, a steady descent over a level database and over a
flight-path one of the same spheres gives the same footprint, to the
interpolation the level build's pitch adds.
"""
import subprocess

import numpy as np
import pytest
from netCDF4 import Dataset

from panam import flight_acoustics as fa
from panam import sphere_statistics
from panam.database import DATABASE_FORMAT_VERSION
from panam.nod_database import validate_nod
from tools import build_empirical_database as script

import sphere_helpers

PHI = np.arange(-180.0, 180.0 + 1e-9, 10.0)
THETA = np.arange(0.0, 180.0 + 1e-9, 10.0)
BANDS = np.array([100.0, 125.0, 160.0, 200.0, 250.0])
CONDITIONS = ((40.0, 0.0), (40.0, -6.0), (80.0, 0.0), (80.0, -6.0))


def _levels(speed):
    # Louder ahead and below, so a pitch error shows.
    theta, phi = np.meshgrid(np.radians(THETA), np.radians(PHI))
    level = (75.0 + 0.04 * speed + 6.0 * np.cos(theta) + 4.0 * np.sin(theta) * np.cos(phi)
             + 2.0 * np.sin(theta) * np.sin(phi))
    levels = np.repeat(level[:, :, None], BANDS.size, axis=2) - 2.0 * np.arange(BANDS.size)
    levels[-1] = levels[0]
    return levels


def _folder(directory, conditions=CONDITIONS):
    directory.mkdir()
    for i, (speed, angle) in enumerate(conditions):
        sphere_helpers.write_wopwop_sphere(directory / f'FPDAT{i:03d}.nc', _levels(speed), PHI, THETA, BANDS,
                                           speed=speed, flight_path_angle=angle, floor=-60.0)
    return directory


def _build(directory, database, **options):
    fa.build_empirical_database(str(directory), str(database), load_factors=None, **options)
    return database


def _measured(database, speed, angle):
    with Dataset(str(database)) as handle:
        for group in handle.groups.values():
            if group.condition_origin == 'measured' and abs(float(group['flight_path_angle'][0]) - angle) < 1e-9 \
                    and abs(float(group['advance_ratio'][0]) - speed * 0.514444 / 230.7) < 1e-6:
                return np.array(group['amplitude'][:])
    raise AssertionError('no group at {} kt, {} deg'.format(speed, angle))


def test_a_flight_path_database_keeps_its_spheres_unpitched(tmp_path):
    directory = _folder(tmp_path / 'spheres')
    flight = _build(directory, tmp_path / 'flight.nod', database_frame='flight_path')
    level = _build(directory, tmp_path / 'level.nod')
    assert validate_nod(str(flight)) == [] and validate_nod(str(level)) == []
    with Dataset(str(flight)) as handle:
        assert handle.sphere_frame == 'flight_path'
        assert int(handle['database_version'][...]) == DATABASE_FORMAT_VERSION == 1
    with Dataset(str(level)) as handle:
        assert handle.sphere_frame == 'level'
    # The descent's sphere as the file holds it, seam row dropped ...
    np.testing.assert_allclose(_measured(flight, 80.0, -6.0), _levels(80.0)[:-1], atol=1e-4)
    # ... where the level build pitches it; level flight is the same in both.
    assert np.abs(_measured(level, 80.0, -6.0) - _levels(80.0)[:-1]).max() > 0.5
    np.testing.assert_allclose(_measured(level, 80.0, 0.0), _measured(flight, 80.0, 0.0), atol=1e-9)


def test_level_spheres_cannot_make_a_flight_path_database(tmp_path):
    directory = tmp_path / 'spheres'
    sphere_helpers.write_sphere_directory(
        directory, [(sphere_helpers.minimal_hemisphere(), 60.0, -6.0), (sphere_helpers.minimal_hemisphere(), 80.0, 0.0)])
    with pytest.raises(ValueError, match='flight_path'):
        _build(directory, tmp_path / 'x.nod', database_frame='flight_path')


def test_validate_nod_refuses_an_unknown_frame(tmp_path):
    database = _build(_folder(tmp_path / 'spheres'), tmp_path / 'x.nod', database_frame='flight_path')
    with Dataset(str(database), 'a') as handle:
        handle.sphere_frame = 'tip_path'
    assert [v.rule for v in validate_nod(str(database))] == ['C2']


def test_the_cli_takes_the_database_frame(tmp_path):
    args = script.parse_arguments([str(tmp_path), str(tmp_path / 'db.nod'), '--database-frame', 'flight_path'])
    assert script.build_options(args)['database_frame'] == 'flight_path'


def test_an_averaged_sphere_keeps_its_type_and_run(tmp_path):
    paths = []
    for i, weight in enumerate((4000.0, 5000.0)):
        path = tmp_path / f'NB{i}.nc'
        sphere_helpers.write_wopwop_sphere(path, _levels(60.0), PHI, THETA, np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
                                           kind='NB', speed=60.0, gross_weight_lb=weight)
        paths.append(str(path))
    average = sphere_statistics.average_spheres(paths)
    out = tmp_path / 'average.nc'
    sphere_statistics.write_averaged_sphere(str(out), average)
    with Dataset(str(out)) as handle:
        assert handle.sphere_type == 'NB'
        assert (float(handle['BB'][...]), float(handle['NB'][...])) == (0.0, 1.0)
        assert handle.sphere_frame == 'flight_path'
        assert float(handle['GROSS_WEIGHT'][...]) == pytest.approx(4500.0)
        assert float(handle['AIR_DENSITY'][...]) == pytest.approx(1.07, rel=1e-6)


# ---- NICEOPS: the same footprint from either frame --------------------------------------------

NICEOPS, NICEOPS_SKIP = sphere_helpers.niceops()


def _descent_track(path, speed_kt=80.0, angle_deg=-6.0, seconds=60.0, step=0.5):
    """A steady straight descent in NICEOPS's 27-column AC format (feet, ft/s)."""
    v = speed_kt * 1.68781
    vx, vz = v * np.cos(np.radians(angle_deg)), v * np.sin(np.radians(angle_deg))
    rows = []
    for t in np.arange(0.0, seconds + 1e-9, step):
        x, z = -0.5 * seconds * vx + vx * t, 1500.0 + vz * (t - 0.5 * seconds)
        row = [t, t, np.nan, np.nan, np.nan, x, 0.0, z, 90.0, 0.0, 0.0, vx, 0.0, 0.0, 0.0, vz, 0.0,
               v, v, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        rows.append(','.join('{:.8f}'.format(value) for value in row))
    header = ('utcsec,time,lat,lon,alt,x,y,z,heading,pitch,roll,vx,vxk,vy,vyk,vz,vzk,V,VG,Vk,VGk,ax,ay,az,'
              'Uatt,Upos,Uvel')
    path.write_text(header + '\n' + '\n'.join(rows) + '\n')
    return path


def _footprint(database, track, cwd, name):
    output = cwd / (name + '.csv')
    result = subprocess.run([NICEOPS, '-d', str(database), '-t', str(track), '-o', str(output),
                             '--speed_reference', 'ground', '--no-ground_effect', '-l', name + '.log'],
                            cwd=cwd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, (cwd / (name + '.log')).read_text()[-2000:] + result.stderr
    # A header line, the x row, the y row, then the levels (x across, y down).
    rows = output.read_text().splitlines()[3:]
    return np.array([[float(v) for v in row.split(',') if v.strip()] for row in rows if row.strip()])


@pytest.mark.skipif(NICEOPS is None, reason=NICEOPS_SKIP)
def test_niceops_hears_the_same_descent_from_either_frame(tmp_path):
    directory = _folder(tmp_path / 'spheres')
    level = _build(directory, tmp_path / 'level.nod')
    flight = _build(directory, tmp_path / 'flight.nod', database_frame='flight_path')
    track = _descent_track(tmp_path / 'descent.csv')
    a = _footprint(level, track, tmp_path, 'level')
    b = _footprint(flight, track, tmp_path, 'flight')
    assert a.shape == b.shape and a.size > 1000
    loud = a > a.max() - 20.0
    difference = np.abs(a - b)[loud]
    print('largest difference within 20 dB of the loudest: {:.3f} dB'.format(difference.max()))
    assert difference.max() < 0.3
