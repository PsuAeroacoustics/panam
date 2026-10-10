import glob
import os
import re

import numpy as np
import pytest

import flight_acoustics as fa
import local_paths
from panam_acoustics.atmosphere import Atmosphere

_NORAH2 = local_paths.data_path('norah2', required=False)
NORAH2_HEMISPHERES = os.path.join(_NORAH2, 'Hemispheres') if _NORAH2 else ''
needs_norah2 = pytest.mark.skipif(not os.path.isdir(NORAH2_HEMISPHERES),
                                  reason='NORAH2 distribution not configured (local_paths: norah2)')

MEASURED = dict(temperature_k=288.15, pressure_kpa=98.0, relative_humidity=45.0)


def synthetic_hemisphere(level_db=70.0, band_centers_hz=None, starboard_boost_db=0.0,
                         apply_absorption_deprop=True):
    """A UMAPR hemisphere, uniform apart from an optional louder starboard side."""
    azi_grid_deg = np.arange(0.0, 360.0 + 1e-9, 5.0)
    elv_grid_deg = np.arange(-5.0, 90.0 + 1e-9, 5.0)
    if band_centers_hz is None:
        # Exact base-10 centers, 12.6 Hz .. 5 kHz: tests the nominal-band match
        # and leaves 10 Hz and 6.3-10 kHz for the writer to mark missing.
        band_centers_hz = 1000.0 * 10.0 ** (np.arange(-19, 8) / 10.0)
    AZI = np.broadcast_to(azi_grid_deg, (elv_grid_deg.size, azi_grid_deg.size))
    starboard = np.sin(np.radians(AZI)) * np.cos(np.radians(elv_grid_deg))[:, None]
    field = level_db + starboard_boost_db * starboard
    bands_db = np.broadcast_to(field, (band_centers_hz.size,) + field.shape).copy()
    return {
        'azi_grid_deg': azi_grid_deg,
        'elv_grid_deg': elv_grid_deg,
        'third_octave': {'band_centers_hz': band_centers_hz, 'bands_db': bands_db},
        'metadata': {
            'r_ref': 100.0,
            'length_units': 'ft',
            'apply_absorption_deprop': apply_absorption_deprop,
            'atmosphere': dict(MEASURED),
        },
    }


def test_norah2umapr_body_axes():
    """theta 0 is the nose, phi > 0 starboard, (phi 0, theta 90) straight down."""
    phi = np.radians([0.0, 0.0, 0.0, 90.0, -90.0])
    theta = np.radians([0.0, 180.0, 90.0, 90.0, 90.0])
    azi, elv = fa.norah2umapr(phi, theta)
    assert np.allclose(np.degrees(azi[[0, 1, 3, 4]]), [180.0, 0.0, 90.0, 270.0], atol=1e-9)
    assert np.allclose(np.degrees(elv), [0.0, 0.0, 90.0, 0.0, 0.0], atol=1e-9)


def test_norah2_and_art_angles_agree():
    """NORAH2 and AAM define (phi, theta) identically: phi > 0 starboard, theta 0 nose."""
    PH, TH = np.meshgrid(np.radians(np.arange(-80.0, 81.0, 20.0)), np.radians(np.arange(10.0, 171.0, 20.0)))
    azi_n, elv_n = fa.norah2umapr(PH, TH)
    azi_a, elv_a = fa.art2umapr(PH, TH)
    assert np.allclose(elv_n, elv_a, atol=1e-9)
    assert np.allclose(np.cos(azi_n - azi_a), 1.0, atol=1e-9)


def test_levels_converted_to_60m_with_icao_absorption(tmp_path):
    hemisphere = synthetic_hemisphere(level_db=70.0)
    path = tmp_path / 'X_Flyover_80kts_0deg.hem'
    written = fa.write_norah2_hemisphere(str(path), hemisphere, speed_knots=80.0,
                                         flight_path_angle_deg=0.0, minimum_level_db=-np.inf)
    hem = fa.load_norah2_hemisphere(str(path))
    assert hem['levels_db'].shape == (19, 19, 31)
    assert np.allclose(hem['frequency_hz'], fa.NORAH2_BAND_CENTERS_HZ)

    measured = Atmosphere(temperature=MEASURED['temperature_k'], pressure=MEASURED['pressure_kpa'],
                          relative_humidity=MEASURED['relative_humidity'])
    reference = Atmosphere(temperature=298.15, pressure=101.325, relative_humidity=70.0)
    r_ref_m = 100.0 * 0.3048
    exact = hemisphere['third_octave']['band_centers_hz']
    nominal = fa.NORAH2_BAND_CENTERS_HZ[1:28]
    expected = (70.0 + 20.0 * np.log10(r_ref_m / 60.0)
                + measured.attenuation_coefficient(exact) * r_ref_m
                - reference.attenuation_coefficient(nominal) * 60.0)

    # Every direction of the lower hemisphere is covered, so all of it is filled.
    levels = hem['levels_db']
    assert np.allclose(levels[:, :, 1:28], expected[None, None, :], atol=0.051)
    assert np.allclose(written[:, :, 1:28], expected[None, None, :], atol=1e-6)
    # Bands the hemisphere does not have are NORAH2's no-value.
    assert np.all(np.isneginf(levels[:, :, 0]))
    assert np.all(np.isneginf(levels[:, :, 28:]))


def test_starboard_stays_starboard(tmp_path):
    hemisphere = synthetic_hemisphere(starboard_boost_db=6.0)
    path = tmp_path / 'asym.hem'
    fa.write_norah2_hemisphere(str(path), hemisphere, speed_knots=60.0, flight_path_angle_deg=-6.0)
    hem = fa.load_norah2_hemisphere(str(path))
    at_90 = hem['levels_db'][:, list(hem['theta_deg']).index(90.0), 10]
    phi = hem['phi_deg']
    assert at_90[phi == 90.0][0] - at_90[phi == -90.0][0] == pytest.approx(12.0, abs=0.2)
    assert np.all(np.diff(at_90) > 0.0)


def test_header_matches_shipped_layout(tmp_path):
    hemisphere = synthetic_hemisphere()
    short = tmp_path / 'short.hem'
    fa.write_norah2_hemisphere(str(short), hemisphere, speed_knots=66.1551, flight_path_angle_deg=-8.34327,
                               title='AS350_Approach_66kts_8.3deg', test_point='TP27+28')
    raw = short.read_bytes()
    assert b'\r\n' in raw and b'\n' not in raw.replace(b'\r\n', b'')
    lines = raw.decode('ascii').split('\r\n')
    assert lines[0] == 'AS350_Approach_66kts_8.3deg\tTP27+28\t'
    assert lines[1].split()[0] == '14'
    names = [line.split()[0] for line in lines[2:16]]
    assert names == ['POLDIST', 'FREEFIELD', 'NOVALUE', 'TAMB', 'RELHUM', 'PAMB', 'Tm', 'RHm', 'Pm',
                     'RmOmega', 'ACSPEED', 'GAMM', 'TW', 'CW']
    assert lines.count('PHIOBSAC= -90.000000') == 1
    assert sum(1 for line in lines if line.startswith('PHIOBSAC=')) == 19

    hem = fa.load_norah2_hemisphere(str(short))
    constants = hem['constants']
    assert constants['POLDIST'] == 60.0 and constants['FREEFIELD'] == 2.0 and constants['NOVALUE'] == -999.0
    assert constants['ACSPEED'] == pytest.approx(66.1551)
    assert constants['GAMM'] == pytest.approx(-8.34327)
    assert constants['Tm'] == pytest.approx(15.0)
    assert constants['RHm'] == pytest.approx(45.0)
    assert constants['Pm'] == pytest.approx(98000.0)

    full = tmp_path / 'full.hem'
    fa.write_norah2_hemisphere(str(full), hemisphere, speed_knots=67.1, flight_path_angle_deg=9.2,
                               pitch_deg=-4.6, roll_deg=1.5, head_wind_knots=0.1)
    names = [line.split()[0] for line in full.read_text().splitlines()[2:19]]
    assert names[12:] == ['PITCH', 'ROLL', 'TW', 'CW', 'HW']
    assert fa.load_norah2_hemisphere(str(full))['constants']['HW'] == pytest.approx(0.1)


def test_requires_third_octave_and_warns_without_absorption_deprop(tmp_path):
    hemisphere = synthetic_hemisphere()
    narrow = {k: v for k, v in hemisphere.items() if k != 'third_octave'}
    with pytest.raises(ValueError, match='third_octave'):
        fa.write_norah2_hemisphere(str(tmp_path / 'n.hem'), narrow, speed_knots=0.0,
                                   flight_path_angle_deg=0.0)
    lossy = synthetic_hemisphere(apply_absorption_deprop=False)
    with pytest.warns(UserWarning, match='absorption depropagation'):
        written = fa.write_norah2_hemisphere(str(tmp_path / 'l.hem'), lossy, speed_knots=0.0,
                                             flight_path_angle_deg=0.0)
    assert np.allclose(written[:, :, 5], 70.0 + 20.0 * np.log10(30.48 / 60.0))


def test_overwrite_false_refuses(tmp_path):
    path = tmp_path / 'x.hem'
    path.write_text('existing')
    with pytest.raises(FileExistsError):
        fa.write_norah2_hemisphere(str(path), synthetic_hemisphere(), speed_knots=0.0,
                                   flight_path_angle_deg=0.0, overwrite=False)


def test_triangulation_file(tmp_path):
    hemispheres = [('A.hem', 50.0, -6.0), ('B.hem', 80.0, -6.0), ('C.hem', 50.0, 0.0),
                   ('D.hem', 80.0, 0.0), ('E.hem', 65.0, 6.0)]
    path = tmp_path / 'X_Triangulation.int'
    triangles = fa.write_norah2_triangulation(str(path), hemispheres)
    text = path.read_bytes().decode('ascii')
    assert '\r\n' in text
    lines = text.split('\r\n')
    assert lines[:5] == ['&HEMISPHERES', '\tNGAD = 5', '&END', '', 'iHem\tHEMSpeed\tHEMAngle\t[Path\\]FileHem']
    assert lines[5] == '1\t50\t-6\tA.hem'
    assert f'\tNTRI = {len(triangles)}' in lines
    assert '8\t"Outoffgroundhover"' in lines
    assert len(triangles) == 3

    with pytest.raises(ValueError, match='share a flight condition'):
        fa.write_norah2_triangulation(str(tmp_path / 'dup.int'), hemispheres + [('F.hem', 80.0, 0.0)])


@pytest.mark.parametrize('conditions', [
    [(40.0, 0.0), (60.0, 0.0), (80.0, 0.0)],          # level runs at one flight path angle
    [(60.0, -6.0), (60.0, 0.0), (60.0, 6.0)],         # one speed
    [(40.0, -3.0), (60.0, 0.0), (80.0, 3.0)],         # a sloping line
])
def test_triangulation_refuses_collinear_conditions(tmp_path, conditions):
    """A ValueError, which build_directory reports as a warning, not scipy's QhullError,
    which would abort it before its manifest is written."""
    path = tmp_path / 'line.int'
    entries = [(f'{i}.hem', speed, angle) for i, (speed, angle) in enumerate(conditions)]
    with pytest.raises(ValueError, match='one line'):
        fa.write_norah2_triangulation(str(path), entries)
    assert not path.exists()


@pytest.mark.data
@needs_norah2
def test_reads_every_shipped_flight_condition_hemisphere():
    files = [f for f in glob.glob(os.path.join(NORAH2_HEMISPHERES, '*.hem'))
             if 'hover' not in os.path.basename(f).lower()]
    assert len(files) > 200
    for path in files:
        hem = fa.load_norah2_hemisphere(path)
        assert hem['levels_db'].shape == (19, 19, 31), path
        assert np.allclose(hem['phi_deg'], fa.NORAH2_PHI_DEG)
        assert np.allclose(hem['theta_deg'], fa.NORAH2_THETA_DEG)
        assert np.allclose(hem['frequency_hz'], fa.NORAH2_BAND_CENTERS_HZ)
        assert hem['constants']['POLDIST'] == 60.0


@pytest.mark.data
@needs_norah2
@pytest.mark.parametrize('path', sorted(glob.glob(os.path.join(NORAH2_HEMISPHERES, '*_[Tt]riangulation.int'))))
def test_reproduces_shipped_triangulations(tmp_path, path):
    text = open(path).read().replace('\r', '')
    table = re.search(r'iHem\tHEMSpeed.*?\n(.*?)\n\s*\n', text, re.S).group(1).strip().split('\n')
    entries = [(row.split()[3], float(row.split()[1]), float(row.split()[2])) for row in table]
    # Special-operation hemispheres (R22's in-ground hover) are listed but not meshed.
    entries = [e for e in entries if 'hover' not in e[0].lower()]
    shipped = re.search(r'iTri\tiHem1.*?\n(.*?)\n\s*\n', text, re.S).group(1).strip().split('\n')
    shipped = {frozenset(int(v) - 1 for v in row.split()[1:4]) for row in shipped}
    triangles = fa.write_norah2_triangulation(str(tmp_path / 'out.int'), entries)
    assert {frozenset(int(v) for v in t) for t in triangles} == shipped
