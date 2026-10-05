"""Per-run metadata: what a sphere records about the run it came from, and what a database
carries per condition group (gross weight, air density, wind at the aircraft,
advance_ratio_air, thrust_coefficient_run).  None of it changes a level."""

import os
import shutil
import warnings

import numpy as np
import pytest
from netCDF4 import Dataset

import flight_acoustics as fa
import noise_abatement_2017 as na
from panam_acoustics.atmosphere import Atmosphere

from test_database_provenance import FREQUENCY, PHI, THETA, VEHICLE_CFG, _write_sphere


# --- air density ------------------------------------------------------------------------

def test_dry_air_density_is_the_standard_atmospheres():
    assert na.air_density_kg_m3(Atmosphere(288.15, 101.325, 0.0)) == pytest.approx(1.2250, abs=2e-4)


def test_humidity_and_height_lower_the_density():
    dry = na.air_density_kg_m3(Atmosphere(303.15, 88.0, 0.0))
    humid = na.air_density_kg_m3(Atmosphere(303.15, 88.0, 80.0))
    e = 0.8 * Atmosphere(303.15, 88.0, 80.0).saturation_pressure * 1000.0
    assert humid == pytest.approx((88000.0 - 0.378 * e) / (287.058 * 303.15), rel=1e-12)
    assert humid < dry
    # Hydrostatic at the measured (virtual) temperature: about 3 % per 1000 ft here.
    tv = 303.15 / (1.0 - 0.378 * e / 88000.0)
    aloft = na.air_density_kg_m3(Atmosphere(303.15, 88.0, 80.0), height_ft=1000.0)
    assert aloft == pytest.approx(humid * np.exp(-9.80665 * 304.8 / (287.058 * tv)), rel=1e-12)
    assert 0.96 < aloft / humid < 0.975


def test_an_impossible_atmosphere_has_no_density():
    assert np.isnan(na.air_density_kg_m3(Atmosphere(0.0, 101.325, 20.0)))


# --- wind -----------------------------------------------------------------------------

def test_a_meteorological_wind_blows_toward_the_opposite_bearing():
    wind = na.normalize_wind(dict(speed=10.0, direction_from_deg=0.0, units='kt', source='LIDAR'))
    assert wind['east'] == pytest.approx(0.0, abs=1e-12) and wind['north'] == pytest.approx(-10.0)
    assert wind['source'] == 'lidar' and wind['units'] == 'kt'
    wind = na.normalize_wind(dict(speed=4.0, direction_from_deg=270.0, units='m/s', source='balloon+lidar'))
    assert wind['east'] == pytest.approx(4.0) and wind['north'] == pytest.approx(0.0, abs=1e-12)
    assert wind['source'] == 'balloon+lidar'
    assert na.normalize_wind(dict(east=1.0, north=2.0, units='mph', source='station'))['north'] == 2.0


@pytest.mark.parametrize('wind, message', [
    (dict(speed=5.0, direction_from_deg=10.0, source='lidar'), 'units must be declared'),
    (dict(speed=5.0, direction_from_deg=10.0, units='knots', source='lidar'), 'units must be declared'),
    (dict(speed=5.0, direction_from_deg=10.0, units='kt', source='radar'), 'wind source'),
    (dict(speed=5.0, direction_from_deg=10.0, units='kt'), 'wind source'),
    (dict(speed=5.0, direction_from_deg=10.0, units='kt', source='none'), 'is no wind'),
    (dict(speed=5.0, direction_from_deg=10.0, east=1.0, north=1.0, units='kt', source='lidar'), 'not both'),
    (dict(speed=5.0, units='kt', source='lidar'), 'not neither'),
    (dict(speed=-5.0, direction_from_deg=10.0, units='kt', source='lidar'), 'negative'),
    (dict(east=np.nan, north=1.0, units='kt', source='lidar'), 'finite'),
])
def test_a_wind_must_say_what_it_is(wind, message):
    with pytest.raises(ValueError, match=message):
        na.normalize_wind(wind)


def test_wind_components_are_along_and_starboard_of_the_reference():
    """Flying east: a wind toward the east is a tailwind; toward the south (the
    starboard side of an eastbound track) a positive crosswind."""
    ground = np.array([[100.0 * 1.68781, 0.0]])                  # 100 kt east, ft/s
    tail = na.normalize_wind(dict(east=10.0, north=0.0, units='kt', source='lidar'))
    along, cross, airspeed = na.wind_components(tail, (1.0, 0.0), ground)
    assert (along, cross) == (pytest.approx(10.0), pytest.approx(0.0, abs=1e-12))
    assert airspeed == pytest.approx(90.0, rel=1e-4)
    south = na.normalize_wind(dict(speed=10.0, direction_from_deg=0.0, units='kt', source='lidar'))
    along, cross, airspeed = na.wind_components(south, (1.0, 0.0), ground)
    assert along == pytest.approx(0.0, abs=1e-12) and cross == pytest.approx(10.0)
    assert airspeed == pytest.approx(np.hypot(100.0, 10.0), rel=1e-4)
    # The components stay in the declared unit; only the airspeed converts.
    metric = na.normalize_wind(dict(east=-5.0, north=0.0, units='m/s', source='lidar'))
    along, _, airspeed = na.wind_components(metric, (1.0, 0.0), ground)
    assert along == pytest.approx(-5.0) and airspeed == pytest.approx(100.0 + 5.0 / 0.514444, rel=1e-4)


def test_a_winds_file_is_read_and_checked(tmp_path):
    path = tmp_path / 'winds.csv'
    path.write_text('run,east,north,speed,direction_from_deg,units,source\n'
                    '284202,,,8.4,250,kt,lidar\n'
                    '284203,1.5,-2.0,,,m/s,balloon\n'
                    '284204,,,,,kt,lidar\n')
    winds = na.read_winds(str(path))
    assert set(winds) == {'284202', '284203'}
    assert winds['284203'] == dict(east=1.5, north=-2.0, units='m/s', source='balloon')
    path.write_text('run,speed,direction_from_deg,units,source\n284202,8.4,250,knots,lidar\n')
    with pytest.raises(ValueError, match='line 2'):
        na.read_winds(str(path))


# --- the sphere file ------------------------------------------------------------------

def _hemisphere():
    azi = np.arange(0.0, 360.0 + 1e-9, 10.0)
    elv = np.arange(0.0, 90.0 + 1e-9, 10.0)
    bands = np.full((FREQUENCY.size, elv.size, azi.size), 70.0)
    return {'azi_grid_deg': azi, 'elv_grid_deg': elv,
            'third_octave': {'band_centers_hz': FREQUENCY, 'bands_db': bands},
            'metadata': {'r_ref': 100.0, 'length_units': 'ft'}}


METADATA = dict(gross_weight_lb=3505.0, air_density_kg_m3=1.0412, air_density_source='ground stations',
                wind_along_track=-8.4, wind_cross_track=2.5, wind_units='kt', wind_source='lidar',
                wind_reference_direction='ground track', airspeed_knots=88.6)


def test_a_sphere_carries_its_run_metadata(tmp_path):
    path = str(tmp_path / 'sphere.nc')
    fa.write_aam_hemisphere_netcdf(path, _hemisphere(), mode='third_octave', speed_knots=80.0,
                                   azimuth_reference='heading', run_metadata=METADATA)
    read = fa.read_run_metadata(path)
    assert read == dict(METADATA, azimuth_reference='heading')
    with Dataset(path) as sphere:
        assert sphere['WIND_ALONG_TRACK'].unit.strip() == 'kt'
        assert sphere['GROSS_WEIGHT'].unit.strip() == 'POUNDS'
        assert sphere['AIR_DENSITY'].unit.strip() == 'KG/M^3'
        assert sphere['GROSS_WEIGHT'].dtype == np.float64
    # Unknowns are NaN, not zero.
    partial = str(tmp_path / 'partial.nc')
    fa.write_aam_hemisphere_netcdf(partial, _hemisphere(), mode='third_octave',
                                   run_metadata=dict(gross_weight_lb=3400.0, air_density_kg_m3=None))
    read = fa.read_run_metadata(partial)
    assert read['gross_weight_lb'] == 3400.0 and np.isnan(read['air_density_kg_m3'])
    assert np.isnan(read['wind_along_track']) and read['wind_units'] == '' and read['azimuth_reference'] is None


def test_a_sphere_without_metadata_reads_as_unknown(tmp_path):
    path = str(tmp_path / 'old.nc')
    fa.write_aam_hemisphere_netcdf(path, _hemisphere(), mode='third_octave')
    read = fa.read_run_metadata(path)
    assert all(np.isnan(read[key]) for _, _, key in fa.RUN_METADATA_VARIABLES)
    assert read['wind_source'] == '' and read['azimuth_reference'] is None


@pytest.mark.parametrize('metadata, message', [
    (dict(METADATA, wind_units=''), 'wind_units'),
    (dict(METADATA, wind_units='knots'), 'wind_units'),
    (dict(METADATA, wind_source='none'), 'no wind_source'),
    (dict(METADATA, wind_source='sodar'), 'wind source'),
    (dict(METADATA, weight=1.0), 'unknown run_metadata'),
])
def test_a_sphere_refuses_metadata_it_cannot_label(tmp_path, metadata, message):
    with pytest.raises(ValueError, match=message):
        fa.write_aam_hemisphere_netcdf(str(tmp_path / 's.nc'), _hemisphere(), mode='third_octave',
                                       run_metadata=metadata)
    with pytest.raises(ValueError, match='azimuth_reference'):
        fa.write_aam_hemisphere_netcdf(str(tmp_path / 's.nc'), _hemisphere(), mode='third_octave',
                                       azimuth_reference='nose')


# --- the database -----------------------------------------------------------------------

def _labeled_sphere(path, speed, fpa, metadata):
    amplitude = np.full((PHI.size, THETA.size, FREQUENCY.size), 80.0)
    _write_sphere(path, amplitude, speed=speed, flight_path_angle=fpa, doppler=0.0)
    with Dataset(str(path), 'a') as ds:
        for name, unit, key in fa.RUN_METADATA_VARIABLES:
            if key in metadata:
                ds.createVariable(name, 'f8').assignValue(metadata[key])
        for name in fa.RUN_METADATA_ATTRIBUTES:
            if name in metadata:
                ds.setncattr(name, metadata[name])


def _groups_by_origin(db):
    out = {}
    for group in db.groups.values():
        out.setdefault(group.getncattr('condition_origin'), []).append(group)
    return out


def test_every_group_carries_its_runs_metadata(tmp_path):
    directory = tmp_path / 'spheres'
    directory.mkdir()
    (directory / 'vehicle.cfg').write_text(VEHICLE_CFG)
    _labeled_sphere(directory / 'A.nc', 60.0, 0.0, METADATA)
    _labeled_sphere(directory / 'B.nc', 90.0, -6.0, dict(gross_weight_lb=3300.0))      # no density, no wind
    database = str(tmp_path / 'db.nod')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fa.build_empirical_database(str(directory), database, load_factors=[1.0, 2.0],
                                    extended_flight_path_angles=(-24.0, 35.0))
    radius, tip_speed = 5.345, 218.0
    weight = 3505.0 * fa.POUND_FORCE_NEWTONS
    ct_run = weight / (1.0412 * np.pi * radius ** 2 * tip_speed ** 2)
    with Dataset(database) as db:
        groups = _groups_by_origin(db)
        measured = {(g.getncattr('source_sphere'), float(g['thrust_coefficient'][0])): g
                    for g in groups['measured']}
        assert len(measured) == 4                                 # two spheres x two load factors
        for load_factor in (1.0, 2.0):
            a = next(g for (name, ct), g in measured.items() if name == 'A.nc'
                     and np.isclose(ct, load_factor * float(db.groups['sphere0']['thrust_coefficient'][0])))
            assert float(a['gross_weight'][0]) == pytest.approx(weight)
            assert float(a['air_density'][0]) == pytest.approx(1.0412)
            assert float(a['wind_along_track'][0]) == -8.4 and float(a['wind_cross_track'][0]) == 2.5
            # The run's own C_T, not scaled by the group's load factor.
            assert float(a['thrust_coefficient_run'][0]) == pytest.approx(ct_run, rel=1e-12)
            assert float(a['advance_ratio_air'][0]) == pytest.approx(0.514444 * 88.6 / tip_speed, rel=1e-12)
            assert a['advance_ratio'][0] == pytest.approx(0.514444 * 60.0 / tip_speed)
            assert a['gross_weight'].units == 'N' and a['air_density'].units == 'kg m-3'
            assert a['wind_along_track'].units == 'kt' and a['advance_ratio_air'].units == '1'
            assert a.getncattr('wind_source') == 'lidar' and a.getncattr('air_density_source') == 'ground stations'
            assert a.getncattr('wind_reference_direction') == 'ground track'
        b = next(g for (name, _), g in measured.items() if name == 'B.nc')
        assert float(b['gross_weight'][0]) == pytest.approx(3300.0 * fa.POUND_FORCE_NEWTONS)
        for name in ('air_density', 'wind_along_track', 'wind_cross_track', 'advance_ratio_air',
                     'thrust_coefficient_run'):
            assert np.isnan(float(b[name][0])), name
        assert b.getncattr('wind_source') == 'none' and b.getncattr('air_density_source') == 'none'

        # The level sphere is re-emitted at the extended angles with its own metadata.
        extended = groups['extended_flight_path_angle']
        assert extended and {g.getncattr('source_sphere') for g in extended} == {'A.nc'}
        assert all(float(g['advance_ratio_air'][0]) == pytest.approx(0.514444 * 88.6 / tip_speed) for g in extended)
        # The hover keeps the source run's loading but no wind or flight condition.
        hover = groups['synthesized_hover']
        assert hover and {g.getncattr('source_sphere') for g in hover} == {'A.nc'}
        for g in hover:
            assert float(g['thrust_coefficient_run'][0]) == pytest.approx(ct_run, rel=1e-12)
            assert np.isnan(float(g['advance_ratio_air'][0])) and np.isnan(float(g['wind_along_track'][0]))
            assert g.getncattr('wind_source') == 'none'


def test_the_metadata_moves_no_level(tmp_path):
    """The same spheres with and without metadata give the same levels and conditions."""
    plain, labeled = tmp_path / 'plain', tmp_path / 'labeled'
    for directory, metadata in ((plain, {}), (labeled, METADATA)):
        directory.mkdir()
        (directory / 'vehicle.cfg').write_text(VEHICLE_CFG)
        _labeled_sphere(directory / 'A.nc', 60.0, 0.0, metadata)
        _labeled_sphere(directory / 'B.nc', 90.0, -6.0, metadata)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for directory in (plain, labeled):
            fa.build_empirical_database(str(directory), str(directory / 'db.nod'), load_factors=None)
    with Dataset(str(plain / 'db.nod')) as a, Dataset(str(labeled / 'db.nod')) as b:
        assert set(a.groups) == set(b.groups)
        for name in a.groups:
            for variable in ('dBA', 'EAA', 'amplitude', 'advance_ratio', 'flight_path_angle',
                             'thrust_coefficient', 'coverage'):
                np.testing.assert_array_equal(a.groups[name][variable][:], b.groups[name][variable][:])


# --- through the build ----------------------------------------------------------------

def test_the_sphere_build_records_weight_density_and_wind(tmp_path):
    """The synthetic archive's pass flies along +x, west on a 270 deg frame, at 150 ft/s.
    A wind from the west at 10 kt is a 10 kt headwind; the airspeed is the ground speed
    plus it.  The archive has no weather stations, so the density is unknown."""
    from test_noise_abatement_2017 import _archive, _write_csv
    _archive(tmp_path)
    base = tmp_path / 'AS350B3'
    _write_csv(base / 'AS350B3FullRefList.csv',
               ['combined', 'test_cond', 'layout', 'bank_ang', 'accel_rate', 'utc_secs_from_mid_start',
                'run_num', 'true_heading', 'gross_weight'],
               [['289101', 'AMB', 'A', '0', '0', '60.0', '101', '270', ''],
                ['289108', 'L1', 'A', '0', '0', '100.0', '108', '270', '4257']])
    manifest = tmp_path / 'manifest.csv'
    records, failures = na.build_all(
        'AS350B3', str(tmp_path / 'out'), root=str(tmp_path), prefetch=False, board_correction='flat',
        manifest_path=str(manifest),
        winds={'289108': dict(speed=10.0, direction_from_deg=270.0, units='kt', source='lidar')})
    assert failures == []
    read = fa.read_run_metadata(records[0]['output'])
    ground_knots = 150.0 * 0.3048 / 0.514444
    assert read['gross_weight_lb'] == 4257.0
    assert read['wind_along_track'] == pytest.approx(-10.0) and read['wind_cross_track'] == pytest.approx(0.0, abs=1e-9)
    assert read['airspeed_knots'] == pytest.approx(ground_knots + 10.0, rel=1e-6)
    assert read['wind_units'] == 'kt' and read['wind_source'] == 'lidar'
    assert read['wind_reference_direction'] == 'ground track'
    assert np.isnan(read['air_density_kg_m3']) and read['air_density_source'] == 'none'
    assert read['azimuth_reference'] == 'track'
    assert records[0]['airspeed_knots'] == pytest.approx(ground_knots + 10.0, rel=1e-6)
    # A caller's atmosphere gives a density, labeled as the caller's.
    test = na.NoiseAbatementTest('AS350B3', root=str(tmp_path))
    record = na.build_sphere(test, '289108', str(tmp_path / 'caller.nc'), board_correction='flat',
                             atmosphere=Atmosphere(293.15, 88.0, 40.0))
    read = fa.read_run_metadata(record['output'])
    assert read['air_density_source'] == 'caller'
    assert read['air_density_kg_m3'] == pytest.approx(na.air_density_kg_m3(Atmosphere(293.15, 88.0, 40.0), 300.0))
    assert np.isnan(read['wind_along_track']) and read['wind_source'] == 'none'


def test_a_wind_needs_the_frame_bearing(tmp_path):
    from test_noise_abatement_2017 import _archive
    _archive(tmp_path)                                    # its reference list has no true_heading
    test = na.NoiseAbatementTest('AS350B3', root=str(tmp_path))
    with pytest.raises(ValueError, match='no true_heading'):
        na.build_sphere(test, '289108', str(tmp_path / 'x.nc'), board_correction='flat',
                        wind=dict(speed=5.0, direction_from_deg=0.0, units='kt', source='lidar'))
