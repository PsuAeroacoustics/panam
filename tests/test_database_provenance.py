"""What build_empirical_database records about how a database was built.

NICE-OPS reads four things that panam now writes: the root attributes
speed_reference and build_temperature_K / build_pressure_kPa /
build_relative_humidity_percent, and per condition group DOPPLER_SHIFT_REMOVED
and coverage.  These tests pin their names, types and shapes, that a mixed
Doppler convention is refused, that coverage marks exactly the gated bins, and
that a build without the new arguments differs from the old one only by the
added items.

The spheres here are written by hand, not by write_aam_hemisphere_netcdf: that
writer interpolates the hemisphere onto the sphere grid, which would smear a
gated bin into its neighbors and make "exactly where the gate zeroes a bin"
untestable.
"""

import os
import shutil
import warnings

import numpy as np
import pytest
from netCDF4 import Dataset

import flight_acoustics as fa
from panam_acoustics.atmosphere import Atmosphere


SPHERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'example_data', 'AS350B3108.nc')

VEHICLE_CFG = """[Main Rotor]
blades = 3
radius = 5.345
tip speed = 218.0
[Tail Rotor]
blades = 2
radius = 0.93
tip speed = 202.0
[Atmosphere]
temperature = 288.15
density = 1.225
[Vehicle]
weight = 2250.0
drag = 1.0
"""

PHI = np.arange(-90.0, 90.0 + 1e-9, 10.0)       # 19 source azimuths, lower half only
THETA = np.arange(0.0, 180.0 + 1e-9, 10.0)      # 19 polar angles
FREQUENCY = np.array([125.0, 500.0, 2000.0, 8000.0])
MISSING = -999.0                                 # the AAM sentinel: reads back as -inf

# Root attributes and group variables that existed before this change.  The
# default build must produce exactly these, plus the added items below.
PRE_EXISTING_ROOT_VARIABLES = {
    'same_grid', 'fixed_load_factor', 'database_version', 'main_rotor_radius_meters',
    'main_rotor_tip_speed_meters_per_sec', 'vehicle_weight_newtons',
    'shared_grid_and_frequency', 'phi', 'theta', 'frequency',
}
PRE_EXISTING_GROUP_VARIABLES = {
    'radius', 'rotor_scale', 'advance_ratio', 'flight_path_angle', 'thrust_coefficient',
    'dBA', 'EAA', 'amplitude',
}
ADDED_ROOT_ATTRIBUTES = {'speed_reference', 'build_temperature_K', 'build_pressure_kPa',
                         'build_relative_humidity_percent'}
ADDED_GROUP_VARIABLES = {'DOPPLER_SHIFT_REMOVED', 'coverage'}


def _write_sphere(path, amplitude, speed=70.0, flight_path_angle=0.0, doppler=None):
    """A sphere file with the AAM variable names, amplitude given as written."""
    with Dataset(str(path), 'w', format='NETCDF3_CLASSIC') as ds:
        ds.createDimension('PHI', PHI.size)
        ds.createDimension('THETA', THETA.size)
        ds.createDimension('FREQUENCY', FREQUENCY.size)
        ds.createVariable('PHI', 'f4', ('PHI',))[:] = PHI
        ds.createVariable('THETA', 'f4', ('THETA',))[:] = THETA
        ds.createVariable('FREQUENCY', 'f4', ('FREQUENCY',))[:] = FREQUENCY
        ds.createVariable('AMPLITUDE', 'f4', ('PHI', 'THETA', 'FREQUENCY'))[:] = amplitude
        ds.createVariable('RADIUS', 'f4').assignValue(100.0)
        ds.createVariable('SPEED', 'f4').assignValue(speed)
        ds.createVariable('FLIGHT_PATH_ANGLE', 'f4').assignValue(flight_path_angle)
        if doppler is not None:
            ds.createVariable('DOPPLER_SHIFT_REMOVED', 'f4').assignValue(doppler)


def _source_directory(tmp_path, spheres):
    """spheres: a mapping of file name to keyword arguments for _write_sphere."""
    directory = tmp_path / 'source'
    directory.mkdir()
    (directory / 'vehicle.cfg').write_text(VEHICLE_CFG)
    for name, options in spheres.items():
        options = dict(options)
        amplitude = options.pop('amplitude', np.full((PHI.size, THETA.size, FREQUENCY.size), 80.0))
        _write_sphere(directory / name, amplitude, **options)
    return directory


def _gated_amplitude():
    """80 dB everywhere, with a known pattern of gated bins.

    Gated: one bin in the middle of the lower hemisphere, one bin at a grid edge,
    one whole direction at the aft (theta = 180) edge with a single band lost,
    and one band gated in a direction whose partner is fully measured, so the
    fore-to-aft averaging has something to fill from.
    """
    amplitude = np.full((PHI.size, THETA.size, FREQUENCY.size), 80.0)
    amplitude[5, 3, 2] = MISSING           # interior bin
    amplitude[0, 0, 0] = MISSING           # corner
    amplitude[12, 18, 1] = MISSING         # aft edge, one band
    amplitude[9, 7, 3] = MISSING           # a band lost in one direction only
    return amplitude


def _gate_mask(amplitude):
    """True where the file's bins are gated, the independent reading of the gate."""
    return amplitude <= fa.AAM_MISSING_THRESHOLD


def _source_index(phi_channel, theta_channel):
    """The source (i_phi, i_theta) of a completed channel, or None if it is mirrored."""
    matches = np.flatnonzero(np.isclose(PHI, phi_channel))
    if matches.size == 0:
        return None
    return int(matches[0]), int(np.flatnonzero(np.isclose(THETA, theta_channel))[0])


def _expected_coverage(source_measured, phi_channels, theta_channels):
    """Per channel and band, what coverage should read, from the source mask alone.

    A channel is measured only if it is a source row (the lower hemisphere) and the
    bin is measured.  Built by lookup from the channel's own angles, so it does not
    depend on how the builder orders its rows.
    """
    expected = np.zeros((phi_channels.size, FREQUENCY.size), dtype=np.int8)
    for c, (phi_c, theta_c) in enumerate(zip(phi_channels, theta_channels)):
        index = _source_index(phi_c, theta_c)
        if index is None:
            continue
        expected[c] = source_measured[index[0], index[1], :]
    return expected


def _fore_aft_partner_mask(measured):
    """A fore-to-aft averaged mask: a pair is measured only where both partners are."""
    return measured & measured[:, ::-1, :]


def _root_channels(db):
    return np.asarray(db['phi'][:], dtype=float), np.asarray(db['theta'][:], dtype=float)


# --- (a) attributes, variables, dtypes and shapes -----------------------------------


def test_root_attributes_and_group_variables_have_their_types_and_shapes(tmp_path):
    directory = _source_directory(tmp_path, {'A100.nc': dict(doppler=0.0)})
    database = tmp_path / 'database.nod'
    fa.build_empirical_database(str(directory), str(database), load_factors=None)

    with Dataset(str(database)) as db:
        assert db.getncattr('speed_reference') == 'ground'
        assert isinstance(db.getncattr('speed_reference'), str)
        for name in ('build_temperature_K', 'build_pressure_kPa',
                     'build_relative_humidity_percent'):
            assert np.asarray(db.getncattr(name)).dtype == np.float64, name
        assert db.getncattr('build_temperature_K') == 293.15
        assert db.getncattr('build_pressure_kPa') == 101.325
        assert db.getncattr('build_relative_humidity_percent') == 20.0
        # Optional items, kept on every build; the format version is unchanged.
        assert int(db['database_version'][()]) == fa.DATABASE_FORMAT_VERSION == 1
        # No rotor RPM in the source spheres, so none is invented.
        assert 'rotor_rpm' not in db.ncattrs()

        # The completed sphere: the 19 source azimuths plus the 17 mirrored ones (the
        # +/-90 edges are shared), 36 in all, each with every polar angle.
        n_channels = (PHI.size + PHI.size - 2) * THETA.size
        assert n_channels == 684
        assert db.dimensions['channels'].size == n_channels
        for group in db.groups.values():
            flag = group['DOPPLER_SHIFT_REMOVED']
            assert flag.dtype == np.int32
            assert flag.dimensions == ()
            assert int(flag[()]) == 0
            coverage = group['coverage']
            assert coverage.dtype == np.int8
            # The spectrum's own dimensions: amplitude is (PHI, THETA, frequency).
            assert coverage.dimensions == ('PHI', 'THETA', 'frequency')
            assert coverage.shape == (PHI.size + PHI.size - 2, THETA.size, FREQUENCY.size)
            assert group['amplitude'].dimensions == coverage.dimensions
            assert set(np.unique(coverage[:])) <= {0, 1}


def test_build_arguments_are_written_as_given(tmp_path):
    directory = _source_directory(tmp_path, {'A100.nc': dict(doppler=0.0)})
    database = tmp_path / 'air.nod'
    atmosphere = Atmosphere(temperature=280.0, pressure=100.0, relative_humidity=50.0)
    fa.build_empirical_database(str(directory), str(database), load_factors=None,
                                speed_reference='air', atmosphere=atmosphere)
    with Dataset(str(database)) as db:
        assert db.getncattr('speed_reference') == 'air'
        # The atmosphere the EAA was computed in, not a copy of the defaults.
        assert db.getncattr('build_temperature_K') == 280.0
        assert db.getncattr('build_pressure_kPa') == 100.0
        assert db.getncattr('build_relative_humidity_percent') == 50.0


def test_unknown_speed_reference_is_refused(tmp_path):
    directory = _source_directory(tmp_path, {'A100.nc': {}})
    database = tmp_path / 'bad.nod'
    with pytest.raises(ValueError, match='speed_reference'):
        fa.build_empirical_database(str(directory), str(database), load_factors=None,
                                    speed_reference='wind')
    assert not database.exists()


def test_doppler_flag_is_written_per_group_from_the_source(tmp_path):
    directory = _source_directory(tmp_path, {'A100.nc': dict(doppler=1.0)})
    database = tmp_path / 'stationary.nod'
    fa.build_empirical_database(str(directory), str(database), load_factors=None)
    with Dataset(str(database)) as db:
        assert all(int(group['DOPPLER_SHIFT_REMOVED'][()]) == 1 for group in db.groups.values())


def test_doppler_argument_fills_in_sources_that_carry_no_flag(tmp_path):
    """A sphere with no flag takes the argument; with none given it is 0, with a warning."""
    directory = _source_directory(tmp_path, {'A100.nc': {}})
    with pytest.warns(UserWarning, match='DOPPLER_SHIFT_REMOVED'):
        fa.build_empirical_database(str(directory), str(tmp_path / 'default.nod'), load_factors=None)
    fa.build_empirical_database(str(directory), str(tmp_path / 'given.nod'), load_factors=None,
                                doppler_shift_removed=1)
    with Dataset(str(tmp_path / 'default.nod')) as default, Dataset(str(tmp_path / 'given.nod')) as given:
        assert int(next(iter(default.groups.values()))['DOPPLER_SHIFT_REMOVED'][()]) == 0
        assert int(next(iter(given.groups.values()))['DOPPLER_SHIFT_REMOVED'][()]) == 1


# --- (b) mixed Doppler conventions are refused --------------------------------------


def test_mixed_doppler_flags_are_refused_and_no_database_is_left(tmp_path):
    directory = _source_directory(tmp_path, {'A100.nc': dict(doppler=0.0),
                                             'A101.nc': dict(doppler=1.0, speed=90.0)})
    database = tmp_path / 'mixed.nod'
    with pytest.raises(ValueError, match='differs between the sources'):
        fa.build_empirical_database(str(directory), str(database), load_factors=None)
    assert not database.exists()


def test_flag_missing_from_one_sphere_counts_as_received_frame_and_still_mixes(tmp_path):
    """A sphere without the flag is taken as 0, so it cannot be combined with a 1."""
    directory = _source_directory(tmp_path, {'A100.nc': dict(doppler=1.0),
                                             'A101.nc': dict(speed=90.0)})
    with pytest.warns(UserWarning, match='DOPPLER_SHIFT_REMOVED'):
        with pytest.raises(ValueError, match='differs between the sources'):
            fa.build_empirical_database(str(directory), str(tmp_path / 'x.nod'), load_factors=None)


def test_argument_that_disagrees_with_a_sphere_is_refused(tmp_path):
    directory = _source_directory(tmp_path, {'A100.nc': dict(doppler=0.0)})
    with pytest.raises(ValueError, match='asked for 1'):
        fa.build_empirical_database(str(directory), str(tmp_path / 'x.nod'), load_factors=None,
                                    doppler_shift_removed=1)
    assert not (tmp_path / 'x.nod').exists()


def test_doppler_argument_must_be_0_or_1(tmp_path):
    directory = _source_directory(tmp_path, {'A100.nc': dict(doppler=0.0)})
    with pytest.raises(ValueError, match='0, 1 or None'):
        fa.build_empirical_database(str(directory), str(tmp_path / 'x.nod'), load_factors=None,
                                    doppler_shift_removed=2)


def test_doppler_flag_outside_0_and_1_in_a_sphere_is_refused(tmp_path):
    directory = _source_directory(tmp_path, {'A100.nc': dict(doppler=0.5)})
    with pytest.raises(ValueError, match='must be 0 or 1'):
        fa.build_empirical_database(str(directory), str(tmp_path / 'x.nod'), load_factors=None)


# --- (c) coverage marks exactly the gated bins --------------------------------------


def test_coverage_is_zero_exactly_where_the_gate_zeroes_a_bin(tmp_path):
    amplitude = _gated_amplitude()
    directory = _source_directory(tmp_path, {'A100.nc': dict(amplitude=amplitude, speed=70.0)})
    database = tmp_path / 'coverage.nod'
    fa.build_empirical_database(str(directory), str(database), load_factors=None,
                                store_spectrum=True)

    gated = _gate_mask(amplitude)
    measured = ~gated
    with Dataset(str(database)) as db:
        phi_channels, theta_channels = _root_channels(db)
        forward = db.groups['sphere0']
        assert float(forward['advance_ratio'][0]) > 0.0            # a forward-flight group
        coverage = np.asarray(forward['coverage'][:], dtype=np.int8).reshape(phi_channels.size, -1)
        expected = _expected_coverage(measured, phi_channels, theta_channels)
        np.testing.assert_array_equal(coverage, expected)

        # The reading that matters, spelled out once: 0 at the four gated bins...
        for i, j, b in [(5, 3, 2), (0, 0, 0), (12, 18, 1), (9, 7, 3)]:
            channel = np.flatnonzero(np.isclose(phi_channels, PHI[i]) & np.isclose(theta_channels, THETA[j]))[0]
            assert coverage[channel, b] == 0, (i, j, b)
        # ...1 at a bin nothing touched...
        channel = np.flatnonzero(np.isclose(phi_channels, PHI[2]) & np.isclose(theta_channels, THETA[2]))[0]
        assert coverage[channel, 0] == 1
        # ...and 0 on every mirrored (never measured) row.
        mirrored = ~np.isin(phi_channels, PHI)
        assert mirrored.any()
        assert np.all(coverage[mirrored] == 0)
        # Gating a band does not empty its direction: the level comes from the bands
        # that were measured, so it stays finite.
        partial = np.flatnonzero(np.isclose(phi_channels, PHI[5]) & np.isclose(theta_channels, THETA[3]))[0]
        assert np.isfinite(forward['dBA'][partial])


def test_hover_coverage_is_measured_only_where_both_fore_and_aft_partners_are(tmp_path):
    """The hover sphere averages each direction with its partner.  A gated bin fills
    its partner, so neither cell is measured in that band."""
    amplitude = _gated_amplitude()
    directory = _source_directory(tmp_path, {'A100.nc': dict(amplitude=amplitude, speed=70.0,
                                                             flight_path_angle=0.0)})
    database = tmp_path / 'hover.nod'
    fa.build_empirical_database(str(directory), str(database), load_factors=None,
                                store_spectrum=True)

    measured = _fore_aft_partner_mask(~_gate_mask(amplitude))
    with Dataset(str(database)) as db:
        phi_channels, theta_channels = _root_channels(db)
        hover_groups = [g for g in db.groups.values() if float(g['advance_ratio'][0]) == 0.0]
        assert hover_groups
        expected = _expected_coverage(measured, phi_channels, theta_channels)
        for group in hover_groups:
            np.testing.assert_array_equal(
                np.asarray(group['coverage'][:], dtype=np.int8).reshape(phi_channels.size, -1), expected)
        # The gated bin at theta index 3 fills its aft partner (index 15), so both are unmeasured.
        assert not measured[5, 3, 2] and not measured[5, 15, 2]


def test_without_spectra_a_cell_is_measured_only_when_every_band_is(tmp_path):
    amplitude = _gated_amplitude()
    directory = _source_directory(tmp_path, {'A100.nc': dict(amplitude=amplitude, speed=70.0)})
    database = tmp_path / 'levels_only.nod'
    fa.build_empirical_database(str(directory), str(database), load_factors=None,
                                store_spectrum=False)
    with Dataset(str(database)) as db:
        phi_channels, theta_channels = _root_channels(db)
        group = db.groups['sphere0']
        assert 'frequency' not in group.dimensions
        assert group['coverage'].dimensions == ('channels',)
        expected = _expected_coverage(~_gate_mask(amplitude), phi_channels, theta_channels).all(axis=1)
        np.testing.assert_array_equal(np.asarray(group['coverage'][:], dtype=np.int8), expected)
        # The cell with one band gated is unmeasured as a whole.
        channel = np.flatnonzero(np.isclose(phi_channels, PHI[5]) & np.isclose(theta_channels, THETA[3]))[0]
        assert group['coverage'][channel] == 0


def test_a_fully_gated_direction_is_unmeasured_and_has_no_energy(tmp_path):
    amplitude = np.full((PHI.size, THETA.size, FREQUENCY.size), 80.0)
    amplitude[4, 6, :] = MISSING
    directory = _source_directory(tmp_path, {'A100.nc': dict(amplitude=amplitude, speed=70.0)})
    database = tmp_path / 'empty.nod'
    fa.build_empirical_database(str(directory), str(database), load_factors=None)
    with Dataset(str(database)) as db:
        phi_channels, theta_channels = _root_channels(db)
        group = db.groups['sphere0']
        channel = np.flatnonzero(np.isclose(phi_channels, PHI[4]) & np.isclose(theta_channels, THETA[6]))[0]
        coverage = np.asarray(group['coverage'][:], dtype=np.int8).reshape(phi_channels.size, -1)
        assert np.all(coverage[channel, :] == 0)
        assert np.isneginf(group['dBA'][channel])
        assert np.isfinite(group['EAA'][channel])


# --- (d) the default build differs from the old one only by the added items ----------


def test_default_build_differs_from_the_old_one_only_by_the_added_items(tmp_path):
    """Not a comparison against a stored file: the pre-existing values are pinned to
    the same pipeline the old builder ran, and the set of names is pinned to what it
    wrote.  The added items are exactly the four attributes and the two variables."""
    if not os.path.exists(SPHERE):
        pytest.skip('example sphere data not available')
    directory = tmp_path / 'source'
    directory.mkdir()
    (directory / 'vehicle.cfg').write_text(VEHICLE_CFG)
    shutil.copy(SPHERE, directory / 'AS350A.nc')

    default_path = tmp_path / 'default.nod'
    explicit_path = tmp_path / 'explicit.nod'
    fa.build_empirical_database(str(directory), str(default_path), load_factors=None)
    fa.build_empirical_database(str(directory), str(explicit_path), load_factors=None,
                                speed_reference='ground', doppler_shift_removed=None)

    with Dataset(str(default_path)) as db:
        root_names = set(db.variables)
        assert root_names == PRE_EXISTING_ROOT_VARIABLES
        assert ADDED_ROOT_ATTRIBUTES <= set(db.ncattrs())
        for group in db.groups.values():
            assert set(group.variables) == PRE_EXISTING_GROUP_VARIABLES | ADDED_GROUP_VARIABLES

        # The pre-existing numbers are the old pipeline's, bit for bit: a forward
        # sphere's dBA is the completed SPLA (no load-factor offset at LF = 1), and
        # EAA is the completed EAA with non-finite values zeroed.
        (_, _, phi_list, theta_list, _, _, spla, eaa, speed, _, frequency, amplitude) = fa.extract_SPL(
            str(directory / 'AS350A.nc'), None, 1000, Atmosphere(temperature=293.15, pressure=101.325,
                                                                  relative_humidity=20.0))
    spla_c, eaa_c = fa._finite_sphere_levels(spla, eaa)
    _, _, spla_full, eaa_full, _ = fa._complete_sphere(phi_list, theta_list, spla_c, eaa_c, amplitude)
    with Dataset(str(default_path)) as db:
        forward = [g for g in db.groups.values() if float(g['advance_ratio'][0]) > 0.0]
        assert forward
        np.testing.assert_array_equal(np.asarray(forward[0]['dBA'][:], dtype=float),
                                      spla_full.reshape(-1))
        np.testing.assert_array_equal(np.asarray(forward[0]['EAA'][:], dtype=float),
                                      eaa_full.reshape(-1))

    with Dataset(str(default_path)) as default, Dataset(str(explicit_path)) as explicit:
        for name in default.variables:
            np.testing.assert_array_equal(np.asarray(default[name][:]), np.asarray(explicit[name][:]))
        assert {k: default.getncattr(k) for k in default.ncattrs()} == \
            {k: explicit.getncattr(k) for k in explicit.ncattrs()}
        for group in default.groups:
            for name in default.groups[group].variables:
                np.testing.assert_array_equal(np.asarray(default.groups[group][name][:]),
                                              np.asarray(explicit.groups[group][name][:]))
