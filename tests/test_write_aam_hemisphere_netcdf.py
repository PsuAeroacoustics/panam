import numpy as np
import pytest

import flight_acoustics as fa
from netCDF4 import Dataset


def test_write_aam_hemisphere_netcdf_roundtrip_third_octave(tmp_path):
    # Minimal synthetic depropagate_hemisphere-like output
    azi_grid_deg = np.array([0.0, 90.0, 180.0, 270.0, 360.0])
    elv_grid_deg = np.array([0.0, 45.0, 90.0])

    # Two bands, simple spatial pattern; include -inf sentinel
    bands_db = np.zeros((2, elv_grid_deg.size, azi_grid_deg.size), dtype=float)
    bands_db[0, :, :] = 50.0
    bands_db[1, :, :] = 60.0
    bands_db[0, 0, 0] = -np.inf

    hemisphere = {
        'azi_grid_deg': azi_grid_deg,
        'elv_grid_deg': elv_grid_deg,
        'third_octave': {
            'band_centers_hz': np.array([100.0, 200.0]),
            'bands_db': bands_db,
        },
        'metadata': {
            'r_ref': 100.0,
            'length_units': 'ft',
            'freq_range_hz': (0.0, 2000.0),
            'window_time': 0.5,
            'window_overlap': 0.5,
            'point_stride': 1,
            'rmax_deg': 25.0,
            'apply_absorption_deprop': False,
            'flip_y_for_geometry': False,
        },
    }

    out_nc = tmp_path / 'synthetic_aam_hemisphere.nc'
    fa.write_aam_hemisphere_netcdf(
        str(out_nc),
        hemisphere,
        mode='third_octave',
        speed_knots=0.0,
        flight_path_angle_deg=0.0,
        overwrite=True,
    )

    amp, phi, theta, freq, radius, speed, fpa = fa.load_nc_sphere(str(out_nc))

    assert amp.ndim == 3
    assert phi.ndim == 1
    assert theta.ndim == 1
    assert freq.ndim == 1
    assert amp.shape == (phi.size, theta.size, freq.size)

    assert freq.size == 2
    assert np.allclose(freq, np.array([100.0, 200.0]))

    # Scalars stored as 1-element arrays in this format
    assert np.asarray(radius).size == 1
    assert np.asarray(speed).size == 1
    assert np.asarray(fpa).size == 1

    # Verify the AAM-style auxiliary fields exist and use 'unit' attrs
    with Dataset(str(out_nc), 'r') as ds:
        assert ds.data_model == 'NETCDF3_CLASSIC'
        for vname in [
            'BB', 'NB', 'PT', 'DOPPLER_SHIFT_REMOVED',
            'EMPTY_WEIGHT', 'FUEL_WEIGHT', 'LOAD_WEIGHT',
            'PYLON_ANGLE', 'MASTTILT', 'XYZ',
        ]:
            assert vname in ds.variables
        assert 'unit' in ds.variables['PHI'].ncattrs()


# The shipped AAM/RNM spheres (Sphere_Database/<aircraft>/*.nc) are the contract
# a regenerated sphere has to satisfy: AAM reads these names and attributes
# directly.  Captured from Be407220.nc, AS350269.nc and RO-66500.nc, which agree
# field for field.
AAM_REQUIRED_DIMENSIONS = {
    'PHI', 'THETA', 'FREQUENCY', 'XYZ', 'BB', 'NB', 'PT',
    'DOPPLER_SHIFT_REMOVED', 'EMPTY_WEIGHT', 'FUEL_WEIGHT', 'LOAD_WEIGHT',
    'RADIUS', 'FLIGHT_PATH_ANGLE', 'PYLON_ANGLE', 'SPEED', 'MASTTILT',
}

AAM_REQUIRED_UNITS = {
    # Array variables carry an unpadded unit...
    'PHI': 'DEGREE',
    'THETA': 'DEGREE',
    'FREQUENCY': 'HERTZ',
    'AMPLITUDE': 'DECIBEL',
    'XYZ': 'FEET',
    # ...scalars are blank padded to 20 characters.
    'RADIUS': 'FEET'.ljust(20),
    'SPEED': 'KNOTS'.ljust(20),
    'FLIGHT_PATH_ANGLE': 'DEGREE'.ljust(20),
    'PYLON_ANGLE': 'DEGREE'.ljust(20),
    'MASTTILT': 'DEGREE'.ljust(20),
    'EMPTY_WEIGHT': 'POUNDS'.ljust(20),
    'FUEL_WEIGHT': 'POUNDS'.ljust(20),
    'LOAD_WEIGHT': 'POUNDS'.ljust(20),
    'BB': ''.ljust(20),
    'NB': ''.ljust(20),
    'PT': ''.ljust(20),
    'DOPPLER_SHIFT_REMOVED': ''.ljust(20),
}


def _minimal_hemisphere():
    band_centers = np.array([100.0, 125.0, 160.0])
    elv = np.arange(0.0, 90.0 + 1e-9, 30.0)
    azi = np.arange(0.0, 360.0 + 1e-9, 60.0)
    bands = np.full((band_centers.size, elv.size, azi.size), 80.0)
    return dict(azi_grid_deg=azi, elv_grid_deg=elv,
                third_octave=dict(band_centers_hz=band_centers, bands_db=bands))


def test_written_sphere_carries_every_field_aam_reads(tmp_path):
    path = tmp_path / 'sphere.nc'
    fa.write_aam_hemisphere_netcdf(
        str(path), _minimal_hemisphere(), mode='third_octave',
        phi_deg=np.arange(-90.0, 90.0 + 1e-9, 10.0),
        theta_deg=np.arange(0.0, 180.0 + 1e-9, 5.0),
        radius_ft=100.0, speed_knots=70.0, flight_path_angle_deg=-6.0,
        title='TEST   Run 1 01/01/2017 x'.ljust(77))

    with Dataset(str(path), 'r') as sphere:
        assert sphere.file_format == 'NETCDF3_CLASSIC'
        assert AAM_REQUIRED_DIMENSIONS <= set(sphere.dimensions)
        assert set(AAM_REQUIRED_UNITS) <= set(sphere.variables)
        for name, unit in AAM_REQUIRED_UNITS.items():
            assert sphere.variables[name].unit == unit, name
        # AMPLITUDE must not declare _FillValue: no shipped sphere has one.
        assert '_FillValue' not in sphere.variables['AMPLITUDE'].ncattrs()
        assert sphere.variables['AMPLITUDE'].dimensions == ('PHI', 'THETA', 'FREQUENCY')
        for name in ('RADIUS', 'SPEED', 'FLIGHT_PATH_ANGLE'):
            assert sphere.variables[name].shape == ()
        assert sphere.variables['XYZ'].shape == (3,)
        assert 'title' in sphere.ncattrs()


def test_missing_bands_are_written_as_the_aam_sentinel(tmp_path):
    """Ambient gating leaves bands with no data; they must read back as no energy."""
    hemisphere = _minimal_hemisphere()
    hemisphere['third_octave']['bands_db'][0, :, :] = -np.inf

    path = tmp_path / 'gapped.nc'
    fa.write_aam_hemisphere_netcdf(str(path), hemisphere, mode='third_octave',
                                   radius_ft=100.0, title='t')

    amplitude, _, _, _, _, _, _ = fa.load_nc_sphere(str(path))
    raw = np.asarray(amplitude, dtype=float)
    # The AAM code masks known-bad values with -999, so that is what goes on disk.
    assert np.allclose(raw[:, :, 0], fa.AAM_MISSING_LEVEL)
    masked = fa.mask_missing_levels(raw)
    assert np.all(np.isneginf(masked[:, :, 0]))
    assert np.all(np.isfinite(masked[:, :, 1]))


def test_database_carries_the_root_vehicle_fields_niceops_reads(tmp_path):
    """The shipped .nod files all carry these; NICE-OPS cross-checks tip speed
    against them and prefers them for --export_aam."""
    import flight_acoustics as fa_mod

    sphere_dir = tmp_path / 'spheres'
    sphere_dir.mkdir()
    (sphere_dir / 'vehicle.cfg').write_text(
        '[Main Rotor]\nradius = 5.334\ntip speed = 230.7\nblades = 4\n'
        '[Tail Rotor]\nradius = 0.8255\ntip speed = 216.1\nblades = 2\n'
        '[Atmosphere]\ndensity = 1.070\ntemperature = 280.37\n'
        '[Vehicle]\nweight = 2250\ndrag = 0.8175\n')

    fa_mod.write_aam_hemisphere_netcdf(
        str(sphere_dir / 'X100.nc'), _minimal_hemisphere(), mode='third_octave',
        phi_deg=np.arange(-90.0, 90.0 + 1e-9, 10.0),
        theta_deg=np.arange(0.0, 180.0 + 1e-9, 10.0),
        radius_ft=100.0, speed_knots=70.0, flight_path_angle_deg=0.0, title='t')

    out = tmp_path / 'db.nod'
    fa_mod.build_empirical_database(str(sphere_dir), str(out),
                                    load_factors=np.array([1.0]), store_spectrum=False)

    with Dataset(str(out), 'r') as db:
        assert 'same_grid' in db.variables
        assert float(db['main_rotor_radius_meters'][0]) == 5.334
        assert float(db['main_rotor_tip_speed_meters_per_sec'][0]) == 230.7
        assert float(db['vehicle_weight_newtons'][0]) == fa.STANDARD_GRAVITY * 2250


def _sphere_dir_with_two_conditions(tmp_path):
    sphere_dir = tmp_path / 'spheres'
    sphere_dir.mkdir()
    (sphere_dir / 'vehicle.cfg').write_text(
        '[Main Rotor]\nradius = 5.334\ntip speed = 230.7\nblades = 4\n'
        '[Tail Rotor]\nradius = 0.8255\ntip speed = 216.1\nblades = 2\n'
        '[Atmosphere]\ndensity = 1.070\ntemperature = 280.37\n'
        '[Vehicle]\nweight = 2250\ndrag = 0.8175\n')
    for name, speed, fpa in [('A100.nc', 70.0, 0.0), ('A101.nc', 90.0, -6.0)]:
        fa.write_aam_hemisphere_netcdf(
            str(sphere_dir / name), _minimal_hemisphere(), mode='third_octave',
            phi_deg=np.arange(-90.0, 90.0 + 1e-9, 10.0),
            theta_deg=np.arange(0.0, 180.0 + 1e-9, 10.0),
            radius_ft=100.0, speed_knots=speed, flight_path_angle_deg=fpa, title='t')
    return sphere_dir


def test_load_factors_none_writes_one_group_per_condition_at_the_lf1_reference(tmp_path):
    """See the docstring: every load factor add_sphere_group would otherwise
    materialise is the *same* spectrum at a uniform dB offset, so with
    load_factors=None NICE-OPS is expected to reproduce them analytically
    instead of finding them stored."""
    sphere_dir = _sphere_dir_with_two_conditions(tmp_path)

    fixed_path = tmp_path / 'fixed.nod'
    fa.build_empirical_database(str(sphere_dir), str(fixed_path), load_factors=None,
                                store_spectrum=False)
    explicit_lf1_path = tmp_path / 'explicit_lf1.nod'
    fa.build_empirical_database(str(sphere_dir), str(explicit_lf1_path),
                                load_factors=np.array([1.0]), store_spectrum=False)
    two_factor_path = tmp_path / 'two_factor.nod'
    fa.build_empirical_database(str(sphere_dir), str(two_factor_path),
                                load_factors=np.array([1.0, 2.0]), store_spectrum=False)

    with Dataset(str(fixed_path)) as fixed, Dataset(str(explicit_lf1_path)) as explicit_lf1, \
            Dataset(str(two_factor_path)) as two_factor:
        assert bool(fixed['fixed_load_factor'][()])
        assert not bool(explicit_lf1['fixed_load_factor'][()])
        assert not bool(two_factor['fixed_load_factor'][()])

        # 2 base conditions + 3 default hover flight-path angles = 5, doubled
        # when the same conditions are written at 2 explicit load factors.
        assert len(fixed.groups) == 5
        assert len(explicit_lf1.groups) == 5
        assert len(two_factor.groups) == 10

        # load_factors=None must be bit-for-bit the LF=1 slice of the old
        # per-load-factor form, not merely "close": every dBA/EAA/thrust_coefficient
        # it writes already is that reference, so there is nothing to average or round.
        for name in fixed.groups:
            fixed_group = fixed.groups[name]
            explicit_group = explicit_lf1.groups[name]
            for var in ('dBA', 'EAA', 'thrust_coefficient', 'advance_ratio', 'flight_path_angle'):
                np.testing.assert_array_equal(fixed_group[var][:], explicit_group[var][:])


def test_denormal_band_power_is_written_as_missing_not_as_minus_3000_db(tmp_path):
    """Gated hemispheres interpolate to tiny positive power, not exact zero.

    Without a floor those become finite levels near -3000 dB, which read back as
    real data: a Be407 descent sphere had 68% of its directions at -3064 dBA.
    """
    hemisphere = _minimal_hemisphere()
    # One band left with essentially no energy anywhere, as gating produces.
    hemisphere['third_octave']['bands_db'][0, :, :] = -3000.0

    path = tmp_path / 'tiny.nc'
    fa.write_aam_hemisphere_netcdf(str(path), hemisphere, mode='third_octave',
                                   radius_ft=100.0, title='t')
    amplitude, _, _, _, _, _, _ = fa.load_nc_sphere(str(path))
    masked = fa.mask_missing_levels(amplitude)
    assert np.all(np.isneginf(masked[:, :, 0])), 'near-zero band must read as missing'
    assert np.all(np.isfinite(masked[:, :, 1]))

    # Opting out keeps the finite level.
    kept = tmp_path / 'kept.nc'
    hemisphere['third_octave']['bands_db'][0, :, :] = -500.0
    fa.write_aam_hemisphere_netcdf(str(kept), hemisphere, mode='third_octave',
                                   radius_ft=100.0, title='t',
                                   minimum_level_db=-np.inf)
    raw = np.asarray(fa.load_nc_sphere(str(kept))[0], dtype=float)
    assert raw[0, 0, 0] < -100.0
    # -500 is a level, not the mask, so it survives the read as one.
    assert np.isfinite(fa.mask_missing_levels(raw)[0, 0, 0])


def test_empty_directions_leave_a_usable_eaa(tmp_path):
    """A gated-out direction must not put a NaN in EAA.

    Checked against NICE-OPS itself: a -inf dBA is harmless (its exposure
    integral sums energy, so the direction simply contributes nothing), but a
    NaN EAA put 886 NaN cells into a 120x120 footprint. So the level is left
    alone, including -inf, and only EAA is given a value.
    """
    import flight_acoustics as fa_mod

    sphere_dir = tmp_path / 'spheres'
    sphere_dir.mkdir()
    (sphere_dir / 'vehicle.cfg').write_text(
        '[Main Rotor]\nradius = 5.334\ntip speed = 230.7\nblades = 4\n'
        '[Tail Rotor]\nradius = 0.8255\ntip speed = 216.1\nblades = 2\n'
        '[Atmosphere]\ndensity = 1.070\ntemperature = 280.37\n'
        '[Vehicle]\nweight = 2250\ndrag = 0.8175\n')

    hemisphere = _minimal_hemisphere()
    hemisphere['third_octave']['bands_db'][:, 0, :] = -np.inf   # a whole elevation ring gated out
    fa_mod.write_aam_hemisphere_netcdf(
        str(sphere_dir / 'X100.nc'), hemisphere, mode='third_octave',
        phi_deg=np.arange(-90.0, 90.0 + 1e-9, 10.0),
        theta_deg=np.arange(0.0, 180.0 + 1e-9, 10.0),
        radius_ft=100.0, speed_knots=70.0, flight_path_angle_deg=0.0, title='t')

    out = tmp_path / 'db.nod'
    fa_mod.build_empirical_database(str(sphere_dir), str(out),
                                    load_factors=np.array([1.0]), store_spectrum=False)
    with Dataset(str(out), 'r') as db:
        assert db.groups
        for name in db.groups:
            dba = np.asarray(db.groups[name]['dBA'][:], dtype=float)
            eaa = np.asarray(db.groups[name]['EAA'][:], dtype=float)
            assert not np.isnan(dba).any(), name
            assert np.all(np.isfinite(eaa)), name
        # The emptied ring is reported as no energy, not clamped up to a level.
        levels = np.concatenate([np.asarray(db.groups[n]['dBA'][:], dtype=float)
                                 for n in db.groups])
        assert np.isneginf(levels).any() or levels.min() < -100.0


def test_every_masking_convention_in_the_wild_reads_as_no_energy():
    """NaN (shipped 2017 spheres), -999 (AAM), and a large positive (old panam)."""
    amplitude = np.array([[[10.0, np.nan, -999.0, 1.0e35, -998.5, -20.0]]])
    masked = fa.mask_missing_levels(amplitude)
    assert masked[0, 0, 0] == 10.0
    assert np.isneginf(masked[0, 0, 1])
    assert np.isneginf(masked[0, 0, 2])
    assert np.isneginf(masked[0, 0, 3])
    assert np.isneginf(masked[0, 0, 4])
    # A genuine low level is not swallowed by the mask.
    assert masked[0, 0, 5] == -20.0
    # The input is left alone and the result is never a masked array.
    assert not isinstance(masked, np.ma.MaskedArray)


def _sphere_dir_with_mismatched_grids(tmp_path):
    """Like _sphere_dir_with_two_conditions, but the second condition's sphere
    is completed onto a different theta grid -- the "not always" case
    shared_grid_and_frequency has to detect rather than assume.
    """
    sphere_dir = tmp_path / 'spheres_mismatched'
    sphere_dir.mkdir()
    (sphere_dir / 'vehicle.cfg').write_text(
        '[Main Rotor]\nradius = 5.334\ntip speed = 230.7\nblades = 4\n'
        '[Tail Rotor]\nradius = 0.8255\ntip speed = 216.1\nblades = 2\n'
        '[Atmosphere]\ndensity = 1.070\ntemperature = 280.37\n'
        '[Vehicle]\nweight = 2250\ndrag = 0.8175\n')
    conditions = [
        ('A100.nc', 70.0, 0.0, np.arange(0.0, 180.0 + 1e-9, 10.0)),
        ('A101.nc', 90.0, -6.0, np.arange(0.0, 180.0 + 1e-9, 15.0)),
    ]
    for name, speed, fpa, theta_deg in conditions:
        fa.write_aam_hemisphere_netcdf(
            str(sphere_dir / name), _minimal_hemisphere(), mode='third_octave',
            phi_deg=np.arange(-90.0, 90.0 + 1e-9, 10.0), theta_deg=theta_deg,
            radius_ft=100.0, speed_knots=speed, flight_path_angle_deg=fpa, title='t')
    return sphere_dir


def test_shared_grid_and_frequency_is_detected_and_deduplicated(tmp_path):
    """When every condition shares one phi/theta grid and one frequency axis
    (the normal case -- see build_empirical_database's docstring), they are
    written once at the root instead of once per condition, and NICE-OPS is
    expected to read them from there instead of expecting them per group.
    """
    sphere_dir = _sphere_dir_with_two_conditions(tmp_path)

    shared_path = tmp_path / 'shared.nod'
    fa.build_empirical_database(str(sphere_dir), str(shared_path), load_factors=None,
                                store_spectrum=True)

    with Dataset(str(shared_path)) as ds:
        assert bool(ds['shared_grid_and_frequency'][()])
        assert 'phi' in ds.variables
        assert 'theta' in ds.variables
        assert 'frequency' in ds.variables
        root_phi = ds['phi'][:]
        root_theta = ds['theta'][:]
        root_frequency = ds['frequency'][:]
        assert len(ds.groups) == 5  # 2 conditions + 3 default hover angles
        for group in ds.groups.values():
            # Deduplicated away: every condition's own copy would otherwise
            # duplicate the root arrays exactly.
            assert 'phi' not in group.variables
            assert 'theta' not in group.variables
            assert 'frequency' not in group.variables
            assert 'dBA' in group.variables
            assert 'amplitude' in group.variables
            # A group's amplitude still resolves the shared root PHI/THETA/
            # frequency dimensions (netCDF4 groups inherit their ancestors'
            # dimensions) rather than defining its own: phi/theta are the
            # flattened per-channel arrays (PHI * THETA long), while
            # amplitude keeps the unflattened (PHI, THETA) grid shape.
            assert group['amplitude'].shape[0] * group['amplitude'].shape[1] == root_phi.size
            assert group['amplitude'].shape[2] == root_frequency.size


def test_mismatched_grids_fall_back_to_per_condition_storage(tmp_path):
    """The "not always" case: conditions whose sphere grids genuinely differ
    must not be deduplicated, or every condition but the first would silently
    lose whatever its grid disagreed on.
    """
    sphere_dir = _sphere_dir_with_mismatched_grids(tmp_path)

    path = tmp_path / 'mismatched.nod'
    fa.build_empirical_database(str(sphere_dir), str(path), load_factors=None,
                                store_spectrum=True)

    with Dataset(str(path)) as ds:
        assert not bool(ds['shared_grid_and_frequency'][()])
        assert 'phi' not in ds.variables
        assert 'theta' not in ds.variables
        assert 'frequency' not in ds.variables
        for group in ds.groups.values():
            assert 'phi' in group.variables
            assert 'theta' in group.variables
            assert 'frequency' in group.variables


def test_export_keeps_starboard_and_ahead_where_they_are(tmp_path):
    """A hemisphere loud to starboard / ahead must export loud at phi > 0 / theta < 90.

    Every other hemisphere in this file is uniform in azimuth, so a mirrored
    export (port for starboard, or tail for nose) passed all of them.
    """
    azi = np.arange(0.0, 361.0, 10.0)
    elv = np.arange(0.0, 91.0, 10.0)
    AZI, _ = np.meshgrid(azi, elv)
    for loud, axis_test in (((AZI > 45.0) & (AZI < 135.0), 'phi'),       # UMAPR 90 = starboard
                            ((AZI > 135.0) & (AZI < 225.0), 'theta')):   # UMAPR 180 = ahead
        bands = np.where(loud, 80.0, 60.0)[None, :, :]
        hemisphere = {'azi_grid_deg': azi, 'elv_grid_deg': elv,
                      'third_octave': {'band_centers_hz': np.array([1000.0]), 'bands_db': bands},
                      'metadata': {'r_ref': 100.0, 'length_units': 'ft'}}
        path = tmp_path / f'loud_{axis_test}.nc'
        fa.write_aam_hemisphere_netcdf(str(path), hemisphere, mode='third_octave', radius_ft=100.0)
        amplitude, phi, theta, _, _, _, _ = fa.load_nc_sphere(str(path))
        level = np.asarray(amplitude, dtype=float)[:, :, 0]

        def at(phi_deg, theta_deg):
            return level[np.argmin(np.abs(phi - phi_deg)), np.argmin(np.abs(theta - theta_deg))]

        if axis_test == 'phi':      # starboard vs port, in the lateral plane
            assert at(60.0, 90.0) == pytest.approx(80.0, abs=0.5)
            assert at(-60.0, 90.0) == pytest.approx(60.0, abs=0.5)
        else:                       # ahead vs behind, straight below the flight path
            assert at(0.0, 30.0) == pytest.approx(80.0, abs=0.5)
            assert at(0.0, 150.0) == pytest.approx(60.0, abs=0.5)
