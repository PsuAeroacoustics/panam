import numpy as np

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
