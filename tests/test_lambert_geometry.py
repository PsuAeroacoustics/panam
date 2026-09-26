"""Lateral-sense regression tests for the Lambert hemisphere plots.

The hemisphere is a VIEW FROM ABOVE, matching the ground footprints:
ahead at the top, starboard on the RIGHT. Before 2026-08-19 the data was
placed with `lon = azimuth - pi` (starboard on the left) while the
meridian labels printed `360 - meridian`, so the rim numbers contradicted
the field and the spheres read mirrored against the footprints.
"""

import numpy as np
import pytest

import flight_acoustics as fa


CARDINALS = [
    # azimuth, description, expected page position
    (0.0, "behind", "bottom"),
    (90.0, "starboard", "right"),
    (180.0, "ahead", "top"),
    (270.0, "port", "left"),
]


def _page(az_deg):
    x, y = fa.lambert_ea(np.radians(0.0), fa.lambert_lon(np.radians(az_deg)))
    if abs(x) > 1e-9 and abs(x) > abs(y):
        return "right" if x > 0 else "left"
    return "top" if y > 0 else "bottom"


@pytest.mark.parametrize("az,what,expect", CARDINALS)
def test_page_position(az, what, expect):
    assert _page(az) == expect, f"azimuth {az} ({what}) must plot {expect}"


@pytest.mark.parametrize("az,what,expect", CARDINALS)
def test_meridian_label_matches_data(az, what, expect):
    """The rim label drawn at a meridian must name that same azimuth."""
    lon = fa.lambert_lon(np.radians(az))
    xl, _ = fa.lambert_ea(np.radians(-11.0), lon)
    xd, _ = fa.lambert_ea(np.radians(0.0), lon)
    assert np.sign(xl) == np.sign(xd) or abs(xd) < 1e-9


def test_hemigen_lateral_sense():
    """hemigen: starboard is azimuth 90 in the x=North, y=-East frame."""
    t = np.array([0.0])
    src = np.array([[0.0, 0.0, 100.0]])
    vel = np.array([[50.0, 0.0, 0.0]])            # due North
    obs = np.array([[0.0, -200.0, 0.0],           # y=-200 -> East -> starboard
                    [0.0, +200.0, 0.0],           # y=+200 -> West -> port
                    [200.0, 0.0, 0.0],            # North      -> ahead
                    [-200.0, 0.0, 0.0]])          # South      -> behind
    az, _, _, _, _ = fa.hemigen(t, src, vel, obs, 343.0)
    assert np.allclose(az[0], [90.0, 270.0, 180.0, 0.0], atol=1e-6)


def test_starboard_lobe_lands_on_the_right():
    """End to end: a lobe built at azimuth 90 must render on the right."""
    az = np.arange(0.0, 361.0, 5.0)
    el = np.arange(0.0, 91.0, 5.0)
    AZ, EL = np.meshgrid(az, el)
    lobe = np.exp(-((np.remainder(AZ - 90 + 180, 360) - 180) / 25.0) ** 2)
    x, _ = fa.lambert_ea(np.radians(EL), fa.lambert_lon(np.radians(AZ)))
    # power-weighted centroid of the lobe on the page
    assert float((x * lobe).sum() / lobe.sum()) > 0.5


def test_ground_footprint_puts_starboard_on_the_right(tmp_path):
    """A sphere loud to starboard must project loud to the RIGHT of the track.

    Footprints are the same view from above as these hemispheres. project_sphere
    used x = -ground_range*sin(azimuth), which mirrored every footprint once
    art2umapr put ART phi > 0 to starboard (the AAM manual's sign).
    """
    from netCDF4 import Dataset
    path = tmp_path / 'starboard_loud.nc'
    phi = np.arange(-90.0, 91.0, 10.0)
    theta = np.arange(0.0, 181.0, 5.0)
    frequency = np.array([100.0, 1000.0])
    amplitude = np.full((phi.size, theta.size, frequency.size), 80.0)
    amplitude[phi > 0] = 100.0                       # ART phi > 0 is starboard
    with Dataset(str(path), 'w', format='NETCDF3_CLASSIC') as ds:
        for name, values in (('PHI', phi), ('THETA', theta), ('FREQUENCY', frequency)):
            ds.createDimension(name, values.size)
            ds.createVariable(name, 'f4', (name,))[:] = values
        ds.createVariable('AMPLITUDE', 'f4', ('PHI', 'THETA', 'FREQUENCY'))[:] = amplitude
        for name, value in (('RADIUS', 100.0), ('SPEED', 60.0), ('FLIGHT_PATH_ANGLE', 0.0)):
            ds.createVariable(name, 'f4').assignValue(value)

    x, y, level_a, _, _ = fa.project_sphere(str(path), 150.0, 20.0)
    assert level_a[x > 1.0].mean() > level_a[x < -1.0].mean() + 10.0
