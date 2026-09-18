"""Pins the azimuth mirror used to synthesise the upper surface of a sphere.

The source spheres cover only the lower half of the roll circle,
phi in [-90, 90] degrees.  The upper half is produced by reflecting through
the horizontal plane (phi -> 180 - phi), NOT by negating phi.

Negation is a left/right flip within the lower half, and since the source
range is symmetric about zero it maps [-90, 90] onto itself.  Using it
produced a second copy of the lower hemisphere under the same (phi, theta)
labels and left the upper surface empty -- which is how the hover spheres in
S-76D_M3.nod came to carry two conflicting levels per point.
"""

import numpy as np

from flight_acoustics import mirror_phi_to_upper_surface


def test_straight_down_maps_to_straight_up():
    # phi = 0 is straight down; its reflection is straight up (180 == -180).
    assert np.isclose(abs(mirror_phi_to_upper_surface(np.array([0.0]))[0]), 180.0)


def test_horizontal_plane_is_fixed():
    # +/-90 lie in the horizontal plane, shared by both halves, so they map to
    # themselves -- which is why they appear twice in a completed sphere.
    got = mirror_phi_to_upper_surface(np.array([-90.0, 90.0]))
    assert np.allclose(got, [-90.0, 90.0])


def test_intermediate_angle():
    assert np.allclose(mirror_phi_to_upper_surface(np.array([45.0])), [135.0])


def test_output_stays_in_canonical_range():
    phi = np.arange(-90.0, 90.1, 10.0)
    got = mirror_phi_to_upper_surface(phi)
    assert np.all(got >= -180.0) and np.all(got < 180.0)


def test_mirror_completes_the_sphere():
    """The whole point: lower half + mirror must cover the full azimuth circle."""
    lower = np.arange(-90.0, 90.1, 10.0)          # 19 values, the source half
    full = np.concatenate((lower, mirror_phi_to_upper_surface(lower)))

    # 19 + 19 slots, with +/-90 shared between the halves -> 36 distinct azimuths
    assert np.unique(full).size == 36
    assert full.min() == -180.0 and full.max() == 170.0

    # The old, broken form collapses onto the source half and covers nothing new.
    broken = np.concatenate((lower, -lower))
    assert np.unique(broken).size == 19
