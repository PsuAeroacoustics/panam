"""Pins how a lower-hemisphere sphere is completed onto its upper surface.

Sphere data covers only phi in [-90, 90] (phi = 0 straight down).  The upper
half is the reflection through the horizontal plane, phi -> 180 - phi.

Two earlier forms produced bad databases and are guarded against here:

* ``concatenate((phi, -phi))`` -- a left/right flip *within* the lower half.
  The source range is symmetric about zero, so it maps [-90, 90] onto itself,
  duplicating the lower hemisphere and leaving the upper surface empty.
* A three-part concatenation slicing phi as ``[h+1:n], [:], [1:h]`` but the data
  arrays as ``[1:h], [:], [h+1:n]``.  Those lengths agree only when the phi axis
  has an even number of entries; real grids have 19, so labels ran one row out
  of step with the acoustic data.
"""

import numpy as np

from flight_acoustics import _complete_sphere, mirror_phi_to_upper_surface


PHI_LIST = np.arange(-90.0, 90.1, 10.0)      # 19 azimuths, as in real spheres
THETA_LIST = np.arange(0.0, 180.1, 5.0)      # 37 polar angles


def test_reflection_values():
    mirror, _ = mirror_phi_to_upper_surface(np.array([0.0, 45.0, -80.0]))
    # 0 (down) -> 180 (up, stored as -180); 45 -> 135; -80 -> 260 == -100
    assert np.allclose(np.abs(mirror[0]), 180.0)
    assert np.isclose(mirror[1], 135.0)
    assert np.isclose(mirror[2], -100.0)


def test_shared_edges_are_not_duplicated():
    """+/-90 lie in the horizontal plane, shared by both halves, so they reflect
    onto themselves and must not be emitted twice."""
    mirror, source_index = mirror_phi_to_upper_surface(PHI_LIST)
    assert mirror.size == PHI_LIST.size - 2          # the two edges dropped
    assert not np.any(np.isclose(np.abs(mirror), 90.0))
    assert source_index.size == mirror.size


def test_completed_sphere_covers_the_full_circle_without_duplicates():
    mirror, _ = mirror_phi_to_upper_surface(PHI_LIST)
    full = np.concatenate((PHI_LIST, mirror))
    assert np.unique(full).size == 36                # full circle at 10 deg
    assert full.size == 36                           # and no repeats at all
    assert full.min() == -180.0 and full.max() == 170.0

    # The old broken form covered nothing new.
    assert np.unique(np.concatenate((PHI_LIST, -PHI_LIST))).size == 19


def test_data_stays_aligned_with_its_azimuth():
    """The regression the three-part slicing introduced: every row of data must
    still belong to the azimuth labeling it."""
    n_phi, n_theta = PHI_LIST.size, THETA_LIST.size
    # Tag each source row with its own azimuth so provenance is traceable.
    spla = np.repeat(PHI_LIST[:, None], n_theta, axis=1)
    eaa = spla.copy()
    amplitude = np.repeat(spla[:, :, None], 4, axis=2)

    phi, theta, spla_f, eaa_f, amp_f = _complete_sphere(
        PHI_LIST, THETA_LIST, spla, eaa, amplitude)

    assert phi.shape == spla_f.shape == eaa_f.shape == (36, n_theta)
    assert theta.shape == phi.shape
    assert amp_f.shape == (36, n_theta, 4)

    # Source rows carry their own azimuth; reflected rows carry the azimuth they
    # were reflected FROM, which must be the reflection of the output azimuth.
    for row in range(phi.shape[0]):
        out = phi[row, 0]
        src = spla_f[row, 0]
        reflected_back = (180.0 - src + 180.0) % 360.0 - 180.0
        assert np.isclose(out, src) or np.isclose(out, reflected_back), (
            f"row {row}: azimuth {out} carries data from {src}")


def test_theta_is_preserved_across_the_mirror():
    spla = np.zeros((PHI_LIST.size, THETA_LIST.size))
    amplitude = np.zeros((PHI_LIST.size, THETA_LIST.size, 2))
    _, theta, _, _, _ = _complete_sphere(PHI_LIST, THETA_LIST, spla, spla.copy(), amplitude)
    # Reflection is in azimuth only; every row spans the same theta axis.
    for row in range(theta.shape[0]):
        assert np.allclose(theta[row, :], THETA_LIST)


def test_completed_sphere_rows_are_sorted_by_phi():
    """NICE-OPS reads the gridded spectra as sorted by phi; they carry no angles
    of their own, so completion order (lower half, then the mirrored upper
    surface) put every spectrum in the wrong direction."""
    n_theta = THETA_LIST.size
    spla = np.repeat(PHI_LIST[:, None], n_theta, axis=1)
    amplitude = np.repeat(spla[:, :, None], 4, axis=2)
    phi, _, spla_f, _, amp_f = _complete_sphere(PHI_LIST, THETA_LIST, spla, spla.copy(), amplitude)
    assert np.all(np.diff(phi[:, 0]) > 0)
    # The spectra ride with the level rows through the sort.
    assert np.array_equal(amp_f[:, :, 0], spla_f)
