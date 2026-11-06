"""A database build opens each sphere once for its metadata (sphere_io.per_file_cache).

Every reader of a sphere's type, vehicle, labels, condition and flags opened the file again
(about 20 opens a sphere), which tells on cloud storage.  Inside a build they share one
handle per file; outside one, each reader opens and closes the file as before.
"""
import collections

import netCDF4

from panam import flight_acoustics as fa
from panam import sphere_io

import test_thrust_from_sphere as sweep


def test_a_build_opens_each_sphere_twice_at_most(tmp_path, monkeypatch):
    opens = collections.Counter()

    def Counting(filename, *args, **kwargs):   # noqa: N802 -- stands in for the class
        if str(filename).endswith('.nc'):
            opens[str(filename)] += 1
        return netCDF4.Dataset(filename, *args, **kwargs)

    import panam.database
    for module in (sphere_io, panam.database):
        monkeypatch.setattr(module, 'Dataset', Counting)
    runs = tuple((speed, weight) for speed in (0.0, 40.0, 80.0) for weight in (4000.0, 5500.0))
    folder = sweep._folder(tmp_path / 's', runs)
    fa.build_empirical_database(str(folder), str(tmp_path / 'x.nod'), load_factors=None, thrust_from='sphere')
    assert len(opens) == len(runs) and max(opens.values()) <= 2
    # The handles are closed when the build ends, and a reader outside one opens its own.
    assert sphere_io._FILE_CACHE is None
    before = sum(opens.values())
    sphere_io.read_sphere_type(str(folder / 'SWEEP000.nc'))
    sphere_io.read_sphere_type(str(folder / 'SWEEP000.nc'))
    assert sum(opens.values()) == before + 2
