"""Build a NICE-OPS sphere database from a directory of AAM sphere files.

Usage:  python build_empirical_database.py SPHERE_DIRECTORY DATABASE_FILE

SPHERE_DIRECTORY holds the .nc spheres and a vehicle.cfg (rotors,
atmosphere, weight and drag; see docs/file_formats.md).
"""
import argparse

import flight_acoustics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('sphere_directory', help='directory of .nc spheres (and vehicle.cfg)')
    parser.add_argument('database_file', help='output NICE-OPS database (.nc/.nod)')
    args = parser.parse_args()
    flight_acoustics.build_empirical_database(args.sphere_directory, args.database_file)
