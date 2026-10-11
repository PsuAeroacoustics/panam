"""Compare two NICE-OPS databases (.nod) and report what differs.

Usage:  python compare_databases.py A.nod B.nod [--tolerance DB] [--validate]

Conditions are matched by advance ratio, flight path angle and thrust
coefficient, not by group name.  The report lists the root variables and
attributes that differ, the conditions found in only one file, and for each
matched condition the per-band energy difference (B minus A), the largest
level differences and the coverage gained and lost.  The exit status is 0
when the databases agree within the tolerance, 1 when they differ, and 2 when
--validate finds a contract violation.  See nod_database.compare_databases.
"""
import argparse
import sys

import nod_database


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('database_a', help='first NICE-OPS database (.nod)')
    parser.add_argument('database_b', help='second NICE-OPS database (.nod)')
    parser.add_argument('--tolerance', type=float, default=0.0,
                        help='level differences up to this many dB (dB per 1000 m for EAA) are '
                             'not reported; a finite number, at least 0 (default %(default)g)')
    parser.add_argument('--validate', action='store_true',
                        help='also check both files against docs/file_formats.md and list the '
                             'violations (default: off)')
    args = parser.parse_args(argv)
    if not 0.0 <= args.tolerance < float('inf'):
        parser.error('--tolerance must be a finite number of dB, at least 0')
    return args


def main(argv=None):
    args = parse_arguments(argv)
    invalid = False
    if args.validate:
        for label, path in (('a', args.database_a), ('b', args.database_b)):
            for violation in nod_database.validate_nod(path):
                invalid = True
                print(f'{label} {path}: {violation}')
    comparison = nod_database.compare_databases(args.database_a, args.database_b)
    differs = comparison.differs(args.tolerance)
    if differs:
        print(comparison.summary(args.tolerance))
    else:
        print('no differences beyond {:g} dB in {} matched conditions'.format(
            args.tolerance, len(comparison.conditions)))
    return 2 if invalid else int(differs)


if __name__ == '__main__':
    sys.exit(main())
