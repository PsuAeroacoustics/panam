"""Read, check and compare NICEOPS databases (``.nod``).

The writers are :func:`flight_acoustics.build_empirical_database` and
:func:`flight_acoustics.build_database_from_norah2`; the file contract is
``docs/file_formats.md``.  This module only reads; ``tools/nod_tool.py`` and
``tools/compare_databases.py`` are its command lines:

- :func:`load_nod` returns the conditions of a database, with both the
  per-group and the ``shared_grid_and_frequency`` layout, and both a load
  factor ladder and a ``fixed_load_factor`` database.
- :func:`validate_nod` lists the ways a file breaks the contract, by the rule
  numbers of ``docs/file_formats.md#checking-a-database``.
- :func:`compare_databases` reports what differs between two databases.
- :func:`to_frame` gives a long pandas DataFrame, when pandas is installed.
- :func:`summarize_condition`, :func:`select_condition`, :func:`plot_condition`
  (a Lambert map of one quantity) and :func:`plot_envelope` (the conditions in
  speed and flight path angle) inspect a database.

Units: speed in knots, angles in degrees, radii in meters, frequency in Hz,
levels in dB, EAA in dB per 1000 m.  Angles are ART angles (phi 0 below,
positive to starboard; theta 0 at the nose).
"""
from collections import Counter, namedtuple
from dataclasses import dataclass, field

import numpy as np
from netCDF4 import Dataset

from panam import provenance
from panam.acoustics.plotting import acoustic_plot_style
from panam.spectra import power_to_db
from panam.geometry import art2umapr
from panam.sphere_io import AZIMUTH_REFERENCES, band_index, KNOT_MPS
from panam.database import DATABASE_FORMAT_VERSION, SPEED_REFERENCES
from panam.lambert_plots import plot_lambert_ea

#: The condition variables every group must carry (R2), each with one value.
CONDITION_VARIABLES = ('radius', 'rotor_scale', 'advance_ratio', 'flight_path_angle',
                       'thrust_coefficient')
#: The values the group attribute ``condition_origin`` takes (C12).
CONDITION_ORIGINS = ('measured', 'extended_flight_path_angle', 'synthesized_hover')


@dataclass
class NodCondition:
    """One condition group of a database.

    Attributes:
        name: the group name, e.g. ``sphere0``.
        speed_kt: advance_ratio × V_tip / 0.514444, knots; NaN when the root has
            no ``main_rotor_tip_speed_meters_per_sec``.
        advance_ratio: as stored, 1.
        flight_path_angle_deg: as stored, deg, negative in descent.
        thrust_coefficient: as stored, 1.
        load_factor: 1 in a ``fixed_load_factor`` database; otherwise
            thrust_coefficient over the ``weight_coefficient`` given to
            :func:`load_nod`, or NaN without one.
        radius_m: the sphere radius, ``radius`` × ``rotor_scale``, m.
        rotor_scale_m: the rotor radius, m.
        phi_deg, theta_deg: (n_channels,) channel directions, deg.
        dBA: (n_channels,) A-weighted level at the sphere radius, dB, with the
            load factor's 20 lg n; -inf where a direction has no energy.
        EAA: (n_channels,) excess atmospheric attenuation, dB per 1000 m.
        frequency_hz: (n_bands,) band centers, Hz, or None without spectra.
        amplitude: (n_phi, n_theta, n_bands) unweighted band levels at the sphere
            radius, dB, as stored (-inf where missing), or None without spectra.
        coverage: bool, True where measured; shaped like ``amplitude`` with
            spectra, (n_channels,) without, or None when the group has none.
        doppler_shift_removed: 0 or 1, or None when the group does not say.
        condition_origin: the group attribute, or None.
        attributes: every group attribute, by name.
    """
    name: str
    speed_kt: float
    advance_ratio: float
    flight_path_angle_deg: float
    thrust_coefficient: float
    load_factor: float
    radius_m: float
    rotor_scale_m: float
    phi_deg: np.ndarray
    theta_deg: np.ndarray
    dBA: np.ndarray
    EAA: np.ndarray
    frequency_hz: np.ndarray = None
    amplitude: np.ndarray = None
    coverage: np.ndarray = None
    doppler_shift_removed: int = None
    condition_origin: str = None
    attributes: dict = field(default_factory=dict)

    @property
    def measured_amplitude(self):
        """``amplitude``, dB, with every coverage-0 cell NaN (no data).

        The database stores a gated cell (measured, no energy) and an unmeasured
        one alike, as -inf with coverage 0 (``docs/file_formats.md``, how a bin's
        state travels), so the reader cannot give a gated cell back its -inf: both
        become NaN here, as do the filled cells (mirrored upper surface, hover
        partners, NORAH2 fills) that hold a level but no measurement.  Without a
        coverage variable, the -inf cells become NaN.  None without spectra.
        """
        if self.amplitude is None:
            return None
        measured = np.isfinite(self.amplitude) if self.coverage is None else self.coverage
        return np.where(measured, self.amplitude, np.nan)

    @property
    def key(self):
        """(advance_ratio, flight_path_angle_deg, thrust_coefficient), rounded to 9, 6
        and 9 decimals: what :func:`compare_databases` matches conditions by."""
        return (round(float(self.advance_ratio), 9), round(float(self.flight_path_angle_deg), 6),
                round(float(self.thrust_coefficient), 9))


@dataclass
class NodDatabase:
    """A database read by :func:`load_nod`: its conditions, in file order, and
    its root.  Indexing, iteration and ``len`` go to ``conditions``.

    Attributes:
        path: the file read.
        conditions: list of :class:`NodCondition`.
        root_variables: the root variables by name (scalars as Python numbers).
        root_attributes: the root attributes by name.
    """
    path: str
    conditions: list
    root_variables: dict
    root_attributes: dict

    def __iter__(self):
        return iter(self.conditions)

    def __len__(self):
        return len(self.conditions)

    def __getitem__(self, index):
        return self.conditions[index]

    @property
    def fixed_load_factor(self):
        return bool(self.root_variables.get('fixed_load_factor', False))

    @property
    def shared_grid_and_frequency(self):
        return bool(self.root_variables.get('shared_grid_and_frequency', False))


def _value(variable):
    """A variable's values as a numpy array (a Python scalar for a scalar)."""
    values = np.array(variable[...])
    return values.item() if values.ndim == 0 else values


def _scalar(group, name, default=np.nan):
    if name not in group.variables:
        return default
    return float(np.asarray(group.variables[name][...]).reshape(-1)[0])


def _group_or_shared(variables, root, name, shared):
    """A group's variable ``name`` as floats, else the root's when ``shared``.

    Args:
        variables: the group's netCDF variables.
        root: the root variables' values by name.
        name: the variable.
        shared: whether the root's copy stands in for a missing group variable
            (the database is ``shared_grid_and_frequency``).

    Returns:
        (values, where): where is 'group', 'root' or None (values None).
    """
    if name in variables:
        return np.asarray(variables[name][...], dtype=float), 'group'
    if shared and name in root:
        return np.asarray(root[name], dtype=float), 'root'
    return None, None


def _root(database):
    variables = {name: _value(variable) for name, variable in database.variables.items()}
    attributes = {name: database.getncattr(name) for name in database.ncattrs()}
    return variables, attributes


def _group_names(database):
    """Group names in sphere order (``sphere2`` before ``sphere10``), then any others."""
    def order(name):
        suffix = name[len('sphere'):]
        return (0, int(suffix), '') if name.startswith('sphere') and suffix.isdigit() else (1, 0, name)
    return sorted(database.groups, key=order)


def load_nod(path, weight_coefficient=None):
    """Read every condition group of a NICEOPS database.

    Args:
        path: the ``.nod`` file.
        weight_coefficient: C_W = W / (rho pi R^2 V_tip^2), 1, which the file
            does not store (it has no density).  Used for the load factor of a
            database that is not ``fixed_load_factor``: n = C_T / C_W.

    Returns:
        :class:`NodDatabase`, list-like over :class:`NodCondition` in group order.
        Values are as stored: ``amplitude`` is -inf where missing; use
        :attr:`NodCondition.measured_amplitude` for NaN at coverage 0.  A group
        that lacks a required variable reads it as NaN (see :func:`validate_nod`).
    """
    with Dataset(str(path)) as database:
        database.set_auto_mask(False)
        root_variables, root_attributes = _root(database)
        shared = bool(root_variables.get('shared_grid_and_frequency', False))
        fixed = bool(root_variables.get('fixed_load_factor', False))
        tip_speed = float(root_variables.get('main_rotor_tip_speed_meters_per_sec', np.nan))
        conditions = []
        for name in _group_names(database):
            group = database.groups[name]

            def array(variable, root_fallback=True):
                return _group_or_shared(group.variables, root_variables, variable,
                                        root_fallback and shared)[0]

            amplitude = array('amplitude', root_fallback=False)
            frequency = array('frequency') if amplitude is not None else None
            coverage = None
            if 'coverage' in group.variables:
                coverage = np.asarray(group.variables['coverage'][...]) == 1
            doppler = None
            if 'DOPPLER_SHIFT_REMOVED' in group.variables:
                doppler = int(np.asarray(group.variables['DOPPLER_SHIFT_REMOVED'][...]))
            attributes = {key: group.getncattr(key) for key in group.ncattrs()}
            advance_ratio = _scalar(group, 'advance_ratio')
            thrust_coefficient = _scalar(group, 'thrust_coefficient')
            if fixed:
                load_factor = 1.0
            elif weight_coefficient:
                load_factor = thrust_coefficient / float(weight_coefficient)
            else:
                load_factor = np.nan
            rotor_scale = _scalar(group, 'rotor_scale')
            empty = np.array([])
            phi, theta = array('phi'), array('theta')
            dba, eaa = array('dBA', False), array('EAA', False)
            conditions.append(NodCondition(
                name=name,
                speed_kt=advance_ratio * tip_speed / KNOT_MPS,
                advance_ratio=advance_ratio,
                flight_path_angle_deg=_scalar(group, 'flight_path_angle'),
                thrust_coefficient=thrust_coefficient,
                load_factor=load_factor,
                radius_m=_scalar(group, 'radius') * rotor_scale,
                rotor_scale_m=rotor_scale,
                phi_deg=empty if phi is None else phi,
                theta_deg=empty if theta is None else theta,
                dBA=empty if dba is None else dba,
                EAA=empty if eaa is None else eaa,
                frequency_hz=frequency,
                amplitude=amplitude,
                coverage=coverage,
                doppler_shift_removed=doppler,
                condition_origin=attributes.get('condition_origin'),
                attributes=attributes))
    return NodDatabase(str(path), conditions, root_variables, root_attributes)


class Violation(namedtuple('Violation', ['rule', 'group', 'message'])):
    """One break of the contract: its rule number (``R1``..., ``C1``...), the group
    it is in (None for the root or the file as a whole) and what is wrong."""
    __slots__ = ()

    def __str__(self):
        where = self.group if self.group is not None else 'root'
        return f'{self.rule} [{where}]: {self.message}'


def validate_nod(path):
    """Check a NICEOPS database against ``docs/file_formats.md``.

    The R rules are what NICEOPS requires; the C rules are the rest of the
    contract the writers keep (both listed in
    ``docs/file_formats.md#checking-a-database``).

    Args:
        path: the ``.nod`` file.

    Returns:
        list of :class:`Violation`, empty for a database that keeps the contract.
    """
    violations = []

    def violate(rule, group, message):
        violations.append(Violation(rule, group, message))

    with Dataset(str(path)) as database:
        database.set_auto_mask(False)
        root, attributes = _root(database)
        names = _group_names(database)

        # Root flags and attributes.
        if 'same_grid' not in root:
            violate('R1', None, 'no same_grid variable')
        elif not root['same_grid']:
            violate('R1', None, 'same_grid is false')
        version = root.get('database_version')
        if version is not None and (not float(version).is_integer()
                                    or not 0 <= version <= DATABASE_FORMAT_VERSION):
            violate('C1', None, f'database_version {version} is not an integer from 0 to '
                                f'{DATABASE_FORMAT_VERSION}')
        for attribute, allowed in (('speed_reference', SPEED_REFERENCES),
                                   ('azimuth_reference', AZIMUTH_REFERENCES)):
            if attribute in attributes and attributes[attribute] not in allowed:
                violate('C2', None, f'{attribute} {attributes[attribute]!r} is not one of {allowed}')
        shared = bool(root.get('shared_grid_and_frequency', False))
        if shared:
            for name in ('phi', 'theta'):
                if name not in root:
                    violate('R4', None, f'shared_grid_and_frequency is set but the root has no {name}')
        if not names:
            violate('R2', None, 'no condition groups')
        expected_names = ['sphere{}'.format(i) for i in range(len(names))]
        if names != expected_names:
            violate('C11', None, 'groups are not named sphere0 to sphere{}: {}'.format(
                len(names) - 1, ', '.join(sorted(set(names) - set(expected_names)))))

        sphere_radii = {}
        thrust_coefficients = {}
        doppler_flags = {}
        for name in names:
            group = database.groups[name]
            variables = group.variables
            n_channels = len(group.dimensions['channels']) if 'channels' in group.dimensions else None

            # R2: one finite value of each condition variable.
            for variable in CONDITION_VARIABLES:
                if variable not in variables:
                    violate('R2', name, f'no {variable}')
                    continue
                values = np.asarray(variables[variable][...], dtype=float).reshape(-1)
                if values.size != 1:
                    violate('R2', name, f'{variable} has {values.size} values, not 1')
                elif not np.isfinite(values[0]):
                    violate('R2', name, f'{variable} is {values[0]}')
            if 'radius' in variables and 'rotor_scale' in variables:
                sphere_radii[name] = _scalar(group, 'radius') * _scalar(group, 'rotor_scale')
            if 'thrust_coefficient' in variables:
                thrust_coefficients[name] = _scalar(group, 'thrust_coefficient')

            # R3, R6: dBA and EAA over the channels.
            levels = {}
            for variable in ('dBA', 'EAA'):
                if variable not in variables:
                    violate('R3', name, f'no {variable}')
                    continue
                if variables[variable].dimensions != ('channels',):
                    violate('R3', name, f'{variable} is over {variables[variable].dimensions}, '
                                        "not ('channels',)")
                    continue
                levels[variable] = np.asarray(variables[variable][...], dtype=float)
            if 'dBA' in levels:
                bad = np.isnan(levels['dBA']) | np.isposinf(levels['dBA'])
                if bad.any():
                    violate('R6', name, f'dBA is NaN or +inf in {bad.sum()} channels')
            if 'EAA' in levels:
                bad = ~np.isfinite(levels['EAA'])
                if bad.any():
                    violate('R6', name, f'EAA is not finite in {bad.sum()} channels')

            # R4, C3: the channel directions, here or at the root.
            directions = {}
            for variable in ('phi', 'theta'):
                values, where = _group_or_shared(variables, root, variable, shared)
                if where is None:
                    violate('R4', name, f'no {variable}, in the group or shared at the root')
                    continue
                if where == 'group' and shared and variable in root \
                        and not np.array_equal(values, root[variable]):
                    violate('C3', name, f'{variable} differs from the shared root {variable}')
                if n_channels is not None and values.size != n_channels:
                    violate('R4', name, f'{variable} has {values.size} values for {n_channels} channels')
                    continue
                directions[variable] = values
            # C13: ART angles, deg.
            for variable, outside in (('phi', lambda a: ~((a >= -180.0) & (a < 180.0))),
                                      ('theta', lambda a: ~((a >= 0.0) & (a <= 180.0)))):
                if variable in directions and np.any(outside(directions[variable])):
                    interval = '[-180, 180)' if variable == 'phi' else '[0, 180]'
                    violate('C13', name, f'{variable} is outside {interval} deg (ART) in '
                                         f'{np.sum(outside(directions[variable]))} channels')

            # C4, C5: the spectrum's grid and values.
            amplitude = None
            if 'amplitude' in variables:
                amplitude_variable = variables['amplitude']
                amplitude = np.asarray(amplitude_variable[...], dtype=float)
                if amplitude_variable.dimensions != ('PHI', 'THETA', 'frequency'):
                    violate('C4', name, f'amplitude is over {amplitude_variable.dimensions}, '
                                        "not ('PHI', 'THETA', 'frequency')")
                frequency, where = _group_or_shared(variables, root, 'frequency', shared)
                if where is None:
                    violate('C3', name, 'amplitude without frequency, in the group or shared at the root')
                elif where == 'group' and shared and 'frequency' in root \
                        and not np.array_equal(frequency, root['frequency']):
                    violate('C3', name, 'frequency differs from the shared root frequency')
                if amplitude.ndim == 3:
                    n_phi, n_theta, n_bands = amplitude.shape
                    if frequency is not None and frequency.size != n_bands:
                        violate('C4', name, f'{frequency.size} frequencies for {n_bands} amplitude bands')
                    if n_channels is not None and n_phi * n_theta != n_channels:
                        violate('C4', name, f'amplitude grid {n_phi} x {n_theta} for {n_channels} channels')
                    elif len(directions) == 2:
                        phi = directions['phi'].reshape(n_phi, n_theta)
                        theta = directions['theta'].reshape(n_phi, n_theta)
                        if (np.any(phi != phi[:, :1]) or np.any(np.diff(phi[:, 0]) <= 0)
                                or np.any(theta != theta[:1, :])):
                            violate('C4', name, 'the channels are not the row-major flatten of a '
                                                'phi-sorted (PHI, THETA) grid')
                    bad = np.isnan(amplitude) | np.isposinf(amplitude)
                    if bad.any():
                        violate('C5', name, f'amplitude is NaN or +inf in {bad.sum()} cells')
                    # C7: a direction with no band left has no energy.
                    if 'dBA' in levels and n_channels == n_phi * n_theta:
                        empty = ~np.isfinite(amplitude).any(axis=2).reshape(-1)
                        if np.any(np.isfinite(levels['dBA'][empty])):
                            violate('C7', name, 'a direction with no finite band has a finite dBA')
                        if 'EAA' in levels and np.any(levels['EAA'][empty] != 0):
                            violate('C7', name, 'a direction with no finite band has a nonzero EAA')

            # C6: coverage marks measured cells only.
            if 'coverage' in variables:
                coverage_variable = variables['coverage']
                coverage = np.asarray(coverage_variable[...])
                expected = ('PHI', 'THETA', 'frequency') if amplitude is not None else ('channels',)
                if coverage_variable.dimensions != expected:
                    violate('C6', name, f'coverage is over {coverage_variable.dimensions}, not {expected}')
                elif not np.isin(coverage, (0, 1)).all():
                    violate('C6', name, 'coverage holds values other than 0 and 1')
                else:
                    levels_covered = amplitude if amplitude is not None else levels.get('dBA')
                    if levels_covered is not None and levels_covered.shape == coverage.shape:
                        unbacked = (coverage == 1) & ~np.isfinite(levels_covered)
                        if unbacked.any():
                            violate('C6', name, f'coverage is 1 in {unbacked.sum()} cells without a '
                                                'finite level')

            if 'DOPPLER_SHIFT_REMOVED' in variables:
                doppler_flags[name] = np.asarray(variables['DOPPLER_SHIFT_REMOVED'][...]).item()
            if 'main_rotor_radius_meters' in root and 'rotor_scale' in variables:
                if not np.isclose(_scalar(group, 'rotor_scale'), root['main_rotor_radius_meters'],
                                  rtol=1e-12, atol=0.0):
                    violate('C10', name, 'rotor_scale {} differs from main_rotor_radius_meters {}'.format(
                        _scalar(group, 'rotor_scale'), root['main_rotor_radius_meters']))
            if 'condition_origin' in group.ncattrs():
                origin = group.getncattr('condition_origin')
                if origin not in CONDITION_ORIGINS:
                    violate('C12', name, f'condition_origin {origin!r} is not one of {CONDITION_ORIGINS}')

        # Rules across groups.
        if sphere_radii:
            first = next(iter(sphere_radii.values()))
            for name, value in sphere_radii.items():
                if not np.isclose(value, first, rtol=1e-9, atol=0.0):
                    violate('R5', name, f'radius x rotor_scale is {value} m, not {first} m as in {names[0]}')
        if root.get('fixed_load_factor') and len(set(thrust_coefficients.values())) > 1:
            violate('C8', None, 'fixed_load_factor is set but thrust_coefficient differs between groups')
        if doppler_flags:
            if not set(doppler_flags.values()) <= {0, 1}:
                violate('C9', None, 'DOPPLER_SHIFT_REMOVED takes values other than 0 and 1')
            if len(set(doppler_flags.values())) > 1 or len(doppler_flags) != len(names):
                violate('C9', None, 'DOPPLER_SHIFT_REMOVED is not the same in every group')
    return violations


@dataclass
class ConditionDifference:
    """How one condition matched by :func:`compare_databases` differs.

    Attributes:
        key: the condition's :attr:`NodCondition.key`.
        group_a, group_b: its group names in the two databases.
        grid_differs: the channel directions, the spectrum grid or the bands
            differ, so the levels are not compared cell by cell.
        frequency_hz: (n_bands,) band centers, Hz, or None without spectra in both.
        band_difference_db: (n_bands,) b minus a, dB, of each band's energy summed
            over the cells finite in both; NaN for a band with no such cell.
        max_abs_band_difference_db: largest |b - a| over the cells finite in both, dB.
        max_abs_dba_difference_db, max_abs_eaa_difference_db: the same over the
            channels, dB and dB per 1000 m.
        finite_only_in_a, finite_only_in_b: cells (bands with spectra, else
            channels' dBA) that hold energy in one database and -inf in the other.
        coverage_gained, coverage_lost: cells with coverage 0 in a and 1 in b, and
            the reverse; None unless both carry coverage of the same shape.
        scalar_differences: {name: (value in a, value in b)} for each of
            :data:`CONDITION_SCALARS` that differs.
    """
    key: tuple
    group_a: str
    group_b: str
    grid_differs: bool = False
    frequency_hz: np.ndarray = None
    band_difference_db: np.ndarray = None
    max_abs_band_difference_db: float = np.nan
    max_abs_dba_difference_db: float = np.nan
    max_abs_eaa_difference_db: float = np.nan
    finite_only_in_a: int = 0
    finite_only_in_b: int = 0
    coverage_gained: int = None
    coverage_lost: int = None
    scalar_differences: dict = field(default_factory=dict)

    def differs(self, tolerance_db=0.0):
        """Whether anything here differs by more than tolerance_db (dB)."""
        return bool(self.scalar_differences) or self.levels_differ(tolerance_db)

    def levels_differ(self, tolerance_db=0.0):
        """Whether the grid, levels (by more than tolerance_db, dB), energy cells
        or coverage differ; :attr:`scalar_differences` aside."""
        largest = np.nanmax([self.max_abs_band_difference_db, self.max_abs_dba_difference_db,
                             self.max_abs_eaa_difference_db, -np.inf])
        return bool(self.grid_differs or largest > tolerance_db or self.finite_only_in_a
                    or self.finite_only_in_b or self.coverage_gained or self.coverage_lost)


@dataclass
class DatabaseComparison:
    """What :func:`compare_databases` found.

    Attributes:
        root_differences: {name: (value in a, value in b)} for each root variable
            or attribute that differs; None for one that is absent.
        only_in_a, only_in_b: (group name, key) of the conditions with no match.
        conditions: list of :class:`ConditionDifference`, one per matched pair.
    """
    root_differences: dict
    only_in_a: list
    only_in_b: list
    conditions: list

    def differs(self, tolerance_db=0.0):
        """Whether the databases differ, levels by more than tolerance_db (dB)."""
        return bool(self.root_differences or self.only_in_a or self.only_in_b
                    or any(c.differs(tolerance_db) for c in self.conditions))

    def summary(self, tolerance_db=0.0):
        """A plain-text report, one line per difference; levels within
        tolerance_db (dB) are not listed."""
        lines = []
        for name, (a, b) in sorted(self.root_differences.items()):
            lines.append(f'root {name}: {_short(a)} -> {_short(b)}')
        for label, unmatched in (('only in a', self.only_in_a), ('only in b', self.only_in_b)):
            for name, key in unmatched:
                lines.append('{} {}: advance ratio {:g}, flight path angle {:g} deg, C_T {:g}'.format(
                    label, name, *key))
        for c in self.conditions:
            if not c.differs(tolerance_db):
                continue
            head = f'{c.group_a} / {c.group_b}:'
            for name, (va, vb) in sorted(c.scalar_differences.items()):
                lines.append(f'{head} {name} {_short(va)} -> {_short(vb)}')
            if not c.levels_differ(tolerance_db):
                continue
            if c.grid_differs:
                lines.append(f'{head} grid or bands differ; levels not compared')
                continue
            parts = []
            if c.band_difference_db is not None and np.isfinite(c.max_abs_band_difference_db):
                worst = int(np.nanargmax(np.abs(np.where(np.isnan(c.band_difference_db), 0.0,
                                                          c.band_difference_db))))
                parts.append('bands up to {:.6g} dB (band energy {:+.6g} dB at {:g} Hz)'.format(
                    c.max_abs_band_difference_db, c.band_difference_db[worst], c.frequency_hz[worst]))
            parts.append('dBA up to {:.6g} dB, EAA up to {:.6g} dB/km'.format(
                c.max_abs_dba_difference_db, c.max_abs_eaa_difference_db))
            if c.finite_only_in_a or c.finite_only_in_b:
                parts.append(f'energy only in a {c.finite_only_in_a}, only in b {c.finite_only_in_b} cells')
            if c.coverage_gained or c.coverage_lost:
                parts.append(f'coverage gained {c.coverage_gained}, lost {c.coverage_lost} cells')
            lines.append(head + ' ' + '; '.join(parts))
        return '\n'.join(lines)


def _short(value):
    if isinstance(value, np.ndarray):
        return f'array of {value.size}'
    return repr(value)


def _differ(a, b):
    if a is None or b is None:
        return a is not b
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape != b.shape or not np.array_equal(a, b, equal_nan=a.dtype.kind == 'f')
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return False
    return a != b


def _energy_db(levels, axis=None):
    return power_to_db(np.sum(10.0 ** (levels / 10.0), axis=axis))


def _max_abs_difference(a, b):
    both = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(b[both] - a[both]))) if both.any() else np.nan


#: NodCondition fields that change how NICEOPS reads a group's levels;
#: :func:`compare_databases` reports each one that differs.
CONDITION_SCALARS = ('radius_m', 'rotor_scale_m', 'doppler_shift_removed', 'condition_origin')


def _compare_condition(a, b):
    result = ConditionDifference(a.key, a.name, b.name)
    result.scalar_differences = {name: (getattr(a, name), getattr(b, name))
                                 for name in CONDITION_SCALARS
                                 if _differ(getattr(a, name), getattr(b, name))}
    if (a.phi_deg.shape != b.phi_deg.shape or not np.array_equal(a.phi_deg, b.phi_deg)
            or not np.array_equal(a.theta_deg, b.theta_deg)
            or (a.amplitude is None) != (b.amplitude is None)):
        result.grid_differs = True
        return result
    if a.amplitude is not None and (a.amplitude.shape != b.amplitude.shape
                                    or not np.array_equal(a.frequency_hz, b.frequency_hz)):
        result.grid_differs = True
        return result
    result.max_abs_dba_difference_db = _max_abs_difference(a.dBA, b.dBA)
    result.max_abs_eaa_difference_db = _max_abs_difference(a.EAA, b.EAA)
    if a.amplitude is not None:
        la, lb = a.amplitude, b.amplitude
        both = np.isfinite(la) & np.isfinite(lb)
        n_bands = la.shape[2]
        flat_a, flat_b = la.reshape(-1, n_bands), lb.reshape(-1, n_bands)
        flat_both = both.reshape(-1, n_bands)
        energy_a = _energy_db(np.where(flat_both, flat_a, -np.inf), axis=0)
        energy_b = _energy_db(np.where(flat_both, flat_b, -np.inf), axis=0)
        with np.errstate(invalid='ignore'):
            band = energy_b - energy_a
        band[~flat_both.any(axis=0)] = np.nan
        result.frequency_hz = a.frequency_hz
        result.band_difference_db = band
        result.max_abs_band_difference_db = _max_abs_difference(la, lb)
        cells_a, cells_b = np.isfinite(la), np.isfinite(lb)
    else:
        cells_a, cells_b = np.isfinite(a.dBA), np.isfinite(b.dBA)
    result.finite_only_in_a = int(np.sum(cells_a & ~cells_b))
    result.finite_only_in_b = int(np.sum(cells_b & ~cells_a))
    if a.coverage is not None and b.coverage is not None and a.coverage.shape == b.coverage.shape:
        result.coverage_gained = int(np.sum(~a.coverage & b.coverage))
        result.coverage_lost = int(np.sum(a.coverage & ~b.coverage))
    return result


#: Root attributes of the provenance stamp that name a file and its build time
#: rather than its content; :func:`compare_databases` does not compare them.
IDENTITY_ATTRIBUTES = provenance.IDENTITY_ATTRIBUTES


def compare_databases(a, b):
    """Compare two NICEOPS databases.

    Conditions are matched by :attr:`NodCondition.key` (advance ratio, flight
    path angle, thrust coefficient), not by group name, so a reordered database
    compares equal.  Conditions with the same key are paired in file order.

    Args:
        a, b: ``.nod`` paths, or :class:`NodDatabase` from :func:`load_nod`.

    Returns:
        :class:`DatabaseComparison`: the root variables and attributes that differ
        (apart from :data:`IDENTITY_ATTRIBUTES`, which differ between any two builds),
        the unmatched conditions of each, and per matched condition the
        :data:`CONDITION_SCALARS` that differ (sphere radius, rotor scale, Doppler
        convention, origin), the per-band energy difference (dB), the largest level
        differences and the coverage gained and lost.
    """
    a = a if isinstance(a, NodDatabase) else load_nod(a)
    b = b if isinstance(b, NodDatabase) else load_nod(b)
    root_a = {**a.root_variables, **{'@' + k: v for k, v in a.root_attributes.items()}}
    root_b = {**b.root_variables, **{'@' + k: v for k, v in b.root_attributes.items()}}
    for name in IDENTITY_ATTRIBUTES:
        root_a.pop('@' + name, None)
        root_b.pop('@' + name, None)
    root_differences = {name.lstrip('@'): (root_a.get(name), root_b.get(name))
                        for name in sorted(set(root_a) | set(root_b))
                        if _differ(root_a.get(name), root_b.get(name))}
    unmatched_b = {}
    for condition in b:
        unmatched_b.setdefault(condition.key, []).append(condition)
    only_in_a, pairs = [], []
    for condition in a:
        partners = unmatched_b.get(condition.key)
        if partners:
            pairs.append((condition, partners.pop(0)))
        else:
            only_in_a.append((condition.name, condition.key))
    remaining = {id(c) for partners in unmatched_b.values() for c in partners}
    only_in_b = [(c.name, c.key) for c in b if id(c) in remaining]
    return DatabaseComparison(root_differences, only_in_a, only_in_b,
                              [_compare_condition(x, y) for x, y in pairs])


def to_frame(database):
    """A long pandas DataFrame of a database: one row per condition, channel and
    band (per condition and channel for a group without spectra).

    Columns: condition, speed_kt, advance_ratio, flight_path_angle_deg,
    thrust_coefficient, load_factor, condition_origin, phi_deg, theta_deg, dBA (dB),
    EAA (dB per 1000 m), coverage (bool; NaN without one) and, with spectra,
    frequency_hz and amplitude_db (as stored, -inf where missing).  The channel
    of a band is its row-major position in the (PHI, THETA) grid.

    Args:
        database: a ``.nod`` path or a :class:`NodDatabase`.

    Raises:
        ImportError: pandas is not installed (it is optional for PANAM).
    """
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError('to_frame needs pandas, which PANAM does not install; '
                          'pip install pandas') from error
    database = database if isinstance(database, NodDatabase) else load_nod(database)
    frames = []
    for c in database:
        columns = dict(phi_deg=c.phi_deg, theta_deg=c.theta_deg, dBA=c.dBA, EAA=c.EAA)
        if c.amplitude is not None:
            n_bands = c.amplitude.shape[2]
            columns = {k: np.repeat(v, n_bands) for k, v in columns.items()}
            columns['frequency_hz'] = np.tile(c.frequency_hz, c.phi_deg.size)
            columns['amplitude_db'] = c.amplitude.reshape(-1)
        columns['coverage'] = (np.full(columns['dBA'].size, np.nan) if c.coverage is None
                               else c.coverage.reshape(-1))
        frame = pd.DataFrame(columns)
        for name in ('load_factor', 'thrust_coefficient', 'flight_path_angle_deg', 'advance_ratio',
                     'speed_kt'):
            frame.insert(0, name, getattr(c, name))
        frame.insert(0, 'condition', c.name)
        frame['condition_origin'] = c.condition_origin
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_condition(condition):
    """One row of ``nod_tool.py list``: what a condition is and how full it is.

    Args:
        condition: a :class:`NodCondition`.

    Returns:
        dict with group, speed_kt (kt), flight_path_angle_deg (deg), load_factor
        (1; NaN when unknown), condition_origin (None when the group does not say),
        coverage_fraction (the fraction of coverage cells set in the lower
        hemisphere, |phi| <= 90 deg, the part a measurement can cover, 1; NaN
        without coverage) and dBA_max (the largest channel dBA, dB; -inf when no
        direction has energy, NaN without channels).
    """
    if condition.coverage is None or condition.coverage.size == 0:
        coverage = np.nan
    else:
        cells = condition.coverage if condition.coverage.ndim == 3 \
            else _channel_grid(condition, condition.coverage)
        coverage = float(np.mean(cells[_lower_rows(condition)]))
    return dict(group=condition.name, speed_kt=condition.speed_kt,
                flight_path_angle_deg=condition.flight_path_angle_deg,
                load_factor=condition.load_factor, condition_origin=condition.condition_origin,
                coverage_fraction=coverage,
                dBA_max=float(np.max(condition.dBA)) if condition.dBA.size else np.nan)


def select_condition(database, speed_kt, flight_path_angle_deg, load_factor=None,
                     speed_tolerance_kt=0.5, angle_tolerance_deg=0.05):
    """The condition of a database at a speed and flight path angle.

    Args:
        database: a :class:`NodDatabase`.
        speed_kt: speed, kt.
        flight_path_angle_deg: flight path angle, deg, negative in descent.
        load_factor: the load factor, 1, the condition must have (it picks one
            of a load factor ladder at that speed and angle); default None,
            which needs the match to be unique.
        speed_tolerance_kt, angle_tolerance_deg: how far a condition may be from
            the speed (kt, default 0.5) and the angle (deg, default 0.05).

    Returns:
        the matching :class:`NodCondition`.

    Raises:
        ValueError: no condition matches (including one at the speed and angle
            with another load factor), or several do and ``load_factor`` does
            not pick one (e.g. the load factors are unknown: pass the
            weight coefficient to :func:`load_nod`).
    """
    matches = [c for c in database
               if abs(c.speed_kt - speed_kt) <= speed_tolerance_kt
               and abs(c.flight_path_angle_deg - flight_path_angle_deg) <= angle_tolerance_deg]
    unknown_load_factor = False
    if load_factor is not None:
        unknown_load_factor = bool(matches) and all(np.isnan(c.load_factor) for c in matches)
        matches = [c for c in matches if np.isclose(c.load_factor, load_factor, rtol=0.0, atol=1e-3)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        message = f'no condition at {speed_kt:g} kt and {flight_path_angle_deg:g} deg'
        if load_factor is not None:
            message += f', load factor {load_factor:g}'
            if unknown_load_factor:
                message += (' (the load factors of this database are unknown: give the weight '
                            'coefficient, --weight-coefficient)')
        raise ValueError(message)
    raise ValueError('{} conditions at {:g} kt and {:g} deg ({}); give the load factor or the group'.format(
        len(matches), speed_kt, flight_path_angle_deg, ', '.join(c.name for c in matches)))


#: The quantities :func:`plot_condition` maps.
PLOT_QUANTITIES = ('dBA', 'EAA', 'band', 'coverage')


def _channel_grid(condition, values):
    """``values`` over the channels as the (n_phi, n_theta) grid they flatten (C4)."""
    n_phi = condition.amplitude.shape[0] if condition.amplitude is not None \
        else np.unique(condition.phi_deg).size
    return np.asarray(values, dtype=float).reshape(n_phi, -1)


def _lower_rows(condition):
    """The phi rows of the channel grid in the lower hemisphere, |phi| <= 90 deg.

    A built database covers the whole sphere; the rows with |phi| > 90 deg are
    the mirrored upper surface, which no measurement covers (coverage 0).
    """
    phi = _channel_grid(condition, condition.phi_deg)[:, 0]
    return np.abs(phi) <= 90.0 + 1e-9


@acoustic_plot_style
def plot_condition(condition, quantity='dBA', band_hz=None, level_range=None, grid_convention='art'):
    """Map one quantity of a condition on a Lambert equal-area projection.

    Args:
        condition: a :class:`NodCondition`.
        quantity: one of :data:`PLOT_QUANTITIES`: 'dBA' (A-weighted level at the
            sphere radius, dB), 'EAA' (dB per 1000 m), 'band' (the stored band
            level at the sphere radius, dB; needs ``band_hz``) or 'coverage' (1
            where measured; with spectra, the band ``band_hz``, or without it the
            fraction of bands measured in each direction).
        band_hz: band center, Hz, for 'band' and 'coverage'; matched to 0.1 %.
        level_range: (low, high) color range in the quantity's unit; default
            the finite data's range (0 to 1 for coverage).
        grid_convention: grid lines, as :func:`flight_acoustics.plot_lambert_ea`
            (default 'art').

    Returns:
        matplotlib Figure of the lower hemisphere, |phi| <= 90 deg (the rows
        with |phi| > 90 deg, the mirrored upper surface, are left out, also from
        the default color range).  -inf levels (no energy) and missing cells
        are blank.

    Raises:
        ValueError: an unknown quantity, a band the condition does not have,
            or a quantity the group does not store.
    """
    if quantity not in PLOT_QUANTITIES:
        raise ValueError(f'quantity {quantity!r} is not one of {PLOT_QUANTITIES}')
    band = None
    if band_hz is not None or quantity == 'band':
        if condition.amplitude is None:
            raise ValueError(f'{condition.name} has no band levels')
        if band_hz is None:
            raise ValueError('quantity band needs a band center')
        band = band_index(condition.frequency_hz, band_hz)
        if band is None:
            raise ValueError(f'{condition.name} has no {band_hz:g} Hz band')
    levels = None
    cmap = None
    radius = f'{condition.radius_m:g} m'
    if quantity == 'dBA':
        field = _channel_grid(condition, condition.dBA)
        label = f'A-weighted level at {radius}, dBA'
    elif quantity == 'EAA':
        field = _channel_grid(condition, condition.EAA)
        label = 'Excess atmospheric attenuation, dB/1000 m'
    elif quantity == 'band':
        field = condition.amplitude[:, :, band]
        label = f'{condition.frequency_hz[band]:g} Hz band level at {radius}, dB'
    else:
        if condition.coverage is None:
            raise ValueError(f'{condition.name} has no coverage')
        if condition.coverage.ndim == 3:
            coverage = condition.coverage
            field = coverage[:, :, band] if band is not None else np.mean(coverage, axis=2)
            label = ('Measured' if band is not None else 'Fraction of bands measured')
            if band is not None:
                label += f', {condition.frequency_hz[band]:g} Hz'
        else:
            field = _channel_grid(condition, condition.coverage)
            label = 'Measured'
        field = np.asarray(field, dtype=float)
        level_range = (0.0, 1.0) if level_range is None else level_range
        levels = np.linspace(level_range[0], level_range[1], 11)
        cmap = 'viridis'
    rows = _lower_rows(condition)
    field = np.asarray(field, dtype=float)[rows]
    if not np.any(np.isfinite(field)):
        raise ValueError(f'{condition.name} has no finite {quantity} to map in the lower hemisphere')
    phi = _channel_grid(condition, condition.phi_deg)[rows]
    theta = _channel_grid(condition, condition.theta_deg)[rows]
    azi, elv = art2umapr(np.radians(phi), np.radians(theta))
    fig, _, _ = plot_lambert_ea(azi, elv, field, level_range, grid_convention=grid_convention,
                                   levels=levels, cmap=cmap, colorbar_label=label)
    return fig


@acoustic_plot_style
def plot_envelope(database):
    """Scatter a database's conditions in speed and flight path angle, colored
    by ``condition_origin``.

    Args:
        database: a :class:`NodDatabase`.

    Returns:
        matplotlib Figure: speed (kt) against flight path angle (deg), or
        advance ratio against flight path angle when the root has no tip speed.
        Conditions of a load factor ladder share a point.
    """
    import matplotlib.pyplot as plt
    speeds = np.array([c.speed_kt for c in database], dtype=float)
    by_speed = bool(speeds.size) and bool(np.all(np.isfinite(speeds)))
    x = speeds if by_speed else np.array([c.advance_ratio for c in database], dtype=float)
    y = np.array([c.flight_path_angle_deg for c in database], dtype=float)
    origins = [c.condition_origin or 'not given' for c in database]
    fig, ax = plt.subplots(facecolor='white')
    known = [o for o in CONDITION_ORIGINS if o in origins]
    for k, origin in enumerate(known + sorted(set(origins) - set(known))):
        chosen = np.array([o == origin for o in origins])
        ax.scatter(x[chosen], y[chosen], color=f'C{k}', label=origin.replace('_', ' '), zorder=3)
    ax.set_xlabel('Speed, kt' if by_speed else 'Advance ratio')
    ax.set_ylabel('Flight path angle, deg')
    if origins:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    return fig
