#!/usr/bin/env python3
"""Microphone array design + coverage visualization CLI.

This tool is meant to support quick iteration on linear microphone array designs
and visualize how well a design covers a *depropagated acoustic hemisphere*
(e.g., an `.npz` product saved by `AS350_289108_demo.py`).

It wires together three core utilities in `flight_acoustics`:
- `linear_array_plan`: compute sideline microphone positions for a target elevation spacing
- `hemigen`: the (azimuth, elevation) each microphone samples along an overflight trajectory
- `lambert_ea_points`: the Lambert equal-area axes the coverage points are drawn on
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np

import flight_acoustics as fa
from cli import colon_pair, save_or_show


def _set_window_title(fig, title: Optional[str]) -> None:
	"""Set the GUI window title (no-op for non-GUI backends).

	We intentionally avoid putting titles inside the Matplotlib axes/figure,
	since they don't render nicely for these Lambert plots.
	"""
	if not title:
		return
	try:
		manager = getattr(fig.canvas, 'manager', None)
		if manager is not None and hasattr(manager, 'set_window_title'):
			manager.set_window_title(str(title))
	except Exception:
		# Best-effort only; some backends don't support window titles.
		pass


def _normalize_offsets(x_offsets: Optional[np.ndarray]) -> np.ndarray:
	arr = np.asarray(x_offsets if x_offsets is not None else np.array([0.0]), dtype=float).ravel()
	if arr.size == 0:
		return np.array([0.0], dtype=float)
	return arr


def _coverage_for_x_offsets(
	*,
	ymics: np.ndarray,
	altitude: float,
	flight_path_angle_deg: float,
	x_offsets: np.ndarray,
	xmin: float,
	xmax: float,
	speed: float,
	rate: float,
	speed_of_sound: float,
) -> tuple[np.ndarray, np.ndarray]:
	"""Coverage for multiple linear arrays placed at different X offsets.

	This generalizes `flight_acoustics.array_coverage`, which assumes the array is at X=0.
	"""
	ymics = np.asarray(ymics, dtype=float).ravel()
	x_offsets = np.asarray(x_offsets, dtype=float).ravel()
	if x_offsets.size == 0:
		x_offsets = np.array([0.0], dtype=float)

	tmax = (float(xmax) - float(xmin)) / float(speed)
	time = np.arange(0.0, tmax, float(rate))

	fpa_rad = np.deg2rad(float(flight_path_angle_deg))
	tan_fpa = float(np.tan(fpa_rad))

	source = np.zeros((len(time), 3), dtype=float)
	source[:, 0] = float(xmin) + time * float(speed)
	# Flight path is a straight line in X-Z with path angle = fpa,
	# and must cross the provided altitude at x=0.
	source[:, 2] = float(altitude) + source[:, 0] * tan_fpa

	velocity = np.zeros_like(source)
	velocity[:, 0] = float(speed)
	velocity[:, 2] = float(speed) * tan_fpa

	az_all: list[np.ndarray] = []
	el_all: list[np.ndarray] = []
	for x0 in x_offsets:
		observers = np.zeros((len(ymics), 3), dtype=float)
		observers[:, 0] = float(x0)
		observers[:, 1] = ymics
		azimuth, elevation, _, _, _ = fa.hemigen(time, source, velocity, observers, float(speed_of_sound))
		az_all.append(np.asarray(azimuth, dtype=float).ravel())
		el_all.append(np.asarray(elevation, dtype=float).ravel())

	return np.concatenate(az_all), np.concatenate(el_all)


def _coverage_by_offset(
	*,
	ymics: np.ndarray,
	altitude: float,
	flight_path_angle_deg: float,
	x_offsets: np.ndarray,
	xmin: float,
	xmax: float,
	speed: float,
	rate: float,
	speed_of_sound: float,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
	results: list[tuple[float, np.ndarray, np.ndarray]] = []
	for x0 in _normalize_offsets(x_offsets):
		az_deg, el_deg = _coverage_for_x_offsets(
			ymics=ymics,
			altitude=altitude,
			flight_path_angle_deg=flight_path_angle_deg,
			x_offsets=np.array([x0], dtype=float),
			xmin=xmin,
			xmax=xmax,
			speed=speed,
			rate=rate,
			speed_of_sound=speed_of_sound,
		)
		results.append((float(x0), az_deg, el_deg))
	return results


def _plot_coverage_points(
	*,
	ax,
	coverage: list[tuple[float, np.ndarray, np.ndarray]],
	marker_size: float,
	marker_alpha: float,
	label_offsets: bool,
) -> None:
	if len(coverage) == 1:
		_, az_deg, el_deg = coverage[0]
		az = np.deg2rad(np.asarray(az_deg, dtype=float).ravel())
		el = np.deg2rad(np.asarray(el_deg, dtype=float).ravel())
		x, y = fa.lambert_ea(el, fa.lambert_lon(az))
		ax.plot(x, y, 'k.', markersize=marker_size, alpha=marker_alpha)
		return

	import matplotlib
	colors = matplotlib.colormaps['tab10']
	for i, (x0, az_deg, el_deg) in enumerate(coverage):
		az = np.deg2rad(np.asarray(az_deg, dtype=float).ravel())
		el = np.deg2rad(np.asarray(el_deg, dtype=float).ravel())
		x, y = fa.lambert_ea(el, fa.lambert_lon(az))
		label = f'x={x0:g}' if label_offsets else None
		ax.plot(x, y, '.', color=colors(i % 10), markersize=marker_size, alpha=marker_alpha, label=label)
	if label_offsets:
		ax.legend(loc='lower center', ncol=min(5, len(coverage)), frameon=False)


def _parse_csv_floats(value: str) -> np.ndarray:
	parts = [p.strip() for p in value.split(',') if p.strip()]
	if not parts:
		raise argparse.ArgumentTypeError('Expected a comma-separated list of floats')
	try:
		return np.array([float(p) for p in parts], dtype=float)
	except ValueError as e:
		raise argparse.ArgumentTypeError(str(e))


@dataclass(frozen=True)
class HemisphereGrid:
	azi_grid_deg: np.ndarray
	elv_grid_deg: np.ndarray
	spl_db: np.ndarray
	label: str


#: Levels at or below this in a hemisphere NPZ are the old no-data floor.
_OLD_NO_DATA_FLOOR_DB = -1000.0


#: --field choices stored as a single (Nelv, Nazi) level map: their npz key and label.
_NPZ_LEVEL_FIELDS = {
	'oaspl_fullband': ('hemisphere_oaspl_fullband_db', 'OASPL (full band), dB'),
	'oaspl_lt2khz': ('hemisphere_oaspl_lt2khz_db', 'OASPL (<2 kHz), dB'),
	'splA_lt2khz': ('hemisphere_splA_lt2khz_db', 'Overall A-weighted (<2 kHz), dBA'),
}


def _load_depropagated_hemisphere_npz(path: str, *, field: str, fc_hz: Optional[float]) -> HemisphereGrid:
	"""Load a depropagated hemisphere product saved as .npz.

	Supports the keys written by `AS350_289108_demo.py`.
	"""
	with np.load(path) as npz:
		required = {'azi_grid_deg', 'elv_grid_deg'}
		missing = sorted([k for k in required if k not in npz])
		if missing:
			raise ValueError(f"{path} is missing required keys: {missing}")

		azi_grid_deg = np.asarray(npz['azi_grid_deg'], dtype=float)
		elv_grid_deg = np.asarray(npz['elv_grid_deg'], dtype=float)

		if field in _NPZ_LEVEL_FIELDS:
			key, label = _NPZ_LEVEL_FIELDS[field]
			if key not in npz:
				raise ValueError(f"{path} does not contain '{key}'")
			spl = np.asarray(npz[key], dtype=float)
		elif field == 'third_octave':
			if fc_hz is None:
				raise ValueError('field=third_octave requires --fc')
			key_bands = 'hemisphere_bands_db'
			key_fc = 'band_centers_hz'
			if key_bands not in npz or key_fc not in npz:
				raise ValueError(f"{path} does not contain '{key_bands}' and '{key_fc}'")
			band_centers = np.asarray(npz[key_fc], dtype=float)
			bands_db = np.asarray(npz[key_bands], dtype=float)
			if bands_db.ndim != 3:
				raise ValueError(f"{key_bands} must have shape (Nb, Nelv, Nazi)")
			idx = int(np.argmin(np.abs(band_centers - float(fc_hz))))
			spl = bands_db[idx, :, :]
			label = f'Third-octave band @ {band_centers[idx]:.0f} Hz, dB'
		else:
			raise ValueError(f'Unknown hemisphere field: {field}')

	# Basic shape sanity check: grid is (Nelv, Nazi)
	if spl.ndim != 2:
		raise ValueError(f'SPL field must be 2D; got shape {spl.shape}')
	if spl.shape != (elv_grid_deg.size, azi_grid_deg.size):
		raise ValueError(
			f"SPL field shape {spl.shape} does not match (Nelv, Nazi)={(elv_grid_deg.size, azi_grid_deg.size)}"
		)
	# Files saved before 2026-09-26 hold -3076.5 dB (10 log10 of the smallest
	# float) where there was no data or no energy.  Read as a level it set the
	# color scale to -3500..500 dB and drew the overlay as one blob.
	spl = np.where(spl > _OLD_NO_DATA_FLOOR_DB, spl, np.nan)
	return HemisphereGrid(azi_grid_deg=azi_grid_deg, elv_grid_deg=elv_grid_deg, spl_db=spl, label=label)


def _check_overflight(args: argparse.Namespace) -> None:
	"""Refuse an altitude, speed or sampling period that is not positive."""
	for option, value in (('--altitude', args.altitude), ('--speed', args.speed), ('--rate', args.rate)):
		if not value > 0:
			raise ValueError(f'{option} must be positive, got {value:g}')


def _design_ymics(args: argparse.Namespace) -> np.ndarray:
	if args.ymics is not None:
		return np.asarray(args.ymics, dtype=float).ravel()
	if args.nmics is None:
		raise ValueError('Provide either --ymics or --nmics')
	if args.nmics < 2:
		raise ValueError(f'--nmics must be at least 2, got {args.nmics} '
						 '(for a single microphone under the track give --ymics 0)')
	if not 0 < args.min_elevation < 90:
		raise ValueError(f'--min-elevation must be between 0 and 90 deg, got {args.min_elevation:g}')
	if not 0 < args.target_elv <= 90:
		raise ValueError(f'--target-elv must be greater than 0 and at most 90 deg, got {args.target_elv:g}')
	return np.asarray(
		fa.linear_array_plan(
			args.nmics,
			args.altitude,
			min_elevation=args.min_elevation,
			target_elv=args.target_elv,
		),
		dtype=float,
	).ravel()


def _build_local_mic_coords(ymics: np.ndarray, x_offsets: np.ndarray, mic_height: float) -> np.ndarray:
	x_offsets = _normalize_offsets(x_offsets)
	n_arrays = int(x_offsets.size)
	n_mics = int(ymics.size)
	local = np.zeros((n_arrays * n_mics, 3), dtype=float)
	local[:, 0] = np.repeat(x_offsets, n_mics)
	local[:, 1] = np.tile(ymics, n_arrays)
	local[:, 2] = float(mic_height)
	return local


def cmd_design(args: argparse.Namespace) -> int:
	ymics = _design_ymics(args)
	if args.format == 'csv':
		print(','.join([f'{y:.6g}' for y in ymics]))
	elif args.format == 'json':
		print(json.dumps({'ymics': ymics.tolist(), 'altitude': float(args.altitude)}))
	else:
		raise ValueError(f'Unsupported format: {args.format}')
	return 0


def cmd_kml(args: argparse.Namespace) -> int:
	"""Write microphone coordinates to a KMZ/KML file.

	Microphone positions are assumed to be in the local array frame:
	- x: along heading / along-track (forward)
	- y: left of heading (cross-track)
	- z: up

	`--x-offsets` places additional identical arrays at different x locations.
	"""
	ymics = _design_ymics(args)
	local = _build_local_mic_coords(ymics, args.x_offsets, args.mic_height)

	reference = np.array([float(args.ref_lat), float(args.ref_lon), float(args.ref_alt)], dtype=float)
	geodetic = fa.array2geodetic(local, reference, float(args.heading), units=str(args.units))

	# `write_kml` writes a KMZ regardless of extension.
	savename = str(args.kmz)
	fa.write_kml(geodetic, savename, testname=str(args.name), channel_prefix=str(args.prefix))
	print(f'Wrote {savename} ({geodetic.shape[0]} points)')
	return 0


def cmd_coverage(args: argparse.Namespace) -> int:
	ymics = _design_ymics(args)
	coverage = _coverage_by_offset(
		ymics=ymics,
		altitude=args.altitude,
		flight_path_angle_deg=args.flight_path_angle,
		x_offsets=args.x_offsets,
		xmin=args.xmin,
		xmax=args.xmax,
		speed=args.speed,
		rate=args.rate,
		speed_of_sound=args.speed_of_sound,
	)
	fig, ax, _ = fa.lambert_ea_points(np.array([], dtype=float), np.array([], dtype=float))
	_plot_coverage_points(
		ax=ax,
		coverage=coverage,
		marker_size=2.0,
		marker_alpha=0.6,
		label_offsets=len(coverage) > 1,
	)
	_set_window_title(fig, args.title)
	save_or_show(fig, args.output, args.plot, bbox_inches='tight', facecolor='none', dpi=args.dpi)
	return 0


def cmd_overlay(args: argparse.Namespace) -> int:
	ymics = _design_ymics(args)

	# Optional background hemisphere plot
	if args.hemisphere:
		hemi = _load_depropagated_hemisphere_npz(args.hemisphere, field=args.field, fc_hz=args.fc)
		AZI_GRID, ELV_GRID = np.meshgrid(hemi.azi_grid_deg, hemi.elv_grid_deg)
		spl = hemi.spl_db.copy()
		fig, ax, _ = fa.plot_lambert_ea(
			np.deg2rad(AZI_GRID),
			np.deg2rad(ELV_GRID),
			spl,
			SPL_range=args.spl_range,
			weight=('A' if args.field == 'splA_lt2khz' else None),
		)
	else:
		# Empty Lambert projection (meridians/parallels) as a base layer
		fig, ax, _ = fa.lambert_ea_points(np.array([], dtype=float), np.array([], dtype=float))

	# Coverage points overlay (supports multiple X-offset arrays)
	coverage = _coverage_by_offset(
		ymics=ymics,
		altitude=args.altitude,
		flight_path_angle_deg=args.flight_path_angle,
		x_offsets=args.x_offsets,
		xmin=args.xmin,
		xmax=args.xmax,
		speed=args.speed,
		rate=args.rate,
		speed_of_sound=args.speed_of_sound,
	)
	_plot_coverage_points(
		ax=ax,
		coverage=coverage,
		marker_size=args.marker_size,
		marker_alpha=args.marker_alpha,
		label_offsets=len(coverage) > 1,
	)

	_set_window_title(fig, args.title)

	save_or_show(fig, args.output, args.plot, bbox_inches='tight', facecolor='none', dpi=args.dpi)
	return 0


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description='Design a linear mic array and visualize hemispherical coverage.',
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog=(
			"Examples:\n"
			"  # Print a 12-mic design (ymics) for a 150 m overflight\n"
			"  %(prog)s design --nmics 12 --altitude 150 --min-elevation 10 --target-elv 90\n\n"
			"  # Plot Lambert coverage points only\n"
			"  %(prog)s coverage --nmics 12 --altitude 150 -o demo_plots/array_coverage_lambert.png\n\n"
			"  # Plot multiple arrays at different X offsets\n"
			"  %(prog)s coverage --nmics 12 --altitude 150 --x-offsets -200,0,200 -o /tmp/coverage.png\n\n"
			"  # Add a climb/descent (flight path angle), crossing altitude at x=0\n"
			"  %(prog)s coverage --nmics 12 --altitude 150 --fpa 6 --x-offsets -200,0,200 -o /tmp/coverage_fpa6.png\n\n"
			"  # Overlay coverage on a depropagated hemisphere NPZ (from AS350_289108_demo.py)\n"
			"  %(prog)s overlay --nmics 12 --altitude 150 --hemisphere demo_plots/hemisphere_third_octave.npz "
			"--field oaspl_fullband -o demo_plots/coverage_on_hemisphere.png\n"
			"\n"
			"  # Export mic coordinates to KMZ (Google Earth)\n"
			"  %(prog)s kml --nmics 12 --altitude 150 --x-offsets -200,0,200 "
			"--ref-lat 34.0001 --ref-lon -118.0002 --ref-alt 250 --heading 90 "
			"--kmz array.kmz\n"
		),
	)

	# Common args are added to each subcommand so they can appear *after* the subcommand
	# (argparse requires global options to appear before subcommands otherwise).
	common = argparse.ArgumentParser(add_help=False)

	# Shared design inputs
	common.add_argument('--altitude', type=float, required=True,
						help='Vehicle altitude above array (defaults assume meters).')
	common.add_argument('--ymics', type=_parse_csv_floats, default=None,
						help='Explicit mic sideline locations as comma-separated list (overrides --nmics design).')
	common.add_argument('--nmics', type=int, default=None,
						help='Number of microphones (used when --ymics is not set).')
	common.add_argument('--min-elevation', type=float, default=10.0,
						help='Elevation (deg) of the outermost microphones above the horizon, measured in the '
							 '--target-elv plane; seen from directly overhead it is '
							 'atan(sin(target_elv) * tan(min_elevation)), which equals it only at --target-elv 90. '
							 'Between 0 and 90. Default: 10.')
	common.add_argument('--target-elv', type=float, default=90.0,
						help='Elevation (deg) of the plane in which the microphones are equally spaced in angle, '
							 'greater than 0 and at most 90. Default: 90.')

	# Shared overflight/coverage settings
	common.add_argument(
		'--flight-path-angle',
		'--fpa',
		dest='flight_path_angle',
		type=float,
		default=0.0,
		help='Flight path angle in degrees (positive climbs as x increases). Crosses --altitude at x=0. Default: 0.',
	)
	common.add_argument(
		'--x-offsets',
		'-x',
		'-x-offsets',
		dest='x_offsets',
		type=_parse_csv_floats,
		default=None,
		help='Comma-separated X offsets for additional arrays (same length units as altitude). Example: -x -200,0,200',
	)
	common.add_argument('--xmin', type=float, default=-1000.0,
						help='Overflight start x (same length units as altitude). Default: -1000.')
	common.add_argument('--xmax', type=float, default=1000.0,
						help='Overflight end x (same length units as altitude). Default: 1000.')
	common.add_argument('--speed', type=float, default=50.0,
						help='Overflight speed (same length units per second). Affects point density via dx=speed*rate. Default: 50 (m/s).')
	common.add_argument('--rate', type=float, default=0.1,
						help='Emission-time sampling period in seconds. Affects point density via dx=speed*rate. Default: 0.1.')
	common.add_argument('--speed-of-sound', type=float, default=343.0,
						help='Speed of sound (same length units per second). Does not affect az/el coverage angles. Default: 343 (m/s).')

	# Output controls (used by coverage/overlay; harmless for design)
	common.add_argument('-o', '--output', type=str, default=None, help='Output image file name.')
	common.add_argument('-p', '--plot', action='store_true', help='Also display plot when saving to file.')
	common.add_argument('--dpi', type=int, default=200, help='Output DPI when saving. Default: 200.')
	common.add_argument('--title', type=str, default=None, help='Optional window title (not drawn on the plot).')

	sub = parser.add_subparsers(dest='command', required=True)

	p_design = sub.add_parser('design', parents=[common], help='Print designed mic sideline positions (ymics).')
	p_design.add_argument('--format', choices=['csv', 'json'], default='csv', help='Output format. Default: csv.')
	p_design.set_defaults(func=cmd_design)

	p_cov = sub.add_parser('coverage', parents=[common],
						   help='Plot Lambert coverage points only (no hemisphere background).')
	p_cov.set_defaults(func=cmd_coverage)

	p_ov = sub.add_parser('overlay', parents=[common],
						  help='Overlay coverage points on a depropagated hemisphere (.npz).')
	p_ov.add_argument('--hemisphere', type=str, default=None,
					  help='Optional path to depropagated hemisphere .npz (if omitted, plots coverage only).')
	p_ov.add_argument(
		'--field',
		choices=['oaspl_fullband', 'oaspl_lt2khz', 'splA_lt2khz', 'third_octave'],
		default='oaspl_fullband',
		help='Hemisphere field to plot. Default: oaspl_fullband.',
	)
	p_ov.add_argument('--fc', type=float, default=None, help='Third-octave center frequency (Hz) when --field third_octave.')
	p_ov.add_argument('--spl-range', type=colon_pair, default=None, help='Color range as "min:max" (dB).')
	p_ov.add_argument('--marker-size', type=float, default=2.0, help='Overlay marker size. Default: 2.')
	p_ov.add_argument('--marker-alpha', type=float, default=0.4, help='Overlay marker alpha. Default: 0.4.')
	p_ov.set_defaults(func=cmd_overlay)

	p_kml = sub.add_parser('kml', parents=[common], help='Write microphone coordinates to a KMZ file.')
	p_kml.add_argument('--kmz', type=str, required=True, help='Output KMZ filename (e.g., array.kmz).')
	p_kml.add_argument('--ref-lat', type=float, required=True, help='Reference latitude (deg).')
	p_kml.add_argument('--ref-lon', type=float, required=True, help='Reference longitude (deg).')
	p_kml.add_argument('--ref-alt', type=float, default=0.0, help='Reference altitude/height (meters). Default: 0.')
	p_kml.add_argument('--heading', type=float, required=True,
					 help='Heading (deg), clockwise from north. Local +x aligns with heading.')
	p_kml.add_argument('--units', type=str, default='m', help='Units for local x/y/z (default: m).')
	p_kml.add_argument('--mic-height', type=float, default=0.0, help='Microphone height above reference in local units. Default: 0.')
	p_kml.add_argument('--name', type=str, default='Array', help='KML document name. Default: Array.')
	p_kml.add_argument('--prefix', type=str, default='M', help='Placemark name prefix. Default: M (M1, M2, ...).')
	p_kml.set_defaults(func=cmd_kml)

	return parser


def main(argv: Optional[list[str]] = None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)
	try:
		_check_overflight(args)
		return int(args.func(args))
	except Exception as e:
		parser.error(str(e))
		return 2


if __name__ == '__main__':
	raise SystemExit(main())

