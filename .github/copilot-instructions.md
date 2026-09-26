# Panam - Aeroacoustic Analysis Tools

## Project Overview
This is a Python-based aeroacoustics analysis toolkit for rotorcraft noise modeling and visualization. Core capabilities include acoustic signal processing, spherical noise source modeling, ground projection analysis, and Lambert equal-area visualization.

## Architecture & Data Flow

### Core Module: `flight_acoustics.py`
Central library containing all acoustic analysis and visualization functions. This is a **monolithic module** that other scripts import from. Key functional areas:

1. **Acoustic Signal Processing**: PSD computation (`psd`, `psd_welch`), spectrogram generation, A-weighting (`dBAw`), filtering
2. **Spherical Noise Sources**: Load/manipulate netCDF acoustic hemispheres (`load_nc_sphere`, `extract_SPL`), coordinate transforms (ART ↔ UMAPR via `art2umapr`)
3. **Ground Projection**: Project hemispherical noise to ground plane (`project_sphere`, `project_directory`) with atmospheric absorption correction
4. **Visualization**: Lambert equal-area projections (`lambert_ea`, `nc_lambert_ea`), "fried egg" flight condition plots (`fried_egg_plot`), spectrograms
5. **Flight Dynamics**: Moving source emissions (`hemigen`, `dedopplerize`), microphone array design (`linear_array_plan`)

### CLI Executables
Thin argparse wrappers around `flight_acoustics.py` functions:
- `nc_lambert_ea.py`: Spherical noise visualization  
- `plot_projection.py`: Ground noise contours
- `fried_egg_plot.py`: Flight condition noise "carpet plots"
- `spectrogram_plot.py`: Time-frequency analysis (supports HDF5/netCDF)
- `atmomap.py`: Atmospheric absorption visualization
- `build_empirical_database.py`: Compile netCDF sphere database

### Supporting Libraries
- `unit_conversion.py`: Aerospace unit conversions (originated from external library, extensive with functions like `avgas_conv`)
- `default_units.py`: Global unit defaults (ft, knots, °C, etc.)

## Key Conventions

### Data Formats
- **netCDF spheres**: Standard input format with variables `PHI`, `THETA`, `FREQUENCY`, `AMPLITUDE`, `RADIUS`, `SPEED`, `FLIGHT_PATH_ANGLE` (units: degrees, Hz, dB, feet, knots)
- **NORAH2 hemispheres**: ASCII `.hem` files (HELENA GAD layout) at 60 m with ICAO-reference absorption included, 10° grid, 31 one-third octave bands 10 Hz–10 kHz; write with `write_norah2_hemisphere`, read with `load_norah2_hemisphere`, and index a set with `write_norah2_triangulation`
- **HDF5 signals**: BKConnect format acoustic time series (use `load_h5_signal` with `datasetname='Table1'`)
- **Config files**: `vehicle.cfg` (INI format) in sphere directories defines vehicle geometry and operating conditions

### Coordinate Systems
- **ART (AAM/RNM/ANOPP)**: Aircraft noise standard with phi (lateral: 0 below, positive to starboard) and theta (longitudinal: 0 at the nose, 180 at the tail), per the AAM v3 Technical Reference sec. 2.4.1
- **UMAPR**: Azimuth/elevation convention used internally (azimuth 180 ahead, 90 starboard, 270 port, 0 behind; elevation positive below the horizon); convert via `art2umapr(phi, theta)`
- **NORAH2**: same (phi, theta) definition as ART; convert via `norah2umapr(phi, theta)`
- **Geodetic ↔ Local Array**: Use `geodetic2array`/`array2geodetic` with reference point and heading for microphone positioning

### Unit Philosophy
All internal calculations use **SI-adjacent units** (meters, Hz, Pa) but I/O defaults to **aerospace units** (feet, knots, °C) per `default_units.py`. Use `unit_conversion.py` functions liberally; they auto-convert via intermediate base units.

### Atmospheric Modeling
The vendored `panam_acoustics` package (derived from python-acoustics, BSD-3) provides ISO 9613-1 atmospheric absorption: `panam_acoustics.atmosphere.Atmosphere(temperature=K, pressure=kPa, relative_humidity=%)`. `atmosorb(freq, temp, humid, pstat)` wraps it and takes °C, % RH and mbar → dB/m.

## Critical Patterns

### Handling Invalid Data in Spheres
netCDF spheres mark missing bands with NaN, -999 (the AAM convention, which this module writes) or a large positive value (>1e34, older output). `mask_missing_levels(amplitude)` turns all three into -inf, i.e. "no energy" for OASPL integration. Suppress the resulting numpy warnings locally with `with np.errstate(invalid='ignore'):`, not the global `np.seterr`.

### Frequency Band Processing
A-weighting and OASPL integrate over frequency bands. Apply weighting **before** OASPL:
```python
Aweight = np.array([dBAw(f) for f in frequency])
SPLA = np.apply_along_axis(OASPL, 2, amplitude + Aweight)
```

### Matplotlib Style
Uses `fivethirtyeight` style globally with `matplotlib.rcParams` adjustments. Custom colormap helper `get_ylorrd_cmap(num_levels)` prefers palettable discrete maps but falls back to continuous.

### Spherical Interpolation
Modified Shepard's IDW used for hemispheric data (`shepIDW`, `IDWweights`). Uses geodesic distances (`geodist`) and Franke-Nielson radius-based weighting.

## Development Workflows

### Environment Setup
Python ≥ 3.11 (`local_paths.py` uses `tomllib`).
```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
pytest                     # from the repository root
```

### Running CLI Tools
The CLI wrappers take `-h` for help. Common pattern:
```bash
./nc_lambert_ea.py input.nc -f 100:10000 -w A -o output.png
./fried_egg_plot.py sphere_dir/ -a 500 -c 30 -m mean -o plot.pdf
```

### Adding Analysis Functions
New acoustic analysis belongs in `flight_acoustics.py`. Follow patterns:
- Use docstrings with Args/Returns sections
- Default to aerospace units in signatures, convert internally
- Return numpy arrays or matplotlib (fig, ax, cs) tuples
- Handle infinities/NaNs explicitly for acoustic data

### Vehicle Configuration
`vehicle.cfg` format (see `read_vehicle_data`). ConfigParser does not strip
inline comments, so keep values bare. Radii in m, tip speeds in m/s,
temperature in K, density in kg/m³, weight in kg, drag (flat-plate area) in m²:
```ini
[Main Rotor]
blades = 4
radius = 5.33
tip speed = 216

[Tail Rotor]
blades = 2
radius = 0.83
tip speed = 216

[Atmosphere]
temperature = 293.15
density = 1.225

[Vehicle]
weight = 2250
drag = 0.82
```

## Dependencies & External Interfaces
- **panam_acoustics** (vendored): ISO 9613-1 absorption, Butterworth filters, one-third octave filter bank
- **pymap3d**: Geodetic transformations (`geodetic2enu`, `enu2geodetic`)
- **simplekml**: KML/KMZ export for microphone arrays (`write_kml`)
- **palettable**: Discrete ColorBrewer palettes (YlOrRd preferred)
- **scipy.signal**: Spectrogram/periodogram/welch methods

## Common Pitfalls
- **Unit confusion**: Radii in netCDF are feet, must convert to meters (`* 0.3048`)
- **Speed conversion**: netCDF speeds in knots, use `0.514444` factor to m/s
- **Reference pressure**: Acoustic calculations use 20 µPa (`pref = 2.0e-5`)
- **Duration correction**: Use `Lmax + 10*log10(ref_speed/speed)` for SEL-like metrics
