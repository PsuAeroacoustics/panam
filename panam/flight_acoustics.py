"""Every name of the PANAM library from when it was this one module.

flight_acoustics held the library's functions until they were split into
topic modules (``spectra``, ``depropagation``, ``database`` and the others;
``docs/DEVELOPING.md`` has the module map).  This module re-imports every
function and constant it defined, private ones included, and the names it
re-exported for its users (the signal loaders of :mod:`panam.acoustics.signal_io`,
``Atmosphere``, ``BAND_ABSORPTION_METHODS`` and :mod:`panam.unit_conversion`), so
that ``from panam import flight_acoustics as fa`` and ``fa.depropagate_hemisphere``
keep working.  Other modules it happened to import (``np``,
``band_absorption_db`` and the like) are not re-exported.  It defines nothing, and new functions are not added to it.

Library modules import from the topic modules, not from here.  Replacing a
function here, as a test's monkeypatch does, does not change what the topic
modules call: patch the module that makes the call.
"""

from panam.acoustics.atmosphere import Atmosphere, BAND_ABSORPTION_METHODS
from panam import unit_conversion
from panam.acoustics.signal_io import load_nc_signal, load_h5_signal, open_h5_signal, load_UFF_signal

from panam.spectra import (P_REF, psd, psd_welch, overall_SPL, level_history, frame_length, nextpow2,
                           BASE2_BAND_CENTERS, _within_quarter_band, _bands_within, third_octave_band_levels,
                           third_octave_band_edges, spectrogram, plot_spectrogram, HANN_ENBW, hann_power,
                           _running_integral_at, _band_integrals, _running_floor, tone_aware_band_power,
                           _tone_aware_bands, _bands_missing, OASPL, power_to_db, safe_log10, dBAw, highpass,
                           lowpass)
from panam.metrics import (PNL_BAND_FREQUENCIES, _INF, _NOY_SPL_A, _NOY_SPL_B, _NOY_SPL_C, _NOY_SPL_D,
                           _NOY_SPL_E, _NOY_M_B, _NOY_M_C, _NOY_M_D, _NOY_M_E, noys, perceived_noise_level,
                           tone_correction, tone_corrected_perceived_noise_level, EPNL_REFERENCE_DURATION_S,
                           EPNL_HALF_SECOND_NORMALIZATION_DB, effective_perceived_noise_level,
                           ten_db_down_interval, sound_exposure_level, MIL_STD_1474E_TABLE_C1_FILE,
                           load_mil_std_1474e_table_c1, _mil_std_interp_log_frequency,
                           _MIL_STD_BAND_TOLERANCE, _mil_std_band_distance,
                           mil_std_1474e_nondetectability_distance)
from panam.geometry import (dedopplerize, hemigen, finite_difference_velocity, norah2umapr, art2umapr,
                            lambert_ea, lambert_lon, _lambert_xy_to_umapr, _umapr2art, geodist)
from panam.absorption import (_check_band_method, _band_attenuation_db, metadata_atmosphere,
                              path_absorption_db, atmosorb)
from panam.gridding import (interpolate_hemisphere_power, _interpolate_measured_power,
                            _sample_hemisphere_levels, ADAPTIVE_INTERPOLATION, adaptive_idw_weights, shepIDW,
                            shepIDW_weights, shepIDW_apply)
from panam.depropagation import (_AIR_RELATIVE_DOPPLER_RAYS, ambient_from_frames, gate_ambient,
                                 range_cap_mask, ray_azimuth_deg, air_relative_doppler_factor,
                                 depropagate_hemisphere)
from panam.sphere_io import (KNOT_MPS, load_nc_sphere, AAM_SCALAR_UNIT_WIDTH, AZIMUTH_REFERENCES,
                             WIND_SPEED_UNITS, WIND_SOURCES, POUND_FORCE_NEWTONS, RUN_METADATA_VARIABLES,
                             RUN_METADATA_ATTRIBUTES, normalize_wind_source, _empty_run_metadata,
                             read_run_metadata, _read_azimuth_reference, _azimuth_reference_of,
                             _read_doppler_shift_removed, _doppler_shift_removed_of, _read_scalar,
                             AAM_MISSING_LEVEL, AAM_VARIABLE_ORDER, AAM_MISSING_THRESHOLD,
                             AAM_LEGACY_MISSING_THRESHOLD, mask_missing_levels, band_index,
                             missing_level_sentinel, _atomic_output, AAM_DEFAULT_PHI_DEG,
                             AAM_DEFAULT_THETA_DEG, write_aam_hemisphere_netcdf)
from panam.database import (read_vehicle_rotor_data, DATABASE_FORMAT_VERSION, SPEED_REFERENCES,
                            LEVEL_FLIGHT_TOLERANCE, _SphereCondition, GROUP_RUN_METADATA_NUMBERS,
                            GROUP_RUN_METADATA_TEXT, CONDITION_ORIGINS, _resolve_azimuth_reference,
                            group_run_metadata, _resolve_doppler_shift_removed, average_fore_and_aft,
                            spla_and_eaa_from_spectrum, _complete_sphere, _complete_sphere_with_coverage,
                            _measured_bins, _fore_aft_measured, HOVER_FLIGHT_PATH_ANGLES,
                            _hover_flight_path_angles, _hover_source_index, _synthesized_hover_sphere,
                            mirror_phi_to_upper_surface, build_empirical_database, _sphere_condition,
                            _group_repeat_conditions, _extract_condition_SPL, _merged_run_metadata,
                            _select_sphere_files, _write_database_root, _write_shared_grid_and_frequency,
                            _grid_and_frequency_are_shared, _finite_sphere_levels, add_sphere_group,
                            STANDARD_GRAVITY, read_vehicle_weight_newtons, read_vehicle_data, extract_SPL,
                            _reduce_sphere_levels)
from panam.norah2 import (NORAH2_BAND_CENTERS_HZ, NORAH2_THETA_DEG, NORAH2_PHI_DEG,
                          NORAH2_REFERENCE_DISTANCE_M, NORAH2_REFERENCE_TEMPERATURE_K,
                          NORAH2_REFERENCE_RELATIVE_HUMIDITY, NORAH2_REFERENCE_PRESSURE_PA,
                          NORAH2_MISSING_LEVEL, NORAH2_DEFAULT_CORRECTIONS, _norah2_band_index,
                          write_norah2_hemisphere, write_norah2_levels, _norah2_numbers,
                          load_norah2_hemisphere, write_norah2_triangulation, NORAH2_SPEED_MAPPINGS,
                          STANDARD_SEA_LEVEL_DENSITY, DRY_AIR_GAS_CONSTANT, WATER_VAPOR_GAS_CONSTANT,
                          DATABASE_SPHERE_RADIUS_M, _NORAH2_REQUIRED_CONSTANTS, humid_air_density,
                          _norah2_reference_atmosphere, norah2_levels_at_radius, norah2_speed_knots,
                          fill_norah2_empty_cells, build_database_from_norah2, _norah2_rotor_speed_fraction,
                          _conditions_are_collinear, _check_build_atmosphere, _write_norah2_database)
from panam.lambert_plots import (_normalize_lambert_grid_convention, _configure_lambert_axes,
                                 _draw_lambert_grid, nice_levels, colorbar_ticks, plot_lambert_ea,
                                 nc_lambert_ea, lambert_ea_points, get_ylorrd_cmap)
from panam.projection import (project_sphere, project_directory, fried_egg_plot, plot_projection, data_filter,
                              is_noisy)
from panam.array_design import (linear_array_plan, array_coverage, hover_array_coverage,
                                takeoff_array_coverage, hover_array_coverage_plot, array_coverage_plot,
                                geodetic2array, array2geodetic, write_kml)
from panam.tracks import load_NASA_track, filter_track
from panam.ground_effect import spherical_reflection_coefficient, CHESSELL_MU, CHESSELL_ETA, ega
