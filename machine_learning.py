from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import export_graphviz

import airspeed
import standard_atmosphere
from flight_acoustics import *


def training_data(directory_names, altitude=500, cutoff=30, input_frequencies=None, fpa_climb_cutoff=5,
                  atmosphere=acoustics.atmosphere.Atmosphere(temperature=293.15, pressure=101.325,
                                                             relative_humidity=20.0), duration_correction=60):
    advance_ratio_sets = []
    weight_coefficient_sets = []
    alpha_sets = []
    flight_path_angle_sets = []
    hover_tip_mach_number_sets = []
    blade_number_sets = []
    Lmax_sets = []
    Lmean_sets = []
    tip_speeds = []
    effective_flat_plate_drag_areas = []
    for directory in directory_names:
        (speeds, fpas, Lmax, Lmean, mus, CWs, MHs, alphas,
         runs, MN, TN, fbar, MT) = project_directory(directory,
                                                     altitude, cutoff,
                                                     input_frequencies,
                                                     fpa_climb_cutoff,
                                                     atmosphere, duration_correction)
        advance_ratio_sets.append(expand_if_single(mus, Lmax))
        weight_coefficient_sets.append(expand_if_single(CWs, Lmax))
        alpha_sets.append(expand_if_single(alphas, Lmax))
        flight_path_angle_sets.append(expand_if_single(fpas, Lmax))
        hover_tip_mach_number_sets.append(expand_if_single(MHs, Lmax))
        blade_number_sets.append(expand_if_single(MN, Lmax))
        Lmax_sets.append(Lmax)
        Lmean_sets.append(Lmean)
        tip_speeds.append(MT)
        effective_flat_plate_drag_areas.append(fbar)
    return (advance_ratio_sets, weight_coefficient_sets, alpha_sets, hover_tip_mach_number_sets, blade_number_sets,
            Lmax_sets, Lmean_sets, effective_flat_plate_drag_areas, tip_speeds, flight_path_angle_sets)


def package_data(advance_ratio_sets, weight_coefficient_sets, alpha_sets, hover_tip_mach_number_sets,
                 blade_number_sets, L_sets, flight_path_angle_sets, threshhold=0.65, cull_noisy_fpa=-1.0,
                 exclude_sets=None, quiet_alpha_buffer=6.0, quiet_advance_ratio=0.0, balance=None):
    advance_ratios = np.array([])
    weight_coefficients = np.array([])
    alphas = np.array([])
    hover_tip_mach_numbers = np.array([])
    numbers_of_blades = np.array([])
    noisy = np.array([])
    if exclude_sets is None:
        exclude_sets = []
    for set_index, _ in enumerate(advance_ratio_sets):
        if not np.isin(set_index, exclude_sets):
            if cull_noisy_fpa is not None:
                mask = data_filter(L_sets[set_index], flight_path_angle_sets[set_index], threshhold, cull_noisy_fpa)
            else:
                mask = np.ones_like(L_sets[set_index])
            advance_ratios = np.append(advance_ratios, advance_ratio_sets[set_index][mask])
            weight_coefficients = np.append(weight_coefficients, weight_coefficient_sets[set_index][mask])
            alphas = np.append(alphas, alpha_sets[set_index][mask])
            max_alpha = np.max(alpha_sets[set_index][mask])
            hover_tip_mach_numbers = np.append(hover_tip_mach_numbers, hover_tip_mach_number_sets[set_index][mask])
            numbers_of_blades = np.append(numbers_of_blades, blade_number_sets[set_index][mask])
            noisy = np.append(noisy, is_noisy(L_sets[set_index][mask], threshhold))
            # Augment with quiet data at high alpha to enclose noisy area
            if quiet_alpha_buffer is not None:
                quiet_advance_ratios = np.linspace(np.min(advance_ratio_sets[set_index][mask]),
                                                   np.max(advance_ratio_sets[set_index][mask]), 10)
                advance_ratios = np.append(advance_ratios, quiet_advance_ratios)
                alphas = np.append(alphas, np.full_like(quiet_advance_ratios, max_alpha + quiet_alpha_buffer))
                weight_coefficients = np.append(weight_coefficients, np.full_like(quiet_advance_ratios, np.asscalar(
                    np.mean(weight_coefficient_sets[set_index][mask]))))
                hover_tip_mach_numbers = np.append(hover_tip_mach_numbers, np.full_like(quiet_advance_ratios,
                                                                                        np.asscalar(np.mean(
                                                                                            hover_tip_mach_number_sets[
                                                                                                set_index][mask]))))
                numbers_of_blades = np.append(numbers_of_blades, np.full_like(quiet_advance_ratios, np.asscalar(
                    np.mean(blade_number_sets[set_index][mask]))))
                noisy = np.append(noisy, np.full_like(quiet_advance_ratios, 0))
            # Augment with quiet data at low advance ratio to enclose noisy area
            if quiet_advance_ratio is not None:
                quiet_alphas = np.linspace(np.min(alpha_sets[set_index][mask]),
                                           np.max(alpha_sets[set_index][mask]), 10)
                alphas = np.append(alphas, quiet_alphas)
                advance_ratios = np.append(advance_ratios, np.full_like(quiet_alphas, quiet_advance_ratio))
                weight_coefficients = np.append(weight_coefficients, np.full_like(quiet_alphas, np.asscalar(
                    np.mean(weight_coefficient_sets[set_index][mask]))))
                hover_tip_mach_numbers = np.append(hover_tip_mach_numbers, np.full_like(quiet_alphas, np.asscalar(
                    np.mean(hover_tip_mach_number_sets[set_index][mask]))))
                numbers_of_blades = np.append(numbers_of_blades, np.full_like(quiet_alphas, np.asscalar(
                    np.mean(blade_number_sets[set_index][mask]))))
                noisy = np.append(noisy, np.full_like(quiet_alphas, 0))
    condition_matrix = np.column_stack(
        (advance_ratios, alphas, weight_coefficients, hover_tip_mach_numbers, numbers_of_blades))
    classifications = noisy.astype(int)
    # Balance undersampled data
    if balance is 'ada':
        sampler = ADASYN(random_state=42, n_jobs=-1)
        condition_matrix, classifications = sampler.fit_sample(condition_matrix, classifications)
    elif balance is 'random-over':
        sampler = RandomOverSampler(random_state=42)
        condition_matrix, classifications = sampler.fit_sample(condition_matrix, classifications)
    elif balance is 'random-under':
        sampler = RandomUnderSampler(random_state=42)
        condition_matrix, classifications = sampler.fit_sample(condition_matrix, classifications)
    elif balance is 'smote':
        sampler = SMOTE(random_state=42, n_jobs=-1)
        condition_matrix, classifications = sampler.fit_sample(condition_matrix, classifications)
    elif balance is 'cluster-majority':
        sampler = ClusterCentroids(ratio='majority', random_state=42, n_jobs=-1)
        condition_matrix, classifications = sampler.fit_sample(condition_matrix, classifications)
    elif balance is 'cluster-minority':
        sampler = ClusterCentroids(ratio='minority', random_state=42, n_jobs=-1)
        condition_matrix, classifications = sampler.fit_sample(condition_matrix, classifications)
    elif balance is 'cluster':
        sampler = ClusterCentroids(ratio='auto', random_state=42, n_jobs=-1)
        condition_matrix, classifications = sampler.fit_sample(condition_matrix, classifications)
    return condition_matrix, classifications


def probability_plot(classifier, scaler=None, samples=100, advance_ratio_range=(0.05, 0.35), alpha_range=(-12.0, 12.0),
                     weight_coefficient=0.006, hover_tip_mach_number=0.6, blade_number=5,
                     advance_ratio_scatter=None, alpha_scatter=None, noisy_index=None, save_name=None,
                     window_name=None):
    # Probability plot
    advance_ratios = np.linspace(advance_ratio_range[0], advance_ratio_range[1], samples)
    alpha = np.linspace(alpha_range[0], alpha_range[1], samples)
    advance_ratio_mesh, alpha_mesh = np.meshgrid(advance_ratios, alpha)
    weight_coefficient_mesh = weight_coefficient * np.ones_like(advance_ratio_mesh)
    hover_tip_mach_number_mesh = hover_tip_mach_number * np.ones_like(advance_ratio_mesh)
    blade_number_mesh = blade_number * np.ones_like(advance_ratio_mesh)

    sample_input = np.column_stack((advance_ratio_mesh.flatten(), alpha_mesh.flatten(),
                                    weight_coefficient_mesh.flatten(), hover_tip_mach_number_mesh.flatten(),
                                    blade_number_mesh.flatten()))
    if scaler is not None:
        sample_input = scaler.transform(sample_input)

    probabilities = classifier.predict_proba(sample_input)
    probability_quiet = probabilities[:, 0]
    probability_noisy = probabilities[:, 1]
    probability_quiet_mesh = np.reshape(probability_quiet, (samples, samples))
    probability_noisy_mesh = np.reshape(probability_noisy, (samples, samples))
    num_levels = 11
    levels = np.linspace(0, 100, num_levels)
    color_map = brewer2mpl.get_map('RdBu', 'diverging', num_levels, reverse=True)
    fig, ax = subplots(facecolor='white')
    cs = ax.contourf(advance_ratios, alpha, 100 * probability_noisy_mesh, levels, colors=color_map.mpl_colors, zorder=0)
    cs.set_clim(vmin=0, vmax=100)
    if advance_ratio_scatter is not None and alpha_scatter is not None and noisy_index is not None:
        quiet_index = np.logical_not(noisy_index)
        ax.plot(advance_ratio_scatter[quiet_index], alpha_scatter[quiet_index], 'ko', alpha=.8,
                markeredgecolor='w', markersize=10)
        ax.plot(advance_ratio_scatter[noisy_index], alpha_scatter[noisy_index], 'ro', alpha=.8,
                markeredgecolor='k', markersize=10)
    ax.set_xlabel('Advance Ratio - μ')
    ax.set_ylabel('Angle of Attack - α, °')
    ax.tick_params(axis='x', pad=6)
    ax.xaxis.grid(True, zorder=1)
    ax.yaxis.grid(True, zorder=1)
    cb = colorbar(cs)
    cb.set_label('Probability of Noisy Condition (%)')
    if window_name is not None:
        fig.canvas.set_window_title(window_name)
    if save_name is not None:
        fig.savefig(save_name, facecolor='none')
    return fig, ax


def classification_plot(classifier, scaler=None, samples=100, speed_range=(40.0, 130.0), climb_rate_range=(-2500, 0),
                        tip_speeds=300, effective_flat_plate_drag=10, weight_coefficient=0.006,
                        hover_tip_mach_number=0.6, blade_number=5, speed_scatter=None, roc_scatter=None,
                        noisy_index=None, save_name=None, window_name=None, fried_egg_directory=None,
                        duration_correction=60):
    # Dimensional classification plot
    speeds = np.linspace(speed_range[0], speed_range[1], samples)
    rates_of_climb = np.linspace(climb_rate_range[0], climb_rate_range[1], samples)

    speed_mesh, rate_of_climb_mesh = np.meshgrid(speeds, rates_of_climb)
    flight_path_angle_mesh = np.degrees(np.arcsin(rate_of_climb_mesh / (1.6878 * 60 * speed_mesh)))
    advance_ratio_mesh = 0.514444 * speed_mesh / tip_speeds
    weight_coefficient_mesh = weight_coefficient * np.ones_like(advance_ratio_mesh)
    drag_to_weight_mesh = 0.5 * effective_flat_plate_drag * advance_ratio_mesh ** 2 / weight_coefficient_mesh
    alpha_mesh = -np.degrees(drag_to_weight_mesh) - flight_path_angle_mesh
    hover_tip_mach_number_mesh = hover_tip_mach_number * np.ones_like(advance_ratio_mesh)
    blade_number_mesh = blade_number * np.ones_like(advance_ratio_mesh)
    sample_input = np.column_stack((advance_ratio_mesh.flatten(), alpha_mesh.flatten(),
                                    weight_coefficient_mesh.flatten(), hover_tip_mach_number_mesh.flatten(),
                                    blade_number_mesh.flatten()))
    if scaler is not None:
        sample_input = scaler.transform(sample_input)
    choices = classifier.predict(sample_input)
    classified_region_mesh = np.reshape(choices, (samples, samples))
    if fried_egg_directory is not None:
        fig, ax, cs = fried_egg_plot(fried_egg_directory, climb_rates=True, cull_noisy_fpa=-1.0,
                                     suppress_classification=True, duration_correction=duration_correction)
        hatch_color = 'none'
    else:
        fig, ax = subplots(facecolor='white')
        hatch_color = 'white'
    ax.contour(speeds, rates_of_climb, classified_region_mesh, levels=[0.5, 1.5], colors='black')
    rcParams['hatch.color'] = 'black'
    ax.contourf(speeds, rates_of_climb, classified_region_mesh, hatches=['xx'], levels=[0.5, 1.5], colors=hatch_color)

    if speed_scatter is not None and roc_scatter is not None and noisy_index is not None:
        quiet_index = np.logical_not(noisy_index)
        ax.plot(speed_scatter[quiet_index], roc_scatter[quiet_index], 'ko',
                alpha=.8, markeredgecolor='w', markersize=10)
        ax.plot(speed_scatter[noisy_index], roc_scatter[noisy_index], 'ro',
                alpha=.8, markeredgecolor='k', markersize=10)
    ax.set_xlabel('Speed, kts')
    ax.set_ylabel('Rate of Climb, fpm')
    ax.set_xlim((35, 140))
    ax.set_ylim((-2000, 750))
    if window_name is not None:
        fig.canvas.set_window_title(window_name)
    if save_name is not None:
        fig.savefig(save_name, facecolor='none')
    return fig, ax


def classification_drag_plot(classifier, scaler=None, samples=100, speed_range=(40.0, 130.0),
                             climb_rate_range=(-2500, 0), tip_speeds=300, effective_flat_plate_drags=(0.04, 0.07, 0.15),
                             labels=None, weight_coefficient=0.006, hover_tip_mach_number=0.6, blade_number=5,
                             save_name=None, window_name=None):
    fig, ax = subplots(facecolor='white')
    colorlist = ['black', 'blue', 'red', 'green', 'magenta', 'cyan']
    hatchlist = ['++', 'xx', 'O', '-', '.', '/']
    handles = []

    for index, effective_flat_plate_drag in enumerate(effective_flat_plate_drags):
        # Dimensional classification plot
        speeds = np.linspace(speed_range[0], speed_range[1], samples)
        rates_of_climb = np.linspace(climb_rate_range[0], climb_rate_range[1], samples)

        speed_mesh, rate_of_climb_mesh = np.meshgrid(speeds, rates_of_climb)
        flight_path_angle_mesh = np.degrees(np.arcsin(rate_of_climb_mesh / (1.6878 * 60 * speed_mesh)))
        advance_ratio_mesh = 0.514444 * speed_mesh / tip_speeds
        weight_coefficient_mesh = weight_coefficient * np.ones_like(advance_ratio_mesh)
        drag_to_weight_mesh = 0.5 * effective_flat_plate_drag * advance_ratio_mesh ** 2 / weight_coefficient_mesh
        alpha_mesh = -np.degrees(drag_to_weight_mesh) - flight_path_angle_mesh
        hover_tip_mach_number_mesh = hover_tip_mach_number * np.ones_like(advance_ratio_mesh)
        blade_number_mesh = blade_number * np.ones_like(advance_ratio_mesh)
        sample_input = np.column_stack((advance_ratio_mesh.flatten(), alpha_mesh.flatten(),
                                        weight_coefficient_mesh.flatten(), hover_tip_mach_number_mesh.flatten(),
                                        blade_number_mesh.flatten()))
        if scaler is not None:
            sample_input = scaler.transform(sample_input)
        choices = classifier.predict(sample_input)
        classified_region_mesh = np.reshape(choices, (samples, samples))
        hatch_color = 'none'
        ax.contour(speeds, rates_of_climb, classified_region_mesh, levels=[0.5, 1.5], colors=colorlist[index])
        rcParams['hatch.color'] = colorlist[index]
        handle = ax.contourf(speeds, rates_of_climb, classified_region_mesh, hatches=hatchlist[index],
                             levels=[0.5, 1.5], colors=hatch_color)
        artists, _ = handle.legend_elements()
        handles.append(artists[0])
    ax.set_xlabel('Speed, kts')
    ax.set_ylabel('Rate of Climb, fpm')
    ax.set_xlim((35, 140))
    ax.set_ylim((-2000, 750))
    legend(handles, labels)
    if window_name is not None:
        fig.canvas.set_window_title(window_name)
    if save_name is not None:
        fig.savefig(save_name, facecolor='none')
    return fig, ax


def classification_altitude_plot(classifier, scaler=None, samples=100, airspeed_range=(40.0, 130.0),
                                 airspeed_type='tas', climb_rate_range=(-2500, 0), altitudes=(0, 2000, 4000),
                                 tip_speeds=300, effective_flat_plate_drag=0.07, labels=None,
                                 sea_level_weight_coefficient=0.006, sea_level_hover_tip_mach_number=0.6,
                                 blade_number=5, save_name=None, window_name=None):
    fig, ax = subplots(facecolor='white')
    colorlist = ['black', 'blue', 'red', 'green', 'magenta', 'cyan']
    hatchlist = ['++', 'xx', 'O', '-', '.', '/']
    handles = []

    sea_level_temperature = standard_atmosphere.isa2temp(ISA_dev=0, altitude=0)
    sea_level_sound_speed = standard_atmosphere.temp2speed_of_sound(sea_level_temperature)
    for index, altitude in enumerate(altitudes):
        density_ratio = standard_atmosphere.alt2density_ratio(altitude)
        weight_coefficient = sea_level_weight_coefficient / density_ratio
        temperature = standard_atmosphere.isa2temp(ISA_dev=0, altitude=altitude)
        sound_speed = standard_atmosphere.temp2speed_of_sound(temperature)
        hover_tip_mach_number = sea_level_hover_tip_mach_number * sea_level_sound_speed / sound_speed

        # Dimensional classification plot
        speeds = np.linspace(airspeed_range[0], airspeed_range[1], samples)
        if airspeed_type is 'cas':
            true_airspeeds = np.array([airspeed.cas2tas(speed, altitude) for speed in speeds])
        else:
            true_airspeeds = speeds
        rates_of_climb = np.linspace(climb_rate_range[0], climb_rate_range[1], samples)
        speed_mesh, rate_of_climb_mesh = np.meshgrid(speeds, rates_of_climb)
        true_airspeed_mesh, rate_of_climb_mesh = np.meshgrid(true_airspeeds, rates_of_climb)
        flight_path_angle_mesh = np.degrees(np.arcsin(rate_of_climb_mesh / (1.6878 * 60 * speed_mesh)))
        advance_ratio_mesh = 0.514444 * true_airspeed_mesh / tip_speeds
        weight_coefficient_mesh = weight_coefficient * np.ones_like(advance_ratio_mesh)
        drag_to_weight_mesh = 0.5 * effective_flat_plate_drag * advance_ratio_mesh ** 2 / weight_coefficient_mesh
        alpha_mesh = -np.degrees(drag_to_weight_mesh) - flight_path_angle_mesh
        hover_tip_mach_number_mesh = hover_tip_mach_number * np.ones_like(advance_ratio_mesh)
        blade_number_mesh = blade_number * np.ones_like(advance_ratio_mesh)
        sample_input = np.column_stack((advance_ratio_mesh.flatten(), alpha_mesh.flatten(),
                                        weight_coefficient_mesh.flatten(), hover_tip_mach_number_mesh.flatten(),
                                        blade_number_mesh.flatten()))
        if scaler is not None:
            sample_input = scaler.transform(sample_input)
        choices = classifier.predict(sample_input)
        classified_region_mesh = np.reshape(choices, (samples, samples))
        hatch_color = 'none'
        ax.contour(speeds, rates_of_climb, classified_region_mesh, levels=[0.5, 1.5], colors=colorlist[index])
        rcParams['hatch.color'] = colorlist[index]
        handle = ax.contourf(speeds, rates_of_climb, classified_region_mesh, hatches=hatchlist[index],
                             levels=[0.5, 1.5], colors=hatch_color)
        artists, _ = handle.legend_elements()
        handles.append(artists[0])
    if airspeed_type is 'cas':
        ax.set_xlabel('Calibrated Airspeed, kts')
    else:
        ax.set_xlabel('True Airspeed, kts')
    ax.set_ylabel('Rate of Climb, fpm')
    ax.set_xlim((35, 140))
    ax.set_ylim((-2000, 750))
    legend(handles, labels)
    if window_name is not None:
        fig.canvas.set_window_title(window_name)
    if save_name is not None:
        fig.savefig(save_name, facecolor='none')
    return fig, ax


def train_data(directory_names, ref_idx=3, duration_correction=60, threshhold=0.65, balance='smote', save_prefix='',
               exclude_reference=True, grid_search=False):
    (advance_ratio_sets, weight_coefficient_sets, alpha_sets, hover_tip_mach_number_sets, blade_number_sets,
     Lmax_sets, Lmean_sets, effective_flat_plate_drag_areas,
     tip_speeds, flight_path_angle_sets) = training_data(directory_names, altitude=500, cutoff=30,
                                                         input_frequencies=None, fpa_climb_cutoff=5,
                                                         atmosphere=acoustics.atmosphere.Atmosphere(temperature=293.15,
                                                                                                    pressure=101.325,
                                                                                                    relative_humidity=20.0),
                                                         duration_correction=duration_correction)
    L_sets = Lmean_sets
    cull_noisy_fpa = -1.0
    if exclude_reference:
        exclusions = [ref_idx]
    else:
        exclusions = []

    conditions_matrix, classifications = package_data(advance_ratio_sets, weight_coefficient_sets,
                                                      alpha_sets, hover_tip_mach_number_sets,
                                                      blade_number_sets, L_sets,
                                                      flight_path_angle_sets, threshhold,
                                                      cull_noisy_fpa, exclusions, balance=balance)

    scaler = preprocessing.StandardScaler().fit(conditions_matrix)
    scaled_conditions = scaler.transform(conditions_matrix)
    if grid_search:
        parameters = {'C': np.logspace(1, 2, 3), 'gamma': np.logspace(-4, -1, 5), 'kernel': ['rbf']}
        SVclassifier = GridSearchCV(estimator=SVC(probability=True), param_grid=parameters, cv=5)
    else:
        SVclassifier = SVC(C=10, gamma=0.1, probability=True)
    SVclassifier.fit(scaled_conditions, classifications)
    SVscores = cross_val_score(SVclassifier, scaled_conditions, classifications, cv=5, n_jobs=-1)
    print("SV Classifier score is %g" % np.mean(SVscores))
    if grid_search:
        print(SVclassifier.best_params_)
    RFclassifier = RandomForestClassifier(n_jobs=-1, random_state=42)
    RFclassifier.fit(scaled_conditions, classifications)
    importances = RFclassifier.feature_importances_
    RFscores = cross_val_score(RFclassifier, scaled_conditions, classifications, cv=5, n_jobs=-1)
    print("RF Classifier score is %g" % np.mean(RFscores))
    print("RF Feature Importances are %s" % str(importances))
    fig, ax = subplots(facecolor='white')
    importance_order = np.argsort(-importances)
    features = np.array([r'$\mu$', r'$\alpha$', r'$C_W$', r'$M_H$', r'$N_b$'])
    ax.bar(range(len(features)), importances[importance_order], tick_label=features[importance_order], color='k')
    ax.set_ylabel('Gini Importance')
    ax.set_xlabel('Feature')
    fig.savefig(save_prefix + 'RF_importance.pdf', facecolor='none')

    export_graphviz(RFclassifier.estimators_[0], out_file='tree.dot',
                    feature_names=['Advance Ratio', 'Angle of Attack', 'Weight Coefficient', 'Hover Tip Mach Number',
                                   'Blade Number'], class_names=['Not Noisy', 'Noisy'], filled=True, rounded=False,
                    rotate=False, precision=2, proportion=True, max_depth=5, label='root')
    os.system('dot -Tps2 tree.dot -o tree.ps')
    print(len(RFclassifier.estimators_))

    NNclassifier = MLPClassifier(max_iter=2000, random_state=42)
    NNclassifier.fit(scaled_conditions, classifications)
    NNscores = cross_val_score(NNclassifier, scaled_conditions, classifications, cv=5, n_jobs=-1)
    print("NN Classifier score is %g" % np.mean(NNscores))

    # Select empirical data to superimpose
    advance_ratio_scatter = advance_ratio_sets[ref_idx]
    alpha_scatter = alpha_sets[ref_idx]
    noisy_index = is_noisy(L_sets[ref_idx])
    # Set data ranges
    advance_ratio_range = (0.05, 0.35)
    alpha_range = (-15.0, 15.0)
    weight_coefficient = np.asscalar(np.mean(weight_coefficient_sets[ref_idx]))
    hover_tip_mach_number = np.asscalar(np.mean(hover_tip_mach_number_sets[ref_idx]))
    blade_number = np.asscalar(np.mean(blade_number_sets[ref_idx]))

    # Plot probabilities
    probability_plot(SVclassifier, scaler, advance_ratio_scatter=advance_ratio_scatter, alpha_scatter=alpha_scatter,
                     advance_ratio_range=advance_ratio_range, alpha_range=alpha_range,
                     hover_tip_mach_number=hover_tip_mach_number, weight_coefficient=weight_coefficient,
                     noisy_index=noisy_index, save_name=save_prefix + 'SVCGS_Probability.pdf',
                     window_name='Support Vector')
    probability_plot(RFclassifier, scaler, advance_ratio_scatter=advance_ratio_scatter, alpha_scatter=alpha_scatter,
                     advance_ratio_range=advance_ratio_range, alpha_range=alpha_range,
                     hover_tip_mach_number=hover_tip_mach_number, weight_coefficient=weight_coefficient,
                     noisy_index=noisy_index, save_name=save_prefix + 'RF_Probability.pdf', window_name='Random Forest')
    probability_plot(NNclassifier, scaler, advance_ratio_scatter=advance_ratio_scatter, alpha_scatter=alpha_scatter,
                     advance_ratio_range=advance_ratio_range, alpha_range=alpha_range,
                     hover_tip_mach_number=hover_tip_mach_number, weight_coefficient=weight_coefficient,
                     noisy_index=noisy_index, save_name=save_prefix + 'NN_Probability.pdf', window_name='Neural Net')

    tip_speeds = tip_speeds[ref_idx]
    effective_flat_plate_drag = effective_flat_plate_drag_areas[ref_idx]
    speed_scatter = advance_ratio_sets[ref_idx] * tip_speeds / 0.514444

    speed_range = (35.0, 130.0)
    climb_rate_range = (-2500, 500)

    data_drag_to_weight = 0.5 * effective_flat_plate_drag * advance_ratio_sets[ref_idx] ** 2 / weight_coefficient_sets[
        ref_idx]
    data_flight_path_angles = -np.degrees(data_drag_to_weight) - alpha_sets[ref_idx]
    roc_scatter = speed_scatter * 1.6878 * 60 * np.sin(np.radians(data_flight_path_angles))

    print('Weight coefficient:    %g' % weight_coefficient)
    print('Hover tip Mach #:      %g' % hover_tip_mach_number)
    print('Flat plate drag f/A:   %g' % effective_flat_plate_drag)

    classification_plot(SVclassifier, scaler, tip_speeds=tip_speeds,
                        effective_flat_plate_drag=effective_flat_plate_drag, speed_range=speed_range,
                        climb_rate_range=climb_rate_range, hover_tip_mach_number=hover_tip_mach_number,
                        weight_coefficient=weight_coefficient, blade_number=blade_number,
                        save_name=save_prefix + 'SVCGS_Classification.pdf', window_name='Support Vector',
                        fried_egg_directory=directory_names[ref_idx], duration_correction=duration_correction)
    classification_plot(RFclassifier, scaler, tip_speeds=tip_speeds,
                        effective_flat_plate_drag=effective_flat_plate_drag, speed_range=speed_range,
                        climb_rate_range=climb_rate_range, hover_tip_mach_number=hover_tip_mach_number,
                        weight_coefficient=weight_coefficient, blade_number=blade_number,
                        save_name=save_prefix + 'RF_Classification.pdf', window_name='Random Forest',
                        fried_egg_directory=directory_names[ref_idx], duration_correction=duration_correction)
    classification_plot(NNclassifier, scaler, tip_speeds=tip_speeds,
                        effective_flat_plate_drag=effective_flat_plate_drag, speed_range=speed_range,
                        climb_rate_range=climb_rate_range, hover_tip_mach_number=hover_tip_mach_number,
                        weight_coefficient=weight_coefficient, blade_number=blade_number,
                        save_name=save_prefix + 'NN_Classification.pdf', window_name='Neural Net',
                        fried_egg_directory=directory_names[ref_idx], duration_correction=duration_correction)

    effective_flat_plate_drags = (0.5 * effective_flat_plate_drag, effective_flat_plate_drag,
                                  2 * effective_flat_plate_drag)
    labels = tuple(map("$f/A = ${:.3f}".format, effective_flat_plate_drags))
    classification_drag_plot(SVclassifier, scaler, tip_speeds=tip_speeds,
                             effective_flat_plate_drags=effective_flat_plate_drags, speed_range=speed_range,
                             climb_rate_range=climb_rate_range, hover_tip_mach_number=hover_tip_mach_number,
                             weight_coefficient=weight_coefficient, blade_number=blade_number,
                             save_name=save_prefix + 'drag_trend_SVGS.pdf', window_name='Support Vector', labels=labels)
    classification_drag_plot(RFclassifier, scaler, tip_speeds=tip_speeds,
                             effective_flat_plate_drags=effective_flat_plate_drags, speed_range=speed_range,
                             climb_rate_range=climb_rate_range, hover_tip_mach_number=hover_tip_mach_number,
                             weight_coefficient=weight_coefficient, blade_number=blade_number,
                             save_name=save_prefix + 'drag_trend_RF.pdf', window_name='Random Forest', labels=labels)
    classification_drag_plot(NNclassifier, scaler, tip_speeds=tip_speeds,
                             effective_flat_plate_drags=effective_flat_plate_drags, speed_range=speed_range,
                             climb_rate_range=climb_rate_range, hover_tip_mach_number=hover_tip_mach_number,
                             weight_coefficient=weight_coefficient, blade_number=blade_number,
                             save_name=save_prefix + 'drag_trend_NN.pdf', window_name='Neural Network', labels=labels)

    altitudes = (0, 4000, 8000)
    labels = tuple(map("{:d} ft MSL".format, altitudes))
    sea_level_weight_coefficient = 1.08 / 1.225 * weight_coefficient
    classification_altitude_plot(SVclassifier, scaler, tip_speeds=tip_speeds, airspeed_type='cas',
                                 effective_flat_plate_drag=effective_flat_plate_drag, airspeed_range=speed_range,
                                 climb_rate_range=climb_rate_range,
                                 sea_level_hover_tip_mach_number=hover_tip_mach_number,
                                 sea_level_weight_coefficient=sea_level_weight_coefficient, blade_number=blade_number,
                                 save_name=save_prefix + 'altitude_trend_cas_SVGS.pdf', window_name='Support Vector',
                                 labels=labels)
    classification_altitude_plot(SVclassifier, scaler, tip_speeds=tip_speeds, airspeed_type='tas',
                                 effective_flat_plate_drag=effective_flat_plate_drag, airspeed_range=speed_range,
                                 climb_rate_range=climb_rate_range,
                                 sea_level_hover_tip_mach_number=hover_tip_mach_number,
                                 sea_level_weight_coefficient=sea_level_weight_coefficient, blade_number=blade_number,
                                 save_name=save_prefix + 'altitude_trend_tas_SVGS.pdf', window_name='Support Vector',
                                 labels=labels)
    classification_altitude_plot(RFclassifier, scaler, tip_speeds=tip_speeds, airspeed_type='cas',
                                 effective_flat_plate_drag=effective_flat_plate_drag, airspeed_range=speed_range,
                                 climb_rate_range=climb_rate_range,
                                 sea_level_hover_tip_mach_number=hover_tip_mach_number,
                                 sea_level_weight_coefficient=sea_level_weight_coefficient, blade_number=blade_number,
                                 save_name=save_prefix + 'altitude_trend_cas_RF.pdf', window_name='Random Forest',
                                 labels=labels)
    classification_altitude_plot(RFclassifier, scaler, tip_speeds=tip_speeds, airspeed_type='tas',
                                 effective_flat_plate_drag=effective_flat_plate_drag, airspeed_range=speed_range,
                                 climb_rate_range=climb_rate_range,
                                 sea_level_hover_tip_mach_number=hover_tip_mach_number,
                                 sea_level_weight_coefficient=sea_level_weight_coefficient, blade_number=blade_number,
                                 save_name=save_prefix + 'altitude_trend_tas_RF.pdf', window_name='Random Forest',
                                 labels=labels)
    classification_altitude_plot(NNclassifier, scaler, tip_speeds=tip_speeds, airspeed_type='cas',
                                 effective_flat_plate_drag=effective_flat_plate_drag, airspeed_range=speed_range,
                                 climb_rate_range=climb_rate_range,
                                 sea_level_hover_tip_mach_number=hover_tip_mach_number,
                                 sea_level_weight_coefficient=sea_level_weight_coefficient, blade_number=blade_number,
                                 save_name=save_prefix + 'altitude_trend_cas_NN.pdf', window_name='Neural Network',
                                 labels=labels)
    classification_altitude_plot(NNclassifier, scaler, tip_speeds=tip_speeds, airspeed_type='tas',
                                 effective_flat_plate_drag=effective_flat_plate_drag, airspeed_range=speed_range,
                                 climb_rate_range=climb_rate_range,
                                 sea_level_hover_tip_mach_number=hover_tip_mach_number,
                                 sea_level_weight_coefficient=sea_level_weight_coefficient, blade_number=blade_number,
                                 save_name=save_prefix + 'altitude_trend_tas_NN.pdf', window_name='Neural Network',
                                 labels=labels)
