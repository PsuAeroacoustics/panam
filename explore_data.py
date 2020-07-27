import matplotlib.font_manager as fm
from matplotlib.animation import FuncAnimation
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
from os.path import expanduser
from sklearn.svm import SVC

import flight_acoustics
import machine_learning

# Configuration
explore_sampling = True
sampling_ref = 3
explore_fried_eggs = True
explore_training_data = False
show_explore = False

threshhold = 0.65
duration_correction = 60

font_dirs = ['/Library/Fonts/', ]
font_files = fm.findSystemFonts(fontpaths=font_dirs)
font_list = fm.createFontList(font_files)
fm.fontManager.ttflist.extend(font_list)

font = {'family': 'Arial',
        'weight': 'normal',
        'size': 10}

matplotlib.rc('font', **font)
style.use('fivethirtyeight')
matplotlib.rcParams.update({'mathtext.fontset': 'dejavuserif'})

directory_names = list(map(expanduser, ['~/Desktop/spheres/R-44', '~/Desktop/spheres/R-66',
                                        '~/Desktop/spheres/Be430', '~/Desktop/spheres/MD-902',
                                        '~/Desktop/spheres/Mi-8', '~/Desktop/spheres/AS350/Amedee',
                                        '~/Desktop/spheres/AS350/SaltonSea',
                                        '~/Desktop/spheres/AS350/Sweetwater',
                                        '~/Desktop/spheres/AS350/B3',
                                        '~/Desktop/spheres/EC130', '~/Desktop/spheres/Be407',
                                        '~/Desktop/spheres/Be206L3']))

if explore_sampling:
    (advance_ratio_sets, weight_coefficient_sets, alpha_sets, hover_tip_mach_number_sets, blade_number_sets,
     Lmax_sets, Lmean_sets, effective_flat_plate_drag_areas,
     tip_speeds, flight_path_angle_sets) = machine_learning.training_data([directory_names[sampling_ref]], altitude=500,
                                                                          cutoff=30, input_frequencies=None,
                                                                          fpa_climb_cutoff=5,
                                                                          duration_correction=duration_correction)
    L_sets = Lmean_sets
    cull_noisy_fpa = -1.0
    quiet_alpha_buffer = 6
    quiet_advance_ratio = None
    (balanced_conditions_matrix,
     balanced_classifications) = machine_learning.package_data(advance_ratio_sets, weight_coefficient_sets, alpha_sets,
                                                               hover_tip_mach_number_sets, blade_number_sets, L_sets,
                                                               flight_path_angle_sets, threshhold, cull_noisy_fpa,
                                                               quiet_alpha_buffer=None,
                                                               quiet_advance_ratio=quiet_advance_ratio, balance='smote')
    (conditions_matrix,
     classifications) = machine_learning.package_data(advance_ratio_sets, weight_coefficient_sets, alpha_sets,
                                                      hover_tip_mach_number_sets, blade_number_sets, L_sets,
                                                      flight_path_angle_sets, threshhold, cull_noisy_fpa,
                                                      quiet_alpha_buffer=None, quiet_advance_ratio=None,
                                                      balance='none')
    (augmented_conditions_matrix,
     augmented_classifications) = machine_learning.package_data(advance_ratio_sets, weight_coefficient_sets, alpha_sets,
                                                                hover_tip_mach_number_sets, blade_number_sets, L_sets,
                                                                flight_path_angle_sets, threshhold, cull_noisy_fpa,
                                                                quiet_alpha_buffer=quiet_alpha_buffer,
                                                                quiet_advance_ratio=None, balance='none')
    # Define classification masks
    noisy = classifications.astype(bool)
    not_noisy = np.logical_not(noisy)
    balanced_noisy = balanced_classifications.astype(bool)
    balanced_not_noisy = np.logical_not(balanced_noisy)

    # Initialize plot
    fig, ax = subplots()
    augmented = ax.plot(augmented_conditions_matrix[:, 0], augmented_conditions_matrix[:, 1], 'o', color='green',
                        alpha=.8, markeredgecolor='k', markersize=10)
    quiet = ax.plot(balanced_conditions_matrix[balanced_not_noisy, 0],
                    balanced_conditions_matrix[balanced_not_noisy, 1], 'o', color='black', alpha=.8,
                    markeredgecolor='k', markersize=10)
    balanced = ax.plot(balanced_conditions_matrix[balanced_noisy, 0], balanced_conditions_matrix[balanced_noisy, 1],
                       'o', color='blue', alpha=.8, markeredgecolor='k', markersize=10)
    unbalanced = ax.plot(conditions_matrix[noisy, 0], conditions_matrix[noisy, 1], 'o', color='red', alpha=.8,
                         markeredgecolor='k', markersize=10)
    ax.set_xlabel('Advance Ratio - $\mu$')
    ax.set_ylabel(r'Angle of Attack - $\alpha$' + ', °')
    ax.set_xlim([0.05, 0.35])
    ax.set_ylim([-10, 20])
    ax.tick_params(axis='x', which='major', pad=15)
    ax.legend((quiet[0], unbalanced[0], augmented[0], balanced[0]),
              ('Not-Noisy', 'Noisy', 'Augmented Not-Noisy', 'Balanced Noisy'),
              loc=9, bbox_to_anchor=(0.5, -0.25), shadow=True, ncol=2)
    if show_explore:
        show(block=True)
    else:
        savefig('balancing.pdf')

    classifier = SVC(C=10000)
    classifier.fit(balanced_conditions_matrix, balanced_classifications)
    z = classifier.decision_function(conditions_matrix)
    fig = figure()
    fig.set_tight_layout(False)
    ax = Axes3D(fig)

    ax.plot3D(conditions_matrix[not_noisy, 0], conditions_matrix[not_noisy, 1], z[not_noisy], 'o',
              color='black', markeredgecolor='white', alpha=.8, label='Not "Noisy"')
    ax.plot3D(conditions_matrix[noisy, 0], conditions_matrix[noisy, 1], z[noisy], 'o',
              color='red', markeredgecolor='k', alpha=.8, label='"Noisy"')

    ax.view_init(elev=15., azim=-60)
    ax.set_xlabel(r'Advance Ratio - $\mu$', labelpad=20.0, fontsize=14)
    ax.set_ylabel(r'Angle of Attack - $\alpha$' + ', °', labelpad=20.0, fontsize=14)
    ax.set_zlabel(r'Kernel function, $k(\mu,\alpha)$', labelpad=20.0, fontsize=14)
    x_ticks = np.linspace(0.05, 0.35, 5)
    y_ticks = np.linspace(-5, 15, 5)
    z_ticks = np.linspace(-40, 0, 5)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_zticks(z_ticks)

    xx, yy = np.meshgrid(x_ticks, y_ticks)
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=.35, color='blue')
    ax.dist = 12

    if show_explore:
        show(block=True)
    else:
        savefig('kernel_trick.pdf')

    classifier = SVC(C=10, kernel='linear')
    classifier.fit(balanced_conditions_matrix[:, 0:2], balanced_classifications)
    x = np.linspace(0.05, 0.35, 200)
    y = np.linspace(-20, 20, 200)
    xx, yy = np.meshgrid(xx, yy)
    w = np.column_stack((xx.flatten(), yy.flatten()))
    zz = np.reshape(classifier.decision_function(w), xx.shape)
    fig, ax = subplots(facecolor='white')

    ax.set_xlabel('Advance Ratio - $\mu$')
    ax.set_ylabel(r'Angle of Attack - $\alpha$' + ', °')
    ax.set_xlim([0.05, 0.35])
    ax.set_ylim([-5, 15])
    ax.tick_params(axis='x', which='major', pad=15)
    ax.plot(balanced_conditions_matrix[balanced_not_noisy, 0],
            balanced_conditions_matrix[balanced_not_noisy, 1], 'o', color='black', alpha=.8,
            markeredgecolor='w', markersize=10, label='Not "Noisy"')
    ax.plot(conditions_matrix[noisy, 0], conditions_matrix[noisy, 1], 'o', color='red', alpha=.8,
            markeredgecolor='k', markersize=10, label='"Noisy"')
    ax.contour(xx, yy, zz, colors=['blue'], linestyles=['-'], levels=[0])

    if show_explore:
        show(block=True)
    else:
        savefig('linear_kernel.pdf')

if explore_fried_eggs:
    flight_acoustics.plot_fried_eggs(directory_names, climb_rates=True, duration_correction=duration_correction,
                                     threshhold=threshhold, cull_noisy_fpa=-1.0, save_figures=True,
                                     fpa_climb_cutoff=5.0, ylim=(-2000, 750))

if explore_training_data:
    (advance_ratio_sets, weight_coefficient_sets, alpha_sets, hover_tip_mach_number_sets, blade_number_sets,
     Lmax_sets, Lmean_sets, effective_flat_plate_drag_areas,
     tip_speeds, flight_path_angle_sets) = machine_learning.training_data(directory_names, altitude=500,
                                                                          cutoff=30, input_frequencies=None,
                                                                          fpa_climb_cutoff=5,
                                                                          duration_correction=duration_correction)
    L_sets = Lmean_sets

    cull_noisy_fpa = -1.0
    conditions_matrix, classifications = machine_learning.package_data(advance_ratio_sets,
                                                                       weight_coefficient_sets, alpha_sets,
                                                                       hover_tip_mach_number_sets,
                                                                       blade_number_sets, L_sets,
                                                                       flight_path_angle_sets, threshhold,
                                                                       cull_noisy_fpa, quiet_alpha_buffer=None,
                                                                       quiet_advance_ratio=None)
    # Define classification masks
    noisy = classifications.astype(bool)
    not_noisy = np.logical_not(noisy)

    # Initialize 3D plot
    fig = figure()
    fig.set_tight_layout(False)
    ax = Axes3D(fig)

    ax.plot3D(conditions_matrix[not_noisy, 0], conditions_matrix[not_noisy, 1], conditions_matrix[not_noisy, 2], 'o',
              color='black')
    ax.plot3D(conditions_matrix[noisy, 0], conditions_matrix[noisy, 1], conditions_matrix[noisy, 2], 'o', color='red')

    ax.set_xlabel(r'$\mu$', labelpad=20.0, fontsize=16)
    ax.set_ylabel(r'$\alpha$', labelpad=20.0, fontsize=16)
    ax.set_zlabel(r'$C_W$', labelpad=20.0, fontsize=16)

    x_ticks = np.linspace(0.05, 0.35, 3)
    y_ticks = np.linspace(-10, 20, 4)
    z_ticks = np.linspace(.002, .006, 3)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_zticks(z_ticks)

    ax.set_xticklabels(map(str, x_ticks), fontsize=6)
    ax.set_yticklabels(map(str, y_ticks), fontsize=6)
    ax.set_zticklabels(map(str, z_ticks), fontsize=6)

    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)

    frames = 360 * 10


    # Rotate
    def update(i):
        ax.view_init(elev=15., azim=i * 360 / frames)
        ax.dist = 12
        return ax,


    animation = FuncAnimation(fig, update, frames=frames, interval=20, blit=False)
    if show_explore:
        show(block=True)
    else:
        animation.save('explore.mp4', dpi=200)
