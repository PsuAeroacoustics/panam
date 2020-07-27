#!/usr/bin/env python
import os.path

import matplotlib.pyplot as plt
import numpy as np

import machine_learning

directory_names = list(map(os.path.expanduser, ['~/Desktop/spheres/R-44', '~/Desktop/spheres/R-66',
                                                '~/Desktop/spheres/Be430', '~/Desktop/spheres/MD-902',
                                                '~/Desktop/spheres/Mi-8', '~/Desktop/spheres/AS350/Amedee',
                                                '~/Desktop/spheres/AS350/SaltonSea',
                                                '~/Desktop/spheres/AS350/Sweetwater',
                                                '~/Desktop/spheres/AS350/B3',
                                                '~/Desktop/spheres/EC130', '~/Desktop/spheres/Be407',
                                                '~/Desktop/spheres/Be206L3']))

single_case = False
if __name__ == '__main__':
    if single_case:
        ref_idx = 10
        name = os.path.basename(os.path.normpath(directory_names[ref_idx]))
        balance = 'smote'
        threshhold = 0.65
        machine_learning.train_data(directory_names, ref_idx=ref_idx, threshhold=threshhold, balance=balance,
                                    exclude_reference=False,
                                    save_prefix='%s_t%d_%s_' % (name, 100 * threshhold, balance))
        plt.show(block=True)
    else:
        balance_methods = ['ada', 'random-over', 'random-under', 'smote', 'cluster-majority', 'cluster-minority',
                           'cluster']
        balance_methods = ['smote']
        threshholds = np.linspace(.55, .75, 5)
        for ref_idx, directory_name in enumerate(directory_names):
            name = os.path.basename(os.path.normpath(directory_name))
            for threshhold in threshholds:
                for balance in balance_methods:
                    prefix = '%s_t%d_%s_noex_' % (name, 100 * threshhold, balance)
                    machine_learning.train_data(directory_names, ref_idx=ref_idx, threshhold=threshhold,
                                                balance=balance, exclude_reference=False,
                                                save_prefix=prefix, grid_search=True)
                    plt.close('all')
# Notes:
# 407 - pretty good!
# 430 - decent
# 206 - poor
# R66 - poor
# Mi-8 Bad--wrong AoA? Fix drag?
# EC130 - need to eliminate very steep conditions?
# R-44 - not great; random forest is close
# AS350B3 - close, but lower value points picked

# smote seems to do fairly well.
