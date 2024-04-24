# -*- coding: utf-8 -*-
"""
@author: Houlberg
"""
# %% Init
import glob
import os

import matplotlib.pyplot as plt
import scienceplots
import mpl_settings

import numpy as np
import pandas as pd
import pyimpspec as pyi
from pyimpspec import DataSet
from bayes_drt2 import utils

import funcs
import statics

pd.options.mode.chained_assignment = None  # default='warn'

# %% Set folders
directory, fig_directory, pkl_directory = statics.set_experiment("IRFB\\240415-Felt-0p5M_FeCl2FeCl23-1M_HCl")
files = glob.glob('Data/*.mpt')

# %% Parsing with pyimpspec. REQUIRES exporting EC-Lab raw binarys (.mpr) as text (.mpt)
dummy_freq = np.logspace(6, -2, 81)
dummy_Z = np.ones_like(dummy_freq, dtype='complex128')
pyi_dummy = DataSet(dummy_freq, dummy_Z)

parses = []
parse_masked = []
for file in files:
    name = os.path.basename(file).split('.')[0]
    string = os.path.join('Parsed\\', name) + '.pkl'
    string_mask = os.path.join('Parsed\\', name + 'notail.pkl')
    if not os.path.isfile(string):
        try:
            parse = pyi.parse_data(file, file_format=".mpt")
            utils.save_pickle(parse, string)
        except:
            print("Pyimpspec could not parse file" + str(file))
    else:
        pyi_pickle = utils.load_pickle(string)
        parse = []
        for cycle in pyi_pickle:
            parse.append(pyi_dummy.from_dict(cycle))
        parses.append(parse)

# %%
for peis in parses:
    eis = peis[0]
    label = eis.get_label().split('C15')[0]
    string_mask = os.path.join('Parsed\\', label + 'notail.pkl')
    if not os.path.isfile(string_mask):
        dataset_masked, mask = funcs.create_mask_notail(eis)
        parse_masked.append(dataset_masked)
        utils.save_pickle(parse_masked, string_mask)
    else:
        pyi_pickle = utils.load_pickle(string_mask)
        parse_masked.append(pyi_pickle)

# %% Selecting the wanted cycles
# Comparing Flow rates with nothing else changed. First [0] cycle each file. Exclude 10mHz at first [-1]

chosen_names = files[:-1]
# idx = list(map(lambda x: files.index(x), chosen_names)) Generalized form perhaps
chosen_parsed = parses[:-1]
chosen_parsed = [x[0] for x in chosen_parsed]
chosen_masked = parse_masked[:-1]

ident = ['5mL/min', '10mL/min', '20mL/min', '50mL/min', '100mL/min']
pyi_columns = ['f', 'z_re', 'z_im', 'mod', 'phz']

# %% Lin - KK ala pyimpspec
mu_criterion = 0.85
explorations = []
if not os.path.isfile(os.path.join(pkl_directory, 'LinKKTests_notail.pkl')):
    for i, eis in enumerate(chosen_masked):
        # os.remove(os.path.join(pkl_directory, 'LinKKTests.pkl'))  # Remove pickle to rerun
        tests = pyi.perform_exploratory_tests(eis, mu_criterion=mu_criterion)
        explorations.append(tests)
    utils.save_pickle(explorations, os.path.join(pkl_directory, 'LinKKTests_notail.pkl'))
else:
    explorations = utils.load_pickle(os.path.join(pkl_directory, 'LinKKTests_notail.pkl'))
# %% Plot Lin - KK explorations
plot_explorations = True
if plot_explorations is True:
    for i, tests in enumerate(explorations):
        fig, ax = pyi.mpl.plot_mu_xps(tests, mu_criterion=mu_criterion)
        plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_ExploratoryTests_notail.png"))
        plt.show()
        plt.close()

# %% Residual plot
# I personally prefer freq going high to low from left to right ("similar" as nyquist)
for i, tests in enumerate(explorations):
    fig, ax = pyi.mpl.plot_residuals(tests[0])
    plt.gca().invert_xaxis()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_LinKKResiduals_notail.png"))
    plt.show()
    plt.close()

# %% Plotting Lin KK fitted EIS
unit_scale = ''  # If you want to swap to different scale. milli, kilo etc

for i, tests in enumerate(explorations):
    eis = tests[0]
    fig, ax = plt.subplots()
    imp = chosen_masked[i].get_impedances()
    ax.scatter(imp.real, -imp.imag, label='Data')
    fit = eis.get_impedances()
    num_RC = eis.num_RC
    ax.plot(fit.real, -fit.imag, label="#(RC)={}".format(num_RC))

    # ax = set_aspect_ratio(ax, peis)
    funcs.set_equal_tickspace(ax, figure=fig)
    ax.grid(visible=True)
    ax.set_xlabel(f'$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega$')
    ax.set_ylabel(f'$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega$')
    ax.legend()

    plt.tight_layout()
    plt.gca().set_aspect('equal')

    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_LinKKImpedanceFit_notail.png"))
    plt.show()
    plt.close()
