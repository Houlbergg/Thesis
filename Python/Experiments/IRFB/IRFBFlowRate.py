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

parsedfiles = []
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
        parsedfiles.append(parse)

# %%
for peis in parsedfiles:
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
#idx = list(map(lambda x: files.index(x), chosen_names)) Generalized form perhaps
chosen_parsed = parsedfiles[:-1]
chosen_masked = parse_masked[:-1]

ident = ['5mL/min', '10mL/min', '20mL/min', '50mL/min', '100mL/min']

# %% Raw Nyquist

unit_scale = ''
area = 2.3 ** 2
fig, ax = plt.subplots()
for i, exp in enumerate(chosen_parsed):
    imp = exp[0].get_impedances() * area
    ax.plot(imp.real, -imp.imag, label=ident[i])
    # ax = set_aspect_ratio(ax, chosen[0])
    funcs.set_equal_tickspace(ax, figure=fig)

    ax.grid(visible=True)
    ax.set_xlabel(f'$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$')
    ax.set_ylabel(f'$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$')
    ax.legend()
    ax.set_axisbelow('line')
    # plt.tight_layout()
    plt.gca().set_aspect('equal')
#remove_legend_duplicate()
#plt.legend(loc='best', bbox_to_anchor=(0., 0.5, 0.5, 0.5))
plt.savefig(os.path.join(fig_directory, 'FlowRate' + "_Nyquist_data_area.png"))
plt.show()
plt.close()

# %% Raw Nyquist without tail

unit_scale = ''
area = 2.3 ** 2
fig, ax = plt.subplots()
for i, exp in enumerate(chosen_masked):
    imp = exp[0].get_impedances() * area
    ax.plot(imp.real, -imp.imag, label=ident[i])
    # ax = set_aspect_ratio(ax, chosen[0])
    #ax.set_xlim((0.4, 1))
    #ax.set_ylim((0, 0.25))
    funcs.set_equal_tickspace(ax, figure=fig)

    ax.grid(visible=True)
    ax.set_xlabel(f'$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$')
    ax.set_ylabel(f'$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$')
    ax.legend()
    ax.set_axisbelow('line')
    # plt.tight_layout()
    plt.gca().set_aspect('equal')
#remove_legend_duplicate()
#plt.legend(loc='best', bbox_to_anchor=(0., 0.5, 0.5, 0.5))
plt.savefig(os.path.join(fig_directory, 'FlowRate_NoTail' + "_Nyquist_data_area.png"))
plt.show()
plt.close()

# %% Raw Nyquist Freq comp. @ 10mL/min
freq_list = [parse_masked[1], parse_masked[-1]]
freq_ident = ['$f_{min}$ = 25mHz', '$f_{min}$ = 10mHz']
unit_scale = ''
area = 2.3 ** 2
fig, ax = plt.subplots()
for i, exp in enumerate(freq_list):
    imp = exp.get_impedances() * area
    ax.plot(imp.real, -imp.imag, label=freq_ident[i])
    # ax = set_aspect_ratio(ax, chosen[0])
    #ax.set_xlim((0.4, 0.9))
    #ax.set_ylim((0, 0.25))
    funcs.set_equal_tickspace(ax, figure=fig)

    ax.grid(visible=True)
    ax.set_xlabel(f'$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$')
    ax.set_ylabel(f'$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$')
    ax.legend()
    ax.set_axisbelow('line')
    # plt.tight_layout()
    plt.gca().set_aspect('equal')
#remove_legend_duplicate()
#plt.legend(loc='best', bbox_to_anchor=(0., 0.5, 0.5, 0.5))
plt.savefig(os.path.join(fig_directory, 'fmin_comparison' + "_Nyquist_data_area.png"))
plt.show()
plt.close()

# %%
pyi_columns = ['f', 'z_re', 'z_im', 'mod', 'phz']
chosen_names = files
chosen_parsed = [x[0] for x in parsedfiles]

# %%
dummy = chosen_parsed[0].to_dataframe(columns=pyi_columns)
mask = dummy['z_im'] < -0
chosen_nonzero = dummy[dummy['z_im'] < -0]
chosen_masked.append(pyi.dataframe_to_data_sets(chosen_nonzero, path=files[4]))
chosen_masked = list(zip(*chosen_masked))[0]
# Construct Pandas df because bayes_drt package likes that
columns = ['Freq', 'Zreal', 'Zimag', 'Zmod', 'Zphs']
dFs = []
dFs_masked = []
for eis in chosen:
    dFs.append(eis.to_dataframe(columns=columns))
for eis in chosen_masked:
    dFs_masked.append(eis.to_dataframe(columns=columns))