# -*- coding: utf-8 -*-
"""
@author: Houlberg
"""
# %% Init

import glob
import os
import matplotlib.pyplot as plt

import mpl_settings
import numpy as np
import pandas as pd
import pyimpspec as pyi
from pyimpspec import DataSet
from bayes_drt2 import utils

import funcs
import statics

pd.options.mode.chained_assignment = None  # default='warn'

plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# %% Set folders
directory, fig_directory, pkl_directory = statics.set_experiment('IRFB/FirstCell')
# %% Load Files and seperate by type.
# REQUIRES exporting EC-Lab raw binarys (.mpr) as text (.mpt)
files = glob.glob('Data/*.mpt')

# %% Parsing with pyimpspec. REQUIRES exporting EC-Lab raw binarys (.mpr) as text (.mpt)
parses = []
for file in files:
    name = os.path.basename(file).split(".")[0]
    string = os.path.join(pkl_directory, name) + ".pkl"
    if not os.path.isfile(string):
        try:
            parse = pyi.parse_data(file, file_format=".mpt")
            utils.save_pickle(parse, string)
            parses.append(parse)
        except:
            print("Pyimpspec could not parse file" + str(file))
    else:
        pyi_pickle = utils.load_pickle(string)
        parses.append(pyi_pickle)

# %%
parse_masked = []
for peis in parses:
    eis = peis[0]
    label = eis.get_label().split("C15")[0]
    string_mask = os.path.join(pkl_directory, label + "notail.pkl")
    if not os.path.isfile(string_mask):
        eis_masked, mask = funcs.create_mask_notail(eis)
        print(eis_masked)
        parse_masked.append(eis_masked)
        utils.save_pickle(parse_masked, string_mask)
    else:
        pyi_pickle_masked = utils.load_pickle(string_mask)
        parse_masked.append(pyi_pickle_masked)
# %% Selecting the wanted cycles
# Comparing Dampened and Un-dampened at various flow rate
# + Weird tail
#
chosen_names = files[7:9]
chosen_parsed = [parses[7][0], parses[8][0]]
first = chosen_parsed[0].get_impedances().real
second = chosen_parsed[1].get_impedances().real
chosen_diff = second[119] - first[119]
ident = ['Base', 'Dampened']

# %% Raw Nyquist

unit_scale = ''
area = 2.3 ** 2
fig, ax = plt.subplots()
for i, exp in enumerate(chosen_parsed):
    imp = exp.get_impedances() * area
    ax.plot(imp.real, -imp.imag, label=ident[i])
    # ax = set_aspect_ratio(ax, chosen[0])
    funcs.set_equal_tickspace(ax, figure=fig)

    ax.grid(visible=True)
    ax.set_xlabel(f'$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$')
    ax.set_ylabel(f'$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$')
    ax.legend()
    ax.set_axisbelow('line')
    plt.gca().set_aspect('equal')
#remove_legend_duplicate()
#plt.legend(loc='best', bbox_to_anchor=(0., 0.5, 0.5, 0.5))
plt.tight_layout()
plt.savefig(os.path.join(fig_directory, 'Dampening' + "_Nyquist_data_area.png"))
plt.show()
plt.close()

# %% Raw Nyquist zoom

unit_scale = ''
area = 2.3 ** 2
fig, ax = plt.subplots()
for i, exp in enumerate(chosen_parsed):
    imp = exp.get_impedances() * area
    ax.plot(imp.real, -imp.imag, label=ident[i])
    # ax = set_aspect_ratio(ax, chosen[0])
    ax.set_xlim((1.8, 2.3))
    ax.set_ylim((0, 0.3))
    funcs.set_equal_tickspace(ax, figure=fig)

    ax.grid(visible=True)
    ax.set_xlabel(f'$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$')
    ax.set_ylabel(f'$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$')
    ax.legend()
    ax.set_axisbelow('line')
    plt.tight_layout()
    plt.gca().set_aspect('equal')
#remove_legend_duplicate()
#plt.legend(loc='best', bbox_to_anchor=(0., 0.5, 0.5, 0.5))
plt.tight_layout()
plt.savefig(os.path.join(fig_directory, 'Dampening_zoom' + "_Nyquist_data_area.png"))
plt.show()
plt.close()

# %% Raw Nyquist shifted

unit_scale = ''
area = 2.3 ** 2
fig, ax = plt.subplots()
for i, exp in enumerate(chosen_parsed):
    imp = exp.get_impedances() * area
    ax.plot(imp.real - i * chosen_diff * area, -imp.imag, label=ident[i])
    # ax = set_aspect_ratio(ax, chosen[0])
    funcs.set_equal_tickspace(ax, figure=fig)

    ax.grid(visible=True)
    ax.set_xlabel(f'$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$')
    ax.set_ylabel(f'$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$')
    ax.legend()
    ax.set_axisbelow('line')
    plt.tight_layout()
    plt.gca().set_aspect('equal')
#remove_legend_duplicate()
#plt.legend(loc='best', bbox_to_anchor=(0., 0.5, 0.5, 0.5))
plt.savefig(os.path.join(fig_directory, 'Dampening_shifted' + "_Nyquist_data_area.png"))
plt.show()
plt.close()

# %% Raw Nyquist shifted zoom

unit_scale = ''
area = 2.3 ** 2
fig, ax = plt.subplots()
for i, exp in enumerate(chosen_parsed):
    imp = exp.get_impedances() * area
    ax.plot(imp.real - i * chosen_diff * area, -imp.imag, label=ident[i],markersize=3.5)
    # ax = set_aspect_ratio(ax, chosen[0])
    ax.set_xlim((1.8, 2.2))
    ax.set_ylim((0, 0.3))
    funcs.set_equal_tickspace(ax, figure=fig)

    ax.grid(visible=True)
    ax.set_xlabel(f'$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$')
    ax.set_ylabel(f'$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$')
    ax.legend()
    ax.set_axisbelow('line')
    plt.tight_layout()
    plt.gca().set_aspect('equal')
#remove_legend_duplicate()
#plt.legend(loc='best', bbox_to_anchor=(0., 0.5, 0.5, 0.5))
plt.savefig(os.path.join(fig_directory, 'Dampening_shifted_zoom' + "_Nyquist_data_area.png"))
plt.show()
plt.close()

# %% Raw Nyquist weird tail
chosen_names = files[7:11]
# chosen_parsed = [parses[7][0], parses[8][0], parses[9][0], parses[10][0]]
chosen_dummy = parses[7:11]
chosen_dummy2 = chosen_dummy[:]
chosen_parsed = [sublist[0] for sublist in parses[7:11]]

tailident = ['20mL/min', '20mL/min Dampened', '30mL/min Dampened', '40mL/min Dampened']
ident = ['Base', 'Dampened']

unit_scale = ''
area = 2.3 ** 2
fig, ax = plt.subplots()
for i, exp in enumerate(chosen_parsed):
    imp = exp.get_impedances() * area
    ax.plot(imp.real, -imp.imag, label=tailident[i], markevery=3, markersize=3)
    # ax = set_aspect_ratio(ax, chosen[0])
    funcs.set_equal_tickspace(ax, figure=fig)

    ax.grid(visible=True)
    ax.set_xlabel(f'$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$')
    ax.set_ylabel(f'$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$')
    ax.legend()
    ax.set_axisbelow('line')
    plt.tight_layout()
    plt.gca().set_aspect('equal')
#remove_legend_duplicate()
#plt.legend(loc='best', bbox_to_anchor=(0., 0.5, 0.5, 0.5))
plt.savefig(os.path.join(fig_directory, 'Dampening_WeirdTail' + "_Nyquist_data_area.png"))
plt.show()
plt.close()
