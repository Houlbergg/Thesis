# -*- coding: utf-8 -*-
"""
@author: Houlberg
"""
# %% Init
import glob
import os

import matplotlib.pyplot as plt
import scienceplots
from matplotlib import ticker

import mpl_settings

import numpy as np
import pandas as pd
import pyimpspec as pyi
from pyimpspec import DataSet
from bayes_drt2 import utils

import funcs
import statics

plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14

pd.options.mode.chained_assignment = None  # default='warn'

# %% Set folders
directory, fig_directory, pkl_directory = statics.set_experiment(
    "IRFB\\240415-Felt-0p5M_FeCl2FeCl23-1M_HCl"
)
files = glob.glob("240503-Felt-1p5M_FeCl2FeCl3-1M_HCl/Data/*.mpt")

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
        # pyi_pickle = pyi.DataSet.from_dict(pyi_pickle)
        parses.append(pyi_pickle)

# %%
parse_masked = []
for peis in parses:
    eis = peis[0]
    label = eis.get_label().split("C15")[0]
    string_mask = os.path.join(pkl_directory, label + "notail.pkl")
    if not os.path.isfile(string_mask):
        eis_masked, mask = funcs.create_mask_notail(eis)
        parse_masked.append(eis_masked)
        utils.save_pickle(parse_masked, string_mask)
    else:
        pyi_pickle_masked = utils.load_pickle(string_mask)
        parse_masked.append(pyi_pickle_masked)
# %% Selecting the wanted cycles
# Baichen was nice to us here, there is only per file

chosen_names = files
# idx = list(map(lambda x: files.index(x), chosen_names)) Generalized form perhaps
chosen_parses = parses
chosen_parses = [x[0] for x in chosen_parses]
chosen_masked = parse_masked

ident = [""]

# %% Nyquist

unit_scale = ""
area = 2.3**2
fig, ax = plt.subplots()
for i, exp in enumerate(chosen_parses):
    imp = exp.get_impedances() * area
    ax.plot(imp.real, -imp.imag, label=ident[i])
    # ax = set_aspect_ratio(ax, chosen[0])
    funcs.set_equal_tickspace(ax, figure=fig)

    ax.grid(visible=True)
    ax.set_xlabel(f"$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$")
    ax.set_ylabel(
        f"$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$"
    )
    ax.legend()
    ax.set_axisbelow("line")


# plt.legend(loc='best', bbox_to_anchor=(0., 0.5, 0.5, 0.5))
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.savefig(os.path.join(fig_directory, "FlowRate" + "_Nyquist_data_area.png"))
plt.show()
plt.close()

# %% Nyquist masked

unit_scale = ""
area = 2.3**2
fig, ax = plt.subplots()
for i, exp in enumerate(chosen_masked):
    imp = exp.get_impedances() * area
    ax.plot(imp.real, -imp.imag, label=ident[i])
    # ax = set_aspect_ratio(ax, chosen[0])
    # ax.set_xlim((0.4, 1))
    # ax.set_ylim((0, 0.25))
    funcs.set_equal_tickspace(ax, figure=fig, space="max")

    ax.set_aspect("equal")
    ax.grid(visible=True)
    ax.set_xlabel(f"$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$")
    ax.set_ylabel(
        f"$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$"
    )
    ax.legend()
    ax.set_axisbelow("line")


plt.tight_layout()
# plt.legend(loc='best', bbox_to_anchor=(0., 0.5, 0.5, 0.5))
plt.savefig(os.path.join(fig_directory, "FlowRate_NoTail" + "_Nyquist_data_area.png"))
plt.show()
plt.close()

# %% Nyquist Freq comp. @ 10mL/min masked
freq_list = [parse_masked[1], parse_masked[-1]]
freq_ident = ["$f_{min}$ = 25mHz", "$f_{min}$ = 10mHz"]
unit_scale = ""
area = 2.3**2
fig, ax = plt.subplots()
for i, exp in enumerate(freq_list):
    imp = exp.get_impedances() * area
    ax.plot(imp.real, -imp.imag, label=freq_ident[i])
    # ax = set_aspect_ratio(ax, chosen[0])
    # ax.set_xlim((0.4, 0.9))
    # ax.set_ylim((0, 0.25))
    funcs.set_equal_tickspace(ax, figure=fig, space=("min", 1.5))

    ax.grid(visible=True)
    ax.set_xlabel(f"$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$")
    ax.set_ylabel(
        f"$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$"
    )
    ax.legend()
    ax.set_axisbelow("line")

plt.tight_layout()
plt.gca().set_aspect("equal")
plt.savefig(os.path.join(fig_directory, "fmin_comparison" + "_Nyquist_data_area.png"))
plt.show()
plt.close()
