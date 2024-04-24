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
directory, fig_directory, pkl_directory = statics.set_experiment(
    "IRFB\\240415-Felt-0p5M_FeCl2FeCl23-1M_HCl"
)
files = glob.glob("Data/*.mpt")

# %% Parsing with pyimpspec. REQUIRES exporting EC-Lab raw binarys (.mpr) as text (.mpt)
parses = []
for file in files:
    name = os.path.basename(file).split(".")[0]
    string = os.path.join("Parsed\\", name) + ".pkl"
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
    string_mask = os.path.join("Parsed\\", label + "notail.pkl")
    if not os.path.isfile(string_mask):
        eis_masked, mask = funcs.create_mask_notail(eis)
        print(eis_masked)
        parse_masked.append(eis_masked)
        utils.save_pickle(parse_masked, string_mask)
    else:
        pyi_pickle_masked = utils.load_pickle(string_mask)
        parse_masked.append(pyi_pickle_masked)
# %% Selecting the wanted cycles
# Comparing Flow rates with nothing else changed. First [0] cycle each file. Exclude 10mHz at first [-1]

chosen_names = files[:-1]
# idx = list(map(lambda x: files.index(x), chosen_names)) Generalized form perhaps
chosen_parses = parses[:-1]
chosen_parses = [x[0] for x in chosen_parses]
chosen_masked = parse_masked[:-1]

ident = ["5mL/min", "10mL/min", "20mL/min", "50mL/min", "100mL/min"]

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
    # plt.tight_layout()
    plt.gca().set_aspect("equal")
# remove_legend_duplicate()
# plt.legend(loc='best', bbox_to_anchor=(0., 0.5, 0.5, 0.5))
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
    funcs.set_equal_tickspace(ax, figure=fig)

    ax.grid(visible=True)
    ax.set_xlabel(f"$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$")
    ax.set_ylabel(
        f"$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$"
    )
    ax.legend()
    ax.set_axisbelow("line")
    # plt.tight_layout()
    plt.gca().set_aspect("equal")
# remove_legend_duplicate()
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
    funcs.set_equal_tickspace(ax, figure=fig)

    ax.grid(visible=True)
    ax.set_xlabel(f"$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$")
    ax.set_ylabel(
        f"$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$"
    )
    ax.legend()
    ax.set_axisbelow("line")
    # plt.tight_layout()
    plt.gca().set_aspect("equal")
# remove_legend_duplicate()
# plt.legend(loc='best', bbox_to_anchor=(0., 0.5, 0.5, 0.5))
plt.savefig(os.path.join(fig_directory, "fmin_comparison" + "_Nyquist_data_area.png"))
plt.show()
plt.close()
