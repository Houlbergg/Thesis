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

plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

pd.options.mode.chained_assignment = None  # default='warn'

home = r"C:\Users\Houlberg\Documents\Thesis\Python\Experiments"
os.chdir(home)
# %% Set folders
felt = "IRFB\\Hackjob\\Felt\\240416-EIS-0p5M_FeCl2FeCl3-1M_HCl-20mLmin_Dampened-25mHz_10ppd_C15.mpt"

cloth1 = "IRFB\\Hackjob\\Cloth1\\240422-EIS-Cloth1-0p5M_FeCl2FeCl3-1M_HCl-20mLmin-25mHz_10ppd-Warmup_C15.mpt"

cloth2 = "IRFB\\Hackjob\\Cloth2\\240418-EIS-Cloth2-0p5M_FeCl2FeCl3-1M_HCl-20mLmin-25mHz_10ppd-Warmup_C15.mpt"
# %% Parsing with pyimpspec. REQUIRES exporting EC-Lab raw binarys (.mpr) as text (.mpt)
parse = pyi.parse_data(felt, file_format=".mpt")[-1]
parse1 = pyi.parse_data(cloth1, file_format=".mpt")[-1]
parse2 = pyi.parse_data(cloth2, file_format=".mpt")[-1]
parse_masked, mask = funcs.create_mask_notail(parse)
parse1_masked, mask1 = funcs.create_mask_notail(parse1)
parse2_masked, mask2 = funcs.create_mask_notail(parse2)

# %%
fig_directory = "figs"
if not os.path.exists(fig_directory):
    os.mkdir(fig_directory)
ident = ["Felt", "Cloth1", "Cloth2"]
parses = [parse, parse1, parse2]
parses_masked = [parse_masked, parse1_masked, parse2_masked]
# %% Nyquist

unit_scale = ""
area = 2.3**2
fig, ax = plt.subplots()
for i, exp in enumerate(parses_masked):
    imp = exp.get_impedances() * area
    ax.plot(imp.real, -imp.imag, label=ident[i])
    # ax = set_aspect_ratio(ax, chosen[0])
    funcs.set_equal_tickspace(ax, figure=fig, space='max')

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
    funcs.set_equal_tickspace(ax, figure=fig, space='ma')

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
    funcs.set_equal_tickspace(ax, figure=fig, space=('min', 1.5))

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
