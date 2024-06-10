"""
@author: Houlberg
"""
import contextlib
import warnings
import glob
import os
import copy

import funcs
import statics
import mpl_settings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyimpspec as pyi

import bayes_drt2.file_load as fl
from bayes_drt2.inversion import Inverter

warnings.showwarning = funcs.FilteredStream().write_warning
pd.options.mode.chained_assignment = None  # default='warn'

# %% Set folders
directory, fig_directory, pkl_directory, parse_directory = statics.set_experiment(
    # "IRFB\\240503-Felt-1p5M_FeCl2FeCl3-1M_HCl"
    "IRFB\\240507-Cloth-1p5M_FeCl2FeCl3-1M_HCl"
    # "IRFB\\240515-Paper-1p5M_FeCl2FeCl3-1M_HCl"
)
# exp = "1.5M Felt"
exp = "1.5M Cloth"
# exp = "1.5M Paper"
files = glob.glob("Data/*.mpt")

# %% Parsing with pyimpspec. REQUIRES exporting EC-Lab raw binaries (.mpr) as text (.mpt)
parses = []
for file in files:
    name = os.path.basename(file).split(".")[0]
    string = os.path.join(parse_directory, name) + ".pkl"
    if not os.path.isfile(string):
        try:
            parse = pyi.parse_data(file, file_format=".mpt")
            funcs.save_pickle(parse, string)
            parses.append(parse)
        except AssertionError as e:
            print("Pyimpspec parse of file: " + str(file) + " failed due to" + str(e))
    else:
        parses.append(funcs.load_pickle(string))

# %%
parses_masked = []
for peis in parses:
    label = peis[0].get_label().split("C15")[0]
    string_mask = os.path.join(parse_directory, label + "_masked.pkl")
    if not os.path.isfile(string_mask):
        parse_masked = []
        for eis in peis:
            eis_masked, mask = funcs.create_mask_notail(eis)
            parse_masked.append(eis_masked)
        funcs.save_pickle(parse_masked, string_mask)
        parses_masked.append(parse_masked)
    else:
        pyi_pickle_masked = funcs.load_pickle(string_mask)
        parses_masked.append(pyi_pickle_masked)

# %% Visual Inspection of the repeated cycles
# First 2 elements of parses are warmup cycles
fig_directory_nyquist = os.path.join(fig_directory, "Nyquist")
if not os.path.isdir(fig_directory_nyquist):
    os.mkdir(fig_directory_nyquist)
vel_label = ["v=0.5cms", "v=1cms", "v=2cms", "v=5cms"]
cycle_label = ["Cycle 1", "Cycle 2", "Cycle 3"]
color_dict = mpl_settings.bright_dict
no_fill_markers = mpl_settings.no_fill_markers
for i, vel in enumerate(parses[-4:]):
    fig, ax = plt.subplots()
    plt.suptitle(vel_label[i])
    for j, eis in enumerate(vel):
        imp = eis.get_impedances()
        label = cycle_label[j]
        unit_scale = ""  # If you want to swap to different scale. milli, kilo etc
        unit = f"\ \mathrm{{{unit_scale}}}\Omega$"
        ax.plot(
            imp.real,
            -imp.imag,
            label=label,
            marker=no_fill_markers[j],
            markersize=5,
            markerfacecolor="w",
            alpha=1,
            zorder=2.5,
        )
    ax.grid(visible=True)
    ax.set_axisbelow("line")
    plt.gca().set_aspect("equal")
    funcs.set_equal_tickspace(ax, figure=fig)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            fig_directory_nyquist,
            f"{vel_label[i]}.png",
        )
    )
    plt.show()
    plt.close()
# %% Selecting the wanted cycles
# Use the last cycle under assumption that the system has "settled"

chosen_names = files[-4:]
# idx = list(map(lambda x: files.index(x), chosen_names)) Generalized form perhaps
chosen_parses = parses[-4:]
chosen_parses = [x[-1] for x in chosen_parses]
chosen_masked = parses_masked[-4:]
chosen_masked = [x[-1] for x in chosen_masked]

ident = ["v=0.5cms", "v=1cms", "v=2cms", "v=5cms"]
pyi_columns = ["f", "z_re", "z_im", "mod", "phz"]

# %% Lin - KK ala pyimpspec
mu_criterion = 0.85
explorations = []
explorations_masked = []
string = pkl_directory + "\\LinKK"
if not os.path.isdir(string):
    os.mkdir(string)

string_exp = string + "\\ExploratoryTests.pkl"
string_exp_masked = string + "\\ExploratoryTests_masked.pkl"
if not os.path.isfile(string_exp):
    for eis in chosen_parses:
        tests = pyi.perform_exploratory_tests(eis, mu_criterion=mu_criterion)
        explorations.append(tests)
    funcs.save_pickle(explorations, string_exp)
else:
    explorations = funcs.load_pickle(string_exp)

if not os.path.isfile(string_exp_masked):
    for eis in chosen_masked:
        tests = pyi.perform_exploratory_tests(eis, mu_criterion=mu_criterion)
        explorations_masked.append(tests)
    funcs.save_pickle(explorations_masked, string_exp_masked)
else:
    explorations_masked = funcs.load_pickle(string_exp_masked)

# %% Plot Lin - KK explorations
fig_directory_linkk = os.path.join(fig_directory, "LinKK")
if not os.path.isdir(fig_directory_linkk):
    os.mkdir(fig_directory_linkk)

plot_explorations = True
if plot_explorations is True:
    for i, tests in enumerate(explorations):
        fig, ax = pyi.mpl.plot_mu_xps(tests, mu_criterion=mu_criterion)
        plt.suptitle(f"{ident[i]}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                fig_directory_linkk,
                f"{ident[i]}-ExploratoryTests.png",
            )
        )
        plt.show()
        plt.close()
    for i, tests in enumerate(explorations_masked):
        fig, ax = pyi.mpl.plot_mu_xps(tests, mu_criterion=mu_criterion)
        plt.suptitle(f"{ident[i]}-masked")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                fig_directory_linkk,
                f"{ident[i]}-masked-ExploratoryTests.png",
            )
        )
        plt.show()
        plt.close()

# %% Residual plot
# I personally prefer freq going high to low from left to right ("similar" as nyquist)
for i, tests in enumerate(explorations):
    fig, ax = pyi.mpl.plot_residuals(tests[0])
    plt.gca().invert_xaxis()
    # plt.suptitle(f"{ident[i]}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            fig_directory_linkk,
            f"{ident[i]}-Residuals.png",
        )
    )
    plt.show()
    plt.close()

for i, tests in enumerate(explorations_masked):
    fig, ax = pyi.mpl.plot_residuals(tests[0])
    plt.gca().invert_xaxis()
    # plt.suptitle(f"{ident[i]}-masked")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            fig_directory_linkk,
            f"{ident[i]}-masked-Residuals.png",
        )
    )
    plt.show()
    plt.close()

# %% Plotting Lin KK fitted EIS
color_dict = mpl_settings.bright_dict
no_fill_markers = mpl_settings.no_fill_markers
unit_scale = ""  # If you want to swap to different scale. milli, kilo etc
unit = f"\ \mathrm{{{unit_scale}}}\Omega$"
for i, tests in enumerate(explorations):
    fit = tests[0]
    fig, ax = plt.subplots()
    imp = chosen_parses[i].get_impedances()
    ax.scatter(
        imp.real,
        -imp.imag,
        label="Data",
        marker=no_fill_markers[0],
        linewidths=1,
        color=color_dict["black"],
        s=10,
        alpha=1,
        zorder=2.5,
    )
    imp_fit = fit.get_impedances()
    num_RC = fit.num_RC
    ax.plot(
        imp_fit.real,
        -imp_fit.imag,
        label="#(RC)={}".format(num_RC),
        marker="",
        color=color_dict["blue"],
    )
    funcs.set_equal_tickspace(ax, figure=fig)
    ax.grid(visible=True)
    ax.set_xlabel(f"$Z^\prime \ /" + unit)
    ax.set_ylabel(f"$-Z^{{\prime\prime}} \ /" + unit)
    ax.legend(
        fontsize=12,
        frameon=True,
        framealpha=1,
        fancybox=False,
        edgecolor="black",
    )
    ax.set_axisbelow("line")
    plt.suptitle(f"{ident[i]}")
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.show()
    plt.close()

# %%
color_dict = mpl_settings.bright_dict
no_fill_markers = mpl_settings.no_fill_markers
for i, tests in enumerate(explorations):
    eis = tests[0]
    fig, ax = plt.subplots()
    imp = chosen_masked[i].get_impedances()
    ax.scatter(imp.real, -imp.imag, label="Data", marker=no_fill_markers[0], color="b")
    fit = eis.get_impedances()
    num_RC = eis.num_RC
    ax.plot(
        fit.real,
        -fit.imag,
        label="#(RC)={}".format(num_RC),
        marker="",
        color=color_dict["blue"],
    )

    # ax = set_aspect_ratio(ax, peis)
    funcs.set_equal_tickspace(ax, figure=fig)
    ax.grid(visible=True)
    ax.set_xlabel(f"$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega$")
    ax.set_ylabel(f"$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega$")
    ax.legend()

    plt.tight_layout()
    plt.gca().set_aspect("equal")

    plt.savefig(
        os.path.join(fig_directory, str(ident[i]) + "_LinKKImpedanceFit_notail.png")
    )
    plt.show()
    plt.close()
