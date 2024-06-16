# -*- coding: utf-8 -*-
"""
@author: Houlberg
"""
# %% Init
import glob
import os
import time
import copy

import matplotlib.pyplot as plt
import scienceplots
import mpl_settings

import numpy as np
import pandas as pd
import pyimpspec as pyi
from pyimpspec import DataSet

import bayes_drt2.file_load as fl
import bayes_drt2.utils as utils
import matplotlib.axes._secondary_axes
from bayes_drt2.inversion import Inverter

import funcs
import statics

pd.options.mode.chained_assignment = None  # default='warn'

# %% Set folders
directory, fig_directory, pkl_directory, parse_directory = statics.set_experiment(
    "IRFB\\240415-Felt-0p5M_FeCl2FeCl23-1M_HCl"
)
files = glob.glob("240503-Felt-1p5M_FeCl2FeCl3-1M_HCl/Data/*.mpt")

# %% Parsing with pyimpspec. REQUIRES exporting EC-Lab raw binarys (.mpr) as text (.mpt)
dummy_freq = np.logspace(6, -2, 81)
dummy_Z = np.ones_like(dummy_freq, dtype="complex128")
pyi_dummy = DataSet(dummy_freq, dummy_Z)

parses = []
parse_masked = []
for file in files:
    name = os.path.basename(file).split(".")[0]
    string = os.path.join("Parsed\\", name) + ".pkl"
    string_mask = os.path.join("Parsed\\", name + "notail.pkl")
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
    label = eis.get_label().split("C15")[0]
    string_mask = os.path.join("Parsed\\", label + "notail.pkl")
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

ident = ["5mL/min", "10mL/min", "20mL/min", "50mL/min", "100mL/min"]
pyi_columns = ["f", "z_re", "z_im", "mod", "phz"]

# %% Construct Pandas df because bayes_drt package likes that
columns = ["Freq", "Zreal", "Zimag", "Zmod", "Zphs"]
dFs = []
dFs_masked = []
for eis in chosen_parsed:
    dFs.append(eis.to_dataframe(columns=columns))
for eis in chosen_masked:
    dFs_masked.append(eis.to_dataframe(columns=columns))

# %% Bayes_DRT fitting
# By default, the Inverter class is configured to fit the DRT (rather than the DDT)
# Create separate Inverter instances for HMC and MAP fits
# Set the basis frequencies equal to the measurement frequencies
# (not necessary in general, but yields faster results here - see Tutorial 1 for more info on basis_freq)
drt_hmc_list = []
drt_map_list = []
drt_hmc_masked_list = []
drt_map_masked_list = []
ddt_hmc_list = []
ddt_map_list = []
ddt_hmc_masked_list = []
ddt_map_masked_list = []
inv_multi_hmc_list = []
inv_multi_map_list = []
inv_multi_hmc_masked_list = []
inv_multi_map_masked_list = []

for i, eis in enumerate(dFs):
    freq, Z = fl.get_fZ(eis)
    freq_mask, Z_mask = fl.get_fZ(dFs_masked[i])

    # DRT
    drt_hmc = Inverter(basis_freq=freq)
    drt_map = Inverter(basis_freq=freq)
    drt_hmc_masked = Inverter(basis_freq=freq_mask)
    drt_map_masked = Inverter(basis_freq=freq_mask)
    # DDT
    ddt_hmc = Inverter(
        distributions={
            "TP-DDT": {  # user-defined distribution name
                "kernel": "DDT",  # indicates that a DDT-type kernel should be used
                "dist_type": "parallel",
                # indicates that the diffusion paths are in parallel
                "symmetry": "planar",  # indicates the geometry of the system
                "bc": "transmissive",  # indicates the boundary condition
                "ct": False,  # indicates no simultaneous charge transfer
                "basis_freq": np.logspace(6, -3, 91),
            }
        },
        basis_freq=np.logspace(
            6, -3, 91
        ),  # use basis range large enough to capture full DDT
    )
    ddt_map = copy.deepcopy(ddt_hmc)
    ddt_hmc_masked = copy.deepcopy(ddt_hmc)
    ddt_map_masked = copy.deepcopy(ddt_hmc)
    # Multi distribution
    # for sp_dr: use xp_scale=0.8
    inv_multi_hmc = Inverter(
        distributions={
            "DRT": {"kernel": "DRT"},
            "TPP-DDT": {
                "kernel": "DDT",
                "symmetry": "planar",
                "bc": "transmissive",
                "dist_type": "parallel",
                "x_scale": 0.8,
                "ct": False,  # indicates no simultaneous charge transfer
                "basis_freq": np.logspace(6, -3, 91),
            },
        }
    )
    inv_multi_map = copy.deepcopy(inv_multi_hmc)
    inv_multi_hmc_masked = copy.deepcopy(inv_multi_hmc)
    inv_multi_map_masked = copy.deepcopy(inv_multi_hmc)

    # Perform HMC fit
    file_pickle = os.path.join(pkl_directory, "{0}_{1}.pkl".format(ident[i], drt_hmc))
    file_pickle_core = os.path.join(
        pkl_directory, "{0}_{1}_core.pkl".format(ident[i], drt_hmc)
    )

    drt_hmc_pickle = os.path.join(pkl_directory, str(ident[i]) + "_drt_hmc.pkl")
    if not os.path.isfile(drt_hmc_pickle):
        start = time.time()
        drt_hmc.fit(freq, Z, mode="sample")
        elapsed = time.time() - start
        print("HMC fit time {:.1f} s".format(elapsed))
        utils.save_pickle(drt_hmc, drt_hmc_pickle)
        drt_hmc.save_fit_data(drt_hmc_pickle + "_core", which="core")
        # Change to Inverter.save_fit_data(..., which= 'core') #load_pickle wont work
    else:
        drt_hmc.load_fit_data(drt_hmc_pickle)
        drt_hmc = utils.load_pickle(drt_hmc_pickle)

    # Perform HMC fit with initial tail removed
    if not os.path.isfile(
        os.path.join(pkl_directory, str(ident[i]) + "_drt_hmc_mask.pkl")
    ):
        start = time.time()
        drt_hmc_masked.fit(freq_mask, Z_mask, mode="sample")
        elapsed = time.time() - start
        print("HMC fit time {:.1f} s".format(elapsed))
        utils.save_pickle(
            drt_hmc_masked,
            os.path.join(pkl_directory, str(ident[i]) + "_drt_hmc_mask.pkl"),
        )
    else:
        drt_hmc_masked = utils.load_pickle(
            os.path.join(pkl_directory, str(ident[i]) + "_drt_hmc_mask.pkl")
        )

    #  Perform MAP fit
    if not os.path.isfile(os.path.join(pkl_directory, str(ident[i]) + "_drt_map.pkl")):
        start = time.time()
        drt_map.fit(freq, Z, mode="optimize")  # initialize from ridge solution
        elapsed = time.time() - start
        print("MAP fit time {:.1f} s".format(elapsed))
        utils.save_pickle(
            drt_map, os.path.join(pkl_directory, str(ident[i]) + "_drt_map.pkl")
        )
    else:
        drt_map = utils.load_pickle(
            os.path.join(pkl_directory, str(ident[i]) + "_drt_map.pkl")
        )

    #  Perform MAP fit with initial tail removed
    if not os.path.isfile(
        os.path.join(pkl_directory, str(ident[i]) + "_drt_map_mask.pkl")
    ):
        start = time.time()
        drt_map_masked.fit(freq_mask, Z_mask, mode="sample")
        elapsed = time.time() - start
        print("MAP fit time {:.1f} s".format(elapsed))
        utils.save_pickle(
            drt_map_masked,
            os.path.join(pkl_directory, str(ident[i]) + "_drt_map_mask.pkl"),
        )
    else:
        drt_map_masked = utils.load_pickle(
            os.path.join(pkl_directory, str(ident[i]) + "_drt_map_mask.pkl")
        )

    # Perform DDT HMC fit
    if not os.path.isfile(os.path.join(pkl_directory, str(ident[i]) + "_ddt_hmc.pkl")):
        start = time.time()
        ddt_hmc.fit(freq, Z, mode="sample")
        elapsed = time.time() - start
        print("HMC fit time {:.1f} s".format(elapsed))
        utils.save_pickle(
            ddt_hmc, os.path.join(pkl_directory, str(ident[i]) + "_ddt_hmc.pkl")
        )
    else:
        ddt_hmc = utils.load_pickle(
            os.path.join(pkl_directory, str(ident[i]) + "_ddt_hmc.pkl")
        )

    # Perform DDT HMC fit with initial tail removed
    if not os.path.isfile(
        os.path.join(pkl_directory, str(ident[i]) + "_ddt_hmc_mask.pkl")
    ):
        start = time.time()
        ddt_hmc_masked.fit(freq_mask, Z_mask, mode="sample")
        elapsed = time.time() - start
        print("HMC fit time {:.1f} s".format(elapsed))
        utils.save_pickle(
            ddt_hmc_masked,
            os.path.join(pkl_directory, str(ident[i]) + "_ddt_hmc_mask.pkl"),
        )
    else:
        ddt_hmc_masked = utils.load_pickle(
            os.path.join(pkl_directory, str(ident[i]) + "_ddt_hmc_mask.pkl")
        )

    #  Perform DDT MAP fit
    if not os.path.isfile(os.path.join(pkl_directory, str(ident[i]) + "_ddt_map.pkl")):
        start = time.time()
        ddt_map.fit(freq, Z, mode="optimize")  # initialize from ridge solution
        elapsed = time.time() - start
        print("MAP fit time {:.1f} s".format(elapsed))
        utils.save_pickle(
            ddt_map, os.path.join(pkl_directory, str(ident[i]) + "_ddt_map.pkl")
        )
    else:
        ddt_map = utils.load_pickle(
            os.path.join(pkl_directory, str(ident[i]) + "_ddt_map.pkl")
        )

    #  Perform DDT MAP fit with initial tail removed
    if not os.path.isfile(
        os.path.join(pkl_directory, str(ident[i]) + "_ddt_map_mask.pkl")
    ):
        start = time.time()
        ddt_map_masked.fit(freq_mask, Z_mask, mode="sample")
        elapsed = time.time() - start
        print("MAP fit time {:.1f} s".format(elapsed))
        utils.save_pickle(
            ddt_map_masked,
            os.path.join(pkl_directory, str(ident[i]) + "_ddt_map_mask.pkl"),
        )
    else:
        ddt_map_masked = utils.load_pickle(
            os.path.join(pkl_directory, str(ident[i]) + "_ddt_map_mask.pkl")
        )

    # Perform DRT+DDT HMC fit
    if not os.path.isfile(
        os.path.join(pkl_directory, str(ident[i]) + "_inv_multi_hmc.pkl")
    ):
        start = time.time()
        inv_multi_hmc.fit(freq, Z, mode="sample", nonneg=True)
        elapsed = time.time() - start
        print("HMC fit time {:.1f} s".format(elapsed))
        utils.save_pickle(
            inv_multi_hmc,
            os.path.join(pkl_directory, str(ident[i]) + "_inv_multi_hmc.pkl"),
        )
    else:
        inv_multi_hmc = utils.load_pickle(
            os.path.join(pkl_directory, str(ident[i]) + "_inv_multi_hmc.pkl")
        )

    #  Perform DRT+DDT MAP fit
    if not os.path.isfile(
        os.path.join(pkl_directory, str(ident[i]) + "_inv_multi_map.pkl")
    ):
        start = time.time()
        inv_multi_map.fit(
            freq, Z, mode="optimize", nonneg=True
        )  # initialize from ridge solution
        elapsed = time.time() - start
        print("MAP fit time {:.1f} s".format(elapsed))
        utils.save_pickle(
            inv_multi_map,
            os.path.join(pkl_directory, str(ident[i]) + "_inv_multi_map.pkl"),
        )
    else:
        inv_multi_map = utils.load_pickle(
            os.path.join(pkl_directory, str(ident[i]) + "_inv_multi_map.pkl")
        )

    # Perform DRT+DDT HMC fit with initial tail removed
    if not os.path.isfile(
        os.path.join(pkl_directory, str(ident[i]) + "_inv_multi_hmc_mask.pkl")
    ):
        start = time.time()
        inv_multi_hmc_masked.fit(freq_mask, Z_mask, mode="sample", nonneg=True)
        elapsed = time.time() - start
        print("HMC fit time {:.1f} s".format(elapsed))
        utils.save_pickle(
            inv_multi_hmc,
            os.path.join(pkl_directory, str(ident[i]) + "_inv_multi_hmc_mask.pkl"),
        )
    else:
        inv_multi_hmc = utils.load_pickle(
            os.path.join(pkl_directory, str(ident[i]) + "_inv_multi_hmc_mask.pkl")
        )

    #  Perform DRT+DDT MAP fit with initial tail removed
    if not os.path.isfile(
        os.path.join(pkl_directory, str(ident[i]) + "_inv_multi_map_mask.pkl")
    ):
        start = time.time()
        inv_multi_map_masked.fit(
            freq_mask, Z_mask, mode="optimize", nonneg=True
        )  # initialize from ridge solution
        elapsed = time.time() - start
        print("MAP fit time {:.1f} s".format(elapsed))
        utils.save_pickle(
            inv_multi_map_masked,
            os.path.join(pkl_directory, str(ident[i]) + "_inv_multi_map_mask.pkl"),
        )
    else:
        inv_multi_map_masked = utils.load_pickle(
            os.path.join(pkl_directory, str(ident[i]) + "_inv_multi_map_mask.pkl")
        )

    drt_hmc_list.append(drt_hmc)
    drt_map_list.append(drt_map)
    drt_hmc_masked_list.append(drt_hmc_masked)
    drt_map_masked_list.append(drt_map_masked)
    ddt_hmc_list.append(ddt_hmc)
    ddt_map_list.append(ddt_map)
    ddt_hmc_masked_list.append(ddt_hmc)
    ddt_map_masked_list.append(ddt_map)
    inv_multi_hmc_list.append(inv_multi_hmc)
    inv_multi_map_list.append(inv_multi_map)
    inv_multi_hmc_masked_list.append(inv_multi_hmc)
    inv_multi_map_masked_list.append(inv_multi_map)

# %% Visualize DRT and impedance fit
# plot impedance fit and recovered DRT

for i, exp in enumerate(dFs):
    fig, axes = plt.subplots()

    drt_hmc_list[i].plot_fit(
        axes=axes, plot_type="nyquist", color="k", label="HMC fit", data_label="Data"
    )
    drt_map_list[i].plot_fit(
        axes=axes, plot_type="nyquist", color="r", label="MAP fit", plot_data=False
    )
    # ax = set_aspect_ratio(ax, peis)
    # axes = set_equal_tickspace(axes, figure=fig)
    axes.grid(visible=True)
    axes.set_xlabel(f"$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega$")
    axes.set_ylabel(f"$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega$")
    axes.legend()
    # plt.tight_layout()
    plt.gca().set_aspect("equal")
    remove_legend_duplicate()
    plt.figure(fig)

    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT_FitImpedance.png"))
    plt.close()

    ddt_hmc_list[i].plot_fit(
        axes=axes, plot_type="nyquist", color="k", label="HMC fit", data_label="Data"
    )
    ddt_map_list[i].plot_fit(
        axes=axes, plot_type="nyquist", color="r", label="MAP fit", plot_data=False
    )
    # ax = set_aspect_ratio(ax, peis)
    # axes = set_equal_tickspace(axes, figure=fig)
    axes.grid(visible=True)
    axes.set_xlabel(f"$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega$")
    axes.set_ylabel(f"$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega$")
    axes.legend()
    # plt.tight_layout()
    plt.gca().set_aspect("equal")
    remove_legend_duplicate()
    plt.figure(fig)

    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DDT_FitImpedance.png"))
    plt.close()

    inv_multi_hmc_list[i].plot_fit(
        axes=axes, plot_type="nyquist", color="k", label="HMC fit", data_label="Data"
    )
    inv_multi_map_list[i].plot_fit(
        axes=axes, plot_type="nyquist", color="r", label="MAP fit", plot_data=False
    )
    # ax = set_aspect_ratio(ax, peis)
    # axes = set_equal_tickspace(axes, figure=fig)
    axes.grid(visible=True)
    axes.set_xlabel(f"$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega$")
    axes.set_ylabel(f"$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega$")
    axes.legend()
    # plt.tight_layout()
    plt.gca().set_aspect("equal")
    remove_legend_duplicate()
    plt.figure(fig)

    plt.savefig(
        os.path.join(fig_directory, str(ident[i]) + "_DRT-DDT_FitImpedance.png")
    )
    plt.close()

# %% Time Constant Distributions
for i, dist in enumerate(dFs):
    # tau_drt = drt_map_list[i].distributions['DRT']['tau']
    tau_drt = None
    fig, axes = plt.subplots()
    drt_hmc_list[i].plot_distribution(
        ax=axes, tau_plot=tau_drt, color="k", label="HMC mean", ci_label="HMC 95% CI"
    )
    drt_map_list[i].plot_distribution(ax=axes, tau_plot=tau_drt, color="r", label="MAP")
    sec_xaxis = [
        x
        for x in axes.get_children()
        if isinstance(x, matplotlib.axes._secondary_axes.SecondaryAxis)
    ][0]
    sec_xaxis.set_xlabel("$f$ / Hz")
    axes.legend()
    plt.figure(fig)

    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT.png"))
    plt.close()

    # tau_ddt = ddt_map_list[i].distributions['TP-DDT']['tau']
    tau_ddt = None
    fig, axes = plt.subplots()
    ddt_hmc_list[i].plot_distribution(
        ax=axes, tau_plot=tau_ddt, color="k", label="HMC mean", ci_label="HMC 95% CI"
    )
    ddt_map_list[i].plot_distribution(ax=axes, tau_plot=tau_ddt, color="r", label="MAP")
    sec_xaxis = [
        x
        for x in axes.get_children()
        if isinstance(x, matplotlib.axes._secondary_axes.SecondaryAxis)
    ][0]
    sec_xaxis.set_xlabel("$f$ / Hz")
    axes.legend()
    plt.figure(fig)

    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DDT.png"))
    plt.close()

    # tau_multi = inv_multi_map_list[i].distributions['DRT']['tau']
    tau_multi = None
    fig, axes = plt.subplots()
    inv_multi_hmc_list[i].plot_distribution(
        ax=axes, tau_plot=tau_multi, color="k", label="HMC mean", ci_label="HMC 95% CI"
    )
    inv_multi_map_list[i].plot_distribution(
        ax=axes, tau_plot=tau_multi, color="r", label="MAP"
    )
    sec_xaxis = [
        x
        for x in axes.get_children()
        if isinstance(x, matplotlib.axes._secondary_axes.SecondaryAxis)
    ][0]
    sec_xaxis.set_xlabel("$f$ / Hz")
    axes.legend()
    plt.figure(fig)

    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT-DDT.png"))
    plt.close()

    # tau_drt_mask = drt_map_masked_list[i].distributions['DRT']['tau']
    tau_drt_mask = None
    fig, axes = plt.subplots()
    drt_hmc_masked_list[i].plot_distribution(
        ax=axes,
        tau_plot=tau_drt_mask,
        color="k",
        label="HMC mean",
        ci_label="HMC 95% CI",
    )
    drt_map_masked_list[i].plot_distribution(
        ax=axes, tau_plot=tau_drt_mask, color="r", label="MAP"
    )
    sec_xaxis = [
        x
        for x in axes.get_children()
        if isinstance(x, matplotlib.axes._secondary_axes.SecondaryAxis)
    ][0]
    sec_xaxis.set_xlabel("$f$ / Hz")
    axes.legend()
    plt.figure(fig)

    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT_masked.png"))
    plt.close()

    # tau_ddt_mask = ddt_map_masked_list[i].distributions['TP-DDT']['tau']
    tau_ddt_mask = None
    fig, axes = plt.subplots()
    ddt_hmc_masked_list[i].plot_distribution(
        ax=axes,
        tau_plot=tau_ddt_mask,
        color="k",
        label="HMC mean",
        ci_label="HMC 95% CI",
    )
    ddt_map_masked_list[i].plot_distribution(
        ax=axes, tau_plot=tau_ddt_mask, color="r", label="MAP"
    )
    sec_xaxis = [
        x
        for x in axes.get_children()
        if isinstance(x, matplotlib.axes._secondary_axes.SecondaryAxis)
    ][0]
    sec_xaxis.set_xlabel("$f$ / Hz")
    axes.legend()
    plt.figure(fig)

    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DDT_masked.png"))
    plt.close()

    # tau_multi_mask = inv_multi_map_masked_list[i].distributions['DRT']['tau']
    tau_multi_mask = None
    fig, axes = plt.subplots()
    inv_multi_hmc_masked_list[i].plot_distribution(
        ax=axes,
        tau_plot=tau_multi_mask,
        color="k",
        label="HMC mean",
        ci_label="HMC 95% CI",
    )
    inv_multi_map_masked_list[i].plot_distribution(
        ax=axes, tau_plot=tau_multi_mask, color="r", label="MAP"
    )
    sec_xaxis = [
        x
        for x in axes.get_children()
        if isinstance(x, matplotlib.axes._secondary_axes.SecondaryAxis)
    ][0]
    sec_xaxis.set_xlabel("$f$ / Hz")
    axes.legend()
    plt.figure(fig)

    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT-DDT_masked.png"))
    plt.close()

    plt.show()

# %% Visualize the recovered error structure
# plot residuals and estimated error structure
for i, dist in enumerate(dFs):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    drt_hmc_list[i].plot_residuals(axes=axes)
    # fig.suptitle("Bayes Estimated Error Structure")
    plt.figure(fig)

    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT_HMC_Residuals.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    drt_hmc_masked_list[i].plot_residuals(axes=axes)
    plt.figure(fig)

    plt.savefig(
        os.path.join(fig_directory, str(ident[i]) + "_DRT_HMC_Residuals_masked.png")
    )
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    ddt_hmc_list[i].plot_residuals(axes=axes)
    # fig.suptitle("Bayes Estimated Error Structure")
    plt.figure(fig)

    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DDT_HMC_Residuals.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    ddt_hmc_masked_list[i].plot_residuals(axes=axes)
    plt.figure(fig)

    plt.savefig(
        os.path.join(fig_directory, str(ident[i]) + "_DDT_HMC_Residuals_masked.png")
    )
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    inv_multi_hmc_list[i].plot_residuals(axes=axes)
    # fig.suptitle("Bayes Estimated Error Structure")
    plt.figure(fig)

    plt.savefig(
        os.path.join(fig_directory, str(ident[i]) + "_DRT-DDT_HMC_Residuals.png")
    )
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    inv_multi_hmc_masked_list[i].plot_residuals(axes=axes)
    plt.figure(fig)

    plt.savefig(
        os.path.join(fig_directory, str(ident[i]) + "_DRT-DDT_HMC_Residuals_masked.png")
    )
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    drt_map_list[i].plot_residuals(axes=axes)
    # fig.suptitle("Bayes Estimated Error Structure")
    plt.figure(fig)

    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT_MAP_Residuals.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    drt_map_masked_list[i].plot_residuals(axes=axes)
    plt.figure(fig)

    plt.savefig(
        os.path.join(fig_directory, str(ident[i]) + "_DRT_MAP_Residuals_masked.png")
    )
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    ddt_map_list[i].plot_residuals(axes=axes)
    # fig.suptitle("Bayes Estimated Error Structure")
    plt.figure(fig)

    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DDT_MAP_Residuals.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    ddt_map_masked_list[i].plot_residuals(axes=axes)
    plt.figure(fig)

    plt.savefig(
        os.path.join(fig_directory, str(ident[i]) + "_DDT_MAP_Residuals_masked.png")
    )
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    inv_multi_map_list[i].plot_residuals(axes=axes)
    # fig.suptitle("Bayes Estimated Error Structure")
    plt.figure(fig)

    plt.savefig(
        os.path.join(fig_directory, str(ident[i]) + "_DRT-DDT_MAP_Residuals.png")
    )
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    inv_multi_map_masked_list[i].plot_residuals(axes=axes)
    plt.figure(fig)

    plt.savefig(
        os.path.join(fig_directory, str(ident[i]) + "_DRT-DDT_MAP_Residuals_masked.png")
    )
    plt.close()
# plot true error structure in miliohms. Cant with this data
# p = axes[0].plot(freq, 3*Zdf['sigma_re'] * 1000, ls='--')
# axes[0].plot(freq, -3*Zdf['sigma_re'] * 1000, ls='--', c=p[0].get_color())
# axes[1].plot(freq, 3*Zdf['sigma_im'] * 1000, ls='--')
# axes[1].plot(freq, -3*Zdf['sigma_im'] * 1000, ls='--', c=p[0].get_color(), label='True $\pm 3\sigma$')

# %% Peak fitting
# Only fit peaks that have a prominence of >= 5% of the estimated polarization resistance
for i, dist in enumerate(dFs):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    drt_map_list[i].fit_peaks(prom_rthresh=0.05)
    drt_map_list[i].plot_peak_fit(ax=axes[0])  # Plot the overall peak fit
    drt_map_list[i].plot_peak_fit(
        ax=axes[1], plot_individual_peaks=True
    )  # Plot the individual peaks
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.figure(fig)

    fig.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT_MAP_PeakFits.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    ddt_map_list[i].fit_peaks(prom_rthresh=0.05)
    ddt_map_list[i].plot_peak_fit(ax=axes[0])  # Plot the overall peak fit
    ddt_map_list[i].plot_peak_fit(
        ax=axes[1], plot_individual_peaks=True
    )  # Plot the individual peaks
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.figure(fig)

    fig.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DDT_MAP_PeakFits.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    inv_multi_map_list[i].fit_peaks(prom_rthresh=0.05)
    inv_multi_map_list[i].plot_peak_fit(ax=axes[0])  # Plot the overall peak fit
    inv_multi_map_list[i].plot_peak_fit(
        ax=axes[1], plot_individual_peaks=True
    )  # Plot the individual peaks
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.figure(fig)

    fig.tight_layout()
    plt.savefig(
        os.path.join(fig_directory, str(ident[i]) + "_DRT-DDT_MAP_PeakFits.png")
    )
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    try:
        drt_map_masked_list[i].fit_peaks(prom_rthresh=0.05)
        drt_map_masked_list[i].plot_peak_fit(ax=axes[0])  # Plot the overall peak fit
        drt_map_masked_list[i].plot_peak_fit(
            ax=axes[1], plot_individual_peaks=True
        )  # Plot the individual peaks
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
        plt.figure(fig)

        fig.tight_layout()
        plt.savefig(
            os.path.join(fig_directory, str(ident[i]) + "_DRT_MAP_PeakFits_masked.png")
        )
        plt.close()
    except:
        print("Error with file " + str(chosen[i]) + "For DRT Non-zero Peak Fitting")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    ddt_map_list[i].fit_peaks(prom_rthresh=0.05)
    ddt_map_list[i].plot_peak_fit(ax=axes[0])  # Plot the overall peak fit
    ddt_map_list[i].plot_peak_fit(
        ax=axes[1], plot_individual_peaks=True
    )  # Plot the individual peaks
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.figure(fig)

    fig.tight_layout()
    plt.savefig(
        os.path.join(fig_directory, str(ident[i]) + "_DDT_MAP_PeakFits_masked.png")
    )
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    inv_multi_map_list[i].fit_peaks(prom_rthresh=0.05)
    inv_multi_map_list[i].plot_peak_fit(ax=axes[0])  # Plot the overall peak fit
    inv_multi_map_list[i].plot_peak_fit(
        ax=axes[1], plot_individual_peaks=True
    )  # Plot the individual peaks
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.figure(fig)

    fig.tight_layout()
    plt.savefig(
        os.path.join(fig_directory, str(ident[i]) + "_DRT-DDT_MAP_PeakFits_masked.png")
    )
    plt.close()

# %% Re sort files

figures = glob.glob(os.path.join(fig_directory, "*"))

for iden in ident:
    if not os.path.isdir(os.path.join(fig_directory, str(iden))):
        os.makedirs(os.path.join(fig_directory, str(iden)))

for file in figures:
    name = os.path.basename(file)
    if "25mHz-20ppd" in name:
        shutil.move(file, os.path.join(fig_directory, "25mHz-20ppd"))
