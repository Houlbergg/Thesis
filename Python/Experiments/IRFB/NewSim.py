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
from scipy.signal import find_peaks

import bayes_drt2.file_load as fl
from bayes_drt2.inversion import Inverter

warnings.showwarning = funcs.FilteredStream().write_warning
pd.options.mode.chained_assignment = None  # default='warn'

# %% Set folders
directory, fig_directory, pkl_directory, parse_directory = statics.set_experiment(
    # "IRFB\\240503-Felt-1p5M_FeCl2FeCl3-1M_HCl"
    # "IRFB\\240507-Cloth-1p5M_FeCl2FeCl3-1M_HCl"
    # "IRFB\\240515-Paper-1p5M_FeCl2FeCl3-1M_HCl"
    "IRFB\\Sim"
)
# exp = "1.5M Felt"
# exp = "1.5M Cloth"
# exp = "1.5M Paper"
exp = "Sim Felt"
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
pyi.mpl.plot_nyquist(parses[0][0])
plt.show()
# %%
"""
label = parses[1][0].get_label().split("C15")[0]
string_ct = os.path.join(parse_directory, label + "_ct.pkl")
string_diffusion = os.path.join(parse_directory, label + "_diffusion.pkl")
if not os.path.isfile(string_ct):
    full_ct, ct_mask = funcs.create_mask_ct(parses[1][0])
    funcs.save_pickle(full_ct, string_ct)
else:
    full_ct = funcs.load_pickle(string_ct)
if not os.path.isfile(string_diffusion):
    full_diff, diff_mask = funcs.create_mask_diffusion(parses[1][0])
    funcs.save_pickle(full_diff, string_diffusion)
else:
    full_diff = funcs.load_pickle(string_diffusion)
"""
# %% Selecting the wanted cycles
# Just the one cycle for sim
chosen = parses[0]
# full = parses[0]
# ct = parses[1]
diff = parses[0][0]
"""
chosen_names = files[-4:]
# idx = list(map(lambda x: files.index(x), chosen_names)) Generalized form perhaps
"""
ident = ["Sim Felt Full"]
pyi_columns = ["f", "z_re", "z_im", "mod", "phz"]

# %% Construct Pandas df because bayes_drt package likes that
columns = ["Freq", "Zreal", "Zimag", "Zmod", "Zphs"]
dFs = []
for eis in chosen:
    dFs.append(eis.to_dataframe(columns=columns))

# %% Bayes_DRT fitting
# By default, the Inverter class is configured to fit the DRT (rather than the DDT)
# Create separate Inverter instances for HMC and MAP fits

with contextlib.redirect_stdout(funcs.FilteredStream(filtered_values=["f"])):
    # Initialize all the Inverters. DO NOT use these directly, create a deep copy instead.
    drt = Inverter()
    ddt = Inverter(
        distributions={
            "TP-DDT": {  # user-defined distribution name
                "kernel": "DDT",  # indicates that a DDT-type kernel should be used
                "dist_type": "parallel",
                "symmetry": "planar",  # indicates the geometry of the system
                "bc": "transmissive",  # indicates the boundary condition
            }
        }
    )
    # Multi distribution D "Multi" Times : dmt :)
    # for sp_dr: use xp_scale=0.8
    dmt = Inverter(
        distributions={
            "DRT": {"kernel": "DRT"},
            "TP-DDT": {
                "kernel": "DDT",
                "symmetry": "planar",
                "bc": "transmissive",
                "dist_type": "parallel",
                "x_scale": 0.8,
            },
        },
    )
    inverter_list = [drt, ddt, dmt]
dist_list_str = ["drt", "ddt", "dmt"]
fit_list = ["hmc", "map", "hmc_ct", "map_ct", "hmc_diffusion", "map_diffusion"]

for direc in dist_list_str:
    if not os.path.isdir(pkl_directory + "\\" + direc):
        os.mkdir(pkl_directory + "\\" + direc)
pkl_direcs = [pkl_directory + "\\" + x for x in dist_list_str]

# %% Fitting Sim
direc = pkl_directory + "\\" + "Diff"
if not os.path.isdir(direc):
    os.mkdir(direc)

dist_dict = {}
dist_keys = ["drt", "ddt", "dmt"]
fit_keys = ["hmc", "map"]
# low_vel = [dFs, dFs_ct, dFs_diffusion]

freq, Z = fl.get_fZ(dFs[0])
with contextlib.redirect_stdout(
    funcs.FilteredStream(filtered_values=["f"])
), warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="overflow encountered")
    inverters = [copy.deepcopy(x) for x in inverter_list]
    for j, inv in enumerate(inverters):
        dist = dist_keys[j]
        fit_dict = {}
        stringy = direc + "\\"

        fitted_list = []
        hmc = copy.deepcopy(inv)
        mAP = copy.deepcopy(inv)

        if len(inv.distributions) != 1:
            nonneg = True
        else:
            nonneg = False
        for k in fit_keys:
            path = stringy + "_" + dist + "_" + k + ".pkl"
            if k == "hmc":
                if os.path.isfile(path):
                    hmc.load_fit_data(path)
                else:
                    hmc.fit(freq, Z, mode="sample", nonneg=nonneg)
                    hmc.save_fit_data(path, which="core")
                fit_dict[k] = hmc

            elif k == "map":
                if os.path.isfile(path):
                    mAP.load_fit_data(path)
                else:
                    mAP.fit(freq, Z, mode="optimize", nonneg=nonneg)
                    mAP.save_fit_data(path, which="core")
                fit_dict[k] = mAP

            else:
                raise ValueError("Invalid fit type")
        dist_dict[dist] = fit_dict

# %% Visualize DRT and impedance fit v=0.5cms
# plot impedance fit and recovered DRT
fig_directory_fit = os.path.join(fig_directory, "FitTest")
if not os.path.isdir(fig_directory_fit):
    os.mkdir(fig_directory_fit)
unit_scale = ""
pruning = ["", " - ct", " - diffusion"]
area = None
if area:
    unit = f"\ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$"
else:
    unit = f"\ \mathrm{{{unit_scale}}}\Omega$"
color_dict = mpl_settings.bright_dict
no_fill_markers = mpl_settings.no_fill_markers
counter = 0
for dist in dist_dict:
    hmc = copy.deepcopy(dist_dict[dist]["hmc"])  # hmc
    mAP = copy.deepcopy(dist_dict[dist]["map"])  # map
    fig, axes = plt.subplots(figsize=(8, 5.5))
    with contextlib.redirect_stdout(
        funcs.FilteredStream(
            filtered_values=[
                "f",
            ]
        )
    ):
        hmc.plot_fit(
            axes=axes,
            area=area,
            plot_type="nyquist",
            label="HMC fit",
            color=color_dict["blue"],
            data_label="Data",
            data_kw={
                "marker": no_fill_markers[0],
                "linewidths": 1,
                "color": color_dict["black"],
                "s": 10,
                "alpha": 1,
                "zorder": 2.5,
            },
            marker="",
        )
        mAP.plot_fit(
            axes=axes,
            area=area,
            plot_type="nyquist",
            label="MAP fit",
            color=color_dict["red"],
            plot_data=False,
            marker="",
            linestyle="dashed",
            alpha=1,
        )

        funcs.set_equal_tickspace(axes, figure=fig)
        axes.grid(visible=True)
        axes.set_xlabel(f"$Z^\prime \ /" + unit)
        axes.set_ylabel(f"$-Z^{{\prime\prime}} \ /" + unit)
        axes.legend(
            fontsize=12,
            frameon=True,
            framealpha=1,
            fancybox=False,
            edgecolor="black",
        )
        axes.set_axisbelow("line")
        # plt.suptitle(f"{exp}{data}-{dist}{pruning[i]}")
        plt.gca().set_aspect("equal")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                fig_directory_fit,
                f"{dist}-FitImpedance.png",
            )
        )
        plt.show()
        plt.close()
counter += 1

# %% Time constant distributions v=0.5cms
fig_directory_time = os.path.join(fig_directory, "TimeConstantDistributions")
if not os.path.isdir(fig_directory_time):
    os.mkdir(fig_directory_time)
unit_scale = ""
area = None
color_dict = mpl_settings.bright_dict
no_fill_markers = mpl_settings.no_fill_markers
fill_markers = mpl_settings.filled_markers

tempo = [data_dict["ct"]["drt"], data_dict["diff"]["ddt"], data_dict["full"]["dmt"]]
tempi = ["ct", "diff", "full"]
counter = 0
for data in tempo:
    hmc = copy.deepcopy(data["hmc"])
    mAP = copy.deepcopy(data["map"])
    fig, axes = plt.subplots(figsize=(8, 5.5))
    with contextlib.redirect_stdout(funcs.FilteredStream(filtered_values=["f"])):
        tau = None
        hmc.plot_distribution(
            ax=axes,
            tau_plot=tau,
            color=color_dict["blue"],
            label="HMC mean",
            ci_label="HMC 95% CI",
            marker="",
        )
        mAP.plot_distribution(
            ax=axes, tau_plot=tau, color=color_dict["red"], label="MAP", marker=""
        )

        axes.legend(
            fontsize=12,
            frameon=True,
            framealpha=1,
            fancybox=False,
            edgecolor="black",
        )
        axes.set_axisbelow("line")
        plt.figure(fig)
        # plt.xticks()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                fig_directory_time,
                f"{tempi[counter]}-TimeConstantDistributions.png",
            )
        )
        plt.show()
        plt.close()
        counter += 1

# %% Peak fits 0.5cms
# Only fit peaks that have a prominence of >= 5% of the estimated polarization resistance
fig_directory_peak = os.path.join(fig_directory, "Peaks")
if not os.path.isdir(fig_directory_peak):
    os.mkdir(fig_directory_peak)

counter = 0
for data in tempo:
    fig, axes = plt.subplots(2, 1, figsize=(8, 9))
    mAP = copy.deepcopy(data["map"])  # map
    mAP.fit_peaks(prom_rthresh=0.05)
    mAP.plot_peak_fit(ax=axes[0], marker="")
    mAP.plot_peak_fit(ax=axes[1], plot_individual_peaks=True, marker="")
    axes[0].legend(
        fontsize=12,
        frameon=True,
        framealpha=1,
        fancybox=False,
        edgecolor="black",
    )
    axes[0].set_axisbelow("line")
    axes[1].legend(
        fontsize=12,
        frameon=True,
        framealpha=1,
        fancybox=False,
        edgecolor="black",
    )
    axes[1].set_axisbelow("line")
    plt.figure(fig)
    # plt.xticks()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            fig_directory_peak,
            f"{tempi[counter]}-PeakFits.png",
        )
    )
    plt.show()
    plt.close()
    counter += 1

# %% Peak fitting
fig_directory_peak = os.path.join(fig_directory, "PeakFits")
if not os.path.isdir(fig_directory_peak):
    os.mkdir(fig_directory_peak)
unit_scale = ""
area = None
color_dict = mpl_settings.bright_dict
no_fill_markers = mpl_settings.no_fill_markers
fill_markers = mpl_settings.filled_markers

for data in data_dict:
    for dist in data_dict[data]:
        for i in range(2):
            idx = i * 2
            hmc = copy.deepcopy(data_dict[data][dist][fit_keys[idx]])  # hmc
            mAP = copy.deepcopy(data_dict[data][dist][fit_keys[idx + 1]])  # map
            with contextlib.redirect_stdout(
                funcs.FilteredStream(filtered_values=["f"])
            ):
                fig, axes = plt.subplots(2, 1, figsize=(8, 9))
                hmc.fit_peaks(prom_rthresh=0.10)
                hmc.plot_peak_fit(ax=axes[0])
                hmc.plot_peak_fit(ax=axes[1], plot_individual_peaks=True)
                axes[1].legend(
                    fontsize=12,
                    frameon=True,
                    framealpha=1,
                    fancybox=False,
                    edgecolor="black",
                )
                plt.figure(fig)
                plt.suptitle(f"{name}{data}-{dist}-{fit_keys[idx]}{pruning[i]}")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        fig_directory_time,
                        f"{data}-{dist}{pruning[i]}-PeakFits.png",
                    )
                )
                plt.show()
                plt.close()

                fig, axes = plt.subplots(2, 1, figsize=(8, 9))
                mAP.fit_peaks(prom_rthresh=0.10)
                mAP.plot_peak_fit(ax=axes[0])
                mAP.plot_peak_fit(ax=axes[1], plot_individual_peaks=True)
                axes.legend(
                    fontsize=12,
                    frameon=True,
                    framealpha=1,
                    fancybox=False,
                    edgecolor="black",
                )
                plt.figure(fig)
                plt.suptitle(f"{name}{data}-{dist}{pruning[i]}")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        fig_directory_time,
                        f"{data}-{dist}{pruning[i]}-PeakFits.png",
                    )
                )
                plt.show()
                plt.close()
                hmc.plot_distribution(
                    ax=axes,
                    tau_plot=tau,
                    color=color_dict["blue"],
                    label="HMC mean",
                    ci_label="HMC 95% CI",
                )
                mAP.plot_distribution(
                    ax=axes, tau_plot=tau, color=color_dict["red"], label="MAP"
                )

# plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT_MAP_PeakFits.png"))
# %% Old full cycle
"""
# %% Fitting Old Full cycle
data_dict = {}
dist_keys = ["drt", "ddt", "dmt"]
fit_keys = ["hmc", "map", "hmc_ct", "map_ct", "hmc_diffusion", "map_diffusion"]
with contextlib.redirect_stdout(
    funcs.FilteredStream(filtered_values=["f"])
), warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="overflow encountered")
    for i, eis in enumerate(dFs):
        freq, Z = fl.get_fZ(eis)
        freq_mask, Z_mask = fl.get_fZ(dFs_masked[i])

        dist_dict = {}

        inverters = [copy.deepcopy(x) for x in inverter_list]
        for j, inv in enumerate(inverters):
            dist = dist_keys[j]
            fit_dict = {}
            direc = pkl_direcs[j]

            stringy = direc + "\\" + ident[i]
            fitted_list = []
            hmc = copy.deepcopy(inv)
            hmc_masked = copy.deepcopy(inv)
            mAP = copy.deepcopy(inv)
            mAP_masked = copy.deepcopy(inv)

            if len(inv.distributions) != 1:
                nonneg = True
            else:
                nonneg = False
            for k in fit_list:
                path = stringy + "_" + k + ".pkl"
                if k == "hmc":
                    if os.path.isfile(path):
                        hmc.load_fit_data(path)
                    else:
                        hmc.fit(freq, Z, mode="sample", nonneg=nonneg)
                        hmc.save_fit_data(path, which="core")
                    fit_dict[k] = hmc

                elif k == "hmc_masked":
                    if os.path.isfile(path):
                        hmc_masked.load_fit_data(path)
                    else:
                        hmc_masked.fit(freq_mask, Z_mask, mode="sample", nonneg=nonneg)
                        hmc_masked.save_fit_data(path, which="core")
                    fit_dict[k] = hmc_masked

                elif k == "map":
                    if os.path.isfile(path):
                        mAP.load_fit_data(path)
                    else:
                        mAP.fit(freq, Z, mode="optimize", nonneg=nonneg)
                        mAP.save_fit_data(path, which="core")
                    fit_dict[k] = mAP

                elif k == "map_masked":
                    if os.path.isfile(path):
                        mAP_masked.load_fit_data(path)
                    else:
                        mAP_masked.fit(
                            freq_mask, Z_mask, mode="optimize", nonneg=nonneg
                        )
                        mAP_masked.save_fit_data(path, which="core")
                    fit_dict[k] = mAP_masked

                else:
                    raise ValueError("Invalid fit type")

            dist_dict[dist] = fit_dict
        data_dict[ident[i]] = dist_dict


# %% Visualize DRT and impedance fit Old full
# plot impedance fit and recovered DRT
fig_directory_fit = os.path.join(fig_directory, "FitImpedance")
if not os.path.isdir(fig_directory_fit):
    os.mkdir(fig_directory_fit)
unit_scale = ""
pruning = ["", " - masked"]
area = None
if area:
    unit = f"\ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$"
else:
    unit = f"\ \mathrm{{{unit_scale}}}\Omega$"
color_dict = mpl_settings.bright_dict
no_fill_markers = mpl_settings.no_fill_markers

for data in data_dict:
    for dist in data_dict[data]:
        for i in range(2):
            idx = i * 2
            hmc = copy.deepcopy(data_dict[data][dist][fit_keys[idx]])  # hmc
            mAP = copy.deepcopy(data_dict[data][dist][fit_keys[idx + 1]])  # map
            fig, axes = plt.subplots(figsize=(8, 5.5))
            with contextlib.redirect_stdout(
                funcs.FilteredStream(
                    filtered_values=[
                        "f",
                    ]
                )
            ):
                hmc.plot_fit(
                    axes=axes,
                    area=area,
                    plot_type="nyquist",
                    label="HMC fit",
                    color=color_dict["blue"],
                    data_label="Data",
                    data_kw={
                        "marker": no_fill_markers[0],
                        "linewidths": 1,
                        "color": color_dict["black"],
                        "s": 10,
                        "alpha": 1,
                        "zorder": 2.5,
                    },
                    marker="",
                )
                mAP.plot_fit(
                    axes=axes,
                    area=area,
                    plot_type="nyquist",
                    label="MAP fit",
                    color=color_dict["red"],
                    plot_data=False,
                    marker="",
                    linestyle="dashed",
                    alpha=1,
                )

                funcs.set_equal_tickspace(axes, figure=fig)
                axes.grid(visible=True)
                axes.set_xlabel(f"$Z^\prime \ /" + unit)
                axes.set_ylabel(f"$-Z^{{\prime\prime}} \ /" + unit)
                axes.legend(
                    fontsize=12,
                    frameon=True,
                    framealpha=1,
                    fancybox=False,
                    edgecolor="black",
                )
                axes.set_axisbelow("line")
                plt.suptitle(f"{exp}{data}-{dist}{pruning[i]}")
                plt.gca().set_aspect("equal")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        fig_directory_fit,
                        f"{data}-{dist}{pruning[i]}-FitImpedance.png",
                    )
                )
                plt.show()
                plt.close()

# %% Time constant distributions Old full
fig_directory_time = os.path.join(fig_directory, "TimeConstantDistributions")
if not os.path.isdir(fig_directory_time):
    os.mkdir(fig_directory_time)
unit_scale = ""
pruning = ["", "-masked"]
area = None
color_dict = mpl_settings.bright_dict
no_fill_markers = mpl_settings.no_fill_markers

for data in data_dict:
    for dist in data_dict[data]:
        for i in range(2):
            idx = i * 2
            hmc = copy.deepcopy(data_dict[data][dist][fit_keys[idx]])  # hmc
            mAP = copy.deepcopy(data_dict[data][dist][fit_keys[idx + 1]])  # map
            fig, axes = plt.subplots(figsize=(8, 5.5))
            with contextlib.redirect_stdout(
                funcs.FilteredStream(filtered_values=["f"])
            ):
                tau = None
                hmc.plot_distribution(
                    ax=axes,
                    tau_plot=tau,
                    color=color_dict["blue"],
                    label="HMC mean",
                    ci_label="HMC 95% CI",
                )
                mAP.plot_distribution(
                    ax=axes, tau_plot=tau, color=color_dict["red"], label="MAP"
                )

                axes.legend(
                    fontsize=12,
                    frameon=True,
                    framealpha=1,
                    fancybox=False,
                    edgecolor="black",
                )
                axes.set_axisbelow("line")
                plt.figure(fig)
                plt.xticks()
                print(axes.get_xticks())
                plt.suptitle(f"{exp}{data}-{dist}{pruning[i]}")
                print(f"{data}-{dist}{pruning[i]}")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        fig_directory_time,
                        f"{data}-{dist}{pruning[i]}-TimeConstantDistributions.png",
                    )
                )
                plt.show()
                plt.close()

# %% Peak fitting
# Only fit peaks that have a prominence of >= 10% of the estimated polarization resistance
fig, axes = plt.subplots(2, 1, figsize=(8, 9))
mAP = copy.deepcopy(data_dict["v=1cms"]["dmt"]["map"])  # map
mAP.fit_peaks(prom_rthresh=0.10)
mAP.plot_peak_fit(ax=axes[0])
mAP.plot_peak_fit(ax=axes[1], plot_individual_peaks=True)
fig.tight_layout()
plt.show()

# %% Peak fitting
fig_directory_peak = os.path.join(fig_directory, "PeakFits")
if not os.path.isdir(fig_directory_time):
    os.mkdir(fig_directory_time)
unit_scale = ""
pruning = ["", "-masked"]
area = None
color_dict = mpl_settings.bright_dict
no_fill_markers = mpl_settings.no_fill_markers

for data in data_dict:
    for dist in data_dict[data]:
        for i in range(2):
            idx = i * 2
            hmc = copy.deepcopy(data_dict[data][dist][fit_keys[idx]])  # hmc
            mAP = copy.deepcopy(data_dict[data][dist][fit_keys[idx + 1]])  # map
            with contextlib.redirect_stdout(
                funcs.FilteredStream(filtered_values=["f"])
            ):
                fig, axes = plt.subplots(2, 1, figsize=(8, 9))
                hmc.fit_peaks(prom_rthresh=0.10)
                hmc.plot_peak_fit(ax=axes[0])
                hmc.plot_peak_fit(ax=axes[1], plot_individual_peaks=True)
                axes[1].legend(
                    fontsize=12,
                    frameon=True,
                    framealpha=1,
                    fancybox=False,
                    edgecolor="black",
                )
                plt.figure(fig)
                plt.suptitle(f"{name}{data}-{dist}-{fit_keys[idx]}{pruning[i]}")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        fig_directory_time,
                        f"{data}-{dist}{pruning[i]}-PeakFits.png",
                    )
                )
                plt.show()
                plt.close()

                fig, axes = plt.subplots(2, 1, figsize=(8, 9))
                mAP.fit_peaks(prom_rthresh=0.10)
                mAP.plot_peak_fit(ax=axes[0])
                mAP.plot_peak_fit(ax=axes[1], plot_individual_peaks=True)
                axes.legend(
                    fontsize=12,
                    frameon=True,
                    framealpha=1,
                    fancybox=False,
                    edgecolor="black",
                )
                plt.figure(fig)
                plt.suptitle(f"{name}{data}-{dist}{pruning[i]}")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        fig_directory_time,
                        f"{data}-{dist}{pruning[i]}-PeakFits.png",
                    )
                )
                plt.show()
                plt.close()
                hmc.plot_distribution(
                    ax=axes,
                    tau_plot=tau,
                    color=color_dict["blue"],
                    label="HMC mean",
                    ci_label="HMC 95% CI",
                )
                mAP.plot_distribution(
                    ax=axes, tau_plot=tau, color=color_dict["red"], label="MAP"
                )

# plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT_MAP_PeakFits.png"))
"""
# : Peak fit loop, Lin KK, review bayes_drt tutorials again to make better fits
