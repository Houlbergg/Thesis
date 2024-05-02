"""
@author: Houlberg
"""
import contextlib
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
# Init

import glob
import os
import copy

import matplotlib.pyplot as plt
import mpl_settings

import numpy as np
import pandas as pd
import pyimpspec as pyi

import bayes_drt2.file_load as fl
import matplotlib.axes._secondary_axes
from bayes_drt2.inversion import Inverter

import funcs
import statics

pd.options.mode.chained_assignment = None  # default='warn'

# %% Set folders
directory, fig_directory, pkl_directory, parse_directory = statics.set_experiment(
    "Baichen\\IronSymmetricNIPSElectrode\\FTFF"
)
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
# %% Selecting the wanted cycles
# Easy in this case for Baichen, there is only a single cycle in each file.

chosen_names = files
# idx = list(map(lambda x: files.index(x), chosen_names)) Generalized form perhaps
chosen_parses = parses
chosen_parses = [x[0] for x in chosen_parses]
chosen_masked = parses_masked
chosen_masked = [x[0] for x in chosen_masked]

ident = ["v=1cms", "v=2cms", "v=5cms"]

# %% Construct Pandas df because bayes_drt package likes that
columns = ["Freq", "Zreal", "Zimag", "Zmod", "Zphs"]
dFs = []
dFs_masked = []
for eis in chosen_parses:
    dFs.append(eis.to_dataframe(columns=columns))
for eis in chosen_masked:
    dFs_masked.append(eis.to_dataframe(columns=columns))

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
                "ct": False,  # indicates no simultaneous charge transfer
                "basis_freq": np.logspace(6, -3, 91),
            },
        },
        basis_freq=np.logspace(
            6, -3, 91
        ),  # use basis range large enough to capture full DDT
    )
    inverter_list = [drt, ddt, dmt]
dist_list_str = ["drt", "ddt", "dmt"]
fit_list = ["hmc", "map", "hmc_masked", "map_masked"]

for direc in dist_list_str:
    if not os.path.isdir(pkl_directory + "\\" + direc):
        os.mkdir(pkl_directory + "\\" + direc)
pkl_direcs = [pkl_directory + "\\" + x for x in dist_list_str]

# %% Fitting
data_dict = {}
dist_keys = ["drt", "ddt", "dmt"]
fit_keys = ["hmc", "map", "hmc_masked", "map_masked"]
with contextlib.redirect_stdout(funcs.FilteredStream(filtered_values=["f"])):
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


# %% Visualize DRT and impedance fit
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
                funcs.FilteredStream(filtered_values=["f"])
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
                plt.suptitle(f"{data} - {dist}{pruning[i]}")
                plt.gca().set_aspect("equal")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        fig_directory_fit,
                        f"{data} - {dist}{pruning[i]}-FitImpedance.png",
                    )
                )
                plt.show()
                plt.close()

# %% Time Constant Tests
tau = None
fig, axes = plt.subplots()
data_dict['v=1cms']['drt']['hmc'].plot_distribution(ax=axes, tau_plot=tau, color='k', label='HMC mean', ci_label='HMC 95% CI')
data_dict['v=1cms']['drt']['map'].plot_distribution(ax=axes, tau_plot=tau, color='r', label='MAP')
axes.legend()
plt.figure(fig)
plt.show()
plt.close()
# %%
fig_directory_time = os.path.join(fig_directory, "TimeConstantDistributions")
if not os.path.isdir(fig_directory_time):
    os.mkdir(fig_directory_time)
unit_scale = ""
pruning = ["", " - masked"]
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
                hmc.plot_distribution(ax=axes, tau_plot=tau, color=color_dict["blue"], label='HMC mean', ci_label='HMC 95% CI')
                mAP.plot_distribution(ax=axes, tau_plot=tau, color=color_dict["red"], label='MAP')

                funcs.set_equal_tickspace(axes, figure=fig)
                axes.legend(
                    fontsize=12,
                    frameon=True,
                    framealpha=1,
                    fancybox=False,
                    edgecolor="black",
                )
                axes.set_axisbelow("line")
                plt.suptitle(f"{data} - {dist}{pruning[i]}")
                plt.gca().set_aspect("equal")
                print(f"{data} - {dist}{pruning[i]}")
                #plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        fig_directory_time,
                        f"{data} - {dist}{pruning[i]}-TimeConstantDistributions.png",
                    )
                )
                plt.show()
                plt.close()

# %% Time Constant Distributions
for i, dist in enumerate(dFs):
    #tau_drt = drt_map_list[i].distributions['DRT']['tau']
    tau_drt = None
    fig, axes = plt.subplots()
    drt_hmc_list[i].plot_distribution(ax=axes, tau_plot=tau_drt, color='k', label='HMC mean', ci_label='HMC 95% CI')
    drt_map_list[i].plot_distribution(ax=axes, tau_plot=tau_drt, color='r', label='MAP')
    sec_xaxis = [x for x in axes.get_children() if isinstance(x, matplotlib.axes._secondary_axes.SecondaryAxis)][0]
    sec_xaxis.set_xlabel('$f$ / Hz')
    axes.legend()
    plt.figure(fig)
plt.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT.png"))
    plt.close()

    #tau_ddt = ddt_map_list[i].distributions['TP-DDT']['tau']
    tau_ddt = None
    fig, axes = plt.subplots()
    ddt_hmc_list[i].plot_distribution(ax=axes, tau_plot=tau_ddt, color='k', label='HMC mean', ci_label='HMC 95% CI')
    ddt_map_list[i].plot_distribution(ax=axes, tau_plot=tau_ddt, color='r', label='MAP')
    sec_xaxis = [x for x in axes.get_children() if isinstance(x, matplotlib.axes._secondary_axes.SecondaryAxis)][0]
    sec_xaxis.set_xlabel('$f$ / Hz')
    axes.legend()
    plt.figure(fig)
plt.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DDT.png"))
    plt.close()

    #tau_multi = inv_multi_map_list[i].distributions['DRT']['tau']
    tau_multi = None
    fig, axes = plt.subplots()
    inv_multi_hmc_list[i].plot_distribution(ax=axes, tau_plot=tau_multi, color='k', label='HMC mean', ci_label='HMC 95% CI')
    inv_multi_map_list[i].plot_distribution(ax=axes, tau_plot=tau_multi, color='r', label='MAP')
    sec_xaxis = [x for x in axes.get_children() if isinstance(x, matplotlib.axes._secondary_axes.SecondaryAxis)][0]
    sec_xaxis.set_xlabel('$f$ / Hz')
    axes.legend()
    plt.figure(fig)
plt.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT-DDT.png"))
    plt.close()

    #tau_drt_mask = drt_map_masked_list[i].distributions['DRT']['tau']
    tau_drt_mask = None
    fig, axes = plt.subplots()
    drt_hmc_masked_list[i].plot_distribution(ax=axes, tau_plot=tau_drt_mask, color='k', label='HMC mean', ci_label='HMC 95% CI')
    drt_map_masked_list[i].plot_distribution(ax=axes, tau_plot=tau_drt_mask, color='r', label='MAP')
    sec_xaxis = [x for x in axes.get_children() if isinstance(x, matplotlib.axes._secondary_axes.SecondaryAxis)][0]
    sec_xaxis.set_xlabel('$f$ / Hz')
    axes.legend()
    plt.figure(fig)
plt.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT_masked.png"))
    plt.close()

    #tau_ddt_mask = ddt_map_masked_list[i].distributions['TP-DDT']['tau']
    tau_ddt_mask = None
    fig, axes = plt.subplots()
    ddt_hmc_masked_list[i].plot_distribution(ax=axes, tau_plot=tau_ddt_mask, color='k', label='HMC mean', ci_label='HMC 95% CI')
    ddt_map_masked_list[i].plot_distribution(ax=axes, tau_plot=tau_ddt_mask, color='r', label='MAP')
    sec_xaxis = [x for x in axes.get_children() if isinstance(x, matplotlib.axes._secondary_axes.SecondaryAxis)][0]
    sec_xaxis.set_xlabel('$f$ / Hz')
    axes.legend()
    plt.figure(fig)
plt.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DDT_masked.png"))
    plt.close()

    #tau_multi_mask = inv_multi_map_masked_list[i].distributions['DRT']['tau']
    tau_multi_mask = None
    fig, axes = plt.subplots()
    inv_multi_hmc_masked_list[i].plot_distribution(ax=axes, tau_plot=tau_multi_mask, color='k', label='HMC mean', ci_label='HMC 95% CI')
    inv_multi_map_masked_list[i].plot_distribution(ax=axes, tau_plot=tau_multi_mask, color='r', label='MAP')
    sec_xaxis = [x for x in axes.get_children() if isinstance(x, matplotlib.axes._secondary_axes.SecondaryAxis)][0]
    sec_xaxis.set_xlabel('$f$ / Hz')
    axes.legend()
    plt.figure(fig)
plt.tight_layout()
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
plt.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT_HMC_Residuals.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    drt_hmc_masked_list[i].plot_residuals(axes=axes)
    plt.figure(fig)
plt.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT_HMC_Residuals_masked.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    ddt_hmc_list[i].plot_residuals(axes=axes)
    # fig.suptitle("Bayes Estimated Error Structure")
    plt.figure(fig)
plt.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DDT_HMC_Residuals.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    ddt_hmc_masked_list[i].plot_residuals(axes=axes)
    plt.figure(fig)
plt.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DDT_HMC_Residuals_masked.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    inv_multi_hmc_list[i].plot_residuals(axes=axes)
    # fig.suptitle("Bayes Estimated Error Structure")
    plt.figure(fig)
plt.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT-DDT_HMC_Residuals.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    inv_multi_hmc_masked_list[i].plot_residuals(axes=axes)
    plt.figure(fig)
plt.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT-DDT_HMC_Residuals_masked.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    drt_map_list[i].plot_residuals(axes=axes)
    # fig.suptitle("Bayes Estimated Error Structure")
    plt.figure(fig)
plt.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT_MAP_Residuals.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    drt_map_masked_list[i].plot_residuals(axes=axes)
    plt.figure(fig)
plt.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT_MAP_Residuals_masked.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    ddt_map_list[i].plot_residuals(axes=axes)
    # fig.suptitle("Bayes Estimated Error Structure")
    plt.figure(fig)
plt.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DDT_MAP_Residuals.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    ddt_map_masked_list[i].plot_residuals(axes=axes)
    plt.figure(fig)
plt.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DDT_MAP_Residuals_masked.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    inv_multi_map_list[i].plot_residuals(axes=axes)
    # fig.suptitle("Bayes Estimated Error Structure")
    plt.figure(fig)
plt.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT-DDT_MAP_Residuals.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    inv_multi_map_masked_list[i].plot_residuals(axes=axes)
    plt.figure(fig)
plt.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT-DDT_MAP_Residuals_masked.png"))
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
    drt_map_list[i].plot_peak_fit(ax=axes[1], plot_individual_peaks=True)  # Plot the individual peaks
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.figure(fig)
plt.tight_layout()
    fig.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT_MAP_PeakFits.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    ddt_map_list[i].fit_peaks(prom_rthresh=0.05)
    ddt_map_list[i].plot_peak_fit(ax=axes[0])  # Plot the overall peak fit
    ddt_map_list[i].plot_peak_fit(ax=axes[1], plot_individual_peaks=True)  # Plot the individual peaks
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.figure(fig)
plt.tight_layout()
    fig.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DDT_MAP_PeakFits.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    inv_multi_map_list[i].fit_peaks(prom_rthresh=0.05)
    inv_multi_map_list[i].plot_peak_fit(ax=axes[0])  # Plot the overall peak fit
    inv_multi_map_list[i].plot_peak_fit(ax=axes[1], plot_individual_peaks=True)  # Plot the individual peaks
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.figure(fig)
plt.tight_layout()
    fig.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT-DDT_MAP_PeakFits.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    try:
        drt_map_masked_list[i].fit_peaks(prom_rthresh=0.05)
        drt_map_masked_list[i].plot_peak_fit(ax=axes[0])  # Plot the overall peak fit
        drt_map_masked_list[i].plot_peak_fit(ax=axes[1], plot_individual_peaks=True)  # Plot the individual peaks
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.figure(fig)
plt.tight_layout()
        fig.tight_layout()
        plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT_MAP_PeakFits_masked.png"))
        plt.close()
    except:
        print("Error with file " + str(chosen[i]) + "For DRT Non-zero Peak Fitting")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    ddt_map_list[i].fit_peaks(prom_rthresh=0.05)
    ddt_map_list[i].plot_peak_fit(ax=axes[0])  # Plot the overall peak fit
    ddt_map_list[i].plot_peak_fit(ax=axes[1], plot_individual_peaks=True)  # Plot the individual peaks
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.figure(fig)
plt.tight_layout()
    fig.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DDT_MAP_PeakFits_masked.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    inv_multi_map_list[i].fit_peaks(prom_rthresh=0.05)
    inv_multi_map_list[i].plot_peak_fit(ax=axes[0])  # Plot the overall peak fit
    inv_multi_map_list[i].plot_peak_fit(ax=axes[1], plot_individual_peaks=True)  # Plot the individual peaks
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.figure(fig)
plt.tight_layout()
    fig.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT-DDT_MAP_PeakFits_masked.png"))
    plt.close()

#%% Re sort files

figures = glob.glob(os.path.join(fig_directory, '*'))

for iden in ident:
    if not os.path.isdir(os.path.join(fig_directory, str(iden))):
        os.makedirs(os.path.join(fig_directory, str(iden)))

for file in figures:
    name = os.path.basename(file)
    if '25mHz-20ppd' in name:
        shutil.move(file, os.path.join(fig_directory, '25mHz-20ppd'))
"""
