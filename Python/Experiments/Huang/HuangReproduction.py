"""
@author: Houlberg
"""
import contextlib
import warnings
import glob
import os
import copy
import time

import funcs
import statics
import mpl_settings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyimpspec as pyi
from scipy.signal import find_peaks

import bayes_drt2.file_load as fl
import bayes_drt2.plotting as bp
from bayes_drt2.inversion import Inverter

warnings.showwarning = funcs.FilteredStream().write_warning
pd.options.mode.chained_assignment = None  # default='warn'

# %% Set folders Huang

directory, fig_directory, pkl_directory, parse_directory = statics.set_experiment(
    "Huang\\Z_BimodalTP-DDT_noiseless"
)
exp = "Huang simulated TP-DDT noiseless"
files = glob.glob("Data/*.csv")

Zdf = pd.read_csv(files[1])
freq, Z = fl.get_fZ(Zdf)

# load true DDT
g_file = files[0]
g_true = pd.read_csv(g_file)

# %% Set folders Sim R+Wd
"""
directory, fig_directory, pkl_directory, parse_directory = statics.set_experiment(
    "Huang\\Sim"
)
exp = "Zsim R+Wd using params from Zfit of Felt electrode at v=0.5cms"
files = glob.glob("Data/*.mpt")
parse = pyi.parse_data(files[0], file_format=".mpt")
columns = ["Freq", "Zreal", "Zimag", "Zmod", "Zphz"]
Zdf = parse[0].to_dataframe(columns=columns)
freq, Z = fl.get_fZ(Zdf)
"""
# %%
axes = bp.plot_eis(Zdf)
plt.show()

# %%
"""
"Fit the data DRT"
# By default, the Inverter class is configured to fit the DRT (rather than the DDT)
# Create separate Inverter instances for HMC and MAP fits
# Set the basis frequencies equal to the measurement frequencies
# (not necessary in general, but yields faster results here - see Tutorial 1 for more info on basis_freq)
inv_hmc = Inverter(basis_freq=freq)
inv_map = Inverter(basis_freq=freq)

# Perform HMC fit
start = time.time()
inv_hmc.fit(freq, Z, mode="sample")
elapsed = time.time() - start
print("HMC fit time {:.1f} s".format(elapsed))

# Perform MAP fit
start = time.time()
inv_map.fit(freq, Z, mode="optimize")  # initialize from ridge solution
elapsed = time.time() - start
print("MAP fit time {:.1f} s".format(elapsed))

# %%
"Visualize DRT and impedance fit"
# plot impedance fit and recovered DRT
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

# plot fits of impedance data
inv_hmc.plot_fit(
    axes=axes[0], plot_type="nyquist", color="k", label="HMC fit", data_label="Data"
)
inv_map.plot_fit(
    axes=axes[0], plot_type="nyquist", color="r", label="MAP fit", plot_data=False
)

# plot true DRT
# p = axes[1].plot(g_true["tau"], g_true["gamma"], label="True", ls="--")
# add Dirac delta function for RC element
# axes[1].plot([np.exp(-2), np.exp(-2)], [0, 10], ls="--", c=p[0].get_color(), lw=1)

# Plot recovered DRT at given tau values
# tau_plot = g_true["tau"].values
tau_plot = None
inv_hmc.plot_distribution(
    ax=axes[1], tau_plot=tau_plot, color="k", label="HMC mean", ci_label="HMC 95% CI"
)
inv_map.plot_distribution(ax=axes[1], tau_plot=tau_plot, color="r", label="MAP")

axes[1].set_ylim(0, 3.5)
axes[1].legend()


fig.tight_layout()
plt.show()

# %%
# Only fit peaks that have a prominence of >= 5% of the estimated polarization resistance
inv_map.fit_peaks(prom_rthresh=0.05)

# plot the peak fit
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
inv_map.plot_peak_fit(ax=axes[0])  # Plot the overall peak fit
inv_map.plot_peak_fit(
    ax=axes[1], plot_individual_peaks=True
)  # Plot the individual peaks

fig.tight_layout()
plt.show()

# %%
inv_hmc.fit_peaks(prom_rthresh=0.05)

# plot the peak fit
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
inv_hmc.plot_peak_fit(ax=axes[0])  # Plot the overall peak fit
inv_hmc.plot_peak_fit(
    ax=axes[1], plot_individual_peaks=True
)  # Plot the individual peaks

fig.tight_layout()
plt.show()
"""
# %%
"Fit the data DDT"
# Define the distribution to be recovered (transmissive planar DDT) in the Inverter initialization
# Use a slightly expanded basis frequency range to fully capture the tail of the low-frequency peak
inv_hmc2 = Inverter(
    distributions={
        "DDT": {
            "kernel": "DDT",
            "dist_type": "parallel",
            "bc": "transmissive",
            "symmetry": "planar",
            "basis_freq": np.logspace(6, -3, 91),
        }
    }
)
inv_map2 = Inverter(
    distributions={
        "DDT": {
            "kernel": "DDT",
            "dist_type": "parallel",
            "bc": "transmissive",
            "symmetry": "planar",
            "basis_freq": np.logspace(6, -3, 91),
        }
    }
)

# %%
# # Perform HMC fit
start = time.time()
inv_hmc2.fit(freq, Z, mode="sample")
elapsed = time.time() - start
print("HMC fit time {:.2f}".format(elapsed))

# Perform MAP fit
start = time.time()
inv_map2.fit(freq, Z)
elapsed = time.time() - start
print("MAP fit time {:.2f}".format(elapsed))

# %%
"Visualize DDT and impedance fit"
# plot impedance fit and recovered DRT

unit_scale = ""
unit = f"\ \mathrm{{{unit_scale}}}\Omega$"
area = None
color_dict = mpl_settings.bright_dict
no_fill_markers = mpl_settings.no_fill_markers
fig, axes = plt.subplots(figsize=(8, 5.5))
with contextlib.redirect_stdout(
    funcs.FilteredStream(
        filtered_values=[
            "f",
        ]
    )
):
    inv_hmc2.plot_fit(
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
    inv_map2.plot_fit(
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
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            fig_directory,
            f"HuangDDTFitImpedance.png",
        )
    )
    plt.show()
    plt.close()

# %%
# plot true DRT
fig, axes = plt.subplots(figsize=(8, 5.5))
p = axes.plot(g_true["tau"], g_true["gamma"] * 1000, label="True", ls="--", marker="")

# Plot recovered DRT at given tau values
tau_plot = g_true["tau"].values
with contextlib.redirect_stdout(funcs.FilteredStream(filtered_values=["f"])):
    tau = tau_plot
    inv_hmc2.plot_distribution(
        ax=axes,
        tau_plot=tau,
        color=color_dict["blue"],
        label="HMC mean",
        ci_label="HMC 95% CI",
        marker="",
    )
    inv_map2.plot_distribution(
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
            fig_directory,
            f"HuangDDTTimeConstantDistributions.png",
        )
    )
    plt.show()
    plt.close()
# %%
# Only fit peaks that have a prominence of >= 5% of the estimated polarization resistance
inv_map2.fit_peaks(prom_rthresh=0.10)

# plot the peak fit
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
inv_map2.plot_peak_fit(ax=axes[0])  # Plot the overall peak fit
inv_map2.plot_peak_fit(
    ax=axes[1], plot_individual_peaks=True
)  # Plot the individual peaks

fig.tight_layout()
plt.show()

# %%
inv_hmc2.fit_peaks(prom_rthresh=0.10)

# plot the peak fit
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
inv_hmc2.plot_peak_fit(ax=axes[0])  # Plot the overall peak fit
inv_hmc2.plot_peak_fit(
    ax=axes[1], plot_individual_peaks=True
)  # Plot the individual peaks

fig.tight_layout()
plt.show()

# %%

with contextlib.redirect_stdout(funcs.FilteredStream(filtered_values=["f"])):
    fig, axes = plt.subplots(2, 1, figsize=(8, 9))
    inv_hmc2.fit_peaks(prom_rthresh=0.20)
    inv_hmc2.plot_peak_fit(ax=axes[0])
    inv_hmc2.plot_peak_fit(ax=axes[1], plot_individual_peaks=True)
    axes[1].legend(
        fontsize=12,
        frameon=True,
        framealpha=1,
        fancybox=False,
        edgecolor="black",
    )
    plt.figure(fig)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            fig_directory,
            f"HuangDDTPeakFitsHMC.png",
        )
    )
    plt.show()
    plt.close()
# %%
with contextlib.redirect_stdout(funcs.FilteredStream(filtered_values=["f"])):
    fig, axes = plt.subplots(2, 1, figsize=(8, 9))
    inv_map2.fit_peaks(prom_rthresh=0.20)
    inv_map2.plot_peak_fit(ax=axes[0], distribution_kw={"marker": ""}, marker="")
    inv_map2.plot_peak_fit(
        ax=axes[1],
        plot_individual_peaks=True,
        distribution_kw={"marker": ""},
        marker="",
    )
    axes[1].legend(
        fontsize=12,
        frameon=True,
        framealpha=1,
        fancybox=False,
        edgecolor="black",
    )
    plt.figure(fig)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            fig_directory,
            f"HuangDDTPeakFitsMAP.png",
        )
    )
    plt.show()
    plt.close()
