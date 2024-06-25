# %% Init
import numpy as np
import pandas as pd
import os

# import sys
# import time
import glob
import gc
import matplotlib.pyplot as plt

import funcs
import statics
import mpl_settings

# import matplotlib.axes.x
from matplotlib import ticker

# import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from cycler import cycler

# from bayes_drt2.inversion import Inverter
# import bayes_drt2.file_load as fl
# import bayes_drt2.plotting as bp
from bayes_drt2 import utils

import pyimpspec as pyi
from pyimpspec import DataSet

# from pyimpspec.plot.mpl.utility import _configure_log_scale, _configure_log_limits

# from pyimpspec import mpl
# import eclabfiles as ecf
# import yadg
# import fnmatch

pd.options.mode.chained_assignment = None  # default='warn'
load_with_captions = True

# %% Set folders
directory, fig_directory, pkl_directory, parse_directory = statics.set_experiment(
    "RDE\\Dicarboxylferrocene"
)

exp = "Dicarboxylferrocene"
files = glob.glob("Data/*.mpt")

peisfiles = []
cvfiles = []

for file in files:
    filename = os.fsdecode(file)
    if "_PEIS_" in filename:
        peisfiles.append(file)
    elif "_CV_" in filename:
        cvfiles.append(file)

# %% Parsing with pyimpspec. REQUIRES exporting EC-Lab raw binaries (.mpr) as text (.mpt)
peisparses = []
for file in peisfiles:
    name = os.path.basename(file).split(".")[0]
    string = os.path.join(parse_directory, name) + ".pkl"
    if not os.path.isfile(string):
        try:
            parse = pyi.parse_data(file, file_format=".mpt")
            funcs.save_pickle(parse, string)
            peisparses.append(parse)
        except AssertionError as e:
            print("Pyimpspec parse of file: " + str(file) + " failed due to" + str(e))
    else:
        peisparses.append(funcs.load_pickle(string))

# %% Parsing with ecdh

cvparses = []
for file in cvfiles:
    cv = funcs.read_mpt(file)
    cv = cv.loc[cv["cycle number"] == cv["cycle number"].max()]
    cvparses.append(cv)

# %% Plotting CV
plot_cycles = [cvparses[0], cvparses[3], cvparses[5]]
fig, ax = plt.subplots()
labels = ["Initial", "Post-EIS", "Re-balanced"]
for i, cycle in enumerate(plot_cycles):
    ax.plot(cycle["Ewe/V"], cycle["<I>/mA"], label=labels[i], marker="")
    ax.set_xlabel(r"$E / V$")
    ax.set_ylabel(r"$I / A$")

    ax.legend()
    ax.grid(visible=True)
fig.suptitle("5mM Dicarboxyferrocene, 1M NaCl")
plt.figure(fig)
plt.tight_layout()
plt.savefig(os.path.join(fig_directory, f"CVs.png"))
plt.show()
# %% Plotting EIS
plot_spectra = [peisparses[2], peisparses[5]]
unit_scale = ""  # If you want to swap to different scale. milli, kilo etc
fig, ax = plt.subplots()
for i, peis in enumerate(plot_spectra[0]):
    imp = peis.get_impedances()
    ax.plot(imp.real, -imp.imag, label="Cycle #" + str(i))

    funcs.set_equal_tickspace(ax, figure=fig, space=("max", 1))
    ax.grid(visible=True)
    ax.set_xlabel(f"$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega$")
    ax.set_ylabel(f"$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega$")
    ax.legend(
        fontsize=12,
        frameon=True,
        framealpha=1,
        fancybox=False,
        edgecolor="black",
    )
ax.set_axisbelow("line")
# plt.suptitle(f"{exp}{data}-{dist}{pruning[i]}")
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.savefig(
    os.path.join(
        fig_directory,
        f"EISDCFC.png",
    )
)
plt.show()
plt.close()

# %% Plotting EIS Multi
"""
unit_scale = ""  # If you want to swap to different scale. milli, kilo etc
labels = labels[::2]
fig, ax = plt.subplots()
for i, meas in enumerate(plot_spectra):
    for peis in meas:
        imp = peis.get_impedances()
        ax.scatter(imp.real, -imp.imag, color=colors[i], label=labels[i])

    ax = set_aspect_ratio(ax, peis)
    ax = set_equal_tickspace(ax, figure=fig)
    ax.grid(visible=True)
    ax.set_xlabel(f"$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega$")
    ax.set_ylabel(f"$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega$")
    ax.legend()

# plt.tight_layout()
plt.gca().set_aspect("equal")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.savefig(os.path.join(fig_directory, f"EISComparison.png"))
plt.show()
"""
# %% Lin - KK ala pyimpspec
mu_criterion = 0.85
subject = plot_spectra[0][-1]
if not os.path.isfile(os.path.join(pkl_directory, "LinKKTests.pkl")):
    tests = pyi.perform_exploratory_tests(subject, mu_criterion=mu_criterion)
    utils.save_pickle(tests, os.path.join(pkl_directory, "LinKKTests.pkl"))
else:
    tests = utils.load_pickle(os.path.join(pkl_directory, "LinKKTests.pkl"))
fig, ax = pyi.mpl.plot_mu_xps(tests, mu_criterion=mu_criterion)
plt.savefig(os.path.join(fig_directory, f"ExploratoryTests.png"))
plt.show()

# %% Residual plot
# I personally prefer freq going high to low from left to right ("similar" as nyquist)
fig, ax = pyi.mpl.plot_residuals(tests[0])
plt.gca().invert_xaxis()
plt.savefig(os.path.join(fig_directory, f"LinKKResiduals.png"))
plt.show()

# %%
"""
figure, axes = pyi.mpl.plot_nyquist(tests[0], line=True)
_ = pyi.mpl.plot_nyquist(subject, figure=figure, axes=axes, colors={
                         "impedance": "black"}, label='Data')
figure.tight_layout()
"""

# %% Plotting Lin KK fitted EIS
unit_scale = ""  # If you want to swap to different scale. milli, kilo etc
labels = ["Data", "#(RC)=30"]
plot_type = ["scatter", "plot"]

color_dict = mpl_settings.bright_dict
no_fill_markers = mpl_settings.no_fill_markers

fig, ax = plt.subplots()
imp_data = subject.get_impedances()
imp_rc = tests[0].get_impedances()
ax.scatter(
    imp_data.real,
    -imp_data.imag,
    color=color_dict["black"],
    marker=no_fill_markers[0],
    linewidths=1,
    label="Data",
)
ax.plot(
    imp_rc.real,
    -imp_rc.imag,
    color=color_dict["red"],
    marker="",
    label="#(RC)=30",
    zorder=0.5,
)
funcs.set_equal_tickspace(ax, figure=fig, space="max")
ax.grid(visible=True)
ax.set_xlabel(f"$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega$")
ax.set_ylabel(f"$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega$")
ax.legend()

line = True
plt.tight_layout()
plt.gca().set_aspect("equal")


plt.savefig(os.path.join(fig_directory, f"LinKKImpedanceFit.png"))
plt.show()

# %% Z-hit
if not os.path.isfile(os.path.join(pkl_directory, "Z-hit.pkl")):
    zHit = pyi.perform_zhit(subject, interpolation="auto")
    utils.save_pickle(zHit, os.path.join(pkl_directory, "Z-hit.pkl"))
else:
    zHit = utils.load_pickle(os.path.join(pkl_directory, "Z-hit.pkl"))

# %%
fig, ax = plt.subplots()
# _, ax = pyi.mpl.plot_bode(subject, fig=fig, axes=[ax, ax])
pyi.mpl.plot_bode(zHit, fig=fig)
plt.show()
