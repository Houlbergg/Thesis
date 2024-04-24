# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:48:04 2024

@author: Houlberg
"""
# %% Init
import numpy as np
import pandas as pd
import os

# import sys
# import time
import glob
import gc
import matplotlib.pyplot as plt

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


# %% Aspect Ratio function
# Blatantly stolen from bayes_drt
# Adjusted for pyimpspec Datasets as compared to pandas DataFrames


def set_aspect_ratio(ax, dataset):
    pyimp = dataset.get_impedances()
    img = pyimp.imag
    real = pyimp.real

    img_max = np.max(img)
    img_min = np.min(img)
    real_max = np.max(real)
    real_min = np.min(real)
    # make scale of x and y axes the same
    fig = ax.get_figure()

    # if data extends beyond axis limits, adjust to capture all data
    ydata_range = img_max - img_min
    xdata_range = real_max - real_min
    if np.min(-img) < ax.get_ylim()[0]:
        if np.min(-img) >= 0:
            # if data doesn't go negative, don't let y-axis go negative
            ymin = max(0, np.min(-img) - ydata_range * 0.1)
        else:
            ymin = np.min(-img) - ydata_range * 0.1
    else:
        ymin = ax.get_ylim()[0]
    if np.max(-img) > ax.get_ylim()[1]:
        ymax = np.max(-img) + ydata_range * 0.1
    else:
        ymax = ax.get_ylim()[1]
    ax.set_ylim(ymin, ymax)

    if real_min < ax.get_xlim()[0]:
        if real_min == False:
            # if data doesn't go negative, don't let x-axis go negative
            xmin = max(0, real_min - xdata_range * 0.1)
        else:
            xmin = real_min - xdata_range * 0.1
    else:
        xmin = ax.get_xlim()[0]
    if real_max > ax.get_xlim()[1]:
        xmax = real_max + xdata_range * 0.1
    else:
        xmax = ax.get_xlim()[1]
    ax.set_xlim(xmin, xmax)

    # get data range
    yrng = ax.get_ylim()[1] - ax.get_ylim()[0]
    xrng = ax.get_xlim()[1] - ax.get_xlim()[0]

    # get axis dimensions

    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height

    yscale = yrng / height
    xscale = xrng / width

    if yscale > xscale:
        # set figsize square
        # expand the x axis
        # width = height
        # fig.set_figwidth(width)
        # diff = (yscale - xscale) * width
        # xmin = max(0, ax.get_xlim()[0] - diff / 2)
        # mindelta = ax.get_xlim()[0] - xmin
        # xmax = ax.get_xlim()[1] + diff - mindelta

        ax.set_xlim(ax.get_ylim()[0], ax.get_ylim()[1])

        # bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        # width, height = bbox.width, bbox.height

        # width = height
        # fig.set_figwidth(width)
    elif xscale > yscale:
        # expand the y axis
        diff = (xscale - yscale) * height
        if min(np.min(-img), ax.get_ylim()[0]) >= 0:
            # if -Zimag doesn't go negative, don't go negative on y-axis
            ymin = max(0, ax.get_ylim()[0] - diff / 2)
            mindelta = ax.get_ylim()[0] - ymin
            ymax = ax.get_ylim()[1] + diff - mindelta
        else:
            negrng = abs(ax.get_ylim()[0])
            posrng = abs(ax.get_ylim()[1])
            negoffset = negrng * diff / (negrng + posrng)
            posoffset = posrng * diff / (negrng + posrng)
            ymin = ax.get_ylim()[0] - negoffset
            ymax = ax.get_ylim()[1] + posoffset

        ax.set_ylim(ymin, ymax)

    return ax


# %% Equal tickspace function
# %% CV parser from ecdh

# Borrowed :) from ecdh https://github.com/amundmr/ecdh/blob/main/ecdh/readers/BioLogic.py


def read_mpt(filepath):
    """
    Author: Amund M. Raniseth
    Reads an mpt file to a pandas dataframe

    .MPT format:
        mode column: 1=Galvanostatic, 2=Linear Potential Sweep, 3=Rest

    """
    modes = {1: "Galvanostatic", 2: "Linear Potential Sweep", 3: "Rest"}

    # with open(filepath, 'r', encoding= "iso-8859-1") as f:  #open the filepath for the mpt file
    #    lines = f.readlines()
    with open(filepath, errors="ignore") as f:
        lines = f.readlines()

    # now we skip all the data in the start and jump straight to the dense data
    headerlines = 0
    for line in lines:
        if "Nb header lines :" in line:
            headerlines = int(line.split(":")[-1])
            break  # breaks for loop when headerlines is found

    for i, line in enumerate(lines):
        # You wont find Characteristic mass outside of headerlines.
        if headerlines > 0:
            if i > headerlines:
                break

        if "Characteristic mass :" in line:
            active_mass = float(line.split(":")[-1][:-3].replace(",", ".")) / 1000
            # print("Active mass found in file to be: " + str(active_mass) + "g")
            break  # breaks loop when active mass is found

    # pandas.read_csv command automatically skips white lines, meaning that we need to subtract this from the amout of headerlines for the function to work.
    whitelines = 0
    for i, line in enumerate(lines):
        # we dont care about outside of headerlines.
        if headerlines > 0:
            if i > headerlines:
                break
        if line == "\n":
            whitelines += 1

    # Remove lines object
    del lines
    gc.collect()

    big_df = pd.read_csv(
        filepath, header=headerlines - whitelines - 1, sep="\t", encoding="ISO-8859-1"
    )
    # print("Dataframe column names: {}".format(big_df.columns))

    # Start filling dataframe
    if "I/mA" in big_df.columns:
        current_header = "I/mA"
    elif "<I>/mA" in big_df.columns:
        current_header = "<I>/mA"
    df = big_df[["mode", "time/s", "Ewe/V", current_header, "cycle number", "ox/red"]]
    # Change headers of df to be correct
    df.rename(columns={current_header: "<I>/mA"}, inplace=True)

    # If it's galvanostatic we want the capacity
    mode = df["mode"].value_counts()  # Find occurences of modes
    # Remove the count of modes with rest (if there are large rests, there might be more rest datapoints than GC/CV steps)
    mode = mode[mode.index != 3]
    mode = mode.idxmax()  # Picking out the mode with the maximum occurences
    # print("Found cycling mode: {}".format(modes[mode]))

    if mode == 1:
        df = df.join(big_df["Capacity/mA.h"])
        df.rename(columns={"Capacity/mA.h": "capacity/mAhg"}, inplace=True)
        # the str.replace only works and is only needed if it is a string
        if df.dtypes["capacity/mAhg"] == str:
            df["capacity/mAhg"] = pd.to_numeric(
                df["capacity/mAhg"].str.replace(",", ".")
            )
    del big_df  # deletes the dataframe
    gc.collect()  # Clean unused memory (which is the dataframe above)
    # Replace , by . and make numeric from strings. Mode is already interpreted as int.
    for col in df.columns:
        if (
            df.dtypes[col] == str
        ):  # the str.replace only works and is only needed if it is a string
            df[col] = pd.to_numeric(df[col].str.replace(",", "."))
    # df['time/s'] = pd.to_numeric(df['time/s'].str.replace(',','.'))
    # df['Ewe/V'] = pd.to_numeric(df['Ewe/V'].str.replace(',','.'))
    # df['<I>/mA'] = pd.to_numeric(df['<I>/mA'].str.replace(',','.'))
    # df['cycle number'] = pd.to_numeric(df['cycle number'].str.replace(',','.')).astype('int32')
    df.rename(columns={"ox/red": "charge"}, inplace=True)
    df["charge"].replace({1: True, 0: False}, inplace=True)

    #    if mode == 2:
    # If it is CV data, then BioLogic counts the cycles in a weird way (starting new cycle when passing the point of the OCV, not when starting a charge or discharge..) so we need to count our own cycles

    #        df['cycle number'] = df['charge'].ne(df['charge'].shift()).cumsum()
    #        df['cycle number'] = df['cycle number'].floordiv(2)

    # Adding attributes must be the last thing to do since it will not be copied when doing operations on the dataframe
    df.experiment_mode = mode

    return df


# %% Homemade tickspace function


def set_equal_tickspace(ax, figure=None):
    assert isinstance(figure, Figure) or figure is None, figure
    if figure is None:
        assert ax is None
        figure, axis = plt.subplots()
        ax = [axis]

    xticks = plt.xticks()[0]
    yticks = plt.yticks()[0]
    xtickspace = xticks[1] - xticks[0]
    ytickspace = yticks[1] - yticks[0]

    spacing = min(xtickspace, ytickspace)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=spacing))

    return ax


# %% Removes legend duplicates


def remove_legend_duplicate():
    try:
        handles, labels = plt.gca().get_legend_handles_labels()
    except ValueError:
        print("No plot available")
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


# %% Establish Static Directories
# os.chdir(__file__)
home = r"C:\Users\Houlberg\Documents\Thesis\Python"
os.chdir(home)

# Load stan files for unpickeling ??????
sms = glob.glob("../bayes-drt2-main/bayes_drt2/stan_model_files/*.pkl")
# sms = glob.glob('Pickles/*.pkl')
for file in sms:
    utils.load_pickle(file)

exp_dir = os.path.join(home, r"Experiments")
fig_dir = os.path.join(home, r"Figures")
pkl_dir = os.path.join(home, r"Pickles")
# pkl = os.path.join(pkl_dir,'obj{}.pkl'.format('MAP'))

tester = home + r"\Experiments"
# %% Selecting Experiment and establishing subfolders

experiment = r"RDE\Dicarboxylferrocene"
if not os.path.isdir(os.path.join(exp_dir, experiment)):
    os.makedirs(os.path.join(exp_dir, experiment))

if not os.path.isdir(os.path.join(fig_dir, experiment)):
    os.makedirs(os.path.join(fig_dir, experiment))

if not os.path.isdir(os.path.join(pkl_dir, experiment)):
    os.makedirs(os.path.join(pkl_dir, experiment))

directory = os.path.join(exp_dir, experiment)
pkl_directory = os.path.join(pkl_dir, experiment)
fig_directory = os.path.join(fig_dir, experiment)
os.chdir(directory)

if not os.path.isdir("Parsed"):
    os.makedirs("Parsed")

# %% Matplotlib parameters
# base = 'seaborn-v0_8' # -paper or -poster for smaller or larger figs ##Doesnt really work
# size = 'seaborn-v0_8-poster'
# ticks = 'seaborn-v0_8-ticks'
# plt.style.use([base,size,ticks])

colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
plt.rcParams["axes.prop_cycle"] = cycler(color=colors)
plt.rcParams["patch.facecolor"] = colors[0]

tick_size = 9
label_size = 11
# fig_width = 10
plt.rcParams["figure.figsize"] = [8, 5.5]
plt.rcParams["figure.dpi"] = 200
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "Liberation Sans",
    "DejaVu Sans",
    "Bitstream Vera Sans",
    "sans-serif",
]
plt.rcParams["mathtext.fontset"] = "dejavuserif"

plt.rcParams["lines.markersize"] = 5
plt.rcParams["xtick.labelsize"] = tick_size
plt.rcParams["ytick.labelsize"] = tick_size
plt.rcParams["axes.labelsize"] = label_size
plt.rcParams["legend.fontsize"] = tick_size + 1
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.framealpha"] = 0.95

# %% Load Files and seperate by type.
# REQUIRES exporting EC-Lab raw binarys (.mpr) as text (.mpt)
files = glob.glob("Data/*.mpt")

peisfiles = []
cvfiles = []

for file in files:
    filename = os.fsdecode(file)
    if "_PEIS_" in filename:
        peisfiles.append(file)
    elif "_CV_" in filename:
        cvfiles.append(file)

# %% Parsing with pyimpspec
dummy_freq = np.logspace(6, -2, 81)
dummy_Z = np.ones_like(dummy_freq, dtype="complex128")
pyi_dummy = DataSet(dummy_freq, dummy_Z)

parses_eis = []
for file in peisfiles:
    string = "Parsed/{}".format(file.split(".")[0].split("\\")[1])
    if not os.path.isdir(string):
        os.makedirs(string)
        try:
            parse = pyi.parse_data(file, file_format=".mpt")
            parses_eis.append(parse)
            for i, cycle in enumerate(parse):
                utils.save_pickle(
                    cycle.to_dict(),
                    os.path.join(  # .to_dict() might be unneccesary)
                        string, "Cycle_{}.pkl".format(i)
                    ),
                )
        except:
            print("Pyimpspec could not parse file" + str(file))
    else:
        temp_list = []
        for i, cycle in enumerate(os.listdir(string)):
            pyi_pickle = utils.load_pickle(
                os.path.join(string, "Cycle_{}.pkl".format(i))
            )
            parse = pyi_dummy.from_dict(pyi_pickle)
            temp_list.append(parse)
        parses_eis.append(temp_list)

# %% Parsing with ecdh

parses_cv = []
for file in cvfiles:
    cv = read_mpt(file)
    cv = cv.loc[cv["cycle number"] == cv["cycle number"].max()]
    parses_cv.append(cv)

# %% Plotting CV
plot_cycles = [parses_cv[0], parses_cv[3], parses_cv[5]]
fig, ax = plt.subplots()
labels = ["Initial", "Post-EIS", "Re-balanced"]
for i, cycle in enumerate(plot_cycles):
    ax.plot(cycle["Ewe/V"], cycle["<I>/mA"], label=labels[i])
    ax.set_xlabel(r"$E / V$")
    ax.set_ylabel(r"$I / A$")

    ax.legend()
    ax.grid(visible=True)
fig.suptitle("5mM Dicarboxyferrocene, 1M NaCl")
plt.figure(fig)
plt.tight_layout()
plt.savefig(os.path.join(fig_directory, f"CVs.png"))

# %% Plotting EIS
plot_spectra = [parses_eis[2], parses_eis[5]]
unit_scale = ""  # If you want to swap to different scale. milli, kilo etc
fig, ax = plt.subplots()
for i, peis in enumerate(plot_spectra[0]):
    imp = peis.get_impedances()
    ax.scatter(imp.real, -imp.imag, label="Cycle #" + str(i))

    ax = set_aspect_ratio(ax, peis)
    ax = set_equal_tickspace(ax, figure=fig)
    ax.grid(visible=True)
    ax.set_xlabel(f"$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega$")
    ax.set_ylabel(f"$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega$")
    ax.legend()

# plt.tight_layout()
plt.gca().set_aspect("equal")

plt.savefig(os.path.join(fig_directory, f"InitialEIS.png"))
plt.show()

# %% Plotting EIS Multi
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

# %% Lin - KK ala pyimpspec
if not os.path.isfile(os.path.join(pkl_directory, "LinKKTests.pkl")):
    subject = plot_spectra[0][-1]
    mu_criterion = 0.85
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

fig, ax = plt.subplots()
for i, peis in enumerate([subject, tests[0]]):
    imp = peis.get_impedances()
    if plot_type[i] == "scatter":
        ax.scatter(imp.real, -imp.imag, color=colors[i], label=labels[i])
    elif plot_type[i] == "plot":
        ax.plot(imp.real, -imp.imag, color=colors[i], label=labels[i])

    ax = set_aspect_ratio(ax, peis)
    ax = set_equal_tickspace(ax, figure=fig)
    ax.grid(visible=True)
    ax.set_xlabel(f"$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega$")
    ax.set_ylabel(f"$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega$")
    ax.legend()

    line = True
# plt.tight_layout()
plt.gca().set_aspect("equal")
remove_legend_duplicate()

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
_, ax = pyi.mpl.plot_bode(subject, fig=fig, axes=[ax, ax])
pyi.mpl.plot_bode(zHit, fig=fig)
plt.show()
