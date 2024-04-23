# -*- coding: utf-8 -*-
"""
@author: Houlberg
"""
# %% Init
import gc
import glob
import os
import time
import copy
import shutil

import bayes_drt2.file_load as fl
import bayes_drt2.utils as utils
import matplotlib.axes._secondary_axes
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pyimpspec as pyi
from bayes_drt2.inversion import Inverter
from cycler import cycler
from matplotlib.figure import Figure
from pyimpspec import DataSet

# from pyimpspec import mpl
# import eclabfiles as ecf
# import yadg
# import fnmatch

pd.options.mode.chained_assignment = None  # default='warn'
load_with_captions = True


# %% Define Custom Functions

def brute_eis(file):
    '''
    Manual approach to parsing .mpt eis files

    Parameter
    ---------
    file: path-like
        Str or bytes representing path to file
    Returns
    ---------
    data: np.ndarray
        Numpy array of the parsed data
    df: pd.DataFrame
        Pandas DataFrame using bayes_drt convention of column naming
        '''
    f = open(file, 'r')
    line_two = f.readlines()[1] #Line two will say: "Nb of header lines: XX"
    f.close()
    header_lines = int(np.array(line_two.split())[np.char.isnumeric(np.array(line_two.split()))].astype(int)) #Grabbing the numeric value of header lines
    data = np.genfromtxt(file, delimiter="\t", skip_header=int(header_lines))

    # Construct Pandas df because the bayes_drt package likes that
    df = pd.DataFrame(data=data[:, 0:5], columns=['Freq', 'Zreal', 'Zimag', 'Zmod', 'Zphs'])
    return data, df

def set_aspect_ratio(ax, dataset):
    '''
    Force the ratio between xmin, xmax, to be equal to ymin, ymax (and vice versa)
    Blatantly stolen from bayes_drt
    Adjusted for pyimpspec Datasets as compared to pandas DataFrames
    Careful with this function its quite buggy

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
    	Axes on which to plot
    dataset : pyimpspec.data.data_set.Dataset
        Pyimpspec DataSet containing the data to be plotted

    Returns
    -------
    None
    '''
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

    return



def read_mpt(filepath):
    """
    CV parser from ecdh
    Borrowed :) from ecdh https://github.com/amundmr/ecdh/blob/main/ecdh/readers/BioLogic.py

    Author: Amund M. Raniseth
    Reads an .mpt file to a pandas dataframe

    Parameter
    ---------
    filepath: path-like
        Str or bytes representing path to file
    Returns
    ---------
    df: pd.DataFrame
        Pandas DataFrame
    """
    modes = {1: "Galvanostatic", 2: "Linear Potential Sweep", 3: "Rest"}

    # with open(filepath, 'r', encoding= "iso-8859-1") as f:  #open the filepath for the mpt file
    #    lines = f.readlines()
    with open(filepath, errors='ignore') as f:
        lines = f.readlines()

    # now we skip all the data in the start and jump straight to the dense data
    headerlines = 0
    for line in lines:
        if "Nb header lines :" in line:
            headerlines = int(line.split(':')[-1])
            break  # breaks for loop when headerlines is found

    for i, line in enumerate(lines):
        # You wont find Characteristic mass outside of headerlines.
        if headerlines > 0:
            if i > headerlines:
                break

        if "Characteristic mass :" in line:
            active_mass = float(line.split(
                ':')[-1][:-3].replace(',', '.')) / 1000
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

    big_df = pd.read_csv(filepath, header=headerlines -
                                          whitelines - 1, sep="\t", encoding="ISO-8859-1")
    # print("Dataframe column names: {}".format(big_df.columns))

    # Start filling dataframe
    if 'I/mA' in big_df.columns:
        current_header = 'I/mA'
    elif '<I>/mA' in big_df.columns:
        current_header = '<I>/mA'
    df = big_df[['mode', 'time/s', 'Ewe/V',
                 current_header, 'cycle number', 'ox/red']]
    # Change headers of df to be correct
    df.rename(columns={current_header: '<I>/mA'}, inplace=True)

    # If it's galvanostatic we want the capacity
    mode = df['mode'].value_counts()  # Find occurences of modes
    # Remove the count of modes with rest (if there are large rests, there might be more rest datapoints than GC/CV steps)
    mode = mode[mode.index != 3]
    mode = mode.idxmax()  # Picking out the mode with the maximum occurences
    # print("Found cycling mode: {}".format(modes[mode]))

    if mode == 1:
        df = df.drt_hmc_pickle(big_df['Capacity/mA.h'])
        df.rename(columns={'Capacity/mA.h': 'capacity/mAhg'}, inplace=True)
        # the str.replace only works and is only needed if it is a string
        if df.dtypes['capacity/mAhg'] == str:
            df['capacity/mAhg'] = pd.to_numeric(
                df['capacity/mAhg'].str.replace(',', '.'))
    del big_df  # deletes the dataframe
    gc.collect()  # Clean unused memory (which is the dataframe above)
    # Replace , by . and make numeric from strings. Mode is already interpreted as int.
    for col in df.columns:
        if df.dtypes[col] == str:  # the str.replace only works and is only needed if it is a string
            df[col] = pd.to_numeric(df[col].str.replace(',', '.'))
    # df['time/s'] = pd.to_numeric(df['time/s'].str.replace(',','.'))
    # df['Ewe/V'] = pd.to_numeric(df['Ewe/V'].str.replace(',','.'))
    # df['<I>/mA'] = pd.to_numeric(df['<I>/mA'].str.replace(',','.'))
    # df['cycle number'] = pd.to_numeric(df['cycle number'].str.replace(',','.')).astype('int32')
    df.rename(columns={'ox/red': 'charge'}, inplace=True)
    df['charge'].replace({1: True, 0: False}, inplace=True)

    #    if mode == 2:
    # If it is CV data, then BioLogic counts the cycles in a weird way (starting new cycle when passing the point of the OCV, not when starting a charge or discharge..) so we need to count our own cycles

    #        df['cycle number'] = df['charge'].ne(df['charge'].shift()).cumsum()
    #        df['cycle number'] = df['cycle number'].floordiv(2)

    # Adding attributes must be the last thing to do since it will not be copied when doing operations on the dataframe
    df.experiment_mode = mode

    return df

def set_equal_tickspace(ax, figure=None):
    '''
    Adjusts x and y tick-spacing to be equal to the lower of the two

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        Axes to adjust tickspace on
    figure : matplotlib.figure.Figure, optional
        Figure containing the chosen Axes. The default is None.

    Returns
    -------
    None.

    '''
    assert isinstance(figure, Figure) or figure is None, figure
    if figure is None:
        assert ax is None
        figure, axis = plt.subplots()
        ax = [axis]

    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    xtickspace = xticks[1] - xticks[0]
    ytickspace = yticks[1] - yticks[0]

    spacing = min(xtickspace, ytickspace)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=spacing))

    return

def remove_legend_duplicate():
    #Removes legend duplicates
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
sms = glob.glob('../bayes-drt2-main/bayes_drt2/stan_model_files/*.pkl')
# sms = glob.glob('Pickles/*.pkl')
for file in sms:
    utils.load_pickle(file)

exp_dir = os.path.join(home, r'Experiments')
fig_dir = os.path.join(home, r'Figures')
pkl_dir = os.path.join(home, r'Pickles')
# pkl = os.path.join(pkl_dir,'obj{}.pkl'.format('MAP'))

tester = home + r'\Experiments'
# %% Selecting Experiment and establishing subfolders

experiment = 'IRFB/FirstCell'
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

if not os.path.isdir('Parsed'):
    os.makedirs('Parsed')

# %% Matplotlib parameters
# base = 'seaborn-v0_8' # -paper or -poster for smaller or larger figs ##Doesnt really work
# size = 'seaborn-v0_8-poster'
# ticks = 'seaborn-v0_8-ticks'
# plt.style.use([base,size,ticks])
bright_colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE',
                 '#AA3377', '#BBBBBB', '#000000']
bright_names = ['blue', 'red', 'green', 'yellow', 'cyan', 'purple', 'grey', 'black']
bright_dict = dict(zip(bright_names, bright_colors))

contrast_colors = ['#004488', '#DDAA33', '#BB5566', '#000000']
contrast_names = ['blue', 'yellow', 'red', 'black']
contrast_dict = dict(zip(contrast_names, contrast_colors))

vibrant_colors = ['#EE7733', '#0077BB', '#33BBEE', '#EE3377', '#CC3311',
                  '#009988', '#BBBBBB', '#000000']
vibrant_names = ['orange', 'blue', 'cyan', 'magenta', 'red', 'teal', 'grey', 'black']
vibrant_dict = dict(zip(vibrant_names, vibrant_colors))

seaborn_colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']

plt.rcParams['axes.prop_cycle'] = cycler(color=vibrant_colors)
plt.rcParams['patch.facecolor'] = vibrant_colors[0]

tick_size = 9
label_size = 11
# fig_width = 10
plt.rcParams['figure.figsize'] = [8, 5.5]
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

plt.rcParams['lines.markersize'] = 5
plt.rcParams['xtick.labelsize'] = tick_size
plt.rcParams['ytick.labelsize'] = tick_size
plt.rcParams['axes.labelsize'] = label_size
plt.rcParams["axes.axisbelow"] = True
plt.rcParams['legend.fontsize'] = tick_size + 1
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.95

# %% Load Files and seperate by type.
# REQUIRES exporting EC-Lab raw binarys (.mpr) as text (.mpt)
files = glob.glob('Data/*.mpt')

# %% Parsing with pyimpspec
dummy_freq = np.logspace(6, -2, 81)
dummy_Z = np.ones_like(dummy_freq, dtype='complex128')
pyi_dummy = DataSet(dummy_freq, dummy_Z)

parsedfiles = []
for file in files:
    string = 'Parsed/{}'.format(file.split('.')[0].split("\\")[1])
    if not os.path.isdir(string):
        os.makedirs(string)
        try:
            parse = pyi.parse_data(file, file_format=".mpt")
            parsedfiles.append(parse)
            for i, cycle in enumerate(parse):
                utils.save_pickle(cycle.to_dict(), os.path.join(  # .to_dict() might be unneccesary)
                    string, 'Cycle_{}.pkl'.format(i)))
        except:
            print("Pyimpspec could not parse file" + str(file))
    else:
        temp_list = []
        for i, cycle in enumerate(os.listdir(string)):
            pyi_pickle = utils.load_pickle(
                os.path.join(string, 'Cycle_{}.pkl'.format(i)))
            parse = pyi_dummy.from_dict(pyi_pickle)
            temp_list.append(parse)
        parsedfiles.append(temp_list)

# %% Selecting the wanted cycles
# For this data set from IRFB First Cell
# We want the 1st cycle for only LowFreq_Full
#parsedfiles[4][0]
pyi_columns = ['f', 'z_re', 'z_im', 'mod', 'phz']
chosen = []
chosen_masked = []
ident = ['25mHz-20ppd']

chosen.append(parsedfiles[4][0])
dummy = parsedfiles[4][0].to_dataframe(columns=pyi_columns)
mask = dummy['z_im'] < -0
chosen_nonzero = dummy[dummy['z_im'] < -0]
chosen_masked.append(pyi.dataframe_to_data_sets(chosen_nonzero, path=files[4]))
chosen_masked = list(zip(*chosen_masked))[0]
# Construct Pandas df because bayes_drt package likes that
columns = ['Freq', 'Zreal', 'Zimag', 'Zmod', 'Zphs']
dFs = []
dFs_masked = []
for eis in chosen:
    dFs.append(eis.to_dataframe(columns=columns))
for eis in chosen_masked:
    dFs_masked.append(eis.to_dataframe(columns=columns))

# %% Raw Nyquist with inset

unit_scale = ''
area = 2.3**2
subregions = [0.205, 0.255, -0.005, 0.045]
subregions_area = [np.multiply(subregions, area)]
for i, exp in enumerate(chosen):
    fig, ax = plt.subplots()
    imp = exp.get_impedances() * area
    ax.scatter(imp.real, -imp.imag)
    # ax = set_aspect_ratio(ax, chosen[0])
    set_equal_tickspace(ax, figure=fig)
    x1, x2, y1, y2 = subregions_area[i]  # Subregion

    ax_inset = ax.inset_axes([0.5, 0.1, 0.42, 0.42], xlim=(x1, x2), ylim=(y1, y2))
    ax_inset.scatter(imp.real, -imp.imag)
    ax.indicate_inset_zoom(ax_inset, edgecolor="black")

    set_equal_tickspace(ax_inset, figure=fig)
    #ax_inset.xaxis.set_major_locator(ticker.MultipleLocator(base=0.05))
    #ax_inset.yaxis.set_major_locator(ticker.MultipleLocator(base=0.05))
    ax.set_axisbelow(True)
    ax.grid(visible=True)
    ax_inset.set_axisbelow(True)
    ax_inset.grid(visible=True)


    #ax_inset = set_equal_tickspace(ax_inset, figure=fig)
    ax.set_xlabel(f'$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$')
    ax.set_ylabel(f'$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$')
    ax.legend()

    # plt.tight_layout()
    plt.gca().set_aspect('equal')
    remove_legend_duplicate()

    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_Nyquist_data_area_inset.png"))
    plt.show()
    plt.close()


# %% Raw Nyquist without inset

unit_scale = ''
area = 2.3**2
fig, ax = plt.subplots()
for i, exp in enumerate(chosen):
    imp = exp.get_impedances() * area
    ax.scatter(imp.real, -imp.imag, label=ident[i])
    # ax = set_aspect_ratio(ax, chosen[0])
    set_equal_tickspace(ax, figure=fig)

    ax.grid(visible=True)
    ax.set_xlabel(f'$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$')
    ax.set_ylabel(f'$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$')
    #ax.legend()

    # plt.tight_layout()
plt.gca().set_aspect('equal')
#remove_legend_duplicate()
#plt.legend(loc='best', bbox_to_anchor=(0., 0.5, 0.5, 0.5))
plt.savefig(os.path.join(fig_directory, str(ident[0]) + "_Nyquist_data_area.png"))
plt.show()
plt.close()

# %% Raw Nyquist without inset tail-removed

unit_scale = ''
area = 2.3**2  # Set to 1 for non-adjusted values
fig, ax = plt.subplots()
for i, exp in enumerate(chosen_masked):
    imp = exp.get_impedances() * area
    ax.scatter(imp.real, -imp.imag, label=ident[i])
    # ax = set_aspect_ratio(ax, chosen[0])
    set_equal_tickspace(ax, figure=fig)

    ax.grid(visible=True)
    ax.set_xlabel(f'$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$')
    ax.set_ylabel(f'$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega \mathrm{{cm^2}}$')
    #ax.legend()

    # plt.tight_layout()
plt.gca().set_aspect('equal')
#remove_legend_duplicate()
#plt.legend(loc='best', bbox_to_anchor=(0., 0.5, 0.5, 0.5))
plt.savefig(os.path.join(fig_directory, str(ident[0]) + "_Nyquist_data_area_nonzero.png"))
plt.show()
plt.close()
# %%
'''
#fig, ax = plt.subplots()
fig, ax = pyi.mpl.plot_nyquist(chosen[0])
plt.show()
#plt.savefig(os.path.join(fig_directory, f"Nyquist_data.png"))
'''
# %% Lin - KK ala pyimpspec
mu_criterion = 0.80
explorations = []
for i, eis in enumerate(chosen):
    # os.remove(os.path.join(pkl_directory, 'LinKKTests.pkl'))  # Remove pickle to rerun
    if not os.path.isfile(os.path.join(pkl_directory, str(ident[i]) + '_LinKKTests.pkl')):
        subject = eis
        mu_criterion = 0.80
        tests = pyi.perform_exploratory_tests(subject, mu_criterion=mu_criterion)
        utils.save_pickle(tests, os.path.join(pkl_directory, str(ident[i]) + '_LinKKTests.pkl'))
        explorations.append(tests[0])
    else:
        tests = utils.load_pickle(os.path.join(pkl_directory, str(ident[i]) + '_LinKKTests.pkl'))
    fig, ax = pyi.mpl.plot_mu_xps(tests, mu_criterion=mu_criterion)
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_ExploratoryTests.png"))
    plt.show()
    plt.close()

# %% Residual plot
# I personally prefer freq going high to low from left to right ("similar" as nyquist)
for i, test in enumerate(explorations):
    fig, ax = pyi.mpl.plot_residuals(test)
    plt.gca().invert_xaxis()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_LinKKResiduals.png"))
    plt.show()
    plt.close()

# %% Plotting Lin KK fitted EIS
unit_scale = ''  # If you want to swap to different scale. milli, kilo etc

for i, eis in enumerate(explorations):
    fig, ax = plt.subplots()
    imp = chosen[i].get_impedances()
    ax.scatter(imp.real, -imp.imag, color=vibrant_colors[0], label='Data')
    fit = eis.get_impedances()
    num_RC = eis.num_RC
    ax.plot(fit.real, -fit.imag, color=vibrant_colors[1], label="#(RC)={}".format(num_RC))

    # ax = set_aspect_ratio(ax, peis)
    set_equal_tickspace(ax, figure=fig)
    ax.grid(visible=True)
    ax.set_xlabel(f'$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega$')
    ax.set_ylabel(f'$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega$')
    ax.legend()

    # plt.tight_layout()
    plt.gca().set_aspect('equal')
    remove_legend_duplicate()

    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_LinKKImpedanceFit.png"))
    plt.show()
    plt.close()

# %% Bayes_DRT fitting
# By default, the Inverter class is configured to fit the DRT (rather than the DDT)
# Create separate Inverter instances for HMC and MAP fits
# Set the basis frequencies equal to the measurement frequencies
# (not necessary in general, but yields faster results here - see Tutorial 1 for more info on basis_freq)
drt_hmc_list = []
drt_map_list = []
drt_hmc_nonzero_list = []
drt_map_nonzero_list = []
ddt_hmc_list = []
ddt_map_list = []
ddt_hmc_nonzero_list = []
ddt_map_nonzero_list = []
inv_multi_hmc_list = []
inv_multi_map_list = []
inv_multi_hmc_nonzero_list = []
inv_multi_map_nonzero_list = []

for i, eis in enumerate(dFs):

    freq, Z = fl.get_fZ(eis)
    freq_nz, Z_nz = fl.get_fZ(dFs_masked[i])

    # DRT
    drt_hmc = Inverter(basis_freq=freq)
    drt_map = Inverter(basis_freq=freq)
    drt_hmc_nonzero = Inverter(basis_freq=freq_nz)
    drt_map_nonzero = Inverter(basis_freq=freq_nz)
    # DDT
    ddt_hmc = Inverter(distributions={'TP-DDT':  # user-defined distribution name
                                          {'kernel': 'DDT',  # indicates that a DDT-type kernel should be used
                                           'dist_type': 'parallel',
                                           # indicates that the diffusion paths are in parallel
                                           'symmetry': 'planar',  # indicates the geometry of the system
                                           'bc': 'transmissive',  # indicates the boundary condition
                                           'ct': False,  # indicates no simultaneous charge transfer
                                           'basis_freq': np.logspace(6, -3, 91)
                                           }
                                      },
                       basis_freq=np.logspace(6, -3, 91)  # use basis range large enough to capture full DDT
                       )
    ddt_map = copy.deepcopy(ddt_hmc)
    ddt_hmc_nonzero = copy.deepcopy(ddt_hmc)
    ddt_map_nonzero = copy.deepcopy(ddt_hmc)
    # Multi distribution
    # for sp_dr: use xp_scale=0.8
    inv_multi_hmc = Inverter(distributions={"DRT": {'kernel': 'DRT'},
                                            'TPP-DDT': {'kernel': 'DDT',
                                                        'symmetry': 'planar',
                                                        'bc': 'transmissive',
                                                        'dist_type': 'parallel',
                                                        'x_scale': 0.8,
                                                        'ct': False,  # indicates no simultaneous charge transfer
                                                        'basis_freq': np.logspace(6, -3, 91)
                                                        }
                                            }
                             )
    inv_multi_map = copy.deepcopy(inv_multi_hmc)
    inv_multi_hmc_nonzero = copy.deepcopy(inv_multi_hmc)
    inv_multi_map_nonzero = copy.deepcopy(inv_multi_hmc)

    # Perform HMC fit
    file_pickle = os.path.join(pkl_directory, '{0}_{1}.pkl'.format(ident[i],drt_hmc))
    file_pickle_core = os.path.join(pkl_directory, '{0}_{1}_core.pkl'.format(ident[i],drt_hmc))

    drt_hmc_pickle = os.path.join(pkl_directory, str(ident[i]) + '_drt_hmc.pkl')
    if not os.path.isfile(drt_hmc_pickle):
        start = time.time()
        drt_hmc.fit(freq, Z, mode='sample')
        elapsed = time.time() - start
        print('HMC fit time {:.1f} s'.format(elapsed))
        utils.save_pickle(drt_hmc, drt_hmc_pickle)
        drt_hmc.save_fit_data(drt_hmc_pickle + '_core', which='core')
        #Change to Inverter.save_fit_data(..., which= 'core') #load_pickle wont work
    else:
        drt_hmc.load_fit_data(drt_hmc_pickle)
        drt_hmc = utils.load_pickle(drt_hmc_pickle)


    # Perform HMC fit with initial tail removed
    if not os.path.isfile(os.path.join(pkl_directory, str(ident[i]) + '_drt_hmc_nz.pkl')):
        start = time.time()
        drt_hmc_nonzero.fit(freq_nz, Z_nz, mode='sample')
        elapsed = time.time() - start
        print('HMC fit time {:.1f} s'.format(elapsed))
        utils.save_pickle(drt_hmc_nonzero, os.path.join(pkl_directory, str(ident[i]) + '_drt_hmc_nz.pkl'))
    else:
        drt_hmc_nonzero = utils.load_pickle(os.path.join(pkl_directory, str(ident[i]) + '_drt_hmc_nz.pkl'))

    #  Perform MAP fit
    if not os.path.isfile(os.path.join(pkl_directory, str(ident[i]) + '_drt_map.pkl')):
        start = time.time()
        drt_map.fit(freq, Z, mode='optimize')  # initialize from ridge solution
        elapsed = time.time() - start
        print('MAP fit time {:.1f} s'.format(elapsed))
        utils.save_pickle(drt_map, os.path.join(pkl_directory, str(ident[i]) + '_drt_map.pkl'))
    else:
        drt_map = utils.load_pickle(os.path.join(pkl_directory, str(ident[i]) + '_drt_map.pkl'))

    #  Perform MAP fit with initial tail removed
    if not os.path.isfile(os.path.join(pkl_directory, str(ident[i]) + '_drt_map_nz.pkl')):
        start = time.time()
        drt_map_nonzero.fit(freq_nz, Z_nz, mode='sample')
        elapsed = time.time() - start
        print('MAP fit time {:.1f} s'.format(elapsed))
        utils.save_pickle(drt_map_nonzero, os.path.join(pkl_directory, str(ident[i]) + '_drt_map_nz.pkl'))
    else:
        drt_map_nonzero = utils.load_pickle(os.path.join(pkl_directory, str(ident[i]) + '_drt_map_nz.pkl'))

    # Perform DDT HMC fit
    if not os.path.isfile(os.path.join(pkl_directory, str(ident[i]) + '_ddt_hmc.pkl')):
        start = time.time()
        ddt_hmc.fit(freq, Z, mode='sample')
        elapsed = time.time() - start
        print('HMC fit time {:.1f} s'.format(elapsed))
        utils.save_pickle(ddt_hmc, os.path.join(pkl_directory, str(ident[i]) + '_ddt_hmc.pkl'))
    else:
        ddt_hmc = utils.load_pickle(os.path.join(pkl_directory, str(ident[i]) + '_ddt_hmc.pkl'))

    # Perform DDT HMC fit with initial tail removed
    if not os.path.isfile(os.path.join(pkl_directory, str(ident[i]) + '_ddt_hmc_nz.pkl')):
        start = time.time()
        ddt_hmc_nonzero.fit(freq_nz, Z_nz, mode='sample')
        elapsed = time.time() - start
        print('HMC fit time {:.1f} s'.format(elapsed))
        utils.save_pickle(ddt_hmc_nonzero, os.path.join(pkl_directory, str(ident[i]) + '_ddt_hmc_nz.pkl'))
    else:
        ddt_hmc_nonzero = utils.load_pickle(os.path.join(pkl_directory, str(ident[i]) + '_ddt_hmc_nz.pkl'))

    #  Perform DDT MAP fit
    if not os.path.isfile(os.path.join(pkl_directory, str(ident[i]) + '_ddt_map.pkl')):
        start = time.time()
        ddt_map.fit(freq, Z, mode='optimize')  # initialize from ridge solution
        elapsed = time.time() - start
        print('MAP fit time {:.1f} s'.format(elapsed))
        utils.save_pickle(ddt_map, os.path.join(pkl_directory, str(ident[i]) + '_ddt_map.pkl'))
    else:
        ddt_map = utils.load_pickle(os.path.join(pkl_directory, str(ident[i]) + '_ddt_map.pkl'))

    #  Perform DDT MAP fit with initial tail removed
    if not os.path.isfile(os.path.join(pkl_directory, str(ident[i]) + '_ddt_map_nz.pkl')):
        start = time.time()
        ddt_map_nonzero.fit(freq_nz, Z_nz, mode='sample')
        elapsed = time.time() - start
        print('MAP fit time {:.1f} s'.format(elapsed))
        utils.save_pickle(ddt_map_nonzero, os.path.join(pkl_directory, str(ident[i]) + '_ddt_map_nz.pkl'))
    else:
        ddt_map_nonzero = utils.load_pickle(os.path.join(pkl_directory, str(ident[i]) + '_ddt_map_nz.pkl'))

    # Perform DRT+DDT HMC fit
    if not os.path.isfile(os.path.join(pkl_directory, str(ident[i]) + '_inv_multi_hmc.pkl')):
        start = time.time()
        inv_multi_hmc.fit(freq, Z, mode='sample', nonneg=True)
        elapsed = time.time() - start
        print('HMC fit time {:.1f} s'.format(elapsed))
        utils.save_pickle(inv_multi_hmc, os.path.join(pkl_directory, str(ident[i]) + '_inv_multi_hmc.pkl'))
    else:
        inv_multi_hmc = utils.load_pickle(os.path.join(pkl_directory, str(ident[i]) + '_inv_multi_hmc.pkl'))

    #  Perform DRT+DDT MAP fit
    if not os.path.isfile(os.path.join(pkl_directory, str(ident[i]) + '_inv_multi_map.pkl')):
        start = time.time()
        inv_multi_map.fit(freq, Z, mode='optimize', nonneg=True)  # initialize from ridge solution
        elapsed = time.time() - start
        print('MAP fit time {:.1f} s'.format(elapsed))
        utils.save_pickle(inv_multi_map, os.path.join(pkl_directory, str(ident[i]) + '_inv_multi_map.pkl'))
    else:
        inv_multi_map = utils.load_pickle(os.path.join(pkl_directory, str(ident[i]) + '_inv_multi_map.pkl'))

    # Perform DRT+DDT HMC fit with initial tail removed
    if not os.path.isfile(os.path.join(pkl_directory, str(ident[i]) + '_inv_multi_hmc_nz.pkl')):
        start = time.time()
        inv_multi_hmc_nonzero.fit(freq_nz, Z_nz, mode='sample', nonneg=True)
        elapsed = time.time() - start
        print('HMC fit time {:.1f} s'.format(elapsed))
        utils.save_pickle(inv_multi_hmc, os.path.join(pkl_directory, str(ident[i]) + '_inv_multi_hmc_nz.pkl'))
    else:
        inv_multi_hmc = utils.load_pickle(os.path.join(pkl_directory, str(ident[i]) + '_inv_multi_hmc_nz.pkl'))

    #  Perform DRT+DDT MAP fit with initial tail removed
    if not os.path.isfile(os.path.join(pkl_directory, str(ident[i]) + '_inv_multi_map_nz.pkl')):
        start = time.time()
        inv_multi_map_nonzero.fit(freq_nz, Z_nz, mode='optimize', nonneg=True)  # initialize from ridge solution
        elapsed = time.time() - start
        print('MAP fit time {:.1f} s'.format(elapsed))
        utils.save_pickle(inv_multi_map_nonzero, os.path.join(pkl_directory, str(ident[i]) + '_inv_multi_map_nz.pkl'))
    else:
        inv_multi_map_nonzero = utils.load_pickle(os.path.join(pkl_directory, str(ident[i]) + '_inv_multi_map_nz.pkl'))

    drt_hmc_list.append(drt_hmc)
    drt_map_list.append(drt_map)
    drt_hmc_nonzero_list.append(drt_hmc_nonzero)
    drt_map_nonzero_list.append(drt_map_nonzero)
    ddt_hmc_list.append(ddt_hmc)
    ddt_map_list.append(ddt_map)
    ddt_hmc_nonzero_list.append(ddt_hmc)
    ddt_map_nonzero_list.append(ddt_map)
    inv_multi_hmc_list.append(inv_multi_hmc)
    inv_multi_map_list.append(inv_multi_map)
    inv_multi_hmc_nonzero_list.append(inv_multi_hmc)
    inv_multi_map_nonzero_list.append(inv_multi_map)

# %% Visualize DRT and impedance fit
# plot impedance fit and recovered DRT

for i, exp in enumerate(dFs):
    fig, axes = plt.subplots()

    drt_hmc_list[i].plot_fit(axes=axes, plot_type='nyquist', color='k', label='HMC fit', data_label='Data')
    drt_map_list[i].plot_fit(axes=axes, plot_type='nyquist', color='r', label='MAP fit', plot_data=False)
    # ax = set_aspect_ratio(ax, peis)
    #axes = set_equal_tickspace(axes, figure=fig)
    axes.grid(visible=True)
    axes.set_xlabel(f'$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega$')
    axes.set_ylabel(f'$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega$')
    axes.legend()
    # plt.tight_layout()
    plt.gca().set_aspect('equal')
    remove_legend_duplicate()
    plt.figure(fig)
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT_FitImpedance.png"))
    plt.close()

    ddt_hmc_list[i].plot_fit(axes=axes, plot_type='nyquist', color='k', label='HMC fit', data_label='Data')
    ddt_map_list[i].plot_fit(axes=axes, plot_type='nyquist', color='r', label='MAP fit', plot_data=False)
    # ax = set_aspect_ratio(ax, peis)
    #axes = set_equal_tickspace(axes, figure=fig)
    axes.grid(visible=True)
    axes.set_xlabel(f'$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega$')
    axes.set_ylabel(f'$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega$')
    axes.legend()
    # plt.tight_layout()
    plt.gca().set_aspect('equal')
    remove_legend_duplicate()
    plt.figure(fig)
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DDT_FitImpedance.png"))
    plt.close()

    inv_multi_hmc_list[i].plot_fit(axes=axes, plot_type='nyquist', color='k', label='HMC fit', data_label='Data')
    inv_multi_map_list[i].plot_fit(axes=axes, plot_type='nyquist', color='r', label='MAP fit', plot_data=False)
    # ax = set_aspect_ratio(ax, peis)
    #axes = set_equal_tickspace(axes, figure=fig)
    axes.grid(visible=True)
    axes.set_xlabel(f'$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega$')
    axes.set_ylabel(f'$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega$')
    axes.legend()
    # plt.tight_layout()
    plt.gca().set_aspect('equal')
    remove_legend_duplicate()
    plt.figure(fig)
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT-DDT_FitImpedance.png"))
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
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT-DDT.png"))
    plt.close()

    #tau_drt_nz = drt_map_nonzero_list[i].distributions['DRT']['tau']
    tau_drt_nz = None
    fig, axes = plt.subplots()
    drt_hmc_nonzero_list[i].plot_distribution(ax=axes, tau_plot=tau_drt_nz, color='k', label='HMC mean', ci_label='HMC 95% CI')
    drt_map_nonzero_list[i].plot_distribution(ax=axes, tau_plot=tau_drt_nz, color='r', label='MAP')
    sec_xaxis = [x for x in axes.get_children() if isinstance(x, matplotlib.axes._secondary_axes.SecondaryAxis)][0]
    sec_xaxis.set_xlabel('$f$ / Hz')
    axes.legend()
    plt.figure(fig)
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT_nonzero.png"))
    plt.close()

    #tau_ddt_nz = ddt_map_nonzero_list[i].distributions['TP-DDT']['tau']
    tau_ddt_nz = None
    fig, axes = plt.subplots()
    ddt_hmc_nonzero_list[i].plot_distribution(ax=axes, tau_plot=tau_ddt_nz, color='k', label='HMC mean', ci_label='HMC 95% CI')
    ddt_map_nonzero_list[i].plot_distribution(ax=axes, tau_plot=tau_ddt_nz, color='r', label='MAP')
    sec_xaxis = [x for x in axes.get_children() if isinstance(x, matplotlib.axes._secondary_axes.SecondaryAxis)][0]
    sec_xaxis.set_xlabel('$f$ / Hz')
    axes.legend()
    plt.figure(fig)
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DDT_nonzero.png"))
    plt.close()

    #tau_multi_nz = inv_multi_map_nonzero_list[i].distributions['DRT']['tau']
    tau_multi_nz = None
    fig, axes = plt.subplots()
    inv_multi_hmc_nonzero_list[i].plot_distribution(ax=axes, tau_plot=tau_multi_nz, color='k', label='HMC mean', ci_label='HMC 95% CI')
    inv_multi_map_nonzero_list[i].plot_distribution(ax=axes, tau_plot=tau_multi_nz, color='r', label='MAP')
    sec_xaxis = [x for x in axes.get_children() if isinstance(x, matplotlib.axes._secondary_axes.SecondaryAxis)][0]
    sec_xaxis.set_xlabel('$f$ / Hz')
    axes.legend()
    plt.figure(fig)
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT-DDT_nonzero.png"))
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
    drt_hmc_nonzero_list[i].plot_residuals(axes=axes)
    plt.figure(fig)
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT_HMC_Residuals_nonzero.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    ddt_hmc_list[i].plot_residuals(axes=axes)
    # fig.suptitle("Bayes Estimated Error Structure")
    plt.figure(fig)
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DDT_HMC_Residuals.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    ddt_hmc_nonzero_list[i].plot_residuals(axes=axes)
    plt.figure(fig)
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DDT_HMC_Residuals_nonzero.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    inv_multi_hmc_list[i].plot_residuals(axes=axes)
    # fig.suptitle("Bayes Estimated Error Structure")
    plt.figure(fig)
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT-DDT_HMC_Residuals.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    inv_multi_hmc_nonzero_list[i].plot_residuals(axes=axes)
    plt.figure(fig)
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT-DDT_HMC_Residuals_nonzero.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    drt_map_list[i].plot_residuals(axes=axes)
    # fig.suptitle("Bayes Estimated Error Structure")
    plt.figure(fig)
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT_MAP_Residuals.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    drt_map_nonzero_list[i].plot_residuals(axes=axes)
    plt.figure(fig)
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT_MAP_Residuals_nonzero.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    ddt_map_list[i].plot_residuals(axes=axes)
    # fig.suptitle("Bayes Estimated Error Structure")
    plt.figure(fig)
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DDT_MAP_Residuals.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    ddt_map_nonzero_list[i].plot_residuals(axes=axes)
    plt.figure(fig)
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DDT_MAP_Residuals_nonzero.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    inv_multi_map_list[i].plot_residuals(axes=axes)
    # fig.suptitle("Bayes Estimated Error Structure")
    plt.figure(fig)
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT-DDT_MAP_Residuals.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)
    inv_multi_map_nonzero_list[i].plot_residuals(axes=axes)
    plt.figure(fig)
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT-DDT_MAP_Residuals_nonzero.png"))
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
    fig.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT_MAP_PeakFits.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    ddt_map_list[i].fit_peaks(prom_rthresh=0.05)
    ddt_map_list[i].plot_peak_fit(ax=axes[0])  # Plot the overall peak fit
    ddt_map_list[i].plot_peak_fit(ax=axes[1], plot_individual_peaks=True)  # Plot the individual peaks
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.figure(fig)
    fig.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DDT_MAP_PeakFits.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    inv_multi_map_list[i].fit_peaks(prom_rthresh=0.05)
    inv_multi_map_list[i].plot_peak_fit(ax=axes[0])  # Plot the overall peak fit
    inv_multi_map_list[i].plot_peak_fit(ax=axes[1], plot_individual_peaks=True)  # Plot the individual peaks
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.figure(fig)
    fig.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT-DDT_MAP_PeakFits.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    try:
        drt_map_nonzero_list[i].fit_peaks(prom_rthresh=0.05)
        drt_map_nonzero_list[i].plot_peak_fit(ax=axes[0])  # Plot the overall peak fit
        drt_map_nonzero_list[i].plot_peak_fit(ax=axes[1], plot_individual_peaks=True)  # Plot the individual peaks
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.figure(fig)
        fig.tight_layout()
        plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT_MAP_PeakFits_nonzero.png"))
        plt.close()
    except:
        print("Error with file " + str(chosen[i]) + "For DRT Non-zero Peak Fitting")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    ddt_map_list[i].fit_peaks(prom_rthresh=0.05)
    ddt_map_list[i].plot_peak_fit(ax=axes[0])  # Plot the overall peak fit
    ddt_map_list[i].plot_peak_fit(ax=axes[1], plot_individual_peaks=True)  # Plot the individual peaks
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.figure(fig)
    fig.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DDT_MAP_PeakFits_nonzero.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    inv_multi_map_list[i].fit_peaks(prom_rthresh=0.05)
    inv_multi_map_list[i].plot_peak_fit(ax=axes[0])  # Plot the overall peak fit
    inv_multi_map_list[i].plot_peak_fit(ax=axes[1], plot_individual_peaks=True)  # Plot the individual peaks
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.figure(fig)
    fig.tight_layout()
    plt.savefig(os.path.join(fig_directory, str(ident[i]) + "_DRT-DDT_MAP_PeakFits_nonzero.png"))
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
