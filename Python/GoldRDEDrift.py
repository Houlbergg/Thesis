# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 08:17:52 2024

@author: Houlberg
"""

#%% Init
import numpy as np
import pandas as pd
import os
#import sys
#import time
import gc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from cycler import cycler


#from bayes_drt2.inversion import Inverter
#import bayes_drt2.file_load as fl
#import bayes_drt2.plotting as bp

import pyimpspec as pyi
#from pyimpspec import mpl
#import eclabfiles as ecf
#import yadg
#import fnmatch

pd.options.mode.chained_assignment = None  # default='warn'

#%% Matplotlib parameters
#base = 'seaborn-v0_8' # -paper or -poster for smaller or larger figs ##Doesnt really work
#size = 'seaborn-v0_8-poster' 
#ticks = 'seaborn-v0_8-ticks'
#plt.style.use([base,size,ticks])

colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
plt.rcParams['patch.facecolor'] = colors[0]

tick_size = 9
label_size = 11
#fig_width = 10
plt.rcParams['figure.figsize'] = [8,5.5]
plt.rcParams['figure.dpi'] = 200    
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

plt.rcParams['lines.markersize'] = 5
plt.rcParams['xtick.labelsize'] = tick_size
plt.rcParams['ytick.labelsize'] = tick_size
plt.rcParams['axes.labelsize'] = label_size
plt.rcParams['legend.fontsize'] = tick_size + 1
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.95


#%% Aspect Ratio function
#Blatantly stolen from bayes_drt
#Adjusted for pyimpspec Datasets as compared to pandas DataFrames
def set_aspect_ratio(ax,dataset):
    
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
        #width = height
        #fig.set_figwidth(width)
        #diff = (yscale - xscale) * width
        #xmin = max(0, ax.get_xlim()[0] - diff / 2)
        #mindelta = ax.get_xlim()[0] - xmin
        #xmax = ax.get_xlim()[1] + diff - mindelta

        ax.set_xlim(ax.get_ylim()[0], ax.get_ylim()[1])
        
        #bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        #width, height = bbox.width, bbox.height
        
        #width = height
        #fig.set_figwidth(width)
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

#%% Equal tickspace function
#Homemade
def set_equal_tickspace(ax, figure = None):
    assert isinstance(figure, Figure) or figure is None, figure
    if figure is None:
        assert ax is None
        figure, axis = plt.subplots()
        ax = [axis]
    
    xticks = plt.xticks()[0]
    yticks = plt.yticks()[0]
    xtickspace = xticks[1] - xticks[0]
    ytickspace = yticks[1] - yticks[0]

    spacing = min(xtickspace,ytickspace)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=spacing))
    
    return ax

#%% Load Files and seperate by type. 
#REQUIRES exporting EC-Lab raw binarys (.mpr) as text (.mpt)
direc = r"C:\Users\Houlberg\Documents\Thesis\Python\Data\GoldElectrode\240319\Gold"
os.chdir(direc)

loadedfiles = []
loadedfilesnames = []
    
for file in os.listdir(direc):
    filename = os.fsdecode(file)
    if filename.endswith(".mpt"): 
        loadedfiles.append(filename)
        loadedfilesnames.append(os.path.basename(file).split('.')[0])
        continue
    else:
        continue

peisfiles_act = []
cvfiles_act = []
peisfiles_post = []
cvfiles_post = []

for file in loadedfiles:
    filename = os.fsdecode(file)
    if "_Activation" in filename:
        if "_PEIS_" in filename:
            peisfiles_act.append(file)
        elif "-EIS_" in filename:
            peisfiles_act.append(file)
        elif "_CV_" in filename:
            cvfiles_act.append(file)
        elif "-CV_" in filename:
            cvfiles_act.append(file)
    elif "PostActivation" in filename:
        if "_PEIS_" in filename:
            peisfiles_post.append(file)
        elif "-EIS_" in filename:
            peisfiles_post.append(file)
        elif "_CV_" in filename:
            cvfiles_post.append(file)
        elif "-CV_" in filename:
            cvfiles_post.append(file)        
#Parse data with ydag
#yadg.parsers.electrochem.process(cvfiles[0],filetype=".mpt")
#%% CV parser from ecdh
#Borrowed :) from ecdh https://github.com/amundmr/ecdh/blob/main/ecdh/readers/BioLogic.py
def read_mpt(filepath):
    """
    Author: Amund M. Raniseth
    Reads an mpt file to a pandas dataframe
    
    .MPT format:
        mode column: 1=Galvanostatic, 2=Linear Potential Sweep, 3=Rest

    """
    modes = {1: "Galvanostatic", 2 : "Linear Potential Sweep", 3 : "Rest"}

    #with open(filepath, 'r', encoding= "iso-8859-1") as f:  #open the filepath for the mpt file
    #    lines = f.readlines()
    with open(filepath, errors = 'ignore') as f:
        lines = f.readlines()

    # now we skip all the data in the start and jump straight to the dense data
    headerlines = 0
    for line in lines:
        if "Nb header lines :" in line:
            headerlines = int(line.split(':')[-1])
            break #breaks for loop when headerlines is found
    
    for i,line in enumerate(lines):
        #You wont find Characteristic mass outside of headerlines.
        if headerlines > 0:
            if i > headerlines:
                break
        
        if "Characteristic mass :" in line:
            active_mass = float(line.split(':')[-1][:-3].replace(',', '.'))/1000
            #print("Active mass found in file to be: " + str(active_mass) + "g")
            break #breaks loop when active mass is found

    # pandas.read_csv command automatically skips white lines, meaning that we need to subtract this from the amout of headerlines for the function to work.
    whitelines = 0
    for i, line in enumerate(lines):
        #we dont care about outside of headerlines.
        if headerlines > 0:
            if i > headerlines:
                break
        if line == "\n":
            whitelines += 1

    #Remove lines object
    del lines
    gc.collect()


    big_df = pd.read_csv(filepath, header=headerlines-whitelines-1, sep = "\t", encoding = "ISO-8859-1")
    #print("Dataframe column names: {}".format(big_df.columns))

    # Start filling dataframe
    if 'I/mA' in big_df.columns:
        current_header = 'I/mA'
    elif '<I>/mA' in big_df.columns:
        current_header = '<I>/mA'
    df = big_df[['mode', 'time/s', 'Ewe/V', current_header, 'cycle number', 'ox/red']]
    # Change headers of df to be correct
    df.rename(columns={current_header: '<I>/mA'}, inplace=True)

    # If it's galvanostatic we want the capacity
    mode = df['mode'].value_counts() #Find occurences of modes
    mode = mode[mode.index != 3] #Remove the count of modes with rest (if there are large rests, there might be more rest datapoints than GC/CV steps)
    mode = mode.idxmax()    # Picking out the mode with the maximum occurences
    #print("Found cycling mode: {}".format(modes[mode]))

    if mode == 1:
        df = df.join(big_df['Capacity/mA.h'])
        df.rename(columns={'Capacity/mA.h': 'capacity/mAhg'}, inplace=True)
        if df.dtypes['capacity/mAhg'] == str: #the str.replace only works and is only needed if it is a string
            df['capacity/mAhg'] = pd.to_numeric(df['capacity/mAhg'].str.replace(',','.'))
    del big_df #deletes the dataframe
    gc.collect() #Clean unused memory (which is the dataframe above)
    # Replace , by . and make numeric from strings. Mode is already interpreted as int.
    for col in df.columns:
        if df.dtypes[col] == str: #the str.replace only works and is only needed if it is a string
            df[col] = pd.to_numeric(df[col].str.replace(',','.'))
    #df['time/s'] = pd.to_numeric(df['time/s'].str.replace(',','.'))
    #df['Ewe/V'] = pd.to_numeric(df['Ewe/V'].str.replace(',','.'))
    #df['<I>/mA'] = pd.to_numeric(df['<I>/mA'].str.replace(',','.'))
    #df['cycle number'] = pd.to_numeric(df['cycle number'].str.replace(',','.')).astype('int32')
    df.rename(columns={'ox/red': 'charge'}, inplace=True)
    df['charge'].replace({1: True, 0: False}, inplace = True)

#    if mode == 2:
        # If it is CV data, then BioLogic counts the cycles in a weird way (starting new cycle when passing the point of the OCV, not when starting a charge or discharge..) so we need to count our own cycles
        
#        df['cycle number'] = df['charge'].ne(df['charge'].shift()).cumsum()
#        df['cycle number'] = df['cycle number'].floordiv(2)

    #Adding attributes must be the last thing to do since it will not be copied when doing operations on the dataframe
    df.experiment_mode = mode
    
    return df
#%% Parse data with pyimpspec and ecdh

parsescv_act = []
parseseis_act = []
parsescv_post = []
parseseis_post = []

for file in cvfiles_act:
    try:
        f = read_mpt(file)
        parsescv_act.append(f)
    except:
        print("Error with file " + str(file))
for file in cvfiles_post:
    try:
        f = read_mpt(file)
        parsescv_post.append(f)
    except:
        print("Error with file " + str(file))
for file in peisfiles_act:
    try:
        parseseis_act.append(pyi.parse_data(file,file_format=".mpt"))
    except:
        print("Error with file " + str(file))
for file in peisfiles_post:
    try:
        parseseis_post.append(pyi.parse_data(file,file_format=".mpt"))
    except:
        print("Error with file " + str(file))

cyclenums = [1, 20, 40, 60, 80]
#No clue why some files dont work

#%% Plotting Directory
if not os.path.isdir('Figures/'):
    os.makedirs('Figures/')
#%% Selecting CV cycles to plot Activation
#Using the first cycle in the first CV file + last cycle from all files
toplots_cv_act = []
for cv in parsescv_act:
    finalcycle = cv.loc[cv['cycle number'] == cv['cycle number'].max()]
    toplots_cv_act.append(finalcycle)


#%% CV Plotting Activation

fig, ax = plt.subplots()
for i,cycle in enumerate(toplots_cv_act):
    num = cyclenums[i]
    ax.plot(cycle['Ewe/V'],cycle['<I>/mA'],label="Cycle " +str(num) )
    ax.set_xlabel(r'$E / V$')
    ax.set_ylabel(r'$I / A$')

    ax.legend()
    ax.grid(visible=True)
plt.figure(fig)
plt.tight_layout()
plt.savefig("Figures\ActivationCV.png")

#%% Selecting EIS spectra to plot Activation

toplots_eis_act = []
for eis in parseseis_act:
    toplots_eis_act.append(eis[0])

#%% EIS Plotting Activation

unit_scale = '' #If you want to swap to different scale. milli, kilo etc
fig, ax = plt.subplots()
for i,peis in enumerate(toplots_eis_act):
    imp = peis.get_impedances()
    ax.scatter(imp.real,-imp.imag,label="After " + str(cyclenums[i]) + " CV Cycles")

    ax = set_aspect_ratio(ax,peis)
    ax = set_equal_tickspace(ax,figure=fig)
    ax.grid(visible=True)
    ax.set_xlabel(f'$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega$')
    ax.set_ylabel(f'$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega$')
    ax.legend()

#plt.tight_layout()
plt.gca().set_aspect('equal')
    
plt.savefig("Figures\ActivationEIS.png")
plt.show()


#%% Selecting CV cycles to plot Post

toplots_cv_post = []
for cv in parsescv_post:
    finalcycle = cv.loc[cv['cycle number'] == cv['cycle number'].max()]
    toplots_cv_post.append(finalcycle)

#%% CV Plotting Post

fig, ax = plt.subplots()
rpms = [0, 225, 625, 1225, 2000]
for i,cycle in enumerate(toplots_cv_post):
    ax.plot(cycle['Ewe/V'],cycle['<I>/mA'],label="RPM " +str(rpms[i]) )
    ax.legend()
    ax.grid(visible=True)
plt.savefig("Figures\PostActivationCV.png")

#%% Selecting EIS spectra to plot Post

toplots_eis_post = []
for eis in parseseis_post:
    toplots_eis_post.append(eis[0])

#%% EIS Plotting Post
unit_scale = ''
fig, ax = plt.subplots()
for i,peis in enumerate(toplots_eis_post):
    imp = peis.get_impedances()
    ax.scatter(imp.real,-imp.imag,label="RPM" + str(rpms[i]))

    ax = set_aspect_ratio(ax,peis)
    ax = set_equal_tickspace(ax,figure=fig)          
    ax.grid(visible=True)
    ax.set_xlabel(f'$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega$')
    ax.set_ylabel(f'$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega$')
    ax.legend()

#plt.tight_layout()
plt.gca().set_aspect('equal')
    
plt.savefig("Figures\PostActivationEIS.png")
plt.show()

#%% The Fast and the Furious: Tokyo Drift

toplots_eis_drift = parseseis_post[0]

unit_scale = ''
fig, ax = plt.subplots()
for i,peis in enumerate(toplots_eis_drift):
    imp = peis.get_impedances()
    ax.scatter(imp.real,-imp.imag,label="Scan " + str(i))

    ax = set_aspect_ratio(ax,peis)
    ax = set_equal_tickspace(ax,figure=fig)          
    ax.grid(visible=True)
    ax.set_xlabel(f'$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega$')
    ax.set_ylabel(f'$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega$')
    ax.legend()

#plt.tight_layout()
plt.gca().set_aspect('equal')
    
plt.savefig("Figures\DriftEIS.png")
plt.show()

#%% Lin - KK ala pyimpspec

EXAMPLE = parseseis_post[0][2]
mu_criterion = 0.85
tests = pyi.perform_exploratory_tests(EXAMPLE, mu_criterion=mu_criterion)

figure, axes = pyi.mpl.plot_mu_xps(tests, mu_criterion=mu_criterion)
figure.tight_layout()

figure_log_limits(
    axis: "Axes",  # noqa: F821
    x: bool = False,
    y: bool = False,
):
    if x:
        x_min: float
        x_max: float
        x_min, x_max = axis.get_xlim()
        if np.log10(x_max) - np.log10(x_min) < 1.0:
            x_min = 10 ** np.floor(np.log10(x_min))
            x_max = 10 ** np.ceil(np.log10(x_max))
            axis.set_xlim(x_min, x_max)
    if y:
        y_min: float
        y_max: float
        y_min, y_max = axis.get_ylim()
        if np.log10(y_max) - np.log10(y_min) < 1.0:
            y_min = 10 ** np.floor(np.log10(y_min))
            y_max = 10 ** np.ceil(np.log10(y_max))
            axis.set_ylim(y_min, y_max)
def _configure_log_scale(
    axis: "Axes",  # noqa: F821
    x: bool = False,
    y: bool = False,
):
    from matplotlib.ticker import FormatStrFormatter

    formatter: FormatStrFormatter = FormatStrFormatter("")
    if x:
        global teste 
        teste = axis.get_xlim()
        axis.set_xscale("log")
        axis.xaxis.set_minor_formatter(formatter)
        global tester 
        tester = axis.get_xlim()
    if y:
        axis.set_yscale("log")
        axis.yaxis.set_minor_formatter(formatter)


#%%
testi = tests[0]
testo = testi.get_residuals_data()
#%%
figure, axes = pyi.mpl.plot_residuals(tests[0])

#%%
figure, axes = pyi.mpl.plot_nyquist(tests[0], line=True)
_ = pyi.mpl.plot_nyquist(EXAMPLE, figure=figure, axes=axes, colors={"impedance": "black"})
figure.tight_layout()