# -*- coding: utf-8 -*-
"""
@author: Houlberg
"""
#%% Init
import numpy as np
import pandas as pd
import os
import sys
import time
import glob
import gc
import matplotlib.pyplot as plt
from matplotlib.axes._secondary_axes import SecondaryAxis
import matplotlib.ticker as ticker
#import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from cycler import cycler


from bayes_drt2.inversion import Inverter
import bayes_drt2.file_load as fl
import bayes_drt2.plotting as bp
import bayes_drt2.utils as utils

import pyimpspec as pyi
from pyimpspec.plot.mpl.utility import _configure_log_scale , _configure_log_limits
#from pyimpspec import mpl
#import eclabfiles as ecf
#import yadg
#import fnmatch

pd.options.mode.chained_assignment = None  # default='warn'
load_with_captions = True

#%%
#%% Move to plotting
#pickle / unpickle models
home = os.getcwd()
pkl_dir = os.path.join(home,r'Pickles')
sms = glob.glob(pkl_dir)
for file in sms:
    utils.load_pickle(file)
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

#%% Load Files and seperate by type. 
#REQUIRES exporting EC-Lab raw binarys (.mpr) as text (.mpt)
direc = r"C:\Users\Houlberg\Documents\Thesis\Python\ImpedanceDataOld\different flow rate_carbon felt_full-cell vanadium tests_1.6 M total vanadium_compression ratio=30%_choose the cycle number=1 or 3\Alt"
os.chdir(direc)

loadedfiles = []
loadedfilesnames = []
    
for file in os.listdir(direc):
    filename = os.fsdecode(file)
    if filename.endswith(".mpt"): 
        ident = os.path.basename(file).split('=')[1]
        ident_digit = ident.split(' ')[0]
        loadedfiles.append([filename,ident_digit])
        continue
    else:
        continue

#%% Parsing with pyimpspec
parsedfiles = []
for i,file in enumerate(loadedfiles):
    try:
        temp = pyi.parse_data(file[0],file_format=".mpt")
        parsedfiles.append([temp,file[1]])
    except:
        print("Error with file " + str(file))
#For this data set from Baichen we want the 1st cycle for all but 2cms, where we want the last
#[0,0,0] [1,0,0] [2,0,3] [3,0,0]
#%% Selecting the wanted cycles
chosen = [parsedfiles[0][0][0],parsedfiles[1][0][0],parsedfiles[2][0][3],parsedfiles[3][0][0]]
# Construct Pandas df because the package likes that
columns=['Freq','Zreal','Zimag','Zmod','Zphs']
dFs = []
for eis in chosen:
    dFs.append(eis.to_dataframe(columns=columns))
#%% Alt loading. Manual dataframe construction
data = np.loadtxt(loadedfiles[0][0],delimiter="\t",skiprows=76)
headers = open(loadedfiles[0][0]).readlines()[75].split("\t")
#headers = np.genfromtxt(loadedfiles[0][0],delimiter="\t",skiprows=75,dtype=str)
'''
Z_file = r"FTFF_ve=1cms.txt"
#Z_file = os.path.join(datadir,'FTFF_ve=1cms.txt')
Z_file
data = np.loadtxt(Z_file,delimiter="\t",skiprows=1)
dataclean = data[12:,] #Remove initial tail

# extract frequency and impedance
freq = data[:,0]
Z = data[:,1:3]
Z = Z[...,0] + -Z[...,1]*1j #convert to complex

freqclean = freq [12:,]
Zclean = Z[12:,]

# Construct Pandas df because the package likes that
df=pd.DataFrame(data=data[:,0:3],columns=['Freq','Zreal','Zimag'])
# Plot the data
#axes = bp.plot_eis(df)
'''
#%% Bayes_DRT fitting
# By default, the Inverter class is configured to fit the DRT (rather than the DDT)
# Create separate Inverter instances for HMC and MAP fits
# Set the basis frequencies equal to the measurement frequencies 
# (not necessary in general, but yields faster results here - see Tutorial 1 for more info on basis_freq)
neo = dFs[0]

mask = neo['Zimag'] > 0 
trinity = neo[~mask]

freq, Z= fl.get_fZ(trinity)

inv_hmc = Inverter(basis_freq=freq)
inv_map = Inverter(basis_freq=freq)
inv_hmcclean = Inverter(basis_freq=freq)
inv_mapclean = Inverter(basis_freq=freq)

#%% Perform HMC fit
start = time.time()
inv_hmc.fit(freq, Z, mode='sample')
elapsed = time.time() - start
print('HMC fit time {:.1f} s'.format(elapsed))

#%% Perform MAP fit
start = time.time()
inv_map.fit(freq, Z, mode='optimize')  # initialize from ridge solution
elapsed = time.time() - start
print('MAP fit time {:.1f} s'.format(elapsed))
#%% Visualize DRT and impedance fit
# plot impedance fit and recovered DRT
fig,axes = plt.subplots()

# plot fits of impedance data
inv_hmc.plot_fit(axes=axes, plot_type='nyquist', color='k', label='HMC fit', data_label='Data')
inv_map.plot_fit(axes=axes, plot_type='nyquist', color='r', label='MAP fit', plot_data=False)

plt.show()
# plot true DRT
#p = axes[1].plot(g_true['tau'],g_true['gamma'],label='True',ls='--')
# add Dirac delta function for RC element
#axes[1].plot([np.exp(-2),np.exp(-2)],[0,10],ls='--',c=p[0].get_color(),lw=1)

# Plot recovered DRT at given tau values
#tau_plot = g_true['tau'].values

#%%
fig,axes = plt.subplots()
inv_hmc.plot_distribution(ax=axes, color='k', label='HMC mean', ci_label='HMC 95% CI')
inv_map.plot_distribution(ax=axes, color='r', label='MAP')
sec_xaxis = [x for x in axes.get_children() if isinstance(x, SecondaryAxis)][0]
sec_xaxis.set_xlabel('$f$ / Hz')
#axes[1].set_ylim(0,3.5)
axes.legend()
#fig.tight_layout()
plt.show()

#%% Visualize the recovered error structure"
# For visual clarity, only MAP results are shown.
# HMC results can be obtained in the same way
fig, axes = plt.subplots(1,2,sharex=True)

# plot residuals and estimated error structure
inv_map.plot_residuals(axes=axes)

# plot true error structure in miliohms. Cant with this data
#p = axes[0].plot(freq, 3*Zdf['sigma_re'] * 1000, ls='--')
#axes[0].plot(freq, -3*Zdf['sigma_re'] * 1000, ls='--', c=p[0].get_color())
#axes[1].plot(freq, 3*Zdf['sigma_im'] * 1000, ls='--')
#axes[1].plot(freq, -3*Zdf['sigma_im'] * 1000, ls='--', c=p[0].get_color(), label='True $\pm 3\sigma$')

axes[1].legend()

fig.tight_layout()

#%% Peak fitting
# Only fit peaks that have a prominence of >= 5% of the estimated polarization resistance
inv_map.fit_peaks()

# plot the peak fit
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
inv_map.plot_peak_fit(ax=axes[0])  # Plot the overall peak fit
inv_map.plot_peak_fit(ax=axes[1], plot_individual_peaks=True)  # Plot the individual peaks

fig.tight_layout()

#%%
homie = os.path.dirname(__file__)
os.chdir(homie)
morty = os.path.join(homie,"pickles")
rick = os.path.join(morty,'TestPickle.pkl')
utils.save_pickle(inv_hmc,rick)