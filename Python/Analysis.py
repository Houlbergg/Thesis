# -*- coding: utf-8 -*-
"""
@author: Houlberg
"""
# %% Init
import numpy as np
import pandas as pd
import os
import sys
import time
import glob
import gc
import matplotlib.pyplot as plt
import matplotlib.axes._secondary_axes
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from cycler import cycler
from bayes_drt2.inversion import Inverter
import bayes_drt2.file_load as fl
import bayes_drt2.plotting as bp
import bayes_drt2.utils as utils

import pyimpspec as pyi
from pyimpspec import DataSet
from pyimpspec.plot.mpl.utility import _configure_log_scale, _configure_log_limits

# from pyimpspec import mpl
# import eclabfiles as ecf
# import yadg
# import fnmatch

pd.options.mode.chained_assignment = None  # default='warn'
load_with_captions = True

#%% Define Custom Functions
def brute_eis(file):
    f = open(file, 'r')
    line_two = f.readlines()[1]
    f.close()
    header_lines = int(np.array(line_two.split())[np.char.isnumeric(np.array(line_two.split()))].astype(int))
    data = np.genfromtxt(file, delimiter="\t", skip_header=int(header_lines))

    # Construct Pandas df because the package likes that
    df = pd.DataFrame(data=data[:, 0:5], columns=['Freq', 'Zreal', 'Zimag', 'Zmod', 'Zphs'])
    return data, df    # I realize this is silly
# %% Establish Static Directories
# os.chdir(__file__)
home = r"C:\Users\Houlberg\Documents\Thesis\Python"
os.chdir(home)

# Load stan files for unpickeling ??????
sms = glob.glob('../bayes-drt2-main/bayes_drt2/stan_model_files/*.pkl')
#sms = glob.glob('Pickles/*.pkl')
for file in sms:
    utils.load_pickle(file)

exp_dir = os.path.join(home, r'Experiments')
fig_dir = os.path.join(home, r'Figures')
pkl_dir = os.path.join(home, r'Pickles')
# pkl = os.path.join(pkl_dir,'obj{}.pkl'.format('MAP'))

tester = home + r'\Experiments'
# %% Selecting Experiment and establishing subfolders

experiment = 'IRFB\FirstCell'
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

colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
plt.rcParams['patch.facecolor'] = colors[0]

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
plt.rcParams['legend.fontsize'] = tick_size + 1
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.95

# %% Load Files and seperate by type.
# REQUIRES exporting EC-Lab raw binarys (.mpr) as text (.mpt)
files = glob.glob('Data/*.mpt')

# %% Parsing with pyimpspec
dummy_freq = np.logspace(6, -2, 81)
dummy_Z = np.ones_like(dummy_freq,dtype='complex128')
pyi_dummy = DataSet(dummy_freq, dummy_Z)

parsedfiles = []
for file in files:
    string = 'Parsed/{}'.format(file.split('.')[0].split("\\")[1])
    if not os.path.isdir(string):
        os.makedirs(string)
        try:
            parse = pyi.parse_data(file, file_format=".mpt")
            parsedfiles.append(parse)
            for i,cycle in enumerate(parse):
                utils.save_pickle(cycle.to_dict(), os.path.join(string, 'Cycle_{}.pkl'.format(i)))
        except:
            print("Pyimpspec could not parse file" + str(file))
    else:
        for i, cycle in enumerate(os.listdir(string)):
            pyi_pickle = utils.load_pickle(os.path.join(string, 'Cycle_{}.pkl'.format(i)))
            parse = pyi_dummy.from_dict(pyi_pickle)
            parsedfiles.append(parse)

# For this data set from Baichen we want the 1st cycle for all but 2cms, where we want the last
# [0,0] [1,0] [2,3] [3,0]
# For IRFB FirstCell start with the 1st cycle of LowFreq_Full


# %% Selecting the wanted cycles

chosen = [parsedfiles[4][0]]
# Construct Pandas df because bayes_drt package likes that
columns = ['Freq', 'Zreal', 'Zimag', 'Zmod', 'Zphs']
dFs = []
for eis in chosen:
    dFs.append(eis.to_dataframe(columns=columns))
'''
file = r'Data\test.txt'
#data, df = brute_eis(file)
#dFs = [df]
#### OOOOOOOOMG ITS BECAUSE OF DECINMAL SEPARATOR FROM ECLAB!!!
test = np.genfromtxt(file, delimiter="\t", skip_header = 66)
'''
# %% Bayes_DRT fitting
# By default, the Inverter class is configured to fit the DRT (rather than the DDT)
# Create separate Inverter instances for HMC and MAP fits
# Set the basis frequencies equal to the measurement frequencies 
# (not necessary in general, but yields faster results here - see Tutorial 1 for more info on basis_freq)
first = dFs[0]

mask = first['Zimag'] > 0
first_nonzero = first[~mask]

freq, Z = fl.get_fZ(first)
freq_nz, Z_nz = fl.get_fZ(first_nonzero)

inv_hmc = Inverter(basis_freq=freq)
inv_map = Inverter(basis_freq=freq)
inv_hmc_nonzero = Inverter(basis_freq=freq_nz)
inv_map_nonzero = Inverter(basis_freq=freq_nz)

# %% Perform HMC fit
if not os.path.isfile(os.path.join(pkl_directory, f'inv_hmc.pkl')):
    start = time.time()
    inv_hmc.fit(freq, Z, mode='sample')
    elapsed = time.time() - start
    print('HMC fit time {:.1f} s'.format(elapsed))
    utils.save_pickle(inv_hmc, os.path.join(pkl_directory, f'inv_hmc.pkl'))
else:
    inv_hmc = utils.load_pickle(os.path.join(pkl_directory, f'inv_hmc.pkl'))

# %% Perform HMC fit with initial tail removed
if not os.path.isfile(os.path.join(pkl_directory, f'inv_hmc_nz.pkl')):
    start = time.time()
    inv_hmc_nonzero.fit(freq_nz, Z_nz, mode='sample')
    elapsed = time.time() - start
    print('HMC fit time {:.1f} s'.format(elapsed))
    utils.save_pickle(inv_hmc_nonzero, os.path.join(pkl_directory, f'inv_hmc_nz.pkl'))
else:
    inv_hmc_nonzero = utils.load_pickle(os.path.join(pkl_directory, f'inv_hmc_nz.pkl'))

# %% Perform MAP fit
if not os.path.isfile(os.path.join(pkl_directory, f'inv_map.pkl')):
    start = time.time()
    inv_map.fit(freq, Z, mode='optimize')  # initialize from ridge solution
    elapsed = time.time() - start
    print('MAP fit time {:.1f} s'.format(elapsed))
    utils.save_pickle(inv_map, os.path.join(pkl_directory, f'inv_map.pkl'))
else:
    inv_map = utils.load_pickle(os.path.join(pkl_directory, f'inv_map.pkl'))

# %% Perform MAP fit with initial tail removed
if not os.path.isfile(os.path.join(pkl_directory, f'inv_map_nz.pkl')):
    start = time.time()
    inv_map_nonzero.fit(freq_nz, Z_nz, mode='sample')
    elapsed = time.time() - start
    print('MAP fit time {:.1f} s'.format(elapsed))
    utils.save_pickle(inv_map_nonzero, os.path.join(pkl_directory, f'inv_map_nz.pkl'))
else:
    inv_map_nonzero = utils.load_pickle(os.path.join(pkl_directory, f'inv_map_nz.pkl'))
# %% Visualize DRT and impedance fit
# plot impedance fit and recovered DRT
fig, axes = plt.subplots()

# plot fits of impedance data
inv_hmc.plot_fit(axes=axes, plot_type='nyquist', color='k', label='HMC fit', data_label='Data')
inv_map.plot_fit(axes=axes, plot_type='nyquist', color='r', label='MAP fit', plot_data=False)

fig.suptitle("Reconstructed Impedance")
plt.figure(fig)
plt.savefig(os.path.join(fig_directory, f"FitImpedance.png"))

plt.show()
# plot true DRT
# p = axes[1].plot(g_true['tau'],g_true['gamma'],label='True',ls='--')
# add Dirac delta function for RC element
# axes[1].plot([np.exp(-2),np.exp(-2)],[0,10],ls='--',c=p[0].get_color(),lw=1)

# Plot recovered DRT at given tau values
# tau_plot = g_true['tau'].values

# %%
tau = inv_map.distributions['DRT']['tau']
fig, axes = plt.subplots()
inv_hmc.plot_distribution(ax=axes, tau_plot=tau, color='k', label='HMC mean', ci_label='HMC 95% CI')
inv_map.plot_distribution(ax=axes, tau_plot=tau, color='r', label='MAP')
sec_xaxis = [x for x in axes.get_children() if isinstance(x, matplotlib.axes._secondary_axes.SecondaryAxis)][0]
sec_xaxis.set_xlabel('$f$ / Hz')
# axes[1].set_ylim(0,3.5)
axes.legend()
# fig.tight_layout()

fig.suptitle("DRT")
plt.figure(fig)
plt.savefig(os.path.join(fig_directory, f"DRT.png"))

plt.show()

# %% Visualize the recovered error structure"
# For visual clarity, only MAP results are shown.
# HMC results can be obtained in the same way
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)

# plot residuals and estimated error structure
inv_hmc.plot_residuals(axes=axes)

# plot true error structure in miliohms. Cant with this data
# p = axes[0].plot(freq, 3*Zdf['sigma_re'] * 1000, ls='--')
# axes[0].plot(freq, -3*Zdf['sigma_re'] * 1000, ls='--', c=p[0].get_color())
# axes[1].plot(freq, 3*Zdf['sigma_im'] * 1000, ls='--')
# axes[1].plot(freq, -3*Zdf['sigma_im'] * 1000, ls='--', c=p[0].get_color(), label='True $\pm 3\sigma$')

axes[1].legend()

fig.suptitle("Bayes Estimated Error Structure")
plt.figure(fig)
plt.savefig(os.path.join(fig_directory, f"BayesResiduals.png"))

plt.show()

# %% Visualize the recovered error structure"
# For visual clarity, only MAP results are shown.
# HMC results can be obtained in the same way
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)

# plot residuals and estimated error structure
inv_map.plot_residuals(axes=axes)

# plot true error structure in miliohms. Cant with this data
# p = axes[0].plot(freq, 3*Zdf['sigma_re'] * 1000, ls='--')
# axes[0].plot(freq, -3*Zdf['sigma_re'] * 1000, ls='--', c=p[0].get_color())
# axes[1].plot(freq, 3*Zdf['sigma_im'] * 1000, ls='--')
# axes[1].plot(freq, -3*Zdf['sigma_im'] * 1000, ls='--', c=p[0].get_color(), label='True $\pm 3\sigma$')

axes[1].legend()

fig.suptitle("MAP Estimated Error Structure")
plt.figure(fig)
plt.savefig(os.path.join(fig_directory, f"MAPResiduals.png"))

plt.show()

# %% Peak fitting
# Only fit peaks that have a prominence of >= 5% of the estimated polarization resistance
inv_hmc.fit_peaks(prom_rthresh=0.05, percentile=99.7)

# plot the peak fit
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
inv_hmc.plot_peak_fit(ax=axes[0])  # Plot the overall peak fit
inv_hmc.plot_peak_fit(ax=axes[1], plot_individual_peaks=True)  # Plot the individual peaks
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
fig.tight_layout()

fig.suptitle("Bayes Peak Fits")
plt.figure(fig)
plt.savefig(os.path.join(fig_directory, f"BayesPeakFits.png"))

plt.show()
# %% Peak fitting
# Only fit peaks that have a prominence of >= 5% of the estimated polarization resistance
inv_map.fit_peaks(prom_rthresh=0.05)

# plot the peak fit
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
inv_map.plot_peak_fit(ax=axes[0])  # Plot the overall peak fit
inv_map.plot_peak_fit(ax=axes[1], plot_individual_peaks=True)  # Plot the individual peaks
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
fig.tight_layout()

fig.suptitle("MAP Peak Fits")
plt.figure(fig)
plt.savefig(os.path.join(fig_directory, f"MAPPeakFits.png"))

plt.show()

# %% Multi distribution
# for sp_dr: use xp_scale=0.8
inv_multi_hmc = Inverter(distributions={"DRT": {'kernel': 'DRT'},
                                    'TP-DDT': {'kernel': 'DDT',
                                               'symmetry': 'planar',
                                               'bc': 'transmissive',
                                               'dist_type': 'parallel',
                                               'x_scale': 0.8}
                                    }
                     )
inv_multi_map = inv_multi_hmc
# %% Perform HMC fit
if not os.path.isfile(os.path.join(pkl_directory, f'inv_multi_hmc.pkl')):
    start = time.time()
    inv_multi_hmc.fit(freq, Z, mode='sample',nonneg=True)
    elapsed = time.time() - start
    print('HMC fit time {:.1f} s'.format(elapsed))
    utils.save_pickle(inv_multi_hmc, os.path.join(pkl_directory, f'inv_multi_hmc.pkl'))
else:
    inv_multi_hmc = utils.load_pickle(os.path.join(pkl_directory, f'inv_multi_hmc.pkl'))

# %% Perform MAP fit
if not os.path.isfile(os.path.join(pkl_directory, f'inv_multi_map.pkl')):
    start = time.time()
    inv_map.fit(freq, Z, mode='optimize', nonneg=True)  # initialize from ridge solution
    elapsed = time.time() - start
    print('MAP fit time {:.1f} s'.format(elapsed))
    utils.save_pickle(inv_multi_map, os.path.join(pkl_directory, f'inv_multi_map.pkl'))
else:
    inv_multi_map = utils.load_pickle(os.path.join(pkl_directory, f'inv_multi_map.pkl'))

# %% Visualize DRT and impedance fit
# plot impedance fit and recovered DRT
fig, axes = plt.subplots()

# plot fits of impedance data
inv_multi_hmc.plot_fit(axes=axes, plot_type='nyquist', color='k', label='HMC fit', data_label='Data')
inv_multi_map.plot_fit(axes=axes, plot_type='nyquist', color='r', label='MAP fit', plot_data=False)

fig.suptitle("Reconstructed Impedance - Multi Distribution")
plt.figure(fig)
plt.savefig(os.path.join(fig_directory, f"FitImpedanceMulti.png"))

plt.show()
# plot true DRT
# p = axes[1].plot(g_true['tau'],g_true['gamma'],label='True',ls='--')
# add Dirac delta function for RC element
# axes[1].plot([np.exp(-2),np.exp(-2)],[0,10],ls='--',c=p[0].get_color(),lw=1)

# Plot recovered DRT at given tau values
# tau_plot = g_true['tau'].values

# %%
tau = inv_multi_map.distributions['DRT']['tau']
fig, axes = plt.subplots()
inv_multi_hmc.plot_distribution(ax=axes, tau_plot=tau, color='k', label='HMC mean', ci_label='HMC 95% CI')
inv_multi_map.plot_distribution(ax=axes, tau_plot=tau, color='r', label='MAP')
sec_xaxis = [x for x in axes.get_children() if isinstance(x, matplotlib.axes._secondary_axes.SecondaryAxis)][0]
sec_xaxis.set_xlabel('$f$ / Hz')
# axes[1].set_ylim(0,3.5)
axes.legend()
# fig.tight_layout()

fig.suptitle("Combined DRT-DDT")
plt.figure(fig)
plt.savefig(os.path.join(fig_directory, f"DRT-DDT.png"))

plt.show()

# %% Visualize the recovered error structure"
# For visual clarity, only MAP results are shown.
# HMC results can be obtained in the same way
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)

# plot residuals and estimated error structure
inv_multi_hmc.plot_residuals(axes=axes)

# plot true error structure in miliohms. Cant with this data
# p = axes[0].plot(freq, 3*Zdf['sigma_re'] * 1000, ls='--')
# axes[0].plot(freq, -3*Zdf['sigma_re'] * 1000, ls='--', c=p[0].get_color())
# axes[1].plot(freq, 3*Zdf['sigma_im'] * 1000, ls='--')
# axes[1].plot(freq, -3*Zdf['sigma_im'] * 1000, ls='--', c=p[0].get_color(), label='True $\pm 3\sigma$')

axes[1].legend()

fig.suptitle("Bayes Estimated Error Structure - Multi Distribution")
plt.figure(fig)
plt.savefig(os.path.join(fig_directory, f"BayesResidualsMulti.png"))

plt.show()

# %% Visualize the recovered error structure"
# For visual clarity, only MAP results are shown.
# HMC results can be obtained in the same way
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)

# plot residuals and estimated error structure
inv_multi_map.plot_residuals(axes=axes)

# plot true error structure in miliohms. Cant with this data
# p = axes[0].plot(freq, 3*Zdf['sigma_re'] * 1000, ls='--')
# axes[0].plot(freq, -3*Zdf['sigma_re'] * 1000, ls='--', c=p[0].get_color())
# axes[1].plot(freq, 3*Zdf['sigma_im'] * 1000, ls='--')
# axes[1].plot(freq, -3*Zdf['sigma_im'] * 1000, ls='--', c=p[0].get_color(), label='True $\pm 3\sigma$')

axes[1].legend()

fig.suptitle("MAP Estimated Error Structure - Multi Distribution")
plt.figure(fig)
plt.savefig(os.path.join(fig_directory, f"MAPResidualsMulti.png"))

plt.show()

# %% Peak fitting
# Only fit peaks that have a prominence of >= 5% of the estimated polarization resistance
inv_multi_hmc.fit_peaks(prom_rthresh=0.05, percentile=99.7)

# plot the peak fit
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
inv_multi_hmc.plot_peak_fit(ax=axes[0])  # Plot the overall peak fit
inv_multi_hmc.plot_peak_fit(ax=axes[1], plot_individual_peaks=True)  # Plot the individual peaks
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
fig.tight_layout()

fig.suptitle("Bayes Peak Fits - Multi Distribution")
plt.figure(fig)
plt.savefig(os.path.join(fig_directory, f"BayesPeakFitsMulti.png"))

plt.show()
# %% Peak fitting
# Only fit peaks that have a prominence of >= 5% of the estimated polarization resistance
inv_multi_map.fit_peaks(prom_rthresh=0.05)

# plot the peak fit
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
inv_multi_map.plot_peak_fit(ax=axes[0])  # Plot the overall peak fit
inv_multi_map.plot_peak_fit(ax=axes[1], plot_individual_peaks=True)  # Plot the individual peaks
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
fig.tight_layout()

fig.suptitle("MAP Peak Fits - Multi Distribution")
plt.figure(fig)
plt.savefig(os.path.join(fig_directory, f"MAPPeakFitsMulti.png"))

plt.show()
