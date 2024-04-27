# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:54:03 2024

@author: Houlberg
"""
#%%
import numpy as np
import pandas as pd
import os
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

import deareis
from bayes_drt2.inversion import Inverter
import bayes_drt2.file_load as fl
import bayes_drt2.plotting as bp

import pyimpspec
from pyimpspec import mpl
import eclabfiles as ecf
import yadg
import fnmatch

#%% Load Files and seperate by type. 
#REQUIRES exporting EC-Lab raw binarys (.mpr) as text (.mpt)
#Parsers for .mpr are broken and fixing them might be a large undertaking
direc = r"C:\Users\Houlberg\Documents\Thesis\Python\Data\RDE"
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

peisfiles = []
cvfiles = []
ocvfiles = []
waitfiles = []

for file in loadedfiles:
    filename = os.fsdecode(file)
    if "_PEIS_" in filename:
        peisfiles.append(file)
    elif "_CV_" in filename:
        cvfiles.append(file)
    elif "_OCV_" in filename:
        ocvfiles.append(file)
    elif "_WAIT_"in filename:
        waitfiles.append(file)

#%%
# Matplotlib parameters

plt.style.use('seaborn-v0_8-colorblind')
tick_size = 9
label_size = 11

plt.rcParams['font.family'] = ['DejaVu Sans','sans-serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

plt.rcParams['xtick.labelsize'] = tick_size
plt.rcParams['ytick.labelsize'] = tick_size
plt.rcParams['axes.labelsize'] = label_size
plt.rcParams['legend.fontsize'] = tick_size + 1
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.95

#%%
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
        # expand the x axis
        diff = (yscale - xscale) * width
        xmin = max(0, ax.get_xlim()[0] - diff / 2)
        mindelta = ax.get_xlim()[0] - xmin
        xmax = ax.get_xlim()[1] + diff - mindelta

        ax.set_xlim(xmin, xmax)
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

#%%
# Prob not needed
'''
def axscale(ax,dataset):
    pyimp = dataset.get_impedances()
    img = pyimp.imag
    
    
    yrng = ax.get_ylim()[1] - ax.get_ylim()[0]
    xrng = ax.get_xlim()[1] - ax.get_xlim()[0]

	# get axis dimensions
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height

    yscale = yrng / height
    xscale = xrng / width

    if yscale > xscale:
        # expand the x axis
        diff = (yscale - xscale) * width
        xmin = max(0, ax.get_xlim()[0] - diff / 2)
        mindelta = ax.get_xlim()[0] - xmin
        xmax = ax.get_xlim()[1] + diff - mindelta

        ax.set_xlim(xmin, xmax)
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
'''
#%%
#Parse data with pyimpspec

parses= []

for file in peisfiles:
    try:
        parses.append(pyimpspec.parse_data(file,file_format=".mpt"))
    except:
        print("Error with file " + str(file))

#No clue why some files dont work

# parses has a wonky structure due to difference in ec-lab experiment setup
# its an array of arrays with a number of rows corrosponding t0 the number of PEIS files
# the number of columns is not consistent but is fixed later
# Based on if the series were generated from a n_cycle > 1 or a new technique in EC-lab

#%%
#Organizing data series based on file name
#Final structure of toplots: array of experiments 
# with identifier in last index and scans in order starting at first index

eids = []
toplots = []
tomerge = []
uniqueindex = []
k = 0

for peis in parses:
    
    label = str(peis[0])
    kword1 = "HCl1M"
    kword2 = "Ag"
    idx1 = label.find(kword1)
    idx2 = label.find(kword2)
    rpm = label[idx1 + len(kword1) + 1 : idx2 - 1]
    
    kword3 = "RDE"
    kword4 = "_0"
    idx3 = label.find(kword3)
    idx4 = label.find(kword4)
    identifier = label[idx3 + len (kword3) +1 : idx4]
    if not identifier:
        identifier = "Standard"
    eid = identifier + " " + rpm
    if eid not in eids:
        eids.append(eid)
        toplots.append(peis)
        uniqueindex.append(k)
    else:
        if len(peis) == 1:
            tomerge.append(peis)
    k+=1
    peis.append(eid)

for x in toplots:
    for val in tomerge:
        if val[-1] == x[-1]:
            x.insert(-2,val[0])
'''
# OK CONSIDER DITCHING THIS WACK PYIMPSPEC PLOTTING
xses = []
tempx = []
tempid = ""
for peis in toplots:
    if len(peis) != 2:
        fig, ax = pyimpspec.mpl.plot_nyquist(peis[0],label="Scan 1",adjust_axes=False,legend=False)
        ax[0] = set_aspect_ratio(ax[0],peis[0])
        #ax[0].axis('equal')
        for i in range(len(peis)-2):
            _,newax = pyimpspec.mpl.plot_nyquist(peis[i+1],label="Scan" +str(i+2),adjust_axes=False,legend=False, figure=fig, axes=ax)
            newax[0] = set_aspect_ratio(newax[0],peis[i+1])
            #newax[0].axis('equal')
            imp2 = peis[i+1].get_impedances()
            tempx.append(np.min(imp2.real))
    else:
        fig, ax = pyimpspec.mpl.plot_nyquist(peis[0],label="Scan 1",adjust_axes=False,legend=False)
        ax[0] = set_aspect_ratio(ax[0],peis[0])
        #ax[0].axis('equal')

    imp = peis[0].get_impedances()
    xses.append(np.min(imp.real))

    fig.legend(loc="outside upper right")
    fig.suptitle(peis[-1])
    #plt.ylim(bottom=-10)
    #plt.tight_layout()
    plt.show()
        '''
#%%
# TOOK WAY TOO FUCKING LONG TO DO
# 

unit_scale = ""
for peis in toplots:
    fig, ax = plt.subplots(figsize=(10, 7))
    xrange = np.max(peis[0].get_impedances().real) - np.min(peis[0].get_impedances().real)
    for i in range(len(peis)-1):
        imp = peis[i].get_impedances()
        ax.scatter(imp.real,-imp.imag,label="Scan" +str(i+1))
        ax = set_aspect_ratio(ax,peis[i])
        
        xticks = plt.xticks()[0]
        yticks = plt.yticks()[0]
        xtickspace = xticks[1] - xticks[0]
        ytickspace = yticks[1] - yticks[0]

        spacing = min(xtickspace,ytickspace)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=spacing))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(base=spacing))
        
        ax.grid()
        ax.set_xlabel(f'$Z^\prime \ / \ \mathrm{{{unit_scale}}}\Omega$')
        ax.set_ylabel(f'$-Z^{{\prime\prime}} \ / \ \mathrm{{{unit_scale}}}\Omega$')
        ax.legend()
        
    #fig.legend(bbox_to_anchor=(1, 1),
    #  bbox_transform=fig.transFigure)
    fig.suptitle(peis[-1])

    plt.tight_layout()
    plt.gca().set_aspect('equal')
    
    plt.xlim(left= - xrange * 0.05)
    plt.show()


#%%
#Taking the nice readings for now
toplotscopy = toplots
nicepeis = []
for peis in toplotscopy:
    if "1minRestVSEoc-3ppf-14p1mVAmplitude" in peis[-1]:
        label = str(peis[-1])
        kword1 = " "
        kword2 = "RPM"
        idx1 = label.find(kword1)
        idx2 = label.find(kword2)
        rpm = label[idx1 + len(kword1) : idx2 ]
        
        peis.insert(-1, int(rpm))
        nicepeis.append(peis)
        
nicepeis.sort(key=lambda x: x[-2])

#%%
#Ok i dunno why i keep overwriting here but saving another copy for safety
backupnice = nicepeis


#%%
#KK test
#Assuming last scan is the best and using that

kkpeis = nicepeis[4][2]
title = str(nicepeis[4][3])

mu_criterion = 0.85
tests = pyimpspec.perform_exploratory_tests(kkpeis,
                                            mu_criterion=mu_criterion)

fig = plt.figure(figsize=(10,7))

ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,2,3)
ax3 = ax2.twinx()
ax4 = plt.subplot(2,2,4)
ax5 = ax4.twinx()
axes = [ax1, ax2, ax3, ax4, ax5]


#figtest, tmp = plt.subplots(2,2)
#axestest = [tmp[0,:],
#        tmp[1][0],tmp[1][0].twinx(),
#        tmp[1][1],tmp[1][1].twinx()]
#plt.show()
#%%
_,_ = mpl.plot_mu_xps(tests,
                      mu_criterion=mu_criterion,
                      figure=fig,
                      axes=[axes[3],axes[4]])

_,_ = mpl.plot_residuals(tests[0],
                         figure=fig,
                         axes=[axes[1],axes[2]])

_,_ = mpl.plot_nyquist(tests[0],
                       line=True,
                       figure=fig,
                       axes=[axes[0]],
                       label = "Fit")

_ = mpl.plot_nyquist(kkpeis, 
                     figure=fig,
                     axes=[axes[0]],
                     colors={"impedance": "black"},
                     legend=True,
                     label="Data")

fig.suptitle(title + "RPM")
fig.tight_layout()
plt.show()
#%%
#pyimpspec implementation of DRT algorithms
#Only using on 2000rpm last scan, for now
drtpeis = kkpeis

bht = pyimpspec.calculate_drt(drtpeis,method = "bht")
#mrq_fit = pyimpspec.calculate_drt(drtpeis,method = "mrq-fit")
tr_nnls = pyimpspec.calculate_drt(drtpeis,method = "tr-nnls")
tr_rbf = pyimpspec.calculate_drt(drtpeis,method = "tr-rbf")

#%%

fig, ax = mpl.plot_gamma(bht)
fig.suptitle("")
fig.tight_layout()

#%%

fig, ax = mpl.plot_gamma(tr_nnls)
fig.suptitle("")
fig.tight_layout()

#%%

fig, ax = mpl.plot_gamma(tr_rbf)
fig.suptitle("")
fig.tight_layout()

#%%
#Bayes_drt
freq = drtpeis.get_frequencies()
Z = drtpeis.get_impedances()
magnitudes = drtpeis.get_magnitudes()
phase = drtpeis.get_phases()



data = np.column_stack((freq,Z.real,Z.imag,magnitudes,phase))

# Construct Pandas df because the package likes that
df=pd.DataFrame(data=data,columns=['Freq','Zreal','Zimag','Zmod','Zphz'])

axes = bp.plot_eis(df)

#%%
"Fit the data"
# By default, the Inverter class is configured to fit the DRT (rather than the DDT)
# Create separate Inverter instances for HMC and MAP fits
# Set the basis frequencies equal to the measurement frequencies 
# (not necessary in general, but yields faster results here - see Tutorial 1 for more info on basis_freq)
inv_hmc = Inverter(basis_freq=freq)
inv_map = Inverter(basis_freq=freq)

# Perform HMC fit
start = time.time()
inv_hmc.fit(freq, Z, mode='sample')
elapsed = time.time() - start
print('HMC fit time {:.1f} s'.format(elapsed))

# Perform MAP fit
start = time.time()
inv_map.fit(freq, Z, mode='optimize')  # initialize from ridge solution
elapsed = time.time() - start
print('MAP fit time {:.1f} s'.format(elapsed))

#%%
"Visualize DRT and impedance fit"
# plot impedance fit and recovered DRT
fig = plt.figure(figsize=(10, 7))
ax = plt.subplot()
axes = [ax, ax.twinx()]
#fig,axes = plt.subplots(1, 2, figsize=(10, 7))
# plot fits of impedance data
#inv_hmc.plot_fit(axes=axes[0], plot_type='nyquist', color='k', label='HMC fit', data_label='Data')
#inv_map.plot_fit(axes=axes[0], plot_type='nyquist', color='r', label='MAP fit', plot_data=False)

# Plot recovered DRT at given tau values
inv_hmc.plot_distribution(ax=axes[0], color='k', label='HMC mean', ci_label='HMC 95% CI')
inv_map.plot_distribution(ax=axes[0], color='r', label='MAP')

axes[1].set_ylim(0,3.5)
axes[1].legend()


fig.tight_layout()

#%%
"Visualize the recovered error structure"
# For visual clarity, only MAP results are shown.
# HMC results can be obtained in the same way
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)

# plot residuals and estimated error structure
inv_map.plot_residuals(axes=axes)

axes[1].legend()

fig.tight_layout()

#inv_hmc.plot_full_results()
#%%
#DDT !!!
"Fit the data"
# Define the distribution to be recovered (transmissive planar DDT) in the Inverter initialization
# Use a slightly expanded basis frequency range to fully capture the tail of the low-frequency peak
inv_hmc2 = Inverter(distributions={'DDT':{'kernel':'DDT','dist_type':'parallel','bc':'transmissive',
                                         'symmetry':'planar','basis_freq':np.logspace(6,-3,91)}
                                 }
                  )
inv_map2 = Inverter(distributions={'DDT':{'kernel':'DDT','dist_type':'parallel','bc':'transmissive',
                                         'symmetry':'planar','basis_freq':np.logspace(6,-3,91)}
                                 }
                  )

# # Perform HMC fit
start = time.time()
inv_hmc2.fit(freq, Z, mode='sample')
elapsed = time.time() - start
print('HMC fit time {:.2f}'.format(elapsed))

# Perform MAP fit
start = time.time()
inv_map2.fit(freq, Z)
elapsed = time.time() - start
print('MAP fit time {:.2f}'.format(elapsed))

#%%

"Visualize DDT and impedance fit"
# plot impedance fit and recovered DRT
fig,axes = plt.subplots(1, 2, figsize=(8, 3.5))

# plot fits of impedance data
inv_hmc2.plot_fit(axes=axes[0], plot_type='nyquist', color='k', label='HMC fit', data_label='Data')
inv_map2.plot_fit(axes=axes[0], plot_type='nyquist', color='r', label='MAP fit', plot_data=False)

# plot true DRT

# Plot recovered DRT at given tau values
inv_hmc2.plot_distribution(ax=axes[1], color='k', label='HMC mean', ci_label='HMC 95% CI')
inv_map2.plot_distribution(ax=axes[1], color='r', label='MAP')

axes[1].legend()

fig.tight_layout()

'''
TODO:
DDT on simulated data. Warburg diffusion eis
'''
