import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as mark
from cycler import cycler
import scienceplots

# Matplotlib parameters
"""
base = 'seaborn-v0_8' # -paper or -poster for smaller or larger figs ##Doesnt really work
size = 'seaborn-v0_8-poster'
ticks = 'seaborn-v0_8-ticks'
plt.style.use([base,size,ticks])
"""
bright_colors = [
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
    "#000000",
]
bright_names = ["blue", "red", "green", "yellow", "cyan", "purple", "grey", "black"]
bright_dict = dict(zip(bright_names, bright_colors))

contrast_colors = ["#004488", "#DDAA33", "#BB5566", "#000000"]
contrast_names = ["blue", "yellow", "red", "black"]
contrast_dict = dict(zip(contrast_names, contrast_colors))

vibrant_colors = [
    "#EE7733",
    "#0077BB",
    "#33BBEE",
    "#EE3377",
    "#CC3311",
    "#009988",
    "#BBBBBB",
    "#000000",
]
vibrant_names = ["orange", "blue", "cyan", "magenta", "red", "teal", "grey", "black"]
vibrant_dict = dict(zip(vibrant_names, vibrant_colors))

seaborn_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]

markers = ["o", "s", "x", "v", "+", "D", "p", "*"]
# plt.rcParams['markers.fillstyle'] = 'none'
plt.rcParams["axes.prop_cycle"] = cycler(color=bright_colors, marker=markers)
plt.rcParams["lines.markerfacecolor"] = "w"
no_fill_markers = [mark.MarkerStyle(marker, fillstyle="none") for marker in markers]

# plt.rcParams['axes.prop_cycle'] = cycler(color=bright_colors)
# plt.rcParams['patch.facecolor'] = bright_colors[0]

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

plt.rcParams["lines.markersize"] = 3
plt.rcParams["xtick.labelsize"] = tick_size
plt.rcParams["ytick.labelsize"] = tick_size

plt.rcParams["axes.labelsize"] = label_size
plt.rcParams["axes.axisbelow"] = True
plt.rcParams["legend.fontsize"] = tick_size + 1
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.framealpha"] = 0.95

plt.style.use("notebook")  # Overrides a lot

plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

# TODO - Rewrite as dicts per object to be passed directly to the plotting functions
