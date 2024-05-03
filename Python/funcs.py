import gc
import matplotlib.axes._secondary_axes
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import glob
import os
import pyimpspec as pyi
from matplotlib.figure import Figure
import pickle
import io
import warnings


# Define Custom Functions (And Class :) )
class FilteredStream(io.StringIO):
    def __init__(self, filtered_values=None):
        super().__init__()
        self.filtered_values = filtered_values if filtered_values else []

    def write(self, msg):
        if not any(filtered_value in msg for filtered_value in self.filtered_values):
            super().write(msg)

    def flush(self):
        pass  # Override flush method to prevent clearing the buffer

    def write_warning(self, message, category, filename, lineno, file=None, line=None):
        if not any(filtered_value in message for filtered_value in self.filtered_values):
            formatted_warning = warnings.formatwarning(message, category, filename, lineno, line)
            self.write(formatted_warning)


def save_pickle(obj, file):
    with open(file, "xb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print("Dumped pickle to {}".format(file))


def load_pickle(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def rename_files(folder_path, common_part):
    for filename in os.listdir(folder_path):
        if common_part in filename:
            new_filename = filename.replace(common_part, "")
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)


def brute_eis(file):
    """
    Manual approach to parsing .mpt eis files

    Parameter
    ---------
    file: path-like
        Str or bytes representing path to file
    Returns
    ---------
    data: np.ndarray
        Numpy array of the parses data
    df: pd.DataFrame
        Pandas DataFrame using bayes_drt convention of column naming
    """
    f = open(file, "r")
    line_two = f.readlines()[1]  # Line two will say: "Nb of header lines: XX"
    f.close()
    header_lines = int(
        np.array(line_two.split())[
            np.char.isnumeric(np.array(line_two.split()))
        ].astype(int)
    )  # Grabbing the numeric value of header lines
    data = np.genfromtxt(file, delimiter="\t", skip_header=int(header_lines))

    # Construct Pandas df because the bayes_drt package likes that
    df = pd.DataFrame(
        data=data[:, 0:5], columns=["Freq", "Zreal", "Zimag", "Zmod", "Zphs"]
    )
    return data, df


def set_aspect_ratio(ax, dataset):
    """
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
    """
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
        df = df.drt_hmc_pickle(big_df["Capacity/mA.h"])
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


def set_equal_tickspace(ax, figure=None, space=("min", 1)):
    """
    Adjusts x and y tick-spacing to be equal to the lower of the two

    Parameters
    ----------
    space : str or tuple
        If str, should be either 'max' or 'min'.
        If tuple, should be a tuple of a string ('max' or 'min') and a value between 0 and 1.
    ax : matplotlib.axes._axes.Axes
        Axes to adjust tickspace on
    figure : matplotlib.figure.Figure, optional
        Figure containing the chosen Axes. The default is None.

    Returns
    -------
    None.

    """
    assert isinstance(figure, Figure) or figure is None, figure
    if figure is None:
        assert ax is None
        figure, axis = plt.subplots()
        ax = axis

    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    xtickspace = xticks[1] - xticks[0]
    ytickspace = yticks[1] - yticks[0]

    if isinstance(space, str):
        if space == "max":
            spacing = max(xtickspace, ytickspace)
        elif space == "min":
            spacing = min(xtickspace, ytickspace)
        else:
            raise ValueError('space must be either "max" or "min"')
    elif (
        isinstance(space, tuple)
        and len(space) == 2
        and isinstance(space[0], str)
        and isinstance(space[1], (int, float))
    ):
        if space[0] == "min":
            spacing = min(xtickspace, ytickspace) * space[1]
        elif space[0] == "max":
            spacing = max(xtickspace, ytickspace) * space[1]
        else:
            raise ValueError(
                'Invalid tuple format. The first element should be either "max" or "min".'
            )
    else:
        raise ValueError("Invalid space parameter format.")

    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=spacing))

    return


def remove_legend_duplicate():
    # Removes legend duplicates
    try:
        handles, labels = plt.gca().get_legend_handles_labels()
    except ValueError:
        print("No plot available")
        return
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    return


def create_mask_notail(dataset):
    dataset_masked = pyi.DataSet.duplicate(dataset)
    tail_end = 0
    mask = {}
    z_im = dataset.get_impedances().imag
    for i, j in enumerate(z_im):
        if j < 0:
            tail_end = i
            break
    for i in range(dataset.get_num_points()):
        mask[i] = i < tail_end
    dataset_masked.set_mask(mask)
    return dataset_masked, mask


def reset_files(path, extension):
    """
    Remove all files of given type in the given directory.

    Args:
        path (str): The path to the directory.
        extension (str): The file extension type to remove.
    """

    files = glob.glob(path + "**." + extension, recursive=True)
    [os.remove(x) for x in files]

    return


def swap_filename_parts(filename):
    """
    Swap the order of parts of a filename separated by underscores, while maintaining the file extension.

    Args:
        filename (str): The input filename.

    Returns:
        str: The modified filename with parts swapped.

    Example:
        swapped_filename = swap_filename_parts('part1_part2_part3.txt')
        print(swapped_filename)  # Output: 'part3_part2_part1.txt'
    """
    name, extension = filename.split(".")
    parts = name.split("_", maxsplit=1)
    swapped_filename = "_".join(parts[::-1]) + "." + extension
    return swapped_filename
