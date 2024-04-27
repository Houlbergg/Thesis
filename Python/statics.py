# Establish Static Directories
import os

home = r'C:\Users\Houlberg\Documents\Thesis\Python'
os.chdir(home)

exp_dir = os.path.join(home, r'Experiments')
fig_dir = os.path.join(home, r'Figures')
pkl_dir = os.path.join(home, r'Pickles')

# %% Selecting Experiment and establishing subfolders

def set_experiment(experiment):
    """
    Set up directories for a new experiment.

    This function creates directories for storing experiment-related files such as data, figures, pickles, and parses.

    Args:
        experiment (str): Name of the experiment.

    Returns:
        tuple: A tuple containing directory paths for the experiment.
            The tuple elements are:
            - directory: Directory path for the experiment.
            - fig_directory: Directory path for storing figures related to the experiment.
            - pkl_directory: Directory path for storing pickled objects related to the experiment.
            - parse_directory: Directory path for storing parsed data related to the experiment.

    Example:
        directory, fig_directory, pkl_directory, parse_directory = set_experiment('experiment1')
    """
    directory = os.path.join(exp_dir, experiment)
    pkl_directory = os.path.join(pkl_dir, experiment)
    fig_directory = os.path.join(fig_dir, experiment)
    parse_directory = os.path.join(pkl_directory, 'Parses')

    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(pkl_directory):
        os.makedirs(pkl_directory)
    if not os.path.exists(fig_directory):
        os.makedirs(fig_directory)
    if not os.path.exists(parse_directory):
        os.makedirs(parse_directory)

    os.chdir(directory)

    return directory, fig_directory, pkl_directory, parse_directory
