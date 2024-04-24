# Establish Static Directories
import os

home = r'C:\Users\Houlberg\Documents\Thesis\Python'
os.chdir(home)

exp_dir = os.path.join(home, r'Experiments')
fig_dir = os.path.join(home, r'Figures')
pkl_dir = os.path.join(home, r'Pickles')

# %% Selecting Experiment and establishing subfolders

def set_experiment(experiment):
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
    if not os.path.isdir('Parses'):
        os.makedirs('Parses')
    return directory, fig_directory, pkl_directory
