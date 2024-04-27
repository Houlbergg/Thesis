import os
import glob
from funcs import rename_files, swap_filename_parts

dists = ["drt", "ddt", "dmt"]
path = "C:\\Users\\Houlberg\\Documents\\Thesis\\Python\\Pickles\\Baichen\\IronSymmetricNIPSElectrode\\FTFF"
pattern = os.path.join(path, "**/*.pkl")
files = glob.glob(pattern, recursive=True)
[os.remove(f) for f in files]
