import os
import glob
import pathlib


path = pathlib.Path().absolute()

files = glob.glob(str(path) + "/*.sh")
files.remove(files[files.index("/scratch/djidje77/KD-Reid/linter.sh")])
for f in files:
    cmd = "sbatch " + f
    os.system(cmd)  # returns the exit code in unix


"""var = os.getenv("DATA_DIR")

datamanager = torchreid.data.ImageDataManager(
        root= var + "/reid-data",
        ...
)
"""