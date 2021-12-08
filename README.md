# EGNN MD trajectories
This is the repository for the work that the SURF HPML team has performed on applying Equivariant Graph Neural Networks (EGNNs) to the [MD trajectories dataset of small molecules](http://quantum-machine.org/datasets/). 

## Initialization
Codes are meant to be run on `Lisa`. Running `init.sh create` will create the virtual environments for you. Training scripts for the individual molecules can be found in the `Job scripts` folder. 

## Data
Download the data [here](http://quantum-machine.org/gdml/#datasets) as `*.npz` files and place these files in the `data` folder. You may need to rename a few files. See `options.py` on how these should be named. 
