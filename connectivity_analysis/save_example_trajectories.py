"""
Created on Wed Jun 01 2022

@author: SLYpma

Creates lon,lat,time trajectories starting at one specific coastal gridpoint. Only those trajectories are saved that beach at one of the Galapagos Islands.

"""

# Packages
import numpy as np
import pandas as pd
import math 
from parcels import rng as random
import os
from pathlib import Path
import sys
import xarray as xr

parent_dir = str(Path(os.path.dirname(os.path.realpath("__file__"))).parents[0])

# Input
grid_start = 107
beaching_timescale = 5

# Functions
def load_dict(path_to_dict):
    import pickle
    with open(path_to_dict, 'rb') as config_dictionary_file:

        dictionary = pickle.load(config_dictionary_file)

        return dictionary  

# Directories
path_to_dict = parent_dir + '/particle_simulation/output/Simulation2008_2012_complete_trajectories.dictionary'
grid_dir = parent_dir + '/particle_simulation/input/GridsDataFrame.csv'
output_dir = 'input/'

# load the trajectory data
traj_data_dict = load_dict(path_to_dict)
total_particles = len(traj_data_dict.get('lon')[:,0])
time_steps = len(traj_data_dict.get('lon')[0,:])
grids = pd.read_csv(grid_dir)
coastcells = traj_data_dict.get('coastgrids')
plon = traj_data_dict.get('lon')
plat = traj_data_dict.get('lat')
minlon = np.nanmin(plon)
maxlon = np.nanmax(plon)
minlat = np.nanmin(plat)
maxlat = np.nanmax(plat)

# beaching params
Prob = math.exp((-60*60)/(beaching_timescale*24*60*60))
np.random.seed(23)
randomprob = np.random.random((total_particles,time_steps))

# check whether particle starts at specific gridnumber and save locations until beaching
traj_lon = np.zeros((int(total_particles/len(grids)),time_steps))
traj_lat = np.zeros((int(total_particles/len(grids)),time_steps))
count = 0
for particle in range(total_particles):
    BeginGrid = int(particle%len(grids))
    if BeginGrid == grid_start:
        
        lons = traj_data_dict.get('lon')[particle,:]
        lats = traj_data_dict.get('lat')[particle,:]
        
        for t in np.arange(time_steps):
            CoastcellValue = coastcells[particle,t]               
            
            if ((lons[t] > minlon+0.05) and
                (lats[t] > minlat+0.05) and
                (lons[t] < maxlon-0.05) and
                (lats[t] < maxlat-0.05)):

                if (CoastcellValue == 1 and randomprob[particle,t] > Prob):
                    traj_lon[count,:t] = lons[:t]
                    traj_lat[count,:t] = lats[:t]
                    count += 1
                    break
            else: #particle outside domain
                break
                
# Save
np.save(output_dir + 'traj_lon' + str(grid_start), traj_lon)
np.save(output_dir + 'traj_lat' + str(grid_start), traj_lat)
