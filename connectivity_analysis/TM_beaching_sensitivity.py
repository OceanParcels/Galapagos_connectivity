"""
Created on Wed Jun 01 2022

@author: SLYpma

Creates all transition matrices from particle trajectories based in different beaching timescales.

"""

import numpy as np
import pandas as pd
import math 
from parcels import rng as random
import os
from pathlib import Path
import sys
import xarray as xr

class Functions:
    
    pass

    @classmethod 
    def ParticleOnLand(self, OutputFile):
        land = OutputFile['beached'].data
        land = np.nan_to_num(land)
        summed = np.sum(land, axis = 1)
        
        count = 0
        for som in summed:
            if som > 0:
                count += 1
        PercentageOnLand = count/len(summed)
    
        return PercentageOnLand


    def load_dict(path_to_dict):
        import pickle
        with open(path_to_dict, 'rb') as config_dictionary_file:
         
            # Step 3
            dictionary = pickle.load(config_dictionary_file)
         
            return dictionary        
    
class CTM:
    # to Construct Transition Matrices  
    
    def __init__(self, traj_data_dict, grid_dir, delta_time, beaching_timescale, endloc=False):
        
        # gridsdataframe, this dataframe includes all information about the coastgrids
        self.gridsdataframe = pd.read_csv(grid_dir + 'GridsDataFrame.csv')
        self.beaching_timescale = beaching_timescale  # in days
        
        # the coastcells dataframe, is a matrix with the same shape as the particle trajectories. 
        # It contains all timesteps for each particle and for each timestep
        # it gives a 1 when the particle is in a coastgrid (potential beaching location) and zero otherwise
        self.coastcells = traj_data_dict.get('coastgrids')
        self.lons = traj_data_dict.get('lon')
        self.lats = traj_data_dict.get('lat')
        self.minlon = np.nanmin(self.lons)
        self.maxlon = np.nanmax(self.lons)
        self.minlat = np.nanmin(self.lats)
        self.maxlat = np.nanmax(self.lats)
        
        # the gridnumber matrix, is a matrix with each row representing a particle and each columns 
        # representing a timestep. The columns values are the gridnumbers of the coastgrids.
        # So this matrix tells us in what coastgrid the particle is a what timestep
        self.gridnumbers = traj_data_dict.get('gridnumber')
        self.dt = 60*60*delta_time #delta_time in hours, dt in seconds
        
        # end location of all particles
        if endloc:
            self.end_lon = traj_data_dict.get('lon')[:,-1]
            self.end_lat = traj_data_dict.get('lat')[:,-1]
            self.start_lon = traj_data_dict.get('lon')[:,0]
            self.start_lat = traj_data_dict.get('lat')[:,0]
            self.endloc = np.array([self.end_lat, self.end_lon, self.start_lat, self.start_lon])
        
        # initialize parameters of interest
        self.transition_matrix =  None
        self.beached = None
    
    def BeachProb(self):
        '''function that computes a beaching probability 
        based on the beaching timescale.'''
        
        Prob = math.exp((-self.dt)/(self.beaching_timescale*24*60*60))
        
        return Prob
    
    def CreateEmptyTransMatrix(self):
        '''function that initializes the empty transition matrix, 
        The row numbers represent the release grid and the columns 
        numbers represent the beach grids'''
        
        self.transition_matrix = np.zeros((len(self.gridsdataframe), len(self.gridsdataframe)))
        
    def CreateEmptyBeached(self):
        '''function that initializes the vector with beaching flag'''
        
        self.beached = np.zeros(self.coastcells.shape[0])
        
    def construct_matrix(self):
        '''function that creates the transition matrix and saves 
        for every particle ID a beaching flag'''
        
        # Initialize transition matrix
        self.CreateEmptyTransMatrix()
        
        # Initialize end locations
        self.CreateEmptyBeached()
        
        # Get pseudo random probabilities
        np.random.seed(23)
        randomprob = np.random.random((self.coastcells.shape[0],self.coastcells.shape[1]))
        
        # loop over all different particles
        for particle in range(self.coastcells.shape[0]):

            print(self.coastcells.shape[0]-particle, ' iterations to go')
            
            # loop over all timesteps of a particle.
            for Timestep in range(len(self.coastcells[particle,:])):

                # check the coastcell value. 
                CoastcellValue = self.coastcells[particle,Timestep]
                
                #compute the beaching probability
                Prob = self.BeachProb()
                    
                # Check that particle is not leaving the domain
                if ((self.lons[particle,Timestep] > self.minlon+0.05) and
                    (self.lats[particle,Timestep] > self.minlat+0.05) and
                    (self.lons[particle,Timestep] < self.maxlon-0.05) and
                    (self.lats[particle,Timestep] < self.maxlat-0.05)):
                    
                    # compute a random number between 0-1, when this random number is 
                    # larger than the beaching probability and particle is at a coast
                    # grid, we mark the particle as beached.
                    if ((CoastcellValue == 1) and (randomprob[particle,Timestep] > Prob)):

                        #when the particle is marked as beached, define the end and starting 
                        # grid. This end grid is the gridnumber of the coastgrid where it beached. 
                        EndGrid = int(self.gridnumbers[particle, Timestep])
                        BeginGrid = int(particle%len(self.gridsdataframe))

                        # fill the transition matrix and add a flag that particle has beached 
                        self.transition_matrix[BeginGrid,EndGrid-1] += 1
                        self.beached[particle] = 1

                        # no need to check for future timesteps
                        break                        
                else:

                    break
                        
  
#############################################

parent_dir = str(Path(os.path.dirname(os.path.realpath("__file__"))).parents[0])

# Directories
grid_dir = parent_dir + '/particle_simulation/input/'
traj_dir = parent_dir + '/particle_simulation/output/'
output_dir = 'output/'

#load the trajectory data
path_to_dict = traj_dir + 'Simulation2008_2012_complete_trajectories.dictionary' 
traj_data_dict = Functions.load_dict(path_to_dict)

beaching_parameters = [1, 2, 5, 10, 26, 35]

# make transition matrix for different values of the beaching parameter
for tbeaching in beaching_parameters:
    savename = 'tbeach' + str(tbeaching)
    print(savename)

    # Make the transiton matrix object, Tm_obj - change to loop over different values for the beaching timescale
    Tm_obj = CTM(traj_data_dict, grid_dir, delta_time = 1, beaching_timescale = tbeaching)

    # Compute the transition matrix
    Tm_obj.construct_matrix()

    # Save the transition matrix and the beaching flag
    tm = Tm_obj.transition_matrix
    bflag = Tm_obj.beached
    np.save(output_dir + 'Tm_' + savename, tm)
    np.save(output_dir + 'bflag_' + savename, bflag)

    # Save all end locations
    Tm_obj2 = CTM(traj_data_dict, grid_dir, delta_time = 1, beaching_timescale = tbeaching, endloc=True)
    endloc = Tm_obj2.endloc
    np.save(output_dir + 'endloc', endloc)
