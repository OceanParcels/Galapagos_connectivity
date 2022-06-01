# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 18:06:18 2021

@author: quint
"""


from parcels import Field, FieldSet, JITParticle, ScipyParticle 
from parcels import ParticleFile, ParticleSet, Variable, VectorField, ErrorCode
from parcels.tools.converters import GeographicPolar 
from datetime import timedelta as delta
from glob import glob
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import math 
import pickle

############ FUNCTIONS ####################
def Convert_to_single_particles(data, length_simulation, advection_duration, output_frequency, repeatdt, deltatime):

    advection_duration_hours = int(advection_duration*(24/output_frequency))+1 
    advected_timesteps = len(data[1,:])
    number_particles = math.ceil(advected_timesteps/advection_duration_hours)

    ReleaseLon = list(np.load('input/ReleaseLat.npy'))

    release_locations = len(ReleaseLon)

    output_trajectory = np.zeros(((len(data[:,1])*number_particles) ,advection_duration_hours))

    for trajectory_number in range(len(data)):
        
        trajectory = data[trajectory_number,:]
    
        remaining_time_steps = len(trajectory)%advection_duration_hours
    
        full_particle_trajectories = trajectory[0:len(trajectory)-remaining_time_steps]
        
        seperate_particles = np.split(full_particle_trajectories,number_particles-1)
        last_particle = trajectory[len(full_particle_trajectories):(len(full_particle_trajectories)+remaining_time_steps)]
    
        particle_set = int(math.floor(trajectory_number/release_locations))
    
        for particle_number in range((number_particles)): 
            
            if particle_number < number_particles-1:
            
                print('part_set:',particle_set)
                indices_current_particle_set = particle_set * (release_locations*number_particles)
                print('current_ind:',indices_current_particle_set)
                index = indices_current_particle_set + ((trajectory_number%release_locations) + (release_locations*particle_number))
                print('index:',index)
                
                output_trajectory[index,:] = seperate_particles[particle_number]
            
            else:
                print('part_set:',particle_set)
                indices_current_particle_set = particle_set * (release_locations*number_particles)
                print('current_ind:',indices_current_particle_set)
                index = indices_current_particle_set + ((trajectory_number%release_locations) + (release_locations*particle_number))
                print('index:',index)
                
                output_trajectory[index,0:len(last_particle)] = last_particle

    return output_trajectory


############ RELEASE LOCATIONS ####################

ReleaseLat = list(np.load('input/ReleaseLat.npy'))
ReleaseLon = list(np.load('input/ReleaseLon.npy'))

############ SIMULATIONS SPECIFICATIONS ####################


length_simulation = 1800 #unit: days (for how long do we deploy particles)
advection_duration = 60 #unit: days (how long does one particle advect in the fields)
output_frequency = 1     #unit: hours
repeatdt = 24        #unit: hours
deltatime = 1           #dt in hours

data_in = 'input/MITgcm4km'
savename = '2008_2012'

domain = [-92, -88, -2, 2]

############ GET INDICES ####################
def getclosest_ij(lats,lons,latpt,lonpt):    
    """Function to find the index of the closest point to a certain lon/lat value."""
    dist_lat = (lats-latpt)**2          # find squared distance of every point on grid
    dist_lon = (lons-lonpt)**2
    minindex_lat = dist_lat.argmin()    # 1D index of minimum dist_sq element
    minindex_lon = dist_lon.argmin()
    return minindex_lat, minindex_lon   # Get 2D index for latvals and lonvals arrays from 1D index

dfile = Dataset(data_in+'/RGEMS3_Surf_grid.nc')
lon = dfile.variables['XC'][:]
lat = dfile.variables['YC'][:]
loncor = dfile.variables['XG'][:]
latcor = dfile.variables['YG'][:]
iy_min, ix_min = getclosest_ij(lat, lon, domain[2], domain[0])
iy_max, ix_max = getclosest_ij(lat, lon, domain[3], domain[1])

londomain = lon[ix_min:ix_max]
latdomain = lat[iy_min:iy_max]

loncordomain = lon[ix_min:ix_max]
latcordomain = lat[iy_min:iy_max]

############ CREATE FIELDSET ####################

PathToVelocity = data_in + '/RGEMS_20*'
PathToGrid = data_in+'/RGEMS3_Surf_grid.nc'

VelocityField = sorted(glob(PathToVelocity))
GridField = glob(PathToGrid)


files_MITgcm = {'U': {'lon': GridField, 'lat': GridField, 'data': VelocityField},                                             
            'V': {'lon': GridField, 'lat': GridField, 'data': VelocityField}}     

variables = {'U': 'UVEL', 'V': 'VVEL'}
dimensions = {'lon': 'XG', 'lat': 'YG', 'time': 'time'}

indices = {'lon': range(ix_min,ix_max), 
           'lat': range(iy_min,iy_max)}


fieldset = FieldSet.from_mitgcm(files_MITgcm,
                                     variables,
                                     dimensions,
                                     indices = indices)

############ ADD EXTRA FIELDS/CONSTANTS TO FIELDSET ####################

#load fields

landmask = np.load('input/Landmask.npy')
coastgrids = np.load('input/Coastgrids.npy')
coastgrids = np.roll(coastgrids,1,0)
gridnumbermask = np.load('input/GridNumberMask.npy')
gridnumbermask = np.roll(gridnumbermask,1,0)

fieldset.add_constant('advection_duration',advection_duration)    
fieldset.add_constant('lon_max',lon[ix_max] - 0.2)
fieldset.add_constant('lon_min',lon[ix_min] + 0.2)
fieldset.add_constant('lat_max',lat[iy_max] - 0.2)
fieldset.add_constant('lat_min',lat[iy_min] + 0.2)

fieldset.add_field(Field('landmask',
                         data = landmask,
                         lon = londomain,
                         lat = latdomain,
                         mesh='spherical',
                         interp_method = 'nearest'))

fieldset.add_field(Field('coastgrids',
                         data = coastgrids,
                         lon = londomain,
                         lat = latdomain,
                         mesh='spherical',
                         interp_method = 'nearest'))

fieldset.add_field(Field('gridnumbermask',
                         data = gridnumbermask,
                         lon = londomain,
                         lat = latdomain,
                         mesh='spherical',
                         interp_method = 'nearest'))


DistanceFromShore = np.load(('input/distance2shore.npy')
x = np.linspace(domain[0],domain[1],len(londomain))
y = np.linspace(domain[2],domain[3],len(londomain))
lon, lat = np.meshgrid(x,y)
fieldset.add_field(Field('distance2shore', DistanceFromShore, lon, lat))


############ ADD KERNELS ####################


def AdvectionRK4(particle, fieldset, time):
    """ Only advect particles that are not out of bounds"""
    
    if (particle.lon < fieldset.lon_max and
        particle.lon > fieldset.lon_min and
        particle.lat < fieldset.lat_max and
        particle.lat > fieldset.lat_min):
    
        (u1, v1) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        lon1, lat1 = (particle.lon + u1*.5*particle.dt, particle.lat + v1*.5*particle.dt)

        if (lon1 < fieldset.lon_max and
            lon1 > fieldset.lon_min and
            lat1 < fieldset.lat_max and
            lat1 > fieldset.lat_min):

            (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat1, lon1]
            lon2, lat2 = (particle.lon + u2*.5*particle.dt, particle.lat + v2*.5*particle.dt)

            if (lon2 < fieldset.lon_max and
                lon2 > fieldset.lon_min and
                lat2 < fieldset.lat_max and
                lat2 > fieldset.lat_min):

                (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat2, lon2]
                lon3, lat3 = (particle.lon + u3*particle.dt, particle.lat + v3*particle.dt)

                if (lon3 < fieldset.lon_max and
                    lon3 > fieldset.lon_min and
                    lat3 < fieldset.lat_max and
                    lat3 > fieldset.lat_min):

                    (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3]
                    
                    particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
                    particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt


def Age(fieldset, particle, time):
    """ Delete particles when reaching age specified by advection_duration """
    
    particle.age = particle.age + particle.delta_time*60*60   
    

def beachtesting(particle, fieldset, time):
    
    landcheck = fieldset.landmask[time, particle.depth, particle.lat, particle.lon]
    
    if landcheck == 1: 

        particle.beached = 1
        
def coasttesting(particle, fieldset, time):
    
    if (particle.lon < fieldset.lon_max and
        particle.lon > fieldset.lon_min and
        particle.lat < fieldset.lat_max and
        particle.lat > fieldset.lat_min):
    
        particle.coastcell = fieldset.coastgrids[time, particle.depth, particle.lat, particle.lon]
        
def gridnumbertesting(particle, fieldset, time):
    
    if (particle.lon < fieldset.lon_max and
        particle.lon > fieldset.lon_min and
        particle.lat < fieldset.lat_max and
        particle.lat > fieldset.lat_min):
    
        particle.gridnumber = fieldset.gridnumbermask[time, particle.depth, particle.lat, particle.lon]

def SetParticleBack(particle,fieldset,time):
    
    if particle.age > fieldset.advection_duration*86400:
        
        print('test')
        particle.lon = particle.startlon
        particle.lat = particle.startlat
        particle.age = 0     
        
        
        
############ CREATE PARTICLE ####################

class GalapagosParticle(JITParticle):
    distance = Variable('distance', initial = 0) 
    age = Variable('age', dtype=np.float32, initial = 0.)
    beached = Variable('beached', dtype=np.int32, initial = 0.) 
    coastcell = Variable('coastcell', dtype=np.int32, initial = 0.)
    gridnumber = Variable('gridnumber', dtype=np.int32, initial = 0.)
    distancetoshore = Variable('distancetoshore', dtype=np.int32, initial = 0.)
    startlon = Variable('startlon', dtype=np.float32, initial = ReleaseLon)
    startlat = Variable('startlat', dtype=np.float32, initial = ReleaseLat)
    delta_time = Variable('delta_time', dtype=np.float32, initial = deltatime)

if repeatdt == None:
    print('NoRepeat')
    pset = ParticleSet(fieldset = fieldset,
                           pclass = GalapagosParticle,
                           lon = ReleaseLon,
                           lat = ReleaseLat,
                           time = 0)
else:
    print('repeat')
    pset = ParticleSet(fieldset = fieldset,
                           pclass = GalapagosParticle,
                           lon = ReleaseLon,
                           lat = ReleaseLat,
                           repeatdt = delta(hours = repeatdt),
                           time = 0)  

############ EXECUTION PARTICLE SET ####################


outfile = pset.ParticleFile('output/' + savename + '.nc' , outputdt = delta(hours = output_frequency))

kernels = (pset.Kernel(AdvectionRK4) + pset.Kernel(Age) + 
           pset.Kernel(beachtesting) + pset.Kernel(coasttesting) +
           pset.Kernel(gridnumbertesting)+ pset.Kernel(SetParticleBack))


pset.execute(kernels,
             runtime=delta(days=advection_duration),
             dt=delta(hours=deltatime),
             recovery = {ErrorCode.ErrorInterpolation : delete_particle},
             output_file=outfile)

pset.repeatdt = None

pset.execute(kernels,
             runtime=delta(days=length_simulation - advection_duration),
             dt=delta(hours=deltatime),
             recovery = {ErrorCode.ErrorInterpolation : delete_particle},
             output_file=outfile)


outfile.export()
outfile.close()


############################################## CONVERT DATA TO SINGLE PARTICLES ##############################################################

Sim_data = xr.open_dataset('output/' + savename + '.nc')
lon_traj = Sim_data['lon'].data
lat_traj = Sim_data['lat'].data 
age_traj = Sim_data['age'].data 
gridnumber_traj = Sim_data['gridnumber'].data 
coastgrid_traj = Sim_data['coastcell'].data 

age_per_particle = Convert_to_single_particles(age_traj, length_simulation, advection_duration, output_frequency, repeatdt, deltatime)
lon_per_particle = Convert_to_single_particles(lon_traj, length_simulation, advection_duration, output_frequency, repeatdt, deltatime)
lat_per_particle = Convert_to_single_particles(lat_traj, length_simulation, advection_duration, output_frequency, repeatdt, deltatime)
grid_numb_per_particle = Convert_to_single_particles(gridnumber_traj, length_simulation, advection_duration, output_frequency, repeatdt, deltatime)
coastgrid_per_particle = Convert_to_single_particles(coastgrid_traj, length_simulation, advection_duration, output_frequency, repeatdt, deltatime)

trajectory_data = {'age':age_per_particle,
                   'lon':lon_per_particle,
                   'lat':lat_per_particle,
                   'gridnumber':grid_numb_per_particle,
                   'coastgrids':coastgrid_per_particle}

with open('output/' + savename +  '.dictionary', 'wb') as config_dictionary_file:
        
         pickle.dump(trajectory_data, config_dictionary_file)













