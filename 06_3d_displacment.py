#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:28:18 2024

@author: matthew
"""


import numpy as np
import numpy.ma as ma
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import pdb
from datetime import datetime


import syinterferopy
from syinterferopy.syinterferopy import deformation_wrapper, coherence_mask
from syinterferopy.atmosphere import atmosphere_topos
from syinterferopy.atmosphere import atmosphere_turb
from syinterferopy.aux import col_to_ma, griddata_plot, plot_ifgs                        
from syinterferopy.temporal import generate_temporal_baselines
from syinterferopy.temporal import tc_uniform_inflation, sample_tc_on_acq_dates
from syinterferopy.temporal import seasonal_topographic_ctc



def project_to_LOS(incidence, heading, x_grid, y_grid, z_grid):
    """
    """
    

    # conversion factor
    deg2rad = np.pi/180
    
    # hmmm - why did TJW im use a 2nd different definition?
    sat_inc = 90 - incidence
    sat_az  = 360 - heading
    # sat_inc=Incidence                                                     
    # sat_az=Heading;        
    los_x = -np.cos(sat_az*deg2rad)*np.cos(sat_inc*deg2rad);
    los_y = -np.sin(sat_az*deg2rad)*np.cos(sat_inc*deg2rad);
    los_z = np.sin(sat_inc*deg2rad);
    
    # Unit vector in satellite line of site    
    los_vector = np.array([[los_x],
                           [los_y],
                           [los_z]])                                          
    
    # dot product between displacement vector and LOS vector.  
    los_grid = x_grid*los_vector[0,0] + y_grid*los_vector[1,0] + z_grid*los_vector[2,0]
    
    return los_grid


#%% ################################ Things to set ################################
np.random.seed(0)

srtm_tools_dir = Path('/home/matthew/university_work/15_my_software_releases/SRTM-DEM-tools-3.0.0')             # SRTM-DEM-tools will be needed for this example.  It can be downloaded from https://github.com/matthew-gaddes/SRTM-DEM-tools
gshhs_dir = Path("/home/matthew/university_work/data/coastlines/gshhg/shapefile/GSHHS_shp")                    # coastline information, available from: http://www.soest.hawaii.edu/pwessel/gshhg/

## Campi Flegrei
dem_loc_size = {'centre'        : (14.14, 40.84),
                'side_length'   : (20e3,20e3)}                                   # lon lat width height (m) of interferogram.  
deformation_ll = (14.14, 40.84,)                                                 # deformation lon lat
mogi_kwargs = {'volume_change' : 1e6,                           
                'depth'         : 2000}                                          # both in metres

dem_settings = {"download"              : False,                                 # if don't need to download anymore, faster to set to false
                "void_fill"             : False,                                 # Best to leave as False here, dems can have voids, which it can be worth filling (but can be slow and requires scipy)
                "SRTM3_tiles_folder"    : Path('./SRTM3/'),                            # folder to keep SRTM3 tiles in 
                "water_mask_resolution" : 'f',                                   # resolution of water mask.  c (crude), l (low), i (intermediate), h (high), f (full)
                'gshhs_dir'             : gshhs_dir}                            # srmt-dem-tools needs access to data about coastlines

n_interferograms = 20                                                            # number of interferograms in the time series


# step 00: temporal stuff
d_start_asc = "20200101"
# may need to be shifted by 6 days to replicate reality 
d_start_dec = "20200101"
d_end = "20240101"


# 

topo_aps_noise = 0.01

# # inc should vary across the scene with real data.  
# inc = 23 
# asc_heading = 192.04
# desc_heading = 348.04



#%% Import srtm_dem_tools

sys.path.append(str(srtm_tools_dir))                         
import srtm_dem_tools
from srtm_dem_tools.constructing import SRTM_dem_make

#%% Login details are now needed to download SRTM3 tiles:
    
# ed_username = input(f'Please enter your USGS Earthdata username:  ')
# ed_password = input(f'Please enter your USGS Earthdata password (NB characters will be visible!   ):  ')

# dem_settings['ed_username'] = ed_username
# dem_settings['ed_password'] = ed_password

dem_settings['ed_username'] = ''
dem_settings['ed_password'] = ''

#%% 01: temporal stuf for deformation and topo. correlated APS


# acquisition dates
acq_dates_asc, _ = generate_temporal_baselines(d_start_asc, d_end,
                                               usual_tbaselines = [12])

acq_dates_dec, _ = generate_temporal_baselines(d_start_dec, d_end,
                                               usual_tbaselines = [12])

# deformation cumulative tc, def rate is m/yr
ctc_def, def_dates = tc_uniform_inflation(0.07, d_start_asc, d_end)

# debug
# f, ax = plt.subplots(1)
# ax.plot(np.arange(ctc_def.shape[0]), ctc_def)

# resample to asc and dec acquisition days
ctc_def_asc = sample_tc_on_acq_dates(acq_dates_asc, ctc_def, def_dates)
ctc_def_dec = sample_tc_on_acq_dates(acq_dates_dec, ctc_def, def_dates)


## Topo. correlated APS
_, topo_aps_dates, ctc_topo_aps = seasonal_topographic_ctc(
    datetime.strftime(acq_dates_asc[0], '%Y%m%d'), d_end,
    noise_level=topo_aps_noise
    )

# resample to asc and dec acquisition days
ctc_topo_aps_asc = sample_tc_on_acq_dates(acq_dates_asc, ctc_topo_aps, 
                                          topo_aps_dates)
ctc_topo_aps_dec = sample_tc_on_acq_dates(acq_dates_dec, ctc_topo_aps, 
                                          topo_aps_dates)


# topo aps cumulative time courses
f, ax = plt.subplots()
ax.scatter(acq_dates_asc, ctc_topo_aps_asc, label='Ascending')
ax.scatter(acq_dates_dec, ctc_topo_aps_dec, label='Descending')
ax.set_xlabel('Date')
ax.set_ylabel('Value')
f.suptitle(f'Topo. correlated APS cumulative time course')
f.legend()
ax.grid(True)



#%%  02: make the DEM

dem, lons_mg, lats_mg = SRTM_dem_make(dem_loc_size, **dem_settings)
griddata_plot(dem, lons_mg, lats_mg,
              "01 A digital elevation model (DEM) of Campi Flegrei.  ")

water_mask = ma.getmask(dem)                                                                    



#%% 03: make a deformation signal (in asc and desc)

signals_m = {}                                                                              
ll_extent = [(lons_mg[-1,0], lats_mg[-1,0]), (lons_mg[1,-1], lats_mg[1,-1])]                

# calculate deformation (in 3D).  
_, x_grid, y_grid, z_grid = deformation_wrapper(lons_mg, lats_mg, 
                                                deformation_ll, 'mogi', 
                                                dem, **mogi_kwargs)

# # method 1: reproject into two LOSs
# # reproject 3D into satellite LOS.  
# def_asc = project_to_LOS(inc, asc_heading, x_grid, y_grid, z_grid)
# def_dec = project_to_LOS(inc, desc_heading, x_grid, y_grid, z_grid)

# # rescale here

# # signals_m["deformation"] = ma.array(los_grid, mask = water_mask)
# griddata_plot(def_asc, lons_mg, lats_mg, 
#               "06a Deformaiton signal - ascending", dem_mode = False)                         

# griddata_plot(def_dec, lons_mg, lats_mg, 
#               "06b Deformaiton signal - descending", dem_mode = False)                         


# Method 2: just U and E
x_grid_rescaled = (x_grid - np.min(x_grid)) / (np.max(x_grid) - np.min(x_grid))
y_grid_rescaled = (y_grid - np.min(y_grid)) / (np.max(y_grid) - np.min(y_grid))




#%% 04:  make a topograhically correlated atmospheric phase screen (APS)

# method 1: no seasonal variation
# note that this functions takes n_ifgs as default (so -1 from n_acq)
# _, atm_topos_asc = atmosphere_topos(len(acq_dates_asc) -1 , dem, 
#                                     delay_grad = -0.0003, delay_var = 0.00005,
#                                     zero_delay_h = 10000.)

# _, atm_topos_dec = atmosphere_topos(len(acq_dates_dec) -1 , dem, 
#                                     delay_grad = -0.0003, delay_var = 0.00005,
#                                     zero_delay_h = 10000.)

# f, ax = plt.subplots(1,1)
# ax.matshow(atm_topos_asc[0,:,:])

# method 2: seasonal variation
# rescale to lie in [0, 1]
dem_rescaled = (dem - ma.min(dem)) / (ma.max(dem) - ma.min(dem))




#%% 05: Make a turbulent APS

# ascending
atm_turbs_asc = atmosphere_turb(len(acq_dates_asc), lons_mg, lats_mg, 
                              method = 'fft', mean_m = 0.02, 
                              water_mask = water_mask, difference = False)                                      

# descending
atm_turbs_dec = atmosphere_turb(len(acq_dates_dec), lons_mg, lats_mg, 
                               method = 'fft', mean_m = 0.02, 
                               water_mask = water_mask, difference = False)                                      

griddata_plot(atm_turbs_asc[0,], lons_mg, lats_mg, f" Turbulent APS, ", 
              dem_mode = False)                       

#%% Combine in two LOSs


# # combine ascending
# los_asc = ((ctc_def_asc[:,:,np.newaxis] * def_asc[np.newaxis,]) + \
#             atm_topos_asc + atm_turbs_asc)

# los_asc_ifgs = ma.diff(los_asc, axis = 0)

# griddata_plot(los_asc_ifgs[1,], lons_mg, lats_mg, f"ascending ifg.", 
#               dem_mode = False)                       


# # combine descending
# los_dec = ((ctc_def_dec[:,:,np.newaxis] * def_dec[np.newaxis,]) + \
#             atm_topos_dec + atm_turbs_dec)

# los_dec_ifgs = ma.diff(los_dec, axis = 0)

# griddata_plot(los_dec_ifgs[1,], lons_mg, lats_mg, f"descending ifg.", 
#               dem_mode = False)                       

#%% Or, make in E and U

e_disp = ((ctc_def_asc[:,:,np.newaxis] * x_grid_rescaled[np.newaxis,]) + \
           (ctc_topo_aps_asc[:,:,np.newaxis] * dem_rescaled[np.newaxis,]) + \
            atm_turbs_asc)


u_disp = ((ctc_def_dec[:,:,np.newaxis] * x_grid_rescaled[np.newaxis,]) + \
           (ctc_topo_aps_dec[:,:,np.newaxis] * dem_rescaled[np.newaxis,]) + \
            atm_turbs_dec)




griddata_plot(e_disp[-1] - e_disp[0] , lons_mg, lats_mg, f"East deformation", 
              dem_mode = False)                       

#%%