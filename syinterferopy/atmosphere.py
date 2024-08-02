#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:46:57 2023

@author: matthew
"""

import pdb


#%%

def atmosphere_topos(n_ifgs, dem_m, delay_grad = -0.0003, delay_var = 0.00005, 
                     zero_delay_h = 10000.):
    """ Create multiple topographically correlated APS for each acquisition.  
    Different to the function below that makes them for a single interferogram. 
    
    Inputs:
        n_ifgs | int | Number of interferogram atmospheres to generate.   
        delay_grad | float | absolute delay gradient.   Units are m delay / m #
                             height.  With -0.0003, ~3000m elevation required 
                             for 1m of delay.  
        delay_var | float | variance of the delays.  Higher value gives more 
                            variety in atmospheres, so bigger interfergoram atmospheres.  
        zero_delay_h | float | Height at which delays tend to zero.  Not tested.  
    Returns:
        atm_topos | r3 ma | interferograms with mask applied.  
        
    History:
        2023_08_24 | MEG | Written
        2024_07_31 | MEG | Update to return atmospheres on each acquisiton.  
        
        
    E.g. GACOS for vesuvius: Vesuvius 1281m high, 2.1m of delay.  
                        sea level: (0m)         2.45m of delay
    """
    
    import numpy as np
    import numpy.ma as ma
    import matplotlib.pyplot as plt
    
    
    # sanity check of how delay is defined.  
    # height on the y axis, delay on the x (as per Bekaert 2015).  
    # Makes sense as heiht is normally vertical.  
    # grad_av =  (0 - 1281) / (2.45 - 2.1)                
    # height at which delays are 0, in m.  Also called zero_delay_height.  
    #  c = 10000                                       
    # # heights = grad_av * delays + c
    # # heights - c / grad_av = delays
    # heights = np.arange(0, 10000, 1)
    # delays = (heights - c) / grad_av
    # f, ax = plt.subplots()
    # ax.plot(delays, heights)

    
    n_atms = n_ifgs + 1                                                                                                     
    delay_grads = delay_grad + (delay_var * np.random.randn(n_atms))
        
    # # debug figure
    # f, ax = plt.subplots(1,1)
    # ax.set_xlabel('Delay (m)')
    # ax.set_ylabel('Height (m)')
    # heights = np.arange(0, 10000, 1)
    # for delay_grad in delay_grads:
    #     delays = (heights - zero_delay_h) / (1 / delay_grad)
    #     ax.plot(delays, heights)
    
    # # generate the delay.  
    # loop through making atms.  Make n_acqs of them, which is 1 more than n_ifgs
    for n_atm in range(n_atms):                                                                                 
        atm_topo = (dem_m - zero_delay_h) / (1/delay_grads[n_atm])
        #atm_topo = (dem_m) / (1/delay_grads[n_atm])
        # if the first atmosphere, initialise array to store outputs.  
        if n_atm == 0:                                                                                          
            atm_topos = ma.zeros((n_atms, atm_topo.shape[0], atm_topo.shape[1]))                                
        atm_topos[n_atm, :, :] = atm_topo                                                       


    
    ## Reference the atmospheres 
    # find the lowest pixel in the DEM
    dem_minimum_arg = np.unravel_index(np.argmin(dem_m), dem_m.shape)                                           
    # use this as a reference pixel for the atmoshperes.  
    atm_topo_offsets = atm_topos[:, dem_minimum_arg[0], dem_minimum_arg[1]]                                     
    # iterate through all atmospheres and set the reference pixel to 0
    for atm_n, atm_topo in enumerate(atm_topos):                                                                
        atm_topo -= atm_topo_offsets[atm_n]                                                                     

    # calculate the difference between the diasy chain of acquisitions 
    # (i.e. shortest temporal baselines)
    atm_topos_diff = ma.diff(atm_topos, axis = 0)                                                               
    
    # # debug figure
    # f, axes = plt.subplots(3,n_atm)
    # for i in range(n_atm):
    #     axes[0,i].plot(ma.ravel(atm_topos[i,]), ma.ravel(dem_m))
    #     axes[0,i].set_xlim(left = ma.min(atm_topos))
    #     axes[1,i].matshow(atm_topos[i,], vmin = np.min(atm_topos), vmax =  np.max(atm_topos))
    #     axes[2,i].matshow(atm_topos_diff[i,], vmin = np.min(atm_topos_diff), vmax =  np.max(atm_topos_diff))
    
    return atm_topos_diff, atm_topos

#%%
    
def atmosphere_topo(dem_m, strength_mean = 56.0, strength_var = 2.0, difference = False):
    """ Given a dem, return a topographically correlated APS, either for a single acquistion
    or for an interferometric pair.
    Inputs:
        dem_m | r4 ma | rank4 masked array, with water masked. Units = metres!  NB.  Surely r2?  
        strength_mean | float | rad/km of delay.  default is 56.0, taken from Fig5 Pinel 2011 (Statovolcanoes...)
        strength_var  | float | variance of rad/km delay.  Default is 2.0, which gives values similar to Fig 5 of above.
        difference | boolean | if False, returns for one acquisitin.  If true, returns for an interferometric pair (ie difference of two acquisitions)

    Outputs:
        ph_topo | r4 ma | topo correlated delay in m.  UNITS ARE M

    2019/09/11 | MEG | written.
    """
    import numpy as np
    import numpy.ma as ma

    satellite_lambda = 0.055465763                  # S1 wavelength in m
    #satellite_lambda = 0.056                       #envisat wavelength in m
    dem = 0.001 * dem_m                        # convert from metres to km

    if difference is False:
        ph_topo = (strength_mean + strength_var * np.random.randn(1)) * dem
        print(f"The use of this function with difference = False is discouraged.  "
              "Use atmosphere_topos instead (and not atmosphere_topo)")
    elif difference is True:
        ph_topo_aq1 = (strength_mean + strength_var * np.random.randn(1)) * dem                         # this is the delay for one acquisition
        ph_topo_aq2 = (strength_mean + strength_var * np.random.randn(1)) * dem                         # and for another
        ph_topo = ph_topo_aq1 - ph_topo_aq2                                                             # interferogram is the difference, still in rad

    else:
        print("'difference' must be either True or False.  Exiting...")
        import sys; sys.exit()

    # convert from rad to m
    ph_topo_m = (ph_topo / (4*np.pi)) * satellite_lambda                               # delay/elevation ratio is taken from a paper (pinel 2011) using Envisat data


    if np.max(ph_topo_m) < 0:                                                       # ensure that it always starts from 0, either increasing or decreasing
        ph_topo_m -= np.max(ph_topo_m)
    else:
        ph_topo_m -= np.min(ph_topo_m)

    ph_topo_m = ma.array(ph_topo_m, mask = ma.getmask(dem_m))
    ph_topo_m -= ma.mean(ph_topo_m)                                                 # mean centre the signal

    return ph_topo_m

#%%
 
def atmosphere_turb(n_atms, lons_mg, lats_mg, method = 'fft', mean_m = 0.02,
                    water_mask = None, difference = False, verbose = False,
                    cov_interpolate_threshold = 1e4, cov_Lc = 2000):
    """ A function to create synthetic turbulent atmospheres based on the  methods in Lohman Simons 2005, or using Andy Hooper and Lin Shen's fft method.  
    Note that due to memory issues, when using the covariance (Lohman) method, largers ones are made by interpolateing smaller ones.  
    Can return atmsopheres for an individual acquisition, or as the difference of two (as per an interferogram).  Units are in metres.  
    
    Inputs:
        n_atms | int | number of atmospheres to generate
        lons_mg | rank 2 array | longitudes of the bottom left corner of each pixel.  
        lats_mg | rank 2 array | latitudes of the bottom left corner of each pixel.  
        method | string | 'fft' or 'cov'.  Cov for the Lohmans Simons (sp?) method, fft for Andy Hooper/Lin Shen's fft method (which is much faster).  Currently no way to set length scale using fft method.  
        mean_m | float | average max or min value of atmospheres that are created.  e.g. if 3 atmospheres have max values of 0.02m, 0.03m, and 0.04m, their mean would be 0.03cm.  
        water_mask | rank 2 array | If supplied, this is applied to the atmospheres generated, convering them to masked arrays.  
        difference | boolean | If difference, two atmospheres are generated and subtracted from each other to make a single atmosphere.  
        verbose | boolean | Controls info printed to screen when running.  
        cov_Lc     | float | length scale of correlation, in metres.  If smaller, noise is patchier, and if larger, smoother.  
        cov_interpolate_threshold | int | if n_pixs is greater than this, images will be generated at size so that the total number of pixels doesn't exceed this.  
                                          e.g. if set to 1e4 (10000, the default) and images are 120*120, they will be generated at 100*100 then upsampled to 120*120.  
        
    
    Outputs:
        ph_turbs | r3 array | n_atms x n_pixs x n_pixs, UNITS ARE M.  Note that if a water_mask is provided, this is applied and a masked array is returned.  
        
    2019/09/13 | MEG | adapted extensively from a simple script
    2020/10/02 | MEG | Change so that a water mask is optional.  
    2020/10/05 | MEG | Change so that meshgrids of the longitudes and latitudes of each pixel are used to set resolution. 
                       Also fix a bug in how cov_Lc is handled, so this is now in meters.  
    2020/10/06 | MEG | Add support for rectangular atmospheres, fix some bugs.  
    2020_03_01 | MEG | Add option to use Lin Shen/Andy Hooper's fft method which is quicker than the covariance method.  
    """
    
    import numpy as np
    import numpy.ma as ma
    from scipy.spatial import distance as sp_distance                                                # geopy also has a distance function.  Rename for safety.  
    from scipy import interpolate as scipy_interpolate
    from syinterferopy.aux import lon_lat_to_ijk
    
    def generate_correlated_noise_cov(pixel_distances, cov_Lc, shape):
        """ given a matrix of pixel distances (in meters) and a length scale for the noise (also in meters),
        generate some 2d spatially correlated noise.  
        Inputs:
            pixel_distances | rank 2 array | pixels x pixels, distance between each on in metres.  
            cov_Lc | float | Length scale over which the noise is correlated.  units are metres.  
            shape | tuple | (nx, ny)  NOTE X FIRST!
        Returns:
            y_2d | rank 2 array | spatially correlated noise.  
        History:
            2019/06/?? | MEG | Written
            2020/10/05 | MEG | Overhauled to be in metres and use scipy cholesky
            2020/10/06 | MEG | Add support for rectangular atmospheres.  
        """
        import scipy
        nx = shape[0]
        ny = shape[1]
        Cd = np.exp((-1 * pixel_distances)/cov_Lc)                                     # from the matrix of distances, convert to covariances using exponential equation
        Cd_L = np.linalg.cholesky(Cd)                                             # ie Cd = CD_L @ CD_L.T      Worse error messages, so best called in a try/except form.  
        #Cd_L = scipy.linalg.cholesky(Cd, lower=True)                               # better error messages than the numpy versio, but can cause crashes on some machines
        x = np.random.randn((ny*nx))                                               # Parsons 2007 syntax - x for uncorrelated noise
        y = Cd_L @ x                                                               # y for correlated noise
        y_2d = np.reshape(y, (ny, nx))                                             # turn back to rank 2
        return y_2d
    
    
    def generate_correlated_noise_fft(nx, ny, std_long, sp):
        """ A function to create synthetic turbulent troposphere delay using an FFT approach. 
        The power of the turbulence is tuned by the weather model at the longer wavelengths.
        
        Inputs:
            nx (int) -- width of troposphere 
            Ny (int) -- length of troposphere 
            std_long (float) -- standard deviation of the weather model at the longer wavelengths. Default = ?
            sp | int | pixel spacing in km
            
        Outputs:
            APS (float): 2D array, Ny * nx, units are m.
            
        History:
            2020_??_?? | LS | Adapted from code by Andy Hooper.  
            2021_03_01 | MEG | Small change to docs and inputs to work with SyInterferoPy
        """
        
        import numpy as np
        import numpy.matlib as npm
        import math
        
        np.seterr(divide='ignore')
    
        cut_off_freq=1/50                                                   # drop wavelengths above 50 km 
        
        x=np.arange(0,int(nx/2))                                            # positive frequencies only
        y=np.arange(0,int(ny/2))                                            # positive frequencies only
        freq_x=np.divide(x,nx*sp)
        freq_y=np.divide(y,ny*sp)
        Y,X=npm.meshgrid(freq_x,freq_y)
        freq=np.sqrt((X*X+Y*Y)/2)                                           # 2D positive frequencies
        
        log_power=np.log10(freq)*-11/3                                      # -11/3 in 2D gives -8/3 in 1D
        ix=np.where(freq<2/3)
        log_power[ix]=np.log10(freq[ix])*-8/3-math.log10(2/3)                 # change slope at 1.5 km (2/3 cycles per km)
        
        bin_power=np.power(10,log_power)
        ix=np.where(freq<cut_off_freq)
        bin_power[ix]=0
        
        APS_power=np.zeros((ny,nx))                                         # mirror positive frequencies into other quadrants
        APS_power[0:int(ny/2), 0:int(nx/2)]=bin_power
        # APS_power[0:int(ny/2), int(nx/2):nx]=npm.fliplr(bin_power)
        # APS_power[int(ny/2):ny, 0:int(nx/2)]=npm.flipud(bin_power)
        # APS_power[int(ny/2):ny, int(nx/2):nx]=npm.fliplr(npm.flipud(bin_power))
        APS_power[0:int(ny/2), int(np.ceil(nx/2)):]=npm.fliplr(bin_power)
        APS_power[int(np.ceil(ny/2)):, 0:int(nx/2)]=npm.flipud(bin_power)
        APS_power[int(np.ceil(ny/2)):, int(np.ceil(nx/2)):]=npm.fliplr(npm.flipud(bin_power))
        APS_filt=np.sqrt(APS_power)
        
        x=np.random.randn(ny,nx)                                            # white noise
        y_tmp=np.fft.fft2(x)
        y_tmp2=np.multiply(y_tmp,APS_filt)                                  # convolve with filter
        y=np.fft.ifft2(y_tmp2)
        APS=np.real(y)
    
        APS=APS/np.std(APS)*std_long                                        #  adjust the turbulence by the weather model at the longer wavelengths.
        APS=APS*0.01                                                        # convert from cm to m
        return APS 
        

    def rescale_atmosphere(atm, atm_mean = 0.02, atm_sigma = 0.005):
        """ a function to rescale a 2d atmosphere with any scale to a mean centered
        one with a min and max value drawn from a normal distribution.  
        Inputs:
            atm | rank 2 array | a single atmosphere.  
            atm_mean | float | average max or min value of atmospheres that are created, in metres.  e.g. if 3 atmospheres have max values of 0.02m, 0.03m, and 0.04m, their mean would be 0.03m
            atm_sigma | float | standard deviation of Gaussian distribution used to generate atmosphere strengths.  
        Returns:
            atm | rank 2 array | a single atmosphere, rescaled to have a maximum signal of around that set by mean_m
        History:
            20YY/MM/DD | MEG | Written
            2020/10/02 | MEG | Standardise throughout to use metres for units.  
        """
        atm -= np.mean(atm)                                                         # mean centre
        atm_strength = (atm_sigma * np.random.randn(1)) + atm_mean                  # maximum strength of signal is drawn from a gaussian distribution, mean and sigma set in metres.  
        if np.abs(np.min(atm)) > np.abs(np.max(atm)):                               # if range of negative numbers is larger
            atm *= (atm_strength / np.abs(np.min(atm)))                              # strength is drawn from a normal distribution  with a mean set by mean_m (e.g. 0.02)
        else:
            atm *= (atm_strength / np.max(atm))                     # but if positive part is larger, rescale in the same way as above.  
        return atm
    
    
    # 0: Check inputs
    if method not in ['fft', 'cov']:
        raise Exception(f"'method' must be either 'fft' (for the fourier transform based method), "
                        f" or 'cov' (for the covariance based method).  {method} was supplied, so exiting.  ")
    
    #1: determine if linear interpolation is required
    ny, nx = lons_mg.shape
    n_pixs = nx * ny
    
    if (n_pixs > cov_interpolate_threshold) and (method == 'cov'):
        if verbose:
            print(f"The number of pixels ({n_pixs}) is larger than 'cov_interpolate_threshold' ({int(cov_interpolate_threshold)}) so images will be created "
                  f"with {int(cov_interpolate_threshold)} pixels and interpolated to the full resolution.  ")
        interpolate = True                                                                                  # set boolean flag
        oversize_factor = n_pixs / cov_interpolate_threshold                                                    # determine how many times too many pixels we have.  
        lons_ds = np.linspace(lons_mg[-1,0], lons_mg[-1,-1], int(nx * (1/np.sqrt(oversize_factor))))        # make a downsampled vector of just the longitudes (square root as number of pixels is a measure of area, and this is length)
        lats_ds = np.linspace(lats_mg[0,0], lats_mg[-1,0], int(ny * (1/np.sqrt(oversize_factor))))          # and for latitudes
        lons_mg_ds = np.repeat(lons_ds[np.newaxis, :], lats_ds.shape, axis = 0)                             # make rank 2 again
        lats_mg_ds = np.repeat(lats_ds[:, np.newaxis], lons_ds.shape, axis = 1)                             # and for latitudes
        ny_generate, nx_generate = lons_mg_ds.shape                                                         # get the size of the downsampled grid we'll be generating at
    else:
        interpolate = False                                                                                 # set boolean flag
        nx_generate = nx                                                                                    # if not interpolating, these don't change.  
        ny_generate = ny
        lons_mg_ds = lons_mg                                                                                # if not interpolating, don't need to downsample.  
        lats_mg_ds = lats_mg
    
    #2: calculate distance between points
    ph_turbs = np.zeros((n_atms, ny_generate, nx_generate))                                                  # initiate output as a rank 3 (ie n_images x ny x nx)
    xyz_m, pixel_spacing = lon_lat_to_ijk(lons_mg_ds, lats_mg_ds)                                           # get pixel positions in metres from origin in lower left corner (and also their size in x and y direction)
    xy = xyz_m[0:2].T                                                                                       # just get the x and y positions (ie discard z), and make lots x 2 (ie two columns)
      
    
    #3: generate atmospheres, using either of the two methods.  
    if difference == True:
        n_atms += 1                                                                                         # if differencing atmospheres, create one extra so that when differencing we are left with the correct number
    
    if method == 'fft':
        for i in range(n_atms):
            ph_turbs[i,:,:] = generate_correlated_noise_fft(nx_generate, ny_generate,    std_long=1, 
                                                           sp = 0.001 * np.mean((pixel_spacing['x'], pixel_spacing['y'])) )      # generate noise using fft method.  pixel spacing is average in x and y direction (and m converted to km) 
            if verbose:
                print(f'Generated {i+1} of {n_atms} single acquisition atmospheres.  ')
            
    else:
        pixel_distances = sp_distance.cdist(xy,xy, 'euclidean')                                                     # calcaulte all pixelwise pairs - slow as (pixels x pixels)       
        Cd = np.exp((-1 * pixel_distances)/cov_Lc)                                     # from the matrix of distances, convert to covariances using exponential equation
        success = False
        while not success:
            try:
                Cd_L = np.linalg.cholesky(Cd)                                             # ie Cd = CD_L @ CD_L.T      Worse error messages, so best called in a try/except form.  
                #Cd_L = scipy.linalg.cholesky(Cd, lower=True)                               # better error messages than the numpy versio, but can cause crashes on some machines
                success = True
            except:
                success = False
        for n_atm in range(n_atms):
            x = np.random.randn((ny_generate*nx_generate))                                               # Parsons 2007 syntax - x for uncorrelated noise
            y = Cd_L @ x                                                               # y for correlated noise
            ph_turb = np.reshape(y, (ny_generate, nx_generate))                                             # turn back to rank 2
            ph_turbs[n_atm,:,:] = ph_turb
            print(f'Generated {n_atm} of {n_atms} single acquisition atmospheres.  ')
        
        
        # nx = shape[0]
        # ny = shape[1]

        

        # return y_2d
        
        # success = 0
        # fail = 0
        # while success < n_atms:
        # #for i in range(n_atms):
        #     try:
        #         ph_turb = generate_correlated_noise_cov(pixel_distances, cov_Lc, (nx_generate,ny_generate))      # generate noise 
        #         ph_turbs[success,:,:] = ph_turb
        #         success += 1
        #         if verbose:
        #             print(f'Generated {success} of {n_atms} single acquisition atmospheres (with {fail} failures).  ')
        #     except:
        #         fail += 0
        #         if verbose:
        #             print(f"'generate_correlated_noise_cov' failed, which is usually due to errors in the cholesky decomposition that Numpy is performing.  The odd failure is normal.  ")
                
        #     # ph_turbs[i,:,:] = generate_correlated_noise_cov(pixel_distances, cov_Lc, (nx_generate,ny_generate))      # generate noise 
        #     # if verbose:
                
                

    #3: possibly interplate to bigger size
    if interpolate:
        if verbose:
            print('Interpolating to the larger size...', end = '')
        ph_turbs_output = np.zeros((n_atms, ny, nx))                                                                          # initiate output at the upscaled size (ie the same as the original lons_mg shape)
        for atm_n, atm in enumerate(ph_turbs):                                                                                # loop through the 1st dimension of the rank 3 atmospheres.  
            f = scipy_interpolate.interp2d(np.arange(0,nx_generate), np.arange(0,ny_generate), atm, kind='linear')           # and interpolate them to a larger size.  First we give it  meshgrids and values for each point
            ph_turbs_output[atm_n,:,:] = f(np.linspace(0, nx_generate, nx), np.linspace(0, ny_generate, ny))                  # then new meshgrids at the original (full) resolution.  
        if verbose:
            print('Done!')
    else:
        ph_turbs_output = ph_turbs                                                                                              # if we're not interpolating, no change needed
       
    # 4: rescale to correct range (i.e. a couple of cm)
    ph_turbs_m = np.zeros(ph_turbs_output.shape)
    for atm_n, atm in enumerate(ph_turbs_output):
        ph_turbs_m[atm_n,] = rescale_atmosphere(atm, mean_m)
                
    # 5: return back to the shape given, which can be a rectangle:
    ph_turbs_m = ph_turbs_m[:,:lons_mg.shape[0],:lons_mg.shape[1]]
    
    if water_mask is not None:
        water_mask_r3 = ma.repeat(water_mask[np.newaxis,], ph_turbs_m.shape[0], axis = 0)
        ph_turbs_m = ma.array(ph_turbs_m, mask = water_mask_r3)
    
    return ph_turbs_m
