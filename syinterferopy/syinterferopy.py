#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 15:22:56 2020

@author: matthew
"""




#%%

def deformation_wrapper(lons_mg, lats_mg, deformation_ll, source, dem = None, 
                        asc_or_desc = 'asc', incidence = 23, **kwargs):
    """ A function to prepare grids of pixels and deformation sources specified in lon lat for use with 
    deformation generating functions that work in metres.  Note that different sources require different arguments
    (e.g. opening makes sense for a dyke or sill, but not for a Mogi source, but volume change does for a Mogi
     source and not for a dyke or sill).  Therefore, after setting "source" a selection of kwargs must be passed 
    to the function.   
    E.g. for a Mogi source:
        mogi_kwargs = {'volume_change' : 1e6,                           
                       'depth'         : 2000}                                                 # both in metres
        
    Or a dyke:
        dyke_kwargs = {'strike' : 0,                                    # degrees
                       'top_depth' : 1000,                              # metres.  
                       'bottom_depth' : 3000,
                       'length' : 5000,
                       'dip' : 80,
                       'opening' : 0.5}
    Or a sill:
        sill_kwargs = {'strike' : 0,
                       'depth' : 3000,
                       'width' : 5000,
                       'length' : 5000,
                       'dip' : 1,
                       'opening' : 0.5}
    Or an earthquake:
        quake_ss_kwargs  = {'strike' : 0,
                            'dip' : 80,
                            'length' : 5000,
                            'rake' : 0,
                            'slip' : 1,
                            'top_depth' : 4000,
                            'bottom_depth' : 8000}
    
    deformation_eq_dyke_sill  and deformation_mogi have more information on these arguments, too.  
    
    Inputs:
        lons_mg | rank 2 array | longitudes of the bottom left corner of each pixel.  
        lats_mg | rank 2 array | latitudes of the bottom left corner of each pixel.  
        deformation_ll | tuple | (lon, lat) of centre of deformation source.  
        source | string | mogi or quake or dyke or sill
        dem | rank 2 masked array or None | The dem, with water masked.   If not supplied (None),
                                            then an array (and not a masked array) of deformation is returned.  
        asc_or_dec | string | 'asc' or 'desc' or 'random'.  If set to 'random', 50% chance of each.  
        incidence | float | satellite incidence angle.  
        **kwargs | various parameters required for each type of source.  E.g. opening, as opposed to slip or volumne change etc.  
    
    Returns:
        los_grid | rank 2 masked array | displacment in satellite LOS at each location, with the same mask as the dem.  
        x_grid | rank 2 array | x displacement at each location
        y_grid | rank 2 array | y displacement at each location
        z_grid | rank 2 array | z displacement at each location
    
    History:
        2020/08/07 | MEG | Written.  
        2020/10/09 | MEG | Update so that the DEM is optional.  
        2021_05_18 | MEG | Fix bug in number of pixels in a degree (hard coded as 1201, now calcualted from lon and lat grids)
    """
    
    import numpy as np    
    import numpy.ma as ma
    from syinterferopy.aux import ll2xy, lon_lat_to_ijk
    
    
    # 1: deal with lons and lats, and get pixels in metres from origin at lower left.  
    pixs2deg_x = 1 / (lons_mg[0,1] - lons_mg[0,0])                                              # the number of pixels in 1 deg of longitude
    pixs2deg_y = 1 / (lats_mg[0,0] - lats_mg[1,0])                                              # the number of pixels in 1 deg of latitude
    pixs2deg = np.mean([pixs2deg_x, pixs2deg_y])
    dem_ll_extent = [(lons_mg[-1,0], lats_mg[-1,-1]), (lons_mg[1,-1], lats_mg[1,0])]                # [lon lat tuple of lower left corner, lon lat tuple of upper right corner]
    xyz_m, pixel_spacing = lon_lat_to_ijk(lons_mg, lats_mg)                                                    # get pixel positions in metres from origin in lower left corner (and also their size in x and y direction)
    
    
    # import sys
    # sys.path.append("/home/matthew/university_work/python_stuff/python_scripts")
    # from small_plot_functions import matrix_show
    # import matplotlib.pyplot as plt
    # f, axes = plt.subplots(1,2)
    # matrix_show(np.reshape(xyz_m[0,:], (lons_mg.shape[0], lons_mg.shape[1])), ax = axes[0], fig = f)
    # matrix_show(np.reshape(xyz_m[1,:], (lons_mg.shape[0], lons_mg.shape[1])), ax = axes[1], fig = f)
    
    # 1: Make a satellite look vector.  
    if asc_or_desc == 'asc':
        heading = 192.04
    elif asc_or_desc == 'desc':
        heading = 348.04
    elif asc_or_desc == 'random':
        if (-0.5+np.random.rand()) < 0.5:
            heading = 192.04                                                            # Heading (azimuth) of satellite measured clockwise from North, in degrees, half are descending
        else:
            heading = 348.04                                                            # Heading (azimuth) of satellite measured clockwise from North, in degrees, half are ascending
    else:
        raise Exception(f"'asc_or_desc' must be either 'asc' or 'desc' or 'random', but is currently {asc_or_desc}.  Exiting.   ")

    # matlab implementation    
    deg2rad = np.pi/180
    sat_inc = 90 - incidence
    sat_az  = 360 - heading
#    sat_inc=Incidence                                                      # hmmm - why did T (TJW) im use a 2nd different definition?
 #   sat_az=Heading;        
    los_x=-np.cos(sat_az*deg2rad)*np.cos(sat_inc*deg2rad);
    los_y=-np.sin(sat_az*deg2rad)*np.cos(sat_inc*deg2rad);
    los_z=np.sin(sat_inc*deg2rad);
    los_vector = np.array([[los_x],
                            [los_y],
                            [los_z]])                                          # Unit vector in satellite line of site    
    # my Python implementaiton
    # look = np.array([[np.sin(heading)*np.sin(incidence)],                     # Looks to be a radians / vs degrees error here.  
    #               [np.cos(heading)*np.sin(incidence)],
    #               [np.cos(incidence)]])                                      # ground to satelite unit vector



    # 2: calculate deformation location in the new metres from lower left coordinate system.  
    deformation_xy = ll2xy(np.asarray(dem_ll_extent[0])[np.newaxis,:], pixs2deg,                # lon lat of lower left coerner, number of pixels in 1 degree
                           np.asarray(deformation_ll)[np.newaxis,:])                            # long lat of point of interext (deformation centre)
                                                                                              
    
    deformation_m = np.array([[deformation_xy[0,0] * pixel_spacing['x'], deformation_xy[0,1] * pixel_spacing['y']]])       # convert from number of pixels from lower left corner to number of metres from lower left corner, 1x2 array.  
    
    
    
    
    #import pdb; pdb.set_trace()
    
    # 3: Calculate the deformation:
    if source == 'mogi':
        model_params = np.array([deformation_m[0,0], deformation_m[0,1], 
                                 kwargs['depth'], kwargs['volume_change']])[:,np.newaxis]
        
        # U = 3d displacement, xyz are rows, each point is a column.  
        U = deformation_Mogi(model_params, xyz_m, 0.25,30e9)                                                                   
    elif (source == 'quake') or (source == 'dyke') or (source == 'sill'):
        U = deformation_eq_dyke_sill(source, (deformation_m[0,0], 
                                              deformation_m[0,1]), xyz_m, 
                                     **kwargs)                        
    else:
        raise Exception(f"'source' can be eitehr 'mogi', 'quake', 'dyke', or"
                        " 'sill', but not {source}.  Exiting.  ")
    
    # 4: convert the xyz deforamtion in movement in direction of satellite 
    # LOS (ie. upwards = positive, despite being range shortening)    
    x_grid = np.reshape(U[0,], (lons_mg.shape[0], lons_mg.shape[1]))
    y_grid = np.reshape(U[1,], (lons_mg.shape[0], lons_mg.shape[1]))
    z_grid = np.reshape(U[2,], (lons_mg.shape[0], lons_mg.shape[1]))
    los_grid = x_grid*los_vector[0,0] + y_grid*los_vector[1,0] + z_grid*los_vector[2,0]

    if dem is not None:
        # mask the water parts of the scene. Note that this can reduce the max  
        # of defo_m as parts of the signal may then be masked out.  
        los_grid = ma.array(los_grid, mask = ma.getmask(dem))                                           
        
    return los_grid, x_grid, y_grid, z_grid
 





#%%


def deformation_eq_dyke_sill(source, source_xy_m, xyz_m, **kwargs):    
    """
    A function to create deformation patterns for either an earthquake, dyke or sill.   Uses the Okada function from PyInSAR: https://github.com/MITeaps/pyinsar
    To aid in readability, different sources take different parameters (e.g. slip for a quake, opening for a dyke)
    are passed separately as kwargs, even if they ultimately go into the same field in the model parameters.  
    
    A quick recap on definitions:
        strike - measured clockwise from 0 at north, 180 at south.  fault dips to the right of this.  hanging adn fo
        dip - measured from horizontal, 0 for horizontal, 90 for vertical.  
        rake - direction the hanging wall moves during rupture, measured relative to strike, anticlockwise is positive, so:
            0 for left lateral ss 
            180 (or -180) for right lateral ss
            -90 for normal
            90 for thrust

    Inputs:
        source | string | quake or dyke or sill
        source_xy_m | tuple | x and y location of centre of source, in metres.  
        xyz_m | rank2 array | x and y locations of all points in metres.  0,0 is top left?  

        
        examples of kwargs:
            
        quake_normal = {'strike' : 0,
                        'dip' : 70,
                        'length' : 5000,
                        'rake' : -90,
                        'slip' : 1,
                        'top_depth' : 4000,
                        'bottom_depth' : 8000}
        
        quake_thrust = {'strike' : 0,
                        'dip' : 30,
                        'length' : 5000,
                        'rake' : 90,
                        'slip' : 1,
                        'top_depth' : 4000,
                        'bottom_depth' : 8000}
        
        quake_ss = {'strike' : 0,
                    'dip' : 80,
                    'length' : 5000,
                    'rake' : 0,
                    'slip' : 1,
                    'top_depth' : 4000,
                    'bottom_depth' : 8000}
        
        dyke = {'strike' : 0,
                'top_depth' : 1000,
                'bottom_depth' : 3000,
                'length' : 5000,
                'dip' : 80,
                'opening' : 0.5}
        
        sill = {'strike' : 0,
                'depth' : 3000,
                'width' : 5000,
                'length' : 5000,
                'dip' : 1,
                'opening' : 0.5}
        
    Returns:
        x_grid | rank 2 array | displacment in x direction for each point (pixel on Earth's surface)
        y_grid | rank 2 array | displacment in y direction for each point (pixel on Earth's surface)
        z_grid | rank 2 array | displacment in z direction for each point (pixel on Earth's surface)
        los_grid | rank 2 array | change in satellite - ground distance, in satellite look angle direction. Need to confirm if +ve is up or down.  
        
    History:
        2020/08/05 | MEG | Written
        2020/08/21 | MEG | Switch from disloc3d.m function to compute_okada_displacement.py functions.  
    """    
    import numpy as np
    from syinterferopy.pyinsar_okada import compute_okada_displacement
    
    # 1:  Setting for elastic parameters.  
    lame = {'lambda' : 2.3e10,                                                         # elastic modulus (Lame parameter, units are pascals)
            'mu'     : 2.3e10}                                                         # shear modulus (Lame parameter, units are pascals)
    v = lame['lambda'] / (2*(lame['lambda'] + lame['mu']))                             #  calculate poisson's ration
      
    
    
    # import matplotlib.pyplot as plt
    # both_arrays = np.hstack((np.ravel(coords), np.ravel(xyz_m)))
    # f, axes = plt.subplots(1,2)
    # axes[0].imshow(coords, aspect = 'auto', vmin = np.min(both_arrays), vmax = np.max(both_arrays))                 # goes from -1e4 to 1e4
    # axes[1].imshow(xyz_m, aspect = 'auto', vmin = np.min(both_arrays), vmax = np.max(both_arrays))                  # goes from 0 to 2e4
    if source == 'quake':
        opening = 0
        slip = kwargs['slip']
        rake = kwargs['rake']
        width = kwargs['bottom_depth'] - kwargs['top_depth']
        # centroid_depth = np.mean((kwargs['bottom_depth'] - kwargs['top_depth']))
        centroid_depth = np.mean((kwargs['bottom_depth'], kwargs['top_depth']))
    elif source == 'dyke':                                                                               # ie dyke or sill
        opening = kwargs['opening']
        slip = 0
        rake = 0
        width = kwargs['bottom_depth'] - kwargs['top_depth']
        centroid_depth = np.mean((kwargs['bottom_depth'], kwargs['top_depth']))
    elif source == 'sill':                                                                               # ie dyke or sill
        opening = kwargs['opening']
        slip = 0
        rake = 0
        centroid_depth = kwargs['depth']
        width = kwargs['width']
    else:
        raise Exception(f"'Source' must be either 'quake', 'dyke', or 'sill', but is set to {source}.  Exiting.")
        
    
        
    # 3:  compute deformation using Okada function
    U = compute_okada_displacement(source_xy_m[0], source_xy_m[1],                    # x y location, in metres
                                   centroid_depth,                                    # fault_centroid_depth, guess metres?  
                                   np.deg2rad(kwargs['strike']),
                                   np.deg2rad(kwargs['dip']),
                                   kwargs['length'], width,                           # length and width, in metres
                                   np.deg2rad(rake),                                  # rake, in rads
                                   slip, opening,                                     # slip (if quake) or opening (if dyke or sill)
                                   v, xyz_m[0,], xyz_m[1,:])                          # poissons ratio, x and y coords of surface locations.
       
    return U



#%%

def deformation_Mogi(m,xloc,nu,mu):
    """
    Computes displacements, strains and stresses from a point (Mogi) source. 
    Inputs m and xloc can be matrices; for multiple models, the deformation fields from each are summed.

    Inputs:
        m = 4xn volume source geometry (length; length; length; length^3)
            (x-coord, y-coord, depth(+), volume change)
     xloc = 3xs matrix of observation coordinates (length)
       nu = Poisson's ratio
       mu = shear modulus (if omitted, default value is unity)
    
    Outputs:
        U = 3xs matrix of displacements (length)
            (Ux,Uy,Uz)
        D = 9xn matrix of displacement derivatives
            (Dxx,Dxy,Dxz,Dyx,Dyy,Dyz,Dzx,Dzy,Dzz)
        S = 6xn matrix of stresses
            (Sxx,Sxy,Sxz,Syy,Syz,Szz)
    
    History:
        For information on the basis for this code see:
        Okada, Y. Internal deformation due to shear and tensile faults in a half-space, Bull. Seismol. Soc. Am., 82, 1018-1049, 1992.
        1998/06/17 | Peter Cervelli.
        2000/11/03 | Peter Cervelli, Revised 
        2001/08/21 | Kaj Johnson,  Fixed a bug ('*' multiplication should have been '.*'), , 2001. Kaj Johnson
        2018/03/31 | Matthw Gaddes, converted to Python3  #, but only for U (displacements)
    """
    #import scipy.io
    import numpy as np
    
    _, n_data = xloc.shape
    _, models = m.shape
    #Lambda=2*mu*nu/(1-2*nu)
    U=np.zeros((3,n_data))                        # set up the array to store displacements
                  
    for i in range(models):                         # loop through each of the defo sources
        C=m[3]/(4*np.pi)
        x=xloc[0,:]-float(m[0,i])                          # difference in distance from centre of source (x)
        y=xloc[1,:]-float(m[1,i])                         # difference in distance from centre of source (y)
        z=xloc[2,:]
        d1=m[2,i]-z
        d2=m[2,i]+z
        R12=x**2+y**2+d1**2
        R22=x**2+y**2+d2**2
        R13=R12**1.5
        R23=R22**1.5
        R15=R12**2.5
        R25=R22**2.5
        R17=R12**3.5
        R27=R12**3.5
            
        #Calculate displacements
        U[0,:] = U[0,:] + C*( (3 - 4*nu)*x/R13 + x/R23 + 6*d1*x*z/R15 )
        U[1,:] = U[1,:] + C*( (3 - 4*nu)*y/R13 + y/R23 + 6*d1*y*z/R15 )
        U[2,:] = U[2,:] + C*( (3 - 4*nu)*d1/R13 + d2/R23 - 2*(3*d1**2 - R12)*z/R15)
    return U

    
#%%

# def atmosphere_topo(dem_m, strength_mean = 56.0, strength_var = 2.0, 
#                     difference = False):
#     """ Given a dem, return a topographically correlated APS, either for a single acquistion
#     or for an interferometric pair.
#     Inputs:
#         dem_m | r4 ma | rank4 masked array, with water masked. Units = metres!
#         strength_mean | float | rad/km of delay.  default is 56.0, taken from Fig5 Pinel 2011 (Statovolcanoes...)
#         strength_var  | float | variance of rad/km delay.  Default is 2.0, which gives values similar to Fig 5 of above.
#         difference | boolean | if False, returns for one acquisitin.  If true, returns for an interferometric pair (ie difference of two acquisitions)

#     Outputs:
#         ph_topo | r4 ma | topo correlated delay in m.  UNITS ARE M

#     2019/09/11 | MEG | written.
#     """
#     import numpy as np
#     import numpy.ma as ma

#     envisat_lambda = 0.056                       #envisat/S1 wavelength in m
#     dem = 0.001 * dem_m                        # convert from metres to km

#     if difference is False:
#         ph_topo = (strength_mean + strength_var * np.random.randn(1)) * dem
#     elif difference is True:
#         ph_topo_aq1 = (strength_mean + strength_var * np.random.randn(1)) * dem                         # this is the delay for one acquisition
#         ph_topo_aq2 = (strength_mean + strength_var * np.random.randn(1)) * dem                         # and for another
#         ph_topo = ph_topo_aq1 - ph_topo_aq2                                                             # interferogram is the difference, still in rad



#     else:
#         print("'difference' must be either True or False.  Exiting...")
#         import sys; sys.exit()

#     # convert from rad to m
#     ph_topo_m = (ph_topo / (4*np.pi)) * envisat_lambda                               # delay/elevation ratio is taken from a paper (pinel 2011) using Envisat data


#     if np.max(ph_topo_m) < 0:                                                       # ensure that it always start from 0, either increasing or decreasing
#         ph_topo_m -= np.max(ph_topo_m)
#     else:
#         ph_topo_m -= np.min(ph_topo_m)

#     ph_topo_m = ma.array(ph_topo_m, mask = ma.getmask(dem_m))
#     ph_topo_m -= ma.mean(ph_topo_m)                                                 # mean centre the signal

#     return ph_topo_m

# #%%
 
# def atmosphere_turb(n_atms, lons_mg, lats_mg, method = 'fft', mean_m = 0.02,
#                     water_mask = None, difference = False, verbose = False,
#                     cov_interpolate_threshold = 1e4, cov_Lc = 2000):
#     """ A function to create synthetic turbulent atmospheres based on the  methods in Lohman Simons 2005, or using Andy Hooper and Lin Shen's fft method.  
#     Note that due to memory issues, when using the covariance (Lohman) method, largers ones are made by interpolateing smaller ones.  
#     Can return atmsopheres for an individual acquisition, or as the difference of two (as per an interferogram).  Units are in metres.  
    
#     Inputs:
#         n_atms | int | number of atmospheres to generate
#         lons_mg | rank 2 array | longitudes of the bottom left corner of each pixel.  
#         lats_mg | rank 2 array | latitudes of the bottom left corner of each pixel.  
#         method | string | 'fft' or 'cov'.  Cov for the Lohmans Simons (sp?) method, fft for Andy Hooper/Lin Shen's fft method (which is much faster).  Currently no way to set length scale using fft method.  
#         mean_m | float | average max or min value of atmospheres that are created.  e.g. if 3 atmospheres have max values of 0.02m, 0.03m, and 0.04m, their mean would be 0.03cm.  
#         water_mask | rank 2 array | If supplied, this is applied to the atmospheres generated, convering them to masked arrays.  
#         difference | boolean | If difference, two atmospheres are generated and subtracted from each other to make a single atmosphere.  
#         verbose | boolean | Controls info printed to screen when running.  
#         cov_Lc     | float | length scale of correlation, in metres.  If smaller, noise is patchier, and if larger, smoother.  
#         cov_interpolate_threshold | int | if n_pixs is greater than this, images will be generated at size so that the total number of pixels doesn't exceed this.  
#                                           e.g. if set to 1e4 (10000, the default) and images are 120*120, they will be generated at 100*100 then upsampled to 120*120.  
        
    
#     Outputs:
#         ph_turbs | r3 array | n_atms x n_pixs x n_pixs, UNITS ARE M.  Note that if a water_mask is provided, this is applied and a masked array is returned.  
        
#     2019/09/13 | MEG | adapted extensively from a simple script
#     2020/10/02 | MEG | Change so that a water mask is optional.  
#     2020/10/05 | MEG | Change so that meshgrids of the longitudes and latitudes of each pixel are used to set resolution. 
#                        Also fix a bug in how cov_Lc is handled, so this is now in meters.  
#     2020/10/06 | MEG | Add support for rectangular atmospheres, fix some bugs.  
#     2020_03_01 | MEG | Add option to use Lin Shen/Andy Hooper's fft method which is quicker than the covariance method.  
#     """
    
#     import numpy as np
#     import numpy.ma as ma
#     from scipy.spatial import distance as sp_distance                                                # geopy also has a distance function.  Rename for safety.  
#     from scipy import interpolate as scipy_interpolate
#     from syinterferopy.aux import lon_lat_to_ijk
    
#     def generate_correlated_noise_cov(pixel_distances, cov_Lc, shape):
#         """ given a matrix of pixel distances (in meters) and a length scale for the noise (also in meters),
#         generate some 2d spatially correlated noise.  
#         Inputs:
#             pixel_distances | rank 2 array | pixels x pixels, distance between each on in metres.  
#             cov_Lc | float | Length scale over which the noise is correlated.  units are metres.  
#             shape | tuple | (nx, ny)  NOTE X FIRST!
#         Returns:
#             y_2d | rank 2 array | spatially correlated noise.  
#         History:
#             2019/06/?? | MEG | Written
#             2020/10/05 | MEG | Overhauled to be in metres and use scipy cholesky
#             2020/10/06 | MEG | Add support for rectangular atmospheres.  
#         """
#         import scipy
#         nx = shape[0]
#         ny = shape[1]
#         Cd = np.exp((-1 * pixel_distances)/cov_Lc)                                     # from the matrix of distances, convert to covariances using exponential equation
#         Cd_L = np.linalg.cholesky(Cd)                                             # ie Cd = CD_L @ CD_L.T      Worse error messages, so best called in a try/except form.  
#         #Cd_L = scipy.linalg.cholesky(Cd, lower=True)                               # better error messages than the numpy versio, but can cause crashes on some machines
#         x = np.random.randn((ny*nx))                                               # Parsons 2007 syntax - x for uncorrelated noise
#         y = Cd_L @ x                                                               # y for correlated noise
#         y_2d = np.reshape(y, (ny, nx))                                             # turn back to rank 2
#         return y_2d
    
    
#     def generate_correlated_noise_fft(nx, ny, std_long, sp):
#         """ A function to create synthetic turbulent troposphere delay using an FFT approach. 
#         The power of the turbulence is tuned by the weather model at the longer wavelengths.
        
#         Inputs:
#             nx (int) -- width of troposphere 
#             Ny (int) -- length of troposphere 
#             std_long (float) -- standard deviation of the weather model at the longer wavelengths. Default = ?
#             sp | int | pixel spacing in km
            
#         Outputs:
#             APS (float): 2D array, Ny * nx, units are m.
            
#         History:
#             2020_??_?? | LS | Adapted from code by Andy Hooper.  
#             2021_03_01 | MEG | Small change to docs and inputs to work with SyInterferoPy
#         """
        
#         import numpy as np
#         import numpy.matlib as npm
#         import math
        
#         np.seterr(divide='ignore')
    
#         cut_off_freq=1/50                                                   # drop wavelengths above 50 km 
        
#         x=np.arange(0,int(nx/2))                                            # positive frequencies only
#         y=np.arange(0,int(ny/2))                                            # positive frequencies only
#         freq_x=np.divide(x,nx*sp)
#         freq_y=np.divide(y,ny*sp)
#         Y,X=npm.meshgrid(freq_x,freq_y)
#         freq=np.sqrt((X*X+Y*Y)/2)                                           # 2D positive frequencies
        
#         log_power=np.log10(freq)*-11/3                                      # -11/3 in 2D gives -8/3 in 1D
#         ix=np.where(freq<2/3)
#         log_power[ix]=np.log10(freq[ix])*-8/3-math.log10(2/3)                 # change slope at 1.5 km (2/3 cycles per km)
        
#         bin_power=np.power(10,log_power)
#         ix=np.where(freq<cut_off_freq)
#         bin_power[ix]=0
        
#         APS_power=np.zeros((ny,nx))                                         # mirror positive frequencies into other quadrants
#         APS_power[0:int(ny/2), 0:int(nx/2)]=bin_power
#         # APS_power[0:int(ny/2), int(nx/2):nx]=npm.fliplr(bin_power)
#         # APS_power[int(ny/2):ny, 0:int(nx/2)]=npm.flipud(bin_power)
#         # APS_power[int(ny/2):ny, int(nx/2):nx]=npm.fliplr(npm.flipud(bin_power))
#         APS_power[0:int(ny/2), int(np.ceil(nx/2)):]=npm.fliplr(bin_power)
#         APS_power[int(np.ceil(ny/2)):, 0:int(nx/2)]=npm.flipud(bin_power)
#         APS_power[int(np.ceil(ny/2)):, int(np.ceil(nx/2)):]=npm.fliplr(npm.flipud(bin_power))
#         APS_filt=np.sqrt(APS_power)
        
#         x=np.random.randn(ny,nx)                                            # white noise
#         y_tmp=np.fft.fft2(x)
#         y_tmp2=np.multiply(y_tmp,APS_filt)                                  # convolve with filter
#         y=np.fft.ifft2(y_tmp2)
#         APS=np.real(y)
    
#         APS=APS/np.std(APS)*std_long                                        #  adjust the turbulence by the weather model at the longer wavelengths.
#         APS=APS*0.01                                                        # convert from cm to m
#         return APS 
        

#     def rescale_atmosphere(atm, atm_mean = 0.02, atm_sigma = 0.005):
#         """ a function to rescale a 2d atmosphere with any scale to a mean centered
#         one with a min and max value drawn from a normal distribution.  
#         Inputs:
#             atm | rank 2 array | a single atmosphere.  
#             atm_mean | float | average max or min value of atmospheres that are created, in metres.  e.g. if 3 atmospheres have max values of 0.02m, 0.03m, and 0.04m, their mean would be 0.03m
#             atm_sigma | float | standard deviation of Gaussian distribution used to generate atmosphere strengths.  
#         Returns:
#             atm | rank 2 array | a single atmosphere, rescaled to have a maximum signal of around that set by mean_m
#         History:
#             20YY/MM/DD | MEG | Written
#             2020/10/02 | MEG | Standardise throughout to use metres for units.  
#         """
#         atm -= np.mean(atm)                                                         # mean centre
#         atm_strength = (atm_sigma * np.random.randn(1)) + atm_mean                  # maximum strength of signal is drawn from a gaussian distribution, mean and sigma set in metres.  
#         if np.abs(np.min(atm)) > np.abs(np.max(atm)):                               # if range of negative numbers is larger
#             atm *= (atm_strength / np.abs(np.min(atm)))                              # strength is drawn from a normal distribution  with a mean set by mean_m (e.g. 0.02)
#         else:
#             atm *= (atm_strength / np.max(atm))                     # but if positive part is larger, rescale in the same way as above.  
#         return atm
    
    
#     # 0: Check inputs
#     if method not in ['fft', 'cov']:
#         raise Exception(f"'method' must be either 'fft' (for the fourier transform based method), "
#                         f" or 'cov' (for the covariance based method).  {method} was supplied, so exiting.  ")
    
#     #1: determine if linear interpolation is required
#     ny, nx = lons_mg.shape
#     n_pixs = nx * ny
    
#     if (n_pixs > cov_interpolate_threshold) and (method == 'cov'):
#         if verbose:
#             print(f"The number of pixels ({n_pixs}) is larger than 'cov_interpolate_threshold' ({int(cov_interpolate_threshold)}) so images will be created "
#                   f"with {int(cov_interpolate_threshold)} pixels and interpolated to the full resolution.  ")
#         interpolate = True                                                                                  # set boolean flag
#         oversize_factor = n_pixs / cov_interpolate_threshold                                                    # determine how many times too many pixels we have.  
#         lons_ds = np.linspace(lons_mg[-1,0], lons_mg[-1,-1], int(nx * (1/np.sqrt(oversize_factor))))        # make a downsampled vector of just the longitudes (square root as number of pixels is a measure of area, and this is length)
#         lats_ds = np.linspace(lats_mg[0,0], lats_mg[-1,0], int(ny * (1/np.sqrt(oversize_factor))))          # and for latitudes
#         lons_mg_ds = np.repeat(lons_ds[np.newaxis, :], lats_ds.shape, axis = 0)                             # make rank 2 again
#         lats_mg_ds = np.repeat(lats_ds[:, np.newaxis], lons_ds.shape, axis = 1)                             # and for latitudes
#         ny_generate, nx_generate = lons_mg_ds.shape                                                         # get the size of the downsampled grid we'll be generating at
#     else:
#         interpolate = False                                                                                 # set boolean flag
#         nx_generate = nx                                                                                    # if not interpolating, these don't change.  
#         ny_generate = ny
#         lons_mg_ds = lons_mg                                                                                # if not interpolating, don't need to downsample.  
#         lats_mg_ds = lats_mg
    
#     #2: calculate distance between points
#     ph_turbs = np.zeros((n_atms, ny_generate, nx_generate))                                                  # initiate output as a rank 3 (ie n_images x ny x nx)
#     xyz_m, pixel_spacing = lon_lat_to_ijk(lons_mg_ds, lats_mg_ds)                                           # get pixel positions in metres from origin in lower left corner (and also their size in x and y direction)
#     xy = xyz_m[0:2].T                                                                                       # just get the x and y positions (ie discard z), and make lots x 2 (ie two columns)
      
    
#     #3: generate atmospheres, using either of the two methods.  
#     if difference == True:
#         n_atms += 1                                                                                         # if differencing atmospheres, create one extra so that when differencing we are left with the correct number
    
#     if method == 'fft':
#         for i in range(n_atms):
#             ph_turbs[i,:,:] = generate_correlated_noise_fft(nx_generate, ny_generate,    std_long=1, 
#                                                            sp = 0.001 * np.mean((pixel_spacing['x'], pixel_spacing['y'])) )      # generate noise using fft method.  pixel spacing is average in x and y direction (and m converted to km) 
#             if verbose:
#                 print(f'Generated {i+1} of {n_atms} single acquisition atmospheres.  ')
            
#     else:
#         pixel_distances = sp_distance.cdist(xy,xy, 'euclidean')                                                     # calcaulte all pixelwise pairs - slow as (pixels x pixels)       
#         Cd = np.exp((-1 * pixel_distances)/cov_Lc)                                     # from the matrix of distances, convert to covariances using exponential equation
#         success = False
#         while not success:
#             try:
#                 Cd_L = np.linalg.cholesky(Cd)                                             # ie Cd = CD_L @ CD_L.T      Worse error messages, so best called in a try/except form.  
#                 #Cd_L = scipy.linalg.cholesky(Cd, lower=True)                               # better error messages than the numpy versio, but can cause crashes on some machines
#                 success = True
#             except:
#                 success = False
#         for n_atm in range(n_atms):
#             x = np.random.randn((ny_generate*nx_generate))                                               # Parsons 2007 syntax - x for uncorrelated noise
#             y = Cd_L @ x                                                               # y for correlated noise
#             ph_turb = np.reshape(y, (ny_generate, nx_generate))                                             # turn back to rank 2
#             ph_turbs[n_atm,:,:] = ph_turb
#             print(f'Generated {n_atm} of {n_atms} single acquisition atmospheres.  ')
        
        
#         # nx = shape[0]
#         # ny = shape[1]

        

#         # return y_2d
        
#         # success = 0
#         # fail = 0
#         # while success < n_atms:
#         # #for i in range(n_atms):
#         #     try:
#         #         ph_turb = generate_correlated_noise_cov(pixel_distances, cov_Lc, (nx_generate,ny_generate))      # generate noise 
#         #         ph_turbs[success,:,:] = ph_turb
#         #         success += 1
#         #         if verbose:
#         #             print(f'Generated {success} of {n_atms} single acquisition atmospheres (with {fail} failures).  ')
#         #     except:
#         #         fail += 0
#         #         if verbose:
#         #             print(f"'generate_correlated_noise_cov' failed, which is usually due to errors in the cholesky decomposition that Numpy is performing.  The odd failure is normal.  ")
                
#         #     # ph_turbs[i,:,:] = generate_correlated_noise_cov(pixel_distances, cov_Lc, (nx_generate,ny_generate))      # generate noise 
#         #     # if verbose:
                
                

#     #3: possibly interplate to bigger size
#     if interpolate:
#         if verbose:
#             print('Interpolating to the larger size...', end = '')
#         ph_turbs_output = np.zeros((n_atms, ny, nx))                                                                          # initiate output at the upscaled size (ie the same as the original lons_mg shape)
#         for atm_n, atm in enumerate(ph_turbs):                                                                                # loop through the 1st dimension of the rank 3 atmospheres.  
#             f = scipy_interpolate.interp2d(np.arange(0,nx_generate), np.arange(0,ny_generate), atm, kind='linear')           # and interpolate them to a larger size.  First we give it  meshgrids and values for each point
#             ph_turbs_output[atm_n,:,:] = f(np.linspace(0, nx_generate, nx), np.linspace(0, ny_generate, ny))                  # then new meshgrids at the original (full) resolution.  
#         if verbose:
#             print('Done!')
#     else:
#         ph_turbs_output = ph_turbs                                                                                              # if we're not interpolating, no change needed
       
#     # 4: rescale to correct range (i.e. a couple of cm)
#     ph_turbs_m = np.zeros(ph_turbs_output.shape)
#     for atm_n, atm in enumerate(ph_turbs_output):
#         ph_turbs_m[atm_n,] = rescale_atmosphere(atm, mean_m)
                
#     # 5: return back to the shape given, which can be a rectangle:
#     ph_turbs_m = ph_turbs_m[:,:lons_mg.shape[0],:lons_mg.shape[1]]
    
#     if water_mask is not None:
#         water_mask_r3 = ma.repeat(water_mask[np.newaxis,], ph_turbs_m.shape[0], axis = 0)
#         ph_turbs_m = ma.array(ph_turbs_m, mask = water_mask_r3)
    
#     return ph_turbs_m

#%%

def coherence_mask(lons_mg, lats_mg, threshold=0.8, turb_method = 'fft',
                   cov_Lc = 5000, cov_interpolate_threshold = 1e4, verbose = False):
    """A function to synthesis a mask of incoherent pixels
    Inputs:
        lons_mg | rank 2 array | longitudes of the bottom left corner of each pixel.  
        lats_mg | rank 2 array | latitudes of the bottom left corner of each pixel.  
        threshold | decimal | value at which deemed incoherent.  Bigger value = less is incoherent
        turb_method | string | 'fft' or 'cov'.  Controls the method used to genete spatially correlated noise which is used here.  fft is normal ~100x faster.  
        cov_Lc     | float | length scale of correlation, in metres.  If smaller, noise is patchier (ie lots of small masked areas), and if larger, smoother (ie a few large masked areas).  
        cov_interpolation_threshold | int | if there are more pixels than this value (ie the number of entries in lons_mg), interpolation is used to create the extra resolution (as generating spatially correlated noise is slow for large images)
        verbose | boolean | True is information required on terminal.  
        
    Returns:
        mask_coh | rank 2 array | 
        
    2019_03_06 | MEG | Written.  
    2020/08/10 | MEG | Update and add to SyInterferoPy.  
    2020/08/12 | MEG | Remove need for water_mask to be passed to function.  
    2020/10/02 | MEG | Update to work with atmosphere_turb after this switched from cm to m.  
    2020/10/07 | MEG | Update to use new atmosphere_turb function
    2020/10/19 | MEG | Add option to set pass interpolation threshold to atmosphere_turb function.  
    2020/03/01 | MEG | Add option to select which method is used to generate the spatialy correlated noise.  
    """
    import numpy as np
    if verbose:
        print(f"Starting to generate a coherence mask... ", end = '')


    if turb_method == 'fft':
        mask_coh_values_r3 = atmosphere_turb(1, lons_mg, lats_mg, method='fft', mean_m = 0.01)                              # generate a single turbulent atmosphere (though it still comes at as rank 3 with first dimension = 1)
    elif turb_method == 'cov':
        mask_coh_values_r3 = atmosphere_turb(1, lons_mg, lats_mg, mean_m = 0.01,
                                             method='cov', cov_Lc=cov_Lc, cov_interpolate_threshold=cov_interpolate_threshold)       # generate a single turbulent atmosphere (though it still comes at as rank 3 with first dimension = 1)
    else:
        print(f"'turb_method' should be either 'fft' or 'cov'.  {turb_method} was supplied, so defaulting to 'fft'.  ")
        mask_coh_values_r3 = atmosphere_turb(1, lons_mg, lats_mg, method='fft', mean_m = 0.01)                              # generate a single turbulent atmosphere (though it still comes at as rank 3 with first dimension = 1)
        
    mask_coh_values = mask_coh_values_r3[0,]                                                                                # convert to rank 2
    mask_coh_values = (mask_coh_values - np.min(mask_coh_values)) / np.max(mask_coh_values - np.min(mask_coh_values))       # rescale to range [0, 1]
    mask_coh = np.where(mask_coh_values > threshold, np.ones(lons_mg.shape), np.zeros(lons_mg.shape))                       # anything above the threshold is masked, creating blothcy areas of incoherence.  

    if verbose:
        print("Done. ")
    return mask_coh

#%%

