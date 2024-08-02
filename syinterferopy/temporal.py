#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:34:39 2023

@author: matthew
"""



#%%


def seasonal_topographic_ctc(day_start, day_end, amplitude = 0.02, 
                            noise_level = 0.01, peak_season='summer',
                            period=365):
    """
    Generate a sinusoid with noise for use as a cumulative time course for
    a topographically correlated phase screen.  
    
    Note that this is cumulatve, and not the 

    Inputs
        day_start (str): Start date in 'yyyymmdd' format.
        day_end (str): End date in 'yyyymmdd' format.
        amplitude (float): Amplitude of the sinusoid, in metres (e.)
        noise_level (float): Standard deviation of the Gaussian noise.
        peak_season (str): 'summer' for peak in summer, 'winter' for peak in winter.
        period (int): Period of the sinusoid (default is 365 days for annual period).

    Returns:
        np.ndarray: Array of generated values.
        
    History:
        2024 | MEG | Written.  
    """
    
    import numpy as np
    from datetime import datetime
    
    # Convert date strings to datetime objects
    start_date = datetime.strptime(day_start, '%Y%m%d')
    end_date = datetime.strptime(day_end, '%Y%m%d')

    # list of all dates 
    dates = datetimes_between_dates(day_start, day_end)

    # Calculate the number of days
    n_days = (end_date - start_date).days + 1

    # Generate an array of days
    days = np.arange(n_days)

    # Calculate the day of the year for the start date (ie if Jan 28, 28 days)
    start_day_of_year = start_date.timetuple().tm_yday

    # Determine the peak day of the year for the specified season
    if peak_season == 'summer':
        peak_day_of_year = 172  # June 21st, approximate middle of summer
    elif peak_season == 'winter':
        peak_day_of_year = 355  # December 21st, approximate middle of winter
    else:
        raise ValueError("peak_season must be 'summer' or 'winter'")

    # Calculate phase shift to align the start date with the desired peak day
    # note that cos has peak alligned on day 0, so translate to desired
    # peak day, and also translate by when in year we start.  
    phase_shift = (peak_day_of_year + start_day_of_year) % period
    
    # Generate the sinusoidal signal with the phase shift
    sinusoid = amplitude * np.cos(2 * np.pi *( (days + phase_shift) / period))

    # Generate the noise
    noise = np.random.normal(0, noise_level, n_days)

    # Combine the sinusoid and noise
    noisy_sinusoid = sinusoid + noise
    
    # set so that zero on first acquisition day
    noisy_sinusoid -= noisy_sinusoid[0]

    return days, dates, noisy_sinusoid


#%%

def generate_random_tcs(n_tcs = 100, d_start = "20141231", d_stop = "20230801",
                        min_def_rate = 1., max_def_rate = 2. ):
    """ Generate n_tcs random time courses (ie temporal behavious of deformation.  )
    
    Inputs:
        n_tcs | int | Number of time series to generate.  
        d_start | string | start date, form YYYYMMDD
        d_end | string | start date, form YYYYMMDD
        min_def_rate | float | m/yr minimum deformation rate.  
        max_def_rate | float | m/yr maximum deformation rate.  
    Returns:
        tcs | rank 2 array | time courses as row vectors.  n_tcs rows.  
        def_dates | list of datetimes | all the dates that we have deformation for.  
    History:
        2023_08_24 | MEG | Written.  
    """
    import numpy as np
    from syinterferopy.temporal import tc_uniform_inflation
    
    for ts_n in range(n_tcs):
        def_rate = np.random.uniform(min_def_rate, max_def_rate)                                       # random deformation rate.  
        tc_def, def_dates = tc_uniform_inflation(def_rate, d_start, d_stop)         # generate time course
        if ts_n == 0:                                                               # if the first time.
            tcs = np.zeros((n_tcs, tc_def.shape[0]))                                # initialise array to store results
        tcs[ts_n, :] = tc_def                                                       # record one result as row vector to array.  
    return tcs, def_dates


#%%

def tc_uniform_inflation(def_rate = 0.07, d_start = "20141231", d_stop = "20230801"):
    """ Calculate the magnitdue of an inflating signal at each day.  Inflation is linear.  
    Inputs:
        def_rate | float | deformation rate, my/yr
        d_start | str | YYYYMMDD of when to start time series.  Inclusive.  
        d_stop | str | YYYYMMDD of when to stop time series.  Not inclusive.  
    Returns:
        tc_def | list of floats | cumulative deformation on each day.  Note that last day is not included.  
        def_dates | list of dates | dateteime for each day there is deformation for.  Note that last day is not includes
    History:
        2023_08_17 | MEG | Written
    """
    from datetime import datetime, timedelta
    import numpy as np


    # make cumulative deformaitn for each day
    dstart = datetime.strptime(d_start, '%Y%m%d')                              # conver to dateteime
    dstop = datetime.strptime(d_stop, '%Y%m%d')                                # convert to datetime
    n_days = (dstop - dstart).days                                             # find number of days between dates
    max_def = def_rate * (n_days / 365)                                         # calculate the maximum deformation
    tc_def = np.linspace(0, max_def, n_days)                                    # divide it up to calucaulte it at each day
    
    # make datetime for each day.  
    def_dates = datetimes_between_dates(d_start, d_stop)

    return tc_def, def_dates




#%%

def generate_temporal_baselines(d_start = "20141231", d_stop = "20230801",
                                usual_tbaselines = None):
    """ Given a date range, generate LiCSAR style short temporal baseline ifgs.
    Takes into account that S1b was operational and there were more 6 day ifg then.  
    
    If the usual_tbaselines that are provided are all the same, uniform baselines
    are created.  
    
    Inputs:
        d_start | str | YYYYMMDD of when to start time series
        d_stop | str | YYYYMMDD of when to stop time series
    Returns:
        acq_dates | list of datetimes | acq dates.  
        tbaselines | list of ints | timeporal baselines of short temporal 
                                    baseline ifgs.  First one is 0.  
    History:
        2023_08_17 | MEG | Written
        2024_07_26 | MEG | Add option for uniform baselines.  
    """
    # temp baseline is chosen from this list at random 
    if usual_tbaselines is None:
        print("No 'tbaselines' (usual temporal baselines in days) were provided, "
              "so using the default values.  ")
        usual_tbaselines = [6, 12, 12, 12, 12, 12, 12, 12, 12, 12,                  
                             24, 24, 24, 24, 24, 24, 36, 48, 60, 72]                

    from datetime import datetime, timedelta
    import numpy as np

    # hard coded.  Not exact (i.e. some frames start and stop at different times.)
    s1b_dstart = datetime.strptime("20160901", '%Y%m%d')                       # launch 20160425, ramp up to fully operational over next few months.  
    s1b_dstop = datetime.strptime("20161223", '%Y%m%d')                        # power failure ended mission

    dstart = datetime.strptime(d_start, '%Y%m%d')                              # 
    dstop = datetime.strptime(d_stop, '%Y%m%d')                                # 

    
    acq_dates = [dstart]
    tbaselines = [0]
    dcurrent = acq_dates[-1]

    while dcurrent < dstop:
        # check if the next date is during the S1b years.  
        if (s1b_dstart < dcurrent) and (dcurrent < s1b_dstop ):                  
            #  if so no need to remove anything
            filtered_tbaselines = usual_tbaselines
        else:
            # remove any 6 day ifgs
            filtered_tbaselines  = [x for x in usual_tbaselines if x != 6]
        
        tbaseline = int(np.random.choice(filtered_tbaselines))       
            
        dnext = dcurrent + timedelta(days = tbaseline)                           # add the temp baseline to find the new date   
        if dnext < dstop:                                                        # check we haven't gone past the end date 
            acq_dates.append(dnext)                                              # if we haven't, record            
            tbaselines.append(tbaseline)
            dcurrent = acq_dates[-1]                                             # record the current date
        else:
            break                                                                # remember to exit the while if we have got to the last date

    return acq_dates, tbaselines


#%%



def sample_tc_on_acq_dates(acq_dates, tc, def_dates):
    """ Given a time course (or cumulative time course) on each day 
    (tc_def and def_dates), find the values for the (cumulative)time course
    on the acquisition days (acq_dates).  
    
    Inputs:
        acq_dates | list of datetimes | acq dates.  
        tc_def | list of floats | cumulative deformation on each day.  Note that last day is not included.  
        def_dates | list of dates | dateteime for each day there is deformation for.  Note that last day is not includes
    Returns:
        tc_def_resampled | r2 numpy array | cumulative deformation on each acquisition day.  
    History:
        2023_08_23 | MEG | Written
        2024_08_01 | MEG | Update naming to work with any time course.  
    """
    
    import numpy as np
    
    n_acq = len(acq_dates)                                                  
    # initialise as empty (zeros)
    tc_resampled = np.zeros((n_acq, 1))                                 
    
    for acq_n, acq_date in enumerate(acq_dates):                        
        # find which day number the acquiisiont day is
        day_arg = def_dates.index(acq_date)                                 

        # get the deformaiton for that day
        tc_resampled[acq_n, 0] = tc[day_arg]
    
    return tc_resampled


#%%

        

def defo_to_ts(defo, tc):
    """ Multiply a rank 2 deformation (ie images) by a time course to make a 
    rank 3 time series.  Time course is cumulative and for each acquisition, so first one is 0.  
    Returns short temporal baseline ifgs, so first one is not zero.  
    
    Inputs:
        defo | r2 ma | deformatin patter, some pixels masked.  
        tc | r1 | cumulative time course.  
    Returns:
        defo_ts | r3 ma | cumulative time course.  
    History:
        2023_08_24 | MEG | Written.  
    """
    import numpy as np
    import numpy.ma as ma
    
    defo_ts = ma.zeros((tc.shape[0], defo.shape[0], defo.shape[1]))
    
    for acq_n, tc_value in enumerate(tc):
        defo_ts[acq_n, ] = defo * tc_value
        
    defo_ts_ifgs = ma.diff(defo_ts, axis = 0)                                   # convert from cumulative to daisy chain of ifgs (i.e. short temporal baseline)
    return defo_ts_ifgs


#%%

def datetimes_between_dates(d_start, d_stop):
    """ Given two dates, find all the dates between them.  
    
    Inputs:
        d_start | str | yyyymmdd
        d_stop | str | as above. 
        
    Returns:
        dates | list of datetimes.  
        
    History:
        2024_08_01 | MEG | Written
    
    """
    from datetime import datetime, timedelta
    
    dstart = datetime.strptime(d_start, '%Y%m%d')                              
    dstop = datetime.strptime(d_stop, '%Y%m%d')                                
    
    dates = [dstart]
    dcurrent = dates[-1]

    #  iterate over days
    while dcurrent < dstop:
        dnext = dcurrent + timedelta(days = 1)                                  
        dates.append(dnext)
        dcurrent = dates[-1]                                                
        
    # # drop the last one so limits are exculsive
    # dates = dates[:-1]                                                  
    
    return dates