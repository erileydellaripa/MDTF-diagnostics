#!/usr/bin/env python
# coding: utf-8

#To test the POD in the model framework, need to use non-interactive backend with matplotlib,
#so comment out the below two lines
#import matplotlib
#matplotlib.use("Agg")
import matplotlib

matplotlib.use("Agg")

import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt

dirname  = '/Users/eriley/NOAA_POD/data/'
filename = 'E3SM.LY04.TAUX.Equator.0362-0400.nc'

#Set the lat and lon limits.
#In the POD framework, these will be read in from the settings file

#lat_lim = [np.float(os.getenv("lat_min")), np.float(os.getenv("lat_max"))]
#lon_lim = [np.float(os.getenv("lon_min")), np.float(os.getenv("lon_max"))]
lat_lim = [-15., 15.]
lon_lim = [130., 270]

#Open the data set
ds = xr.open_dataset(dirname + filename)

#Find region of interest
ds_region = ((ds).where(
    (ds.lat >= np.array(lat_lim).min()) &
    (ds.lat <= np.array(lat_lim).max()) &
    (ds.lon >= np.array(lon_lim).min()) &
    (ds.lon <= np.array(lon_lim).max()),
    drop = True))

#Take mean over time and latitude
ds_daily_mean = ds_region.groupby('time.dayofyear').mean(dim = ('time', 'lat'))

#Get the data and reverse the sign of the stress, so that a westward stress is positive
factor = -1
taux_data = ds_daily_mean["TAUX"]*factor

#Attach the original variable attributes to the time and lat averaged attributes
taux_data.attrs = ds.TAUX.attrs

#Save the taux_data to a netcdf
taux_data.to_netcdf('annual_cycle_taux_EQ_130to260.nc')

##Plot hovmoller
#Plot details
levs    = np.linspace(-0.1, 0.1, 21)
fig, ax = plt.subplots(figsize=(5, 5))

#Make the plot
cf = taux_data.plot(robust = True, x = 'lon', y = 'dayofyear', 
                             levels = levs, cmap = 'bwr')

#Remove the colorbar, so that details can be added later
#cb = cf.colorbar
#cb.remove()

#Regine plot labels
plt.ylabel("day of year")
plt.xlabel("longitude")
plt.title("Annual Cycle Latitude-Averaged Taux")

#Output the figure
fig.savefig("LatAvg_Taux_vs_Time.eps", bbox_inches="tight")
