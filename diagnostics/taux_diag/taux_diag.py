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

taux_path = os.environ["TAUX_FILE"]
workdir_name = os.environ["WK_DIR"]
print('taux_path:', taux_path)
print('wk_dir:', workdir_name)

#Set the lat and lon limits.
lat_lim = [float(os.getenv("lat_min")), float(os.getenv("lat_max"))]
lon_lim = [float(os.getenv("lon_min")), float(os.getenv("lon_max"))]

#Open the data set
model_ds = xr.open_dataset(taux_path)

#Find region of interest
ds_region = ((model_ds).where(
    (model_ds.lat >= np.array(lat_lim).min()) &
    (model_ds.lat <= np.array(lat_lim).max()) &
    (model_ds.lon >= np.array(lon_lim).min()) &
    (model_ds.lon <= np.array(lon_lim).max()),
    drop = True))

taux_var_name   = os.environ["TAUX_var"]
time_coord_name = os.environ["time_coord"]
lat_coord_name  = os.environ["lat_coord"]

#Take mean over time and latitude
ds_daily_mean = ds_region.groupby('time.dayofyear').mean(dim = (time_coord_name, lat_coord_name))

#Get the data and reverse the sign of the stress, so that a westward stress is positive
factor    = -1
taux_data = ds_daily_mean[taux_var_name]*factor

print("Computed latitudinally average annual cycle of {TAUX_var} for {CASENAME}.".format(**os.environ))

#Attach the original variable attributes to the time and lat averaged attributes
taux_data.attrs = model_ds.TAUX.attrs

#Save the taux_data to a netcdf
out_path = workdir_name + "/model/netCDF/annual_cycle_taux_EQ_130to260.nc".format(**os.environ)
taux_data.to_netcdf(out_path)

###############################################################
##Plot Model hovmoller
###############################################################
#Plot details
levs    = np.linspace(-0.1, 0.1, 21)
fig, ax = plt.subplots(figsize=(5, 5))

##Make the plot
cf = taux_data.plot(robust = True, x = 'lon', y = 'dayofyear', 
                             levels = levs, cmap = 'bwr')

#Regine plot labels
plt.ylabel("day of year")
plt.xlabel("longitude")
title_string = "{CASENAME}: Mean {TAUX_var} Annual Cycle".format(**os.environ)
plt.title(title_string)

#Output the figure
fname        = "{CASENAME}.AnnualCycle_TAUX.eps".format(**os.environ) #"LatAvg_Taux_Time_vs_Lon.eps"
output_fname = os.path.join(workdir_name, "model", "PS", fname)
plt.savefig(output_fname, format="eps", bbox_inches="tight")


###############################################################
##Plot Observations hovmoller
###############################################################
input_path = "{OBS_DATA}/avg_annual_cycle_1980-2018_TropFlux_tauxPa_EQ_130to260.nc".format(**os.environ)

# command to load the netcdf file
obs_dataset = xr.open_dataset(input_path)
obs_taux    = obs_dataset['taux']

########################
##Make the plot
obs_fig, obs_ax = plt.subplots(figsize=(5, 5))

obs_cf = obs_taux.plot(robust = True, x = 'lon', y = 'dayofyear', levels = levs, cmap = 'bwr')

#Remove the colorbar, so that details can be added later
#cb = cf.colorbar
#cb.remove()

#Regine plot labels
plt.ylabel("day of year")
plt.xlabel("longitude")
title_string = "TropFlux: Mean Taux Annual Cycle".format(**os.environ)
plt.title(title_string)

#Output the figure
fname        = "TropFlux.AnnualCycle_TAUX_avgeraged_1980-2018.eps".format(**os.environ) 
output_fname = os.path.join(workdir_name, "obs", "PS", fname)
plt.savefig(output_fname, format="eps", bbox_inches="tight")

#Close the files
model_ds.close()
obs_ds.close()
