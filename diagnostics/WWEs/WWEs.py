import matplotlib
matplotlib.use("Agg")  # non-X windows backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import glob
import os
import time 
import xesmf as xe
import scipy
from scipy import stats
from functools import partial
import intake
import sys
import yaml

from WWE_diag_tools import (
    land_mask_using_etopo,
    regridder_model2obs,
    nharm,
    calc_raw_and_smth_annual_cycle,
    isolate_WWEs,
    WWE_characteristics,
    WWE_statistics, #We don't need to do the statistics to make the likelihood by longitude plot
    find_WWE_time_lon)


####################################################################################
#Define some paths and functions
####################################################################################
def plot_model_Hovmollers_by_year(data = None, wwe_mask = None, lon_vals = None,
                                  tauu_time = None, savename = '',
                                  first_year = '', last_year = ''):
    
    year_array = np.unique(tauu_time.dt.year)
    nyears     = np.unique(tauu_time.dt.year).size

    fig, ax = plt.subplots(ncols=5, nrows=4, figsize = (15, 16), sharex = True, sharey = True) 
    axlist = ax.flatten()
    shade_choice     = 'bwr'
    levs             = np.linspace(-0.1, 0.1, 21)

    kwargs = {'fontsize':12}
    ####################################################################################
    #Loop through each year to make a Hovmoller panel of filtered zonal wind stress
    #for each year overlapped with WWE blobs
    ####################################################################################
    for iyear in range(20):
        wiyear = np.where((np.asarray(tauu_time.dt.year) == year_array[iyear]))
        
        ########################################################################           
        #Plot details
        ########################################################################=
        cf = axlist[iyear].contourf(np.asarray(lon_vals), np.arange(0, tauu_time[wiyear[0]].size),
                                    np.asarray(data[wiyear[0], :]), levels = levs, 
                                    cmap = shade_choice, extend = 'both')

        cl = axlist[iyear].contour(np.asarray(lon_vals), np.arange(0, tauu_time[wiyear[0]].size),  
                                   wwe_mask[wiyear[0], :], cmap = 'binary', linewidths = 1)

        axlist[iyear].grid(alpha = 0.5)
        
        if iyear >=15 :axlist[iyear].set_xlabel('longitude', **kwargs)
        if iyear%5 == 0: axlist[iyear].set_ylabel('day of year', **kwargs)
        axlist[iyear].set_title(str(year_array[iyear]), fontsize=12, loc = 'left')
        axlist[iyear].tick_params(axis='y', labelsize=12)
        axlist[iyear].tick_params(axis='x', labelsize=12)
        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

    cbar_ax = fig.add_axes([0.81, 0.35, 0.015, 0.3])
    cbar_ax.tick_params(labelsize=12)
    cb = plt.colorbar(cf, cax=cbar_ax)
    cb.set_label(label = '$\u03C4_x$ (N $m^{-2}$)', fontsize = 12)
    
    endof20yrs = str(int(first_year) + 19)
    plt.savefig(savename + first_year + '-' + endof20yrs + '.YearlyHovmollers.png', bbox_inches='tight')
    
    if year_array.size > 20:
        fig, ax = plt.subplots(ncols=5, nrows=4, figsize = (15, 16), sharex = True, sharey = True) 
        axlist = ax.flatten()

        for iyear in range(year_array.size - 20):
            wiyear = np.where((np.asarray(tauu_time.dt.year) == year_array[iyear + 20]))
                
            ####################################################################           
            #Plot details
            ########################################################################
            cf = axlist[iyear].contourf(np.asarray(lon_vals), np.arange(0, tauu_time[wiyear[0]].size),
                                    np.asarray(data[wiyear[0], :]), levels = levs, 
                                    cmap = shade_choice, extend = 'both')
            
            cl = axlist[iyear].contour(np.asarray(lon_vals), np.arange(0, tauu_time[wiyear[0]].size),  
                                       wwe_mask[wiyear[0], :], cmap = 'binary', linewidths = 1)

            axlist[iyear].grid(alpha = 0.5)
            
            if iyear >=15 :axlist[iyear].set_xlabel('longitude', **kwargs)
            if iyear%5 == 0: axlist[iyear].set_ylabel('day of year', **kwargs)
            axlist[iyear].set_title(str(year_array[iyear + 20]), fontsize=12, loc = 'left')
            axlist[iyear].tick_params(axis='y', labelsize=12)
            axlist[iyear].tick_params(axis='x', labelsize=12)
            plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

        cbar_ax = fig.add_axes([0.81, 0.35, 0.015, 0.3])
        cbar_ax.tick_params(labelsize=12)
        cb = plt.colorbar(cf, cax=cbar_ax)
        cb.set_label(label = '$\u03C4_x$ (N $m^{-2}$)', fontsize = 12)
        
        start2ndpage = str(int(first_year) + 20)
        plt.savefig(savename + start2ndpage + '-' + last_year + '.YearlyHovmollers.png', bbox_inches='tight')
    
    return cf

def _preprocess(x, lon_bnds, lat_bnds):
    return x.sel(lon=slice(*lon_bnds), lat=slice(*lat_bnds))

print("\n=======================================")
print("BEGIN WWEs.py ")
print("=======================================")

work_dir     = os.environ["WORK_DIR"]
obs_dir      = os.environ["OBS_DATA"]
casename     = os.environ["CASENAME"]
first_year   = os.environ["first_yr"]
last_year    = os.environ["last_yr"]
min_lat      = float(os.environ["min_lat"])
max_lat      = float(os.environ["max_lat"])
min_lon      = float(os.environ["min_lon"])
max_lon      = float(os.environ["max_lon"])
regrid_method= os.environ["regrid_method"]

#Define lats to average tauu over and lon range to analyze
lat_lim_list = [min_lat, max_lat]
lon_lim_list = [min_lon, max_lon]

###########################################################################
##############Part 1: Get, Plot Observations ##############################
###########################################################################
print(f'*** Now working on obs data\n------------------------------')
obs_file_WWEs = obs_dir + '/TropFlux_120-dayHPfiltered_tauu_1980-2014.nc'

print(f'*** Reading obs data from {obs_file_WWEs}')
obs_WWEs    = xr.open_dataset(obs_file_WWEs)
print(obs_WWEs, 'obs_WWEs')

# Subset the data for the user defined first and last years #
obs_WWEs = obs_WWEs.sel(time=slice(first_year, last_year))

obs_lons = obs_WWEs.lon
obs_lats = obs_WWEs.lat
obs_time = obs_WWEs.time
Pac_lons = obs_WWEs.Pac_lon
obs_WWE_mask        = obs_WWEs.WWE_mask
TropFlux_filt_tauu  = obs_WWEs.filtered_tauu
TropFlux_WWEsperlon = obs_WWEs.WWEs_per_lon

plot_model_Hovmollers_by_year(data = TropFlux_filt_tauu, wwe_mask = obs_WWE_mask,
                                  lon_vals = Pac_lons, tauu_time = obs_time,
                                  savename = f"{work_dir}/obs/PS/TropFlux_",
                                  first_year = first_year, last_year = last_year)

###########################################################################
###########Parse MDTF-set environment variables############################
###########################################################################
#These variables come from the case_env_file that the framework creates
#the case_env_file points to the csv file, which in turn points to the data files.
#Variables from the data files are then read in. See example_multicase.py
print("*** Parse MDTF-set environment variables ...")

case_env_file = os.environ["case_env_file"]
assert os.path.isfile(case_env_file), f"case environment file not found"
with open(case_env_file, 'r') as stream:
    try:
        case_info = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

cat_def_file = case_info['CATALOG_FILE']
case_list    = case_info['CASE_LIST']

#Use partial function to only load part of the data file
lon_bnds, lat_bnds = (0, 360), (-32.5, 32.5)
partial_func       = partial(_preprocess, lon_bnds=lon_bnds, lat_bnds=lat_bnds)

# open the csv file using information provided by the catalog definition file
cat = intake.open_esm_datastore(cat_def_file)

# all cases share variable names and dimension coords in this example, so just get first result for each
tauu_var   = [case['tauu_var'] for case in case_list.values()][0]
time_coord = [case['time_coord'] for case in case_list.values()][0]
lat_coord  = [case['lat_coord'] for case in case_list.values()][0]
lon_coord  = [case['lon_coord'] for case in case_list.values()][0]

###########################################################
#Filter catalog by desired variable and output frequency
###########################################################
#Get tauu (zonal wind stress) variable
tauu_subset = cat.search(variable_id=tauu_var, frequency="day")

# convert tauu_subset catalog to an xarray dataset dict
tauu_dict = tauu_subset.to_dataset_dict(preprocess = partial_func,
                                        xarray_open_kwargs={"decode_times": True, "use_cftime": True})

for k, v in tauu_dict.items(): 
    tauu_arr = tauu_dict[k][tauu_var]

#Get sftlf (land fraction) variable if it exists & mask out land 
key = 'sftlf_var'
x = list(case_list[casename].keys())

if(x.count(key) == 1):
    print("Using model land fraction variable")
    sftlf_var    = [case['sftlf_var'] for case in case_list.values()][0]
    sftlf_subset = cat.search(variable_id=sftlf_var, frequency="fx")
    # convert sftlf_subset catalog to an xarray dataset dict
    sftlf_dict   = sftlf_subset.to_dataset_dict(preprocess = partial_func)

    for k, v in sftlf_dict.items():
        sftlf_arr = sftlf_dict[k][sftlf_var]

    #mask out land in tauu
    masked_tauu = tauu_arr.where(sftlf_arr < 10)

if(x.count(key) == 0):
    print("Need to use etopo.nc file to mask out the land")
    print('Program will exit for now, as need to build in more code')
    #ls_mask = land_mask_using_etopo(ds = model_ds, topo_latgrid_1D = topo_latgrid_1D, 
    #                                    topo_longrid_1D = topo_longrid_1D,
    #                                    topo_data1D = topo_data1D, lf_cutoff = 10)
    #masked_tauu = model_ds[tauu_name].where(ls_mask == 1)
    sys.exit()

if(x.count(key) > 1):
    print('Error: Multiple land fraction (sftlf) files found. There should only be one!')
    print('Program will exit')
    sys.exit()

#Convert masked_tauu dataaray back to dataset    
tauu_ds = masked_tauu.to_dataset()

#Create a mask variable for the tauu and ws dataset
tauu_ds["mask"] = xr.where(~np.isnan(tauu_ds[tauu_var].isel(time = 0)), 1, 0)

print('tauu_ds.lat.size:', tauu_ds.lat.size)
print('regrid method:', regrid_method)
##################################################
#Regrid tauu to the observations
##################################################
if tauu_ds.lat.size > 1:
    print('tauu_ds.lat.size > 1')
    regridder_tauu = regridder_model2obs(lon_vals = obs_lons, lat_vals = obs_lats,
                                        in_data = tauu_ds, type_name = regrid_method,
                                        isperiodic = True)
    re_model_var   = regridder_tauu(tauu_ds[tauu_var], skipna = True)

re_model_var
    
tauu_arrays = {}
for k, v in tauu_dict.items(): 
    arr = tauu_dict[k][tauu_var]
    arr = arr.sel(lon = slice(120,280), lat = slice(-2.5, 2.5),
                      time = slice(first_year, last_year))
    arr = arr.mean(dim = (tauu_dict[k][lat_coord].name,tauu_dict[k][time_coord].name))

    tauu_arrays[k] = arr

###########################################################################
# Part 3: Make a plot that contains results from each case
# --------------------------------------------------------

# set up the figure
fig = plt.figure(figsize=(12, 4))
ax = plt.subplot(1, 1, 1)

# loop over cases
for k, v in tauu_arrays.items():
    v.plot(ax=ax, label=k)

# add legend
plt.legend()

# add title
plt.title("Mean Zonal Wind Stress")

assert os.path.isdir(f"{work_dir}/model/PS"), f'Assertion error: {work_dir}/model/PS not found'
plt.savefig(f"{work_dir}/model/PS/{casename}.Mean_TAUU.eps", bbox_inches="tight")

# Part 4: Close the catalog files and
# release variable dict reference for garbage collection
# ------------------------------------------------------
cat.close()
tauu_dict = None
# Part 5: Confirm POD executed successfully
# ----------------------------------------
print("Last log message by example_multicase POD: finished successfully!")
sys.exit(0)

