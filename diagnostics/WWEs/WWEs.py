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

work_dir = os.environ["WORK_DIR"]
casename = os.environ["CASENAME"]

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

print("\n=======================================")
print("BEGIN WWEs.py ")
print("=======================================")

def _preprocess(x, lon_bnds, lat_bnds):
    return x.sel(lon=slice(*lon_bnds), lat=slice(*lat_bnds))

work_dir  = os.environ["WORK_DIR"]
obs_dir   = os.environ["OBS_DATA"]
casename  = os.environ["CASENAME"]
first_year= os.environ["first_yr"]
last_year = os.environ["last_yr"]

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
# all cases share variable names and dimension coords in this example, so just get first result for each
tauu_var   = [case['tauu_var'] for case in case_list.values()][0]
time_coord = [case['time_coord'] for case in case_list.values()][0]
lat_coord  = [case['lat_coord'] for case in case_list.values()][0]
lon_coord  = [case['lon_coord'] for case in case_list.values()][0]

# open the csv file using information provided by the catalog definition file
cat = intake.open_esm_datastore(cat_def_file)

# filter catalog by desired variable and output frequency
tauu_subset = cat.search(variable_id=tauu_var, frequency="day")

# examine assets for a specific file
#tas_subset['CMIP.synthetic.day.r1i1p1f1.day.gr.atmos.r1i1p1f1.1980-01-01-1984-12-31'].df

#Use partial function to only load part of the data file
lon_bnds, lat_bnds = (0, 360), (-32.5, 32.5)
partial_func       = partial(_preprocess, lon_bnds=lon_bnds, lat_bnds=lat_bnds)

# convert tas_subset catalog to an xarray dataset dict
tauu_dict = tauu_subset.to_dataset_dict(preprocess = partial_func,
    xarray_open_kwargs={"decode_times": True, "use_cftime": True}
)

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

