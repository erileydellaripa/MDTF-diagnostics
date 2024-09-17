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
    filter_data,
    isolate_WWEs,
    WWE_characteristics,
    WWE_statistics, #We don't need to do the statistics to make the likelihood by longitude plot
    find_WWE_time_lon)

####################################################################################
#Define some paths and functions
####################################################################################
def find_WWEs_and_characteristics(in_data = None, tauu_thresh = 0.04, mintime = 5, minlons = 10,
                                 xminmax = (3, 3), yminmax = (3, 3), minmax_dur_bins = (5, 27),
                                 dur_bin_space = 2, minmax_IWW_bins = (1, 42), IWW_bin_space = 4,
                                 xtend_past_lon = 140):
    '''
    This function call the following functions within WWE_diag_tools.py
    - isolate_WWEs
        - find_nearby_wwes_merge
        - renumber_wwes
    - WWE_chracteristics
    - find_WWE_time_lon
    '''
    
    start_time = time.time()
    
    # 1) Find WWEs
    #The isolate_WWEs function uses the find_nearby_wwes_merge, renumber_wwes functions 
    WWE_labels, WWE_mask = isolate_WWEs(data = in_data, tauu_thresh = tauu_thresh, mintime = mintime, 
                                        minlons = minlons, xmin = xminmax[0], xmax = xminmax[1], 
                                        ymin = yminmax[0], ymax = yminmax[1], xtend_past_lon = xtend_past_lon)

    # 2) Find characteristics (i.e., duration, zonal extent, integrated wind work sum and mean) of each WWE
    #Uses WWE_characteristics function
    duration, zonal_extent, IWW, tauu_mean = WWE_characteristics(wwe_labels = WWE_labels, data = in_data)
    
    # 3) Find central, min, and max time and longitude of each WWE
    #Uses find_WWE_time_lon function
    tauu_time   = in_data["time"]
    tauu_lon    = in_data["lon"]
    lon_array   = np.asarray(tauu_lon)
       
    center_lons, center_times, min_times, max_times, min_lons, max_lons \
    = find_WWE_time_lon(data = in_data, wwe_labels = WWE_labels, 
                         lon = lon_array, time_array = tauu_time)

    print("--- %s seconds to ID WWEs and compute characteristics---" % (time.time() - start_time))
    
    return duration, IWW, zonal_extent, tauu_mean, WWE_labels, WWE_mask, center_lons, \
           center_times, min_times, max_times, min_lons, max_lons, \
           lon_array, tauu_time
           
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

print("*** Parse MDTF-set environment variables ...")
work_dir     = os.environ["WORK_DIR"]
obs_dir      = os.environ["OBS_DATA"]
casename     = os.environ["CASENAME"]
first_year   = os.environ["first_yr"]
last_year    = os.environ["last_yr"]
static_thresh= os.environ['do_static_threshold']
min_lat      = float(os.environ["min_lat"])
max_lat      = float(os.environ["max_lat"])
min_lon      = float(os.environ["min_lon"])
max_lon      = float(os.environ["max_lon"])
regrid_method= os.environ["regrid_method"]

#Define lats to average tauu over and lon range to analyze
lat_lim_list = [min_lat, max_lat]
lon_lim_list = [min_lon, max_lon]

###########################################################################
##################### Get & Plot Observations #############################
###########################################################################
print(f'*** Now working on obs data\n------------------------------')
obs_file_WWEs = obs_dir + '/TropFlux_120-dayHPfiltered_tauu_1980-2014.nc'

print(f'*** Reading obs data from {obs_file_WWEs}')
obs_WWEs    = xr.open_dataset(obs_file_WWEs)
print(obs_WWEs)

# Subset the data for the user defined first and last years #
obs_WWEs = obs_WWEs.sel(time=slice(first_year, last_year))

obs_lons = obs_WWEs.lon
obs_lats = obs_WWEs.lat
obs_time = obs_WWEs.time
Pac_lons = obs_WWEs.Pac_lon
obs_WWE_mask        = obs_WWEs.WWE_mask
TropFlux_filt_tauu  = obs_WWEs.filtered_tauu
TropFlux_WWEsperlon = obs_WWEs.WWEs_per_lon

#Plot the yearly Hovmollers for observations
plot_model_Hovmollers_by_year(data = TropFlux_filt_tauu, wwe_mask = obs_WWE_mask,
                                  lon_vals = Pac_lons, tauu_time = obs_time,
                                  savename = f"{work_dir}/obs/PS/TropFlux_",
                                  first_year = first_year, last_year = last_year)

###################################################################################
######### PART 2 ##################################################################
######### Prepare Model output for WWE ID code#####################################
###################################################################################
print(f'*** Now starting work on {casename}\n------------------------------')
print('*** Reading variables ...')

#These variables come from the case_env_file that the framework creates
#the case_env_file points to the csv file, which in turn points to the data files.
#Variables from the data files are then read in. See example_multicase.py
# Read the input model data

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

############################################################################
#Filter catalog by desired variable and output frequency
############################################################################
#Get tauu (zonal wind stress) variable
tauu_subset = cat.search(variable_id=tauu_var, frequency="day")

# convert tauu_subset catalog to an xarray dataset dict
tauu_dict = tauu_subset.to_dataset_dict(preprocess = partial_func,
                                        xarray_open_kwargs={"decode_times": True, "use_cftime": True})

for k, v in tauu_dict.items(): 
    tauu_arr = tauu_dict[k][tauu_var]

##################################################################
#Get sftlf (land fraction) variable if it exists & mask out land
##################################################################
key = 'sftlf_var'
x = list(case_list[casename].keys())

if(x.count(key) == 1):
    print("Using model land fraction variable to mask out land")
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
    
##################################################
#Convert masked_tauu dataaray back to dataset    
tauu_ds = masked_tauu.to_dataset()

##################################################
#Only keep data during desired time range
tauu_ds = tauu_ds.where((tauu_ds.time.dt.year >= int(first_year)) &
                       (tauu_ds.time.dt.year <= int(last_year)), drop = True)

##################################################
#Create a mask variable for the tauu dataset
tauu_ds["mask"] = xr.where(~np.isnan(tauu_ds[tauu_var].isel(time = 0)), 1, 0)

##################################################
#Regrid tauu to the TropFlux obs grid
##################################################
print('lon size before regridding:', tauu_ds.lon.size)
print('Start regrid code using the following method:', regrid_method)

if tauu_ds.lat.size > 1:
    print('tauu_ds.lat.size > 1')
    regridder_tauu = regridder_model2obs(lon_vals = np.asarray(obs_lons), lat_vals = np.asarray(obs_lats),
                                        in_data = tauu_ds, type_name = regrid_method,
                                        isperiodic = True)
    re_model_tauu  = regridder_tauu(tauu_ds[tauu_var], skipna = True)

print('lon size after regridding:', re_model_tauu.lon.size)
    
##################################################
#Find region of interest
#At this point, re_model_tauu is a DataArray
##################################################
tauu_region = ((re_model_tauu).where(
    (re_model_tauu.lat >= np.array(lat_lim_list).min()) &
    (re_model_tauu.lat <= np.array(lat_lim_list).max()) &
    (re_model_tauu.lon >= np.array(lon_lim_list).min()) &
    (re_model_tauu.lon <= np.array(lon_lim_list).max()),
    drop = True))

print('tauu_region:', tauu_region)

##################################################
#Average over the latitudes
##################################################
#The xarray mean function ignores the nans
tauu_region_latavg = tauu_region.mean(dim = 'lat') 
    
###################################################################################
#Check to see if westerly zonal wind stresses are recorded as positive or negative
###################################################################################
mean_lon220p5 = np.array(np.mean(tauu_region_latavg.sel(lon = 220.5)))
print('mean tauu at 220.5E:', mean_lon220p5)
factor = -1 if mean_lon220p5 > 0 else 1
tauu   = tauu_region_latavg * factor
print('tauu after lat averaging:', tauu)
print('At this point, tauu is a DataArray with time longitude dimensions on the TropFlux grid')

###################################################################################
#Filter tauu to use as input to find WWEs and their chracteristics
###################################################################################
#filt_dataLP = filter_data(data = tauu, nweights = 201, a = 5)
#For now the only option is to apply a 120-day highpass filter
filt_dataHP = filter_data(data = tauu, nweights = 201, a = 120) 

data2use        = tauu - filt_dataHP
obs_tauu_thresh = 0.04 #Nm-2 Two standard deviations of the TropFlux lat-averaged 120E-280E zonal wind stress.
tauu_thresh2use = obs_tauu_thresh if static_thresh is True else np.round(data2use.std()*2, decimals = 2)

print('tauu_thresh2use:', tauu_thresh2use)
print('data2use', data2use)

###################################################################################
######### PART 3 ##################################################################
#########Find WWEs and their characteristics and compute statistics################
###################################################################################
duration, IWW, zonal_extent, tauu_mean, WWE_labels, WWE_mask, center_lons, \
center_times, min_times, max_times, min_lons, max_lons, lon_array, tauu_time = \
find_WWEs_and_characteristics(in_data = data2use, tauu_thresh = tauu_thresh2use, mintime = 5, minlons = 10,
                              xminmax = (3, 3), yminmax = (3, 3), minmax_dur_bins = (5, 27),
                              dur_bin_space = 2, minmax_IWW_bins = (1, 42), IWW_bin_space = 4,
                              xtend_past_lon = 140)

durationB, zonal_extentB, tauu_sum, tauu_abs_mean = WWE_characteristics(wwe_labels = WWE_labels, data = tauu)

print('nWWEs:', duration.size)
#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------
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

