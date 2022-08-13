#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 11:49:08 2022

@author: fraukaiser
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import geopandas as gpd
import datetime
#from shapely.geometry import MultiPoint
import shapely




#%%
path_in = '/home/fraukaiser/Desktop/Clouds/2022-02-25_QualityComparison/1_results/'
today = datetime.date.today()

#%% Load Datasets after processing and calculation of s_min
    
change = path_in + '20201027_2018_DH4_RGB_camfile_II_feature.kNN4_arosicsshifted.registeredBySegmentation.M3C2-1.23273_2022-05-05_Dxyz_2D3D_masked_zoom.csv'

change_df = pd.read_csv(change).replace([np.inf, -np.inf], np.nan)
#%% 
#change_gdf = gpd.GeoDataFrame(change_df, geometry=gpd.points_from_xy(change_df['X'], change_df['Y']))

#%% statistics
#stats = pd.DataFrame(change_df.M3C2_dista.describe())


#%%
#p_movement = pd.DataFrame({'X_0': change_df.X, 'Y_0': change_df.Y, 'Z_0': change_df.Z, 'Dx': change_df.Dx*(-1), 'Dy': change_df.Dy*(-1), 'Dz': change_df.Dz*(-1)})
p_movement = pd.DataFrame({'X_2018': change_df.X, 'Y_2018': change_df.Y, 'Z_2018': change_df.Z, 'Dx': change_df.Dx*(-1), 'Dy': change_df.Dy*(-1), 'Dz': change_df.Dz*(-1)})

#%%
years = np.arange(2019, 2041, 1)

for i in years:

    p_movement['X_%s' %(str(i))] = p_movement['X_%s' %str(i-1)] + p_movement.Dx
    p_movement['Y_%s' %(str(i))] = p_movement['Y_%s' %str(i-1)] + p_movement.Dy
    p_movement['Z_%s' %(str(i))] = p_movement['Z_%s' %str(i-1)] + p_movement.Dz

p_movement.to_csv(path_in + str(today) + '_p_movement.csv')
    
#%% #%% Create MultiPoint

coords= list(zip(p_movement.X_2018, p_movement.Y_2018))
mp = shapely.geometry.MultiPoint(coords)
XY_0 = mp.convex_hull
gdf = gpd.GeoDataFrame(index=[0], crs='epsg:32606', geometry=[XY_0])    
#gdf.to_file(path_in + str(today) + '_X_0.shp')
    

for i in years:
    coords= list(zip(p_movement['X_%s' %(str(i))], p_movement['Y_%s' %(str(i))]))
    mp = shapely.geometry.MultiPoint(coords)
    XY_hull = mp.convex_hull
#    gdf = gpd.GeoDataFrame(index=[0], crs='epsg:32606', geometry=[XY_hull]) # output every timestep as shapefile
#    gdf.to_file(path_in + str(today) + '_X_%s.shp' %(str(i)))
    gdf.at[i, 'geometry'] = XY_hull
#    gdf.to_file(path_in + str(today) + '_X_complete.shp') # output all the timesteps as shapefile
    gdf.to_file(path_in + str(today) + '_X_complete_clip.shp') # output all the timesteps as shapefile
