#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 11:55:33 2022

@author: fraukaiser
"""


import numpy as np
from sklearn import cluster
from osgeo import gdal
import matplotlib.pyplot as plt
import datetime, time, ogr
import pandas as pd
import geopandas as gpd
from math import sqrt
from skimage import filters
import fiona
import rasterio
import rasterio.features
from shapely.geometry import shape, mapping
from shapely.geometry.multipolygon import MultiPolygon
from affine import Affine


#%% functions 
def readRaster(fname):
    gtif = gdal.Open(fname)
    geotransform = gtif.GetGeoTransform()
    epsg = gtif.GetProjection()
    
    resolution = geotransform[1]
    xSize = gtif.RasterXSize
    ySize = gtif.RasterYSize
    bands = gtif.RasterCount
    original_img = np.array(gtif.GetRasterBand(1).ReadAsArray())
    return epsg, resolution, xSize, ySize, bands, geotransform, original_img
    gtif = None

def saveRaster(fname_out, array, xSize, ySize, nr_bands, spatial, epsg, format='GTiff', dtype=gdal.GDT_Float32):
    dst_ds = gdal.GetDriverByName(format).Create(fname_out, xSize, ySize, nr_bands, dtype)   
    dst_ds.SetGeoTransform(spatial)
    dst_ds.SetProjection(epsg) 
    dst_ds.GetRasterBand(1).WriteArray(array)

    dst_ds.FlushCache()
    dst_ds = None

def magnitude3D(Dx, Dy, Dz):
    length = []
    for i, j, k in zip(Dx, Dy, Dz):
        try:
            p = sqrt(i**2 + j**2 + k**2)
        except:
            p = np.nan
        length.append(p)
    return length

def magnitude2D(Dx, Dy):
    length = []
    for i, j in zip(Dx, Dy):
        try:
            p = sqrt(i**2 + j**2)
        except:
            p = np.nan
        length.append(p)
    return length

projection_3Dto2D = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 0]])

#%% Set working directory/ paths
    
path_in = '/home/fraukaiser/Desktop/Clouds/2022-02-25_QualityComparison/'
#os.mkdir(path_in + '/1_results')
today = datetime.date.today()   
start = datetime.datetime.now()
start_time = time.time()

#%% Load .csv M3C2 distance cloud (Output from Cloud Compare, unprocessed)

files = [path_in + i for i in [
                               '20201027_2018_DH4_RGB_camfile_II_feature.M3C2-1.23297.csv',                                                         # ppl I
                               '20201027_2018_DH4_RGB_camfile_II_feature.kNN4.M3C2-1.23273.csv',                                                    # ppl II
                               '20201027_2018_DH4_RGB_camfile_II_feature.kNN4.registeredBySegmentation.M3C2-1.23273.csv',                           # ppl III
                               '20201027_2018_DH4_RGB_camfile_II_feature.kNN4_arosicsshifted.registeredBySegmentation.rot_trans.M3C2-1.22772.csv',  # validation
                               '20201027_2018_DH4_RGB_camfile_II_feature.kNN4_arosicsshifted.registeredBySegmentation.M3C2-1.23273.csv',            # ppl IV
                               ]]

fend = '_%s_Dxyz_2D3D.csv' %str(today) # latest version (used for publication):  PPL I-IV 2022-10-25, for validation 2022-10-24
#%%
for fin in files:
    cloud = pd.read_csv(fin).replace([np.inf, -np.inf], np.nan)
    
    # Reverse sign for ['M3C2 distance', 'Nx', 'Ny', 'Nz'] since earlier cloud (2018) was compared to later cloud (2019)
    
    cloud.columns = ['X', 'Y', 'Z', 'Npoints_cloud1', 'Npoints_cloud2', 'STD_cloud1',
           'STD_cloud2', 'significant change', 'distance uncertainty',
           'M3C2_distance', 'Nx', 'Ny', 'Nz']
    
    for i in ['M3C2_distance', 'Nx', 'Ny', 'Nz']:
        cloud[i] = cloud[i] * (-1)
    
    # calcuate displacement vectors Dx, Dy, Dz and vector magnitude 2D and 3D
    
    try:    
        cloud['Dx'] = cloud['M3C2_distance'] * cloud['Nx']
        cloud['Dy'] = cloud['M3C2_distance'] * cloud['Ny']
        cloud['Dz'] = cloud['M3C2_distance'] * cloud['Nz']
    except:    
        cloud['M3C2_distance'] = pd.to_numeric(cloud.M3C2_distance)
        
        cloud['Dx'] = cloud['M3C2_distance'] * cloud['Nx']
        cloud['Dy'] = cloud['M3C2_distance'] * cloud['Ny']
        cloud['Dz'] = cloud['M3C2_distance'] * cloud['Nz']
            
    threeD_vector = np.matrix([cloud.Dx, cloud.Dy, cloud.Dz])
    twoD_vector = np.dot(projection_3Dto2D, threeD_vector)
    cloud['Dx2D'] = np.reshape(twoD_vector[0], (len(cloud), 1))
    cloud['Dy2D'] = np.reshape(twoD_vector[1], (len(cloud), 1))
    
    length = magnitude2D(np.array(cloud['Dx2D']), np.array(cloud['Dy2D']))
    cloud['2D_magnitude'] = length
    
    length = magnitude3D(np.array(cloud['Dx']), np.array(cloud['Dy']), np.array(cloud['Dz']))
    cloud['3D_magnitude'] = length  
    
#    fname = fin.rsplit('/', 1)[1][:-4]
    fout = fin[:-4] + fend
    print('saving... ' + fout)
    cloud.to_csv(fout)
#%%  write point cloud to shapefile for rasterization (see from line 173)

cloud_gdf = gpd.GeoDataFrame(cloud, geometry=gpd.points_from_xy(cloud['X'], cloud['Y'])) # loads point cloud of ppl IV
cloud_gdf = cloud_gdf.set_crs("EPSG:32606")
cloud_gdf.to_file(fout[:-4] + '.shp')

#%% Load Orthophoto for parameters

x = readRaster('/home/fraukaiser/Desktop/Clouds/2022-02-08_arosics/20201027_2018_DH4_RGB_camfile_II_feature_max_50_globally_shifted_to__20201103_2019_DH4_RGB_camfile_III_feature.tif') # dimension from AROSICS shifted raster

epsg = x[0]
resolution = x[1]
xSize = x[2]
ySize = x[3]
bands = x[4]
gtransform = x[5]
orthophoto = x[6]

#%%
attribute = 'Dz'
power = 3

to_resolution = 5      
# other tested spatial resolution:
#to_resolution = 2.5
#to_resolution = 10
#to_resolution = 20

algorithm = "invdist:power=3.0:smoothing=0.0:radius1=0.0:radius2=0.0:angle=0.0:max_points=0:min_points=0:nodata=0.0"

ulx = gtransform[0]
uly = gtransform[3]

lrx = ulx + xSize*resolution
lry = uly - ySize*resolution

to_xSize = int((lrx-ulx)/to_resolution)
to_ySize = int((lry-uly)/(-to_resolution))

bounds = [ulx, uly, lrx, lry]

#%% define shapefile input for rasterization and name of output raster
shp = fout[:-4] + '.shp'
ras_out = path_in + '/1_results/%s_%s_fctr%s_%sm.tif' %(str(today), attribute, str(power), str(to_resolution))

#%% optional: show field names of Shapefile just for double checking 

pts = ogr.Open(shp, 0)
layer = pts.GetLayer()
for field in layer.schema:
    print(field.name)
    
#%% 

#%% #%% Rasterize Cloud via IDW, factor 3 and resample to coarser spatial reolution of 5 m/px 

idw = gdal.Grid(ras_out, 
                shp,  # this line wasn't automated
                format = "GTiff", 
                outputBounds = bounds, 
                outputSRS = epsg, 
                width = to_xSize, 
                height = to_ySize, 
                zfield = attribute, 
                algorithm = algorithm)
idw = None
#%% Load IDW Dz Raster 
x = readRaster(ras_out)
original_img = x[6]

#%% apply Sobel Filter for edge detection (high image gradients)

sobel_img = filters.sobel(original_img)
plt.imshow(sobel_img)

#%% write sobel to .tif
gtransform = (ulx, to_resolution, 0.0, uly, 0.0, to_resolution * (-1))   
saveRaster(ras_out[:-4] + '_sobel.tif', sobel_img, to_xSize, to_ySize, 1, gtransform, epsg)

#%% apply kMeans classification (two classes: high image gradient, low image gradient)

X_single = sobel_img.reshape((-1,1))
#%% Run kMeans cluster analysis ONE BAND (adapted from: https://www.acgeospatial.co.uk/k-means-sentinel-2-python/)

k_means_single = cluster.KMeans(n_clusters=2)
k_means_single.fit(X_single)

X_cluster_single = k_means_single.labels_
X_cluster_single = X_cluster_single.reshape(original_img.shape)

#%% optional: for visualization
for i in [original_img, sobel_img, X_cluster_single]:
    plt.imshow(i)
    plt.show()
#%% save raster after Sobel and kMeans to raster file
gtransform = (ulx, to_resolution, 0.0, uly, 0.0, to_resolution * (-1))   
saveRaster(ras_out[:-4] + '_sobel_kMeans.tif', X_cluster_single, to_xSize, to_ySize, 1, gtransform, epsg)
#%% Read input band with Rasterio (adapted from https://gis.stackexchange.com/questions/295362/polygonize-raster-file-according-to-band-values)
# Keep track of unique pixel values in the input band
unique_values = np.unique(X_cluster_single)
# Polygonize with Rasterio. `shapes()` returns an iterable
# of (geom, value) as tuples
afn = Affine.from_gdal(*gtransform)
shapes = list(rasterio.features.shapes(X_cluster_single, transform=afn))

shp_schema = {
    'geometry': 'MultiPolygon',
    'properties': {'pixelvalue': 'int'}
}

# Write to shapefile
with fiona.open('%s_sobel_kMeans_mask.shp' %ras_out[:-4], 'w', 'ESRI Shapefile', shp_schema, epsg) as shp:
    for pixel_value in unique_values:
        polygons = [shape(geom) for geom, value in shapes
                    if value == 1]
        multipolygon = MultiPolygon(polygons)
        shp.write({
            'geometry': mapping(multipolygon),
            'properties': {'pixelvalue': int(pixel_value)}
        })
    
#%% Clip point cloud ppl 4 data to polygon mask of Sobel/ kMeans -> high image gradient

mask_gdf = gpd.read_file('%s_sobel_kMeans_mask.shp' %ras_out[:-4])

#%%
# Clip the data using GeoPandas clip
points_clip = gpd.clip(cloud_gdf, mask_gdf)
points_clip.to_file('%s_masked_%s.shp'% (fout[:-4], str(to_resolution)))

#%% 

#%%

now = datetime.datetime.now()
print('started at %s,  finished at %s' % (str(start), str(now))) 
print('Total computing time: --------------- %s seconds -----------------' % (time.time() - start_time))
