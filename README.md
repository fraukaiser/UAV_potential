# The Potential of UAV Imagery for the Detection of Rapid Permafrost Degradation

Python code to detect land surface displacements on the basis of distance point clouds derived from UAV.

## Prerequisites

- Python 3.6+ with the following packages installed: affine, Fiona, GDAL, geopandas, matplotlib, numpy, pandas, rasterio, scikit-image, scikit-learn, Shapely

## Input

Distance Point Clouds after Multiscale Model to Model Cloud Comparison (M3C2 after Lague et al., 2013) in CloudCompare in a .csv format
- post-processing level I: raw point clouds compared
- post-processing level II: denoised point clouds compared
- post-processing level III: denoised and Iterative Closest Point (ICP) aligned clouds compared
- post-processing level IV: denoised, AROSICS shifted and ICP aligned point clouds compared


## Workflow

The provided python script:
- calculates the displacement vectors Dx, Dy, Dz for each distance point cloud
- saves the output in a .csv file
- rasterizes attribute "vertical displacement (Dz)" of the distance point cloud with highest accuracy (post-processing level IV)
- applies a Sobel edge detection filter to highlight high image gradients 
- applies kMeans algorithm to cluster image into two categories: change (high image gradient) and no change (low image gradient)
