import os
import glob
import subprocess
import json
import warnings
from typing import List, Union
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import sklearn.metrics as metrics
import pickle

import ee
import rasterio
import rioxarray as rxr
import sankee
import fiona
from osgeo import gdal, osr, gdalconst
import numpy.ma as ma

import geemap

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import statsmodels
import matplotlib.cm as cm
import matplotlib.ticker as ticker

from glob import glob
from scipy import stats
from shapely.geometry import Polygon, Point, box

from rasterio import features, mask, plot as rplot, merge
from rasterio.plot import show
from rasterstats import zonal_stats

from rio_cogeo.profiles import cog_profiles

from geopandas import GeoDataFrame
from geocube.api.core import make_geocube

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.offsetbox import AnchoredText
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.lines import Line2D


from sklearn import ensemble
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


def getFactorImg(factorNames: List[str], image: ee.Image) -> ee.Image:
    """
    Create an image from a list of factor names extracted from an image.

    Args:
        factorNames: List of factor names to select from the image.
        image: The input image.

    Returns:
        The image with the selected factors as constant bands.

    Raises:
        None.
    """
    factorList = image.toDictionary().select(factorNames).values()
    return ee.Image.constant(factorList)


def NormalizeData(data: Union[list, np.ndarray]) -> Union[list, np.ndarray]:
    """
    Normalize the given data by scaling it between 0 and 1.

    Args:
        data: The input data as a list or NumPy array.

    Returns:
        The normalized data as a list or NumPy array.

    Raises:
        TypeError: If the input data is not a list or NumPy array.
        ValueError: If the input data has insufficient elements for normalization.
    """
    # Convert input data to NumPy array if it's a list
    if isinstance(data, list):
        data = np.array(data)

    # Check if the input data is a NumPy array
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a list or NumPy array.")

    # Check if the input data has at least two elements
    if data.size < 2:
        raise ValueError("Input data must have at least two elements for normalization.")

    # Perform data normalization
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # Return the normalized data
    return normalized_data


def getQABits(image: ee.Image, start: int, end: int, mask: str) -> ee.Image:
    """
    Extract bits from an image based on a given range and apply a mask.

    Args:
        image: The input image.
        start: The starting bit index.
        end: The ending bit index (exclusive).
        mask: The name of the mask band.

    Returns:
        The image with the extracted bits and mask applied.

    Raises:
        None.
    """
    pattern = 0
    for i in range(start, end):
        pattern += 2 ** i
    return image.select([0], [mask]).bitwiseAnd(pattern).rightShift(start)



# Function to add raster to map
def addL8(Map, layer, mi, ma, name):
    Map.addLayer(layer,
                 {'bands': ['SR_B4', 'SR_B3', 'SR_B2'],'max': ma, 'min' : mi},
                 name=name)
    
# Function to add raster to map
def addL5(Map, layer, mi, ma, name):
    Map.addLayer(layer,
                 {'bands': ['SR_B3', 'SR_B2', 'SR_B1'],'max': ma, 'min' : mi},
                 name=name)
    
def fmask(image):
    # see https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2
    # Bit 0 - Fill
    # Bit 1 - Dilated Cloud
    # Bit 2 - Cirrus
    # Bit 3 - Cloud
    # Bit 4 - Cloud Shadow
    qaMask = image.select("QA_PIXEL").bitwiseAnd(int("11111", 2)).eq(0)

    # Apply the scaling factors to the appropriate bands.
    opticalBands = image.select("SR_B.").multiply(0.0000275).add(-0.2)

    # Replace the original bands with the scaled ones and apply the masks.
    return image.addBands(opticalBands, None, True).updateMask(qaMask)




def remove_sunglint_L8(image: ee.Image, glint_geo: ee.Geometry) -> ee.Image:
    """
    Remove sunglint from an image using the glint removal technique.

    Args:
        image: The input image with sunglint.
        glint_geo: Geometry to clip the image for slope calculation.

    Returns:
        The image with sunglint removed.

    Raises:
        ee.EEException: If an error occurs during Earth Engine computation.
    """
    # Band selection
    B2 = image.select(['SR_B5', 'SR_B2']).clip(glint_geo)
    B3 = image.select(['SR_B5', 'SR_B3']).clip(glint_geo)
    B4 = image.select(['SR_B5', 'SR_B4']).clip(glint_geo)
    B5 = image.select(['SR_B5']).clip(glint_geo)

    # Linear fit reduction
    lfitB2 = B2.reduceRegion(
        reducer=ee.Reducer.linearFit(),
        geometry=glint_geo,
        scale=30,
        bestEffort=True
    )
    lfitB3 = B3.reduceRegion(
        reducer=ee.Reducer.linearFit(),
        geometry=glint_geo,
        scale=30,
        bestEffort=True
    )
    lfitB4 = B4.reduceRegion(
        reducer=ee.Reducer.linearFit(),
        geometry=glint_geo,
        scale=30,
        bestEffort=True
    )

    # Extract slope values
    slope_B2 = ee.Image.constant(lfitB2.get('scale')).clip(glint_geo).rename('slope_B2')
    slope_B3 = ee.Image.constant(lfitB3.get('scale')).clip(glint_geo).rename('slope_B3')
    slope_B4 = ee.Image.constant(lfitB4.get('scale')).clip(glint_geo).rename('slope_B4')

    # Extract minimum B8 value
    min_B5 = ee.Image.constant(image.select('SR_B5').reduceRegion(
        ee.Reducer.min(),
        geometry=glint_geo,
        scale=30
    ).get('SR_B5')).rename('min_B5')

    # Create glint factors image
    glint_factors = ee.Image([slope_B2, slope_B3, slope_B4, min_B5])
    image_add = image.addBands(glint_factors)

    # Perform deglinting
    deglint_B2 = image_add.expression(
        'Blue - (Slope * (NIR - MinNIR))', {
        'Blue': image_add.select('SR_B2'),
        'NIR': image_add.select('SR_B5'),
        'MinNIR': image_add.select('min_B5'),
        'Slope': image_add.select('slope_B2')
    }).rename('SR_B2')

    deglint_B3 = image_add.expression(
        'Green - (Slope * (NIR - MinNIR))', {
        'Green': image_add.select('SR_B3'),
        'NIR': image_add.select('SR_B5'),
        'MinNIR': image_add.select('min_B5'),
        'Slope': image_add.select('slope_B3')
    }).rename('SR_B3')

    deglint_B4 = image_add.expression(
        'Red - (Slope * (NIR - MinNIR))', {
        'Red': image_add.select('SR_B4'),
        'NIR': image_add.select('SR_B5'),
        'MinNIR': image_add.select('min_B5'),
        'Slope': image_add.select('slope_B4')
    }).rename('SR_B4')

    # Create deglinted image
    image_deglint = ee.Image([deglint_B2, deglint_B3, deglint_B4, B5])

    return image_deglint

# Function to mosaic images
def mosaicImages(image1, image2):
    return ee.Image(image1).addBands(image2)


# Function to blend images iteratively
def blend_images(image_list):
    # Start with the first image in the list
    blended_image = image_list[0]
    
    # Blend each subsequent image in the list
    for i in range(1, len(image_list)):
        blended_image = blended_image.blend(image_list[i])
    
    return blended_image

def get_LS8_image(sites: ee.Geometry, start_date: str, end_date: str, cloud: int) -> ee.Image:
    """
    Retrieve a single Landsat 8 image from the year 2022, filtered by location, cloud cover, and date range.

    Args:
        sites: An Earth Engine Geometry object representing the region of interest.
        start_date: The start date of the image collection filtering in 'YYYY-MM-DD' format.
        end_date: The end date of the image collection filtering in 'YYYY-MM-DD' format.

    Returns:
        The Landsat 8 image from 2022, filtered by location, cloud cover, and date range.

    Raises:
        None.
    """
    # Filter Landsat 8 Collection
    l8_collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')

    # Apply filters
    filtered_collection = l8_collection.filterBounds(sites)\
        .filterMetadata('CLOUD_COVER', 'less_than', cloud)\
        .filterDate(start_date, end_date)\
        .map(fmask)

    # Compute median and clip to the region of interest
    median_image = filtered_collection.median().select(['SR_B2', 'SR_B3' ,'SR_B4', 'SR_B5']).clip(sites)
    
    don = sites.filter(ee.Filter.eq('Location', 'Dongara'))
    cliff = sites.filter(ee.Filter.eq('Location', 'Cliff Head'))
    tr = sites.filter(ee.Filter.eq('Location', 'Two Rocks'))
    jur = sites.filter(ee.Filter.eq('Location', 'Jurien Bay'))
    cerv = sites.filter(ee.Filter.eq('Location', 'Cervantes'))
    lanc = sites.filter(ee.Filter.eq('Location', 'Lancelin'))
    
    
    de_don = remove_sunglint_L8(image = median_image, glint_geo = don)
    de_cliff = remove_sunglint_L8(image = median_image, glint_geo = cliff)
    de_tr = remove_sunglint_L8(image = median_image, glint_geo = tr)
    de_jur = remove_sunglint_L8(image = median_image, glint_geo = jur)
    de_cerv = remove_sunglint_L8(image = median_image, glint_geo = cerv)
    de_lance = remove_sunglint_L8(image = median_image, glint_geo = lanc)
    
    image_collection = [de_don, de_cliff, de_tr, de_jur, de_cerv, de_lance]
    merged_image = blend_images(image_collection)

    return median_image



def export_l8_to_cloud_storage(image: ee.Image, description: str, file_name_prefix: str, crs: str,
                              scale: int, region: ee.Geometry, file_format: str, bucket: str,
                              max_pixels: int, skip_empty_tiles: bool, format_options: dict) -> ee.batch.Task:
    """
    Export Landsat 8 image to Cloud Storage.

    Args:
        image (ee.Image): Landsat 8 image.
        description (str): Description for the export task.
        file_name_prefix (str): Prefix for the output file name.
        crs (str): Coordinate Reference System (e.g., 'EPSG:4326').
        scale (int): Scale in meters.
        region (ee.Geometry): Region of interest.
        file_format (str): Output file format (e.g., 'GeoTIFF').
        bucket (str): Cloud Storage bucket name.
        max_pixels (int): Maximum number of pixels allowed in the export.
        skip_empty_tiles (bool): Whether to skip empty tiles in the export.
        format_options (dict): Format-specific options.

    Returns:
        ee.batch.Task: Export task.

    """
    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description=description,
        fileNamePrefix=file_name_prefix,
        crs=crs,
        scale=scale,
        region=region,
        fileFormat=file_format,
        bucket=bucket,
        maxPixels=max_pixels,
        skipEmptyTiles=skip_empty_tiles,
        formatOptions=format_options
    )
    task.start()
    return task


def get_LS5_image(sites: ee.Geometry, start_date: str, end_date: str, cloud: int) -> ee.Image:
    """
    Retrieve a single Landsat 5 image from the year 2022, filtered by location, cloud cover, and date range.

    Args:
        sites: An Earth Engine Geometry object representing the region of interest.
        start_date: The start date of the image collection filtering in 'YYYY-MM-DD' format.
        end_date: The end date of the image collection filtering in 'YYYY-MM-DD' format.

    Returns:
        The Landsat 8 image from 2022, filtered by location, cloud cover, and date range.

    Raises:
        None.
    """
    # Filter Landsat 5 Collection
    l5_collection = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")

    # Apply filters
    filtered_collection = l5_collection.filterBounds(sites)\
        .filterMetadata('CLOUD_COVER', 'less_than', cloud)\
        .filterDate(start_date, end_date)\
        .map(fmask)

    # Compute median and clip to the region of interest
    median_image = filtered_collection.median().select(['SR_B1', 'SR_B2' ,'SR_B3', 'SR_B4']).clip(sites)
    
    don = sites.filter(ee.Filter.eq('Location', 'Dongara'))
    cliff = sites.filter(ee.Filter.eq('Location', 'Cliff Head'))
    tr = sites.filter(ee.Filter.eq('Location', 'Two Rocks'))
    jur = sites.filter(ee.Filter.eq('Location', 'Jurien Bay'))
    cerv = sites.filter(ee.Filter.eq('Location', 'Cervantes'))
    lanc = sites.filter(ee.Filter.eq('Location', 'Lancelin'))
    
    
    de_don = remove_sunglint_L8(image = median_image, glint_geo = don)
    de_cliff = remove_sunglint_L8(image = median_image, glint_geo = cliff)
    de_tr = remove_sunglint_L8(image = median_image, glint_geo = tr)
    de_jur = remove_sunglint_L8(image = median_image, glint_geo = jur)
    de_cerv = remove_sunglint_L8(image = median_image, glint_geo = cerv)
    de_lance = remove_sunglint_L8(image = median_image, glint_geo = lanc)
    

    image_collection = [de_don, de_cliff, de_tr, de_jur, de_cerv, de_lance]
    merged_image = blend_images(image_collection)

    return median_image



def remove_sunglint_L5(image: ee.Image, glint_geo: ee.Geometry) -> ee.Image:
    """
    Remove sunglint from an image using the glint removal technique.

    Args:
        image: The input image with sunglint.
        glint_geo: Geometry to clip the image for slope calculation.

    Returns:
        The image with sunglint removed.

    Raises:
        ee.EEException: If an error occurs during Earth Engine computation.
    """
    # Band selection
    B1 = image.select(['SR_B4', 'SR_B1'])
    B2 = image.select(['SR_B4', 'SR_B2'])
    B3 = image.select(['SR_B4', 'SR_B3'])
    B4 = image.select(['SR_B4'])

    # Linear fit reduction
    lfitB1 = B1.reduceRegion(
        reducer=ee.Reducer.linearFit(),
        geometry=glint_geo,
        scale=30,
        bestEffort=True
    )
    lfitB2 = B2.reduceRegion(
        reducer=ee.Reducer.linearFit(),
        geometry=glint_geo,
        scale=30,
        bestEffort=True
    )
    lfitB3 = B3.reduceRegion(
        reducer=ee.Reducer.linearFit(),
        geometry=glint_geo,
        scale=30,
        bestEffort=True
    )

    # Extract slope values
    slope_B1 = ee.Image.constant(lfitB1.get('scale')).clip(glint_geo).rename('slope_B1')
    slope_B2 = ee.Image.constant(lfitB2.get('scale')).clip(glint_geo).rename('slope_B2')
    slope_B3 = ee.Image.constant(lfitB3.get('scale')).clip(glint_geo).rename('slope_B3')

    # Extract minimum B8 value
    min_B4 = ee.Image.constant(image.select('SR_B4').reduceRegion(
        ee.Reducer.min(),
        geometry=glint_geo,
        scale=30
    ).get('SR_B4')).rename('min_B4')

    # Create glint factors image
    glint_factors = ee.Image([slope_B1, slope_B2, slope_B3, min_B4])
    image_add = image.addBands(glint_factors)

    # Perform deglinting
    deglint_B1 = image_add.expression(
        'Blue - (Slope * (NIR - MinNIR))', {
        'Blue': image_add.select('SR_B1'),
        'NIR': image_add.select('SR_B4'),
        'MinNIR': image_add.select('min_B4'),
        'Slope': image_add.select('slope_B1')
    }).rename('SR_B1')

    deglint_B2 = image_add.expression(
        'Green - (Slope * (NIR - MinNIR))', {
        'Green': image_add.select('SR_B2'),
        'NIR': image_add.select('SR_B4'),
        'MinNIR': image_add.select('min_B4'),
        'Slope': image_add.select('slope_B2')
    }).rename('SR_B2')

    deglint_B3 = image_add.expression(
        'Red - (Slope * (NIR - MinNIR))', {
        'Red': image_add.select('SR_B3'),
        'NIR': image_add.select('SR_B4'),
        'MinNIR': image_add.select('min_B4'),
        'Slope': image_add.select('slope_B3')
    }).rename('SR_B3')

    # Create deglinted image
    image_deglint = ee.Image([deglint_B1, deglint_B2, deglint_B3, B4])

    return image_deglint



def export_l5_to_cloud_storage(image: ee.Image, description: str, file_name_prefix: str, crs: str,
                              scale: int, region: ee.Geometry, file_format: str, bucket: str,
                              max_pixels: int, skip_empty_tiles: bool, format_options: dict) -> ee.batch.Task:
    """
    Export Landsat 5 image to Cloud Storage.

    Args:
        image (ee.Image): Landsat 5 image.
        description (str): Description for the export task.
        file_name_prefix (str): Prefix for the output file name.
        crs (str): Coordinate Reference System (e.g., 'EPSG:4326').
        scale (int): Scale in meters.
        region (ee.Geometry): Region of interest.
        file_format (str): Output file format (e.g., 'GeoTIFF').
        bucket (str): Cloud Storage bucket name.
        max_pixels (int): Maximum number of pixels allowed in the export.
        skip_empty_tiles (bool): Whether to skip empty tiles in the export.
        format_options (dict): Format-specific options.

    Returns:
        ee.batch.Task: Export task.

    """
    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description=description,
        fileNamePrefix=file_name_prefix,
        crs=crs,
        scale=scale,
        region=region,
        fileFormat=file_format,
        bucket=bucket,
        maxPixels=max_pixels,
        skipEmptyTiles=skip_empty_tiles,
        formatOptions=format_options
    )
    task.start()
    return task




def process_rgb_stack(rgb_stack: rasterio.DatasetReader, dat_dir: str, year: str, geom_path: str) -> None:
    """
    Process RGB stack and generate kndavi TIFF file.

    Args:
        rgb_stack (rasterio.DatasetReader): The RGB stack dataset.
        dat_dir (str): The directory where the output files will be saved.
        year (str): The year associated with the output files.
    """
    # Update profile metadata
    profile = rgb_stack.meta
    profile.update(
        dtype=rasterio.float64,
        count=1,
        compress='lzw'
    )

    # Read individual bands from RGB stack
    b1 = rgb_stack.read(1)
    b2 = rgb_stack.read(2)
    b3 = rgb_stack.read(3)
    b4 = rgb_stack.read(4)

    # Calculate distance
    dist = np.abs(b4 - b1)

    # Save distance as a TIFF file
    dist_out = dat_dir
    with rasterio.open(os.path.join(dist_out, f'dist_{year}.tif'), 'w', **profile) as dst:
        dst.write_band(1, dist)

    # Perform zonal statistics
    zon = zonal_stats(
        geom_path,
        os.path.join(dist_out, f'dist_{year}.tif'),
        stats="mean",
        geojson_out=True
    )

    result = {"type": "FeatureCollection", "features": zon}

    # Save zonal statistics as GeoJSON
    with open(os.path.join(dat_dir, 'geoj.geojson'), 'w') as outfile:
        json.dump(result, outfile)

    # Convert GeoJSON to shapefile
    geoj = gpd.read_file(os.path.join(dat_dir, 'geoj.geojson'))
    geoj.to_file(os.path.join(dat_dir, f'mu_dist_{year}.shp'))

    vector_fn = os.path.join(dat_dir, f'mu_dist_{year}.shp')

    # Read vector file
    vector = gpd.read_file(vector_fn)

    # Create geometry-value pairs
    geom_value = ((geom, value) for geom, value in zip(vector.geometry, vector['mean']))

    # Rasterize the geometry-value pairs
    rasterized = features.rasterize(
        geom_value,
        out_shape=rgb_stack.shape,
        transform=rgb_stack.transform,
        all_touched=True,
        dtype=np.float64
    )

    # Save rasterized image as TIFF
    with rasterio.open(
            os.path.join(dat_dir, f'Sigma_{year}.tif'), "w",
            driver="GTiff",
            transform=rgb_stack.transform,
            dtype=rasterio.float64,
            count=1,
            width=rgb_stack.width,
            height=rgb_stack.height
    ) as dst:
        dst.write(rasterized, indexes=1)

    # Calculate kndavi
    sig_ras = rasterio.open(os.path.join(dat_dir, f'Sigma_{year}.tif'))
    s = sig_ras.read(1)
    sigma = s
    knr = np.exp(-(b4 - b1) ** 2 / (2 * sigma ** 2))
    kndavi = ((1 - knr) / (1 + knr))
              
    # Save kndavi as a TIFF file
    kndavi_out = dat_dir
    with rasterio.open(os.path.join(kndavi_out, f'kndavi_{year}.tif'), 'w', **profile) as dst:
        dst.write_band(1, kndavi)
        

        
        
def filter_and_dropna(gdf: pd.DataFrame, sub: List[str], query: List[str]) -> pd.DataFrame:
    """
    Filters a GeoDataFrame based on a list of conditions and drops rows with missing values in selected columns.
    
    Args:
        gdf (GeoDataFrame): The input GeoDataFrame containing the data.
        sub (List[str]): List of column names to be used for filtering and dropping rows.
        query (List[str]): List of conditions for filtering the GeoDataFrame.

    Returns:
        GeoDataFrame: The filtered and cleaned GeoDataFrame.
    """
    filtered_gdf = gdf.query(query)
    filtered_gdf = filtered_gdf.dropna(subset=sub)
    
    return filtered_gdf        
        
        
def generate_dendrogram(df: pd.DataFrame, sub: List[str]) -> None:
    """
    Generates a dendrogram based on clustering of columns in a GeoDataFrame.
    
    Args:
        gdf (GeoDataFrame): The input GeoDataFrame containing the data.
        sub (List[str]): List of column names to be used for clustering.

    Returns:
        None.
    """
    hr_df = df[sub].copy()

    fig_dendo = plt.figure(figsize=(10, 10))
    ax_dendo = fig_dendo.add_subplot(111)
    linkage_data = linkage(hr_df, method='ward', metric='euclidean')
    dendrogram(linkage_data)
    plt.show()
    
    
def perform_clustering(clust_df: pd.DataFrame, sub: List[str], sub_c: List[str],  dat_dir: str, gdf: gpd.GeoDataFrame, que: List[str]) -> pd.DataFrame:
    """
    Performs hierarchical clustering on a DataFrame, assigns cluster labels, and saves the results to CSV and shapefile.
    
    Args:
        clust_df (pd.DataFrame): The DataFrame containing the data for clustering.
        prop_c (List[str]): List of property columns to include in the analysis.
        dat_dir (str): The directory path to save the output CSV file.
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing the spatial data.
        que (List[str]): List of query conditions to filter the data.

    Returns:
        pd.DataFrame: The concatenated DataFrame with cluster labels.

    """
    hierarchical_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    labels = hierarchical_cluster.fit_predict(clust_df[sub]).round(0)
    clust_df["Cluster"] = list(map(str, labels))

    #sumdf = clust_df[prop_c].groupby('Cluster').describe(percentiles=[0.50])
    
    outdf = os.path.join(dat_dir, 'Dataframes\\BOSS_clust.csv')
    clust_df.to_csv(outdf)

    gdf_sub = gdf.query(que)
    gdf_sub = gdf_sub.dropna(subset=sub)
    gdf_sub["Cluster"] = list(map(str, labels))
    boss_out = os.path.join(dat_dir, 'BOSS\\Cleaned\\BOSS_kndavi.shp')
    gdf_sub.to_file(boss_out)

    df_duplicate = clust_df.copy()
    df_duplicate['site'] = 'All sites'
    frames = [clust_df, df_duplicate]
    df_dup = pd.concat(frames)
    df_dup = df_dup.reset_index()
    
    return df_dup


def plot_box_strip(bdf: pd.DataFrame, loc: str) -> None:
    """
    Plots a boxplot and stripplot for the kNDAVI values grouped by site and cluster.

    Args:
        bdf (pd.DataFrame): The DataFrame containing the data for plotting.

    Returns:
        None
    """

    bdf.loc[(bdf.Cluster == "0") & (bdf.kndavi > 0.5), "kndavi"] = np.nan

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)

    sns.boxplot(x=bdf['site'],
                y=bdf['kndavi'],
                whis=15,
                dodge=True,
                showfliers=False,
                hue=bdf['Cluster'],
                palette=['#22BF45', '#D4C27E'],
                ax=ax,
                linewidth=1)

    sns.stripplot(data=bdf, x='site', y='kndavi', palette=['#039a25', '#c2a742'],
                  jitter=True, edgecolor='black', marker='o', linewidth=0.3, alpha=0.7, s=4,
                  hue='Cluster', dodge=True, ax=ax)

    ax.legend(title='Cluster',
              loc=loc,
              labels=['SAV', 'Other'])

    leg = ax.get_legend()
    leg.legendHandles[0].set_color('#22BF45')
    leg.legendHandles[1].set_color('#D4C27E')

    ax.annotate('A)',
                xy=(0.033, 0.94),
                xycoords='subfigure fraction',
                ha='center', va='center',
                fontsize=10)

    ax.set_xlabel('Site')
    fig.suptitle('Landsat 8 (2021) kNDAVI')
    ax.set_ylabel('kNDAVI')
    plt.tight_layout()
    plt.show()

    
def reclassify_kndavi(years: List[str], dat_dir: str, thresh: float):
    """
    Reclassify kNDVI raster images for multiple years.
    
    Args:
        years: List of years as strings.
        dat_dir: Directory path where the kNDVI files are located.
    """
    for year in years:
        year_str = str(year)
        kndavi_file = f'kndavi_{year_str}.tif'
        reclass_file = f'reclass/reclass_{year_str}.tif'
        
        kndavi_path = os.path.join(dat_dir, kndavi_file)
        reclass_path = os.path.join(dat_dir, reclass_file)
        
        with rasterio.open(kndavi_path) as src:
            # Read as numpy array
            array = src.read()
            profile = src.profile

            # Reclassify
            array[np.where(array > thresh)] = 2 
            array[np.where(array <= thresh)] = 1
            # and so on ...

        with rasterio.open(reclass_path, 'w', **profile) as dst:
            # Write to disk
            dst.write(array)
            
            
            
def calculate_sav_area(sites: List[str], sav_list: List[str], cwd: str, site_path: str) -> pd.DataFrame:
    import glob
    """
    Calculate the area and percentage of SAV (Submerged Aquatic Vegetation) for each site and year.

    Args:
        sites (List[str]): List of site names.
        sav_list (List[str]): List of file paths to SAV raster files.
        cwd (str): Directory path.

    Returns:
        pd.DataFrame: DataFrame containing the site, year, area of SAV (in km2), and percentage of SAV.

    Raises:
        FileNotFoundError: If the file paths provided in `sav_list` do not exist.
    """
    site_list = []
    year_list = []
    area_list = []
    perc_list = []
    err_list = []

    for shp in sites:
        bound_p = os.path.join(cwd, 'data\\ICoAST sites\\Sites')
        shp_file = glob.glob(os.path.join(bound_p, shp + '*.shp'))
        ex = gpd.read_file(*shp_file)
        ext = ex.explode().geometry

        for tif in sav_list:
            # nm = '_'.join(tif.split('\\')[-1].split('.')[0].split('_')[1:])
            nm = str(20) + '_'.join(tif.split('\\')[-1].split('.')[0].split('_')[-1:])
            class_ras = rasterio.open(tif)
            cl_crop, cl_transform = mask(class_ras, ext, crop=True, nodata=255, filled=False, all_touched=False)
            sav_pix = ma.masked_not_equal(cl_crop, 1)
            area = ma.MaskedArray.count(sav_pix, keepdims=False)
            count = ma.MaskedArray.count(cl_crop, keepdims=False)
            err = area * 0.08
            m2 = area * (28.6 * 28.6)
            ha = m2 / 10000
            km2 = m2 * 0.000001
            perc = area / count * 100
            err_tot = err/ count * 100

            if shp == 'Cliff_head':
                shp = 'Freshwater'

            site_list.append(shp)
            year_list.append(nm)
            area_list.append(km2)
            perc_list.append(perc)
            err_list.append(err_tot)

    areadf = pd.DataFrame(list(zip(site_list, year_list, area_list, perc_list, err_list)), columns=['Site', 'Year', 'Area SAV', 'Percentage SAV','Error'])
    return areadf



def plot_cropped_landsat(
    landsat_path: str,
    shapefile_path: str,
    title: str,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot the cropped Landsat image.

    Args:
        landsat_path (str): Path to the Landsat image file.
        shapefile_path (str): Path to the shapefile.
        title (str): Title of the plot.
        ax (plt.Axes, optional): Matplotlib Axes object to plot on. If not provided, a new Axes object is created.

    Returns:
        plt.Axes: Matplotlib Axes object containing the plot.
    """
    # Read the shapefile
    shapefile = gpd.read_file(shapefile_path)

    # Open the Landsat image
    landsat = rxr.open_rasterio(landsat_path, masked=True)

    # Crop the Landsat image using the shapefile
    cropped_landsat = landsat.rio.clip(shapefile.geometry)

    # Select the first three bands (RGB)
    cropped_rgb = cropped_landsat[:3]

    # Reverse the order of the bands
    cropped_rgb = cropped_rgb[::-1]

    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    cropped_rgb.plot.imshow(robust=True, ax=ax)

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Remove lon and lat labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Remove the box around the plot
    ax.set_frame_on(False)
    
    # Add scale bar
    scalebar = ScaleBar(111, "km", frameon=False, location='lower right', color='black', box_alpha=0)
    ax.add_artist(scalebar)

    ax.set_title(title)
    
    return ax


# def plot_cropped_landsat(landsat_path, shapefile_path, title, ax):
#     # Read the shapefile
#     shapefile = gpd.read_file(shapefile_path)

#     # Open the Landsat image
#     landsat = rxr.open_rasterio(landsat_path, masked=True)

#     # Crop the Landsat image using the shapefile
#     cropped_landsat = landsat.rio.clip(shapefile.geometry)

#     # Select the first three bands (RGB)
#     cropped_rgb = cropped_landsat[:3]

#     # Reverse the order of the bands
#     cropped_rgb = cropped_rgb[::-1]

#     # Create the plot
#     #fig, ax = plt.subplots(figsize=(10, 10))
#     cropped_rgb.plot.imshow(robust=True, ax=ax)

#     # Remove axis ticks and labels
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
    
#     # Remove lon and lat labels
#     ax.set_xlabel('')
#     ax.set_ylabel('')
    
#     #Remove the box around the plot
#     ax.set_frame_on(False)
    
#     #Add scale bar
#     scalebar = ScaleBar(111, "km", frameon=False, location='lower right', color='black', box_alpha=0)
#     ax.add_artist(scalebar)

#     ax.set_title(title)



def plot_cropped_raster(
    raster_path: str,
    shapefile_path: str,
    title: str,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot the cropped raster.

    Args:
        raster_path (str): Path to the raster file.
        shapefile_path (str): Path to the shapefile.
        title (str): Title of the plot.
        ax (plt.Axes, optional): Matplotlib Axes object to plot on. If not provided, a new Axes object is created.

    Returns:
        plt.Axes: Matplotlib Axes object containing the plot.
    """
    # Read the shapefile
    shapefile = gpd.read_file(shapefile_path)

    # Open the raster
    raster = rxr.open_rasterio(raster_path, masked=True)

    # Crop the raster using the shapefile
    cropped_raster = raster.rio.clip(shapefile.geometry)

    # Set colormap
    cmap = cm.get_cmap('plasma')

    # Normalize the data to range between 0 and 1
    norm = colors.Normalize(vmin=0, vmax=1)

    # Create the plot
    if ax is None:
        fig, ax = plt.subplots()

    # Plot the cropped raster
    im = ax.imshow(cropped_raster.squeeze(), cmap=cmap, vmin=0, vmax=0.1)
    ax.set_title(title)
    ax.axis('off')

    # Create a ScalarMappable object for the colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Set an empty array to ensure correct mapping of values

    # Add colorbar
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    return ax

# def plot_cropped_raster(raster_path, shapefile_path, title, ax):
#     # Read the shapefile
#     shapefile = gpd.read_file(shapefile_path)

#     # Open the raster
#     raster = rxr.open_rasterio(raster_path, masked=True)

#     # Crop the raster using the shapefile
#     cropped_raster = raster.rio.clip(shapefile.geometry)

#     # Set colormap
#     cmap = cm.get_cmap('plasma')

#     # Normalize the data to range between 0 and 1
#     norm = colors.Normalize(vmin=0, vmax=1)

#     # Plot the cropped raster
#     im = ax.imshow(cropped_raster.squeeze(), cmap=cmap,  vmin = 0,vmax = 0.1)
#     ax.set_title(title)
#     ax.axis('off')

#     # Create a ScalarMappable object for the colorbar
#     sm = cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])  # Set an empty array to ensure correct mapping of values

#     # Add colorbar
#     cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)



def plot_class_raster(
    class_path: str,
    shapefile_path: str,
    title: str,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot the classified raster.

    Args:
        class_path (str): Path to the classified raster file.
        shapefile_path (str): Path to the shapefile.
        title (str): Title of the plot.
        ax (plt.Axes, optional): Matplotlib Axes object to plot on. If not provided, a new Axes object is created.

    Returns:
        plt.Axes: Matplotlib Axes object containing the plot.
    """
    # Read the shapefile
    shapefile = gpd.read_file(shapefile_path)

    # Open the raster
    raster = rxr.open_rasterio(class_path, masked=True)

    # Crop the raster using the shapefile
    cropped_raster = raster.rio.clip(shapefile.geometry)

    # Create custom colormap
    cmap_colors = ['green', 'tan']
    cmap = ListedColormap(cmap_colors)

    # Set normalization for the classified values
    norm = Normalize(vmin=0, vmax=1)

    # Create the plot
    if ax is None:
        fig, ax = plt.subplots()

    # Plot the cropped raster
    ax.imshow(cropped_raster.squeeze(), cmap=cmap)
    ax.set_title(title)
    ax.axis('off')

    # Create custom legend
    custom_lines = [Line2D([0], [0], color='#D4C27E', lw=6),
                    Line2D([0], [0], color='#22BF45', lw=6)]
    leg_names = ['Other', 'SAV']

    # Add legend within the Axes object
    ax.legend(custom_lines, leg_names, title='', loc='lower left', frameon=True)

    return ax



def plot_class_raster(class_path, shapefile_path, title, ax):
    # Read the shapefile
    shapefile = gpd.read_file(shapefile_path)

    # Open the raster
    raster = rxr.open_rasterio(class_path, masked=True)

    # Crop the raster using the shapefile
    cropped_raster = raster.rio.clip(shapefile.geometry)

    # Create custom colormap
    cmap_colors = ['green', 'tan']
    cmap = ListedColormap(cmap_colors)

    # Set normalization for the classified values
    norm = Normalize(vmin=0, vmax=1)

    # Plot the cropped raster
    #fig, ax1 = plt.subplots(figsize=(10, 10))
    ax.imshow(cropped_raster.squeeze(), cmap=cmap)
    ax.set_title(title)
    ax.axis('off')

    # Create custom legend
    custom_lines = [Line2D([0], [0], color='#D4C27E', lw=6),
                    Line2D([0], [0], color='#22BF45', lw=6)]
    leg_names = ['Other', 'SAV']

    # Add legend within ax1
    ax.legend(custom_lines, leg_names, title='', loc='lower left', frameon=True)

    #plt.show()
    
    
    
def calculate_pixel_change(folder_path: str, output_path: str) -> None:
    """
    Calculate the pixel change between consecutive GeoTIFF files in a folder and output the result as a new GeoTIFF.

    Args:
        folder_path (str): Path to the folder containing the GeoTIFF files.
        output_path (str): Path to save the output GeoTIFF file.

    Returns:
        None
    """
    # List all GeoTIFF files in the folder
    file_names = [file for file in os.listdir(folder_path) if file.endswith('.tif')]
    file_names.sort()  # Sort the file names to ensure sequential comparison

    # Read the first GeoTIFF file to get the dimensions
    with rasterio.open(os.path.join(folder_path, file_names[0])) as src:
        # Read the raster data as a numpy array
        previous_data = src.read(1)

        # Initialize an empty array to store the sum of pixel change
        change_array = np.zeros(previous_data.shape, dtype=np.int32)

    # Iterate over the remaining GeoTIFF files
    for file_name in file_names[1:]:
        file_path = os.path.join(folder_path, file_name)
        with rasterio.open(file_path) as src:
            # Read the raster data as a numpy array
            current_data = src.read(1)

            # Calculate the changes from 1 to 2
            changes = np.where((previous_data == 1) & (current_data == 2), 1, 0)

            # Accumulate the changes to the change array
            change_array += changes

            # Update the previous data for the next comparison
            previous_data = current_data

    # Output the change array as a new GeoTIFF
    with rasterio.open(os.path.join(folder_path, file_names[0])) as src:
        # Create a new raster with the same dimensions and metadata as the input
        profile = src.profile
        profile.update(dtype=rasterio.int32)

        # Write the change array to the output GeoTIFF
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(change_array, 1)    
            
def plot_cropped_change(raster_path: str, shapefile_path: str, title: str, ax: plt.Axes) -> None:
    """
    Plot a cropped change raster overlaid on a shapefile.

    Args:
        raster_path (str): Path to the change raster GeoTIFF file.
        shapefile_path (str): Path to the shapefile for cropping.
        title (str): Title of the plot.
        ax (plt.Axes): Matplotlib Axes object for plotting.

    Returns:
        None
    """
    # Read the shapefile
    shapefile = gpd.read_file(shapefile_path)

    # Open the raster using rioxarray
    raster = rxr.open_rasterio(raster_path, masked=True)

    # Crop the raster using the shapefile
    cropped_raster = raster.rio.clip(shapefile.geometry)
    min_value = np.nanmin(cropped_raster)
    max_value = np.nanmax(cropped_raster)

    # Set colormap
    cmap = cm.get_cmap('plasma')

    # Plot the cropped raster
    im = ax.imshow(cropped_raster.squeeze(), cmap=cmap, vmin=min_value, vmax=max_value, interpolation=None)
    ax.set_title(title)
    ax.axis('off')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)