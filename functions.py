import numpy as np, pandas as pd, geopandas as gpd, sklearn.metrics as metrics
import geemap, ee , rasterio, sankee, warnings, pickle, os, glob, subprocess, json, fiona
from matplotlib.lines import Line2D
from rasterio import features
from geopandas import GeoDataFrame
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from rasterio.mask import mask
from osgeo import gdal, osr, gdalconst
import numpy.ma as ma
import rasterio, pickle
import matplotlib as mpl
import matplotlib.pyplot as plt, plotly.express as px
from sklearn import ensemble
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from matplotlib_scalebar.scalebar import ScaleBar
from scipy import stats
import seaborn as sns
import statsmodels
import glob
from glob import glob
import rasterio.plot as rplot
from scipy.cluster.hierarchy import dendrogram, linkage
from shapely.geometry import Polygon, Point, box
from rasterio.plot import show
import rioxarray as rxr
from rio_cogeo.profiles import cog_profiles
from rasterio.merge import merge
from rasterstats import zonal_stats
from geocube.api.core import make_geocube
from typing import List
from typing import Union
warnings.filterwarnings('ignore')


######################## factor image   ########################################
##################################################################################
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

######################## pandas normalise   ########################################
##################################################################################
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

######################## L8 cloud mask   ########################################
##################################################################################
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
        S2S: The input image with sunglint.
        randomPoints: Geometry representing random points for linear fit reduction.
        glint_geo: Geometry to clip the image for slope calculation.

    Returns:
        The image with sunglint removed.

    Raises:
        ee.EEException: If an error occurs during Earth Engine computation.
    """
    # Band selection
    B2 = image.select(['SR_B5', 'SR_B2'])
    B3 = image.select(['SR_B5', 'SR_B3'])
    B4 = image.select(['SR_B5', 'SR_B4'])
    B5 = image.select(['SR_B5'])

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
    min_B8 = ee.Image.constant(image.select('SR_B5').reduceRegion(
        ee.Reducer.min(),
        geometry=glint_geo,
        scale=30
    ).get('SR_B5')).rename('min_B5')

    # Create glint factors image
    glint_factors = ee.Image([slope_B2, slope_B3, slope_B4, min_B8])
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

def get_LS8_image(sites: ee.Geometry, start_date: str, end_date: str) -> ee.Image:
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
        .filterMetadata('CLOUD_COVER', 'less_than', 5)\
        .filterDate(start_date, end_date)\
        .map(fmask)

    # Compute median and clip to the region of interest
    median_image = filtered_collection.median().clip(sites)
    
#     don = sites.filter(ee.Filter.eq('Location', 'Dongara'))
#     cliff = sites.filter(ee.Filter.eq('Location', 'Cliff Head'))
#     tr = sites.filter(ee.Filter.eq('Location', 'Two Rocks'))
#     jur = sites.filter(ee.Filter.eq('Location', 'Jurien Bay'))
#     cerv = sites.filter(ee.Filter.eq('Location', 'Cervantes'))
#     lanc = sites.filter(ee.Filter.eq('Location', 'Lancelin'))
    
    
#     de_don = remove_sunglint_L8(image = median_image, glint_geo = don)
#     de_cliff = remove_sunglint_L8(image = median_image, glint_geo = cliff)
#     de_tr = remove_sunglint_L8(image = median_image, glint_geo = tr)
#     de_jur = remove_sunglint_L8(image = median_image, glint_geo = jur)
#     de_cerv = remove_sunglint_L8(image = median_image, glint_geo = cerv)
#     de_lance = remove_sunglint_L8(image = median_image, glint_geo = lanc)
    de_image = remove_sunglint_L8(median_image, glint_geo = sites)
    

    return de_image



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


