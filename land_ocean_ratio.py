#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:35:24 2021

@author: huw
"""
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pytesseract
from itertools import repeat
from PIL import Image
from osgeo import gdal, osr
import cartopy.crs as ccrs
from shapely.geometry.polygon import Polygon
import shapely.vectorized

# Options: 'no mask' or 'yes mask'
use_mask = 'yes mask'

# Options: 'no' or 'yes'
save_table = 'yes'

georeference_images = 'yes'


# Search this directory for the image that will be processed
os.chdir(r'/home/huw/Dropbox/Sophie/SeaLevelChange/Images2')

image_names = os.listdir()
image_names_no_extension = [os.path.splitext(i)[0] for i in image_names]
image_names_no_extension.sort()
image_names.sort()


# %%


def map_extent(input_raster):
    """
    A method for providing the top righ, left, and bottom right, left
    coordinates of the input raster image.

    Parameters
    ----------
    input_raster : string
        Directory to the raster which should be in tiff format.

    Returns
    -------
    raster_extent : tuple
        the top left righ and bottom left right corner coordinates of
        the input raster.

    """

    gdal.UseExceptions()
    raster = gdal.Open(input_raster)
    raster_geotransform = raster.GetGeoTransform()
    raster_extent = (raster_geotransform[0],
                     raster_geotransform[0]
                     + raster.RasterXSize * raster_geotransform[1],
                     raster_geotransform[3]
                     + raster.RasterYSize * raster_geotransform[5],
                     raster_geotransform[3])

    return raster_extent


def georeferenced_images(png_image, tif_image):
    os.chdir(r'/home/huw/Dropbox/Sophie/SeaLevelChange/Images2')
    ds = gdal.Translate('temporary.tif', png_image)

    # Set spatial reference:
    sr = osr.SpatialReference()

    sr.ImportFromEPSG(4326)
    
    # Pixel coordinates in the pre-georeferenced image
    left_x = 427.59
    right_x = 763.168
    bottom_y = 736.604
    top_y = 568.527

    # Enter the ground control points (GCPs)
    #   Format: [map x-coordinate(longitude)], [map y-coordinate (latitude)], [elevation],
    #   [image column index(x)], [image row index (y)]
    gcps = [gdal.GCP(0, 30, 0, left_x, top_y),
            gdal.GCP(180, 30, 0, right_x, top_y),
            gdal.GCP(180, -60, 0, right_x, bottom_y),
            gdal.GCP(0, -60, 0, left_x, bottom_y)]

    ds.SetProjection(sr.ExportToWkt())
    wkt = ds.GetProjection()
    ds.SetGCPs(gcps, wkt)
    os.chdir(r'/home/huw/Dropbox/Sophie/SeaLevelChange/georeferenced')
    gdal.Warp(f"{tif_image}.tif", ds, dstSRS='EPSG:4326', format='gtiff')
    os.chdir(r'/home/huw/Dropbox/Sophie/SeaLevelChange/Images2')
    os.remove('temporary.tif')
    ds= None

if georeference_images == 'yes':
    for i, w in zip(image_names, image_names_no_extension):
        georeferenced_images(i, w)


# %%

# starts in the top left going clockwise finishing at top left (x, y).
# coordinates in decimal degrees.
irregular_study_area = Polygon([(98, 13.5),  
                               (125, 13.5),
                               (145, -3),
                               (145, -18),
                               (122, -18),
                               (98, -3),
                               (98, 13.5)])

# %%

def raster_to_array(input_raster):
    """
    Convert a raster tiff image to a numpy array. Input Requires the
    address to the tiff image.

    Parameters
    ----------
    input_raster : string
        Directory to the raster which should be in tiff format.

    Returns
    -------
    converted_array : numpy array
        A numpy array of the input raster.

    """

    raster = gdal.Open(input_raster)
    band = raster.GetRasterBand(1)
    converted_array = band.ReadAsArray()

    return converted_array


os.chdir(r'/home/huw/Dropbox/Sophie/SeaLevelChange/georeferenced')
georeferenced_images = os.listdir()
georeferenced_images.sort()
test = raster_to_array(georeferenced_images[0])
test_extent = map_extent(georeferenced_images[0])


x0, x1 = test_extent[0], test_extent[1]
y0, y1 = test_extent[2], test_extent[3]


def linear_interpolation_of_x_y(georeferenced_array, extent_minimum,
                                extent_maximum):
    """
    A rather cluncky method. The purpose is to create an array of
    longitude and latitude values that have the same length as the input
    georeferenced array. This method linearly interpolates the longitude
    and latitude values.

    Parameters
    ----------
    georeferenced_array : 2D Array
        DESCRIPTION.
    loc0 : float or integer
        The first extent value of the georeferenced array.
    loc1 : float or integer
        The second extent value of the georeferenced array.

    Returns
    -------
    interpolated_coordinates : 1D array
        Interpolated latitude or longitude values for the length of the
        georeferenced array.

    """
    # Extracts axis 1 (columns) from the input array.
    # This represents the longitude.
    if extent_minimum == x0:
        inn = georeferenced_array[0, :]

    #  Extracts axis 0 (rows) from the input array.
    # This represents the latitude.
    elif extent_minimum == y0:
        inn = georeferenced_array[:, 0]

    #
    linear_interpolation = [((i-0)*(extent_maximum-extent_minimum)/(
        (len(inn)-1)-0)+extent_minimum) for i, r in enumerate(inn)]

    # Claculates the difference between the value in front and the value
    # behind in the list
    difference = [y - x for x, y in zip(linear_interpolation,
                                        linear_interpolation[1:])]

    # Calculates the size of each array so to compare it to the size of the
    # input array.
    array_length = [np.size(np.arange(
        extent_minimum, extent_maximum, i)) for i in difference]

    # Select values that only match the longitude/latitude length then return
    # the first index in the list of matched values.
    # This list is a list of indexes that correspond to the index in the
    # variable difference.
    index_of_correct_value = [i for i, v in enumerate(
        array_length) if v == len(inn)][0]

    #
    interpolated_coordinates = np.arange(extent_minimum,
                                         extent_maximum,
                                         difference[index_of_correct_value])

    return interpolated_coordinates


x_longitude = linear_interpolation_of_x_y(test, x0, x1)
y_latitude = linear_interpolation_of_x_y(test, y0, y1)
xx_longitude, yy_longitude = np.meshgrid(x_longitude, y_latitude[::-1])


mask = shapely.vectorized.contains(irregular_study_area,
                                   xx_longitude,
                                   yy_longitude)





def mask_and_binarize(polygon_mask, area_of_interest_raster, threshold):

    raster_array = raster_to_array(area_of_interest_raster)
    
    # Pixels outside of the polygon are assigned nan values.
    masked = np.where(mask == True, raster_array, np.nan)

    binerized_array = np.where(masked >= threshold, 255, 0)

    box_top, box_bottom, box_left, box_right = 616, 649, 637, 681

    # Draw hollow rectangle with 2px border width on left and 1px for rest.
    # -9999 is a random value I chose. Easier to detect in image.
    binerized_array[box_top:box_bottom, box_left:box_left+2] = -9999
    binerized_array[box_top:box_bottom, box_right-1:box_right] = -9999
    binerized_array[box_top:box_top+1, box_left:box_right] = -9999
    binerized_array[box_bottom-1:box_bottom, box_left:box_right] = -9999

    # If pixels are not equal to -9999 keep the pixel value.
    # Pixels that are equal to -9999 are assigned 'nan'.
    binerized_array = np.where(binerized_array != -9999, binerized_array, np.nan)
    binerized_array = np.ma.array(binerized_array, mask=np.isnan(masked))

    return binerized_array


sample_mask = mask_and_binarize(mask, georeferenced_images[0], 150)




# %% Recreating the sea_land_ratio method
def sea_land_ratio_calculation(masked_array, box_perimeter_fill_value):
    
    cleaned_array = np.nan_to_num(masked_array,
                                      copy=True,
                                      nan=box_perimeter_fill_value,
                                      posinf=None,
                                      neginf=None)
        
    image_pixels = cleaned_array.count()
    image_nans = np.isnan(cleaned_array).sum()
    non_nan_pixels = image_pixels-image_nans
    land_pixels = np.count_nonzero(cleaned_array == 255)
    ocean_pixels = np.count_nonzero(cleaned_array == 0)
    
    if type(box_perimeter_fill_value) == float:
        land_percentage = round((land_pixels/non_nan_pixels)*100, 4)
        ocean_percentage = round((ocean_pixels/non_nan_pixels)*100, 4)
    
    elif type(box_perimeter_fill_value) == int:
        land_percentage = round((land_pixels/image_pixels)*100, 4)
        ocean_percentage = round((ocean_pixels/image_pixels)*100, 4)
                 
    land_ocean_ratio = round(land_percentage/ocean_percentage, 4)
    
    return land_percentage, ocean_percentage, land_ocean_ratio

# %% testing the sea_land_ratio definition 
os.chdir(r'/home/huw/Dropbox/Sophie/SeaLevelChange/georeferenced')
# apply_box_perimeter_mask = 'yes'
box_perimeter_fill_value = np.nan

cleaned_array = np.nan_to_num(sample_mask,
                                  copy=True,
                                  nan=box_perimeter_fill_value,
                                  posinf=None,
                                  neginf=None)
    
image_pixels = cleaned_array.count()
image_nans = np.isnan(cleaned_array).sum()
non_nan_pixels = image_pixels-image_nans
land_pixels = np.count_nonzero(cleaned_array == 255)
ocean_pixels = np.count_nonzero(cleaned_array == 0)

if type(box_perimeter_fill_value) == float:
    land_percentage = round((land_pixels/non_nan_pixels)*100, 4)
    ocean_percentage = round((ocean_pixels/non_nan_pixels)*100, 4)

elif type(box_perimeter_fill_value) == int:
    land_percentage = round((land_pixels/image_pixels)*100, 4)
    ocean_percentage = round((ocean_pixels/image_pixels)*100, 4)

# land_percentage = round((land_pixels/image_pixels)*100, 4)
# ocean_percentage = round((ocean_pixels/image_pixels)*100, 4)
    
    
land_ocean_ratio = round(land_percentage/ocean_percentage, 10)
print(f'{box_perimeter_fill_value}', land_ocean_ratio) 
 
map_projection = ccrs.PlateCarree()

fig, axes = plt.subplots(nrows=2,
                         ncols=1,
                         figsize=(15, 5), subplot_kw={'projection': map_projection})

from matplotlib import colors
cmap = colors.ListedColormap(['dodgerblue', 'tan'])

import cartopy.feature as cfeature

georeferenced_images = os.listdir()
georeferenced_images.sort()
img = Image.open(georeferenced_images[1])

ax = axes[0]
ax.imshow(img,
          origin='upper',
          extent=map_extent(georeferenced_images[0]),
          cmap=cmap)

continents = cfeature.NaturalEarthFeature(category='physical',
                                          name='land',
                                          scale='50m',
                                          edgecolor='face')

ax.add_feature(continents,
                 facecolor='none',
                 edgecolor='grey',
                 lw=1)

ax.set_extent([90, 155, -25, 20], crs=map_projection)

ax = axes[1]
ax.imshow(cleaned_array,
          origin='upper',
          extent=map_extent(georeferenced_images[0]),
          cmap=cmap)

ax.add_feature(continents,
                 facecolor='none',
                 edgecolor='grey',
                 lw=1)

ax.set_extent([90, 155, -25, 20], crs=map_projection)
# ax.set_extent([91, 150.59, -20.35, 20.61], crs=map_projection)

unmodified_image = raster_array = raster_to_array(georeferenced_images[0])


# %%
def multi_process_mask_and_binarize(polygon_mask,
                                    area_of_interest_raster,
                                    threshold):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        processed_image = executor.map(mask_and_binarize,
                                       repeat(polygon_mask),
                                       area_of_interest_raster,
                                       repeat(threshold))
        
        return processed_image
    
processed_image = multi_process_mask_and_binarize(mask,
                                                  georeferenced_images,
                                                  150)

sea_land_ratio = [sea_land_ratio_calculation(i, 0) for i in processed_image]
# sea_land_ratio = [sea_land_ratio_calculation(i, 'yes mask') for i in processed_image]

# %%
os.chdir(r'/home/huw/Dropbox/Sophie/SeaLevelChange/Images2')

def CropImage(image, left, top, right, bottom):
    open_image = Image.open(image)
    # Cropped image of above dimension
    # (It will not change orginal image)
    cropped_image = open_image.crop((left,
                                     top,
                                     right,
                                     bottom))

    return cropped_image


def MultiProcessCrop(names, im_left, im_top, im_right, im_bottom):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        processed_image = executor.map(CropImage,
                                       names,
                                       repeat(im_left),
                                       repeat(im_top),
                                       repeat(im_right),
                                       repeat(im_bottom))

        return processed_image


cropped_year = MultiProcessCrop(image_names, 300, 0, 395, 50)
cropped_eustatic = MultiProcessCrop(image_names, 336, 45, 377, 62)

threshold_value = 150


# %%

model_year = (float(pytesseract.image_to_string(i)) for i in cropped_year)

# %%


def CleanEustaticNumbersInImages(cropped_eustatic_images):

    test_list = []
    index_missing_numbers = []

    for i, r in enumerate(cropped_eustatic_images):
        try:
            test_list.append(float(pytesseract.image_to_string(
                r, config='--psm 6')))
        except ValueError:
            test_list.append(-9999)
            index_missing_numbers.append(i)

    # images_missing_eustatic = [image_names[i] for i in index_missing_numbers]

    replacement_eustatic = [7.176,
                            4.585,
                            13.111,
                            50.435,
                            36.167,
                            6.645,
                            6.253,
                            7.721,
                            7.721,
                            9.185,
                            9.185,
                            7.512,
                            7.512,
                            36.945,
                            36.945,
                            43.045]

    for (i, r) in zip(index_missing_numbers, replacement_eustatic):
        test_list[i] = r

    index_wrong_numbers = [76, 138, 196, 197, 400, 509, 510]

    replacement_numbers = [37.173, 31.124, 31.144, 31.144, 5.291, 71.277,
                           71.277]

    for (i, r) in zip(index_wrong_numbers, replacement_numbers):
        test_list[i] = r

    test_list_divide = [x/1000 if x > 100 else x for x in test_list]

    cleaned_eustatic = (-x if x > 0 else x for x in test_list_divide)

    return cleaned_eustatic


eustatic_clean = CleanEustaticNumbersInImages(cropped_eustatic)


def DataToPandasDataFrame(name, year, eustatic, sea_land_and_ratio):
    variable_locations = [[i]*len(name) for i in [0, 1, 2]]

    variables_to_insert = [name, year, eustatic]

    image_index = [np.arange(0, len(name), 1)]*3

    sea_land_ratio_list = [list(i) for i in sea_land_and_ratio]

    for i, r, t in zip(image_index, variable_locations, variables_to_insert):
        for w, c, b in zip(i, r, t):
            sea_land_ratio_list[w].insert(c, b)

    df = pd.DataFrame(sea_land_ratio_list,
                      columns=[
                          'Image',
                          'Years ago [kyr]',
                          'Eustatic [m]',
                          'Land [%]',
                          'Ocean [%]',
                          'Ratio'])

    return df


summary_table = DataToPandasDataFrame(image_names_no_extension,
                                      model_year,
                                      eustatic_clean,
                                      sea_land_ratio)


# %%

"""Saving tables and figures"""


# Saving figure and table in a different directory

if save_table == 'yes' and use_mask == 'yes mask':
    os.chdir(r'/home/huw/Dropbox/Sophie/SeaLevelChange/Excel')
    summary_table.to_excel(f'RSL_with_polgon_AOI_Ocean.xlsx',
                           index=False)
    print(f'Table saved: {os.getcwd()}/RSL_with_mask_small_AOI.xlsx ')

elif save_table == 'yes' and use_mask == 'no mask':
    os.chdir(r'/home/huw/Dropbox/Sophie/SeaLevelChange/Excel')
    summary_table.to_excel(f'Sea_level_threshold{threshold_value}.xlsx',
                           index=False)


os.chdir(r'/home/huw/Dropbox/Sophie/SeaLevelChange/Excel')

