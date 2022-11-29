#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 16:57:51 2020

@author: huw
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.signal import find_peaks

# Change the working directory to where the excel files are located
os.chdir(r'/home/huw/Dropbox/Sophie/SeaLevelChange/Excel')

sea_level_data = pd.read_excel('RSL_with_polgon_AOI_Land.xlsx')  # Polygon
# sea_level_data = pd.read_excel('RSL_with_mask_small_AOI.xlsx')  # Small Rectangle AOI
# sea_level_data = pd.read_excel('Sea_level_threshold150_mask.xlsx')  # Large AOI

# Values represent the column number in the numpy array.
# 0=Image, 1=Years ago[kyr], 2=Eustatic[m], 3=Land[%], 4=Ocean[%], 5=Ratio.
data_numpy = sea_level_data.to_numpy()


def remove_duplicate_rows(numpy_data):
    """
    Removes duplicate rows of data that share the same year.

    Parameters
    ----------
    numpy_data : ndarray object
        The multidimensional array to have duplicate rows removed.

    Returns
    -------
    eustatic : array
        Relative sea level values with deuplicate rows removed.
    ratio : array
        Ratio values with duplicate ows removed.
    year : array
        Year values with duplicate rows removed.

    """

    year = data_numpy[:, 1]

    # [1] selects the second return from np.unique. [::-1] orders the return
    # so that lowest value at the top.
    data_numpy_cleaned = data_numpy[np.unique(
        year, return_index=True)[1][::-1]]

    ratio = data_numpy_cleaned[:, 5]
    year = data_numpy_cleaned[:, 1]
    eustatic = data_numpy_cleaned[:, 2]

    return eustatic, ratio, year


# '[0]' returns the second element of the method.'[1]' returns the
# second element of the method etc.
eustatic = remove_duplicate_rows(data_numpy)[0]
ratio = remove_duplicate_rows(data_numpy)[1]
year = remove_duplicate_rows(data_numpy)[2]


def derive_slope(rise, run):
    """
    Calculates the slope derivative of the input data by calculating the
    rise/run.

    Parameters
    ----------
    data1_rise : 1D array
        DESCRIPTION.
    data2_run : 1D array
        DESCRIPTION.

    Returns
    -------
    derivative : 1D array
        The derivative of thee rise and run datasets. 1st derivative (slope)
        of the ratio/eustatic

    """
    difference_run = np.diff(run)
    difference_rise = np.diff(rise)
    derivative = difference_rise/difference_run

    return derivative


derivative_ratio_eustatic = derive_slope(ratio, eustatic)


# %%

def find_peaks_and_troughs(eustatic_data):
    """
    Finds the index locations of peaks and troughs in the array. This is
    useful to seperate the eustatic, land, and ocean into rise and fall
    categories.

    Parameters
    ----------
    eustatic_data : 1D array
        The array to be processed.

    Returns
    -------
    index_of_peaks_and_troughs : array of int64
        Index locations of the apex of troughs and peaks.
    peaks : array of int64
        Index locations of the apex of peaks only.
    troughs : array of int64
        Index locations of the apex of troughs only.

    """

    peaks, _ = find_peaks(x=eustatic_data, distance=1)
    negative = [i*-1 for i in eustatic_data]
    troughs, _ = find_peaks(negative, distance=1)

    # The scipy module,'find_peaks', doesn't recognise the first and last
    # element in the wave-form therefore I have added  the first '[0]' and
    # last '[len(eustatic_data)-1]' indicies manually using 'np.concatenate'.
    index_of_peaks_and_troughs = np.sort(
        np.concatenate((
            peaks, troughs, [0], [len(eustatic_data)-1])))

    return index_of_peaks_and_troughs, peaks, troughs


# '[0]' returns the second element of the method.'[1]' returns the
# second element of the method etc.
eustatic_peaks_and_troughs_index = find_peaks_and_troughs(eustatic)[0]
eustatic_peaks_index = find_peaks_and_troughs(eustatic)[1]
eustatic_troughs_index = find_peaks_and_troughs(eustatic)[2]


def SplitIntoRiseAndFall(data, peak_and_trough_indices):
    """Splits the data into rise and fall based on the index location of peaks
    and troughs. Note that every rise line will start at the same index as the
    last fall index i.e. there is overlap with the index located at the apex
    of the peaks and troughs."""

    peaks_and_troughs_length = np.arange(2,
                                         len(peak_and_trough_indices)+1,
                                         1)
    # 0 represents even indices.
    fall = [data[peak_and_trough_indices[-2+i]:
                 peak_and_trough_indices[-1+i]+1]
            for i in peaks_and_troughs_length if i % 2 == 0]

    # 1 represents odd indices.
    rise = [data[peak_and_trough_indices[-2+i]:
                 peak_and_trough_indices[-1+i]+1]
            for i in peaks_and_troughs_length if i % 2 == 1]

    return fall, rise


eustatic_fall = SplitIntoRiseAndFall(eustatic,
                                     eustatic_peaks_and_troughs_index)[0]

eustatic_rise = SplitIntoRiseAndFall(eustatic,
                                     eustatic_peaks_and_troughs_index)[1]

year_fall = SplitIntoRiseAndFall(year,
                                 eustatic_peaks_and_troughs_index)[0]

year_rise = SplitIntoRiseAndFall(year,
                                 eustatic_peaks_and_troughs_index)[1]


def SplitIntoRiseAndFallMod(data, peak_and_trough_indices):
    """Splits the data into rise and fall based on the index location of peaks
    and troughs. Each rising curve starts one index after the index of the
    lowest falling curve point. Likewise the falling curve starts one index
    after the highest rising curve value. This means there is no overlap
    between the start and end of each falling and rising curve. This resulted
    in the first falling curve being ignored. Therefore, I included an 'if'
    statement that doesn't modify the first falling curve."""

    peaks_and_troughs_length = np.arange(2,
                                         len(peak_and_trough_indices)+1,
                                         1)

    # I avoided a list comprehension to make the code more readable with the
    # 'if' and 'elif statements.
    fall = []
    for i in peaks_and_troughs_length:
        if i == 2:
            fall.append(
                data[peak_and_trough_indices[-2+i]:
                     peak_and_trough_indices[-1+i]+1])

        elif i % 2 == 0:
            fall.append(
               data[peak_and_trough_indices[-2+i]+1:
                    peak_and_trough_indices[-1+i]+1])

    # The addition of '+1' after '[-2+i] avoids overlaping of first values of
    # the rise curve. Same as the 'elif' statement above.
    rise = [data[peak_and_trough_indices[-2+i]+1:
                 peak_and_trough_indices[-1+i]+1]
            for i in peaks_and_troughs_length if i % 2 == 1]

    return fall, rise


eustatic_fall1 = SplitIntoRiseAndFallMod(eustatic,
                                         eustatic_peaks_and_troughs_index)[0]
eustatic_rise1 = SplitIntoRiseAndFallMod(eustatic,
                                         eustatic_peaks_and_troughs_index)[1]

year_fall1 = SplitIntoRiseAndFallMod(year,
                                     eustatic_peaks_and_troughs_index)[0]
year_rise1 = SplitIntoRiseAndFallMod(year,
                                     eustatic_peaks_and_troughs_index)[1]

ratio_fall = SplitIntoRiseAndFallMod(ratio,
                                     eustatic_peaks_and_troughs_index)[0]
ratio_rise = SplitIntoRiseAndFallMod(ratio,
                                     eustatic_peaks_and_troughs_index)[1]


def FilterLittleFallsAndRises(split_data, threshold, split_name):
    """Removes any falls and rises that are below the threshold.
    The threshold is based on the number of data point. Small rises and
    falls have fewer data points and can be removed using a filter. The
    filter is based on the size i.e. number of datapoints in each fall and
    rise curve"""

    if split_name == 'fall':
        fall_filter = [i for i in split_data if np.size(i) >= threshold]

        return fall_filter

    elif split_name == 'rise':
        rise_filter = [i for i in split_data if np.size(i) >= threshold]

        return rise_filter


eustatic_fall_filter = FilterLittleFallsAndRises(eustatic_fall1, 10, 'fall') ###
eustatic_rise_filter = FilterLittleFallsAndRises(eustatic_rise1, 5, 'rise') ###

eustatic_fall_year_filter = FilterLittleFallsAndRises(year_fall1, 10, 'fall') ###
eustatic_rise_year_filter = FilterLittleFallsAndRises(year_rise1, 5, 'rise') ###

eustatic_fall_ratio_filter = FilterLittleFallsAndRises(ratio_fall, 10, 'fall')
eustatic_rise_ratio_filter = FilterLittleFallsAndRises(ratio_rise, 5, 'rise')


def collapse_list_of_arrays(list_of_arrays):
    concatenated_array = np.concatenate(list_of_arrays, axis=None)
    return concatenated_array

eustatic_rise_year_filter_concatenated = collapse_list_of_arrays(eustatic_rise_year_filter)
eustatic_rise_filter_concatenated = collapse_list_of_arrays(eustatic_rise_filter)
eustatic_rise_ratio_filter_concatenated = collapse_list_of_arrays(eustatic_rise_ratio_filter)
eustatic_fall_year_filter_concatenated = collapse_list_of_arrays(eustatic_fall_year_filter)
eustatic_fall_filter_concatenated = collapse_list_of_arrays(eustatic_fall_filter)
eustatic_fall_ratio_filter_concatenated = collapse_list_of_arrays(eustatic_fall_ratio_filter)

array_list = {'RSL rise year [kyr]':eustatic_rise_year_filter_concatenated,
              'RSL rise [m]':eustatic_rise_filter_concatenated,
              'Ratio rise':eustatic_rise_ratio_filter_concatenated,
             'RSL fall year [kyr]':eustatic_fall_year_filter_concatenated,
            'RSL fall [m]':eustatic_fall_filter_concatenated,
            'Ratio fall':eustatic_fall_ratio_filter_concatenated }


summary_table = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in array_list.items() ]))


# %%
"""Plotting figures"""

fig, axes = plt.subplots(nrows=3,
                         ncols=1,
                         sharex=False,
                         sharey=False,
                         figsize=(10, 15))  # width x height

# First plot
ax = axes[0]

# ax.plot(year, eustatic)

ax.scatter(year[eustatic_peaks_index], eustatic[eustatic_peaks_index])
ax.scatter(year[eustatic_troughs_index], eustatic[eustatic_troughs_index])

# Annotate plot with index values where there's a peak and trough.
# for numbers in eustatic_peaks_index:
#     ax.annotate(numbers,
#                 xy=[year[numbers], eustatic[numbers]],
#                 xytext=[year[numbers], eustatic[numbers]+1.5],
#                 ha='center')

# for numbers in eustatic_troughs_index:
#     ax.annotate(numbers,
#                 xy=[year[numbers], eustatic[numbers]],
#                 xytext=[year[numbers], eustatic[numbers]-3],
#                 ha='center')

for i in (eustatic_peaks_index):
    ax.annotate(year[i],
                xy=[year[i], eustatic[i]],
                xytext=[year[i], eustatic[i]+2.5],
                ha='left',
                color='C0')

for i in (eustatic_troughs_index):
    ax.annotate(year[i],
                xy=[year[i], eustatic[i]],
                xytext=[year[i], eustatic[i]-5],
                ha='right',
                color='C1')


# for i, r in zip(year_fall_label, :
#     ax.annotate(i,
#                 xy=[year_fall_label, eustatic_troughs_index],
#                 xytext=[year_fall_label, eustatic_troughs_index],
#                 ha='center')

# The section 'label='Rise' if i == 0 else ""' allows me to have one item in
# the legend without duplicate items.
for i, (r, t) in enumerate(zip(year_rise1, eustatic_rise1)):
    ax.plot(r, t, color='C1', label='Rise' if i == 0 else "")

for i, (r, t) in enumerate(zip(year_fall1, eustatic_fall1)):
    ax.plot(r, t, color='C0', label='Fall' if i == 0 else "")

ax.set_ylabel('Eustatic [m]')
ax.set_xlabel('Years [kyr]')

ax.annotate('Before removing little rises and falls',
            [1270, -70],
            ha='right',
            va='top',
            size=12,
            color='black')

ax.legend(prop={'size': 12})


# Second plot
ax = axes[1]

for i, (r, t) in enumerate(zip(
        eustatic_rise_year_filter, eustatic_rise_filter)):
    ax.plot(r, t, color='C1', label='Rise' if i == 0 else "")

for i, (r, t) in enumerate(zip(
        eustatic_fall_year_filter, eustatic_fall_filter)):
    ax.plot(r, t, color='C0', label='Fall' if i == 0 else "")

ax.set_ylabel('Eustatic [m]')
ax.set_xlabel('Years [kyr]')

ax.annotate('After removing little rises and falls',
            [1270, -70],
            ha='right',
            va='top',
            size=12,
            color='black')

ax.legend(prop={'size': 12})


# Third plot
ax = axes[2]

# 'C1' is matplotlib's default orange colour.
for i, (r, t) in enumerate(zip(
        eustatic_rise_ratio_filter, eustatic_rise_filter)):
    ax.scatter(r, t, color='C1', label='Rise' if i == 0 else "")


# 'C0' is matplotlib's default blue colour.
for i, (r, t) in enumerate(zip(
        eustatic_fall_ratio_filter, eustatic_fall_filter)):
    ax.scatter(r, t, color='C0', alpha=0.4, label='Fall' if i == 0 else "")

ax.set_ylabel('Eustatic [m]')
ax.set_xlabel('Ratio [Land/Ocean]')

ax.annotate('Using data after removing little rises and falls',
            [0.247, -70],
            ha='left',
            va='top',
            size=12,
            color='black')

ax.legend(prop={'size': 12})


fig, axes = plt.subplots(nrows=2,
                          ncols=2,
                          sharex=False,
                          sharey=False,
                          figsize=(18, 10))  # width x height

from matplotlib.collections import LineCollection
# import matplotlib.colors as colors



ax = axes[0, 0]
# ax.plot(year, eustatic)
ax.set_ylabel('Eustatic Sea Level [m]')
ax.set_xlabel('Years [kyr]')

# Convert array of objects to an array of floats.
x = np.array([i[0] if isinstance(i, np.ndarray) else i for i in year])
y = np.array([i[0] if isinstance(i, np.ndarray) else i for i in eustatic])
z = np.array([i[0] if isinstance(i, np.ndarray) else i for i in ratio])

# Create a set of line segments so that we can color them individually
# This creates the points as a N x 1 x 2 array so that we can stack points
# together easily to get the segments. The segments array for line collection
# needs to be (numlines) x (points per line) x 2 (for x and y)
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)


# Create a continuous norm to map from data points to colors
norm = plt.Normalize(z.min(), z.max())
lc = LineCollection(segments, cmap='viridis', norm=norm)
# Set the values used for colormapping
lc.set_array(z)
lc.set_linewidth(2)
line = ax.add_collection(lc)
fig.colorbar(line, ax=ax)

ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())


# Sort the points by density, so that the densest points are plotted last.
idx = np.argsort(year)
x1, y1, z1 = eustatic[idx], ratio[idx], year[idx]

ax = axes[0, 1]
sc = ax.scatter(x1, y1, c=z1, cmap='viridis')
ax.set_ylabel('Ratio')
ax.set_xlabel('Eustatic [m]')
fig.colorbar(sc, ax=ax)


ax = axes[1, 0]
# [1:] Select items from array starting at index 1.
ax.scatter(eustatic[1:], derivative_ratio_eustatic)
ax.axhline(0, color='black', ls='--')
ax.set_ylabel('Derivative')
ax.set_xlabel('Ratio')

ax = axes[1, 1]
ax.plot(year, eustatic)
ax.set_ylabel('Eustatic [m]')
ax.set_xlabel('Years [kyr]')

ax.scatter(year[eustatic_peaks_index], eustatic[eustatic_peaks_index])
ax.scatter(year[eustatic_troughs_index], eustatic[eustatic_troughs_index])


# %%
"""Saving tables and figures"""


def save_table(yes_or_no, folder_directory, pandas_table, save_table_name):
    """
    Converts a pandas table to excel and saves in a folder.
    Prints the directory into which the excel table is saved.

    Parameters
    ----------
    yes_or_no : string
        Yes for save no for not.
    folder_directory : string
        Full directory to the folder where you want to save the table.
    pandas_table : dataframe
        The pandas dataframe that you want to save.
    save_table_name : string
        Name of the saved excel file.

    Returns
    -------
    None.

    """
    if yes_or_no == 'yes':
        os.chdir(fr'{folder_directory}')
        pandas_table.to_excel(f'{save_table_name}.xlsx', index=False)
        print(f'Table saved: {os.getcwd()}/{save_table_name}.xlsx')


def save_figure(yes_or_no, folder_directory, save_figure_name):
    """
    Save the current figure in the plot object as a png file at 300 dpi.
    Prints the directory into which the figure is saved.

    Parameters
    ----------
    yes_or_no : string
        Yes for save no for not.
    folder_directory : string
        Full directory to the folder where you want to save the figure.
    save_figure_name : string
        Name of the saved figure file.

    Returns
    -------
    None.

    """
    if yes_or_no == 'yes':
        os.chdir(fr'{folder_directory}')
        plt.savefig(f'{save_figure_name}.png', dpi=300, bbox_inches='tight')
        print(f'Figure saved: {os.getcwd()}/{save_figure_name}.png')


save_table('yes',
           '/home/huw/Dropbox/Sophie/SeaLevelChange/Excel',
           summary_table,
           'Rise_and_Fall_Land')

save_figure('no',
            '/home/huw/Dropbox/Sophie/SeaLevelChange/Figures',
            'Rise_and_Fall_small_AOI1')
