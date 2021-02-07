import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wget
from sklearn.preprocessing import MinMaxScaler


def download_csv(url, save_location):
    """Downloads a csv file from the specified url to the specified save location.

    Args:
        url (str): url to download csv file.
        save_location (str): absolute path to save the downloaded file.
    """
    if os.path.exists(save_location):
        os.remove(save_location)
    wget.download(url, save_location)


def get_date_list(start="2020-01-23", end=datetime.now().strftime("%Y-%m-%d")):
    """Generates a list of dates in the format YYYY-MM-DD between the specified start and end dates.

    Args:
        start (str, optional): Start date in the format 'YYYY-MM-DD'. Defaults to '2020-01-23'.
        end (str, optional): End date in the format 'YYYY-MM-DD'. Defaults to the current day.

    Returns:
        [str]: Numpy array of dates between the start and end date. Formatted as 'YYYY-MM-DD'.
    """
    return pd.date_range(start=start, end=end).strftime("%Y-%m-%d").to_numpy()


def process_csv(save_location):
    """Retrives a list of dates and confimed cases from a downloaded csv file.

    Args:
        save_location (str): Absolute path of the csv file location.

    Returns:
        [str], [int]: A list of unique dates representing the dates where there were confirmed cases, the number of confirmed cases for each given date.
    """
    cr = csv.reader(open(save_location, "r"))
    next(cr)
    formatted = []
    for row in cr:
        if row[2] != "":
            formatted.append(row[2])
    np_array = np.array(formatted)
    unique_elements, counts_elements = np.unique(np_array, return_counts=True)
    return unique_elements, counts_elements


coordinates = {}


def process_csv_locations(save_location, coordinates=coordinates):
    """Retrives a list of dates and confimed cases from a downloaded csv file on a per location basis.

    Args:
        save_location (str): Absolute path of the csv file location.

    Returns:
        [str], [int]: A list of unique dates representing the dates where there were confirmed cases, the number of confirmed cases for each given date.
    """
    cr = csv.reader(open(save_location, "r"))
    next(cr)
    # Pull relevant data from the CSV in tuple form
    formatted = {}
    for row in cr:
        if row[13] in formatted: # If the location already exists in the dict
            formatted[row[13]].append(row[1])
        else: # If the location does not exist in the dict
            formatted[row[13]] = [row[1]]
            coordinates[row[13]] = {"x": row[17], "y": row[16]}

    locations_dict = {}
    for location in formatted:
        np_array = np.array(formatted[location])
        unique, counts = np.unique(np_array, return_counts=True)
        locations_dict[location] = [unique, counts]

    return locations_dict


def interpolate_cases(unique, counts, zeros=False, start=None, end=None):
    """Interpolates number of confirmed cases for dates that did not have a recorded number of confirmed cases.

    Args:
        unique ([str]): List of dates having number of confirmed COVID-19 cases.
        counts ([int]): List of corresponding counts of cases for the given date.

    Returns:
        [[str], [int]]: Complete list of dates with corresponding case numbers.
    """
    if not end:
        end = str(unique[-1])
    if not start:
        start = "2020-01-23"
    full_date_list = get_date_list(start=start, end=end)
    complete_date_array = [[], []]

    if zeros:
        for date in full_date_list:
            try:
                complete_date_array[0].append(date)
                complete_date_array[1].append(
                    int(counts[np.where(unique == date)]))
            except:
                complete_date_array[1].append(0)
                continue

    else:
        for date in full_date_list:
            try:
                complete_date_array[0].append(date)
                complete_date_array[1].append(
                    int(counts[np.where(unique == date)]))
            except:
                complete_date_array[1].append(np.nan)
                continue

        s = pd.Series(complete_date_array[1])
        complete_date_array[1] = s.interpolate(
            limit_direction="backward").to_list()

    return complete_date_array


def rolling_mean(data, window):
    """
    Args:
        data (list): List of numbers to compute the rolling mean
        window (int): Size of the rolling mean window (note that for the first
            few entries in data where the number of elements before the current
            element is less than window, only those existing elements shall be
            considered).

    Returns:
        list: A list of the rolling mean numbers
    """
    if len(data) <= 0:
        return []

    cumsums, means = [data[0]], [data[0]]
    for x in data[1:]:
        new_sum = cumsums[-1] + x
        cumsums.append(new_sum)

        new_mean = new_sum
        if len(cumsums) <= window:
            new_mean /= len(cumsums)
        else:
            new_mean -= cumsums[-(window + 1)]
            new_mean /= window
        means.append(new_mean)

    return means


def rate_of_change(data, window=7, average=False):
    """Calculates the rate of change per iteration of a given input list.
        Can specify whether the list is a rolling average or not.

    Args:
        data ([type]): List of numbers to compute the rate of change per index.
        window ([type], optional): Window to use for calculating the rolling
            average, if required. Defaults to 7.
        average (bool, optional): If set to False, the data will have its
            rolling average computed before calculating the rate of change.
            Defaults to False.

    Returns:
        [list]: List of rates of change based on previous value
    """
    if not average:
        # calculate rolling average
        data = rolling_mean(data, window)

    def avg(lst): return (sum(lst) / len(lst)) if len(lst) > 0 else 0

    diff = [0] * (window + 1)
    for i, _ in enumerate(data[: -(window + 1)]):
        prev_window = data[i: i + window]
        prev_window_avg = avg(prev_window)

        if prev_window_avg:
            change = (data[i + window + 1] / prev_window_avg) * 100
        else:
            change = 0.0
        diff.append(change)

    return diff


# def plot(unique_elements, counts_elements):
#     """Simple plot for testing preprocessing.

#     Args:
#         unique_elements ([str]): String list of dates.
#         counts_elements ([int]): Int list of confirmed COVID-19 cases.
#     """
#     plt.plot(unique_elements, counts_elements)
#     plt.show()


# def main():
#     url = "https://data.ontario.ca/dataset/f4112442-bdc8-45d2-be3c-12efae72fb27/resource/455fd63b-603d-4608-8216-7d8647f43350/download/conposcovidloc.csv"
#     save_location = os.path.join(os.getcwd(), "data/conposcovidloc.csv")

#     download_csv(url, save_location)

#     each_location = False
#     if not each_location:
#         # perform for all of ontario
#         unique, counts = process_csv(save_location)
#         data_array = interpolate_cases(unique, counts, zeros=True)
#         # demonstrate rolling mean
#         plt.close()
#         plt.plot(data_array[0], data_array[1], label="counts")
#         plt.plot(
#             data_array[0], rolling_mean(data_array[1], window=7), label="means",
#         )
#         plt.plot(data_array[0], rate_of_change(
#             data_array[1], window=7, average=False))
#         plt.title("rolling mean (window=7)")
#         plt.legend()
#         plt.show()

#         print()
#         # print(np.array(data_array[1]))
#         scaler = create_scaler()
#         normalized_data = normalize_data(np.array(data_array[1]), scaler)
#         train_window = 7
#         inout_seq = create_tensors(normalized_data, train_window)
#         # print(inout_seq)
#     else:
#         # perform for distinct locations
#         locations_dict = process_csv_locations(save_location)
#         interpolated_dict = {}
#         inout_locations = {}
#         for location in locations_dict:
#             unique = locations_dict[location][0]
#             counts = locations_dict[location][1]
#             interpolated_dict[location] = interpolate_cases(
#                 unique, counts, zeros=True)
#             plt.close()
#             plt.plot(
#                 interpolated_dict[location][0],
#                 interpolated_dict[location][1],
#                 label="counts",
#             )
#             plt.plot(
#                 interpolated_dict[location][0],
#                 rolling_mean(interpolated_dict[location][1], window=7),
#                 label="means",
#             )
#             plt.title(location)
#             plt.legend()
#             plt.show()

#             scaler = create_scaler()
#             normalized_data = normalize_data(
#                 np.array(interpolated_dict[location][1]), scaler
#             )
#             train_window = 7
#             inout_seq = create_tensors(normalized_data, train_window)
#             inout_locations[location] = inout_seq


# if __name__ == "__main__":
#     main()
