import argparse
import os
import shutil
import sys

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from scipy import signal
import re

import zipfile
import io
import math

from scipy.signal import butter, lfilter, filtfilt, iirnotch


from lib_graph.calculate_peak_alpha import calculate_peak_alpha_simple, calculate_peak_alpha_welch, \
    calculate_peak_alpha_window, calculate_periods_peak_alpha_simple, calculate_periods_peak_alpha_welch, \
    calculate_periods_peak_alpha_window
from lib_graph.func_eeg_data import remove_non_connected_electrode_parts, add_average_to_data

from lib_graph.func_signal_quality import identify_bad_electrodes, signal_quality_statistics
from lib_graph.html_templates import generate_detail_html_file, generate_index_file, generate_css_file
from lib_graph.load_drlref_data import load_drlref
from lib_graph.load_eeg_data import load_data
from lib_graph.load_ica_data import load_ica
from lib_graph.load_signal_quality_data import load_signal_quality
from lib_graph.plot_amplitude_distribution_histogram_1 import plot_amplitude_distribution_histogram_1
from lib_graph.plot_frequency_domain_1 import plot_frequency_domain_1
from lib_graph.plot_powerbands import plot_powerbands_1
from lib_graph.plot_powerbands_hilbert_envelope_1 import plot_powerbands_hilbert_envelope_1
from lib_graph.plot_powerbands_hilbert_envelope_moveing_average_1 import \
    plot_powerbands_hilbert_envelope_moveing_average_1
from lib_graph.plot_psd__power_spectral_density_1 import plot_psd__power_spectral_density_1
from lib_graph.plot_time_frequency_analysis_1 import plot_time_frequency_analysis_1
from lib_graph.save_json import save_dict_to_json_pretty
from lib_graph.util import generate_img_thumbnail, is_running_in_pycharm, get_script_memory_usage, print_mem_usage


def parse_args():
    parser = argparse.ArgumentParser(description="Process file calculation options.")

    # Add argument for recalculating the last N files
    parser.add_argument('-r', '--recalculate', type=int, metavar='N', help='Recalculate the last N files')

    # Add argument for limiting the number of files to process
    parser.add_argument('-l', '--limit', type=int, metavar='N', help='Calculate max N files')

    args = parser.parse_args()

    return args

def mk_dir(folder):
    try:
        # Attempt to create the directory
        os.makedirs(folder, exist_ok=True)
    except OSError as e:
        pass


def rm_dir(folder):
    try:
        # Use shutil.rmtree to delete a directory and all its contents
        shutil.rmtree(folder)
    except OSError as e:
        pass


import os
from datetime import datetime

import os

def folder_list(root_folder):
    """
    Lists all folders within the root_folder that contain a 'statistics.json' file larger than 0 bytes.

    Parameters:
    root_folder (str): The path to the directory to start searching from.

    Returns:
    list: A list of folder paths that meet the criteria.

    Raises:
    FileNotFoundError: If the root_folder does not exist.
    """
    # Check if the root folder exists
    if not os.path.exists(root_folder):
        return []

    valid_folders = []

    # Walk through all directories in the root folder
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # Check if 'statistics.json' exists and is not empty
        if 'statistics.json' in filenames:
            stats_file_path = os.path.join(dirpath, 'statistics.json')
            # Check if the file size is greater than 0 bytes
            if os.path.getsize(stats_file_path) > 0:
                # Get the relative path from base to folder
                p = os.path.relpath(dirpath, root_folder)
                valid_folders.append(p)

    return valid_folders


def file_list(folder):
    """
    Generate a list of all .zip files in the specified folder, sorted by creation time.

    Parameters:
    folder (str): The path to the directory to search for zip files.

    Returns:
    list: A list of tuples containing (filename, creation_time) sorted by creation time,
          with the most recent first. The creation time is a datetime object.

    Raises:
    FileNotFoundError: If the specified folder does not exist.
    """
    # Ensure the folder path exists
    if not os.path.exists(folder):
        raise FileNotFoundError(f"The folder {folder} does not exist.")

    # List to store zip file names with their creation times
    zip_files = []

    # Iterate over all entries in the directory
    for file in os.listdir(folder):
        # Full file path
        full_path = os.path.join(folder, file)
        # Check if the entry is a file and has a .zip extension
        if os.path.isfile(full_path) and file.endswith('.zip'):
            # Get the creation time of the file (note: this is platform dependent,
            # on Unix, it returns the last metadata change, on Windows, the creation time)
            creation_time = os.path.getctime(full_path)
            # Convert to datetime object for better readability
            creation_time = datetime.fromtimestamp(creation_time)
            zip_files.append((file, creation_time))

    # Sort the files by creation time, most recent first
    zip_files.sort(key=lambda x: x[1], reverse=True)

    # Return only the filenames in the sorted order
    return [file for file, _ in zip_files]


def generate_img_report_for(file='tho_eeglab_2024.09.04_22.02.zip', cache_dir_base='cache', data_dir='out_eeg'):
    base_name = os.path.splitext(file)[0]
    cache_dir = f'{cache_dir_base}/{base_name}'
    rm_dir(cache_dir)
    mk_dir(cache_dir)

    sample_rate = 256  # Hz

    # Band colors
    band_colors = {
        'alpha': 'navy',
        'delta': 'green',
        'theta': 'turquoise',
        'beta': 'violet',
        'gamma': '#808080'  # Hex value for gray
        }

    # todo: warning if eeg_data is empty (file shorter than load_from)
    # load_from = 300
    # load_until = 1600
    load_from = 0
    load_until = None
    eeg_data = load_data(f'{data_dir}/{file}', load_from=load_from, load_until=load_until)
    print_mem_usage('eeg loaded')

    signal_quality_data = load_signal_quality(f'{data_dir}/{file}', load_from=load_from, load_until=load_until)
    ica_data = load_ica(f'{data_dir}/{file}', load_from=load_from, load_until=load_until)
    drlref_data = load_drlref(f'{data_dir}/{file}', load_from=load_from, load_until=load_until)
    if signal_quality_data is None:
        return None

    print_mem_usage('signal quality files loaded')

    # Identify bad electrodes
    bad_electrodes = identify_bad_electrodes(signal_quality_data)
    if len(bad_electrodes) > 3:
        return None

    eeg_data_trunc, signal_quality_data_trunc = remove_non_connected_electrode_parts(eeg_data, signal_quality_data,
                                                                                     bad_electrodes)

    statis_good_el, statis_bad_el, stats_json = signal_quality_statistics(signal_quality_data, bad_electrodes)
    # signal_quality_statis_trunc = signal_quality_statistics(signal_quality_data_trunc)

    # add electrode average
    add_average_to_data(eeg_data_trunc, bad_electrodes)
    print_mem_usage('eeg signals processed')

    #### eeg_data_filterd = filter_eeg_data(eeg_data_trunc, sample_rate=sample_rate, ignored_electrodes=ignored_electrodes)

    plot_frequency_domain_1(eeg_data_trunc, location=cache_dir)
    print_mem_usage('plot (frequency_domain)')

    plot_psd__power_spectral_density_1(eeg_data_trunc, location=cache_dir)
    print_mem_usage('plot (power_spectral_density)')

    plot_time_frequency_analysis_1(eeg_data_trunc, location=cache_dir)
    print_mem_usage('plot (time_frequency_analysis)')

    plot_amplitude_distribution_histogram_1(eeg_data_trunc, location=cache_dir)
    print_mem_usage('plot (amplitude_distribution_histogram)')

    plot_powerbands_1(eeg_data_trunc, location=cache_dir)
    print_mem_usage('plot (powerbands)')

    plot_powerbands_hilbert_envelope_1(eeg_data_trunc, location=cache_dir)
    print_mem_usage('plot (powerbands_hilbert_envelope)')

    icon_name = plot_powerbands_hilbert_envelope_moveing_average_1(eeg_data_trunc, location=cache_dir)
    generate_img_thumbnail(f'{cache_dir}/{icon_name}', f'{cache_dir}/icon.png')
    print_mem_usage('plot (icon)')


    # nperseg = 256   # resolution of 1hz
    nperseg = 1024  # resolution of .25hz
    # nperseg = 2560  # resolution of 0.1hz - not so good, because the function assumes a stationary over this timeframe.. 10s seems too long, mostly its 1s, 4s seems to be okayisch
    pa_simple = calculate_peak_alpha_simple(eeg_data_trunc)
    ppa_simple = calculate_periods_peak_alpha_simple(eeg_data_trunc, periode_length=300)

    print_mem_usage('peak alpha (simple) processed')

    # nperseg=256 -> each segment is 1s long,  nperseg=1024 -> each segment is 4s long. (the welch function assumes that
    # the waveform is static, which is only true for short periods of time, so 1s is better than 4s
    # (10s would be too error-prone), but 4s gives 0.25Hz resolution while 1s only gives 1Hz resolution..
    pa_welch = calculate_peak_alpha_welch(eeg_data_trunc, nperseg=256)
    pa_welch4s = calculate_peak_alpha_welch(eeg_data_trunc, nperseg=1024)
    ppa_welch = calculate_periods_peak_alpha_welch(eeg_data_trunc, nperseg=256, periode_length=300)
    ppa_welch4s = calculate_periods_peak_alpha_welch(eeg_data_trunc, nperseg=1024, periode_length=300)
    pa_window = calculate_peak_alpha_window(eeg_data_trunc)
    ppa_window = calculate_periods_peak_alpha_window(eeg_data_trunc, periode_length=300)

    print_mem_usage('peak alpha (all) processed')

    statistics_json = {
        'peak_alpha_simple': pa_simple,
        'peak_alpha_welch': pa_welch,
        'peak_alpha_welch4s': pa_welch4s,
        'peak_alpha_window': pa_window,
        'periods_peak_alpha_simple': ppa_simple,
        'periods_peak_alpha_welch': ppa_welch,
        'periods_peak_alpha_welch4s': ppa_welch4s,
        'periods_peak_alpha_window': ppa_window,
        'table_good_electrodes': stats_json['good_electrodes'],
        'table_bad_electrodes': stats_json['bad_electrodes']
    }
    save_dict_to_json_pretty(statistics_json, filename='statistics.json', location=cache_dir)


    # print(statis_good_el)
    # print(statis_bad_el)
    print(' -------------')



def main(limit=None, recalculate=None):
    data_dir = 'out_eeg'
    cache_dir_base = f'cache'

    files = file_list(data_dir)
    already_processed_folders = folder_list(cache_dir_base)

    i = 1
    if recalculate is not None:
        already_processed_folders = []
    for f in files:

        # process all files that where not yet processed (are not present in the cache dir)
        if f.replace('.zip', '') in already_processed_folders:
            continue
        if limit is not None:
            # limit the processing of files (mostly for testing)
            if i > limit:
                break
            i += 1

        generate_img_report_for(f, cache_dir_base, data_dir)
        generate_detail_html_file(f, f'{cache_dir_base}')

    generate_index_file(files, f'{cache_dir_base}')
    generate_css_file(f'{cache_dir_base}')


if __name__ == "__main__":

    limit = recalculate = None
    args = parse_args()
    if args.recalculate is not None:
        recalculate = args.recalculate
    if args.limit is not None:
        limit = args.limit

    if  is_running_in_pycharm():    # if run through pycharm
        limit = 2                   # only process most recent file
        # recalculate = 1


    main(limit=limit, recalculate=recalculate)
