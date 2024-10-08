import os
import shutil

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from scipy import signal
import re

import zipfile
import io
import math

from scipy.signal import butter, lfilter, filtfilt, iirnotch
from sklearn.decomposition import FastICA

from lib_graph.calculate_peak_alpha import calculate_peak_alpha_simple, calculate_peak_alpha_welch, \
    calculate_peak_alpha_window, calculate_periods_peak_alpha_simple, calculate_periods_peak_alpha_welch, \
    calculate_periods_peak_alpha_window
from lib_graph.func_eeg_data import remove_non_connected_electrode_parts, add_average_to_data

from lib_graph.func_signal_quality import identify_bad_electrodes, signal_quality_statistics
from lib_graph.html_templates import generate_detail_html_file, generate_index_file
from lib_graph.load_eeg_data import load_data
from lib_graph.load_signal_quality_data import load_signal_quality
from lib_graph.plot_amplitude_distribution_histogram_1 import plot_amplitude_distribution_histogram_1
from lib_graph.plot_frequency_domain_1 import plot_frequency_domain_1
from lib_graph.plot_powerbands import plot_powerbands_1
from lib_graph.plot_powerbands_hilbert_envelope_1 import plot_powerbands_hilbert_envelope_1
from lib_graph.plot_powerbands_hilbert_envelope_moveing_average_1 import plot_powerbands_hilbert_envelope_moveing_average_1
from lib_graph.plot_psd__power_spectral_density_1 import plot_psd__power_spectral_density_1
from lib_graph.plot_time_frequency_analysis_1 import plot_time_frequency_analysis_1
from lib_graph.save_json import save_dict_to_json_pretty
from lib_graph.util import generate_img_thumbnail


# Define filter functions
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def notch_filter(data, freq, fs, quality_factor=30):
    b, a = iirnotch(freq, quality_factor, fs)
    y = lfilter(b, a, data)
    return y


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


def file_list(folder):
    """
    Generate a list of all .zip files in the specified folder.

    Parameters:
    folder (str): The path to the directory to search for zip files.

    Returns:
    list: A list of filenames (strings) that end with '.zip' in the given folder.
    """
    # Ensure the folder path exists
    if not os.path.exists(folder):
        raise FileNotFoundError(f"The folder {folder} does not exist.")

    # List to store zip file names
    zip_files = []

    # Iterate over all entries in the directory
    for file in os.listdir(folder):
        # Check if the entry is a file and has a .zip extension
        if os.path.isfile(os.path.join(folder, file)) and file.endswith('.zip'):
            zip_files.append(file)

    return zip_files



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



    #todo: warning if eeg_data is empty (file shorter than load_from)
    eeg_data = load_data(f'{data_dir}/{file}', load_from=300, load_until=1600) #, col_separator='\t')
    print('eeg loaded')

    signal_quality_data = load_signal_quality(f'{data_dir}/{file}', load_from=65, load_until=220) #, col_separator='\t')
    print('signal quality loaded')

    # Identify bad electrodes
    bad_electrodes = identify_bad_electrodes(signal_quality_data)
    if len(bad_electrodes) > 3:
        return False


    eeg_data_trunc, signal_quality_data_trunc  = remove_non_connected_electrode_parts(eeg_data, signal_quality_data, bad_electrodes)

    statis_good_el, statis_bad_el = signal_quality_statistics(signal_quality_data, bad_electrodes)
    signal_quality_statis_trunc = signal_quality_statistics(signal_quality_data_trunc)



    # add electrode average
    add_average_to_data(eeg_data_trunc, bad_electrodes)

    #### eeg_data_filterd = filter_eeg_data(eeg_data_trunc, sample_rate=sample_rate, ignored_electrodes=ignored_electrodes)

    plot_frequency_domain_1(eeg_data_trunc, location=cache_dir)
    plot_psd__power_spectral_density_1(eeg_data_trunc, location=cache_dir)
    plot_time_frequency_analysis_1(eeg_data_trunc, location=cache_dir)
    plot_amplitude_distribution_histogram_1(eeg_data_trunc, location=cache_dir)

    plot_powerbands_1(eeg_data_trunc, location=cache_dir)
    plot_powerbands_hilbert_envelope_1(eeg_data_trunc, location=cache_dir)
    icon_name = plot_powerbands_hilbert_envelope_moveing_average_1(eeg_data_trunc, location=cache_dir)
    generate_img_thumbnail(f'{cache_dir}/{icon_name}',f'{cache_dir}/icon.png')

    # nperseg = 256   # resolution of 1hz
    nperseg = 1024  # resolution of .25hz
    # nperseg = 2560  # resolution of 0.1hz - not so good, because the function assumes a stationary over this timeframe.. 10s seems too long, mostly its 1s, 4s seems to be okayisch
    pa_simple = calculate_peak_alpha_simple(eeg_data_trunc)
    ppa_simple = calculate_periods_peak_alpha_simple(eeg_data_trunc, periode_length=300)
    pa_welch = calculate_peak_alpha_welch(eeg_data_trunc, nperseg=nperseg)
    ppa_welch = calculate_periods_peak_alpha_welch(eeg_data_trunc, nperseg=nperseg, periode_length=300)
    pa_window = calculate_peak_alpha_window(eeg_data_trunc)
    ppa_window = calculate_periods_peak_alpha_window(eeg_data_trunc, periode_length=300)

    statistics_json = {'peak_alpha_simple':pa_simple, 'peak_alpha_welch':pa_welch, 'peak_alpha_window':pa_window, 'periods_peak_alpha_simple':ppa_simple, 'periods_peak_alpha_welch':ppa_welch, 'periods_peak_alpha_window':ppa_window,  'table_good_electrodes':statis_good_el, 'table_bad_electrodes':statis_bad_el}
    save_dict_to_json_pretty(statistics_json, filename='statistics.json', location=cache_dir)

    # TODO: 1) generate '{cache_dir}/statistics.json' and create a {cache_dir_base}/summary.csv
    #       2) peak alpha stats


    print(statis_good_el)
    print(statis_bad_el)


def main():


    data_dir = 'out_eeg'
    cache_dir_base = f'cache'



    files = file_list(data_dir)

    # generate_img_report_for(files[1], cache_dir_base, data_dir)
    # generate_detail_html_file(files[1], f'{cache_dir_base}')

    for f in files:
        generate_img_report_for(f, cache_dir_base, data_dir)
        generate_detail_html_file(f, f'{cache_dir_base}')

    generate_index_file(files, f'{cache_dir_base}')


if __name__ == "__main__":
    main()