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

from lib_graph.func_eeg_data import remove_non_connected_electrode_parts, add_average_to_data

from lib_graph.func_signal_quality import identify_bad_electrodes, signal_quality_statistics
from lib_graph.load_eeg_data import load_data
from lib_graph.load_signal_quality_data import load_signal_quality
from lib_graph.plot_amplitude_distribution_histogram_1 import plot_amplitude_distribution_histogram_1
from lib_graph.plot_frequency_domain_1 import plot_frequency_domain_1
from lib_graph.plot_powerbands import plot_powerbands_1
from lib_graph.plot_powerbands_hilbert_envelope_1 import plot_powerbands_hilbert_envelope_1
from lib_graph.plot_powerbands_hilbert_envelope_moveing_average_1 import plot_powerbands_hilbert_envelope_moveing_average_1
from lib_graph.plot_psd__power_spectral_density_1 import plot_psd__power_spectral_density_1
from lib_graph.plot_time_frequency_analysis_1 import plot_time_frequency_analysis_1

from PIL import Image

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

def generate_thumbnail(file_name,thumb_name):
    # Your existing plotting code...


    # Load the image you just saved
    img = Image.open(f'{file_name}')

    # Define the width of your thumbnail
    thumb_width = 100

    # Calculate the height to maintain aspect ratio
    thumb_height = int((thumb_width / img.width) * img.height)

    # Resize the image to thumbnail size
    img.thumbnail((thumb_width, thumb_height))

    # Save the thumbnail
    img.save(f'{thumb_name}', 'PNG')



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
    eeg_data = load_data(f'{data_dir}/{file}', load_from=65, load_until=220) #, col_separator='\t')
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
    generate_thumbnail(f'{cache_dir}/{icon_name}',f'{cache_dir}/icon.png')


    print(statis_good_el)
    print(statis_bad_el)

def find_min(text):
    pattern = r'\d+min'
    # Use re.findall to find all non-overlapping matches of pattern in string
    matches = re.findall(pattern, text)
    if len(matches)>0:
        return f"({matches[0]})"

def find_date_pattern(text):
    """
    Find all occurrences of the date pattern YYYY.MM.DD in the given text.

    Parameters:
    text (str): The string to search for date patterns.

    Returns:
    list: A list of all matches found in the text.
    """
    # Define the pattern.
    # \d{4} matches four digits, \. matches a literal dot, \d{2} matches two digits
    pattern = r'\d{4}\.\d{2}\.\d{2}_\d{2}\.\d{2}'

    # Use re.findall to find all non-overlapping matches of pattern in string
    matches = re.findall(pattern, text)
    if len(matches)>0:
        datetime = matches[0].split('_')
        date = datetime[0].split('.')
        datetime[0] = f"{date[2]}.{date[1]}"
        datetime[1] = datetime[1].replace('.', ':')
        return datetime

def save_html_file(html: str, file: str) -> None:
    """
    Saves the provided HTML content to a file with UTF-8 encoding.

    Args:
        html (str): The HTML content as a string to save.
        file (str): The filename or path where the HTML should be saved.

    Returns:
        None

    Example:
        save_html_file("<html><body>Hello World!</body></html>", "index.html")
    """
    try:
        with open(file, 'w', encoding='utf-8') as f:
            # Write the HTML content to the file
            f.write(html)
        #print(f"HTML content saved successfully to {file}")
    except IOError as e:
        print(f"An error occurred while saving the file: {e}")



def generate_index_file(files, cache_dir_base):

    ul = ''
    for f in files:
        date = find_date_pattern(f)
        min = find_min(f)
        base_name = os.path.splitext(f)[0]
        ul += f'<li><a href="{base_name}/index.html"><img src="{base_name}/icon.png">{date[0]} {date[1]}h {min}</a></li>\n'

    # print(ul)
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>overview</title>
    <script src="main.js" defer></script>
    <link rel="stylesheet" href="main.css">
</head>
<body>
    <h1>Logs</h1>
    <ul>
        {ul}
    </ul>
</body>
</html>
        
        """

    save_html_file(html, f"{cache_dir_base}/index.html")

def generate_detail_html_file(file, cache_dir_base):

    ul = ''

    date = find_date_pattern(file)
    min = find_min(file)
    base_name = os.path.splitext(file)[0]
    ul += f'<li><a href="detail.html?folder={base_name}"><img src="{base_name}/icon.png">{date[0]} {date[1]}h {min}</a></li>\n'

    # print(ul)

    save_html_file(html, f"{cache_dir_base}/{base_name}/index.html")
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{date[0]} {date[1]}h {min}</title>
    <script src="main.js" defer></script>
    <link rel="stylesheet" href="main.css">
</head>
<body>
    <h1>Logs</h1>
    <ul>
        <li><img src='plot_powerbands_hilbert_envelope_moveing_average_1.png'></li>
        <li><img src='plot_frequency_domain_1.png'></li>
        <li><img src='plot_time_frequency_analysis_1.png'></li>
        <li><img src='plot_psd__power_spectral_density_1.png'></li>
        <li><img src='plot_amplitude_distribution_histogram_1.png'></li>
        <li><img src='plot_powerbands_hilbert_envelope_1.png'></li>
    </ul>
</body>
</html>
        
        """

    # return html

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