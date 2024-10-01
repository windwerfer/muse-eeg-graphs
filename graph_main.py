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

from lib.func_eeg_data import remove_non_connected_electrode_parts

from lib.func_signal_quality import identify_bad_electrodes, signal_quality_statistics
from lib.load_eeg_data import load_data
from lib.load_signal_quality_data import load_signal_quality
from lib.plot_amplitude_distribution_histogram_1 import plot_amplitude_distribution_histogram_1
from lib.plot_frequency_domain_1 import plot_frequency_domain_1
from lib.plot_powerbands import plot_powerbands_1
from lib.plot_powerbands_hilbert_envelope_1 import plot_powerbands_hilbert_envelope_1
from lib.plot_powerbands_hilbert_envelope_moveing_average_1 import plot_powerbands_hilbert_envelope_moveing_average_1
from lib.plot_psd__power_spectral_density_1 import plot_psd__power_spectral_density_1
from lib.plot_time_frequency_analysis_1 import plot_time_frequency_analysis_1


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






if __name__ == "__main__":
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
    eeg_data = load_data('out_eeg/tho_eeglab_2024.09.04_22.02.zip', load_from=65, load_until=220) #, col_separator='\t')
    print('eeg loaded')

    signal_quality_data = load_signal_quality('out_eeg/tho_eeglab_2024.09.04_22.02.zip', load_from=65, load_until=220) #, col_separator='\t')
    print('signal quality loaded')

    # Identify bad electrodes
    ignored_electrodes = identify_bad_electrodes(signal_quality_data)

    eeg_data_trunc, signal_quality_data_trunc  = remove_non_connected_electrode_parts(eeg_data, signal_quality_data, ignored_electrodes)

    statis_good_el, statis_bad_el = signal_quality_statistics(signal_quality_data, ignored_electrodes)
    signal_quality_statis_trunc = signal_quality_statistics(signal_quality_data_trunc)



    #### eeg_data_filterd = filter_eeg_data(eeg_data_trunc, sample_rate=sample_rate, ignored_electrodes=ignored_electrodes)

    # plot_frequency_domain_1(eeg_data_trunc)
    # plot_psd__power_spectral_density_1(eeg_data_trunc)
    # plot_time_frequency_analysis_1(eeg_data_trunc)
    # plot_amplitude_distribution_histogram_1(eeg_data_trunc)

    plot_powerbands_1(eeg_data_trunc)
    plot_powerbands_hilbert_envelope_1(eeg_data_trunc)
    plot_powerbands_hilbert_envelope_moveing_average_1(eeg_data_trunc)


    print(statis_good_el)
    print(statis_bad_el)
