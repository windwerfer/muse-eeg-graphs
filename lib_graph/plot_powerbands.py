import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import welch

from scipy.signal import spectrogram

from lib_graph.func_filters import bandpass_filter_filtfilt


def plot_powerbands_1(eeg_data, location='.cache/', sampling_rate = 256):

    file = 'plot_powerbands_1.png'

    eeg_signal = eeg_data['electrodes_average'].values

    # Define the frequency range for the Alpha band (8-13 Hz)
    alpha_low = 8
    alpha_high = 13

    # Apply a bandpass filter to isolate the Alpha band
    alpha_signal = bandpass_filter_filtfilt(eeg_signal, alpha_low, alpha_high, sampling_rate)

    # Plot the Alpha band signal in the time domain
    plt.figure(figsize=(14, 6))
    plt.plot(alpha_signal, color='blue', label='Alpha Band (8-13 Hz)')
    plt.title('Alpha Band of EEG Signal (TP9) - Time Domain')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    #plt.show()

    # Save the figure
    plt.savefig(f'{location}/{file}', dpi=300, bbox_inches='tight')

    # Close the figure to free up memory
    plt.close()

    return file
