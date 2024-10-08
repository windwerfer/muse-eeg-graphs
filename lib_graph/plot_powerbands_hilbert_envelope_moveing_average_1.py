import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import welch

from scipy.signal import spectrogram
from scipy.signal import hilbert

from lib_graph.func_filters import bandpass_filter_filtfilt



# Define a function for moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def plot_powerbands_hilbert_envelope_moveing_average_1(eeg_data, location='.cache/', sampling_rate = 256, only_hilbert=True):

    file = 'plot_powerbands_hilbert_envelope_moveing_average_1.png'

    eeg_signal = eeg_data['electrodes_average'].values

    # Define the frequency range for the Alpha band (8-13 Hz)
    alpha_low = 8
    alpha_high = 13

    # Apply a bandpass filter to isolate the Alpha band
    alpha_signal = bandpass_filter_filtfilt(eeg_signal, alpha_low, alpha_high, sampling_rate)


    # Calculate the analytical signal using the Hilbert transform
    analytic_signal = hilbert(alpha_signal)
    envelope = np.abs(analytic_signal)



    # Apply moving average to the envelope to smooth it
    window_size = 1000  # Adjust this value for more or less smoothing
    smoothed_envelope = moving_average(envelope, window_size)

    # Plot the Alpha band signal with the smoothed envelope
    plt.figure(figsize=(14, 6))
    plt.plot(alpha_signal, color='blue', label='Alpha Band (8-13 Hz)')
    plt.plot(smoothed_envelope, color='red', label='Smoothed Envelope', linewidth=2)
    plt.title('Alpha Band of EEG Signal (TP9) - Time Domain with Smoothed Envelope')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    # plt.show()

    # Save the figure
    plt.savefig(f'{location}/{file}', dpi=300, bbox_inches='tight')

    # Close the figure to free up memory
    plt.close()

    return file