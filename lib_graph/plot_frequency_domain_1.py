# https://chatgpt.com/c/66e799d5-07fc-8011-81ad-58fa682a7dd6
#   can you make a graph that displays the powerbands?


import numpy as np
from matplotlib import pyplot as plt

# Define the sampling rate and the EEG data
  # Hz


def plot_frequency_domain_1(eeg_data, location='.cache/',  sampling_rate = 256):

    file = 'plot_frequency_domain_1.png'

    eeg_signal = eeg_data['electrodes_average'].values

    # Perform FFT
    n = len(eeg_signal)
    frequencies = np.fft.rfftfreq(n, d=1/sampling_rate)
    fft_values = np.abs(np.fft.rfft(eeg_signal))**2  # Power spectrum

    # Define frequency bands
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 45)
    }

    # Calculate power in each band
    power_bands = {band: np.mean(fft_values[(frequencies >= low) & (frequencies < high)])
                   for band, (low, high) in bands.items()}

    # Create a time series plot of the power bands over time
    plt.figure(figsize=(14, 8))

    # Plot the power of each band
    for band in bands.keys():
        band_power = fft_values[(frequencies >= bands[band][0]) & (frequencies < bands[band][1])]
        plt.plot(frequencies[(frequencies >= bands[band][0]) & (frequencies < bands[band][1])],
                 band_power, label=f'{band} band')

    plt.title('Brainwave Power Bands (TP9)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.legend()
    #plt.show()

    # Save the figure
    plt.savefig(f'{location}/{file}', dpi=300, bbox_inches='tight')
    return file
