import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import welch

from scipy.signal import spectrogram

def plot_time_frequency_analysis_1(eeg_data, location='.cache/', sampling_rate = 256):

    file = 'plot_time_frequency_analysis_1.png'

    eeg_signal = eeg_data['electrodes_average'].values

    # Calculate the spectrogram
    frequencies, times, Sxx = spectrogram(eeg_signal, fs=sampling_rate, nperseg=512, noverlap=256, nfft=1024)

    # Plot the spectrogram
    plt.figure(figsize=(14, 6))
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
    plt.title('Spectrogram of EEG Signal - TP9')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.colorbar(label='Power (dB)')
    plt.ylim(0, 50)  # Focus on the range of 0-50 Hz
    # plt.show()

    # Save the figure
    plt.savefig(f'{location}/{file}', dpi=300, bbox_inches='tight')

    # Close the figure to free up memory
    plt.close()

    return file
