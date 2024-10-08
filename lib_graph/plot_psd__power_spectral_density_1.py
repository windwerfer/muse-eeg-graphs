from matplotlib import pyplot as plt
from scipy.signal import welch


def plot_psd__power_spectral_density_1(eeg_data, location='.cache/', sampling_rate = 256):

    file = 'plot_psd__power_spectral_density_1.png'

    eeg_signal = eeg_data['electrodes_average'].values

    frequencies, psd = welch(eeg_signal, fs=sampling_rate, nperseg=1024)

    # Plot the Power Spectral Density (PSD)
    plt.figure(figsize=(14, 6))
    plt.semilogy(frequencies, psd)
    plt.title('Power Spectral Density (PSD) of EEG Signal - TP9')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (V^2/Hz)')
    plt.xlim(0, 50)  # Focus on the range of 0-50 Hz
    plt.grid(True)
    #plt.show()

    # Save the figure
    # Save the figure
    plt.savefig(f'{location}/{file}', dpi=300, bbox_inches='tight')

    # Close the figure to free up memory
    plt.close()

    return file