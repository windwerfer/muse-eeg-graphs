
from matplotlib import pyplot as plt




# what is the graph about?

# 1. Signal Quality
#
#     Normal Distribution: The histogram shows a roughly normal (bell-shaped) distribution centered around a mean value.
#     This suggests that the signal is relatively clean and free from large artifacts or extreme noise.
#     Lack of Artifacts: Artifacts (e.g., muscle movements, eye blinks) usually introduce large spikes or skewness in the signal.
#     A clean EEG signal like this, with no extreme peaks or unusual skewness in the amplitude distribution,
#     suggests that the recording conditions were good and that the data is suitable for further analysis.
#
# 2. Amplitude Range
#
#     The histogram shows that most of the EEG amplitudes are within a certain range (740-840).
#     This range is typical for EEG signals, where the amplitudes are often measured in microvolts (µV).
#     This indicates that the signal falls within the expected range of EEG recordings,
#     which further suggests the absence of excessive noise or artifacts.


def plot_amplitude_distribution_histogram_1(eeg_data, location='.cache/', sampling_rate = 256):

    file = 'plot_amplitude_distribution_histogram_1.png'

    eeg_signal = eeg_data['electrodes_average'].values

    plt.figure(figsize=(10, 6))
    plt.hist(eeg_signal, bins=50, color='c', edgecolor='black', alpha=0.7)
    plt.title('Amplitude Distribution of EEG Signal - TP9')
    plt.xlabel('Amplitude')
    plt.ylabel('Frequency')
    plt.grid(True)
    # plt.show()


    # Save the figure
    plt.savefig(f'{location}/{file}', dpi=300, bbox_inches='tight')
    return file