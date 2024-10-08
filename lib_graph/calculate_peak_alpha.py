import pandas as pd
import numpy as np
from scipy import signal


#  This example doesn't apply windowing or overlapping segments which are common in spectral analysis for more
#  accurate results, especially for shorter segments or when looking for changes over time. If you need to
#  calculate for different intervals with high precision, consider using techniques like Welch's method
#  or applying a window function.


def calculate_peak_alpha_simple(eeg_data, sample_rate=256, flatness_threshold=0.1, power_threshold=1e-5):
    channels = ['tp9', 'af7', 'af8', 'tp10']
    peak_alphas = {}

    for channel in channels:
        channel_data = eeg_data[channel].values

        # Compute the FFT
        freqs = np.fft.fftfreq(len(channel_data), d=1 / sample_rate)
        fft_values = np.abs(np.fft.fft(channel_data))

        # Consider only positive frequencies
        positive_freqs = freqs[freqs >= 0]
        fft_values = fft_values[freqs >= 0]

        # Define the alpha band
        alpha_band = (positive_freqs >= 8) & (positive_freqs <= 13)

        # Extract alpha band data
        alpha_fft = fft_values[alpha_band]
        alpha_freqs = positive_freqs[alpha_band]

        # Check for a discernible peak in the alpha band
        if np.max(alpha_fft) - np.min(alpha_fft) < flatness_threshold or np.max(alpha_fft) < power_threshold:
            peak_alpha_freq = None  # Indicating no significant peak detected
        else:
            peak_alpha_freq = alpha_freqs[np.argmax(alpha_fft)]

        peak_alphas[channel] = peak_alpha_freq

    # Calculate the mean, filtering out None values
    valid_peaks = [v for v in peak_alphas.values() if v is not None]
    if not valid_peaks:  # If all are None
        overall_peak_alpha = None
    else:
        overall_peak_alpha = np.mean(valid_peaks)

    return {'peak_alphas': peak_alphas, 'mean_peak_alpha': overall_peak_alpha}



def calculate_peak_alpha_welch(eeg_data, sample_rate=256, nperseg=256, noverlap=None, flatness_threshold=0.1,
                               power_threshold=1e-5):
    channels = ['tp9', 'af7', 'af8', 'tp10']

    peak_alphas = {}

    if noverlap is None:
        noverlap = nperseg // 2  # Default overlap

    for channel in channels:
        channel_data = eeg_data[channel].values

        # Using Welch's method to compute PSD
        freqs, psd = signal.welch(channel_data, sample_rate, nperseg=nperseg, noverlap=noverlap)

        # Define alpha band
        alpha_band = (freqs >= 8) & (freqs <= 13)

        # Check for peak within the alpha band
        alpha_psd = psd[alpha_band]

        # If the power spectrum is flat or below threshold, consider no peak
        if np.max(alpha_psd) - np.min(alpha_psd) < flatness_threshold or np.max(alpha_psd) < power_threshold:
            peak_alpha_freq = None
        else:
            peak_alpha_freq = freqs[alpha_band][np.argmax(alpha_psd)]

        peak_alphas[channel] = peak_alpha_freq

    # Calculate mean, ignoring None values
    valid_peaks = [v for v in peak_alphas.values() if v is not None]
    if not valid_peaks:
        overall_peak_alpha = None
    else:
        overall_peak_alpha = np.mean(valid_peaks)

    return {'peak_alphas': peak_alphas, 'mean_peak_alpha': overall_peak_alpha}




def calculate_peak_alpha_window(eeg_data, sample_rate=256, window='hann', flatness_threshold=0.1, power_threshold=1e-5):
    channels = ['tp9', 'af7', 'af8', 'tp10']
    peak_alphas = {}

    for channel in channels:
        channel_data = eeg_data[channel].values

        # Create and apply window
        win = signal.get_window(window, len(channel_data))
        windowed_data = channel_data * win

        # Compute the FFT
        freqs = np.fft.fftfreq(len(windowed_data), d=1 / sample_rate)
        fft_values = np.abs(np.fft.fft(windowed_data))

        # Consider only positive frequencies
        positive_freqs = freqs[freqs >= 0]
        fft_values = fft_values[freqs >= 0]

        # Define alpha band
        alpha_band = (positive_freqs >= 8) & (positive_freqs <= 13)

        # Extract alpha band data
        alpha_fft = fft_values[alpha_band]
        alpha_freqs = positive_freqs[alpha_band]

        # Check if there's a discernible peak
        if np.max(alpha_fft) - np.min(alpha_fft) < flatness_threshold or np.max(alpha_fft) < power_threshold:
            peak_alpha_freq = None  # No significant peak detected
        else:
            peak_alpha_freq = alpha_freqs[np.argmax(alpha_fft)]

        peak_alphas[channel] = peak_alpha_freq

    # Check if all channels returned None, if not calculate mean
    if all(v is None for v in peak_alphas.values()):
        overall_peak_alpha = None
    else:
        # Filter out None values before calculating mean
        overall_peak_alpha = np.mean([v for v in peak_alphas.values() if v is not None])

    return {'peak_alphas': peak_alphas, 'mean_peak_alpha': overall_peak_alpha}


def calculate_periods_peak_alpha_simple(eeg_data, periode_length=600, sample_rate=256, flatness_threshold=0.1, power_threshold=1e-5):
    # Ensure periode_length is in samples, not seconds
    periode_length_samples = periode_length * sample_rate

    # Check if the data length allows for at least one complete period
    if len(eeg_data) < periode_length_samples:
        raise ValueError("The EEG data is shorter than the specified period length.")

    # Number of periods we can fit into the EEG data length
    num_periods = len(eeg_data) // periode_length_samples

    #  If there's any remaining data less than periode_length, add an additional period
    if len(eeg_data) % periode_length_samples != 0:
        num_periods += 1

    results = []


    for i in range(num_periods):
        start_from = i * periode_length_samples
        end_at = start_from + periode_length_samples
        if end_at > len(eeg_data):      # if end_at is bigger than eeg_data, reduce to len(eeg_data)
            end_at = len(eeg_data)
        # Slice the data for this period
        eeg_slice = eeg_data.iloc[start_from:end_at]

        channels = ['tp9', 'af7', 'af8', 'tp10']
        peak_alphas = {}

        for channel in channels:
            channel_data = eeg_slice[channel].values

            # Compute the FFT
            freqs = np.fft.fftfreq(len(channel_data), d=1 / sample_rate)
            fft_values = np.abs(np.fft.fft(channel_data))

            # Consider only positive frequencies
            positive_freqs = freqs[freqs >= 0]
            fft_values = fft_values[freqs >= 0]

            # Define the alpha band
            alpha_band = (positive_freqs >= 8) & (positive_freqs <= 13)

            # Extract alpha band data
            alpha_fft = fft_values[alpha_band]
            alpha_freqs = positive_freqs[alpha_band]

            # Check for a discernible peak in the alpha band
            if np.max(alpha_fft) - np.min(alpha_fft) < flatness_threshold or np.max(alpha_fft) < power_threshold:
                peak_alpha_freq = None  # Indicating no significant peak detected
            else:
                peak_alpha_freq = alpha_freqs[np.argmax(alpha_fft)]

            peak_alphas[channel] = peak_alpha_freq

        # Calculate the mean, filtering out None values
        valid_peaks = [v for v in peak_alphas.values() if v is not None]
        if not valid_peaks:  # If all are None
            overall_peak_alpha = None
        else:
            overall_peak_alpha = np.mean(valid_peaks)

        # Append results for this slice
        results.append({
            'periode_start': periode_length * i,
            'periode_length': int((end_at - start_from) / sample_rate),
            'peak_aplhas': peak_alphas,
            'mean_peak_alpha': overall_peak_alpha
        })

    return results

def calculate_periods_peak_alpha_welch(eeg_data, periode_length=600, sample_rate=256, nperseg=256, noverlap=None, flatness_threshold=0.1, power_threshold=1e-5):
    # Ensure periode_length is in samples, not seconds
    periode_length_samples = periode_length * sample_rate

    # Check if the data length allows for at least one complete period
    if len(eeg_data) < periode_length_samples:
        raise ValueError("The EEG data is shorter than the specified period length.")

    # Number of periods we can fit into the EEG data length
    num_periods = len(eeg_data) // periode_length_samples

    #  If there's any remaining data less than periode_length, add an additional period
    if len(eeg_data) % periode_length_samples != 0:
        num_periods += 1

    results = []


    for i in range(num_periods):
        start_from = i * periode_length_samples
        end_at = start_from + periode_length_samples
        if end_at > len(eeg_data):      # if end_at is bigger than eeg_data, reduce to len(eeg_data)
            end_at = len(eeg_data)

        # Slice the data for this period
        eeg_slice = eeg_data.iloc[start_from:end_at]

        channels = ['tp9', 'af7', 'af8', 'tp10']

        peak_alphas = {}

        if noverlap is None:
            noverlap = nperseg // 2  # Default overlap

        for channel in channels:
            channel_data = eeg_slice[channel].values

            # Using Welch's method to compute PSD
            freqs, psd = signal.welch(channel_data, sample_rate, nperseg=nperseg, noverlap=noverlap)

            # Define alpha band
            alpha_band = (freqs >= 8) & (freqs <= 13)

            # Check for peak within the alpha band
            alpha_psd = psd[alpha_band]

            # If the power spectrum is flat or below threshold, consider no peak
            if np.max(alpha_psd) - np.min(alpha_psd) < flatness_threshold or np.max(alpha_psd) < power_threshold:
                peak_alpha_freq = None
            else:
                peak_alpha_freq = freqs[alpha_band][np.argmax(alpha_psd)]

            peak_alphas[channel] = peak_alpha_freq

        # Calculate mean, ignoring None values
        valid_peaks = [v for v in peak_alphas.values() if v is not None]
        if not valid_peaks:
            overall_peak_alpha = None
        else:
            overall_peak_alpha = np.mean(valid_peaks)

        # Append results for this slice
        results.append({
            'periode_start': periode_length * i,
            'periode_length': int((end_at - start_from) / sample_rate),
            'peak_aplhas': peak_alphas,
            'mean_peak_alpha': overall_peak_alpha
        })

    return results



def calculate_periods_peak_alpha_window(eeg_data, periode_length=600, sample_rate=256, window='hann', flatness_threshold=0.1, power_threshold=1e-5):
    # Ensure periode_length is in samples, not seconds
    periode_length_samples = periode_length * sample_rate

    # Check if the data length allows for at least one complete period
    if len(eeg_data) < periode_length_samples:
        raise ValueError("The EEG data is shorter than the specified period length.")

    # Number of periods we can fit into the EEG data length
    num_periods = len(eeg_data) // periode_length_samples

    #  If there's any remaining data less than periode_length, add an additional period
    if len(eeg_data) % periode_length_samples != 0:
        num_periods += 1

    results = []

    for i in range(num_periods):
        start_from = i * periode_length_samples
        end_at = start_from + periode_length_samples
        if end_at > len(eeg_data):  # if end_at is bigger than eeg_data, reduce to len(eeg_data)
            end_at = len(eeg_data)

        # Slice the data for this period
        eeg_slice = eeg_data.iloc[start_from:end_at]

        channels = ['tp9', 'af7', 'af8', 'tp10']
        peak_alphas = {}

        for channel in channels:
            channel_data = eeg_slice[channel].values

            # Create and apply window
            win = signal.get_window(window, len(channel_data))
            windowed_data = channel_data * win

            # Compute the FFT
            freqs = np.fft.fftfreq(len(windowed_data), d=1 / sample_rate)
            fft_values = np.abs(np.fft.fft(windowed_data))

            # Consider only positive frequencies
            positive_freqs = freqs[freqs >= 0]
            fft_values = fft_values[freqs >= 0]

            # Define alpha band
            alpha_band = (positive_freqs >= 8) & (positive_freqs <= 13)

            # Extract alpha band data
            alpha_fft = fft_values[alpha_band]
            alpha_freqs = positive_freqs[alpha_band]

            # Check if there's a discernible peak
            if np.max(alpha_fft) - np.min(alpha_fft) < flatness_threshold or np.max(alpha_fft) < power_threshold:
                peak_alpha_freq = None  # No significant peak detected
            else:
                peak_alpha_freq = alpha_freqs[np.argmax(alpha_fft)]

            peak_alphas[channel] = peak_alpha_freq

        # Check if all channels returned None, if not calculate mean
        if all(v is None for v in peak_alphas.values()):
            overall_peak_alpha = None
        else:
            # Filter out None values before calculating mean
            overall_peak_alpha = np.mean([v for v in peak_alphas.values() if v is not None])

        # Append results for this slice
        results.append({
            'periode_start': periode_length * i,
            'periode_length': int((end_at - start_from) / sample_rate),
            'peak_aplhas': peak_alphas,
            'mean_peak_alpha': overall_peak_alpha
        })

    return results
