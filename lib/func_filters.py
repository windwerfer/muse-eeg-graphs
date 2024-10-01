from scipy import signal
from scipy.signal import filtfilt, butter, lfilter


def notch_filter(data, sample_rate, ignored_electrodes, freq=50.0, quality_factor=30.0):
    b, a = signal.iirnotch(freq, quality_factor, sample_rate)
    return {channel: (lfilter(b, a, data[channel]) if channel not in ignored_electrodes else data[channel])
            for channel in data.columns if channel != 'time_seconds'}




def bandpass_filter_butter(data, lowcut, highcut, sample_rate, ignored_electrodes, order=5):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return {channel: (lfilter(b, a, data[channel]) if channel not in ignored_electrodes else data[channel])
            for channel in data.columns if channel != 'time_seconds'}

def bandpass_filter_filtfilt(data, lowcut, highcut, sample_rate, order=5):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)  # Use filtfilt for zero-phase filtering


def bandpass_filter_advanced(data, lowcut, highcut, sample_rate, ignored_electrodes, order=5, filter_type='filtfilt'):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')

    def apply_filter(channel_data):
        if filter_type == 'filtfilt':
            return filtfilt(b, a, channel_data)
        elif filter_type == 'lfilter':
            return lfilter(b, a, channel_data)
        else:
            raise ValueError("Invalid filter type. Use 'filtfilt' or 'lfilter'.")

    filtered_data = {}
    for channel in data.columns:
        if channel != 'time_seconds' and channel not in ignored_electrodes:
            filtered_data[channel] = apply_filter(data[channel].values)
        else:
            filtered_data[channel] = data[channel].values

    return filtered_data
