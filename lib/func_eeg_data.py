import pandas as pd


def remove_non_connected_electrode_parts(eeg_data, signal_quality_data, ignored_electrodes=None, truncate_only_beginning_and_end=True, sample_frequency_data=256, sample_frequency_signal_quality=256):
    """
    Remove parts of the EEG data where the electrodes were not connected and return both EEG and signal quality data.

    Parameters:
    - eeg_data: DataFrame, the EEG data.
    - signal_quality_data: DataFrame, the signal quality data.
    - ignored_electrodes: list of str, electrodes to ignore in the analysis.
    - truncate_only_beginning_and_end: bool, if True, only truncate non-connected parts at the beginning and end.
    - sample_frequency_data: int, the sampling frequency of the EEG data.
    - sample_frequency_signal_quality: int, the sampling frequency of the signal quality data.

    Returns:
    - eeg_data_filtered: DataFrame, the EEG data with non-connected parts removed.
    - signal_quality_data_filtered: DataFrame, the signal quality data corresponding to the filtered EEG data.
    """
    if ignored_electrodes is None:
        ignored_electrodes = []

    # Resample signal quality data if frequencies do not match
    if sample_frequency_data != sample_frequency_signal_quality:
        signal_quality_data = signal_quality_data.set_index('time_seconds').resample(f'{1/sample_frequency_data}S').ffill().reset_index()

    # Merge EEG data with signal quality data
    merged_data = pd.merge_asof(eeg_data.sort_values('time_seconds'),
                                signal_quality_data.sort_values('time_seconds'),
                                on='time_seconds',
                                direction='nearest')

    # Identify non-connected parts for each electrode
    electrodes = [electrode for electrode in ['tp9', 'af7', 'af8', 'tp10'] if electrode not in ignored_electrodes]
    non_connected = merged_data[[f'signal_quality_{electrode}' for electrode in electrodes]].gt(1).any(axis=1)

    if truncate_only_beginning_and_end:
        # Find the first and last good signal
        first_good_index = non_connected.idxmin()
        last_good_index = non_connected[::-1].idxmin() + 1

        # Ensure valid indices
        if first_good_index >= last_good_index:
            return pd.DataFrame(columns=eeg_data.columns), pd.DataFrame(columns=signal_quality_data.columns)

        # Truncate data
        eeg_data_filtered = merged_data.iloc[first_good_index:last_good_index].copy()
    else:
        # Remove all non-connected parts
        eeg_data_filtered = merged_data[~non_connected].copy()

    # Separate EEG data and signal quality data
    signal_quality_columns = [col for col in merged_data.columns if 'signal_quality' in col]
    signal_quality_data_filtered = eeg_data_filtered[signal_quality_columns + ['time_seconds']].copy()
    eeg_data_filtered.drop(columns=signal_quality_columns, inplace=True)

    return eeg_data_filtered, signal_quality_data_filtered