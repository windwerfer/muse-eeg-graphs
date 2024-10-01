import io
import math
import re
import zipfile

import pandas as pd


def load_data(filename, keep_channels=['tp9', 'af7', 'af8', 'tp10'], sample_rate=256, load_from=0, load_until=None, max_duration=None, col_separator=','):
    """
    Load EEG data from a CSV file, which might be inside a zip archive. Assumes column order if no header is present.

    Parameters:
    - filename: str, path to the CSV or ZIP file containing the CSV.
    - keep_channels: list of str, names of the channels to keep (must match predefined names if no header).
    - sample_rate: int, the sampling rate of the EEG data in Hz.
    - load_from: float, start loading data from this time in seconds (default 0).
    - load_until: float, stop loading data at this time in seconds (default None, load until end).
    - max_duration: float, maximum duration to load in seconds (overrides load_until if set).

    Returns:
    - eeg_data: DataFrame, contains the time series data for the specified channels within the time range.
    """
    # Define default column names if no header is provided
    default_columns = ['tp9', 'af7', 'af8', 'tp10']


    # Function to check if a line contains letters
    def contains_letters(line):
        return bool(re.search('[a-zA-Z]', line))

    # Determine if the file is a zip or a csv
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            csv_file_name = [name for name in zip_ref.namelist() if name.endswith('_eeg.csv')][0]
            with zip_ref.open(csv_file_name) as csv_file:
                # Check if the CSV has a header
                first_line = csv_file.readline().decode('utf-8').strip()
                csv_file.seek(0)  # Reset file pointer to the start
                has_header = contains_letters(first_line)

                if has_header:
                    eeg_df = pd.read_csv(io.TextIOWrapper(csv_file), sep=col_separator)
                else:
                    eeg_df = pd.read_csv(io.TextIOWrapper(csv_file), sep=col_separator, header=None, names=default_columns)
    else:
        # Check if the CSV file has a header
        with open(filename, 'r') as f:
            first_line = f.readline().strip()
        has_header = contains_letters(first_line)

        if has_header:
            eeg_df = pd.read_csv(filename, sep=col_separator)
        else:
            eeg_df = pd.read_csv(filename, sep=col_separator, header=None, names=default_columns)

    # Calculate sample indices based on time parameters
    start_sample = math.floor(load_from * sample_rate)
    if max_duration is not None:
        end_sample = start_sample + math.floor(max_duration * sample_rate)
    elif load_until is not None:
        end_sample = math.floor(load_until * sample_rate)
    else:
        end_sample = None

    # Slice the dataframe based on calculated samples
    eeg_df = eeg_df.iloc[start_sample:end_sample]

    # Add sample number and convert to time in seconds
    eeg_df['sample_number'] = range(start_sample, start_sample + len(eeg_df))
    eeg_df['time_seconds'] = eeg_df['sample_number'] / sample_rate

    # Ensure keep_channels are valid
    valid_channels = [channel for channel in keep_channels if channel in eeg_df.columns]
    if not valid_channels:
        raise ValueError("None of the keep_channels are recognized or present in the data.")

    # Select only the channels we want to keep
    eeg_data = eeg_df[valid_channels + ['time_seconds']].copy()

    return eeg_data