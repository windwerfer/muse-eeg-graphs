import io
import math
import re
import zipfile

import pandas as pd


def load_ica(filename, sample_rate=256, load_from=0, load_until=None, max_duration=None, col_separator=','):
    """
    Load ica data from a CSV file inside a zip archive.

    Parameters:
    - filename: str, path to the CSV or ZIP file containing the CSV.
    - sample_rate: int, the sampling rate of the data in Hz.
    - load_from: float, start loading data from this time in seconds (default 0).
    - load_until: float, stop loading data at this time in seconds (default None, load until end).
    - max_duration: float, maximum duration to load in seconds (overrides load_until if set).

    Returns:
    - signal_quality_data: DataFrame, contains the signal quality data within the time range.
    """
    # Define default column names for signal quality data
    default_columns = ['ica']

    # Function to check if a line contains letters
    def contains_letters(line):
        return bool(re.search('[a-zA-Z]', line))

    def count_columns(line, col_separator=col_separator):
        return len(line.split(col_separator))

    # Determine if the file is a zip or a csv
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            csv_file_name = [name for name in zip_ref.namelist() if name.endswith('_ica.csv')][0]
            with zip_ref.open(csv_file_name) as csv_file:
                # Check if the CSV has a header
                first_line = csv_file.readline().decode('utf-8').strip()
                csv_file.seek(0)  # Reset file pointer to the start
                has_header = contains_letters(first_line)
                count_col = count_columns(first_line)

                if count_col != len(default_columns):
                    print(f"    WARNING: {filename} has {count_col} columns, but {len(default_columns)} are expected.")
                    return None

                if has_header:
                    ica_df = pd.read_csv(io.TextIOWrapper(csv_file), sep=col_separator)
                else:
                    ica_df = pd.read_csv(io.TextIOWrapper(csv_file), sep=col_separator, header=None, names=default_columns)


    # Calculate sample indices based on time parameters
    start_sample = math.floor(load_from * sample_rate)
    if max_duration is not None:
        end_sample = start_sample + math.floor(max_duration * sample_rate)
    elif load_until is not None:
        end_sample = math.floor(load_until * sample_rate)
    else:
        end_sample = None

    # Slice the dataframe based on calculated samples
    ica_df = ica_df.iloc[start_sample:end_sample]

    # Add sample number and convert to time in seconds
    ica_df['sample_number'] = range(start_sample, start_sample + len(ica_df))
    ica_df['time_seconds'] = ica_df['sample_number'] / sample_rate

    return ica_df
