import pandas as pd


def signal_quality_statistics(signal_quality_data, ignored_electrodes=None):
    """
    Calculate statistics for signal quality data, considering ignored electrodes.

    Parameters:
    - signal_quality_data: DataFrame, the signal quality data.
    - ignored_electrodes: list of str, electrodes to ignore in the analysis.

    Returns:
    - stats_df: DataFrame, statistics for non-ignored electrodes.
    - stats_df_bad_electrodes: DataFrame, statistics for ignored electrodes.
    """
    if ignored_electrodes is None:
        ignored_electrodes = []

    # Initialize results dictionaries
    result = {}
    bad_result = {}
    electrodes = ['tp9', 'af7', 'af8', 'tp10']

    for electrode in electrodes:
        # Extract signal quality for the electrode
        signal_quality = signal_quality_data[f'signal_quality_{electrode}']

        # Count non-good signals
        non_good_signals = (signal_quality != 1).sum()

        # Detect blocks of non-good signals
        blocks = (signal_quality != 1).astype(int).groupby(signal_quality.ne(signal_quality.shift()).cumsum()).sum()
        non_good_blocks = blocks[blocks > 0]
        total_non_good_blocks = len(non_good_blocks)
        average_block_length = non_good_blocks.mean() if total_non_good_blocks > 0 else 0

        # Calculate percentages
        total_signals = len(signal_quality)
        good_percentage = 100 * (total_signals - non_good_signals) / total_signals
        non_good_percentage = 100 - good_percentage

        # Store results in the appropriate dictionary
        if electrode in ignored_electrodes:
            bad_result[electrode] = {
                'Non-Good Signals': non_good_signals,
                'Average Block Length': average_block_length,
                'Good Percentage': good_percentage,
                'Non-Good Percentage': non_good_percentage
            }
        else:
            result[electrode] = {
                'Non-Good Signals': non_good_signals,
                'Average Block Length': average_block_length,
                'Good Percentage': good_percentage,
                'Non-Good Percentage': non_good_percentage
            }

    # Convert results to DataFrames and transpose for better representation
    stats_df = pd.DataFrame(result).T
    stats_df_bad_electrodes = pd.DataFrame(bad_result).T

    # Set pandas display options to show all columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    return stats_df, stats_df_bad_electrodes



def identify_bad_electrodes(signal_quality_data, threshold=90):
    """
    Identify electrodes with more than a specified percentage of non-good signals.

    Parameters:
    - signal_quality_data: DataFrame, the signal quality data.
    - threshold: float, the percentage threshold above which an electrode is considered bad.

    Returns:
    - ignored_electrodes: list of str, names of electrodes to ignore.
    """
    electrodes = ['tp9', 'af7', 'af8', 'tp10']
    ignored_electrodes = []

    for electrode in electrodes:
        signal_quality = signal_quality_data[f'signal_quality_{electrode}']
        non_good_percentage = (signal_quality > 1).mean() * 100

        if non_good_percentage > threshold:
            ignored_electrodes.append(electrode)

    return ignored_electrodes