import json

from numpy import int64


def convert_int64(obj):
    if isinstance(obj, dict):
        return {k: convert_int64(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_int64(item) for item in obj]
    elif isinstance(obj, int64):
        return int(obj)
    else:
        return obj

def save_dict_to_json_pretty(dict_to_save, filename, location='cache'):
    """
    Save a dictionary to a JSON file with pretty formatting.

    :param dict_to_save: Dictionary to be saved into the JSON file.
    :param filename: Name of the file where the JSON will be saved.
    """
    dict_to_save_converted = convert_int64(dict_to_save)
    try:
        with open(f'{location}/{filename}', 'w', encoding='utf-8') as file:
            # Using indent for pretty print, sort_keys to sort the keys,
            # and ensure_ascii=False to allow non-ASCII characters
            json.dump(dict_to_save_converted, file, indent=4, sort_keys=True, ensure_ascii=False)
            # Add newline at the end of the file for better readability in some editors
            file.write('\n')
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")
    except TypeError as e:
        print(f"The dictionary contains objects that are not JSON serializable: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")