import json


def save_dict_to_json_pretty(dict_to_save, filename, location='cache'):
    """
    Save a dictionary to a JSON file with pretty formatting.

    :param dict_to_save: Dictionary to be saved into the JSON file.
    :param filename: Name of the file where the JSON will be saved.
    """
    try:
        with open(f'{location}/{filename}', 'w', encoding='utf-8') as file:
            # Using indent for pretty print, sort_keys to sort the keys,
            # and ensure_ascii=False to allow non-ASCII characters
            json.dump(dict_to_save, file, indent=4, sort_keys=True, ensure_ascii=False)
            # Add newline at the end of the file for better readability in some editors
            file.write('\n')
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")
    except TypeError as e:
        print(f"The dictionary contains objects that are not JSON serializable: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")