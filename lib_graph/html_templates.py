import json
import os

from lib_graph.util import find_date_pattern, find_min


def save_html_file(html: str, file: str) -> None:
    """
    Saves the provided HTML content to a file with UTF-8 encoding.

    Args:
        html (str): The HTML content as a string to save.
        file (str): The filename or path where the HTML should be saved.

    Returns:
        None

    Example:
        save_html_file("<html><body>Hello World!</body></html>", "index.html")
    """
    try:
        with open(file, 'w', encoding='utf-8') as f:
            # Write the HTML content to the file
            f.write(html)
        # print(f"HTML content saved successfully to {file}")
    except IOError as e:
        print(f"An error occurred while saving the file: {e}")


def load_and_validate_json(filepath: str):
    """
    Opens a file encoded in UTF-8, reads its JSON content,
    and validates it.

    Args:
    filepath (str): The path to the JSON file.

    Returns:
    Union[Dict[str, Any], None]: A dictionary representing the JSON data if valid,
    or None if the JSON is invalid or if any error occurred during file operations.

    Raises:
    FileNotFoundError: If the specified file does not exist.
    json.JSONDecodeError: If the file content is not valid JSON.
    """
    try:
        # Open the file with UTF-8 encoding
        with open(filepath, 'r', encoding='utf-8') as file:
            # Attempt to load the JSON
            data = json.load(file)

            # If we've reached here, the JSON is valid
            return data

    except FileNotFoundError:
        print(f"The file {filepath} was not found.")
        return None

    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        return None

    except UnicodeDecodeError:
        print("The file is not encoded in UTF-8 or contains invalid UTF-8 sequences.")
        return None

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def extract_keys(d: dict) -> list:
    """
    Extracts and returns the keys from the given dictionary.

    Args:
    d (dict): The dictionary from which to extract keys.

    Returns:
    list: A list containing all the keys from the dictionary.

    Example:
    >>> extract_keys({'a': 1, 'b': 3})
    ['a', 'b']
    """
    # Initialize an empty list to store the keys
    keys = []


    # Loop through each key in the dictionary
    for key in d:
        # Append each key to the list
        keys.append(key)

    # Return the list of keys
    return keys

def extract_statistics_summary(file):

    data = load_and_validate_json(file)
    if data is None:
        return None

    good_el = extract_keys(data['table_good_electrodes'])
    bad_el = extract_keys(data['table_bad_electrodes'])

    ret = {
        'peak_alpha_welch': data['peak_alpha_welch'],
        'peak_alpha_welch4s': data['peak_alpha_welch4s'],
        'peak_alpha_window': data['peak_alpha_window'],
        'good_electrodes': good_el,
        'bad_electrodes': bad_el,
        }

    return ret

def extract_statistics_detail(file):

    data = load_and_validate_json(file)

    return data


def generate_index_file(files, cache_dir_base):
    ul = ul_pa = ''
    stats_summery = []
    for f in files:
        date = find_date_pattern(f)
        mi = find_min(f)
        base_name = os.path.splitext(f)[0]
        s = extract_statistics_summary(f"{cache_dir_base}/{base_name}/statistics.json")
        if s is None:
            continue
        stats_summery.append(s)
        if len(s['bad_electrodes']) > 0:
            sie = ', '.join(s['bad_electrodes'])
            bad_el = f'(ignore weak signal in electrode: <b>{sie}</b>)'
        else:
            bad_el = ''
        ul += f'<li><a href="{base_name}/index.html"><img src="{base_name}/icon.png">{date[0]} {date[1]}h {mi}</a>{bad_el}</li>\n'

        ul_pa += f"<li><span>{date[0]} {date[1]}h {mi}</span><span>{s['peak_alpha_welch']['mean_peak_alpha']}</span><span>{s['peak_alpha_welch4s']['mean_peak_alpha']}</span><span>{s['peak_alpha_window']['mean_peak_alpha']}</span></li>\n"



    # print(ul)
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>overview</title>
    <script src="main.js" defer></script>
    <link rel="stylesheet" href="main.css">
</head>
<body>
    <h1>Logs</h1>
    
    <ul class='peak_alpha'>
        <li><span>File</span><span>Peak Alpha Welch 1s</span><span>Peak Alpha Welch 4s</span><span>Peak Alpha Window method</span></li>
        {ul_pa}
    </ul>
    <ul class='img_list'>
        {ul}
    </ul>
</body>
</html>

        """

    save_html_file(html, f"{cache_dir_base}/index.html")

def generate_css_file(cache_dir_base):
    css = """
a {
    /* Remove the underline */
    text-decoration: none;
    /* Set the color to 90% gray, which would be #E6E6E6 in hexadecimal */
    color: #1A1A1A;
}
/* Set the width of the ul to be 50% of the screen width */
ul.peak_alpha {
    width: 70%;
    display: table; /* Behaves like a table */
    table-layout: fixed; /* Ensures equal width columns */
    padding: 0; /* Remove default padding */
}

/* Each li acts as a table row */
ul.peak_alpha li {
    display: table-row;
    list-style-type: none; /* Remove bullet points */
}

/* Each span within li acts as a table cell */
ul.peak_alpha li span {
    display: table-cell;
    width: 25%; /* Each column takes up 25% of the ul width */
    border: 1px solid #ddd; /* Optional: for visual separation of cells */
    padding: 5px; /* Optional: adds some space inside each cell */
    text-align: center; /* Optional: centers the text in each cell */
}

/* Set the width of the ul to be 50% of the screen width */
ul.detail_peak_alpha {
    width: 70%;
    display: table; /* Behaves like a table */
    table-layout: fixed; /* Ensures equal width columns */
    padding: 0; /* Remove default padding */
}

/* Each li acts as a table row */
ul.detail_peak_alpha li {
    display: table-row;
    list-style-type: none; /* Remove bullet points */
}

/* Each span within li acts as a table cell */
ul.detail_peak_alpha li span {
    display: table-cell;
    width: 16.5%; /* Each column takes up 16.5% of the ul width */
    border: 1px solid #ddd; /* Optional: for visual separation of cells */
    padding: 5px; /* Optional: adds some space inside each cell */
    text-align: center; /* Optional: centers the text in each cell */
}
ul.img_details img {
	 height: 400px;
}

ul.img_list li {
    /* Removes the bullets from the list items */
    list-style-type: none;
    /* Optionally, you might want to remove padding or margins */
    padding: 5px;
    margin: 5px;
}
ul.img_list img {
    vertical-align:middle;
	padding: 5px;

}

    
    
    """
    save_html_file(css, f"{cache_dir_base}/main.css")

def table_detail_peak_alpha(pa):
    ret = ''
    for i, p in enumerate(pa):
        minute = p['periode_start'] // 60
        ret += f"""<li><span>{minute}</span>
                         <span>{p['mean_peak_alpha']}</span>
                         <span>{p['peak_aplhas']['tp9']}</span>
                         <span>{p['peak_aplhas']['af7']}</span>
                         <span>{p['peak_aplhas']['af8']}</span>
                         <span>{p['peak_aplhas']['tp10']}</span></li>\n"""
    return ret


def generate_detail_html_file(file, cache_dir_base):
    ul = ul_pa = ''

    date = find_date_pattern(file)
    mi = find_min(file)
    base_name = os.path.splitext(file)[0]

    s = extract_statistics_detail(f"{cache_dir_base}/{base_name}/statistics.json")
    if s is None:
        return

    ul += f'<li><a href="detail.html?folder={base_name}"><img src="{base_name}/icon.png">{date[0]} {date[1]}h {mi}</a></li>\n'

    ul_pa_welch = table_detail_peak_alpha(s['periods_peak_alpha_welch'])


    # print(ul)

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{date[0]} {date[1]}h {mi}</title>
    <script src="main.js" defer></script>
    <link rel="stylesheet" href="../main.css">
</head>
<body>
    <h1>Logs</h1>
    
    <ul class='detail_peak_alpha'>
        <li><span>time (min)</span><span>PA mean</span><span>PA tp9</span><span>PA af7</span><span>PA af8</span><span>PA tp10</span></li>
        {ul_pa_welch} 
    <ul>    
    <ul class='img_details'>
        <li><img src='plot_powerbands_hilbert_envelope_moveing_average_1.png'></li>
        <li><img src='plot_frequency_domain_1.png'></li>
        <li><img src='plot_time_frequency_analysis_1.png'></li>
        <li><img src='plot_psd__power_spectral_density_1.png'></li>
        <li><img src='plot_amplitude_distribution_histogram_1.png'></li>
        <li><img src='plot_powerbands_hilbert_envelope_1.png'></li>
    </ul>
</body>
</html>

        """
    save_html_file(html, f"{cache_dir_base}/{base_name}/index.html")
    # return html