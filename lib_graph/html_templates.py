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


def generate_index_file(files, cache_dir_base):
    ul = ''
    for f in files:
        date = find_date_pattern(f)
        min = find_min(f)
        base_name = os.path.splitext(f)[0]
        ul += f'<li><a href="{base_name}/index.html"><img src="{base_name}/icon.png">{date[0]} {date[1]}h {min}</a></li>\n'

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
    <ul>
        {ul}
    </ul>
</body>
</html>

        """

    save_html_file(html, f"{cache_dir_base}/index.html")


def generate_detail_html_file(file, cache_dir_base):
    ul = ''

    date = find_date_pattern(file)
    min = find_min(file)
    base_name = os.path.splitext(file)[0]
    ul += f'<li><a href="detail.html?folder={base_name}"><img src="{base_name}/icon.png">{date[0]} {date[1]}h {min}</a></li>\n'

    # print(ul)

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{date[0]} {date[1]}h {min}</title>
    <script src="main.js" defer></script>
    <link rel="stylesheet" href="main.css">
</head>
<body>
    <h1>Logs</h1>
    <ul>
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