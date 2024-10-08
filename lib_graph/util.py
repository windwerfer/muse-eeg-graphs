import re
from PIL import Image

def find_min(text):
    pattern = r'\d+min'
    # Use re.findall to find all non-overlapping matches of pattern in string
    matches = re.findall(pattern, text)
    if len(matches)>0:
        return f"({matches[0]})"

def find_date_pattern(text):
    """
    Find all occurrences of the date pattern YYYY.MM.DD in the given text.

    Parameters:
    text (str): The string to search for date patterns.

    Returns:
    list: A list of all matches found in the text.
    """
    # Define the pattern.
    # \d{4} matches four digits, \. matches a literal dot, \d{2} matches two digits
    pattern = r'\d{4}\.\d{2}\.\d{2}_\d{2}\.\d{2}'

    # Use re.findall to find all non-overlapping matches of pattern in string
    matches = re.findall(pattern, text)
    if len(matches)>0:
        datetime = matches[0].split('_')
        date = datetime[0].split('.')
        datetime[0] = f"{date[2]}.{date[1]}"
        datetime[1] = datetime[1].replace('.', ':')
        return datetime

def generate_img_thumbnail(file_name,thumb_name):
    # Your existing plotting code...


    # Load the image you just saved
    img = Image.open(f'{file_name}')

    # Define the width of your thumbnail
    thumb_width = 100

    # Calculate the height to maintain aspect ratio
    thumb_height = int((thumb_width / img.width) * img.height)

    # Resize the image to thumbnail size
    img.thumbnail((thumb_width, thumb_height))

    # Save the thumbnail
    img.save(f'{thumb_name}', 'PNG')

