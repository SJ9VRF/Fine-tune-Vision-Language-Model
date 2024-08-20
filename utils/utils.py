# Utility functions, like id_from_filename

import json
import re
from os import listdir
from os.path import isfile, join

def id_from_filename(filename):
    """
    Extracts the numerical ID from a filename using a regex pattern.
    """
    filename_re = re.compile(r".*(\d{12})\.((jpg)|(png))")
    match = filename_re.fullmatch(filename)
    if match is None:
        return None
    return int(match.group(1))

def load_data(question_file, annotation_file):
    """
    Loads the question and annotation data from JSON files.
    """
    with open(question_file, 'r') as f:
        data_questions = json.load(f)

    with open(annotation_file, 'r') as f:
        data_annotations = json.load(f)

    return data_questions['questions'], data_annotations['annotations']

def list_files_in_directory(directory_path):
    """
    Lists all files in a given directory.
    """
    return [f for f in listdir(directory_path) if isfile(join(directory_path, f))]
"""

from utils.utils import load_data, id_from_filename, list_files_in_directory

# Example of using load_data
questions, annotations = load_data('/path/to/questions.json', '/path/to/annotations.json')

# Example of using id_from_filename
file_id = id_from_filename('COCO_val2014_000000501080.jpg')
print(f"Extracted file ID: {file_id}")

# Example of listing files
image_files = list_files_in_directory('/path/to/images')
print(f"Found {len(image_files)} image files in directory.")

"""
