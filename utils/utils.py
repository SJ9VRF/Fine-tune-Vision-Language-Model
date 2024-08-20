# Utility functions, like id_from_filename

from utils.utils import load_data, id_from_filename, list_files_in_directory

# Example of using load_data
questions, annotations = load_data('/path/to/questions.json', '/path/to/annotations.json')

# Example of using id_from_filename
file_id = id_from_filename('COCO_val2014_000000501080.jpg')
print(f"Extracted file ID: {file_id}")

# Example of listing files
image_files = list_files_in_directory('/path/to/images')
print(f"Found {len(image_files)} image files in directory.")
