import os

def save_file(content_to_save, file_name):
    directory = os.path.dirname(file_name)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(file_name, "w") as text_file:
        text_file.write(content_to_save)