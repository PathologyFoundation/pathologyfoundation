import os
from ..utils import get_default_cache_dir
import requests
import gdown
import pandas as pd
import zipfile


def download_file(url, folder_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        # Get the filename from the Content-Disposition header
        content_disposition = response.headers.get("content-disposition")
        if content_disposition:
            filename = content_disposition.split("filename=")[1].strip('"')
        else:
            filename = os.path.basename(url)
        
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        print(f"Downloaded '{filename}' into '{folder_path}'.")
    else:
        print("Failed to download the file.")

def download_google_drive_file(file_id, output_path):
    if os.path.exists(output_path):
        print(f"File '{output_path}' already exists. Skipping download.")
        return False
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)
    return True

def unzip_file(zip_file_path, extract_folder):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

def load_example(dataset_name="CRC-VAL-HE-7K",
                 download_folder=None):
    

    if download_folder is None:
        download_folder = os.path.join(get_default_cache_dir(), "example_data")
    os.makedirs(download_folder, exist_ok=True)

    if dataset_name == "CRC-VAL-HE-7K":
        file_id = "1nzLEA0daP8cs3I5yw_Na8rtWXBM9nogr"
        output_path = os.path.join(download_folder, "CRC-VAL-HE-7K.zip")
        newly_download = download_google_drive_file(file_id, output_path)
        extract_folder = os.path.join(download_folder, dataset_name)
        if newly_download:
            os.makedirs(extract_folder, exist_ok=True)
            unzip_file(output_path, extract_folder)
        
        dataset_folder = os.path.join(extract_folder, "CRC-VAL-HE-7K")

        data = []
        class_mapping = {
            "ADI": 0, "BACK": 1, "DEB": 2, "LYM": 3,
            "MUC": 4, "MUS": 5, "NORM": 6, "STR": 7, "TUM": 8
        }

        for class_name in class_mapping.keys():
            class_path = os.path.join(dataset_folder, class_name)
            if os.path.exists(class_path):
                for root, _, files in os.walk(class_path):
                    for file_name in files:
                        image_path = os.path.join(root, file_name)
                        data.append((image_path, class_name, class_mapping[class_name]))
        
        df = pd.DataFrame(data, columns=["image", "class", "label"])
    
    return df
        
