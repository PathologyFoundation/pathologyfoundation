import os
import requests
import appdirs


def download_file(url, folder_path):
    print("Start downloading ...")
    response = requests.get(url)
    if response.status_code == 200:
        filename = url.split("/")[-1].split("?")[0]  # Extracting the filename from the URL
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "wb") as file:
            file.write(response.content)
        
        print(f"Downloaded '{filename}' into '{folder_path}'.")
    else:
        print("Failed to download the file.")


def get_default_cache_dir():
    app_name = "PathologyFoundation"
    default_cache_dir = appdirs.user_cache_dir(app_name)
    return default_cache_dir