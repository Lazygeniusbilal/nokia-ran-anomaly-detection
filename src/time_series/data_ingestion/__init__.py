import os
from dotenv import load_dotenv
import gdown
import zipfile
import pandas as pd
from pathlib import Path


def data_ingestion(url: str):
    os.makedirs('data/zip', exist_ok=True)
    return gdown.download_folder(url=url, output='data/zip', quiet=False, use_cookies=False)

def unzip_data(zip_dir: Path) -> None:
    # get the data and unzip it
    # make directory to save data
    os.makedirs('data/unzip', exist_ok=True)
    # get the path of the file
    for zip_path in Path(zip_dir).glob('*.zip'):
        with zipfile.ZipFile(file=zip_path, mode='r') as file:
            file.extractall('data/unzip')

if __name__ == "__main__":
    # load the env vars
    load_dotenv()
    google_drive= os.getenv('GOOGLE_DRIVE')
    data_ingestion(url= google_drive)
    unzip_data(zip_dir='data/zip')