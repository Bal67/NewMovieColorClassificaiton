import pandas as pd
import os
import requests
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_data_from_github(url):
    return pd.read_csv(url)

def preprocess_data(df):
    # Drop rows with missing 'Poster' URLs
    df = df.dropna(subset=['Poster'])
    columns_to_drop = ['imdbld', 'Imdb Link', 'Title', 'IMDB Score']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    return df

def download_image(url):
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None

def save_image_locally(img, directory, filename):
    os.makedirs(directory, exist_ok=True)
    img_path = os.path.join(directory, filename)
    img.save(img_path, 'JPEG')

def download_and_save_image(row, directory):
    img = download_image(row['Poster'])
    if img:
        # Create a unique filename using the 'md5hash' or another unique identifier
        filename = f"{row['md5hash']}.jpg"
        save_image_locally(img, directory, filename)
        return True
    return False

def filter_rows_with_unavailable_images(df, directory):
    available_rows = []

    def check_image_availability(row):
        if download_and_save_image(row, directory):
            available_rows.append(row)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(check_image_availability, row) for _, row in df.iterrows()]
        for future in as_completed(futures):
            future.result()

    return pd.DataFrame(available_rows)

def save_data(df, filename, directory):
    os.makedirs(directory, exist_ok=True)
    df.to_csv(os.path.join(directory, filename), index=False)

if __name__ == "__main__":
    github_url = 'https://raw.githubusercontent.com/Bal67/NewMovieColorClassification/main/data/MovieGenre.csv'
    save_directory = '/content/drive/MyDrive/MovieColorClassification/NewMovieColorClassification/images'  # Specify the Google Drive folder for images
    data_save_directory = '/content/drive/MyDrive/MovieColorClassification/NewMovieColorClassification/data'  # Specify the Google Drive folder for data

    # Load the data from GitHub
    df = load_data_from_github(github_url)

    # Preprocess the data (drop rows with missing Poster URLs)
    df = preprocess_data(df)
    
    # Filter rows where images cannot be downloaded and save available images locally
    df = filter_rows_with_unavailable_images(df, save_directory)
    
    # Save the processed DataFrame to a CSV file
    save_data(df, 'moviecolorclassification_processed.csv', data_save_directory)
