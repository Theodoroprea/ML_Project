import os
import pandas as pd
from PIL import Image
import requests
from io import BytesIO

# Ensure 'data' directory exists
data_directory = 'data'
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

# Load the dataset, get image urls and respective user ratings
dataset = pd.read_csv('dataset.csv') # Original dataset downloaded from kaggle.com
image_urls = dataset['image_url'].tolist()
ratings = dataset['avg_rating'].tolist()

total_images = len(image_urls)

# Function to resize image with aspect ratio preservation and padding
def resize_with_padding(img, target_size):
    original_width, original_height = img.size
    target_width, target_height = target_size

    # Calculate aspect ratio
    aspect_ratio = original_width / original_height

    # Calculate new dimensions with padding
    if aspect_ratio > 1:  # Landscape orientation
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:  # Portrait or square orientation
        new_width = int(target_height * aspect_ratio)
        new_height = target_height

    # Resize image
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    # Create a new blank image with the target size
    padded_img = Image.new("RGB", target_size, (0, 0, 0))  # Black background chosen to have all padding pixels have 0 intensity

    # Calculate the position to paste the resized image onto background
    paste_position = ((target_width - new_width) // 2, (target_height - new_height) // 2)

    # Paste the resized image onto the blank image
    padded_img.paste(resized_img, paste_position)

    return padded_img

# Resize images and save them in data directory
for i, (image_url, rating) in enumerate(zip(image_urls, ratings)):
    try:
        # Download image using image_url
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))

        # Resize image 224x224 with padding, the final dimensions can be changed if needed here
        target_size = (224, 224)
        padded_img = resize_with_padding(img, target_size)

        # Save resized image with rating encoded into title to simplify access during model operation
        filename = f'{data_directory}/boardgame_{i}_rating_{rating}.jpg'
        padded_img.save(filename)

          # Print progress
        progress = (i + 1) / total_images * 100
        print(f'[{i + 1}/{total_images}] Downloaded and resized {image_url}. Progress: {progress:.2f}%')
    except Exception as e:
        print(f'Error processing {image_url}: {str(e)}')

print('Image download and resize process completed.')