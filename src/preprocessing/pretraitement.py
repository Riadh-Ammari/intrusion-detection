import os
import cv2
from pathlib import Path

# Define paths
project_root = Path(__file__).parent.parent.parent
train_images_path = project_root / "data" / "dataset_pre" / "train" / "images"
val_images_path = project_root / "data" / "dataset_pre" / "val" / "images"

# Target image size (match train.py img_size)
target_size = (416, 416)

# Function to preprocess and overwrite image
def preprocess_image(image_path):
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not read {image_path}, skipping.")
        return
    
    # Resize image
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # Overwrite the original image
    cv2.imwrite(str(image_path), img_resized)
    print(f"Preprocessed and overwrote {image_path}")

# Process all images in train and val folders
for folder_path in [train_images_path, val_images_path]:
    if folder_path.exists():
        for image_file in folder_path.glob("*.[jp][pn][gf]"):  # Match .jpg, .jpeg, .png
            preprocess_image(image_file)
    else:
        print(f"Warning: Folder {folder_path} does not exist.")

print("Preprocessing completed.")