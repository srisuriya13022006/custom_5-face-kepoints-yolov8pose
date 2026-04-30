import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

# Folder paths
CSV_PATH = "dataset/annotations.csv"
IMAGE_DIR = "dataset/raw_images"

OUTPUT_IMAGE_TRAIN = "dataset/yolo_dataset/images/train"
OUTPUT_IMAGE_VAL = "dataset/yolo_dataset/images/val"

OUTPUT_LABEL_TRAIN = "dataset/yolo_dataset/labels/train"
OUTPUT_LABEL_VAL = "dataset/yolo_dataset/labels/val"

# Create directories
os.makedirs(OUTPUT_IMAGE_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_IMAGE_VAL, exist_ok=True)
os.makedirs(OUTPUT_LABEL_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_LABEL_VAL, exist_ok=True)

# Load annotations
df = pd.read_csv(CSV_PATH)

# Split dataset
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

def process_row(row, image_output_dir, label_output_dir):
    img_name = row["image"]
    img_path = os.path.join(IMAGE_DIR, img_name)

    if not os.path.exists(img_path):
        return

    # Open image to get dimensions
    image = Image.open(img_path)
    w, h = image.size

    # Extract keypoints
    keypoints = [
        (row["left_eye_x"], row["left_eye_y"]),
        (row["right_eye_x"], row["right_eye_y"]),
        (row["nose_tip_x"], row["nose_tip_y"]),
        (row["left_mouth_x"], row["left_mouth_y"]),
        (row["right_mouth_x"], row["right_mouth_y"]),
    ]

    xs = [kp[0] for kp in keypoints]
    ys = [kp[1] for kp in keypoints]

    # Calculate Bounding Box (tight fit around keypoints)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_center = ((x_min + x_max) / 2) / w
    y_center = ((y_min + y_max) / 2) / h
    box_width = (x_max - x_min) / w
    box_height = (y_max - y_min) / h

    # YOLO Pose Format: class x_center y_center width height k1_x k1_y k1_v ...
    label_data = f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"

    for x, y in keypoints:
        x_norm = x / w
        y_norm = y / h
        visibility = 2  # 2 means labeled and visible
        label_data += f" {x_norm:.6f} {y_norm:.6f} {visibility}"

    # Generate .txt file path
    base_name = os.path.splitext(img_name)[0]
    txt_path = os.path.join(label_output_dir, f"{base_name}.txt")

    # Save label file
    with open(txt_path, "w") as f:
        f.write(label_data)

    # Copy image to YOLO directory
    shutil.copy(img_path, os.path.join(image_output_dir, img_name))

# Run processing for training set
print("Processing training samples...")
for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
    process_row(row, OUTPUT_IMAGE_TRAIN, OUTPUT_LABEL_TRAIN)

# Run processing for validation set
print("Processing validation samples...")
for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
    process_row(row, OUTPUT_IMAGE_VAL, OUTPUT_LABEL_VAL)

print("\nYOLO Pose dataset prepared successfully!")
print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")