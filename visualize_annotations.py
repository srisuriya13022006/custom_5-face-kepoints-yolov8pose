import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

IMAGE_DIR = "dataset/raw_images"
CSV_PATH = "dataset/annotations.csv"

# Load data
df = pd.read_csv(CSV_PATH)

# Select 5 random samples
samples = df.sample(5)

for _, row in samples.iterrows():
    img_path = os.path.join(IMAGE_DIR, row["image"])

    # Load and check image
    image = cv2.imread(img_path)
    if image is None:
        print(f"Skipping: {row['image']} (not found)")
        continue
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define keypoints from the CSV columns
    keypoints = [
        (row["left_eye_x"], row["left_eye_y"]),
        (row["right_eye_x"], row["right_eye_y"]),
        (row["nose_tip_x"], row["nose_tip_y"]),
        (row["left_mouth_x"], row["left_mouth_y"]),
        (row["right_mouth_x"], row["right_mouth_y"]),
    ]

    # Draw circles for each keypoint
    for x, y in keypoints:
        # We use int() because pixel coordinates must be whole numbers
        cv2.circle(image, (int(x), int(y)), 4, (255, 0, 0), -1)

    # Show the result for THIS specific iteration
    plt.figure(figsize=(6,6))
    plt.imshow(image)
    plt.title(row["image"])
    plt.axis("off")
    plt.show()