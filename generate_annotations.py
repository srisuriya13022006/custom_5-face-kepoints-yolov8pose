import cv2
import mediapipe as mp
import os
import pandas as pd
from tqdm import tqdm

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

# Folder paths
IMAGE_DIR = "dataset/raw_images"
OUTPUT_CSV = "dataset/annotations.csv"

# Ensure the output directory exists
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Selected landmark indices (Standard stable points)
LANDMARKS = {
    "left_eye": 33,
    "right_eye": 263,
    "nose_tip": 1,
    "left_mouth": 61,
    "right_mouth": 291
}

data = []

# Verify directory exists before listing
if not os.path.exists(IMAGE_DIR):
    print(f"Error: The directory {IMAGE_DIR} does not exist.")
else:
    image_files = os.listdir(IMAGE_DIR)[:2000]

    for img_name in tqdm(image_files, desc="Processing Images"):
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        image = cv2.imread(img_path)
        if image is None:
            continue

        h, w, _ = image.shape
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        # If no face is detected, skip the image
        if not results.multi_face_landmarks:
            continue

        face_landmarks = results.multi_face_landmarks[0]
        row = {"image": img_name}

        for name, idx in LANDMARKS.items():
            landmark = face_landmarks.landmark[idx]
            # Convert normalized coordinates to pixel coordinates
            x = int(landmark.x * w)
            y = int(landmark.y * h)

            row[f"{name}_x"] = x
            row[f"{name}_y"] = y

        data.append(row)

    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved annotations to {OUTPUT_CSV}")
        print(f"Total annotated images: {len(df)}")
    else:
        print("\nNo landmarks were detected in any of the images.")