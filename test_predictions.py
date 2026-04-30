from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import random

# Model and Directory Paths
MODEL_PATH = r"G:\bv\bigvision_keypoint_project\runs\pose\train-5\weights\best.pt"
IMAGE_DIR = "test_images"

# Load model
model = YOLO(MODEL_PATH)

# Get random sample images
image_files = os.listdir(IMAGE_DIR)
sample_images = random.sample(
    image_files,
    max(5, len(image_files))
)

for img_name in sample_images:
    img_path = os.path.join(IMAGE_DIR, img_name)

    print(f"Testing: {img_name}")

    # Run inference
    results = model(img_path)

    # Read image
    image = cv2.imread(img_path)
    if image is None:
        print(f"Could not load: {img_name}")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get image dimensions
    h, w = image.shape[:2]

    # Dynamic circle radius based on image size
    # You can tune divisor (150~250)
    radius = max(4, int(min(w, h) / 150))

    print(f"Image Size: {w}x{h} | Circle Radius: {radius}")

    for result in results:
        if result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()

            for person_keypoints in keypoints:
                for x, y in person_keypoints:
                    if x > 0 and y > 0:
                        cv2.circle(
                            image,
                            (int(x), int(y)),
                            radius,
                            (255, 0, 0),
                            -1
                        )

    # Dynamic figure size for better display
    fig_size = max(6, min(w, h) / 150)

    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(image)
    plt.title(f"Testing: {img_name}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()