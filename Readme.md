# README — How to Run This Project (Colab Guide)

## 1. Open in Google Colab

* Open the provided notebook (`.ipynb`) in Google Colab
* Make sure runtime is enabled

---

## 2. Enable Runtime (GPU Recommended)

Go to:

```text
Runtime → Change runtime type → Hardware accelerator → GPU
```

If GPU is not available, you can use CPU:

```python
device = "cpu"
```

Else:

```python
device = 0   # GPU
```

---

## 3. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 4. Dataset Setup
* Download the dataset from provide link in colab
* Upload or place your dataset inside your Google Drive
* Example path:

```python
DATASET_PATH = "/content/drive/MyDrive/your_folder/dataset/raw_images"
```

⚠️ IMPORTANT:
Update all dataset paths in the notebook according to your Drive location.

---

## 5. Install Required Libraries

Run the setup cell:

```python
!pip install ultralytics mediapipe opencv-python pandas matplotlib tqdm scikit-learn
```

---

## 6. Run Notebook Step-by-Step

Execute all phases in order:

1. Dataset loading
2. Remove corrupted images
3. Generate annotations
4. Visualize annotations
5. Convert to YOLO format
6. Train model

⚠️ Do NOT skip steps.

---

## 7. Model Training

Training will automatically start using:

```python
model.train(...)
```

* Model will be saved in:

```text
runs/pose/bigvision_face_keypoints_final/
```

---

## 8. Use Trained Model (50 Epoch Model)

You can directly use the trained model from:

```text
enhanced_model/best.pt
```

---

## 9. Run Prediction

```python
from ultralytics import YOLO

model = YOLO("enhanced_model/best.pt")

results = model.predict(
    source="path_to_test_images",
    save=True,
    conf=0.25
)
```

---

## 10. View Results

* Output images will be saved in:

```text
runs/pose/predict/
```

* Pre-generated results are available in:

```text
prediction_results/
```

---

## 11. Important Notes

* Update all paths based on your Drive location
* Use GPU for faster training

---

## Done ✅

You can now:

* Train your own model
* Test on new images
* Visualize predictions

---
