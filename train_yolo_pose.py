from ultralytics import YOLO

def main():
    model = YOLO("yolov8n-pose.pt")

    model.train(
        data="dataset.yaml",
        epochs=50,

        imgsz=896,        # slightly reduced
        batch=8,          # safer for 12GB GPU
        device=0,

        optimizer="AdamW",
        lr0=0.001,

        fliplr=0.5,
        degrees=20,
        translate=0.15,
        scale=0.5,

        multi_scale=False,   # IMPORTANT disable first

        perspective=0.0005,
        shear=2.0,
        erasing=0.2,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        mosaic=0.0,

        patience=25,
        workers=4,

        cache="disk",    # much safer than RAM
        amp=True
    )

if __name__ == "__main__":
    main()