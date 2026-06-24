from ultralytics import YOLO
from pathlib import Path
import yaml

def main():

    model = YOLO("yolov8n.pt")

    yaml_path = Path("dataset/surface_defects/weld.yaml")

    print("Exists:", yaml_path.exists())
    print("Absolute path:", yaml_path.resolve())

    with open(yaml_path, "r") as f:
        print("Raw file contents:")
        print(f.read())

    with open(yaml_path, "r") as f:
        print("Parsed YAML:")
        print(yaml.safe_load(f))

    model.train(
        data="dataset/surface_defects/weld.yaml",
        epochs=100,
        imgsz=640,
        batch=4,
        device=0,    
        workers=0,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        patience=20,
        project="runs",
        name="surface_defect",
        pretrained=True,
        save=True,
        verbose=True,

        mosaic=0.0,
        fliplr=0.5,
        translate=0.1,
        scale=0.3
    )

if __name__ == "__main__":
    main()