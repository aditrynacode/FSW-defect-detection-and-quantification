from ultralytics import YOLO

def main():

    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset/weld.yaml",
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
        name="weld_defect",
        pretrained=True,
        save=True,
        verbose=True
    )

if __name__ == "__main__":
    main()