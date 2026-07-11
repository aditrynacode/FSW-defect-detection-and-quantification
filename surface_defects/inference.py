from ultralytics import YOLO
import cv2

MODEL_PATH = r"runs/detect/runs/surface_defect-16/weights/best.pt"
IMAGE_PATH = r"C:\Users\adity\FSW_Defect_Detection\dataset\surface_defects\unlabelled_images\DSC_0289.JPG"

model = YOLO(MODEL_PATH)

results = model.predict(
    source=IMAGE_PATH,
    conf=0.25,       # Confidence threshold
    save=False,      # Don't save output image
    verbose=False
)

result = results[0]

annotated_image = result.plot()

display = cv2.resize(annotated_image, None, fx=0.2, fy=0.2)

cv2.imshow("Prediction", display)
cv2.waitKey(0)
cv2.destroyAllWindows()

for box in result.boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])

    x1, y1, x2, y2 = box.xyxy[0].tolist()

    class_name = model.names[cls]

    print(f"Class      : {class_name}")
    print(f"Confidence : {conf:.3f}")
    print(f"Bounding Box: ({x1:.1f}, {y1:.1f}) ({x2:.1f}, {y2:.1f})")
    print("-" * 40)