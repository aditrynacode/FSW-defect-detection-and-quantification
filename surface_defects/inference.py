from ultralytics import YOLO
import cv2

MODEL_PATH = r"runs/detect/runs/surface_defect-16/weights/best.pt"
IMAGE_PATH = r"C:\Users\adity\FSW_Defect_Detection\dataset\surface_defects\unlabelled_images\DSC_0289.JPG"

model = YOLO(MODEL_PATH)
model.names[3] = "Weld Junction"

results = model.predict(
    source=IMAGE_PATH,
    conf=0.4,       # Confidence threshold
    save=False,      # Don't save output image
    verbose=False
)

result = results[0]

annotated_image = result.plot()

display = cv2.resize(annotated_image, None, fx=0.2, fy=0.2)

cv2.imshow("Prediction", display)
cv2.waitKey(0)
cv2.destroyAllWindows()

measurements = {}

for box in result.boxes:
    cls = int(box.cls[0])
    class_name = model.names[cls]

    x1, y1, width, height = box.xywh[0].tolist()

    length = max(width, height)
    breadth = min(width, height)
    area = width * height

    if class_name not in measurements:
        measurements[class_name] = []

    measurements[class_name].append({
        "length": length,
        "width": breadth,
        "area": area
    })

print("\nQuantification Results:\n")

if "Weld Jumction" not in measurements:
    print("Weld Junction not detected. Cannot quantify.")
else:
    weld = measurements["Weld Jumction"][0]

    defect_classes = [
        "Excess Flash",
        "Key Hole",
        "Grooves"
    ]

    for defect in defect_classes:
        if defect in measurements:
            for idx, d in enumerate(measurements[defect], start=1):

                print(f"{defect} #{idx}:")

                if defect == "Key Hole":

                    dia_ratio = d['width'] / weld['width']
                    area_ratio = d['area'] / weld['area']

                    print(f"  Diameter/Width Ratio = {dia_ratio:.4f}")
                    print(f"  Diameter in mm = {dia_ratio:.4f}*20 mm = {dia_ratio*20:.4f} mm")
                    print(f"  Area Ratio = {area_ratio:.4f}")
                    print(f"  Area in mm² = {area_ratio:.4f}*2000 mm² = {area_ratio*2000:.4f} mm²")

                else:

                    len_ratio = d['length'] / weld['length']
                    width_ratio = d['width'] / weld['width']
                    area_ratio = d['area'] / weld['area']

                    print(f"  Length Ratio = {len_ratio:.4f}")
                    print(f"  Length in mm = {len_ratio:.4f}*100 mm = {len_ratio*100:.4f} mm")
                    print(f"  Width Ratio = {width_ratio:.4f}")
                    print(f"  Width in mm = {width_ratio:.4f}*20 mm = {width_ratio*20:.4f} mm")
                    print(f"  Area Ratio = {area_ratio:.4f}")
                    print(f"  Area in mm² = {area_ratio:.4f}*2000 mm² = {area_ratio*2000:.4f} mm²")

                print("-" * 40)

        else:
            print(f"{defect}: Not detected")
            print("-" * 40)