from ultralytics import YOLO
import os
import cv2
import shutil
from collections import Counter

MODEL_PATH = r"runs/detect/runs/surface_defect-6/weights/best.pt"
IMAGE_FOLDER = r"dataset/surface_defects/unlabelled_images"
LABEL_FOLDER = r"dataset/surface_defects/pseudo_labels"

if os.path.exists(LABEL_FOLDER):
    shutil.rmtree(LABEL_FOLDER)

CONFIDENCE_THRESHOLD = 0.2
prediction_counter = Counter()

os.makedirs(LABEL_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)

print(model.names)
print(model.model.nc)

extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

image_files = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith(extensions)
])

print(f"Found {len(image_files)} images.\n")

for image_name in image_files:

    image_path = os.path.join(IMAGE_FOLDER, image_name)

    results = model.predict(
        source=image_path,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    label_path = os.path.join(
        LABEL_FOLDER,
        os.path.splitext(image_name)[0] + ".txt"
    )

    with open(label_path, "w") as f:

        boxes = results[0].boxes

        if boxes is not None:

            for box in boxes:

                cls = int(box.cls.item())
                conf = float(box.conf.item())

                if conf < CONFIDENCE_THRESHOLD:
                    continue

                prediction_counter[cls] += 1

                x, y, w, h = box.xywhn[0].tolist()

                f.write(
                    f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                )

print("\nPredicted objects:")

for cls, count in sorted(prediction_counter.items()):
    print(f"{model.names[cls]} : {count}")

print("\nPseudo-label generation complete!")

output_dir = r"Pseudo_labels_archive"

class_names = [
    "Excess Flash",
    "Key Hole",
    "Grooves"
]

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.makedirs(output_dir, exist_ok=True)

train_out = os.path.join(output_dir, "obj_train_data")
os.makedirs(train_out, exist_ok=True)

image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

image_files = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith(image_extensions)
])

with open(os.path.join(output_dir, "train.txt"), "w") as train_list:

    for image in image_files:

        shutil.copy2(
            os.path.join(IMAGE_FOLDER, image),
            os.path.join(train_out, image)
        )

        label = os.path.splitext(image)[0] + ".txt"

        src_label = os.path.join(LABEL_FOLDER, label)
        dst_label = os.path.join(train_out, label)

        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
        else:
            open(dst_label, "w").close()

        train_list.write(f"obj_train_data/{image}\n")

with open(os.path.join(output_dir, "obj.names"), "w") as f:
    for cls in class_names:
        f.write(cls + "\n")

with open(os.path.join(output_dir, "obj.data"), "w") as f:
    f.write(f"classes = {len(class_names)}\n")
    f.write("names = obj.names\n")
    f.write("train = train.txt\n")
    f.write("backup = backup/\n")

print("Annotation folder created!")