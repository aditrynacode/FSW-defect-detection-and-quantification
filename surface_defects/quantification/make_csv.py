import os
import pandas as pd

labels_folder = r"C:\Users\adity\FSW_Defect_Detection\dataset\surface_defects\labels\train"
dataset_folder = r"C:\Users\adity\FSW_Defect_Detection\dataset\surface_defects"

output_csv = os.path.join(dataset_folder, "train_labels.csv")

class_mapping = {
    0: ("excess_flash_width", "excess_flash_height"),
    1: ("key_hole_width", "key_hole_height"),
    2: ("grooves_width", "grooves_height"),
    3: ("weld_junction_width", "weld_junction_height"),
}

rows = []

for filename in os.listdir(labels_folder):
    if not filename.endswith(".txt"):
        continue

    txt_path = os.path.join(labels_folder, filename)
    image_name = os.path.splitext(filename)[0]

    # Initialize one row for this image
    row = {
        "image": image_name,
        "excess_flash_width": None,
        "excess_flash_height": None,
        "key_hole_width": None,
        "key_hole_height": None,
        "grooves_width": None,
        "grooves_height": None,
        "weld_junction_width": None,
        "weld_junction_height": None,
    }

    with open(txt_path, "r") as f:
        for line in f:
            values = line.strip().split()

            if len(values) != 5:
                print(f"Skipping malformed line in {filename}: {line}")
                continue

            class_id = int(values[0])
            width = float(values[3])
            height = float(values[4])

            if class_id in class_mapping:
                width_col, height_col = class_mapping[class_id]
                row[width_col] = width
                row[height_col] = height

    rows.append(row)

df = pd.DataFrame(rows)

df = df.sort_values("image").reset_index(drop=True)

df.to_csv(output_csv, index=False)

print(f"CSV saved to: {output_csv}")
print(f"Total images processed: {len(df)}")