import cv2
import numpy as np
import os


def warp_to_paper_triangle(
    img,
    output_width=469,
    output_height=252
):

    V1 = (637, 0)      # Top left
    V2 = (743, 0)      # Top right
    V2 = (1327, 692)     # Bottom Left
    V3 = (1743, 309)     # Bottom Right

    src_pts = np.float32([V1, V2, V3])

    dst_pts = np.float32([
        [234, 1],
        [1, 251],
        [468, 251]
    ])

    M = cv2.getAffineTransform(
        src_pts,
        dst_pts
    )

    warped = cv2.warpAffine(
        img,
        M,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC
    )

    # trianglular mask

    mask = np.zeros(
        (output_height, output_width),
        dtype=np.uint8
    )

    triangle = np.array([
        [234, 1],
        [1, 251],
        [468, 251]
    ], dtype=np.int32)

    cv2.fillPoly(
        mask,
        [triangle],
        255
    )

    warped_masked = cv2.bitwise_and(
        warped,
        warped,
        mask=mask
    )

    return warped_masked


def process_dataset(
    source_root,
    destination_root
):

    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    for root, dirs, files in os.walk(source_root):

        relative_path = os.path.relpath(
            root,
            source_root
        )

        save_dir = os.path.join(
            destination_root,
            relative_path
        )

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        for file in files:

            if not file.lower().endswith(valid_extensions):
                continue

            img_path = os.path.join(
                root,
                file
            )

            img = cv2.imread(img_path)

            if img is None:
                print(f"Could not read: {img_path}")
                continue

            warped = warp_to_paper_triangle(img)

            save_path = os.path.join(
                save_dir,
                file
            )

            cv2.imwrite(
                save_path,
                warped
            )

            print(f"Saved: {save_path}")


SOURCE_ROOT = r"dataset\raw_images"
DESTINATION_ROOT = r"dataset\images"

process_dataset(SOURCE_ROOT, DESTINATION_ROOT)

print("\nProcessing Complete.")