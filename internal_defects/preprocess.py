import cv2
import numpy as np
import os

src = cv2.imread(r"C:\Users\adity\FSW_Defect_Detection\dataset\internal_defects\raw_images\train\BI19-D3.png")

if src is None:
    raise ValueError(f"Could not load image.")

height, width = src.shape[:2]

src = src[
    0:src.shape[0]-34,
    0:src.shape[1]-170
]

gray = cv2.cvtColor(
    src,
    cv2.COLOR_BGR2GRAY
)

blurred = cv2.GaussianBlur(
    gray,
    (5, 5),
    1.5
)

clahe = cv2.createCLAHE(
    clipLimit=1.0,
    tileGridSize=(8, 8)
)

enhanced = clahe.apply(
    blurred
)

_, thresh = cv2.threshold(
    enhanced,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if len(contours) == 0:
    raise RuntimeError(
        "No contours found"
    )

largest = max(
    contours,
    key=cv2.contourArea
)

hull = cv2.convexHull(
    largest
)

area, triangle = cv2.minEnclosingTriangle(
    hull
)

triangle = triangle.reshape(
    3,
    2
)

triangle = np.int32(
    triangle
)

triangle = triangle[
    np.argsort(
        triangle[:, 1]
    )
]

top = triangle[0]

bottom = triangle[1:]

bottom = bottom[
    np.argsort(
        bottom[:, 0]
    )
]

V1 = tuple(top)         # Apex
V2 = tuple(bottom[0])   # Bottom Left
V3 = tuple(bottom[1])   # Bottom Right

# We only calculate vertices once as all the images are of same composition, hence we need not compute them again and again for every image

def warp_to_paper_triangle(
    img,
    V1, V2, V3,
    output_width=469,
    output_height=252
):
    """
    V1 = (637, 0)      # Top left
    V2 = (1327, 692)     # Bottom Left
    V3 = (1743, 309)     # Bottom Right
    """
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

    remove_triangle = np.array([
        [234, 1],    # top
        [198, 40],   # bottom left
        [271, 40]    # bottom right
    ], dtype=np.int32)

    remove_mask = np.zeros(
        (output_height, output_width),
        dtype=np.uint8
    )

    cv2.fillPoly(
        remove_mask,
        [remove_triangle],
        255
    )
    warped_masked[remove_mask == 255] = 0

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

            img = img[
                0:img.shape[0]-34,
                0:img.shape[1]-170
            ]
            
            warped = warp_to_paper_triangle(img, V1, V2, V3)

            save_path = os.path.join(
                save_dir,
                file
            )

            cv2.imwrite(
                save_path,
                warped
            )

            print(f"Saved: {save_path}")


SOURCE_ROOT = r"dataset\internal_defects\raw_images"
DESTINATION_ROOT = r"dataset\internal_defects\images"

process_dataset(SOURCE_ROOT, DESTINATION_ROOT)

print("\nProcessing Complete.")