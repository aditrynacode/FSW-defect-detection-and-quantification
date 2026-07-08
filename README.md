# Friction Stir Weld Defect Detection and Quantification using YOLOv8

## Overview

This repository presents two computer vision pipelines developed for automated inspection of Friction Stir Welded (FSW) joints:

- **Internal Defect Detection** using Ultrasonic Non-Destructive Testing (NDT) S-scan images.
- **Surface Defect Detection** using weld surface images.

Developed during an internship at **IIT Goa**, the project aims to reduce manual inspection effort through deep learning-based defect detection and automated defect quantification.

---

## Features

- YOLOv8-based internal and surface defect detection
- Ultrasonic S-scan preprocessing and ROI extraction
- Semi-supervised annotation workflow for surface defect dataset
- Regression-based internal defect dimension estimation
- Surface defect quantification relative to weld geometry
- Transfer learning using pretrained YOLOv8 models

---

# Internal Defect Detection

The internal defect pipeline processes ultrasonic S-scan images to detect subsurface weld defects and estimate their physical dimensions.

## Pipeline

```text
Raw Ultrasonic S-Scan
        │
        ▼
Image Preprocessing
        │
        ▼
ROI Extraction
        │
        ▼
YOLOv8 Detection
        │
        ▼
Bounding Box Features
(cx, cy, w, h, area)
        │
        ▼
Regression Models
        │
        ▼
Width • Height • Depth
```

## Image Preprocessing

The preprocessing stage consists of:

- Image cropping
- Grayscale conversion
- Gaussian filtering
- CLAHE contrast enhancement
- ROI extraction
- Geometric normalization

## Defect Quantification

Bounding box features extracted from detected defects are used to estimate:

- Defect Width
- Defect Height
- Defect Depth

Linear regression models were evaluated using Leave-One-Out Cross Validation (LOOCV).

### Detection Performance

| Metric | Value |
|---------|------:|
| Precision | 1.000 |
| Recall | 0.912 |
| mAP@50 | 0.984 |
| mAP@50-95 | 0.511 |

### Quantification Performance

| Target | MAE | RMSE |
|---------|----:|-----:|
| Width | 0.7359 | 0.9714 |
| Height | 0.1491 | 0.2072 |
| Depth | 0.2205 | 0.3247 |

---

# Surface Defect Detection

The surface defect pipeline detects visible weld defects directly from weld surface images using YOLOv8.

## Semi-Supervised Annotation Workflow

The dataset consists of **1,145 weld surface images**.

Instead of manually annotating every image, a semi-supervised annotation strategy was adopted:

1. Manually annotate 300 images.
2. Train an initial YOLOv8 detector.
3. Automatically annotate 100-200 more images using the trained model.
4. Review and correct the generated annotations.
5. Retrain YOLOv8 using the initial + newly labeled dataset.
6. Repeat steps 3-5 until complete dataset is annotated.
7. Retrain YOLOv8 using the the complete labeled dataset.

This significantly reduced annotation effort while maintaining annotation quality.

## Surface Defect Quantification

Following defect detection, bounding boxes are used to quantify surface defects relative to the weld geometry.

Measurements include:

- Excess flash length relative to weld length
- Groove area relative to weld area
- Keyhole defect area relative to weld area

Using normalized measurements enables consistent comparison of weld quality across specimens of different sizes.

---

## Repository Structure

```text
FSW_Defect_Detection/
│
├── dataset/
│   ├── surface_defects/
│   │   ├── images/
│   │   ├── labels/
│   │   └── weld.yaml
│   │
│   └── internal_defects/
│       ├── images/
│       ├── labels/
│       ├── raw_images/
│       ├── quantification/
│       │   └── quantification_ds.csv
│       └── weld.yaml
│
├── internal_defects/
│   ├── preprocess.py
│   ├── train.py
│   └── linear_regressor.py
│
├── surface_defects/
│   └── train.py
│
├── requirements.txt
├── yolov8n.pt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<username>/<repository>.git
cd FSW_Defect_Detection

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

---

## Training

### Internal Defect Detection

```bash
python internal_defects/train.py
```

### Surface Defect Detection

```bash
python surface_defects/train.py
```

---

## Future Work

- Develop a desktop interface that accepts ultrasonic S-scan images and automatically predicts internal defect dimensions.
- Integrate the surface defect detection model with a real-time camera for live weld inspection.
- Investigate nonlinear regression and deep learning methods for improved defect quantification.
- Expand the datasets to improve detection robustness and quantification accuracy.

---

## Acknowledgements

Developed during an internship at **Indian Institute of Technology Goa** as part of research on automated weld inspection using computer vision and machine learning.

---

## Author

**Aditya Singh**

B.Tech Mechanical Engineering  
Indian Institute of Technology Goa