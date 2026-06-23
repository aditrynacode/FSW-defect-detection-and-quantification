# Friction Stir Weld Defect Detection and Quantification using YOLOv8 and Ultrasonic NDT Imaging

## Overview

This project presents an end-to-end computer vision pipeline for automated defect detection and dimensional quantification in Friction Stir Welded (FSW) joints using Ultrasonic Non-Destructive Testing (NDT) S-scan images.

The system combines image preprocessing, region-of-interest extraction, defect localization using YOLOv8, and regression-based dimensional estimation to identify weld defects and predict their physical dimensions from ultrasonic inspection data.

Developed during an internship at IIT Goa, the project aims to reduce dependence on manual inspection while providing a scalable framework for automated weld quality assessment.

---

## Key Features

- Automated preprocessing of ultrasonic S-scan images
- Triangular ROI extraction and geometric transformation
- YOLOv8-based defect detection and localization
- Manual and semi-automatic annotation workflow
- Extraction of defect geometric features from detected bounding boxes
- Regression-based estimation of:
  - Defect Width
  - Defect Height
  - Defect Depth
- Leave-One-Out Cross Validation (LOOCV) based model evaluation

---

## System Pipeline

```text
Raw Ultrasonic S-Scan
          │
          ▼
Image Preprocessing
          │
          ▼
ROI Extraction & Enhancement
          │
          ▼
YOLOv8 Defect Detection
          │
          ▼
Bounding Box Feature Extraction
(cx, cy, w, h, area)
          │
          ▼
Regression Models
          │
          ▼
Defect Width
Defect Height
Defect Depth
```

---

## Dataset

The dataset consists of ultrasonic inspection scans collected from Friction Stir Welded specimens.

The dataset contains both defective and defect-free weld scans to improve detector robustness and reduce false positives.

### Quantification Dataset

A separate quantification dataset was created containing:

- Bounding box features extracted from defect regions
- Measured defect dimensions obtained from inspection records

Features investigated include:

- Bounding box center coordinates (`cx`, `cy`)
- Bounding box dimensions (`w`, `h`)
- Bounding box area

---

## Image Preprocessing

A custom preprocessing pipeline was developed to isolate relevant ultrasonic information and improve defect visibility.

### Processing Steps

1. Cropping to remove Scales
2. Grayscale Conversion
3. Gaussian Noise Reduction
4. Contrast Enhancement using CLAHE
5. Thresholding and Contour Detection
6. ROI Identification 
7. Triangular ROI Extraction using OpenCV's `minEnclosingTriangle()`
8. Geometric Transformation and Normalization

The resulting images contain only the relevant ultrasonic inspection region used for training and inference.

---

## Data Annotation

Defect regions were annotated using YOLO-format bounding boxes.

### Annotation Workflow

#### Manual Annotation

- Expert-guided defect localization
- Bounding box generation for training data
- Quality verification of labels

#### Semi-Automatic Annotation

Several computer vision approaches were explored:

- Thresholding
- Contour Detection
- Edge-Based Segmentation
- ROI-Based Candidate Extraction

These methods were explored to accelerate dataset annotation, but manual annotation was used in the end because of insufficient technical accuracy in the automatic approaches.

---

## Defect Detection Model

### YOLOv8 Nano

The detection stage utilizes the YOLOv8 Nano architecture.

#### Advantages

- Lightweight architecture
- Fast training and inference
- Strong performance on limited dataset
- Suitable for future deployment applications

### Transfer Learning

Training was performed using pretrained YOLOv8 weights to improve convergence and detection performance, given the constraint of a small dataset.

---

## Defect Quantification

Following defect localization, bounding box features are extracted and used as inputs to regression models.

### Target Variables

- Defect Width
- Defect Height
- Defect Depth

### Regression Workflow

```text
Detected Bounding Box
        │
        ▼
Feature Extraction
(cx, cy, w, h, area)
        │
        ▼
Feature Selection
        │
        ▼
Linear Regression Models
        │
        ▼
Dimension Estimation
```

Feature importance studies were conducted to determine the most informative geometric features for each target dimension.

---

## Training Configuration

| Parameter | Value |
|------------|---------|
| Model | YOLOv8n |
| Image Size | 640 × 640 |
| Batch Size | 4 |
| Epochs | 100 |
| Framework | Ultralytics YOLOv8 |
| Hardware | NVIDIA RTX 3050 Laptop GPU |

---

## Detection Results

| Metric | Value |
|----------|----------|
| Precision | 1 |
| Recall | 0.912 |
| mAP@50 | 0.984 |
| mAP@50-95 | 0.511 |

These results demonstrate reliable defect localization despite a relatively limited training dataset.

---

## Quantification Results

Defect dimension estimation models were evaluated using Leave-One-Out Cross Validation (LOOCV).

Performance metrics reported include:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

The regression framework enables automated estimation of physical defect dimensions directly from detected ultrasonic indications.

| Target | MAE | RMSE |
|---------|---------|---------|
| Width | 0.7359 | 0.9714 |
| Height | 0.1491 | 0.2072 |
| Depth | 0.2205 | 0.3247 |

The moderate R² values obtained are largely attributable to the limited dataset size (49 samples). With such a small number of observations, regression models are more susceptible to noise, outliers, and overfitting, making highly accurate dimensional prediction challenging. Increasing the size and diversity of the quantification dataset is expected to significantly improve performance.

---

## Repository Structure

```text
FSW_Defect_Detection/
│
├── dataset/
│   ├── images/
│   ├── labels/
│   ├── raw_images/
│   ├── quantification/
│   │   └── quantification_ds.csv
│   └── weld.yaml
│
├── preprocessing/
│   └── preprocess.py
│
├── yolo/
│   └── train.py
│
├── linear_regressor/
│   └── linear_regressor.py
│
├── runs/
│
├── yolov8n.pt
├── yolo26n.pt
│
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

```bash
python yolo/train.py
```

---

## Quantification

```bash
python linear_regressor/linear_regressor.py
```

This script performs feature evaluation, model training, and dimensional prediction experiments.

---

## Reproducibility

To ensure compatibility and reproduce the same software environment used during development and experimentation, install all dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

This installs the exact package versions used for preprocessing, model training, defect detection, and dimensional quantification.

---

## Future Work

- Expand dataset size
- Explore nonlinear regression models
- Investigate neural-network-based quantification methods
- Extend framework to additional weld inspection techniques

---

## Acknowledgements

Developed during an internship at IIT Goa as part of research on automated ultrasonic weld inspection, defect localization, and dimensional quantification using computer vision and machine learning.

---

## Author

**Aditya Singh**

B.Tech Mechanical Engineering  
Indian Institute of Technology Goa