# Friction Stir Weld Defect Detection using YOLOv8 and Ultrasonic NDT Imaging

## Overview

This project presents an automated computer vision pipeline for detecting defects in Friction Stir Welded (FSW) joints using Ultrasonic Non-Destructive Testing (NDT) scan images.

The system combines image preprocessing, region-of-interest extraction, dataset annotation, and YOLOv8-based object detection to automatically identify defect regions within ultrasonic weld inspection scans.

Developed during an internship at IIT Goa, this work aims to reduce dependence on manual inspection and provide a foundation for automated defect localization and quantification in industrial weld quality assessment.

---

## Objectives

- Detect weld defects from ultrasonic S-scan images.
- Automate defect localization using deep learning.
- Reduce manual inspection effort.
- Develop a preprocessing pipeline tailored to ultrasonic NDT scans.
- Create a framework that can later be extended for defect dimension estimation and quantification.

---

## Project Pipeline

```text
Raw Ultrasonic Scan
        │
        ▼
Image Preprocessing
        │
        ▼
ROI Extraction
(Triangular Scan Region)
        │
        ▼
Image Enhancement
        │
        ▼
Bounding Box Annotation
        │
        ▼
YOLOv8 Training
        │
        ▼
Defect Localization
        │
        ▼
Defect Quantification (Future Work)
```

---

## Dataset

The dataset consists of ultrasonic inspection scans acquired from Friction Stir Welded specimens.

### Classes

| Class ID | Class Name |
|-----------|------------|
| 0 | Defect |

### Dataset Split

| Split | Purpose |
|---------|----------|
| Train | Model training |
| Validation | Performance evaluation |

The dataset contains both:

- Defective weld scans
- Defect-free weld scans (background images)

to improve detector robustness and reduce false positives.

---

## Image Preprocessing

Raw ultrasonic scans contain significant irrelevant regions that do not contribute to defect detection.

A custom preprocessing pipeline was developed to:

### 1. Region of Interest Extraction

Extract only the triangular ultrasonic scan region containing relevant inspection information.

### 2. Noise Reduction

Reduce unwanted artifacts and improve signal visibility.

### 3. Image Enhancement

Improve contrast and feature visibility for defect patterns.

### 4. Geometric Filtering

Remove unnecessary background regions while preserving weld indications.

This preprocessing stage ensures that the detector focuses on meaningful ultrasonic information rather than irrelevant image content.

---

## Data Annotation

Defect regions were annotated using bounding boxes in YOLO format.

### Challenges Encountered

#### Manual Annotation

- Time-consuming process
- Required careful interpretation of ultrasonic indications
- Limited availability of labeled data

#### Automatic Annotation Attempts

Computer vision techniques were explored to generate annotations automatically.

Approaches investigated included:

- Thresholding
- Edge detection
- Contour extraction
- ROI-based segmentation

While useful for initial experimentation, manual verification remained necessary to ensure annotation quality.

---

## Model Architecture

### YOLOv8 Nano

The project utilizes the YOLOv8 Nano architecture for real-time object detection.

Reasons for selection:

- Lightweight architecture
- Fast training and inference
- Strong performance on limited datasets
- Easy deployment

### Transfer Learning

Pretrained YOLOv8 weights were used as the initialization point to improve performance despite limited training data.

---

## Training Configuration

| Parameter | Value |
|------------|---------|
| Model | YOLOv8n |
| Image Size | 640 × 640 |
| Batch Size | 4 |
| Optimizer | AdamW |
| Epochs | Up to 100 |
| Early Stopping | Enabled |
| Framework | Ultralytics YOLOv8 |
| Hardware | NVIDIA RTX 3050 Laptop GPU |

---

## Results

Best validation results obtained during training:

| Metric | Value |
|-----------|-----------|
| Precision | 0.901 |
| Recall | 0.833 |
| mAP@50 | 0.946 |
| mAP@50-95 | 0.492 |

### Interpretation

- High mAP@50 indicates strong defect localization capability.
- Precision above 90% indicates low false-positive detections.
- Recall above 83% indicates the majority of defects were successfully detected.
- Performance was achieved despite a limited labeled dataset.

---

## Repository Structure

```text
FSW_Defect_Detection/
│
├── dataset/
│   ├── images/
│   ├── labels/
│   ├── raw_images/
│   └── weld.yaml
│
├── preprocessing/
│   └── preprocess.py
│
├── yolo/
│   └── train.py
│
├── cfnn/
│
├── outputs/
│
├── runs/
│
├── yolov8n.pt
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<username>/<repository>.git
cd FSW_Defect_Detection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Run:

```bash
python yolo/train.py
```

or

```bash
yolo detect train model=yolov8n.pt data=dataset/weld.yaml
```

---

## Inference

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="image.png",
    save=True,
    conf=0.25
)
```

---

## Future Work

- Increase dataset size.
- Improve annotation quality.
- Implement defect dimension estimation.
- Integrate CFNN-based defect quantification.
- Explore instance segmentation models.
- Evaluate performance on additional weld types.
- Develop a complete automated NDT analysis pipeline.

---

## Acknowledgements

This project was developed during an internship at IIT Goa as part of research in automated ultrasonic weld inspection and defect detection using deep learning and computer vision techniques.

---

## Author

**Aditya [Last Name]**

B.Tech Student  
Computer Vision and Machine Learning Enthusiast

Developed a YOLO-CFNN Hybrid Framework for Ultrasonic Weld Defect Detection and Quantification.