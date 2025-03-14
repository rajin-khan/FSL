<div align="center">

<img src="https://inkbotdesign.com/wp-content/uploads/2023/08/Unilever-Logo-PNG-File-881x1024.webp" alt="Unilever Bangladesh Retail Analysis" width="500">

# Unilever Bangladesh 
## Retail Analysis Project

## 📌 Overview
This is a YOLO-based object detection pipeline for Unilever Bangladesh, designed to classify and differentiate **UBL vs. Non-UBL** products, specifically focusing on sachets and shelf products. The system supports **multiple models (YOLOv8, Transformer-based approaches, etc.)** and provides a structured dataset for training, testing, and validation.

## 🔍 Project Objectives
- Detect and classify **UBL vs. Non-UBL** products in retail environments.
- Develop an adaptable **YOLOv8-based detection pipeline**.
- Automate **image augmentation, annotation, training, and evaluation**.
- Provide a reusable framework for **training and inference** on custom datasets.

## 📁 Repository Structure

</div>

```
UBLRetailAnalysis/
│── Main/
│   ├── dataset/              # Annotated dataset (linked below)
│   ├── models/               # Trained model weights
│   ├── selected_results_v1/  # Results from UBLModel_v1 (50 epochs)
│   ├── selected_results_v2/  # Results from UBLModel_v2 (500 epochs)
│   ├── training_logs/        # Logs from model training
│── Pipeline/
│   ├── pipeline.py           # Detection pipeline for inference
│   ├── README.md             # Documentation for running the pipeline
│── notebooks/                # Jupyter Notebooks for analysis
│── README.md                 # Main documentation
```
<div align="center">

---

## 📊 Dataset & Annotation
The dataset consists of **sachets and shelf product images**, categorized as **UBL vs. Non-UBL** products. 

### 🔹 **Classes for Detection**
| **Category**          | **Class Name**               | **Description**                           |
|----------------------|----------------------------|-------------------------------------------|
| **Sachet Detection** | `Clear`                    | Identify Clear sachets                   |
|                      | `Dove`                     | Identify Dove sachets                    |
|                      | `Horlicks`                 | Identify Horlicks sachets                |
| **UBL vs. Non-UBL**  | `UBL Sachet`               | Unilever sachet products                 |
|                      | `Non-UBL Sachet`           | Non-Unilever sachet products             |
| **Shelf Detection**  | `UBL Shelf`                | Unilever shelf products                  |
|                      | `Non-UBL Shelf`            | Non-Unilever shelf products              |
|                      | `Dove (Hair Care)`         | Identify Dove Hair Care shelf products   |
|                      | `Glow & Lovely (Skin Care)`| Identify Glow & Lovely Skin Care products |
|                      | `Horlicks (Nutrition)`     | Identify Horlicks Nutrition shelf products |

📌 **Annotated Dataset:**
[Google Drive Link](https://drive.google.com/drive/folders/1t93-ZDU6EJ9sKtC4nnadoTLlwy_sUOzj?usp=drive_link)

---

## 🏗️ Model Training Workflow

### **🔹 Step 1: Data Preparation**
- **Collected images** for training and validation.
- **Applied augmentations** (rotation, cropping, brightness variations, background noise) using `albumentations`.
- **Labeled and annotated** images using `LabelImg`.

### **🔹 Step 2: Training**
- **First training**: 50 epochs locally using YOLOv8.
- **Extended training**: 500 epochs on Google Colab with a TPU/GPU for better generalization.
- **Hyperparameter tuning**: Adjusted **batch size, learning rate, augmentation strategies**.

📌 **Trained Models Available:**
| Model Version | Epochs | Download Link | Sample Results |
|--------------|--------|---------------|----------------|
| `UBLModel_v1.pt` | 50 | [GitHub - Models](https://github.com/rajin-khan/UBLRetailAnalysis/tree/main/Main/models) | [Selected Results v1](https://github.com/rajin-khan/UBLRetailAnalysis/tree/main/Main/selected_results_v1) |
| `UBLModel_v2.pt` | 500 | [GitHub - Models](https://github.com/rajin-khan/UBLRetailAnalysis/tree/main/Main/models) | [Selected Results v2](https://github.com/rajin-khan/UBLRetailAnalysis/tree/main/Main/selected_results_v2) |

---

## 🏃🏻 Running the YOLOv8 Detection Pipeline

The pipeline allows **easy inference on new images** using trained YOLOv8 models.

### 🔹 **Step 1: Clone the Repository**

</div>

```bash
git clone https://github.com/rajin-khan/UBLRetailAnalysis.git
cd UBLRetailAnalysis/Pipeline
```
<div align="center">

### 🔹 **Step 2: Install Dependencies**

</div>

```bash
pip install ultralytics opencv-python numpy tqdm pyyaml
```
<div align="center">

### 🔹 **Step 3: Run the Pipeline**
```bash
python pipeline.py --model ../Main/models/UBLModel_v2.pt --input ../test_images/ --output ../predictions/
```
</div>

**Options:**
- `--model`: Path to the trained YOLOv8 `.pt` model.
- `--input`: Directory containing images for detection.
- `--output`: Directory where results will be saved.

<div align="center">

### 🔹 **Example Output**

</div>

```bash
✔ Processed 100 images
✔ Detections saved in `predictions/`
✔ Final summary stored in `detections_summary.txt`
```

---
