# 🌿 Complete Project Context — Crop Disease Detection & Classification System

> **Purpose**: This document contains the **complete context** of the final year project including all architecture details, source code summaries, training configurations, performance metrics, and dataset information. Copy this entire document into GPT/AI to generate an official academic report.

---

## 1. PROJECT OVERVIEW

**Title**: End-to-End Crop Disease Detection & Classification System

**Objective**: Build a two-stage deep learning pipeline that can (1) **detect** diseased leaf regions in real-world field images using object detection, and (2) **classify** the specific disease from the detected/uploaded leaf image using image classification. The system provides both bounding-box localization and a top-5 probability distribution of diseases.

**Crops Covered**: Cassava, Corn (Maize), and Tomato

**Frontend**: Gradio Web Interface (interactive, browser-based)

**Deployment**: Docker containerization, runs locally on port 7860

**Team**: 4 members — Soumya (Intro/Datasets), Lokesh (Architecture/Detection), Kedar (Classification/Optimizations), Shaurya (Deployment/Demo)

---

## 2. SYSTEM ARCHITECTURE

### Two-Stage Pipeline Design

```
Input Image ─┬─→ [Track A] YOLOv11m Detection    → Bounding Boxes (annotated image)
             └─→ [Track B] EfficientNet-B0       → Disease Classification (probability distribution)
                                                  → Gradio UI → Combined Results Display
```

**Why Two Stages Instead of One?**
- Object Detection (YOLO) excels at **localizing** diseased regions in complex field images with multiple leaves, varying lighting, and cluttered backgrounds.
- Image Classification (EfficientNet) excels at **identifying the specific disease** with high accuracy from a cleaner leaf image.
- Running both tracks independently on the same input provides complementary information: **where** the disease is (detection) and **what** the disease is (classification).

### Track A — Object Detection (YOLOv11m)
- **Model**: YOLO11m (upgraded from YOLOv8m — better C3k2 backbone + SPPF neck)
- **Parameters**: ~25M parameters
- **Input Size**: 640×640 pixels
- **Task**: Locates diseased leaf regions → draws bounding boxes + per-box labels
- **Dataset**: FieldPlant (30 classes across Cassava, Corn, Tomato)
- **Training Platform**: Kaggle (GPU T4 × 2)

### Track B — Image Classification (EfficientNet-B0)
- **Model**: EfficientNet-B0 (compound-scaled CNN with efficient channel attention)
- **Parameters**: ~5.3M parameters (base) + modified classifier head
- **Input Size**: 224×224 pixels
- **Task**: Whole-image disease classification → probability distribution (top-5)
- **Dataset**: PlantVillage (Corn + Tomato) — 14 classes
- **Training Platform**: Kaggle (GPU T4)
- **Note**: Cassava classes (5) were prepared but excluded from the final checkpoint due to Kaggle session constraints. The current deployed model classifies **14 classes** (4 Corn + 10 Tomato).

---

## 3. DATASETS

### 3.1 FieldPlant — Object Detection (Track A)

| Property | Value |
|----------|-------|
| **Source** | [Roboflow Universe — FieldPlant](https://universe.roboflow.com/plant-disease-detection/fieldplant) |
| **Total Images** | ~5,170 |
| **Annotated Leaves** | ~8,629 |
| **Classes** | 30 |
| **Annotation Format** | YOLO (bounding boxes) |
| **Image Size** | 640×640 (resized for training) |
| **Crops Covered** | Cassava, Corn, Tomato |
| **License** | CC BY 4.0 |
| **Key Characteristic** | Real-world field images captured directly from plantations (not lab images). Complex backgrounds, varying lighting, multiple leaves per image. Manual annotation under supervision of plant pathologists. |

**30 Detection Classes**:
- **Cassava (5)**: Bacterial Disease, Brown Leaf Spot, Healthy, Mosaic, Root Rot
- **Corn (19)**: Blight, Brown Spots, Cercosporiose, Charcoal, Chlorotic Leaf Spot, Healthy, Insects Damages, Mildiou, Purple Discoloration, Rust, Smut, Streak, Stripe, Violet Decoloration, Yellow Spots, Yellowing (+ duplicates: "Corn Healthy", "Corn Smut", "Corn Streak" from Roboflow)
- **Tomato (6)**: Brown Spots, Leaf Curling, Mildiou, Mosaic, Bacterial Wilt, Healthy

### 3.2 New Plant Diseases Dataset — Classification (Track B)

| Property | Value |
|----------|-------|
| **Source** | [Kaggle — New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) |
| **Original Source** | PlantVillage repository (Hughes & Salathé, 2015) |
| **Total Images** | ~87,000 (augmented) |
| **Total Classes** | 38 (we use 14: 4 Corn + 10 Tomato) |
| **Split** | 80% Train / 20% Validation |
| **Image Type** | RGB leaf images (lab/controlled background) |
| **License** | Data files © Original Authors |

**14 Classes Used (Corn + Tomato)**:
1. `Corn_Cercospora_Leaf_Spot`
2. `Corn_Common_Rust`
3. `Corn_Healthy`
4. `Corn_Northern_Leaf_Blight`
5. `Tomato_Bacterial_Spot`
6. `Tomato_Early_Blight`
7. `Tomato_Healthy`
8. `Tomato_Late_Blight`
9. `Tomato_Leaf_Mold`
10. `Tomato_Mosaic_Virus`
11. `Tomato_Septoria_Leaf_Spot`
12. `Tomato_Spider_Mites`
13. `Tomato_Target_Spot`
14. `Tomato_Yellow_Leaf_Curl_Virus`

### 3.3 Cassava Leaf Disease Classification — Classification (Track B, prepared but not in final checkpoint)

| Property | Value |
|----------|-------|
| **Source** | [Kaggle — Cassava Leaf Disease Classification](https://www.kaggle.com/datasets/nirmalsankalana/cassava-leaf-disease-classification) |
| **Total Images** | ~21,397 |
| **Classes** | 5 (Bacterial Blight, Brown Streak Disease, Green Mottle, Mosaic Disease, Healthy) |
| **Split** | 80% Train / 20% Validation (manual split) |
| **Image Type** | RGB field images |
| **License** | CC0: Public Domain |
| **Status** | Dataset was merged during training pipeline preparation but the final EfficientNet checkpoint was trained on 14 classes (Corn + Tomato only) due to Kaggle session/resource constraints. |

---

## 4. MODEL PERFORMANCE — METRICS & SCORES

### 4.1 Track A — YOLOv11m (Object Detection)

| Metric | Value |
|--------|-------|
| **Model** | YOLO11m |
| **Dataset** | FieldPlant — 30 classes |
| **Image Size** | 640×640 |
| **Total Epochs Trained** | 50 (with early stopping patience=15) |
| **Batch Size** | 32 (dual GPU) / 16 (single GPU) |
| **mAP@50** | **99.36%** |
| **mAP@50-95** | **96.09%** |
| **Precision** | **96.69%** |
| **Recall** | **97.88%** |
| **Validation Box Loss** | 0.3047 |
| **Validation Classification Loss** | 0.2014 |
| **Mixed Precision** | Yes (AMP enabled) |
| **Cosine LR Annealing** | Yes |
| **Cache** | RAM |
| **Confidence Threshold** | 0.25 |
| **Model File Size** | ~40.6 MB (`best_yolo11m.pt`) |

**Training Configuration (YOLO)**:
- Base model: `yolo11m.pt` (pretrained on COCO)
- Total target epochs: 100 (early stopped at ~50)
- Image size: 640×640
- Workers: 4
- Augmentation: Ultralytics default + cosine LR
- Save period: every 5 epochs
- Platform: Kaggle GPU T4 × 2

### 4.2 Track B — EfficientNet-B0 (Classification)

| Metric | Value |
|--------|-------|
| **Model** | EfficientNet-B0 |
| **Backbone Pretrained On** | ImageNet |
| **Dataset** | PlantVillage (Corn + Tomato subset) |
| **Total Classes** | 14 (4 Corn + 10 Tomato) |
| **Image Size** | 224×224 |
| **Loss Function** | CrossEntropyLoss |
| **Optimizer** | AdamW (lr=1e-3, weight_decay=1e-4) |
| **Scheduler** | ReduceLROnPlateau (mode='max', factor=0.5, patience=2) |
| **Mixed Precision** | FP16 via `torch.amp.autocast('cuda')` + `GradScaler` |
| **Batch Size** | 32 |
| **Num Workers** | 0 (disabled to fix Kaggle RAM crashes) |
| **Total Epochs Trained** | 11 (out of 20 target) |
| **Best Epoch** | 8 |
| **Best Validation Accuracy** | **99.35%** |
| **Weighted F1 Score** | **0.9934** |
| **Best Validation Loss** | 0.0203 |
| **Final Training Accuracy** | 98.72% |
| **GPU Peak Memory** | 2.28 GB |
| **Model File Size** | ~16.4 MB (`best_efficientnet.pth`) |

**Training Data Augmentation Pipeline**:
```
Training Transforms:
  1. Resize(224×224)
  2. RandomHorizontalFlip(p=0.5)
  3. RandomVerticalFlip(p=0.3)
  4. RandomRotation(degrees=30)
  5. ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
  6. RandomAffine(degrees=0, translate=(0.1, 0.1))
  7. ToTensor()
  8. Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

Validation Transforms:
  1. Resize(224×224)
  2. ToTensor()
  3. Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

**EfficientNet Architecture Details**:
- Backbone: `torchvision.models.efficientnet_b0` with `EfficientNet_B0_Weights.DEFAULT`
- Classifier Head: Original final Linear layer replaced with `nn.Linear(1280, 14)`
- The codebase also includes a `MultiClassFocalLoss` (gamma=2, alpha=0.25) implementation, but the final training used standard `CrossEntropyLoss` for greater stability.

---

## 5. CODEBASE ARCHITECTURE

### Project Structure
```
FinalProject/
├── app.py                        # Entry point — launches the Gradio frontend
├── frontend/
│   └── app.py                    # Gradio web interface (248 lines)
├── src/
│   ├── __init__.py               # Package marker
│   ├── config.py                 # Centralized settings via Pydantic (53 lines)
│   ├── model.py                  # EfficientNet-B0 architecture + Focal Loss (52 lines)
│   ├── train.py                  # EfficientNet training loop (156 lines)
│   ├── dataset.py                # DataLoader & augmentation pipeline (83 lines)
│   ├── yolo_manager.py           # YOLOv11m training & inference wrapper (58 lines)
│   ├── inference_engine.py       # Two-track inference orchestrator (147 lines)
│   └── logger.py                 # Structured logging — file + console (37 lines)
├── models/
│   ├── best_yolo11m.pt           # Trained YOLOv11m weights (40.6 MB)
│   ├── best_yolov8m.pt           # Legacy YOLOv8m weights (52.1 MB)
│   └── best_efficientnet.pth     # Trained EfficientNet-B0 weights (16.4 MB)
├── notebooks/
│   ├── train_model.ipynb         # Kaggle notebook — YOLOv11m training (353 lines)
│   └── train_efficientnet.ipynb  # Kaggle notebook — EfficientNet-B0 training (552 lines)
├── scripts/
│   ├── prepare_datasets.py       # Dataset preparation & XML→YOLO conversion (221 lines)
│   ├── setup_colab.py            # Google Colab environment setup (49 lines)
│   └── verify_setup.py           # End-to-end installation verification (143 lines)
├── data/                         # FieldPlant dataset (Roboflow YOLO format)
│   ├── data.yaml                 # 30-class YOLO config
│   ├── README.dataset.txt        # Dataset documentation
│   └── train/                    # Training images + labels
├── docs/
│   └── presentation_prompt.md    # Presentation slide generation prompt
├── logs/                         # Runtime logs (auto-created)
├── Dockerfile                    # Docker container deployment (41 lines)
├── requirements.txt              # 14 Python dependencies
├── .env.example                  # Environment variable template
└── .gitignore
```

### Module Descriptions

#### `src/config.py` — Centralized Configuration
- Uses **Pydantic Settings** (`BaseSettings`) for type-safe configuration
- Loads from `.env` file or environment variables
- Defines: app title, port (7860), model paths, input size (224), confidence threshold (0.25), all 14 CLASS_NAMES, NUM_CLASSES

#### `src/model.py` — Neural Network Architecture
- **`FLEfficientNet`**: Custom wrapper around `torchvision.models.efficientnet_b0`
  - Loads pretrained ImageNet weights
  - Replaces `classifier[1]` (final Linear layer) with `nn.Linear(1280, num_classes)`
- **`MultiClassFocalLoss`**: Implementation of Focal Loss (gamma=2, alpha=0.25) for handling class imbalance (available but not used in final training)

#### `src/train.py` — Training Pipeline
- **`EfficientNetTrainer`** class with:
  - train_one_epoch() with progress bars
  - validate() with F1 score computation
  - run() with best-model checkpointing
- Uses: Focal Loss, AdamW optimizer, OneCycleLR scheduler
- Computes: training loss/accuracy, validation loss/accuracy, weighted F1 score

#### `src/dataset.py` — Data Loading
- **`PlantVillageClassifierDataset`**: Wraps `ImageFolder` with train/val augmentation pipelines
- ImageNet normalization constants: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- **`create_yolo_yaml()`**: Generates `data.yaml` for YOLO training

#### `src/yolo_manager.py` — YOLO Wrapper
- **`YOLOManager`**: Manages YOLO model lifecycle
  - `train()`: Launches YOLO training with configurable epochs, image size, device
  - `predict()`: Runs inference on a single image with confidence threshold
  - `validate()`: Runs validation and returns mAP metrics

#### `src/inference_engine.py` — Two-Track Orchestrator
- **`DiseaseInferenceEngine`**: Core inference class
  - Loads both models on initialization (with graceful fallback)
  - `predict(image)` runs both tracks independently:
    - Track A: YOLO detection → annotated image with bounding boxes
    - Track B: EfficientNet classification → top-5 probability distribution
  - Handles state_dict key remapping (`backbone.` prefix) for compatibility between training notebook (raw efficientnet_b0) and inference (FLEfficientNet wrapper)

#### `src/logger.py` — Logging
- Dual handler: console (stdout) + file (`logs/app.log`)
- Format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

#### `frontend/app.py` — Gradio Web Interface
- Premium dark theme with glassmorphism CSS
- Google Font: Inter
- Input: Image upload (upload, webcam, clipboard)
- Output tabs: Detection (annotated image) + Classification (label probabilities)
- Model badges for YOLOv11m and EfficientNet-B0

---

## 6. KEY TECHNIQUES & TECHNOLOGIES

### Deep Learning Techniques
| Technique | Purpose | Details |
|-----------|---------|---------|
| **Transfer Learning** | Reduce training time, improve performance | EfficientNet pretrained on ImageNet, YOLO pretrained on COCO |
| **Mixed Precision Training (AMP)** | ~2× GPU speedup, reduced memory | FP16 forward pass via `torch.amp.autocast('cuda')` |
| **GradScaler** | Prevent underflow in FP16 gradients | Scales loss before backward, unscales gradients before optimizer step |
| **AdamW Optimizer** | Decoupled weight decay | lr=1e-3, weight_decay=1e-4 |
| **ReduceLROnPlateau** | Adaptive learning rate | Halves LR when val accuracy plateaus (patience=2) |
| **CrossEntropyLoss** | Multi-class classification | Stable, proven loss for balanced datasets |
| **Focal Loss (available)** | Class imbalance handling | gamma=2, alpha=0.25 (implemented but not used in final training) |
| **Data Augmentation** | Prevent overfitting | Flip, rotation, color jitter, affine transforms |
| **Early Stopping** | Prevent overfitting in YOLO | patience=15 epochs |
| **Cosine LR Annealing** | Smooth LR decay for YOLO | Enabled via `cos_lr=True` |
| **Compound Scaling** | EfficientNet architecture | Balances depth, width, and resolution scaling |

### Technology Stack
| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.10 |
| **Deep Learning Framework** | PyTorch ≥ 2.0, TorchVision ≥ 0.15 |
| **Object Detection** | Ultralytics ≥ 8.0 (YOLO11m) |
| **Web Framework** | Gradio ≥ 4.0 |
| **Configuration** | Pydantic Settings |
| **Image Processing** | Pillow, OpenCV |
| **Metrics** | scikit-learn (F1, classification_report) |
| **Augmentation** | TorchVision transforms, Albumentations |
| **Containerization** | Docker (Python 3.10-slim) |
| **Training Platform** | Kaggle (NVIDIA T4 GPUs) |
| **Dataset Hosting** | Kaggle Datasets, Roboflow Universe |

---

## 7. DEPLOYMENT ARCHITECTURE

### Local Deployment
```bash
pip install -r requirements.txt
python app.py
# → Gradio UI at http://localhost:7860
```

### Docker Deployment
```dockerfile
FROM python:3.10-slim
# System deps for OpenCV + ultralytics
RUN apt-get install libgl1-mesa-glx libglib2.0-0 ...
COPY requirements.txt → pip install
COPY src/ frontend/ app.py models/ → /app
EXPOSE 7860
HEALTHCHECK every 30s
CMD ["python", "app.py"]
```
```bash
docker build -t crop-disease .
docker run -p 7860:7860 crop-disease
```

### Environment Variables (`.env`)
```
APP_TITLE=Crop Disease Diagnosis System
DEBUG=false
PORT=7860
YOLO_PATH=models/best_yolo11m.pt
EFFICIENTNET_PATH=models/best_efficientnet.pth
YOLO_FALLBACK_MODEL=yolo11m.pt
INPUT_SIZE=224
CONFIDENCE_THRESHOLD=0.25
```

---

## 8. TRAINING WORKFLOW

### YOLOv11m Training (Kaggle Notebook: `train_model.ipynb`)
1. Install ultralytics on Kaggle
2. Locate FieldPlant dataset and read `data.yaml`
3. Rebuild `data.yaml` with absolute Kaggle paths
4. Load `yolo11m.pt` pretrained model
5. Train with: epochs=100, imgsz=640, batch=32, amp=True, cos_lr=True, patience=15, cache='ram'
6. Save `best.pt` and `last.pt` for download
7. Supports checkpoint resume if Kaggle session expires

### EfficientNet-B0 Training (Kaggle Notebook: `train_efficientnet.ipynb`)
1. Install pydantic-settings, albumentations on Kaggle
2. GPU diagnostic check
3. Build unified dataset by merging:
   - PlantVillage → extract 14 Corn+Tomato classes with renamed folders
   - Cassava → read train.csv, split 80/20, map labels to class names
4. Build DataLoaders with augmentation transforms
5. Load EfficientNet-B0 with ImageNet weights, replace classifier head
6. Train with: CrossEntropyLoss, AdamW(lr=1e-3), ReduceLROnPlateau, GradScaler, AMP
7. Save best model based on validation accuracy
8. Save full checkpoint (model + optimizer + scheduler + scaler + history) every epoch
9. Generate training curves (Loss, Accuracy, F1) and classification report

---

## 9. DATASET CITATIONS

### FieldPlant
```bibtex
@article{moupojou2023fieldplant,
  title   = {FieldPlant: A Dataset of Field Plant Images for Plant Disease
             Detection and Classification With Deep Learning},
  author  = {Moupojou, Emmanuel and Tagne, Appolinaire and Retraint, Florent
             and Tadonkemwa, Anicet and Wilfried, Dongmo and Tapamo, Hyppolite
             and Nkenlifack, Marcellin},
  journal = {IEEE Access},
  volume  = {11},
  pages   = {35398--35410},
  year    = {2023},
  publisher = {IEEE}
}
```

### PlantVillage (New Plant Diseases Dataset)
```bibtex
@article{hughes2015plantvillage,
  title   = {An open access repository of images on plant health to enable the
             development of mobile disease diagnostics},
  author  = {Hughes, David and Salathé, Marcel},
  journal = {arXiv preprint arXiv:1511.08060},
  year    = {2015}
}
```

### Cassava Leaf Disease Classification
```bibtex
@misc{cassava2020,
  title     = {Cassava Leaf Disease Classification},
  author    = {Nirmal Sankalana},
  year      = {2023},
  publisher = {Kaggle},
  url       = {https://www.kaggle.com/datasets/nirmalsankalana/cassava-leaf-disease-classification},
  note      = {CC0: Public Domain}
}
```

---

## 10. PERFORMANCE SUMMARY TABLE

| Metric | Track A (YOLOv11m) | Track B (EfficientNet-B0) |
|--------|---------------------|---------------------------|
| **Task** | Object Detection | Image Classification |
| **Dataset** | FieldPlant (30 classes) | PlantVillage (14 classes) |
| **Primary Metric** | mAP@50 = **99.36%** | Val Accuracy = **99.35%** |
| **Secondary Metric** | mAP@50-95 = **96.09%** | Weighted F1 = **0.9934** |
| **Precision** | **96.69%** | — |
| **Recall** | **97.88%** | — |
| **Best Loss** | Box=0.3047, Cls=0.2014 | Val Loss = **0.0203** |
| **Training Accuracy** | — | 98.72% |
| **Epochs** | 50 | 11 (best at epoch 8) |
| **GPU Memory** | — | Peak 2.28 GB |
| **Model Size** | 40.6 MB | 16.4 MB |
| **Input Resolution** | 640×640 | 224×224 |
| **Parameters** | ~25M | ~5.3M |

---

## 11. COMPARISON WITH EXISTING WORK

The FieldPlant paper (Moupojou et al., 2023) benchmarked several models:
- FieldPlant was created because PlantVillage (lab images) models had very low accuracy on real field images
- PlantDoc had issues: included some lab images + no plant pathologist supervision during annotation
- FieldPlant solved both: 5,170 real field images, annotated under pathologist supervision
- Our YOLOv11m achieved **99.36% mAP@50** on FieldPlant, demonstrating excellent detection in real-world conditions

For classification, EfficientNet-B0 on PlantVillage subset achieved **99.35% accuracy**, consistent with state-of-the-art results on this benchmark (typical range: 97-99.5%).

---

## 12. LIMITATIONS & FUTURE WORK

### Current Limitations
1. **Cassava classes not in final classifier**: The 5 Cassava classes were prepared but excluded from the final EfficientNet checkpoint (14 classes instead of 19) due to Kaggle training session constraints.
2. **Lab vs Field gap**: EfficientNet was trained on PlantVillage (lab images), while YOLO was trained on FieldPlant (real field images). The classification may be less accurate on field-captured images.
3. **No cross-track integration**: Detection and classification run independently — the system doesn't crop YOLO-detected regions and feed them to EfficientNet.

### Future Work
1. Integrate Cassava classes into the EfficientNet classifier (19 total classes)
2. Implement **crop-and-classify**: Use YOLO detections as input to EfficientNet
3. Add **Grad-CAM** visual explanations for classification decisions
4. Deploy on mobile devices using TensorRT/ONNX optimization
5. Expand to more crop species and diseases
6. Real-time video stream processing for field deployment

---

## 13. LICENSE

This project is for **academic purposes**. Dataset licenses:
- FieldPlant: **CC BY 4.0**
- PlantVillage: **Data files © Original Authors**
- Cassava: **CC0: Public Domain**
