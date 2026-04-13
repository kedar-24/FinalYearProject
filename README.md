# 🌿 Crop Disease Detection & Classification

An end-to-end two-stage deep learning pipeline for **detecting** and **classifying** crop diseases from field images of Cassava, Corn, and Tomato crops.

---

## Architecture

| Track | Model | Purpose | Dataset |
|-------|-------|---------|---------| 
| **A — Detection** | **YOLOv11m** | Locates diseased leaf regions via bounding boxes | [FieldPlant](https://universe.roboflow.com/plant-disease-detection/fieldplant) (30 classes) |
| **B — Classification** | **EfficientNet-B0** | Classifies the specific disease (top-5 probabilities) | [Crop Pest and Disease Detection](https://www.kaggle.com/datasets/nirmalsankalana/crop-pest-and-disease-detection) — Cassava, Maize & Tomato (17 classes) |

### Pipeline Flow
```
Input Image ─┬─→ YOLOv11m Detection    → Bounding Boxes (annotated image)
             └─→ EfficientNet-B0       → Disease Classification (probability distribution)
                                       → Gradio UI → Combined Results Display
```

---

## Datasets

### 1. FieldPlant — Object Detection (Track A)
> **Source**: [Roboflow Universe — FieldPlant](https://universe.roboflow.com/plant-disease-detection/fieldplant)

The FieldPlant dataset contains real-world field images of diseased and healthy crop leaves captured directly from plantations. Unlike lab-based datasets, these images feature complex backgrounds, varying lighting, and multiple leaves per image — making them ideal for training robust detection models.

| Property | Value |
|----------|-------|
| Total Images | ~5,170 |
| Annotated Leaves | ~8,629 |
| Classes | 30 (custom Roboflow version) |
| Annotation Format | YOLO (bounding boxes) |
| Image Size | 640×640 (resized for training) |
| Crops Covered | Cassava, Corn, Tomato |
| License | CC BY 4.0 |

### 2. New Plant Diseases Dataset — Classification (Track B)
> **Source**: [Kaggle — New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

Recreated using offline augmentation from the original [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) repository. Contains ~87K RGB images of healthy and diseased crop leaves across 38 classes with an 80/20 train/validation split. We use only the **Corn (4 classes)** and **Tomato (10 classes)** subsets for our pipeline.

| Property | Value |
|----------|-------|
| Total Images | ~87,000 (augmented) |
| Total Classes | 38 (we use 14: 4 Corn + 10 Tomato) |
| Split | 80% Train / 20% Validation |
| Image Type | RGB leaf images (lab background) |
| License | Data files © Original Authors |

### 3. Cassava Leaf Disease Classification — Classification (Track B)
> **Source**: [Kaggle — Cassava Leaf Disease Classification](https://www.kaggle.com/datasets/nirmalsankalana/cassava-leaf-disease-classification)

Contains images of cassava plant leaves organized into 5 disease/healthy categories. Originally sourced from the [Kaggle Cassava Competition](https://www.kaggle.com/competitions/cassava-leaf-disease-classification). Images are captured using relatively inexpensive cameras, simulating real-world agricultural conditions in African countries.

| Property | Value |
|----------|-------|
| Total Images | ~21,397 |
| Classes | 5 (Bacterial Blight, Brown Streak Disease, Green Mottle, Mosaic Disease, Healthy) |
| Split | 80% Train / 20% Validation (manual split) |
| Image Type | RGB field images |
| License | CC0: Public Domain |

---

## Model Performance

### Track A — YOLOv11m (Object Detection)
| Metric | Value |
|--------|-------|
| Dataset | FieldPlant — 30 classes |
| Image Size | 640×640 |
| Total Epochs | 50 |
| **mAP@50** | **99.36%** |
| **mAP@50-95** | **96.09%** |
| **Precision** | **96.69%** |
| **Recall** | **97.88%** |
| Val Box Loss | 0.3047 |
| Val Cls Loss | 0.2014 |

### Track B — EfficientNet-B0 (Classification)
| Metric | Value |
|--------|-------|
| Dataset | [Crop Pest and Disease Detection](https://www.kaggle.com/datasets/nirmalsankalana/crop-pest-and-disease-detection) (Cassava + Maize + Tomato) |
| Total Classes | 17 (5 Cassava + 7 Maize + 5 Tomato) |
| Image Size | 224×224 |
| Loss Function | CrossEntropyLoss |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Mixed Precision | FP16 via `torch.amp` |
| Train/Val Split | 80% / 20% (stratified, seed=42) |
| **Validation Accuracy** | **91%** |
| **Macro F1 Score** | **0.90** |
| **Weighted F1 Score** | **0.91** |
| Val Samples | 3,734 |

#### Per-Class Performance
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Cassava_Bacterial_Blight | 0.94 | 0.96 | 0.95 |
| Cassava_Brown_Spot | 0.95 | 0.94 | 0.95 |
| Cassava_Green_Mite | 0.94 | 0.93 | 0.94 |
| Cassava_Healthy | 0.97 | 0.97 | 0.97 |
| Cassava_Mosaic | 0.97 | 0.97 | 0.97 |
| Maize_Fall_Armyworm | 0.96 | 0.89 | 0.92 |
| Maize_Grasshoper | 0.98 | 0.99 | 0.99 |
| Maize_Healthy | 0.82 | 0.95 | 0.88 |
| Maize_Leaf_Beetle | 0.98 | 0.99 | 0.99 |
| Maize_Leaf_Blight | 0.76 | 0.85 | 0.80 |
| Maize_Leaf_Spot | 0.84 | 0.77 | 0.80 |
| Maize_Streak_Virus | 0.87 | 0.85 | 0.86 |
| Tomato_Healthy | 0.95 | 0.97 | 0.96 |
| Tomato_Leaf_Blight | 0.85 | 0.83 | 0.84 |
| Tomato_Leaf_Curl | 0.79 | 0.80 | 0.80 |
| Tomato_Septoria_Leaf_Spot | 0.90 | 0.92 | 0.91 |
| Tomato_Verticulium_Wilt | 0.78 | 0.74 | 0.76 |

---

## Disease Classes

### Detection — FieldPlant (30 classes)
| Crop | Count | Classes |
|------|-------|---------|
| **Cassava** | 5 | Bacterial Disease, Brown Leaf Spot, Healthy, Mosaic, Root Rot |
| **Corn** | 19 | Blight, Brown Spots, Cercosporiose, Charcoal, Chlorotic Leaf Spot, Healthy, Insects Damages, Mildiou, Purple Discoloration, Rust, Smut, Streak, Stripe, Violet Decoloration, Yellow Spots, Yellowing |
| **Tomato** | 6 | Brown Spots, Leaf Curling, Mildiou, Mosaic, Bacterial Wilt, Healthy |

### Classification — Crop Pest and Disease Detection (17 classes)
| Crop | Count | Classes |
|------|-------|---------|
| **Cassava** | 5 | Bacterial Blight, Brown Spot, Green Mite, Healthy, Mosaic |
| **Maize** | 7 | Fall Armyworm, Grasshoper, Healthy, Leaf Beetle, Leaf Blight, Leaf Spot, Streak Virus |
| **Tomato** | 5 | Healthy, Leaf Blight, Leaf Curl, Septoria Leaf Spot, Verticulium Wilt |

---

## Project Structure
```
FinalProject/
├── app.py                      # Entry point — launches the Gradio frontend
├── frontend/
│   └── app.py                  # Gradio web interface (swappable UI layer)
├── src/
│   ├── config.py               # Centralized settings (Pydantic)
│   ├── model.py                # EfficientNet-B0 architecture + Focal Loss
│   ├── train.py                # EfficientNet training loop
│   ├── dataset.py              # DataLoader & augmentation pipeline
│   ├── yolo_manager.py         # YOLOv11m training & inference wrapper
│   ├── inference_engine.py     # Two-track inference orchestrator
│   └── logger.py               # Structured logging (file + console)
├── models/
│   ├── best_yolo11m.pt         # Trained YOLOv11m weights
│   └── best_efficientnet.pth   # Trained EfficientNet-B0 weights
├── notebooks/
│   ├── train_model.ipynb       # Kaggle notebook — YOLOv11m training
│   └── train_efficientnet.ipynb # Kaggle notebook — EfficientNet-B0 training
├── scripts/
│   ├── prepare_datasets.py     # Dataset preparation & conversion
│   ├── setup_colab.py          # Google Colab environment setup
│   └── verify_setup.py         # Installation verification
├── data/                       # FieldPlant dataset (Roboflow YOLO format)
│   ├── data.yaml
│   └── train/
├── docs/
│   └── presentation_prompt.md  # Slide deck generation prompt
├── logs/                       # Runtime logs (auto-created)
├── Dockerfile                  # Container deployment
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
└── .gitignore
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models on Kaggle

**YOLOv11m (Detection):**
1. Download FieldPlant from [Roboflow](https://universe.roboflow.com/plant-disease-detection/fieldplant) in YOLOv8 format
2. Upload the dataset to Kaggle
3. Open `notebooks/train_model.ipynb` → Enable GPU T4 × 2 → Run all cells
4. Download `best.pt` → place in `models/best_yolo11m.pt`

**EfficientNet-B0 (Classification):**
1. Add Kaggle datasets:
   - [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
   - [Cassava Leaf Disease Classification](https://www.kaggle.com/datasets/nirmalsankalana/cassava-leaf-disease-classification)
2. Open `notebooks/train_efficientnet.ipynb` → Enable GPU T4 → Run all cells
3. Download `best_efficientnet.pth` → place in `models/best_efficientnet.pth`

### 3. Run Inference Locally
```bash
python app.py
```
Open `http://localhost:7860` in your browser.

### 4. Docker Deployment
```bash
docker build -t crop-disease .
docker run -p 7860:7860 crop-disease
```

---

## Key Techniques
- **YOLOv11m**: Latest-generation YOLO architecture for real-time object detection at 640px
- **EfficientNet-B0**: Compound-scaled CNN backbone with efficient channel attention
- **CrossEntropyLoss**: Stable and proven loss function for multi-class classification
- **ReduceLROnPlateau**: Adaptive learning rate reduction when validation accuracy plateaus
- **AdamW**: Decoupled weight decay optimizer (lr=1e-3, wd=1e-4)
- **Mixed Precision (AMP)**: FP16 training for ~2× speedup on T4 GPUs
- **GradScaler**: Prevents underflow in FP16 gradient computation

---

## Dataset Citations

### FieldPlant
```bibtex
@article{moupojou2023fieldplant,
  title   = {FieldPlant: A Dataset of Field Plant Images for Plant Disease
             Detection and Classification With Deep Learning},
  author  = {Moupojou, Emmanuel and Tagne, Appolinaire and Retraint, Florent
             and others},
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

## License
This project is for academic purposes. Dataset licenses: FieldPlant (CC BY 4.0), PlantVillage (Data files © Original Authors), Cassava (CC0: Public Domain).
