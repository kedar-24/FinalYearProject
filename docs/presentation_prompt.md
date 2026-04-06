# Final College Presentation: Crop Disease Diagnosis System

## Speaker Breakdown & Strategy
Based on your team dynamics, here is the optimal distribution of the presentation to make it engaging, technical, and professional for your final college defense:

**1. Soumya (The Hook & Introduction - Low Technical Burden)**
*   **Role:** Sets the stage and introduces the problem.
*   **Topics:** Agricultural challenges, the impact of crop diseases, Introduction to the datasets, and the mission of the project.
*   **Datasets to cover:**
    - [FieldPlant](https://universe.roboflow.com/plant-disease-detection/fieldplant) — ~5,170 field images with 8,629 annotations across 30 disease classes (Cassava, Corn, Tomato). Real-world plantation images.
    - [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) — ~87K augmented RGB images across 38 classes from PlantVillage (we use 14 Corn + Tomato classes).
    - [Cassava Leaf Disease Classification](https://www.kaggle.com/datasets/nirmalsankalana/cassava-leaf-disease-classification) — ~21K images across 5 cassava disease classes. Field-captured in African conditions.
*   **Why:** Requires no deep ML knowledge, but relies on good presentation skills to engage the audience initially.

**2. Lokesh (The Architecture & Detection - High Technical Burden)**
*   **Role:** Explains the "Why" and the first half of the pipeline.
*   **Topics:** The Two-Stage Architecture (Why not just one model?). Deep dive into YOLOv11m for object detection. How the model locates the diseased leaf regions in the wild using the FieldPlant dataset.
*   **Why:** Highlights core engineering decisions and the mechanics of the object detection track.

**3. Kedar (Classification & Advanced Optimizations - High Technical Burden)**
*   **Role:** Explains the hardcore deep learning optimizations.
*   **Topics:** EfficientNet-B0 classification. CrossEntropyLoss for stable multi-class training. Merging PlantVillage + Cassava datasets into a unified 19-class classifier. Advanced training strategies: ReduceLROnPlateau, AdamW, Mixed Precision (AMP), GradScaler.
*   **Why:** You and Lokesh did the heavy lifting, so you take the most complex SOTA training methodologies to impress the evaluators.

**4. Shaurya (The Manager: Deployment, Demo & Impact - Medium Technical/Business Burden)**
*   **Role:** Brings it all together, shows the working model, and concludes.
*   **Topics:** Gradio Web Interface, Docker Containerization, Local/Cloud deployment workflow, Live System Demo, and the final Business/Real-World impact.
*   **Why:** Fits the "manager" persona perfectly—focusing on the final product, user experience, and the overarching value of the system.

---

## The Prompt for Slide Generation

Copy and paste the entire block below into an AI assistant to generate your actual slide decks (whether you want it in Marp Markdown, Reveal.js, or just as detailed textual slides with speaker notes).

```text
Please generate a State-Of-The-Art (SOTA) slide deck for my final year college project presentation. The project is an "End-to-End Crop Disease Detection & Classification System". 

Here is the context of our project:
- We built a two-stage deep learning pipeline.
- Stage 1 (Detection): YOLOv11m trained on the FieldPlant dataset (~5,170 field images, 8,629 annotated leaves, 30 classes across Cassava, Corn, and Tomato).
- Stage 2 (Classification): EfficientNet-B0 with CrossEntropyLoss trained on a merged dataset:
  - New Plant Diseases Dataset (~87K augmented PlantVillage images — we use 14 classes: 4 Corn + 10 Tomato)
  - Cassava Leaf Disease Classification (~21K images, 5 classes)
  - Total: 19 unified classification classes
- Advanced Training Techniques: CrossEntropyLoss for stability, ReduceLROnPlateau (adaptive LR), AdamW (lr=1e-3, wd=1e-4), Mixed Precision (AMP) with GradScaler for 2x T4 GPU speedup.
- Deployment: Gradio Web UI and Docker containerization.
- Dataset sources:
  - Detection: https://universe.roboflow.com/plant-disease-detection/fieldplant
  - Classification: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
  - Classification: https://www.kaggle.com/datasets/nirmalsankalana/cassava-leaf-disease-classification

We are a team of 4 people. Please generate the slides and include **Speaker Notes** for each slide assigned exactly to the following people based on their roles:

1. **Soumya (Non-technical / Intro)**: Needs simple, engaging content. Assign her the Problem Statement, Introduction, and Dataset Overview.
2. **Lokesh (Core Developer)**: Assign him the System Architecture overview and the YOLOv11m Object Detection mechanics.
3. **Kedar (Core Developer)**: Assign him the hardcore DL specifics. EfficientNet-B0, CrossEntropyLoss, ReduceLROnPlateau, AdamW, and Mixed Precision optimizations. 
4. **Shaurya (Manager / Presenter)**: Assign him the Architecture Pipeline Flow, Gradio UI Demo setup, Deployment (Docker), system performance/results, and Business Impact/Conclusion.

Requirements for the output:
- Create around 15-20 highly professional, visually descriptive slides.
- Provide a slide title, the visual elements/bullet points for the slide, and a detailed "Speaker Notes" section script tailored to that specific person's capabilities.
- Keep the tone highly academic but extremely modern and "SOTA" to wow our college evaluators. Give it a Silicon Valley tech-launch vibe.
- Format the output so I can easily use it to create a Reveal.js application or put it into PowerPoint.
```
