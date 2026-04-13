import gradio as gr
import numpy as np
import os
import sys
from pathlib import Path

# Ensure project root is in path since frontend/ is isolated
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference_engine import DiseaseInferenceEngine
from src.config import settings
from src.logger import logger


# ── Initialize Inference Engine ──
engine = None
try:
    engine = DiseaseInferenceEngine()
    logger.info("Inference Engine started successfully.")
except Exception as e:
    logger.critical(f"Failed to start Inference Engine: {e}")


def diagnosis_pipeline(image: np.ndarray):
    """
    Gradio callback: runs image through YOLOv11m detection + EfficientNet-B0 classification.
    Returns annotated image and classification probabilities.
    """
    if engine is None:
        return None, {"Error": "System initialization failed. Check logs."}

    if image is None:
        return None, {"Error": "No image provided."}

    try:
        detection_img, classification_results, gradcam_img = engine.predict(image)
        return detection_img, classification_results, gradcam_img
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return image, {"Error": str(e)}, None


# ── Custom CSS for premium look ──
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; }

.gradio-container {
    max-width: 1280px !important;
    margin: auto !important;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%) !important;
}

footer { display: none !important; }

.app-header {
    text-align: center;
    padding: 2rem 1rem 1rem;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(59, 130, 246, 0.06));
    border-radius: 16px;
    border: 1px solid rgba(16, 185, 129, 0.15);
    margin-bottom: 1.5rem;
}

.app-header h1 {
    background: linear-gradient(135deg, #10b981, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.3rem !important;
}

.app-header p {
    color: #94a3b8 !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
}

.model-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px 4px;
    border: 1px solid;
}

.badge-yolo {
    background: rgba(59, 130, 246, 0.1);
    color: #60a5fa;
    border-color: rgba(59, 130, 246, 0.3);
}

.badge-effnet {
    background: rgba(16, 185, 129, 0.1);
    color: #34d399;
    border-color: rgba(16, 185, 129, 0.3);
}

.app-footer {
    text-align: center;
    padding: 1rem;
    color: #475569;
    font-size: 0.8rem;
    border-top: 1px solid rgba(148, 163, 184, 0.1);
    margin-top: 1rem;
}
"""

# ── Gradio UI ──
theme = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

with gr.Blocks(theme=theme, title=settings.APP_TITLE, css=CUSTOM_CSS) as demo:

    # ── Header ──
    gr.HTML("""
    <div class="app-header">
        <h1>🌿 Crop Disease Diagnosis System</h1>
        <p>
            Upload a clear image of a crop leaf. Our two-stage deep learning pipeline will
            <strong>detect</strong> diseased regions and <strong>classify</strong> the specific disease.
        </p>
        <div style="margin-top: 0.8rem;">
            <span class="model-badge badge-yolo">🎯 YOLOv11m — Detection</span>
            <span class="model-badge badge-effnet">🧠 EfficientNet-B0 — Classification</span>
            <span class="model-badge" style="background: rgba(168, 85, 247, 0.1); color: #c084fc; border-color: rgba(168, 85, 247, 0.3);">🔮 Grad-CAM — Explainability</span>
        </div>
    </div>
    """)

    # ── Main Interface ──
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="📸 Upload or Capture Leaf Image",
                type="numpy",
                height=420,
                sources=["upload", "webcam", "clipboard"],
            )
            submit_btn = gr.Button(
                "🔬 Analyze Leaf",
                variant="primary",
                size="lg",
            )

            # Show example images if available
            example_dir = os.path.join(PROJECT_ROOT, "examples")
            example_files = []
            if os.path.exists(example_dir):
                example_files = sorted([
                    os.path.join(example_dir, f)
                    for f in os.listdir(example_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ])
            if example_files:
                gr.Examples(examples=example_files, inputs=input_image, label="📂 Example Images")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("🎯 Detection (YOLOv11m)"):
                    output_detect = gr.Image(
                        label="YOLOv11m Object Detection — Diseased Region Localization",
                        height=420,
                    )
                with gr.TabItem("📊 Classification (EfficientNet-B0)"):
                    output_class = gr.Label(
                        num_top_classes=5,
                        label="Disease Probability Distribution",
                    )
                with gr.TabItem("🔮 Explainability (Grad-CAM)"):
                    output_gradcam = gr.Image(
                        label="Grad-CAM Heatmap (Focus regions for the predicted class)",
                        height=420,
                    )

    submit_btn.click(
        fn=diagnosis_pipeline,
        inputs=input_image,
        outputs=[output_detect, output_class, output_gradcam],
    )

    # ── Footer ──
    gr.HTML("""
    <div class="app-footer">
        <strong>Crop Disease Diagnosis System</strong><br/>
        Powered by <strong>YOLOv11m</strong> &amp; <strong>EfficientNet-B0</strong> ·
        FieldPlant Dataset (30 classes) · © 2026
    </div>
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=settings.PORT,
        share=False,
    )
