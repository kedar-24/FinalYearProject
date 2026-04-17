import gradio as gr
import numpy as np
import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference_engine import DiseaseInferenceEngine
from src.config import settings
from src.logger import logger


engine = None
try:
    engine = DiseaseInferenceEngine()
    logger.info("Inference Engine started successfully.")
except Exception as e:
    logger.critical(f"Failed to start Inference Engine: {e}")


def diagnosis_pipeline(image: np.ndarray):
    if engine is None:
        return None, {"Error": "System initialization failed. Check logs."}, None
    if image is None:
        return None, {"Error": "No image provided."}, None
    try:
        detection_img, classification_results, gradcam_img = engine.predict(image)
        return detection_img, classification_results, gradcam_img
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return image, {"Error": str(e)}, None


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif !important;
    box-sizing: border-box;
}

body, .gradio-container {
    background: #f8fafc !important;
    min-height: 100vh;
}

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 2rem 1.5rem !important;
}

footer { display: none !important; }

/* ── Header ── */
.header-wrap {
    text-align: center;
    padding: 2.5rem 1rem 2rem;
    margin-bottom: 2rem;
}

.header-wrap h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 0.5rem;
    letter-spacing: -0.02em;
}

.header-wrap p {
    font-size: 1rem;
    color: #64748b;
    margin: 0 auto;
    max-width: 520px;
    line-height: 1.6;
}

.pill-row {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    margin-top: 1.25rem;
    flex-wrap: wrap;
}

.pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid transparent;
}

.pill-blue  { background: #eff6ff; color: #3b82f6; border-color: #bfdbfe; }
.pill-green { background: #f0fdf4; color: #16a34a; border-color: #bbf7d0; }
.pill-purple{ background: #faf5ff; color: #9333ea; border-color: #e9d5ff; }

/* ── Cards ── */
.card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

/* ── Upload zone overrides ── */
.upload-card { padding: 1.25rem; }

/* ── Analyze button ── */
.analyze-btn {
    width: 100% !important;
    margin-top: 0.85rem !important;
    height: 48px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    background: #10b981 !important;
    border: none !important;
    color: #fff !important;
    cursor: pointer;
    transition: background 0.18s;
}

.analyze-btn:hover { background: #059669 !important; }

/* ── Section label ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 0.6rem;
}

/* ── Results tabs override ── */
.tab-nav button {
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
}

.tab-nav button.selected {
    color: #0f172a !important;
    border-bottom: 2px solid #10b981 !important;
}

/* ── Footer ── */
.footer-wrap {
    text-align: center;
    margin-top: 2.5rem;
    padding-top: 1.25rem;
    border-top: 1px solid #e2e8f0;
    font-size: 0.8rem;
    color: #94a3b8;
}

.footer-wrap strong { color: #64748b; }
"""

with gr.Blocks(title=settings.APP_TITLE) as demo:

    # ── Header ──
    gr.HTML("""
    <div class="header-wrap">
        <h1>Crop Disease Diagnosis</h1>
        <p>Upload a leaf image. Our AI pipeline detects diseased regions, classifies the disease, and highlights the decision area.</p>
        <div class="pill-row">
            <span class="pill pill-blue">YOLOv11m &mdash; Detection</span>
            <span class="pill pill-green">EfficientNet-B0 &mdash; Classification</span>
            <span class="pill pill-purple">Grad-CAM &mdash; Explainability</span>
        </div>
    </div>
    """)

    # ── Main layout ──
    with gr.Row(equal_height=False):

        # Left — upload
        with gr.Column(scale=4, min_width=300):
            with gr.Group(elem_classes="card upload-card"):
                gr.HTML('<div class="section-label">Input</div>')
                input_image = gr.Image(
                    label="",
                    type="numpy",
                    height=360,
                    sources=["upload", "webcam", "clipboard"],
                    show_label=False,
                )
                submit_btn = gr.Button(
                    "Analyze Leaf",
                    variant="primary",
                    elem_classes="analyze-btn",
                )

            # Example images
            example_dir = os.path.join(PROJECT_ROOT, "examples")
            example_files = []
            if os.path.exists(example_dir):
                example_files = sorted([
                    os.path.join(example_dir, f)
                    for f in os.listdir(example_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ])
            if example_files:
                gr.Examples(
                    examples=example_files,
                    inputs=input_image,
                    label="Examples",
                )

        # Right — results
        with gr.Column(scale=6, min_width=400):
            with gr.Group(elem_classes="card"):
                gr.HTML('<div class="section-label">Results</div>')
                with gr.Tabs():
                    with gr.TabItem("Detection"):
                        output_detect = gr.Image(
                            label="",
                            show_label=False,
                            height=360,
                        )
                    with gr.TabItem("Classification"):
                        output_class = gr.Label(
                            num_top_classes=5,
                            label="",
                            show_label=False,
                        )
                    with gr.TabItem("Grad-CAM"):
                        output_gradcam = gr.Image(
                            label="",
                            show_label=False,
                            height=360,
                        )

    submit_btn.click(
        fn=diagnosis_pipeline,
        inputs=input_image,
        outputs=[output_detect, output_class, output_gradcam],
    )

    # ── Footer ──
    gr.HTML("""
    <div class="footer-wrap">
        <strong>Crop Disease Diagnosis System</strong> &nbsp;&middot;&nbsp;
        YOLOv11m &amp; EfficientNet-B0 &nbsp;&middot;&nbsp; 17 disease classes &nbsp;&middot;&nbsp; &copy; 2026
    </div>
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=settings.PORT,
        share=False,
        css=CSS,
    )
