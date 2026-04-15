import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from typing import Dict, Tuple, Optional
from .model import FLEfficientNet
from .yolo_manager import YOLOManager
from .config import settings
from .logger import logger
from .dataset import IMAGENET_MEAN, IMAGENET_STD
from .gradcam import GradCAM


class DiseaseInferenceEngine:
    """
    Two-Track Crop Disease Diagnosis Pipeline.

    Track A — Detection (YOLOv8m, 30 FieldPlant classes):
        Locates diseased regions in field images → bounding boxes + per-box labels.

    Track B — Classification (EfficientNet-B0, 17 classes):
        Whole-image disease classification → probability distribution.
        Trained on Crop Pest and Disease Detection (Cassava + Maize + Tomato).

    Track C — Explainability (Grad-CAM):
        Gradient-weighted Class Activation Mapping on the top predicted class.
        Overlaid on the original image to highlight the most influential regions.

    All three tracks run on every prediction and are displayed in the Gradio UI.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Inference Engine initializing on device: {self.device}")

        self.detector   = None
        self.classifier = None
        self.gradcam    = None

        # EfficientNet preprocessing (must match training transforms)
        self.transform = transforms.Compose([
            transforms.Resize((settings.INPUT_SIZE, settings.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self._load_models()

    # ── Model Loading ──────────────────────────────────────────────────────────
    def _load_models(self):
        """Load both models with graceful fallback."""

        # ── Track A: YOLO Detector ──
        yolo_path = Path(settings.YOLO_PATH)
        if yolo_path.exists():
            try:
                self.detector = YOLOManager(model_path=str(yolo_path))
                logger.info(f"YOLO loaded from trained weights: {yolo_path}")
            except Exception as e:
                logger.error(f"Failed to load YOLO from {yolo_path}: {e}")

        if self.detector is None:
            logger.info(f"Falling back to pretrained {settings.YOLO_FALLBACK_MODEL}")
            try:
                self.detector = YOLOManager(model_path=settings.YOLO_FALLBACK_MODEL)
            except Exception as e:
                logger.critical(f"Fallback YOLO also failed: {e}")

        # ── Track B: EfficientNet Classifier ──
        eff_path = Path(settings.EFFICIENTNET_PATH)
        try:
            self.classifier = FLEfficientNet(
                num_classes=settings.NUM_CLASSES,
                pretrained=True,
            )
            if eff_path.exists():
                state_dict = torch.load(str(eff_path), map_location=self.device)

                # Remap keys: if saved without 'backbone.' prefix, add it
                new_state_dict = {}
                for k, v in state_dict.items():
                    if not k.startswith("backbone."):
                        new_state_dict[f"backbone.{k}"] = v
                    else:
                        new_state_dict[k] = v

                self.classifier.load_state_dict(new_state_dict)
                logger.info(f"EfficientNet loaded with trained weights from {eff_path}")
            else:
                logger.warning(
                    f"EfficientNet weights not found at {eff_path}. "
                    "Using pretrained ImageNet backbone (classification head is random)."
                )
            self.classifier.to(self.device)
            self.classifier.eval()

            # ── Track C: Grad-CAM (attached to the loaded classifier) ──
            self.gradcam = GradCAM(model=self.classifier, device=self.device)
            logger.info("Grad-CAM initialised on backbone.features[-1]")

        except Exception as e:
            logger.error(f"Failed to initialize EfficientNet: {e}")
            self.classifier = None
            self.gradcam    = None

    # ── Inference ──────────────────────────────────────────────────────────────
    def predict(
        self,
        image: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Dict[str, float], Optional[np.ndarray]]:
        """
        Run all three tracks on the input image.

        Args:
            image: Input image as numpy array (H, W, 3) in RGB, uint8.

        Returns:
            annotated_image:  YOLO bounding-box overlay (numpy array, RGB).
            classification:   Dict of {class_name: probability} — top-5 results.
            gradcam_overlay:  Grad-CAM heatmap blended onto the original image.
        """
        annotated_image  = image.copy()
        classification   = {}
        gradcam_overlay  = None

        # ── Track A: Detection ──
        if self.detector:
            try:
                yolo_results    = self.detector.predict(image)
                annotated_image = yolo_results[0].plot()
                num_boxes       = len(yolo_results[0].boxes)
                logger.info(f"Detection: {num_boxes} diseased regions found")
            except Exception as e:
                logger.error(f"Detection failed: {e}")

        # ── Tracks B + C: Classification + Grad-CAM ──
        if self.classifier:
            try:
                pil_image  = Image.fromarray(image) if isinstance(image, np.ndarray) else image
                img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

                # ── Track B: Classification (no-grad, fast) ──
                with torch.no_grad():
                    logits = self.classifier(img_tensor)
                    probs  = torch.nn.functional.softmax(logits, dim=1)[0]

                top_k = torch.topk(probs, min(5, len(probs)))
                top_class_idx = top_k.indices[0].item()
                classification = {
                    settings.CLASS_NAMES[idx]: round(float(prob), 4)
                    for idx, prob in zip(top_k.indices.tolist(), top_k.values.tolist())
                }
                logger.info(f"Classification: top result = {list(classification.items())[0]}")

                # ── Track C: Grad-CAM (requires grad, separate pass) ──
                if self.gradcam is not None:
                    try:
                        # Fresh tensor with grad enabled
                        img_tensor_grad = self.transform(pil_image).unsqueeze(0).to(self.device)
                        cam = self.gradcam.generate(img_tensor_grad, class_idx=top_class_idx)
                        gradcam_overlay = self.gradcam.overlay(image, cam, alpha=0.45)
                        logger.info("Grad-CAM heatmap generated")
                    except Exception as e:
                        logger.error(f"Grad-CAM failed: {e}")
                        gradcam_overlay = None

            except Exception as e:
                logger.error(f"Classification failed: {e}")
                classification = {"Error: Classification Failed": 1.0}
        else:
            classification = {"Info: EfficientNet not loaded": 1.0}

        return annotated_image, classification, gradcam_overlay
