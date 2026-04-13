"""
Grad-CAM for EfficientNet-B0.

Hooks into the last convolutional block of the backbone to produce
a class-discriminative heatmap overlaid on the original image.
"""

import cv2
import torch
import numpy as np
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for EfficientNet-B0.

    The target layer is `backbone.features[-1]` — the last MBConv block
    before global average pooling, which contains the richest spatial info.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device

        # Hook into the last feature block of efficientnet_b0
        self.target_layer = model.backbone.features[-1]

        self._gradients: torch.Tensor | None = None
        self._activations: torch.Tensor | None = None

        # Register hooks
        self._fwd_hook = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

    # ── Hooks ──────────────────────────────────────────────────────
    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    # ── Core ───────────────────────────────────────────────────────
    def generate(
        self,
        img_tensor: torch.Tensor,
        class_idx: int | None = None,
    ) -> np.ndarray:
        """
        Run a forward+backward pass and return the raw Grad-CAM map (H×W, float32, 0–1).

        Args:
            img_tensor: Preprocessed tensor of shape (1, C, H, W) on self.device.
            class_idx:  Target class index. If None, uses the predicted class.

        Returns:
            cam: Grad-CAM heatmap as a numpy array (H_orig×W_orig), normalised to [0, 1].
        """
        self.model.eval()

        # Enable gradients for this forward pass
        img_tensor = img_tensor.requires_grad_(True)
        logits = self.model(img_tensor)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # Zero existing grads, then backprop on the target class score
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        # Grad-CAM: global-average-pool gradients → channel weights
        # gradients: (1, C, H, W) → weights: (C,)
        weights = self._gradients.mean(dim=(2, 3))[0]          # (C,)
        activations = self._activations[0]                      # (C, H, W)

        # Weighted combination of activation maps
        cam = torch.einsum("c,chw->hw", weights, activations)  # (H, W)
        cam = torch.relu(cam)                                   # keep only positive influence

        # Normalise to [0, 1]
        cam = cam.cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay(
        self,
        original_image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Blend Grad-CAM heatmap onto the original image.

        Args:
            original_image: RGB numpy array (H, W, 3), uint8.
            cam:            Normalised Grad-CAM map (any spatial size).
            alpha:          Heatmap blend weight (0 = original, 1 = heatmap only).
            colormap:       OpenCV colormap constant.

        Returns:
            blended: RGB numpy array (H, W, 3), uint8.
        """
        h, w = original_image.shape[:2]

        # Resize CAM to match the original image
        cam_resized = cv2.resize(cam, (w, h))

        # Apply colormap (returns BGR)
        heatmap_bgr = cv2.applyColorMap(
            np.uint8(255 * cam_resized), colormap
        )

        # Convert heatmap to RGB
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Blend
        blended = cv2.addWeighted(
            original_image.astype(np.uint8), 1 - alpha,
            heatmap_rgb,                      alpha,
            0,
        )
        return blended

    def remove_hooks(self):
        """Call when done to free hook memory."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()
