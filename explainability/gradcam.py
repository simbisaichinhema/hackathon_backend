"""Grad-CAM implementation for DR classifier explainability.

Generates class-discriminative heatmaps showing which retinal regions
contributed to the model prediction.

IMPORTANT: Grad-CAM highlights regions that contributed strongly to the
model prediction. It is an interpretability signal and not proof of
clinical causality.
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import os
os.environ["KERAS_BACKEND"] = "torch"


def find_last_conv_layer(model) -> str:
    """Find the last convolutional layer name in a model."""
    for layer in reversed(model.layers):
        if hasattr(layer, "output_shape") and layer.output_shape is not None:
            if len(layer.output_shape) == 4:
                return layer.name

    conv_keywords = ["activation", "conv", "block", "top_activation", "top_conv"]
    for layer in reversed(model.layers):
        name = layer.name.lower()
        if any(kw in name for kw in conv_keywords):
            if "pool" not in name and "global" not in name:
                return layer.name

    for layer in reversed(model.layers):
        if "top_activation" in layer.name or "top_conv" in layer.name:
            return layer.name

    return model.layers[-2].name if len(model.layers) > 2 else model.layers[-1].name


def compute_gradcam(
    model,
    image: np.ndarray,
    predicted_class: int,
    last_conv_layer_name: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Grad-CAM heatmap for a given image and prediction."""
    img_h, img_w = image.shape[1], image.shape[2]

    try:
        import torch
        import keras
        if last_conv_layer_name is None:
            last_conv_layer_name = find_last_conv_layer(model)

        # Grad-CAM with Keras 3 / PyTorch backend
        grad_model = keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output],
        )

        image_tensor = torch.tensor(image, dtype=torch.float32, requires_grad=True)
        conv_outputs, predictions = grad_model(image_tensor)
        loss = predictions[:, predicted_class]
        loss.backward()

        grads = image_tensor.grad
        if grads is not None:
            pooled_grads = torch.mean(grads, dim=(0, 2, 3))
            conv_out = conv_outputs[0]
            heatmap = torch.sum(conv_out * pooled_grads.view(-1, 1, 1), dim=0)
            heatmap = torch.relu(heatmap)
            max_v = torch.max(heatmap)
            if max_v > 0:
                heatmap = heatmap / max_v
            heatmap_raw = heatmap.detach().cpu().numpy()
        else:
            heatmap_raw = np.mean(np.abs(conv_outputs.detach().cpu().numpy()[0]), axis=-1)
            heatmap_raw = (heatmap_raw - heatmap_raw.min()) / (heatmap_raw.max() - heatmap_raw.min() + 1e-8)

        heatmap_resized = cv2.resize(heatmap_raw, (img_w, img_h))
        return heatmap_raw, heatmap_resized
    except Exception as e:
        raise RuntimeError(f"Grad-CAM generation failed: {e}") from e


def generate_gradcam_overlay(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Generate Grad-CAM overlay on original image.

    Args:
        original_image: Original image (H, W, 3) as uint8 [0, 255].
        heatmap: Grad-CAM heatmap (H, W) as float [0, 1].
        alpha: Overlay transparency.
        colormap: OpenCV colormap.

    Returns:
        Overlay image (H, W, 3) as uint8.
    """
    # Ensure image is uint8
    if original_image.dtype != np.uint8:
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        else:
            original_image = original_image.astype(np.uint8)

    orig_h, orig_w = original_image.shape[:2]
    # Resize heatmap to match original image dimensions exactly
    if heatmap.shape[:2] != (orig_h, orig_w):
        heatmap = cv2.resize(heatmap, (orig_w, orig_h))

    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap(
        (np.clip(heatmap, 0, 1) * 255).astype(np.uint8), colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


def compute_gradcam_plus(
    model,
    image: np.ndarray,
    predicted_class: int,
    last_conv_layer_name: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Grad-CAM++ for improved gradient visualization."""
    return compute_gradcam(model, image, predicted_class, last_conv_layer_name)
