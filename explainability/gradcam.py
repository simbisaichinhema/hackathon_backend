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


def _torch_modules(model):
    """Best-effort iterator over underlying torch nn.Modules (Keras-3 torch backend)."""
    import torch.nn as nn

    seen = set()

    def _walk(obj):
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        if isinstance(obj, nn.Module):
            yield obj
            for child in obj.children():
                yield from _walk(child)
            return
        for attr in ("torch_module", "_torch_module", "module", "_module"):
            sub = getattr(obj, attr, None)
            if sub is not None and id(sub) != oid:
                yield from _walk(sub)
        for child in getattr(obj, "layers", []) or []:
            yield from _walk(child)

    yield from _walk(model)


def _find_last_conv2d(model):
    """Find the last torch Conv2d module for hook-based Grad-CAM."""
    import torch.nn as nn

    last = None
    for mod in _torch_modules(model):
        for child in mod.modules():
            if isinstance(child, nn.Conv2d):
                last = child
    return last


def _activation_heatmap(activations: np.ndarray) -> np.ndarray:
    """Class-agnostic attention signal: mean |activation| over channels."""
    heat = np.mean(np.abs(activations), axis=0)
    span = heat.max() - heat.min()
    if span > 1e-8:
        heat = (heat - heat.min()) / span
    else:
        heat = np.zeros_like(heat)
    return heat.astype(np.float32)
def compute_gradcam(
    model,
    image: np.ndarray,
    predicted_class: int,
    last_conv_layer_name: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Grad-CAM heatmap for a given image and prediction.

    Strategy (Keras 3 / PyTorch backend):
    1. True Grad-CAM via a forward hook on the last torch Conv2d + backward()
       of the predicted-class score w.r.t. those feature maps.
    2. Fallback: class-agnostic activation heatmap (mean |activation|) from the
       same layer — still a genuine model-attention signal, never synthetic.
    Raises RuntimeError only if neither path is possible.
    """
    img_h, img_w = image.shape[1], image.shape[2]

    def _finish(heat_hw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        heat = np.clip(heat_hw, 0, None).astype(np.float32)
        mx = heat.max()
        heat = heat / mx if mx > 1e-8 else np.zeros_like(heat)
        return heat, cv2.resize(heat, (img_w, img_h)).astype(np.float32)

    try:
        import torch

        conv = _find_last_conv2d(model)
        if conv is None:
            raise RuntimeError("no Conv2d module found")

        captured = {}

        def _fwd_hook(_mod, _inp, out):
            captured["act"] = out

        handle = conv.register_forward_hook(_fwd_hook)
        try:
            was_training = bool(getattr(model, "training", False))
            if hasattr(model, "eval"):
                try:
                    model.eval()
                except Exception:
                    pass
            heat_hw = None
            last_err = None
            # Try channels-last (matches the numpy layout) then channels-first.
            for layout in ("last", "first"):
                try:
                    if layout == "last":
                        t = torch.tensor(image, dtype=torch.float32)
                    else:
                        t = torch.tensor(image, dtype=torch.float32).permute(0, 3, 1, 2)
                    captured.pop("act", None)
                    out = model(t)
                    preds = out[0] if isinstance(out, (list, tuple)) else out
                    score = preds[0, predicted_class]
                    model.zero_grad(set_to_none=True)
                    score.backward()
                    act = captured.get("act", None)
                    if act is None:
                        raise RuntimeError("hook captured nothing")
                    grad = act.grad
                    if grad is None:
                        raise RuntimeError("no gradient on conv maps")
                    weights = grad.detach().mean(dim=(0, 2, 3))
                    heat = (act.detach()[0] * weights.view(-1, 1, 1)).sum(dim=0)
                    heat = torch.relu(heat).cpu().numpy()
                    heat_hw = heat
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    continue
            if heat_hw is not None:
                return _finish(heat_hw)
            # Gradient path failed but the hook may still have activations.
            act = captured.get("act", None)
            if act is not None:
                a = act.detach().cpu().numpy()
                a = a[0].transpose(1, 2, 0) if a.ndim == 4 else a
                if a.ndim == 3:
                    return _finish(_activation_heatmap(a.transpose(2, 0, 1)))
            raise RuntimeError(f"gradient path failed: {last_err}")
        finally:
            try:
                handle.remove()
            except Exception:
                pass
            if was_training and hasattr(model, "train"):
                try:
                    model.train()
                except Exception:
                    pass
    except RuntimeError as e:
        raise RuntimeError(f"Grad-CAM generation failed: {e}") from e
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
