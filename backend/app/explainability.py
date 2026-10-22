
from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance


def png_base64(image: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG string."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def analyze_visual_features(image: Image.Image) -> dict:
    """
    Calculate descriptive pixel statistics.

    These are image measurements, not independent coral-health diagnoses.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    gray = (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    )

    histograms = {}

    for index, channel in enumerate(("red", "green", "blue")):
        counts, _ = np.histogram(
            rgb[:, :, index],
            bins=8,
            range=(0.0, 1.0),
        )
        total = max(int(counts.sum()), 1)
        histograms[channel] = [
            round(float(value / total), 4)
            for value in counts
        ]

    gy, gx = np.gradient(gray)
    edge_strength = np.sqrt(gx * gx + gy * gy)

    return {
        "image_width": image.width,
        "image_height": image.height,
        "mean_rgb": {
            "red": round(float(rgb[:, :, 0].mean()), 4),
            "green": round(float(rgb[:, :, 1].mean()), 4),
            "blue": round(float(rgb[:, :, 2].mean()), 4),
        },
        "mean_brightness": round(float(gray.mean()), 4),
        "brightness_std": round(float(gray.std()), 4),
        "edge_strength_mean": round(
            float(edge_strength.mean()), 5
        ),
        "rgb_histogram_8_bins": histograms,
        "interpretation_note": (
            "These measurements describe image pixels only. "
            "They do not independently diagnose bleaching or coral health."
        ),
    }


def enhance_image(image: Image.Image) -> Image.Image:
    """Apply conservative visual enhancement for human inspection."""
    result = ImageEnhance.Color(image.convert("RGB")).enhance(1.12)
    result = ImageEnhance.Contrast(result).enhance(1.12)
    result = ImageEnhance.Sharpness(result).enhance(1.15)
    return result


def _extract_tensor(output):
    """Extract a tensor from a transformer module's output."""
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item

    raise RuntimeError(
        "The selected transformer layer did not return a tensor."
    )


def _normalize_attribution(cam: np.ndarray) -> np.ndarray:
    """
    Normalize non-negative attribution for visualization.

    Percentile scaling improves display contrast without changing
    the classifier's prediction.
    """
    cam = np.asarray(cam, dtype=np.float32)
    cam = np.nan_to_num(
        cam,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    cam = np.maximum(cam, 0.0)

    low = float(np.percentile(cam, 2))
    high = float(np.percentile(cam, 99))

    if high - low > 1e-8:
        cam = np.clip(
            (cam - low) / (high - low),
            0.0,
            1.0,
        )
    else:
        maximum = float(cam.max())
        if maximum > 1e-8:
            cam = cam / maximum
        else:
            cam = np.zeros_like(cam, dtype=np.float32)

    return cam


def _create_heatmap(cam: np.ndarray) -> Image.Image:
    """Convert normalized attribution into an RGB heatmap."""
    x = np.clip(cam, 0.0, 1.0)

    # Blue -> cyan -> yellow -> red.
    red = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)

    heat_rgb = np.stack([red, green, blue], axis=-1)
    heat_rgb = np.uint8(np.clip(heat_rgb * 255.0, 0, 255))

    return Image.fromarray(heat_rgb)


def make_heatmap_overlay(
    image: Image.Image,
    model,
    image_processor,
    target_class: int | None = None,
) -> dict:
    """
    Generate gradient-based transformer patch attribution.

    Returns:
        heatmap_png_base64:
            Color-coded model attribution.
        overlay_png_base64:
            Attribution blended with the original image.
        attribution_map_png_base64:
            Grayscale attribution intensity map.

    This is a model-attribution visualization, not a segmentation mask
    or an independent ecological diagnosis.
    """
    model.eval()

    original = image.convert("RGB")
    device = next(model.parameters()).device
    vision_model = getattr(model, "vision_model", None)

    if vision_model is None:
        raise RuntimeError(
            "This model does not expose a compatible vision_model."
        )

    encoder = getattr(vision_model, "encoder", None)
    layers = getattr(encoder, "layers", None)

    if layers is None or len(layers) == 0:
        raise RuntimeError(
            "Could not locate transformer encoder layers for attribution."
        )

    # Capture the final transformer block output.
    # Its output generally contains spatial tokens and may also contain
    # a class token, depending on the model architecture.
    target_layer = layers[-1]
    captured: dict[str, torch.Tensor] = {}

    def capture_activation(_module, _inputs, output):
        activation = _extract_tensor(output)

        if not activation.requires_grad:
            raise RuntimeError(
                "Captured activations do not support gradients."
            )

        captured["activation"] = activation
        activation.retain_grad()

    hook = target_layer.register_forward_hook(capture_activation)

    try:
        inputs = image_processor(
            images=original,
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"].to(device)

        model.zero_grad(set_to_none=True)

        # Gradients must remain enabled for attribution.
        with torch.enable_grad():
            output = model(pixel_values=pixel_values)
            logits = output.logits

            if logits.ndim != 2 or logits.shape[0] != 1:
                raise RuntimeError(
                    f"Unexpected classifier logits shape: {tuple(logits.shape)}"
                )

            probabilities = torch.softmax(
                logits.detach(),
                dim=-1,
            )[0]

            selected_class = (
                int(probabilities.argmax().item())
                if target_class is None
                else int(target_class)
            )

            if not 0 <= selected_class < logits.shape[-1]:
                raise ValueError(
                    f"Invalid target class index {selected_class}; "
                    f"model has {logits.shape[-1]} output classes."
                )

            score = logits[0, selected_class]
            score.backward()

        activation = captured.get("activation")

        if activation is None:
            raise RuntimeError(
                "The target transformer layer was not captured."
            )

        gradient = activation.grad

        if gradient is None:
            raise RuntimeError(
                "No gradients were produced for the target layer."
            )

        activations = activation.detach()[0]
        gradients = gradient.detach()[0]

        if activations.ndim != 2 or gradients.shape != activations.shape:
            raise RuntimeError(
                "Unexpected transformer activation shape: "
                f"{tuple(activations.shape)}."
            )

        token_count, hidden_size = activations.shape

        # Identify a square patch-token grid.
        # Some vision transformers prepend one class token.
        grid_size = int(np.sqrt(token_count))

        if grid_size * grid_size != token_count:
            grid_size = int(np.sqrt(token_count - 1))

            if (
                token_count > 1
                and grid_size * grid_size == token_count - 1
            ):
                activations = activations[1:, :]
                gradients = gradients[1:, :]
            else:
                raise RuntimeError(
                    "Cannot reshape transformer tokens into a square "
                    f"spatial grid. Token count: {token_count}."
                )

        # Grad-CAM channel weights: average gradients over spatial tokens.
        channel_weights = gradients.mean(dim=0)

        # Positive gradient-weighted activation map.
        cam = torch.relu(
            (activations * channel_weights.unsqueeze(0)).sum(dim=-1)
        )

        cam = cam.reshape(1, 1, grid_size, grid_size)

        # Upsample to the original image dimensions.
        cam = F.interpolate(
            cam,
            size=(original.height, original.width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        cam_array = cam.detach().cpu().numpy()
        cam_array = _normalize_attribution(cam_array)

        # Build visualization outputs.
        heatmap_image = _create_heatmap(cam_array)
        grayscale_map = Image.fromarray(
            np.uint8(cam_array * 255.0),
            mode="L",
        )

        original_rgb = np.asarray(
            original,
            dtype=np.float32,
        )

        heat_rgb = np.asarray(
            heatmap_image,
            dtype=np.float32,
        )

        # Use stronger color blending for higher attribution.
        # Low-attribution regions retain more original-image detail.
        alpha = (0.65 * cam_array)[..., np.newaxis]

        overlay_rgb = np.clip(
            original_rgb * (1.0 - alpha) + heat_rgb * alpha,
            0.0,
            255.0,
        ).astype(np.uint8)

        overlay_image = Image.fromarray(overlay_rgb, mode="RGB")

        id2label = getattr(model.config, "id2label", {}) or {}

        def class_label(index: int) -> str:
            return str(id2label.get(index, id2label.get(str(index), index)))

        return {
            "target_class": class_label(selected_class),
            "target_class_index": selected_class,
            "probabilities": {
                class_label(index): round(float(probabilities[index]), 6)
                for index in range(len(probabilities))
            },
            "heatmap_png_base64": png_base64(heatmap_image),
            "overlay_png_base64": png_base64(overlay_image),
            "attribution_map_png_base64": png_base64(grayscale_map),
            "patch_grid_size": [grid_size, grid_size],
            "method": "Gradient-weighted transformer patch attribution",
            "interpretation_note": (
                "Highlighted regions represent positive gradient-weighted "
                "activations associated with the selected class score. "
                "The heatmap is an approximate model-attribution display, "
                "not a coral segmentation mask, proof of bleaching, or "
                "causal ecological diagnosis. Color scaling is adjusted "
                "for visualization. Validate explanations on representative "
                "images before relying on them."
            ),
        }

    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)
