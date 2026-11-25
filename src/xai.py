"""Image and text explainability utilities for IndicSent-M."""

import html
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn

from .model import MAX_LEN, IndicSentModel


def _last_vit_block(vit_model: nn.Module) -> nn.Module:
    layers = getattr(vit_model, "layers", None)
    if layers is not None:
        return layers[-1]
    encoder = getattr(vit_model, "encoder", None)
    blocks = getattr(encoder, "layer", None)
    if blocks is not None:
        return blocks[-1]
    raise AttributeError("Could not locate the final ViT encoder block.")


def _attention_rollout(vit_model: nn.Module, pixel_values: Tensor) -> Tensor:
    output = vit_model(pixel_values=pixel_values, output_attentions=True)
    attentions = getattr(output, "attentions", None)
    if not attentions:
        raise RuntimeError("ViT did not return attentions for rollout.")

    token_count = attentions[-1].shape[-1]
    joint = torch.eye(token_count, device=pixel_values.device)
    for attention in attentions:
        averaged = attention[0].mean(dim=0)
        averaged = averaged + torch.eye(token_count, device=averaged.device)
        averaged = averaged / averaged.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        joint = averaged @ joint
    patch_values = joint[0, 1:]
    side = int(patch_values.numel() ** 0.5)
    if side * side != patch_values.numel():
        raise RuntimeError("Attention rollout patch count is not a square grid.")
    return patch_values.reshape(1, 1, side, side)


def _make_overlay(image: Image.Image, patch_cam: Tensor) -> Image.Image:
    original = image.convert("RGB")
    cam = F.interpolate(
        patch_cam.unsqueeze(0).unsqueeze(0),
        size=(original.height, original.width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    cam = cam - cam.min()
    cam = cam / cam.max().clamp(min=1e-8)
    heat = cam.detach().cpu().numpy()

    colors = np.array(
        [[0.08, 0.12, 0.65], [0.0, 0.85, 0.9], [0.95, 0.9, 0.1], [0.9, 0.05, 0.02]],
        dtype=np.float32,
    )
    positions = np.linspace(0.0, 1.0, len(colors))
    red = np.interp(heat, positions, colors[:, 0])
    green = np.interp(heat, positions, colors[:, 1])
    blue = np.interp(heat, positions, colors[:, 2])
    rgb = (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)
    heatmap = Image.fromarray(rgb, mode="RGB")
    return Image.blend(original, heatmap, alpha=0.42)


def generate_gradcam(
    model: IndicSentModel,
    pixel_values: Tensor,
    image: Image.Image,
    target_class: Optional[int] = None,
    input_ids: Optional[Tensor] = None,
    attention_mask: Optional[Tensor] = None,
) -> Image.Image:
    """Return a ViT patch-CAM overlay, using attention rollout for flat gradients."""
    if model.image_encoder is None:
        raise ValueError("Grad-CAM requires a model with an image encoder.")
    layer = _last_vit_block(model.image_encoder)
    norm = getattr(layer, "layernorm_before", None)
    if norm is None:
        raise AttributeError("Final ViT block has no layernorm_before module.")

    captured: dict[str, Tensor] = {}

    def capture_activation(_module: nn.Module, _inputs: tuple, output: Tensor) -> None:
        captured["activation"] = output
        if output.requires_grad:
            output.retain_grad()

    hook = norm.register_forward_hook(capture_activation)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            class_index = int(logits.argmax(dim=-1).item()) if target_class is None else target_class
            logits[0, class_index].backward()
        activation = captured.get("activation")
        gradient = None if activation is None else activation.grad
        if (
            activation is None
            or gradient is None
            or not torch.isfinite(gradient).all()
            or float(gradient.abs().max().item()) < 1e-10
        ):
            patch_cam = _attention_rollout(model.image_encoder, pixel_values)
        else:
            weights = gradient[:, 1:, :].mean(dim=1, keepdim=True)
            cam = F.relu((weights * activation[:, 1:, :]).sum(dim=-1))
            side = int(cam.shape[-1] ** 0.5)
            if side * side != cam.shape[-1]:
                raise RuntimeError("Grad-CAM patch count is not a square grid.")
            patch_cam = cam.reshape(1, 1, side, side)
        return _make_overlay(image, patch_cam[0, 0])
    finally:
        model.zero_grad(set_to_none=True)
        hook.remove()


def leave_one_out_attributions(
    model: IndicSentModel,
    tokenizer: object,
    text: str,
    target_class: int,
    pixel_values: Optional[Tensor] = None,
    batch_size: int = 8,
) -> List[Tuple[str, float]]:
    """Measure each word's effect on the selected class probability."""
    words = list(re.finditer(r"\b[\w'-]+\b", text, flags=re.UNICODE))
    if not words:
        return []
    if model.mode == "multimodal" and pixel_values is None:
        raise ValueError("Multimodal word attribution requires pixel_values.")

    def encode(value: str) -> dict[str, Tensor]:
        encoded = tokenizer(
            value,
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    original = encode(text)
    device = next(model.parameters()).device
    original = {key: value.to(device) for key, value in original.items()}
    image_batch = None if pixel_values is None else pixel_values.to(device)
    model.eval()
    with torch.inference_mode():
        original_logits = model(
            **original,
            **({"pixel_values": image_batch} if image_batch is not None else {}),
        )
        base_probability = torch.softmax(original_logits, dim=-1)[0, target_class].item()

    variants = [
        text[:match.start()] + (getattr(tokenizer, "mask_token", None) or "") + text[match.end():]
        for match in words
    ]
    probabilities: List[float] = []
    with torch.inference_mode():
        for start in range(0, len(variants), batch_size):
            batch = [encode(item) for item in variants[start : start + batch_size]]
            inputs = {
                key: torch.cat([item[key] for item in batch], dim=0).to(device)
                for key in ("input_ids", "attention_mask")
            }
            kwargs = {}
            if image_batch is not None:
                kwargs["pixel_values"] = image_batch.expand(len(batch), -1, -1, -1)
            logits = model(**inputs, **kwargs)
            probabilities.extend(
                torch.softmax(logits, dim=-1)[:, target_class].cpu().tolist()
            )
    return [
        (match.group(), float(base_probability - probability))
        for match, probability in zip(words, probabilities)
    ]


def render_word_attributions(
    text: str,
    attributions: Sequence[Tuple[str, float]],
    predicted_label: str,
) -> str:
    """Render word-level evidence with green support and red opposition."""
    queues: dict[str, List[float]] = {}
    for word, delta in attributions:
        queues.setdefault(word.casefold(), []).append(delta)
    pieces = []
    cursor = 0
    for match in re.finditer(r"\b[\w'-]+\b", text, flags=re.UNICODE):
        pieces.append(html.escape(text[cursor : match.start()]))
        word = match.group()
        deltas = queues.get(word.casefold(), [])
        delta = deltas.pop(0) if deltas else 0.0
        color = "#16803c" if delta >= 0 else "#bd3030"
        pieces.append(
            f'<span title="Effect on {html.escape(predicted_label)} probability: '
            f'{delta:+.3f}" style="color:{color};font-weight:600">'
            f"{html.escape(word)}</span>"
        )
        cursor = match.end()
    pieces.append(html.escape(text[cursor:]))
    return (
        '<div style="line-height:1.9;font-size:1.05rem">'
        + "".join(pieces)
        + "</div><small>Green supports the predicted sentiment; red opposes it.</small>"
    )
