"""MuRIL + ViT models for IndicSent-M."""

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
from torch import Tensor, nn
from transformers import AutoModel, ViTModel

TEXT_MODEL = "google/muril-base-cased"
IMAGE_MODEL = "google/vit-base-patch16-224"
MAX_LEN = 128
CLASS_NAMES = ["Negative", "Neutral", "Positive"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IndicSentModel(nn.Module):
    """Unimodal or gated multimodal classifier."""

    def __init__(
        self,
        mode: str = "multimodal",
        dropout: float = 0.3,
        image_dim: int = 768,
    ) -> None:
        super().__init__()
        if mode not in {"text", "image", "multimodal"}:
            raise ValueError(f"Unsupported mode: {mode}")

        self.mode = mode
        self.text_encoder: Optional[nn.Module] = None
        self.image_encoder: Optional[nn.Module] = None

        if mode in {"text", "multimodal"}:
            self.text_encoder = AutoModel.from_pretrained(TEXT_MODEL)
        if mode in {"image", "multimodal"}:
            try:
                self.image_encoder = ViTModel.from_pretrained(
                    IMAGE_MODEL, attn_implementation="eager"
                )
            except TypeError:
                self.image_encoder = ViTModel.from_pretrained(IMAGE_MODEL)

        if mode == "text":
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(768, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 3),
            )
        elif mode == "image":
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(image_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 3),
            )
        else:
            self.text_proj = nn.Sequential(
                nn.Linear(768, 256),
                nn.LayerNorm(256),
                nn.GELU(),
            )
            self.image_proj = nn.Sequential(
                nn.Linear(image_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
            )
            self.gate = nn.Sequential(nn.Linear(512, 256), nn.Sigmoid())
            self.classifier = nn.Sequential(
                nn.LayerNorm(1024),
                nn.Dropout(dropout),
                nn.Linear(1024, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 3),
            )

    def encode_text(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if self.text_encoder is None:
            raise RuntimeError("This model has no text encoder.")
        hidden = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    def encode_image(self, pixel_values: Tensor) -> Tensor:
        if self.image_encoder is None:
            raise RuntimeError("This model has no image encoder.")
        return self.image_encoder(pixel_values=pixel_values).last_hidden_state[:, 0]

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        pixel_values: Optional[Tensor] = None,
    ) -> Tensor:
        if self.mode == "text":
            if input_ids is None or attention_mask is None:
                raise ValueError("Text mode requires input_ids and attention_mask.")
            return self.classifier(self.encode_text(input_ids, attention_mask))
        if self.mode == "image":
            if pixel_values is None:
                raise ValueError("Image mode requires pixel_values.")
            return self.classifier(self.encode_image(pixel_values))

        if input_ids is None or attention_mask is None or pixel_values is None:
            raise ValueError(
                "Multimodal mode requires input_ids, attention_mask, and pixel_values."
            )
        text_features = self.text_proj(self.encode_text(input_ids, attention_mask))
        image_features = self.image_proj(self.encode_image(pixel_values))
        gate = self.gate(torch.cat([text_features, image_features], dim=-1))
        fused = torch.cat(
            [
                gate * text_features + (1.0 - gate) * image_features,
                text_features,
                image_features,
                torch.abs(text_features - image_features),
            ],
            dim=-1,
        )
        return self.classifier(fused)


def _extract_state_dict(checkpoint: Any) -> Mapping[str, Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping) and candidate:
                return candidate
        if checkpoint and all(isinstance(value, Tensor) for value in checkpoint.values()):
            return checkpoint
    raise TypeError("Checkpoint does not contain a PyTorch model state_dict.")


def load_checkpoint(
    model: IndicSentModel, checkpoint_path: str | Path, device: torch.device = DEVICE
) -> IndicSentModel:
    """Load a checkpoint, requiring an exact state-dict key and shape match."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
    return model.to(device).eval()


def load_models(checkpoint_paths: Dict[str, str | Path]) -> Dict[str, IndicSentModel]:
    """Create and strictly load the text, image, and multimodal checkpoints."""
    expected_modes = {
        "text": "text",
        "image": "image",
        "multimodal": "multimodal",
    }
    missing = [
        str(checkpoint_paths[name])
        for name in expected_modes
        if name not in checkpoint_paths or not Path(checkpoint_paths[name]).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Required model checkpoint(s) not found: " + ", ".join(missing)
        )

    models: Dict[str, IndicSentModel] = {}
    for name, mode in expected_modes.items():
        if name not in checkpoint_paths:
            raise KeyError(f"Missing checkpoint path for {name!r}.")
        model = IndicSentModel(mode=mode)
        models[name] = load_checkpoint(model, checkpoint_paths[name], DEVICE)
    return models
