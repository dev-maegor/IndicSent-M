"""OCR helpers for English and Romanized-Hinglish meme text."""

import re
from functools import lru_cache
from typing import Any, List

import emoji
import numpy as np
from PIL import Image, ImageEnhance


@lru_cache(maxsize=1)
def get_reader() -> Any:
    import easyocr

    return easyocr.Reader(["en"], verbose=False)


def clean_text(text: str) -> str:
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text(image: Image.Image) -> str:
    img = image.convert("RGB")
    width, height = img.size
    if max(width, height) < 900:
        img = img.resize((width * 3, height * 3), Image.Resampling.LANCZOS)
    img = ImageEnhance.Sharpness(img).enhance(1.6)

    detections = get_reader().readtext(
        np.asarray(img),
        detail=1,
        paragraph=False,
        contrast_ths=0.2,
        width_ths=0.5,
    )
    lines: List[str] = [
        detection[1].strip()
        for detection in detections
        if len(detection) >= 3 and float(detection[2]) >= 0.30 and detection[1].strip()
    ]
    return clean_text(" ".join(lines))
