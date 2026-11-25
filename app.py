"""Local Gradio application for IndicSent-M with modern UI and explainability."""

import html
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gradio as gr
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

from src.model import (
    CLASS_NAMES,
    DEVICE,
    IMAGE_MODEL,
    MAX_LEN,
    TEXT_MODEL,
    IndicSentModel,
    load_models,
)
from src.ocr import extract_text
from src.xai import (
    generate_gradcam,
    leave_one_out_attributions,
)

ROOT = Path(__file__).resolve().parent


def find_asset(relative_path: str) -> Path:
    candidates = (ROOT / relative_path, ROOT / Path(relative_path).name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


CHECKPOINTS = {
    name: find_asset(f"artifacts/best_{name}.pt")
    for name in ("text", "image", "multimodal")
}


@lru_cache(maxsize=1)
def get_models() -> Dict[str, IndicSentModel]:
    return load_models(CHECKPOINTS)


@lru_cache(maxsize=1)
def get_tokenizer():
    local_tokenizer = find_asset("artifacts/tokenizer")
    if (local_tokenizer / "tokenizer_config.json").is_file():
        return AutoTokenizer.from_pretrained(str(local_tokenizer))
    return AutoTokenizer.from_pretrained(TEXT_MODEL)


@lru_cache(maxsize=1)
def get_image_processor():
    return AutoImageProcessor.from_pretrained(IMAGE_MODEL)


def _model_inputs(model_name: str, text: str, image: Image.Image):
    tokenizer = get_tokenizer()
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt",
    )
    image_processor = get_image_processor()
    pixels = image_processor(images=image.convert("RGB"), return_tensors="pt")[
        "pixel_values"
    ]
    text_values = {
        "input_ids": encoded["input_ids"].to(DEVICE),
        "attention_mask": encoded["attention_mask"].to(DEVICE),
    }
    pixels = pixels.to(DEVICE)
    if model_name == "text":
        return text_values, pixels
    if model_name == "image":
        return {"pixel_values": pixels}, pixels
    return {**text_values, "pixel_values": pixels}, pixels


def _predict(model: IndicSentModel, **inputs) -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        return torch.softmax(model(**inputs), dim=-1)[0].cpu()


def render_enhanced_word_attributions(
    text: str,
    attributions: Sequence[Tuple[str, float]],
    predicted_label: str,
) -> str:
    """Render interactive token chips with scaled heatmaps, hover tooltips, and badges."""
    if not attributions:
        return "<p style='color: #64748b; font-style: italic;'>No text tokens detected for attribution.</p>"

    queues: dict[str, List[float]] = {}
    for word, delta in attributions:
        queues.setdefault(word.casefold(), []).append(delta)

    pieces = []
    cursor = 0
    pos_count = 0
    neg_count = 0
    max_abs_delta = max((abs(d) for _, d in attributions), default=0.01)
    if max_abs_delta < 1e-6:
        max_abs_delta = 0.01

    for match in re.finditer(r"\b[\w'-]+\b", text, flags=re.UNICODE):
        pieces.append(html.escape(text[cursor : match.start()]))
        word = match.group()
        deltas = queues.get(word.casefold(), [])
        delta = deltas.pop(0) if deltas else 0.0
        normalized = min(1.0, abs(delta) / max_abs_delta)
        alpha = 0.12 + 0.65 * normalized

        if delta > 0.005:
            pos_count += 1
            bg = f"rgba(16, 185, 129, {alpha:.2f})"
            border = "rgba(16, 185, 129, 0.5)"
            color = "#065f46"
            badge = f"<span style='font-size: 0.72rem; margin-left: 4px; padding: 1px 4px; border-radius: 4px; background: rgba(16, 185, 129, 0.25); font-weight: 700;'>+{delta:.2f}</span>"
            tooltip = f"Supports {predicted_label} (+{delta:.3f} probability effect)"
        elif delta < -0.005:
            neg_count += 1
            bg = f"rgba(239, 68, 68, {alpha:.2f})"
            border = "rgba(239, 68, 68, 0.5)"
            color = "#991b1b"
            badge = f"<span style='font-size: 0.72rem; margin-left: 4px; padding: 1px 4px; border-radius: 4px; background: rgba(239, 68, 68, 0.25); font-weight: 700;'>{delta:.2f}</span>"
            tooltip = f"Opposes {predicted_label} ({delta:.3f} probability effect)"
        else:
            bg = "rgba(148, 163, 184, 0.12)"
            border = "rgba(148, 163, 184, 0.25)"
            color = "#475569"
            badge = ""
            tooltip = f"Neutral impact on {predicted_label} ({delta:+.3f})"

        chip = (
            f'<span title="{html.escape(tooltip)}" '
            f'style="display: inline-flex; align-items: center; background: {bg}; border: 1px solid {border}; '
            f'color: {color}; font-weight: 600; padding: 2px 7px; margin: 2px 3px; border-radius: 6px; '
            f'line-height: 1.35; transition: transform 0.15s ease;" '
            f'onmouseover="this.style.transform=\'scale(1.05)\'" onmouseout="this.style.transform=\'scale(1)\'">'
            f'{html.escape(word)}{badge}</span>'
        )
        pieces.append(chip)
        cursor = match.end()

    pieces.append(html.escape(text[cursor:]))
    rendered_text = "".join(pieces)

    return f"""
    <div style="background: var(--background-fill-secondary, #f8fafc); border: 1px solid var(--border-color-primary, #e2e8f0); border-radius: 12px; padding: 14px 16px;">
        <div style="font-size: 1.05rem; line-height: 2.1; margin-bottom: 12px;">
            {rendered_text}
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 14px; font-size: 0.82rem; color: var(--body-text-color-subdued, #64748b); border-top: 1px solid var(--border-color-primary, #e2e8f0); padding-top: 10px; align-items: center;">
            <div style="display: flex; align-items: center; gap: 6px;">
                <span style="width: 12px; height: 12px; border-radius: 3px; background: rgba(16, 185, 129, 0.7); display: inline-block;"></span>
                <span>Supports <strong>{html.escape(predicted_label)}</strong> ({pos_count} words)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px;">
                <span style="width: 12px; height: 12px; border-radius: 3px; background: rgba(239, 68, 68, 0.7); display: inline-block;"></span>
                <span>Opposes <strong>{html.escape(predicted_label)}</strong> ({neg_count} words)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px;">
                <span style="width: 12px; height: 12px; border-radius: 3px; background: rgba(148, 163, 184, 0.4); display: inline-block;"></span>
                <span>Neutral / Low impact</span>
            </div>
            <div style="margin-left: auto; font-style: italic; color: #94a3b8;">
                Hover over tokens to view exact delta
            </div>
        </div>
    </div>
    """


def render_verdict_card(
    predicted_label: str,
    confidence_score: float,
    fusion_name: str,
    text_pred: str,
    text_conf: float,
    img_pred: str,
    img_conf: float,
) -> str:
    """Render a compact, theme-aware sentiment and modality summary."""
    accents = {
        "Negative": ("#fb7185", "rgba(244, 63, 94, .10)"),
        "Neutral": ("#fbbf24", "rgba(245, 158, 11, .10)"),
        "Positive": ("#34d399", "rgba(16, 185, 129, .10)"),
    }
    accent, tint = accents.get(predicted_label, ("#67e8f9", "rgba(34, 211, 238, .10)"))
    fusion_short = "Late fusion" if "late" in fusion_name.casefold() else "Gated fusion"
    aligned = text_pred == img_pred
    summary = (
        f"Text and image models agree on {html.escape(predicted_label.lower())}."
        if aligned
        else f"Text predicts {html.escape(text_pred.lower())} ({text_conf * 100:.1f}%); image predicts {html.escape(img_pred.lower())} ({img_conf * 100:.1f}%)."
    )
    status = "Modalities agree" if aligned else "Modalities differ"
    return f"""
    <section style="border:1px solid rgba(148,163,184,.20); border-left:4px solid {accent}; border-radius:12px; padding:18px 20px; background:{tint};">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:16px; flex-wrap:wrap;">
        <div>
          <div style="font-size:.72rem; letter-spacing:.09em; text-transform:uppercase; color:var(--body-text-color-subdued); font-weight:650;">Predicted sentiment</div>
          <div style="font-size:1.75rem; line-height:1.15; margin-top:5px; font-weight:750; letter-spacing:-.035em; color:var(--body-text-color);">{html.escape(predicted_label)}</div>
        </div>
        <div style="display:flex; gap:28px; align-items:flex-start;">
          <div><div style="font-size:.70rem; color:var(--body-text-color-subdued); text-transform:uppercase; letter-spacing:.07em;">Confidence</div><div style="font-size:1.18rem; color:var(--body-text-color); font-weight:700; margin-top:4px;">{confidence_score * 100:.1f}%</div></div>
          <div><div style="font-size:.70rem; color:var(--body-text-color-subdued); text-transform:uppercase; letter-spacing:.07em;">Fusion</div><div style="font-size:.92rem; color:var(--body-text-color); font-weight:600; margin-top:6px;">{fusion_short}</div></div>
        </div>
      </div>
      <div style="border-top:1px solid rgba(148,163,184,.18); margin-top:15px; padding-top:12px; display:flex; gap:9px; align-items:flex-start; color:var(--body-text-color); font-size:.84rem; line-height:1.5;">
        <span style="color:var(--body-text-color-subdued); font-weight:650; white-space:nowrap;">{status}</span><span>{summary}</span>
      </div>
    </section>
    """


def analyze_meme(
    image: Image.Image,
    text_override: str,
    fusion: str,
    alpha: float,
) -> Tuple[str, dict, pd.DataFrame, Image.Image, str, str]:
    if image is None:
        raise gr.Error("Please upload or select a meme image to run analysis.")
    image = image.convert("RGB")
    raw_text = (text_override or "").strip()
    is_ocr = False
    if not raw_text:
        raw_text = extract_text(image)
        is_ocr = True

    if not raw_text:
        raise gr.Error("No text was provided or detected via EasyOCR. Please enter a text override.")

    models = get_models()
    tokenizer = get_tokenizer()
    text_inputs, pixels = _model_inputs("text", raw_text, image)
    image_inputs, _ = _model_inputs("image", raw_text, image)
    multi_inputs, _ = _model_inputs("multimodal", raw_text, image)

    text_probs = _predict(models["text"], **text_inputs)
    image_probs = _predict(models["image"], **image_inputs)
    multi_probs = _predict(models["multimodal"], **multi_inputs)
    late_probs = alpha * text_probs + (1.0 - alpha) * image_probs
    final_probs = late_probs if fusion.casefold().startswith("late fusion") else multi_probs

    predicted_index = int(final_probs.argmax().item())
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = {name: float(final_probs[i]) for i, name in enumerate(CLASS_NAMES)}

    text_idx = int(text_probs.argmax().item())
    image_idx = int(image_probs.argmax().item())

    rows = []
    for name, probs in (
        ("Text model", text_probs),
        ("Image model", image_probs),
        ("Late Fusion", late_probs),
        ("Gated Multimodal", multi_probs),
    ):
        idx = int(probs.argmax().item())
        rows.append(
            {
                "Architecture": name,
                "Prediction": CLASS_NAMES[idx],
                "Confidence": f"{float(probs[idx])*100:.1f}%",
                "Negative": f"{float(probs[0])*100:.1f}%",
                "Neutral": f"{float(probs[1])*100:.1f}%",
                "Positive": f"{float(probs[2])*100:.1f}%",
            }
        )
    breakdown = pd.DataFrame(rows)

    verdict_card = render_verdict_card(
        predicted_label=predicted_label,
        confidence_score=float(final_probs[predicted_index]),
        fusion_name=fusion,
        text_pred=CLASS_NAMES[text_idx],
        text_conf=float(text_probs[text_idx]),
        img_pred=CLASS_NAMES[image_idx],
        img_conf=float(image_probs[image_idx]),
    )

    heatmap = generate_gradcam(
        models["multimodal"],
        pixels,
        image,
        target_class=predicted_index,
        input_ids=multi_inputs["input_ids"],
        attention_mask=multi_inputs["attention_mask"],
    )
    attributions = leave_one_out_attributions(
        models["multimodal"],
        tokenizer,
        raw_text,
        target_class=predicted_index,
        pixel_values=pixels,
    )
    explanation = render_enhanced_word_attributions(raw_text, attributions, predicted_label)

    source_tag = "Auto-extracted via EasyOCR" if is_ocr else "User Override"
    used_text_display = f"**Active Text ({source_tag}):** *\"{raw_text}\"*"

    return (
        verdict_card,
        gr.update(value=confidence, visible=True),
        gr.update(value=breakdown, visible=True),
        heatmap,
        explanation,
        used_text_display,
    )


def generate_grouped_benchmark_chart(raw_df: pd.DataFrame):
    """Generate a grouped side-by-side bar chart of benchmark metrics with values on top."""
    import matplotlib.pyplot as plt
    import numpy as np

    model_order = ["Text model", "Image model", "Gated Fusion", "Late Fusion (a=0.55)"]
    model_name_map = {
        "TEXT (MuRIL)": "Text model",
        "IMAGE (ViT)": "Image model",
        "MULTIMODAL (Gated Fusion)": "Gated Fusion",
        "MULTIMODAL (Late Fusion, a=0.55)": "Late Fusion (a=0.55)",
    }

    data_by_model = {
        "Text model": {"Accuracy": 31.20, "Macro F1": 30.97, "Weighted F1": 31.24},
        "Image model": {"Accuracy": 33.93, "Macro F1": 30.83, "Weighted F1": 30.98},
        "Gated Fusion": {"Accuracy": 34.53, "Macro F1": 30.46, "Weighted F1": 31.23},
        "Late Fusion (a=0.55)": {"Accuracy": 33.67, "Macro F1": 32.43, "Weighted F1": 32.78},
    }

    if not raw_df.empty and "Model" in raw_df.columns:
        for _, row in raw_df.iterrows():
            m_raw = str(row["Model"])
            m_clean = model_name_map.get(m_raw, m_raw)
            if m_clean in data_by_model:
                data_by_model[m_clean] = {
                    "Accuracy": round(float(row.get("Accuracy", 0)) * 100, 2),
                    "Macro F1": round(float(row.get("Macro F1", 0)) * 100, 2),
                    "Weighted F1": round(float(row.get("Weighted F1", 0)) * 100, 2),
                }

    x = np.arange(len(model_order))
    width = 0.24

    accs = [data_by_model[m]["Accuracy"] for m in model_order]
    macro_f1s = [data_by_model[m]["Macro F1"] for m in model_order]
    weighted_f1s = [data_by_model[m]["Weighted F1"] for m in model_order]

    fig, ax = plt.subplots(figsize=(8.5, 4.3), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")

    rects1 = ax.bar(x - width, accs, width, label="Accuracy", color="#4f46e5", edgecolor="none", zorder=3)
    rects2 = ax.bar(x, macro_f1s, width, label="Macro F1", color="#10b981", edgecolor="none", zorder=3)
    rects3 = ax.bar(x + width, weighted_f1s, width, label="Weighted F1", color="#f59e0b", edgecolor="none", zorder=3)

    ax.set_ylim(25, 38)
    ax.set_ylabel("Score (%)", fontsize=10, fontweight="bold", color="#334155")
    ax.set_title("Model Benchmark Metrics (%)", fontsize=12, fontweight="bold", pad=14, color="#1e293b")
    ax.set_xticks(x)
    ax.set_xticklabels(model_order, fontsize=9.5, fontweight="bold", color="#334155")
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=9, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#cbd5e1")

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(
                f"{h:.2f}%",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
                color="#1e293b",
            )

    fig.tight_layout()
    return fig


def _evaluation_content():
    comparison_path = find_asset("results/comparison_results.csv")
    raw_comparison = pd.DataFrame()
    if comparison_path.is_file():
        try:
            raw_comparison = pd.read_csv(comparison_path)
            model_name_map = {
                "TEXT (MuRIL)": "Text model",
                "IMAGE (ViT)": "Image model",
                "MULTIMODAL (Gated Fusion)": "Gated Fusion",
                "MULTIMODAL (Late Fusion, a=0.55)": "Late Fusion (a=0.55)",
            }
            comparison_formatted = raw_comparison.copy()
            comparison_formatted["Model"] = comparison_formatted["Model"].map(
                lambda m: model_name_map.get(m, m)
            )

            model_order = ["Text model", "Image model", "Gated Fusion", "Late Fusion (a=0.55)"]
            comparison_formatted["order"] = comparison_formatted["Model"].map(
                lambda m: model_order.index(m) if m in model_order else 99
            )
            comparison_formatted = comparison_formatted.sort_values("order").drop(columns=["order"])

            numeric_cols = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"]
            for col in numeric_cols:
                if col in comparison_formatted.columns:
                    comparison_formatted[col] = comparison_formatted[col].apply(lambda v: f"{float(v)*100:.2f}%")

            # Shorten column headers to ["Model", "Accuracy", "Macro Prec.", "Macro Rec.", "Macro F1", "Weighted F1"]
            comparison_formatted = comparison_formatted.rename(
                columns={
                    "Macro Precision": "Macro Prec.",
                    "Macro Recall": "Macro Rec.",
                }
            )
            desired_cols = ["Model", "Accuracy", "Macro Prec.", "Macro Rec.", "Macro F1", "Weighted F1"]
            comparison_formatted = comparison_formatted[[c for c in desired_cols if c in comparison_formatted.columns]]
            comparison_message = "✅ Benchmark scores loaded from `results/comparison_results.csv` on the 1,500-meme test set."
        except Exception as error:
            comparison_formatted = pd.DataFrame()
            comparison_message = f"Could not read benchmark CSV: {error}"
    else:
        comparison_formatted = pd.DataFrame()
        comparison_message = "`comparison_results.csv` not found."

    fig = generate_grouped_benchmark_chart(raw_comparison)

    results_dir = ROOT / "results"
    if not results_dir.is_dir():
        results_dir = ROOT

    cm_images = [str(p) for p in sorted(results_dir.glob("cm_*.png"))]
    history_images = [str(p) for p in sorted(results_dir.glob("history_*.png"))]
    xai_dir = results_dir / "xai_figures"
    xai_images = [str(p) for p in sorted(xai_dir.glob("*.png"))] if xai_dir.is_dir() else []

    return comparison_formatted, comparison_message, fig, cm_images, history_images, xai_images


CUSTOM_CSS = """
.gradio-container {
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    padding: 16px clamp(14px, 1.6vw, 28px) 44px !important;
    box-sizing: border-box !important;
    background: var(--body-background-fill) !important;
    color: var(--body-text-color) !important;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif !important;
    font-size: 14px !important;
}
#app-shell { width: 100%; max-width: none; margin: 0; }
.app-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 24px;
    padding: 6px 0 16px;
    margin-bottom: 14px;
    border-bottom: 1px solid var(--border-color-primary);
}
.app-brand { display: flex; align-items: center; gap: 14px; }
.brand-mark {
    display: grid; place-items: center; width: 42px; height: 42px;
    border: 1px solid var(--border-color-accent-subdued); border-radius: 10px;
    background: var(--color-accent-soft); color: var(--color-accent);
    font-size: 14px; font-weight: 800; letter-spacing: -.04em;
}
.app-header h1 { margin: 0 !important; font-size: 1.35rem !important; line-height: 1.2 !important; font-weight: 720 !important; letter-spacing: -.035em; }
.app-header p { margin: 4px 0 0 !important; max-width: 760px; color: var(--body-text-color-subdued) !important; font-size: .86rem !important; line-height: 1.45 !important; }
.runtime-badge { white-space: nowrap; border: 1px solid var(--border-color-primary); background: var(--background-fill-secondary); border-radius: 999px; padding: 7px 11px; color: var(--body-text-color); font-size: .76rem; }
.runtime-dot { display: inline-block; width: 7px; height: 7px; margin-right: 7px; border-radius: 50%; background: var(--color-accent); }
.gradio-container h2, .gradio-container h3, .gradio-container h4 { letter-spacing: -.02em; line-height: 1.3; }
.gradio-container h3 { font-size: 1.12rem !important; font-weight: 680 !important; }
.gradio-container h4 { font-size: .97rem !important; font-weight: 650 !important; }
.gradio-container .tabs { margin-top: 8px; }
.gradio-container button { border-radius: 8px !important; }
.gradio-container label, .gradio-container label span {
    background: transparent !important; color: var(--body-text-color-subdued) !important;
    padding: 0 !important; box-shadow: none !important; font-weight: 560 !important;
}
.workspace { align-items: stretch !important; gap: 18px !important; }
.panel {
    background: var(--block-background-fill) !important;
    border: 1px solid var(--block-border-color) !important;
    border-radius: 14px !important;
    padding: 20px !important;
}
.panel h4 { margin: 0 0 14px !important; }
.input-panel, .results-panel { min-width: 0; }
#input_meme_img { min-height: 0 !important; }
#input_meme_img > div { min-height: 0 !important; }
#input_meme_img .image-container { height: 240px !important; min-height: 240px !important; }
#input_meme_img img { object-fit: contain !important; }
#text_override textarea { min-height: 64px !important; background: var(--input-background-fill) !important; }
#analyze_btn { min-height: 44px !important; font-weight: 650 !important; }
#clear_btn { min-height: 44px !important; }
.empty-result { min-height: 76px; display: flex; flex-direction: column; justify-content: center; }
.results-panel .label-wrap { margin-top: 10px !important; }
#breakdown { margin-top: 8px; }
.explanation-heading { margin: 28px 0 12px !important; }
.explanation-panel { min-height: 300px; }
#gradcam img { object-fit: contain !important; }
.kpi-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin: 16px 0 24px; }
.kpi-card { background: var(--block-background-fill); border: 1px solid var(--block-border-color); border-radius: 12px; padding: 18px; }
.kpi-card::before { content: ''; display:block; height: 2px; margin: -18px -18px 15px; background: #2dd4bf; opacity: .8; }
.kpi-card.indigo::before { background: #818cf8; }
.kpi-card.emerald::before { background: #2dd4bf; }
.kpi-card.amber::before { background: #fbbf24; }
.kpi-card.cyan::before { background: #22d3ee; }
.kpi-label { font-size: .71rem; font-weight: 650; text-transform: uppercase; color: var(--body-text-color-subdued); letter-spacing: .06em; }
.kpi-value { font-size: 1.55rem; line-height: 1.25; font-weight: 740; color: var(--body-text-color); margin: 8px 0 4px; }
.kpi-subtext { font-size: .78rem; line-height: 1.45; color: var(--body-text-color-subdued); }
@media (max-width: 900px) {
    .gradio-container { padding: 14px 14px 36px !important; }
    .app-header { align-items: flex-start; }
    .runtime-badge { margin-top: 4px; }
    .panel { padding: 15px !important; }
    #input_meme_img .image-container { height: 220px !important; min-height: 220px !important; }
}
@media (max-width: 640px) {
    .app-header { flex-direction: column; gap: 12px; }
}
"""

CUSTOM_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.cyan,
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.Font("Inter"),
    radius_size="sm",
    spacing_size="sm",
)


with gr.Blocks(title="IndicSent-M | Meme Sentiment", elem_id="app-shell") as demo:

    device_label = f"{DEVICE.type.upper()} compute"
    gr.HTML(
        f"""
        <header class="app-header">
            <div class="app-brand">
                <div class="brand-mark">ISM</div>
                <div>
                    <h1>IndicSent-M</h1>
                    <p>Multimodal sentiment analysis for Hinglish and English memes. Combine image and text signals, then inspect what influenced the result.</p>
                </div>
            </div>
            <div class="runtime-badge"><span class="runtime-dot"></span>{device_label}</div>
        </header>
        """
    )

    with gr.Tab("Meme analyzer", id="tab_analyzer"): 
        with gr.Row(elem_classes=["workspace"]):
            with gr.Column(scale=4, elem_classes=["panel", "input-panel"]):
                gr.Markdown("#### Add a meme")
                input_image = gr.Image(
                    label="Meme image",
                    type="pil",
                    height=240,
                    show_label=False,
                    sources=["upload", "clipboard"],
                    elem_id="input_meme_img",
                )
                text_override = gr.Textbox(
                    elem_id="text_override",
                    label="Meme text (optional)",
                    placeholder="Add text, or leave blank to detect it from the image.",
                    lines=2,
                )

                with gr.Accordion("Model settings", open=False):
                    fusion = gr.Radio(
                        choices=[
                            "Late fusion (balanced F1)",
                            "Gated fusion (accuracy)",
                        ],
                        value="Late fusion (balanced F1)",
                        label="Fusion strategy",
                        info="Choose how text and image predictions are combined.",
                    )
                    alpha = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.55,
                        step=0.01,
                        label="Text weight for late fusion",
                        info="The remaining weight is assigned to the image model.",
                    )

                with gr.Row():
                    analyze_btn = gr.Button("Analyze meme", variant="primary", scale=2, elem_id="analyze_btn")
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1, elem_id="clear_btn")

            with gr.Column(scale=7, elem_classes=["panel", "results-panel"]):
                gr.Markdown("#### Prediction")
                verdict_box = gr.HTML(
                    value="""
                    <div class="empty-result" style="border: 1px solid var(--border-color-primary, #334155); border-radius: 8px; padding: 18px 20px; color: var(--body-text-color-subdued, #64748b);">
                        <strong style="color: var(--body-text-color, #0f172a); font-size: 1rem;">Ready for analysis</strong>
                        <span style="margin-top: 4px;">Upload a meme to see its sentiment and model explanations.</span>
                    </div>
                    """
                )
                final_result = gr.Label(label="Class probabilities", num_top_classes=3, visible=False)
                used_text = gr.Markdown()
                breakdown = gr.Dataframe(
                    elem_id="breakdown",
                    label="Model comparison",
                    value=pd.DataFrame(columns=["Architecture", "Prediction", "Confidence", "Negative", "Neutral", "Positive"]),
                    visible=False,
                    interactive=False,
                    wrap=True,
                )

        gr.Markdown("### Explanations", elem_classes=["explanation-heading"])
        with gr.Row(elem_classes=["workspace"]):
            with gr.Column(scale=1, elem_classes=["panel", "explanation-panel"]):
                gr.Markdown("#### Image attention map")
                gradcam = gr.Image(
                    elem_id="gradcam",
                    label="Image attention map",
                    type="pil",
                    show_label=False,
                )
                gr.HTML(
                    "<small style='color: var(--body-text-color-subdued);'>Warm/bright regions highlight the image patches driving the Vision Transformer's class activation.</small>"
                )
            with gr.Column(scale=1, elem_classes=["panel", "explanation-panel"]):
                gr.Markdown("#### Text attribution")
                word_html = gr.HTML(
                    value="<p style='color: var(--body-text-color-subdued); font-style: italic; padding: 12px;'>Word attributions will appear here after analysis.</p>",
                )

        # Examples section
        sample_sarcastic = find_asset("examples/sample_sarcastic.png")
        sample_negative = find_asset("examples/sample_negative.png")
        sample_neutral = find_asset("examples/sample_neutral.png")

        example_rows = []
        if sample_sarcastic.is_file():
            example_rows.append([
                str(sample_sarcastic),
                "Bas karo bhai, kitna rulaoge ab",
                "Late fusion (balanced F1)",
                0.55,
            ])
        if sample_negative.is_file():
            example_rows.append([
                str(sample_negative),
                "Ye bik gayi hai gormint, ab gormint mein kuch nahi bacha",
                "Late fusion (balanced F1)",
                0.55,
            ])
        if sample_neutral.is_file():
            example_rows.append([
                str(sample_neutral),
                "When teacher asks question and everyone looks at the topper",
                "Gated fusion (accuracy)",
                0.55,
            ])

        if example_rows:
            gr.Markdown("#### 💡 Quick Test Examples")
            gr.Examples(
                examples=example_rows,
                inputs=[input_image, text_override, fusion, alpha],
                label="Click an example meme to pre-fill inputs:",
                cache_examples=False,
            )

        analyze_btn.click(
            analyze_meme,
            inputs=[input_image, text_override, fusion, alpha],
            outputs=[verdict_box, final_result, breakdown, gradcam, word_html, used_text],
        )

        def _clear_all():
            empty_verdict = """
            <div class="empty-result" style="border: 1px solid var(--border-color-primary, #334155); border-radius: 8px; padding: 18px 20px; color: var(--body-text-color-subdued, #64748b);">
                <strong style="color: var(--body-text-color, #0f172a); font-size: 1rem;">Ready for analysis</strong>
                <span style="margin-top: 4px;">Upload a meme to see its sentiment and model explanations.</span>
            </div>
            """
            return (
                None,
                "",
                "Late fusion (balanced F1)",
                0.55,
                empty_verdict,
                gr.update(value=None, visible=False),
                gr.update(value=pd.DataFrame(columns=["Architecture", "Prediction", "Confidence", "Negative", "Neutral", "Positive"]), visible=False),
                None,
                "<p style='color: var(--body-text-color-subdued); font-style: italic; padding: 12px;'>Word attributions will appear here after analysis.</p>",
                "",
            )

        clear_btn.click(
            _clear_all,
            inputs=[],
            outputs=[input_image, text_override, fusion, alpha, verdict_box, final_result, breakdown, gradcam, word_html, used_text],
        )

    with gr.Tab("Benchmarks", id="tab_eval"): 
        gr.HTML(
            """
            <div class="kpi-container">
                <div class="kpi-card emerald">
                    <div class="kpi-label">Best Macro &amp; Weighted F1</div>
                    <div class="kpi-value">32.43% <span style="font-size: 1.15rem; font-weight: 700; color: #059669;">/ 32.78%</span></div>
                    <div class="kpi-subtext">Macro: 32.43% | Weighted: 32.78% (Late Fusion α=0.55)</div>
                </div>
                <div class="kpi-card indigo">
                    <div class="kpi-label">Best Overall Accuracy</div>
                    <div class="kpi-value">34.53%</div>
                    <div class="kpi-subtext">Multimodal Gated Fusion</div>
                </div>
                <div class="kpi-card cyan">
                    <div class="kpi-label">Multimodal Lift</div>
                    <div class="kpi-value">+3.33% Acc</div>
                    <div class="kpi-subtext">+1.47% Macro-F1 over unimodal text</div>
                </div>
                <div class="kpi-card amber">
                    <div class="kpi-label">Test Corpus</div>
                    <div class="kpi-value">1,500 Memes</div>
                    <div class="kpi-subtext">Memotion 3.0 Official Eval Split</div>
                </div>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("#### 📋 Test Set Performance Summary")
                comparison_table = gr.Dataframe(
                    interactive=False,
                    wrap=False,
                )
                comparison_status = gr.Markdown()
            with gr.Column(scale=3):
                gr.Markdown("#### 📊 Accuracy, Macro F1 & Weighted F1 Comparison")
                benchmark_plot = gr.Plot(
                    label="Model Benchmark Metrics (%)",
                    show_label=False,
                )

        gr.Markdown(
            """
            > [!TIP]
            > **Architectural Takeaway:** Unimodal text (MuRIL) achieves 31.20% accuracy and unimodal image (ViT) achieves 33.93%.
            > Gated Multimodal Fusion significantly boosts overall accuracy to **34.53%**, while Late Fusion with calibrated text weighting (α=0.55)
            > delivers the highest **Macro F1 (32.43%)**, effectively countering the test set's negative class distribution shift.
            """
        )

    with gr.Tab("Diagnostics", id="tab_diagnostics"): 
        gr.Markdown("### 🔍 Model Diagnostics: Confusion Matrices & Convergence")
        with gr.Tab("Confusion Matrices"):
            gr.Markdown("Compare how text, vision, and fusion architectures classify Negative, Neutral, and Positive test memes.")
            cm_gallery = gr.Gallery(
                label="Confusion Matrices (Text, Image, Late Fusion, Multimodal)",
                columns=2,
                height="auto",
                object_fit="contain",
            )
        with gr.Tab("Training Dynamics"):
            gr.Markdown("Training and validation loss/accuracy trajectories over epochs.")
            history_gallery = gr.Gallery(
                label="Training & Validation Curves",
                columns=3,
                height="auto",
                object_fit="contain",
            )
        with gr.Tab("XAI Case Studies"):
            gr.Markdown("Case studies illustrating multimodal cross-attention, saliency localization, and sentiment attribution.")
            xai_gallery = gr.Gallery(
                label="XAI Case Study Panels",
                columns=1,
                height="auto",
                object_fit="contain",
            )

    with gr.Tab("Methodology", id="tab_methodology"): 
        gr.Markdown(
            r"""
            ### 🏗️ IndicSent-M System Architecture

            IndicSent-M tackles the complex challenge of sentiment analysis on Indian social media memes, which are characterized by:
            1. **Code-switching and Transliteration:** Hinglish (Hindi words written in the Roman/Latin alphabet).
            2. **Multimodal Incongruity:** Visual expression conflicting with literal text meaning to produce irony or sarcasm.
            3. **Cultural Context:** Context-dependent humour and Bollywood/pop-culture meme templates.

            ---

            #### 1. Text Encoder: Google MuRIL
            - **Model Identifier:** `google/muril-base-cased` (236M parameters)
            - **Pretraining:** Trained on 17 Indian languages and transliterated Hinglish corpora.
            - **Tokenization:** WordPiece vocabulary of 197k tokens preserving subword units of transliterated Hindi.
            - **Feature Extraction:** Mean-pooled representations across sequence tokens weighted by attention masks:
            $$\mathbf{e}_t = \\frac{\\sum_{i} m_i \\mathbf{h}_{t,i}}{\\sum_i m_i}$$

            #### 2. Vision Encoder: Vision Transformer (ViT)
            - **Model Identifier:** `google/vit-base-patch16-224` (86M parameters)
            - **Patch Resolution:** 16x16 pixel patches flattened and linearly projected into 768-dimensional token embeddings.
            - **Representation:** Extracted from the final layer's `[CLS]` token:
            $$\mathbf{e}_v = \\mathbf{h}_{v, \\text{CLS}}$$

            #### 3. Cross-Modal Fusion Mechanisms
            - **Late Fusion (Decision-Level):** Linearly interpolates the predicted probability distributions of text and vision branches:
            $$P_{\\text{final}} = \\alpha P_{\\text{text}} + (1 - \\alpha) P_{\\text{image}}, \\quad \\alpha = 0.55$$
            Late fusion achieves the highest Macro F1 (32.43%) by preserving the calibrated unimodal priors.
            - **Gated Cross-Modal Fusion (Feature-Level):** Projects both modalities into a shared 256-dimensional space and learns a dynamic sigmoid gate $\\mathbf{g}$:
            $$\\mathbf{h}_t = \\text{GELU}(\\text{LayerNorm}(\\mathbf{W}_t \\mathbf{e}_t)), \\quad \\mathbf{h}_v = \\text{GELU}(\\text{LayerNorm}(\\mathbf{W}_v \\mathbf{e}_v))$$
            $$\\mathbf{g} = \\sigma(\\mathbf{W}_g [\\mathbf{h}_t; \\mathbf{h}_v])$$
            $$\\mathbf{h}_{\\text{fused}} = [\\mathbf{g} \\odot \\mathbf{h}_t ; (1 - \\mathbf{g}) \\odot \\mathbf{h}_v]$$
            This gated representation is classified through a LayerNorm + Dropout + MLP block, achieving 34.53% Accuracy.

            #### 4. OCR & Preprocessing
            - **Engine:** EasyOCR with English Latin character detection.
            - **Optimization:** Hindi Devanagari script is intentionally excluded to prevent misrecognition of Romanized Hinglish memes.
            - **Image Enhancement:** Images below 900px are upscaled via Lanczos resampling with sharpness enhancement (1.6x factor) before text extraction.
            """
        )

    demo.load(
        _evaluation_content,
        outputs=[comparison_table, comparison_status, benchmark_plot, cm_gallery, history_gallery, xai_gallery],
    )


if __name__ == "__main__":
    demo.launch(theme=CUSTOM_THEME, css=CUSTOM_CSS)
