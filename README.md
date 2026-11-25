---
title: IndicSent-M
emoji: ""
colorFrom: teal
colorTo: blue
sdk: gradio
app_file: app.py
python_version: "3.11"
---

# IndicSent-M

Multimodal sentiment analysis for Hinglish and English memes. The Gradio app combines OCR, a MuRIL text encoder, and a ViT image encoder, and includes model comparisons and explanations.

## Run locally

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

All three trained checkpoints are stored with Git LFS in `artifacts/`. A clone needs Git LFS installed to download them. The app also downloads the public MuRIL and ViT encoder weights on first run.

## Deploy the Gradio app

GitHub Pages serves static files and cannot run Python or Gradio. The interactive app is configured for Hugging Face Spaces (Gradio SDK).

1. Create a new Gradio Space on Hugging Face.
2. Clone the Space repository and add this project as its source, or push this repository to the Space's Git remote.
3. Ensure Git LFS is installed and pull the checkpoint files before pushing.
4. Push to the Space's `main` branch. Hugging Face builds the app from the Space metadata above and `requirements.txt`.

The checkpoints total about 2.6 GB, so allow time for the initial LFS upload and Space build. The Space must have enough storage and memory for the checkpoints and model encoders.

## Deploy the project page

The `docs/` folder contains a static project page. The included GitHub Actions workflow publishes it with GitHub Pages after each push to `main`.

In the GitHub repository, open **Settings > Pages** and select **GitHub Actions** as the build and deployment source. The site will then be available at `https://dev-maegor.github.io/IndicSent-M/`.

## Project contents

- `app.py`: Gradio application and interactive analysis UI.
- `src/`: model loading, OCR, and explainability code.
- `artifacts/`: trained checkpoints, tokenizer, and training configuration.
- `results/`: benchmark tables, training curves, confusion matrices, and explanation examples.
- `examples/`: sample memes for the analyzer.
- `notebooks/`: training and explainability notebooks.
- `docs/`: static GitHub Pages project page.

## Evaluation

The benchmark scores below are from `results/comparison_results.csv`.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Text (MuRIL) | 31.20% | 30.97% | 31.24% |
| Image (ViT) | 33.93% | 30.83% | 30.98% |
| Gated multimodal fusion | 34.53% | 30.46% | 31.23% |
| Late fusion (alpha=0.55) | 33.67% | 32.43% | 32.78% |
