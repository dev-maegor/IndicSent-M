# IndicSent-M Data Card

## Dataset and splits

The project uses Memotion 3.0, with a reported 7,000-example training split and
1,500-example validation split. The model-training pipeline makes an 85/15
stratified train-validation split of the 7,000 training examples, yielding
5,950 training and 1,050 validation examples; `training_config.json` records
these internal split sizes and a 1,500-example test split.

The reported label-prior analysis shows a train-to-test shift in the Negative
class from 25.1% to 38.7%. This distribution shift should be considered when
interpreting aggregate accuracy and class-wise metrics.

Labels are ordered as Negative, Neutral, Positive (class ids 0, 1, and 2 in the
available prediction CSVs). The task uses meme images and associated
Romanized-Hinglish or English text. OCR is configured for English characters:
the Hindi Devanagari OCR language is intentionally excluded because much of
the target text is Romanized Hinglish.

## Benchmark results

The following four-model results are from `results/comparison_results.csv`.
Scores are reported as fractions on a 0-1 scale.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| TEXT (MuRIL) | 0.312 | 0.32642822862114135 | 0.3267601115036665 | 0.3096529389648596 | 0.3123668498824606 |
| IMAGE (ViT) | 0.3393333333333333 | 0.3279744455716675 | 0.34359873583536027 | 0.30833790480731876 | 0.3097918528072571 |
| MULTIMODAL (Gated Fusion) | 0.3453333333333333 | 0.32640457513921045 | 0.3355123505918575 | 0.30461116274592365 | 0.31226764103079724 |
| MULTIMODAL (Late Fusion, a=0.55) | 0.33666666666666667 | 0.34094645008181396 | 0.3444667356302045 | 0.32434037688095857 | 0.32776000815956385 |

Late fusion has the highest macro F1 and weighted F1 in this comparison;
gated fusion has the highest accuracy.

## Limitations and use

Sentiment in memes can depend on sarcasm, cultural references, code-switching,
and image-text interactions. OCR errors and the reported train/test class-prior
shift can affect predictions. Explanations are model-attribution signals, not
causal evidence or guarantees that an image region or word caused the sentiment
classification. This project is intended for research and educational
demonstration, not consequential decisions about individuals.
