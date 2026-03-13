# beinex-ai-Sentiment-Analysis-model-comparison
Comparative analysis of DistilBERT and Twitter RoBERTa for sentiment analysis using HuggingFace inference APIs. Conducted as part of the Beinex AI Internship Level-2 evaluation.

This repository contains the source code and experimental data for comparing two HuggingFace NLP models: **DistilBERT** vs **Twitter RoBERTa**. 

## Files Included
* `run_inference.py`: Connects to the HuggingFace Inference API, tests 10 diverse emotional/sarcastic inputs, handles cold-start latency mitigation, and logs accuracy/speed.
* `results.json`: The raw, exact timing and prediction data dumped from the HuggingFace server.

## How to Run
1. Get a free HuggingFace API key and place it in `run_inference.py`.
2. Run `pip install requests`.
3. Execute `python run_inference.py` to recreate the metric data.

## Example Results

| ID | Input Text                             | DistilBERT | RoBERTa    |
| -- | -------------------------------------- | ---------- | ---------- |
| 1  | The new update is fantastic            | POS (1.00) | POS (0.99) |
| 2  | Worst customer service ever            | NEG (1.00) | NEG (0.96) |
| 3  | The package arrived on Tuesday         | NEG (0.98) | NEU (0.67) |
| 4  | The UI looks great but the app crashes | NEG (1.00) | NEG (0.82) |
| 5  | Oh brilliant, another error            | POS (0.99) | POS (0.73) |
| 6  | This feature is totally fire 🔥        | POS (1.00) | POS (0.98) |
| 7  | ok                                     | POS (1.00) | NEU (0.51) |
| 8  | Damaged box but item unharmed          | NEG (1.00) | NEU (0.48) |
| 9  | It's not that I hate it, but...        | NEG (1.00) | NEG (0.86) |
| 10 | Battery life is acceptable             | POS (1.00) | POS (0.74) |

## Model Comparison

| Metric                  | DistilBERT                 | Twitter RoBERTa                        |
| ----------------------- | -------------------------- | -------------------------------------- |
| Output Format           | Binary (POS / NEG) + Score | Three-class (POS / NEU / NEG) + Score  |
| Average Inference Speed | 0.35s                      | 0.35s after warm-up                    |
| Model Size              | 67M parameters             | 125M parameters                        |
| Integration             | Lightweight and easy       | Heavier but more robust                |
| Domain Strength         | Standard English reviews   | Social media language, sarcasm, emojis |
