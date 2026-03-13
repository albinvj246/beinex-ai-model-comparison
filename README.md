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
