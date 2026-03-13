import requests
import time
import json

# ---------------------------------------------------------
# EXPERIMENTAL SETUP CONFIGURATION
# ---------------------------------------------------------
# Get your API key from https://huggingface.co/settings/tokens

API_TOKEN = "YOUR_HF_API_TOKEN" 
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# 1. Select the Models
MODELS = {
    "DistilBERT (Fast)": "https://router.huggingface.co/hf-inference/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "RoBERTa (Nuanced)": "https://router.huggingface.co/hf-inference/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
}

# 2. Add 10 Diverse Test Inputs
TEST_INPUTS = [
    "The new update is fantastic! Everything loads instantly.", # strongly positive
    "Worst customer service ever. I've been on hold for an hour.", # strongly negative
    "The package arrived on Tuesday via standard shipping.", # neutral
    "The UI looks great but the app crashes every 5 minutes.", # mixed
    "Oh brilliant, another unexpected error. Just what I needed today.", # sarcasm
    "This feature is totally fire (fire) pure W", # slang / emojis
    "ok", # short/ambiguous
    "While I appreciate the prompt delivery, the product box was damaged, though the item itself seems unharmed.", # complex context
    "It's not that I hate it, but I definitely don't love it either.", # subtle
    "The battery life is acceptable.", # mildly positive/neutral
]

def query_model(api_url, payload, max_retries=3):
    """Sends a request to the HF Inference API and measures inference latency."""
    for attempt in range(max_retries):
        start_time = time.time()
        response = requests.post(api_url, headers=HEADERS, json=payload)
        end_time = time.time()
        latency = end_time - start_time
        
        if response.status_code == 200:
            return response.json(), latency
        elif response.status_code == 503:
            # HuggingFace Free Tier puts unused models to sleep. 
            # 503 means it is waking up.
            est_time = response.json().get("estimated_time", 10)
            print(f"  [!] Model is waking up. Waiting {est_time:.1f} seconds...")
            time.sleep(est_time)
        else:
            return f"Error: {response.text}", latency
    return "Failed", 0

def run_experiment():
    print("Starting Model Inference Comparison...\n")
    
    # PRE-RUN: Wake up models to avoid skewing our latency data.
    print("Waking up models (First call latency)...")
    for name, url in MODELS.items():
        query_model(url, {"inputs": "warmup"})
    print("Models are ready!\n" + "-"*60)
    
    final_output = []

    # ACTUAL TEST
    for idx, text in enumerate(TEST_INPUTS, 1):
        test_case = {"id": idx, "text": text, "results": {}}
        print(f"\n[Test {idx}] Input: '{text}'")
        
        for model_name, model_url in MODELS.items():
            result, latency_sec = query_model(model_url, {"inputs": text})
            
            # Formatting the output nicely
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                # Sort by the highest confidence score
                top_prediction = max(result[0], key=lambda x: x['score'])
                label = top_prediction['label']
                score = top_prediction['score']
                print(f"  -> {model_name:<18} | Label: {label:<10} | Score: {score:.4f} | Time: {latency_sec:.3f}s")
                test_case["results"][model_name] = {"label": label, "score": score, "time": latency_sec}
            else:
                print(f"  -> {model_name:<18} | Error: {result}")
                test_case["results"][model_name] = {"error": str(result), "time": latency_sec}
        
        final_output.append(test_case)
        time.sleep(0.5) # Sleep half a second to respect free API rate-limits

    with open("results.json", "w") as f:
        json.dump(final_output, f, indent=4)
    print("\nSaved structured results to results.json")

if __name__ == "__main__":
    if API_TOKEN == "YOUR_HF_API_TOKEN_HERE":
        print("⚠️ PLEASE SET YOUR HUGGINGFACE API TOKEN AT THE TOP OF THE SCRIPT BEFORE RUNNING! ⚠️")
    else:
        run_experiment()
