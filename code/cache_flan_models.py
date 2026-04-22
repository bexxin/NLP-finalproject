import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

models_to_cache = {
    "google/flan-t5-base": "./hf_cache",
    "google/flan-t5-large": "./hf_cache_large",
}

def cache_has_model_files(cache_dir):
    """
    Check whether the cache folder already has model files.
    This avoids downloading again before the demo.
    """
    if not os.path.exists(cache_dir):
        return False

    needed_indicators = [
        "config.json",
        "model.safetensors",
    ]

    for root, dirs, files in os.walk(cache_dir):
        if all(file_name in files for file_name in needed_indicators):
            return True

    return False


for model_name, cache_dir in models_to_cache.items():
    print("\n" + "=" * 60)
    print(f"Checking cache for {model_name}")
    print(f"Cache folder: {cache_dir}")

    if cache_has_model_files(cache_dir):
        print("Model already appears to be cached. Skipping download.")
        continue

    print("Model not found in cache. Downloading now...")

    AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )

    AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )

    print(f"Finished caching {model_name}")

print("\nCache check complete.")