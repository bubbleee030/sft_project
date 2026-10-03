import json
import numpy as np
import logging
from transformers import AutoTokenizer
from huggingface_hub import login
import os

def run_length_analysis(dataset_path: str, model_name: str, hf_token: str = None):
    """
    Analyzes the token length of each entry in a dataset file.
    """
    logging.info(f"--- Running Token Length Analysis ---")
    
    # Handle Hugging Face token securely
    if hf_token:
        logging.info("Hugging Face token provided. Logging in.")
        login(token=hf_token)
    else:
        logging.info("No Hugging Face token provided. Using cached token if available.")

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logging.error(f"Failed to load tokenizer for '{model_name}'. If it's a private model, please provide a token. Error: {e}")
        return

    token_lengths = []
    
    logging.info(f"Reading dataset: {dataset_path}...")
    if not os.path.exists(dataset_path):
        logging.error(f"Dataset file not found at: {dataset_path}")
        return

    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data_point = json.loads(line)
                
                # Supports both flat text and ShareGPT message format
                if "messages" in data_point:
                    full_text = tokenizer.apply_chat_template(data_point["messages"], tokenize=False)
                elif "text" in data_point:
                    full_text = data_point.get("text", "")
                else:
                    logging.warning(f"Skipping line, no 'messages' or 'text' field found: {line.strip()}")
                    continue

                if full_text:
                    tokens = tokenizer.encode(full_text)
                    token_lengths.append(len(tokens))

            except json.JSONDecodeError:
                logging.warning(f"Skipping unparsable line: {line.strip()}")
                continue

    if not token_lengths:
        logging.error("No token lengths were calculated. Please check the dataset format.")
        return

    lengths_array = np.array(token_lengths)
    
    # Generate and print the report
    report = f"""

--- Token Length Analysis Report ---
Model Used: {model_name}
Dataset: {dataset_path}

Total Items: {len(lengths_array)}
Min Length:  {np.min(lengths_array)}
Max Length:  {np.max(lengths_array)}
Mean Length: {np.mean(lengths_array):.2f}
Median:      {np.median(lengths_array)}

--- Percentiles ---
90th: {np.percentile(lengths_array, 90):.0f}
95th: {np.percentile(lengths_array, 95):.0f}
98th: {np.percentile(lengths_array, 98):.0f}
99th: {np.percentile(lengths_array, 99):.0f}

--- Recommendation ---
Review the 95th or 98th percentile value and choose a slightly larger power-of-two number
(e.g., 2048, 4096, 8192) as your 'cutoff_len' for training.
------------------------------------
"""
    print(report)
    logging.info(report)
