import os
import json
import logging

# --- Constants ---
# Template for inference is simpler, as it doesn't include the reference answer.
INFERENCE_PROMPT_TEMPLATE = """
### 原始任務指令:
{original_prompt}

### 待評分的回覆:
{response_to_be_judged}
"""

def run_inference_formatting(source_dir: str, output_dir: str, output_filename: str, limit_per_file: int = 100):
    """
    Converts raw data into the format required for judge model inference.
    """
    logging.info(f"--- Running Inference Data Preparation ---")
    if not os.path.exists(source_dir):
        logging.error(f"Error: Source directory '{source_dir}' does not exist!")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    formatted_data = []
    total_processed_count = 0

    logging.info(f"Formatting data for inference from '{source_dir}'. Max {limit_per_file} items per file.")

    # In the original script, this targeted .jsonl files
    target_files = [f for f in os.listdir(source_dir) if f.endswith(".jsonl")]

    for filename in target_files:
        file_path = os.path.join(source_dir, filename)
        logging.info(f"  Processing file: {filename}")
        
        file_processed_count = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue

                    if file_processed_count >= limit_per_file:
                        logging.info(f"Reached limit of {limit_per_file} for file {filename}. Moving to next file.")
                        break
                        
                    item = json.loads(line)
                    total_processed_count += 1
                    file_processed_count += 1
                    
                    original_prompt = item.get('prompt')
                    response_to_judge = item.get('model_resp') # Original script used 'model_resp'

                    if all([original_prompt, response_to_judge]):
                        user_content = INFERENCE_PROMPT_TEMPLATE.format(
                            original_prompt=original_prompt,
                            response_to_be_judged=response_to_judge
                        ).strip()
                        
                        # For inference, we only provide the user message
                        formatted_item = {
                            "messages": [
                                {"role": "user", "content": user_content},
                            ]
                        }
                        formatted_data.append(formatted_item)
                    else:
                        logging.warning(f"    Skipping an item due to missing 'prompt' or 'model_resp' (QID: {item.get('qid')})")
            
        except Exception as e:
            logging.error(f"Error processing file {filename}: {e}")

    # Write the final .jsonl file
    output_path = os.path.join(output_dir, output_filename)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in formatted_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        logging.info("="*50)
        logging.info("Judge Model inference dataset preparation complete!")
        logging.info(f"Total source items processed: {total_processed_count}")
        logging.info(f"Successfully converted and wrote {len(formatted_data)} items to:")
        logging.info(f"  --> {os.path.abspath(output_path)}")
        logging.info("="*50)

    except Exception as e:
        logging.error(f"Error writing final inference file {output_path}: {e}")
