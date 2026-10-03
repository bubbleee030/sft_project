import os
import json
import logging

# --- Constants ---
JUDGE_PROMPT_TEMPLATE = """
### 原始任務指令:
{original_prompt}

### 標準參考答案 (Ground Truth):
{reference_answer}

### 待評分的回覆:
{response_to_be_judged}
"""

def run_formatting_pipeline(source_dir: str, output_dir: str, output_filename: str, keep_simplified: bool = False):
    """
    Converts cleaned and balanced data into the judge model's fine-tuning format.
    """
    logging.info(f"--- Running Formatting Pipeline ---")
    if not os.path.exists(source_dir):
        logging.error(f"Error: Source directory '{source_dir}' does not exist!")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    formatted_data = []
    total_processed_count = 0

    logging.info("Formatting data into user/assistant conversation format for the Judge Model...")

    # Target the specific files produced by the balancing step
    target_files = [f for f in os.listdir(source_dir) if (f.startswith("traditional_") or (keep_simplified and f.startswith("simplified_"))) and f.endswith(".json")]

    for filename in target_files:
        file_path = os.path.join(source_dir, filename)
        logging.info(f"  Processing file: {filename}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for item in data:
                # Ensure all required fields are present
                original_prompt = item.get('prompt')
                reference_answer = item.get('resp')
                response_to_judge = item.get('model_response')
                ground_truth_judgement = item.get('full_output')

                if all([original_prompt, reference_answer, response_to_judge, ground_truth_judgement]):
                    user_content = JUDGE_PROMPT_TEMPLATE.format(
                        original_prompt=original_prompt,
                        reference_answer=reference_answer,
                        response_to_be_judged=response_to_judge
                    ).strip()
                    
                    assistant_content = ground_truth_judgement

                    formatted_item = {
                        "messages": [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": assistant_content}
                        ]
                    }
                    formatted_data.append(formatted_item)
                else:
                    logging.warning(f"    Skipping an item due to missing fields (QID: {item.get('qid')})")
            
            total_processed_count += len(data)

        except Exception as e:
            logging.error(f"Error processing file {filename}: {e}")

    # Write the final .jsonl file
    output_path = os.path.join(output_dir, output_filename)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in formatted_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        logging.info("="*50)
        logging.info("Judge Model training dataset preparation complete!")
        logging.info(f"Total source items processed: {total_processed_count}")
        logging.info(f"Successfully converted and wrote {len(formatted_data)} items to:")
        logging.info(f"  --> {os.path.abspath(output_path)}")
        logging.info("="*50)

    except Exception as e:
        logging.error(f"Error writing final training file {output_path}: {e}")
