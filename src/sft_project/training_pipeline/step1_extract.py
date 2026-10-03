import os
import json
import logging
import time
import pandas as pd
from simhash import Simhash
from datetime import datetime
from tqdm import tqdm

# --- Constants ---
missions = ['petition', 'od', 'press', 'qa']

# --- Helper Functions ---
def findMission(filename):
    """
    Parses the mission name from the filename.
    """
    for mission in missions:
        if mission in filename.lower():
            return mission
    return 'unknown'

def load_simpCharSet(filepath):
    """
    Loads simplified characters from a file into a set.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            chars = set(f.read().replace(" ", "").replace("\n", ""))
            logging.info(f"Successfully loaded {len(chars)} unique simplified characters from {filepath}.")
            return chars
    except FileNotFoundError:
        logging.error(f"Error: Simplified characters file not found at {filepath}. Terminating.")
        exit()

def containSimp(text, simplifiedCharSet):
    """
    Checks if the text contains any simplified characters.
    """
    if not isinstance(text, str) or not text: return None
    for char in text:
        if char in simplifiedCharSet: return char
    return None

def parse_model_from_path(normalized_path):
    """
    Extracts genmodel and judgemodel from the normalized path.
    """
    try:
        path_parts = normalized_path.split('/')
        eval_results_index = path_parts.index('eval_results')
        genmodel = path_parts[eval_results_index + 1]
        judgemodel = path_parts[eval_results_index + 2]
        return genmodel, judgemodel
    except (ValueError, IndexError):
        logging.warning(f"Could not parse genmodel or judgemodel from path '{normalized_path}'.")
        return 'unknown', 'unknown'

# --- Core Logic Functions ---
def extract_and_validate(root_directory, keep_think=False):
    """
    Traverses directories, finds all .jsonl/.json files, and extracts relevant fields.
    Performs format validation.
    """
    validated_data = []
    format_error = []
    all_items_metadata = [] # For statistics
    file_paths = []
    patterns = ['petition', 'od', 'press', 'qa', 'score']
    
    logging.info(f"Starting search in '{root_directory}'...")
    
    for subdir, _, files in os.walk(root_directory):
        for filename in files:
            file_paths.append(os.path.join(subdir, filename))
    
    for file_path in tqdm(file_paths, desc="Scanning all files"):
        normalized_path = file_path.replace('\\', '/')
        genmodel, judgemodel = parse_model_from_path(normalized_path)
        mission = findMission(os.path.basename(normalized_path))
    
        if 'raft' in os.path.basename(normalized_path).lower():
            logging.debug(f"Skipping file: {normalized_path} (contains 'raft')")
            continue
        if not any(pattern in os.path.basename(normalized_path).lower() for pattern in patterns):
            logging.debug(f"Skipping file: {normalized_path} (does not match required patterns)")
            continue

        try:
            current_items = []
            if normalized_path.endswith('.jsonl'):
                with open(normalized_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip(): continue
                        current_items.append(json.loads(line))
            elif normalized_path.endswith('.json'):
                with open(normalized_path, 'r', encoding='utf-8') as f:
                    json_content = json.load(f)
                    current_items = json_content.get("eval_result_from_first_iteration", [])
                if not current_items:
                    logging.warning(f"In file {normalized_path}, 'eval_result_from_first_iteration' not found or is empty.")
                    continue
            else:
                continue # Skip non-json/jsonl files

            for i, item in enumerate(current_items):
                internal_id = f"{normalized_path}_{i}"
                base_metadata = {'internal_id': internal_id, 'mission': mission, 'genmodel': genmodel, 'judgemodel': judgemodel}

                if normalized_path.endswith('.json'):
                    model_responses = item.get('model_responses', {})
                    processed_item = {
                        'qid': item.get('qid'),
                        'prompt': item.get('question'),
                        'resp': model_responses.get('ground_truth'),
                        'model_response': model_responses.get('Llama-4-Maverick-17B-128E-Instruct-FP8') or model_responses.get('Llama-3.1-8B-Instruct'),
                        'full_output': item.get('judge_response')
                    }
                else:  # .jsonl
                    processed_item = {
                        'qid': item.get('qid'),
                        'prompt': item.get('prompt'),
                        'resp': item.get('resp'),
                        'model_response': item.get('model_response'),
                        'full_output': item.get('full_output'),
                    }
                
                processed_item.update(base_metadata)
                status, error_reason = "Valid", "N/A"
                
                full_output = processed_item.get('full_output')
                model_response = processed_item.get('model_response')
                required_tags = ['【給分原因】', '【分數】', '題文匹配度(MatchPoint)', '文本格式(TextFormat)']
                
                if not isinstance(full_output, str) or not full_output.strip():
                    status, error_reason = "FormatError", "'full_output' is missing, not a string, or empty"
                elif not all(tag in full_output for tag in required_tags):
                    status, error_reason = "FormatError", "'full_output' is missing required tags"
                elif not keep_think and 'think' in full_output.lower():
                    status, error_reason = "FormatError", "'full_output' contains 'think'"
                elif not isinstance(model_response, str) or not model_response.strip():
                    status, error_reason = "FormatError", "'model_response' is missing, not a string, or empty"

                metadata_entry = {**base_metadata, 'qid': processed_item.get('qid'), 'status': status, 'error_reason': error_reason}
                all_items_metadata.append(metadata_entry)

                if status == 'FormatError':
                    format_error.append({'reason': error_reason, 'format_error_item_details': processed_item})
                    logging.warning(f"[Format Error] {error_reason} (QID: {processed_item.get('qid')}), File: {normalized_path}")
                else:
                    validated_data.append(processed_item)

        except Exception as e:
            logging.error(f"Error processing file {normalized_path}: {e}", exc_info=True)

    logging.info(f"Step 1 (Extraction) finished. Found {len(validated_data)} valid items and {len(format_error)} format errors.")
    return validated_data, format_error, all_items_metadata

def find_and_remove_duplicates(data_list, threshold):
    """
    Uses SimHash to find and remove near-duplicate entries.
    """
    if not data_list:
        logging.warning("No valid data to deduplicate.")
        return [], []
    logging.info(f"Step 2 (Deduplication) starting with SimHash (threshold: {threshold})...")
    
    hashes = [(item['internal_id'], Simhash(list(item.get('full_output', '')))) for item in tqdm(data_list, desc="Calculating SimHashes")]
    id_to_item_map = {item['internal_id']: item for item in data_list}
    hashes.sort(key=lambda x: x[1].value)
    
    ids_to_remove = set()
    deduplicated_log = []
    
    for i in tqdm(range(len(hashes) - 1), desc="Comparing Hashes"):
        if hashes[i][0] in ids_to_remove: continue
        for j in range(i + 1, len(hashes)):
            if hashes[j][0] in ids_to_remove: continue
            # Remove the flawed early exit: if hashes[j][1].value - hashes[i][1].value > threshold: break
            # Limit the search window to speed up instead
            if j - i > 200: break
            
            distance = hashes[i][1].distance(hashes[j][1])
            if distance <= threshold:
                ids_to_remove.add(hashes[j][0])
                item_i_details = id_to_item_map[hashes[i][0]]
                item_j_details = id_to_item_map[hashes[j][0]]
                deduplicated_log.append({
                    'removed_qid': item_j_details.get('qid'),
                    'reason': f'SimHash Similar (distance: {distance})',
                    'compared_with_qid': item_i_details.get('qid'),
                    'removed_item_details': item_j_details
                })

    final_data = [item for item in data_list if item['internal_id'] not in ids_to_remove]
    logging.info(f"Step 2 (Deduplication) finished. Removed {len(deduplicated_log)} duplicates, {len(final_data)} items remaining.")
    
    # Update metadata for removed items
    for log_entry in deduplicated_log:
        log_entry['removed_item_details']['status'] = 'Duplicate'

    return final_data, deduplicated_log

def classify_and_save(data_list, simplifiedCharSet, output_dir):
    """
    Classifies data by language and mission, then saves to separate files.
    """
    if not data_list:
        logging.info("No data to classify and save.")
        return {}

    logging.info(f"Step 3 (Classification) starting. Saving results to '{output_dir}'...")
    final_data_buckets = {}
    language_buckets = {'traditional': [], 'simplified': []}

    for item in data_list:
        item.pop('internal_id', None)
        full_output_simp = containSimp(item.get('full_output', ''), simplifiedCharSet)
        model_response_simp = containSimp(item.get('model_response', ''), simplifiedCharSet)
        language_type = 'simplified' if full_output_simp or model_response_simp else 'traditional'
        mission_type = item.get('mission', 'unknown')
        
        if full_output_simp or model_response_simp:
           item['trigger_char'] = full_output_simp or model_response_simp
           item['trigger_source'] = "full_output" if full_output_simp else "model_response"
        
        language_buckets[language_type].append(item)
        bucket_key = f"{language_type}_{mission_type}"
        if bucket_key not in final_data_buckets:
            final_data_buckets[bucket_key] = []
        final_data_buckets[bucket_key].append(item)

    # Save classified data files
    for bucket_name, data in final_data_buckets.items():
        if not data: continue
        filename = os.path.join(output_dir, f"{bucket_name}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"Successfully saved {len(data)} items to: {filename}")

    # Save combined language files
    for lang, lang_data in language_buckets.items():
        if not lang_data: continue
        lang_filename = os.path.join(output_dir, f"classified_data_{lang}.json")
        with open(lang_filename, 'w', encoding='utf-8') as f:
            json.dump(lang_data, f, ensure_ascii=False, indent=4)
        logging.info(f"Successfully saved {len(lang_data)} language items to: {lang_filename}")
    
    logging.info("Step 3 (Classification) finished.")
    return final_data_buckets

def run_extraction_pipeline(input_dir, output_dir, simplified_chars_file, dedup_threshold, 
                            run_extraction=True, run_deduplication=True, run_classification=True,
                            no_simhash=False, keep_simplified=False, keep_think=False):
    """
    Orchestrates the entire data extraction, deduplication, and classification pipeline.
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Data Extraction and Validation ---
    if run_extraction:
        validated_data, format_error, all_items_metadata = extract_and_validate(input_dir, keep_think=keep_think)
        # Save intermediate results and logs
        with open(os.path.join(output_dir, 'validated_data.json'), 'w', encoding='utf-8') as f:
            json.dump(validated_data, f, ensure_ascii=False, indent=4)
        with open(os.path.join(output_dir, 'format_error.json'), 'w', encoding='utf-8') as f:
            json.dump(format_error, f, ensure_ascii=False, indent=4)
        with open(os.path.join(output_dir, 'full_metadata_log.json'), 'w', encoding='utf-8') as f:
            json.dump(all_items_metadata, f, ensure_ascii=False, indent=4)
    else:
        logging.info("Skipping Step 1: Loading previous results.")
        try:
            with open(os.path.join(output_dir, 'validated_data.json'), 'r', encoding='utf-8') as f:
                validated_data = json.load(f)
        except FileNotFoundError:
            logging.error("Error: 'validated_data.json' not found. Cannot skip extraction.")
            return

    # --- Step 2: Deduplication ---
    if run_deduplication:
        if no_simhash:
            logging.info("Ablation: Skipping SimHash deduplication (--no-simhash enabled).")
            deduplicated_data = validated_data
            deduplicated_log = []
        else:
            deduplicated_data, deduplicated_log = find_and_remove_duplicates(validated_data, threshold=dedup_threshold)
            with open(os.path.join(output_dir, 'deduplicated_log.json'), 'w', encoding='utf-8') as f:
                json.dump(deduplicated_log, f, ensure_ascii=False, indent=4)
        with open(os.path.join(output_dir, 'deduplicated_data.json'), 'w', encoding='utf-8') as f:
            json.dump(deduplicated_data, f, ensure_ascii=False, indent=4)
    else:
        logging.info("Skipping Step 2: Deduplication.")
        try:
            with open(os.path.join(output_dir, 'deduplicated_data.json'), 'r', encoding='utf-8') as f:
                deduplicated_data = json.load(f)
        except FileNotFoundError:
            logging.error("Error: 'deduplicated_data.json' not found. Cannot skip deduplication.")
            return

    # --- Step 3: Language/Mission Classification ---
    if run_classification:
        simplifiedCharSet = load_simpCharSet(simplified_chars_file)
        classify_and_save(deduplicated_data, simplifiedCharSet, output_dir)
    else:
        logging.info("Skipping Step 3: Classification.")

    duration = time.time() - start_time
    logging.info(f"Extraction pipeline finished in {duration:.2f} seconds.")
    print(f"Extraction pipeline finished. Check logs and the '{output_dir}' directory for results.")
