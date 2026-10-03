import os
import json
import logging
import time
import pandas as pd

def balance_dataset_by_mission(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategically balances the dataset to ensure each mission has a similar count.
    """
    if df.empty:
        logging.warning("Warning: Input DataFrame is empty, cannot perform balancing.")
        return pd.DataFrame()

    logging.info("--- Starting strategic dataset balancing ---")
    
    mission_counts = df['mission'].value_counts()
    logging.info("Mission distribution of valid data before balancing:\n%s", mission_counts)
    
    if mission_counts.empty:
        logging.warning("No data to balance, returning original DataFrame.")
        return df
        
    target_count = mission_counts.min()
    logging.info(f"Target count (based on the smallest mission): {target_count}")

    df['combination'] = df['genmodel'] + " / " + df['judgemodel']
    combination_priority = df['combination'].value_counts().reset_index()
    combination_priority.columns = ['combination', 'total_count']
    combination_priority = combination_priority.sort_values('total_count', ascending=True)
    logging.info(f"Model combination priority (from lowest to highest):\n{combination_priority.to_string()}")

    balanced_dfs = []
    for mission, group_df in df.groupby('mission'):
        if len(group_df) <= target_count:
            logging.info(f"Mission '{mission}' (count: {len(group_df)}) does not need sampling, keeping all.")
            balanced_dfs.append(group_df)
            continue

        logging.info(f"Mission '{mission}' (count: {len(group_df)}) needs down-sampling to {target_count} items.")
        samples_to_keep = pd.DataFrame()
        for _, row in combination_priority.iterrows():
            priority_combo = row['combination']
            combo_in_group = group_df[group_df['combination'] == priority_combo]
            if combo_in_group.empty: continue
            
            if len(samples_to_keep) + len(combo_in_group) <= target_count:
                samples_to_keep = pd.concat([samples_to_keep, combo_in_group])
            else:
                remaining_needed = target_count - len(samples_to_keep)
                if remaining_needed > 0:
                    samples_to_add = combo_in_group.sample(n=remaining_needed, random_state=42)
                    samples_to_keep = pd.concat([samples_to_keep, samples_to_add])
                break
        balanced_dfs.append(samples_to_keep)

    final_balanced_df = pd.concat(balanced_dfs).reset_index(drop=True)
    if 'combination' in final_balanced_df.columns:
        final_balanced_df = final_balanced_df.drop(columns=['combination'])
    
    logging.info("--- Dataset balancing finished ---")
    logging.info("Mission distribution after balancing:\n%s", final_balanced_df['mission'].value_counts())

    return final_balanced_df

def save_balanced_files(final_df: pd.DataFrame, output_dir: str, keep_simplified: bool = False):
    """
    Saves the final, balanced DataFrame into multiple classified JSON files.
    """
    if final_df.empty:
        logging.warning("Final DataFrame is empty. No files will be saved.")
        return

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving balanced and classified results to '{output_dir}'...")

    all_items = final_df.to_dict('records')

    # Prepare buckets
    output_buckets = {}
    for item in all_items:
        status = item.get('status', 'Valid') # Default to valid if status is missing
        if status == 'Valid':
            mission = item.get('mission')
            # Use original bucket keys for traditional, and handle simplified based on flag
            prefix = "simplified" if item.get('trigger_char') else "traditional"
            if not keep_simplified and prefix == "simplified":
                continue # Skip simplified items if not keeping them (safety net)
            bucket_key = f"{prefix}_{mission}"
            if bucket_key not in output_buckets:
                output_buckets[bucket_key] = []
            # Clean up metadata before saving
            item.pop('status', None); item.pop('error_reason', None)
            output_buckets[bucket_key].append(item)
        else:
            # Handle logs like format_error and deduplicated_log
            log_key = 'format_error' if status == 'FormatError' else 'deduplicated_log'
            if log_key not in output_buckets:
                output_buckets[log_key] = []
            output_buckets[log_key].append(item)

    # Save all buckets to files
    for bucket_name, data in output_buckets.items():
        if not data: continue
        filename = os.path.join(output_dir, f"{bucket_name}.json")
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logging.info(f"Successfully saved {len(data)} items to: {filename}")
        except Exception as e:
            logging.error(f"Error saving file {filename}: {e}")

def run_balancing_pipeline(input_dir: str, output_dir: str, keep_simplified: bool = False):
    """
    Orchestrates the data balancing process.
    """
    start_time = time.time()
    logging.info(f"--- Running Balancing Pipeline ---")
    logging.info(f"Reading data from: {input_dir}")
    logging.info(f"Will save balanced data to: {output_dir}")

    # --- Step 1: Load all valid target data ---
    target_files = ['traditional_od.json', 'traditional_press.json', 'traditional_petition.json', 'traditional_qa.json']
    if keep_simplified:
        target_files.extend(['simplified_od.json', 'simplified_press.json', 'simplified_petition.json', 'simplified_qa.json'])
    all_valid_data = []
    for filename in target_files:
        file_path = os.path.join(input_dir, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_valid_data.extend(data)
                    logging.info(f"Successfully loaded {len(data)} items from '{filename}'.")
            except Exception as e:
                logging.error(f"Error reading file {filename}: {e}")
        else:
            logging.warning(f"Warning: Target file not found, skipping: {file_path}")

    if not all_valid_data:
        logging.error("No valid data was loaded. Terminating balancing process.")
        return

    valid_df = pd.DataFrame(all_valid_data)
    valid_df['status'] = 'Valid' # Ensure status column exists for processing

    # --- Step 2: Balance the valid data ---
    balanced_valid_df = balance_dataset_by_mission(valid_df)

    # --- Step 3: Load log data (errors, duplicates) ---
    all_other_data = []
    log_files = ['format_error.json', 'deduplicated_log.json']
    for log_file in log_files:
        file_path = os.path.join(input_dir, log_file)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
                # The structure of these logs is a list of dicts, where the actual data is nested
                for item in log_data:
                    details = item.get('format_error_item_details') or item.get('removed_item_details')
                    if details:
                        details['status'] = 'FormatError' if log_file == 'format_error.json' else 'Duplicate'
                        all_other_data.append(details)
    
    other_df = pd.DataFrame(all_other_data)
    logging.info(f"Loaded {len(other_df)} log items (errors/duplicates).")

    # --- Step 4: Combine and Save ---
    final_df = pd.concat([balanced_valid_df, other_df], ignore_index=True)
    logging.info(f"Re-combined data. Final total items: {len(final_df)}")

    save_balanced_files(final_df, output_dir, keep_simplified=keep_simplified)
    
    duration = time.time() - start_time
    logging.info(f"Balancing pipeline finished in {duration:.2f} seconds.")
    print(f"Balancing pipeline finished. Check logs and the '{output_dir}' directory for results.")
