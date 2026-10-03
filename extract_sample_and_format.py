import os
import json
import logging
import random
from datetime import datetime
from pathlib import Path

# --- 日誌設定 ---
log_time = datetime.now().strftime('%Y%m%d_%H%M%S')
log_directory = 'logs'
os.makedirs(log_directory, exist_ok=True)
log_filename = f"extract_sample_{log_time}.log"
log_filepath = os.path.join(log_directory, log_filename)

logging.basicConfig(
    level=logging.INFO,
    filename=log_filepath,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# show logs at terminal
logger = logging.getLogger()
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


JUDGE_PROMPT_TEMPLATE = """
### 原始任務指令:
{original_prompt}

### 標準參考答案 (Ground Truth):
{reference_answer}

### 待評分的回覆:
{response_to_be_judged}
"""


def find_jsonl_files(directory, limit=4):
    """
    找出目錄下的所有 .jsonl 檔案。
    
    Args:
        directory (str): 要搜尋的目錄路徑
        limit (int): 最多要找的檔案數量
        
    Returns:
        list: .jsonl 檔案的完整路徑列表
    """
    jsonl_files = []
    try:
        for file in os.listdir(directory):
            if file.endswith('.jsonl'):
                jsonl_files.append(os.path.join(directory, file))
                if len(jsonl_files) >= limit:
                    break
    except Exception as e:
        logging.error(f"搜尋目錄 {directory} 時發生錯誤: {e}")
    
    return jsonl_files


def extract_samples_from_jsonl(jsonl_filepath, num_samples=3):
    """
    從 JSONL 檔案中隨機提取指定數量的樣本，並記錄來源。
    
    Args:
        jsonl_filepath (str): JSONL 檔案路徑
        num_samples (int): 要提取的樣本數量
        
    Returns:
        tuple: (samples_list, source_info_list)
               samples_list: 提取的資料項目列表
               source_info_list: 記錄每筆樣本的來源資訊
    """
    all_items = []
    try:
        # 先讀取所有行
        with open(jsonl_filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    all_items.append(item)
                except json.JSONDecodeError as e:
                    logging.warning(f"無法解析 {jsonl_filepath} 中第 {i+1} 行: {e}")
        
        # 從所有項目中隨機抽取樣本
        if not all_items:
            logging.warning(f"檔案 {os.path.basename(jsonl_filepath)} 中沒有有效資料")
            return [], []
        
        # 如果檔案中的項目數小於需要的樣本數，則返回所有項目；否則隨機抽取
        samples_count = min(num_samples, len(all_items))
        samples = random.sample(all_items, samples_count)
        
        # 記錄來源資訊
        source_info = []
        for sample in samples:
            source_info.append({
                "source_file": os.path.basename(jsonl_filepath),
                "qid": sample.get('qid', 'unknown'),
                "prompt_preview": str(sample.get('prompt', 'N/A'))[:50]
            })
        
        logging.info(f"成功從 {os.path.basename(jsonl_filepath)} 隨機提取 {len(samples)} 筆資料 (共 {len(all_items)} 筆)")
        return samples, source_info
    except Exception as e:
        logging.error(f"讀取檔案 {jsonl_filepath} 時發生錯誤: {e}")
        return [], []


def extract_from_directories(directory_list, samples_per_jsonl=3):
    """
    從多個目錄中的 JSONL 檔案提取樣本。
    
    Args:
        directory_list (list): 目錄路徑列表
        samples_per_jsonl (int): 每個 JSONL 檔案提取的樣本數
        
    Returns:
        tuple: (all_samples_list, all_source_info_list)
               all_samples_list: 所有提取的資料項目列表
               all_source_info_list: 所有來源資訊列表
    """
    all_samples = []
    all_source_info = []
    total_processed = 0
    
    for directory in directory_list:
        if not os.path.isdir(directory):
            logging.warning(f"目錄不存在或無法訪問: {directory}")
            continue
        
        logging.info(f"\n開始處理目錄: {directory}")
        jsonl_files = find_jsonl_files(directory, limit=4)
        
        if not jsonl_files:
            logging.warning(f"在目錄 {directory} 中未找到 JSONL 檔案")
            continue
        
        logging.info(f"在目錄中找到 {len(jsonl_files)} 個 JSONL 檔案")
        
        for jsonl_path in jsonl_files:
            samples, source_info = extract_samples_from_jsonl(jsonl_path, num_samples=samples_per_jsonl)
            all_samples.extend(samples)
            all_source_info.extend(source_info)
            total_processed += len(samples)
    
    logging.info(f"\n共從所有目錄提取 {total_processed} 筆資料")
    return all_samples, all_source_info


def validate_item(item):
    """
    驗證資料項目是否包含必要欄位。
    
    Args:
        item (dict): 資料項目
        
    Returns:
        tuple: (is_valid, error_message)
    """
    # 必須的欄位
    required_fields = ['prompt', 'resp', 'full_output']
    
    # 檢查必須欄位
    for field in required_fields:
        if field not in item or not item.get(field):
            return False, f"缺少或為空的欄位: {field}"
    
    # 檢查 model_response 或 model_resp (至少需要其中一個)
    has_model_response = item.get('model_response') or item.get('model_resp')
    if not has_model_response:
        return False, f"缺少或為空的欄位: model_response 或 model_resp"
    
    return True, "OK"


def format_samples_for_judge_model(samples, source_info):
    """
    將提取的樣本格式化為「裁判模型」的 user/assistant 對話微調格式。
    
    Args:
        samples (list): 原始樣本列表
        source_info (list): 來源資訊列表
        
    Returns:
        list: 格式化後的資料列表
    """
    formatted_data = []
    skipped_count = 0
    
    logging.info(f"\n開始格式化 {len(samples)} 筆資料...")
    
    for i, item in enumerate(samples):
        # 驗證資料
        is_valid, error_msg = validate_item(item)
        if not is_valid:
            logging.warning(f"資料 #{i+1} 無效: {error_msg}")
            skipped_count += 1
            continue
        
        original_prompt = item.get('prompt')
        reference_answer = item.get('resp')
        # 優先使用 model_response，如果不存在則使用 model_resp
        response_to_judge = item.get('model_response') or item.get('model_resp')
        ground_truth_judgement = item.get('full_output')
        
        # 從 source_info 中取得對應的來源資訊
        item_source = source_info[i] if i < len(source_info) else {}
        
        # 記錄在日誌中
        logging.info(f"[資料 #{len(formatted_data)+1}] 來源: {item_source.get('source_file', 'unknown')} | QID: {item_source.get('qid', 'unknown')}")
        
        # 建立 user/assistant 格式
        user_content = JUDGE_PROMPT_TEMPLATE.format(
            original_prompt=original_prompt,
            reference_answer=reference_answer,
            response_to_be_judged=response_to_judge
        ).strip()
        
        assistant_content = ground_truth_judgement
        
        # 組成 LLaMA-Factory 偏好的 ShareGPT/messages 格式
        formatted_item = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }
        formatted_data.append(formatted_item)
    
    logging.info(f"成功格式化 {len(formatted_data)} 筆資料，跳過 {skipped_count} 筆無效資料")
    return formatted_data


def save_formatted_data(formatted_data, output_dir, output_filename="sampled_judge_dataset.jsonl"):
    """
    將格式化後的資料儲存為 JSONL 檔案。
    
    Args:
        formatted_data (list): 格式化後的資料列表
        output_dir (str): 輸出目錄
        output_filename (str): 輸出檔案名稱
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in formatted_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        logging.info("="*60)
        logging.info("✓ 樣本提取與格式化完成！")
        logging.info(f"  總共成功轉換: {len(formatted_data)} 筆資料")
        logging.info(f"  輸出檔案: {os.path.abspath(output_path)}")
        logging.info("="*60)
        print(f"\n✓ 完成！資料已儲存至: {os.path.abspath(output_path)}")
        
    except Exception as e:
        logging.error(f"寫入檔案 {output_path} 時發生錯誤: {e}")


def main():
    """主程式：從使用者輸入的多個目錄提取並格式化資料"""
    
    print("\n" + "="*60)
    print("JSONL 樣本提取與格式化工具")
    print("="*60)
    print("\n此工具將從指定目錄的 JSONL 檔案中提取樣本，")
    print("並轉換為可用於微調的格式。\n")
    
    # 取得使用者輸入的目錄
    directories = []
    print("請輸入要處理的目錄路徑 (每次一個)")
    print("輸入完成後按 Enter 留空後再按 Enter 以開始處理")
    print("-" * 60)
    
    while True:
        user_input = input("目錄路徑 (或留空完成): ").strip()
        if not user_input:
            break
        if os.path.isdir(user_input):
            directories.append(user_input)
            print(f"✓ 已新增: {user_input}")
        else:
            print(f"✗ 目錄不存在: {user_input}")
    
    if not directories:
        print("\n✗ 未輸入任何有效目錄，程式結束。")
        logging.warning("使用者未輸入任何目錄")
        return
    
    print(f"\n即將處理 {len(directories)} 個目錄:")
    for d in directories:
        print(f"  - {d}")
    
    # 取得使用者輸入：每個檔案提取的樣本數
    while True:
        try:
            samples_per_file = int(input(f"\n每個 JSONL 檔案提取多少筆資料? (預設: 3): ") or "3")
            if samples_per_file > 0:
                break
            else:
                print("✗ 請輸入正整數")
        except ValueError:
            print("✗ 無效輸入，請輸入數字")
    
    # 取得使用者輸入：輸出檔案名稱
    output_filename = input(f"\n輸出檔案名稱 (預設: sampled_judge_dataset.jsonl): ").strip()
    if not output_filename:
        output_filename = "sampled_judge_dataset.jsonl"
    
    output_dir = input(f"輸出目錄 (預設: judge_training_data): ").strip()
    if not output_dir:
        output_dir = "judge_training_data"
    
    # 執行提取與格式化
    print("\n" + "="*60)
    print("開始處理...")
    print("="*60 + "\n")
    
    # Step 1: 從目錄提取樣本
    samples, source_info = extract_from_directories(directories, samples_per_jsonl=samples_per_file)
    
    if not samples:
        print("\n✗ 沒有提取到任何資料，程式結束。")
        logging.error("未能提取任何資料")
        return
    
    # Step 2: 格式化樣本
    formatted_data = format_samples_for_judge_model(samples, source_info)
    
    if not formatted_data:
        print("\n✗ 格式化後沒有有效資料，程式結束。")
        logging.error("格式化後沒有有效資料")
        return
    
    # Step 3: 儲存結果
    save_formatted_data(formatted_data, output_dir, output_filename)
    
    logging.info(f"程式執行完成，詳細日誌已記錄在: {os.path.abspath(log_filepath)}")


if __name__ == "__main__":
    main()
