import os
import json
import logging
from datetime import datetime

# --- 日誌設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

JUDGE_PROMPT_TEMPLATE = """
### 原始任務指令:
{original_prompt}

### 待評分的回覆:
{response_to_be_judged}
"""

def trace_source_with_mapping(source_dir, output_dir):
    """
    重新處理資料，並生成兩個檔案：
    1. judge_inference_dataset.jsonl - 原始輸出（加上 source_file 欄位）
    2. source_mapping.json - 完整的來源映射表
    """
    if not os.path.exists(source_dir):
        logging.error(f"錯誤：來源目錄 '{source_dir}' 不存在！")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    formatted_data = []
    line_mapping = []  # 記錄每一行的來源檔案和行號
    total_line_number = 0
    
    logging.info("開始追蹤資料來源...")

    # **固定排序檔案，確保可重現性**
    source_files = sorted([f for f in os.listdir(source_dir) if f.endswith(".jsonl")])
    
    for filename in source_files:
        file_path = os.path.join(source_dir, filename)
        logging.info(f"  正在處理檔案: {filename}")
        
        file_line_number = 0
        file_processed_count = 0
        max_per_file = 100
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    file_line_number += 1
                    
                    if line.strip():  # 跳過空行
                        if file_processed_count >= max_per_file:
                            logging.info(f"檔案 {filename} 已處理100筆資料，停止處理此檔案")
                            break
                            
                        item = json.loads(line)
                        total_line_number += 1
                        file_processed_count += 1
                        
                        mission_type = item.get('mission', 'unknown')
                        original_prompt = item.get('prompt')
                        reference_answer = item.get('resp')
                        response_to_judge = item.get('model_resp')
                        qid = item.get('qid', 'unknown')

                        if all([mission_type, original_prompt, reference_answer, response_to_judge]):
                            user_content = JUDGE_PROMPT_TEMPLATE.format(
                                original_prompt=original_prompt,
                                response_to_be_judged=response_to_judge
                            ).strip()
                            
                            formatted_item = {
                                "messages": [
                                    {"role": "user", "content": user_content},
                                ],
                                "source_file": filename,
                                "qid": qid
                            }
                            formatted_data.append(formatted_item)
                            
                            # 記錄映射關係
                            line_mapping.append({
                                "output_line_number": total_line_number,
                                "source_file": filename,
                                "source_file_line": file_line_number,
                                "qid": qid
                            })
                        else:
                            logging.warning(f"    跳過一筆資料，因為缺少必要欄位。(QID: {qid})")
            
            logging.info(f"檔案 {filename} 處理完成，共處理 {file_processed_count} 筆資料")

        except Exception as e:
            logging.error(f"處理檔案 {filename} 時發生錯誤: {e}")

    # 輸出1: JSONL檔案（加上來源資訊）
    output_jsonl_path = os.path.join(output_dir, "judge_inference_dataset.jsonl")
    try:
        with open(output_jsonl_path, 'w', encoding='utf-8') as f:
            for entry in formatted_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        logging.info(f"✓ JSONL檔案已寫入: {os.path.abspath(output_jsonl_path)}")
    except Exception as e:
        logging.error(f"寫入 JSONL 檔案時發生錯誤: {e}")

    # 輸出2: 來源映射表（JSON格式）
    output_mapping_path = os.path.join(output_dir, "source_mapping.json")
    try:
        mapping_data = {
            "generated_at": datetime.now().isoformat(),
            "total_lines": total_line_number,
            "source_files": source_files,
            "line_mapping": line_mapping
        }
        with open(output_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)
        logging.info(f"✓ 來源映射表已寫入: {os.path.abspath(output_mapping_path)}")
    except Exception as e:
        logging.error(f"寫入映射表時發生錯誤: {e}")

    # 輸出3: 簡單對照表（CSV/TSV格式）
    output_csv_path = os.path.join(output_dir, "source_mapping.tsv")
    try:
        with open(output_csv_path, 'w', encoding='utf-8') as f:
            f.write("輸出行號\t來源檔案\t來源檔案行號\tQID\n")
            for mapping in line_mapping:
                f.write(f"{mapping['output_line_number']}\t{mapping['source_file']}\t{mapping['source_file_line']}\t{mapping['qid']}\n")
        logging.info(f"✓ TSV對照表已寫入: {os.path.abspath(output_csv_path)}")
    except Exception as e:
        logging.error(f"寫入TSV檔案時發生錯誤: {e}")

    logging.info("="*60)
    logging.info("資料來源追蹤完成！")
    logging.info(f"總共處理了 {total_line_number} 筆資料")
    logging.info(f"輸出檔案：")
    logging.info(f"  1. {output_jsonl_path}")
    logging.info(f"  2. {output_mapping_path}")
    logging.info(f"  3. {output_csv_path}")
    logging.info("="*60)


if __name__ == "__main__":
    SOURCE_DIRECTORY = "eval_results/gen_Gemma-3-27B-it/judge_llama-3.3-70B-Instruct"
    OUTPUT_DIRECTORY = "judge_training_data"
    
    trace_source_with_mapping(
        source_dir=SOURCE_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY
    )
