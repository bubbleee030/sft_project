import json
import random

def create_mismatched_dataset(input_file, output_file):
    """
    讀取一個 .jsonl 檔案，將其 'messages' 中 'content' 內的 '### 待評分的回覆:' 後面的內容隨機打亂，
    生成一個新的文不對題的 .jsonl 檔案。
    """
    # 1. 讀取原始資料並抽取出所有 '待評分的回覆' 內容
    original_data = []
    all_responses = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                original_data.append(item)
                # 確保 messages 欄位存在且有內容
                if 'messages' in item and item['messages']:
                    content = item['messages'][0].get('content', '')
                    # 提取 '### 待評分的回覆:' 後面的內容
                    if '### 待評分的回覆:' in content:
                        response_part = content.split('### 待評分的回覆:', 1)[1].strip()
                        all_responses.append(response_part)
    except FileNotFoundError:
        print(f"錯誤：找不到輸入檔案 '{input_file}'")
        return
    except json.JSONDecodeError as e:
        print(f"錯誤：解析檔案 '{input_file}' 時發生錯誤: {e}")
        return

    if not all_responses:
        print("錯誤：輸入檔案中沒有找到任何 '### 待評分的回覆:' 欄位可供打亂。")
        return

    # 2. 隨機打亂 '待評分的回覆' 列表
    random.shuffle(all_responses)

    # 3. 寫入新的檔案，用打亂後的回覆替換原來的
    mismatched_count = 0
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i, item in enumerate(original_data):
            # 建立一個新的 item，以防修改原始資料
            new_item = item.copy()
            
            # 從打亂的列表中取一個回覆
            # 使用 i % len(all_responses) 確保索引不會超出範圍
            new_response = all_responses[i % len(all_responses)]
            
            # 替換 content 中的回覆部分
            original_content = new_item['messages'][0]['content']
            if '### 待評分的回覆:' in original_content:
                task_part = original_content.split('### 待評分的回覆:', 1)[0]
                new_content = task_part + '### 待評分的回覆:\n' + new_response
                new_item['messages'][0]['content'] = new_content
                
                # 檢查是否真的文不對題 (避免隨機後剛好配到自己)
                original_response = original_content.split('### 待評分的回覆:', 1)[1].strip()
                if new_response != original_response:
                    mismatched_count += 1
            
            f_out.write(json.dumps(new_item, ensure_ascii=False) + '\n')
            
    print("\n==================== 資料集生成報告 ====================")
    print(f"成功從 '{input_file}' 讀取 {len(original_data)} 筆資料。")
    print(f"已生成文不對題的資料集，並儲存至 '{output_file}'。")
    print(f"確保 {mismatched_count} / {len(original_data)} 筆資料的 '待評分的回覆' 與原始不同。")
    print("========================================================\n")


if __name__ == '__main__':
    # === 使用方法 ===
    # 原始的、內容正確的資料集
    source_dataset = './judge_training_data/judge_inference_dataset.jsonl' 
    # 希望儲存文不對題資料集的新檔名
    mismatched_output_dataset = 'adversarial_test.jsonl'
    
    create_mismatched_dataset(source_dataset, mismatched_output_dataset)