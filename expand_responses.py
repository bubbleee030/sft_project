import json

def expand_responses(input_file: str, output_file: str):
    """
    將 JSONL 檔案中的 'responses' 列表展開為單獨的 'response_0', 'response_1' 等字段。

    Args:
        input_file: 來源 JSONL 檔案路徑
        output_file: 輸出 JSONL 檔案路徑
    """
    print(f"正在讀取檔案: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"正在寫入到檔案: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines:
            try:
                item = json.loads(line.strip())
                if 'responses' in item and isinstance(item['responses'], list):
                    # 展開 responses 列表
                    for i, response in enumerate(item['responses']):
                        item[f'response_{i}'] = response
                    # 移除原始的 responses 字段
                    del item['responses']
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            except json.JSONDecodeError:
                print(f"警告：跳過無法解析的行: {line.strip()}")

    print("資料轉換完成！")

if __name__ == '__main__':
    # 設定檔案路徑
    INPUT_JSONL_PATH = 'argilla_ready_tem0.87_p0.95.jsonl'  # 來源檔案
    OUTPUT_JSONL_PATH = 'argilla_ready_tem0.87_p0.95_expanded.jsonl'  # 輸出檔案
    # -------------------------

    expand_responses(INPUT_JSONL_PATH, OUTPUT_JSONL_PATH)