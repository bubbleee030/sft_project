import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert Argilla records to DPO training dataset.")
    parser.add_argument('--input_file', type=str, default='../模型回答偏好選擇_整合_20260220_231157/records.json', help='Path to Argilla records.json')
    parser.add_argument('--output_file', type=str, default='dpo_training_data.jsonl', help='Path to output DPO JSONL file')
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found.")
        return

    with open(args.input_file, 'r', encoding='utf-8') as f:
        records = json.load(f)

    dpo_records = []
    for record in records:
        responses = record.get("responses", {})
        pref_selections = responses.get("preference_selection", [])
        if not pref_selections:
            continue
        
        pref = pref_selections[0].get("value")
        if pref in ["resp_0_better", "resp_1_better"]:
            fields = record.get("fields", {})
            prompt = fields.get("prompt", "")
            resp_0 = fields.get("response_0", "")
            resp_1 = fields.get("response_1", "")
            
            if pref == "resp_0_better":
                chosen = resp_0
                rejected = resp_1
            else:
                chosen = resp_1
                rejected = resp_0
                
            # LLaMA-Factory standard DPO format
            dpo_records.append({
                "conversations": [
                    {"role": "user", "content": prompt}
                ],
                "chosen": {"role": "assistant", "content": chosen},
                "rejected": {"role": "assistant", "content": rejected}
            })

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in dpo_records:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Successfully converted {len(dpo_records)} records to DPO format.")
    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    main()
