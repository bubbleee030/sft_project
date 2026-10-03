import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert Argilla records to a Judge Testset.")
    parser.add_argument('--input_file', type=str, default='../模型回答偏好選擇_整合_20260220_231157/records.json', help='Path to Argilla records.json')
    parser.add_argument('--output_file', type=str, default='judge_testset.jsonl', help='Path to output JSONL file')
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found.")
        return

    with open(args.input_file, 'r', encoding='utf-8') as f:
        records = json.load(f)

    valid_records = []
    for record in records:
        responses = record.get("responses", {})
        pref_selections = responses.get("preference_selection", [])
        if not pref_selections:
            continue
        
        pref = pref_selections[0].get("value")
        # We only want clear preferences, not "equal"
        if pref in ["resp_0_better", "resp_1_better"]:
            fields = record.get("fields", {})
            prompt = fields.get("prompt", "")
            resp_0 = fields.get("response_0", "")
            resp_1 = fields.get("response_1", "")
            
            chosen = resp_0 if pref == "resp_0_better" else resp_1
            
            valid_records.append({
                "id": record.get("id"),
                "prompt": prompt,
                "response_0": resp_0,
                "response_1": resp_1,
                "human_preference": pref,
                "golden_evaluation": chosen
            })

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in valid_records:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Successfully extracted {len(valid_records)} records with clear human preferences.")
    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    main()
