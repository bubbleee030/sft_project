import json
import random
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Generate a Blind A/B Testing Markdown form.")
    parser.add_argument("--sft_file", type=str, required=True, help="JSONL file with SFT model evaluations")
    parser.add_argument("--rl_file", type=str, required=True, help="JSONL file with RL model evaluations")
    parser.add_argument("--output_md", type=str, default="blind_test_form.md", help="Output markdown file")
    parser.add_argument("--answer_key", type=str, default="blind_test_answers.json", help="Secret mapping key")
    
    args = parser.parse_args()
    
    with open(args.sft_file, 'r', encoding='utf-8') as f:
        sft_data = [json.loads(line) for line in f]
        
    with open(args.rl_file, 'r', encoding='utf-8') as f:
        rl_data = [json.loads(line) for line in f]
        
    if len(sft_data) != len(rl_data):
        print("Warning: The two files have different lengths!")
        
    length = min(len(sft_data), len(rl_data))
    
    answers = {}
    md_content = "# LLM-as-a-Judge 盲測表單 (Blind A/B Test)\n\n"
    md_content += "請詳細閱讀每一題的「原始任務指令」與「待評分回覆」，並比較底下兩種模型的評分與評語（模型 A 與模型 B）。請在方框內打勾 [x] 選擇您認為較合理的一方。\n\n---\n\n"
    
    for i in range(length):
        sft_item = sft_data[i]
        rl_item = rl_data[i]
        
        # We assume the prompt is identical in both files
        # Extract the original question and response from the prompt
        # If it's a standard conversation format:
        if "messages" in sft_item:
            prompt_text = sft_item["messages"][0]["content"]
            sft_response = sft_item["messages"][1]["content"] if len(sft_item["messages"]) > 1 else sft_item.get("generated_text", "")
        else:
            prompt_text = sft_item.get("prompt", "")
            sft_response = sft_item.get("generated_text", sft_item.get("response", ""))
            
        rl_response = rl_item.get("generated_text", rl_item.get("response", ""))
        
        if "messages" in rl_item and not rl_response:
            rl_response = rl_item["messages"][1]["content"] if len(rl_item["messages"]) > 1 else ""

        # Randomize A and B
        is_sft_a = random.choice([True, False])
        
        if is_sft_a:
            model_A_resp = sft_response
            model_B_resp = rl_response
            answers[f"Question_{i+1}"] = {"A": "SFT", "B": "RL"}
        else:
            model_A_resp = rl_response
            model_B_resp = sft_response
            answers[f"Question_{i+1}"] = {"A": "RL", "B": "SFT"}
            
        md_content += f"## 題目 {i+1}\n\n"
        md_content += "### 📜 評分輸入 (Prompt)\n"
        md_content += "```text\n" + prompt_text.strip() + "\n```\n\n"
        
        md_content += "### 💡 評估結果 A (Model A)\n"
        md_content += "```text\n" + model_A_resp.strip() + "\n```\n\n"
        
        md_content += "### 💡 評估結果 B (Model B)\n"
        md_content += "```text\n" + model_B_resp.strip() + "\n```\n\n"
        
        md_content += "### ✅ 您的選擇\n"
        md_content += "- [ ] 模型 A 較好\n"
        md_content += "- [ ] 模型 B 較好\n"
        md_content += "- [ ] 兩者差不多 / Tie\n\n"
        md_content += "---\n\n"
        
    with open(args.output_md, 'w', encoding='utf-8') as f:
        f.write(md_content)
        
    with open(args.answer_key, 'w', encoding='utf-8') as f:
        json.dump(answers, f, indent=4)
        
    print(f"✅ 盲測表單已生成: {args.output_md}")
    print(f"🔒 答案對應表已安全儲存: {args.answer_key} (請在評估完成後再開啟！)")

if __name__ == "__main__":
    main()
