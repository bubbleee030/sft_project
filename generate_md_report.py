import os
import re

def parse_log_to_md(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract file name
    file_match = re.search(r'檔案: (.+\.jsonl)', content)
    file_name = file_match.group(1) if file_match else "Unknown"

    # Extract total samples
    total_match = re.search(r'總樣本數: (\d+)', content)
    total_samples = total_match.group(1) if total_match else "Unknown"

    # Extract format compliance
    format_match = re.search(r'符合指定格式的樣本數: (\d+) / (\d+) \(([\d.]+)%\)', content)
    if format_match:
        format_valid = format_match.group(1)
        format_total = format_match.group(2)
        format_percent = format_match.group(3)
    else:
        format_valid = format_total = format_percent = "Unknown"

    # Extract score parsing success
    parse_match = re.search(r'可成功解析分數的樣本數: (\d+) / (\d+) \(([\d.]+)%\)', content)
    if parse_match:
        parse_valid = parse_match.group(1)
        parse_total = parse_match.group(2)
        parse_percent = parse_match.group(3)
    else:
        parse_valid = parse_total = parse_percent = "Unknown"

    # Extract MatchPoint distribution
    mp_section = re.search(r'--- MatchPoint 分數分佈 ---(.*?)(?=---|$)', content, re.DOTALL)
    mp_dist = []
    if mp_section:
        for line in mp_section.group(1).strip().split('\n'):
            match = re.search(r'分數 ([\d.]+): (\d+) 次', line)
            if match:
                score = match.group(1)
                count = match.group(2)
                mp_dist.append((score, count))

    # Extract TextFormat distribution
    tf_section = re.search(r'--- TextFormat 分數分佈 ---(.*?)(?=---|$)', content, re.DOTALL)
    tf_dist = []
    if tf_section:
        for line in tf_section.group(1).strip().split('\n'):
            match = re.search(r'分數 ([\d.]+): (\d+) 次', line)
            if match:
                score = match.group(1)
                count = match.group(2)
                tf_dist.append((score, count))

    # Extract combinations
    combo_section = re.search(r'--- 分數組合分佈 \(MatchPoint, TextFormat\) ---(.*?)(?=---|$)', content, re.DOTALL)
    combos = []
    if combo_section:
        for line in combo_section.group(1).strip().split('\n'):
            match = re.search(r'組合 \(([\d.]+), ([\d.]+)\): (\d+) 次', line)
            if match:
                mp = match.group(1)
                tf = match.group(2)
                count = match.group(3)
                combos.append((mp, tf, count))

    # Generate Markdown
    md = f"# 模型輸出分析報告\n\n"
    md += f"**檔案**: {file_name}\n\n"
    md += f"**總樣本數**: {total_samples}\n\n"
    md += "## 格式符合度分析\n\n"
    md += f"符合指定格式的樣本數: {format_valid} / {format_total} ({format_percent}%)\n\n"
    md += "## 分數解析成功率\n\n"
    md += f"可成功解析分數的樣本數: {parse_valid} / {parse_total} ({parse_percent}%)\n\n"
    md += "## MatchPoint 分數分布\n\n"
    md += "| 分數 | 次數 |\n|------|------|\n"
    for score, count in mp_dist:
        md += f"| {score} | {count} |\n"
    md += "\n"
    md += "## TextFormat 分數分布\n\n"
    md += "| 分數 | 次數 |\n|------|------|\n"
    for score, count in tf_dist:
        md += f"| {score} | {count} |\n"
    md += "\n"
    md += "## 分數組合分布 (MatchPoint, TextFormat)\n\n"
    md += "| MatchPoint | TextFormat | 次數 |\n|------------|------------|------|\n"
    for mp, tf, count in combos:
        md += f"| {mp} | {tf} | {count} |\n"
    md += "\n"

    # Add charts
    base_name = log_file.replace('.log', '')
    md += "## 圖表\n\n"
    md += f"### MatchPoint 分數分布圖\n\n![MatchPoint Distribution]({base_name}_matchpoint.png)\n\n"
    md += f"### TextFormat 分數分布圖\n\n![TextFormat Distribution]({base_name}_textformat.png)\n\n"
    md += f"### 分數組合熱力圖\n\n![Score Combinations Heatmap]({base_name}_combinations.png)\n\n"

    return md

# Process all log files
log_files = ['analysis_tem0.9.log', 'analysis_tem0.85.log', 'analysis_tem0.95.log']
all_md = "# 模型分析報告總結\n\n"

for log_file in log_files:
    if os.path.exists(log_file):
        md_content = parse_log_to_md(log_file)
        all_md += md_content + "\n---\n\n"

# Write to file
with open('analysis_report.md', 'w', encoding='utf-8') as f:
    f.write(all_md)

print("Markdown report generated: analysis_report.md")